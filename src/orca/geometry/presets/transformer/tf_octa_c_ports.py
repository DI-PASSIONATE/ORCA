import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orca import BaseGeometry
from orca.geometry.cells.transformer import check_tf_octa_c_parameters, tf_octa_c
from orca.geometry.constraints import is_feasible_by
from orca.geometry.input_parameters import InputParameterIterator, RangeParameter
from orca.geometry.presets.paths import StackupXML

if TYPE_CHECKING:
    from orca.training.datasets.base_dataset import BaseDataset

# Layout choices fixed for every sample (µm). They are part of the device the model
# describes, not inputs, so changing one calls for new simulations.
#: Each center tap only carries bias current, so it is narrow; it runs through the
#: other winding's feed gap on the other metal layer.
CENTER_TAP_WIDTH = 3.0
#: Gap between the two feed lines of a winding: the crossing 3 µm tap plus 1 µm on
#: either side. Close feeds are a tight differential pair with a small loop inductance.
FEED_GAP = 5.0
#: The ports sit on the ground ring's inner edge, GROUND_SPACING - GROUND_RING_WIDTH =
#: 10 µm beyond the windings' vertices, so short feeds are part of every model.
GROUND_SPACING = 20.0
GROUND_RING_WIDTH = 10.0

# Which combinations make a useful transformer. The coupling factor depends on the
# ratio of the winding diameters and on their offset relative to the size: in the
# earlier campaign, windings under 0.7 of their partner's diameter or offset by more
# than a fifth of it reached k ~ 0.05-0.3, where matched, centred ones reached ~0.5.
MIN_DIAMETER_RATIO = 0.75
MAX_RELATIVE_DISPLACEMENT = 0.2

# Built per instance rather than shared as a class attribute: a dataclass default
# holds one object for every instance of the class, so two geometries would share
# one iterator. The dataset is built per instance too, by create_dataset().


def _input_parameters() -> InputParameterIterator:
    # All in µm, on a 0.1 µm grid. Sized for mm-wave and sub-THz transformers (D-band
    # to ~300 GHz): their self-resonance must lie well above the operating band, which
    # takes windings of roughly 20-90 µm. The diameters are log-sampled, so the small
    # end gets as many samples per octave as the large one.
    return InputParameterIterator(
        RangeParameter("bottom_winding_diameter", 20.0, 90.0, step=0.1, sampling="log"),
        RangeParameter("top_winding_diameter", 20.0, 90.0, step=0.1, sampling="log"),
        RangeParameter("center_displacement", 0.0, 15.0, step=0.1),
        RangeParameter("bottom_linewidth", 2.0, 8.0, step=0.1),
        RangeParameter("top_linewidth", 2.0, 8.0, step=0.1),
        picking_strategy="sobol",
        frequency=[1e9, 500e9],  # 1 GHz to 500 GHz
    )


@dataclass
class TransformerOcta(BaseGeometry):
    """
    Stacked octagonal transformer (IHP SG13G2) for mm-wave and sub-THz circuits.

    One single-turn winding on TopMetal2 (ports ``op``/``on`` on the right, center tap
    ``oci`` to the left) over one on TopMetal1 (ports ``ip``/``in`` on the left, center
    tap ``ico`` to the right), inside a Metal5 ground ring the ports refer to.
    """

    name: str = "tf_octa_c_ports"
    stackup_xml: str = StackupXML.SG13G2_FEM_200um
    simconfig_filename: str = os.path.join(os.path.dirname(__file__), "tf_octa_c_ports.simcfg")
    input_parameter_iterator: InputParameterIterator = field(
        default_factory=_input_parameters
    )

    def create_dataset(self) -> "BaseDataset":
        # Imported here so the geometry can be drawn and simulated without the
        # "train" extra (PyTorch) installed.
        from orca.training.codecs import FlatReImCodec
        from orca.training.datasets.geo_to_s_param_single_f import (
            GeoToSParamDatasetSingleFrequency,
        )
        from orca.training.normalize import MinMaxNormalizer, StandardNormalizer

        return GeoToSParamDatasetSingleFrequency(
            codec=FlatReImCodec(n_ports=6),
            # Scale inputs with the declared parameter ranges rather than the
            # min/max of whatever subset happens to be trained on.
            input_normalizer=MinMaxNormalizer(self.input_parameter_iterator),
            output_normalizer=StandardNormalizer(),
        )

    @staticmethod
    def _cell_arguments(params: dict[str, Any]) -> dict[str, Any]:
        """The tf_octa_c arguments for a parameter draw; the rest is fixed for this preset."""
        return {
            "bottom_winding_diameter": params["bottom_winding_diameter"],
            "top_winding_diameter": params["top_winding_diameter"],
            "center_displacement": params["center_displacement"],
            "bottom_linewidth": params["bottom_linewidth"],
            "top_linewidth": params["top_linewidth"],
            "bottom_center_tap_width": CENTER_TAP_WIDTH,
            "upper_center_tap_width": CENTER_TAP_WIDTH,
            "lower_feed_type": 1,
            "upper_feed_type": 1,
            "feedline_spacing": FEED_GAP,
            "gnd_upper_spacing": GROUND_SPACING,
            "gnd_lower_spacing": GROUND_SPACING,
            "gnd_side_spacing": GROUND_SPACING,
            "gnd_ring_width": GROUND_RING_WIDTH,
        }

    @staticmethod
    def _coupling_constraints() -> list[str]:
        """The rules that keep the windings well coupled, as constraint expressions."""
        return [
            (
                "min(bottom_winding_diameter, top_winding_diameter) >= "
                f"{MIN_DIAMETER_RATIO:g} * max(bottom_winding_diameter, top_winding_diameter)"
            ),
            (
                f"center_displacement <= {MAX_RELATIVE_DISPLACEMENT:g} * "
                "(bottom_winding_diameter + top_winding_diameter) / 2"
            ),
        ]

    def feasibility_constraints(self) -> list[str]:
        # The binding rules of check_tf_octa_c_parameters for this preset's fixed
        # arguments, then the coupling rules. Left out because they always hold over
        # the ranges above: the 3 µm center taps fit both windings (d >= 9), the 5 µm
        # feed gap fits the octagon's side (d >= 13.1) and the ground ring is valid.
        # Kept in step with is_feasible by a test.
        return [
            "bottom_linewidth <= bottom_winding_diameter / 3",
            "top_linewidth <= top_winding_diameter / 3",
            *self._coupling_constraints(),
        ]

    def is_feasible(self, params: dict[str, Any]) -> bool:
        try:
            check_tf_octa_c_parameters(**self._cell_arguments(params))
        except ValueError:
            return False
        return is_feasible_by(self._coupling_constraints(), params)

    @staticmethod
    def create_gds_file(name: str, output_path: str, params: dict[str, Any]) -> str:
        c = tf_octa_c(name=name, **TransformerOcta._cell_arguments(params))
        c.write_gds(output_path, with_metadata=False)
        return output_path
