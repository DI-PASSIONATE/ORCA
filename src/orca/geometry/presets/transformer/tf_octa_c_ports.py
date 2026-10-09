import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

from orca import BaseGeometry
from orca.geometry.cells.transformer import check_tf_octa_c_parameters, tf_octa_c
from orca.geometry.constraints import is_feasible_by
from orca.geometry.input_parameters import InputParameterIterator, RangeParameter
from orca.geometry.presets.paths import StackupXML

if TYPE_CHECKING:
    import numpy as np
    import skrf as rf

    from orca.training.datasets.base_dataset import BaseDataset

# Layout choices fixed for every sample (µm). They are part of the device the model
# describes, not inputs, so changing one calls for new simulations.
#: Each center tap only carries bias current, so it is narrow; it runs through the
#: other winding's feed gap on the other metal layer.
CENTER_TAP_WIDTH = 3.0
#: Gap between the two feed lines of a winding: the crossing 3 µm tap plus 1 µm on
#: either side. Close feeds are a tight differential pair with a small loop inductance.
FEED_GAP = 5.0
#: The Metal5 ground ring keeps GROUND_SPACING = 20 µm clear of the windings' outermost
#: metal and is GROUND_RING_WIDTH = 10 µm wide, as the inductor's. The feeds run across it
#: to its outer edge, 30 µm beyond the windings, where the ports sit: the reference planes
#: are the boundary of the cell, so simulated cells (inductors too) can be abutted with
#: touching ports. The feeds and their crossing of the ring are part of every model.
GROUND_SPACING = 20.0
GROUND_RING_WIDTH = 10.0

# Which combinations make a useful transformer. The coupling factor depends on the
# ratio of the winding diameters and on their offset relative to the size.
MIN_DIAMETER_RATIO = 0.55
#: Upper bound of relative_displacement, the offset between the winding centres as a
#: share of their mean diameter.
MAX_RELATIVE_DISPLACEMENT = 0.3


def _input_parameters() -> InputParameterIterator:
    # Lengths in µm, on a 0.1 µm grid. Sized for mm-wave and sub-THz transformers
    # (D-band to ~300 GHz): their self-resonance must lie well above the operating band,
    # which takes windings of roughly 20-90 µm. The diameters are log-sampled, and since
    # nothing else rejects small windings more often than large ones, the share of
    # layouts falls with size: about 29 % have a winding of 20-30 µm, 8 % of 70-90 µm.
    return InputParameterIterator(
        RangeParameter("bottom_winding_diameter", 20.0, 90.0, step=0.1, sampling="log"),
        RangeParameter("top_winding_diameter", 20.0, 90.0, step=0.1, sampling="log"),
        RangeParameter("relative_displacement", 0.0, MAX_RELATIVE_DISPLACEMENT, step=0.005),
        RangeParameter("bottom_linewidth", 2.0, 8.0, step=0.1),
        RangeParameter("top_linewidth", 2.0, 8.0, step=0.1),
        picking_strategy="sobol",
        frequency=[0.0, 500e9],  # DC to 500 GHz, as the simcfg sweeps it
    )


@dataclass
class TransformerOcta(BaseGeometry):
    """
    Stacked octagonal transformer (IHP SG13G2) for mm-wave and sub-THz circuits.

    One single-turn winding on TopMetal2 (ports ``op``/``on`` on the right, center tap
    ``oci`` to the left) over one on TopMetal1 (ports ``ip``/``in`` on the left, center
    tap ``ico`` to the right), inside a Metal5 ground ring the ports refer to. The ports
    sit on the ring's outer edge.
    """

    name: str = "tf_octa_c_ports"
    # Conformal SiO2/passivation over TopMetal2, as for the inductor (gds2palace L6n2
    # study). The simcfg meshes it with refined_cellsize = 2: the ports are flush with
    # the ring's outer edge, and at coarser sizes gmsh can fill that plane with flat
    # tetrahedra that Palace cannot solve (GDSConverter drops such meshes).
    stackup_xml: str = StackupXML.SG13G2_FEM_200um_passi3D
    simconfig_filename: str = os.path.join(os.path.dirname(__file__), "tf_octa_c_ports.simcfg")
    input_parameter_iterator: InputParameterIterator = field(
        default_factory=_input_parameters
    )

    # Close to zero for weakly coupled windings, where a relative error says nothing
    absolute_error_parameters: ClassVar[frozenset[str]] = frozenset({"k"})

    def electrical_parameters(self, ntwk: "rf.Network") -> dict[str, "np.ndarray"]:
        from orca.utils.postprocessing import transformer_parameters

        # Ports (simcfg order): 1 op, 2 on (top winding), 3 ip, 4 in (bottom winding),
        # 5 oci and 6 ico (the center taps), which are AC-grounded as in differential use.
        # The top winding (op/on) is the primary, as in COBRA's Lp/Qp goals.
        return transformer_parameters(ntwk, primary=(0, 1), secondary=(2, 3), shorted=(4, 5))

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
    def center_displacement(params: dict[str, Any]) -> float:
        """The offset between the winding centres in µm, from ``relative_displacement``."""
        mean_diameter = (params["bottom_winding_diameter"] + params["top_winding_diameter"]) / 2
        return params["relative_displacement"] * mean_diameter

    @staticmethod
    def _cell_arguments(params: dict[str, Any]) -> dict[str, Any]:
        """The tf_octa_c arguments for a parameter draw; the rest is fixed for this preset."""
        return {
            "bottom_winding_diameter": params["bottom_winding_diameter"],
            "top_winding_diameter": params["top_winding_diameter"],
            "center_displacement": TransformerOcta.center_displacement(params),
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
