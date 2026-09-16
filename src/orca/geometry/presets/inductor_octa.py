import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orca import BaseGeometry
from orca.geometry.cells.inductor import get_min_outer_diameter, symmetric_octa_IHP
from orca.geometry.input_parameters import InputParameterIterator

if TYPE_CHECKING:
    from orca.training.datasets.base_dataset import BaseDataset

# Ground ring is drawn on Metal5 (matches "from_layername": "Metal5" in the
# simcfg). Feed is on TopMetal1, which requires N >= 2 turns (see
# symmetric_octa_IHP: N == 1 feeds on TopMetal2 instead).
GROUND_LAYER = 67       # Metal5
RING_SPACING = 20.0     # µm, gap between inductor outer edge and ground ring
RING_WIDTH = 20.0       # µm, ground-ring thickness


# Built per instance rather than shared as a class attribute - see the note in
# tf_octa_c_ports.py. The dataset is built per instance too, by create_dataset().


def _input_parameters() -> InputParameterIterator:
    return InputParameterIterator(
        picking_strategy="random",
        frequency=[1e9, 500e9],  # 1 GHz to 500 GHz
        turns=[2, 3, 4, 5],
        width=[x / 100 for x in range(201, 1501, 1)],   # 2.01 .. 15.00 µm
        space=[x / 100 for x in range(201, 601, 1)],    # 2.01 ..  6.00 µm
        diameter=[float(x) for x in range(30, 301, 1)],  # 30 .. 300 µm
    )


@dataclass
class InductorOcta(BaseGeometry):
    """
    Represents a symmetric octagonal spiral inductor geometry (IHP SG13G2).

    2-port (LA/LB) spiral inductor, ported from the gds2palace IHP example by
    Volker Muehlhaus. Requires N >= 2 turns, since the feedline sits on
    TopMetal1 (single-turn inductors feed on TopMetal2 instead).
    """

    name: str = "inductor_octa"
    stackup_xml: str = os.path.join(os.path.dirname(__file__), "SG13G2_nosub.xml")
    simconfig_filename: str = os.path.join(
        os.path.dirname(__file__), "inductor_octa.simcfg"
    )
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
            codec=FlatReImCodec(n_ports=2),
            # Scale inputs with the declared parameter ranges rather than the
            # min/max of whatever subset happens to be trained on.
            input_normalizer=MinMaxNormalizer(self.input_parameter_iterator),
            output_normalizer=StandardNormalizer(),
        )

    @staticmethod
    def create_gds_file(name: str, output_path: str, params: dict[str, Any]) -> str:  # noqa: ARG004 - the cell name is derived from the parameters
        N = round(params["turns"])
        w = float(params["width"])
        s = float(params["space"])
        D = float(params["diameter"])

        # clamp the outer diameter to the minimum buildable (DRC-valid) value
        do_min = get_min_outer_diameter(N, w, s)
        D = max(D, do_min)

        symmetric_octa_IHP(
            N=N, D=D, w=w, s=s,
            includeCenterTap=False,
            LBE=False,
            forEM=True,
            ground_layer=GROUND_LAYER,
            ring_spacing=RING_SPACING,
            ring_width=RING_WIDTH,
            filename=output_path,
        )
        return output_path
