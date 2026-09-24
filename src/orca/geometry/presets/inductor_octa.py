import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from orca import BaseGeometry
from orca.geometry.cells.inductor import (
    VIA_GAP,
    VIA_MARGIN,
    VIA_SIZE,
    get_min_outer_diameter,
    symmetric_octa_IHP,
)
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

    def feasibility_constraints(self) -> list[str]:
        # get_min_outer_diameter as one expression; kept in step with it by a test.
        two_vias = 2 * VIA_SIZE + VIA_GAP + 2 * VIA_MARGIN
        overlap = f"(width if width >= {two_vias:g} else 1.1 * {two_vias:g})"
        crossover = (
            f"max(3 * width + 2 * space, "
            f"(2 * space + width) * (sqrt2 - 1) + (space + width) + 2 * {overlap})"
        )
        inner_segment = f"({crossover} + (0 if turns < 3 else width + 2 * space))"
        inner_diameter = (
            f"({inner_segment} * (1 + sqrt2) if turns > 1 else 2 * (width + space) * (1 + sqrt2))"
        )
        outer_diameter = f"{inner_diameter} + 2 * turns * width + 2 * (turns - 1) * space"
        return [f"diameter >= ceil(100 * ({outer_diameter})) / 100"]

    def is_feasible(self, params: dict[str, Any]) -> bool:
        # The windings, crossovers and feed vias must fit inside the outer diameter.
        N = round(params["turns"])
        return float(params["diameter"]) >= get_min_outer_diameter(
            N, float(params["width"]), float(params["space"])
        )

    @staticmethod
    def create_gds_file(name: str, output_path: str, params: dict[str, Any]) -> str:  # noqa: ARG004 - the cell name is derived from the parameters
        N = round(params["turns"])
        w = float(params["width"])
        s = float(params["space"])
        D = float(params["diameter"])

        # Refuse, rather than clamp, a diameter below the buildable minimum: the
        # parameter table records the requested value, so a clamped layout would
        # train the model on a diameter the layout does not have. is_feasible
        # rejects such draws before they get here.
        do_min = get_min_outer_diameter(N, w, s)
        if do_min > D:
            raise ValueError(
                f"diameter={D:g} is below the minimum {do_min:g} for turns={N}, "
                f"width={w:g}, space={s:g}."
            )

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
