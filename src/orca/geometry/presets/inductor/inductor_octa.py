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
from orca.geometry.input_parameters import InputParameterIterator, RangeParameter
from orca.geometry.presets.paths import StackupXML

if TYPE_CHECKING:
    import numpy as np
    import skrf as rf

    from orca.training.datasets.base_dataset import BaseDataset

# All ports run up from a ground on Metal5 (matches "from_layername": "Metal5" in
# the simcfg): ports 1/2 to the feeds on TopMetal1, port 3 to the center tap on
# TopMetal2. The ground is a closed square ring around the spiral for every N, so
# one connected reference serves all three ports whether the center tap leaves
# between the feeds (even N) or at the top (odd N). The feeds and the center tap run out to the ring's outer edge,
# where the ports sit. The feed on TopMetal1 requires N >= 2 turns (see
# symmetric_octa_IHP: N == 1 feeds on TopMetal2 instead).
GROUND_LAYER = 67       # Metal5
GROUND_SPACING = 20.0   # µm, gap between inductor outer edge and the ground
GROUND_DEPTH = 20.0     # µm, width of the ground ring bars


# Built per instance rather than shared as a class attribute - see the note in
# tf_octa_c_ports.py. The dataset is built per instance too, by create_dataset().


def _input_parameters() -> InputParameterIterator:
    # Lengths in µm. The diameter is drawn on a log scale and kept when is_feasible rejects
    # a draw: turns, width and space are redrawn around it instead. A small spiral only
    # fits with one turn and thin lines, so a plain rejection threw out most small
    # diameters and left the median at 182 µm; kept, the diameter follows its log
    # distribution (seed 40: median 96 µm, 51% of 6000 samples below 100 µm, 29% below
    # 60 µm). Spirals under ~60 µm are then nearly all single-turn, as only those fit.
    # geometries/inductor_octa_coverage.png shows the coverage after a run.
    return InputParameterIterator(
        RangeParameter("turns", 1, 5, dtype=int),
        # 0.02 µm steps: symmetric_octa_IHP draws w and s on even hundredths,
        # so odd values would be built 0.01 µm off from what the table records.
        RangeParameter("width", 2.0, 15.0, step=0.02, sampling="log"),
        RangeParameter("space", 2.0, 6.0, step=0.02),
        RangeParameter("diameter", 30.0, 300.0, step=2.0, sampling="log"),
        picking_strategy="sobol",
        frequency=[1e9, 500e9],  # 1 GHz to 500 GHz
        kept_on_rejection=("diameter",),
    )


@dataclass
class InductorOcta(BaseGeometry):
    """
    Represents a symmetric octagonal spiral inductor geometry (IHP SG13G2).

    3-port spiral inductor (LA, LB and the center tap LC), ported from the
    gds2palace IHP example by Volker Muehlhaus. Requires N >= 2 turns, since the feedline sits on
    TopMetal1 (single-turn inductors feed on TopMetal2 instead).
    """

    name: str = "inductor_octa"
    # Conformal SiO2/passivation over TopMetal2 (gds2palace L6n2 study): the planar
    # stackup fills the gaps between turns with oxide and overstates the turn-to-turn
    # capacitance. Paired with refined_cellsize = 5 in the simcfg, the study's fast
    # "daily driver" setting; it meshes smaller than planar at 2 µm.
    stackup_xml: str = StackupXML.SG13G2_FEM_200um_passi3D
    simconfig_filename: str = os.path.join(os.path.dirname(__file__), "inductor_octa.simcfg")
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
            codec=FlatReImCodec(n_ports=3),
            # Scale inputs with the declared parameter ranges rather than the
            # min/max of whatever subset happens to be trained on.
            input_normalizer=MinMaxNormalizer(self.input_parameter_iterator),
            output_normalizer=StandardNormalizer(),
        )

    def electrical_parameters(self, ntwk: "rf.Network") -> dict[str, "np.ndarray"]:
        from orca.utils.postprocessing import inductor_parameters

        # Ports 1 and 2 feed the two ends (LA, LB); port 3, the center tap, is AC-grounded
        # as in differential use
        return inductor_parameters(ntwk, ends=(0, 1), shorted=(2,))

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
            includeCenterTap=True,
            LBE=False,
            forEM=True,
            ground_layer=GROUND_LAYER,
            ground_style="ring",
            ring_spacing=GROUND_SPACING,
            ring_width=GROUND_DEPTH,
            filename=output_path,
        )
        return output_path
