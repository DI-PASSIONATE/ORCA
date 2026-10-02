"""End-to-end test of the GDS generation stage with a geometry that writes stub layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from orca.geometry.base_geometry import BaseGeometry
from orca.geometry.input_parameters import ChoiceParameter, InputParameterIterator, RangeParameter
from orca.pipeline.context import PipelineContext
from orca.pipeline.gds_gen_stage import GDSGenerator


def _parameters() -> InputParameterIterator:
    return InputParameterIterator(
        RangeParameter("turns", 1, 4, dtype=int),
        RangeParameter("diameter", 20.0, 200.0, step=1.0, sampling="log"),
        RangeParameter("width", 2.0, 8.0),
        ChoiceParameter("layer", [126, 134]),
    )


@dataclass
class StubGeometry(BaseGeometry):
    name: str = "stub"
    stackup_xml: str = ""
    simconfig_filename: str = ""
    input_parameter_iterator: InputParameterIterator = field(default_factory=_parameters)

    def is_feasible(self, params: dict[str, Any]) -> bool:
        return params["diameter"] >= 10 * params["turns"] * params["width"] / 2

    @staticmethod
    def create_gds_file(name: str, output_path: str, params: dict[str, Any]) -> str:  # noqa: ARG004
        with open(output_path, "w") as f:
            f.write(repr(params))
        return output_path

    def create_dataset(self):
        raise NotImplementedError


def test_generator_lays_out_the_draws_and_plots_their_coverage(tmp_path):
    context = PipelineContext(
        geometry=StubGeometry(), base_dir=str(tmp_path), num_processes=2, seed=4
    )

    context = GDSGenerator(num_samples=40).run(context)

    with open(context.gds_csv_path) as f:
        assert len(f.readlines()) == 41  # header + one row per layout
    with open(context.gds_coverage_plot_path, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_coverage_plot_can_be_switched_off(tmp_path):
    context = PipelineContext(geometry=StubGeometry(), base_dir=str(tmp_path), num_processes=1)

    context = GDSGenerator(num_samples=5, plot_coverage=False).run(context)

    assert not (tmp_path / "geometries" / "stub_coverage.png").exists()
