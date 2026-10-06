"""Tests for simulate_geometry, the single-sample Palace entry point COBRA uses."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import skrf as rf

from orca.geometry.presets import InductorOcta
from orca.simulation import geometry_simulation
from orca.simulation.combine_snp_results import touchstone_filename
from orca.simulation.geometry_simulation import SimulationError, simulate_geometry

# A single turn feeds on TopMetal2, so InductorOcta changes its ports for these
SINGLE_TURN = {"turns": 1, "width": 5.0, "space": 3.0, "diameter": 200.0}
# Every Touchstone variant gets its own S value, so a test can tell which one was read
S_VALUE = {"dc_deembedded": 0.1, "deembedded": 0.2, "dc": 0.3, "normal": 0.4}


def _write_result(output_dir: str, name: str, variant: str) -> None:
    ntwk = rf.Network(
        frequency=rf.Frequency.from_f(np.array([1e9, 2e9]), unit="Hz"),
        s=np.full((2, 3, 3), S_VALUE[variant], dtype=complex),
        z0=50,
    )
    path = os.path.join(output_dir, touchstone_filename(name, 3, variant))
    ntwk.write_touchstone(os.path.splitext(path)[0])


@pytest.fixture
def fake_palace(tmp_path, monkeypatch):
    """Stubs meshing and Palace; returns the calls they received."""
    calls: dict = {"quality": 0.01, "succeeded": True, "variants": ["dc_deembedded"]}
    sim_path = tmp_path / "out" / "palace_model"

    def create_palace_model(**kwargs):
        calls["mesh"] = kwargs
        sim_path.mkdir(parents=True, exist_ok=True)
        (sim_path / "config.json").write_text("{}")
        return kwargs["geometry_name"], kwargs["params"], "config.json", str(sim_path), "output", calls["quality"]

    def run_palace(**kwargs):
        calls["palace"] = kwargs
        if calls["succeeded"]:
            for variant in calls["variants"]:
                _write_result(kwargs["result_dir"], "sample", variant)
        return calls["succeeded"]

    monkeypatch.setattr(geometry_simulation, "_create_palace_model", create_palace_model)
    monkeypatch.setattr(geometry_simulation, "run_palace", run_palace)
    calls["sim_path"] = sim_path
    return calls


def test_simulates_with_the_samples_ports_and_patched_config(tmp_path, fake_palace):
    geometry = InductorOcta()
    output_dir = str(tmp_path / "out")

    ntwk = simulate_geometry(
        geometry, SINGLE_TURN, output_dir, name="sample", palace_executable="palace", num_processes=4
    )

    assert ntwk.nports == 3
    assert ntwk.s[0, 0, 0] == pytest.approx(S_VALUE["dc_deembedded"])
    assert os.path.isfile(os.path.join(output_dir, "sample.gds"))
    assert fake_palace["mesh"]["ports"] == geometry.ports_for(SINGLE_TURN)
    assert fake_palace["mesh"]["simconfig_filename"] == geometry.simconfig_filename
    assert fake_palace["palace"]["cmd"] == "palace -np 4 config.json"
    config = json.loads((fake_palace["sim_path"] / "config.json").read_text())
    assert config["Solver"]["Linear"]["Type"] == "SuperLU"


def test_falls_back_to_a_less_corrected_result(tmp_path, fake_palace):
    fake_palace["variants"] = ["deembedded", "normal"]
    output_dir = str(tmp_path / "out")

    ntwk = simulate_geometry(InductorOcta(), SINGLE_TURN, output_dir, name="sample")

    assert ntwk.s[0, 0, 0] == pytest.approx(S_VALUE["deembedded"])


@pytest.mark.parametrize("quality", [0.0, float("nan"), float("-inf")])
def test_degenerate_mesh_is_not_simulated(tmp_path, fake_palace, quality):
    fake_palace["quality"] = quality

    with pytest.raises(SimulationError, match="element quality"):
        simulate_geometry(InductorOcta(), SINGLE_TURN, str(tmp_path / "out"), name="sample")
    assert "palace" not in fake_palace


def test_failed_palace_run_raises(tmp_path, fake_palace):
    fake_palace["succeeded"] = False

    with pytest.raises(SimulationError, match="failed"):
        simulate_geometry(InductorOcta(), SINGLE_TURN, str(tmp_path / "out"), name="sample")


def test_missing_result_raises(tmp_path, fake_palace):
    fake_palace["variants"] = []

    with pytest.raises(SimulationError, match="no Touchstone file"):
        simulate_geometry(InductorOcta(), SINGLE_TURN, str(tmp_path / "out"), name="sample")


def test_infeasible_parameters_are_not_drawn(tmp_path, fake_palace):
    infeasible = {"turns": 5, "width": 15.0, "space": 6.0, "diameter": 30.0}

    with pytest.raises(SimulationError, match="is_feasible"):
        simulate_geometry(InductorOcta(), infeasible, str(tmp_path / "out"), name="sample")
    assert not os.path.exists(tmp_path / "out" / "sample.gds")
    assert "mesh" not in fake_palace
