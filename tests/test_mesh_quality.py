"""Tests for the mesh quality check of the GDS conversion stage."""

from __future__ import annotations

import math
import os

import pandas as pd

import orca.pipeline.gds_conversion_stage as conversion_stage
from orca.geometry.presets import InductorOcta
from orca.pipeline.context import PipelineContext
from orca.pipeline.gds_conversion_stage import GDSConverter


def _tetrahedron_mesh(path, apex_height: float) -> str:
    """A one-tetrahedron mesh in gmsh's MSH 2.2 format; height 0 makes it flat."""
    with open(path, "w") as f:
        f.write(
            "$MeshFormat\n2.2 0 8\n$EndMeshFormat\n"
            "$Nodes\n4\n1 0 0 0\n2 1 0 0\n3 0 1 0\n"
            f"4 0.25 0.25 {apex_height}\n$EndNodes\n"
            "$Elements\n1\n1 4 2 1 1 1 2 3 4\n$EndElements\n"
        )
    return str(path)


QUALITY = {"good.gds": 0.01, "poor.gds": 5e-5, "flat.gds": -1.7e-14, "corrupt.gds": float("-inf")}


def _fake_conversion(geometry_name, params, output_dir, gds_filename, stackup_xml,
                     simconfig_filename, show_mesh_results=False, ports=None):
    sim_path = os.path.join(output_dir, "palace_sims", f"{geometry_name[:-4]}_data")
    os.makedirs(sim_path, exist_ok=True)
    open(os.path.join(sim_path, "config.json"), "w").close()
    return geometry_name, params, "config.json", sim_path, sim_path, QUALITY[geometry_name]


def test_flat_meshes_are_left_out_and_reported(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(conversion_stage, "create_palace_model_from_gds", _fake_conversion)
    context = PipelineContext(geometry=InductorOcta(), base_dir=str(tmp_path), num_processes=1)
    os.makedirs(context.geometry_dir, exist_ok=True)
    params = {"turns": 2, "width": 4.0, "space": 3.0, "diameter": 120.0}
    pd.DataFrame([{"name": name} | params for name in QUALITY]).to_csv(
        context.gds_csv_path, index=False
    )

    context = GDSConverter().run(context)

    assert sorted(pd.read_csv(context.palace_csv_path)["name"]) == ["good.gds", "poor.gds"]
    report = pd.read_csv(context.mesh_report_path).set_index("name")["worst_element_quality"]
    assert all(
        report[name] == q or math.isclose(report[name], q, abs_tol=1e-15)
        for name, q in QUALITY.items()
    )
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "2 of 4 meshes contain flat, inverted or corrupt elements" in messages
    assert "refined_cellsize = 2 µm" in messages
    assert "1 of 4 meshes have poorly shaped elements" in messages
