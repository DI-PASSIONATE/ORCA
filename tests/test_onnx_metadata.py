"""Tests for the metadata OnnxExporter writes for COBRA beyond the ranges and constraints:
the ports of each winding and the Touchstone variant the model was trained on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from orca.geometry.presets import InductorOcta, TransformerOcta
from orca.pipeline.export_onnx_stage import training_touchstone_type, validate_windings
from orca.simulation.combine_snp_results import touchstone_filename, touchstone_type_of


@pytest.mark.parametrize(
    ("geometry", "windings"),
    [(InductorOcta, [[1, 2, 3]]), (TransformerOcta, [[1, 2, 5], [3, 4, 6]])],
    ids=["inductor", "transformer"],
)
def test_presets_declare_their_windings(geometry, windings):
    instance = geometry()

    assert instance.windings() == windings
    validate_windings(instance.windings(), instance.n_ports)


@pytest.mark.parametrize(
    "windings",
    [[[1, 2], []], [[0, 1]], [[1, 4]], [[1, 2], [2, 3]], [[1.0, 2]]],
    ids=["empty-winding", "port-zero", "beyond-the-ports", "shared-port", "float-port"],
)
def test_malformed_windings_are_refused(windings):
    with pytest.raises(ValueError, match="windings"):
        validate_windings(windings, n_ports=3)


@pytest.mark.parametrize("variant", ["normal", "dc", "deembedded", "dc_deembedded"])
def test_the_variant_is_read_back_from_the_file_name(variant):
    assert touchstone_type_of(touchstone_filename("tf_octa_c_ports_12", 6, variant)) == variant


def test_all_writes_the_fully_corrected_variant():
    assert touchstone_type_of(touchstone_filename("inductor_octa_3", 3, "all")) == "dc_deembedded"


def _table(tmp_path, names):
    path = tmp_path / "results.csv"
    pd.DataFrame({"name": names, "turns": range(len(names))}).to_csv(path, index=False)
    return str(path)


def test_the_training_variant_comes_from_the_result_table(tmp_path):
    names = [touchstone_filename(f"inductor_octa_{index}", 3, "dc") for index in range(3)]

    assert training_touchstone_type(_table(tmp_path, names)) == "dc"


def test_a_mixed_or_missing_table_declares_no_variant(tmp_path):
    mixed = [touchstone_filename("a_1", 3, "dc"), touchstone_filename("a_2", 3, "dc_deembedded")]

    assert training_touchstone_type(_table(tmp_path, mixed)) is None
    assert training_touchstone_type(str(tmp_path / "absent.csv")) is None


def test_the_version_names_the_commit_only_in_a_checkout(monkeypatch):
    from importlib.metadata import version

    from orca.pipeline import export_onnx_stage

    release = version("orca-rfic")
    assert export_onnx_stage.orca_version().startswith(release)

    monkeypatch.setattr(export_onnx_stage.shutil, "which", lambda _name: None)
    assert export_onnx_stage.orca_version() == release
