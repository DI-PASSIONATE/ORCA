"""The state that flows through every stage of an ORCA pipeline run.

One :class:`PipelineContext` is created by :meth:`orca.orca.ORCA.run` and handed
to each stage in turn. A stage reads the fields it needs, writes the fields it
owns, and returns the same object.

Field ownership is what keeps the pipeline honest, so it is recorded per group
below: everything a stage does not own it should treat as read-only. A stage
that runs without its inputs present (because an earlier stage was left out of
the pipeline) is expected to say so, rather than to fail on a missing key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    import torch.nn as nn

    from orca.geometry.base_geometry import BaseGeometry
    from orca.training.datasets.base_dataset import BaseDataset
    from orca.training.trainer import EpochResult

#: Fields left out of :meth:`PipelineContext.to_json_dict`, because they are
#: large binary or tabular objects whose ``str()`` says nothing useful. The
#: trained model and the dataset are written to disk by their own stages; the
#: test split is reproducible from the result CSV and the split seed.
_JSON_EXCLUDED = frozenset({"trained_model", "dataset", "test_df"})


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert *obj* to a JSON-safe structure.

    Enum keys and values become their ``.value``; anything else that is not a
    native JSON type is left for ``json.dump``'s ``default=str`` fallback.
    """
    if isinstance(obj, dict):
        return {
            (k.value if isinstance(k, Enum) else k): sanitize_for_json(v)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, Enum):
        return obj.value
    return obj


@dataclass(kw_only=True)
class PipelineContext:
    """Mutable state shared by the stages of a single pipeline run.

    The folder layout is derived from :attr:`base_dir` by the ``*_dir``/``*_csv``
    properties, so stages ask the context where files live instead of rebuilding
    paths. The four override fields let a run point at results that were produced
    elsewhere, which is what :meth:`orca.orca.ORCA.run` exposes to callers.
    """

    # --- Fixed for the whole run, written by ORCA.run -----------------------
    geometry: BaseGeometry
    """Geometry class being characterised. Read by every stage."""
    num_processes: int = 1
    """MPI ranks per Palace simulation, and worker count for GDS generation."""
    base_dir: str = "."
    """Root output directory; every derived path below hangs off this."""

    # --- Path overrides, written by ORCA.run --------------------------------
    result_dir_override: str | None = None
    """Directory of existing Touchstone results, instead of ``<base_dir>/results``."""
    result_csv_override: str | None = None
    """Parameter table describing those results, instead of ``<result_dir>/<name>.csv``."""
    model_dir_override: str | None = None
    """Directory for the exported model, instead of ``<base_dir>/models``."""
    model_path_override: str | None = None
    """Exported model file, instead of ``<model_dir>/<name>.onnx``."""

    # --- Written by GDSGenerator --------------------------------------------
    gds_csv: str | None = None
    """Parameter table for the generated GDS files. ``None`` until the stage runs."""

    # --- Written by GDSConverter --------------------------------------------
    palace_csv: str | None = None
    """Parameter table for the generated Palace models."""

    # --- Written by ModelTrainer --------------------------------------------
    trained_model: nn.Module | None = None
    """Best model from the training run, restored to its best-validation weights."""
    dataset: BaseDataset | None = None
    """The training split, carrying the feature pipeline and fitted normalizers.

    The ONNX exporter needs these to bake the same preprocessing into the graph,
    so this is the training split rather than a fresh dataset.
    """
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    """Hyperparameters actually trained with, tuned or supplied."""
    final_val_loss: float | None = None
    training_history: list[EpochResult] = field(default_factory=list)
    test_df: pd.DataFrame | None = None
    """Held-out split, kept so the testing stage evaluates the same rows."""

    # --- Written by OnnxExporter --------------------------------------------
    model_path: str | None = None
    """Where the ONNX model was written. ``None`` until the export stage runs."""

    # --- Written by ModelTester ---------------------------------------------
    test_results: dict[str, Any] = field(default_factory=dict)
    """Error metrics per electrical parameter; empty when nothing was evaluated."""

    # --- Derived folder layout -----------------------------------------------

    @property
    def geometry_dir(self) -> str:
        return os.path.join(self.base_dir, "geometries")

    @property
    def gds_csv_path(self) -> str:
        """Where the GDS generation stage writes its parameter table."""
        return os.path.join(self.geometry_dir, f"{self.geometry.name}.csv")

    @property
    def palace_sim_dir(self) -> str:
        return os.path.join(self.base_dir, "palace_sims")

    @property
    def palace_csv_path(self) -> str:
        """Where the GDS conversion stage writes its parameter table."""
        return os.path.join(self.palace_sim_dir, f"{self.geometry.name}.csv")

    @property
    def result_dir(self) -> str:
        """Directory holding the Touchstone results the model is trained on."""
        return self.result_dir_override or os.path.join(self.base_dir, "results")

    @property
    def result_csv(self) -> str:
        """Parameter table describing the files in :attr:`result_dir`."""
        return self.result_csv_override or os.path.join(
            self.result_dir, f"{self.geometry.name}.csv"
        )

    @property
    def model_dir(self) -> str:
        return self.model_dir_override or os.path.join(self.base_dir, "models")

    @property
    def onnx_path(self) -> str:
        """Where the export stage writes, and the testing stage looks for, the model."""
        return self.model_path_override or os.path.join(
            self.model_dir, f"{self.geometry.name}.onnx"
        )

    def to_json_dict(self) -> dict[str, Any]:
        """A JSON-serialisable view of this context, for the run record.

        Fields in :data:`_JSON_EXCLUDED` are skipped, and the derived paths are
        included so the record says where everything actually went.
        """
        record = {
            f.name: sanitize_for_json(getattr(self, f.name))
            for f in fields(self)
            if f.name not in _JSON_EXCLUDED
        }
        record["geometry"] = self.geometry.name
        record["paths"] = {
            "geometry_dir": self.geometry_dir,
            "palace_sim_dir": self.palace_sim_dir,
            "result_dir": self.result_dir,
            "result_csv": self.result_csv,
            "model_dir": self.model_dir,
            "onnx_path": self.onnx_path,
        }
        return record
