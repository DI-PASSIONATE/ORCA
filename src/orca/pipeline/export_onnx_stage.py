import json
import os
import shutil
import subprocess
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import onnx
import pandas as pd
import torch

from orca.geometry.constraints import validate_constraint
from orca.logger import logger
from orca.pipeline.pipeline_stage import PipelineStage
from orca.simulation.combine_snp_results import touchstone_type_of
from orca.training.onnx_wrapper import ONNXWrapper

if TYPE_CHECKING:
    from orca.geometry.base_geometry import BaseGeometry
    from orca.pipeline.context import PipelineContext


class OnnxExporter(PipelineStage):
    """
    Pipeline stage for exporting trained models to ONNX format.
    """

    def __init__(self):
        super().__init__(name="ONNX Exporter", index=5)

    def run(
        self,
        context: "PipelineContext",
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> "PipelineContext":
        geometry: BaseGeometry = context.geometry

        trained_model = context.trained_model
        dataset = context.dataset

        if trained_model is None or dataset is None:
            logger.error(
                "Trained model or dataset not found in context. Cannot export to ONNX."
            )
            return context

        output_dir = context.model_dir
        output_path = context.onnx_path

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Exporting trained model to ONNX format at {output_path}.")

        wrapped_model = ONNXWrapper(
            trained_model.eval(),
            input_normalizer=dataset.input_normalizer,
            output_denormalizer=dataset.output_normalizer,
        ).eval()

        batch = torch.export.Dim("batch_size")

        # Export to ONNX with multiple inputs/outputs using ONNXWrapper
        torch.onnx.export(
            wrapped_model,
            args=tuple(
                torch.randn(1, 1, device=dataset.device)
                for _ in dataset.input_param_names
            ),
            input_names=dataset.input_param_names,
            output_names=dataset.output_param_names,
            dynamic_shapes=(tuple({0: batch} for _ in dataset.input_param_names),),
            f=output_path,
            external_data=False,
            dynamo=True,
        )

        onnx_model = onnx.load(output_path)

        # The dynamo exporter annotates every node with the Python stack trace
        # FX provenance it came from.
        # bloats the file and leaks local paths into a model that may be shared.
        for node in onnx_model.graph.node:
            del node.metadata_props[:]

        # Add valid ranges as metadata to the ONNX model
        ranges = geometry.input_parameter_iterator.get_ranges()
        meta = onnx_model.metadata_props.add()
        meta.key = "input_parameter_ranges"
        meta.value = json.dumps(ranges)

        # The ranges are a box; the geometry may only be buildable in part of it,
        # and the model was trained on that part alone. Record the constraints so
        # a consumer can tell an unbuildable query from a bad prediction.
        constraints = geometry.feasibility_constraints()
        if constraints:
            for expression in constraints:
                validate_constraint(expression, list(ranges))
            meta = onnx_model.metadata_props.add()
            meta.key = "input_constraints"
            meta.value = json.dumps(constraints)

        # Record which physical properties are guaranteed by construction, so consumers
        # (e.g. COBRA) know whether the predicted S-matrix is passive/reciprocal by
        # construction. The architecture and the output codec both contribute: an
        # upper-triangle codec makes the response reciprocal whatever sits behind it.
        guarantees = getattr(trained_model, "guarantees", None)
        if guarantees is not None:
            guarantees = guarantees | dataset.codec.guarantees
            meta = onnx_model.metadata_props.add()
            meta.key = "physics_guarantees"
            meta.value = json.dumps(guarantees.as_dict())

        # The ports of each winding, so COBRA can give the predicted 0 Hz point the DC
        # structure of real windings (see BaseGeometry.windings).
        windings = geometry.windings()
        if windings:
            validate_windings(windings, geometry.n_ports)
            meta = onnx_model.metadata_props.add()
            meta.key = "windings"
            meta.value = json.dumps(windings)

        # Which Palace result the model learned (raw or de-embedded, with or without the
        # 0 Hz point), so an EM verification compares against the same reference plane.
        touchstone_type = training_touchstone_type(context.result_csv)
        if touchstone_type is not None:
            meta = onnx_model.metadata_props.add()
            meta.key = "touchstone_type"
            meta.value = touchstone_type

        # Where the model came from, for reports and for tracing a model back to the ORCA
        # that built it (e.g. one trained on data from before a simulation fix).
        meta = onnx_model.metadata_props.add()
        meta.key = "orca_version"
        meta.value = orca_version()

        onnx.save(onnx_model, output_path)

        if progress_callback:
            progress_callback(self.name, 1, 1, f"Exported ONNX model to {output_path}.")

        context.model_path = output_path
        return context


def validate_windings(windings: list[list[int]], n_ports: int) -> None:
    """
    Check that *windings* are disjoint, non-empty groups of ports 1 to *n_ports*.

    Raises:
        ValueError: If they are not.
    """
    ports = [port for winding in windings for port in winding]
    if (
        not all(winding for winding in windings)
        or not all(type(port) is int and 1 <= port <= n_ports for port in ports)
        or len(set(ports)) != len(ports)
    ):
        raise ValueError(
            f"windings() returned {windings}; it must be disjoint, non-empty lists of port "
            f"numbers 1 to {n_ports}."
        )


def training_touchstone_type(result_csv: str) -> str | None:
    """
    The Touchstone variant of the result files in *result_csv*, the table the model was
    trained on, or None if the table is missing or mixes variants.
    """
    if not os.path.exists(result_csv):
        return None
    variants = {touchstone_type_of(name) for name in pd.read_csv(result_csv)["name"]}
    if len(variants) != 1:
        if variants:
            logger.warning(
                f"The training data mixes Touchstone variants {sorted(variants)}; the "
                "touchstone_type metadata is left out."
            )
        return None
    return variants.pop()


def orca_version() -> str:
    """
    The installed ORCA release, plus ``git describe`` of its checkout when it runs from one.

    An editable install reports the last release even with changes on top of it, so the
    commit is what identifies such a build, e.g. ``"2.2.0 (git v2.2.0-1-gccf1103-dirty)"``.
    """
    try:
        release = version("orca-rfic")
    except PackageNotFoundError:
        release = "unknown"
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # src/orca -> the repository root, which only a source checkout has
    git = shutil.which("git")
    if git is None or not os.path.exists(os.path.join(package_dir, "..", "..", ".git")):
        return release
    try:
        described = subprocess.run(
            [git, "-C", package_dir, "describe", "--always", "--dirty", "--tags"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return release
    return f"{release} (git {described})" if described else release
