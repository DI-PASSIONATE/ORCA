import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import onnx
import torch

from orca.geometry.constraints import validate_constraint
from orca.logger import logger
from orca.pipeline.pipeline_stage import PipelineStage
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

        onnx.save(onnx_model, output_path)

        if progress_callback:
            progress_callback(self.name, 1, 1, f"Exported ONNX model to {output_path}.")

        context.model_path = output_path
        return context
