from typing import Optional, Any, Dict, Callable
import os
import torch

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.training.onnx_wrapper import ONNXWrapper


class OnnxExporter(PipelineStage):
    """
    Pipeline stage for exporting trained models to ONNX format.
    """

    def __init__(self):
        super().__init__(name="ONNX Exporter", index=5)

    def run(
        self,
        context: Dict[str, Any],
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        geometry: BaseGeometry = context["geometry"]
        base_dir: str = context.get("base_dir", os.getcwd())

        trained_model = context.get("trained_model", None)
        dataset = context.get("dataset", None)

        if trained_model is None or dataset is None:
            logger.error(
                "Trained model or dataset not found in context. Cannot export to ONNX."
            )
            return context

        output_dir = os.path.join(base_dir, "models")
        output_path = os.path.join(output_dir, f"{geometry.name}.onnx")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info(f"Exporting trained model to ONNX format at {output_path}.")

        # Export to ONNX with multiple inputs/outputs using ONNXWrapper
        torch.onnx.export(
            ONNXWrapper(
                trained_model,
                features=dataset.features,
                input_normalizer=dataset.input_normalizer,
                output_denormalizer=dataset.output_normalizer,
            ),
            tuple(
                torch.randn(1, 1, device=dataset.device)
                for _ in dataset.input_param_names
            ),
            input_names=dataset.input_param_names,
            output_names=dataset.output_param_names,
            f=output_path,
            external_data=False,
            dynamo=True,
        )

        context["onnx_model_path"] = output_path
        return context
