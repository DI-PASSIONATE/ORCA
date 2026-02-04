from typing import Optional, Any, Dict, Callable
import os

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger


class ModelTester(PipelineStage):
    """
    Pipeline stage for exporting trained models to ONNX format.
    """

    def __init__(self):
        super().__init__(name="Model Testing", index=6)

    def run(
        self,
        context: Dict[str, Any],
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        geometry: BaseGeometry = context["geometry"]
        base_dir: str = context.get("base_dir", os.getcwd())

        trained_model = context.get("trained_model", None)
        onnx_model_path = context.get("onnx_model_path", None)
        dataset = context.get("dataset", None)

        if trained_model is None or dataset is None:
            logger.error(
                "Trained model or dataset not found in context. Cannot export to ONNX."
            )
            return context

        logger.info("Testing not implemented yet.")

        return context
