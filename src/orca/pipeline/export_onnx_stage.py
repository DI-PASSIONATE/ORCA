from typing import Optional, Any, Dict, Callable
import os
import torch
import onnx

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.training.onnx_wrapper import ONNXWrapper
from orca.utils.folder_structure import OrcaFolderStructure


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

        output_dir = OrcaFolderStructure.get_model_dir(context)
        output_path = OrcaFolderStructure.get_model_path(context)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logger.info(f"Exporting trained model to ONNX format at {output_path}.")

        wrapped_model = ONNXWrapper(
            trained_model.eval(),
            features=dataset.features,
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

        # Add valid ranges as metadata to the ONNX model
        onnx_model = onnx.load(output_path)
        ranges = geometry.input_parameter_iterator.get_ranges()

        for name in dataset.input_param_names:
            if name in ranges:
                min_val, max_val = ranges[name]
                meta = onnx_model.metadata_props.add()
                meta.key = f"{name}_min"
                meta.value = f"{min_val}"
                meta = onnx_model.metadata_props.add()
                meta.key = f"{name}_max"
                meta.value = f"{max_val}"

        onnx.save(onnx_model, output_path)

        context["model_path"] = output_path
        return context
