from typing import Optional, Any, Dict, Callable
import os
import pandas as pd

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.training.datasets.dataloader import train_val_test_dataset
from orca.training.train import test_model
from orca.utils.folder_structure import OrcaFolderStructure


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
        base_dir: str = OrcaFolderStructure.get_base_dir(context)

        test_dataset = context.get("test_dataset", None)

        # If test_dataset is not in context (i.e., training stage was skipped), we need to reload the result CSV and split again to get the test set
        if test_dataset is None:
            result_dir = context.get("result_dir", os.path.join(base_dir, "results"))
            result_csv = context.get(
                "result_csv", os.path.join(result_dir, f"{geometry.name}.csv")
            )

            if not os.path.exists(result_csv):
                logger.error(f"No result CSV file found for training at {result_csv}.")
                return context

            result_df = pd.read_csv(result_csv)  # Information
            _, _, test_dataset = train_val_test_dataset(result_df, geometry, result_dir)


        trained_model = context.get("trained_model", None)

        if trained_model is None:
            logger.error("No trained model found in context for testing.")
            return context

        loss = test_model(test_dataset=test_dataset, model=trained_model)


        logger.info(f"Testing completed with loss: {loss}")

        return context
