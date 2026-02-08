from typing import Optional, Any, Dict, Callable
import os
import onnxruntime
import pandas as pd

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.training.datasets.dataloader import train_val_test_dataset
from orca.training.train import test_model
from orca.utils.folder_structure import OrcaFolderStructure
from orca.training.datasets.geo_to_ntwk import GeoToNtwkDataset

from orca.utils.postprocessing import *

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

        test_df = context.get("test_df", None)

        # If test_dataset is not in context (i.e., training stage was skipped), we need to reload the result CSV and split again to get the test set
        if test_df is None:
            result_dir = OrcaFolderStructure.get_result_dir(context)
            result_csv = OrcaFolderStructure.get_result_csv(context)

            if not os.path.exists(result_csv):
                logger.error(f"No result CSV file found for training at {result_csv}.")
                return context

            result_df = pd.read_csv(result_csv)  # Information
            _, _, test_df = train_val_test_dataset(result_df, geometry, result_dir)

        test_dataset = GeoToNtwkDataset(directory=OrcaFolderStructure.get_result_dir(context), data_df=test_df)


        trained_model = OrcaFolderStructure.get_model_path(context)

        if trained_model is None:
            logger.error("No trained model found in context for testing.")
            return context

        loss = self.test_model(test_dataset=test_dataset, geometry=geometry, onnx_path=OrcaFolderStructure.get_model_path(context))

        logger.info(f"Testing completed with loss: {loss}")

        return context
    
    def test_model(self, test_dataset, geometry, onnx_path):
        """
        Evaluates the trained model on the test dataset and returns the loss.
        """
        onnx_session = onnxruntime.InferenceSession(onnx_path)
        total_loss = 0.0
        num_samples = len(test_dataset)

        for i in range(num_samples):
            input_params, ntwk_gt = test_dataset[i]
            ntwk_pred = geometry.inference_snp(onnx_session, input_params)
            ntwk_pred.name = "Predicted"
            ntwk_gt.name = "Ground Truth"

            plot_rfic_transformer_metrics(ntwk_pred)
            plot_rfic_transformer_metrics(ntwk_gt)


