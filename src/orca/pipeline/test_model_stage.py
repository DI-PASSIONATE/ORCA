from typing import Optional, Any, Dict, Callable
import os
import onnxruntime
import pandas as pd
import tqdm

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
            _, _, test_df = train_val_test_dataset(result_df)

        test_dataset = GeoToNtwkDataset(directory=OrcaFolderStructure.get_result_dir(context), data_df=test_df)


        trained_model = OrcaFolderStructure.get_model_path(context)

        if trained_model is None:
            logger.error("No trained model found in context for testing.")
            return context

        loss = self.test_model(test_dataset=test_dataset, geometry=geometry, onnx_path=OrcaFolderStructure.get_model_path(context))

        logger.info(f"Testing completed with loss: {loss}")

        return context
    
    def test_model(self, test_dataset, geometry, onnx_path, plot=False):
        """
        Evaluates the trained model on the test dataset and returns the loss.
        """
        onnx_session = onnxruntime.InferenceSession(onnx_path)
        num_samples = len(test_dataset)

        param_errors =  {}

        for i in tqdm.tqdm(range(num_samples), desc="Testing samples"):
            input_params, ntwk_gt = test_dataset[i]
            ntwk_pred = geometry.inference_snp(onnx_session, input_params)
            ntwk_pred.name = "Predicted"
            ntwk_gt.name = "Ground Truth"

            if plot:
                plot_rfic_transformer_metrics(ntwk_gt)
                plot_rfic_transformer_metrics(ntwk_pred)
                continue

            ep1 = calculate_electrical_parameters(ntwk_pred)
            ep2 = calculate_electrical_parameters(ntwk_gt)


            # Calculate percentage error for each parameter and average them
            for param in ep1.keys():
                pred = ep1[param]
                gt = ep2[param]


                # Calculate mean error without NaN or inf values
                try:
                    logger.debug(f"pred: {pred}, gt: {gt}")
                    error_per_freq_point = np.abs(pred - gt) / (np.mean(np.abs(gt)) + 1e-10) * 100 
                    valid_mask = ~np.isinf(error_per_freq_point) & ~np.isnan(error_per_freq_point)
                    error = np.mean(error_per_freq_point[valid_mask])# * 100  # Convert to percentage
                    logger.debug(f"Sample {input_params}, Parameter {param}, Error per frequency point: {error_per_freq_point[valid_mask]}, Mean Error: {error:.2f}%")
                    if not np.isinf(error) and not np.isnan(error):
                        if param not in param_errors:
                            param_errors[param] = []
                        param_errors[param].append(error)
                        
                except Exception as e:
                    logger.debug(f"Error calculating percentage error for sample {i}, parameter {param}: {e}")
                
        # Calculate average error for each parameter across all samples
        average_errors = {param: np.mean(errors) for param, errors in param_errors.items()}
        for param, avg_error in average_errors.items():
            logger.info(f"Average percentage error for {param} across all test samples: {avg_error:.2f}%")
        return average_errors
        

            


