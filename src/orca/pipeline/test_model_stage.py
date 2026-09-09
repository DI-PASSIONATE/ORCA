from typing import Optional, Any, Dict, Callable
import os

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split

from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.pipeline.pipeline_stage import PipelineStage
from orca.training.datasets.geo_to_ntwk import GeoToNtwkDataset
from orca.training.predictors import (
    NetworkPredictor,
    OnnxNetworkPredictor,
    TorchNetworkPredictor,
)
from orca.utils.folder_structure import OrcaFolderStructure
from orca.utils.postprocessing import (
    calculate_electrical_parameters,
    median_relative_error,
    plot_rfic_transformer_metrics,
)


class ModelTester(PipelineStage):
    """
    Pipeline stage for evaluating a trained model against held-out simulation results.

    Predictions come from whatever is available: the model still held in the pipeline
    context, or otherwise an exported ONNX file. Either way the comparison is made in
    physical units, on geometries the model was never trained on.
    """

    def __init__(self, n_test_samples: Optional[int] = None, plot: bool = False):
        """
        Args:
            n_test_samples: Limit the evaluation to the first N held-out geometries.
                None evaluates all of them.
            plot: Plot predicted and reference metrics for each sample instead of
                computing errors.
        """
        super().__init__(name="Model Testing", index=6)
        self.n_test_samples = n_test_samples
        self.plot = plot

    def run(
        self,
        context: Dict[str, Any],
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        result_dir = OrcaFolderStructure.get_result_dir(context)
        test_df = context.get("test_df", None)

        # If the training stage was skipped, the test split has to be reproduced from the CSV
        if test_df is None:
            result_csv = OrcaFolderStructure.get_result_csv(context)

            if not os.path.exists(result_csv):
                logger.error(f"No result CSV file found for testing at {result_csv}.")
                return context

            _train_val_df, test_df = train_test_split(
                pd.read_csv(result_csv),
                test_size=0.15,  # Use the same test fraction as in training stage
                random_state=11,
            )

        test_dataset = GeoToNtwkDataset(directory=result_dir, data_df=test_df)
        if len(test_dataset) == 0:
            logger.error(f"No test networks could be loaded from {result_dir}.")
            return context

        try:
            predictor = self._build_predictor(context)
        except FileNotFoundError as e:
            logger.error(str(e))
            return context

        results = self.test_model(test_dataset, predictor, progress_callback)

        if results:
            logger.info(
                f"Mean absolute S-parameter error over {results['n_samples']} test geometries: "
                f"{results['mean_abs_s_error']:.4f}"
            )
            for param, error in results["electrical_parameters"].items():
                logger.info(f"Median relative error for {param}: {error:.2f}%")

        context["test_results"] = results
        return context

    def _build_predictor(self, context: Dict[str, Any]) -> NetworkPredictor:
        """
        Predict with the trained model if the training stage ran in this pipeline,
        otherwise fall back to the exported ONNX model on disk.
        """
        trained_model = context.get("trained_model", None)
        dataset = context.get("dataset", None)

        if trained_model is not None and dataset is not None:
            logger.info("Testing the trained model directly (no ONNX round-trip).")
            return TorchNetworkPredictor(trained_model, dataset)

        model_path = OrcaFolderStructure.get_model_path(context)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model in the pipeline context and no exported model at {model_path}. "
                "Run the ModelTrainer stage, or export a model first."
            )

        import onnxruntime

        logger.info(f"Testing the exported ONNX model at {model_path}.")
        geometry: BaseGeometry = context["geometry"]
        return OnnxNetworkPredictor(
            onnxruntime.InferenceSession(model_path), geometry.dataset.codec
        )

    def test_model(
        self,
        test_dataset: GeoToNtwkDataset,
        predictor: NetworkPredictor,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the predictor on the test dataset.

        Returns:
            dict: Mean absolute S-parameter error, the median relative error per
            electrical parameter, and how many geometries were evaluated.
        """
        num_samples = len(test_dataset)
        if self.n_test_samples is not None:
            num_samples = min(num_samples, self.n_test_samples)

        s_errors: list[float] = []
        param_errors: dict[str, list[float]] = {}

        for i in tqdm.tqdm(range(num_samples), desc="Testing samples"):
            input_params, ntwk_gt = test_dataset[i]
            ntwk_pred = predictor.predict(input_params, ntwk_gt.f)
            ntwk_pred.name = "Predicted"
            ntwk_gt.name = "Ground Truth"

            if self.plot:
                plot_rfic_transformer_metrics(ntwk_gt)
                plot_rfic_transformer_metrics(ntwk_pred)
                continue

            s_errors.append(float(np.abs(ntwk_pred.s - ntwk_gt.s).mean()))

            try:
                predicted = calculate_electrical_parameters(ntwk_pred)
                reference = calculate_electrical_parameters(ntwk_gt)
            except Exception as e:
                logger.debug(f"Could not compute electrical parameters for sample {i}: {e}")
                continue

            for param, gt in reference.items():
                error = median_relative_error(predicted[param], gt)
                if np.isfinite(error):
                    param_errors.setdefault(param, []).append(error)

            if progress_callback:
                progress_callback(
                    self.name, i + 1, num_samples, f"Tested {i + 1} of {num_samples} geometries."
                )

        if not s_errors:
            return {}

        return {
            "n_samples": len(s_errors),
            "mean_abs_s_error": float(np.mean(s_errors)),
            "electrical_parameters": {
                param: float(np.mean(errors)) for param, errors in sorted(param_errors.items())
            },
        }
