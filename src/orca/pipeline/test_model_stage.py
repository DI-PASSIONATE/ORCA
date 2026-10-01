import itertools
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import tqdm

from orca.logger import logger
from orca.pipeline.pipeline_stage import PipelineStage
from orca.pipeline.training_stage import order_parameter_columns
from orca.training.datasets.geo_to_ntwk import GeoToNtwkDataset
from orca.training.predictors import (
    NetworkPredictor,
    OnnxNetworkPredictor,
    TorchNetworkPredictor,
)
from orca.utils.postprocessing import (
    calculate_electrical_parameters,
    median_relative_error,
    plot_rfic_transformer_metrics,
)

if TYPE_CHECKING:
    import skrf as rf

    from orca.geometry.base_geometry import BaseGeometry
    from orca.pipeline.context import PipelineContext


class ModelTester(PipelineStage):
    """
    Pipeline stage for evaluating a trained model against held-out simulation results.

    Predictions come from whatever is available: the model still held in the pipeline
    context, or otherwise an exported ONNX file. Either way the comparison is made in
    physical units, on geometries the model was never trained on.
    """

    def __init__(
        self, n_test_samples: int | None = None, plot: bool = False, n_frequency_bands: int = 5
    ):
        """
        Args:
            n_test_samples: Limit the evaluation to the first N held-out geometries.
                None evaluates all of them.
            plot: Plot predicted and reference metrics for each sample instead of
                computing errors.
            n_frequency_bands: Number of equally wide frequency bands the S-parameter
                error is also reported for.
        """
        super().__init__(name="Model Testing", index=6)
        self.n_test_samples = n_test_samples
        self.plot = plot
        self.n_frequency_bands = n_frequency_bands

    def run(
        self,
        context: "PipelineContext",
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> "PipelineContext":
        result_dir = context.result_dir
        test_df = context.test_df

        # If the training stage was skipped, the held-out rows come from the split it recorded
        if test_df is None:
            test_df = self._load_test_rows(context)
            if test_df is None:
                return context
        test_df = order_parameter_columns(test_df, context.geometry)
        if self.n_test_samples is not None:
            # Trimmed before loading, so the Touchstone files left out are never parsed
            test_df = test_df.head(self.n_test_samples)

        test_dataset = GeoToNtwkDataset(directory=result_dir, data_df=test_df)
        if len(test_dataset) == 0:
            logger.error(f"No test networks could be loaded from {result_dir}.")
            return context

        try:
            predictor = self._build_predictor(context)
        except FileNotFoundError as e:
            logger.error(str(e))
            return context

        per_geometry = self.evaluate(test_dataset, predictor, progress_callback)
        results = self.summarize(per_geometry)

        if results:
            os.makedirs(context.model_dir, exist_ok=True)
            per_geometry.to_csv(context.test_errors_csv_path, index=False)
            self._log_summary(results)
            logger.info(f"Errors of every test geometry written to {context.test_errors_csv_path}.")

        context.test_results = results
        return context

    def _load_test_rows(self, context: "PipelineContext") -> pd.DataFrame | None:
        """
        The rows of the result table the model was not trained on, read from the split
        the training stage recorded. Without that record every row is used, which only
        gives honest errors for results the model has never seen.
        """
        result_csv = context.result_csv
        if not os.path.exists(result_csv):
            logger.error(f"No result CSV file found for testing at {result_csv}.")
            return None
        result_df = pd.read_csv(result_csv)

        split_csv = context.split_csv_path
        if not os.path.exists(split_csv):
            logger.warning(
                f"No split record at {split_csv}, so the Model Trainer did not run for this "
                f"model folder. Testing on all {len(result_df)} rows of {result_csv}; if the "
                "model was trained on any of them, the reported errors are too optimistic."
            )
            return result_df

        split = pd.read_csv(split_csv)
        test_names = set(split.loc[split["split"] == "test", "name"])
        test_df = result_df[result_df["name"].isin(test_names)]
        logger.info(f"Testing on the {len(test_df)} held-out geometries listed in {split_csv}.")
        return test_df

    def _build_predictor(self, context: "PipelineContext") -> NetworkPredictor:
        """
        Predict with the trained model if the training stage ran in this pipeline,
        otherwise fall back to the exported ONNX model on disk.
        """
        trained_model = context.trained_model
        dataset = context.dataset

        if trained_model is not None and dataset is not None:
            logger.info("Testing the trained model directly (no ONNX round-trip).")
            return TorchNetworkPredictor(trained_model, dataset)

        model_path = context.onnx_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model in the pipeline context and no exported model at {model_path}. "
                "Run the ModelTrainer stage, or export a model first."
            )

        import onnxruntime

        logger.info(f"Testing the exported ONNX model at {model_path}.")
        geometry: BaseGeometry = context.geometry
        return OnnxNetworkPredictor(
            onnxruntime.InferenceSession(model_path), geometry.dataset.codec
        )

    def test_model(
        self,
        test_dataset: GeoToNtwkDataset,
        predictor: NetworkPredictor,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluates the predictor on the test dataset.

        Returns:
            dict: The summary of :meth:`summarize`; empty if nothing was evaluated.
        """
        return self.summarize(self.evaluate(test_dataset, predictor, progress_callback))

    def evaluate(
        self,
        test_dataset: GeoToNtwkDataset,
        predictor: NetworkPredictor,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> pd.DataFrame:
        """
        The errors of every test geometry, one row each.

        Returns:
            pd.DataFrame: ``name``, the geometry parameters, the mean and maximum
            absolute S-parameter error, the mean absolute error per frequency band
            (``s_error <band>``) and the median relative error of each electrical
            parameter in percent (``<param> error %``). Empty in plot mode.
        """
        num_samples = len(test_dataset)
        if self.n_test_samples is not None:
            num_samples = min(num_samples, self.n_test_samples)

        rows: list[dict[str, Any]] = []
        for i in tqdm.tqdm(range(num_samples), desc="Testing samples"):
            input_params, ntwk_gt = test_dataset[i]
            ntwk_pred = predictor.predict(input_params, ntwk_gt.f)
            ntwk_pred.name = "Predicted"
            ntwk_gt.name = "Ground Truth"

            if self.plot:
                plot_rfic_transformer_metrics(ntwk_gt)
                plot_rfic_transformer_metrics(ntwk_pred)
                continue

            abs_error = np.abs(ntwk_pred.s - ntwk_gt.s)  # (n_freq, n_ports, n_ports)
            per_frequency = abs_error.mean(axis=(1, 2))
            row: dict[str, Any] = {
                "name": test_dataset.names[i],
                **dict(zip(test_dataset.input_param_names, map(float, input_params), strict=True)),
                "mean_abs_s_error": float(abs_error.mean()),
                "max_abs_s_error": float(abs_error.max()),
            }
            for label, in_band in self._frequency_bands(ntwk_gt.f).items():
                row[f"s_error {label}"] = float(per_frequency[in_band].mean())
            row |= self._electrical_errors(ntwk_pred, ntwk_gt, test_dataset.names[i])
            rows.append(row)

            if progress_callback:
                progress_callback(
                    self.name, i + 1, num_samples, f"Tested {i + 1} of {num_samples} geometries."
                )

        return pd.DataFrame(rows)

    @staticmethod
    def summarize(per_geometry: pd.DataFrame, n_worst: int = 5) -> dict[str, Any]:
        """
        Condense the per-geometry errors of :meth:`evaluate`.

        Means hide the geometries a model gets badly wrong, so the spread is reported
        too: percentiles of the S-parameter error, the worst geometries, and the
        error per frequency band.

        Returns:
            dict: ``n_samples``; ``mean_abs_s_error``; ``s_error_percentiles`` (p50, p95,
            max); ``s_error_by_band``; ``worst_geometries`` (name to error, worst first);
            and per electrical parameter the mean (``electrical_parameters``) and 95th
            percentile (``electrical_parameters_p95``) over the geometries of the
            median relative error in percent. Empty if nothing was evaluated.
        """
        if per_geometry.empty:
            return {}

        s_error = per_geometry["mean_abs_s_error"]
        band_columns = [c for c in per_geometry.columns if c.startswith("s_error ")]
        param_columns = sorted(c for c in per_geometry.columns if c.endswith(" error %"))
        worst = per_geometry.nlargest(n_worst, "mean_abs_s_error")

        return {
            "n_samples": len(per_geometry),
            "mean_abs_s_error": float(s_error.mean()),
            "s_error_percentiles": {
                "p50": float(s_error.quantile(0.5)),
                "p95": float(s_error.quantile(0.95)),
                "max": float(s_error.max()),
            },
            "s_error_by_band": {
                c.removeprefix("s_error "): float(per_geometry[c].mean()) for c in band_columns
            },
            "worst_geometries": dict(
                zip(worst["name"], map(float, worst["mean_abs_s_error"]), strict=True)
            ),
            "electrical_parameters": {
                c.removesuffix(" error %"): float(per_geometry[c].mean()) for c in param_columns
            },
            "electrical_parameters_p95": {
                c.removesuffix(" error %"): float(per_geometry[c].quantile(0.95))
                for c in param_columns
            },
        }

    def _frequency_bands(self, frequencies: np.ndarray) -> dict[str, np.ndarray]:
        """Masks selecting each of ``n_frequency_bands`` equally wide bands, by label."""
        edges = np.linspace(frequencies.min(), frequencies.max(), self.n_frequency_bands + 1)
        bands = {}
        for i, (low, high) in enumerate(itertools.pairwise(edges)):
            last = i == self.n_frequency_bands - 1
            in_band = (frequencies >= low) & ((frequencies <= high) if last else (frequencies < high))
            if in_band.any():
                bands[f"{low / 1e9:.3g}-{high / 1e9:.3g} GHz"] = in_band
        return bands

    @staticmethod
    def _electrical_errors(
        predicted_ntwk: "rf.Network", reference_ntwk: "rf.Network", name: str
    ) -> dict[str, float]:
        """Median relative error in percent of each electrical parameter that can be derived."""
        try:
            predicted = calculate_electrical_parameters(predicted_ntwk)
            reference = calculate_electrical_parameters(reference_ntwk)
        except Exception as e:  # noqa: BLE001 - skip samples whose metrics cannot be derived
            logger.debug(f"Could not compute electrical parameters for {name}: {e}")
            return {}

        errors = {}
        for param, gt in reference.items():
            error = median_relative_error(predicted[param], gt)
            if np.isfinite(error):
                errors[f"{param} error %"] = error
        return errors

    def _log_summary(self, results: dict[str, Any]) -> None:
        percentiles = results["s_error_percentiles"]
        logger.info(
            f"Mean absolute S-parameter error over {results['n_samples']} test geometries: "
            f"{results['mean_abs_s_error']:.4f} (median {percentiles['p50']:.4f}, "
            f"95th percentile {percentiles['p95']:.4f}, worst {percentiles['max']:.4f})"
        )
        for band, error in results["s_error_by_band"].items():
            logger.info(f"Mean absolute S-parameter error at {band}: {error:.4f}")
        worst = ", ".join(f"{name} ({e:.4f})" for name, e in results["worst_geometries"].items())
        logger.info(f"Worst geometries: {worst}")
        p95 = results["electrical_parameters_p95"]
        for param, error in results["electrical_parameters"].items():
            logger.info(
                f"Median relative error for {param}: {error:.2f}% on average, "
                f"{p95[param]:.2f}% at the 95th percentile"
            )
