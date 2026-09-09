from typing import Optional, Any, Callable, TYPE_CHECKING
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.basis_expansion import BasisExpansion, get_basis_class
from orca.training.models.base_model import OrcaModel, get_model_class
from orca.training.trainer import Trainer, TrainingConfig
from orca.training.tuner import HyperparameterTuner

if TYPE_CHECKING:
    from orca.pipeline.context import PipelineContext


class ModelTrainer(PipelineStage):
    """
    Pipeline stage for training AI/ML models based on simulation results.
    """

    def __init__(
        self,
        model: str | type[OrcaModel] = "mlp",
        basis: str | type[BasisExpansion] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        test_frac: float = 0.15,
        n_train_samples: Optional[int] = None,
        n_fold_cv: int = 5,
        n_trials: int = 200,
    ):
        """
        Initializes the ModelTrainer stage with the architecture to train and optional
        hyperparameters. By default, optuna tuning is used to search for optimal
        hyperparameters, but specific hyperparameters can be provided to override the search space.

        The model is chosen here rather than on the geometry, so swapping architectures
        never requires touching a geometry class.

        Args:
            model: An OrcaModel subclass or a registered model name (default: "mlp").
            basis: Optional basis expansion of the model inputs, as a
                BasisExpansion subclass or a registered name (e.g. "chebyshev").
            hyperparameters: Optional predefined hyperparameters. If None, hyperparameter tuning will be performed.
            test_frac: Fraction of data to use for testing (default: 0.15).
            n_train_samples: Optional limit on the number of samples to use for training.
            n_fold_cv: Number of folds for cross-validation during hyperparameter tuning (default: 5).
            n_trials: Number of optuna trials during hyperparameter tuning (default: 200).
        """
        super().__init__(name="Model Trainer", index=4)
        self.model_cls = get_model_class(model)
        self.basis_cls = get_basis_class(basis) if basis is not None else None
        self.hyperparameters = hyperparameters
        self.test_frac = test_frac
        self.n_samples = n_train_samples
        self.n_fold_cv = n_fold_cv
        self.n_trials = n_trials

    def run(
        self,
        context: "PipelineContext",
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> "PipelineContext":
        geometry: BaseGeometry = context.geometry
        result_dir = context.result_dir
        result_csv = context.result_csv

        if not os.path.exists(result_csv):
            logger.error(f"No result CSV file found for training at {result_csv}.")
            return context

        result_df = pd.read_csv(result_csv)

        # Split into train_val and test using sklearn - train_val is used for n-fold-cross-validation during hyperparameter tuning
        train_val_df, test_df = train_test_split(
            result_df,
            test_size=self.test_frac,
            random_state=11
        )

        if self.n_samples is not None and self.n_samples < len(train_val_df):
            train_val_df = train_val_df.head(self.n_samples)
            logger.info(f"Using only the first {self.n_samples} samples for training as specified in the ModelTrainer initialization.")

        # Perform hyperparameter tuning if needed. The result stays local rather than
        # being stored on the stage, so re-running the same ModelTrainer tunes again
        # instead of silently reusing the previous run's best parameters.
        if isinstance(self.hyperparameters, dict):
            hyperparameters = self.hyperparameters
            logger.info(f"Using provided hyperparameters for training: {hyperparameters}")
        else:
            logger.info("No hyperparameters provided, starting hyperparameter tuning with optuna...")
            train_val_dataset = geometry.dataset.new_split(
                directory=result_dir, data_df=train_val_df, fit_normalizers=True
            )
            self._check_compatibility(train_val_dataset)
            tuner = HyperparameterTuner(
                model_cls=self.model_cls,
                dataset=train_val_dataset,
                basis_cls=self.basis_cls,
                n_fold_cv=self.n_fold_cv,
                n_trials=self.n_trials,
            )
            hyperparameters = tuner.tune()
            logger.info(f"Hyperparameter tuning completed. Best hyperparameters: {hyperparameters}")

        # Split train_val_df into train and val for actual model training
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.test_frac,
            random_state=11
        )

        # The training split owns the normalization statistics; validation reuses them,
        # so no validation data leaks into the normalization of the training data.
        train_dataset = geometry.dataset.new_split(
            directory=result_dir, data_df=train_df, fit_normalizers=True
        )
        val_dataset = geometry.dataset.new_split(directory=result_dir, data_df=val_df)
        self._check_compatibility(train_dataset)

        logger.info(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples for model training. Beginning training..."
        )

        trainer = Trainer(
            config=TrainingConfig.from_hyperparameters(hyperparameters),
            progress_callback=progress_callback,
            stage_name=self.name,
        )
        spec = train_dataset.io_spec
        basis = self.basis_cls.from_spec(spec, hyperparameters) if self.basis_cls else None

        result = trainer.fit(
            model=self.model_cls.from_spec(spec, hyperparameters, basis),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        context.trained_model = result.model
        context.dataset = train_dataset
        context.hyperparameters = hyperparameters
        context.final_val_loss = result.best_loss
        context.training_history = result.history
        context.test_df = test_df
        return context

    def _check_compatibility(self, dataset: BaseDataset) -> None:
        """
        Verifies that the model and the dataset agree on how frequency is laid out,
        so a mismatch fails here rather than as a shape error mid-training.
        """
        dataset_mode = type(dataset).frequency_mode
        if not self.model_cls.accepts_frequency_mode(dataset_mode):
            raise ValueError(
                f"{self.model_cls.__name__} consumes {self.model_cls.frequency_mode.name} samples, "
                f"but {type(dataset).__name__} produces {dataset_mode.name}. "
                "Pick a dataset and a model that agree on the frequency layout."
            )
