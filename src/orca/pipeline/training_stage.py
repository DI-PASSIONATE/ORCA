from typing import Optional, Any, Dict, Callable
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.training.datasets.base_dataset import BaseDataset
from orca.training.models.base_model import OrcaModel, get_model_class
from orca.training.trainer import Trainer, TrainingConfig
from orca.training.tuner import HyperparameterTuner
from orca.utils.folder_structure import OrcaFolderStructure


class ModelTrainer(PipelineStage):
    """
    Pipeline stage for training AI/ML models based on simulation results.
    """

    def __init__(
        self,
        model: str | type[OrcaModel] = "mlp",
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
            hyperparameters: Optional predefined hyperparameters. If None, hyperparameter tuning will be performed.
            test_frac: Fraction of data to use for testing (default: 0.15).
            n_train_samples: Optional limit on the number of samples to use for training.
            n_fold_cv: Number of folds for cross-validation during hyperparameter tuning (default: 5).
            n_trials: Number of optuna trials during hyperparameter tuning (default: 200).
        """
        super().__init__(name="Model Trainer", index=4)
        self.model_cls = get_model_class(model)
        self.hyperparameters = hyperparameters
        self.test_frac = test_frac
        self.n_samples = n_train_samples
        self.n_fold_cv = n_fold_cv
        self.n_trials = n_trials

    def run(
        self,
        context: Dict[str, Any],
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        geometry: BaseGeometry = context["geometry"]
        result_dir = OrcaFolderStructure.get_result_dir(context)
        result_csv = OrcaFolderStructure.get_result_csv(context)

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

        # Perform hyperparameter tuning if needed
        if self.hyperparameters is None or type(self.hyperparameters) is not dict:
            logger.info("No hyperparameters provided, starting hyperparameter tuning with optuna...")
            train_val_dataset = geometry.dataset.new_split(directory=result_dir, data_df=train_val_df)
            self._check_compatibility(train_val_dataset)
            tuner = HyperparameterTuner(
                model_cls=self.model_cls,
                dataset=train_val_dataset,
                n_fold_cv=self.n_fold_cv,
                n_trials=self.n_trials,
            )
            self.hyperparameters = tuner.tune()
            logger.info(f"Hyperparameter tuning completed. Best hyperparameters: {self.hyperparameters}")
        else:
            logger.info(f"Using provided hyperparameters for training: {self.hyperparameters}")

        # Split train_val_df into train and val for actual model training
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=self.test_frac,
            random_state=11
        )

        train_dataset = geometry.dataset.new_split(directory=result_dir, data_df=train_df)
        val_dataset = geometry.dataset.new_split(directory=result_dir, data_df=val_df)
        self._check_compatibility(train_dataset)

        logger.info(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples for model training. Beginning training..."
        )

        trainer = Trainer(
            config=TrainingConfig.from_hyperparameters(self.hyperparameters),
            progress_callback=progress_callback,
            stage_name=self.name,
        )
        result = trainer.fit(
            model=self.model_cls.from_spec(train_dataset.io_spec, self.hyperparameters),
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        context["trained_model"] = result.model
        context["dataset"] = train_dataset
        context["hyperparameters"] = self.hyperparameters
        context["final_val_loss"] = result.best_loss
        context["training_history"] = result.history
        context["test_df"] = test_df
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
