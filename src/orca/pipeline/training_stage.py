from typing import Optional, Any, Dict, Callable
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from orca.training.train import hyperparameter_tuning, train_model

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.utils.folder_structure import OrcaFolderStructure


class ModelTrainer(PipelineStage):
    """
    Pipeline stage for training AI/ML models based on simulation results.
    """

    def __init__(self, hyperparameters: dict[str, Any] | None = None, test_frac: float = 0.15, n_train_samples: Optional[int] = None, n_fold_cv: int = 5):
        """
        Initializes the ModelTrainer stage with optional hyperparameters for model training.
        By default, optuna tuning is used to search for optimal hyperparameters, 
        but specific hyperparameters can be provided to override the search space.
        
        Args:
            hyperparameters: Optional predefined hyperparameters. If None, hyperparameter tuning will be performed.
            val_frac: Fraction of data to use for validation (default: 0.15).
            test_frac: Fraction of data to use for testing (default: 0.15).
            n_samples: Optional limit on the number of samples to use for training.
            n_fold_cv: Number of folds for cross-validation during hyperparameter tuning (default: 5).
        """
        super().__init__(name="Model Trainer", index=4)
        self.hyperparameters = hyperparameters
        self.test_frac = test_frac
        self.n_samples = n_train_samples
        self.n_fold_cv = n_fold_cv

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
        if self.hyperparameters is None or type(self.hyperparameters) != dict:
            logger.info("No hyperparameters provided, starting hyperparameter tuning with optuna...")
            self.hyperparameters = hyperparameter_tuning(
                train_val_df, result_dir, geometry, n_fold_cv=self.n_fold_cv
            )
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

        logger.info(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples for model training. Beginning training..."
        )

        trained_model, best_loss = train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=geometry.get_model(self.hyperparameters),
            epochs=self.hyperparameters["epochs"],
            batch_size=self.hyperparameters["batch_size"],
            learning_rate=self.hyperparameters["learning_rate"],
            progress_callback=progress_callback,
            stage_name=self.name,
        )

        context["trained_model"] = trained_model
        context["dataset"] = train_dataset
        context["hyperparameters"] = self.hyperparameters
        context["final_val_loss"] = best_loss
        context["test_df"] = test_df
        return context
    
