from typing import Optional, Any, Dict, Callable
import os
import pandas as pd
import optuna
from orca.training.datasets.dataloader import train_val_test_dataset
from orca.training.train import hyperparameter_tuning, train_model

from sklearn.model_selection import train_test_split

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.utils.folder_structure import OrcaFolderStructure


class ModelTrainer(PipelineStage):
    """
    Pipeline stage for training AI/ML models based on simulation results.
    """

    def __init__(self, hyperparameters: dict[str, Any] | None = None):
        """
        Initializes the ModelTrainer stage with optional hyperparameters for model training.
        By default, optuna tuning is used to search for optimal hyperparameters, 
        but specific hyperparameters can be provided to override the search space.
        """
        super().__init__(name="Model Trainer", index=4)
        self.hyperparameters = hyperparameters

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

        result_df = pd.read_csv(result_csv)  # Information

        train_dataset, val_dataset, test_dataset = train_val_test_dataset(result_df, geometry, result_dir)

        logger.info(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples for model training. Beginning training..."
        )

        if self.hyperparameters is None or type(self.hyperparameters) != dict:
            logger.info("No hyperparameters provided, starting hyperparameter tuning with optuna...")
            self.hyperparameters = hyperparameter_tuning(train_dataset, val_dataset, geometry)
            logger.info(f"Hyperparameter tuning completed. Best hyperparameters: {self.hyperparameters}")
        else:
            logger.info(f"Using provided hyperparameters for training: {self.hyperparameters}")

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
        context["train_dataset"] = train_dataset
        context["val_dataset"] = val_dataset
        context["test_dataset"] = test_dataset
        return context
    
