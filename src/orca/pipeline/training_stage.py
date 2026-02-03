from typing import Optional, Any, Dict, Callable
import os
import pandas as pd
from orca.training.train import train_model

from sklearn.model_selection import train_test_split

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger


class ModelTrainer(PipelineStage):
    """
    Pipeline stage for training AI/ML models based on simulation results.
    """

    def __init__(self):
        super().__init__(name="Model Trainer", index=4)

    def run(
        self,
        context: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        geometry: BaseGeometry = context["geometry"]
        base_dir: str = context.get("base_dir", os.getcwd())
        result_dir = context.get("result_dir", os.path.join(base_dir, "results"))
        result_csv = context.get(
            "result_csv", os.path.join(result_dir, f"{geometry.name}.csv")
        )

        if not os.path.exists(result_csv):
            logger.error(f"No result CSV file found for training at {result_csv}.")
            return context

        result_df = pd.read_csv(result_csv)  # Information

        train_df, val_df = train_test_split(result_df, test_size=0.2, random_state=11)

        train_dataset = geometry.dataset.new_split(
            directory=result_dir, data_df=train_df
        )
        val_dataset = geometry.dataset.new_split(directory=result_dir, data_df=val_df)

        logger.info(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples for model training. Beginning training..."
        )

        trained_model = train_model(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=geometry.model,
            epochs=15,
            batch_size=32,
            learning_rate=1e-3,
            progress_callback=None,
        )

        context["trained_model"] = trained_model
        context["dataset"] = train_dataset
        return context
