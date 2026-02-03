from typing import Optional, Any, Dict, Callable
import os
import pandas as pd
import tqdm

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.simulation.simulate import run_palace

class ModelTrainer(PipelineStage):
    """
    Pipeline stage for training AI/ML models based on simulation results.
    """
    def __init__(self):
        super().__init__(name="Model Trainer", index=4)

    def run(self, context: Dict[str, Any], progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        geometry: BaseGeometry = context["geometry"]
        base_dir: str = context.get("base_dir", os.getcwd())
        output_dir = os.path.join(base_dir, "models") # Model gets stored here
        result_csv = context.get("result_csv", os.path.join(base_dir, "results", f"{geometry.name}.csv"))

        if not os.path.exists(result_csv):
            logger.error(f"No result CSV file found for training at {result_csv}.")
            return context
        
        result_df = pd.read_csv(result_csv) # Information

        return context