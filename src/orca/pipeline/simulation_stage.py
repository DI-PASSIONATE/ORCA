from typing import Optional, Any, Dict, Callable
import os

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.simulation.simulate import run_palace

class PalaceSimulator(PipelineStage):
    """
    Pipeline stage for running Palace EM simulations.
    """
    def __init__(self):
        super().__init__(name="Palace EM Simulator", index=2)

    def run(self, context: Dict[str, Any], progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        geometry: BaseGeometry = context["geometry"]
        cpu_cores: int = context.get("cpu_cores", 16)
        base_dir: str = context.get("base_dir", os.getcwd())
        output_dir = os.path.join(base_dir, "results", geometry.name)

        return context

        
   