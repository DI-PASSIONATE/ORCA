from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from typing import Any, Dict, Callable, Optional
from orca.geometry.base_geometry import BaseGeometry
from orca.pipeline.pipeline_stage import PipelineStage
from orca.logger import logger

class GDSConverter(PipelineStage):
    """
    Pipeline stage for converting GDS files to Palace-compatible format.
    """
    def __init__(self, num_samples: int = 1000):
        super().__init__(name="GDS Converter", index=1)
        self.num_samples = num_samples

    def run(self, context: Dict[str, Any], progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        geometry: BaseGeometry = context["geometry"]
        cpu_cores: int = context.get("cpu_cores", 16)
        gds_path = context.get("gds_path", "")

        if not gds_path:
            logger.error("No GDS path provided in context for conversion.")
            return context
        
        try:
            from ihp import PDK
            PDK.activate()
        except ImportError:
            logger.error("IHP PDK not found. Please install the IHP PDK to use GDS conversion.")
            return context