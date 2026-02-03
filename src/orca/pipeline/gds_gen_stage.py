from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from typing import Any, Dict, Callable, Optional, Type
from orca.geometry.base_geometry import BaseGeometry
from orca.pipeline.pipeline_stage import PipelineStage
from orca.logger import logger

class GDSGenerator(PipelineStage):
    """
    Pipeline stage for generating GDS files from trained models.
    """
    def __init__(self, num_samples: int = 1000):
        super().__init__(name="GDS Generator", index=0)
        self.num_samples = num_samples

    def run(self, context: Dict[str, Any], progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        geometry: BaseGeometry = context["geometry"]
        cpu_cores: int = context.get("cpu_cores", 16)
        logger.info(f"Starting GDS generation for {self.num_samples} samples using {cpu_cores} CPU cores.")
        
        try:
            from ihp import PDK
            PDK.activate()
        except ImportError:
            logger.error("IHP PDK not found. Please install the IHP PDK to use GDS conversion.")
            return context

        futures = []
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            # Create cpu_cores processes to generate GDS files in parallel
            for i, input_params in enumerate(geometry.input_iterator):
                if i >= self.num_samples:
                    break
                
                logger.info(f"Submitting GDS generation task for sample {i}")
                # Pass geometry class and name separately to avoid pickling the whole instance (with locks)
                future = executor.submit(
                    geometry.create_gds_file,
                    f"{geometry.name}_{i}.gds",
                    input_params
                )
                futures.append(future)

            # Update progress as tasks complete
            for i, future in enumerate(as_completed(futures)):
                try:
                    gds_path = future.result()
                    logger.info(f"GDS file generated: {gds_path}")
                except Exception as e:
                    logger.error(f"Worker task failed: {e}")
                finally:
                    if progress_callback:
                        progress = (i + 1) / len(futures)
                        progress_callback(progress, f"GDS Generation Progress: {i + 1}/{len(futures)}")

        logger.info("GDS generation completed.")
        return context