from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from typing import Any, Dict, Callable, Optional, Type
from orca.geometry.base_geometry import BaseGeometry
from orca.pipeline.pipeline_stage import PipelineStage
from orca.logger import logger
import tqdm
import os

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
        base_dir: str = context.get("base_dir", os.getcwd())
        output_dir = os.path.join(base_dir, "geometries", geometry.name)
        gds_csv = os.path.join(output_dir, f"{geometry.name}.csv")
        logger.info(f"Starting GDS generation for {self.num_samples} samples using {cpu_cores} CPU cores.")
        
        try:
            from ihp import PDK
            PDK.activate()
        except ImportError:
            logger.error("IHP PDK not found. Please install the IHP PDK to use GDS conversion.")
            return context

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Remove existing CSV file if it exists
        if os.path.exists(gds_csv):
            os.remove(gds_csv)

        # Tell the input parameter iterator the number of samples to generate
        geometry.input_parameter_iterator.set_sample_count(self.num_samples)

        futures = []
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            # Create cpu_cores processes to generate GDS files in parallel
            for i, input_params in enumerate(geometry.input_iterator):
                if i >= self.num_samples:
                    break
                
                # Pass geometry class and name separately to avoid pickling the whole instance (with locks)
                future = executor.submit(
                    GDSGenerator._generate_gds_file,
                    geometry.create_gds_file,
                    f"{geometry.name}_{i}.gds",
                    output_dir,
                    input_params
                )
                futures.append(future)


            # Print progress bar using tqdm and call progress_callback
            for i, future in enumerate(tqdm.tqdm(as_completed(futures), total=len(futures), desc="GDS Generation Progress")):
                try:
                    gds_path, name, params = future.result()

                    # Save instance name + input parameters to CSV
                    self._save_csv(gds_csv, name, params)
                except Exception as e:
                    logger.debug(f"Worker task failed: {e}")
                finally:
                    if progress_callback:
                        progress = (i + 1) / len(futures)
                        progress_callback(progress, f"GDS Generation Progress: {i + 1}/{len(futures)}")

        logger.info("GDS generation completed.")
        context["gds_csv"] = gds_csv
        return context
    
    @staticmethod
    def _generate_gds_file(gds_method: Callable, name: str, output_dir: str, params: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        """
        Static method to generate a GDS file given a geometry class, name, and parameters.
        This is used for multiprocessing to avoid pickling issues with instance methods.

        Args:
            geometry_class (Type[BaseGeometry]): The geometry class to instantiate.
            name (str): The name of the GDS file to create.
            output_dir (str): The directory to save the GDS file.
            params (dict[str, Any]): The input parameters for the geometry.

        Returns:
            str: Path to the created GDS file.
        """
        output_path = os.path.join(output_dir, name)
        path = gds_method(name, output_path, params)
        return path, name, params
    
    def _save_csv(self, csv_path, name: str, params: dict[str, Any]):
        """
        Saves the input parameters to a CSV file.

        Args:
            csv_path (str): Path to the CSV file.
            name (str): Name of the geometry instance.
            params (dict[str, Any]): Input parameters to save.
        """
        df = pd.DataFrame([{"name": name} | params])
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, header=True, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)