from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import os
import tqdm

from typing import Any, Dict, Callable, Optional
from orca.geometry.base_geometry import BaseGeometry
from orca.pipeline.pipeline_stage import PipelineStage
from orca.logger import logger
from orca.utils.gds_converter import create_palace_model_from_gds

class GDSConverter(PipelineStage):
    """
    Pipeline stage for converting GDS files to Palace-compatible format.
    """
    def __init__(self):
        super().__init__(name="GDS Converter", index=1)

    def run(self, context: Dict[str, Any], progress_callback: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
        geometry: BaseGeometry = context["geometry"]
        cpu_cores: int = context.get("cpu_cores", 16)
        base_dir: str = context.get("base_dir", os.getcwd())
        gds_csv = context.get("gds_csv", base_dir + f"/geometries/{geometry.name}.csv")
        output_dir = os.path.join(base_dir, "palace_sims") # Palace simulations get stored here
        palace_csv = os.path.join(output_dir, f"{geometry.name}.csv")

        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)

        os.makedirs(output_dir)
        
        gds_data = pd.read_csv(gds_csv)
        
        logger.info(f"Starting GDS conversion for {len(gds_data)} files using {cpu_cores} CPU cores.")
        
        with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
            futures = []
            gds_dir = os.path.dirname(gds_csv)
            for i, row in gds_data.iterrows():
                # CSV layout:
                # name,input_winding_diameter,output_winding_diameter,center_displacement,bottom_linewidth,upper_linewidth
                # everything after name is input parameters
                name = row["name"] # GDS path
                gds_path = os.path.join(gds_dir, name)
                params = row.to_dict()
                del params["name"]

                # Submit GDS conversion tasks
                future = executor.submit(
                    create_palace_model_from_gds,
                    geometry_name=name,
                    params=params,
                    output_dir=base_dir,
                    gds_filename=gds_path,
                    stackup_xml=geometry.stackup_xml,
                    simconfig_filename=geometry.simconfig_filename,
                    show_mesh_results=False
                )
                futures.append(future)

            # Collect finished results
            # Print progress bar using tqdm
            for i, future in enumerate(tqdm.tqdm(as_completed(futures), total=len(futures), desc="GDS Conversion")):
                try:
                    geo_name, params, config_name, sim_path, data_dir = future.result()
                    # Save input parameters to CSV
                    self._save_csv(palace_csv, geo_name, params, data_dir, sim_path, config_name)
                except Exception as e:
                    logger.error(f"GDS conversion failed for file index {i} with error: {e}")
                finally: # and call progress_callback even on failure
                    if progress_callback:
                        percentage = (i + 1) / len(futures) * 100
                        progress_callback(percentage, f"Converted {i + 1} of {len(futures)} GDS files.")

            logger.info("GDS conversion completed.")

        context["palace_csv"] = palace_csv
        return context
    

    def _save_csv(self, csv_path, name: str, params: dict[str, Any], data_dir: str, sim_path: str, config_name: str):
        """
        Saves the input parameters to a CSV file.

        Args:
            csv_path (str): Path to the CSV file.
            name (str): Name of the geometry instance.
            params (dict[str, Any]): Input parameters to save.
            data_dir (str): Directory where data is stored.
            sim_path (str): Path to the simulation file.
        """
        df = pd.DataFrame([{"name": name, "data_dir": data_dir, "sim_path": sim_path, "config_name": config_name} | params])
        if not os.path.exists(csv_path):
            df.to_csv(csv_path, header=True, index=False)
        else:
            df.to_csv(csv_path, mode='a', header=False, index=False)