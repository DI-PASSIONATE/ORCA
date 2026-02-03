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
        gds_csv = context.get("gds_csv", "")

        if not gds_csv:
            logger.error("No GDS path provided in context for conversion.")
            return context
        else:
            gds_data = pd.read_csv(gds_csv)
            logger.info(f"Starting GDS conversion for {len(gds_data)} files using {cpu_cores} CPU cores.")
        
        try:
            from ihp import PDK
            PDK.activate()
        except ImportError:
            logger.error("IHP PDK not found. Please install the IHP PDK to use GDS conversion.")
            return context
        
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
                    output_dir=context.get("base_dir", ""),
                    gds_filename=gds_path,
                    stackup_xml=geometry.stackup_xml,
                    simconfig_filename=geometry.simconfig_filename,
                    show_mesh_results=False
                )
                futures.append(future)

            # Collect finished results
            # Print progress bar using tqdm
            for i, future in enumerate(tqdm.tqdm(as_completed(futures), total=len(futures), desc="GDS Conversion Progress")):
                try:
                    config_name, sim_path, data_dir = future.result()
                except Exception as e:
                    logger.error(f"GDS conversion failed for file index {i} with error: {e}")
                finally: # and call progress_callback even on failure
                    if progress_callback:
                        percentage = (i + 1) / len(futures) * 100
                        progress_callback(percentage, f"Converted {i + 1} of {len(futures)} GDS files.")

            logger.info("GDS conversion completed.")
        return context