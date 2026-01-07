
from concurrent.futures import ProcessPoolExecutor, as_completed
from ihp import PDK
import multiprocessing
import pandas as pd
import os

from orca.logger import logger
from orca.geometry.base_geometry import BaseGeometry
from orca.simulation.simulate import create_palace_model_from_gds, run_palace

def _convert_gds_worker(geo_inst, gds_filename, simconfig_filename):
    """
    Worker function for parallel GDS to Palace conversion.
    
    Args:
        geo_inst: Geometry instance
        gds_filename: Path to GDS file
        simconfig_filename: Path to simulation config file
        
    Returns:
        Tuple of (geo_inst, gds_filename, output, params) or (geo_inst, gds_filename, error, None) on failure
    """
    try:
        output = create_palace_model_from_gds(
            geometry=geo_inst,
            gds_filename=gds_filename,
            simconfig_filename=simconfig_filename
        )
        return (output, geo_inst.params)
    except Exception as e:
        return (e, None)


class ORCA:
    """
    This is the main ORCA class. It serves as the entry point for the ORCA framework and runs the entire pipeline, from 
    data generation, simulation, training, to evaluation.

    Attributes:
        geometry (BaseGeometry): An instance of a geometry class that defines the geometry to be used
    """

    def __init__(self, geometry: BaseGeometry):
        self.geometry: BaseGeometry = geometry
        self.geometry_instances: list[tuple[BaseGeometry, str]] = []
        self.palace_models: list[tuple[str, str, str]] = []
        self.working_geometries = pd.DataFrame(columns=["name"] + list(geometry.get_input_parameters().input_values.keys()))
        self.process_pool_executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        PDK.activate()

    def run(self, cpu_cores: int = multiprocessing.cpu_count(), num_samples: int = 1000, palace_executable: str = "apptainer exec ~/Documents/git/palace/palace.sif palace"):
        """
        Runs the ORCA pipeline, including data generation, simulation, training, and evaluation.
        """
        self.print_super_cool_logo_art()

        if cpu_cores < 1:
            raise ValueError("cpu_cores must be at least 1.")

        logger.info(f"Running {num_samples} ORCA simulations of {self.geometry.name} with {cpu_cores} CPU cores...")         

        self.generate_gds_data(num_samples)
        self.convert_gds_to_palace()
        self.run_simulation(palace_executable, cpu_cores)
        self.train_model()
        self.evaluate_model()

        logger.info("ORCA pipeline finished successfully.")

    def print_super_cool_logo_art(self):
        logger.info("###########################################################")  
        logger.info(" ██████╗ ██████╗  ██████╗ █████╗ ")
        logger.info("██╔═══██╗██╔══██╗██╔════╝██╔══██╗")
        logger.info("██║   ██║██████╔╝██║     ███████║")
        logger.info("██║   ██║██╔══██╗██║     ██╔══██║")
        logger.info("╚██████╔╝██║  ██║╚██████╗██║  ██║")
        logger.info(" ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝")       
        logger.info("###########################################################")
        logger.info("Welcome to ORCA - Open RF Integrated Circuit Automation")
        logger.info("")

    def generate_gds_data(self, num_samples: int):
        """
        Generates data based on the defined geometry.
        """
        logger.info("Starting data generation using gdsfactory...")
        for i, input_params in enumerate(self.geometry.input_iterator):
            if i >= num_samples:
                break
            geo_inst = self.geometry.create_geometry_instance(name=f"{self.geometry.name}_{i}", params=input_params)
            gds_filename = geo_inst.create_gds_file(params=input_params)
            self.geometry_instances.append((geo_inst, gds_filename))                 
            
        logger.info("#----------- Data generation completed. -----------#")

    def convert_gds_to_palace(self):
        """
        Converts GDS files to Palace models in parallel.
        """
        logger.info("Starting GDS to Palace conversion...")
        
        with self.process_pool_executor as executor:
            # Submit all conversion tasks
            futures = {
                executor.submit(
                    _convert_gds_worker,
                    geo_inst,
                    gds_filename,
                    geo_inst.simconfig_filename
                ): geo_inst
                for geo_inst, gds_filename in self.geometry_instances
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                geo_inst = futures[future]
                try:
                    result = future.result()
                    output, params = result
                    
                    # Check if the result was an error
                    if isinstance(output, Exception):
                        logger.error(f"## ERROR ## Conversion of GDS to Palace model failed for {geo_inst.name} with error: {output}")
                        continue
                    
                    # If conversion was successful, append to palace_models
                    self.palace_models.append(output)
                    
                    # Append to working_geometries DataFrame
                    row = {"name": geo_inst.name}
                    row.update(params if params is not None else {})
                    self.working_geometries = pd.concat(
                        [self.working_geometries, pd.DataFrame([row])],
                        ignore_index=True
                    )
                    
                except Exception as e:
                    logger.error(f"## ERROR ## Failed to process result for {geo_inst.name}: {e}")
                    continue
        
        # Save the working geometries to CSV after all conversions are complete
        save_path = os.path.join(os.getcwd(), "geometries", "parameters.csv")
        self.working_geometries.to_csv(save_path, index=False)
        logger.info("#----------- GDS to Palace conversion completed. -----------#")

    def run_simulation(self, palace_executable: str, cpu_cores: int):
        """
        Runs simulations on the generated data.
        """
        logger.info("Starting simulations with palace...")
        for config_name, sim_path, data_dir in self.palace_models:
            run_palace(
                sim_path=sim_path,
                data_dir=data_dir,
                result_dir=self.geometry.name,
                config_name=config_name,
                palace_executable=palace_executable,
                cpu_cores=cpu_cores
            )
        logger.info("#----------- Simulations completed. -----------#")

    def train_model(self):
        """
        Trains the ORCA model using the simulation data.
        """
        logger.warning("Starting model training... (NOT IMPLEMENTED YET)")
        logger.info("#----------- Model training completed. -----------#")

    def evaluate_model(self):
        """
        Evaluates the trained ORCA model.
        """
        logger.warning("Starting model evaluation... (NOT IMPLEMENTED YET)")
        logger.info("#----------- Model evaluation completed. -----------#")