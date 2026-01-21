
from concurrent.futures import ProcessPoolExecutor, as_completed
from ihp import PDK
import multiprocessing
import pandas as pd
import os

from orca.logger import logger
from orca.geometry.base_geometry import BaseGeometry

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

    from orca.simulation.simulate import create_palace_model_from_gds
    try:
        config_name, sim_path, data_dir = create_palace_model_from_gds(
            geometry=geo_inst,
            gds_filename=gds_filename,
            simconfig_filename=simconfig_filename,
            show_mesh_results=False
        )
        return config_name, sim_path, data_dir, geo_inst.params
    except Exception as e:
        return e, None, None, None


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
        self.palace_models: list[tuple[str, str, str, dict]] = []
        self.working_geometries = pd.DataFrame(columns=["name"] + list(geometry.get_input_parameters().input_values.keys()))
        self.process_pool_executor = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.progress_callback = None
        PDK.activate()

    def run(self, cpu_cores: int = multiprocessing.cpu_count(), epochs=50, palace_executable: str = "apptainer exec ~/Documents/git/palace/palace.sif palace", progress_callback=None):
        """
        Runs the ORCA pipeline, including data generation, simulation, training, and evaluation.
        
        Args:
            cpu_cores: Number of CPU cores to use
            num_samples: Number of geometry samples to generate
            palace_executable: Command to execute Palace
            progress_callback: Optional callback function(step, current, total, message) for progress updates
        """
        self.progress_callback = progress_callback
        self.print_super_cool_logo_art()

        if cpu_cores < 1:
            raise ValueError("cpu_cores must be at least 1.")
        
        num_samples = self.geometry.n_samples

        logger.info(f"Running {num_samples} ORCA simulations of {self.geometry.name} with {cpu_cores} CPU cores...")         
        self._emit_progress("Initializing", 0, num_samples, "Starting ORCA pipeline...")

        cwd = os.getcwd()
        save_path = os.path.join(cwd, "results", self.geometry.name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.generate_gds_data(num_samples)
        self.convert_gds_to_palace()
        self.run_simulation(save_path, palace_executable, cpu_cores)
        model = self.train(self.geometry.get_dataset(), cwd=cwd, epochs=epochs)
        self.evaluate_model(self.geometry.get_dataset(), model = model)

        logger.info("ORCA pipeline finished successfully.")
        self._emit_progress("Complete", num_samples, num_samples, f"Successfully trained model of {self.geometry.name}.")

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
        self._emit_progress("GDS Generation", 0, num_samples, "Starting GDS generation...")
        
        for i, input_params in enumerate(self.geometry.input_iterator):
            if i >= num_samples:
                break
            try:
                geo_inst = self.geometry.create_geometry_instance(name=f"{self.geometry.name}_{i}", params=input_params)
                gds_filename = geo_inst.create_gds_file(params=input_params)
                self.geometry_instances.append((geo_inst, gds_filename))
            except Exception as e:
                logger.error(f"Error generating GDS for sample {i}: {e}")
                continue
            
            # Progress update every sample or every 10% for large batches
            if i % max(1, num_samples // 10) == 0 or i == num_samples - 1:
                self._emit_progress("GDS Generation", i + 1, num_samples, f"Generated {i + 1}/{num_samples} GDS files...")
            
        logger.info("#----------- Data generation completed. -----------#")
        self._emit_progress("GDS Generation", num_samples, num_samples, f"Generated {num_samples} GDS files successfully")

    def convert_gds_to_palace(self):
        """
        Converts GDS files to Palace models in parallel.
        """
        total_gds = len(self.geometry_instances)
        logger.info(f"Starting GDS to Palace conversion for {total_gds} files...")
        self._emit_progress("Palace Conversion", 0, total_gds, "Starting GDS to Palace conversion...")
        
        completed = 0
        failed = 0
        
        with self.process_pool_executor as executor:
            # Submit all conversion tasks
            futures = {
                executor.submit(
                    _convert_gds_worker,
                    geo_inst,
                    gds_filename,
                    geo_inst.simconfig_filename
                ): (geo_inst, gds_filename)
                for geo_inst, gds_filename in self.geometry_instances
            }
            
            # Process results as they complete
            for future in as_completed(futures):
                geo_inst, gds_filename = futures[future]
                try:
                    result = future.result()
                    config_name, sim_path, data_dir, params = result
                    
                    # Check if the result was an error
                    if isinstance(config_name, Exception):
                        logger.error(f"## ERROR ## Conversion of GDS to Palace model failed for {geo_inst.name} with error: {config_name}")
                        failed += 1
                        completed += 1
                        self._emit_progress("Palace Conversion", completed, total_gds, 
                                          f"Converting GDS to Palace ({completed}/{total_gds}, {failed} failed)...")
                        continue
                    
                    # Save parameters
                    row = {"name": geo_inst.name}
                    row.update(params if params is not None else {})
                    self.working_geometries = pd.concat(
                        [self.working_geometries, pd.DataFrame([row])],
                        ignore_index=True
                    )

                    # If conversion was successful, append to palace_models
                    self.palace_models.append((config_name, sim_path, data_dir, row))
                    completed += 1
                    
                    # Update progress
                    self._emit_progress("Palace Conversion", completed, total_gds, 
                                      f"Converting GDS to Palace ({completed}/{total_gds}, {failed} failed)...")
                    
                except Exception as e:
                    logger.warning(f"## ERROR ## Failed to process result for {geo_inst.name}: {e}")
                    os.remove(gds_filename)
                    failed += 1
                    completed += 1
                    self._emit_progress("Palace Conversion", completed, total_gds, 
                                      f"Converting GDS to Palace ({completed}/{total_gds}, {failed} failed)...")
                    continue
        
        # Save the working geometries to CSV after all conversions are complete
        save_path = os.path.join(os.getcwd(), "geometries", "params.csv")
        self.working_geometries.to_csv(save_path, index=False)
        
        successful = len(self.palace_models)
        logger.info(f"#----------- GDS to Palace conversion completed. {successful} successful, {failed} failed -----------#")
        self._emit_progress("Palace Conversion", total_gds, total_gds, 
                          f"Conversion complete: {successful} successful, {failed} failed")

    def run_simulation(self, save_path, palace_executable: str, cpu_cores: int):
        """
        Runs simulations on the generated data.
        """

        from orca.simulation.simulate import run_palace
        total_sims = len(self.palace_models)
        logger.info(f"Starting simulations with palace for {total_sims} models...")
        self._emit_progress("Palace Simulation", 0, total_sims, "Starting Palace simulations...")
        
        working_geoms = pd.DataFrame(self.working_geometries)
        
        completed = 0
        failed = 0

        
        for i, (config_name, sim_path, data_dir, row) in enumerate(self.palace_models):
            try:
                self._emit_progress("Palace Simulation", i, total_sims, 
                                  f"Running simulation {i + 1}/{total_sims} ({config_name})...")
                
                run_palace(
                    sim_path=sim_path,
                    data_dir=data_dir,
                    result_dir=save_path,
                    config_name=config_name,
                    palace_executable=palace_executable,
                    cpu_cores=cpu_cores
                )
                # Update the matching row by name
                if "name" in row:
                    mask = working_geoms["name"] == row["name"]
                    for key, value in row.items():
                        working_geoms.loc[mask, key] = value
                working_geoms.to_csv(os.path.join(save_path, "params.csv"), index=False)
                completed += 1
                
                self._emit_progress("Palace Simulation", i + 1, total_sims, 
                                  f"Completed simulation {i + 1}/{total_sims} ({failed} failed)")
            except Exception as e:
                logger.warning(f"## ERROR ## Simulation failed for {config_name} with error: {e}")
                failed += 1
                self._emit_progress("Palace Simulation", i + 1, total_sims, 
                                  f"Simulation failed for {config_name} ({failed} total failures)")
                continue
        
        logger.info(f"#----------- Simulations completed. {completed} successful, {failed} failed -----------#")
        self._emit_progress("Palace Simulation", total_sims, total_sims, 
                          f"Simulations complete: {completed} successful, {failed} failed")

    def train(self, dataset, cwd: str = os.getcwd(), epochs: int = 30):
        """
        Trains the ORCA model using the simulation data.
        """
        from orca.training.train import train_model
        import torch
        import torch.nn as nn
        from orca.training.onnx_wrapper import ONNXWrapper
        logger.info(f"Starting model training with {len(dataset)} samples...")
        self._emit_progress("Model Training", 0, 1, f"{len(dataset)} samples loaded for training.")

        model = self.geometry.create_model()

        trained_model = train_model(dataset, model=model, epochs=epochs, batch_size=32, learning_rate=1e-3, progress_callback=self._emit_progress)
        
        model_save_dir = os.path.join(cwd, "models", self.geometry.name)
        model_save_path = os.path.join(model_save_dir, f"{self.geometry.name}.onnx")

        if not os.path.exists(model_save_dir): # Create model directory if it doesn't exist
            os.makedirs(model_save_dir)
        elif os.path.exists(model_save_path): # Check if model already exists under this path -> rename old model
            import time
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            archived_model_path = model_save_path.replace(".onnx", f"_{timestamp}.onnx")
            os.rename(model_save_path, archived_model_path)
            logger.info(f"Existing model found. Archived previous model to {archived_model_path}")

        # Export to ONNX with multiple inputs/outputs using ONNXWrapper
        torch.onnx.export(
            ONNXWrapper(trained_model, output_denormalizer=dataset.output_normalizer),
            tuple(torch.randn(1, 1, device=dataset.device) for _ in dataset.input_param_names),
            input_names=dataset.input_param_names,
            output_names=dataset.output_param_names,
            f=model_save_path,
            external_data=False,
            dynamo=True
        )

        self._emit_progress("Model Training", 1, 1, "Model training completed successfully.")
        logger.info("#----------- Model training completed. -----------#")
        return trained_model

    def evaluate_model(self, dataset, model):
        """
        Evaluates the trained ORCA model.
        """
        from orca.training.train import test_model
        logger.warning("Starting model evaluation...")
        self._emit_progress("Model Evaluation", 0, 1, "Model evaluation not yet implemented")
        test_loss = test_model(dataset, model=model, batch_size=32)
        logger.info(f"Model evaluation completed. Test Loss: {test_loss:.4f}")
        self._emit_progress("Model Evaluation", 1, 1, f"Model evaluation completed. Test Loss: {test_loss:.4f}")
        logger.info("#----------- Model evaluation completed. -----------#")
    
    def _emit_progress(self, step: str, current: int, total: int, message: str):
        """
        Emit progress update through callback if available.
        
        Args:
            step: Name of the current step (e.g., "GDS Generation")
            current: Current progress count
            total: Total items to process
            message: Detailed message about current progress
        """
        if self.progress_callback:
            try:
                self.progress_callback(step, current, total, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")