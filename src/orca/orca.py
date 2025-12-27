from .geometry.base_geometry import BaseGeometry
from .simulation.simulate import create_palace_model_from_gds

import multiprocessing

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
        self.palace_models: list[tuple[str, str]] = []

    def run(self, cpu_cores: int = multiprocessing.cpu_count(), num_samples: int = 1000):
        """
        Runs the ORCA pipeline, including data generation, simulation, training, and evaluation.
        """
        self.print_super_cool_logo_art()

        print(f"Running {num_samples} ORCA simulations of {self.geometry.name} with {cpu_cores} CPU cores...")                
        self.generate_gds_data(num_samples)
        self.convert_gds_to_palace()
        self.run_simulation()
        self.train_model()
        self.evaluate_model()

        print("ORCA pipeline finished successfully.")

    def print_super_cool_logo_art(self):
        print("###########################################################")
        print("░▒▓███████▓▒░ ░▒▓██████▓▒░░▒▓███████▓▒░░▒▓█▓▒░▒▓███████▓▒░")  
        print("░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░")
        print("░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░")
        print("░▒▓███████▓▒░░▒▓████████▓▒░▒▓███████▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░")
        print("░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░")
        print("░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░")
        print("░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░▒▓███████▓▒░") 
        print("###########################################################")
        print("Welcome to ORCA - Open RF Integrated Circuit Automation")
        print("")

    def generate_gds_data(self, num_samples: int):
        """
        Generates data based on the defined geometry.
        """
        print("Starting data generation using gdsfactory...")
        for i in range(num_samples):
            # TODO: Generate input parameters for the geometry
            input_params = self.geometry.get_next_input_parameters()
            geo_inst = self.geometry.create_geometry_instance(name=f"{self.geometry.name}_{i}", input_parameters=input_params)
            gds_filename = geo_inst.create_gds_file()
            self.geometry_instances.append((geo_inst, gds_filename))                 
            
        print("#----------- Data generation completed. -----------#")

    def convert_gds_to_palace(self):
        """
        Converts GDS files to Palace models.
        """
        print("Starting GDS to Palace conversion...")
        for geo_inst, gds_filename in self.geometry_instances:
            config_name, palace_folder = create_palace_model_from_gds(
                geometry=geo_inst,
                gds_filename=gds_filename,
                simconfig_filename=geo_inst.simconfig_filename
            )
            self.palace_models.append((config_name, palace_folder))
        print("#----------- GDS to Palace conversion completed. -----------#")

    def run_simulation(self):
        """
        Runs simulations on the generated data.
        """
        print("Starting simulations with palace...")
        
        print("#----------- Simulations completed. -----------#")

    def train_model(self):
        """
        Trains the ORCA model using the simulation data.
        """
        print("Starting model training...")
        print("#----------- Model training completed. -----------#")

    def evaluate_model(self):
        """
        Evaluates the trained ORCA model.
        """
        print("Starting model evaluation...")
        print("#----------- Model evaluation completed. -----------#")