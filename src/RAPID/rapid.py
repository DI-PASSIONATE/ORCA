from .geometry.base_geometry import BaseGeometry
from .simulation.simulate import create_palace_model_from_gds

import multiprocessing

class RAPID:
    """
    This is the main RAPID class. It serves as the entry point for the RAPID framework and runs the entire pipeline, from 
    data generation, simulation, training, to evaluation.

    Attributes:
        geometry (BaseGeometry): An instance of a geometry class that defines the geometry to be used
    """

    def __init__(self, geometry: BaseGeometry):
        self.geometry = geometry
        self.geometry_instances = []

    def run(self, cpu_cores: int = multiprocessing.cpu_count(), num_samples: int = 1000):
        """
        Runs the RAPID pipeline, including data generation, simulation, training, and evaluation.
        """
        self.print_super_cool_logo_art()

        print(f"Running {num_samples} RAPID simulations of {self.geometry.name} with {cpu_cores} CPU cores...")                

        self.generate_data(num_samples)
        self.run_simulation()
        self.train_model()
        self.evaluate_model()

        print("RAPID pipeline finished successfully.")

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
        print("Welcome to RAPID - RF AI Pipeline for Integrated Circuit Design")
        print("")

    def generate_data(self, num_samples: int):
        """
        Generates data based on the defined geometry.
        """
        print("Starting data generation using gdsfactory...")
        for i in range(num_samples):
            # TODO: Generate input parameters for the geometry
            input_params = self.geometry.get_next_input_parameters()
            geo_inst = self.geometry.create_geometry_instance(name=f"{self.geometry.name}_{i}", input_parameters=input_params)
            self.geometry_instances.append(geo_inst)                   
            
        print("#----------- Data generation completed. -----------#")

    def run_simulation(self):
        """
        Runs simulations on the generated data.
        """
        print("Starting simulations with palace...")
        create_palace_model_from_gds(
            geometry=self.geometry,
            gds_filename=self.geometry.create_geometry_instance([]),
            simconfig_filename=self.geometry.simconfig_filename
        )
        print("#----------- Simulations completed. -----------#")

    def train_model(self):
        """
        Trains the RAPID model using the simulation data.
        """
        print("Starting model training...")
        print("#----------- Model training completed. -----------#")

    def evaluate_model(self):
        """
        Evaluates the trained RAPID model.
        """
        print("Starting model evaluation...")
        print("#----------- Model evaluation completed. -----------#")