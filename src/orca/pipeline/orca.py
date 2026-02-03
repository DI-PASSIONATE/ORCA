import os
import multiprocessing
from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger

class ORCA:
    """
    Main class for the ORCA framework. Manages all pipeline stages.
    """
    def __init__(self, stages: list[PipelineStage]):
        self.stages = stages
        # Sort stages by their index to ensure correct execution order
        self.stages.sort(key=lambda stage: stage.index)

    def run(self, geometry: BaseGeometry, cpu_cores: int = multiprocessing.cpu_count(), progress_callback=None):
        """
        Runs the ORCA pipeline with the specified geometry and CPU cores.

        Args:
            geometry (BaseGeometry): The geometry to be used in the pipeline.
            cpu_cores (int): Number of CPU cores to utilize.
        """
        self.print_super_cool_logo_art()

        context = {
            "geometry": geometry,
            "cpu_cores": cpu_cores,
            "base_dir": os.path.join(os.getcwd(), "output", geometry.name)
        }

        if os.path.exists(context["base_dir"]):
            # Ask user to confirm overwriting existing output directory
            response = input(f"Output directory {context['base_dir']} already exists. Overwrite? (y/n): ")
            if response.lower() != 'y':
                logger.info("Aborting pipeline run.")
                return
            else:
                # Clear the existing directory
                import shutil
                shutil.rmtree(context["base_dir"])

        for stage in self.stages:
            def stage_callback(percentage: float, message: str):
                if progress_callback:
                    progress_callback(stage.name, percentage, message)
            
            context = stage.run(context, progress_callback=stage_callback)


    def print_super_cool_logo_art(self):
        logger.info("###########################################################")  
        logger.info(" ██████╗ ██████╗  ██████╗ █████╗ ")
        logger.info("██╔═══██╗██╔══██╗██╔════╝██╔══██╗")
        logger.info("██║   ██║██████╔╝██║     ███████║")
        logger.info("██║   ██║██╔══██╗██║     ██╔══██║")
        logger.info("╚██████╔╝██║  ██║╚██████╗██║  ██║")
        logger.info(" ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝")       
        logger.info("###########################################################")
        logger.info("Welcome to ORCA - Open AI-Assisted RF Circuit Automation")
        logger.info("###########################################################")