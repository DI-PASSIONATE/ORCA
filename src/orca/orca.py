import os
import multiprocessing
from typing import Callable, Optional
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

    def run(
        self,
        geometry: BaseGeometry,
        num_processes: int = multiprocessing.cpu_count(),
        force_overwrite: bool = False,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        overwrite_callback: Optional[Callable[[str], bool]] = None,
        base_dir: Optional[str] = None,
        result_dir: Optional[str] = None,
        result_csv: Optional[str] = None,
    ):
        """
        Runs the ORCA pipeline with the specified geometry and CPU cores.

        Args:
            geometry (BaseGeometry): The geometry to be used in the pipeline.
            num_processes (int): Number of MPI processes to use for parallel execution. Default is the number of available CPU cores.
            force_overwrite (bool): Skip the confirmation prompt when the output directory already exists.
            progress_callback: Called as (stage_name, current, total, message) while stages run.
            overwrite_callback: Asked whether to continue when the output directory exists.
            base_dir (str|None): Output directory. Defaults to ./output/<geometry name>.
            result_dir (str|None): Directory of existing simulation results. Set this to train on
                results that were simulated earlier, instead of the pipeline's own results folder.
            result_csv (str|None): Parameter CSV describing those results. Defaults to
                <result_dir>/<geometry name>.csv.
        """
        self.print_super_cool_logo_art()

        context = {
            "geometry": geometry,
            "num_processes": num_processes,
            "base_dir": base_dir or os.path.join(os.getcwd(), "output", geometry.name),
        }

        if result_dir is not None:
            context["result_dir"] = result_dir
        if result_csv is not None:
            context["result_csv"] = result_csv

        if os.path.exists(context["base_dir"]) and not force_overwrite:
            # Ask user to confirm overwriting existing output directory
            if overwrite_callback:
                if not overwrite_callback(context["base_dir"]):
                    logger.info("Aborting pipeline run.")
                    return
            else:
                response = input(
                    f"Output directory {context['base_dir']} already exists. Stages may overwrite existing files. Continue? (y/n): "
                )
                if response.lower() != "y":
                    logger.info("Aborting pipeline run.")
                    return

        for stage in self.stages:
            def stage_callback(stage_name: str, current: int, total: int, message: str):
                if progress_callback:
                    progress_callback(stage_name, current, total, message)

            context = stage.run(context, progress_callback=stage_callback)

        logger.info("ORCA pipeline completed successfully.")
        # Save context to a file for debugging and reproducibility
        import json
        os.makedirs(context["base_dir"], exist_ok=True)
        context_save_path = os.path.join(context["base_dir"], "context.json")
        with open(context_save_path, "w") as f:
            json.dump(context, f, default=str, indent=4)
        logger.info(f"Context saved to {context_save_path}")

    def print_super_cool_logo_art(self):
        logger.info("##########################################")
        logger.info("#               Welcome to               #")
        logger.info("##########################################")
        logger.info("#    ██████╗ ██████╗  ██████╗ █████╗     #")
        logger.info("#   ██╔═══██╗██╔══██╗██╔════╝██╔══██╗    #")
        logger.info("#   ██║   ██║██████╔╝██║     ███████║    #")
        logger.info("#   ██║   ██║██╔══██╗██║     ██╔══██║    #")
        logger.info("#   ╚██████╔╝██║  ██║╚██████╗██║  ██║    #")
        logger.info("#    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝    #")
        logger.info("##########################################")
        logger.info("# Open AI-Assisted RF Circuit Automation #")
        logger.info("##########################################")
