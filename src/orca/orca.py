import json
import os
from typing import Callable, Optional
from orca.pipeline.context import PipelineContext
from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger


def default_process_count() -> int:
    """Cores this process may actually use.

    ``os.cpu_count()`` reports the machine's cores, which overcommits under a
    Slurm allocation or a container CPU limit; the affinity mask respects both.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Not available on macOS/Windows
        return os.cpu_count() or 1


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
        num_processes: Optional[int] = None,
        force_overwrite: bool = False,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
        overwrite_callback: Optional[Callable[[str], bool]] = None,
        base_dir: Optional[str] = None,
        result_dir: Optional[str] = None,
        result_csv: Optional[str] = None,
    ) -> Optional[PipelineContext]:
        """
        Runs the ORCA pipeline with the specified geometry and CPU cores.

        Args:
            geometry (BaseGeometry): The geometry to be used in the pipeline.
            num_processes (int|None): Number of MPI processes to use for parallel execution.
                Defaults to the number of CPU cores available to this process.
            force_overwrite (bool): Skip the confirmation prompt when the output directory already exists.
            progress_callback: Called as (stage_name, current, total, message) while stages run.
            overwrite_callback: Asked whether to continue when the output directory exists.
            base_dir (str|None): Output directory. Defaults to ./output/<geometry name>.
            result_dir (str|None): Directory of existing simulation results. Set this to train on
                results that were simulated earlier, instead of the pipeline's own results folder.
            result_csv (str|None): Parameter CSV describing those results. Defaults to
                <result_dir>/<geometry name>.csv.

        Returns:
            PipelineContext|None: The final context, or None if the run was aborted.
        """
        self.print_super_cool_logo_art()

        context = PipelineContext(
            geometry=geometry,
            num_processes=num_processes or default_process_count(),
            base_dir=base_dir or os.path.join(os.getcwd(), "output", geometry.name),
            result_dir_override=result_dir,
            result_csv_override=result_csv,
        )

        if os.path.exists(context.base_dir) and not force_overwrite:
            # Ask user to confirm overwriting existing output directory
            if overwrite_callback:
                if not overwrite_callback(context.base_dir):
                    logger.info("Aborting pipeline run.")
                    return None
            else:
                response = input(
                    f"Output directory {context.base_dir} already exists. Stages may overwrite existing files. Continue? (y/n): "
                )
                if response.lower() != "y":
                    logger.info("Aborting pipeline run.")
                    return None

        for stage in self.stages:
            def stage_callback(stage_name: str, current: int, total: int, message: str):
                if progress_callback:
                    progress_callback(stage_name, current, total, message)

            context = stage.run(context, progress_callback=stage_callback)

        logger.info("ORCA pipeline completed successfully.")
        self._save_context(context)
        return context

    @staticmethod
    def _save_context(context: PipelineContext) -> None:
        """Write a JSON record of the run, for debugging and reproducibility."""
        os.makedirs(context.base_dir, exist_ok=True)
        context_save_path = os.path.join(context.base_dir, "context.json")
        with open(context_save_path, "w") as f:
            json.dump(context.to_json_dict(), f, default=str, indent=4)
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
