from typing import Optional, Any, Dict, Callable
import os
import pandas as pd
import tqdm

from orca.pipeline.pipeline_stage import PipelineStage
from orca.logger import logger
from orca.simulation.simulate import run_palace
from orca.utils.folder_structure import OrcaFolderStructure


class PalaceSimulator(PipelineStage):
    """
    Pipeline stage for running Palace EM simulations.
    """

    def __init__(self, palace_executable: str = "palace", touchstone_type: str = "dc_deembedded"):
        """
        Initializes the PalaceSimulator stage.

        Args:
            palace_executable (str): Path to the Palace executable. Default is "palace".
            touchstone_type (str): Type of Touchstone file to generate. One of "all", "normal", "deembedded", "dc", "dc_deembedded". 
        """
        super().__init__(name="Palace EM Simulator", index=2)
        self.palace_executable = palace_executable
        self.touchstone_type = touchstone_type

    def run(
        self,
        context: Dict[str, Any],
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        cpu_cores: int = context.get("cpu_cores", 16)
        output_dir = OrcaFolderStructure.get_result_dir(context)
        palace_csv = OrcaFolderStructure.get_palace_csv(context)
        result_csv = OrcaFolderStructure.get_result_csv(context)
        palace_data = pd.read_csv(palace_csv)  # Information

        if os.path.exists(output_dir):
            import shutil

            shutil.rmtree(output_dir)

        os.makedirs(output_dir)

        # Prepare output CSV
        result_data = palace_data.copy()  # Drop columns data_dir,sim_path,config_name
        result_data.drop(
            columns=["data_dir", "sim_path", "config_name"],
            inplace=True,
            errors="ignore",
        )

        logger.info(
            f"Starting Palace EM simulations for {len(palace_data)} models using {cpu_cores} CPU cores."
        )

        for i, (index, row) in tqdm.tqdm(
            enumerate(palace_data.iterrows()),
            total=len(palace_data),
            desc="Palace Simulations",
        ):
            palace_config_name = row["config_name"]
            data_directory = row["data_dir"]
            sim_path = row["sim_path"]

            # Runs palace and converts results to Touchstone format
            # This is not done in parallel because Palace is already parallelized very well internally, so we run one simulation with max cores
            # Also, we convert results after each simulation instead of an extra stage to allow using intermediate results
            success = run_palace(
                sim_path=sim_path,
                data_dir=data_directory,
                result_dir=output_dir,
                config_name=palace_config_name,
                palace_executable=self.palace_executable,
                cpu_cores=cpu_cores,
                touchstone_type=self.touchstone_type,
            )

            if not success:
                # Remove failed simulation from the CSV
                logger.error(
                    f"Simulation for config {palace_config_name} failed. Removing from results."
                )
                result_data.drop(index, inplace=True)

            if progress_callback:
                progress_callback(
                    self.name,
                    i + 1,
                    len(palace_data),
                    f"Simulated {i + 1} of {len(palace_data)} models.",
                )

        # Save updated results CSV
        result_data.to_csv(result_csv, index=False)

        context["result_dir"] = output_dir
        context["result_csv"] = result_csv
        logger.info("Palace EM simulations completed.")
        return context
