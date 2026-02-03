from typing import Optional, Any, Dict, Callable
import os
import pandas as pd
import tqdm

from orca.pipeline.pipeline_stage import PipelineStage
from orca.geometry.base_geometry import BaseGeometry
from orca.logger import logger
from orca.simulation.simulate import run_palace


class PalaceSimulator(PipelineStage):
    """
    Pipeline stage for running Palace EM simulations.
    """

    def __init__(self, palace_executable: str = "palace"):
        super().__init__(name="Palace EM Simulator", index=2)
        self.palace_executable = palace_executable

    def run(
        self,
        context: Dict[str, Any],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Dict[str, Any]:
        geometry: BaseGeometry = context["geometry"]
        cpu_cores: int = context.get("cpu_cores", 16)
        base_dir: str = context.get("base_dir", os.getcwd())
        output_dir = os.path.join(
            base_dir, "results"
        )  # Touchstone results get stored here
        palace_csv = context.get(
            "palace_csv", base_dir + f"/palace_sims/{geometry.name}.csv"
        )
        result_csv = os.path.join(output_dir, f"{geometry.name}.csv")

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
            )

            if not success:
                # Remove failed simulation from the CSV
                logger.error(
                    f"Simulation for config {palace_config_name} failed. Removing from results."
                )
                result_data.drop(index, inplace=True)

            if progress_callback:
                percentage = (i + 1) / len(palace_data) * 100
                progress_callback(
                    percentage, f"Simulated {i + 1} of {len(palace_data)} models."
                )

        # Save updated results CSV
        result_data.to_csv(result_csv, index=False)

        context["result_dir"] = output_dir
        context["result_csv"] = result_csv
        logger.info("Palace EM simulations completed.")
        return context
