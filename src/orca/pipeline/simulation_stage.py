from typing import Optional, Any, Dict, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
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

    def __init__(
        self,
        palace_executable: str = "palace",
        touchstone_type: str = "dc_deembedded",
        num_parallel_palace_sims: int = 1,
    ):
        """
        Initializes the PalaceSimulator stage.

        Args:
            palace_executable (str): Path to the Palace executable. Default is "palace".
            touchstone_type (str): Type of Touchstone file to generate. One of "all", "normal", "deembedded", "dc", "dc_deembedded".
            num_parallel_palace_sims (int): Number of Palace simulations to run concurrently. Default is 1
                (sequential, single node/machine). When set to a value greater than 1 while running inside a
                Slurm allocation (i.e. multiple nodes requested via `#SBATCH --nodes=N`), each concurrent
                simulation is pinned to its own dedicated node via `srun`, so that one node hosts exactly one
                whole Palace simulation (using `num_processes` MPI ranks local to that node). If no Slurm
                allocation is detected, simulations are still run in parallel on the local machine, but without
                per-node isolation.
        """
        super().__init__(name="Palace EM Simulator", index=2)
        self.palace_executable = palace_executable
        self.touchstone_type = touchstone_type
        self.num_parallel_palace_sims = num_parallel_palace_sims

    def run(
        self,
        context: Dict[str, Any],
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        num_processes: int = context.get("num_processes", 1)
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
            f"Starting Palace EM simulations for {len(palace_data)} models using {num_processes} MPI processes."
        )

        running_under_slurm = "SLURM_JOB_ID" in os.environ
        parallel_sims = max(1, self.num_parallel_palace_sims)

        if parallel_sims > 1 and not running_under_slurm:
            logger.warning(
                f"num_parallel_palace_sims={parallel_sims} was requested, but no Slurm allocation was "
                "detected (SLURM_JOB_ID not set). Simulations will still run in parallel on this machine, "
                "but without per-node isolation, so make sure num_processes leaves enough cores free for "
                "all simulations running at once."
            )

        # When multiple nodes are available in a Slurm allocation, pin each concurrent simulation to
        # its own dedicated node so that one node runs exactly one whole (MPI-parallel) simulation.
        # --mpi=none stops srun from setting up its PMIx server for this step; without it, mpirun
        # inside would still connect to Slurm's PMIx and see only the 1 task/slot this step
        # requested (regardless of stripping SLURM_* env vars), causing "not enough slots" errors.
        command_prefix = (
            "srun --exclusive --nodes=1 --ntasks=1 --cpu-bind=none --mpi=none"
            if parallel_sims > 1 and running_under_slurm
            else ""
        )

        if parallel_sims <= 1:
            # Sequential execution: one Palace simulation at a time on the current node/machine
            for i, (index, row) in tqdm.tqdm(
                enumerate(palace_data.iterrows()),
                total=len(palace_data),
                desc="Palace Simulations",
            ):
                success = self._run_single_simulation(row, output_dir, num_processes, command_prefix)

                if not success:
                    # Remove failed simulation from the CSV
                    logger.error(
                        f"Simulation for config {row['config_name']} failed. Removing from results."
                    )
                    result_data.drop(index, inplace=True)

                if progress_callback:
                    progress_callback(
                        self.name,
                        i + 1,
                        len(palace_data),
                        f"Simulated {i + 1} of {len(palace_data)} models.",
                    )
        else:
            # Parallel execution: up to `num_parallel_palace_sims` simulations run at once
            with ThreadPoolExecutor(max_workers=parallel_sims) as executor:
                future_to_info = {
                    executor.submit(
                        self._run_single_simulation, row, output_dir, num_processes, command_prefix
                    ): (index, row["config_name"])
                    for index, row in palace_data.iterrows()
                }

                for i, future in enumerate(
                    tqdm.tqdm(
                        as_completed(future_to_info),
                        total=len(future_to_info),
                        desc="Palace Simulations",
                    )
                ):
                    index, palace_config_name = future_to_info[future]

                    try:
                        success = future.result()
                    except Exception as e:
                        logger.error(
                            f"Simulation for config {palace_config_name} raised an exception: {e}"
                        )
                        success = False

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

    def _run_single_simulation(
        self,
        row: "pd.Series",
        output_dir: str,
        num_processes: int,
        command_prefix: str,
    ) -> bool:
        """
        Patches the Palace config for a single model and runs the simulation.

        Args:
            row (pd.Series): Row from the Palace CSV describing one model (config_name, data_dir, sim_path).
            output_dir (str): Directory to write Touchstone results to.
            num_processes (int): Number of MPI processes to use for this simulation.
            command_prefix (str): Optional command prefix (e.g. an `srun` node-pinning command) prepended
                before the Palace executable.

        Returns:
            bool: True if the simulation was successful, False otherwise.
        """
        palace_config_name = row["config_name"]
        data_directory = row["data_dir"]
        sim_path = row["sim_path"]

        # Patch the auto-generated config.json to speed up simulation time
        self._patch_palace_config(os.path.join(sim_path, palace_config_name))

        # Runs palace and converts results to Touchstone format
        # Also, we convert results after each simulation instead of an extra stage to allow using intermediate results
        return run_palace(
            sim_path=sim_path,
            data_dir=data_directory,
            result_dir=output_dir,
            config_name=palace_config_name,
            palace_executable=self.palace_executable,
            touchstone_type=self.touchstone_type,
            num_processes=num_processes,
            command_prefix=command_prefix,
        )

    def _patch_palace_config(self, config_path: str) -> None:
        """
        Applies ORCA-specific overrides to a Palace config.json generated by gds2palace.

        Args:
            config_path (str): Path to the Palace config.json file to patch.
        """
        with open(config_path, "r") as f:
            config = json.load(f)

        config.setdefault("Problem", {})["OutputFormats"] = {"Paraview": False}
        config.setdefault("Model", {})["ReorderElements"] = True
        solver = config.setdefault("Solver", {})
        solver["PartialAssemblyOrder"] = 3
        solver.setdefault("Linear", {})["Type"] = "SuperLU"
        config.setdefault("Problem", {})["Verbose"] = 1

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
