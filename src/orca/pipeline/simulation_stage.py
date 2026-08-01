from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Any, Dict, Callable
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
        extra_srun_args: str = "",
    ):
        """
        Initializes the PalaceSimulator stage.

        Args:
            palace_executable (str): Path to the Palace executable. Default is "palace".
            touchstone_type (str): Type of Touchstone file to generate. One of "all", "normal", "deembedded", "dc", "dc_deembedded". 
            num_parallel_palace_sims (int): Number of Palace simulations to run in parallel. When > 1,
                each simulation bypasses the Palace wrapper script's own `mpirun` call and is instead
                launched directly as `srun --exclusive --nodes=1 --ntasks=num_processes <palace-bin> config`,
                so Slurm packs each simulation onto its own dedicated node (see `run_palace` in
                simulate.py for details). Requires running inside a Slurm allocation with at least this
                many nodes (e.g. `sbatch --nodes=<num_parallel_palace_sims>`), and `palace_executable`
                to be a direct filesystem path to the Palace wrapper script (not a container-wrapped
                compound command). Default is 1, which runs simulations sequentially on the current
                node without Slurm, using `palace_executable` as-is.
            extra_srun_args (str): Additional arguments appended to the `srun` command used when
                `num_parallel_palace_sims` > 1 (e.g. "--cpu-bind=cores"). Ignored otherwise.
        """
        super().__init__(name="Palace EM Simulator", index=2)
        self.palace_executable = palace_executable
        self.touchstone_type = touchstone_type
        if num_parallel_palace_sims < 1:
            raise ValueError("num_parallel_palace_sims must be at least 1.")
        self.num_parallel_palace_sims = num_parallel_palace_sims
        self.extra_srun_args = extra_srun_args

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

        use_slurm = self.num_parallel_palace_sims > 1

        if use_slurm:
            logger.info(
                f"Starting Palace EM simulations for {len(palace_data)} models, "
                f"{self.num_parallel_palace_sims} in parallel across Slurm nodes, "
                f"each using {num_processes} MPI processes."
            )
        else:
            logger.info(
                f"Starting Palace EM simulations for {len(palace_data)} models using {num_processes} MPI processes."
            )

        completed = 0
        if use_slurm:
            # Run simulations in parallel, one per Slurm node (via srun), since a single simulation
            # is already parallelized internally via MPI and does not benefit from more than one node.
            with ThreadPoolExecutor(max_workers=self.num_parallel_palace_sims) as executor:
                futures = {
                    executor.submit(
                        self._run_single_simulation, index, row, output_dir, num_processes, use_slurm
                    ): index
                    for index, row in palace_data.iterrows()
                }
                for future in tqdm.tqdm(
                    as_completed(futures), total=len(futures), desc="Palace Simulations"
                ):
                    index, palace_config_name, success = future.result()
                    if not success:
                        logger.error(
                            f"Simulation for config {palace_config_name} failed. Removing from results."
                        )
                        result_data.drop(index, inplace=True)

                    completed += 1
                    if progress_callback:
                        progress_callback(
                            self.name,
                            completed,
                            len(palace_data),
                            f"Simulated {completed} of {len(palace_data)} models.",
                        )
        else:
            for index, row in tqdm.tqdm(
                palace_data.iterrows(), total=len(palace_data), desc="Palace Simulations"
            ):
                index, palace_config_name, success = self._run_single_simulation(
                    index, row, output_dir, num_processes, use_slurm
                )

                if not success:
                    # Remove failed simulation from the CSV
                    logger.error(
                        f"Simulation for config {palace_config_name} failed. Removing from results."
                    )
                    result_data.drop(index, inplace=True)

                completed += 1
                if progress_callback:
                    progress_callback(
                        self.name,
                        completed,
                        len(palace_data),
                        f"Simulated {completed} of {len(palace_data)} models.",
                    )

        # Save updated results CSV
        result_data.to_csv(result_csv, index=False)

        context["result_dir"] = output_dir
        context["result_csv"] = result_csv
        logger.info("Palace EM simulations completed.")
        return context

    def _run_single_simulation(
        self,
        index: Any,
        row: "pd.Series",
        output_dir: str,
        num_processes: int,
        use_srun: bool,
    ) -> tuple[Any, str, bool]:
        """
        Patches the config for and runs a single Palace simulation.

        Args:
            index (Any): The row index of this simulation in the palace_data DataFrame, passed through
                so failures can be mapped back to the correct row.
            row (pd.Series): Row from the palace_data DataFrame describing the simulation to run.
            output_dir (str): Directory to write Touchstone results to.
            num_processes (int): Number of MPI processes to use for the simulation.
            use_srun (bool): Whether to pin this simulation to its own Slurm node via `srun`. See
                `run_palace` for details.

        Returns:
            tuple[Any, str, bool]: The row index, the Palace config name, and whether the simulation succeeded.
        """
        palace_config_name = row["config_name"]
        data_directory = row["data_dir"]
        sim_path = row["sim_path"]

        # Patch the auto-generated config.json to speed up simulation time
        self._patch_palace_config(os.path.join(sim_path, palace_config_name))

        # This is not parallelized across CPUs because Palace is already parallelized very well
        # internally via MPI; num_parallel_palace_sims instead parallelizes across Slurm nodes.
        # Also, we convert results after each simulation instead of an extra stage to allow using intermediate results
        success = run_palace(
            sim_path=sim_path,
            data_dir=data_directory,
            result_dir=output_dir,
            config_name=palace_config_name,
            palace_executable=self.palace_executable,
            touchstone_type=self.touchstone_type,
            num_processes=num_processes,
            use_srun=use_srun,
            extra_srun_args=self.extra_srun_args,
        )

        return index, palace_config_name, success

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
