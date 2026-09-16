import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import TYPE_CHECKING, Any

import pandas as pd
import tqdm

from orca.logger import logger
from orca.pipeline.pipeline_stage import PipelineStage
from orca.simulation.combine_snp_results import touchstone_filename
from orca.simulation.launchers import BIND_CHOICES, LocalLauncher, SimulationLauncher
from orca.simulation.simulate import run_palace

if TYPE_CHECKING:
    from orca.pipeline.context import PipelineContext


class PalaceSimulator(PipelineStage):
    """
    Pipeline stage for running Palace EM simulations.
    """

    def __init__(
        self,
        palace_executable: str = "palace",
        touchstone_type: str = "dc_deembedded",
        launcher: str = "local",
        num_parallel_sims: int = 1,
        bind: str = "node",
        extra_srun_args: str = "",
        save_log: bool = False,
        hyperthreads: bool = False,
    ):
        """
        Initializes the PalaceSimulator stage.

        Args:
            palace_executable (str): Path to the Palace executable. Default is "palace".
            touchstone_type (str): Type of Touchstone file to generate. One of "all", "normal", "deembedded", "dc", "dc_deembedded".
            launcher (str): Where simulations run. "local" (default) runs them on this machine
                through the Palace wrapper (`palace -np num_processes config`); `palace_executable`
                may then also be a container invocation. "slurm" runs one simulation per node of the
                current Slurm allocation, each as its own `srun` job step (see
                `orca.simulation.slurm.SlurmLauncher`); `palace_executable` must then be a direct
                filesystem path to the Palace wrapper script.
            num_parallel_sims (int): Simulations to run at once. 0 means "as many slots as there
                are": one per `bind` domain of every Slurm node for "slurm", one per `bind` domain of
                this machine for "local". A larger value is clamped with a warning. Default is 1
                (sequential). `num_processes` (passed to `ORCA.run`) is the number of MPI ranks *per
                simulation* and is capped to the cores of one slot.
            bind (str): What one simulation is bound to: "node" (default, a whole node or this whole
                machine), "socket" or "numa" (one socket / NUMA domain, so several simulations run per
                node with their own cores and local memory). For a memory-bandwidth-bound solver like
                Palace, "numa" with num_parallel_sims=0 usually gives the best throughput, as long as
                one simulation fits into the memory of a NUMA domain; use "socket" otherwise.
            extra_srun_args (str): "slurm" only: additional arguments appended to each `srun`
                command (e.g. "--mpi=pmi2" or "--mem-per-cpu=2G").
            save_log (bool): Write each simulation's full Palace/srun output to `palace.log` in its
                simulation folder. Off by default (Palace prints a lot); errors are still reported.
            hyperthreads (bool): Allow one MPI rank per hardware thread instead of per physical
                core. Palace is memory-bandwidth bound and usually gains nothing from SMT (two
                ranks then share one core's execution units, caches and bandwidth), so this is
                off by default; enable it to measure the difference on your machine.
        """
        super().__init__(name="Palace EM Simulator", index=2)
        self.palace_executable = palace_executable
        self.touchstone_type = touchstone_type
        if launcher not in ("local", "slurm"):
            raise ValueError(f"launcher must be 'local' or 'slurm', got {launcher!r}.")
        if num_parallel_sims < 0:
            raise ValueError("num_parallel_sims must be >= 0 (0 = one per slot).")
        if bind not in BIND_CHOICES:
            raise ValueError(f"bind must be one of {BIND_CHOICES}, got {bind!r}.")
        self.launcher = launcher
        self.num_parallel_sims = num_parallel_sims
        self.bind = bind
        self.extra_srun_args = extra_srun_args
        self.save_log = save_log
        self.hyperthreads = hyperthreads

    def _create_launcher(self) -> SimulationLauncher:
        """Builds the launcher for this run (reads the Slurm allocation / NUMA topology now, not at construction)."""
        if self.launcher == "slurm":
            from orca.simulation.slurm import SlurmLauncher

            return SlurmLauncher(
                num_parallel_sims=self.num_parallel_sims,
                bind=self.bind,
                extra_srun_args=self.extra_srun_args,
                hyperthreads=self.hyperthreads,
            )
        return LocalLauncher(
            num_parallel_sims=self.num_parallel_sims, bind=self.bind, hyperthreads=self.hyperthreads
        )

    def run(
        self,
        context: "PipelineContext",
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> "PipelineContext":
        num_processes: int = context.num_processes
        output_dir = context.result_dir
        palace_csv = context.palace_csv_path
        result_csv = context.result_csv
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

        # The name column arrives as "<geometry>.gds" from the GDS generation stage. Replace it
        # with the Touchstone file this stage actually produces, so downstream datasets can open
        # the file directly
        n_ports = context.geometry.n_ports
        result_data["name"] = result_data["name"].apply(
            lambda name: touchstone_filename(name, n_ports, self.touchstone_type)
        )

        launcher = self._create_launcher()
        num_processes = launcher.ranks_per_simulation(num_processes)
        logger.info(
            f"Starting Palace EM simulations for {len(palace_data)} models, "
            f"{launcher.describe()}, each using {num_processes} MPI processes."
        )
        # Fail fast if the launcher cannot start simulations with these settings, instead of hanging
        # silently in the first simulation (whose output is not shown on the console).
        launcher.check(num_processes)

        # One worker per slot: each takes a free slot from the queue, runs its simulation there and
        # hands the slot back afterwards, so simulations never share a slot (node / NUMA domain).
        # A single simulation is already parallelized internally via MPI, so this is the only
        # parallelism on top.
        free_slots: Queue[str] = Queue()
        for slot in launcher.slots:
            free_slots.put(slot)

        def run_in_free_slot(index: Any, row: "pd.Series") -> tuple[Any, str, bool]:
            slot = free_slots.get()
            try:
                return self._run_single_simulation(index, row, output_dir, num_processes, launcher, slot)
            finally:
                free_slots.put(slot)

        completed = 0
        with ThreadPoolExecutor(max_workers=len(launcher.slots)) as executor:
            futures = {
                executor.submit(run_in_free_slot, index, row): index
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

        # Save updated results CSV
        result_data.to_csv(result_csv, index=False)

        logger.info("Palace EM simulations completed.")
        return context

    def _run_single_simulation(
        self,
        index: Any,
        row: "pd.Series",
        output_dir: str,
        num_processes: int,
        launcher: SimulationLauncher,
        slot: str,
    ) -> tuple[Any, str, bool]:
        """
        Patches the config for and runs a single Palace simulation.

        Args:
            index (Any): The row index of this simulation in the palace_data DataFrame, passed through
                so failures can be mapped back to the correct row.
            row (pd.Series): Row from the palace_data DataFrame describing the simulation to run.
            output_dir (str): Directory to write Touchstone results to.
            num_processes (int): Number of MPI processes to use for the simulation.
            launcher (SimulationLauncher): Builds the launch command for the simulation.
            slot (str): The launcher slot (Slurm node, NUMA domain, ...) to run in.

        Returns:
            tuple[Any, str, bool]: The row index, the Palace config name, and whether the simulation succeeded.
        """
        palace_config_name = row["config_name"]
        data_directory = row["data_dir"]
        sim_path = row["sim_path"]

        # Patch the auto-generated config.json to speed up simulation time
        self._patch_palace_config(os.path.join(sim_path, palace_config_name))

        # Results are converted right after each simulation instead of in an extra stage, so
        # intermediate results are usable while the stage is still running.
        success = run_palace(
            sim_path=sim_path,
            data_dir=data_directory,
            result_dir=output_dir,
            cmd=launcher.command(slot, self.palace_executable, num_processes, palace_config_name),
            touchstone_type=self.touchstone_type,
            save_log=self.save_log,
        )

        return index, palace_config_name, success

    def _patch_palace_config(self, config_path: str) -> None:
        """
        Applies ORCA-specific overrides to a Palace config.json generated by gds2palace.

        Args:
            config_path (str): Path to the Palace config.json file to patch.
        """
        with open(config_path) as f:
            config = json.load(f)

        config.setdefault("Problem", {})["OutputFormats"] = {"Paraview": False}
        config.setdefault("Model", {})["ReorderElements"] = True
        solver = config.setdefault("Solver", {})
        solver["PartialAssemblyOrder"] = 3
        solver.setdefault("Linear", {})["Type"] = "SuperLU"
        config.setdefault("Problem", {})["Verbose"] = 1

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
