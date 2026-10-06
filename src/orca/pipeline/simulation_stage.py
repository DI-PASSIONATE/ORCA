import os
import shutil
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import TYPE_CHECKING, Any

import pandas as pd
import tqdm

from orca.logger import logger
from orca.pipeline.pipeline_stage import PipelineStage
from orca.pipeline.resume import append_row, read_table, resume_table, write_table
from orca.simulation.combine_snp_results import touchstone_filename
from orca.simulation.launchers import BIND_CHOICES, LocalLauncher, SimulationLauncher
from orca.simulation.simulate import patch_palace_config, run_palace

if TYPE_CHECKING:
    from orca.pipeline.context import PipelineContext


class PalaceSimulator(PipelineStage):
    """
    Pipeline stage for running Palace EM simulations.

    Each finished simulation is added to the result table right away, so the table
    stays valid when the run is killed (e.g. by a Slurm time limit). A later run
    keeps those results and only simulates the models that are missing, that failed,
    or whose parameters changed since.
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
        overwrite: bool = False,
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
            overwrite (bool): Delete the results folder first instead of keeping the results an
                earlier run left there. Refused for a results folder passed to `ORCA.run`, which
                is never deleted.
        """
        super().__init__(name="Palace EM Simulator", index=3)
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
        self.overwrite = overwrite

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
        if not os.path.exists(palace_csv):
            logger.error(
                f"No Palace model table found at {palace_csv}. The GDS Converter stage was "
                "skipped or produced no models; nothing to simulate."
            )
            return context
        palace_data = pd.read_csv(palace_csv)  # Information

        if self.overwrite and os.path.exists(output_dir):
            if context.result_dir_override:
                raise ValueError(
                    f"PalaceSimulator(overwrite=True) would delete {output_dir}, the results "
                    "folder passed to ORCA.run. That folder is never deleted; turn overwrite off "
                    "to add to it, or pass another folder."
                )
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        # Prepare output CSV
        result_data = palace_data.drop(
            columns=["data_dir", "sim_path", "config_name"], errors="ignore"
        )

        # The name column arrives as "<geometry>.gds" from the GDS generation stage. Replace it
        # with the Touchstone file this stage actually produces, so downstream datasets can open
        # the file directly
        n_ports = context.geometry.n_ports
        result_data["name"] = result_data["name"].apply(
            lambda name: touchstone_filename(name, n_ports, self.touchstone_type)
        )

        # Keep the results an earlier run finished for the same parameters; the table then
        # grows by one row per simulation that finishes in this run.
        simulated = resume_table(
            result_csv,
            result_data,
            list(result_data.columns),
            lambda row: os.path.exists(os.path.join(output_dir, row["name"])),
        )
        pending = palace_data[~result_data["name"].isin(simulated)]
        if simulated:
            logger.info(
                f"{len(simulated)} of {len(palace_data)} models were simulated by an earlier run "
                f"(listed in {result_csv}); skipping them."
            )

        launcher = self._create_launcher()
        num_processes = launcher.ranks_per_simulation(num_processes)
        logger.info(
            f"Starting Palace EM simulations for {len(pending)} models, "
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
                for index, row in pending.iterrows()
            }
            for future in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Palace Simulations"
            ):
                index = futures[future]
                palace_config_name = palace_data.loc[index, "config_name"]
                touchstone_name = result_data.loc[index, "name"]
                try:
                    _, _, success = future.result()
                    reason = "failed"
                    if success and not os.path.exists(os.path.join(output_dir, touchstone_name)):
                        success, reason = False, f"did not produce {touchstone_name}"
                except Exception as e:  # noqa: BLE001 - one bad simulation must not abort the batch
                    success, reason = False, f"raised {e!r}"
                if success:
                    # Written as soon as the Touchstone file exists, so the table lists every
                    # finished simulation even if this run is killed before the others finish.
                    append_row(result_csv, result_data.loc[index].to_dict())
                else:
                    logger.error(
                        f"Simulation for config {palace_config_name} {reason}. Leaving it out of "
                        "the results; a later run retries it."
                    )

                completed += 1
                if progress_callback:
                    progress_callback(
                        self.name,
                        completed,
                        len(pending),
                        f"Simulated {completed} of {len(pending)} models.",
                    )

        # Rows were appended in completion order; restore the order of the Palace table so the
        # table (and a split drawn from it) does not depend on which simulation finished first.
        results = read_table(result_csv)
        if results is not None:
            order = {name: i for i, name in enumerate(result_data["name"])}
            write_table(results.sort_values("name", key=lambda names: names.map(order)), result_csv)

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
        patch_palace_config(os.path.join(sim_path, palace_config_name))

        # Results are converted right after each simulation instead of in an extra stage, so
        # intermediate results are usable while the stage is still running.
        success = run_palace(
            sim_path=sim_path,
            data_dir=data_directory,
            result_dir=output_dir,
            cmd=launcher.command(slot, self.palace_executable, num_processes, palace_config_name),
            touchstone_type=self.touchstone_type,
            save_log=self.save_log,
            launch_failed=launcher.launch_failed,
        )

        return index, palace_config_name, success
