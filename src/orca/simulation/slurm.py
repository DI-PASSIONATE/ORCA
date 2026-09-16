"""
Slurm support: runs one Palace simulation per node of the current allocation, each launched as
its own job step with `srun`, bypassing the Palace wrapper script's `mpirun`.
"""

import glob
import os
import platform
import subprocess

from orca.logger import logger
from orca.simulation.launchers import CpuDomain, SimulationLauncher, cpu_domains


def slurm_allocation() -> tuple[list[str], int]:
    """
    Reads the node list and the number of CPUs per node of the current Slurm allocation.

    Returns:
        tuple[list[str], int]: The hostnames of all allocated nodes, and the smallest number of
            CPUs allocated on any of them (so a per-node value is valid on every node).

    Raises:
        RuntimeError: If not running inside a Slurm allocation or the node list cannot be read.
    """
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    cpus_spec = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if not nodelist or not cpus_spec:
        raise RuntimeError(
            "Not inside a Slurm allocation (SLURM_JOB_NODELIST/SLURM_JOB_CPUS_PER_NODE are not set). "
            "launcher='slurm' requires running under sbatch/salloc."
        )

    # Expand the compact node list (e.g. "f[0101-0102,0110]") into hostnames
    ret = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],  # noqa: S607 - scontrol comes from the Slurm PATH
        capture_output=True,
        text=True,
        check=False,
    )
    if ret.returncode != 0:
        raise RuntimeError(f"Could not expand Slurm node list '{nodelist}': {ret.stderr.strip()}")
    nodes = ret.stdout.split()

    # SLURM_JOB_CPUS_PER_NODE looks like "72(x2)" or "72,36(x3)"
    cpus_per_node = min(int(part.split("(")[0]) for part in cpus_spec.split(","))
    return nodes, cpus_per_node


def resolve_palace_binary(palace_executable: str) -> str:
    """
    Mimics the Palace wrapper script's own binary lookup (`find $PALACE_DIR -name "palace-*.bin"`)
    to find the real MPI-enabled Palace binary next to the wrapper script, so it can be launched
    directly via `srun` instead of through the wrapper's own `mpirun` call.

    Args:
        palace_executable (str): Path to the Palace wrapper script (e.g. "~/palace/build/bin/palace").
            Must be a plain filesystem path to the wrapper, not a compound command (e.g. via
            apptainer/singularity), since there is no wrapper script to look next to in that case.

    Returns:
        str: Absolute path to the resolved `palace-*.bin` binary.
    """
    wrapper_path = os.path.expanduser(palace_executable.strip())
    palace_dir = os.path.dirname(wrapper_path)
    candidates = glob.glob(os.path.join(palace_dir, "**", "palace-*.bin"), recursive=True)

    arch = "arm64" if platform.machine() in ("arm64", "aarch64") else "x86_64"
    matches = [path for path in candidates if arch in os.path.basename(path)]

    if not matches:
        raise FileNotFoundError(
            f"Could not locate a 'palace-*.bin' executable for architecture '{arch}' under '{palace_dir}'. "
            "Launching via srun requires palace_executable to be a direct filesystem path to the Palace wrapper script."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple 'palace-*.bin' candidates under '{palace_dir}': {matches}. "
            "Could not unambiguously resolve the Palace binary for launching via srun."
        )
    return matches[0]


class SlurmLauncher(SimulationLauncher):
    """
    Runs simulations on the nodes of the current Slurm allocation, one per node, socket or NUMA
    domain (`bind`). Each simulation is a job step pinned to its node and CPUs:

        PMIX_MCA_psec=native srun --nodes=1 --nodelist=<node> --ntasks=<ranks> --overlap
            --cpu-bind=<cores | map_cpu:<cpus of the domain>> --mpi=pmix <palace-*.bin> <config>

    The step is pinned to the node explicitly (--nodelist) instead of letting Slurm pick a node with
    free CPUs, and uses --overlap instead of --exclusive so it never waits on Slurm's step-level
    CPU/memory accounting (the batch step running the Python script also occupies CPUs on its
    node, which made --exclusive steps sized to a full node wait forever). Only one simulation runs
    per slot at a time, and sub-node slots are pinned to disjoint CPU sets, so steps never compete
    for cores. The CPU topology is read from the node running the Python script and assumed to be
    the same on all allocated nodes.

    --mpi=pmix is required so Open MPI can rendezvous through Slurm's PMIx server; without it,
    Open MPI silently launches num_processes independent single-rank "singleton" processes
    instead of one coordinated job, which then race each other (e.g. concurrently
    creating/checking the output directory) and fail unpredictably.
    PMIX_MCA_psec=native tells the PMIx client to skip the "munge" security component (which
    may not be built/loadable in this environment, causing "component was not found" warnings
    and potentially failing the handshake) and fall back to PMIx's always-available basic
    UID/GID-based security check instead.
    """

    def __init__(
        self,
        num_parallel_sims: int = 0,
        bind: str = "node",
        extra_srun_args: str = "",
        hyperthreads: bool = False,
    ):
        """
        Args:
            num_parallel_sims (int): Simulations to run at once. 0 (default) uses every slot of the
                allocation (nodes x domains per node); more than that is clamped with a warning.
                Slots are filled node by node.
            bind (str): "node" gives each simulation a whole node; "socket" or "numa" gives it one
                socket / NUMA domain of a node, so several simulations run per node, each with its
                own cores and local memory.
            extra_srun_args (str): Additional arguments appended to each `srun` command, e.g.
                "--mpi=pmi2" to override the PMI type or "--mem-per-cpu=2G".
            hyperthreads (bool): Count hardware threads instead of physical cores, allowing one
                MPI rank per thread (only meaningful on nodes with SMT enabled). Usually not
                faster for Palace; see `orca.simulation.launchers.cpu_domains`.

        Raises:
            RuntimeError: If not running inside a Slurm allocation.
        """
        nodes, self._cpus_per_node = slurm_allocation()
        self.bind = bind
        self.extra_srun_args = extra_srun_args
        domains = cpu_domains(bind, hyperthreads)

        all_slots = {f"{node}:{d.name}": (node, d) for node in nodes for d in domains}
        if num_parallel_sims > len(all_slots):
            logger.warning(
                f"num_parallel_sims={num_parallel_sims} exceeds the {len(all_slots)} {bind} slot(s) "
                f"of the {len(nodes)} allocated Slurm node(s); running {len(all_slots)} in parallel."
            )
        n = num_parallel_sims if num_parallel_sims > 0 else len(all_slots)
        self._slots: dict[str, tuple[str, CpuDomain]] = dict(list(all_slots.items())[:n])

    @property
    def slots(self) -> list[str]:
        return list(self._slots)

    def ranks_per_simulation(self, requested: int) -> int:
        cores = min(len(d.cpus) for _, d in self._slots.values())
        available = min(cores, self._cpus_per_node)
        return self._cap(requested, available, f"usable CPUs per {self.bind} slot")

    def _srun_prefix(self, slot: str, num_processes: int) -> str:
        node, domain = self._slots[slot]
        if self.bind == "node":
            cpu_bind = "cores"
        else:
            # Task i runs on the i-th core of the domain; memory follows by first-touch
            cpu_bind = "map_cpu:" + ",".join(map(str, domain.cpus[:num_processes]))
        cmd = (
            "PMIX_MCA_psec=native "
            f"srun --nodes=1 --nodelist={node} --ntasks={num_processes} --overlap "
            f"--cpu-bind={cpu_bind} --mpi=pmix"
        )
        if self.extra_srun_args:
            cmd += f" {self.extra_srun_args}"
        return cmd

    def command(self, slot: str, palace_executable: str, num_processes: int, config_name: str) -> str:
        palace_bin = resolve_palace_binary(palace_executable)
        return f"{self._srun_prefix(slot, num_processes)} {palace_bin} {config_name}"

    def check(self, num_processes: int, timeout: float = 120) -> None:
        """
        Runs one trivial job step in every slot concurrently, with the same `srun` options Palace
        will be launched with, to fail fast with srun's own diagnostics instead of hanging silently
        in the first simulation. Launching them concurrently reproduces the real workload shape, so
        a configuration that only works for a single step is caught too.

        Args:
            num_processes (int): MPI ranks per simulation, i.e. `--ntasks` of each step.
            timeout (float): Seconds to wait for all test steps before giving up.

        Raises:
            RuntimeError: If any test step fails or does not complete within `timeout`.
        """
        procs = {
            slot: subprocess.Popen(  # noqa: S602 - the srun prefix is built by this launcher
                f"{self._srun_prefix(slot, num_processes)} hostname",
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
            )
            for slot in self._slots
        }
        outputs: dict[str, str] = {}
        try:
            for slot, proc in procs.items():
                outputs[slot] = proc.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            stuck = [slot for slot, proc in procs.items() if proc.poll() is None]
            for proc in procs.values():
                proc.kill()
            raise RuntimeError(
                f"Test job steps in slots {stuck} did not complete within {timeout:.0f} s "
                f"(e.g. {self._srun_prefix(stuck[0], num_processes)} hostname).\n"
                "srun is waiting for resources inside the allocation or the PMIx handshake hangs. Check "
                "`squeue -s` for the step state and try the same srun command by hand in the allocation."
            ) from None

        failed = {slot: out for slot, out in outputs.items() if procs[slot].returncode != 0}
        if failed:
            details = "\n".join(f"[{slot}] {out.strip()}" for slot, out in failed.items())
            raise RuntimeError(
                f"Test job steps failed in slots {list(failed)}:\n{details}\n"
                f"(e.g. {self._srun_prefix(next(iter(failed)), num_processes)} hostname)"
            )
        logger.info(f"srun launch check passed in {len(self._slots)} slot(s).")

    def describe(self) -> str:
        nodes = sorted({node for node, _ in self._slots.values()})
        per_node = "one per node" if self.bind == "node" else f"one per {self.bind} domain"
        return f"{len(self._slots)} in parallel ({per_node}) on Slurm nodes {', '.join(nodes)}"
