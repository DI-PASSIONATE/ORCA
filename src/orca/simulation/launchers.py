"""
Launchers decide *where* Palace simulations run and how many can run at once.

A launcher exposes a fixed set of *slots*. Each slot can run one simulation at a time; the
simulation stage keeps one worker per slot and asks the launcher for the shell command that runs
a given Palace config in a given slot. A slot is a whole machine, a socket or a NUMA domain
(`bind`), on this machine for `LocalLauncher` below or on each node of a Slurm allocation for
`orca.simulation.slurm.SlurmLauncher`.

Binding a simulation to a socket or NUMA domain matters for memory-bandwidth-bound solvers like
Palace: several smaller simulations, each confined to its own domain, keep all memory traffic
local and avoid the poor parallel efficiency of one small mesh spread over a whole node. Only the
CPUs are pinned; memory then lands in the same domain through the kernel's first-touch policy and
spills over to a neighbouring domain (slower, but not fatal) if a simulation needs more than the
domain has.
"""

import glob
import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass

from orca.logger import logger

BIND_CHOICES = ("node", "socket", "numa")


@dataclass(frozen=True)
class CpuDomain:
    """A set of CPUs a simulation can be bound to (a whole node, one socket or one NUMA domain)."""

    name: str
    cpus: tuple[int, ...]


def _parse_cpulist(cpulist: str) -> list[int]:
    """Parses a sysfs CPU list like "0-7,16-23" into CPU ids."""
    cpus: list[int] = []
    for part in cpulist.strip().split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def cpu_domains(
    bind: str, hyperthreads: bool = False, sysfs_root: str = "/sys/devices/system"
) -> list[CpuDomain]:
    """
    Reads the CPU topology of this machine from sysfs and groups the CPUs by `bind`.

    By default only one hardware thread per core is returned, so `len(domain.cpus)` is the number
    of cores an MPI job can use in that domain without two ranks sharing a core's execution units
    and caches. Memory-bandwidth-bound solvers like Palace gain little or nothing from SMT, so
    `hyperthreads=True` is mainly there to measure that on a given machine.

    Args:
        bind (str): "node" (one domain with every core), "socket" (one per physical package) or
            "numa" (one per NUMA node).
        hyperthreads (bool): Also return the sibling hardware threads of every core.
        sysfs_root (str): The sysfs directory holding the `cpu` and `node` topology entries.

    Returns:
        list[CpuDomain]: The domains in id order, named "node", "socket<N>" or "numa<N>".
    """
    if bind not in BIND_CHOICES:
        raise ValueError(f"bind must be one of {BIND_CHOICES}, got {bind!r}.")

    # First hardware thread of every core, with its socket id
    core_cpus: list[int] = []
    socket_of: dict[int, int] = {}
    for cpu_dir in glob.glob(os.path.join(sysfs_root, "cpu", "cpu[0-9]*")):
        cpu = int(os.path.basename(cpu_dir)[len("cpu"):])
        topology = os.path.join(cpu_dir, "topology")
        if not os.path.isdir(topology):
            continue  # offline CPU
        if not hyperthreads:
            with open(os.path.join(topology, "thread_siblings_list")) as f:
                if _parse_cpulist(f.read())[0] != cpu:
                    continue  # hyperthread sibling of a core already counted
        with open(os.path.join(topology, "physical_package_id")) as f:
            socket_of[cpu] = int(f.read())
        core_cpus.append(cpu)
    core_cpus.sort()
    if not core_cpus:
        core_cpus = sorted(os.sched_getaffinity(0))
        socket_of = dict.fromkeys(core_cpus, 0)

    if bind == "node":
        return [CpuDomain("node", tuple(core_cpus))]

    groups: dict[int, list[int]] = {}
    if bind == "socket":
        for cpu in core_cpus:
            groups.setdefault(socket_of[cpu], []).append(cpu)
    else:
        cores = set(core_cpus)
        for node_dir in glob.glob(os.path.join(sysfs_root, "node", "node[0-9]*")):
            node_id = int(os.path.basename(node_dir)[len("node"):])
            with open(os.path.join(node_dir, "cpulist")) as f:
                cpus = [cpu for cpu in _parse_cpulist(f.read()) if cpu in cores]
            if cpus:
                groups[node_id] = cpus
        if not groups:
            groups[0] = core_cpus
    return [CpuDomain(f"{bind}{i}", tuple(groups[i])) for i in sorted(groups)]


class SimulationLauncher(ABC):
    """Base class for the different ways of launching Palace simulations."""

    @property
    @abstractmethod
    def slots(self) -> list[str]:
        """Names of the slots that can each run one simulation at a time."""

    @abstractmethod
    def ranks_per_simulation(self, requested: int) -> int:
        """
        Number of MPI ranks each simulation gets, given the requested `num_processes`. Launchers
        cap this to what a slot actually provides and log a warning if they do.
        """

    @abstractmethod
    def command(self, slot: str, palace_executable: str, num_processes: int, config_name: str) -> str:
        """Shell command that runs `config_name` with `num_processes` MPI ranks in `slot`."""

    def check(self, num_processes: int) -> None:
        """
        Pre-flight check run once before the first simulation, so a broken launch setup fails
        fast with a clear message instead of a hanging or failing simulation. No-op by default.
        """

    def describe(self) -> str:
        """One-line description of the slots for the log."""
        return f"{len(self.slots)} in parallel ({', '.join(self.slots)})"

    @staticmethod
    def _cap(requested: int, available: int, what: str) -> int:
        """Caps the requested ranks to what is available, with a warning."""
        if requested > available:
            logger.warning(
                f"num_processes={requested} exceeds the {available} {what}; "
                f"using {available} MPI processes per simulation."
            )
            return max(available, 1)
        return requested


class LocalLauncher(SimulationLauncher):
    """
    Runs simulations on the current machine through the Palace wrapper (`palace -np N config`),
    optionally several at once, each pinned to its own socket or NUMA domain with `numactl`.
    """

    def __init__(self, num_parallel_sims: int = 1, bind: str = "node", hyperthreads: bool = False):
        """
        Args:
            num_parallel_sims (int): Simulations to run at once. 0 means one per domain selected by
                `bind` (i.e. 1 for "node"). Default is 1 (sequential).
            bind (str): "node" runs the simulations unpinned, sharing all cores. "socket" or "numa"
                pins each simulation to its own socket / NUMA domain (`numactl --physcpubind`), so
                parallel simulations do not compete for the same cores and memory controllers; at
                most one simulation per domain then runs at a time. Requires `numactl`.
            hyperthreads (bool): Count hardware threads instead of physical cores, allowing one
                MPI rank per thread. Usually not faster for Palace; see `cpu_domains`.

        Raises:
            RuntimeError: If binding is requested but `numactl` is not available.
        """
        if bind != "node" and shutil.which("numactl") is None:
            raise RuntimeError(f"bind={bind!r} requires `numactl` to be installed and on PATH.")
        self.bind = bind
        self._unit = "hardware threads" if hyperthreads else "physical cores"

        # Only cores this process may use (e.g. inside a container or cpuset)
        allowed = os.sched_getaffinity(0)
        domains = [
            CpuDomain(d.name, tuple(cpu for cpu in d.cpus if cpu in allowed))
            for d in cpu_domains(bind, hyperthreads)
        ]
        domains = [d for d in domains if d.cpus]

        if bind == "node":
            # All simulations share the whole machine, unpinned
            n = max(num_parallel_sims, 1)
            self._slots = {f"slot{i}": domains[0] for i in range(n)}
        else:
            n = num_parallel_sims if num_parallel_sims > 0 else len(domains)
            if n > len(domains):
                logger.warning(
                    f"num_parallel_sims={n} exceeds the {len(domains)} {bind} domain(s) of this "
                    f"machine; running {len(domains)} in parallel."
                )
            self._slots = {d.name: d for d in domains[:n]}

    @property
    def slots(self) -> list[str]:
        return list(self._slots)

    def ranks_per_simulation(self, requested: int) -> int:
        if self.bind == "node":
            available = len(next(iter(self._slots.values())).cpus) // len(self._slots)
            return self._cap(requested, available, f"{self._unit} available per parallel simulation")
        available = min(len(d.cpus) for d in self._slots.values())
        return self._cap(requested, available, f"{self._unit} per {self.bind} domain")

    def command(self, slot: str, palace_executable: str, num_processes: int, config_name: str) -> str:
        cmd = f"{palace_executable} -np {num_processes} {config_name}"
        if self.bind != "node":
            # numactl restricts the CPU affinity of the wrapper script; the MPI ranks it launches
            # inherit it and Open MPI binds within the allowed cpuset.
            cpus = ",".join(map(str, self._slots[slot].cpus))
            cmd = f"numactl --physcpubind={cpus} {cmd}"
        return cmd

    def describe(self) -> str:
        if self.bind == "node":
            return f"{len(self._slots)} in parallel on this machine"
        return f"{len(self._slots)} in parallel, one per {self.bind} domain ({', '.join(self._slots)})"
