import glob
import os
import platform
import subprocess
import json

from orca.logger import logger
from orca.simulation.combine_snp_results import convert_to_touchstone


def _resolve_palace_binary(palace_executable: str) -> str:
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
            "use_srun requires palace_executable to be a direct filesystem path to the Palace wrapper script."
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Found multiple 'palace-*.bin' candidates under '{palace_dir}': {matches}. "
            "Could not unambiguously resolve the Palace binary for use_srun."
        )
    return matches[0]


def run_palace(
    sim_path: str,
    data_dir: str,
    result_dir: str,
    config_name: str,
    palace_executable: str,
    touchstone_type: str,
    num_processes: int,
    use_srun: bool = False,
    extra_srun_args: str = "",
) -> bool:
    """
    Runs Palace simulation for the given model.

    Args:
        data_dir (str): Directory where the Palace model is stored.
        config_name (str): Name of the Palace configuration to run.
        palace_executable (str): Path to the Palace executable (e.g. "apptainer exec ~/path/to/palace.sif palace").
        num_processes (int): Number of MPI processes to use for the simulation.
        use_srun (bool): If True, bypass the Palace wrapper script's own `mpirun` call and instead
            resolve the real `palace-*.bin` binary and launch it directly via
            `PMIX_MCA_psec=native srun --exclusive --nodes=1 --ntasks=num_processes --mpi=pmix`, so
            Slurm's own PMIx becomes the MPI launcher for this simulation, pinned to a single,
            dedicated node, without requiring a "munge" security plugin that may not be
            loadable/available. Running several simulations like this in parallel (e.g. via a thread
            pool) lets Slurm pack each one onto its own free node within a multi-node allocation (e.g.
            `sbatch --nodes=N`). Requires `palace_executable` to be a direct filesystem path to the
            wrapper script (see `_resolve_palace_binary`), not a container-wrapped compound command.
            Pass `extra_srun_args="--mpi=pmi2"` (or another value) to override the default PMI type if
            a given cluster needs a different one.
        extra_srun_args (str): Additional arguments appended to the `srun` command when `use_srun` is
            True (e.g. "--cpu-bind=cores"). Ignored otherwise.
        touchstone_type (str): Type of Touchstone file to generate. One of "all", "normal", "deembedded", "dc", "dc_deembedded".

    Returns:
        bool: True if simulation was successful, False otherwise.
    """
    if use_srun:
        palace_bin = _resolve_palace_binary(palace_executable)
        # --mpi=pmix is required so Open MPI can rendezvous through Slurm's PMIx server; without it,
        # Open MPI silently launches num_processes independent single-rank "singleton" processes
        # instead of one coordinated job, which then race each other (e.g. concurrently
        # creating/checking the output directory) and fail unpredictably.
        # PMIX_MCA_psec=native tells the PMIx client to skip the "munge" security component (which
        # may not be built/loadable in this environment, causing "component was not found" warnings
        # and potentially failing the handshake) and fall back to PMIx's always-available basic
        # UID/GID-based security check instead.
        cmd = (
            "PMIX_MCA_psec=native "
            f"srun --exclusive --nodes=1 --ntasks={num_processes} --cpu-bind=cores --mpi=pmix"
        )
        if extra_srun_args:
            cmd += f" {extra_srun_args}"
        cmd += f" {palace_bin} {config_name}"
    else:
        cmd = f"{palace_executable} -np {num_processes} {config_name}"

    # execute the command, hide output and save return code
    # cwd is passed explicitly (instead of os.chdir) so this is safe to call concurrently from
    # multiple threads when running several simulations in parallel across Slurm nodes.
    ret = subprocess.run(cmd, shell=True, cwd=sim_path, capture_output=True) # USUALLY: SET capture_output=True to avoid palace output, only for debugging

    if ret.returncode != 0:
        logger.error(f"Palace simulation failed with return code {ret.returncode} for command: {cmd}")
        return False

    # data_dir (from gds2palace) is relative to sim_path (matching Palace's own relative
    # Problem.Output config, written while its cwd was set to sim_path above), so resolve it to an
    # absolute path here since this process's own cwd was never changed (unlike the subprocess).
    convert_to_touchstone(
        workdir=os.path.join(sim_path, data_dir), output_dir=result_dir, touchstone_type=touchstone_type
    )
    return True


def read_simconfig(simconfig_filename: str) -> dict:
    """Reads simulation configuration from a file and returns it as a dictionary.

    Args:
        simconfig_filename (str): Path to the simulation configuration file.
    Returns:
        dict: A dictionary containing simulation configuration parameters.
    """
    with open(simconfig_filename, "r") as file:
        simconfig = json.load(file)

    # Add e9 suffix to frequency values if they are in GHz
    if "fstart" in simconfig["saved_values"]:
        fstart = simconfig["saved_values"]["fstart"]
        if fstart < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig["saved_values"]["fstart"] = fstart * 1e9
    if "fstop" in simconfig["saved_values"]:
        fstop = simconfig["saved_values"]["fstop"]
        if fstop < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig["saved_values"]["fstop"] = fstop * 1e9
    if "fstep" in simconfig["saved_values"]:
        fstep = simconfig["saved_values"]["fstep"]
        if fstep < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig["saved_values"]["fstep"] = fstep * 1e9
    if "fdump" in simconfig["saved_values"]:
        fdump = simconfig["saved_values"]["fdump"]
        if fdump < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig["saved_values"]["fdump"] = fdump * 1e9
    return simconfig
