import os
import subprocess
import json

from orca.logger import logger
from orca.simulation.combine_snp_results import convert_to_touchstone


def run_palace(
    sim_path: str,
    data_dir: str,
    result_dir: str,
    config_name: str,
    palace_executable: str,
    touchstone_type: str,
    num_processes: int,
    use_srun: bool = False,
) -> bool:
    """
    Runs Palace simulation for the given model.

    Args:
        data_dir (str): Directory where the Palace model is stored.
        config_name (str): Name of the Palace configuration to run.
        palace_executable (str): Path to the Palace executable (e.g. "apptainer exec ~/path/to/palace.sif palace").
        num_processes (int): Number of MPI processes to use for the simulation.
        use_srun (bool): If True, pin this simulation to a single, dedicated Slurm node via
            `srun --exclusive --nodes=1 --ntasks=1`, so several simulations can run in parallel, each
            on its own node, within a multi-node Slurm allocation (e.g. `sbatch --nodes=N`). Palace's
            own launcher script reads `$SLURM_JOB_NODELIST`/`-np` directly and Open MPI's PMIx client
            connects to the srun step's PMIx server, both of which reflect the *whole job* allocation
            (or just 1 task for this step) rather than the single node reserved for this simulation.
            To avoid "not enough slots" errors, all `SLURM_*` env vars are unset for the inner process
            and the srun step is launched with `--mpi=none` so it doesn't set up PMIx at all, forcing
            `mpirun` (called internally by the Palace launcher) to fall back to plain local execution
            on the one exclusively-reserved node.
        touchstone_type (str): Type of Touchstone file to generate. One of "all", "normal", "deembedded", "dc", "dc_deembedded".

    Returns:
        bool: True if simulation was successful, False otherwise.
    """
    palace_cmd = f"{palace_executable} -np {num_processes} {config_name}"
    if use_srun:
        # Strip SLURM_* env vars so Palace/Open MPI treat this as a plain local run on the one node
        # reserved by srun, instead of trying to use the whole job's node list / task count.
        inner_cmd = f"unset ${{!SLURM_@}}; exec {palace_cmd}"
        cmd = (
            "srun --exclusive --nodes=1 --ntasks=1 --cpu-bind=none --mpi=none "
            f"bash -c '{inner_cmd}'"
        )
    else:
        cmd = palace_cmd

    # execute the command, hide output and save return code
    # cwd is passed explicitly (instead of os.chdir) so this is safe to call concurrently from
    # multiple threads when running several simulations in parallel across Slurm nodes.
    ret = subprocess.run(cmd, shell=True, cwd=sim_path) # USUALLY: SET capture_output=True to avoid palace output, only for debugging

    if ret.returncode != 0:
        logger.error(f"Palace simulation failed with return code {ret.returncode} for command: {cmd}")
        return False

    convert_to_touchstone(workdir=data_dir, output_dir=result_dir, touchstone_type=touchstone_type)
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
