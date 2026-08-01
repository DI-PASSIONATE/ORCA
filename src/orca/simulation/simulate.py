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
    command_prefix: str = "",
) -> bool:
    """
    Runs Palace simulation for the given model.

    Args:
        data_dir (str): Directory where the Palace model is stored.
        config_name (str): Name of the Palace configuration to run.
        palace_executable (str): Path to the Palace executable (e.g. "apptainer exec ~/path/to/palace.sif palace").
        num_processes (int): Number of MPI processes to use for the simulation.
        touchstone_type (str): Type of Touchstone file to generate. One of "all", "normal", "deembedded", "dc", "dc_deembedded".
        command_prefix (str): Optional command prefix, prepended before the Palace executable
            (e.g. "srun --exclusive --nodes=1 --ntasks=1" to pin this simulation to a single
            Slurm node when running several simulations in parallel across a cluster).

    Returns:
        bool: True if simulation was successful, False otherwise.
    """
    inner_cmd = f"{palace_executable} -np {num_processes} {config_name}"

    if command_prefix:
        # Palace's own launcher script builds its MPI hostfile straight from $SLURM_JOB_NODELIST
        # (the whole job's node list), and Open MPI's Slurm integration also reads Slurm env vars
        # to size the resource allocation. Since command_prefix (an `srun --nodes=1 --ntasks=1 ...`
        # step) only owns a single task/node from Slurm's point of view, both of those would think
        # far fewer than `num_processes` slots are available. Stripping all SLURM_* variables
        # inside the step forces Palace/MPI to treat it as a plain local run on the one (exclusive)
        # node srun placed it on, using all of that node's cores.
        escaped_inner_cmd = inner_cmd.replace("'", "'\\''")
        cmd = f"{command_prefix} bash -c 'unset ${{!SLURM_@}}; exec {escaped_inner_cmd}'"
    else:
        cmd = inner_cmd

    # execute the command, hide output and save return code
    # cwd is used instead of os.chdir so this remains safe when run concurrently from multiple threads
    ret = subprocess.run(cmd, shell=True, cwd=sim_path) # USUALLY: SET capture_output=True to avoid palace output, only for debugging

    if ret.returncode != 0:
        logger.error(f"Palace simulation failed with exit code {ret.returncode}: {cmd}")
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
