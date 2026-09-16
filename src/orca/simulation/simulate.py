import json
import os
import subprocess
from typing import Any

from orca.logger import logger
from orca.simulation.combine_snp_results import convert_to_touchstone

PALACE_LOG_NAME = "palace.log"


def run_palace(
    sim_path: str,
    data_dir: str,
    result_dir: str,
    cmd: str,
    touchstone_type: str,
    save_log: bool = False,
) -> bool:
    """
    Runs one Palace simulation and converts its results to a Touchstone file.

    Args:
        sim_path (str): Simulation folder (working directory of the command).
        data_dir (str): Directory where Palace writes its results, relative to `sim_path`.
        result_dir (str): Directory to write the Touchstone file to.
        cmd (str): Shell command that runs the simulation, as built by a
            `SimulationLauncher` (e.g. `palace -np 16 config.json`).
        touchstone_type (str): Type of Touchstone file to generate. One of "all", "normal", "deembedded", "dc", "dc_deembedded".
        save_log (bool): Write the command and everything Palace/srun/MPI print to
            `<sim_path>/palace.log`. Off by default, since Palace prints a lot; then only stderr is
            kept (in memory) and shown if the simulation fails.

    Returns:
        bool: True if simulation was successful, False otherwise.
    """
    # cwd is passed explicitly (instead of os.chdir) so this is safe to call concurrently from
    # multiple threads when running several simulations in parallel.
    run_kwargs: dict[str, Any] = {"shell": True, "cwd": sim_path, "stdin": subprocess.DEVNULL}
    if save_log:
        log_path = os.path.join(sim_path, PALACE_LOG_NAME)
        with open(log_path, "w") as log_file:
            log_file.write(f"$ {cmd}\n")
            log_file.flush()
            ret = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False, **run_kwargs)
        if ret.returncode != 0:
            with open(log_path, errors="replace") as log_file:
                diagnostics = f"Last lines of {log_path}:\n" + "".join(log_file.readlines()[-15:])
    else:
        ret = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            check=False,
            **run_kwargs,
        )
        if ret.returncode != 0:
            diagnostics = "stderr:\n" + "".join(ret.stderr.splitlines(keepends=True)[-15:])

    if ret.returncode != 0:
        logger.error(
            f"Palace simulation failed with return code {ret.returncode} for command: {cmd}\n{diagnostics}"
        )
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
    with open(simconfig_filename) as file:
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
