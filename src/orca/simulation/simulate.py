import json
import os
import re
import subprocess
import time
from collections.abc import Callable
from typing import Any

from orca.logger import logger
from orca.simulation.combine_snp_results import convert_to_touchstone

PALACE_LOG_NAME = "palace.log"
# Seconds to wait before each retry of a launch the scheduler refused (see run_palace).
LAUNCH_RETRY_DELAYS = (10.0, 30.0, 90.0, 300.0)
# Printed once per rank by PMIx when a failed job step is torn down; carries no information.
_TEARDOWN_NOISE = re.compile(r"pmix_ptl_base: send_msg: write failed")


def run_palace(
    sim_path: str,
    data_dir: str,
    result_dir: str,
    cmd: str,
    touchstone_type: str,
    save_log: bool = False,
    launch_failed: Callable[[str], bool] | None = None,
    retry_delays: tuple[float, ...] = LAUNCH_RETRY_DELAYS,
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
        launch_failed (Callable[[str], bool] | None): Tells from the output of a failed command
            whether the launcher refused to start it (see `SimulationLauncher.launch_failed`).
            Such a launch is retried after each of `retry_delays` instead of failing the
            simulation. None never retries.
        retry_delays (tuple[float, ...]): Seconds to wait before each retry of a refused launch.

    Returns:
        bool: True if simulation was successful, False otherwise.
    """
    for attempt in range(len(retry_delays) + 1):
        returncode, diagnostics = _run_command(sim_path, cmd, save_log)
        if returncode == 0:
            break
        if attempt == len(retry_delays) or launch_failed is None or not launch_failed(diagnostics):
            logger.error(
                f"Palace simulation failed with return code {returncode} for command: {cmd}\n"
                f"{diagnostics}"
            )
            return False
        # The scheduler refused the launch, e.g. a transient Slurm step-creation error. Waiting in
        # this worker keeps the slot busy, so the queue does not drain through the same refusal.
        delay = retry_delays[attempt]
        logger.warning(
            f"Launch refused in {sim_path} ({diagnostics.strip().splitlines()[-1]}); "
            f"retrying in {delay:g} s ({attempt + 1}/{len(retry_delays)})."
        )
        time.sleep(delay)

    # data_dir (from gds2palace) is relative to sim_path (matching Palace's own relative
    # Problem.Output config, written while its cwd was set to sim_path in _run_command), so resolve it to an
    # absolute path here since this process's own cwd was never changed (unlike the subprocess).
    convert_to_touchstone(
        workdir=os.path.join(sim_path, data_dir), output_dir=result_dir, touchstone_type=touchstone_type
    )
    return True


def _run_command(sim_path: str, cmd: str, save_log: bool) -> tuple[int, str]:
    """
    Runs `cmd` in `sim_path` and returns its return code and, if it failed, the last lines of its
    output (without MPI teardown noise) for the log.
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
        if ret.returncode == 0:
            return 0, ""
        with open(log_path, errors="replace") as log_file:
            return ret.returncode, f"Last lines of {log_path}:\n" + _tail(log_file.read())

    ret = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        check=False,
        **run_kwargs,
    )
    if ret.returncode == 0:
        return 0, ""
    return ret.returncode, "stderr:\n" + _tail(ret.stderr)


def _tail(output: str, n_lines: int = 15) -> str:
    """
    Last `n_lines` lines of `output`, skipping blank lines and the PMIx messages every rank prints
    when a failed step is torn down. Those used to fill the tail and hide Palace's own error.
    """
    lines = [
        line
        for line in output.splitlines(keepends=True)
        if line.strip() and not _TEARDOWN_NOISE.search(line)
    ]
    return "".join(lines[-n_lines:])


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
