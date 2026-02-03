import os
import subprocess
import json

from orca.logger import logger
from orca.geometry.base_geometry import BaseGeometry
from orca.simulation.combine_snp_results import convert_to_touchstone


def run_palace(sim_path: str, data_dir: str, result_dir: str, config_name: str, palace_executable: str, cpu_cores: int) -> bool:
    """
    Runs Palace simulation for the given model.

    Args:
        data_dir (str): Directory where the Palace model is stored.
        config_name (str): Name of the Palace configuration to run.
        palace_executable (str): Path to the Palace executable (e.g. "apptainer exec ~/path/to/palace.sif palace").
        cpu_cores (int): Number of CPU cores to use for the simulation.

    Returns:
        bool: True if simulation was successful, False otherwise.
    """
    os.chdir(sim_path)
    cmd = f"{palace_executable} -np {cpu_cores} {config_name}"

    # execute the command, hide output and save return code
    ret = subprocess.run(cmd, shell=True, capture_output=True)

    if ret.returncode != 0:
        logger.error(f"Palace simulation failed: {ret.stderr.decode('utf-8')}")
        return False
    
    convert_to_touchstone(workdir=data_dir, output_dir=result_dir)
    return True


def read_simconfig(simconfig_filename: str) -> dict:
    """Reads simulation configuration from a file and returns it as a dictionary.

    Args:
        simconfig_filename (str): Path to the simulation configuration file.
    Returns:
        dict: A dictionary containing simulation configuration parameters.
    """
    with open(simconfig_filename, 'r') as file:
        simconfig = json.load(file)

    # Add e9 suffix to frequency values if they are in GHz
    if 'fstart' in simconfig['saved_values']:
        fstart = simconfig['saved_values']['fstart']
        if fstart < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig['saved_values']['fstart'] = fstart * 1e9
    if 'fstop' in simconfig['saved_values']:
        fstop = simconfig['saved_values']['fstop']
        if fstop < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig['saved_values']['fstop'] = fstop * 1e9
    if 'fstep' in simconfig['saved_values']:
        fstep = simconfig['saved_values']['fstep']
        if fstep < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig['saved_values']['fstep'] = fstep * 1e9
    if "fdump" in simconfig['saved_values']:
        fdump = simconfig['saved_values']['fdump']
        if fdump < 1e6:  # assuming values less than 1 MHz are in GHz
            simconfig['saved_values']['fdump'] = fdump * 1e9
    return simconfig