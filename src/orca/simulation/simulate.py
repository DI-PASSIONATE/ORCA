import os
import subprocess
from orca.logger import logger
from orca.geometry.base_geometry import BaseGeometry
from orca.simulation.read_simconfig import read_simconfig
from orca.simulation.combine_snp_results import convert_to_touchstone


def run_palace(sim_path: str, data_dir: str, result_dir: str, config_name: str, palace_executable: str, cpu_cores: int) -> None:
    """
    Runs Palace simulation for the given model.

    Args:
        data_dir (str): Directory where the Palace model is stored.
        config_name (str): Name of the Palace configuration to run.
        palace_executable (str): Path to the Palace executable (e.g. "apptainer exec ~/path/to/palace.sif palace").
        cpu_cores (int): Number of CPU cores to use for the simulation.
    """
    os.chdir(sim_path)
    cmd = f"{palace_executable} -np {cpu_cores} {config_name}"

    # execute the command, hide output and save return code
    ret = subprocess.run(cmd, shell=True, capture_output=True)

    if ret.returncode != 0:
        logger.error(f"Palace simulation failed: {ret.stderr.decode('utf-8')}")
        return
    
    convert_to_touchstone(workdir=data_dir, output_dir=result_dir)
    