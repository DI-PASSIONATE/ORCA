import os
from orca.logger import logger
from orca.geometry.base_geometry import BaseGeometry
from orca.simulation.read_simconfig import read_simconfig
from orca.simulation.combine_snp_results import convert_to_touchstone
from gds2palace import gds_reader, stackup_reader, utilities, simulation_setup

def create_palace_model_from_gds(geometry: BaseGeometry, gds_filename: str, simconfig_filename: str) -> tuple[str, str, str]:
    """
    Uses gds2palace to create a Palace model from a GDS file and simulation configuration.
    The simconfig is a json and can either be created manually or by using setupEM GUI and saving the configuration.

    Based on: https://github.com/VolkerMuehlhaus/gds2palace_ihp_sg13g2/blob/main/workflow/palace_L2n0.py
    
    Args:
        gds_filename (str): Path to the GDS file.
        simconfig_filename (str): Path to the simulation configuration file (json).

    Returns:
        tuple[str, str]: Palace config name and data directory of the created Palace model.
    """
    # Path where the output simulation files will be stored -> use current working directory
    script_path = os.getcwd()
    # Model basename is taken from geometry name
    model_basename = geometry.name
    # set and create directory for simulation output
    sim_path = utilities.create_sim_path(script_path,model_basename)

    # Read the simconfig file
    simconfig = read_simconfig(simconfig_filename)

    # The settings dictionary contains all simulation parameters (e.g. frequency range, mesh settings...)
    settings = simconfig["saved_values"]

    # Add all ports from simconfig
    simulation_ports = simulation_setup.all_simulation_ports()
    for port in simconfig["ports"]:
        simulation_ports.add_port(
            simulation_setup.simulation_port(
                portnumber=port["portnumber"],
                voltage=port["voltage"],
                port_Z0=port["port_Z0"],
                source_layernum=port["source_layernum"],
                from_layername=port["from_layername"],
                to_layername=port["to_layername"],
                direction=port["direction"]
            )
        )

    materials_list, dielectrics_list, metals_list = stackup_reader.read_substrate(geometry.stackup_xml)
    layernumbers = metals_list.getlayernumbers()
    layernumbers.extend(simulation_ports.portlayers)

    # read geometries from GDSII
    allpolygons = gds_reader.read_gds(gds_filename, 
        layernumbers,
        purposelist=settings['purpose'], 
        metals_list=metals_list, 
        preprocess=settings['preprocess_gds'], 
        merge_polygon_size=settings['merge_polygon_size'],
        gds_boundary_layers=dielectrics_list.get_boundary_layers(),
        mirror=False, 
        offset_x=0, offset_y=0,
        layernumber_offset=0)
    
    settings['simulation_ports'] = simulation_ports
    settings['materials_list'] = materials_list
    settings['dielectrics_list'] = dielectrics_list
    settings['metals_list'] = metals_list
    settings['layernumbers'] = layernumbers
    settings['allpolygons'] = allpolygons
    settings['sim_path'] = sim_path
    settings['model_basename'] = model_basename
    settings['nogui'] = True  # create files without showing 3D model

    # list of ports that are excited (set voltage to zero in port excitation to skip an excitation!)
    excite_ports = simulation_ports.all_active_excitations()
    config_name, data_dir = simulation_setup.create_palace(excite_ports, settings)

    # for convenience, write run script to model directory
    utilities.create_run_script(settings['sim_path'])

    return config_name, sim_path, data_dir


def run_palace(sim_path: str, data_dir: str, result_dir: str, config_name: str, palace_executable: str, cpu_cores: int) -> None:
    """
    Runs Palace simulation for the given model.

    Args:
        data_dir (str): Directory where the Palace model is stored.
        config_name (str): Name of the Palace configuration to run.
        palace_executable (str): Path to the Palace executable (e.g. "apptainer exec ~/path/to/palace.sif palace").
        cpu_cores (int): Number of CPU cores to use for the simulation.
    """
    logger.info(f"Running Palace simulation for model: {config_name} in folder: {data_dir} using {cpu_cores} CPU cores...")
    os.chdir(sim_path)
    cmd = f"{palace_executable} -np {cpu_cores} {config_name}"
    # execute the command, hide output and save return code
    ret_code = os.system(cmd)
    if ret_code != 0:
        raise RuntimeError(f"Palace simulation for model: {config_name} failed with return code: {ret_code}")
    
    logger.info(f"Palace simulation for model: {config_name} completed.")
    logger.debug(f"Converting Palace CSV results at {sim_path}/{data_dir} to Touchstone format...")
    convert_to_touchstone(workdir=data_dir, output_dir=result_dir)