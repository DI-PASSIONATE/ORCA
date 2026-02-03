from gds2palace import gds_reader, stackup_reader, utilities, simulation_setup
import os
from contextlib import redirect_stdout, ExitStack

import gmsh
from typing import Any
from orca.simulation.read_simconfig import read_simconfig

def create_palace_model_from_gds(
        geometry_name: str, 
        params: dict[str, Any],
        output_dir: str,
        gds_filename: str, 
        stackup_xml: str,
        simconfig_filename: str, 
        show_mesh_results: bool = False
    ) -> tuple[str, dict[str, Any], str, str, str]:
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
    # ExitStack is used to suppress stdout output in the ProcessPoolExecutor workers to avoid cluttering the console
    with ExitStack() as stack:
        if not show_mesh_results:
            null_file = stack.enter_context(open(os.devnull, "w"))
            stack.enter_context(redirect_stdout(null_file))

        model_basename = utilities.get_basename(gds_filename)

        # set and create directory for simulation output
        sim_path = utilities.create_sim_path(script_path=output_dir, model_basename=model_basename, dirname="palace_sims")

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

        materials_list, dielectrics_list, metals_list = stackup_reader.read_substrate(stackup_xml)
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
            layernumber_offset=0
        )
        
        settings['simulation_ports'] = simulation_ports
        settings['materials_list'] = materials_list
        settings['dielectrics_list'] = dielectrics_list
        settings['metals_list'] = metals_list
        settings['layernumbers'] = layernumbers
        settings['allpolygons'] = allpolygons
        settings['sim_path'] = sim_path
        settings['model_basename'] = model_basename
        settings['no_gui'] = not show_mesh_results  # create files without showing 3D model

        # list of ports that are excited (set voltage to zero in port excitation to skip an excitation!)
        excite_ports = simulation_ports.all_active_excitations()
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        config_name, data_dir = simulation_setup.create_palace(excite_ports, settings)

        # for convenience, write run script to model directory
        utilities.create_run_script(settings['sim_path'])

        return geometry_name, params, config_name, sim_path, data_dir
