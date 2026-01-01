# Model for IHP OpenPDK EM workflow created using setupEM
import os, sys, subprocess

from gds2palace import *

# get path for this simulation file
script_path = utilities.get_script_path(__file__)
# use script filename as model basename
model_basename = utilities.get_basename(__file__)
# set and create directory for simulation output
sim_path = utilities.create_sim_path (script_path,model_basename)

# ========================= workflow settings ==========================

# ===================== input files and settings =======================
settings={}
settings['GdsFile'] = '/home/david/Documents/git/ORCA/tf_octa_c.gds'
settings['SubstrateFile'] = '/home/david/Documents/git/ORCA/.venv11/lib/python3.11/site-packages/setupEM/data/SG13G2_nosub.xml'
settings['preprocess_gds'] = True
settings['merge_polygon_size'] = 0.5
settings['purpose'] = [0]
settings['fstart'] = 0.0e9
settings['fstop'] = 50.0e9
settings['refined_cellsize'] = 5.0
settings['order'] = 2
settings['cells_per_wavelength'] = 10.0
settings['meshsize_max'] = 100.0
settings['adaptive_mesh_iterations'] = 0
settings['boundary'] = ['ABC', 'ABC', 'ABC', 'ABC', 'ABC', 'ABC']
settings['margin'] = 200.0
settings['air_around'] = 200.0
settings['ELMER_MPI_THREADS'] = 4
settings['preview_only'] = True
settings['no_preview'] = False

# ===================== port definitions =======================
simulation_ports = simulation_setup.all_simulation_ports()

# ================= read stackup and geometries =================
materials_list, dielectrics_list, metals_list = stackup_reader.read_substrate (settings['SubstrateFile'])
layernumbers = metals_list.getlayernumbers()
layernumbers.extend(simulation_ports.portlayers)

# read geometries from GDSII
allpolygons = gds_reader.read_gds(settings['GdsFile'], 
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

# list of ports that are excited (set voltage to zero in port excitation to skip an excitation!)
excite_ports = simulation_ports.all_active_excitations()
config_name, data_dir = simulation_setup.create_palace (excite_ports, settings)

# for convenience, write run script to model directory
utilities.create_run_script(settings['sim_path'])