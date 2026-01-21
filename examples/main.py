from orca import ORCA
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from orca.utils.postprocessing import *

np.random.seed(11)
### Example of using a custom geometry
# geometry = MyCustomGeometry( # Python class that inherits from BaseGeometry
#     name = "my_geometry",
#     stackup_xml = "/path/to/stackup.xml", # XML file defining the physical layer stackup
#     simconfig_filename = "/path/to/simconfig.simcfg" # Simulation configuration file generated manually or with setupEM
# )


# Use predefined geometry from examples
from orca.geometry.presets.tf_octa_c_ports import TransformerOcta
geometry = TransformerOcta(n_samples=0)

#orca_instance = ORCA(geometry)

#orca_instance.run(cpu_cores=16, epochs=0, palace_executable="apptainer exec ~/Documents/git/palace/palace.sif palace")

# # Load and plot the results
ntwk = rf.Network("/home/david/Documents/git/ORCA/results/tf_octa_c_ports_6/tf_octa_c_ports_6_4_dc_deembedded.s6p")
# Plot S-parameters
plt.figure()
ntwk.plot_s_db(m=0, n=1)
N, ntwk = single_ended_to_mixed_mode(ntwk)
plot_diff_s_params_and_k(ntwk)
#print("Ground truth network S-parameters at position 10:")
#print(ntwk.s[10])

# plt.figure()
# ntwk2 = rf.Network("/home/david/Documents/git/ORCA/tf_octa_c_ports.s4p")
# ntwk2 = single_ended_to_mixed_mode(ntwk2)
# plot_diff_s_params_and_k(ntwk2)
# #print(ntwk2.s[10])
# plt.show()
plt.show()