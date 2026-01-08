from orca import ORCA
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt

### Example of using a custom geometry
# geometry = MyCustomGeometry( # Python class that inherits from BaseGeometry
#     name = "my_geometry",
#     stackup_xml = "/path/to/stackup.xml", # XML file defining the physical layer stackup
#     simconfig_filename = "/path/to/simconfig.simcfg" # Simulation configuration file generated manually or with setupEM
# )

# Use predefined geometry from examples
from orca.geometry.presets.tf_octa_c_ports import TransformerOcta
geometry = TransformerOcta()

orca_instance = ORCA(geometry)

orca_instance.run(num_samples=0, cpu_cores=16, palace_executable="apptainer exec ~/Documents/git/palace/palace.sif palace")

# # Load and plot the results
# ntwk = rf.Network("/home/david/Documents/git/ORCA/palace_model/tf_octa_c_ports_0_data/output/tf_octa_c_ports_0/tf_octa_c_ports_0.s4p")

# plt.figure()
# plt.subplot(2, 1, 1)
# plt.title("S-parameters that should be minimized")
# # Self-reflections
# ntwk.plot_s_db(0, 0) # S11 \approx S22
# ntwk.plot_s_db(1, 1) # S22 \approx S11

# ntwk.plot_s_db(2, 2) # S33 \approx S44
# ntwk.plot_s_db(3, 3) # S44 \approx S33

# # Same-octagon couplings
# ntwk.plot_s_db(0, 1) # S12 \approx S34
# ntwk.plot_s_db(2, 3) # S34 \approx S12

# # Add line at -7 dB
# plt.axhline(y=-7, color='r', linestyle='--', label='-7 dB Threshold')
# plt.legend()

# plt.subplot(2, 1, 2)
# plt.title("S-parameters that should be maximized")

# # Add line at -3 dB
# plt.axhline(y=-3, color='r', linestyle='--', label='-3 dB Threshold')
# plt.legend()

# ntwk.plot_s_db(0, 2) # S13
# ntwk.plot_s_db(0, 3) # S14
# ntwk.plot_s_db(1, 2) # S23
# ntwk.plot_s_db(1, 3) # S24

# # Symmetry
# # ntwk.plot_s_db(2, 0) # S31
# # ntwk.plot_s_db(3, 0) # S41
# # ntwk.plot_s_db(3, 1) # S42
# # ntwk.plot_s_db(2, 1) # S32
# plt.show()

# mm = ntwk.se2gmm(p=2)
# port_labels = ['d0', 'c0', 'd1', 'c1']

# # Plot S-parameters magnitude (in dB) for differential ports
# fig, ax = plt.subplots(1, 1, figsize=(8,6))
# for i, j in [(0,0), (2,2), (0,2), (2,0)]:
#     ax.plot(ntwk.f/1e9, 20*np.log10(abs(ntwk.s[:,i,j])), label=f'S({port_labels[i]},{port_labels[j]})')

# ax.set_xlabel('Frequency (GHz)')
# ax.set_ylabel('|S| (dB)')
# ax.set_title('Differential S-parameters')
# ax.legend()
# ax.grid(True)
# plt.show()