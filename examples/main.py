from orca import ORCA
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
from orca.utils.postprocessing import *

PLOT = False

### Example of using a custom geometry
# geometry = MyCustomGeometry( # Python class that inherits from BaseGeometry
#     name = "my_geometry",
#     stackup_xml = "/path/to/stackup.xml", # XML file defining the physical layer stackup
#     simconfig_filename = "/path/to/simconfig.simcfg" # Simulation configuration file generated manually or with setupEM
# )


# Use predefined geometry from examples
np.random.seed(11)
from orca.geometry.presets.tf_octa_c_ports import TransformerOcta
geometry = TransformerOcta(n_samples=100, name="tf_octa_c_ports_testing")

orca_instance = ORCA(geometry)

orca_instance.run(cpu_cores=16, epochs=15, stages=["train", "evaluate"], palace_executable="apptainer exec ~/Documents/git/palace/palace.sif palace")


if PLOT:
    # # Load and plot the results
    ntwk = rf.Network("/home/david/Documents/git/ORCA/results/tf_octa_c_ports/tf_octa_c_ports_110_dc_deembedded.s6p")
    # Plot S-parameters
    plot_rfic_transformer_metrics(ntwk)
    #ntwk.plot_s_db()
    #N, ntwk = single_ended_to_mixed_mode(ntwk)
    print("Ground truth network S-parameters at position 10:")

    ntwk2 = rf.Network("/home/david/Documents/git/ORCA/tf_octa_c_ports.s6p")
    #N, ntwk2 = single_ended_to_mixed_mode(ntwk2)
    plot_rfic_transformer_metrics(ntwk2)
    plt.show()