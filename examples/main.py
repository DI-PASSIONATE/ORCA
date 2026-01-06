from orca import ORCA
import skrf as rf
import matplotlib.pyplot as plt

### Example of using a custom geometry
# geometry = MyCustomGeometry( # Python class that inherits from BaseGeometry
#     name = "my_geometry",
#     stackup_xml = "/path/to/stackup.xml", # XML file defining the physical layer stackup
#     simconfig_filename = "/path/to/simconfig.simcfg" # Simulation configuration file generated manually or with setupEM
# )

# Use predefined geometry from examples
from orca.geometry.examples.transformer.tf_octa_c_ports import TransformerOcta
geometry = TransformerOcta()

orca_instance = ORCA(geometry)

orca_instance.run(num_samples=1, cpu_cores=16)

# Load and plot the results
ntwk = rf.Network("/home/david/Documents/git/ORCA/palace_model/tf_octa_c_ports_0_data/output/tf_octa_c_ports_0/tf_octa_c_ports_0.s4p")
ntwk.plot_s_db()
plt.show()
