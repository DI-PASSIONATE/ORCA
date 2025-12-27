from orca import ORCA
from orca.geometry.examples.transformer.tf_octa_c_ports import TransformerOcta
import skrf as rf
import matplotlib.pyplot as plt

geometry = TransformerOcta()

orca_instance = ORCA(geometry)

orca_instance.run(num_samples=1, cpu_cores=12)

# Load and plot the results
ntwk = rf.Network("/home/david/Documents/git/ORCA/palace_model/tf_octa_c_ports_0_data/output/tf_octa_c_ports_0/tf_octa_c_ports_0.s4p")
ntwk.plot_s_db()
plt.show()
