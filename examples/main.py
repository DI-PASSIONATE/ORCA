from orca import ORCA
import skrf as rf
import numpy as np
import matplotlib.pyplot as plt
import torch

from orca.utils.postprocessing import (
    plot_rfic_transformer_metrics,
)
from orca.geometry.presets.tf_octa_c_ports import TransformerOcta
import orca

PLOT = False

### Example of using a custom geometry
# geometry = MyCustomGeometry( # Python class that inherits from BaseGeometry
#     name = "my_geometry",
#     stackup_xml = "/path/to/stackup.xml", # XML file defining the physical layer stackup
#     simconfig_filename = "/path/to/simconfig.simcfg" # Simulation configuration file generated manually or with setupEM
# )

# hyperparameters = {'learning_rate': 0.0008166998266605425, 'batch_size': 128, 'epochs': 10, 'num_layers': 4, 'hidden_size': 512, 'activation_function': 'GELU'}
hyperparameters = {
    "learning_rate": 0.0005,
    "batch_size": 256,
    "epochs": 15,
    "num_layers": 5,
    "hidden_size": 800,
    "activation_function": "GELU"
}

# Use predefined geometry from examples
np.random.seed(11)
torch.manual_seed(11)
geometry = TransformerOcta()

orca_instance = ORCA(
    [
        # orca.GDSGenerator(num_samples=1000),
        # orca.GDSConverter(),
        # orca.PalaceSimulator(palace_executable="apptainer exec ~/Documents/git/palace/palace.sif palace"),
        orca.ModelTrainer(n_train_samples=1000),
        orca.OnnxExporter(),
        orca.ModelTester(),
    ]
)

orca_instance.run(geometry=geometry, cpu_cores=16)