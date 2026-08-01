import orca
import numpy as np
import torch

from orca.geometry.presets.tf_octa_c_ports import TransformerOcta

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
np.random.seed(40)
torch.manual_seed(40)
geometry = TransformerOcta()

orca_instance = orca.ORCA(
    [
        orca.GDSGenerator(num_samples=40),
        orca.GDSConverter(),
        # num_parallel_palace_sims runs that many simulations in parallel, each pinned to its own
        # Slurm node via srun (requires #SBATCH --nodes=<num_parallel_palace_sims> in the job script).
        # Set to 1 (default) to run sequentially on the current node, with or without Slurm.
        orca.PalaceSimulator(palace_executable="~/palace/build/bin/palace", num_parallel_palace_sims=10),
        # orca.ModelTrainer(n_train_samples=1000),
        # orca.OnnxExporter(),
        # orca.ModelTester(),
    ]
)

orca_instance.run(geometry=geometry, num_processes=72, force_overwrite=True)