import orca
import numpy as np
import torch

from orca.geometry.presets.tf_octa_c_ports import TransformerOcta

def main():
    # Use predefined geometry from examples
    np.random.seed(40)
    torch.manual_seed(40)
    geometry = TransformerOcta(name="tf_octa_c_ports")

    orca_instance = orca.ORCA(
        [
            orca.GDSGenerator(num_samples=430),
            orca.GDSConverter(),
            # num_parallel_palace_sims runs that many simulations in parallel, each pinned to its own
            # Slurm node via srun (requires #SBATCH --nodes=<num_parallel_palace_sims> in the job script).
            # Set to 1 (default) to run sequentially on the current node, with or without Slurm.
            orca.PalaceSimulator(palace_executable="~/palace/build/bin/palace", num_parallel_palace_sims=2),
            # orca.ModelTrainer(n_train_samples=1000),
            # orca.OnnxExporter(),
            # orca.ModelTester(),
        ]
    )

    orca_instance.run(geometry=geometry, num_processes=36, force_overwrite=True)


if __name__ == "__main__":
    main()