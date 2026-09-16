import orca
import numpy as np

from orca.geometry.presets.tf_octa_c_ports import TransformerOcta

def main():
    # Use predefined geometry from examples
    np.random.seed(40)
    try:
        import torch  # optional: only installed with ORCA's "train" extra
        torch.manual_seed(40)
    except ModuleNotFoundError:
        pass
    geometry = TransformerOcta(name="tf_octa_c_ports")

    orca_instance = orca.ORCA(
        [
            orca.GDSGenerator(num_samples=6000),
            orca.GDSConverter(),
            # launcher="slurm" runs the simulations as srun job steps on the nodes of this allocation
            # (#SBATCH --nodes in the job script). bind="numa" with num_parallel_sims=0 runs one
            # simulation per NUMA domain of every node (4 x 18 ranks on a Fritz node), which suits the
            # memory-bandwidth-bound solver best as long as one simulation fits into a domain's memory;
            # use bind="socket" (2 x 36) or "node" (1 x 72) otherwise. Leave launcher/num_parallel_sims
            # at their defaults to run sequentially on the current node, with or without Slurm.
            orca.PalaceSimulator(
                palace_executable="~/palace/build/bin/palace",
                launcher="slurm",
                num_parallel_sims=0,
                bind="numa",
            ),
            # orca.ModelTrainer(n_train_samples=1000),
            # orca.OnnxExporter(),
            # orca.ModelTester(),
        ]
    )

    # num_processes=None uses all cores of the current node for the GDS stages. Each Palace simulation
    # gets at most that many MPI ranks, capped to the cores of its slot (18 for a NUMA domain).
    orca_instance.run(geometry=geometry, num_processes=None, force_overwrite=True)


if __name__ == "__main__":
    main()
