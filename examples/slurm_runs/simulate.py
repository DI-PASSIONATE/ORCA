import orca

# from orca.geometry.presets import TransformerOcta
from orca.geometry.presets import InductorOcta


def main():
    # Same geometry (and name) as in prepare.py, so its Palace models are found
    # geometry = TransformerOcta(name="tf_octa_c_ports")
    geometry = InductorOcta(name="inductor_octa")

    orca_instance = orca.ORCA(
        [
            # Simulates the models prepare.py wrote to output/<geometry name>/palace_sims.
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
        ]
    )

    # Each Palace simulation gets at most num_processes MPI ranks (None: all cores of the node this
    # script runs on), capped to the cores of its slot (18 for a NUMA domain). If the job hits its
    # time limit, submit it again (sbatch orca_slurm_simulate.sh): results/<name>.csv lists each
    # completed simulation, and only the rest are run.
    orca_instance.run(geometry=geometry, num_processes=None, force_overwrite=True)


if __name__ == "__main__":
    main()
