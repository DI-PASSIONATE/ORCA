#!/bin/bash -l
#
# Request one node per parallel Palace simulation (Fritz icelake nodes have 72 cores/2 sockets each).
# Must match the `num_parallel_palace_sims` passed to orca.PalaceSimulator(...) in main.py.
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=72
#SBATCH --time=01:00:00
#SBATCH --job-name=ORCA-FEM-SIM
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

###### START YOUR ACTUAL JOB SCRIPT BELOW THIS LINE ######

module load python
module load intel/2025.2.0
module load openmpi/5.0.8-intel2025.2.0

# Activate the conda environment
conda activate orca

# Run Python directly. With num_parallel_palace_sims > 1, PalaceSimulator uses `srun --exclusive
# --nodes=1 --ntasks=1` internally to pin each simulation to its own node within this allocation,
# and Palace manages mpirun across the 72 cores of that node.
python ./main.py