#!/bin/bash -l
#
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

# Run Python directly (no srun prefix here). Palace internally manages mpirun across the 72
# allocated Slurm slots of a node; ORCA itself uses srun internally per-simulation only when
# num_parallel_palace_sims > 1.
python ./main.py