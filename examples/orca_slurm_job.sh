#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --time=00:30:00
#SBATCH --job-name=ORCA-FEM-SIM
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

###### START YOUR ACTUAL JOB SCRIPT BELOW THIS LINE ######

module load user-spack
module load openmpi/5.0.8-gcc11.5.0
module load palace
module load python

# Activate the conda environment
conda activate orca

# Run Python directly (Palace internally manages mpirun across the 72 allocated Slurm slots)
python ./main.py