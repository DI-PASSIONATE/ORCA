#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=36
#SBATCH --time=01:00:00
#SBATCH --job-name=ORCA-FEM-SIM
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

###### START YOUR ACTUAL JOB SCRIPT BELOW THIS LINE ######

module load python
module load intel/2025.2.0
module load openmpi/5.0.8-intel2025.2.0

# Set the number of threads for OpenMP
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# for Slurm version >22.05: cpus-per-task has to be set again for srun
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Pin OpenMP threads to cores and close them together to avoid thread migration
export OMP_PLACES=cores
export OMP_PROC_BIND=close


# Activate the conda environment
conda activate orca

# Run Python directly (Palace internally manages mpirun across the 72 allocated Slurm slots)
python ./main.py