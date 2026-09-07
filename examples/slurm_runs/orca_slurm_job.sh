#!/bin/bash -l
#
# Request one node per parallel Palace simulation (Fritz icelake nodes have 72 cores/2 sockets each).
# Must match the `num_parallel_palace_sims` passed to orca.PalaceSimulator(...) in main.py.
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=72
#SBATCH --time=24:00:00
#SBATCH --job-name=ORCA-FEM-SIM
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

###### START YOUR ACTUAL JOB SCRIPT BELOW THIS LINE ######

module load python
module load intel/2025.2.0
module load openmpi/5.0.8-intel2025.2.0

# Activate the conda environment
conda activate orca

# Run Python directly. With num_parallel_palace_sims > 1, PalaceSimulator bypasses the Palace
# wrapper's own mpirun call and launches the resolved palace-*.bin binary directly via
# `srun --exclusive --nodes=1 --ntasks=<num_processes>`, so Slurm packs each simulation onto its
# own node within this allocation.
python ./main.py