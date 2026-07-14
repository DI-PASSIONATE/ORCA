#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --time=01:00:00
#SBATCH --job-name=ORCA-FEM-SIM
#SBATCH --export=NONE
#
# first non-empty non-comment line ends SBATCH options
# nodes: number of nodes to use
# ntasks-per-node: number of MPI processes to launch per node, should be equal to the number of physical cores on the node (72 for Fritz Ice Lake nodes)

unset SLURM_EXPORT_ENV

###### START YOUR ACTUAL JOB SCRIPT BELOW THIS LINE ######

module load openmpi
module load palace
module load python

# Activate the conda environment
conda activate orca

# Calls palace internally, which calls mpirun. Make sure to use srun to launch the job, otherwise it will not work properly.
srun python ./ORCA/examples/main.py