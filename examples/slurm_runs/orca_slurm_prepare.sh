#!/bin/bash -l
#
# ###### THIS SCRIPT IS EXPLICITLY FOR THE FRITZ CLUSTER @ NHR/FAU #######
# You may need to adjust the modules, nodes, tasks etc. for your specific cluster setup.
# ########################################################################
# First of the two simulation jobs: GDS generation, DRC and the GDS -> Palace conversion. These
# stages use the cores of one machine only, so they run on a single node instead of keeping the
# nodes of the simulation job idle. Submit both jobs with submit.sh, which starts
# orca_slurm_simulate.sh once this job has finished successfully.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=72
#SBATCH --partition=singlenode
#SBATCH --time=02:00:00
#SBATCH --job-name=ORCA-PREPARE
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

###### START YOUR ACTUAL JOB SCRIPT BELOW THIS LINE ######

module load python
module load intel/2025.2.0
module load openmpi/5.0.8-intel2025.2.0

# Activate the conda environment
conda activate orca

python ./prepare.py
