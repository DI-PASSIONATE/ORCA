#!/bin/bash -l
#
# ###### THIS SCRIPT IS EXPLICITLY FOR THE FRITZ CLUSTER @ NHR/FAU #######
# You may need to adjust the modules, nodes, tasks etc. for your specific cluster setup.
# ########################################################################
# Second of the two simulation jobs: the Palace simulations of the models orca_slurm_prepare.sh
# wrote. Submit both jobs with submit.sh, which holds this job until the first one has finished
# successfully. On its own (e.g. to resume after a time limit) submit it from the same folder:
#     sbatch orca_slurm_simulate.sh
#
# Palace simulations run on the allocated nodes (Fritz icelake: 2 sockets x 36 cores, 4 NUMA domains).
# With launcher="slurm", num_parallel_sims=0, bind="numa" in simulate.py, ORCA runs 4 simulations
# per node on every node of this allocation, so the node count only has to be set here (2 to 64
# nodes in the multinode partition).
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=72
#SBATCH --partition=multinode
#SBATCH --time=12:00:00
#SBATCH --job-name=ORCA-FEM-SIM
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

###### START YOUR ACTUAL JOB SCRIPT BELOW THIS LINE ######

module load python
module load intel/2025.2.0
module load openmpi/5.0.8-intel2025.2.0

# Activate the conda environment
conda activate orca

# Run Python directly. With launcher="slurm", PalaceSimulator bypasses the Palace
# wrapper's own mpirun call and launches the resolved palace-*.bin binary directly via
# `srun --nodes=1 --nodelist=<node> --ntasks=<cores per slot> --cpu-bind=map_cpu:<cores>`, one
# simulation per slot (node / socket / NUMA domain) of this allocation. Each simulation writes its
# srun/Palace output to <sim dir>/palace.log if save_log=True is passed to PalaceSimulator.
python ./simulate.py
