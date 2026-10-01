#!/bin/bash -l
#
# ###### THIS SCRIPT IS EXPLICITLY FOR THE TINYGPU CLUSTER @ NHR/FAU #######
# You may need to adjust the partition, GPU type, modules etc. for your specific cluster setup.
# ##########################################################################
# Trains a model on the results of a finished simulation job (orca_slurm_simulate.sh on Fritz).
# $HOME and $WORK are shared between the NHR@FAU clusters, so submit from the folder simulate.py
# ran in, from the TinyGPU frontend (tinyx.nhr.fau.de):
#     sbatch.tinygpu orca_slurm_train.sh
#
# ORCA trains one model at a time on one GPU, so more GPUs would sit idle. The training data of
# the presets fits into a few GB of GPU memory, so every TinyGPU type works: the A100 is the
# fastest, the other partitions often have shorter queues, e.g.
#     --gres=gpu:rtx3080:1 --partition=rtx3080   or   --gres=gpu:v100:1 --partition=v100
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=ORCA-TRAIN
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

###### START YOUR ACTUAL JOB SCRIPT BELOW THIS LINE ######

module load python

# A separate environment with a CUDA build of PyTorch (see HPC_INSTALL.md); the Fritz
# environment "orca" is simulation-only.
conda activate orca-gpu

# Fail now rather than train on the CPU for 24 hours
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 'PyTorch cannot see a GPU: install a CUDA build of torch into orca-gpu (see HPC_INSTALL.md).')" || exit 1
nvidia-smi

# PyTorch otherwise starts one CPU thread per core of the whole node, not of this job's share
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

python ./train.py
