#!/usr/bin/env bash
#
# Automates the steps from HPC_INSTALL.md to install AWS Palace and ORCA
# on the Fritz HPC cluster.
#
# IMPORTANT: This script must be run on an already allocated compute node,
# e.g. after:
#   salloc -N 1 --partition=singlenode --time=01:00:00
#
# Usage: bash hpc_install.sh

set -euo pipefail

# --- Sanity check: make sure we're running on an allocated compute node ---
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "Error: no SLURM_JOB_ID found. Allocate a node first, e.g.:" >&2
  echo "  salloc -N 1 --partition=singlenode --time=01:00:00" >&2
  exit 1
fi

# --- Make the internet accessible from the allocated node ---
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80

# --- Load compilers and MPI modules ---
# (module load can reference unset variables internally, so relax -u here)
set +u
module load cmake
module load intel/2025.2.0
module load openmpi/5.0.8-intel2025.2.0
set -u

# --- Set the MKLROOT environment variable ---
export MKLROOT=/apps/spack/1.0.2/opt/linux-almalinux9-icelake/none-none/intel-oneapi-mkl-2024.2.2-bdh2w4w5yar6xnpkwig2qb6i3i6vbxuz/mkl/2024.2

# --- Clone and build Palace ---
if [ ! -d palace ]; then
  git clone https://github.com/awslabs/palace.git
fi
cd palace
mkdir -p build && cd build

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_Fortran_COMPILER=ifx \
  -DCMAKE_C_FLAGS="-O3 -xHost" \
  -DCMAKE_CXX_FLAGS="-O3 -xHost" \
  -DCMAKE_Fortran_FLAGS="-O3 -xHost" \
  -DPALACE_WITH_LIBXSMM=ON \
  -DPALACE_WITH_MUMPS=ON \
  ..
make -j "$(nproc)"

# --- Check that Palace is working ---
palace --version

cd ../..

# --- Clone ORCA ---
if [ ! -d ORCA ]; then
  git clone https://github.com/DI-PASSIONATE/ORCA.git
fi

# --- First time conda setup ---
if [ ! -f ~/.bash_profile ]; then
  echo "if [ -f ~/.bashrc ]; then . ~/.bashrc; fi" > ~/.bash_profile
fi
set +u
module add python
set -u
conda config --add pkgs_dirs "$WORK/software/private/conda/pkgs"
conda config --add envs_dirs "$WORK/software/private/conda/envs"

# --- Create a conda environment for ORCA and install dependencies ---
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -qE '^orca[[:space:]]'; then
  conda create -y -n orca python=3.12
fi
conda activate orca
conda install -y -c conda-forge mesalib libglu

# --- Install ORCA dependencies ---
pip install -e ./ORCA/

# --- Install PyTorch (CPU version) ---
# Replace with the command from https://pytorch.org/get-started/locally/
# if you need a different (e.g. GPU) build.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# --- Make sure everything is set up correctly ---
which palace
which python
python --version
