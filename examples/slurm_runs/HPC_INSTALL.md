## How to install ORCA on the Fritz HPC Cluster
### Install AWS Palace
1. Connect to the frontend node of Fritz using SSH:

```sh
ssh <username>@fritz.nhr.fau.de
```

2. Allocate an interactive node and wait until you are granted access

```sh
salloc -N 1 --partition=singlenode --time=01:00:00
```

Now you have two options: Either perform the following steps by hand on the allocated node, or try out the hpc_install.sh script on the allocated node.

3. Make the internet accessible from the allocated node by running:

```sh
export http_proxy=http://proxy.nhr.fau.de:80 && export https_proxy=http://proxy.nhr.fau.de:80
```

4. Load compilers and MPI modules

```sh
module load cmake
module load intel/2025.2.0
module load openmpi/5.0.8-intel2025.2.0
```

5. Set the MKLROOT environment variable (Intel Math Kernel Library). Palace uses MKL as a dependency for some of its linear algebra operations (Alternative to OpenBLAS and LAPACK libraries optimized for Intel CPUs).

```sh
export MKLROOT=/apps/spack/1.0.2/opt/linux-almalinux9-icelake/none-none/intel-oneapi-mkl-2024.2.2-bdh2w4w5yar6xnpkwig2qb6i3i6vbxuz/mkl/2024.2
```

6. Clone the Palace repository from GitHub

```sh
git clone https://github.com/awslabs/palace.git
cd palace
mkdir build && cd build
```

7. Build Palace using CMake and Make

You have more configuration options available (see https://awslabs.github.io/palace/stable/install/#Build-from-source), e.g. if you want to build for GPU support, but the following is a simple example for building Palace with Intel compilers and MPI support for Fritz HPC cluster.

```sh
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
```

```sh
make -j 72
```

8. Check if palace is working by running:

```sh
./bin/palace --version
```

### Install ORCA

1. Download ORCA from GitHub (with HTTPS!)

```sh
git clone https://github.com/DI-PASSIONATE/ORCA.git
```

2. First time conda setup

The following steps only have to be performed once on the system. They will initialize conda and cause conda to store packages and environments under $WORK instead of $HOME in order to save space in the latter.

Summary of all the following steps for easy copy and paste:

```sh
if [ ! -f ~/.bash_profile ]; then
  echo "if [ -f ~/.bashrc ]; then . ~/.bashrc; fi" > ~/.bash_profile
fi
module add python
conda config --add pkgs_dirs $WORK/software/private/conda/pkgs
conda config --add envs_dirs $WORK/software/private/conda/envs
```

3. Create a conda environment for ORCA and install dependencies

```sh
conda create -n orca python=3.13
conda activate orca
conda install -c conda-forge mesalib libglu
```

4. Install ORCA dependencies

```sh
pip install -e ./ORCA/
```

   This is a simulation-only install: GDS generation, conversion and Palace simulation, which is all the Slurm example runs. PyTorch is not needed for it.

5. (Optional) To also train models on the cluster, install the CPU build of PyTorch (or a GPU build if you have access to a GPU node) from [PyTorch.org](https://pytorch.org/get-started/locally/), then ORCA's training extra:

```sh
# Example only, replace with the command from PyTorch.org for your system!
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e "./ORCA/[train]"
```

6. Make sure everything is set up correctly

```sh
which palace
which python
python --version
```

## Training on the TinyGPU cluster

Training needs no Palace, but a CUDA build of PyTorch, so it gets its own environment. $HOME and $WORK are shared with Fritz, as is the conda setup above, so the simulation results can be trained on in place.

1. Connect to the TinyGPU frontend and allocate a GPU node, so the installed PyTorch can be checked against a real GPU

```sh
ssh <username>@tinyx.nhr.fau.de
salloc.tinygpu --gres=gpu:1 --time=01:00:00
export http_proxy=http://proxy.nhr.fau.de:80 && export https_proxy=http://proxy.nhr.fau.de:80
```

2. Create the environment and install a CUDA build of PyTorch, then ORCA's training extra

```sh
module load python
conda create -n orca-gpu python=3.13
conda activate orca-gpu
# Example only, take the command for your CUDA version from PyTorch.org; check the node's driver with nvidia-smi
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e "./ORCA/[train]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

3. Submit the training from the folder the simulation ran in (it reads `output/<geometry>/results` there)

```sh
cd ORCA/examples/slurm_runs
sbatch.tinygpu orca_slurm_train.sh
```

Each job writes its model, train/validation/test split and test errors to `output/<geometry>/training_<job id>/`.
