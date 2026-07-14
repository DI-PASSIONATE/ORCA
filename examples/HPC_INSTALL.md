## How to install ORCA on the Fritz HPC Cluster
### Install AWS Palace
1. Connect to the frontend node of Fritz using SSH:

```sh
ssh <username>@fritz.nhr.fau.de
```

2. Allocate an interactive node

```sh
salloc -N 1 --partition=singlenode --time=01:00:00
```

3. Make the internet accessible from the allocated node by running:

```sh
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
```

4. Load spack user installation

```sh
module load user-spack
```

5. Download and add the newest spack repository to get a palace version more recent than from 2023

```sh
mkdir -p ~/spack-repos
cd ~/spack-repos
git clone https://github.com/spack/spack-packages.git
spack repo add ~/spack-repos/spack-packages/repos/spack_repo/builtin/
cd ..
```

6. Install palace using spack (this make take a very long time, spack is very very slow sometimes, so be patient).

```sh
spack install palace@0.16.0 target=icelake ^openmpi ^openblas ^gcc
```

7. Load MPI and palace

```sh
module load openmpi
module load palace
```

8. Check if palace is working by running:

```sh
palace --version
```

### Install ORCA

1. Download ORCA from GitHub (with HTTPS!)

```sh
git clone https://github.com/DI-PASSIONATE/ORCA.git
```

2. Load python and palace

```sh
module load python
module load palace
```

3. Make some extremely unnecessary steps to create a virtual conda environment with version 3.12

```sh
unset CONDA_ENVS_PATH
unset CONDA_PKGS_DIRS
mkdir -p ~/conda_pkgs
mkdir -p ~/conda_envs
conda config --add pkgs_dirs ~/conda_pkgs
conda config --add envs_dirs ~/conda_envs
conda create -n orca python=3.12
conda activate orca
conda install -c conda-forge mesalib
conda install -c conda-forge libglu
```

4. Install ORCA dependencies

```sh
pip install -e ./ORCA/
```

5. Make sure everything is set up correctly

```sh
which palace
which python
python --version
```

### Run a job

To run a job, you need to create a job script. Here is an example of a job script for ORCA:

```bash
#!/bin/bash -l
#
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --job-name=ORCA-FEM-SIM
#SBATCH --export=NONE
#
# first non-empty non-comment line ends SBATCH options

unset SLURM_EXPORT_ENV

###### START YOUR ACTUAL JOB SCRIPT BELOW THIS LINE ######

module load openmpi
module load palace
module load python

# Activate the conda environment
conda activate orca

# Calls palace internally, which calls mpirun
srun python ./ORCA/examples/main.py
```