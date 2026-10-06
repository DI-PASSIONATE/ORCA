#!/bin/bash
#
# Submits the two simulation jobs on Fritz: orca_slurm_prepare.sh on a single node (GDS generation,
# DRC, GDS conversion), then orca_slurm_simulate.sh on many nodes (Palace). Both are queued now; the
# simulation job only starts once the preparation job has finished successfully (afterok). If the
# preparation job fails or hits its time limit, the simulation job is cancelled instead
# (--kill-on-invalid-dep) rather than left pending; run this script again to resume, every stage
# keeps what earlier jobs finished.
#
# Run it from this folder on the Fritz frontend; both jobs write to ./output/<geometry name>:
#     bash submit.sh

set -euo pipefail
cd "$(dirname "$0")"

prepare_id=$(sbatch --parsable orca_slurm_prepare.sh)
echo "Submitted preparation job ${prepare_id}"

simulate_id=$(sbatch --parsable --dependency="afterok:${prepare_id}" --kill-on-invalid-dep=yes \
  orca_slurm_simulate.sh)
echo "Submitted simulation job ${simulate_id}, starts after job ${prepare_id} succeeds"
