#!/bin/bash
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --job-name=per-item-store
#SBATCH --output=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/signal-and-noise/analysis/rq08_subset_selection/per_item_store/%x_%j.out

# build_per_item_store.sbatch — the rq08 per-item store from the ladder's
# lm-eval samples files (see build_per_item_store.py). Resumable: a second
# submission skips every (model, step, task) already in the manifest.
#
#   sbatch analysis/rq08_subset_selection/build_per_item_store.sbatch [--pool predictivity_seeds]
#
# CPU-only: json decoding of ~500k samples files, parallel over checkpoint
# dirs. Never run this on the login node (the 1000-pid slice, the 4 M-file tree).

set -euo pipefail
echo "START $(date) on $(hostname)"
export PATH=/users/mariagrandury/miniconda3/envs/snr/bin:$PATH
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/signal-and-noise
export PYTHONPATH=$PWD:$PWD/../../src
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 SOURCE_DATE_EPOCH=0
mkdir -p analysis/rq08_subset_selection/per_item_store

python analysis/rq08_subset_selection/build_per_item_store.py \
    --workers "${SLURM_CPUS_PER_TASK:-64}" "$@"
echo "END $(date)"
