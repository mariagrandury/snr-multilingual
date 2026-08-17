#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=11:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=24000
#SBATCH --output=/iopsstor/scratch/cscs/mariagrandury/data/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/mariagrandury/data/logs/%x-%j.out
#SBATCH --no-requeue

# Build ONE data mixture (english, or one FineWeb-2 setting of one scheme), then
# self-chain past the 12h wall. Driven by --export vars so a single script backs
# every per-mix job launched by launch_builds.sh:
#   BUILD_SCHEME  A|B
#   BUILD_STAGE   english|fineweb
#   BUILD_SETTING L value (fineweb only)
#   BUILD_OUT     output --data_dir
set -euo pipefail
source ~/.bashrc
conda activate snr
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data

# The tokenizer (Rust/rayon) peaks at ~16-32 threads and slows past ~64, and the
# serial write path is 20x faster than tokenization, so a build needs only a
# handful of cores. Cap threads to the allocation so ~9 builds pack per 288-core
# node instead of one build hogging (and running slow on) a whole node.
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OMP_NUM_THREADS=$RAYON_NUM_THREADS
export TOKENIZERS_PARALLELISM=true

# Without this, Python block-buffers stdout when it is a file, so a 12h build
# shows NOTHING in its log until the process exits (and a wall-clock kill loses
# the buffer entirely). Set for the wrapper and, via the environment, its
# create_data_mixture.py child.
export PYTHONUNBUFFERED=1

SCRIPT=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data/submit_build_one.sh
LOGDIR=/iopsstor/scratch/cscs/mariagrandury/data/logs
: "${BUILD_SCHEME:?set via --export}"; : "${BUILD_STAGE:?set via --export}"; : "${BUILD_OUT:?set via --export}"

if [ "$BUILD_STAGE" = english ]; then
  PREFIX=$BUILD_OUT/english_dclm
  STAGE_ARGS=(--stage english)
else
  : "${BUILD_SETTING:?set via --export for fineweb}"
  PREFIX=$BUILD_OUT/fineweb_L$BUILD_SETTING
  STAGE_ARGS=(--stage fineweb --settings "$BUILD_SETTING")
fi

# Complete already? (.idx present, checkpoint gone.) Skip and DON'T requeue —
# this ends the singleton chain and prevents rebuilding a finished dataset.
if [ -f "$PREFIX.idx" ] && [ ! -f "$PREFIX.checkpoint.json" ]; then
  echo "[$(date)] $PREFIX already built — nothing to do."
  exit 0
fi

# Survive the 12h wall: queue a singleton successor UP FRONT (same job name, so
# only one runs at a time). The idempotent build resumes from its checkpoint;
# the guard above no-ops the successor once done. Capped against a failure loop.
n_attempts=$(find "$LOGDIR" -name "${SLURM_JOB_NAME}-*.out" 2>/dev/null | wc -l)
if [ "$n_attempts" -lt 25 ]; then
  echo "[$(date)] queuing singleton successor (attempt $n_attempts)"
  sbatch --dependency=singleton --job-name="$SLURM_JOB_NAME" --export=ALL "$SCRIPT"
fi

python build_data_mixtures.py --scheme "$BUILD_SCHEME" --output_dir "$BUILD_OUT" "${STAGE_ARGS[@]}"
echo "[$(date)] $PREFIX build complete"
