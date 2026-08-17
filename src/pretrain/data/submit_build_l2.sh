#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=build-l2
#SBATCH --time=11:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48000  # doc-count-driven peak headroom; packing is cpu-bound (9/node), see submit_build_one.sh
#SBATCH --output=/iopsstor/scratch/cscs/mariagrandury/data/build-l2-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/mariagrandury/data/build-l2-%j.out
#SBATCH --no-requeue

set -euo pipefail
source ~/.bashrc
conda activate snr
cd /iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data

# Tokenizer peaks at ~16-32 threads; cap so builds pack many-per-node (see
# submit_build_one.sh). ~32 cores is plenty for one build.
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OMP_NUM_THREADS=$RAYON_NUM_THREADS
export TOKENIZERS_PARALLELISM=true
# Line-level progress in the log instead of one flush at exit (see
# submit_build_one.sh).
export PYTHONUNBUFFERED=1

SCRIPT=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data/submit_build_l2.sh
OUT=/iopsstor/scratch/cscs/mariagrandury/data
PREFIX=$OUT/fineweb_L2

# Scheme A L2 = Russian (rus_Cyrl). Built in its own job, in parallel with the
# main sweep's english build, because it is independent — it only needs the
# validation manifest (already written), not the english data. Sized for the
# LARGEST run we intend at L=2 (the 1.7B model), straight from
# build_data_mixtures' own sizing so this job and `--settings 2` can never
# disagree (~92B).
TARGET=$(python -c "import build_data_mixtures as b; print(b.fineweb_target_tokens(2))")

# Already complete? (.idx present and the resume checkpoint removed.) Skip and,
# crucially, do NOT requeue — this is what ends the singleton chain and prevents
# a post-completion successor from rebuilding from scratch.
if [ -f "$PREFIX.idx" ] && [ ! -f "$PREFIX.checkpoint.json" ]; then
  echo "[$(date)] fineweb_L2 already built ($PREFIX.idx present) — nothing to do."
  exit 0
fi

# Survive the 12h wall: queue a singleton successor UP FRONT (runs only after
# this job ends; the guard above no-ops it once the build is done). Capped to
# avoid a runaway loop on a hard failure.
n_attempts=$(find "$OUT" -maxdepth 1 -name 'build-l2-*.out' | wc -l)
if [ "$n_attempts" -lt 20 ]; then
  echo "[$(date)] queuing singleton successor (attempt $n_attempts) to continue past the wall"
  sbatch --dependency=singleton "$SCRIPT"
fi

# create_data_mixture resumes from its own checkpoint if this is a continuation.
python create_data_mixture.py \
  --target_tokens "$TARGET" \
  --fineweb_pct 100 --dclm_pct 0 \
  --languages rus_Cyrl \
  --temperature 1.0 \
  --validation_manifest "$OUT/validation.manifest.json" \
  --output_prefix "$PREFIX"

echo "[$(date)] fineweb_L2 build complete (target $TARGET tokens)"
