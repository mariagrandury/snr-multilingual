#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=11:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
# 32 cpus caps packing at ~9 builds per 288-core node; a node has ~460GB, so mem
# is not the binding constraint. 48GB is generous headroom for the doc-count-
# driven peak of the big multilingual builds (the writer holds all
# sequence_lengths in memory, plus one streamed parquet batch), without reducing
# packing (9 x 48 < 460).
#SBATCH --mem=48000
#SBATCH --output=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data/logs/%x-%j.out
#SBATCH --error=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data/logs/%x-%j.out
#SBATCH --no-requeue

# Build ONE data mixture (english, or one FineWeb-2 setting of one scheme),
# then self-chain past the 12h wall. Driven by --export vars so a single script
# backs every per-mix job launched by launch_builds.sh:
#   BUILD_SCHEME  one of launch_trainings.DATA_SCHEMES
#                 (A|AT3|B|ZH|ES|DCLMP|FWEB) — it carries the language
#                 lists, the temperature, the subdir and, for DCLMP/FWEB, the
#                 English corpus
#   BUILD_STAGE   english|fineweb
#   BUILD_SETTING L value (fineweb only)
#   BUILD_OUT     output --data_dir (the scheme's dir, i.e. <root>/<subdir>)
#   BUILD_DST     optional iopsstor stage root; only the 92B rebuilds set it,
#                 so a rebuilt mixture lands beside — never on top of — the
#                 copy the running cells are memmapping
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
# Logs live on capstor beside the data. They are the only place print_plan's
# per-source token allocation was recorded before <prefix>.plan.json existed,
# and the analysis wants it long after iopsstor scratch is swept. Measured
# build output is ~4 lines/min (~3 bytes/s), so this costs nothing.
LOGDIR=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data/logs
: "${BUILD_SCHEME:?set via --export}"; : "${BUILD_STAGE:?set via --export}"; : "${BUILD_OUT:?set via --export}"

if [ "$BUILD_STAGE" = english ]; then
  PREFIX=$BUILD_OUT/english_dclm
  STAGE_ARGS=(--stage english)
else
  : "${BUILD_SETTING:?set via --export for fineweb}"
  PREFIX=$BUILD_OUT/fineweb_L$BUILD_SETTING
  STAGE_ARGS=(--stage fineweb --settings "$BUILD_SETTING")
fi

# Every scheme but A builds into <root>/<subdir>, so a prefix taken relative
# to BUILD_OUT would lose the subdir component and the stager would act on the
# scheme-A mixture instead. Strip the scheme's own subdir (empty for A, which
# builds into the root), keeping e.g. schemeB/fineweb_LN — and pass that root
# to the stager as SRC, so a build sent somewhere other than the capstor master
# (a 92B rebuild root) stages from where it actually wrote.
SUBDIR=$(python -c "import sys; sys.path.insert(0, '..'); from launch_trainings import DATA_SCHEMES; print(DATA_SCHEMES['$BUILD_SCHEME']['subdir'])")
DATA_ROOT=${BUILD_OUT%${SUBDIR:+/$SUBDIR}}
# That strip is a silent no-op when BUILD_OUT does not end in exactly /<subdir>
# (a trailing slash, a typo): the stager would then run with SRC=<scheme dir>
# and stage e.g. ZH/fineweb_L2 over the root's fineweb_L2. Refuse instead.
if [ "$DATA_ROOT${SUBDIR:+/$SUBDIR}" != "$BUILD_OUT" ]; then
  echo "BUILD_OUT=$BUILD_OUT does not end in /$SUBDIR (scheme $BUILD_SCHEME) — refusing" >&2
  exit 1
fi

# Rebuilds stage into their own tree; everything else takes the stager's default.
if [ -n "${BUILD_DST:-}" ]; then export DST="$BUILD_DST"; fi

# What to stage: the mixture itself, plus — for a scheme subdir going to the
# training stage — its english_dclm symlink. Training reads
# <data_dir>/<subdir>/english_dclm, and the stager mirrors only the prefixes it
# is given, so without this a new scheme's cells die on "... cannot be found
# at .../ZH/english_dclm" (2026-09-11, all six ZH/ES trainings). Not for
# rebuilds: their english link points outside SRC, where the stager would copy
# the 736 GB target instead of linking it.
TO_STAGE=("${PREFIX#$DATA_ROOT/}")
# (not for an english build: its own PREFIX is already that path)
if [ -n "$SUBDIR" ] && [ -z "${BUILD_DST:-}" ] && [ "$BUILD_STAGE" != english ]; then
  TO_STAGE+=("$SUBDIR/english_dclm")
fi

# Complete already? (.idx present, checkpoint gone.) Skip and DON'T requeue —
# this ends the singleton chain and prevents rebuilding a finished dataset.
if [ -f "$PREFIX.idx" ] && [ ! -f "$PREFIX.checkpoint.json" ]; then
  echo "[$(date)] $PREFIX already built — nothing to do."
  # Still make sure it is staged: a mixture built before staging existed (or
  # staged onto a since-swept iopsstor) would otherwise never reach training.
  SRC="$DATA_ROOT" bash "$(dirname "$SCRIPT")/stage_to_iopsstor.sh" "${TO_STAGE[@]}"
  exit $?   # FAILED if the staging did — --no-requeue, so this only sets the job state
fi

# Survive the wall: queue a singleton successor UP FRONT (same job name, so
# only one runs at a time). The idempotent build resumes from its checkpoint;
# the guard above no-ops the successor once done. Queuing it first is also what
# makes this preemption-safe: a killed attempt leaves its successor pending.
# Capped against a failure loop — and a preemption spends one of those
# attempts, so BUILD_MAX_ATTEMPTS raises the cap for a heavily preempted chain.
# The successor inherits the partition this attempt actually ran in (a drainer
# may have moved it) and, on `preemptable`, its longer wall — a limit cannot be
# raised later, so it has to be asked for at submission.
CHAIN_PART=${BUILD_PARTITION:-$SLURM_JOB_PARTITION}
CHAIN_TIME=${BUILD_TIME:-11:59:59}
[ "$CHAIN_PART" = preemptable ] && [ -z "${BUILD_TIME:-}" ] && CHAIN_TIME=23:59:00
n_attempts=$(find "$LOGDIR" -name "${SLURM_JOB_NAME}-[0-9]*.out" 2>/dev/null | wc -l)
if [ "$n_attempts" -lt "${BUILD_MAX_ATTEMPTS:-25}" ]; then
  echo "[$(date)] queuing singleton successor (attempt $n_attempts, $CHAIN_PART, $CHAIN_TIME)"
  sbatch --dependency=singleton --job-name="$SLURM_JOB_NAME" \
         --partition="$CHAIN_PART" --time="$CHAIN_TIME" \
         ${BUILD_EXCLUSIVE:---exclusive} --export=ALL "$SCRIPT"
fi

python build_data_mixtures.py --scheme "$BUILD_SCHEME" --output_dir "$BUILD_OUT" "${STAGE_ARGS[@]}"
echo "[$(date)] $PREFIX build complete"

# Builds write to the capstor master, but training reads from the iopsstor
# stage (Megatron memmaps the .bin and reads it shuffled; capstor is ~28x
# slower per random read — ../CLAUDE.md #8). Without this copy the new mixture
# exists but every cell at that setting dies on "One or both of the .idx and
# .bin files cannot be found". Copy, never move: capstor is what survives the
# scratch sweep. No-ops when the build did not finish, or is already staged.
SRC="$DATA_ROOT" bash "$(dirname "$SCRIPT")/stage_to_iopsstor.sh" "${TO_STAGE[@]}"
