#!/bin/bash
# score_bpb.sbatch — per-language BPB + perplexity for every due checkpoint of
# one cell, the predictivity study's outcome metric (see scripts/score_bpb.py).
#
# One job per CELL, not per checkpoint: the model load is seconds against
# minutes of scoring, so paying the container start and pip install once for
# ~10 checkpoints is most of the wall-clock saving available here.
#
#   sbatch --job-name=bpb-<cell> scripts/score_bpb.sbatch <cell> [iters...]
#   sbatch scripts/score_bpb.sbatch lm-90M-L2-deep-seed1904 450 900
#
# With no iters it scores every checkpoint converted for the cell.
#
# A cell does not fit one debug slot. Scoring costs a flat ~448 s/ckpt at 90M
# and ~816 s at 350M (measured off the bpb.json mtimes), so debug's 1:30 wall
# gets through ~11 or ~6 of the 20 planned checkpoints. The first round of
# these therefore ended 8 TIMEOUTs with 81/780 scored and nothing left in the
# queue: per-checkpoint idempotency kept the work, but nothing ever asked for
# the rest. Two additions fix that, both borrowed from
# ../../pretrain/data/submit_build_one.sh, which survives the 12 h wall the
# same way:
#
#   * a SINGLETON SUCCESSOR queued up front (same job name, so only one runs
#     at a time). Queuing it before the work starts is what makes it survive a
#     kill -9 at the wall. The successor re-derives what is still due and exits
#     without chaining when nothing is, so the chain ends itself.
#   * a DEADLINE GUARD that will not start a checkpoint it cannot finish.
#     Without it every slot burns up to 13 min of GPU on a scoring pass that
#     gets killed before it writes.
#
#SBATCH --account=infra01
#SBATCH --job-name=score-bpb
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200000
#SBATCH --time=03:00:00
#SBATCH --output=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals/logs/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals/logs/%x_%j.err
set -uo pipefail

CELL=${1:?usage: sbatch score_bpb.sbatch <cell> [iters...]}
ORIG_ARGS=("$@")   # replayed verbatim by the successor, so it re-derives the
shift || true      # due set instead of freezing today's list of converted iters
ITERS=("$@")

# $0 inside a batch job is /var/spool/slurmd/job<id>/slurm_script, so the
# successor cannot find this file by a path relative to it (../CLAUDE.md, the
# SCRIPT_PATH gotcha). Absolute, like submit_build_one.sh.
SCRIPT=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/evals/scripts/score_bpb.sbatch
MAX_CHAIN=${MAX_CHAIN:-12}

REPO_DIR=${REPO_DIR:-$(dirname "$(dirname "$SCRIPT")")}
STAGING=${STAGING:-/capstor/store/cscs/swissai/infra01/msnr-hf-models}
# Results live beside the harness results, under the same cell-iter name, so
# the eval tooling and the analysis find them in one place.
OUT_ROOT=${OUT_ROOT:-/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/msnr}
DATA_DIR=${DATA_DIR:-/iopsstor/scratch/cscs/mariagrandury/data}
MAX_TOKENS=${MAX_TOKENS:-1000000}

if [ ${#ITERS[@]} -eq 0 ]; then
    # .hf_complete is convert-snr.sh's last write, so its presence means the
    # snapshot is whole — config.json alone can match a half-written save.
    mapfile -t ITERS < <(find "$STAGING/$CELL" -name .hf_complete -printf '%h\n' 2>/dev/null \
        | xargs -r -n1 basename | sed 's/^iter_0*//' | sort -n)
fi
echo "cell=$CELL  converted=${#ITERS[@]}  max_tokens=$MAX_TOKENS"
[ ${#ITERS[@]} -eq 0 ] && { echo "no converted checkpoints for $CELL"; exit 1; }

# What is still DUE. Deciding this here rather than only inside the loop is
# what lets the chain terminate: a successor that finds nothing due exits
# without queuing another one.
DUE=()
for it in "${ITERS[@]}"; do
    [ -s "$OUT_ROOT/$CELL-iter$it/bpb/bpb.json" ] || DUE+=("$it")
done
echo "due=${#DUE[@]}: ${DUE[*]:-none}"
[ ${#DUE[@]} -eq 0 ] && { echo "$CELL fully scored — chain ends here."; exit 0; }

# Queue the successor NOW, before any scoring: at the wall Slurm kills the
# batch script outright, so anything submitted after the srun would never run.
# --dependency=singleton + the shared job name keep exactly one alive.
# The depth is threaded through the successor's env, NOT counted from the
# logs/ dir: log files accumulate forever, so a lifetime tally would let a
# cell that ever used MAX_CHAIN slots never chain again — a re-launch weeks
# later (new checkpoints, new --max-tokens) must start a fresh chain at 0.
CHAIN_DEPTH=${CHAIN_DEPTH:-0}
if [ "$CHAIN_DEPTH" -lt "$MAX_CHAIN" ]; then
    echo "queuing singleton successor (chain depth $CHAIN_DEPTH/$MAX_CHAIN)"
    sbatch --dependency=singleton --job-name="${SLURM_JOB_NAME:-bpb-$CELL}" \
           --chdir="$REPO_DIR" --export=ALL,CHAIN_DEPTH=$((CHAIN_DEPTH + 1)) \
           "$SCRIPT" "${ORIG_ARGS[@]}"
else
    echo "WARN: chain cap $MAX_CHAIN reached for $CELL — not queuing a successor"
fi

# Wall-clock deadline, so the loop can stop instead of being killed mid-scoring.
# Read at runtime, not from the #SBATCH line, because debug_drain.sh rewrites
# the limit to 01:30:00 while the job is still pending.
DEADLINE=${SLURM_JOB_END_TIME:-0}
if ! [ "$DEADLINE" -gt 0 ] 2>/dev/null; then
    DEADLINE=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o '%e' 2>/dev/null \
        | xargs -r -I{} date -d '{}' +%s 2>/dev/null)
fi
[ -n "${DEADLINE:-}" ] || DEADLINE=0
echo "deadline=$DEADLINE ($([ "$DEADLINE" -gt 0 ] && date -d "@$DEADLINE" || echo unknown))"

# Same container as the harness: it already carries torch + transformers, and
# reusing it keeps the numerics identical to what lm_eval sees.
srun -n 1 --mpi=pmix -ul --environment=./containers/env_vllm.toml \
     --export=ALL,HF_TOKEN,HF_HOME,HF_HUB_CACHE bash -lc "
    cd '$REPO_DIR'
    set -uo pipefail
    unset SSL_CERT_FILE
    export HF_HOME='${HF_HOME:-/iopsstor/scratch/cscs/mariagrandury/hf_home}'
    # Compute nodes have no internet. Without these, from_pretrained on a LOCAL
    # path still calls the Hub to check for a newer revision and blocks on the
    # connect timeout — minutes per checkpoint, for nothing (evals CLAUDE.md
    # bug 12, the same fix convert-snr.sh needed).
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}
    rc=0; last=0
    for it in ${DUE[*]}; do
        padded=\$(printf 'iter_%07d' \$it)
        model='$STAGING/$CELL'/\$padded
        dest='$OUT_ROOT'/'$CELL'-iter\$it/bpb
        mkdir -p \"\$dest\"
        # Idempotent, like every other stage: a re-run after a walltime kill
        # skips what already landed instead of rescoring it. Still checked here
        # even though DUE was filtered on the host — a chained predecessor of
        # this job may have written it since.
        if [ -s \"\$dest/bpb.json\" ]; then
            echo \"== \$padded already scored, skipping\"; continue
        fi
        # Per-checkpoint cost is flat within a cell (448 s at 90M, 816 s at
        # 350M, <2% spread), so the previous checkpoint is a good predictor.
        # +20% margin. The first one always runs: no measurement yet, and even
        # the slowest size fits a fresh debug slot.
        left=\$(( $DEADLINE - \$(date +%s) ))
        if [ $DEADLINE -gt 0 ] && [ \$last -gt 0 ] && [ \$left -lt \$(( last * 6 / 5 )) ]; then
            echo \"== stopping before \$padded: \${left}s left, need ~\$(( last * 6 / 5 ))s — successor takes it\"
            break
        fi
        echo \"== scoring \$padded (\${left}s of wall left)\"
        t0=\$(date +%s)
        python3 scripts/score_bpb.py --model \"\$model\" \
            --data-dir '$DATA_DIR' --max-tokens $MAX_TOKENS \
            --out \"\$dest/bpb.json\" && last=\$(( \$(date +%s) - t0 )) || rc=1
    done
    exit \$rc
"
echo "END: $(date)"
