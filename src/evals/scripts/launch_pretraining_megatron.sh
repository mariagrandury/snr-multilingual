#!/bin/bash
# launch_pretraining_megatron.sh - Idempotent SNR pretraining eval launcher
# (megatron_lm backend, raw Megatron checkpoints under
#  /iopsstor/.../data-mix-small/Megatron-LM/logs/Meg-Runs/.../checkpoints/).
#
# Mirror of launch_pretraining_hf.sh for the Megatron backend. Use the HF
# launcher as the primary path (vLLM is faster and dodges the cgroup OOMs
# that took out ~196 megatron eval jobs on 2026-05-04). Reach for this one
# only when:
#   * the HF-converted snapshot for a (cell, iter) hasn't been produced yet
#     and you can't wait for the conversion job, OR
#   * you specifically want to evaluate against the raw megatron weights
#     (e.g. to confirm the converted ckpt is faithful).
#
# Reads the same snr_progress.csv (auto-refreshed at script start). For
# every row that is NOT `completed` and has NO active_jobids, submits one
# evaluate.sbatch per (cell, iter) with TASKS=<remaining>.
#
# Megatron path runs `torchrun --nproc-per-node=4` so TP/PP fan-out is
# baked into the harness invocation, not exposed via env. Walltime is
# scaled per-size (megatron eval is slower than vLLM, so estimates are ~2x
# the HF per-task minutes).
#
# Usage:
#   bash scripts/launch_pretraining_megatron.sh                # submit
#   bash scripts/launch_pretraining_megatron.sh --dry-run      # preview
#   bash scripts/launch_pretraining_megatron.sh --filter seed28
#   bash scripts/launch_pretraining_megatron.sh --no-refresh   # use existing CSV
#
set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
FILTER=""
PARTITION="normal"
RESERVATION=""
REFRESH=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=1; shift ;;
        --filter)      FILTER="$2"; shift 2 ;;
        --partition)   PARTITION="$2"; shift 2 ;;
        --reservation) RESERVATION="$2"; shift 2 ;;
        --no-refresh)  REFRESH=0; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

REPO_DIR=$PWD
MEG_BASE=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small
MODELS_FILE=$REPO_DIR/configs/signal_to_ratio/models_pretraining_custom_all.txt
TASKS_FILE=$REPO_DIR/configs/signal_to_ratio/tasks_pretraining_full.txt
CSV=$REPO_DIR/snr_progress.csv

if (( REFRESH )); then
    echo "[meg] refreshing $CSV ..."
    python3.11 scripts/snr_progress.py \
        --models "$MODELS_FILE" \
        --tasks-file "$TASKS_FILE" \
        --seed-iters seed28=6000,28000,42000,44000,46000,48000,50000 \
        > /dev/null
fi
[[ -f "$CSV" ]] || { echo "ERROR: $CSV missing (run without --no-refresh)" >&2; exit 1; }

# Megatron eval is ~2x slower than vLLM at same model size (no batched
# kv-cache reuse, more all-reduce traffic). Estimates from observed runs.
per_task_min()  { case "$1" in 175M) echo 4;; 350M) echo 6;; 600M) echo 8;; 1B) echo 10;; esac; }
COLD_START_MIN=20   # megatron has no vLLM compile cache to warm
CAP_MIN=719

walltime_for() {
    local size=$1 remaining=$2
    local pt; pt=$(per_task_min "$size")
    local m=$(( COLD_START_MIN + remaining * pt ))
    (( m > CAP_MIN )) && m=$CAP_MIN
    printf "%02d:%02d:00" $(( m / 60 )) $(( m % 60 ))
}

SBATCH_RES_ARGS=()
[[ -n "$RESERVATION" ]] && SBATCH_RES_ARGS=(--reservation="$RESERVATION")

# Same iter-desc / size-desc submit order as the HF launcher.
SORTED=$(awk -F, 'NR>1 { print $0 }' "$CSV" | python3.11 -c "
import sys, csv
rows = list(csv.reader(sys.stdin))
size_rank = {'1B': 0, '600M': 1, '350M': 2, '175M': 3}
def key(r):
    name = r[0]
    parts = name.split('-')
    size = parts[1]
    iter_n = int(parts[-1].replace('iter', ''))
    return (-iter_n, size_rank.get(size, 9))
rows.sort(key=key)
for r in rows:
    print('\t'.join([r[0], r[1], r[2], r[3], r[4], r[5]]))
")

submitted=0; skipped_active=0; skipped_done=0; skipped_no_ckpt=0; skipped_filter=0
echo "[meg] partition=$PARTITION dry_run=$DRY_RUN filter=${FILTER:-<none>}"
echo ""

while IFS=$'\t' read -r name status done total remaining active_jobids; do
    [[ -z "$name" ]] && continue
    if [[ -n "$FILTER" && "$name" != *"$FILTER"* ]]; then
        skipped_filter=$((skipped_filter + 1)); continue
    fi
    if [[ "$status" == "completed" ]]; then
        skipped_done=$((skipped_done + 1)); continue
    fi
    if [[ -n "$active_jobids" ]]; then
        echo "  SKIP active: $name (jobs=$active_jobids)"
        skipped_active=$((skipped_active + 1)); continue
    fi

    if [[ "$name" =~ ^(apertus-([0-9]+[MB])-fwEdu[0-9]+-fw2[0-9]+-seed[0-9]+)-iter([0-9]+)$ ]]; then
        cell=${BASH_REMATCH[1]}
        size=${BASH_REMATCH[2]}
        iter=${BASH_REMATCH[3]}
    else
        echo "  WARN: unparseable name: $name"; continue
    fi

    ckpt_path="$MEG_BASE/$cell/checkpoints"
    iter_dir=$(printf "%s/iter_%07d" "$ckpt_path" "$iter")
    if [[ ! -d "$iter_dir" ]]; then
        skipped_no_ckpt=$((skipped_no_ckpt + 1)); continue
    fi

    n_remaining=$(awk -F, '{print NF}' <<<"$remaining")
    wall=$(walltime_for "$size" "$n_remaining")

    if (( DRY_RUN )); then
        echo "  would submit: $name  --time=$wall  remaining=$n_remaining"
        submitted=$((submitted + 1)); continue
    fi

    # Prefix-export rather than --export=ALL,K=V,...: sbatch's --export uses
    # commas as separators between vars, so TASKS=a,b,c gets truncated at the
    # first comma (silently submits "1 task per ckpt" jobs). Putting the vars
    # in sbatch's process env via the prefix form below works because sbatch's
    # default --export=ALL snapshots them intact.
    jid=$(LM_EVAL_BACKEND=megatron_lm \
          TOKENIZER=alehc/swissai-tokenizer \
          BOS=true \
          APPLY_CHAT_TEMPLATE=false \
          WANDB_ENTITY=mariagrandury-epflnlp \
          WANDB_PROJECT=snr-experiments \
          TASKS="$remaining" \
          CKPT_ITER=$iter \
        sbatch --parsable \
            --job-name="eval-${name}" \
            --partition="$PARTITION" \
            --time="$wall" \
            "${SBATCH_RES_ARGS[@]}" \
            scripts/evaluate.sbatch "$ckpt_path" "$name") \
        && {
            echo "  $jid  $name  $wall  remaining=$n_remaining"
            submitted=$((submitted + 1))
        } || {
            echo "  sbatch FAILED: $name"
        }
    sleep 1
done <<< "$SORTED"

echo ""
echo "submitted=$submitted  skipped_active=$skipped_active  skipped_done=$skipped_done  skipped_no_ckpt=$skipped_no_ckpt  skipped_filter=$skipped_filter"
