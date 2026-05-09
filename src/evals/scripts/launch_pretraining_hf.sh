#!/bin/bash
# launch_pretraining_hf.sh - Idempotent SNR pretraining eval launcher (vLLM
# backend, HF-converted checkpoints under /iopsstor/.../snr-hf-checkpoints/).
#
# Reads the canonical (cell, iter, status, remaining_tasks, active_jobids)
# matrix from snr_progress.csv (auto-refreshed at script start, written by
# scripts/snr_progress.py — committed alongside the script). For every row
# that is NOT `completed` and has NO active_jobids, submits one
# `evaluate.sbatch` with TASKS=<remaining> so lm_eval (BATCH_TASKS=1) loads
# only what's missing. Walltime is sized to remaining_tasks × per-size cost.
#
# Idempotency layers:
#   1. snr_progress.csv `status=completed`            → SKIP
#   2. snr_progress.csv `active_jobids` non-empty     → SKIP
#   3. HF iter dir missing on disk (conversion lag)   → SKIP
#
# Per-size vLLM parallelism (CLAUDE.md bug 14: vLLM rejects TP > kv_heads):
#   175M  TP=4 PP=1   (kv=4 → TP=4 OK; uses all 4 GPUs via TP)
#   350M  TP=1 PP=4   (kv=5 → only TP=1; PP=4 fills the node)
#   600M  TP=2 PP=2   (kv=6 → TP=2 max; PP=2 uses remaining 2 GPUs)
#   1B    TP=1 PP=4   (kv=7 → only TP=1; PP=4 fills the node)
#
# Walltime sizing (re-fit on 2026-05-09 from 4 size-test jobs after fixing
# the cache/offline cascade — see notes in evaluate.sbatch). With BATCH_TASKS=1
# + HF_DATASETS_OFFLINE=1 + populated cache, ALL sizes finished 67 tasks in
# ~23-25 min. Cold start (pip install + vLLM init + dataset load) dominates;
# per-task generation is fast and roughly size-independent because vLLM batches
# efficiently. Old estimates (cold=25, per_task=2-8 min/size) over-allocated by
# 2-9x and hurt queue priority. New shape:
#   wall = max(MIN_WALL, COLD_START + remaining_tasks * per_task_min) capped at 11:59:00
#   COLD_START = 15 min; per_task_min = 0.5 (single value, all sizes); MIN_WALL = 20 min.
# So 67 tasks → 15 + 33.5 ≈ 49 min walltime (2x observed), 5 tasks → 20 min floor.
#
# Canonical seed → iter policy (encoded in the snr_progress.csv refresh):
#   seeds 1904, 1797 → all 13 canonical iters (2k, 6k, 12k, 18k, 22k, 28k,
#                       34k, 38k, 42k, 44k, 46k, 48k, 50k)
#   seed  28         → narrow set: 6k, 28k, 42k, 44k, 46k, 48k, 50k
#
# Usage:
#   bash scripts/launch_pretraining_hf.sh                # submit
#   bash scripts/launch_pretraining_hf.sh --dry-run      # preview
#   bash scripts/launch_pretraining_hf.sh --filter seed1797   # NAME substring
#   bash scripts/launch_pretraining_hf.sh --no-refresh   # use existing CSV
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
HF_BASE=/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints
MODELS_FILE=$REPO_DIR/configs/signal_to_ratio/models_pretraining_custom_all.txt
TASKS_FILE=$REPO_DIR/configs/signal_to_ratio/tasks_pretraining_full.txt
CSV=$REPO_DIR/snr_progress.csv

# Refresh the snapshot once at script start. Same flags the dashboard uses,
# so the on-disk file stays consistent for both humans and launchers.
if (( REFRESH )); then
    echo "[hf] refreshing $CSV ..."
    python3.11 scripts/snr_progress.py \
        --models "$MODELS_FILE" \
        --tasks-file "$TASKS_FILE" \
        --seed-iters seed28=6000,28000,42000,44000,46000,48000,50000 \
        > /dev/null
fi
[[ -f "$CSV" ]] || { echo "ERROR: $CSV missing (run without --no-refresh)" >&2; exit 1; }

# Per-size config. TP/PP picked per CLAUDE.md bug 14 (vLLM kv_heads constraint).
tp_for()        { case "$1" in 175M) echo 4;; 350M) echo 1;; 600M) echo 2;; 1B) echo 1;; esac; }
pp_for()        { case "$1" in 175M) echo 1;; 350M) echo 4;; 600M) echo 2;; 1B) echo 4;; esac; }
COLD_START_MIN=15   # pip install + vLLM init + dataset load (offline cache hit)
PER_TASK_MIN=0.5    # batch-mode generation; ~size-independent
MIN_WALL_MIN=25     # floor — even 1-task jobs need vLLM init time
CAP_MIN=719         # 11:59:00 — normal-partition wall cap

walltime_for() {
    local remaining=$1
    # Use awk for floating-point math (PER_TASK_MIN is fractional).
    local m=$(awk -v c=$COLD_START_MIN -v p=$PER_TASK_MIN -v n=$remaining \
                  'BEGIN { printf "%d\n", c + n * p + 0.999 }')
    (( m < MIN_WALL_MIN )) && m=$MIN_WALL_MIN
    (( m > CAP_MIN ))      && m=$CAP_MIN
    printf "%02d:%02d:00" $(( m / 60 )) $(( m % 60 ))
}

SBATCH_RES_ARGS=()
[[ -n "$RESERVATION" ]] && SBATCH_RES_ARGS=(--reservation="$RESERVATION")

# Submit order: iter desc (50k → 2k), size desc (1B → 175M). 1B jobs are the
# longest so they hit the queue first; 175M jobs backfill while the big ones
# wait. Sort the CSV rows accordingly via awk.
SORTED=$(awk -F, 'NR>1 { print $0 }' "$CSV" | python3.11 -c "
import sys, csv
rows = list(csv.reader(sys.stdin))
size_rank = {'1B': 0, '600M': 1, '350M': 2, '175M': 3}
def key(r):
    name = r[0]
    # apertus-{size}-fwEdu{edu}-fw2{fw2}-seed{seed}-iter{iter}
    parts = name.split('-')
    size = parts[1]
    iter_n = int(parts[-1].replace('iter', ''))
    return (-iter_n, size_rank.get(size, 9))
rows.sort(key=key)
for r in rows:
    # Re-quote the remaining-task field (column 5) since it embeds commas.
    print('\t'.join([r[0], r[1], r[2], r[3], r[4], r[5]]))
")

submitted=0; skipped_active=0; skipped_done=0; skipped_no_ckpt=0; skipped_filter=0
echo "[hf] partition=$PARTITION dry_run=$DRY_RUN filter=${FILTER:-<none>}"
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

    # Parse size + cell + iter from the canonical NAME format.
    # apertus-{size}-fwEdu{edu}-fw2{fw2}-seed{seed}-iter{iter}
    if [[ "$name" =~ ^(apertus-([0-9]+[MB])-fwEdu[0-9]+-fw2[0-9]+-seed[0-9]+)-iter([0-9]+)$ ]]; then
        cell=${BASH_REMATCH[1]}
        size=${BASH_REMATCH[2]}
        iter=${BASH_REMATCH[3]}
    else
        echo "  WARN: unparseable name: $name"; continue
    fi

    iter_dir=$(printf "%s/%s/iter_%07d" "$HF_BASE" "$cell" "$iter")
    if [[ ! -f "$iter_dir/config.json" ]] || ! ls "$iter_dir"/model.safetensors* >/dev/null 2>&1; then
        skipped_no_ckpt=$((skipped_no_ckpt + 1)); continue
    fi

    n_remaining=$(awk -F, '{print NF}' <<<"$remaining")
    tp=$(tp_for "$size")
    pp=$(pp_for "$size")
    wall=$(walltime_for "$n_remaining")

    if (( DRY_RUN )); then
        echo "  would submit: $name  TP=$tp PP=$pp  --time=$wall  remaining=$n_remaining"
        submitted=$((submitted + 1)); continue
    fi

    # Export env via the parent shell rather than `--export=ALL,K=V,K=V`.
    # sbatch's --export uses commas as the separator BETWEEN vars, so a value
    # containing commas (TASKS=a,b,c) gets truncated at the first comma —
    # which silently submitted "1 task per ckpt" jobs the first time around
    # (caught by re-reading the .out logs after launch). The prefix-export
    # form below puts the vars in sbatch's process env; sbatch's default
    # --export=ALL then snapshots them intact for the slurm job.
    jid=$(LM_EVAL_BACKEND=vllm \
          TOKENIZER=alehc/swissai-tokenizer \
          BOS=true \
          APPLY_CHAT_TEMPLATE=false \
          BATCH_TASKS=1 \
          TP=$tp \
          PP=$pp \
          WANDB_ENTITY=mariagrandury-epflnlp \
          WANDB_PROJECT=snr-experiments \
          TASKS="$remaining" \
        sbatch --parsable \
            --job-name="eval-${name}" \
            --partition="$PARTITION" \
            --time="$wall" \
            "${SBATCH_RES_ARGS[@]}" \
            scripts/evaluate.sbatch "$iter_dir" "$name") \
        && {
            echo "  $jid  $name  TP=$tp PP=$pp  $wall  remaining=$n_remaining"
            submitted=$((submitted + 1))
        } || {
            echo "  sbatch FAILED: $name"
        }
    sleep 1
done <<< "$SORTED"

echo ""
echo "submitted=$submitted  skipped_active=$skipped_active  skipped_done=$skipped_done  skipped_no_ckpt=$skipped_no_ckpt  skipped_filter=$skipped_filter"
