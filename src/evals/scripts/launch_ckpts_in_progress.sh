#!/bin/bash
# Submit one sbatch per ckpt that needs work — both [in_progress] (partial
# results, no active job) and [not_submitted] (no results, no job) entries
# from snr_progress.py. Already-running ckpts are skipped (they have a
# `jobs=` annotation in the dashboard).
#
# Walltime sizing:
#   * partial-progress ckpts: max(30 min, remaining * per_task + cold_start + buffer)
#   * no-progress ckpts:      12h (full normal-partition cap, default for new ckpts)
#
# Usage: bash scripts/launch_ckpts_in_progress.sh [--dry-run] [--filter SUBSTR]
#                                                 [--reservation RES]
#
# --filter SUBSTR restricts to ckpt names containing SUBSTR (passed through
# to snr_progress.py --filter). Useful for per-seed runs, e.g.
# `--filter seed1904` to skip seed 28 / 1797 entries entirely.
#
# Per-task minute estimates (logprob-only, since mgsm is no longer in the
# task list) are coarse averages from observed runs. Adjust if the typical
# task mix changes again.
set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
FILTER=""
RESERVATION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)     DRY_RUN=1; shift ;;
        --filter)      FILTER="$2"; shift 2 ;;
        --reservation) RESERVATION="$2"; shift 2 ;;
        *) echo "Unknown flag: $1" >&2; exit 1 ;;
    esac
done

SBATCH_RES_ARG=()
[[ -n "$RESERVATION" ]] && SBATCH_RES_ARG=(--reservation="$RESERVATION")

PROGRESS_ARGS=()
[[ -n "$FILTER" ]] && PROGRESS_ARGS+=(--filter "$FILTER")

declare -A PER_TASK_MIN=(["175M"]=4 ["350M"]=6 ["600M"]=8 ["1B"]=10)
COLD_START_MIN=15      # checkpoint load + container setup
BUFFER_MIN=10
MIN_WALL_MIN=30
CAP_MIN=719            # 11:59:00 — max for normal partition
DEFAULT_NEW_WALL="11:59:00"   # for ckpts with no progress yet

CKPT_BASE=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small

# Pick from snr_progress.py:
#   * [in_progress] with NO `jobs=` annotation (stuck) — sized walltime
#   * [not_submitted]                                 — 12h walltime
TARGETS=$(python3.11 scripts/snr_progress.py "${PROGRESS_ARGS[@]}" 2>&1 | awk '
    /\[in_progress\]/ && !/jobs=/ {
        status = "in_progress"
    }
    /\[not_submitted\]/ {
        status = "not_submitted"
    }
    /\[in_progress\]/ || /\[not_submitted\]/ {
        if (/\[in_progress\]/ && /jobs=/) next
        match($0, /([0-9]+)\/([0-9]+)/, m)
        done = m[1]; total = m[2]
        name = ""
        for (i = 1; i <= NF; i++) if ($i ~ /apertus|SmolLM|Olmo|Apertus/) { name = $i; break }
        if (name != "") print status, done "/" total, name
    }
' | sort -u)

if [[ -z "$TARGETS" ]]; then
    echo "No ckpts need launching — everything is either complete or already queued."
    exit 0
fi

submitted=0
skipped=0
while IFS=' ' read -r status progress name; do
    [[ -z "$name" ]] && continue

    size=""
    for s in 175M 350M 600M 1B; do
        if [[ "$name" == *"-$s-"* ]]; then size=$s; break; fi
    done
    if [[ -z "$size" ]]; then
        echo "skip $name (no size match — only Megatron apertus ckpts supported)"
        skipped=$((skipped + 1))
        continue
    fi

    iter=${name##*-iter}
    model_dir=${name%-iter*}
    ckpt_path="$CKPT_BASE/$model_dir/checkpoints"
    iter_dir="$ckpt_path/iter_$(printf '%07d' $iter)"
    if [[ ! -d "$iter_dir" ]]; then
        echo "skip $name (iter dir missing: $iter_dir)"
        skipped=$((skipped + 1))
        continue
    fi

    done_count=${progress%%/*}
    total_count=${progress##*/}
    remaining=$(( total_count - done_count ))

    if [[ "$status" == "not_submitted" || "$done_count" -eq 0 ]]; then
        walltime="$DEFAULT_NEW_WALL"
    else
        per_task=${PER_TASK_MIN[$size]}
        wall_min=$(( remaining * per_task + COLD_START_MIN + BUFFER_MIN ))
        (( wall_min < MIN_WALL_MIN )) && wall_min=$MIN_WALL_MIN
        (( wall_min > CAP_MIN ))      && wall_min=$CAP_MIN
        hh=$(( wall_min / 60 ))
        mm=$(( wall_min % 60 ))
        walltime=$(printf "%02d:%02d:00" $hh $mm)
    fi

    if (( DRY_RUN )); then
        res_note=""
        [[ -n "$RESERVATION" ]] && res_note=" reservation=$RESERVATION"
        echo "would submit  $walltime  $name  ($status, $progress done, $remaining left)$res_note"
        submitted=$((submitted + 1))
        continue
    fi

    JOB_ID=$(TASKS=./configs/signal_to_ratio/tasks_pretraining_full.txt \
             LM_EVAL_BACKEND=megatron_lm \
             TOKENIZER=alehc/swissai-tokenizer \
             BOS=true \
             APPLY_CHAT_TEMPLATE=false \
             WANDB_ENTITY=mariagrandury-epflnlp \
             WANDB_PROJECT=snr-experiments \
             sbatch --parsable \
                 --job-name "eval-$name" \
                 --time "$walltime" \
                 --partition=normal \
                 "${SBATCH_RES_ARG[@]}" \
                 --export=ALL,CKPT_ITER=$iter \
                 scripts/evaluate.sbatch "$ckpt_path" "$name") \
        || { echo "sbatch FAILED for $name"; skipped=$((skipped + 1)); continue; }
    echo "Submitted $JOB_ID  $name  ($walltime, $status, $progress done, $remaining left)"
    submitted=$((submitted + 1))
done <<< "$TARGETS"

echo ""
echo "Total submitted: $submitted ; skipped: $skipped"
