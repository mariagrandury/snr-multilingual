#!/usr/bin/env bash
# Drive every (size, mix, seed) cell to iter 50000 in one pass:
#
#   [done]         → skip (already at target)
#   in squeue      → skip (job already queued/running for this cell)
#   [in_progress]  → submit resume (--load picks up the latest valid ckpt)
#   [corrupt]      → wipe checkpoints/ and submit a fresh from-scratch run
#   [no_ckpts] /
#   no exp dir     → submit a fresh from-scratch run
#
# Walltime is auto-computed per model from remaining iterations and per-size
# steady-state iter rates, with a 2h30m conservative margin (covers the 1h
# SIGUSR2 grace + cold-start + headroom). Override with --time HH:MM:SS.
#
# Idempotent: re-running is safe — already-active jobs are skipped via the
# canonical job name (apertus-<size>-edu<edu>-fw2<fw2>-seed<seed>).
#
# Usage:
#   bash launch_resumes.sh [--filter SUBSTR] [--time HH:MM:SS] [--dry-run]
#
# Examples:
#   bash launch_resumes.sh --dry-run
#   bash launch_resumes.sh --filter seed28
#   bash launch_resumes.sh --filter 175M --time 06:00:00

set -euo pipefail

FILTER_SUBSTR=""
FILTER_ARGS=()
TIME_OVERRIDE=""
DRY_RUN_ARGS=()
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --filter)   FILTER_ARGS=(--filter "$2"); FILTER_SUBSTR="$2"; shift 2 ;;
    --time)     TIME_OVERRIDE="$2"; shift 2 ;;
    --dry-run)  DRY_RUN_ARGS=(--dry-run); DRY_RUN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

PROGRESS=/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/pretrain_progress.py
LAUNCHER_DIR=/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain
CKPT_ROOT=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small
TARGET=50000

# Canonical 4×3×3 cross product.
SIZES=(175M 350M 600M 1B)
EDUS=(30 60 90)
SEEDS=(28 1797 1904)

# Per-size steady-state iter time (ms). Sampled from current training logs.
declare -A ITER_MS=( [175M]=800 [350M]=565 [600M]=520 [1B]=715 )

MARGIN_SEC=9000     # 2h30m: 1h SIGUSR2 grace + cold-start + buffer
MIN_SEC=5400        # 1h30m
MAX_SEC=43199       # 11:59:59 (slurm normal queue cap)

fmt_hms() {
  local s=$1
  printf "%02d:%02d:%02d" $((s/3600)) $(((s%3600)/60)) $((s%60))
}

auto_time() {
  # $1 = size, $2 = remaining iters
  local size=$1 remaining=$2 ms total
  ms=${ITER_MS[$size]:-800}
  total=$(( remaining * ms / 1000 + MARGIN_SEC ))
  total=$(( (total + 899) / 900 * 900 ))   # round UP to next 15 min
  (( total < MIN_SEC )) && total=$MIN_SEC
  (( total > MAX_SEC )) && total=$MAX_SEC
  fmt_hms "$total"
}

fw2_for_edu() {
  case $1 in
    30) echo 70 ;;
    60) echo 40 ;;
    90) echo 10 ;;
    *)  echo "unknown edu=$1" >&2; return 1 ;;
  esac
}

# launch_trainings.py uses Python 3.7+ syntax — the snr env has it
source /users/mariagrandury/miniconda3/etc/profile.d/conda.sh
conda activate snr

cd "$LAUNCHER_DIR"

# Snapshot of currently queued/running job names for this user.
ACTIVE_JOBS=$(squeue --user="$USER" -h --format="%j" 2>/dev/null | sort -u || true)

# Parse the progress dashboard once and build a map: model_name -> "STATUS<TAB>CUR_ITER".
#   STATUS ∈ {done, in_progress, corrupt, no_ckpts}
declare -A STATUS_OF
declare -A CUR_OF
while IFS= read -r line; do
  [[ "$line" != *"apertus-"* ]] && continue
  cur=$(awk '{print $2}' <<<"$line")
  model=$(awk '{ for (i=1;i<=NF;i++) if ($i ~ /^apertus-/) { print $i; exit } }' <<<"$line")
  [[ -z "$model" ]] && continue
  if   [[ "$line" == *"[done]"* ]];        then status=done
  elif [[ "$line" == *"[corrupt]"* ]];     then status=corrupt
  elif [[ "$line" == *"[no_ckpts]"* ]];    then status=no_ckpts
  elif [[ "$line" == *"[in_progress]"* ]]; then status=in_progress
  else continue
  fi
  STATUS_OF[$model]=$status
  CUR_OF[$model]=$cur
done < <(python3.11 "$PROGRESS" "${FILTER_ARGS[@]}" --target "$TARGET")

submit_one() {
  # $1 = size, $2 = edu, $3 = seed, $4 = remaining iters, $5 = label
  local size=$1 edu=$2 seed=$3 remaining=$4 label=$5 size_lc fw2 jobname tstr
  size_lc=$(printf '%s' "$size" | tr '[:upper:]' '[:lower:]')
  fw2=$(fw2_for_edu "$edu") || return
  jobname="apertus-${size_lc}-edu${edu}-fw2${fw2}-seed${seed}"

  if grep -Fxq "$jobname" <<<"$ACTIVE_JOBS"; then
    echo "    skip [$label]: $jobname already in squeue"
    return
  fi

  if [[ -n "$TIME_OVERRIDE" ]]; then
    tstr="$TIME_OVERRIDE"
  else
    tstr=$(auto_time "$size" "$remaining")
  fi
  echo ">>> [$label] apertus-${size}-fwEdu${edu}-fw2${fw2}-seed${seed}  remaining=${remaining}  --time=${tstr}"
  python launch_trainings.py --size "$size" --mix_en "$edu" --seed "$seed" \
      --time "$tstr" "${DRY_RUN_ARGS[@]}"
}

wipe_checkpoints() {
  # $1 = model dir name
  local ckpt_dir="$CKPT_ROOT/$1/checkpoints"
  if (( DRY_RUN )); then
    echo "    (dry-run) rm -rf '$ckpt_dir'"
  elif [[ -d "$ckpt_dir" ]]; then
    rm -rf "$ckpt_dir"
  fi
}

# Iterate the canonical 36 cells.
for size in "${SIZES[@]}"; do
  for edu in "${EDUS[@]}"; do
    fw2=$(fw2_for_edu "$edu")
    for seed in "${SEEDS[@]}"; do
      model="apertus-${size}-fwEdu${edu}-fw2${fw2}-seed${seed}"
      [[ -n "$FILTER_SUBSTR" && "$model" != *"$FILTER_SUBSTR"* ]] && continue

      status=${STATUS_OF[$model]:-no_dir}
      cur=${CUR_OF[$model]:-0}

      case $status in
        done)
          continue
          ;;
        in_progress)
          remaining=$(( TARGET - cur ))
          submit_one "$size" "$edu" "$seed" "$remaining" "resume"
          ;;
        corrupt)
          echo "*** corrupt: $model — wiping checkpoints and starting fresh"
          wipe_checkpoints "$model"
          submit_one "$size" "$edu" "$seed" "$TARGET" "fresh"
          ;;
        no_ckpts|no_dir)
          submit_one "$size" "$edu" "$seed" "$TARGET" "fresh"
          ;;
      esac
    done
  done
done
