#!/usr/bin/env bash
# Drive every (size, mix, seed) cell to having all canonical checkpoints saved.
#
# Per-model action comes from `pretrain_progress.py --actions`:
#
#   done                              → skip (all canonical iters ≤ target valid)
#   fresh   <target>                  → submit a fresh from-scratch run to <target>
#   corrupt <n_iters>                 → SKIP + warn (iter dirs exist but none valid;
#                                         we never auto-rm checkpoints — manual review)
#   resume  <load_iter> <target>      → submit a resume to <target>; if <load_iter>
#                                         is below the current latest_checkpointed_
#                                         iteration.txt marker, the marker is rewound
#                                         to <load_iter> first (mid-gap backfill).
#
# Walltime is auto-computed per model from remaining iterations and per-size
# steady-state iter rates, with a 2h30m conservative margin (covers the 1h
# SIGUSR2 grace + cold-start + headroom). Override with --time HH:MM:SS.
#
# Idempotent: re-runnable. Already-active jobs are skipped via the canonical
# job name (apertus-<size>-edu<edu>-fw2<fw2>-seed<seed>); we never rm anything.
#
# Usage:
#   bash launch_resumes.sh [--filter SUBSTR] [--time HH:MM:SS] [--target N] [--dry-run]
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
TARGET=50000
while [[ $# -gt 0 ]]; do
  case $1 in
    --filter)   FILTER_ARGS=(--filter "$2"); FILTER_SUBSTR="$2"; shift 2 ;;
    --time)     TIME_OVERRIDE="$2"; shift 2 ;;
    --target)   TARGET="$2"; shift 2 ;;
    --dry-run)  DRY_RUN_ARGS=(--dry-run); DRY_RUN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

LAUNCHER_DIR=/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain
PROGRESS=$LAUNCHER_DIR/pretrain_progress.py
CKPT_ROOT=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small

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

# Parse a canonical model name into (size, edu, seed). Echoes them
# space-separated; returns non-zero on no match.
parse_model() {
  local model=$1
  if [[ "$model" =~ ^apertus-(175M|350M|600M|1B)-fwEdu([0-9]+)-fw2[0-9]+-seed([0-9]+)$ ]]; then
    printf '%s %s %s\n' "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}" "${BASH_REMATCH[3]}"
    return 0
  fi
  return 1
}

# Read marker file (latest_checkpointed_iteration.txt). Echoes "" if absent
# or unparseable.
read_marker() {
  local f=$1
  [[ -f "$f" ]] || { echo ""; return; }
  local raw
  raw=$(<"$f") || { echo ""; return; }
  raw=${raw//[$'\t\r\n ']/}
  if [[ "$raw" =~ ^[0-9]+$ ]]; then
    echo "$raw"
  else
    echo ""
  fi
}

# launch_trainings.py uses Python 3.7+ syntax — the snr env has it.
source /users/mariagrandury/miniconda3/etc/profile.d/conda.sh
conda activate snr

cd "$LAUNCHER_DIR"

# Snapshot of currently queued/running job names for this user.
ACTIVE_JOBS=$(squeue --user="$USER" -h --format="%j" 2>/dev/null | sort -u || true)

submit_one() {
  # $1 = size, $2 = edu, $3 = seed, $4 = remaining iters,
  # $5 = label, $6 = target iter (passed as --training-steps if != 50000)
  local size=$1 edu=$2 seed=$3 remaining=$4 label=$5 tgt=$6
  local size_lc fw2 jobname tstr extra_args=()
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
  if [[ "$tgt" != "50000" ]]; then
    extra_args+=(--training-steps "$tgt")
  fi
  echo ">>> [$label] apertus-${size}-fwEdu${edu}-fw2${fw2}-seed${seed}  remaining=${remaining}  target=${tgt}  --time=${tstr}"
  python launch_trainings.py --size "$size" --mix_en "$edu" --seed "$seed" \
      --time "$tstr" "${extra_args[@]}" "${DRY_RUN_ARGS[@]}"
}

rewind_marker() {
  # $1 = model dir name, $2 = desired marker iter
  local model=$1 want=$2
  local ckpt_dir="$CKPT_ROOT/$model/checkpoints"
  local marker_file="$ckpt_dir/latest_checkpointed_iteration.txt"
  local cur
  cur=$(read_marker "$marker_file")
  if [[ "$cur" == "$want" ]]; then
    return 0
  fi
  if (( DRY_RUN )); then
    echo "    (dry-run) would rewind $marker_file: ${cur:-<missing>} → $want"
    return 0
  fi
  # Sanity check: the iter dir we're pointing at must exist and contain a
  # .distcp shard. If it doesn't, refuse to rewind — abort the model.
  local iter_dir
  iter_dir=$(printf '%s/iter_%07d' "$ckpt_dir" "$want")
  if [[ ! -f "$iter_dir/.metadata" ]] || ! ls "$iter_dir"/*.distcp >/dev/null 2>&1; then
    echo "    !! refusing to rewind marker for $model: $iter_dir is not a valid checkpoint" >&2
    return 1
  fi
  printf '%s\n' "$want" > "$marker_file"
  echo "    rewound marker for $model: ${cur:-<missing>} → $want"
}

# Pull per-model actions from pretrain_progress.py (one tab-separated line each).
# Format: <model>\tdone | fresh\t<target> | corrupt\t<n_iters> |
#         resume\t<load_iter>\t<target>
ACTIONS_OUT=$(python3.11 "$PROGRESS" "${FILTER_ARGS[@]}" --target "$TARGET" --actions)

while IFS=$'\t' read -r model action a b _rest; do
  [[ -z "${model:-}" ]] && continue
  [[ -n "$FILTER_SUBSTR" && "$model" != *"$FILTER_SUBSTR"* ]] && continue

  if ! read -r size edu seed <<<"$(parse_model "$model")"; then
    echo "    skip: cannot parse model name '$model'" >&2
    continue
  fi

  case "$action" in
    done)
      continue
      ;;
    corrupt)
      n_iters=${a:-?}
      echo "*** corrupt: $model — $n_iters iter dir(s) on disk but none valid; SKIPPING (manual review)"
      ;;
    fresh)
      tgt=${a:-$TARGET}
      submit_one "$size" "$edu" "$seed" "$tgt" "fresh" "$tgt"
      ;;
    resume)
      load_iter=${a:-0}
      tgt=${b:-$TARGET}
      remaining=$(( tgt - load_iter ))
      (( remaining < 0 )) && remaining=0

      # If load_iter is below the current marker, this is a mid-gap backfill:
      # rewind the marker so Megatron resumes from the right iter on next load.
      ckpt_dir="$CKPT_ROOT/$model/checkpoints"
      cur_marker=$(read_marker "$ckpt_dir/latest_checkpointed_iteration.txt")
      if [[ -n "$cur_marker" && "$cur_marker" != "$load_iter" && "$cur_marker" -gt "$load_iter" ]]; then
        # Don't rewind if a job is already queued for this cell — let it run.
        size_lc=$(printf '%s' "$size" | tr '[:upper:]' '[:lower:]')
        fw2=$(fw2_for_edu "$edu")
        jobname="apertus-${size_lc}-edu${edu}-fw2${fw2}-seed${seed}"
        if grep -Fxq "$jobname" <<<"$ACTIVE_JOBS"; then
          echo "    skip [resume]: $jobname already in squeue (no rewind)"
          continue
        fi
        rewind_marker "$model" "$load_iter" || continue
        submit_one "$size" "$edu" "$seed" "$remaining" "resume-mid" "$tgt"
      else
        submit_one "$size" "$edu" "$seed" "$remaining" "resume" "$tgt"
      fi
      ;;
    *)
      echo "    skip: unknown action '$action' for $model" >&2
      ;;
  esac
done <<<"$ACTIONS_OUT"
