#!/bin/bash
# Launch a focused eval subset on `normal` (350M/600M/1B) and `debug` (175M).
#
# Scope:
#   * Iters     : 6000, 28000, 50000
#   * Tasks     : configs/signal_to_ratio/tasks_pretraining_full.txt (current trim)
#   * seed1797  : all 4 sizes × all 3 mixes (edu30/60/90)
#   * seed28    : only edu90 × all 4 sizes
#   * Excluded  : a few (cell, iter) entries already evaluated elsewhere
#                 (see EXCLUDE list below). Plus any cell whose ckpt isn't on disk.
#
# Per-size optimized parallelism (uses all 4 GPUs/node, see CLAUDE.md bug 14):
#   * 175M  TP=4 PP=1   (kv=4 → TP=4 OK)
#   * 350M  TP=1 PP=4   (kv=5 → only TP=1; PP=4 fills the node)
#   * 600M  TP=2 PP=2   (kv=6 allows TP=2; PP=2 to keep all 4 GPUs busy)
#   * 1B    TP=1 PP=4   (kv=7 → only TP=1; PP=4 fills the node)
#
# Per-size partition + walltime:
#   * 175M  debug  01:25:00   (debug 1h30m cap; debug-qos allows max 2 active per user)
#   * 350M  normal 01:30:00
#   * 600M  normal 02:00:00
#   * 1B    normal 02:30:00
#
# Queue order: outer = iter desc (50k → 28k → 6k); inner = size desc
# (1B → 600M → 350M → 175M). 175M is throttled to debug-qos (2 active) — the
# script blocks before each debug submission until a slot is free, so once the
# 175M phase starts the script becomes long-running. To keep the shell free:
#   nohup bash scripts/launch_eval_subset.sh > launch_eval_subset.log 2>&1 &
#
# Usage:
#   bash scripts/launch_eval_subset.sh           # submit
#   bash scripts/launch_eval_subset.sh --dry-run # show what would be submitted

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

HF_BASE=/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints
TASKS_FILE=$PWD/configs/signal_to_ratio/tasks_pretraining_full.txt
ITERS=(6000 28000 50000)

# Hard-coded (cell, iter) exclusions — already evaluated or otherwise covered.
# Format: full job NAME ("apertus-<size>-fwEdu<edu>-fw2<fw2>-seed<seed>-iter<iter>")
EXCLUDE=(
  "apertus-175M-fwEdu90-fw210-seed28-iter50000"
  "apertus-175M-fwEdu90-fw210-seed1797-iter28000"
  "apertus-175M-fwEdu90-fw210-seed1797-iter50000"
)
is_excluded() {
  local n=$1 e
  for e in "${EXCLUDE[@]}"; do [[ "$n" == "$e" ]] && return 0; done
  return 1
}

# Build the (size, seed, edu) matrix.
declare -a CELLS=()
for size in 175M 350M 600M 1B; do
  # seed1797: all 3 mixes
  for edu in 30 60 90; do CELLS+=("${size}|1797|${edu}"); done
  # seed28: only edu90
  CELLS+=("${size}|28|90")
done

# Per-size config.
tp_for()        { case "$1" in 175M) echo 4;; 350M) echo 1;; 600M) echo 2;; 1B) echo 1;; esac; }
pp_for()        { case "$1" in 175M) echo 1;; 350M) echo 4;; 600M) echo 2;; 1B) echo 4;; esac; }
time_for()      { case "$1" in 175M) echo 01:25:00;; 350M) echo 01:30:00;; 600M) echo 02:00:00;; 1B) echo 02:30:00;; esac; }
partition_for() { case "$1" in 175M) echo debug;; *) echo normal;; esac; }
fw2_for()       { case "$1" in 30) echo 70;; 60) echo 40;; 90) echo 10;; esac; }

# debug-qos allows max 1 running + 1 queued per user. Block before each
# debug submission until at most 1 of our eval-* debug jobs is active (so
# the new submission fills the second slot).
debug_active() {
  squeue -u "$USER" -h --format="%j %P" 2>/dev/null \
    | awk '$2=="debug" && $1 ~ /^eval-/ {n++} END {print n+0}'
}
wait_for_debug_slot() {
  while [[ "$(debug_active)" -ge 2 ]]; do sleep 60; done
}

# Snapshot of currently-queued eval-* job names (refreshed once at script
# start); used to skip resubmissions when re-running.
ACTIVE_JOBS=$(squeue -u "$USER" -h --format="%j" 2>/dev/null | sort -u)
is_active() { grep -Fxq "eval-$1" <<<"$ACTIVE_JOBS"; }

submit_one() {
  local size=$1 seed=$2 edu=$3 iter=$4
  local fw2 cell name iter_dir tp pp wall part
  fw2=$(fw2_for "$edu")
  cell="apertus-${size}-fwEdu${edu}-fw2${fw2}-seed${seed}"
  name="${cell}-iter${iter}"
  if is_excluded "$name"; then
    echo "  SKIP (excluded): ${name}"
    return 1
  fi
  if is_active "$name"; then
    echo "  SKIP (already queued): ${name}"
    return 1
  fi
  iter_dir=$(printf "%s/%s/iter_%07d" "$HF_BASE" "$cell" "$iter")
  if [[ ! -f "$iter_dir/config.json" ]] || ! ls "$iter_dir"/model.safetensors* >/dev/null 2>&1; then
    echo "  SKIP (no ckpt): ${name}"
    return 1
  fi
  tp=$(tp_for "$size")
  pp=$(pp_for "$size")
  wall=$(time_for "$size")
  part=$(partition_for "$size")
  if (( DRY_RUN )); then
    echo "  would submit: ${name}  TP=${tp} PP=${pp}  --time=${wall}  --partition=${part}"
    return 0
  fi
  if [[ "$part" == "debug" ]]; then
    wait_for_debug_slot
  fi
  local jid
  jid=$(sbatch --parsable \
    --job-name="eval-${name}" \
    --partition="$part" \
    --time="$wall" \
    --export=ALL,LM_EVAL_BACKEND=vllm,TOKENIZER=alehc/swissai-tokenizer,BOS=true,APPLY_CHAT_TEMPLATE=false,BATCH_TASKS=1,TP=$tp,PP=$pp,WANDB_ENTITY=mariagrandury-epflnlp,WANDB_PROJECT=snr-experiments,TASKS=$TASKS_FILE \
    scripts/evaluate.sbatch "$iter_dir" "$name") \
    && echo "  ${jid}  ${name}  TP=${tp} PP=${pp}  ${wall}  ${part}"
}

submit=0; skip=0
# Outer loop: iter descending (50k → 28k → 6k).
# Inner loop: size descending (1B → 600M → 350M → 175M).
for iter in 50000 28000 6000; do
  echo "=== iter ${iter} ==="
  for size in 1B 600M 350M 175M; do
    for entry in "${CELLS[@]}"; do
      IFS='|' read -r s seed edu <<<"$entry"
      [[ "$s" == "$size" ]] || continue
      if submit_one "$size" "$seed" "$edu" "$iter"; then
        submit=$((submit+1))
      else
        skip=$((skip+1))
      fi
    done
  done
  echo ""
done

echo "submitted=$submit  skipped=$skip"
