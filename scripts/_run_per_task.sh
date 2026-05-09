#!/bin/bash
# Run lm_eval once per task with fault tolerance: a failure in one task is
# logged and the loop continues so survivor metrics still reach W&B.
# Idempotent: tasks whose results already exist on disk (in any sibling
# eval_*/ run for this checkpoint) are skipped, so re-launching the same
# command after a partial run / timeout doesn't redo finished work.
#
# Reads from env (exported by evaluate.sbatch):
#   CMD_BASE          lm_eval invocation without --tasks / --output_path
#   TASKS             comma-separated task names
#   HARNESS_EVAL_DIR  base output dir; per-task subdirs are merged into here
#   NAME              checkpoint name (model-ckpt) — used for status lookup
#   WANDB_ENTITY      W&B entity
#   WANDB_PROJECT     W&B project
#   LOGS_ROOT         eval_logs root (optional, has a sane default)
set -uo pipefail

PER_TASK_DIR="$HARNESS_EVAL_DIR/per_task"
FAILED_LOG="$HARNESS_EVAL_DIR/failed_tasks.log"
SKIPPED_LOG="$HARNESS_EVAL_DIR/skipped_tasks.log"
mkdir -p "$PER_TASK_DIR"
: > "$FAILED_LOG"
: > "$SKIPPED_LOG"

# Filter out tasks that already have results in any prior eval_*/ run.
# _eval_status.py contract: exit 0 + non-empty stdout = tasks remaining,
# exit 1 = all done, exit 2+ = crash. Don't conflate the crash case with
# "all done" — fall back to running everything if the filter misbehaves.
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
set +e
REMAINING=$(python3 "$REPO_DIR/scripts/_eval_status.py" \
    --name "$NAME" --tasks "$TASKS" \
    --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" 2>/dev/null)
EVAL_STATUS_RC=$?
set -uo pipefail

if [[ $EVAL_STATUS_RC -eq 1 ]]; then
    echo "All tasks for $NAME already have results — nothing to do."
    exit 0
elif [[ $EVAL_STATUS_RC -ne 0 ]]; then
    echo "WARNING: _eval_status.py crashed (rc=$EVAL_STATUS_RC); running all tasks without filtering."
    REMAINING=$(echo "$TASKS" | tr ',' '\n')
elif [[ -z "$REMAINING" ]]; then
    echo "All tasks for $NAME already have results — nothing to do."
    exit 0
fi

# Log which tasks we're skipping vs running
IFS=',' read -ra _ALL_TASKS_ARR <<< "$TASKS"
declare -A _REMAINING_SET
while IFS= read -r t; do
    [[ -n "$t" ]] && _REMAINING_SET["$t"]=1
done <<< "$REMAINING"
for t in "${_ALL_TASKS_ARR[@]}"; do
    if [[ -z "${_REMAINING_SET[$t]:-}" ]]; then
        echo "$t" >> "$SKIPPED_LOG"
    fi
done

if [[ -s "$SKIPPED_LOG" ]]; then
    echo "Skipping $(wc -l < "$SKIPPED_LOG") task(s) with existing results:"
    sed 's/^/  - /' "$SKIPPED_LOG"
fi

SUCCESS_DIRS=()
mapfile -t TASKS_TO_RUN <<< "$REMAINING"

# BATCH_TASKS=1 (set by the vLLM/HF SNR runner): run ALL remaining tasks in
# ONE lm_eval invocation. vLLM model load is the dominant per-task cost
# (~30-90s) and dwarfs actual inference on small models, so a single call
# ~10x's per-ckpt throughput. lm_eval handles per-task errors internally
# (logs and moves on within the run). The single results_*.json is recognized
# by _eval_status.py's `.results` listing, so per-task idempotency on re-runs
# still works. Trade-off: a mid-run crash loses all unwritten results in
# that call (vs the per-task loop's "lose only the in-progress task").
if [[ "${BATCH_TASKS:-0}" == "1" ]]; then
    TASKS_CSV=$(IFS=,; echo "${TASKS_TO_RUN[*]}")
    echo "=== Running ${#TASKS_TO_RUN[@]} tasks in a single lm_eval call (BATCH_TASKS=1) ==="
    if $CMD_BASE --tasks "$TASKS_CSV" --output_path "$HARNESS_EVAL_DIR"; then
        SUCCESS_DIRS+=("$HARNESS_EVAL_DIR")
        echo "=== OK: batch of ${#TASKS_TO_RUN[@]} tasks ==="
    else
        rc=$?
        printf '%s\n' "${TASKS_TO_RUN[@]}" >> "$FAILED_LOG"
        echo "=== FAILED (rc=$rc): batch lm_eval call — see $FAILED_LOG ===" >&2
    fi
else
    for t in "${TASKS_TO_RUN[@]}"; do
        [[ -z "$t" ]] && continue
        out="$PER_TASK_DIR/$t"
        echo "=== Running task: $t ==="
        if $CMD_BASE --tasks "$t" --output_path "$out"; then
            SUCCESS_DIRS+=("$out")
            echo "=== OK: $t ==="
        else
            rc=$?
            echo "$t" >> "$FAILED_LOG"
            echo "=== FAILED (rc=$rc): $t — logged and continuing ===" >&2
        fi
    done
fi

if (( ${#SUCCESS_DIRS[@]} == 0 )); then
    echo "ERROR: every attempted task failed; nothing to merge or upload." >&2
    exit 1
fi

# In BATCH_TASKS mode the single lm_eval call already wrote results_*.json
# directly into $HARNESS_EVAL_DIR, so the per-task merge step is unnecessary
# (and would be a no-op anyway since SUCCESS_DIRS == [$HARNESS_EVAL_DIR]).
if [[ "${BATCH_TASKS:-0}" != "1" ]]; then
    echo "Merging ${#SUCCESS_DIRS[@]} successful task dirs into $HARNESS_EVAL_DIR"
    python -m scripts.alignment.merge_split_results \
        --split_dirs "${SUCCESS_DIRS[@]}" \
        --output_dir "$HARNESS_EVAL_DIR"
    rm -rf "$PER_TASK_DIR"
fi

# Flatten any sanitized-model subdir that lm_eval's vLLM backend creates.
# vLLM writes results to <output_path>/<sanitized_model_path>/, while the
# megatron_lm backend writes them directly to <output_path>/. Flattening
# here unifies the two layouts so downstream tooling (push_all_results.py,
# build_hf_dataset.py, _eval_status.py) can rely on a single shape.
shopt -s nullglob
for inner in "$HARNESS_EVAL_DIR"/*/; do
    base=$(basename "$inner")
    [[ "$base" == "per_task" ]] && continue
    # Only flatten if it actually contains a results_*.json (defensive guard).
    if compgen -G "$inner"/results_*.json > /dev/null; then
        echo "Flattening lm_eval subdir: $base/"
        mv "$inner"/* "$HARNESS_EVAL_DIR"/ 2>/dev/null
        rmdir "$inner" 2>/dev/null || echo "  warn: $inner not empty after flatten"
    fi
done
shopt -u nullglob

if [[ -s "$FAILED_LOG" ]]; then
    echo "WARNING: $(wc -l < "$FAILED_LOG") task(s) failed (see $FAILED_LOG):"
    sed 's/^/  - /' "$FAILED_LOG"
else
    rm -f "$FAILED_LOG"
fi

[[ -s "$SKIPPED_LOG" ]] || rm -f "$SKIPPED_LOG"
