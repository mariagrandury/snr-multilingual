#!/bin/bash
# Run the remaining tasks of one checkpoint on every GPU of the node, each
# task's results landing on disk the moment it finishes.
#
# Starts EVAL_WORKERS copies of scripts/eval_worker.py, one per group of
# GPUS_PER_WORKER devices (pinned with CUDA_VISIBLE_DEVICES), all drawing from
# the same task queue: a worker claims a task, runs it against its once-loaded
# model and publishes per_task/<task>/ atomically (the protocol is in
# eval_worker.py). When all workers are done the per-task results are merged
# into one results_<ts>.json at the top of the eval dir and the samples files
# moved up beside it — the layout every reader (push_all_results.py,
# build_hf_dataset.py, _eval_status.py) already expects. The per-task results
# files stay in per_task/ (they carry each task's own timing and config).
#
# Idempotent across jobs: tasks with results in any earlier eval_*/ of this
# checkpoint (_eval_status.py) are skipped up front, so resubmitting after a
# walltime kill runs only what is missing.
#
# Reads from env (exported by evaluate.sbatch):
#   CMD_BASE          eval_worker.py invocation without --tasks / --output_path
#   TASKS             comma-separated task names
#   HARNESS_EVAL_DIR  this job's eval dir
#   NAME              checkpoint name (model-ckpt) — used for the status lookup
#   WANDB_ENTITY, WANDB_PROJECT, LOGS_ROOT   where the status lookup scans
#   EVAL_WORKERS      worker processes (default 1); GPUS_PER_WORKER each (default 1)
set -uo pipefail

PER_TASK_DIR="$HARNESS_EVAL_DIR/per_task"
INFLIGHT_DIR="$HARNESS_EVAL_DIR/inflight"
FAILED_LOG="$HARNESS_EVAL_DIR/failed_tasks.log"
SKIPPED_LOG="$HARNESS_EVAL_DIR/skipped_tasks.log"
EVAL_WORKERS=${EVAL_WORKERS:-1}
GPUS_PER_WORKER=${GPUS_PER_WORKER:-1}
mkdir -p "$PER_TASK_DIR" "$INFLIGHT_DIR"
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

mapfile -t TASKS_TO_RUN <<< "$REMAINING"
TASKS_CSV=$(IFS=,; echo "${TASKS_TO_RUN[*]}")
echo "=== ${#TASKS_TO_RUN[@]} task(s) on $EVAL_WORKERS worker(s) x $GPUS_PER_WORKER GPU(s), results per task in $PER_TASK_DIR ==="

# One process per worker, its stream prefixed and mirrored to worker_<i>.log.
# A single worker gets no CUDA_VISIBLE_DEVICES: under torchrun/accelerate
# (hf, megatron_lm) the launcher itself spans the node's GPUs.
for (( w=0; w<EVAL_WORKERS; w++ )); do
    if (( EVAL_WORKERS > 1 )); then
        gpus=$(seq -s, $(( w * GPUS_PER_WORKER )) $(( (w + 1) * GPUS_PER_WORKER - 1 )))
        CUDA_VISIBLE_DEVICES=$gpus $CMD_BASE --tasks "$TASKS_CSV" --output_path "$HARNESS_EVAL_DIR" \
            --worker "$w" --num_workers "$EVAL_WORKERS" 2>&1 \
            | sed -u "s/^/[w$w] /" | tee "$HARNESS_EVAL_DIR/worker_$w.log" &
    else
        $CMD_BASE --tasks "$TASKS_CSV" --output_path "$HARNESS_EVAL_DIR" \
            --worker 0 --num_workers 1 2>&1 | tee "$HARNESS_EVAL_DIR/worker_0.log" &
    fi
done
wait

# What landed is the truth, not the exit codes: every published task is a
# directory under per_task/, every failure a line in failed_tasks.log.
mapfile -t SUCCESS_DIRS < <(find "$PER_TASK_DIR" -mindepth 1 -maxdepth 1 -type d | sort)
if (( ${#SUCCESS_DIRS[@]} == 0 )); then
    echo "ERROR: no task finished; nothing to merge or upload." >&2
    exit 1
fi

echo "Merging ${#SUCCESS_DIRS[@]} finished task dir(s) into $HARNESS_EVAL_DIR"
python -m scripts.alignment.merge_split_results \
    --split_dirs "${SUCCESS_DIRS[@]}" \
    --output_dir "$HARNESS_EVAL_DIR" --move-samples \
    || echo "WARNING: merge failed (rc=$?) — the per-task results stay in $PER_TASK_DIR, where every reader also looks" >&2

# Flatten any sanitized-model subdir a CLI-style lm_eval run leaves at the top
# level (vLLM writes results to <output_path>/<sanitized_model_path>/), so
# downstream tooling can rely on a single shape. per_task/ and inflight/ are
# this script's own directories.
shopt -s nullglob
for inner in "$HARNESS_EVAL_DIR"/*/; do
    base=$(basename "$inner")
    [[ "$base" == "per_task" || "$base" == "inflight" ]] && continue
    if compgen -G "$inner"/results_*.json > /dev/null; then
        echo "Flattening lm_eval subdir: $base/"
        mv "$inner"/* "$HARNESS_EVAL_DIR"/ 2>/dev/null
        rmdir "$inner" 2>/dev/null || echo "  warn: $inner not empty after flatten"
    fi
done
shopt -u nullglob

# Any claimed task still sitting in inflight/ once every worker has exited was
# NOT published and NOT logged as failed: its worker died without raising —
# a CUDA abort, the OOM killer, a vLLM engine crash. Record it, or the watcher
# sees a run that made progress, never counts a failure against that task, and
# resubmits the checkpoint every pass forever (the 196-job pathology the
# --max-attempts gate exists to stop). The directory stays as evidence.
shopt -s nullglob
for d in "$INFLIGHT_DIR"/*/; do
    t=$(basename "$d")
    grep -q "^$t	" "$FAILED_LOG" 2>/dev/null \
        || printf '%s\tworker died without an exception (no results written)\n' "$t" >> "$FAILED_LOG"
done
shopt -u nullglob
rm -f "$INFLIGHT_DIR"/*.claim
rmdir "$INFLIGHT_DIR" 2>/dev/null \
    || echo "inflight/ kept: partial output of $(ls -1 "$INFLIGHT_DIR" | wc -l | tr -d ' ') task(s) whose worker died"

if [[ -s "$FAILED_LOG" ]]; then
    echo "WARNING: $(wc -l < "$FAILED_LOG") task(s) failed (see $FAILED_LOG):"
    sed 's/^/  - /' "$FAILED_LOG"
else
    rm -f "$FAILED_LOG"
fi

[[ -s "$SKIPPED_LOG" ]] || rm -f "$SKIPPED_LOG"
