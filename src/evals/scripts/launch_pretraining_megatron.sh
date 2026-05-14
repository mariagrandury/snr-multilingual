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
# Checkpoint paths are no longer hardcoded — each model's megatron checkpoint
# dir comes from configs/models.json `backends.megatron` (so custom apertus
# AND a06 main-run cells both resolve correctly). Pick the a06 pool with
# `POOL=pretraining_a06 bash scripts/launch_pretraining_megatron.sh`.
POOL=${POOL:-seeds_28_1797_1904}        # default pool — all Apertus custom seeds
TASKS_GROUP=${TASKS_GROUP:-pretraining_full}
CSV=$REPO_DIR/snr_progress.csv

if (( REFRESH )); then
    echo "[meg] refreshing $CSV ..."
    python3.11 scripts/snr_progress.py \
        --pool "$POOL" \
        --tasks-group "$TASKS_GROUP" \
        > /dev/null
fi
[[ -f "$CSV" ]] || { echo "ERROR: $CSV missing (run without --no-refresh)" >&2; exit 1; }

# Megatron eval is ~2x slower than vLLM at same model size (no batched
# kv-cache reuse, more all-reduce traffic). Estimates from observed runs;
# 3B is an estimate (no observed a06-3B run yet), `*` is a safe default.
per_task_min()  { case "$1" in 175M) echo 4;; 350M) echo 6;; 600M) echo 8;; 1B) echo 10;; 3B) echo 16;; *) echo 10;; esac; }
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

# Resolve model / size / megatron-checkpoint dir per CSV row from
# configs/models.json — no name-regex parsing, so custom apertus AND a06
# main-run cells both work. Emits the original 6 CSV columns plus
# model, size, ckpt_path, iter, \x1f-delimited (non-whitespace separator
# so empty fields like active_jobids survive the bash `read`). Same
# iter-desc / size-desc submit order as the HF launcher.
SORTED=$(awk -F, 'NR>1 { print $0 }' "$CSV" | python3.11 -c "
import sys, csv
sys.path.insert(0, 'scripts/utils')
from configs import get_model
size_rank = {'3B': 0, '1B': 1, '600M': 2, '350M': 3, '175M': 4}
out = []
for r in csv.reader(sys.stdin):
    if len(r) < 6:
        continue
    model, sep, it = r[0].rpartition('-iter')
    if not sep or not it.isdigit():
        print('  WARN: name is not <model>-iter<N>: ' + r[0], file=sys.stderr)
        continue
    try:
        e = get_model(model)
    except KeyError:
        print('  WARN: model not in models.json: ' + model, file=sys.stderr)
        continue
    ckpt = (e.get('backends') or {}).get('megatron')
    if not ckpt:
        print('  WARN: no megatron backend for ' + model, file=sys.stderr)
        continue
    out.append((r, model, e['size'], ckpt.rstrip('/'), int(it)))
out.sort(key=lambda x: (-x[4], size_rank.get(x[2], 9)))
for r, model, size, ckpt, it in out:
    print('\x1f'.join([r[0], r[1], r[2], r[3], r[4], r[5],
                       model, size, ckpt, str(it)]))
")

submitted=0; skipped_active=0; skipped_done=0; skipped_no_ckpt=0; skipped_filter=0
echo "[meg] partition=$PARTITION dry_run=$DRY_RUN filter=${FILTER:-<none>}"
echo ""

while IFS=$'\x1f' read -r name status done total remaining active_jobids model size ckpt_path iter; do
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

    # model / size / ckpt_path / iter were resolved from configs/models.json
    # by the python sort block above — works for custom apertus + a06 alike.
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
