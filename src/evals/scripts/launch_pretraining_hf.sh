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
# Per-cell hf_local dir is no longer hardcoded — each model's converted-HF
# checkpoint dir is read from configs/models.json `backends.hf_local`, so
# custom apertus AND a06 (and any other model) work without a name-regex.
# Pick the a06 pool with `POOL=pretraining_a06 bash scripts/launch_pretraining_hf.sh`.
POOL=${POOL:-seeds_28_1797_1904}        # default pool — all Apertus custom seeds
TASKS_GROUP=${TASKS_GROUP:-pretraining_full}
CSV=$REPO_DIR/snr_progress.csv

# Posttraining tasks (ifeval/gsm8k_cot/humaneval_instruct/…) need a chat
# template; pretraining/midtraining base evals don't.
APPLY_CHAT=false
[[ "$TASKS_GROUP" == posttraining* ]] && APPLY_CHAT=true

# Refresh the snapshot once at script start. snr_progress.py reads pool
# membership + per-cell iter lists from configs/models.json, so the
# on-disk file stays consistent for both humans and launchers.
if (( REFRESH )); then
    echo "[hf] refreshing $CSV ..."
    python3.11 scripts/snr_progress.py \
        --pool "$POOL" \
        --tasks-group "$TASKS_GROUP" \
        > /dev/null
fi
[[ -f "$CSV" ]] || { echo "ERROR: $CSV missing (run without --no-refresh)" >&2; exit 1; }

# TP/PP picked per CLAUDE.md bug 14 (vLLM kv_heads constraint).
# Custom apertus kv heads: 175M=4, 350M=5, 600M=6, 1B=7.
# a06 apertus3-1b-21-nodes kv=8 (TP=4 OK, all 4 GPUs via TP).
# a06 apertus3-3b-64-nodes kv unknown until first conversion lands — safe TP=1 PP=4.
# Distill ap-from8b-TOP256: 0.6b kv=4, 1b kv=8 — both fit TP=4 cleanly.
# Per-model overrides take precedence over the size-based defaults.
tp_for() {
    local size=$1 model=${2:-}
    case "$model" in
        apertus3-1b-21-nodes) echo 4; return;;
        apertus3-3b-64-nodes) echo 1; return;;
        apertus-0.6b-from8b-TOP256-long) echo 4; return;;
        apertus-1b-from8b-TOP256-long)   echo 4; return;;
    esac
    case "$size" in 175M) echo 4;; 350M) echo 1;; 600M) echo 2;; 1B) echo 1;; *) echo 1;; esac
}
pp_for() {
    local size=$1 model=${2:-}
    case "$model" in
        apertus3-1b-21-nodes) echo 1; return;;
        apertus3-3b-64-nodes) echo 4; return;;
        apertus-0.6b-from8b-TOP256-long) echo 1; return;;
        apertus-1b-from8b-TOP256-long)   echo 1; return;;
    esac
    case "$size" in 175M) echo 1;; 350M) echo 4;; 600M) echo 2;; 1B) echo 4;; *) echo 4;; esac
}
# Hub (vLLM) path: vLLM rejects offline DP>1 for dense models (CLAUDE bug 11),
# so we fill the 4-GPU node with tensor parallelism instead; PP>1 adds pipeline
# bubbles AND crashes some archs with `'PPMissingLayer' has no attribute
# attention_type`. So PP=1 and TP = largest of {4,2,1} dividing the model's
# num_key_value_heads (CLAUDE bug 14: TP must divide kv-heads). Reads the cached
# HF config; falls back to TP=1 if heads can't be read (e.g. gemma-3-4b nests them).
tp_from_kv() {
    HF_HUB_CACHE="${HF_HUB_CACHE:-/capstor/store/cscs/swissai/infra01/users/mariagrandury/hf_models}" \
    python3.11 - "$1" <<'PY'
import json, glob, os, sys
g = glob.glob(f"{os.environ['HF_HUB_CACHE']}/models--{sys.argv[1].replace('/','--')}/snapshots/*/config.json")
kv = None
if g:
    tc = json.load(open(g[0])); tc = tc.get("text_config", tc)
    kv = tc.get("num_key_value_heads", tc.get("num_attention_heads"))
print(next((t for t in (4, 2, 1) if isinstance(kv, int) and kv % t == 0), 1))
PY
}
# Per-size per-task minute estimates (re-fit 2026-05-10 after the iter50000
# batch hit 13/36 TIMEOUT on the new 52-task pretraining-full mix). The flat
# 0.5 min/task estimate worked for 175M but undershot for larger sizes where
# generation cost grows: 1B is ~3x slower than 175M, 600M/350M ~2x. 3B is
# extrapolated (no a06 vLLM eval observed yet).
# min/task (LOGPROB baseline), calibrated from REAL completed-eval walltimes
# (sacct, 2026-05/06 full ~120-task pretraining runs): pure per-task (total minus
# ~18m cold) ≈ 0.6 (≤1.7B), 0.9 (4B), 1.2 (7B), 1.4 (12B). Values below add a
# ~1.3x safety margin and extrapolate 24-70B. Posttraining is generative
# (autoregressive) and slower — see GEN_MULT in walltime_for (estimate, no
# completed posttraining-full run to calibrate against yet).
per_task_min()  { case "$1" in
    175M|350M|600M|270M|0.6B|1B|1.7B) echo 1.0;;
    3B|4B)        echo 1.3;;
    7B|8B|30B-A3B) echo 1.7;;   # 30B-A3B is MoE, ~3B active compute
    13B|14B)      echo 2.0;;
    24B|27B|32B)  echo 3.0;;
    70B)          echo 5.0;;
    *)            echo 2.0;;
  esac; }
# cold start = pip install + vLLM init + dataset load + shard-load across 4 GPUs.
# Real small full evals ran ~83-110m/120 tasks ⇒ ~18m cold; bigger models load slower.
cold_start_for() { case "$1" in 70B) echo 35;; 24B|27B|32B|30B-A3B) echo 25;; *) echo 18;; esac; }
MIN_WALL_MIN=25     # floor — even 1-task jobs need vLLM init time
CAP_MIN=719         # 11:59:00 — normal-partition wall cap

walltime_for() {
    local size=$1 remaining=$2
    local pt; pt=$(per_task_min "$size")
    local c; c=$(cold_start_for "$size")
    # Generative groups (posttraining/midtraining: mgsm/ifeval/humaneval CoT) run
    # far slower per task than logprob. ESTIMATE (2.5x) — refine once a full
    # posttraining sweep completes and we can re-read sacct.
    local mult=1
    [[ "$TASKS_GROUP" == posttraining* || "$TASKS_GROUP" == midtraining* ]] && mult=2.5
    # all_stages = pretraining_full ∪ mid ∪ post (213 tasks, ~120 logprob + ~93
    # generative) → blended factor between the logprob (1) and generative (2.5).
    [[ "$TASKS_GROUP" == all_stages ]] && mult=1.8
    local m=$(awk -v c=$c -v p=$pt -v n=$remaining -v g=$mult \
                  'BEGIN { printf "%d\n", c + n * p * g + 0.999 }')
    (( m < MIN_WALL_MIN )) && m=$MIN_WALL_MIN
    (( m > CAP_MIN ))      && m=$CAP_MIN
    printf "%02d:%02d:00" $(( m / 60 )) $(( m % 60 ))
}

SBATCH_RES_ARGS=()
[[ -n "$RESERVATION" ]] && SBATCH_RES_ARGS=(--reservation="$RESERVATION")

# Resolve model / size / checkpoint-source per CSV row from
# configs/models.json. Dispatches on `checkpoint_kind`:
#   * hf_branch  → mode=hub:   evaluate the HF-Hub repo at REVISION=<branch>
#                  (Qwen / OLMo / gemma / aya reference models). No local
#                  iter dir; the model's own tokenizer is used.
#   * else       → mode=local: evaluate the converted snapshot under
#                  backends.hf_local/iter_NNNNNNN (apertus / a06 / distill).
# NAME → (model, ckpt) by longest models.json-key prefix, so neither the
# `-iter<N>` nor the `-<branch>` form needs a regex. Emits the 6 CSV columns
# plus model, size, mode, path, branch, iter (\x1f-delimited). Sort: iter
# desc then size desc (hub rows have iter=-1 → sorted last, by size).
SORTED=$(awk -F, 'NR>1 { print $0 }' "$CSV" | python3.11 -c "
import sys, csv
sys.path.insert(0, 'scripts/utils')
from configs import get_model, load_models
MODELS = load_models()
size_rank = {'70B': -2, '32B': -1, '30B-A3B': -1, '3B': 0, '1B': 1,
             '600M': 2, '350M': 3, '175M': 4}
def split_name(name):
    best = None
    for k in MODELS:
        if name == k or name.startswith(k + '-'):
            if best is None or len(k) > len(best):
                best = k
    return (best, name[len(best)+1:]) if best else (None, None)
out = []
for r in csv.reader(sys.stdin):
    if len(r) < 6:
        continue
    model, ckpt = split_name(r[0])
    if model is None:
        print('  WARN: no models.json key matches: ' + r[0], file=sys.stderr); continue
    e = get_model(model); be = e.get('backends') or {}
    if not any((s or {}).get('eval_groups') for s in (e.get('stages') or {}).values()):
        continue  # no eval_groups (unsupported arch, e.g. gemma-4 / Qwen3.5) → skip
    if e['checkpoint_kind'] == 'hf_branch':
        repo = (be.get('hf') or '').replace('https://huggingface.co/', '')
        if not repo:
            print('  WARN: no hf backend for ' + model, file=sys.stderr); continue
        out.append((r, model, e['size'], 'hub', repo, ckpt, -1))
    else:
        hf = be.get('hf_local')
        if not hf:
            print('  WARN: no hf_local backend for ' + model, file=sys.stderr); continue
        if not (ckpt.startswith('iter') and ckpt[4:].isdigit()):
            print('  WARN: local ckpt not iter<N>: ' + r[0], file=sys.stderr); continue
        out.append((r, model, e['size'], 'local', hf.rstrip('/'), '', int(ckpt[4:])))
out.sort(key=lambda x: (-x[6], size_rank.get(x[2], 9)))
for r, model, size, mode, path, branch, it in out:
    print('\x1f'.join([r[0], r[1], r[2], r[3], r[4], r[5],
                       model, size, mode, path, branch, str(it)]))
")

submitted=0; skipped_active=0; skipped_done=0; skipped_no_ckpt=0; skipped_filter=0
echo "[hf] partition=$PARTITION dry_run=$DRY_RUN filter=${FILTER:-<none>} tasks=$TASKS_GROUP chat=$APPLY_CHAT"
echo ""

while IFS=$'\x1f' read -r name status done total remaining active_jobids model size mode path branch iter; do
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

    # Resolve the positional checkpoint arg per mode. local = converted iter
    # dir (must exist on disk); hub = HF-Hub repo id evaluated at REVISION.
    if [[ "$mode" == "local" ]]; then
        ckpt_arg=$(printf "%s/iter_%07d" "$path" "$iter")
        if [[ ! -f "$ckpt_arg/config.json" ]] || ! ls "$ckpt_arg"/model.safetensors* >/dev/null 2>&1; then
            skipped_no_ckpt=$((skipped_no_ckpt + 1)); continue
        fi
    else
        ckpt_arg="$path"   # HF-Hub repo id; evaluate.sbatch loads it at REVISION
    fi

    n_remaining=$(awk -F, '{print NF}' <<<"$remaining")
    if [[ "$mode" == "hub" ]]; then
        tp=$(tp_from_kv "$path"); pp=1
    else
        tp=$(tp_for "$size" "$model"); pp=$(pp_for "$size" "$model")
    fi
    wall=${WALL_OVERRIDE:-$(walltime_for "$size" "$n_remaining")}

    if (( DRY_RUN )); then
        ref=$([[ "$mode" == hub ]] && echo "@$branch" || echo "@iter$iter")
        echo "  would submit: $name  [$mode$ref]  TP=$tp PP=$pp  --time=$wall  remaining=$n_remaining"
        submitted=$((submitted + 1)); continue
    fi

    # Prefix-export (not --export=ALL,K=V): sbatch's --export splits vars on
    # commas, truncating a comma-bearing TASKS=a,b,c at the first comma. The
    # prefix form puts vars in sbatch's process env, snapshot intact by the
    # default --export=ALL. Env differs by mode: local apertus needs the
    # swissai tokenizer + BOS; hub models use their OWN tokenizer (NEVER the
    # swissai one — wrong vocab) and pin REVISION (+ tokenizer_revision in
    # evaluate.sbatch, CLAUDE bug 1).
    ENVV=(LM_EVAL_BACKEND=vllm BATCH_TASKS=1 TP=$tp PP=$pp
          APPLY_CHAT_TEMPLATE=$APPLY_CHAT
          WANDB_ENTITY=mariagrandury-epflnlp WANDB_PROJECT=snr-experiments
          TASKS="$remaining")
    if [[ "$mode" == "local" ]]; then
        ENVV+=(TOKENIZER=alehc/swissai-tokenizer BOS=true)
    else
        ENVV+=(REVISION="$branch")
    fi
    jid=$(env "${ENVV[@]}" \
        sbatch --parsable \
            --job-name="eval-${name}" \
            --partition="$PARTITION" \
            --time="$wall" \
            "${SBATCH_RES_ARGS[@]}" \
            scripts/evaluate.sbatch "$ckpt_arg" "$name") \
        && {
            echo "  $jid  $name  [$mode]  TP=$tp PP=$pp  $wall  remaining=$n_remaining"
            submitted=$((submitted + 1))
        } || {
            echo "  sbatch FAILED: $name"
        }
    sleep 1
done <<< "$SORTED"

echo ""
echo "submitted=$submitted  skipped_active=$skipped_active  skipped_done=$skipped_done  skipped_no_ckpt=$skipped_no_ckpt  skipped_filter=$skipped_filter"
