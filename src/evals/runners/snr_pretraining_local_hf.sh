#!/bin/bash
# SNR pretraining runner — vLLM eval against locally-staged HF checkpoints
# (the converted-from-Megatron snapshots under
#  /iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints/).
#
# Uses local paths instead of snr-models HF URLs so we skip the HF API
# round-trip entirely (dodges 429 rate-limits during a large sweep, and
# avoids re-downloading several GB per ckpt). The container mounts
# /iopsstor so the path resolves inside the eval container.
#
# Submit through:
#   bash scripts/launch_evaluations.sh snr-pretraining-full \
#       --script runners/snr_pretraining_local_hf.sh --time 06:00:00
#
# Each (cell, iter) becomes one sbatch via hf_base_runner.sh; per-task
# idempotency in evaluate.sbatch skips checkpoints already evaluated.

unset MODEL_CHECKPOINTS MODEL_ITERATIONS REVISION
export TOKENIZER=${TOKENIZER:-alehc/swissai-tokenizer}
export BOS=${BOS:-true}
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}
# Force TP=1 — these models all fit on one GPU; DP=GPUS_PER_NODE/TP=4 then
# gives near-linear throughput on batched loglikelihood scoring (no per-step
# all-reduce). evaluate.sbatch reads $TP from env (line ~177).
# Don't force TP=1 — vLLM rejects offline data_parallel_size>1 for dense
# models with: "Offline data parallel mode is not supported/useful for dense
# models." (caused 599/810 splits to fail in the first sweep). Let evaluate.sbatch
# default to TP=GPUS_PER_NODE=4 (DP=1). Per-node throughput is comparable —
# 4 GPUs do tensor-parallel inference instead of 4 data-parallel replicas.
# Run all remaining tasks in ONE lm_eval call per split (rather than one
# call per task). vLLM model load is the dominant per-task cost on small
# models, so this ~10x's per-ckpt throughput. _run_per_task.sh implements
# the toggle; per-task idempotency on re-runs still works because the single
# results_*.json lists all tasks under .results.
export BATCH_TASKS=${BATCH_TASKS:-1}

STAGING_BASE=${STAGING_BASE:-/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints}
# Limit to specific seeds (default: 1797 28; 1904 is excluded because it
# already has megatron-backed eval results and we don't want duplicate
# W&B history steps). Override with SEEDS_FILTER="1904 1797 28" to include all.
SEEDS_FILTER=${SEEDS_FILTER:-"1797 28"}

declare -A MODEL_CHECKPOINTS=()
for cell_dir in "$STAGING_BASE"/apertus-*; do
    [[ -d "$cell_dir" ]] || continue
    cell=$(basename "$cell_dir")
    # Skip helper/staging dirs.
    [[ "$cell" == _tmp_torch* || "$cell" == _tokenizer* ]] && continue
    # Apply seed filter (cell ends with -seed<N>).
    cell_seed="${cell##*-seed}"
    matched=0
    for s in $SEEDS_FILTER; do
        [[ "$cell_seed" == "$s" ]] && { matched=1; break; }
    done
    (( matched )) || continue
    for iter_dir in "$cell_dir"/iter_*; do
        [[ -d "$iter_dir" ]] || continue
        # A converted iter dir always has config.json + safetensors.
        [[ -f "$iter_dir/config.json" ]] || continue
        ls "$iter_dir"/model.safetensors* >/dev/null 2>&1 || continue
        iter_num=$(basename "$iter_dir" | sed 's/iter_0*//')
        # Name matches the existing snr_pretraining_all.sh convention so
        # eval_logs / W&B run names line up across megatron and vllm runs.
        name="${cell}-iter${iter_num}"
        MODEL_CHECKPOINTS["$name"]="$iter_dir"
    done
done

if [[ ${#MODEL_CHECKPOINTS[@]} -eq 0 ]]; then
    echo "[snr-local-hf] no converted checkpoints found under $STAGING_BASE" >&2
    exit 1
fi

echo "[snr-local-hf] found ${#MODEL_CHECKPOINTS[@]} converted (cell, iter) checkpoints to evaluate"

source runners/hf_base_runner.sh "SNR local-HF (vLLM) checkpoints"
