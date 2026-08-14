#!/bin/bash
# SNR posttraining runner — HF-Hub instruct models, one branch per entry
# (NAME|repo|branch). NAME matches the configs/models.json key so the
# <NAME>-<branch> eval dir lines up with snr_progress / build_hf_dataset.
#
# Two groups:
#  1. Original anchors (Apertus / Olmo) — kept as-is.
#  2. Curated ≤9B subset of posttraining_hf_reference (added 2026-05-31).
#     Text CausalLM only — the Qwen3.5-* entries are multimodal `qwen3_5`
#     and are intentionally excluded. All have kv_heads % 4 == 0 so the
#     evaluate.sbatch default TP=4 is valid (CLAUDE bug 14); all pre-cached
#     in HF_HUB_CACHE (no compute-node internet, CLAUDE bug 16).

# ===== HuggingFace revisions (REVISION is singleton; loop per branch) =====
unset MODEL_CHECKPOINTS MODEL_ITERATIONS
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}

HF_ENTRIES=(
    # --- original anchors ---
    "Apertus-8B-Instruct-2509|swiss-ai/Apertus-8B-Instruct-2509|sft"
    "Olmo-3-7B-Instruct|allenai/Olmo-3-7B-Instruct|step_300"
    "Olmo-3-7B-Instruct|allenai/Olmo-3-7B-Instruct|step_350"
    "Olmo-3-7B-Instruct|allenai/Olmo-3-7B-Instruct|step_400"
    "Apertus-70B-Instruct-2509|swiss-ai/Apertus-70B-Instruct-2509|sft"
    # --- curated ≤9B subset of posttraining_hf_reference ---
    "Qwen3-0.6B|Qwen/Qwen3-0.6B|main"
    "Qwen3-1.7B|Qwen/Qwen3-1.7B|main"
    "Qwen3-4B|Qwen/Qwen3-4B|main"
    "Qwen3-8B|Qwen/Qwen3-8B|main"
    "SmolLM3-3B|HuggingFaceTB/SmolLM3-3B|main"
    "aya-expanse-8b|CohereLabs/aya-expanse-8b|main"
    # --- larger scaling-extension models (all kv%4==0 → default TP=4) ---
    "Qwen3-14B|Qwen/Qwen3-14B|main"
    "Qwen3-30B-A3B|Qwen/Qwen3-30B-A3B|main"
    "aya-expanse-32b|CohereLabs/aya-expanse-32b|main"
)

for ENTRY in "${HF_ENTRIES[@]}"; do
    IFS="|" read -r NAME REPO BRANCH <<< "$ENTRY"
    export REVISION="$BRANCH"
    unset MODEL_CHECKPOINTS
    declare -A MODEL_CHECKPOINTS=(["${NAME}-${BRANCH}"]="$REPO")
    source runners/hf_base_runner.sh "SNR HuggingFace checkpoints"
done
unset REVISION MODEL_CHECKPOINTS
