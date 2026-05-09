#!/bin/bash
# SNR stage runner - GENERATED from configs/signal_to_ratio/models_posttraining_hf.txt (selection: --last 3)
# Regenerate: bash scripts/generate_snr_runner.sh --models configs/signal_to_ratio/models_posttraining_hf.txt --last 3 > $0

# ===== HuggingFace revisions (REVISION is singleton; loop per branch) =====
unset MODEL_CHECKPOINTS MODEL_ITERATIONS
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}

HF_ENTRIES=(
    "Apertus-8B-Instruct-2509|swiss-ai/Apertus-8B-Instruct-2509|sft"
    "Olmo-3-7B-Instruct|allenai/Olmo-3-7B-Instruct|step_300"
    "Olmo-3-7B-Instruct|allenai/Olmo-3-7B-Instruct|step_350"
    "Olmo-3-7B-Instruct|allenai/Olmo-3-7B-Instruct|step_400"
    "Apertus-70B-Instruct-2509|swiss-ai/Apertus-70B-Instruct-2509|sft"
)

for ENTRY in "${HF_ENTRIES[@]}"; do
    IFS="|" read -r NAME REPO BRANCH <<< "$ENTRY"
    export REVISION="$BRANCH"
    unset MODEL_CHECKPOINTS
    declare -A MODEL_CHECKPOINTS=(["${NAME}-${BRANCH}"]="$REPO")
    source runners/hf_base_runner.sh "SNR HuggingFace checkpoints"
done
unset REVISION MODEL_CHECKPOINTS
