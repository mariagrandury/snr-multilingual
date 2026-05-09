#!/bin/bash
# SNR stage runner - GENERATED from configs/signal_to_ratio/models_midtraining_hf.txt (selection: --last 3)
# Regenerate: bash scripts/generate_snr_runner.sh --models configs/signal_to_ratio/models_midtraining_hf.txt --last 3 > $0

# ===== HuggingFace revisions (REVISION is singleton; loop per branch) =====
unset MODEL_CHECKPOINTS MODEL_ITERATIONS
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}

HF_ENTRIES=(
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage3-step-4640000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage3-step-4680000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage3-step-4720000"
    "Apertus-8B-2509|swiss-ai/Apertus-8B-2509|step850000-tokens3570B"
    "Apertus-8B-2509|swiss-ai/Apertus-8B-2509|step900000-tokens3780B"
    "Apertus-8B-2509|swiss-ai/Apertus-8B-2509|step950000-tokens3990B"
    "Olmo-3-1025-7B|allenai/Olmo-3-1025-7B|stage3-step7000"
    "Olmo-3-1025-7B|allenai/Olmo-3-1025-7B|stage3-step8000"
    "Olmo-3-1025-7B|allenai/Olmo-3-1025-7B|stage3-step9000"
)

for ENTRY in "${HF_ENTRIES[@]}"; do
    IFS="|" read -r NAME REPO BRANCH <<< "$ENTRY"
    export REVISION="$BRANCH"
    unset MODEL_CHECKPOINTS
    declare -A MODEL_CHECKPOINTS=(["${NAME}-${BRANCH}"]="$REPO")
    source runners/hf_base_runner.sh "SNR HuggingFace checkpoints"
done
unset REVISION MODEL_CHECKPOINTS
