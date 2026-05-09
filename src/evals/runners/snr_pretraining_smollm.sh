#!/bin/bash
# SNR stage runner - GENERATED from configs/signal_to_ratio/models_pretraining_smollm.txt (selection: --include stage1 --total 30)
# Regenerate: bash scripts/generate_snr_runner.sh --models configs/signal_to_ratio/models_pretraining_smollm.txt --include stage1 --total 30 > $0

# ===== HuggingFace revisions (REVISION is singleton; loop per branch) =====
unset MODEL_CHECKPOINTS MODEL_ITERATIONS
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}

HF_ENTRIES=(
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-40000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-120000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-240000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-360000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-480000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-600000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-720000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-840000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-960000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1080000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1200000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1320000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1440000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1560000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1680000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1760000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-1880000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2000000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2120000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2240000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2360000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2480000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2600000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2720000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2840000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-2960000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-3080000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-3200000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-3320000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-3440000"
)

for ENTRY in "${HF_ENTRIES[@]}"; do
    IFS="|" read -r NAME REPO BRANCH <<< "$ENTRY"
    export REVISION="$BRANCH"
    unset MODEL_CHECKPOINTS
    declare -A MODEL_CHECKPOINTS=(["${NAME}-${BRANCH}"]="$REPO")
    source runners/hf_base_runner.sh "SNR HuggingFace checkpoints"
done
unset REVISION MODEL_CHECKPOINTS
