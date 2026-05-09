#!/bin/bash
# SNR runner — Apertus-70B last pretraining checkpoint. Separate from
# snr_pretraining_hf_top.sh so it gets its own 12h slot (the largest
# the normal partition allows).
# Submit through:
#   bash scripts/launch_evaluations.sh snr-pretraining-full \
#       --script runners/snr_pretraining_hf_70b.sh --time 12:00:00

unset MODEL_CHECKPOINTS MODEL_ITERATIONS
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}

HF_ENTRIES=(
    "Apertus-70B-2509|swiss-ai/Apertus-70B-2509|main"
)

for ENTRY in "${HF_ENTRIES[@]}"; do
    IFS="|" read -r NAME REPO BRANCH <<< "$ENTRY"
    export REVISION="$BRANCH"
    unset MODEL_CHECKPOINTS
    declare -A MODEL_CHECKPOINTS=(["${NAME}-${BRANCH}"]="$REPO")
    source runners/hf_base_runner.sh "SNR HuggingFace 70B"
done
unset REVISION MODEL_CHECKPOINTS
