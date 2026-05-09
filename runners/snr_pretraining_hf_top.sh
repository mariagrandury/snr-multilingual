#!/bin/bash
# SNR runner — last checkpoint of each named pretraining stage for three
# of the reference HF models. Olmo stage1 is excluded (covered by the
# Olmo-3-7B debug resubmit-loop). Apertus-70B is in its own runner
# because it needs a 12h slot on its own.
# Submit through:
#   bash scripts/launch_evaluations.sh snr-pretraining-full \
#       --script runners/snr_pretraining_hf_top.sh --time 12:00:00

# ===== HuggingFace revisions (REVISION is singleton; loop per branch) =====
unset MODEL_CHECKPOINTS MODEL_ITERATIONS
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}

# Order: Olmo first (largest practical priority for the project),
# then SmolLM3 stages, then Apertus-8B.
HF_ENTRIES=(
    "Olmo-3-1025-7B|allenai/Olmo-3-1025-7B|stage2-step47684-mix-round5-from-2T-ckpt"
    "Olmo-3-1025-7B|allenai/Olmo-3-1025-7B|stage3-step11921"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage1-step-3440000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage2-step-4200000"
    "SmolLM3-3B-checkpoints|HuggingFaceTB/SmolLM3-3B-checkpoints|stage3-step-4720000"
    "Apertus-8B-2509|swiss-ai/Apertus-8B-2509|main"
)

for ENTRY in "${HF_ENTRIES[@]}"; do
    IFS="|" read -r NAME REPO BRANCH <<< "$ENTRY"
    export REVISION="$BRANCH"
    unset MODEL_CHECKPOINTS
    declare -A MODEL_CHECKPOINTS=(["${NAME}-${BRANCH}"]="$REPO")
    source runners/hf_base_runner.sh "SNR HuggingFace top checkpoints"
done
unset REVISION MODEL_CHECKPOINTS
