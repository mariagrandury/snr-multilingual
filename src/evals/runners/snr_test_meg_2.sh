#!/bin/bash
# SNR stage runner - GENERATED from configs/signal_to_ratio/models_test_megatron.txt (selection: --last 2)
# Regenerate: bash scripts/generate_snr_runner.sh --models configs/signal_to_ratio/models_test_megatron.txt --last 2 > $0

# ===== Megatron checkpoints (one sbatch per iter via MODEL_ITERATIONS) =====
export TOKENIZER=${TOKENIZER:-alehc/swissai-tokenizer}
export BOS=${BOS:-true}
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-megatron_lm}

declare -A MODEL_CHECKPOINTS=(
    ["apertus-350M-fwEdu60-fw240-seed1904-iter48000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-350M-fwEdu60-fw240-seed1904/checkpoints"
    ["apertus-350M-fwEdu60-fw240-seed1904-iter50000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-350M-fwEdu60-fw240-seed1904/checkpoints"
)

declare -A MODEL_ITERATIONS=(
    ["apertus-350M-fwEdu60-fw240-seed1904-iter48000-iter"]="48000"
    ["apertus-350M-fwEdu60-fw240-seed1904-iter50000-iter"]="50000"
)

source runners/hf_base_runner.sh "SNR Megatron checkpoints"
