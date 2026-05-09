#!/bin/bash
# SNR stage runner - 175M, canonical 10 iters (matches the 175M section of
# snr_pretraining_all.sh — NOT what generate_snr_runner.sh produces, since
# the 175M models stopped at ~48000-49000 and the generator's "evenly spaced
# from on-disk" rule otherwise picks non-canonical iters).

# ===== Megatron checkpoints (one sbatch per iter via MODEL_ITERATIONS) =====
export TOKENIZER=${TOKENIZER:-alehc/swissai-tokenizer}
export BOS=${BOS:-true}
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-megatron_lm}

declare -A MODEL_CHECKPOINTS=(
    ["apertus-175M-fwEdu30-fw270-seed1904-iter2000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1904/checkpoints"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter6000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1904/checkpoints"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter12000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1904/checkpoints"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter18000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1904/checkpoints"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter22000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1904/checkpoints"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter28000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1904/checkpoints"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter34000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1904/checkpoints"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter38000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1904/checkpoints"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter44000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1904/checkpoints"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter50000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1904/checkpoints"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter2000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed1904/checkpoints"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter6000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed1904/checkpoints"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter12000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed1904/checkpoints"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter18000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed1904/checkpoints"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter22000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed1904/checkpoints"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter28000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed1904/checkpoints"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter34000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed1904/checkpoints"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter38000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed1904/checkpoints"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter44000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed1904/checkpoints"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter50000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed1904/checkpoints"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter2000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu90-fw210-seed1904/checkpoints"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter6000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu90-fw210-seed1904/checkpoints"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter12000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu90-fw210-seed1904/checkpoints"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter18000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu90-fw210-seed1904/checkpoints"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter22000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu90-fw210-seed1904/checkpoints"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter28000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu90-fw210-seed1904/checkpoints"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter34000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu90-fw210-seed1904/checkpoints"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter38000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu90-fw210-seed1904/checkpoints"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter44000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu90-fw210-seed1904/checkpoints"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter50000"]="/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu90-fw210-seed1904/checkpoints"
)

declare -A MODEL_ITERATIONS=(
    ["apertus-175M-fwEdu30-fw270-seed1904-iter2000-iter"]="2000"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter6000-iter"]="6000"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter12000-iter"]="12000"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter18000-iter"]="18000"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter22000-iter"]="22000"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter28000-iter"]="28000"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter34000-iter"]="34000"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter38000-iter"]="38000"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter44000-iter"]="44000"
    ["apertus-175M-fwEdu30-fw270-seed1904-iter50000-iter"]="50000"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter2000-iter"]="2000"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter6000-iter"]="6000"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter12000-iter"]="12000"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter18000-iter"]="18000"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter22000-iter"]="22000"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter28000-iter"]="28000"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter34000-iter"]="34000"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter38000-iter"]="38000"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter44000-iter"]="44000"
    ["apertus-175M-fwEdu60-fw240-seed1904-iter50000-iter"]="50000"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter2000-iter"]="2000"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter6000-iter"]="6000"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter12000-iter"]="12000"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter18000-iter"]="18000"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter22000-iter"]="22000"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter28000-iter"]="28000"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter34000-iter"]="34000"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter38000-iter"]="38000"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter44000-iter"]="44000"
    ["apertus-175M-fwEdu90-fw210-seed1904-iter50000-iter"]="50000"
)

source runners/hf_base_runner.sh "SNR Megatron 175M checkpoints"
