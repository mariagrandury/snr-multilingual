#!/bin/bash
# Azure ML wrapper of the predictivity training pair — the Azure half
# (launch_pretraining_cscs.sh is the SLURM half). Every Megatron argument
# lives in megatron_args.sh so both platforms train identically; this file
# adds only the Azure machinery: the pinned Megatron checkout, a GPU-count-
# aware micro-batch, and torchrun. Submitted by `launch_trainings.py azure`
# via azure/jobs/pretrain.yml, which injects the cell's env vars.
#
# Required env (set by the job yml): CKPT_DIR, LOG_DIR, CACHE_DIR, and
# DATA_BLEND unless MOCK_DATA=true.
set -euo pipefail

echo "START TIME: $(date)"
cd "$(dirname "$0")"
source azure/get_megatron.sh

################ Configs ################
SEED=${SEED:-1904}
EXP_NAME=${EXP_NAME:-apertus-${MODEL_SIZE:-175M}-manual-seed${SEED}}
PROJECT_NAME=${PROJECT_NAME:-msnr}
MOCK_DATA=${MOCK_DATA:-false}
#########################################

TENSORBOARD_DIR=$LOG_DIR/tensorboard
DATA_CACHE_DIR=$CACHE_DIR
mkdir -p "$CKPT_DIR" "$DATA_CACHE_DIR" "$TENSORBOARD_DIR"

# Set up ENV
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export HF_HOME=${HF_HOME:-/tmp/hf_home}   # tokenizer download cache
# JIT extensions (xielu) must compile for the local GPU (A100=8.0, H100=9.0)
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

# Build the (platform-identical) training command; then launch with torchrun.
source megatron_args.sh

# The per-size MBS values were tuned for the cluster's node counts; on Azure
# nodes DP is the local GPU count, so shrink MBS to the nearest value that
# keeps GBS % (DP * MBS) == 0 (e.g. on 8 GPUs: 21->21, 7->7, 6->3, 2->1).
NPROC=$(nvidia-smi --list-gpus | wc -l)
MBS=${MBS:-7}
while (( GBS % (NPROC * MBS) != 0 )); do MBS=$((MBS - 1)); done
echo "[$(date)] micro batch size resolved to $MBS on $NPROC GPUs"

RUN_NAME=$EXP_NAME-azure-${AZUREML_RUN_ID:-local}
WANDB_SAVE_DIR=$LOG_DIR
build_megatron_cmd

echo "[$(date)] $EXP_NAME: ${TRAINING_STEPS:-?} iters on $NPROC GPUs (grad accum $(( GBS / (NPROC * MBS) )))"
torchrun --standalone --nproc-per-node=$NPROC $TRAINING_CMD

echo "END TIME: $(date)"
