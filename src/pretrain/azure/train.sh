#!/bin/bash
# Azure ML training entrypoint — a direct port of ../submit-apertus-data-mix.sh
# with the SLURM/CSCS machinery removed (srun -> torchrun, no numactl, no
# SIGUSR2 trigger files, no pyxis). Same Megatron arguments, same env-var
# knobs, so launch_azure_trainings.py can drive any size x mix x seed cell
# exactly like launch_trainings.py does on the cluster.
#
# Required env (set by jobs/*.yml): CKPT_DIR, LOG_DIR, CACHE_DIR, and
# DATA_DIR unless MOCK_DATA=true.
set -euo pipefail

echo "START TIME: $(date)"
cd "$(dirname "$0")"
source get_megatron.sh

################ Configs ################
# Defaults = the guide's walkthrough cell: 175M, mix 30/70, seed 28
# (values from ../hyperparams/hyperparams_deep.json — overridden per cell by the launcher).
MODEL_SIZE=${MODEL_SIZE:-175M}
NUM_LAYERS=${NUM_LAYERS:-16}
HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-4096}
NUM_ATTENTION_HEADS=${NUM_ATTENTION_HEADS:-16}
NUM_QUERY_GROUPS=${NUM_QUERY_GROUPS:-4}
LR=${LR:-0.00097919}
MBS=${MBS:-7}    # Micro batch size — must divide GBS/DP
GBS=504          # Global batch size (504 * 4096 = 2_064_384 tokens per step)
SEQ_LEN=4096
TRAINING_STEPS=${TRAINING_STEPS:-50000}  # Must be > (lr_warmup + lr-wsd-decay-iters)
LR_WARMUP_ITERS=${LR_WARMUP_ITERS:-2000}
LR_WSD_DECAY_ITERS=${LR_WSD_DECAY_ITERS:-10000}
CHECKPOINT_STEPS=${SAVE_INTERVAL:-2000}

FW_EDU_RATIO=${FW_EDU_RATIO:-30}
FW2_RATIO=${FW2_RATIO:-70}
SEED=${SEED:-28}
MOCK_DATA=${MOCK_DATA:-false}

WANDB_ENTITY=mariagrandury-epflnlp   # constant, not a launcher-passed variable
# Predictivity-sweep env hooks (same names the sbatch script honours):
# PROJECT_NAME, TOKENIZER_MODEL, DATA_BLEND (a ready --data-path value).
# PROJECT_NAME is injected by the launcher from configs/hf_wandb.json; this
# default is the fallback for a raw job run.
PROJECT_NAME=${PROJECT_NAME:-msnr}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-alehc/swissai-tokenizer}
DATA_MIX_LABEL=${DATA_MIX_LABEL:-"fwEdu${FW_EDU_RATIO}-fw2${FW2_RATIO}"}
EXP_NAME=apertus-${MODEL_SIZE}-${DATA_MIX_LABEL}-seed${SEED}
#########################################

TENSORBOARD_DIR=$LOG_DIR/tensorboard
mkdir -p "$CKPT_DIR" "$CACHE_DIR" "$TENSORBOARD_DIR"

# Set up ENV
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=8
export HF_HOME=${HF_HOME:-/tmp/hf_home}   # tokenizer download cache
# JIT extensions (xielu) must compile for the local GPU (A100=8.0, H100=9.0)
export TORCH_CUDA_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)

NPROC=$(nvidia-smi --list-gpus | wc -l)
# The per-size MBS values were tuned for the cluster's node counts; on Azure
# nodes DP is 1/2/4/8, so shrink MBS to the nearest value that keeps
# GBS % (DP * MBS) == 0 (e.g. on 8 GPUs: 28->21, 14->9, 8->7, 4->3, 2->1).
while (( GBS % (NPROC * MBS) != 0 )); do MBS=$((MBS - 1)); done
echo "[$(date)] micro batch size resolved to $MBS on $NPROC GPUs"

#### Megatron Args #### (same groups as ../submit-apertus-data-mix.sh)
TRANSFORMER_ENGINE_ARGS=(
	--main-grads-dtype fp32
)

NETWORK_SIZE_ARGS=(
	--num-layers $NUM_LAYERS
	--hidden-size $HIDDEN_SIZE
	--ffn-hidden-size $FFN_HIDDEN_SIZE
	--num-attention-heads $NUM_ATTENTION_HEADS
	--group-query-attention
	--num-query-groups $NUM_QUERY_GROUPS
	--max-position-embeddings $SEQ_LEN
	--position-embedding-type rope
	--rotary-base 500000
	--use-rope-scaling
	--rope-scaling-factor 32
	--make-vocab-size-divisible-by 128
	--normalization RMSNorm
	--xielu
	--qk-layernorm
	--qknorm-impl apex
)

LOGGING_ARGS=(
	--log-throughput
	--log-progress
	--tensorboard-dir $TENSORBOARD_DIR
	--no-log-loss-scale-to-tensorboard
	--log-memory-to-tensorboard
)

REGULARIZATION_ARGS=(
	--attention-dropout 0.0
	--hidden-dropout 0.0
	--weight-decay 0.1
	--clip-grad 0.1
	--adam-beta1 0.9
	--adam-beta2 0.999
	--ademamix-alpha 8
	--ademamix-beta3 0.9999
	--ademamix-beta3-warmup 100000
	--ademamix-alpha-warmup 100000
)

TRAINING_ARGS=(
	--micro-batch-size $MBS
	--global-batch-size $GBS
	--no-check-for-nan-in-loss-and-grad
	--train-iters $TRAINING_STEPS
	--log-interval 1
	--cross-entropy-loss-fusion
	--disable-bias-linear
	--optimizer ademamix
	--dataloader-type single
	--manual-gc
	--manual-gc-interval 500
)

INITIALIZATION_ARGS=(
	--seed $SEED
	--init-method-std 0.008944
)

LEARNING_RATE_ARGS=(
	--lr $LR
	--min-lr 0.0
	--lr-decay-style WSD
	--lr-warmup-iters $LR_WARMUP_ITERS
	--lr-wsd-decay-style 1-sqrt
	--lr-wsd-decay-iters $LR_WSD_DECAY_ITERS
)

# --save and --load point at the same dir, so resubmitting the job resumes
# from the latest checkpoint (the jobs pin this dir to a fixed blob path).
CHECKPOINTING_ARGS=(
	--save $CKPT_DIR
	--save-interval $CHECKPOINT_STEPS
	--ckpt-format torch_dist
	--load $CKPT_DIR
	--async-save
	--dist-ckpt-strictness log_unexpected
	--use-checkpoint-opt_param-scheduler
)

MIXED_PRECISION_ARGS=(
	--bf16
)

DISTRIBUTED_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
	--use-distributed-optimizer
	--overlap-grad-reduce
	--overlap-param-gather
)

TOKENIZER_ARGS=(
	--tokenizer-type HuggingFaceTokenizer
	--tokenizer-model $TOKENIZER_MODEL
)

DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
	--num-workers 4
	--num-dataset-builder-threads 1
)

# Data Args, one of three sources:
#  - MOCK_DATA=true: Megatron's synthetic data (smoke tests);
#  - DATA_BLEND set: a ready "--data-path" weight/prefix list (predictivity
#    sweep — composed by launch_azure_predictivity.py from the job's mounts);
#  - else: the tokenized dataset dir's data_path.txt manifest of
#    "<weight> <relative prefix>" pairs shipped with the data (README §5).
if [ "$MOCK_DATA" = true ]; then
  DATA_ARGS+=( --mock-data )
elif [ -n "${DATA_BLEND:-}" ]; then
  DATA_ARGS+=( --data-path $DATA_BLEND --data-cache-path $CACHE_DIR )
else
  DATA_ARGS+=( --data-path $(awk -v d="$DATA_DIR" '{print $1, d"/"$2}' "$DATA_DIR/data_path.txt") )
  DATA_ARGS+=( --data-cache-path $CACHE_DIR )
fi

TRAINING_CMD="$MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
    ${DATA_ARGS[@]}"

# WANDB Logging (primary monitoring; entity via env since --wandb-entity is
# not a pretrain_gpt.py flag)
if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
  export WANDB_ENTITY
  TRAINING_CMD="$TRAINING_CMD \
    --wandb-save-dir $LOG_DIR \
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $EXP_NAME-azure-${AZUREML_RUN_ID:-local}"
else
  echo "[$(date)] No WANDB API key found. Logging to tensorboard only."
  export WANDB_MODE=disabled
fi

echo "[$(date)] $EXP_NAME: $TRAINING_STEPS iters on $NPROC GPUs (grad accum $(( GBS / (NPROC * MBS) )))"
torchrun --standalone --nproc-per-node=$NPROC $TRAINING_CMD

echo "END TIME: $(date)"
