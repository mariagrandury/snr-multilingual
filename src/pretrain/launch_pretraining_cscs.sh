#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=11:59:59
#SBATCH --job-name=apertus-pretrain
#SBATCH --output=/iopsstor/scratch/cscs/%u/data-mix-small/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/data-mix-small/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --signal=SIGUSR2@3600	# Send SIGUSR2 1h before hitting the time limit
#SBATCH --no-requeue	# Don't requeue on node failure so we don't lose the logs

# CSCS wrapper of the predictivity training pair — the SLURM half
# (launch_pretraining_azure.sh is the Azure half). Every Megatron argument
# lives in megatron_args.sh so both platforms train identically; this file
# adds only the SLURM machinery: directories, container launch via srun/pyxis,
# the SIGUSR2 graceful-exit trigger, and debug logging. Submitted by
# `launch_trainings.py cscs`, which injects the cell's env vars via --export.

echo "START TIME: $(date)"
SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR=$(dirname $SCRIPT_PATH)

################ Configs ################
SEED=${SEED:-1904}
EXP_NAME=${EXP_NAME:-apertus-${MODEL_SIZE:-175M}-manual-seed${SEED}}
PROJECT_NAME=${PROJECT_NAME:-msnr}
MOCK_DATA=${MOCK_DATA:-false}

# Megatron source and dataset cache
MEGATRON_LM_DIR=/iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM
DATA_CACHE_DIR=/iopsstor/scratch/cscs/$USER/datasets/cache

#### Debugging ####
LOG_NCCL=false # Log NCCL_DEBUG=info into per-process files under $DEBUG_DIR
NSYS_PROFILER=false # Turn on the NSYS profiler (check --profile-* megatron args)
#########################################

PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME
EXP_DIR=$PROJECT_DIR/$EXP_NAME
CKPT_DIR=$EXP_DIR/checkpoints
TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard

mkdir -p $CKPT_DIR $TRIGGER_DIR $DEBUG_DIR $LOGGING_DIR $DATA_CACHE_DIR

# Set up ENV
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
export TRITON_HOME=/iopsstor/scratch/cscs/$USER/.triton  # /dev/shm is noexec on compute nodes

# torch.distributed rendezvous (RANK/LOCAL_RANK are set at the srun command)
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=8963
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# Sync any previous run's W&B data before starting (wandb is in the container
# on compute nodes — guard so a missing host binary doesn't pollute the log).
if [ -n "$WANDB_API_KEY" ] && [ -d "$LOGGING_DIR/wandb/latest-run" ] && command -v wandb >/dev/null 2>&1; then
  echo "[$(date)] Syncing WANDB from previous run"
  wandb sync "$LOGGING_DIR/wandb/latest-run"
fi

# Build the (platform-identical) training command. TRIGGER_PATH adds the
# SLURM graceful-exit flags — the one intentional CSCS-only delta.
TRIGGER_PATH=$TRIGGER_DIR
RUN_NAME=$EXP_NAME   # stable: resumes append to ONE W&B run (see megatron_args.sh)
WANDB_SAVE_DIR=$LOGGING_DIR
source $SCRIPT_DIR/megatron_args.sh
build_megatron_cmd || exit 1
TRAINING_CMD="python3 $TRAINING_CMD"

CMD_PREFIX="numactl --membind=0-3"

# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

# NSYS profiler
if [ "$NSYS_PROFILER" = true ]; then
    NSYS_LAUNCHER="nsys profile -s none --trace='nvtx,cudnn,cublas,cuda' --output=$DEBUG_DIR/nsys-trace-hostname-\$SLURMD_NODENAME-procid-\$SLURM_PROCID.nsys-rep --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
    TRAINING_CMD="$NSYS_LAUNCHER $TRAINING_CMD --profile"
fi

# Clean triggers left over from a previous run
rm -f $TRIGGER_DIR/save $TRIGGER_DIR/exit

# Record the compute environment for debugging (command, code version, GPUs)
cp $SCRIPT_PATH $DEBUG_DIR
{
  date
  echo "CMD: $CMD_PREFIX $TRAINING_CMD"
  echo "NODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)"
  echo "Megatron path: $MEGATRON_LM_DIR ($(git -C $MEGATRON_LM_DIR rev-parse --verify HEAD))"
  nvidia-smi
  echo "Environment Variables:"
  printenv
} > $COMPUTE_ENVIRONMENT_DIR

srun --mpi=pmix \
	--network=disable_rdzv_get \
	--cpus-per-task $SLURM_CPUS_PER_TASK \
	--environment=/capstor/store/cscs/swissai/a139/containers/ngc_25-11-nemo-alps3.toml \
	-lu bash \
	-c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX $TRAINING_CMD"

echo "END TIME: $(date)"

if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -f $TRIGGER_DIR/exit
   scancel --jobname $SLURM_JOB_NAME
fi
