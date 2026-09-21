#!/bin/bash

#SBATCH --account=infra01
#SBATCH --time=11:59:59
#SBATCH --job-name=pretrain-manual
#SBATCH --output=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/training/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/training/%x-%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --signal=SIGUSR2@3600	# Send SIGUSR2 1h before hitting the time limit
#SBATCH --no-requeue	# Don't requeue on node failure so we don't lose the logs
#SBATCH --open-mode=append	# A requeue keeps the jobid, so %x-%j would be
				# reopened: append, or the first attempt's log is
				# truncated. (launch_trainings.py --partition
				# preemptable passes --requeue, which outranks
				# the directive above.)

# CSCS wrapper of the predictivity training pair — the SLURM half
# (launch_pretraining_azure.sh is the Azure half). Every Megatron argument
# lives in megatron_args.sh so both platforms train identically; this file
# adds only the SLURM machinery: directories, container launch via srun/pyxis,
# the SIGUSR2 graceful-exit trigger, and debug logging. Submitted by
# `launch_trainings.py cscs`, which injects the cell's env vars via --export.

echo "START TIME: $(date)"
SCRIPT_PATH=$(realpath $0)
# sbatch spools this script to /var/spool/slurmd, so $0's dir is NOT the repo.
# launch_trainings.py passes the real checkout dir via PRETRAIN_DIR (--export);
# fall back to $0's dir for a manual `sbatch launch_pretraining_cscs.sh` run.
SCRIPT_DIR="${PRETRAIN_DIR:-$(dirname "$SCRIPT_PATH")}"

################ Configs ################
SEED=${SEED:-1904}
EXP_NAME=${EXP_NAME:-lm-${MODEL_SIZE:-175M}-manual-seed${SEED}}
PROJECT_NAME=${PROJECT_NAME:-msnr}
MOCK_DATA=${MOCK_DATA:-false}

# Megatron source and dataset cache. One shared checkout for the whole sweep,
# not per-$USER: identical args don't guarantee identical code (see #6), and a
# collaborator's own scratch is an empty tree — the job then dies at
# `python3 $MEGATRON_LM_DIR/pretrain_gpt.py` with exit 2 having trained nothing.
# The SBATCH log paths above are pinned to the same tree for the same reason.
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM}
DATA_CACHE_DIR=${DATA_CACHE_DIR:-/iopsstor/scratch/cscs/mariagrandury/datasets/cache}

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

# Container. The a139 capstor toml is the default; when capstor is unavailable
# fall back to the repo's copy, which pins the same image from the local EDF
# image store (see container/ngc_nemo_iopsstor.toml). CONTAINER_TOML overrides
# both, like convert-snr.sh.
CONTAINER_TOML=${CONTAINER_TOML:-/capstor/store/cscs/swissai/a139/containers/ngc_25-11-nemo-alps3.toml}
[ -f "$CONTAINER_TOML" ] || CONTAINER_TOML=$SCRIPT_DIR/container/ngc_nemo_iopsstor.toml

# ~/.bashrc points HF_HUB_CACHE at capstor, where the tokenizer of
# --tokenizer-model is cached. Unreadable (capstor down) -> drop the override
# so huggingface_hub uses $HF_HOME/hub on iopsstor instead.
if [ -n "$HF_HUB_CACHE" ] && [ ! -d "$HF_HUB_CACHE" ]; then
  echo "[$(date)] HF_HUB_CACHE=$HF_HUB_CACHE unreachable - using \$HF_HOME/hub"
  unset HF_HUB_CACHE
fi

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
  echo "Container: $CONTAINER_TOML"
  nvidia-smi
  echo "Environment Variables:"
  printenv
} > $COMPUTE_ENVIRONMENT_DIR

# Preemption (only on `preemptable`) arrives as SIGTERM with GraceTime=240s
# before the kill. The ranks handle it themselves: EXIT_ON_SIGTERM (set by
# launch_trainings.py --partition preemptable) becomes MEGATRON_EXIT_ON_SIGTERM
# on the srun line below, and the patched DistributedSignalHandler then adds
# SIGTERM to what --exit-signal-handler catches, so Megatron checkpoints at the
# next iteration boundary and exits. With --requeue the job comes back with the
# same jobid and partition and resumes from that save, which is off the
# checkpoint grid — `run_interval`/`due_iters` already expect that from the
# walltime SIGUSR2, and an off-grid save is never due for eval.
#
# This trap is NOT that mechanism, and on its own it does not work: SLURM
# signals the step's tasks directly, so by the time a batch-shell trap runs the
# ranks are already gone (2026-09-21: it fired only during checkpoint loading,
# and a 1.7B lost 334 iterations to a preemption it could not reach). It is
# kept for the log line, which is the cheapest record of when a preemption
# landed, and because a USR2 during startup is harmless.
term_handler() {
	echo "[$(date)] SIGTERM (preemption or scancel) — ranks handle it via MEGATRON_EXIT_ON_SIGTERM; nudging USR2 too"
	scancel --signal=USR2 "$SLURM_JOB_ID"
}
trap term_handler TERM

# Backgrounded so the trap runs when the signal lands rather than after srun
# returns; `wait` is itself interrupted by the trap (returns >128 with the
# step still alive), hence the loop.
srun --mpi=pmix \
	--network=disable_rdzv_get \
	--cpus-per-task $SLURM_CPUS_PER_TASK \
	--environment=$CONTAINER_TOML \
	-lu bash \
	-c "RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID ${EXIT_ON_SIGTERM:+MEGATRON_EXIT_ON_SIGTERM=1 }$CMD_PREFIX $TRAINING_CMD" &
SRUN_PID=$!
while :; do
	wait "$SRUN_PID"; SRUN_RC=$?
	(( SRUN_RC > 128 )) && kill -0 "$SRUN_PID" 2>/dev/null && continue
	break
done
echo "srun exited $SRUN_RC"

echo "END TIME: $(date)"

if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -f $TRIGGER_DIR/exit
   scancel --jobname $SLURM_JOB_NAME
fi
