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
#SBATCH --open-mode=append	# Belt and braces: nothing requeues these any more
				# (preemptable self-chains instead, one new jobid
				# per link), but a requeue would keep the jobid and
				# reopen %x-%j, truncating the first attempt's log.

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

################ Self-chain ################
# Clariden's MaxBatchRequeue is 5 and a preemption spends one, so a --requeue'd
# run on `preemptable` is HELD on its sixth — reported, misleadingly, as
# "launch failure limit exceeded requeued held" even when every attempt ended
# in a clean preemption (three runs stopped that way on 2026-09-21). A chain
# never meets that cap, because each link is a NEW jobid. This is the shape
# data/submit_build_one.sh has always used, and it is proven at length:
# build-a-L8-165b took 29 links over 36h (24 PREEMPTED, 3 FAILED, 1 CANCELLED,
# 1 COMPLETED) to finish one 165B mixture.
#
# The successor is queued UP FRONT, before training starts, with
# --dependency=singleton — same job name, so only one link of a cell ever runs.
# Queuing it first is what makes it preemption-safe: a killed attempt leaves
# its successor already pending (holding a queue position, which a resubmit
# would forfeit), and the idempotent resume picks up from the checkpoint the
# patched signal handler wrote on the way out.
#
# Opt-in via PRETRAIN_CHAIN, which launch_trainings.py sets for
# --partition preemptable: a manual `sbatch` of this wrapper must never
# silently queue a second 21-node job.
CHAIN_SCRIPT=$SCRIPT_DIR/launch_pretraining_cscs.sh
# Where THIS job's %x-%j.out actually lands, asked of the controller rather
# than assumed: counting attempts in a directory that holds none reads 0 every
# time, so the cap never fires and the chain runs unbounded at 21 nodes a link.
# The #SBATCH --output above does pin every user's logs to this one tree today
# (8 of aromanou's are in it), but a chain that cannot find its own bound must
# not guess at one.
if [ -z "${CHAIN_LOGDIR:-}" ]; then
	CHAIN_LOGDIR=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
	               | sed -n 's/.*StdOut=\([^[:space:]]*\).*/\1/p' | head -1)
	CHAIN_LOGDIR=${CHAIN_LOGDIR%/*}
fi

# done | fresh | resume | corrupt for this cell, or EMPTY when the check could
# not run at all. Empty is NOT "not done": chaining on a state we cannot read
# would spend the whole attempt budget on links that allocate nodes and exit,
# which is why pretrain_progress.py --cell-action prints the word instead of
# encoding it in an exit status an ImportError could forge. python3.11 first —
# the system python3 is 3.6 and cannot parse launch_trainings.py, which
# pretrain_progress imports (scripts/preempt_drain.sh has the same note).
cell_action() {
	local py
	# No positive target, no verdict: against a target of 0 cell_action calls
	# every cell with a checkpoint "done", and the wrapper would skip training
	# and cancel the chain. -gt, not -n: `[ -n "0" ]` is true.
	[ "${TRAINING_STEPS:-0}" -gt 0 ] 2>/dev/null || return 0
	py=$(command -v python3.11 || command -v python) || return 0
	"$py" "$SCRIPT_DIR/pretrain_progress.py" --cell-action \
	      "$EXP_DIR" "$TRAINING_STEPS" 2>/dev/null
}
# "<word> <latest valid iter>", or both empty when the check could not run.
read -r CELL_ACTION CELL_ITER <<<"$(cell_action)"

# Cancelling the successor is a chain operation: a plain `normal` or manual run
# must not scancel jobs it never queued. Empty name guarded too — scancel with
# no name filter is not a mistake worth risking.
cancel_successor() {
	[ -n "${PRETRAIN_CHAIN:-}" ] && [ -n "${SLURM_JOB_NAME:-}" ] \
		&& scancel --state=PENDING --name="$SLURM_JOB_NAME"
	return 0
}

# Already at the target: end the chain rather than start Megatron on a finished
# cell. The successor this link would otherwise inherit has nothing to do.
if [ "$CELL_ACTION" = done ]; then
	echo "[$(date)] $EXP_NAME is already at its target of ${TRAINING_STEPS:-?} iters — nothing to train; ending the chain."
	cancel_successor
	exit 0
fi

if [ -n "${PRETRAIN_CHAIN:-}" ]; then
	# The successor inherits the partition this attempt actually ran in (a
	# drainer may have moved it) and its walltime — a limit cannot be raised
	# after submission, so it has to be asked for here.
	CHAIN_PART=${CHAIN_PARTITION:-$SLURM_JOB_PARTITION}
	CHAIN_TIME=${CHAIN_WALLTIME:-$(squeue -h -j "$SLURM_JOB_ID" -O TimeLimit 2>/dev/null | tr -d ' ')}
	[ -n "$CHAIN_TIME" ] || CHAIN_TIME=11:59:59
	# The no-progress guard, and the real bound on this chain. Each link records
	# the iteration it STARTED from; when that has not moved, the link before it
	# achieved nothing, and a chain that achieves nothing four times running is
	# retrying a failure, not surviving preemption. On 2026-09-22 one cell burned
	# 17 links x 21 nodes this way: its logging/tensorboard belonged to a
	# collaborator, so rank 83 died of PermissionError ~70s in, every time, and
	# Slurm killed the step for TASK FAILURE while the wrapper still exited 0.
	# Progress resets the counter, so a genuinely preempted run is never capped.
	# Per-USER filename. This directory carries a default ACL granting each
	# collaborator rwx, but a file one of them creates belongs to them and group
	# a139 gets only r-x — and the dir is sticky, so nobody can replace it. A
	# shared name would let whoever chained first freeze everyone else's counter
	# at its value forever.
	PROGRESS_FILE=$CHAIN_LOGDIR/.chain-$SLURM_JOB_NAME-${USER:-$(id -un)}.progress
	stalls=0; progress_recorded=
	if [ -n "$CELL_ACTION" ] && [ -d "$CHAIN_LOGDIR" ]; then
		read -r prev_iter prev_stalls 2>/dev/null < "$PROGRESS_FILE" || prev_iter=
		# Anything but digits restarts the count rather than poisoning `-ge`.
		case ${prev_stalls:-} in ''|*[!0-9]*) prev_stalls=0 ;; esac
		if [ "$CELL_ITER" = "$prev_iter" ]; then stalls=$((prev_stalls + 1)); fi
		echo "$CELL_ITER $stalls" > "$PROGRESS_FILE" 2>/dev/null && progress_recorded=1
	fi
	# The attempt cap is now only a backstop against something the stall counter
	# cannot see, so it is generous: a 3-day 3B at the observed ~1 preemption/hour
	# needs 30-70 links, and build-a-L8-165b needed 29 for a ONE-node build.
	# One %x-%j.out per attempt — each link is its own jobid, so nothing appends.
	n_attempts=$(find "$CHAIN_LOGDIR" -maxdepth 1 -name "${SLURM_JOB_NAME}-[0-9]*.out" 2>/dev/null | wc -l)
	if [ ! -r "$CHAIN_SCRIPT" ]; then
		echo "[$(date)] WARN: no chain successor — $CHAIN_SCRIPT unreadable (PRETRAIN_DIR wrong?)"
	elif [ -z "$CELL_ACTION" ]; then
		echo "[$(date)] WARN: no chain successor — could not read this cell's state; re-run launch_trainings.py to resume it"
	elif [ ! -d "$CHAIN_LOGDIR" ]; then
		echo "[$(date)] WARN: no chain successor — cannot count attempts in '$CHAIN_LOGDIR'; an uncountable chain is an unbounded one"
	elif [ "$CELL_ACTION" = corrupt ]; then
		echo "[$(date)] WARN: no chain successor — cell is corrupt (iter dirs on disk, none loadable); launch_trainings.py refuses these too, they want manual review"
	elif [ -z "$progress_recorded" ]; then
		echo "[$(date)] WARN: no chain successor — cannot record progress in $PROGRESS_FILE; a chain whose progress cannot be recorded cannot be stopped by one"
	elif [ "$stalls" -ge "${CHAIN_MAX_STALLS:-4}" ]; then
		echo "[$(date)] WARN: no chain successor — $stalls links in a row started at iter $CELL_ITER without advancing; this is a failing run, not a preempted one. Read $SLURM_JOB_NAME-$SLURM_JOB_ID.err, fix it, then re-run launch_trainings.py."
	elif [ "$n_attempts" -ge "${CHAIN_MAX_ATTEMPTS:-200}" ]; then
		echo "[$(date)] WARN: no chain successor — $n_attempts attempts already (raise CHAIN_MAX_ATTEMPTS, default 200)"
	else
		echo "[$(date)] queuing singleton successor (attempt $n_attempts, $CHAIN_PART, $CHAIN_TIME, $SLURM_JOB_NUM_NODES nodes)"
		sbatch --dependency=singleton --job-name="$SLURM_JOB_NAME" \
		       --partition="$CHAIN_PART" --time="$CHAIN_TIME" \
		       --nodes="$SLURM_JOB_NUM_NODES" --account="$SLURM_JOB_ACCOUNT" \
		       --comment=selfchain --export=ALL "$CHAIN_SCRIPT"
	fi
fi

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
# next iteration boundary and exits. The singleton successor queued at the top
# then starts and resumes from that save, which is off the checkpoint grid —
# `run_interval`/`due_iters` already expect that from the walltime SIGUSR2, and
# an off-grid save is never due for eval.
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

# Reached the target: the successor queued before training has nothing left to
# do, and letting it start would allocate $SLURM_JOB_NUM_NODES nodes to print
# one line and exit. A preempted or failed attempt deliberately leaves it
# pending — that is the whole point of queuing it up front.
read -r CELL_ACTION CELL_ITER <<<"$(cell_action)"
if [ "$CELL_ACTION" = done ]; then
	echo "[$(date)] $EXP_NAME reached ${TRAINING_STEPS:-?} iters — cancelling the pending chain successor"
	cancel_successor
fi

if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -f $TRIGGER_DIR/exit
   scancel --jobname $SLURM_JOB_NAME
fi
