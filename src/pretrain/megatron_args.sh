# Shared Megatron-LM training arguments — the single source of the
# predictivity-sweep training logic. Sourced by the two platform wrappers:
#
#   launch_pretraining_cscs.sh   (SLURM / srun / pyxis container)
#   launch_pretraining_azure.sh  (Azure ML / torchrun)
#
# After calling `build_megatron_cmd`, $TRAINING_CMD holds pretrain_gpt.py plus
# every training argument. Both platforms produce an IDENTICAL command except
# for one intentional delta: when TRIGGER_PATH is set (CSCS only), the SLURM
# graceful-exit flags (--exit-signal-handler --trigger-path) are appended so
# SIGUSR2 checkpoints before walltime. Everything else — architecture,
# optimizer, LR schedule, checkpointing, data blend — is byte-identical, so a
# cell trained on CSCS and a cell trained on Azure follow the same math.
#
# Env contract (set by the wrapper; per-cell values injected by
# launch_trainings.py):
#   Architecture   MODEL_SIZE NUM_LAYERS HIDDEN_SIZE FFN_HIDDEN_SIZE
#                  NUM_ATTENTION_HEADS NUM_QUERY_GROUPS
#   Schedule       TRAINING_STEPS LR LR_WARMUP_ITERS LR_WSD_DECAY_ITERS MBS SEED
#   Data           DATA_BLEND ("w1 prefix1 [w2 prefix2]") or MOCK_DATA=true
#   Paths          MEGATRON_LM_DIR CKPT_DIR TENSORBOARD_DIR DATA_CACHE_DIR
#   W&B (optional) WANDB_API_KEY PROJECT_NAME RUN_NAME WANDB_SAVE_DIR
#   CSCS only      TRIGGER_PATH (enables the graceful-exit flags)
#
# Defaults below are the 175M walkthrough cell so a bare smoke run works; real
# runs always get explicit values from the launcher.

GBS=504          # Global batch size (504 x 4096 = 2_064_384 tokens per step)
SEQ_LEN=4096     # Sequence length
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
WANDB_ENTITY=mariagrandury-epflnlp   # constant — every run logs to this entity

build_megatron_cmd() {
	: "${MEGATRON_LM_DIR:?set by the wrapper (megatron checkout)}"
	: "${CKPT_DIR:?set by the wrapper (checkpoint dir)}"
	: "${TENSORBOARD_DIR:?set by the wrapper (tensorboard dir)}"

	#### Megatron Args #### Check megatron/training/arguments.py
	TRANSFORMER_ENGINE_ARGS=(
		--main-grads-dtype fp32
	)

	NETWORK_SIZE_ARGS=(
		--num-layers ${NUM_LAYERS:-16}
		--hidden-size ${HIDDEN_SIZE:-1024}
		--ffn-hidden-size ${FFN_HIDDEN_SIZE:-4096}
		--num-attention-heads ${NUM_ATTENTION_HEADS:-16}
		--group-query-attention
		--num-query-groups ${NUM_QUERY_GROUPS:-4}
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
		--attention-backend flash
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
		# beta3's ENDPOINT is 0.9999 for every ladder cell and the default
		# here is what makes that true — a run that does not set the env var
		# (every grid cell, every resume, both platforms) is unchanged.
		# ADEMAMIX_BETA3 is set ONLY by launch_trainings.py's diagnostic
		# --ademamix-beta3-factor, which also forces a diag- run name; see
		# plan/90M-rung-anomaly.md for why the endpoint is under suspicion
		# (1/(1-0.9999) = 10000 steps of slow-EMA memory against a 4500-iter
		# 90M run) and why fixing it would mean re-running the ladder.
		--ademamix-beta3 ${ADEMAMIX_BETA3:-0.9999}
		# AdEMAMix alpha/beta3 WARMUP spans the FULL run (the launcher sets
		# ADEMAMIX_WARMUP to the cell's target iters), so the optimizer
		# behaves the same at every ladder size — a fixed 100000 would leave
		# the short runs (4500..81000 iters) with wildly different warmed-up
		# fractions and confound the size-scaling fit. Resumes must pass the
		# same value (the launcher derives it from the cell, so they do).
		--ademamix-beta3-warmup ${ADEMAMIX_WARMUP:-100000}
		--ademamix-alpha-warmup ${ADEMAMIX_WARMUP:-100000}
	)

	TRAINING_ARGS=(
		--micro-batch-size ${MBS:-7}
		--global-batch-size $GBS
		--no-check-for-nan-in-loss-and-grad
		--train-iters ${TRAINING_STEPS:-50}
		--log-interval 1
		--cross-entropy-loss-fusion
		--disable-bias-linear
		--optimizer ademamix
		--dataloader-type single
		--manual-gc
		--manual-gc-interval 500
	)
	if [ -n "${TRIGGER_PATH:-}" ]; then
		# SLURM graceful exit: SIGUSR2 (sent 1h before walltime) makes Megatron
		# checkpoint and exit; the trigger dir enables manual save/exit files.
		TRAINING_ARGS+=(
			--exit-signal-handler
			--trigger-path $TRIGGER_PATH
		)
	fi

	INITIALIZATION_ARGS=(
		--seed ${SEED:-1904}
		# Width-scaled init: the launcher sets INIT_STD = 0.008944 x
		# sqrt(1792 / hidden_size) — 1/sqrt(d) scaling anchored at the 1B
		# (d=1792, which keeps the reviewed 0.008944 exactly), so the init
		# is consistent across the 768..3072 ladder widths instead of one
		# fixed value. Default = the old fixed value for raw runs.
		--init-method-std ${INIT_STD:-0.008944}
	)

	LEARNING_RATE_ARGS=(
		--lr ${LR:-0.00097919}
		--min-lr 0.0
		--lr-decay-style WSD
		--lr-warmup-iters ${LR_WARMUP_ITERS:-10}
		--lr-wsd-decay-style 1-sqrt
		--lr-wsd-decay-iters ${LR_WSD_DECAY_ITERS:-20}
	)

	# --save and --load point at the same dir, so resubmitting a cell resumes
	# it from its latest checkpoint on either platform.
	CHECKPOINTING_ARGS=(
		--save $CKPT_DIR
		--save-interval $SAVE_INTERVAL
		--ckpt-format torch_dist
		--load $CKPT_DIR
		--async-save
		# Tolerate missing transformer-engine `_extra_state` keys when
		# resuming checkpoints saved with a different TE version. Real model
		# weights still load; only TE bookkeeping is skipped.
		--dist-ckpt-strictness log_unexpected
		# Preserve the saved LR/WSD schedule across resumes even when
		# --train-iters is reduced (mid-gap backfill via the idempotent
		# launch_trainings.py). Bypasses the OptimizerParamScheduler
		# train_samples assertion. Safe for fresh starts and end-gap
		# resumes (CLI matches the checkpoint).
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
		# Must match the tokenizer that produced the .bin token IDs
		# (data/build_data_mixtures.py builds with swiss-ai/Apertus-70B-2509).
		--tokenizer-model ${TOKENIZER_MODEL:-swiss-ai/Apertus-70B-2509}
	)

	DATA_ARGS=(
		--split 100,0,0
		--seq-length $SEQ_LEN
		--num-workers 4
		--num-dataset-builder-threads 1
	)
	if [ "${MOCK_DATA:-false}" = true ]; then
		DATA_ARGS+=( --mock-data )
	elif [ -n "${DATA_BLEND:-}" ]; then
		# DATA_BLEND is a ready Megatron --data-path value over pre-built
		# .bin/.idx datasets: "w1 prefix1 [w2 prefix2]" (English DCLM + a
		# setting's FineWeb-2), composed by launch_trainings.py.
		: "${DATA_CACHE_DIR:?set by the wrapper (dataset index cache)}"
		DATA_ARGS+=( --data-path $DATA_BLEND --data-cache-path $DATA_CACHE_DIR )
	else
		echo "megatron_args.sh: set DATA_BLEND or MOCK_DATA=true" >&2
		return 1
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

	# W&B (entity via env since --wandb-entity is not a pretrain_gpt.py flag)
	if [ -n "${WANDB_API_KEY:-}" ]; then
		echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
		export WANDB_ENTITY
		# One CONTINUOUS run per cell across resumes: a deterministic run id
		# (the cell name; dots sanitized — "1.7B" is not a valid id char) +
		# resume=allow makes every resubmission append to the same curve
		# instead of opening a new fragment per job.
		export WANDB_RUN_ID="${RUN_NAME//./-}"
		export WANDB_RESUME=allow
		TRAINING_CMD="$TRAINING_CMD \
			--wandb-save-dir ${WANDB_SAVE_DIR:?set by the wrapper when W&B is on} \
			--wandb-project ${PROJECT_NAME:-msnr} \
			--wandb-exp-name ${RUN_NAME:?set by the wrapper when W&B is on}"
	else
		echo "[$(date)] No WANDB API key found. Logging to tensorboard only."
		export WANDB_MODE=disabled
	fi
}
