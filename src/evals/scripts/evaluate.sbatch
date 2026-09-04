#!/bin/bash
#SBATCH --account=infra01
#SBATCH --cpus-per-task=200
#SBATCH --gres=gpu:4
#SBATCH --job-name=evaluation
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --time=11:59:00
#SBATCH --exclude=nid006784
#SBATCH --exclude=nid007014
#SBATCH --exclude=nid007127
#SBATCH --exclude=nid007348
#SBATCH --exclude=nid007449
#SBATCH --exclude=nid007548
#SBATCH --exclude=nid007587
#SBATCH --exclude=nid006679
#SBATCH --exclude=nid006975
#SBATCH --exclude=nid007620

# Aux functions.
usage() {
	echo "Usage: sbatch evaluate.sbatch <model> <name>"
	echo "Runs evaluation of the specified model. The 'model' should a huggingface path/name."
	echo "The 'name' shall be used as an unique identifier."
	echo "In addition of these three necessary positional arguments, you can specify the following bash environment variables. Unless explicitly stated, all of these are optional:"
	echo " TOKENIZER: A huggingface tokenizer path/name. Needed if 'model' is a megatron checkpoint, otherwise the default will be set as the same value as 'model'."
	echo " BS: Batch size."
	echo " REVISION: Only used in huggingface models. If set, this revision of the model will be evaluated."
	echo " SIZE: The (approximate) size of the model in billions of parameters. Used to set model parallelism (needs to be set in larger models)."
	echo " MAX_LENGTH: The maximum length of the input sequence to the model. By default, it is unset."
	echo " MAX_NEW_TOKENS: The maximum number of new tokens to generate. By default, it is unset."
	echo " BOS: Set this to 'true' if you wish to prepend the BOS token when evaluating models."
	echo " LOGS_ROOT: Where are your evaluation wandb&harness logs going to."
	echo " TASKS: Tasks to run with lm eval harness."
	echo " WANDB_ENTITY: WandB entity name for uploading results (default: apertus)."
	echo " WANDB_PROJECT: WandB project name for uploading results (default: swissai-evals)."
	echo " LM_EVAL_BACKEND: Backend to use for lm-eval (default: hf). - options are: hf, vllm, megatron_lm"
    echo " CKPT_ITER: Megatron checkpoint iteration (integer). Required when LM_EVAL_BACKEND=megatron_lm."
    echo " HARNESS_LIMIT: Optional argument to pass as --limit to the lm-evaluation-harness, to limit the number of samples per task (default: no limit)."
    echo " LM_EVAL_HARNESS_BRANCH: Branch/ref to install from swiss-ai/lm-evaluation-harness (default: repository default branch)."
    echo " EVAL_WORKERS: vllm only. Worker processes per job, each with its own model copy on its own GPU group, sharing one task queue (default: GPUs / (TP x PP), i.e. every GPU busy)."
	echo "For more information see the README: https://github.com/swiss-ai/evals?tab=readme-ov-file."
}
die() {
	echo "$*" >& 2
	exit 1
}

# Wakeup logs.
set -e
echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"

# Grab variables and arguments.
if (( $# != 2 )); then
	usage
	die "Invalid usage: Invalid argument count"
fi
MODEL=$1
NAME=$2
CKPT_ITER=${CKPT_ITER:-"latest"}
BS=${BS:-"auto:20"}
SIZE=${SIZE:-1}
BOS=${BOS:-false}
LOGS_ROOT=${LOGS_ROOT:-/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs}
TASKS=${TASKS:-./configs/apertus/tasks_default.txt}
TABLE_METRICS=${TABLE_METRICS:-./configs/apertus/tasks_default_main_table.txt}
WANDB_ENTITY=${WANDB_ENTITY:-apertus}
WANDB_PROJECT=${WANDB_PROJECT:-swissai-evals-test}
LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-true}
HARNESS_LIMIT=${HARNESS_LIMIT:-""}
LM_EVAL_HARNESS_BRANCH=${LM_EVAL_HARNESS_BRANCH:-""}

export HF_TOKEN="${HF_TOKEN:-$(tr -d '\r\n' < ./scripts/hf_token.txt)}"
export WANDB_API_KEY="${WANDB_API_KEY:-$(tr -d '\r\n' < ./scripts/wandb_api_key.txt)}"
export CSCS_SERVING_API="${CSCS_SERVING_API:-$(tr -d '\r\n' < ./scripts/cscs_serving_api_key.txt)}"

warn_if_empty() {
  local var="$1"
  local msg="$2"

  if [[ -z "${!var:-}" ]]; then
    echo "WARNING: $var is not set. $msg" >&2
  fi
}

warn_if_empty HF_TOKEN \
  "This may cause Hugging Face rate-limit issues or prevent access to gated models/datasets."

warn_if_empty WANDB_API_KEY \
  "Weights & Biases logging is disabled. Results will NOT be uploaded."

warn_if_empty CSCS_SERVING_API \
  "Optional key missing. LLM-as-a-judge benchmarks via CSCS serving API will fail."

# optional eval arguments
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
NUM_FEWSHOT=${NUM_FEWSHOT:-""}

# Task splitting support: set NUM_SPLITS and SPLIT_INDEX to distribute tasks across jobs
NUM_SPLITS=${NUM_SPLITS:-1}
SPLIT_INDEX=${SPLIT_INDEX:-0}

if [ -f "$TASKS" ]; then
    echo "Reading task list from file: $TASKS"
    # Filter out comments and blank lines, then join with commas
    TASKS=$(grep -v '^\s*#' "$TASKS" | grep -v '^\s*$' | paste -sd, -)

    # Split tasks across jobs if NUM_SPLITS > 1
    if (( NUM_SPLITS > 1 )); then
        IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
        TOTAL_TASKS=${#TASK_ARRAY[@]}
        CHUNK_SIZE=$(( (TOTAL_TASKS + NUM_SPLITS - 1) / NUM_SPLITS ))
        START=$(( SPLIT_INDEX * CHUNK_SIZE ))
        END=$(( START + CHUNK_SIZE ))
        if (( END > TOTAL_TASKS )); then END=$TOTAL_TASKS; fi
        SPLIT_TASKS=()
        for (( i=START; i<END; i++ )); do
            SPLIT_TASKS+=("${TASK_ARRAY[$i]}")
        done
        TASKS=$(IFS=','; echo "${SPLIT_TASKS[*]}")
        echo "Split $((SPLIT_INDEX+1))/$NUM_SPLITS: running ${#SPLIT_TASKS[@]}/$TOTAL_TASKS tasks (indices $START-$((END-1)))"
    fi
fi

if [ -f "$TABLE_METRICS" ]; then
	echo "Reading table metrics from file: $TABLE_METRICS"
	TABLE_METRICS=$(grep -v '^\s*#' "$TABLE_METRICS" | grep -v '^\s*$' | paste -sd' ' -)
fi

# HF caching — point at the populated cache to avoid hammering the HF API
# during a sweep (the 429 cascade that killed ~600 jobs on 2026-05-04 was
# caused by an unconfigured cache + concurrent first-time fetches across the
# whole burst). The capstor path holds ~258 GB of pre-fetched models +
# datasets; the iopsstor path holds the legacy datasets/modules cache.
# Both are exported to the inner container via INNER_EXPORTS below — without
# that, host env vars don't propagate through `srun --environment=...`
# (CLAUDE.md eval bug 5).
# Shared caches, set unconditionally — deliberately NOT ${VAR:-...}. sbatch
# --export=ALL carries the submitter's login shell into the job, and ~/.bashrc
# exports HF_HOME=/iopsstor/scratch/cscs/$USER/hf_home, so a default would
# always lose to a collaborator's own empty paths (17 MB of models vs 3.6 TB,
# and no datasets/ at all). HF_DATASETS_OFFLINE=1 below leaves no hub fallback,
# so every task then dies in load_dataset() before generating a token — that is
# what killed jobs 3255016-3255025. Move these by editing here, not via the env.
export HF_HOME=/iopsstor/scratch/cscs/mariagrandury/hf_home
export HF_HUB_CACHE=/capstor/store/cscs/swissai/infra01/users/mariagrandury/hf_models
# Datasets-only offline mode. The 2026-05-09 SNR launch cascaded ~140 jobs to
# FAILED inside `datasets.load_dataset()` because the burst made `datasets`
# probe the hub for the latest revision of m_truthfulqa / m_hellaswag /
# multiblimp ⇒ 429 ⇒ "FileNotFoundError: hf://datasets/.../val.jsonl". The
# datasets are pre-staged at HF_HOME/datasets/<repo>/<lang>/0.0.0/<rev>/, so
# skipping the hub probe here is the right thing. We deliberately do NOT set
# HF_HUB_OFFLINE — that broke `AutoTokenizer.from_pretrained("alehc/swissai-
# tokenizer")` because transformers' cache layout doesn't satisfy strict
# offline lookups.
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
GPUS_PER_NODE=4

# Print configuration.
echo "Configuration set:"
printf "MODEL=$MODEL\nNAME=$NAME\nTOKENIZER=$TOKENIZER\nBS=$BS\nREVISION=$REVISION\nSIZE=$SIZE\nHARNESS_LIMIT=$HARNESS_LIMIT\nLM_EVAL_HARNESS_BRANCH=$LM_EVAL_HARNESS_BRANCH\nBOS=$BOS\nTASKS=$TASKS\nMAX_LENGTH=$MAX_LENGTH\nWANDB_ENTITY=$WANDB_ENTITY\nWANDB_PROJECT=$WANDB_PROJECT\nLM_EVAL_BACKEND=$LM_EVAL_BACKEND\nLHF_HOME=$HF_HOME\n\n"

# Generic envs.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=25678
export CUDA_DEVICE_MAX_CONNECTIONS=1

export WORLD_SIZE=$GPUS_PER_NODE

# export HF_ALLOW_CODE_EVAL=1

echo "Huggingface checkpoint detected!"
if [ ! -z ${REVISION+x} ]; then
    echo "Using revision=$REVISION"
    MAYBE_REVISION=",revision=$REVISION"
    MAYBE_DOWNLOAD_REVISION="--revision=$REVISION"
    # Some repos (e.g. HuggingFaceTB/SmolLM3-3B-checkpoints) keep all
    # checkpoint files on revision branches and leave `main` empty. vLLM
    # loads the tokenizer separately and does not auto-derive
    # tokenizer_revision from revision, so AutoTokenizer fetches `main`,
    # finds no tokenizer.json, and dies with the misleading "need
    # sentencepiece or tiktoken" error. Pin tokenizer_revision explicitly.
    MAYBE_TOKENIZER_REVISION=",tokenizer_revision=$REVISION"
fi
if [ -z ${TOKENIZER+x} ]; then
    echo "TOKENIZER not set, using TOKENIZER=$MODEL"
    TOKENIZER=$MODEL
fi


# Tensor-parallel size. Default to all GPUs on the node (legacy behavior, right
# for big models that don't fit on one GPU). Override with TP=1 for small models
# where DP=GPUS_PER_NODE/TP gives near-linear throughput on batch loglikelihood
# scoring (no per-step all-reduce).
MP=${TP:-$GPUS_PER_NODE}
echo "Using model size ~${SIZE}B, tensor parallel size of $MP"

# Prepare logs root.
RUN_ROOT=$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/$NAME
HARNESS_DIR=$RUN_ROOT/harness
export WANDB_DIR=$RUN_ROOT
mkdir -p $HARNESS_DIR

# export HF_HUB_OFFLINE=1
export HF_ALLOW_CODE_EVAL=1
# export PERSPECTIVE_API_KEY="" # Uncomment if you want to use perspective API, needed for realtoxicityprompts task

# Finally, run the actual evaluations.
HF_CHECKPOINT_PATH=$MODEL
# Pipeline parallel: complements TP when TP can't fully use the node (e.g. SNR
# 350M kv=5 → TP=1 ⇒ PP=4 to fill 4 GPUs; 600M kv=6 → TP=2 ⇒ PP=2; 1B kv=7
# → TP=1 ⇒ PP=4; 175M kv=4 → TP=4 ⇒ PP=1). PP has no kv_heads constraint, just
# needs num_hidden_layers % PP == 0. Default 1 (= no PP). See CLAUDE.md bug 14.
PP=${PP:-1}
# vLLM rejects offline data_parallel_size>1 for dense models with
# "Offline data parallel mode is not supported/useful for dense models."
# (CLAUDE.md bug 11), so data_parallel_size stays 1 and the GPUs that TP x PP
# leave over are used the other way round: that many independent
# scripts/eval_worker.py processes, each with its own model copy on its own
# GPU group, all drawing from one task queue (scripts/_run_per_task.sh). The
# ladder's KV-head counts force TP=1, so on a 4-GPU node that is 4 workers.
DP=1
EVAL_WORKERS=${EVAL_WORKERS:-$(( GPUS_PER_NODE / (MP * PP) ))}
if [[ "$LM_EVAL_BACKEND" != "vllm" ]]; then
    EVAL_WORKERS=1    # accelerate / torchrun already span the node's GPUs
elif (( EVAL_WORKERS < 1 )); then
    # TP x PP claims more GPUs than the node has: the division floors to 0,
    # which would start no workers at all and burn the allocation on a job
    # that reports "no task finished". One worker, and let vLLM complain
    # about the layout if it is genuinely impossible.
    echo "WARNING: TP=$MP x PP=$PP exceeds $GPUS_PER_NODE GPUs; running 1 worker"
    EVAL_WORKERS=1
fi
echo "Eval workers: $EVAL_WORKERS x (TP=$MP PP=$PP) GPU(s)"
COMMON_MODEL_ARGS="dtype=bfloat16"
if [[ $BOS = true ]]; then
	echo "Adding BOS token."
	COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,add_bos_token=True"
fi

# Add vLLM specific arguments.
export OMP_NUM_THREADS=$(( 8 * $GPUS_PER_NODE ))
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_COMPILE_CACHE=1

#export VLLM_TORCH_COMPILE=0
#export NCCL_P2P_DISABLE=1

# export NCCL_IB_DISABLE=1
# export NCCL_NET="Socket"
# export NCCL_DEBUG=WARN
# Reduce memory fraction to avoid OOM during initialization
VLLM_MEMORY_FRACTION=0.75
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

if [[ $LM_EVAL_BACKEND == "vllm" ]]; then
    # enable_prefix_caching: the choices of a multiple-choice doc share the
    # whole context, so its KV blocks are computed once and reused per choice.
    # Ported from include-private's run_include_eval.sbatch. vLLM's V1 engine
    # turns this on by default, so on the current container it should be a
    # no-op that only pins the behaviour — NOT verified against this image's
    # swiss-ai fork. Verify before trusting a cross-time comparison: re-run an
    # already-evaluated checkpoint and diff the scores. Drop the flag if they
    # move. The include-private max_model_len cap is deliberately NOT ported —
    # the ladder's config already says 4096 (megatron_args.sh), and a lower
    # cap would truncate prompts differently from every past result.
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,pretrained=$HF_CHECKPOINT_PATH,tokenizer=$TOKENIZER$MAYBE_REVISION${MAYBE_TOKENIZER_REVISION:-},data_parallel_size=$DP,tensor_parallel_size=$MP,pipeline_parallel_size=$PP,gpu_memory_utilization=$VLLM_MEMORY_FRACTION,enable_prefix_caching=True,enable_thinking=False"
elif [[ $LM_EVAL_BACKEND == "megatron_lm" ]]; then
    # `--dist-ckpt-strictness=log_unexpected` tells Megatron's distributed
    # checkpoint loader to log (instead of raise) on missing keys like
    # `q_layernorm._extra_state` — these are TE bookkeeping fields the
    # current Megatron-LM expects but older checkpoints don't carry. The
    # default ASSUME_OK_UNEXPECTED delegates to the underlying strategy,
    # which fatally raises in torch.distributed.checkpoint's planner. We
    # use `=` (not space) so the token survives bash word-splitting in
    # CMD_BASE. Real model weights still load unchanged.
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,load=$MODEL,tokenizer_type=HuggingFaceTokenizer,tokenizer_model=$TOKENIZER$MAYBE_REVISION,devices=4,extra_args=--dist-ckpt-strictness=log_unexpected"
    if [[ "$CKPT_ITER" =~ ^[0-9]+$ ]]; then
        COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,ckpt_step=$CKPT_ITER"
    elif [[ "$CKPT_ITER" == "latest" ]]; then
        echo "CKPT_ITER=latest: omitting ckpt_step from Megatron model args."
    else
        die "CKPT_ITER must be an integer or 'latest' when LM_EVAL_BACKEND=megatron_lm (got '$CKPT_ITER')."
    fi
elif [[ $LM_EVAL_BACKEND == "hf" ]]; then
    COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,pretrained=$HF_CHECKPOINT_PATH,tokenizer=$TOKENIZER$MAYBE_REVISION"
else
    die "Unsupported LM_EVAL_BACKEND: $LM_EVAL_BACKEND"
fi
#,tokenizer_mode=slow"

HARNESS_EVAL_DIR=$HARNESS_DIR/eval_$(date +%Y%m%d_%H%M%S)_$SLURM_JOBID

COMMON_EVAL_ARGS=(
	--trust_remote_code
	--batch_size $BS
	--max_batch_size 32
	--log_samples
	--write_out
	--confirm_run_unsafe_code
    # --cache_requests true - causes problems for long model names due to file name length limits
    --gen_kwargs max_gen_toks=$MAX_NEW_TOKENS
)
# --tasks and --output_path are appended by scripts/_run_per_task.sh.

if [[ -n "${HARNESS_LIMIT}" ]]; then
    COMMON_EVAL_ARGS+=(--limit $HARNESS_LIMIT)
fi

if [[ -n "${NUM_FEWSHOT}" ]]; then
	COMMON_EVAL_ARGS+=(--num_fewshot $NUM_FEWSHOT)
fi

if [[ $APPLY_CHAT_TEMPLATE = true ]]; then
	COMMON_EVAL_ARGS+=(--apply_chat_template)
	COMMON_EVAL_ARGS+=(--fewshot_as_multiturn)
fi

# Detect long context length only when 'ruler' is among tasks.
CONTEXT_LEN=""
if echo "$TASKS" | grep -q "ruler"; then
    if echo "$TASKS" | grep -qE "/[0-9]+"; then
        CONTEXT_LEN=$(echo "$TASKS" | grep -oE '/[0-9]+' | head -n1 | tr -d '/')
    fi
    # Fallback: support underscore or dash pattern like ruler_8192 or ruler-8192
    if [[ -z "$CONTEXT_LEN" ]]; then
        if echo "$TASKS" | grep -qE "ruler[_-][0-9]+"; then
            CONTEXT_LEN=$(echo "$TASKS" | grep -oE 'ruler[_-][0-9]+' | head -n1 | grep -oE '[0-9]+')
        fi
    fi
    if [[ -n "$CONTEXT_LEN" ]]; then
        export MAX_LENGTH="$CONTEXT_LEN"
        # Only add metadata and model arg if CONTEXT_LEN is numeric
        if [[ "$CONTEXT_LEN" =~ ^[0-9]+$ ]]; then
			COMMON_EVAL_ARGS+=( --metadata \'{\"max_seq_lengths\":[${CONTEXT_LEN}]}\' )

            MAX_MODEL_LEN_WITH_MARGIN=$(( CONTEXT_LEN + 600 ))
            export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
            COMMON_MODEL_ARGS="$COMMON_MODEL_ARGS,max_model_len=$MAX_MODEL_LEN_WITH_MARGIN"
            echo "Detected ruler context length: $CONTEXT_LEN (set MAX_LENGTH, max_model_len=$MAX_MODEL_LEN_WITH_MARGIN and lm-eval metadata)"
        else
            echo "Warning: Ruler detected but no numeric context length found; skipping metadata and max_model_len."
        fi
    fi
fi

LM_EVAL_HARNESS_PIP_SPEC="git+https://github.com/swiss-ai/lm-evaluation-harness.git"
if [[ -n "$LM_EVAL_HARNESS_BRANCH" ]]; then
    LM_EVAL_HARNESS_PIP_SPEC="${LM_EVAL_HARNESS_PIP_SPEC}@${LM_EVAL_HARNESS_BRANCH}"
fi

INSTALL_CMD="pip install --no-cache-dir sentencepiece tiktoken protobuf && \
pip install --no-cache-dir --upgrade \"$LM_EVAL_HARNESS_PIP_SPEC\" && \
pip install --no-cache-dir --upgrade \
    \"huggingface-hub>=1.3.0,<2.0\" \
    \"transformers==5.1.0\" \
    \"accelerate>=1.1.0\" \
    \"antlr4-python3-runtime==4.11\" \
    datasketch \
    segtok \
    \"tokenizers==0.22.2\" \
    mistral-common \
    sympy \
    spacy \
    math_verify \
    latex2sympy2_extended \
    wonderwords \
    nltk \
    immutabledict \
    langdetect \
    hf-transfer \
    subset2evaluate \
    \"alpaca-eval>=0.6.2\" \
    \"peft>=0.18.0\" && \
python -c 'import sentencepiece, tiktoken; print(f\"sentencepiece {sentencepiece.__version__} OK\"); print(f\"tiktoken {tiktoken.__version__} OK\")'"

if [[ $LM_EVAL_BACKEND == "megatron_lm" ]]; then
  # The local Megatron-LM enforces strict checkpoint loading and chokes on
  # transformer-engine `_extra_state` fields (fp8 amax history etc.) that
  # are missing from older checkpoints — known upstream issue. The harness
  # passes strict=True at lm_eval/models/megatron_lm.py:541; flip it to
  # strict=False so missing _extra_state keys become warnings instead of
  # fatal RuntimeErrors. Safe for bf16 eval: real weights still load, only
  # TE bookkeeping is skipped.
  PATCH_HARNESS_CMD="LM_EVAL_DIR=\$(python -c 'import lm_eval, os; print(os.path.dirname(lm_eval.__file__))') && \
sed -i 's/load_checkpoint(self._model, None, None, strict=True)/load_checkpoint(self._model, None, None, strict=False)/' \"\$LM_EVAL_DIR/models/megatron_lm.py\" && \
echo \"Patched \$LM_EVAL_DIR/models/megatron_lm.py: load_checkpoint strict=True -> strict=False\""
  INSTALL_CMD="$INSTALL_CMD && pip install megatron-core --no-deps && $PATCH_HARNESS_CMD && export MEGATRON_PATH=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM"
fi

# scripts/eval_worker.py takes the lm_eval CLI's flags and runs the tasks one
# simple_evaluate() call at a time against a model it loads once, writing each
# task's results as it finishes; only the launcher in front of it differs per
# backend (vLLM manages its own GPUs, hf/megatron span the node via
# accelerate/torchrun with a single worker).
if [[ $LM_EVAL_BACKEND == "vllm" ]]; then
    CMD_BASE="python scripts/eval_worker.py --model $LM_EVAL_BACKEND --model_args=$COMMON_MODEL_ARGS ${COMMON_EVAL_ARGS[@]}"
elif [[ $LM_EVAL_BACKEND == "hf" ]]; then
    CMD_BASE="accelerate launch --num_processes=4 scripts/eval_worker.py --model $LM_EVAL_BACKEND --model_args=$COMMON_MODEL_ARGS ${COMMON_EVAL_ARGS[@]}"
elif [[ $LM_EVAL_BACKEND == "megatron_lm" ]]; then
    CMD_BASE="torchrun --nproc-per-node=4 scripts/eval_worker.py --model megatron_lm --model_args=$COMMON_MODEL_ARGS ${COMMON_EVAL_ARGS[@]}"
fi
# scripts/_run_per_task.sh reads CMD_BASE/TASKS/HARNESS_EVAL_DIR from env.
export CMD_BASE TASKS HARNESS_EVAL_DIR NAME WANDB_ENTITY WANDB_PROJECT LOGS_ROOT

echo "Installation command: $INSTALL_CMD"
echo "Worker command base: $CMD_BASE"

# Start timing for lines 182-189
echo "Starting timing for installation and evaluation..."
START_TIME=$(date +%s)

# The container's WORKDIR is /workspace/vllm, not our repo. Cd into the
# repo on the host side so `./scripts/_run_per_task.sh` resolves and the
# `python -m scripts.alignment.*` calls inside it find their package.
REPO_DIR="$PWD"

# pyxis (--environment=...) doesn't forward all host env vars into the
# container, so re-export the ones _run_per_task.sh needs by host-side
# interpolation into the bash -lc heredoc. Single-quote the values so
# arbitrary characters in TASKS/CMD_BASE survive intact.
INNER_EXPORTS="export NAME='$NAME' WANDB_ENTITY='$WANDB_ENTITY' WANDB_PROJECT='$WANDB_PROJECT' LOGS_ROOT='$LOGS_ROOT' TASKS='$TASKS' HARNESS_EVAL_DIR='$HARNESS_EVAL_DIR' CMD_BASE='$CMD_BASE' EVAL_WORKERS='$EVAL_WORKERS' GPUS_PER_WORKER='$(( MP * PP ))' HF_HOME='$HF_HOME' HF_HUB_CACHE='$HF_HUB_CACHE' HF_DATASETS_OFFLINE='$HF_DATASETS_OFFLINE'"

# job.json — what only the job knows, beside the harness's own files: Slurm
# ids, node, walltime, backend/model/parallelism, repo commit, container, and
# at the end the outcome. Written before the eval (a walltime kill still
# leaves the header) and rewritten after it with the counts.
job_json() {
    command -v python3 >/dev/null || return 0
    python3 - "$HARNESS_EVAL_DIR/job.json" "$@" <<'PY'
import json, sys
d = {}
for kv in sys.argv[2:]:
    k, v = kv.split("=", 1)
    d[k] = int(v) if v.lstrip("-").isdigit() else v
json.dump(d, open(sys.argv[1], "w"), indent=1, sort_keys=True)
PY
}
count_lines() { if [ -f "$1" ]; then grep -c . "$1" || true; else echo 0; fi; }
CONTAINER_TOML=./containers/env_vllm.toml
[[ $LM_EVAL_BACKEND == "hf" ]] && CONTAINER_TOML=./containers/env.toml
JOB_FACTS=(job_id="$SLURM_JOBID" job_name="${SLURM_JOB_NAME:-}" node="$SLURM_JOB_NODELIST"
    partition="${SLURM_JOB_PARTITION:-}"
    time_limit="$(squeue -h -j "$SLURM_JOBID" -o %l 2>/dev/null || true)"
    name="$NAME" model="$MODEL" tokenizer="$TOKENIZER" backend="$LM_EVAL_BACKEND"
    tp="$MP" pp="$PP" workers="$EVAL_WORKERS"
    tasks_requested="$(tr ',' '\n' <<<"$TASKS" | grep -c . || true)"
    repo_commit="$(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || true)"
    container="$(sed -n 's/^image = "\(.*\)"/\1/p' "$CONTAINER_TOML" || true)"
    eval_dir="$HARNESS_EVAL_DIR" started="$(date -Iseconds)")
mkdir -p "$HARNESS_EVAL_DIR"
job_json "${JOB_FACTS[@]}" status=started

EVAL_RC=0
if [[ $LM_EVAL_BACKEND == "megatron_lm" ]]; then
  srun -n 1 --mpi=pmix -ul --environment=./containers/env_vllm.toml bash -lc "
    cd '$REPO_DIR'
    $INNER_EXPORTS
    set -euo pipefail
    unset SSL_CERT_FILE

    $INSTALL_CMD
    # Make torch CUDA libs discoverable for vllm._C import
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}
    bash ./scripts/_run_per_task.sh
    " || EVAL_RC=$?
elif [[ $LM_EVAL_BACKEND == "vllm" ]]; then
  # The install and the eval run in the same srun so the pip-installed harness
  # is there for the workers; _run_per_task.sh starts one worker per GPU group.
  srun -n 1 --mpi=pmix -ul --environment=./containers/env_vllm.toml bash -lc "
    cd '$REPO_DIR'
    $INNER_EXPORTS
    set -euo pipefail
    unset SSL_CERT_FILE

    $INSTALL_CMD
    # Make torch CUDA libs discoverable for vllm._C import
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}
    bash ./scripts/_run_per_task.sh
    " || EVAL_RC=$?
else
  # hf (via accelerate)
  srun -n 1 --mpi=pmix -ul --environment=./containers/env.toml --export=ALL,HF_TOKEN,HF_HOME,WANDB_API_KEY bash -lc "
    cd '$REPO_DIR'
    $INNER_EXPORTS
    set -euo pipefail
    unset SSL_CERT_FILE

    $INSTALL_CMD

    # Make torch CUDA libs discoverable for vllm._C import - is just ignored for Huggingface
    export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:\${LD_LIBRARY_PATH:-}

    bash ./scripts/_run_per_task.sh
    " || EVAL_RC=$?
fi

# End timing and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Time taken for installation and evaluation: ${DURATION} seconds ($(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds)"

job_json "${JOB_FACTS[@]}" status=$([[ $EVAL_RC -eq 0 ]] && echo finished || echo failed) \
    exit_code="$EVAL_RC" finished="$(date -Iseconds)" duration_s="$DURATION" \
    tasks_done="$(find "$HARNESS_EVAL_DIR/per_task" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')" \
    tasks_failed="$(count_lines "$HARNESS_EVAL_DIR/failed_tasks.log")" \
    tasks_skipped="$(count_lines "$HARNESS_EVAL_DIR/skipped_tasks.log")" \
    tasks_unfinished="$(find "$HARNESS_EVAL_DIR/inflight" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')"
# A failed eval step means NO task finished (_run_per_task.sh exits 1 only
# then); whatever did land is on disk and the next submission picks it up.
(( EVAL_RC == 0 )) || die "Evaluation step failed (rc=$EVAL_RC) — nothing to upload; see $HARNESS_EVAL_DIR"

# Goodbye.
echo "Evaluation finished"

if (( NUM_SPLITS > 1 )); then
    # When running as a split job, write a marker file with the eval dir path
    # so the aggregation job can find all split results
    SPLIT_MARKER_DIR=$HARNESS_DIR/split_markers
    mkdir -p $SPLIT_MARKER_DIR
    echo "$HARNESS_EVAL_DIR" > "$SPLIT_MARKER_DIR/split_${SPLIT_INDEX}.txt"
    echo "Split job $((SPLIT_INDEX+1))/$NUM_SPLITS done. Skipping W&B upload (aggregation job will handle it)."
    echo "Results in: $HARNESS_EVAL_DIR"
    echo "Marker written to: $SPLIT_MARKER_DIR/split_${SPLIT_INDEX}.txt"
else
    echo "Uploading results to wandb (per-model curve, one step per ckpt)"
    WANDB_CMD="cd $PWD && python scripts/push_all_results.py --entity $WANDB_ENTITY --project $WANDB_PROJECT --name $NAME --eval-duration $DURATION"
    echo "Running command to upload results to wandb:"
    echo "$WANDB_CMD"
    # pyxis doesn't forward host env into the container (same as the eval
    # sruns above) — thread LOGS_ROOT + the W&B key in explicitly, or
    # push_all_results.py silently falls back to its hardcoded eval_logs path.
    # Key export is conditional: an exported-but-EMPTY WANDB_API_KEY would
    # break the ~/.netrc fallback the push used before.
    PUSH_CREDS=""
    [[ -n "${WANDB_API_KEY:-}" ]] && PUSH_CREDS="export WANDB_API_KEY='$WANDB_API_KEY'"
    srun -ul --mpi=pmix --environment=./containers/env.toml bash -c " \
        $INNER_EXPORTS
        $PUSH_CREDS
        $WANDB_CMD
    "
fi
echo "END TIME: $(date)"
