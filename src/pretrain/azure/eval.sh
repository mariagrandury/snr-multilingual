#!/bin/bash
# Azure ML lm-eval entrypoint — the vLLM branch of
# src/evals/scripts/evaluate.sbatch with the SNR pretraining runner defaults
# (BOS=true, APPLY_CHAT_TEMPLATE=false, TOKENIZER=alehc/swissai-tokenizer, see
# runners/snr_pretraining_local_hf.sh), running the cluster's own
# scripts/_run_per_task.sh worker pool. Same knobs, same argument surface,
# same results layout, so the repo's downstream tooling (push_all_results.py,
# _eval_status.py) reads the output unchanged.
#
# Required env: MODEL (HF snapshot dir or hub repo id), NAME (results id —
# must be "<configs/models.json key>-iter<N>" for the W&B push to resolve),
# RESULTS_DIR.
set -euo pipefail

TASKS=${TASKS:-hellaswag}                 # comma-separated lm-eval task names
TOKENIZER=${TOKENIZER:-alehc/swissai-tokenizer}
BOS=${BOS:-true}
APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
REVISION=${REVISION:-}                    # HF branch, for hub checkpoint repos
BS=${BS:-auto:20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
NUM_FEWSHOT=${NUM_FEWSHOT:-}
HARNESS_LIMIT=${HARNESS_LIMIT:-}          # e.g. 10 for a smoke eval
TP=${TP:-1}                               # TP=1 works for every SNR size
PP=${PP:-1}                               # (350M/1B have 5/7 KV heads)
VLLM_MEMORY_FRACTION=${VLLM_MEMORY_FRACTION:-0.75}
WANDB_ENTITY=mariagrandury-epflnlp        # constant — same entity as training
# Eval curves land in the SAME W&B project as the training runs (msnr) —
# single source of truth is configs/hf_wandb.json in the repo-root snapshot.
REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
WANDB_PROJECT=${WANDB_PROJECT:-$(python3 -c "import json;print(json.load(open('$REPO_ROOT/configs/hf_wandb.json'))['wandb']['project'])" 2>/dev/null || echo msnr)}

export HF_HOME=${HF_HOME:-/tmp/hf_home}
export HF_ALLOW_CODE_EVAL=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_DISABLE_COMPILE_CACHE=1

# Task definitions come from the swiss-ai lm-eval fork (plain `hellaswag` also
# exists upstream, but hellaswag_eu / multiblimp_* etc. are fork-only).
# The cluster additionally pins transformers==5.1.0 inside its own vLLM-fork
# image; here we keep the versions the stock vLLM image ships (Apertus needs
# transformers>=4.56, which it has) to avoid breaking vLLM.
pip install --no-cache-dir "lm-eval @ git+https://github.com/swiss-ai/lm-evaluation-harness.git${LM_EVAL_HARNESS_BRANCH:+@$LM_EVAL_HARNESS_BRANCH}" \
    sentencepiece tiktoken protobuf

# Model args, as evaluate.sbatch builds COMMON_MODEL_ARGS for the vllm
# backend. enable_thinking=False is a swiss-ai-vLLM-fork kwarg and is dropped
# on stock vLLM; harmless for base models, which have no chat template.
COMMON_MODEL_ARGS="dtype=bfloat16"
[ "$BOS" = true ] && COMMON_MODEL_ARGS+=",add_bos_token=True"
COMMON_MODEL_ARGS+=",pretrained=$MODEL,tokenizer=$TOKENIZER"
[ -n "$REVISION" ] && COMMON_MODEL_ARGS+=",revision=$REVISION,tokenizer_revision=$REVISION"
COMMON_MODEL_ARGS+=",data_parallel_size=1,tensor_parallel_size=$TP,pipeline_parallel_size=$PP"
COMMON_MODEL_ARGS+=",gpu_memory_utilization=$VLLM_MEMORY_FRACTION"

COMMON_EVAL_ARGS=(
  --trust_remote_code
  --batch_size "$BS"
  --max_batch_size 32
  --log_samples
  --write_out
  --confirm_run_unsafe_code
  --gen_kwargs max_gen_toks=$MAX_NEW_TOKENS
)
[ -n "$HARNESS_LIMIT" ] && COMMON_EVAL_ARGS+=( --limit "$HARNESS_LIMIT" )
[ -n "$NUM_FEWSHOT" ] && COMMON_EVAL_ARGS+=( --num_fewshot "$NUM_FEWSHOT" )
[ "$APPLY_CHAT_TEMPLATE" = true ] && COMMON_EVAL_ARGS+=( --apply_chat_template --fewshot_as_multiturn )

# Cluster results layout: $LOGS_ROOT/<entity>/<project>/<NAME>/harness/eval_<ts>_<id>
HARNESS_EVAL_DIR="$RESULTS_DIR/$WANDB_ENTITY/$WANDB_PROJECT/$NAME/harness/eval_$(date +%Y%m%d_%H%M%S)_${AZUREML_RUN_ID:-azure}"
mkdir -p "$HARNESS_EVAL_DIR"

# Same inner runner as the cluster: scripts/_run_per_task.sh starts one
# eval_worker.py per GPU group (each loads the model once and writes every
# task's results the moment it finishes, then merges them into the eval dir),
# so a preempted Spot job keeps what it finished and the watcher's next
# submission runs only the rest — the results blobs it gates on include the
# per-task files.
GPUS=$(nvidia-smi -L 2>/dev/null | grep -c . || true)
EVAL_WORKERS=${EVAL_WORKERS:-$(( ${GPUS:-1} / (TP * PP) ))}
(( EVAL_WORKERS > 0 )) || EVAL_WORKERS=1
CMD_BASE="python scripts/eval_worker.py --model vllm --model_args=$COMMON_MODEL_ARGS ${COMMON_EVAL_ARGS[*]}"
export CMD_BASE TASKS HARNESS_EVAL_DIR NAME WANDB_ENTITY WANDB_PROJECT EVAL_WORKERS
export GPUS_PER_WORKER=$(( TP * PP )) LOGS_ROOT="$RESULTS_DIR"
(cd "$REPO_ROOT/src/evals" && bash scripts/_run_per_task.sh)

echo "Results in $HARNESS_EVAL_DIR:"
ls "$HARNESS_EVAL_DIR"
python - "$HARNESS_EVAL_DIR" <<'EOF'
import json, sys
from pathlib import Path
results = sorted(Path(sys.argv[1]).glob("results_*.json"))[-1]
for task, metrics in json.load(open(results))["results"].items():
    print(task, {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
EOF

# Push to W&B (same project as the training runs — msnr) right from the job —
# required for auto-evals, convenient for manual ones. Needs the repo-root
# code snapshot (configs/ + src/evals) and a WANDB_API_KEY.
if [ -n "${WANDB_API_KEY:-}" ] && [ -f "$REPO_ROOT/configs/models.json" ]; then
  pip install --no-cache-dir -q wandb pandas
  LOGS_ROOT="$RESULTS_DIR" python "$REPO_ROOT/src/evals/scripts/push_all_results.py" \
      --entity "$WANDB_ENTITY" --project "$WANDB_PROJECT" --name "$NAME"
else
  echo "Skipping W&B push (no WANDB_API_KEY or repo snapshot); push locally via push_all_results.py"
fi
