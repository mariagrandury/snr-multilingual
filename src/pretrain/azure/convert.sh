#!/bin/bash
# Azure ML conversion entrypoint: Megatron torch_dist checkpoint -> HF
# (ApertusForCausalLM). Same two steps as ../conversion/convert-snr.sh mode 1,
# without the SLURM/plan-file machinery. Runs on 1 GPU in the apertus-nemo
# container (jobs/convert.yml).
#
# Required env: CKPT_ROOT (dir holding iter_* subdirs), CKPT_STEP, HF_OUT.
set -euo pipefail

cd "$(dirname "$0")"
source get_megatron.sh

HF_TOKENIZER=${HF_TOKENIZER:-alehc/swissai-tokenizer}
export HF_HOME=${HF_HOME:-/tmp/hf_home}
TMP_TORCH=/tmp/torch_ckpt

# The saver needs a transformers with ApertusForCausalLM (see ../conversion/README.md);
# installed only inside this job's container, nothing else is affected.
pip install --no-cache-dir transformers==4.57.6

# Step 1: torch_dist -> plain torch (--load is the checkpoints ROOT, not one iter dir)
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc-per-node=1 \
    "$MEGATRON_LM_DIR/scripts/conversion/torchdist_2_torch.py" \
    --bf16 --load "$CKPT_ROOT" --ckpt-step "$CKPT_STEP" --ckpt-convert-save "$TMP_TORCH"

# Step 2: torch -> HF
python "$MEGATRON_LM_DIR/tools/checkpoint/convert.py" \
    --model-type GPT --loader core --saver swissai_hf \
    --load-dir "$TMP_TORCH/torch" --save-dir "$HF_OUT" \
    --hf-tokenizer "$HF_TOKENIZER"

echo "Converted iter $CKPT_STEP -> $HF_OUT"
ls -la "$HF_OUT"

# Written LAST (same contract as convert-snr.sh): the watcher treats a
# snapshot as converted only once this exists — config.json lands on the
# rw_mount long before the weight shards, and a preempted spot job must not
# leave a half-written snapshot that looks done forever. Non-empty on
# purpose: blobfuse has been flaky about flushing 0-byte creates.
echo done > "$HF_OUT/.hf_complete"
