# Megatron → HuggingFace checkpoint conversion

> Two-step pipeline to convert Megatron `torch_dist` checkpoints to HF
> format so they can be evaluated with `lm-evaluation-harness` (via the
> vLLM backend) and pushed to the HF Hub.

The sweep-level driver
([`convert-snr.sh`](convert-snr.sh)) wraps both steps and walks all 36
cells × 13 canonical iters; this README documents the underlying single-
checkpoint flow.

## Step 1 — `torch_dist` → `torch`

If you trained with `--ckpt-format=torch_dist` (the swiss-ai template
default), first convert to the plain `torch` format:

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun scripts/conversion/torchdist_2_torch.py \
    --bf16 \
    --load CHECKPOINT_PATH \
    --ckpt-convert-save INTERMEDIATE_CHECKPOINT_PATH \
    --ckpt-step ITERATION_STEP   # optional; defaults to latest
```

`CHECKPOINT_PATH` is the **root** directory holding all the iter dirs
(`iter_0001000/ iter_0002000/ … latest_checkpointed_iteration.txt
progress.txt`), not an individual iter dir.

If you get `ModuleNotFoundError: No module named 'megatron'`, set
`PYTHONPATH=$PWD`.

For larger models (e.g. 70B), pass `--nproc-per-node=4` to `torchrun`
and `--pipeline-model-parallel-size=4` to the script. The PP value does
**not** carry over to `convert.py` below — don't set it there.

The `INTERMEDIATE_CHECKPOINT_PATH` is consumed by Step 2 and can be
removed afterward.

## Step 2 — `torch` → HuggingFace

```bash
python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader core \
    --saver swissai_hf \
    --load-dir CHECKPOINT_PATH \
    --save-dir SAVE_DIR \
    --hf-tokenizer HF_TOKENIZER_NAME   # optional; saves tokenizer in SAVE_DIR
```

If Step 1 was run, replace `CHECKPOINT_PATH` with
`INTERMEDIATE_CHECKPOINT_PATH`.

Add `--test-logits` to verify that the HF model's logits match the
Megatron implementation (TP1, PP1 only).

Apertus models use a custom `ApertusForCausalLM`. Install the latest
swiss-ai transformers fork before converting:

```bash
git clone https://github.com/swiss-ai/transformers.git
cd transformers
pip install -e .
```

## Use the converted model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    SAVE_DIR, device_map="auto", torch_dtype="auto"
)

prompt = "What's the best way to get in shape?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
output = model.generate(
    inputs.input_ids,
    attention_mask=inputs.attention_mask,
    max_new_tokens=256,
    do_sample=True,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

End-to-end single-checkpoint example: `scripts/conversion/do-convert.sh`.
For the full 36-cell × 13-iter sweep, use the wrapper
[`convert-snr.sh`](convert-snr.sh) — three modes (per-iter, sbatch
wrapper, launcher with `--submit`).
