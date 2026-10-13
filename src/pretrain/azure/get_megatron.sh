# Fetch the swiss-ai Megatron-LM fork at the commit this project's data
# tooling was written against (cited in ../create_data_mixture.py).
# Sourced (not executed) by train.sh / convert.sh / the data-prep job so
# MEGATRON_LM_DIR and PYTHONPATH land in the caller's environment.
MEGATRON_COMMIT=c92402e39ef3c8e69ea378a59e79059dc14541f4
MEGATRON_LM_DIR=${MEGATRON_LM_DIR:-/tmp/Megatron-LM}

if [ ! -f "$MEGATRON_LM_DIR/pretrain_gpt.py" ]; then
  git init -q "$MEGATRON_LM_DIR"
  git -C "$MEGATRON_LM_DIR" remote add origin https://github.com/swiss-ai/Megatron-LM.git
  git -C "$MEGATRON_LM_DIR" fetch -q --depth 1 origin "$MEGATRON_COMMIT"
  git -C "$MEGATRON_LM_DIR" checkout -q FETCH_HEAD
fi

# Same patch the CSCS checkout carries (README "Before the first CSCS run"):
# without it any resume that needs the legacy-metadata fallback dies in
# get_reformulation_metadata, and "both platforms run the same training code"
# is false by construction. BASH_SOURCE, not $0: this file is `source`d.
cp "$(dirname "${BASH_SOURCE[0]}")/../patches/dist_checkpointing_strategies_torch.py" \
   "$MEGATRON_LM_DIR/megatron/core/dist_checkpointing/strategies/torch.py"
# Lets MEGATRON_EXIT_ON_SIGTERM=1 add SIGTERM to the signals --exit-signal-handler
# catches, so a preempted run checkpoints instead of dying where it stands. Inert
# unless that variable is set, and nothing on Azure sets it today — it is copied
# for parity, so a low-priority preemption here can use the same mechanism.
cp "$(dirname "${BASH_SOURCE[0]}")/../patches/training_dist_signal_handler.py" \
   "$MEGATRON_LM_DIR/megatron/training/dist_signal_handler.py"
# The HF saver writes a swiglu checkpoint as Qwen3ForCausalLM (Apertus' MLP is
# ungated, so the stock saver dropped every gate_proj and still reported
# success), and raises when its reload drops or misses a weight. convert.py
# runs the saver in a child process and exits 0 regardless, so convert.sh and
# convert-snr.sh refuse to mark a save dir with no config.json complete; a
# failed --test-logits check (asserted after the save) is still not caught.
cp "$(dirname "${BASH_SOURCE[0]}")/../patches/tools_checkpoint_saver_swissai_hf.py" \
   "$MEGATRON_LM_DIR/tools/checkpoint/saver_swissai_hf.py"
# --optimizer muon, ported from upstream NVIDIA Megatron-LM (Newton-Schulz
# vendored from Emerging-Optimizers v0.3.0, which the image lacks). Inert for
# every other --optimizer: adds args/config fields and one dispatch branch.
cp "$(dirname "${BASH_SOURCE[0]}")/../patches/optimizer_muon.py" \
   "$MEGATRON_LM_DIR/megatron/core/optimizer/muon.py"
cp "$(dirname "${BASH_SOURCE[0]}")/../patches/optimizer_optimizer_config.py" \
   "$MEGATRON_LM_DIR/megatron/core/optimizer/optimizer_config.py"
cp "$(dirname "${BASH_SOURCE[0]}")/../patches/training_arguments.py" \
   "$MEGATRON_LM_DIR/megatron/training/arguments.py"
cp "$(dirname "${BASH_SOURCE[0]}")/../patches/training_training.py" \
   "$MEGATRON_LM_DIR/megatron/training/training.py"
export MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR${PYTHONPATH:+:$PYTHONPATH}
