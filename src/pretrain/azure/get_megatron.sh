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
export MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR${PYTHONPATH:+:$PYTHONPATH}
