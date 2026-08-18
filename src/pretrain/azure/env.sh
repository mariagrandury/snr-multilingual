# Azure names used by every script and job spec in this azure/ directory.
# Edit once, then `source azure/env.sh` (from src/pretrain) before running any
# command from the guide (azure/README.md).

# Shell compatibility: the guide's `az ml` commands splat $AZ_ML_ARGS (a
# string of two flags) and rely on word splitting. bash splits unquoted
# scalars into separate words; zsh does not by default, so under zsh the two
# flags arrive as ONE argument and az reports "--resource-group/-w required".
# Enable bash-style splitting when this file is sourced into zsh (no-op in
# bash, where $ZSH_VERSION is unset).
if [ -n "${ZSH_VERSION:-}" ]; then
  setopt SH_WORD_SPLIT
fi

# Find yours with: az account list --output table
export AZ_SUBSCRIPTION="ef1ff20e-1168-4846-a78e-47d102dd35f6"

# --- The two workspaces (final region/machine plan) -------------------------
# Decided 2026-08-13 from this subscription's deployable SKUs and real meters
# (see plan/compute-budget.md):
#   Spain Central — NC80adis_H100_v5, low-priority meter ($1.82/GPU-h):
#                   economy pool for every size <=600M, plus evals/conversion.
#   UK South      — ND96isr_H100_v5 (8x H100 + InfiniBand), Spot ($2.84/GPU-h):
#                   the 1B and 1.7B pool.
# An AML compute cluster lives in its workspace's region, hence one workspace
# per region. Run setup_azure.sh once per workspace (it targets AZ_LOCATION/
# AZ_RG/AZ_WS below; computes whose SKU a region lacks are skipped).
export AZ_ES_LOCATION="spaincentral"
export AZ_ES_RG="snr-es-rg"
export AZ_ES_WS="snr-es-ws"
export AZ_UK_LOCATION="uksouth"
export AZ_UK_RG="snr-uk-rg"
export AZ_UK_WS="snr-uk-ws"
export AZ_ML_ARGS_ES="--resource-group $AZ_ES_RG --workspace-name $AZ_ES_WS"
export AZ_ML_ARGS_UK="--resource-group $AZ_UK_RG --workspace-name $AZ_UK_WS"

# Primary workspace = Spain Central; the guide's single-workspace commands and
# AZ_ML_ARGS point here. Re-export these three to the AZ_UK_* values before
# running setup_azure.sh for the UK workspace.
export AZ_LOCATION="$AZ_ES_LOCATION"
export AZ_RG="$AZ_ES_RG"
export AZ_WS="$AZ_ES_WS"

# Compute clusters (created by setup_azure.sh, referenced by job_*_azure.yml):
# gpu-nc80-lp in Spain Central, gpu-nd96-spot in UK South — see the
# compute-*.yml files.

# W&B: the entity is the constant "mariagrandury-epflnlp" (hardcoded in
# megatron_args.sh) and the training project comes from configs/hf_wandb.json
# via launch_trainings.py — nothing to export here. The WANDB_API_KEY is NOT
# in any file and is never committed: it lives in your shell/login on your
# laptop and is read from $WANDB_API_KEY when a job needs it.

# Convenience: every `az ml` call needs these two flags.
export AZ_ML_ARGS="--resource-group $AZ_RG --workspace-name $AZ_WS"
