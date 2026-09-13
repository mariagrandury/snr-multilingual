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

# --- The workspaces (region/machine plan) -----------------------------------
# Decided 2026-08-13 from this subscription's deployable SKUs and real meters
# (see plan/compute-budget.md):
#   Spain Central — NC80adis_H100_v5, low-priority meter ($1.82/GPU-h):
#                   economy pool for every size <=600M, plus evals/conversion.
#   UK South      — ND96isr_H100_v5 (8x H100 + InfiniBand), Spot ($2.84/GPU-h):
#                   the 1B and 1.7B pool.
#
# Canada Central added 2026-08-26. After 13 days and 23 quota requests the
# subscription still holds 0 dedicated H100 cores, so the ND pool moves to the
# region that is actually usable today (`quota_status.sh board`):
#   * ND96isr_H100_v5 is allow-list clear here (UK South is too, but Italy
#     North and Norway East have NO retail meters at all — quota there could
#     never be billed, which is why those three big tickets cannot land).
#   * Its meters are the cheapest of any clear region: Spot $21.80/node-h
#     ($2.73/GPU-h) vs UK South's $23.53, and dedicated $122.40 vs $122.90.
#
# NOTE (2026-08-26): the 300 regional low-priority vCPUs this region reports
# CANNOT be used for H100. AML rejects every H100 SKU for low-priority with
# UnsupportedVMSizeForLowPriority, so DEDICATED quota is the only path — see
# compute-nd96-spot.yml. The Spot/Low-Priority meters in the price list are
# real but unreachable through AmlCompute for these sizes.
#
# An AML compute cluster lives in its workspace's region, hence one workspace
# per region. Run setup.sh once per workspace (it targets AZ_LOCATION/
# AZ_RG/AZ_WS below; computes whose SKU a region lacks are skipped).
export AZ_ES_LOCATION="spaincentral"
export AZ_ES_RG="snr-es-rg"
export AZ_ES_WS="snr-es-ws"
export AZ_UK_LOCATION="uksouth"
export AZ_UK_RG="snr-uk-rg"
export AZ_UK_WS="snr-uk-ws"
export AZ_CA_LOCATION="canadacentral"
export AZ_CA_RG="snr-ca-rg"
export AZ_CA_WS="snr-ca-ws"
export AZ_ML_ARGS_ES="--resource-group $AZ_ES_RG --workspace-name $AZ_ES_WS"
export AZ_ML_ARGS_UK="--resource-group $AZ_UK_RG --workspace-name $AZ_UK_WS"
export AZ_ML_ARGS_CA="--resource-group $AZ_CA_RG --workspace-name $AZ_CA_WS"

# Switch the whole shell to the Canada Central (ND) workspace:
#   source azure/env.sh && use_ca
use_ca() {
  export AZ_LOCATION="$AZ_CA_LOCATION" AZ_RG="$AZ_CA_RG" AZ_WS="$AZ_CA_WS"
  export AZ_ML_ARGS="$AZ_ML_ARGS_CA"
  echo "workspace -> $AZ_WS ($AZ_LOCATION)"
}

# Primary workspace = Spain Central; the guide's single-workspace commands and
# AZ_ML_ARGS point here. For the ND workspace use `use_ca` above — AZ_ML_ARGS
# below is expanded once at source time, so the guide's `az ml ... $AZ_ML_ARGS`
# commands keep pointing at Spain otherwise (setup.sh recomputes it itself).
# The AZ_UK_* values are kept because the UK workspace still exists (it holds
# the older uploads); nothing in the sweep targets it any more.
export AZ_LOCATION="$AZ_ES_LOCATION"
export AZ_RG="$AZ_ES_RG"
export AZ_WS="$AZ_ES_WS"

# Compute clusters (created by setup.sh, referenced by jobs/*.yml):
# gpu-nc80-lp in Spain Central; gpu-nd96-spot and the A100 fallbacks
# (gpu-nc96-a100-lp / gpu-nc96-a100-ded) in Canada Central — see the
# compute-*.yml files.

# W&B: the entity is the constant "mariagrandury-epflnlp" (hardcoded in
# megatron_args.sh) and the training project comes from configs/hf_wandb.json
# via launch_trainings.py — nothing to export here. The WANDB_API_KEY is NOT
# in any file and is never committed: it lives in your shell/login on your
# laptop and is read from $WANDB_API_KEY when a job needs it.

# Convenience: every `az ml` call needs these two flags.
export AZ_ML_ARGS="--resource-group $AZ_RG --workspace-name $AZ_WS"
