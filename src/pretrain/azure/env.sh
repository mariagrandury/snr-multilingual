# Azure names used by every script and job in this directory.
# Edit once, then `source env.sh` before running any command from the guide.

# Find yours with: az account list --output table
export AZ_SUBSCRIPTION="ef1ff20e-1168-4846-a78e-47d102dd35f6"

# --- The two workspaces (final region/machine plan) -------------------------
# Decided 2026-08-13 from this subscription's deployable SKUs and real meters
# (see .claude-shared/plans/predictivity-compute-budget.md):
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

# Compute cluster names (created by setup_azure.sh, referenced by jobs/*.yml).
# gpu-train/gpu-single are the guide's original A100 clusters — their SKU
# (NCads_A100_v4) is not offered in Spain Central, so setup skips them there;
# the H100 pools (gpu-nc80-lp / gpu-nd96-spot) are the ones the plan uses.
export AZ_COMPUTE_TRAIN="gpu-train"   # 4x A100 80GB — data prep + training
export AZ_COMPUTE_SINGLE="gpu-single" # 1x A100 80GB — smoke test, conversion, eval

# W&B (primary monitoring). Set the key in your shell, never in a committed file:
#   export WANDB_API_KEY=...
export WANDB_ENTITY="mariagrandury-epflnlp"

# Convenience: every `az ml` call needs these two flags.
export AZ_ML_ARGS="--resource-group $AZ_RG --workspace-name $AZ_WS"
