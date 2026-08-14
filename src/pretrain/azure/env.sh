# Azure names used by every script and job in this directory.
# Edit once, then `source env.sh` before running any command from the guide.

# Find yours with: az account list --output table
export AZ_SUBSCRIPTION="00000000-0000-0000-0000-000000000000"

# Region must be one where you obtained NCADS_A100_v4 quota (see README step 3).
export AZ_LOCATION="francecentral"

export AZ_RG="snr-azure-rg"          # resource group (the "project folder")
export AZ_WS="snr-azure-ws"          # Azure ML workspace

# Compute cluster names (created by setup_azure.sh, referenced by jobs/*.yml)
export AZ_COMPUTE_TRAIN="gpu-train"   # 4x A100 80GB — data prep + training
export AZ_COMPUTE_SINGLE="gpu-single" # 1x A100 80GB — smoke test, conversion, eval

# W&B (primary monitoring). Set the key in your shell, never in a committed file:
#   export WANDB_API_KEY=...
export WANDB_ENTITY="mariagrandury-epflnlp"

# Convenience: every `az ml` call needs these two flags.
export AZ_ML_ARGS="--resource-group $AZ_RG --workspace-name $AZ_WS"

# --- Predictivity sweep: two workspaces -------------------------------------
# Spain Central (NC80adis H100, low-priority — economy pool: <=600M + evals)
# and UK South (ND96isr 8xH100, Spot — 1B/1.4B pool). An AML compute cluster
# lives in its workspace's region, hence one workspace per region. Run
# setup_azure.sh once per workspace by exporting AZ_LOCATION/AZ_RG/AZ_WS to
# the values below before calling it.
export AZ_ES_LOCATION="spaincentral"
export AZ_ES_RG="snr-es-rg"
export AZ_ES_WS="snr-es-ws"
export AZ_UK_LOCATION="uksouth"
export AZ_UK_RG="snr-uk-rg"
export AZ_UK_WS="snr-uk-ws"
export AZ_ML_ARGS_ES="--resource-group $AZ_ES_RG --workspace-name $AZ_ES_WS"
export AZ_ML_ARGS_UK="--resource-group $AZ_UK_RG --workspace-name $AZ_UK_WS"
