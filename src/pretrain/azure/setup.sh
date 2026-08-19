#!/bin/bash
# One-time Azure setup: resource group, ML workspace, environments, compute.
# Idempotent — safe to re-run. Requires: az CLI, `source azure/env.sh`, and GPU
# quota already granted in $AZ_LOCATION (README step 3).
set -euo pipefail
cd "$(dirname "$0")"

: "${AZ_SUBSCRIPTION:?source azure/env.sh first}"

# env.sh expands AZ_ML_ARGS once at source time from the Spain defaults; the
# UK bring-up re-exports only AZ_LOCATION/AZ_RG/AZ_WS, so recompute here or
# the environment/compute registrations below silently land in Spain while
# the UK workspace stays empty.
AZ_ML_ARGS="--resource-group $AZ_RG --workspace-name $AZ_WS"

az account set --subscription "$AZ_SUBSCRIPTION"
az extension add --name ml --upgrade --yes

# Resource providers (no-op if already registered; first registration can take minutes)
for ns in Microsoft.MachineLearningServices Microsoft.Storage Microsoft.KeyVault \
          Microsoft.ContainerRegistry Microsoft.insights; do
  az provider register --namespace "$ns" --wait
done

az group create --name "$AZ_RG" --location "$AZ_LOCATION" --output none
echo "Resource group $AZ_RG ready"

# Workspace auto-creates its storage account (-> the `workspaceblobstore`
# datastore all jobs write to), key vault, and app insights.
az ml workspace create --name "$AZ_WS" --resource-group "$AZ_RG" --location "$AZ_LOCATION" --output none
echo "Workspace $AZ_WS ready"

az ml environment create --file environment-train.yml $AZ_ML_ARGS --output none
az ml environment create --file environment-eval.yml $AZ_ML_ARGS --output none
echo "Environments ready"

# Each compute needs its SKU offered in the workspace's region plus quota —
# create whatever this region supports and skip the rest with a warning
# (the NC80adis cluster exists in Spain Central, the ND96isr cluster only
# in UK South).
for c in compute-*.yml; do
  az ml compute create --file "$c" $AZ_ML_ARGS --output none \
    && echo "Compute $c ready" \
    || echo "WARNING: $c not created here (SKU not offered in this region, or no quota yet)"
done

echo "Setup complete."
