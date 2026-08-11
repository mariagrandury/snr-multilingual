#!/bin/bash
# One-time Azure setup: resource group, ML workspace, environments, compute.
# Idempotent — safe to re-run. Requires: az CLI, `source env.sh`, and GPU
# quota already granted in $AZ_LOCATION (README step 3).
set -euo pipefail
cd "$(dirname "$0")"

: "${AZ_SUBSCRIPTION:?source env.sh first}"

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

az ml compute create --file compute-train.yml $AZ_ML_ARGS --output none
echo "Compute $AZ_COMPUTE_TRAIN ready"
# The 1-GPU cluster needs 24 extra vCPUs of quota; skip gracefully if denied
# (README explains how to point every job at gpu-train instead).
az ml compute create --file compute-single.yml $AZ_ML_ARGS --output none \
  || echo "WARNING: could not create $AZ_COMPUTE_SINGLE (quota?). Use $AZ_COMPUTE_TRAIN for all jobs."

echo "Setup complete."
