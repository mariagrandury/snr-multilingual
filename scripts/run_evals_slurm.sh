#!/bin/bash
#SBATCH --account=infra01
#SBATCH --job-name=eval_snr
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=200
#SBATCH --gres=gpu:4
#SBATCH --output=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/%x_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --partition=normal
#SBATCH --time=04:00:00

# ── Parse our custom args from those forwarded to the Python script ──
# Usage: sbatch scripts/run_evals_slurm.sh --tasks KEY --models KEY --last N [--limit L] [--time HH:MM:SS]

PYTHON_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --time)
            # Already handled by SBATCH override: sbatch --time=VALUE scripts/run_evals_slurm.sh ...
            shift 2
            ;;
        *)
            PYTHON_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Environment setup ──
set -euo pipefail
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Date: $(date)"
echo "Args: ${PYTHON_ARGS[*]}"

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate snr

# ── Run evaluation ──
REPO_DIR=/iopsstor/scratch/cscs/mariagrandury/snr-multilingual
cd "$REPO_DIR"

python scripts/run_evals_local.py \
    --device cuda \
    --batch-size auto \
    "${PYTHON_ARGS[@]}"

echo "Done: $(date)"
