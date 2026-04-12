# SNR Multilingual - LM Evaluation Harness Runner

## Overview

Evaluates HuggingFace model checkpoints on lm-evaluation-harness tasks. Results saved locally and pushed to W&B (entity: mariagrandury-epflnlp, project: snr-experiments).

## Usage

### Local (CPU/testing)

```bash
python scripts/run_local.py --tasks test --models test --last 2 --limit 2
python scripts/run_local.py --tasks pretraining --models pretraining --total 5
python scripts/run_local.py --tasks test --models test --names --no-wandb
```

### SLURM

Set up the environment:

```bash
conda create -n snr python=3.11 -y
conda activate snr
pip install -r requirements.txt
```

Launch a quick test (<10s):

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/ \
&& git pull \
&& sbatch --time=00:30:00 /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/scripts/run_slurm.sh --tasks test --models test --last 2 --limit 2
```

Review the logs:

```bash
cd /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs && ls
```

Evaluate multiple checkpoints:

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/ \
&& git pull \
&& sbatch --time=02:00:00 /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/scripts/run_slurm.sh --tasks pretraining --models SmolLM3-3B --total 10
```

Evaluate multiple checkpoints in parallel:

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/ \
&& git pull && \
for i in $(seq 0 9); do
    sbatch --time=04:00:00 --job-name="eval_SmolLM3_${i}" \
        scripts/run_slurm.sh --tasks pretraining --models SmolLM3-3B --total 10 --checkpoint-index "$i"
done
```

### Import from another WandB project

```bash
python scripts/import_wandb.py --source ist/SwissAI-QAT-evals --tag QAT 2>&1
```

## Checkpoint strategies

- `--last N`: last N branches (alphabetically sorted, excludes 'main')
- `--total T`: T evenly spaced from all branches
- `--names`: use exact names from `"checkpoints"` key in models.json

## Project structure

- `configs/` - tasks.json and models.json define what to evaluate
- `src/` - core logic (config loading, checkpoint resolution, evaluation)
- `scripts/` - thin runner wrappers (local + SLURM)
- `results/` - local output (gitignored)
