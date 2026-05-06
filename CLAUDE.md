# CLAUDE.md

## Project Overview

This project extends the Signal-and-Noise (SNR) framework (Heineman et al., 2025) from English-only to multilingual settings.

**Research question:** Which (subsets of) benchmarks provide reliable signal at each stage of multilingual model training?

The project trains 36 small multilingual models (4 sizes x 3 data mixtures x 3 seeds), evaluates them on 40+ benchmarks, computes SNR metrics, and produces stage-specific evaluation recommendations.

## Architecture

```
configs/          # JSON configs for tasks, models, hyperparams, wandb
documents/        # Slidev presentation (scholarly theme) + project documents
scripts/          # Thin wrapper scripts with CLI args, minimal logic
src/
  evals/          # Evaluation harness wrapper (lm_eval integration)
  pretrain/       # Pretraining configs and hyperparameter calculations
  snr/            # SNR metric computation (signal, noise, decision accuracy)
  analysis/       # Statistical analysis and visualization
results/          # Local eval outputs (gitignored)
preliminary-analysis/  # Prior work reference (signal-and-noise repo + thesis PDF)
```

## Key Concepts

- **Signal**: Relative dispersion of benchmark scores across training data mixtures (how well a benchmark separates models): `(max - min) / mean`
- **Noise**: Two variants:
  - Checkpoint noise: Score variability across late training checkpoints (requires many checkpoints)
  - Benchmark noise: k-fold split variability from a single evaluation run (more practical)
- **SNR**: Signal / Noise — higher means more reliable benchmark
- **Decision Accuracy**: Does the benchmark correctly rank model pairs? Pairwise ranking agreement between small and large models
- **Scaling-Law Error**: Can small model results predict large model performance?

## External Dependencies

- **lm_eval** (lm-evaluation-harness): Evaluation framework, used via `lm_eval.simple_evaluate()`
- **signal-and-noise** (Allen AI): Reference implementation at `preliminary-analysis/signal-and-noise/`.
- **wandb**: Experiment tracking — entity `mariagrandury-epflnlp`, project `snr-experiments`

## Configuration

- `configs/tasks.json`: Task lists by training stage (pretraining/midtraining/posttraining)
- `configs/models.json`: Model definitions with HF repo IDs and checkpoint branches
- `configs/wandb.json`: W&B entity and project names
- `configs/hyperparams.json`: Model architecture configs (175M, 350M, 600M, 1B)

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run evaluations locally
python scripts/run_evals_local.py --tasks test --models SmolLM3-3B --names --limit 10 --no-wandb

# Run evaluations on SLURM
sbatch scripts/run_evals_slurm.sh --tasks pretraining --models SmolLM3-3B --names

# Compute SNR metrics
python scripts/compute_snr.py --wandb-project snr-experiments

# Generate analysis plots
python scripts/run_analysis.py --wandb-project snr-experiments

# Import external W&B results
python scripts/import_wandb.py --source ist/SwissAI-QAT-evals --tag QAT

# Slides development
cd documents && npx slidev --open
```

## Conventions

- Source modules (`src/`) contain core logic with no CLI parsing
- Scripts (`scripts/`) are thin wrappers with argparse and minimal logic
- W&B logging is optional (--no-wandb flag) for local development
- Results are saved locally AND to W&B when enabled
- Checkpoint resolution supports: last N, total T (evenly spaced), or named list
- Model sizes: 175M, 350M, 600M, 1B (custom); up to 70B (open-source)
- Data mixtures: FineWeb-Edu (EN) + FineWeb2 (multilingual) in 30/70, 60/40, 90/10 ratios
- Presentation uses Slidev with `slidev-theme-scholarly`, config in `documents/`
