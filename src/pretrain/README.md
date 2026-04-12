# Small Multilingual Pretrained Models

Pretraining of small-scale multilingual Apertus models. Initial set:

- 4 model sizes: 100M, 300M, 500M, 1B
- 3 data mixtures of FineWeb2-Edu (English) and FineWeb2-HQ (non-English), ratios: 30/70, 60/40, and 90/10
- 3 seeds per data mixture
- total: 4 sizes × 3 mixtures × 3 seeds = 36 models

## Files

### Set hyperparams

- `calculate_params_lr_bs.py`: Calculates the number of total and non-embedding parameters for a model, as well as the learning rate and batch size to train the model.
- `find_hyperparams.py`: Computes architecture and training hyperparameters for each model size, writes `hyperparams_explanation.txt` and saves final `hyperparams.json`.
- `hyperparams_explanation.txt`: Hyperparameter summary, stdout of `find_hyperparams.py`.
- `hyperparams.json`: Hyperparameter configs consumed by the launch script `launch_trainings.py`.
- `fetch_hf_model_hyperparams.py`: Fetches architecture hyperparameters from reference HuggingFace models for comparison.
- `hf_models.txt`: List of reference HuggingFace model IDs.
- `hf_model_hyperparams.csv`: Saved hyperparameters from reference HuggingFace models.

### Train models

- `launch_trainings.py`: Submits one Slurm job per (model size × data ratio)
  combination via `sbatch --export`, supports `--dry-run` and `--test` flags.
- `submit-apertus-data-mix.sh`: Slurm sbatch template, model-size and data-ratio parameters are injected at submission time via environment variables.

## Launching jobs

```bash
# One command
cd /iopsstor/scratch/cscs/mariagrandury/pretrain/megatron/data-mix-small/ && git pull && conda activate && python launch_trainings.py --mix_en 60 --seed 1904

# See training logs
cd /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/training/ && tail -100 apertus-175m-edu60-fw240-seed1797-1830890.err

grep -i "error" /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/training/apertus-175m-edu60-fw240-seed1797-1830890.err


# See checkpoints
ls /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu60-fw240-seed28/checkpoints/
```

## Debugging

Remove checkpoints that are not n\*2000:

```bash
for dir in /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/*/checkpoints; do
  iter=$(cat "$dir/latest_checkpointed_iteration.txt" 2>/dev/null) || continue
  if (( iter % 2000 != 0 )); then
    last_good=$(( (iter / 2000) * 2000 ))
    echo "Fixing $dir: removing iter_$(printf '%07d' $iter), setting latest to $last_good"
  fi
done
```

```bash
for dir in /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/*/checkpoints; do
  iter=$(cat "$dir/latest_checkpointed_iteration.txt" 2>/dev/null) || continue
  if (( iter % 2000 != 0 )); then
    last_good=$(( (iter / 2000) * 2000 ))
    echo "Fixing $dir: removing iter_$(printf '%07d' $iter), setting latest to $last_good"
    rm -rf "$dir/iter_$(printf '%07d' $iter)"
    echo "$last_good" > "$dir/latest_checkpointed_iteration.txt"
  fi
done
```
