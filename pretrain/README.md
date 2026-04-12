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

1. cd into the folder and switch to the branch

```bash
cd /iopsstor/scratch/cscs/mariagrandury/pretrain/
git switch data-mix-small
```

2. Review the `--time` and `--nodes`

3. Launch the job with sbatch:

````bash
sbatch pretrain/megatron/data-mix-small/submit-apertus-data-mix.sh
```

or in an interactive node:

```bash
srun \
 --account=infra01 \
 --time=0:09:59 \
 --job-name=apertus-100m-test \
 --nodes=1 \
 --ntasks-per-node=4 \
 --gpus-per-node=4 \
 --cpus-per-task=72 \
 --mem=460000 \
 --mpi=pmix \
 --network=disable_rdzv_get \
 --environment=/capstor/store/cscs/swissai/a139/containers/ngc_25-11-nemo-alps3.toml \
 --output=/iopsstor/scratch/cscs/%u/data-mix-small/Megatron-LM/logs/slurm/training/%x-%j.out \
 --error=/iopsstor/scratch/cscs/%u/data-mix-small/Megatron-LM/logs/slurm/training/%x-%j.err \
 --signal=SIGUSR2@600 \
 --kill-on-bad-exit=1 \
 /iopsstor/scratch/cscs/mariagrandury/pretrain/megatron/data-mix-small/submit-apertus-data-mix.sh
````

The command includes all `sbatch` directives except `--no-requeue` because
`srun: unrecognized option '--no-requeue'`.

4. The output and error will be saved under:

```bash
cd /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/slurm/training
```

## Launching job

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
