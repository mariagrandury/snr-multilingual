# W&B Visualization Guide

How to create useful plots in the W&B UI to visualize evaluation and SNR results in the `snr-experiments` project.

> **Project URL**: https://wandb.ai/mariagrandury-epflnlp/snr-experiments

## Data Sources in W&B

| Tag | Source | Models | What it contains |
|---|---|---|---|
| `preliminary` | `allenai/signal-and-noise` HF dataset | 109 DataDecide runs (150M–1B) | 244 English tasks, 28 data mixes, multi-checkpoint |
| `QAT` | `ist/SwissAI-QAT-evals` W&B import | 22 Apertus/SmolLM/EuroLLM runs | 572 tasks (incl. multilingual), training variant checkpoints |
| `eval` | Our own evaluations | Custom models + open-source | Tasks from configs/tasks.json, logged per checkpoint |
| `snr` | SNR analysis runs | — | SNR result tables (signal, noise, DA per task) |

## Recommended Plots

### 1. Score Progression Across Checkpoints

Shows how a benchmark score evolves during training.

**Setup:**
- **Panel type**: Line plot
- **X-axis**: `checkpoint_index`
- **Y-axis**: Pick a task metric, e.g., `hellaswag/primary_score` or `mmlu/acc_norm`
- **Group by**: `run.group` (groups runs by model family)
- **Filter**: Tag = `preliminary` (or `QAT`)

This reveals whether a benchmark tracks training progress smoothly or is noisy.

### 2. Model Comparison Table

Compare all models on key benchmarks at their final checkpoint.

**Setup:**
- **Panel type**: Table
- **Columns**: Pick summary metrics like `hellaswag/primary_score`, `mmlu/primary_score`, `arc_easy/primary_score`
- **Group by**: `config.model_size`
- **Sort by**: Any metric column
- **Filter**: Tag = `preliminary`, then refine by size

### 3. Size Scaling Plot

Shows how benchmark scores scale with model size.

**Setup:**
- **Panel type**: Scatter plot
- **X-axis**: `config.model_size` (or sort runs by size)
- **Y-axis**: Target metric, e.g., `mmlu/primary_score`
- **Color by**: `config.data_mix`
- **Filter**: Tag = `preliminary`, pick a single `checkpoint_index` (e.g., the max)

### 4. Data Mix Comparison

Compare how different data mixtures affect benchmark scores.

**Setup:**
- **Panel type**: Bar chart or grouped line plot
- **X-axis**: `config.data_mix`
- **Y-axis**: Target metric
- **Group by**: `config.model_size`
- **Filter**: Tag = `preliminary`

This directly shows the "signal" — the separation across data mixes.

### 5. SNR Results Table

After running `compute_snr.py` with `--no-wandb` off, the SNR analysis run logs a `snr_results` table.

**Setup:**
- Go to the SNR analysis run (tag = `snr`)
- Open the **Tables** tab
- Sort by `snr` descending to see most reliable benchmarks
- Filter by `small_size` to compare proxy scales

### 6. Multilingual Task Comparison (QAT)

**Setup:**
- **Panel type**: Bar chart
- **X-axis**: Run names (each run = one model variant)
- **Y-axis**: Pick multilingual tasks: `xcopa_id/acc`, `hellaswag_vi/acc_norm`, `xnli_th/acc`
- **Filter**: Tag = `QAT`

Shows which training variants (base, SFT, long) perform best on multilingual tasks — this is the "signal" in the QAT analysis.

## Creating a Dashboard

1. Go to **snr-experiments** → **Workspaces**
2. Create sections:
   - **Pretraining SNR**: Score progressions + size scaling for preliminary data
   - **QAT Multilingual**: Multilingual task comparison + training variant analysis
   - **SNR Analysis**: Link to SNR result tables from analysis runs
3. Pin key charts to the workspace for quick access

## Useful Filters

| Filter | Use case |
|---|---|
| `tag:preliminary` | DataDecide models only |
| `tag:QAT` | Apertus/SwissAI QAT models only |
| `tag:snr` | SNR analysis result runs |
| `jobType:import` | All imported (non-native) runs |
| `jobType:eval` | Our own evaluation runs |
| `config.model_size:150M` | Specific model size |
| `config.data_mix:DCLM-baseline` | Specific data mixture |

## Key Metrics to Track

### For English benchmarks (preliminary tag)
- `arc_easy/primary_score` — highest SNR (74.4), highest DA (94.7%)
- `mmlu/primary_score` — SNR 41.2, DA 90.7%
- `hellaswag/primary_score` — SNR 41.5, DA 80.3%
- `arc_challenge/primary_score` — SNR 32.7, DA 85.2%

### For multilingual benchmarks (QAT tag)
- `xcopa_id/acc` — SNR 16.7, DA 100%
- `hellaswag_vi/acc_norm` — SNR 10.3, DA 100%
- `xnli_th/acc` — SNR 8.2, DA 100%
- `hellaswag_bn/acc_norm` — SNR 8.3, DA 100%

## Reproducing Results

```bash
# Import AllenAI data and push to W&B
python scripts/import_hf_to_wandb.py --split core --tag preliminary --match-sizes

# Import QAT data from W&B (already done if QAT tag exists)
python scripts/import_wandb.py --source ist/SwissAI-QAT-evals --tag QAT

# Compute SNR and generate figures locally
python scripts/analyze_imported.py --output results/figures/

# Compute SNR for a specific stage and push to W&B
python scripts/compute_snr.py --stage pretraining --output results/snr/

# Generate analysis plots from saved CSVs
python scripts/run_analysis.py --input results/snr/ --output results/figures/
```
