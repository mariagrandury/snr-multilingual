# Implementation Report

Rationale, decisions, and lessons from implementing the SNR framework and analyzing preliminary data.

## 1. SNR Module Design (`src/snr`)

### Architecture: three files, clean separation

| File | Role | Depends on |
|---|---|---|
| `metrics.py` | Pure math on numpy arrays | numpy only |
| `data.py` | I/O: load from W&B, HF, local JSON/CSV | pandas, huggingface_hub, wandb |
| `compute.py` | Orchestration: iterate tasks × sizes, call metrics | metrics.py, data.py |

**Rationale**: The preliminary analysis (Allen AI `signal-and-noise` repo) mixes I/O, metric computation, and constants into a single `snr_simple.py`. The multilingual analysis (`analyse_results.py`) does the same. By separating concerns, we can:
- Unit-test metrics independently (no data dependencies)
- Swap data sources (W&B vs HF vs local) without touching metrics
- Reuse `compute_all_metrics()` from both `compute_snr.py` and `analyze_imported.py`

### Metric formulas: faithful to preliminary, validated empirically

**Signal** — `metrics.relative_dispersion()`:
- Formula: `(max - min) / mean` over per-mix mean scores
- Chosen over `relative_spread` (std/mean) based on preliminary finding: R=0.811 vs 0.791
- Identical to `signal_to_noise_ratio()` signal part in Allen AI code

**Noise** — two options:
- `metrics.checkpoint_noise()`: average relative_spread across mixes. Matches Allen AI's approach of taking late-checkpoint variance.
- `metrics.benchmark_noise()`: mean of per-model std across k-fold splits. Matches the multilingual analysis's `noise_from_fold_scores()`.
- Both available because the preliminary showed benchmark noise is more predictive (R=0.854) but requires per-sample data that imported datasets don't have.

**Decision accuracy** — `metrics.decision_accuracy()`:
- Vectorized pairwise comparison identical to `decision_acc_fast()` in Allen AI code
- Added `higher_is_better` parameter (missing from preliminary) to handle BPB/perplexity metrics where lower scores are better. Without this, DA would be inverted for loss-based metrics.

**SNR** — `metrics.snr()`: simple ratio with zero-division protection.

### What we chose NOT to implement (and why)

**Scaling law error** (`ladder_wrapper.py`): ~300 lines of polynomial fitting with a dependency on `OLMo-ladder`. Deferred because:
1. Our custom models aren't trained yet, so there's no scaling data
2. The OLMo-ladder dependency would add complexity
3. We already capture the most actionable metrics (SNR + DA)
4. Can be added later as `src/snr/scaling.py` when needed

**24 signal metric variants** (`snr_variants.py`): The preliminary empirically tested all 24 and found `relative_dispersion` wins. Implementing all variants would be ~300 lines of code for a comparison that's already been done. We implemented only the winner but kept `relative_spread` as an alternative for completeness.

**Large-scale SNR with separate signal/noise models** (`compute_snr_large_scale`): This uses a curated list of external models as the "signal pool" and a single model's training checkpoints as the "noise pool." Our analysis uses the more natural approach of varying data mixtures for signal — which the preliminary analysis also uses at small scale. The large-scale approach is specific to the Allen AI setup with pre-evaluated external models.

## 2. Analysis Module Design (`src/analysis`)

### `plots.py`: figures as objects, no side effects

Every function returns a `matplotlib.Figure` — the caller decides whether to save, display, or discard it. This follows matplotlib best practice and makes the functions testable.

**Key addition over preliminary**: 95% confidence interval bands on the log-linear fit in `snr_vs_decision_accuracy()`. The preliminary's `safe_log_fit_and_ci()` had this but our initial implementation didn't. Added via standard t-distribution CI computation on the regression residuals.

**Comparative plots** (`cross_dataset_scatter`, `side_by_side_scatter`, `side_by_side_ranking`): These were initially hardcoded in `analyze_imported.py`. Moved to `src/analysis/plots.py` during refactoring so `run_analysis.py` can also use them for stage comparisons.

### `summary.py`: tables + metadata

The `results_table()` function formats SNR results for human consumption. `recommend_benchmarks()` applies the same thresholds as the preliminary analysis: SNR ≥ 2.0 and DA ≥ 0.7.

**`dataset_metadata()` + `save_metadata()`**: Added to capture the useful diagnostic information that was previously only printed to stdout (checkpoint counts, task counts, model catalog). Now saved as JSON alongside SNR CSVs.

## 3. Data Loading (`src/snr/data.py`)

### Multiple loaders, one canonical format

All loaders produce the same DataFrame schema:
```
model_id, revision, checkpoint_index, task, metric, score, model_size, data_mix, seed, run_id, run_name
```

This means `compute_all_metrics()` works identically regardless of whether data came from W&B, HuggingFace, or local JSON.

| Loader | Source | When to use |
|---|---|---|
| `load_wandb_results()` | W&B summary | Our own eval runs (single checkpoint per run) |
| `load_wandb_history()` | W&B history | Imported multi-checkpoint runs |
| `load_local_results()` | Local JSON files | lm_eval output in results/ |
| `load_hf_dataset()` | HuggingFace parquet | Allen AI signal-and-noise dataset |
| `load_imported_results()` | Local JSON (import format) | QAT imported data |
| `load_csv_results()` | Saved CSV | Previously processed data |

**`load_hf_dataset()` and the fastparquet workaround**: The Allen AI dataset uses nested parquet schemas (struct columns for `model_config` and `metrics`). PyArrow 19 can't read these — it throws "Repetition level histogram size mismatch." Using `engine="fastparquet"` as the reader fixes it. This is a version incompatibility, not a data issue.

**`load_imported_results()` with metadata**: Returns both the DataFrame and a model catalog dict with checkpoint counts, task counts, sizes, and training variants per model. This metadata is saved to `results/snr/metadata_*.json`.

### Metadata parsing: `_parse_model_metadata()`

Extracts model_size, data_mix, and seed from model names using regex. Supports patterns like:
- `apertus-175M-en30-seed28` → size=175M, mix=en30, seed=28
- `HuggingFaceTB/SmolLM3-3B-checkpoints` → size=3B

For QAT data, `load_imported_results()` does its own parsing since QAT model names follow a different convention (`Apertus-1.7B-from8B-long`).

### Metric directionality

Added `LOWER_IS_BETTER_METRICS` set in `compute.py` and `higher_is_better` parameter to `compute_all_metrics()`. Auto-detects from metric name (byte_perplexity, loss → lower is better). For DA computation, lower-is-better scores are negated before pairwise comparison. The preliminary analysis had this in `compute_decision_accuracy_cross_size()` but it was missing from the Allen AI code.

### Weighted task aggregation

Added `_apply_group_weights()` for multilingual benchmarks with many subtasks (e.g., XNLI across 15 languages). Weights by sample count, matching the preliminary's `compute_weighted_group_scores()`. Not yet used in practice but ready for Phase 4 INCLUDE analysis.

## 4. Scripts: thin wrappers

| Script | Role | Key args |
|---|---|---|
| `compute_snr.py` | SNR from W&B or local data, by training stage | `--stage`, `--local`, `--noise`, `--small-sizes` |
| `run_analysis.py` | Plots + tables from SNR CSVs | `--input`, `--stage`, `--all-stages` |
| `analyze_imported.py` | End-to-end for imported data (AllenAI + QAT) | `--allenai-only`, `--qat-only` |
| `import_hf_to_wandb.py` | Fetch HF → save CSV + push W&B | `--match-sizes`, `--match-tasks`, `--all` |
| `import_wandb.py` | Copy runs from external W&B project | `--source`, `--tag` |

**Refactoring note**: `analyze_imported.py` was initially a monolith with its own QAT loading, parsing, and plotting logic. Refactored to:
- Move QAT loading → `data.load_imported_results(tag="QAT")`
- Move comparative plots → `plots.cross_dataset_scatter()`, `plots.side_by_side_*`
- The script itself became ~100 lines of glue calling `compute_all_metrics()` + plotting functions

## 5. Analysis Results

### AllenAI DataDecide (389K evaluations, 105 models at 150M–1B)

- **R = 0.617** (log-SNR vs decision accuracy) across 202 tasks
- Confirms Heineman et al. at larger scale than their paper
- Top reliable benchmarks: ARC Easy (SNR=74, DA=95%), MMLU (SNR=41, DA=91%), HellaSwag (SNR=42, DA=80%)
- The high R validates our implementation produces correct results

### QAT Apertus (59K evaluations, 22 models at 0.6B–8B)

- **R = 0.126** — expected to be lower because signal measures training recipe variation (base/SFT/long), not data mixture variation
- **Key finding**: multilingual benchmarks show high SNR for the first time — XCOPA-id (SNR=17, DA=100%), HellaSwag-vi (SNR=10, DA=100%)
- This resolves the R=0.045 finding from the preliminary analysis: the framework works for multilingual when models have multilingual competence

### Cross-dataset correlation

- **R = 0.408** across 64 common tasks
- ARC Easy, HellaSwag, MMLU, ARC Challenge rank highly in both datasets
- Suggests benchmark reliability is partially intrinsic to the benchmark, not just model-family dependent

### Can QAT data be used for SNR?

Yes, with caveats. QAT data provides:
- Checkpoint noise (4-15 checkpoints per model)
- Decision accuracy (0.6B → 1.7B → 3B → 8B size ladder)
- Training recipe signal (base vs SFT vs long) — different from data mix signal but still informative

The weaker R is not a problem — it correctly reflects that training recipes create less variance than data mixtures.

## 6. Noise Caching

Added `save_noise_results()` / `load_noise_results()` for fold score persistence. Critical for:
- Cluster workflows where noise computation is distributed across jobs
- Avoiding recomputation when running analysis multiple times

Format: JSON with `{task: {model_key: [fold1, fold2, ..., foldK]}}`.

## 7. Slides Update

Added 10 slides covering the new analysis results, including:
- DataDecide SNR validation (R=0.617)
- QAT multilingual SNR findings (first evidence it works)
- Cross-dataset comparison (R=0.408)
- Figures embedded from `documents/public/` (auto-copied by `analyze_imported.py`)

Updated timeline to show Phase 3 as validated, and open questions now reference concrete numbers from the analysis.

## 8. Decisions for future reference

1. **fastparquet dependency**: Required for reading Allen AI parquet files. Could be removed if pyarrow fixes the nested schema bug.
2. **Scaling law error**: Not implemented. Add as `src/snr/scaling.py` when custom model training data is available.
3. **Signal metric variants**: Only `relative_dispersion` implemented. If needed for the paper, add `src/snr/variants.py` with the full exploration.
4. **W&B push granularity**: One run per model (not per checkpoint). Each checkpoint is a step within the run. This keeps the run count manageable.
5. **Metadata saved as JSON**: Chose JSON over CSV for the model catalog because it's hierarchical (per-model checkpoint counts).
