# INSTRUCTIONS — `snr_definition`

Owner: one Claude session. Don't write outside this directory.

## Research questions

Per language, which benchmarks have the strongest **SNR ↔ DA**
correlation?

1. Across **all SNR definitions** (the 22 aggregators in
   `snr.snr_variants.AGGREGATION_FUNCTIONS`): which definition's SNR
   correlates best with decision accuracy?
2. Holding the best-performing SNR definition fixed: which
   **benchmarks** have the highest SNR (per language)? These are the
   "reliable for ranking data mixes" benchmarks.

## Inputs

- `analysis/data/multilingual_snr/data/pretraining_custom-*.parquet`
  (12 Apertus pretrained models). Loaded via
  `snr.download.apertus.load_apertus_eval_results`.
- `reference_hf-*.parquet` — 4 external HF models with multi-step
  trajectories (SmolLM3-3B-checkpoints, Olmo-3-1025-7B,
  Apertus-8B-2509, plus SmolLM3-3B-Base which is a single-step
  snapshot). **Use these as extra DA-ckpt data points** (see Step 6
  below). They cannot enter the per-mix SNR computation directly —
  only Olmo-3-7B has more than one mix (`main` + `stage1`) — so the
  signal/noise/SNR columns of the variants table are still
  Apertus-only.

## Existing code (REUSE)

- [multilingual/run_apertus_snr_variants.py](../../multilingual/run_apertus_snr_variants.py)
  — emits `snr_variants_per_task.csv` (one row per task with
  `signal_<v>_<size>`, `noise_<v>_<size>`, `snr_<v>_<size>` columns
  for every variant × size, plus DA columns).
- [multilingual/analyze_snr_variants.py](../../multilingual/analyze_snr_variants.py)
  — reads the CSV and renders every grid currently committed under
  `results/snr_definition/`.

Both scripts already work end-to-end on the local parquet (the loader
was patched). Re-running gives reproducible artifacts.

## Plan

### Step 1 — refresh the variants CSV

```bash
python multilingual/run_apertus_snr_variants.py
```

Confirms the loader works against the local parquet. Should write
`results/snr_definition/snr_variants_per_task.csv` (~280 columns × ~820
tasks).

### Step 2 — render existing plot grid

```bash
python multilingual/analyze_snr_variants.py
```

Produces:
- `da_size/snr_vs_decision_accuracy.png` (rows = variants, cols = small
  sizes; ranked by mean Pearson r).
- `da_size/snr_vs_decision_accuracy_<lang>.png` for each language
  with ≥5 valid DA points.
- `da_size/heatmap_pearson_r.png` — variant × language Pearson r.
- `da_ckpt/{da_ckpt_mix,da_ckpt_175M,...}/snr_vs_decision_accuracy*.png`
  — same idea but DA defined within a single size (early-ckpt vs
  max-ckpt).
- `variant_correlation_matrix.png` — which variants are algebraically
  redundant.
- `da_size_vs_da_ckpt.png` — per-variant: agreement on DA-size axis vs
  DA-ckpt axis.

### Step 3 — answer Q1 (best SNR definition)

From `da_size/heatmap_pearson_r.png` and the per-language ranking
inside `analyze_snr_variants.py:rank_variants`, identify:
- **Globally best variant** = top row of the heatmap (highest mean
  Pearson r across languages).
- **Per-language best variant** = argmax per column.

Save a new file `results/snr_definition/best_variant_per_language.csv`
with columns `language, best_variant, mean_pearson_r,
runner_up_variant`. Implement as a small `<10`-line script that reads
`snr_variants_per_task.csv` and re-uses `_per_language_pearson_table`
from `analyze_snr_variants.py`.

### Step 4 — answer Q2 (best benchmarks per language, fixed variant)

Pick the global winner from Step 3 (call it `<V>`). For each language:

1. Subset `snr_variants_per_task.csv` to tasks in that language.
2. Sort by `snr_<V>_1B` (or whichever single size you commit to —
   `1B` is the largest available; document the choice in the README).
3. Output a CSV `top_benchmarks_per_language.csv` with columns
   `language, rank, task, snr_<V>_<size>, decision_acc_size_600M`.

Cap the per-language list at the top 5 tasks. Render a faceted bar
chart `top_benchmarks_per_language.png` (one panel per language, bars
= top 5 tasks colored by SNR magnitude).

### Step 6 — extend DA-ckpt with reference_hf step-series

The DA-ckpt definition (rank agreement at early-ckpt vs late-ckpt for
a fixed model identity) is well-defined for a single-mix model: pick
`early` and `late` steps, ask whether the per-task ordering at `early`
matches the ordering at `late`. This adds 4 more (model, task) data
points to the per-variant regression without touching the SNR axis.

Concrete steps (one new script, ~60 lines):

1. Load `load_reference_hf_eval_results()`.
2. For each model in `{Apertus-8B-2509, Olmo-3-1025-7B,
   SmolLM3-3B-checkpoints}` (skip `SmolLM3-3B-Base` — only 1 step):
   pick `early_step` = ~⅓ of max step, `late_step` = max step. For
   each shared task with Apertus, compute Pearson r between
   `score(task, early_step)` and `score(task, late_step)` *across
   tasks* — this gives a per-model "DA-ckpt-like" reliability score.
3. Append the resulting `(task, model, da_ckpt_ref)` rows to a wide
   table `snr_variants_per_task_with_refhf.csv` with the SNR columns
   from Apertus (joined on `task`).
4. Recompute the heatmap using the pooled DA points and write
   `da_ckpt/heatmap_pearson_r_with_refhf.png`. The variant order may
   shift; document any change in the README.

Skip Step 6 if time-constrained — the headline finding can stand on
Apertus alone. The reference HF extension is for the robustness
section of the README.

### Step 7 — README

Write `results/snr_definition/README.md`:

- **Research questions** — verbatim from above.
- **Setup** — 3–4 sentences: "12 Apertus pretrains, 3 mixes × 4 sizes,
  X SNR variants from snr_variants.py, DA-size = mix-rank agreement
  small-vs-1B, DA-ckpt = early-vs-late within size."
- **Main results** — embed:
  - `da_size/heatmap_pearson_r.png` (top finding).
  - `da_size/snr_vs_decision_accuracy.png` (for the headline grid).
  - `top_benchmarks_per_language.png` (Q2).
  Keep prose to one paragraph per plot.
- **Directory contents** — bulleted list naming every file/subdir.

## Notes / gotchas

- The variants CSV is the **single source of truth**. If you change
  the SNR definition, regenerate it.
- 1B-fwEdu90 has only 8 ckpts (steps stop at 38000); the per-mix
  jagged-array tolerance in `compute_snr_small_scale` already handles
  this. Don't refactor it.
- DA needs ≥2 mixes alive at the relevant ckpt — undercovered cells
  return NaN. The Pearson-r helper masks NaNs.
- Don't add scaling-law-error columns. We don't have a ladder of
  Apertus models.

## Definition of done

- `snr_variants_per_task.csv` regenerated from the local parquet.
- All grids/heatmaps under `da_size/` and `da_ckpt/` regenerated.
- `best_variant_per_language.csv` and
  `top_benchmarks_per_language.{csv,png}` produced.
- `README.md` references and embeds the headline plots and lists every
  file in the directory.
