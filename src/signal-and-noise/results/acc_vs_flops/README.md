# Accuracy-vs-FLOPs curves (scaling view)

> How does benchmark accuracy move with compute across the three data mixtures
> and across model scale, and which benchmarks separate the mixtures most?

Outputs under `pretraining/<pool>/{per_benchmark,per_language}/`. Each panel is
accuracy vs FLOPs (log-x); the custom models are drawn as per-mixture curves
(seed 1904), and for `custom_swissai_hf` the external pretraining models
(a06, distillation, swiss-ai/HF base) are **overlaid as final-checkpoint
markers out to 70B** to extend the compute axis past the custom 1B ceiling.

## How the plots are selected (fast by construction)

- Tasks are **parent-aggregated** (mmlu / agieval / arabic_leaderboard subjects
  collapse into the parent; languages stay distinct) → 950 → 118 tasks.
- Every task's **Signal** = (max−min)/mean of the per-mixture final scores at
  1B is written to [`acc_vs_flops_signal.csv`](pretraining/custom_swissai_hf/acc_vs_flops_signal.csv);
  only the **top-3 families by Signal** get curve grids (307 → 3 per-benchmark
  PNGs).

## TL;DR (paper takeaways)

1. **The benchmarks that separate data mixtures most are `belebele`,
   `agieval_sat`, and `arabic_leaderboard`** — top-3 by mixture-Signal at 1B.
2. **Mixture-Signal ≠ SNR.** These top-Signal families are exactly RQ2's
   *lowest-SNR* families (`belebele` is last by SNR). They swing a lot with the
   data mixture (high signal) **but are also high-noise**, so signal-to-noise
   stays low. The lesson for the paper: *raw mixture sensitivity is not
   reliability* — a benchmark must be divided by its noise (SNR), which is why
   RQ1/RQ2 rank `multiblimp`/`hellaswag` (low absolute swing, very low noise)
   above `belebele`.
3. **Scaling overlay**: on `custom_swissai_hf` the external models extend the
   curves to 8B–70B, but cross-size *decision accuracy* above 1B stays
   family-coverage-limited (see RQ1's scaling-DA note) — the overlay is for
   visual scaling, not a DA claim.

## Top benchmarks by mixture-Signal (pool `custom_swissai_hf`)

| task | family | lang | Signal |
|---|---|---|---:|
| **`agieval_sat_en`** | agieval_sat | en | **0.268** |
| `belebele_hin_Deva` | belebele | hi | 0.245 |
| `belebele_zho_Hans` | belebele | zh | 0.236 |
| `global_piqa_completions_arb_arab` | global_piqa | ar | 0.229 |
| `belebele_eng_Latn` | belebele | en | 0.222 |

![belebele accuracy vs FLOPs (per language)](pretraining/custom_swissai_hf/per_benchmark/belebele.png)

![English benchmarks accuracy vs FLOPs](pretraining/custom_swissai_hf/per_language/en.png)

## Above-random gate (foundational — RQs depend on it)

Before a benchmark can carry signal the models must clear **chance**
(`1 / n_options`). [`above_random.py`](../../multilingual/above_random.py)
computes this at the raw-metric level, depending **only** on raw eval scores
and the intrinsic per-family answer-option counts (`N_OPTIONS` in that file) —
it reads no RQ output, so every RQ depends on the gate, never the reverse.

```bash
python multilingual/above_random.py --pool custom_swissai_hf
```

For each (benchmark, custom size group) it averages the raw `primary_score`
over all models at that size (final ckpt per model). A cell is **above random**
iff `mean > 1/n_options + 0.05`. `run_apertus_snr_variants.py` imports
`scores_and_mask` and NaN-s every random `(benchmark, size)` SNR cell, so the
gate propagates to all downstream analyses — **random benchmarks are never used.**

Of **118 benchmarks, 44 clear chance at ≥ 1 size** (74 are random everywhere);
it's almost entirely an answer-count effect:

| options | chance | above at ≥1 size | above at 1B |
|---:|---:|---:|---:|
| 2 | 0.50 | 28 / 42 | 28 / 42 |
| 3 | 0.33 | 7 / 11 | 7 / 11 |
| 4 | 0.25 | **9 / 63** | 7 / 63 |
| 5 | 0.20 | 0 / 2 | 0 / 2 |

4-option knowledge MCQA (`belebele`, multilingual `arc`, `global_mmlu`) sit at
chance; `hellaswag` (4-option but contentful) is the exception. Some tasks clear
chance only with scale (`arc_challenge`, `xnli_th` at 600M+; `paws_en` at 1B).

## Files

- `pretraining/<pool>/acc_vs_flops_signal.csv` — per-task mixture-Signal (full
  ranking; all 118 parent tasks).
- `…/above_random_scores.csv` — row per benchmark, column per size group, value =
  mean score across all models at that size (+ `n_options`, `random_baseline`,
  `options_exact`). From `multilingual/above_random.py`.
- `…/above_random_mask.csv` — same shape; value = 1 (above random) / 0 (random) /
  blank (no models). The gate consumed by the SNR pipeline.
- `…/per_benchmark/<family>.png` — top-3 families, subplots per language,
  external scaling markers overlaid.
- `…/per_language/<lang>.png` — per language, subplots = top-3 families.
