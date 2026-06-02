# RQ1 — Which SNR definition best predicts decision accuracy?

> Of 22 candidate SNR definitions, which best correlates with **decision
> accuracy** (DA) — the probability that a benchmark ranks two models the way
> a larger-model evaluation would — across 12 languages, and does the answer
> survive a change of training seed and the addition of external models?

Outputs live under `pretraining/<pool>/` for the four model-set tiers:
`seeds_1904` (1 seed) · `seeds_28_1797` (2 seeds) · `seeds_28_1797_1904`
(3 seeds) · `custom_swissai_hf` (3 seeds + a06 + distillation + swiss-ai/HF
**pretraining** references, instruct excluded). DA has two flavours: **DA-size**
(small→1B ranking) and **DA-ckpt** (early-→late-checkpoint ranking, early ckpt
picked at relative training fractions so external trajectories participate).

## TL;DR (paper takeaways)

1. **`rel_mpd` (relative mean pairwise distance) is the global-best SNR
   definition.** On the full pool it reaches mean Pearson r **+0.40 (DA-size)**
   and **+0.52 (DA-ckpt)** across 12 languages — top of a tight
   relative-spread / dispersion block. Use it (or any dispersion-family member)
   as the default; **never `tukey` / `projection`** (anti-correlated, r ≤ 0).
2. **Checkpoint-DA is the more SNR-aligned target than size-DA** — the winners'
   DA-ckpt r (0.46–0.52) exceeds their DA-size r (~0.40). SNR tracks "does this
   benchmark separate a run's own early vs late checkpoints" better than "does
   it predict the 1B model".
3. **More model variation sharpens the relative variants.** The global winner
   moves from absolute `mpd` (1 seed) to **`rel_mpd`** (3 seeds + externals),
   and the top correlation rises from +0.31 to +0.52.
4. **The variant *ranking* generalizes across seeds** (Spearman ρ = **+0.80**
   DA-size, **+0.93** DA-ckpt, train 2 seeds → held-out seed) — but the
   *per-language argmax does not* (0/14 languages keep the exact pick). Report
   a single global default; treat per-language tuning as overfitting unless
   re-validated on a held-out seed.

## Global variant ranking (pool `custom_swissai_hf`)

Mean Pearson r of log10(SNR) vs DA across 12 languages — top and bottom of
[`top_variants_overall.csv`](pretraining/custom_swissai_hf/top_variants_overall.csv):

| variant | DA-size r | DA-ckpt r | overall |
|---|---:|---:|---:|
| **`rel_mpd`** | **0.400** | **0.519** | **0.460** |
| `rel_std` | 0.396 | 0.455 | 0.426 |
| `iqr` | 0.389 | 0.390 | 0.390 |
| `quartile_deviation` | 0.386 | 0.410 | 0.398 |
| `aad` | 0.382 | 0.484 | 0.433 |
| `mpd` | 0.368 | 0.512 | 0.440 |
| `rms_deviation` | 0.367 | 0.461 | 0.414 |
| … | | | |
| `projection` | 0.012 | −0.391 | −0.189 |
| **`tukey`** | **−0.006** | **−0.034** | **−0.020** |

The top block (`rel_mpd`/`rel_std` + the dispersion cluster `quartile_deviation`/
`aad`/`mpd`/`rms_deviation`/`dist_std`) is algebraically near-redundant
(inter-variant r ≥ 0.999), so any member works as the default. Depth metrics
(`tukey`, `projection`) need more dimensions than the model pool provides and
collapse.

![SNR variants ranked by correlation with DA](pretraining/custom_swissai_hf/top_variants_overall.png)

![Inter-variant redundancy of log10(SNR)](pretraining/custom_swissai_hf/variant_correlation_matrix.png)

## Statistical power: more seeds + external models

Top-variant correlation by tier (each cell is that tier's best variant's mean r):

| pool | best variant | DA-size r | DA-ckpt r |
|---|---|---:|---:|
| `seeds_1904` (1 seed) | `mpd` | 0.312 | 0.296 |
| `seeds_28_1797` (2 seeds) | `rel_mpd` | 0.326 | 0.276 |
| `seeds_28_1797_1904` (3 seeds) | `rel_mpd` | 0.392 | 0.379 |
| **`custom_swissai_hf`** (3 seeds + externals) | **`rel_mpd`** | **0.400** | **0.519** |

Adding the external pretraining models is what unlocks the large **DA-ckpt**
gain (0.379 → **0.519**): their multi-checkpoint trajectories give the
relative-fraction ckpt-DA far more, more-diverse points to fit.

## Seed generalization (holdout: `seeds_28_1797` → `seeds_1904`)

From [`seeds_28_1797__vs__seeds_1904/summary.md`](pretraining/seeds_28_1797__vs__seeds_1904/summary.md):

| metric | DA-size | DA-ckpt |
|---|---:|---:|
| **Spearman ρ on global variant ranking** | **+0.797** | **+0.925** |
| Pearson r between splits (all variant×lang cells) | +0.569 | +0.725 |
| Exact-variant agreement (per language) | 0/14 | 1/14 |
| Family-level agreement (per language) | 2/14 | 5/14 |
| Retention of train-best r on the test seed | 62% | 78% |

**Read:** the *global* recommendation is seed-robust (ρ up to 0.93); the
*per-language* best variant is not — so the paper should claim a single global
default, not language-specific SNR definitions.

![Per-(language,variant) r: train seed vs held-out seed](pretraining/seeds_28_1797__vs__seeds_1904/variant_r_train_vs_test.png)

## Best benchmark in each language (highest SNR, with mean DA-ckpt@1B)

From [`top_benchmarks_per_language.csv`](pretraining/custom_swissai_hf/top_benchmarks_per_language.csv):

| lang | top benchmark | SNR | DA-ckpt |
|---|---|---:|---:|
| ar | `hellaswag_ar` | 6.88 | **0.89** |
| en | `mmlu` | 6.04 | 0.65 |
| es | `hellaswag_es` | 4.88 | **0.86** |
| eu | `xnli_eu` | **8.51** | 0.70 |
| hi | `hellaswag_hi` | 7.83 | **0.88** |
| ja | `xwinograd_jp` | 5.24 | 0.76 |
| ru | `multiblimp_rus` | **8.84** | **0.86** |
| sw | `xstorycloze_sw` | 5.18 | 0.73 |
| th | `xnli_th` | 4.71 | 0.75 |
| tr | `multiblimp_tur` | 4.58 | 0.79 |
| vi | `hellaswag_vi` | 5.08 | **0.84** |
| zh | `belebele_zho_Hans` | 3.72 | 0.53 |

**`hellaswag_<lang>`, `multiblimp_<lang>` and `xstorycloze_<lang>` are the
recurring high-signal, high-DA benchmarks.** `xnli_<lang>` reaches the highest
raw SNR (eu 8.5) but its DA-size is often 0 (mis-ranks vs the 1B target) — keep
it only where DA is non-trivial.

![Top-5 benchmarks per language by SNR](pretraining/custom_swissai_hf/top_benchmarks_per_language.png)

## Which DA target does each variant serve?

![Variant agreement: DA-size vs DA-ckpt](pretraining/custom_swissai_hf/da_size_vs_da_ckpt.png)

Most of the dispersion / relative-spread family sits above the diagonal —
agreeing more with DA-ckpt than DA-size, consistent with takeaway #2.

## Files

- `pretraining/<pool>/snr_variants_per_task.csv` — per-task SNR (every
  variant × size-bucket) + DA columns. Single source of truth.
- `…/snr_variant_ranking.csv` — full per-(variant, DA-def, scope) Pearson r.
- `…/top_variants_overall.csv`, `best_variant_per_language.csv`,
  `variant_clusters.csv`, `top_benchmarks_per_language.csv` — RQ1 tables.
- `…/{da_size,da_ckpt}/…` — top-3 scatter grids + per-language heatmaps.
- `seeds_28_1797__vs__seeds_1904/` — the holdout generalization report.
