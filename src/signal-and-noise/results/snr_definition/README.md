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

1. **The dispersion family is the global-best SNR definition — but two
   different members lead the two DA targets.** `dist_std` dominates **DA-size**
   (mean Pearson r **+0.32**, far ahead of the runner-up at +0.14), while the
   mean-pairwise-distance / relative-spread cluster (`rel_mpd` / `mpd` / `mpsd`)
   leads **DA-ckpt** (≈ **+0.51**). All are dispersion-family; recommend the
   *family*, not an exact variant. **Never `tukey` / `projection`** (r ≤ 0).
2. **Checkpoint-DA is the more SNR-aligned target than size-DA** — the DA-ckpt
   leaders (≈0.50) exceed the DA-size leader (0.32). SNR tracks "does this
   benchmark separate a run's own early vs late checkpoints" better than "does
   it predict the 1B model".
3. **More model variation sharpens the signal.** The DA-size winner moves from
   absolute `mpd` (1 seed) to `rel_mpd` (3 pure seeds) to **`dist_std`** (3
   seeds + externals), and the top DA-size r climbs **+0.31 → +0.39**.
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
| **`dist_std`** | **0.318** | **0.434** | **0.376** |
| `star_discrepancy_shifted` | 0.141 | 0.154 | 0.148 |
| `gini` | 0.134 | 0.119 | 0.126 |
| `rel_mpd` | 0.112 | **0.506** | 0.309 |
| `rel_std` | 0.111 | 0.500 | 0.305 |
| `mpd` | 0.105 | 0.504 | 0.304 |
| `mpsd` | 0.059 | **0.507** | 0.283 |
| … | | | |
| `tukey` | 0.055 | 0.220 | 0.137 |
| **`projection`** | **−0.026** | **−0.266** | **−0.146** |

Two regimes: **`dist_std` alone dominates DA-size** (0.318 vs 0.14 for the next
variant — it is *not* redundant with the other dispersion members here), while
the **mean-pairwise-distance / relative-spread cluster** (`rel_mpd` / `mpd` /
`mpsd` / `aad` / `rms_deviation` / `mad`) ties at ≈ 0.50 for **DA-ckpt**. All
are dispersion-family, so the family-level recommendation is robust even though
the exact argmax differs by target. Depth metrics (`tukey`, `projection`) need
more dimensions than the model pool provides and collapse (r ≤ 0).

![SNR variants ranked by correlation with DA](pretraining/custom_swissai_hf/top_variants_overall.png)

![Inter-variant redundancy of log10(SNR)](pretraining/custom_swissai_hf/variant_correlation_matrix.png)

## Statistical power: more seeds + external models

Top-variant correlation by tier (each cell is that tier's best variant's mean r):

| pool | best variant (DA-size) | DA-size r | DA-ckpt r |
|---|---|---:|---:|
| `seeds_1904` (1 seed) | `mpd` | 0.312 | 0.296 |
| `seeds_28_1797` (2 seeds) | `rel_mpd` | 0.326 | 0.276 |
| `seeds_28_1797_1904` (3 seeds) | `rel_mpd` | 0.392 | 0.379 |
| **`custom_swissai_hf`** (3 seeds + externals) | **`dist_std`** | **0.318** | 0.434 |

The externals don't lift the DA-size leader (the winner flips to `dist_std`,
0.32, on par with the pure 3-seed 0.39), but their multi-checkpoint
trajectories give the relative-fraction ckpt-DA far more points to fit: the
relative cluster's **DA-ckpt** r rises from 0.379 (3 pure seeds) to **0.51**
(`rel_mpd` / `mpsd`) once externals are folded in.

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

SNR is the global-best variant (`dist_std`) @ 1B over the above-random tasks:

| lang | top benchmark | SNR | DA-ckpt |
|---|---|---:|---:|
| ar | `multiblimp_arb` | 2.65 | **0.87** |
| en | `xwinograd_en` | 2.40 | **0.83** |
| es | `multiblimp_spa` | **3.37** | **0.85** |
| eu | `multiblimp_eus` | 1.28 | 0.64 |
| hi | `multiblimp_hin` | **4.95** | **0.85** |
| ja | `xwinograd_jp` | 2.28 | 0.76 |
| ru | `multiblimp_rus` | **7.08** | **0.86** |
| th | `xnli_th` | 1.28 | 0.75 |
| tr | `multiblimp_tur` | 2.75 | 0.79 |
| vi | `xcopa_vi` | 1.61 | 0.76 |
| zh | `xcopa_zh` | 1.57 | 0.61 |

**`multiblimp_<lang>` is the most reliable benchmark in 7 of 11 languages**,
with `xwinograd` / `xcopa` recurring; `xnli_th` leads only where those are
absent. `sw` has no above-random benchmark left and drops out. (DA-size is NaN
by definition at the 1B target, so DA-ckpt@1B is shown.)

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
