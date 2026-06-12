# RQ1 — Which SNR definition best predicts decision accuracy?

## Research question

> Of 22 candidate SNR definitions, which best correlates with **decision
> accuracy** (DA) — the probability that a benchmark ranks two models the way
> a larger-model evaluation would — across 12 languages, and does the answer
> survive a change of training seed and the addition of external models?

<!-- BEGIN auto:highlight (snr_definition_postprocess.py --pool custom_swissai_hf) -->
## Highlighted result

- **Global-best SNR definition (`custom_swissai_hf`): `dist_std`** — mean Pearson r of log₁₀(SNR) vs decision accuracy **0.32** (DA-size), **0.43** (DA-ckpt), 0.38 overall. DA-ckpt is led by the mean-pairwise-distance / relative-spread cluster (`mpsd`/`rel_mpd`/`rel_mpsd` ≈ 0.51) — all dispersion-family, so recommend the *family*, not an exact variant.
- **Per-language anchor: `multiblimp`** — the highest-SNR above-random benchmark in **6 of 11** languages (`dist_std` SNR @ 1B).
- **Variant ranking generalizes across seeds under DA-ckpt** (holdout Spearman ρ **0.81**) **but not under DA-size** (ρ **-0.07**): the DA-size per-variant correlations are small and near-tied, so their ranking is noise-dominated and does not survive a seed swap — treat DA-size variant ranking as unreliable. The exact per-language argmax never transfers. **Never `tukey` / `projection`** (r ≤ 0).
<!-- END auto:highlight -->

## Experimental setup

Outputs live under `pretraining/<pool>/` for the four model-set tiers:
`seeds_1904` (1 seed) · `seeds_28_1797` (2 seeds) · `seeds_28_1797_1904`
(3 seeds) · `custom_swissai_hf` (3 seeds + a06 + distillation + swiss-ai/HF
**pretraining** references, instruct excluded). DA has two flavours: **DA-size**
(small→1B ranking) and **DA-ckpt** (early-→late-checkpoint ranking, early ckpt
picked at relative training fractions so external trajectories participate).
The 22 variants are grouped into families (dispersion / relative-spread /
discrepancy / robust / depth); the headline numbers reflect the comprehensive
`custom_swissai_hf` pool.

<!-- BEGIN auto:results (snr_definition_postprocess.py --pool custom_swissai_hf) -->
## Results

Headline numbers from the `custom_swissai_hf` pool. Regenerate with `python analysis/rq02_snr_definition/snr_definition_postprocess.py --pool custom_swissai_hf`.

**Global variant ranking** — mean Pearson r of log₁₀(SNR) vs DA across languages (top of a tight dispersion block; depth metrics collapse):

| variant | DA-size r | DA-ckpt r | overall |
|---|---|---|---|
| `dist_std` | 0.32 | 0.43 | 0.38 |
| `star_discrepancy_shifted` | 0.14 | 0.15 | 0.15 |
| `gini` | 0.13 | 0.12 | 0.13 |
| `discrepancy` | 0.13 | -0.04 | 0.05 |
| `star_discrepancy` | 0.12 | -0.00 | 0.06 |
| `rel_mpd` | 0.11 | 0.51 | 0.31 |
| `rel_std` | 0.11 | 0.50 | 0.31 |
| … |  |  |  |
| `tukey` | 0.05 | 0.22 | 0.14 |
| `projection` | -0.03 | -0.27 | -0.15 |

![SNR variants ranked by correlation with DA](pretraining/custom_swissai_hf/top_variants_overall.png)

**Statistical power by pool** — each pool's best DA-size variant:

| pool | best variant (DA-size) | DA-size r | DA-ckpt r |
|---|---|---|---|
| `seeds_1904` (1 seed) | `mad` | 0.50 | 0.28 |
| `seeds_28_1797` (2 seeds) | `rel_mpd` | 0.31 | 0.37 |
| `seeds_28_1797_1904` (3 seeds) | `rel_std` | 0.43 | 0.48 |
| `custom_swissai_hf` (3 seeds + externals) | `dist_std` | 0.32 | 0.43 |

**Most reliable benchmark per language** — `dist_std` SNR @ 1B over above-random tasks (DA-size is NaN at the 1B target, so DA-ckpt@1B is shown):

| lang | top benchmark | SNR | DA-ckpt@1B |
|---|---|---|---|
| ar | `multiblimp_arb` | 2.65 | 0.87 |
| en | `xwinograd_en` | 2.40 | 0.83 |
| es | `multiblimp_spa` | 3.37 | 0.85 |
| eu | `multiblimp_eus` | 1.28 | 0.64 |
| hi | `multiblimp_hin` | 4.95 | 0.85 |
| ja | `xwinograd_jp` | 2.28 | 0.76 |
| ru | `multiblimp_rus` | 7.08 | 0.86 |
| th | `xnli_th` | 1.28 | 0.75 |
| tr | `multiblimp_tur` | 2.75 | 0.79 |
| vi | `xcopa_vi` | 1.61 | 0.76 |
| zh | `xcopa_zh` | 1.57 | 0.61 |

![Top-5 benchmarks per language by SNR](pretraining/custom_swissai_hf/top_benchmarks_per_language.png)

**Seed generalization** — holdout `seeds_28_1797` → `seeds_1904`. DA-ckpt ranking transfers; **DA-size does not** — its per-variant correlations are small and clustered (top variants within ~0.1), so the global ranking is noise-dominated and its Spearman ρ is unstable run-to-run (don't read it as a real effect):

| metric | DA-size | DA-ckpt |
|---|---|---|
| Spearman ρ on global variant ranking | -0.07 | 0.81 |
| Pearson r between splits (all cells) | 0.48 | 0.60 |
| Exact-variant agreement (per lang) | 7% | 7% |
| Family-level agreement (per lang) | 21% | 29% |
| Retention of train-best r on test | 61% | 79% |
<!-- END auto:results -->

## TODO

- [ ] Bootstrap CIs on per-language Pearson r and cross-pool Spearman ρ.
- [ ] Recommend a *family* (dispersion / relative-spread), not an exact variant
      — only the family transfers across seeds.
- [ ] Use a larger DA-size target (e.g. Apertus-8B) instead of the
      not-fully-converged 1B custom model.

## Files

- `pretraining/<pool>/snr_variants_per_task.csv` — per-task SNR (every
  variant × size-bucket) + DA columns. Single source of truth.
- `…/snr_variant_ranking.csv` — full per-(variant, DA-def, scope) Pearson r.
- `…/top_variants_overall.csv`, `best_variant_per_language.csv`,
  `variant_clusters.csv`, `top_benchmarks_per_language.csv` — RQ1 tables.
- `…/variant_correlation_matrix.png`, `best_variant_family_per_language.png`,
  `da_size_vs_da_ckpt.png` — supporting figures.
- `…/{da_size,da_ckpt}/…` — top-3 scatter grids + per-language heatmaps.
- `seeds_28_1797__vs__seeds_1904/` — the holdout generalization report.
