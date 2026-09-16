# RQ2 — Which SNR definition best predicts decision accuracy?

## Research question

> Of 22 candidate SNR definitions, which best correlates with **decision
> accuracy** (DA) — the probability that a benchmark ranks two models the way
> a larger-model evaluation would — across the ladder's languages, and does the
> answer survive a change of training seed?

<!-- BEGIN auto:highlight (snr_definition_postprocess.py --pool predictivity) -->
## Highlighted result

- **Global-best SNR definition (`predictivity`): `discrepancy`** — mean Pearson r of log₁₀(SNR) vs decision accuracy **0.03** (DA-size), **-0.05** (DA-ckpt), -0.01 overall. DA-ckpt is led by `mpsd`/`rel_mpsd`/`aad` (≈ 0.27; families: dispersion, rel_spread) — recommend the *family*, not an exact variant.
- **Per-language anchor: `multiblimp`** — the highest-SNR above-random benchmark in **55 of 65** languages (`discrepancy` SNR @ 1B). Weakest variants overall: `tukey`, `projection`.
- **Seed holdout (predictivity_seeds_train → predictivity_seeds_test)**: Spearman ρ of the global variant ranking **-0.21** (DA-ckpt), **0.01** (DA-size); family-level per-language agreement 15% / 3%. A ranking that does not survive the seed swap is noise-dominated — only the *family* recommendation transfers.
<!-- END auto:highlight -->

## Experimental setup

Outputs live under `pretraining/<pool>/` for the ladder pools: `predictivity`
(the plan grid, seed 1904 — the headline), `predictivity_seeds` (every seed;
replicates enter the signal pool as separate models), and the seed holdout
`predictivity_seeds_train` (seeds 64/313 at the ×3 cells) → `_test` (seed 1904
on the same cells). The signal population at a size is every design variant
trained there (language setting × depth × scheme); the noise is the
late-checkpoint std over the last 5 checkpoints on the shared grid (rq06
carries the seed-replicate noise). DA has two flavours: **DA-size**
(small→1B ranking, plus every other bucket pair up to 1.7B) and **DA-ckpt**
(20/40/60/80 % → final within a size). The 22 variants are grouped into
families (dispersion / relative-spread / discrepancy / robust / depth) and
correlated with DA per language as Pearson r of log₁₀(SNR) vs DA; the
per-language BPB tasks (`bpb_<subset>`) take part like any benchmark.
The seed holdout is English-heavy by construction: among the ×3 cells
(L ∈ {1, 2, 50, 100}) a non-English harness task is only evaluated at L50 and
L100, so its per-language variant ranking rests on two language settings per
seed, while English and the BPB tasks cover all four.

<!-- BEGIN auto:results (snr_definition_postprocess.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq02_snr_definition/snr_definition_postprocess.py --pool predictivity`.

**Global variant ranking** — mean Pearson r of log₁₀(SNR) vs DA across languages:

| variant | DA-size r | DA-ckpt r | overall |
|---|---|---|---|
| `discrepancy` | 0.03 | -0.05 | -0.01 |
| `rel_mpsd` | -0.00 | 0.27 | 0.13 |
| `star_discrepancy_shifted` | -0.00 | 0.11 | 0.05 |
| `mpsd` | -0.00 | 0.27 | 0.13 |
| `dist_std` | -0.01 | 0.24 | 0.12 |
| `rms_deviation` | -0.01 | 0.24 | 0.11 |
| `aad` | -0.02 | 0.25 | 0.12 |
| … |  |  |  |
| `projection` | -0.17 | -0.03 | -0.10 |
| `tukey` | -0.21 | -0.10 | -0.16 |

![SNR variants ranked by correlation with DA](pretraining/predictivity/top_variants_overall.png)

**Statistical power by pool** — each pool's best DA-size variant:

| pool | best variant (DA-size) | DA-size r | DA-ckpt r |
|---|---|---|---|
| `predictivity` (grid, seed 1904) | `discrepancy` | 0.03 | -0.05 |
| `predictivity_seeds` (all seeds) | `star_discrepancy_shifted` | 0.04 | 0.11 |

**Most reliable benchmark per language** — `discrepancy` SNR @ 1B over above-random tasks (DA-size is undefined at the reference size itself, so DA-ckpt@1B is shown):

| lang | top benchmark | SNR | DA-ckpt@1B |
|---|---|---|---|
| am | `multiblimp_amh` | 6.55 | 0.46 |
| ar | `multiblimp_arb` | 16.45 | 0.84 |
| be | `multiblimp_bel` | 5.85 | 0.76 |
| bg | `multiblimp_bul` | 9.08 | 0.92 |
| bn | `bpb_ben_Beng` | 2.77 | 0.96 |
| ca | `multiblimp_cat` | 12.03 | 0.87 |
| cs | `multiblimp_ces` | 7.50 | 0.88 |
| cy | `multiblimp_cym` | 3.49 | 0.71 |
| da | `multiblimp_dan` | 8.08 | 0.72 |
| de | `multiblimp_deu` | 51.07 | 0.82 |
| el | `multiblimp_ell` | 24.70 | 0.82 |
| en | `multiblimp_eng` | 113.20 | 0.46 |
| es | `multiblimp_spa` | 36.66 | 0.85 |
| et | `multiblimp_est` | 9.72 | 0.74 |
| eu | `multiblimp_eus` | 28.68 | 0.70 |
| fa | `multiblimp_fas` | 8.09 | 0.83 |
| fi | `multiblimp_fin` | 7.03 | 0.78 |
| fo | `multiblimp_fao` | 1.96 | 0.60 |
| fr | `multiblimp_fra` | 39.99 | 0.72 |
| ga | `multiblimp_gle` | 2.00 | 0.69 |
| gd | `multiblimp_gla` | 14.07 | 0.56 |
| gl | `multiblimp_glg` | 9.94 | 0.79 |
| grc | `multiblimp_grc` | 7.20 | 0.65 |
| gu | `multiblimp_guj` | 6.91 | 0.84 |
| hbo | `multiblimp_hbo` | 7.90 | 0.74 |
| he | `multiblimp_heb` | 7.62 | 0.84 |
| hi | `multiblimp_hin` | 17.87 | 0.78 |
| hu | `multiblimp_hun` | 14.08 | 0.85 |
| hy | `multiblimp_hye` | 4.81 | 0.62 |
| hyw | `multiblimp_hyw` | 2.99 | 0.59 |
| id | `hellaswag_id` | 9.43 | 0.86 |
| is | `multiblimp_isl` | 10.00 | 0.69 |
| it | `multiblimp_ita` | 14.51 | 0.81 |
| ja | `xwinograd_jp` | 4.75 | 0.72 |
| ka | `multiblimp_kat` | 7.66 | 0.78 |
| kk | `multiblimp_kaz` | 5.30 | 0.76 |
| kmr | `multiblimp_kmr` | 6.22 | 0.65 |
| kn | `bpb_kan_Knda` | 12.31 | 0.71 |
| ky | `multiblimp_kir` | 28.90 | 0.41 |
| la | `multiblimp_lat` | 5.25 | 0.80 |
| lt | `multiblimp_lit` | 10.99 | 0.61 |
| mk | `multiblimp_mkd` | 4.54 | 0.77 |
| ml | `bpb_mal_Mlym` | 3.76 | 0.93 |
| mr | `bpb_mar_Deva` | 20.52 | 0.86 |
| nds | `multiblimp_nds` | 13.36 | 0.62 |
| ne | `bpb_npi_Deva` | 9.03 | 0.92 |
| nl | `multiblimp_nld` | 12.55 | 0.88 |
| pl | `multiblimp_pol` | 11.71 | 0.93 |
| pt | `multiblimp_por` | 33.51 | 0.87 |
| ro | `multiblimp_ron` | 18.05 | 0.88 |
| ru | `multiblimp_rus` | 61.02 | 0.86 |
| sa | `multiblimp_san` | 10.19 | 0.64 |
| sah | `multiblimp_sah` | 7.85 | 0.62 |
| se | `multiblimp_sme` | 12.44 | 0.43 |
| sk | `multiblimp_slk` | 3.96 | 0.90 |
| sl | `multiblimp_slv` | 8.23 | 0.85 |
| sv | `multiblimp_swe` | 73.09 | 0.87 |
| ta | `multiblimp_tam` | 11.41 | 0.86 |
| th | `xcopa_th` | 4.68 | 0.61 |
| tr | `multiblimp_tur` | 12.12 | 0.75 |
| ug | `multiblimp_uig` | 13.79 | 0.62 |
| uk | `multiblimp_ukr` | 12.64 | 0.90 |
| ur | `multiblimp_urd` | 5.81 | 0.78 |
| vi | `hellaswag_vi` | 8.83 | 0.79 |
| zh | `xstorycloze_zh` | 8.85 | 0.84 |

![Top-5 benchmarks per language by SNR](pretraining/predictivity/top_benchmarks_per_language.png)

**Seed generalization** — holdout `predictivity_seeds_train` → `predictivity_seeds_test` (the ×3 cells only). A variant ranking whose Spearman ρ is low here is noise-dominated; recommend the family that transfers, not the argmax:

| metric | DA-size | DA-ckpt |
|---|---|---|
| Spearman ρ on global variant ranking | 0.01 | -0.21 |
| Pearson r between splits (all cells) | -0.43 | 0.05 |
| Exact-variant agreement (per lang) | 2% | 7% |
| Family-level agreement (per lang) | 3% | 15% |
| Retention of train-best r on test | 68% | 39% |
<!-- END auto:results -->

## Preliminary findings (ladder snapshot, 2026-09-01)

Noise on the ≤ 600M ladder before any seed replicate existed
(`plan/status-09-01.md`, §5; checkpoint noise = std over the last 3 evaluated
checkpoints, median 0.004): the across-L range of final benchmark scores at
600M has median 0.022 ≈ 4.4× that noise (56/60 benchmarks above 2×; top:
`xwinograd_en` 17×, `hellaswag_ru` 13×), and on BPB the separation is one to
two orders of magnitude above noise — the language-count axis gives genuinely
distinct models for SNR. The transformation table on the 14 September report
(18 seed pairs) puts the seed effect at Δ final loss −0.061 … +0.046,
Δ macro BPB −0.050 … +0.034 and Δ mean benchmark −0.004 … +0.007; the depth
effect on benchmarks is of the same order (−0.009 … +0.014 over the tasks each
pair shares), so
whether deep vs shallow is a distinct model for SNR is exactly what rq06's
effect-vs-noise table decides per task.

## External model-set tier (`all/external`, 36-sweep)

The `external` tier pools every non-custom model (reference HF, a06,
distillation, posttraining; sizes 270M…70B) with **no data-mixture axis**, so
"Signal" here is **cross-model dispersion** of final-checkpoint scores across the
external ladder — *how far different model families separate on a benchmark*,
rather than how far the three FineWeb mixtures separate. SNR magnitudes are
consequently an order larger than the custom pool's (e.g. `multiblimp_rus` SNR
≈ 119) and DA is the within-family scaling-DA over that ladder. The headline —
**recommend the dispersion *family*, not an exact variant** — holds on this
disjoint model set, the strongest robustness check we have. Regenerate with
`python analysis/rq02_snr_definition/snr_definition_postprocess.py --pool external`.

**Global variant ranking** — mean Pearson r of log₁₀(SNR) vs DA across languages.
The dispersion and relative-spread clusters lead overall and on DA-ckpt; the
discrepancy cluster leads DA-size only; depth metrics fail; the custom-pool winner
`dist_std` is undefined on this tier:

| variant (family) | DA-size r | DA-ckpt r | overall |
|---|---|---|---|
| `dispersion` / `mpd` / `range` / `mad` (dispersion) | 0.16 | **0.44** | **0.30** |
| `rel_std` / `rel_mpd` / `iqr` (relative-spread) | 0.16 | 0.43 | 0.30 |
| `mpsd` (dispersion) | 0.10 | **0.45** | 0.27 |
| `star_discrepancy` (discrepancy) | **0.20** | 0.05 | 0.12 |
| `discrepancy` / `gini` / `dispersion_shifted` (discrepancy) | 0.19 | 0.07 | 0.13 |
| `projection` (depth) | 0.06 | −0.21 | −0.07 |
| `dist_std`, `tukey` | — | — | — |

![SNR variants ranked by correlation with DA (external)](all/external/top_variants_overall.png)

DA-size is sparse on the external ladder (most cross-bucket pairs lack ≥2
spanning families), so **DA-ckpt is the more trustworthy axis here** — and on it
the dispersion/relative-spread families win cleanly, agreeing with the custom
pool's family-level recommendation.

**Most reliable benchmark per language** — rank-1 benchmark by `dist_std` SNR @ 1B
over above-random tasks. With capable models clearing the gate, the
long-completion 4-option `hellaswag_<lang>` joins MultiBLiMP at the top:

| lang | top benchmark | SNR | | lang | top benchmark | SNR |
|---|---|---|---|---|---|---|
| ar | `hellaswag_ar` | 78.9 | | ru | `multiblimp_rus` | 119.5 |
| en | `multiblimp_eng` | 82.1 | | th | `xcopa_th` | 10.1 |
| es | `multiblimp_spa` | 87.0 | | tr | `multiblimp_tur` | 40.5 |
| eu | `multiblimp_eus` | 38.4 | | vi | `hellaswag_vi` | 38.5 |
| hi | `multiblimp_hin` | 73.7 | | zh | `xstorycloze_zh` | 25.4 |
| ja | `xwinograd_jp` | 20.5 | | | | |

![Top benchmarks per language by SNR (external)](all/external/top_benchmarks_per_language.png)

The exact per-language argmax still does not transfer across tiers — only the
dispersion/relative-spread *family* does — so the paper-level claim is the family
recommendation, with HellaSwag and MultiBLiMP as the durable multilingual anchors.


## Results from the 36-model sweep (2026-06, superseded)

The numbers below were generated on the 36-model sweep (4 sizes × 3 data mixtures × 3 seeds, 12 languages, pool `custom_swissai_hf` unless stated) and are kept as history; the predictivity ladder regenerates the blocks above.

### Highlighted result

- **Global-best SNR definition (`custom_swissai_hf`): `dist_std`** — mean Pearson r of log₁₀(SNR) vs decision accuracy **0.32** (DA-size), **0.43** (DA-ckpt), 0.38 overall. DA-ckpt is led by the mean-pairwise-distance / relative-spread cluster (`mpsd`/`rel_mpd`/`rel_mpsd` ≈ 0.51) — all dispersion-family, so recommend the *family*, not an exact variant.
- **Per-language anchor: `multiblimp`** — the highest-SNR above-random benchmark in **6 of 11** languages (`dist_std` SNR @ 1B).
- **Variant ranking generalizes across seeds under DA-ckpt** (holdout Spearman ρ **0.81**) **but not under DA-size** (ρ **-0.07**): the DA-size per-variant correlations are small and near-tied, so their ranking is noise-dominated and does not survive a seed swap — treat DA-size variant ranking as unreliable. The exact per-language argmax never transfers. **Never `tukey` / `projection`** (r ≤ 0).

### Results

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
