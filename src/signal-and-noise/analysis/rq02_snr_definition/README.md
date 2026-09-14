# RQ1 — Which SNR definition best predicts decision accuracy?

## Research question

> Of 22 candidate SNR definitions, which best correlates with **decision
> accuracy** (DA) — the probability that a benchmark ranks two models the way
> a larger-model evaluation would — across the ladder's languages, and does the
> answer survive a change of training seed?

<!-- BEGIN auto:highlight (snr_definition_postprocess.py --pool predictivity) -->
## Highlighted result

- **Global-best SNR definition (`predictivity`): `discrepancy`** — mean Pearson r of log₁₀(SNR) vs decision accuracy **0.06** (DA-size), **-0.05** (DA-ckpt), 0.00 overall. DA-ckpt is led by `dist_std`/`mad`/`rel_mpsd` (≈ 0.20; families: dispersion, rel_spread, robust) — recommend the *family*, not an exact variant.
- **Per-language anchor: `multiblimp`** — the highest-SNR above-random benchmark in **55 of 68** languages (`discrepancy` SNR @ 1B). Weakest variants overall: `projection`, `rel_star_discrepancy`.
- **Seed holdout (predictivity_seeds_train → predictivity_seeds_test)**: Spearman ρ of the global variant ranking **0.01** (DA-ckpt), **-0.20** (DA-size); family-level per-language agreement 11% / 1%. A ranking that does not survive the seed swap is noise-dominated — only the *family* recommendation transfers.
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
| `discrepancy` | 0.06 | -0.05 | 0.00 |
| `rel_mpsd` | -0.00 | 0.07 | 0.03 |
| `mpsd` | -0.00 | 0.05 | 0.02 |
| `dist_std` | -0.01 | 0.20 | 0.09 |
| `rel_std` | -0.01 | -0.04 | -0.03 |
| `rel_dispersion` | -0.01 | -0.08 | -0.05 |
| `rms_deviation` | -0.01 | -0.04 | -0.03 |
| … |  |  |  |
| `projection` | -0.09 | -0.35 | -0.22 |
| `tukey` | -0.12 | -0.12 | -0.12 |

![SNR variants ranked by correlation with DA](pretraining/predictivity/top_variants_overall.png)

**Statistical power by pool** — each pool's best DA-size variant:

| pool | best variant (DA-size) | DA-size r | DA-ckpt r |
|---|---|---|---|
| `predictivity` (grid, seed 1904) | `discrepancy` | 0.06 | -0.05 |
| `predictivity_seeds` (all seeds) | `discrepancy` | 0.12 | -0.05 |

**Most reliable benchmark per language** — `discrepancy` SNR @ 1B over above-random tasks (DA-size is undefined at the reference size itself, so DA-ckpt@1B is shown):

| lang | top benchmark | SNR | DA-ckpt@1B |
|---|---|---|---|
| am | `multiblimp_amh` | 5.46 | 0.75 |
| ar | `multiblimp_arb` | 15.37 | 1.00 |
| be | `multiblimp_bel` | 6.45 | 0.75 |
| bg | `multiblimp_bul` | 14.52 | 1.00 |
| bn | `bpb_ben_Beng` | 3.09 | 1.00 |
| bo | `bpb_bod_Tibt` | 6.38 | 0.50 |
| ca | `multiblimp_cat` | 24.81 | 1.00 |
| cs | `multiblimp_ces` | 9.81 | 1.00 |
| cy | `multiblimp_cym` | 5.12 | 0.75 |
| da | `multiblimp_dan` | 11.32 | 1.00 |
| de | `multiblimp_deu` | 59.34 | 0.50 |
| el | `multiblimp_ell` | 21.51 | 1.00 |
| en | `multiblimp_eng` | 103.90 | 0.75 |
| es | `multiblimp_spa` | 122.55 | 0.50 |
| et | `multiblimp_est` | 12.43 | 1.00 |
| eu | `multiblimp_eus` | 32.68 | 1.00 |
| fa | `multiblimp_fas` | 7.67 | 1.00 |
| fi | `multiblimp_fin` | 13.22 | 1.00 |
| fo | `multiblimp_fao` | 1.76 | 0.50 |
| fr | `multiblimp_fra` | 164.21 | 0.75 |
| ga | `multiblimp_gle` | 2.59 | 0.75 |
| gd | `multiblimp_gla` | 16.74 | 1.00 |
| gl | `multiblimp_glg` | 13.22 | 0.25 |
| grc | `multiblimp_grc` | 6.87 | 1.00 |
| gu | `multiblimp_guj` | 5.67 | 1.00 |
| hbo | `multiblimp_hbo` | 7.02 | 1.00 |
| he | `multiblimp_heb` | 6.22 | 1.00 |
| hi | `multiblimp_hin` | 18.06 | 1.00 |
| hu | `multiblimp_hun` | 13.68 | 1.00 |
| hy | `multiblimp_hye` | 7.72 | 0.50 |
| hyw | `multiblimp_hyw` | 3.07 | 0.50 |
| id | `hellaswag_id` | 8.57 | 1.00 |
| is | `multiblimp_isl` | 7.29 | 0.50 |
| it | `multiblimp_ita` | 51.85 | 0.75 |
| ja | `bpb_jpn_Jpan` | 15.94 | 1.00 |
| ka | `multiblimp_kat` | 16.54 | 1.00 |
| kk | `multiblimp_kaz` | 7.21 | 1.00 |
| kmr | `multiblimp_kmr` | 12.17 | 0.75 |
| kn | `bpb_kan_Knda` | 14.30 | 0.75 |
| ky | `multiblimp_kir` | 24.45 | 1.00 |
| la | `multiblimp_lat` | 3.91 | 1.00 |
| lt | `multiblimp_lit` | 14.57 | 1.00 |
| mk | `multiblimp_mkd` | 7.28 | 0.50 |
| ml | `bpb_mal_Mlym` | 3.80 | 1.00 |
| mr | `bpb_mar_Deva` | 7.13 | 1.00 |
| nds | `multiblimp_nds` | 16.56 | 1.00 |
| ne | `bpb_npi_Deva` | 3.03 | 1.00 |
| nl | `multiblimp_nld` | 13.19 | 1.00 |
| pl | `multiblimp_pol` | 22.59 | 1.00 |
| pt | `multiblimp_por` | 29.55 | 1.00 |
| ro | `multiblimp_ron` | 18.35 | 1.00 |
| ru | `multiblimp_rus` | 116.85 | 0.75 |
| sa | `multiblimp_san` | 12.16 | 0.00 |
| sah | `multiblimp_sah` | 8.65 | 0.50 |
| se | `multiblimp_sme` | 12.20 | 0.50 |
| sk | `multiblimp_slk` | 8.23 | 1.00 |
| sl | `multiblimp_slv` | 21.29 | 1.00 |
| sv | `multiblimp_swe` | 200.48 | 1.00 |
| ta | `multiblimp_tam` | 13.76 | 1.00 |
| te | `bpb_tel_Telu` | 14.03 | 1.00 |
| th | `bpb_tha_Thai` | 2.91 | 1.00 |
| tr | `multiblimp_tur` | 15.30 | 1.00 |
| ug | `multiblimp_uig` | 11.93 | 1.00 |
| uk | `multiblimp_ukr` | 20.51 | 1.00 |
| ur | `multiblimp_urd` | 10.82 | 1.00 |
| uz | `include_base_44_uzbek` | 0.45 | 1.00 |
| vi | `hellaswag_vi` | 8.49 | 1.00 |
| zh | `xstorycloze_zh` | 13.94 | 0.50 |

![Top-5 benchmarks per language by SNR](pretraining/predictivity/top_benchmarks_per_language.png)

**Seed generalization** — holdout `predictivity_seeds_train` → `predictivity_seeds_test` (the ×3 cells only). A variant ranking whose Spearman ρ is low here is noise-dominated; recommend the family that transfers, not the argmax:

| metric | DA-size | DA-ckpt |
|---|---|---|
| Spearman ρ on global variant ranking | -0.20 | 0.01 |
| Pearson r between splits (all cells) | 0.27 | -0.06 |
| Exact-variant agreement (per lang) | 1% | 5% |
| Family-level agreement (per lang) | 1% | 11% |
| Retention of train-best r on test | 100% | 41% |
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
