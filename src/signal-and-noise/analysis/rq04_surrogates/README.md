# RQ4 — Surrogates: which cheap statistic predicts decision accuracy?

## Research question

> Before training the reference, can a statistic computed on the proxy alone
> tell us that the proxy's decision will match the reference's? First among
> the 22 SNR definitions of rq03: which best correlates with **decision
> accuracy** (rq02) across the ladder's languages, and does the answer
> survive a change of training seed? Then beyond SNR (the paper's RQ3,
> [`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex)):
> its signal or noise part alone, the proxy's early-checkpoint agreement, how
> well its scores fit a scaling trend (rq01), its margin above chance (rq00).

<!-- BEGIN auto:highlight (snr_definition_postprocess.py --pool predictivity) -->
## Highlighted result

- **Global-best SNR definition (`predictivity`): `mpsd`** — mean Pearson r of log₁₀(SNR) vs decision accuracy **0.05** (DA-size), **0.32** (DA-ckpt), 0.19 overall. DA-ckpt is led by `mpsd`/`rel_mpsd`/`aad` (≈ 0.32; families: dispersion, rel_spread) — recommend the *family*, not an exact variant.
- **Per-language anchor: `multiblimp`** — the highest-SNR above-random benchmark in **55 of 66** languages (`mpsd` SNR @ 1B); the language's own BPB, ungated and on its own noise scale, outranks that benchmark in 55 of the 57 languages that have both. Weakest variants overall: `tukey`, `projection`.
- **Seed holdout (predictivity_seeds_train → predictivity_seeds_test)**: Spearman ρ of the global variant ranking **0.07** (DA-ckpt), **0.75** (DA-size); family-level per-language agreement 29% / 57%. A ranking that does not survive the seed swap is noise-dominated — only the *family* recommendation transfers.
<!-- END auto:highlight -->

## Experimental setup

Outputs live under `pretraining/<pool>/` for the ladder pools: `predictivity`
(the plan grid, seed 1904 — the headline), `predictivity_seeds` (every seed;
replicates enter the signal pool as separate models), and the seed holdout
`predictivity_seeds_train` (seeds 64/313 at the ×3 cells) → `_test` (seed 1904
on the same cells). The signal population at a size is every design variant
trained there (language setting × depth × scheme); the noise is the
late-checkpoint std over the last 5 checkpoints on the shared grid (rq03
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

## Methodology

- **SNR per (task, size bucket)** is rq03's table (`snr_variants_per_task.csv`:
  22 variants, gated cells NaN, coverage per variant recorded there).
- **Decision accuracy** is joined from rq02 (`decision_acc_size_*`,
  `decision_acc_ckpt_*`; pair counts in `da_n_pairs_per_task.csv`).
- **Ranking.** Per language, Pearson r of log₁₀(SNR) against DA over the
  (task, bucket) cells; the global variant is the highest mean r over
  languages and over both DA kinds (`top_variants_overall.csv`), and the
  per-language "most reliable benchmark" table reads that variant at the
  reference size.
- **Seed holdout.** The same per-language table on the replicate seeds
  (64/313) and on seed 1904 of the same cells; agreement is counted over
  the languages that have a best variant on both splits.

<!-- BEGIN auto:results (snr_definition_postprocess.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq04_surrogates/snr_definition_postprocess.py --pool predictivity`.

**Global variant ranking** — mean Pearson r of log₁₀(SNR) vs DA across languages:

| variant | DA-size r | DA-ckpt r | overall |
|---|---|---|---|
| `mpsd` | 0.05 | 0.32 | 0.19 |
| `rel_mpsd` | 0.05 | 0.31 | 0.18 |
| `aad` | 0.04 | 0.30 | 0.17 |
| `dist_std` | 0.04 | 0.29 | 0.17 |
| `rms_deviation` | 0.04 | 0.29 | 0.17 |
| `mpd` | 0.04 | 0.29 | 0.16 |
| `dispersion` | 0.03 | 0.28 | 0.16 |
| … |  |  |  |
| `projection` | -0.16 | -0.07 | -0.11 |
| `tukey` | -0.20 | -0.12 | -0.16 |

![SNR variants ranked by correlation with DA](pretraining/predictivity/top_variants_overall.png)

**Statistical power by pool** — each pool's best variant (mean r over both DA kinds):

| pool | best variant (overall) | DA-size r | DA-ckpt r |
|---|---|---|---|
| `predictivity` (grid, seed 1904) | `mpsd` | 0.05 | 0.32 |
| `predictivity_seeds` (all seeds) | `mpsd` | 0.08 | 0.30 |

**Most reliable benchmark per language** — `mpsd` SNR @ 1B over the above-random benchmarks, with the language's own BPB SNR alongside (ungated, on its own noise scale; DA-size is undefined at the reference size itself, so DA-ckpt@1B is shown):

| lang | top benchmark | SNR | DA-ckpt@1B | BPB SNR |
|---|---|---|---|---|
| am | `multiblimp_amh` | 0.10 | 0.42 | 0.11 |
| ar | `multiblimp_arb` | 1.57 | 0.84 | 10.22 |
| be | `multiblimp_bel` | 0.40 | 0.76 | 3.35 |
| bg | `multiblimp_bul` | 2.54 | 0.91 | 8.32 |
| bn | `multiblimp_ben` | 0.62 | 0.68 | 5.64 |
| bs | `global_piqa_nonparallel_cloze_bos_latn` | 0.27 | 0.62 | 17.41 |
| ca | `multiblimp_cat` | 1.25 | 0.83 | 10.97 |
| cs | `multiblimp_ces` | 4.12 | 0.88 | 28.78 |
| cy | `multiblimp_cym` | 0.09 | 0.71 | 0.47 |
| da | `multiblimp_dan` | 0.34 | 0.64 | 16.78 |
| de | `multiblimp_deu` | 1.52 | 0.82 | 7.15 |
| el | `multiblimp_ell` | 1.82 | 0.78 | 5.90 |
| en | `xnli_en` | 0.03 | 0.72 | 0.04 |
| es | `multiblimp_spa` | 0.57 | 0.85 | 4.06 |
| et | `multiblimp_est` | 1.81 | 0.74 | 42.67 |
| eu | `multiblimp_eus` | 0.00 | 0.53 | 0.79 |
| fa | `multiblimp_fas` | 1.18 | 0.83 | 7.96 |
| fi | `multiblimp_fin` | 3.87 | 0.77 | 35.62 |
| fo | `multiblimp_fao` | 0.02 | 0.54 |  |
| fr | `multiblimp_fra` | 0.67 | 0.72 | 3.48 |
| ga | `multiblimp_gle` | 0.10 | 0.55 | 0.32 |
| gd | `multiblimp_gla` | 0.06 | 0.39 |  |
| gl | `multiblimp_glg` | 0.23 | 0.79 | 3.88 |
| grc | `multiblimp_grc` | 0.01 | 0.64 |  |
| gu | `multiblimp_guj` | 0.22 | 0.44 | 0.02 |
| hbo | `multiblimp_hbo` | 0.04 | 0.72 |  |
| he | `multiblimp_heb` | 1.03 | 0.83 | 13.93 |
| hi | `multiblimp_hin` | 2.34 | 0.77 | 2.68 |
| hr | `hellaswag_hr` | 0.16 | 0.75 | 18.26 |
| hu | `multiblimp_hun` | 3.48 | 0.82 | 57.00 |
| hy | `multiblimp_hye` | 0.06 | 0.60 | 0.27 |
| hyw | `multiblimp_hyw` | 0.07 | 0.59 |  |
| id | `xstorycloze_id` | 0.48 | 0.84 | 7.79 |
| is | `multiblimp_isl` | 0.00 | 0.65 | 0.12 |
| it | `multiblimp_ita` | 2.32 | 0.81 | 7.66 |
| ja | `xwinograd_jp` | 0.40 | 0.71 | 11.10 |
| ka | `multiblimp_kat` | 0.30 | 0.71 | 5.13 |
| kk | `multiblimp_kaz` | 0.31 | 0.70 | 6.47 |
| kmr | `multiblimp_kmr` | 0.01 | 0.61 | 0.26 |
| ky | `multiblimp_kir` | 0.01 | 0.27 | 1.31 |
| la | `multiblimp_lat` | 0.02 | 0.80 | 0.01 |
| lt | `multiblimp_lit` | 0.95 | 0.61 | 37.12 |
| mk | `multiblimp_mkd` | 0.14 | 0.68 | 1.50 |
| mr | `multiblimp_mar` | 0.09 | 0.60 | 3.95 |
| multi | `train_loss` | 1.37 | 0.97 | 2.18 |
| nds | `multiblimp_nds` | 0.01 | 0.60 |  |
| nl | `multiblimp_nld` | 1.82 | 0.88 | 12.69 |
| pl | `multiblimp_pol` | 3.89 | 0.92 | 27.28 |
| pt | `multiblimp_por` | 1.91 | 0.87 | 5.85 |
| ro | `multiblimp_ron` | 3.47 | 0.87 | 23.15 |
| ru | `multiblimp_rus` | 3.32 | 0.85 | 4.05 |
| sa | `multiblimp_san` | 0.02 | 0.64 |  |
| sah | `multiblimp_sah` | 0.04 | 0.57 |  |
| se | `multiblimp_sme` | 0.02 | 0.42 |  |
| sk | `multiblimp_slk` | 1.98 | 0.90 | 31.50 |
| sl | `multiblimp_slv` | 1.36 | 0.85 | 22.54 |
| sr | `hellaswag_sr` | 0.14 | 0.78 | 21.32 |
| sv | `hellaswag_sv` | 0.33 | 0.92 | 19.11 |
| ta | `multiblimp_tam` | 0.18 | 0.86 | 2.33 |
| th | `xnli_th` | 0.14 | 0.87 | 4.90 |
| tr | `multiblimp_tur` | 1.11 | 0.74 | 22.81 |
| ug | `multiblimp_uig` | 0.03 | 0.61 | 0.14 |
| uk | `multiblimp_ukr` | 2.04 | 0.90 | 6.28 |
| ur | `multiblimp_urd` | 0.24 | 0.77 | 4.32 |
| vi | `xcopa_vi` | 0.44 | 0.79 | 9.86 |
| zh | `xstorycloze_zh` | 0.19 | 0.82 | 10.08 |

![Top-5 benchmarks per language by SNR](pretraining/predictivity/top_benchmarks_per_language.png)

**Seed generalization** — holdout `predictivity_seeds_train` → `predictivity_seeds_test` (the ×3 cells only). A variant ranking whose Spearman ρ is low here is noise-dominated; recommend the family that transfers, not the argmax:

| metric | DA-size | DA-ckpt |
|---|---|---|
| Spearman ρ on global variant ranking | 0.75 | 0.07 |
| Pearson r between splits (all cells) | -0.14 | 0.19 |
| Exact-variant agreement (per lang) | 43% | 15% |
| Family-level agreement (per lang) | 57% | 29% |
| Retention of train-best r on test | 82% | 41% |
<!-- END auto:results -->

### Statistics beyond SNR — setup

The truth is DA-size from [`rq02_decision_accuracy/`](../rq02_decision_accuracy/):
per task, the agreement between the ranking of the design variants at a proxy
size and at the target size, as carried in rq03's per-task table
(`snr_variants_per_task.csv`, `decision_acc_size_<proxy>`). The candidates are
read from the same table (`snr_*`, `signal_*`, `noise_*`, `decision_acc_ckpt_*`),
from rq00's above-random scores (margin above chance) and from rq01's log-N
fits (R²). The population per proxy size is the set of tasks with an SNR at
that size — the above-random survivors — so every candidate is scored on the
same tasks; benchmarks and per-language BPB are scored separately. Spearman ρ
between the candidate and DA-size over the population, per proxy size (the
`snr` block's `small_sizes` that have a DA-size column), with the p-value and
n; a cell needs at least 8 tasks; the noise part is inverted so that "higher
is better" holds for every candidate.

<!-- BEGIN auto:surrogates (analyze.py --pool predictivity) -->
## Statistics beyond SNR (paper RQ3)

Numbers from the `predictivity` pool's rq03 table. Regenerate with `python analysis/rq04_surrogates/analyze.py --pool predictivity`.

- **benchmark tasks** — strongest surrogate of DA-size (mean ρ over proxies): `SNR, relative std` 0.54; weakest: `margin above chance` -0.02.
- **per-language bits per byte** — strongest surrogate of DA-size (mean ρ over proxies): `SNR, relative std` 0.61; weakest: `SNR, discrepancy` -0.28.

**benchmark tasks** (Spearman ρ of the statistic with DA-size, per proxy size):

| metric | 175M | 350M | 600M |
|---|---|---|---|
| SNR, relative std | 0.50 | 0.63 | 0.50 |
| early-checkpoint agreement (20 %) | 0.37 | 0.65 | 0.57 |
| signal alone (relative std) | 0.49 | 0.55 | 0.31 |
| scaling-fit R² | 0.32 | 0.43 | 0.57 |
| SNR, dist_std | 0.38 | 0.49 | 0.37 |
| noise alone (relative std, inverted) | 0.04 | 0.17 | 0.23 |
| SNR, discrepancy | -0.00 | 0.06 | 0.09 |
| margin above chance | 0.17 | -0.03 | -0.19 |

**per-language bits per byte** (Spearman ρ of the statistic with DA-size, per proxy size):

| metric | 175M | 350M | 600M |
|---|---|---|---|
| SNR, relative std | 0.69 | 0.60 | 0.52 |
| early-checkpoint agreement (20 %) | 0.61 | 0.62 | 0.55 |
| SNR, dist_std | 0.64 | 0.58 | 0.52 |
| signal alone (relative std) | 0.62 | 0.50 | 0.48 |
| scaling-fit R² | 0.22 | 0.14 | 0.10 |
| noise alone (relative std, inverted) | 0.23 | 0.10 | 0.06 |
| SNR, discrepancy |  |  | -0.28 |

![Surrogates](pretraining/predictivity/rq3_surrogates.png)
<!-- END auto:surrogates -->

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
`python analysis/rq04_surrogates/snr_definition_postprocess.py --pool external`.

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

Headline numbers from the `custom_swissai_hf` pool. Regenerate with `python analysis/rq04_surrogates/snr_definition_postprocess.py --pool custom_swissai_hf`.

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

- `pretraining/<pool>/snr_variant_ranking.csv` — per-(variant, DA-def, scope)
  Pearson r (`analyze_snr_variants.py`).
- `…/top_variants_overall.csv`, `best_variant_per_language.csv`,
  `best_variant_family_per_language.csv`, `variant_clusters.csv`,
  `top_benchmarks_per_language.csv` — the variant ranking and the per-language
  anchor (`snr_definition_postprocess.py`).
- `…/top_variants_overall.png`, `best_variant_per_language.png`,
  `best_variant_family_per_language.png`, `top_benchmarks_per_language.png`,
  `variant_correlation_matrix.png`, `da_size_vs_da_ckpt.png`,
  `{da_size,da_ckpt}/…` — supporting figures.
- `…/rq3_surrogates.csv`, `rq3_surrogates.png/.pdf`, `facts.json` — the
  statistics beyond SNR (`analyze.py`; the paper's RQ3 figure).
- Inputs: rq03's `snr_variants_per_task.csv` and holdout
  `headline_metrics.csv`, rq00's `above_random_scores.csv`, rq01's `rq1_fits.csv`.
