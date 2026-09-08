# RQ1 — Which SNR definition best predicts decision accuracy?

## Research question

> Of 22 candidate SNR definitions, which best correlates with **decision
> accuracy** (DA) — the probability that a benchmark ranks two models the way
> a larger-model evaluation would — across the ladder's languages, and does the
> answer survive a change of training seed?

<!-- BEGIN auto:highlight (snr_definition_postprocess.py --pool predictivity) -->
## Highlighted result

- **Global-best SNR definition (`predictivity`): `rel_mpsd`** — mean Pearson r of log₁₀(SNR) vs decision accuracy **0.06** (DA-size), **0.32** (DA-ckpt), 0.19 overall. DA-ckpt is led by `rel_mpsd`/`rel_mpd`/`mad` (≈ 0.32; families: rel_spread, robust) — recommend the *family*, not an exact variant.
- **Per-language anchor: `bpb`** — the highest-SNR above-random benchmark in **90 of 96** languages (`rel_mpsd` SNR @ 1B). Weakest variants overall: `tukey`, `projection`.
- **Seed holdout (predictivity_seeds_train → predictivity_seeds_test)**: Spearman ρ of the global variant ranking **0.29** (DA-ckpt), **-0.09** (DA-size); family-level per-language agreement 10% / 1%. A ranking that does not survive the seed swap is noise-dominated — only the *family* recommendation transfers.
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
| `rel_mpsd` | 0.06 | 0.32 | 0.19 |
| `mpsd` | 0.06 | 0.27 | 0.16 |
| `rel_dispersion` | 0.05 | 0.06 | 0.05 |
| `rel_std` | 0.04 | 0.16 | 0.10 |
| `dispersion` | 0.04 | 0.03 | 0.04 |
| `range` | 0.04 | 0.03 | 0.04 |
| `rms_deviation` | 0.04 | 0.09 | 0.07 |
| … |  |  |  |
| `projection` | -0.20 | -0.29 | -0.24 |
| `tukey` | -0.35 | -0.22 | -0.29 |

![SNR variants ranked by correlation with DA](pretraining/predictivity/top_variants_overall.png)

**Statistical power by pool** — each pool's best DA-size variant:

| pool | best variant (DA-size) | DA-size r | DA-ckpt r |
|---|---|---|---|
| `predictivity` (grid, seed 1904) | `rel_mpsd` | 0.06 | 0.32 |
| `predictivity_seeds` (all seeds) | `rel_mpsd` | 0.28 | 0.42 |

**Most reliable benchmark per language** — `rel_mpsd` SNR @ 1B over above-random tasks (DA-size is undefined at the reference size itself, so DA-ckpt@1B is shown):

| lang | top benchmark | SNR | DA-ckpt@1B |
|---|---|---|---|
| af | `bpb_afr_Latn` | 0.97 | 1.00 |
| am | `bpb_amh_Ethi` | 0.00 | 0.25 |
| ar | `bpb_ary_Arab` | 13.71 | 1.00 |
| as | `bpb_asm_Beng` | 1.81 | 1.00 |
| az | `bpb_azj_Latn` | 48.42 | 1.00 |
| be | `bpb_bel_Cyrl` | 0.19 | 1.00 |
| bew | `bpb_bew_Latn` | 7.00 | 1.00 |
| bg | `bpb_bul_Cyrl` | 20.00 | 1.00 |
| bn | `bpb_ben_Beng` | 13.92 | 1.00 |
| bo | `bpb_bod_Tibt` | 0.00 | 0.50 |
| bs | `bpb_bos_Latn` | 43.76 | 1.00 |
| ca | `bpb_cat_Latn` | 21.13 | 1.00 |
| ckb | `bpb_ckb_Arab` | 0.73 | 1.00 |
| cs | `bpb_ces_Latn` | 42.66 | 1.00 |
| cy | `bpb_cym_Latn` | 0.00 | 0.75 |
| da | `bpb_dan_Latn` | 32.49 | 1.00 |
| de | `hellaswag_de` | 0.02 | 1.00 |
| dv | `bpb_div_Thaa` | 16.12 | 1.00 |
| el | `bpb_ell_Grek` | 14.99 | 1.00 |
| en | `arc_challenge` | 0.20 | 1.00 |
| eo | `bpb_epo_Latn` | 0.18 | 1.00 |
| es | `hellaswag_es` | 0.03 | 1.00 |
| et | `bpb_ekk_Latn` | 78.27 | 1.00 |
| eu | `bpb_eus_Latn` | 0.15 | 0.75 |
| fa | `bpb_fas_Arab` | 18.45 | 1.00 |
| fi | `bpb_fin_Latn` | 48.83 | 1.00 |
| fr | `xwinograd_fr` | 0.11 | 0.50 |
| ga | `bpb_gle_Latn` | 0.00 | 1.00 |
| gl | `bpb_glg_Latn` | 0.12 | 1.00 |
| gmh | `bpb_gmh_Latn` | 0.06 | 1.00 |
| gu | `bpb_guj_Gujr` | 0.00 | 1.00 |
| he | `bpb_heb_Hebr` | 23.07 | 1.00 |
| hi | `bpb_hin_Deva` | 7.33 | 1.00 |
| hif | `bpb_hif_Latn` | 0.11 | 1.00 |
| hr | `bpb_hrv_Latn` | 46.11 | 1.00 |
| hu | `bpb_hun_Latn` | 64.61 | 1.00 |
| hy | `bpb_hye_Armn` | 0.26 | 1.00 |
| id | `bpb_ind_Latn` | 13.43 | 1.00 |
| is | `bpb_isl_Latn` | 0.06 | 1.00 |
| it | `bpb_ita_Latn` | 0.02 | 1.00 |
| ja | `bpb_jpn_Jpan` | 0.02 | 1.00 |
| ka | `bpb_kat_Geor` | 8.83 | 1.00 |
| kk | `bpb_kaz_Cyrl` | 20.20 | 1.00 |
| km | `bpb_khm_Khmr` | 0.01 | 0.75 |
| kmr | `bpb_kmr_Latn` | 0.11 | 1.00 |
| kn | `bpb_kan_Knda` | 0.01 | 0.75 |
| ko | `bpb_kor_Hang` | 14.15 | 1.00 |
| ky | `bpb_kir_Cyrl` | 0.28 | 1.00 |
| la | `bpb_lat_Latn` | 0.00 | 0.25 |
| lb | `bpb_ltz_Latn` | 0.02 | 0.50 |
| lo | `bpb_lao_Laoo` | 0.03 | 0.50 |
| lt | `bpb_lit_Latn` | 77.33 | 1.00 |
| lv | `bpb_lvs_Latn` | 56.98 | 1.00 |
| mg | `bpb_plt_Latn` | 0.02 | 0.75 |
| mk | `bpb_mkd_Cyrl` | 3.95 | 1.00 |
| ml | `bpb_mal_Mlym` | 15.36 | 1.00 |
| mn | `bpb_khk_Cyrl` | 0.01 | 0.25 |
| mr | `bpb_mar_Deva` | 17.78 | 1.00 |
| ms | `bpb_zsm_Latn` | 11.28 | 1.00 |
| mt | `bpb_mlt_Latn` | 0.02 | 1.00 |
| multi | `bpb_macro` | 7.94 | 1.00 |
| my | `bpb_mya_Mymr` | 0.01 | 1.00 |
| ne | `bpb_npi_Deva` | 10.77 | 1.00 |
| nl | `bpb_nld_Latn` | 16.22 | 1.00 |
| nn | `bpb_nno_Latn` | 33.22 | 1.00 |
| no | `bpb_nob_Latn` | 34.21 | 1.00 |
| nrm | `bpb_nrm_Latn` | 1.08 | 1.00 |
| or | `bpb_ory_Orya` | 0.02 | 0.75 |
| pa | `bpb_pan_Guru` | 0.07 | 0.50 |
| pl | `bpb_pol_Latn` | 33.80 | 1.00 |
| ps | `bpb_pbt_Arab` | 0.08 | 1.00 |
| pt | `bpb_por_Latn` | 5.89 | 1.00 |
| ro | `bpb_ron_Latn` | 32.21 | 1.00 |
| ru | `hellaswag_ru` | 0.03 | 1.00 |
| sd | `bpb_snd_Arab` | 1.28 | 0.75 |
| si | `bpb_sin_Sinh` | 0.00 | 0.75 |
| sk | `bpb_slk_Latn` | 66.30 | 1.00 |
| sl | `bpb_slv_Latn` | 40.83 | 1.00 |
| so | `bpb_som_Latn` | 0.03 | 1.00 |
| sq | `bpb_als_Latn` | 78.86 | 1.00 |
| sr | `bpb_srp_Latn` | 47.48 | 1.00 |
| sv | `bpb_swe_Latn` | 27.46 | 1.00 |
| sw | `bpb_swh_Latn` | 0.02 | 0.75 |
| ta | `bpb_tam_Taml` | 6.76 | 1.00 |
| te | `bpb_tel_Telu` | 0.00 | 1.00 |
| tg | `bpb_tgk_Cyrl` | 0.02 | 0.00 |
| th | `bpb_tha_Thai` | 11.39 | 1.00 |
| tl | `bpb_fil_Latn` | 0.00 | 0.75 |
| tr | `bpb_tur_Latn` | 37.25 | 1.00 |
| tt | `bpb_tat_Cyrl` | 0.02 | 1.00 |
| ug | `bpb_uig_Arab` | 0.01 | 1.00 |
| uk | `bpb_ukr_Cyrl` | 7.83 | 1.00 |
| ur | `bpb_urd_Arab` | 13.22 | 1.00 |
| uz | `bpb_uzn_Latn` | 0.23 | 1.00 |
| vi | `bpb_vie_Latn` | 16.67 | 1.00 |
| zh | `xcopa_zh` | 0.01 | 1.00 |

![Top-5 benchmarks per language by SNR](pretraining/predictivity/top_benchmarks_per_language.png)

**Seed generalization** — holdout `predictivity_seeds_train` → `predictivity_seeds_test` (the ×3 cells only). A variant ranking whose Spearman ρ is low here is noise-dominated; recommend the family that transfers, not the argmax:

| metric | DA-size | DA-ckpt |
|---|---|---|
| Spearman ρ on global variant ranking | -0.09 | 0.29 |
| Pearson r between splits (all cells) | 0.52 | 0.03 |
| Exact-variant agreement (per lang) | 1% | 4% |
| Family-level agreement (per lang) | 1% | 10% |
| Retention of train-best r on test | 58% | 35% |
<!-- END auto:results -->

## Preliminary findings (ladder snapshot, 2026-09-01)

Noise on the ≤ 600M ladder before any seed replicate existed
(`plan/status-09-01.md`, §5; checkpoint noise = std over the last 3 evaluated
checkpoints, median 0.004): the across-L range of final benchmark scores at
600M has median 0.022 ≈ 4.4× that noise (56/60 benchmarks above 2×; top:
`xwinograd_en` 17×, `hellaswag_ru` 13×), and on BPB the separation is one to
two orders of magnitude above noise — the language-count axis gives genuinely
distinct models for SNR. The `ladder_report.md` transformation table
(2026-09-03, six seed pairs) puts the seed effect at |Δ final loss| ≤ 0.043,
|Δ macro BPB| ≤ 0.024 and |Δ mean benchmark| ≈ 0.013; the depth effect on
benchmarks is of the same order (−0.005 … +0.003 on the mean over tasks), so
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
