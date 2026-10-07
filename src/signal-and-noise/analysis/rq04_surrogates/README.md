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
> Finally, ~210 statistics from the literature and the AllenAI signal × noise grid (`catalogue.py`, `search.py`,
> [`literature.md`](literature.md)), per language, tier and benchmark.

<!-- BEGIN auto:highlight (snr_definition_postprocess.py --pool predictivity) -->
## Highlighted result

- **Global-best SNR definition (`predictivity`): `aad`** — mean Pearson r of log₁₀(SNR) vs decision accuracy **0.34** (DA-size, proxy → 1.7B, 50 languages), **0.47** (DA-ckpt, proxy sizes pooled, 50 languages), 0.41 overall. DA-ckpt is led by `aad`/`quartile_deviation`/`mpd` (≈ 0.47; one family: dispersion) — recommend the *family*, not an exact variant.
- **Per-language anchor: `bbpb_multiblimp`** — the highest-SNR above-random benchmark in **12 of 50** languages (`aad` SNR @ 1.7B; `train_loss` and `bpb_macro` are not a language's and are left out); the language's own BPB, ungated and on its own noise scale, outranks that benchmark in 1 of the 50 languages that have both. Weakest variants overall: `projection`, `tukey`.
- **Seed holdout (predictivity_seeds_train → predictivity_seeds_test)**: Spearman ρ of the global variant ranking **0.82** (DA-ckpt), **-0.28** (DA-size); family-level per-language agreement 100% / 0%. A ranking that does not survive the seed swap is noise-dominated — only the *family* recommendation transfers.
<!-- END auto:highlight -->

## Experimental setup

Outputs live under `pretraining/<pool>/` for the ladder pools
([RULES.md](../RULES.md), Definitions): `predictivity` (the grid seed, every
ladder and data build — the headline), `predictivity_seeds` (every seed;
replicates enter the signal pool as separate models), and the seed holdout
`predictivity_seeds_train` (the replicate seeds 64/313 at 175M/600M and 28/1797
at 1B) → `_test` (seed 1904 on exactly the same cells; until 2026-10-05 the
holdout was the 175M/600M cells alone). The signal population at a size is every design variant
trained there (language setting × ladder × data build); the noise is the
late-checkpoint std over the noise window, the k/20 points in the last 20 %
of the run — 80/85/90/95/100 % (rule 4) — (rq03
carries the seed-replicate noise). DA has two flavours: **DA-size**
(small→1.7B ranking, plus every other bucket pair) and **DA-ckpt**
(20/40/60/80 % → final within a size). The 22 variants are grouped into
families (dispersion / relative-spread / discrepancy / robust / depth) and
correlated with DA per language as Pearson r of log₁₀(SNR) vs DA; the
per-language BPB tasks (`bpb_<subset>`) take part like any benchmark.
The seed holdout is English-heavy by construction: among the ×3 cells
(L ∈ {1, 2, 50} at 175M/600M, L ∈ {1, 2, 30} at 1B) a non-English harness task
is only trained at L30/L50 (Russian also at L2), so its per-language variant
ranking rests on few language settings per seed, while English and the BPB
tasks cover them all.

## Methodology

- **SNR per (task, size bucket)** is rq03's table (`snr_variants_per_task.csv`:
  22 variants, gated cells NaN, coverage per variant recorded there).
- **Decision accuracy** is joined from rq02 (`decision_acc_size_*`,
  `decision_acc_ckpt_*`; pair counts in `da_all_n_pairs_per_task_both_axes.csv`).
- **Ranking.** Per language, Pearson r of log₁₀(SNR) against DA over the
  (task, bucket) cells; the global variant is the highest mean r over
  languages and over both DA kinds (`top_variants_overall.csv`), and the
  per-language "most reliable benchmark" table reads that variant at the
  reference size.
- **Seed holdout.** The same per-language table on the replicate seeds
  (64/313/28/1797) and on seed 1904 of the same cells; agreement is counted over
  the languages that have a best variant on both splits.

Hand-written numbers in this README are from the ladder-report snapshot
**2026-09-30 23:54**.

<!-- BEGIN auto:results (snr_definition_postprocess.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq04_surrogates/snr_definition_postprocess.py --pool predictivity`.

**Global variant ranking** — mean Pearson r of log₁₀(SNR) vs DA across the trained languages with ≥ 3 tasks (rule 8; 50 languages under DA-size, 50 under DA-ckpt). DA-size = proxy final → 1.7B final only (the proxy-to-proxy scaling pairs are not DA-size, rule 9); DA-ckpt = a proxy size's early checkpoints → its final, the proxy sizes 90M, 175M, 350M, 600M, 1B pooled, never the 1.7B run's own checkpoints (rule 11):

| variant | DA-size r | DA-ckpt r | overall |
|---|---|---|---|
| `aad` | 0.34 | 0.47 | 0.41 |
| `quartile_deviation` | 0.35 | 0.47 | 0.41 |
| `rms_deviation` | 0.34 | 0.47 | 0.40 |
| `mpd` | 0.34 | 0.47 | 0.40 |
| `dist_std` | 0.34 | 0.46 | 0.40 |
| `dispersion` | 0.32 | 0.45 | 0.39 |
| `range` | 0.32 | 0.45 | 0.39 |
| … |  |  |  |
| `tukey` | -0.18 | -0.27 | -0.23 |
| `projection` | -0.21 | -0.26 | -0.24 |

![SNR variants ranked by correlation with DA](pretraining/predictivity/top_variants_overall.png)

**Statistical power by pool** — each pool's best variant (mean r over both DA kinds):

| pool | best variant (overall) | DA-size r | DA-ckpt r |
|---|---|---|---|
| `predictivity` (grid, seed 1904, every design) | `aad` | 0.34 | 0.47 |
| `predictivity_seeds` (every seed) | `aad` | 0.34 | 0.46 |

**Most reliable benchmark per language** — `aad` SNR @ 1.7B over the above-random benchmarks, with the language's own BPB SNR alongside (ungated, on its own noise scale; DA-size is undefined at the reference size itself, so DA-ckpt@1.7B is shown; `train_loss` and `bpb_macro` measure the whole mixture and are not a row, rule 7):

| lang | top benchmark | SNR | DA-ckpt@1.7B | BPB SNR |
|---|---|---|---|---|
| ar | `bbpb_rf_belebele_arb_Latn` | 3.33 | 0.60 | 0.75 |
| az | `bbpb_include_v2_og_azerbaijani_azerbaijan` | 12.65 | 0.98 | 1.71 |
| bg | `multiblimp_bul` | 2.62 | 0.81 | 0.80 |
| bn | `bbpb_global_piqa_parallel_cloze_ben_latn` | 3.09 | 0.67 | 0.74 |
| bs | `bbpb_global_piqa_nonparallel_cloze_bos_latn` | 2.49 | 0.87 | 2.06 |
| ca | `bbpb_xnli_ca` | 5.88 | 0.89 | 1.65 |
| cs | `bbpb_multiblimp_ces` | 1.95 | 0.76 | 0.44 |
| da | `multiblimp_dan` | 3.21 | 0.46 | 0.97 |
| de | `bbpb_multiblimp_deu` | 2.07 | 0.73 | 0.63 |
| el | `bbpb_belebele_ell_Grek` | 2.56 | 0.67 | 0.27 |
| en | `bbpb_acp_bench_cloze_val` | 4.33 | 0.59 | 1.41 |
| es | `bbpb_xnli_es` | 3.04 | 0.79 | 0.83 |
| et | `bbpb_multiblimp_est` | 7.11 | 0.91 | 2.36 |
| fa | `bbpb_belebele_pes_Arab` | 2.00 | 0.71 | 0.65 |
| fi | `bbpb_multiblimp_fin` | 3.01 | 0.80 | 1.29 |
| fr | `bbpb_include_v2_og_french_canada` | 3.68 | 0.65 | 0.48 |
| he | `bbpb_multiblimp_heb` | 5.18 | 0.79 | 1.21 |
| hi | `bbpb_include_base_44_hindi` | 3.09 | 0.66 | 0.74 |
| hr | `bbpb_global_piqa_parallel_cloze_hrv_latn` | 2.53 | 0.87 | 1.72 |
| hu | `bbpb_include_v2_og_hungarian_hungary` | 2.60 | 0.72 | 0.70 |
| id | `bbpb_include_base_44_indonesian` | 2.17 | 0.61 | 0.55 |
| it | `bbpb_multiblimp_ita` | 2.47 | 0.76 | 0.53 |
| ja | `bbpb_include_v2_og_japanese_japan` | 2.46 | 0.63 | 0.79 |
| ka | `bbpb_belebele_kat_Geor` | 1.71 | 0.75 | 0.65 |
| kk | `bbpb_rf_include_base_44_kazakh` | 4.61 | 0.83 | 1.37 |
| ko | `bbpb_belebele_kor_Hang` | 1.82 | 0.63 | 0.73 |
| lt | `bbpb_include_v2_og_lithuanian_lithuania` | 3.63 | 0.85 | 1.67 |
| lv | `bbpb_rfgm_belebele_lvs_Latn` | 1.59 | 0.98 | 2.09 |
| ml | `bbpb_include_base_44_malayalam` | 2.80 | 0.72 | 0.72 |
| mr | `bbpb_belebele_mar_Deva` | 2.70 | 0.69 | 1.23 |
| ms | `bbpb_include_v2_og_malay_singapore` | 5.90 | 0.67 | 1.65 |
| ne | `bbpb_rf_belebele_npi_Latn` | 2.61 | 0.81 | 1.17 |
| nl | `bbpb_multiblimp_nld` | 2.24 | 0.73 | 0.61 |
| no | `bbpb_global_piqa_parallel_cloze_nob_latn` | 1.49 | 0.63 | 0.89 |
| pl | `bbpb_multiblimp_pol` | 3.32 | 0.70 | 0.49 |
| pt | `bbpb_multiblimp_por` | 3.08 | 0.64 | 0.46 |
| ro | `bbpb_multiblimp_ron` | 2.28 | 0.71 | 0.71 |
| ru | `bbpb_include_base_44_russian` | 2.73 | 0.72 | 0.82 |
| sk | `bbpb_global_piqa_parallel_cloze_slk_latn_sari` | 3.46 | 0.89 | 1.32 |
| sl | `bbpb_global_piqa_parallel_cloze_slv_latn_cerk` | 7.65 | 0.85 | 2.44 |
| sq | `bbpb_include_v2_og_albanian_albania` | 5.81 | 0.85 | 2.37 |
| sr | `bbpb_global_piqa_parallel_cloze_srp_latn` | 3.11 | 0.83 | 1.37 |
| sv | `bbpb_multiblimp_swe` | 2.56 | 0.81 | 1.03 |
| ta | `bbpb_include_base_44_tamil` | 3.05 | 0.72 | 0.64 |
| th | `bbpb_belebele_tha_Thai` | 1.65 | 0.66 | 0.40 |
| tr | `bbpb_multiblimp_tur` | 3.46 | 0.55 | 0.60 |
| uk | `bbpb_belebele_ukr_Cyrl` | 2.27 | 0.73 | 0.18 |
| ur | `bbpb_rf_include_base_44_urdu` | 6.22 | 0.81 | 1.79 |
| vi | `bbpb_include_v2_og_vietnamese_vietnam` | 1.87 | 0.70 | 0.88 |
| zh | `bbpb_paws_zh` | 4.87 | 0.86 | 1.52 |

![Top-5 benchmarks per language by SNR](pretraining/predictivity/top_benchmarks_per_language.png)

**Seed generalization** — holdout `predictivity_seeds_train` → `predictivity_seeds_test` (the ×3 cells only). A variant ranking whose Spearman ρ is low here is noise-dominated; recommend the family that transfers, not the argmax:

| metric | DA-size | DA-ckpt |
|---|---|---|
| Spearman ρ on global variant ranking | -0.28 | 0.82 |
| Pearson r between splits (all cells) | -0.17 | 0.68 |
| Exact-variant agreement (per lang) | 0% | 0% |
| Family-level agreement (per lang) | 0% | 100% |
| Retention of train-best r on test | 43% | 94% |
<!-- END auto:results -->

GitHub: [top_variants_overall.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_variants_overall.png) · [top_variants_overall.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_variants_overall.csv) ·
GitHub: [top_benchmarks_per_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_benchmarks_per_language.png) · [top_benchmarks_per_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_benchmarks_per_language.csv) ·
[snr_variant_ranking.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_ranking.csv) ·
[best_variant_per_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/best_variant_per_language.csv)

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

Numbers from the `predictivity` pool's rq03 table. Regenerate with `python analysis/rq04_surrogates/analyze.py --pool predictivity`. The population at a proxy size is the tasks above chance at the proxy and at 1.7B (rule 1); each candidate is scored on the tasks of it where the candidate has a value, so n differs per candidate and is given next to every ρ. The scaling-fit R² is rq01's log-N fit refitted on the rungs up to the proxy only (rule 11; it needs 3 rungs, so it starts at 350M). `bpb_macro` and `train_loss` are in neither population (rule 7).

- **benchmark tasks** — strongest surrogate of DA-size (mean ρ over proxies): `early-checkpoint agreement (10 %)` 0.42; weakest: `signal alone (relative std)` -0.06.
- **per-language bits per byte** — strongest surrogate of DA-size (mean ρ over proxies): `SNR, dist_std` 0.31; weakest: `early-checkpoint agreement (10 %)` -0.01.

**benchmark tasks** (Spearman ρ of the statistic with DA-size, per proxy size; n = the tasks behind the ρ):

| metric | 90M ρ (n) | 175M ρ (n) | 350M ρ (n) | 600M ρ (n) | 1B ρ (n) |
|---|---|---|---|---|---|
| early-checkpoint agreement (10 %) | 0.25 (298) | 0.64 (1154) | 0.33 (892) | 0.31 (413) | 0.57 (1267) |
| SNR, relative std | 0.42 (298) | 0.36 (1154) | 0.34 (915) | 0.32 (413) | 0.26 (1267) |
| scaling-fit R² (proxy rungs only) |  |  | 0.35 (296) | 0.31 (335) | 0.32 (918) |
| noise alone (relative std, inverted) | 0.22 (298) | 0.31 (1154) | 0.41 (915) | 0.30 (413) | 0.31 (1267) |
| SNR, discrepancy | 0.23 (298) | 0.29 (338) | 0.33 (371) | 0.28 (413) | 0.33 (451) |
| SNR, dist_std | 0.29 (298) | 0.22 (1154) | 0.24 (915) | 0.25 (413) | 0.20 (1267) |
| margin above chance | 0.02 (293) | -0.07 (333) | -0.02 (366) | 0.04 (408) | -0.05 (446) |
| signal alone (relative std) | 0.05 (298) | -0.00 (1154) | -0.10 (915) | -0.13 (413) | -0.10 (1267) |

**per-language bits per byte** (Spearman ρ of the statistic with DA-size, per proxy size; n = the tasks behind the ρ):

| metric | 90M ρ (n) | 175M ρ (n) | 350M ρ (n) | 600M ρ (n) | 1B ρ (n) |
|---|---|---|---|---|---|
| SNR, dist_std | 0.45 (50) | 0.53 (50) | 0.51 (50) | -0.43 (50) | 0.47 (50) |
| SNR, relative std | 0.31 (50) | 0.41 (50) | 0.30 (50) | -0.38 (50) | 0.25 (50) |
| signal alone (relative std) | 0.30 (50) | 0.40 (50) | 0.26 (50) | -0.40 (50) | 0.25 (50) |
| noise alone (relative std, inverted) | -0.02 (50) | -0.03 (50) | 0.01 (50) | 0.41 (50) | -0.02 (50) |
| scaling-fit R² (proxy rungs only) |  |  | 0.31 (50) | -0.64 (50) | 0.34 (50) |
| early-checkpoint agreement (10 %) | 0.55 (50) | 0.53 (50) | -0.13 (50) | -0.47 (50) | -0.56 (50) |

![Surrogates](pretraining/predictivity/rq3_surrogates.png)
<!-- END auto:surrogates -->

GitHub: [rq3_surrogates.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/rq3_surrogates.png) · [rq3_surrogates.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/rq3_surrogates.csv) ·
[facts.json](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/facts.json)

### FineTasks' criteria — setup

A selection rule computed on one size, with no larger model in sight, is the
practical alternative to SNR: FineTasks (Kydlíček, Penedo et al., 2024)
selects tasks per language with four such statistics. `finetasks_criteria.py`
(moved here from rq09 on 2026-09-23; its outputs are
`pretraining/predictivity/finetasks_*`) computes them per (task, size) on the
`predictivity` pool — every design variant at each size at the grid seed (18
of schemes A/B on the tables quoted here; every data build since 2026-10-05),
on the ten evaluated tenths — and judges them by DA-size against the
1.7B reference, which the criteria never see. The gate is not applied (non-random
is one of the criteria under test) but the mask is carried; every candidate is
scored as Spearman ρ with DA-size across tasks, the way the SNR definitions
above are. Their "noise" is the spread across recipes at one step, which this
framework calls signal, so their SNR is the inverse of ours and the two should
not be expected to agree.

<!-- BEGIN auto:finetasks-criteria (finetasks_criteria.py --pool predictivity) -->
## FineTasks' criteria on the ladder

FineTasks selects tasks with four statistics computed on single-seed runs at one size (monotonicity, a cross-run SNR, a non-random margin, consecutive-step ordering). Computed here per (task, size) on the `predictivity` variants and judged by DA-size against 1.7B, which the criteria never see. Columns: share passing each criterion, all three, and our gate; mean DA-size of passers vs failers; Spearman of each statistic with DA-size (monotonicity / SNR / non-random / ordering). Regenerate with `python analysis/rq04_surrogates/finetasks_criteria.py --pool predictivity`.

| size | tasks | monotone | SNR > 20 | non-random | all three | our gate | DA-size pass vs fail | ρ with DA-size |
|---|---|---|---|---|---|---|---|---|
| 90M | 846 | 19% | 62% | 44% | 16% | 37% | 0.58 [135] vs 0.46 [711] | +0.24 / +0.21 / +0.12 / +0.22 |
| 175M | 1662 | 14% | 62% | 24% | 11% | 40% | 0.57 [179] vs 0.56 [1483] | -0.37 / +0.14 / +0.21 / +0.51 |
| 350M | 1367 | 22% | 64% | 30% | 18% | 44% | 0.56 [245] vs 0.54 [1122] | -0.26 / +0.27 / +0.24 / +0.42 |
| 600M | 846 | 44% | 62% | 53% | 36% | 49% | 0.57 [302] vs 0.49 [544] | +0.29 / +0.31 / +0.23 / +0.22 |
| 1B | 1662 | 26% | 62% | 29% | 21% | 54% | 0.59 [356] vs 0.59 [1306] | -0.33 / +0.27 / +0.28 / +0.56 |
| 1.7B | 1662 | 27% | 62% | 31% | 24% | 58% | — | +nan / +nan / +nan / +nan |

![FineTasks criteria](pretraining/predictivity/finetasks_criteria.png)

![DA against each surrogate](pretraining/predictivity/finetasks_surrogates_scatter.png)
<!-- END auto:finetasks-criteria -->

*`finetasks_criteria.png`: (a) the share of tasks passing each criterion and
all three, with our gate; (b) each criterion's Spearman correlation with
DA-size across tasks; (c) DA-size of the tasks passing all three against the
rest; (d) panel (a) with the share of tasks clearing rq02's two DA cuts at that
size (DA-size against the reference ≥ 0.66; median DA-ckpt of the size's own
run ≥ 0.66); (e) panel (b) with rq03's 22 SNR definitions behind it and, for
scale, rq02's own ρ and τ_b of the proxy ranking against the reference's (an
oracle) and rq01's ρ of score with size; (f) the FineTasks picks with a
counterpart in our registry, originals and cloze twins.
`finetasks_surrogates_scatter.png`: DA-size against each surrogate, one point
per (task, proxy size) coloured by the proxy — FineTasks' four criteria, our
gate, rq01's ρ, the three rq03 SNR definitions with the largest |Spearman|
here, rq02's own ρ (the ceiling), and ten further reference-free candidates
(ordering in the noise window, the 10 % and 50 % rankings against the final,
the proxy's own DA-ckpt, DA between the two rungs below the proxy, relative
signal and noise, cross-variant std, margin over chance, item count); Pearson
r, Spearman ρ and the cell count in each corner. Pooled ranking in
`finetasks_surrogates_rank.csv`, the FineTasks picks in `finetasks_overlap.csv`
(folder root).*

**Key findings**

- A selection rule that never sees a larger model recovers about a third of
  the chance-to-reference gap: the composite (monotonicity ≥ 0.5, cross-run
  SNR > 20, best margin > 3 std) passes 17 % of tasks at 90M, 23 % at 175M
  and 43 % at 1B, below our gate's 36 %, 39 % and 53 %, and the passers read
  DA-size 0.53–0.54 at every proxy against 0.45–0.48 for the tasks that
  fail. Their SNR criterion passes 62–63 % of tasks at every size and does
  not discriminate; monotonicity is the size-sensitive one.
- As surrogates of DA-size the four statistics correlate at Spearman
  0.04–0.37 across tasks per proxy, the same range as our 22 SNR
  definitions (`dispersion_shifted` 0.19–0.35, the best at 1B) and as rq01's
  ρ of score with size (0.19–0.33).
- Pooled over the five proxies (`finetasks_surrogates_rank.csv`) the best
  reference-free statistics are the task's item count (0.29), FineTasks' SNR
  (0.28), the proxy's own median DA-ckpt (0.28), `dispersion_shifted` (0.25)
  and the DA between the two rungs below the proxy (0.23); the gate reads
  0.13 and the ordering statistics 0.03–0.10; the cross-variant std of the
  final score, the relative noise and the relative signal correlate
  negatively (−0.25, −0.25, −0.20), so more spread across variants at the
  proxy goes with less agreement with the reference — against the
  framework's premise. The statistics that read the reference are higher:
  rq02's own ρ 0.75 and rq01's trajectory R² over every size 0.35. No
  reference-free statistic exceeds 0.30.
- rq02's two DA cuts (panel d) barely move with size: the DA-size cut passes
  13–15 % of tasks at every proxy, the DA-ckpt cut 17–20 % except at 350M
  (28 %). The DA-ckpt cut passing more tasks than the DA-size one at every
  size is the within-run persistence of
  [rq02 figure 7](../rq02_decision_accuracy/README.md#7-seed-uncertainty-and-the-da-ckpt-null)
  in another guise.
- Of their 96 picks, 55 exist in our registry (35 with results on the
  ladder); at 1B 49 % of the originals pass their own criteria on our ladder
  against 100 % of the 13 cloze twins, with the same mean DA-size (0.53). The format choice they make
  implicitly (cloze for selection) is the one decision that moves the
  population; the thresholds move it little.

**Follow-ups**

- Bootstrap the ρ over tasks (a 90 % interval per statistic): the top
  candidates are within 0.05 of each other, and the seed holdout above says
  the exact ranking does not transfer — claim the family ("relative
  dispersion over checkpoint noise", or here "item count and persistence"),
  not a winner.
- The negative correlation of the cross-variant spread with DA-size deserves
  its own figure: signal at the proxy against the reference's agreement, per
  family, to see whether it is the twins (large spread, re-sorting with
  scale; [rq00 figure 3](../rq00_gate_and_curves/README.md#3-the-reformulated-twins-move-whole-families-across-the-gate))
  or general.
- A per-size and per-population version of the pooled ranking, as the `_b`
  panels do for the SNR definitions: the pooled Spearman mixes BPB with
  benchmarks and small with large proxies.

GitHub: [finetasks_criteria.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_criteria.png) · [finetasks_criteria.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_criteria.csv) ·
GitHub: [finetasks_surrogates_scatter.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_surrogates_scatter.png) · [finetasks_surrogates_scatter.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_surrogates_scatter.csv) ·
[finetasks_surrogates_rank.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_surrogates_rank.csv) ·
[finetasks_overlap.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/finetasks_overlap.csv)

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

The rankings above, without the aggregation (`predictivity` pool); a surrogate subplot needs 8 tasks at a proxy size. Regenerate with `python analysis/rq04_surrogates/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq04 in one figure](pretraining/predictivity/highlights.png)

![SNR definition per language](pretraining/predictivity/snr_definition_by_language.png)

![Surrogates per benchmark](pretraining/predictivity/surrogates_by_benchmark.png)

![Surrogates per language](pretraining/predictivity/surrogates_by_language.png)

**Per language count** (rq02's `da_all_by_L_per_task_multi_axes.csv`: pairs of design variants sharing the L, on the tasks in the languages every variant at the L trains on (the intersection of the L's lists, English always); a level counts when it holds at every larger level with a value; DA ≥ 0.75, an SNR definition tracks DA at ρ ≥ 0.3; an L with fewer than 3 pairs against 1.7B is named in the figure's caption):

![Smallest safe level per measurement and L](pretraining/predictivity/min_level_by_L.png)

![the same as lines](pretraining/predictivity/min_level_by_L_lines.png)

![Smallest size at which an SNR definition tracks DA, per L](pretraining/predictivity/snr_variant_min_size_by_L.png)

![the same as lines](pretraining/predictivity/snr_variant_min_size_by_L_lines.png)

**Version B — every pair pooled, the size axis instead of the language count** (`da_all_pooled_per_task_multi_axes.csv`, ten checkpoints; the population is every design-variant pair of the pool on the tasks of the languages each cell trains (rule 2), parent tasks only (rule 6), gated at the proxy and at 1.7B; the benchmark mean's task count per size differs with the gate (rule 13) and is in each figure's caption and on the DA-at-1C panel):

![Earliest checkpoint per proxy size and DA at 1C](pretraining/predictivity/min_level_by_L_b.png)

![the same as lines](pretraining/predictivity/min_level_by_L_lines_b.png)

![Spearman rho of each SNR definition with DA per proxy size](pretraining/predictivity/snr_variant_min_size_by_L_b.png)

![the same as lines](pretraining/predictivity/snr_variant_min_size_by_L_lines_b.png)

**Version per FLOPs** — every (proxy size, checkpoint) cell at its training compute, the same population as version B:

![DA of every cell against compute](pretraining/predictivity/min_level_by_L_flops.png)

![rho of each SNR definition against compute](pretraining/predictivity/snr_variant_min_size_by_L_flops.png)
<!-- END auto:panels -->

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/highlights.csv) ·
[snr_definition_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_definition_by_language.png) ·
[snr_definition.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_definition.csv) ·
[surrogates_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/surrogates_by_benchmark.png) ·
[surrogates_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/surrogates_by_language.png) ·
[surrogates.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/surrogates.csv) ·
GitHub: [min_level_by_L.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L.png) · [min_level_by_L.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L.csv) ·
GitHub: [min_level_by_L_lines.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_lines.png) · [min_level_by_L_lines.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_lines.csv) ·
GitHub: [snr_variant_min_size_by_L.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L.png) · [snr_variant_min_size_by_L.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L.csv) ·
GitHub: [snr_variant_min_size_by_L_lines.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_lines.png) · [snr_variant_min_size_by_L_lines.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_lines.csv) ·
GitHub: [min_level_by_L_b.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_b.png) · [min_level_by_L_b.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_b.csv) ·
GitHub: [min_level_by_L_lines_b.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_lines_b.png) · [min_level_by_L_lines_b.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_lines_b.csv) ·
GitHub: [snr_variant_min_size_by_L_b.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_b.png) · [snr_variant_min_size_by_L_b.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_b.csv) ·
GitHub: [snr_variant_min_size_by_L_lines_b.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_lines_b.png) · [snr_variant_min_size_by_L_lines_b.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_lines_b.csv) ·
GitHub: [min_level_by_L_flops.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_flops.png) · [min_level_by_L_flops.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_flops.csv) ·
GitHub: [snr_variant_min_size_by_L_flops.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_flops.png) · [snr_variant_min_size_by_L_flops.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_flops.csv)

Two readings of the version-B and per-FLOPs panels carried over from the
2026-09-18 figure review, still true on today's tables: the per-L SNR
correlations (`snr_variant_min_size_by_L`) are "never" for DA-size almost
everywhere because per-L DA is a lattice with ≤ 3 pairs, so the L-dependence
of SNR needs SNR computed over the L's own variants (degenerate below three);
and the per-FLOPs line joins a bigger model's first (peak-LR) checkpoint to a
smaller model's annealed final, which explains the zigzags near 10⁻² — one
line per proxy size with the annealed points marked, plus the compute
frontier (the best DA reachable at or below each compute), is the version to
draw for the paper.

## Surrogates from the literature — summary

`catalogue.py` computes 211 statistics read on the proxy alone, save
`pseudo_ref_da`, which reads the 1B rung below the reference (labelled so): the literature catalogue
([`literature.md`](literature.md); every definition, its source and how our port
differs from it are in the paper's appendix,
`documents/paper/sections/app_snr_new.tex`), every AllenAI signal over every
noise (22 × 6, the k-fold benchmark noise included), and each signal and noise
alone. `search.py` scores them against 30 truths from rq02 (DA / τ-b / ρ for
DA-size, DA-goal and DA-ckpt, both pair sets, every pair or the pairs sharing
one L) on all the data: Spearman over one point per (benchmark, language)
cluster, permutation p below 30 clusters, Benjamini–Hochberg over every
configuration, and a split-half confirmation as an extra. The numbers are in
the generated block below; what they say:

- **The ceiling.** A truth re-read at 90 % keeps ρ = 0.66 (DA-size), 0.81
  (DA-goal) and 0.84 (DA-ckpt) with itself on the benchmarks.
- **No surrogate reaches it.** Over all benchmarks (239–245 clusters) the best
  are rank-stability statistics of the proxy's own run: late consecutive-step
  Kendall τ for DA-ckpt (0.63), sign persistence for DA-goal (0.59) and the
  proxy's mean DA-ckpt for DA-size (0.57) — 73–86 % of the ceiling. (DA-ckpt
  is read before the noise window: at 80 % and 90 % it shares its checkpoints
  with the window statistics, which made window sign consistency look like
  0.73.)
- **SNR is weak here.** AllenAI's `rel_std` / checkpoint noise reads 0.25 on
  DA-size (0.23 for relative dispersion, the paper's headline, against their
  R = 0.79). The k-fold benchmark noise beats the checkpoint noise for every
  signal (best ratios 0.48 / 0.51 / 0.60 on DA-size / goal / ckpt), but it is
  mostly the item count: `n_items` alone reads 0.51 on DA-size, and the best
  DA-size ratio, `dispersion_shifted / kfold_abs`, is the inverse noise
  (that signal is constant).
- **Dispersion alone points the wrong way.** Range, standard deviation and
  pairwise distance across variants correlate negatively with every truth
  (−0.16 to −0.31).
- **Particular cases** reach 0.93–1.00 on single languages or benchmarks
  (10–19 clusters) and on rq02's reliable subsets; the latter are selected on
  the truth, so their ρ is conditional on it.
- **Language count.** Over pairs of variants sharing L (L8, L15, L30) mean
  DA-size is 0.51 / 0.48 / 0.48: no trend, on a coarse test.
- **Split-half check.** 627 of 1,861 configurations chosen on one half hold on
  the other at q < 0.05.

<!-- BEGIN auto:catalogue (search.py --pool predictivity) -->
## A catalogue of surrogates beyond SNR

Numbers from the `predictivity` pool. Regenerate with `python analysis/rq04_surrogates/catalogue.py --pool predictivity` then `search.py --pool predictivity`. Surrogates are read on the proxy alone (rule 11), except `pseudo_ref_da`, which reads the 1B rung; the truths are rq02's DA (DA-size; DA-goal and DA-ckpt at every early checkpoint of the proxy; both pair sets; every pair or the pairs sharing L) and Kendall τ-b / Spearman ρ on the same rankings, over cells above chance (rule 1). ρ is a Spearman over one point per (benchmark, language) cluster, pooled over proxies and checkpoints; populations differ per configuration (rule 13) and n is in the CSVs. Definitions and sources: [`literature.md`](literature.md); method: `search.py`'s docstring.

**Main analysis (all the data).** 745,992 configurations (truths × subsets × surrogates, the threshold filters included); 438,471 have a p-value (≥ 10 units); **100,559 hold at BH q < 0.05**. A best-of-many ρ is optimistic even when it is significant; the extra below measures by how much.

**The ceiling** — each truth against itself re-read at 90 % (benchmarks):

| truth | ρ | ρ over cells | units |
|---|---|---|---|
| DA-size da (multi-axis) | 0.90 | 0.80 | 420 |
| DA-goal da (multi-axis) | 0.94 | 0.80 | 420 |
| DA-ckpt da (multi-axis) | 0.97 | 0.86 | 420 |
| DA-size tau_b (multi-axis) | 0.90 | 0.79 | 420 |
| DA-size rho (multi-axis) | 0.91 | 0.81 | 420 |
| DA-goal tau_b (multi-axis) | 0.93 | 0.79 | 420 |
| DA-goal rho (multi-axis) | 0.94 | 0.80 | 420 |
| DA-ckpt tau_b (multi-axis) | 0.97 | 0.85 | 420 |
| DA-ckpt rho (multi-axis) | 0.97 | 0.86 | 420 |

**Strongest surrogate per truth**, every benchmark:

| truth | surrogate | ρ | q | ρ over cells | ρ within stratum | units |
|---|---|---|---|---|---|---|
| DA-ckpt da (mono-axis, pairs within L1) | `snr__rel_dispersion__kfold_rel` | 0.77 | 3.9e-03 | 0.36 | 0.37 | 17 |
| DA-ckpt da (mono-axis, pairs within L15) | `consecutive_kendall` | 0.60 | 4.3e-08 | 0.27 | 0.22 | 81 |
| DA-ckpt da (mono-axis, pairs within L2) | `rung_sign_consistency` | 0.83 | 3.8e-04 | 0.20 | 0.26 | 16 |
| DA-ckpt da (mono-axis, pairs within L30) | `sign_consistency_window` | 0.55 | 1.3e-21 | 0.27 | 0.24 | 274 |
| DA-ckpt da (mono-axis, pairs within L50) | `consecutive_kendall` | 0.80 | 1.1e-89 | 0.36 | 0.33 | 420 |
| DA-ckpt da (mono-axis, pairs within L8) | `sign_consistency_window` | 0.54 | 7.7e-06 | 0.22 | 0.19 | 74 |
| DA-ckpt da (mono-axis) | `consecutive_kendall` | 0.86 | 1.7e-118 | 0.48 | 0.47 | 420 |
| DA-ckpt da (multi-axis, pairs within L1) | `item_total_corr` | 0.81 | 3.8e-04 | 0.26 | 0.24 | 24 |
| DA-ckpt da (multi-axis, pairs within L15) | `snr__gini__kfold_rel` | 0.57 | 4.1e-12 | 0.22 | 0.22 | 143 |
| DA-ckpt da (multi-axis, pairs within L2) | `rung_sign_consistency` | 0.80 | 1.9e-03 | 0.28 | 0.32 | 16 |
| DA-ckpt da (multi-axis, pairs within L30) | `consecutive_kendall` | 0.70 | 3.0e-42 | 0.40 | 0.37 | 298 |
| DA-ckpt da (multi-axis, pairs within L50) | `consecutive_kendall` | 0.86 | 4.0e-118 | 0.46 | 0.44 | 420 |
| DA-ckpt da (multi-axis, pairs within L8) | `sign_consistency_window` | 0.59 | 4.4e-07 | 0.24 | 0.21 | 74 |
| DA-ckpt da (multi-axis) | `consecutive_kendall` | 0.94 | 1.4e-187 | 0.66 | 0.65 | 420 |
| DA-ckpt rho (multi-axis) | `consecutive_kendall` | 0.94 | 1.2e-187 | 0.64 | 0.64 | 420 |
| DA-ckpt tau_b (multi-axis) | `consecutive_kendall` | 0.94 | 1.9e-186 | 0.62 | 0.62 | 420 |
| DA-goal da (mono-axis, pairs within L1) | `prev_rung_da` | 0.86 | 7.1e-04 | 0.39 | 0.34 | 16 |
| DA-goal da (mono-axis, pairs within L15) | `tie_rate_items` | -0.26 | 1.5e-01 | -0.02 | -0.02 | 53 |
| DA-goal da (mono-axis, pairs within L2) | `snr__star_discrepancy_shifted__ckpt_abs` | 0.77 | 3.6e-03 | 0.27 | 0.27 | 16 |
| DA-goal da (mono-axis, pairs within L30) | `scale_gain_over_noise` | 0.39 | 4.8e-08 | 0.15 | 0.14 | 220 |
| DA-goal da (mono-axis, pairs within L50) | `consecutive_kendall` | 0.67 | 2.0e-52 | 0.31 | 0.30 | 420 |
| DA-goal da (mono-axis, pairs within L8) | `snr__iqr__kfold_abs` | 0.44 | 6.6e-03 | 0.09 | 0.08 | 50 |
| DA-goal da (mono-axis) | `consecutive_kendall` | 0.64 | 4.2e-48 | 0.36 | 0.35 | 420 |
| DA-goal da (multi-axis, pairs within L1) | `n_items` | 0.75 | 1.0e-02 | 0.47 | 0.47 | 15 |
| DA-goal da (multi-axis, pairs within L15) | `item_total_corr` | 0.52 | 6.2e-13 | 0.18 | 0.18 | 185 |
| DA-goal da (multi-axis, pairs within L2) | `snr__star_discrepancy_shifted__tukey_depth` | 0.77 | 3.4e-03 | 0.27 | 0.28 | 16 |
| DA-goal da (multi-axis, pairs within L30) | `sign_persistence` | 0.57 | 3.3e-25 | 0.29 | 0.28 | 298 |
| DA-goal da (multi-axis, pairs within L50) | `consecutive_kendall` | 0.75 | 3.9e-73 | 0.41 | 0.39 | 420 |
| DA-goal da (multi-axis, pairs within L8) | `snr__iqr__kfold_abs` | 0.44 | 7.5e-03 | 0.09 | 0.08 | 50 |
| DA-goal da (multi-axis) | `consecutive_kendall` | 0.82 | 5.7e-98 | 0.55 | 0.54 | 420 |
| DA-goal rho (multi-axis) | `consecutive_kendall` | 0.81 | 8.5e-96 | 0.52 | 0.52 | 420 |
| DA-goal tau_b (multi-axis) | `consecutive_kendall` | 0.80 | 1.7e-90 | 0.51 | 0.51 | 420 |
| DA-size da (mono-axis, pairs within L1) | `prev_rung_da` | 0.79 | 3.4e-03 | 0.43 | 0.35 | 16 |
| DA-size da (mono-axis, pairs within L15) | `emerged_share` | -0.32 | 5.9e-02 | -0.04 | -0.04 | 53 |
| DA-size da (mono-axis, pairs within L2) | `snr__rel_star_discrepancy__ckpt_abs` | 0.79 | 3.2e-03 | 0.32 | 0.33 | 16 |
| DA-size da (mono-axis, pairs within L30) | `pseudo_ref_da` | 0.26 | 1.4e-04 | 0.10 | 0.10 | 274 |
| DA-size da (mono-axis, pairs within L50) | `consecutive_kendall` | 0.60 | 8.1e-41 | 0.33 | 0.33 | 420 |
| DA-size da (mono-axis, pairs within L8) | `snr__iqr__kfold_abs` | 0.33 | 6.6e-02 | 0.10 | 0.09 | 50 |
| DA-size da (mono-axis) | `bpb_rank_agreement` | 0.59 | 8.3e-38 | 0.38 | 0.38 | 420 |
| DA-size da (multi-axis, pairs within L1) | `snr__dispersion__kfold_abs` | 0.76 | 6.2e-03 | 0.47 | 0.50 | 16 |
| DA-size da (multi-axis, pairs within L15) | `item_total_corr` | 0.51 | 3.3e-12 | 0.17 | 0.18 | 185 |
| DA-size da (multi-axis, pairs within L2) | `snr__rel_star_discrepancy__ckpt_abs` | 0.79 | 2.1e-03 | 0.34 | 0.35 | 16 |
| DA-size da (multi-axis, pairs within L30) | `pseudo_ref_da` | 0.48 | 2.9e-17 | 0.32 | 0.31 | 298 |
| DA-size da (multi-axis, pairs within L50) | `consecutive_kendall` | 0.70 | 1.4e-59 | 0.42 | 0.40 | 420 |
| DA-size da (multi-axis, pairs within L8) | `snr__mad__kfold_abs` | 0.27 | 1.5e-01 | 0.05 | 0.04 | 50 |
| DA-size da (multi-axis) | `da_ckpt_mean` | 0.76 | 4.3e-76 | 0.58 | 0.58 | 420 |
| DA-size rho (multi-axis) | `bpb_rank_agreement` | 0.76 | 9.5e-78 | 0.56 | 0.56 | 420 |
| DA-size tau_b (multi-axis) | `bpb_rank_agreement` | 0.74 | 1.0e-71 | 0.55 | 0.55 | 420 |

**The AllenAI grid** (22 signals × 6 noises, DA, every pair, benchmarks): best combination against AllenAI's own `rel_std` / checkpoint noise:

| truth | best signal / noise | ρ | rel_std / ckpt_rel ρ |
|---|---|---|---|
| DA-ckpt | `star_discrepancy_shifted / kfold_rel` | 0.67 | 0.57 |
| DA-goal | `aad / ckpt_abs` | 0.56 | 0.54 |
| DA-size | `aad / ckpt_abs` | 0.52 | 0.50 |

**Particular cases** — the strongest significant configurations outside "every benchmark", two per (truth, subset type); `reliable (on the truth)` subsets are selected on the truth itself, so their ρ is conditional on it; all of them in `surrogate_correlations.csv` (`q` column):

| truth | subset | surrogate | ρ | q | ρ over cells | units |
|---|---|---|---|---|---|---|
| DA-ckpt da (multi-axis) | language: ru | `window_kendall` | 0.99 | 3.8e-04 | 0.80 | 13 |
| DA-ckpt da (multi-axis) | language: ru | `kendall_w_window` | 0.99 | 3.8e-04 | 0.82 | 13 |
| DA-goal da (multi-axis, pairs within L15) | language: vi | `cronbach_alpha` | 0.99 | 3.8e-04 | 0.40 | 12 |
| DA-size tau_b (multi-axis) | language: ar | `late_slope_to_noise` | 0.98 | 3.8e-04 | 0.41 | 13 |
| DA-goal da (multi-axis) | language: ar | `late_slope_to_noise` | 0.98 | 3.8e-04 | 0.53 | 13 |
| DA-ckpt tau_b (multi-axis) | language: zh | `consecutive_kendall_late` | 0.98 | 3.8e-04 | 0.74 | 15 |
| DA-ckpt rho (multi-axis) | language: zh | `consecutive_kendall_late` | 0.98 | 3.8e-04 | 0.73 | 15 |
| DA-ckpt tau_b (multi-axis) | language: zh | `consecutive_kendall` | 0.97 | 3.8e-04 | 0.77 | 15 |
| DA-ckpt rho (multi-axis) | language: zh | `consecutive_kendall` | 0.97 | 3.8e-04 | 0.76 | 15 |
| DA-size da (multi-axis) | language: ar | `late_slope_to_noise` | 0.97 | 3.8e-04 | 0.48 | 13 |
| DA-goal tau_b (multi-axis) | language: ar | `late_slope_to_noise` | 0.97 | 3.8e-04 | 0.48 | 13 |
| DA-size da (multi-axis, pairs within L15) | language: vi | `cronbach_alpha` | 0.97 | 7.1e-04 | 0.39 | 12 |
| DA-ckpt rho (multi-axis) | benchmark: multiblimp | `consecutive_kendall` | 0.96 | 2.6e-18 | 0.67 | 34 |
| DA-goal rho (multi-axis) | language: ar | `late_slope_to_noise` | 0.96 | 3.8e-04 | 0.49 | 13 |
| DA-goal rho (multi-axis) | language: vi | `cronbach_alpha` | 0.96 | 3.8e-04 | 0.58 | 12 |
| DA-size rho (multi-axis) | language: ar | `late_slope_to_noise` | 0.96 | 3.8e-04 | 0.44 | 13 |
| DA-goal tau_b (multi-axis) | benchmark: hellaswag | `da_ckpt_mean` | 0.95 | 3.8e-04 | 0.58 | 25 |
| DA-goal tau_b (multi-axis) | benchmark: hellaswag | `sign_persistence` | 0.95 | 3.8e-04 | 0.58 | 25 |
| DA-size tau_b (multi-axis) | language: es | `bpb_rank_agreement` | 0.95 | 3.8e-04 | 0.50 | 17 |
| DA-goal rho (multi-axis) | benchmark: hellaswag | `sign_persistence` | 0.95 | 3.8e-04 | 0.58 | 25 |
| DA-goal rho (multi-axis) | benchmark: hellaswag | `da_ckpt_mean` | 0.95 | 3.8e-04 | 0.58 | 25 |
| DA-size rho (multi-axis) | language: pt | `settling_time` | -0.95 | 3.8e-04 | -0.40 | 11 |
| DA-ckpt rho (multi-axis) | tier: L50 | `consecutive_kendall` | 0.95 | 1.8e-57 | 0.60 | 122 |
| DA-goal tau_b (multi-axis) | language: ru | `settling_time` | -0.95 | 7.1e-04 | -0.63 | 13 |
| DA-ckpt tau_b (multi-axis) | benchmark: multiblimp | `consecutive_kendall` | 0.94 | 1.7e-15 | 0.63 | 34 |
| DA-ckpt da (multi-axis) | benchmark: multiblimp | `consecutive_kendall` | 0.94 | 2.6e-15 | 0.69 | 34 |
| DA-ckpt tau_b (multi-axis) | tier: L8 | `consecutive_kendall` | 0.94 | 1.7e-52 | 0.63 | 116 |
| DA-ckpt tau_b (multi-axis) | tier: L50 | `consecutive_kendall` | 0.94 | 3.4e-55 | 0.58 | 122 |
| DA-ckpt da (multi-axis) | benchmark: multiblimp | `prev_rung_da` | 0.94 | 6.3e-15 | 0.59 | 34 |
| DA-ckpt tau_b (multi-axis) | benchmark: arc_mt | `snr__aad__tukey_depth` | 0.94 | 1.0e-03 | 0.49 | 10 |

**Per language count** (pairs of variants sharing L; benchmarks):

| truth | L | mean DA | sd | tasks | best surrogate | ρ |
|---|---|---|---|---|---|---|
| DA-ckpt | 1 | 0.69 | 0.26 | 138 | `item_total_corr` | 0.81 |
| DA-ckpt | 2 | 0.61 | 0.26 | 138 | `rung_sign_consistency` | 0.80 |
| DA-ckpt | 8 | 0.57 | 0.25 | 263 | `sign_consistency_window` | 0.59 |
| DA-ckpt | 15 | 0.62 | 0.28 | 678 | `snr__gini__kfold_rel` | 0.57 |
| DA-ckpt | 30 | 0.62 | 0.22 | 967 | `consecutive_kendall` | 0.70 |
| DA-ckpt | 50 | 0.64 | 0.26 | 1295 | `consecutive_kendall` | 0.86 |
| DA-goal | 1 | 0.60 | 0.28 | 129 | `n_items` | 0.75 |
| DA-goal | 2 | 0.51 | 0.25 | 129 | `snr__star_discrepancy_shifted__tukey_depth` | 0.77 |
| DA-goal | 8 | 0.49 | 0.24 | 250 | `snr__iqr__kfold_abs` | 0.44 |
| DA-goal | 15 | 0.56 | 0.28 | 659 | `item_total_corr` | 0.52 |
| DA-goal | 30 | 0.54 | 0.22 | 947 | `sign_persistence` | 0.57 |
| DA-goal | 50 | 0.57 | 0.26 | 1269 | `consecutive_kendall` | 0.75 |
| DA-size | 1 | 0.62 | 0.26 | 129 | `snr__dispersion__kfold_abs` | 0.76 |
| DA-size | 2 | 0.49 | 0.26 | 129 | `snr__rel_star_discrepancy__ckpt_abs` | 0.79 |
| DA-size | 8 | 0.51 | 0.25 | 250 | `snr__mad__kfold_abs` | 0.27 |
| DA-size | 15 | 0.55 | 0.29 | 659 | `item_total_corr` | 0.51 |
| DA-size | 30 | 0.54 | 0.23 | 947 | `pseudo_ref_da` | 0.48 |
| DA-size | 50 | 0.58 | 0.27 | 1269 | `consecutive_kendall` | 0.70 |

**Extra: held-out confirmation.** The (benchmark, language) clusters split in two halves; 5319 configurations ranked best on one half (within-stratum ρ) were tested once on the other (one-sided cluster permutation test, BH): **1627 hold at q < 0.05**. The strongest:

| truth | subset | surrogate | ρ disc. | ρ val. [90 %] | q | tasks val. |
|---|---|---|---|---|---|---|
| DA-ckpt da (multi-axis) | proxy: 175M | `consecutive_kendall` | 0.77 | 0.76 [0.73, 0.78] | 2.7e-03 | 500 |
| DA-ckpt rho (multi-axis) | proxy: 175M | `consecutive_kendall` | 0.75 | 0.75 [0.72, 0.77] | 2.7e-03 | 500 |
| DA-ckpt da (multi-axis) | proxy: 175M | `crossings` | -0.75 | -0.74 [-0.77, -0.72] | 2.7e-03 | 500 |
| DA-ckpt rho (multi-axis) | proxy: 175M | `crossings` | -0.74 | -0.74 [-0.76, -0.72] | 2.7e-03 | 500 |
| DA-ckpt tau_b (multi-axis) | proxy: 175M | `consecutive_kendall` | 0.74 | 0.73 [0.71, 0.76] | 2.7e-03 | 500 |
| DA-size rho (multi-axis) | benchmark: hellaswag | `pseudo_ref_da` | 0.86 | 0.73 [0.51, 0.89] | 2.7e-03 | 12 |
| DA-ckpt tau_b (multi-axis) | proxy: 175M | `crossings` | -0.73 | -0.73 [-0.75, -0.70] | 2.7e-03 | 500 |
| DA-size da (multi-axis) | benchmark: multiblimp | `sign_persistence` | 0.73 | 0.72 [0.52, 0.84] | 2.7e-03 | 17 |
| DA-size da (multi-axis) | benchmark: multiblimp | `da_ckpt_mean` | 0.73 | 0.72 [0.52, 0.84] | 2.7e-03 | 17 |
| DA-size da (mono-axis) | benchmark: hellaswag | `pseudo_ref_da` | 0.72 | 0.72 [0.47, 0.87] | 2.7e-03 | 12 |
| DA-ckpt da (multi-axis) | proxy: 175M | `consecutive_kendall_late` | 0.73 | 0.71 [0.68, 0.74] | 2.7e-03 | 500 |
| DA-ckpt da (multi-axis) | tier: L8 | `consecutive_kendall` | 0.69 | 0.71 [0.66, 0.74] | 2.7e-03 | 153 |
| DA-size tau_b (multi-axis) | benchmark: hellaswag | `pseudo_ref_da` | 0.88 | 0.71 [0.49, 0.87] | 2.7e-03 | 12 |
| DA-ckpt rho (multi-axis) | proxy: 175M | `consecutive_kendall_late` | 0.71 | 0.70 [0.67, 0.73] | 2.7e-03 | 500 |
| DA-size da (multi-axis) | benchmark: hellaswag | `pseudo_ref_da` | 0.88 | 0.70 [0.46, 0.86] | 2.7e-03 | 12 |

![surrogates_significant](pretraining/predictivity/surrogates_significant.png)

![surrogates_snr_grid](pretraining/predictivity/surrogates_snr_grid.png)

![surrogates_catalogue](pretraining/predictivity/surrogates_catalogue.png)

![surrogates_by_L](pretraining/predictivity/surrogates_by_L.png)

![surrogates_catalogue_by_language](pretraining/predictivity/surrogates_catalogue_by_language.png)

![surrogates_validated](pretraining/predictivity/surrogates_validated.png)
<!-- END auto:catalogue -->

## TODO

- [ ] **The seed holdout ranks most tasks on a single model pair (open, 2026-09-17).**
      *What it is.* rq04 claims one SNR definition tracks decision accuracy
      best; the holdout (`rq03_noise_and_snr/compare_seed_splits.py`, reported
      in rq04's highlight and "Seed generalization" table) asks whether that
      ranking of definitions survives a change of seed: it is computed on
      `predictivity_seeds_train` (the replicate seeds 64, 313, 28, 1797) and again on
      `predictivity_seeds_test` (seed 1904 of the same cells) and the two are
      compared.
      *The problem.* (2026-10-05: the train split is now every replicate
      seed, so the 1B ×3 cells, seeds 28/1797, join both splits and the
      text below predates that.) Replicate seeds exist only at 175M and 600M, so both
      splits hold those two sizes and DA-size there is 175M → 600M (the 1.7B
      reference never enters). English tasks and BPB rest on 15 model pairs.
      A non-English benchmark is trained only in the L50 cell, so its train
      split holds one pair — the two seeds of the same cell — and its
      "decision accuracy" is 0 or 1 and measures seed noise, not a ranking.
      The median task has one pair.
      *Implications.* The per-language agreement numbers (29 % / 57 %) and the
      low DA-ckpt ρ (0.07) are dominated by those one-pair tasks, so "the
      ranking does not survive a seed swap" may say more about the holdout
      than about the definitions; the DA-size ρ (0.75) leans on English + BPB.
      Nothing in the main `predictivity` tables is affected.
      *Options.* (A) keep only tasks with ≥ 6 pairs on both splits: an honest
      check, but English + BPB only. (B) treat replicate seeds as replicates,
      not as models: average them before ranking, and use the holdout only
      for the noise estimate — changes what the holdout means. (C) report the
      holdout for DA-ckpt only, whose pairs are checkpoints × variants within
      a size and do not shrink to one. (D) drop the holdout numbers from the
      highlight until more sizes have replicate seeds.
      *Recommendation.* A now (small change, honest scope, say "English and
      BPB" in the highlight), and revisit once the 90M / 1.7B-L15 cells and
      any further replicate seeds exist. Not implemented yet.
- [ ] Bootstrap CIs on per-language Pearson r and cross-pool Spearman ρ.
- [ ] Recommend a *family* (dispersion / relative-spread), not an exact variant
      — only the family transfers across seeds.
- [ ] Use a larger DA-size target (e.g. Apertus-8B) instead of the
      not-fully-converged 1B custom model.

## Extensions from other sweeps

Everything below comes from the **36-model sweep** (2026-04…06, 4 sizes × 3
data mixtures × 3 seeds, pools `seeds_1904`, `seeds_28_1797`,
`seeds_28_1797_1904`, `custom_swissai_hf`; 12 languages, reference **1B**)
or from the **external tier** (`all/external`: the public and reference
models, 270M–70B, no data-mixture axis). Its harness, task set, checkpoint
window and reference differ from the ladder's, and its 36-sweep DA-size is
a 3-mixture ranking against 1B, so its numbers are replications of the
family-level recommendation above, never rows of the same table. The dated
comparison note that once tracked these tiers (`ANALYSIS_new_vs_previous.md`, removed 2026-09-23:
`rel_mpd` on top in every tier, DA-ckpt r 0.379 → 0.519 with the externals
folded in, holdout ρ 0.80 / 0.93) predates the tie fix in `decision_acc_fast`
and the parent-only filter and is superseded by the blocks below.

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

[top_variants_overall.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/all/external/top_variants_overall.png) ·
[top_benchmarks_per_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/all/external/top_benchmarks_per_language.png)

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

[top_variants_overall.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/custom_swissai_hf/top_variants_overall.png) ·
[top_benchmarks_per_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/custom_swissai_hf/top_benchmarks_per_language.png)

## Files

- `pretraining/<pool>/snr_variant_ranking.csv` — per-(variant, DA-def, scope)
  Pearson r (`analyze_snr_variants.py`).
- `…/top_variants_overall.csv`, `best_variant_per_language.csv`,
  `best_variant_family_per_language.csv`, `variant_clusters.csv`,
  `top_benchmarks_per_language.csv` — the variant ranking and the per-language
  anchor (`snr_definition_postprocess.py`).
- `…/top_variants_overall.png`, `best_variant_per_language.png`,
  `best_variant_family_per_language.png`, `top_benchmarks_per_language.png`,
  `variant_correlation_matrix.png`, `da_size_vs_da_ckpt_multi_axes.png`,
  `{da_size,da_ckpt}/…` — supporting figures.
- `…/rq3_surrogates.csv`, `rq3_surrogates.png/.pdf`, `facts.json` — the
  statistics beyond SNR (`analyze.py`; the paper's RQ3 figure).
- `…/surrogate_values.csv`, `surrogate_targets.csv`, `surrogate_definitions.csv` — the
  surrogates per (task, proxy, pair set) and the truths, long (`catalogue.py`; sources in
  `literature.md`); `surrogate_correlations.csv`, `surrogate_filters.csv`,
  `surrogate_validated.csv`, `da_all_retest_multi_axes.csv` and
  `surrogates_{validated,snr_grid,catalogue,by_L,catalogue_by_language}.png` — the
  validated search (`search.py`).
- Inputs: rq03's `snr_variants_per_task.csv` and holdout
  `headline_metrics.csv`, rq00's `above_random_scores.csv`, rq01's `rq1_fits.csv`.
- `…/finetasks_criteria.{csv,png}`, `finetasks_surrogates_scatter.{csv,png}`,
  `finetasks_surrogates_rank.csv`, `../finetasks_overlap.csv` — FineTasks'
  criteria on the ladder and every surrogate against DA-size
  (`finetasks_criteria.py`; moved from rq09 on 2026-09-23, the copies left
  under `rq09_benchmark_design/pretraining/predictivity/` are stale).
