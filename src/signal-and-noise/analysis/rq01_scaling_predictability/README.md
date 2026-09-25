# RQ1 — What scales predictably? (paper RQ1)

## Research question

> Which benchmarks, and which per-language measurements, move with model size
> in a way a log-linear fit captures, so that a small-model measurement has a
> trend to extrapolate from at all? The paper's RQ1
> ([`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex)).
> The folder also holds the loss scaling fit and, in `scaling_law_error.py`,
> the prediction test: how well a power law fitted on the proxy rungs
> predicts the reference rung's per-language BPB.

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_all) -->
## Highlighted result

- **1156 (task, L) fits over 410 tasks**, each on the rungs where the task is above chance (rule 1, rq00's mask): the gate removed 510 rungs from the fitted series and left 1276 (task, L) series with fewer than 3 rungs (no fit, `gated` in `rq1_fits.csv`), 417 tasks without any fit. Best-scaling families (median R²): `lambada_openai_mt` 0.99, `arc` 0.99, `xstorycloze` 0.99, `rf_mmlu` 0.98; worst: `truthfulqa_mc2` 0.60, `include_base_44` 0.53, `cultural_bench_easy` 0.21.
- **Median R² by answer count** (over the gated benchmark fits): 0.92 over the 2-option fits, 0.88 over the 3-option fits, 0.94 over the 4-option fits, 0.93 over the 5-option fits, 0.94 over the 6-option fits, 0.68 over the 7-option fits, 0.77 over the 8-option fits.
- **Loss exponent α per (L, arch, scheme)**, the seed-1904 cells, sizes in the fit in brackets: L1 deep/A 0.125 (5), L1 shallow/A 0.127 (5), L2 deep/A 0.144 (5), L2 deep/ES 0.153 (4), L2 deep/ZH 0.138 (5), L2 shallow/A 0.166 (5), L8 deep/A 0.148 (5), L8 deep/B 0.143 (5), L8 shallow/A 0.158 (5), L8 shallow/B 0.154 (5), L15 deep/A 0.147 (5), L15 deep/AT3 0.144 (5), L15 deep/B 0.153 (5), L15 shallow/A 0.154 (5), L15 shallow/B 0.153 (5), L30 deep/A 0.141 (5), L30 deep/AT3 0.157 (5), L30 deep/B 0.149 (5), L30 shallow/A 0.161 (5), L30 shallow/B 0.155 (5), L50 deep/A 0.145 (5), L50 deep/AT3 0.160 (5), L50 shallow/A 0.154 (5), L50 shallow/AT3 0.158 (5).
<!-- END auto:highlight -->

## Setup

Pool `predictivity_all` (every seed and scheme; the fits use the deep,
scheme-A, seed-1904 cells), sizes 175M–1.7B, the rq00 gate per (task, size)
(`predictivity` mask), trained languages and parent tasks only; no design
pairs are involved, so no pair set. Hand-written numbers are from the
ladder-report snapshot **2026-09-23 06:16**.

## Experimental setup

The fits read the plan grid — deep, scheme A, seed 1904 — at each cell's
final checkpoint, from 175M up (the 90M rung diverged and is dropped at load,
[`plan/90M-rung-anomaly.md`](../../../../plan/90M-rung-anomaly.md)); the
loader keeps parent tasks and trained languages only
([`RULES.md`](../RULES.md), rules 6 and 2). Every task is one series per
language setting L; per-language BPB and the training loss enter as tasks too
(they fall with size, so ρ = −1 is their ideal). The above-random gate (rule 1)
applies per (task, size): a rung enters a task's fit only where rq00's mask
says the task is above chance there; tasks without a chance level (BPB, the
loss, generative tasks) are never gated. The scaling-law error uses the deep,
scheme-A, seed-1904 cells at every L on the languages the cell trains, with
the 1.7B rung as the only reference (rule 9); the loss scaling fit the
seed-1904 cell of every (L, arch, scheme) the pool holds.

## Methodology

- **Log-N fit.** Per (task, L): score = a + b·log₁₀ N over the rungs where
  the task is above chance (≥ 3 of them; `n_rungs` and `gated_rungs` in the
  table, and `gated` where the gate left fewer than 3 — the row stays, with
  NaN statistics, and is grey in the panels), with R² and the Spearman ρ
  between score and N; medians per family over its fits (≥ 3), the family's
  modal option count alongside (`rq1_fits.csv`, `rq1_families.csv`).
- **Loss scaling fit.** Final training loss of the seed-1904 cells against N
  per (L, arch, scheme), `pretrain.ladder_report._fit` over every rung present
  (the count per fit is in the legend and the README bullet) — the health
  check's law, but with every rung in the fit rather than the smallest
  predicted.
- **Scaling-law error.** Per (L, arch, scheme, language), log BPB = a − α log N
  is fitted on the proxy rungs up to a ladder top (≥ 3 points, the same
  `_fit`) and predicts the 1.7B BPB; a chain without a 1.7B final is dropped
  and named, never referenced at a smaller size. The relative error is signed
  ((predicted − observed) / observed, negative = the law under-predicts BPB)
  and reported per ladder top, so the table reads "how far up the ladder must
  one train before the reference is predicted within x %". A constant offset
  between small and large models shows up here but not in decision accuracy
  (rq05), so the two reads can disagree.

## Figures, in storyline order

### 1. Scaling regimes: the paper figure

*`predictivity_all`, deep scheme-A seed-1904 cells, gated log-N fits
(`rq1_fits.csv`) and trajectory fits over each run's saved checkpoints; one
point per task with a median over ≥ 2 language settings.* The paper's RQ1
figure is `scaling_regimes_outliers_paper` (copied by
`documents/paper/figures/make_rq_figures.py`), its appendix twin
`scaling_regimes_by_family_paper`.

<!-- BEGIN auto:regimes (regimes.py --pool predictivity_all) -->
## Scaling regimes per benchmark-language pair

`regimes.py`: one point per task, medians over the deep scheme-A seed-1904 cells (parent tasks, trained languages) — the R² and (oriented) Spearman ρ of the gated log-N fits of `rq1_fits.csv`, one per L, and the R² of the training-trajectory fit (score ~ log tokens over a run's checkpoints, ≥ 5 points), one per (L, size). Both use only the sizes where the task is above chance (rule 1, rq00's mask): 306 tasks have a point; the gate removed 417 tasks that are at chance at every size where a fit was possible (per family: acp_bench_cloze 7, acp_bench_mcq 7, arc 26, bbh_cloze 6, bbh_mcq 17, belebele 59, blend_sample 5, commonsense_qa 1, cultural_bench_easy 18, cultural_bench_hard 19, global_mmlu_full 29, global_piqa_nonparallel_cloze 1, global_piqa_parallel_cloze 63, hellaswag 2, include_base_44 33, include_v2_en 16, include_v2_og 32, mmlu 1, openbookqa 1, paws 5, rf_acp_bench_mcq 2, rf_bbh_mcq 8, rf_belebele 4, rf_cultural_bench_easy 13, rf_global_mmlu_full 4, rf_include_base_44 17, rfgm_include_base_44 14, toxigen 1, truthfulqa-multi_mc1 2, xcopa 1, xnli 3); the training loss is left out (rule 7). The quadrants of (b) split at R² = 0.5, a heuristic; a task with median ρ < 0 is the fifth regime `declines with size` whatever its quadrant (6 tasks, black edge in panel (b)). Table: `scaling_regimes.csv`. Regenerate with `python analysis/rq01_scaling_predictability/regimes.py --pool predictivity_all`.

![Scaling regimes](pretraining/predictivity_all/scaling_regimes.png)

Named variants of the same points: `scaling_regimes_families.png` (one label per family at its median point, `scaling_regimes_families.csv`), `scaling_regimes_outliers.png` (plus the tasks in another quadrant than their family's majority and > 0.25 from its median point, `scaling_regimes_outliers.csv`; `scaling_regimes_outliers_paper.png/.pdf/.svg` is its bare, square-panel version for the paper, the label text pulled 45% towards the ink), `scaling_regimes_by_family.png` (panel (b) per family, tasks named by language, its per-task table with the labels next to it; `_paper.png/.pdf/.svg/.csv` is its bare version for the paper's appendix) and `scaling_regimes.html` (hover names, click-to-highlight legend; for the project site).

![Scaling regimes, outliers named](pretraining/predictivity_all/scaling_regimes_outliers.png)

![Scaling regimes per family](pretraining/predictivity_all/scaling_regimes_by_family.png)
<!-- END auto:regimes -->

**Key findings**

- Scaling is predictable on the tasks that survive the gate: of the 306 tasks
  with a regime, 242 are predictable across both size and training, 38
  across size only, 20 weak in both and 6 decline with size (three
  `truthfulqa_*_mc2`, two `rf_bbh_mcq` subtasks, `include_base_44_ukrainian`).
  Medians: R² 0.936 (IQR 0.847–0.968) for the size fit, 0.776 (0.563–0.887)
  for the trajectory fit, Spearman ρ with size 1.0 (IQR 0.9–1.0).
- The fit is made on the rungs that already cleared chance — 110 of the 136
  three-rung fits lost their two smallest rungs to the gate — so a line
  through the rising top of a sigmoid is fitted and R² is inflated by
  construction; and the reference is inside every fit, so R² is a property of
  the ladder including 1.7B, not a statement about predicting 1.7B from
  below (figure 3 is that statement).
- With and without the twins: the 306 tasks are 201 originals (median
  size-fit R² 0.943) and 105 twins (0.921), with the same regime shares to
  within a few points, so the twins change the population, not the verdict
  ([rq00 figure 3](../rq00_gate_and_curves/README.md#3-the-reformulated-twins-move-whole-families-across-the-gate)).
- Spearman ρ over five points saturates (ρ = 1 in 67 % of the 1 044 benchmark
  fits), and the trajectory fit runs over all saved points under a WSD decay
  that is not log-linear in tokens, so part of panel (b)'s spread is
  schedule.

**Follow-ups**

- The paper panel is crowded (306 points, 25 labelled families). In order of
  preference: (1) the twins as a second panel beside the originals (a
  `--twins {all,originals,twins}` switch on `regimes.py`, no new computation;
  also the figure that shows the reformulation working); (2) label a family
  only from five tasks up, the rest in a footer, hollow markers for the twins;
  (3) one marker per family at its median, area ∝ task count, outlier labels
  kept.
- Carry the fit's slope per decade beside R², or drop the ρ panel: ρ over
  five rungs carries almost no information.
- The paper text (`04_analysis.tex`) still describes weak fits for
  `global_mmlu_full`, `belebele` and `truthfulqa`; on the current tables
  those families have no point at all and their twins fit at R² 0.91–0.95.
  The paragraph and the caption need the population statement of figure 2.

GitHub: [scaling_regimes_outliers_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_outliers_paper.png) · [scaling_regimes_outliers_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_outliers_paper.csv) ·
GitHub: [scaling_regimes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes.png) · [scaling_regimes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes.csv) ·
GitHub: [scaling_regimes_outliers.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_outliers.png) · [scaling_regimes_outliers.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_outliers.csv) ·
GitHub: [scaling_regimes_families.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_families.png) · [scaling_regimes_families.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_families.csv) ·
GitHub: [scaling_regimes_by_family.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_by_family.png) · [scaling_regimes_by_family.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_by_family.csv) ·
GitHub: [scaling_regimes_by_family_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_by_family_paper.png) · [scaling_regimes_by_family_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_by_family_paper.csv)

### 2. Survivorship: what the gate and the fit minimum removed

*The same population as figure 1, per family: the tasks the regimes figure
draws against the two ways a task loses its point.* It exists because
figure 1 shows the 306 tasks that have a regime and not the 520 of 826 that
do not.

![Survivorship](pretraining/predictivity_all/scaling_regimes_survivorship.png)

*`regimes_survivorship.py` reads `scaling_regimes.csv` and `rq1_fits.csv`:
(a) per family, the tasks in the figure against those at chance at every
size where a fit was possible (rule 1, `gated`) or fitted at a single L and
so without a median (`below_min_fits`); (b) figure 1's panel with kept/total
in every label. It does not replace `regimes.py`, whose table it reads.*

**Key findings**

- 417 tasks have no fit at any L because they are at chance wherever a fit
  was possible, and a further 103 have a fit at a single L and so no median.
  15 of the 40 families lose every task, five of them with ≥ 17 tasks
  (`global_piqa_parallel_cloze` 0/63, `belebele` 0/59, `global_mmlu_full`
  0/29, `cultural_bench_hard` 0/19, `bbh_mcq` 0/17); `arc` keeps 2 of 28.
- The reformulated twins put those families back (`rf_belebele` 35/59,
  `rf_global_mmlu_full` 21/29, `rf_include_base_44` 13/36,
  `rfgm_include_base_44` 14/36, `rf_bbh_mcq` 9/17, `rf_cultural_bench_easy`
  6/19), and the probe families evaluated from 600M add `include_v2_en` 48/77
  and `include_v2_og` 38/77.
- The 20 BPB tasks below the fit minimum are the languages only the L50
  mixture trains, which have one L setting by construction.

**Follow-ups**

- Make the population statement part of the paper figure's caption
  (`plan/decision_accuracy.md` §5 has the recommendation).

GitHub: [scaling_regimes_survivorship.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_survivorship.png) · [scaling_regimes_survivorship.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_survivorship.csv)

### 3. Scaling-law error: predicting the 1.7B BPB from below

*`predictivity_all`, deep scheme-A seed-1904 cells, per-language BPB
(ungated: no chance level), a power law fitted on the proxy rungs up to a
ladder top and evaluated at the 1.7B reference (rule 11: the reference never
enters the fit).* The leave-top-out error is the prediction test figure 1's
R² cannot be.

<!-- BEGIN auto:scaling-law-error (scaling_law_error.py --pool predictivity_all) -->
## Scaling-law error on per-language BPB

Numbers from the `predictivity_all` pool (deep, scheme A, seed 1904; the loader keeps trained languages only). The reference is 1.7B for every chain (rule 9); a chain without a 1.7B final is dropped, never referenced at a smaller size: dropped L2 deep/ES, L2 deep/ZH, L15 deep/AT3, L30 deep/AT3. Regenerate with `python analysis/rq01_scaling_predictability/scaling_law_error.py --pool predictivity_all`.

- **Scaling-law error** — median |relative error| of the 1.7B per-language BPB predicted from the proxy ladder: L1 0.038, L2 0.052, L8 0.049, L15 0.049, L30 0.043, L50 0.047 (largest proxy ladder at that L).

- **Sign** — `rel_error` in the CSV is signed, (predicted − observed) / observed: 212 of 212 plotted fits are negative, the power law under-predicts the 1.7B BPB; median signed error at the largest proxy ladder: L1 -0.038, L2 -0.052, L8 -0.049, L15 -0.049, L30 -0.043, L50 -0.047.

**Median |relative error|** (columns: largest proxy rung in the fit; languages = trained languages behind the median):

| L | languages | 600M | 1B |
|---|---|---|---|
| L1 | 1 | 0.077 | 0.038 |
| L2 | 2 | 0.098 | 0.052 |
| L8 | 8 | 0.100 | 0.049 |
| L15 | 15 | 0.092 | 0.049 |
| L30 | 30 | 0.086 | 0.043 |
| L50 | 50 | 0.087 | 0.047 |

![Scaling-law error](pretraining/predictivity_all/scaling_law_error.png)
<!-- END auto:scaling-law-error -->

**Key findings**

- The 1.7B BPB is under-predicted from every ladder top: median absolute
  relative error 11.5 % with three rungs (175M–600M) and 5.7 % with four
  (175M–1B), and every fit over-predicts the improvement (the signed error
  is negative throughout). The exponent fitted on the small rungs is too
  steep for the reference's regime — a curvature finding, not a symmetric
  error bar.
- A constant offset between small and large models shows up here but not in
  decision accuracy (rq02, rq05), so the two reads can disagree.

**Follow-ups**

- Extend the leave-top-out error from BPB to the gated benchmark tasks and
  make it, rather than R², the headline predictability statistic (R² in the
  appendix).
- rq06's leave-language-out transfer (`../rq06_language_transfer/README.md`)
  is the same fit with the exponent pooled over languages; the two tables
  should quote each other's error at k = 4 rungs.

GitHub: [scaling_law_error.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_law_error.png) · [scaling_law_error.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_law_error.csv)

### 4. The fits per family and per language

*The gated (task, L) fits behind figure 1, aggregated per family
(`rq1_families.csv`) and drawn per benchmark and per language; the loss
scaling fit per (L, arch, scheme).*

<!-- BEGIN auto:results (analyze.py --pool predictivity_all) -->
## Results

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq01_scaling_predictability/analyze.py --pool predictivity_all`.

**Per family** (median over its gated (task, L) fits, ≥ 3 fits; ρ = −1 is the ideal for BPB and loss):

| family | R² | ρ | fits | options |
|---|---|---|---|---|
| cultural_bench_easy | 0.21 | 0.50 | 3 | 4 |
| include_base_44 | 0.53 | -0.20 | 4 | 4 |
| truthfulqa_mc2 | 0.60 | -0.60 | 13 |  |
| rf_acp_bench_mcq | 0.64 | 0.72 | 30 | 4 |
| rf_bbh_mcq | 0.65 | 0.62 | 54 | 3 |
| rf_cultural_bench_easy | 0.77 | 0.87 | 27 | 4 |
| multiblimp | 0.88 | 1.00 | 76 | 2 |
| xcopa | 0.89 | 1.00 | 19 | 2 |
| paws | 0.89 | 1.00 | 11 | 2 |
| xnli | 0.91 | 0.90 | 37 | 3 |
| rf_global_mmlu_full | 0.92 | 1.00 | 71 | 4 |
| rf_include_base_44 | 0.92 | 1.00 | 47 | 4 |
| include_v2_og | 0.94 | 1.00 | 138 | 4 |
| rfgm_include_base_44 | 0.94 | 1.00 | 54 | 4 |
| loss | 0.94 | -1.00 | 6 |  |
| mathqa | 0.94 | 1.00 | 6 | 5 |
| include_v2_en | 0.94 | 1.00 | 169 | 4 |
| rf_belebele | 0.94 | 1.00 | 122 | 4 |
| bpb | 0.95 | -1.00 | 106 |  |
| xwinograd | 0.95 | 1.00 | 26 | 2 |
| hellaswag | 0.97 | 1.00 | 62 | 4 |
| rf_commonsense_qa | 0.98 | 1.00 | 6 | 5 |
| rf_mmlu | 0.98 | 1.00 | 6 | 4 |
| xstorycloze | 0.99 | 1.00 | 28 | 2 |
| arc | 0.99 | 1.00 | 12 | 4 |
| lambada_openai_mt | 0.99 | 1.00 | 22 |  |

![RQ1 scaling](pretraining/predictivity_all/rq1_scaling.png)

![Scaling fit](pretraining/predictivity_all/scaling_fit.png)
<!-- END auto:results -->

<!-- BEGIN auto:panels (panels.py --pool predictivity_all) -->
## Per benchmark and per language

The family medians above, without the aggregation (`predictivity_all` pool). Regenerate with `python analysis/rq01_scaling_predictability/panels.py --pool predictivity_all`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq01 in one figure](pretraining/predictivity_all/highlights.png)

![Median R² per benchmark and L](pretraining/predictivity_all/fit_r2_median.png)

![Fit R² per benchmark](pretraining/predictivity_all/fit_r2_by_benchmark.png)

![Fit R² per language](pretraining/predictivity_all/fit_r2_by_language.png)
<!-- END auto:panels -->

**Key findings**

- Per-language BPB, LAMBADA, XStoryCloze, XWinograd, HellaSwag and ARC follow
  a log-linear law at every L (family medians R² ≥ 0.95); the families that
  once read as unpredictable (Belebele, Global-MMLU, INCLUDE) have no gated
  fit at all, and their twins fit at 0.92–0.94.
- The answer count no longer splits the families once the gate is applied to
  the fit (median R² 0.92 / 0.88 / 0.94 for 2 / 3 / 4 options): the earlier
  "3 options 0.81, 2 options 0.64, 4 options 0.39" ordering was the gate's
  casualties measuring noise around chance.

**Follow-ups**

- Grey the cells of `fit_r2_median` whose task is at chance at 1.7B
  (`grids.mark_gated`) so the per-family grid carries the same population
  statement as figure 2.

GitHub: [rq1_scaling.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/rq1_scaling.png) · [rq1_scaling.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/rq1_scaling.csv) ·
GitHub: [scaling_fit.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_fit.png) · [scaling_fit.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/scaling_fit.csv) ·
GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/highlights.csv) ·
GitHub: [fit_r2_median.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/fit_r2_median.png) · [fit_r2_median.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/fit_r2_median.csv) ·
[fit_r2_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/fit_r2_by_benchmark.png) ·
[fit_r2_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/fit_r2_by_language.png) ·
[fit_r2.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/fit_r2.csv) ·
[rq1_fits.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/rq1_fits.csv) ·
[rq1_families.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_all/rq1_families.csv)

### 5. Scaling cleanly and ranking like the reference are different properties

*Per task: figure 1's ρ (score with model size along the ladder, `rho_size`)
against rq02's ρ (the proxy's ranking of the design variants with the 1.7B
ranking, `agreement_per_cell.csv`; DA-size, multi-axis pairs from
`predictivity_schemes` at seed 1904, gate `predictivity`). The script and its
outputs live in rq02's folder
(`../rq02_decision_accuracy/scaling_vs_ranking.py`; auto block in
[rq02's README](../rq02_decision_accuracy/README.md#11-read-next-in-the-other-rqs)).*

![Scaling against ranking](../rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking.png)

*One point per task with both a scaling regime (rq01) and a decision-accuracy
cell at the proxy (rq02). Top: rq01's ρ (jittered, it lives on a lattice)
against rq02's ρ. Bottom: rq01's R² against DA-size. The corner number is
the Spearman correlation across tasks.*

**Key findings**

- A benchmark that scales cleanly ranks the variants only somewhat better:
  across tasks, Spearman between rq01's ρ and rq02's ρ is 0.12 at the 175M
  proxy, 0.23 at 350M, 0.28 at 600M and 0.37 at 1B (167–274 tasks); between
  rq01's R² and DA-size 0.16 → 0.34.
- The tasks rq01 calls "predictable across both" read DA-size 0.54–0.59 at
  the four proxies against 0.44–0.46 for the other regimes; the trajectory
  fit is the better of rq01's statistics as a surrogate (0.38 → 0.62 with
  DA-size), the same monotonicity property FineTasks selects on
  ([rq04, FineTasks' criteria](../rq04_surrogates/README.md#finetasks-criteria-on-the-ladder)).
- Scaling smoothly with size is necessary for ranking on this ladder (almost
  every task with DA-size above 0.7 sits at ρ = 1) but far from sufficient:
  at ρ = 1 the ranking correlation spans −0.5 to 0.9. Choosing benchmarks by
  their scaling curves alone keeps many tasks that decide nothing.

**Follow-ups**

- The same scatter with the trajectory R² on the x axis, which is the
  stronger surrogate, as the panel the paper quotes.

GitHub: [scaling_vs_ranking.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking.png) · [scaling_vs_ranking.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking.csv)

## Extensions from other sweeps

None. rq01 exists on the ladder only (`predictivity_all`, `predictivity_seeds`):
the 36-model sweep had three data mixtures at four sizes with no language-count
axis and no per-language BPB, so no fit of this kind was ever made on it, and
its numbers would not be pooled with the ladder's in any case (a different
harness, task set and reference size).

## Files

- `pretraining/<pool>/rq1_fits.csv`, `rq1_families.csv`, `rq1_scaling.png/.pdf/.csv`
  — the gated (task, L) fits, the family medians, and the two-panel figure
  with its plotted values. The paper's RQ1 figure is
  `scaling_regimes_outliers_paper` (appendix: `scaling_regimes_by_family_paper`),
  copied by `documents/paper/figures/make_rq_figures.py`.
- `…/scaling_fit.csv`, `scaling_fit.png` — final loss vs N per L, one power
  law per (arch, scheme).
- `…/scaling_law_error.csv`, `scaling_law_error.png` — per (L, arch, scheme,
  language, ladder top): fitted α, predicted vs observed 1.7B BPB, signed
  relative error (`scaling_law_error.py`).
- `…/scaling_regimes*.csv` — one table next to every `scaling_regimes*.png`
  (`regimes.py`; the `_paper` twins carry the same table as their non-paper
  figure).
- `…/facts.json` — the numbers the paper quotes (merged into `rq_facts.json`).
- `…/scaling_regimes_survivorship.csv`, `.png` — per family, kept / gated /
  below the fit minimum (`regimes_survivorship.py`).
- `../rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking.{png,csv}`
  — figure 5 (`scaling_vs_ranking.py`, in rq02's folder because it reads
  rq02's per-cell agreement table).
- `…/da_goal_multi_axes_across_langs_bpb.png/.pdf/.csv`, `_cells.csv` — a
  language's BPB DA-goal against the tokens of the language the proxies had
  seen, mean over languages per (proxy size, tenth) with its standard error;
  the cells table has the per-language cells (`tokens_seen.py`).
- `…/pass_prob_vs_train_tokens_by_benchmark_<deep_A_1904|1904>[_ckpts].png/.pdf/.csv`,
  `_points.csv` — per benchmark, the share of (language, L) cells above chance
  against the tokens of the language, one line per size; the `_ckpts` twins
  read the same runs at every evaluated tenth; the `.csv` is the cell table,
  `_points.csv` the binned values drawn (`tokens_seen.py`).

<!-- BEGIN auto:tokens-seen (tokens_seen.py --pool predictivity_all) -->
## Tokens seen: exposure to a language against its evaluation

`tokens_seen.py`, `predictivity_all` pool: both figures put a language's evaluation against the training tokens of that language the model had seen (its share of the mixture from the build's plan × the cell's budget × the checkpoint's share of the run). Regenerate with `python analysis/rq01_scaling_predictability/tokens_seen.py --pool predictivity_all`.

**A. DA-goal of a language's BPB against the tokens seen.** Per proxy size and tenth of the run, the mean over languages of the share of design-variant pairs the proxy's BPB orders like the 1.7B final (rq02's kernel, every pair at seed 1904 of every scheme on the variants that train the language, ≥ 3 pairs; rule 15's multi-axis set), with its standard error over languages; the x of a cell is the mean over the pair set's proxies of the tokens of the language they had seen, and a point's x the geometric mean over languages (50 languages; `da_goal_multi_axes_across_langs_bpb_cells.csv` has the per-language cells with the min and max over variants). The 1.7B line is its own early checkpoints against its final.

| proxy size | tokens of a language at 1C | DA at 1C | tokens at 5C | DA at 5C | languages |
|---|---|---|---|---|---|
| 175M | 0.03 B | 0.74 | 0.14 B | 0.88 | 50 |
| 350M | 0.05 B | 0.83 | 0.27 B | 0.91 | 50 |
| 600M | 0.10 B | 0.68 | 0.48 B | 0.70 | 50 |
| 1B | 0.15 B | 0.75 | 0.75 B | 0.91 | 50 |

![DA-goal of BPB vs tokens seen](pretraining/predictivity_all/da_goal_multi_axes_across_langs_bpb.png)

**B. Share of a benchmark's cells above chance against the tokens seen.** One (task, size, L) cell per benchmark, language and language setting; above chance by rule 1's Wilson test on the cell's runs; cells binned 3 per decade of tokens, a point = the share of the bin's cells above chance with the cell count, one line per size. Two populations: `deep_A_1904`, the plan grid, deep / scheme A / seed 1904: one run per cell; `1904`, every seed-1904 run at the (size, L) that trains the language, every scheme and architecture. The `.csv` next to each figure is the cell table (benchmark, task, language, model_size, language_scheme, train_tokens, task_score, above_chance, share_above, n_runs), `_points.csv` the binned values drawn. The `_ckpts` twins read the same runs at every evaluated tenth (a cell = (task, size, L, tenth), x = the tokens seen by that checkpoint): ten times the cells, and a token axis that runs through every training run. Cells above chance: `deep_A_1904` 4936 of 11135 (768 tasks), `deep_A_1904_ckpts` 46054 of 111350 (768 tasks), `1904` 5684 of 12761 (768 tasks), `1904_ckpts` 53176 of 127610 (768 tasks).

![Share above chance vs tokens seen, deep_A_1904](pretraining/predictivity_all/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904.png)

![Share above chance vs tokens seen, deep_A_1904_ckpts](pretraining/predictivity_all/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904_ckpts.png)

![Share above chance vs tokens seen, 1904](pretraining/predictivity_all/pass_prob_vs_train_tokens_by_benchmark_1904.png)

![Share above chance vs tokens seen, 1904_ckpts](pretraining/predictivity_all/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.png)
<!-- END auto:tokens-seen -->
