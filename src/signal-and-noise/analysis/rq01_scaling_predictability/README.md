# What scales predictably?

## Research question

> Which benchmarks, and which per-language measurements, move with model size
> in a way a log-linear fit captures, so that a small-model measurement has a
> trend to extrapolate from at all (the paper's scaling-predictability section,
> [`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex))?
> The folder also holds the loss scaling fit and, in `scaling_law_error.py`,
> the prediction test: how well a power law fitted on the proxy rungs
> predicts the reference rung's per-language BPB.

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_seeds) -->
## Highlighted result

- **3628 (task, L) fits over 1286 tasks**, each on the rungs where the task is above chance (rule 1, rq00's mask): the gate removed 712 rungs from the fitted series and left 1312 (task, L) series with fewer than 3 rungs (no fit, `gated` in `rq1_fits.csv`), 427 tasks without any fit. Best-scaling families (median R²): `bpb` 1.00, `bbpb_hellaswag` 1.00, `bbpb_xstorycloze` 0.99, `loss` 0.99; worst: `bbpb_cultural_bench_easy` 0.12, `bbpb_bbh_mcq` 0.09, `bbpb_blend_sample` 0.03.
- **Median R² by answer count** (over the gated benchmark fits): 0.93 over the 2-option fits, 0.84 over the 3-option fits, 0.87 over the 4-option fits, 0.90 over the 5-option fits, 0.88 over the 6-option fits, 0.75 over the 7-option fits, 0.67 over the 8-option fits.
- **Loss exponent α per (L, ladder, data build)**, the seed-1904 cells, sizes in the fit in brackets: L1 deep/A 0.085 (6), L1 deep/DCLMP 0.077 (6), L1 deep/FWEB 0.082 (6), L1 shallow/A 0.089 (6), L2 deep/A 0.094 (6), L2 deep/ES 0.091 (6), L2 deep/ZH 0.093 (6), L2 shallow/A 0.101 (6), L8 deep/A 0.097 (6), L8 deep/B 0.099 (6), L8 shallow/A 0.104 (6), L8 shallow/B 0.096 (6), L8 swiglu/A 0.092 (5), L15 deep/A 0.100 (6), L15 deep/AT3 0.101 (6), L15 deep/B 0.101 (6), L15 shallow/A 0.093 (6), L15 shallow/B 0.102 (6), L15 swiglu/A 0.103 (6), L30 deep/A 0.103 (6), L30 deep/AT3 0.103 (6), L30 deep/B 0.100 (6), L30 shallow/A 0.096 (6), L30 shallow/B 0.106 (6), L30 swiglu/A 0.102 (5), L50 deep/A 0.104 (6), L50 deep/AT3 0.108 (6), L50 shallow/A 0.102 (6), L50 shallow/AT3 0.102 (6).
<!-- END auto:highlight -->

## Setup

- **Pool** `predictivity_seeds`: every seed of every cell (all seven data
  builds, read as scheme A/B/C × temperature T, and the deep, shallow and
  swiglu ladders), 181 cells, 26–34 per rung, 90M–1.7B. The log-N fits, the
  regimes and the scaling-law figure read its deep data-A seed-1904 cells; the
  loss fit and the scaling-law table read every seed-1904 (L, ladder, data
  build).
- **Gate** per (task, size): the `predictivity` mask (seed 1904, every data
  build; rule 1). Trained languages and parent tasks only; no design pairs
  here, so no pair set (figure 5 borrows decision accuracy's, stated there).
- **Grid**: the fits read each cell's final checkpoint, 90M–1.7B, each rung
  at its own batch since the 2026-09-23 retrain
  ([`plan/90M-rung-anomaly.md`](../../../../plan/90M-rung-anomaly.md);
  [`RULES.md`](../RULES.md), rules 2 and 6 for the population).
- **Series**: every task is one series per language setting L. Per-language
  BPB and the training loss enter as tasks too, never gated (no chance level,
  like the generative tasks), and fall with size, so ρ = −1 is their ideal.
- **Other populations**: the scaling-law error is tabulated for every
  seed-1904 (L, ladder, data build) chain with a 1.7B final (469 chains) and
  drawn for the deep data-A ones, with 1.7B as the only reference (rule 9).
  The loss fit covers every seed-1904 (L, ladder, data build) with ≥ 3
  rungs: 26 fits.
- **bBPB twins**: each benchmark's gold-answer bits per byte, `bbpb_*`, never
  gated (no chance level). In today's outputs their size fits run over 3–4
  rungs (1369 three-rung and 978 four-rung fits), against six rungs for 801 of
  the 1165 gated accuracy fits.
- **bBPB caveat**: the twins are still being extended to every checkpoint (a
  second refresh), so their trajectory fits and figure-1 regimes are not
  quoted here; the accuracy and BPB numbers do not depend on them.
- **Snapshot**: ladder report **2026-10-06 04:26**, analysis refresh of
  2026-10-07 (commit `b316f53b`), for every hand-written number.

## Opening figure and key findings

![Scaling regimes, the paper figure](pretraining/predictivity_seeds/scaling_regimes_outliers_paper.png)

*Figure 1's paper version: deep data-A seed-1904 cells, 90M–1.7B, one point
per task (median over ≥ 2 language settings) of the gated log-N fits and the
trajectory fits; today it also draws the bBPB twins (951 points, 342 without
them).*

- **Accuracy scales log-linearly where it is above chance:** 1165 gated
  (task, L) fits over 418 benchmark tasks, median R² 0.88 (IQR 0.77–0.94);
  per-language BPB 1.00 (106 fits, 50 languages), training loss 0.98–1.00 at
  every L. Of the 342 accuracy and BPB tasks with a regime, 263 (77 %) are
  predictable across both size and training
  ([figure 1](#1-scaling-regimes-the-paper-figure)).
- **The gate decides which benchmarks have a law at all:** 428 of 896
  accuracy and BPB tasks are at chance wherever a fit was possible, and 17 of
  42 families lose every task, among them Global-PIQA parallel (0/63),
  Belebele (0/59), INCLUDE (0/36) and Global-MMLU (0/29). Belebele's,
  Global-MMLU's and INCLUDE's reformulated twins fit at family median R²
  0.84–0.91 (BBH-MCQ's at 0.65, CulturalBench-easy's at 0.77), and
  Global-PIQA parallel has no reformulated twin
  ([figure 2](#2-survivorship-what-the-gate-and-the-fit-minimum-removed)).
- **A letter-answer benchmark's bBPB scales far better once reformulated:**
  over the eight letter-format families the original's bBPB fits at median
  R² 0.81 and its `rf_` twin's at 0.98 (530 fits each, 3–4 rungs).
  Global-PIQA parallel, at chance in all 63 tasks, scales at 0.99 in bBPB
  (147 fits; [figure 4](#4-the-fits-per-family-and-per-language)).
- **A power law fitted on the proxies over-predicts the 1.7B BPB:** in all
  318 deep data-A per-language fits (106 chains), median |error| 7.7 % with
  the fit to 350M, 4.4 % to 600M, 2.3 % to 1B
  ([figure 3](#3-scaling-law-error-predicting-the-17b-bpb-from-below)).
- **The loss exponent grows with the language count on the deep data-A
  ladder:** α = 0.085 (L1), 0.094 (L2), 0.097 (L8), 0.100 (L15), 0.103 (L30),
  0.104 (L50); 0.077–0.108 over all 26 six-rung (L, ladder, data build) fits.
  Shallow A is not monotone, and the swiglu ladder (1–2 rungs per L so far)
  has no fit.

GitHub: [scaling_regimes_outliers_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_outliers_paper.png) · [scaling_regimes_outliers_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_outliers_paper.csv)

## Methodology

- **Log-N fit.** Per (task, L): score = a + b·log₁₀ N over the rungs where
  the task is above chance (≥ 3 of them; `n_rungs` and `gated_rungs` in the
  table, and `gated` where the gate left fewer than 3 — the row stays, with
  NaN statistics, and is grey in the panels), with R² and the Spearman ρ
  between score and N; medians per family over its fits (≥ 3), the family's
  modal option count alongside (`rq1_fits.csv`, `rq1_families.csv`).
- **Loss scaling fit.** Final training loss of the seed-1904 cells against N
  per (L, ladder, data build), `pretrain.ladder_report._fit` over every rung present
  (the count per fit is in the legend and the README bullet) — the health
  check's law, but with every rung in the fit rather than the smallest
  predicted.
- **Scaling-law error.** Per (L, ladder, data build, language), log BPB = a − α log N
  is fitted on the proxy rungs up to a ladder top (≥ 3 points, the same
  `_fit`) and predicts the 1.7B BPB; a chain without a 1.7B final is dropped
  and named, never referenced at a smaller size. The relative error is signed
  ((predicted − observed) / observed, negative = the law under-predicts BPB)
  and reported per ladder top, so the table reads "how far up the ladder must
  one train before the reference is predicted within x %".

## Figures, in storyline order

### 1. Scaling regimes: the paper figure

*`predictivity_seeds`, deep data-A seed-1904 cells, 90M–1.7B, gate
`predictivity`, gated log-N fits (`rq1_fits.csv`) and trajectory fits over
each run's saved checkpoints; one point per task with a median over ≥ 2
language settings.* The paper figure for this question is
`scaling_regimes_outliers_paper` (copied by
`documents/paper/figures/make_rq_figures.py`), its appendix twin
`scaling_regimes_by_family_paper`; both draw the bBPB twins too today.

<!-- BEGIN auto:regimes (regimes.py --pool predictivity_seeds) -->
## Scaling regimes per benchmark-language pair

`regimes.py`: one point per task, medians over the deep data-A seed-1904 cells (parent tasks, trained languages) — the R² and (oriented) Spearman ρ of the gated log-N fits of `rq1_fits.csv`, one per L, and the R² of the training-trajectory fit (score ~ log tokens over a run's checkpoints, ≥ 5 points), one per (L, size). Both use only the sizes where the task is above chance (rule 1, rq00's mask): 952 tasks have a point; the gate removed 427 tasks that are at chance at every size where a fit was possible (per family: acp_bench_cloze 7, acp_bench_mcq 7, arc 26, arc_mt 9, bbh_cloze 6, bbh_mcq 17, belebele 59, blend_sample 5, commonsense_qa 1, cultural_bench_easy 19, cultural_bench_hard 19, global_mmlu_full 29, global_piqa_nonparallel_cloze 1, global_piqa_parallel_cloze 63, hellaswag 2, include_base_44 35, include_v2_en 15, include_v2_og 30, mmlu 1, openbookqa 1, paws 5, rf_acp_bench_mcq 2, rf_bbh_mcq 8, rf_belebele 3, rf_cultural_bench_easy 14, rf_global_mmlu_full 4, rf_include_base_44 16, rfgm_include_base_44 15, toxigen 1, truthfulqa-multi_mc1 1, truthfulqa_mc2 3, xnli 3); the training loss is left out (rule 7). The quadrants of (b) split at R² = 0.5, a heuristic; a task with median ρ < 0 is the fifth regime `declines with size` whatever its quadrant (29 tasks, black edge in panel (b)). Table: `scaling_regimes.csv`. Regenerate with `python analysis/rq01_scaling_predictability/regimes.py --pool predictivity_seeds`.

![Scaling regimes](pretraining/predictivity_seeds/scaling_regimes.png)

Named variants of the same points: `scaling_regimes_families.png` (one label per family at its median point, `scaling_regimes_families.csv`; `scaling_regimes_families_paper.png/.pdf/.svg` its bare, square version for the paper), `scaling_regimes_outliers.png` (plus the tasks in another quadrant than their family's majority and > 0.25 from its median point, `scaling_regimes_outliers.csv`; `scaling_regimes_outliers_paper.png/.pdf/.svg` is its bare, square-panel version for the paper, the label text pulled 45% towards the ink), `scaling_regimes_by_family.png` (panel (b) per family, tasks named by language, its per-task table with the labels next to it; `_paper.png/.pdf/.svg/.csv` is its bare version for the paper's appendix) and `scaling_regimes.html` (hover names, click-to-highlight legend; for the project site).

![Scaling regimes, outliers named](pretraining/predictivity_seeds/scaling_regimes_outliers.png)

![Scaling regimes per family](pretraining/predictivity_seeds/scaling_regimes_by_family.png)
<!-- END auto:regimes -->

**Key findings**

The bullets cover the 342 accuracy and per-language BPB tasks; the 609 bBPB
points of the figure (951 in all) are left out until the twins' checkpoint
refresh lands.

- Scaling is predictable on the tasks that survive the gate: of the 342 tasks
  with a regime, 263 (77 %) are predictable across both size and training, 51
  across size only, 23 weak in both, 1 predictable during training only
  (`multiblimp_ben`) and 4 decline with size (`rf_bbh_mcq_movie_recommendation`,
  `rf_bbh_mcq_ruin_names`, `truthfulqa-multi_mc1_es`,
  `include_v2_en_arabic_morocco`).
- Medians over the 342: R² 0.887 (IQR 0.812–0.949) for the size fit, 0.758
  (0.535–0.880) for the trajectory fit, Spearman ρ with size 0.986
  (0.921–1.000). All 30 per-language BPB tasks are predictable across both.
- The fit is made on the rungs that already cleared chance. Of the 125
  three-rung accuracy fits, 120 lost their 90M rung to the gate (111 only a
  bottom block), so a line through the rising top of a sigmoid is fitted and
  R² is inflated by construction.
- The other 5: `truthfulqa-multi_mc1_es` at L8–L50 lost its top rungs (at
  chance from 600M up, so its "declines with size" regime rests on 90M–350M
  fits), and `rf_include_base_44_nepali` at L50 lost its middle ones
  (350M–1B).
- The reference is inside all but 8 of the 1165 gated accuracy fits (where
  1.7B is at chance it drops out). R² is therefore mostly a property of the
  ladder including 1.7B, not a statement about predicting 1.7B from below
  (figure 3 is that statement).
- With and without the reformulated twins: the 342 tasks are 199 originals
  (30 of them per-language BPB; median size-fit R² 0.911) and 143 twins
  (0.854). The twins are less often predictable across both (70 % against
  82 %) and more often across size only (20 % against 11 %): they widen the
  population at a slightly weaker fit
  ([the gate's figure 3](../rq00_gate_and_curves/README.md#3-the-reformulated-twins-move-whole-families-across-the-gate)).
- Spearman ρ saturates: ρ = 1 in 612 of the 1165 gated accuracy fits (53 %;
  407 of the 801 six-rung ones, 51 %). The trajectory fit runs over all saved
  points under a WSD decay that is not log-linear in tokens, so part of panel
  (b)'s spread is schedule.

**Follow-ups**

- The paper panel is crowded (951 points, 62 families, 37 of them bBPB). In
  order of preference: (1) the accuracy tasks and the bBPB twins in separate
  panels, and the reformulated twins beside the originals (a
  `--twins {all,originals,twins}` switch on `regimes.py`, no new computation);
  (2) label a family only from five tasks up, the rest in a footer, hollow
  markers for the twins; (3) one marker per family at its median, area ∝ task
  count, outlier labels kept.
- Carry the fit's slope per decade beside R², or drop the ρ panel: ρ over
  at most six rungs saturates (53 % of the accuracy fits sit at ρ = 1).
- `04_analysis.tex` against today's family table
  (`scaling_regimes_families.csv`): HellaSwag 0.88, LAMBADA 0.96, BPB 1.00 and
  TruthfulQA 0.97 at ρ −0.75 still hold. Its "INCLUDE keeps a single task, with
  R² = 0.17" no longer does: `include_base_44` has no point (0/36); the
  paragraph also needs figure 2's population statement and that TruthfulQA's
  ρ rests on 90M–350M fits.

GitHub: [scaling_regimes_outliers_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_outliers_paper.png) · [scaling_regimes_outliers_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_outliers_paper.csv) ·
GitHub: [scaling_regimes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes.png) · [scaling_regimes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes.csv) ·
GitHub: [scaling_regimes_outliers.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_outliers.png) · [scaling_regimes_outliers.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_outliers.csv) ·
GitHub: [scaling_regimes_families.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_families.png) · [scaling_regimes_families.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_families.csv) ·
GitHub: [scaling_regimes_by_family.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_by_family.png) · [scaling_regimes_by_family.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_by_family.csv) ·
GitHub: [scaling_regimes_by_family_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_by_family_paper.png) · [scaling_regimes_by_family_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_by_family_paper.csv)

### 2. Survivorship: what the gate and the fit minimum removed

*The same population as figure 1, per family: the tasks the regimes figure
draws against the two ways a task loses its point.* It exists because
figure 1 shows the 342 accuracy and BPB tasks that have a regime and not the
554 of 896 that do not.

![Survivorship](pretraining/predictivity_seeds/scaling_regimes_survivorship.png)

*`regimes_survivorship.py` reads `scaling_regimes.csv` and `rq1_fits.csv`:
(a) per family, the tasks in the figure against those at chance at every
size where a fit was possible (rule 1, `gated`) or fitted at a single L and
so without a median (`below_min_fits`); (b) figure 1's panel with kept/total
in every label. It does not replace `regimes.py`, whose table it reads.*

**Key findings**

- Of the 896 accuracy and per-language BPB tasks, 428 have no fit at any L
  because they are at chance wherever a fit was possible, and a further 126
  have a fit at a single L and so no median.
- 17 of the 42 families lose every task, seven of them with ≥ 17 tasks
  (`global_piqa_parallel_cloze` 0/63, `belebele` 0/59, `include_base_44`
  0/36, `global_mmlu_full` 0/29, `cultural_bench_easy` 0/19,
  `cultural_bench_hard` 0/19, `bbh_mcq` 0/17); `arc` keeps 2 of 28.
- The reformulated twins put those families back (`rf_belebele` 36/59,
  `rfgm_belebele` 38/59, `rf_global_mmlu_full` 22/29, `rf_include_base_44`
  13/36, `rfgm_include_base_44` 13/36, `rf_bbh_mcq` 9/17,
  `rf_cultural_bench_easy` 5/19), and INCLUDE v2 adds `include_v2_en` 48/77
  and `include_v2_og` 38/77.
- The 20 BPB tasks below the fit minimum (30 of 50 kept) are the languages
  only the L50 mixture trains, which have one L setting by construction.
- The CSV and panel also carry the 38 bBPB families (816 tasks: 609 in the
  figure, 207 below the fit minimum, none gated); the counts above leave them
  out.

**Follow-ups**

- Make the population statement part of the paper figure's caption
  (`plan/decision_accuracy.md` §5 has the recommendation).
- Draw the bBPB families in a panel of their own once the twins cover every
  checkpoint; mixed with the accuracy families they double panel (a)'s rows.

GitHub: [scaling_regimes_survivorship.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_survivorship.png) · [scaling_regimes_survivorship.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_survivorship.csv)

### 3. Scaling-law error: predicting the 1.7B BPB from below

*`predictivity_seeds`, deep data-A seed-1904 cells, per-language BPB
(ungated: no chance level), a power law fitted on the proxy rungs up to a
ladder top and evaluated at the 1.7B reference (rule 11: the reference never
enters the fit).* The leave-top-out error is the prediction test figure 1's
R² cannot be.

<!-- BEGIN auto:scaling-law-error (scaling_law_error.py --pool predictivity_seeds) -->
## Scaling-law error on per-language BPB

Numbers from the `predictivity_seeds` pool (deep, data A, seed 1904; the loader keeps trained languages only). The reference is 1.7B for every chain (rule 9); a chain without a 1.7B final is dropped, never referenced at a smaller size: dropped L8 swiglu/A, L15 swiglu/A, L30 swiglu/A. Regenerate with `python analysis/rq01_scaling_predictability/scaling_law_error.py --pool predictivity_seeds`.

- **Scaling-law error** — median |relative error| of the 1.7B per-language BPB predicted from the proxy ladder: L1 0.011, L2 0.015, L8 0.022, L15 0.019, L30 0.024, L50 0.026 (largest proxy ladder at that L).

- **Sign** — `rel_error` in the CSV is signed, (predicted − observed) / observed: 0 of 318 plotted fits are negative, the power law under-predicts the 1.7B BPB; median signed error at the largest proxy ladder: L1 0.011, L2 0.015, L8 0.022, L15 0.019, L30 0.024, L50 0.026.

**Median |relative error|** (columns: largest proxy rung in the fit; languages = trained languages behind the median):

| L | languages | 350M | 600M | 1B |
|---|---|---|---|---|
| L1 | 1 | 0.047 | 0.021 | 0.011 |
| L2 | 2 | 0.061 | 0.031 | 0.015 |
| L8 | 8 | 0.077 | 0.038 | 0.022 |
| L15 | 15 | 0.071 | 0.039 | 0.019 |
| L30 | 30 | 0.077 | 0.043 | 0.024 |
| L50 | 50 | 0.083 | 0.051 | 0.026 |

![Scaling-law error](pretraining/predictivity_seeds/scaling_law_error.png)
<!-- END auto:scaling-law-error -->

**Key findings**

- The 1.7B BPB is over-predicted from every ladder top (deep, data A, 318
  fits over 106 language chains): median absolute relative error 7.7 % with
  three rungs (90M–350M), 4.4 % with four (to 600M) and 2.3 % with five (to
  1B). The signed error is positive in all 318, so every fit under-predicts
  the improvement: a curvature finding, not a symmetric error bar.
- Over every seed-1904 chain with a 1.7B final (deep and shallow, all data
  builds; 1407 fits, 469 chains) the medians are 7.4 %, 3.6 % and 2.0 %. Six
  fits are negative, all within 0.4 % (five shallow-A fits to 1B, one FWEB fit
  to 350M); the shallow data-A ladder reaches 1.1 % at the 1B top.
- A constant offset between small and large models shows up here but not in
  decision accuracy (nor in the design-decisions analysis), so the two reads
  can disagree.

**Follow-ups**

- Extend the leave-top-out error from BPB to the gated benchmark tasks and
  make it, rather than R², the headline predictability statistic (R² in the
  appendix).
- The language-transfer analysis's leave-language-out fit
  (`../rq06_language_transfer/README.md`) is the same fit with the exponent
  pooled over languages; the two tables should quote each other's error at
  k = 4 rungs.
- Fix the sign sentence of the block and the caption in
  `scaling_law_error.py`: "0 of 318 negative, the power law under-predicts the
  1.7B BPB" should say that no fit is negative, so the law over-predicts it.

GitHub: [scaling_law_error.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_law_error.png) · [scaling_law_error.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_law_error.csv)

### 4. The fits per family and per language

*`predictivity_seeds`, deep data-A seed-1904 cells, 90M–1.7B, gate
`predictivity`: the (task, L) fits behind figure 1 (gated accuracy, ungated
BPB) plus the loss and the ungated bBPB twins, aggregated per family
(`rq1_families.csv`) and drawn per benchmark and per language; the loss
scaling fit per seed-1904 (L, ladder, data build).*

<!-- BEGIN auto:results (analyze.py --pool predictivity_seeds) -->
## Results

Numbers from the `predictivity_seeds` pool. Regenerate with `python analysis/rq01_scaling_predictability/analyze.py --pool predictivity_seeds`.

**Per family** (median over its gated (task, L) fits, ≥ 3 fits; ρ = −1 is the ideal for BPB and loss):

| family | R² | ρ | fits | options |
|---|---|---|---|---|
| bbpb_blend_sample | 0.03 | -0.03 | 21 |  |
| bbpb_bbh_mcq | 0.09 | 0.03 | 102 |  |
| bbpb_cultural_bench_easy | 0.12 | -0.09 | 86 |  |
| bbpb_acp_bench_cloze | 0.13 | -0.03 | 42 |  |
| bbpb_bbh_cloze | 0.19 | -0.30 | 36 |  |
| bbpb_acp_bench_mcq | 0.39 | -0.60 | 42 |  |
| bbpb_commonsense_qa | 0.44 | -0.57 | 6 |  |
| rf_acp_bench_mcq | 0.49 | 0.65 | 30 | 4 |
| bbpb_toxigen | 0.52 | -0.66 | 6 |  |
| bbpb_include_base_44 | 0.56 | -0.77 | 79 |  |
| rf_bbh_mcq | 0.65 | 0.80 | 54 | 3 |
| rf_cultural_bench_easy | 0.77 | 0.87 | 23 | 4 |
| bbpb_global_mmlu_full | 0.80 | -0.83 | 79 |  |
| bbpb_rf_bbh_mcq | 0.80 | -0.90 | 102 |  |
| bbpb_mmlu | 0.81 | -0.94 | 6 |  |
| bbpb_belebele | 0.83 | -0.94 | 130 |  |
| rf_include_base_44 | 0.83 | 0.97 | 48 | 4 |
| rf_global_mmlu_full | 0.84 | 1.00 | 71 | 4 |
| mathqa | 0.84 | 0.97 | 6 | 5 |
| rf_belebele | 0.85 | 0.94 | 124 | 4 |
| xnli | 0.86 | 0.94 | 37 | 3 |
| bbpb_mathqa | 0.87 | -0.91 | 6 |  |
| bbpb_rf_commonsense_qa | 0.88 | -0.94 | 6 |  |
| xcopa | 0.88 | 0.94 | 20 | 2 |
| paws | 0.88 | 1.00 | 11 | 2 |
| include_v2_og | 0.88 | 1.00 | 141 | 4 |
| hellaswag | 0.89 | 1.00 | 62 | 4 |
| include_v2_en | 0.89 | 0.94 | 172 | 4 |
| rfgm_belebele | 0.89 | 0.99 | 130 | 4 |
| rf_mmlu | 0.90 | 1.00 | 6 | 4 |
| xwinograd | 0.92 | 1.00 | 26 | 2 |
| rfgm_include_base_44 | 0.92 | 0.94 | 49 | 4 |
| bbpb_rf_acp_bench_mcq | 0.94 | -1.00 | 42 |  |
| multiblimp | 0.94 | 1.00 | 77 | 2 |
| bbpb_xnli | 0.95 | -1.00 | 45 |  |
| rf_commonsense_qa | 0.96 | 1.00 | 6 | 5 |
| xstorycloze | 0.96 | 1.00 | 28 | 2 |
| lambada_openai_mt | 0.96 | 1.00 | 22 |  |
| bbpb_paws | 0.96 | -1.00 | 29 |  |
| bbpb_include_v2_og | 0.96 | -1.00 | 211 |  |
| bbpb_rf_cultural_bench_easy | 0.97 | -1.00 | 86 |  |
| bbpb_multiblimp | 0.97 | -1.00 | 77 |  |
| bbpb_include_v2_en | 0.97 | -1.00 | 211 |  |
| truthfulqa-multi_mc1 | 0.97 | -0.75 | 4 | 5 |
| arc | 0.97 | 1.00 | 12 | 4 |
| bbpb_openbookqa | 0.98 | -1.00 | 6 |  |
| bbpb_rf_belebele | 0.98 | -1.00 | 130 |  |
| bbpb_rf_include_base_44 | 0.98 | -1.00 | 79 |  |
| bbpb_global_piqa_parallel_cloze | 0.98 | -1.00 | 147 |  |
| arc_mt | 0.99 | 1.00 | 8 | 4 |
| bbpb_arc | 0.99 | -1.00 | 74 |  |
| bbpb_truthfulqa_mc2 | 0.99 | -1.00 | 13 |  |
| bbpb_rf_global_mmlu_full | 0.99 | -1.00 | 79 |  |
| bbpb_rf_mmlu | 0.99 | -1.00 | 6 |  |
| bbpb_rfgm_include_base_44 | 0.99 | -1.00 | 79 |  |
| bbpb_xcopa | 0.99 | -1.00 | 20 |  |
| bbpb_arc_mt | 0.99 | -1.00 | 30 |  |
| bbpb_rfgm_belebele | 0.99 | -1.00 | 130 |  |
| bbpb_truthfulqa-multi_mc1 | 0.99 | -1.00 | 10 |  |
| loss | 0.99 | -1.00 | 6 |  |
| bbpb_xstorycloze | 0.99 | -1.00 | 28 |  |
| bbpb_hellaswag | 1.00 | -1.00 | 64 |  |
| bpb | 1.00 | -1.00 | 106 |  |

![RQ1 scaling](pretraining/predictivity_seeds/rq1_scaling.png)

![Scaling fit](pretraining/predictivity_seeds/scaling_fit.png)
<!-- END auto:results -->

<!-- BEGIN auto:panels (panels.py --pool predictivity_seeds) -->
## Per benchmark and per language

The family medians above, without the aggregation (`predictivity_seeds` pool). Regenerate with `python analysis/rq01_scaling_predictability/panels.py --pool predictivity_seeds`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq01 in one figure](pretraining/predictivity_seeds/highlights.png)

![Median R² per benchmark and L](pretraining/predictivity_seeds/fit_r2_median.png)

![Fit R² per benchmark](pretraining/predictivity_seeds/fit_r2_by_benchmark.png)

![Fit R² per language](pretraining/predictivity_seeds/fit_r2_by_language.png)
<!-- END auto:panels -->

**Key findings**

- The rows pool gated accuracy fits with ungated BPB, loss and bBPB fits, so
  the population differs by row (the `fits` column); the bBPB rows rest on 3–4
  rungs each.
- Per-language BPB (family median R² 1.00, 106 fits), the loss (0.99), ARC
  (0.97; ARC-MT 0.99), LAMBADA and XStoryCloze (0.96) follow a log-linear law;
  XWinograd (0.90, 26 fits) and HellaSwag (0.89, 62) sit a step below.
- Belebele and Global-MMLU have no gated fit, and INCLUDE (`include_base_44`)
  one (Urdu at L50, R² 0.01), too few for a family median. Their reformulated
  twins fit at 0.84–0.91.
- The answer count does not order the gated accuracy fits: median R² 0.93 /
  0.84 / 0.87 / 0.90 over the 2- / 3- / 4- / 5-option fits (165 / 49 / 889 /
  22). The 6-, 7- and 8-option medians (0.88, 0.75, 0.67) rest on 6 fits each.
- bBPB fits at median R² 0.98 (2347 fits, 816 tasks) against 0.88 over the
  1165 gated accuracy fits, but on 3–4 rungs against mostly six, so the gap is
  not yet a like-for-like comparison.
- The reformulation is what makes a letter-answer benchmark's bBPB scale. On
  the eight letter-format families (Belebele, Global-MMLU, INCLUDE, BBH-MCQ,
  CulturalBench-easy, ACP-Bench-MCQ, MMLU, CommonsenseQA) the original's bBPB
  fits at 0.81 (IQR 0.29–0.96) and its `rf_` twin's at 0.98 (530 fits each;
  e.g. `bbpb_bbh_mcq` 0.21 against `bbpb_rf_bbh_mcq` 0.87).
- The weakest bBPB families are BBH-MCQ (0.21), BLEnD (0.30), CommonsenseQA
  (0.43, 6 fits) and ACP-Bench-MCQ (0.44); the cloze arms sit between (BBH
  0.72, ACP-Bench 0.60). Global-PIQA parallel, at chance in all 63 tasks,
  scales at 0.99 in bBPB (147 fits).

**Follow-ups**

- Grey the cells of `fit_r2_median` whose task is at chance at 1.7B
  (`grids.mark_gated`) so the per-family grid carries the same population
  statement as figure 2.
- Split panel (b) of `rq1_scaling` into accuracy and bBPB families. With 63
  families its labels overlap.
- Compare bBPB with accuracy on the same rungs (refit accuracy on the 3–4
  sizes the twins cover) before quoting bBPB as the better-scaling metric.

GitHub: [rq1_scaling.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/rq1_scaling.png) · [rq1_scaling.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/rq1_scaling.csv) ·
GitHub: [scaling_fit.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_fit.png) · [scaling_fit.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_fit.csv) ·
GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/highlights.csv) ·
GitHub: [fit_r2_median.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/fit_r2_median.png) · [fit_r2_median.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/fit_r2_median.csv) ·
[fit_r2_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/fit_r2_by_benchmark.png) ·
[fit_r2_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/fit_r2_by_language.png) ·
[fit_r2.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/fit_r2.csv) ·
[rq1_fits.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/rq1_fits.csv) ·
[rq1_families.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq01_scaling_predictability/pretraining/predictivity_seeds/rq1_families.csv)

### 5. Scaling cleanly and ranking like the reference are different properties

*Per task: figure 1's ρ (score with model size along the ladder, `rho_size`;
`predictivity_seeds`, deep data-A seed-1904 cells) against decision
accuracy's ρ (the proxy's ranking of the design variants with the 1.7B
ranking, `agreement_da_size_per_cell_multi_axes.csv`; DA-size, no filter,
multi-axis pairs from `predictivity` at seed 1904, gate `predictivity`). The
script and its outputs live in the decision-accuracy folder
(`../rq02_decision_accuracy/scaling_vs_ranking.py`; auto block in
[the decision-accuracy README](../rq02_decision_accuracy/README.md#12-read-next-in-the-other-rqs)).*

![Scaling against ranking](../rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking_da_size_multi_axes.png)

*One point per task with both a scaling regime (figure 1) and a
decision-accuracy cell at the proxy. Top: the size ρ (jittered, it lives on a
lattice) against the ranking ρ; bottom: the size-fit R² against DA-size; the
corner number is the Spearman correlation across tasks.*

**Key findings**

The bullets cover the accuracy and per-language BPB tasks, 220–309 per proxy
at the five proxies 90M–1B. The CSV also holds bBPB twins at 175M, 350M and
1B, which the decision-accuracy README's auto block counts in (804–918 tasks
there); they are left out here until the twins' checkpoint refresh.

- A benchmark that scales cleanly ranks the variants only somewhat better.
  Across tasks, Spearman between the size ρ and the ranking ρ is +0.31 to
  +0.40, and between the size-fit R² and DA-size +0.29 to +0.40.
- Tasks predictable across both read DA-size 0.56–0.60 at the five proxies,
  against 0.44–0.48 for the other regimes. The trajectory R² is the better
  surrogate (+0.51 to +0.57 with DA-size), the monotonicity property FineTasks
  selects on
  ([surrogates, FineTasks' criteria](../rq04_surrogates/README.md#finetasks-criteria-on-the-ladder)).
- Scaling smoothly with size is close to necessary for ranking but far from
  sufficient: 78 % of the 204 (task, proxy) cells with DA-size above 0.7 sit
  at ρ = 1. Yet over the 578 cells at ρ = 1 the ranking ρ spans −0.9 to 1.0
  (5th–95th percentile −0.27 to 0.86), so choosing benchmarks by their
  scaling curves alone keeps many tasks that decide nothing.

**Follow-ups**

- The same scatter with the trajectory R² on the x axis, which is the
  stronger surrogate, as the panel the paper quotes.

GitHub: [scaling_vs_ranking_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking_da_size_multi_axes.png) · [scaling_vs_ranking_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking_da_size_multi_axes.csv)

## Extensions from other sweeps

None. This analysis exists on the ladder only (`predictivity_seeds`):
the 36-model sweep had three data mixtures at four sizes with no language-count
axis and no per-language BPB, so no fit of this kind was ever made on it, and
its numbers would not be pooled with the ladder's in any case (a different
harness, task set and reference size).

## Files

- `pretraining/<pool>/rq1_fits.csv`, `rq1_families.csv`, `rq1_scaling.png/.pdf/.csv`
  — the gated (task, L) fits, the family medians, and the two-panel figure
  with its plotted values. The paper figure for this question is
  `scaling_regimes_outliers_paper` (appendix: `scaling_regimes_by_family_paper`),
  copied by `documents/paper/figures/make_rq_figures.py`.
- `…/scaling_fit.csv`, `scaling_fit.png` — final loss vs N per L, one power
  law per (ladder, data build).
- `…/scaling_law_error.csv`, `scaling_law_error.png` — per (L, ladder, data build,
  language, ladder top): fitted α, predicted vs observed 1.7B BPB, signed
  relative error (`scaling_law_error.py`).
- `…/scaling_regimes*.csv` — one table next to every `scaling_regimes*.png`
  (`regimes.py`; the `_paper` twins carry the same table as their non-paper
  figure).
- `…/facts.json` — the numbers the paper quotes (`analyze.py`).
- `…/scaling_regimes_survivorship.csv`, `.png` — per family, kept / gated /
  below the fit minimum (`regimes_survivorship.py`).
- `../rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking_da_size_multi_axes.{png,csv}`
  — figure 5 (`scaling_vs_ranking.py`, in the decision-accuracy folder
  because it reads that analysis's per-cell agreement table).
- `tokens_seen.py` writes its two figure families elsewhere since 2026-10-07: the share of cells above chance
  against the tokens of the language seen in [`../rq00_chance_vs_train_tokens/`](../rq00_chance_vs_train_tokens/README.md),
  a language's BPB decision accuracy against the same tokens in [`../rq02_da_vs_train_tokens/`](../rq02_da_vs_train_tokens/README.md).
