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

- **408 (task, L) fits over 155 tasks**, each on the rungs where the task is above chance (rule 1, rq00's mask): the gate removed 98 rungs from the fitted series and left 535 (task, L) series with fewer than 3 rungs (no fit, `gated` in `rq1_fits.csv`), 225 tasks without any fit. Best-scaling families (median R²): `lambada_openai_mt` 0.99, `arc` 0.99, `xstorycloze` 0.99, `hellaswag` 0.97; worst: `paws` 0.89, `xcopa` 0.89, `multiblimp` 0.88.
- **Median R² by answer count** (over the gated benchmark fits): 0.92 over the 2-option fits, 0.91 over the 3-option fits, 0.97 over the 4-option fits.
- **Loss exponent α per (L, arch, scheme)**, the seed-1904 cells, sizes in the fit in brackets: L1 deep/A 0.125 (5), L1 shallow/A 0.127 (5), L2 deep/A 0.144 (5), L2 deep/ES 0.189 (3), L2 deep/ZH 0.154 (4), L2 shallow/A 0.166 (4), L8 deep/A 0.148 (5), L8 deep/B 0.143 (5), L8 shallow/A 0.157 (4), L15 deep/A 0.147 (5), L15 deep/B 0.165 (4), L15 shallow/A 0.182 (4), L30 deep/A 0.141 (5), L30 deep/B 0.149 (5), L30 shallow/A 0.161 (5), L50 deep/A 0.145 (5), L50 deep/AT3 0.160 (5), L50 shallow/A 0.154 (5).
<!-- END auto:highlight -->

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

<!-- BEGIN auto:results (analyze.py --pool predictivity_all) -->
## Results

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq01_scaling_predictability/analyze.py --pool predictivity_all`.

**Per family** (median over its gated (task, L) fits, ≥ 3 fits; ρ = −1 is the ideal for BPB and loss):

| family | R² | ρ | fits | options |
|---|---|---|---|---|
| multiblimp | 0.88 | 1.00 | 76 | 2 |
| xcopa | 0.89 | 1.00 | 19 | 2 |
| paws | 0.89 | 1.00 | 11 | 2 |
| xnli | 0.91 | 0.90 | 37 | 3 |
| loss | 0.94 | -1.00 | 6 |  |
| xwinograd | 0.95 | 1.00 | 26 | 2 |
| bpb | 0.95 | -1.00 | 106 |  |
| hellaswag | 0.97 | 1.00 | 62 | 4 |
| xstorycloze | 0.99 | 1.00 | 28 | 2 |
| arc | 0.99 | 1.00 | 12 | 4 |
| lambada_openai_mt | 0.99 | 1.00 | 22 |  |

![RQ1 scaling](pretraining/predictivity_all/rq1_scaling.png)

![Scaling fit](pretraining/predictivity_all/scaling_fit.png)
<!-- END auto:results -->

<!-- BEGIN auto:scaling-law-error (scaling_law_error.py --pool predictivity_all) -->
## Scaling-law error on per-language BPB

Numbers from the `predictivity_all` pool (deep, scheme A, seed 1904; the loader keeps trained languages only). The reference is 1.7B for every chain (rule 9); a chain without a 1.7B final is dropped, never referenced at a smaller size: dropped L2 deep/ES, L2 deep/ZH, L2 shallow/A, L8 shallow/A, L15 deep/A, L15 deep/B, L15 shallow/A, L30 shallow/A, L50 deep/A, L50 deep/AT3, L50 shallow/A. Regenerate with `python analysis/rq01_scaling_predictability/scaling_law_error.py --pool predictivity_all`.

- **Scaling-law error** — median |relative error| of the 1.7B per-language BPB predicted from the proxy ladder: L1 0.038, L2 0.052, L8 0.049, L30 0.043 (largest proxy ladder at that L).

- **Sign** — `rel_error` in the CSV is signed, (predicted − observed) / observed: 82 of 82 plotted fits are negative, the power law under-predicts the 1.7B BPB; median signed error at the largest proxy ladder: L1 -0.038, L2 -0.052, L8 -0.049, L30 -0.043.

**Median |relative error|** (columns: largest proxy rung in the fit; languages = trained languages behind the median):

| L | languages | 600M | 1B |
|---|---|---|---|
| L1 | 1 | 0.077 | 0.038 |
| L2 | 2 | 0.098 | 0.052 |
| L8 | 8 | 0.100 | 0.049 |
| L30 | 30 | 0.086 | 0.043 |

![Scaling-law error](pretraining/predictivity_all/scaling_law_error.png)
<!-- END auto:scaling-law-error -->

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

<!-- BEGIN auto:panels (panels.py --pool predictivity_all) -->
## Per benchmark and per language

The family medians above, without the aggregation (`predictivity_all` pool). Regenerate with `python analysis/rq01_scaling_predictability/panels.py --pool predictivity_all`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq01 in one figure](pretraining/predictivity_all/highlights.png)

![Median R² per benchmark and L](pretraining/predictivity_all/fit_r2_median.png)

![Fit R² per benchmark](pretraining/predictivity_all/fit_r2_by_benchmark.png)

![Fit R² per language](pretraining/predictivity_all/fit_r2_by_language.png)
<!-- END auto:panels -->

<!-- BEGIN auto:regimes (regimes.py --pool predictivity_all) -->
## Scaling regimes per benchmark-language pair

`regimes.py`: one point per task, medians over the deep scheme-A seed-1904 cells (parent tasks, trained languages) — the R² and (oriented) Spearman ρ of the gated log-N fits of `rq1_fits.csv`, one per L, and the R² of the training-trajectory fit (score ~ log tokens over a run's checkpoints, ≥ 5 points), one per (L, size). Both use only the sizes where the task is above chance (rule 1, rq00's mask): 109 tasks have a point; the gate removed 225 tasks that are at chance at every size where a fit was possible (per family: arc 26, belebele 59, global_mmlu_full 29, global_piqa_nonparallel_cloze 1, global_piqa_parallel_cloze 63, hellaswag 2, include_base_44 34, paws 5, truthfulqa-multi_mc1 2, xcopa 1, xnli 3); the training loss is left out (rule 7). The quadrants of (b) split at R² = 0.5, a heuristic; a task with median ρ < 0 is the fifth regime `declines with size` whatever its quadrant (0 tasks, black edge in panel (b)). Table: `scaling_regimes.csv`. Regenerate with `python analysis/rq01_scaling_predictability/regimes.py --pool predictivity_all`.

![Scaling regimes](pretraining/predictivity_all/scaling_regimes.png)

Named variants of the same points: `scaling_regimes_families.png` (one label per family at its median point, `scaling_regimes_families.csv`), `scaling_regimes_outliers.png` (plus the tasks in another quadrant than their family's majority and > 0.25 from its median point, `scaling_regimes_outliers.csv`; `scaling_regimes_outliers_paper.png/.pdf/.svg` is its bare, square-panel version for the paper, the label text pulled 45% towards the ink), `scaling_regimes_by_family.png` (panel (b) per family, tasks named by language, its per-task table with the labels next to it; `_paper.png/.pdf/.svg/.csv` is its bare version for the paper's appendix) and `scaling_regimes.html` (hover names, click-to-highlight legend; for the project site).

![Scaling regimes, outliers named](pretraining/predictivity_all/scaling_regimes_outliers.png)

![Scaling regimes per family](pretraining/predictivity_all/scaling_regimes_by_family.png)
<!-- END auto:regimes -->
