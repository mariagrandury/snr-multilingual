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

- **3408 (task, L) fits over 568 tasks**; best-scaling families (median R²): `lambada_openai_mt` 0.99, `bpb` 0.95, `xstorycloze` 0.94, `loss` 0.94; worst: `belebele` 0.29, `include_base_44` 0.25, `global_mmlu_full` 0.22.
- **The answer count splits the families**: median R² 0.64 over the 2-option fits, 0.81 over the 3-option fits, 0.39 over the 4-option fits.
- **Loss exponent α per (L, arch, scheme)**: L1 deep/A 0.125, L1 shallow/A 0.128, L2 deep/A 0.144, L2 deep/ES 0.189, L2 deep/ZH 0.154, L2 shallow/A 0.272, L8 deep/A 0.150, L8 deep/B 0.143, L8 shallow/A 0.157, L15 deep/A 0.156, L15 deep/B 0.165, L15 shallow/A 0.225, L30 deep/A 0.141, L30 deep/B 0.149, L30 shallow/A 0.161, L50 deep/A 0.145, L50 deep/AT3 0.160, L50 shallow/A 0.215.
<!-- END auto:highlight -->

## Experimental setup

The fits read the plan grid — deep, scheme A, seed 1904 — at each cell's
final checkpoint, from 175M up (the 90M rung diverged,
[`plan/90M-rung-anomaly.md`](../../../../plan/90M-rung-anomaly.md)). Every
task is one series per language setting L; per-language BPB and the training
loss enter as tasks too (they fall with size, so ρ = −1 is their ideal). The
scaling-law error uses the deep, scheme-A, seed-1904 cells at every L on
the languages the cell trains; the loss scaling fit every cell the pool
holds (all seeds and schemes for `predictivity_all`).

## Methodology

- **Log-N fit.** Per (task, L): score = a + b·log₁₀ N over the rungs present
  (≥ 3), with R² and the Spearman ρ between score and N; medians per family,
  the family's modal option count alongside (`rq1_fits.csv`,
  `rq1_families.csv`).
- **Loss scaling fit.** Final training loss against N per (L, arch, scheme),
  `pretrain.ladder_report._fit` over every rung present — the health check's
  law, but with every rung in the fit rather than the smallest predicted.
- **Scaling-law error.** Per (L, arch, scheme, language), log BPB = a − α log N
  is fitted on the proxy rungs up to a ladder top (≥ 3 points, the same
  `_fit`) and predicts the reference's BPB; the relative error per ladder top
  reads "how far up the ladder must one train before the reference is
  predicted within x %". A constant offset between small and large models
  shows up here but not in decision accuracy (rq05), so the two reads can
  disagree.

<!-- BEGIN auto:results (analyze.py --pool predictivity_all) -->
## Results

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq01_scaling_predictability/analyze.py --pool predictivity_all`.

**Per family** (median over its (task, L) fits; ρ = −1 is the ideal for BPB and loss):

| family | R² | ρ | fits | options |
|---|---|---|---|---|
| global_mmlu_full | 0.22 | -0.20 | 222 | 4 |
| include_base_44 | 0.25 | 0.00 | 258 | 4 |
| belebele | 0.29 | 0.40 | 630 | 4 |
| global_piqa_nonparallel_cloze | 0.33 | 0.13 | 24 | 2 |
| global_piqa_parallel_cloze | 0.34 | 0.21 | 546 | 2 |
| xcopa | 0.57 | 0.50 | 66 | 2 |
| truthfulqa-multi_mc1 | 0.68 | -0.57 | 18 | 4 |
| paws | 0.68 | 0.80 | 60 | 2 |
| arc | 0.76 | 0.80 | 198 | 4 |
| xnli | 0.81 | 0.80 | 108 | 3 |
| multiblimp | 0.85 | 0.97 | 342 | 2 |
| hellaswag | 0.91 | 1.00 | 186 | 4 |
| xwinograd | 0.94 | 1.00 | 36 | 2 |
| loss | 0.94 | -1.00 | 6 |  |
| xstorycloze | 0.94 | 1.00 | 78 | 2 |
| bpb | 0.95 | -1.00 | 600 |  |
| lambada_openai_mt | 0.99 | 1.00 | 30 |  |

![RQ1 scaling](pretraining/predictivity_all/rq1_scaling.png)

![Scaling fit](pretraining/predictivity_all/scaling_fit.png)
<!-- END auto:results -->

<!-- BEGIN auto:scaling-law-error (scaling_law_error.py --pool predictivity_all) -->
## Scaling-law error on per-language BPB

Numbers from the `predictivity_all` pool (deep, scheme A, seed 1904; trained languages). Regenerate with `python analysis/rq01_scaling_predictability/scaling_law_error.py --pool predictivity_all`.

- **Scaling-law error** — median |relative error| of the reference's per-language BPB predicted from the proxy ladder: L1 0.038, L2 0.052, L8 0.049, L15 0.047, L30 0.043, L50 0.046 (largest proxy ladder at that L).

**Median |relative error|** (columns: largest proxy rung in the fit):

| L | 600M | 1B |
|---|---|---|
| L1 | 0.077 | 0.038 |
| L2 | 0.098 | 0.052 |
| L8 | 0.100 | 0.049 |
| L15 | 0.047 |  |
| L30 | 0.086 | 0.043 |
| L50 | 0.046 |  |

![Scaling-law error](pretraining/predictivity_all/scaling_law_error.png)
<!-- END auto:scaling-law-error -->

## Files

- `pretraining/<pool>/rq1_fits.csv`, `rq1_families.csv`, `rq1_scaling.png/.pdf`
  — the paper's RQ1 table and figure (copied to `documents/paper/figures/`).
- `…/scaling_fit.csv`, `scaling_fit.png` — final loss vs N per L, one power
  law per (arch, scheme).
- `…/scaling_law_error.csv`, `scaling_law_error.png` — per (L, arch, scheme,
  language, ladder top): fitted α, predicted vs observed reference BPB,
  relative error (`scaling_law_error.py`).
- `…/facts.json` — the numbers the paper quotes (merged into `rq_facts.json`).

<!-- BEGIN auto:panels (panels.py --pool predictivity_all) -->
## Per benchmark and per language

The family medians above, without the aggregation (`predictivity_all` pool). Regenerate with `python analysis/rq01_scaling_predictability/panels.py --pool predictivity_all`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq01 in one figure](pretraining/predictivity_all/highlights.png)

![Median R² per benchmark and L](pretraining/predictivity_all/fit_r2_median.png)

![Fit R² per benchmark](pretraining/predictivity_all/fit_r2_by_benchmark.png)

![Fit R² per language](pretraining/predictivity_all/fit_r2_by_language.png)
<!-- END auto:panels -->
