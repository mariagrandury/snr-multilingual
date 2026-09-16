# RQ7 — What scales predictably? (paper RQ1)

## Research question

> Which benchmarks, and which per-language measurements, move with model size
> in a way a log-linear fit captures, so that a small-model measurement has a
> trend to extrapolate from at all? The paper's RQ1
> ([`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex)).
> This folder also holds the ladder's descriptive curves — loss, scaling fit
> and benchmark trajectories — drawn from the analysis loader instead of the
> raw report, so they share every assumption of the other RQs.

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
curves use every cell the pool holds (all seeds and schemes for
`predictivity_all`), which is what makes them the detailed counterpart of
the progress report's figures: the same cells, after the loader has dropped
diverged and unfinished runs and restricted checkpoints to the shared grid.

## Methodology

- **Log-N fit.** Per (task, L): score = a + b·log₁₀ N over the rungs present
  (≥ 3), with R² and the Spearman ρ between score and N; medians per family,
  the family's modal option count alongside (`rq1_fits.csv`,
  `rq1_families.csv`).
- **Loss curves.** `ladder_report_curve.csv` (the dense per-window loss the
  report writes) restricted to the pool's cells; full run and the last 10 %
  per L, the close-up capped at 3.5 nats so the arch and scheme differences
  read (the report's `plot_loss`, same conventions).
- **Loss scaling fit.** Final training loss against N per (L, arch, scheme),
  `pretrain.ladder_report._fit` over every rung present — the health check's
  law, but with every rung in the fit rather than the smallest predicted.
- **Benchmark curves.** Mean accuracy over the tasks in the cell's trained
  languages (the watcher's list, `_trained_tasks`) against the fraction of the
  run, one line per cell, chance from `tasks.json`'s option count.

<!-- BEGIN auto:results (analyze.py --pool predictivity_all) -->
## Results

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq07_scaling_predictability/analyze.py --pool predictivity_all`.

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

![Loss curves](pretraining/predictivity_all/loss_curves.png)

![Scaling fit](pretraining/predictivity_all/scaling_fit.png)

![Benchmark curves](pretraining/predictivity_all/benchmark_curves.png)
<!-- END auto:results -->

## Files

- `pretraining/<pool>/rq1_fits.csv`, `rq1_families.csv`, `rq1_scaling.png/.pdf`
  — the paper's RQ1 table and figure (copied to `documents/paper/figures/`).
- `…/loss_curves.png`, `scaling_fit.csv`, `scaling_fit.png`,
  `benchmark_curves.png` — the ladder's curves on the analysis' cells.
- `…/facts.json` — the numbers the paper quotes (merged into `rq_facts.json`).
