# RQ9 — Can we know cheaply whether the decision will be reliable? (paper RQ3)

## Research question

> Before training the reference, can a statistic computed on the proxy alone
> — an SNR, its signal or noise part, the proxy's early-checkpoint agreement,
> how well its scores fit a scaling trend, its margin above chance — tell us
> that the proxy's decision will match the reference's? The paper's RQ3
> ([`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex)).

<!-- BEGIN auto:highlight (analyze.py --pool predictivity) -->
## Highlighted result

- **benchmark tasks** — strongest surrogate of DA-size (mean ρ over proxies): `SNR, relative std` 0.51; weakest: `SNR, discrepancy` 0.09.
- **per-language bits per byte** — strongest surrogate of DA-size (mean ρ over proxies): `SNR, relative std` 0.61; weakest: `SNR, discrepancy` -0.28.
<!-- END auto:highlight -->

## Experimental setup

The truth is DA-size from [`rq01_decision_accuracy/`](../rq01_decision_accuracy/):
per task, the agreement between the ranking of the design variants at a proxy
size and at the target size, as carried in rq02's per-task table
(`snr_variants_per_task.csv`, `decision_acc_size_<proxy>`). The candidates are
read from the same table (`snr_*`, `signal_*`, `noise_*`, `decision_acc_ckpt_*`),
from rq00's above-random scores (margin above chance) and from rq07's log-N
fits (R²). The population per proxy size is the set of tasks with an SNR at
that size — the above-random survivors — so every candidate is scored on the
same tasks; benchmarks and per-language BPB are scored separately.

## Methodology

Spearman ρ between the candidate and DA-size over the population, per proxy
size (the `snr` block's `small_sizes` that have a DA-size column), with the
p-value and n. A cell needs at least 8 tasks. The noise part is inverted so
that "higher is better" holds for every candidate.

<!-- BEGIN auto:results (analyze.py --pool predictivity) -->
## Results

Numbers from the `predictivity` pool's rq02 table. Regenerate with `python analysis/rq09_surrogates/analyze.py --pool predictivity`.

**benchmark tasks** (Spearman ρ of the statistic with DA-size, per proxy size):

| metric | 175M | 350M | 600M |
|---|---|---|---|
| SNR, relative std | 0.52 | 0.60 | 0.41 |
| early-checkpoint agreement (20 %) | 0.33 | 0.59 | 0.57 |
| signal alone (relative std) | 0.48 | 0.54 | 0.36 |
| SNR, dist_std | 0.43 | 0.53 | 0.33 |
| scaling-fit R² | 0.21 | 0.34 | 0.48 |
| noise alone (relative std, inverted) | 0.09 | 0.18 | 0.08 |
| margin above chance | 0.27 | 0.14 | -0.12 |
| SNR, discrepancy | 0.11 | 0.16 | -0.01 |

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
<!-- END auto:results -->

## Files

- `pretraining/<pool>/rq3_surrogates.csv` — ρ, p, n per (proxy, kind, metric).
- `…/rq3_surrogates.png/.pdf` — the paper figure; `facts.json` the numbers it quotes.
