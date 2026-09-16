# RQ10 — Does it generalize to languages we have not measured? (paper RQ5)

## Research question

> If a language's BPB was measured on one or two small rungs only, can the
> exponent pooled over the other languages predict its BPB at the reference
> size — and does that hold for languages the model never trained on? The
> paper's RQ5
> ([`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex)).
> The folder also holds the detailed per-language BPB curves of every cell.

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_all) -->
## Highlighted result

- **trained languages (106 (L, language) cases)** — one 175M rung plus the pooled exponent predicts the reference within 3.4 % (median); with every proxy rung, transferred 2.1 % vs own fit 4.5 % vs largest proxy as is 7.3 %.
- **never-trained languages (494 (L, language) cases)** — one 175M rung plus the pooled exponent predicts the reference within 6.7 % (median); with every proxy rung, transferred 5.1 % vs own fit 3.2 % vs largest proxy as is 6.5 %.
- **Pooled exponent α by L**: L1 0.141, L2 0.163, L8 0.163, L15 0.165, L30 0.160, L50 0.203.
<!-- END auto:highlight -->

## Experimental setup

Per-language BPB of the plan grid (deep, scheme A, seed 1904) at the final
checkpoint, from 175M up, at every L with at least four sizes (three proxy
rungs and a reference, the largest size at that L). Every one of the 100
validation languages is a series; "trained" means the language is in the
cell's FineWeb-2 list at that L (`launch_trainings.cell_fineweb_subsets`).

## Methodology

- **Own exponent.** log BPB = a − α log N per language over the proxy rungs.
- **Leave-one-language-out.** For each language the exponent is the median
  of the other languages' α at that L; the intercept comes from the held-out
  language's k smallest rungs (k = 1 … all); the prediction at the reference
  is compared with the language's own fit (k ≥ 3) and with the largest proxy's
  BPB taken as is. Relative errors, medians per (trained, k).
- **Decision transfer.** rq06's final-checkpoint agreement on the
  `bpb_untrained` population against `bpb_trained` — whether a decision read
  on the trained languages also holds on the ones neither level trains.
- **BPB curves.** One panel per cell of the pool, per-language BPB against
  fraction of run: trained languages blue, unseen grey, macro BPB dashed
  (the progress report's `plot_bpb` on the analysis' cells).

<!-- BEGIN auto:results (analyze.py --pool predictivity_all) -->
## Results

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq10_language_transfer/analyze.py --pool predictivity_all`.

**Median |relative error| of the reference's BPB** (k = rungs measured for the held-out language):

| languages | k | transferred α | own fit | largest proxy | n |
|---|---|---|---|---|---|
| never trained | 1 | 0.067 |  | 0.358 | 494 |
| never trained | 2 | 0.065 |  | 0.163 | 494 |
| never trained | 3 | 0.058 | 0.058 | 0.102 | 494 |
| never trained | 4 | 0.051 | 0.032 | 0.065 | 359 |
| trained | 1 | 0.034 |  | 0.454 | 106 |
| trained | 2 | 0.024 |  | 0.194 | 106 |
| trained | 3 | 0.021 | 0.051 | 0.094 | 106 |
| trained | 4 | 0.021 | 0.045 | 0.073 | 41 |

![Transfer](pretraining/predictivity_all/rq5_transfer.png)

**Decision transfer** (rq06's final-checkpoint agreement, mean over interventions and L, on the languages both levels train vs the languages neither does):

| population | proxy | agreement | cells |
|---|---|---|---|
| bpb_trained | 175M | 0.26 | 8 |
| bpb_untrained | 175M | 0.42 | 12 |
| bpb_trained | 350M | 0.75 | 8 |
| bpb_untrained | 350M | 0.69 | 12 |
| bpb_trained | 600M | 0.94 | 3 |
| bpb_untrained | 600M | 0.74 | 4 |
| bpb_trained | 1B | 0.65 | 2 |
| bpb_untrained | 1B | 0.76 | 2 |

![BPB curves](pretraining/predictivity_all/bpb_curves.png)
<!-- END auto:results -->

## Files

- `pretraining/<pool>/rq5_transfer.csv`, `rq5_transfer_summary.csv`,
  `rq5_transfer.png/.pdf` — the paper's RQ5 table and figure.
- `…/bpb_curves.png` — per-cell per-language BPB curves.
- `…/facts.json` — the numbers the paper quotes.
