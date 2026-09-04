# RQ6 — Which proxy sizes rank a design choice like the reference, and how does that depend on the number of languages?

## Research question

> At a given number of languages L, when does a small model rank a design
> choice the way the largest model trained at that L does — and how does the
> answer move with L? This is the predictivity question of
> [`plan/small-to-large-predictivity-training-plan.md`](../../../../plan/small-to-large-predictivity-training-plan.md):
> rq00–rq05 ask which *benchmarks* carry reliable signal, this RQ asks which
> *model sizes* do.

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_seeds) -->
## Highlighted result

_Not generated yet — the ladder report (`msnr-data/ladder-report`) was not
published when this README was written. Run
`bash run_all_predictivity.sh` (or
`python analysis/rq06_proxy_predictivity/analyze.py --pool predictivity_seeds`)
to fill this block._
<!-- END auto:highlight -->

## Experimental setup

The grid is the predictivity ladder: sizes 90M–1.7B (non-embedding), language
settings L ∈ {1, 2, 8, 15, 30, 50, 100}, two intervention axes — model depth
(deep, width/depth ≈ 64, vs shallow, ≈ 128, at equal non-embedding size) and
data scheme (A resource-ranked vs B diversity-first language sets, which differ
only at L ∈ {8, 15, 30}) — and seed replicates on the ×3 cells. Every read uses
each cell's final checkpoint (D = 100·N tokens, WSD-annealed). The reference at
each L is the largest size trained there (1.7B where it exists, else 1B); a
proxy is every smaller size. Diverged runs (the 90M rung, see
[`plan/90M-rung-anomaly.md`](../../../../plan/90M-rung-anomaly.md)) and runs that
have not reached their target are excluded by the loader.

Populations for the decision: per-language BPB on the languages both levels
train (`bpb_trained`, the plan's primary outcome), on all 100 validation
languages (`bpb_all`, zero-shot transfer included), the benchmark tasks the
cell was evaluated on (`benchmark`), and the single macro-BPB decision
(`bpb_macro`, the "aggregate criterion" the plan asks to compare against the
per-language one).

## Methodology

- **Intervention decision accuracy.** For each item of a population, the
  decision is which level of the intervention is better; DA(proxy, L) is the
  fraction of items on which the proxy agrees with the reference
  (`snr.metrics.decision_acc_fast` on the two models of one item). With two
  levels the pairwise-ranking definition of Heineman et al. (2025) reduces to
  this sign agreement. `n_items` is reported with every cell; a benchmark cell
  needs ≥ 3 items.
- **Scaling-law error.** Per (L, arch, scheme, language), log BPB = a − α log N
  is fitted on the proxy rungs up to a ladder top (≥ 3 points) and predicts
  the reference's BPB; the relative error is reported per ladder top, so the
  table reads "how far up the ladder must one train before the reference is
  predicted within x %". The fit is `pretrain.ladder_report._fit`, the same
  power law the ladder health check uses on the training loss. A constant
  offset between small and large models shows up here but not in DA — the two
  reads can disagree, and the plan says so.
- **Effect vs noise.** For every (size, L, task) the intervention's |Δ| is put
  against the seed noise (sample std over the seed replicates, where ≥ 2
  seeds exist) and the late-checkpoint noise (std over the last `last_n`
  checkpoints of the baseline cell, raw and detrended — under WSD the final
  window is still descending, so the raw std carries trend). A ratio near 1
  means the two levels are the same model as far as a ranking is concerned
  (the "read this against the seed row" rule of `ladder_report.md`); a
  decision on such a cell is a coin flip whatever its DA.

<!-- BEGIN auto:results (analyze.py --pool predictivity_seeds) -->
## Results

_Not generated yet — see the highlight block above._
<!-- END auto:results -->

## Caveats to carry into the paper

- The depth intervention's effect is small by design (aspect ratio near the
  optimum); the effect-vs-noise table decides whether its DA is interpretable
  at all. The scheme intervention changes the language set, so its
  `bpb_trained` population is the languages both schemes train.
- Checkpoint noise windows differ by size unless the shared grid is used
  (the loader's default): 5 late checkpoints span the final 25 % of a
  20-checkpoint run and 12.5 % of a 40-checkpoint one
  ([`plan/1b-models.md`](../../../../plan/1b-models.md)).
- Reference = 1B at L ∈ {15, 50} and, until the 1.7B row lands, everywhere.
  The `reference_size` column names it per cell.

## Files

- `pretraining/<pool>/intervention_da.csv` — one row per (intervention,
  population, L, proxy size): `decision_acc`, `n_items`, mean |Δ| at proxy and
  reference, the level the reference prefers.
- `…/scaling_law_error.csv` — per (L, arch, scheme, language, ladder top):
  fitted α, predicted vs observed reference BPB, relative error.
- `…/effect_vs_noise.csv` — per (size, L, task): |Δ| per intervention, seed
  noise, raw and detrended checkpoint noise, and their ratios.
- `…/intervention_da.png`, `scaling_law_error.png`, `effect_vs_noise.png`.
