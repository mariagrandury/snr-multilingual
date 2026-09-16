# RQ6 — Which proxy sizes rank a design choice like the reference, and how does that depend on the number of languages?

## Research question

> At a given number of languages L, when does a small model rank a design
> choice the way the largest model trained at that L does — and how does the
> answer move with L? This is the predictivity question of
> [`plan/small-to-large-predictivity-training-plan.md`](../../../../plan/small-to-large-predictivity-training-plan.md):
> rq00–rq05 ask which *benchmarks* carry reliable signal, this RQ asks which
> *model sizes* do.

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_all) -->
## Highlighted result

- **depth (deep vs shallow) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L8: 350M, L15: 350M, L30: —, L50: —.
- **language lists (A vs B) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L8: 175M, L15: 350M, L30: 350M.
- **temperature (T=1 vs T=3) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L50: 175M.
- **Depth decision on benchmarks** — mean DA over L by proxy: 175M 0.49, 350M 0.48, 600M 0.51.
- **Is there a decision to make?** median |Δ| at the reference in seed sds — depth (deep vs shallow): benchmarks 1.1×, bits per byte 1.5×; 2nd language (ru vs es): benchmarks 1.7×, bits per byte 3.4×; language lists (A vs B): benchmarks 1.5×, bits per byte 1.6×; temperature (T=1 vs T=3): benchmarks 1.4×, bits per byte 3.9×; 2nd language (ru vs zh): benchmarks 1.1×, bits per byte 4.9×.
- **Scaling-law error** — median |relative error| of the reference's per-language BPB predicted from the proxy ladder: L1 0.038, L2 0.052, L8 0.049, L15 0.047, L30 0.043, L50 0.046 (largest proxy ladder at that L).
- **Seed noise vs detrended checkpoint noise** — median ratio 1.88 over 7870 (size, L, task) cells with seed replicates.
- **Depth effect vs seed noise** — median |Δ|/seed-std 1.39; 34% of 5552 cells above 2× (a distinct model for SNR, not a re-roll).
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
  With two levels the pairwise-ranking definition of Heineman et al. (2025)
  reduces to sign agreement on the two models of one item, which is
  `snr.metrics.decision_acc_fast` per item; items the reference ties are
  dropped rather than counted as misses, since they leave no decision to
  agree with (the one place this RQ departs from the kernel). `n_items` is reported with every cell; a benchmark cell
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

<!-- BEGIN auto:results (analyze.py --pool predictivity_all) -->
## Results

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq06_proxy_predictivity/analyze.py --pool predictivity_all`.

**depth (deep vs shallow), per-language BPB** (rows: proxy size; columns: L; reference L8 → 600M, L15 → 600M, L30 → 600M, L50 → 600M):

| proxy | L8 | L15 | L30 | L50 |
|---|---|---|---|---|
| 175M | 0.00 | 0.00 | 0.00 | 0.00 |
| 350M | 0.75 | 1.00 | 0.03 | 0.26 |

**language lists (A vs B), per-language BPB** (rows: proxy size; columns: L; reference L8 → 1.7B, L15 → 1B, L30 → 1.7B):

| proxy | L8 | L15 | L30 |
|---|---|---|---|
| 175M | 1.00 | 0.17 | 0.04 |
| 350M | 1.00 | 1.00 | 1.00 |
| 600M | 1.00 | 0.83 | 1.00 |
| 1B | 1.00 |  | 0.31 |

**temperature (T=1 vs T=3), per-language BPB** (rows: proxy size; columns: L; reference L50 → 600M):

| proxy | L50 |
|---|---|
| 175M | 0.88 |
| 350M | 0.96 |

**depth (deep vs shallow), benchmarks** (rows: proxy size; columns: L; reference L1 → 1.7B, L2 → 1.7B, L8 → 600M, L15 → 600M, L30 → 1.7B, L50 → 600M):

| proxy | L1 | L2 | L8 | L15 | L30 | L50 |
|---|---|---|---|---|---|---|
| 175M | 0.47 | 0.51 | 0.47 | 0.52 | 0.51 | 0.49 |
| 350M | 0.51 | 0.44 | 0.47 | 0.56 | 0.46 | 0.42 |
| 600M | 0.60 | 0.52 |  |  | 0.42 |  |

**language lists (A vs B), benchmarks** (rows: proxy size; columns: L; reference L8 → 1.7B, L15 → 1B, L30 → 1.7B):

| proxy | L8 | L15 | L30 |
|---|---|---|---|
| 175M | 0.50 | 0.45 | 0.43 |
| 350M | 0.52 | 0.48 | 0.52 |
| 600M | 0.48 | 0.50 | 0.37 |
| 1B |  |  | 0.38 |

**temperature (T=1 vs T=3), benchmarks** (rows: proxy size; columns: L; reference L50 → 1B):

| proxy | L50 |
|---|---|
| 175M | 0.35 |
| 350M | 0.42 |
| 600M | 0.48 |

**2nd language (ru vs zh), benchmarks** (rows: proxy size; columns: L; reference L2 → 1B):

| proxy | L2 |
|---|---|
| 175M | 0.52 |
| 350M | 0.53 |
| 600M | 0.54 |

**2nd language (ru vs es), benchmarks** (rows: proxy size; columns: L; reference L2 → 600M):

| proxy | L2 |
|---|---|
| 175M | 0.30 |
| 350M | 0.43 |

![Intervention DA grid](pretraining/predictivity_all/intervention_da.png)

**Effect at the reference in seed standard deviations** (median over items and L):

| intervention | benchmarks | bits per byte |
|---|---|---|
| depth (deep vs shallow) | 1.1 | 1.5 |
| 2nd language (ru vs es) | 1.7 | 3.4 |
| language lists (A vs B) | 1.5 | 1.6 |
| temperature (T=1 vs T=3) | 1.4 | 3.9 |
| 2nd language (ru vs zh) | 1.1 | 4.9 |

![Interventions](pretraining/predictivity_all/rq4_interventions.png)

**Scaling-law error on trained-language BPB** (median |relative error|; columns: largest proxy rung in the fit):

| L | 600M | 1B |
|---|---|---|
| L1 | 0.077 | 0.038 |
| L2 | 0.098 | 0.052 |
| L8 | 0.100 | 0.049 |
| L15 | 0.047 |  |
| L30 | 0.086 | 0.043 |
| L50 | 0.046 |  |

![Scaling-law error](pretraining/predictivity_all/scaling_law_error.png)

**Intervention effect against noise** (median over (size, L, task) cells):

| population | effect / noise | median | n |
|---|---|---|---|
| benchmark | arch / seed | 1.32 | 4946 |
| benchmark | arch / ckpt | 2.24 | 20080 |
| benchmark | scheme / seed | 1.09 | 1802 |
| benchmark | scheme / ckpt | 1.91 | 34260 |
| benchmark | temperature / seed | 1.13 | 4473 |
| benchmark | temperature / ckpt | 2.07 | 8990 |
| benchmark | zh / seed | 1.31 | 221 |
| benchmark | zh / ckpt | 2.01 | 539 |
| benchmark | es / seed | 1.49 | 146 |
| benchmark | es / ckpt | 2.77 | 379 |
| bpb | arch / seed | 1.86 | 606 |
| bpb | arch / ckpt | 4.34 | 1919 |
| bpb | scheme / seed | 0.49 | 101 |
| bpb | scheme / ckpt | 5.52 | 1313 |
| bpb | temperature / seed | 4.05 | 202 |
| bpb | temperature / ckpt | 12.05 | 303 |
| bpb | zh / seed | 2.77 | 202 |
| bpb | zh / ckpt | 14.46 | 303 |
| bpb | es / seed | 3.04 | 202 |
| bpb | es / ckpt | 14.66 | 303 |

![Effect vs noise](pretraining/predictivity_all/effect_vs_noise.png)
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
