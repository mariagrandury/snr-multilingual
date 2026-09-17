# RQ5 — Which proxy sizes rank a design decision like the reference, and how does that depend on the number of languages?

## Research question

> At a given number of languages L, when does a small model rank a design
> choice the way the largest model trained at that L does — and how does the
> answer move with L? This is the predictivity question of
> [`plan/small-to-large-predictivity-training-plan.md`](../../../../plan/small-to-large-predictivity-training-plan.md):
> rq00–rq04 ask which *benchmarks* carry reliable signal, this RQ asks which
> *model sizes* do — and, in `early_decision.py`, how early in the proxy's
> run (the paper's RQ2 and RQ4).

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_all) -->
## Highlighted result

- **depth (deep vs shallow) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L8: 350M, L15: 350M, L30: —, L50: —.
- **language lists (A vs B) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L8: 175M, L15: 350M, L30: 350M.
- **temperature (T=1 vs T=3) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L50: 175M.
- **Depth decision on benchmarks** — mean DA over L by proxy: 175M 0.49, 350M 0.48, 600M 0.51.
- **Is there a decision to make?** median |Δ| at the reference in seed sds — depth (deep vs shallow): benchmarks 1.1×, bits per byte 1.5×; 2nd language (ru vs es): benchmarks 1.7×, bits per byte 3.4×; language lists (A vs B): benchmarks 1.5×, bits per byte 1.6×; temperature (T=1 vs T=3): benchmarks 1.4×, bits per byte 3.9×; 2nd language (ru vs zh): benchmarks 1.1×, bits per byte 4.9×.
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
- **Effect at the reference.** Per intervention and L, the median |Δ| at
  the reference in per-task seed standard deviations (the seed sd of the
  baseline cells) — the paper's "is there a decision to make?". The two
  other reads this folder used to carry live with their themes: the
  scaling-law error in rq01 (`scaling_law_error.py`) and the effect against
  seed and checkpoint noise per cell in rq03 (`effect_vs_noise.py`).

<!-- BEGIN auto:results (analyze.py --pool predictivity_all) -->
## Results

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq05_design_decisions/analyze.py --pool predictivity_all`.

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
<!-- END auto:results -->

## How small, and how early (paper RQ2)

### Setup

Everything comes from the decision table above: with two levels,
decision accuracy is the share of population items on which the proxy
prefers the level the reference prefers. The reference is the largest size
trained at both levels, at its final checkpoint; the proxy is every smaller
size, read at the checkpoint nearest 1C, 2C, 3C, 4C and 5C of training (C = the
Chinchilla-optimal 20 tokens per parameter; 5C is the full run) — one grid answers both halves of the question, how small and how
early. Two populations: the per-language BPB of the languages both levels
train, and the benchmark tasks both levels were evaluated on. A cell needs
at least three items.

### Methodology

The per-(intervention, L, population, proxy size, fraction) rows of the two
planned decisions are averaged over L, so that every language setting counts
once. `cells` in the table is the number of settings behind a mean; the
reference each setting resolved against is carried in `refs`.

<!-- BEGIN auto:early-decision (early_decision.py --pool predictivity_all) -->
## How small, and how early (paper RQ2)

Numbers from the `predictivity_all` decision table above. Regenerate with `python analysis/rq05_design_decisions/early_decision.py --pool predictivity_all`.

- **depth (deep vs shallow), per-language bits per byte** — final-checkpoint agreement by proxy: 175M 0.00, 350M 0.51; no proxy reaches 0.75.
- **depth (deep vs shallow), benchmark tasks** — final-checkpoint agreement by proxy: 175M 0.49, 350M 0.48, 600M 0.51; no proxy reaches 0.75.
- **language lists (A vs B), per-language bits per byte** — final-checkpoint agreement by proxy: 175M 0.40, 350M 1.00, 600M 0.94, 1B 0.65; smallest proxy at ≥ 0.75: **350M**, which reaches it at 1C of training (5C = the full run).
- **language lists (A vs B), benchmark tasks** — final-checkpoint agreement by proxy: 175M 0.46, 350M 0.51, 600M 0.45, 1B 0.38; no proxy reaches 0.75.

**depth (deep vs shallow) — per-language bits per byte** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 1C | 2C | 3C | 4C | 5C |
|---|---|---|---|---|---|
| 175M | 0.50 | 0.01 | 0.00 | 0.00 | 0.00 |
| 350M | 0.74 | 0.83 | 0.55 | 0.69 | 0.51 |
| 600M | 1.00 | 1.00 | 1.00 | 0.99 |  |

**depth (deep vs shallow) — benchmark tasks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 1C | 2C | 3C | 4C | 5C |
|---|---|---|---|---|---|
| 175M | 0.39 | 0.45 | 0.47 | 0.49 | 0.49 |
| 350M | 0.50 | 0.49 | 0.48 | 0.49 | 0.48 |
| 600M | 0.46 | 0.45 | 0.53 | 0.51 | 0.51 |
| 1.7B | 0.58 | 0.46 | 0.55 | 0.51 |  |

**language lists (A vs B) — per-language bits per byte** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 1C | 2C | 3C | 4C | 5C |
|---|---|---|---|---|---|
| 175M | 0.86 | 0.68 | 0.40 | 0.40 | 0.40 |
| 350M | 0.88 | 0.94 | 1.00 | 0.97 | 1.00 |
| 600M | 0.94 | 0.94 | 0.87 | 0.88 | 0.94 |
| 1B | 0.87 | 0.75 | 0.75 | 0.58 | 0.65 |
| 1.7B | 1.00 | 1.00 | 1.00 | 0.98 |  |

**language lists (A vs B) — benchmark tasks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 1C | 2C | 3C | 4C | 5C |
|---|---|---|---|---|---|
| 175M | 0.46 | 0.50 | 0.47 | 0.45 | 0.46 |
| 350M | 0.48 | 0.48 | 0.46 | 0.49 | 0.51 |
| 600M | 0.50 | 0.49 | 0.49 | 0.49 | 0.45 |
| 1B | 0.46 | 0.44 | 0.53 | 0.53 | 0.38 |
| 1.7B | 0.43 | 0.49 | 0.54 | 0.57 |  |

![Early and small](pretraining/predictivity_all/rq2_early_small.png)
<!-- END auto:early-decision -->

## Caveats to carry into the paper

- The depth intervention's effect is small by design (aspect ratio near the
  optimum); the effect-vs-noise table decides whether its DA is interpretable
  at all. The scheme intervention changes the language set, so its
  `bpb_trained` population is the languages both schemes train.
- Checkpoint noise windows differ by size unless the shared grid is used
  (the loader's default): 5 late checkpoints span the final 25 % of a
  20-checkpoint run and 12.5 % of a 40-checkpoint one
  ([`plan/1b-models.md`](../../../../plan/1b-models.md)).
- The reference is the largest size trained at both levels: 1.7B wherever it
  exists, 1B where it does not yet (L15, and the temperature and
  second-language decisions). The `reference_size` column names it per cell.

## Files

- `pretraining/<pool>/intervention_da.csv` — one row per (intervention,
  population, L, proxy size): `decision_acc`, `n_items`, mean |Δ| at proxy and
  reference, the level the reference prefers.
- `…/rq4_da_by_intervention.csv`, `rq4_effect_vs_seed.csv`, `rq4_interventions.png/.pdf`,
  `facts.json` — the paper's RQ4 figure and the numbers it quotes.
- `…/rq2_decisions.csv`, `rq2_early_small.csv`, `rq2_early_small.png/.pdf`,
  `early_decision_facts.json` — the paper's RQ2 figure (`early_decision.py`).
- `…/intervention_da.png`.

<!-- BEGIN auto:panels (panels.py --pool predictivity_all) -->
## Per benchmark and per language

The decision table and the early read above, without pooling the benchmarks (`predictivity_all` pool). Language aggregates and per-subject facets are left out here (about 60 % of the pooled benchmark items remain), and a language's subplot averages its BPB item with its benchmark items. Regenerate with `python analysis/rq05_design_decisions/panels.py --pool predictivity_all`. White cells have no value; each figure's table sits next to it under the same name (`intervention_da_by_<unit>.csv`).

![rq05 in one figure](pretraining/predictivity_all/highlights.png)

![Decisions per benchmark](pretraining/predictivity_all/intervention_da_by_benchmark.png)

![Early and small per benchmark](pretraining/predictivity_all/intervention_da_by_benchmark_early.png)

![Decisions per language](pretraining/predictivity_all/intervention_da_by_language.png)

![Early and small per language](pretraining/predictivity_all/intervention_da_by_language_early.png)
<!-- END auto:panels -->
