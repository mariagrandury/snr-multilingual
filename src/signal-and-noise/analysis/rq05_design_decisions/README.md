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

- **depth (deep vs shallow) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L8: 175M, L15: 175M, L30: 175M, L50: 175M.
- **language lists (A vs B) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L8: 175M, L15: 350M, L30: 350M.
- **temperature (T=1 vs T=3) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L15: 175M, L30: 175M, L50: 175M.
- **Depth decision on benchmarks** — mean DA over L by proxy: 175M 0.51, 350M 0.49, 600M 0.46, 1B 0.49.
- **Is there a decision to make?** median |Δ| at the reference in seed sds — depth (deep vs shallow): benchmarks 1.3×, bits per byte 1.0×; 2nd language (ru vs es): benchmarks 1.1×; language lists (A vs B): benchmarks 1.4×, bits per byte 1.1×; temperature (T=1 vs T=3): benchmarks 1.4×, bits per byte 5.9×; 2nd language (ru vs zh): benchmarks 1.1×.
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

Hand-written numbers in this README are from the ladder-report snapshot
**2026-09-23 06:16**.

<!-- BEGIN auto:results (analyze.py --pool predictivity_all) -->
## Results

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq05_design_decisions/analyze.py --pool predictivity_all`.

**depth (deep vs shallow), per-language BPB** (rows: proxy size; columns: L; reference L8 → 1.7B, L15 → 1.7B, L30 → 1.7B, L50 → 1.7B):

| proxy | L8 | L15 | L30 | L50 |
|---|---|---|---|---|
| 175M | 1.00 | 1.00 | 1.00 | 1.00 |
| 350M | 0.25 | 0.00 | 0.97 | 0.74 |
| 600M | 0.00 | 0.00 | 0.00 | 0.00 |
| 1B | 1.00 | 1.00 | 1.00 | 0.78 |

**language lists (A vs B), per-language BPB** (rows: proxy size; columns: L; reference L8 → 1.7B, L15 → 1.7B, L30 → 1.7B):

| proxy | L8 | L15 | L30 |
|---|---|---|---|
| 175M | 1.00 | 0.17 | 0.04 |
| 350M | 1.00 | 1.00 | 1.00 |
| 600M | 1.00 | 0.83 | 1.00 |
| 1B | 1.00 | 1.00 | 0.31 |

**temperature (T=1 vs T=3), per-language BPB** (rows: proxy size; columns: L; reference L15 → 1B, L30 → 600M, L50 → 1.7B):

| proxy | L15 | L30 | L50 |
|---|---|---|---|
| 175M | 0.93 | 0.83 | 0.90 |
| 350M | 0.93 | 0.97 | 0.98 |
| 600M | 0.93 |  | 0.98 |
| 1B |  |  | 0.98 |

**depth (deep vs shallow), benchmarks** (rows: proxy size; columns: L; reference L1 → 1.7B, L2 → 1.7B, L8 → 1.7B, L15 → 1.7B, L30 → 1.7B, L50 → 1.7B):

| proxy | L1 | L2 | L8 | L15 | L30 | L50 |
|---|---|---|---|---|---|---|
| 175M | 0.56 | 0.44 | 0.55 | 0.52 | 0.52 | 0.48 |
| 350M | 0.56 | 0.50 | 0.53 | 0.45 | 0.47 | 0.41 |
| 600M | 0.44 | 0.46 | 0.47 | 0.44 | 0.49 | 0.46 |
| 1B | 0.56 | 0.43 | 0.51 | 0.51 | 0.46 | 0.49 |

**language lists (A vs B), benchmarks** (rows: proxy size; columns: L; reference L8 → 1.7B, L15 → 1.7B, L30 → 1.7B):

| proxy | L8 | L15 | L30 |
|---|---|---|---|
| 175M | 0.46 | 0.49 | 0.45 |
| 350M | 0.51 | 0.49 | 0.46 |
| 600M | 0.37 | 0.39 | 0.47 |
| 1B | 0.48 | 0.47 | 0.42 |

**temperature (T=1 vs T=3), benchmarks** (rows: proxy size; columns: L; reference L15 → 1.7B, L30 → 1.7B, L50 → 1.7B):

| proxy | L15 | L30 | L50 |
|---|---|---|---|
| 175M | 0.50 | 0.45 | 0.41 |
| 350M | 0.54 | 0.48 | 0.46 |
| 600M | 0.53 | 0.54 | 0.55 |
| 1B | 0.53 | 0.51 | 0.55 |

**2nd language (ru vs zh), benchmarks** (rows: proxy size; columns: L; reference L2 → 1.7B):

| proxy | L2 |
|---|---|
| 175M | 0.33 |
| 350M | 0.51 |
| 600M | 0.49 |
| 1B | 0.50 |

**2nd language (ru vs es), benchmarks** (rows: proxy size; columns: L; reference L2 → 1B):

| proxy | L2 |
|---|---|
| 175M | 0.44 |
| 350M | 0.57 |
| 600M | 0.46 |

![Intervention DA grid](pretraining/predictivity_all/intervention_da.png)

**Effect at the reference in seed standard deviations** (median over items and L):

| intervention | benchmarks | bits per byte |
|---|---|---|
| depth (deep vs shallow) | 1.3 | 1.0 |
| 2nd language (ru vs es) | 1.1 |  |
| language lists (A vs B) | 1.4 | 1.1 |
| temperature (T=1 vs T=3) | 1.4 | 5.9 |
| 2nd language (ru vs zh) | 1.1 |  |

![Interventions](pretraining/predictivity_all/rq4_interventions.png)
<!-- END auto:results -->

GitHub: [intervention_da.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/intervention_da.png) · [intervention_da.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/intervention_da.csv) ·
GitHub: [rq4_interventions.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/rq4_interventions.png) · [rq4_interventions.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/rq4_interventions.csv) ·
[rq4_effect_vs_seed.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/rq4_effect_vs_seed.csv) ·
[rq4_da_by_intervention.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/rq4_da_by_intervention.csv)

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

- **depth (deep vs shallow), per-language bits per byte** — final-checkpoint agreement by proxy: 175M 1.00, 350M 0.49, 600M 0.00, 1B 0.95; smallest proxy at ≥ 0.75: **175M**, which reaches it at 0.5C of training (5C = the full run).
- **depth (deep vs shallow), benchmark tasks** — final-checkpoint agreement by proxy: 175M 0.51, 350M 0.49, 600M 0.46, 1B 0.49; no proxy reaches 0.75.
- **language lists (A vs B), per-language bits per byte** — final-checkpoint agreement by proxy: 175M 0.40, 350M 1.00, 600M 0.94, 1B 0.77; smallest proxy at ≥ 0.75: **350M**, which reaches it at 0.5C of training (5C = the full run).
- **language lists (A vs B), benchmark tasks** — final-checkpoint agreement by proxy: 175M 0.47, 350M 0.49, 600M 0.41, 1B 0.46; no proxy reaches 0.75.

**depth (deep vs shallow) — per-language bits per byte** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.91 | 0.50 | 1.00 | 0.99 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 |
| 350M | 0.25 | 0.26 | 0.20 | 0.17 | 0.36 | 0.45 | 0.29 | 0.31 | 0.44 | 0.49 |
| 600M | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 |
| 1B | 0.27 | 0.28 | 0.54 | 0.83 | 0.85 | 0.87 | 0.90 | 0.92 | 0.94 | 0.95 |
| 1.7B | 0.69 | 0.98 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 |  |

**depth (deep vs shallow) — benchmark tasks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.50 | 0.40 | 0.47 | 0.45 | 0.50 | 0.45 | 0.50 | 0.50 | 0.52 | 0.51 |
| 350M | 0.43 | 0.53 | 0.47 | 0.50 | 0.50 | 0.42 | 0.51 | 0.47 | 0.49 | 0.49 |
| 600M | 0.42 | 0.46 | 0.44 | 0.46 | 0.46 | 0.46 | 0.47 | 0.47 | 0.48 | 0.46 |
| 1B | 0.48 | 0.47 | 0.48 | 0.48 | 0.46 | 0.45 | 0.51 | 0.51 | 0.49 | 0.49 |
| 1.7B | 0.47 | 0.49 | 0.50 | 0.53 | 0.52 | 0.54 | 0.58 | 0.59 | 0.71 |  |

**language lists (A vs B) — per-language bits per byte** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.19 | 0.86 | 0.57 | 0.68 | 1.00 | 0.40 | 0.34 | 0.40 | 0.40 | 0.40 |
| 350M | 0.86 | 0.88 | 0.94 | 0.94 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 |
| 600M | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.87 | 0.94 | 0.88 | 0.94 | 0.94 |
| 1B | 0.79 | 0.91 | 0.87 | 0.83 | 0.83 | 0.83 | 0.81 | 0.72 | 0.77 | 0.77 |
| 1.7B | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |  |

**language lists (A vs B) — benchmark tasks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.42 | 0.43 | 0.45 | 0.44 | 0.47 | 0.52 | 0.46 | 0.45 | 0.48 | 0.47 |
| 350M | 0.45 | 0.45 | 0.47 | 0.43 | 0.43 | 0.47 | 0.50 | 0.48 | 0.50 | 0.49 |
| 600M | 0.48 | 0.47 | 0.47 | 0.50 | 0.48 | 0.47 | 0.45 | 0.44 | 0.43 | 0.41 |
| 1B | 0.50 | 0.50 | 0.50 | 0.48 | 0.47 | 0.46 | 0.48 | 0.46 | 0.48 | 0.46 |
| 1.7B | 0.47 | 0.45 | 0.49 | 0.59 | 0.53 | 0.58 | 0.54 | 0.57 | 0.71 |  |

![Early and small](pretraining/predictivity_all/rq2_early_small.png)
<!-- END auto:early-decision -->

GitHub: [rq2_early_small.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/rq2_early_small.png) · [rq2_early_small.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/rq2_early_small.csv) ·
[rq2_decisions.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/rq2_decisions.csv) ·
[early_decision_facts.json](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/early_decision_facts.json)

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

<!-- BEGIN auto:panels (panels.py --pool predictivity_all) -->
## Per benchmark and per language

The decision table and the early read above, without pooling the benchmarks (`predictivity_all` pool). Language aggregates and per-subject facets are left out here (about 60 % of the pooled benchmark items remain), and a language's subplot averages its BPB item with its benchmark items. Regenerate with `python analysis/rq05_design_decisions/panels.py --pool predictivity_all`. White cells have no value; each figure's table sits next to it under the same name (`intervention_da_by_<unit>.csv`).

![rq05 in one figure](pretraining/predictivity_all/highlights.png)

![Decisions by proxy size and checkpoint](pretraining/predictivity_all/da_lines.png)

![The same on the items decided outside seed noise](pretraining/predictivity_all/da_lines_decided.png)

![Which depth wins, in seed sds](pretraining/predictivity_all/depth_crossover.png)

![Decisions by compute](pretraining/predictivity_all/da_lines_flops.png)

![Decisions per benchmark](pretraining/predictivity_all/intervention_da_by_benchmark.png)

![Early and small per benchmark](pretraining/predictivity_all/intervention_da_early_by_benchmark.png)

![Decisions per language](pretraining/predictivity_all/intervention_da_by_language.png)

![Early and small per language](pretraining/predictivity_all/intervention_da_early_by_language.png)
<!-- END auto:panels -->

**Key findings** (`da_lines`, `da_lines_decided`, `depth_crossover`;
population, sizes and reference as in the setup above: `predictivity_all`,
the reference per (intervention, L) the largest size trained at both levels,
items the reference ties dropped)

- Most decisions on the shared languages are not decisions at the reference:
  the effect table above puts depth at 1.0 seed sds on BPB and the language
  lists at 1.1, against temperature at 5.9 (benchmarks 1.1–1.4 for every
  intervention). `da_lines_decided` keeps only the items whose reference |Δ|
  clears 2 sds of the two-run difference; on the benchmark population that
  restriction does not help — the benchmarks' failure is not noise at the
  reference — while on `bpb_macro` and the loss it leaves one item per L,
  a 0/1 reading.
- Depth is a vanishing advantage, not a crossover (`depth_crossover.csv`):
  deep beats shallow by 4.5–11.0 difference sds at 175M, by |z| ≤ 0.44 at
  350M, shallow is ahead by ≤ 1.16 sds at 600M, and deep by 0.37–0.89 at 1B
  and 0.66–0.80 at 1.7B — all inside noise from 350M on. The depth DA of
  1.00 / 0.49 / 0.00 / 0.95 at 175M / 350M / 600M / 1B on BPB reads a
  reference that has no real preference.
- On every language's BPB (`bpb_all`, the languages only one level trains
  included) the decided items are read well, but that population is
  dominated by "the model that saw the language wins" —
  [rq06](../rq06_language_transfer/README.md).

**Follow-ups**

- Items are not independent: a language's BPB items move together, so 37
  languages agreeing is closer to one decision measured 37 times than to 37
  decisions. Report the number of decisions (intervention × L) that agree
  with the item share as the secondary number, and bootstrap over L.
- The reference changes between lines (600M or 1B for some interventions,
  1.7B for others; `refs` in the table): name it in the legend, and re-read
  every line against 1.7B once the missing 1.7B cells finish.
- The seed sd is the median over the replicated deep scheme-A cells applied
  to every size and scheme, each on 3 seeds (the median-of-sd is biased low
  by ~17 %); use the size's own sd where the ×3 cells exist at the
  reference's size and widen the decided cut to cover its sampling error.
- The training loss is one item per L, so its line is a 0/0.5/1 step
  function; drop it from the paper version. `da_lines_flops` (5 × 3 lines of
  60 cells) is not readable; one line per size, or per-size markers.

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/highlights.csv) ·
GitHub: [da_lines.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/da_lines.png) · [da_lines.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/da_lines.csv) ·
GitHub: [da_lines_decided.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/da_lines_decided.png) · [da_lines_decided.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/da_lines_decided.csv) ·
GitHub: [depth_crossover.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/depth_crossover.png) · [depth_crossover.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/depth_crossover.csv) ·
GitHub: [da_lines_flops.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/da_lines_flops.png) · [da_lines_flops.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/da_lines_flops.csv) ·
GitHub: [intervention_da_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/intervention_da_by_benchmark.png) · [intervention_da_by_benchmark.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/intervention_da_by_benchmark.csv) ·
GitHub: [intervention_da_early_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/intervention_da_early_by_benchmark.png) · [intervention_da_early_by_benchmark.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/intervention_da_early_by_benchmark.csv) ·
GitHub: [intervention_da_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/intervention_da_by_language.png) · [intervention_da_by_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/intervention_da_by_language.csv) ·
GitHub: [intervention_da_early_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/intervention_da_early_by_language.png) · [intervention_da_early_by_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/intervention_da_early_by_language.csv)

<!-- BEGIN auto:transformations (transformations.py --pool predictivity_all) -->
## Transformations on one item set

Mean decision accuracy over each transformation's pairs, on the items every transformation decides somewhere (benchmarks gated by rq00's above-random mask at the proxy and the reference); `transformation_da.csv` has every pair. Regenerate with `python analysis/rq05_design_decisions/transformations.py --pool predictivity_all`.

**benchmarks** (rows: transformation; columns: proxy size; mean over pairs, shared items):

| transformation | pairs (with data) | items | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| depth (deep vs shallow) | 6 (6) | 256 | 0.54 | 0.57 | 0.45 | 0.51 |
| language count (L vs next L) | 5 (5) | 254 | 0.64 | 0.62 | 0.52 | 0.55 |
| language lists (A vs B) | 3 (3) | 261 | 0.51 | 0.47 | 0.44 | 0.49 |
| temperature (T=1 vs T=3) | 3 (3) | 257 | 0.52 | 0.56 | 0.57 | 0.57 |

**per-language BPB (trained languages)** (rows: transformation; columns: proxy size; mean over pairs, shared items):

| transformation | pairs (with data) | items | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| depth (deep vs shallow) | 6 (4) | 26 | 1.00 | 0.49 | 0.00 | 0.96 |
| language count (L vs next L) | 5 (3) | 26 | 0.58 | 0.74 | 0.81 | 0.64 |
| language lists (A vs B) | 3 (3) | 26 | 0.40 | 1.00 | 0.94 | 0.77 |
| temperature (T=1 vs T=3) | 3 (1) | 26 | 0.81 | 0.96 | 0.96 | 0.96 |

![Transformations](pretraining/predictivity_all/transformation_da.png)
<!-- END auto:transformations -->

GitHub: [transformation_da.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/transformation_da.png) · [transformation_da.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_all/transformation_da.csv)

## Extensions from other sweeps

None. rq05 exists on the ladder only (`predictivity_all`, `predictivity_seeds`):
the 36-model sweep had one intervention (three FineWeb-edu mixtures) with no
depth, language-count, list or temperature axis, so no decision table of this
kind was made on it, and its numbers would not be pooled with the ladder's in
any case (a different harness, task set and reference size).

## Files

- `pretraining/<pool>/intervention_da.csv` — one row per (intervention,
  population, L, proxy size): `decision_acc`, `n_items`, mean |Δ| at proxy and
  reference, the level the reference prefers.
- `…/rq4_da_by_intervention.csv`, `rq4_effect_vs_seed.csv`, `rq4_interventions.png/.pdf`,
  `facts.json` — the paper's RQ4 figure and the numbers it quotes.
- `…/rq2_decisions.csv`, `rq2_early_small.csv`, `rq2_early_small.png/.pdf`,
  `early_decision_facts.json` — the paper's RQ2 figure (`early_decision.py`).
- `pretraining/<pool>/transformation_da.csv`, `transformation_da.png` — the four
  transformations (language count, temperature, depth, language lists) on one
  gated item set, so their predictability can be compared (`transformations.py`;
  the block "Transformations on one item set" below).
- `…/intervention_da.png`.
