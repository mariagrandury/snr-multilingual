# Design decisions — which proxy sizes rank a design decision like the reference, and how does that depend on the number of languages?

## Research question

> At a given number of languages L, when does a small model rank a design
> choice (depth, data scheme, temperature) the way the 1.7B reference
> (TARGET_SIZE, rule 9) does, and how does the answer move with L?
> This is the predictivity question of
> [`plan/small-to-large-predictivity-training-plan.md`](../../../../plan/small-to-large-predictivity-training-plan.md):
> the gate, scaling, decision-accuracy and noise analyses ask which *benchmarks*
> carry reliable signal, this one asks which *model sizes* do, and
> (`early_decision.py`) how early in the proxy's run.

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_seeds) -->
## Highlighted result

- **depth (deep vs shallow) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L8: 90M, L15: 90M, L30: 90M, L50: 90M.
- **data scheme (A vs B) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L8: 90M, L15: 90M, L30: 90M.
- **temperature (T=1 vs T=3) on per-language BPB** — smallest proxy reaching DA ≥ 0.75 against the reference: L15: 90M, L30: 90M, L50: 90M.
- **Depth decision on benchmarks** — mean DA over L by proxy: 90M 0.55, 175M 0.52, 350M 0.58, 600M 0.47, 1B 0.49.
- **Is there a decision to make?** median |Δ| at the reference in seed sds — depth (deep vs shallow): benchmarks 1.3×, bits per byte 1.3×; data scheme (A vs B, B at L8–L30): benchmarks 1.3×, bits per byte 1.6×; data scheme (A vs B, ZH at L2): benchmarks 1.0×; data scheme (A vs B, DCLMP at L1): benchmarks 1.4×; data scheme (A vs C, ES at L2): benchmarks 1.3×; data scheme (A vs C, FWEB at L1): benchmarks 1.8×; temperature (T=1 vs T=3): benchmarks 1.4×, bits per byte 5.4×.
<!-- END auto:highlight -->

## Experimental setup

The grid is the predictivity ladder: sizes 90M–1.7B (non-embedding), language
settings L ∈ {1, 2, 8, 15, 30, 50}, and four interventions on the design axes
([RULES.md](../RULES.md), Definitions), each read against the deep, scheme-A,
T = 1 baseline (the data-A cell): model depth (deep, width/depth ≈ 64, vs
shallow, ≈ 128, at equal non-embedding size); data scheme A vs B and A vs C at
T = 1, read per L because a letter names a different recipe at each L (B: the
diversity-first lists at L8–L30, ZH at L2, DCLMP at L1; C: ES at L2, FWEB at
L1) — a figure or table that averages over L draws a scheme decision per build
(`analyze.by_recipe`), and the early-decision heat map keeps the planned list
decision (A vs B at L8–L30); and temperature T = 1 vs T = 3 at scheme A (AT3 at
L15, L30, L50).

The data interventions are read on the single data-scheme axis (scheme × T) of
the refresh of 2026-10-06. The swiglu ladder is in the pool but moves the
activation, which is not one of the four interventions.

The pool is `predictivity_seeds` (every seed, ladder and data build), gated with
`predictivity`'s mask, and every read uses each cell's final checkpoint
(D = 100·N tokens, WSD-annealed) unless it says otherwise. The reference at each
L is TARGET_SIZE, 1.7B (rule 9), which all 16 (intervention, L) settings have
at both levels; the proxies are 90M, 175M, 350M, 600M and 1B.

The diverged batch-504 90M and 175M runs (see
[`plan/90M-rung-anomaly.md`](../../../../plan/90M-rung-anomaly.md)) are dropped
at load in favour of their batch-84 / batch-168 retrains (rule 10), and runs
that have not reached their target are excluded by the loader.

Populations for the decision: per-language BPB on the languages both levels
train (`bpb_trained`, the plan's primary outcome), the benchmark tasks the
cell was evaluated on (`benchmark`, gated at the proxy and at the cell's
reference, rule 1), and the single macro-BPB decision
(`bpb_macro`, the "aggregate criterion" the plan asks to compare against the
per-language one).

## Key figure

![Design decisions read by proxy size and by the reference's checkpoints](pretraining/predictivity_seeds/da_all_lines_mono_axis_paper.png)

Population: pool `predictivity_seeds` (every seed, ladder and data build; benchmarks gated with `predictivity`'s mask), DA-size (left) and DA-ckpt (right) on mono-axis decisions of the four interventions (a scheme decision per data build), proxies 90M–1B at their final checkpoint (left) and the reference's own nine earlier checkpoints, 0.5C–4.5C (right), each line against one reference size and the L's that share it, mean over L; solid = per-language BPB of the languages both levels train, dashed = the gated benchmark tasks; dotted line = 0.75.

**Key finding (2026-10-07 refresh).** Per-language BPB of the trained languages reads the temperature decision from every proxy (0.95–0.99, mean over L15–L50) and the language-list decision at 0.77–1.00 (L8–L30), while on the gated benchmarks only two proxies read a decision at 0.75, both L1 English-data swaps (DCLMP at 350M, 0.79; FWEB at 600M, 0.75): 0.45–0.64 for every intervention but those two swaps (DCLMP 0.50–0.79, FWEB 0.57–0.75).

The depth line on per-language BPB swings from 0.00 (600M) to 0.98 (175M) against a reference whose depth effect is 1.3 seed sds, and the reference's own checkpoints reach 0.76–0.85 on the benchmarks at 90 % of its run, where before 80 % only the L1 FWEB swap reaches 0.75.

GitHub: [da_all_lines_mono_axis_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_mono_axis_paper.png) · [da_all_lines_mono_axis_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_mono_axis_paper.csv). The same on the decided items and per benchmark: [Per benchmark and per language](#per-benchmark-and-per-language).

## Methodology

- **Intervention decision accuracy.** For each item of a population, the
  decision is which level of the intervention is better; DA(proxy, L) is the
  fraction of items on which the proxy agrees with the reference.
  With two levels the pairwise-ranking definition of Heineman et al. (2025)
  reduces to sign agreement on the two models of one item, which is
  `snr.metrics.decision_acc_fast` per item; items the reference ties are
  dropped rather than counted as misses, since they leave no decision to
  agree with (the one place this analysis departs from the kernel).
- **Items per cell.** `n_items` is reported with every cell and a benchmark
  cell needs ≥ 3 items. At the final checkpoint a benchmark cell holds 14–1,252
  gated tasks (14–128 at L1, 1,107–1,252 at L50), and a `bpb_trained` cell
  5–50 languages (5, 6 and 26 for the language lists at L8, L15, L30).
- **Effect at the reference.** Per intervention and L, the median |Δ| at
  the reference in per-task seed standard deviations (the seed sd of the
  baseline deep data-A cells with replicates): "is there a decision to make?". The two
  other reads this folder used to carry live with their themes: the
  scaling-law error in the scaling analysis (`scaling_law_error.py`) and the effect against
  seed and checkpoint noise per cell in the noise-and-SNR analysis (`effect_vs_noise.py`).

Hand-written numbers in this README are from the ladder-report snapshot
**2026-10-07 15:51** (commit 7966367c), re-read on 2026-10-07. The bBPB twins
now exist at every checkpoint of every size, so every proxy's benchmark cell
holds the same task set: before this refresh the 90M, 350M and 600M cells
lacked the twins (288 tasks at 90M, L50, against 1,107 now), which is why
their benchmark DAs moved while 175M, 1B and the 1.7B checkpoints did not.

<!-- BEGIN auto:results (analyze.py --pool predictivity_seeds) -->
## Results

Numbers from the `predictivity_seeds` pool. Regenerate with `python analysis/rq05_design_decisions/analyze.py --pool predictivity_seeds`.

**depth (deep vs shallow), per-language BPB** (rows: proxy size; columns: L; reference L8 → 1.7B, L15 → 1.7B, L30 → 1.7B, L50 → 1.7B):

| proxy | L8 | L15 | L30 | L50 |
|---|---|---|---|---|
| 90M | 1.00 | 1.00 | 0.97 | 0.92 |
| 175M | 1.00 | 1.00 | 1.00 | 0.94 |
| 350M | 0.25 | 0.00 | 0.97 | 0.74 |
| 600M | 0.00 | 0.00 | 0.00 | 0.00 |
| 1B | 1.00 | 1.00 | 1.00 | 0.78 |

**data scheme (A vs B), per-language BPB** (rows: proxy size; columns: L; reference L8 → 1.7B, L15 → 1.7B, L30 → 1.7B):

| proxy | L8 | L15 | L30 |
|---|---|---|---|
| 90M | 1.00 | 1.00 | 1.00 |
| 175M | 1.00 | 1.00 | 1.00 |
| 350M | 1.00 | 1.00 | 1.00 |
| 600M | 1.00 | 0.83 | 1.00 |
| 1B | 1.00 | 1.00 | 0.31 |

**temperature (T=1 vs T=3), per-language BPB** (rows: proxy size; columns: L; reference L15 → 1.7B, L30 → 1.7B, L50 → 1.7B):

| proxy | L15 | L30 | L50 |
|---|---|---|---|
| 90M | 0.87 | 1.00 | 1.00 |
| 175M | 1.00 | 1.00 | 0.98 |
| 350M | 0.87 | 1.00 | 0.98 |
| 600M | 1.00 | 0.97 | 0.98 |
| 1B | 0.93 | 1.00 | 0.98 |

**depth (deep vs shallow), benchmarks** (rows: proxy size; columns: L; reference L1 → 1.7B, L2 → 1.7B, L8 → 1.7B, L15 → 1.7B, L30 → 1.7B, L50 → 1.7B):

| proxy | L1 | L2 | L8 | L15 | L30 | L50 |
|---|---|---|---|---|---|---|
| 90M | 0.53 | 0.53 | 0.53 | 0.58 | 0.57 | 0.56 |
| 175M | 0.55 | 0.58 | 0.52 | 0.47 | 0.55 | 0.47 |
| 350M | 0.86 | 0.55 | 0.49 | 0.54 | 0.53 | 0.51 |
| 600M | 0.48 | 0.45 | 0.47 | 0.48 | 0.54 | 0.43 |
| 1B | 0.48 | 0.44 | 0.47 | 0.50 | 0.56 | 0.49 |

**data scheme (A vs B), benchmarks** (rows: proxy size; columns: L; reference L1 → 1.7B, L2 → 1.7B, L8 → 1.7B, L15 → 1.7B, L30 → 1.7B):

| proxy | L1 | L2 | L8 | L15 | L30 |
|---|---|---|---|---|---|
| 90M | 0.60 | 0.48 | 0.44 | 0.44 | 0.49 |
| 175M | 0.58 | 0.49 | 0.55 | 0.48 | 0.41 |
| 350M | 0.79 | 0.51 | 0.45 | 0.51 | 0.42 |
| 600M | 0.52 | 0.57 | 0.48 | 0.46 | 0.49 |
| 1B | 0.50 | 0.49 | 0.50 | 0.55 | 0.50 |

**data scheme (A vs C), benchmarks** (rows: proxy size; columns: L; reference L1 → 1.7B, L2 → 1.7B):

| proxy | L1 | L2 |
|---|---|---|
| 90M | 0.57 | 0.51 |
| 175M | 0.68 | 0.52 |
| 350M | 0.71 | 0.52 |
| 600M | 0.75 | 0.48 |
| 1B | 0.74 | 0.45 |

**temperature (T=1 vs T=3), benchmarks** (rows: proxy size; columns: L; reference L15 → 1.7B, L30 → 1.7B, L50 → 1.7B):

| proxy | L15 | L30 | L50 |
|---|---|---|---|
| 90M | 0.56 | 0.63 | 0.72 |
| 175M | 0.58 | 0.61 | 0.68 |
| 350M | 0.59 | 0.61 | 0.67 |
| 600M | 0.59 | 0.60 | 0.67 |
| 1B | 0.56 | 0.65 | 0.64 |

![Intervention DA grid](pretraining/predictivity_seeds/intervention_da_all_mono_axis.png)

**Effect at the reference in seed standard deviations** (median over items and L; a scheme decision per recipe, since the letter names a different build at each L):

| intervention | benchmarks | bits per byte |
|---|---|---|
| depth (deep vs shallow) | 1.3 | 1.3 |
| data scheme (A vs B, B at L8–L30) | 1.3 | 1.6 |
| data scheme (A vs B, ZH at L2) | 1.0 |  |
| data scheme (A vs B, DCLMP at L1) | 1.4 |  |
| data scheme (A vs C, ES at L2) | 1.3 |  |
| data scheme (A vs C, FWEB at L1) | 1.8 |  |
| temperature (T=1 vs T=3) | 1.4 | 5.4 |

![Interventions](pretraining/predictivity_seeds/rq4_interventions.png)
<!-- END auto:results -->

GitHub: [intervention_da_all_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_all_mono_axis.png) · [intervention_da_all_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_all_mono_axis.csv) ·
GitHub: [rq4_interventions.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq4_interventions.png) · [rq4_interventions.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq4_interventions.csv) ·
[rq4_effect_vs_seed.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq4_effect_vs_seed.csv) ·
[rq4_da_size_by_intervention_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq4_da_size_by_intervention_mono_axis.csv)

**Key findings** (`intervention_da_all_mono_axis`, `rq4_effect_vs_seed`; pool
`predictivity_seeds`, proxies 90M–1B at their final checkpoint, reference
1.7B at every (intervention, L), DA-size per L)

- On per-language BPB the temperature decision is read at 0.87–1.00 by every
  proxy at every L, and the language-list decision at 0.83–1.00 except 1B at
  L30 (0.31).
- The depth decision on per-language BPB is 0.00 for 600M at all four L and
  0.00–0.97 for 350M, while 90M, 175M and 1B read it at 0.78–1.00; the depth
  bullet under the per-benchmark figures explains why.
- On the gated benchmarks the per-L cells span 0.41–0.86, and the only ones at
  ≥ 0.75 are at L1: 350M (depth 0.86, A vs DCLMP 0.79) and 600M (A vs FWEB 0.75).
- The effect at the reference is 1.0–1.8 seed sds on the benchmarks for every
  intervention; on per-language BPB it is 1.3 for depth, 1.6 for the language
  lists and 5.4 for temperature, the one decision far outside seed noise.

**Follow-ups**

- Bootstrap the per-L benchmark DA over tasks: the L1 cells hold 14–128
  tasks, so the three ≥ 0.75 cells there need an interval before they are quoted.

## How small, and how early

### Setup

Everything comes from the decision table above: with two levels,
decision accuracy is the share of population items on which the proxy
prefers the level the reference prefers. The reference is TARGET_SIZE (1.7B,
rule 9) at its final checkpoint — an (intervention, L) without a 1.7B cell at
both levels is skipped (none is in this refresh); the proxy is every smaller
size, read at its ten evaluated checkpoints, 0.5C to 5C of training (C = the
Chinchilla-optimal 20 tokens per parameter; 5C is the full run), and the 1.7B row
is the reference's own checkpoints against its final ranking (DA-ckpt).

One grid answers both halves of the question, how small and how early. Two
populations, each cell needing at least three items: the per-language BPB of
the languages both levels train, and the gated benchmark tasks both levels
were evaluated on.

### Methodology

The per-(intervention, L, population, proxy size, fraction) rows of the two
planned decisions, depth and the A vs B language lists, are averaged over L so
that every language setting counts once. `cells` in the table is the number of
settings behind a mean (depth: 4 on BPB, L8–L50, and 6 on benchmarks, L1–L50;
lists: 3, L8–L30); the reference, 1.7B for all of them, is carried in `refs`.

<!-- BEGIN auto:early-decision (early_decision.py --pool predictivity_seeds) -->
## How small, and how early (paper RQ2)

Numbers from the `predictivity_seeds` decision table above. Regenerate with `python analysis/rq05_design_decisions/early_decision.py --pool predictivity_seeds`.

- **depth (deep vs shallow), per-language bits per byte** — final-checkpoint agreement by proxy: 90M 0.97, 175M 0.98, 350M 0.49, 600M 0.00, 1B 0.95; smallest proxy at ≥ 0.75: **90M**, which reaches it at 0.5C of training (5C = the full run).
- **depth (deep vs shallow), benchmark tasks** — final-checkpoint agreement by proxy: 90M 0.55, 175M 0.52, 350M 0.58, 600M 0.47, 1B 0.49; no proxy reaches 0.75.
- **data scheme (A vs B, B at L8–L30), per-language bits per byte** — final-checkpoint agreement by proxy: 90M 1.00, 175M 1.00, 350M 1.00, 600M 0.94, 1B 0.77; smallest proxy at ≥ 0.75: **90M**, which reaches it at 0.5C of training (5C = the full run).
- **data scheme (A vs B, B at L8–L30), benchmark tasks** — final-checkpoint agreement by proxy: 90M 0.46, 175M 0.48, 350M 0.46, 600M 0.48, 1B 0.52; no proxy reaches 0.75.

**depth (deep vs shallow) — per-language bits per byte** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.97 | 0.97 | 0.98 | 0.94 | 0.93 | 0.96 | 0.95 | 0.93 | 0.96 | 0.97 |
| 175M | 0.94 | 0.86 | 0.90 | 0.98 | 0.93 | 0.95 | 0.92 | 0.93 | 0.96 | 0.98 |
| 350M | 0.25 | 0.26 | 0.20 | 0.17 | 0.36 | 0.45 | 0.29 | 0.31 | 0.44 | 0.49 |
| 600M | 0.19 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 |
| 1B | 0.27 | 0.28 | 0.54 | 0.83 | 0.85 | 0.87 | 0.90 | 0.92 | 0.94 | 0.95 |
| 1.7B | 0.69 | 0.98 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 |  |

**depth (deep vs shallow) — benchmark tasks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.54 | 0.52 | 0.53 | 0.48 | 0.51 | 0.52 | 0.54 | 0.52 | 0.56 | 0.55 |
| 175M | 0.53 | 0.51 | 0.49 | 0.52 | 0.50 | 0.52 | 0.53 | 0.50 | 0.52 | 0.52 |
| 350M | 0.47 | 0.49 | 0.46 | 0.52 | 0.51 | 0.53 | 0.57 | 0.55 | 0.55 | 0.58 |
| 600M | 0.46 | 0.47 | 0.45 | 0.47 | 0.49 | 0.47 | 0.48 | 0.46 | 0.47 | 0.47 |
| 1B | 0.46 | 0.49 | 0.47 | 0.48 | 0.48 | 0.49 | 0.52 | 0.55 | 0.49 | 0.49 |
| 1.7B | 0.51 | 0.47 | 0.53 | 0.56 | 0.59 | 0.61 | 0.64 | 0.64 | 0.76 |  |

**data scheme (A vs B, B at L8–L30) — per-language bits per byte** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.96 | 0.97 | 0.97 | 0.97 | 0.97 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| 175M | 0.96 | 0.99 | 1.00 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 350M | 0.86 | 0.88 | 0.94 | 0.94 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 |
| 600M | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.87 | 0.94 | 0.88 | 0.94 | 0.94 |
| 1B | 0.79 | 0.91 | 0.87 | 0.83 | 0.83 | 0.83 | 0.81 | 0.72 | 0.77 | 0.77 |
| 1.7B | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |  |

**data scheme (A vs B, B at L8–L30) — benchmark tasks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples, 5C = the full run; mean over L of the per-L agreement):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.49 | 0.52 | 0.51 | 0.46 | 0.49 | 0.51 | 0.47 | 0.44 | 0.46 | 0.46 |
| 175M | 0.48 | 0.53 | 0.47 | 0.51 | 0.50 | 0.41 | 0.48 | 0.45 | 0.46 | 0.48 |
| 350M | 0.48 | 0.45 | 0.45 | 0.48 | 0.44 | 0.46 | 0.47 | 0.44 | 0.47 | 0.46 |
| 600M | 0.49 | 0.48 | 0.52 | 0.50 | 0.48 | 0.51 | 0.50 | 0.48 | 0.48 | 0.48 |
| 1B | 0.51 | 0.48 | 0.51 | 0.51 | 0.49 | 0.48 | 0.48 | 0.54 | 0.53 | 0.52 |
| 1.7B | 0.48 | 0.48 | 0.51 | 0.60 | 0.59 | 0.61 | 0.60 | 0.62 | 0.76 |  |

![Early and small](pretraining/predictivity_seeds/rq2_da_goal_early_small_mono_axis.png)
<!-- END auto:early-decision -->

GitHub: [rq2_da_goal_early_small_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq2_da_goal_early_small_mono_axis.png) · [rq2_da_goal_early_small_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq2_da_goal_early_small_mono_axis.csv) ·
[rq2_da_all_decisions_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/rq2_da_all_decisions_mono_axis.csv) ·
[early_decision_facts.json](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/early_decision_facts.json)

**Key findings** (`rq2_da_goal_early_small_mono_axis`; pool `predictivity_seeds`,
proxies 90M–1B at ten checkpoints, 0.5C–5C, against the 1.7B final ranking,
mean over L; the 1.7B row is DA-ckpt)

- On per-language BPB the smallest proxy is enough and early is enough: 90M
  reads the depth decision at 0.93–0.98 and the list decision at 0.96–1.00 from
  0.5C on.
- Larger proxies are not safer on BPB: the 1B depth read climbs from 0.27 at
  0.5C to 0.95 at 5C, the 1B list read falls from 0.91 at 1C to 0.72–0.77 from 4C on, and 600M
  reads depth at 0.00–0.19 throughout.
- On the gated benchmarks no proxy at any checkpoint reaches 0.75 (0.41–0.58
  over both decisions), and the reference's own checkpoints sit at 0.47–0.64
  until they reach 0.76 at 4.5C.

**Follow-ups**

- The benchmark rows now carry the bBPB twins at every checkpoint (a depth
  proxy row holds 3,148–3,535 task-cells, against 793–1,735 at 90M–600M
  before), and the 0.41–0.58 range did not move, so the twins do not rescue
  the early benchmark decision; split the rows by twin vs native task to check
  that neither half alone reaches 0.75.

## Caveats to carry into the paper

- The depth intervention's effect is small by design (aspect ratio near the
  optimum); the effect-vs-noise table decides whether its DA is interpretable
  at all. A scheme intervention can change the language set (A vs B at
  L8–L30, A vs ZH/ES at L2), so its `bpb_trained` population is the languages
  both builds train.
- Checkpoint noise windows differ by size unless the shared grid is used
  (the loader's default): 5 late checkpoints span the final 25 % of a
  20-checkpoint run and 12.5 % of a 40-checkpoint one
  ([`plan/1b-models.md`](../../../../plan/1b-models.md)).
- The reference is TARGET_SIZE, 1.7B (rule 9); an (intervention, L) without a
  1.7B cell at both levels is skipped. In the 2026-10-07 refresh all 16
  (intervention, L) settings have one, and the `reference_size` column names
  it per cell.

<!-- BEGIN auto:panels (panels.py --pool predictivity_seeds) -->
## Per benchmark and per language

The decision table and the early read above, without pooling the benchmarks (`predictivity_seeds` pool). Language aggregates and per-subject facets are left out here (about 60 % of the pooled benchmark items remain), and a language's subplot averages its BPB item with its benchmark items. Regenerate with `python analysis/rq05_design_decisions/panels.py --pool predictivity_seeds`. White cells have no value; each figure's table sits next to it under the same name (`intervention_da_size_by_<unit>_mono_axis.csv`).

![rq05 in one figure](pretraining/predictivity_seeds/highlights.png)

![Decisions by proxy size and checkpoint](pretraining/predictivity_seeds/da_all_lines_mono_axis.png)

![The same on the items decided outside seed noise](pretraining/predictivity_seeds/da_all_lines_decided_mono_axis.png)

![Which depth wins, in seed sds](pretraining/predictivity_seeds/depth_crossover.png)

![Decisions by compute](pretraining/predictivity_seeds/da_all_lines_flops_mono_axis.png)

![Decisions per benchmark](pretraining/predictivity_seeds/intervention_da_size_by_benchmark_mono_axis.png)

![Early and small per benchmark](pretraining/predictivity_seeds/intervention_da_goal_early_by_benchmark_mono_axis.png)

![Decisions per language](pretraining/predictivity_seeds/intervention_da_size_by_language_mono_axis.png)

![Early and small per language](pretraining/predictivity_seeds/intervention_da_goal_early_by_language_mono_axis.png)
<!-- END auto:panels -->

**Key findings** (`da_all_lines_mono_axis`, `da_all_lines_decided_mono_axis`, `depth_crossover`;
DA-size and DA-ckpt on mono-axis decisions, final checkpoint unless stated;
population, sizes and reference as in the setup above: `predictivity_seeds`,
the reference per (intervention, L) TARGET_SIZE, 1.7B (rule 9),
items the reference ties dropped)

- Most decisions on the shared languages are not decisions at the reference:
  the effect table puts depth at 1.3 seed sds on BPB (per-L medians
  1.15–1.35) and the language lists at 1.6 (0.59–2.87), against temperature
  at 5.4 (4.36–7.53). On the benchmarks every intervention sits at 1.0–1.8.
- `da_all_lines_decided_mono_axis` keeps only the items whose reference |Δ|
  clears 2 sds of the two-run difference: on the 73 gated benchmark cells
  (proxy × intervention × L) that keep ≥ 3 decided tasks, the final-checkpoint
  DA rises from 0.54 to 0.61 (means, a median of 21 decided tasks per cell).
  That is still short of 0.75, so noise at the reference explains part of the
  benchmarks' failure but not most of it.
- On per-language BPB the decided cut keeps 10–41 languages per cell for
  temperature but at most 2 for depth, so the decided figure has no depth BPB
  line; on `bpb_macro` and the loss it leaves one item per L, a 0/1 reading.
- Depth is a vanishing advantage, not a crossover (`depth_crossover.csv`,
  the batch-84 / batch-168 retrains): deep beats shallow by 1.9–2.1
  difference sds at 90M and 1.1–1.6 at 175M, by |z| ≤ 0.51 at 350M, shallow
  is ahead by 0.28–1.46 sds at 600M, and deep by 0.48–1.03 at 1B and 0.81–0.95
  at 1.7B, all inside 2 sds from 350M on. The depth DA on the trained
  languages' BPB (0.92–1.00 at 90M, 0.94–1.00 at 175M, 0.00–0.97 at 350M,
  0.00 at 600M, 0.78–1.00 at 1B over L8–L50) therefore reads a reference that
  has no real preference.
- Every language's BPB, the languages only one level trains included, is
  not a population here (rule 2: a score on an untrained language is the
  [language-transfer analysis](../rq06_language_transfer/README.md)'s
  measurement, and the loader no longer delivers those rows).

**Follow-ups**

- Items are not independent: a language's BPB items move together, so 50
  languages agreeing at L50 is closer to one decision measured 50 times than
  to 50 decisions. Report the number of decisions (intervention × L) that agree
  with the item share as the secondary number, and bootstrap over L.
- The reference is 1.7B on every line since 2026-09-26 (`refs` in the
  table); keep naming it in the legend.
- The seed sd is the median over the replicated deep data-A cells applied
  to every size and scheme, on 3 seeds where they exist (the median-of-sd is
  biased low by ~17 %); use the size's own sd where the ×3 cells exist at the
  reference's size and widen the decided cut to cover its sampling error.
- The training loss is one item per L, so its line is a 0/0.5/1 step
  function; drop it from the paper version. `da_all_lines_flops_mono_axis` (17 lines of
  59 points) is not readable; one line per size, or per-size markers.

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/highlights.csv) ·
GitHub: [da_all_lines_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_mono_axis.png) · [da_all_lines_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_mono_axis.csv) ·
GitHub: [da_all_lines_decided_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_decided_mono_axis.png) · [da_all_lines_decided_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_decided_mono_axis.csv) ·
GitHub: [depth_crossover.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/depth_crossover.png) · [depth_crossover.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/depth_crossover.csv) ·
GitHub: [da_all_lines_flops_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_flops_mono_axis.png) · [da_all_lines_flops_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_flops_mono_axis.csv) ·
GitHub: [intervention_da_size_by_benchmark_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_size_by_benchmark_mono_axis.png) · [intervention_da_size_by_benchmark_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_size_by_benchmark_mono_axis.csv) ·
GitHub: [intervention_da_goal_early_by_benchmark_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_goal_early_by_benchmark_mono_axis.png) · [intervention_da_goal_early_by_benchmark_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_goal_early_by_benchmark_mono_axis.csv) ·
GitHub: [intervention_da_size_by_language_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_size_by_language_mono_axis.png) · [intervention_da_size_by_language_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_size_by_language_mono_axis.csv) ·
GitHub: [intervention_da_goal_early_by_language_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_goal_early_by_language_mono_axis.png) · [intervention_da_goal_early_by_language_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/intervention_da_goal_early_by_language_mono_axis.csv)

<!-- BEGIN auto:transformations (transformations.py --pool predictivity_seeds) -->
## Transformations on one item set

Mean decision accuracy over each transformation's pairs, on the items every transformation decides somewhere (benchmarks gated by rq00's above-random mask at the proxy and the reference); `transformation_da_size_mono_axis.csv` has every pair. Regenerate with `python analysis/rq05_design_decisions/transformations.py --pool predictivity_seeds`.

**benchmarks** (rows: transformation; columns: proxy size; mean over pairs, shared items):

| transformation | pairs (with data) | items | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|---|
| data scheme (A vs B) | 5 (5) | 129 | 0.47 | 0.54 | 0.54 | 0.50 | 0.53 |
| data scheme (A vs C) | 2 (2) | 128 | 0.54 | 0.60 | 0.62 | 0.61 | 0.59 |
| depth (deep vs shallow) | 6 (6) | 128 | 0.51 | 0.52 | 0.57 | 0.45 | 0.46 |
| language count (L vs next L) | 5 (5) | 129 | 0.48 | 0.57 | 0.60 | 0.48 | 0.51 |
| temperature (T=1 vs T=3) | 3 (3) | 129 | 0.48 | 0.49 | 0.49 | 0.42 | 0.42 |

**per-language BPB (trained languages)** (rows: transformation; columns: proxy size; mean over pairs, shared items):

| transformation | pairs (with data) | items |
|---|---|---|

![Transformations](pretraining/predictivity_seeds/transformation_da_size_mono_axis.png)
<!-- END auto:transformations -->

GitHub: [transformation_da_size_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/transformation_da_size_mono_axis.png) · [transformation_da_size_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq05_design_decisions/pretraining/predictivity_seeds/transformation_da_size_mono_axis.csv)

**Key findings** (`transformation_da_size_mono_axis`; pool `predictivity_seeds`,
DA-size of proxies 90M–1B against 1.7B, mono-axis pairs, 128–129 gated
benchmark tasks shared by every transformation)

- On one task set no transformation is read reliably by any proxy: the means
  span 0.42 (temperature, 600M and 1B) to 0.62 (A vs C, 350M), and the
  language-count pairs (L vs next L) sit at 0.48–0.60 like the four design
  interventions.
- The per-language BPB table is empty in this refresh: the CSV carries only
  `bpb_all` rows (every language), not the trained-language population.

**Follow-ups**

- Have `transformations.py` write the trained-language BPB rows, so the BPB
  side of this comparison exists on one language set.

## Extensions from other sweeps

None. The design-decision analysis exists on the ladder only (`predictivity_seeds`):
the 36-model sweep had one intervention (three FineWeb-edu mixtures) with no
depth, language-count, list or temperature axis, so no decision table of this
kind was made on it, and its numbers would not be pooled with the ladder's in
any case (a different harness, task set and reference size).

## Files

- `pretraining/<pool>/intervention_da_all_mono_axis.csv` — one row per (intervention,
  population, L, proxy size): `decision_acc`, `n_items`, mean |Δ| at proxy and
  reference, the level the reference prefers.
- `…/rq4_da_size_by_intervention_mono_axis.csv`, `rq4_effect_vs_seed.csv`, `rq4_interventions.png`,
  `facts.json` — the paper's design-decision figure and the numbers it quotes.
- `…/rq2_da_all_decisions_mono_axis.csv`, `rq2_da_goal_early_small_mono_axis.csv`, `rq2_da_goal_early_small_mono_axis.png`,
  `early_decision_facts.json` — the paper's how-small-and-how-early figure (`early_decision.py`).
- `pretraining/<pool>/transformation_da_size_mono_axis.csv`, `transformation_da_size_mono_axis.png` — the five
  transformations (language count, temperature, depth, data scheme A vs B and A vs C) on one
  gated item set, so their predictability can be compared (`transformations.py`;
  the block "Transformations on one item set" above).
- `…/intervention_da_all_mono_axis.png`.
