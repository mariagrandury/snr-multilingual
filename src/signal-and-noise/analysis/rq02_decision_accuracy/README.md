# RQ2 — Does a benchmark rank models at a small size the way the reference size does?

## Research question

> Decision accuracy (DA) is the ground truth of the whole framework: the
> probability that a benchmark orders a pair of models the way an evaluation
> of larger models would (Heineman et al., 2025). Which benchmarks, in which
> languages, keep the ranking of the ladder's design variants across sizes
> (**DA-size**) and across training (**DA-ckpt**)?

<!-- BEGIN auto:highlight (da_per_benchmark.py --pool predictivity) -->
## Highlighted result

- **DA-size, proxy → 1.7B** (mean over the above-random benchmark tasks / over the per-language BPB tasks): 175M → 1.7B 0.56 / 0.72; 350M → 1.7B 0.56 / 0.96; 600M → 1.7B 0.58 / 0.51; 1B → 1.7B 0.58 / 0.84.
- **DA-size of `bpb_macro`** (one task, kept out of the means above): 175M 0.78; 350M 0.93; 600M 0.93; 1B 0.94.
- **DA-size of `train_loss`** (one task, kept out of the means above): 175M 0.83; 350M 0.88; 600M 0.82; 1B 0.82.
- **DA-ckpt** (early checkpoint vs final, above-random benchmark tasks): highest at 175M 90 % (0.87).
<!-- END auto:highlight -->

## Experimental setup

Models are the predictivity ladder's cells (`configs/models.json` pool
`predictivity`: 90M–1.7B × L ∈ {1, 2, 8, 15, 30, 50, 100} × deep/shallow ×
scheme A/B, seed 1904; `predictivity_seeds` adds the seed replicates as
separate models). The cross-size identity is the cell's `family`
(`lm-L8-schemeB-deep-seed1904`: everything but the size), so a design variant
present at two sizes is one pair.

- **DA-size** — `decision_acc_size_<small>`: the families' ranking at
  `<small>`'s final checkpoint vs at the reference size's (1.7B) final checkpoint;
  `decision_acc_size_<a>_to_<b>` for every other bucket pair with ≥ 2 shared
  families (90M→175M … 1B→1.7B). Multilingual tasks are only evaluated on
  cells that train the language, so each task's pair set is the families
  that exist at both sizes *and* were evaluated on it.
- **DA-ckpt** — `decision_acc_ckpt_f<frac>_<size>`: within one size, the
  ranking at the checkpoint nearest 20/40/60/80 % of each run (1×C … 4×C on
  the WSD schedule) vs at the final checkpoint.

Per-language BPB (`bpb_<subset>`) and the training loss are tasks too, so DA
is computed for the plan's outcome metric alongside the benchmarks. DA is
never gated: it is the truth the SNR proxies of rq03 and rq04 are scored against.

## Methodology

[`compute_da.py`](compute_da.py) writes `pretraining/<pool>/da_per_task.csv`
(one row per parent task, one column per DA definition) from
`snr.metrics.decision_acc_fast`; [`da_per_benchmark.py`](da_per_benchmark.py)
melts it into a long (language, benchmark, comparison) table and the wide
`_size` / `_ckpt` pivots, and rewrites the deck's appendix slides for the
canonical pool. rq03 joins the SNR variants onto this table; rq05 asks the
complementary question — which proxy *size* ranks an intervention like the
reference, with languages as the population.

With few families at a bucket (the 1.7B row has five language settings; a
task in one language may have three), DA is quantised to 1/#pairs: read the
family-level averages, and `n` alongside every value
(`da_n_pairs_per_task.csv`).

<!-- BEGIN auto:results (da_per_benchmark.py --pool predictivity) -->
## Results

Numbers from the `predictivity` pool (`da_per_task.csv`, pairs from `da_n_pairs_per_task.csv`, gate from rq00). Regenerate with `python analysis/rq02_decision_accuracy/da_per_benchmark.py --pool predictivity`.

**DA-size by proxy size** (`n` tasks; median pairs per cell):

| comparison | benchmarks | n | pairs | BPB | n |
|---|---|---|---|---|---|
| 175M → 1.7B | 0.56 | 64 | 21 | 0.72 | 26 |
| 350M → 1.7B | 0.56 | 80 | 21 | 0.96 | 26 |
| 600M → 1.7B | 0.58 | 85 | 21 | 0.51 | 26 |
| 1B → 1.7B | 0.58 | 91 | 45 | 0.84 | 26 |

![DA-size by family](pretraining/predictivity/da_size_by_family.png)

**DA-ckpt by bucket and fraction of the run** (mean over the above-random benchmark tasks):

| bucket | 10 % | 20 % | 30 % | 40 % | 50 % | 60 % | 70 % | 80 % | 90 % |
|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.59 | 0.58 | 0.59 | 0.73 | 0.75 | 0.73 | 0.76 | 0.81 | 0.87 |
| 350M | 0.52 | 0.57 | 0.61 | 0.61 | 0.64 | 0.65 | 0.71 | 0.76 | 0.83 |
| 600M | 0.54 | 0.57 | 0.58 | 0.60 | 0.59 | 0.64 | 0.62 | 0.67 | 0.79 |
| 1B | 0.51 | 0.58 | 0.61 | 0.59 | 0.62 | 0.65 | 0.66 | 0.69 | 0.76 |
| 1.7B | 0.56 | 0.60 | 0.60 | 0.64 | 0.62 | 0.63 | 0.63 | 0.68 | 0.74 |
<!-- END auto:results -->

## Departure from upstream: tie handling in `decision_acc_fast`

Upstream's kernel (`snr/metrics.py`, kept verbatim as
`decision_acc_fast_upstream`) compares `a > b` on both vectors, so a pair the
proxy ties but the target decides counts as an agreement or a disagreement
depending on which model is listed first — the value depends on the listing
order (alphabetical by family here), not on the benchmark. Since 2026-09-16
`decision_acc_fast` compares the sign of the difference on both sides
instead: tied in both agrees, tied in one and decided in the other is a miss,
and the result is order-invariant. Without ties the two are identical.

Measured on the `predictivity` pool at the final checkpoints (2026-09-16):
3,301 of 119,456 family pairs are exact ties (2.76 %); 4.05 % of benchmark
pairs, where accuracy over a fixed item count ties easily, and none of the
per-language BPB or loss pairs; 1,225 of 2,845 (task, size) cells contain at
least one tie. The tables committed on 2026-09-16 were produced with the
upstream kernel and are regenerated by `run_all_predictivity.sh`.

## Preliminary findings (ladder snapshot, 2026-09-01)

Computed on the eval results of the ≤ 600M ladder before this pipeline existed
(`plan/status-09-01.md`, §4): ranking the language-count recipes (L1…L50,
deep, seed 1904) by final score at a small size vs at 600M, 60 of 219 parent
benchmarks had ≥ 4 recipes in common.

- Best: `hellaswag_de` (0.94), `xwinograd_jp` (0.78), `arc_it` (0.78),
  `hellaswag_ru` (0.77), `global_mmlu_full_en` (0.73); the HellaSwag family is
  the most decision-reliable overall (mean 0.72; 0.91 deciding from 350M).
- At or below the coin flip: the Global-MMLU family (0.48), several
  INCLUDE / Belebele / XNLI variants, `multiblimp_spa` (0.22) — a benchmark
  at chance has nothing to rank (their across-recipe range at 600M is ~0.01).
- DA improves with the deciding size for the emerged families (HellaSwag
  0.52 → 0.91 from 90M → 350M; ARC 0.52 → 0.64) but not for chance-level ones.
- English benchmarks average 0.63 vs 0.55 for non-English, mostly the
  chance-level knowledge tasks dragging the non-English pool.

These numbers use the 4–6 language-count recipes as the model population;
the pipeline above uses every design variant at a size and will supersede
them.

## Files

- `pretraining/<pool>/da_per_task.csv` — the DA table (single source of truth
  for rq03).
- `…/da_per_benchmark.csv`, `da_per_benchmark_size.csv`,
  `da_per_benchmark_ckpt.csv` — long and wide per-(language, benchmark) views.

<!-- BEGIN auto:early-small (early_small.py --pool predictivity) -->
## Early and small, as a ranking

Numbers from the `predictivity` pool: every design variant at a proxy size, read at 1C–5C of training (C = the Chinchilla-optimal 20 tokens per parameter; every run trains 5C, so 1C is 20 % of it), ranked against the same variants at the 1.7B final checkpoint (the 5C column is DA-size, the 1.7B row is that size's DA-ckpt). A benchmark task counts only where it clears chance at the proxy size and at 1.7B. Regenerate with `python analysis/rq02_decision_accuracy/early_small.py --pool predictivity`.

- **bpb** — smallest proxy whose mean agreement with the 1.7B final ranking reaches 0.75: **175M at 1C** (0.89).
- **all benchmarks** — no (proxy, checkpoint) reaches a mean agreement of 0.75.
- **Smallest safe size per (benchmark, language)** — never: 85, 1B: 18, 350M: 7, 175M: 4, 600M: 3 of 117 cells.

![rq02 in one figure](pretraining/predictivity/highlights.png)

**bpb** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 30 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.66 | 0.89 | 0.54 | 0.70 | 0.94 | 0.72 | 0.63 | 0.72 | 0.72 | 0.72 |
| 350M | 0.93 | 0.96 | 0.90 | 0.79 | 0.93 | 0.94 | 0.87 | 0.86 | 0.93 | 0.96 |
| 600M | 0.53 | 0.51 | 0.51 | 0.51 | 0.51 | 0.44 | 0.51 | 0.52 | 0.53 | 0.51 |
| 1B | 0.65 | 0.85 | 0.91 | 0.88 | 0.87 | 0.87 | 0.86 | 0.82 | 0.85 | 0.84 |
| 1.7B | 0.88 | 0.92 | 0.92 | 0.98 | 1.00 | 0.99 | 1.00 | 0.99 | 1.00 |  |

**all benchmarks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 107 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.54 | 0.50 | 0.55 | 0.56 | 0.57 | 0.57 | 0.56 | 0.54 | 0.55 | 0.56 |
| 350M | 0.46 | 0.49 | 0.54 | 0.55 | 0.56 | 0.58 | 0.56 | 0.56 | 0.56 | 0.56 |
| 600M | 0.51 | 0.50 | 0.53 | 0.53 | 0.55 | 0.56 | 0.55 | 0.57 | 0.56 | 0.58 |
| 1B | 0.50 | 0.55 | 0.56 | 0.55 | 0.55 | 0.58 | 0.58 | 0.58 | 0.58 | 0.58 |
| 1.7B | 0.56 | 0.60 | 0.60 | 0.64 | 0.62 | 0.63 | 0.63 | 0.68 | 0.74 |  |

![Early and small](pretraining/predictivity/early_small.png)

![Early and small per benchmark](pretraining/predictivity/early_small_by_benchmark.png)

![Early and small per language](pretraining/predictivity/early_small_by_language.png)

**Smallest safe level per language and benchmark** (DA ≥ 0.75 over ≥ 3 pairs, held at every larger level — for FLOPs, at every costlier (size, checkpoint) cell; red = never, grey = filtered out by the above-random gate, white = no value). Each figure's table sits next to it under the same name:

![Smallest safe size](pretraining/predictivity/safe_size.png)

![Smallest safe checkpoint](pretraining/predictivity/safe_checkpoint.png)

![Smallest safe FLOPs](pretraining/predictivity/safe_flops.png)

![DA-size per benchmark](pretraining/predictivity/da_size_by_benchmark.png)

![DA-size per language](pretraining/predictivity/da_size_by_language.png)
<!-- END auto:early-small -->

## Why the pooled panel keeps every scheme

The `all pairs` panel of `by_L.py` and the black `all pairs` line of
`scale_convergence.py` are one population: **every pair of design variants at the
grid seed (1904), every data scheme** — A, B, AT3, ZH and ES — cross-L and
cross-architecture included. Until 2026-09-22 both held the scheme to A and B, the
`predictivity` headline pool's filter, and that was wrong here for two reasons.

**It made the pooled panel a different population from the panels beside it.** The
per-L panels have always pooled the schemes (`L_POOL = predictivity_all` at the grid
seed), so L50's cell was comparing A against AT3 while the first panel, drawn on the
same axes and read as its pooled version, excluded exactly that comparison. The same
held between `scale_convergence`'s black line and its coloured ones.

**The reason for the A/B filter does not transfer.** The headline pool excludes AT3,
ZH and ES because they widen the *dispersion* an SNR signal is measured over without
widening the decision the pool exists to measure. Decision accuracy is a rank
agreement over a pair set: a temperature change or a swapped second language adds
decisions to that set, it does not inflate a spread. A practitioner choosing T = 1
over T = 3 is making a design decision exactly as much as one choosing deep over
shallow.

What it changes today: **37 pairs at the reference**, all of them the two L50 AT3
families against the A/B families and each other. ZH and ES contribute nothing yet —
their 1.7B cells are not in the report — and will contribute without a code change
once they land (rule 9). Replicate seeds are unaffected either way: they exist only
below the reference, so no pair containing one is ever comparable.

The consequence is not cosmetic. Pooled over the gated benchmark tasks, the plain
scale-convergence line across 175M → 1B goes from **0.590, 0.599, 0.600, 0.598**
under the A/B filter to **0.586, 0.613, 0.623, 0.629** with every scheme — a slope
of **+0.010** against **+0.057** per decade of non-embedding parameters. On the
`above_66_both` population it is +0.100 against +0.180. The A/B filter was
suppressing measured convergence.

The second number of each pair is `scale_convergence[_above_66_both].csv` as it
stands; the first is a counterfactual, since no A/B-only table is kept. Reproduce it
by restricting `pairs_by_group`'s OVERALL branch to
`{ra["scheme"], rb["scheme"]} <= {"A", "B"}` and rerunning — do not hand-carry these
numbers, which is how three of them were wrong in the first draft of this section.

## What counts as one design axis

`scale_convergence.py --by transformation` draws one line per axis a pair differs
on, and drops any pair that moves two at once — such a pair is a decision about
neither. That only works if the axes are actually independent, and the `scheme`
token is not: it encodes **three** design choices at once. It is unpacked through
`DATA_SCHEMES` (`sets`, `temp`), the registry that defines the grid, so the axes
are `L`, `arch`, `list`, `T`, `lang2`, `seed`:

| scheme | `list` | `T` | `lang2` | reading |
|---|---|---|---|---|
| A | A | 1 | ru | resource-ranked list, English + Russian at L = 2 |
| AT3 | A | 3 | ru | A's list sampled at T = 3 |
| B | B | 1 | ru | diversity-first list |
| BT3 | B | 3 | ru | B's list at T = 3 (built 2026-09-21, not yet launched) |
| ZH | A | 1 | zh | A's list with Chinese in the second slot |
| ES | A | 1 | es | A's list with Spanish in the second slot |

Two consequences, both of which were wrong before 2026-09-22:

**ZH and ES are not list designs.** At L = 2 every setting is "English + one other
language"; scheme A's second language is simply Russian. So A, ZH and ES share a
list and differ on `lang2` alone, and A's L2 cell is the `ru` level of the
second-language axis rather than a scheme of its own. They used to be three
separate one-pair groups; they are now one 3-pair axis. None of them reaches the
reference yet — ZH's 1.7B eval is not in the report and ES is capped at 1B
(rule 9) — so the axis has no line today.

**B vs AT3 moves two axes, not one.** Under the old `scheme` key it differed on a
single key and was dropped only because no label existed for the level pair — the
right outcome for the wrong reason, and the same accident hid ZH vs ES. Every axis
now has a label and nothing is dropped for want of one; a pair is dropped when, and
only when, it moves more than one axis.

The split costs nothing today: `language count` (36 pairs at the reference),
`depth` (10) and `language list` (6) are the same lines as before. It pays off
when **BT3** lands, where `B vs BT3` becomes a pure temperature pair and takes that
axis from 2 pairs to the 3 that `MIN_PAIRS` needs — the temperature line becomes
drawable with no code change — and `AT3 vs BT3` joins the language-list axis.

<!-- BEGIN auto:by-L (by_L.py --pool predictivity) -->
## Per language count

**Pairs per L** — the design variants the grid plans at seed 1904, and per proxy size the pairs usable against 1.7B (both members planned at that size and at 1.7B): planned / with data today on BPB / on the benchmarks / on the training loss. ES stops at 1B, so it never pairs against the reference; ZH runs to 1.7B (2026-09-20) and is the third L2 family. A cell below MIN_PAIRS (3) families is left empty (rule 5), so a thin L shows blanks rather than a 0/1 reading.

| L | variants | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| 1 | L1-deep, L1-shallow | 1 / 0/0/0 | 1 / 0/0/0 | 1 / 0/0/0 | 1 / 0/0/0 |
| 2 | L2-ES-deep, L2-ZH-deep, L2-deep, L2-shallow | 3 / 0/0/0 | 3 / 0/0/0 | 3 / 0/0/0 | 3 / 0/0/0 |
| 8 | L8-deep, L8-schemeB-deep, L8-schemeB-shallow, L8-shallow | 6 / 3/3/3 | 6 / 3/3/3 | 6 / 3/3/3 | 6 / 0/6/6 |
| 15 | L15-AT3-deep, L15-deep, L15-schemeB-deep, L15-schemeB-shallow, L15-shallow | 10 / 0/3/3 | 10 / 0/3/3 | 10 / 0/3/3 | 10 / 0/6/6 |
| 30 | L30-AT3-deep, L30-BT3-deep, L30-deep, L30-schemeB-deep, L30-schemeB-shallow, L30-shallow | 15 / 3/3/3 | 15 / 3/3/3 | 15 / 3/3/3 | 15 / 3/6/6 |
| 50 | L50-AT3-deep, L50-AT3-shallow, L50-deep, L50-shallow | 6 / 0/3/3 | 6 / 0/3/3 | 6 / 0/3/3 | 6 / 0/6/6 |

The early-and-small reading one L at a time: pairs of design variants that share the L (seed 1904 of every scheme, `predictivity_all`), against the 1.7B final ranking, on the ten evaluated checkpoints of every run; a cell needs ≥ 3 pairs (rq02's rule), which today leaves out every L with one pair (the table above); the first panel pools every pair at that seed, every scheme included (`da_pooled_per_task.csv`). `da_by_L_per_task.csv` also carries each size's DA-ckpt within the L (`da_own`); rq04 reads both tables. Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool predictivity`.

Two of rq02's three decision accuracies have a checkpoint axis and so a figure here. **DA-goal** ranks the proxy at any checkpoint against the 1.7B final checkpoint; **DA-ckpt** ranks it against its own size's final checkpoint, so the 175M line asks what 175M would have decided early and what it misses is the checkpoint alone. The distance between the two is what the proxy *size* costs, and the 1.7B line is the same curve in both — at the reference the definitions coincide. DA-ckpt has no 5C column: a run's final checkpoint is its own reference. The third, **DA-size**, is DA-goal read at 5C alone and lives in `da_per_task.csv`. Each figure comes in a benchmarks-only version and a `_with_bpb` one that adds the solid per-size BPB lines; all four share the y axis, so any two overlay.

![DA-goal per L](pretraining/predictivity/early_small_by_L_goal.png)

![DA-goal per L, with BPB](pretraining/predictivity/early_small_by_L_goal_with_bpb.png)

![DA-ckpt per L](pretraining/predictivity/early_small_by_L_ckpt.png)

![DA-ckpt per L, with BPB](pretraining/predictivity/early_small_by_L_ckpt_with_bpb.png)

A third variant of each restricts the mean to the (benchmark, language) cells that rank reliably on BOTH axes (DA-size and DA-ckpt each ≥ 0.8, `reliable_tasks.py`): the plain panels average over every gated benchmark, these average over the benchmarks that work.

![DA-goal per L, reliable cells only](pretraining/predictivity/early_small_by_L_goal_above_80.png)

![DA-ckpt per L, reliable cells only](pretraining/predictivity/early_small_by_L_ckpt_above_80.png)
<!-- END auto:by-L -->

<!-- BEGIN auto:cross-task (cross_task.py --pool predictivity) -->
## Cross-task predictability

Every parent task as the proxy for every other one (381 x 381): the cell is the smallest proxy size (DA-size, 18 variants at 1.7B, 153 pairs) or the earliest checkpoint (DA-ckpt, the within-size pairs of every size pooled, 604 pairs, the nine checkpoints before the final) at which the ranking on task x (columns) safely predicts the final ranking on task y (rows): DA >= 0.75 over >= 3 pairs there and at every larger level with a value. The diagonal is rq02's own-task DA; the gate empties a benchmark's pairs at every size where it is at chance. The `_by_family` maps take the median level over the task pairs of two benchmarks, the `_by_language` maps over the same-benchmark task pairs of two languages (resource order of the scheme-A lists). Regenerate with `python analysis/rq02_decision_accuracy/cross_task.py --pool predictivity`.

The same two maps over the benchmark tasks that are above chance at some size — BPB, the loss and the benchmarks the gate finds at chance everywhere are dropped, so what is left is the sub-map where a transfer result is possible at all: [`cross_task_size_benchmarks.png`](pretraining/predictivity/cross_task_size_benchmarks.png), [`cross_task_ckpt_benchmarks.png`](pretraining/predictivity/cross_task_ckpt_benchmarks.png).

![Cross-task DA-size by benchmark](pretraining/predictivity/cross_task_size_by_family.png)

![Cross-task DA-ckpt by benchmark](pretraining/predictivity/cross_task_ckpt_by_family.png)

![Cross-task DA-size by language](pretraining/predictivity/cross_task_size_by_language.png)

![Cross-task DA-ckpt by language](pretraining/predictivity/cross_task_ckpt_by_language.png)

Full task-level maps: [`cross_task_size.png`](pretraining/predictivity/cross_task_size.png), [`cross_task_ckpt.png`](pretraining/predictivity/cross_task_ckpt.png).
<!-- END auto:cross-task -->

<!-- BEGIN auto:scale-convergence (scale_convergence.py) -->
## Scale convergence — the minimum useful scale

How small a **fully trained** model may be and still decide the way the 1.7B final checkpoint does: R_size(N) = matching decisions / comparable decisions over the gated tasks, and N_min(τ) = the smallest size with R ≥ τ (τ = 0.9). Same kernel, gate and pair minimum as the rest of rq02 — this is DA-size pooled over decisions rather than averaged over tasks, so the counts behind a point are in the CSV. The 1.7B point is 1.0 by construction. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py` (all three groupings).

**Pooled over every pair** (benchmarks; `scale_convergence.csv` carries BPB and the decision counts):

| group | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| all pairs | 0.55 | 0.57 | 0.59 | 0.6 | 1.0 | — |

![Scale convergence, overall](pretraining/predictivity/scale_convergence.png)

**By language count** (benchmarks; `scale_convergence_L.csv` carries BPB and the decision counts):

| group | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| L15 | 0.62 | 0.52 | 0.42 | 0.54 | 1.0 | — |
| L30 | 0.58 | 0.55 | 0.54 | 0.49 | 1.0 | — |
| L50 | 0.5 | 0.55 | 0.58 | 0.63 | 1.0 | — |
| L8 | 0.41 | 0.51 | 0.41 | 0.59 | 1.0 | — |
| all pairs | 0.55 | 0.57 | 0.59 | 0.6 | 1.0 | — |

![Scale convergence, L](pretraining/predictivity/scale_convergence_L.png)

**By design axis** (benchmarks; `scale_convergence_transformation.csv` carries BPB and the decision counts):

| group | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| all pairs | 0.55 | 0.57 | 0.59 | 0.6 | 1.0 | — |
| depth (deep vs shallow) | 0.61 | 0.57 | 0.45 | 0.54 | 1.0 | — |
| language count | 0.54 | 0.58 | 0.6 | 0.57 | 1.0 | — |
| language list (A vs B) | 0.52 | 0.52 | 0.44 | 0.53 | 1.0 | — |

![Scale convergence, transformation](pretraining/predictivity/scale_convergence_transformation.png)
<!-- END auto:scale-convergence -->

<!-- BEGIN auto:reliable-tasks (reliable_tasks.py --pool predictivity) -->
## Which benchmark-language cells rank reliably

Per language, how many benchmarks clear DA ≥ 0.8 on DA-size (a proxy size's final ranking vs the reference's) and on DA-ckpt (an earlier checkpoint vs the same size's final), reducing each task's cells with `late` (one fixed cell per axis, so no cell is chosen by its value). Cells need ≥ 3 pairs (rule 5) and must survive the above-random gate (rule 1). `da_reliable_tasks.csv` holds the per-task values for every reduction and is threshold-free — each figure is one view of it. Regenerate with `python analysis/rq02_decision_accuracy/reliable_tasks.py --pool predictivity`.

| language | benchmarks evaluated | DA-size | DA-ckpt | either | both |
|---|---|---|---|---|---|
| en | 10 | 2 | 5 | 5 | 2 |
| ru | 6 | 1 | 2 | 2 | 1 |
| zh | 5 | 0 | 1 | 1 | 0 |
| de | 7 | 1 | 5 | 5 | 1 |
| ja | 1 | 0 | 1 | 1 | 0 |
| es | 7 | 1 | 5 | 5 | 1 |
| fr | 7 | 1 | 2 | 2 | 1 |
| it | 5 | 1 | 3 | 3 | 1 |
| pt | 4 | 1 | 3 | 3 | 1 |
| pl | 1 | 0 | 1 | 1 | 0 |
| nl | 3 | 1 | 1 | 1 | 1 |
| id | 4 | 1 | 0 | 1 | 0 |
| vi | 4 | 1 | 1 | 1 | 1 |
| fa | 2 | 0 | 1 | 1 | 0 |
| tr | 3 | 0 | 1 | 1 | 0 |
| th | 2 | 0 | 1 | 1 | 0 |
| uk | 3 | 1 | 2 | 2 | 1 |
| el | 3 | 0 | 0 | 0 | 0 |
| cs | 1 | 0 | 1 | 1 | 0 |
| sv | 3 | 0 | 1 | 1 | 0 |
| hu | 2 | 0 | 1 | 1 | 0 |
| ro | 3 | 1 | 2 | 2 | 1 |
| no | 1 | 0 | 0 | 0 | 0 |
| da | 4 | 1 | 1 | 1 | 1 |
| bg | 2 | 0 | 1 | 1 | 0 |
| fi | 1 | 0 | 1 | 1 | 0 |
| hi | 4 | 0 | 1 | 1 | 0 |
| bn | 4 | 0 | 0 | 0 | 0 |
| he | 2 | 0 | 1 | 1 | 0 |
| ta | 4 | 0 | 0 | 0 | 0 |
| ka | 2 | 0 | 0 | 0 | 0 |
| ar | 3 | 0 | 2 | 2 | 0 |

**How many cells pass, by cut and reduction** — the cut is a choice, and this is its whole sensitivity:

| threshold | reduction | tasks passing both | languages | benchmarks |
|---|---|---|---|---|
| 0.8 | late | 13 | 12 | hellaswag, lambada_openai_mt |
| 0.8 | mean | 2 | 2 | hellaswag |
| 0.8 | median | 4 | 4 | hellaswag |
| 0.8 | max | 23 | 17 | hellaswag, lambada_openai_mt, multiblimp, xnli, xstorycloze |
| 0.75 | late | 15 | 13 | hellaswag, lambada_openai_mt |
| 0.75 | mean | 6 | 6 | hellaswag |
| 0.75 | median | 11 | 10 | hellaswag, xstorycloze |
| 0.75 | max | 29 | 19 | hellaswag, lambada_openai_mt, multiblimp, xnli, xstorycloze |
| 0.66 | late | 32 | 19 | arc, hellaswag, lambada_openai_mt, multiblimp, paws, xcopa, xnli, xstorycloze, xwinograd |
| 0.66 | mean | 22 | 14 | arc, hellaswag, lambada_openai_mt, multiblimp, xstorycloze |
| 0.66 | median | 23 | 14 | arc, hellaswag, lambada_openai_mt, multiblimp, xstorycloze |
| 0.66 | max | 57 | 26 | arc, hellaswag, lambada_openai_mt, multiblimp, paws, xcopa, xnli, xstorycloze, xwinograd |

![Reliable benchmark-language cells](pretraining/predictivity/da_reliable_tasks_80_late.png)
<!-- END auto:reliable-tasks -->

## The RQ2 figure

`rq2_above_66_both_transformation.png` is the figure RQ2 reports. It reads the three
decision accuracies over a single population: the 23 (benchmark, language) cells whose
median decision accuracy clears 0.66 on both DA-size and DA-ckpt, covering 14 languages
and 5 benchmark families (`arc`, `hellaswag`, `lambada_openai_mt`, `multiblimp`,
`xstorycloze`). Restricting to cells that rank reliably is what makes the three panels
comparable, because the unfiltered versions average over benchmarks that do not rank at
all. (Before 2026-09-22 this read 26 cells over 17 languages: `reliable_tasks.py` gated
DA-size without the reference, so three cells at chance at 1.7B — `belebele_ben_Beng`,
`include_base_44_georgian`, `include_base_44_tamil` — counted as reliable. Rule 1.)

![RQ2](pretraining/predictivity/rq2_above_66_both_transformation.png)

**Figure.** Decision accuracy of a cheap proxy against the 1.7B reference, under the
three definitions, on the cells reliable on both axes. **Left (DA-size):** a fully
trained proxy of each size against the reference's final ranking, one line per design
axis the pair differs on and one pooled black line over every pair; x is non-embedding
parameters on a log scale, the hollow 1.7B point is 1.0 by construction, and the dotted
line is the threshold, tau = 0.90. **Middle (DA-ckpt):** each proxy size against its own
final ranking, read at the ten evaluated checkpoints; x is training tokens in Chinchilla
multiples and the dotted line is 0.75. **Right (DA-goal):** the same checkpoints against
the reference's final ranking. The middle and right panels share the y axis with the
left, so the vertical distance between them is what the proxy's SIZE costs on top of
reading it early.

**Reading a proxy early is cheap.** Under DA-ckpt every proxy size reaches 0.85 to 0.90
by 4.5C (0.850, 0.890, 0.895, 0.888, 0.871 from 175M to 1.7B), and the gap between the
smallest proxy and the reference closes from 0.151 at 0.5C to 0.021 at 4.5C. The 600M
and 1B proxies clear 0.75 by 2C and the 350M by 2.5C; only the 175M lags, reaching it
at 4C. This indicates that for every proxy at 350M and above, stopping a run around two
Chinchilla multiples costs little, and that by the end of training the proxy's size
barely affects how well it reproduces its own final ranking.

**Reading a smaller model is not.** Under DA-goal the sizes separate and stay separated:
at 4.5C the 175M proxy reaches 0.619 against 0.795 for the 1B proxy and 0.871 for the
reference's own trajectory. Training the small proxy for longer does not close the gap:
from 2C to 5C the 175M line reads 0.620, 0.625, 0.594, 0.583, 0.588, 0.619, 0.650, and
never clears 0.75 at any checkpoint, while the 1B proxy clears it by 1.5C. We read this
as evidence that the binding constraint is the proxy's scale rather than its training
length, and that the two readings should not be traded against each other.

**The pooled DA-size line hides which decisions are recoverable.** Over every pair,
reliability rises from 0.647 at 175M to 0.786 at 1B and reaches tau = 0.90 at no size
below the reference. The decomposition shows that this aggregate averages two opposite
behaviours. The language-count decision improves with scale over most of the ladder,
from 0.580 at 175M to 0.837 at 600M, though it falls back to 0.774 at 1B. By contrast,
the language-list decision (scheme A against scheme B) degrades to 0.417 at 600M and
0.479 at 1B, below the 0.5 a coin flip would give, and depth is non-monotonic (0.793,
0.647, 0.520, 0.785). Consequently a fully trained proxy smaller than the reference is
informative about how many languages to train on and uninformative about which list to
train on, and reporting only the pooled line would have concealed that.

**Caveat.** The language-list axis rests on 18 comparable decisions at 175M and 48 at 1B,
against 219 and 310 for the language count, so its trajectory is the least stable line in
the panel. A 600M dip appears on two of the four lines (depth and the language list) and
is not yet explained; the language count instead peaks at 600M and falls at 1B. We leave
both to future work.
