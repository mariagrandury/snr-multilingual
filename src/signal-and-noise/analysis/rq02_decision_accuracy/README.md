# RQ2 — Does a benchmark rank models at a small size the way the reference size does?

## Research question

> Decision accuracy (DA) is the ground truth of the whole framework: the
> probability that a benchmark orders a pair of models the way an evaluation
> of larger models would (Heineman et al., 2025). Which benchmarks, in which
> languages, keep the ranking of the ladder's design variants across sizes
> (**DA-size**) and across training (**DA-ckpt**)?

<!-- BEGIN auto:highlight (da_per_benchmark.py --pool predictivity) -->
## Highlighted result

- **DA-size, proxy → 1.7B** (mean over the above-random benchmark tasks / over the per-language BPB tasks): 175M → 1.7B 0.54 / 0.78; 350M → 1.7B 0.52 / 0.82; 600M → 1.7B 0.53 / 0.58; 1B → 1.7B 0.54 / 0.82.
- **DA-size of `bpb_macro`** (one task, kept out of the means above): 175M 0.91; 350M 0.97; 600M 0.92; 1B 0.94.
- **DA-size of `train_loss`** (one task, kept out of the means above): 175M 0.81; 350M 0.86; 600M 0.81; 1B 0.83.
- **DA-ckpt** (early checkpoint vs final, above-random benchmark tasks): highest at 175M 90 % (0.84).
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
| 175M → 1.7B | 0.54 | 113 | 28 | 0.78 | 34 |
| 350M → 1.7B | 0.52 | 157 | 28 | 0.82 | 34 |
| 600M → 1.7B | 0.53 | 173 | 28 | 0.58 | 34 |
| 1B → 1.7B | 0.54 | 188 | 28 | 0.82 | 34 |

![DA-size by family](pretraining/predictivity/da_size_by_family.png)

**DA-ckpt by bucket and fraction of the run** (mean over the above-random benchmark tasks):

| bucket | 10 % | 20 % | 30 % | 40 % | 50 % | 60 % | 70 % | 80 % | 90 % |
|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.54 | 0.56 | 0.58 | 0.68 | 0.70 | 0.73 | 0.74 | 0.79 | 0.84 |
| 350M | 0.50 | 0.53 | 0.56 | 0.58 | 0.61 | 0.63 | 0.69 | 0.73 | 0.80 |
| 600M | 0.50 | 0.54 | 0.55 | 0.58 | 0.57 | 0.61 | 0.61 | 0.66 | 0.75 |
| 1B | 0.49 | 0.53 | 0.57 | 0.57 | 0.59 | 0.61 | 0.64 | 0.66 | 0.75 |
| 1.7B | 0.52 | 0.56 | 0.57 | 0.59 | 0.59 | 0.60 | 0.62 | 0.65 | 0.72 |
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

- **bpb** — smallest proxy whose mean agreement with the 1.7B final ranking reaches 0.75: **175M at 2C** (0.77).
- **all benchmarks** — no (proxy, checkpoint) reaches a mean agreement of 0.75.
- **Smallest safe size per (benchmark, language)** — never: 166, 1B: 34, 350M: 7, 175M: 5, 600M: 3 of 215 cells.

![rq02 in one figure](pretraining/predictivity/highlights.png)

**bpb** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 34 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.67 | 0.74 | 0.66 | 0.77 | 0.83 | 0.66 | 0.75 | 0.79 | 0.78 | 0.78 |
| 350M | 0.71 | 0.73 | 0.72 | 0.70 | 0.77 | 0.79 | 0.75 | 0.76 | 0.80 | 0.82 |
| 600M | 0.52 | 0.53 | 0.54 | 0.57 | 0.57 | 0.59 | 0.56 | 0.57 | 0.58 | 0.58 |
| 1B | 0.65 | 0.66 | 0.70 | 0.75 | 0.74 | 0.80 | 0.77 | 0.80 | 0.81 | 0.82 |
| 1.7B | 0.81 | 0.90 | 0.93 | 0.95 | 0.96 | 0.95 | 0.97 | 0.96 | 0.99 |  |

**all benchmarks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 215 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.52 | 0.49 | 0.52 | 0.52 | 0.53 | 0.53 | 0.51 | 0.51 | 0.52 | 0.54 |
| 350M | 0.47 | 0.48 | 0.50 | 0.49 | 0.49 | 0.52 | 0.51 | 0.52 | 0.51 | 0.52 |
| 600M | 0.48 | 0.47 | 0.49 | 0.50 | 0.52 | 0.50 | 0.52 | 0.52 | 0.53 | 0.53 |
| 1B | 0.50 | 0.52 | 0.52 | 0.52 | 0.54 | 0.55 | 0.53 | 0.55 | 0.55 | 0.54 |
| 1.7B | 0.52 | 0.56 | 0.57 | 0.59 | 0.59 | 0.60 | 0.62 | 0.65 | 0.72 |  |

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
| 1 | L1-dclmP-deep, L1-deep, L1-fweb-deep, L1-shallow | 6 / 0/0/0 | 6 / 0/0/0 | 6 / 0/0/0 | 6 / 0/0/0 |
| 2 | L2-ES-deep, L2-ZH-deep, L2-deep, L2-shallow | 6 / 0/0/0 | 6 / 0/0/0 | 6 / 0/0/0 | 6 / 0/0/0 |
| 8 | L8-deep, L8-schemeB-deep, L8-schemeB-shallow, L8-shallow | 6 / 4/4/4 | 6 / 4/4/4 | 6 / 4/4/4 | 6 / 4/4/4 |
| 15 | L15-AT3-deep, L15-deep, L15-schemeB-deep, L15-schemeB-shallow, L15-shallow | 10 / 4/5/5 | 10 / 4/5/5 | 10 / 4/5/5 | 10 / 4/5/5 |
| 30 | L30-AT3-deep, L30-BT3-deep, L30-deep, L30-schemeB-deep, L30-schemeB-shallow, L30-shallow | 15 / 4/5/5 | 15 / 4/5/5 | 15 / 4/5/5 | 15 / 4/5/5 |
| 50 | L50-AT3-deep, L50-AT3-shallow, L50-deep, L50-shallow | 6 / 4/4/4 | 6 / 4/4/4 | 6 / 4/4/4 | 6 / 4/4/4 |

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

Every parent task as the proxy for every other one (562 x 562): the cell is the smallest proxy size (DA-size, 18 variants at 1.7B, 153 pairs) or the earliest checkpoint (DA-ckpt, the within-size pairs of every size pooled, 765 pairs, the nine checkpoints before the final) at which the ranking on task x (columns) safely predicts the final ranking on task y (rows): DA >= 0.75 over >= 3 pairs there and at every larger level with a value. The diagonal is rq02's own-task DA; the gate empties a benchmark's pairs at every size where it is at chance. The `_by_family` maps take the median level over the task pairs of two benchmarks, the `_by_language` maps over the same-benchmark task pairs of two languages (resource order of the scheme-A lists). Regenerate with `python analysis/rq02_decision_accuracy/cross_task.py --pool predictivity`.

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
| all pairs | 0.55 | 0.58 | 0.59 | 0.61 | 1.0 | — |

![Scale convergence, overall](pretraining/predictivity/scale_convergence.png)

**By language count** (benchmarks; `scale_convergence_L.csv` carries BPB and the decision counts):

| group | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| L15 | 0.61 | 0.57 | 0.54 | 0.58 | 1.0 | — |
| L30 | 0.55 | 0.54 | 0.54 | 0.52 | 1.0 | — |
| L50 | 0.52 | 0.54 | 0.56 | 0.61 | 1.0 | — |
| L8 | 0.46 | 0.52 | 0.49 | 0.59 | 1.0 | — |
| all pairs | 0.55 | 0.58 | 0.59 | 0.61 | 1.0 | — |

![Scale convergence, L](pretraining/predictivity/scale_convergence_L.png)

**By design axis** (benchmarks; `scale_convergence_transformation.csv` carries BPB and the decision counts):

| group | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| all pairs | 0.55 | 0.58 | 0.59 | 0.61 | 1.0 | — |
| depth (deep vs shallow) | 0.57 | 0.51 | 0.5 | 0.54 | 1.0 | — |
| language count | 0.53 | 0.57 | 0.59 | 0.58 | 1.0 | — |
| language list (A vs B) | 0.5 | 0.51 | 0.48 | 0.52 | 1.0 | — |
| temperature (T=1 vs T=3) | 0.5 | 0.61 | 0.62 | 0.62 | 1.0 | — |

![Scale convergence, transformation](pretraining/predictivity/scale_convergence_transformation.png)
<!-- END auto:scale-convergence -->

<!-- BEGIN auto:reliable-tasks (reliable_tasks.py --pool predictivity) -->
## Which benchmark-language cells rank reliably

Per language, how many benchmarks clear DA ≥ 0.8 on DA-size (a proxy size's final ranking vs the reference's) and on DA-ckpt (an earlier checkpoint vs the same size's final), reducing each task's cells with `late` (one fixed cell per axis, so no cell is chosen by its value). Cells need ≥ 3 pairs (rule 5) and must survive the above-random gate (rule 1). The decisions come from the `predictivity_schemes` pool — every data scheme at the grid seed, which is the population `by_L` and `scale_convergence` pair over — while the gate and this folder stay with `predictivity`; the table carries one row per pair set (rule 15) and the figures show `multi-axis`. `da_reliable_tasks.csv` holds the per-task values for every reduction and is threshold-free — each figure is one view of it. Regenerate with `python analysis/rq02_decision_accuracy/reliable_tasks.py --pool predictivity`.

| language | benchmarks evaluated | DA-size | DA-ckpt | either | both |
|---|---|---|---|---|---|
| en | 12 | 2 | 9 | 9 | 2 |
| ru | 10 | 2 | 4 | 4 | 2 |
| zh | 10 | 0 | 1 | 1 | 0 |
| de | 10 | 1 | 6 | 6 | 1 |
| ja | 5 | 0 | 2 | 2 | 0 |
| es | 11 | 1 | 6 | 6 | 1 |
| fr | 11 | 2 | 5 | 5 | 2 |
| it | 9 | 2 | 3 | 3 | 2 |
| pt | 8 | 1 | 4 | 4 | 1 |
| pl | 5 | 0 | 2 | 2 | 0 |
| nl | 5 | 1 | 3 | 3 | 1 |
| id | 8 | 2 | 3 | 3 | 2 |
| vi | 8 | 1 | 3 | 3 | 1 |
| fa | 6 | 0 | 2 | 2 | 0 |
| tr | 7 | 0 | 3 | 3 | 0 |
| th | 3 | 0 | 1 | 1 | 0 |
| uk | 7 | 1 | 2 | 3 | 0 |
| el | 6 | 0 | 0 | 0 | 0 |
| ko | 4 | 0 | 1 | 1 | 0 |
| cs | 3 | 0 | 2 | 2 | 0 |
| sv | 5 | 1 | 2 | 2 | 1 |
| hu | 5 | 1 | 1 | 1 | 1 |
| ro | 5 | 1 | 2 | 2 | 1 |
| no | 2 | 0 | 0 | 0 | 0 |
| da | 5 | 1 | 1 | 1 | 1 |
| bg | 5 | 0 | 2 | 2 | 0 |
| fi | 2 | 0 | 1 | 1 | 0 |
| hi | 9 | 0 | 4 | 4 | 0 |
| bn | 8 | 0 | 1 | 1 | 0 |
| sk | 3 | 3 | 3 | 3 | 3 |
| he | 4 | 2 | 1 | 2 | 1 |
| lt | 6 | 0 | 2 | 2 | 0 |
| bs | 1 | 0 | 0 | 0 | 0 |
| sl | 2 | 1 | 2 | 2 | 1 |
| et | 6 | 3 | 2 | 3 | 2 |
| ca | 8 | 3 | 4 | 5 | 2 |
| ta | 4 | 0 | 0 | 0 | 0 |
| hr | 5 | 2 | 3 | 3 | 2 |
| lv | 2 | 0 | 0 | 0 | 0 |
| ms | 5 | 2 | 3 | 3 | 2 |
| az | 4 | 0 | 1 | 1 | 0 |
| ka | 5 | 0 | 2 | 2 | 0 |
| ne | 8 | 2 | 2 | 2 | 2 |
| mr | 4 | 2 | 2 | 2 | 2 |
| ml | 1 | 0 | 1 | 1 | 0 |
| kk | 7 | 2 | 3 | 4 | 1 |
| ur | 6 | 1 | 4 | 4 | 1 |
| sq | 4 | 1 | 1 | 1 | 1 |
| ar | 11 | 0 | 5 | 5 | 0 |
| sr | 8 | 3 | 2 | 4 | 1 |

**How many cells pass, by cut and reduction** — the cut is a choice, and this is its whole sensitivity:

| threshold | reduction | tasks passing both | languages | benchmarks |
|---|---|---|---|---|
| 0.8 | late | 40 | 27 | hellaswag, lambada_openai_mt, multiblimp, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xstorycloze |
| 0.8 | mean | 7 | 7 | hellaswag, multiblimp, rfgm_include_base_44 |
| 0.8 | median | 19 | 16 | global_piqa_nonparallel_cloze, hellaswag, multiblimp, rf_include_base_44, rfgm_include_base_44, xstorycloze |
| 0.8 | max | 62 | 32 | global_piqa_nonparallel_cloze, hellaswag, include_base_44, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xnli, xstorycloze |
| 0.75 | late | 51 | 34 | hellaswag, lambada_openai_mt, multiblimp, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xcopa, xstorycloze |
| 0.75 | mean | 28 | 24 | global_piqa_nonparallel_cloze, hellaswag, multiblimp, paws, rf_global_mmlu_full, rfgm_include_base_44, xstorycloze |
| 0.75 | median | 30 | 24 | global_piqa_nonparallel_cloze, hellaswag, multiblimp, paws, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xstorycloze |
| 0.75 | max | 78 | 38 | global_piqa_nonparallel_cloze, hellaswag, include_base_44, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xcopa, xnli, xstorycloze |
| 0.66 | late | 91 | 41 | arc, belebele, global_piqa_nonparallel_cloze, hellaswag, lambada_openai_mt, multiblimp, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xcopa, xstorycloze, xwinograd |
| 0.66 | mean | 55 | 33 | arc, belebele, global_piqa_nonparallel_cloze, hellaswag, include_base_44, lambada_openai_mt, multiblimp, paws, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xstorycloze |
| 0.66 | median | 62 | 35 | arc, belebele, global_piqa_nonparallel_cloze, hellaswag, include_base_44, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xnli, xstorycloze, xwinograd |
| 0.66 | max | 146 | 46 | arc, belebele, global_piqa_nonparallel_cloze, hellaswag, include_base_44, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xcopa, xnli, xstorycloze, xwinograd |

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

`rq2_above_66_either_transformation.png` is the same axis breakdown over the per-panel
populations of `rq2_above_66_one` — each definition read on the cells reliable for it.
Its left panel is therefore not the same cells as its middle and right, so it is three
claims rather than one population seen three ways.

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

## Exploratory: multi-axis against mono-axis pairs

`pair_axes.py` computes the three decision accuracies on the `above_66_both` cells
twice — over every pair at the grid seed (multi-axis, rq02's convention) and over
the pairs that move exactly one of L, depth, list, temperature, second language
(mono-axis, which is what DataDecide's "all pairs" are by construction) — and
writes `rq2_above_66_both_axes.png/.csv`. DA-ckpt is indifferent to the pair set;
DA-size and DA-goal read 0.04–0.05 lower under mono-axis with the same trend, on a
third of the decisions. The proposal that follows from it (an `axes` column in
`da_per_task.csv`, mono-axis as the headline for decisions) is in
`plan/decision_accuracy.md` (§2, §6).

<!-- BEGIN auto:agreement-measures (agreement.py --pool predictivity) -->
## Decision accuracy is Kendall's τ under another tie convention

Over the 820 (benchmark task, proxy size) cells of DA-size's population (every pair of the grid-seed variants, ≥ 3 pairs, above chance at the proxy and at 1.7B): 2·DA − 1 = τ_a + (T_both − T_one)/n exactly, where T_both / T_one are the pairs tied on both / one side. Ties are 2,781 of 55,381 pairs (5.0%), 95% of them one-sided, and touch 70% of the cells — so the two statistics correlate at r = 0.976 by construction, and the number with content is how often the tie convention changes a reliability verdict (DA ≥ 0.66, i.e. τ ≥ 0.32), in % of cells per proxy size:

| statistic | 175M | 350M | 600M | 1B |
|---|---|---|---|---|
| da | 0.0 | 0.0 | 0.0 | 0.0 |
| da_drop_ref_ties | 4.0 | 2.4 | 3.2 | 2.9 |
| gamma | 6.0 | 5.9 | 7.2 | 6.5 |
| rho | 10.1 | 10.7 | 14.0 | 13.9 |
| tau_a | 4.7 | 2.9 | 5.4 | 3.3 |
| tau_b | 5.4 | 5.4 | 6.8 | 4.5 |

Spearman ρ and Pearson r on the raw scores are the two statistics that are NOT a rescaling — they weight a pair by its displacement — and sit at r = 0.965 and 0.909 against DA. Median 11 models per cell. Values in `agreement_per_cell.csv`; regenerate with `python analysis/rq02_decision_accuracy/agreement.py --pool predictivity`.

![DA against Kendall's tau](pretraining/predictivity/agreement_identity.png)

![Cut sensitivity](pretraining/predictivity/agreement_cut_sensitivity.png)
<!-- END auto:agreement-measures -->

<!-- BEGIN auto:scale-convergence-by-language (by_language.py --pool predictivity) -->
## Scale convergence per language

For each language of the L8 setting, the `--by L` lines read on that language's benchmarks alone: R at the smallest → largest proxy [tasks], per regime that trains the language, on every gated task (no selection on DA — the inference version; the `above_66_size` twin is the conditional one). The last column is the collapse test: R² of one log-linear line through every regime's points with x = model size, then with x = tokens of the language (share × 0.50 × D(N)); a rise under tokens says exposure explains what language count does not. Regimes pool arch, list and temperature decisions at once. Regenerate with `python analysis/rq02_decision_accuracy/by_language.py --pool predictivity`; `scale_convergence_lang_all_coverage.csv` says why a cell is empty.

| language | L1 | L2 | L8 | L15 | L30 | L50 | R² size / tokens |
|---|---|---|---|---|---|---|---|
| en | — | — | 0.50→1.00 [11] | 0.57→1.00 [11] | 0.58→0.83 [11] | 0.43→1.00 [11] | 0.00 / 0.01 |
| ru | — | — | 0.47→1.00 [9] | 0.70→1.00 [9] | 0.62→0.83 [9] | 0.67→1.00 [9] | 0.00 / 0.01 |
| zh | — | — | 0.42→0.83 [8] | 0.55→1.00 [8] | 0.45→0.83 [8] | 0.42→1.00 [8] | 0.00 / 0.00 |
| de | — | — | 0.50→1.00 [7] | 0.62→1.00 [7] | 0.48→0.83 [7] | 0.67→1.00 [7] | 0.00 / 0.01 |
| ja | — | — | 0.00→1.00 [5] | 0.70→1.00 [5] | 0.40→0.83 [5] | 0.33→1.00 [5] | 0.03 / 0.02 |
| es | — | — | — | 0.60→0.64 [11] | 0.64→0.83 [11] | 0.63→1.00 [11] | 0.04 / 0.07 |
| fr | — | — | — | 0.67→0.47 [10] | 0.45→0.83 [10] | 0.56→1.00 [10] | 0.02 / 0.06 |
| it | — | — | — | 0.67→0.74 [9] | 0.65→0.83 [9] | 0.71→0.83 [9] | 0.09 / 0.09 |

![Scale convergence per language](pretraining/predictivity/scale_convergence_lang_all.png)

![Scale convergence per language, tokens axis](pretraining/predictivity/scale_convergence_lang_all_tokens.png)
<!-- END auto:scale-convergence-by-language -->

<!-- BEGIN auto:seed-uncertainty (seed_uncertainty.py --pool predictivity) -->
## Seed uncertainty

What the replicate seeds say about decision accuracy — English at three proxy sizes and Russian at 1B, the only (size, language) cells where three replicated designs give ≥ 3 cross-L pairs under rule 2. Per row: DA of those decisions against the 1.7B final with the proxy at each of its three seeds (proxy-side noise), and the DA between two seeds' rankings of the same designs at that size (the reference-side ceiling). Three pairs put a DA on {0, ⅓, ⅔, 1}: read the spread, not a mean. The seed null (DA-ckpt over pairs of two seeds of one design, mean over fractions and gated tasks) is 175M: 0.62 vs real 0.69; 600M: 0.58 vs real 0.60; 1B: 0.59 vs real 0.61. Regenerate with `python analysis/rq02_decision_accuracy/seed_uncertainty.py --pool predictivity`.

| size | language | designs | DA at the 3 proxy seeds | test-retest (3 seed pairs) | tasks |
|---|---|---|---|---|---|
| 175M | en | 3 | 0.74, 0.70, 0.83 | 0.83–0.89 | 9 |
| 175M | ru | 3 | nan, nan, nan | — | 0 |
| 600M | en | 3 | 0.85, 0.85, 0.85 | 0.82–0.85 | 11 |
| 600M | ru | 3 | nan, nan, nan | — | 0 |
| 1B | en | 4 | 0.80, 0.73, 0.68 | 0.77–0.80 | 11 |
| 1B | ru | 4 | 0.81, 0.85, 0.78 | 0.78–0.89 | 9 |

![Seed uncertainty](pretraining/predictivity/seed_uncertainty.png)
<!-- END auto:seed-uncertainty -->

<!-- BEGIN auto:scale-convergence-L8-common (scale_convergence.py --by L --langs L8 --common-tasks) -->
## Scale convergence by language count, on the L8 languages, common tasks

The `--by L` lines read over the tasks in the 8 languages of the L8 setting (de, en, es, fr, it, ja, ru, zh), which every regime from L8 up trains, and further over the tasks with ≥ 3 pairs in every regime at every proxy size — one task set for the whole figure, so a gap between lines is a gap on the same benchmarks. What this cannot fix: a regime pools arch, list and temperature decisions at once and the mix differs by regime (`share_*` in the CSV); rule 5 forbids holding it fixed. Under `--axes mono-axis` the regimes keep only their one-axis pairs and rule 5 empties every proxy size but 1B, so the `_one_axis` L figures draw the pooled line alone. Numbers below are the unfiltered population; the `above_66_size` twin (`scale_convergence_L8common_above_66_size.png`) is the conditional one. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by L --langs L8 --common-tasks`.

| group | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| L15 | 0.61 | 0.51 | 0.48 | 0.53 | 1.0 | — |
| L30 | 0.55 | 0.57 | 0.55 | 0.53 | 1.0 | — |
| L50 | 0.53 | 0.61 | 0.61 | 0.61 | 1.0 | — |
| L8 | 0.46 | 0.52 | 0.51 | 0.61 | 1.0 | — |
| all pairs | 0.56 | 0.62 | 0.64 | 0.65 | 1.0 | — |

![Scale convergence, L8 common tasks](pretraining/predictivity/scale_convergence_L8common.png)
<!-- END auto:scale-convergence-L8-common -->
