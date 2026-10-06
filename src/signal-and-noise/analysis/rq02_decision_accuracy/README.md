# RQ2 — Does a benchmark rank models at a small size, or early in a run, the way the reference does?

## Research question

> Decision accuracy (DA) is the ground truth of the whole framework: the
> probability that a benchmark orders a pair of models the way an evaluation
> of larger models would (Heineman et al., 2025). Which benchmarks, in which
> languages, keep the ranking of the ladder's design variants across sizes
> (**DA-size**), across training (**DA-ckpt**), and early *and* small at once
> (**DA-goal**)?

<!-- BEGIN auto:highlight (da_per_benchmark.py --pool predictivity) -->
## Highlighted result

- **DA-size, proxy → 1.7B** (mean over the above-random benchmark tasks / over the per-language BPB tasks): 90M → 1.7B 0.55 / 0.91; 175M → 1.7B 0.53 / 0.91; 350M → 1.7B 0.54 / 0.82; 600M → 1.7B 0.54 / 0.58; 1B → 1.7B 0.53 / 0.82.
- **DA-size of `bpb_macro`** (one task, kept out of the means above): 90M 0.94; 175M 0.97; 350M 0.97; 600M 0.92; 1B 0.94.
- **DA-size of `train_loss`** (one task, kept out of the means above): 90M 0.79; 175M 0.93; 350M 0.86; 600M 0.81; 1B 0.83.
- **DA-ckpt** (early checkpoint vs final, above-random benchmark tasks): highest at 350M 90 % (0.79).
<!-- END auto:highlight -->

**In one paragraph** (ladder-report snapshot **2026-09-30 23:54**, the one
the tables on disk were built from — every cell of the grid evaluated except
FWEB and the 3B L30/L50 rungs, the 90M and 175M rungs at their own batch;
every number below is read from the CSV beside the figure it describes).
Ranking design variants from a smaller fully trained model is close to a
coin flip on the full gated population (0.54 at 90M, 0.55 at 175M, 0.56 at
1B, jackknife ± 0.02, ± 0.03 at 1B); the 0.64 → 0.76 rise of the paper's filtered figure
(0.72 → 0.76 from 175M) is a conditional statement on the tasks whose
DA-size cleared 0.66. Ranking from an early checkpoint of the same run
reaches 0.74–0.82 at 90 % of training (0.75 at 90M, 0.82 at 350M), but two
seeds of one design reach up to 0.75 on the same axis (the seed null,
0.56–0.58 on average): most of DA-ckpt's rise is within-run persistence, and
on English at 175M–1B two seeds of the same designs disagree with each other
(test-retest 0.55–0.89, 0.63–0.73 on average) as much as the proxy disagrees
with the reference. Restricting to the eight high-resource languages, to a
common task set, or to one language or one language tier does not order the
per-L lines, and a language's token share does not predict how reliably its
benchmarks rank (Spearman ρ 0.02 to 0.27 over 45–48 languages). Decision
accuracy, Kendall's τ and Spearman's ρ are one statistic (r ≥ 0.96 over
1 883 cells), and the tie convention alone moves the reliable-task verdict
on 2–7 % of cells.

## Naming

Every artifact of this folder says in its name what it holds (rule 16), in one
order: `<subject>_da_<kind>[_<breakdown>][_<filter>]_<pair set>[_<view>]`.

- **DA kind**: `da_size`, `da_ckpt`, `da_goal`; `da_all` when one file holds
  more than one (the per-task tables, the three-panel `rq2_da_all_*` figures);
  `da_size_vs_da_ckpt` when it sets two against each other.
- **Breakdown**: `by_L`, `by_transformation`, `by_benchmark`, `by_language`,
  `L` / `transformation` / `lang_*` for the scale-convergence lines.
- **Filter**: `above_66_size`, `above_66_ckpt`, `above_66_either`,
  `above_66_both`, `above_80` — the reliability cut the population passed.
- **Pair set** (rule 15): `multi_axes` (every pair of design variants),
  `mono_axis` (pairs moving exactly one axis), `seed_null` (two seeds of one
  design); `both_axes` marks a table whose `axes` column carries every pair
  set, `mono_vs_multi_axes` a figure that compares the two.
- **View**: `_flops`, `_tokens`, `_coverage`.

A facet pair shares one table: `<name>_by_benchmark_<pair set>.png` and
`<name>_by_language_<pair set>.png` read `<name>_<pair set>.csv`.

## Setup

**Pools and sizes.** Models are the predictivity ladder's cells (pool
`predictivity`: 90M–1.7B, each rung at its own batch, rule 10;
L ∈ {1, 2, 8, 15, 30, 50} × every ladder × every data build, seed 1904 —
[RULES.md](../RULES.md), Definitions), the pair set every decision figure
from figure 1 on is computed over (`reliable_tasks.py`, `by_L.py`,
`scale_convergence.py` and the extensions; rule 15); `predictivity_seeds`
adds the replicate seeds as separate models (figure 7). The gate pool is
`predictivity` everywhere; the output folder is `pretraining/predictivity/`.
The reference is 1.7B (rule 9). Until 2026-10-05 the `predictivity` pool held
schemes A/B only and the decisions came from a separate all-builds pool whose
tables sat in another folder (plan/decision_accuracy.md §9); the figures
below were drawn on that all-builds population, which is today's `predictivity`.

**Three decision accuracies.** DA is the share of design-variant pairs a
proxy orders the same way as the reference; a pair tied on both sides counts
as agreement and a pair tied on one side as a miss (the kernel's one departure
from upstream, below). **DA-size** compares the proxy size's final checkpoint
with the 1.7B final; **DA-ckpt** an early checkpoint of a run with that run's
own final; **DA-goal** an early checkpoint of the proxy with the 1.7B final.
Two identities are verified on every run: DA-size equals DA-goal at 100 % of
the proxy's run, and DA-ckpt of the 1.7B run equals DA-goal of the 1.7B run.
Pooled figures (`scale_convergence_da_size*`, `rq2_da_all*`, `reliability_da_size*`) report
matching decisions over comparable decisions summed over tasks; the
`early_small_da_*` families (`early_small.py`, `by_L.py`) report the mean over tasks. The two agree
to two decimals on today's tables but are different estimands.

**Pair sets (rule 15).** `multi-axis`: every pair of design variants at the
grid seed (two thirds move more than one axis at once). `mono-axis`
(`_mono_axis` stems; the multi-axis files carry `_multi_axes`): the pairs moving exactly one of L, depth,
activation, optimizer, data scheme (A/B/C, the recipe at that L), T — the
decision a practitioner makes, and what upstream's "every pair" is by
construction. `seed`: two draws of one design (the null; DA-ckpt only,
figure 7). On the 2026-10-05 report the mono-axis pairs at the grid seed are
`language count` 39, `depth` 10, `data scheme` 12 (A vs B at L8–L30, A/ZH/ES
at L2, A/DCLMP/FWEB at L1 — until that date three axes, language list 6,
second language 3 and English corpus 3) and `temperature` 4; the other 260 of
the 325 pairs move two or more axes.

**Filters.** An `above_*` stem keeps the tasks whose DA cleared a cut on a
reduction of their cells (`reliable_tasks.py`): `above_66_size` /
`above_66_ckpt` / `above_66_either` (median DA over the proxy cells ≥ 0.66
on that axis, or on either), `above_66_both` (both axes; never quoted, RULES.md),
`above_80` (0.80 on the `late` reduction, both axes). Every filtered figure
plots the quantity it selected on and is read as conditional (bug #17,
`../CLAUDE.md`); the plain stem is the unconditional one.

## Experimental setup

Models are the predictivity ladder's cells (`configs/models.json` pool
`predictivity`: 90M–1.7B (each rung at its own batch, rule 10) ×
L ∈ {1, 2, 8, 15, 30, 50} × every ladder × every data build, seed 1904, the
pool `reliable_tasks.py`, `by_L.py` and `scale_convergence.py` pair over,
rule 15; `predictivity_seeds` adds the seed replicates as separate models). The cross-size identity is the cell's `family`
(`lm-L8-schemeB-deep-seed1904`: everything but the size), so a design variant
present at two sizes is one pair.

- **DA-size** — `decision_acc_size_<small>`: the families' ranking at
  `<small>`'s final checkpoint vs at the reference size's (1.7B) final checkpoint;
  `decision_acc_size_<a>_to_<b>` for every other size pair with ≥ 2 shared
  families (175M→350M … 600M→1B). Multilingual tasks are only evaluated on
  cells that train the language, so each task's pair set is the families
  that exist at both sizes *and* were evaluated on it.
- **DA-ckpt** — `decision_acc_ckpt_f<frac>_<size>`: within one size, the
  ranking at the checkpoint nearest 10, 20, … 90 % of each run
  (`da_early_fracs`; 0.5C … 4.5C on the WSD schedule) vs at the final
  checkpoint.

Per-language BPB (`bpb_<subset>`) and the training loss are tasks too, so DA
is computed for the plan's outcome metric alongside the benchmarks. DA is
never gated: it is the truth the SNR proxies of rq03 and rq04 are scored against.

## Methodology

[`compute_da.py`](compute_da.py) writes `pretraining/<pool>/da_all_per_task_both_axes.csv`
(one row per (parent task, pair set `axes`), one column per DA definition) from
`utils.pair_agreement`; [`da_per_benchmark.py`](da_per_benchmark.py)
melts it into a long (language, benchmark, comparison) table and the wide
`_size` / `_ckpt` pivots, and rewrites the deck's appendix slides for the
canonical pool. rq03 joins the SNR variants onto this table; rq05 asks the
complementary question — which proxy *size* ranks an intervention like the
reference, with languages as the population.

With few families at a size (26 at 1.7B on the 2026-10-05 report, every
build at the grid seed; a task in a language only the L50 mixture trains rests
on three or four),
DA is quantised to 1/#pairs: read the
family-level averages, and `n` alongside every value
(`da_all_n_pairs_per_task_both_axes.csv`).

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

## Why the pooled panel keeps every scheme

The `all pairs` panel of `by_L.py` and the black `all pairs` line of
`scale_convergence.py` are one population: **every pair of design variants at the
grid seed (1904), every data build** — A, AT3, B, ZH, ES, DCLMP and FWEB — cross-L and
cross-architecture included. Until 2026-09-22 both held the scheme to A and B, the
`predictivity` headline pool's filter at the time, and that was wrong here for two
reasons. (Since 2026-10-05 the `predictivity` pool itself carries every build, so
the section below is the history of why.)

**It made the pooled panel a different population from the panels beside it.** The
per-L panels have always pooled the schemes (`L_POOL = predictivity_seeds` at the grid
seed), so L50's cell was comparing A against AT3 while the first panel, drawn on the
same axes and read as its pooled version, excluded exactly that comparison. The same
held between `scale_convergence`'s black line and its coloured ones.

**The reason for the A/B filter does not transfer.** The headline pool excluded AT3,
ZH and ES because they widen the *dispersion* an SNR signal is measured over without
widening the decision the pool exists to measure. Decision accuracy is a rank
agreement over a pair set: a temperature change or a swapped second language adds
decisions to that set, it does not inflate a spread. A practitioner choosing T = 1
over T = 3 is making a design decision exactly as much as one choosing deep over
shallow.

What it changes today: **123 of the 276 pairs at the reference** (24 families
against the 18 of schemes A and B): the four AT3 families (L15, L30 and L50 deep,
L50 shallow) and the two L2 second-language families (ZH, ES), all at 1.7B in the
2026-09-23 report, against the A/B families and each other. Replicate seeds are
unaffected either way: they exist only
below the reference, so no pair containing one is ever comparable.

The consequence is not cosmetic. On the 2026-09-22 tables, pooled over the gated
benchmark tasks, the plain scale-convergence line across 175M → 1B went from
**0.590, 0.599, 0.600, 0.598** under the A/B filter to **0.586, 0.613, 0.623,
0.629** with every scheme — a slope of **+0.010** against **+0.057** per decade of
non-embedding parameters; on the `above_66_both` population +0.100 against +0.180.
The A/B filter was suppressing measured convergence. On the 2026-09-30 tables
(90M in the ladder, the twins at every size, ZH and ES at 1.7B) the every-scheme
line reads **0.540, 0.548, 0.539, 0.550, 0.563** from 90M to 1B
(`scale_convergence_da_size_multi_axes.csv`) and 0.687 → 0.776 on `above_66_both`;
the A/B counterfactual has not been re-measured on them.

The every-scheme numbers are `scale_convergence_da_size[_above_66_both]_multi_axes.csv` as it stands;
the A/B numbers are a counterfactual, since no A/B-only table is kept. Reproduce it
by restricting `pairs_by_group`'s OVERALL branch to
`{ra["data"], rb["data"]} <= {"A", "B"}` and rerunning — do not hand-carry these
numbers, which is how three of them were wrong in the first draft of this section.

## What counts as one design axis

`scale_convergence.py --by transformation` draws one line per axis a pair differs
on, and drops any pair that moves two at once — such a pair is a decision about
neither. That only works if the axes are actually independent, and a cell's name
is not: its ladder token and its data build each encode more than one design
choice. Both are read through the registries that define the grid
([RULES.md](../RULES.md), Definitions): the ladder as `arch`, `activation`,
`optimizer` (`LADDERS`), the build as `scheme` — the recipe at that L, A/B/C —
and `T` (`DATA_SCHEMES` `letter`, `temp`). The axes are `L`, `arch`,
`activation`, `optimizer`, `scheme`, `T`, `seed`:

| build | L | `scheme` | `T` | reading |
|---|---|---|---|---|
| A | 1–50 | A | 1 | resource-ranked list, English + Russian at L = 2 |
| AT3 | 15, 30, 50 | A | 3 | A's list sampled at T = 3 |
| B | 8, 15, 30 | B | 1 | diversity-first list |
| ZH | 2 | B | 1 | English + Chinese |
| DCLMP | 1 | B | 1 | DCLM without the edu filter |
| ES | 2 | C | 1 | English + Spanish |
| FWEB | 1 | C | 1 | FineWeb |

A letter names one recipe within an L, not across L (B is DCLMP at L1, ZH at L2,
the diversity-first list at L8–L30), so a cross-L pair moves `scheme` when the
two cells read different builds at one T (`utils.moved_axes`): L2-ZH vs L8-B
moves L and scheme, L8-B vs L15-B moves L alone. Until 2026-10-05 the build was
read as three axes — the language list (A vs B), the second language (ru vs zh vs
es at L2) and the English corpus (the L1 DCLMP/FWEB cells) — and they are now
one: the multi-axis, mono-axis and seed pair sets are identical as sets before
and after, only the grouping moved (12 = 6 + 3 + 3 mono-axis pairs at the grid
seed). **B vs AT3 moves two axes** (scheme and T) and is dropped from every
per-axis line, as before.

On the 2026-10-05 report the mono-axis pairs at the reference are `language
count` 39, `depth` 10, `data scheme` 12 and `temperature` 4 (the L15, L30 and L50
AT3 cells against their scheme-A twins); the other 260 of the 325 pairs move two
or more axes.

<!-- BEGIN auto:da-explainer (da_explainer.py --pool predictivity) -->
### Decision accuracy on a toy ladder

**Toy, not measured.** Four labelled variants (Deep-A-T1, Deep-B-T1, Shallow-A-T1, Deep-A-T3) with hand-written scores; every DA in the figure is the pipeline's kernel run on them. (a) the finals per size and (b) one run along training show the two ways a ranking moves; (c) decides the six pairs three ways over both pair sets of rule 15 and rings the values at or above the reliability cut; (d) is where each definition sits on the size × checkpoint grid, with the two identities. Regenerate with `python analysis/rq02_decision_accuracy/da_explainer.py --pool predictivity`.

![Decision accuracy explained on a toy ladder](pretraining/predictivity/da_all_explainer_both_axes.png)

Key findings (definitions, so nothing to measure):

- On the toy, DA-size 0.50, DA-ckpt 0.17 and DA-goal 0.67 over the six multi-axis pairs; over the three mono-axis pairs 0.33 / 0.00 / 0.67. The tied pair (Deep-A-T1 = Deep-A-T3 at the proxy's final) is a miss wherever the other side decides it, an agreement only if both sides tie — `decision_acc_fast`'s convention, order-invariant.
- A cell of n pairs takes the values k/n: at the minimum of 3 pairs that is 0, ⅓, ⅔, 1, so a per-cell DA is read on its lattice and the figures draw the pooled ratio over tasks (`scale_convergence.py`) or the mean over cells (`by_L.py`), never one cell. The ringed cells (DA-goal multi-axis 0.67, DA-goal mono-axis 0.67) are the ones the `above_66_*` filters would keep (cut 0.66).
- 0.5 is a coin flip on every untied pair; since a one-sided tie is a miss, an uninformative proxy sits below it, at 0.5 × (1 − the share of pairs one side ties): ≈ 0.47 on the ladder (6 % one-sided ties, `agreement_da_size_per_cell_multi_axes.csv`). The seed null of `seed_uncertainty.py` is a different baseline (two seeds of one design, read in its own section).
- DA-goal at the final checkpoint is DA-size, and at the reference size DA-ckpt is DA-goal: the early-and-small grid's last column and last row are the other two figures' numbers.
- `by transformation` is the mono-axis set split by the axis a pair moves; each group needs its own three pairs. On the ladder a (task, size) cell holds 0–4 pairs for the temperature axis, 0–6 for the list, 0–10 for depth and 0–39 for the language count, so the temperature and depth groups often fall below the minimum and are NaN (`early_small_da_*_by_transformation_*`, `scale_convergence_da_size_transformation_panels*`).

Follow-ups:

- A measured twin: the same panels on one real task (`hellaswag_de`, say) with the ladder's families, so the toy orders become the observed ones.
- The lattice of the ladder's actual pair counts per cell (`median_pairs` in the scale-convergence CSVs), to show how coarse a per-cell DA is on each axis.

Files: [`da_all_explainer_both_axes.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_explainer_both_axes.png), [`da_all_explainer_both_axes.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_explainer_both_axes.csv).
<!-- END auto:da-explainer -->

## Figures, in storyline order

### 1. The full population: a smaller model is close to a coin flip

**DA-size · no filter · multi-axis pairs (`_mono_axis` twin: mono-axis) ·
pairs from `predictivity` at seed 1904 · gate `predictivity`.**
Pooled over decisions, gated at the proxy and at the reference, ≥ 3 pairs
per task; the band is the 90 % leave-one-family-out jackknife.

<!-- BEGIN auto:scale-convergence (scale_convergence.py) -->
## Scale convergence — the minimum useful scale

How small a **fully trained** model may be and still decide the way the 1.7B final checkpoint does: R_size(N) = matching decisions / comparable decisions over the gated tasks, and N_min(τ) = the smallest size with R ≥ τ (τ = 0.9). Same kernel, gate and pair minimum as the rest of rq02 — this is DA-size pooled over decisions rather than averaged over tasks, so the counts behind a point are in the CSV. The 1.7B point is 1.0 by construction. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py` (all three groupings).

**Pooled over every pair** (benchmarks; `scale_convergence_da_size_multi_axes.csv` carries BPB and the decision counts):

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| all pairs | 0.6 | 0.58 | 0.58 | 0.58 | 0.58 | 1.0 | — |

![Scale convergence, overall](pretraining/predictivity/scale_convergence_da_size_multi_axes.png)

**By language count** (benchmarks; `scale_convergence_da_size_L_multi_axes.csv` carries BPB and the decision counts):

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| L1 | 0.55 | 0.59 | 0.69 | 0.51 | 0.56 | 1.0 | — |
| L15 | 0.55 | 0.52 | 0.55 | 0.53 | 0.56 | 1.0 | — |
| L2 | 0.53 | 0.51 | 0.51 | 0.53 | 0.46 | 1.0 | — |
| L30 | 0.56 | 0.55 | 0.54 | 0.55 | 0.55 | 1.0 | — |
| L50 | 0.64 | 0.59 | 0.61 | 0.58 | 0.62 | 1.0 | — |
| L8 | 0.51 | 0.51 | 0.5 | 0.52 | 0.51 | 1.0 | — |
| all pairs | 0.6 | 0.58 | 0.58 | 0.58 | 0.58 | 1.0 | — |

![Scale convergence, L](pretraining/predictivity/scale_convergence_da_size_L_multi_axes.png)

**By design axis** (benchmarks; `scale_convergence_da_size_transformation_multi_axes.csv` carries BPB and the decision counts):

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| 2nd language (ru vs zh vs es) | 0.53 | 0.51 | 0.49 | 0.53 | 0.48 | 1.0 | — |
| English corpus (edu filter on vs off) | 0.55 | 0.61 | 0.78 | 0.62 | 0.66 | 1.0 | — |
| all pairs | 0.6 | 0.58 | 0.58 | 0.58 | 0.58 | 1.0 | — |
| depth (deep vs shallow) | 0.55 | 0.5 | 0.51 | 0.47 | 0.5 | 1.0 | — |
| language count | 0.58 | 0.57 | 0.57 | 0.56 | 0.55 | 1.0 | — |
| language list (A vs B) | 0.5 | 0.47 | 0.46 | 0.48 | 0.53 | 1.0 | — |
| temperature (T=1 vs T=3) | 0.63 | 0.6 | 0.62 | 0.62 | 0.62 | 1.0 | — |

![Scale convergence, transformation](pretraining/predictivity/scale_convergence_da_size_transformation_multi_axes.png)
<!-- END auto:scale-convergence -->

<!-- BEGIN auto:scale-convergence-transformation-panels (scale_convergence.py --by transformation) -->
## Scale convergence per design axis, one panel each

The `--by transformation` lines above drawn one axis per panel, with the panel's own leave-one-family-out band and the pooled `all pairs` line faint behind it. DA-size pooled over decisions, every pair at seed 1904 (`predictivity_all`), gated with `predictivity`'s mask, ≥ 3 pairs per task; task counts under the points. Same table as `scale_convergence_da_size_transformation_multi_axes.csv`. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by transformation`.

![Scale convergence per design axis](pretraining/predictivity/scale_convergence_da_size_transformation_panels_multi_axes.png)

| axis | families | 90M R [lo, hi] (tasks) | 175M R [lo, hi] (tasks) | 350M R [lo, hi] (tasks) | 600M R [lo, hi] (tasks) | 1B R [lo, hi] (tasks) | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| all pairs | 26 | 0.60 [0.57, 0.62] (1118) | 0.58 [0.55, 0.60] (1157) | 0.58 [0.56, 0.61] (1193) | 0.58 [0.55, 0.60] (1227) | 0.58 [0.55, 0.61] (1269) | never |
| 2nd language (ru vs zh vs es) | 3 | 0.53 [nan, nan] (121) | 0.51 [nan, nan] (122) | 0.49 [nan, nan] (126) | 0.53 [nan, nan] (129) | 0.48 [nan, nan] (129) | never |
| English corpus (edu filter on vs off) | 3 | 0.55 [nan, nan] (23) | 0.61 [nan, nan] (24) | 0.78 [nan, nan] (6) | 0.62 [nan, nan] (31) | 0.66 [nan, nan] (31) | never |
| depth (deep vs shallow) | 20 | 0.55 [0.50, 0.60] (890) | 0.50 [0.47, 0.54] (924) | 0.51 [0.50, 0.53] (953) | 0.47 [0.42, 0.52] (987) | 0.50 [0.46, 0.55] (1015) | never |
| language count | 21 | 0.58 [0.54, 0.61] (830) | 0.57 [0.54, 0.59] (861) | 0.57 [0.54, 0.60] (891) | 0.56 [0.54, 0.59] (922) | 0.55 [0.52, 0.58] (948) | never |
| language list (A vs B) | 12 | 0.50 [0.44, 0.55] (242) | 0.47 [0.43, 0.51] (247) | 0.46 [0.43, 0.50] (257) | 0.48 [0.46, 0.50] (266) | 0.53 [0.50, 0.55] (268) | never |
| temperature (T=1 vs T=3) | 8 | 0.63 [0.57, 0.69] (830) | 0.60 [0.55, 0.65] (861) | 0.62 [0.59, 0.64] (891) | 0.62 [0.58, 0.66] (922) | 0.62 [0.56, 0.67] (948) | never |

Key findings:

- **2nd language (ru vs zh vs es)** (3 families): R = 0.48 at 1B [nan, nan] over 129 tasks; no proxy reaches τ.
- **English corpus (edu filter on vs off)** (3 families): R = 0.66 at 1B [nan, nan] over 31 tasks; no proxy reaches τ.
- **depth (deep vs shallow)** (20 families): R = 0.50 at 1B [0.46, 0.55] over 1015 tasks; no proxy reaches τ.
- **language count** (21 families): R = 0.55 at 1B [0.52, 0.58] over 948 tasks; no proxy reaches τ.
- **language list (A vs B)** (12 families): R = 0.53 at 1B [0.50, 0.55] over 268 tasks; no proxy reaches τ.
- **temperature (T=1 vs T=3)** (8 families): R = 0.62 at 1B [0.56, 0.67] over 948 tasks; no proxy reaches τ.
- The bands are leave-one-design-variant-out over the families in the column, not seed noise; a band on 4 families is four numbers and only says which variant the line hinges on.

Follow-ups:

- The `above_66_size` twin per panel (`scale_convergence_da_size_transformation_panels_above_66_size_multi_axes.png`): the same split on the cells that rank reliably.
- A per-axis panel grid on the L8 languages only (`--langs L8`), so the language-count panel is read on one task set.
- The second-language and English-corpus panels fill in once their 1.7B cells are in the report (rule 9).

Files: [`scale_convergence_da_size_transformation_panels_multi_axes.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_panels_multi_axes.png), [`scale_convergence_da_size_transformation_panels_multi_axes.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_panels_multi_axes.csv).
<!-- END auto:scale-convergence-transformation-panels -->

**Key findings**

- On every gated task DA-size is 0.54 at 90M and 0.56 at 1B: 0.540
  [0.516, 0.564], 0.548 [0.525, 0.571], 0.539 [0.517, 0.561], 0.550
  [0.527, 0.573], 0.563 [0.529, 0.598] at 90M, 175M, 350M, 600M, 1B on
  302–453 tasks (25 families); the mono-axis pairs read 0.506, 0.519, 0.507,
  0.518, 0.523 (`scale_convergence_da_size_mono_axis.csv`). Nowhere near τ = 0.90,
  so N_min is undefined.
- Per-language BPB does better (0.91 / 0.91 / 0.82 / 0.58 / 0.82 at 90M …
  1B on the BPB panel) and the two aggregates best of all: `bpb_macro`
  0.92–0.97 and `train_loss` 0.79–0.93 from any size (highlight block). The
  design differences the
  ladder measures move most benchmark scores by less than their noise at any
  one size; BPB, which sums over every token, sees them.
- By design axis (`scale_convergence_da_size_transformation_multi_axes.csv`) the unfiltered
  lines sit at 0.48–0.57 at every size; by language count the regime lines
  are unordered (figure 4).
- With and without the twins: on the ungated multi-axis table the originals
  alone read 0.50 → 0.52 from 175M to 1B and the twins alone 0.47 → 0.48
  ([rq00 figure 3](../rq00_gate_and_curves/README.md#3-the-reformulated-twins-move-whole-families-across-the-gate)),
  so every pooled figure here is a little lower with the twins in, and no
  conclusion depends on them.

**Follow-ups**

- Effect-size-resolved DA: per pair the reference's gap in seed sds (rq05's
  `seed_sd`), DA on the pairs above 2 sds and DA against the gap per proxy
  size — most pairs differ in L or list membership, so on a language's BPB
  one variant saw the language and the other did not, which any proxy orders
  correctly and which dominates the BPB line (cross-task shows it, figure 10).
- A noise ceiling on the DA-size panel: no seed replicate exists at 1.7B, so
  the test-retest ceiling of figure 7 stops at 1B; a second 1.7B seed on two
  cells would put a band on this figure.
- Read the `_flops` twins (`scale_convergence_da_size_*_flops.png`) with one line per
  size and the annealed points marked; a bigger model's first (peak-LR)
  checkpoint beside a smaller model's annealed final is what makes the
  compute axis zigzag.

GitHub: [scale_convergence_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_multi_axes.png) · [scale_convergence_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_mono_axis.png) · [scale_convergence_da_size_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_mono_axis.csv) ·
GitHub: [scale_convergence_da_size_L_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L_multi_axes.png) · [scale_convergence_da_size_L_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_transformation_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_multi_axes.png) · [scale_convergence_da_size_transformation_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_transformation_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_mono_axis.png) · [scale_convergence_da_size_transformation_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_mono_axis.csv)

What this leaves open is whether the coin flip is the average of a few
reliable tasks and many hopeless ones — figure 2.

### 2. The paper figure is a conditional statement

**Left panel DA-size, middle DA-ckpt, right DA-goal · filter per panel:
`above_66_size` / `above_66_ckpt` / `above_66_either` (the
`_either_transformation` stem) · mono-axis pairs (`_mono_axis`) · pairs from
`predictivity` at seed 1904 · gate `predictivity`.** The paper
embeds `rq2_da_all_multi_axes.png` (`paper_rq2.py`, no filter, multi-axis, copied by
`documents/paper/figures/make_rq_figures.py`); its variants share the
composition and differ only in filter and pair set: `rq2_da_all_mono_axis`
(mono-axis, no filter), `rq2_da_all_above_80_*` (the `late` reduction at 0.80, both
axes), `rq2_da_all_above_66_both[_transformation]_*` (one population on all three
panels, kept for comparison only) and `rq2_da_all_above_66_own_*` (each panel its own
cut).

![RQ2, per-panel cuts, mono-axis pairs](pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis.png)

*Left: DA-size per design axis (mono-axis pairs) and pooled, over the tasks
whose median DA-size over the proxy sizes is ≥ 0.66 (49 tasks at 90M, 72 at
1.7B); x is non-embedding parameters, the hollow 1.7B point is 1.0 by
construction, dotted τ = 0.90. Middle: DA-ckpt over the tasks whose median
DA-ckpt is ≥ 0.66 (86–114 tasks), each proxy against its own final, x in
Chinchilla multiples, dotted 0.75. Right: DA-goal over the tasks passing
either cut (90–136), the same checkpoints against the reference's final; the
panels share the y axis, so the vertical distance between middle and right is
what the proxy's SIZE costs on top of reading it early. The left panel is
therefore not the same cells as the middle and right: three claims, not one
population seen three ways. CSVs: `rq2_da_all_above_66_either_transformation_mono_axis.csv`,
`early_small_da_ckpt_by_L_above_66_ckpt_mono_axis.csv`,
`early_small_da_goal_by_L_above_66_either_mono_axis.csv`.*

| line (mono-axis, `above_66_size` tasks) | 90M | 175M | 350M | 600M | 1B | tasks |
|---|---|---|---|---|---|---|
| all pairs | 0.644 | 0.720 | 0.699 | 0.742 | 0.759 | 49–70 |
| temperature (T = 1 vs 3) | 0.683 | 0.804 | 0.821 | 0.827 | 0.828 | 29–41 |
| language count | 0.678 | 0.756 | 0.729 | 0.829 | 0.789 | 29–41 |
| depth (deep vs shallow) | 0.658 | 0.661 | 0.643 | 0.569 | 0.746 | 29–43 |
| language list (A vs B) | 0.389 | 0.583 | 0.479 | 0.537 | 0.500 | 6–10 |
| second language (ru vs zh vs es) | 0.556 | 0.778 | 0.778 | 0.833 | 0.833 | 3–4 |

**Key findings**

- On the filtered population the pooled DA-size rises from 0.64 at 90M
  (0.72 at 175M) to 0.76 at 1B; temperature and language count are the
  decisions a 175M proxy already reads at 0.76–0.83; depth dips to 0.57 at
  600M (unexplained); the language-list decision never leaves 0.39–0.58 on
  6–10 tasks — a coin flip,
  not below it. On the unfiltered tables the same four lines sit at 0.48–0.57
  (figure 1).
- The rise is partly the cut: the filter keeps the tasks whose DA-size
  cleared 0.66 and then plots DA-size on them; on `predictivity` the
  82 passers read 0.78 at 1B against 0.52 for the other 746 tasks. Every
  filtered `rq2_*` figure, including `above_80` whose 1B point is truncated
  at 0.80 by construction, is "on the cells that rank reliably, this is how
  the reliability scales"; figure 1 is the unconditional one.
- Reading a proxy early is cheap, but so is reading a second seed: DA-ckpt
  on its own filtered population climbs from 0.56–0.61 at 0.5C to 0.81–0.88
  at 4.5C (0.863, 0.875, 0.806, 0.809, 0.839 from 175M to 1.7B); on every
  gated task the 175M run reads 0.83 at 90 % and two seeds of one design
  read 0.81 there (figure 7).
- Reading a smaller model is not: under DA-goal the sizes separate and stay
  separated — the 175M line reads 0.48 at 0.5C and 0.52–0.57 thereafter
  (0.53 at 5C), the 1B line 0.56 → 0.67, the reference's own run 0.58 → 0.82.
  Training the small proxy longer does not close the gap; the binding
  constraint is its scale.
- Which tasks rank reliably: on the multi-axis tables 82 of 473 gated tasks
  pass the DA-size cut, 178 the DA-ckpt cut, 189 either and 71 both; mono-axis
  63 / 141 / 160 / 44. The DA-ckpt cut is the permissive one because its
  median runs over up to 45 cells, the reference's own run included.
- Caveat: the language-list line rests on 24 comparable decisions at 175M and
  42 at 1B (4–7 tasks, 12 families), the temperature line on 8 families; those
  two are the least stable lines in the panel.

**Follow-ups**

- Quote figure 1 beside every filtered `rq2_*` figure in the paper, or restate
  the paper figure as conditional in its caption; never call a filtered
  figure "free of selection bias".
- Draw the seed null of figure 7 on the middle panel, so the "reading early
  is cheap" claim is read against within-run persistence.
- Mark the annealed points (≥ 4C and the finals): every checkpoint up to 4C
  is read at the peak learning rate under WSD, so DA-ckpt compares two
  regimes. Short cooldown branches from the 1C/2C checkpoints of a few
  350M/600M cells would measure the cost of the missing anneal (training
  compute; a planning decision).
- Bootstrap over design variants (not tasks, which share the variants) for a
  90 % interval on every line: with 11 variants a DA moves in steps of 1/55.

GitHub: [rq2_da_all_above_66_either_transformation_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis.png) · [rq2_da_all_above_66_either_transformation_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis.csv) ·
GitHub: [rq2_da_all_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_multi_axes.png) · [rq2_da_all_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_multi_axes.csv) ·
GitHub: [rq2_da_all_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_mono_axis.png) · [rq2_da_all_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_mono_axis.csv) ·
GitHub: [rq2_da_all_above_80_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_80_multi_axes.png) · [rq2_da_all_above_80_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_80_multi_axes.csv) ·
GitHub: [rq2_da_all_above_66_both_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_multi_axes.png) · [rq2_da_all_above_66_both_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_multi_axes.csv) ·
GitHub: [rq2_da_all_above_66_both_transformation_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_transformation_multi_axes.png) · [rq2_da_all_above_66_both_transformation_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_transformation_multi_axes.csv) ·
GitHub: [rq2_da_all_above_66_own_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_own_multi_axes.png) · [rq2_da_all_above_66_own_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_own_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_above_66_size_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_above_66_size_mono_axis.png) · [scale_convergence_da_size_above_66_size_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_above_66_size_mono_axis.csv)

The population behind the cuts — which (benchmark, language) cells rank
reliably at all, and in which languages — is the block below.

**Reliable cells · DA-size and DA-ckpt · every cut × reduction ·
multi-axis (the table carries mono-axis too) · pairs from
`predictivity` · gate `predictivity`.** The verdicts from 2026-09-22
on gate DA-size at the reference too (rule 1): before that, three cells at
chance at 1.7B counted as reliable, and the `above_66_both` population read
23 cells; the 2026-09-23 snapshot with the twins at 175M and 350M took it
to 71.

<!-- BEGIN auto:reliable-tasks (reliable_tasks.py --pool predictivity) -->
## Which benchmark-language cells rank reliably

Per language, how many benchmarks clear DA ≥ 0.8 on DA-size (a proxy size's final ranking vs the reference's) and on DA-ckpt (an earlier checkpoint vs the same size's final), reducing each task's cells with `late` (one fixed cell per axis, so no cell is chosen by its value). Cells need ≥ 3 pairs (rule 5) and must survive the above-random gate (rule 1). The decisions come from the `predictivity` pool — every data build at the grid seed, which is the population `by_L` and `scale_convergence` pair over — while the gate and this folder stay with `predictivity`; the table carries one row per pair set (rule 15) and the figures show `multi-axis`. `da_all_reliable_tasks_both_axes.csv` holds the per-task values for every reduction and is threshold-free — each figure is one view of it. Regenerate with `python analysis/rq02_decision_accuracy/reliable_tasks.py --pool predictivity`.

| language | benchmarks evaluated | DA-size | DA-ckpt | either | both |
|---|---|---|---|---|---|
| en | 138 | 3 | 9 | 10 | 2 |
| ru | 32 | 4 | 6 | 8 | 2 |
| zh | 44 | 1 | 3 | 4 | 0 |
| de | 31 | 1 | 7 | 7 | 1 |
| ja | 24 | 0 | 3 | 3 | 0 |
| es | 125 | 2 | 15 | 16 | 1 |
| fr | 33 | 2 | 5 | 6 | 1 |
| it | 29 | 2 | 9 | 10 | 1 |
| pt | 31 | 2 | 5 | 6 | 1 |
| pl | 22 | 0 | 3 | 3 | 0 |
| nl | 23 | 2 | 3 | 4 | 1 |
| id | 26 | 1 | 2 | 3 | 0 |
| vi | 30 | 2 | 3 | 3 | 2 |
| fa | 20 | 0 | 2 | 2 | 0 |
| tr | 24 | 1 | 5 | 6 | 0 |
| th | 10 | 1 | 1 | 2 | 0 |
| uk | 24 | 1 | 6 | 6 | 1 |
| el | 27 | 1 | 0 | 1 | 0 |
| ko | 19 | 1 | 2 | 3 | 0 |
| cs | 16 | 1 | 2 | 3 | 0 |
| sv | 21 | 5 | 6 | 9 | 2 |
| hu | 22 | 0 | 2 | 2 | 0 |
| ro | 15 | 1 | 2 | 2 | 1 |
| no | 9 | 0 | 1 | 1 | 0 |
| da | 18 | 3 | 4 | 6 | 1 |
| bg | 19 | 4 | 3 | 7 | 0 |
| fi | 16 | 3 | 2 | 5 | 0 |
| hi | 36 | 1 | 3 | 4 | 0 |
| bn | 32 | 0 | 3 | 3 | 0 |
| he | 19 | 3 | 3 | 5 | 1 |
| ta | 19 | 3 | 1 | 4 | 0 |
| ka | 18 | 3 | 4 | 7 | 0 |
| ml | 14 | 3 | 2 | 5 | 0 |
| ar | 87 | 1 | 9 | 10 | 0 |

**How many cells pass, by cut and reduction** — the cut is a choice, and this is its whole sensitivity:

| threshold | reduction | tasks passing both | languages | benchmarks |
|---|---|---|---|---|
| 0.8 | late | 18 | 14 | hellaswag, include_v2_en, include_v2_og, lambada_openai_mt, rf_global_mmlu_full |
| 0.8 | mean | 3 | 3 | hellaswag |
| 0.8 | median | 4 | 4 | hellaswag |
| 0.8 | max | 42 | 23 | hellaswag, include_base_44, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, rf_belebele, rf_global_mmlu_full, rfgm_belebele, rfgm_include_base_44, xstorycloze |
| 0.75 | late | 22 | 16 | hellaswag, include_v2_en, include_v2_og, lambada_openai_mt, rf_global_mmlu_full |
| 0.75 | mean | 4 | 4 | hellaswag |
| 0.75 | median | 7 | 7 | hellaswag |
| 0.75 | max | 67 | 29 | hellaswag, include_base_44, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_belebele, rfgm_include_base_44, xnli, xstorycloze |
| 0.66 | late | 66 | 27 | arc, arc_mt, hellaswag, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rfgm_belebele, rfgm_include_base_44, xcopa, xnli, xstorycloze, xwinograd |
| 0.66 | mean | 18 | 11 | arc, hellaswag, include_v2_og, lambada_openai_mt, rf_global_mmlu_full, xstorycloze |
| 0.66 | median | 24 | 15 | arc, hellaswag, include_v2_og, lambada_openai_mt, multiblimp, rf_global_mmlu_full, xstorycloze, xwinograd |
| 0.66 | max | 158 | 33 | arc, arc_mt, hellaswag, include_base_44, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_bbh_mcq, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rf_mmlu, rfgm_belebele, rfgm_include_base_44, xcopa, xnli, xstorycloze, xwinograd |

![Reliable benchmark-language cells](pretraining/predictivity/da_size_vs_da_ckpt_reliable_tasks_80_late_multi_axes.png)
<!-- END auto:reliable-tasks -->

[da_all_reliable_tasks_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_reliable_tasks_both_axes.csv) ·
[da_all_reliable_by_language_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_reliable_by_language_multi_axes.csv) ·
[da_size_vs_da_ckpt_reliable_tasks_80_late_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_vs_da_ckpt_reliable_tasks_80_late_multi_axes.png) ·
[da_size_vs_da_ckpt_reliable_tasks_66_median_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_vs_da_ckpt_reliable_tasks_66_median_multi_axes.png) ·
[da_size_vs_da_ckpt_reliable_tasks_75_late_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_vs_da_ckpt_reliable_tasks_75_late_multi_axes.png)
(one PNG per cut × reduction, `da_size_vs_da_ckpt_reliable_tasks_<t>_<red>_multi_axes.png`, all views of
`da_all_reliable_tasks_both_axes.csv`).

### 3. Early and small: DA-goal and DA-ckpt on the ten checkpoints

**`early_small_da_{ckpt,goal}_by_L_*`: DA-ckpt / DA-goal · mean over tasks ·
filter per stem (none, `_above_80`, `_above_66_ckpt`, `_above_66_either`,
`_above_66_both`; `_with_bpb` adds the per-size BPB lines) · multi-axis
unless `_mono_axis` · pairs from `predictivity` (`by_L.py` pools every
scheme at the grid seed) · gate `predictivity`.** The `early_small_da_goal_multi_axes.png` grid,
the `safe_*` maps and the Results block below read `compute_da.py`'s own
tables on `predictivity` (every build since 2026-10-05, the A/B pool before;
multi-axis, no filter, gate `predictivity`) — DA-goal at 1C–5C per proxy, the mean over tasks, and the
smallest safe level (DA ≥ 0.75 over ≥ 3 pairs, held at every larger level).

<!-- BEGIN auto:results (da_per_benchmark.py --pool predictivity) -->
## Results

Numbers from the `predictivity` pool (`da_all_per_task_both_axes.csv`, pairs from `da_all_n_pairs_per_task_both_axes.csv`, gate from rq00). Regenerate with `python analysis/rq02_decision_accuracy/da_per_benchmark.py --pool predictivity`.

**DA-size by proxy size** (`n` tasks; median pairs per cell):

| comparison | benchmarks | n | pairs | BPB | n |
|---|---|---|---|---|---|
| 90M → 1.7B | 0.55 | 890 | 28 | 0.91 | 34 |
| 175M → 1.7B | 0.53 | 924 | 28 | 0.91 | 34 |
| 350M → 1.7B | 0.54 | 953 | 28 | 0.82 | 34 |
| 600M → 1.7B | 0.54 | 987 | 28 | 0.58 | 34 |
| 1B → 1.7B | 0.53 | 1015 | 28 | 0.82 | 34 |

![DA-size by family](pretraining/predictivity/da_size_by_family_multi_axes.png)

**DA-ckpt by bucket and fraction of the run** (mean over the above-random benchmark tasks):

| bucket | 10 % | 20 % | 30 % | 40 % | 50 % | 60 % | 70 % | 80 % | 90 % |
|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.48 | 0.51 | 0.52 | 0.56 | 0.57 | 0.60 | 0.62 | 0.65 | 0.75 |
| 175M | 0.51 | 0.51 | 0.53 | 0.54 | 0.59 | 0.59 | 0.62 | 0.66 | 0.75 |
| 350M | 0.49 | 0.51 | 0.54 | 0.57 | 0.59 | 0.62 | 0.67 | 0.72 | 0.79 |
| 600M | 0.48 | 0.50 | 0.53 | 0.55 | 0.57 | 0.59 | 0.60 | 0.64 | 0.74 |
| 1B | 0.48 | 0.51 | 0.53 | 0.54 | 0.57 | 0.59 | 0.61 | 0.64 | 0.74 |
| 1.7B | 0.50 | 0.54 | 0.54 | 0.56 | 0.56 | 0.59 | 0.59 | 0.63 | 0.71 |
<!-- END auto:results -->

<!-- BEGIN auto:early-small (early_small.py --pool predictivity) -->
## Early and small, as a ranking

Numbers from the `predictivity` pool: every design variant at a proxy size, read at 1C–5C of training (C = the Chinchilla-optimal 20 tokens per parameter; every run trains 5C, so 1C is 20 % of it), ranked against the same variants at the 1.7B final checkpoint (the 5C column is DA-size, the 1.7B row is that size's DA-ckpt). A benchmark task counts only where it clears chance at the proxy size and at 1.7B. Regenerate with `python analysis/rq02_decision_accuracy/early_small.py --pool predictivity`.

- **bpb** — smallest proxy whose mean agreement with the 1.7B final ranking reaches 0.75: **90M at 0.5C** (0.90).
- **all benchmarks** — no (proxy, checkpoint) reaches a mean agreement of 0.75.
- **Smallest safe size per (benchmark, language)** — never: 686, 1B: 77, 90M: 14, 600M: 12, 350M: 4, 175M: 2 of 795 cells.

![rq02 in one figure](pretraining/predictivity/highlights_da_all_multi_axes.png)

**bpb** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 34 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.90 | 0.92 | 0.91 | 0.90 | 0.90 | 0.88 | 0.88 | 0.89 | 0.91 | 0.91 |
| 175M | 0.89 | 0.84 | 0.88 | 0.90 | 0.86 | 0.88 | 0.86 | 0.88 | 0.90 | 0.91 |
| 350M | 0.71 | 0.73 | 0.72 | 0.70 | 0.77 | 0.79 | 0.75 | 0.76 | 0.80 | 0.82 |
| 600M | 0.52 | 0.53 | 0.54 | 0.57 | 0.57 | 0.59 | 0.56 | 0.57 | 0.58 | 0.58 |
| 1B | 0.65 | 0.66 | 0.70 | 0.75 | 0.74 | 0.80 | 0.77 | 0.80 | 0.81 | 0.82 |
| 1.7B | 0.81 | 0.90 | 0.93 | 0.95 | 0.96 | 0.95 | 0.97 | 0.96 | 0.99 |  |

**all benchmarks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 1015 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.49 | 0.47 | 0.49 | 0.48 | 0.50 | 0.51 | 0.50 | 0.51 | 0.51 | 0.55 |
| 175M | 0.51 | 0.50 | 0.49 | 0.50 | 0.50 | 0.50 | 0.49 | 0.49 | 0.48 | 0.53 |
| 350M | 0.47 | 0.47 | 0.48 | 0.47 | 0.47 | 0.49 | 0.49 | 0.49 | 0.49 | 0.54 |
| 600M | 0.47 | 0.47 | 0.48 | 0.49 | 0.50 | 0.49 | 0.50 | 0.50 | 0.51 | 0.54 |
| 1B | 0.49 | 0.50 | 0.50 | 0.50 | 0.52 | 0.51 | 0.51 | 0.52 | 0.51 | 0.53 |
| 1.7B | 0.50 | 0.54 | 0.54 | 0.56 | 0.56 | 0.59 | 0.59 | 0.63 | 0.71 |  |

![Early and small](pretraining/predictivity/early_small_da_goal_multi_axes.png)

![Early and small per benchmark](pretraining/predictivity/early_small_da_goal_by_benchmark_multi_axes.png)

![Early and small per language](pretraining/predictivity/early_small_da_goal_by_language_multi_axes.png)

**Smallest safe level per language and benchmark** (DA ≥ 0.75 over ≥ 3 pairs, held at every larger level — for FLOPs, at every costlier (size, checkpoint) cell; red = never, grey = filtered out by the above-random gate, white = no value). Each figure's table sits next to it under the same name:

![Smallest safe size](pretraining/predictivity/safe_size_da_size_multi_axes.png)

![Smallest safe checkpoint](pretraining/predictivity/safe_checkpoint_da_ckpt_multi_axes.png)

![Smallest safe FLOPs](pretraining/predictivity/safe_flops_da_goal_multi_axes.png)

![DA-size per benchmark](pretraining/predictivity/da_size_by_benchmark_multi_axes.png)

![DA-size per language](pretraining/predictivity/da_size_by_language_multi_axes.png)
<!-- END auto:early-small -->

<!-- BEGIN auto:by-L (by_L.py --pool predictivity) -->
## Per language count

**Pairs per L** — the design variants the grid plans at seed 1904, and per proxy size the pairs usable against 1.7B (both members planned at that size and at 1.7B): planned / with data today on BPB / on the benchmarks / on the training loss. ZH and ES run to 1.7B and are the second and third L2 families. A cell below MIN_PAIRS (3) families is left empty (rule 5), so a thin L shows blanks rather than a 0/1 reading.

| L | variants | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| 1 | L1-dclmP-deep, L1-deep, L1-fweb-deep, L1-shallow | 6 / 3/3/3 | 6 / 3/3/3 | 6 / 3/3/3 | 6 / 3/3/3 | 6 / 3/3/3 |
| 2 | L2-ES-deep, L2-ZH-deep, L2-deep, L2-shallow | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 |
| 8 | L8-deep, L8-schemeB-deep, L8-schemeB-shallow, L8-shallow | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 |
| 15 | L15-AT3-deep, L15-deep, L15-schemeB-deep, L15-schemeB-shallow, L15-shallow | 10 / 10/10/10 | 10 / 10/10/10 | 10 / 10/10/10 | 10 / 10/10/10 | 10 / 10/10/10 |
| 30 | L30-AT3-deep, L30-deep, L30-schemeB-deep, L30-schemeB-shallow, L30-shallow | 10 / 10/10/10 | 10 / 10/10/10 | 10 / 10/10/10 | 10 / 10/10/10 | 10 / 10/10/10 |
| 50 | L50-AT3-deep, L50-AT3-shallow, L50-deep, L50-shallow | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 |

The early-and-small reading one L at a time: pairs of design variants that share the L (seed 1904 of every scheme, `predictivity_all`), against the 1.7B final ranking, on the ten evaluated checkpoints of every run; a cell needs ≥ 3 pairs (rq02's rule), which today leaves out every L with one pair (the table above); the first panel pools every pair at that seed, every scheme included (`da_all_pooled_per_task_multi_axes.csv`). `da_all_by_L_per_task_multi_axes.csv` also carries each size's DA-ckpt within the L (`da_own`); rq04 reads both tables. The `_mono_axis` twins of every table and figure are the same over the one-axis pairs (rule 15). Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool predictivity [--axes mono-axis]`.

Two of rq02's three decision accuracies have a checkpoint axis and so a figure here. **DA-goal** ranks the proxy at any checkpoint against the 1.7B final checkpoint; **DA-ckpt** ranks it against its own size's final checkpoint, so the 175M line asks what 175M would have decided early and what it misses is the checkpoint alone. The distance between the two is what the proxy *size* costs, and the 1.7B line is the same curve in both — at the reference the definitions coincide. DA-ckpt has no 5C column: a run's final checkpoint is its own reference. The third, **DA-size**, is DA-goal read at 5C alone and lives in `da_all_per_task_both_axes.csv`. Each figure comes in a benchmarks-only version and a `_with_bpb` one that adds the solid per-size BPB lines; all four share the y axis, so any two overlay.

![DA-goal per L](pretraining/predictivity/early_small_da_goal_by_L_multi_axes.png)

![DA-goal per L, with BPB](pretraining/predictivity/early_small_da_goal_by_L_with_bpb_multi_axes.png)

![DA-ckpt per L](pretraining/predictivity/early_small_da_ckpt_by_L_multi_axes.png)

![DA-ckpt per L, with BPB](pretraining/predictivity/early_small_da_ckpt_by_L_with_bpb_multi_axes.png)

A third variant of each restricts the mean to the (benchmark, language) cells that rank reliably on BOTH axes (DA-size and DA-ckpt each ≥ 0.8, `reliable_tasks.py`): the plain panels average over every gated benchmark, these average over the benchmarks that work.

![DA-goal per L, reliable cells only](pretraining/predictivity/early_small_da_goal_by_L_above_80_multi_axes.png)

![DA-ckpt per L, reliable cells only](pretraining/predictivity/early_small_da_ckpt_by_L_above_80_multi_axes.png)
<!-- END auto:by-L -->

<!-- BEGIN auto:early-small-by-transformation (by_L.py --pool predictivity --by transformation) -->
## Early and small per design axis

The per-L reading above pools every design axis inside an L; this one splits the MONO-AXIS pairs at seed 1904 (`predictivity_all`, every scheme) by the one axis each pair moves — language count, depth (deep vs shallow), language list (A vs B), temperature (T=1 vs T=3), 2nd language (ru vs zh vs es), English corpus (edu filter on vs off) — one panel per axis and a first panel over every mono-axis pair (median pairs per cell up to 20). Same gate (rule 1), pair minimum (rule 5) and filter variants as the per-L figures; the `above_66_ckpt` twin filters the DA-ckpt figure and `above_66_either` the DA-goal one, both on the mono-axis reliability (rule 15). Task counts sit at the end of every line and the populations differ between panels and sizes (rule 13). Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool predictivity --by transformation`.

![DA-ckpt per design axis](pretraining/predictivity/early_small_da_ckpt_by_transformation_mono_axis.png)

**DA-ckpt** (against the proxy size's own final; cell = mean DA at the first → last drawn checkpoint, tasks behind the line in brackets):

| axis (DA-ckpt, 0.5C → 4.5C, tasks) | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| all | 0.50 → 0.75 (324) | 0.51 → 0.75 (349) | 0.50 → 0.81 (386) | 0.51 → 0.75 (417) | 0.51 → 0.74 (458) | 0.51 → 0.72 (503) |
| language count | 0.48 → 0.76 (235) | 0.48 → 0.73 (256) | 0.48 → 0.79 (285) | 0.49 → 0.75 (316) | 0.50 → 0.74 (343) | 0.51 → 0.72 (374) |
| depth (deep vs shallow) | 0.48 → 0.74 (250) | 0.49 → 0.75 (273) | 0.48 → 0.79 (301) | 0.48 → 0.73 (335) | 0.47 → 0.73 (365) | 0.49 → 0.72 (397) |
| language list (A vs B) | 0.39 → 0.75 (63) | 0.49 → 0.70 (61) | 0.49 → 0.81 (70) | 0.52 → 0.74 (79) | 0.46 → 0.73 (83) | 0.45 → 0.69 (92) |
| temperature (T=1 vs T=3) | 0.49 → 0.72 (235) | 0.52 → 0.77 (256) | 0.52 → 0.84 (285) | 0.54 → 0.77 (316) | 0.52 → 0.73 (343) | 0.54 → 0.73 (374) |
| 2nd language (ru vs zh vs es) | 0.42 → 0.71 (28) | 0.49 → 0.83 (25) | 0.39 → 0.70 (28) | 0.55 → 0.77 (31) | 0.46 → 0.78 (32) | 0.61 → 0.75 (34) |
| English corpus (edu filter on vs off) | 0.56 → 0.82 (28) | 0.63 → 0.75 (25) | 0.56 → 0.83 (6) | 0.58 → 0.90 (31) | 0.60 → 0.79 (32) | 0.67 → 0.78 (34) |

Key findings:

- At 90M the DA-ckpt line clears 0.75 and stays there — language count: from 4.5C (0.76); depth (deep vs shallow): never (max 0.74); language list (A vs B): from 4.5C (0.75); temperature (T=1 vs T=3): never (max 0.72); 2nd language (ru vs zh vs es): never (max 0.71); English corpus (edu filter on vs off): from 4C (0.76).
- At 1.7B — language count: 0.51 → 0.72 (374); depth (deep vs shallow): 0.49 → 0.72 (397); language list (A vs B): 0.45 → 0.69 (92); temperature (T=1 vs T=3): 0.54 → 0.73 (374); 2nd language (ru vs zh vs es): 0.61 → 0.75 (34); English corpus (edu filter on vs off): 0.67 → 0.78 (34).

Follow-ups:

- A `_with_bpb` variant per axis, to see whether BPB decides the temperature and the second language earlier than the benchmarks do.
- The same panels on the L8 languages only (`scale_convergence.py --langs L8` does it for DA-size), so the language-count panel is read on one task set.
- Once BT3 trains, the temperature panel gains the B-vs-BT3 pairs and the list panel AT3-vs-BT3 with no code change.

![DA-goal per design axis](pretraining/predictivity/early_small_da_goal_by_transformation_mono_axis.png)

**DA-goal** (against the 1.7B final):

| axis (DA-goal, 0.5C → 5C, tasks) | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| all | 0.50 → 0.59 (302–1118) | 0.49 → 0.56 (341–1157) | 0.47 → 0.57 (377–1193) | 0.49 → 0.56 (411–1227) | 0.51 → 0.58 (453–1269) | 0.51 → 0.72 (503) |
| language count | 0.51 → 0.56 (221–830) | 0.50 → 0.56 (252–861) | 0.47 → 0.56 (282–891) | 0.53 → 0.57 (313–922) | 0.52 → 0.55 (339–948) | 0.51 → 0.72 (374) |
| depth (deep vs shallow) | 0.48 → 0.56 (235–890) | 0.50 → 0.50 (269–924) | 0.47 → 0.52 (298–953) | 0.42 → 0.47 (332–987) | 0.46 → 0.51 (360–1015) | 0.49 → 0.72 (397) |
| language list (A vs B) | 0.50 → 0.50 (55–242) | 0.47 → 0.47 (60–247) | 0.47 → 0.47 (70–257) | 0.44 → 0.48 (79–266) | 0.48 → 0.53 (81–268) | 0.45 → 0.69 (92) |
| temperature (T=1 vs T=3) | 0.54 → 0.64 (221–830) | 0.50 → 0.60 (252–861) | 0.49 → 0.62 (282–891) | 0.53 → 0.62 (313–922) | 0.55 → 0.62 (339–948) | 0.54 → 0.73 (374) |
| 2nd language (ru vs zh vs es) | 0.45 → 0.53 (23–121) | 0.50 → 0.51 (24–122) | 0.43 → 0.49 (28–126) | 0.52 → 0.53 (31–129) | 0.49 → 0.48 (31–129) | 0.61 → 0.75 (34) |
| English corpus (edu filter on vs off) | 0.54 → 0.55 (23) | 0.56 → 0.61 (24) | 0.50 → 0.78 (6) | 0.47 → 0.62 (31) | 0.55 → 0.66 (31) | 0.67 → 0.78 (34) |

Key findings:

- At 90M the DA-goal line clears 0.75 and stays there — language count: never (max 0.56); depth (deep vs shallow): never (max 0.56); language list (A vs B): never (max 0.51); temperature (T=1 vs T=3): never (max 0.64); 2nd language (ru vs zh vs es): never (max 0.62); English corpus (edu filter on vs off): never (max 0.65).
- The distance between the DA-goal and DA-ckpt cell of an axis at a proxy size is what the size costs; the 1.7B column is the same line in both.

Follow-ups:

- The size axis of the same split is `scale_convergence_da_size_transformation_panels_multi_axes.png` (DA-size pooled over decisions).
- A jackknife band per axis line (leave one family out), as the scale-convergence panels carry.

Filtered twins (reliable cells only): [`early_small_da_goal_by_transformation_above_80_mono_axis.png`](pretraining/predictivity/early_small_da_goal_by_transformation_above_80_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_both_mono_axis.png`](pretraining/predictivity/early_small_da_goal_by_transformation_above_66_both_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_either_mono_axis.png`](pretraining/predictivity/early_small_da_goal_by_transformation_above_66_either_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_80_mono_axis.png`](pretraining/predictivity/early_small_da_ckpt_by_transformation_above_80_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png`](pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png`](pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png)

Files: [`early_small_da_goal_by_transformation_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_mono_axis.png), [`early_small_da_goal_by_transformation_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_mono_axis.csv), [`early_small_da_goal_by_transformation_above_80_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_80_mono_axis.png), [`early_small_da_goal_by_transformation_above_80_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_80_mono_axis.csv), [`early_small_da_goal_by_transformation_above_66_both_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_both_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_both_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_both_mono_axis.csv), [`early_small_da_goal_by_transformation_above_66_either_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_either_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_either_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_either_mono_axis.csv), [`early_small_da_ckpt_by_transformation_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_mono_axis.png), [`early_small_da_ckpt_by_transformation_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_mono_axis.csv), [`early_small_da_ckpt_by_transformation_above_80_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_80_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_80_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_80_mono_axis.csv), [`early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_both_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_both_mono_axis.csv), [`early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.csv), [`da_all_by_transformation_per_task_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_by_transformation_per_task_mono_axis.csv).
<!-- END auto:early-small-by-transformation -->

**Key findings**

- The mean-over-tasks reading on the A/B pool (2026-09-30 tables, before
  `predictivity` took every build) is 0.46–0.53 at every proxy
  size and checkpoint for the benchmarks (`early_small_da_goal_multi_axes.csv`),
  the pooled-over-decisions reading of figure 1
  0.53–0.56: two estimands, one verdict.
- BPB reads the 1.7B ranking at ≥ 0.75 from 175M at 2C (0.77) and from 350M
  at 2.5C; `bpb_macro` and `train_loss` from any size and checkpoint; no
  (proxy, checkpoint) cell reaches 0.75 for the benchmark mean.
- DA-ckpt rises along every run (175M: 0.54 at 10 % → 0.76 at 90 %; 1.7B:
  0.53 → 0.74) and the reference's own curve is its DA-ckpt; read against the
  seed null of figure 7 (0.51 → 0.75 at 175M), the design signal at 90 % is
  −0.01 to +0.02.
- Per L, only the regimes with ≥ 3 usable pairs draw a panel (`pairs_da_all_by_L_multi_axes.csv`:
  L1 and L2 had one pair against 1.7B on the A/B pool; the DCLMP/FWEB and
  ZH/ES cells give them their three families on `predictivity`, figure 4).
- Smallest safe level per (benchmark, language): never 228, 1B 38, 350M 7,
  175M 5, 600M 4 of 282 cells; "safe" with 3 pairs is a weak guarantee.

**Follow-ups**

- The `_flops` reading (`safe_flops_da_goal_multi_axes.png`) joins un-annealed and annealed
  points; one line per size with the annealed points marked, plus the
  compute frontier (the best DA reachable at or below each compute), is the
  practical answer.
- `early_small_by_L_own` (the size's cost separated from the checkpoint's) as
  an appendix pair with `early_small_da_goal_by_L_multi_axes`, once the 6-pair L's are
  complete; `pairs_da_all_by_L_multi_axes.csv` justifies why per-L DA is coarse.

GitHub: [early_small_da_goal_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_multi_axes.png) · [early_small_da_goal_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_multi_axes.csv) ·
[early_small_da_goal_by_benchmark_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_benchmark_multi_axes.png) ·
[early_small_da_goal_by_language_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_language_multi_axes.png) ·
GitHub: [highlights_da_all_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/highlights_da_all_multi_axes.png) · [highlights_da_all_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/highlights_da_all_multi_axes.csv) ·
GitHub: [safe_size_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_size_da_size_multi_axes.png) · [safe_size_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_size_da_size_multi_axes.csv) ·
GitHub: [safe_checkpoint_da_ckpt_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_checkpoint_da_ckpt_multi_axes.png) · [safe_checkpoint_da_ckpt_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_checkpoint_da_ckpt_multi_axes.csv) ·
GitHub: [safe_flops_da_goal_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_flops_da_goal_multi_axes.png) · [safe_flops_da_goal_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_flops_da_goal_multi_axes.csv) ·
GitHub: [da_size_by_family_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_by_family_multi_axes.png) · [da_size_by_family_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_by_family_multi_axes.csv) ·
[da_size_by_benchmark_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_by_benchmark_multi_axes.png) ·
[da_size_by_language_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_by_language_multi_axes.png) ·
[da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_multi_axes.csv) ·
GitHub: [early_small_da_goal_by_L_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_L_multi_axes.png) · [early_small_da_goal_by_L_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_L_multi_axes.csv) ·
GitHub: [early_small_da_goal_by_L_with_bpb_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_L_with_bpb_multi_axes.png) · [early_small_da_goal_by_L_with_bpb_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_L_with_bpb_multi_axes.csv) ·
GitHub: [early_small_da_ckpt_by_L_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_L_multi_axes.png) · [early_small_da_ckpt_by_L_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_L_multi_axes.csv) ·
GitHub: [early_small_da_ckpt_by_L_with_bpb_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_L_with_bpb_multi_axes.png) · [early_small_da_ckpt_by_L_with_bpb_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_L_with_bpb_multi_axes.csv) ·
GitHub: [early_small_da_goal_by_L_above_80_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_L_above_80_multi_axes.png) · [early_small_da_goal_by_L_above_80_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_L_above_80_multi_axes.csv) ·
GitHub: [early_small_da_ckpt_by_L_above_80_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_L_above_80_multi_axes.png) · [early_small_da_ckpt_by_L_above_80_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_L_above_80_multi_axes.csv) ·
GitHub: [early_small_da_ckpt_by_L_above_66_ckpt_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_L_above_66_ckpt_mono_axis.png) · [early_small_da_ckpt_by_L_above_66_ckpt_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_L_above_66_ckpt_mono_axis.csv) ·
GitHub: [early_small_da_goal_by_L_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_L_above_66_either_mono_axis.png) · [early_small_da_goal_by_L_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_L_above_66_either_mono_axis.csv) ·
[pairs_da_all_by_L_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/pairs_da_all_by_L_multi_axes.csv) ·
[da_all_by_L_per_task_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_by_L_per_task_multi_axes.csv) ·
[da_all_pooled_per_task_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_pooled_per_task_multi_axes.csv)

### 4. Language count: unordered, on any task set

**DA-size · `scale_convergence_da_size_L8*`: no filter (the `_above_66_size` twins
conditional) · multi-axis (`scale_convergence_da_size_L_mono_axis` mono-axis, every
language) · pairs sharing the L from `predictivity` at seed 1904 ·
gate `predictivity`.** One line per language-count regime; a regime pools
depth, scheme and temperature decisions (`share_*` in the CSV) and the mix
differs by regime, which rule 5 forbids holding fixed.

<!-- BEGIN auto:scale-convergence-L8 (scale_convergence.py --by L --langs L8) -->
## Scale convergence by language count, on the L8 languages

The `--by L` lines read over the tasks in the 8 languages of the L8 setting (de, en, es, fr, it, ja, ru, zh), which every regime from L8 up trains. What this cannot fix: a regime pools arch, list and temperature decisions at once and the mix differs by regime (`share_*` in the CSV); rule 5 forbids holding it fixed. Under `--axes mono-axis` the regimes keep only their one-axis pairs (L8 6 → 4, L30 10 → 5) and rest on fewer tasks. Numbers below are the unfiltered population; the `above_66_size` twin (`scale_convergence_da_size_L8_above_66_size_multi_axes.png`) is the conditional one. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by L --langs L8`.

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| L1 | 0.55 | 0.59 | 0.69 | 0.51 | 0.56 | 1.0 | — |
| L15 | 0.52 | 0.49 | 0.51 | 0.51 | 0.54 | 1.0 | — |
| L2 | 0.53 | 0.51 | 0.51 | 0.53 | 0.46 | 1.0 | — |
| L30 | 0.54 | 0.52 | 0.51 | 0.52 | 0.52 | 1.0 | — |
| L50 | 0.6 | 0.57 | 0.58 | 0.55 | 0.56 | 1.0 | — |
| L8 | 0.51 | 0.51 | 0.5 | 0.52 | 0.51 | 1.0 | — |
| all pairs | 0.59 | 0.57 | 0.58 | 0.57 | 0.57 | 1.0 | — |

![Scale convergence, L8](pretraining/predictivity/scale_convergence_da_size_L8_multi_axes.png)
<!-- END auto:scale-convergence-L8 -->

<!-- BEGIN auto:scale-convergence-L8-common (scale_convergence.py --by L --langs L8 --common-tasks) -->
## Scale convergence by language count, on the L8 languages, common tasks

The `--by L` lines read over the tasks in the 8 languages of the L8 setting (de, en, es, fr, it, ja, ru, zh), which every regime from L8 up trains, and further over the tasks with ≥ 3 pairs in every regime at every proxy size — one task set for the whole figure, so a gap between lines is a gap on the same benchmarks. What this cannot fix: a regime pools arch, list and temperature decisions at once and the mix differs by regime (`share_*` in the CSV); rule 5 forbids holding it fixed. Under `--axes mono-axis` the regimes keep only their one-axis pairs (L8 6 → 4, L30 10 → 5) and rest on fewer tasks. Numbers below are the unfiltered population; the `above_66_size` twin (`scale_convergence_da_size_L8common_above_66_size_multi_axes.png`) is the conditional one. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by L --langs L8 --common-tasks`.

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| L1 | 0.62 | 0.62 | 0.72 | 0.59 | 0.69 | 1.0 | — |
| L15 | 0.48 | 0.47 | 0.4 | 0.47 | 0.54 | 1.0 | — |
| L2 | 0.53 | 0.53 | 0.55 | 0.58 | 0.6 | 1.0 | — |
| L30 | 0.51 | 0.5 | 0.46 | 0.48 | 0.51 | 1.0 | — |
| L50 | 0.53 | 0.39 | 0.54 | 0.46 | 0.51 | 1.0 | — |
| L8 | 0.53 | 0.56 | 0.42 | 0.46 | 0.51 | 1.0 | — |
| all pairs | 0.56 | 0.54 | 0.55 | 0.55 | 0.58 | 1.0 | — |

![Scale convergence, L8 common tasks](pretraining/predictivity/scale_convergence_da_size_L8common_multi_axes.png)
<!-- END auto:scale-convergence-L8-common -->

| regime, L8-language tasks (unfiltered) | 175M | 350M | 600M | 1B | tasks | decision mix at 1B |
|---|---|---|---|---|---|---|
| L2 | 0.389 | 0.583 | 0.527 | 0.484 | 6–31 | depth ⅓, second language ⅓, two-axis ⅓ |
| L8 | 0.472 | 0.492 | 0.464 | 0.568 | 41–71 | depth ⅓, list ⅓, two-axis ⅓ |
| L15 | 0.538 | 0.508 | 0.497 | 0.517 | 79–141 | depth 0.23, list 0.15, T 0.15, two-axis 0.46 |
| L30 | 0.534 | 0.469 | 0.517 | 0.447 | 79–141 | depth 0.20, list 0.20, T 0.10, two-axis 0.50 |
| L50 | 0.542 | 0.526 | 0.502 | 0.564 | 79–141 | depth ⅓, T ⅓, two-axis ⅓ |
| all pairs | 0.539 | 0.543 | 0.543 | 0.552 | 79–141 | L 0.15, two-axis 0.74 |

**Key findings**

- The per-L lines do not order by L and do not move when the task set is
  fixed: on the L8 languages every regime line lies between 0.39 and 0.58
  with no monotone relation to L, the pooled line at 0.54–0.55 at every
  proxy; on the 6 tasks common to every regime and size the lines read L8
  0.47 → 0.44, L15 0.62 → 0.60, L30 0.67 → 0.42, L50 0.39 → 0.53 (a lattice,
  not a trend). Filtered on `above_66_size` they read 0.44–0.80 on 11–28
  tasks and are again unordered.
- The mono-axis pairs keep every L line (L8 6 → 4 pairs, L30 10 → 5); the L2
  regime has a line because the ZH and ES second-language cells give it three
  families.
- A regime line is not the effect of language count: within a regime the
  decisions are depth, scheme and temperature in a mix that differs by regime;
  what the figure shows is that the decisions within any regime are read
  equally poorly from a smaller model, on 4–5 families per regime.

**Follow-ups**

- An L8-only replicate of the depth and list decisions in every regime would
  hold the decision mix fixed — the one thing this grid cannot do above three
  pairs.
- Bootstrap over L, not items, for the per-L comparison (items within an L
  move together).

GitHub: [scale_convergence_da_size_L8_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8_multi_axes.png) · [scale_convergence_da_size_L8_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_L8common_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8common_multi_axes.png) · [scale_convergence_da_size_L8common_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8common_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_L8_above_66_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8_above_66_size_multi_axes.png) · [scale_convergence_da_size_L8_above_66_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8_above_66_size_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_L8common_above_66_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8common_above_66_size_multi_axes.png) · [scale_convergence_da_size_L8common_above_66_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8common_above_66_size_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_L_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L_mono_axis.png) · [scale_convergence_da_size_L_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L_mono_axis.csv)

### 5. One language at a time, and the tokens axis

**DA-size · `scale_convergence_da_size_lang_all*`: no filter (`_above_66_size`
conditional) · multi-axis · pairs sharing the L from `predictivity` ·
gate `predictivity`.** One panel per L8 language, one line per regime that
trains it plus the pooled line; the `_tokens` twin relabels x as the tokens of
the language the decision's two members trained on (share × D(N)), and the
panel title carries the collapse test.

<!-- BEGIN auto:scale-convergence-by-language (by_language.py --pool predictivity) -->
## Scale convergence per language

For each language of the L8 setting, the `--by L` lines read on that language's benchmarks alone: R at the smallest → largest proxy [tasks], per regime that trains the language, on every gated task (no selection on DA — the inference version; the `above_66_size` twin is the conditional one). The last column is the collapse test: R² of one log-linear line through every regime's points with x = model size, then with x = tokens of the language (its share of all tokens × D(N)); a rise under tokens says exposure explains what language count does not. Regimes pool arch, list and temperature decisions at once. Regenerate with `python analysis/rq02_decision_accuracy/by_language.py --pool predictivity`; `scale_convergence_da_size_lang_all_multi_axes_coverage.csv` says why a cell is empty.

| language | L1 | L2 | L8 | L15 | L30 | L50 | R² size / tokens |
|---|---|---|---|---|---|---|---|
| en | 0.62→0.60 [31] | 0.46→0.51 [31] | 0.55→0.56 [31] | 0.48→0.51 [31] | 0.51→0.44 [31] | 0.49→0.51 [31] | 0.00 / 0.03 |
| ru | — | — | 0.52→0.46 [12] | 0.43→0.53 [12] | 0.59→0.57 [12] | 0.58→0.74 [12] | 0.03 / 0.02 |
| zh | — | — | 0.63→0.55 [13] | 0.51→0.51 [13] | 0.54→0.52 [13] | 0.56→0.60 [13] | 0.01 / 0.00 |
| de | — | — | 0.64→0.56 [11] | 0.49→0.62 [11] | 0.50→0.61 [11] | 0.57→0.61 [11] | 0.04 / 0.01 |
| ja | — | — | 0.56→0.63 [9] | 0.47→0.43 [9] | 0.40→0.41 [9] | 0.33→0.65 [9] | 0.03 / 0.05 |
| es | — | — | — | 0.42→0.50 [49] | 0.50→0.39 [49] | 0.41→0.54 [49] | 0.08 / 0.08 |
| fr | — | — | — | 0.57→0.49 [13] | 0.44→0.50 [13] | 0.52→0.59 [13] | 0.01 / 0.03 |
| it | — | — | — | 0.62→0.72 [13] | 0.40→0.50 [13] | 0.60→0.62 [13] | 0.02 / 0.02 |

![Scale convergence per language](pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes.png)

![Scale convergence per language, tokens axis](pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_tokens.png)
<!-- END auto:scale-convergence-by-language -->

**Key findings**

- Every one of the eight languages draws a panel and none is ordered by L:
  per language the pooled line at 1B reads en 0.51 (31 tasks), ru 0.69 (11),
  zh 0.60 (12), de 0.66 (9), ja 0.54 (8), es 0.50 (47), fr 0.58 (12), it 0.59
  (11); the regime lines within a panel cross at every size. Spanish, French
  and Italian have no L8 line (scheme B's L8 list does not contain them, so
  their L8 cells have one pair).
- Exposure does not collapse the lines: the collapse R² is 0.00–0.07 under
  size and 0.00–0.09 under tokens, rising under tokens for two languages only
  (ru 0.00 → 0.08, es 0.04 → 0.09). If reliability were a function of how
  many of a language's tokens the models saw, the tokens axis would order the
  points; it does not — the same result as figure 1, per language.

**Follow-ups**

- The zh and ja panels rest on one to two families below 1B because the
  reformulated twins were evaluated on a subset of cells; an evaluation
  top-up, not analysis code, fills them.

GitHub: [scale_convergence_da_size_lang_all_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes.png) · [scale_convergence_da_size_lang_all_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_lang_all_multi_axes_tokens.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_tokens.png) · [scale_convergence_da_size_lang_all_multi_axes_tokens.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_tokens.csv) ·
[scale_convergence_da_size_lang_all_multi_axes_coverage.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_coverage.csv) ·
GitHub: [scale_convergence_da_size_lang_above_66_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes.png) · [scale_convergence_da_size_lang_above_66_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.png) · [scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.csv)

### 6. High-resource languages are not easier to read

**DA-size · `reliability_da_size_by_language_tier_multi_axes`: panel (a) no filter, panel (b)
`above_66_size` · multi-axis · pairs from `predictivity` · gate
`predictivity`; `reliability_da_size_vs_language_share_multi_axes`: no filter, per language with
≥ 3 gated tasks (rule 8).** A tier is the smallest scheme-A regime that trains
the language (L8 = the eight high-resource languages, L50 = the twenty only
the L50 mixture trains).

<!-- BEGIN auto:language-tier (language_tier.py --pool predictivity) -->
## Decision reliability by language tier

The pooled `all pairs` line of `scale_convergence.py` read over the gated tasks of one language TIER — the smallest scheme-A regime that trains the language (L8: the eight high-resource languages every regime trains; L50: the twenty only the L50 mixture trains). Reliability at the smallest → largest proxy [task count]. `reliability_da_size_vs_language_share_multi_axes.png` is the per-language version: reliability against the language's share of the L50 mixture, Spearman ρ over languages 175M 0.24, 1B 0.02, 350M 0.18, 600M 0.27, 90M 0.24. Both are unfiltered; a tier also differs in benchmark mix. Regenerate with `python analysis/rq02_decision_accuracy/language_tier.py --pool predictivity`.

| tier | all gated tasks | above_66_size |
|---|---|---|
| L8 languages (in every regime) | 0.56 → 0.57 [100–151 tasks] | 0.70 → 0.76 [23–30 tasks] |
| L15-only languages | 0.51 → 0.56 [53–80 tasks] | 0.65 → 0.80 [6–9 tasks] |
| L30-only languages | 0.49 → 0.56 [68–108 tasks] | 0.61 → 0.77 [13–16 tasks] |
| L50-only languages | 0.50 → 0.56 [81–114 tasks] | 0.65 → 0.76 [26–36 tasks] |

![Reliability by language tier](pretraining/predictivity/reliability_da_size_by_language_tier_multi_axes.png)

![Reliability against language share](pretraining/predictivity/reliability_da_size_vs_language_share_multi_axes.png)
<!-- END auto:language-tier -->

**Key findings**

- The L8 tier leads at 175M by 0.01–0.04 and the tiers coincide by 1B:
  unfiltered L8 0.539 → 0.552, L15 0.525 → 0.569, L30 0.505 → 0.563, L50
  0.495 → 0.570 from 175M to 1B on 79–141 / 39–67 / 51–86 / 58–93 tasks; at
  1B the four jackknife intervals overlap (L8 [0.51, 0.59], L50 [0.48, 0.66]).
  Filtered on `above_66_size` every tier reads 0.60–0.66 at 175M and
  0.76–0.84 at 1B with no order.
- Across 45 languages the share of the mixture explains nothing of the
  per-language reliability: Spearman 0.20 (175M, 36 languages), 0.10 (350M,
  42), 0.02 (600M, 44), −0.07 (1B, 45).
- One reading: the low-resource tier reads its decisions from the L50 cells
  alone (6 pairs per task, mostly the temperature and depth decisions at L50),
  which are the decisions figure 2 reads best; the tiers differ in decision
  mix as well as in resource level, and the two pull in opposite directions.

**Follow-ups**

- The same decision set in every tier (rule 5 forbids it on this grid; the
  L8-only replicate of figure 4's follow-up would allow it).
- A tier also differs in benchmark mix (the L50 tier is mostly Belebele and
  Global-MMLU twins); a per-family version of panel (a) separates the two.

GitHub: [reliability_da_size_by_language_tier_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_da_size_by_language_tier_multi_axes.png) · [reliability_da_size_by_language_tier_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_da_size_by_language_tier_multi_axes.csv) ·
GitHub: [reliability_da_size_vs_language_share_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_da_size_vs_language_share_multi_axes.png) · [reliability_da_size_vs_language_share_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_da_size_vs_language_share_multi_axes.csv)

### 7. Seed uncertainty and the DA-ckpt null

**`seed_uncertainty_da_all_seed_null`: proxy-seed DA = DA-size at each of the proxy's three
seeds against the fixed 1.7B (seed 1904) final; test-retest = the same
designs ranked from seed s1 against seed s2 at one size; seed null = DA-ckpt
over the `seed` pairs (two seeds of ONE design) against the real design pairs
· no filter · pool `predictivity_seeds` · gate `predictivity`.** English at
175M, 600M and 1B and Russian at 1B are the only (size, language) cells where
three replicated designs give ≥ 3 cross-L pairs under rule 2.

<!-- BEGIN auto:seed-uncertainty (seed_uncertainty.py --pool predictivity) -->
## Seed uncertainty

What the replicate seeds say about decision accuracy — English at three proxy sizes and Russian at 1B, the only (size, language) cells where three replicated designs give ≥ 3 cross-L pairs under rule 2. Per row: DA of those decisions against the 1.7B final with the proxy at each of its three seeds (proxy-side noise), and the DA between two seeds' rankings of the same designs at that size (the reference-side ceiling). Three pairs put a DA on {0, ⅓, ⅔, 1}: read the spread, not a mean. The seed null (DA-ckpt over pairs of two seeds of one design, mean over fractions and gated tasks) is 175M: 0.58 vs real 0.59; 600M: 0.56 vs real 0.58; 1B: 0.56 vs real 0.58. Regenerate with `python analysis/rq02_decision_accuracy/seed_uncertainty.py --pool predictivity`.

| size | language | designs | DA at the 3 proxy seeds | test-retest (3 seed pairs) | tasks |
|---|---|---|---|---|---|
| 175M | en | 3 | 0.60, 0.74, 0.68 | 0.64–0.71 | 24 |
| 600M | en | 3 | 0.67, 0.68, 0.59 | 0.55–0.71 | 31 |
| 1B | en | 4 | 0.63, 0.53, 0.58 | 0.56–0.62 | 31 |
| 1B | ru | 4 | 0.83, 0.83, 0.81 | 0.81–0.89 | 12 |

![Seed uncertainty](pretraining/predictivity/seed_uncertainty_da_all_seed_null.png)
<!-- END auto:seed-uncertainty -->

**Key findings**

- Proxy-seed noise is up to ± 0.07 on English: the three seeds read 0.60 /
  0.74 / 0.68 at 175M (72 decisions on 24 tasks), 0.67 / 0.68 / 0.59 at 600M
  (93 decisions, 31 tasks), 0.63 / 0.53 / 0.58 at 1B (186 decisions, 31
  tasks); Russian at 1B 0.83 / 0.83 / 0.81 on 12 tasks. Three pairs put a DA
  on {0, ⅓, ⅔, 1}: read the spread, not a mean.
- The English test-retest ceiling is low at every size: 0.64–0.71 at 175M,
  0.55–0.71 at 600M and 0.56–0.62 at 1B (Russian 0.81–0.89). On the 24–31
  English tasks readable there (`include_v2_en`, `bbh`, `acp_bench` among
  them) two seeds of the same designs disagree with each other as much as the
  proxy disagrees with the reference: the design differences are below the
  seed noise of those benchmarks.
- DA-ckpt is mostly within-run persistence: two seeds of one design, which
  have nothing to decide, read 0.51 at 10 % of the 175M run and 0.75 at 90 %,
  against 0.51 and 0.75 for the real pairs, and 0.48 → 0.72 against 0.48 →
  0.74 at 1B; at 90 % the real pairs exceed the null by −0.01 (175M), +0.01
  (600M) and +0.02 (1B), the largest gap mid-run (+0.05: 0.51 against 0.55
  at 40 % of the 600M run, 0.54 against 0.59 at 50 % of the 175M one). The
  "reading early is cheap" panel of figure 2 measures how much a run's
  ranking at 90 % resembles its ranking at 100 %.
- These replicated cells are three or four cross-L deep designs on one or two
  languages: they give the scale of seed noise, not an interval for the pooled
  lines. No Wilson or binomial interval is drawn anywhere: the pairs share
  models and are not independent.

**Follow-ups**

- The null exists for DA-ckpt only (no seed replicate at 1.7B); a second 1.7B
  seed on two cells would give the DA-size and DA-goal panels the same floor.
- Draw the null on every checkpoint-axis figure the paper shows.

GitHub: [seed_uncertainty_da_all_seed_null.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/seed_uncertainty_da_all_seed_null.png) · [seed_uncertainty_da_all_seed_null.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/seed_uncertainty_da_all_seed_null.csv)

### 8. Decision accuracy is Kendall's τ under another tie convention

**DA-size's population · no filter · multi-axis pairs · pairs from
`predictivity` · gate `predictivity`; one point per (task, proxy
size) cell, median 12 models per cell.**

<!-- BEGIN auto:agreement-measures (agreement.py --pool predictivity) -->
## Decision accuracy is Kendall's τ under another tie convention

Over the 5,964 (benchmark task, proxy size) cells of DA-size's population (every pair of the grid-seed variants, ≥ 3 pairs, above chance at the proxy and at 1.7B): 2·DA − 1 = τ_a + (T_both − T_one)/n exactly, where T_both / T_one are the pairs tied on both / one side. Ties are 10,977 of 518,627 pairs (2.1%), 95% of them one-sided, and touch 24% of the cells — so the two statistics correlate at r = 0.989 by construction, and the number with content is how often the tie convention changes a reliability verdict (DA ≥ 0.66, i.e. τ ≥ 0.32), in % of cells per proxy size:

| statistic | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| da | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| da_drop_ref_ties | 0.6 | 0.7 | 1.1 | 1.0 | 1.4 |
| gamma | 2.1 | 1.4 | 1.6 | 2.4 | 2.2 |
| tau_a | 1.5 | 1.2 | 1.1 | 1.8 | 1.7 |
| tau_b | 1.7 | 1.2 | 1.5 | 2.2 | 1.9 |

Spearman ρ and Pearson r on the raw scores are the two statistics that are NOT a rescaling — they weight a pair by its displacement — and sit at r = 0.973 and 0.897 against DA. Median 12 models per cell. Values in `agreement_da_size_per_cell_multi_axes.csv`; regenerate with `python analysis/rq02_decision_accuracy/agreement.py --pool predictivity`.

![DA, Kendall tau and Spearman rho against each other](pretraining/predictivity/agreement_da_size_correlation_multi_axes.png)

![DA against Kendall's tau](pretraining/predictivity/agreement_da_size_identity_multi_axes.png)

![Cut sensitivity](pretraining/predictivity/agreement_da_size_cut_sensitivity_multi_axes.png)
<!-- END auto:agreement-measures -->

**Key findings**

- The three statistics are one statistic: over the 1 286 cells r(DA, τ_b) =
  0.971, r(DA, ρ) = 0.961, r(τ_b, ρ) = 0.986 (Spearman 0.966, 0.961, 0.990).
  The relation to τ_a is exact — 2·DA − 1 = τ_a + (T_both − T_one)/n to
  2.5e−16 on every cell — so DA and τ_a differ only in how tied pairs are
  counted. Ties are 7.2 % of the pairs, touch 73 % of the cells, and 94 %
  are one-sided.
- The convention moves the verdict on 2.5–6 % of cells: against the DA ≥ 0.66
  set, τ_a flips 2.8–4.8 %, τ_b 4.0–5.7 %, γ 4.7–6.2 % and DA with the
  reference's ties dropped 2.5–3.5 %, per proxy size. The high correlations
  are guaranteed by the identity; the flip rate is the number with content,
  and it says a published reliable-task list depends on the tie convention at
  the margin.

**Follow-ups**

- Report the reliable-task list under two conventions (DA and γ) in the
  paper's appendix, or state the flip rate beside it.

GitHub: [agreement_da_size_correlation_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_correlation_multi_axes.png) · [agreement_da_size_correlation_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_correlation_multi_axes.csv) ·
GitHub: [agreement_da_size_identity_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_identity_multi_axes.png) · [agreement_da_size_identity_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_identity_multi_axes.csv) ·
GitHub: [agreement_da_size_cut_sensitivity_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_cut_sensitivity_multi_axes.png) · [agreement_da_size_cut_sensitivity_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_cut_sensitivity_multi_axes.csv) ·
[agreement_da_size_per_cell_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_per_cell_multi_axes.csv)

### 9. Multi-axis against mono-axis pairs

**`rq2_da_all_above_66_both_mono_vs_multi_axes`: the three definitions · filter `above_66_both`
(71 cells; the one place it is drawn, for the pair-set comparison only) ·
rows multi-axis and mono-axis · pairs from `predictivity` · gate
`predictivity`.** Exploratory (`pair_axes.py`); it fed `plan/decision_accuracy.md`
and the `axes` column of rule 15.

![Multi-axis against mono-axis pairs](pretraining/predictivity/rq2_da_all_above_66_both_mono_vs_multi_axes.png)

`pair_axes.py` computes the three decision accuracies on the `above_66_both` cells
twice — over every pair at the grid seed (multi-axis, rq02's convention) and over
the pairs that move exactly one design axis — L, depth, list, temperature, second
language on those tables; L, depth, activation, data scheme, temperature since
2026-10-05, the same pairs (mono-axis, which is what DataDecide's "all pairs" are by construction) — and
writes `rq2_da_all_above_66_both_mono_vs_multi_axes.png/.csv` (71 cells on the 2026-09-23 tables).
DA-ckpt is indifferent to the pair set (the 1.7B run 0.68 → 0.83 mono-axis against
0.73 → 0.84 multi-axis from 10 % to 90 %); DA-size reads 0.586, 0.679, 0.698, 0.721
mono-axis against 0.637, 0.726, 0.742, 0.772 multi-axis at 175M … 1B — 0.04–0.05
lower with the same trend — and DA-goal likewise, on 28 % of the decisions (999 of
3,552 at 175M). The proposal that follows from it (an `axes` column in
`da_all_per_task_both_axes.csv`, mono-axis as the headline for decisions) is in
`plan/decision_accuracy.md` (§2, §6).

**Key findings**

- DA-ckpt moves little with the pair set (the 1.7B run 0.68 → 0.83 mono-axis
  against 0.73 → 0.84 multi-axis from 10 % to 90 %); DA-size reads 0.04–0.05
  higher under the multi-axis pairs on the same cells (0.64 → 0.77 against
  0.59 → 0.72 from 175M to 1B), and DA-goal likewise; the mono-axis pairs are
  the stricter and smaller set (28 % of the decisions, 999 of 3 552 at 175M).

**Follow-ups**

- Mono-axis as the headline for decisions (the paper's left panel already
  draws it); the multi-axis pooled line as the population statement.

GitHub: [rq2_da_all_above_66_both_mono_vs_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_mono_vs_multi_axes.png) · [rq2_da_all_above_66_both_mono_vs_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_mono_vs_multi_axes.csv)

### 10. Cross-task predictability

**DA-size and DA-ckpt · no filter · multi-axis pairs of `predictivity` (the
A/B pool on the tables below: 18 variants at 1.7B, 153 pairs; DA-ckpt pools the
within-size pairs of every size, 765; every build since 2026-10-05) · gate
`predictivity` on both sides.** Every parent
task as the proxy for every other one; a cell is the smallest size or earliest
checkpoint at which x's ranking safely predicts y's final ranking (DA ≥ 0.75
over ≥ 3 pairs, held at every larger level).

<!-- BEGIN auto:cross-task (cross_task.py --pool predictivity) -->
## Cross-task predictability

Every parent task as the proxy for every other one (1714 x 1714): the cell is the smallest proxy size (DA-size, 18 variants at 1.7B, 153 pairs) or the earliest checkpoint (DA-ckpt, the within-size pairs of every size pooled, 918 pairs, the nine checkpoints before the final) at which the ranking on task x (columns) safely predicts the final ranking on task y (rows): DA >= 0.75 over >= 3 pairs there and at every larger level with a value. The diagonal is rq02's own-task DA; the gate empties a benchmark's pairs at every size where it is at chance. The `_by_family` maps take the median level over the task pairs of two benchmarks, the `_by_language` maps over the same-benchmark task pairs of two languages (resource order of the scheme-A lists). Regenerate with `python analysis/rq02_decision_accuracy/cross_task.py --pool predictivity`.

The same two maps over the benchmark tasks that are above chance at some size — BPB, the loss and the benchmarks the gate finds at chance everywhere are dropped, so what is left is the sub-map where a transfer result is possible at all: [`cross_task_da_size_benchmarks_multi_axes.png`](pretraining/predictivity/cross_task_da_size_benchmarks_multi_axes.png), [`cross_task_da_ckpt_benchmarks_multi_axes.png`](pretraining/predictivity/cross_task_da_ckpt_benchmarks_multi_axes.png).

![Cross-task DA-size by benchmark](pretraining/predictivity/cross_task_da_size_by_family_multi_axes.png)

![Cross-task DA-ckpt by benchmark](pretraining/predictivity/cross_task_da_ckpt_by_family_multi_axes.png)

![Cross-task DA-size by language](pretraining/predictivity/cross_task_da_size_by_language_multi_axes.png)

![Cross-task DA-ckpt by language](pretraining/predictivity/cross_task_da_ckpt_by_language_multi_axes.png)

Full task-level maps: [`cross_task_da_size_multi_axes.png`](pretraining/predictivity/cross_task_da_size_multi_axes.png), [`cross_task_da_ckpt_multi_axes.png`](pretraining/predictivity/cross_task_da_ckpt_multi_axes.png).
<!-- END auto:cross-task -->

**Key findings**

- Cross-task predictability is low, and where it works it is late and
  large: the safe-level maps are mostly "never" off the diagonal, and the
  family maps put the median benchmark → benchmark cell at 1B / 4C; the only
  sub-1B blocks are BPB → BPB and hellaswag/xnli → hellaswag/xnli.
- Between two BPB languages the ranking is unrelated by construction — the
  design variants are language mixes, so the ranking on one language is not
  the ranking on another — and high-resource languages do not predict
  low-resource ones; what transfers is membership in the same L list (the
  blocks between the lines), because the same design variants train both.
- Most cells are gated (the letter-format originals are at chance at 1.7B for
  every language), so the full maps mostly say which tasks have a ranking at
  all; `cross_task_da_{size,ckpt}_benchmarks_multi_axes.png` is the sub-map where a
  transfer result is possible.

**Follow-ups**

- The family maps show the median of the pairs that succeed while most pairs
  never do; draw the share of task pairs reaching a safe size (in the CSV) as
  the cell and keep the median level as the small number.
- Same-L blocks share the design variants, so "same L list transfers" is
  partly by construction: compare against a permutation baseline (the
  cross-task DA of two languages whose L lists are shuffled).
- Recompute on every build, as the other figures are, so the diagonal is
  figure 1's population — what the next refresh does, since `predictivity`
  holds every build from 2026-10-05.

GitHub: [cross_task_da_size_by_family_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_family_multi_axes.png) · [cross_task_da_size_by_family_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_family_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_by_family_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_family_multi_axes.png) · [cross_task_da_ckpt_by_family_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_family_multi_axes.csv) ·
GitHub: [cross_task_da_size_by_language_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_language_multi_axes.png) · [cross_task_da_size_by_language_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_language_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_by_language_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_language_multi_axes.png) · [cross_task_da_ckpt_by_language_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_language_multi_axes.csv) ·
GitHub: [cross_task_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_multi_axes.png) · [cross_task_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_multi_axes.png) · [cross_task_da_ckpt_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_multi_axes.csv) ·
GitHub: [cross_task_da_size_benchmarks_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_benchmarks_multi_axes.png) · [cross_task_da_size_benchmarks_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_benchmarks_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_benchmarks_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_benchmarks_multi_axes.png) · [cross_task_da_ckpt_benchmarks_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_benchmarks_multi_axes.csv)

### 11. Benchmark BPB: a continuous score on the same items

**DA-size · no filter · multi-axis pairs of `predictivity` (25
variants at 1.7B, 300 pairs) · final checkpoints only · gate `predictivity`
on the accuracy side only.** Figures 1–10 read every benchmark through its
accuracy, a step function of the log-likelihoods the harness already
computed. This figure reads the same items, the same checkpoints and the same
pairs through the gold answer's bits-per-byte (**bBPB**). Does a continuous
score rank like the reference where accuracy is a coin flip?

How bBPB is computed (no model is re-run; `build_per_item_store.py` +
`bench_bpb_da.py`):

- **Source.** The lm-eval samples files of every final checkpoint,
  `eval_logs/.../<cell>-iter<N>/harness/eval_*/samples_<task>_*.jsonl`. The
  newest file per task wins across the top-up eval dirs. A parent with no file
  of its own is the union of its subject files (rule 6). Only the iopsstor
  copies work: the HF-pushed samples drop `arguments`.
- **Per item.** `ll_gold` is the summed natural-log log-likelihood of the gold
  choice, `resps[target]`. `target` is a digit or a letter (A = 0).
  `bytes_gold` is the UTF-8 length of that choice's continuation,
  `arguments.gen_args_<target>.arg_1`. It is the same string, with its leading
  space, that the harness scores and that `acc_bytes` normalises by.
- **Item bBPB** = `-ll_gold / ln 2 / bytes_gold`, the bits the model spends
  per byte of the correct answer given the prompt. It is the OLMES / Heineman
  et al. (2025) `correct_bpb`, and it is tokenizer-free like the FineWeb2 BPB.
- **Task bBPB** is the mean over the task's items (OLMES's item mean, not
  lm-eval's total-over-total `bits_per_byte`), at the run's final checkpoint.
  Lower is better. DA compares signs of differences, so no flip is needed when
  both sides are bBPB. Against accuracy the proxy side is negated.
- **Which tasks.** Every parent task with per-choice log-likelihoods: the
  trained-language parents of `ladder_frame` (rules 2 and 6).
  - Generative tasks have no per-choice log-likelihoods and drop out.
  - xwinograd and lambada drop out too, because their `target` is the answer
    string, not a choice index. That leaves 816 tasks.
  - *Lettered* tasks (mean gold length ≤ 2.5 bytes) score the continuation
    `" A"`. Their bBPB is the letter's surprisal, not the answer text's.
    `belebele`, `global_mmlu_full` and `include_base_44` are examples.
  - Lettered tasks are summarised apart from the cloze ones. The `rf_` twins
    are the cloze reading of the same items.
- **Three readings, one kernel.** `pair_agreement` over the same families and
  pairs for all three, with rule 5's NaN below three pairs:
  - acc → 1.7B acc;
  - bBPB → 1.7B acc, the practical question: does a small model's bBPB predict
    the reference's accuracy ranking?
  - bBPB → 1.7B bBPB.
- **The gate (rule 1) applies to accuracy only.** bBPB has no chance level.
  - The head-to-head population is the tasks above chance at 1.7B, where the
    target ranking is not noise.
  - acc → acc also needs the task above chance at the proxy, as in every rq02
    figure, so its task count is smaller.
  - bBPB is not gated at the proxy. That would discard exactly the regime it
    is for: a proxy at chance on accuracy whose bBPB still separates the
    designs.
  - The paired test (Wilcoxon, bBPB → acc minus acc → acc) runs on the tasks
    where both are defined.
  - The raw ungated DAs and both gate flags stay in `bench_bpb_da_size_multi_axes.csv`.

<!-- BEGIN auto:bench-bpb (bench_bpb_da.py --pool predictivity_schemes) -->
## Benchmark BPB against accuracy

DA-size, final checkpoints, multi-axis pairs of `predictivity_schemes` (325 pairs), gate `predictivity` on the accuracy side only: every reading counts the tasks above chance at 1.7B, and acc → acc also needs the task above chance at the proxy (its task count is the smaller one). The paired gain is bBPB → acc minus acc → acc on the tasks where both are defined. FineWeb2 val BPB is `bpb_macro`'s DA-size from `da_per_task.csv`. Regenerate with `python analysis/rq02_decision_accuracy/bench_bpb_da.py --pool predictivity_schemes` (after `build_per_item_store.py --pool predictivity_schemes --finals-only`).

**all benchmarks**

| proxy | acc → acc (tasks) | bBPB → acc (tasks) | bBPB → bBPB | paired gain (tasks) | bBPB better / worse | Wilcoxon p | FineWeb2 val BPB |
|---|---|---|---|---|---|---|---|
| 90M | 0.52 (294) | 0.61 (492) | 0.70 | +0.11 (294) | 71% / 20% | <0.001 | 0.95 |
| 175M | 0.54 (330) | 0.59 (492) | 0.65 | +0.06 (330) | 59% / 33% | <0.001 | 0.97 |
| 350M | 0.53 (367) | 0.59 (492) | 0.68 | +0.07 (367) | 63% / 28% | <0.001 | 0.98 |
| 600M | 0.55 (400) | 0.59 (492) | 0.66 | +0.05 (400) | 58% / 34% | <0.001 | 0.95 |
| 1B | 0.57 (442) | 0.60 (492) | 0.67 | +0.03 (442) | 53% / 36% | <0.001 | 0.97 |

**all cloze** (the answer text is the continuation)

| proxy | acc → acc (tasks) | bBPB → acc (tasks) | bBPB → bBPB | paired gain (tasks) | bBPB better / worse | Wilcoxon p | FineWeb2 val BPB |
|---|---|---|---|---|---|---|---|
| 90M | 0.52 (291) | 0.61 (478) | 0.70 | +0.11 (291) | 72% / 20% | <0.001 | 0.95 |
| 175M | 0.53 (328) | 0.59 (478) | 0.66 | +0.06 (328) | 59% / 33% | <0.001 | 0.97 |
| 350M | 0.53 (365) | 0.60 (478) | 0.69 | +0.07 (365) | 64% / 28% | <0.001 | 0.98 |
| 600M | 0.55 (397) | 0.60 (478) | 0.66 | +0.05 (397) | 58% / 34% | <0.001 | 0.95 |
| 1B | 0.57 (438) | 0.60 (478) | 0.67 | +0.03 (438) | 53% / 36% | <0.001 | 0.97 |

**all lettered** (the continuation is the letter: bBPB is the letter's surprisal)

| proxy | acc → acc (tasks) | bBPB → acc (tasks) | bBPB → bBPB | paired gain (tasks) | bBPB better / worse | Wilcoxon p | FineWeb2 val BPB |
|---|---|---|---|---|---|---|---|
| 90M | 0.56 (3) | 0.54 (14) | 0.52 | -0.11 (3) | 33% / 33% | 1.000 | 0.95 |
| 175M | 0.75 (2) | 0.48 (14) | 0.52 | -0.15 (2) | 50% / 50% | 1.000 | 0.97 |
| 350M | 0.33 (2) | 0.50 (14) | 0.52 | +0.08 (2) | 50% / 0% | 1.000 | 0.98 |
| 600M | 0.49 (3) | 0.52 (14) | 0.56 | +0.03 (3) | 67% / 33% | 0.750 | 0.95 |
| 1B | 0.55 (4) | 0.45 (14) | 0.61 | -0.14 (4) | 25% / 50% | 0.500 | 0.97 |

**Per benchmark**, mean over the five proxy sizes (tasks: the parent tasks with bBPB; cells: task-mean DA over those above chance at 1.7B, blank = all gated):

| benchmark | tasks | acc → 1.7B acc | bBPB → 1.7B acc | bBPB → 1.7B bBPB |
|---|---|---|---|---|
| acp_bench_cloze | 7 |  |  |  |
| arc | 28 | 0.56 | 0.64 | 0.63 |
| arc_mt | 11 | 0.58 | 0.63 | 0.64 |
| bbh_cloze | 6 |  |  |  |
| bbh_mcq | 17 |  |  |  |
| global_piqa_nonparallel_cloze | 2 | 0.79 | 0.75 | 0.90 |
| global_piqa_parallel_cloze | 63 |  | 0.51 | 0.54 |
| hellaswag | 26 | 0.76 | 0.82 | 0.87 |
| include_v2_en | 77 | 0.48 | 0.49 | 0.52 |
| include_v2_og | 77 | 0.54 | 0.59 | 0.64 |
| mathqa | 1 | 0.53 | 0.51 | 0.49 |
| multiblimp | 34 | 0.65 | 0.66 | 0.69 |
| openbookqa | 1 |  | 0.56 | 0.57 |
| paws | 8 | 0.58 | 0.54 | 0.68 |
| acp_bench_mcq-rf | 7 | 0.43 | 0.50 | 0.54 |
| bbh_mcq-rf | 17 | 0.46 | 0.47 | 0.54 |
| belebele-rf | 59 | 0.49 | 0.58 | 0.71 |
| commonsense_qa-rf | 1 | 0.53 | 0.57 | 0.58 |
| cultural_bench_easy-rf | 19 | 0.36 | 0.43 | 0.53 |
| global_mmlu_full-rf | 29 | 0.55 | 0.68 | 0.64 |
| include_base_44-rf | 36 | 0.53 | 0.62 | 0.69 |
| mmlu-rf | 1 | 0.64 | 0.67 | 0.60 |
| belebele-rfgm | 59 | 0.50 | 0.59 | 0.74 |
| include_base_44-rfgm | 36 | 0.55 | 0.63 | 0.80 |
| toxigen | 1 |  |  |  |
| truthfulqa-multi_mc1 | 2 |  |  |  |
| truthfulqa_mc2 | 3 |  |  |  |
| xcopa | 8 | 0.49 | 0.63 | 0.75 |
| xnli | 15 | 0.51 | 0.53 | 0.70 |
| xstorycloze | 8 | 0.66 | 0.74 | 0.86 |
| acp_bench_mcq (letter) | 7 |  |  |  |
| belebele (letter) | 59 | 0.58 | 0.51 | 0.55 |
| blend_sample (letter) | 5 |  |  |  |
| commonsense_qa (letter) | 1 |  |  |  |
| cultural_bench_easy (letter) | 19 | 0.30 | 0.46 | 0.58 |
| global_mmlu_full (letter) | 29 |  |  |  |
| include_base_44 (letter) | 36 | 0.57 | 0.49 | 0.54 |
| mmlu (letter) | 1 |  |  |  |

![bBPB DA, overall](pretraining/predictivity_schemes/bench_bpb_da_size_bars_multi_axes.png)

![bBPB DA, per benchmark](pretraining/predictivity_schemes/bench_bpb_da_size_bars_benchmarks_multi_axes.png)

![bBPB DA, heat map](pretraining/predictivity_schemes/bench_bpb_da_size_heatmap_multi_axes.png)
<!-- END auto:bench-bpb -->

Snapshot: ladder report cached 2026-10-02 06:24, samples read from
`eval_logs` on 2026-10-02 (816 parent tasks with bBPB at the finals of 150
cells). Raw per-task DAs and both gate flags: [`bench_bpb_da_size_multi_axes.csv`](pretraining/predictivity_schemes/bench_bpb_da_size_multi_axes.csv);
every number below is in [`bench_bpb_da_size_summary_multi_axes.csv`](pretraining/predictivity_schemes/bench_bpb_da_size_summary_multi_axes.csv).

**Key findings**

- **bBPB predicts the 1.7B accuracy ranking better than accuracy does.**
  - Paired over the tasks where both readings are defined, bBPB → acc minus
    acc → acc is +0.11 at 90M (0.61 against 0.52; bBPB better on 71 % of
    tasks, worse on 20 %).
  - The gain shrinks with the proxy size: +0.06 at 175M, +0.07 at 350M, +0.05
    at 600M and +0.03 at 1B, where it is better on 53 % and worse on 36 %.
    Wilcoxon p < 0.001 at every size.
- **It reads more tasks.** bBPB → acc is defined on 492 tasks at every proxy
  size. acc → acc is defined on 294 at 90M and 442 at 1B, because accuracy
  must also clear chance at the proxy.
- **It is flat in size, accuracy is not.** bBPB → acc stays at 0.59–0.61
  from 90M to 1B, while acc → acc only climbs from 0.52 to 0.57.
  - A 90M model's bBPB is as useful a proxy as a 1B model's.
  - Most of the gain is where accuracy is still noise.
- **bBPB ranks itself more consistently than it predicts accuracy.**
  - bBPB → 1.7B bBPB is 0.65–0.70, above bBPB → acc (0.59–0.61): part of
    what a small model's bBPB gets right is the reference's bBPB ranking,
    which the reference's accuracy does not share. How much is a follow-up.
  - It is still far below FineWeb2 validation BPB (`bpb_macro`, 0.95–0.98),
    which averages 1M tokens per language against a benchmark's few hundred
    items.
- **Per benchmark.** Means are over the five proxy sizes, and the arrows read
  as acc → acc, then bBPB → acc.
  - The largest gains are on the multilingual cloze sets: global_mmlu_full-rf
    0.55 → 0.68, xcopa 0.49 → 0.63, include_base_44-rf 0.53 → 0.62 and
    belebele-rf 0.49 → 0.58.
  - hellaswag (0.76 → 0.82) and xstorycloze (0.66 → 0.74) were already
    reliable and stay the best.
  - The include_v2 English questions do not move (0.48 → 0.49).
  - The Gemini-rewritten twins rank themselves best on bBPB: include_base_44
    rfgm 0.80, belebele rfgm 0.74.
- **The lettered tasks give no reading.**
  - Only 14 of 157 lettered tasks are above chance at 1.7B, and their
    continuation is a letter.
  - bBPB → acc there is 0.45–0.54, i.e. a coin flip. The `rf_` twins are the
    way to read those items continuously.

**Follow-ups**

- **The full checkpoint grid.** Run `build_per_item_store.py` without
  `--finals-only` (normal partition, about 4 h). That gives bBPB's DA-ckpt,
  DA-goal and SNR on the ten checkpoints, the readings figures 3 and 7 use for
  accuracy.
- **Feed bBPB into the shared pipeline.** Write it into `ladder_report.csv` as
  a `bbpb__<task>` kind (not `bpb__`, which is the FineWeb2 family in
  `snr/download/ladder.py`). Then `compute_da.py`, the scale-convergence
  figure and rq03's SNR read it without a separate script.
- **xwinograd and lambada.** Their `target` is the answer string, not a
  choice index, so they have no bBPB yet.
  - xwinograd's gold index is the doc's `answer` field.
  - lambada has one continuation, so its gold is that continuation.
  - Both are a change to `gold_index` and a re-extract of those two families.
- **Seed null.** Repeat on `predictivity_seeds` to place bBPB's 0.61 against
  the two-seeds-of-one-design null of figure 7, as DA-ckpt was.
- **The reference's two rankings.** The DA between the 1.7B bBPB and the
  1.7B accuracy rankings, per task, bounds what bBPB → acc can reach and
  separates the likelihood-vs-accuracy gap from proxy noise.
- **Cross-reading.** bBPB → acc against figure 10's cross-task map: a task's
  bBPB as the proxy for another task's accuracy.

GitHub: [bench_bpb_da_size_bars_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity_schemes/bench_bpb_da_size_bars_multi_axes.png) · [bench_bpb_da_size_bars_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity_schemes/bench_bpb_da_size_bars_multi_axes.csv) ·
GitHub: [bench_bpb_da_size_bars_benchmarks_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity_schemes/bench_bpb_da_size_bars_benchmarks_multi_axes.png) · [bench_bpb_da_size_bars_benchmarks_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity_schemes/bench_bpb_da_size_bars_benchmarks_multi_axes.csv) ·
GitHub: [bench_bpb_da_size_heatmap_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity_schemes/bench_bpb_da_size_heatmap_multi_axes.png) · [bench_bpb_da_size_heatmap_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity_schemes/bench_bpb_da_size_heatmap_multi_axes.csv)

### 12. Read next in the other RQs

Three rq02 readings live where their figures belong; the auto block of the
first stays here because its script reads rq02's per-cell table.

- **Scaling cleanly and ranking like the reference are different
  properties** — rq01's ρ of score with size against rq02's ρ of the proxy
  ranking with the reference's, per task: Spearman across tasks 0.12 (175M)
  → 0.37 (1B); the trajectory R² is the better surrogate (0.38 → 0.62 with
  DA-size). Write-up: [rq01 figure 5](../rq01_scaling_predictability/README.md#5-scaling-cleanly-and-ranking-like-the-reference-are-different-properties).
- **FineTasks' selection criteria, judged by the reference they cannot see**
  — their composite passes 13 % of tasks at 175M and 38 % at 1B (our gate 31 %
  → 49 %); passers read DA-size 0.52–0.54 against 0.45–0.47 for the rest; no
  reference-free statistic exceeds Spearman 0.33 with DA-size, and the
  cross-variant spread at the proxy correlates negatively. Write-up:
  [rq04, FineTasks' criteria](../rq04_surrogates/README.md#finetasks-criteria-on-the-ladder).
- **The minimal language panel** — against the 1.7B macro ranking over the
  L8 panel, English alone is the worst single-language proxy at 600M and 1B
  (0.59 at 1B against 0.61–0.67 for de/ru/zh/it/es/fr) and the proxy's own
  macro the safest. Write-up:
  [rq06, the minimal language panel](../rq06_language_transfer/README.md#the-minimal-language-panel).

<!-- BEGIN auto:scaling-vs-ranking (scaling_vs_ranking.py --pool predictivity) -->
## Scaling utility against ranking utility

Per proxy size, the Spearman correlation ACROSS TASKS between rq01's scaling statistics (ρ of score with model size; R² of the size fit; R² of the trajectory fit) and rq02's ranking statistics (ρ of the proxy's ranking of the design variants with the 1.7B ranking; DA-size), and the mean DA-size of the tasks rq01 calls 'predictable across both' against the rest. Regenerate with `python analysis/rq02_decision_accuracy/scaling_vs_ranking.py --pool predictivity`.

| proxy | tasks | ρ_size vs ρ_ranking | R²_size vs DA-size | R²_traj vs DA-size | DA-size, predictable both | DA-size, other regimes |
|---|---|---|---|---|---|---|
| 90M | 221 | +0.40 | +0.40 | +0.55 | 0.55 | 0.45 |
| 175M | 252 | +0.31 | +0.29 | +0.57 | 0.57 | 0.43 |
| 350M | 282 | +0.34 | +0.29 | +0.52 | 0.56 | 0.46 |
| 600M | 312 | +0.32 | +0.28 | +0.51 | 0.58 | 0.47 |
| 1B | 313 | +0.39 | +0.31 | +0.58 | 0.60 | 0.48 |

![Scaling against ranking](pretraining/predictivity/scaling_vs_ranking_da_size_multi_axes.png)
<!-- END auto:scaling-vs-ranking -->

GitHub: [scaling_vs_ranking_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking_da_size_multi_axes.png) · [scaling_vs_ranking_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking_da_size_multi_axes.csv)

## Extensions from other sweeps

### The public model lines, one rung above the ladder

From the **external tier** (public base models, `all/external/` gate;
gemma-3, Qwen3-Base and OLMo-2, each with a base model near 1B and near 13B),
not the ladder: the "decision" is between whole lab recipes that differ in
everything at once, the token budgets differ by line and size, and three
lines are three pairs (rule 5's minimum), so the per-task DA is a lattice and
the numbers are never pooled with the ladder's one-axis decisions.

**DA-size between size buckets · tasks above chance at the 1B, 1.7B and
12–14B buckets of the external gate (54 tasks) · three line pairs · no ladder
pairs.**

<!-- BEGIN auto:public-ladders (public_ladders.py --pool predictivity) -->
## Decision accuracy one rung above the ladder: the public lines

gemma-3, Qwen3-Base and OLMo-2 each have a base model near 1B and near 13B in the external tier. Per task above chance at both buckets, DA is the share of the three line pairs the small models order like the large ones: pooled 0.73 over 54 tasks, against the ladder's own 1B → 1.7B DA-size of 0.58 on the same tasks. Per line pair: Qwen3 vs OLMo-2 0.87 against a majority baseline of 0.80 (11 minority tasks, DA there 0.55); gemma-3 vs OLMo-2 0.76 against a majority baseline of 0.81 (10 minority tasks, DA there 0.80); gemma-3 vs Qwen3 0.57 against a majority baseline of 0.56 (24 minority tasks, DA there 0.79). The majority baseline is what a proxy scores by always naming the line that usually wins at 12–14B, so only DA above it is information the small models add. Between-lab decisions on public sizes, not the ladder's one-axis ones; three pairs, so per-task values are a lattice. Regenerate with `python analysis/rq02_decision_accuracy/public_ladders.py --pool predictivity`.

| family | tasks | DA public lines (1B–1.7B → 12–14B) | DA ladder (1B → 1.7B) |
|---|---|---|---|
| multiblimp | 7 | 0.67 | 0.43 |
| global_piqa_completions | 5 | 0.73 | — |
| hellaswag | 5 | 0.67 | 0.82 |
| truthfulqa | 5 | 0.67 | — |
| xnli | 5 | 0.80 | 0.52 |
| xstorycloze | 5 | 0.87 | 0.60 |
| xcopa | 4 | 0.83 | 0.55 |
| xwinograd | 4 | 0.75 | 0.62 |
| belebele | 3 | 0.67 | — |

![Public ladders](pretraining/predictivity/public_ladders_da_size_multi_axes.png)
<!-- END auto:public-ladders -->

**Verdict.** The pooled 0.73 is within 0.07 of a majority-order baseline —
a proxy that always names the line that usually wins at 12–14B scores 0.80 /
0.81 / 0.56 on the same three pairs (DA +0.07, −0.05, +0.01 against it) — so
the figure shows lab-level differences (Qwen3 above OLMo-2 on four tasks in
five at 13B and already at 1.7B, which any leaderboard gives), not
task-level ranking preservation. On the 10–24 minority tasks, where the
reference order is the uncommon one, the small models read it at 0.55–0.80,
better than the ladder's gated 0.58 but on too few tasks to be a finding. As
evaluated today the public tier cannot say whether benchmarks read decisions
above 1.7B.

**Follow-ups** (the better routes to "which benchmarks preserve ranking at
larger scale"; `plan/next_analyses.md` §7c)

- The ladder's own 3B rung: 1.7B → 3B on four families (deep, L8/L15, A/B),
  now evaluated and read in [rq10](../rq10_size_generalisation/README.md) —
  the one within-recipe generalisation above the current reference
  (`above_reference=True` in `build_snr_pool`).
- The trend of DA-size across proxies per task: does agreement rise
  monotonically towards the reference? A task whose DA-size climbs 175M →
  1B is the one whose ranking is converging.
- More public lines with both a small and a large base release on the same
  task list (an eval launch of the `auto` list on ~15 public checkpoints),
  close enough in level that the majority baseline is near 0.5.

GitHub: [public_ladders_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/public_ladders_da_size_multi_axes.png) · [public_ladders_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/public_ladders_da_size_multi_axes.csv) ·
[public_ladders_da_size_pairs_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/public_ladders_da_size_pairs_multi_axes.csv)

### The 36-model sweep

From the **36-model sweep** (2026-04…06, 4 sizes × 3 data mixtures × 3 seeds,
pools `seeds_1904`, `seeds_28_1797`, `seeds_28_1797_1904`,
`custom_swissai_hf`): `pretraining/<pool>/da_all_per_task_both_axes.csv` and the
per-benchmark views are its decision tables with the **1B** reference and the
three FineWeb mixtures as the design variants (three pairs). They are kept as
history and not regenerated — rerun today their `decision_acc_size_*`
columns would point at a 1.7B rung the sweep never trained — and they are
never pooled with the ladder's: a different harness, task set (the 86-task
old list, no twins) and reference. The seed-holdout and per-tier readings
built on them are in rq04's extensions.

## Files

- `da_explainer.py` → `…/da_all_explainer_both_axes.{png,csv}` — the toy explainer of the
  three DA kinds, the pair sets and the value lattice (Setup; no measured number).
- `pretraining/<pool>/da_all_per_task_both_axes.csv` — the DA table (single source of truth
  for rq03; the `axes` column names the pair set, rule 15);
  `da_all_n_pairs_per_task_both_axes.csv` the pair count behind every cell;
  `da_goal_early_small_per_task_both_axes.csv` the ten-checkpoint grid.
- `…/da_all_per_benchmark_multi_axes.csv`, `da_size_per_benchmark_multi_axes.csv`,
  `da_ckpt_per_benchmark_multi_axes.csv` — long and wide per-(language, benchmark) views
  (`da_per_benchmark.py`).
- `…/early_small_da_goal*_multi_axes.csv/.png`, `safe_{size,checkpoint,flops}_da_*_multi_axes.*`,
  `highlights_da_all_multi_axes.*`, `da_size*_multi_axes.*` — `early_small.py` (figure 3).
- `…/da_all_reliable_tasks_both_axes.csv`, `da_all_reliable_by_language_multi_axes.csv`,
  `da_size_vs_da_ckpt_reliable_tasks_<t>_<red>_multi_axes.png` — `reliable_tasks.py` (figure 2).
- `…/da_all_by_L_per_task_multi_axes.csv`, `da_all_pooled_per_task_multi_axes.csv`, `pairs_da_all_by_L_multi_axes.csv`,
  `early_small_da_{goal,ckpt}_by_{L,transformation}_*` — `by_L.py` (figure 3; rq04's panels read the tables).
- `…/scale_convergence_da_size*.csv/.png` — `scale_convergence.py` (figures 1, 4;
  `_L8`, `_L8common`, `_L`, `_transformation`, `_mono_axis`, `_above_*`,
  `_flops`); `scale_convergence_da_size_lang_*` — `by_language.py` (figure 5).
- `…/rq2_da_all*.csv/.png/.svg` — `paper_rq2.py` (figure 2; the paper embeds
  `rq2_da_all_above_66_either_transformation_mono_axis`), `rq2_da_all_above_66_both_mono_vs_multi_axes.*` —
  `pair_axes.py` (figure 9).
- `…/reliability_da_size_{by_language_tier,vs_language_share}_multi_axes.*` —
  `language_tier.py` (figure 6); `seed_uncertainty_da_all_seed_null.*` — `seed_uncertainty.py`
  (figure 7); `agreement_da_size_*_multi_axes.*` — `agreement.py` (figure 8);
  `cross_task_da_{size,ckpt}*_multi_axes.*` — `cross_task.py` (figure 10);
  `bench_bpb_da*.*` — `bench_bpb_da.py` on `predictivity` (figure 11);
  `scaling_vs_ranking_da_size_multi_axes.*` — `scaling_vs_ranking.py` (rq01 figure 5);
  `public_ladders_da_size*_multi_axes.*` — `public_ladders.py` (extension).
- `pretraining/predictivity_seeds*/` — the same tables on the other ladder
  pools (Definitions in [RULES.md](../RULES.md)); the all-builds folder of
  before 2026-10-05 is an orphan the refresh lists (rule 17); `pretraining/seeds_*`,
  `custom_swissai_hf/`, `all/external/` — the 36-sweep's and the public
  models' (extensions).
- `pretraining/predictivity/README.md` — removed 2026-09-23: its write-up of
  rq00–rq02 was folded into the three RQ READMEs (README rule 1: no README
  under a pool folder).
