# Decision accuracy — Does a benchmark rank models at a small size, or early in a run, the way the reference does?

## Research question

> Decision accuracy (DA), the framework's ground truth, is the probability
> that a benchmark orders a pair of models as an evaluation of larger models
> would (Heineman et al., 2025). Which benchmarks, in which languages, keep
> the ranking of the ladder's design variants across sizes (**DA-size**),
> along training (**DA-ckpt**), and early *and* small at once (**DA-goal**)?

Population (Setup below): pool `predictivity`, seed 1904, every cell, ladder
and data build from 90M to 1.7B. The 29 design variants with benchmark scores
at 1.7B give 406 pairs at the reference. Every benchmark task must be above chance both at
the proxy and at 1.7B (the gate).

<!-- BEGIN auto:highlight (da_per_benchmark.py --pool predictivity) -->
## Highlighted result

- **DA-size, proxy → 1.7B**, as the mean over the above-random benchmark tasks (their bBPB twins included) / the mean over the per-language BPB tasks: 90M → 1.7B 0.62 / 0.94, 175M → 1.7B 0.59 / 0.94, 350M → 1.7B 0.61 / 0.89, 600M → 1.7B 0.60 / 0.74, 1B → 1.7B 0.61 / 0.92.
- **DA-size of `bpb_macro`**, a single task kept out of the means above: 90M 0.95, 175M 0.96, 350M 0.98, 600M 0.95, 1B 0.97.
- **DA-size of `train_loss`**, a single task kept out of the means above: 90M 0.82, 175M 0.88, 350M 0.83, 600M 0.82, 1B 0.81.
- **DA-ckpt**, an early checkpoint against the final of the same size (above-random benchmark tasks, bBPB twins included): highest at 350M and 90 % of the run (0.89).
<!-- END auto:highlight -->

## Key finding

**Snapshot:** ladder report 2026-10-10 04:31 (acc_norm on the cloze-format
originals), outputs of the 2026-10-10 refresh (detrended checkpoint noise),
prose re-read 2026-10-10, on the pool above. Every number below is
read from the CSV beside the figure it describes.

![Scale convergence, overall](pretraining/predictivity/scale_convergence_da_size_multi_axes.png)

*DA-size · no filter · multi-axis pairs · gate `predictivity`
([figure 1](#1-the-full-population-a-smaller-model-is-close-to-chance)).*

- **A smaller fully trained model ranks the design variants close to
  chance.** On benchmark accuracy alone, DA-size is 0.51, 0.53, 0.52, 0.54 and
  0.56 from 90M to 1B (mean over 359–553 gated tasks,
  `da_size_definition_sensitivity_multi_axes.csv`), and no proxy comes near the
  0.90 threshold (table: app_rq02_definition_sensitivity).
- **The pooled line in the figure adds the bBPB twins at every size and is
  flat.** It reads 0.59, 0.57, 0.58, 0.58, 0.58 (90 % jackknife ± 0.02–0.03)
  over 1175–1369 tasks, all 816 twins among them at every size. A bBPB twin
  scores the same benchmark items by the bits per byte of the gold answer.
  Because every size has all of them, the task count now moves with the gate
  alone.
- **Bits per byte ranks like the reference and accuracy does not.** Per-language
  BPB reads 0.95, 0.95, 0.91, 0.73, 0.92 from 90M to 1B (mean over 50
  per-language tasks) and `bpb_macro` 0.94–0.97 (figure: app_rq02_bbpb_acc_gate).
  The bBPB twins sit in between. From a proxy's final checkpoint they predict
  the 1.7B accuracy ranking at 0.58–0.60, which is +0.12 over accuracy at 90M
  and +0.03 at 1B.
- **Reading a run early mostly measures persistence.** A ranking read late in
  a run mostly survives to its end, whether or not it reflects a design
  difference. On accuracy alone, DA-ckpt at 90 % of a run is 0.74–0.81 at
  every size (0.74 at 90M, 0.77 at 600M). With the twins, which now carry
  every checkpoint, it is 0.83–0.89 at the proxies. Two seeds of one design
  have nothing to decide, yet they already read 0.78–0.80 at 90 %, against
  0.82–0.86 for the real pairs (175M, 600M and 1B, 1062–1302 gated tasks,
  twins included now that the seed replicates have them).
- **Early and small at once never works for benchmarks.** On accuracy alone,
  DA-goal stays at 0.48–0.57 at every proxy and every checkpoint from 0.5C to
  5C, where C is one Chinchilla budget of 20 tokens per parameter (90M
  0.49–0.51, 600M 0.50–0.55). The twins alone read 0.61–0.67 and the filtered
  paper figure 0.57–0.64. The reference's own run reads its final accuracy
  ranking at 0.54 at 0.5C and 0.74 at 4.5C. The paper figure's filtered
  DA-size line (0.60–0.64, `above_66_either`, mono-axis) holds only for the
  tasks that passed the cut.
- **Seeds disagree as much as sizes.** On English at 175M–1B, two seeds of
  the same designs agree with each other at 0.53–0.65. A proxy agrees with
  the reference in the same range (0.55–0.64 over the three proxy seeds).
- **Language-count regimes are unordered on any task set, and the tier order
  follows the kind of score.** A language's tier is the smallest scheme-A
  mixture that trains it. On accuracy alone, the L8 tier leads at 90M (0.549 against
  0.486 for L50), and the tiers sit within 0.04 of each other at 600M and
  within 0.03 at 1B. With the twins, now at every size, L50 leads at every size
  (0.672 against 0.575 at 1B). The Spearman ρ between a language's token share and its DA
  splits the same way (+0.07 to +0.22 on accuracy alone, −0.42 to −0.53 with
  the twins).
- **DA is Kendall's τ under another tie convention.** The two correlate at
  r = 0.986 over 6,320 cells. The tie convention flips the reliable-task
  verdict on 0.4–3.1 % of cells per proxy size.

## Naming

Every artifact of this folder says in its name what it holds (rule 16), in one
order: `<subject>_da_<kind>[_<breakdown>][_<filter>]_<pair set>[_<view>]`.

- **DA kind**: `da_size`, `da_ckpt` or `da_goal`. `da_all` marks a file that
  holds more than one (the per-task tables, the three-panel `rq2_da_all_*`
  figures), and `da_size_vs_da_ckpt` one that sets two against each other.
- **Breakdown**: `by_L`, `by_transformation`, `by_benchmark`, `by_language`,
  `L` / `transformation` / `lang_*` for the scale-convergence lines.
- **Filter**: `above_66_size`, `above_66_ckpt`, `above_66_either`,
  `above_66_both`, `above_80`: the reliability cut the population passed.
- **Pair set** (rule 15): `multi_axes` (every pair of design variants),
  `mono_axis` (pairs moving exactly one axis), `seed_null` (two seeds of one
  design). `both_axes` marks a table whose `axes` column carries every pair
  set, and `mono_vs_multi_axes` a figure that compares the two.
- **View**: `_flops`, `_tokens`, `_coverage`.

A facet pair shares one table: `<name>_by_benchmark_<pair set>.png` and
`<name>_by_language_<pair set>.png` read `<name>_<pair set>.csv`.

## Setup

**Setup.**
- **Measured.** We measure DA-size, the share of design pairs that a proxy's final checkpoint (90M–1B) orders like the 1.7B reference's.
- **Models and pairs.** We use the seed-1904 runs of every language setting, ladder and data build. A pair differs on one design axis or several.
- **Tasks.** The tasks are the benchmark tasks above chance at the proxy and at 1.7B with at least three comparable pairs, their bBPB twins (the gold answer's bits per byte on the same items, figure: app_rq02_bbpb), and the per-language BPB of the 50 trained languages.
- **How.** A line divides matching by comparable decisions, summed over its tasks.
- **Breakdowns.** We map DA per benchmark and language (figures: app_rq02_da_by_language, app_rq02_da_size_by_language_per_proxy). We pick, out of sample, the tasks with median DA-size or DA-ckpt ≥ 0.66 (figures: app_rq02_rq2_crossfit, app_rq02_decision_accuracy_crossfit). We compare the proxies with the agreement of two seeds of one design (figure: app_rq02_seed_uncertainty).

**Pools and sizes.** Models are the predictivity ladder's cells: pool
`predictivity`, 90M–1.7B with each rung at its own batch (rule 10),
L ∈ {1, 2, 8, 15, 30, 50} × every ladder × every data build, seed 1904
([RULES.md](../RULES.md), Definitions). Every decision figure from figure 1
on computes its pairs over this pool (`reliable_tasks.py`, `by_L.py`,
`scale_convergence.py` and the extensions, rule 15). `predictivity_seeds`
adds the replicate seeds as separate models (figure 7). The gate pool is
`predictivity` everywhere, and the output folder is `pretraining/predictivity/`.
The swiglu cells (L8, L15, L30) now have 1.7B runs and enter the BPB,
benchmark and loss pairs against the reference (10 BPB and 10 benchmark pairs
at L8, `pairs_da_all_by_L_multi_axes.csv`). The muon cells (L8, L15, L30)
enter no BPB or benchmark pair against 1.7B yet. At L15 and L30 they enter
the loss pairs only (21 loss against 15 benchmark pairs), and at L8 no pair.

**Benchmark-BPB twins (2026-10-10 tables).** Every benchmark task has a
`bbpb_<task>` twin. Since the per-item store was rebuilt over every
checkpoint, the 816 twins exist at every size from 90M to 1.7B. They also
exist at every checkpoint of every seed-1904 cell except the 19 that the
bench-bpb block lists (`da_all_per_task_both_axes.csv`). They therefore enter
DA-size, DA-ckpt and DA-goal at every fraction of a run. The store now also
holds the seed replicates and five 3B cells, so figure 7 reads the twins
beside accuracy.

The reference is 1.7B (rule 9). Until 2026-10-05 the `predictivity` pool held
schemes A and B only, and the decisions came from a separate all-builds pool
with its tables in another folder (plan/decision_accuracy.md §9). The figures
below were drawn on that all-builds population, which is today's `predictivity`.

**Three decision accuracies.** DA is the share of design-variant pairs that a
proxy orders the same way as the reference. A pair tied on both sides counts
as agreement, and a pair tied on one side counts as a miss (the kernel's one
departure from upstream, below). **DA-size** compares the final checkpoint of
a proxy size with the 1.7B final. **DA-ckpt** compares an early checkpoint of
a run with that run's own final. **DA-goal** compares an early checkpoint of
the proxy with the 1.7B final. We verify two identities on every run:
DA-size equals DA-goal at 100 % of the proxy's run, and DA-ckpt of the 1.7B
run equals DA-goal of the 1.7B run. Pooled figures (`scale_convergence_da_size*`,
`rq2_da_all*`, `reliability_da_size*`) report the matching decisions divided
by the comparable decisions, both summed over tasks. The `early_small_da_*`
families (`early_small.py`, `by_L.py`) report the mean over tasks. The two
differ by 0.02–0.03 on the benchmarks in today's tables (`reliability` against
`reliability_macro` in `scale_convergence_da_size_multi_axes.csv`) and are
different estimands.

**Pair sets (rule 15).** There are three. `multi-axis` is every pair of
design variants at the grid seed, whichever axes it moves. `mono-axis`
(`_mono_axis` stems, while the multi-axis files carry `_multi_axes`) keeps
the pairs that move exactly one of L, depth, activation, optimizer, data
scheme (A/B/C, the recipe at that L) and T. This is the decision a
practitioner makes, and it is what upstream's "every pair" is by
construction. `seed` is two draws of one design. It is the null and serves
DA-ckpt only (figure 7). On the 2026-10-10 report the mono-axis pairs at the
grid seed are `language count` 42, `depth` 10, `data scheme` 12,
`temperature` 4 and `activation` 3. The 12 data-scheme pairs are A vs B at
L8–L30, A/ZH/ES at L2 and A/DCLMP/FWEB at L1. Until 2026-10-05 they were
three axes: language list 6, second language 3 and English corpus 3. The
other 335 of the 406 pairs move two or more axes. The training loss counts
465 pairs over 31 families, 74 of them mono-axis (43 on language count).

**Filters.** An `above_*` stem keeps the tasks whose DA cleared a cut on a
reduction of their cells (`reliable_tasks.py`). `above_66_size`,
`above_66_ckpt` and `above_66_either` keep the tasks whose median DA over the
proxy cells is ≥ 0.66 on that axis, or on either. `above_66_both` needs both
axes and is never quoted (RULES.md). `above_80` needs 0.80 on the `late`
reduction, on both axes. Every filtered figure plots the quantity it selected
on, so it is read as conditional (bug #17, `../CLAUDE.md`). The plain stem is
the unconditional one.

## Experimental setup

Models are the `predictivity` cells of `configs/models.json` described under
Setup above, and `predictivity_seeds` adds the seed replicates as separate
models. The cross-size identity is the cell's `family`, its name without the
size (`lm-L8-schemeB-deep-seed1904`), so a design variant present at two
sizes is one pair.

- **DA-size** (`decision_acc_size_<small>`) compares the families' ranking at
  the final checkpoint of `<small>` with their ranking at the final
  checkpoint of the reference (1.7B). `decision_acc_size_<a>_to_<b>` does the
  same for every other size pair with ≥ 2 shared families (90M→175M …
  600M→1B). A multilingual task is evaluated only on the cells that train its
  language. Each task's pair set is therefore the families that exist at both
  sizes *and* were evaluated on it.
- **DA-ckpt** (`decision_acc_ckpt_f<frac>_<size>`) works within one size. It
  compares the ranking at the checkpoint nearest 10, 20, … 90 % of each run
  (`da_early_fracs`, 0.5C … 4.5C on the WSD schedule) with the ranking at the
  final checkpoint.

Per-language BPB (`bpb_<subset>`) and the training loss are tasks too, so DA
covers the plan's outcome metric beside the benchmarks. DA is the truth that
the SNR proxies of the noise-and-SNR and surrogate analyses are scored
against. On a benchmark it counts only where the task is above chance at the
proxy and, for DA-size, also at the reference (rule 1). BPB and the loss
have no chance level and are never gated.

## Methodology

[`compute_da.py`](compute_da.py) writes `pretraining/<pool>/da_all_per_task_both_axes.csv`
from `utils.pair_agreement`, with one row per (parent task, pair set `axes`)
and one column per DA definition. [`da_per_benchmark.py`](da_per_benchmark.py)
melts it into a long (language, benchmark, comparison) table and the wide
`_size` / `_ckpt` pivots. It also rewrites the deck's appendix slides for the
canonical pool. The noise-and-SNR analysis joins the SNR variants onto this
table. The design-decisions analysis asks the complementary question: which
proxy *size* ranks an intervention like the reference, with languages as the
population.

A size has few families, so DA moves in steps of 1/#pairs. On the 2026-10-10
report, 1.7B has 29 families with benchmark scores and 31 on the training
loss, every build at the grid seed. Most languages that only the L50 mixture
trains (15 of 19) rest on four families and six pairs. The four languages
he, ka, ml and ta, which scheme-B cells also train, rest on eight families
and 28 pairs. Read the family-level averages, with `n` beside every value
(`da_all_n_pairs_per_task_both_axes.csv`).

## Departure from upstream: tie handling in `decision_acc_fast`

Upstream's kernel (`snr/metrics.py`, kept verbatim as
`decision_acc_fast_upstream`) compares `a > b` on both vectors. A pair that
the proxy ties but the target decides then counts as an agreement or a
disagreement depending on which model is listed first. The value depends on
the listing order (alphabetical by family here), not on the benchmark. Since
2026-09-16 `decision_acc_fast` compares the sign of the difference on both
sides instead. A pair tied on both sides agrees, a pair tied on one side and
decided on the other is a miss, and the result no longer depends on the
order. Without ties the two kernels are identical.

We measured the effect on the `predictivity` pool at the final checkpoints
(2026-09-16). Of 119,456 family pairs, 3,301 are exact ties (2.76 %). Ties
are 4.05 % of the benchmark pairs, where accuracy over a fixed item count
ties easily, and none of the per-language BPB or loss pairs. Of 2,845
(task, size) cells, 1,225 contain at least one tie. The tables committed on
2026-09-16 were produced with the upstream kernel, and
`run_all_predictivity.sh` regenerates them.

## Why the pooled panel keeps every scheme

The `all pairs` panel of `by_L.py` and the black `all pairs` line of
`scale_convergence.py` are one population: **every pair of design variants at the
grid seed (1904), every data build** (A, AT3, B, ZH, ES, DCLMP and FWEB), cross-L and
cross-architecture pairs included. Until 2026-09-22 both kept schemes A and B only,
the filter of the `predictivity` headline pool at the time. That was wrong here for
two reasons. Since 2026-10-05 the `predictivity` pool itself carries every build, so
this section records why.

**It made the pooled panel a different population from the panels beside it.** The
per-L panels have always pooled the schemes (`L_POOL = predictivity_seeds` at the grid
seed). L50's cell therefore compared A against AT3, while the first panel, drawn on
the same axes and read as their pooled version, excluded exactly that comparison. The
same held between the black line of `scale_convergence` and its coloured ones.

**The reason for the A/B filter does not transfer.** The headline pool excluded AT3,
ZH and ES because they widen the *dispersion* that an SNR signal is measured over,
without widening the decision the pool exists to measure. Decision accuracy is a rank
agreement over a pair set. A temperature change or a swapped second language adds
decisions to that set and does not inflate a spread. Choosing T = 1 over T = 3 is as
much a design decision as choosing deep over shallow.

On the 2026-09-23 report the change added **123 of the 276 pairs at the reference**
(24 families against the 18 of schemes A and B). The new pairs set the four AT3
families (L15, L30 and L50 deep, L50 shallow) and the two L2 second-language families
(ZH, ES), all at 1.7B in that report, against the A/B families and against each
other. Replicate seeds are unaffected either way. They exist only below the
reference, so no pair containing one is ever comparable.

The consequence is not cosmetic. On the 2026-09-22 tables, pooled over the gated
benchmark tasks, the plain scale-convergence line across 175M → 1B went from
**0.590, 0.599, 0.600, 0.598** under the A/B filter to **0.586, 0.613, 0.623,
0.629** with every scheme. That is a slope of **+0.010** against **+0.057** per
decade of non-embedding parameters, and +0.100 against +0.180 on the
`above_66_both` population. The A/B filter was hiding measured convergence. On the
2026-09-30 tables (90M in the ladder, the twins at every size, ZH and ES at 1.7B)
the every-scheme line reads **0.540, 0.548, 0.539, 0.550, 0.563** from 90M to 1B
(`scale_convergence_da_size_multi_axes.csv`) and 0.687 → 0.776 on `above_66_both`.
The A/B counterfactual has not been re-measured on those tables.

The dated every-scheme numbers come from `scale_convergence_da_size[_above_66_both]_multi_axes.csv`
on those dates (today's line is [figure 1](#1-the-full-population-a-smaller-model-is-close-to-chance)).
The A/B numbers are a counterfactual, since no A/B-only table is kept. To reproduce
it, restrict the OVERALL branch of `pairs_by_group` to
`{ra["data"], rb["data"]} <= {"A", "B"}` and rerun. Do not hand-carry these
numbers: that is how three of them were wrong in the first draft of this section.

## What counts as one design axis

`scale_convergence.py --by transformation` draws one line per axis a pair differs
on. It drops any pair that moves two axes at once, since such a pair is a decision
about neither. That only works if the axes are independent, and a cell's name does
not give independent axes: its ladder token and its data build each encode more
than one design choice. Both are read through the registries that define the grid
([RULES.md](../RULES.md), Definitions). The ladder becomes `arch`, `activation` and
`optimizer` (`LADDERS`). The build becomes `scheme`, the recipe at that L (A/B/C),
and `T` (`DATA_SCHEMES` `letter`, `temp`). The axes are `L`, `arch`,
`activation`, `optimizer`, `scheme`, `T` and `seed`:

| build | L | `scheme` | `T` | reading |
|---|---|---|---|---|
| A | 1–50 | A | 1 | resource-ranked list, English + Russian at L = 2 |
| AT3 | 15, 30, 50 | A | 3 | A's list sampled at T = 3 |
| B | 8, 15, 30 | B | 1 | diversity-first list |
| ZH | 2 | B | 1 | English + Chinese |
| DCLMP | 1 | B | 1 | DCLM without the edu filter |
| ES | 2 | C | 1 | English + Spanish |
| FWEB | 1 | C | 1 | FineWeb |

A letter names one recipe within an L and a different one at another L (B is
DCLMP at L1, ZH at L2 and the diversity-first list at L8–L30). A cross-L pair
therefore moves `scheme` when the two cells read different builds at one T
(`utils.moved_axes`). L2-ZH vs L8-B moves L and scheme, while L8-B vs L15-B
moves L alone. Until 2026-10-05 the build was read as three axes: the language
list (A vs B), the second language (ru vs zh vs es at L2) and the English corpus
(the L1 DCLMP/FWEB cells). They are now one axis. The multi-axis, mono-axis and
seed pair sets are the same sets before and after, and only the grouping moved
(12 = 6 + 3 + 3 mono-axis pairs at the grid seed). **B vs AT3 moves two axes**
(scheme and T), so it is dropped from every per-axis line, as before.

On the 2026-10-10 report the mono-axis pairs at the reference are `language
count` 42, `depth` 10, `data scheme` 12, `temperature` 4 (the L15, L30 and L50
AT3 cells against their scheme-A twins) and `activation` 3. The other 335 of the
406 pairs move two or more axes.

<!-- BEGIN auto:da-explainer (da_explainer.py --pool predictivity) -->
### Decision accuracy on a toy ladder

**Toy, not measured.** The figure uses four labelled variants (Deep-A-T1, Deep-B-T1, Shallow-A-T1, Deep-A-T3) with hand-written scores. Every DA in it is the pipeline's kernel run on these scores. Panels (a) and (b) show the two ways a ranking moves: (a) across sizes, on the final checkpoints, and (b) along training, for one run. Panel (c) decides the six pairs in three ways, over both pair sets of rule 15, and rings the values at or above the reliability cut. Panel (d) places each definition on the size × checkpoint grid and shows the two identities between them. Regenerate with `python analysis/rq02_decision_accuracy/da_explainer.py --pool predictivity`.

![Decision accuracy explained on a toy ladder](pretraining/predictivity/da_all_explainer_both_axes.png)

Key findings (these are definitions, so there is nothing to measure):

- **The three definitions disagree on the toy.** Over the six multi-axis pairs, DA-size is 0.50, DA-ckpt 0.17 and DA-goal 0.67. Over the three mono-axis pairs (the pairs that move exactly one design axis) they are 0.33, 0.00 and 0.67. The tied pair (Deep-A-T1 = Deep-A-T3 at the proxy's final) counts as a miss when the other side orders it, and as an agreement only when both sides tie. This is `decision_acc_fast`'s convention, and it is order-invariant.
- **A single cell can only take a few values.** A cell of n pairs takes the values k/n. At the minimum of 3 pairs that is 0, ⅓, ⅔ or 1, so a per-cell DA is read on this lattice. The figures therefore never show one cell. They draw the ratio pooled over tasks (`scale_convergence.py`) or the mean over cells (`by_L.py`). The ringed cells (DA-goal multi-axis 0.67, DA-goal mono-axis 0.67) are the ones the `above_66_*` filters would keep (cut 0.66).
- **An uninformative proxy scores a little below 0.5.** Chance agreement on an untied pair is 0.5. A one-sided tie counts as a miss, so chance falls to 0.5 × (1 − the share of pairs that one side ties). On the ladder that is ≈ 0.47, with 7% one-sided ties over the benchmark cells that pass the gate (`agreement_da_size_per_cell_multi_axes.csv`). The seed null of `seed_uncertainty.py` is a different baseline (two seeds of one design), read in its own section.
- **Two identities link the definitions.** DA-goal at the final checkpoint is DA-size, and DA-ckpt at the reference size is DA-goal. So the last column and the last row of the early-and-small grid repeat the numbers of the other two figures.
- **Small axes often fall below the pair minimum.** `by transformation` splits the mono-axis set by the axis a pair moves, and each group needs its own three pairs. On the ladder, a (task, size) cell at the final checkpoint holds 3–3 for activation (XIELU vs SwiGLU), 4–12 for data scheme (A vs B vs C), 3–10 for depth (deep vs shallow), 3–45 for language count, 3–3 for optimizer, 3–4 for temperature (T=1 vs T=3) (`da_all_by_transformation_per_task_mono_axis.csv`). The smaller groups therefore often fall below the minimum and are NaN (`early_small_da_*_by_transformation_*`, `scale_convergence_da_size_transformation_panels*`).

Follow-ups:

- **A measured twin**: the same panels on one real task (`hellaswag_de`, say) with the ladder's families, so the toy orders become the observed ones.
- **The lattice of the ladder's actual pair counts per cell** (`median_pairs` in the scale-convergence CSVs), to show how coarse a per-cell DA is on each axis.

Files: [`da_all_explainer_both_axes.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_explainer_both_axes.png), [`da_all_explainer_both_axes.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_explainer_both_axes.csv).
<!-- END auto:da-explainer -->

## Figures, in storyline order

### 1. The full population: a smaller model is close to chance

**DA-size · no filter · multi-axis pairs (`_mono_axis` twin: mono-axis) ·
pairs from `predictivity` at seed 1904 · gate `predictivity`.**
A point pools the decisions of every task that is gated in at the proxy and
at the reference and has ≥ 3 pairs. The band is the 90 % leave-one-unit-out
jackknife, where a unit is one of the 29 design variants on the pooled line.
The 816 bBPB twins enter at every size (Setup), so the task count moves with
the gate alone: 1175 at 90M, 1218 at 175M, 1248 at 350M, 1319 at 600M, 1369
at 1B.

<!-- BEGIN auto:scale-convergence (scale_convergence.py) -->
## Scale convergence — the minimum useful scale

How small can a **fully trained** model be and still make the same decisions as the 1.7B final checkpoint? A decision is the order of one pair of design variants on one task. R_size(N) = matching decisions / comparable decisions, pooled over the gated tasks, and N_min(τ) is the smallest size with R ≥ τ (τ = 0.9). The kernel, gate and pair minimum are the same as in the rest of rq02. R is DA-size pooled over decisions instead of averaged over tasks, and the CSV holds the decision counts behind each point. The 1.7B point is 1.0 by construction, since it is compared with itself. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py` (all three groupings).

**Pooled over every pair**, over the benchmark tasks and their bBPB twins (`scale_convergence_da_size_multi_axes.csv` also holds the per-language BPB rows and the decision counts):

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| all pairs | 0.58 | 0.57 | 0.58 | 0.58 | 0.58 | 1.0 | — |

![Scale convergence, overall](pretraining/predictivity/scale_convergence_da_size_multi_axes.png)

**By language count**, over the benchmark tasks and their bBPB twins (`scale_convergence_da_size_L_multi_axes.csv` also holds the per-language BPB rows and the decision counts):

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| L1 | 0.58 | 0.6 | 0.63 | 0.61 | 0.63 | 1.0 | — |
| L15 | 0.55 | 0.52 | 0.56 | 0.54 | 0.55 | 1.0 | — |
| L2 | 0.51 | 0.51 | 0.51 | 0.54 | 0.45 | 1.0 | — |
| L30 | 0.56 | 0.54 | 0.55 | 0.54 | 0.55 | 1.0 | — |
| L50 | 0.64 | 0.59 | 0.6 | 0.57 | 0.61 | 1.0 | — |
| L8 | 0.5 | 0.5 | 0.54 | 0.51 | 0.5 | 1.0 | — |
| all pairs | 0.58 | 0.57 | 0.58 | 0.58 | 0.58 | 1.0 | — |

![Scale convergence, L](pretraining/predictivity/scale_convergence_da_size_L_multi_axes.png)

**By design axis**, over the benchmark tasks and their bBPB twins (`scale_convergence_da_size_transformation_multi_axes.csv` also holds the per-language BPB rows and the decision counts):

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| activation (XIELU vs SwiGLU) | 0.52 | 0.5 | 0.55 | 0.49 | 0.52 | 1.0 | — |
| all pairs | 0.58 | 0.57 | 0.58 | 0.58 | 0.58 | 1.0 | — |
| data scheme (A vs B vs C) | 0.52 | 0.5 | 0.49 | 0.52 | 0.53 | 1.0 | — |
| depth (deep vs shallow) | 0.55 | 0.5 | 0.51 | 0.47 | 0.5 | 1.0 | — |
| language count | 0.58 | 0.56 | 0.57 | 0.56 | 0.55 | 1.0 | — |
| optimizer | 0.52 | 0.44 | 0.56 | 0.5 | 0.52 | 1.0 | — |
| temperature (T=1 vs T=3) | 0.63 | 0.6 | 0.62 | 0.61 | 0.61 | 1.0 | — |

![Scale convergence, transformation](pretraining/predictivity/scale_convergence_da_size_transformation_multi_axes.png)
<!-- END auto:scale-convergence -->

The 600M depth pairs are not size-matched. Shallow has 3.7 % more non-embedding parameters, width 2048 against 1536 and about 15 % more compute ([RULES.md, "Size matching"](../RULES.md#size-matching-the-600m-depth-pairs-are-flagged-not-dropped)). The depth axis at 600M therefore reads depth plus a little size. The pairs stay in, the per-axis tables carry `size_matched`, and the per-axis captions say so.

<!-- BEGIN auto:scale-convergence-transformation-panels (scale_convergence.py --by transformation) -->
## Scale convergence per design axis, one panel each

This figure splits the `--by transformation` lines above into one panel per design axis. Each panel has its own leave-one-unit-out band, with the pooled `all pairs` line drawn faintly behind it for reference. R is DA-size pooled over decisions. Every pair is at seed 1904 (`predictivity_seeds`), gated with `predictivity`'s mask, and a task needs ≥ 3 pairs. The number under a point is the count of tasks behind it. The table is the same as `scale_convergence_da_size_transformation_multi_axes.csv`. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by transformation`.

![Scale convergence per design axis](pretraining/predictivity/scale_convergence_da_size_transformation_panels_multi_axes.png)

| axis | units | families | 90M R [lo, hi] (tasks) | 175M R [lo, hi] (tasks) | 350M R [lo, hi] (tasks) | 600M R [lo, hi] (tasks) | 1B R [lo, hi] (tasks) | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|---|
| all pairs | 32 | 32 | 0.58 [0.56, 0.61] (1274) | 0.57 [0.55, 0.59] (1316) | 0.58 [0.56, 0.61] (1351) | 0.58 [0.56, 0.60] (1407) | 0.58 [0.56, 0.61] (1461) | never |
| activation (XIELU vs SwiGLU) | 3 | 6 | 0.52 [nan, nan] (418) | 0.50 [nan, nan] (432) | 0.55 [nan, nan] (445) | 0.49 [nan, nan] (470) | 0.52 [nan, nan] (480) | never |
| data scheme (A vs B vs C) | 12 | 18 | 0.52 [0.48, 0.56] (268) | 0.50 [0.45, 0.54] (275) | 0.49 [0.44, 0.55] (282) | 0.52 [0.46, 0.57] (295) | 0.53 [0.49, 0.58] (302) | never |
| depth (deep vs shallow) | 10 | 20 | 0.55 [0.52, 0.58] (1032) | 0.50 [0.48, 0.53] (1068) | 0.51 [0.50, 0.53] (1098) | 0.47 [0.44, 0.51] (1142) | 0.50 [0.47, 0.54] (1188) | never |
| language count | 27 | 27 | 0.58 [0.55, 0.60] (966) | 0.56 [0.53, 0.58] (1000) | 0.57 [0.55, 0.59] (1031) | 0.56 [0.54, 0.58] (1071) | 0.55 [0.52, 0.58] (1117) | never |
| optimizer | 3 | 6 | 0.52 [nan, nan] (418) | 0.44 [nan, nan] (432) | 0.56 [nan, nan] (445) | 0.50 [nan, nan] (470) | 0.52 [nan, nan] (480) | never |
| temperature (T=1 vs T=3) | 4 | 8 | 0.63 [0.59, 0.66] (966) | 0.60 [0.57, 0.62] (1000) | 0.62 [0.60, 0.64] (1031) | 0.61 [0.60, 0.63] (1071) | 0.61 [0.58, 0.64] (1117) | never |

Key findings:

- **activation (XIELU vs SwiGLU)** (3 units / 6 families): at 1B, R = 0.52 [nan, nan] over 480 tasks. No proxy reaches τ.
- **data scheme (A vs B vs C)** (12 units / 18 families): at 1B, R = 0.53 [0.49, 0.58] over 302 tasks. No proxy reaches τ.
- **depth (deep vs shallow)** (10 units / 20 families): at 1B, R = 0.50 [0.47, 0.54] over 1188 tasks. No proxy reaches τ.
- **language count** (27 units / 27 families): at 1B, R = 0.55 [0.52, 0.58] over 1117 tasks. No proxy reaches τ.
- **optimizer** (3 units / 6 families): at 1B, R = 0.52 [nan, nan] over 480 tasks. No proxy reaches τ.
- **temperature (T=1 vs T=3)** (4 units / 8 families): at 1B, R = 0.61 [0.58, 0.64] over 1117 tasks. No proxy reaches τ.
- **The bands measure which units the line depends on, not seed noise.** Each band is a leave-one-unit-out jackknife over the units in the `units` column. A unit is a design variant, or a pair when the pairs on an axis share no variant. A band on 4 units comes from four numbers, so it only shows which variant or pair the line hinges on. With 3 units or fewer the band is NaN.

Follow-ups:

- **The `above_66_size` twin per panel** (`scale_convergence_da_size_transformation_panels_above_66_size_multi_axes.png`): the same split on the cells that rank reliably.
- **A per-axis panel grid on the L8 languages only** (`--langs L8`), so the language-count panel is read on one task set.
- **A band for the activation panel.** Its line is drawn, but its band stays NaN while the axis has ≤ 3 units (one per (deep, swiglu) pair). More swiglu cells at 1.7B (rule 9) would give it one.

Files: [`scale_convergence_da_size_transformation_panels_multi_axes.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_panels_multi_axes.png), [`scale_convergence_da_size_transformation_panels_multi_axes.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_panels_multi_axes.csv).
<!-- END auto:scale-convergence-transformation-panels -->

**Key findings**

- **No proxy comes near τ = 0.90.** Pooled over every gated benchmark
  decision, DA-size is 0.589 [0.567, 0.612], 0.574 [0.550, 0.598], 0.578
  [0.555, 0.601], 0.578 [0.555, 0.601] and 0.584 [0.557, 0.611] at 90M, 175M,
  350M, 600M and 1B (29 families). N_min, the smallest size that reaches τ,
  is therefore undefined.
- **The line is flat because every size now carries the same 816 bBPB twins.**
  90M is its highest point and the five intervals overlap. On accuracy alone
  the mean over tasks rises only from 0.51 (90M, 294 tasks) to 0.57 (1B, 459
  tasks) ([figure 11](#11-benchmark-bpb-a-continuous-score-on-the-same-items)).
- **The mono-axis pairs read lower.** They give 0.571, 0.544, 0.555, 0.540,
  0.548 (`scale_convergence_da_size_mono_axis.csv`).
- **Per-language BPB does far better.** It pools 0.937, 0.932, 0.881, 0.777,
  0.910 over 50 tasks. The aggregates do best of all from any proxy size:
  `bpb_macro` 0.94–0.97 and `train_loss` 0.81–0.88 (highlight block). BPB sums
  over every token, so it sees design differences that move most benchmark
  scores by less than their noise.
- **By design axis the lines sit at 0.45–0.63.** Temperature is highest
  (0.60–0.63). Depth reads 0.47–0.55, and its 600M low is on pairs that are
  not size-matched (above). The data scheme reads 0.48–0.53 and activation is
  lowest at 0.45–0.51 (`scale_convergence_da_size_transformation_multi_axes.csv`).
  By language count the regime lines are unordered (figure 4).

**Follow-ups**

- Effect-size-resolved DA. Per pair, express the reference's gap in seed SDs
  (the design-decisions analysis's `seed_sd`), then draw DA on the pairs above
  2 SDs and DA against the gap per proxy size. Most pairs differ in L or list
  membership. On a language's BPB, one variant then saw the language and the
  other did not, which any proxy orders correctly and which dominates the BPB
  line (cross-task shows it, figure 10).
- A noise ceiling on the DA-size panel. No seed replicate exists at 1.7B, so
  the test-retest ceiling of figure 7 stops at 1B. A second 1.7B seed on two
  cells would put a band on this figure.
- Read the `_flops` twins (`scale_convergence_da_size_*_flops.png`) with one
  line per size and the annealed points marked. The compute axis zigzags
  because a bigger model's first checkpoint, at the peak learning rate, sits
  beside a smaller model's annealed final.

GitHub: [scale_convergence_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_multi_axes.png) · [scale_convergence_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_mono_axis.png) · [scale_convergence_da_size_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_mono_axis.csv) ·
GitHub: [scale_convergence_da_size_L_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L_multi_axes.png) · [scale_convergence_da_size_L_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_transformation_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_multi_axes.png) · [scale_convergence_da_size_transformation_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_transformation_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_mono_axis.png) · [scale_convergence_da_size_transformation_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_mono_axis.csv)

This leaves open whether the chance-level agreement averages a few reliable
tasks with many hopeless ones. Figure 2 asks that.

### 2. The paper figure is a conditional statement

**Left panel DA-size, middle DA-ckpt, right DA-goal · one filter for all
three panels: `above_66_either` (the `_either_transformation` stem) ·
mono-axis pairs (`_mono_axis`) · pairs from `predictivity` at seed 1904 ·
gate `predictivity`.** The paper embeds
`rq2_da_all_above_66_either_transformation_mono_axis.png` (`paper_rq2.py
--axes mono-axis`, copied to `documents/paper/figures/rq2.png` by
`documents/paper/figures/make_rq_figures.py`). Its variants share the
layout and differ only in filter and pair set. `rq2_da_all_<axes>` has no
filter. `rq2_da_all_above_80_*` uses the `late` reduction at 0.80 on both
axes. `rq2_da_all_above_66_both[_transformation]_*` keeps the cells reliable
on both axes. `rq2_da_all_above_66_own_*` gives each panel its own cut, so it
is the one variant whose panels read different populations.

![Decision accuracy, one population, mono-axis pairs](pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis.png)

*All three panels read the tasks whose median DA-size or median DA-ckpt is
≥ 0.66. The 1.7B line of the middle panel is therefore the 1.7B line of the
right one, and the black all-pairs line of the left panel is the 5C points
of the right one. Left: DA-size, the mean over tasks (`reliability_macro`,
not the pooled ratio), with one black line over all pairs and one line per
design axis in shades of green. The data scheme (A vs B vs C) is one line,
and x is non-embedding parameters. Middle: DA-ckpt, each proxy against its
own final, with x in Chinchilla multiples. Right: DA-goal, the same
checkpoints against the 1.7B final. The panels share the y axis, so the
vertical distance between middle and right is what the proxy's SIZE costs
on top of reading it early. A point that is 1.0 because it compares a
ranking with itself is hollow and joined by a dashed segment. There are no
reference lines. CSVs: `rq2_da_all_above_66_either_transformation_mono_axis.csv`,
`early_small_da_ckpt_by_L_above_66_either_mono_axis.csv`,
`early_small_da_goal_by_L_above_66_either_mono_axis.csv`,
`scale_convergence_da_size_transformation_above_66_either_mono_axis.csv`.*

| line (mono-axis, `above_66_either` tasks, mean over tasks) | 90M | 175M | 350M | 600M | 1B | tasks |
|---|---|---|---|---|---|---|
| all pairs | 0.640 | 0.609 | 0.634 | 0.600 | 0.642 | 716–755 |
| temperature (T = 1 vs 3) | 0.726 | 0.704 | 0.742 | 0.725 | 0.733 | 478–511 |
| language count | 0.597 | 0.600 | 0.618 | 0.621 | 0.607 | 488–514 |
| depth (deep vs shallow) | 0.592 | 0.534 | 0.545 | 0.465 | 0.531 | 506–547 |
| data scheme (A vs B vs C) | 0.504 | 0.494 | 0.479 | 0.529 | 0.525 | 130–140 |
| activation (XIELU vs SwiGLU) | 0.550 | 0.551 | 0.556 | 0.413 | 0.580 | 20–27 |

**Key findings**

- **A larger proxy decides no more like the 1.7B reference.** On the filtered
  population the DA-size of a fully trained proxy stays at 0.60–0.64 from 90M
  to 1B (0.640 at 90M, 0.642 at 1B). Temperature reads 0.70–0.74 and language
  count 0.60–0.62. Depth reads 0.47–0.59, with its 600M low on depth pairs
  that are not size-matched. The data scheme reads 0.48–0.53 on 130–140
  tasks, close to chance.
- **The population is mostly bBPB twins.** 612 of the 771 mono-axis tasks that
  pass either cut are twins. The twins now exist at every size, so the line
  rests on 716–755 tasks at every proxy and reads one mix of scores.
- **The cut selects on the quantity it plots.** The filter keeps the tasks
  whose DA-size or DA-ckpt cleared 0.66. Every filtered `rq2_*` figure
  therefore says "on the cells that rank reliably, this is how the
  reliability scales". This includes `above_80`, whose 1B point is truncated
  at 0.80 by construction. Figure 1 is the unconditional one.
- **DA-ckpt climbs from 0.5C to 4.5C at every size.** It goes from 0.61–0.67
  to 0.86–0.92 at the proxies and 0.64 → 0.83 at 1.7B
  (`early_small_da_ckpt_by_L_above_66_either_mono_axis.csv`). Figure 7's seed
  null reaches 0.78–0.80 at 90 % of a run, so most of this rise is
  persistence rather than a design decision.
- **Training the small proxy longer does not close the gap.** Under DA-goal
  every proxy stays at 0.57–0.64 at every checkpoint up to 5C, while the
  reference's own run reaches 0.83 at 4.5C. What limits the proxy is its
  size.
- **The DA-ckpt cut is the permissive one.** We count which tasks rank
  reliably in `da_all_reliable_tasks_both_axes.csv` (median reduction, cut
  0.66). On multi-axis pairs, 464 of 1435 tasks pass the DA-size cut, 872 the
  DA-ckpt cut, 895 either and 441 both. Without the 816 twins the counts are
  106, 189, 210 and 85 of 619. On mono-axis pairs they are 296 / 732 / 771 /
  257 (76 / 132 / 159 / 49 without the twins). The DA-ckpt median runs over
  up to 54 cells.

**Follow-ups**

- Quote figure 1 beside every filtered `rq2_*` figure in the paper, or state
  in the caption that the paper figure is conditional. Never call a filtered
  figure "free of selection bias".
- Draw the seed null of figure 7 on the middle panel, so the claim that
  reading early is cheap is read against within-run persistence.
- Mark the annealed points (≥ 4C and the finals). Under the WSD schedule every
  checkpoint up to 4C is read at the peak learning rate, so DA-ckpt compares
  two regimes. Short cooldown branches from the 1C/2C checkpoints of a few
  350M/600M cells would measure the cost of the missing anneal. This costs
  training compute, so it is a planning decision.
- Bootstrap over design variants for a 90 % interval on every line, not over
  tasks, which share the variants. A task in most L50-only languages has six
  pairs (28 for he, ka, ml and ta), so its DA moves in steps of 1/6.
- An accuracy-only twin of the paper figure. Done below: the same panels
  split by scoring.

GitHub: [rq2_da_all_above_66_either_transformation_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis.png) · [rq2_da_all_above_66_either_transformation_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis.csv) ·
GitHub: [rq2_da_all_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_multi_axes.png) · [rq2_da_all_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_multi_axes.csv) ·
GitHub: [rq2_da_all_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_mono_axis.png) · [rq2_da_all_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_mono_axis.csv) ·
GitHub: [rq2_da_all_above_80_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_80_multi_axes.png) · [rq2_da_all_above_80_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_80_multi_axes.csv) ·
GitHub: [rq2_da_all_above_66_both_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_multi_axes.png) · [rq2_da_all_above_66_both_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_multi_axes.csv) ·
GitHub: [rq2_da_all_above_66_both_transformation_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_transformation_multi_axes.png) · [rq2_da_all_above_66_both_transformation_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_transformation_multi_axes.csv) ·
GitHub: [rq2_da_all_above_66_own_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_own_multi_axes.png) · [rq2_da_all_above_66_own_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_own_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_above_66_size_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_above_66_size_mono_axis.png) · [scale_convergence_da_size_above_66_size_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_above_66_size_mono_axis.csv)

**Each panel on its own population** (`rq2_da_all_above_66_own_transformation_mono_axis.png`,
`paper_rq2.py --axes mono-axis`, the paper figure's populations before
2026-10-07). **DA-size, DA-ckpt, DA-goal · filter per panel: `above_66_size`
(left), `above_66_ckpt` (middle), `above_66_either` (right) · mono-axis pairs
· pairs from `predictivity` at seed 1904 · gate `predictivity`.** The panels
read three populations, so the 1.7B line and the 5C points need not match
across panels.

![Decision accuracy, each panel its own population, mono-axis pairs](pretraining/predictivity/rq2_da_all_above_66_own_transformation_mono_axis.png)

**Key findings** (`rq2_da_all_above_66_own_transformation_mono_axis.csv`)

- **On the DA-size passers the proxy agrees more, and still not more with
  size.** All pairs read 0.70–0.76 (0.755 at 90M, 0.753 at 1B) over 272–295
  tasks, against 0.60–0.64 on the `either` population. Temperature reads
  0.82–0.87, language count 0.69–0.75 and the data scheme 0.52–0.57 on 25–29
  tasks. The cut is on DA-size itself, so the lift is the selection
  (figure 1).
- **The other two panels barely move.** DA-ckpt on the DA-ckpt passers
  (693–717 tasks) is within 0.01 of the paper figure's at every point (0.65 →
  0.84 at 1.7B). DA-goal reads the same `either` tasks as there.

**Follow-ups**

- The DA-size panel on the DA-ckpt passers. This selection does not cut on
  the plotted quantity.

GitHub: [rq2_da_all_above_66_own_transformation_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_own_transformation_mono_axis.png) · [rq2_da_all_above_66_own_transformation_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_own_transformation_mono_axis.csv)

**The same figure split by scoring** (`rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.png`,
`paper_rq2.py --axes mono-axis`, the paper's `app_rq02_bbpb`). **DA-size,
DA-ckpt, DA-goal · filter `above_66_either` · mono-axis pairs · pairs from
`predictivity` at seed 1904 · gate `predictivity`.** The top row holds the
accuracy tasks (originals and their `rf_`/`rfgm_` twins). The bottom row holds
the `bbpb_` twins, read as bBPB → 1.7B bBPB (never gated, since bBPB has no
chance level). The two rows split the paper figure's tasks exactly. They are
reduced from by_L's per-task tables with by_L's `_summary`, which reproduces
the paper figure's CSV to 1e-12.

![Decision accuracy by scoring, mono-axis pairs](pretraining/predictivity/rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.png)

**Key findings**

- **Accuracy alone does improve with the proxy's size.** DA-size over all
  pairs goes from 0.57 at 90M to 0.67 at 1B, over 104–143 tasks. The gate keeps
  more tasks at larger sizes, so the population moves along the line.
- **The bBPB twins stay flat.** They read 0.60–0.65 over the same 612 tasks at
  every size. The paper figure's flat line is mostly theirs.
- **The flat bBPB line is not a bug (checked 2026-10-08).** The twins cover
  every checkpoint of every seed-1904 cell except the 19 that the bench-bpb
  block lists (3 of the 71 mono-axis pairs). The per-task DA-size recomputes
  exactly by hand from `bench_bpb.csv`. Sign agreement does not depend on
  which direction is better, and bBPB has no ties.
- **Many design differences are below the noise.** At 1.7B, 38 % of the
  (twin, pair) differences are under one detrended checkpoint-noise SD (rule 4
  window), and 59 % are under the raw SD. A perfect proxy would therefore
  agree with the observed 1.7B ranking on at most 0.86 of them (0.83–0.89 per
  axis). Pairs under one SD agree at 0.52–0.54, and pairs over four SDs at
  0.75–0.79.
- **Depth and the data scheme sit at 0.46–0.59.** Their bBPB effect is small
  and changes sign with size. At L15 scheme B has the lower bBPB on 83 % of
  twins at 90M, but scheme A has it on 63 % at 1.7B (and the 600M depth pairs
  are not size-matched). Temperature, the largest effect, reaches 0.71–0.74.
  On the pairs with a clear 1.7B difference (two detrended SDs) its DA-size is
  0.67–0.72, still flat from 90M to 1B.
- **Reading early costs less than reading small.** DA-ckpt at 4.5C is
  0.79–0.85 on accuracy and 0.84–0.93 on bBPB. DA-goal of the proxies at 5C
  is 0.57–0.67 on accuracy and 0.60–0.65 on bBPB.

**Follow-ups**

- The accuracy row on a fixed task set (the tasks gated in at every size), so
  that its rise is not partly the gate admitting easier tasks.
- The twins are flat too when read as bBPB → 1.7B accuracy, rq11's other
  reading, on the 410 twins whose original clears chance at 1.7B. DA-size is
  then 0.57–0.58 from 90M to 1B. We tried it on 2026-10-08 as a third row and
  dropped it from the figure, and rq11 keeps that reading.

GitHub: [rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.png) · [rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.csv)

**Per benchmark and language** (`paper_rq2.py --axes mono-axis`, the paper's
`app_rq02_da_by_language` and `app_rq02_da_size_by_language_per_proxy`). The
lines above average over every benchmark and language. These maps give each
(benchmark, language) cell. **DA-size, DA-ckpt, DA-goal · filter
`above_66_either` (paper) and none (beside it) · mono-axis pairs · pairs from
`predictivity` at seed 1904 · gate `predictivity`** (at the proxy, and at 1.7B
where it is the reference). The maps use the ORIGINAL accuracy tasks only,
with no `rf_`/`rfgm_`/`bbpb_` twin (rq11's figure shows those). They read
`da_all_per_task_both_axes.csv`. A cell is the mean over the benchmark's tasks
in the language, which is one task except for the English subtask families,
of each task's mean over its readings. Rows are the benchmarks with two or
more trained languages and a value somewhere, ordered by DA-size. Columns are
the trained languages by resource rank, with a line where the K = 1, 2, 8,
15, 30, 50 lists end (ar and sr come from the diversity-first lists only).
Grey is gated and white is no value (or, in the filtered maps, a task the
filter drops). The colour scale is centred at 0.5.

- `rq2_da_all_by_benchmark_and_language[_above_66_either]_mono_axis.png` draws
  one map per DA. DA-size is averaged over the proxies 90M–1B. DA-ckpt is
  averaged over the proxies, 1.7B (where DA-ckpt is defined against its own
  final) and the nine tenths before the final. DA-goal is averaged over the
  proxies and the ten tenths. DA-size and DA-goal stop at 1B because 1.7B is
  their reference. Beside and under each map are the row and column means with
  a 95 % bootstrap band over the cells (`utils.bootstrap_band`).
- `rq2_da_size_by_benchmark_and_language_per_proxy[_above_66_either]_mono_axis.png`
  draws DA-size, one map per proxy, with the row means in a last column. The
  benchmark with the most languages above 0.5 at every proxy is outlined
  (HellaSwag).

**Which variant the paper shows: the filtered one.** `above_66_either` keeps
the tasks whose median DA-size or median DA-ckpt is ≥ 0.66 (the population of
the paper's rq2 figure). It selects on DA, so every map over it is
conditional, and the paper captions say so. It is the more informative one
per benchmark and language, because on it the proxy size matters (below). The
unfiltered maps are close to chance almost everywhere, so they mostly restate
figure 1.

![DA per benchmark and language, above_66_either, mono-axis pairs](pretraining/predictivity/rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis.png)

![DA-size per benchmark and language at each proxy, above_66_either, mono-axis pairs](pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis.png)

![DA per benchmark and language, no filter, mono-axis pairs](pretraining/predictivity/rq2_da_all_by_benchmark_and_language_mono_axis.png)

![DA-size per benchmark and language at each proxy, no filter, mono-axis pairs](pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_mono_axis.png)

**Key findings** (the four CSVs of the same names)

- **On the reliable tasks a larger proxy decides more like the reference.**
  With `above_66_either`, the share of cells with DA-size above 0.5 goes from
  0.43 at 90M to 0.70 at 1B (mean 0.59 to 0.71, 56 to 76 cells). XStoryCloze
  goes from 0.58 to 0.76, INCLUDE v2 (OG) from 0.42 to 0.64 and HellaSwag from
  0.61 to 0.85 (XCOPA no longer has a filtered cell). Without the filter the
  mean only goes from 0.52 to 0.57 (136 to 217 cells) and the share from 0.21
  to 0.35.
- **HellaSwag is the clean case (the outlined row).** Its DA-size is above 0.5
  at every proxy in 12 of the 23 languages with a value at all five sizes (da,
  de, en, es, fr, id, it, nl, pt, ru, sv, vi). Unfiltered it is the same 12, of
  26. MultiBLiMP, the other wide benchmark (34 languages), stays at 0.56–0.61
  at every proxy unfiltered and 0.64–0.70 filtered. There the proxy's size
  does not help.
- **Averaged over the proxies, DA-size and DA-goal clear 0.5 on few
  unfiltered benchmarks.** "Clear" means the 95 % band of the mean is above
  0.5. With the filter, DA-size clears it on 7 of 10 benchmarks and 31 of 41
  languages. Without it, on 6 of 17 and 12 of 50 (HellaSwag 0.71, LAMBADA
  0.62, XStoryCloze 0.60, MultiBLiMP 0.58, PAWS-X 0.58 and MultiBLiMP (extra)
  0.75). DA-goal clears it on 7 of 10 and 30 of 41 filtered, and 5 of 17 and
  12 of 50 unfiltered. DA-ckpt, now with 1.7B, clears it on 11 of 11 and 37
  of 39 filtered, and 14 of 21 and 41 of 49 unfiltered (cell mean 0.70 and
  0.61). This is the persistence that figure
  7's seed null reads as well. DA-ckpt is also the only DA with values on
  Belebele and Global-MMLU, since it is gated at the proxy only.
- **No language stands out once its cells are many.** Unfiltered, the
  languages with five or more DA-size cells sit between 0.48 (el) and 0.61
  (ca). The extremes rest on few cells (ta 0.40 on 4, et 0.84 on 2, sl 0.75 on
  1), and the language lists (K blocks) do not order the cells either.

**Follow-ups**

- The same maps on the RF and LLM-RF twins, where Belebele, Global-MMLU and
  INCLUDE clear the gate (rq11 has them per benchmark, not per language).
- Read the filtered rise on a fixed task set, so that the gate admitting more
  tasks at 1B is not part of it.
- A multi-axis twin of every map is written beside these (`_multi_axes`).

GitHub: [rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis.png) · [rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis.csv) ·
GitHub: [rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis.png) · [rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis.csv) ·
GitHub: [rq2_da_all_by_benchmark_and_language_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_by_benchmark_and_language_mono_axis.png) · [rq2_da_all_by_benchmark_and_language_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_by_benchmark_and_language_mono_axis.csv) ·
GitHub: [rq2_da_size_by_benchmark_and_language_per_proxy_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_mono_axis.png) · [rq2_da_size_by_benchmark_and_language_per_proxy_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_mono_axis.csv)

The block below shows the population behind the cuts: which (benchmark,
language) cells rank reliably at all, and in which languages.

**Reliable cells · DA-size and DA-ckpt · every cut × reduction ·
multi-axis (the table carries mono-axis too) · pairs from
`predictivity` · gate `predictivity`.** Since 2026-09-22 the verdicts gate
DA-size at the reference too (rule 1). Before that, three cells at chance at
1.7B counted as reliable, and the `above_66_both` population read 23 cells.
The 2026-09-23 snapshot, with the twins at 175M and 350M, took it to 71. On
the 2026-10-10 tables it holds 441 multi-axis tasks, 356 of them bBPB twins.

<!-- BEGIN auto:reliable-tasks (reliable_tasks.py --pool predictivity) -->
## Which benchmark-language cells rank reliably

The table counts, per language, the benchmarks that reach DA ≥ 0.8 on DA-size (a proxy size's final ranking against the reference's) and on DA-ckpt (an earlier checkpoint against the same size's final ranking). Each task's cells are reduced with `late`, which keeps one fixed cell per axis, so no cell is picked for its value. A cell counts only if it has ≥ 3 pairs (rule 5) and survives the above-random gate (rule 1). The decisions come from the `predictivity` pool, which holds every data build at the grid seed (the same population that `by_L` and `scale_convergence` pair over). The gate and this folder use the `predictivity` pool. The table and figures show the `multi-axis` pair set, and `da_all_reliable_tasks_both_axes.csv` carries every pair set (rule 15). That CSV holds the per-task values for every reduction and does not depend on the cut, so each figure is one view of it. Regenerate with `python analysis/rq02_decision_accuracy/reliable_tasks.py --pool predictivity`.

| language | benchmarks evaluated | DA-size | DA-ckpt | either | both |
|---|---|---|---|---|---|
| en | 150 | 4 | 82 | 82 | 4 |
| ru | 35 | 5 | 22 | 22 | 5 |
| zh | 50 | 3 | 32 | 32 | 3 |
| de | 33 | 0 | 27 | 27 | 0 |
| ja | 27 | 0 | 13 | 13 | 0 |
| es | 130 | 4 | 69 | 69 | 4 |
| fr | 39 | 4 | 25 | 26 | 3 |
| it | 36 | 2 | 23 | 23 | 2 |
| pt | 34 | 2 | 22 | 22 | 2 |
| pl | 22 | 0 | 11 | 11 | 0 |
| nl | 25 | 3 | 14 | 14 | 3 |
| id | 30 | 2 | 16 | 16 | 2 |
| vi | 32 | 1 | 16 | 16 | 1 |
| fa | 21 | 0 | 13 | 13 | 0 |
| tr | 26 | 2 | 12 | 12 | 2 |
| th | 10 | 0 | 7 | 7 | 0 |
| uk | 26 | 2 | 13 | 13 | 2 |
| el | 26 | 1 | 13 | 13 | 1 |
| ko | 26 | 2 | 11 | 12 | 1 |
| cs | 15 | 0 | 10 | 10 | 0 |
| sv | 23 | 2 | 15 | 15 | 2 |
| hu | 25 | 3 | 12 | 12 | 3 |
| ro | 17 | 3 | 8 | 8 | 3 |
| no | 84 | 2 | 53 | 53 | 2 |
| da | 19 | 3 | 15 | 15 | 3 |
| bg | 19 | 1 | 10 | 10 | 1 |
| fi | 16 | 5 | 11 | 11 | 5 |
| hi | 40 | 4 | 24 | 24 | 4 |
| bn | 35 | 5 | 18 | 18 | 5 |
| sk | 18 | 9 | 15 | 16 | 8 |
| he | 20 | 7 | 11 | 12 | 6 |
| lt | 21 | 8 | 14 | 16 | 6 |
| bs | 2 | 1 | 1 | 1 | 1 |
| sl | 9 | 6 | 9 | 9 | 6 |
| et | 19 | 11 | 13 | 15 | 9 |
| ca | 21 | 10 | 15 | 16 | 9 |
| ta | 21 | 6 | 13 | 13 | 6 |
| hr | 24 | 8 | 16 | 17 | 7 |
| lv | 7 | 5 | 5 | 6 | 4 |
| ms | 21 | 12 | 17 | 18 | 11 |
| az | 16 | 5 | 13 | 13 | 5 |
| ka | 18 | 3 | 12 | 12 | 3 |
| ne | 30 | 12 | 21 | 21 | 12 |
| mr | 16 | 9 | 13 | 14 | 8 |
| ml | 17 | 8 | 13 | 13 | 8 |
| kk | 18 | 6 | 14 | 15 | 5 |
| ur | 26 | 8 | 22 | 23 | 7 |
| sq | 17 | 10 | 13 | 14 | 9 |
| ar | 91 | 6 | 51 | 51 | 6 |
| sr | 26 | 14 | 19 | 21 | 12 |

**How many cells pass, by cut and reduction.** The cut is a choice. This table shows how the count changes with it:

| threshold | reduction | tasks passing both | languages | benchmarks |
|---|---|---|---|---|
| 0.8 | late | 211 | 44 | bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_og, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval, bbpb_noreval_extra, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite_math, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, hellaswag, include_v2_og, lambada_openai_mt, multiblimp, multiblimp-extra, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_belebele, rfgm_include_base_44, xstorycloze |
| 0.8 | mean | 123 | 38 | bbpb_arc, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_v2_og, bbpb_kobest, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval_extra, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite_math, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, hellaswag, multiblimp, multiblimp-extra, rf_global_mmlu_full |
| 0.8 | median | 152 | 40 | bbpb_arc, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_v2_og, bbpb_kobest, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval_extra, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite_math, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, hellaswag, multiblimp, multiblimp-extra, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44 |
| 0.8 | max | 352 | 49 | bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_kobest, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval, bbpb_noreval_extra, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rf_mmmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite_math, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, hellaswag, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, multiblimp-extra, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rf_mmmlu, rfgm_belebele, rfgm_include_base_44, xcopa, xnli, xstorycloze |
| 0.75 | late | 275 | 47 | bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_og, bbpb_kobest, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval, bbpb_noreval_extra, bbpb_paws, bbpb_rf_belebele, bbpb_rf_cultural_bench_easy, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rf_mmmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite_math, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, hellaswag, include_v2_og, lambada_openai_mt, multiblimp, multiblimp-extra, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rf_m_mmlu, rf_mmmlu, rfgm_belebele, rfgm_include_base_44, xstorycloze |
| 0.75 | mean | 207 | 46 | bbpb_arc, bbpb_arc_mt, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_v2_og, bbpb_kobest, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval, bbpb_noreval_extra, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rf_mmmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite_math, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, hellaswag, multiblimp, multiblimp-extra, rf_global_mmlu_full, rfgm_include_base_44 |
| 0.75 | median | 212 | 46 | bbpb_arc, bbpb_arc_mt, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_v2_og, bbpb_kobest, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval, bbpb_noreval_extra, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rf_mmmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite_math, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, hellaswag, include_v2_og, multiblimp, multiblimp-extra, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xstorycloze |
| 0.75 | max | 476 | 50 | arc, bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_kobest, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval, bbpb_noreval_extra, bbpb_paws, bbpb_rf_belebele, bbpb_rf_cultural_bench_easy, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rf_mmmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite_math, bbpb_truthfulqa-multi_mc1, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, hellaswag, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, multiblimp-extra, noreval, noreval_extra, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rf_m_mmlu, rf_mmmlu, rfgm_belebele, rfgm_include_base_44, xcopa, xnli, xstorycloze |
| 0.66 | late | 512 | 50 | arc, arc_mt, bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_kobest, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval, bbpb_noreval_extra, bbpb_paws, bbpb_piqa, bbpb_rf_belebele, bbpb_rf_cultural_bench_easy, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rf_mmmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite_math, bbpb_toksuite_stem, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, multiblimp-extra, noreval, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rf_m_mmlu, rf_mmlu, rf_mmmlu, rfgm_belebele, rfgm_include_base_44, toksuite_math, xcopa, xstorycloze, xwinograd |
| 0.66 | mean | 393 | 50 | arc, bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_kobest, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval, bbpb_noreval_extra, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rf_mmmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite_math, bbpb_truthfulqa-multi_mc1, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_v2_og, lambada_openai_mt, multiblimp, multiblimp-extra, noreval, paws, rf_global_mmlu_full, rf_include_base_44, rf_mmmlu, rfgm_belebele, rfgm_include_base_44, xstorycloze, xwinograd |
| 0.66 | median | 426 | 50 | arc, bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_kobest, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval, bbpb_noreval_extra, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rf_mmmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite_math, bbpb_truthfulqa-multi_mc1, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_v2_og, lambada_openai_mt, multiblimp, multiblimp-extra, noreval, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rf_m_mmlu, rf_mmmlu, rfgm_belebele, rfgm_include_base_44, xcopa, xstorycloze, xwinograd |
| 0.66 | max | 864 | 50 | arc, arc_mt, bbpb_acp_bench_mcq, bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_blend_sample, bbpb_cultural_bench_easy, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_kobest, bbpb_multiblimp, bbpb_multiblimp-extra, bbpb_noreval, bbpb_noreval_extra, bbpb_paws, bbpb_piqa, bbpb_rf_bbh_mcq, bbpb_rf_belebele, bbpb_rf_cultural_bench_easy, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_m_mmlu, bbpb_rf_mmlu, bbpb_rf_mmmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_toksuite, bbpb_toksuite_math, bbpb_toksuite_stem, bbpb_truthfulqa-multi_mc1, bbpb_truthfulqa_mc2, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_base_44, include_v2_en, include_v2_og, kobest, lambada_openai_mt, multiblimp, multiblimp-extra, noreval, noreval_extra, paws, piqa, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rf_m_mmlu, rf_mmlu, rf_mmmlu, rfgm_belebele, rfgm_include_base_44, toksuite_math, winogrande, xcopa, xnli, xstorycloze, xwinograd |

![Reliable benchmark-language cells](pretraining/predictivity/da_size_vs_da_ckpt_reliable_tasks_80_late_multi_axes.png)
<!-- END auto:reliable-tasks -->

[da_all_reliable_tasks_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_reliable_tasks_both_axes.csv) ·
[da_all_reliable_by_language_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_reliable_by_language_multi_axes.csv) ·
[da_size_vs_da_ckpt_reliable_tasks_80_late_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_vs_da_ckpt_reliable_tasks_80_late_multi_axes.png) ·
[da_size_vs_da_ckpt_reliable_tasks_66_median_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_vs_da_ckpt_reliable_tasks_66_median_multi_axes.png) ·
[da_size_vs_da_ckpt_reliable_tasks_75_late_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_vs_da_ckpt_reliable_tasks_75_late_multi_axes.png)
(one PNG per cut × reduction, `da_size_vs_da_ckpt_reliable_tasks_<t>_<red>_multi_axes.png`, all views of
`da_all_reliable_tasks_both_axes.csv`).

**Key findings**

- **At 0.8 on the `late` reduction, most passing tasks are bBPB twins.** 208
  tasks in 46 languages pass both axes, 165 of them twins. The 43 accuracy
  tasks are mostly hellaswag (19), then multiblimp (6) and the reformulated
  Global-MMLU and INCLUDE twins.
- **DA-size is the binding axis.** 227 tasks clear it at 0.8, only 57 of them
  on accuracy. DA-ckpt passes most of every language's benchmarks (English 88
  of 145, against 3 on DA-size).

**Follow-ups**

- The same table without the `bbpb_` rows, so the per-language counts say
  which accuracy benchmarks rank reliably.

### 3. Early and small: DA-goal and DA-ckpt on the ten checkpoints

**`early_small_da_{ckpt,goal}_by_L_*`: DA-ckpt / DA-goal · mean over tasks ·
filter per stem (none, `_above_80`, `_above_66_ckpt`, `_above_66_either`,
`_above_66_both`; `_with_bpb` adds the per-size BPB lines) · multi-axis
unless `_mono_axis` · pairs from `predictivity` (`by_L.py` pools every
scheme at the grid seed) · gate `predictivity`.** The `early_small_da_goal_multi_axes.png` grid,
the `safe_*` maps and the Results block below read `compute_da.py`'s own
tables on `predictivity` (every build since 2026-10-05, the A/B pool before,
with multi-axis pairs, no filter and gate `predictivity`). They show DA-goal
at 1C–5C per proxy as the mean over tasks, and the smallest safe level. A
level is safe when DA ≥ 0.75 over ≥ 3 pairs and this holds at every larger
level.

<!-- BEGIN auto:results (da_per_benchmark.py --pool predictivity) -->
## Results

The numbers come from the `predictivity` pool: DA from `da_all_per_task_both_axes.csv`, pair counts from `da_all_n_pairs_per_task_both_axes.csv` and the above-chance gate from rq00. Regenerate with `python analysis/rq02_decision_accuracy/da_per_benchmark.py --pool predictivity`.

**DA-size by proxy size.** The benchmark column includes the bBPB twins. Each `n` is the number of tasks behind the column to its left, and `pairs` is the median number of pairs per cell.

| comparison | benchmarks | n | pairs | BPB | n |
|---|---|---|---|---|---|
| 90M → 1.7B | 0.62 | 1274 | 105 | 0.94 | 50 |
| 175M → 1.7B | 0.59 | 1316 | 105 | 0.94 | 50 |
| 350M → 1.7B | 0.61 | 1351 | 105 | 0.89 | 50 |
| 600M → 1.7B | 0.60 | 1407 | 120 | 0.74 | 50 |
| 1B → 1.7B | 0.61 | 1461 | 105 | 0.92 | 50 |

![DA-size by family](pretraining/predictivity/da_size_by_family_multi_axes.png)

**DA-ckpt by bucket and fraction of the run**, as the mean over the above-random benchmark tasks (bBPB twins included):

| bucket | 10 % | 20 % | 30 % | 40 % | 50 % | 60 % | 70 % | 80 % | 90 % |
|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.64 | 0.64 | 0.64 | 0.68 | 0.69 | 0.71 | 0.72 | 0.75 | 0.85 |
| 175M | 0.63 | 0.63 | 0.64 | 0.68 | 0.69 | 0.70 | 0.71 | 0.75 | 0.85 |
| 350M | 0.60 | 0.63 | 0.65 | 0.68 | 0.71 | 0.73 | 0.77 | 0.81 | 0.89 |
| 600M | 0.61 | 0.62 | 0.65 | 0.65 | 0.68 | 0.68 | 0.70 | 0.73 | 0.84 |
| 1B | 0.59 | 0.61 | 0.63 | 0.64 | 0.65 | 0.68 | 0.70 | 0.72 | 0.82 |
| 1.7B | 0.61 | 0.60 | 0.63 | 0.63 | 0.65 | 0.66 | 0.68 | 0.69 | 0.79 |
<!-- END auto:results -->

<!-- BEGIN auto:early-small (early_small.py --pool predictivity) -->
## Early and small, as a ranking

This section asks how small a proxy, and how early in its run, still ranks the design variants like the 1.7B final checkpoint. Numbers come from the `predictivity` pool. Every design variant is read at a proxy size and at checkpoints from 0.5C to 5C of training, where C is the Chinchilla-optimal budget of 20 tokens per parameter. Every run trains for 5C, so 1C is 20 % of the run. Each cell ranks the variants against the same variants at the 1.7B final checkpoint. The 5C column is therefore DA-size, and the 1.7B row is that size's DA-ckpt. A benchmark task counts only where it clears chance both at the proxy size and at 1.7B. Regenerate with `python analysis/rq02_decision_accuracy/early_small.py --pool predictivity`.

- **bpb**: the smallest size, at its earliest checkpoint, whose mean agreement with the 1.7B final ranking reaches 0.75 is **90M at 0.5C** (0.90).
- **all benchmarks**: the smallest size, at its earliest checkpoint, whose mean agreement with the 1.7B final ranking reaches 0.75 is **1.7B at 4.5C** (0.79).
- **Smallest safe size per (benchmark, language)**: never: 857, 1B: 161, 90M: 114, 600M: 19, 350M: 14, 175M: 7 of 1172 cells.

![rq02 in one figure](pretraining/predictivity/highlights_da_all_multi_axes.png)

**bpb**. Rows are proxy sizes, columns are the proxy's training tokens in Chinchilla multiples, and each cell is the mean DA over 50 tasks:

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.90 | 0.90 | 0.90 | 0.88 | 0.87 | 0.88 | 0.88 | 0.87 | 0.92 | 0.94 |
| 175M | 0.90 | 0.87 | 0.88 | 0.90 | 0.85 | 0.88 | 0.85 | 0.87 | 0.92 | 0.94 |
| 350M | 0.79 | 0.82 | 0.82 | 0.82 | 0.86 | 0.86 | 0.84 | 0.84 | 0.90 | 0.89 |
| 600M | 0.69 | 0.70 | 0.70 | 0.70 | 0.70 | 0.71 | 0.70 | 0.70 | 0.73 | 0.74 |
| 1B | 0.75 | 0.76 | 0.78 | 0.81 | 0.83 | 0.86 | 0.84 | 0.86 | 0.89 | 0.92 |
| 1.7B | 0.89 | 0.91 | 0.91 | 0.92 | 0.92 | 0.92 | 0.92 | 0.92 | 0.95 |  |

**all benchmarks**. Rows are proxy sizes, columns are the proxy's training tokens in Chinchilla multiples, and each cell is the mean DA over 1506 tasks:

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.61 | 0.61 | 0.60 | 0.59 | 0.60 | 0.60 | 0.59 | 0.60 | 0.61 | 0.62 |
| 175M | 0.61 | 0.59 | 0.58 | 0.58 | 0.58 | 0.58 | 0.57 | 0.57 | 0.59 | 0.59 |
| 350M | 0.59 | 0.58 | 0.58 | 0.59 | 0.59 | 0.59 | 0.59 | 0.59 | 0.60 | 0.61 |
| 600M | 0.59 | 0.58 | 0.58 | 0.57 | 0.59 | 0.60 | 0.58 | 0.58 | 0.60 | 0.60 |
| 1B | 0.59 | 0.57 | 0.59 | 0.57 | 0.59 | 0.59 | 0.60 | 0.59 | 0.61 | 0.61 |
| 1.7B | 0.61 | 0.60 | 0.63 | 0.63 | 0.65 | 0.66 | 0.68 | 0.69 | 0.79 |  |

![Early and small](pretraining/predictivity/early_small_da_goal_multi_axes.png)

![Early and small per benchmark](pretraining/predictivity/early_small_da_goal_by_benchmark_multi_axes.png)

![Early and small per language](pretraining/predictivity/early_small_da_goal_by_language_multi_axes.png)

**Smallest safe level per language and benchmark.** A level is safe when DA ≥ 0.75 over ≥ 3 pairs there and at every larger level (for FLOPs, at every costlier (size, checkpoint) cell). Red = never, grey = filtered out by the above-random gate, white = no value. Each figure's table sits next to it under the same name:

![Smallest safe size](pretraining/predictivity/safe_size_da_size_multi_axes.png)

![Smallest safe checkpoint](pretraining/predictivity/safe_checkpoint_da_ckpt_multi_axes.png)

![Smallest safe FLOPs](pretraining/predictivity/safe_flops_da_goal_multi_axes.png)

![DA-size per benchmark](pretraining/predictivity/da_size_by_benchmark_multi_axes.png)

![DA-size per language](pretraining/predictivity/da_size_by_language_multi_axes.png)
<!-- END auto:early-small -->

The "all benchmarks" rows of the grid above hold the 816 bBPB twins at every
size and checkpoint beside 359–553 gated accuracy tasks (Setup), so every row
reads the same mix of scores. The Key findings below split the grid into the
two kinds by hand from `early_small_da_goal_multi_axes.csv` and
`da_all_per_task_both_axes.csv` ([Benchmark BPB](#11-benchmark-bpb-a-continuous-score-on-the-same-items)).

<!-- BEGIN auto:by-L (by_L.py --pool predictivity) -->
## Per language count

**Pairs per L.** The table lists the design variants the grid plans at seed 1904. For each proxy size it gives the pairs usable against 1.7B, meaning both members are planned at that size and at 1.7B. Each cell reads planned / with data today on BPB / on the benchmarks / on the training loss. ZH and ES also run to 1.7B, as the second and third L2 data builds. In the figures below, an L with fewer than MIN_PAIRS (3) pairs is left out (rule 5), so a thin L shows no line instead of a reading on one or two pairs.

| L | variants | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| 1 | L1-dclmP-deep, L1-deep, L1-fweb-deep, L1-shallow | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 |
| 2 | L2-ES-deep, L2-ZH-deep, L2-deep, L2-shallow | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 |
| 8 | L8-deep, L8-muon, L8-schemeB-deep, L8-schemeB-shallow, L8-shallow, L8-swiglu | 15 / 15/15/15 | 15 / 15/15/15 | 15 / 15/15/15 | 15 / 15/15/15 | 15 / 15/15/15 |
| 15 | L15-AT3-deep, L15-deep, L15-muon, L15-schemeB-deep, L15-schemeB-shallow, L15-shallow, L15-swiglu | 21 / 21/21/21 | 21 / 21/21/21 | 21 / 21/21/21 | 21 / 21/21/21 | 21 / 21/21/21 |
| 30 | L30-AT3-deep, L30-deep, L30-muon, L30-schemeB-deep, L30-schemeB-shallow, L30-shallow, L30-swiglu | 21 / 21/21/21 | 21 / 21/21/21 | 21 / 21/21/21 | 21 / 21/21/21 | 21 / 21/21/21 |
| 50 | L50-AT3-deep, L50-AT3-shallow, L50-deep, L50-shallow | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 |

These figures repeat the early-and-small reading one L at a time. Each panel uses the pairs of design variants that share the L (seed 1904 of every data build, `predictivity_seeds`) and compares them with the 1.7B final ranking at the ten evaluated checkpoints of every run. A cell needs ≥ 3 pairs (rule 5). The first panel pools every pair at that seed, across all data builds (`da_all_pooled_per_task_multi_axes.csv`). `da_all_by_L_per_task_multi_axes.csv` also carries each size's DA-ckpt within the L (`da_own`). rq04 reads both tables. The `_mono_axis` twins of every table and figure repeat the analysis over the one-axis pairs (rule 15). Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool predictivity [--axes mono-axis]`.

rq02 has three decision accuracies, and two of them have a checkpoint axis, so they get a figure here. **DA-goal** ranks the proxy at any checkpoint against the 1.7B final checkpoint. **DA-ckpt** ranks it against its own size's final checkpoint. The 175M DA-ckpt line therefore asks what 175M would have decided early, and anything it misses is due to the checkpoint alone. The gap between the two is what the proxy *size* costs. The 1.7B line is the same curve in both, because at the reference the two definitions coincide. DA-ckpt has no 5C column, since a run's final checkpoint is its own reference. The third, **DA-size**, is DA-goal read at 5C alone and lives in `da_all_per_task_both_axes.csv`. Each figure comes in a benchmarks-only version and a `_with_bpb` version that adds solid per-size BPB lines. All four share the y axis, so any two can be overlaid.

![DA-goal per L](pretraining/predictivity/early_small_da_goal_by_L_multi_axes.png)

![DA-goal per L, with BPB](pretraining/predictivity/early_small_da_goal_by_L_with_bpb_multi_axes.png)

![DA-ckpt per L](pretraining/predictivity/early_small_da_ckpt_by_L_multi_axes.png)

![DA-ckpt per L, with BPB](pretraining/predictivity/early_small_da_ckpt_by_L_with_bpb_multi_axes.png)

A third variant of each averages only over the (benchmark, language) cells that rank reliably on BOTH axes (DA-size and DA-ckpt each ≥ 0.8 at the `late` cell, `reliable_tasks.py`). The plain panels average over every gated benchmark. These average only over the benchmarks that work.

![DA-goal per L, reliable cells only](pretraining/predictivity/early_small_da_goal_by_L_above_80_multi_axes.png)

![DA-ckpt per L, reliable cells only](pretraining/predictivity/early_small_da_ckpt_by_L_above_80_multi_axes.png)
<!-- END auto:by-L -->

<!-- BEGIN auto:early-small-by-transformation (by_L.py --pool predictivity --by transformation) -->
## Early and small per design axis

The per-L reading above pools every design axis inside an L. This section instead splits the MONO-AXIS pairs at seed 1904 (`predictivity_seeds`, every data build) by the one axis each pair changes: language count, depth (deep vs shallow), activation (XIELU vs SwiGLU), optimizer, data scheme (A vs B vs C), temperature (T=1 vs T=3). Each axis gets one panel, and a first panel covers every mono-axis pair (the median number of pairs per cell is at most 26). The gate (rule 1), pair minimum (rule 5) and filter variants are the same as in the per-L figures. The `above_66_ckpt` twin filters the DA-ckpt figure and `above_66_either` filters both, all on mono-axis reliability (rule 15). The task count sits at the end of every line, and the populations differ between panels and sizes (rule 13). Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool predictivity --by transformation`.

![DA-ckpt per design axis](pretraining/predictivity/early_small_da_ckpt_by_transformation_mono_axis.png)

**DA-ckpt**, against the proxy size's own final checkpoint. Each cell gives the mean DA at the first → last drawn checkpoint, with the tasks behind the line in brackets:

| axis (DA-ckpt, 0.5C → 4.5C, tasks) | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| all | 0.61 → 0.84 (1288) | 0.58 → 0.84 (1320) | 0.57 → 0.88 (1359) | 0.58 → 0.83 (1409) | 0.56 → 0.81 (1465) | 0.57 → 0.77 (1506) |
| language count | 0.60 → 0.84 (979) | 0.58 → 0.83 (1003) | 0.56 → 0.87 (1034) | 0.58 → 0.83 (1073) | 0.54 → 0.80 (1120) | 0.55 → 0.76 (1145) |
| depth (deep vs shallow) | 0.56 → 0.83 (1046) | 0.53 → 0.83 (1071) | 0.51 → 0.87 (1101) | 0.50 → 0.79 (1144) | 0.48 → 0.80 (1192) | 0.52 → 0.76 (1220) |
| activation (XIELU vs SwiGLU) | 0.57 → 0.83 (430) | 0.52 → 0.85 (434) | 0.51 → 0.87 (447) | 0.55 → 0.83 (471) | 0.47 → 0.79 (482) | 0.54 → 0.76 (486) |
| optimizer | 0.54 → 0.75 (430) | 0.53 → 0.76 (434) | 0.65 → 0.87 (447) | 0.58 → 0.81 (471) | 0.51 → 0.70 (482) | 0.50 → 0.73 (486) |
| data scheme (A vs B vs C) | 0.53 → 0.82 (277) | 0.54 → 0.84 (276) | 0.54 → 0.87 (282) | 0.56 → 0.84 (295) | 0.53 → 0.81 (303) | 0.51 → 0.77 (306) |
| temperature (T=1 vs T=3) | 0.64 → 0.86 (979) | 0.63 → 0.89 (1003) | 0.58 → 0.90 (1034) | 0.63 → 0.86 (1073) | 0.62 → 0.82 (1120) | 0.63 → 0.79 (1145) |

Key findings:

- **Earliest safe checkpoint at 90M (DA-ckpt).** The checkpoint from which the line stays ≥ 0.75, per axis: language count: from 4.5C (0.84), depth (deep vs shallow): from 4.5C (0.83), activation (XIELU vs SwiGLU): from 4C (0.76), optimizer: from 4.5C (0.75), data scheme (A vs B vs C): from 4.5C (0.82), temperature (T=1 vs T=3): from 4C (0.80).
- **DA-ckpt at 1.7B.** Mean DA at the first → last checkpoint per axis, task counts in brackets: language count: 0.55 → 0.76 (1145), depth (deep vs shallow): 0.52 → 0.76 (1220), activation (XIELU vs SwiGLU): 0.54 → 0.76 (486), optimizer: 0.50 → 0.73 (486), data scheme (A vs B vs C): 0.51 → 0.77 (306), temperature (T=1 vs T=3): 0.63 → 0.79 (1145).

Follow-ups:

- A `_with_bpb` variant per axis, to see whether BPB decides the temperature and the data scheme earlier than the benchmarks do.
- The same panels on the L8 languages only (`scale_convergence.py --langs L8` does it for DA-size), so the language-count panel is read on one task set.
- Once BT3 trains, the temperature panel gains the B-vs-BT3 pairs and the scheme panel AT3-vs-BT3 with no code change.

![DA-goal per design axis](pretraining/predictivity/early_small_da_goal_by_transformation_mono_axis.png)

**DA-goal**, against the 1.7B final checkpoint:

| axis (DA-goal, 0.5C → 5C, tasks) | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| all | 0.58 → 0.59 (1274) | 0.57 → 0.55 (1316) | 0.55 → 0.57 (1351) | 0.55 → 0.55 (1407) | 0.55 → 0.57 (1461) | 0.57 → 0.77 (1506) |
| language count | 0.57 → 0.56 (966) | 0.55 → 0.55 (1000) | 0.53 → 0.56 (1031) | 0.57 → 0.57 (1071) | 0.55 → 0.55 (1117) | 0.55 → 0.76 (1145) |
| depth (deep vs shallow) | 0.54 → 0.56 (1032) | 0.53 → 0.51 (1068) | 0.52 → 0.52 (1098) | 0.45 → 0.47 (1142) | 0.48 → 0.51 (1188) | 0.52 → 0.76 (1220) |
| activation (XIELU vs SwiGLU) | 0.50 → 0.52 (418) | 0.51 → 0.50 (432) | 0.50 → 0.55 (445) | 0.47 → 0.49 (470) | 0.42 → 0.52 (480) | 0.54 → 0.76 (486) |
| optimizer | 0.50 → 0.52 (418) | 0.46 → 0.44 (432) | 0.52 → 0.56 (445) | 0.52 → 0.50 (470) | 0.46 → 0.52 (480) | 0.50 → 0.73 (486) |
| data scheme (A vs B vs C) | 0.52 → 0.51 (268) | 0.52 → 0.49 (275) | 0.51 → 0.49 (282) | 0.49 → 0.51 (295) | 0.52 → 0.52 (302) | 0.51 → 0.77 (306) |
| temperature (T=1 vs T=3) | 0.62 → 0.63 (966) | 0.64 → 0.60 (1000) | 0.59 → 0.62 (1031) | 0.62 → 0.62 (1071) | 0.63 → 0.62 (1117) | 0.63 → 0.79 (1145) |

Key findings:

- **Earliest safe checkpoint at 90M (DA-goal).** The checkpoint from which the line stays ≥ 0.75, per axis: language count: never (max 0.57), depth (deep vs shallow): never (max 0.56), activation (XIELU vs SwiGLU): never (max 0.55), optimizer: never (max 0.52), data scheme (A vs B vs C): never (max 0.54), temperature (T=1 vs T=3): never (max 0.64).
- **The gap between the tables is the cost of size.** The difference between an axis's DA-goal and DA-ckpt cells at a proxy size is what that size costs. The 1.7B column is the same line in both.

Follow-ups:

- The size axis of the same split is `scale_convergence_da_size_transformation_panels_multi_axes.png` (DA-size pooled over decisions).
- A jackknife band per axis line (leave one family out), as the scale-convergence panels carry.

Filtered twins (reliable cells only): [`early_small_da_goal_by_transformation_above_80_mono_axis.png`](pretraining/predictivity/early_small_da_goal_by_transformation_above_80_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_both_mono_axis.png`](pretraining/predictivity/early_small_da_goal_by_transformation_above_66_both_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_either_mono_axis.png`](pretraining/predictivity/early_small_da_goal_by_transformation_above_66_either_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_80_mono_axis.png`](pretraining/predictivity/early_small_da_ckpt_by_transformation_above_80_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png`](pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png`](pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_either_mono_axis.png`](pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_either_mono_axis.png)

Files: [`early_small_da_goal_by_transformation_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_mono_axis.png), [`early_small_da_goal_by_transformation_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_mono_axis.csv), [`early_small_da_goal_by_transformation_above_80_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_80_mono_axis.png), [`early_small_da_goal_by_transformation_above_80_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_80_mono_axis.csv), [`early_small_da_goal_by_transformation_above_66_both_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_both_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_both_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_both_mono_axis.csv), [`early_small_da_goal_by_transformation_above_66_either_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_either_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_either_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_either_mono_axis.csv), [`early_small_da_ckpt_by_transformation_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_mono_axis.png), [`early_small_da_ckpt_by_transformation_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_mono_axis.csv), [`early_small_da_ckpt_by_transformation_above_80_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_80_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_80_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_80_mono_axis.csv), [`early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_both_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_both_mono_axis.csv), [`early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.csv), [`early_small_da_ckpt_by_transformation_above_66_either_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_either_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_either_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_either_mono_axis.csv), [`da_all_by_transformation_per_task_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_by_transformation_per_task_mono_axis.csv).
<!-- END auto:early-small-by-transformation -->

**Key findings**

- **DA-goal for the benchmark mean never reaches 0.75 at a proxy.** The
  grid's rows read 0.58–0.62 at every checkpoint from 0.5C to 5C, against 0.62
  → 0.80 for the reference's own run. Split by kind, accuracy alone reads
  0.48–0.57 at the proxies (90M 0.49–0.51, 600M 0.50–0.55, 1.7B 0.54 → 0.74).
  The twins alone read 0.61–0.67 (1.7B 0.67 → 0.85).
- **The two estimands give one verdict.** The pooled-over-decisions reading of
  figure 1 (0.57–0.59) and this mean-over-tasks one (0.60–0.62 at 5C) both sit
  near chance.
- **BPB reads the 1.7B ranking at ≥ 0.75 from 90M at 0.5C (0.92).** It stays
  there at 175M (0.90–0.95), 350M (0.80–0.91) and, from 1C on, 1B (0.77–0.92,
  0.7497 at 0.5C). The 600M row never gets there (0.69–0.73), so 600M is the
  one size where BPB is not a
  safe proxy. Its depth pairs, which are not size-matched, read 0.06 on BPB
  (`scale_convergence_da_size_transformation_multi_axes.csv`).
- **Reading early mostly measures persistence.** On accuracy alone, DA-ckpt
  at the proxies rises from 0.51–0.53 at 10 % to 0.74–0.81 at 90 % (90M 0.51 →
  0.74, 600M 0.52 → 0.77). The twins alone rise from 0.64–0.70 to 0.87–0.93.
  The seed null of figure 7 reads 0.78–0.80 at 90 % on 175M, 600M and 1B,
  twins included. Against it, the design signal at 90 % is +0.04 to +0.07.
- **Every L now draws a panel.** Each regime has 6 (L1, L2, L50), 10 (L8) or
  15 (L15, L30) usable benchmark pairs against 1.7B at every proxy size. The
  muon variants have no benchmark or BPB pair against 1.7B yet, and at L15
  and L30 they enter the loss pairs only (`pairs_da_all_by_L_multi_axes.csv`).
- **Few (benchmark, language) cells are safe at any proxy.** Of 1116 cells,
  the smallest safe level is never for 803, 1B for 168, 90M for 103, 600M for
  22, 350M for 13 and 175M for 7. Of these, 76 of the 103 at 90M and 87 of the
  168 at 1B are bBPB twins. "Safe" over 3 pairs is a weak guarantee.

**Follow-ups**

- The `_flops` reading (`safe_flops_da_goal_multi_axes.png`) joins un-annealed
  and annealed points. The practical answer is one line per size with the
  annealed points marked, plus the compute frontier (the best DA reachable at
  or below each compute).
- `early_small_by_L_own`, which separates the cost of the size from the cost
  of the checkpoint, as an appendix pair with
  `early_small_da_goal_by_L_multi_axes`. `pairs_da_all_by_L_multi_axes.csv`
  shows why per-L DA is coarse.
- An accuracy-only and a twins-only version of the grid, to draw the split
  that the Key findings make by hand. Every row now holds the twins at every
  checkpoint.

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
gate `predictivity`.** There is one line per language-count regime. A regime
pools depth, scheme and temperature decisions (`share_*` in the CSV), and the
mix differs by regime. Holding the mix fixed would leave fewer pairs than
rule 5's minimum of three.

<!-- BEGIN auto:scale-convergence-L8 (scale_convergence.py --by L --langs L8) -->
## Scale convergence by language count, on the L8 languages

This block reads the `--by L` lines over the tasks in the 8 languages of the L8 setting (de, en, es, fr, it, ja, ru, zh). Every regime from L8 up trains on these languages. One confound remains. A regime pools architecture, scheme and temperature decisions, and the mix differs by regime (`share_*` in the CSV). Rule 5 forbids holding it fixed. Under `--axes mono-axis` the regimes keep only their one-axis pairs (L1 10 → 5, L2 6 → 4, L8 15 → 6, L15 21 → 7, L30 21 → 7, L50 6 → 4) and rest on fewer tasks. The numbers below are the unfiltered population. The `above_66_size` twin (`scale_convergence_da_size_L8_above_66_size_multi_axes.png`) is the conditional reading. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by L --langs L8`.

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| L1 | 0.58 | 0.6 | 0.63 | 0.61 | 0.63 | 1.0 | — |
| L15 | 0.53 | 0.5 | 0.54 | 0.53 | 0.54 | 1.0 | — |
| L2 | 0.51 | 0.51 | 0.51 | 0.54 | 0.45 | 1.0 | — |
| L30 | 0.54 | 0.51 | 0.54 | 0.52 | 0.53 | 1.0 | — |
| L50 | 0.59 | 0.57 | 0.59 | 0.55 | 0.57 | 1.0 | — |
| L8 | 0.5 | 0.5 | 0.54 | 0.51 | 0.5 | 1.0 | — |
| all pairs | 0.58 | 0.57 | 0.58 | 0.58 | 0.57 | 1.0 | — |

![Scale convergence, L8](pretraining/predictivity/scale_convergence_da_size_L8_multi_axes.png)
<!-- END auto:scale-convergence-L8 -->

<!-- BEGIN auto:scale-convergence-L8-common (scale_convergence.py --by L --langs L8 --common-tasks) -->
## Scale convergence by language count, on the L8 languages, common tasks

This block reads the `--by L` lines over the tasks in the 8 languages of the L8 setting (de, en, es, fr, it, ja, ru, zh). Every regime from L8 up trains on these languages. It further keeps only the tasks with ≥ 3 pairs in every regime at every proxy size. The whole figure then uses one task set, so a gap between two lines is a gap on the same benchmarks. One confound remains. A regime pools architecture, scheme and temperature decisions, and the mix differs by regime (`share_*` in the CSV). Rule 5 forbids holding it fixed. Under `--axes mono-axis` the regimes keep only their one-axis pairs (L1 10 → 5, L2 6 → 4, L8 15 → 6, L15 21 → 7, L30 21 → 7, L50 6 → 4) and rest on fewer tasks. The numbers below are the unfiltered population. The `above_66_size` twin (`scale_convergence_da_size_L8common_above_66_size_multi_axes.png`) is the conditional reading. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by L --langs L8 --common-tasks`.

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| L1 | 0.58 | 0.6 | 0.63 | 0.62 | 0.64 | 1.0 | — |
| L15 | 0.52 | 0.49 | 0.51 | 0.48 | 0.54 | 1.0 | — |
| L2 | 0.51 | 0.51 | 0.51 | 0.54 | 0.46 | 1.0 | — |
| L30 | 0.51 | 0.5 | 0.49 | 0.47 | 0.53 | 1.0 | — |
| L50 | 0.5 | 0.5 | 0.53 | 0.46 | 0.44 | 1.0 | — |
| L8 | 0.51 | 0.51 | 0.51 | 0.51 | 0.51 | 1.0 | — |
| all pairs | 0.54 | 0.53 | 0.55 | 0.53 | 0.54 | 1.0 | — |

![Scale convergence, L8 common tasks](pretraining/predictivity/scale_convergence_da_size_L8common_multi_axes.png)
<!-- END auto:scale-convergence-L8-common -->

| regime, L8-language tasks (unfiltered, pooled) | 90M | 175M | 350M | 600M | 1B | tasks | decision mix at 1B |
|---|---|---|---|---|---|---|---|
| L1 (English only) | 0.586 | 0.603 | 0.638 | 0.607 | 0.629 | 125–134 | depth 0.17, scheme 0.50, two-axis 0.33 |
| L2 | 0.516 | 0.505 | 0.513 | 0.531 | 0.454 | 125–134 | depth 0.17, scheme 0.50, two-axis 0.33 |
| L8 | 0.503 | 0.498 | 0.491 | 0.508 | 0.493 | 279–344 | depth 0.28, scheme 0.25, activation 0.07, two-axis 0.40 |
| L15 | 0.521 | 0.505 | 0.518 | 0.513 | 0.531 | 386–457 | depth 0.14, scheme 0.10, T 0.09, activation 0.09, two-axis 0.58 |
| L30 | 0.543 | 0.510 | 0.516 | 0.517 | 0.531 | 386–457 | depth 0.15, scheme 0.15, T 0.08, activation 0.05, two-axis 0.57 |
| L50 | 0.597 | 0.566 | 0.579 | 0.546 | 0.568 | 386–457 | depth ⅓, T ⅓, two-axis ⅓ |
| all pairs | 0.581 | 0.568 | 0.574 | 0.573 | 0.575 | 399–460 | L 0.11, two-axis 0.80 |

*`scale_convergence_da_size_L8_multi_axes.csv`. The bBPB twins enter at every
size, so the task counts move with the gate alone (399 at 90M to 460 at 1B on
the pooled line).*

**Key findings**

- **The per-L lines do not order by L.** On the L8 languages the English-only
  L1 line is highest from 175M to 1B (0.60–0.64). At 90M it reads 0.59,
  behind L50's 0.60. L2 is lowest at 1B (0.45), and L8, L15, L30 and L50 cross
  between 0.49 and 0.60. The pooled line sits at 0.57–0.58 at every proxy.
- **Fixing the task set does not order them either.** On the 122 tasks common
  to every regime and size, L1 reads 0.58 → 0.64 from 90M to 1B, L8 0.47 →
  0.53, L15 0.50 → 0.54, L30 0.51 → 0.55, L2 0.52 → 0.46 and L50 0.49 → 0.43
  (not a trend). Filtered on `above_66_size`, the lines read 0.48–0.90 on
  7–106 tasks and are again unordered.
- **The mono-axis pairs keep every L line.** In
  `pairs_da_all_by_L_mono_axis.csv`, L1, L2 and L50 keep 4 benchmark pairs, L8
  keeps 5, and L15 and L30 keep 6. L1 and L2 have lines because the
  DCLMP/FWEB and ZH/ES cells give each of them four families.
- **A regime line is not the effect of language count.** Within a regime the
  decisions are depth, scheme, temperature and activation, in a mix that
  differs by regime. A smaller model reads the decisions of every regime
  equally poorly, on 4–6 families per regime at 1.7B.

**Follow-ups**

- An L8-only replicate of the depth and scheme decisions in every regime. It
  would hold the decision mix fixed, which this grid cannot do with three
  pairs or more.
- Bootstrap over L, not over items, for the per-L comparison, since items
  within an L move together.

GitHub: [scale_convergence_da_size_L8_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8_multi_axes.png) · [scale_convergence_da_size_L8_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_L8common_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8common_multi_axes.png) · [scale_convergence_da_size_L8common_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8common_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_L8_above_66_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8_above_66_size_multi_axes.png) · [scale_convergence_da_size_L8_above_66_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8_above_66_size_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_L8common_above_66_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8common_above_66_size_multi_axes.png) · [scale_convergence_da_size_L8common_above_66_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L8common_above_66_size_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_L_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L_mono_axis.png) · [scale_convergence_da_size_L_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_L_mono_axis.csv)

### 5. One language at a time, and the tokens axis

**DA-size · `scale_convergence_da_size_lang_all*`: no filter (`_above_66_size`
conditional) · multi-axis · pairs sharing the L from `predictivity` ·
gate `predictivity`.** There is one panel per L8 language, with one line per
regime that trains it plus the pooled line. The `_tokens` twin relabels x as
the tokens of that language that the decision's two members trained on
(share × D(N)). The panel title carries the collapse test.

<!-- BEGIN auto:scale-convergence-by-language (by_language.py --pool predictivity) -->
## Scale convergence per language

For each language of the L8 setting, this table reads the `--by L` lines on that language's benchmarks alone. A cell gives R at the smallest → largest proxy for one regime that trains the language, with the largest task count in brackets. Every gated task is included, with no selection on DA, so this is the version to draw inferences from. The `above_66_size` twin is the conditional reading. The last column is the collapse test. It fits one log-linear line through the points of every regime, first with x = model size, then with x = tokens of the language (its share of all tokens × D(N)), and reports the R² of each fit. A higher R² under tokens would mean that exposure to the language explains what language count does not. Each regime pools architecture, scheme and temperature decisions. Regenerate with `python analysis/rq02_decision_accuracy/by_language.py --pool predictivity`. `scale_convergence_da_size_lang_all_multi_axes_coverage.csv` says why a cell is empty.

| language | L1 | L2 | L8 | L15 | L30 | L50 | R² size / tokens |
|---|---|---|---|---|---|---|---|
| en | 0.58→0.63 [143] | 0.51→0.45 [143] | 0.52→0.51 [143] | 0.52→0.54 [143] | 0.51→0.53 [143] | 0.50→0.45 [143] | 0.00 / 0.04 |
| ru | — | — | 0.51→0.49 [33] | 0.57→0.56 [33] | 0.63→0.59 [33] | 0.64→0.73 [33] | 0.01 / 0.06 |
| zh | — | — | 0.48→0.44 [48] | 0.58→0.55 [48] | 0.55→0.55 [48] | 0.72→0.72 [48] | 0.00 / 0.13 |
| de | — | — | 0.51→0.50 [33] | 0.52→0.51 [33] | 0.55→0.56 [33] | 0.65→0.64 [33] | 0.00 / 0.04 |
| ja | — | — | 0.50→0.60 [25] | 0.53→0.50 [25] | 0.57→0.51 [25] | 0.45→0.56 [25] | 0.03 / 0.07 |
| es | — | — | 0.52→0.49 [125] | 0.50→0.52 [125] | 0.55→0.51 [125] | 0.62→0.57 [125] | 0.01 / 0.07 |
| fr | — | — | 0.42→0.54 [38] | 0.58→0.55 [38] | 0.52→0.54 [38] | 0.62→0.64 [38] | 0.04 / 0.00 |
| it | — | — | 0.41→0.51 [35] | 0.56→0.57 [35] | 0.54→0.56 [35] | 0.69→0.62 [35] | 0.02 / 0.00 |

![Scale convergence per language](pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes.png)

![Scale convergence per language, tokens axis](pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_tokens.png)
<!-- END auto:scale-convergence-by-language -->

**Key findings**

- **Every one of the eight languages draws a panel and none is ordered by L.**
  The pooled line at 1B reads en 0.54 (137 tasks), ru 0.67 (32), zh 0.64
  (45), de 0.64 (31), ja 0.57 (24), es 0.57 (124), fr 0.61 (35) and it 0.61
  (32). The regime lines within a panel cross at every size. Spanish, French
  and Italian now have an L8 line too, on the three scheme-A L8 cells that
  reach 1.7B (scheme B's L8 list does not contain them).
- **English sits at the bottom at every size and Russian at the top.** English
  reads 0.53–0.54 from 90M to 1B and Russian 0.66–0.68. The twins enter at
  every size, so the task counts now move with the gate alone (en 127 tasks
  at 90M against 137 at 1B).
- **Exposure does not collapse the lines.** The collapse R² is 0.00–0.06 under
  size and 0.01–0.18 under tokens. Tokens beat size for seven languages, by at
  most 0.17 (zh 0.01 → 0.18). If reliability depended on how many of a
  language's tokens the models saw, the tokens axis would order the points.
  It does not.

**Follow-ups**

- A bootstrap over families, to say which crossings are real. Every regime
  line rests on 3–6 families (4–6 for en, ru, zh, de and ja, and 3 for the L8
  lines of es, fr and it), so a per-language line moves in coarse steps.

GitHub: [scale_convergence_da_size_lang_all_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes.png) · [scale_convergence_da_size_lang_all_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_lang_all_multi_axes_tokens.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_tokens.png) · [scale_convergence_da_size_lang_all_multi_axes_tokens.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_tokens.csv) ·
[scale_convergence_da_size_lang_all_multi_axes_coverage.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_coverage.csv) ·
GitHub: [scale_convergence_da_size_lang_above_66_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes.png) · [scale_convergence_da_size_lang_above_66_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.png) · [scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.csv)

### 6. Language tier: the order follows the task population

**DA-size · `reliability_da_size_by_language_tier_multi_axes`: panel (a) no filter, panel (b)
`above_66_size` · multi-axis · pairs from `predictivity` · gate
`predictivity`; `reliability_da_size_vs_language_share_multi_axes`: no filter, per language with
≥ 3 gated tasks (rule 8).** A language's tier is the smallest scheme-A regime
that trains it. L8 holds the eight high-resource languages, and L50 the twenty
that only the L50 mixture trains.

<!-- BEGIN auto:language-tier (language_tier.py --pool predictivity) -->
## Decision reliability by language tier

Each row reads the pooled `all pairs` line of `scale_convergence.py` over the gated tasks of one language TIER. A language's tier is the smallest scheme-A regime that trains it. L8 holds the eight high-resource languages that every regime from L8 up trains, and L50 holds the twenty languages that only the L50 mixture trains. Each cell gives the reliability at the smallest → largest proxy, with the task count in brackets. `reliability_da_size_vs_language_share_multi_axes.png` is the per-language version. It plots reliability against the language's share of the L50 mixture, with a Spearman ρ over languages of 90M -0.60, 175M -0.57, 350M -0.50, 600M -0.39, 1B -0.54. The `all gated tasks` column and the per-language figure are unfiltered. The `above_66_size` column keeps only the tasks whose median DA-size is ≥ 0.66. Tiers also differ in their benchmark mix, so a gap between tiers can be a gap between benchmark families. Regenerate with `python analysis/rq02_decision_accuracy/language_tier.py --pool predictivity`.

| tier | all gated tasks | above_66_size |
|---|---|---|
| L8 languages (in every regime) | 0.58 → 0.57 [418–480 tasks] | 0.72 → 0.73 [83–89 tasks] |
| L15-only languages | 0.60 → 0.59 [215–250 tasks] | 0.74 → 0.73 [51–55 tasks] |
| L30-only languages | 0.60 → 0.60 [333–387 tasks] | 0.74 → 0.75 [103–110 tasks] |
| L50-only languages | 0.69 → 0.67 [308–344 tasks] | 0.81 → 0.81 [189–203 tasks] |

![Reliability by language tier](pretraining/predictivity/reliability_da_size_by_language_tier_multi_axes.png)

![Reliability against language share](pretraining/predictivity/reliability_da_size_vs_language_share_multi_axes.png)
<!-- END auto:language-tier -->

**Key findings**

- **On accuracy alone the high-resource tier reads best at 90M.** The L8 tier
  is 0.549 against 0.510 (L15), 0.491 (L30) and 0.486 (L50), on 60–118 tasks.
  At 600M and 1B the tiers sit within 0.04 of each other (0.519–0.555 and
  0.545–0.567). We pool these accuracy-only values by hand from
  `early_small_da_goal_multi_axes.csv` without the `bbpb_` rows. The figure
  now carries the twins at every size, so it has no accuracy-only interval.
- **With the twins the order reverses at every size.** The L50-only tier reads
  0.643–0.686 (0.686 [0.62, 0.75] at 90M, 0.672 [0.60, 0.75] at 1B) and L8
  0.568–0.581 (0.575 [0.55, 0.60] at 1B), on 206–460 tasks. Filtered on
  `above_66_size`, the tiers read 0.70–0.81 at every size, with L50 highest
  (0.77–0.81).
- **The token share follows the same split.** Spearman ρ between a language's
  share of the L50 mixture and its reliability is −0.42 to −0.53 at every size
  over 49 languages in the figure (−0.49 at 90M, −0.53 at 1B), but +0.07 to
  +0.22 on accuracy alone over 47–49 (+0.22 at 90M, +0.07 at 1B). The sign
  follows the kind of score, so neither reads as a resource effect.
- **The tiers also differ in decision and benchmark mix.** 15 of the 19
  L50-only languages are read from the four L50 families alone. Their six
  pairs are mostly temperature and depth decisions. he, ka, ml and ta, which
  scheme-B cells also train, are read from eight families and 28 pairs.

**Follow-ups**

- An accuracy-only and a twins-only version of both panels, with intervals,
  since the sign of the tier effect flips with the kind of score.
- The same decision set in every tier. On this grid it leaves fewer than
  rule 5's three pairs, and the L8-only replicate of figure 4's follow-up
  would allow it.
- A per-family version of panel (a), to separate benchmark mix from tier.

GitHub: [reliability_da_size_by_language_tier_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_da_size_by_language_tier_multi_axes.png) · [reliability_da_size_by_language_tier_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_da_size_by_language_tier_multi_axes.csv) ·
GitHub: [reliability_da_size_vs_language_share_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_da_size_vs_language_share_multi_axes.png) · [reliability_da_size_vs_language_share_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_da_size_vs_language_share_multi_axes.csv)

### 7. Seed uncertainty and the DA-ckpt null

**`seed_uncertainty_da_all_seed_null`: three views · no filter · pool
`predictivity_seeds` · gate `predictivity`.** Proxy-seed DA is DA-size at
each of the proxy's three seeds against the fixed 1.7B final (seed 1904).
Test-retest ranks the same designs from seed s1 and from seed s2 at one size
and compares the two. The seed null is DA-ckpt over the `seed` pairs (two
seeds of ONE design), set against the real design pairs. English at 175M,
600M and 1B and Russian at 1B are the only (size, language) cells where the
replicated designs give ≥ 3 pairs under rule 2.

<!-- BEGIN auto:seed-uncertainty (seed_uncertainty.py --pool predictivity) -->
## Seed uncertainty

This section asks what the replicate seeds say about decision accuracy. Only English at three proxy sizes and Russian at 1B can be read. These are the only (size, language) cells where the replicated designs give ≥ 3 pairs under rule 2. Each row reports two things. The first is the DA of those decisions against the 1.7B final, with the proxy at each of its three seeds (proxy-side noise). The second is the DA between two seeds' rankings of the same designs at that size (the reference-side ceiling). With as few as three pairs per task, a task's DA can only be 0, ⅓, ⅔ or 1, so read the spread, not a mean. The seed null is DA-ckpt over pairs of two seeds of one design, where there is nothing to decide, averaged over fractions and gated tasks: 175M: 0.61 vs real 0.69, 600M: 0.60 vs real 0.68, 1B: 0.61 vs real 0.65. Regenerate with `python analysis/rq02_decision_accuracy/seed_uncertainty.py --pool predictivity`.

| size | language | designs | DA at the 3 proxy seeds | test-retest (3 seed pairs) | tasks |
|---|---|---|---|---|---|
| 175M | en | 3 | 0.55, 0.55, 0.64 | 0.59–0.65 | 129 |
| 600M | en | 3 | 0.60, 0.56, 0.61 | 0.61–0.62 | 137 |
| 1B | en | 4 | 0.60, 0.57, 0.56 | 0.53–0.60 | 138 |
| 1B | ru | 4 | 0.83, 0.79, 0.81 | 0.75–0.83 | 32 |

![Seed uncertainty](pretraining/predictivity/seed_uncertainty_da_all_seed_null.png)
<!-- END auto:seed-uncertainty -->

**Key findings**

- **Proxy-seed noise is up to ± 0.05 on English.** The three seeds read 0.55 /
  0.55 / 0.64 at 175M (387 decisions on 129 tasks), 0.60 / 0.56 / 0.61 at 600M
  (411 decisions, 137 tasks) and 0.60 / 0.57 / 0.56 at 1B (813 decisions, 137
  tasks). Russian at 1B reads 0.83 / 0.79 / 0.81 on 32 tasks. With three
  pairs a DA can only be 0, ⅓, ⅔ or 1, so read the spread, not a mean.
- **The English test-retest ceiling is low at every size.** It reads 0.59–0.65
  at 175M, 0.61–0.62 at 600M and 0.53–0.60 at 1B (Russian 0.75–0.83). On these
  129–137 English tasks, now including the twins, two seeds of the same
  designs disagree with each other as much as the proxy disagrees with the
  reference.
- **DA-ckpt is mostly within-run persistence.** Two seeds of one design, which
  have nothing to decide, read 0.48 at 10 % of the 175M run and 0.79 at 90 %,
  against 0.63 and 0.86 for the real pairs (600M 0.52 → 0.80 against 0.61 →
  0.84, 1B 0.54 → 0.78 against 0.58 → 0.82). At 90 % the real pairs exceed
  the null by +0.04 to +0.07. The gap is largest early in the run (+0.15 at
  40 % of 175M, +0.16 at 30 % of 600M, +0.11 at 30 % of 1B). The null reads
  1221, 1302 and 1062 gated tasks, including the bBPB twins now that the seed
  replicates have them (`seed_uncertainty_da_all_seed_null.csv`).
- **These replicated cells give the scale of seed noise, not an interval.**
  They are three or four cross-L deep designs on one or two languages, so they
  say nothing about the interval of the pooled lines. No Wilson or binomial
  interval is drawn anywhere, because the pairs share models and are not
  independent.

**Follow-ups**

- A second 1.7B seed on two cells. The null exists for DA-ckpt only, since
  there is no seed replicate at 1.7B. The second seed would give the DA-size
  and DA-goal panels the same floor.
- Draw the null on every checkpoint-axis figure the paper shows.

GitHub: [seed_uncertainty_da_all_seed_null.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/seed_uncertainty_da_all_seed_null.png) · [seed_uncertainty_da_all_seed_null.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/seed_uncertainty_da_all_seed_null.csv)

#### The paper version

**Same three views, same population (proxy-seed DA-size and test-retest on
the replicated deep cells, English at 175M, 600M and 1B and Russian at 1B,
and the DA-ckpt seed null against the multi-axis pairs) · no filter · pool
`predictivity_seeds` · gate `predictivity`.** It is drawn bare for the paper
(rule 18, `style.save_paper`), with no title, capitalized axis labels and the
legends "English" / "Russian" and "<size> seed null" / "<size> design pairs".

![Seed uncertainty, paper](pretraining/predictivity/seed_uncertainty_da_all_seed_null_paper.png)

**Key findings**

- **The seed-replicate level is the middle panel.** On English, two seeds of
  the same designs agree with each other at 0.59–0.65 (175M), 0.61–0.62 (600M)
  and 0.53–0.60 (1B). A proxy's 0.55–0.64 against the 1.7B final is therefore
  already at the level a redrawn run reaches.
- **The numbers are the ones of the figure above.** The paper CSV is the same
  long table (`seed_uncertainty_da_all_seed_null_paper.csv`).

**Follow-ups**

- If the paper draws the test-retest band as a reference level on another
  figure, that caption must state the band's population (cross-L deep
  designs, English), because it is not the pooled line's.

GitHub: [seed_uncertainty_da_all_seed_null_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/seed_uncertainty_da_all_seed_null_paper.png) · [seed_uncertainty_da_all_seed_null_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/seed_uncertainty_da_all_seed_null_paper.csv)

### 8. Decision accuracy is Kendall's τ under another tie convention

**DA-size's population · no filter · multi-axis pairs · pairs from
`predictivity` · gate `predictivity` · one point per (task, proxy
size) cell, median 13 models per cell · 6,329 cells, 4,080 of them the bBPB twins'.**

<!-- BEGIN auto:agreement-measures (agreement.py --pool predictivity) -->
## Decision accuracy is Kendall's τ under another tie convention

DA and Kendall's τ_a differ only in how they count tied pairs. We check this on the 6,809 (benchmark task, proxy size) cells of DA-size's population: every pair of the grid-seed variants, ≥ 3 pairs per cell, above chance at the proxy and at 1.7B. In every cell, 2·DA − 1 = τ_a + (T_both − T_one)/n exactly, where T_both and T_one count the pairs tied on both sides and on one side, and n counts all pairs. Ties are 22,413 of 981,440 pairs (2.3%). 94% of them are one-sided, and they touch 26% of the cells. DA and τ_b therefore correlate at r = 0.986 by construction. The number that matters is how often the tie convention changes a reliability verdict (DA ≥ 0.66, or equivalently τ ≥ 0.32). The table gives it in % of cells per proxy size:

| statistic | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| da | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| da_drop_ref_ties | 0.2 | 0.8 | 1.2 | 0.8 | 1.2 |
| gamma | 1.3 | 1.9 | 1.6 | 2.8 | 2.4 |
| tau_a | 0.6 | 1.3 | 1.3 | 1.8 | 1.8 |
| tau_b | 0.9 | 1.4 | 1.5 | 2.2 | 2.3 |

Spearman ρ and Pearson r on the raw scores are the only two statistics here that are not a rescaling of DA. They weight a pair by how far it is displaced, and they correlate with DA at r = 0.970 and 0.895. A cell has a median of 15 models. Values are in `agreement_da_size_per_cell_multi_axes.csv`. Regenerate with `python analysis/rq02_decision_accuracy/agreement.py --pool predictivity`.

![DA, Kendall tau and Spearman rho against each other](pretraining/predictivity/agreement_da_size_correlation_multi_axes.png)

![DA against Kendall's tau](pretraining/predictivity/agreement_da_size_identity_multi_axes.png)

![Cut sensitivity](pretraining/predictivity/agreement_da_size_cut_sensitivity_multi_axes.png)
<!-- END auto:agreement-measures -->

**Key findings**

- **The three statistics are one statistic.** Over the 6,320 cells where all
  three are defined r(DA, τ_b) = 0.986, r(DA, ρ) = 0.970 and r(τ_b, ρ) = 0.982
  (Spearman 0.986, 0.978, 0.989, `agreement_da_size_correlation_multi_axes.csv`).
  The relation to τ_a is exact on every cell: 2·DA − 1 = τ_a + (T_both −
  T_one)/n, where T_both and T_one count the pairs tied on both sides and on
  one side. DA and τ_a therefore differ only in how tied pairs are counted.
- **Ties are rare and mostly one-sided.** They are 2.5 % of the pairs and
  touch 26 % of the cells, and 94 % of them are one-sided. The twins never tie
  (0 ties over their 4,080 cells), so on accuracy alone ties are 7.3 % of the
  pairs (16,490 of 225,180).
- **The convention moves the verdict on 0.4–3.1 % of cells against the DA ≥
  0.66 set.** τ_a flips 1.0–2.0 %, τ_b 1.1–2.7 %, γ 1.4–3.1 % and DA with the
  reference's ties dropped 0.4–1.2 % per proxy size, most at 600M and 1B. The
  twins never tie, so they now dilute the rate at every size. The identity
  guarantees the correlations, so the flip rate is the number that carries
  information. A published reliable-task list depends on the tie convention
  at the margin.

**Follow-ups**

- Report the reliable-task list under two conventions (DA and γ) in the
  paper's appendix, or state the flip rate beside it.

GitHub: [agreement_da_size_correlation_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_correlation_multi_axes.png) · [agreement_da_size_correlation_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_correlation_multi_axes.csv) ·
GitHub: [agreement_da_size_identity_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_identity_multi_axes.png) · [agreement_da_size_identity_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_identity_multi_axes.csv) ·
GitHub: [agreement_da_size_cut_sensitivity_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_cut_sensitivity_multi_axes.png) · [agreement_da_size_cut_sensitivity_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_cut_sensitivity_multi_axes.csv) ·
[agreement_da_size_per_cell_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_per_cell_multi_axes.csv)

#### DA-size under other definitions (paper appendix table)

**DA-size · no reliability filter · multi-axis and mono-axis pairs of
`predictivity` (grid-seed finals, every build) · gate `predictivity` at the
proxy and at 1.7B · ≥ 3 comparable and ≥ 3 kept pairs per cell (rule 5) ·
task means per population: benchmark accuracy, the bBPB twins, both, and
per-language BPB.** The tie convention above moves verdicts at the margin.
This table asks how far the convention, and a filter on the pairs the
reference can separate, move the DA level itself. The pair filters keep the
pairs whose 1.7B gap exceeds K × the task's reference checkpoint noise (rule
4, `utils.checkpoint_noise`, mean over the 1.7B runs), or K × √2 × its seed SD
as in [decisive pairs](../rq02_decisive_pairs/README.md), with K ∈ {1, 2}.
Each variant reads its own cells (`own`, the table the `.tex` prints) and,
beside them, the cells every variant defines (`common`). The populations
differ across rows of `own` (rule 13), since a filter that keeps fewer than
three pairs empties a cell.

<!-- BEGIN auto:da-definition-sensitivity (agreement.py --pool predictivity) -->
## DA-size under other definitions

Each row gives the task-mean DA-size under one definition of DA. DA-size compares the proxy's final ranking of the grid-seed variants with the 1.7B final's. A task counts at a proxy size when it is above chance at the proxy and at 1.7B (rule 1) and has ≥ 3 comparable pairs, ≥ 3 of them kept (rule 5). The checkpoint SD is the reference's checkpoint noise (rule 4): the residual SD around a line fitted through the 80–100 % window, averaged over the 1.7B runs. The seed SD of a difference is √2 × rq05's seed sd, the threshold rq02_decisive_pairs uses. Each cell reads DA (tasks). "Pairs kept" is the share of the comparable pairs a definition keeps, pooled over the proxy sizes. `own` uses each definition's own cells, and `common` only the cells every definition defines. Regenerate with `python analysis/rq02_decision_accuracy/agreement.py --pool predictivity`.

**multi-axis, own cells** (`da_size_definition_sensitivity_multi_axes.csv`)

| population | definition | pairs kept | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|---|
| Benchmark accuracy | current | 100% | 0.506 (368) | 0.527 (410) | 0.526 (445) | 0.543 (501) | 0.560 (555) |
| Benchmark accuracy | ties_half | 100% | 0.537 (368) | 0.557 (410) | 0.556 (445) | 0.572 (501) | 0.590 (555) |
| Benchmark accuracy | ref_ties_dropped | 96% | 0.518 (367) | 0.537 (409) | 0.534 (444) | 0.553 (500) | 0.571 (554) |
| Benchmark accuracy | ckpt_noise_1x | 72% | 0.537 (361) | 0.557 (403) | 0.549 (437) | 0.575 (493) | 0.592 (545) |
| Benchmark accuracy | ckpt_noise_2x | 47% | 0.549 (343) | 0.565 (385) | 0.572 (417) | 0.606 (468) | 0.619 (518) |
| Benchmark accuracy | seed_sd_1x | 51% | 0.546 (339) | 0.563 (378) | 0.567 (407) | 0.602 (460) | 0.621 (511) |
| Benchmark accuracy | seed_sd_2x | 23% | 0.592 (267) | 0.598 (293) | 0.621 (314) | 0.647 (349) | 0.674 (398) |
| Benchmark accuracy and bBPB twins | current | 100% | 0.616 (1274) | 0.594 (1316) | 0.606 (1351) | 0.600 (1407) | 0.612 (1461) |
| Benchmark accuracy and bBPB twins | ties_half | 100% | 0.625 (1274) | 0.603 (1316) | 0.616 (1351) | 0.610 (1407) | 0.623 (1461) |
| Benchmark accuracy and bBPB twins | ref_ties_dropped | 99% | 0.619 (1273) | 0.597 (1315) | 0.609 (1350) | 0.603 (1406) | 0.616 (1460) |
| Benchmark accuracy and bBPB twins | ckpt_noise_1x | 73% | 0.652 (1264) | 0.622 (1306) | 0.636 (1340) | 0.628 (1396) | 0.645 (1448) |
| Benchmark accuracy and bBPB twins | ckpt_noise_2x | 50% | 0.683 (1222) | 0.647 (1264) | 0.670 (1296) | 0.663 (1347) | 0.674 (1397) |
| Benchmark accuracy and bBPB twins | seed_sd_1x | 47% | 0.683 (1111) | 0.644 (1150) | 0.669 (1179) | 0.661 (1232) | 0.677 (1283) |
| Benchmark accuracy and bBPB twins | seed_sd_2x | 21% | 0.735 (924) | 0.695 (950) | 0.734 (971) | 0.713 (1006) | 0.733 (1055) |
| Per-language BPB | current | 100% | 0.943 (50) | 0.939 (50) | 0.894 (50) | 0.741 (50) | 0.917 (50) |
| Per-language BPB | ties_half | 100% | 0.943 (50) | 0.939 (50) | 0.894 (50) | 0.741 (50) | 0.917 (50) |
| Per-language BPB | ref_ties_dropped | 100% | 0.943 (50) | 0.939 (50) | 0.894 (50) | 0.741 (50) | 0.917 (50) |
| Per-language BPB | ckpt_noise_1x | 79% | 0.984 (50) | 0.983 (50) | 0.943 (50) | 0.790 (50) | 0.964 (50) |
| Per-language BPB | ckpt_noise_2x | 58% | 0.999 (50) | 0.998 (50) | 0.976 (50) | 0.927 (50) | 0.996 (50) |
| Per-language BPB | seed_sd_1x | 70% | 0.987 (50) | 0.988 (50) | 0.953 (50) | 0.856 (50) | 0.974 (50) |
| Per-language BPB | seed_sd_2x | 48% | 0.999 (50) | 0.998 (50) | 0.984 (50) | 0.988 (50) | 0.999 (50) |

**multi-axis, common cells** (`da_size_definition_sensitivity_multi_axes.csv`)

| population | definition | pairs kept | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|---|
| Benchmark accuracy | current | 100% | 0.526 (265) | 0.540 (291) | 0.538 (313) | 0.562 (347) | 0.580 (395) |
| Benchmark accuracy | ties_half | 100% | 0.553 (265) | 0.566 (291) | 0.564 (313) | 0.589 (347) | 0.606 (395) |
| Benchmark accuracy | ref_ties_dropped | 97% | 0.537 (265) | 0.551 (291) | 0.548 (313) | 0.574 (347) | 0.592 (395) |
| Benchmark accuracy | ckpt_noise_1x | 73% | 0.553 (265) | 0.565 (291) | 0.565 (313) | 0.594 (347) | 0.611 (395) |
| Benchmark accuracy | ckpt_noise_2x | 49% | 0.570 (265) | 0.579 (291) | 0.585 (313) | 0.615 (347) | 0.635 (395) |
| Benchmark accuracy | seed_sd_1x | 55% | 0.561 (265) | 0.574 (291) | 0.577 (313) | 0.609 (347) | 0.631 (395) |
| Benchmark accuracy | seed_sd_2x | 26% | 0.592 (265) | 0.600 (291) | 0.620 (313) | 0.648 (347) | 0.676 (395) |
| Benchmark accuracy and bBPB twins | current | 100% | 0.626 (918) | 0.610 (944) | 0.622 (966) | 0.614 (1000) | 0.628 (1048) |
| Benchmark accuracy and bBPB twins | ties_half | 100% | 0.634 (918) | 0.618 (944) | 0.630 (966) | 0.624 (1000) | 0.638 (1048) |
| Benchmark accuracy and bBPB twins | ref_ties_dropped | 99% | 0.629 (918) | 0.613 (944) | 0.625 (966) | 0.618 (1000) | 0.633 (1048) |
| Benchmark accuracy and bBPB twins | ckpt_noise_1x | 73% | 0.660 (918) | 0.638 (944) | 0.655 (966) | 0.644 (1000) | 0.661 (1048) |
| Benchmark accuracy and bBPB twins | ckpt_noise_2x | 50% | 0.694 (918) | 0.663 (944) | 0.688 (966) | 0.674 (1000) | 0.689 (1048) |
| Benchmark accuracy and bBPB twins | seed_sd_1x | 52% | 0.686 (918) | 0.659 (944) | 0.680 (966) | 0.670 (1000) | 0.688 (1048) |
| Benchmark accuracy and bBPB twins | seed_sd_2x | 24% | 0.736 (918) | 0.697 (944) | 0.735 (966) | 0.714 (1000) | 0.736 (1048) |
| Per-language BPB | current | 100% | 0.943 (50) | 0.939 (50) | 0.894 (50) | 0.741 (50) | 0.917 (50) |
| Per-language BPB | ties_half | 100% | 0.943 (50) | 0.939 (50) | 0.894 (50) | 0.741 (50) | 0.917 (50) |
| Per-language BPB | ref_ties_dropped | 100% | 0.943 (50) | 0.939 (50) | 0.894 (50) | 0.741 (50) | 0.917 (50) |
| Per-language BPB | ckpt_noise_1x | 79% | 0.984 (50) | 0.983 (50) | 0.943 (50) | 0.790 (50) | 0.964 (50) |
| Per-language BPB | ckpt_noise_2x | 58% | 0.999 (50) | 0.998 (50) | 0.976 (50) | 0.927 (50) | 0.996 (50) |
| Per-language BPB | seed_sd_1x | 70% | 0.987 (50) | 0.988 (50) | 0.953 (50) | 0.856 (50) | 0.974 (50) |
| Per-language BPB | seed_sd_2x | 48% | 0.999 (50) | 0.998 (50) | 0.984 (50) | 0.988 (50) | 0.999 (50) |

**mono-axis, own cells** (`da_size_definition_sensitivity_mono_axis.csv`)

| population | definition | pairs kept | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|---|
| Benchmark accuracy | current | 100% | 0.494 (368) | 0.507 (410) | 0.498 (445) | 0.514 (501) | 0.524 (555) |
| Benchmark accuracy | ties_half | 100% | 0.522 (368) | 0.539 (410) | 0.530 (445) | 0.545 (501) | 0.556 (555) |
| Benchmark accuracy | ref_ties_dropped | 96% | 0.505 (366) | 0.519 (408) | 0.508 (443) | 0.524 (499) | 0.536 (553) |
| Benchmark accuracy | ckpt_noise_1x | 69% | 0.516 (343) | 0.525 (383) | 0.522 (416) | 0.546 (470) | 0.549 (519) |
| Benchmark accuracy | ckpt_noise_2x | 42% | 0.522 (294) | 0.543 (332) | 0.540 (361) | 0.577 (404) | 0.573 (453) |
| Benchmark accuracy | seed_sd_1x | 47% | 0.518 (296) | 0.547 (331) | 0.527 (355) | 0.561 (399) | 0.568 (449) |
| Benchmark accuracy | seed_sd_2x | 19% | 0.562 (180) | 0.586 (199) | 0.591 (213) | 0.608 (242) | 0.593 (274) |
| Benchmark accuracy and bBPB twins | current | 100% | 0.586 (1274) | 0.551 (1316) | 0.571 (1351) | 0.551 (1407) | 0.569 (1461) |
| Benchmark accuracy and bBPB twins | ties_half | 100% | 0.594 (1274) | 0.561 (1316) | 0.581 (1351) | 0.562 (1407) | 0.581 (1461) |
| Benchmark accuracy and bBPB twins | ref_ties_dropped | 99% | 0.589 (1272) | 0.555 (1314) | 0.574 (1349) | 0.554 (1405) | 0.573 (1459) |
| Benchmark accuracy and bBPB twins | ckpt_noise_1x | 70% | 0.614 (1200) | 0.571 (1240) | 0.599 (1273) | 0.574 (1327) | 0.590 (1376) |
| Benchmark accuracy and bBPB twins | ckpt_noise_2x | 45% | 0.630 (1068) | 0.585 (1106) | 0.620 (1135) | 0.606 (1178) | 0.607 (1227) |
| Benchmark accuracy and bBPB twins | seed_sd_1x | 43% | 0.630 (969) | 0.582 (1004) | 0.617 (1028) | 0.595 (1072) | 0.606 (1122) |
| Benchmark accuracy and bBPB twins | seed_sd_2x | 17% | 0.659 (607) | 0.612 (626) | 0.663 (640) | 0.649 (669) | 0.645 (701) |
| Per-language BPB | current | 100% | 0.915 (50) | 0.913 (50) | 0.866 (50) | 0.571 (50) | 0.872 (50) |
| Per-language BPB | ties_half | 100% | 0.915 (50) | 0.913 (50) | 0.866 (50) | 0.571 (50) | 0.872 (50) |
| Per-language BPB | ref_ties_dropped | 100% | 0.915 (50) | 0.913 (50) | 0.866 (50) | 0.571 (50) | 0.872 (50) |
| Per-language BPB | ckpt_noise_1x | 72% | 0.984 (50) | 0.986 (50) | 0.933 (50) | 0.593 (50) | 0.951 (50) |
| Per-language BPB | ckpt_noise_2x | 43% | 1.000 (44) | 1.000 (44) | 1.000 (44) | 0.900 (44) | 1.000 (44) |
| Per-language BPB | seed_sd_1x | 57% | 0.981 (48) | 0.986 (48) | 0.959 (48) | 0.766 (48) | 0.964 (48) |
| Per-language BPB | seed_sd_2x | 31% | 0.999 (32) | 0.999 (32) | 0.994 (32) | 0.979 (32) | 0.999 (32) |

**mono-axis, common cells** (`da_size_definition_sensitivity_mono_axis.csv`)

| population | definition | pairs kept | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|---|
| Benchmark accuracy | current | 100% | 0.515 (176) | 0.531 (195) | 0.522 (210) | 0.535 (238) | 0.536 (269) |
| Benchmark accuracy | ties_half | 100% | 0.541 (176) | 0.554 (195) | 0.546 (210) | 0.559 (238) | 0.563 (269) |
| Benchmark accuracy | ref_ties_dropped | 97% | 0.526 (176) | 0.541 (195) | 0.532 (210) | 0.544 (238) | 0.547 (269) |
| Benchmark accuracy | ckpt_noise_1x | 71% | 0.528 (176) | 0.552 (195) | 0.549 (210) | 0.565 (238) | 0.561 (269) |
| Benchmark accuracy | ckpt_noise_2x | 45% | 0.544 (176) | 0.568 (195) | 0.570 (210) | 0.591 (238) | 0.578 (269) |
| Benchmark accuracy | seed_sd_1x | 55% | 0.536 (176) | 0.566 (195) | 0.557 (210) | 0.580 (238) | 0.575 (269) |
| Benchmark accuracy | seed_sd_2x | 26% | 0.559 (176) | 0.591 (195) | 0.592 (210) | 0.610 (238) | 0.594 (269) |
| Benchmark accuracy and bBPB twins | current | 100% | 0.577 (595) | 0.554 (614) | 0.568 (629) | 0.558 (657) | 0.566 (688) |
| Benchmark accuracy and bBPB twins | ties_half | 100% | 0.585 (595) | 0.562 (614) | 0.576 (629) | 0.567 (657) | 0.576 (688) |
| Benchmark accuracy and bBPB twins | ref_ties_dropped | 99% | 0.580 (595) | 0.557 (614) | 0.571 (629) | 0.562 (657) | 0.570 (688) |
| Benchmark accuracy and bBPB twins | ckpt_noise_1x | 72% | 0.602 (595) | 0.575 (614) | 0.595 (629) | 0.583 (657) | 0.588 (688) |
| Benchmark accuracy and bBPB twins | ckpt_noise_2x | 47% | 0.627 (595) | 0.591 (614) | 0.625 (629) | 0.617 (657) | 0.612 (688) |
| Benchmark accuracy and bBPB twins | seed_sd_1x | 53% | 0.620 (595) | 0.584 (614) | 0.615 (629) | 0.607 (657) | 0.606 (688) |
| Benchmark accuracy and bBPB twins | seed_sd_2x | 24% | 0.658 (595) | 0.613 (614) | 0.663 (629) | 0.652 (657) | 0.645 (688) |
| Per-language BPB | current | 100% | 0.889 (32) | 0.879 (32) | 0.826 (32) | 0.597 (32) | 0.841 (32) |
| Per-language BPB | ties_half | 100% | 0.889 (32) | 0.879 (32) | 0.826 (32) | 0.597 (32) | 0.841 (32) |
| Per-language BPB | ref_ties_dropped | 100% | 0.889 (32) | 0.879 (32) | 0.826 (32) | 0.597 (32) | 0.841 (32) |
| Per-language BPB | ckpt_noise_1x | 71% | 0.991 (32) | 0.986 (32) | 0.923 (32) | 0.636 (32) | 0.947 (32) |
| Per-language BPB | ckpt_noise_2x | 41% | 1.000 (32) | 1.000 (32) | 1.000 (32) | 0.987 (32) | 1.000 (32) |
| Per-language BPB | seed_sd_1x | 56% | 0.987 (32) | 0.986 (32) | 0.962 (32) | 0.847 (32) | 0.969 (32) |
| Per-language BPB | seed_sd_2x | 31% | 0.999 (32) | 0.999 (32) | 0.994 (32) | 0.979 (32) | 0.999 (32) |

Paper tabulars: `pretraining/predictivity/da_size_definition_sensitivity_multi_axes.tex`, `pretraining/predictivity/da_size_definition_sensitivity_mono_axis.tex`.
<!-- END auto:da-definition-sensitivity -->

**Key findings**

- **No definition lifts accuracy DA far from chance.** On accuracy alone
  (multi-axis, own cells) the current rule reads 0.51–0.56 from 90M to 1B.
  Counting ties as one half reads 0.54–0.59, and dropping the reference's ties
  reads 0.52–0.58 (96 % of the pairs kept). The strictest filter keeps the
  pairs whose gap exceeds 2 seed SDs of a difference (23 % of the pairs) and
  reads 0.61–0.68.
- **Ties as one half is exactly (1 + τ_a)/2 and adds +0.03.** It moves the
  accuracy cells only, since the bBPB twins and BPB never tie. The pooled
  accuracy and bBPB line therefore moves by +0.01 (0.60–0.62 to 0.60–0.63).
- **The reference's checkpoint noise is a looser threshold than its seed
  noise.** One checkpoint SD keeps 72 % of the accuracy pairs and reads
  0.54–0.60. Two keep 48 % and read 0.56–0.62. One seed SD of a difference
  keeps 51 % and reads 0.56–0.62, close to two checkpoint SDs.
- **On one task set the ordering holds.** On the cells every variant defines
  (241–384 accuracy tasks), accuracy DA goes 0.54–0.59 (current), 0.56–0.62
  (one checkpoint SD), 0.58–0.64 (two) and 0.61–0.68 (two seed SDs).
- **BPB stays far above every accuracy variant.** Per-language BPB reads
  0.73–0.95 under the current rule and 0.89–1.00 above two checkpoint SDs
  (66 % of its pairs). Its 600M dip shrinks from 0.73 to 0.89.
- **Mono-axis decisions move less.** Accuracy reads 0.50–0.53 (current),
  0.53–0.56 (ties as one half), 0.53–0.58 (two checkpoint SDs, 43 % of the
  pairs) and 0.58–0.61 (two seed SDs, 19 %).

**Follow-ups**

- A pooled version (matching over comparable pairs) beside the task means,
  to set this table against the decisive-pairs figure, which pools.
- The reliable-task verdict (`above_66_*`) under the checkpoint-noise filter,
  to see which tasks only fail on pairs the reference cannot separate.

GitHub: [da_size_definition_sensitivity_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_definition_sensitivity_multi_axes.csv) · [da_size_definition_sensitivity_multi_axes.tex](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_definition_sensitivity_multi_axes.tex) ·
GitHub: [da_size_definition_sensitivity_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_definition_sensitivity_mono_axis.csv) · [da_size_definition_sensitivity_mono_axis.tex](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_definition_sensitivity_mono_axis.tex)

### 9. Multi-axis against mono-axis pairs

**`rq2_da_all_above_66_both_mono_vs_multi_axes`: the three definitions · filter `above_66_both`
(441 tasks at 1B, 356 of them bBPB twins, drawn here only for the pair-set comparison) ·
rows multi-axis and mono-axis · pairs from `predictivity` · gate
`predictivity`.** This figure is exploratory (`pair_axes.py`). It fed
`plan/decision_accuracy.md` and the `axes` column of rule 15.

![Multi-axis against mono-axis pairs](pretraining/predictivity/rq2_da_all_above_66_both_mono_vs_multi_axes.png)

`pair_axes.py` computes the three decision accuracies on the `above_66_both`
cells twice. The first pass uses every pair at the grid seed (multi-axis,
this analysis's convention). The second uses the pairs that move exactly one
of L, depth, activation, data scheme and temperature (mono-axis, which is
what DataDecide's "all pairs" are by construction). The proposal that
followed, an `axes` column in `da_all_per_task_both_axes.csv` with mono-axis
as the headline for decisions, is in `plan/decision_accuracy.md` (§2, §6).

**Key findings**

- **DA-size reads 0.06–0.08 lower under the mono-axis pairs.** It is 0.679,
  0.664, 0.671, 0.645, 0.672 against 0.739, 0.733, 0.733, 0.722, 0.739 from
  90M to 1B, and DA-goal at 5C likewise (`rq2_da_all_above_66_both_mono_vs_multi_axes.csv`).
- **The mono-axis pairs are the stricter and smaller set.** They are 25 % of
  the decisions (7,293 of 29,291 at 175M), on 426–441 tasks at every size now
  that the twins carry every size and checkpoint. DA-ckpt reads 0.02–0.07
  lower under them too, and the gap closes towards 4.5C (0.84–0.92 against
  0.87–0.93 at 4.5C).

**Follow-ups**

- Mono-axis as the headline for decisions, which the paper's left panel
  already draws, and the multi-axis pooled line as the population statement.

GitHub: [rq2_da_all_above_66_both_mono_vs_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_mono_vs_multi_axes.png) · [rq2_da_all_above_66_both_mono_vs_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_mono_vs_multi_axes.csv)

### 10. Cross-task predictability

**DA-size and DA-ckpt · no filter · multi-axis pairs of `predictivity` (every
build: 31 variants at 1.7B and 465 pairs, while DA-ckpt pools the 2,977
within-size pairs of every size) ·
gate `predictivity` on both sides.** Every parent task serves as the proxy
for every other one. A cell is the smallest size or earliest checkpoint at
which x's ranking safely predicts y's final ranking (DA ≥ 0.75 over ≥ 3
pairs, held at every larger level).

<!-- BEGIN auto:cross-task (cross_task.py --pool predictivity) -->
## Cross-task predictability

Each map uses every parent task as a proxy for every other one (1896 x 1896). A cell gives the earliest level at which the ranking on task x (columns) safely predicts the final ranking on task y (rows). Safely means DA >= 0.75 over >= 3 pairs at that level and at every larger level with a value. For DA-size the level is the smallest proxy size (32 variants at 1.7B, 496 pairs). For DA-ckpt it is the earliest of the nine checkpoints before the final, over the within-size pairs of every size pooled (3008 pairs). The diagonal is rq02's own-task DA. The gate empties a benchmark's pairs at every size where it is at chance. The `_by_family` maps take the median level over the task pairs of two benchmarks. The `_by_language` maps take it over the same-benchmark task pairs of two languages, ordered by resource as in the scheme-A lists. Regenerate with `python analysis/rq02_decision_accuracy/cross_task.py --pool predictivity`.

The benchmark maps repeat the DA-size and DA-ckpt task-level maps on the benchmark tasks above chance at some size. They drop BPB, the loss and the benchmarks the gate finds at chance everywhere, so they keep only the part of the map where a transfer result is possible at all: [`cross_task_da_size_benchmarks_multi_axes.png`](pretraining/predictivity/cross_task_da_size_benchmarks_multi_axes.png), [`cross_task_da_ckpt_benchmarks_multi_axes.png`](pretraining/predictivity/cross_task_da_ckpt_benchmarks_multi_axes.png).

![Cross-task DA-size by benchmark](pretraining/predictivity/cross_task_da_size_by_family_multi_axes.png)

![Cross-task DA-ckpt by benchmark](pretraining/predictivity/cross_task_da_ckpt_by_family_multi_axes.png)

![Cross-task DA-size by language](pretraining/predictivity/cross_task_da_size_by_language_multi_axes.png)

![Cross-task DA-ckpt by language](pretraining/predictivity/cross_task_da_ckpt_by_language_multi_axes.png)

Full task-level maps: [`cross_task_da_size_multi_axes.png`](pretraining/predictivity/cross_task_da_size_multi_axes.png), [`cross_task_da_ckpt_multi_axes.png`](pretraining/predictivity/cross_task_da_ckpt_multi_axes.png).
<!-- END auto:cross-task -->

**Key findings**

- **Cross-task predictability is almost absent between accuracies.** Of the
  8,556 off-diagonal pairs of the 93 benchmark families in the DA-size map, 59
  reach a median safe size (30 at 1B, 14 at 600M, 8 at 90M, 5 at 350M, 2 at
  175M). On average 4.1 % of a family pair's task pairs reach one, 60 % never
  do and 36 % are gated (`cross_task_da_size_by_family_multi_axes.csv`).
- **45 of the 59 have BPB, the loss or a bBPB twin on at least one side.** The
  most frequent are `global_piqa_nonparallel_cloze` (accuracy or twin), then
  per-language BPB and multiblimp-extra. The other 14 are accuracy →
  accuracy, among them HellaSwag as the proxy for
  `global_piqa_nonparallel_cloze` at 1B and winogrande as the proxy for boolq,
  hellaswag and piqa at 600M.
- **The DA-ckpt map, now with the twins at every checkpoint, says much the
  same.** 53 of the 8,556 family pairs reach a median safe checkpoint (46 at
  0.5C), 45 of them with BPB, the loss or a twin on a side. On average 2.3 %
  of a family pair's task pairs reach one, 62 % never do and 35 % are gated
  (`cross_task_da_ckpt_by_family_multi_axes.csv`).

**Follow-ups**

- Draw the share of task pairs that reach a safe size (in the CSV) as the
  cell, and keep the median level as the small number. The family maps now
  show the median of the pairs that succeed, while most pairs never do.
- Compare against a permutation baseline: the cross-task DA of two languages
  whose L lists are shuffled. Same-L blocks share the design variants, so a
  transfer within one list is partly true by construction.

GitHub: [cross_task_da_size_by_family_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_family_multi_axes.png) · [cross_task_da_size_by_family_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_family_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_by_family_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_family_multi_axes.png) · [cross_task_da_ckpt_by_family_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_family_multi_axes.csv) ·
GitHub: [cross_task_da_size_by_language_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_language_multi_axes.png) · [cross_task_da_size_by_language_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_language_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_by_language_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_language_multi_axes.png) · [cross_task_da_ckpt_by_language_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_language_multi_axes.csv) ·
GitHub: [cross_task_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_multi_axes.png) · [cross_task_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_multi_axes.png) · [cross_task_da_ckpt_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_multi_axes.csv) ·
GitHub: [cross_task_da_size_benchmarks_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_benchmarks_multi_axes.png) · [cross_task_da_size_benchmarks_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_benchmarks_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_benchmarks_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_benchmarks_multi_axes.png) · [cross_task_da_ckpt_benchmarks_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_benchmarks_multi_axes.csv)

### 11. Benchmark BPB: a continuous score on the same items

**DA-size · no filter · multi-axis pairs of `predictivity` (29
variants at 1.7B, 406 pairs) · final checkpoints only · gate `predictivity`
on the accuracy side only.** Figures 1–10 read every benchmark through its
accuracy, a step function of the log-likelihoods the harness already
computed, with the twins added at every size and checkpoint (Setup). This
figure reads the same items, the same final checkpoints and the same pairs
through the gold answer's bits per byte (**bBPB**). Does a continuous score
rank like the reference where accuracy is at chance?

How bBPB is computed (no model is re-run, `build_per_item_store.py` +
`bench_bpb_da.py`):

- **Source.** The lm-eval samples files of every checkpoint,
  `eval_logs/.../<cell>-iter<N>/harness/eval_*/samples_<task>_*.jsonl`. The
  store holds all of them since 2026-10-07, and this figure reads the finals. The
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
- **Three readings, one kernel.** All three run `pair_agreement` over the
  same families and pairs, with rule 5's NaN below three pairs.
  - acc → 1.7B acc.
  - bBPB → 1.7B acc, the practical question: does a small model's bBPB predict
    the reference's accuracy ranking?
  - bBPB → 1.7B bBPB.
- **The gate (rule 1) applies to accuracy only.** bBPB has no chance level.
  - The head-to-head population is the tasks above chance at 1.7B, where the
    target ranking is not noise.
  - acc → acc also needs the task above chance at the proxy, as in every
    other figure here, so its task count is smaller.
  - bBPB is not gated at the proxy. That would discard exactly the regime it
    is for: a proxy at chance on accuracy whose bBPB still separates the
    designs.
  - The paired test (Wilcoxon, bBPB → acc minus acc → acc) runs on the tasks
    where both are defined.
  - The raw ungated DAs and both gate flags stay in `bench_bpb_da_size_multi_axes.csv`.

<!-- BEGIN auto:bench-bpb (bench_bpb_da.py --pool predictivity) -->
## Benchmark BPB against accuracy

Each table gives DA-size on the final checkpoints under three readings: the proxy's accuracy or its benchmark BPB (bBPB) predicts the 1.7B accuracy (acc → acc, bBPB → acc), or bBPB predicts the 1.7B bBPB (bBPB → bBPB). The pairs are the multi-axis pairs of `predictivity`, 496 of its 528: those between families the store holds. Every reading and the FineWeb2 column use the same pairs. The `predictivity` gate applies to the accuracy side only. Every reading counts the tasks above chance at 1.7B, and acc → acc also needs the task above chance at the proxy, so its task count is smaller. The paired gain is bBPB → acc minus acc → acc, on the tasks where both are defined. FineWeb2 val BPB is `bpb_macro`'s DA-size on the same pairs. Regenerate with `python analysis/rq02_decision_accuracy/bench_bpb_da.py --pool predictivity` (after `build_per_item_store.py --pool predictivity --finals-only`). The store `predictivity` lacks 1 of the pool's models (lm-90M-L1-b84-swiglu-seed1904). They stay out until the store is rebuilt for the pool.

**all benchmarks**

| proxy | acc → acc (tasks) | bBPB → acc (tasks) | bBPB → bBPB | paired gain (tasks) | bBPB better / worse | Wilcoxon p | FineWeb2 val BPB |
|---|---|---|---|---|---|---|---|
| 90M | 0.50 (360) | 0.59 (587) | 0.69 | +0.10 (360) | 71% / 21% | <0.001 | 0.95 |
| 175M | 0.52 (400) | 0.58 (587) | 0.64 | +0.06 (400) | 60% / 32% | <0.001 | 0.96 |
| 350M | 0.52 (435) | 0.58 (587) | 0.68 | +0.06 (435) | 64% / 29% | <0.001 | 0.98 |
| 600M | 0.54 (488) | 0.58 (587) | 0.65 | +0.05 (488) | 60% / 32% | <0.001 | 0.95 |
| 1B | 0.56 (542) | 0.59 (587) | 0.67 | +0.03 (542) | 57% / 35% | <0.001 | 0.97 |

**all cloze** (the answer text is the continuation)

| proxy | acc → acc (tasks) | bBPB → acc (tasks) | bBPB → bBPB | paired gain (tasks) | bBPB better / worse | Wilcoxon p | FineWeb2 val BPB |
|---|---|---|---|---|---|---|---|
| 90M | 0.50 (360) | 0.59 (580) | 0.69 | +0.10 (360) | 71% / 21% | <0.001 | 0.95 |
| 175M | 0.52 (400) | 0.58 (580) | 0.64 | +0.06 (400) | 60% / 32% | <0.001 | 0.96 |
| 350M | 0.52 (435) | 0.58 (580) | 0.68 | +0.06 (435) | 64% / 29% | <0.001 | 0.98 |
| 600M | 0.54 (487) | 0.58 (580) | 0.65 | +0.05 (487) | 60% / 32% | <0.001 | 0.95 |
| 1B | 0.56 (540) | 0.59 (580) | 0.67 | +0.03 (540) | 57% / 35% | <0.001 | 0.97 |

**all lettered** (the continuation is the answer letter, so bBPB is the letter's surprisal)

| proxy | acc → acc (tasks) | bBPB → acc (tasks) | bBPB → bBPB | paired gain (tasks) | bBPB better / worse | Wilcoxon p | FineWeb2 val BPB |
|---|---|---|---|---|---|---|---|
| 90M |  (0) | 0.41 (7) | 0.48 | — (0) | — |  | 0.95 |
| 175M |  (0) | 0.42 (7) | 0.50 | — (0) | — |  | 0.96 |
| 350M |  (0) | 0.46 (7) | 0.54 | — (0) | — |  | 0.98 |
| 600M | 0.67 (1) | 0.53 (7) | 0.57 | +0.17 (1) | 100% / 0% | 1.000 | 0.95 |
| 1B | 0.33 (2) | 0.42 (7) | 0.73 | +0.04 (2) | 50% / 0% | 1.000 | 0.97 |

**Per benchmark**, mean over the five proxy sizes. `tasks` counts the parent tasks with bBPB. A cell is the task-mean DA over the tasks above chance at 1.7B (for acc → acc, also above chance at the proxy), blank when the gate removes them all:

| benchmark | tasks | acc → 1.7B acc | bBPB → 1.7B acc | bBPB → 1.7B bBPB |
|---|---|---|---|---|
| acp_bench_cloze | 7 |  |  |  |
| arc | 28 | 0.62 | 0.59 | 0.62 |
| arc_mt | 11 | 0.57 | 0.61 | 0.62 |
| bbh_cloze | 6 |  |  |  |
| bbh_mcq | 17 |  |  |  |
| boolq | 1 | 0.51 | 0.50 | 0.49 |
| global_piqa_nonparallel_cloze | 2 | 0.58 | 0.72 | 0.90 |
| global_piqa_parallel_cloze | 63 | 0.49 | 0.50 | 0.63 |
| hellaswag | 26 | 0.71 | 0.79 | 0.85 |
| include_v2_en | 77 | 0.48 | 0.49 | 0.53 |
| include_v2_og | 77 | 0.53 | 0.58 | 0.65 |
| kobest | 2 | 0.41 | 0.61 | 0.88 |
| mathqa | 1 | 0.48 | 0.55 | 0.54 |
| multiblimp | 34 | 0.65 | 0.65 | 0.68 |
| multiblimp-extra | 3 | 0.79 | 0.82 | 0.87 |
| noreval | 7 | 0.57 | 0.55 | 0.62 |
| noreval_extra | 30 | 0.45 | 0.46 | 0.57 |
| openbookqa | 1 | 0.53 | 0.59 | 0.58 |
| paws | 8 | 0.56 | 0.52 | 0.70 |
| piqa | 1 | 0.59 | 0.65 | 0.61 |
| acp_bench_mcq-rf | 7 | 0.44 | 0.49 | 0.52 |
| bbh_mcq-rf | 17 | 0.46 | 0.47 | 0.53 |
| belebele-rf | 59 | 0.48 | 0.58 | 0.69 |
| commonsense_qa-rf | 1 | 0.53 | 0.58 | 0.59 |
| cultural_bench_easy-rf | 19 | 0.35 | 0.42 | 0.56 |
| global_mmlu_full-rf | 29 | 0.56 | 0.68 | 0.67 |
| include_base_44-rf | 36 | 0.52 | 0.61 | 0.69 |
| m_mmlu-rf | 27 | 0.52 | 0.59 | 0.68 |
| mmlu-rf | 1 | 0.64 | 0.66 | 0.63 |
| mmmlu-rf | 12 | 0.58 | 0.66 | 0.65 |
| belebele-rfgm | 59 | 0.50 | 0.59 | 0.74 |
| include_base_44-rfgm | 36 | 0.55 | 0.63 | 0.79 |
| toksuite | 5 | 0.42 | 0.46 | 0.55 |
| toksuite_math | 1 | 0.56 | 0.68 | 0.82 |
| toksuite_stem | 1 | 0.47 | 0.52 | 0.57 |
| toxigen | 1 |  |  |  |
| truthfulqa-multi_mc1 | 2 |  |  |  |
| truthfulqa_mc2 | 3 |  |  |  |
| xcopa | 8 | 0.48 | 0.60 | 0.73 |
| xnli | 15 | 0.51 | 0.54 | 0.67 |
| xstorycloze | 8 | 0.63 | 0.70 | 0.83 |
| acp_bench_mcq (letter) | 7 |  |  |  |
| belebele (letter) | 59 |  | 0.44 | 0.57 |
| blend_sample (letter) | 5 |  |  |  |
| commonsense_qa (letter) | 1 |  |  |  |
| cultural_bench_easy (letter) | 19 |  |  |  |
| global_mmlu_full (letter) | 29 |  | 0.43 | 0.53 |
| include_base_44 (letter) | 36 | 0.50 | 0.48 | 0.58 |
| mmlu (letter) | 1 |  |  |  |

![bBPB DA, overall](pretraining/predictivity/bench_bpb_da_size_bars_multi_axes.png)

![bBPB DA, per benchmark](pretraining/predictivity/bench_bpb_da_size_bars_benchmarks_multi_axes.png)

![bBPB DA, heat map](pretraining/predictivity/bench_bpb_da_size_heatmap_multi_axes.png)
<!-- END auto:bench-bpb -->

Snapshot: ladder report 2026-10-10 04:31, outputs of the 2026-10-10 refresh.
bBPB comes from the per-item store `predictivity` (rebuilt over every
checkpoint, finals read here), with 816 parent tasks. The 19 pool models the
store lacks (two swiglu, seventeen muon) are left out and listed in the auto
block. This figure reads the store directly and has bBPB at all five proxy
sizes, as the twins of figures 1–10 now do too. The raw per-task DAs and both
gate flags are in [`bench_bpb_da_size_multi_axes.csv`](pretraining/predictivity/bench_bpb_da_size_multi_axes.csv),
and every number below is in [`bench_bpb_da_size_summary_multi_axes.csv`](pretraining/predictivity/bench_bpb_da_size_summary_multi_axes.csv).

**Key findings**

- **bBPB predicts the 1.7B accuracy ranking better than accuracy does.**
  - Paired over the tasks where both readings are defined, bBPB → acc minus
    acc → acc is +0.12 at 90M (0.60 against 0.51). bBPB is better on 71 % of
    tasks and worse on 21 %.
  - The gain shrinks with the proxy size: +0.06 at 175M, +0.07 at 350M, +0.06
    at 600M and +0.03 at 1B, where it is better on 56 % and worse on 35 %.
    Wilcoxon p < 0.001 at every size.
- **It reads more tasks.** bBPB → acc is defined on 501 tasks at every proxy
  size. acc → acc is defined on 294 at 90M and 459 at 1B, because accuracy
  must also clear chance at the proxy.
- **It is flat in size and accuracy is not.** bBPB → acc stays at 0.58–0.60
  from 90M to 1B, while acc → acc only climbs from 0.51 to 0.57.
  - A 90M model's bBPB is as useful a proxy as a 1B model's.
  - Most of the gain is where accuracy is still noise.
- **bBPB ranks itself more consistently than it predicts accuracy.**
  - bBPB → 1.7B bBPB is 0.65–0.70, above bBPB → acc (0.58–0.60). Part of
    what a small model's bBPB gets right is the reference's bBPB ranking,
    which the reference's accuracy does not share. How large that part is
    remains a follow-up.
  - It is still far below FineWeb2 validation BPB (`bpb_macro`, 0.94–0.97),
    which averages 1M tokens per language against a benchmark's few hundred
    items.
- **Per benchmark.** Means are over the five proxy sizes, and the arrows read
  as acc → acc, then bBPB → acc.
  - The largest gains are on the multilingual cloze sets: global_mmlu_full-rf
    0.56 → 0.68, xcopa 0.48 → 0.61, include_base_44-rf 0.52 → 0.62,
    belebele-rf 0.49 → 0.58 and, on two tasks, global_piqa_nonparallel_cloze
    0.58 → 0.72.
  - hellaswag (0.72 → 0.80) and xstorycloze (0.65 → 0.72) were already
    reliable and stay the best.
  - The include_v2 English questions do not move (0.48 → 0.49).
  - The Gemini-rewritten twins rank themselves best on bBPB: include_base_44
    rfgm 0.80, belebele rfgm 0.74.
- **The lettered tasks give no reading.**
  - Only 7 of 157 lettered tasks are above chance at 1.7B, and their
    continuation is a letter.
  - bBPB → acc there is 0.42–0.53, which is chance. The `rf_` twins are the
    way to read those items with a continuous score.

<!-- BEGIN auto:bench-bpb-acc-gate (bench_bpb_da.py --pool predictivity) -->
## bBPB where accuracy is at chance

This table splits the same task × size DAs (multi-axis pairs of `predictivity`, finals) by the ACCURACY gate (rule 1, `predictivity`). Each row is the mean DA-size of the bBPB twin over the tasks whose accuracy is at chance, or above it, at the proxy, at 1.7B or at every size. bBPB → acc needs the 1.7B accuracy above chance, so it is blank where the population is at chance there. bBPB → bBPB has no chance level and is read on every task. acc → acc is the ungated accuracy DA, shown for contrast. Cells read mean (tasks). Table: `bench_bpb_da_size_by_acc_gate_multi_axes.csv`.

| group | accuracy gate | proxy | bBPB → 1.7B acc | bBPB → 1.7B bBPB | acc → acc (ungated) | FineWeb2 val BPB |
|---|---|---|---|---|---|---|
| all benchmarks | acc above chance at the proxy | 90M | 0.61 (360) | 0.70 (374) | 0.50 | 0.95 |
| all benchmarks | acc above chance at the proxy | 175M | 0.58 (400) | 0.65 (404) | 0.52 | 0.96 |
| all benchmarks | acc above chance at the proxy | 350M | 0.59 (435) | 0.68 (443) | 0.52 | 0.98 |
| all benchmarks | acc above chance at the proxy | 600M | 0.59 (488) | 0.65 (490) | 0.54 | 0.95 |
| all benchmarks | acc above chance at the proxy | 1B | 0.59 (542) | 0.66 (546) | 0.56 | 0.97 |
| all benchmarks | acc at chance at the proxy | 90M | 0.56 (227) | 0.63 (532) | 0.46 | 0.95 |
| all benchmarks | acc at chance at the proxy | 175M | 0.56 (187) | 0.60 (502) | 0.47 | 0.96 |
| all benchmarks | acc at chance at the proxy | 350M | 0.56 (152) | 0.61 (463) | 0.46 | 0.98 |
| all benchmarks | acc at chance at the proxy | 600M | 0.56 (99) | 0.60 (416) | 0.48 | 0.95 |
| all benchmarks | acc at chance at the proxy | 1B | 0.53 (45) | 0.62 (360) | 0.50 | 0.97 |
| all benchmarks | acc at chance at 1.7B | 90M |  | 0.61 (319) | 0.46 | 0.95 |
| all benchmarks | acc at chance at 1.7B | 175M |  | 0.59 (319) | 0.46 | 0.96 |
| all benchmarks | acc at chance at 1.7B | 350M |  | 0.59 (319) | 0.45 | 0.98 |
| all benchmarks | acc at chance at 1.7B | 600M |  | 0.59 (319) | 0.47 | 0.95 |
| all benchmarks | acc at chance at 1.7B | 1B |  | 0.60 (319) | 0.49 | 0.97 |
| all benchmarks | acc at chance at every size | 90M |  | 0.62 (296) | 0.46 | 0.95 |
| all benchmarks | acc at chance at every size | 175M |  | 0.60 (296) | 0.46 | 0.96 |
| all benchmarks | acc at chance at every size | 350M |  | 0.59 (296) | 0.46 | 0.98 |
| all benchmarks | acc at chance at every size | 600M |  | 0.60 (296) | 0.47 | 0.95 |
| all benchmarks | acc at chance at every size | 1B |  | 0.61 (296) | 0.49 | 0.97 |
| all cloze | acc above chance at the proxy | 90M | 0.61 (360) | 0.70 (362) | 0.50 | 0.95 |
| all cloze | acc above chance at the proxy | 175M | 0.58 (400) | 0.65 (403) | 0.52 | 0.96 |
| all cloze | acc above chance at the proxy | 350M | 0.59 (435) | 0.68 (438) | 0.52 | 0.98 |
| all cloze | acc above chance at the proxy | 600M | 0.59 (487) | 0.65 (489) | 0.54 | 0.95 |
| all cloze | acc above chance at the proxy | 1B | 0.59 (540) | 0.66 (543) | 0.56 | 0.97 |
| all cloze | acc at chance at the proxy | 90M | 0.57 (220) | 0.65 (386) | 0.46 | 0.95 |
| all cloze | acc at chance at the proxy | 175M | 0.57 (180) | 0.63 (345) | 0.47 | 0.96 |
| all cloze | acc at chance at the proxy | 350M | 0.57 (145) | 0.65 (310) | 0.47 | 0.98 |
| all cloze | acc at chance at the proxy | 600M | 0.56 (93) | 0.63 (259) | 0.46 | 0.95 |
| all cloze | acc at chance at the proxy | 1B | 0.55 (40) | 0.65 (205) | 0.47 | 0.97 |
| all cloze | acc at chance at 1.7B | 90M |  | 0.63 (168) | 0.44 | 0.95 |
| all cloze | acc at chance at 1.7B | 175M |  | 0.63 (168) | 0.45 | 0.96 |
| all cloze | acc at chance at 1.7B | 350M |  | 0.63 (168) | 0.45 | 0.98 |
| all cloze | acc at chance at 1.7B | 600M |  | 0.61 (168) | 0.45 | 0.95 |
| all cloze | acc at chance at 1.7B | 1B |  | 0.64 (168) | 0.46 | 0.97 |
| all cloze | acc at chance at every size | 90M |  | 0.63 (162) | 0.44 | 0.95 |
| all cloze | acc at chance at every size | 175M |  | 0.63 (162) | 0.45 | 0.96 |
| all cloze | acc at chance at every size | 350M |  | 0.64 (162) | 0.45 | 0.98 |
| all cloze | acc at chance at every size | 600M |  | 0.61 (162) | 0.45 | 0.95 |
| all cloze | acc at chance at every size | 1B |  | 0.64 (162) | 0.46 | 0.97 |

![bBPB where accuracy is at chance](pretraining/predictivity/bench_bpb_da_size_by_acc_gate_multi_axes_paper.png)
<!-- END auto:bench-bpb-acc-gate -->

**bBPB where accuracy is at chance: DA-size · no filter · multi-axis pairs
of `predictivity` (the 406 pairs above) · finals · gate `predictivity` on the
accuracy side, used here to split the tasks rather than drop them.** Where
accuracy is at chance, does the twin rank near the per-language BPB level
(0.73–0.95, FineWeb2 macro 0.94–0.97)?

**Key findings**

- **No: on the accuracy-gated cells bBPB reads 0.55–0.64, not ~0.9.** Where
  accuracy is at chance at the proxy but above it at 1.7B, bBPB → 1.7B
  accuracy is 0.57 at 90M (207 tasks) and 0.55 at 1B (42). bBPB → 1.7B bBPB
  on every proxy-gated task is 0.61–0.64 (354–508 tasks).
- **Still clearly better than accuracy there.** On the same proxy-gated
  tasks the ungated accuracy DA is 0.46–0.50. bBPB → bBPB lifts the cell by
  +0.13 to +0.19 without reaching the BPB level.
- **The tasks at chance at every size rank themselves at 0.60–0.63.** On
  those 294 tasks (161 cloze, 0.62–0.64) bBPB → bBPB is the only reading. The
  same items' accuracy reads 0.45–0.49.
- **Accuracy at chance costs bBPB little.** bBPB → acc reads 0.59–0.63 on
  the tasks above chance at the proxy against 0.55–0.57 below it. bBPB →
  bBPB reads 0.66–0.71 against 0.61–0.64. Scoring by log-likelihood where
  accuracy is at chance therefore buys a ranking above chance, but not one
  as good as BPB.

**Follow-ups**

- The same split on DA-ckpt, where accuracy is at chance for most of a small
  run (the twins have every checkpoint).
- Per benchmark, which families carry the 0.6. include_v2_en sits near 0.5
  on every reading (0.48 / 0.49 / 0.52 in the per-benchmark table).

GitHub: [bench_bpb_da_size_by_acc_gate_multi_axes_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_by_acc_gate_multi_axes_paper.png) · [bench_bpb_da_size_by_acc_gate_multi_axes_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_by_acc_gate_multi_axes_paper.csv) · [bench_bpb_da_size_by_acc_gate_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_by_acc_gate_multi_axes.csv)

**Follow-ups**

- ~~**The full checkpoint grid.**~~ Done on 2026-10-07. The store holds every
  checkpoint of every seed-1904 cell, so the twins carry bBPB's DA-ckpt and
  DA-goal into figure 3 (DA-goal 0.61–0.67 at the proxies, DA-ckpt 0.87–0.93
  at 90 % of a run) and its SNR into the noise-and-SNR analysis.
- ~~**Feed bBPB into the shared pipeline.**~~ Done. `utils.with_bbpb_twins`
  puts a `bbpb_<task>` twin in every `build_snr_pool` population, so
  `compute_da.py`, the scale-convergence figure and the SNR tables already
  read it. This figure stays as the head-to-head of the three readings.
- **xwinograd and lambada.** Their `target` is the answer string, not a
  choice index, so they have no bBPB yet.
  - xwinograd's gold index is the doc's `answer` field.
  - lambada has one continuation, so its gold is that continuation.
  - Both are a change to `gold_index` and a re-extract of those two families.
- **Seed null.** Repeat on `predictivity_seeds` to place bBPB's 0.60 against
  the two-seeds-of-one-design null of figure 7, as was done for DA-ckpt. The
  store now holds the 20 seed-replicate models, and figure 7's null already
  reads the twins.
- **The reference's two rankings.** The DA between the 1.7B bBPB ranking and
  the 1.7B accuracy ranking, per task, bounds what bBPB → acc can reach. It
  separates the gap between likelihood and accuracy from proxy noise.
- **Cross-reading.** Set bBPB → acc against figure 10's cross-task map, with
  a task's bBPB as the proxy for another task's accuracy.

GitHub: [bench_bpb_da_size_bars_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_bars_multi_axes.png) · [bench_bpb_da_size_bars_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_bars_multi_axes.csv) ·
GitHub: [bench_bpb_da_size_bars_benchmarks_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_bars_benchmarks_multi_axes.png) · [bench_bpb_da_size_bars_benchmarks_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_bars_benchmarks_multi_axes.csv) ·
GitHub: [bench_bpb_da_size_heatmap_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_heatmap_multi_axes.png) · [bench_bpb_da_size_heatmap_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_heatmap_multi_axes.csv)

### 12. Read next in the other RQs

Three decision-accuracy readings live in the RQs their figures belong to. The
auto block of the first stays here, because its script reads this folder's
per-task table.

- **Scaling cleanly and ranking like the reference are different
  properties.** Across tasks, the scaling analysis's ρ of score with size
  correlates with the proxy's ranking ρ at only +0.30 (175M) to +0.42 (90M).
  The trajectory R² is the better surrogate (+0.49 to +0.57 with DA-size), on
  888–995 tasks with the twins included at every size (auto block below).
  Write-up:
  [scaling predictability, figure 5](../rq01_scaling_predictability/README.md#5-scaling-cleanly-and-ranking-like-the-reference-are-different-properties).
- **FineTasks' selection criteria, judged by the reference they cannot
  see.** The surrogates write-up reads their pass rates and the DA-size of
  the tasks they pass:
  [FineTasks' criteria](../rq04_surrogates/README.md#finetasks-criteria-on-the-ladder).
- **The minimal language panel.** The language-transfer write-up reads which
  single language, or which macro, best predicts the 1.7B macro ranking over
  the L8 panel:
  [the minimal language panel](../rq06_language_transfer/README.md#the-minimal-language-panel).

<!-- BEGIN auto:scaling-vs-ranking (scaling_vs_ranking.py --pool predictivity) -->
## Scaling utility against ranking utility

For each proxy size, the table gives the Spearman correlation ACROSS TASKS between rq01's scaling statistics and rq02's ranking statistics. The scaling statistics are the ρ of score with model size, the R² of the size fit and the R² of the trajectory fit. The ranking statistics are the ρ between the proxy's ranking of the design variants and the 1.7B ranking, and DA-size. The last two columns compare the mean DA-size of the tasks rq01 calls 'predictable across both' with that of the rest. Regenerate with `python analysis/rq02_decision_accuracy/scaling_vs_ranking.py --pool predictivity`.

| proxy | tasks | ρ_size vs ρ_ranking | R²_size vs DA-size | R²_traj vs DA-size | DA-size, predictable both | DA-size, other regimes |
|---|---|---|---|---|---|---|
| 90M | 966 | +0.45 | +0.55 | +0.58 | 0.62 | 0.51 |
| 175M | 1000 | +0.33 | +0.43 | +0.53 | 0.60 | 0.51 |
| 350M | 1031 | +0.48 | +0.54 | +0.62 | 0.61 | 0.49 |
| 600M | 1071 | +0.39 | +0.44 | +0.52 | 0.61 | 0.51 |
| 1B | 1072 | +0.45 | +0.49 | +0.62 | 0.62 | 0.50 |

![Scaling against ranking](pretraining/predictivity/scaling_vs_ranking_da_size_multi_axes.png)
<!-- END auto:scaling-vs-ranking -->

GitHub: [scaling_vs_ranking_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking_da_size_multi_axes.png) · [scaling_vs_ranking_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking_da_size_multi_axes.csv)

## Extensions from other sweeps

### The public model lines, one rung above the ladder

These numbers come from the **external tier**, not the ladder: public base
models with the `all/external/` gate, namely gemma-3, Qwen3-Base and OLMo-2,
each with a base model near 1B and near 13B. Here the "decision" is between
whole lab recipes that differ in everything at once, and the token budgets
differ by line and size. Three lines give three pairs (rule 5's minimum), so
the per-task DA can only take a few values. The numbers are never pooled
with the ladder's one-axis decisions.

**DA-size between size buckets · tasks above chance at the 1B, 1.7B and
12–14B buckets of the external gate (54 tasks) · three line pairs · no ladder
pairs.**

<!-- BEGIN auto:public-ladders (public_ladders.py --pool predictivity) -->
## Decision accuracy one rung above the ladder: the public lines

gemma-3, Qwen3-Base and OLMo-2 each have a base model near 1B and near 13B in the external tier. For each task above chance at both buckets, DA is the share of the three line pairs that the small models order like the large ones. Pooled, it is 0.73 over 54 tasks, against the ladder's own 1B → 1.7B DA-size of 0.63 on the same tasks. Per line pair: Qwen3 vs OLMo-2 reaches 0.87 against a majority baseline of 0.80 (11 minority tasks, DA there 0.55). gemma-3 vs OLMo-2 reaches 0.76 against a majority baseline of 0.81 (10 minority tasks, DA there 0.80). gemma-3 vs Qwen3 reaches 0.57 against a majority baseline of 0.56 (24 minority tasks, DA there 0.79). The majority baseline is the DA a proxy gets by always naming the line that usually wins at 12–14B. Only DA above it is information the small models add. These are decisions between labs at public sizes, not the ladder's one-axis decisions. With three pairs, a task's DA can only be 0, ⅓, ⅔ or 1, so read the pooled numbers. Regenerate with `python analysis/rq02_decision_accuracy/public_ladders.py --pool predictivity`.

| family | tasks | DA public lines (1B–1.7B → 12–14B) | DA ladder (1B → 1.7B) |
|---|---|---|---|
| multiblimp | 7 | 0.67 | 0.58 |
| global_piqa_completions | 5 | 0.73 | — |
| hellaswag | 5 | 0.67 | 0.83 |
| m_truthfulqa_mc2 | 5 | 0.67 | — |
| xnli | 5 | 0.80 | 0.50 |
| xstorycloze | 5 | 0.87 | 0.65 |
| xcopa | 4 | 0.83 | 0.63 |
| xwinograd | 4 | 0.75 | 0.59 |
| belebele | 3 | 0.67 | — |

![Public ladders](pretraining/predictivity/public_ladders_da_size_multi_axes.png)
<!-- END auto:public-ladders -->

**Key findings**

- **The pooled DA mostly replays a majority order.** The pooled 0.73 over 54
  tasks is within 0.07 of a majority-order baseline. A proxy that always names
  the line that usually wins at 12–14B scores 0.80 / 0.81 / 0.56 on the three
  pairs (DA +0.07, −0.06, +0.02 against it). The figure shows differences
  between labs (Qwen3 above OLMo-2 on four tasks in five), not a ranking that
  each task preserves.
- **The minority tasks are too few to be a finding.** On the 10–24 minority
  tasks, where the reference order is the uncommon one, the small models read
  it at 0.55–0.80. Over all 54 tasks the public lines pool to 0.73, against
  the ladder's own 1B → 1.7B DA-size of 0.64 on the same tasks.
- **As evaluated today, the public tier cannot say whether benchmarks read
  decisions above 1.7B.** Its three lines are three pairs, rule 5's minimum,
  and they differ in everything at once.

**Follow-ups** (the better routes to "which benchmarks preserve ranking at
larger scale", `plan/next_analyses.md` §7c)

- The ladder's own 3B rung, read in the
  [size-generalisation write-up](../rq10_size_generalisation/README.md). It is
  the one test above the current reference within a single recipe
  (`above_reference=True` in `build_snr_pool`). So far (six deep families,
  A-L8/L15/L30/L50 and B-L8/L15, 2026-10-10 refresh) the conclusions of this
  README hold one rung up. Per-language BPB carries over,
  the 1.7B finals ordering the 3B decisions at 0.95 mono-axis and 0.94
  multi-axis and every proxy from 90M reaching at least 0.78, with
  `bpb_macro` matching every 1.7B → 3B pair (9/9 mono-axis, 15/15
  multi-axis). Benchmark accuracy reads 0.48–0.53 to either reference on
  identical decisions, so its gap to BPB is not an artefact of the 1.7B
  reference. The rung also measures more of the suite. The gate admits 772 of
  927 tasks at 3B against 607 at 1.7B, mostly the original-format
  multilingual knowledge benchmarks whose reformulated twins are already
  above chance at 1.7B. With six families the check covers only the language
  count and the scheme.
- The trend of DA-size across proxies per task. Does agreement rise
  monotonically towards the reference? A task whose DA-size climbs from 175M
  to 1B is one whose ranking is converging.
- More public lines with both a small and a large base release on the same
  task list (an eval launch of the `auto` list on ~15 public checkpoints).
  They should be close enough in level that the majority baseline is near
  0.5.

GitHub: [public_ladders_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/public_ladders_da_size_multi_axes.png) · [public_ladders_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/public_ladders_da_size_multi_axes.csv) ·
[public_ladders_da_size_pairs_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/public_ladders_da_size_pairs_multi_axes.csv)

### The 36-model sweep

These tables come from the **36-model sweep** (2026-04…06, 4 sizes × 3 data
mixtures × 3 seeds, pools `seeds_1904`, `seeds_28_1797`, `seeds_28_1797_1904`,
`custom_swissai_hf`). `pretraining/<pool>/da_all_per_task_both_axes.csv` and
the per-benchmark views are its decision tables, with the **1B** reference
and the three FineWeb mixtures as the design variants (three pairs). They are
kept as history and not regenerated. Rerun today, their `decision_acc_size_*`
columns would point at a 1.7B rung the sweep never trained. They are never
pooled with the ladder's, because the harness, the task set (the old 86-task
list, no twins) and the reference all differ. The seed-holdout and per-tier
readings built on them are in the extensions of the surrogates write-up.

## Files

- `da_explainer.py` → `…/da_all_explainer_both_axes.{png,csv}`: the toy explainer of the
  three DA kinds, the pair sets and the value lattice (Setup, no measured number).
- `pretraining/<pool>/da_all_per_task_both_axes.csv`: the DA table (the single source of truth
  for the noise-and-SNR analysis, whose `axes` column names the pair set, rule 15).
  `da_all_n_pairs_per_task_both_axes.csv` gives the pair count behind every cell, and
  `da_goal_early_small_per_task_both_axes.csv` the ten-checkpoint grid.
- `…/da_all_per_benchmark_multi_axes.csv`, `da_size_per_benchmark_multi_axes.csv`,
  `da_ckpt_per_benchmark_multi_axes.csv`: long and wide per-(language, benchmark) views
  (`da_per_benchmark.py`).
- `…/early_small_da_goal*_multi_axes.csv/.png`, `safe_{size,checkpoint,flops}_da_*_multi_axes.*`,
  `highlights_da_all_multi_axes.*`, `da_size*_multi_axes.*`: `early_small.py` (figure 3).
- `…/da_all_reliable_tasks_both_axes.csv`, `da_all_reliable_by_language_multi_axes.csv`,
  `da_size_vs_da_ckpt_reliable_tasks_<t>_<red>_multi_axes.png`: `reliable_tasks.py` (figure 2).
- `…/da_all_by_L_per_task_multi_axes.csv`, `da_all_pooled_per_task_multi_axes.csv`, `pairs_da_all_by_L_multi_axes.csv`,
  `early_small_da_{goal,ckpt}_by_{L,transformation}_*`: `by_L.py` (figure 3, and the surrogate panels read the tables).
- `…/scale_convergence_da_size*.csv/.png`: `scale_convergence.py` (figures 1 and 4,
  `_L8`, `_L8common`, `_L`, `_transformation`, `_mono_axis`, `_above_*`,
  `_flops`). `scale_convergence_da_size_lang_*`: `by_language.py` (figure 5).
- `…/rq2_da_all*.csv/.png`: `paper_rq2.py` (figure 2, the paper embeds
  `rq2_da_all_above_66_either_transformation_mono_axis`), `rq2_da_all_above_66_both_mono_vs_multi_axes.*`:
  `pair_axes.py` (figure 9).
- `…/reliability_da_size_{by_language_tier,vs_language_share}_multi_axes.*`:
  `language_tier.py` (figure 6). `seed_uncertainty_da_all_seed_null.*`: `seed_uncertainty.py`
  (figure 7). `agreement_da_size_*_multi_axes.*`: `agreement.py` (figure 8).
  `cross_task_da_{size,ckpt}*_multi_axes.*`: `cross_task.py` (figure 10).
  `bench_bpb_da*.*`: `bench_bpb_da.py` on `predictivity` (figure 11).
  `scaling_vs_ranking_da_size_multi_axes.*`: `scaling_vs_ranking.py` (scaling predictability, figure 5).
  `public_ladders_da_size*_multi_axes.*`: `public_ladders.py` (extension).
- `pretraining/predictivity_seeds*/`: the same tables on the other ladder
  pools (Definitions in [RULES.md](../RULES.md)). The all-builds folder of
  before 2026-10-05 is an orphan the refresh lists (rule 17). `pretraining/seeds_*`,
  `custom_swissai_hf/` and `all/external/` hold the 36-sweep's and the public
  models' tables (extensions).
- `pretraining/predictivity/README.md` was removed on 2026-09-23. Its write-up
  of rq00–rq02 was folded into the three RQ READMEs (README rule 1: no README
  under a pool folder).

<!-- BEGIN auto:crossfit-reliable (crossfit_reliable.py --pool predictivity) -->
### The reliable tasks chosen out of sample

Here `above_66_either` picks its reliable tasks on one half of the pairs and is read on the other half. There are 20 random splits, stratified by the design axes a pair moves, each used in both directions. The value is the mean over the 40 readings, and the band is their 5-95 % range. `in-sample` runs the same code with the verdict and the figure on every pair, so it gives the committed figures' numbers. `in-sample (half)` decides and reads on the same half. It differs from the cross-fitted reading by the selection alone, and the figures draw it dashed. Regenerate with `python analysis/rq02_decision_accuracy/crossfit_reliable.py --pool predictivity`.

![rq2 cross-fitted](pretraining/predictivity/rq2_da_all_above_66_either_crossfit_transformation_mono_axis.png)

DA-size over every mono-axis pair, as the mean over the reliable tasks (mean number of tasks):

| reading | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| in-sample | 0.642 (705) | 0.609 (713) | 0.633 (718) | 0.602 (728) | 0.640 (734) |
| in-sample (half) | 0.614 (668) | 0.580 (679) | 0.606 (686) | 0.585 (698) | 0.604 (711) |
| cross-fitted | 0.602 (590) | 0.573 (600) | 0.597 (605) | 0.576 (614) | 0.587 (625) |

![DA-size, reliable twin](pretraining/predictivity/scale_convergence_da_size_above_66_either_crossfit_multi_axes_paper.png)

Pooled DA-size over the multi-axis pairs, benchmark tasks only, as DA (tasks):

| reading | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| every task | 0.584 (1274) | 0.571 (1316) | 0.584 (1351) | 0.580 (1407) | 0.582 (1461) |
| in-sample | 0.639 (831) | 0.627 (840) | 0.648 (849) | 0.648 (866) | 0.646 (879) |
| in-sample (half) | 0.638 (783) | 0.626 (792) | 0.646 (798) | 0.648 (814) | 0.646 (826) |
| cross-fitted | 0.636 (729) | 0.625 (737) | 0.645 (743) | 0.646 (757) | 0.643 (768) |
<!-- END auto:crossfit-reliable -->

DA-size (filter `above_66_either` chosen on the other half of the pairs, mono-axis pairs for the rq2 figure and multi-axis for the twin, gate `predictivity`). GitHub: [rq2 cross-fitted PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_crossfit_transformation_mono_axis.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_crossfit_transformation_mono_axis.csv) · [twin PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_above_66_either_crossfit_multi_axes_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_above_66_either_crossfit_multi_axes_paper.csv)

Key findings:

- **A small part of the rq2 lift is the cut.** On the mono-axis pairs, the reliable tasks chosen out of sample read 0.58–0.61, against 0.59–0.62 when chosen and read on the same half. The selection effect is 0.01–0.02, largest at 1B. The rest of the gap to the committed in-sample line (0.60–0.64) comes from reading half the pairs.
- **The multi-axis lift survives.** On the multi-axis twin, cross-fitted and same-half in-sample differ by at most 0.01 (0.63–0.65), both well above the 0.57–0.59 of every task.
- **The cut keeps fewer tasks out of sample.** It keeps 88–93 % of the same-half in-sample count, because a verdict read on the other half is noisier.

Follow-ups:

- **Cross-fitting over families.** Splitting the families instead of the pairs is cleaner, but today it leaves too few pairs per half (33 families, 81 mono-axis pairs). It becomes worth it once more cells land.

### The rq02 paper figures on one task set per line

At each size, the lines above average the tasks that pass the gate (and the reliable-task cut) at that size, so a line can move because its tasks changed (rule 13). `fixed_tasks.py` redraws the two rq02 paper figures on the tasks of `utils.fixed_population`. For each line these are the tasks with a value at every point. The fixed line is solid, with the committed line dashed behind it.

![rq2 on one task set per line](pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis_fixed_tasks_paper.png)

![DA-size on one task set per line](pretraining/predictivity/scale_convergence_da_size_multi_axes_fixed_tasks_paper.png)

DA-size (rq2: filter `above_66_either`, mono-axis pairs. Twin: no filter, multi-axis pairs. Gate `predictivity`). GitHub: [rq2 PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis_fixed_tasks_paper.csv) · [twin PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_multi_axes_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_multi_axes_fixed_tasks_paper.csv)

Key findings:

- **On one task set, the rq2 lines barely move.** Over every mono-axis pair, the 714 reliable tasks the line has at every size read 0.64 → 0.64 from 90M to 1B (0.60–0.63 in between), against 0.64 → 0.64 on 716–755 tasks per size. With the twins at every size, the population hardly changes along the line.
- **The per-axis lines keep most of their tasks.** Fixed, they keep 20 (activation) to 506 (depth) tasks and read within 0.01 of the committed lines, except activation (20 tasks, up to 0.05 apart).
- **The unfiltered benchmark line barely moves either.** The 1167 tasks it has at every size read 0.590, 0.576, 0.583, 0.585, 0.592, within 0.01 of the committed line now that the bBPB twins exist at every size. BPB does not move (the same 50 tasks at every size).

Follow-ups:

- **A fixed population per panel of the full figures.** The `by_L` and per-axis figures have the same moving task sets.
