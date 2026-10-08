# Decision accuracy — Does a benchmark rank models at a small size, or early in a run, the way the reference does?

## Research question

> Decision accuracy (DA) is the ground truth of the whole framework: the
> probability that a benchmark orders a pair of models the way an evaluation
> of larger models would (Heineman et al., 2025). Which benchmarks, in which
> languages, keep the ranking of the ladder's design variants across sizes
> (**DA-size**), across training (**DA-ckpt**), and early *and* small at once
> (**DA-goal**)?

Population (Setup below): pool `predictivity`, seed 1904, every cell, ladder
and data build from 90M to 1.7B; the 26 design variants with a 1.7B run give
325 pairs at the reference, and every benchmark task is gated above chance at
the proxy and at 1.7B.

<!-- BEGIN auto:highlight (da_per_benchmark.py --pool predictivity) -->
## Highlighted result

- **DA-size, proxy → 1.7B** (mean over the above-random benchmark tasks / over the per-language BPB tasks): 90M → 1.7B 0.63 / 0.96; 175M → 1.7B 0.61 / 0.96; 350M → 1.7B 0.62 / 0.92; 600M → 1.7B 0.61 / 0.74; 1B → 1.7B 0.62 / 0.92.
- **DA-size of `bpb_macro`** (one task, kept out of the means above): 90M 0.95; 175M 0.97; 350M 0.98; 600M 0.95; 1B 0.97.
- **DA-size of `train_loss`** (one task, kept out of the means above): 90M 0.81; 175M 0.90; 350M 0.87; 600M 0.83; 1B 0.85.
- **DA-ckpt** (early checkpoint vs final, above-random benchmark tasks): highest at 350M 90 % (0.90).
<!-- END auto:highlight -->

## Key finding

**Snapshot:** ladder report 2026-10-07 15:51, outputs regenerated in
`7966367c`, prose re-read 2026-10-07, on the pool above. Every number below is
read from the CSV beside the figure it describes.

![Scale convergence, overall](pretraining/predictivity/scale_convergence_da_size_multi_axes.png)

*DA-size · no filter · multi-axis pairs · gate `predictivity`
([figure 1](#1-the-full-population-a-smaller-model-is-close-to-a-coin-flip)).*

- **A smaller fully trained model ranks the design variants close to a coin
  flip.** On benchmark accuracy alone, DA-size is 0.52, 0.54, 0.53, 0.55 and
  0.57 from 90M to 1B (mean over 293–439 gated tasks,
  `bench_bpb_da_size_summary_multi_axes.csv`); no proxy comes near τ = 0.90.
- **The pooled line in the figure adds the bBPB twins at every size and is
  flat.** It reads 0.60, 0.58, 0.59, 0.58, 0.59 (90 % jackknife ± 0.02–0.03)
  over 1117–1266 tasks, the 816 twins among them at every size, so the task
  count now moves with the gate alone.
- **Bits per byte ranks like the reference; accuracy does not.** FineWeb2 BPB
  reads 0.96, 0.96, 0.92, 0.74, 0.92 (mean over 50 per-language tasks) and
  `bpb_macro` 0.95–0.98. The gold answer's bits per byte on the same benchmark
  items (bBPB) predicts the 1.7B accuracy ranking at 0.59–0.61 from a proxy's
  final checkpoint, +0.11 over accuracy at 90M and +0.03 at 1B.
- **Reading a run early mostly measures persistence.** On accuracy alone
  DA-ckpt at 90 % of a run is 0.74–0.82 at every size (0.75 at 90M, 0.77 at
  600M), and with the twins, which now carry every checkpoint, 0.83–0.90 at
  the proxies. Two seeds of one design, which have nothing to decide, already
  read 0.71–0.73 at 90 % against 0.75–0.77 for the real pairs (175M, 600M,
  1B; 339–417 gated accuracy tasks, the seed replicates having no twin).
- **Early and small at once never works for benchmarks.** On accuracy alone
  DA-goal stays at 0.48–0.58 at every proxy and checkpoint from 0.5C to 5C
  (90M 0.51–0.53, 600M 0.50–0.55), the twins alone at 0.61–0.68, and the
  filtered paper figure at 0.58–0.65. The reference's own run reads its final
  accuracy ranking at 0.54 at 0.5C and 0.74 at 4.5C; the paper figure's
  filtered DA-size line (0.61–0.65, `above_66_either`, mono-axis) is
  conditional on the tasks that passed the cut.
- **Seeds disagree as much as sizes.** On English at 175M–1B two seeds of
  the same designs agree at 0.55–0.71, the range in which the proxy agrees
  with the reference (0.53–0.73 over the three proxy seeds).
- **Language-count regimes are unordered on any task set; the tier order
  follows the kind of score.** On accuracy alone the L8 tier leads at 90M
  (0.559 against 0.495 for L50) and the tiers sit within 0.03 at 600M and 1B,
  while with the twins, now at every size, L50 leads at every size (0.675
  against 0.576 at 1B). The token-share ρ splits the same way (+0.02 to +0.22
  on accuracy alone, −0.39 to −0.54 with the twins).
- **DA is Kendall's τ under another tie convention** (r = 0.989 over 5,954
  cells), and the convention flips the reliable-task verdict on 0.6–2.4 % of
  cells per proxy size.

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
The swiglu cells (L8, L15, L30) enter DA-ckpt but no benchmark or BPB pair
against the reference: L8's and L30's have no 1.7B run, and L15's 1.7B run
has only its training loss so far (15 loss pairs at L15,
`pairs_da_all_by_L_multi_axes.csv`).

**Benchmark-BPB twins (2026-10-07 tables).** Every benchmark task has a
`bbpb_<task>` twin, and since the per-item store was rebuilt over every
checkpoint the 816 twins exist at every size from 90M to 1.7B and at every
checkpoint of every seed-1904 cell (`da_all_per_task_both_axes.csv`). They
therefore enter DA-size, DA-ckpt and DA-goal at every fraction; the seed
replicates and the 3B cells still have no twin, so figure 7 holds no bBPB row.

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
grid seed (260 of the 325 at the reference move more than one axis at once). `mono-axis`
(`_mono_axis` stems; the multi-axis files carry `_multi_axes`): the pairs moving exactly one of L, depth,
activation, optimizer, data scheme (A/B/C, the recipe at that L), T — the
decision a practitioner makes, and what upstream's "every pair" is by
construction. `seed`: two draws of one design (the null; DA-ckpt only,
figure 7). On the 2026-10-07 report the mono-axis pairs at the grid seed are
`language count` 39, `depth` 10, `data scheme` 12 (A vs B at L8–L30, A/ZH/ES
at L2, A/DCLMP/FWEB at L1 — until 2026-10-05 three axes, language list 6,
second language 3 and English corpus 3) and `temperature` 4; the other 260 of
the 325 pairs move two or more axes. The training loss alone, which also has
the L15 swiglu run at 1.7B, counts 351 pairs, 66 of them mono-axis.

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
  families (90M→175M … 600M→1B). Multilingual tasks are only evaluated on
  cells that train the language, so each task's pair set is the families
  that exist at both sizes *and* were evaluated on it.
- **DA-ckpt** — `decision_acc_ckpt_f<frac>_<size>`: within one size, the
  ranking at the checkpoint nearest 10, 20, … 90 % of each run
  (`da_early_fracs`; 0.5C … 4.5C on the WSD schedule) vs at the final
  checkpoint.

Per-language BPB (`bpb_<subset>`) and the training loss are tasks too, so DA
is computed for the plan's outcome metric alongside the benchmarks. DA is
never gated: it is the truth the SNR proxies of the noise-and-SNR and
surrogate analyses are scored against.

## Methodology

[`compute_da.py`](compute_da.py) writes `pretraining/<pool>/da_all_per_task_both_axes.csv`
(one row per (parent task, pair set `axes`), one column per DA definition) from
`utils.pair_agreement`; [`da_per_benchmark.py`](da_per_benchmark.py)
melts it into a long (language, benchmark, comparison) table and the wide
`_size` / `_ckpt` pivots, and rewrites the deck's appendix slides for the
canonical pool. The noise-and-SNR analysis joins the SNR variants onto this
table; the design-decisions analysis asks the complementary question — which
proxy *size* ranks an intervention like the reference, with languages as the
population.

With few families at a size (26 at 1.7B with benchmark scores on the
2026-10-07 report, 27 on the training loss, every build at the grid seed; most languages only the L50 mixture trains, 15 of 19,
rest on four families and six pairs; he, ka, ml and ta, also trained by
scheme-B cells, on eight families and 28 pairs),
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

What it changed on the 2026-09-23 report: **123 of the 276 pairs at the reference** (24 families
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

The dated every-scheme numbers were `scale_convergence_da_size[_above_66_both]_multi_axes.csv`
on those dates (today's line is [figure 1](#1-the-full-population-a-smaller-model-is-close-to-a-coin-flip));
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

On the 2026-10-07 report the mono-axis pairs at the reference are `language
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
- 0.5 is a coin flip on every untied pair; since a one-sided tie is a miss, an uninformative proxy sits below it, at 0.5 × (1 − the share of pairs one side ties): ≈ 0.47 on the ladder (7% one-sided ties over the benchmark cells that pass the gate, `agreement_da_size_per_cell_multi_axes.csv`). The seed null of `seed_uncertainty.py` is a different baseline (two seeds of one design, read in its own section).
- DA-goal at the final checkpoint is DA-size, and at the reference size DA-ckpt is DA-goal: the early-and-small grid's last column and last row are the other two figures' numbers.
- `by transformation` is the mono-axis set split by the axis a pair moves; each group needs its own three pairs. On the ladder a (task, size) cell at the final checkpoint holds 3–12 for data scheme (A vs B vs C), 3–10 for depth (deep vs shallow), 3–39 for language count, 3–4 for temperature (T=1 vs T=3) (`da_all_by_transformation_per_task_mono_axis.csv`), so the smaller groups often fall below the minimum and are NaN (`early_small_da_*_by_transformation_*`, `scale_convergence_da_size_transformation_panels*`).

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
per task; the band is the 90 % leave-one-unit-out jackknife (a unit is one
of the 26 design variants on the pooled line). The task count moves with the
gate alone, the 816 bBPB twins entering at every size (Setup): 1117 at 90M,
1152 at 175M, 1188 at 350M, 1232 at 600M, 1266 at 1B.

<!-- BEGIN auto:scale-convergence (scale_convergence.py) -->
## Scale convergence — the minimum useful scale

How small a **fully trained** model may be and still decide the way the 1.7B final checkpoint does: R_size(N) = matching decisions / comparable decisions over the gated tasks, and N_min(τ) = the smallest size with R ≥ τ (τ = 0.9). Same kernel, gate and pair minimum as the rest of rq02 — this is DA-size pooled over decisions rather than averaged over tasks, so the counts behind a point are in the CSV. The 1.7B point is 1.0 by construction. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py` (all three groupings).

**Pooled over every pair** (benchmarks; `scale_convergence_da_size_multi_axes.csv` carries BPB and the decision counts):

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| all pairs | 0.6 | 0.58 | 0.59 | 0.58 | 0.59 | 1.0 | — |

![Scale convergence, overall](pretraining/predictivity/scale_convergence_da_size_multi_axes.png)

**By language count** (benchmarks; `scale_convergence_da_size_L_multi_axes.csv` carries BPB and the decision counts):

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| L1 | 0.58 | 0.6 | 0.64 | 0.6 | 0.63 | 1.0 | — |
| L15 | 0.55 | 0.52 | 0.55 | 0.53 | 0.56 | 1.0 | — |
| L2 | 0.53 | 0.51 | 0.51 | 0.53 | 0.46 | 1.0 | — |
| L30 | 0.56 | 0.55 | 0.54 | 0.55 | 0.55 | 1.0 | — |
| L50 | 0.64 | 0.59 | 0.61 | 0.58 | 0.62 | 1.0 | — |
| L8 | 0.51 | 0.51 | 0.5 | 0.52 | 0.51 | 1.0 | — |
| all pairs | 0.6 | 0.58 | 0.59 | 0.58 | 0.59 | 1.0 | — |

![Scale convergence, L](pretraining/predictivity/scale_convergence_da_size_L_multi_axes.png)

**By design axis** (benchmarks; `scale_convergence_da_size_transformation_multi_axes.csv` carries BPB and the decision counts):

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| all pairs | 0.6 | 0.58 | 0.59 | 0.58 | 0.59 | 1.0 | — |
| data scheme (A vs B vs C) | 0.52 | 0.5 | 0.48 | 0.52 | 0.54 | 1.0 | — |
| depth (deep vs shallow) | 0.55 | 0.5 | 0.51 | 0.47 | 0.5 | 1.0 | — |
| language count | 0.58 | 0.57 | 0.57 | 0.56 | 0.55 | 1.0 | — |
| temperature (T=1 vs T=3) | 0.63 | 0.6 | 0.62 | 0.62 | 0.62 | 1.0 | — |

![Scale convergence, transformation](pretraining/predictivity/scale_convergence_da_size_transformation_multi_axes.png)
<!-- END auto:scale-convergence -->

The 600M depth pairs are not size-matched (shallow has 3.7 % more non-embedding parameters, width 2048 against 1536 and about 15 % more compute; [RULES.md, "Size matching"](../RULES.md#size-matching-the-600m-depth-pairs-are-flagged-not-dropped)), so the depth axis at 600M reads depth plus a little size: the pairs stay in, the per-axis tables carry `size_matched` and the per-axis captions say so.

<!-- BEGIN auto:scale-convergence-transformation-panels (scale_convergence.py --by transformation) -->
## Scale convergence per design axis, one panel each

The `--by transformation` lines above drawn one axis per panel, with the panel's own leave-one-unit-out band and the pooled `all pairs` line faint behind it. DA-size pooled over decisions, every pair at seed 1904 (`predictivity_seeds`), gated with `predictivity`'s mask, ≥ 3 pairs per task; task counts under the points. Same table as `scale_convergence_da_size_transformation_multi_axes.csv`. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by transformation`.

![Scale convergence per design axis](pretraining/predictivity/scale_convergence_da_size_transformation_panels_multi_axes.png)

| axis | units | families | 90M R [lo, hi] (tasks) | 175M R [lo, hi] (tasks) | 350M R [lo, hi] (tasks) | 600M R [lo, hi] (tasks) | 1B R [lo, hi] (tasks) | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|---|
| all pairs | 26 | 26 | 0.60 [0.57, 0.62] (1117) | 0.58 [0.55, 0.61] (1152) | 0.59 [0.56, 0.61] (1188) | 0.58 [0.56, 0.61] (1232) | 0.59 [0.56, 0.62] (1266) | never |
| data scheme (A vs B vs C) | 12 | 18 | 0.52 [0.48, 0.57] (241) | 0.50 [0.46, 0.54] (248) | 0.48 [0.45, 0.51] (257) | 0.52 [0.47, 0.57] (264) | 0.54 [0.49, 0.58] (268) | never |
| depth (deep vs shallow) | 10 | 20 | 0.55 [0.52, 0.59] (892) | 0.50 [0.48, 0.53] (922) | 0.51 [0.50, 0.53] (953) | 0.47 [0.44, 0.50] (986) | 0.50 [0.47, 0.54] (1012) | never |
| language count | 21 | 21 | 0.58 [0.55, 0.61] (832) | 0.57 [0.54, 0.60] (859) | 0.57 [0.54, 0.60] (891) | 0.56 [0.54, 0.59] (920) | 0.55 [0.52, 0.58] (945) | never |
| temperature (T=1 vs T=3) | 4 | 8 | 0.63 [0.59, 0.67] (832) | 0.60 [0.57, 0.63] (859) | 0.62 [0.60, 0.63] (891) | 0.62 [0.60, 0.64] (920) | 0.62 [0.58, 0.65] (945) | never |

Key findings:

- **data scheme (A vs B vs C)** (12 units / 18 families): R = 0.54 at 1B [0.49, 0.58] over 268 tasks; no proxy reaches τ.
- **depth (deep vs shallow)** (10 units / 20 families): R = 0.50 at 1B [0.47, 0.54] over 1012 tasks; no proxy reaches τ.
- **language count** (21 units / 21 families): R = 0.55 at 1B [0.52, 0.58] over 945 tasks; no proxy reaches τ.
- **temperature (T=1 vs T=3)** (4 units / 8 families): R = 0.62 at 1B [0.58, 0.65] over 945 tasks; no proxy reaches τ.
- The bands are leave-one-unit-out over the units in the column — the design variant, or the pair on an axis whose pairs share no variant — not seed noise; a band on 4 units is four numbers and only says which variant or pair the line hinges on.

Follow-ups:

- The `above_66_size` twin per panel (`scale_convergence_da_size_transformation_panels_above_66_size_multi_axes.png`): the same split on the cells that rank reliably.
- A per-axis panel grid on the L8 languages only (`--langs L8`), so the language-count panel is read on one task set.
- The activation panel fills in once the swiglu cells reach 1.7B (rule 9).

Files: [`scale_convergence_da_size_transformation_panels_multi_axes.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_panels_multi_axes.png), [`scale_convergence_da_size_transformation_panels_multi_axes.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_transformation_panels_multi_axes.csv).
<!-- END auto:scale-convergence-transformation-panels -->

**Key findings**

- Pooled over every gated benchmark decision, DA-size is 0.596 [0.571,
  0.620], 0.580 [0.555, 0.606], 0.587 [0.564, 0.609], 0.582 [0.558, 0.606]
  and 0.587 [0.558, 0.617] at 90M, 175M, 350M, 600M and 1B (26 families).
  Nowhere near τ = 0.90, so N_min is undefined.
- The line is flat because every size now carries the same 816 bBPB twins:
  90M is its highest point and the five intervals overlap. On accuracy alone
  the mean over tasks rises only from 0.52 (90M, 293 tasks) to 0.57 (1B, 439
  tasks) ([figure 11](#11-benchmark-bpb-a-continuous-score-on-the-same-items)).
- The mono-axis pairs read lower: 0.575, 0.548, 0.558,
  0.546, 0.551 (`scale_convergence_da_size_mono_axis.csv`).
- Per-language BPB does far better, pooled 0.945, 0.944, 0.885, 0.785, 0.910
  over 50 tasks, and the aggregates best of all: `bpb_macro` 0.95–0.98 and
  `train_loss` 0.81–0.90 from any proxy size (highlight block). The design
  differences move most benchmark scores by less than their noise; BPB, which
  sums over every token, sees them.
- By design axis (`scale_convergence_da_size_transformation_multi_axes.csv`)
  the lines sit at 0.47–0.63: temperature highest (0.60–0.63), depth at
  0.47–0.55, data scheme 0.48–0.54. By language count the regime lines are
  unordered (figure 4).

**Follow-ups**

- Effect-size-resolved DA: per pair the reference's gap in seed sds (the
  design-decisions analysis's `seed_sd`), DA on the pairs above 2 sds and DA against the gap per proxy
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

**Left panel DA-size, middle DA-ckpt, right DA-goal · one filter for all
three panels: `above_66_either` (the `_either_transformation` stem) ·
mono-axis pairs (`_mono_axis`) · pairs from `predictivity` at seed 1904 ·
gate `predictivity`.** The paper embeds
`rq2_da_all_above_66_either_transformation_mono_axis.png` (`paper_rq2.py
--axes mono-axis`, copied to `documents/paper/figures/rq2.png` by
`documents/paper/figures/make_rq_figures.py`); its variants share the
composition and differ only in filter and pair set: `rq2_da_all_<axes>`
(no filter), `rq2_da_all_above_80_*` (the `late` reduction at 0.80, both
axes), `rq2_da_all_above_66_both[_transformation]_*` (the cells reliable on
both axes) and `rq2_da_all_above_66_own_*` (each panel its own cut, the one
variant whose panels read different populations).

![Decision accuracy, one population, mono-axis pairs](pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis.png)

*All three panels read the tasks whose median DA-size or median DA-ckpt is
≥ 0.66, so the 1.7B line of the middle panel is the 1.7B line of the right
one, and the black all-pairs line of the left panel is the 5C points of the
right one. Left: DA-size, the mean over tasks (`reliability_macro`, not the
pooled ratio), one black line over all pairs and one line per design axis in
shades of green; the data scheme (A vs B vs C) is one line on the
scheme axis; x is non-embedding parameters. Middle: DA-ckpt, each proxy
against its own final, x in Chinchilla multiples. Right: DA-goal, the same
checkpoints against the 1.7B final; the panels share the y axis, so the
vertical distance between middle and right is what the proxy's SIZE costs
on top of reading it early. A point that is 1.0 by comparing a ranking with
itself is hollow and joined by a dashed segment; there are no reference
lines. CSVs: `rq2_da_all_above_66_either_transformation_mono_axis.csv`,
`early_small_da_ckpt_by_L_above_66_either_mono_axis.csv`,
`early_small_da_goal_by_L_above_66_either_mono_axis.csv`,
`scale_convergence_da_size_transformation_above_66_either_mono_axis.csv`.*

| line (mono-axis, `above_66_either` tasks, mean over tasks) | 90M | 175M | 350M | 600M | 1B | tasks |
|---|---|---|---|---|---|---|
| all pairs | 0.648 | 0.611 | 0.634 | 0.607 | 0.643 | 706–742 |
| temperature (T = 1 vs 3) | 0.734 | 0.706 | 0.738 | 0.733 | 0.737 | 485–503 |
| language count | 0.595 | 0.602 | 0.614 | 0.625 | 0.610 | 485–503 |
| depth (deep vs shallow) | 0.590 | 0.527 | 0.548 | 0.471 | 0.529 | 520–539 |
| data scheme (A vs B vs C) | 0.501 | 0.499 | 0.494 | 0.536 | 0.516 | 127–131 |

**Key findings**

- On the filtered population the DA-size of a fully trained proxy stays at
  0.61–0.65 from 90M to 1B: a larger proxy decides no more like the 1.7B
  reference (0.648 at 90M, 0.643 at 1B). Temperature reads 0.71–0.74,
  language count 0.59–0.62, depth 0.47–0.59 and the data scheme 0.49–0.54 on
  127–131 tasks, close to a coin flip.
- The population is mostly bBPB twins: 621 of the 757 mono-axis tasks that
  pass either cut are twins. Since the twins now exist at every size, the
  line rests on 706–742 tasks at every proxy and reads one mix of scores.
- The cut selects on the quantity it plots: the filter keeps the tasks whose
  DA-size or DA-ckpt cleared 0.66. Every filtered `rq2_*` figure, including
  `above_80` whose 1B point is truncated at 0.80 by construction, is "on the
  cells that rank reliably, this is how the reliability scales"; figure 1 is
  the unconditional one.
- DA-ckpt climbs from 0.5C to 4.5C at every size, from 0.61–0.67 to
  0.86–0.92 at the proxies and 0.64 → 0.83 at 1.7B
  (`early_small_da_ckpt_by_L_above_66_either_mono_axis.csv`). Figure 7's seed
  null reaches 0.71–0.73 at 90 % of a run, so most of this rise is
  persistence, not decision.
- Under DA-goal every proxy stays at 0.58–0.65 at every checkpoint up to 5C,
  while the reference's own run reaches 0.83 at 4.5C. Training the small
  proxy longer does not close the gap; the binding constraint is its scale.
- Which tasks rank reliably (`da_all_reliable_tasks_both_axes.csv`, median
  reduction, cut 0.66): multi-axis 466 of 1335 tasks pass the DA-size cut,
  851 the DA-ckpt cut, 870 either and 447 both; without the 816 twins, 94,
  166, 182 and 78 of 519. Mono-axis 319 / 717 / 757 / 279 (77 / 107 / 136 / 48
  without the twins); the DA-ckpt cut is the permissive one because its
  median runs over up to 54 cells.

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
  90 % interval on every line: a task in most L50-only languages has six
  pairs (28 for he, ka, ml and ta), so its DA moves in steps of 1/6.
- An accuracy-only twin of the paper figure: done below, the same panels split
  by scoring.

GitHub: [rq2_da_all_above_66_either_transformation_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis.png) · [rq2_da_all_above_66_either_transformation_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis.csv) ·
GitHub: [rq2_da_all_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_multi_axes.png) · [rq2_da_all_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_multi_axes.csv) ·
GitHub: [rq2_da_all_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_mono_axis.png) · [rq2_da_all_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_mono_axis.csv) ·
GitHub: [rq2_da_all_above_80_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_80_multi_axes.png) · [rq2_da_all_above_80_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_80_multi_axes.csv) ·
GitHub: [rq2_da_all_above_66_both_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_multi_axes.png) · [rq2_da_all_above_66_both_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_multi_axes.csv) ·
GitHub: [rq2_da_all_above_66_both_transformation_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_transformation_multi_axes.png) · [rq2_da_all_above_66_both_transformation_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_transformation_multi_axes.csv) ·
GitHub: [rq2_da_all_above_66_own_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_own_multi_axes.png) · [rq2_da_all_above_66_own_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_own_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_above_66_size_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_above_66_size_mono_axis.png) · [scale_convergence_da_size_above_66_size_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_above_66_size_mono_axis.csv)

**The same figure split by scoring** (`rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.png`,
`paper_rq2.py --axes mono-axis`, the paper's `app_rq02_bbpb`). **DA-size,
DA-ckpt, DA-goal · filter `above_66_either` · mono-axis pairs · pairs from
`predictivity` at seed 1904 · gate `predictivity`.** Top row: the accuracy tasks
(originals and their `rf_`/`rfgm_` twins); bottom row: the `bbpb_` twins, read
as bBPB → 1.7B bBPB (never gated, no chance level). The two rows partition the
paper figure's tasks exactly; they are reduced from by_L's per-task tables
with by_L's `_summary`, which reproduces the paper figure's CSV to 1e-12.

![Decision accuracy by scoring, mono-axis pairs](pretraining/predictivity/rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.png)

**Key findings**

- Accuracy alone does improve with the proxy's size: DA-size over all pairs
  goes from 0.58 at 90M to 0.70 at 1B, over 85–121 tasks (the gate
  keeps more tasks at larger sizes, so the population moves along the line).
- The bBPB twins stay flat at 0.60–0.66 over the same 621 tasks at every size:
  the paper figure's flat line is mostly theirs.
- The flat bBPB line is not a bug (checked 2026-10-08): the twins cover
  every checkpoint of every seed-1904 cell but seven SwiGLU ones (3.9 % of the
  rows, 3 of the 68 mono-axis pairs), the per-task DA-size recomputes by hand
  from `bench_bpb.csv` exactly, sign agreement is direction-free, and bBPB has
  no ties. The design differences are below the noise: at 1.7B 59 % of the
  (twin, pair) differences are under one checkpoint-noise SD (rule 4 window),
  and a perfect proxy would agree with the observed 1.7B ranking on at most
  0.78–0.86 of them. Pairs under one SD agree at 0.52–0.58, pairs over four
  at 0.65–0.81.
- Depth and the data scheme sit at 0.46–0.59 because their bBPB effect is
  small and changes sign with size: at L15 scheme B has the lower bBPB on 84 %
  of twins at 90M but scheme A on 65 % at 1.7B. Temperature, the largest
  effect, reaches 0.70–0.74. On the pairs with a clear 1.7B difference
  (two detrended SD) DA-size is 0.67–0.73, still flat from 90M to 1B.
- DA-ckpt at 4.5C is 0.78–0.86 on accuracy and 0.84–0.93 on bBPB; DA-goal of
  the proxies at 5C is 0.58–0.70 on accuracy and 0.60–0.66 on bBPB.

**Follow-ups**

- The accuracy row on a fixed task set (the tasks gated in at every size), so
  its rise is not partly the gate admitting easier tasks.
- Read as bBPB → 1.7B accuracy (rq11's other reading, on the 410 twins whose
  original clears chance at 1.7B) the twins are flat too, DA-size 0.57–0.58
  from 90M to 1B (tried 2026-10-08 as a third row and dropped from the
  figure; rq11 keeps that reading).

GitHub: [rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.png) · [rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.csv)

**Per benchmark and language** (`paper_rq2.py --axes mono-axis`, the paper's
`app_rq02_da_by_language` and `app_rq02_da_size_by_language_per_proxy`). The
lines above average over every benchmark and language; these maps give each
(benchmark, language) cell. **DA-size, DA-ckpt, DA-goal · filter
`above_66_either` (paper) and none (beside it) · mono-axis pairs · pairs from
`predictivity` at seed 1904 · gate `predictivity`** (the proxy, and 1.7B where
it is the reference), on the ORIGINAL accuracy tasks only (no
`rf_`/`rfgm_`/`bbpb_` twin; the variants are rq11's figure). Read from
`da_all_per_task_both_axes.csv`; a cell is the mean over the benchmark's tasks
in the language (one task, but for the English subtask families) of each task's
mean over its readings. Rows are the benchmarks with two or more trained
languages and a value somewhere, ordered by DA-size; columns the trained
languages by resource rank, a line where the K = 1, 2, 8, 15, 30, 50 lists end
(ar and sr come from the diversity-first lists only). Grey = gated, white = no
value (or, in the filtered maps, a task the filter drops), the scale centred at
0.5.

- `rq2_da_all_by_benchmark_and_language[_above_66_either]_mono_axis.png`: one
  map per DA, DA-size averaged over the proxies 90M–1B, DA-ckpt over the
  proxies, 1.7B (DA-ckpt is defined there, against its own final) and the nine
  tenths before the final, DA-goal over the proxies and the ten tenths
  (DA-size and DA-goal stop at 1B: 1.7B is their reference); beside and under
  each map the row and column means with a 95 % bootstrap band over the cells
  (`utils.bootstrap_band`).
- `rq2_da_size_by_benchmark_and_language_per_proxy[_above_66_either]_mono_axis.png`:
  DA-size, one map per proxy, the row means in a last column. The benchmark
  with the most languages above 0.5 at every proxy is outlined (HellaSwag).

**Which variant the paper shows: the filtered one.** `above_66_either` keeps
the tasks whose median DA-size or median DA-ckpt is ≥ 0.66 (the paper rq2
figure's population), so it selects on DA and every map over it is
conditional; the paper captions say so. It is the more informative one per
benchmark and language: on it the proxy size matters (below), while the
unfiltered maps are close to a coin flip almost everywhere, so they mostly
restate figure 1.

![DA per benchmark and language, above_66_either, mono-axis pairs](pretraining/predictivity/rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis.png)

![DA-size per benchmark and language at each proxy, above_66_either, mono-axis pairs](pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis.png)

![DA per benchmark and language, no filter, mono-axis pairs](pretraining/predictivity/rq2_da_all_by_benchmark_and_language_mono_axis.png)

![DA-size per benchmark and language at each proxy, no filter, mono-axis pairs](pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_mono_axis.png)

**Key findings** (the four CSVs of the same names)

- **On the reliable tasks a larger proxy decides more like the reference.**
  With `above_66_either`, the share of cells with DA-size above 0.5 goes from
  0.52 at 90M to 0.73 at 1B (mean 0.60 to 0.71, 65 to 83 cells): XCOPA 0.58 to
  0.83, XStoryCloze 0.58 to 0.78, INCLUDE v2 (OG) 0.46 to 0.66, HellaSwag 0.66
  to 0.85. Without the filter the mean only goes from 0.53 to 0.58 (130 to 191
  cells) and the share from 0.26 to 0.40.
- **HellaSwag is the clean case** (the outlined row): its DA-size is above 0.5
  at every proxy in 16 of the 23 languages with a value at all five (ar, da,
  de, en, es, fr, hi, hr, id, it, nl, pt, ro, ru, sv, vi), in both variants.
  MultiBLiMP, the other wide benchmark (34 languages), stays at 0.57–0.60 at
  every proxy unfiltered and 0.62–0.67 filtered: there the proxy's size does
  not help.
- **Averaged over the proxies** (the mean maps), DA-size clears 0.5 (95 % band
  above it) on 8 of 11 benchmarks and 30 of 41 languages with the filter, 4 of
  14 and 15 of 49 without (HellaSwag 0.73, LAMBADA 0.64, XStoryCloze 0.63,
  MultiBLiMP 0.58). DA-goal: 9 of 11 and 31 of 41 filtered, 5 of 14 and 16 of
  49 unfiltered. DA-ckpt, now with 1.7B: 13 of 13 and 40 of 41 filtered, 14 of
  20 and 43 of 49 unfiltered (cell mean 0.69 and 0.62): the persistence that
  figure 7's seed null reads as well, and the only DA with values on Belebele
  and Global-MMLU, as it is gated at the proxy only.
- No language stands out once its cells are many: unfiltered, the languages
  with five or more DA-size cells sit between 0.48 (el, hu) and 0.63 (ca); the
  extremes rest on few cells (ta 0.37 on 4, et 0.84 on 2, sl 0.75 on 1). The
  language lists (K blocks) do not order the cells either.

**Follow-ups**

- The same maps on the RF and LLM-RF twins, where Belebele, Global-MMLU and
  INCLUDE clear the gate (rq11 has them per benchmark, not per language).
- Read the filtered rise on a fixed task set, so the gate admitting more
  tasks at 1B is not part of it.
- A multi-axis twin of every map is written beside these (`_multi_axes`).

GitHub: [rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis.png) · [rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis.csv) ·
GitHub: [rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis.png) · [rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis.csv) ·
GitHub: [rq2_da_all_by_benchmark_and_language_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_by_benchmark_and_language_mono_axis.png) · [rq2_da_all_by_benchmark_and_language_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_by_benchmark_and_language_mono_axis.csv) ·
GitHub: [rq2_da_size_by_benchmark_and_language_per_proxy_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_mono_axis.png) · [rq2_da_size_by_benchmark_and_language_per_proxy_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_size_by_benchmark_and_language_per_proxy_mono_axis.csv)

The population behind the cuts — which (benchmark, language) cells rank
reliably at all, and in which languages — is the block below.

**Reliable cells · DA-size and DA-ckpt · every cut × reduction ·
multi-axis (the table carries mono-axis too) · pairs from
`predictivity` · gate `predictivity`.** The verdicts from 2026-09-22
on gate DA-size at the reference too (rule 1): before that, three cells at
chance at 1.7B counted as reliable, and the `above_66_both` population read
23 cells; the 2026-09-23 snapshot with the twins at 175M and 350M took it
to 71. On the 2026-10-07 tables it holds 447 multi-axis tasks, 369 of them
bBPB twins.

<!-- BEGIN auto:reliable-tasks (reliable_tasks.py --pool predictivity) -->
## Which benchmark-language cells rank reliably

Per language, how many benchmarks clear DA ≥ 0.8 on DA-size (a proxy size's final ranking vs the reference's) and on DA-ckpt (an earlier checkpoint vs the same size's final), reducing each task's cells with `late` (one fixed cell per axis, so no cell is chosen by its value). Cells need ≥ 3 pairs (rule 5) and must survive the above-random gate (rule 1). The decisions come from the `predictivity` pool — every data build at the grid seed, which is the population `by_L` and `scale_convergence` pair over — while the gate and this folder stay with `predictivity`; the table carries one row per pair set (rule 15) and the figures show `multi-axis`. `da_all_reliable_tasks_both_axes.csv` holds the per-task values for every reduction and is threshold-free — each figure is one view of it. Regenerate with `python analysis/rq02_decision_accuracy/reliable_tasks.py --pool predictivity`.

| language | benchmarks evaluated | DA-size | DA-ckpt | either | both |
|---|---|---|---|---|---|
| en | 138 | 3 | 80 | 80 | 3 |
| ru | 32 | 7 | 22 | 22 | 7 |
| zh | 44 | 4 | 28 | 28 | 4 |
| de | 30 | 2 | 25 | 25 | 2 |
| ja | 24 | 0 | 14 | 14 | 0 |
| es | 126 | 4 | 71 | 71 | 4 |
| fr | 33 | 5 | 23 | 23 | 5 |
| it | 29 | 3 | 20 | 20 | 3 |
| pt | 30 | 2 | 20 | 20 | 2 |
| pl | 22 | 0 | 12 | 12 | 0 |
| nl | 23 | 2 | 15 | 15 | 2 |
| id | 26 | 5 | 14 | 14 | 5 |
| vi | 30 | 3 | 17 | 17 | 3 |
| fa | 19 | 1 | 13 | 13 | 1 |
| tr | 24 | 2 | 15 | 15 | 2 |
| th | 10 | 0 | 7 | 7 | 0 |
| uk | 24 | 2 | 16 | 17 | 1 |
| el | 26 | 1 | 14 | 14 | 1 |
| ko | 19 | 1 | 13 | 13 | 1 |
| cs | 15 | 0 | 10 | 10 | 0 |
| sv | 21 | 3 | 15 | 15 | 3 |
| hu | 22 | 4 | 13 | 13 | 4 |
| ro | 15 | 2 | 10 | 10 | 2 |
| no | 8 | 1 | 7 | 7 | 1 |
| da | 17 | 3 | 13 | 13 | 3 |
| bg | 19 | 1 | 10 | 10 | 1 |
| fi | 16 | 5 | 8 | 9 | 4 |
| hi | 36 | 4 | 23 | 23 | 4 |
| bn | 31 | 6 | 17 | 17 | 6 |
| sk | 15 | 9 | 13 | 14 | 8 |
| he | 19 | 7 | 11 | 12 | 6 |
| lt | 21 | 8 | 14 | 16 | 6 |
| bs | 2 | 1 | 1 | 1 | 1 |
| sl | 9 | 6 | 9 | 9 | 6 |
| et | 19 | 11 | 13 | 15 | 9 |
| ca | 19 | 10 | 14 | 15 | 9 |
| ta | 19 | 6 | 12 | 12 | 6 |
| hr | 19 | 7 | 13 | 14 | 6 |
| lv | 5 | 3 | 3 | 4 | 2 |
| ms | 21 | 12 | 17 | 18 | 11 |
| az | 16 | 5 | 13 | 13 | 5 |
| ka | 18 | 3 | 12 | 12 | 3 |
| ne | 28 | 11 | 19 | 19 | 11 |
| mr | 14 | 9 | 11 | 12 | 8 |
| ml | 14 | 7 | 12 | 12 | 7 |
| kk | 18 | 6 | 14 | 15 | 5 |
| ur | 26 | 8 | 22 | 23 | 7 |
| sq | 15 | 8 | 11 | 12 | 7 |
| ar | 86 | 9 | 53 | 53 | 9 |
| sr | 23 | 14 | 17 | 19 | 12 |

**How many cells pass, by cut and reduction** — the cut is a choice, and this is its whole sensitivity:

| threshold | reduction | tasks passing both | languages | benchmarks |
|---|---|---|---|---|
| 0.8 | late | 218 | 46 | bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_og, bbpb_multiblimp, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, hellaswag, include_v2_og, lambada_openai_mt, multiblimp, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_belebele, rfgm_include_base_44, xstorycloze |
| 0.8 | mean | 144 | 43 | bbpb_arc, bbpb_arc_mt, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_v2_og, bbpb_multiblimp, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, hellaswag, multiblimp, rf_global_mmlu_full |
| 0.8 | median | 168 | 44 | bbpb_arc, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_v2_og, bbpb_multiblimp, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, multiblimp, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44 |
| 0.8 | max | 385 | 50 | bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_cultural_bench_easy, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_multiblimp, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_truthfulqa-multi_mc1, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_belebele, rfgm_include_base_44, xcopa, xnli, xstorycloze |
| 0.75 | late | 277 | 47 | arc_mt, bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_og, bbpb_multiblimp, bbpb_paws, bbpb_rf_belebele, bbpb_rf_cultural_bench_easy, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, hellaswag, include_v2_og, lambada_openai_mt, multiblimp, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_belebele, rfgm_include_base_44, xstorycloze |
| 0.75 | mean | 224 | 47 | bbpb_arc, bbpb_arc_mt, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_v2_og, bbpb_multiblimp, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_v2_og, multiblimp, rf_global_mmlu_full, rfgm_include_base_44, xstorycloze |
| 0.75 | median | 238 | 48 | bbpb_arc, bbpb_arc_mt, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_v2_og, bbpb_multiblimp, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_v2_og, multiblimp, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xstorycloze |
| 0.75 | max | 511 | 50 | arc_mt, bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_cultural_bench_easy, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_multiblimp, bbpb_paws, bbpb_rf_belebele, bbpb_rf_cultural_bench_easy, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_truthfulqa-multi_mc1, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_belebele, rfgm_include_base_44, xcopa, xnli, xstorycloze, xwinograd |
| 0.66 | late | 504 | 50 | arc, arc_mt, bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_cultural_bench_easy, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_multiblimp, bbpb_paws, bbpb_rf_bbh_mcq, bbpb_rf_belebele, bbpb_rf_cultural_bench_easy, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rf_mmlu, rfgm_belebele, rfgm_include_base_44, xcopa, xstorycloze, xwinograd |
| 0.66 | mean | 410 | 50 | arc, arc_mt, bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_multiblimp, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_truthfulqa-multi_mc1, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_global_mmlu_full, rf_include_base_44, rf_mmlu, rfgm_belebele, rfgm_include_base_44, xstorycloze, xwinograd |
| 0.66 | median | 447 | 50 | arc, arc_mt, bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_multiblimp, bbpb_paws, bbpb_rf_belebele, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_truthfulqa-multi_mc1, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rf_mmlu, rfgm_belebele, rfgm_include_base_44, xcopa, xstorycloze, xwinograd |
| 0.66 | max | 830 | 50 | arc, arc_mt, bbpb_acp_bench_mcq, bbpb_arc, bbpb_arc_mt, bbpb_belebele, bbpb_blend_sample, bbpb_cultural_bench_easy, bbpb_global_mmlu_full, bbpb_global_piqa_nonparallel_cloze, bbpb_global_piqa_parallel_cloze, bbpb_hellaswag, bbpb_include_base_44, bbpb_include_v2_en, bbpb_include_v2_og, bbpb_multiblimp, bbpb_openbookqa, bbpb_paws, bbpb_rf_bbh_mcq, bbpb_rf_belebele, bbpb_rf_cultural_bench_easy, bbpb_rf_global_mmlu_full, bbpb_rf_include_base_44, bbpb_rf_mmlu, bbpb_rfgm_belebele, bbpb_rfgm_include_base_44, bbpb_truthfulqa-multi_mc1, bbpb_truthfulqa_mc2, bbpb_xcopa, bbpb_xnli, bbpb_xstorycloze, global_piqa_nonparallel_cloze, hellaswag, include_base_44, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rf_mmlu, rfgm_belebele, rfgm_include_base_44, xcopa, xnli, xstorycloze, xwinograd |

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

- At 0.8 on the `late` reduction, 218 tasks in 46 languages pass both axes,
  173 of them bBPB twins; the 45 accuracy tasks are mostly hellaswag (19),
  then multiblimp (6) and the reformulated Global-MMLU and INCLUDE twins.
- DA-size is the binding axis: 236 tasks clear it at 0.8, only 58 of them on
  accuracy, while DA-ckpt passes most of every language's benchmarks
  (English 80 of 138 against 3 on DA-size).

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
tables on `predictivity` (every build since 2026-10-05, the A/B pool before;
multi-axis, no filter, gate `predictivity`) — DA-goal at 1C–5C per proxy, the mean over tasks, and the
smallest safe level (DA ≥ 0.75 over ≥ 3 pairs, held at every larger level).

<!-- BEGIN auto:results (da_per_benchmark.py --pool predictivity) -->
## Results

Numbers from the `predictivity` pool (`da_all_per_task_both_axes.csv`, pairs from `da_all_n_pairs_per_task_both_axes.csv`, gate from rq00). Regenerate with `python analysis/rq02_decision_accuracy/da_per_benchmark.py --pool predictivity`.

**DA-size by proxy size** (`n` tasks; median pairs per cell):

| comparison | benchmarks | n | pairs | BPB | n |
|---|---|---|---|---|---|
| 90M → 1.7B | 0.63 | 1117 | 66 | 0.96 | 50 |
| 175M → 1.7B | 0.61 | 1152 | 66 | 0.96 | 50 |
| 350M → 1.7B | 0.62 | 1188 | 66 | 0.92 | 50 |
| 600M → 1.7B | 0.61 | 1232 | 66 | 0.74 | 50 |
| 1B → 1.7B | 0.62 | 1266 | 66 | 0.92 | 50 |

![DA-size by family](pretraining/predictivity/da_size_by_family_multi_axes.png)

**DA-ckpt by bucket and fraction of the run** (mean over the above-random benchmark tasks):

| bucket | 10 % | 20 % | 30 % | 40 % | 50 % | 60 % | 70 % | 80 % | 90 % |
|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.65 | 0.65 | 0.65 | 0.70 | 0.71 | 0.73 | 0.75 | 0.78 | 0.86 |
| 175M | 0.64 | 0.65 | 0.67 | 0.71 | 0.72 | 0.74 | 0.74 | 0.79 | 0.87 |
| 350M | 0.59 | 0.64 | 0.67 | 0.70 | 0.73 | 0.76 | 0.80 | 0.84 | 0.90 |
| 600M | 0.62 | 0.63 | 0.67 | 0.67 | 0.70 | 0.71 | 0.73 | 0.77 | 0.84 |
| 1B | 0.60 | 0.62 | 0.65 | 0.65 | 0.67 | 0.70 | 0.72 | 0.75 | 0.83 |
| 1.7B | 0.62 | 0.63 | 0.66 | 0.67 | 0.68 | 0.69 | 0.71 | 0.72 | 0.81 |
<!-- END auto:results -->

<!-- BEGIN auto:early-small (early_small.py --pool predictivity) -->
## Early and small, as a ranking

Numbers from the `predictivity` pool: every design variant at a proxy size, read at 1C–5C of training (C = the Chinchilla-optimal 20 tokens per parameter; every run trains 5C, so 1C is 20 % of it), ranked against the same variants at the 1.7B final checkpoint (the 5C column is DA-size, the 1.7B row is that size's DA-ckpt). A benchmark task counts only where it clears chance at the proxy size and at 1.7B. Regenerate with `python analysis/rq02_decision_accuracy/early_small.py --pool predictivity`.

- **bpb** — smallest proxy whose mean agreement with the 1.7B final ranking reaches 0.75: **90M at 0.5C** (0.95).
- **all benchmarks** — smallest proxy whose mean agreement with the 1.7B final ranking reaches 0.75: **1.7B at 4.5C** (0.81).
- **Smallest safe size per (benchmark, language)** — never: 723, 1B: 161, 90M: 113, 600M: 26, 350M: 13, 175M: 10 of 1046 cells.

![rq02 in one figure](pretraining/predictivity/highlights_da_all_multi_axes.png)

**bpb** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 50 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.95 | 0.95 | 0.95 | 0.94 | 0.94 | 0.94 | 0.94 | 0.94 | 0.96 | 0.96 |
| 175M | 0.94 | 0.92 | 0.93 | 0.96 | 0.91 | 0.94 | 0.92 | 0.93 | 0.95 | 0.96 |
| 350M | 0.83 | 0.85 | 0.85 | 0.85 | 0.90 | 0.90 | 0.88 | 0.89 | 0.91 | 0.92 |
| 600M | 0.70 | 0.72 | 0.72 | 0.73 | 0.74 | 0.74 | 0.73 | 0.74 | 0.74 | 0.74 |
| 1B | 0.76 | 0.79 | 0.82 | 0.86 | 0.88 | 0.91 | 0.90 | 0.92 | 0.92 | 0.92 |
| 1.7B | 0.91 | 0.96 | 0.97 | 0.98 | 0.99 | 0.98 | 0.99 | 0.99 | 1.00 |  |

**all benchmarks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 1310 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 90M | 0.64 | 0.63 | 0.62 | 0.61 | 0.63 | 0.62 | 0.63 | 0.63 | 0.63 | 0.63 |
| 175M | 0.63 | 0.61 | 0.61 | 0.60 | 0.60 | 0.61 | 0.59 | 0.59 | 0.60 | 0.61 |
| 350M | 0.59 | 0.59 | 0.60 | 0.60 | 0.60 | 0.61 | 0.61 | 0.61 | 0.61 | 0.62 |
| 600M | 0.60 | 0.59 | 0.60 | 0.59 | 0.61 | 0.60 | 0.59 | 0.61 | 0.61 | 0.61 |
| 1B | 0.62 | 0.60 | 0.60 | 0.59 | 0.62 | 0.61 | 0.62 | 0.62 | 0.62 | 0.62 |
| 1.7B | 0.62 | 0.63 | 0.66 | 0.67 | 0.68 | 0.69 | 0.71 | 0.72 | 0.81 |  |

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

The "all benchmarks" rows of the grid above hold the 816 bBPB twins at every
size and checkpoint beside 301–494 gated accuracy tasks (Setup), so every row
reads the same mix of scores. The Key findings below split the grid into the
two kinds by hand from `early_small_da_goal_multi_axes.csv` and
`da_all_per_task_both_axes.csv` ([Benchmark BPB](#11-benchmark-bpb-a-continuous-score-on-the-same-items)).

<!-- BEGIN auto:by-L (by_L.py --pool predictivity) -->
## Per language count

**Pairs per L** — the design variants the grid plans at seed 1904, and per proxy size the pairs usable against 1.7B (both members planned at that size and at 1.7B): planned / with data today on BPB / on the benchmarks / on the training loss. ZH and ES run to 1.7B and are the second and third L2 families. A cell below MIN_PAIRS (3) families is left empty (rule 5), so a thin L shows blanks rather than a 0/1 reading.

| L | variants | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| 1 | L1-dclmP-deep, L1-deep, L1-fweb-deep, L1-shallow | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 |
| 2 | L2-ES-deep, L2-ZH-deep, L2-deep, L2-shallow | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 |
| 8 | L8-deep, L8-schemeB-deep, L8-schemeB-shallow, L8-shallow, L8-swiglu | 10 / 6/6/6 | 10 / 6/6/6 | 10 / 6/6/6 | 10 / 6/6/6 | 10 / 6/6/6 |
| 15 | L15-AT3-deep, L15-deep, L15-schemeB-deep, L15-schemeB-shallow, L15-shallow, L15-swiglu | 15 / 10/10/15 | 15 / 10/10/15 | 15 / 10/10/15 | 15 / 10/10/15 | 15 / 10/10/15 |
| 30 | L30-AT3-deep, L30-deep, L30-schemeB-deep, L30-schemeB-shallow, L30-shallow, L30-swiglu | 15 / 10/10/10 | 15 / 10/10/10 | 15 / 10/10/10 | 15 / 10/10/10 | 15 / 10/10/10 |
| 50 | L50-AT3-deep, L50-AT3-shallow, L50-deep, L50-shallow | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 | 6 / 6/6/6 |

The early-and-small reading one L at a time: pairs of design variants that share the L (seed 1904 of every data build, `predictivity_seeds`), against the 1.7B final ranking, on the ten evaluated checkpoints of every run; a cell needs ≥ 3 pairs (rq02's rule), which today leaves out every L with one pair (the table above); the first panel pools every pair at that seed, every data build included (`da_all_pooled_per_task_multi_axes.csv`). `da_all_by_L_per_task_multi_axes.csv` also carries each size's DA-ckpt within the L (`da_own`); rq04 reads both tables. The `_mono_axis` twins of every table and figure are the same over the one-axis pairs (rule 15). Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool predictivity [--axes mono-axis]`.

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

The per-L reading above pools every design axis inside an L; this one splits the MONO-AXIS pairs at seed 1904 (`predictivity_seeds`, every data build) by the one axis each pair moves — language count, depth (deep vs shallow), activation (XIELU vs SwiGLU), data scheme (A vs B vs C), temperature (T=1 vs T=3) — one panel per axis and a first panel over every mono-axis pair (median pairs per cell up to 23). Same gate (rule 1), pair minimum (rule 5) and filter variants as the per-L figures; the `above_66_ckpt` twin filters the DA-ckpt figure and `above_66_either` both, all on the mono-axis reliability (rule 15). Task counts sit at the end of every line and the populations differ between panels and sizes (rule 13). Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool predictivity --by transformation`.

![DA-ckpt per design axis](pretraining/predictivity/early_small_da_ckpt_by_transformation_mono_axis.png)

**DA-ckpt** (against the proxy size's own final; cell = mean DA at the first → last drawn checkpoint, tasks behind the line in brackets):

| axis (DA-ckpt, 0.5C → 4.5C, tasks) | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| all | 0.61 → 0.85 (1134) | 0.59 → 0.85 (1155) | 0.56 → 0.89 (1195) | 0.59 → 0.83 (1233) | 0.56 → 0.82 (1271) | 0.58 → 0.78 (1310) |
| language count | 0.60 → 0.85 (848) | 0.59 → 0.84 (861) | 0.56 → 0.87 (893) | 0.59 → 0.83 (921) | 0.55 → 0.81 (949) | 0.55 → 0.77 (976) |
| depth (deep vs shallow) | 0.55 → 0.84 (909) | 0.53 → 0.83 (924) | 0.51 → 0.87 (955) | 0.51 → 0.79 (987) | 0.48 → 0.80 (1017) | 0.53 → 0.77 (1045) |
| activation (XIELU vs SwiGLU) | 0.56 → 0.84 (395) | — | — | — | — | — |
| data scheme (A vs B vs C) | 0.53 → 0.83 (250) | 0.55 → 0.84 (249) | 0.54 → 0.88 (257) | 0.56 → 0.84 (264) | 0.52 → 0.81 (270) | 0.51 → 0.77 (276) |
| temperature (T=1 vs T=3) | 0.64 → 0.86 (848) | 0.63 → 0.89 (861) | 0.59 → 0.90 (893) | 0.65 → 0.86 (921) | 0.62 → 0.81 (949) | 0.64 → 0.80 (976) |

Key findings:

- At 90M the DA-ckpt line clears 0.75 and stays there — language count: from 4C (0.75); depth (deep vs shallow): from 4.5C (0.84); activation (XIELU vs SwiGLU): from 4C (0.77); data scheme (A vs B vs C): from 4.5C (0.83); temperature (T=1 vs T=3): from 3.5C (0.75).
- At 1.7B — language count: 0.55 → 0.77 (976); depth (deep vs shallow): 0.53 → 0.77 (1045); activation (XIELU vs SwiGLU): —; data scheme (A vs B vs C): 0.51 → 0.77 (276); temperature (T=1 vs T=3): 0.64 → 0.80 (976).

Follow-ups:

- A `_with_bpb` variant per axis, to see whether BPB decides the temperature and the data scheme earlier than the benchmarks do.
- The same panels on the L8 languages only (`scale_convergence.py --langs L8` does it for DA-size), so the language-count panel is read on one task set.
- Once BT3 trains, the temperature panel gains the B-vs-BT3 pairs and the scheme panel AT3-vs-BT3 with no code change.

![DA-goal per design axis](pretraining/predictivity/early_small_da_goal_by_transformation_mono_axis.png)

**DA-goal** (against the 1.7B final):

| axis (DA-goal, 0.5C → 5C, tasks) | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| all | 0.60 → 0.60 (1117) | 0.58 → 0.56 (1152) | 0.55 → 0.58 (1188) | 0.56 → 0.56 (1232) | 0.57 → 0.58 (1266) | 0.58 → 0.78 (1310) |
| language count | 0.57 → 0.56 (832) | 0.57 → 0.56 (859) | 0.53 → 0.56 (891) | 0.57 → 0.57 (920) | 0.56 → 0.55 (945) | 0.55 → 0.77 (976) |
| depth (deep vs shallow) | 0.54 → 0.56 (892) | 0.53 → 0.50 (922) | 0.52 → 0.52 (953) | 0.45 → 0.47 (986) | 0.49 → 0.51 (1012) | 0.53 → 0.77 (1045) |
| data scheme (A vs B vs C) | 0.52 → 0.52 (241) | 0.51 → 0.49 (248) | 0.50 → 0.48 (257) | 0.50 → 0.51 (264) | 0.53 → 0.53 (268) | 0.51 → 0.77 (276) |
| temperature (T=1 vs T=3) | 0.64 → 0.64 (832) | 0.65 → 0.60 (859) | 0.60 → 0.62 (891) | 0.64 → 0.62 (920) | 0.64 → 0.63 (945) | 0.64 → 0.80 (976) |

Key findings:

- At 90M the DA-goal line clears 0.75 and stays there — language count: never (max 0.58); depth (deep vs shallow): never (max 0.56); data scheme (A vs B vs C): never (max 0.54); temperature (T=1 vs T=3): never (max 0.66).
- The distance between the DA-goal and DA-ckpt cell of an axis at a proxy size is what the size costs; the 1.7B column is the same line in both.

Follow-ups:

- The size axis of the same split is `scale_convergence_da_size_transformation_panels_multi_axes.png` (DA-size pooled over decisions).
- A jackknife band per axis line (leave one family out), as the scale-convergence panels carry.

Filtered twins (reliable cells only): [`early_small_da_goal_by_transformation_above_80_mono_axis.png`](pretraining/predictivity/early_small_da_goal_by_transformation_above_80_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_both_mono_axis.png`](pretraining/predictivity/early_small_da_goal_by_transformation_above_66_both_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_either_mono_axis.png`](pretraining/predictivity/early_small_da_goal_by_transformation_above_66_either_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_80_mono_axis.png`](pretraining/predictivity/early_small_da_ckpt_by_transformation_above_80_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png`](pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png`](pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_either_mono_axis.png`](pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_either_mono_axis.png)

Files: [`early_small_da_goal_by_transformation_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_mono_axis.png), [`early_small_da_goal_by_transformation_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_mono_axis.csv), [`early_small_da_goal_by_transformation_above_80_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_80_mono_axis.png), [`early_small_da_goal_by_transformation_above_80_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_80_mono_axis.csv), [`early_small_da_goal_by_transformation_above_66_both_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_both_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_both_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_both_mono_axis.csv), [`early_small_da_goal_by_transformation_above_66_either_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_either_mono_axis.png), [`early_small_da_goal_by_transformation_above_66_either_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_goal_by_transformation_above_66_either_mono_axis.csv), [`early_small_da_ckpt_by_transformation_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_mono_axis.png), [`early_small_da_ckpt_by_transformation_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_mono_axis.csv), [`early_small_da_ckpt_by_transformation_above_80_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_80_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_80_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_80_mono_axis.csv), [`early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_both_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_both_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_both_mono_axis.csv), [`early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_ckpt_mono_axis.csv), [`early_small_da_ckpt_by_transformation_above_66_either_mono_axis.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_either_mono_axis.png), [`early_small_da_ckpt_by_transformation_above_66_either_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_da_ckpt_by_transformation_above_66_either_mono_axis.csv), [`da_all_by_transformation_per_task_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_all_by_transformation_per_task_mono_axis.csv).
<!-- END auto:early-small-by-transformation -->

**Key findings**

- DA-goal for the benchmark mean never reaches 0.75 at a proxy: the grid's
  rows read 0.59–0.64 at every checkpoint from 0.5C to 5C, against 0.62 →
  0.81 for the reference's own run. Split by kind, accuracy alone reads
  0.48–0.58 at the proxies (90M 0.51–0.53, 600M 0.50–0.55, 1.7B 0.54 → 0.74)
  and the twins alone 0.61–0.68 (1.7B 0.68 → 0.85).
- The pooled-over-decisions reading of figure 1 (0.58–0.60) and this
  mean-over-tasks one (0.61–0.63 at 5C) are two estimands with one verdict.
- BPB (mean over 50 tasks) reads the 1.7B ranking at ≥ 0.75 from 90M at 0.5C
  (0.95) and stays there at 175M (0.91–0.96), 350M (0.83–0.92) and 1B
  (0.76–0.92). The 600M row never gets there (0.70–0.74), the one size where
  BPB is not a safe proxy.
- On accuracy alone DA-ckpt rises from 0.51–0.54 at 10 % to 0.75–0.82 at
  90 % at the proxies (90M 0.53 → 0.75, 600M 0.52 → 0.77), and the twins
  alone from 0.63–0.70 to 0.87–0.93. Read against the seed null of figure 7
  (0.71–0.73 at 90 % on 175M, 600M, 1B, accuracy only), the design signal at
  90 % is +0.02 to +0.04.
- Every L now draws a panel: each regime has 6 (L1, L2, L8, L50) or 10 (L15,
  L30) usable benchmark pairs against 1.7B at every proxy size; the L8 and
  L30 swiglu variants have no 1.7B run and L15's only its training loss
  (`pairs_da_all_by_L_multi_axes.csv`).
- Smallest safe level per (benchmark, language): never 723, 1B 161, 90M 113,
  600M 26, 350M 13, 175M 10 of 1046 cells; 80 of the 113 at 90M and 87 of the
  161 at 1B are bBPB twins. "Safe" with 3 pairs is a weak guarantee.

**Follow-ups**

- The `_flops` reading (`safe_flops_da_goal_multi_axes.png`) joins un-annealed and annealed
  points; one line per size with the annealed points marked, plus the
  compute frontier (the best DA reachable at or below each compute), is the
  practical answer.
- `early_small_by_L_own` (the size's cost separated from the checkpoint's) as
  an appendix pair with `early_small_da_goal_by_L_multi_axes`;
  `pairs_da_all_by_L_multi_axes.csv` justifies why per-L DA is coarse.
- An accuracy-only and a twins-only version of the grid, so the split the
  Key findings make by hand is drawn; every row now holds the twins at every
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
gate `predictivity`.** One line per language-count regime; a regime pools
depth, scheme and temperature decisions (`share_*` in the CSV) and the mix
differs by regime, which rule 5 forbids holding fixed.

<!-- BEGIN auto:scale-convergence-L8 (scale_convergence.py --by L --langs L8) -->
## Scale convergence by language count, on the L8 languages

The `--by L` lines read over the tasks in the 8 languages of the L8 setting (de, en, es, fr, it, ja, ru, zh), which every regime from L8 up trains. What this cannot fix: a regime pools arch, scheme and temperature decisions at once and the mix differs by regime (`share_*` in the CSV); rule 5 forbids holding it fixed. Under `--axes mono-axis` the regimes keep only their one-axis pairs (L1 10 → 5, L2 6 → 4, L8 10 → 5, L15 15 → 6, L30 15 → 6, L50 6 → 4) and rest on fewer tasks. Numbers below are the unfiltered population; the `above_66_size` twin (`scale_convergence_da_size_L8_above_66_size_multi_axes.png`) is the conditional one. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by L --langs L8`.

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| L1 | 0.58 | 0.6 | 0.64 | 0.6 | 0.63 | 1.0 | — |
| L15 | 0.52 | 0.49 | 0.51 | 0.51 | 0.54 | 1.0 | — |
| L2 | 0.53 | 0.51 | 0.51 | 0.53 | 0.46 | 1.0 | — |
| L30 | 0.54 | 0.52 | 0.51 | 0.52 | 0.52 | 1.0 | — |
| L50 | 0.6 | 0.57 | 0.58 | 0.55 | 0.56 | 1.0 | — |
| L8 | 0.51 | 0.51 | 0.5 | 0.52 | 0.51 | 1.0 | — |
| all pairs | 0.59 | 0.57 | 0.58 | 0.57 | 0.58 | 1.0 | — |

![Scale convergence, L8](pretraining/predictivity/scale_convergence_da_size_L8_multi_axes.png)
<!-- END auto:scale-convergence-L8 -->

<!-- BEGIN auto:scale-convergence-L8-common (scale_convergence.py --by L --langs L8 --common-tasks) -->
## Scale convergence by language count, on the L8 languages, common tasks

The `--by L` lines read over the tasks in the 8 languages of the L8 setting (de, en, es, fr, it, ja, ru, zh), which every regime from L8 up trains, and further over the tasks with ≥ 3 pairs in every regime at every proxy size — one task set for the whole figure, so a gap between lines is a gap on the same benchmarks. What this cannot fix: a regime pools arch, scheme and temperature decisions at once and the mix differs by regime (`share_*` in the CSV); rule 5 forbids holding it fixed. Under `--axes mono-axis` the regimes keep only their one-axis pairs (L1 10 → 5, L2 6 → 4, L8 10 → 5, L15 15 → 6, L30 15 → 6, L50 6 → 4) and rest on fewer tasks. Numbers below are the unfiltered population; the `above_66_size` twin (`scale_convergence_da_size_L8common_above_66_size_multi_axes.png`) is the conditional one. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by L --langs L8 --common-tasks`.

| group | 90M | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|---|
| L1 | 0.58 | 0.6 | 0.65 | 0.61 | 0.63 | 1.0 | — |
| L15 | 0.5 | 0.47 | 0.45 | 0.48 | 0.55 | 1.0 | — |
| L2 | 0.53 | 0.51 | 0.51 | 0.53 | 0.46 | 1.0 | — |
| L30 | 0.5 | 0.5 | 0.47 | 0.47 | 0.53 | 1.0 | — |
| L50 | 0.49 | 0.51 | 0.53 | 0.45 | 0.43 | 1.0 | — |
| L8 | 0.47 | 0.52 | 0.48 | 0.51 | 0.52 | 1.0 | — |
| all pairs | 0.54 | 0.54 | 0.54 | 0.53 | 0.54 | 1.0 | — |

![Scale convergence, L8 common tasks](pretraining/predictivity/scale_convergence_da_size_L8common_multi_axes.png)
<!-- END auto:scale-convergence-L8-common -->

| regime, L8-language tasks (unfiltered, pooled) | 90M | 175M | 350M | 600M | 1B | tasks | decision mix at 1B |
|---|---|---|---|---|---|---|---|
| L1 (English only) | 0.583 | 0.602 | 0.643 | 0.599 | 0.628 | 121–129 | depth 0.17, scheme 0.50, two-axis 0.33 |
| L2 | 0.528 | 0.514 | 0.513 | 0.530 | 0.456 | 121–129 | depth 0.17, scheme 0.50, two-axis 0.33 |
| L8 | 0.508 | 0.511 | 0.497 | 0.515 | 0.505 | 226–250 | depth ⅓, scheme ⅓, two-axis ⅓ |
| L15 | 0.524 | 0.492 | 0.514 | 0.510 | 0.543 | 380–429 | depth 0.22, scheme 0.16, T 0.14, two-axis 0.47 |
| L30 | 0.543 | 0.521 | 0.508 | 0.518 | 0.522 | 380–429 | depth 0.20, scheme 0.20, T 0.10, two-axis 0.50 |
| L50 | 0.598 | 0.568 | 0.582 | 0.546 | 0.562 | 380–429 | depth ⅓, T ⅓, two-axis ⅓ |
| all pairs | 0.586 | 0.573 | 0.580 | 0.574 | 0.576 | 380–429 | L 0.13, two-axis 0.77 |

*`scale_convergence_da_size_L8_multi_axes.csv`; the task counts move with the
gate alone, the bBPB twins entering at every size (380 at 90M to 429 at 1B on
the pooled line).*

**Key findings**

- The per-L lines do not order by L: on the L8 languages the English-only L1
  line is highest from 175M to 1B (0.60–0.64; 0.58 at 90M, behind L50's
  0.60), L2 is lowest at 1B (0.46), and L8, L15, L30 and L50 cross between
  0.49 and 0.60. The pooled line sits at 0.57–0.59 at every proxy.
- Fixing the task set does not order them either: on the 121 tasks common to
  every regime and size, L1 reads 0.58 → 0.64 from 90M to 1B, L8 0.47 → 0.52,
  L15 0.50 → 0.55, L30 0.50 → 0.54, L2 0.53 → 0.46 and L50 0.49 → 0.43 (not a
  trend). Filtered on `above_66_size` the lines read 0.50–0.94 on 7–107 tasks
  and are again unordered.
- The mono-axis pairs keep every L line (`pairs_da_all_by_L_mono_axis.csv`:
  on the benchmarks L1, L2, L8 and L50 keep 4 pairs, L15 and L30 5); L1 and L2
  have lines because the DCLMP/FWEB and ZH/ES cells give each four families.
- A regime line is not the effect of language count: within a regime the
  decisions are depth, scheme and temperature in a mix that differs by
  regime. The decisions within any regime are read equally poorly from a
  smaller model, on 4–5 families per regime at 1.7B.

**Follow-ups**

- An L8-only replicate of the depth and scheme decisions in every regime would
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

For each language of the L8 setting, the `--by L` lines read on that language's benchmarks alone: R at the smallest → largest proxy [tasks], per regime that trains the language, on every gated task (no selection on DA — the inference version; the `above_66_size` twin is the conditional one). The last column is the collapse test: R² of one log-linear line through every regime's points with x = model size, then with x = tokens of the language (its share of all tokens × D(N)); a rise under tokens says exposure explains what language count does not. Regimes pool arch, scheme and temperature decisions at once. Regenerate with `python analysis/rq02_decision_accuracy/by_language.py --pool predictivity`; `scale_convergence_da_size_lang_all_multi_axes_coverage.csv` says why a cell is empty.

| language | L1 | L2 | L8 | L15 | L30 | L50 | R² size / tokens |
|---|---|---|---|---|---|---|---|
| en | 0.58→0.63 [129] | 0.53→0.46 [129] | 0.47→0.52 [129] | 0.50→0.55 [129] | 0.50→0.52 [129] | 0.49→0.44 [129] | 0.00 / 0.05 |
| ru | — | — | 0.51→0.43 [30] | 0.54→0.58 [30] | 0.64→0.60 [30] | 0.64→0.72 [30] | 0.00 / 0.09 |
| zh | — | — | 0.52→0.46 [41] | 0.57→0.54 [41] | 0.62→0.57 [41] | 0.74→0.69 [41] | 0.02 / 0.23 |
| de | — | — | 0.60→0.53 [28] | 0.47→0.55 [28] | 0.56→0.57 [28] | 0.68→0.65 [28] | 0.00 / 0.06 |
| ja | — | — | 0.59→0.56 [22] | 0.54→0.45 [22] | 0.61→0.51 [22] | 0.46→0.56 [22] | 0.07 / 0.02 |
| es | — | — | — | 0.53→0.50 [120] | 0.54→0.49 [120] | 0.63→0.57 [120] | 0.03 / 0.09 |
| fr | — | — | — | 0.66→0.55 [31] | 0.49→0.49 [31] | 0.66→0.62 [31] | 0.04 / 0.09 |
| it | — | — | — | 0.68→0.67 [28] | 0.54→0.49 [28] | 0.69→0.61 [28] | 0.02 / 0.00 |

![Scale convergence per language](pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes.png)

![Scale convergence per language, tokens axis](pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_tokens.png)
<!-- END auto:scale-convergence-by-language -->

**Key findings**

- Every one of the eight languages draws a panel and none is ordered by L:
  the pooled line at 1B reads en 0.54 (129 tasks), ru 0.68 (30), zh 0.65
  (41), de 0.66 (28), ja 0.58 (22), es 0.58 (120), fr 0.61 (31) and it 0.60
  (28), and the regime lines within a panel cross at every size. Spanish,
  French and Italian have no L8 line (scheme B's L8 list does not contain
  them, so their L8 cells have one pair).
- English sits at the bottom at every size (0.53–0.54 from 90M to 1B) and
  Russian at the top (0.67–0.70). The task counts now move with the gate
  alone, the twins entering at every size (en 121 tasks at 90M against 129 at
  1B).
- Exposure does not collapse the lines: the collapse R² is 0.00–0.07 under
  size and 0.00–0.23 under tokens, and tokens beat size for six languages by
  at most 0.21 (zh 0.02 → 0.23). If reliability were a function of how many
  of a language's tokens the models saw, the tokens axis would order the
  points; it does not.

**Follow-ups**

- Every regime line rests on 3–5 families (4–5 for en, ru, zh, de, ja; the
  L15 lines of es, fr and it on 3), so a per-language line moves in coarse
  steps; a bootstrap over families would say which crossings are real.

GitHub: [scale_convergence_da_size_lang_all_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes.png) · [scale_convergence_da_size_lang_all_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_lang_all_multi_axes_tokens.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_tokens.png) · [scale_convergence_da_size_lang_all_multi_axes_tokens.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_tokens.csv) ·
[scale_convergence_da_size_lang_all_multi_axes_coverage.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_all_multi_axes_coverage.csv) ·
GitHub: [scale_convergence_da_size_lang_above_66_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes.png) · [scale_convergence_da_size_lang_above_66_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes.csv) ·
GitHub: [scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.png) · [scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_lang_above_66_size_multi_axes_tokens.csv)

### 6. Language tier: the order follows the task population

**DA-size · `reliability_da_size_by_language_tier_multi_axes`: panel (a) no filter, panel (b)
`above_66_size` · multi-axis · pairs from `predictivity` · gate
`predictivity`; `reliability_da_size_vs_language_share_multi_axes`: no filter, per language with
≥ 3 gated tasks (rule 8).** A tier is the smallest scheme-A regime that trains
the language (L8 = the eight high-resource languages, L50 = the twenty only
the L50 mixture trains).

<!-- BEGIN auto:language-tier (language_tier.py --pool predictivity) -->
## Decision reliability by language tier

The pooled `all pairs` line of `scale_convergence.py` read over the gated tasks of one language TIER — the smallest scheme-A regime that trains the language (L8: the eight high-resource languages every regime trains; L50: the twenty only the L50 mixture trains). Reliability at the smallest → largest proxy [task count]. `reliability_da_size_vs_language_share_multi_axes.png` is the per-language version: reliability against the language's share of the L50 mixture, Spearman ρ over languages 175M -0.39, 1B -0.54, 350M -0.48, 600M -0.39, 90M -0.46. Both are unfiltered; a tier also differs in benchmark mix. Regenerate with `python analysis/rq02_decision_accuracy/language_tier.py --pool predictivity`.

| tier | all gated tasks | above_66_size |
|---|---|---|
| L8 languages (in every regime) | 0.59 → 0.58 [380–429 tasks] | 0.74 → 0.73 [100–107 tasks] |
| L15-only languages | 0.61 → 0.60 [199–228 tasks] | 0.74 → 0.75 [64–69 tasks] |
| L30-only languages | 0.61 → 0.61 [253–288 tasks] | 0.74 → 0.76 [98–101 tasks] |
| L50-only languages | 0.69 → 0.68 [285–321 tasks] | 0.81 → 0.81 [177–189 tasks] |

![Reliability by language tier](pretraining/predictivity/reliability_da_size_by_language_tier_multi_axes.png)

![Reliability against language share](pretraining/predictivity/reliability_da_size_vs_language_share_multi_axes.png)
<!-- END auto:language-tier -->

**Key findings**

- On accuracy alone the high-resource tier reads best at 90M: the L8 tier is
  0.559 against 0.511 (L15), 0.497 (L30) and 0.495 (L50), on 53–99 tasks; at
  600M and 1B the tiers sit within 0.03 (0.529–0.557 and 0.557–0.571). These
  accuracy-only values are pooled by hand from `early_small_da_goal_multi_axes.csv`
  without the `bbpb_` rows; the figure, which now carries the twins at every
  size, has no accuracy-only interval.
- With the twins the order reverses at every size: the L50-only tier reads
  0.643–0.692 (0.692 [0.63, 0.75] at 90M, 0.675 [0.60, 0.75] at 1B) and L8
  0.573–0.586 (0.576 [0.55, 0.60] at 1B), on 199–429 tasks. Filtered on
  `above_66_size` the tiers read 0.70–0.81 at every size, L50 highest
  (0.77–0.81).
- The token share follows the same split: Spearman ρ between a language's
  share of the L50 mixture and its reliability is −0.39 to −0.54 at every size
  over 49 languages in the figure (−0.46 at 90M, −0.54 at 1B), but +0.02 to
  +0.22 on accuracy alone over 46–48 (+0.22 at 90M, +0.02 at 1B). The sign
  follows the kind of score, so neither reads as a resource effect.
- The tiers also differ in decision mix (15 of the 19 L50-only languages are read
  from the four L50 families alone, six pairs that are mostly temperature and
  depth decisions; he, ka, ml and ta, also trained by scheme-B cells, from
  eight families and 28 pairs) and in benchmark mix.

**Follow-ups**

- An accuracy-only and a twins-only version of both panels, with intervals,
  since the sign of the tier effect flips with the kind of score.
- The same decision set in every tier (rule 5 forbids it on this grid; the
  L8-only replicate of figure 4's follow-up would allow it).
- A per-family version of panel (a), to separate benchmark mix from tier.

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

What the replicate seeds say about decision accuracy — English at three proxy sizes and Russian at 1B, the only (size, language) cells where three replicated designs give ≥ 3 cross-L pairs under rule 2. Per row: DA of those decisions against the 1.7B final with the proxy at each of its three seeds (proxy-side noise), and the DA between two seeds' rankings of the same designs at that size (the reference-side ceiling). Three pairs put a DA on {0, ⅓, ⅔, 1}: read the spread, not a mean. The seed null (DA-ckpt over pairs of two seeds of one design, mean over fractions and gated tasks) is 175M: 0.57 vs real 0.61; 600M: 0.56 vs real 0.62; 1B: 0.56 vs real 0.61. Regenerate with `python analysis/rq02_decision_accuracy/seed_uncertainty.py --pool predictivity`.

| size | language | designs | DA at the 3 proxy seeds | test-retest (3 seed pairs) | tasks |
|---|---|---|---|---|---|
| 175M | en | 3 | 0.59, 0.73, 0.65 | 0.64–0.69 | 25 |
| 600M | en | 3 | 0.67, 0.68, 0.59 | 0.55–0.71 | 31 |
| 1B | en | 4 | 0.63, 0.53, 0.58 | 0.56–0.62 | 31 |
| 1B | ru | 4 | 0.83, 0.83, 0.81 | 0.81–0.89 | 12 |

![Seed uncertainty](pretraining/predictivity/seed_uncertainty_da_all_seed_null.png)
<!-- END auto:seed-uncertainty -->

**Key findings**

- Proxy-seed noise is up to ± 0.07 on English: the three seeds read 0.59 /
  0.73 / 0.65 at 175M (75 decisions on 25 tasks), 0.67 / 0.68 / 0.59 at 600M
  (93 decisions, 31 tasks) and 0.63 / 0.53 / 0.58 at 1B (186 decisions, 31
  tasks); Russian at 1B reads 0.83 / 0.83 / 0.81 on 12 tasks. Three pairs put
  a DA on {0, ⅓, ⅔, 1}: read the spread, not a mean.
- The English test-retest ceiling is low at every size: 0.64–0.69 at 175M,
  0.55–0.71 at 600M and 0.56–0.62 at 1B (Russian 0.81–0.89). On these 25–31
  English tasks two seeds of the same designs disagree with each other as
  much as the proxy disagrees with the reference.
- DA-ckpt is mostly within-run persistence: two seeds of one design, which
  have nothing to decide, read 0.50 at 10 % of the 175M run and 0.73 at 90 %,
  against 0.54 and 0.76 for the real pairs (600M 0.47 → 0.73 against 0.52 →
  0.77; 1B 0.48 → 0.71 against 0.52 → 0.75). At 90 % the real pairs exceed
  the null by +0.02 to +0.04, the gap is largest mid-run (+0.08 at 50 % of
  175M, +0.09 at 40 % of 600M, +0.06 at 30 % of 1B), and the null reads 339,
  417 and 362 gated accuracy tasks, the seed replicates having no bBPB twin
  (`seed_uncertainty_da_all_seed_null.csv`).
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
size) cell, median 12 models per cell; 5,955 cells, 4,080 of them the bBPB twins'.**

<!-- BEGIN auto:agreement-measures (agreement.py --pool predictivity) -->
## Decision accuracy is Kendall's τ under another tie convention

Over the 5,955 (benchmark task, proxy size) cells of DA-size's population (every pair of the grid-seed variants, ≥ 3 pairs, above chance at the proxy and at 1.7B): 2·DA − 1 = τ_a + (T_both − T_one)/n exactly, where T_both / T_one are the pairs tied on both / one side. Ties are 10,836 of 530,169 pairs (2.0%), 95% of them one-sided, and touch 24% of the cells — so the two statistics correlate at r = 0.989 by construction, and the number with content is how often the tie convention changes a reliability verdict (DA ≥ 0.66, i.e. τ ≥ 0.32), in % of cells per proxy size:

| statistic | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| da | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| da_drop_ref_ties | 0.6 | 0.8 | 1.1 | 1.0 | 1.4 |
| gamma | 2.0 | 1.3 | 1.6 | 2.4 | 2.2 |
| tau_a | 1.4 | 1.2 | 1.1 | 1.7 | 1.7 |
| tau_b | 1.6 | 1.2 | 1.5 | 2.1 | 1.9 |

Spearman ρ and Pearson r on the raw scores are the two statistics that are NOT a rescaling — they weight a pair by its displacement — and sit at r = 0.973 and 0.893 against DA. Median 12 models per cell. Values in `agreement_da_size_per_cell_multi_axes.csv`; regenerate with `python analysis/rq02_decision_accuracy/agreement.py --pool predictivity`.

![DA, Kendall tau and Spearman rho against each other](pretraining/predictivity/agreement_da_size_correlation_multi_axes.png)

![DA against Kendall's tau](pretraining/predictivity/agreement_da_size_identity_multi_axes.png)

![Cut sensitivity](pretraining/predictivity/agreement_da_size_cut_sensitivity_multi_axes.png)
<!-- END auto:agreement-measures -->

**Key findings**

- The three statistics are one statistic: over the 5,954 cells where all
  three are defined r(DA, τ_b) = 0.989, r(DA, ρ) = 0.973 and r(τ_b, ρ) = 0.982
  (Spearman 0.988, 0.981, 0.989; `agreement_da_size_correlation_multi_axes.csv`).
  The relation to τ_a is exact, 2·DA − 1 = τ_a + (T_both − T_one)/n on every
  cell, so DA and τ_a differ only in how tied pairs are counted.
- Ties are 2.0 % of the pairs, touch 24 % of the cells, and 95 % are
  one-sided. The twins never tie (0 ties over their 4,080 cells), so on
  accuracy alone ties are 7.1 % of the pairs (10,836 of 152,809).
- The convention moves the verdict on 0.6–2.4 % of cells against the DA ≥
  0.66 set: τ_a flips 1.1–1.7 %, τ_b 1.2–2.1 %, γ 1.3–2.4 % and DA with the
  reference's ties dropped 0.6–1.4 % per proxy size, most at 600M and 1B; the
  twins, which never tie, now dilute the rate at every size. The correlations
  are guaranteed by the identity; the flip rate is the number with content,
  and a published reliable-task list depends on the tie convention at the
  margin.

**Follow-ups**

- Report the reliable-task list under two conventions (DA and γ) in the
  paper's appendix, or state the flip rate beside it.

GitHub: [agreement_da_size_correlation_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_correlation_multi_axes.png) · [agreement_da_size_correlation_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_correlation_multi_axes.csv) ·
GitHub: [agreement_da_size_identity_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_identity_multi_axes.png) · [agreement_da_size_identity_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_identity_multi_axes.csv) ·
GitHub: [agreement_da_size_cut_sensitivity_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_cut_sensitivity_multi_axes.png) · [agreement_da_size_cut_sensitivity_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_cut_sensitivity_multi_axes.csv) ·
[agreement_da_size_per_cell_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_da_size_per_cell_multi_axes.csv)

### 9. Multi-axis against mono-axis pairs

**`rq2_da_all_above_66_both_mono_vs_multi_axes`: the three definitions · filter `above_66_both`
(447 tasks at 1B, 369 of them bBPB twins; the one place it is drawn, for the pair-set comparison only) ·
rows multi-axis and mono-axis · pairs from `predictivity` · gate
`predictivity`.** Exploratory (`pair_axes.py`); it fed `plan/decision_accuracy.md`
and the `axes` column of rule 15.

![Multi-axis against mono-axis pairs](pretraining/predictivity/rq2_da_all_above_66_both_mono_vs_multi_axes.png)

`pair_axes.py` computes the three decision accuracies on the `above_66_both`
cells twice: over every pair at the grid seed (multi-axis, this analysis's
convention) and over the pairs that move exactly one of L, depth, activation,
data scheme and temperature (mono-axis, what DataDecide's "all pairs" are by
construction). The proposal that followed from it (an `axes` column in
`da_all_per_task_both_axes.csv`, mono-axis as the headline for decisions) is
in `plan/decision_accuracy.md` (§2, §6).

**Key findings**

- DA-size reads 0.07–0.08 lower under the mono-axis pairs:
  0.682, 0.666, 0.674, 0.654, 0.671 against 0.748, 0.742, 0.748, 0.736, 0.748
  from 90M to 1B, and DA-goal at 5C likewise (`rq2_da_all_above_66_both_mono_vs_multi_axes.csv`).
- The mono-axis pairs are the stricter and smaller set: 28 % of the
  decisions (7,106 of 25,278 at 175M), on 431–447 tasks at every size now
  that the twins carry every size and checkpoint. DA-ckpt reads 0.01–0.08
  lower under them too, the gap closing towards 4.5C (0.84–0.92 against
  0.87–0.93 at 4.5C).

**Follow-ups**

- Mono-axis as the headline for decisions (the paper's left panel already
  draws it); the multi-axis pooled line as the population statement.

GitHub: [rq2_da_all_above_66_both_mono_vs_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_mono_vs_multi_axes.png) · [rq2_da_all_above_66_both_mono_vs_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_both_mono_vs_multi_axes.csv)

### 10. Cross-task predictability

**DA-size and DA-ckpt · no filter · multi-axis pairs of `predictivity` (every
build: 27 variants at 1.7B and 351 pairs counting the L15 swiglu run, whose
1.7B cell has only the training loss, 26 and 325 on the benchmarks; DA-ckpt
pools the within-size pairs of every size, 2,410) · gate `predictivity` on
both sides.** Every parent
task as the proxy for every other one; a cell is the smallest size or earliest
checkpoint at which x's ranking safely predicts y's final ranking (DA ≥ 0.75
over ≥ 3 pairs, held at every larger level).

<!-- BEGIN auto:cross-task (cross_task.py --pool predictivity) -->
## Cross-task predictability

Every parent task as the proxy for every other one (1714 x 1714): the cell is the smallest proxy size (DA-size, 27 variants at 1.7B, 351 pairs) or the earliest checkpoint (DA-ckpt, the within-size pairs of every size pooled, 2410 pairs, the nine checkpoints before the final) at which the ranking on task x (columns) safely predicts the final ranking on task y (rows): DA >= 0.75 over >= 3 pairs there and at every larger level with a value. The diagonal is rq02's own-task DA; the gate empties a benchmark's pairs at every size where it is at chance. The `_by_family` maps take the median level over the task pairs of two benchmarks, the `_by_language` maps over the same-benchmark task pairs of two languages (resource order of the scheme-A lists). Regenerate with `python analysis/rq02_decision_accuracy/cross_task.py --pool predictivity`.

The same two maps over the benchmark tasks that are above chance at some size — BPB, the loss and the benchmarks the gate finds at chance everywhere are dropped, so what is left is the sub-map where a transfer result is possible at all: [`cross_task_da_size_benchmarks_multi_axes.png`](pretraining/predictivity/cross_task_da_size_benchmarks_multi_axes.png), [`cross_task_da_ckpt_benchmarks_multi_axes.png`](pretraining/predictivity/cross_task_da_ckpt_benchmarks_multi_axes.png).

![Cross-task DA-size by benchmark](pretraining/predictivity/cross_task_da_size_by_family_multi_axes.png)

![Cross-task DA-ckpt by benchmark](pretraining/predictivity/cross_task_da_ckpt_by_family_multi_axes.png)

![Cross-task DA-size by language](pretraining/predictivity/cross_task_da_size_by_language_multi_axes.png)

![Cross-task DA-ckpt by language](pretraining/predictivity/cross_task_da_ckpt_by_language_multi_axes.png)

Full task-level maps: [`cross_task_da_size_multi_axes.png`](pretraining/predictivity/cross_task_da_size_multi_axes.png), [`cross_task_da_ckpt_multi_axes.png`](pretraining/predictivity/cross_task_da_ckpt_multi_axes.png).
<!-- END auto:cross-task -->

**Key findings**

- Cross-task predictability is almost absent between accuracies: of the
  6,480 off-diagonal benchmark-family pairs of the DA-size map, 32 reach a
  median safe size (17 at 1B, 6 at 90M, 3 each at 175M, 350M and 600M), and
  on average 3.8 % of a family pair's task pairs reach one, 53 % never and
  43 % are gated (`cross_task_da_size_by_family_multi_axes.csv`).
- Every one of the 32 has BPB, the loss or a bBPB twin on at least one side,
  most often `global_piqa_nonparallel_cloze` (accuracy or twin) and
  per-language BPB. No accuracy → accuracy family pair is safe at any size.
- The DA-ckpt map, now with the twins at every checkpoint, says the same: 39
  of the 6,480 family pairs reach a median safe checkpoint (37 at 0.5C, 2 at
  4.5C), every one with BPB, the loss or a twin on a side, and on average 2.0 % of a
  family pair's task pairs reach one, 58 % never and 40 % are gated
  (`cross_task_da_ckpt_by_family_multi_axes.csv`).

**Follow-ups**

- The family maps show the median of the pairs that succeed while most pairs
  never do; draw the share of task pairs reaching a safe size (in the CSV) as
  the cell and keep the median level as the small number.
- Same-L blocks share the design variants, so a same-list transfer is partly
  by construction: compare against a permutation baseline (the cross-task DA
  of two languages whose L lists are shuffled).

GitHub: [cross_task_da_size_by_family_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_family_multi_axes.png) · [cross_task_da_size_by_family_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_family_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_by_family_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_family_multi_axes.png) · [cross_task_da_ckpt_by_family_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_family_multi_axes.csv) ·
GitHub: [cross_task_da_size_by_language_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_language_multi_axes.png) · [cross_task_da_size_by_language_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_by_language_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_by_language_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_language_multi_axes.png) · [cross_task_da_ckpt_by_language_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_by_language_multi_axes.csv) ·
GitHub: [cross_task_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_multi_axes.png) · [cross_task_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_multi_axes.png) · [cross_task_da_ckpt_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_multi_axes.csv) ·
GitHub: [cross_task_da_size_benchmarks_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_benchmarks_multi_axes.png) · [cross_task_da_size_benchmarks_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_size_benchmarks_multi_axes.csv) ·
GitHub: [cross_task_da_ckpt_benchmarks_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_benchmarks_multi_axes.png) · [cross_task_da_ckpt_benchmarks_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_da_ckpt_benchmarks_multi_axes.csv)

### 11. Benchmark BPB: a continuous score on the same items

**DA-size · no filter · multi-axis pairs of `predictivity` (26
variants at 1.7B, 325 pairs) · final checkpoints only · gate `predictivity`
on the accuracy side only.** Figures 1–10 read every benchmark through its
accuracy, a step function of the log-likelihoods the harness already
computed, with the twins added at every size and checkpoint (Setup). This figure reads the same items, the same final checkpoints and the same
pairs through the gold answer's bits-per-byte (**bBPB**). Does a continuous
score rank like the reference where accuracy is a coin flip?

How bBPB is computed (no model is re-run; `build_per_item_store.py` +
`bench_bpb_da.py`):

- **Source.** The lm-eval samples files of every checkpoint (the store holds
  all of them since 2026-10-07; this figure reads the finals),
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

DA-size, final checkpoints, multi-axis pairs of `predictivity` (406 of its 435 pairs: those between families the store holds, for every reading and the FineWeb2 tick), gate `predictivity` on the accuracy side only: every reading counts the tasks above chance at 1.7B, and acc → acc also needs the task above chance at the proxy (its task count is the smaller one). The paired gain is bBPB → acc minus acc → acc on the tasks where both are defined. FineWeb2 val BPB is `bpb_macro`'s DA-size on the same pairs. Regenerate with `python analysis/rq02_decision_accuracy/bench_bpb_da.py --pool predictivity` (after `build_per_item_store.py --pool predictivity --finals-only`). The store `predictivity` lacks 7 of the pool's models (lm-1.7B-L15-swiglu-seed1904, lm-175M-L30-b168-swiglu-seed1904, lm-1B-L8-swiglu-seed1904, lm-350M-L8-swiglu-seed1904, lm-600M-L15-swiglu-seed1904, lm-600M-L8-swiglu-seed1904, lm-90M-L1-b84-swiglu-seed1904); they are left out until the store is rebuilt for the pool.

**all benchmarks**

| proxy | acc → acc (tasks) | bBPB → acc (tasks) | bBPB → bBPB | paired gain (tasks) | bBPB better / worse | Wilcoxon p | FineWeb2 val BPB |
|---|---|---|---|---|---|---|---|
| 90M | 0.52 (293) | 0.61 (483) | 0.70 | +0.11 (293) | 71% / 20% | <0.001 | 0.95 |
| 175M | 0.54 (327) | 0.59 (483) | 0.65 | +0.06 (327) | 59% / 33% | <0.001 | 0.97 |
| 350M | 0.53 (362) | 0.60 (483) | 0.68 | +0.07 (362) | 63% / 28% | <0.001 | 0.98 |
| 600M | 0.55 (405) | 0.60 (483) | 0.66 | +0.05 (405) | 57% / 35% | <0.001 | 0.95 |
| 1B | 0.57 (439) | 0.60 (483) | 0.68 | +0.03 (439) | 53% / 36% | <0.001 | 0.97 |

**all cloze** (the answer text is the continuation)

| proxy | acc → acc (tasks) | bBPB → acc (tasks) | bBPB → bBPB | paired gain (tasks) | bBPB better / worse | Wilcoxon p | FineWeb2 val BPB |
|---|---|---|---|---|---|---|---|
| 90M | 0.52 (293) | 0.61 (476) | 0.70 | +0.11 (293) | 71% / 20% | <0.001 | 0.95 |
| 175M | 0.54 (327) | 0.59 (476) | 0.66 | +0.06 (327) | 59% / 33% | <0.001 | 0.97 |
| 350M | 0.53 (362) | 0.60 (476) | 0.69 | +0.07 (362) | 63% / 28% | <0.001 | 0.98 |
| 600M | 0.55 (403) | 0.60 (476) | 0.66 | +0.05 (403) | 57% / 35% | <0.001 | 0.95 |
| 1B | 0.57 (438) | 0.60 (476) | 0.68 | +0.03 (438) | 53% / 36% | <0.001 | 0.97 |

**all lettered** (the continuation is the letter: bBPB is the letter's surprisal)

| proxy | acc → acc (tasks) | bBPB → acc (tasks) | bBPB → bBPB | paired gain (tasks) | bBPB better / worse | Wilcoxon p | FineWeb2 val BPB |
|---|---|---|---|---|---|---|---|
| 90M |  (0) | 0.45 (7) | 0.48 | +nan (0) | nan% / nan% |  | 0.95 |
| 175M |  (0) | 0.46 (7) | 0.50 | +nan (0) | nan% / nan% |  | 0.97 |
| 350M |  (0) | 0.53 (7) | 0.56 | +nan (0) | nan% / nan% |  | 0.98 |
| 600M | 0.58 (2) | 0.51 (7) | 0.58 | +0.03 (2) | 50% / 50% | 1.000 | 0.95 |
| 1B | 0.33 (1) | 0.42 (7) | 0.73 | +0.00 (1) | 0% / 0% |  | 0.97 |

**Per benchmark**, mean over the five proxy sizes (tasks: the parent tasks with bBPB; cells: task-mean DA over those above chance at 1.7B, blank = all gated):

| benchmark | tasks | acc → 1.7B acc | bBPB → 1.7B acc | bBPB → 1.7B bBPB |
|---|---|---|---|---|
| acp_bench_cloze | 7 |  |  |  |
| arc | 28 | 0.60 | 0.65 | 0.63 |
| arc_mt | 11 | 0.58 | 0.63 | 0.64 |
| bbh_cloze | 6 |  |  |  |
| bbh_mcq | 17 |  |  |  |
| global_piqa_nonparallel_cloze | 2 | 0.81 | 0.75 | 0.90 |
| global_piqa_parallel_cloze | 63 |  | 0.50 | 0.53 |
| hellaswag | 26 | 0.76 | 0.82 | 0.87 |
| include_v2_en | 77 | 0.48 | 0.49 | 0.52 |
| include_v2_og | 77 | 0.54 | 0.59 | 0.64 |
| mathqa | 1 | 0.52 | 0.55 | 0.53 |
| multiblimp | 34 | 0.65 | 0.66 | 0.69 |
| openbookqa | 1 |  |  |  |
| paws | 8 | 0.55 | 0.54 | 0.69 |
| acp_bench_mcq-rf | 7 | 0.43 | 0.50 | 0.54 |
| bbh_mcq-rf | 17 | 0.47 | 0.46 | 0.54 |
| belebele-rf | 59 | 0.49 | 0.58 | 0.71 |
| commonsense_qa-rf | 1 | 0.54 | 0.60 | 0.61 |
| cultural_bench_easy-rf | 19 | 0.35 | 0.43 | 0.54 |
| global_mmlu_full-rf | 29 | 0.56 | 0.68 | 0.64 |
| include_base_44-rf | 36 | 0.54 | 0.63 | 0.70 |
| mmlu-rf | 1 | 0.66 | 0.70 | 0.64 |
| belebele-rfgm | 59 | 0.50 | 0.59 | 0.74 |
| include_base_44-rfgm | 36 | 0.55 | 0.64 | 0.80 |
| toxigen | 1 |  |  |  |
| truthfulqa-multi_mc1 | 2 |  |  |  |
| truthfulqa_mc2 | 3 |  |  |  |
| xcopa | 8 | 0.49 | 0.63 | 0.75 |
| xnli | 15 | 0.51 | 0.53 | 0.70 |
| xstorycloze | 8 | 0.66 | 0.74 | 0.86 |
| acp_bench_mcq (letter) | 7 |  |  |  |
| belebele (letter) | 59 |  | 0.44 | 0.57 |
| blend_sample (letter) | 5 |  |  |  |
| commonsense_qa (letter) | 1 |  |  |  |
| cultural_bench_easy (letter) | 19 |  |  |  |
| global_mmlu_full (letter) | 29 |  | 0.43 | 0.53 |
| include_base_44 (letter) | 36 | 0.46 | 0.52 | 0.58 |
| mmlu (letter) | 1 |  |  |  |

![bBPB DA, overall](pretraining/predictivity/bench_bpb_da_size_bars_multi_axes.png)

![bBPB DA, per benchmark](pretraining/predictivity/bench_bpb_da_size_bars_benchmarks_multi_axes.png)

![bBPB DA, heat map](pretraining/predictivity/bench_bpb_da_size_heatmap_multi_axes.png)
<!-- END auto:bench-bpb -->

Snapshot: ladder report 2026-10-07 15:51, outputs regenerated in `7966367c`;
bBPB from the per-item store `predictivity` (rebuilt over every checkpoint,
finals read here), 816 parent tasks with bBPB, the seven swiglu pool models
the store lacks left out (listed in the auto block). This figure reads the
store directly and has bBPB at all five proxy sizes, as the twins of
figures 1–10 now do too. Raw per-task DAs and both gate flags: [`bench_bpb_da_size_multi_axes.csv`](pretraining/predictivity/bench_bpb_da_size_multi_axes.csv);
every number below is in [`bench_bpb_da_size_summary_multi_axes.csv`](pretraining/predictivity/bench_bpb_da_size_summary_multi_axes.csv).

**Key findings**

- **bBPB predicts the 1.7B accuracy ranking better than accuracy does.**
  - Paired over the tasks where both readings are defined, bBPB → acc minus
    acc → acc is +0.11 at 90M (0.61 against 0.52; bBPB better on 71 % of
    tasks, worse on 20 %).
  - The gain shrinks with the proxy size: +0.06 at 175M, +0.07 at 350M, +0.05
    at 600M and +0.03 at 1B, where it is better on 53 % and worse on 36 %.
    Wilcoxon p < 0.001 at every size.
- **It reads more tasks.** bBPB → acc is defined on 483 tasks at every proxy
  size. acc → acc is defined on 293 at 90M and 439 at 1B, because accuracy
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
    0.56 → 0.68, xcopa 0.49 → 0.63, include_base_44-rf 0.54 → 0.63 and
    belebele-rf 0.49 → 0.58.
  - hellaswag (0.76 → 0.82) and xstorycloze (0.66 → 0.74) were already
    reliable and stay the best.
  - The include_v2 English questions do not move (0.48 → 0.49).
  - The Gemini-rewritten twins rank themselves best on bBPB: include_base_44
    rfgm 0.80, belebele rfgm 0.74.
- **The lettered tasks give no reading.**
  - Only 7 of 157 lettered tasks are above chance at 1.7B, and their
    continuation is a letter.
  - bBPB → acc there is 0.42–0.53, i.e. a coin flip. The `rf_` twins are the
    way to read those items continuously.

**Follow-ups**

- ~~**The full checkpoint grid.**~~ Done on 2026-10-07: the store holds every
  checkpoint of every seed-1904 cell, so the twins carry bBPB's DA-ckpt and
  DA-goal into figure 3 (DA-goal 0.61–0.68 at the proxies, DA-ckpt 0.87–0.93
  at 90 % of a run) and its SNR into the noise-and-SNR analysis.
- ~~**Feed bBPB into the shared pipeline.**~~ Done: `utils.with_bbpb_twins`
  puts a `bbpb_<task>` twin in every `build_snr_pool` population, so
  `compute_da.py`, the scale-convergence figure and the SNR tables already
  read it; this figure stays as the head-to-head of the three readings.
- **xwinograd and lambada.** Their `target` is the answer string, not a
  choice index, so they have no bBPB yet.
  - xwinograd's gold index is the doc's `answer` field.
  - lambada has one continuation, so its gold is that continuation.
  - Both are a change to `gold_index` and a re-extract of those two families.
- **Seed null.** Repeat on `predictivity_seeds` to place bBPB's 0.61 against
  the two-seeds-of-one-design null of figure 7, as DA-ckpt was; it waits on a
  twin for the 20 seed-replicate models, which the store does not hold yet.
- **The reference's two rankings.** The DA between the 1.7B bBPB and the
  1.7B accuracy rankings, per task, bounds what bBPB → acc can reach and
  separates the likelihood-vs-accuracy gap from proxy noise.
- **Cross-reading.** bBPB → acc against figure 10's cross-task map: a task's
  bBPB as the proxy for another task's accuracy.

GitHub: [bench_bpb_da_size_bars_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_bars_multi_axes.png) · [bench_bpb_da_size_bars_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_bars_multi_axes.csv) ·
GitHub: [bench_bpb_da_size_bars_benchmarks_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_bars_benchmarks_multi_axes.png) · [bench_bpb_da_size_bars_benchmarks_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_bars_benchmarks_multi_axes.csv) ·
GitHub: [bench_bpb_da_size_heatmap_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_heatmap_multi_axes.png) · [bench_bpb_da_size_heatmap_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/bench_bpb_da_size_heatmap_multi_axes.csv)

### 12. Read next in the other RQs

Three decision-accuracy readings live where their figures belong; the auto
block of the first stays here because its script reads this folder's per-task table.

- **Scaling cleanly and ranking like the reference are different
  properties.** Across tasks, the scaling analysis's ρ of score with size
  correlates with the proxy's ranking ρ at only +0.33 (175M) to +0.41 (1B);
  the trajectory R² is the better surrogate (+0.49 to +0.57 with DA-size), on
  832–919 tasks, the twins included at every size (auto block below). Write-up:
  [scaling predictability, figure 5](../rq01_scaling_predictability/README.md#5-scaling-cleanly-and-ranking-like-the-reference-are-different-properties).
- **FineTasks' selection criteria, judged by the reference they cannot
  see.** Their pass rates and the DA-size of the tasks they pass are read in
  the surrogates write-up:
  [FineTasks' criteria](../rq04_surrogates/README.md#finetasks-criteria-on-the-ladder).
- **The minimal language panel.** Which single language, or which macro,
  best predicts the 1.7B macro ranking over the L8 panel is read in the
  language-transfer write-up:
  [the minimal language panel](../rq06_language_transfer/README.md#the-minimal-language-panel).

<!-- BEGIN auto:scaling-vs-ranking (scaling_vs_ranking.py --pool predictivity) -->
## Scaling utility against ranking utility

Per proxy size, the Spearman correlation ACROSS TASKS between rq01's scaling statistics (ρ of score with model size; R² of the size fit; R² of the trajectory fit) and rq02's ranking statistics (ρ of the proxy's ranking of the design variants with the 1.7B ranking; DA-size), and the mean DA-size of the tasks rq01 calls 'predictable across both' against the rest. Regenerate with `python analysis/rq02_decision_accuracy/scaling_vs_ranking.py --pool predictivity`.

| proxy | tasks | ρ_size vs ρ_ranking | R²_size vs DA-size | R²_traj vs DA-size | DA-size, predictable both | DA-size, other regimes |
|---|---|---|---|---|---|---|
| 90M | 832 | +0.40 | +0.50 | +0.55 | 0.63 | 0.53 |
| 175M | 859 | +0.33 | +0.42 | +0.54 | 0.61 | 0.51 |
| 350M | 891 | +0.39 | +0.46 | +0.56 | 0.62 | 0.51 |
| 600M | 919 | +0.36 | +0.40 | +0.49 | 0.62 | 0.51 |
| 1B | 919 | +0.41 | +0.46 | +0.57 | 0.63 | 0.51 |

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

gemma-3, Qwen3-Base and OLMo-2 each have a base model near 1B and near 13B in the external tier. Per task above chance at both buckets, DA is the share of the three line pairs the small models order like the large ones: pooled 0.73 over 54 tasks, against the ladder's own 1B → 1.7B DA-size of 0.64 on the same tasks. Per line pair: Qwen3 vs OLMo-2 0.87 against a majority baseline of 0.80 (11 minority tasks, DA there 0.55); gemma-3 vs OLMo-2 0.76 against a majority baseline of 0.81 (10 minority tasks, DA there 0.80); gemma-3 vs Qwen3 0.57 against a majority baseline of 0.56 (24 minority tasks, DA there 0.79). The majority baseline is what a proxy scores by always naming the line that usually wins at 12–14B, so only DA above it is information the small models add. Between-lab decisions on public sizes, not the ladder's one-axis ones; three pairs, so per-task values are a lattice. Regenerate with `python analysis/rq02_decision_accuracy/public_ladders.py --pool predictivity`.

| family | tasks | DA public lines (1B–1.7B → 12–14B) | DA ladder (1B → 1.7B) |
|---|---|---|---|
| multiblimp | 7 | 0.67 | 0.58 |
| global_piqa_completions | 5 | 0.73 | — |
| hellaswag | 5 | 0.67 | 0.88 |
| truthfulqa | 5 | 0.67 | — |
| xnli | 5 | 0.80 | 0.49 |
| xstorycloze | 5 | 0.87 | 0.68 |
| xcopa | 4 | 0.83 | 0.66 |
| xwinograd | 4 | 0.75 | 0.59 |
| belebele | 3 | 0.67 | — |

![Public ladders](pretraining/predictivity/public_ladders_da_size_multi_axes.png)
<!-- END auto:public-ladders -->

**Key findings**

- The pooled 0.73 over 54 tasks is within 0.07 of a majority-order baseline:
  a proxy that always names the line that usually wins at 12–14B scores 0.80
  / 0.81 / 0.56 on the three pairs (DA +0.07, −0.05, +0.01 against it). The
  figure shows lab-level differences (Qwen3 above OLMo-2 on four tasks in
  five), not task-level ranking preservation.
- On the 10–24 minority tasks, where the reference order is the uncommon
  one, the small models read it at 0.55–0.80 (on too few tasks to be a
  finding); over all 54 tasks the public lines pool to 0.73 against the
  ladder's own 1B → 1.7B DA-size of 0.64 on the same tasks.
- As evaluated today the public tier cannot say whether benchmarks read
  decisions above 1.7B.

**Follow-ups** (the better routes to "which benchmarks preserve ranking at
larger scale"; `plan/next_analyses.md` §7c)

- The ladder's own 3B rung, read in the
  [size-generalisation write-up](../rq10_size_generalisation/README.md): the
  one within-recipe generalisation above the current reference
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
built on them are in the surrogates write-up's extensions.

## Files

- `da_explainer.py` → `…/da_all_explainer_both_axes.{png,csv}` — the toy explainer of the
  three DA kinds, the pair sets and the value lattice (Setup; no measured number).
- `pretraining/<pool>/da_all_per_task_both_axes.csv` — the DA table (single source of truth
  for the noise-and-SNR analysis; the `axes` column names the pair set, rule 15);
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
  `early_small_da_{goal,ckpt}_by_{L,transformation}_*` — `by_L.py` (figure 3; the surrogate panels read the tables).
- `…/scale_convergence_da_size*.csv/.png` — `scale_convergence.py` (figures 1, 4;
  `_L8`, `_L8common`, `_L`, `_transformation`, `_mono_axis`, `_above_*`,
  `_flops`); `scale_convergence_da_size_lang_*` — `by_language.py` (figure 5).
- `…/rq2_da_all*.csv/.png` — `paper_rq2.py` (figure 2; the paper embeds
  `rq2_da_all_above_66_either_transformation_mono_axis`), `rq2_da_all_above_66_both_mono_vs_multi_axes.*` —
  `pair_axes.py` (figure 9).
- `…/reliability_da_size_{by_language_tier,vs_language_share}_multi_axes.*` —
  `language_tier.py` (figure 6); `seed_uncertainty_da_all_seed_null.*` — `seed_uncertainty.py`
  (figure 7); `agreement_da_size_*_multi_axes.*` — `agreement.py` (figure 8);
  `cross_task_da_{size,ckpt}*_multi_axes.*` — `cross_task.py` (figure 10);
  `bench_bpb_da*.*` — `bench_bpb_da.py` on `predictivity` (figure 11);
  `scaling_vs_ranking_da_size_multi_axes.*` — `scaling_vs_ranking.py` (scaling predictability, figure 5);
  `public_ladders_da_size*_multi_axes.*` — `public_ladders.py` (extension).
- `pretraining/predictivity_seeds*/` — the same tables on the other ladder
  pools (Definitions in [RULES.md](../RULES.md)); the all-builds folder of
  before 2026-10-05 is an orphan the refresh lists (rule 17); `pretraining/seeds_*`,
  `custom_swissai_hf/`, `all/external/` — the 36-sweep's and the public
  models' (extensions).
- `pretraining/predictivity/README.md` — removed 2026-09-23: its write-up of
  rq00–rq02 was folded into the three RQ READMEs (README rule 1: no README
  under a pool folder).

<!-- BEGIN auto:crossfit-reliable (crossfit_reliable.py --pool predictivity) -->
### The reliable tasks chosen out of sample

`above_66_either` decided on one half of the pairs and read on the other (20 random splits stratified by the design axes a pair moves, both directions; the value is the mean over the 40 readings, the band its 5-95 % range). `in-sample` is the same code with the verdict and the figure on every pair, i.e. the committed figures' numbers; `in-sample (half)` decides and reads on the same half, so it differs from the cross-fitted reading by the selection alone (the figures draw it dashed). Regenerate with `python analysis/rq02_decision_accuracy/crossfit_reliable.py --pool predictivity`.

![rq2 cross-fitted](pretraining/predictivity/rq2_da_all_above_66_either_crossfit_transformation_mono_axis.png)

DA-size over every mono-axis pair, mean over the reliable tasks (mean number of tasks):

| reading | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| in-sample | 0.579 (90) | 0.604 (716) | 0.590 (411) | 0.639 (113) | 0.641 (743) |
| in-sample (half) | 0.574 (92) | 0.590 (628) | 0.584 (357) | 0.618 (121) | 0.612 (658) |
| cross-fitted | 0.564 (77) | 0.579 (562) | 0.573 (339) | 0.591 (101) | 0.593 (587) |

![DA-size, reliable twin](pretraining/predictivity/scale_convergence_da_size_above_66_either_crossfit_multi_axes_paper.png)

Pooled DA-size over the multi-axis pairs, benchmarks (tasks):

| reading | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| every task | 0.543 (298) | 0.580 (1154) | 0.564 (915) | 0.553 (413) | 0.587 (1267) |
| in-sample | 0.644 (112) | 0.630 (815) | 0.645 (569) | 0.688 (148) | 0.643 (854) |
| in-sample (half) | 0.644 (106) | 0.631 (750) | 0.643 (472) | 0.685 (138) | 0.643 (785) |
| cross-fitted | 0.642 (95) | 0.629 (689) | 0.640 (457) | 0.678 (122) | 0.639 (719) |
<!-- END auto:crossfit-reliable -->

DA-size (filter `above_66_either` chosen on the other half of the pairs, mono-axis pairs for the rq2 figure and multi-axis for the twin, gate `predictivity`). GitHub: [rq2 cross-fitted PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_crossfit_transformation_mono_axis.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_crossfit_transformation_mono_axis.csv) · [twin PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_above_66_either_crossfit_multi_axes_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_above_66_either_crossfit_multi_axes_paper.csv)

Key findings:

- **A small part of the rq2 lift is the cut.** On the mono-axis pairs, the reliable tasks chosen out of sample read 0.56–0.59, against 0.57–0.62 when chosen and read on the same half; the selection effect is 0.01–0.03, largest at 600M and 1B. The rest of the gap to the committed in-sample line (0.58–0.64) comes from reading half the pairs.
- **The multi-axis lift survives.** On the multi-axis twin, cross-fitted and same-half in-sample differ by at most 0.01 (0.63–0.68), both well above the 0.54–0.59 of every task.
- **The cut keeps fewer tasks out of sample.** 83–95 % of the same-half in-sample count, because a verdict read on the other half is noisier.

Follow-ups:

- **Cross-fitting over families.** Splitting the families instead of the pairs is cleaner but leaves too few pairs per half today (30 families, 75 mono-axis pairs); worth it once more cells land.

### The rq02 paper figures on one task set per line

The lines above average, at each size, the tasks that pass the gate (and the reliable-task cut) there, so a line can move because its tasks changed (rule 13). `fixed_tasks.py` redraws the two rq02 paper figures on `utils.fixed_population`'s tasks: per line, the tasks with a value at every point, solid, with the committed line dashed behind it.

![rq2 on one task set per line](pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis_fixed_tasks_paper.png)

![DA-size on one task set per line](pretraining/predictivity/scale_convergence_da_size_multi_axes_fixed_tasks_paper.png)

DA-size (rq2: filter `above_66_either`, mono-axis pairs; twin: no filter, multi-axis pairs; gate `predictivity`). GitHub: [rq2 PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis_fixed_tasks_paper.csv) · [twin PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_multi_axes_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_multi_axes_fixed_tasks_paper.csv)

Key findings:

- **On one task set, the rq2 lines rise more steadily.** Over every mono-axis pair, the 90 reliable tasks the line has at every size read 0.58 → 0.68 from 90M to 1B, against 0.58 → 0.64 on 90–743 tasks per size; the 350M dip is the population, not the size.
- **The per-axis lines rest on few tasks.** Fixed, they keep 18 (data scheme) to 57 (depth) tasks, so their wiggles are within task-sampling noise.
- **The unfiltered benchmark line becomes smooth.** The 296 tasks it has at every size read 0.544, 0.552, 0.551, 0.565, 0.587; the 175M bump of the committed line (0.580) is the bBPB twins that exist only at 175M, 350M and 1B. BPB does not move (the same 50 tasks at every size).

Follow-ups:

- **A fixed population per panel of the full figures.** The `by_L` and per-axis figures have the same moving task sets.
