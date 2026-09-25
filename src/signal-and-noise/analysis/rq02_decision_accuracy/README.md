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

- **DA-size, proxy → 1.7B** (mean over the above-random benchmark tasks / over the per-language BPB tasks): 175M → 1.7B 0.52 / 0.78; 350M → 1.7B 0.50 / 0.82; 600M → 1.7B 0.51 / 0.58; 1B → 1.7B 0.52 / 0.82.
- **DA-size of `bpb_macro`** (one task, kept out of the means above): 175M 0.91; 350M 0.97; 600M 0.92; 1B 0.94.
- **DA-size of `train_loss`** (one task, kept out of the means above): 175M 0.81; 350M 0.86; 600M 0.81; 1B 0.83.
- **DA-ckpt** (early checkpoint vs final, above-random benchmark tasks): highest at 175M 90 % (0.83).
<!-- END auto:highlight -->

**In one paragraph** (ladder-report snapshot **2026-09-23 06:16**, the one
the tables on disk were built from; every number below is read from the CSV
beside the figure it describes). Ranking design variants from a smaller
fully trained model is close to a coin flip on the full gated population
(0.53 at 175M, 0.56 at 1B, jackknife ± 0.03); the 0.60 → 0.76 rise of the
paper's filtered figure is a conditional statement on the tasks whose
DA-size cleared 0.66. Ranking from an early checkpoint of the same run
reaches 0.86 at 90 % of training, but two seeds of one design reach 0.81 on
the same axis: most of DA-ckpt's rise is within-run persistence, and on
English at 600M and 1B two seeds of the same designs disagree with each
other (test-retest 0.54–0.71) as much as the proxy disagrees with the
reference. Restricting to the eight high-resource languages, to a common
task set, or to one language or one language tier does not order the per-L
lines, and a language's token share does not predict how reliably its
benchmarks rank (Spearman ρ −0.07 to 0.20 over 36–45 languages). Decision
accuracy, Kendall's τ and Spearman's ρ are one statistic (r ≥ 0.961 over
1 286 cells), and the tie convention alone moves the reliable-task verdict
on 2.5–6 % of cells.

## Setup

**Pools and sizes.** Models are the predictivity ladder's cells (pool
`predictivity`: 175M–1.7B, the 90M rung dropped at load, rule 10;
L ∈ {1, 2, 8, 15, 30, 50} × deep/shallow × scheme A/B, seed 1904).
`predictivity_schemes` adds the AT3, ZH and ES cells at the grid seed and is
the pair set every decision figure from figure 1 on is computed over
(`reliable_tasks.py`, `by_L.py`, `scale_convergence.py` and the extensions;
rule 15); `predictivity_seeds` adds the replicate seeds as separate models
(figure 7). The gate pool is `predictivity` everywhere; the output folder is
`pretraining/predictivity/`. The reference is 1.7B (rule 9).

**Three decision accuracies.** DA is the share of design-variant pairs a
proxy orders the same way as the reference; a pair tied on both sides counts
as agreement and a pair tied on one side as a miss (the kernel's one departure
from upstream, below). **DA-size** compares the proxy size's final checkpoint
with the 1.7B final; **DA-ckpt** an early checkpoint of a run with that run's
own final; **DA-goal** an early checkpoint of the proxy with the 1.7B final.
Two identities are verified on every run: DA-size equals DA-goal at 100 % of
the proxy's run, and DA-ckpt of the 1.7B run equals DA-goal of the 1.7B run.
Pooled figures (`scale_convergence*`, `rq2_*`, `reliability_*`) report
matching decisions over comparable decisions summed over tasks; the
`early_small*` and `by_L` families report the mean over tasks. The two agree
to two decimals on today's tables but are different estimands.

**Pair sets (rule 15).** `multi-axis`: every pair of design variants at the
grid seed (two thirds move more than one axis at once). `mono-axis`
(`_one_axis` stems): the pairs moving exactly one of L, depth, list, T,
lang2 — the decision a practitioner makes, and what upstream's "every pair"
is by construction. `seed`: two draws of one design (the null; DA-ckpt only,
figure 7). At the reference the mono-axis pairs are `language count` 39,
`depth` 10, `language list` 6, `temperature` 4 and `second language` 3; the
other 214 of the 276 pairs move two or more axes.

**Filters.** An `above_*` stem keeps the tasks whose DA cleared a cut on a
reduction of their cells (`reliable_tasks.py`): `above_66_size` /
`above_66_ckpt` / `above_66_either` (median DA over the proxy cells ≥ 0.66
on that axis, or on either), `above_66_both` (both axes; never quoted, RULES.md),
`above_80` (0.80 on the `late` reduction, both axes). Every filtered figure
plots the quantity it selected on and is read as conditional (bug #17,
`../CLAUDE.md`); the plain stem is the unconditional one.

## Experimental setup

Models are the predictivity ladder's cells (`configs/models.json` pool
`predictivity`: 175M–1.7B (the 90M rung is dropped at load, rule 10) ×
L ∈ {1, 2, 8, 15, 30, 50} × deep/shallow × scheme A/B, seed 1904;
`predictivity_seeds` adds the seed replicates as separate models;
`predictivity_schemes` adds the AT3, ZH and ES cells at the grid seed and is
the pool `reliable_tasks.py`, `by_L.py` and `scale_convergence.py` pair over,
rule 15). The cross-size identity is the cell's `family`
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

[`compute_da.py`](compute_da.py) writes `pretraining/<pool>/da_per_task.csv`
(one row per parent task, one column per DA definition) from
`snr.metrics.decision_acc_fast`; [`da_per_benchmark.py`](da_per_benchmark.py)
melts it into a long (language, benchmark, comparison) table and the wide
`_size` / `_ckpt` pivots, and rewrites the deck's appendix slides for the
canonical pool. rq03 joins the SNR variants onto this table; rq05 asks the
complementary question — which proxy *size* ranks an intervention like the
reference, with languages as the population.

With few families at a size (18 A/B variants at 1.7B, 24 with every scheme;
a task in a language only the L50 mixture trains rests on three or four),
DA is quantised to 1/#pairs: read the
family-level averages, and `n` alongside every value
(`da_n_pairs_per_task.csv`).

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
The A/B filter was suppressing measured convergence. On the 2026-09-23 tables
(the `rf_` twins at 175M and 350M, ZH and ES at 1.7B) the every-scheme line reads
**0.531, 0.536, 0.544, 0.557** (`scale_convergence.csv`, +0.034 per decade) and
0.637 → 0.772 on `above_66_both` (+0.172); the A/B counterfactual has not been
re-measured on them.

The every-scheme numbers are `scale_convergence[_above_66_both].csv` as it stands;
the A/B numbers are a counterfactual, since no A/B-only table is kept. Reproduce it
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
| BT3 | B | 3 | ru | B's list at T = 3 (registered 2026-09-21; no BT3 cell in the 2026-09-23 report) |
| ZH | A | 1 | zh | A's list with Chinese in the second slot |
| ES | A | 1 | es | A's list with Spanish in the second slot |

Two consequences, both of which were wrong before 2026-09-22:

**ZH and ES are not list designs.** At L = 2 every setting is "English + one other
language"; scheme A's second language is simply Russian. So A, ZH and ES share a
list and differ on `lang2` alone, and A's L2 cell is the `ru` level of the
second-language axis rather than a scheme of its own. They used to be three
separate one-pair groups; they are now one 3-pair axis, and all three reach the
reference in the 2026-09-23 report (ZH and ES at 1.7B) — exactly `MIN_PAIRS`.
`scale_convergence_transformation.csv` still draws no line for it: the axis is
0.1 % of the pooled decisions (`share_lang2`).

**B vs AT3 moves two axes, not one.** Under the old `scheme` key it differed on a
single key and was dropped only because no label existed for the level pair — the
right outcome for the wrong reason, and the same accident hid ZH vs ES. Every axis
now has a label and nothing is dropped for want of one; a pair is dropped when, and
only when, it moves more than one axis.

At the reference the mono-axis pairs are `language count` 39, `depth` 10,
`language list` 6, `temperature` 4 (the L15, L30 and L50 AT3 cells against their
scheme-A twins; the line became drawable once the L15 and L30 AT3 cells reached
1.7B) and `second language` 3; the other 214 of the 276 pairs move two or more
axes. When **BT3** lands, `B vs BT3` joins the temperature axis and `AT3 vs BT3`
the language-list axis with no code change.

<!-- BEGIN auto:da-explainer (da_explainer.py --pool predictivity) -->
### Decision accuracy on a toy ladder

**Toy, not measured.** Four labelled variants (Deep-A-T1, Deep-B-T1, Shallow-A-T1, Deep-A-T3) with hand-written scores; every DA in the figure is the pipeline's kernel run on them. (a) the finals per size and (b) one run along training show the two ways a ranking moves; (c) decides the six pairs three ways over both pair sets of rule 15 and rings the values at or above the reliability cut; (d) is where each definition sits on the size × checkpoint grid, with the two identities. Regenerate with `python analysis/rq02_decision_accuracy/da_explainer.py --pool predictivity`.

![Decision accuracy explained on a toy ladder](pretraining/predictivity/da_explainer.png)

Key findings (definitions, so nothing to measure):

- On the toy, DA-size 0.50, DA-ckpt 0.17 and DA-goal 0.67 over the six multi-axis pairs; over the three mono-axis pairs 0.33 / 0.00 / 0.67. The tied pair (Deep-A-T1 = Deep-A-T3 at the proxy's final) is a miss wherever the other side decides it, an agreement only if both sides tie — `decision_acc_fast`'s convention, order-invariant.
- A cell of n pairs takes the values k/n: at the minimum of 3 pairs that is 0, ⅓, ⅔, 1, so a per-cell DA is read on its lattice and the figures draw the pooled ratio over tasks (`scale_convergence.py`) or the mean over cells (`by_L.py`), never one cell. The ringed cells (DA-goal multi-axis 0.67, DA-goal mono-axis 0.67) are the ones the `above_66_*` filters would keep (cut 0.66).
- 0.5 is a coin flip on every untied pair; since a one-sided tie is a miss, an uninformative proxy sits below it — ≈ 0.47 on the ladder (7 % of pairs tied, the seed null of `seed_uncertainty.py`).
- DA-goal at the final checkpoint is DA-size, and at the reference size DA-ckpt is DA-goal: the early-and-small grid's last column and last row are the other two figures' numbers.
- `by transformation` is the mono-axis set split by the axis a pair moves; each group needs its own three pairs. On the ladder a (task, size) cell holds 0–4 pairs for the temperature axis, 0–6 for the list, 0–10 for depth and 0–39 for the language count, so the temperature and depth groups often fall below the minimum and are NaN (`early_small_by_transformation_*`, `scale_convergence_transformation_panels*`).

Follow-ups:

- A measured twin: the same panels on one real task (`hellaswag_de`, say) with the ladder's families, so the toy orders become the observed ones.
- The lattice of the ladder's actual pair counts per cell (`median_pairs` in the scale-convergence CSVs), to show how coarse a per-cell DA is on each axis.

Files: [`da_explainer.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_explainer.png), [`da_explainer.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_explainer.csv).
<!-- END auto:da-explainer -->

## Figures, in storyline order

### 1. The full population: a smaller model is close to a coin flip

**DA-size · no filter · multi-axis pairs (`_one_axis` twin: mono-axis) ·
pairs from `predictivity_schemes` at seed 1904 · gate `predictivity`.**
Pooled over decisions, gated at the proxy and at the reference, ≥ 3 pairs
per task; the band is the 90 % leave-one-family-out jackknife.

<!-- BEGIN auto:scale-convergence (scale_convergence.py) -->
## Scale convergence — the minimum useful scale

How small a **fully trained** model may be and still decide the way the 1.7B final checkpoint does: R_size(N) = matching decisions / comparable decisions over the gated tasks, and N_min(τ) = the smallest size with R ≥ τ (τ = 0.9). Same kernel, gate and pair minimum as the rest of rq02 — this is DA-size pooled over decisions rather than averaged over tasks, so the counts behind a point are in the CSV. The 1.7B point is 1.0 by construction. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py` (all three groupings).

**Pooled over every pair** (benchmarks; `scale_convergence.csv` carries BPB and the decision counts):

| group | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| all pairs | 0.53 | 0.54 | 0.54 | 0.56 | 1.0 | — |

![Scale convergence, overall](pretraining/predictivity/scale_convergence.png)

**By language count** (benchmarks; `scale_convergence_L.csv` carries BPB and the decision counts):

| group | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| L15 | 0.54 | 0.53 | 0.51 | 0.54 | 1.0 | — |
| L2 | 0.39 | 0.58 | 0.53 | 0.48 | 1.0 | — |
| L30 | 0.54 | 0.48 | 0.52 | 0.49 | 1.0 | — |
| L50 | 0.51 | 0.52 | 0.53 | 0.57 | 1.0 | — |
| L8 | 0.47 | 0.49 | 0.46 | 0.57 | 1.0 | — |
| all pairs | 0.53 | 0.54 | 0.54 | 0.56 | 1.0 | — |

![Scale convergence, L](pretraining/predictivity/scale_convergence_L.png)

**By design axis** (benchmarks; `scale_convergence_transformation.csv` carries BPB and the decision counts):

| group | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| all pairs | 0.53 | 0.54 | 0.54 | 0.56 | 1.0 | — |
| depth (deep vs shallow) | 0.53 | 0.48 | 0.48 | 0.51 | 1.0 | — |
| language count | 0.52 | 0.53 | 0.53 | 0.53 | 1.0 | — |
| language list (A vs B) | 0.49 | 0.49 | 0.48 | 0.49 | 1.0 | — |
| temperature (T=1 vs T=3) | 0.49 | 0.55 | 0.57 | 0.57 | 1.0 | — |

![Scale convergence, transformation](pretraining/predictivity/scale_convergence_transformation.png)
<!-- END auto:scale-convergence -->

<!-- BEGIN auto:scale-convergence-transformation-panels (scale_convergence.py --by transformation) -->
## Scale convergence per design axis, one panel each

The `--by transformation` lines above drawn one axis per panel, with the panel's own leave-one-family-out band and the pooled `all pairs` line faint behind it. DA-size pooled over decisions, every pair at seed 1904 (`predictivity_all`), gated with `predictivity`'s mask, ≥ 3 pairs per task; task counts under the points. Same table as `scale_convergence_transformation.csv`. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by transformation`.

![Scale convergence per design axis](pretraining/predictivity/scale_convergence_transformation_panels.png)

| axis | families | 175M R [lo, hi] (tasks) | 350M R [lo, hi] (tasks) | 600M R [lo, hi] (tasks) | 1B R [lo, hi] (tasks) | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| all pairs | 23 | 0.53 [0.51, 0.56] (227) | 0.54 [0.51, 0.56] (321) | 0.54 [0.52, 0.56] (352) | 0.56 [0.52, 0.59] (387) | never |
| depth (deep vs shallow) | 20 | 0.53 [0.49, 0.58] (179) | 0.48 [0.46, 0.50] (259) | 0.48 [0.44, 0.52] (290) | 0.51 [0.46, 0.56] (311) | never |
| language count | 21 | 0.52 [0.48, 0.56] (169) | 0.53 [0.50, 0.55] (247) | 0.53 [0.50, 0.56] (275) | 0.53 [0.50, 0.56] (294) | never |
| language list (A vs B) | 12 | 0.49 [0.40, 0.57] (43) | 0.49 [0.44, 0.54] (65) | 0.48 [0.41, 0.55] (73) | 0.49 [0.44, 0.53] (75) | never |
| temperature (T=1 vs T=3) | 8 | 0.49 [0.44, 0.55] (169) | 0.55 [0.52, 0.58] (247) | 0.57 [0.55, 0.58] (275) | 0.57 [0.52, 0.62] (294) | never |

Key findings:

- **depth (deep vs shallow)** (20 families): R = 0.51 at 1B [0.46, 0.56] over 311 tasks; no proxy reaches τ.
- **language count** (21 families): R = 0.53 at 1B [0.50, 0.56] over 294 tasks; no proxy reaches τ.
- **language list (A vs B)** (12 families): R = 0.49 at 1B [0.44, 0.53] over 75 tasks; no proxy reaches τ.
- **temperature (T=1 vs T=3)** (8 families): R = 0.57 at 1B [0.52, 0.62] over 294 tasks; no proxy reaches τ.
- The bands are leave-one-design-variant-out over the families in the column, not seed noise; a band on 4 families is four numbers and only says which variant the line hinges on.

Follow-ups:

- The `above_66_size` twin per panel (`scale_convergence_transformation_panels_above_66_size.png`): the same split on the cells that rank reliably.
- A per-axis panel grid on the L8 languages only (`--langs L8`), so the language-count panel is read on one task set.
- The second-language and English-corpus panels fill in once their 1.7B cells are in the report (rule 9).

Files: [`scale_convergence_transformation_panels.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_transformation_panels.png), [`scale_convergence_transformation_panels.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_transformation_panels.csv).
<!-- END auto:scale-convergence-transformation-panels -->

**Key findings**

- On every gated task DA-size is 0.53 at 175M and 0.56 at 1B: 0.531
  [0.506, 0.556], 0.536 [0.512, 0.559], 0.544 [0.523, 0.565], 0.557
  [0.522, 0.591] at 175M, 350M, 600M, 1B on 227–387 tasks (23 families); the
  mono-axis pairs read 0.518, 0.511, 0.517, 0.524 (`scale_convergence_one_axis.csv`).
  Nowhere near τ = 0.90, so N_min is undefined.
- Per-language BPB does better (0.81 / 0.87 / 0.73 / 0.89 on the BPB panel)
  and the two aggregates best of all: `bpb_macro` 0.91–0.97 and `train_loss`
  0.81–0.86 from any size (highlight block). The design differences the
  ladder measures move most benchmark scores by less than their noise at any
  one size; BPB, which sums over every token, sees them.
- By design axis (`scale_convergence_transformation.csv`) the unfiltered
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
- Read the `_flops` twins (`scale_convergence_*_flops.png`) with one line per
  size and the annealed points marked; a bigger model's first (peak-LR)
  checkpoint beside a smaller model's annealed final is what makes the
  compute axis zigzag.

GitHub: [scale_convergence.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence.png) · [scale_convergence.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence.csv) ·
GitHub: [scale_convergence_one_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_one_axis.png) · [scale_convergence_one_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_one_axis.csv) ·
GitHub: [scale_convergence_L.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L.png) · [scale_convergence_L.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L.csv) ·
GitHub: [scale_convergence_transformation.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_transformation.png) · [scale_convergence_transformation.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_transformation.csv) ·
GitHub: [scale_convergence_transformation_one_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_transformation_one_axis.png) · [scale_convergence_transformation_one_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_transformation_one_axis.csv)

What this leaves open is whether the coin flip is the average of a few
reliable tasks and many hopeless ones — figure 2.

### 2. The paper figure is a conditional statement

**Left panel DA-size, middle DA-ckpt, right DA-goal · filter per panel:
`above_66_size` / `above_66_ckpt` / `above_66_either` (the
`_either_transformation` stem) · mono-axis pairs (`_one_axis`) · pairs from
`predictivity_schemes` at seed 1904 · gate `predictivity`.** The paper
embeds `rq2.png` (`paper_rq2.py`, no filter, multi-axis, copied by
`documents/paper/figures/make_rq_figures.py`); its variants share the
composition and differ only in filter and pair set: `rq2_one_axis`
(mono-axis, no filter), `rq2_above_80` (the `late` reduction at 0.80, both
axes), `rq2_above_66_both[_transformation]` (one population on all three
panels, kept for comparison only), `rq2_above_66_one` (each panel its own
cut), and `rq2_ten_checkpoints` (`paper_ten_checkpoints.py`: the mean over
tasks on the A/B pool's ten checkpoints, BPB and benchmarks, no filter).

![RQ2, per-panel cuts, mono-axis pairs](pretraining/predictivity/rq2_above_66_either_transformation_one_axis.png)

*Left: DA-size per design axis (mono-axis pairs) and pooled, over the tasks
whose median DA-size over the proxy sizes is ≥ 0.66 (38 tasks at 175M, 63 at
1.7B); x is non-embedding parameters, the hollow 1.7B point is 1.0 by
construction, dotted τ = 0.90. Middle: DA-ckpt over the tasks whose median
DA-ckpt is ≥ 0.66 (99–130 tasks), each proxy against its own final, x in
Chinchilla multiples, dotted 0.75. Right: DA-goal over the tasks passing
either cut (97–149), the same checkpoints against the reference's final; the
panels share the y axis, so the vertical distance between middle and right is
what the proxy's SIZE costs on top of reading it early. The left panel is
therefore not the same cells as the middle and right: three claims, not one
population seen three ways. CSVs: `rq2_above_66_either_transformation_one_axis.csv`,
`early_small_by_L_ckpt_above_66_ckpt_one_axis.csv`,
`early_small_by_L_goal_above_66_either_one_axis.csv`.*

| line (mono-axis, `above_66_size` tasks) | 175M | 350M | 600M | 1B | tasks |
|---|---|---|---|---|---|
| all pairs | 0.599 | 0.717 | 0.738 | 0.761 | 38–62 |
| temperature (T = 1 vs 3) | 0.576 | 0.829 | 0.793 | 0.811 | 24–37 |
| language count | 0.577 | 0.755 | 0.821 | 0.796 | 24–37 |
| depth (deep vs shallow) | 0.682 | 0.689 | 0.572 | 0.744 | 24–39 |
| language list (A vs B) | 0.500 | 0.533 | 0.583 | 0.571 | 4–7 |

**Key findings**

- On the filtered population the pooled DA-size rises from 0.60 to 0.76;
  temperature and language count are the decisions a 350M proxy already
  reads at 0.76–0.83; depth dips to 0.57 at 600M (unexplained); the
  language-list decision never leaves 0.50–0.58 on 4–7 tasks — a coin flip,
  not below it. On the unfiltered tables the same four lines sit at 0.48–0.57
  (figure 1).
- The rise is partly the cut: the filter keeps the tasks whose DA-size
  cleared 0.66 and then plots DA-size on them; on `predictivity_schemes` the
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

GitHub: [rq2_above_66_either_transformation_one_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_66_either_transformation_one_axis.png) · [rq2_above_66_either_transformation_one_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_66_either_transformation_one_axis.csv) ·
GitHub: [rq2.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2.png) · [rq2.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2.csv) ·
GitHub: [rq2_one_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_one_axis.png) · [rq2_one_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_one_axis.csv) ·
GitHub: [rq2_above_80.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_80.png) · [rq2_above_80.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_80.csv) ·
GitHub: [rq2_above_66_both.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_66_both.png) · [rq2_above_66_both.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_66_both.csv) ·
GitHub: [rq2_above_66_both_transformation.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_66_both_transformation.png) · [rq2_above_66_both_transformation.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_66_both_transformation.csv) ·
GitHub: [rq2_above_66_one.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_66_one.png) · [rq2_above_66_one.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_66_one.csv) ·
GitHub: [rq2_ten_checkpoints.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_ten_checkpoints.png) · [rq2_ten_checkpoints.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_ten_checkpoints.csv) ·
GitHub: [scale_convergence_above_66_size_one_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_above_66_size_one_axis.png) · [scale_convergence_above_66_size_one_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_above_66_size_one_axis.csv)

The population behind the cuts — which (benchmark, language) cells rank
reliably at all, and in which languages — is the block below.

**Reliable cells · DA-size and DA-ckpt · every cut × reduction ·
multi-axis (the table carries mono-axis too) · pairs from
`predictivity_schemes` · gate `predictivity`.** The verdicts from 2026-09-22
on gate DA-size at the reference too (rule 1): before that, three cells at
chance at 1.7B counted as reliable, and the `above_66_both` population read
23 cells; the 2026-09-23 snapshot with the twins at 175M and 350M took it
to 71.

<!-- BEGIN auto:reliable-tasks (reliable_tasks.py --pool predictivity) -->
## Which benchmark-language cells rank reliably

Per language, how many benchmarks clear DA ≥ 0.8 on DA-size (a proxy size's final ranking vs the reference's) and on DA-ckpt (an earlier checkpoint vs the same size's final), reducing each task's cells with `late` (one fixed cell per axis, so no cell is chosen by its value). Cells need ≥ 3 pairs (rule 5) and must survive the above-random gate (rule 1). The decisions come from the `predictivity_schemes` pool — every data scheme at the grid seed, which is the population `by_L` and `scale_convergence` pair over — while the gate and this folder stay with `predictivity`; the table carries one row per pair set (rule 15) and the figures show `multi-axis`. `da_reliable_tasks.csv` holds the per-task values for every reduction and is threshold-free — each figure is one view of it. Regenerate with `python analysis/rq02_decision_accuracy/reliable_tasks.py --pool predictivity`.

| language | benchmarks evaluated | DA-size | DA-ckpt | either | both |
|---|---|---|---|---|---|
| en | 44 | 2 | 13 | 13 | 2 |
| ru | 13 | 3 | 6 | 6 | 3 |
| zh | 15 | 0 | 3 | 3 | 0 |
| de | 12 | 1 | 7 | 7 | 1 |
| ja | 9 | 0 | 3 | 3 | 0 |
| es | 53 | 1 | 13 | 13 | 1 |
| fr | 13 | 2 | 5 | 5 | 2 |
| it | 11 | 2 | 5 | 5 | 2 |
| pt | 11 | 1 | 4 | 4 | 1 |
| pl | 7 | 0 | 3 | 3 | 0 |
| nl | 9 | 1 | 3 | 3 | 1 |
| id | 10 | 2 | 4 | 4 | 2 |
| vi | 12 | 1 | 5 | 5 | 1 |
| fa | 7 | 0 | 3 | 3 | 0 |
| tr | 9 | 0 | 4 | 4 | 0 |
| th | 3 | 0 | 1 | 1 | 0 |
| uk | 9 | 1 | 3 | 4 | 0 |
| el | 9 | 0 | 0 | 0 | 0 |
| ko | 6 | 0 | 1 | 1 | 0 |
| cs | 5 | 0 | 2 | 2 | 0 |
| sv | 7 | 1 | 3 | 3 | 1 |
| hu | 7 | 1 | 2 | 2 | 1 |
| ro | 5 | 1 | 2 | 2 | 1 |
| no | 2 | 0 | 0 | 0 | 0 |
| da | 7 | 1 | 3 | 3 | 1 |
| bg | 7 | 0 | 3 | 3 | 0 |
| fi | 3 | 0 | 1 | 1 | 0 |
| hi | 13 | 0 | 5 | 5 | 0 |
| bn | 10 | 1 | 2 | 2 | 1 |
| sk | 4 | 3 | 4 | 4 | 3 |
| he | 6 | 2 | 2 | 3 | 1 |
| lt | 8 | 2 | 2 | 4 | 0 |
| bs | 1 | 0 | 0 | 0 | 0 |
| sl | 2 | 1 | 2 | 2 | 1 |
| et | 8 | 3 | 2 | 3 | 2 |
| ca | 8 | 3 | 4 | 5 | 2 |
| ta | 5 | 0 | 0 | 0 | 0 |
| hr | 7 | 2 | 3 | 3 | 2 |
| lv | 2 | 0 | 0 | 0 | 0 |
| ms | 8 | 2 | 4 | 4 | 2 |
| az | 6 | 0 | 3 | 3 | 0 |
| ka | 7 | 0 | 2 | 2 | 0 |
| ne | 10 | 2 | 2 | 2 | 2 |
| mr | 5 | 3 | 2 | 3 | 2 |
| ml | 2 | 0 | 1 | 1 | 0 |
| kk | 9 | 2 | 4 | 5 | 1 |
| ur | 7 | 1 | 5 | 5 | 1 |
| sq | 6 | 1 | 2 | 2 | 1 |
| ar | 24 | 0 | 6 | 6 | 0 |
| sr | 10 | 3 | 2 | 4 | 1 |

**How many cells pass, by cut and reduction** — the cut is a choice, and this is its whole sensitivity:

| threshold | reduction | tasks passing both | languages | benchmarks |
|---|---|---|---|---|
| 0.8 | late | 42 | 28 | hellaswag, include_v2_og, lambada_openai_mt, multiblimp, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xstorycloze |
| 0.8 | mean | 7 | 7 | hellaswag, multiblimp, rfgm_include_base_44 |
| 0.8 | median | 19 | 16 | global_piqa_nonparallel_cloze, hellaswag, multiblimp, rf_include_base_44, rfgm_include_base_44, xstorycloze |
| 0.8 | max | 71 | 34 | global_piqa_nonparallel_cloze, hellaswag, include_base_44, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xnli, xstorycloze |
| 0.75 | late | 54 | 35 | hellaswag, include_v2_og, lambada_openai_mt, multiblimp, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xcopa, xstorycloze |
| 0.75 | mean | 29 | 25 | global_piqa_nonparallel_cloze, hellaswag, include_v2_og, multiblimp, paws, rf_global_mmlu_full, rfgm_include_base_44, xstorycloze |
| 0.75 | median | 32 | 24 | global_piqa_nonparallel_cloze, hellaswag, include_v2_og, multiblimp, paws, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xstorycloze |
| 0.75 | max | 91 | 40 | global_piqa_nonparallel_cloze, hellaswag, include_base_44, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xcopa, xnli, xstorycloze |
| 0.66 | late | 106 | 42 | belebele, global_piqa_nonparallel_cloze, hellaswag, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xcopa, xstorycloze, xwinograd |
| 0.66 | mean | 63 | 35 | arc, belebele, global_piqa_nonparallel_cloze, hellaswag, include_base_44, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xstorycloze |
| 0.66 | median | 71 | 36 | arc, belebele, global_piqa_nonparallel_cloze, hellaswag, include_base_44, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, xnli, xstorycloze, xwinograd |
| 0.66 | max | 193 | 46 | arc, belebele, global_piqa_nonparallel_cloze, hellaswag, include_base_44, include_v2_en, include_v2_og, lambada_openai_mt, multiblimp, paws, rf_acp_bench_mcq, rf_belebele, rf_global_mmlu_full, rf_include_base_44, rfgm_include_base_44, truthfulqa_mc2, xcopa, xnli, xstorycloze, xwinograd |

![Reliable benchmark-language cells](pretraining/predictivity/da_reliable_tasks_80_late.png)
<!-- END auto:reliable-tasks -->

[da_reliable_tasks.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_reliable_tasks.csv) ·
[da_reliable_by_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_reliable_by_language.csv) ·
[da_reliable_tasks_80_late.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_reliable_tasks_80_late.png) ·
[da_reliable_tasks_66_median.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_reliable_tasks_66_median.png) ·
[da_reliable_tasks_75_late.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_reliable_tasks_75_late.png)
(one PNG per cut × reduction, `da_reliable_tasks_<t>_<red>.png`, tables
`da_reliable_tasks_66.csv`, `da_reliable_tasks_75.csv`).

### 3. Early and small: DA-goal and DA-ckpt on the ten checkpoints

**`early_small_by_L_{ckpt,goal}_*`: DA-ckpt / DA-goal · mean over tasks ·
filter per stem (none, `_above_80`, `_above_66_ckpt`, `_above_66_either`,
`_above_66_both`; `_with_bpb` adds the per-size BPB lines) · multi-axis
unless `_one_axis` · pairs from `predictivity_schemes` (`by_L.py` pools every
scheme at the grid seed) · gate `predictivity`.** The `early_small.png` grid,
the `safe_*` maps and the Results block below read `compute_da.py`'s own
tables on the A/B pool `predictivity` (multi-axis, no filter, gate
`predictivity`) — DA-goal at 1C–5C per proxy, the mean over tasks, and the
smallest safe level (DA ≥ 0.75 over ≥ 3 pairs, held at every larger level).

<!-- BEGIN auto:results (da_per_benchmark.py --pool predictivity) -->
## Results

Numbers from the `predictivity` pool (`da_per_task.csv`, pairs from `da_n_pairs_per_task.csv`, gate from rq00). Regenerate with `python analysis/rq02_decision_accuracy/da_per_benchmark.py --pool predictivity`.

**DA-size by proxy size** (`n` tasks; median pairs per cell):

| comparison | benchmarks | n | pairs | BPB | n |
|---|---|---|---|---|---|
| 175M → 1.7B | 0.52 | 179 | 28 | 0.78 | 34 |
| 350M → 1.7B | 0.50 | 259 | 45 | 0.82 | 34 |
| 600M → 1.7B | 0.51 | 290 | 45 | 0.58 | 34 |
| 1B → 1.7B | 0.52 | 311 | 28 | 0.82 | 34 |

![DA-size by family](pretraining/predictivity/da_size_by_family.png)

**DA-ckpt by bucket and fraction of the run** (mean over the above-random benchmark tasks):

| bucket | 10 % | 20 % | 30 % | 40 % | 50 % | 60 % | 70 % | 80 % | 90 % |
|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.51 | 0.53 | 0.56 | 0.64 | 0.67 | 0.70 | 0.72 | 0.78 | 0.83 |
| 350M | 0.49 | 0.51 | 0.55 | 0.57 | 0.60 | 0.63 | 0.68 | 0.72 | 0.80 |
| 600M | 0.49 | 0.51 | 0.54 | 0.56 | 0.57 | 0.59 | 0.60 | 0.65 | 0.74 |
| 1B | 0.48 | 0.51 | 0.54 | 0.55 | 0.57 | 0.59 | 0.61 | 0.64 | 0.74 |
| 1.7B | 0.50 | 0.54 | 0.55 | 0.56 | 0.56 | 0.58 | 0.59 | 0.62 | 0.71 |
<!-- END auto:results -->

<!-- BEGIN auto:early-small (early_small.py --pool predictivity) -->
## Early and small, as a ranking

Numbers from the `predictivity` pool: every design variant at a proxy size, read at 1C–5C of training (C = the Chinchilla-optimal 20 tokens per parameter; every run trains 5C, so 1C is 20 % of it), ranked against the same variants at the 1.7B final checkpoint (the 5C column is DA-size, the 1.7B row is that size's DA-ckpt). A benchmark task counts only where it clears chance at the proxy size and at 1.7B. Regenerate with `python analysis/rq02_decision_accuracy/early_small.py --pool predictivity`.

- **bpb** — smallest proxy whose mean agreement with the 1.7B final ranking reaches 0.75: **175M at 2C** (0.77).
- **all benchmarks** — no (proxy, checkpoint) reaches a mean agreement of 0.75.
- **Smallest safe size per (benchmark, language)** — never: 228, 1B: 38, 350M: 7, 175M: 5, 600M: 4 of 282 cells.

![rq02 in one figure](pretraining/predictivity/highlights.png)

**bpb** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 34 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.67 | 0.74 | 0.66 | 0.77 | 0.83 | 0.66 | 0.75 | 0.79 | 0.78 | 0.78 |
| 350M | 0.71 | 0.73 | 0.72 | 0.70 | 0.77 | 0.79 | 0.75 | 0.76 | 0.80 | 0.82 |
| 600M | 0.52 | 0.53 | 0.54 | 0.57 | 0.57 | 0.59 | 0.56 | 0.57 | 0.58 | 0.58 |
| 1B | 0.65 | 0.66 | 0.70 | 0.75 | 0.74 | 0.80 | 0.77 | 0.80 | 0.81 | 0.82 |
| 1.7B | 0.81 | 0.90 | 0.93 | 0.95 | 0.96 | 0.95 | 0.97 | 0.96 | 0.99 |  |

**all benchmarks** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over 347 tasks):

| proxy | 0.5C | 1C | 1.5C | 2C | 2.5C | 3C | 3.5C | 4C | 4.5C | 5C |
|---|---|---|---|---|---|---|---|---|---|---|
| 175M | 0.51 | 0.47 | 0.51 | 0.50 | 0.51 | 0.50 | 0.48 | 0.50 | 0.51 | 0.52 |
| 350M | 0.46 | 0.46 | 0.48 | 0.47 | 0.47 | 0.49 | 0.49 | 0.49 | 0.49 | 0.50 |
| 600M | 0.47 | 0.47 | 0.48 | 0.50 | 0.50 | 0.49 | 0.50 | 0.51 | 0.51 | 0.51 |
| 1B | 0.49 | 0.50 | 0.50 | 0.50 | 0.52 | 0.52 | 0.51 | 0.53 | 0.51 | 0.52 |
| 1.7B | 0.50 | 0.54 | 0.55 | 0.56 | 0.56 | 0.58 | 0.59 | 0.62 | 0.71 |  |

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

<!-- BEGIN auto:early-small-by-transformation (by_L.py --pool predictivity --by transformation) -->
## Early and small per design axis

The per-L reading above pools every design axis inside an L; this one splits the MONO-AXIS pairs at seed 1904 (`predictivity_all`, every scheme) by the one axis each pair moves — language count, depth (deep vs shallow), language list (A vs B), temperature (T=1 vs T=3), 2nd language (ru vs zh vs es) — one panel per axis and a first panel over every mono-axis pair (median pairs per cell up to 20). Same gate (rule 1), pair minimum (rule 5) and filter variants as the per-L figures; the `above_66_ckpt` twin filters the DA-ckpt figure and `above_66_either` the DA-goal one, both on the mono-axis reliability (rule 15). Task counts sit at the end of every line and the populations differ between panels and sizes (rule 13). Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool predictivity --by transformation`.

![DA-ckpt per design axis](pretraining/predictivity/early_small_by_transformation_ckpt.png)

**DA-ckpt** (against the proxy size's own final; cell = mean DA at the first → last drawn checkpoint, tasks behind the line in brackets):

| axis (DA-ckpt, 0.5C → 4.5C, tasks) | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|
| all | 0.50 → 0.82 (254) | 0.50 → 0.82 (329) | 0.52 → 0.75 (358) | 0.51 → 0.74 (392) | 0.50 → 0.72 (436) |
| language count | 0.52 → 0.81 (165–189) | 0.49 → 0.79 (249) | 0.50 → 0.74 (278) | 0.51 → 0.75 (298) | 0.51 → 0.73 (328) |
| depth (deep vs shallow) | 0.50 → 0.83 (193–200) | 0.48 → 0.79 (261) | 0.48 → 0.73 (293) | 0.47 → 0.73 (316) | 0.49 → 0.72 (347) |
| language list (A vs B) | 0.43 → 0.81 (55) | 0.47 → 0.81 (65) | 0.52 → 0.75 (73) | 0.46 → 0.74 (77) | 0.44 → 0.69 (86) |
| temperature (T=1 vs T=3) | 0.53 → 0.81 (189) | 0.53 → 0.84 (249) | 0.55 → 0.77 (278) | 0.53 → 0.74 (298) | 0.55 → 0.73 (328) |
| 2nd language (ru vs zh vs es) | 0.33 → 0.78 (6) | 0.39 → 0.71 (28) | 0.56 → 0.76 (31) | 0.46 → 0.78 (32) | — |

Key findings:

- At 175M the DA-ckpt line clears 0.75 and stays there — language count: from 4C (0.77); depth (deep vs shallow): from 4C (0.76); language list (A vs B): from 4C (0.77); temperature (T=1 vs T=3): from 3.5C (0.75); 2nd language (ru vs zh vs es): from 3.5C (0.78).
- At 1.7B — language count: 0.51 → 0.73 (328); depth (deep vs shallow): 0.49 → 0.72 (347); language list (A vs B): 0.44 → 0.69 (86); temperature (T=1 vs T=3): 0.55 → 0.73 (328); 2nd language (ru vs zh vs es): —.

Follow-ups:

- A `_with_bpb` variant per axis, to see whether BPB decides the temperature and the second language earlier than the benchmarks do.
- The same panels on the L8 languages only (`scale_convergence.py --langs L8` does it for DA-size), so the language-count panel is read on one task set.
- Once BT3 trains, the temperature panel gains the B-vs-BT3 pairs and the list panel AT3-vs-BT3 with no code change.

![DA-goal per design axis](pretraining/predictivity/early_small_by_transformation_goal.png)

**DA-goal** (against the 1.7B final):

| axis (DA-goal, 0.5C → 5C, tasks) | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|
| all | 0.48 → 0.50 (227) | 0.47 → 0.50 (321) | 0.50 → 0.53 (352) | 0.50 → 0.54 (387) | 0.50 → 0.72 (436) |
| language count | 0.53 → 0.50 (145–169) | 0.47 → 0.52 (247) | 0.52 → 0.53 (275) | 0.52 → 0.53 (294) | 0.51 → 0.73 (328) |
| depth (deep vs shallow) | 0.47 → 0.53 (172–179) | 0.46 → 0.48 (259) | 0.42 → 0.48 (290) | 0.45 → 0.52 (311) | 0.49 → 0.72 (347) |
| language list (A vs B) | 0.54 → 0.49 (43) | 0.46 → 0.49 (65) | 0.43 → 0.48 (73) | 0.47 → 0.49 (75) | 0.44 → 0.69 (86) |
| temperature (T=1 vs T=3) | 0.46 → 0.49 (169) | 0.49 → 0.55 (247) | 0.54 → 0.57 (275) | 0.55 → 0.58 (294) | 0.55 → 0.73 (328) |

Key findings:

- At 175M the DA-goal line clears 0.75 and stays there — language count: never (max 0.53); depth (deep vs shallow): never (max 0.55); language list (A vs B): never (max 0.57); temperature (T=1 vs T=3): never (max 0.53).
- The distance between the DA-goal and DA-ckpt cell of an axis at a proxy size is what the size costs; the 1.7B column is the same line in both.

Follow-ups:

- The size axis of the same split is `scale_convergence_transformation_panels.png` (DA-size pooled over decisions).
- A jackknife band per axis line (leave one family out), as the scale-convergence panels carry.

Filtered twins (reliable cells only): [`early_small_by_transformation_goal_above_80.png`](pretraining/predictivity/early_small_by_transformation_goal_above_80.png), [`early_small_by_transformation_goal_above_66_both.png`](pretraining/predictivity/early_small_by_transformation_goal_above_66_both.png), [`early_small_by_transformation_goal_above_66_either.png`](pretraining/predictivity/early_small_by_transformation_goal_above_66_either.png), [`early_small_by_transformation_ckpt_above_80.png`](pretraining/predictivity/early_small_by_transformation_ckpt_above_80.png), [`early_small_by_transformation_ckpt_above_66_both.png`](pretraining/predictivity/early_small_by_transformation_ckpt_above_66_both.png), [`early_small_by_transformation_ckpt_above_66_ckpt.png`](pretraining/predictivity/early_small_by_transformation_ckpt_above_66_ckpt.png)

Files: [`early_small_by_transformation_goal.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_goal.png), [`early_small_by_transformation_goal.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_goal.csv), [`early_small_by_transformation_goal_above_80.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_goal_above_80.png), [`early_small_by_transformation_goal_above_80.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_goal_above_80.csv), [`early_small_by_transformation_goal_above_66_both.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_goal_above_66_both.png), [`early_small_by_transformation_goal_above_66_both.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_goal_above_66_both.csv), [`early_small_by_transformation_goal_above_66_either.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_goal_above_66_either.png), [`early_small_by_transformation_goal_above_66_either.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_goal_above_66_either.csv), [`early_small_by_transformation_ckpt.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_ckpt.png), [`early_small_by_transformation_ckpt.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_ckpt.csv), [`early_small_by_transformation_ckpt_above_80.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_ckpt_above_80.png), [`early_small_by_transformation_ckpt_above_80.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_ckpt_above_80.csv), [`early_small_by_transformation_ckpt_above_66_both.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_ckpt_above_66_both.png), [`early_small_by_transformation_ckpt_above_66_both.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_ckpt_above_66_both.csv), [`early_small_by_transformation_ckpt_above_66_ckpt.png`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_ckpt_above_66_ckpt.png), [`early_small_by_transformation_ckpt_above_66_ckpt.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_transformation_ckpt_above_66_ckpt.csv), [`da_by_transformation_per_task.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_by_transformation_per_task.csv).
<!-- END auto:early-small-by-transformation -->

**Key findings**

- The mean-over-tasks reading on the A/B pool is 0.46–0.53 at every proxy
  size and checkpoint for the benchmarks (`early_small.csv`,
  `rq2_ten_checkpoints.csv`), the pooled-over-decisions reading of figure 1
  0.53–0.56: two estimands, one verdict.
- BPB reads the 1.7B ranking at ≥ 0.75 from 175M at 2C (0.77) and from 350M
  at 2.5C; `bpb_macro` and `train_loss` from any size and checkpoint; no
  (proxy, checkpoint) cell reaches 0.75 for the benchmark mean.
- DA-ckpt rises monotonically along every run (175M: 0.51 at 10 % → 0.83 at
  90 %; 1.7B: 0.50 → 0.71) and the reference's own curve is its DA-ckpt; read
  against the seed null of figure 7 (0.47 → 0.81 at 175M), the design signal
  in that rise is 0.01–0.03.
- Per L, only the regimes with ≥ 3 usable pairs draw a panel (`pairs_by_L.csv`:
  L1 and L2 have one pair against 1.7B on the A/B pool; the ZH/ES cells give
  L2 its three families on `predictivity_schemes`, figure 4).
- Smallest safe level per (benchmark, language): never 228, 1B 38, 350M 7,
  175M 5, 600M 4 of 282 cells; "safe" with 3 pairs is a weak guarantee.

**Follow-ups**

- The `_flops` reading (`safe_flops.png`) joins un-annealed and annealed
  points; one line per size with the annealed points marked, plus the
  compute frontier (the best DA reachable at or below each compute), is the
  practical answer.
- `early_small_by_L_own` (the size's cost separated from the checkpoint's) as
  an appendix pair with `early_small_by_L_goal`, once the 6-pair L's are
  complete; `pairs_by_L.csv` justifies why per-L DA is coarse.

GitHub: [early_small.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small.png) · [early_small.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small.csv) ·
[early_small_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_benchmark.png) ·
[early_small_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_language.png) ·
GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/highlights.csv) ·
GitHub: [safe_size.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_size.png) · [safe_size.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_size.csv) ·
GitHub: [safe_checkpoint.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_checkpoint.png) · [safe_checkpoint.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_checkpoint.csv) ·
GitHub: [safe_flops.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_flops.png) · [safe_flops.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/safe_flops.csv) ·
GitHub: [da_size_by_family.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_by_family.png) · [da_size_by_family.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_by_family.csv) ·
[da_size_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_by_benchmark.png) ·
[da_size_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size_by_language.png) ·
[da_size.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_size.csv) ·
GitHub: [early_small_by_L_goal.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_goal.png) · [early_small_by_L_goal.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_goal.csv) ·
GitHub: [early_small_by_L_goal_with_bpb.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_goal_with_bpb.png) · [early_small_by_L_goal_with_bpb.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_goal_with_bpb.csv) ·
GitHub: [early_small_by_L_ckpt.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_ckpt.png) · [early_small_by_L_ckpt.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_ckpt.csv) ·
GitHub: [early_small_by_L_ckpt_with_bpb.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_ckpt_with_bpb.png) · [early_small_by_L_ckpt_with_bpb.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_ckpt_with_bpb.csv) ·
GitHub: [early_small_by_L_goal_above_80.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_goal_above_80.png) · [early_small_by_L_goal_above_80.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_goal_above_80.csv) ·
GitHub: [early_small_by_L_ckpt_above_80.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_ckpt_above_80.png) · [early_small_by_L_ckpt_above_80.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_ckpt_above_80.csv) ·
GitHub: [early_small_by_L_ckpt_above_66_ckpt_one_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_ckpt_above_66_ckpt_one_axis.png) · [early_small_by_L_ckpt_above_66_ckpt_one_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_ckpt_above_66_ckpt_one_axis.csv) ·
GitHub: [early_small_by_L_goal_above_66_either_one_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_goal_above_66_either_one_axis.png) · [early_small_by_L_goal_above_66_either_one_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/early_small_by_L_goal_above_66_either_one_axis.csv) ·
[pairs_by_L.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/pairs_by_L.csv) ·
[da_by_L_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_by_L_per_task.csv) ·
[da_pooled_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/da_pooled_per_task.csv)

### 4. Language count: unordered, on any task set

**DA-size · `scale_convergence_L8*`: no filter (the `_above_66_size` twins
conditional) · multi-axis (`scale_convergence_L_one_axis` mono-axis, every
language) · pairs sharing the L from `predictivity_schemes` at seed 1904 ·
gate `predictivity`.** One line per language-count regime; a regime pools
depth, list and temperature decisions (`share_*` in the CSV) and the mix
differs by regime, which rule 5 forbids holding fixed.

<!-- BEGIN auto:scale-convergence-L8 (scale_convergence.py --by L --langs L8) -->
## Scale convergence by language count, on the L8 languages

The `--by L` lines read over the tasks in the 8 languages of the L8 setting (de, en, es, fr, it, ja, ru, zh), which every regime from L8 up trains. What this cannot fix: a regime pools arch, list and temperature decisions at once and the mix differs by regime (`share_*` in the CSV); rule 5 forbids holding it fixed. Under `--axes mono-axis` the regimes keep only their one-axis pairs (L8 6 → 4, L30 10 → 5) and rest on fewer tasks. Numbers below are the unfiltered population; the `above_66_size` twin (`scale_convergence_L8_above_66_size.png`) is the conditional one. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by L --langs L8`.

| group | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| L15 | 0.54 | 0.51 | 0.5 | 0.52 | 1.0 | — |
| L2 | 0.39 | 0.58 | 0.53 | 0.48 | 1.0 | — |
| L30 | 0.53 | 0.47 | 0.52 | 0.45 | 1.0 | — |
| L50 | 0.54 | 0.53 | 0.5 | 0.56 | 1.0 | — |
| L8 | 0.47 | 0.49 | 0.46 | 0.57 | 1.0 | — |
| all pairs | 0.54 | 0.54 | 0.54 | 0.55 | 1.0 | — |

![Scale convergence, L8](pretraining/predictivity/scale_convergence_L8.png)
<!-- END auto:scale-convergence-L8 -->

<!-- BEGIN auto:scale-convergence-L8-common (scale_convergence.py --by L --langs L8 --common-tasks) -->
## Scale convergence by language count, on the L8 languages, common tasks

The `--by L` lines read over the tasks in the 8 languages of the L8 setting (de, en, es, fr, it, ja, ru, zh), which every regime from L8 up trains, and further over the tasks with ≥ 3 pairs in every regime at every proxy size — one task set for the whole figure, so a gap between lines is a gap on the same benchmarks. What this cannot fix: a regime pools arch, list and temperature decisions at once and the mix differs by regime (`share_*` in the CSV); rule 5 forbids holding it fixed. Under `--axes mono-axis` the regimes keep only their one-axis pairs (L8 6 → 4, L30 10 → 5) and rest on fewer tasks. Numbers below are the unfiltered population; the `above_66_size` twin (`scale_convergence_L8common_above_66_size.png`) is the conditional one. Regenerate with `python analysis/rq02_decision_accuracy/scale_convergence.py --by L --langs L8 --common-tasks`.

| group | 175M | 350M | 600M | 1B | 1.7B | N_min(τ=0.9) |
|---|---|---|---|---|---|---|
| L15 | 0.62 | 0.4 | 0.4 | 0.6 | 1.0 | — |
| L2 | 0.39 | 0.44 | 0.61 | 0.39 | 1.0 | — |
| L30 | 0.67 | 0.62 | 0.53 | 0.42 | 1.0 | — |
| L50 | 0.39 | 0.67 | 0.53 | 0.53 | 1.0 | — |
| L8 | 0.47 | 0.53 | 0.36 | 0.44 | 1.0 | — |
| all pairs | 0.57 | 0.61 | 0.59 | 0.57 | 1.0 | — |

![Scale convergence, L8 common tasks](pretraining/predictivity/scale_convergence_L8common.png)
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
  decisions are depth, list and temperature in a mix that differs by regime;
  what the figure shows is that the decisions within any regime are read
  equally poorly from a smaller model, on 4–5 families per regime.

**Follow-ups**

- An L8-only replicate of the depth and list decisions in every regime would
  hold the decision mix fixed — the one thing this grid cannot do above three
  pairs.
- Bootstrap over L, not items, for the per-L comparison (items within an L
  move together).

GitHub: [scale_convergence_L8.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L8.png) · [scale_convergence_L8.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L8.csv) ·
GitHub: [scale_convergence_L8common.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L8common.png) · [scale_convergence_L8common.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L8common.csv) ·
GitHub: [scale_convergence_L8_above_66_size.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L8_above_66_size.png) · [scale_convergence_L8_above_66_size.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L8_above_66_size.csv) ·
GitHub: [scale_convergence_L8common_above_66_size.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L8common_above_66_size.png) · [scale_convergence_L8common_above_66_size.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L8common_above_66_size.csv) ·
GitHub: [scale_convergence_L_one_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L_one_axis.png) · [scale_convergence_L_one_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_L_one_axis.csv)

### 5. One language at a time, and the tokens axis

**DA-size · `scale_convergence_lang_all*`: no filter (`_above_66_size`
conditional) · multi-axis · pairs sharing the L from `predictivity_schemes` ·
gate `predictivity`.** One panel per L8 language, one line per regime that
trains it plus the pooled line; the `_tokens` twin relabels x as the tokens of
the language the decision's two members trained on (share × D(N)), and the
panel title carries the collapse test.

<!-- BEGIN auto:scale-convergence-by-language (by_language.py --pool predictivity) -->
## Scale convergence per language

For each language of the L8 setting, the `--by L` lines read on that language's benchmarks alone: R at the smallest → largest proxy [tasks], per regime that trains the language, on every gated task (no selection on DA — the inference version; the `above_66_size` twin is the conditional one). The last column is the collapse test: R² of one log-linear line through every regime's points with x = model size, then with x = tokens of the language (its share of all tokens × D(N)); a rise under tokens says exposure explains what language count does not. Regimes pool arch, list and temperature decisions at once. Regenerate with `python analysis/rq02_decision_accuracy/by_language.py --pool predictivity`; `scale_convergence_lang_all_coverage.csv` says why a cell is empty.

| language | L1 | L2 | L8 | L15 | L30 | L50 | R² size / tokens |
|---|---|---|---|---|---|---|---|
| en | — | 0.39→0.48 [31] | 0.51→0.56 [31] | 0.49→0.51 [31] | 0.57→0.43 [31] | 0.48→0.52 [31] | 0.00 / 0.00 |
| ru | — | — | 0.40→0.45 [11] | 0.66→0.56 [11] | 0.63→0.55 [11] | 0.57→0.74 [11] | 0.00 / 0.08 |
| zh | — | — | 0.40→0.60 [12] | 0.56→0.47 [12] | 0.56→0.48 [12] | 0.50→0.58 [12] | 0.04 / 0.00 |
| de | — | — | 0.50→0.59 [9] | 0.62→0.61 [9] | 0.48→0.59 [9] | 0.67→0.56 [9] | 0.00 / 0.00 |
| ja | — | — | 0.42→0.69 [8] | 0.55→0.42 [8] | 0.35→0.42 [8] | 0.67→0.62 [8] | 0.02 / 0.01 |
| es | — | — | — | 0.43→0.49 [47] | 0.52→0.37 [47] | 0.55→0.53 [47] | 0.04 / 0.09 |
| fr | — | — | — | 0.67→0.50 [12] | 0.45→0.49 [12] | 0.56→0.58 [12] | 0.04 / 0.03 |
| it | — | — | — | 0.73→0.70 [11] | 0.56→0.50 [11] | 0.57→0.56 [11] | 0.07 / 0.01 |

![Scale convergence per language](pretraining/predictivity/scale_convergence_lang_all.png)

![Scale convergence per language, tokens axis](pretraining/predictivity/scale_convergence_lang_all_tokens.png)
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

GitHub: [scale_convergence_lang_all.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_lang_all.png) · [scale_convergence_lang_all.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_lang_all.csv) ·
GitHub: [scale_convergence_lang_all_tokens.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_lang_all_tokens.png) · [scale_convergence_lang_all_tokens.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_lang_all_tokens.csv) ·
[scale_convergence_lang_all_coverage.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_lang_all_coverage.csv) ·
GitHub: [scale_convergence_lang_above_66_size.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_lang_above_66_size.png) · [scale_convergence_lang_above_66_size.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_lang_above_66_size.csv) ·
GitHub: [scale_convergence_lang_above_66_size_tokens.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_lang_above_66_size_tokens.png) · [scale_convergence_lang_above_66_size_tokens.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scale_convergence_lang_above_66_size_tokens.csv)

### 6. High-resource languages are not easier to read

**DA-size · `reliability_by_language_tier`: panel (a) no filter, panel (b)
`above_66_size` · multi-axis · pairs from `predictivity_schemes` · gate
`predictivity`; `reliability_vs_language_share`: no filter, per language with
≥ 3 gated tasks (rule 8).** A tier is the smallest scheme-A regime that trains
the language (L8 = the eight high-resource languages, L50 = the twenty only
the L50 mixture trains).

<!-- BEGIN auto:language-tier (language_tier.py --pool predictivity) -->
## Decision reliability by language tier

The pooled `all pairs` line of `scale_convergence.py` read over the gated tasks of one language TIER — the smallest scheme-A regime that trains the language (L8: the eight high-resource languages every regime trains; L50: the twenty only the L50 mixture trains). Reliability at the smallest → largest proxy [task count]. `reliability_vs_language_share.png` is the per-language version: reliability against the language's share of the L50 mixture, Spearman ρ over languages 175M 0.20, 1B -0.07, 350M 0.10, 600M 0.02. Both are unfiltered; a tier also differs in benchmark mix. Regenerate with `python analysis/rq02_decision_accuracy/language_tier.py --pool predictivity`.

| tier | all gated tasks | above_66_size |
|---|---|---|
| L8 languages (in every regime) | 0.54 → 0.55 [79–141 tasks] | 0.64 → 0.76 [19–28 tasks] |
| L15-only languages | 0.52 → 0.57 [39–67 tasks] | 0.65 → 0.84 [5–7 tasks] |
| L30-only languages | 0.51 → 0.56 [51–86 tasks] | 0.64 → 0.78 [10–13 tasks] |
| L50-only languages | 0.49 → 0.57 [58–93 tasks] | 0.59 → 0.77 [21–33 tasks] |

![Reliability by language tier](pretraining/predictivity/reliability_by_language_tier.png)

![Reliability against language share](pretraining/predictivity/reliability_vs_language_share.png)
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

GitHub: [reliability_by_language_tier.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_by_language_tier.png) · [reliability_by_language_tier.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_by_language_tier.csv) ·
GitHub: [reliability_vs_language_share.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_vs_language_share.png) · [reliability_vs_language_share.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/reliability_vs_language_share.csv)

### 7. Seed uncertainty and the DA-ckpt null

**`seed_uncertainty`: proxy-seed DA = DA-size at each of the proxy's three
seeds against the fixed 1.7B (seed 1904) final; test-retest = the same
designs ranked from seed s1 against seed s2 at one size; seed null = DA-ckpt
over the `seed` pairs (two seeds of ONE design) against the real design pairs
· no filter · pool `predictivity_seeds` · gate `predictivity`.** English at
175M, 600M and 1B and Russian at 1B are the only (size, language) cells where
three replicated designs give ≥ 3 cross-L pairs under rule 2.

<!-- BEGIN auto:seed-uncertainty (seed_uncertainty.py --pool predictivity) -->
## Seed uncertainty

What the replicate seeds say about decision accuracy — English at three proxy sizes and Russian at 1B, the only (size, language) cells where three replicated designs give ≥ 3 cross-L pairs under rule 2. Per row: DA of those decisions against the 1.7B final with the proxy at each of its three seeds (proxy-side noise), and the DA between two seeds' rankings of the same designs at that size (the reference-side ceiling). Three pairs put a DA on {0, ⅓, ⅔, 1}: read the spread, not a mean. The seed null (DA-ckpt over pairs of two seeds of one design, mean over fractions and gated tasks) is 175M: 0.59 vs real 0.66; 600M: 0.56 vs real 0.58; 1B: 0.56 vs real 0.58. Regenerate with `python analysis/rq02_decision_accuracy/seed_uncertainty.py --pool predictivity`.

| size | language | designs | DA at the 3 proxy seeds | test-retest (3 seed pairs) | tasks |
|---|---|---|---|---|---|
| 175M | en | 3 | 0.78, 0.72, 0.83 | 0.83–0.94 | 6 |
| 600M | en | 3 | 0.68, 0.67, 0.61 | 0.54–0.71 | 31 |
| 1B | en | 4 | 0.64, 0.52, 0.57 | 0.56–0.62 | 31 |
| 1B | ru | 4 | 0.82, 0.82, 0.79 | 0.79–0.88 | 11 |

![Seed uncertainty](pretraining/predictivity/seed_uncertainty.png)
<!-- END auto:seed-uncertainty -->

**Key findings**

- Proxy-seed noise is ± 0.05 on English: the three seeds read 0.72 / 0.78 /
  0.83 at 175M (18 decisions on 6 tasks), 0.61 / 0.67 / 0.68 at 600M (93
  decisions, 31 tasks), 0.52 / 0.57 / 0.64 at 1B (186 decisions, 31 tasks);
  Russian at 1B 0.79 / 0.82 / 0.82 on 11 tasks. Three pairs put a DA on
  {0, ⅓, ⅔, 1}: read the spread, not a mean.
- The English test-retest ceiling collapses once the probe families count:
  0.83–0.94 at 175M but 0.54–0.71 at 600M and 0.56–0.62 at 1B (Russian
  0.79–0.88). On the 31 English tasks readable at 600M and 1B (`include_v2_en`,
  `bbh`, `acp_bench` among them) two seeds of the same designs disagree with
  each other as much as the proxy disagrees with the reference: the design
  differences are below the seed noise of those benchmarks.
- DA-ckpt is mostly within-run persistence: two seeds of one design, which
  have nothing to decide, read 0.47 at 10 % of the 175M run and 0.81 at 90 %,
  against 0.51 and 0.83 for the real pairs, and 0.48 → 0.72 against 0.48 →
  0.74 at 1B; at 90 % the real pairs exceed the null by 0.01–0.02 at every
  size, the largest gap mid-run at 175M (0.54 against 0.64 at 40 %). The
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

GitHub: [seed_uncertainty.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/seed_uncertainty.png) · [seed_uncertainty.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/seed_uncertainty.csv)

### 8. Decision accuracy is Kendall's τ under another tie convention

**DA-size's population · no filter · multi-axis pairs · pairs from
`predictivity_schemes` · gate `predictivity`; one point per (task, proxy
size) cell, median 12 models per cell.**

<!-- BEGIN auto:agreement-measures (agreement.py --pool predictivity) -->
## Decision accuracy is Kendall's τ under another tie convention

Over the 1,287 (benchmark task, proxy size) cells of DA-size's population (every pair of the grid-seed variants, ≥ 3 pairs, above chance at the proxy and at 1.7B): 2·DA − 1 = τ_a + (T_both − T_one)/n exactly, where T_both / T_one are the pairs tied on both / one side. Ties are 7,246 of 100,697 pairs (7.2%), 94% of them one-sided, and touch 73% of the cells — so the two statistics correlate at r = 0.971 by construction, and the number with content is how often the tie convention changes a reliability verdict (DA ≥ 0.66, i.e. τ ≥ 0.32), in % of cells per proxy size:

| statistic | 175M | 350M | 600M | 1B |
|---|---|---|---|---|
| da | 0.0 | 0.0 | 0.0 | 0.0 |
| da_drop_ref_ties | 3.5 | 2.5 | 2.8 | 3.4 |
| gamma | 6.2 | 4.7 | 6.2 | 6.2 |
| tau_a | 4.8 | 2.8 | 4.5 | 3.9 |
| tau_b | 5.3 | 4.0 | 5.7 | 4.7 |

Spearman ρ and Pearson r on the raw scores are the two statistics that are NOT a rescaling — they weight a pair by its displacement — and sit at r = 0.961 and 0.904 against DA. Median 12 models per cell. Values in `agreement_per_cell.csv`; regenerate with `python analysis/rq02_decision_accuracy/agreement.py --pool predictivity`.

![DA, Kendall tau and Spearman rho against each other](pretraining/predictivity/agreement_correlation.png)

![DA against Kendall's tau](pretraining/predictivity/agreement_identity.png)

![Cut sensitivity](pretraining/predictivity/agreement_cut_sensitivity.png)
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

GitHub: [agreement_correlation.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_correlation.png) · [agreement_correlation.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_correlation.csv) ·
GitHub: [agreement_identity.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_identity.png) · [agreement_identity.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_identity.csv) ·
GitHub: [agreement_cut_sensitivity.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_cut_sensitivity.png) · [agreement_cut_sensitivity.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_cut_sensitivity.csv) ·
[agreement_per_cell.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/agreement_per_cell.csv)

### 9. Multi-axis against mono-axis pairs

**`rq2_above_66_both_axes`: the three definitions · filter `above_66_both`
(71 cells; the one place it is drawn, for the pair-set comparison only) ·
rows multi-axis and mono-axis · pairs from `predictivity_schemes` · gate
`predictivity`.** Exploratory (`pair_axes.py`); it fed `plan/decision_accuracy.md`
and the `axes` column of rule 15.

![Multi-axis against mono-axis pairs](pretraining/predictivity/rq2_above_66_both_axes.png)

`pair_axes.py` computes the three decision accuracies on the `above_66_both` cells
twice — over every pair at the grid seed (multi-axis, rq02's convention) and over
the pairs that move exactly one of L, depth, list, temperature, second language
(mono-axis, which is what DataDecide's "all pairs" are by construction) — and
writes `rq2_above_66_both_axes.png/.csv` (71 cells on the 2026-09-23 tables).
DA-ckpt is indifferent to the pair set (the 1.7B run 0.68 → 0.83 mono-axis against
0.73 → 0.84 multi-axis from 10 % to 90 %); DA-size reads 0.586, 0.679, 0.698, 0.721
mono-axis against 0.637, 0.726, 0.742, 0.772 multi-axis at 175M … 1B — 0.04–0.05
lower with the same trend — and DA-goal likewise, on 28 % of the decisions (999 of
3,552 at 175M). The proposal that follows from it (an `axes` column in
`da_per_task.csv`, mono-axis as the headline for decisions) is in
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

GitHub: [rq2_above_66_both_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_66_both_axes.png) · [rq2_above_66_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/rq2_above_66_both_axes.csv)

### 10. Cross-task predictability

**DA-size and DA-ckpt · no filter · multi-axis pairs of the A/B pool
`predictivity` (18 variants at 1.7B, 153 pairs; DA-ckpt pools the within-size
pairs of every size, 765) · gate `predictivity` on both sides.** Every parent
task as the proxy for every other one; a cell is the smallest size or earliest
checkpoint at which x's ranking safely predicts y's final ranking (DA ≥ 0.75
over ≥ 3 pairs, held at every larger level).

<!-- BEGIN auto:cross-task (cross_task.py --pool predictivity) -->
## Cross-task predictability

Every parent task as the proxy for every other one (828 x 828): the cell is the smallest proxy size (DA-size, 18 variants at 1.7B, 153 pairs) or the earliest checkpoint (DA-ckpt, the within-size pairs of every size pooled, 765 pairs, the nine checkpoints before the final) at which the ranking on task x (columns) safely predicts the final ranking on task y (rows): DA >= 0.75 over >= 3 pairs there and at every larger level with a value. The diagonal is rq02's own-task DA; the gate empties a benchmark's pairs at every size where it is at chance. The `_by_family` maps take the median level over the task pairs of two benchmarks, the `_by_language` maps over the same-benchmark task pairs of two languages (resource order of the scheme-A lists). Regenerate with `python analysis/rq02_decision_accuracy/cross_task.py --pool predictivity`.

The same two maps over the benchmark tasks that are above chance at some size — BPB, the loss and the benchmarks the gate finds at chance everywhere are dropped, so what is left is the sub-map where a transfer result is possible at all: [`cross_task_size_benchmarks.png`](pretraining/predictivity/cross_task_size_benchmarks.png), [`cross_task_ckpt_benchmarks.png`](pretraining/predictivity/cross_task_ckpt_benchmarks.png).

![Cross-task DA-size by benchmark](pretraining/predictivity/cross_task_size_by_family.png)

![Cross-task DA-ckpt by benchmark](pretraining/predictivity/cross_task_ckpt_by_family.png)

![Cross-task DA-size by language](pretraining/predictivity/cross_task_size_by_language.png)

![Cross-task DA-ckpt by language](pretraining/predictivity/cross_task_ckpt_by_language.png)

Full task-level maps: [`cross_task_size.png`](pretraining/predictivity/cross_task_size.png), [`cross_task_ckpt.png`](pretraining/predictivity/cross_task_ckpt.png).
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
  all; `cross_task_{size,ckpt}_benchmarks.png` is the sub-map where a
  transfer result is possible.

**Follow-ups**

- The family maps show the median of the pairs that succeed while most pairs
  never do; draw the share of task pairs reaching a safe size (in the CSV) as
  the cell and keep the median level as the small number.
- Same-L blocks share the design variants, so "same L list transfers" is
  partly by construction: compare against a permutation baseline (the
  cross-task DA of two languages whose L lists are shuffled).
- Recompute on `predictivity_schemes` (every scheme), as the other figures
  are, so the diagonal is figure 1's population.

GitHub: [cross_task_size_by_family.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_size_by_family.png) · [cross_task_size_by_family.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_size_by_family.csv) ·
GitHub: [cross_task_ckpt_by_family.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_ckpt_by_family.png) · [cross_task_ckpt_by_family.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_ckpt_by_family.csv) ·
GitHub: [cross_task_size_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_size_by_language.png) · [cross_task_size_by_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_size_by_language.csv) ·
GitHub: [cross_task_ckpt_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_ckpt_by_language.png) · [cross_task_ckpt_by_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_ckpt_by_language.csv) ·
GitHub: [cross_task_size.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_size.png) · [cross_task_size.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_size.csv) ·
GitHub: [cross_task_ckpt.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_ckpt.png) · [cross_task_ckpt.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_ckpt.csv) ·
GitHub: [cross_task_size_benchmarks.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_size_benchmarks.png) · [cross_task_size_benchmarks.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_size_benchmarks.csv) ·
GitHub: [cross_task_ckpt_benchmarks.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_ckpt_benchmarks.png) · [cross_task_ckpt_benchmarks.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/cross_task_ckpt_benchmarks.csv)

### 11. Read next in the other RQs

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
| 175M | 167 | +0.12 | +0.16 | +0.38 | 0.54 | 0.46 |
| 350M | 245 | +0.23 | +0.21 | +0.52 | 0.55 | 0.44 |
| 600M | 274 | +0.28 | +0.26 | +0.53 | 0.57 | 0.45 |
| 1B | 274 | +0.37 | +0.34 | +0.62 | 0.59 | 0.46 |

![Scaling against ranking](pretraining/predictivity/scaling_vs_ranking.png)
<!-- END auto:scaling-vs-ranking -->

GitHub: [scaling_vs_ranking.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking.png) · [scaling_vs_ranking.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/scaling_vs_ranking.csv)

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

gemma-3, Qwen3-Base and OLMo-2 each have a base model near 1B and near 13B in the external tier. Per task above chance at both buckets, DA is the share of the three line pairs the small models order like the large ones: pooled 0.73 over 54 tasks, against the ladder's own 1B → 1.7B DA-size of 0.55 on the same tasks. Per line pair: Qwen3 vs OLMo-2 0.87 against a majority baseline of 0.80 (11 minority tasks, DA there 0.55); gemma-3 vs OLMo-2 0.76 against a majority baseline of 0.81 (10 minority tasks, DA there 0.80); gemma-3 vs Qwen3 0.57 against a majority baseline of 0.56 (24 minority tasks, DA there 0.79). The majority baseline is what a proxy scores by always naming the line that usually wins at 12–14B, so only DA above it is information the small models add. Between-lab decisions on public sizes, not the ladder's one-axis ones; three pairs, so per-task values are a lattice. Regenerate with `python analysis/rq02_decision_accuracy/public_ladders.py --pool predictivity`.

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
| belebele | 3 | 0.67 | 0.47 |

![Public ladders](pretraining/predictivity/public_ladders.png)
<!-- END auto:public-ladders -->

**Verdict.** The pooled 0.73 is within 0.07 of a majority-order baseline —
a proxy that always names the line that usually wins at 12–14B scores 0.80 /
0.82 / 0.56 on the same three pairs (DA +0.07, −0.06, +0.02 against it) — so
the figure shows lab-level differences (Qwen3 above OLMo-2 on four tasks in
five at 13B and already at 1.7B, which any leaderboard gives), not
task-level ranking preservation. On the 10–24 minority tasks, where the
reference order is the uncommon one, the small models read it at 0.55–0.80,
better than the ladder's 0.55 but on too few tasks to be a finding. As
evaluated today the public tier cannot say whether benchmarks read decisions
above 1.7B.

**Follow-ups** (the better routes to "which benchmarks preserve ranking at
larger scale"; `plan/next_analyses.md` §7c)

- The ladder's own 3B rung: 1.7B → 3B on four families (deep, L8/L15, A/B)
  once its evals land — the one within-recipe generalisation above the
  current reference (`above_reference=True` in `build_snr_pool`).
- The trend of DA-size across proxies per task: does agreement rise
  monotonically towards the reference? A task whose DA-size climbs 175M →
  1B is the one whose ranking is converging.
- More public lines with both a small and a large base release on the same
  task list (an eval launch of the `auto` list on ~15 public checkpoints),
  close enough in level that the majority baseline is near 0.5.

GitHub: [public_ladders.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/public_ladders.png) · [public_ladders.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/public_ladders.csv) ·
[public_ladders_pairs.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_decision_accuracy/pretraining/predictivity/public_ladders_pairs.csv)

### The 36-model sweep

From the **36-model sweep** (2026-04…06, 4 sizes × 3 data mixtures × 3 seeds,
pools `seeds_1904`, `seeds_28_1797`, `seeds_28_1797_1904`,
`custom_swissai_hf`): `pretraining/<pool>/da_per_task.csv` and the
per-benchmark views are its decision tables with the **1B** reference and the
three FineWeb mixtures as the design variants (three pairs). They are kept as
history and not regenerated — rerun today their `decision_acc_size_*`
columns would point at a 1.7B rung the sweep never trained — and they are
never pooled with the ladder's: a different harness, task set (the 86-task
old list, no twins) and reference. The seed-holdout and per-tier readings
built on them are in rq04's extensions.

## Files

- `da_explainer.py` → `…/da_explainer.{png,csv}` — the toy explainer of the
  three DA kinds, the pair sets and the value lattice (Setup; no measured number).
- `pretraining/<pool>/da_per_task.csv` — the DA table (single source of truth
  for rq03; the `axes` column names the pair set, rule 15);
  `da_n_pairs_per_task.csv` the pair count behind every cell;
  `da_early_small_per_task.csv` the ten-checkpoint grid.
- `…/da_per_benchmark.csv`, `da_per_benchmark_size.csv`,
  `da_per_benchmark_ckpt.csv` — long and wide per-(language, benchmark) views
  (`da_per_benchmark.py`).
- `…/early_small*.csv/.png`, `safe_{size,checkpoint,flops}.*`, `safe_levels.csv`,
  `highlights.*`, `da_size*.*` — `early_small.py` (figure 3).
- `…/da_reliable_tasks.csv`, `da_reliable_by_language.csv`,
  `da_reliable_tasks_<t>_<red>.png` — `reliable_tasks.py` (figure 2).
- `…/da_by_L_per_task.csv`, `da_pooled_per_task.csv`, `pairs_by_L.csv`,
  `early_small_by_L_*` — `by_L.py` (figure 3; rq04's panels read the tables).
- `…/scale_convergence*.csv/.png` — `scale_convergence.py` (figures 1, 4;
  `_L8`, `_L8common`, `_L`, `_transformation`, `_one_axis`, `_above_*`,
  `_flops`); `scale_convergence_lang_*` — `by_language.py` (figure 5).
- `…/rq2*.csv/.png/.svg` — `paper_rq2.py` (figure 2; `rq2.*` is what the
  paper embeds), `rq2_ten_checkpoints.*` — `paper_ten_checkpoints.py`,
  `rq2_above_66_both_axes.*` — `pair_axes.py` (figure 9).
- `…/reliability_by_language_tier.*`, `reliability_vs_language_share.*` —
  `language_tier.py` (figure 6); `seed_uncertainty.*` — `seed_uncertainty.py`
  (figure 7); `agreement_*.*` — `agreement.py` (figure 8);
  `cross_task_*.*` — `cross_task.py` (figure 10);
  `scaling_vs_ranking.*` — `scaling_vs_ranking.py` (rq01 figure 5);
  `public_ladders*.*` — `public_ladders.py` (extension).
- `pretraining/predictivity_schemes/`, `predictivity_seeds*/` — the same
  tables on the other ladder pools; `pretraining/seeds_*`,
  `custom_swissai_hf/`, `all/external/` — the 36-sweep's and the public
  models' (extensions).
- `pretraining/predictivity/README.md` — removed 2026-09-23: its write-up of
  rq00–rq02 was folded into the three RQ READMEs (README rule 1: no README
  under a pool folder).
