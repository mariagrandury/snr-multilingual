# Above-chance items: how much do decision accuracy and SNR rise on the items the reference answers above chance?

## Research question

> If every benchmark-language task keeps only the items its 1.7B runs answer
> above chance, how much do decision accuracy, SNR, the above-random gate and
> the scaling fit improve, and does it matter whether the gate is applied
> before or after the items are chosen?

## Experimental setup

- **Rule 11 is waived by request: the selection reads the reference by
  design; the numbers measure how much that inflates DA and SNR.** Nothing
  here is a held-out estimate; the subset selection's
  [held-out reading of the solved items](../rq08_subset_selection/README.md#items-the-reference-solves)
  is the leakage-free counterpart.
- **Pool** `predictivity`: seed 1904, every cell (every data build, read as
  scheme A/B/C × temperature T, and every ladder, swiglu included), sizes
  90M–1.7B (rule 10), 90M–1B proxies against the 1.7B reference, the
  loader's parent tasks in trained languages only (rules 2 and 6).
- **Per-item store** `predictivity_schemes` (this run's `--store-pool`; the
  code default is now the `predictivity` store), read with its rows filtered
  to the pool's models; it skips a truncated part with a warning. The store
  lacks 3 of the pool's models (lm-90M-L1-fweb-b84-deep, lm-350M-L1-fweb-deep
  and lm-600M-L1-fweb-deep, seed 1904); they are left out of every number
  here until the store is rebuilt for the pool. The store holds checkpoints, so
  DA-ckpt and the checkpoint SNR are computed (tables only, not drawn yet).
- **Gate pool** `predictivity` (the committed mask of the
  [above-random gate](../rq00_gate_and_curves/README.md)), and for one ordering
  the same gate recomputed on the kept items.
- **Snapshot**: the ladder report of 2026-10-06 04:26, outputs regenerated
  and prose re-read on 2026-10-07 (the results block states the report it
  was built from). The gate recomputed on the store's full-benchmark scores
  matches the committed mask on 5,042 of 5,046 (task, size) cells (results
  block, "Checks").

## Key figure

![SNR on the above-chance items](pretraining/predictivity/above_chance_items_snr.png)

**Key finding.** Keeping only the items the 1.7B runs answer above chance
raises the median paired SNR from ×1.26 at 90M to ×1.52 at 1.7B, almost all
through a lower k-fold noise (×0.57–0.85) while the signal stays flat
(×0.87–1.07). Decision accuracy barely follows, although the selection reads
the reference it is scored against: the paired DA-size gain is +0.043 at 90M
and +0.000 to +0.020 at 175M–1B (multi-axis pairs).

Population: pool `predictivity`, finals, *gate, then items* against the full
benchmark on the cells the committed gate passes (SNR on 303–475 tasks per
size, DA-size on 293–446 tasks per proxy); details in
[figure 2](#2-decision-accuracy-rises-a-little-at-the-small-proxies-on-the-same-tasks-only)
and [figure 4](#4-snr-rises-through-the-noise-and-not-the-signal).

## Methodology

- **Above chance at 1.7B.** An item is kept when the mean of its per-item
  metric (the task's own, acc or acc_norm, as the subset selection reads it)
  over the pool's 1.7B final runs that train the task's language (rule 2)
  exceeds the task's chance level, the one the gate tests
  (`above_random.task_chance`). Each task is reduced on its own; a run's
  sub-benchmark score is its mean over the kept items, its item count the
  number kept.
- **How many runs select.** From 4 runs for a language only the widest
  mixtures train (Azerbaijani, Catalan, Slovak, ...) to 26 for English; 161
  of the 841 tasks select on 4 (results block, `n_reference_runs` in the
  selection table). With 4 runs an item's mean moves in steps of 0.25, so
  below a chance level of 0.25 an item is kept as soon as one of the four
  runs answers it.
- **Three orderings**, same pool, families and sizes. *Full benchmark*: the
  full tasks under the committed gate, the reference point. *Gate, then
  items*: the cells the committed gate passes, read on their kept items (the
  same cells as the full benchmark, so its gain is paired). *Items, then
  gate*: every task reduced first, then the gate recomputed on the kept
  items by its own code (`scores_and_mask` with the kept item counts).
- **Decision accuracy** ([decision accuracy](../rq02_decision_accuracy/README.md)'s
  decision rows and kernel): DA-size against the 1.7B final on the kept items
  at both ends, multi-axis and mono-axis pairs, ≥ 3 pairs (rule 5), rule 1 at
  the proxy and the reference, pooled over the tasks with the
  leave-one-family-out jackknife band.
- **SNR** at the final checkpoint: the `rel_std` signal of
  [noise and SNR](../rq03_noise_and_snr/README.md) over the relative k-fold
  benchmark noise on the task's (kept) items, the one noise a single
  checkpoint carries (the surrogate catalogue's `snr__rel_std__kfold_rel`),
  and its Spearman ρ with DA-size over the tasks.
- **Scaling predictability**: the log-N fit of the final scores per
  (task, L) of [scaling predictability](../rq01_scaling_predictability/README.md)
  (`fit_table`), over the rungs the ordering's gate passes.
- **Not computed**: the surrogates, because the catalogue reads a task's item
  count by its name and its truths from the decision-accuracy table on disk.
- **Caveat on the chance level.** The store has no per-item option count, so
  an item of a variable-option family is held to the family's chance level
  (TruthfulQA mc1's E[1/n], mc2's mean true-option share), not its own.

<!-- BEGIN auto:results (above_chance_items.py --pool predictivity --store-pool predictivity_schemes) -->
## Results

Pool `predictivity` (90M, 175M, 350M, 600M, 1B, 1.7B), per-item store `predictivity_schemes`, ladder report of 2026-10-06 04:26. **Rule 11 is waived by design:** the items are chosen on the pool's 1.7B runs that train the task's language and DA is scored against the 1.7B reference: the selection reads the reference by design, and the numbers measure how much that inflates DA and SNR. DA-ckpt and the checkpoint SNR are in `above_chance_items_da_ckpt_*` and the `*_ckpt` columns of the SNR table. The surrogates are not computed (the catalogue reads a task's item count by its name and its truths from the decision-accuracy table on disk). Checks: the store's full-benchmark score equals the ladder report's within 1e-3 on 100.0% of 63155 finals; the gate recomputed on them by the same code agrees with the committed mask on 5042 of 5046 (task, size) cells. Regenerate with `python analysis/rq12_above_chance_items/above_chance_items.py --pool predictivity --store-pool predictivity_schemes`.

The runs that select a task's items are its 1.7B finals in the cells that train its language (rule 2): 4–26 per task, 161 of 841 tasks on 4 (`n_reference_runs` in the selection table).

**What survives each step** (items kept = items above chance at the reference, over the tasks of the row):

| ordering | step | tasks | items | items kept |
|---|---|---|---|---|
| full benchmark | tasks with a chance level and a reference run | 841 | 1847293 | 1847293 (100%) |
| full benchmark | ... the gate passes at 1.7B | 489 | 1271289 | 1271289 (100%) |
| gate, then items | tasks the gate passes at 1.7B | 489 | 1271289 | 544479 (43%) |
| gate, then items | ... with an item above chance at 1.7B | 489 | 1271289 | 544479 (43%) |
| items, then gate | tasks with an item above chance at 1.7B | 840 | 1847043 | 771156 (42%) |
| items, then gate | ... the recomputed gate passes at 1.7B | 826 | 1846036 | 770788 (42%) |

**The gate's pass share** per size (tasks passing, over every task with a chance level and a reference run; gate, then items reads the committed gate and coincides with the full benchmark):

| ordering | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| full benchmark | 0.37 (312) | 0.40 (336) | 0.44 (373) | 0.49 (409) | 0.54 (450) | 0.58 (489) |
| gate, then items | 0.37 (312) | 0.40 (336) | 0.44 (373) | 0.49 (409) | 0.54 (450) | 0.58 (489) |
| items, then gate | 0.85 (715) | 0.82 (693) | 0.84 (706) | 0.88 (736) | 0.90 (757) | 0.98 (826) |

**DA-size** against the 1.7B final, pooled over the tasks, its 90 % leave-one-family-out jackknife band and the task count; rule 1 at the proxy and at 1.7B, ≥ 3 pairs (rule 5: no cell has fewer and are NaN); the task count moves along a row and between the orderings (rule 13):

| pairs | ordering | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| multi-axis | full benchmark | 0.536 [0.51, 0.56] (293) | 0.545 [0.52, 0.57] (333) | 0.536 [0.51, 0.56] (366) | 0.548 [0.53, 0.57] (408) | 0.562 [0.53, 0.60] (446) |
| multi-axis | gate, then items | 0.553 [0.53, 0.58] (293) | 0.546 [0.52, 0.57] (333) | 0.539 [0.51, 0.57] (366) | 0.551 [0.53, 0.58] (408) | 0.552 [0.52, 0.59] (446) |
| multi-axis | items, then gate | 0.506 [0.49, 0.52] (712) | 0.512 [0.50, 0.53] (692) | 0.513 [0.49, 0.53] (705) | 0.525 [0.51, 0.54] (735) | 0.527 [0.50, 0.55] (756) |
| mono-axis | full benchmark | 0.504 [0.48, 0.52] (293) | 0.517 [0.49, 0.54] (333) | 0.506 [0.49, 0.52] (366) | 0.517 [0.50, 0.54] (408) | 0.522 [0.49, 0.56] (446) |
| mono-axis | gate, then items | 0.514 [0.49, 0.54] (293) | 0.513 [0.49, 0.54] (333) | 0.509 [0.49, 0.53] (366) | 0.517 [0.50, 0.54] (408) | 0.515 [0.48, 0.55] (446) |
| mono-axis | items, then gate | 0.482 [0.47, 0.50] (712) | 0.490 [0.47, 0.51] (692) | 0.491 [0.47, 0.51] (705) | 0.502 [0.49, 0.52] (735) | 0.502 [0.48, 0.53] (756) |

**The DA-size gain**: the paired difference over the full benchmark averaged over the (task, proxy) cells both readings have, the difference of the pooled lines (which weight a task by its pairs), and the cells items, then gate admits that the committed gate blanks on the full benchmark. Items, then gate − full equals gate, then items − full on 3690 of 3692 paired (task, pair set, proxy) cells (the same sub-scores wherever both have a value), so only the latter is shown; both are in `above_chance_items_da_size_both_axes.csv`:

| pairs | reading | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| multi-axis | gate, then items − full: mean paired Δ over the tasks (cells) | +0.043 (293) | +0.018 (333) | +0.020 (366) | +0.020 (408) | +0.000 (446) |
| multi-axis | gate, then items − full: Δ of the pooled DA-size (the figure's lines) | +0.017 | +0.001 | +0.002 | +0.003 | -0.010 |
| multi-axis | items, then gate: mean DA on the admitted cells (cells) | 0.470 (420) | 0.494 (359) | 0.485 (339) | 0.503 (327) | 0.498 (310) |
| mono-axis | gate, then items − full: mean paired Δ over the tasks (cells) | +0.031 (293) | +0.004 (333) | +0.019 (366) | +0.009 (408) | +0.001 (446) |
| mono-axis | gate, then items − full: Δ of the pooled DA-size (the figure's lines) | +0.011 | -0.004 | +0.004 | +0.000 | -0.007 |
| mono-axis | items, then gate: mean DA on the admitted cells (cells) | 0.456 (420) | 0.478 (359) | 0.467 (339) | 0.491 (327) | 0.488 (310) |

The admitted cells by the number of items their task keeps (mean DA-size over every proxy, cells):

| pairs | 1-30 kept items | 31-100 kept items | 101-300 kept items | 301-1000 kept items | > 1000 kept items |
|---|---|---|---|---|---|
| multi-axis | 0.410 (302) | 0.477 (378) | 0.494 (541) | 0.530 (318) | 0.546 (216) |
| mono-axis | 0.397 (302) | 0.473 (378) | 0.482 (541) | 0.509 (318) | 0.517 (216) |

**Tied sub-benchmarks.** Where the models answer a two-option task by a constant bias, the items above chance are the ones whose gold matches it, and every design variant scores alike on them; a pair tied at the proxy and at the reference counts as agreeing (decision accuracy's tie convention), so such a cell reads DA-size 1. Cells where every variant ties at the proxy or at 1.7B (`tied` in the per-task table):

| pairs | ordering | cells with a DA-size value | of which every variant ties | their mean DA-size |
|---|---|---|---|---|
| multi-axis | full benchmark | 1846 | 1 | 0.52 |
| multi-axis | gate, then items | 1846 | 6 | 0.44 |
| multi-axis | items, then gate | 3600 | 49 | 0.87 |
| mono-axis | full benchmark | 1846 | 1 | 0.56 |
| mono-axis | gate, then items | 1846 | 6 | 0.46 |
| mono-axis | items, then gate | 3600 | 49 | 0.87 |

**SNR** at the final checkpoint (rel_std signal over the relative k-fold noise of the task's items; tasks above chance at the size) and its gain on the paired cells:

| reading | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| full benchmark: median SNR (tasks) | 0.300 (309) | 0.290 (334) | 0.290 (371) | 0.318 (407) | 0.331 (448) | 0.353 (486) |
| gate, then items: median SNR (tasks) | 0.398 (303) | 0.407 (333) | 0.413 (369) | 0.475 (405) | 0.509 (444) | 0.515 (475) |
| items, then gate: median SNR (tasks) | 0.380 (690) | 0.386 (672) | 0.384 (682) | 0.447 (709) | 0.499 (721) | 0.541 (773) |
| gate, then items / full: median paired SNR ratio | 1.26 | 1.36 | 1.36 | 1.43 | 1.51 | 1.52 |
| gate, then items / full: median paired signal ratio | 1.06 | 1.07 | 1.05 | 1.02 | 0.99 | 0.87 |
| gate, then items / full: median paired k-fold noise ratio | 0.85 | 0.82 | 0.78 | 0.73 | 0.67 | 0.57 |
| items, then gate: median SNR on the admitted cells (cells) | 0.372 (392) | 0.360 (340) | 0.368 (317) | 0.411 (304) | 0.468 (277) | 0.672 (298) |

**Does SNR track DA-size?** Spearman ρ(SNR, DA-size multi-axis) over the tasks at each proxy (tasks):

| ordering | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| full benchmark | 0.29 (291) | 0.32 (331) | 0.31 (364) | 0.41 (406) | 0.34 (444) |
| gate, then items | 0.39 (290) | 0.37 (330) | 0.36 (362) | 0.47 (404) | 0.42 (440) |
| items, then gate | 0.27 (689) | 0.33 (671) | 0.27 (682) | 0.33 (708) | 0.34 (720) |

**Scaling predictability**: the log-N fit of the final scores per (task, L) on the deep data-A cells at the grid seed, over the rungs the ordering's gate passes:

| ordering | fits | median R² | median ρ with size | median ΔR² (paired) | paired fits |
|---|---|---|---|---|---|
| full benchmark | 1143 | 0.88 | 1.00 |  |  |
| gate, then items | 1141 | 0.93 | 1.00 | +0.034 | 1141 |
| items, then gate | 2095 | 0.87 | 0.94 | +0.028 | 1143 |
<!-- END auto:results -->

## Figures, in storyline order

### 1. Choosing the items first makes the gate almost vacuous

![The gate's pass share](pretraining/predictivity/above_chance_items_gate_pass_share.png)

Pool `predictivity`, per-item store finals, every task with a chance level
and a 1.7B run (841); the gate is the committed mask, or the one recomputed
on the kept items for *items, then gate*.

**Key findings**

- The kept items are 43 % of the items of the 489 tasks the committed gate
  passes at 1.7B, and 42 % over all 840 tasks with an item above chance.
- Recomputed on the kept items, the gate passes 826 of those 840 tasks at
  1.7B and 82–90 % of the 841 at the proxies, against 37 % (90M) to 58 %
  (1.7B) for the full benchmarks: the in-sample selection all but guarantees
  the reference's verdict and lifts every proxy's.
- So *items, then gate* scores DA-size on 692–756 tasks per proxy against
  293–446 for the full benchmark: the next figure asks whether the tasks it
  adds rank anything.

**Follow-ups**

- The pass share as a function of each task's kept-item count (or of the
  1.7B margin above chance), to show how few items the in-sample selection
  needs before the gate's verdict at the proxies stops tracking the full
  benchmark's.

GitHub: [above_chance_items_gate_pass_share.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_gate_pass_share.png) · [above_chance_items_gate_pass_share.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_gate_pass_share.csv)

### 2. Decision accuracy rises a little at the small proxies, on the same tasks only

![DA-size on the above-chance items](pretraining/predictivity/above_chance_items_da_size_both_axes.png)

DA-size against the 1.7B final, no reliability filter, multi-axis (left) and
mono-axis (right) pairs, gate pool `predictivity` at the proxy and the
reference, pooled over the tasks with the 90 % leave-one-family-out band.

**Key findings**

- On the same cells, the kept items rank slightly better: the paired gain of
  *gate, then items*, averaged over the 293–446 tasks per proxy, is +0.043 at
  90M, +0.018 to +0.020 at 175M–600M and +0.000 at 1B (multi-axis; mono-axis
  +0.001 to +0.031). The pooled lines move −0.010 to +0.017 (multi-axis) and
  −0.007 to +0.011 (mono-axis), inside the full benchmark's jackknife band at
  every proxy.
- Pooled, the full benchmark reads 0.536–0.562 (multi-axis), *gate, then
  items* 0.539–0.553 and *items, then gate* 0.506–0.527 (692–756 tasks): the
  tasks the recomputed gate admits pull the pooled line down, not up.
- The 310–420 admitted cells per proxy rank at or below a coin flip,
  0.470–0.503 (multi-axis, 0.503 at 600M; mono-axis 0.456–0.491), and read higher the more items their
  task keeps: 0.410 at 1–30 kept items, 0.477, 0.494 and 0.530 in the middle
  bins, 0.546 above 1,000.
- 49 of the 3,600 *items, then gate* cells have every design variant tied at
  the proxy or the reference (a two-option task answered by a constant bias
  keeps exactly the items whose gold matches it); they read a mean DA-size of
  0.87 from the tie convention alone.
- The subset selection's
  [held-out reading of the solved items](../rq08_subset_selection/README.md#items-the-reference-solves)
  (same pool and store, multi-axis, 234–358 paired tasks) gains +0.022 to
  +0.033 at 90M–600M and +0.002 at 1B: about the in-sample gain here. The
  selections differ (items at least half of the 1.7B runs solve, against
  items above chance), so the closeness bounds the leakage only loosely.

**Follow-ups**

- DA-size with the reference read on the full benchmark (only the proxy on
  the kept items), to separate a moved target from a better proxy.
- A figure of DA-ckpt on the kept items (`above_chance_items_da_ckpt_*`, now
  computed), read against the seed null.

GitHub: [above_chance_items_da_size_both_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_both_axes.png) · [above_chance_items_da_size_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_both_axes.csv)

### 3. Per benchmark: the small-proxy gains fade by 1B

![DA-size per benchmark, multi-axis](pretraining/predictivity/above_chance_items_da_size_by_benchmark_multi_axes.png)

The same cells per benchmark (mean over its languages), multi-axis pairs;
the mono-axis twin is
[above_chance_items_da_size_by_benchmark_mono_axis.png](pretraining/predictivity/above_chance_items_da_size_by_benchmark_mono_axis.png).

**Key findings**

- Over the benchmarks with at least five paired tasks, the paired gain stays
  within −0.07 and +0.23 at 90M–600M, and within −0.12 and +0.06 at 1B.
- Global-MMLU-RF gains most at the small proxies (+0.23 at 90M on 12 tasks,
  +0.07 to +0.09 at 175M–600M) and loses at 1B (−0.02 on 28); HellaSwag
  likewise (+0.15 at 90M on 23, +0.01 at 1B on 25). XNLI loses at 175M–1B
  (−0.02 to −0.07, 12–13 tasks) and XCOPA at 1B (−0.12, 8 tasks): the items
  chosen at 1.7B make the smallest proxies agree with it more, not the
  larger ones.
- The *items, then gate* panel fills in benchmarks the full benchmark does
  not read at that proxy, near or below a coin flip: Global-MMLU 0.46–0.55
  (19–29 tasks), Global PIQA parallel 0.40–0.47 (58–63), CulturalBench-hard
  0.39–0.44 (10–18). ACP-Bench cloze reads 0.94–0.98 on its 7 tasks only
  through the tie convention: 34 of its 35 multi-axis cells have every variant tied at
  the proxy or at 1.7B.

**Follow-ups**

- Rank the benchmarks by their paired gain with a per-benchmark jackknife, so
  a ±0.05 cell can be told from noise.

GitHub: [above_chance_items_da_size_by_benchmark_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_by_benchmark_multi_axes.png) · [above_chance_items_da_size_by_benchmark_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_by_benchmark_multi_axes.csv)
GitHub: [above_chance_items_da_size_by_benchmark_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_by_benchmark_mono_axis.png) · [above_chance_items_da_size_by_benchmark_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_by_benchmark_mono_axis.csv)

### 4. SNR rises, through the noise and not the signal

![SNR on the above-chance items](pretraining/predictivity/above_chance_items_snr.png)

Final-checkpoint SNR (rel_std signal over the relative k-fold noise), cells
above chance at the size under each ordering's gate, pool `predictivity`;
ratios on the cells both readings have (drawn for *gate, then items* only:
*items, then gate* reads the same sub-scores there); ρ with DA-size
multi-axis.

**Key findings**

- The median SNR rises from 0.290–0.353 (full benchmark, 309–486 tasks per
  size) to 0.398–0.515 (*gate, then items*, 303–475) and 0.380–0.541
  (*items, then gate*, 672–773); the paired ratio grows with size, ×1.26 at
  90M to ×1.52 at 1.7B.
- The signal barely moves (×0.87–1.07): the gain is the relative k-fold noise
  falling (×0.57–0.85), because the kept items are answered right more often,
  not because the design variants separate more.
- SNR tracks DA-size a little better on the kept items: ρ = 0.36–0.47 for
  *gate, then items* against 0.29–0.41 for the full benchmark, and 0.27–0.34
  for *items, then gate*. SNR rises by a quarter to a half while DA-size
  gains at most +0.043: most of the higher SNR is the SNR of an easier test,
  not of a better ranking.

**Follow-ups**

- A figure of the checkpoint SNR (the 20 % window, now in the `*_ckpt`
  columns of the SNR table): the k-fold noise is a function of the accuracy and the
  item count, so it is the reading most exposed to this selection.

GitHub: [above_chance_items_snr.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr.png) · [above_chance_items_snr.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr.csv)

### 5. SNR per benchmark

![SNR per benchmark](pretraining/predictivity/above_chance_items_snr_by_benchmark.png)

Median final-checkpoint SNR per benchmark and size under each ordering, and
the median paired log2 ratio for *gate, then items*.

**Key findings**

- Per benchmark, the median paired log2 ratio over the sizes with at least
  five paired tasks, read as a ratio, runs from ×0.86 (CulturalBench-easy,
  90M only) to ×2.40 (Global-MMLU-RF, over all six sizes), with HellaSwag at
  ×1.89 over all six; among the benchmarks that gain, the lowest is ×1.11
  (MultiBLiMP).
- Every such benchmark gains SNR but CulturalBench-easy (×0.86, its one such
  size, 90M, 8 tasks).
- The largest SNRs of *items, then gate* sit on benchmarks the full benchmark
  barely passes: Global-MMLU 4.5–7.0 on 19–29 tasks (the full benchmark
  passes in at most two languages at any size, none at 90M, 600M or 1B) and
  MMLU 2.4–5.5 (one task, never passed). Thousands of kept items shrink the
  k-fold noise, while Global-MMLU's DA-size in figure 3 stays at 0.46–0.55.

**Follow-ups**

- Put the SNR gain against the DA-size gain per (task, proxy), to show
  directly that the two move independently on this selection.

GitHub: [above_chance_items_snr_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr_by_benchmark.png) · [above_chance_items_snr_by_benchmark.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr_by_benchmark.csv)

### Scaling predictability

The results block's last table: on the deep data-A cells the median R² of
the log-N fit rises from 0.88 (1,143 fits) to 0.93 for *gate, then items*
(+0.034 on 1,141 paired fits, read on the same rungs as the full benchmark),
and *items, then gate* fits 2,095 (task, L) series at a median of 0.87.
Easier items track size more smoothly once the gate is fixed; admitting the
tasks the recomputed gate adds brings the median back to about the full
benchmark's.

## Outputs

Under `pretraining/<pool>/`, every name prefixed `above_chance_items_`:

| file | what |
|---|---|
| `selection.csv` | per task: items, kept items, chance level, metric, 1.7B runs, the committed and the recomputed gate at 1.7B |
| `survival.csv` | per ordering: tasks and items left after each step |
| `gate_pass_share.png/.csv` | per ordering and size: tasks the gate passes |
| `da_size_per_task_both_axes.csv` | per ordering, task, size, pair set: DA-size, pairs, `gated`, `tied` |
| `da_size_both_axes.png/.csv` | per ordering, pair set, size: pooled DA-size, jackknife band, mean over tasks, paired gain, admitted cells |
| `da_size_by_benchmark_multi_axes.png/.csv`, `..._mono_axis` | per benchmark and proxy: mean DA-size per ordering and the paired gain |
| `snr_per_task.csv` | per ordering, task, size: signal, k-fold noise, SNR, and the checkpoint SNR (`*_ckpt`) |
| `snr.png/.csv` | per ordering and size: median SNR, paired ratios of SNR, signal and noise, ρ(SNR, DA-size), admitted cells |
| `snr_by_benchmark.png/.csv` | per benchmark and size: median SNR per ordering and the paired log2 ratio |
| `scaling_fits.csv` | per ordering, task, L: the log-N fit |
| `da_ckpt_per_task_both_axes.csv`, `da_ckpt_both_axes.csv` | DA-ckpt per ordering, task, size, pair set, and pooled (written when the store holds checkpoints) |

## How to run

From `src/signal-and-noise` with the `snr` env (about ten minutes on the
store with checkpoints; `run_all_predictivity.sh` runs it after the per-item
store steps; `--store-pool` reads another store folder):

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
  python analysis/rq12_above_chance_items/above_chance_items.py --pool predictivity --store-pool predictivity
```

Without a store, or with one that does not cover the pool, it prints why and
writes nothing; `--out-dir` writes the tables and figures elsewhere and
leaves this README alone. `tests/test_above_chance_items.py` pins the
selection.
