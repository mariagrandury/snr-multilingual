# Above-chance items: how much do decision accuracy and SNR rise on the items the reference answers above chance?

## Research question

> If every benchmark-language task keeps only the items its 1.7B runs answer
> above chance, how much do decision accuracy, SNR, the above-random gate and
> the scaling fit improve, and does it matter whether the gate is applied
> before or after the items are chosen?

## Experimental setup

**Setup.**
- **Measured.** We measure what keeping only the items that the 1.7B runs answer above chance does to a task's SNR, signal and noise, and to the agreement of SNR with DA-size.
- **Models.** We use the seed-1904 runs of every cell, ladder and data build, final checkpoints, proxies 90M–1B against the 1.7B reference.
- **Tasks.** The tasks are every benchmark task in a trained language, with the above-chance gate applied before the item selection or recomputed on the kept items after it.
- **Leakage.** The items are chosen with the reference's own answers, so the gains are an upper bound, not a held-out estimate.
- **Two truths.** Decision accuracy is scored against the 1.7B ranking on the full task and on the kept items, and the gap between them is the part of the gain that comes from re-scoring the reference.

- **Rule 11 is waived on request.** The selection uses the reference by
  design, and the numbers measure how much that inflates DA and SNR. Nothing
  here is a held-out estimate. The leakage-free counterpart is the
  [held-out reading of the solved items](../rq08_subset_selection/README.md#items-the-reference-solves)
  in the subset selection analysis.
- **Pool** `predictivity`: seed 1904, every cell (every data build, read as
  scheme A/B/C × temperature T, and every ladder, swiglu included), sizes
  90M–1.7B (rule 10), with the 90M–1B proxies against the 1.7B reference.
  Only the loader's parent tasks in trained languages count (rules 2 and 6).
- **Per-item store** `predictivity` (this run's `--store-pool`, the code
  default), read with its rows filtered to the pool's models. It skips a
  truncated part with a warning. It holds every checkpoint of 173 of the
  pool's 186 runs (the run's log lists the 13 it lacks: the twelve muon
  proxies and `lm-1.7B-L8-swiglu-seed1904`), so DA-ckpt and the checkpoint
  SNR are computed on every size. They are in tables only and not drawn yet,
  but figures 2 and 4 quote them.
- **Gate pool** `predictivity` (the committed mask of the
  [above-random gate](../rq00_gate_and_curves/README.md)). For one ordering
  the same gate is recomputed on the kept items.
- **Snapshot**: the ladder report of 2026-10-10 04:31 and the outputs of the
  full refresh of 2026-10-10 (detrended checkpoint noise). The results block
  states the report it was built from. The gate recomputed on the store's
  full-benchmark scores matches the committed mask on 5,027 of 5,046
  (task, size) cells (results block, "Checks").

## Key figure

![SNR on the above-chance items](pretraining/predictivity/above_chance_items_snr.png)

**Key finding.** **Keeping only the items the 1.7B runs answer above chance
raises the SNR by lowering the noise, not by improving the ranking.** The
median paired SNR ratio grows from ×1.27 at 90M to ×1.50 at 1.7B, because the
k-fold noise falls (×0.58–0.86) while the signal stays flat (×0.86–1.07).
Decision accuracy barely follows, even though the items are chosen with the
reference it is scored against: the paired DA-size gain is +0.041 at 90M and
+0.003 to +0.023 at 175M–1B (multi-axis pairs).

Population: pool `predictivity`, finals, *gate, then items* against the full
benchmark on the cells the committed gate passes (SNR on 304–489 tasks per
size, DA-size on 297–465 tasks per proxy). Details are in
[figure 2](#2-decision-accuracy-rises-a-little-at-the-small-proxies-on-the-same-tasks-only)
and [figure 4](#4-snr-rises-through-the-noise-and-not-the-signal).

## Methodology

- **Above chance at 1.7B.** We keep an item when its per-item metric (the
  task's own, acc or acc_norm, as in the subset selection), averaged over the
  pool's 1.7B final runs that train the task's language (rule 2), exceeds the
  task's chance level. This is the chance level the gate tests
  (`above_random.task_chance`). Each task is reduced on its own. A run's
  sub-benchmark score is its mean over the kept items, and its item count is
  the number of kept items.
- **How many runs select.** The count goes from 4 runs, for a language that
  only the widest mixtures train (Azerbaijani, Catalan, Slovak, ...), to 29
  for English. 161 of the 841 tasks select on 4 runs (results block,
  `n_reference_runs` in the selection table). With 4 runs an item's mean
  moves in steps of 0.25, so when the chance level is below 0.25, an item is
  kept as soon as one of the four runs answers it right.
- **Three orderings**, on the same pool, families and sizes. *Full
  benchmark* is the reference point: the full tasks under the committed gate.
  *Gate, then items* reads the cells the committed gate passes on their kept
  items. These are the same cells as the full benchmark, so its gain is
  paired. *Items, then gate* reduces every task first, then recomputes the
  gate on the kept items with the gate's own code (`scores_and_mask` with the
  kept item counts).
- **Decision accuracy.** We use the decision rows and kernel of
  [decision accuracy](../rq02_decision_accuracy/README.md): DA-size against
  the 1.7B final, with the kept items at both ends, on multi-axis and
  mono-axis pairs, with ≥ 3 pairs (rule 5) and rule 1 at the proxy and the
  reference. It is pooled over the tasks with the leave-one-family-out
  jackknife band.
- **SNR** at the final checkpoint is the `rel_std` signal of
  [noise and SNR](../rq03_noise_and_snr/README.md) over the relative k-fold
  benchmark noise on the task's (kept) items. The k-fold noise is the only
  noise a single checkpoint carries (the surrogate catalogue's
  `snr__rel_std__kfold_rel`). We also report the Spearman ρ of this SNR with
  DA-size over the tasks.
- **Scaling predictability**: the log-N fit of the final scores per
  (task, L) of [scaling predictability](../rq01_scaling_predictability/README.md)
  (`fit_table`), over the rungs the ordering's gate passes.
- **Not computed**: the surrogates, because the catalogue reads a task's item
  count by its name and its truths from the decision-accuracy table on disk.
- **Caveat on the chance level.** The store has no per-item option count, so
  an item of a variable-option family is held to the family's chance level
  (TruthfulQA mc1's E[1/n], mc2's mean true-option share), not its own.

<!-- BEGIN auto:results (above_chance_items.py --pool predictivity --store-pool predictivity) -->
## Results

Pool `predictivity` (90M, 175M, 350M, 600M, 1B, 1.7B), per-item store `predictivity`, ladder report of 2026-10-11 04:43. **Rule 11 is waived by design:** the items are chosen on the pool's 1.7B runs that train the task's language, and DA is scored against the 1.7B reference. The selection uses the reference by design, and the numbers measure how much that inflates DA and SNR. DA-ckpt and the checkpoint SNR are in `above_chance_items_da_ckpt_*` and the `*_ckpt` columns of the SNR table. The surrogates are not computed (the catalogue reads a task's item count by its name and its truths from the decision-accuracy table on disk). Checks: the store's full-benchmark score equals the ladder report's within 1e-3 on 100.0% of 87397 finals; the gate recomputed on them by the same code agrees with the committed mask on 5598 of 5598 (task, size) cells. Regenerate with `python analysis/rq12_above_chance_items/above_chance_items.py --pool predictivity --store-pool predictivity`.

The runs that select a task's items are its 1.7B finals in the cells that train its language (rule 2): 4–32 per task, 170 of 933 tasks on 4 (`n_reference_runs` in the selection table).

**What survives each step** (items kept = items above chance at the reference, over the tasks of the row):

| ordering | step | tasks | items | items kept |
|---|---|---|---|---|
| full benchmark | tasks with a chance level and a reference run | 933 | 2419903 | 2419903 (100%) |
| full benchmark | ... the gate passes at 1.7B | 595 | 1860222 | 1860222 (100%) |
| gate, then items | tasks the gate passes at 1.7B | 595 | 1860222 | 790767 (43%) |
| gate, then items | ... with an item above chance at 1.7B | 595 | 1860222 | 790767 (43%) |
| items, then gate | tasks with an item above chance at 1.7B | 932 | 2419653 | 1020152 (42%) |
| items, then gate | ... the recomputed gate passes at 1.7B | 910 | 2417720 | 1019312 (42%) |

**The gate's pass share** per size: the tasks passing, over every task with a chance level and a reference run. Gate, then items uses the committed gate, so it coincides with the full benchmark:

| ordering | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| full benchmark | 0.40 (377) | 0.44 (409) | 0.48 (448) | 0.53 (498) | 0.59 (554) | 0.64 (595) |
| gate, then items | 0.40 (377) | 0.44 (409) | 0.48 (448) | 0.53 (498) | 0.59 (554) | 0.64 (595) |
| items, then gate | 0.86 (804) | 0.84 (783) | 0.85 (797) | 0.88 (824) | 0.91 (848) | 0.98 (910) |

**DA-size** against the 1.7B final, pooled over the tasks, with its 90 % leave-one-family-out jackknife band and the task count. Rule 1 applies at the proxy and at 1.7B, with ≥ 3 pairs (rule 5, cells with fewer pairs are NaN: no cell has fewer). The task count moves along a row and between the orderings (rule 13):

| pairs | ordering | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| multi-axis | full benchmark | 0.525 [0.51, 0.54] (363) | 0.534 [0.51, 0.55] (405) | 0.534 [0.51, 0.55] (440) | 0.547 [0.53, 0.57] (496) | 0.555 [0.53, 0.58] (550) |
| multi-axis | gate, then items | 0.548 [0.53, 0.57] (363) | 0.547 [0.52, 0.57] (405) | 0.544 [0.52, 0.57] (440) | 0.555 [0.53, 0.58] (496) | 0.549 [0.52, 0.58] (550) |
| multi-axis | items, then gate | 0.512 [0.50, 0.53] (797) | 0.518 [0.50, 0.53] (779) | 0.519 [0.50, 0.54] (797) | 0.526 [0.51, 0.54] (824) | 0.526 [0.51, 0.54] (847) |
| mono-axis | full benchmark | 0.506 [0.49, 0.52] (363) | 0.506 [0.48, 0.53] (405) | 0.504 [0.49, 0.52] (440) | 0.511 [0.49, 0.53] (496) | 0.517 [0.49, 0.54] (550) |
| mono-axis | gate, then items | 0.513 [0.49, 0.54] (363) | 0.507 [0.48, 0.53] (405) | 0.516 [0.50, 0.53] (440) | 0.513 [0.49, 0.54] (496) | 0.510 [0.48, 0.54] (550) |
| mono-axis | items, then gate | 0.484 [0.47, 0.50] (797) | 0.493 [0.48, 0.51] (779) | 0.500 [0.48, 0.52] (797) | 0.496 [0.48, 0.51] (824) | 0.497 [0.48, 0.52] (847) |

**The DA-size gain**, read three ways: the paired difference from the full benchmark, averaged over the (task, proxy) cells both readings have, the difference of the pooled lines (which weight a task by its pairs), and the DA on the cells that items, then gate admits but the committed gate blanks on the full benchmark. Items, then gate − full equals gate, then items − full on 4506 of 4508 paired (task, pair set, proxy) cells (the same sub-scores wherever both have a value), so only the latter is shown. Both are in `above_chance_items_da_size_both_axes.csv`:

| pairs | reading | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| multi-axis | gate, then items − full: mean paired Δ over the tasks (cells) | +0.045 (363) | +0.030 (405) | +0.033 (440) | +0.025 (496) | +0.002 (550) |
| multi-axis | gate, then items − full: Δ of the pooled DA-size (the figure's lines) | +0.024 | +0.012 | +0.010 | +0.008 | -0.006 |
| multi-axis | items, then gate: mean DA on the admitted cells (cells) | 0.472 (435) | 0.496 (374) | 0.489 (357) | 0.485 (328) | 0.487 (297) |
| multi-axis | gate, then items, against the full task's 1.7B ranking − against its kept items' (pooled) | +0.011 | +0.016 | +0.007 | +0.007 | +0.015 |
| multi-axis | gate, then items − full, full-task truth: mean paired Δ (the gain the re-scored reference does not supply) | +0.053 | +0.038 | +0.034 | +0.026 | +0.011 |
| mono-axis | gate, then items − full: mean paired Δ over the tasks (cells) | +0.023 (363) | +0.011 (405) | +0.032 (440) | +0.013 (496) | +0.001 (550) |
| mono-axis | gate, then items − full: Δ of the pooled DA-size (the figure's lines) | +0.007 | +0.001 | +0.013 | +0.002 | -0.007 |
| mono-axis | items, then gate: mean DA on the admitted cells (cells) | 0.452 (435) | 0.481 (374) | 0.473 (357) | 0.470 (328) | 0.470 (297) |
| mono-axis | gate, then items, against the full task's 1.7B ranking − against its kept items' (pooled) | +0.013 | +0.016 | -0.003 | +0.004 | +0.010 |
| mono-axis | gate, then items − full, full-task truth: mean paired Δ (the gain the re-scored reference does not supply) | +0.031 | +0.023 | +0.026 | +0.014 | +0.004 |

The admitted cells by the number of items their task keeps (mean DA-size over every proxy, cells):

| pairs | 1-30 kept items | 31-100 kept items | 101-300 kept items | 301-1000 kept items | > 1000 kept items |
|---|---|---|---|---|---|
| multi-axis | 0.399 (239) | 0.466 (456) | 0.503 (498) | 0.513 (343) | 0.529 (255) |
| mono-axis | 0.382 (239) | 0.450 (456) | 0.491 (498) | 0.494 (343) | 0.506 (255) |

**Tied sub-benchmarks.** Where the models answer a two-option task with a constant bias, the items above chance are the ones whose gold matches the bias, and every design variant scores alike on them. A pair tied at the proxy and at the reference counts as agreeing (decision accuracy's tie convention), so such a cell has a DA-size of 1. Cells where every variant ties at the proxy or at 1.7B (`tied` in the per-task table):

| pairs | ordering | cells with a DA-size value | of which every variant ties | their mean DA-size |
|---|---|---|---|---|
| multi-axis | full benchmark | 2254 | 6 | 0.55 |
| multi-axis | gate, then items | 2254 | 11 | 0.49 |
| multi-axis | items, then gate | 4044 | 58 | 0.77 |
| mono-axis | full benchmark | 2254 | 6 | 0.55 |
| mono-axis | gate, then items | 2254 | 11 | 0.50 |
| mono-axis | items, then gate | 4044 | 58 | 0.77 |

**SNR** at the final checkpoint (rel_std signal over the relative k-fold noise of the task's items, on the tasks above chance at the size) and its gain on the paired cells:

| reading | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| full benchmark: median SNR (tasks) | 0.304 (374) | 0.294 (407) | 0.311 (446) | 0.329 (496) | 0.339 (550) | 0.349 (590) |
| gate, then items: median SNR (tasks) | 0.416 (363) | 0.425 (401) | 0.459 (439) | 0.505 (489) | 0.515 (543) | 0.524 (570) |
| items, then gate: median SNR (tasks) | 0.394 (776) | 0.397 (757) | 0.415 (768) | 0.479 (790) | 0.509 (807) | 0.563 (842) |
| gate, then items / full: median paired SNR ratio | 1.29 | 1.35 | 1.40 | 1.44 | 1.50 | 1.51 |
| gate, then items / full: median paired signal ratio | 1.07 | 1.09 | 1.09 | 1.05 | 1.00 | 0.89 |
| gate, then items / full: median paired k-fold noise ratio | 0.86 | 0.82 | 0.79 | 0.73 | 0.67 | 0.58 |
| items, then gate: median SNR on the admitted cells (cells) | 0.382 (417) | 0.369 (357) | 0.389 (333) | 0.430 (301) | 0.494 (264) | 0.803 (272) |

**Does SNR track DA-size?** Spearman ρ(SNR, DA-size multi-axis) over the tasks at each proxy (tasks):

| ordering | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| full benchmark | 0.37 (361) | 0.33 (403) | 0.37 (438) | 0.44 (494) | 0.30 (546) |
| gate, then items | 0.42 (355) | 0.40 (397) | 0.44 (431) | 0.45 (487) | 0.41 (539) |
| items, then gate | 0.30 (771) | 0.35 (753) | 0.33 (768) | 0.34 (790) | 0.35 (806) |

**Scaling predictability**: the log-N fit of the final scores per (task, L) on the deep data-A cells at the grid seed, over the rungs the ordering's gate passes:

| ordering | fits | median R² | median ρ with size | median ΔR² (paired) | paired fits |
|---|---|---|---|---|---|
| full benchmark | 1391 | 0.87 | 1.00 |  |  |
| gate, then items | 1389 | 0.93 | 1.00 | +0.036 | 1389 |
| items, then gate | 2293 | 0.87 | 0.94 | +0.032 | 1390 |
<!-- END auto:results -->

## Figures, in storyline order

### 1. Choosing the items first makes the gate almost vacuous

![The gate's pass share](pretraining/predictivity/above_chance_items_gate_pass_share.png)

Pool `predictivity`, per-item store finals, every task with a chance level
and a 1.7B run (841). The gate is the committed mask, or for *items, then
gate* the one recomputed on the kept items.

**Key findings**

- **Fewer than half the items are kept.** The kept items are 45 % of the
  items of the 507 tasks the committed gate passes at 1.7B, and 44 % over all
  840 tasks with an item above chance.
- **Recomputed on the kept items, the gate passes almost every task.** It
  passes 816 of those 840 tasks at 1.7B and 83–90 % of the 841 at the
  proxies, against 37 % (90M) to 60 % (1.7B) for the full benchmarks.
  Choosing the items on the reference all but guarantees that the reference
  passes, and it raises the pass share at every proxy too.
- **As a result, *items, then gate* scores DA-size on more than half again
  as many tasks.** It scores 695–755 tasks per proxy, against 297–465 for the
  full benchmark. The next figure asks whether the added tasks rank anything.

**Follow-ups**

- The pass share as a function of each task's kept-item count (or of its
  1.7B margin above chance), to show how few items the in-sample selection
  needs before the gate's verdict at the proxies stops tracking that of the
  full benchmark.

GitHub: [above_chance_items_gate_pass_share.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_gate_pass_share.png) · [above_chance_items_gate_pass_share.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_gate_pass_share.csv)

### 2. Decision accuracy rises a little at the small proxies, on the same tasks only

![DA-size on the above-chance items](pretraining/predictivity/above_chance_items_da_size_both_axes.png)

DA-size against the 1.7B final, no reliability filter, multi-axis (left) and
mono-axis (right) pairs, gate pool `predictivity` at the proxy and the
reference, pooled over the tasks with the 90 % leave-one-family-out band.

**Key findings**

- **On the same cells, the kept items rank slightly better.** The paired
  gain of *gate, then items*, averaged over the 297–465 tasks per proxy, is
  +0.041 at 90M, +0.018 to +0.023 at 175M–600M and +0.003 at 1B (multi-axis,
  and +0.006 to +0.029 on mono-axis pairs). The pooled lines move by −0.009
  to +0.014 (multi-axis) and −0.006 to +0.009 (mono-axis), inside the full
  benchmark's jackknife band at every proxy.
- **The gain does not come from re-scoring the reference.** When the truth
  stays the full task's 1.7B ranking and only the proxy is read on the kept
  items, the paired gain is +0.051 at 90M, +0.026 to +0.036 at 175M–600M and
  +0.007 at 1B (multi-axis, and +0.003 to +0.036 on mono-axis pairs). This is
  slightly more than against the kept items' own ranking (results block).
- **The tasks admitted by the recomputed gate pull the pooled line down.**
  Pooled, the full benchmark reaches 0.536–0.560 (multi-axis), *gate, then
  items* 0.536–0.552 and *items, then gate* 0.511–0.527 (695–755 tasks).
- **The admitted cells rank at or below chance.** The 290–408 admitted
  cells per proxy reach 0.470–0.500 (multi-axis, 0.500 at 175M, and
  0.452–0.485 on mono-axis pairs). They rank better the more items their task
  keeps: 0.398 at 1–30 kept items, 0.469, 0.502 and 0.520 in the middle bins
  and 0.537 above 1,000.
- **Some admitted cells score high only because of the tie convention.** In
  53 of the 3,593 *items, then gate* cells, every design variant ties at the
  proxy or at the reference. This happens when the models answer a
  two-option task with a constant bias: the kept items are then exactly those
  whose gold matches the bias. These cells have a mean DA-size of 0.79.
- **The held-out counterpart gains about as much.** The
  [held-out reading of the solved items](../rq08_subset_selection/README.md#items-the-reference-solves)
  in the subset selection analysis (same pool and store, multi-axis, 238–375
  paired tasks) gains +0.015 to +0.033 at 90M–600M and +0.004 at 1B. The two
  selections differ (items that at least half of the 1.7B runs solve, against
  items above chance), so the closeness bounds the leakage only loosely.
- **Along training, the kept items agree with the final ranking earlier at
  every proxy.** On DA-ckpt (each size's checkpoints at 10–90 % of its run
  against its own final ranking, multi-axis, `above_chance_items_da_ckpt_*`),
  the mean paired gain is +0.02 to +0.06 over the nine checkpoints at 90M–1B.
  At 1.7B, the size the items were chosen on, it is only −0.010 to +0.026. At
  90 % of the run the pooled DA-ckpt is 0.757–0.818 for the full benchmark at
  90M–1B (0.737 at 1.7B) and 0.767–0.831 for *gate, then items* (0.725).

**Follow-ups**

- A figure of DA-ckpt on the kept items (`above_chance_items_da_ckpt_*`, the
  bullet above), read against the seed null.

GitHub: [above_chance_items_da_size_both_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_both_axes.png) · [above_chance_items_da_size_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_both_axes.csv)

### 3. Per benchmark: the small-proxy gains fade by 1B

![DA-size per benchmark, multi-axis](pretraining/predictivity/above_chance_items_da_size_by_benchmark_multi_axes.png)

The same cells per benchmark (mean over its languages), on multi-axis
pairs. The mono-axis twin is
[above_chance_items_da_size_by_benchmark_mono_axis.png](pretraining/predictivity/above_chance_items_da_size_by_benchmark_mono_axis.png).

**Key findings**

- **The per-benchmark gains are small and shrink by 1B.** Over the
  benchmarks with at least five paired tasks, the paired gain stays between
  −0.08 and +0.19 at 90M–600M, and between −0.11 and +0.06 at 1B.
- **The items chosen at 1.7B raise agreement mostly at the smallest
  proxies.** Global-MMLU-RF gains most at the small proxies (+0.19 at 90M on
  12 tasks, +0.06 to +0.07 at 175M–600M) and loses at 1B (−0.02 on 28). HellaSwag
  follows the same pattern (+0.15 at 90M on 26 tasks, +0.02 at 1B on 26).
  XNLI loses at 175M–1B (−0.02 to −0.08, 12–13 tasks), and XCOPA loses at 1B
  (−0.11, 8 tasks).
- **The *items, then gate* panel adds benchmarks that the full benchmark
  cannot score at that proxy, and they sit near or below chance.** Global-MMLU
  reaches 0.45–0.57 (23–29 tasks), Global PIQA parallel 0.39–0.44 (60–63) and
  CulturalBench-hard 0.38–0.44 (10–18). ACP-Bench cloze reaches 0.91–0.98 on
  its 7 tasks only because of the tie convention (34 of its 35 multi-axis
  cells have every variant tied at the proxy or at 1.7B).

**Follow-ups**

- Rank the benchmarks by their paired gain with a per-benchmark jackknife, so
  a ±0.05 cell can be told from noise.

GitHub: [above_chance_items_da_size_by_benchmark_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_by_benchmark_multi_axes.png) · [above_chance_items_da_size_by_benchmark_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_by_benchmark_multi_axes.csv)
GitHub: [above_chance_items_da_size_by_benchmark_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_by_benchmark_mono_axis.png) · [above_chance_items_da_size_by_benchmark_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_by_benchmark_mono_axis.csv)

### 4. SNR rises, through the noise and not the signal

![SNR on the above-chance items](pretraining/predictivity/above_chance_items_snr.png)

Final-checkpoint SNR (rel_std signal over the relative k-fold noise), cells
above chance at the size under each ordering's gate, pool `predictivity`.
The ratios are on the cells both readings have and are drawn for *gate, then
items* only, because *items, then gate* has the same sub-scores there. ρ is
with DA-size on multi-axis pairs.

**Key findings**

- **The median SNR rises under both item orderings, and more at larger
  sizes.** It goes from 0.287–0.347 (full benchmark, 308–504 tasks per size)
  to 0.392–0.519 (*gate, then items*, 304–489) and 0.382–0.547 (*items, then
  gate*, 676–759). The paired ratio grows with size, from ×1.27 at 90M to
  ×1.50 at 1.7B.
- **The gain comes from the noise falling, not from the signal rising.** The
  signal barely moves (×0.86–1.07) while the relative k-fold noise falls
  (×0.58–0.86). The kept items are answered right more often, which lowers
  the noise, but the design variants do not separate more.
- **SNR tracks DA-size a little better on the kept items.** ρ is 0.39–0.47
  for *gate, then items*, against 0.35–0.45 for the full benchmark and
  0.26–0.34 for *items, then gate*. SNR rises by a quarter to a half while
  DA-size gains at most +0.041. So most of the higher SNR reflects an easier
  test, not a better ranking.
- **On the checkpoint SNR, the kept items gain little.** On the 20 % window
  of the noise and SNR analysis (the `*_ckpt` columns of the SNR table), the
  median paired ratio is ×1.03–1.10 at 90M–1B and ×0.97 at 1.7B (signal
  ×0.86–1.07, checkpoint noise ×0.89–0.99). So the final-checkpoint gain comes
  mostly from the k-fold noise, which depends on the accuracy and the item
  count, both of which the selection changes.

**Follow-ups**

- A figure of the checkpoint SNR beside the final one, per size, to show the
  two readings part on the kept items.

GitHub: [above_chance_items_snr.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr.png) · [above_chance_items_snr.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr.csv)

### 5. SNR per benchmark

![SNR per benchmark](pretraining/predictivity/above_chance_items_snr_by_benchmark.png)

Median final-checkpoint SNR per benchmark and size under each ordering, and
the median paired log2 ratio for *gate, then items*.

**Key findings**

- **The SNR gain varies widely across benchmarks.** Take the median paired
  log2 ratio over the sizes with at least five paired tasks, read as a ratio.
  It runs from ×0.82 (CulturalBench-easy, 90M only) to ×2.40 (Global-MMLU-RF,
  over all six sizes), with HellaSwag at ×1.65 over all six. Among the
  benchmarks that gain, the lowest is ×1.10 (MultiBLiMP).
- **Every such benchmark gains SNR except one.** The exception is
  CulturalBench-easy (×0.82, at its only such size, 90M, on 6 tasks).
- **The largest *items, then gate* SNRs are on benchmarks that barely pass
  the gate on their full items.** Global-MMLU reaches 4.95–6.45 on 23–29
  tasks, while its full benchmark passes in at most two languages at any size
  and in none at 90M, 600M or 1B. MMLU reaches 1.90–5.23 (one task, never
  passed). Thousands of kept items shrink the k-fold noise, while Global-MMLU's
  DA-size in figure 3 stays at 0.45–0.57.

**Follow-ups**

- Put the SNR gain against the DA-size gain per (task, proxy), to show
  directly that the two move independently on this selection.

GitHub: [above_chance_items_snr_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr_by_benchmark.png) · [above_chance_items_snr_by_benchmark.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr_by_benchmark.csv)

### Scaling predictability

The last table of the results block covers the deep data-A cells. There
the median R² of the log-N fit rises from 0.88 (1,188 fits) to 0.93 for
*gate, then items* (+0.033 on 1,187 paired fits, read on the same rungs as
the full benchmark). *Items, then gate* fits 2,082 (task, L) series at a
median of 0.87. Easier items track size more smoothly once the gate is
fixed. Admitting the tasks that the recomputed gate adds brings the median
back to about that of the full benchmark.

### The key figure on one task set per line

Each line of the key figure takes the median over the tasks with a value at that size. The gate keeps more tasks at larger sizes (308 → 504 on the full benchmark), so a line can move just because its tasks changed (rule 13). This twin figure keeps, for each ordering, only the tasks with a value at every size (solid), with the committed line dashed behind it. It is rebuilt from the per-task CSVs, and its moving reading reproduces `above_chance_items_snr.csv` exactly.

![SNR panels on one task set per line](pretraining/predictivity/above_chance_items_snr_fixed_tasks_paper.png)

Pool `predictivity`, seed 1904, finals, ρ against DA-size on multi-axis pairs (each ordering's own truth). GitHub: [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr_fixed_tasks_paper.csv)

Key findings:

- **Part of the SNR rise under *items, then gate* comes from its changing task set.** On its 616 tasks with a value at every size, the median SNR goes from 0.37 to 0.50 between 90M and 1.7B, against 0.39 to 0.55 on 676–759 tasks. The full and *gate, then items* lines move by 0.02 at most.
- **The paired gain still grows with size, though less steeply.** *Gate, then items* goes from x1.30 to x1.51 on its 287 tasks, against x1.27 to x1.50.
- **ρ(SNR, DA-size) stays positive and is a little higher.** It is 0.29–0.49 on fixed tasks, against 0.26–0.47. The full benchmark gains the most (up to +0.08 at 1B).

Follow-ups:

- **The fixed twin against the full-task truth.** The SNR table now carries the full-truth ρ (`rho_snr_da_size_full_truth`). The fixed twin does not read it yet, so the dashed ρ lines have no fixed version.

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

From `src/signal-and-noise` with the `snr` env. It takes about ten minutes
on the store with checkpoints. `run_all_predictivity.sh` runs it after the
per-item store steps, and `--store-pool` reads another store folder:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 \
  python analysis/rq12_above_chance_items/above_chance_items.py --pool predictivity --store-pool predictivity
```

Without a store, or with one that does not cover the pool, it prints why and
writes nothing. `--out-dir` writes the tables and figures elsewhere and
leaves this README alone. `tests/test_above_chance_items.py` pins the
selection.
