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
- **Per-item store** `predictivity` (this run's `--store-pool`, the code
  default), read with its rows filtered to the pool's models; it skips a
  truncated part with a warning. It holds every checkpoint of 164 of the
  pool's 173 runs (the run's log lists the nine it lacks: the eight muon
  proxies and `lm-1.7B-L8-swiglu-seed1904`), so DA-ckpt and the checkpoint
  SNR are computed on every size (tables only, not drawn yet; figures 2 and 4
  quote them).
- **Gate pool** `predictivity` (the committed mask of the
  [above-random gate](../rq00_gate_and_curves/README.md)), and for one ordering
  the same gate recomputed on the kept items.
- **Snapshot**: the ladder report of 2026-10-08 12:06 (regenerated locally:
  acc_norm on the cloze-format originals) and the outputs of the full refresh
  of 2026-10-08 (detrended checkpoint noise); the results block states the
  report it was built from. The gate recomputed on the store's full-benchmark
  scores matches the committed mask on 5,033 of 5,046 (task, size) cells
  (results block, "Checks").

## Key figure

![SNR on the above-chance items](pretraining/predictivity/above_chance_items_snr.png)

**Key finding.** **Keeping only the items the 1.7B runs answer above chance
raises the SNR through a lower noise, not a better ranking.** The median
paired SNR rises from ×1.27 at 90M to ×1.50 at 1.7B as the k-fold noise falls
(×0.58–0.86) and the signal stays flat (×0.86–1.07), while decision accuracy
barely follows, although the selection reads the reference it is scored
against: the paired DA-size gain is +0.041 at 90M and +0.003 to +0.023 at
175M–1B (multi-axis pairs).

Population: pool `predictivity`, finals, *gate, then items* against the full
benchmark on the cells the committed gate passes (SNR on 304–489 tasks per
size, DA-size on 297–465 tasks per proxy); details in
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
  mixtures train (Azerbaijani, Catalan, Slovak, ...) to 29 for English; 161
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

<!-- BEGIN auto:results (above_chance_items.py --pool predictivity --store-pool predictivity) -->
## Results

Pool `predictivity` (90M, 175M, 350M, 600M, 1B, 1.7B), per-item store `predictivity`, ladder report of 2026-10-08 12:06. **Rule 11 is waived by design:** the items are chosen on the pool's 1.7B runs that train the task's language and DA is scored against the 1.7B reference: the selection reads the reference by design, and the numbers measure how much that inflates DA and SNR. DA-ckpt and the checkpoint SNR are in `above_chance_items_da_ckpt_*` and the `*_ckpt` columns of the SNR table. The surrogates are not computed (the catalogue reads a task's item count by its name and its truths from the decision-accuracy table on disk). Checks: the store's full-benchmark score equals the ladder report's within 1e-3 on 100.0% of 71204 finals; the gate recomputed on them by the same code agrees with the committed mask on 5033 of 5046 (task, size) cells. Regenerate with `python analysis/rq12_above_chance_items/above_chance_items.py --pool predictivity --store-pool predictivity`.

The runs that select a task's items are its 1.7B finals in the cells that train its language (rule 2): 4–29 per task, 161 of 841 tasks on 4 (`n_reference_runs` in the selection table).

**What survives each step** (items kept = items above chance at the reference, over the tasks of the row):

| ordering | step | tasks | items | items kept |
|---|---|---|---|---|
| full benchmark | tasks with a chance level and a reference run | 841 | 1847293 | 1847293 (100%) |
| full benchmark | ... the gate passes at 1.7B | 507 | 1288682 | 1288682 (100%) |
| gate, then items | tasks the gate passes at 1.7B | 507 | 1288682 | 577420 (45%) |
| gate, then items | ... with an item above chance at 1.7B | 507 | 1288682 | 577420 (45%) |
| items, then gate | tasks with an item above chance at 1.7B | 840 | 1847043 | 804664 (44%) |
| items, then gate | ... the recomputed gate passes at 1.7B | 816 | 1844732 | 803711 (44%) |

**The gate's pass share** per size (tasks passing, over every task with a chance level and a reference run; gate, then items reads the committed gate and coincides with the full benchmark):

| ordering | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| full benchmark | 0.37 (311) | 0.40 (340) | 0.45 (378) | 0.50 (424) | 0.56 (468) | 0.60 (507) |
| gate, then items | 0.37 (311) | 0.40 (340) | 0.45 (378) | 0.50 (424) | 0.56 (468) | 0.60 (507) |
| items, then gate | 0.84 (709) | 0.83 (697) | 0.84 (705) | 0.87 (734) | 0.90 (756) | 0.97 (816) |

**DA-size** against the 1.7B final, pooled over the tasks, its 90 % leave-one-family-out jackknife band and the task count; rule 1 at the proxy and at 1.7B, ≥ 3 pairs (rule 5: no cell has fewer and are NaN); the task count moves along a row and between the orderings (rule 13):

| pairs | ordering | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| multi-axis | full benchmark | 0.536 [0.52, 0.56] (297) | 0.544 [0.52, 0.57] (337) | 0.535 [0.51, 0.56] (372) | 0.545 [0.52, 0.57] (424) | 0.560 [0.53, 0.59] (465) |
| multi-axis | gate, then items | 0.551 [0.53, 0.57] (297) | 0.546 [0.52, 0.57] (337) | 0.535 [0.51, 0.56] (372) | 0.546 [0.52, 0.57] (424) | 0.552 [0.52, 0.58] (465) |
| multi-axis | items, then gate | 0.511 [0.50, 0.52] (704) | 0.515 [0.50, 0.53] (695) | 0.514 [0.49, 0.54] (705) | 0.522 [0.50, 0.54] (734) | 0.527 [0.51, 0.55] (755) |
| mono-axis | full benchmark | 0.507 [0.49, 0.53] (297) | 0.519 [0.49, 0.54] (337) | 0.505 [0.49, 0.52] (372) | 0.513 [0.49, 0.53] (424) | 0.523 [0.49, 0.56] (465) |
| mono-axis | gate, then items | 0.517 [0.49, 0.54] (297) | 0.513 [0.49, 0.54] (337) | 0.508 [0.49, 0.53] (372) | 0.510 [0.49, 0.54] (424) | 0.519 [0.49, 0.55] (465) |
| mono-axis | items, then gate | 0.485 [0.47, 0.50] (704) | 0.495 [0.48, 0.51] (695) | 0.492 [0.47, 0.51] (705) | 0.495 [0.48, 0.51] (734) | 0.504 [0.48, 0.53] (755) |

**The DA-size gain**: the paired difference over the full benchmark averaged over the (task, proxy) cells both readings have, the difference of the pooled lines (which weight a task by its pairs), and the cells items, then gate admits that the committed gate blanks on the full benchmark. Items, then gate − full equals gate, then items − full on 3788 of 3790 paired (task, pair set, proxy) cells (the same sub-scores wherever both have a value), so only the latter is shown; both are in `above_chance_items_da_size_both_axes.csv`:

| pairs | reading | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| multi-axis | gate, then items − full: mean paired Δ over the tasks (cells) | +0.041 (297) | +0.023 (337) | +0.022 (372) | +0.017 (424) | +0.003 (465) |
| multi-axis | gate, then items − full: Δ of the pooled DA-size (the figure's lines) | +0.014 | +0.001 | -0.000 | +0.001 | -0.009 |
| multi-axis | items, then gate: mean DA on the admitted cells (cells) | 0.470 (408) | 0.500 (358) | 0.490 (333) | 0.490 (310) | 0.491 (290) |
| multi-axis | gate, then items, against the full task's 1.7B ranking − against its kept items' (pooled) | +0.011 | +0.020 | +0.011 | +0.013 | +0.013 |
| multi-axis | gate, then items − full, full-task truth: mean paired Δ (the gain the re-scored reference does not supply) | +0.051 | +0.036 | +0.031 | +0.025 | +0.007 |
| mono-axis | gate, then items − full: mean paired Δ over the tasks (cells) | +0.029 (297) | +0.007 (337) | +0.022 (372) | +0.007 (424) | +0.006 (465) |
| mono-axis | gate, then items − full: Δ of the pooled DA-size (the figure's lines) | +0.009 | -0.006 | +0.003 | -0.002 | -0.004 |
| mono-axis | items, then gate: mean DA on the admitted cells (cells) | 0.452 (408) | 0.485 (358) | 0.470 (333) | 0.475 (310) | 0.474 (290) |
| mono-axis | gate, then items, against the full task's 1.7B ranking − against its kept items' (pooled) | +0.009 | +0.019 | +0.000 | +0.009 | +0.006 |
| mono-axis | gate, then items − full, full-task truth: mean paired Δ (the gain the re-scored reference does not supply) | +0.036 | +0.023 | +0.021 | +0.014 | +0.003 |

The admitted cells by the number of items their task keeps (mean DA-size over every proxy, cells):

| pairs | 1-30 kept items | 31-100 kept items | 101-300 kept items | 301-1000 kept items | > 1000 kept items |
|---|---|---|---|---|---|
| multi-axis | 0.399 (231) | 0.469 (426) | 0.502 (516) | 0.520 (316) | 0.537 (210) |
| mono-axis | 0.387 (231) | 0.449 (426) | 0.488 (516) | 0.504 (316) | 0.512 (210) |

**Tied sub-benchmarks.** Where the models answer a two-option task by a constant bias, the items above chance are the ones whose gold matches it, and every design variant scores alike on them; a pair tied at the proxy and at the reference counts as agreeing (decision accuracy's tie convention), so such a cell reads DA-size 1. Cells where every variant ties at the proxy or at 1.7B (`tied` in the per-task table):

| pairs | ordering | cells with a DA-size value | of which every variant ties | their mean DA-size |
|---|---|---|---|---|
| multi-axis | full benchmark | 1895 | 1 | 0.46 |
| multi-axis | gate, then items | 1895 | 6 | 0.42 |
| multi-axis | items, then gate | 3593 | 53 | 0.79 |
| mono-axis | full benchmark | 1895 | 1 | 0.50 |
| mono-axis | gate, then items | 1895 | 6 | 0.45 |
| mono-axis | items, then gate | 3593 | 53 | 0.79 |

**SNR** at the final checkpoint (rel_std signal over the relative k-fold noise of the task's items; tasks above chance at the size) and its gain on the paired cells:

| reading | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| full benchmark: median SNR (tasks) | 0.306 (308) | 0.287 (338) | 0.290 (376) | 0.321 (422) | 0.331 (466) | 0.347 (504) |
| gate, then items: median SNR (tasks) | 0.392 (304) | 0.406 (337) | 0.415 (374) | 0.478 (420) | 0.501 (462) | 0.519 (489) |
| items, then gate: median SNR (tasks) | 0.386 (688) | 0.382 (676) | 0.391 (683) | 0.455 (705) | 0.493 (722) | 0.547 (759) |
| gate, then items / full: median paired SNR ratio | 1.27 | 1.35 | 1.35 | 1.42 | 1.50 | 1.50 |
| gate, then items / full: median paired signal ratio | 1.07 | 1.06 | 1.05 | 1.02 | 0.99 | 0.86 |
| gate, then items / full: median paired k-fold noise ratio | 0.86 | 0.82 | 0.79 | 0.73 | 0.67 | 0.58 |
| items, then gate: median SNR on the admitted cells (cells) | 0.380 (390) | 0.371 (340) | 0.374 (313) | 0.421 (285) | 0.490 (260) | 0.845 (270) |

**Does SNR track DA-size?** Spearman ρ(SNR, DA-size multi-axis) over the tasks at each proxy (tasks):

| ordering | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| full benchmark | 0.38 (295) | 0.38 (335) | 0.37 (370) | 0.44 (422) | 0.35 (463) |
| gate, then items | 0.42 (294) | 0.41 (334) | 0.40 (368) | 0.47 (420) | 0.40 (459) |
| items, then gate | 0.28 (684) | 0.31 (674) | 0.26 (683) | 0.34 (705) | 0.34 (721) |

**Scaling predictability**: the log-N fit of the final scores per (task, L) on the deep data-A cells at the grid seed, over the rungs the ordering's gate passes:

| ordering | fits | median R² | median ρ with size | median ΔR² (paired) | paired fits |
|---|---|---|---|---|---|
| full benchmark | 1185 | 0.88 | 1.00 |  |  |
| gate, then items | 1184 | 0.93 | 1.00 | +0.033 | 1184 |
| items, then gate | 2082 | 0.87 | 0.94 | +0.028 | 1185 |
<!-- END auto:results -->

## Figures, in storyline order

### 1. Choosing the items first makes the gate almost vacuous

![The gate's pass share](pretraining/predictivity/above_chance_items_gate_pass_share.png)

Pool `predictivity`, per-item store finals, every task with a chance level
and a 1.7B run (841); the gate is the committed mask, or the one recomputed
on the kept items for *items, then gate*.

**Key findings**

- **Fewer than half the items are kept.** They are 45 % of the items of the
  507 tasks the committed gate passes at 1.7B, and 44 % over all 840 tasks
  with an item above chance.
- **Recomputed on the kept items, the gate passes almost every task.** It
  passes 816 of those 840 tasks at 1.7B and 83–90 % of the 841 at the
  proxies, against 37 % (90M) to 60 % (1.7B) for the full benchmarks: the
  in-sample selection all but guarantees the reference's verdict and lifts
  every proxy's.
- **So *items, then gate* scores DA-size on more than half again as many
  tasks.** It reads 695–755 tasks per proxy against 297–465 for the full
  benchmark; the next figure asks whether the tasks it adds rank anything.

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

- **On the same cells, the kept items rank slightly better.** The paired
  gain of *gate, then items*, averaged over the 297–465 tasks per proxy, is
  +0.041 at 90M, +0.017 to +0.023 at 175M–600M and +0.003 at 1B (multi-axis;
  mono-axis +0.006 to +0.029); the pooled lines move −0.009 to +0.014
  (multi-axis) and −0.006 to +0.009 (mono-axis), inside the full benchmark's
  jackknife band at every proxy.
- **The gain is not the re-scored reference moving towards the proxy.**
  Read against the full task's 1.7B ranking (only the proxy on the kept
  items), the paired gain is +0.051 at 90M, +0.025 to +0.036 at 175M–600M and
  +0.007 at 1B (multi-axis; mono-axis +0.003 to +0.036), slightly more than
  against the kept items' own ranking (results block).
- **The tasks the recomputed gate admits pull the pooled line down, not
  up.** Pooled, the full benchmark reads 0.535–0.560 (multi-axis), *gate,
  then items* 0.535–0.552 and *items, then gate* 0.511–0.527 (695–755 tasks).
- **The admitted cells rank at or below a coin flip.** The 290–408 admitted
  cells per proxy read 0.470–0.500 (multi-axis, 0.500 at 175M; mono-axis
  0.452–0.485), and higher the more items their task keeps: 0.399 at 1–30
  kept items, 0.469, 0.502 and 0.520 in the middle bins, 0.537 above 1,000.
- **Some admitted cells read high from the tie convention alone.** 53 of
  the 3,593 *items, then gate* cells have every design variant tied at the
  proxy or the reference (a two-option task answered by a constant bias keeps
  exactly the items whose gold matches it); they read a mean DA-size of 0.79.
- **The held-out counterpart gains about as much.** The subset selection's
  [held-out reading of the solved items](../rq08_subset_selection/README.md#items-the-reference-solves)
  (same pool and store, multi-axis, 238–375 paired tasks) gains +0.013 to
  +0.033 at 90M–600M and +0.004 at 1B; the selections differ (items at least
  half of the 1.7B runs solve, against items above chance), so the closeness
  bounds the leakage only loosely.
- **Along training the kept items agree with the final ranking earlier at
  every proxy.** On DA-ckpt (each size's checkpoints at 10–90 % of its run
  against its own final ranking, multi-axis, `above_chance_items_da_ckpt_*`)
  the mean paired gain is +0.02 to +0.06 over the nine checkpoints at 90M–1B,
  but −0.010 to +0.026 at 1.7B, the size the items were chosen on. At 90 % of
  the run the pooled DA-ckpt is 0.757–0.818 for the full benchmark at 90M–1B
  (0.737 at 1.7B) and 0.767–0.831 for *gate, then items* (0.725).

**Follow-ups**

- A figure of DA-ckpt on the kept items (`above_chance_items_da_ckpt_*`, the
  bullet above), read against the seed null.

GitHub: [above_chance_items_da_size_both_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_both_axes.png) · [above_chance_items_da_size_both_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_da_size_both_axes.csv)

### 3. Per benchmark: the small-proxy gains fade by 1B

![DA-size per benchmark, multi-axis](pretraining/predictivity/above_chance_items_da_size_by_benchmark_multi_axes.png)

The same cells per benchmark (mean over its languages), multi-axis pairs;
the mono-axis twin is
[above_chance_items_da_size_by_benchmark_mono_axis.png](pretraining/predictivity/above_chance_items_da_size_by_benchmark_mono_axis.png).

**Key findings**

- **The per-benchmark gains are small and shrink by 1B.** Over the
  benchmarks with at least five paired tasks, the paired gain stays within
  −0.08 and +0.19 at 90M–600M, and within −0.11 and +0.06 at 1B.
- **The items chosen at 1.7B make the smallest proxies agree with it more,
  not the larger ones.** Global-MMLU-RF gains most at the small proxies
  (+0.19 at 90M on 12 tasks, +0.06 at 175M–600M) and loses at 1B (−0.02 on
  28); HellaSwag likewise (+0.15 at 90M on 26, +0.02 at 1B on 26), while XNLI
  loses at 175M–1B (−0.02 to −0.08, 12–13 tasks) and XCOPA at 1B (−0.11, 8
  tasks).
- **The *items, then gate* panel fills in benchmarks the full benchmark does
  not read at that proxy, near or below a coin flip.** Global-MMLU reads
  0.45–0.57 (23–29 tasks), Global PIQA parallel 0.39–0.44 (60–63) and
  CulturalBench-hard 0.38–0.44 (10–18); ACP-Bench cloze reads 0.91–0.98 on
  its 7 tasks only through the tie convention (34 of its 35 multi-axis cells
  have every variant tied at the proxy or at 1.7B).

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

- **The median SNR rises under both item orderings, more with size.** It
  goes from 0.287–0.347 (full benchmark, 308–504 tasks per size) to
  0.392–0.519 (*gate, then items*, 304–489) and 0.382–0.547 (*items, then
  gate*, 676–759); the paired ratio grows with size, ×1.27 at 90M to ×1.50 at
  1.7B.
- **The gain is the noise falling, not the signal rising.** The signal barely
  moves (×0.86–1.07) while the relative k-fold noise falls (×0.58–0.86),
  because the kept items are answered right more often, not because the
  design variants separate more.
- **SNR tracks DA-size a little better on the kept items.** ρ = 0.40–0.47
  for *gate, then items* against 0.35–0.44 for the full benchmark, and
  0.26–0.34 for *items, then gate*; SNR rises by a quarter to a half while
  DA-size gains at most +0.041, so most of the higher SNR is the SNR of an
  easier test, not of a better ranking.
- **On the checkpoint SNR the kept items gain little.** On noise and SNR's
  20 % window (the `*_ckpt` columns of the SNR table) the median paired ratio
  is ×1.03–1.10 at 90M–1B and ×0.97 at 1.7B (signal ×0.86–1.07, checkpoint
  noise ×0.89–0.99); the final-checkpoint gain is mostly the k-fold noise's,
  which depends on the accuracy and the item count the selection changes.

**Follow-ups**

- A figure of the checkpoint SNR beside the final one, per size, to show the
  two readings part on the kept items.

GitHub: [above_chance_items_snr.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr.png) · [above_chance_items_snr.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr.csv)

### 5. SNR per benchmark

![SNR per benchmark](pretraining/predictivity/above_chance_items_snr_by_benchmark.png)

Median final-checkpoint SNR per benchmark and size under each ordering, and
the median paired log2 ratio for *gate, then items*.

**Key findings**

- **The SNR gain varies widely across benchmarks.** The median paired
  log2 ratio over the sizes with at least five paired tasks, read as a ratio,
  runs from ×0.82 (CulturalBench-easy, 90M only) to ×2.40 (Global-MMLU-RF,
  over all six sizes), with HellaSwag at ×1.65 over all six; among the
  benchmarks that gain, the lowest is ×1.10 (MultiBLiMP).
- **Every such benchmark gains SNR but one.** The exception is
  CulturalBench-easy (×0.82, its one such size, 90M, 6 tasks).
- **The largest SNRs of *items, then gate* sit on benchmarks the full
  benchmark barely passes.** Global-MMLU reads 4.95–6.45 on 23–29 tasks (the
  full benchmark passes in at most two languages at any size, none at 90M,
  600M or 1B) and MMLU 1.90–5.23 (one task, never passed); thousands of kept
  items shrink the k-fold noise, while Global-MMLU's DA-size in figure 3 stays
  at 0.45–0.57.

**Follow-ups**

- Put the SNR gain against the DA-size gain per (task, proxy), to show
  directly that the two move independently on this selection.

GitHub: [above_chance_items_snr_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr_by_benchmark.png) · [above_chance_items_snr_by_benchmark.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr_by_benchmark.csv)

### Scaling predictability

The results block's last table: on the deep data-A cells the median R² of
the log-N fit rises from 0.88 (1,185 fits) to 0.93 for *gate, then items*
(+0.033 on 1,184 paired fits, read on the same rungs as the full benchmark),
and *items, then gate* fits 2,082 (task, L) series at a median of 0.87.
Easier items track size more smoothly once the gate is fixed; admitting the
tasks the recomputed gate adds brings the median back to about the full
benchmark's.

### The key figure on one task set per line

Each line of the key figure takes the median over the tasks with a value at that size, and the gate keeps more tasks at larger sizes (308 → 504 on the full benchmark), so a line can move because its tasks changed (rule 13). The twin keeps, per ordering, the tasks with a value at every size (solid), the committed line dashed behind. It is rebuilt from the per-task CSVs, and its moving reading reproduces `above_chance_items_snr.csv` exactly.

![SNR panels on one task set per line](pretraining/predictivity/above_chance_items_snr_fixed_tasks_paper.png)

Pool `predictivity`, seed 1904, finals; ρ against DA-size multi-axis (each ordering's own truth). GitHub: [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr_fixed_tasks_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr_fixed_tasks_paper.csv)

Key findings:

- **Part of the items-then-gate SNR rise is its population.** On its 616 tasks at every size the median SNR reads 0.37 → 0.50 from 90M to 1.7B, against 0.39 → 0.55 on 676–759 tasks; the full and gate-then-items lines move by 0.02 at most.
- **The paired gain still grows with size, less steeply.** Gate then items reads x1.30 → x1.51 on its 287 tasks, against x1.27 → x1.50.
- **ρ(SNR, DA-size) stays positive and a little higher.** It reads 0.29–0.49 on fixed tasks, against 0.26–0.47; the full benchmark gains most (up to +0.08 at 1B).

Follow-ups:

- **The fixed twin against the full-task truth.** The SNR table now carries the full-truth ρ (`rho_snr_da_size_full_truth`); the fixed twin does not read it yet, so the dashed ρ lines have no fixed version.

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
