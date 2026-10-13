# Proxy-only item selection

**Question.** Can we shorten a benchmark using only the proxy runs, so that the short version still ranks the design variants as the 1.7B reference does on the full task?

Both earlier item selections look at the reference. [rq12](../rq12_above_chance_items/README.md) keeps the items the 1.7B runs answer above chance and then scores DA against those same runs. That is circular, so its result is an upper bound. [rq08's solved items](../rq08_subset_selection/README.md#items-the-reference-solves) are chosen on the 1.7B runs of half of the families. Here the selection reads only proxy runs, so it could be made before the reference is trained.

## Setup

**Setup.**
- **Measured.** We shorten each benchmark using only the proxy runs and measure the held-out DA-size of the kept items against the 1.7B ranking on the full task.
- **Models and pairs.** We use the final checkpoints of the seed-1904 runs of every cell, ladder and data build that the per-item store holds (all but the twelve Muon runs at 90M–600M and the 1.7B L8 SwiGLU run). A pair may differ on any number of design axes, and a task must be above chance at the proxy and at 1.7B.
- **Selection.** We split the design families into two halves. On one half, we rank the items by their discrimination at 600M and 1B and keep the top quarter, half or three quarters of the candidates, the items whose outcome varies across those runs (about 36 % of a task's items). An item's discrimination is the correlation of its correctness with the run's task score, both centred within each size.
- **How.** On the other half's pairs, we score the DA-size of the proxies 90M–1B on the kept items against the 1.7B ranking on the full task. Then the halves swap and we average the two readings. The baseline is random subsets of the same size, drawn from the same items.
- **Leakage.** The selection never reads the 1.7B runs, so its DA is a held-out estimate that is available before the reference is trained. Only the task set, through the above-chance gate at 1.7B that every DA uses, depends on the reference.

- **Snapshot.** Ladder report of **2026-10-10 04:31** and the outputs of the 2026-10-10 refresh (detrended checkpoint noise, acc_norm on the cloze-format originals). The results block (below) names the per-item store it read. That store still lacks 13 of the pool's models (the twelve Muon cells at 90M–600M and the 1.7B L8 SwiGLU run), so they are left out.
- **Pool.** `predictivity` (seed 1904, every ladder and data build), final checkpoints, parent tasks in trained languages only (rules 2 and 6, applied by the loader). Proxies 90M–1B, reference 1.7B (rules 9 and 10). Per-item outcomes come from the rq08 store (`build_per_item_store.py`) and use the task's own metric (acc or acc_norm).
- **Split.** We split the families with a 1.7B final into two halves, alternating along L, arch, scheme and T. This is the stratified split of `reference_solved.py` and `per_item_ladder.heldout_da`. One half selects and the other is scored. Then the halves swap and we average the two readings.
- **Discrimination.** An item's discrimination is the point-biserial correlation of its outcome with the run's total task score, on the selecting half at 600M and 1B only (never the reference). Item and total are first centred within each size, so an item that only separates 600M from 1B does not count. An item with no variance gets NaN and is never kept. The items with a value are the candidates.
- **Kept items.** We keep the top q of the candidates by discrimination, q ∈ {25 %, 50 %, 75 %}. A run's score on the short benchmark is its mean over the kept items. Only 35–36 % of a task's items are candidates (median over the tasks, per-task CSV), so q = 25 / 50 / 75 % keeps about 9 / 18 / 26–27 % of the items (`kept_share`).
- **Decision accuracy.** DA-size of each proxy size on the kept items against the 1.7B final of the **full** task, on the held-out half's pairs, with multi-axis and mono-axis pair sets (rule 15). There is no reliability filter. A half needs 3 held-out pairs (rule 5), and the per-task table gives the pair count either way. The gate is `predictivity` at the proxy and at 1.7B (rule 1). Gated rows are kept and left blank.
- **Baselines on the same held-out pairs.** The first is the full task. The second is random subsets: 20 draws of the kept count from the same candidates, averaged. The third is rq12's reference-selected items, those whose mean over every 1.7B final is above the task's chance level. This last baseline reads the reference, so we label it the circular upper bound. It is empty where a task has no chance level or no such item.
- **Rule 11.** The selection reads only proxy runs of the other half, so its DA is a true held-out estimate, available before the reference is trained. It is still scored against the 1.7B full task. The analysis therefore needs the reference to *evaluate* the method, not to *apply* it.
- **Populations move (rule 13).** The gate keeps different tasks (benchmark × language) at different sizes: from 238 at 90M up to 375 at 1B on multi-axis pairs, one fewer on mono-axis pairs, and 5 fewer for rq12's baseline. A held-out half has a median of 43 multi-axis and 15 mono-axis pairs (`n_pairs_median`).

## Held-out decision accuracy of the proxy-selected items

![Proxy-only item selection, DA-size, both pair sets](pretraining/predictivity/proxy_item_selection_da_size_both_axes.png)

DA-size (no filter, multi-axis and mono-axis pairs, gate `predictivity` at the proxy and at 1.7B), mean over the tasks, per q and proxy size. The lines are the kept items, the full task, random subsets of the same count, and rq12's reference-selected items (dotted, the upper bound). [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_both_axes.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_both_axes.csv)

Key findings:

- **Proxy-chosen items never beat the full task.** On multi-axis pairs the kept items reach a DA-size of 0.470–0.559 and the full task 0.528–0.571. The kept items are below the full task at every q, proxy size and pair set. On multi-axis pairs the gap runs from 0.004 (600M, q = 75 %) to 0.058 (90M, q = 25 %), and on mono-axis pairs it reaches 0.061 (90M, q = 25 %).
- **They beat random items of the same count, but only just.** The multi-axis gain over random is positive in 14 of the 15 (q, size) cells and at most +0.017. It is largest at 1B for every q (+0.017 / +0.016 / +0.008 at q = 25 / 50 / 75 %). Per task, the kept items beat random on 55–60 % of the multi-axis tasks at 600M and 1B, the sizes they were chosen at, and on 47–51 % at 90M (per-task CSV).
- **Mono-axis pairs leave nothing to select for.** The full task itself is at 0.492–0.526 on mono-axis pairs. The gain over random swings between −0.015 and +0.018 with no pattern in q or size, on a median of 15 held-out pairs.
- **The circular upper bound stays out of reach.** rq12's reference-selected items reach 0.555–0.574 on multi-axis pairs, above the full task at every size (by 0.004 at 1B up to 0.028 at 90M). The proxy selection closes none of that gap, because it never sees which items the 1.7B runs solve.

Follow-ups:

- **The selection at the proxy that is scored.** Compute the discrimination at each proxy size separately, so that 90M–350M are not judged on items chosen at 600M and 1B. This shows whether the small proxies need their own items.
- **A split that keeps mono-axis pairs together.** The alternating split tends to put the two members of a mono-axis pair (deep vs shallow at one L, A vs AT3) in different halves, so few mono-axis pairs remain in either half. A split by L would keep them together, at the cost of stratification.
- **The store rebuilt for the whole pool.** The 13 models the store lacks (Muon cells, 1.7B L8 SwiGLU) are in neither half, so the Muon families do not enter the selection or the scoring yet.

## For the paper

![Proxy-only item selection, paper figure](pretraining/predictivity/proxy_item_selection_da_size_multi_axes_paper.png)

Multi-axis pairs only, gate `predictivity`. Left: DA-size against the 1.7B full task for the full task, for random items and for the items chosen on the proxies at q = 50 %. Right: the gain of the proxy-chosen items over random items, for each q. [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_multi_axes_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_multi_axes_paper.csv)

Key findings:

- **The top half of the candidates is barely better than random items.** At q = 50 % (about 18 % of a task's items), the proxy-chosen items reach 0.490 at 90M to 0.554 at 1B, and random items 0.484 to 0.538. The gain over random goes from −0.002 (175M) to +0.016 (1B).
- **Neither subset reaches the full task.** The full task reaches 0.528 at 90M to 0.571 at 1B, which is 0.013–0.038 above the proxy-chosen half. The gap narrows with size, from 0.038 at 90M to 0.013–0.017 at 600M and 1B.
- **The gain over random is positive almost everywhere, but small.** It is positive in 14 of the 15 (q, size) cells and never above +0.017. It is largest at 1B, one of the sizes the items were chosen at, for every q (+0.017 / +0.016 / +0.008 at q = 25 / 50 / 75 %).
- **Dropping items costs more than the selection wins back.** At q = 25 % the kept items sit 0.032–0.058 below the full task, and at q = 75 % they sit 0.004–0.030 below. The gain over random stays between −0.002 and +0.017.

Follow-ups:

- **Per benchmark family.** Show the gain over random per family, to see which benchmarks can be shortened and which cannot.
- **SNR of the short benchmark.** Compute rq12's final-checkpoint SNR on the kept items, to test whether proxy-chosen items also lower the k-fold noise.
- **DA-ckpt.** Add it now that the store holds every checkpoint, because the selection must hold along a run as well as across sizes.

<!-- BEGIN auto:proxy-item-selection (proxy_item_selection.py --pool predictivity) -->
## Results

Pool `predictivity`, per-item store `predictivity`. DA-size of the finals, pool predictivity (seed 1904), on the held-out half's pairs (>= 3 per half), the two halves averaged. The items are ranked by point-biserial discrimination (within size) on the other half's 600M and 1B runs, the top q are kept, and the kept items are scored against the 1.7B final of the full task. Random = 20 draws of the kept count from the items with a discrimination. Dotted = rq12's items above chance over every 1.7B final (they read the reference, so they are an upper bound). Gate predictivity at the proxy and the reference (rule 1). The task count moves across sizes (rule 13). Regenerate with `python analysis/rq14_proxy_item_selection/proxy_item_selection.py --pool predictivity --store-pool predictivity`.

**DA-size on the held-out half**, kept / full task / random / rq12's reference-selected items (tasks), mean over the tasks. The task count moves along a row (rule 13):

| pairs | kept | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| mono-axis | 25% | 0.460 / 0.502 / 0.463 / 0.512 (236) | 0.472 / 0.506 / 0.467 / 0.510 (266) | 0.476 / 0.491 / 0.461 / 0.499 (296) | 0.487 / 0.503 / 0.473 / 0.499 (334) | 0.502 / 0.520 / 0.485 / 0.519 (371) |
| mono-axis | 50% | 0.467 / 0.502 / 0.478 / 0.512 (236) | 0.494 / 0.506 / 0.479 / 0.510 (266) | 0.499 / 0.491 / 0.472 / 0.499 (296) | 0.501 / 0.503 / 0.491 / 0.499 (334) | 0.508 / 0.520 / 0.501 / 0.519 (371) |
| mono-axis | 75% | 0.476 / 0.502 / 0.481 / 0.512 (236) | 0.489 / 0.506 / 0.490 / 0.510 (266) | 0.492 / 0.491 / 0.479 / 0.499 (296) | 0.514 / 0.503 / 0.498 / 0.499 (334) | 0.522 / 0.520 / 0.513 / 0.519 (371) |
| multi-axis | 25% | 0.477 / 0.521 / 0.470 / 0.554 (296) | 0.492 / 0.530 / 0.482 / 0.560 (332) | 0.500 / 0.519 / 0.482 / 0.549 (362) | 0.525 / 0.542 / 0.504 / 0.558 (406) | 0.531 / 0.555 / 0.508 / 0.562 (452) |
| multi-axis | 50% | 0.493 / 0.521 / 0.485 / 0.554 (296) | 0.515 / 0.530 / 0.504 / 0.560 (332) | 0.522 / 0.519 / 0.497 / 0.549 (362) | 0.542 / 0.542 / 0.527 / 0.558 (406) | 0.547 / 0.555 / 0.529 / 0.562 (452) |
| multi-axis | 75% | 0.500 / 0.521 / 0.492 / 0.554 (296) | 0.522 / 0.530 / 0.514 / 0.560 (332) | 0.519 / 0.519 / 0.507 / 0.549 (362) | 0.550 / 0.542 / 0.537 / 0.558 (406) | 0.547 / 0.555 / 0.543 / 0.562 (452) |

**Gain over random items** (mean paired difference over the same tasks):

| pairs | kept | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| mono-axis | 25% | -0.004 | +0.005 | +0.014 | +0.014 | +0.017 |
| mono-axis | 50% | -0.011 | +0.015 | +0.026 | +0.010 | +0.008 |
| mono-axis | 75% | -0.005 | -0.001 | +0.013 | +0.016 | +0.010 |
| multi-axis | 25% | +0.007 | +0.010 | +0.018 | +0.020 | +0.024 |
| multi-axis | 50% | +0.008 | +0.011 | +0.025 | +0.016 | +0.017 |
| multi-axis | 75% | +0.008 | +0.008 | +0.013 | +0.014 | +0.005 |

![Proxy-only item selection](pretraining/predictivity/proxy_item_selection_da_size_both_axes.png)
<!-- END auto:proxy-item-selection -->
