# Proxy-only item selection

**Question.** Can a benchmark be shortened using the proxies alone, so that the shortened benchmark still ranks the design variants like the 1.7B reference does on the full task?

The two earlier item selections read the reference. [rq12](../rq12_above_chance_items/README.md) keeps the items the 1.7B runs answer above chance and scores DA against those same runs, which makes it circular: it is an upper bound. [rq08's solved items](../rq08_subset_selection/README.md#items-the-reference-solves) select on the 1.7B runs of half of the families. Here the selection reads only proxy runs, so it could be done before the reference is trained.

## Setup

- **Snapshot.** Ladder report of **2026-10-08 12:06** and the outputs of the 2026-10-08 refresh (detrended checkpoint noise, acc_norm on the cloze-format originals), the first run of `proxy_item_selection.py` with the per-item store present. The results block (below) states the store it read, which still lacks 9 of the pool's models (the eight Muon cells at 90M–350M and the 1.7B L8 SwiGLU run), so those are left out.
- **Pool.** `predictivity` (seed 1904, every ladder and data build), final checkpoints, parent tasks in trained languages only (rules 2 and 6, applied by the loader). Proxies 90M–1B, reference 1.7B (rules 9 and 10). Per-item outcomes come from the rq08 store (`build_per_item_store.py`), using the task's own metric (acc or acc_norm).
- **Split.** The families with a 1.7B final are split in two halves, alternating along L, arch, scheme and T. This is the stratified split of `reference_solved.py` and `per_item_ladder.heldout_da`. One half selects, the other is scored, then the halves swap and the two readings are averaged.
- **Discrimination.** On the selecting half, at 600M and 1B only (never the reference), an item's discrimination is the point-biserial correlation of its outcome with the run's total task score. Item and total are centred within each size first, so an item that only separates 600M from 1B does not count. An item with no variance gets NaN and is never kept; the items with a value are the candidates.
- **Kept items.** The top q of the candidates by discrimination, q ∈ {25 %, 50 %, 75 %}, and a run's sub-benchmark score is its mean over the kept items. Only 37–38 % of a task's items are candidates (median over the tasks, per-task CSV), so q = 25 / 50 / 75 % keeps about 9 / 18 / 26–27 % of the items (`kept_share`).
- **Decision accuracy.** DA-size of each proxy size on the kept items against the 1.7B final of the **full** task, on the held-out half's pairs, with multi-axis and mono-axis pair sets (rule 15). There is no reliability filter. A half needs 3 held-out pairs (rule 5), and the per-task table carries the pair count either way. The gate is `predictivity` at the proxy and at 1.7B (rule 1): gated rows are kept and blanked.
- **Baselines on the same held-out pairs.** The full task. Random subsets: 20 draws of the kept count from the same candidates, averaged. rq12's reference-selected items: those whose mean over every 1.7B final is above the task's chance level. That baseline reads the reference and is labelled as the circular upper bound; it is empty where a task has no chance level or no such item.
- **Rule 11.** The selection reads only proxy runs of the other half, so its DA is a genuine held-out estimate, available before the reference is trained. The truth it is scored against is still the 1.7B full task, so the analysis needs the reference to *evaluate* the method, not to *apply* it.
- **Populations move (rule 13).** The gate keeps different tasks (benchmark × language) at different sizes: 238 at 90M up to 375 at 1B on multi-axis pairs, one fewer on mono-axis pairs, and 5 fewer for rq12's baseline. A held-out half has a median of 43 multi-axis and 15 mono-axis pairs (`n_pairs_median`).

## Held-out decision accuracy of the proxy-selected items

![Proxy-only item selection, DA-size, both pair sets](pretraining/predictivity/proxy_item_selection_da_size_both_axes.png)

DA-size (no filter, multi-axis and mono-axis pairs, gate `predictivity` at the proxy and at 1.7B), mean over the tasks, per q and proxy size; the full task, random subsets of the same count, and rq12's reference-selected items (dotted, upper bound). [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_both_axes.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_both_axes.csv)

Key findings:

- **Proxy-chosen items never beat the full task.** On multi-axis pairs the kept items read 0.470–0.559 DA-size against 0.528–0.571 for the full task, below it at every q, proxy size and pair set (by 0.005 at 600M with q = 75 % up to 0.058 at 90M with q = 25 %).
- **They beat random items of the same count, but only just.** The multi-axis gain over random is positive in 14 of the 15 (q, size) cells and at most +0.017, the largest at 1B for every q (+0.017 / +0.016 / +0.008 at q = 25 / 50 / 75 %). Per task, the kept items beat random on 55–60 % of the multi-axis tasks at 600M and 1B, the sizes they were chosen at, against 47–51 % at 90M (per-task CSV).
- **Mono-axis pairs leave nothing to select for.** The full task itself reads 0.491–0.526 on mono-axis pairs, and the gain over random swings between −0.015 and +0.018 with no pattern in q or size, on a median of 15 held-out pairs.
- **The circular upper bound stays out of reach.** rq12's reference-selected items read 0.554–0.574 on multi-axis pairs, above the full task at every size (by 0.004 at 1B up to 0.028 at 90M). The proxy selection closes none of that gap, since it never sees which items the 1.7B runs solve.

Follow-ups:

- **The selection at the proxy that is scored.** Read the discrimination at each proxy size separately, so 90M–350M are not judged by items chosen at 600M and 1B. This shows whether the small proxies need their own items.
- **A split that keeps mono-axis pairs together.** The alternating split tends to put the two members of a mono-axis pair (deep vs shallow at one L, A vs AT3) in different halves, so few mono-axis pairs remain in either half. A split by L would keep them, at the cost of stratification.
- **The store rebuilt for the whole pool.** The 9 models the store lacks (Muon cells, 1.7B L8 SwiGLU) are out of both halves, so the Muon families do not enter the selection or the scoring yet.

## For the paper

![Proxy-only item selection, paper figure](pretraining/predictivity/proxy_item_selection_da_size_multi_axes_paper.png)

Multi-axis pairs only, gate `predictivity`: on the left, DA-size against the 1.7B full task for the full task, random items and items chosen on the proxies at q = 50 %; on the right, the gain of the proxy-selected items over random items for each q. [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_multi_axes_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_multi_axes_paper.csv)

Key findings:

- **Keeping the top half of the candidates helps little over random items.** At q = 50 % (about 18 % of a task's items) the proxy-chosen items read 0.490 at 90M to 0.554 at 1B against 0.484 to 0.538 for random ones, a gain of −0.002 (175M) to +0.016 (1B).
- **Neither subset reaches the full task.** The full task reads 0.528 at 90M to 0.571 at 1B, 0.014–0.038 above the proxy-chosen half. The gap narrows with size, from 0.038 at 90M to 0.014–0.017 at 600M and 1B.
- **The gain over random is positive almost everywhere but small.** It is positive in 14 of the 15 (q, size) cells, never above +0.017, and largest at 1B for every q (+0.017 / +0.016 / +0.008 at q = 25 / 50 / 75 %), the size the items were chosen at.
- **Fewer items cost more than the selection wins back.** At q = 25 % the kept items sit 0.033–0.058 below the full task and at q = 75 % 0.005–0.030 below, while the gain over random stays within −0.002 to +0.017.

Follow-ups:

- **Per benchmark family.** Show the gain over random per family, to see which benchmarks can be shortened and which cannot.
- **SNR of the short benchmark.** Compute rq12's final-checkpoint SNR on the kept items, to test whether proxy-chosen items also lower the k-fold noise.
- **DA-ckpt.** Add it now that the store holds every checkpoint, since the selection has to hold along a run as well as across sizes.

<!-- BEGIN auto:proxy-item-selection (proxy_item_selection.py --pool predictivity) -->
## Results

Pool `predictivity`, per-item store `predictivity`. DA-size, finals, pool predictivity (seed 1904), held-out half's pairs (>= 3 per half), both halves averaged; items ranked by point-biserial discrimination (within size) on the other half's 600M and 1B runs, the top q kept; scored against the 1.7B final of the full task. Random = 20 draws of the kept count from the items with a discrimination; dotted = rq12's items above chance over every 1.7B final (reads the reference, upper bound). Gate predictivity at the proxy and the reference (rule 1); the task count moves across sizes (rule 13). Regenerate with `python analysis/rq14_proxy_item_selection/proxy_item_selection.py --pool predictivity --store-pool predictivity`. The store lacks 9 of the pool's models (lm-1.7B-L8-swiglu-seed1904, lm-175M-L15-b168-muon-seed1904, lm-175M-L30-b168-muon-seed1904, lm-175M-L8-b168-muon-seed1904, lm-350M-L15-muon-seed1904, lm-350M-L30-muon-seed1904, lm-90M-L15-b84-muon-seed1904, lm-90M-L30-b84-muon-seed1904, lm-90M-L8-b84-muon-seed1904); they are left out until the store is rebuilt for the pool.

**DA-size on the held-out half**, kept / full task / random / rq12's reference-selected items (tasks), mean over the tasks; the task count moves along a row (rule 13):

| pairs | kept | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| mono-axis | 25% | 0.441 / 0.502 / 0.450 / 0.513 (237) | 0.467 / 0.513 / 0.463 / 0.522 (271) | 0.456 / 0.491 / 0.451 / 0.494 (301) | 0.480 / 0.519 / 0.477 / 0.510 (342) | 0.500 / 0.526 / 0.481 / 0.531 (374) |
| mono-axis | 50% | 0.465 / 0.502 / 0.464 / 0.513 (237) | 0.466 / 0.513 / 0.481 / 0.522 (271) | 0.478 / 0.491 / 0.470 / 0.494 (301) | 0.492 / 0.519 / 0.494 / 0.510 (342) | 0.508 / 0.526 / 0.503 / 0.531 (374) |
| mono-axis | 75% | 0.470 / 0.502 / 0.465 / 0.513 (237) | 0.481 / 0.513 / 0.490 / 0.522 (271) | 0.480 / 0.491 / 0.476 / 0.494 (301) | 0.507 / 0.519 / 0.505 / 0.510 (342) | 0.518 / 0.526 / 0.512 / 0.531 (374) |
| multi-axis | 25% | 0.470 / 0.528 / 0.468 / 0.556 (238) | 0.497 / 0.539 / 0.483 / 0.566 (272) | 0.491 / 0.531 / 0.480 / 0.554 (302) | 0.514 / 0.547 / 0.502 / 0.558 (343) | 0.528 / 0.571 / 0.511 / 0.574 (375) |
| multi-axis | 50% | 0.490 / 0.528 / 0.484 / 0.556 (238) | 0.503 / 0.539 / 0.505 / 0.566 (272) | 0.512 / 0.531 / 0.501 / 0.554 (302) | 0.533 / 0.547 / 0.526 / 0.558 (343) | 0.554 / 0.571 / 0.538 / 0.574 (375) |
| multi-axis | 75% | 0.498 / 0.528 / 0.490 / 0.556 (238) | 0.516 / 0.539 / 0.513 / 0.566 (272) | 0.514 / 0.531 / 0.510 / 0.554 (302) | 0.542 / 0.547 / 0.536 / 0.558 (343) | 0.559 / 0.571 / 0.550 / 0.574 (375) |

**Gain over random items** (mean paired difference over the same tasks):

| pairs | kept | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| mono-axis | 25% | -0.009 | +0.005 | +0.005 | +0.003 | +0.018 |
| mono-axis | 50% | +0.001 | -0.015 | +0.008 | -0.002 | +0.005 |
| mono-axis | 75% | +0.006 | -0.009 | +0.004 | +0.002 | +0.006 |
| multi-axis | 25% | +0.002 | +0.013 | +0.011 | +0.012 | +0.017 |
| multi-axis | 50% | +0.006 | -0.002 | +0.011 | +0.007 | +0.016 |
| multi-axis | 75% | +0.008 | +0.003 | +0.005 | +0.006 | +0.008 |

![Proxy-only item selection](pretraining/predictivity/proxy_item_selection_da_size_both_axes.png)
<!-- END auto:proxy-item-selection -->
