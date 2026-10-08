# Proxy-only item selection

**Question.** Can a benchmark be shortened using the proxies alone, so that the shortened benchmark still ranks the design variants like the 1.7B reference does on the full task?

The two earlier item selections read the reference. [rq12](../rq12_above_chance_items/README.md) keeps the items the 1.7B runs answer above chance and scores DA against those same runs, which makes it circular: it is an upper bound. [rq08's solved items](../rq08_subset_selection/README.md#items-the-reference-solves) select on the 1.7B runs of half of the families. Here the selection reads only proxy runs, so it could be done before the reference is trained.

## Setup

- **Snapshot.** No numbers yet. The per-item store exists only on the cluster, so every table, figure and number here fills in after the first cluster run of `proxy_item_selection.py`. Its results block (below, once generated) states the store it read.
- **Pool.** `predictivity` (seed 1904, every ladder and data build), final checkpoints, parent tasks in trained languages only (rules 2 and 6, applied by the loader). Proxies 90M–1B, reference 1.7B (rules 9 and 10). Per-item outcomes come from the rq08 store (`build_per_item_store.py`), using the task's own metric (acc or acc_norm).
- **Split.** The families with a 1.7B final are split in two halves, alternating along L, arch, scheme and T. This is the stratified split of `reference_solved.py` and `per_item_ladder.heldout_da`. One half selects, the other is scored, then the halves swap and the two readings are averaged.
- **Discrimination.** On the selecting half, at 600M and 1B only (never the reference), an item's discrimination is the point-biserial correlation of its outcome with the run's total task score. Item and total are centred within each size first, so an item that only separates 600M from 1B does not count. An item with no variance gets NaN and is never kept; the items with a value are the candidates.
- **Kept items.** The top q of the candidates by discrimination, q ∈ {25 %, 50 %, 75 %}. A run's sub-benchmark score is its mean over the kept items.
- **Decision accuracy.** DA-size of each proxy size on the kept items against the 1.7B final of the **full** task, on the held-out half's pairs, with multi-axis and mono-axis pair sets (rule 15). There is no reliability filter. A half needs 3 held-out pairs (rule 5), and the per-task table carries the pair count either way. The gate is `predictivity` at the proxy and at 1.7B (rule 1): gated rows are kept and blanked.
- **Baselines on the same held-out pairs.** The full task. Random subsets: 20 draws of the kept count from the same candidates, averaged. rq12's reference-selected items: those whose mean over every 1.7B final is above the task's chance level. That baseline reads the reference and is labelled as the circular upper bound; it is empty where a task has no chance level or no such item.
- **Rule 11.** The selection reads only proxy runs of the other half, so its DA is a genuine held-out estimate, available before the reference is trained. The truth it is scored against is still the 1.7B full task, so the analysis needs the reference to *evaluate* the method, not to *apply* it.
- **Populations move (rule 13).** The gate keeps different tasks at different sizes, so the task count behind a mean changes along a row. Each figure and table carries it.

## Held-out decision accuracy of the proxy-selected items

![Proxy-only item selection, DA-size, both pair sets](pretraining/predictivity/proxy_item_selection_da_size_both_axes.png)

DA-size (no filter, multi-axis and mono-axis pairs, gate `predictivity` at the proxy and at 1.7B), mean over the tasks, per q and proxy size; the full task, random subsets of the same count, and rq12's reference-selected items (dotted, upper bound). [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_both_axes.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_both_axes.csv)

Key findings:

- Pending the first cluster run; the numbers will be read from the CSV, not written ahead of it.

Follow-ups:

- **The selection at the proxy that is scored.** Read the discrimination at each proxy size separately, so 90M–350M are not judged by items chosen at 600M and 1B. This shows whether the small proxies need their own items.
- **A split that keeps mono-axis pairs together.** The alternating split tends to put the two members of a mono-axis pair (deep vs shallow at one L, A vs AT3) in different halves, so few mono-axis pairs remain in either half. A split by L would keep them, at the cost of stratification.

## For the paper

![Proxy-only item selection, paper figure](pretraining/predictivity/proxy_item_selection_da_size_multi_axes_paper.png)

Multi-axis pairs only, gate `predictivity`: on the left, DA-size against the 1.7B full task for the full task, random items and items chosen on the proxies at q = 50 %; on the right, the gain of the proxy-selected items over random items for each q. [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_multi_axes_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_multi_axes_paper.csv)

Key findings:

- Pending the first cluster run.

Follow-ups:

- **Per benchmark family.** Show the gain over random per family, to see which benchmarks can be shortened and which cannot.
- **SNR of the short benchmark.** Compute rq12's final-checkpoint SNR on the kept items, to test whether proxy-chosen items also lower the k-fold noise.
- **DA-ckpt.** Add it once the store holds every checkpoint, since the selection then has to hold along a run as well as across sizes.
