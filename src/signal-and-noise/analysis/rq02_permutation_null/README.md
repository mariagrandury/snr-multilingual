# Decision accuracy against its own no-signal null

**Question.** How far above chance does a proxy rank the design decisions, once chance is measured cell by cell rather than read off a flat 0.5?

The [decision-accuracy figures](../rq02_decision_accuracy/README.md) read DA-size against 0.5. That line is not what a proxy without signal scores: a pair the proxy ties is a miss unless the reference ties it too, and a cell with few families can reach a high DA by luck. This analysis gives every cell its own null.

## Setup

- **Snapshot.** Ladder report of **2026-10-08 12:06** and the outputs of the 2026-10-08 refresh (detrended checkpoint noise), pool `predictivity` (seed 1904, every ladder and data build), proxies 90M–1B against the 1.7B final.
- **Decision accuracy.** DA-size, no reliability filter, multi-axis and mono-axis pairs (rule 15), the pipeline's sign rule with ties (`compute_da.pair_agree`). Gate `predictivity` at the proxy and at 1.7B (rule 1); a cell needs 3 comparable pairs (rule 5).
- **Null.** Per (task, proxy size, pair set): 1000 shuffles of the proxy's final scores across the cell's families, the 1.7B ranking unchanged. A cell's p is the share of shuffles that match at least as many pairs; q is its Benjamini-Hochberg correction over the gated cells of one pair set.
- **Pooled lines.** Matching pairs over comparable pairs, as [`scale_convergence.py`](../rq02_decision_accuracy/README.md) pools; the band is the 2.5–97.5 % range of the summed shuffles.
- **Populations.** Those of the rq02 figures: "Benchmarks" holds every benchmark variant, the bBPB twins included, and "BPB" the 50 per-language BPB tasks. The twins (816 tasks) now exist at every proxy size, so the benchmark task count runs from 1118 at 90M to 1286 at 1B.

## The no-signal null

![DA-size against its no-signal null, multi-axis](pretraining/predictivity/permutation_null_da_size_paper.png)

DA-size (no filter, multi-axis pairs, gate `predictivity`) against the null of each cell. [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_permutation_null/pretraining/predictivity/permutation_null_da_size_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_permutation_null/pretraining/predictivity/permutation_null_da_size_paper.csv)

Key findings:

- **The benchmark null sits below 0.5.** Pooled over the benchmark tasks, a shuffled proxy scores 0.49, because proxy ties count as misses; the observed 0.58–0.59 is 0.08–0.10 above it, not 0.08–0.09.
- **Few benchmark cells beat their own null.** Only 8–10 % of the multi-axis benchmark cells (1–2 % of the mono-axis ones) are above their null at q < 0.05; the pooled lines are above the null because many weak cells add up, not because single tasks rank the decisions.
- **BPB is far above its null.** Per-language BPB reaches 0.79–0.95 against a null of 0.50, and 24–66 % of its multi-axis cells are significant (0–44 % mono-axis, where a cell has fewer pairs).
- **The bBPB twins lift the benchmark line.** Without the twins, benchmarks read 0.53–0.56 multi-axis (null 0.47); the twins alone read 0.59–0.61 against a null of 0.50.

Follow-ups:

- **A null per benchmark family.** It would show which families carry the few significant cells.
- **The null for DA-ckpt.** The checkpoint axis already has the seed null; a permutation null would put both axes on one scale.

<!-- BEGIN auto:permutation-null (permutation_null.py --pool predictivity) -->
## Results

Pool `predictivity`; per cell (task, proxy size, pair set) 1000 shuffles of the proxy's final scores across the cell's families, scored by the pipeline's sign rule against the unchanged 1.7B ranking. Cells are gated at the proxy and at 1.7B (rule 1) and need 3 pairs (rule 5). Regenerate with `python analysis/rq02_permutation_null/permutation_null.py --pool predictivity`.

![DA-size against its null](pretraining/predictivity/permutation_null_da_size_both_axes.png)

Per cell: observed pooled DA-size / pooled null mean [2.5 %, 97.5 %] / share of cells above their own null at BH q < 0.05 (tasks):

| pairs | population | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| multi-axis | Benchmarks | 0.591 / 0.492 [0.486, 0.499] / 9% (1118) | 0.576 / 0.492 [0.485, 0.497] / 9% (1158) | 0.579 / 0.490 [0.484, 0.495] / 8% (1193) | 0.578 / 0.488 [0.482, 0.494] / 9% (1245) | 0.584 / 0.488 [0.483, 0.494] / 10% (1286) |
| multi-axis | BPB | 0.945 / 0.500 [0.463, 0.536] / 66% (50) | 0.944 / 0.500 [0.463, 0.537] / 64% (50) | 0.885 / 0.499 [0.461, 0.538] / 66% (50) | 0.785 / 0.500 [0.465, 0.536] / 24% (50) | 0.910 / 0.499 [0.462, 0.538] / 60% (50) |
| mono-axis | Benchmarks | 0.572 / 0.492 [0.485, 0.499] / 1% (1118) | 0.545 / 0.492 [0.484, 0.499] / 2% (1158) | 0.556 / 0.490 [0.482, 0.497] / 1% (1193) | 0.541 / 0.489 [0.481, 0.496] / 1% (1245) | 0.548 / 0.489 [0.481, 0.495] / 1% (1286) |
| mono-axis | BPB | 0.924 / 0.500 [0.464, 0.537] / 38% (50) | 0.931 / 0.500 [0.464, 0.540] / 44% (50) | 0.844 / 0.500 [0.464, 0.536] / 26% (50) | 0.661 / 0.500 [0.463, 0.537] / 0% (50) | 0.868 / 0.501 [0.465, 0.537] / 20% (50) |

Tables: `permutation_null_da_size_per_task_both_axes.csv` (every cell's DA, null mean, null p95, p and q) and `permutation_null_da_size_both_axes.csv` (the pooled lines).
<!-- END auto:permutation-null -->
