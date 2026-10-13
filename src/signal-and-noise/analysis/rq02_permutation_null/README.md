# Decision accuracy against its own no-signal null

**Question.** How far above chance does a proxy rank the design decisions, once chance is measured cell by cell rather than read off a flat 0.5?

The [decision-accuracy figures](../rq02_decision_accuracy/README.md) read DA-size against 0.5. But a proxy without signal does not score 0.5. A pair the proxy ties counts as a miss unless the reference ties it too, and a cell with few families can reach a high DA by luck. This analysis therefore gives every cell its own null.

## Setup

**Setup.**
- **Measured.** We measure how far DA-size lies above the agreement that a proxy without signal reaches in the same cell.
- **Models and pairs.** We use the seed-1904 runs of every language setting, ladder and data build. The proxies (90M–1B) are compared against the final 1.7B checkpoint, on pairs that differ on any number of design axes.
- **Tasks.** The tasks are the benchmark tasks above chance at the proxy and at 1.7B, the bBPB twins included, and the per-language BPB of the 50 trained languages. A (task, proxy size) cell needs three comparable pairs.
- **Null.** We shuffle the proxy's final scores across the design variants 1000 times. Each shuffle is scored against the unchanged 1.7B ranking with the same sign rule, ties included.
- **How.** A cell's p-value is the share of shuffles that match at least as many pairs as the real proxy, corrected over the cells with the Benjamini-Hochberg procedure. Each pooled line divides the matching pairs by the comparable pairs, summed over cells, and its band is the 95 % range of the pooled shuffles.

- **Snapshot.** Ladder report of **2026-10-10 04:31** and the outputs of the 2026-10-10 refresh (detrended checkpoint noise, prose re-read on 2026-10-10), pool `predictivity` (seed 1904, every ladder and data build), proxies 90M–1B against the 1.7B final.
- **Decision accuracy.** DA-size, no reliability filter, multi-axis and mono-axis pairs (rule 15), the pipeline's sign rule with ties (`compute_da.pair_agree`). Gate `predictivity` at the proxy and at 1.7B (rule 1). A cell needs 3 comparable pairs (rule 5).
- **Null.** Per (task, proxy size, pair set): 1000 shuffles of the proxy's final scores across the cell's families, with the 1.7B ranking unchanged. A cell's p is the share of shuffles that match at least as many pairs. Its q is the Benjamini-Hochberg correction of p over the gated cells of one pair set.
- **Pooled lines.** Matching pairs over comparable pairs, pooled as [`scale_convergence.py`](../rq02_decision_accuracy/README.md) pools them. The band is the 2.5–97.5 % range of the summed shuffles.
- **Populations.** Those of the rq02 figures: "Benchmarks" holds every benchmark variant, the bBPB twins included, and "BPB" the 50 per-language BPB tasks. The twins (816 tasks) now exist at every proxy size, so the benchmark task count runs from 1175 at 90M to 1369 at 1B (multi-axis).

## The no-signal null

![DA-size against its no-signal null, multi-axis](pretraining/predictivity/permutation_null_da_size_paper.png)

DA-size (no filter, multi-axis pairs, gate `predictivity`) against the null of each cell. [PNG](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_permutation_null/pretraining/predictivity/permutation_null_da_size_paper.png) · [CSV](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq02_permutation_null/pretraining/predictivity/permutation_null_da_size_paper.csv)

Key findings:

- **The benchmark null sits below 0.5.** Pooled over the benchmark tasks, a shuffled proxy scores 0.49 because proxy ties count as misses, so the observed 0.57–0.59 lies 0.08–0.10 above its null rather than 0.07–0.09 above 0.5.
- **Few benchmark cells beat their own null.** Only 8–10 % of the multi-axis benchmark cells (0–2 % of the mono-axis ones) are above their null at q < 0.05, so the pooled lines clear the null because many weak cells add up, not because single tasks rank the decisions.
- **BPB is far above its null.** Per-language BPB reaches 0.78–0.94 against a null of 0.50, and 30–64 % of its multi-axis cells are significant (4–40 % mono-axis, where a cell has fewer pairs).
- **The bBPB twins lift the benchmark line.** Without the twins, benchmarks read 0.53–0.56 multi-axis (null 0.47). The twins alone read 0.59–0.61 against a null of 0.50.

Follow-ups:

- **A null per benchmark family.** It would show which families carry the few significant cells.
- **The null for DA-ckpt.** The checkpoint axis already has the seed null. A permutation null would put both axes on one scale.

<!-- BEGIN auto:permutation-null (permutation_null.py --pool predictivity) -->
## Results

Pool `predictivity`. For each cell (task, proxy size, pair set) we shuffle the proxy's final scores across the cell's families 1000 times and score each shuffle against the unchanged 1.7B ranking with the pipeline's sign rule. Cells are gated at the proxy and at 1.7B (rule 1) and need 3 pairs (rule 5). Regenerate with `python analysis/rq02_permutation_null/permutation_null.py --pool predictivity`.

![DA-size against its null](pretraining/predictivity/permutation_null_da_size_both_axes.png)

Each entry reads: observed pooled DA-size / pooled null mean [2.5 %, 97.5 %] / share of cells above their own null at BH q < 0.05 (number of tasks).

| pairs | population | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| multi-axis | Benchmarks | 0.584 / 0.492 [0.487, 0.497] / 11% (1274) | 0.571 / 0.491 [0.486, 0.497] / 10% (1316) | 0.584 / 0.490 [0.485, 0.496] / 11% (1351) | 0.580 / 0.489 [0.484, 0.494] / 12% (1407) | 0.582 / 0.488 [0.483, 0.493] / 12% (1461) |
| multi-axis | BPB | 0.926 / 0.500 [0.466, 0.532] / 66% (50) | 0.916 / 0.500 [0.464, 0.534] / 68% (50) | 0.848 / 0.500 [0.467, 0.533] / 62% (50) | 0.789 / 0.500 [0.470, 0.531] / 34% (50) | 0.899 / 0.500 [0.465, 0.533] / 64% (50) |
| mono-axis | Benchmarks | 0.568 / 0.492 [0.485, 0.498] / 0% (1274) | 0.538 / 0.491 [0.485, 0.498] / 0% (1316) | 0.557 / 0.490 [0.484, 0.497] / 0% (1351) | 0.541 / 0.489 [0.483, 0.496] / 0% (1407) | 0.547 / 0.488 [0.482, 0.494] / 0% (1461) |
| mono-axis | BPB | 0.894 / 0.500 [0.466, 0.534] / 0% (50) | 0.892 / 0.501 [0.469, 0.535] / 0% (50) | 0.827 / 0.499 [0.465, 0.534] / 0% (50) | 0.630 / 0.500 [0.467, 0.533] / 0% (50) | 0.843 / 0.500 [0.466, 0.533] / 0% (50) |

Tables: `permutation_null_da_size_per_task_both_axes.csv` (every cell's DA, null mean, null p95, p and q) and `permutation_null_da_size_both_axes.csv` (the pooled lines).
<!-- END auto:permutation-null -->
