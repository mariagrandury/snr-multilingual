# Per-sample (Option D) — highlights from the committed intermediates

_320 (language, benchmark, size) cells across 80 benchmarks, reusing `variance_prefilter/` outputs (no raw samples)._

> **Scope.** Methods **B** and **C** need the per-checkpoint acc matrix (cluster-only) and are not reconstructable here; **A** is provably identical to **D**. Per-sample SNR is on binary item accuracy — not comparable to subtask-level (Case 1–3) SNR.

## Headline

- A per-sample subset beats the full set in **100%** of cells; median gain **+0.641**, max **+2.679**.
- Only a median of **40%** of items carry any cross-mix signal (the rest are dead and dropped).
- The winning subset is tiny — a median of **2%** of all items.
- **Neither the subset nor the ranking transfers across scale.** Median best-subset Jaccard across sizes is **0.03** and median per-sample SNR-rank Spearman is **0.05** — both ≈0. Per-sample SNR is estimated from only ~5 ckpts × 3 mixes per item, so this near-zero transfer likely reflects estimation noise as much as true scale-dependence, and argues for the subtask-level (Case 1–3) approach over per-item selection.

## Distribution by size

| size   |   n_cells |   median_informative_pct |   median_subset_pct |   median_gain |   max_gain |   pct_gain_pos |
|:-------|----------:|-------------------------:|--------------------:|--------------:|-----------:|---------------:|
| 175M   |        80 |                     38.1 |                 2.9 |         0.571 |      2.28  |            100 |
| 350M   |        80 |                     38.5 |                 2.8 |         0.637 |      2.246 |            100 |
| 600M   |        80 |                     38.8 |                 2.9 |         0.676 |      1.809 |            100 |
| 1B     |        80 |                     43.9 |                 1.4 |         0.667 |      2.679 |            100 |
| ALL    |       320 |                     39.6 |                 2.5 |         0.641 |      2.679 |            100 |

## Biggest gains (top 10 cells)

| language   | task                             | size   |   n_total |   n_candidates |   best_n |   full_set_snr |   best_snr |   snr_gain |
|:-----------|:---------------------------------|:-------|----------:|---------------:|---------:|---------------:|-----------:|-----------:|
| sw         | xcopa_sw                         | 1B     |       500 |            193 |       12 |           0.78 |       3.46 |       2.68 |
| eu         | paws_eu                          | 175M   |      1994 |            567 |        3 |           0.63 |       2.91 |       2.28 |
| es         | paws_es                          | 1B     |      2000 |           1497 |       18 |           0.63 |       2.89 |       2.27 |
| th         | belebele_tha_Thai                | 350M   |       900 |            596 |        6 |           0.2  |       2.45 |       2.25 |
| eu         | paws_eu                          | 1B     |      1994 |           1488 |       97 |           0.91 |       3.14 |       2.23 |
| en         | xwinograd_en                     | 1B     |      2325 |           1087 |      105 |           0.69 |       2.92 |       2.22 |
| sw         | xnli_sw                          | 350M   |      2490 |           1013 |       26 |           0.42 |       2.45 |       2.03 |
| ru         | belebele_rus_Cyrl                | 350M   |       900 |            656 |       31 |           0.49 |       2.45 |       1.96 |
| vi         | global_piqa_completions_vie_latn | 350M   |       100 |             38 |        3 |           0.51 |       2.45 |       1.94 |
| es         | xnli_es                          | 1B     |      2490 |           1133 |       25 |           0.51 |       2.43 |       1.92 |

## Cross-size stability of the selected subset (Jaccard of best_subset doc-ids)

| pair      |   median_jaccard |
|:----------|-----------------:|
| 175M|350M |            0.029 |
| 175M|600M |            0.026 |
| 175M|1B   |            0.012 |
| 350M|600M |            0.029 |
| 350M|1B   |            0.026 |
| 600M|1B   |            0.024 |

Low Jaccard = the items that best separate data mixtures differ by model size — a subset tuned at a small size won't transfer verbatim to a larger one.

## Cross-size stability of per-sample SNR ranking (Spearman of snr_<size>)

| pair      |   median_spearman |
|:----------|------------------:|
| 175M|350M |             0.038 |
| 175M|600M |             0.029 |
| 175M|1B   |             0.023 |
| 350M|600M |             0.054 |
| 350M|1B   |             0.06  |
| 600M|1B   |             0.098 |

All ≈0: an item's SNR ranking does not survive a change in model size. With only ~5 ckpts × 3 mixes behind each item's SNR, the per-sample estimate is itself very noisy, so this is a lower bound on true scale-dependence rather than proof of it — but either way per-item subsets don't transfer across scale.

## Files

- `size_distribution.csv` — the per-size table above.
- `cross_size_subset_jaccard.csv` — per (task, size-pair) subset overlap.
- `cross_size_snr_spearman.csv` — per (task, size-pair) SNR-rank correlation.
