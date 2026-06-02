# Accuracy-vs-FLOPs curves (scaling view)

> How does benchmark accuracy move with compute across the three data mixtures
> and across model scale, and which benchmarks separate the mixtures most?

Outputs under `pretraining/<pool>/{per_benchmark,per_language}/`. Each panel is
accuracy vs FLOPs (log-x); the custom models are drawn as per-mixture curves
(seed 1904), and for `custom_swissai_hf` the external pretraining models
(a06, distillation, swiss-ai/HF base) are **overlaid as final-checkpoint
markers out to 70B** to extend the compute axis past the custom 1B ceiling.

## How the plots are selected (fast by construction)

- Tasks are **parent-aggregated** (mmlu / agieval / arabic_leaderboard subjects
  collapse into the parent; languages stay distinct) → 950 → 118 tasks.
- Every task's **Signal** = (max−min)/mean of the per-mixture final scores at
  1B is written to [`acc_vs_flops_signal.csv`](pretraining/custom_swissai_hf/acc_vs_flops_signal.csv);
  only the **top-3 families by Signal** get curve grids (307 → 3 per-benchmark
  PNGs).

## TL;DR (paper takeaways)

1. **The benchmarks that separate data mixtures most are `belebele`,
   `agieval_sat`, and `arabic_leaderboard`** — top-3 by mixture-Signal at 1B.
2. **Mixture-Signal ≠ SNR.** These top-Signal families are exactly RQ2's
   *lowest-SNR* families (`belebele` is last by SNR). They swing a lot with the
   data mixture (high signal) **but are also high-noise**, so signal-to-noise
   stays low. The lesson for the paper: *raw mixture sensitivity is not
   reliability* — a benchmark must be divided by its noise (SNR), which is why
   RQ1/RQ2 rank `multiblimp`/`hellaswag` (low absolute swing, very low noise)
   above `belebele`.
3. **Scaling overlay**: on `custom_swissai_hf` the external models extend the
   curves to 8B–70B, but cross-size *decision accuracy* above 1B stays
   family-coverage-limited (see RQ1's scaling-DA note) — the overlay is for
   visual scaling, not a DA claim.

## Top benchmarks by mixture-Signal (pool `custom_swissai_hf`)

| task | family | lang | Signal |
|---|---|---|---:|
| **`agieval_sat_en`** | agieval_sat | en | **0.268** |
| `belebele_hin_Deva` | belebele | hi | 0.245 |
| `belebele_zho_Hans` | belebele | zh | 0.236 |
| `global_piqa_completions_arb_arab` | global_piqa | ar | 0.229 |
| `belebele_eng_Latn` | belebele | en | 0.222 |

![belebele accuracy vs FLOPs (per language)](pretraining/custom_swissai_hf/per_benchmark/belebele.png)

![English benchmarks accuracy vs FLOPs](pretraining/custom_swissai_hf/per_language/en.png)

## Files

- `pretraining/<pool>/acc_vs_flops_signal.csv` — per-task mixture-Signal (full
  ranking; all 118 parent tasks).
- `…/per_benchmark/<family>.png` — top-3 families, subplots per language,
  external scaling markers overlaid.
- `…/per_language/<lang>.png` — per language, subplots = top-3 families.
