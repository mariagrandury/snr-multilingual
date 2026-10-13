# Top-K reliability agreement

Variant used: **Rel. Std. Dev.** (`rel_std`)
Apertus SNR column: `snr_rel_std_1B`  ·  AllenAI SNR column: `snr_rel_std_1B`
Shared-task universe: **9** tasks.

## Like for like

Both sides read the same noise window (the k/20 points in the last 20 % of each run, 5 points at 1B), the same checkpoint noise (the residual SD around a line through it, RULES.md rule 4) and the same metric (our `acc_norm` is DataDecide's `acc_per_char`, our `acc` its `acc_raw`; `build_allenai_variants.py`). MMLU and CSQA are our cloze twins `rf_mmlu` / `rf_commonsense_qa` against DataDecide's RC format; our letter-format originals are not shared. The signal populations still differ: 25 pretraining corpora against our design variants.

Other Apertus → AllenAI aliases that hit the shared set: `arc_challenge → arc_challenge`, `arc_easy → arc_easy`, `boolq → boolq`, `hellaswag → hellaswag`, `openbookqa → openbookqa`, `piqa → piqa`, `rf_commonsense_qa → csqa`, `rf_mmlu → mmlu`, `winogrande → winogrande`.

## Cross-corpus agreement over the shared tasks (the result)

Best variant `rel_std`, n = 9 shared tasks:

| metric | value |
|---|---:|
| **Pearson r** (log₁₀ SNR values) | **+0.750** |
| **Spearman ρ** (rank order) | **+0.783** |

> With only 9 shared tasks, **top-K set overlap is NOT a result** — any K ≥ 9 spans the whole universe, so Jaccard is trivially 1.0. Only K < 9 is reported below.

## Top-K agreement (non-trivial K only)

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 4 | 0.80 | 0.67 | arc_challenge, arc_easy, hellaswag, mmlu |

## Full ranking per corpus (all shared tasks)

### Apertus

| task          |   snr |
|:--------------|------:|
| hellaswag     | 6.979 |
| mmlu          | 4.712 |
| arc_easy      | 3.915 |
| winogrande    | 3.285 |
| arc_challenge | 2.727 |
| csqa          | 2.423 |
| piqa          | 2.268 |
| openbookqa    | 1.734 |
| boolq         | 1.621 |

### AllenAI

| task          |    snr |
|:--------------|-------:|
| arc_easy      | 15.669 |
| hellaswag     | 15.343 |
| mmlu          | 13.296 |
| arc_challenge | 12.61  |
| csqa          |  9.293 |
| piqa          |  6.529 |
| openbookqa    |  4.094 |
| winogrande    |  3.997 |
| boolq         |  3.56  |
