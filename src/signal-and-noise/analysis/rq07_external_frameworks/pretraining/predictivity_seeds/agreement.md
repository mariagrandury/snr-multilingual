# Top-K reliability agreement

Variant used: **Distance Standard Deviation** (`dist_std`)
Apertus SNR column: `snr_dist_std_1B`  ·  AllenAI SNR column: `snr_dist_std_1B`
Shared-task universe: **6** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 6 total.

Other Apertus → AllenAI aliases that hit the shared set: `commonsense_qa → csqa`.

## Cross-corpus agreement over the shared tasks (the result)

Best variant `dist_std`, n = 3 shared tasks:

| metric | value |
|---|---:|
| **Pearson r** (log₁₀ SNR values) | **+0.988** |
| **Spearman ρ** (rank order) | **+1.000** |

> With only 6 shared tasks, **top-K set overlap is NOT a result** — any K ≥ 6 spans the whole universe, so Jaccard is trivially 1.0. Only K < 6 is reported below.

## Top-K agreement (non-trivial K only)

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 3 | 0.60 | 0.60 | arc_challenge, arc_easy, hellaswag |

## Full ranking per corpus (all shared tasks)

### Apertus

| task          |   snr |
|:--------------|------:|
| arc_easy      | 1.471 |
| hellaswag     | 1.128 |
| arc_challenge | 0.802 |

### AllenAI

| task          |   snr |
|:--------------|------:|
| arc_easy      | 9.775 |
| hellaswag     | 7.441 |
| mmlu          | 4.841 |
| arc_challenge | 3.945 |
| csqa          | 3.438 |
| openbookqa    | 1.705 |
