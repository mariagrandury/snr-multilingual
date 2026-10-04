# Top-K reliability agreement

Variant used: **Average Absolute Deviation** (`aad`)
Apertus SNR column: `snr_aad_1B`  ·  AllenAI SNR column: `snr_aad_1B`
Shared-task universe: **6** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 6 total.

Other Apertus → AllenAI aliases that hit the shared set: `commonsense_qa → csqa`.

## Cross-corpus agreement over the shared tasks (the result)

Best variant `aad`, n = 3 shared tasks:

| metric | value |
|---|---:|
| **Pearson r** (log₁₀ SNR values) | **+0.947** |
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
| arc_easy      | 0.981 |
| hellaswag     | 0.786 |
| arc_challenge | 0.618 |

### AllenAI

| task          |   snr |
|:--------------|------:|
| arc_easy      | 9.76  |
| hellaswag     | 5.039 |
| mmlu          | 5.011 |
| arc_challenge | 4.157 |
| csqa          | 3.454 |
| openbookqa    | 1.691 |
