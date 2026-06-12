# Top-K reliability agreement

Variant used: **Dispersion** (`dispersion`)
Apertus SNR column: `snr_dispersion_1B`  ·  AllenAI SNR column: `snr_dispersion_1B`
Shared-task universe: **7** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 7 total.

Other Apertus → AllenAI aliases that hit the shared set: `commonsense_qa → csqa`.

## Cross-corpus agreement over the shared tasks (the result)

Best variant `dispersion`, n = 7 shared tasks:

| metric | value |
|---|---:|
| **Pearson r** (log₁₀ SNR values) | **+0.751** |
| **Spearman ρ** (rank order) | **+0.786** |

> With only 7 shared tasks, **top-K set overlap is NOT a result** — any K ≥ 7 spans the whole universe, so Jaccard is trivially 1.0. Only K < 7 is reported below.

## Top-K agreement (non-trivial K only)

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 4 | 0.80 | 0.67 | arc_easy, hellaswag, mmlu, piqa |

## Full ranking per corpus (all shared tasks)

### Apertus

| task          |   snr |
|:--------------|------:|
| piqa          | 5.906 |
| arc_easy      | 5.242 |
| hellaswag     | 3.238 |
| arc_challenge | 3.112 |
| mmlu          | 0.633 |
| csqa          | 0.616 |
| openbookqa    | 0.486 |

### AllenAI

| task          |    snr |
|:--------------|-------:|
| arc_easy      | 43.499 |
| hellaswag     | 34.187 |
| piqa          | 24.594 |
| mmlu          | 18.497 |
| csqa          | 16.496 |
| arc_challenge | 15.74  |
| openbookqa    |  8.347 |
