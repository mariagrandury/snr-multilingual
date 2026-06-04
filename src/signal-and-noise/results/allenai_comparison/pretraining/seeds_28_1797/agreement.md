# Top-K reliability agreement

Variant used: **Discrepancy** (`discrepancy`)
Apertus SNR column: `snr_discrepancy_1B`  ·  AllenAI SNR column: `snr_discrepancy_1B`
Shared-task universe: **7** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 7 total.

Other Apertus → AllenAI aliases that hit the shared set: `commonsense_qa → csqa`.

## Cross-corpus agreement over the shared tasks (the result)

Best variant `discrepancy`, n = 7 shared tasks:

| metric | value |
|---|---:|
| **Pearson r** (log₁₀ SNR values) | **+0.836** |
| **Spearman ρ** (rank order) | **+0.643** |

> With only 7 shared tasks, **top-K set overlap is NOT a result** — any K ≥ 7 spans the whole universe, so Jaccard is trivially 1.0. Only K < 7 is reported below.

## Top-K agreement (non-trivial K only)

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 5 | 1.00 | 1.00 | arc_easy, csqa, hellaswag, mmlu, piqa |

## Full ranking per corpus (all shared tasks)

### Apertus

| task          |    snr |
|:--------------|-------:|
| piqa          | 20.501 |
| arc_easy      |  8.963 |
| csqa          |  7.748 |
| hellaswag     |  7.5   |
| mmlu          |  5.937 |
| openbookqa    |  5.886 |
| arc_challenge |  4.305 |

### AllenAI

| task          |    snr |
|:--------------|-------:|
| piqa          | 47.692 |
| mmlu          | 19.878 |
| hellaswag     | 18.196 |
| arc_easy      | 16.954 |
| csqa          | 11.754 |
| arc_challenge |  6.795 |
| openbookqa    |  5.334 |
