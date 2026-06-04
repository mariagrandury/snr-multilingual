# Top-K reliability agreement

Variant used: **Star Discrepancy (Shift+Scale)** (`star_discrepancy_shifted`)
Apertus SNR column: `snr_star_discrepancy_shifted_1B`  ·  AllenAI SNR column: `snr_star_discrepancy_shifted_1B`
Shared-task universe: **7** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 7 total.

Other Apertus → AllenAI aliases that hit the shared set: `commonsense_qa → csqa`.

## Cross-corpus agreement over the shared tasks (the result)

Best variant `star_discrepancy_shifted`, n = 7 shared tasks:

| metric | value |
|---|---:|
| **Pearson r** (log₁₀ SNR values) | **+0.924** |
| **Spearman ρ** (rank order) | **+0.929** |

> With only 7 shared tasks, **top-K set overlap is NOT a result** — any K ≥ 7 spans the whole universe, so Jaccard is trivially 1.0. Only K < 7 is reported below.

## Top-K agreement (non-trivial K only)

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 5 | 1.00 | 1.00 | arc_easy, csqa, hellaswag, mmlu, piqa |

## Full ranking per corpus (all shared tasks)

### Apertus

| task          |    snr |
|:--------------|-------:|
| piqa          | 10.019 |
| mmlu          |  8.365 |
| hellaswag     |  8.095 |
| csqa          |  7.32  |
| arc_easy      |  6.377 |
| arc_challenge |  6.279 |
| openbookqa    |  4.479 |

### AllenAI

| task          |    snr |
|:--------------|-------:|
| piqa          | 89.572 |
| hellaswag     | 51.433 |
| mmlu          | 22.14  |
| arc_easy      | 12.259 |
| csqa          | 11.346 |
| arc_challenge |  8.428 |
| openbookqa    |  3.902 |
