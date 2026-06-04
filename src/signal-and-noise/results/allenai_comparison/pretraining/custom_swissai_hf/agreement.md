# Top-K reliability agreement

Variant used: **RMS Deviation** (`rms_deviation`)
Apertus SNR column: `snr_rms_deviation_1B`  ·  AllenAI SNR column: `snr_rms_deviation_1B`
Shared-task universe: **7** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 7 total.

Other Apertus → AllenAI aliases that hit the shared set: `commonsense_qa → csqa`.

## Cross-corpus agreement over the shared tasks (the result)

Best variant `rms_deviation`, n = 7 shared tasks:

| metric | value |
|---|---:|
| **Pearson r** (log₁₀ SNR values) | **+0.837** |
| **Spearman ρ** (rank order) | **+0.643** |

> With only 7 shared tasks, **top-K set overlap is NOT a result** — any K ≥ 7 spans the whole universe, so Jaccard is trivially 1.0. Only K < 7 is reported below.

## Top-K agreement (non-trivial K only)

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 4 | 0.80 | 0.67 | arc_easy, hellaswag, mmlu, piqa |

## Full ranking per corpus (all shared tasks)

### Apertus

| task          |   snr |
|:--------------|------:|
| piqa          | 2.175 |
| arc_easy      | 1.968 |
| hellaswag     | 1.91  |
| csqa          | 1.508 |
| mmlu          | 1.444 |
| arc_challenge | 1.304 |
| openbookqa    | 0.699 |

### AllenAI

| task          |    snr |
|:--------------|-------:|
| arc_easy      | 11.836 |
| hellaswag     |  7.248 |
| mmlu          |  5.903 |
| piqa          |  5.369 |
| arc_challenge |  4.787 |
| csqa          |  4.107 |
| openbookqa    |  2.077 |
