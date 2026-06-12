# Top-K reliability agreement

Variant used: **Discrepancy** (`discrepancy`)
Apertus SNR column: `snr_discrepancy_1B`  ·  AllenAI SNR column: `snr_discrepancy_1B`
Shared-task universe: **7** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 7 total.

Other Apertus → AllenAI aliases that hit the shared set: `commonsense_qa → csqa`.

## Top-K agreement

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 5 | 1.00 | 1.00 | arc_easy, csqa, hellaswag, mmlu, piqa |
| 10 | 7 | 0.70 | 1.00 | arc_challenge, arc_easy, csqa, hellaswag, mmlu, openbookqa, piqa |
| 20 | 7 | 0.35 | 1.00 | arc_challenge, arc_easy, csqa, hellaswag, mmlu, openbookqa, piqa |

## Top-20 per corpus

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
