# Top-K reliability agreement

Variant used: **Dispersion** (`dispersion`)
Apertus SNR column: `snr_dispersion_1B`  ·  AllenAI SNR column: `snr_dispersion_1B`
Shared-task universe: **7** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 7 total.

Other Apertus → AllenAI aliases that hit the shared set: `commonsense_qa → csqa`.

## Top-K agreement

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 4 | 0.80 | 0.67 | arc_easy, hellaswag, mmlu, piqa |
| 10 | 7 | 0.70 | 1.00 | arc_challenge, arc_easy, csqa, hellaswag, mmlu, openbookqa, piqa |
| 20 | 7 | 0.35 | 1.00 | arc_challenge, arc_easy, csqa, hellaswag, mmlu, openbookqa, piqa |

## Top-20 per corpus

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
