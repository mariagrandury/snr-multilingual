# Top-K reliability agreement

Variant used: **Halfspace Depth** (`tukey`)
Apertus SNR column: `snr_tukey_1B`  ·  AllenAI SNR column: `snr_tukey_1B`
Shared-task universe: **7** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 7 total.

Other Apertus → AllenAI aliases that hit the shared set: `commonsense_qa → csqa`.

## Top-K agreement

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 4 | 0.80 | 0.80 | arc_challenge, arc_easy, hellaswag, mmlu |
| 10 | 4 | 0.40 | 0.57 | arc_challenge, arc_easy, hellaswag, mmlu |
| 20 | 4 | 0.20 | 0.57 | arc_challenge, arc_easy, hellaswag, mmlu |

## Top-20 per corpus

### Apertus

| task          |    snr |
|:--------------|-------:|
| mmlu          | 21.252 |
| hellaswag     | 12.071 |
| arc_challenge | 10.394 |
| arc_easy      |  4.777 |

### AllenAI

| task          |     snr |
|:--------------|--------:|
| mmlu          | 105.617 |
| piqa          |  39.104 |
| hellaswag     |  37.98  |
| arc_challenge |  36.058 |
| arc_easy      |  22.826 |
| csqa          |  21.233 |
| openbookqa    |  20.335 |
