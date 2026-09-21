# Top-K reliability agreement

Variant used: **Star Rel. Discrepancy** (`rel_star_discrepancy`)
Apertus SNR column: `snr_rel_star_discrepancy_1B`  ·  AllenAI SNR column: `snr_rel_star_discrepancy_1B`
Shared-task universe: **4** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 4 total.

Other Apertus → AllenAI aliases that hit the shared set: _none_.

## Cross-corpus agreement over the shared tasks (the result)

Best variant `rel_star_discrepancy`, n = 3 shared tasks:

| metric | value |
|---|---:|
| **Pearson r** (log₁₀ SNR values) | **+0.684** |
| **Spearman ρ** (rank order) | **+0.500** |

> With only 4 shared tasks, **top-K set overlap is NOT a result** — any K ≥ 4 spans the whole universe, so Jaccard is trivially 1.0. Only K < 4 is reported below.

## Top-K agreement (non-trivial K only)

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 3 | 2 | 0.67 | 0.50 | arc_easy, hellaswag |

## Full ranking per corpus (all shared tasks)

### Apertus

| task          |    snr |
|:--------------|-------:|
| hellaswag     | 45.717 |
| arc_challenge | 38.331 |
| arc_easy      | 32.896 |

### AllenAI

| task          |     snr |
|:--------------|--------:|
| mmlu          | 155.287 |
| hellaswag     |  86.073 |
| arc_easy      |  62.626 |
| arc_challenge |  53.364 |
