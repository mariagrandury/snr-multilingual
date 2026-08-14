# Top-K reliability agreement

Variant used: **Projection Depth** (`projection`)
Apertus SNR column: `snr_projection_1B`  ·  AllenAI SNR column: `snr_projection_1B`
Shared-task universe: **7** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 7 total.

Other Apertus → AllenAI aliases that hit the shared set: `commonsense_qa → csqa`.

## Cross-corpus agreement over the shared tasks (the result)

Best variant `projection`, n = 4 shared tasks:

| metric | value |
|---|---:|
| **Pearson r** (log₁₀ SNR values) | **+0.901** |
| **Spearman ρ** (rank order) | **+0.800** |

> With only 7 shared tasks, **top-K set overlap is NOT a result** — any K ≥ 7 spans the whole universe, so Jaccard is trivially 1.0. Only K < 7 is reported below.

## Top-K agreement (non-trivial K only)

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 4 | 0.80 | 0.80 | arc_challenge, arc_easy, hellaswag, piqa |

## Full ranking per corpus (all shared tasks)

### Apertus

| task          |     snr |
|:--------------|--------:|
| arc_challenge | 122.087 |
| hellaswag     |  67.281 |
| piqa          |  60.396 |
| arc_easy      |  31.952 |

### AllenAI

| task          |     snr |
|:--------------|--------:|
| mmlu          | 331.53  |
| arc_challenge |  97.848 |
| piqa          |  94.35  |
| hellaswag     |  87.793 |
| arc_easy      |  71.498 |
| openbookqa    |  65.421 |
| csqa          |  45.718 |
