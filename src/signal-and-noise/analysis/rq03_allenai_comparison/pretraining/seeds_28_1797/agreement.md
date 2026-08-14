# Top-K reliability agreement

Variant used: **Dispersion (Shift+Scale)** (`dispersion_shifted`)
Apertus SNR column: `snr_dispersion_shifted_1B`  ·  AllenAI SNR column: `snr_dispersion_shifted_1B`
Shared-task universe: **7** tasks.

## ⚠️ Methodological caveat — MMLU aliasing

Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison can use the ~60 MMLU subjects. **The two are not the same content.** Apertus runs the **Cohere Full** translation/post-edit of MMLU (English split), AllenAI runs the original Hendrycks et al. MMLU. Question wording, post-edits, and sample coverage may differ. Plan: re-run the original `mmlu` lm-eval task on the multilingual Apertus checkpoints; once that lands, drop the alias and compare like-for-like.

MMLU rows aliased into the shared set: **1** of 7 total.

Other Apertus → AllenAI aliases that hit the shared set: `commonsense_qa → csqa`.

## Cross-corpus agreement over the shared tasks (the result)

Best variant `dispersion_shifted`, n = 4 shared tasks:

| metric | value |
|---|---:|
| **Pearson r** (log₁₀ SNR values) | **+0.998** |
| **Spearman ρ** (rank order) | **+1.000** |

> With only 7 shared tasks, **top-K set overlap is NOT a result** — any K ≥ 7 spans the whole universe, so Jaccard is trivially 1.0. Only K < 7 is reported below.

## Top-K agreement (non-trivial K only)

| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |
|---|---:|---:|---:|---|
| 5 | 3 | 0.60 | 0.50 | arc_easy, hellaswag, piqa |

## Full ranking per corpus (all shared tasks)

### Apertus

| task          |     snr |
|:--------------|--------:|
| piqa          | 159.718 |
| hellaswag     |  96.76  |
| arc_easy      |  85.958 |
| arc_challenge |  45.681 |

### AllenAI

| task          |     snr |
|:--------------|--------:|
| piqa          | 350.423 |
| hellaswag     | 226.902 |
| mmlu          | 218.486 |
| arc_easy      | 188.945 |
| csqa          | 141.846 |
| arc_challenge | 103.055 |
| openbookqa    |  77.285 |
