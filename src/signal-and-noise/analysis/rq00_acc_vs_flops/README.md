# Accuracy-vs-FLOPs curves + the above-random gate

## Research question

> How does benchmark accuracy move with compute across the three data mixtures
> and across model scale, which benchmarks separate the mixtures most, and which
> benchmarks even clear chance? (The above-random gate is foundational — every
> RQ depends on it.)

<!-- BEGIN auto:highlight (run_apertus.py --pool custom_swissai_hf) -->
## Highlighted result

- **The benchmarks that separate data mixtures most: `agieval_sat`, `belebele`, `arabic_leaderboard_alghafa_mcq_exams_test`** — top-3 families by mixture-Signal ((max−min)/mean of per-mix final scores) at 1B.
- **Mixture-Signal ≠ reliability.** These top-Signal families are exactly the ones the above-random gate **removes** — they sit at chance, so they never enter the SNR analysis. Of **118 benchmarks, 44 clear chance at ≥1 size** (74 are random everywhere) — almost entirely an answer-count effect.
<!-- END auto:highlight -->

## Experimental setup

Curves under `pretraining/<pool>/{per_benchmark,per_language}/`: accuracy vs
FLOPs (log-x), custom models as per-mixture curves (seed 1904); on
`custom_swissai_hf` the external pretraining models (a06, distillation,
swiss-ai/HF base) overlay as final-checkpoint markers out to 70B. Tasks are
parent-aggregated (subjects collapse into the parent; languages stay distinct);
each task's **Signal** = (max−min)/mean of per-mix final scores at 1B; only the
top-3 families by Signal get curve grids.

The **above-random gate** ([`above_random.py`](../../multilingual/above_random.py))
is foundational and depends **only** on raw eval scores and the intrinsic
per-family answer-option counts (`N_OPTIONS` in that file) — it reads no RQ
output, so every RQ depends on the gate, never the reverse. A `(benchmark, size)`
cell is above random iff `mean score > 1/n_options + 0.05`;
`run_apertus_snr_variants.py` NaN-s every random cell so the gate propagates to
all downstream RQs. Numbers below are the `custom` report (custom pretrains,
buckets 175M…1B — the SNR gate's domain).

<!-- BEGIN auto:results (run_apertus.py --pool custom_swissai_hf) -->
## Results

Headline numbers from the `custom_swissai_hf` pool (Signal) and the `custom` above-random report. Regenerate: `python multilingual/run_apertus.py --pool custom_swissai_hf` and `python multilingual/above_random.py`.

**Top benchmarks by mixture-Signal** (full ranking in `pretraining/custom_swissai_hf/acc_vs_flops_signal.csv`):

| task | family | lang | Signal |
|---|---|---|---|
| `agieval_sat_en` | agieval_sat | en | 0.268 |
| `belebele_hin_Deva` | belebele | hi | 0.245 |
| `belebele_zho_Hans` | belebele | zh | 0.236 |
| `global_piqa_completions_arb_arab` | global_piqa_completions | ar | 0.229 |
| `belebele_eng_Latn` | belebele | en | 0.222 |

![top-Signal family accuracy vs FLOPs](pretraining/custom_swissai_hf/per_benchmark/agieval_sat.png)

**Above-random gate** — a benchmark must beat chance (`1/n_options`) by +0.05; `run_apertus_snr_variants.py` NaN-s every random `(benchmark, size)` SNR cell, so the gate propagates to all RQs. Almost entirely an answer-count effect:

| options | chance | above ≥1 size | above @1B |
|---|---|---|---|
| 2 | 0.50 | 28 / 42 | 28 / 42 |
| 3 | 0.33 | 7 / 11 | 7 / 11 |
| 4 | 0.25 | 9 / 63 | 7 / 63 |
| 5 | 0.20 | 0 / 2 | 0 / 2 |
<!-- END auto:results -->

## TODO

- [ ] Per-language curve panels for the gated-out families to visualise *how far*
      below chance they sit (not just that they fail the gate).
- [ ] Annotate the scale at which late-blooming benchmarks (`arc_challenge`,
      `xnli_th`, `paws_en`) cross chance.

## Files

- `pretraining/<pool>/acc_vs_flops_signal.csv` — per-task mixture-Signal (full
  ranking; all parent tasks).
- `pretraining/custom/`, `pretraining/custom_swiss_hf/` —
  `above_random_scores.csv` / `above_random_mask.csv` (the gate; custom-only and
  all-models reports). From `multilingual/above_random.py`.
- `…/per_benchmark/<family>.png` — top-3 families, subplots per language,
  external scaling markers overlaid.
- `…/per_language/<lang>.png` — per language, subplots = top-3 families.
