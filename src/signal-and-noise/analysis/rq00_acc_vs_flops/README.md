# Accuracy-vs-FLOPs curves + the above-random gate

## Research question

> How does benchmark accuracy move with compute across the language settings
> and across model scale, which benchmarks separate the settings most, and
> which benchmarks even clear chance? (The above-random gate is foundational —
> every RQ depends on it.)

<!-- BEGIN auto:highlight (run_apertus.py --pool predictivity) -->
## Highlighted result

_Not generated yet for the predictivity ladder — the ladder report (`msnr-data/ladder-report`) was not published when this README was written. `bash run_all_predictivity.sh` fills this block from the `predictivity` pool._
<!-- END auto:highlight -->

## Experimental setup

Curves under `pretraining/<pool>/{per_benchmark,per_language}/`: accuracy vs
FLOPs (log-x, `6 × (N_non_emb + d·V) × D`), one curve per language setting
(`plotted_mixes` in the `snr` config: L1 … L100, deep, scheme A, seed 1904)
and per size 90M–1.7B. Tasks are parent-aggregated (subjects collapse into the
parent; languages stay distinct); each task's **Signal** = (max−min)/mean of
the per-setting final scores at the target size (1B, or the largest size with
data); only the top-3 families by Signal get curve grids. The 36-sweep pools
still draw their three data mixtures and overlay the external models to 70B.

The **above-random gate** ([`above_random.py`](above_random.py)) is
foundational and depends **only** on raw eval scores and the answer-option
counts — `n_options` in `configs/tasks.json` where it was derived from the
evaluated samples, the per-family table in that file otherwise — it reads no
RQ output, so every RQ depends on the gate, never the reverse. A
`(benchmark, size)` cell is above random iff `mean score > 1/n_options + 0.05`;
`run_apertus_snr_variants.py` NaN-s every at-chance cell so the gate propagates
to all downstream RQs. Per-language BPB and generative tasks have no chance
level and are never gated.

## Preliminary findings (ladder snapshot, 2026-09-01)

From the ≤ 600M ladder's eval results (`plan/status-09-01.md`, §3; 90M
excluded as diverged):

- Real signal already at these sizes: MultiBLiMP (0.65 → 0.92 from 90M →
  600M, chance 0.5) and clear size-monotone growth on HellaSwag (0.25 → 0.30),
  XNLI (0.33 → 0.42), XStoryCloze (0.48 → 0.57), XWinograd (0.51 → 0.64),
  XCOPA.
- Still at chance even at 600M: Belebele, Global-MMLU, INCLUDE and
  ARC-multilingual all sit at 0.24–0.26 (chance 0.25) — the knowledge-heavy
  4-option benchmarks have not emerged at this compute, which is the gate's
  domain.
- BPB (the plan's outcome metric): non-English macro BPB falls monotonically
  with size (L50: 7.03 → 1.61 → 1.38 across 90M/175M/350M) and over training
  within each healthy run; English BPB is identical between L2 and L50 at every
  size (0.947 vs 0.946 at 350M) while L50 beats L2 on 81–89 of 99 non-English
  languages by ~0.4–0.5 bits/byte — enormous against a checkpoint noise of
  ~0.002.

## Methodology — scaling beyond 1B (36-sweep pools)

The `custom_swissai_hf` pool extends the compute axis past the custom 1B ceiling
by folding in the apertus3 a06, distilled, and reference-HF trajectories. Three
mechanisms make that work:

- **Bucketed size axis.** The four custom sizes (175M / 350M / 600M / 1B) stay
  singletons; nearby external/large sizes pool into buckets so each holds ≥2
  models. A bucket with ≥2 models contributes a computable cross-model signal and
  decision accuracy; a singleton bucket (e.g. an isolated 70B) still plots its
  final-checkpoint marker on the curves but is dropped from the noise/DA pool.
- **Relative-fraction ckpt-DA.** External trajectories never hit the custom
  absolute megatron iters, so checkpoint-DA selects each model's early checkpoint
  at a *fraction* of its own max step rather than a fixed iter — letting the a06,
  distillation, SmolLM3, Olmo-3 and Apertus-8B series participate and extending
  ckpt-DA into the large buckets. (The computation lives in
  [compute_da.py](../rq01_decision_accuracy/compute_da.py); the gate and curves
  here just consume its trajectories.)
- **Cross-bucket scaling-DA is family-coverage-limited.** Scaling-DA auto-detects
  bucket pairs where ≥2 model *families* span both sizes (via within-family
  ladders such as gemma-3 / OLMo). Above 1B this is sparse — few families ship
  multiple sizes — so the few detectable pairs saturate at DA 1.0 on a handful of
  shared tasks: directionally useful, not yet statistically strong. acc-vs-FLOPs
  curves and the signal pool carry the >1B scaling story for now.

<!-- BEGIN auto:results (run_apertus.py --pool predictivity) -->
## Results

_Not generated yet for the predictivity ladder — the ladder report (`msnr-data/ladder-report`) was not published when this README was written. `bash run_all_predictivity.sh` fills this block from the `predictivity` pool._
<!-- END auto:results -->

## Custom vs. external: the at-chance problem is a capability artifact

This is the foundational result the rest of the paper rests on. The
above-random gate is identical for every model set, but what it removes depends
entirely on *who is being evaluated*.

**On the custom pretrains the gate is brutally selective.** Of 118 benchmarks,
only **44 clear chance at ≥1 size and 74 are random everywhere** — the 175M–1B
models we train are simply too weak to register signal on most translated
knowledge MCQA. The loss is concentrated in the 4-option families: only **9 / 63**
clear chance, so `belebele`, `global_mmlu_full`, `arc`, and `truthfulqa` are
NaN-ed out before any SNR is computed. The surviving pool is therefore *almost
entirely 2-option*, which is exactly what later biases the design-feature
analyses (RQ2/RQ5) toward "fewer options ⇒ higher SNR".

**Re-running the identical gate on the external tier dissolves the penalty.** The
`external` model set (every non-custom model — `reference_hf` + a06 + distillation
+ posttraining, sizes 270M…70B; gate report in `all/external/`, curves skipped)
clears chance on **122 / 124** benchmarks, including **68 / 69** four-option
families. A 4-option translated benchmark is not intrinsically low-signal — it
only looks that way under models too small to beat chance.

| model set | models | benchmarks | above ≥1 size | 2-opt | 3-opt | 4-opt | 5-opt |
|---|---|---|---|---|---|---|---|
| `custom` (SNR-gate domain) | 175M–1B custom pretrains | 118 | **44 (37%)** | 28 / 42 | 7 / 11 | **9 / 63** | 0 / 2 |
| `external` (`all/external`) | 270M–70B reference / a06 / distill / post | 124 | **122 (98%)** | 42 / 42 | 10 / 11 | **68 / 69** | 2 / 2 |

The same benchmark tells the story directly: on the `custom_swissai_hf`
acc-vs-FLOPs curves, Belebele sits flat at chance across the custom 175M–1B
sweep, then the overlaid external final-checkpoint markers climb steeply toward
0.8+ out to 70B.

![Belebele: custom at chance, externals climb to 0.8+](pretraining/custom_swissai_hf/per_benchmark/belebele.png)

**Implication for the paper.** The custom-pool gate measures *our models' scale*,
not benchmark reliability; the external tier is the control that separates the
two. This single distinction drives the external-tier findings downstream: option
count stops predicting SNR (RQ2), and 4-option HellaSwag becomes the
*highest*-SNR family once it clears the gate (RQ5).


## Results from the 36-model sweep (2026-06, superseded)

The numbers below were generated on the 36-model sweep (4 sizes × 3 data mixtures × 3 seeds, 12 languages, pool `custom_swissai_hf` unless stated) and are kept as history; the predictivity ladder regenerates the blocks above.

### Highlighted result

- **The benchmarks that separate data mixtures most: `agieval_sat`, `belebele`, `arabic_leaderboard_alghafa_mcq_exams_test`** — top-3 families by mixture-Signal ((max−min)/mean of per-mix final scores) at 1B.
- **Mixture-Signal ≠ reliability.** These top-Signal families are exactly the ones the above-random gate **removes** — they sit at chance, so they never enter the SNR analysis. Of **118 benchmarks, 44 clear chance at ≥1 size** (74 are random everywhere) — almost entirely an answer-count effect.

### Results

Headline numbers from the `custom_swissai_hf` pool (Signal) and the `custom` above-random report. Regenerate: `python analysis/rq00_acc_vs_flops/run_apertus.py --pool custom_swissai_hf` and `python analysis/rq00_acc_vs_flops/above_random.py`.

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
## TODO

- [ ] Per-language curve panels for the gated-out families to visualise *how far*
      below chance they sit (not just that they fail the gate).
- [ ] Annotate the scale at which late-blooming benchmarks (`arc_challenge`,
      `xnli_th`, `paws_en`) cross chance.

## Files

- `pretraining/<pool>/acc_vs_flops_signal.csv` — per-task mixture-Signal (full
  ranking; all parent tasks).
- `pretraining/seeds_28_1797_1904/`, `pretraining/custom_swissai_hf/`,
  `all/external/` — `above_random_scores.csv` / `above_random_mask.csv` (the
  gate; pure-custom, all-models, and non-custom reports, pool-named). From
  `analysis/rq00_acc_vs_flops/above_random.py`.
- `…/per_benchmark/<family>.png` — top-3 families, subplots per language,
  external scaling markers overlaid.
- `…/per_language/<lang>.png` — per language, subplots = top-3 families.
