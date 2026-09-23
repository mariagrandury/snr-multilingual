# RQ0 — The above-random gate and the ladder's curves

## Research question

> How does benchmark accuracy move with compute across the language settings
> and across model scale, which benchmarks separate the settings most, and
> which benchmarks even clear chance? (The above-random gate is foundational —
> every RQ depends on it.) The folder also holds the ladder's descriptive
> curves — training loss and benchmark accuracy against the fraction of the
> run — drawn from the analysis loader, so they share every assumption of
> the other RQs.

<!-- BEGIN auto:highlight (run_apertus.py --pool predictivity) -->
## Highlighted result

- **The benchmarks that separate the language settings most: `cultural_bench_easy`, `cultural_bench_hard`, `acp_bench_mcq`** — top-3 families by Signal ((max−min)/mean of per-setting final scores) at 1.7B.
- **Above-random gate.** Of **975 benchmarks, 543 clear chance at ≥1 size** and 499 at 1.7B (432 are random everywhere). The at-chance cells are removed before any SNR is computed; the breakdown by answer count below shows how much of the gate is an option-count effect.
<!-- END auto:highlight -->

## Experimental setup

Curves under `pretraining/<pool>/{per_benchmark,per_language}/`: accuracy vs
FLOPs (log-x, `6 × (N_non_emb + d·V) × D`), one curve per language setting
(`plotted_mixes` in the `snr` config: L1 … L50, deep, scheme A, seed 1904)
and per size 175M–1.7B (the 90M rung is dropped at load, rule 10). Tasks are
parent-aggregated (subjects collapse into the parent; languages stay distinct); each task's **Signal** = (max−min)/mean of
the per-setting final scores at the reference size (1.7B, or the largest size with
data); only the top-3 families by Signal get curve grids. The 36-sweep pools
still draw their three data mixtures and overlay the external models to 70B.

The **above-random gate** ([`above_random.py`](above_random.py)) is
foundational and depends **only** on raw eval scores and the answer-option
counts — `n_options` in `configs/tasks.json` where it was derived from the
evaluated samples, the per-family table in that file otherwise — it reads no
RQ output, so every RQ depends on the gate, never the reverse. A run is above
chance on a task when the one-sided 95 % Wilson lower bound of its accuracy
over the task's `n_items` clears `1/n_options`, and a `(benchmark, size)` cell
is above random when at least half of the size's runs that train the language
are (`MIN_SHARE`). There is no fixed margin: the earlier
`mean score > 1/n_options + 0.05` rule is gone, so a 2-option task with 500
items and a 4-option task with 100 are held to the same evidence.
`run_apertus_snr_variants.py` NaN-s every at-chance cell so the gate propagates
to all downstream RQs. Per-language BPB and generative tasks have no chance
level and are never gated.

## Findings (ladder snapshot 2026-09-23 06:16)

Read from the `predictivity` tables next to the figures (the ≤ 600M snapshot
of 2026-09-01 that stood here is superseded); the write-up is
[`../rq02_decision_accuracy/pretraining/predictivity/README.md`](../rq02_decision_accuracy/pretraining/predictivity/README.md), §1.

- **Half of the evaluation is at chance somewhere on the ladder.** Of the 975
  benchmark-language tasks with a chance level in `above_random_mask.csv`,
  297 / 378 / 408 / 448 / 499 are above chance at 175M / 350M / 600M / 1B /
  1.7B; 543 clear the gate at ≥ 1 size, 432 nowhere, and 44 are above chance
  at a smaller size but at chance at 1.7B. Of the 775 (family, language)
  cells of `first_size_above_random.csv` (38 families), 220 hold from 175M
  on, 80 from 350M, 30 from 600M, 39 from 1B, 55 only at 1.7B and 351 never.
- **The gate is an option-count effect, and the reformulation undoes it.** At
  1.7B the letter-answer four-option families are absent — `global_mmlu_full`
  0 / 37 languages, `global_piqa_parallel_cloze` 1 / 91, `include_base_44`
  4 / 43, `belebele` 11 / 105 — against `multiblimp` 57 / 57, `xwinograd`
  6 / 6, `hellaswag` 26 / 31, `xnli` 15 / 18; the reformulated twins pass
  where the originals fail (`rf_global_mmlu_full` 35 / 37, `rf_belebele`
  86 / 105, `rf_include_base_44` 31 / 43, `rfgm_include_base_44` 33 / 43).
- **What the Wilson bound asks** (`above_random_thresholds.csv`, median task
  per family): the margin over chance a single run needs is +0.04 on
  `xwinograd` (409 items), +0.05–0.08 on the 130–250-item `bbh` / `acp_bench`
  probes, +0.09 on the 100-item cloze probes and +0.16 on `cultural_bench_easy`
  (25 items) — which a model below 1B rarely delivers on a four-option task in
  a non-English language.
- BPB and the training loss have no chance level and are never gated; how
  they scale is rq01's question (`scaling_law_error.csv`).

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
  [compute_da.py](../rq02_decision_accuracy/compute_da.py); the gate and curves
  here just consume its trajectories.)
- **Cross-bucket scaling-DA is family-coverage-limited.** Scaling-DA auto-detects
  bucket pairs where ≥2 model *families* span both sizes (via within-family
  ladders such as gemma-3 / OLMo). Above 1B this is sparse — few families ship
  multiple sizes — so the few detectable pairs saturate at DA 1.0 on a handful of
  shared tasks: directionally useful, not yet statistically strong. acc-vs-FLOPs
  curves and the signal pool carry the >1B scaling story for now.

<!-- BEGIN auto:results (run_apertus.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate: `python analysis/rq00_gate_and_curves/above_random.py --only predictivity` and `python analysis/rq00_gate_and_curves/run_apertus.py --pool predictivity`.

**Top benchmarks by Signal across language settings** (full ranking in `pretraining/predictivity/acc_vs_flops_signal.csv`):

| task | family | lang | Signal |
|---|---|---|---|
| `rf_bbh_mcq_temporal_sequences` | rf_bbh_mcq | en | 3.391 |
| `cultural_bench_easy_australia` | cultural_bench_easy | en | 2.526 |
| `cultural_bench_easy_canada` | cultural_bench_easy | en | 2.308 |
| `cultural_bench_easy_united_states` | cultural_bench_easy | en | 2.143 |
| `cultural_bench_easy_south_africa` | cultural_bench_easy | en | 2.113 |

![top-Signal family accuracy vs FLOPs](pretraining/predictivity/per_benchmark/cultural_bench_easy.png)

**Above-random gate** — a (benchmark, size) cell is kept when at least 50% of the size's runs clear chance (`1/n_options`) with the Wilson 90% lower bound of their accuracy over the task's items; `run_apertus_snr_variants.py` NaN-s every at-chance `(benchmark, size)` SNR cell, so the gate propagates to all RQs:

| options | chance | above ≥1 size | above @1.7B |
|---|---|---|---|
| 2 | 0.50 | 92 / 138 | 91 / 138 |
| 3 | 0.33 | 19 / 24 | 17 / 24 |
| 4 | 0.25 | 423 / 791 | 383 / 791 |
| 5 | 0.20 | 4 / 9 | 4 / 9 |
| 6 | 0.17 | 1 / 4 | 1 / 4 |
| 7 | 0.14 | 1 / 4 | 1 / 4 |
| 8 | 0.12 | 2 / 2 | 1 / 2 |
| 10 | 0.10 | 0 / 2 | 0 / 2 |
| 12 | 0.08 | 1 / 1 | 1 / 1 |
<!-- END auto:results -->

<!-- BEGIN auto:curves (curves.py --pool predictivity_all) -->
## Curves on the analysis' cells

Every cell the `predictivity_all` pool holds (all seeds and schemes), after the loader has dropped diverged and unfinished runs and restricted checkpoints to the shared grid: the detailed counterpart of the progress report's figures. Loss per L, the whole run and its last 10 % (capped at 3.5 nats, where arch and scheme separate); benchmark accuracy as the mean over the tasks in the languages the cell trains on, one line per cell, chance from the option count. Regenerate with `python analysis/rq00_gate_and_curves/curves.py --pool predictivity_all`.

![Loss curves](pretraining/predictivity_all/loss_curves.png)

![Benchmark curves](pretraining/predictivity_all/benchmark_curves.png)
<!-- END auto:curves -->

## Custom vs. external: the at-chance problem is a capability artifact

*36-sweep numbers (pools `custom` and `external`, 2026-06, custom sizes
175M–1B, RQ numbers of that layout); the ladder's counterpart is the
benchmark-floor block below.*

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

Headline numbers from the `custom_swissai_hf` pool (Signal) and the `custom` above-random report. Regenerate: `python analysis/rq00_gate_and_curves/run_apertus.py --pool custom_swissai_hf` and `python analysis/rq00_gate_and_curves/above_random.py`.

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

- [x] How far below chance the gated-out families sit:
      `gate_margin_by_benchmark.png` / `gate_margin_by_language.png` (panels block).
- [x] The scale at which late-blooming benchmarks cross chance:
      `first_size_above_random.png` / `.csv` and `benchmark_floor.png` (blocks below).

## Files

- `pretraining/<pool>/acc_vs_flops_signal.csv` — per-task mixture-Signal (full
  ranking; all parent tasks).
- `pretraining/predictivity/above_random_{scores,share,mask,runs,thresholds}.csv`
  — the ladder's gate (`above_random.py`): mean score, share of runs above
  chance, the 1/0/blank mask every RQ reads, the per-run Wilson bounds, and
  the margin each rule implies. `first_size_above_random.csv`,
  `gate_margin.csv`, `highlights.csv`, `score_curves.csv`, `benchmark_floor.csv`
  sit next to the panels of the same name.
- `pretraining/seeds_28_1797_1904/`, `pretraining/custom_swissai_hf/`,
  `all/external/` — the 36-sweep's `above_random_scores.csv` /
  `above_random_mask.csv` (pure-custom, all-models, and non-custom reports,
  pool-named).
- `…/per_benchmark/<family>.png` — top-3 families, subplots per language,
  external scaling markers overlaid.
- `…/per_language/<lang>.png` — per language, subplots = top-3 families.
- `pretraining/predictivity_all/loss_curves.png`, `benchmark_curves.png` — the
  ladder's curves on the analysis' cells (`curves.py`).

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

The gate and the curves without the aggregation (`predictivity` pool). Regenerate with `python analysis/rq00_gate_and_curves/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq00 in one figure](pretraining/predictivity/highlights.png)

![Smallest size above chance](pretraining/predictivity/first_size_above_random.png)

![Margin above chance per benchmark](pretraining/predictivity/gate_margin_by_benchmark.png)

![Margin above chance per language](pretraining/predictivity/gate_margin_by_language.png)

![What the gate asks, and what each rule keeps](pretraining/predictivity/above_random_thresholds.png)

Score along the run, one figure per language (50 languages, one subplot per benchmark, one line per size): `pretraining/predictivity/score_curves/<language>.png`, e.g.

![Score curves, German](pretraining/predictivity/score_curves/de.png)
<!-- END auto:panels -->

<!-- BEGIN auto:benchmark-floor (benchmark_floor.py --pool predictivity) -->
## Benchmark floors: the ladder's gate against the public models'

Of the 84 tasks both tiers score, 35 are at chance at every ladder size. Where the public base models (270M–70B, same gate) first read them: ≤ 600M 6, 1B–1.7B 19, 3B–4B 0, 7B–14B 5, ≥ 27B 2, never 3. A task readable at ≤ 1.7B by a public model is a size/recipe floor (the 5×-Chinchilla ladder does not reach it; public models of that size train on 10–36 T tokens); a task that needs ≥ 3B or is never read is a benchmark or language-resource floor. Overlap is the 36-sweep's 86-task list only. Regenerate with `python analysis/rq00_gate_and_curves/benchmark_floor.py --pool predictivity`.

| family | ≤ 600M | 1B–1.7B | 3B–4B | 7B–14B | ≥ 27B | never |
|---|---|---|---|---|---|---|
| belebele | 3 | 8 | 0 | 0 | 0 | 0 |
| global_mmlu_full | 1 | 9 | 0 | 0 | 0 | 0 |
| arc | 0 | 1 | 0 | 2 | 0 | 1 |
| xnli | 0 | 0 | 0 | 2 | 0 | 1 |
| paws | 0 | 1 | 0 | 0 | 0 | 1 |
| xstorycloze | 0 | 0 | 0 | 1 | 1 | 0 |
| commonsense_qa | 1 | 0 | 0 | 0 | 0 | 0 |
| mmlu | 1 | 0 | 0 | 0 | 0 | 0 |
| xcopa | 0 | 0 | 0 | 0 | 1 | 0 |

![Benchmark floors](pretraining/predictivity/benchmark_floor.png)
<!-- END auto:benchmark-floor -->
