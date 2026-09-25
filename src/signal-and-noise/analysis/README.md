# Analysis — the research questions

Which (subsets of) benchmarks give a reliable signal at each stage of
multilingual pretraining? The study extends the Signal-and-Noise framework
(Heineman et al., 2025) to multilingual models: a benchmark is useful when a
cheap measurement — a smaller model, an earlier checkpoint, a statistic of the
proxy alone — makes the decision the reference-size model would make. Ten
research questions in four themes, one folder each; `rqNN_*/README.md` is the
single document of its question (the README rules at the end of
[RULES.md](RULES.md)), and the outputs live under `<rq>/<stage>/<pool>/`.
Every hand-written number in these READMEs comes from the ladder-report
snapshot **2026-09-23 06:16**, the one the tables on disk were built from; a
refresh regenerates the auto blocks and moves them.

## The questions

| theme | RQ | question (one sentence) | main figure |
|---|---|---|---|
| **A. Is the evaluation predictable?** | [rq00_gate_and_curves](rq00_gate_and_curves/README.md) | Which benchmark-language tasks are above chance at each size, and is what the ladder cannot read a size floor or a benchmark floor? | [first_size_above_random](rq00_gate_and_curves/pretraining/predictivity/first_size_above_random.png) |
| | [rq00_task_reformulation](rq00_task_reformulation/README.md) | Do the letter-format families clear chance once scored on their answer strings (`rf_`) or on Gemini-rewritten items (`rfgm_`), and does that change any headline reading? | [reformulations_gate](rq00_task_reformulation/reformulations_gate.png) |
| | [rq01_scaling_predictability](rq01_scaling_predictability/README.md) | Which tasks move with model size in a way a log-linear fit captures, and how well does a fit on the proxy rungs predict the reference? | [scaling_regimes_outliers_paper](rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_outliers_paper.png) |
| | [rq02_decision_accuracy](rq02_decision_accuracy/README.md) | Does a benchmark rank the ladder's design variants at a smaller size, or at an earlier checkpoint, the way the 1.7B reference does? | [scale_convergence](rq02_decision_accuracy/pretraining/predictivity/scale_convergence.png) |
| **B. Can it be measured cheaply?** | [rq03_noise_and_snr](rq03_noise_and_snr/README.md) | How much does a score move with the seed or the checkpoint alone, how large is a design effect against that noise, and what is each benchmark's SNR under 22 definitions? | [effect_vs_noise](rq03_noise_and_snr/pretraining/predictivity_all/effect_vs_noise.png) |
| | [rq04_surrogates](rq04_surrogates/README.md) | Which statistic computed on the proxy alone — an SNR definition, its signal or noise part, early-checkpoint agreement, a scaling fit, FineTasks' criteria — predicts decision accuracy? | [rq3_surrogates](rq04_surrogates/pretraining/predictivity/rq3_surrogates.png) |
| **C. Does the framework generalise?** | [rq05_design_decisions](rq05_design_decisions/README.md) | For one design decision at a time (depth, language list, temperature, second language), which proxy sizes, and how early in their run, read the reference's preference at each language count? | [rq4_interventions](rq05_design_decisions/pretraining/predictivity_all/rq4_interventions.png) |
| | [rq06_language_transfer](rq06_language_transfer/README.md) | Does per-language scaling transfer to languages never measured or never trained, and which languages must a developer evaluate to recover the multilingual decision? | [rq5_transfer](rq06_language_transfer/pretraining/predictivity_all/rq5_transfer.png) |
| | [rq07_external_frameworks](rq07_external_frameworks/README.md) | Do our SNR values agree with AllenAI DataDecide on the English tasks both corpora evaluate? | [snr_apertus_vs_snr_allenai_grid](rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_grid.png) |
| **D. Can the benchmarks be improved?** | [rq08_subset_selection](rq08_subset_selection/README.md) | Can a language, subject or item subset of a benchmark beat the full set's SNR by more than selection alone gives for free? | [gain_over_null](rq08_subset_selection/pretraining/predictivity/gain_over_null.png) |
| | [rq09_benchmark_design](rq09_benchmark_design/README.md) | Which design features of a benchmark — curation, source, format, option count, item length — go with a high SNR? | [snr_per_family_ranked](rq09_benchmark_design/pretraining/predictivity/snr_per_family_ranked.png) |
| **E. Past the reference** | [rq10_size_generalisation](rq10_size_generalisation/README.md) | Does a ranking that holds at the 1.7B reference still hold one rung above it, at 3B (the only reader of `above_reference=True`; waiting for the 3B evaluations)? | [above_reference_3B](rq10_size_generalisation/pretraining/predictivity/above_reference_3B.png) |

**The paper's figures.** `documents/paper/figures/make_rq_figures.py` copies
them from the analysis, never the reverse: `rq1` ← rq01
`scaling_regimes_outliers_paper` (appendix `scaling_regimes_by_family_paper`);
`rq2` ← rq02 `rq2` (`paper_rq2.py`, the three decision accuracies; the
ten-checkpoint read `rq2_ten_checkpoints` sits beside it) and `rq2_early_small`
← rq05 `early_decision.py`; `rq3` ← rq04 `rq3_surrogates`; `rq4` ← rq05
`rq4_interventions`; `rq5` ← rq06 `rq5_transfer`. The four report figures
(`report_figures/make_figures.py`) are the 36-sweep's.

## Pools

One input, the published ladder report (`ladder_report.csv`, loaded by
`snr/download/ladder.py`; diverged and unfinished runs dropped, checkpoints on
the shared k/10 and k/20 grid). The models are the predictivity ladder — sizes
175M–1.7B × L ∈ {1, 2, 8, 15, 30, 50} × depth (deep, shallow) × data scheme
(A, B, AT3, ZH, ES) × seeds; a "design variant" is one such cell, and its
cross-size identity (`family`, everything but the size) is what a decision
pairs. The pools (`configs/models.json`): **`predictivity`** is the plan grid
at seed 1904 with schemes A and B, the headline pool of rq00, rq02, rq03,
rq04, rq07, rq08 and rq09 and the gate pool of every rq02-family figure;
**`predictivity_schemes`** adds AT3, ZH and ES at the grid seed and is the pair
set the rq02 decision figures are computed over (a temperature or a second
language is a design decision; it would only widen an SNR signal, which is why
the headline pool keeps the A/B filter); **`predictivity_seeds`** adds every
replicate seed as a separate model, and **`predictivity_seeds_train` /
`_test`** split the ×3 cells (seeds 64/313 against seed 1904) for the seed
holdout; **`predictivity_all`** is every trained cell, all seeds and schemes,
read by the ladder-frame scripts (rq00 curves, rq01, rq03 effect-vs-noise,
rq05, rq06). The 36-model sweep and the public models (`seeds_*`,
`custom_swissai_hf`, `external`) are another period of the project — a
different harness, task set and reference size (1B) — and are never pooled
with the ladder: each RQ keeps them in its final section "Extensions from
other sweeps".

## Rules

Every number follows [RULES.md](RULES.md), verified by `check_rules.py` at
the end of the driver and by the review skill before a commit: the
above-random gate (a Wilson 95 % lower bound over the task's items, at the
proxy and at the reference; gated cells grey, never dropped), trained
languages only (rq06 opts out), the ten evaluated tenths as the checkpoint
axis in Chinchilla multiples 1C–5C, one noise window (the k/20 points in the
last 20 % of a run), at least three design-variant pairs per decision cell,
parent tasks only (rq08 opts out), `multi` is not a language, three tasks per
language for a per-language correlation, one reference (1.7B; L2 ES stops at
1B), sizes 175M–1.7B (no 90M, the 3B rung only for the size-generalisation
question), no leakage from the reference into a proxy statistic, a CSV beside
every PNG, and — since 2026-09-22 — an `axes` column naming a decision
table's pair set (`multi-axis`, `mono-axis`, `seed`). The reformulated twins
are ordinary benchmarks in every population since 2026-09-22 and the twenty
promoted probe families since 2026-09-23, so a table regenerated after those
dates is not comparable with one regenerated before.

## Flow

```
rq00 gate ──► rq02 DA per task ──► rq03 SNR table (22 definitions + DA) ──► rq04 variant ranking, surrogates, FineTasks
   │                 │                      │                                  ├─► rq07 DataDecide agreement
   │                 │                      └─► rq03 seed holdout ──► rq04      ├─► rq08 subset sweeps (own SNR, rq00 gate)
   │                 └─► rq02 extensions (by_L, scale_convergence, …)          └─► rq09 design features
   └─► rq01 fits ──► rq02 scaling_vs_ranking ──► rq04 surrogates (fit R² as a candidate)
rq05 decision table ──► rq05 early decision, rq03 effect_vs_noise, rq06 decision transfer, rq06 language panel
```

## How to regenerate

**Everything**, from `src/signal-and-noise` with the `snr` env (about two
hours; from the repo root `scripts/refresh_analysis.sh` also fetches the
report and rebuilds the documents):

```bash
FORCE=1 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 bash run_all_predictivity.sh
```

`run_all_predictivity.sh` runs the passes in research-question order, which
is also the dependency order: rq00 (`above_random.py`, the gate every later
step reads; `run_apertus.py`, `curves.py`, `panels.py`; the twin comparison,
`reformulations_gate.py` and `above_random_external.py`) → rq01 (`analyze.py`,
`panels.py`, `regimes.py`, `regimes_survivorship.py`, `scaling_law_error.py`)
→ rq02 (`compute_da.py` per pool and for `predictivity_schemes`,
`da_per_benchmark.py`, `early_small.py`, `reliable_tasks.py`, `by_L.py`,
`cross_task.py`, `scale_convergence.py`, `paper_ten_checkpoints.py`,
`paper_rq2.py`, the `--axes mono-axis` twins, then `scale_convergence.py --by L
--langs L8 [--common-tasks]`, `by_language.py`, `agreement.py`,
`seed_uncertainty.py`, `scaling_vs_ranking.py`, `public_ladders.py`,
`language_tier.py`, `pair_axes.py`) → rq03 (`run_apertus_snr_variants.py` per
pool, `compare_seed_splits.py`, `panels.py`) → rq04 (`analyze_snr_variants.py`,
`snr_definition_postprocess.py`, `analyze.py`, `finetasks_criteria.py`,
`panels.py`) → rq05 (`analyze.py`, `early_decision.py`, `transformations.py`,
`panels.py`; then rq03's `effect_vs_noise.py`, which reads rq05's table) →
rq06 (`analyze.py`, `panels.py`, `language_panel.py`) → rq07 (reads rq04's
ranking) → rq08 → rq09 → rq10 (`above_reference.py`, the 3B rung) →
`report_figures/make_figures.py` → `check_rules.py`.
`FORCE=1` is needed whenever the report was regenerated since the last run
(a cached table carries the report's commit time and looks newer);
`CURVES=1` also redraws rq00's ~140 accuracy-vs-FLOPs grids (an hour).

**One figure**, without the driver (the normal way to iterate; the thread caps
are mandatory on the login node — without them OpenBLAS spawns one thread per
core and the 1000-pid slice kills the process), from `src/signal-and-noise`:

```bash
export PATH=/users/mariagrandury/miniconda3/envs/snr/bin:$PATH
PYTHONPATH=$PWD:$PWD/../../src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 SOURCE_DATE_EPOCH=0 \
  python analysis/rqNN_*/<script>.py --pool predictivity
```

Scripts that read `da_reliable_tasks.csv` (everything with an `above_*`
variant) need `reliable_tasks.py` to have run on the current DA tables first,
and a table regenerated by hand needs the panels that read it regenerated in
the same breath (bug #16 in `../CLAUDE.md`). The canonical pool
(`autodoc.CANONICAL_POOL = predictivity`) is the one whose README auto blocks
the scripts write; on other pools the generators no-op.

## Other files here

- `paths.py` (the one map from constant to folder), `utils.py` (pools, rules,
  ladder-frame helpers), `grids.py` (the per-benchmark / per-language panels
  and the smallest-level maps), `autodoc.py` (README block writer),
  `style.py` (the palette), `check_rules.py` (rule 14).
- `report_figures/make_figures.py` — the 36-sweep's report figures fig1–fig4.
- `rq00_task_reformulation/` also holds the reformulation pipeline's notes
  and pilot files (the Gemini rewrite, the probe); `rq08_subset_selection/per_sample/`
  the 36-sweep's cluster-only per-item outputs.
