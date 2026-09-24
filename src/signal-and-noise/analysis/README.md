# Analysis — the research questions

Ten folders, one per question, numbered by the four themes of the study. Each
folder holds its scripts, a README (question · highlighted result · setup ·
methodology · results · files) and its outputs under `<stage>/<pool>/`.
`paths.py` is the only module that knows the folder names; `run_all_predictivity.sh`
(one level up) runs everything in dependency order.

| theme | folder | question |
|---|---|---|
| **A. Is the evaluation predictable?** | [`rq00_gate_and_curves/`](rq00_gate_and_curves/) | Which benchmarks clear chance at each size, and how do scores move with compute and along the run? |
| | [`rq00_task_reformulation/`](rq00_task_reformulation/) | Do the letter-format families (belebele, Global-MMLU, INCLUDE) clear chance once scored on their answer strings instead of A–D? Before/after gate, per family and language. |
| | [`rq01_scaling_predictability/`](rq01_scaling_predictability/) | What moves with size in a way a power law captures, and how well does a fit on the proxy rungs predict the reference? |
| | [`rq02_decision_accuracy/`](rq02_decision_accuracy/) | Does a benchmark rank the design variants at a small size, or at an early checkpoint, the way the reference does? |
| **B. Can it be measured cheaply?** | [`rq03_noise_and_snr/`](rq03_noise_and_snr/) | How noisy is a measurement (seed, checkpoint), how big is a design effect against that noise, and what is each benchmark's SNR? |
| | [`rq04_surrogates/`](rq04_surrogates/) | Which cheap statistic predicts decision accuracy: one of the 22 SNR definitions, or something simpler? |
| **C. Does the framework generalise?** | [`rq05_design_decisions/`](rq05_design_decisions/) | Which proxy sizes, and how early in their run, rank a design decision like the reference, at each number of languages? |
| | [`rq06_language_transfer/`](rq06_language_transfer/) | Does per-language scaling transfer to languages never measured or never trained? |
| | [`rq07_external_frameworks/`](rq07_external_frameworks/) | Do our SNR values agree with AllenAI DataDecide on the shared English tasks? |
| **D. Can the benchmarks be improved?** | [`rq08_subset_selection/`](rq08_subset_selection/) | Can a language or subject subset of a benchmark beat the full set's SNR? |
| | [`rq09_benchmark_design/`](rq09_benchmark_design/) | Which design features of a benchmark predict its SNR? |

The paper's five questions map to rq01 (RQ1), rq05 (RQ2 and RQ4), rq04 (RQ3)
and rq06 (RQ5); their figures are copied to `documents/paper/figures/` as `rq1.png` … `rq5.png` by
`make_rq_figures.py` there.

## Shared setup

- **Data.** One input: the published ladder report (`ladder_report.csv`), read by
  `snr/download/ladder.py`. The loader drops diverged and unfinished runs and
  keeps the checkpoints of the shared k/10 and k/20 grid.
- **Models.** The predictivity ladder: sizes 175M–1.7B × language settings
  L ∈ {1, 2, 8, 15, 30, 50, 100} × depth (deep, shallow) × data scheme
  (A, B, AT3, ZH, ES) × seeds. A "design variant" is one such cell.
- **Pools** (`configs/models.json`):

  | pool | members | used by |
  |---|---|---|
  | `predictivity` | the plan grid, seed 1904; the headline pool | rq00, rq02, rq03, rq04, rq07, rq08, rq09 |
  | `predictivity_seeds` | every seed; replicates enter as separate models | the same scripts, as a power check |
  | `predictivity_seeds_train` / `_test` | the ×3 cells, seeds 64/313 vs seed 1904 | the seed holdout (rq03, reported in rq04) |
  | `predictivity_all` | every seed and all five schemes | rq00 curves, rq01, rq03 effect-vs-noise, rq05, rq06 |

- **Conventions.** Tasks are benchmark tasks per language plus per-language bits
  per byte (`bpb_*`) and the training loss. The rq00 gate (score above
  `1/n_options + 0.05`) sets at-chance SNR cells to NaN everywhere downstream.
  Every number here follows [RULES.md](RULES.md), the 14 analysis-wide rules
  `check_rules.py` verifies before each commit.
  Decision accuracy is `snr.metrics.decision_acc_fast`, the upstream kernel with
  the documented tie fix. Noise is the std over the noise window — the k/20
  points in the last 20 % of a run, 80/85/90/95/100 % (rule 4) — unless a
  script says seed noise. Figures use the one palette in `style.py`.
- **Reference size.** 1.7B (`snr.target_size`); the proxies are 175M–1B (the 90M rung
  trains but is dropped at load). Cells with no information yet (L15 at 1.7B)
  stay in every grid as white cells. rq07 alone reads 1B, the largest rung DataDecide has.
- **Units of training.** Every run trains D(N) = 100 N tokens, five times the
  Chinchilla-optimal 20 N. A point along a run is written as a multiple of
  Chinchilla, 1C–5C (20 %–100 % of the run; `grids.chinchilla`); the tables keep
  the fraction in `frac`.
- **A table per figure.** Everything a PNG draws is saved as a CSV of the same
  name. PNGs that show one table from different angles start with the table's
  name: `early_small.csv` → `early_small.png`, `early_small_by_benchmark.png`,
  `early_small_by_language.png`; `score_curves.csv` → `score_curves/<lang>.png`.
  This holds for the `panels.py` / `early_small.py` figures; the older aggregated
  figures still read tables under other names (listed per RQ below).
- **Empty cells.** In a grid white means no value (the benchmark does not exist
  in that language, or the cell is not trained yet) and grey means the
  above-random gate filtered the value out; in a smallest-level map red means
  no level reaches it. The line under a figure's title says how a cell is computed.
- **One page per RQ.** `highlights.png` (+ `highlights.csv`) in each RQ's pool
  folder condenses the long grids: family-level means, the best and worst
  benchmarks or languages, and how a benchmark's languages split over the levels.
- **Per benchmark and per language.** An aggregate over all benchmarks hides
  which one carries a result, so every aggregated figure has two siblings,
  `<name>_by_benchmark.png` (bits per byte first, then the benchmarks
  alphabetically) and `<name>_by_language.png`: one long figure with a cell
  grid per subplot, drawn by `grids.py` from each folder's `panels.py`.

## Flow

```
rq00 gate ──► rq02 DA per task ──► rq03 SNR table (22 variants + DA columns) ──► rq04 variant ranking, anchor, surrogates
                                        │                                         ├─► rq07 DataDecide comparison
                                        │                                         ├─► rq08 subset sweeps (own SNR, rq00 gate)
                                        └─► rq03 seed holdout ──► rq04 README      └─► rq09 design features
rq05 decision table ──► rq05 early decision, rq06 decision transfer
rq01 fits ──► rq04 surrogates (fit R² as a candidate)
```

## The folders

Outputs listed are those of the canonical pool; other pools write the same
files under their own directory.

### rq00 — The above-random gate and the ladder's curves (A)

- **Intention.** Decide which (benchmark, size) cells carry any information,
  before anything is ranked; show how accuracy and loss move with compute and
  along the run.
- **Setup.** `predictivity` for the gate and the accuracy-vs-FLOPs curves (a
  benchmark is averaged over the cells that trained its language);
  `predictivity_all` for the run curves.
- **Scripts.** `above_random.py` (gate), `run_apertus.py` (accuracy vs FLOPs,
  Signal per task), `curves.py` (loss and benchmark accuracy vs fraction of run),
  `panels.py` (the gate per benchmark and per language, score curves per language).
- **Tables.** `above_random_scores.csv`, `above_random_mask.csv`,
  `acc_vs_flops_signal.csv`.
- **Figures.** `highlights.png` (rq00 on one page); `per_benchmark/<family>.png`
  (every benchmark family and BPB), `per_language/<lang>.png` (every benchmark
  the language has); `predictivity_all/loss_curves.png`, `benchmark_curves.png`;
  `first_size_above_random.png` (language × benchmark: smallest size from which
  the score stays above chance), `gate_margin_by_benchmark.png`,
  `gate_margin_by_language.png`; `score_curves/<language>.png` (score vs training
  tokens in Chinchilla multiples, one line per size, one subplot per benchmark).

### rq01 — What scales predictably? (A, paper RQ1)

- **Intention.** Find which benchmarks and per-language measurements follow a
  trend in model size, and test the trend: fit on the proxy rungs, predict the
  reference rung.
- **Setup.** `predictivity_all`; the fits use the deep, scheme-A, seed-1904
  cells from 175M up, one series per (task, L).
- **Scripts.** `analyze.py` (log-N fits per task, loss power law per
  (L, arch, scheme)), `scaling_law_error.py` (per-language BPB prediction error
  by largest proxy rung in the fit), `panels.py`.
- **Tables.** `rq1_fits.csv`, `rq1_families.csv`, `scaling_fit.csv`,
  `scaling_law_error.csv`, `facts.json`.
- **Figures.** `scaling_regimes_outliers_paper.png/.pdf/.svg` (the paper's rq1; `scaling_regimes_by_family_paper` its appendix twin, `scaling_regimes*.csv` their tables), `rq1_scaling.png/.csv`, `scaling_fit.png`,
  `scaling_law_error.png`, `fit_r2_by_benchmark.png`, `fit_r2_by_language.png`.

### rq02 — Decision accuracy (A)

- **Intention.** The ground truth of the study: per task, does the ranking of
  the design variants at a small size (**DA-size**), at 20/40/60/80 % of a
  run (**DA-ckpt**), or early *and* small at once, match the ranking at the
  reference's final checkpoint? How small, how early and how cheaply can the
  ranking be called, per language and benchmark?
- **Setup.** `predictivity` (plus the seed pools for the holdout). Every design
  variant at a size is ranked, so L, depth and scheme are pooled in one ranking.
  DA is not gated; the gate is applied when it is summarised.
- **Scripts.** `compute_da.py` (per-task DA and the pair count behind every
  cell, and the early-and-small grid: every proxy size at 1C–5C of training
  against the reference final), `da_per_benchmark.py` (family roll-ups, README
  blocks, appendix slides), `early_small.py` (the grid's figures and the
  smallest-safe maps; safe = DA ≥ 0.75 over ≥ 3 pairs, held at every larger level).
- **Tables.** `da_per_task.csv`, `da_n_pairs_per_task.csv`,
  `da_per_benchmark.csv`, `da_per_benchmark_size.csv`, `da_per_benchmark_ckpt.csv`,
  `da_early_small_per_task.csv`, `early_small_summary.csv`, and one table per
  figure (`early_small.csv`, `da_size.csv`, `safe_size.csv`, `safe_checkpoint.csv`,
  `safe_flops.csv`, `highlights.csv`).
- **Figures.** `highlights.png` (rq02 on one page); `da_size_by_family.png`; `early_small.png` (BPB and all
  benchmarks), `early_small_by_benchmark.png`, `early_small_by_language.png`,
  `da_size_by_benchmark.png`, `da_size_by_language.png`; language × benchmark
  maps `safe_size.png` (smallest size that predicts the 1.7B ranking),
  `safe_checkpoint.png` (fewest training tokens, in Chinchilla multiples, that predict the size's own
  final ranking, one subplot per size), `safe_flops.png` (cheapest share of the
  1.7B run's FLOPs that predicts its ranking).

### rq03 — Noise and SNR (B)

- **Intention.** Measure how much a score moves when only the seed or the
  checkpoint changes, put design effects against that noise, and compute every
  benchmark's SNR under 22 definitions. Its per-task table feeds rq04, rq07 and
  rq09 (rq08 computes its own SNR per subset).
- **Setup.** SNR per (task, size bucket) on `predictivity` and the seed pools:
  signal = dispersion across the design variants at the size, noise = std over
  the noise window (the k/20 points in the last 20 %: 80/85/90/95/100 %). Effect-vs-noise on `predictivity_all`, where seed
  replicates exist.
- **Scripts.** `run_apertus_snr_variants.py` (the table), `effect_vs_noise.py`
  (|Δ| of each intervention over seed and checkpoint noise),
  `compare_seed_splits.py` (variant ranking on seeds 64/313 vs seed 1904).
- **Tables.** `snr_variants_per_task.csv`, `snr_variants_definitions.csv`,
  `snr_variant_coverage.csv`, `effect_vs_noise.csv`; holdout folder
  `predictivity_seeds_train__vs__predictivity_seeds_test/` with
  `headline_metrics.csv`, `per_language_agreement_da_{size,ckpt}.csv`,
  `variant_r_train_vs_test.csv`, `top_variants_train_vs_test.csv`, `summary.md`.
- **Figures.** `effect_vs_noise.png`; `snr_by_benchmark.png`, `snr_by_language.png`,
  `effect_over_seed_by_benchmark.png`, `effect_over_seed_by_language.png`
  (`panels.py`); holdout `per_language_agreement_da_{size,ckpt}.png`,
  `variant_r_train_vs_test.png`.

### rq04 — Surrogates (B, paper RQ3)

- **Intention.** Before training the reference, which statistic computed on the
  proxy alone tells us its decision will hold? First among the 22 SNR
  definitions, then against simpler candidates: signal alone, noise alone,
  early-checkpoint agreement, scaling-fit R², margin above chance; then a
  catalogue of ~50 statistics from the literature (`literature.md`).
- **Setup.** `predictivity`; reads rq03's table (SNR and DA columns), rq00's
  scores, rq01's fits, and the rq03 holdout metrics. Per language, Pearson r of
  log₁₀(SNR) vs DA; for the wider candidate list, Spearman ρ with DA-size per
  proxy size, benchmarks and BPB apart.
- **Scripts.** `analyze_snr_variants.py` (per-variant correlations and grids),
  `snr_definition_postprocess.py` (global ranking, best variant per language,
  per-language anchor benchmark, README and slide), `analyze.py` (statistics
  beyond SNR), `catalogue.py` (the literature catalogue, the AllenAI signal ×
  noise grid with the k-fold benchmark noise, and the truths: DA / Kendall τ /
  Spearman ρ for DA-size, -goal, -ckpt, both pair sets, per L), `search.py`
  (every truth × subset × surrogate and two-surrogate threshold filters,
  validated on a held-out half of the benchmarks).
- **Tables.** `snr_variant_ranking.csv`, `top_variants_overall.csv`,
  `best_variant_per_language.csv`, `best_variant_family_per_language.csv`,
  `variant_clusters.csv`, `top_benchmarks_per_language.csv`,
  `rq3_surrogates.csv`, `facts.json`.
- **Figures.** `snr_definition_by_language.png` (SNR definition × language:
  correlation with DA), `surrogates_by_benchmark.png`, `surrogates_by_language.png`
  (`panels.py`); `top_variants_overall.png`, `best_variant_per_language.png`,
  `best_variant_family_per_language.png`, `top_benchmarks_per_language.png`,
  `variant_correlation_matrix.png`, `da_size_vs_da_ckpt.png`,
  `da_size/` and `da_ckpt/<bucket>/` scatter grids and heat maps,
  `rq3_surrogates.png/.pdf` (paper).

### rq05 — Design decisions (C, paper RQ2 and RQ4)

- **Intention.** One decision at a time instead of a pooled ranking: for depth,
  language lists, temperature and the second language of L2, does a proxy
  prefer the level the reference prefers, how small can the proxy be, how early
  in its run, and is the effect larger than seed noise at all?
- **Setup.** `predictivity_all`. Reference = the largest size trained at both
  levels at that L, final checkpoint; proxy = every smaller size at 1C–5C of
  training. Populations: BPB on trained languages, on untrained languages, on
  all, benchmark tasks, macro BPB. Items the reference ties are dropped.
- **Scripts.** `analyze.py` (decision table, effect at the reference in seed
  standard deviations), `early_decision.py` (the two planned decisions, proxy
  size × Chinchilla multiple, mean over L).
- **Tables.** `intervention_da.csv`, `rq4_da_by_intervention.csv`,
  `rq4_effect_vs_seed.csv`, `facts.json`; `rq2_decisions.csv`,
  `rq2_early_small.csv`, `early_decision_facts.json`.
- **Figures.** `intervention_da_by_benchmark.png`, `intervention_da_by_language.png`,
  `intervention_da_by_benchmark_early.png`, `intervention_da_by_language_early.png`
  (`panels.py`, from `intervention_da_by_benchmark.csv` / `_by_language.csv`);
  `intervention_da.png`, `rq4_interventions.png/.pdf` (paper),
  `rq2_early_small.png/.pdf` (the paper's rq2 is rq02's `paper_ten_checkpoints.py` → `rq2.png/.svg/.csv`).

### rq06 — Language transfer (C, paper RQ5)

- **Intention.** Can what is learned on measured languages be carried to
  languages without a measurement, or never trained?
- **Setup.** `predictivity_all`; per-language BPB power laws, leave-language-out
  prediction, and rq05's decision table restricted to never-trained languages.
- **Scripts.** `analyze.py`.
- **Tables.** `rq5_transfer.csv`, `rq5_transfer_summary.csv`, `facts.json`.
- **Figures.** `rq5_transfer.png/.pdf` (paper), `bpb_curves.png`,
  `transfer_error_by_L.png` (language × rungs used, one subplot per L).

### rq07 — External frameworks (C)

- **Intention.** Check our SNR against AllenAI DataDecide on the English tasks
  both corpora evaluate.
- **Setup.** `predictivity` (and `predictivity_seeds`) against
  `allenai_snr_variants_per_task.csv`, built once from DataDecide. The headline
  variant is rq04's choice; with three shared tasks after the gate the
  correlations are reported as not comparable.
- **Scripts.** `build_allenai_variants.py`, `analyze.py`. The headline is the
  matched 1B↔1B pair; 1.7B↔1B is a row of the size sweep. The `allenai` group of
  `configs/tasks.json` lists the harness tasks to evaluate so that the shared
  set grows from 3 to 11 families.
- **Tables.** `task_overlap.csv`, `pearson_r_per_variant.csv`,
  `pearson_r_size_sweep.csv`, `top_apertus.csv`, `top_allenai.csv`,
  `agreement.csv`, `shared_task_agreement.csv`, `agreement.md`.
- **Figures.** `snr_apertus_vs_snr_allenai_<variant>.png`,
  `snr_apertus_vs_snr_allenai_grid.png`.

### rq08 — Subset selection (D)

- **Intention.** Can dropping noisy languages or subjects raise a benchmark's
  SNR, beyond what picking the best prefix on the same numbers gives for free?
- **Setup.** `predictivity`; three cases: languages of a multilingual family,
  Global-MMLU subjects pooled over languages, subjects per language. The rq00
  gate applies, MMLU category roll-ups are excluded, and each swept cell gets a
  100-draw random-subset null.
- **Scripts.** `smooth_subtasks.py`; the per-sample scripts
  (`smooth_subtasks_per_sample.py`, `compare_per_sample_methods.py`,
  `analyze_per_sample_d.py`) are cluster-only and not part of the driver.
- **Tables.** `per_benchmark.csv`, `global_mmlu_full.csv`,
  `global_mmlu_full_per_language.csv`, `summary.csv` (with `null_snr_p95`,
  `gain_over_null`).
- **Figures.** `gain_over_null.png` (every swept cell in one grid),
  `per_benchmark_plots/<family>.png`,
  `global_mmlu_full_subjects.png`, `global_mmlu_full_per_language_plots/<lang>.png`.

### rq09 — Benchmark design (D)

- **Intention.** Which properties of a benchmark go with high SNR: curation
  method, source, format, option count, passage, context and option length?
- **Setup.** `predictivity`; per-family SNR = median over the family's
  per-language tasks at the reference size (rq03's table), Kruskal–Wallis per
  feature, Spearman for the length features (`length_features.csv`).
- **Scripts.** `analyze.py`, `length_features.py`.
- **Tables.** `per_task_snr.csv`, `per_family_snr.csv`, `group_stats.csv`.
- **Figures.** `snr_family_by_language.png` (benchmark × language),
  `snr_per_family_ranked.png`, `snr_by_curation_process.png`,
  `snr_by_curation_per_task.png`, `snr_by_data_source.png`, `snr_by_format.png`,
  `snr_by_n_options.png`, `snr_by_passage.png`, `snr_vs_random_baseline.png`,
  `snr_vs_context_len.png`, `snr_vs_option_len.png`,
  `snr_vs_context_option_ratio.png`, `snr_vs_length_features.png`.

## Other files here

- `paths.py` (folder map), `utils.py` (pools, ladder-frame helpers), `grids.py`
  (the per-benchmark / per-language panel figures and the smallest-level maps),
  `autodoc.py` (README and slide block writer), `style.py` (palette).
- `report_figures/make_figures.py` — publication figures fig1–fig5 from the
  tables above.
- `ANALYSIS_new_vs_previous.md`, `PARALLEL_SESSIONS.md` — dated working notes;
  their prose keeps the numbering of their date.
