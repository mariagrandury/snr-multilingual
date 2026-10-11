# Analysis: the research questions

Which (subsets of) benchmarks give a reliable signal at each stage of
multilingual pretraining? The study extends the Signal-and-Noise framework
(Heineman et al., 2025) to multilingual models. A benchmark is useful when a
cheap measurement (a smaller model, an earlier checkpoint, or a statistic
computed on the proxy alone) leads to the decision the reference-size model
would make.

The analyses fall into six themes, plus a seventh that checks the ladder itself. Each analysis has its own folder.
A folder's `README.md` is the single document of its question (see the README rules at the end of [RULES.md](RULES.md)), and its outputs live under `<folder>/<stage>/<pool>/`.

Artifact names follow rule 16. A decision-accuracy file name states its DA kind (`da_size`, `da_ckpt`, `da_goal`, `da_all`), its pair set (`multi_axes`, `mono_axis`, `seed_null`, `both_axes`) and its reliability filter.
The convention is spelled out in [decision accuracy's README](rq02_decision_accuracy/README.md#naming).

**Snapshot.** Every hand-written number in these READMEs comes from the ladder report **2026-10-10 04:31** (acc_norm on the cloze-format originals ARC, HellaSwag, OpenBookQA, MathQA and Global-PIQA) and the outputs of its 2026-10-10 refresh on the pools of 2026-10-05 below.
The prose of every README was re-read against those tables on 2026-10-10.
The refresh reads the detrended checkpoint noise (the residual SD around a line through the 80–100 % window, with the raw SD kept as a labelled variant). It also flags the 600M depth pairs as not size-matched ([RULES.md](RULES.md#size-matching-the-600m-depth-pairs-are-flagged-not-dropped)).
Each refresh regenerates the auto blocks, so their numbers move.

## The questions

| theme | analysis (folder) | question (one sentence) | opening figure |
|---|---|---|---|
| **A. Is the evaluation predictable?** | [Above-random gate and curves](rq00_gate_and_curves/README.md) | Which benchmark-language tasks clear chance at which size, and is what the ladder cannot read a size floor or a benchmark floor? | [highlights](rq00_gate_and_curves/pretraining/predictivity/highlights.png) |
| | [Task reformulation](rq00_task_reformulation/README.md) | Do the letter-format families leave chance once the letters are dropped (`rf_`) or the items are rewritten by Gemini (`rfgm_`), and do the twins then rank designs better? | [reformulations_gate_paper](rq00_task_reformulation/reformulations_gate_paper.png) |
| | [Chance vs training tokens](rq00_chance_vs_train_tokens/README.md) | Does a task clear chance once the model has seen enough tokens of its language, whatever the size? | [pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper](rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper.png) |
| | [Scaling predictability](rq01_scaling_predictability/README.md) | Which tasks move with model size in a way a log-linear fit captures, and how well does a fit on the proxy rungs predict the reference? | [scaling_regimes_outliers_paper](rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes_outliers_paper.png) |
| | [Decision accuracy](rq02_decision_accuracy/README.md) | Does a benchmark rank the ladder's design variants at a smaller size, or at an earlier checkpoint, the way the 1.7B reference does? | [scale_convergence_da_size_multi_axes](rq02_decision_accuracy/pretraining/predictivity/scale_convergence_da_size_multi_axes.png) |
| | [DA vs training tokens](rq02_da_vs_train_tokens/README.md) | Does a language's BPB rank the design variants like the reference once the proxy has seen enough of that language? | [da_goal_multi_axes_across_langs_bpb_paper](rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_paper.png) |
| | [DA against its null](rq02_permutation_null/README.md) | How far above chance does a proxy rank the design decisions, once chance is measured cell by cell (proxy scores shuffled) rather than read off a flat 0.5? | [permutation_null_da_size_paper](rq02_permutation_null/pretraining/predictivity/permutation_null_da_size_paper.png) |
| | [Decisive pairs](rq02_decisive_pairs/README.md) | Does a proxy rank the design decisions better once the pairs the 1.7B reference cannot separate from seed noise are left out? | [decisive_pairs_da_size_paper](rq02_decisive_pairs/pretraining/predictivity/decisive_pairs_da_size_paper.png) |
| **B. Can it be measured cheaply?** | [Noise and SNR](rq03_noise_and_snr/README.md) | How much does a score move with the seed or the checkpoint alone, how large is a design effect against that noise, and what is each benchmark's SNR under 22 definitions? | [highlights](rq03_noise_and_snr/pretraining/predictivity/highlights.png) |
| | [Surrogates](rq04_surrogates/README.md) | Which statistic computed on the proxy alone (an SNR definition, its signal or noise part, early-checkpoint agreement, a scaling fit, FineTasks' criteria) predicts decision accuracy? | [highlights](rq04_surrogates/pretraining/predictivity/highlights.png) |
| **C. Does the framework generalise?** | [Design decisions](rq05_design_decisions/README.md) | For one design decision at a time (depth, data scheme A vs B and A vs C, temperature), which proxy sizes, and how early in their run, read the reference's preference at each language count? | [da_all_lines_mono_axis_paper](rq05_design_decisions/pretraining/predictivity_seeds/da_all_lines_mono_axis_paper.png) |
| | [Language transfer](rq06_language_transfer/README.md) | Does per-language scaling transfer to languages never measured or never trained, and which languages must a developer evaluate to recover the multilingual decision? | [transfer_da_all_lines_mono_axis_paper](rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis_paper.png) |
| | [External frameworks](rq07_external_frameworks/README.md) | Do our SNR values agree with AllenAI DataDecide on the English tasks both corpora evaluate? | [snr_apertus_vs_snr_allenai_paper](rq07_external_frameworks/pretraining/predictivity/snr_apertus_vs_snr_allenai_paper.png) |
| **D. Can the benchmarks be improved?** | [Subset selection](rq08_subset_selection/README.md) | Can a language, subject or item subset of a benchmark beat the full set's SNR by more than selection alone gives for free? | [gain_over_null_paper](rq08_subset_selection/pretraining/predictivity/gain_over_null_paper.png) |
| | [Benchmark design](rq09_benchmark_design/README.md) | Which design features of a benchmark (curation, source, format, option count, item length) go with a high SNR? | [snr_per_family_ranked_paper](rq09_benchmark_design/pretraining/predictivity/snr_per_family_ranked_paper.png) |
| | [Above-chance items](rq12_above_chance_items/README.md) | If every benchmark-language task keeps only the items its 1.7B runs answer above chance (chosen in sample, on purpose), how much do decision accuracy, SNR and the above-random gate rise, with the gate applied before or after the items are chosen? | [above_chance_items_snr](rq12_above_chance_items/pretraining/predictivity/above_chance_items_snr.png) |
| | [Proxy item selection](rq14_proxy_item_selection/README.md) | Can a benchmark be shortened from the proxies alone (item discrimination at 600M–1B, chosen on half the families) and still rank the held-out designs like the 1.7B reference on the full task? It needs the cluster's per-item store. | [proxy_item_selection_da_size_multi_axes_paper](rq14_proxy_item_selection/pretraining/predictivity/proxy_item_selection_da_size_multi_axes_paper.png) |
| **E. Past the reference** | [Size generalisation](rq10_size_generalisation/README.md) | Does a ranking that holds at the 1.7B reference still hold one rung above it, at 3B? It reads the six deep 3B cells scored at both rungs (scheme A at L8/L15/L30/L50, scheme B at L8/L15) and is the only reader of `above_reference=True`. | [above_reference_3B_paper](rq10_size_generalisation/pretraining/predictivity/above_reference_3B_paper.png) |
| **F. The recommendation** | [Evaluation recipe](rq11_evaluation_recipe/README.md) | Which benchmark, posed how (original, RF, LLM-RF) and scored how (accuracy, bBPB), reads the 1.7B decision from the smallest proxy (DA-size ≥ τ = 0.75, `utils.RELIABLE_DA`)? | [recipe_da_size_variants_multi_axes_paper](rq11_evaluation_recipe/pretraining/predictivity/recipe_da_size_variants_multi_axes_paper.png) |
| **G. Checks on the ladder itself** | [English only (L1)](rq13_english_only/README.md) | Do the monolingual-English cells score higher on the English benchmarks than the multilingual cells of their size, clear chance earlier, rank their design variants as reliably and with as high an SNR, and land where AllenAI DataDecide puts the same tasks? | [english_only_scores](rq13_english_only/pretraining/predictivity/english_only_scores.png) |

## Headline findings (ladder report 2026-10-10 04:31)

One bullet per analysis, in the table's order, each quoted from that folder's README (read 2026-10-10). The folder README gives the full population and the caveats.
The populations differ between analyses (pool, sizes, gate, pair set), so numbers are never compared across bullets.

- **Above-random gate.** Of the 1,696 chance-level tasks, 619 (of 1,511 scored) are above chance at 90M and 946 (of 1,667) at 1.7B, and 710 never pass (pool `predictivity`, 192 seed-1904 runs, 90M–1.7B, one-sided 95 % Wilson gate, report of 2026-10-10). The answer format drives the gate more than the option count: on trained languages at 1.7B `belebele` passes 3/59 against 57/59 for its cloze twin, and two- and four-option trained tasks pass at similar rates (83/122 against 488/759).
- **Task reformulation.** At 1.7B the `rf_` twins pass in 86 of 105 belebele, 35 of 37 Global-MMLU and 31 of 43 INCLUDE tasks, against 5, 1 and 3 originals (McNemar 82:1, 34:0 and 29:1, all p ≤ 5.8e-08). Passing does not make them rank designs better: mean DA-size on the twins is 0.489 at 90M and 0.555 at 1B, against 0.529 and 0.571 on the non-twin tasks (multi-axis pairs, 166–249 twin and 193–304 non-twin tasks). The two means cover different tasks, so they do not compare a twin with its own original.
- **Chance vs training tokens.** Outside English, the share of (task, size, L) cells above chance stays near 40 % (40–43 %) up to 10^8 tokens of the language and rises with every decade after, to 63 % at 10^10–10^11 (seed-1904 runs of `predictivity_seeds`, 90M–1.7B, all ten checkpoints, 139,848 cells from 821 tasks, report of 2026-10-10). Tokens are not sufficient: at 10^8–10^9 tokens 43 % of 90M cells are above chance against 63 % at 1.7B, and INCLUDE's original stays at 5–11 % from 10^7 tokens on while its reformulated twin reaches 99 %.
- **Scaling predictability.** Where accuracy is above chance it scales log-linearly: 1,417 gated (task, L) fits over 509 benchmark tasks reach a median R² of 0.87, while 429 of 988 accuracy and BPB tasks are at chance wherever a fit was possible (pool `predictivity_seeds`, deep scheme-A seed-1904 cells, 90M–1.7B). A power law fitted on the proxies over-predicts the 1.7B BPB in all 318 per-language fits, with a median error of 7.7 % up to 350M, 4.4 % up to 600M and 2.3 % up to 1B.
- **Decision accuracy.** On benchmark accuracy a proxy trained to the end ranks the design variants barely better than chance: DA-size is 0.51 / 0.53 / 0.53 / 0.54 / 0.57 from 90M to 1B against the 1.7B reference (pool `predictivity`, 29 variants, multi-axis pairs, 294–459 gated tasks), while per-language BPB reads 0.95 / 0.95 / 0.91 / 0.73 / 0.92 (50 tasks). Reading a run early mostly measures persistence: two seeds of one design already agree at 0.78–0.80 at 90 % of a run, against 0.82–0.86 for real design pairs (bBPB twins included).
- **DA vs training tokens.** At the final checkpoint a language's BPB reads the reference's ranking at DA-goal 0.95 at 90M and 175M, 0.91 at 350M, 0.73 at 600M and 0.92 at 1B (pool `predictivity_seeds`, multi-axis seed-1904 pairs, 50 languages). The tokens of the language seen do not set it: 90M reads 0.92 at its first tenth after 0.008 B tokens of the language, while 600M reads 0.73 at its final after 0.50 B.
- **DA against its null.** Measured cell by cell, chance sits below 0.5: a proxy with its scores shuffled reads 0.49 pooled over the benchmark tasks, so the observed 0.57–0.59 is 0.08–0.10 above it, yet only 8–10 % of the multi-axis benchmark cells beat their own null at q < 0.05 (pool `predictivity`, DA-size, 1,175–1,369 tasks with the bBPB twins). Per-language BPB reads 0.78–0.94 against a null of 0.50, with 30–64 % of its cells significant.
- **Decisive pairs.** On the pairs the 1.7B reference separates by more than one or two seed sds, benchmark DA-size rises from 0.58–0.59 on every pair to 0.64–0.66 and 0.71–0.73, but those pairs are only 52–53 % and 25 % of the comparable ones (pool `predictivity`, multi-axis pairs, 876–1,020 tasks). On them, per-language BPB reads 0.89–0.98 and 0.95–0.995.
- **Noise and SNR.** The median log10 SNR peaks at 350M (0.41, 446 tasks) and is lowest at the 1.7B reference (0.23, 588 tasks), where 75 of the 588 ungated benchmark tasks (13 %) are below the noise (pool `predictivity`, `rel_std`). On nine three-seed cells of `predictivity_seeds` the seed noise is a median 1.75x the detrended checkpoint noise, so depth's effect is 2.12x the checkpoint noise but only 1.30x the seed noise, and no design axis clears 2x the seed noise in half its cells.
- **Surrogates.** The relative-spread SNR definitions lead only as a block: `rel_std`, `rel_mpd` and `rel_dispersion` have a mean per-language r of 0.51–0.52 with DA-size and 0.55–0.56 with DA-ckpt, and `rel_std` wins by less than 0.01 (pool `predictivity`, 50 languages). In the catalogue the proxy's own rank stability beats every SNR (0.92 against DA-ckpt, 0.81 against DA-size), and FineTasks' three criteria pass 16 % (90M) to 44 % (1B) of the 846–913 accuracy tasks.
- **Design decisions.** On per-language BPB every proxy reads the temperature decision at 0.95–0.99 and the language-list decision (scheme A vs B) at 0.77–1.00 (pool `predictivity_seeds`, DA-size, mono-axis, finals against the 1.7B reference). On the gated benchmarks only three cells reach 0.75, all at L1: 350M on the depth decision and on the DCLMP swap (0.79 each, on 14 tasks) and 600M on the FWEB swap (0.75, on 133 tasks). On the items decided outside seed noise the final-checkpoint DA rises only from 0.54 to 0.60.
- **Language transfer.** The leave-one-language-out exponent predicts the 1.7B BPB from the 175M rung alone with a median error of 4.4 % on trained and 5.0 % on never-trained languages, against 7.6 % and 6.6 % for taking the 1B proxy's BPB as the reference's (deep, scheme A, seed 1904). The list decision transfers only partly: on untrained languages whose script a list trains, DA-size reaches 0.81 only at 1B, and on unseen scripts it never reaches 0.75.
- **External frameworks.** At 1B all 9 English tasks shared with DataDecide clear the gate (ARC Easy, ARC Challenge, HellaSwag, OpenBookQA, PIQA, Winogrande, BoolQ, and MMLU and CSQA through their cloze twins), and both corpora rank them alike (`rel_std`, Spearman rho 0.77, Pearson r of log10 SNR 0.77, p = 0.016), which is indicative on nine points. Our `rel_std` SNR is 1.2–4.2x lower than DataDecide's, so only the task order compares across corpora, never the SNR level.
- **Subset selection.** A chosen language subset beats the random-subset null in 94 of the 254 swept (benchmark, size) cells, while the naive comparison with the full set claims 221 of 254 (pool `predictivity`, 90M–1.7B, 48 gated benchmarks). Picking items by SNR wins SNR but loses the reference: held out at 90M–1B its DA is 0.32–0.37, against 0.53–0.57 for the full set.
- **Benchmark design.** Option count is the only family-level design axis that reaches p < 0.05: the six two-option families have a median SNR of 1.91 against 0.75 for the 18 four-option ones (H = 11.32, p = 0.003, still under 0.05 after a Bonferroni correction for the five axes, pool `predictivity`, 29 families with a 1.7B SNR, 512 tasks). The two-option families are also all non-lettered formats, so option count and format cannot be told apart. The four-option HellaSwag is second by SNR (2.29) and first by DA-size (0.75). Curation, source origin and reading passage come nowhere near (p ≥ 0.74), and neither format (p = 0.10) nor item length (p ≥ 0.14) reaches 0.05.
- **Above-chance items.** Keeping only the items the 1.7B runs answer above chance raises the median paired SNR from x1.27 at 90M to x1.50 at 1.7B, almost all of it from lower k-fold noise (pool `predictivity`, 304–489 tasks per size). On the checkpoint-noise SNR the gain is only x0.97–1.10. Decision accuracy barely moves even though the selection reads the reference (+0.041 at 90M, +0.003 to +0.023 at 175M–1B), and the gate recomputed on the kept items passes 816 of 840 tasks at 1.7B.
- **Proxy item selection.** Items chosen on the proxies alone (the top half of the candidates by discrimination at 600M–1B, about 18 % of a task's items) never beat the full task held out: their DA-size is 0.490 at 90M to 0.554 at 1B against 0.528–0.571 for the full task (pool `predictivity`, multi-axis pairs, 238–375 tasks), and they beat random items of the same count by at most +0.017.
- **Size generalisation.** One rung above the reference, benchmark accuracy ranks the six 3B families at chance from every proxy (DA-size → 3B 0.48–0.53 from 90M–1.7B, 164–303 gated tasks, bBPB twins left out), while per-language BPB ranks them at 0.81–0.96 on only 10–12 BPB tasks (pool `predictivity`). On those families the gate admits 687 tasks at 3B against 522 at 1.7B (+32 %).
- **Evaluation recipe.** No way of evaluating a benchmark reads the 1.7B accuracy decision at τ = 0.75 by 1B: the best mean DA-size is 0.63, for LLM-RF bBPB at 90M (pool `predictivity`, multi-axis pairs, 89 tasks). The rewrites help only when scored by bBPB, and on original accuracy only 4 of 27 benchmarks have half their tasks reliable at 1B (HellaSwag and LAMBADA, plus BoolQ on one task and MultiBLiMP (extra) on three).
- **English only.** The L1 cells beat every other language setting on English accuracy, but by little: deep L1 wins more than half the English accuracy tasks in all 25 (size, L) cells, by 1.1 points at 1.7B on 40 tasks, and has the lower `bpb_dclm` in all 30 comparisons (pool `predictivity`, finals, 90M–1.7B). Its DA-size is 0.58–0.63 at 90M–1B, against 0.45–0.53 for L2/L15/L30, but the decisions differ by L: on the depth pair alone, the one every L shares, L1 reads 0.51 against 0.44–0.47.

## The paper's figures

`documents/paper/figures/make_rq_figures.py` copies them from the analysis, never the reverse.
Main text: `rq0` ← chance vs training tokens' INCLUDE-against-twin panel (`pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical`), `rq1` ← scaling predictability's `scaling_regimes_outliers_paper`, `rq2` ← decision accuracy's `rq2_da_all_above_66_either_transformation_mono_axis` (`paper_rq2.py`).

The appendix (`app_*`) takes the other `_paper` opening figures of the table above, plus the gate's `first_size_share_paper`, `first_size_above_random_paper`, `benchmark_curves_paper`, `above_random_external_paper` and its external-models LaTeX table.
The four report figures (`report_figures/make_figures.py`) are the 36-sweep's.

## Pools

There is one input, the published ladder report (`ladder_report.csv`, loaded by `snr/download/ladder.py`). The loader drops diverged and unfinished runs and keeps the checkpoints on the shared k/10 and k/20 grid.
The models are the predictivity ladder: sizes 90M–1.7B × L ∈ {1, 2, 8, 15, 30, 50} × ladder (deep, shallow, swiglu) × data build (A, AT3, B, ZH, ES, DCLMP, FWEB) × seeds.

A "design variant" is one such cell. Its cross-size identity (`family`: everything but the size) is the unit a decision pairs.
The analysis reads a cell on its design axes, never on its build or ladder label. The axes are L, arch, activation, optimizer, seed and ONE data-scheme axis, scheme A/B/C × temperature T (AT3 is A at T 3, ZH and DCLMP are B, ES and FWEB are C, see [RULES.md](RULES.md), Definitions).

The four pools are defined in `configs/models.json`. Since 2026-10-05 every pool is gated with `predictivity`'s mask (before that date the headline pool held schemes A/B only, plan/decision_accuracy.md §9).

- **`predictivity`**: seed 1904, every cell of every ladder and data build (swiglu included), 90M–1.7B. The above-random gate is computed on it, and it is the headline pool of the gate, decision accuracy, noise and SNR, surrogates, external frameworks, subset selection, benchmark design, above-chance items, size generalisation, the evaluation recipe and the English-only check.
- **`predictivity_seeds`**: every seed, every cell. It carries the seed-noise columns and the reads that take every run: the gate's curves, scaling predictability, the two training-token analyses (their seed-1904 runs), effect against noise, design decisions and language transfer.
- **`predictivity_seeds_train` / `_test`**: the seed holdout. It compares the replicate seeds (64, 313, 28, 1797) with seed 1904 on exactly the same cells.

The 36-model sweep and the public models (`seeds_*`, `custom_swissai_hf`, `external`) are another period of the project, with a different harness, task set and reference size (1B).
They are never pooled with the ladder: each folder keeps them in its final section "Extensions from other sweeps".

## Rules

Every number follows [RULES.md](RULES.md). `check_rules.py` verifies the
tables at the end of the driver, and the review skill checks them before a
commit. The rules are:

- the above-random gate: a Wilson 95 % lower bound over the task's items, at
  the proxy and at the reference. Gated cells are grey, never dropped.
- trained languages only (language transfer opts out).
- the ten evaluated tenths as the checkpoint axis, in Chinchilla multiples 1C–5C.
- one noise window (the k/20 points in the last 20 % of a run) and one noise
  (since 2026-10-08, the residual SD around a line through that window).
- at least three design-variant pairs per decision cell.
- parent tasks only (subset selection opts out).
- `multi` is not a language.
- three tasks per language for a per-language correlation.
- one reference: 1.7B at every L, L2 ZH and ES included since 2026-09-26.
- sizes 90M–1.7B, each rung at its own batch (the 3B rung only for the
  size-generalisation question).
- no leakage from the reference into a proxy statistic.
- a CSV beside every PNG.
- since 2026-09-22, an `axes` column that names a decision table's pair set
  (`multi-axis`, `mono-axis`, `seed`).

The reformulated twins are ordinary benchmarks in every population since
2026-09-22, and the twenty promoted probe families since 2026-09-23. A table
regenerated after those dates is therefore not comparable with one
regenerated before.

## Flow

```
rq00 gate ──► rq02 DA per task ──► rq03 SNR table (22 definitions + DA) ──► rq04 variant ranking, surrogates, FineTasks
   │                 │                      │                                  ├─► rq07 DataDecide agreement
   │                 │                      └─► rq03 seed holdout ──► rq04      ├─► rq08 subset sweeps (own SNR, rq00 gate)
   │                 └─► rq02 extensions (by_L, scale_convergence, …)          └─► rq09 design features
   └─► rq01 fits ──► rq02 scaling_vs_ranking ──► rq04 surrogates (fit R² as a candidate)
rq05 decision table ──► rq05 early decision, rq03 effect_vs_noise
rq05 intervention_da ──► rq06 decision transfer (recomputed there on every language's BPB, rule 2's exception)
```

## How to regenerate

**Everything** runs from `src/signal-and-noise` with the `snr` env and takes
about two hours. From the repo root, `scripts/refresh_analysis.sh` runs it too,
and also fetches the report and rebuilds the documents:

```bash
FORCE=1 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 bash run_all_predictivity.sh
```

`run_all_predictivity.sh` runs the passes in research-question order, which
is also the dependency order: first the bBPB twins' table
(`build_per_item_store.py --bench-bpb` rewrites `rq08_subset_selection/bench_bpb.csv`
from the per-item store before any loader reads it) → rq00 (`above_random.py`, the gate every later
step reads, then `run_apertus.py`, `curves.py`, `panels.py`, then the twin comparison
`reformulations_gate.py` and `above_random_external.py`) → rq01 (`analyze.py`,
`panels.py`, `regimes.py`, `regimes_survivorship.py`, `scaling_law_error.py`,
`tokens_seen.py`) → rq02 (`compute_da.py` per pool, `bench_bpb_da.py`,
`da_per_benchmark.py`, `early_small.py`, `reliable_tasks.py`, `by_L.py`,
`cross_task.py`, `scale_convergence.py`, `paper_rq2.py`, the `--axes mono-axis` twins, then `scale_convergence.py --by L
--langs L8 [--common-tasks]`, `by_language.py`, `agreement.py`,
`seed_uncertainty.py`, `scaling_vs_ranking.py`, `public_ladders.py`,
`language_tier.py`, `pair_axes.py`, `crossfit_reliable.py`, `fixed_tasks.py`, then the null and decisive-pair
readings `rq02_permutation_null/permutation_null.py` and `rq02_decisive_pairs/decisive_pairs.py`) → rq03 (`run_apertus_snr_variants.py` per
pool, `compare_seed_splits.py`) → rq04 (`analyze_snr_variants.py`,
`snr_definition_postprocess.py`, `analyze.py`, `finetasks_criteria.py`,
`panels.py`, `catalogue.py` + `search.py`) → rq05 (`analyze.py`, `early_decision.py`, `transformations.py`,
`panels.py`, then rq03's `effect_vs_noise.py`, which reads rq05's table, and
rq03's `panels.py`, which draws from it) →
rq06 (`analyze.py`, `panels.py`, `language_panel.py`) → rq07 (reads rq04's
ranking) → the English-only check (`english_only.py`, after the AllenAI table it reads) → rq08
(`smooth_subtasks.py`, `panels.py`, `per_item_ladder.py`, `reference_solved.py`) → the
above-chance items (`above_chance_items.py`, after the per-item store steps) → the proxy item selection
(`rq14_proxy_item_selection/proxy_item_selection.py`, also store-only) → rq09 → rq10
(`above_reference.py`, the 3B rung, and `gate_crossover.py`) → the evaluation recipe (`recipe.py`, which
reads decision accuracy's per-task early-small table) →
`report_figures/make_figures.py` → `check_rules.py`.
A cached table is rebuilt when the ladder report or `bench_bpb.csv` is newer
than it (the first pass rewrites `bench_bpb.csv` only when its content changes).
`FORCE=1` is needed whenever the report was regenerated since the last run,
because a cached table carries the report's commit time and looks newer.
`CURVES=1` also redraws rq00's ~150 accuracy-vs-FLOPs grids. They took ~11 h on 2026-10-10, so the nightly draws them in two jobs (`run_apertus.py --only benchmark|language`).
The loader keeps its melted frame as a pickle under `src/signal-and-noise/.cache/` (git-ignored), keyed by a hash of the report and the loader's inputs. `SNR_CACHE=0` bypasses it and `SNR_CACHE_DIR` moves it.
`compute_da.py` builds one score cube per pool and takes `--workers N` or `COMPUTE_DA_WORKERS` (default 1, output identical).

**One figure** can be redrawn without the driver, from `src/signal-and-noise`.
This is the normal way to iterate. The thread caps are mandatory on the login
node: without them OpenBLAS spawns one thread per core and the 1000-pid slice
kills the process.

```bash
export PATH=/users/mariagrandury/miniconda3/envs/snr/bin:$PATH
PYTHONPATH=$PWD:$PWD/../../src OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 SOURCE_DATE_EPOCH=0 \
  python analysis/rqNN_*/<script>.py --pool predictivity
```

Scripts that read `da_all_reliable_tasks_both_axes.csv` (everything with an `above_*`
variant) need `reliable_tasks.py` to have run on the current DA tables first,
and a table regenerated by hand needs the panels that read it regenerated at
the same time (bug #16 in `../CLAUDE.md`). The canonical pool
(`autodoc.CANONICAL_POOL = predictivity`) is the one whose README auto blocks
the scripts write. On other pools the generators do nothing.

## Other files here

- `paths.py` (the one map from constant to folder), `utils.py` (pools, rules,
  ladder-frame helpers), `grids.py` (the per-benchmark / per-language panels
  and the smallest-level maps), `autodoc.py` (README block writer),
  `style.py` (the palette), `check_rules.py` (rule 14).
- `report_figures/make_figures.py`: the 36-sweep's report figures fig1–fig4.
- `rq00_task_reformulation/` also holds the reformulation pipeline's notes
  and pilot files (the Gemini rewrite, the probe), and `rq08_subset_selection/per_sample/`
  holds the 36-sweep's cluster-only per-item outputs.
