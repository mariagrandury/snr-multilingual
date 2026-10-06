# Analysis-wide rules

Every number an `rqNN_*` script writes, plots or puts in a README follows the
rules below. They exist because the September 2026 review found each one
broken somewhere, silently. A script that cannot follow a rule says so in its
output and its README, in a sentence a reader cannot miss; it never quietly
does something else. `check_rules.py` (this folder) verifies what can be
verified from the output tables and is run by `run_all_predictivity.sh` and by
the review skill before any commit.

The constants live in one place, `configs/models.json` → `snr`, and reach the
code through `analysis/utils.py`. The helpers named below are the only
implementation of each rule: use them, do not re-derive.

## Definitions

A **cell** is one trained run, named by the launcher: (size, L, ladder, data
build, seed), e.g. `lm-600M-L2-ZH-deep-seed1904`. Its **family** is the cell
without the size. The analysis never reads the ladder or the build label as a
level: it reads a family on its **design axes**, L, arch, activation,
optimizer, scheme, T and seed (`utils.DESIGN_AXES`), through two registry
mappings the loader applies (`snr/download/ladder.py`; the frame keeps
`ladder` and `data` beside the levels).

| ladder (`launch_trainings.LADDERS`) | arch | activation | optimizer |
|---|---|---|---|
| deep | deep | xielu | ademamix |
| shallow | shallow | xielu | ademamix |
| swiglu | deep | swiglu | ademamix |

| data build (`DATA_SCHEMES`: `letter`, `temp`) | L | scheme | T |
|---|---|---|---|
| A | 1, 2, 8, 15, 30, 50 | A | 1 |
| AT3 | 15, 30, 50 | A | 3 |
| B (diversity-first lists) | 8, 15, 30 | B | 1 |
| ZH (Chinese second language) | 2 | B | 1 |
| DCLMP (DCLM without the edu filter) | 1 | B | 1 |
| ES (Spanish second language) | 2 | C | 1 |
| FWEB (FineWeb) | 1 | C | 1 |

The scheme is the data recipe **at that L**: within one L a (scheme, T) names
one build (`utils.data_build`), but across L a letter does not (B is DCLMP at
L1, ZH at L2, the diversity-first lists at L8–L30), so a pair moves `scheme` when the letters differ or when it
reads two builds at one T (`utils.moved_axes`); A at L8 and A at L15 differ on
L alone. Anything a build decides by its label — its language list, its token
shares, which tasks it trains (rule 2) — reads `data`, never the letter, and a
baseline "the A cell" is `data == "A"` (scheme A alone also matches AT3).

The four **pools** (`configs/models.json`, resolved by `utils.build_snr_pool`):

| pool | filter | purpose |
|---|---|---|
| `predictivity` | seed 1904 | the grid seed, every design (every ladder and build): the headline SNR and DA pool (rq00 gate, rq02–rq04, rq07–rq09), and the one the gate (rule 1) is computed on |
| `predictivity_seeds` | none | every seed: the seed-noise columns, and the reads that take every run |
| `predictivity_seeds_train` | the replicate seeds 64, 313, 28, 1797 | the seed holdout's train split |
| `predictivity_seeds_test` | seed 1904 on exactly the train split's cells (`cells_of`) | the matched holdout's test split |

Every pool is gated with `predictivity`'s mask. Without AT3, ZH, ES, DCLMP and
FWEB, L1, L2 and T = 3 do not reach rule 5's three pairs, which is why the
headline pool carries every build.

## The rules

| # | rule | constant / helper |
|---|---|---|
| 1 | **The above-random gate.** A benchmark's number at a size counts only where the task is above chance at that size: the Wilson 95 % lower bound of a run's accuracy clears the chance level for at least `MIN_SHARE` (half) of the size's runs. A quantity that ranks against the reference (DA-size, a surrogate of it) also needs the task above chance at the reference. A cross-size fit uses only sizes where the task is above chance. Gated cells are drawn grey and kept in the CSV, never dropped from a grid. Tasks without a chance level (BPB, the loss, generative tasks) are never gated: a mask of NA passes. The chance level is uniform guessing: 1/n_options, or E[1/n_i] over the items when the count varies (TruthfulQA mc1, 0.2253), or the mean true-option share for `truthfulqa_mc2`, whose score is probability mass rather than a pick-one accuracy (0.449; `above_random.CHANCE`, read through `task_chance`). It does not test a constant answer, which scores the majority gold label's share (hellaswag_ta: 0.2585 against 0.25), and the runs of a size answer the same items, so their verdicts are correlated, not independent trials. | `rq00 above_random.load_mask`, `utils.passes_gate`, `grids.mark_gated` |
| 2 | **Trained languages only.** A model's score on a language its mixture does not train is a transfer measurement. It is used in rq06 (language transfer) and nowhere else, not even pooled into a mean. `bpb_macro` and `train_loss` are measurements of the whole mixture and always count. | enforced in `utils.build_snr_pool`; opt out with `untrained=True` (rq06, and the gate, which must cover every task) |
| 3 | **Ten checkpoints.** Every quantity read along a run uses the ten evaluated tenths (10 %, 20 %, …, 100 %), the grid every size shares; `da_early_fracs` is the nine before the final. BPB is also scored on the twentieths; those rows are left out wherever BPB and benchmarks are compared or a checkpoint axis is drawn. Progress is labelled in Chinchilla multiples (1C–5C), never as a share of the run. | `utils.CKPT_DA_EARLY_FRACS`, `utils.SHARED_FRACS`, `utils.on_shared_grid`, `grids.chinchilla` |
| 4 | **One noise window.** Checkpoint noise is the standard deviation over the `k`/`NOISE_GRID` (k/20) points in the last `NOISE_WINDOW` (20 %) of a run, the WSD decay, and the same window and grid for every kind of measurement: 80, 85, 90, 95 and 100 %, five points. BPB has always been scored there; `launch_trainings.due_iters` adds the benchmark evals at 85 % and 95 %, which the size grids alone miss (a 20-save size lands on the tenths, a 60-save size on the thirtieths). A run whose two new evals have not landed contributes three points, which is a noisier estimate of the same quantity, not a second definition. | `utils.NOISE_WINDOW`, `utils.NOISE_GRID`, `utils.noise_checkpoints`, `launch_trainings.due_iters`, `ladder._on_shared_grid` |
| 5 | **Three pairs.** A decision-accuracy cell needs at least `MIN_PAIRS` (3) design-variant pairs; below that it is NaN and its pair count is still written next to it. The rq02 kernels enforce this and print how many cells it emptied; any other DA computation applies the same constant and reports the same way. A cell's pair count is `k(k-1)/2` over its `k` families, so it is triangular — 0, 1, 3, 6, 10, … and never 2. `MIN_PAIRS` = 3 therefore means "three families"; setting it to 2 changes nothing, and the only value that admits a two-family cell is 1, where DA can only be 0 or 1. | `utils.MIN_PAIRS`, `compute_da` kernels, `da_all_n_pairs_per_task_both_axes.csv` |
| 6 | **One task per benchmark and language.** A benchmark with sub-benchmarks (MMLU subjects, INCLUDE domains) is read as its per-language parent only. Sub-benchmarks are used in rq08 (subset selection) and nowhere else. | enforced in `utils.build_snr_pool`; opt out with `facets=True` (rq08 only); `utils.parents_only`, `utils._is_parent_task` |
| 7 | **`multi` is not a language.** The cross-language aggregates (`bpb_macro`, `train_loss`, `include_base_44`) and unresolved tasks (`??`) are never a row of a per-language table, never one of "N languages", never a language in a correlation. | `utils.languages_only`, `utils.LANGUAGE_AGGREGATES` |
| 8 | **Three tasks per language.** A per-language correlation (a Pearson or Spearman r over the language's tasks) needs at least `MIN_LANG_TASKS` (3) distinct tasks with a value; fewer is NaN, not a point in a mean. This is a stability floor, not a significance one — see [Why three tasks per language](#why-three-tasks-per-language) below. A per-language r is descriptive and is labelled as such; the significance claim belongs to the r pooled over languages. | `utils.MIN_LANG_TASKS` |
| 9 | **One reference.** The reference is `TARGET_SIZE` (1.7B) for every question. L2 ZH (2026-09-20) and L2 ES (2026-09-26, its 23.9B Spanish source repeated 3.5×) run to 1.7B like scheme A, the three L2 families that let rule 5 accept L2 at the reference; no setting stops below it any more. Where the 1.7B cell of a setting has no result yet, the cell is empty, never filled from a smaller size. Any comparison of the L2 schemes states the epoch count — see [Second-language repetition at L = 2](#second-language-repetition-at-l--2). | `utils.TARGET_SIZE`; `pretrain.launch_trainings.scheme_sizes` |
| 10 | **Sizes 90M–1.7B, at each rung's own batch.** Every analysis reads `ANALYSIS_SIZES`, the ladder from 90M up to the reference. The 90M and 175M rungs were retrained at their own batch (84 / 168) on 2026-09-23; the diverged batch-504 runs they replace stay on disk and in older reports, and the loader keeps only the runs at the batch a rung uses now (`ladder_report.on_grid`, applied in `snr/download/ladder.py`), so an old run never appears: not as a value, an empty column, an axis tick or a README column. The 3B rung sits ABOVE the reference and is dropped the same way: it exists for the size-generalization question, which is its own RQ and opts in with `above_reference=True`. Nothing else reads a size above the reference, because a table whose reference is 1.7B cannot carry a column the reference does not cover. `ANALYSIS_SIZES` is derived from `TARGET_SIZE`, so moving the reference moves the ladder with it. | `utils.ANALYSIS_SIZES`, applied in `utils.build_snr_pool`; `above_reference=True` for the size-generalization RQ alone (`rq10_size_generalisation/above_reference.py`, exempt in `check_rules.EXEMPT`) |
| 11 | **No leakage.** A statistic presented as available at a proxy size, or before the reference is trained, is computed from that proxy's data alone. A fit over sizes that includes the reference is a reference-size quantity and is labelled as one. | rq04 |
| 12 | **Figures.** A CSV of the same name next to every PNG; white = no value, grey = gated; a line under the title saying how a cell is computed; the population (tasks, pairs, languages) stated wherever a mean is shown. | `grids`, `style.save_figure` |
| 13 | **Populations move; say so.** When the set of tasks or pairs behind a cell differs across a row (the gate keeps different tasks at different sizes), the figure or table carries the count, and the README says the populations differ. | `grids` count overlays |
| 14 | **Outputs follow the code.** After a change to the loader, `configs/models.json` → `snr`, or a helper above, the pipeline is re-run before any table is read or cited; `check_rules.py` is the test that the tables on disk obey the rules. | `run_all_predictivity.sh`, `check_rules.py` |
| 15 | **Say which pairs a decision accuracy is over.** A DA table carries an `axes` column naming its pair set: `multi-axis` (every pair of design variants at the pool's seed — the convention to 2026-09-22, in which two thirds of the pairs move more than one axis at once), `mono-axis` (the pairs moving exactly one of the design axes L, arch, activation, optimizer, scheme, T — the seed held — the decision a practitioner makes, and what upstream's "every pair" is by construction) and `seed` (the null: two draws of ONE design, emitted only where the pool has replicate seeds). The build label is never an axis; scheme (A/B/C, the recipe at that L) and T are (Definitions above; `DATA_SCHEMES` `letter` and `temp` are the source of truth), so A vs AT3 moves T, A vs ZH moves the scheme and B vs AT3 moves two axes. Nor is a cell's ladder (the `deep`/`shallow`/`swiglu` token in its name): the loader's frame carries its levels as the `arch` (depth: deep|shallow), `activation` and `optimizer` columns (`launch_trainings.LADDERS`), and `design_axes` reads them, so a (deep, swiglu) pair moves `activation` and a (shallow, swiglu) pair moves two axes and is no mono-axis pair. A consumer that does not ask reads `multi-axis`, so a table written before this rule needs no migration; a figure drawn over one pair set is filtered by a reliability computed on the same one (one exception: `rq02/pair_axes.py` filters both of its rows by the multi-axis reliability on purpose, so the pair set is the only thing that differs between them), and its twin sits beside it under `AXES_SUFFIX`. | `utils.DESIGN_AXES`, `utils.design_axes`, `utils.moved_axes`, `utils.pair_sets`, `utils.pair_agreement`, `utils.one_axes`, `utils.AXES_SUFFIX` |
| 16 | **A name says what the artifact is.** Figure and table names are descriptive and coherent within and across RQs: the same quantity carries the same token in every folder, in one order — `<subject>_da_<kind>[_<breakdown>][_<filter>]_<pair set>[_<view>]` (`scale_convergence_da_size_L_above_66_both_mono_axis_flops`, `early_small_da_goal_by_L_above_80_multi_axes`). A decision-accuracy artifact always names which DA it holds (`da_size`, `da_ckpt`, `da_goal`; `da_all` when one file holds more than one, `da_size_vs_da_ckpt` when it sets two against each other), which pair set (`AXES_SUFFIX`: `_mono_axis`, `_multi_axes`, `_seed_null`; `_both_axes` when an `axes` column carries every pair set, `_mono_vs_multi_axes` when the figure compares them) and, where a reliability filter applies, which one (`above_66_size`, `above_66_ckpt`, `above_66_either`, `above_66_both`, `above_80`). A facet pair shares one table: `<name>_by_benchmark_<pair set>.png` and `<name>_by_language_<pair set>.png` read `<name>_<pair set>.csv` (`grids._csv_path`). A rename changes the writer, every reader and the files (`git mv`) in one change. | `utils.AXES_SUFFIX`, `grids._csv_path`, `check_rules.py --names` (lists the names that break the rule) |
| 17 | **Nothing is generated outside the regeneration, and nothing orphaned stays unflagged.** Every figure, table, README auto block and paper block is written by a script that `run_all_predictivity.sh` or `scripts/refresh_analysis.sh` runs, so none can diverge from the code or from the report; a generated block names its generator in its marker (`<!-- BEGIN auto:KEY (script) -->`, `% BEGIN generated: KEY (script)`, `% Generated by script`). An orphan — an artifact no generator writes any more (the old twin of a renamed file, the output of a script dropped from the driver) — is flagged, never left to look current: the refresh lists every artifact a `FORCE=1` run did not write, and a rename lists the files it leaves behind with the command that removes them (deleting is the user's call). The slides are outside this rule. | `check_rules.py` (generators), `scripts/refresh_analysis.sh` (the `ORPHAN` lines) |
| 18 | **Paper figures are bare.** Every `_paper` figure (the files `documents/paper/figures/make_rq_figures.py` copies) has no figure title and no description: the caption is the paper's. Its axis labels are capitalized; no label, panel title or legend entry uses a dash as punctuation (`—`, `–`, ` - `; a hyphen inside a word such as Global-MMLU is fine) or a `;`; and legend columns should hold the same number of entries where possible (choose `ncol` to divide the entry count). `style.save_paper` refuses, with a `RULE 18` error, to write a figure with a title, a description, an uncapitalized axis label, a dash or a `;`, and only warns about unequal legend columns (an odd entry count cannot always be split evenly), and `check_rules.py` flags a script that writes a `_paper` figure without it (`EXEMPT[18]` lists the generators not yet converted). | `style.save_paper`, `style.paper_problems`, `check_rules.py` |
| 19 | **Double-blind anonymity.** Until the reviews are out, the website, the docs and the paper carry NO link to any of our Hugging Face organisations or profiles, W&B, GitHub, or any personal or professional site, and no name, handle or affiliation that identifies us: one such link is grounds for automatic rejection. Where such a link would go, write the sentence `REMOVED_LINK` holds ("Link momentarily removed for double blind review"); its wording lives in that one constant. The analysis READMEs' GitHub links (README rule 6) are internal and must not reach a published page. | `anonymity.REMOVED_LINK` (repo root) |

## The reformulated twins are in the populations

Since 2026-09-22 the `rf_*` twins of the three letter-format families, and
`rfgm_include_base_44`, are ordinary benchmarks in the `auto` group: they are
gated, plotted and counted like any other. Two consequences a reader has to
be told (rule 13):

- **Benchmark populations grew.** Any unqualified "over the benchmarks" mean
  — rq02's per-benchmark DA, rq03's SNR table, rq04's per-language r, rq09's
  family aggregate (the twins with a `FAMILY_META` entry; rq09's README lists the
  families it leaves out) — now includes the twins. Figures carry their task counts,
  as rule 12 requires, and a twin reads as `belebele-rf`.
- **A twin is not an independent task.** It is the same items in another
  formulation, so a language can clear rule 8's three-task floor with twins of
  one benchmark. That floor is a stability floor, not a significance one, and
  per-language correlations stay descriptive — but it is a weaker three than
  three unrelated benchmarks, and a per-language panel that rests on twins
  alone should say so.

This matters because the originals barely survive the gate: at 1.7B the gate
keeps 0 of 37 Global-MMLU tasks and 11 of 105 belebele tasks, against 35 and
86 of their `rf_` twins, and 4 of 43 INCLUDE tasks against 31 of the `rf_`
twins and 33 of the Gemini-rewritten ones. Before the twins entered the pool
those families contributed almost nothing to any RQ.

## The benchmark-BPB twins are in the populations too

Since 2026-10-04 every benchmark task with a gold-answer log-likelihood has a
third member, `bbpb_<task>`: the item-mean bits per byte of the gold answer
(Heineman et al. 2025's `correct_bpb`), read from the lm-eval samples by the
rq08 per-item store (`build_per_item_store.py --bench-bpb` writes
`rq08_subset_selection/bench_bpb.csv`) and added by the loader
(`utils.with_bbpb_twins`) as a copy of the original's row. So it inherits that
row's rules 2, 6 and 10, reads as `belebele-bbpb` (paper: "Belebele bBPB"), and
an `rf_` twin has one as well (`belebele-rf-bbpb`). What a reader has to be told:

- **Lower is better.** Every score oriented by direction goes through
  `utils.lower_is_better` (per-language BPB, the benchmark BPB, the loss);
  DA and the dispersion SNRs do not care.
- **No chance level, so no gate.** Like per-language BPB, the twin has a mask
  of NA and passes rule 1 everywhere, including where its original is at chance.
- **The store holds finals only.** It was built with `--finals-only` on
  `predictivity` (seed 1904), so a twin exists at each cell's final
  checkpoint and nowhere else: it enters DA-size and every final-checkpoint
  read, and is empty in DA-ckpt, DA-goal before 100 %, the checkpoint noise
  and the seed replicates until the store is rebuilt over every checkpoint.
- **Same items again.** Like an `rf_` twin it is not an independent task
  (rule 8's caveat above); the twin cluster of rq04 holds the original, its
  `rf_` twins and all their `bbpb_` twins (`grids.base`).
- `reformulations_gate.py` leaves it out: its question is the accuracy
  formulations, and the twin would count as an "original".
- **Every benchmark row carries its variant.** The loader writes `format`
  (original / rf / rfgm) and `scoring` (acc / bbpb) on every benchmark row
  (`utils.variant`, `utils.with_variant_columns`), so a table splits by variant
  with a groupby. A pooled "all benchmarks" number pools every variant;
  rq11 (`rq11_evaluation_recipe`) gives every finding per variant and pooled,
  and the head-to-head on the paired cells.
- **One reliability cut, τ = 0.75** (`utils.RELIABLE_DA`): rq02's safe sizes
  and rq11's recommendation read it from there, and every figure, table and
  README block that uses it states it.

## The probe candidates are not in the populations

`groups.auto_probe` (2026-09-23) holds benchmarks being screened — BBH and
ACP-Bench as cloze arms, mmlu, INCLUDE v2, and the `rf_` twins of the ones
that ask for a letter — at the last checkpoint of the 600M–1.7B cells only.
A screened benchmark is outside `auto`, so rule 2's trained set does not know
it: every pool built without `untrained=True` drops it, and no RQ's population
moves while it is only a candidate. The one pass that reads them,
`rq00_task_reformulation/probe.sh`, opts in with `SNR_TRAINED_GROUPS=auto,auto_probe`
(`pretrain.ladder_report._trained_tasks`), which only widens the trained set
for that process; the gate it rewrites differs from the committed one in the
probe rows alone.

**Promoted 2026-09-23**, and this moves every population: twenty of the
twenty-one candidates are now in `auto` — the BBH / ACP-Bench cloze arms and
their `rf_` twins, mmlu + rf_mmlu, commonsense_qa + rf_commonsense_qa,
cultural_bench easy/hard + rf_cultural_bench_easy, INCLUDE v2 (OG and EN),
blend_sample, mathqa, openbookqa, toxigen and truthfulqa_mc2. Every
unqualified "benchmark" mean is over a wider set from the first checkpoint
they land on, so a table regenerated after that point is not comparable with
one regenerated before it; say which side of the promotion a number comes
from. `bbq` was not promoted (23 min per checkpoint, a third of the top-up
bill, and it clears a 1/12 chance trivially) and left the candidates on
2026-10-01.

## Why three tasks per language

`MIN_LANG_TASKS` trades coverage against stability, and it is worth being
explicit that it never buys significance. In rq04 it is the binding
constraint — `ar` has 22 tasks in the pool and still went NaN at 5 — because
a task counts only when it has both an SNR and a decision-accuracy value.
Measured when the threshold was lowered (2026-09-20 tables):

| `MIN_LANG_TASKS` | languages with an rq04 correlation |
| ---: | ---: |
| 5 (until 2026-09-20) | 6 |
| 4 | 10 |
| **3 (now)** | **17** |
| 2 | 26 |

On the 2026-09-30 tables, with 90M in the ladder and the twins in the pool,
three tasks give 34 languages under both DA kinds (`snr_definition.csv`).

Against that, the |r| a Pearson correlation must exceed to reach p < 0.05:

| n | 3 | 4 | 5 | 6 | 10 | 50 | 100 | 300 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| critical \|r\| | 0.997 | 0.950 | 0.878 | 0.811 | 0.632 | 0.279 | 0.197 | 0.113 |

When the threshold was lowered the best per-language r was 0.24, so no
per-language cell was significant at 5 and none would have been at 3: the
threshold was not doing inferential work, and holding it at 5 cost 11 of 17
languages. On the 2026-09-30 tables single cells reach r = 0.72 (Korean,
DA-size, `dist_std`) and 0.63 (Persian, DA-ckpt), but each is the maximum
over 22 variants in one language, read on a few tasks pooled over the proxy
sizes, so it is a selected value and not a test. Per-language panels carry
their n and say they are descriptive. The pooled correlation is the one to
quote: over every language, `rel_star_discrepancy` reaches r = 0.215 against
DA-size (mean over the proxy sizes) and 0.122 against DA-ckpt
(`snr_definition.csv`, scope `all`). Its points are not independent (one
task appears at several sizes), so a nominal p is optimistic; rq04's
catalogue search (`search.py`) is the clustered version, one point per
(benchmark, language).

## Second-language repetition at L = 2

Rule 9 asks for the epoch count at L2 because the L2 builds are capped by the SOURCE,
not the budget: the swiss-ai filtered subset holds 71.8B tokens of Russian
(scheme A), 59.9B of Chinese and 23.4B of Spanish, against the 50 × N tokens
of second language a run draws (the exact budget, `models.json` `stages.pretraining.tokens` / 2, not 50 x the label size). What a cell repeats is set by the BUILD it
reads, which is smaller still — A 72.8B, ZH 52.0B, ES 23.9B — and every rung
of a scheme reads the same build file, so these are the epoch counts the
trained models actually have:

| size | draw | Russian (A) | Chinese (ZH) | Spanish (ES) |
| --- | ---: | ---: | ---: | ---: |
| 175M | 8.8B | 0.12 | 0.17 | 0.37 |
| 350M | 17.2B | 0.24 | 0.33 | 0.72 |
| 600M | 29.7B | 0.41 | 0.57 | 1.25 |
| 1B | 47.2B | 0.65 | 0.91 | 1.98 |
| 1.7B | 83.6B | 1.15 | **1.61** | 3.51 |

ZH's 52.0B build is smaller than the 59.9B Chinese there is: it was sized
when ZH stopped at 1B. The 1.7B cell trains on it anyway
(`launch_trainings.py --allow-undersized lm-1.7B-L2-ZH-deep-seed1904`,
2026-09-20 — the flag names the one cell it applies to) rather than on a
59.9B rebuild, because every other ZH rung reads this file and moving only
the top rung would put it on data the rest of its own ladder never saw —
1.61 epochs instead of 1.40, and one consistent ZH ladder.

Repetition up to ~4 epochs costs little (Muennighoff et al., 2023), so none
of these is bad training, and the scheme-A baseline itself repeats at 1.7B
(1.15). What matters for decision accuracy is that the repetition be
**flat across the ladder**, because DA compares a proxy rung against the
reference rung: Russian and Chinese stay under 1.7 everywhere, so a rank flip
between rungs is about the benchmark. Spanish goes from 0.73 at 350M to 3.51
at 1.7B, so a flip could be the data repeating instead. ES was capped at 1B
for that reason until 2026-09-26; it now reaches 1.7B (rule 9), since 3.51
epochs stays under the ~4 where repeated tokens are still worth close to
fresh ones. Any table that compares the L2 schemes states the epoch count.

ZH and ES are **schemes B and C at L = 2** (Definitions): A trains English
plus Russian, ZH plus Chinese, ES plus Spanish, and all three exist in the
deep architecture only. They are three levels of the scheme axis at that L —
the L2 analogue of A vs B at higher L — and that is what makes them the three
families rule 5 counts at L2.

## Where a rule cannot be followed

Say it in three places: the script's output (a line starting with `!!! RULE n:`),
the README block the script generates, and the figure's caption line. An
analysis that needs an exception (rq06 for rule 2, rq08 for rule 6) passes the
opt-out explicitly and its docstring names the rule it opts out of.

## What the review checks

The review skill reads this file, runs `python analysis/check_rules.py`, and
treats every rule above as a review criterion for the changed scripts: a new
per-language table without `languages_only`, a new DA computation without
`MIN_PAIRS`, a new checkpoint axis on the twentieths, a new mean over an
unstated population, a table that reads `da_all_per_task_both_axes.csv` without `one_axes`,
a DA figure whose name does not say which DA and which pairs (rule 16), a
new output or generated block that no driver step writes (rule 17), a
`_paper` figure saved without `style.save_paper` (rule 18), a link to one of
our profiles in anything the website or the paper publishes (rule 19),
are each a finding.


## README rules (every `analysis/README.md` and `analysis/rqNN_*/README.md`)

A session that touches a README follows these; the review skill checks them.

1. **One README per level.** `analysis/README.md` describes the research questions and points to each RQ; each `rqNN_*/README.md` is the single document of that RQ. No READMEs under `pretraining/<pool>/` — a write-up belongs in the RQ's README.
2. **The current sweep first.** The body is about the predictivity ladder (pool `predictivity` and its siblings). Anything from the 36-model sweep or the external/public models goes in a final section titled "Extensions from other sweeps", each item opening with the sweep it comes from and why its numbers are not pooled with the ladder's.
3. **Every figure states its population.** Beside each rq02-family figure: which decision accuracy (DA-size, DA-ckpt, DA-goal), which filter (none, `above_66_size`, `above_66_ckpt`, `above_66_either`, `above_66_both`, `above_80`), which pair set (multi-axis, mono-axis, seed), and the gate pool. For other RQs: the pool, sizes, seeds, gate and any task filter.
4. **Storyline order.** Figures follow the argument, not the script order; each figure's text says what the previous one left open and links forward to the figure that.. takes the finding further, within the README and, where it exists, across RQs (`../rqNN_*/README.md#anchor`).
5. **After every figure: bullets, not prose.** A "Key findings" bullet list with the numbers and interpretations, then a "Follow-ups" bullet list of figure variants worth generating with the reason for each. No paragraphs longer than two sentences.
6. **Link the figure on GitHub.** After every figure, a link to its file on the `main` branch: `https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/<rq>/<stage>/<pool>/<name>.png` (the CSV beside it the same way).
7. **Auto blocks stay generated.** Text between `<!-- BEGIN auto:` and `<!-- END auto:` is written by the script named in the marker; hand-written text goes around it, and a number in prose is read from the CSV, never from memory.
8. **Numbers move.** Each README states the ladder-report snapshot its numbers come from; a refresh that regenerates the auto blocks also triggers a re-read of the prose numbers.
