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

| # | rule | constant / helper |
|---|---|---|
| 1 | **The above-random gate.** A benchmark's number at a size counts only where the task is above chance at that size: the Wilson 95 % lower bound of a run's accuracy clears the chance level for at least `MIN_SHARE` (half) of the size's runs. A quantity that ranks against the reference (DA-size, a surrogate of it) also needs the task above chance at the reference. A cross-size fit uses only sizes where the task is above chance. Gated cells are drawn grey and kept in the CSV, never dropped from a grid. Tasks without a chance level (BPB, the loss, generative tasks) are never gated: a mask of NA passes. The chance level is uniform guessing: 1/n_options, or E[1/n_i] over the items when the count varies (TruthfulQA mc1, 0.2253), or the mean true-option share for `truthfulqa_mc2`, whose score is probability mass rather than a pick-one accuracy (0.449; `above_random.CHANCE`, read through `task_chance`). It does not test a constant answer, which scores the majority gold label's share (hellaswag_ta: 0.2585 against 0.25), and the runs of a size answer the same items, so their verdicts are correlated, not independent trials. | `rq00 above_random.load_mask`, `utils.passes_gate`, `grids.mark_gated` |
| 2 | **Trained languages only.** A model's score on a language its mixture does not train is a transfer measurement. It is used in rq06 (language transfer) and nowhere else, not even pooled into a mean. `bpb_macro` and `train_loss` are measurements of the whole mixture and always count. | enforced in `utils.build_snr_pool`; opt out with `untrained=True` (rq06, and the gate, which must cover every task) |
| 3 | **Ten checkpoints.** Every quantity read along a run uses the ten evaluated tenths (10 %, 20 %, …, 100 %), the grid every size shares; `da_early_fracs` is the nine before the final. BPB is also scored on the twentieths; those rows are left out wherever BPB and benchmarks are compared or a checkpoint axis is drawn. Progress is labelled in Chinchilla multiples (1C–5C), never as a share of the run. | `utils.CKPT_DA_EARLY_FRACS`, `utils.SHARED_FRACS`, `utils.on_shared_grid`, `grids.chinchilla` |
| 4 | **One noise window.** Checkpoint noise is the standard deviation over the `k`/`NOISE_GRID` (k/20) points in the last `NOISE_WINDOW` (20 %) of a run, the WSD decay, and the same window and grid for every kind of measurement: 80, 85, 90, 95 and 100 %, five points. BPB has always been scored there; `launch_trainings.due_iters` adds the benchmark evals at 85 % and 95 %, which the size grids alone miss (a 20-save size lands on the tenths, a 60-save size on the thirtieths). A run whose two new evals have not landed contributes three points, which is a noisier estimate of the same quantity, not a second definition. | `utils.NOISE_WINDOW`, `utils.NOISE_GRID`, `utils.noise_checkpoints`, `launch_trainings.due_iters`, `ladder._on_shared_grid` |
| 5 | **Three pairs.** A decision-accuracy cell needs at least `MIN_PAIRS` (3) design-variant pairs; below that it is NaN and its pair count is still written next to it. The rq02 kernels enforce this and print how many cells it emptied; any other DA computation applies the same constant and reports the same way. A cell's pair count is `k(k-1)/2` over its `k` families, so it is triangular — 0, 1, 3, 6, 10, … and never 2. `MIN_PAIRS` = 3 therefore means "three families"; setting it to 2 changes nothing, and the only value that admits a two-family cell is 1, where DA can only be 0 or 1. | `utils.MIN_PAIRS`, `compute_da` kernels, `da_n_pairs_per_task.csv` |
| 6 | **One task per benchmark and language.** A benchmark with sub-benchmarks (MMLU subjects, INCLUDE domains) is read as its per-language parent only. Sub-benchmarks are used in rq08 (subset selection) and nowhere else. | enforced in `utils.build_snr_pool`; opt out with `facets=True` (rq08 only); `utils.parents_only`, `utils._is_parent_task` |
| 7 | **`multi` is not a language.** The cross-language aggregates (`bpb_macro`, `train_loss`, `include_base_44`) and unresolved tasks (`??`) are never a row of a per-language table, never one of "N languages", never a language in a correlation. | `utils.languages_only`, `utils.LANGUAGE_AGGREGATES` |
| 8 | **Three tasks per language.** A per-language correlation (a Pearson or Spearman r over the language's tasks) needs at least `MIN_LANG_TASKS` (3) distinct tasks with a value; fewer is NaN, not a point in a mean. This is a stability floor, not a significance one — see [Why three tasks per language](#why-three-tasks-per-language) below. A per-language r is descriptive and is labelled as such; the significance claim belongs to the r pooled over languages. | `utils.MIN_LANG_TASKS` |
| 9 | **One reference.** The reference is `TARGET_SIZE` (1.7B) for every question. The only exception is the L2 **ES** setting, which stops at 1B for lack of data; a table or figure that includes it names the 1B reference next to it. L2 ZH runs to 1.7B (2026-09-20) and is the third family that lets rule 5 accept L2 at the reference. Where the 1.7B cell of a setting has no result yet, the cell is empty, never filled from a smaller size. Any comparison of the L2 schemes states the epoch count — see [Second-language repetition at L = 2](#second-language-repetition-at-l--2). | `utils.TARGET_SIZE`; `pretrain.launch_trainings.scheme_sizes` |
| 10 | **Sizes 90M–1.7B, at each rung's own batch.** Every analysis reads `ANALYSIS_SIZES`, the ladder from 90M up to the reference. The 90M and 175M rungs were retrained at their own batch (84 / 168) on 2026-09-23; the diverged batch-504 runs they replace stay on disk and in older reports, and the loader keeps only the runs at the batch a rung uses now (`ladder_report.on_grid`, applied in `snr/download/ladder.py`), so an old run never appears: not as a value, an empty column, an axis tick or a README column. The 3B rung sits ABOVE the reference and is dropped the same way: it exists for the size-generalization question, which is its own RQ and opts in with `above_reference=True`. Nothing else reads a size above the reference, because a table whose reference is 1.7B cannot carry a column the reference does not cover. `ANALYSIS_SIZES` is derived from `TARGET_SIZE`, so moving the reference moves the ladder with it. | `utils.ANALYSIS_SIZES`, applied in `utils.build_snr_pool`; `above_reference=True` for the size-generalization RQ alone (`rq10_size_generalisation/above_reference.py`, exempt in `check_rules.EXEMPT`) |
| 11 | **No leakage.** A statistic presented as available at a proxy size, or before the reference is trained, is computed from that proxy's data alone. A fit over sizes that includes the reference is a reference-size quantity and is labelled as one. | rq04 |
| 12 | **Figures.** A CSV of the same name next to every PNG; white = no value, grey = gated; a line under the title saying how a cell is computed; the population (tasks, pairs, languages) stated wherever a mean is shown. | `grids`, `style.save_figure` |
| 13 | **Populations move; say so.** When the set of tasks or pairs behind a cell differs across a row (the gate keeps different tasks at different sizes), the figure or table carries the count, and the README says the populations differ. | `grids` count overlays |
| 14 | **Outputs follow the code.** After a change to the loader, `configs/models.json` → `snr`, or a helper above, the pipeline is re-run before any table is read or cited; `check_rules.py` is the test that the tables on disk obey the rules. | `run_all_predictivity.sh`, `check_rules.py` |
| 15 | **Say which pairs a decision accuracy is over.** A DA table carries an `axes` column naming its pair set: `multi-axis` (every pair of design variants at the pool's seed — the convention to 2026-09-22, in which two thirds of the pairs move more than one axis at once), `mono-axis` (the pairs moving exactly one of L, arch, list, T, lang2, en — the English corpus, DCLMP/FWEB — the seed held — the decision a practitioner makes, and what upstream's "every pair" is by construction) and `seed` (the null: two draws of ONE design, emitted only where the pool has replicate seeds). `scheme` is never an axis: it encodes the language list, the sampling temperature and the second language, and `DATA_SCHEMES` is the source of truth for the first two. A consumer that does not ask reads `multi-axis`, so a table written before this rule needs no migration; a figure drawn over one pair set is filtered by a reliability computed on the same one, and its twin sits beside it under `AXES_SUFFIX`. | `utils.DESIGN_AXES`, `utils.design_axes`, `utils.pair_sets`, `utils.pair_agreement`, `utils.one_axes`, `utils.AXES_SUFFIX` |

## The reformulated twins are in the populations

Since 2026-09-22 the `rf_*` twins of the three letter-format families, and
`rfgm_include_base_44`, are ordinary benchmarks in the `auto` group: they are
gated, plotted and counted like any other. Two consequences a reader has to
be told (rule 13):

- **Benchmark populations grew.** Any unqualified "over the benchmarks" mean
  — rq02's per-benchmark DA, rq03's SNR table, rq04's per-language r, rq09's
  family aggregate — now includes the twins. Figures carry their task counts,
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
twins and 34 of the Gemini-rewritten ones. Before the twins entered the pool
those families contributed almost nothing to any RQ.

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
from. `bbq` alone stays a candidate (23 min per checkpoint, a third of the
top-up bill, and it clears a 1/12 chance trivially).

## Why three tasks per language

`MIN_LANG_TASKS` trades coverage against stability, and it is worth being
explicit that it never buys significance. In rq04 it is the binding
constraint — `ar` has 22 tasks in the pool and still went NaN at 5 — because
a task counts only when it has both an SNR and a decision-accuracy value:

| `MIN_LANG_TASKS` | languages with an rq04 correlation |
| ---: | ---: |
| 5 (until 2026-09-20) | 6 |
| 4 | 10 |
| **3 (now)** | **17** |
| 2 | 26 |

Against that, the |r| a Pearson correlation must exceed to reach p < 0.05:

| n | 3 | 4 | 5 | 6 | 10 | 50 | 100 | 300 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| critical \|r\| | 0.997 | 0.950 | 0.878 | 0.811 | 0.632 | 0.279 | 0.197 | 0.113 |

The best per-language r in this study is 0.24, so no per-language cell is
significant at 5 and none would be at 3 either. The threshold was never doing
inferential work, and holding it at 5 bought nothing while costing 11 of 17
languages. Significance lives in the pooled correlation, which is where the
paper makes its claim: over every language, `rel_star_discrepancy` gives
r = 0.194 at n = 324 (p = 4.5e-4) against DA-size and r = 0.138 at n = 2906
(p = 8.7e-14) against DA-ckpt. Those points are not independent (one task
appears at several sizes), so the nominal p is optimistic; the effect
survives a large deflation, but a clustered test is the rigorous version and
is not done yet. Per-language panels carry their n and say they are
descriptive.

## Second-language repetition at L = 2

Rule 9's L2 exception exists because the L2 builds are capped by the SOURCE,
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
at 1.7B, so a flip could be the data repeating instead — which is why ES is
capped at 1B and ZH is not. Any table that compares the L2 schemes states the
epoch count.

ZH and ES are the **second-language axis at L = 2**: scheme A trains English
plus Russian, ZH plus Chinese, ES plus Spanish, and all three exist in the
deep architecture only. They are three levels of one intervention, not three
unrelated schemes — the L2 analogue of the B / AT3 data axis at higher L —
and that is what makes them the three families rule 5 counts at L2.

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
unstated population, a table that reads `da_per_task.csv` without `one_axes`,
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
