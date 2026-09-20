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
| 1 | **The above-random gate.** A benchmark's number at a size counts only where the task is above chance at that size: the Wilson 95 % lower bound of a run's accuracy clears the chance level for at least `MIN_SHARE` (half) of the size's runs. A quantity that ranks against the reference (DA-size, a surrogate of it) also needs the task above chance at the reference. A cross-size fit uses only sizes where the task is above chance. Gated cells are drawn grey and kept in the CSV, never dropped from a grid. Tasks without a chance level (BPB, the loss, generative tasks) are never gated: a mask of NA passes. | `rq00 above_random.load_mask`, `utils.passes_gate`, `grids.mark_gated` |
| 2 | **Trained languages only.** A model's score on a language its mixture does not train is a transfer measurement. It is used in rq06 (language transfer) and nowhere else, not even pooled into a mean. `bpb_macro` and `train_loss` are measurements of the whole mixture and always count. | enforced in `utils.build_snr_pool`; opt out with `untrained=True` (rq06, and the gate, which must cover every task) |
| 3 | **Ten checkpoints.** Every quantity read along a run uses the ten evaluated tenths (10 %, 20 %, …, 100 %), the grid every size shares; `da_early_fracs` is the nine before the final. BPB is also scored on the twentieths; those rows are left out wherever BPB and benchmarks are compared or a checkpoint axis is drawn. Progress is labelled in Chinchilla multiples (1C–5C), never as a share of the run. | `utils.CKPT_DA_EARLY_FRACS`, `utils.SHARED_FRACS`, `utils.on_shared_grid`, `grids.chinchilla` |
| 4 | **One noise window.** Checkpoint noise is the standard deviation over the `k`/`NOISE_GRID` (k/20) points in the last `NOISE_WINDOW` (20 %) of a run, the WSD decay, and the same window and grid for every kind of measurement: 80, 85, 90, 95 and 100 %, five points. BPB has always been scored there; `launch_trainings.due_iters` adds the benchmark evals at 85 % and 95 %, which the size grids alone miss (a 20-save size lands on the tenths, a 60-save size on the thirtieths). A run whose two new evals have not landed contributes three points, which is a noisier estimate of the same quantity, not a second definition. | `utils.NOISE_WINDOW`, `utils.NOISE_GRID`, `utils.noise_checkpoints`, `launch_trainings.due_iters`, `ladder._on_shared_grid` |
| 5 | **Three pairs.** A decision-accuracy cell needs at least `MIN_PAIRS` (3) design-variant pairs; below that it is NaN and its pair count is still written next to it. The rq02 kernels enforce this and print how many cells it emptied; any other DA computation applies the same constant and reports the same way. A cell's pair count is `k(k-1)/2` over its `k` families, so it is triangular — 0, 1, 3, 6, 10, … and never 2. `MIN_PAIRS` = 3 therefore means "three families"; setting it to 2 changes nothing, and the only value that admits a two-family cell is 1, where DA can only be 0 or 1. | `utils.MIN_PAIRS`, `compute_da` kernels, `da_n_pairs_per_task.csv` |
| 6 | **One task per benchmark and language.** A benchmark with sub-benchmarks (MMLU subjects, INCLUDE domains) is read as its per-language parent only. Sub-benchmarks are used in rq08 (subset selection) and nowhere else. | enforced in `utils.build_snr_pool`; opt out with `facets=True` (rq08 only); `utils.parents_only`, `utils._is_parent_task` |
| 7 | **`multi` is not a language.** The cross-language aggregates (`bpb_macro`, `train_loss`, `include_base_44`) and unresolved tasks (`??`) are never a row of a per-language table, never one of "N languages", never a language in a correlation. | `utils.languages_only`, `utils.LANGUAGE_AGGREGATES` |
| 8 | **Three tasks per language.** A per-language correlation (a Pearson or Spearman r over the language's tasks) needs at least `MIN_LANG_TASKS` (3) distinct tasks with a value; fewer is NaN, not a point in a mean. This is a stability floor, not a significance one — see [Why three tasks per language](#why-three-tasks-per-language) below. A per-language r is descriptive and is labelled as such; the significance claim belongs to the r pooled over languages. | `utils.MIN_LANG_TASKS` |
| 9 | **One reference.** The reference is `TARGET_SIZE` (1.7B) for every question. The only exception is the L2 **ES** setting, which stops at 1B for lack of data; a table or figure that includes it names the 1B reference next to it. L2 ZH runs to 1.7B (2026-09-20) and is the third family that lets rule 5 accept L2 at the reference. Where the 1.7B cell of a setting has no result yet, the cell is empty, never filled from a smaller size. Any comparison of the L2 schemes states the epoch count — see [Second-language repetition at L = 2](#second-language-repetition-at-l--2). | `utils.TARGET_SIZE`; `pretrain.launch_trainings.scheme_sizes` |
| 10 | **Sizes 175M–1.7B.** The 90M rung is dropped at load and never appears: not as a value, an empty column, an axis tick or a README column. **Open:** `EVAL_SIZES` is `LADDER` minus 90M, and `LADDER` gained a 3B rung on 2026-09-19, so the first 3B eval to land will enter every ladder pool as a sixth size, above the rule-9 reference, with nothing announcing it. No 3B is trained yet. Decide before one is: either `build_snr_pool` filters to the five analysis sizes, or rule 9 says what a size above the reference means. | `pretrain.launch_trainings.EVAL_SIZES`, applied in `utils.build_snr_pool` |
| 11 | **No leakage.** A statistic presented as available at a proxy size, or before the reference is trained, is computed from that proxy's data alone. A fit over sizes that includes the reference is a reference-size quantity and is labelled as one. | rq04 |
| 12 | **Figures.** A CSV of the same name next to every PNG; white = no value, grey = gated; a line under the title saying how a cell is computed; the population (tasks, pairs, languages) stated wherever a mean is shown. | `grids`, `style.save_figure` |
| 13 | **Populations move; say so.** When the set of tasks or pairs behind a cell differs across a row (the gate keeps different tasks at different sizes), the figure or table carries the count, and the README says the populations differ. | `grids` count overlays |
| 14 | **Outputs follow the code.** After a change to the loader, `configs/models.json` → `snr`, or a helper above, the pipeline is re-run before any table is read or cited; `check_rules.py` is the test that the tables on disk obey the rules. | `run_all_predictivity.sh`, `check_rules.py` |

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
unstated population, are each a finding.
