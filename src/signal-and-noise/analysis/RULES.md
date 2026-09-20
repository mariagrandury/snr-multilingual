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
| 4 | **One noise window.** Checkpoint noise is the standard deviation over the shared tenths in the last `NOISE_WINDOW` (20 %) of a run, the WSD decay, and the same window for every kind of measurement. Today that is the checkpoints at 80, 90 and 100 %; evaluating 85 % and 95 % would make it five. | `utils.NOISE_WINDOW`, `utils.noise_checkpoints` |
| 5 | **Three pairs.** A decision-accuracy cell needs at least `MIN_PAIRS` (3) design-variant pairs; below that it is NaN and its pair count is still written next to it. The rq02 kernels enforce this and print how many cells it emptied; any other DA computation applies the same constant and reports the same way. A cell's pair count is `k(k-1)/2` over its `k` families, so it is triangular — 0, 1, 3, 6, 10, … and never 2. `MIN_PAIRS` = 3 therefore means "three families"; setting it to 2 changes nothing, and the only value that admits a two-family cell is 1, where DA can only be 0 or 1. | `utils.MIN_PAIRS`, `compute_da` kernels, `da_n_pairs_per_task.csv` |
| 6 | **One task per benchmark and language.** A benchmark with sub-benchmarks (MMLU subjects, INCLUDE domains) is read as its per-language parent only. Sub-benchmarks are used in rq08 (subset selection) and nowhere else. | enforced in `utils.build_snr_pool`; opt out with `facets=True` (rq08 only); `utils.parents_only`, `utils._is_parent_task` |
| 7 | **`multi` is not a language.** The cross-language aggregates (`bpb_macro`, `train_loss`, `include_base_44`) and unresolved tasks (`??`) are never a row of a per-language table, never one of "N languages", never a language in a correlation. | `utils.languages_only`, `utils.LANGUAGE_AGGREGATES` |
| 8 | **Five tasks per language.** A per-language correlation (a Pearson or Spearman r over the language's tasks) needs at least `MIN_LANG_TASKS` (5) distinct tasks with a value; fewer is NaN, not a point in a mean. This is a stability floor, not a significance one: at n = 5 an r must exceed 0.88 to reach p < 0.05, and no per-language r in this study is near that. A per-language r is read as descriptive; the significance claim belongs to the r pooled over languages, where n is in the hundreds. | `utils.MIN_LANG_TASKS` |
| 9 | **One reference.** The reference is `TARGET_SIZE` (1.7B) for every question. The only exception is the L2 ZH and ES settings, which stop at 1B for lack of data; a table or figure that includes them names the 1B reference next to them. Where the 1.7B cell of a setting has no result yet, the cell is empty, never filled from a smaller size. | `utils.TARGET_SIZE`; `pretrain.launch_trainings.scheme_sizes` |
| 10 | **Sizes 175M–1.7B.** The 90M rung is dropped at load and never appears: not as a value, an empty column, an axis tick or a README column. | `pretrain.launch_trainings.EVAL_SIZES`, applied in `utils.build_snr_pool` |
| 11 | **No leakage.** A statistic presented as available at a proxy size, or before the reference is trained, is computed from that proxy's data alone. A fit over sizes that includes the reference is a reference-size quantity and is labelled as one. | rq04 |
| 12 | **Figures.** A CSV of the same name next to every PNG; white = no value, grey = gated; a line under the title saying how a cell is computed; the population (tasks, pairs, languages) stated wherever a mean is shown. | `grids`, `style.save_figure` |
| 13 | **Populations move; say so.** When the set of tasks or pairs behind a cell differs across a row (the gate keeps different tasks at different sizes), the figure or table carries the count, and the README says the populations differ. | `grids` count overlays |
| 14 | **Outputs follow the code.** After a change to the loader, `configs/models.json` → `snr`, or a helper above, the pipeline is re-run before any table is read or cited; `check_rules.py` is the test that the tables on disk obey the rules. | `run_all_predictivity.sh`, `check_rules.py` |

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
