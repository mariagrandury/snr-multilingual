# Above chance against the tokens of the language seen

Does a benchmark task clear the chance threshold once the model has seen enough
of its language? Per benchmark and size, the share of the task's (language,
language setting) cells above chance against the training tokens of the task's
language, read at the final checkpoint and at every evaluated tenth of the run.
The figures and tables are written by
[`../rq01_scaling_predictability/tokens_seen.py`](../rq01_scaling_predictability/tokens_seen.py)
(pool `predictivity_all`), from the ladder-report snapshot of 2026-10-06; they
were moved here from `rq01_scaling_predictability/` on 2026-10-07. The paper
copies are `pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper.png` and
`pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical.png`.
The decision-accuracy counterpart is in
[`../rq02_da_vs_train_tokens/`](../rq02_da_vs_train_tokens/README.md).

<!-- BEGIN auto:chance-vs-train-tokens (tokens_seen.py --pool predictivity_all) -->
## Share of cells above chance against the tokens of the language seen

`tokens_seen.py`, `predictivity_all` pool: a language's evaluation against the training tokens of that language the model had seen (its share of the mixture from the build's plan × the cell's budget × the checkpoint's share of the run). Regenerate with `python analysis/rq01_scaling_predictability/tokens_seen.py --pool predictivity_all`.

**Share of a benchmark's cells above chance against the tokens seen.** One (task, size, L) cell per benchmark, language and language setting; above chance by rule 1's Wilson test on the cell's runs; cells binned 3 per decade of tokens, a point = the share of the bin's cells above chance with the cell count, one line per size. Two populations: `deep_A_1904`, the plan grid, deep / scheme A / seed 1904: one run per cell; `1904`, every seed-1904 run at the (size, L) that trains the language, every scheme and architecture. The `.csv` next to each figure is the cell table (benchmark, task, language, model_size, language_scheme, train_tokens, task_score, above_chance, share_above, n_runs), `_points.csv` the binned values drawn. The `_ckpts` twins read the same runs at every evaluated tenth (a cell = (task, size, L, tenth), x = the tokens seen by that checkpoint): ten times the cells, and a token axis that runs through every training run. Each cell table is drawn per benchmark (`_by_benchmark_`), per language (`_by_language_`, panels from the best- to the least-resourced language) and pooled (`_all_`, one panel; their `.csv` is the binned points drawn). Cells above chance: `deep_A_1904` 6711 of 14657 (841 tasks), `deep_A_1904_ckpts` 63070 of 146570 (841 tasks), `1904` 7719 of 16602 (841 tasks), `1904_ckpts` 72949 of 166020 (841 tasks).

![Share above chance vs tokens seen, deep_A_1904](pretraining/predictivity_all/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904.png)

![Share above chance vs tokens seen, deep_A_1904_ckpts](pretraining/predictivity_all/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904_ckpts.png)

![Share above chance vs tokens seen, 1904](pretraining/predictivity_all/pass_prob_vs_train_tokens_by_benchmark_1904.png)

![Share above chance vs tokens seen, 1904_ckpts](pretraining/predictivity_all/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.png)
<!-- END auto:chance-vs-train-tokens -->
