# Above chance against the tokens of the language seen

**Question.** Does a benchmark task clear chance once the model has seen enough
of the task's language, and is that exposure or model size?

**Setup.** A cell is one (task, size, language setting L), read at each of the ten evaluated tenths of the run; it is above chance by the gate's per-run Wilson test, applied to the cell's runs with the gate's half-of-the-runs rule. Its x is the tokens of the task's language the checkpoint had seen: the language's share of the build's mixture × the size's budget × the tenth.

Population: the seed-1904 runs of pool `predictivity_seeds` (the runs of the headline `predictivity` pool: every data build and ladder), sizes 90M–1.7B, L ∈ {1, 2, 8, 15, 30, 50}, 1–5 runs per cell (median 4). That is 841 tasks of 40 benchmarks in 50 languages, 16,602 cells at the final and 166,020 over the tenths; BPB and the benchmark-BPB twins have no chance level and are not in it.

Numbers below are from the ladder-report snapshot of 2026-10-06 04:26 (refresh of 2026-10-07), read from the CSVs on 2026-10-07. The tables are written by [`../rq01_scaling_predictability/tokens_seen.py`](../rq01_scaling_predictability/tokens_seen.py); the decision-accuracy counterpart is [`../rq02_da_vs_train_tokens/`](../rq02_da_vs_train_tokens/README.md).

**Key finding.** Outside English, the share of cells above chance rises with every decade of the language's tokens, from 39 % at 10^7–10^8 tokens to 60 % at 10^10–10^11, but at matched tokens the larger model still clears chance more often (40 % at 90M against 57 % at 1.7B at 10^8–10^9 tokens): exposure is necessary, not sufficient.

## Share above chance against tokens seen, per benchmark

![Share above chance vs tokens seen, per benchmark, every tenth](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper.csv)

The paper copy of the `1904_ckpts` population: one panel per benchmark (40), one line per size, 3 token bins per decade, the cell count on each point. The bins mix languages, L and sizes, so the population behind a point differs along a line and across lines.

Key findings (numbers from `pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.csv`, 166,020 cells; final-only shares from the `_1904.csv` twin, 16,602 cells):
- Pooled over every language the curve peaks at 48 % (10^8–10^9), dips to 47 % (10^9–10^10) and drops to 36 % in 10^10–10^11 (23,897 cells), because 80 % of that bin is English: English cells clear chance in 29 % of 38,160 (106 tasks), the other languages in 48 % of 127,860 (735 tasks).
- Without English the share rises monotonically per decade: 34 % (10^5, 77 cells), 37 % (10^6), 39 % (10^7), 48 % (10^8), 57 % (10^9), 60 % (10^10, 4,711 cells).
- Within one (task, language), the third of its cells with the most tokens is above chance in 51 % against 37 % for the third with the fewest (non-English 57 % against 40 %); the thirds mix size, L and tenth.
- Size matters at matched tokens (non-English, 10^8–10^9): 90M 40 %, 175M 43 %, 350M 46 %, 600M 50 %, 1B 53 %, 1.7B 57 %; at 10^9–10^10 the spread is 50 % (90M) to 63 % (1.7B).
- Along the run the share rises from 39 % at the first tenth to 46 % at the final (1.7B 47 % → 57 %, 90M 33 % → 38 %); 11 % of cells go from below to above chance between the first and the last tenth, 3 % the other way, 44 % are never above chance and 33 % always are.
- The letter-choice originals of the multilingual knowledge benchmarks stay low at the 1.7B final: INCLUDE 12 % (36 languages), Belebele 19 % (49), Global-MMLU 22 % (29); their reformulated twins reach 83 %, 97 % and 100 %. Separately, the cloze-scored Global PIQA parallel stays at 4 % (46 languages) and has no twin.
- At the 1.7B final xcopa, xstorycloze, xwinograd, MultiBLiMP and ARC-MT are above chance in every cell, HellaSwag in 96 %, XNLI in 92 %.

Follow-ups:
- The same figure without English, so the pooled line is not bent by the one language whose cells hold the hardest English-only benchmarks.
- A size-by-token-decade heatmap per benchmark, to separate the effect of exposure from the effect of size within a benchmark rather than pooled.
- Whether the low letter-choice originals are a matter of format or of tokens is taken further in [INCLUDE against its reformulated twin](#include-against-its-reformulated-twin).

## INCLUDE against its reformulated twin

The first figure leaves open whether a flat line is the language or the format; INCLUDE, scored as letter choice and as its reformulated twin on the same 36 languages, answers that for one benchmark.

![INCLUDE over its reformulated twin, share above chance vs tokens seen](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical.csv)

Population: `1904_ckpts`, `include_base_44` (top) over `rf_include_base_44` (bottom), 5,700 cells each, every size and tenth.

Key findings (from the `_1904_ckpts.csv` cell table):
- The original stays at 5–11 % above chance from 10^7 tokens on (11 %, 6 %, 7 %, 5 % per decade up to 10^10–10^11), 9 % of its cells overall.
- The twin rises with tokens on the same cells: 13 % (10^7), 40 % (10^8), 85 % (10^9), 99 % (10^10), 41 % overall; at 1.7B alone 34 %, 64 %, 86 %, 98 %.
- Exposure moves the twin and not the original, so the original's flat line is its format, not too few tokens of the language.

Follow-ups:
- The same pair for Belebele and Global-MMLU, the other originals that stay low at 1.7B, to check that the twin's rise with tokens is general.

## Every population and view

The generated block below carries the four populations (the plan grid `deep_A_1904` and every build `1904`, each at the final and over the tenths) and their per-size totals.

<!-- BEGIN auto:chance-vs-train-tokens (tokens_seen.py --pool predictivity_seeds) -->
## Share of cells above chance against the tokens of the language seen

`tokens_seen.py`, `predictivity_seeds` pool: a language's evaluation against the training tokens of that language the model had seen (its share of the mixture from the build's plan × the cell's budget × the checkpoint's share of the run). Regenerate with `python analysis/rq01_scaling_predictability/tokens_seen.py --pool predictivity_seeds`.

**Share of a benchmark's cells above chance against the tokens seen.** One (task, size, L) cell per benchmark, language and language setting; above chance by rule 1's Wilson test on the cell's runs; cells binned 3 per decade of tokens, a point = the share of the bin's cells above chance with the cell count, one line per size. Two populations: `deep_A_1904`, the plan grid, deep / data A / seed 1904: one run per cell; `1904`, every seed-1904 run at the (size, L) that trains the language, every data build and ladder. The `.csv` next to each figure is the cell table (benchmark, task, language, model_size, language_scheme, train_tokens, task_score, above_chance, share_above, n_runs), `_points.csv` the binned values drawn. The `_ckpts` twins read the same runs at every evaluated tenth (a cell = (task, size, L, tenth), x = the tokens seen by that checkpoint): ten times the cells, and a token axis that runs through every training run. Each cell table is drawn per benchmark (`_by_benchmark_`), per language (`_by_language_`, panels from the best- to the least-resourced language) and pooled (`_all_`, one panel; their `.csv` is the binned points drawn). Cells above chance: `deep_A_1904` 6711 of 14657 (841 tasks), `deep_A_1904_ckpts` 63070 of 146570 (841 tasks), `1904` 7772 of 16602 (841 tasks), `1904_ckpts` 73478 of 166020 (841 tasks).

![Share above chance vs tokens seen, deep_A_1904](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904.csv)

![Share above chance vs tokens seen, deep_A_1904_ckpts](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904_ckpts.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904_ckpts.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904_ckpts.csv)

![Share above chance vs tokens seen, 1904](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904.csv)

![Share above chance vs tokens seen, 1904_ckpts](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.csv)

Key findings:
- `deep_A_1904`: 46% of 14657 cells above chance; per size 90M 37%, 175M 39%, 350M 43%, 600M 47%, 1B 51%, 1.7B 58%.
- `deep_A_1904_ckpts`: 43% of 146570 cells above chance; per size 90M 35%, 175M 37%, 350M 40%, 600M 44%, 1B 48%, 1.7B 54%.
- `1904`: 47% of 16602 cells above chance; per size 90M 38%, 175M 40%, 350M 45%, 600M 49%, 1B 52%, 1.7B 57%.
- `1904_ckpts`: 44% of 166020 cells above chance; per size 90M 37%, 175M 38%, 350M 42%, 600M 46%, 1B 50%, 1.7B 54%.
<!-- END auto:chance-vs-train-tokens -->

Follow-ups:
- The `_by_language_` figures with the token bins pooled across sizes, to separate the effect of exposure from the effect of size within a language.
- A twin restricted to the gate's surviving tasks, to check that the share rises with tokens on the tasks the later analyses keep.
- A twin over the reformulated (`rf_`) tasks alone, since the letter-choice originals of INCLUDE, Belebele and Global-MMLU stay at 12–22 % above chance even at 1.7B.
