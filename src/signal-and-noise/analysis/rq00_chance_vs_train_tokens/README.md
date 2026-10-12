# Above chance against the tokens of the language seen

**Question.** Does a benchmark task clear chance once the model has seen enough
tokens of the task's language, and is it the exposure or the model size that decides?

**Setup.**
- **Measured.** We measure the share of cells above chance. A cell is one (task, size, language setting L), read at one of the ten evaluated tenths of the run.
- **Models.** We use the seed-1904 runs of every data build and ladder, 90M–1.7B, at L ∈ {1, 2, 8, 15, 30, 50}.
- **Tasks.** The tasks are the benchmark tasks with a chance level in trained languages. BPB and the bBPB twins have no chance level and are left out.
- **Rule.** A cell is above chance when it passes the gate (the per-run Wilson test and the half-of-the-runs rule), applied at that tenth.
- **Tokens.** A cell sits at the number of tokens of the task's language that the checkpoint had seen: the language's share of the build's mixture × the size's token budget × the tenth. We bin the cells three per decade of tokens.
- **Thresholds.** Per benchmark, answer format and model size, we fit a logistic curve of the cells' above-chance verdict on the log of the language's tokens, on the non-English cells, and read the token count at which it reaches 50 % and 90 % of the cells, inside the token range that size spans (table: rq00_thresholds). We then test whether leaving out the cells below that threshold changes how well an aggregate score ranks the designs (table: app_rq00_token_exclusion).

Population: the seed-1904 runs of pool `predictivity_seeds` (the runs of the headline `predictivity` pool: every data build and ladder), sizes 90M–1.7B, L ∈ {1, 2, 8, 15, 30, 50}, 1–7 runs per cell (median 4). That is 933 tasks of 52 benchmarks in 50 languages, 18,069 cells at the final and 180,111 over the tenths. BPB and the benchmark-BPB twins have no chance level and are not in it.

Numbers below are from the ladder-report snapshot of 2026-10-10 04:31 and the outputs of the 2026-10-10 refresh (detrended checkpoint noise). The tables are written by [`../rq01_scaling_predictability/tokens_seen.py`](../rq01_scaling_predictability/tokens_seen.py), and the exclusion section by [`token_exclusion.py`](token_exclusion.py). The decision-accuracy counterpart is [`../rq02_da_vs_train_tokens/`](../rq02_da_vs_train_tokens/README.md).

**Key finding.** **Outside English, the share of cells above chance rises with every decade of the language's tokens from 10^7 on, but at matched tokens the larger model still clears chance more often.** The share goes from 42 % at 10^7–10^8 tokens to 64 % at 10^10–10^11. At 10^8–10^9 tokens it is 43 % at 90M against 62 % at 1.7B, so exposure is necessary but not sufficient.

## Share above chance against tokens seen, per benchmark

![Share above chance vs tokens seen, per benchmark, every tenth](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper.csv)

This is the paper copy of the `1904_ckpts` population: one panel per benchmark (52), one line per size, 3 token bins per decade, and the cell count on each point. The bins mix languages, L and sizes, so the population behind a point differs along a line and across lines.

Key findings (numbers from `pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.csv`, 180,111 cells, and the final-only shares from the `_1904.csv` twin, 18,069 cells):
- **Pooled over every language, the curve peaks at 10^8–10^9 tokens and falls after, because the top bins are mostly English.** It reads 52 % at 10^8–10^9, dips to 51 % at 10^9–10^10 and drops to 41 % at 10^10–10^11 (25,217 cells, 80 % of them English). The last bin, 10^11–10^12, holds 560 English cells at 40 %. English cells clear chance in 34 % of 40,263 (112 tasks), the other languages in 52 % of 139,848 (821 tasks).
- **Without English, the share stays near 40 % up to 10^8 tokens and rises with every decade after.** Per decade: 43 % (10^5, 82 cells), 42 % (10^6), 42 % (10^7), 52 % (10^8), 62 % (10^9), 64 % (10^10, 4,945 cells).
- **Within one (task, language), more tokens means more cells above chance.** The third of its cells with the most tokens is above chance in 57 %, against 40 % for the third with the fewest (non-English 62 % against 43 %). Each third mixes sizes, L and tenths.
- **Size matters at matched tokens.** Outside English at 10^8–10^9 tokens: 90M 43 %, 175M 46 %, 350M 50 %, 600M 54 %, 1B 58 %, 1.7B 62 %. At 10^9–10^10 the spread is 51 % (90M) to 69 % (1.7B).
- **Along the run, the share rises from 42 % at the first tenth to 51 % at the final.** Per size it goes from 52 % to 62 % at 1.7B and from 36 % to 41 % at 90M. Between the first and the last tenth, 12 % of cells go from below to above chance and 3 % the other way. 40 % are never above chance and 37 % always are.
- **The letter-choice originals of the multilingual knowledge benchmarks stay low at the 1.7B final.** INCLUDE is above chance in 12 % of its cells (36 languages), Belebele in 23 % (49) and Global-MMLU in 19 % (29). Their reformulated twins reach 83 %, 97 % and 100 %. Separately, the cloze-scored Global PIQA parallel reaches only 28 % (46 languages) and has no twin.
- **The cloze and minimal-pair benchmarks are above chance almost everywhere at the 1.7B final.** XCOPA, XStoryCloze, XWinograd, MultiBLiMP (with its extra set), ARC-MT, HellaSwag and TokSuite are above chance in every cell, XNLI in 92 %.

Follow-ups:
- The same figure without English, so that the one language whose cells hold the hardest English-only benchmarks does not bend the pooled line.
- A size-by-token-decade heatmap per benchmark, to separate the effect of exposure from the effect of size within a benchmark rather than pooled.
- Whether the low letter-choice originals are a matter of format or of tokens is taken further in [INCLUDE against its reformulated twin](#include-against-its-reformulated-twin).

## INCLUDE against its reformulated twin

The first figure leaves open whether a flat line comes from the language or from the format. INCLUDE, scored as letter choice and as its reformulated twin on the same 36 languages, answers that for one benchmark.

![INCLUDE over its reformulated twin, share above chance vs tokens seen](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical.csv)

Population: `1904_ckpts`, `include_base_44` (top) over `rf_include_base_44` (bottom), 5,700 cells each, every size and tenth.

Key findings (from the `_1904_ckpts.csv` cell table):
- **The original stays at 5–11 % above chance from 10^7 tokens on.** Per decade it reads 11 %, 6 %, 7 % and 5 % up to 10^10–10^11, and 9 % of its cells overall.
- **The twin rises with tokens on the same cells.** Per decade it reads 13 % (10^7), 41 % (10^8), 85 % (10^9) and 99 % (10^10), and 41 % overall. At 1.7B alone it reads 35 %, 65 %, 86 % and 98 %.
- **Exposure moves the twin and not the original.** The original's flat line comes from its format, not from too few tokens of the language.

Follow-ups:
- The same pair for Belebele and Global-MMLU, the other originals that stay low at 1.7B, to check that the twin's rise with tokens is general. The [token thresholds per benchmark and format](#token-thresholds-per-benchmark-and-format) below give their reading as one number per family.

## Token thresholds per benchmark and format

The INCLUDE pair shows the format deciding for one benchmark. The table below reads every multilingual family and format the same way, as the token count of the language from which most of its cells clear chance.

[CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_threshold_tokens_by_benchmark_1904_ckpts.csv) · [LaTeX on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_threshold_tokens_by_benchmark_1904_ckpts.tex)

Population: the `1904_ckpts` cell table of the figures above (seed-1904 runs, every build and ladder, 90M–1.7B, every evaluated tenth), the 31 benchmark families with cells in at least two languages, and the format (Original, RF, LLM-RF) read off the family name. Left out (one language, English unless noted): ACP-Bench (cloze, MCQ and its RF), BBH (cloze, MCQ and its RF), BoolQ, CommonsenseQA and its RF, KoBEST (Korean), MathQA, MMLU and its RF, NorEval and NorEval (extra) (Norwegian), OpenBookQA, PIQA, TokSuite math and STEM, ToxiGen, WinoGrande.

Definition (`pass_thresholds` in [`tokens_seen.py`](../rq01_scaling_predictability/tokens_seen.py)):
- **Bins.** The figure's bins, 3 per decade of the language's tokens. A bin with fewer than 20 cells has no value (the edge bins hold 1–19 cells and flipped thresholds by a decade or more).
- **Threshold.** The smallest bin from which the share of cells above chance is at or above the level (50 %, 90 %) in that bin and in every higher bin with a value. The table gives the bin's lower edge. "never": no such bin. "≤": already at the level in the first bin with a value. "–": no bin with 20 cells. The CSV carries the cell count of the threshold bin and every size's thresholds.
- **Views.** Pooled over sizes ("all"), and per size. The pooled bins hold small sizes at low tokens and large sizes at high tokens, and from 10^10 tokens on 53–100 % of a bin's cells are English, so the population moves along the token axis (rule 13).
- **Margin.** The family's median per-task margin over chance that a run must reach for its one-sided 95 % Wilson bound to clear chance (the gate's `needed_score` − chance, as in `rq00_gate_and_curves`' `above_random_thresholds`), beside the median item count.

| Benchmark | Format | Lang. | 50 % all | 90 % all | 50 % 90M | 90 % 90M | 50 % 1.7B | 90 % 1.7B | Margin | Items |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ARC | Original | 27 | 10^9 | 2.2×10^10 | never | never | 4.6×10^8 | 10^9 | 0.021 | 1,169 |
| ARC (MT) | Original | 11 | 10^9 | never | never | never | ≤ 2.2×10^8 | 4.6×10^8 | 0.021 | 1,172 |
| Belebele | Original | 49 | never | never | never | never | never | never | 0.024 | 900 |
| Belebele | RF | 49 | ≤ 2.2×10^6 | 2.2×10^8 | ≤ 2.2×10^6 | 2.2×10^9 | ≤ 4.6×10^7 | ≤ 4.6×10^7 | 0.024 | 900 |
| Belebele | LLM-RF | 49 | ≤ 2.2×10^6 | ≤ 2.2×10^6 | ≤ 2.2×10^6 | ≤ 2.2×10^6 | ≤ 4.6×10^7 | ≤ 4.6×10^7 | 0.024 | 898 |
| BLEnD | Original | 4 | never | never | 2.2×10^9 | never | never | never | 0.032 | 500 |
| CulturalBench-easy | Original | 8 | never | never | never | never | never | never | 0.157 | 25 |
| CulturalBench-easy | RF | 8 | never | never | never | never | never | never | 0.157 | 25 |
| CulturalBench-hard | Original | 8 | never | never | never | never | never | never | 0.090 | 100 |
| Global PIQA (non-parallel) | Original | 2 | 4.6×10^7 | 2.2×10^8 | – | – | – | – | 0.090 | 100 |
| Global PIQA (parallel) | Original | 46 | 2.2×10^10 | never | never | never | 2.2×10^10 | never | 0.070 | 103 |
| Global-MMLU | Original | 29 | never | never | never | never | never | never | 0.006 | 14,042 |
| Global-MMLU | RF | 29 | 10^8 | 4.6×10^8 | 4.6×10^7 | 2.2×10^9 | ≤ 10^8 | 2.2×10^8 | 0.006 | 14,042 |
| HellaSwag | Original | 26 | ≤ 2.2×10^6 | ≤ 2.2×10^6 | ≤ 2.2×10^6 | ≤ 2.2×10^6 | ≤ 4.6×10^7 | ≤ 4.6×10^7 | 0.007 | 9,276 |
| INCLUDE | Original | 36 | never | never | never | never | never | never | 0.032 | 548 |
| INCLUDE | RF | 36 | 4.6×10^8 | 4.6×10^9 | 4.6×10^8 | never | 2.2×10^8 | 10^10 | 0.032 | 548 |
| INCLUDE | LLM-RF | 36 | 4.6×10^8 | 4.6×10^9 | 4.6×10^8 | never | 10^8 | 10^10 | 0.032 | 546 |
| INCLUDE v2 (EN) | Original | 42 | ≤ 2.2×10^6 | never | 4.6×10^6 | never | ≤ 4.6×10^7 | never | 0.024 | 886 |
| INCLUDE v2 (OG) | Original | 42 | 4.6×10^8 | never | 10^9 | never | 2.2×10^8 | never | 0.024 | 886 |
| MMLU (Okapi) | RF | 27 | ≤ 2.2×10^6 | 2.2×10^8 | ≤ 4.6×10^6 | never | ≤ 2.2×10^7 | 2.2×10^8 | 0.006 | 13,101 |
| MMMLU | RF | 12 | 4.6×10^7 | 2.2×10^8 | 2.2×10^7 | 2.2×10^8 | 2.2×10^8 | 2.2×10^8 | 0.006 | 14,042 |
| MultiBLiMP | Original | 34 | ≤ 2.2×10^6 | ≤ 2.2×10^6 | ≤ 2.2×10^6 | ≤ 2.2×10^6 | ≤ 4.6×10^7 | ≤ 4.6×10^7 | 0.017 | 2,291 |
| MultiBLiMP (extra) | Original | 3 | ≤ 10^7 | ≤ 10^7 | – | – | – | – | 0.015 | 3,032 |
| PAWS-X | Original | 8 | 2.2×10^10 | 4.6×10^10 | never | never | 2.2×10^10 | 4.6×10^10 | 0.018 | 2,000 |
| TokSuite | Original | 5 | ≤ 2.2×10^7 | ≤ 2.2×10^7 | ≤ 2.2×10^7 | 10^8 | ≤ 4.6×10^8 | ≤ 4.6×10^8 | 0.125 | 40 |
| TruthfulQA (mc2) | Original | 3 | never | never | never | never | never | never | 0.030 | 788 |
| TruthfulQA-Multi | Original | 2 | never | never | never | never | never | never | 0.024 | 817 |
| XCOPA | Original | 8 | ≤ 4.6×10^6 | 4.6×10^9 | never | never | ≤ 10^8 | ≤ 10^8 | 0.038 | 500 |
| XNLI | Original | 15 | 4.6×10^7 | 2.2×10^10 | 10^8 | never | ≤ 10^8 | 2.2×10^10 | 0.016 | 2,490 |
| XStoryCloze | Original | 8 | 2.2×10^8 | 2.2×10^9 | 10^9 | never | ≤ 2.2×10^8 | ≤ 2.2×10^8 | 0.022 | 1,511 |
| XWinograd | Original | 6 | 2.2×10^8 | 10^9 | 2.2×10^8 | 10^9 | ≤ 10^9 | ≤ 10^9 | 0.043 | 410 |

Key findings (numbers from `pass_threshold_tokens_by_benchmark_1904_ckpts.csv`, and the matched-token shares from `pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_points.csv`, bins with at least 20 cells at both sizes):
- **Pooled over sizes, 9 of the 31 families never reach 50 % above chance and 13 never reach 90 %.** The nine are the letter-choice originals of Belebele, Global-MMLU and INCLUDE, CulturalBench-easy (both formats) and -hard, BLEnD and the two TruthfulQA families. Nine are at 50 % from their first bin (HellaSwag, MultiBLiMP and its extra set, Belebele RF and LLM-RF, INCLUDE v2 EN, XCOPA, MMLU (Okapi) RF, TokSuite), and thirteen have a threshold bin.
- **For the letter-choice families the format decides, not the tokens.** The originals of Belebele, Global-MMLU and INCLUDE never reach 50 %. Their RF twins reach 50 % at ≤ 2.2×10^6, 10^8 and 4.6×10^8 tokens and 90 % at 2.2×10^8, 4.6×10^8 and 4.6×10^9. Belebele LLM-RF is at 90 % from its first bin, and INCLUDE LLM-RF has the same pooled thresholds as RF. CulturalBench-easy's RF twin does not help (never at either level).
- **The minimum detectable margin does not order the thresholds.** An original and its twin share the items and so the margin (Global-MMLU 0.006 over 14,042 items, Belebele 0.024 over 900). Yet the original never clears 50 % and the twin does, by 10^8 and ≤ 2.2×10^6 tokens. HellaSwag (0.007) clears from its first bin. The largest margin is CulturalBench-easy's 0.157 over 25 items, which reaches 50 % in neither format, while TokSuite, second at 0.125 over 40 items, is at 50 % from its first bin.
- **Among the cloze and minimal-pair families, the 90 % thresholds spread over four decades.** HellaSwag and MultiBLiMP reach 90 % from ≤ 2.2×10^6, XWinograd at 10^9, XStoryCloze at 2.2×10^9, XCOPA at 4.6×10^9, ARC and XNLI at 2.2×10^10 and PAWS-X at 4.6×10^10. Global PIQA (parallel) reaches 50 % at 2.2×10^10 and never 90 %.
- **One threshold per family does not hold across the proxy sizes.** At 50 %, 90M has a threshold in 16 families and 1.7B in 20. ARC goes from never (90M and 175M, whose bins with a value end at 4.6×10^9 and 10^10 tokens) to 10^10 (350M), 4.6×10^9 (600M) and 4.6×10^8 (1B, 1.7B). At 90 %, Belebele RF needs 2.2×10^9 tokens at 90M against ≤ 4.6×10^7 at 1.7B, and Global-MMLU RF 2.2×10^9 against 2.2×10^8.
- **At matched tokens, 1.7B clears chance more often.** Over the 118 (family, bin) pairs that both sizes populate (27 families), 1.7B is at least 10 points higher in 56 and within 10 points in 55 (31 of them at 0–5 % or 95–100 % at both sizes). It is at least 10 points lower in 7: five CulturalBench-easy bins, one BLEnD bin and one MMMLU RF bin (87 % of 109 cells at 90M against 38 % of 21 at 1.7B, at 10^8–2.2×10^8 tokens). At 10^9–2.2×10^9 tokens ARC is above chance in 31 % of 90M cells (36) against 99 % of 1.7B cells (144).
- **The sizes coincide mostly where a family is saturated or at chance throughout.** HellaSwag, MultiBLiMP and Belebele LLM-RF are above chance from the first bin at every size and at both levels. The letter-choice originals, CulturalBench and TruthfulQA never are. Among the rest, only XNLI (50 % at 10^8 at 90M and ≤ 10^8 at 1.7B) and Global-MMLU RF (4.6×10^7 and ≤ 10^8) roughly agree at 50 %, and MMMLU RF agrees at 90 % (2.2×10^8 at both sizes) but is reversed at 50 % (2.2×10^7 at 90M, 2.2×10^8 at 1.7B).

Follow-ups:
- The table without English, since from 10^10 tokens on most cells are English and the high thresholds (ARC, XNLI, PAWS-X, Global PIQA at 2.2×10^10 and above) may be English thresholds.
- Per-size thresholds on the fixed set of languages each size covers at a token bin (`utils.fixed_population`), so that a threshold shift between sizes is not a shift in which languages fill the bin.
- CulturalBench-easy, where at 2.2×10^8–4.6×10^9 tokens 90M clears chance in 72–95 % of its cells against 11–29 % at 1.7B: check the 25-item tasks for a constant-answer artefact (rule 1 does not test a constant answer). Done: the early passes are a constant answer on an unbalanced key ([the gate README, section 1b](../rq00_gate_and_curves/README.md#1b-the-early-passes-a-constant-answer-the-longest-option-or-a-related-language)).
- A cluster bootstrap over (task, size, L) runs for the threshold bin, since a run's ten tenths are not independent cells.

## What the exclusion does to an aggregate score

The thresholds above say from which token count a benchmark is measurable. The paper leaves out every (task, language) cell below its family's threshold, both from the decisions and from the aggregate scores. This section asks whether the aggregate gains from that exclusion, in its standard error and in the decisions it ranks.

<!-- BEGIN auto:token-exclusion (token_exclusion.py --pool predictivity) -->
## Aggregate scores with and without the cells below the token threshold

**DA-size of an aggregate ranking · pool `predictivity` (seed 1904), finals 90M–1.7B · multi-axis and mono-axis pairs · the 22 benchmark families every policy scores · no filter.** Per run and family, the aggregate is the mean of the family's per-language tasks over the languages the run trains: every cell (`all`), the cells at or above the family's pooled 50% token threshold (`tokens`), or the cells above chance at the size and at 1.7B (`gate`). A family that never reaches the threshold has no `tokens` aggregate and is left out of every pooled row (rule 13). The `reference` column says whether the 1.7B aggregate uses the same policy or every cell. The `all` policy keeps the at-chance cells on purpose, as the counterfactual to rule 1. Regenerate with `python analysis/rq00_chance_vs_train_tokens/token_exclusion.py --pool predictivity`.

| reference | axes | proxy | DA all | DA tokens | DA gate | pairs (min) |
|---|---|---|---|---|---|---|
| all cells | mono-axis | 90M | 0.748 | 0.676 | 0.691 | 1171 |
| all cells | mono-axis | 175M | 0.739 | 0.707 | 0.705 | 1190 |
| all cells | mono-axis | 350M | 0.761 | 0.728 | 0.723 | 1248 |
| all cells | mono-axis | 600M | 0.783 | 0.740 | 0.744 | 1401 |
| all cells | mono-axis | 1B | 0.810 | 0.759 | 0.775 | 1401 |
| all cells | multi-axis | 90M | 0.802 | 0.723 | 0.747 | 7247 |
| all cells | multi-axis | 175M | 0.826 | 0.789 | 0.765 | 7410 |
| all cells | multi-axis | 350M | 0.845 | 0.810 | 0.793 | 7730 |
| all cells | multi-axis | 600M | 0.869 | 0.814 | 0.828 | 8779 |
| all cells | multi-axis | 1B | 0.881 | 0.825 | 0.846 | 8779 |
| same policy | mono-axis | 90M | 0.748 | 0.677 | 0.692 | 1171 |
| same policy | mono-axis | 175M | 0.739 | 0.701 | 0.700 | 1190 |
| same policy | mono-axis | 350M | 0.761 | 0.736 | 0.730 | 1248 |
| same policy | mono-axis | 600M | 0.783 | 0.747 | 0.750 | 1401 |
| same policy | mono-axis | 1B | 0.810 | 0.762 | 0.774 | 1401 |
| same policy | multi-axis | 90M | 0.802 | 0.722 | 0.742 | 7247 |
| same policy | multi-axis | 175M | 0.826 | 0.783 | 0.769 | 7410 |
| same policy | multi-axis | 350M | 0.845 | 0.816 | 0.802 | 7730 |
| same policy | multi-axis | 600M | 0.869 | 0.820 | 0.828 | 8779 |
| same policy | multi-axis | 1B | 0.881 | 0.832 | 0.850 | 8779 |

| size | median SE all | median SE tokens | median SE gate | cells kept by tokens |
|---|---|---|---|---|
| 90M | 0.0083 | 0.0097 | 0.0079 | 0.60 |
| 175M | 0.0092 | 0.0115 | 0.0088 | 0.68 |
| 350M | 0.0098 | 0.0114 | 0.0095 | 0.74 |
| 600M | 0.0117 | 0.0126 | 0.0110 | 0.80 |
| 1B | 0.0136 | 0.0140 | 0.0129 | 0.84 |
| 1.7B | 0.0154 | 0.0159 | 0.0145 | 0.87 |

Files: [`token_exclusion_da_size_both_axes.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity/token_exclusion_da_size_both_axes.csv), [`token_exclusion_da_size_mono_axis.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity/token_exclusion_da_size_mono_axis.csv) / [`.tex`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity/token_exclusion_da_size_mono_axis.tex) (the paper table), [`token_exclusion_da_size_per_family_both_axes.csv`](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity/token_exclusion_da_size_per_family_both_axes.csv).
<!-- END auto:token-exclusion -->

Key findings (`token_exclusion_da_size_both_axes.csv`, `token_exclusion_da_size_per_family_both_axes.csv`, snapshot of the 2026-10-10 refresh):
- **On the families that keep an aggregate, leaving out the cells below the threshold lowers the decision accuracy of the aggregate.** Over the 22 families every policy scores, mono-axis DA-size is 0.75–0.82 with every cell and 0.71–0.77 without the excluded ones (−0.02 to −0.06 per proxy size), and multi-axis DA-size is 0.81–0.88 against 0.74–0.83. Holding the 1.7B aggregate at every cell gives the same picture (0.71–0.77), so the drop is on the proxy side. The pooled rows do not hold one family set (17–22 families per row, at 90M mono-axis 21 for `all` against 17 for `tokens`), so part of the drop at 90M–350M is a change in which families count.
- **The drop sits in the families the exclusion cuts deepest.** Mono-axis, pooled over the proxies: Global PIQA (parallel) 0.74 → 0.47, ARC 0.86 → 0.68, PAWS-X 0.66 → 0.49, INCLUDE v2 (original) 0.79 → 0.63. The families with no excluded cell (HellaSwag, MultiBLiMP and its extra set, Belebele RF and LLM-RF, XCOPA, INCLUDE v2 EN, MMLU (Okapi) RF, TokSuite) do not move.
- **The standard error of the aggregate rises.** The median SE goes from 0.0088 to 0.0112 at 90M and from 0.0154 to 0.0159 at 1.7B, since the exclusion keeps 58 % of the cells at 90M and 87 % at 1.7B. The loss of languages outweighs the smaller spread.
- **What the exclusion does remove is the families at chance.** The nine families that never reach the threshold lose their aggregate. With every cell, that aggregate ranks the designs at 0.44–0.65 (mono-axis), so the exclusion keeps these near-chance aggregates out of a decision. Within a kept family, the at-chance languages still help the ranking.
- **The gate behaves like the token threshold.** Keeping only the cells above chance gives 0.72–0.78 mono-axis, and a median SE at or below the all-cells one at every size but 350M (0.0105 against 0.0103).

Follow-ups:
- Per-size thresholds (the 90M and 1.7B columns of the threshold table) instead of the pooled one, since one threshold per family does not hold across sizes.
- A pair-wise aggregate over the languages both runs train, so that a pair that changes L compares the same language set.

## Every population and view

The generated block below carries the four populations (the plan grid `deep_A_1904` and every build `1904`, each at the final and over the tenths) and their per-size totals.

<!-- BEGIN auto:chance-vs-train-tokens (tokens_seen.py --pool predictivity_seeds) -->
## Share of cells above chance against the tokens of the language seen

`tokens_seen.py`, `predictivity_seeds` pool: a language's evaluation against the training tokens of that language the model had seen (its share of the mixture from the build's plan × the cell's budget × the checkpoint's share of the run). Regenerate with `python analysis/rq01_scaling_predictability/tokens_seen.py --pool predictivity_seeds`.

**Share of a benchmark's cells above chance against the tokens seen.** One (task, size, L) cell per benchmark, language and language setting; above chance by rule 1's Wilson test on the cell's runs; cells binned 3 per decade of tokens, a point = the share of the bin's cells above chance with the cell count, one line per size. Two populations: `deep_A_1904`, the plan grid, deep / data A / seed 1904: one run per cell; `1904`, every seed-1904 run at the (size, L) that trains the language, every data build and ladder. The `.csv` next to each figure is the cell table (benchmark, task, language, model_size, language_scheme, train_tokens, task_score, above_chance, share_above, n_runs), `_points.csv` the binned values drawn. The `_ckpts` twins read the same runs at every evaluated tenth (a cell = (task, size, L, tenth), x = the tokens seen by that checkpoint): ten times the cells, and a token axis that runs through every training run. Each cell table is drawn per benchmark (`_by_benchmark_`), per language (`_by_language_`, panels from the best- to the least-resourced language) and pooled (`_all_`, one panel; their `.csv` is the binned points drawn). Cells above chance: `deep_A_1904` 8120 of 16164 (933 tasks), `deep_A_1904_ckpts` 76392 of 161640 (933 tasks), `1904` 9237 of 18120 (933 tasks), `1904_ckpts` 87055 of 181200 (933 tasks).

![Share above chance vs tokens seen, deep_A_1904](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904.csv)

![Share above chance vs tokens seen, deep_A_1904_ckpts](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904_ckpts.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904_ckpts.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_deep_A_1904_ckpts.csv)

![Share above chance vs tokens seen, 1904](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904.csv)

![Share above chance vs tokens seen, 1904_ckpts](pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.png)

[PNG on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.png) · [CSV on GitHub](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq00_chance_vs_train_tokens/pretraining/predictivity_seeds/pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.csv)

Key findings:
- `deep_A_1904`: 50% of 16164 cells above chance; per size 90M 41%, 175M 43%, 350M 46%, 600M 52%, 1B 57%, 1.7B 63%.
- `deep_A_1904_ckpts`: 47% of 161640 cells above chance; per size 90M 38%, 175M 40%, 350M 44%, 600M 48%, 1B 54%, 1.7B 59%.
- `1904`: 51% of 18120 cells above chance; per size 90M 41%, 175M 44%, 350M 48%, 600M 53%, 1B 57%, 1.7B 62%.
- `1904_ckpts`: 48% of 181200 cells above chance; per size 90M 40%, 175M 42%, 350M 45%, 600M 50%, 1B 54%, 1.7B 59%.
<!-- END auto:chance-vs-train-tokens -->

Follow-ups:
- The `_by_language_` figures with the token bins pooled across sizes, to separate the effect of exposure from the effect of size within a language.
- A twin restricted to the gate's surviving tasks, to check that the share rises with tokens on the tasks the later analyses keep.
- A twin over the reformulated (`rf_`) tasks alone, since the letter-choice originals of INCLUDE, Belebele and Global-MMLU stay at 12–23 % above chance even at 1.7B.
