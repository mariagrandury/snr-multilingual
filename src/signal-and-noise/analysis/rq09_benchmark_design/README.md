# Benchmark design: what makes a benchmark high-SNR?

## Research question

> Why do some multilingual benchmarks separate models cleanly under the SNR
> framework while others do not? Is it curation quality, task format, item
> length or the answer space?

<!-- BEGIN auto:highlight (analyze.py --pool predictivity) -->
## Highlighted result

- **The above-random gate decides which families are read at all.** Every family whose tasks sit at chance at the reference size is dropped before SNR is computed, leaving **41 families** that clear the gate (10 with 2 options, 23 with 4).
- **Among survivors, only option count and task format separates the families** (five Kruskal–Wallis tests on the same families, uncorrected): curation (H = 5.45, p = 0.364), source origin (H = 1.31, p = 0.520), option count (H = 15.76, p = 0.000), task format (H = 19.13, p = 0.002), passage flag (H = 0.00, p = 1.000). Option count, task format stays under 0.05 after a Bonferroni correction for the five.
- **Per-task curation test** (tasks as observations, 600 tasks of which 68 are `include_v2_en`): H = 92.89, p = 0.000. This is nominally significant, but the tasks of one family are not independent observations, so the test says which family dominates, not which curation works.
<!-- END auto:highlight -->

## Experimental setup

**Setup.**
- **Measured.** We relate the characteristics of each benchmark family to its SNR, DA-size, DA-ckpt and the R² of its scaling fit.
- **Models.** We use the seed-1904 runs of every language setting, ladder and data build, 90M–1.7B.
- **SNR.** SNR takes the mean pairwise distance between designs as its signal and is read at the 1.7B reference. A family's SNR is the median over its per-language tasks, for the 40 families that clear the above-chance gate.
- **DA-size.** DA-size compares each proxy's final checkpoint with the 1.7B final, on the mono-axis design pairs and the tasks above chance at both sizes. A family's DA-size is the median over its tasks at each proxy, averaged over the proxies 90M–1B at which it has a task (21 of the 29 reliable families have all five).
- **Characteristics.** We describe each family by its item count, skill, curation, source, task format, option count, reading passage, domain (what its own paper says it tests) and two item lengths (table: app_rq09_benchmark_characteristics). Skill splits the domains into LM-like (commonsense, grammaticality, language modelling) and the rest.
- **How.** We compute each characteristic's Spearman ρ with SNR and DA-size over families and over tasks, with a 95 % bootstrap interval over whole families (figures: app_rq09_design_da_correlation, app_rq09_design_da_by_level, app_rq09_design_da_quadrant). For item count we also fit the slope within each benchmark, across its languages and subtasks (table: app_rq09_item_count).
- **Robustness.** We compare the benchmarks of the sweep's first task list (2026-09-07) with the ones added later, each benchmark counted once (table: app_rq09_wave_robustness).

Pool `predictivity` is seed 1904 with every cell, data build and ladder (swiglu included). The 36-sweep pools are history, under [Extensions](#extensions-from-other-sweeps). The signal is `snr_mpd_1.7B` (mean pairwise distance, the headline variant of the noise and SNR analysis), read at the 1.7B reference only and after the above-random gate, so every family here has passed the gate.

A family's value is the median over its per-language aggregate tasks. We group the families by curation method, source origin, task format, option count and a reading-passage flag, and test each grouping with a family-level Kruskal–Wallis test (a group of one family is skipped). Of the 47 families tagged in `FAMILY_META` ([analyze.py](analyze.py), provenance in [data_info.md](data_info.md)), 40 keep at least one task, for 592 tasks over 50 languages. TruthfulQA-Multi and MultiBLiMP (extra) have no 1.7B SNR (MultiBLiMP (extra) has a DA-size), and AfriMMLU, AfriXNLI, Global-MMLU Lite, Global-PIQA completions and the LLM-rewritten Global-MMLU twin are not in the pool. Since 2026-10-10 every accuracy family of the `auto` group is tagged. The figure review found twelve families of the 2026-10-09 promotion (80 tasks with an SNR) silently left out, so `analyze.py` and `design_da.py` now print any family with a value and no entry in a `!!! RULE 13` line. The suites KoBEST, NorEval and NorEval (extra) mix formats, option counts and domains: those characteristics are `None` at the family level, and each subtask has its own domain (`SUBTASK_DOMAIN`). [data_info.md](data_info.md#families-added-on-2026-10-08) lists the 12 families added on 2026-10-08. The `bbpb_` twins stay out by design. A twin scores the same items as its original, so each of its characteristics would repeat the original's row (see [Inputs & caveats](#methodology)).

## Key figure

![Median SNR per benchmark family, coloured by answer-option count](pretraining/predictivity/snr_per_family_ranked_paper.png)

Population: pool `predictivity` (seed 1904), SNR `snr_mpd_1.7B` (mean pairwise distance at the 1.7B reference), median over each family's per-language tasks, only the 40 families (592 tasks, 50 languages) that clear the above-random gate. The number above a bar counts its tasks. Snapshot: the 2026-10-10 refresh.

**Key finding.** **Among the families with at least five tasks, MultiBLiMP, HellaSwag and XStoryCloze separate the designs most sharply at 1.7B, and seven of the nine sharpest families have two answer options.** The two highest bars rest on one task each, TokSuite math (3.19) and PIQA (2.79). Next come MultiBLiMP 2.39 (34 tasks), HellaSwag 2.29 (26), XStoryCloze 2.18 (8) and Global PIQA non-parallel 2.06 (2). The reformulated twins sit between 0.56 and 1.09, below every 2-option family except WinoGrande (0.78, one task).

GitHub: [snr_per_family_ranked_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_per_family_ranked_paper.png) · [snr_per_family_ranked_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_per_family_ranked_paper.csv). The other design axes and their tests: [Results](#results). The same characteristics against DA-size: [Benchmark characteristics against DA-size](#benchmark-characteristics-against-da-size).

Key findings:

- **Benchmarks with more items rank the designs more like the reference, within a benchmark as well as across benchmarks.** Within a benchmark, DA-size rises by 0.08 per tenfold more items (CI 0.04 to 0.11, 558 tasks, 19 benchmarks whose tasks differ in item count), and over the 39 benchmarks with a DA-size on every gated task the item count correlates with DA-size at ρ = 0.71 (CI 0.50 to 0.84). DA-ckpt follows it too (ρ = 0.76, within benchmarks 0.09 per tenfold), and so does the R² of the scaling fit within benchmarks (0.20 per tenfold, CI 0.15 to 0.24), but SNR does not (ρ = 0.01). The slope is 0.08 without the KoBEST and NorEval suites, whose subtasks differ in size (`item_count_da_all_mono_axis.csv`).
- **Skill is the second axis: LM-like benchmarks have a higher SNR and DA-size than the knowledge, reading and reasoning ones.** LM-like (commonsense, grammaticality, language modelling) correlates with SNR at ρ = 0.46 (CI 0.15 to 0.72, 38 benchmarks), with DA-size at 0.36 (CI 0.01 to 0.63) and with the R² of the scaling fit at 0.40. It adds to item count in a joint fit (DA-size +0.08, CI 0.04 to 0.12), and on the reliable tasks the LM-like tasks have a median DA-size of 0.72 against 0.60. KoBEST and NorEval mix both groups and are left out.
- **The benchmarks added after the sweep began decide like the original ones, but have a lower SNR.** Counting each benchmark once, the 12 original benchmarks have a median DA-size of 0.53 (CI 0.50 to 0.58) against 0.52 (0.49 to 0.55) for the 27 added later (Mann–Whitney p = 0.21), and a median R² of 0.89 against 0.86 (p = 0.19). Their median SNR is 1.29 against 0.82 (p = 0.04), and the gap holds after item count (p = 0.03). The later benchmarks also have fewer items (median 898 against 1,292).
- **The benchmark whose proxies decide most like 1.7B is also an SNR leader.** HellaSwag has the highest DA-size over every task above chance (0.75, mean over 90M–1B, tied with MultiBLiMP (extra) on three tasks read at two proxies, then PIQA 0.71 on one task and LAMBADA 0.64), and it is fourth by SNR (2.29). On the 23 of its 26 tasks that the paper's reliability filter keeps, it reaches 0.77. Only XCOPA and KoBEST are higher there (0.88 each), each on one task read at two proxies. The other 2-option SNR leaders reach only 0.47 to 0.60 over every task above chance (Global PIQA non-parallel 0.47, XCOPA 0.48, XWinograd 0.51, PAWS-X 0.52, MultiBLiMP 0.57 and XStoryCloze 0.60, in `design_da_size_per_family_mono_axis.csv`).
- **The reformulated twins do not lift their families into the top band.** Belebele has an SNR of 0.51 (3 tasks), against 0.61 for Belebele RF (57) and 0.83 for LLM-RF (59). INCLUDE has 0.66 (3), against 0.58 for RF (29) and 0.72 for LLM-RF (30). Global-MMLU RF has 0.84 (29), against 1.26 for Global-MMLU (1). The original Belebele is the second lowest family, above NorEval (0.41, 7 tasks).
- **The 2-option families sit above the 4-option ones.** The nine 2-option families have a median SNR of 1.77 (0.78 to 2.79), the 23 four-option ones 0.76 (0.51 to 3.19, with TokSuite math and HellaSwag the two above 2) and the two 5-option ones 0.75 (Kruskal–Wallis H = 11.09, p = 0.004, under 0.05 after a Bonferroni correction for the five axes, Spearman ρ = −0.58 with a 95 % interval of −0.76 to −0.30, 35 families). The 2-option families are all continuation, minimal-pair or classification formats with no lettered options, so format and option count cannot be told apart here. Option count does not carry over to DA-size (ρ = −0.20 on every gated task).
- **The same items have a higher SNR in their original language.** INCLUDE v2 in the item's own language reaches 0.89 (56 tasks), against 0.68 for the English translation of the same items (68). ARC in Okapi's ChatGPT translation (0.73, 22 tasks) also sits above the DeepL translation (0.64, 11).
- **Task counts differ by a factor of 68 across families (1 to 68).** The ten single-task medians (Global-MMLU and the English-only MMLU RF, CommonsenseQA RF, OpenBookQA, MathQA, TokSuite math, TokSuite STEM, PIQA, BoolQ and WinoGrande) and the two- and three-task ones (Global PIQA non-parallel, Belebele, INCLUDE and KoBEST) are the least stable bars.

Follow-ups:

- Bootstrap CIs on each family median, to show which bars the single-task families can be told apart from.
- The same ranking at each rung from 90M to 1.7B, to see whether the 2-vs-4-option gap is a reference-size effect.
- The option-count contrast at a fixed format (a 2-option and a 4-option continuation benchmark on the same items), since every 2-option family here is also a non-lettered format.

<!-- BEGIN auto:design-da (design_da.py --pool predictivity) -->
## Benchmark characteristics against DA-size

Pool `predictivity`: rq02's per-task DA-size (mono-axis pairs, each proxy's final checkpoint against the 1.7B final, gated at the proxy and at 1.7B, ≥ 3 pairs) over the paper's `above_66_either` tasks (22 families, up to 111 tasks), with the unfiltered reading (every gated task) beside it. A benchmark's DA-size is the median over its tasks at a proxy, averaged over 90M–1B, and a task's DA-size is its mean over the proxies that have one. Statistic: Spearman ρ against the centred code of an ordinal or binary characteristic (options 2 → 5, no passage → passage) or against a continuous one (the item count, the median `n_items` over a benchmark's languages, and the two lengths). Skill is knowledge, reading, reasoning → LM-like (`analyze.DOMAINS`), None for a suite whose subtasks fall in both groups. Curation, source, task format and domain are binarised: translated → native (written in the language, English originals included, or generated from treebanks or planning problems), translated from English → not, continuation → lettered MCQ, and every other domain → knowledge (the most populous of the eleven domains, chosen by its count). A characteristic that no source settles (`FAMILY_META` None: BBH's option count and passage, LAMBADA's option count) leaves that benchmark out of that row. BBH's curation method is unknown too, but its items were written in English, so the binary curation codes it native (`features`). The interval is a 95 % bootstrap over the families, which for the per-task readings resamples each benchmark with all of its tasks. The by-level figures group each characteristic's levels under a header row. The quadrant figures draw one dot per proxy, coloured by its size, with each size in a band of its own and its median marked. They read every task above chance, so benchmarks fall on both sides of chance agreement. Their `above_66_either` twins, where 19 of the 22 benchmarks sit right of 0.5, stay as analysis figures. Regenerate with `python analysis/rq09_benchmark_design/design_da.py --pool predictivity`.

![Characteristics against DA-size](pretraining/predictivity/design_da_size_correlation_above_66_either_mono_axis.png)

![DA-size by characteristic level, one dot per benchmark](pretraining/predictivity/design_da_size_by_level_above_66_either_mono_axis.png)

![DA-size by characteristic level, one dot per task](pretraining/predictivity/design_da_size_by_level_per_task_above_66_either_mono_axis.png)

![DA-size against the coded characteristics, one dot per benchmark and proxy, every task above chance](pretraining/predictivity/design_da_size_quadrant_mono_axis.png)

![DA-size against the coded characteristics, one dot per task and proxy, every task above chance](pretraining/predictivity/design_da_size_quadrant_per_task_mono_axis.png)

Key findings (generated):

- **DA-size, `above_66_either`** (mean over 90M–1B): Item count ρ = +0.08 [-0.36, +0.49] (n = 22), Skill ρ = +0.51 [+0.09, +0.78] (n = 22), Curation ρ = +0.08 [-0.35, +0.50] (n = 21), Source ρ = -0.06 [-0.52, +0.40] (n = 21), Answer options ρ = -0.32 [-0.71, +0.17] (n = 20), Reading passage ρ = +0.41 [+0.08, +0.70] (n = 21), Domain ρ = -0.37 [-0.70, +0.08] (n = 21), Context length ρ = -0.67 [-1.00, +0.10] (n = 8), Option length ρ = +0.76 [+0.09, +1.00] (n = 8).
- **DA-size, every gated task**: Item count ρ = +0.63 [+0.40, +0.81] (n = 39), Skill ρ = +0.39 [+0.04, +0.67] (n = 37), Curation ρ = -0.10 [-0.38, +0.22] (n = 37), Source ρ = -0.13 [-0.42, +0.21] (n = 37), Answer options ρ = -0.20 [-0.51, +0.16] (n = 34), Reading passage ρ = +0.00 [-0.39, +0.41] (n = 35), Domain ρ = +0.01 [-0.30, +0.34] (n = 36), Context length ρ = +0.24 [-0.80, +1.00] (n = 8), Option length ρ = +0.60 [-0.33, +1.00] (n = 8).
- **SNR at 1.7B** (the same statistics on rq09's family medians): Item count ρ = +0.10 [-0.21, +0.40] (n = 41), Skill ρ = +0.51 [+0.23, +0.73] (n = 39), Curation ρ = +0.23 [-0.08, +0.54] (n = 39), Source ρ = +0.18 [-0.14, +0.50] (n = 39), Task format ρ = -0.23 [-0.50, +0.03] (n = 38), Answer options ρ = -0.67 [-0.82, -0.44] (n = 36), Reading passage ρ = +0.00 [-0.35, +0.36] (n = 37), Domain ρ = -0.38 [-0.65, -0.08] (n = 38), Context length ρ = -0.43 [-0.95, +0.33] (n = 10), Option length ρ = +0.47 [-0.54, +0.94] (n = 10).
- **DA-size per task, `above_66_either`** (tasks as observations, the Kruskal–Wallis p treats them as independent, the interval does not): Item count ρ = +0.17 [-0.16, +0.36] (n = 113 tasks of 22), Skill ρ = +0.36 [+0.08, +0.49] (n = 113 tasks of 22, Kruskal–Wallis p = 0.000), Curation ρ = +0.07 [-0.17, +0.36] (n = 109 tasks of 21, Kruskal–Wallis p = 0.477), Source ρ = +0.01 [-0.22, +0.33] (n = 109 tasks of 21, Kruskal–Wallis p = 0.900), Answer options ρ = -0.11 [-0.38, +0.09] (n = 107 tasks of 20, Kruskal–Wallis p = 0.296), Reading passage ρ = +0.10 [-0.27, +0.35] (n = 109 tasks of 21, Kruskal–Wallis p = 0.312), Domain ρ = -0.15 [-0.32, +0.17] (n = 109 tasks of 21, Kruskal–Wallis p = 0.113), Context length ρ = -0.12 [-0.56, +0.25] (n = 53 tasks of 8), Option length ρ = +0.09 [-0.18, +0.51] (n = 53 tasks of 8).
- **DA-size per task, every gated task**: Item count ρ = +0.33 [+0.18, +0.47] (n = 560 tasks of 39), Skill ρ = +0.31 [+0.10, +0.46] (n = 553 tasks of 37, Kruskal–Wallis p = 0.000), Curation ρ = -0.01 [-0.22, +0.15] (n = 526 tasks of 37, Kruskal–Wallis p = 0.878), Source ρ = -0.10 [-0.29, +0.09] (n = 526 tasks of 37, Kruskal–Wallis p = 0.023), Answer options ρ = -0.15 [-0.27, +0.00] (n = 509 tasks of 34, Kruskal–Wallis p = 0.003), Reading passage ρ = -0.01 [-0.23, +0.34] (n = 514 tasks of 35, Kruskal–Wallis p = 0.744), Domain ρ = -0.02 [-0.24, +0.20] (n = 524 tasks of 36, Kruskal–Wallis p = 0.685), Context length ρ = -0.03 [-0.37, +0.33] (n = 117 tasks of 8), Option length ρ = +0.27 [-0.13, +0.59] (n = 117 tasks of 8).
- **Option count per proxy** (`above_66_either`): 90M ρ = -0.10, 175M ρ = -0.33, 350M ρ = -0.03, 600M ρ = -0.58, 1B ρ = -0.15.
- **Aggregating pulls the benchmarks toward chance** (DA-size − 0.5). Every gated task: 39 benchmarks median +0.01 (IQR -0.02 to +0.05, range -0.15 to +0.23), 64% within 0.05 of chance and 5% at DA-size ≥ 0.66. 560 tasks median +0.00 (IQR -0.06 to +0.07, range -0.50 to +0.50), 41% within 0.05 of chance and 9% at DA-size ≥ 0.66. `above_66_either`: 22 benchmarks median +0.10 (IQR +0.03 to +0.15, range -0.14 to +0.38), 23% within 0.05 of chance and 23% at DA-size ≥ 0.66. 113 tasks median +0.15 (IQR +0.05 to +0.25, range -0.25 to +0.50), 13% within 0.05 of chance and 42% at DA-size ≥ 0.66.
- **Domain levels** (SNR median over the families at 1.7B, then the median task DA-size on every gated task): grammaticality 4.03 (2 families, task DA-size 0.57 over 37 tasks), commonsense 1.61 (9 families, task DA-size 0.60 over 57 tasks), math 1.45 (2 families, task DA-size 0.53 over 2 tasks), paraphrase 1.41 (1 family, task DA-size 0.52 over 5 tasks), NLI 0.99 (1 family, task DA-size 0.50 over 13 tasks), reasoning 0.92 (2 families, task DA-size 0.42 over 15 tasks), language modelling 0.90 (1 family, task DA-size 0.58 over 5 tasks), culture 0.81 (1 family, task DA-size 0.31 over 5 tasks), knowledge 0.73 (15 families, task DA-size 0.51 over 268 tasks), reading comprehension 0.71 (4 families, task DA-size 0.47 over 117 tasks).

GitHub: [design_da_size_correlation_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_correlation_above_66_either_mono_axis.png) · [design_da_size_correlation_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_correlation_above_66_either_mono_axis.csv) · [design_da_size_by_level_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_by_level_above_66_either_mono_axis.png) · [design_da_size_by_level_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_by_level_above_66_either_mono_axis.csv) · [design_da_size_by_level_per_task_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_by_level_per_task_above_66_either_mono_axis.png) · [design_da_size_by_level_per_task_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_by_level_per_task_above_66_either_mono_axis.csv) · [design_da_size_quadrant_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_mono_axis.png) · [design_da_size_quadrant_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_mono_axis.csv) · [design_da_size_quadrant_per_task_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_per_task_mono_axis.png) · [design_da_size_quadrant_per_task_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_per_task_mono_axis.csv) · [design_da_size_quadrant_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_above_66_either_mono_axis.png) · [design_da_size_quadrant_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_above_66_either_mono_axis.csv) · [design_da_size_quadrant_per_task_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_per_task_above_66_either_mono_axis.png) · [design_da_size_quadrant_per_task_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_per_task_above_66_either_mono_axis.csv) · [design_da_size_per_family_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_per_family_mono_axis.csv) · [design_da_size_per_task_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_per_task_mono_axis.csv) · [benchmark_characteristics.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/benchmark_characteristics.csv)

Table: `benchmark_characteristics.csv` (`.tex` for the paper) lists each benchmark's characteristics.
<!-- END auto:design-da -->

Key findings (`design_da_size_correlation_above_66_either_mono_axis.csv`, `design_da_size_per_family_mono_axis.csv`, `design_da_size_per_task_mono_axis.csv`, with 29 families and 145 tasks in the `above_66_either` reading, 39 families and 558 tasks on every gated task, 40 families for the SNR, snapshot: the 2026-10-10 refresh):

- **Item count and skill are the only characteristics that order the benchmarks by DA-size, and item count only on every gated task.** On every gated task item count reaches ρ = 0.71 (CI 0.50 to 0.84) and skill 0.36 (CI 0.01 to 0.63). On the reliable tasks skill keeps 0.51 (CI 0.10 to 0.80), while item count falls to 0.18 (CI −0.21 to 0.57): those tasks are chosen by their DA, so the benchmarks left differ less. Option length reaches 0.76 there (CI 0.10 to 1.00), on the 8 original families with sampled items. No other interval excludes 0 in either reading. The [item count and skill](#item-count-and-skill) section has the within-benchmark slopes.
- **Read per task, the benchmarks are not all near chance, and skill separates the tasks.** A benchmark's DA-size is a median over its tasks, averaged over the proxies, and this pulls it toward 0.5. On every gated task, 59 % of the 39 benchmarks sit within 0.05 of chance (IQR −0.02 to +0.06 around it), against 35 % of the 558 tasks (IQR −0.06 to +0.09, range −0.50 to +0.50). On the reliable tasks, 24 % of the 29 benchmarks reach a DA-size of 0.66, against 43 % of the 145 tasks (median 0.64, IQR 0.55 to 0.72). There the LM-like tasks have a median of 0.72 against 0.60 (per-task ρ = 0.45, CI 0.16 to 0.58), and the levels of every other characteristic with at least five tasks lie between 0.61 and 0.67. On every gated task item count (ρ = 0.34, CI 0.19 to 0.47) and skill (0.31, CI 0.10 to 0.46) keep intervals that exclude 0. The option count and the source reach Kruskal–Wallis p = 0.001 and 0.013, but that test counts the tasks of one benchmark as independent, and their resampled intervals contain 0.
- **In the per-proxy quadrants, the median task sits at chance at every proxy size.** On every gated task the median (task, proxy) DA-size is 0.50 at 90M, 175M, 350M and 1B and 0.51 at 600M (2,247 dots, IQR 0.39 to 0.62, 12 % exactly at 0.5). The binary levels have medians of 0.50 to 0.61, the option counts 0.49 to 0.56, and every (level, proxy) band 0.45 to 0.62.
- **Knowledge benchmarks have a lower SNR, but not a lower DA-size.** We code the domain as knowledge (the most populous domain, 15 of the 40 SNR families) against the other domains. It correlates with SNR at ρ = −0.39 (CI −0.66 to −0.08): the knowledge families have a median SNR of 0.73, against 1.77 for the 9 commonsense ones. With DA-size, ρ is −0.23 on the reliable tasks and 0.00 on every gated task, and both intervals include 0. Per task on every gated task, commonsense reaches a DA-size of 0.59 (57 tasks), knowledge and NLI 0.50 (267 and 13), reading comprehension 0.47 (117) and the two reasoning benchmarks 0.43 (14).
- **Option count orders the families by SNR, but not by DA-size.** Its ρ with SNR has an interval that excludes 0 (ρ = −0.58, CI −0.76 to −0.30, 35 families). With DA-size it is −0.27 (CI −0.64 to 0.17) on the reliable tasks and −0.20 (CI −0.49 to 0.10) on every gated task.
- **One benchmark, HellaSwag, lifts every level it belongs to.** HellaSwag is a 4-option, passage-reading, machine-translated benchmark with 26 tasks. It is the top DA-size family on every gated task (0.75, tied with MultiBLiMP (extra) on three tasks) and third on the reliable tasks (0.77), so its levels of those characteristics are lifted by this one benchmark. The by-level figure labels the highest dot on the reliable tasks once per characteristic. XCOPA and KoBEST reach 0.88 there, each on one task read at two proxies.
- **Curation and source are binarised separately, and they no longer coincide.** Curation is translated against native, source is translated from English against not, and task format is continuation against lettered MCQ. INCLUDE v2 (EN) is a translation of natively sourced items, and the English-only benchmarks are native and untranslated. Task format cannot be tested in either DA reading: INCLUDE (one task, every gated task only) is the one lettered-MCQ family with a DA-size, because Belebele and Global-MMLU have no task above chance at both a proxy and 1.7B.
- **On every task above chance, 16 of the 39 families fall below 0.5 in the quadrant figure.** They fall on both levels of every characteristic, so no pair of quadrants dominates. The lowest are CulturalBench-easy RF 0.34, KoBEST 0.38, ACP-Bench RF 0.42 and TokSuite 0.45. On the reliable tasks 26 of the 29 sit right of 0.5. The three others are XNLI (0.48 on one task), TokSuite (0.46 on one) and INCLUDE v2 (EN) (0.38 on six): the filter keeps tasks with a median DA ≥ 0.66 on either axis, so a task can pass on DA-ckpt alone. That reliable-task version stays as an analysis figure. The per-benchmark quadrant draws one dot per proxy, 174 dots for the 39 benchmarks.
- **The paper shows the per-task figures.** These are the per-task by-level and quadrant figures (`design_da_size_by_level_per_task_above_66_either_mono_axis_paper.png`, `design_da_size_quadrant_per_task_mono_axis_paper.png`), chosen for the spread of the second finding. The per-benchmark points hide that spread behind a median, and with 29 or 39 points, a level of one to five benchmarks has no distribution to read. The per-benchmark figures stay on disk beside them.

Follow-ups:

- Length features for the twins and the 12 families added on 2026-10-08, so the length readings rest on more than eight families.
- BBH's option count and passage per subtask (2 to 8 options, 55 to 549-character stems), which the family-level table can only mark unknown.
- A mixed model of task DA-size with the benchmark as a random effect, the test the per-task Kruskal–Wallis approximates without the clustering.

<!-- BEGIN auto:item-count (design_da.py --pool predictivity) -->
## Item count and skill

Pool `predictivity`. A benchmark's item count is the median `n_items` (configs/tasks.json) over its per-language tasks, and its skill is the group of its domain (the table below): LM-like for commonsense, grammaticality and language modelling, against every other domain. A suite whose subtasks fall in both groups (KoBEST, NorEval) has no skill. Per metric, a benchmark's value is its DA-size on every gated task (as in the hollow dots above), the median over its tasks of DA-ckpt (each task's median over its DA-ckpt cells) and of the R² of rq01's power-law fit over size (pool `predictivity_seeds`), and its median SNR at 1.7B (log10 in the fits). The ρ are the Spearman correlations of the figure above with their bootstrap intervals over the benchmarks. Within benchmarks: an OLS of the task values on log10 items with a fixed effect per benchmark, the interval clustered by benchmark, so only the benchmarks whose tasks differ in item count inform it. Joint fit: a family-level OLS on log10 items and LM-like (HC1 interval). Regenerate with `python analysis/rq09_benchmark_design/design_da.py --pool predictivity`.

| domain | skill | benchmarks with a value | basis |
|---|---|---|---|
| commonsense | LM-like | Global PIQA (non-parallel), Global PIQA (parallel), HellaSwag, KoBEST `kobest_copa`, KoBEST `kobest_hellaswag`, NorEval `norcommonsenseqa`, PIQA, CommonsenseQA RF, WinoGrande, XCOPA, XStoryCloze, XWinograd | a plausible continuation, cause or referent (HellaSwag, COPA, Winograd schemas, PIQA, CommonsenseQA, KoBEST COPA and HellaSwag, NorCommonsenseQA) |
| grammaticality | LM-like | MultiBLiMP, MultiBLiMP (extra), NorEval `ncb` | minimal pairs of a grammatical and an ungrammatical sentence (MultiBLiMP, NCB) |
| language modelling | LM-like | LAMBADA | the last word of a passage (LAMBADA) |
| knowledge | knowledge, reading, reasoning | ARC, ARC (MT), Global-MMLU, INCLUDE, INCLUDE v2 (EN), INCLUDE v2 (OG), NorEval (extra) `noropenbookqa`, NorEval (extra) `nrk_quiz_qa`, OpenBookQA, Global-MMLU RF, INCLUDE RF, MMLU (Okapi) RF, MMLU RF, MMMLU RF, INCLUDE LLM-RF, TokSuite, TokSuite STEM | exam, quiz or science questions (MMLU and its translations, INCLUDE, ARC, OpenBookQA, NRK-Quiz-QA) and TokSuite's general-knowledge completions |
| reading comprehension | knowledge, reading, reasoning | Belebele, BoolQ, NorEval `norbelebele`, Belebele RF, Belebele LLM-RF | a question about a passage (Belebele, BoolQ, KoBEST BoolQ) |
| NLI | knowledge, reading, reasoning | XNLI | entailment between two sentences (XNLI, AfriXNLI) |
| paraphrase | knowledge, reading, reasoning | PAWS-X | whether two sentences mean the same (PAWS-X) |
| word sense | knowledge, reading, reasoning | none | whether a word has one sense in two sentences (KoBEST WiC) |
| sentiment | knowledge, reading, reasoning | KoBEST `kobest_sentineg`, NorEval (extra) `norec` | the polarity of a review or sentence (KoBEST SentiNeg, NoReC) |
| reasoning | knowledge, reading, reasoning | ACP-Bench (MCQ) RF, BBH (MCQ) RF | planning and multi-step reasoning (ACPBench, BBH) |
| math | knowledge, reading, reasoning | MathQA, TokSuite math | math word problems and arithmetic (MathQA, TokSuite math) |
| culture | knowledge, reading, reasoning | CulturalBench-easy RF | cultural knowledge of a region (CulturalBench) |
| truthfulness | knowledge, reading, reasoning | NorEval (extra) `nortruthfulqa`, TruthfulQA-Multi | imitative falsehoods (TruthfulQA, NorTruthfulQA) |

Item count and skill (generated, `item_count_da_all_mono_axis.csv`):

- **DA-size** (39 benchmarks): items ρ = +0.63 [+0.40, +0.81], LM-like ρ = +0.39 [+0.04, +0.67] (37 benchmarks with one skill). Within benchmarks +0.080 per tenfold items [+0.047, +0.113], p = 2.37e-06 (560 tasks, 19 benchmarks whose tasks differ in item count). Joint fit over 37 benchmarks: items +0.06 [+0.04, +0.09], LM-like +0.07 [+0.03, +0.10].
- **DA-ckpt** (42 benchmarks): items ρ = +0.74 [+0.57, +0.85], LM-like ρ = +0.23 [-0.16, +0.56] (40 benchmarks with one skill). Within benchmarks +0.079 per tenfold items [+0.048, +0.110], p = 5.61e-07 (610 tasks, 20 benchmarks whose tasks differ in item count). Joint fit over 40 benchmarks: items +0.08 [+0.05, +0.11], LM-like +0.05 [-0.00, +0.09].
- **SNR at 1.7B** (41 benchmarks): items ρ = +0.10 [-0.21, +0.40], LM-like ρ = +0.51 [+0.23, +0.73] (39 benchmarks with one skill). Within benchmarks +0.038 per tenfold items [-0.015, +0.092], p = 0.159 (599 tasks, 20 benchmarks whose tasks differ in item count). Joint fit over 39 benchmarks: items +0.02 [-0.08, +0.11], LM-like +0.26 [+0.11, +0.41].
- **R² over size** (37 benchmarks): items ρ = +0.19 [-0.14, +0.50], LM-like ρ = +0.41 [+0.10, +0.66] (35 benchmarks with one skill). Within benchmarks +0.187 per tenfold items [+0.148, +0.225], p = 4.3e-21 (389 tasks, 18 benchmarks whose tasks differ in item count). Joint fit over 35 benchmarks: items +0.05 [-0.01, +0.12], LM-like +0.09 [+0.01, +0.18].

Original against later-added benchmarks (generated, `wave_robustness_da_size_mono_axis.csv`; a benchmark counts once, its value as above; `analyze.WAVES` dates each family from configs/tasks.json):

- **DA-size**: original (2026-09-07) 0.52 [0.50, 0.58] (12 benchmarks, median 1292 items), later, all 0.51 [0.49, 0.53] (27 benchmarks, median 898 items), promoted 2026-09-23 0.49 [0.48, 0.55] (5 benchmarks, median 886 items), reformulated twins 0.50 [0.47, 0.53] (12 benchmarks, median 899 items), probes promoted 2026-10-09 0.53 [0.47, 0.59] (10 benchmarks, median 756 items). Original against later: Mann–Whitney p = 0.24. After log10 items the later benchmarks differ by -0.03 (p = 0.07).
- **SNR at 1.7B**: original (2026-09-07) 1.29 [0.71, 1.84] (14 benchmarks, median 1292 items), later, all 0.83 [0.73, 1.00] (27 benchmarks, median 898 items), promoted 2026-09-23 0.70 [0.55, 0.83] (5 benchmarks, median 886 items), reformulated twins 0.82 [0.70, 0.94] (12 benchmarks, median 899 items), probes promoted 2026-10-09 1.51 [0.89, 2.01] (10 benchmarks, median 756 items). Original against later: Mann–Whitney p = 0.22. After log10 items the later benchmarks differ by -0.08 (p = 0.29).
- **R² over size**: original (2026-09-07) 0.89 [0.84, 0.97] (11 benchmarks, median 1415 items), later, all 0.86 [0.83, 0.90] (26 benchmarks, median 892 items), promoted 2026-09-23 0.88 [0.72, 0.99] (5 benchmarks, median 886 items), reformulated twins 0.85 [0.83, 0.90] (12 benchmarks, median 899 items), probes promoted 2026-10-09 0.83 [0.63, 0.99] (9 benchmarks, median 613 items). Original against later: Mann–Whitney p = 0.18. After log10 items the later benchmarks differ by -0.07 (p = 0.08).

GitHub: [item_count_da_all_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/item_count_da_all_mono_axis.csv) · [item_count_da_all_mono_axis.tex](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/item_count_da_all_mono_axis.tex) · [wave_robustness_da_size_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/wave_robustness_da_size_mono_axis.csv) · [wave_robustness_da_size_mono_axis.tex](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/wave_robustness_da_size_mono_axis.tex) · [item_count_per_task_da_all_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/item_count_per_task_da_all_mono_axis.csv)
<!-- END auto:item-count -->

Key findings (`item_count_da_all_mono_axis.csv`, `wave_robustness_da_size_mono_axis.csv`, `item_count_per_task_da_all_mono_axis.csv`, snapshot: the 2026-10-10 refresh):

- **Item count is the design characteristic that tracks DA-size best, and it holds within a benchmark.** Across benchmarks it correlates with DA-size at ρ = 0.71 and with DA-ckpt at 0.76. Within a benchmark, where the languages or subtasks differ in size but the items are of one kind, DA-size rises by 0.08 per tenfold more items (CI 0.04 to 0.11) and DA-ckpt by 0.09. The slope is the same without the three suites (0.08, CI 0.04 to 0.12), so it is not a subtask effect. A benchmark with more items ranks two models with less sampling noise, which is what DA-size rewards.
- **SNR does not follow item count.** Across benchmarks ρ = 0.01, and within benchmarks the slope on log10 SNR is 0.02 (CI −0.03 to 0.06). SNR divides the spread between designs by the checkpoint noise, and both shrink with more items.
- **The R² of the scaling fit rises with item count within benchmarks (0.20 per tenfold, CI 0.15 to 0.24) but only weakly across them (ρ = 0.19, CI −0.14 to 0.51).** Skill orders it more clearly (ρ = 0.40).
- **Skill adds to item count.** In the joint fit over the benchmarks with one skill, LM-like adds 0.08 to DA-size (CI 0.04 to 0.12) and 0.24 to log10 SNR (CI 0.09 to 0.39), while item count keeps 0.07 per tenfold on DA-size. KoBEST and NorEval, whose subtasks fall in both groups, are left out of every skill reading.
- **The original benchmarks and the later ones decide alike, and the later ones have a lower SNR.** DA-size 0.53 against 0.52 (p = 0.21) and R² 0.89 against 0.86 (p = 0.19), each benchmark counted once. The SNR median is 1.29 for the 14 original benchmarks against 0.82 for the 26 later ones (p = 0.04, and p = 0.03 after log10 items). Among the later waves the 2026-10-09 probes have the highest SNR (1.00) and the 2026-09-23 promotion the lowest (0.68).

Follow-ups:

- The item-count slope on subsampled items of one task (DA-size of the same task at 100, 300 and 1,000 items), which separates the count from what else differs between languages.
- Skill per subtask inside the suites, so KoBEST and NorEval enter the skill readings task by task.

## Methodology

The analysis has three phases. All of them use the per-family
`snr_mpd_<reference>` signal, the median across a family's per-language
aggregate tasks, at the reference size (1.7B, or the largest bucket in the
table while 1.7B is missing):

- **Phase 0: curation process.** We group families by how their items were
  produced (machine translation / human translation / template generation /
  originally-multilingual authoring) and by source origin (English-translated
  vs originally-multilingual), and test with a family-level Kruskal–Wallis test.
- **Phase A: task format.** Axes: task format (`minimal_pair`, `completion`,
  `classification`, `cloze_completion`, `statement_continuation`, `last_word`,
  `mcq_question_only`, `mrc_passage`, read from the prompt the harness builds,
  `configs/tasks.json` `example`), answer-option count
  (`random_baseline = 1/n_options`, tested categorically 2 to 5 and
  continuously via Spearman), and a reading-passage flag.
- **Phase B: item lengths.** [length_features.py](length_features.py) pulls
  100 English/default items per family from each benchmark's HF dataset and
  computes character-length statistics for the context and the options,
  which we correlate with SNR.

**Mechanism (a hypothesis).** We hypothesize that reliability depends on the
*answer space* rather than on curation. A benchmark would be sharper when the
model compares **fewer, longer** completions scored by log-likelihood: each
extra option adds another noisy log-likelihood estimate to rank, and longer
options hold more tokens that tell the models apart. On the ladder one
illustration holds and one does not:

- HellaSwag (long 4-option completions, 2.29) sits above ARC (short noun-phrase options, 0.73), but PAWS-X (options `Yes`/`No`, ~2.5 chars) is the sixth-sharpest family (1.55, behind MultiBLiMP 2.39, HellaSwag 2.29, XStoryCloze 2.18, Global PIQA non-parallel 2.06 and XWinograd 1.77).
- Option length does not track sharpness significantly over the 10 original families with sampled items (option length ρ = 0.48, p = 0.16, and context length ρ = −0.48, p = 0.16, in `group_stats.csv`). The answer-space account therefore remains a hypothesis to test.

The `passage` flag itself does not separate families: XStoryCloze (4-sentence context, completion, 2.18) is high and Belebele (long passage, MRC, 0.51) is low. We suspect that what matters is what the prompt does with the passage.

**Inputs & caveats.** Per-family metadata is hand-curated in
[data_info.md](data_info.md) (paper-style paragraphs cross-referenced against the
lm-eval task READMEs). The `FAMILY_META` dict in [analyze.py](analyze.py) is its
machine-readable mirror, with a task-level `xnli_eu` override re-tagged
`mt_post_edited`. Length features exist only for the 10 original families with sampled items (`length_features.csv`), not for the twins, INCLUDE, Global PIQA or the 12 families added on 2026-10-08.

**Characteristics that no source settles are `None`, never guessed.** BBH (RF) mixes human-written and programmatically generated subtasks, 2 to 8 options and stems of 55 to 549 characters, so its curation method, option count and passage flag are unknown (its curation still binarises as native, because it is an English original). LAMBADA scores a last word, so it has no option count. INCLUDE v2 (EN) ships an English translation whose method the dataset does not document. A `None` drops the benchmark only from that characteristic's test and figure row.

**Domain and skill (added 2026-10-09, skill 2026-10-10).** Each family has one `domain` in `FAMILY_META`: what the benchmark tests, as its own paper or dataset card describes it, compared with the `categories` of `configs/multilingual_benchmarks.csv`. A twin (RF, LLM-RF, INCLUDE v2) keeps its original's domain, and a suite (KoBEST, NorEval, NorEval (extra)) has one domain per subtask (`SUBTASK_DOMAIN`). The taxonomy, its skill groups and the basis of each domain live in one place, `DOMAINS` in [analyze.py](analyze.py), and the [item count and skill](#item-count-and-skill) block prints them as a table. The DA-size and SNR tests code the domain as knowledge against the rest, because knowledge is the most populous level (18 of the 47 tagged families). The level is picked by its count, not by its result. Checked against the papers on 2026-10-10: TokSuite's canonical questions are simple general-knowledge completions ("The capital of France is"), so TokSuite is knowledge, not a linguistic test, and KoBEST's five subtasks are reading comprehension (BoolQ), commonsense (COPA, HellaSwag), sentiment (SentiNeg) and word sense (WiC).

Hand-written numbers in this README are from the ladder report **2026-10-10 04:31** and the outputs of the 2026-10-10 refresh (detrended checkpoint noise since 2026-10-08, which moved every `snr_mpd_1.7B`). FineTasks' selection criteria, judged on this ladder, are in the surrogates analysis ([README](../rq04_surrogates/README.md#finetasks-criteria-on-the-ladder)).

**Families without metadata are left out, and the scripts say so.** `load_per_task_snr` and `design_da.per_task` keep a family only if it has a `FAMILY_META` entry, and print every family they drop in a `!!! RULE 13` line. Until 2026-10-08 that left 12 accuracy families out, and until 2026-10-10 the twelve of the 2026-10-09 promotion. They are tagged now. The only families with a 1.7B SNR in the pool and no entry are the 14 `bbpb_` benchmark-BPB twins (plus per-language `bpb`, which is not a benchmark). A `bbpb_` twin scores its original's items with another metric, so each of its characteristics would repeat its original's row. Adding the twins would count each benchmark twice in every test.

<!-- BEGIN auto:results (analyze.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq09_benchmark_design/analyze.py --pool predictivity`.

**Per-family SNR ranking.** The median `snr_mpd_1.7B` over each family's per-language tasks, above-random survivors only:

| family | median SNR | n | format | n_opts |
|---|---|---|---|---|
| `multiblimp-extra` | 5.86 | 3 | minimal_pair | 2 |
| `toksuite_math` | 2.35 | 1 | completion | 4 |
| `multiblimp` | 2.21 | 34 | minimal_pair | 2 |
| `global_piqa_nonparallel_cloze` | 2.06 | 2 | completion | 2 |
| `boolq` | 2.01 | 1 | classification | 2 |
| `xstorycloze` | 1.97 | 8 | completion | 2 |
| `hellaswag` | 1.84 | 26 | completion | 4 |
| `xwinograd` | 1.79 | 6 | completion | 2 |
| `piqa` | 1.61 | 1 | completion | 2 |
| `toksuite` | 1.59 | 5 | completion | 4 |
| `winogrande` | 1.43 | 1 | completion | 2 |
| `paws` | 1.41 | 5 | classification | 2 |
| `xcopa` | 1.33 | 8 | completion | 2 |
| `global_mmlu_full` | 1.26 | 1 | mcq_question_only | 4 |
| `rf_mmlu` | 1.06 | 1 | cloze_completion | 4 |
| `kobest` | 1.03 | 2 | nan |  |
| `toksuite_stem` | 1.00 | 1 | completion | 4 |
| `xnli` | 0.99 | 14 | classification | 3 |
| `rf_commonsense_qa` | 0.97 | 1 | cloze_completion | 5 |
| `rf_acp_bench_mcq` | 0.96 | 5 | cloze_completion | 4 |
| `rf_mmmlu` | 0.93 | 12 | cloze_completion | 4 |
| `lambada_openai_mt` | 0.90 | 5 | last_word |  |
| `rf_bbh_mcq` | 0.89 | 10 | cloze_completion |  |
| `include_v2_og` | 0.83 | 55 | cloze_completion | 4 |
| `rfgm_belebele` | 0.82 | 59 | statement_continuation | 4 |
| `rf_global_mmlu_full` | 0.82 | 29 | cloze_completion | 4 |
| `rf_cultural_bench_easy` | 0.81 | 7 | cloze_completion | 4 |
| `noreval_extra` | 0.75 | 30 | nan |  |
| `openbookqa` | 0.73 | 1 | completion | 4 |
| `rfgm_include_base_44` | 0.72 | 30 | statement_continuation | 4 |
| `arc` | 0.71 | 22 | cloze_completion | 4 |
| `include_v2_en` | 0.70 | 68 | cloze_completion | 4 |
| `rf_m_mmlu` | 0.68 | 27 | cloze_completion | 4 |
| `include_base_44` | 0.67 | 2 | mcq_question_only | 4 |
| `arc_mt` | 0.62 | 11 | cloze_completion | 4 |
| `global_piqa_parallel_cloze` | 0.59 | 9 | completion | 4 |
| `rf_belebele` | 0.59 | 57 | cloze_completion | 4 |
| `rf_include_base_44` | 0.58 | 29 | cloze_completion | 4 |
| `mathqa` | 0.55 | 1 | cloze_completion | 5 |
| `noreval` | 0.53 | 7 | nan |  |
| `belebele` | 0.51 | 3 | mrc_passage | 4 |

![Per-family SNR ranking](pretraining/predictivity/snr_per_family_ranked.png)

**Significance of each design axis.** A family-level Kruskal–Wallis test over the survivors (the families the gate removed are not tested):

| axis | H | p |
|---|---|---|
| n_options | 15.76 | 0.00 |
| format | 19.13 | 0.00 |
| data source | 1.31 | 0.52 |
| curation method | 5.45 | 0.36 |
| reading passage | 0.00 | 1.00 |
<!-- END auto:results -->

[snr_per_family_ranked.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_per_family_ranked.png) ·
[snr_family.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_family.csv) ·
[group_stats.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/group_stats.csv) ·
[snr_by_n_options.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_by_n_options.png) ·
[snr_by_format.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_by_format.png) ·
[snr_by_curation_process.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_by_curation_process.png) ·
[snr_by_data_source.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_by_data_source.png) ·
[snr_by_passage.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_by_passage.png) ·
[snr_vs_length_features.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_vs_length_features.png)

Key findings (40 families, 592 tasks, `snr_mpd_1.7B`):

- **Option count separates the families.** The 9 two-option families have a median family SNR of 1.77, the 23 four-option ones 0.76 and the 2 five-option ones 0.75. The test gives H = 11.09, p = 0.004 over these 34 families, under 0.05 after a Bonferroni correction for the five axes (XNLI, the only 3-option family, is a group of one and is skipped, and LAMBADA, BBH and the three suites have no single count).
- **Task format is under 0.05 only before the correction.** Format gives H = 11.48, p = 0.022 (5 groups), above the Bonferroni cut of 0.01. Curation gives H = 2.43, p = 0.79 (6 groups), source origin H = 0.47, p = 0.79 (3 groups) and the reading passage H = 0.09, p = 0.76.
- **The per-task curation test measures one family, not a curation method.** It gives H = 83.18, p < 0.001 (592 tasks, 7 groups). But its top group, template-generated (median 2.29), is made of MultiBLiMP (34 tasks) and ACP-Bench RF (5). The other six groups have medians of 0.68 to 0.87.
- **No length feature reaches significance.** Over the 10 original families, context length gives ρ = −0.48 (p = 0.16), option length ρ = 0.48 (p = 0.16) and the context-to-option ratio ρ = −0.50 (p = 0.14).

Follow-ups:

- The option-count contrast within one family and format (an original against its reformulated twin, on the same items), the cleanest test the twins allow.
- Length features for the twins, INCLUDE, Global PIQA and the 12 families added on 2026-10-08, to put all 40 families on the length plot.

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

The family medians above, per language (`predictivity` pool). Regenerate with `python analysis/rq09_benchmark_design/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate". Each figure's table sits next to it under the same name.

![rq09 in one figure](pretraining/predictivity/highlights.png)

![SNR per benchmark and language](pretraining/predictivity/snr_family_by_language.png)
<!-- END auto:panels -->

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/highlights.csv) ·
[snr_family_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_family_by_language.png)

Population (both figures): pool `predictivity` (seed 1904), `log10` of `snr_mpd_1.7B` per (benchmark, language) task at the 1.7B reference, gate survivors only: 40 families, 588 per-language tasks (592 with the cross-language aggregates), 50 languages. Snapshot: the 2026-10-10 refresh. The two rank panels of `highlights.png` show the top and bottom eight of each list, so the 24 middle families and the 34 middle languages are not drawn.

Key findings (`highlights.csv`, `snr_family.csv`):

- **14 of the 40 families have a median log10 SNR above 0 (signal above noise).** The highest are TokSuite math 0.50 and PIQA 0.45 (one task each), then MultiBLiMP 0.38, HellaSwag 0.36 and XStoryCloze 0.34. Belebele is lowest (−0.29).
- **10 of 50 languages have a median log10 SNR above 0.** The highest are Bosnian 0.25 (1 task), Slovenian 0.22 (3) and Kazakh 0.10 (8), and the lowest are Malayalam −0.28 (4) and Lithuanian −0.26 (8). English, which now includes the English-only benchmarks (40 tasks), is at 0.01.
- **By construction method, template-generated benchmarks (MultiBLiMP and ACP-Bench RF) are highest.** They are at 0.36, against −0.06 for English originals and for machine-translated, −0.11 for MT-post-edited and human-translated and −0.12 for originally multilingual benchmarks. The one translation of unknown method (INCLUDE v2 (EN)) is the lowest, at −0.17.
- **No cell of the benchmark-by-language grid is grey.** `gated` is False for all 588 tasks in `snr_family.csv`, so a white cell is a (benchmark, language) pair with no 1.7B SNR.

Follow-ups:

- The per-language ranking restricted to languages with at least five tasks, since several extremes rest on very few tasks (Bosnian 1, Slovenian 3, Malayalam 4).
- The per-language ranking within one family with wide coverage (`rfgm_belebele`, 59 tasks), to separate a language effect from the set of benchmarks a language happens to have.

## TODO

- [ ] Bootstrap CIs on the per-family SNR medians and on each Kruskal–Wallis H.
- [ ] Recover statistical power: bring back the gated high-option families as a
      *separate* above-random-vs-gated contrast, rather than testing option
      count only among the (mostly 2-option) survivors.
- [ ] Disentangle the task-level curation confound (curation method is tied to
      format/option count) with a controlled within-format comparison.
- [ ] Finish Phase B length features (context/option length, ratio) and fold
      them into the design-axis significance table.
- [ ] Controlled within-format curation contrasts to test the "curation does not
      matter" claim. HellaSwag (MT) against XStoryCloze (human translation), both
      completion. ARC (MT) against Global-MMLU-Full (MT + post-edit), both 4-option
      MCQ from the same source family (this would also expose the subject-fragmentation
      effect: ARC's single domain against MMLU's ~57 subjects).

## Files

- `pretraining/<pool>/per_family_snr.csv`, `per_task_snr.csv`: SNR and design
  attributes per family / task.
- `…/group_stats.csv`: Kruskal–Wallis H/p for each design axis.
- `…/snr_by_*.png`: the SNR distribution by curation, format, option count,
  passage and data source, plus `snr_per_family_ranked.png`,
  `snr_vs_random_baseline.png` and `snr_vs_length_features.png`.
- `…/design_da_size_*_above_66_either_mono_axis.png/.csv` (+ `_paper`, and the `_per_task` twins of the
  by-level and quadrant figures), `design_da_size_per_family_mono_axis.csv`, `design_da_size_per_task_mono_axis.csv`,
  `benchmark_characteristics.csv/.tex`: from [design_da.py](design_da.py), the characteristics against DA-size,
  the per-family and per-task DA-size, and the characteristics table the paper prints.
- `…/item_count_da_all_mono_axis.csv/.tex`, `item_count_per_task_da_all_mono_axis.csv`,
  `wave_robustness_da_size_mono_axis.csv/.tex`: from [design_da.py](design_da.py), the item-count and skill
  table (family-level ρ, the within-benchmark slope, the joint fit) with its per-task input, and the original
  against later-added benchmarks (both tables the paper prints).
- `…/snr_per_family_ranked_paper_horizontal.png/.csv`: the family ranking with the families along x, the
  appendix page's headline figure ([analyze.py](analyze.py) `--paper`).

## Extensions from other sweeps

Everything below comes from the **36-model sweep** (2026-04…06, 4 sizes × 3
data mixtures × 3 seeds, pool `custom_swissai_hf`, reference **1B**, twelve
families on the 86-task old list) or from the **external tier**
(`all/external`: the public and reference models, 270M–70B, cross-model
dispersion with no mixture axis). The sweep's SNR is a three-mixture
dispersion at 1B under its fixed-margin gate, on families without the
reformulated twins. Its Kruskal–Wallis tests therefore replicate the analysis
on a different survivor set and are never rows of the ladder's table.

### External model-set tier (`all/external`, 36-sweep)

Re-running the design-axis analysis on the `external` tier (cross-model
dispersion over the 270M…70B external ladder) is the sharpest test of the
answer-space mechanism. The capable external models clear the above-random
gate on the 4-option translated MCQA that the custom pool drops. The outputs
are in `all/external/`. Regenerate them with
`python analysis/rq09_benchmark_design/analyze.py --pool external`.

The above-random gate now passes **11 families, including the 4-option ones**
(`hellaswag`, `global_mmlu_full`, `arc`, `belebele`) that the custom pool gated
out. The survivor set is no longer almost all 2-option, and the answer-count
penalty disappears entirely.

**Per-family SNR ranking.** The median per-family SNR over each family's
per-language tasks, above-random survivors only. The highest-SNR family is now
the **4-option** `hellaswag`, and the 4-option families (★) are spread through
the ranking instead of clustered at the bottom:

| family | median SNR | n | format | n_opts |
|---|---|---|---|---|
| `hellaswag` ★ | **5.98** | 4 | completion | 4 |
| `paws` | 4.69 | 2 | classification | 2 |
| `xwinograd` | 4.61 | 4 | completion | 2 |
| `global_mmlu_full` ★ | 4.58 | 1 | mcq_question_only | 4 |
| `xstorycloze` | 2.61 | 5 | completion | 2 |
| `multiblimp` | 2.59 | 7 | minimal_pair | 2 |
| `arc` ★ | 2.30 | 2 | mcq_question_only | 4 |
| `xcopa` | 1.87 | 4 | completion | 2 |
| `belebele` ★ | 1.68 | 3 | mrc_passage | 4 |
| `xnli` | 0.69 | 6 | classification | 3 |
| `global_piqa_completions` | 0.49 | 5 | completion | 2 |

![Per-family SNR ranking (external)](all/external/snr_per_family_ranked.png)

A 4-option benchmark with long completions at the very top is what the
mechanism predicts. On this reading, option count hurts only when the model is
too weak to clear chance (the above-random gate), not by itself. What matters
is comparing few, *long*, information-dense completions.

**Significance of each design axis.** Family-level Kruskal–Wallis. With the
4-option families back in the survivor set, option count loses what little
predictive power it had in the custom pool (H 1.78 → **0.05**), and no design
axis is significant:

| axis | H | p |
|---|---|---|
| n_options | **0.05** | 0.83 |
| format | 0.00 | 1.00 |
| data source | 0.17 | 0.68 |
| curation method | 1.44 | 0.49 |
| reading passage | 0.38 | 0.54 |

![SNR by answer-option count (external), no penalty](all/external/snr_by_n_options.png)

**Paper-level claim.** Comparing the custom and external survivor sets isolates the
confound. The apparent "fewer options ⇒ higher SNR" effect in the custom pool is an
artifact of the above-random gate, which depends on model capability, and not a
property of benchmark design. Once capable models clear the gate, answer-option
count carries no signal, and the sharpest benchmarks span both 2- and 4-option
formats.

[snr_per_family_ranked.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/all/external/snr_per_family_ranked.png) ·
[snr_by_n_options.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/all/external/snr_by_n_options.png)

### Results from the 36-model sweep (2026-06, superseded)

The numbers below were generated on the 36-model sweep (4 sizes × 3 data mixtures × 3 seeds, 12 languages, pool `custom_swissai_hf` unless stated) and are kept as history; the predictivity ladder regenerates the blocks above.

#### Highlighted result

- **The answer-count penalty lives in the above-random gate, upstream of SNR.** Every at-chance 4-option *translated knowledge* MCQA (`belebele`, `global_mmlu_full`, `truthfulqa`) is dropped before SNR is computed, leaving **9 families** that clear the gate — most of them 2-option.
- **Among survivors, no single design feature is individually significant.** Family-level Kruskal–Wallis on option count is **H = 1.78, p = 0.18** and on task format **H = 0.00, p = 1.00** — too little variation left (mostly 2-option) to resolve them.
- **Curation method explains nothing** — family-level Kruskal–Wallis on curation is **H = 0.50, p = 0.78**. Once the gate fixes the answer space, how a benchmark was built does not predict its reliability.

#### Results

Headline numbers from the `custom_swissai_hf` pool. Regenerate with `python analysis/rq09_benchmark_design/analyze.py --pool custom_swissai_hf`.

**Per-family SNR ranking** — median `snr_mpd_1B` over each family's per-language tasks, above-random survivors only:

| family | median SNR | n | format | n_opts |
|---|---|---|---|---|
| `multiblimp` | 3.85 | 7 | minimal_pair | 2 |
| `paws` | 2.55 | 2 | classification | 2 |
| `xwinograd` | 2.48 | 4 | completion | 2 |
| `xstorycloze` | 2.27 | 5 | completion | 2 |
| `xcopa` | 2.06 | 4 | completion | 2 |
| `hellaswag` | 2.05 | 4 | completion | 4 |
| `arc` | 2.05 | 2 | mcq_question_only | 4 |
| `global_piqa_completions` | 1.45 | 5 | completion | 2 |
| `xnli` | 1.15 | 7 | classification | 3 |

![Per-family SNR ranking](pretraining/custom_swissai_hf/snr_per_family_ranked.png)

**Significance of each design axis** — family-level Kruskal–Wallis over the survivors (high-option families already removed by the gate):

| axis | H | p |
|---|---|---|
| n_options | 1.78 | 0.18 |
| format | 0.00 | 1.00 |
| data source | 0.60 | 0.44 |
| curation method | 0.50 | 0.78 |
| reading passage | 0.00 | 1.00 |

[snr_per_family_ranked.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/custom_swissai_hf/snr_per_family_ranked.png)
