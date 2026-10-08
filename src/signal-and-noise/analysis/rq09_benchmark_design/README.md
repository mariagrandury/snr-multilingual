# Benchmark design — what makes a benchmark high-SNR?

## Research question

> Why do some multilingual benchmarks separate models cleanly under the SNR
> framework and others don't? Is it curation quality, task format, item
> length, or the answer space?

<!-- BEGIN auto:highlight (analyze.py --pool predictivity) -->
## Highlighted result

- **The above-random gate decides which families are read at all.** Every family whose tasks sit at chance at the reference size is dropped before SNR is computed, leaving **17 families** that clear the gate — most of them 2-option.
- **Among survivors, no family-level design axis separates the families convincingly** (five Kruskal–Wallis tests on the same families, uncorrected): curation H = 2.45, p = 0.485; source origin H = 0.49, p = 0.482; option count H = 3.81, p = 0.051; task format H = 2.46, p = 0.293; passage flag H = 0.01, p = 0.916. Too little variation is left among the survivors (mostly 2-option) to resolve any axis.
- **Per-task curation test** (tasks as observations, 344 tasks of which 59 are `rfgm_belebele`): H = 93.27, p = 0.000 — nominally significant, but the tasks of one family are not independent observations, so it says which family dominates, not which curation works.
<!-- END auto:highlight -->

## Experimental setup

Pool `predictivity`: seed 1904, every cell, data build and ladder (swiglu included); the 36-sweep pools are history under [Extensions](#extensions-from-other-sweeps). The signal is `snr_mpd_1.7B` (mean pairwise distance, the noise-and-SNR analysis's headline variant) at the 1.7B reference only, after the above-random gate, so every family here is a gate survivor.

Each family is the median over its per-language aggregate tasks, grouped by curation method, source origin, task format, option count and a reading-passage flag, and tested with a family-level Kruskal–Wallis (groups of one family skipped). Of the 23 families tagged in `FAMILY_META` ([analyze.py](analyze.py), provenance in [data_info.md](data_info.md)), 17 keep at least one task: 344 tasks over 49 languages; TruthfulQA-Multi has no 1.7B SNR, and AfriMMLU, AfriXNLI, Global-MMLU Lite, Global-PIQA completions and the LLM-rewritten Global-MMLU twin are not in the pool.

## Key figure

![Median SNR per benchmark family, coloured by answer-option count](pretraining/predictivity/snr_per_family_ranked_paper.png)

Population: pool `predictivity` (seed 1904), SNR `snr_mpd_1.7B` (mean pairwise distance at the 1.7B reference), median over each family's per-language tasks, the 17 families (344 tasks, 49 languages) that clear the above-random gate only; snapshot: the 2026-10-08 refresh.

**Key finding.** **No family-level design characteristic separates the families convincingly, by SNR or by DA-size.** The four sharpest families by SNR are MultiBLiMP 2.39 (34 tasks), HellaSwag 2.29 (26), XStoryCloze 2.18 (8) and XWinograd 1.77 (6), and the reformulated twins fill the lower half (0.58 to 0.84).

GitHub: [snr_per_family_ranked_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_per_family_ranked_paper.png) · [snr_per_family_ranked_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_per_family_ranked_paper.csv). The other design axes and their tests: [Results](#results); the same characteristics against DA-size: [Benchmark characteristics against DA-size](#benchmark-characteristics-against-da-size).

Key findings:

- **The benchmark whose proxies decide most like 1.7B is now also an SNR leader.** HellaSwag has the highest DA-size on every task above chance (0.75 mean over 90M–1B; 0.77 on the 23 of its 26 tasks that the paper's reliability filter keeps, second there to XCOPA's single task read at two proxies, 0.88) and sits second by SNR (2.29); the 2-option SNR leaders read 0.51–0.60 on every task above chance (PAWS-X 0.52, XWinograd 0.51, MultiBLiMP 0.57, XStoryCloze 0.60; `design_da_size_per_family_mono_axis.csv`).
- **The reformulated twins do not lift their families into the top band.** Belebele 0.51 (3 tasks) vs Belebele RF 0.61 (57) and LLM-RF 0.83 (59); INCLUDE 0.66 (3) vs RF 0.58 (29) and LLM-RF 0.72 (30); Global-MMLU RF 0.84 (29) vs Global-MMLU 1.26 (1), and the original Belebele is the lowest family.
- **The 2- and 3-option families mostly sit above the 4-option ones by SNR.** They read 1.01 to 2.39 against 0.51 to 1.26, with HellaSwag (4 options, 2.29) and Global PIQA parallel (2 options, 0.59 on 10 tasks) the exceptions; the option-count test just misses 0.05 (H = 3.81, p = 0.051, one of five uncorrected tests), and option count does not carry over to DA-size (ρ = −0.06 on every gated task).
- **Task counts differ by a factor of 59 across families (1 to 59).** The single-task median (Global-MMLU) and the three-task ones (Belebele, INCLUDE) are the least stable bars.

Follow-ups:

- Bootstrap CIs on each family median, to show which bars the single-task families can be told apart from.
- The same ranking at each rung from 90M to 1.7B, to see whether the 2-vs-4-option gap is a reference-size effect.

<!-- BEGIN auto:design-da (design_da.py --pool predictivity) -->
## Benchmark characteristics against DA-size

Pool `predictivity`: rq02's per-task DA-size (mono-axis pairs, each proxy's final checkpoint against the 1.7B final, gated at the proxy and at 1.7B, ≥ 3 pairs) over the paper's `above_66_either` tasks (13 families, up to 85 tasks), with the unfiltered reading (every gated task) beside it; a benchmark's DA-size is the median over its tasks at a proxy and the mean of those medians over 90M–1B. Statistic: Spearman ρ against the centred code of an ordinal or binary characteristic (options 2 → 4, no passage → passage, translated → originally multilingual) or a length; curation and task format are binarised, translated → native (written in the language or generated from its treebanks) and continuation → lettered MCQ, and on these families native curation picks the same benchmarks as an originally multilingual source, so those two rows coincide; 95 % bootstrap over the families. The quadrant figure reads every task above chance, so benchmarks fall on both sides of chance agreement; its `above_66_either` twin, where every benchmark sits right of 0.5, stays as an analysis figure. Regenerate with `python analysis/rq09_benchmark_design/design_da.py --pool predictivity`.

![Characteristics against DA-size](pretraining/predictivity/design_da_size_correlation_above_66_either_mono_axis.png)

![DA-size by characteristic level](pretraining/predictivity/design_da_size_by_level_above_66_either_mono_axis.png)

![DA-size against the coded characteristics, every task above chance](pretraining/predictivity/design_da_size_quadrant_mono_axis.png)

Key findings (generated):

- **DA-size, `above_66_either`** (mean over 90M–1B): Curation ρ = +0.09 [-0.48, +0.63] (n = 13); Source ρ = +0.09 [-0.48, +0.63] (n = 13); Answer options ρ = -0.07 [-0.64, +0.63] (n = 13); Reading passage ρ = +0.18 [-0.36, +0.70] (n = 13); Context length ρ = -0.67 [-1.00, +0.11] (n = 8); Option length ρ = +0.76 [+0.17, +1.00] (n = 8).
- **DA-size, every gated task**: Curation ρ = -0.16 [-0.62, +0.42] (n = 15); Source ρ = -0.16 [-0.62, +0.42] (n = 15); Task format ρ = +0.05 (n = 15); Answer options ρ = -0.06 [-0.57, +0.47] (n = 15); Reading passage ρ = +0.17 [-0.50, +0.70] (n = 15); Context length ρ = +0.31 [-0.60, +0.96] (n = 8); Option length ρ = +0.50 [-0.33, +0.94] (n = 8).
- **SNR at 1.7B** (the same statistics on rq09's family medians): Curation ρ = -0.18 [-0.69, +0.38] (n = 17); Source ρ = -0.18 [-0.69, +0.38] (n = 17); Task format ρ = -0.34 [-0.68, +0.09] (n = 17); Answer options ρ = -0.52 [-0.88, +0.00] (n = 17); Reading passage ρ = -0.03 [-0.58, +0.53] (n = 17); Context length ρ = -0.48 [-0.94, +0.30] (n = 10); Option length ρ = +0.48 [-0.51, +1.00] (n = 10).
- **Option count per proxy** (`above_66_either`): 90M ρ = +0.35, 175M ρ = -0.03, 350M ρ = +0.15, 600M ρ = -0.54, 1B ρ = +0.15.

GitHub: [design_da_size_correlation_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_correlation_above_66_either_mono_axis.png) · [design_da_size_correlation_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_correlation_above_66_either_mono_axis.csv) · [design_da_size_by_level_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_by_level_above_66_either_mono_axis.png) · [design_da_size_by_level_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_by_level_above_66_either_mono_axis.csv) · [design_da_size_quadrant_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_mono_axis.png) · [design_da_size_quadrant_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_mono_axis.csv) · [design_da_size_quadrant_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_above_66_either_mono_axis.png) · [design_da_size_quadrant_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_above_66_either_mono_axis.csv) · [design_da_size_per_family_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_per_family_mono_axis.csv) · [benchmark_characteristics.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/benchmark_characteristics.csv)

Table: `benchmark_characteristics.csv` (`.tex` for the paper) lists each benchmark's characteristics.
<!-- END auto:design-da -->

Key findings (`design_da_size_correlation_above_66_either_mono_axis.csv`, `design_da_size_per_family_mono_axis.csv`; 13 families in the `above_66_either` reading, 15 on every gated task, 17 for the SNR):

- **Only option length orders the benchmarks by DA-size, and only on the reliable tasks.** Its ρ = 0.76 (CI 0.17 to 1.00) rests on the 8 original families with sampled items and drops to 0.50 (CI −0.33 to 0.94) on every gated task; every other signed interval crosses 0 in both readings, and option count, whose SNR interval now reaches 0 (ρ = −0.52, CI −0.88 to 0.00), is −0.07 on the reliable tasks and −0.06 on every gated task.
- **HellaSwag is the clean example.** A 4-option, passage-reading, machine-translated benchmark, it is the top DA-size family on every gated task (0.75) and second on the reliable tasks (0.77), so the levels it carries are lifted by one benchmark; the outlined dot in the by-level figure.
- **Curation and task format are binarised** (native vs translated, lettered MCQ vs continuation). On these families native curation picks exactly the benchmarks with an originally multilingual source, so the two rows coincide (ρ = 0.09 on the reliable tasks, −0.16 on every task above chance). ARC is the only lettered-MCQ family among the reliable tasks, so task format has no value there; on every task above chance (ARC and INCLUDE) it is 0.05.
- **The quadrant figure reads every task above chance, where six families fall below 0.5.** They are XCOPA 0.48, INCLUDE LLM-RF 0.48, INCLUDE 0.49, Belebele RF 0.49, Belebele LLM-RF 0.49 and XNLI 0.49, on both levels of every characteristic (task format too: INCLUDE is lettered MCQ), so no pair of quadrants dominates. On the reliable tasks every family but XNLI (0.48, one task) sits right of 0.5 (the filter keeps tasks with a median DA ≥ 0.66 on one axis); that twin stays as an analysis figure.

Follow-ups:

- Length features for the twins, so the strongest DA-size reading (option length) is read on more than eight families.

## Methodology

Three phases, all on the per-family `snr_mpd_<reference>` signal (the reference
size, 1.7B; the largest bucket in the table while it is missing) (median across a
family's per-language aggregate tasks):

- **Phase 0 — curation process.** Group families by how their items were
  produced (machine translation / human translation / template generation /
  originally-multilingual authoring) and by source origin (English-translated
  vs originally-multilingual); test with a family-level Kruskal–Wallis.
- **Phase A — task format.** Axes: task format (`minimal_pair`, `completion`,
  `classification`, `mcq_question_only`, `mrc_passage`), answer-option count
  (`random_baseline = 1/n_options`, tested categorically 2-vs-4 and
  continuously via Spearman), and a reading-passage flag.
- **Phase B — item lengths.** [length_features.py](length_features.py) pulls
  100 English/default items per family from each benchmark's HF dataset and
  computes character-length statistics for context vs options; correlate with SNR.

**Mechanism (a hypothesis).** Reliability tracks the *answer space*, not
curation: a benchmark is sharper when the model compares **fewer, longer**
log-likelihood-scored completions, because each extra option adds another noisy LL
estimate to rank and longer options concentrate more discriminating tokens. On the ladder one illustration holds and one does not:

- HellaSwag (long 4-option completions, 2.29) sits above ARC (short noun-phrase options, 0.73), but PAWS-X (options `Yes`/`No`, ~2.5 chars) is the fifth-sharpest family (1.55, behind MultiBLiMP 2.39, HellaSwag 2.29, XStoryCloze 2.18 and XWinograd 1.77).
- Option length does not track sharpness significantly over the 10 original families with sampled items (Spearman ρ = 0.48, p = 0.16; context length ρ = −0.48, p = 0.16; `group_stats.csv`), so the answer-space account is a hypothesis to test, not a finding.

The `passage` flag itself does not separate families: XStoryCloze (4-sentence context, completion, 2.18) is high and Belebele (long passage, MRC, 0.51) is low. What the prompt does with the passage is the candidate explanation.

**Inputs & caveats.** Per-family metadata is hand-curated in
[data_info.md](data_info.md) (paper-style paragraphs cross-referenced against the
lm-eval task READMEs); the `FAMILY_META` dict in [analyze.py](analyze.py) is its
machine-readable mirror, with a task-level `xnli_eu` override re-tagged
`mt_post_edited`. Length features exist only for the 10 original families with sampled items (`length_features.csv`), not for the twins, INCLUDE or Global PIQA.

Hand-written numbers in this README are from the ladder report **2026-10-08 12:06** (regenerated locally: acc_norm on the cloze-format originals) and the outputs of the 2026-10-08 refresh (detrended checkpoint noise, so every `snr_mpd_1.7B` moved). FineTasks' selection criteria, judged on this ladder, live in the surrogates analysis ([README](../rq04_surrogates/README.md#finetasks-criteria-on-the-ladder)).

**Families without metadata are left out.** `load_per_task_snr` keeps a family only if it has a `FAMILY_META` entry. As of 2026-10-08, 26 benchmark families with a 1.7B SNR in the pool have none, so they are in no table here: the 14 `bbpb_` benchmark-BPB twins, `arc_mt`, `global_piqa_nonparallel_cloze`, `include_v2_en`, `include_v2_og`, `lambada_openai_mt`, `mathqa`, `openbookqa` and the `rf_` twins of acp_bench, bbh, commonsense_qa, cultural_bench_easy and mmlu (plus per-language `bpb`, which is not a benchmark).

<!-- BEGIN auto:results (analyze.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq09_benchmark_design/analyze.py --pool predictivity`.

**Per-family SNR ranking** — median `snr_mpd_1.7B` over each family's per-language tasks, above-random survivors only:

| family | median SNR | n | format | n_opts |
|---|---|---|---|---|
| `multiblimp` | 2.39 | 34 | minimal_pair | 2 |
| `hellaswag` | 2.29 | 26 | completion | 4 |
| `xstorycloze` | 2.18 | 8 | completion | 2 |
| `xwinograd` | 1.77 | 6 | completion | 2 |
| `paws` | 1.55 | 5 | classification | 2 |
| `xcopa` | 1.32 | 8 | completion | 2 |
| `global_mmlu_full` | 1.26 | 1 | mcq_question_only | 4 |
| `xnli` | 1.01 | 14 | classification | 3 |
| `rf_global_mmlu_full` | 0.84 | 29 | cloze_completion | 4 |
| `rfgm_belebele` | 0.83 | 59 | statement_continuation | 4 |
| `arc` | 0.73 | 22 | mcq_question_only | 4 |
| `rfgm_include_base_44` | 0.72 | 30 | statement_continuation | 4 |
| `include_base_44` | 0.66 | 3 | mcq_question_only | 4 |
| `rf_belebele` | 0.61 | 57 | cloze_completion | 4 |
| `global_piqa_parallel_cloze` | 0.59 | 10 | completion | 2 |
| `rf_include_base_44` | 0.58 | 29 | cloze_completion | 4 |
| `belebele` | 0.51 | 3 | mrc_passage | 4 |

![Per-family SNR ranking](pretraining/predictivity/snr_per_family_ranked.png)

**Significance of each design axis** — family-level Kruskal–Wallis over the survivors (high-option families already removed by the gate):

| axis | H | p |
|---|---|---|
| n_options | 3.81 | 0.05 |
| format | 2.46 | 0.29 |
| data source | 0.49 | 0.48 |
| curation method | 2.45 | 0.48 |
| reading passage | 0.01 | 0.92 |
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

Key findings (17 families, 344 tasks, `snr_mpd_1.7B`):

- **Option count comes closest but misses 0.05.** The 6 two-option families have a median family SNR of 1.66 and the 10 four-option families 0.73; H = 3.81, p = 0.051 over these 16 (XNLI, the only 3-option family, is a group of one and skipped).
- **No other family-level axis comes near.** Curation H = 2.45, p = 0.48 (4 groups); source origin H = 0.49, p = 0.48; format H = 2.46, p = 0.29 (3 groups); reading passage H = 0.01, p = 0.92.
- **The per-task curation test measures one family, not a curation method.** It gives H = 93.27, p = 2.7e-19 (344 tasks, 5 groups), but its top group, template-generated (median 2.39), is MultiBLiMP alone (34 tasks), against 0.68 to 1.36 for the other four.
- **No length feature reaches significance.** Over the 10 original families, context length ρ = −0.48 (p = 0.16), option length ρ = 0.48 (p = 0.16), context-to-option ratio ρ = −0.50 (p = 0.14).

Follow-ups:

- The option-count contrast within one family and format (an original vs its reformulated twin, same items), the cleanest test the twins allow.
- Length features for the five twins, INCLUDE and Global PIQA (parallel), to put all 17 families on the length plot.

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

The family medians above, per language (`predictivity` pool). Regenerate with `python analysis/rq09_benchmark_design/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq09 in one figure](pretraining/predictivity/highlights.png)

![SNR per benchmark and language](pretraining/predictivity/snr_family_by_language.png)
<!-- END auto:panels -->

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/highlights.csv) ·
[snr_family_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_family_by_language.png)

Population (both figures): pool `predictivity` (seed 1904), `log10` of `snr_mpd_1.7B` per (benchmark, language) task at the 1.7B reference, gate survivors only: 17 families, 344 tasks, 49 languages; snapshot: the 2026-10-08 refresh. The two rank panels of `highlights.png` show the top and bottom eight of each list, so the 17th family, `rf_global_mmlu_full` (the median one, log10 −0.07), and the 33 middle languages are not drawn.

Key findings (`highlights.csv`, `snr_family.csv`):

- **Eight of the 17 families have a median log10 SNR above 0 (signal above noise).** They are MultiBLiMP 0.38, HellaSwag 0.36, XStoryCloze 0.34, XWinograd 0.25, PAWS-X 0.19, XCOPA 0.12, Global-MMLU 0.10 and XNLI 0.01; Belebele is lowest (−0.29).
- **12 of 49 languages have a median log10 SNR above 0.** The highest are Slovenian 0.22 (3 tasks), Croatian 0.16 (7) and English 0.15 (12), and the lowest Malayalam −0.35 (3) and Korean −0.30 (5).
- **By construction method, template-generated (MultiBLiMP alone) is highest.** It is at 0.38 against 0.13 for machine-translated, −0.07 for MT-post-edited, −0.12 for human-translated and −0.17 for originally multilingual, the lowest.
- **No cell of the benchmark-by-language grid is grey.** `gated` is False for all 344 tasks in `snr_family.csv`, so a white cell is a (benchmark, language) pair with no 1.7B SNR.

Follow-ups:

- The per-language ranking restricted to languages with at least five tasks, since several extremes (Slovenian and Malayalam, 3 tasks each) rest on very few.
- The per-language ranking within one family with wide coverage (`rfgm_belebele`, 59 tasks), to separate a language effect from which benchmarks a language happens to have.

## TODO

- [ ] Bootstrap CIs on the per-family SNR medians and on each Kruskal–Wallis H.
- [ ] Recover statistical power: bring back the gated high-option families as a
      *separate* above-random-vs-gated contrast, rather than testing option
      count only among the (mostly 2-option) survivors.
- [ ] Disentangle the task-level curation confound (curation method is tied to
      format/option count) with a controlled within-format comparison.
- [ ] Finish Phase B length features (context/option length, ratio) and fold
      them into the design-axis significance table.
- [ ] Controlled within-format curation contrasts to nail the "curation doesn't
      matter" claim: HellaSwag (MT) vs XStoryCloze (human translation) — both
      completion; ARC (MT) vs Global-MMLU-Full (MT + post-edit) — both 4-option
      MCQ from the same source family (would also expose the subject-fragmentation
      effect: ARC's single domain vs MMLU's ~57 subjects).

## Files

- `pretraining/<pool>/per_family_snr.csv`, `per_task_snr.csv` — SNR + design
  attributes per family / task.
- `…/group_stats.csv` — Kruskal–Wallis H/p for each design axis.
- `…/snr_by_*.png` — SNR distribution by curation, format, option count,
  passage, data source; `snr_per_family_ranked.png`; `snr_vs_random_baseline.png`;
  `snr_vs_length_features.png`.
- `…/design_da_size_*_above_66_either_mono_axis.png/.csv` (+ `_paper`), `design_da_size_per_family_mono_axis.csv`,
  `benchmark_characteristics.csv/.tex` — [design_da.py](design_da.py): the characteristics against DA-size,
  the per-family DA-size, the characteristics table the paper prints.

## Extensions from other sweeps

Everything below comes from the **36-model sweep** (2026-04…06, 4 sizes × 3
data mixtures × 3 seeds, pool `custom_swissai_hf`; reference **1B**, twelve
families on the 86-task old list) or from the **external tier**
(`all/external`: the public and reference models, 270M–70B, cross-model
dispersion with no mixture axis). Its SNR is a three-mixture dispersion at
1B under the sweep's fixed-margin gate, on families without the reformulated
twins, so its Kruskal–Wallis tests are a replication of the mechanism on a
different survivor set, never rows of the ladder's table.

### External model-set tier (`all/external`, 36-sweep)

Re-running the design-axis analysis on the `external` tier (cross-model
dispersion over the 270M…70B external ladder) is the sharpest test of the answer-space
mechanism, because the capable external models clear the above-random gate on the
4-option translated MCQA the custom pool drops. Outputs in `all/external/`;
regenerate with `python analysis/rq09_benchmark_design/analyze.py --pool external`.

The above-random gate now passes **11 families, including the 4-option ones**
(`hellaswag`, `global_mmlu_full`, `arc`, `belebele`) the custom pool gated out, so
the survivor set is no longer almost-all-2-option — and the answer-count penalty
disappears entirely.

**Per-family SNR ranking** — median per-family SNR over each family's
per-language tasks, above-random survivors only. The highest-SNR family is now the
**4-option** `hellaswag`, and 4-option families (★) are interleaved throughout
rather than clustered at the bottom:

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

A long-completion 4-option benchmark sitting at the very top is the mechanism's
prediction: option count only hurts when the model is too weak to clear chance
(the above-random gate), not intrinsically — what matters is comparing few, *long*, information-dense
completions.

**Significance of each design axis** — family-level Kruskal–Wallis. With the
4-option families restored to the survivor set, option count loses what little
predictive power it had in the custom pool (H 1.78 → **0.05**); no design axis is
significant:

| axis | H | p |
|---|---|---|
| n_options | **0.05** | 0.83 |
| format | 0.00 | 1.00 |
| data source | 0.17 | 0.68 |
| curation method | 1.44 | 0.49 |
| reading passage | 0.38 | 0.54 |

![SNR by answer-option count (external) — no penalty](all/external/snr_by_n_options.png)

**Paper-level claim.** Comparing the custom and external survivor sets isolates the
confound: the apparent "fewer options ⇒ higher SNR" effect in the custom pool is an
artifact of the capability-driven above-random gate, not a property of
benchmark design. Once capable models clear the gate, answer-option count carries
no signal and the sharpest benchmarks span both 2- and 4-option formats.

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
