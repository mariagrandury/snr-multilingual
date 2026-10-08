# Benchmark design — what makes a benchmark high-SNR?

## Research question

> Why do some multilingual benchmarks separate models cleanly under the SNR
> framework and others don't? Is it curation quality, task format, item
> length, or the answer space?

<!-- BEGIN auto:highlight (analyze.py --pool predictivity) -->
## Highlighted result

- **The above-random gate decides which families are read at all.** Every family whose tasks sit at chance at the reference size is dropped before SNR is computed, leaving **29 families** that clear the gate (6 with 2 options, 18 with 4).
- **Among survivors, only option count separates the families** (five Kruskal–Wallis tests on the same families, uncorrected): curation H = 2.77, p = 0.736; source origin H = 0.09, p = 0.954; option count H = 11.32, p = 0.003; task format H = 7.68, p = 0.104; passage flag H = 0.06, p = 0.811. Option count stays under 0.05 after a Bonferroni correction for the five.
- **Per-task curation test** (tasks as observations, 512 tasks of which 68 are `include_v2_en`): H = 93.14, p = 0.000 — nominally significant, but the tasks of one family are not independent observations, so it says which family dominates, not which curation works.
<!-- END auto:highlight -->

## Experimental setup

Pool `predictivity`: seed 1904, every cell, data build and ladder (swiglu included); the 36-sweep pools are history under [Extensions](#extensions-from-other-sweeps). The signal is `snr_mpd_1.7B` (mean pairwise distance, the noise-and-SNR analysis's headline variant) at the 1.7B reference only, after the above-random gate, so every family here is a gate survivor.

Each family is the median over its per-language aggregate tasks, grouped by curation method, source origin, task format, option count and a reading-passage flag, and tested with a family-level Kruskal–Wallis (groups of one family skipped). Of the 35 families tagged in `FAMILY_META` ([analyze.py](analyze.py), provenance in [data_info.md](data_info.md)), 29 keep at least one task: 512 tasks over 50 languages; TruthfulQA-Multi has no 1.7B SNR, and AfriMMLU, AfriXNLI, Global-MMLU Lite, Global-PIQA completions and the LLM-rewritten Global-MMLU twin are not in the pool. Since 2026-10-08 every accuracy family of the `auto` group is tagged; the 12 added that day (the English-only benchmarks, LAMBADA, ARC (MT), INCLUDE v2, Global PIQA non-parallel) and two corrections are listed in [data_info.md](data_info.md#families-added-on-2026-10-08). Only the `bbpb_` twins stay out: they score the same items as their original, so every characteristic would repeat the original's row (see [Inputs & caveats](#methodology)).

## Key figure

![Median SNR per benchmark family, coloured by answer-option count](pretraining/predictivity/snr_per_family_ranked_paper.png)

Population: pool `predictivity` (seed 1904), SNR `snr_mpd_1.7B` (mean pairwise distance at the 1.7B reference), median over each family's per-language tasks, the 29 families (512 tasks, 50 languages) that clear the above-random gate only; snapshot: the 2026-10-08 refresh.

**Key finding.** **MultiBLiMP, HellaSwag and XStoryCloze separate the designs most sharply at 1.7B, but no characteristic orders the families by DA-size in both readings.** The five sharpest families are MultiBLiMP 2.39 (34 tasks), HellaSwag 2.29 (26), XStoryCloze 2.18 (8), Global PIQA non-parallel 2.06 (2) and XWinograd 1.77 (6), and the reformulated twins and the English-only benchmarks fill the lower half (0.56 to 1.09).

GitHub: [snr_per_family_ranked_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_per_family_ranked_paper.png) · [snr_per_family_ranked_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_per_family_ranked_paper.csv). The other design axes and their tests: [Results](#results); the same characteristics against DA-size: [Benchmark characteristics against DA-size](#benchmark-characteristics-against-da-size).

Key findings:

- **The benchmark whose proxies decide most like 1.7B is also an SNR leader.** HellaSwag has the highest DA-size on every task above chance (0.75 mean over 90M–1B, LAMBADA second at 0.64) and sits second by SNR (2.29). On the 23 of its 26 tasks that the paper's reliability filter keeps it reads 0.77, second there to XCOPA's single task read at two proxies (0.88). The 2-option SNR leaders read 0.47–0.60 on every task above chance (Global PIQA non-parallel 0.47, XWinograd 0.51, PAWS-X 0.53, MultiBLiMP 0.57, XStoryCloze 0.60; `design_da_size_per_family_mono_axis.csv`).
- **The reformulated twins do not lift their families into the top band.** Belebele 0.51 (3 tasks) vs Belebele RF 0.61 (57) and LLM-RF 0.83 (59); INCLUDE 0.66 (3) vs RF 0.58 (29) and LLM-RF 0.72 (30); Global-MMLU RF 0.84 (29) vs Global-MMLU 1.26 (1), and the original Belebele is the lowest family.
- **The 2-option families sit above the 4-option ones, with HellaSwag the one exception.** All six 2-option families read 1.32 to 2.39, and the 4-option ones read 0.51 to 1.26 except HellaSwag (2.29), with the 5-option CommonsenseQA RF and MathQA (0.95, 0.56) among them: a median of 1.91 against 0.75 (Kruskal-Wallis H = 11.32, p = 0.003, under 0.05 after a Bonferroni correction for the five axes, and Spearman ρ = −0.65 with a 95 % interval of −0.84 to −0.33). Both changes of 2026-10-08 move the test from p = 0.051 (17 families): correcting Global PIQA parallel to 4 options (it was tagged 2, and sat at the bottom at 0.59) gives p = 0.008 on the old 17, the twelve new families with the old tag give p = 0.029, and both together 0.003. The six 2-option families are also all continuation, minimal-pair or classification formats with no lettered options, so format and option count are not separable here, and option count does not carry over to DA-size (ρ = −0.03 on every gated task).
- **The same items read sharper in their own language.** INCLUDE v2 in the item's language reaches 0.89 (56 tasks) against 0.68 for the English translation of the same items (68); ARC read through Okapi's ChatGPT translation (0.73, 22) sits above the DeepL one (0.64, 11).
- **Task counts differ by a factor of 68 across families (1 to 68).** The single-task medians (Global-MMLU and the English-only MMLU RF, CommonsenseQA RF, OpenBookQA, MathQA) and the two- and three-task ones (Global PIQA non-parallel, Belebele, INCLUDE) are the least stable bars.

Follow-ups:

- Bootstrap CIs on each family median, to show which bars the single-task families can be told apart from.
- The same ranking at each rung from 90M to 1.7B, to see whether the 2-vs-4-option gap is a reference-size effect.
- The option-count contrast at a fixed format (a 2-option and a 4-option continuation benchmark on the same items), since every 2-option family here is also a non-lettered format.

<!-- BEGIN auto:design-da (design_da.py --pool predictivity) -->
## Benchmark characteristics against DA-size

Pool `predictivity`: rq02's per-task DA-size (mono-axis pairs, each proxy's final checkpoint against the 1.7B final, gated at the proxy and at 1.7B, ≥ 3 pairs) over the paper's `above_66_either` tasks (19 families, up to 111 tasks), with the unfiltered reading (every gated task) beside it; a benchmark's DA-size is the median over its tasks at a proxy and the mean of those medians over 90M–1B. Statistic: Spearman ρ against the centred code of an ordinal or binary characteristic (options 2 → 5, no passage → passage) or a length; curation, source and task format are binarised, translated → native (written in the language, English originals included, or generated from treebanks or planning problems), translated from English → not, and continuation → lettered MCQ; a characteristic no source settles (`FAMILY_META` None: BBH's curation method, option count and passage, LAMBADA's option count) leaves that benchmark out of that row; 95 % bootstrap over the families. The quadrant figure reads every task above chance, so benchmarks fall on both sides of chance agreement; its `above_66_either` twin, where 17 of the 19 benchmarks sit right of 0.5, stays as an analysis figure. Regenerate with `python analysis/rq09_benchmark_design/design_da.py --pool predictivity`.

![Characteristics against DA-size](pretraining/predictivity/design_da_size_correlation_above_66_either_mono_axis.png)

![DA-size by characteristic level](pretraining/predictivity/design_da_size_by_level_above_66_either_mono_axis.png)

![DA-size against the coded characteristics, every task above chance](pretraining/predictivity/design_da_size_quadrant_mono_axis.png)

Key findings (generated):

- **DA-size, `above_66_either`** (mean over 90M–1B): Curation ρ = -0.08 [-0.50, +0.33] (n = 19); Source ρ = -0.25 [-0.66, +0.21] (n = 19); Answer options ρ = -0.30 [-0.72, +0.23] (n = 18); Reading passage ρ = +0.44 [+0.01, +0.75] (n = 19); Context length ρ = -0.67 [-1.00, +0.22] (n = 8); Option length ρ = +0.76 [+0.00, +1.00] (n = 8).
- **DA-size, every gated task**: Curation ρ = -0.33 [-0.64, +0.04] (n = 27); Source ρ = -0.39 [-0.68, -0.04] (n = 27); Answer options ρ = -0.03 [-0.40, +0.30] (n = 25); Reading passage ρ = +0.17 [-0.32, +0.61] (n = 26); Context length ρ = +0.31 [-0.68, +1.00] (n = 8); Option length ρ = +0.50 [-0.40, +1.00] (n = 8).
- **SNR at 1.7B** (the same statistics on rq09's family medians): Curation ρ = +0.02 [-0.34, +0.41] (n = 29); Source ρ = -0.03 [-0.39, +0.36] (n = 29); Task format ρ = -0.20 [-0.55, +0.10] (n = 29); Answer options ρ = -0.65 [-0.84, -0.33] (n = 27); Reading passage ρ = +0.05 [-0.36, +0.47] (n = 28); Context length ρ = -0.48 [-0.97, +0.28] (n = 10); Option length ρ = +0.48 [-0.56, +1.00] (n = 10).
- **Option count per proxy** (`above_66_either`): 90M ρ = +0.15, 175M ρ = -0.23, 350M ρ = -0.17, 600M ρ = -0.53, 1B ρ = -0.11.

GitHub: [design_da_size_correlation_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_correlation_above_66_either_mono_axis.png) · [design_da_size_correlation_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_correlation_above_66_either_mono_axis.csv) · [design_da_size_by_level_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_by_level_above_66_either_mono_axis.png) · [design_da_size_by_level_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_by_level_above_66_either_mono_axis.csv) · [design_da_size_quadrant_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_mono_axis.png) · [design_da_size_quadrant_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_mono_axis.csv) · [design_da_size_quadrant_above_66_either_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_above_66_either_mono_axis.png) · [design_da_size_quadrant_above_66_either_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_quadrant_above_66_either_mono_axis.csv) · [design_da_size_per_family_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/design_da_size_per_family_mono_axis.csv) · [benchmark_characteristics.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/benchmark_characteristics.csv)

Table: `benchmark_characteristics.csv` (`.tex` for the paper) lists each benchmark's characteristics.
<!-- END auto:design-da -->

Key findings (`design_da_size_correlation_above_66_either_mono_axis.csv`, `design_da_size_per_family_mono_axis.csv`; 19 families in the `above_66_either` reading, 27 on every gated task, 29 for the SNR):

- **No characteristic orders the benchmarks by DA-size in both readings.** On the reliable tasks only the passage flag's interval clears 0, barely (ρ = 0.44, CI 0.01 to 0.75; Kruskal–Wallis p = 0.064), and option length's touches it (0.76, CI 0.00 to 1.00, on the 8 original families with sampled items). On every gated task only the source does (ρ = −0.39, CI −0.68 to −0.04, p = 0.045: the benchmarks translated from English decide better), and the passage flag falls to 0.17. Neither finding holds in the other reading, so neither is a design rule.
- **Option count orders the SNR, not the DA-size.** Its SNR interval now excludes 0 (ρ = −0.65, CI −0.84 to −0.33, 27 families), while it is −0.30 (CI −0.72 to 0.23) on the reliable tasks and −0.03 on every gated task.
- **HellaSwag is the clean example.** A 4-option, passage-reading, machine-translated benchmark, it is the top DA-size family on every gated task (0.75) and second on the reliable tasks (0.77), so the levels it carries are lifted by one benchmark. The by-level figure names XCOPA, the highest on the reliable tasks (0.88), on its dot once per characteristic; that value rests on one task read at two proxies.
- **Curation, source and task format are binarised** (translated vs native, translated from English vs not, continuation vs lettered MCQ), and curation and source no longer coincide: INCLUDE v2 (EN) is a translation of natively sourced items, and the English-only benchmarks are native and untranslated. Task format has no value in either DA reading: correcting ARC to its cloze prompt leaves INCLUDE (two tasks, every gated task only) as the one lettered-MCQ family with a DA-size, since Belebele and Global-MMLU have no task above chance at both a proxy and 1.7B.
- **The quadrant figure reads every task above chance, where 13 of the 27 families fall below 0.5**, on both levels of every characteristic, so no pair of quadrants dominates; the lowest are the English-only cloze twins CulturalBench-easy RF 0.34, ACP-Bench RF 0.42 and BBH RF 0.46. On the reliable tasks 17 of the 19 sit right of 0.5 (XNLI 0.48 on one task, INCLUDE v2 (EN) 0.41 on seven; the filter keeps tasks with a median DA ≥ 0.66 on either axis, so a task can pass on DA-ckpt alone); that twin stays as an analysis figure.

Follow-ups:

- Length features for the twins and the 12 families added on 2026-10-08, so the length readings rest on more than eight families.
- BBH's option count and passage per subtask (2 to 8 options, 55 to 549-character stems), which the family-level table can only mark unknown.

## Methodology

Three phases, all on the per-family `snr_mpd_<reference>` signal (the reference
size, 1.7B; the largest bucket in the table while it is missing) (median across a
family's per-language aggregate tasks):

- **Phase 0 — curation process.** Group families by how their items were
  produced (machine translation / human translation / template generation /
  originally-multilingual authoring) and by source origin (English-translated
  vs originally-multilingual); test with a family-level Kruskal–Wallis.
- **Phase A — task format.** Axes: task format (`minimal_pair`, `completion`,
  `classification`, `cloze_completion`, `statement_continuation`, `last_word`,
  `mcq_question_only`, `mrc_passage`; read from the prompt the harness builds,
  `configs/tasks.json` `example`), answer-option count
  (`random_baseline = 1/n_options`, tested categorically 2 to 5 and
  continuously via Spearman), and a reading-passage flag.
- **Phase B — item lengths.** [length_features.py](length_features.py) pulls
  100 English/default items per family from each benchmark's HF dataset and
  computes character-length statistics for context vs options; correlate with SNR.

**Mechanism (a hypothesis).** Reliability tracks the *answer space*, not
curation: a benchmark is sharper when the model compares **fewer, longer**
log-likelihood-scored completions, because each extra option adds another noisy LL
estimate to rank and longer options concentrate more discriminating tokens. On the ladder one illustration holds and one does not:

- HellaSwag (long 4-option completions, 2.29) sits above ARC (short noun-phrase options, 0.73), but PAWS-X (options `Yes`/`No`, ~2.5 chars) is the sixth-sharpest family (1.55, behind MultiBLiMP 2.39, HellaSwag 2.29, XStoryCloze 2.18, Global PIQA non-parallel 2.06 and XWinograd 1.77).
- Option length does not track sharpness significantly over the 10 original families with sampled items (Spearman ρ = 0.48, p = 0.16; context length ρ = −0.48, p = 0.16; `group_stats.csv`), so the answer-space account is a hypothesis to test, not a finding.

The `passage` flag itself does not separate families: XStoryCloze (4-sentence context, completion, 2.18) is high and Belebele (long passage, MRC, 0.51) is low. What the prompt does with the passage is the candidate explanation.

**Inputs & caveats.** Per-family metadata is hand-curated in
[data_info.md](data_info.md) (paper-style paragraphs cross-referenced against the
lm-eval task READMEs); the `FAMILY_META` dict in [analyze.py](analyze.py) is its
machine-readable mirror, with a task-level `xnli_eu` override re-tagged
`mt_post_edited`. Length features exist only for the 10 original families with sampled items (`length_features.csv`), not for the twins, INCLUDE, Global PIQA or the 12 families added on 2026-10-08.

**Characteristics no source settles are `None`, never guessed.** BBH (RF) mixes human-written and programmatically generated subtasks, 2 to 8 options and stems of 55 to 549 characters, so its curation method, option count and passage flag are unknown (its curation still binarises as native: it is an English original); LAMBADA scores a last word, so it has no option count; INCLUDE v2 (EN) ships an English translation whose method the dataset does not document. A `None` drops the benchmark from that characteristic's test and figure row only.

Hand-written numbers in this README are from the ladder report **2026-10-08 12:06** (regenerated locally: acc_norm on the cloze-format originals) and the outputs of the 2026-10-08 refresh (detrended checkpoint noise, so every `snr_mpd_1.7B` moved). FineTasks' selection criteria, judged on this ladder, live in the surrogates analysis ([README](../rq04_surrogates/README.md#finetasks-criteria-on-the-ladder)).

**Families without metadata are left out.** `load_per_task_snr` keeps a family only if it has a `FAMILY_META` entry. Until 2026-10-08 that left 12 accuracy families out (so the DA-size figures read 13 families on the reliable tasks); they are tagged now, and the only families with a 1.7B SNR in the pool and no entry are the 14 `bbpb_` benchmark-BPB twins (plus per-language `bpb`, which is not a benchmark). A `bbpb_` twin scores its original's items with another metric, so its every characteristic would repeat its original's row: adding them would count each benchmark twice in every test.

<!-- BEGIN auto:results (analyze.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq09_benchmark_design/analyze.py --pool predictivity`.

**Per-family SNR ranking** — median `snr_mpd_1.7B` over each family's per-language tasks, above-random survivors only:

| family | median SNR | n | format | n_opts |
|---|---|---|---|---|
| `multiblimp` | 2.39 | 34 | minimal_pair | 2 |
| `hellaswag` | 2.29 | 26 | completion | 4 |
| `xstorycloze` | 2.18 | 8 | completion | 2 |
| `global_piqa_nonparallel_cloze` | 2.06 | 2 | completion | 2 |
| `xwinograd` | 1.77 | 6 | completion | 2 |
| `paws` | 1.55 | 5 | classification | 2 |
| `xcopa` | 1.32 | 8 | completion | 2 |
| `global_mmlu_full` | 1.26 | 1 | mcq_question_only | 4 |
| `rf_mmlu` | 1.09 | 1 | cloze_completion | 4 |
| `xnli` | 1.01 | 14 | classification | 3 |
| `rf_acp_bench_mcq` | 0.97 | 5 | cloze_completion | 4 |
| `rf_commonsense_qa` | 0.95 | 1 | cloze_completion | 5 |
| `lambada_openai_mt` | 0.94 | 5 | last_word |  |
| `rf_bbh_mcq` | 0.90 | 10 | cloze_completion |  |
| `include_v2_og` | 0.89 | 56 | cloze_completion | 4 |
| `rf_global_mmlu_full` | 0.84 | 29 | cloze_completion | 4 |
| `rfgm_belebele` | 0.83 | 59 | statement_continuation | 4 |
| `rf_cultural_bench_easy` | 0.83 | 7 | cloze_completion | 4 |
| `openbookqa` | 0.76 | 1 | completion | 4 |
| `arc` | 0.73 | 22 | cloze_completion | 4 |
| `rfgm_include_base_44` | 0.72 | 30 | statement_continuation | 4 |
| `include_v2_en` | 0.68 | 68 | cloze_completion | 4 |
| `include_base_44` | 0.66 | 3 | mcq_question_only | 4 |
| `arc_mt` | 0.64 | 11 | cloze_completion | 4 |
| `rf_belebele` | 0.61 | 57 | cloze_completion | 4 |
| `global_piqa_parallel_cloze` | 0.59 | 10 | completion | 4 |
| `rf_include_base_44` | 0.58 | 29 | cloze_completion | 4 |
| `mathqa` | 0.56 | 1 | cloze_completion | 5 |
| `belebele` | 0.51 | 3 | mrc_passage | 4 |

![Per-family SNR ranking](pretraining/predictivity/snr_per_family_ranked.png)

**Significance of each design axis** — family-level Kruskal–Wallis over the survivors (high-option families already removed by the gate):

| axis | H | p |
|---|---|---|
| n_options | 11.32 | 0.00 |
| format | 7.68 | 0.10 |
| data source | 0.09 | 0.95 |
| curation method | 2.77 | 0.74 |
| reading passage | 0.06 | 0.81 |
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

Key findings (29 families, 512 tasks, `snr_mpd_1.7B`):

- **Option count separates the families.** The 6 two-option families have a median family SNR of 1.91, the 18 four-option ones 0.75 and the 2 five-option ones 0.75; H = 11.32, p = 0.003 over these 26 (XNLI, the only 3-option family, is a group of one and skipped; LAMBADA and BBH have no single count), under 0.05 after a Bonferroni correction for the five axes. It was H = 3.81, p = 0.051 on the 17 families before 2026-10-08.
- **No other family-level axis comes near.** Curation H = 2.77, p = 0.74 (6 groups); source origin H = 0.09, p = 0.95 (3 groups); format H = 7.68, p = 0.10 (5 groups); reading passage H = 0.06, p = 0.81.
- **The per-task curation test measures one family, not a curation method.** It gives H = 93.14, p = 6.7e-18 (512 tasks, 7 groups), but its top group, template-generated (median 2.29), is MultiBLiMP (34 tasks) with ACP-Bench RF (5), against 0.68 to 1.04 for the other six.
- **No length feature reaches significance.** Over the 10 original families, context length ρ = −0.48 (p = 0.16), option length ρ = 0.48 (p = 0.16), context-to-option ratio ρ = −0.50 (p = 0.14).

Follow-ups:

- The option-count contrast within one family and format (an original vs its reformulated twin, same items), the cleanest test the twins allow.
- Length features for the twins, INCLUDE, Global PIQA and the 12 families added on 2026-10-08, to put all 29 families on the length plot.

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

The family medians above, per language (`predictivity` pool). Regenerate with `python analysis/rq09_benchmark_design/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq09 in one figure](pretraining/predictivity/highlights.png)

![SNR per benchmark and language](pretraining/predictivity/snr_family_by_language.png)
<!-- END auto:panels -->

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/highlights.csv) ·
[snr_family_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_family_by_language.png)

Population (both figures): pool `predictivity` (seed 1904), `log10` of `snr_mpd_1.7B` per (benchmark, language) task at the 1.7B reference, gate survivors only: 29 families, 512 tasks, 50 languages; snapshot: the 2026-10-08 refresh. The two rank panels of `highlights.png` show the top and bottom eight of each list, so the 13 middle families and the 34 middle languages are not drawn.

Key findings (`highlights.csv`, `snr_family.csv`):

- **Ten of the 29 families have a median log10 SNR above 0 (signal above noise).** They are MultiBLiMP 0.38, HellaSwag 0.36, XStoryCloze 0.34, Global PIQA non-parallel 0.31, XWinograd 0.25, PAWS-X 0.19, XCOPA 0.12, Global-MMLU 0.10, MMLU RF 0.04 and XNLI 0.01; Belebele is lowest (−0.29).
- **9 of 50 languages have a median log10 SNR above 0.** The highest are Bosnian 0.25 (1 task), Slovenian 0.22 (3) and Kazakh 0.10 (8), and the lowest Malayalam −0.28 (4) and Lithuanian −0.26 (8); English, now with the English-only benchmarks (34 tasks), is at −0.01.
- **By construction method, template-generated (MultiBLiMP and ACP-Bench RF) is highest.** It is at 0.36 against 0.02 for machine-translated, −0.08 for English originals, −0.11 for MT-post-edited, −0.12 for human-translated, −0.12 for originally multilingual and −0.17 for the one translation of unknown method (INCLUDE v2 (EN)), the lowest.
- **No cell of the benchmark-by-language grid is grey.** `gated` is False for all 512 tasks in `snr_family.csv`, so a white cell is a (benchmark, language) pair with no 1.7B SNR.

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
