# Benchmark design — what makes a benchmark high-SNR?

## Research question

> Why do some multilingual benchmarks separate models cleanly under the SNR
> framework and others don't? Is it curation quality, task format, item
> length, or the answer space?

<!-- BEGIN auto:highlight (analyze.py --pool predictivity) -->
## Highlighted result

- **The answer-count penalty lives in the above-random gate, upstream of SNR.** Every family whose tasks sit at chance at the reference size is dropped before SNR is computed, leaving **17 families** that clear the gate — most of them 2-option.
- **Among survivors, option count reaches p < 0.05 at the family level** — with five uncorrected tests on the same families, one such hit is what chance produces: curation H = 1.62, p = 0.655; source origin H = 0.16, p = 0.688; option count H = 5.69, p = 0.017; task format H = 1.73, p = 0.422; passage flag H = 0.40, p = 0.527. Too little variation is left among the survivors (mostly 2-option) to resolve any axis.
- **Per-task curation test** (tasks as observations, 328 tasks of which 59 are `rfgm_belebele`): H = 91.44, p = 0.000 — nominally significant, but the tasks of one family are not independent observations, so it says which family dominates, not which curation works.
<!-- END auto:highlight -->

## Experimental setup

Pool `predictivity`: seed 1904, every cell, data build and ladder (swiglu included); the 36-sweep pools are history under [Extensions](#extensions-from-other-sweeps). The signal is `snr_mpd_1.7B` (mean pairwise distance, the noise-and-SNR analysis's headline variant) at the 1.7B reference only, after the above-random gate, so every family here is a gate survivor.

Each family is the median over its per-language aggregate tasks, grouped by curation method, source origin, task format, option count and a reading-passage flag, and tested with a family-level Kruskal–Wallis (groups of one family skipped). Of the 23 families tagged in `FAMILY_META` ([analyze.py](analyze.py), provenance in [data_info.md](data_info.md)), 17 keep at least one task: 328 tasks over 49 languages; TruthfulQA-Multi has no 1.7B SNR, and AfriMMLU, AfriXNLI, Global-MMLU Lite, Global-PIQA completions and the LLM-rewritten Global-MMLU twin are not in the pool.

## Key figure

![Median SNR per benchmark family, coloured by answer-option count](pretraining/predictivity/snr_per_family_ranked_paper.png)

Population: pool `predictivity` (seed 1904), SNR `snr_mpd_1.7B` (mean pairwise distance at the 1.7B reference), median over each family's per-language tasks, the 17 families (328 tasks, 49 languages) that clear the above-random gate only; snapshot 2026-10-06 04:26.

**Key finding.** Among the 17 families that clear the gate, every two- and three-option family except Global PIQA (parallel, one task, 0.48) has a median SNR of 0.89 to 1.70, and every four-option family sits at 0.41 to 0.83. Option count is the only family-level design axis with p < 0.05 (H = 5.69, p = 0.017), which one of five uncorrected tests can reach by chance.

GitHub: [snr_per_family_ranked_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_per_family_ranked_paper.png) · [snr_per_family_ranked_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq09_benchmark_design/pretraining/predictivity/snr_per_family_ranked_paper.csv). The other design axes and their tests: [Results](#results).

Key findings:

- The top four families are all 2-option: MultiBLiMP 1.70 (34 tasks), XStoryCloze 1.53 (8), XWinograd 1.45 (6), PAWS-X 1.37 (5).
- The best 4-option family, Global-MMLU 0.83, rests on one task; among 4-option families with at least three tasks the top is INCLUDE at 0.70 (3 tasks).
- The reformulated twins do not lift their families into the 2-option band: Belebele 0.46 (3 tasks) vs Belebele RF 0.49 (57) and LLM-RF 0.62 (59); INCLUDE 0.70 (3) vs RF 0.60 (29) and LLM-RF 0.58 (30); Global-MMLU RF 0.41 (29) is the lowest family.
- Task counts differ by a factor of 59 across families (1 to 59), so the single-task medians (Global-MMLU, Global PIQA) are the least stable bars.

Follow-ups:

- Bootstrap CIs on each family median, to show which bars the single-task families can be told apart from.
- The same ranking at each rung from 90M to 1.7B, to see whether the 2-vs-4-option gap is a reference-size effect.

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

- HellaSwag (long 4-option completions, 0.61) sits above ARC (short noun-phrase options, 0.51), but PAWS-X (options `Yes`/`No`, ~2.5 chars) is the fourth-sharpest family (1.37, behind MultiBLiMP 1.70, XStoryCloze 1.53 and XWinograd 1.45).
- Option length does not track sharpness over the 10 original families with sampled items (Spearman ρ = 0.12, p = 0.75; context length ρ = −0.35, p = 0.33; `group_stats.csv`), so the answer-space account is a hypothesis to test, not a finding.

The `passage` flag itself does not separate families: XStoryCloze (4-sentence context, completion, 1.53) is high and Belebele (long passage, MRC, 0.46) is low. What the prompt does with the passage is the candidate explanation.

**Inputs & caveats.** Per-family metadata is hand-curated in
[data_info.md](data_info.md) (paper-style paragraphs cross-referenced against the
lm-eval task READMEs); the `FAMILY_META` dict in [analyze.py](analyze.py) is its
machine-readable mirror, with a task-level `xnli_eu` override re-tagged
`mt_post_edited`. Length features exist only for the 10 original families with sampled items (`length_features.csv`), not for the twins, INCLUDE or Global PIQA.

Hand-written numbers in this README are from the ladder-report snapshot **2026-10-06 04:26** (refresh commit b316f53b, outputs regenerated 2026-10-07). FineTasks' selection criteria, judged on this ladder, live in the surrogates analysis ([README](../rq04_surrogates/README.md#finetasks-criteria-on-the-ladder)).

**Families without metadata are left out.** `load_per_task_snr` keeps a family only if it has a `FAMILY_META` entry. As of 2026-10-07, 25 benchmark families with a 1.7B SNR in the pool have none, so they are in no table here: the 14 `bbpb_` benchmark-BPB twins, `arc_mt`, `global_piqa_nonparallel_cloze`, `include_v2_en`, `include_v2_og`, `lambada_openai_mt`, `mathqa` and the `rf_` twins of acp_bench, bbh, commonsense_qa, cultural_bench_easy and mmlu (plus per-language `bpb`, which is not a benchmark).

<!-- BEGIN auto:results (analyze.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq09_benchmark_design/analyze.py --pool predictivity`.

**Per-family SNR ranking** — median `snr_mpd_1.7B` over each family's per-language tasks, above-random survivors only:

| family | median SNR | n | format | n_opts |
|---|---|---|---|---|
| `multiblimp` | 1.70 | 34 | minimal_pair | 2 |
| `xstorycloze` | 1.53 | 8 | completion | 2 |
| `xwinograd` | 1.45 | 6 | completion | 2 |
| `paws` | 1.37 | 5 | classification | 2 |
| `xcopa` | 0.98 | 8 | completion | 2 |
| `xnli` | 0.89 | 14 | classification | 3 |
| `global_mmlu_full` | 0.83 | 1 | mcq_question_only | 4 |
| `include_base_44` | 0.70 | 3 | mcq_question_only | 4 |
| `rfgm_belebele` | 0.62 | 59 | statement_continuation | 4 |
| `hellaswag` | 0.61 | 25 | completion | 4 |
| `rf_include_base_44` | 0.60 | 29 | cloze_completion | 4 |
| `rfgm_include_base_44` | 0.58 | 30 | statement_continuation | 4 |
| `arc` | 0.51 | 16 | mcq_question_only | 4 |
| `rf_belebele` | 0.49 | 57 | cloze_completion | 4 |
| `global_piqa_parallel_cloze` | 0.48 | 1 | completion | 2 |
| `belebele` | 0.46 | 3 | mrc_passage | 4 |
| `rf_global_mmlu_full` | 0.41 | 29 | cloze_completion | 4 |

![Per-family SNR ranking](pretraining/predictivity/snr_per_family_ranked.png)

**Significance of each design axis** — family-level Kruskal–Wallis over the survivors (high-option families already removed by the gate):

| axis | H | p |
|---|---|---|
| n_options | 5.69 | 0.02 |
| format | 1.73 | 0.42 |
| data source | 0.16 | 0.69 |
| curation method | 1.62 | 0.66 |
| reading passage | 0.40 | 0.53 |
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

Key findings (17 families, 328 tasks, `snr_mpd_1.7B`):

- Option count: the 6 two-option families have a median family SNR of 1.41 and the 10 four-option families 0.59; H = 5.69, p = 0.017 over these 16 (XNLI, the only 3-option family, is a group of one and skipped).
- No other family-level axis comes near: curation H = 1.62, p = 0.66 (4 groups); source origin H = 0.16, p = 0.69; format H = 1.73, p = 0.42 (3 groups); reading passage H = 0.40, p = 0.53.
- The per-task curation test (328 tasks, 5 groups) gives H = 91.44, p = 6.5e-19, but its top group, template-generated (median 1.70), is MultiBLiMP alone (34 tasks), against 0.41 to 0.63 for the other four; it measures one family, not a curation method.
- Length features (10 original families): context length ρ = −0.35 (p = 0.33), option length ρ = 0.12 (p = 0.75), context-to-option ratio ρ = −0.24 (p = 0.51).

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

Population (both figures): pool `predictivity` (seed 1904), `log10` of `snr_mpd_1.7B` per (benchmark, language) task at the 1.7B reference, gate survivors only: 17 families, 328 tasks, 49 languages; snapshot 2026-10-06 04:26. The two rank panels of `highlights.png` show the top and bottom eight of each list, so the 17th family, `rfgm_belebele` (the median one, log10 −0.20), and the 33 middle languages are not drawn.

Key findings (`highlights.csv`, `snr_family.csv`):

- Only four families have a median log10 SNR above 0 (signal above noise): MultiBLiMP 0.23, XStoryCloze 0.18, XWinograd 0.16, PAWS-X 0.14; XCOPA (−0.01) and XNLI (−0.05) sit just below, and Global-MMLU RF is lowest (−0.39).
- Only 2 of 49 languages have a median log10 SNR above 0: Slovenian 0.07 (3 tasks) and Croatian 0.03 (6); English is at −0.02 (12 tasks), and the lowest are Malay −0.51 (5) and Malayalam −0.47 (2).
- By construction method, template-generated (MultiBLiMP alone) is at 0.23 against −0.20 to −0.39 for originally multilingual, machine-translated, human-translated and MT-post-edited, with MT-post-edited lowest.
- No cell of the benchmark-by-language grid is grey (`gated` is False for all 328 tasks in `snr_family.csv`), so a white cell is a (benchmark, language) pair with no 1.7B SNR.

Follow-ups:

- The per-language ranking restricted to languages with at least five tasks, since several extremes (Slovenian 3 tasks, Malayalam 2) rest on very few.
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
