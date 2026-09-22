# RQ9 — What makes a benchmark high-SNR?

## Research question

> Why do some multilingual benchmarks separate models cleanly under the SNR
> framework and others don't? Is it curation quality, task format, item
> length, or the answer space?

<!-- BEGIN auto:highlight (analyze.py --pool predictivity) -->
## Highlighted result

- **The answer-count penalty lives in the above-random gate, upstream of SNR.** Every family whose tasks sit at chance at the reference size is dropped before SNR is computed, leaving **15 families** that clear the gate — most of them 2-option.
- **Among survivors, option count reaches p < 0.05 at the family level** — with five uncorrected tests on the same families, one such hit is what chance produces: curation H = 3.39, p = 0.184; source origin H = 0.89, p = 0.346; option count H = 5.40, p = 0.020; task format H = 0.89, p = 0.640; passage flag H = 0.84, p = 0.361. Too little variation is left among the survivors (mostly 2-option) to resolve any axis.
- **Per-task curation test** (tasks as observations, 264 tasks of which 57 are `rf_belebele`): H = 73.30, p = 0.000 — nominally significant, but the tasks of one family are not independent observations, so it says which family dominates, not which curation works.
<!-- END auto:highlight -->

## Experimental setup

Outputs live under `pretraining/<pool>/`; headline numbers reflect the ladder
pool `predictivity`, with the 36-sweep pools as history. The SNR signal is
`snr_mpd_<ref>` (Q1's headline pick: mean pairwise distance, dispersion
cluster) at the target size, or at the largest size with data while the
reference rungs train. Each family's per-language aggregate tasks are grouped
along several design axes — curation method, source origin, task format,
answer-option count, reading-passage flag — and tested with a family-level
Kruskal–Wallis. The ladder adds the natively-sourced INCLUDE and Global-PIQA
families, the human-translated IrokoBench families (AfriMMLU, AfriXNLI) and
TruthfulQA-Multi to the twelve 36-sweep families (`FAMILY_META` in
[analyze.py](analyze.py), provenance in `plan/benchmark_selection.md`). The
above-random gate (computed upstream of SNR) drops every at-chance benchmark
before this analysis, so the families seen here are the gate's survivors.

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

**Mechanism (the durable finding).** Reliability tracks the *answer space*, not
curation: a benchmark is sharper when the model compares **fewer, longer**
log-likelihood-scored completions — each extra option adds another noisy LL
estimate to rank, and longer options concentrate more discriminating tokens.
Illustrations at fixed option count: PAWS (options `Yes`/`No`, ~2 chars) is low
despite being binary; MultiBLiMP (full-sentence minimal pairs) is the sharpest;
HellaSwag (long 4-option completions) escapes the 4-option penalty that sinks
ARC (short noun-phrase options). The `passage` flag itself doesn't matter —
XStoryCloze (4-sentence context, completion) is high, Belebele (long passage,
MRC) is low — what the prompt *does* with the passage is what counts.

**Inputs & caveats.** Per-family metadata is hand-curated in
[data_info.md](data_info.md) (paper-style paragraphs cross-referenced against the
lm-eval task READMEs); the `FAMILY_META` dict in [analyze.py](analyze.py) is its
machine-readable mirror, with a task-level `xnli_eu` override re-tagged
`mt_post_edited`. `global_mmlu` (Lite, one Apertus model) and `arc_de/fr` /
`hellaswag_de/fr` were NaN at the reference when this was written and excluded.

<!-- BEGIN auto:results (analyze.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq09_benchmark_design/analyze.py --pool predictivity`.

**Per-family SNR ranking** — median `snr_mpd_1.7B` over each family's per-language tasks, above-random survivors only:

| family | median SNR | n | format | n_opts |
|---|---|---|---|---|
| `xwinograd` | 1.41 | 6 | completion | 2 |
| `multiblimp` | 1.24 | 34 | minimal_pair | 2 |
| `xstorycloze` | 1.20 | 8 | completion | 2 |
| `paws` | 1.19 | 5 | classification | 2 |
| `xcopa` | 0.84 | 8 | completion | 2 |
| `xnli` | 0.80 | 14 | classification | 3 |
| `rf_include_base_44` | 0.56 | 29 | cloze_completion | 4 |
| `include_base_44` | 0.56 | 4 | mcq_question_only | 4 |
| `rfgm_include_base_44` | 0.53 | 20 | statement_continuation | 4 |
| `belebele` | 0.47 | 9 | mrc_passage | 4 |
| `arc` | 0.47 | 15 | mcq_question_only | 4 |
| `rf_belebele` | 0.47 | 57 | cloze_completion | 4 |
| `global_piqa_parallel_cloze` | 0.42 | 1 | completion | 2 |
| `hellaswag` | 0.40 | 25 | completion | 4 |
| `rf_global_mmlu_full` | 0.30 | 29 | cloze_completion | 4 |

![Per-family SNR ranking](pretraining/predictivity/snr_per_family_ranked.png)

**Significance of each design axis** — family-level Kruskal–Wallis over the survivors (high-option families already removed by the gate):

| axis | H | p |
|---|---|---|
| n_options | 5.40 | 0.02 |
| format | 0.89 | 0.64 |
| data source | 0.89 | 0.35 |
| curation method | 3.39 | 0.18 |
| reading passage | 0.84 | 0.36 |
<!-- END auto:results -->

## External model-set tier (`all/external`, 36-sweep)

Re-running the design-axis analysis on the `external` tier (cross-model
dispersion over the 270M…70B external ladder) is the sharpest test of RQ2's
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
(RQ0), not intrinsically — what matters is comparing few, *long*, information-dense
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
artifact of the capability-driven above-random gate (RQ0), not a property of
benchmark design. Once capable models clear the gate, answer-option count carries
no signal and the sharpest benchmarks span both 2- and 4-option formats.


## Results from the 36-model sweep (2026-06, superseded)

The numbers below were generated on the 36-model sweep (4 sizes × 3 data mixtures × 3 seeds, 12 languages, pool `custom_swissai_hf` unless stated) and are kept as history; the predictivity ladder regenerates the blocks above.

### Highlighted result

- **The answer-count penalty lives in the above-random gate, upstream of SNR.** Every at-chance 4-option *translated knowledge* MCQA (`belebele`, `global_mmlu_full`, `truthfulqa`) is dropped before SNR is computed, leaving **9 families** that clear the gate — most of them 2-option.
- **Among survivors, no single design feature is individually significant.** Family-level Kruskal–Wallis on option count is **H = 1.78, p = 0.18** and on task format **H = 0.00, p = 1.00** — too little variation left (mostly 2-option) to resolve them.
- **Curation method explains nothing** — family-level Kruskal–Wallis on curation is **H = 0.50, p = 0.78**. Once the gate fixes the answer space, how a benchmark was built does not predict its reliability.

### Results

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

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

The family medians above, per language (`predictivity` pool). Regenerate with `python analysis/rq09_benchmark_design/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq09 in one figure](pretraining/predictivity/highlights.png)

![SNR per benchmark and language](pretraining/predictivity/snr_family_by_language.png)
<!-- END auto:panels -->
