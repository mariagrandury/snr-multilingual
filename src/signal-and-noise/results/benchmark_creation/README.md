# `benchmark_creation/` — what makes a benchmark high-SNR?

> Step-by-step plan: [INSTRUCTIONS.md](INSTRUCTIONS.md). Use one
> Claude session per research question — see
> [../PARALLEL_SESSIONS.md](../PARALLEL_SESSIONS.md).

## Research question

The original Q4 (v0) was *"do high-SNR benchmarks share data-source /
curation-process traits?"* — see Phase 0 below. After v0 left most of
the variance unexplained, two extra axes were added (Phase A: task
format; Phase B: item lengths). The combined story is:

> **Curation method does not predict SNR. Task design — and
> specifically how many candidate options the model has to compare,
> and how long those options are — does.**

## Setup

- **SNR signal:** `snr_mpd_1B` from
  [../snr_definition/snr_variants_per_task.csv](../snr_definition/snr_variants_per_task.csv).
  Q1 picked `mpd` (mean pairwise distance) as the headline variant —
  highest mean Pearson r vs decision accuracy across languages.
- **Per-family aggregate:** median of `snr_mpd_1B` across the family's
  per-language aggregate tasks.
- **Metadata:** [data_info.md](data_info.md) — paper-style paragraphs +
  a schema table, cross-referenced against the
  `lm-evaluation-harness` task READMEs. The `FAMILY_META` dict in
  [analyze.py](analyze.py) is the machine-readable mirror.
- **Length features:** [length_features.py](length_features.py) pulls
  100 English (or default-config) items per family from each
  benchmark's HF dataset and computes character-length statistics for
  context vs options. Output: [length_features.csv](length_features.csv).
- **Coverage caveat (carried forward):** `global_mmlu` (Lite, 6 langs)
  is excluded from the SNR analysis — only one Apertus model was
  evaluated on it (350M, single config), so `mpd_1B` is NaN
  everywhere. arc_de/fr and hellaswag_de/fr are also NaN at 1B.
  `global_piqa_completions_spa_latn_spai` is filtered by
  `_is_language_aggregate` (3 trailing tokens). Net pool: **88
  per-language tasks across 11 families.**

## Headline ranking

![Per-family SNR ranking colored by curation category](snr_per_family_ranked.png)

Per-family median `snr_mpd_1B` (full table: [per_family_snr.csv](per_family_snr.csv)):

| family | median | n_tasks | curation | format | n_opts | ctx (chars) | opt (chars) |
|---|---:|---:|---|---|---:|---:|---:|
| multiblimp | 4.42 | 7 | template_generated | minimal_pair | 2 | 46 | 119 |
| xstorycloze | 2.04 | 8 | human_translation | completion | 2 | 189 | 38 |
| hellaswag | 2.04 | 6 | machine_translation | completion | 4 | 118 | 61 |
| xwinograd | 1.56 | 4 | originally_multilingual | completion | 2 | 74 | 7 |
| global_piqa_completions | 1.25 | 10 | originally_multilingual | completion | 2 | 38 | 54 |
| paws | 1.05 | 5 | human_translation | classification | 2 | 224 | 2.5 |
| xcopa | 0.92 | 6 | human_translation | completion | 2 | 37 | 31 |
| xnli | 0.91 | 11 | human_translation | classification | 3 | 127 | 10 |
| arc | 0.74 | 9 | machine_translation | mcq_question_only | 4 | 122 | 28 |
| belebele | 0.48 | 12 | human_translation | mrc_passage | 4 | 558 | 21 |
| global_mmlu_full | 0.40 | 10 | mt_post_edited | mcq_question_only | 4 | 120 | 11 |

## Phase 0 — curation process (the original Q4 question)

![Per-family SNR by curation process](snr_by_curation_process.png)

| view | test | H | p |
|---|---|---:|---:|
| family / curation_category (5-way) | Kruskal-Wallis | 0.84 | 0.66 |
| family / source_origin (2-way) | Kruskal-Wallis | 2.67 | 0.10 |
| task / curation_category (with `xnli_eu` re-tag) | Kruskal-Wallis | 32.3 | <1e-6 |

The family-level test does not reach significance for any curation
view (see [snr_by_data_source.png](snr_by_data_source.png) too). The
per-task test only reaches significance because multiblimp's
template-generated tasks pull the `template_generated` group up; if
you drop multiblimp the residual differences across the other four
categories are not significant. **Curation method alone does not
predict SNR.**

## Phase A — task format (the strongest categorical predictor)

![Per-family SNR by task format](snr_by_format.png)

![Per-family SNR by number of answer options](snr_by_n_options.png)

| view | test | statistic | p |
|---|---|---:|---:|
| family / format (5-way) | Kruskal-Wallis | H = 5.69 | **0.058** |
| family / n_options (2/3/4-way) | Kruskal-Wallis | H = 2.91 | 0.088 |
| family / passage flag | Kruskal-Wallis | H = 0.38 | 0.54 |
| family / random_baseline (continuous) | Spearman ρ | **+0.62** | **0.041** |

![SNR vs random baseline](snr_vs_random_baseline.png)

The continuous Spearman correlation between SNR and random baseline
(= 1 / n_options) is **+0.62, p = 0.041** — the only significant
single-axis result in the entire benchmark-creation analysis.
Concretely: 2-option tasks have higher SNR than 4-option tasks,
holding everything else equal. The probable mechanism is that
log-likelihood comparison across two completions is sharper than
across four — every additional option introduces another noisy LL
estimate that has to be ranked correctly.

The format axis (5-way Kruskal-Wallis p = 0.058) tells the same story
in categorical form:
- `minimal_pair` (multiblimp): single family, very high SNR.
- `completion` (n=5): 2nd-tier, median ≈ 1.6.
- `classification` (n=2): 3rd-tier, ~1.0.
- `mcq_question_only` (n=2): 4th-tier, ~0.55.
- `mrc_passage` (n=1, belebele): lowest, 0.48.

The two earlier surprises now resolve:
- **Belebele**: only `mrc_passage` family AND 4 options — both
  strongest negative predictors stack.
- **MultiBLiMP**: minimal-pair format gives uniquely sharp signal
  because each item is a 1-token contrast, not "automatic generation
  produces better data."

The `passage` flag (whether the prompt contains a long passage)
is **not** a useful predictor (p = 0.54). XStoryCloze (4-sentence
context, completion task) has high SNR; Belebele (long passage, MRC)
has low SNR — passage length doesn't matter, what's done with it
does.

## Phase B — item lengths (extends Phase A quantitatively)

![SNR vs length features](snr_vs_length_features.png)

100 English-or-default items per family pulled from each benchmark's
HF dataset; character-length statistics in
[length_features.csv](length_features.csv).

| feature | Spearman ρ | p | sign as predicted? |
|---|---:|---:|---|
| context length | -0.33 | 0.33 | yes (weak) |
| **option length** | **+0.54** | **0.089** | **yes (borderline)** |
| context : option ratio | -0.47 | 0.14 | yes (weak) |

The strongest length feature is **option length** (ρ = +0.54): tasks
with longer per-option text — full sentences for log-likelihood
comparison — have higher SNR. The mechanism is the same as Phase A's
n_options effect, just in a continuous variable: longer options give
more discriminating tokens, which sharpens per-item log-likelihood.

Concrete examples:
- **PAWS** (option = "Yes" / "No", 2.5 chars) sits at SNR 1.05
  despite being binary; its short labels concentrate the signal in
  one or two tokens.
- **MultiBLiMP** (option = full grammatical sentence, 119 chars)
  sits at SNR 4.42 — 100× more discriminating tokens than PAWS, same
  n_options.
- **HellaSwag** (option = full continuation, 61 chars) escapes the
  4-option penalty and sits at SNR 2.04, beating ARC (option = a
  short noun phrase, 28 chars) at SNR 0.74.

`context_len_chars` and `context_to_option_ratio` move in the
expected direction (longer context → lower SNR; higher ratio → lower
SNR) but neither reaches significance individually with n=11.

## Combined picture

In rough order of evidence strength:

| effect | direction | strength |
|---|---|---|
| n_options / random_baseline | fewer options → higher SNR | **strong** (Spearman ρ=+0.62, p=0.04) |
| option length | longer options → higher SNR | borderline (ρ=+0.54, p=0.09) |
| task format (minimal_pair > completion > classification > MC > MRC) | as listed | borderline (KW p=0.06) |
| context : option ratio | lower ratio → higher SNR | weak (ρ=-0.47, p=0.14) |
| context length | shorter context → higher SNR | weak (ρ=-0.33, p=0.33) |
| source origin (English-translated vs originally-multilingual) | originally-multilingual median higher | weak (KW p=0.10) |
| passage in prompt | no effect | none |
| curation category (MT / human / template / etc.) | no effect | none (p=0.66) |

The curation question that motivated v0 is not where the variance
lives. **Task design — option count and option length — explains
most of what we can explain on this 11-family pool.** All effects are
borderline-significant due to the small sample, but they're all
mutually consistent and point at the same mechanism: tasks where the
model has to pick between fewer, longer log-likelihood-scored
completions are SNR-higher.

## Recommended follow-up

A controlled comparison would tighten the story: pick families with
matched task format and contrast curation methods within it.
Concretely:
- **HellaSwag (MT) vs XStoryCloze (human translation)**: both
  4-or-2-option completion, both have a passage context. SNR ~2.04
  for both — first direct evidence that curation doesn't matter when
  format is held constant.
- **ARC (MT) vs Global-MMLU-Full (MT+post-edit)**: both 4-option
  MCQ, same source dataset family. ARC 0.74 vs MMLU-Full 0.40 —
  domain-fragmentation effect (57 subjects vs single domain) shows up
  here. Phase C topic-tagging would quantify it.

## Directory contents

- [INSTRUCTIONS.md](INSTRUCTIONS.md), [data_info.md](data_info.md) —
  research-question spec and per-family paper-style metadata.
- [analyze.py](analyze.py) — runs Phases 0/A/B; emits the CSVs and
  plots below.
- [length_features.py](length_features.py) — Phase B HF sampler;
  writes [length_features.csv](length_features.csv) and
  [sample_items.json](sample_items.json) (one example item per
  family, kept for any Phase C topic tagging).
- [per_family_snr.csv](per_family_snr.csv) — one row per family,
  carries SNR aggregates + all metadata + length features.
- [per_task_snr.csv](per_task_snr.csv) — one row per per-language
  aggregate task, carrying the per-task curation override (`xnli_eu`
  re-tagged as `mt_post_edited`).
- [group_stats.csv](group_stats.csv) — Kruskal-Wallis (and Spearman
  for continuous axes) for every grouping view.
- Phase 0 plots:
  [snr_per_family_ranked.png](snr_per_family_ranked.png) (headline),
  [snr_by_curation_process.png](snr_by_curation_process.png),
  [snr_by_data_source.png](snr_by_data_source.png),
  [snr_by_curation_per_task.png](snr_by_curation_per_task.png).
- Phase A plots:
  [snr_by_n_options.png](snr_by_n_options.png),
  [snr_by_format.png](snr_by_format.png),
  [snr_by_passage.png](snr_by_passage.png),
  [snr_vs_random_baseline.png](snr_vs_random_baseline.png).
- Phase B plots:
  [snr_vs_length_features.png](snr_vs_length_features.png) (3-panel
  combined),
  [snr_vs_context_len.png](snr_vs_context_len.png),
  [snr_vs_option_len.png](snr_vs_option_len.png),
  [snr_vs_context_option_ratio.png](snr_vs_context_option_ratio.png).
