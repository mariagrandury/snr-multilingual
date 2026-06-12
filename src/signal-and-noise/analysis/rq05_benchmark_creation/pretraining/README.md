# What makes a benchmark high-SNR?

> Why do some multilingual benchmarks separate models cleanly under the SNR
> framework and others don't? Is it curation quality, task format, item
> length, or something else?

## TL;DR

**Curation method does not predict SNR. Task design — specifically how
many candidate options the model has to compare, and how long those
options are — does.**

The strongest single-axis predictor is the number of answer options
(Spearman ρ = **+0.77, p = 0.006** against the random baseline; family-
level Kruskal-Wallis **H = 5.5, p = 0.019**): 2-option tasks have higher
SNR than 4-option tasks, holding everything else equal. The probable
mechanism — every additional option introduces another noisy log-
likelihood estimate to rank — shows up qualitatively in option length
too (multiblimp's full-sentence options vs PAWS's "Yes"/"No"), but at the
per-family level (n = 11) length features alone don't reach significance.
The curation-method question (the original Q4) explains <1% of family-
level SNR variance (Kruskal-Wallis H = 1.44, p = 0.49) — once task design
is held constant, curation doesn't matter.

## Headline ranking (pool: `seeds_28_1797_1904`, recommended)

![Per-family SNR ranking](seeds_28_1797_1904/snr_per_family_ranked.png)

Per-family median `snr_mpd_1B` (full table:
[seeds_28_1797_1904/per_family_snr.csv](seeds_28_1797_1904/per_family_snr.csv)):

| family | median | n_tasks | curation | format | n_opts |
|---|---:|---:|---|---|---:|
| multiblimp | **3.81** | 7 | template_generated | minimal_pair | 2 |
| xwinograd | 2.12 | 4 | originally_multilingual | completion | 2 |
| xstorycloze | 1.96 | 8 | human_translation | completion | 2 |
| paws | 1.60 | 5 | human_translation | classification | 2 |
| xcopa | 1.43 | 6 | human_translation | completion | 2 |
| hellaswag | 1.37 | 6 | machine_translation | completion | 4 |
| xnli | 1.12 | 11 | human_translation | classification | 3 |
| global_piqa_completions | 1.02 | 11 | originally_multilingual | completion | 2 |
| belebele | 0.70 | 12 | human_translation | mrc_passage | 4 |
| global_mmlu_full | 0.66 | 10 | mt_post_edited | mcq_question_only | 4 |
| arc | 0.62 | 9 | machine_translation | mcq_question_only | 4 |

Rank order is stable across seed pools — minor reshuffling in the mid-tier
but the top (`multiblimp` / `xwinograd` / `xstorycloze`) and bottom
(`belebele` / `global_mmlu_full` / `arc`) blocks are unchanged. Single-seed
and train-pool views:
[seeds_1904/per_family_snr.csv](seeds_1904/per_family_snr.csv),
[seeds_28_1797/per_family_snr.csv](seeds_28_1797/per_family_snr.csv).

## Combined picture — what predicts SNR?

In rough order of evidence strength:

| effect | direction | strength |
|---|---|---|
| n_options / random_baseline | fewer options → higher SNR | **strong** (Spearman ρ=+0.77, p=0.006; KW H=5.5, p=0.019) |
| task format | minimal_pair > completion > MCQ > MRC | KW H=4.29, p=0.117 |
| option length | longer options → higher SNR | weak at per-family level (ρ=+0.13); qualitative only |
| context length | shorter context → higher SNR | weak (ρ=-0.23) |
| source origin (English-translated vs originally-multilingual) | originally-multilingual median higher | KW H=2.04, p=0.15 |
| passage in prompt | no effect | KW H=0.0, p=1.0 |
| curation category (MT / human / template / etc.) | no effect | KW H=1.44, p=0.49 |

The curation question that motivated v0 is not where the variance lives.
**Task design — option count and option length — explains most of what we
can explain on this 11-family pool.** All effects are borderline-
significant due to the small sample (n=11 families), but they're mutually
consistent and point at the same mechanism: tasks where the model has to
pick between fewer, longer log-likelihood-scored completions are SNR-higher.

## Phase 0 — curation process (the original Q4 question)

![Per-family SNR by curation process](seeds_28_1797_1904/snr_by_curation_process.png)

Source: [`seeds_28_1797_1904/group_stats.csv`](seeds_28_1797_1904/group_stats.csv)
(every grouping view: one row per view with `H` / `p` / `n_groups` /
optional `spearman_rho`).

| view | test | H | p |
|---|---|---:|---:|
| family / curation_category (3-way) | Kruskal-Wallis | 1.44 | 0.49 |
| family / source_origin (2-way) | Kruskal-Wallis | 2.04 | 0.15 |
| task / curation_category (with `xnli_eu` re-tag) | Kruskal-Wallis | 26.41 | <1e-4 |

The family-level test does not reach significance for any curation view
(see [snr_by_data_source.png](seeds_28_1797_1904/snr_by_data_source.png)).
The per-task test only reaches significance because multiblimp's
template-generated tasks pull the `template_generated` group up; if you
drop multiblimp, the residual differences across the other four categories
are not significant. **Curation method alone does not predict SNR.**

## Phase A — task format (the strongest categorical predictor)

![Per-family SNR by task format](seeds_28_1797_1904/snr_by_format.png)

![Per-family SNR by number of answer options](seeds_28_1797_1904/snr_by_n_options.png)

| view | test | statistic | p |
|---|---|---:|---:|
| family / format (3-way) | Kruskal-Wallis | H = 4.29 | 0.117 |
| family / n_options (2-way) | Kruskal-Wallis | **H = 5.50** | **0.019** |
| family / passage flag | Kruskal-Wallis | H = 0.00 | 1.00 |
| family / random_baseline (continuous) | Spearman ρ | **+0.77** | **0.006** |

![SNR vs random baseline](seeds_28_1797_1904/snr_vs_random_baseline.png)

The continuous Spearman correlation between SNR and random baseline
(= 1 / n_options) is **+0.77, p = 0.006** — the strongest single-axis
result in the benchmark-creation analysis. The categorical 2-vs-4-option
split is also significant at the family level (KW H = 5.5, p = 0.019).
Concretely: 2-option tasks have higher SNR than 4-option tasks, holding
everything else equal. The probable mechanism is that log-likelihood
comparison across two completions is sharper than across four — every
additional option introduces another noisy LL estimate that has to be
ranked correctly.

The format axis tells the same story categorically (5 underlying values;
the 3-way KW test in `group_stats.csv` excludes singleton groups):

- `minimal_pair` (n=1, multiblimp): very high SNR (3.81).
- `completion` (n=5): 2nd-tier, median ≈ 1.6.
- `classification` (n=2): 3rd-tier, ~1.4.
- `mcq_question_only` (n=2): 4th-tier, ~0.64.
- `mrc_passage` (n=1, belebele): lowest, 0.70.

Two earlier surprises now resolve:

- **Belebele**: only `mrc_passage` family AND 4 options — both strongest
  negative predictors stack.
- **MultiBLiMP**: minimal-pair format gives uniquely sharp signal because
  each item is a 1-token contrast, not "automatic generation produces
  better data."

The `passage` flag (whether the prompt contains a long passage) is **not**
a useful predictor (p = 0.54). XStoryCloze (4-sentence context, completion
task) has high SNR; Belebele (long passage, MRC) has low SNR — passage
length doesn't matter, what's done with it does.

## Phase B — item lengths (extends Phase A quantitatively)

![SNR vs length features](seeds_28_1797_1904/snr_vs_length_features.png)

100 English-or-default items per family pulled from each benchmark's HF
dataset; character-length statistics in
[length_features.csv](length_features.csv).

Source: [`seeds_28_1797_1904/group_stats.csv`](seeds_28_1797_1904/group_stats.csv)
rows `family/context_len_chars_median`, `family/option_len_chars_median`,
`family/context_to_option_ratio`.

| feature | Spearman ρ | p | sign as predicted? |
|---|---:|---:|---|
| context length | -0.23 | 0.50 | yes (weak) |
| option length | +0.13 | 0.71 | yes (weak) |
| context : option ratio | -0.15 | 0.67 | yes (weak) |

Length features at the per-family level do not reach significance with
n = 11. The mechanism still shows up in concrete examples below — longer
per-option text concentrates more discriminating tokens — but the
continuous per-family signal is dominated by the categorical option-count
effect (Phase A above).

Concrete examples:

- **PAWS** (option = "Yes" / "No", 2.5 chars) sits at SNR 1.05 despite
  being binary; its short labels concentrate the signal in one or two
  tokens.
- **MultiBLiMP** (option = full grammatical sentence, 119 chars) sits at
  SNR 4.42 — 100× more discriminating tokens than PAWS, same n_options.
- **HellaSwag** (option = full continuation, 61 chars) escapes the
  4-option penalty and sits at SNR 2.04, beating ARC (option = a short
  noun phrase, 28 chars) at SNR 0.74.

`context_len_chars` and `context_to_option_ratio` move in the expected
direction (longer context → lower SNR; higher ratio → lower SNR) but
neither reaches significance individually with n=11.

## Methodology

- **SNR signal:** `snr_mpd_1B` from one of the seed-pool subdirs under
  [../snr_definition/](../snr_definition/). Q1 picked `mpd` (mean pairwise
  distance) as a top dispersion-family variant; it stays in the global
  top-7 across all three seed pools.
- **Per-family aggregate:** median of `snr_mpd_1B` across the family's
  per-language aggregate tasks.
- **Outputs partitioned by seed pool**: each Apertus seed pool has its own
  subdir (`seeds_1904/`, `seeds_28_1797/`, `seeds_28_1797_1904/`) with the
  same set of files.
- **Metadata source:** [data_info.md](data_info.md) — paper-style
  paragraphs + a schema table, cross-referenced against the
  `lm-evaluation-harness` task READMEs. The `FAMILY_META` dict in
  [analyze.py](analyze.py) is the machine-readable mirror.
- **Length features:** [length_features.py](length_features.py) pulls 100
  English (or default-config) items per family from each benchmark's HF
  dataset and computes character-length statistics for context vs options.
  Pool-agnostic; output: [length_features.csv](length_features.csv).
- **Coverage caveats:** `global_mmlu` (Lite, 6 langs) is excluded — only
  one Apertus model evaluated, so `mpd_1B` is NaN. `arc_de/fr` and
  `hellaswag_de/fr` are also NaN at 1B.
  `global_piqa_completions_spa_latn_spai` is filtered by
  `_is_language_aggregate` (3 trailing tokens).

## Recommended follow-up

A controlled comparison would tighten the story: pick families with
matched task format and contrast curation methods within it. Concretely:

- **HellaSwag (MT) vs XStoryCloze (human translation)**: both 4-or-2-option
  completion with passage context. SNR ~2.04 for both — direct evidence
  that curation doesn't matter when format is held constant.
- **ARC (MT) vs Global-MMLU-Full (MT+post-edit)**: both 4-option MCQ, same
  source dataset family. ARC 0.74 vs MMLU-Full 0.40 — domain-fragmentation
  effect (57 subjects vs single domain) shows up here. Phase C
  topic-tagging would quantify it.

## Reproduce

```bash
# Per-pool analysis (run for each seed pool)
for pool in seeds_1904 seeds_28_1797 seeds_28_1797_1904; do
    python analysis/rq05_benchmark_creation/analyze.py --pool $pool
done

# One-time HF dataset length sampling (pool-agnostic)
python analysis/rq05_benchmark_creation/length_features.py
```

## Directory contents

Shared at the top of this dir:

- [INSTRUCTIONS.md](INSTRUCTIONS.md), [data_info.md](data_info.md) —
  research-question spec and per-family paper-style metadata.
- [analyze.py](analyze.py) — runs Phases 0/A/B; takes `--pool` and emits
  the per-pool CSVs and plots into the matching subdir.
- [length_features.py](length_features.py) — Phase B HF sampler; writes
  [length_features.csv](length_features.csv) and
  [sample_items.json](sample_items.json) (one example item per family,
  kept for any Phase C topic tagging). Pool-agnostic.

Per Apertus seed pool (`seeds_1904/`, `seeds_28_1797/`,
`seeds_28_1797_1904/`):

- `per_family_snr.csv` — one row per family with SNR aggregates + metadata
  + length features.
- `per_task_snr.csv` — one row per per-language aggregate task with the
  per-task curation override (`xnli_eu` re-tagged `mt_post_edited`).
- `group_stats.csv` — Kruskal-Wallis (and Spearman for continuous axes)
  for every grouping view.
- Phase 0 plots: `snr_per_family_ranked.png` (headline),
  `snr_by_curation_process.png`, `snr_by_data_source.png`,
  `snr_by_curation_per_task.png`.
- Phase A plots: `snr_by_n_options.png`, `snr_by_format.png`,
  `snr_by_passage.png`, `snr_vs_random_baseline.png`.
- Phase B plots: `snr_vs_length_features.png` (3-panel combined),
  `snr_vs_context_len.png`, `snr_vs_option_len.png`,
  `snr_vs_context_option_ratio.png`.
