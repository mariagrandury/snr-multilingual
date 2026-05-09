# `smooth_subtasks/` — find subsets that elevate SNR and DA

> Step-by-step plan: [INSTRUCTIONS.md](INSTRUCTIONS.md). Use one
> Claude session per research question — see
> [../PARALLEL_SESSIONS.md](../PARALLEL_SESSIONS.md).

## Research questions

Per benchmark, can a subset elevate SNR (and thus DA), per checkpoint
or per size? Two flavors:

1. **Logical subset** — language inside a multilingual family, or the
   MMLU subject taxonomy inside `global_mmlu_full`.
2. **Per-sample subset** — find the doc-id subset that maximises SNR.

## Setup

- Apertus 12-model parquet, last-5 ckpts per (size, mix); SNR computed
  via `snr.metrics.signal_to_noise_ratio` exactly as in
  `compute_snr_small_scale`.
- Combined-subset score = mean across (mix, step) of the included
  subtasks (relaxed inner-join — strict join leaves arc /
  global_mmlu mostly empty, since not every language is evaluated at
  every ckpt).
- Sweep: each subtask's standalone SNR is computed first; subtasks are
  added in descending-SNR order and the cumulative subset-SNR is
  recorded. The best subset is the prefix that maximises that
  cumulative curve. A random-order baseline is computed alongside.
- Per-sample analysis (Option D): variance prefilter to drop "dead"
  samples + vectorised per-sample SNR, then the same sorted-by-SNR
  cumulative sweep on doc-ids. **Cluster-only** (needs
  `samples_*.jsonl`).

## Main results

`summary.csv` ranks every (case, task, size) by
`snr_gain = best_snr − full_set_snr` across the three logical-subset
cases — 96 rows (13 multilingual families × 4 sizes for Case 1, 4 sizes
for Case 2, 10 languages × 4 sizes for Case 3). The pattern: the best
subset usually beats the full set substantially at smaller sizes
(175M–600M) and the gap narrows at 1B as the full-set SNR rises on its
own.

### Case 1 — per-benchmark family (subtask = language)

For most multilingual families a 1–4-language subset beats the full
language set, often by ≥1 SNR point. The strongest case is **xcopa**:
across 175M / 350M / 600M the entire benchmark's signal collapses to
`xcopa_tr` (Turkish) — every other language adds noise. At 1B the best
subset shifts to `xcopa_vi`. xcopa is also the family with the lowest
full-set SNR, so the gain matters most here.

![xcopa — combined SNR vs. subset size, per model size](per_benchmark_plots/xcopa.png)

Top-5 (case, task, size) by `snr_gain`:

| task | size | full_set_snr | best_n | best_snr | snr_gain | best subset |
|---|---|---:|---:|---:|---:|---|
| xcopa | 350M | 0.24 | 1 | 2.14 | **1.90** | `xcopa_tr` |
| arc | 175M | 0.63 | 1 | 2.38 | **1.75** | `arc_easy` |
| arc | 350M | 0.70 | 2 | 2.36 | **1.66** | `arc_challenge\|arc_easy` |
| xwinograd | 1B | 0.92 | 1 | 2.56 | **1.64** | `xwinograd_zh` |
| truthfulqa | 350M | 0.58 | 1 | 2.15 | **1.57** | `truthfulqa_hi_mc1` |

Negative cases worth noting (full set ≈ best): `multiblimp`,
`xstorycloze`, and `hellaswag` already saturate near SNR ≈ 2.2–2.4
with the full language set, so any one-language subset can only match
it. `global_mmlu` (the short variant) yields no signal at any size —
only the `_full` variant has usable SNR on these models. `mgsm_direct`
is in the parquet (7 languages) but is excluded because every row's
`primary_score` is NaN — see "Limitations" below.

### Case 2 — `global_mmlu_full` subjects (subtask = subject, mean across 10 langs)

Treating each MMLU subject as a subtask (averaged across the 10 GMF
languages) reproduces the upstream finding that a top-N subject subset
tracks the full set, but the gains are modest on Apertus — the
saturating averaged-across-languages signal already pushes full-set
SNR above 1.5 at every size, so there's less headroom. Best subsets
range from 1 to 4 subjects; `snr_gain` peaks at 0.84 at 600M
(`marketing|world_religions|clinical_knowledge`).

![global_mmlu_full subjects (subtask = subject, mean across 10 langs)](global_mmlu_full_subjects.png)

| size | full_set_snr | best_n | best_snr | snr_gain | best subset |
|---|---:|---:|---:|---:|---|
| 175M | 1.68 | 2 | 2.11 | 0.42 | `high_school_us_history\|miscellaneous` |
| 350M | 2.23 | 4 | 2.33 | 0.10 | `high_school_chemistry\|stem\|elementary_mathematics\|professional_accounting` |
| 600M | 1.52 | 3 | 2.36 | **0.84** | `marketing\|world_religions\|clinical_knowledge` |
| 1B | 1.48 | 1 | 1.92 | 0.44 | `world_religions` |

Mean #languages averaged per (mix, step, subject) = 10.0 across all
sizes (full coverage), so no language-coverage caveat applies here.

### Case 3 — `global_mmlu_full` subjects per language

The strongest per-language gain is on **Vietnamese at 600M**: the full
set scores SNR ≈ 0.29, and a two-subject subset
(`high_school_computer_science|college_computer_science`) lifts it to
2.24 — a gain of 1.95, the largest in the entire `summary.csv`.

![global_mmlu_full Vietnamese — subject sweep, per model size](global_mmlu_full_per_language_plots/vi.png)

Top-5 per-language gains:

| language | size | full_set_snr | best_n | best_snr | snr_gain | best subset |
|---|---|---:|---:|---:|---:|---|
| vi | 600M | 0.29 | 2 | 2.24 | **1.95** | `high_school_computer_science\|college_computer_science` |
| es | 600M | 0.61 | 1 | 2.32 | **1.71** | `human_aging` |
| zh | 175M | 0.89 | 1 | 2.11 | **1.22** | `high_school_computer_science` |
| en | 1B | 0.96 | 1 | 2.18 | **1.22** | `moral_disputes` |
| ar | 600M | 1.02 | 5 | 2.22 | **1.20** | `professional_law\|world_religions\|marketing\|professional_psychology\|…` |

Plots for the other nine languages live under
`global_mmlu_full_per_language_plots/`.

### Per-sample (Option D, cluster-only)

The per-sample sweep was run on the cluster (eval_logs samples are not
in the parquet) and its outputs are committed under [per_sample/](per_sample/)
for audit. Method: drop "dead" samples with the variance prefilter,
then rank surviving samples by per-sample SNR and walk the cumulative
sweep — same shape as the logical-subset sweep, but with one doc-id as
the atomic unit.

`per_sample/summary_all.csv` has 328 rows (320 `ok`, 8 `no_data`). The
gains are larger than the logical-subset cases — but note that
per-sample acc is binary 0/1, so the relative-std SNR primitive
operates on a different scale than per-task SNR (a single sample with
all-correct ckpts has noise=0 → infinite SNR, guarded to NaN). The
absolute SNR values are not directly comparable to Case 1–3 numbers.

Top-5 (lang, task, size) by `snr_gain` (status = ok):

| lang | task | size | n_total | n_after_prefilter | best_n | full_set_snr | best_snr | snr_gain |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sw | xcopa_sw | 1B | 500 | 193 | 12 | 0.78 | 3.46 | **2.68** |
| eu | paws_eu | 175M | 1994 | 567 | 3 | 0.63 | 2.91 | **2.28** |
| es | paws_es | 1B | 2000 | 1497 | 18 | 0.63 | 2.89 | **2.27** |
| th | belebele_tha_Thai | 350M | 900 | 596 | 6 | 0.20 | 2.45 | **2.25** |
| eu | paws_eu | 1B | 1994 | 1488 | 97 | 0.91 | 3.14 | **2.23** |

To regenerate (cluster only):
```bash
python multilingual/smooth_subtasks_per_sample.py
```

## Limitations

- **Relaxed inner-join in `snr_for_subset`**. The combined-subset
  score at each `(mix, step)` is the mean of *whichever subtasks
  happen to be present* at that cell, not a fixed-set average. For
  sparse-coverage families like `arc` (not every language at every
  ckpt) the score's denominator changes from cell to cell, which can
  inflate or deflate both the signal (between-mix dispersion) and the
  noise (pooled std). Documented in `snr_for_subset`'s docstring; the
  strict alternative leaves most cells empty.
- **English `truthfulqa_mc1` is silently excluded from Case 1.**
  `assign_language` recognises it as English (via
  `_ENGLISH_ONLY_TASKS`) but its task name carries no language token,
  so `benchmark_family` puts it in a singleton family
  (`truthfulqa_mc1`), which the `len(ts) > 1` filter then drops. The
  multilingual `truthfulqa_<lang>_mc1` variants form the `truthfulqa`
  family (7 langs) without an English baseline.
- **`mgsm_direct` is dropped at the loader.** All 157 mgsm rows in
  `pretraining_custom-*.parquet` have `primary_score = NaN` even
  though the metrics dict carries an `exact_match` value. The parquet
  was built with `primary_metric = acc`, but mgsm reports
  `exact_match`. The fix belongs in the parquet generator
  (`swissai-evals-post-train`), not in this repo.
- **Per-sample SNR is on binary acc.** The 2.68 headline gain for
  `xcopa_sw` is real but lives on a different scale than the per-task
  SNR figures in Cases 1–3.

## Directory contents

- [INSTRUCTIONS.md](INSTRUCTIONS.md) — execution plan.
- [summary.csv](summary.csv) — every (case, task, size) ranked by
  `snr_gain` (96 rows). Built by `build_summary` in
  [`multilingual/smooth_subtasks.py`](../../multilingual/smooth_subtasks.py).
- [per_benchmark.csv](per_benchmark.csv) +
  [per_benchmark_plots/](per_benchmark_plots/) — Case 1 outputs (one
  PNG per multilingual family, 13 in total: arc, belebele,
  global_mmlu, global_mmlu_full, global_piqa_completions, hellaswag,
  multiblimp, paws, truthfulqa, xcopa, xnli, xstorycloze, xwinograd).
- [global_mmlu_full.csv](global_mmlu_full.csv) +
  [global_mmlu_full_subjects.png](global_mmlu_full_subjects.png) —
  Case 2 outputs.
- [global_mmlu_full_per_language.csv](global_mmlu_full_per_language.csv)
  + [global_mmlu_full_per_language_plots/](global_mmlu_full_per_language_plots/)
  — Case 3 outputs (one PNG per language, 10 in total).
- [per_sample/](per_sample/) — Option D cluster-run outputs:
  `PROPOSALS.md`, `summary_all.csv`, plus one subdir per language
  (`ar`, `de`, `en`, `es`, `eu`, `fr`, `hi`, `ja`, `ru`, `sw`, `th`,
  `tr`, `vi`, `zh`) each holding one subdir per task.
