# Subsets that elevate SNR (and decision accuracy)

> Per benchmark, can a subset of subtasks — a subset of languages within a
> multilingual family, or a subset of MMLU subjects — give a higher SNR
> than the full set? And does the same subset hold across seeds?

## TL;DR

**Yes — the best subset usually beats the full set by a substantial
margin**, especially for multilingual MMLU-style benchmarks.

Three cases:

- **Case 1 — language subset of a multilingual family**: 1–5 languages
  beat the full set. Top gains: Belebele 350M `+0.89` SNR (full 1.97 → best
  4-lang subset 2.86); Global-MMLU 175M `+0.87` (full 2.14 → `global_mmlu_full_vi`
  alone 3.01); XWinograd 350M `+0.83`.
- **Case 2 — MMLU subject (averaged across 10 languages)**: a top-N subject
  subset tracks the full 57-subject set. 175M: `+0.96` SNR with
  `international_law` alone. 1B: `+0.88` with `virology|human_aging`.
- **Case 3 — MMLU subject per language**: per-language gains up to
  `+1.28` SNR (Spanish 175M, `formal_logic`).
- **Per-sample (cluster-only; default Option D `variance_prefilter`)**
  finds doc-id subsets with even larger gains — up to `+2.68` on
  `xcopa_sw` — but on a different scale (binary acc per item vs aggregate
  scores). Three more proposers (A/B/C) are available; see below.

**Stability across seed pools is uneven.** Case 2 (subject-level) is
highly stable: `world_religions`, `international_law`, `human_aging`,
`marketing`, `jurisprudence` recur as best 1–4-subject picks across
pools. Case 1 (language-level) is partially stable — `xcopa` collapsing
to one language and `arc` favouring English reproduces, but the
_winning_ language can flip. Case 3 (language+subject) is the most
pool-sensitive.

**Selection recipe:** treat best-subset picks as candidates; prefer
subsets that recur in both train and test pools.

## Main results — train pool (`seeds_28_1797`)

`summary.csv` ranks every (case, task, size) by
`snr_gain = best_snr − full_set_snr` across the three logical-subset cases
— 100 rows (13 multilingual families × 4 sizes for Case 1, 4 sizes for
Case 2, 10 languages × 4 sizes for Case 3).

The pattern stays consistent with prior single-seed runs: **the best
subset usually beats the full set substantially**, with the strongest
gains in `global_mmlu_full` per-language and the multilingual `arc`
family. With 6 (mix, seed) units per size (train pool) vs 3
(single-seed), the _absolute_ SNR levels are higher across the board
(roughly 1.4× the single-seed levels) and the gain magnitudes are
smaller in absolute terms but similar in _direction_.

### Case 1 — per-benchmark family (subtask = language)

For most multilingual families a 1–5-language subset beats the full
language set. Source:
[`seeds_28_1797/per_benchmark.csv`](seeds_28_1797/per_benchmark.csv)
(rows of the `summary.csv` filter `case == "per_benchmark"`):

| task             | size | full_set_snr | best_n | best_snr | snr_gain | best subset                                              |
| ---------------- | ---- | -----------: | -----: | -------: | -------: | -------------------------------------------------------- |
| belebele         | 350M |         1.97 |      4 |     2.86 | **0.89** | `belebele_tur\|belebele_eus\|belebele_swh\|belebele_hin` |
| global_mmlu_full | 175M |         2.14 |      1 |     3.01 | **0.87** | `global_mmlu_full_vi`                                    |
| xwinograd        | 350M |         2.04 |      2 |     2.87 | **0.83** | `xwinograd_en\|xwinograd_jp`                             |
| paws             | 175M |         1.61 |      1 |     2.21 | **0.61** | `paws_es`                                                |
| xnli             | 600M |         1.91 |      4 |     2.43 | **0.52** | `xnli_de\|xnli_hi\|xnli_ru\|xnli_zh`                     |
| xcopa            | 350M |         2.54 |      1 |     2.84 | **0.30** | `xcopa_tr`                                               |

xcopa's "collapses to `xcopa_tr`" pattern from prior runs holds weakly
(the gain is smaller at 0.30 because the full-set SNR is already at 2.54
— close to saturation).

Negative cases: `multiblimp`, `xstorycloze`, `hellaswag` saturate above
SNR ≈ 2.7 with the full language set, so no subset materially improves on
them. `mgsm_direct` is in the parquet but excluded (every row's
`primary_score` is NaN — see Limitations).

### Case 2 — `global_mmlu_full` subjects (subtask = subject, mean across 10 langs)

![Global-MMLU subject subset curves per size](seeds_28_1797_1904/global_mmlu_full_subjects.png)

Treating each MMLU subject as a subtask (averaged across the 10 GMF
languages) reproduces the upstream finding that a top-N subject subset
tracks the full set. Source:
[`seeds_28_1797/global_mmlu_full.csv`](seeds_28_1797/global_mmlu_full.csv).

| size | full_set_snr | best_n | best_snr | snr_gain | best subset                                                                            |
| ---- | -----------: | -----: | -------: | -------: | -------------------------------------------------------------------------------------- |
| 175M |         2.13 |      1 |     3.09 | **0.96** | `international_law`                                                                    |
| 350M |         2.20 |      5 |     2.83 |     0.63 | `marketing\|world_religions\|professional_law\|business_ethics\|jurisprudence`         |
| 600M |         2.06 |      4 |     2.89 | **0.84** | `logical_fallacies\|college_mathematics\|high_school_european_history\|moral_disputes` |
| 1B   |         2.15 |      2 |     3.03 | **0.88** | `virology\|human_aging`                                                                |

Best 1-subject subset typically picks a low-noise / high-spread subject
(`international_law`, `marketing`, `world_religions`). 10 GMF languages
averaged per cell — full coverage.

### Case 3 — `global_mmlu_full` subjects per language

Per-language gains are smaller than under single-seed because the
multi-seed signal pool already lifts the full-set SNR. But the top picks
per language are stable. Source:
[`seeds_28_1797/global_mmlu_full_per_language.csv`](seeds_28_1797/global_mmlu_full_per_language.csv).

| language | size | full_set_snr | best_n | best_snr | snr_gain | best subset                                                                                    |
| -------- | ---- | -----------: | -----: | -------: | -------: | ---------------------------------------------------------------------------------------------- |
| es       | 175M |         1.74 |      1 |     3.02 | **1.28** | `formal_logic`                                                                                 |
| sw       | 350M |         1.88 |      3 |     3.15 | **1.27** | `management\|miscellaneous\|prehistory`                                                        |
| sw       | 600M |         1.72 |      5 |     2.88 | **1.16** | `public_relations\|high_school_computer_science\|college_medicine\|electrical_engineering\|+1` |
| vi       | 1B   |         1.90 |      2 |     3.03 | **1.13** | `human_aging\|virology`                                                                        |
| en       | 175M |         1.86 |      1 |     2.89 | **1.03** | `international_law`                                                                            |
| ru       | 600M |         2.12 |      2 |     3.07 | **0.95** | `international_law\|philosophy`                                                                |

Plots per language under
`seeds_28_1797/global_mmlu_full_per_language_plots/`.

### Per-sample (Options A/B/C/D, cluster-only)

The per-sample sweep is cluster-only (needs `samples_*.jsonl`).
[`smooth_subtasks_per_sample.py`](../../multilingual/smooth_subtasks_per_sample.py)
implements four interchangeable proposers, one output dir each under
[per_sample/](per_sample/) (see its
[PROPOSALS.md](per_sample/PROPOSALS.md)):

- `variance_prefilter` (**D**, default) — drop "dead" samples, then rank
  survivors by per-sample SNR and sweep cumulatively on doc-ids.
- `greedy_snr_rank` (**A**) — same sweep, no prefilter (upstream baseline).
- `forward_greedy` (**B**) — forward-select the sample that maximises
  combined SNR, capped pool/budget (`--b-pool` / `--b-budget`).
- `irt_discrimination` (**C**) — `girth` 2PL fit, keep high-discrimination
  items, then rank by SNR. Exploratory: the checkpoint examinee pool is
  thin and correlated.

The numbers below are the **D** (`variance_prefilter`) run.
`per_sample/variance_prefilter/summary_all.csv` has 328 rows. Per-sample acc is binary
0/1, so the relative-std SNR primitive operates on a different scale
than per-task SNR (a single sample with
all-correct ckpts has noise=0 → infinite SNR, guarded to NaN). **Absolute SNR values are not directly comparable
to Case 1–3 numbers** above.

Top-5 (lang, task, size) by `snr_gain` (status = ok):

| lang | task              | size | n_total | n_after_prefilter | best_n | full_set_snr | best_snr | snr_gain |
| ---- | ----------------- | ---- | ------: | ----------------: | -----: | -----------: | -------: | -------: |
| sw   | xcopa_sw          | 1B   |     500 |               193 |     12 |         0.78 |     3.46 | **2.68** |
| eu   | paws_eu           | 175M |    1994 |               567 |      3 |         0.63 |     2.91 | **2.28** |
| es   | paws_es           | 1B   |    2000 |              1497 |     18 |         0.63 |     2.89 | **2.27** |
| th   | belebele_tha_Thai | 350M |     900 |               596 |      6 |         0.20 |     2.45 | **2.25** |
| eu   | paws_eu           | 1B   |    1994 |              1488 |     97 |         0.91 |     3.14 | **2.23** |

## Stability across seed pools

Compare `seeds_28_1797/summary.csv` with `seeds_1904/summary.csv`.

- **Case 2 (GMF subjects) is highly stable** — `world_religions`,
  `high_school_computer_science`, `human_aging`, `international_law`,
  `marketing`, `jurisprudence` recur as best 1–4-subject picks across
  both seed pools.
- **Case 1 (multilingual families) is partially stable** — `xcopa`
  collapsing to one language and `arc` favouring English variants
  reproduces in both pools, but the _winning_ language can flip (e.g.
  `xcopa_tr` at 350M in train, `xcopa_vi` at 350M in test).
- **Case 3 (GMF per-language) is the most pool-sensitive** — the argmax
  subject for a (lang, size) cell rarely repeats across pools except for
  the strongest cases (`human_aging` at 600M for es and vi;
  `international_law` for several languages at 175M).

Consistent with the broader `snr_definition/` finding: the overall
_ranking_ of subtasks by SNR is partially stable across seed pools, but
the _exact_ argmax is sensitive to which seed pool you trained on.
**For benchmark selection, use both pools' winners as candidates and
prefer subsets that recur in both lists.**

## Methodology

- **Apertus multi-seed parquet** (3 mixes × 3 seeds × 4 sizes = 36 models),
  last-5 ckpts per (size, mix, seed). SNR computed via
  `snr.metrics.signal_to_noise_ratio` exactly as in
  `compute_snr_small_scale`, but **grouped by `model` (each (mix, seed)
  is a separate training run)** so the signal pool has up to 9 model
  units per size instead of 3. External reference models (SmolLM3-3B,
  Olmo-3-7B, …) are included via the same `model` grouping but only
  contribute at their native sizes (3B/7B/8B/70B).
- **Combined-subset score** = mean across (model, step) of the included
  subtasks (relaxed inner-join — strict join leaves `arc` /
  `global_mmlu_full` mostly empty, since not every language is evaluated
  at every ckpt).
- **Sweep**: each subtask's standalone SNR is computed first; subtasks are
  added in descending-SNR order and the cumulative subset-SNR is
  recorded. Best subset = prefix that maximises the cumulative curve. A
  random-order baseline is computed alongside.
- **Per-sample analysis (Option D)**: variance prefilter to drop "dead"
  samples + vectorised per-sample SNR, then the same sorted-by-SNR
  cumulative sweep on doc-ids. **Cluster-only** (needs `samples_*.jsonl`).

## Limitations

- **Relaxed inner-join in `snr_for_subset`.** The combined-subset score at
  each `(model, step)` is the mean of _whichever subtasks happen to be
  present_ at that cell, not a fixed-set average. For sparse-coverage
  families like `arc` (not every language at every ckpt) the score's
  denominator changes from cell to cell, which can inflate or deflate
  both the signal and the noise. The strict alternative leaves most cells
  empty.
- **English `truthfulqa_mc1` is silently excluded from Case 1.**
  `assign_language` recognises it as English but its task name carries no
  language token, so `benchmark_family` puts it in a singleton family
  (`truthfulqa_mc1`), which the `len(ts) > 1` filter drops. The
  multilingual `truthfulqa_<lang>_mc1` variants form the `truthfulqa`
  family (7 langs) without an English baseline.
- **`mgsm_direct` is dropped at the loader.** All mgsm rows in
  `pretraining_custom-*.parquet` have `primary_score = NaN` even though
  the metrics dict carries an `exact_match` value. The parquet was built
  with `primary_metric = acc`, but mgsm reports `exact_match`. Fix
  belongs in the parquet generator, not in this repo.
- **Cases 2 & 3 stay Apertus-only.** The per-(lang, subject)
  `global_mmlu_full_<lang>_<subject>` facets only exist for the Apertus
  pretrains in the parquet — external reference models evaluate the
  per-language aggregate but not the subject facets.
- **Per-sample SNR is on binary acc.** The 2.68 headline gain for
  `xcopa_sw` is real but lives on a different scale than the per-task SNR
  figures in Cases 1–3.

## Reproduce

```bash
# Subtask-level subset search (Cases 1-3) — each Apertus seed pool
for pool in seeds_1904 seeds_28_1797 seeds_28_1797_1904; do
    python multilingual/smooth_subtasks.py --pool $pool   # → per_subtask/<pool>/
done

# Per-sample subset search — runs all four proposers (A/B/C/D) by default,
# one output dir each under per_sample/. Cluster only (needs samples_*.jsonl).
for pool in seeds_1904 seeds_28_1797 seeds_28_1797_1904; do
    python multilingual/smooth_subtasks_per_sample.py --pool $pool
done
# A single method, e.g. just the default D:
#   ... --method variance_prefilter   (or --method D)

# Compare the four proposers + extract paper highlights
#   → per_sample/comparison/{method_comparison.csv,method_summary.csv,highlights.md}
python multilingual/compare_per_sample_methods.py

# Reuse only the committed Option-D intermediates (no raw samples needed):
# informative-fraction, gain distribution, cross-size subset/SNR stability
#   → per_sample/variance_prefilter/analysis/{*.csv,highlights.md}
python multilingual/analyze_per_sample_d.py
```

## Directory layout

```
results/smooth_subtasks/
├── per_subtask/                  ← subtask-level subset search (Cases 1-3)
│   ├── seeds_28_1797/            ← train pool
│   │   ├── summary.csv           ← 100 rows ranked by snr_gain
│   │   ├── per_benchmark.csv + per_benchmark_plots/      ← Case 1
│   │   ├── global_mmlu_full.csv + .png                   ← Case 2
│   │   └── global_mmlu_full_per_language.csv + plots/    ← Case 3
│   ├── seeds_1904/               ← held-out test pool (same layout)
│   └── seeds_28_1797_1904/       ← pooled all seeds (recommended)
└── per_sample/                   ← per-sample subset search, cluster-only
    ├── PROPOSALS.md              ← A/B/C/D design notes
    ├── comparison/               ← cross-method tables + paper highlights
    │   └── method_comparison.csv, method_summary.csv, highlights.md
    └── <method>/                 ← one per proposer: variance_prefilter (D),
        ├── summary_all.csv         greedy_snr_rank (A), forward_greedy (B),
        └── <lang>/<task>/          irt_discrimination (C). <lang>/<task>
                                     subdirs hold summary.csv, ranked_samples.csv,
                                     best_subset_<size>.txt, cumulative_snr.png
```

The `comparison/` dir is built by
[`multilingual/compare_per_sample_methods.py`](../../multilingual/compare_per_sample_methods.py),
which merges the four `summary_all.csv` roll-ups into a per-cell table,
per-method aggregates (win rate, median gain, subset size), best-subset
overlap (A vs D, A vs C), and a `highlights.md` with the headline numbers
worth lifting into a paper.

Per seed pool (under `per_subtask/<pool>/`):

- `summary.csv` — every (case, task, size) ranked by `snr_gain`. Built by
  `build_summary` in
  [`multilingual/smooth_subtasks.py`](../../multilingual/smooth_subtasks.py).
- `per_benchmark.csv` + `per_benchmark_plots/` — Case 1 outputs (one PNG
  per multilingual family).
- `global_mmlu_full.csv` + `global_mmlu_full_subjects.png` — Case 2
  outputs.
- `global_mmlu_full_per_language.csv` +
  `global_mmlu_full_per_language_plots/` — Case 3 outputs (one PNG per
  language, 10 in total).

`per_sample/<method>/` (cluster-only) — one dir per proposer
(`variance_prefilter`, `greedy_snr_rank`, `forward_greedy`,
`irt_discrimination`), each holding a `summary_all.csv` roll-up plus one
`<lang>/<task>/` subdir per language-benchmark. Each per-task summary.csv
now carries `language` and `task` columns. `PROPOSALS.md` (the design
notes) sits at the `per_sample/` root.
