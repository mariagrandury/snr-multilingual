# RQ4 — Can a subset of subtasks give higher SNR than the full set?

> Per benchmark, does a subset of subtasks — a subset of *languages* in a
> multilingual family, or a subset of *subjects* in MMLU, or a subset of
> individual *items* — give higher SNR than the full set, and does the same
> subset hold across seeds and scales?

Subtask-level outputs under `pretraining/<pool>/`; per-item (Option D) under
`per_sample/variance_prefilter/`. The three subtask cases: **Case 1** language
subset of a family · **Case 2** MMLU subject subset (mean over 10 languages) ·
**Case 3** MMLU subject subset per language.

## TL;DR (paper takeaways)

1. **A subset beats the full set in essentially every (benchmark, size) cell,
   usually a *single* subtask, by a large margin.** Top subtask-level gains on
   `custom_swissai_hf` (`snr_gain = best − full`):

   | case | task / size | full → best SNR | gain | winning subset |
   |---|---|---|---:|---|
   | Case 2 | `global_mmlu_full` 175M | 2.12 → **3.65** | **+1.52** | `medical_genetics` (1 subject) |
   | Case 3 | `global_mmlu_full_tr` 1B | 1.85 → 3.41 | +1.56 | 8 subjects |
   | Case 1 | `xstorycloze` 3B | 0.54 → 1.99 | +1.45 | `xstorycloze_eu` (1 lang) |

2. **Case 2 (MMLU subjects) is the strongest and most stable lever:** a 1–2
   subject subset matches or beats the full ~48-subject set across sizes
   (175M `medical_genetics` alone; 600M `human_sexuality|clinical_knowledge`;
   1B `human_aging|virology`). Recurring high-SNR subjects across pools:
   **`medical_genetics`, `human_aging`, `international_law`, world-history**.
3. **Case 1 (languages): a few languages carry the family.** `eu`, `zh`, `vi`
   recur as high-signal picks; `arc` collapses to a single language and `xcopa`
   to one or two. The *winning* language can flip across seeds — treat picks as
   candidates and prefer those that recur in train+test pools.
4. **Per-item selection (Option D) gives even larger gains but does NOT
   transfer across scale — so prefer subtask-level selection.** A per-sample
   subset beats the full set in **100%** of 320 cells (median **+0.64**, max
   **+2.68** on `xcopa_sw` 1B), keeping a median of just **2%** of items. But
   cross-size best-subset **Jaccard ≈ 0.03** and SNR-rank **Spearman ≈ 0.05**
   (≈ 0) — the per-item picks are scale-specific (and estimation-noisy), which
   is the empirical argument for operating at the subtask level (Cases 1–3),
   not the item level.

## Case 2 — MMLU subject subset (pool `custom_swissai_hf`)

A handful of subjects reaches the full-set SNR ceiling
([`global_mmlu_full.csv`](pretraining/custom_swissai_hf/global_mmlu_full.csv)):

| size | full-set SNR | best subset | best SNR | gain |
|---|---:|---|---:|---:|
| 175M | 2.12 | **`medical_genetics`** | **3.65** | **+1.52** |
| 600M | 2.18 | `human_sexuality \| clinical_knowledge` | 3.45 | +1.27 |
| 1B | 2.43 | `human_aging \| virology` | 3.37 | +0.93 |

![MMLU subject-subset SNR sweep](pretraining/custom_swissai_hf/global_mmlu_full_subjects.png)

## Case 1 — language subset of a family

Full `summary.csv` ranks all (case, task, size) by `snr_gain`
([`summary.csv`](pretraining/custom_swissai_hf/summary.csv)); per-family sweep
curves in [`per_benchmark_plots/`](pretraining/custom_swissai_hf/per_benchmark_plots/).
Representative single-language winners: `xstorycloze_eu`, `xcopa_vi`,
`multiblimp_eng`, `arc_eu`.

## Per-item (Option D) — `per_sample/variance_prefilter/analysis/`

From [`highlights.md`](per_sample/variance_prefilter/analysis/highlights.md):

| metric | value |
|---|---|
| cells where a subset beats the full set | **100%** (320/320) |
| median SNR gain | **+0.64** |
| max SNR gain | **+2.68** (`xcopa_sw`, 1B) |
| median % of items kept | **2%** |
| cross-size subset **Jaccard** | **0.03** |
| cross-size SNR-rank **Spearman** | **0.05** |

(Per-sample SNR is on binary item accuracy — not comparable to subtask-level
SNR. Proposers B/C are cluster-only; A ≡ D on committed data.)

## Selection recipe

Treat best-subset picks as candidates; **prefer subsets that recur in both
train and test seed pools** (Case 2 subjects are the safest; per-item picks the
least transferable). Outputs regenerated for `seeds_28_1797_1904` and
`custom_swissai_hf`.

## Files

- `pretraining/<pool>/summary.csv` — every (case, task, size) by `snr_gain`.
- `…/per_benchmark.csv` (Case 1), `global_mmlu_full.csv` (Case 2),
  `global_mmlu_full_per_language.csv` (Case 3) + their `*_plots/`.
- `per_sample/variance_prefilter/analysis/` — Option-D size distribution,
  cross-size Jaccard/Spearman, `highlights.md`.
