# RQ2 — What makes a benchmark high-SNR?

> Why do some multilingual benchmarks separate models cleanly under the SNR
> framework and others don't? Is it curation quality, task format, item
> length, or the answer space?

Outputs under `pretraining/<pool>/`; headline numbers below are the
comprehensive pool `custom_swissai_hf` (3 seeds + external pretraining models),
cross-checked against the pure `seeds_28_1797_1904` pool.

## TL;DR (paper takeaways)

1. **Curation method does *not* predict SNR.** At the family level (the honest
   unit — tasks inside a family share a translation pipeline) curation explains
   essentially none of the SNR variance: Kruskal–Wallis **H = 0.77, p = 0.68**.
   Data source (originally-multilingual vs translated) is also non-significant
   (**p = 0.10**), and presence of a reading passage has **zero** effect
   (**p = 1.0**).
2. **Task design — the answer space — is what predicts SNR.** Fewer candidate
   options ⇒ higher SNR: the top of the ranking is all **2-option** families
   (`multiblimp`, `xwinograd`, `xstorycloze`), the bottom is all **4-option**
   (`global_mmlu_full`, `arc`, `belebele`). Family-level Kruskal–Wallis on
   option count is **H = 3.7, p = 0.055** and on format **H = 5.0, p = 0.08** —
   the strongest axes available, borderline only because n = 11 families.
   Mechanism: every extra option adds another noisy log-likelihood to rank.
3. **The one 4-option outlier is `hellaswag`** (SNR 2.05) — long, contentful
   completions, not a question-only MCQ — which is consistent with #2: it's the
   *comparison* that matters, and HellaSwag's options are information-rich.

## Headline ranking (pool `custom_swissai_hf`)

Per-family median `snr_mpd_1B`
([`per_family_snr.csv`](pretraining/custom_swissai_hf/per_family_snr.csv)):

| family | median SNR | n | curation | format | n_opts |
|---|---:|---:|---|---|---:|
| **`multiblimp`** | **3.85** | 7 | template_generated | minimal_pair | **2** |
| `xwinograd` | 2.48 | 4 | originally_multilingual | completion | **2** |
| `xstorycloze` | 2.36 | 8 | human_translation | completion | **2** |
| `hellaswag` | 2.05 | 6 | machine_translation | completion | 4 |
| `paws` | 1.94 | 5 | human_translation | classification | **2** |
| `xcopa` | 1.51 | 6 | human_translation | completion | **2** |
| `global_piqa_completions` | 1.29 | 11 | originally_multilingual | completion | **2** |
| `xnli` | 1.15 | 11 | human_translation | classification | 3 |
| `global_mmlu_full` | 0.81 | 10 | mt_post_edited | mcq_question_only | 4 |
| `arc` | 0.78 | 9 | machine_translation | mcq_question_only | 4 |
| **`belebele`** | **0.70** | 12 | human_translation | mrc_passage | 4 |

`multiblimp` (template-generated minimal pairs, 2 options) tops every pool;
`belebele`/`arc`/`global_mmlu_full` (4-option question-only MCQ / passage MRC)
sit at the bottom regardless of curation quality. Rank order is stable across
seed pools.

![Per-family SNR ranking](pretraining/custom_swissai_hf/snr_per_family_ranked.png)

## Significance of each design axis (family-level Kruskal–Wallis)

From [`group_stats.csv`](pretraining/custom_swissai_hf/group_stats.csv):

| axis | H | p | verdict |
|---|---:|---:|---|
| **n_options** | **3.68** | **0.055** | strongest; fewer ⇒ higher SNR |
| format | 5.04 | 0.080 | minimal-pair/completion > MCQ |
| data source | 2.67 | 0.10 | n.s. |
| **curation method** | **0.77** | **0.68** | **no effect** |
| reading passage | 0.00 | 1.0 | no effect |

(At the *task* level, curation reaches significance — H = 19.7, p = 6e-4 — but
that is the format/option confound leaking through, since each curation method
is tied to a format; the family-level test controls for it.)

![SNR by answer-option count](pretraining/custom_swissai_hf/snr_by_n_options.png)

![SNR by curation process (no trend)](pretraining/custom_swissai_hf/snr_by_curation_process.png)

## SNR vs the random-chance baseline

![SNR vs random baseline](pretraining/custom_swissai_hf/snr_vs_random_baseline.png)

2-option (chance 0.5) families cluster at high SNR, 4-option (chance 0.25) at
low SNR — the same option-count effect viewed against the per-task random
baseline.

## Files

- `pretraining/<pool>/per_family_snr.csv`, `per_task_snr.csv` — SNR + design
  attributes per family / task.
- `…/group_stats.csv` — Kruskal–Wallis H/p for each design axis.
- `…/snr_by_*.png` — SNR distribution by curation, format, option count,
  passage, data source; `snr_per_family_ranked.png`; `snr_vs_random_baseline.png`;
  `snr_vs_length_features.png`.
