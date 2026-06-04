# RQ2 — What makes a benchmark high-SNR?

> Why do some multilingual benchmarks separate models cleanly under the SNR
> framework and others don't? Is it curation quality, task format, item
> length, or the answer space?

Outputs under `pretraining/<pool>/`; headline numbers below are the
comprehensive pool `custom_swissai_hf` (3 seeds + external pretraining models),
cross-checked against the pure `seeds_28_1797_1904` pool.

## TL;DR (paper takeaways)

1. **The answer-count effect now lives in the above-random gate, upstream of
   SNR.** Every 4-option *translated knowledge* MCQA — `belebele`,
   `global_mmlu_full`, `truthfulqa`, `agieval`, `arabic_leaderboard` — sits at
   chance and is dropped before SNR is computed. Only **9 families survive the
   gate**, and 7 of them are 2-option. The penalty for a large answer space is
   real, but it's expressed as *exclusion*, not as a low SNR score.
2. **Among survivors, no single design feature is individually significant.**
   With the high-option families already gone, family-level Kruskal–Wallis on
   option count drops to **H = 1.78, p = 0.18** and on format to **H = 0, p =
   1.0** — there's too little variation left (n = 9, mostly 2-option) to resolve
   them. The mechanism still holds qualitatively (the survivors' top is all
   2-option), but the within-survivor test is underpowered.
3. **Curation method still explains nothing.** Family-level Kruskal–Wallis on
   curation is **H = 0.5, p = 0.78**; data source **p = 0.44**; reading passage
   **p = 1.0**. Once the gate fixes the answer space, how a benchmark was built
   does not predict its reliability.
4. **The two 4-option survivors are `hellaswag` and English `arc`** (SNR ≈ 2.05)
   — contentful completions / standards that clear chance, consistent with #1:
   it's the information in the *comparison* that matters, not the option count
   alone.

## Headline ranking (pool `custom_swissai_hf`)

Per-family median SNR @ 1B, **above-random families only** (the gate drops the
at-chance 4-option MCQA before this table)
([`per_family_snr.csv`](pretraining/custom_swissai_hf/per_family_snr.csv)):

| family | median SNR | n | curation | format | n_opts |
|---|---:|---:|---|---|---:|
| **`multiblimp`** | **3.85** | 7 | template_generated | minimal_pair | **2** |
| `paws` | 2.55 | 2 | human_translation | classification | **2** |
| `xwinograd` | 2.48 | 4 | originally_multilingual | completion | **2** |
| `xstorycloze` | 2.27 | 5 | human_translation | completion | **2** |
| `xcopa` | 2.06 | 4 | human_translation | completion | **2** |
| `hellaswag` | 2.05 | 4 | machine_translation | completion | 4 |
| `arc` | 2.05 | 2 | machine_translation | mcq_question_only | 4 |
| `global_piqa_completions` | 1.45 | 5 | originally_multilingual | completion | **2** |
| **`xnli`** | **1.15** | 7 | human_translation | classification | 3 |

`multiblimp` (template-generated minimal pairs, 2 options) tops the table; the
4-option survivors `hellaswag` / English `arc` sit mid-pack; `xnli` (3-option
NLI) is last. The previously bottom-ranked `belebele` / `global_mmlu_full` no
longer appear — they're gated out as at-chance.

![Per-family SNR ranking](pretraining/custom_swissai_hf/snr_per_family_ranked.png)

## Significance of each design axis (family-level Kruskal–Wallis)

From [`group_stats.csv`](pretraining/custom_swissai_hf/group_stats.csv) — over
the **9 above-random survivors** (high-option families already removed by the
gate, which is why the option/format axes lose power):

| axis | H | p | verdict |
|---|---:|---:|---|
| n_options | 1.78 | 0.18 | n.s. among survivors (penalty is upstream, in the gate) |
| format | 0.00 | 1.0 | n.s. (survivors are nearly all completion / 2-option) |
| data source | 0.60 | 0.44 | n.s. |
| **curation method** | **0.50** | **0.78** | **no effect** |
| reading passage | 0.00 | 1.0 | no effect |

(At the *task* level, curation still reaches significance — H = 13.5, p =
0.0036 — but that is the format/option confound leaking through, since each
curation method is tied to a format; the family-level test controls for it.)

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
