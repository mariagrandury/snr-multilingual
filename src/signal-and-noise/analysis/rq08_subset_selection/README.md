# RQ8 — Can a subset of subtasks give higher SNR than the full set?

## Research question

> Per benchmark, does a subset of subtasks — a subset of *languages* in a
> multilingual family, or a subset of *subjects* in MMLU, or a subset of
> individual *items* — give higher SNR than the full set, and does the same
> subset hold across seeds and scales?

<!-- BEGIN auto:highlight (smooth_subtasks.py --pool predictivity) -->
## Highlighted result

- **`multiblimp` 350M (per_benchmark)** — a subset beats the full set: SNR **2.99 → 4.40** (**+1.42**) with `multiblimp_deu|multiblimp_eng|multiblimp_fra`.
- **`arc` 1B (per_benchmark)** — a subset beats the full set: SNR **2.61 → 3.83** (**+1.22**) with `arc_challenge`.
- **`global_mmlu_full` 600M (global_mmlu_full_subjects)** — a subset beats the full set: SNR **2.14 → 3.33** (**+1.18**) with `human_aging`.
- **Median gain by case** — global_mmlu_full_subjects 1.03; global_mmlu_full_per_language 0.83; per_benchmark 0.24 (SNR units; a subset only helps where the gain clears the seed noise reported in rq03).
- **Selection null** — the best prefix is chosen on the numbers it is scored on, so `best ≥ full` always; against 100 random subsets of the same size, **19 of 53** swept cells beat the null's 95th percentile: `multiblimp` 350M, `global_mmlu_full` 600M, `multiblimp` 175M, `global_mmlu_full_sr` 350M, `multiblimp` 1.7B.
<!-- END auto:highlight -->

## Experimental setup

Subtask-level outputs under `pretraining/<pool>/` (ladder pool `predictivity`;
the 36-sweep pools as history); per-item (Option D) under
`per_sample/variance_prefilter/` (36-sweep only — it needs the per-sample
files that live on the cluster). Three subtask cases plus a per-item view;
on the ladder the multilingual families span up to 100 languages, so Case 1
asks which language subset of Belebele, Global-PIQA, MultiBLiMP or
Global-MMLU carries the family's signal, and the per-language BPB family
(`bpb`) is swept the same way (which languages' BPB make the sharpest
macro-average):

- **Case 1** — language subset of a multilingual family. `task` = the benchmark
  family (arc, belebele, global_mmlu, xnli, …); `subtask` = the per-language
  tasks in that family (arc_de, arc_es, …). Which language subset, ordered by
  per-language SNR, gives the highest combined SNR for the family?
- **Case 2** — MMLU subject subset (mean over the global_mmlu languages the
  pool carries, 37 on the ladder). `task` = `global_mmlu_full`; `subtask` = one
  subject (leaves only), whose per-(model, ckpt) score is the mean across languages.
- **Case 3** — MMLU subject subset per language (no cross-language averaging).
  `task` = `global_mmlu_full_<lang>`; `subtask` = one subject within that
  language.
- **Per-item (Option D)** — `per_sample/variance_prefilter/`. Subset selection at
  the individual-item level (per-sample SNR on binary item accuracy — not
  comparable to subtask-level SNR).

## Methodology

- **Subset SNR.** `signal_to_noise_ratio` over per-*model* last-N-checkpoint
  arrays, one array per training run (seed replicates and external models
  are separate runs), with the noise pooled across runs as in upstream's
  `compute_snr_small_scale` (see the comparability note below). A
  combined subset averages the per-(model, step) scores across its subtasks
  first.
- **Sweep.** Per (task, size) the subtasks are ranked by standalone SNR and
  the cumulative prefixes 1..N are scored; `best_n` / `best_subset` is the
  prefix with the highest SNR and `snr_gain = best − full`.
- **Gate.** The rq00 above-random mask: a language task at chance at a size
  is not swept (Case 1), and a language whose `global_mmlu_full_<lang>`
  aggregate is at chance is skipped (Cases 2 and 3). The four MMLU category
  roll-ups are excluded from the subject lists.
- **Selection null.** The prefix is chosen on the same numbers it is scored
  on, so `best ≥ full` by construction. For every swept cell, 100 random
  subsets of the best size give `null_snr_p95`; `gain_over_null =
  best − null_p95` is the part of the gain that is not selection.

<!-- BEGIN auto:results (smooth_subtasks.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq08_subset_selection/smooth_subtasks.py --pool predictivity`.

**Top subset gains** — every (case, task, size) ranked by `snr_gain = best − full`:

| case | task | size | full → best SNR | +gain | null p95 | best subset |
|---|---|---|---|---|---|---|
| per_benchmark | `multiblimp` | 350M | 2.99 → 4.40 | +1.42 | 3.63 | `multiblimp_deu` \| `multiblimp_eng` \| `multiblimp_fra` |
| per_benchmark | `arc` | 1B | 2.61 → 3.83 | +1.22 | 3.83 | `arc_challenge` |
| global_mmlu_full_subjects | `global_mmlu_full` | 600M | 2.14 → 3.33 | +1.18 | 3.22 | `human_aging` |
| per_benchmark | `multiblimp` | 175M | 3.22 → 4.24 | +1.02 | 3.67 | `multiblimp_rus` |
| global_mmlu_full_per_language | `global_mmlu_full_sr` | 350M | 2.66 → 3.65 | +0.99 | 3.48 | `formal_logic` |
| per_benchmark | `bpb` | 175M | 3.11 → 4.09 | +0.99 | 4.09 | `bpb_rus_Cyrl` |
| per_benchmark | `multiblimp` | 1.7B | 3.03 → 3.94 | +0.91 | 3.31 | `multiblimp_deu` \| `multiblimp_rus` \| `multiblimp_eng` |
| global_mmlu_full_per_language | `global_mmlu_full_ms` | 350M | 2.53 → 3.43 | +0.91 | 2.96 | `prehistory` \| `professional_psychology` |
| global_mmlu_full_subjects | `global_mmlu_full` | 350M | 3.23 → 4.09 | +0.87 | 3.91 | `nutrition` |
| per_benchmark | `multiblimp` | 600M | 2.92 → 3.73 | +0.81 | 3.36 | `multiblimp_eng` \| `multiblimp_deu` |
| global_mmlu_full_per_language | `global_mmlu_full_lt` | 350M | 2.69 → 3.44 | +0.75 | 3.00 | `anatomy` \| `international_law` \| `professional_psychology` \| `formal_logic` \| `… (+4)` |
| per_benchmark | `xcopa` | 175M | 2.54 → 3.20 | +0.66 | 3.20 | `xcopa_id` |

![](pretraining/predictivity/global_mmlu_full_subjects.png)
<!-- END auto:results -->

## External model-set tier (`all/external`, 36-sweep)

Run on the `external` tier the SNR is **cross-model dispersion** over the
270M…70B external ladder (no mixture axis), and only the three subtask cases run
(no per-item Option-D pass). Outputs in `all/external/`; regenerate with
`python analysis/rq08_subset_selection/smooth_subtasks.py --pool external`.

**Subset-beats-full survives the change of model set.** Even though the external
ladder's full-set SNRs are low (the heterogeneous model families wash out a
benchmark's combined signal), a small subtask subset recovers it — the gains are
as large as on the custom pool:

| case | task | size | full → best SNR | +gain | best subset |
|---|---|---|---|---|---|
| case3_global_mmlu_full_per_language | `global_mmlu_full_ar` | 1B | 0.18 → 1.95 | +1.77 | `high_school_chemistry` |
| case1_per_benchmark | `truthfulqa` | 1B | 0.52 → 1.96 | +1.44 | `truthfulqa_hi_mc2` \| `truthfulqa_vi_mc2` |
| case1_per_benchmark | `paws` | 3B | 0.37 → 1.81 | +1.44 | `paws_eu` |
| case1_per_benchmark | `truthfulqa` | 3B | 0.66 → 1.92 | +1.26 | `truthfulqa_es_mc1` |
| case1_per_benchmark | `global_piqa_completions` | 1B | 0.56 → 1.73 | +1.16 | `global_piqa_completions_eng_latn` |
| case3_global_mmlu_full_per_language | `global_mmlu_full_vi` | 1B | 0.87 → 1.93 | +1.06 | `high_school_chemistry` \| `elementary_mathematics` |

![MMLU subject subsets vs full set (external)](all/external/global_mmlu_full_subjects.png)

The recurring levers carry over (MMLU subject subsets, per-language TruthfulQA
`mc` splits, `paws_eu`), but the **exact** winning subjects differ from the custom
pool (`high_school_chemistry` recurs here rather than world-history), reinforcing
RQ4's headline: treat subset picks as a robust *lever*, not a transferable exact
subset. Per-benchmark curves of the sweep (full vs best subset across the external
ladder) for the highest-gain families:

![TruthfulQA subset sweep (external)](all/external/per_benchmark_plots/truthfulqa.png)


## Results from the 36-model sweep (2026-06, superseded)

The numbers below were generated on the 36-model sweep (4 sizes × 3 data mixtures × 3 seeds, 12 languages, pool `custom_swissai_hf` unless stated) and are kept as history; the predictivity ladder regenerates the blocks above.

### Highlighted result

- **`global_mmlu_full_vi` 1B (global_mmlu_full_per_language)** — a subset beats the full set: SNR **2.05 → 4.01** (**+1.95**) with `high_school_world_history|business_ethics|marketing`.
- **`global_mmlu_full` 175M (global_mmlu_full_subjects)** — a subset beats the full set: SNR **2.12 → 3.65** (**+1.52**) with `medical_genetics`.
- **`paws` 3B (per_benchmark)** — a subset beats the full set: SNR **0.37 → 1.81** (**+1.44**) with `paws_eu`.
- **MMLU subject subsets are the most/most-stable lever** — a 1–2 subject subset matches or beats the full ~48-subject set across sizes (`medical_genetics`, `human_aging`, `international_law`, world-history recur).
- **Per-item (per-sample) ranking is mostly noise / overfits across scale** — per-sample subsets give even larger gains but their best picks barely overlap across sizes (Jaccard ≈ 0.03, SNR-rank Spearman ≈ 0.05), so prefer subtask-level selection.

### Results

Headline numbers from the `custom_swissai_hf` pool. Regenerate with `python analysis/rq08_subset_selection/smooth_subtasks.py --pool custom_swissai_hf`.

**Top subset gains** — every (case, task, size) ranked by `snr_gain = best − full`:

| case | task | size | full → best SNR | +gain | best subset |
|---|---|---|---|---|---|
| global_mmlu_full_per_language | `global_mmlu_full_vi` | 1B | 2.05 → 4.01 | +1.95 | `high_school_world_history` \| `business_ethics` \| `marketing` |
| global_mmlu_full_subjects | `global_mmlu_full` | 175M | 2.12 → 3.65 | +1.52 | `medical_genetics` |
| per_benchmark | `paws` | 3B | 0.37 → 1.81 | +1.44 | `paws_eu` |
| global_mmlu_full_per_language | `global_mmlu_full_sw` | 600M | 1.71 → 3.07 | +1.36 | `public_relations` \| `philosophy` |
| global_mmlu_full_per_language | `global_mmlu_full_vi` | 350M | 1.97 → 3.31 | +1.34 | `prehistory` \| `college_medicine` \| `high_school_geography` |
| global_mmlu_full_per_language | `global_mmlu_full_zh` | 175M | 2.15 → 3.46 | +1.31 | `high_school_world_history` \| `international_law` |
| per_benchmark | `truthfulqa` | 3B | 0.66 → 1.92 | +1.26 | `truthfulqa_es_mc1` |
| per_benchmark | `belebele` | 350M | 2.28 → 3.44 | +1.16 | `belebele_swh_Latn` \| `belebele_hin_Deva` \| `belebele_eus_Latn` |
| global_mmlu_full_per_language | `global_mmlu_full_ru` | 600M | 2.09 → 3.24 | +1.15 | `medical_genetics` \| `international_law` \| `high_school_statistics` |
| global_mmlu_full_per_language | `global_mmlu_full_zh` | 1B | 3.15 → 4.27 | +1.13 | `other` \| `high_school_world_history` \| `marketing` \| `human_aging` \| `… (+4)` |
| global_mmlu_full_per_language | `global_mmlu_full_hi` | 1B | 2.91 → 4.03 | +1.12 | `marketing` \| `high_school_world_history` |
| global_mmlu_full_per_language | `global_mmlu_full_sw` | 350M | 2.41 → 3.53 | +1.12 | `management` |

![](pretraining/custom_swissai_hf/global_mmlu_full_subjects.png)
## TODO

- [ ] Recommend a *family* of robust subjects (e.g. `medical_genetics`,
      `human_aging`, `international_law`, world-history), not an exact subset —
      only subsets that recur in both train and test seed pools transfer.
- [ ] Treat best-subset picks as candidates; prefer subsets that recur in both
      train and test seed pools (Case 2 subjects are the safest; per-item picks
      the least transferable).
- [ ] Bootstrap CIs on `snr_gain` per (case, task, size) and check cross-seed
      Jaccard / SNR-rank Spearman of the winning subsets.

## Files

- `pretraining/<pool>/summary.csv` — every (case, task, size) by `snr_gain`.
- `…/per_benchmark.csv` (Case 1), `global_mmlu_full.csv` (Case 2),
  `global_mmlu_full_per_language.csv` (Case 3) + their `*_plots/`.
- `per_sample/variance_prefilter/analysis/` — Option-D size distribution,
  cross-size Jaccard/Spearman, `highlights.md`.

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

Every swept cell in one grid (`predictivity` pool). Regenerate with `python analysis/rq08_subset_selection/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq08 in one figure](pretraining/predictivity/highlights.png)

![Gain over the null](pretraining/predictivity/gain_over_null.png)
<!-- END auto:panels -->
