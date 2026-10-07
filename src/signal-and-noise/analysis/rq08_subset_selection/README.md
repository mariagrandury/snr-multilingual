# RQ8 — Can a subset of subtasks give higher SNR than the full set?

## Research question

> Per benchmark, does a subset of subtasks — a subset of *languages* in a
> multilingual family, or a subset of *subjects* in MMLU, or a subset of
> individual *items* — give higher SNR than the full set, and does the same
> subset hold across seeds and scales?

<!-- BEGIN auto:highlight (smooth_subtasks.py --pool predictivity) -->
## Highlighted result

- **`bbpb_rf_belebele` 1B (per_benchmark)** — a subset beats the full set: SNR **3.05 → 5.46** (**+2.41**) with `bbpb_rf_belebele_eng_Latn|bbpb_rf_belebele_rus_Cyrl|bbpb_rf_belebele_zho_Hant`.
- **`bbpb_paws` 175M (per_benchmark)** — a subset beats the full set: SNR **3.32 → 5.63** (**+2.31**) with `bbpb_paws_es|bbpb_paws_de|bbpb_paws_zh`.
- **`arc` 1B (per_benchmark)** — a subset beats the full set: SNR **3.02 → 5.31** (**+2.29**) with `arc_easy|arc_challenge`.
- **Median gain by case** — global_mmlu_full_subjects 1.03; global_mmlu_full_per_language 0.83; per_benchmark 0.68 (SNR units; a subset only helps where the gain clears the seed noise reported in rq03).
- **Selection null** — the best prefix is chosen on the numbers it is scored on, so `best ≥ full` always; against 100 random subsets of the same size, **78 of 169** swept cells beat the null's 95th percentile: `bbpb_rf_belebele` 1B, `bbpb_paws` 175M, `bbpb_rfgm_belebele` 1B, `bbpb_rfgm_belebele` 175M, `bbpb_rfgm_belebele` 1.7B.
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

## Key figure

![Best language subset against the random-subset null, per benchmark and size](pretraining/predictivity/gain_over_null_paper.png)

Population: pool `predictivity` (seed 1904), sizes 90M–1.7B, the language-subset case (one subset of each benchmark's per-language tasks), the above-random gate at each size; cell = SNR of the best prefix minus the 95th percentile of 100 random subsets of the same size; white = no swept cell.

**Key finding.** A chosen language subset beats the random-subset null in 54 of 118 (benchmark, size) cells, by more than 0.25 SNR in 36, mostly on the reformulated knowledge benchmarks (Global-MMLU RF median +0.74, Belebele RF +0.41) and MultiBLiMP, while XStoryCloze, LAMBADA and PAWS-X never beat it.

GitHub: [gain_over_null_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/gain_over_null_paper.png) · [gain_over_null_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/gain_over_null_paper.csv). The subject cases and every swept cell: [Per benchmark and per language](#per-benchmark-and-per-language).

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

Hand-written numbers in this README are from the ladder-report snapshot
**2026-09-30 23:54**.

<!-- BEGIN auto:results (smooth_subtasks.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq08_subset_selection/smooth_subtasks.py --pool predictivity`.

**Top subset gains** — every (case, task, size) ranked by `snr_gain = best − full`:

| case | task | size | full → best SNR | +gain | null p95 | best subset |
|---|---|---|---|---|---|---|
| per_benchmark | `bbpb_rf_belebele` | 1B | 3.05 → 5.46 | +2.41 | 4.31 | `bbpb_rf_belebele_eng_Latn` \| `bbpb_rf_belebele_rus_Cyrl` \| `bbpb_rf_belebele_zho_Hant` |
| per_benchmark | `bbpb_paws` | 175M | 3.32 → 5.63 | +2.31 | 4.81 | `bbpb_paws_es` \| `bbpb_paws_de` \| `bbpb_paws_zh` |
| per_benchmark | `arc` | 1B | 3.02 → 5.31 | +2.29 | 5.31 | `arc_easy` \| `arc_challenge` |
| per_benchmark | `bbpb_rfgm_belebele` | 1B | 3.11 → 5.35 | +2.24 | 5.32 | `bbpb_rfgm_belebele_rus_Cyrl` \| `bbpb_rfgm_belebele_spa_Latn` \| `bbpb_rfgm_belebele_eng_Latn` \| `bbpb_rfgm_belebele_zho_Hant` \| `… (+4)` |
| per_benchmark | `bbpb_rfgm_belebele` | 175M | 2.97 → 5.21 | +2.23 | 4.24 | `bbpb_rfgm_belebele_spa_Latn` \| `bbpb_rfgm_belebele_rus_Cyrl` \| `bbpb_rfgm_belebele_zho_Hant` |
| per_benchmark | `arc` | 1.7B | 3.18 → 5.37 | +2.19 | 5.37 | `arc_easy` |
| per_benchmark | `bbpb_rfgm_belebele` | 1.7B | 3.30 → 5.46 | +2.15 | 5.15 | `bbpb_rfgm_belebele_zho_Hant` \| `bbpb_rfgm_belebele_spa_Latn` \| `bbpb_rfgm_belebele_eng_Latn` \| `bbpb_rfgm_belebele_rus_Cyrl` |
| per_benchmark | `hellaswag` | 350M | 2.97 → 5.04 | +2.06 | 4.36 | `hellaswag_it` \| `hellaswag_es` \| `hellaswag_ru` |
| per_benchmark | `hellaswag` | 175M | 2.96 → 4.96 | +2.00 | 4.30 | `hellaswag_es` \| `hellaswag_ru` \| `hellaswag_it` |
| per_benchmark | `rfgm_belebele` | 90M | 2.93 → 4.92 | +2.00 | 3.74 | `rfgm_belebele_spa_Latn` \| `rfgm_belebele_zho_Hans` |
| per_benchmark | `bbpb_rf_belebele` | 175M | 3.17 → 5.13 | +1.96 | 4.87 | `bbpb_rf_belebele_spa_Latn` \| `bbpb_rf_belebele_zho_Hans` \| `bbpb_rf_belebele_zho_Hant` \| `bbpb_rf_belebele_rus_Cyrl` \| `… (+5)` |
| per_benchmark | `hellaswag` | 90M | 3.07 → 4.96 | +1.89 | 4.23 | `hellaswag_ru` \| `hellaswag_es` |

![](pretraining/predictivity/global_mmlu_full_subjects.png)
<!-- END auto:results -->

[global_mmlu_full_subjects.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/global_mmlu_full_subjects.png) ·
[summary.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/summary.csv) ·
[per_benchmark.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/per_benchmark.csv) ·
[global_mmlu_full.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/global_mmlu_full.csv) ·
[global_mmlu_full_per_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/global_mmlu_full_per_language.csv)

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

Every swept cell in one grid (`predictivity` pool). Regenerate with `python analysis/rq08_subset_selection/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq08 in one figure](pretraining/predictivity/highlights.png)

![Gain over the null](pretraining/predictivity/gain_over_null.png)
<!-- END auto:panels -->

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/highlights.csv) ·
GitHub: [gain_over_null.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/gain_over_null.png) · [gain_over_null.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/gain_over_null.csv)

<!-- BEGIN auto:reference-solved (reference_solved.py --pool predictivity) -->
## Items the reference solves

DA-size, final checkpoints, multi-axis pairs of `predictivity` (grid seed), gate `predictivity` at the proxy and the reference, 453 tasks with per-item outputs. An item is solved when at least 0.5 of the 1.7B runs of the selecting half answer it right; the subset's proxy mean is scored against the 1.7B final of the full task on the other half's pairs, both ways round (held out, rule 11), beside the full set on the same pairs; the in-sample column selects with every 1.7B run and has seen the truth. SNR = relative dispersion of the design means over the relative k-fold noise, read at the proxy alone; the solved set's SNR is on the items at least half of all the 1.7B runs solve, while the DA it is correlated with is the held-out one. Regenerate with `python analysis/rq08_subset_selection/reference_solved.py --pool predictivity --store predictivity_schemes`. The store `predictivity_schemes` lacks 3 of the pool's models (lm-350M-L1-fweb-deep-seed1904, lm-600M-L1-fweb-deep-seed1904, lm-90M-L1-fweb-b84-deep-seed1904); they are left out until the store is rebuilt for the pool.

| proxy | tasks | solved share | DA all (held out) | DA solved (held out) | Δ | Wilcoxon p | DA solved (in sample) | ρ(SNR, DA) all | ρ(SNR, DA) solved |
|---|---|---|---|---|---|---|---|---|---|
| 90M | 234 | 0.40 | 0.54 | 0.57 | +0.033 | 0.000 | 0.57 | 0.26 | 0.35 |
| 175M | 269 | 0.40 | 0.54 | 0.57 | +0.026 | 0.000 | 0.57 | 0.31 | 0.39 |
| 350M | 297 | 0.40 | 0.53 | 0.56 | +0.028 | 0.000 | 0.57 | 0.32 | 0.38 |
| 600M | 328 | 0.40 | 0.55 | 0.57 | +0.022 | 0.000 | 0.58 | 0.41 | 0.45 |
| 1B | 358 | 0.39 | 0.57 | 0.57 | +0.002 | 0.684 | 0.58 | 0.41 | 0.45 |

![Items the reference solves](pretraining/predictivity/reference_solved_da_size_multi_axes.png)
<!-- END auto:reference-solved -->

**Key findings** (the block above: pool `predictivity`, ladder report cached 2026-10-06 04:26, finals read from the `predictivity_schemes` store until the `predictivity` store is built)

- Held out, the solved items read the reference better at 90M–600M: +0.021
  to +0.035 over the full set on the same pairs (Wilcoxon p ≤ 0.001), and
  not at 1B (+0.003, p = 0.78).
- About 40 % of the items are solved at every proxy, and the in-sample DA
  (0.57–0.58) is barely above the held-out one: the selection does not lean
  on the reference by much.
- SNR tracks the held-out DA a little more closely on the solved items
  (ρ 0.36–0.45 against 0.27–0.40), so a higher SNR on them is not only noise
  removed.
- The in-sample counterpart, every task reduced to the items its 1.7B runs
  answer above chance, is the [above-chance items](../rq12_above_chance_items/README.md)
  analysis.

**Follow-ups**

- The same table per benchmark, to see whether the gain comes from a few
  families with many easy items.
- Once the store holds every checkpoint, the held-out DA-ckpt of the solved
  items, to see whether the gain also comes earlier in training.

GitHub: [reference_solved_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/reference_solved_da_size_multi_axes.png) · [reference_solved_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/reference_solved_da_size_multi_axes.csv) · [reference_solved_summary_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/reference_solved_summary_da_size_multi_axes.csv) · [reference_solved_per_task_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/reference_solved_per_task_da_size_multi_axes.csv)

## TODO

- [ ] Recommend a *family* of robust subjects (e.g. `medical_genetics`,
      `human_aging`, `international_law`, world-history), not an exact subset —
      only subsets that recur in both train and test seed pools transfer.
- [ ] Treat best-subset picks as candidates; prefer subsets that recur in both
      train and test seed pools (Case 2 subjects are the safest; per-item picks
      the least transferable).
- [ ] Bootstrap CIs on `snr_gain` per (case, task, size) and check cross-seed
      Jaccard / SNR-rank Spearman of the winning subsets.

## Extensions from other sweeps

Everything below comes from the **36-model sweep** (2026-04…06, 4 sizes × 3
data mixtures × 3 seeds, pools `seeds_28_1797_1904`, `custom_swissai_hf`;
reference **1B**, 12 languages, the 86-task old list) or from the **external
tier** (`all/external`: the public and reference models, 270M–70B,
cross-model dispersion with no mixture axis). Its SNR is computed over three
mixtures with the sweep's checkpoint window, on a task list without the
twins, so its subset gains are a replication of the lever, never rows of the
ladder's table. The per-item (Option D) pass under `per_sample/` is
36-sweep-only as well: it needs the per-sample files that live on the
cluster, and the ladder's per-item store (`build_per_item_store.sbatch`,
`per_item_ladder.py`) is its successor. That store is cluster-only too, so
`per_item_snr.csv` and `per_item_summary.csv` exist only where it was built;
off the cluster the driver's step writes nothing.

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

[global_mmlu_full_subjects.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/all/external/global_mmlu_full_subjects.png) ·
[truthfulqa.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/all/external/per_benchmark_plots/truthfulqa.png)

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

[global_mmlu_full_subjects.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/custom_swissai_hf/global_mmlu_full_subjects.png)

## Files

- `per_item_store/<store>/` (git-ignored, cluster-only) — one row per (model,
  checkpoint, task, item), built by `build_per_item_store.sbatch`
  (`build_per_item_store.py`, resumable; finals only for now). Read by
  `per_item_ladder.py`, `reference_solved.py`, the bBPB DA comparison and the
  above-chance items analysis.
- `bench_bpb.csv` — the store reduced to one bits-per-byte value of the gold
  answer per (model, step, task) by `build_per_item_store.py --bench-bpb`
  (the driver's first pass, rewritten only when its content changes);
  `utils.with_bbpb_twins` reads it to add every loader's `bbpb_` twins.
- `pretraining/<pool>/reference_solved_per_task_da_size_multi_axes.csv` — per
  task and proxy: pairs, the DAs and SNRs of the full and the solved items,
  `gated`; `reference_solved_summary_da_size_multi_axes.csv` per proxy; the
  figure `reference_solved_da_size_multi_axes.png` with its CSV.
- `pretraining/<pool>/summary.csv` — every (case, task, size) by `snr_gain`.
- `…/per_benchmark.csv` (Case 1), `global_mmlu_full.csv` (Case 2),
  `global_mmlu_full_per_language.csv` (Case 3) + their `*_plots/`.
- `per_sample/variance_prefilter/analysis/` — Option-D size distribution,
  cross-size Jaccard/Spearman, `highlights.md`.

<!-- BEGIN auto:per-item-ladder (per_item_ladder.py --pool predictivity) -->
## Per item on the ladder

Per-item subset selection over the `predictivity` design variants, from the per-item store (`build_per_item_store.sbatch`). Regenerate with `python analysis/rq08_subset_selection/per_item_ladder.py --pool predictivity`. Grid-seed (1904) design variants, 520 tasks over 26 runs at most; per-item SNR on the rule-4 window (80-100 % of the run, k/20 points); dead = the same mean outcome in every run. Gain = best-prefix SNR minus the 95th percentile of 100 random subsets of the same size. Held-out DA: items chosen on half of the families (stratified over L, arch, scheme, T), scored on the other half's multi-axis pairs (>= 3, proxy final vs the 1.7B final of the full task), both halves averaged; random = 20 draws. Grey = at chance at that size (rule 1); the task set differs across sizes (counts in the cells, rule 13).

| task | size | items | dead | full -> best SNR | best n | null p95 | held-out DA full / subset / random |
|---|---|---|---|---|---|---|---|
| `mathqa` | 1B | 2985 | 0.54 | 3.26 -> 7.21 | 2 | 3.49 | 0.64 / 0.01 / 0.10 |
| `xnli_en` | 1.7B | 2490 | 0.19 | 2.93 -> 7.21 | 2 | 3.60 | 1.00 / 0.08 / 0.25 |
| `rf_global_mmlu_full_en` | 1B | 14042 | 0.47 | 5.39 -> 7.21 | 6 | 3.72 | 0.74 / 0.53 / 0.47 |
| `rf_commonsense_qa` | 1.7B | 1221 | 0.52 | 3.06 -> 7.21 | 2 | 3.79 | 1.00 / 0.38 / 0.39 |
| `rf_mmlu` | 1.7B | 14042 | 0.45 | 4.83 -> 7.21 | 2 | 3.81 | 1.00 / 0.48 / 0.48 |
| `xwinograd_en` | 600M | 2325 | 0.25 | 3.80 -> 7.21 | 2 | 3.89 | 0.65 / 0.42 / 0.43 |
| `xstorycloze_en` | 1.7B | 1511 | 0.64 | 3.65 -> 7.21 | 2 | 3.90 | 1.00 / 0.23 / 0.17 |
| `arc_easy` | 1B | 2376 | 0.39 | 5.13 -> 7.21 | 2 | 3.91 | 0.59 / 0.51 / 0.49 |
| `rfgm_belebele_eng_Latn` | 175M | 898 | 0.51 | 4.03 -> 7.21 | 4 | 3.92 | 0.48 / 0.47 / 0.38 |
| `xwinograd_en` | 1B | 2325 | 0.28 | 4.35 -> 7.21 | 2 | 3.94 | 0.69 / 0.39 / 0.37 |

| size | tasks | DA full set | DA held-out subset | DA random subset |
|---|---|---|---|---|
| 90M | 234 | 0.54 | 0.37 | 0.30 |
| 175M | 269 | 0.54 | 0.36 | 0.30 |
| 350M | 297 | 0.53 | 0.38 | 0.32 |
| 600M | 328 | 0.55 | 0.36 | 0.29 |
| 1B | 358 | 0.57 | 0.33 | 0.29 |

![per-item ladder](pretraining/predictivity/per_item_ladder.png)
<!-- END auto:per-item-ladder -->
