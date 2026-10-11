# Subset selection: can a subset of subtasks give a higher SNR than the full set?

## Research question

> For each benchmark, does a subset of its subtasks (the *languages* of a
> multilingual family, the *subjects* of MMLU or single *items*) give a
> higher SNR than the full set, and does the same subset hold across seeds
> and scales?

<!-- BEGIN auto:highlight (smooth_subtasks.py --pool predictivity) -->
## Highlighted result

- **`bbpb_paws` 175M (per_benchmark).** A subset beats the full set, with SNR **3.59 → 6.55** (**+2.96**) on `bbpb_paws_es|bbpb_paws_zh|bbpb_paws_de|bbpb_paws_ja`.
- **`bbpb_rf_belebele` 1B (per_benchmark).** A subset beats the full set, with SNR **3.17 → 5.99** (**+2.82**) on `bbpb_rf_belebele_eng_Latn|bbpb_rf_belebele_rus_Cyrl|bbpb_rf_belebele_zho_Hant`.
- **`bbpb_rfgm_belebele` 350M (per_benchmark).** A subset beats the full set, with SNR **3.17 → 5.96** (**+2.80**) on `bbpb_rfgm_belebele_spa_Latn|bbpb_rfgm_belebele_rus_Cyrl|bbpb_rfgm_belebele_zho_Hans`.
- **Median gain by case.** global_mmlu_full_subjects 1.03, global_mmlu_full_per_language 0.83, per_benchmark 0.73 (in SNR units). A subset helps only where its gain clears the seed noise reported in rq03.
- **Selection null.** The best prefix is chosen on the numbers it is scored on, so `best ≥ full` always. A cell gets a null only where a win is possible: the best prefix beats the full set, holds two or more subtasks, and there are more than 20 subsets of its size (a single subtask is the maximum of the null's own draws, and with 20 subsets or fewer the chosen one is drawn too often to clear the 95th percentile). Against 100 random subsets of the same size, **83 of the 104** testable cells (every case) beat the null's 95th percentile, 77 of 98 in Case 1. They include `bbpb_rf_belebele` 1B, `bbpb_rfgm_belebele` 350M, `bbpb_rfgm_belebele` 175M, `bbpb_rfgm_belebele` 1.7B, `bbpb_rfgm_belebele` 1B.
<!-- END auto:highlight -->

## Experimental setup

**Setup.**
- **Measured.** We test whether the best subset of a benchmark's languages separates the designs better than a random subset of the same size, comparing the SNR of the two.
- **Models.** We use the seed-1904 runs of every cell, ladder and data build, 90M–1.7B.
- **Tasks.** The tasks are the per-language tasks of each multilingual benchmark that pass the above-chance gate at each size, plus their bBPB versions.
- **How.** We rank the languages by their own SNR and score each prefix of that ranking (the top language, the top two, and so on). The prefix with the highest SNR is the chosen subset, and we compare it with the 95th percentile of 100 random subsets of the same size.

The subtask-level outputs are under `pretraining/predictivity/`. Pool
`predictivity` is seed 1904 with every cell, data build and ladder (swiglu
included), at sizes 90M–1.7B. It reads the data build as one axis, scheme
A/B/C × temperature T. The 36-sweep pools are kept as history in the final
section, "Extensions from other sweeps".

The analysis has three subtask cases and two per-item views, listed below.
Case 1 asks which language subset of a multilingual family (Belebele,
Global-PIQA, MultiBLiMP, Global-MMLU, the per-language BPB family `bpb`, …)
carries the family's signal. Since the 2026-10-07 refresh the benchmark-BPB
(`bbpb_`) twins are scored at every checkpoint of every size. Their cells
therefore use the same noise window as the accuracy tasks, and they fill the
90M and 600M columns.

Cases 2 and 3 were **not** regenerated in this refresh.
`global_mmlu_full.csv` (2 rows) and `global_mmlu_full_per_language.csv`
(4 rows) date from 2026-09-21, and their 6 rows still enter `summary.csv`,
the highlight block and `highlights.png`. Do not read anything about MMLU
subjects from this README until these two cases are rerun on the current pool.

- **Case 1: a language subset of a multilingual family.** `task` is the
  benchmark family (arc, belebele, global_mmlu, xnli, …) and `subtask` is
  one of its per-language tasks (arc_de, arc_es, …). Which language subset,
  ordered by per-language SNR, gives the family the highest combined SNR?
- **Case 2: an MMLU subject subset, averaged over languages.** `task` is
  `global_mmlu_full` and `subtask` is one subject (leaves only). A subject's
  per-(model, ckpt) score is its mean over the global_mmlu languages the
  pool carries.
- **Case 3: an MMLU subject subset per language**, with no averaging across
  languages. `task` is `global_mmlu_full_<lang>` and `subtask` is one
  subject in that language.
- **Per item on the ladder.** `per_item_ladder.py` and `reference_solved.py`
  read the per-item store `predictivity`. It holds every checkpoint of 198
  runs: 178 at the grid seed, the swiglu runs included, and 20 seed
  replicates. 13 of the pool's grid-seed models (lm-1.7B-L8-swiglu and the
  twelve muon cells) are not in it yet, so they are missing from these numbers.
  Per-item SNR on item accuracy is not comparable to subtask-level SNR.
- **Per item (Option D).** `per_sample/variance_prefilter/`, 36-sweep only
  (see the final section, "Extensions from other sweeps").

## Key figure

![Best language subset against the random-subset null, per benchmark and size](pretraining/predictivity/gain_over_null_paper.png)

Population: pool `predictivity` (seed 1904, every cell, data build and ladder), sizes 90M–1.7B, Case 1 only (the language subsets of each benchmark), above-random gate on each per-language task. This gives 254 swept (benchmark, size) cells with an SNR, over 48 benchmarks. The figure shows the 221 cells where the best prefix beats the full set, because only those get a null. They cover 47 benchmarks: 33 accuracy or BPB families and 14 bBPB twins. A cell is the SNR of the best prefix minus the 95th percentile of 100 random subsets of the same size. White means no null value (61 of the 282 cells): either the full set is already the best prefix, or the cell was not swept.

**Key finding.** **A chosen language subset beats the random-subset null in a minority of cells.** It does so in 94 of the 254 swept Case-1 (benchmark, size) cells, which are 94 of the 221 Case-1 cells with a positive raw gain. The highlighted result above counts every case, so it reads 100 of 227 (the six Case 2 and 3 cells all beat their null). In Case 1 the subset beats the null by more than 0.25 SNR in 66 cells. HellaSwag (median +1.12 over six sizes), ARC (+0.97 over four), MultiBLiMP (+0.65) and Global-MMLU RF (+0.61) gain the most. XStoryCloze, LAMBADA, XWinograd and the BPB family never beat the null, XCOPA beats it only once (+0.02), and PAWS-X sits below it (median −0.13).

**Key findings**

- **Without the null, a subset seems to help almost everywhere.** The best prefix beats the full set in 221 of the 254 Case-1 cells that have an SNR (median +0.80 SNR), but only 94 of these 221 cells also beat the null.
- **Accuracy families and bBPB twins beat the null about equally often.** In the accuracy and BPB families 67 of 151 cells beat the null (33 benchmarks), against 27 of 70 for the bBPB twins (14 twins, read on their full noise window).
- **The largest gaps over the null are on INCLUDE RF, ARC and Belebele RF.** The five largest are INCLUDE RF at 600M (+1.65), ARC at 1.7B (+1.53), Belebele RF at 350M (+1.39), ARC at 1B (+1.36) and HellaSwag at 1.7B (+1.33), with PAWS-X bBPB at 600M just behind (+1.33).
- **The share of benchmarks that beat the null does not grow with size.** Over all swept benchmarks it is 0.38 at 90M, 0.36 at 175M, 0.37 at 350M, 0.34 at 600M, 0.43 at 1B and 0.33 at 1.7B (39, 39, 43, 44, 44 and 45 benchmarks, since the task set differs across sizes, rule 13).
- **`highlights.csv` gives higher shares because of its denominator.** Its shares are 0.42, 0.39, 0.46, 0.41, 0.49 and 0.39, because it divides by the benchmarks whose best prefix beats the full set at that size (35–39 at 90M–1B, 38 at 1.7B), not by every swept benchmark.

**Follow-ups**

- Rerun Cases 2 and 3 on the current pool, so the MMLU subject subsets can join the figure.

GitHub: [gain_over_null_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/gain_over_null_paper.png) · [gain_over_null_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/gain_over_null_paper.csv). The subject cases and every swept cell: [Per benchmark and per language](#per-benchmark-and-per-language).

## Methodology

- **Subset SNR.** We compute `signal_to_noise_ratio` over one array of
  last-N-checkpoint scores per training run (seed replicates and external
  models count as separate runs). The noise is pooled across runs, as in
  upstream's `compute_snr_small_scale`. This SNR keeps upstream's pooled SD of
  the window scores on purpose. It does not use the detrended checkpoint
  noise that the other analyses use since 2026-10-08. A combined subset first
  averages the per-(model, step) scores over its subtasks.
- **Sweep.** For each (task, size), we rank the subtasks by their standalone
  SNR and score the cumulative prefixes 1..N. `best_n` / `best_subset` is the
  prefix with the highest SNR, and `snr_gain = best − full`.
- **Gate.** We apply the above-random gate ([gate and curves](../rq00_gate_and_curves/README.md)).
  In Case 1, a language task at chance at a size is not swept. In Cases 2
  and 3, a language whose `global_mmlu_full_<lang>` aggregate is at chance is
  skipped. The four MMLU category roll-ups are left out of the subject lists.
- **Selection null.** The prefix is chosen on the same numbers it is scored
  on, so `best ≥ full` by construction. For every swept cell where the best
  prefix beats the full set (221 of the 254 Case-1 cells), we draw 100
  random subsets of the best size and keep their 95th percentile,
  `null_snr_p95`. `gain_over_null = best − null_p95` is the part of the gain
  that does not come from the selection. Where the full set is the best
  prefix, there is no null.

Hand-written numbers in this README are from the ladder report **2026-10-10
04:31** and the outputs of the 2026-10-10 refresh, except the Case 2 and
Case 3 rows, which date from 2026-09-21.

<!-- BEGIN auto:results (smooth_subtasks.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq08_subset_selection/smooth_subtasks.py --pool predictivity`.

**Top subset gains.** The twelve (case, task, size) cells with the largest `snr_gain = best − full`:

| case | task | size | full → best SNR | +gain | null p95 | best subset |
|---|---|---|---|---|---|---|
| per_benchmark | `bbpb_paws` | 175M | 3.59 → 6.55 | +2.96 |  | `bbpb_paws_es` \| `bbpb_paws_zh` \| `bbpb_paws_de` \| `bbpb_paws_ja` |
| per_benchmark | `bbpb_rf_belebele` | 1B | 3.17 → 5.99 | +2.82 | 4.35 | `bbpb_rf_belebele_eng_Latn` \| `bbpb_rf_belebele_rus_Cyrl` \| `bbpb_rf_belebele_zho_Hant` |
| per_benchmark | `bbpb_rfgm_belebele` | 350M | 3.17 → 5.96 | +2.80 | 5.16 | `bbpb_rfgm_belebele_spa_Latn` \| `bbpb_rfgm_belebele_rus_Cyrl` \| `bbpb_rfgm_belebele_zho_Hans` |
| per_benchmark | `bbpb_rfgm_belebele` | 175M | 3.14 → 5.86 | +2.73 | 4.36 | `bbpb_rfgm_belebele_spa_Latn` \| `bbpb_rfgm_belebele_rus_Cyrl` \| `bbpb_rfgm_belebele_zho_Hant` |
| per_benchmark | `bbpb_paws` | 350M | 3.67 → 6.28 | +2.60 |  | `bbpb_paws_zh` \| `bbpb_paws_es` \| `bbpb_paws_ja` |
| per_benchmark | `bbpb_rfgm_belebele` | 1.7B | 3.48 → 6.02 | +2.54 | 5.52 | `bbpb_rfgm_belebele_eng_Latn` \| `bbpb_rfgm_belebele_zho_Hant` \| `bbpb_rfgm_belebele_spa_Latn` \| `bbpb_rfgm_belebele_rus_Cyrl` |
| per_benchmark | `arc` | 350M | 2.86 → 5.36 | +2.51 |  | `arc_easy` |
| per_benchmark | `bbpb_rfgm_belebele` | 1B | 3.35 → 5.83 | +2.49 | 5.61 | `bbpb_rfgm_belebele_eng_Latn` \| `bbpb_rfgm_belebele_spa_Latn` \| `bbpb_rfgm_belebele_rus_Cyrl` \| `bbpb_rfgm_belebele_zho_Hant` |
| per_benchmark | `bbpb_rfgm_belebele` | 600M | 3.15 → 5.55 | +2.40 | 4.75 | `bbpb_rfgm_belebele_spa_Latn` \| `bbpb_rfgm_belebele_zho_Hans` \| `bbpb_rfgm_belebele_zho_Hant` |
| per_benchmark | `bbpb_rfgm_belebele` | 90M | 3.16 → 5.51 | +2.34 | 5.76 | `bbpb_rfgm_belebele_spa_Latn` \| `bbpb_rfgm_belebele_eng_Latn` \| `bbpb_rfgm_belebele_zho_Hans` \| `bbpb_rfgm_belebele_rus_Cyrl` |
| per_benchmark | `bbpb_rf_belebele` | 90M | 3.23 → 5.56 | +2.34 | 4.49 | `bbpb_rf_belebele_spa_Latn` \| `bbpb_rf_belebele_zho_Hans` \| `bbpb_rf_belebele_zho_Hant` |
| per_benchmark | `hellaswag` | 600M | 3.20 → 5.51 | +2.31 | 4.25 | `hellaswag_es` \| `hellaswag_ru` |

![](pretraining/predictivity/global_mmlu_full_subjects.png)
<!-- END auto:results -->

[global_mmlu_full_subjects.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/global_mmlu_full_subjects.png) ·
[summary.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/summary.csv) ·
[per_benchmark.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/per_benchmark.csv) ·
[global_mmlu_full.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/global_mmlu_full.csv) ·
[global_mmlu_full_per_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/global_mmlu_full_per_language.csv)

**Key findings** (the block above: pool `predictivity`, every case, ranked by the raw gain)

- **The largest raw gains are not the largest gains over the null.** Of the twelve largest raw gains, ARC at 350M (+2.51, `arc_easy` alone, third row) gains nothing over the null. ARC at 1.7B (+2.21, the last row) has the largest gain over it among the twelve (+1.53).
- **Half of the twelve rows are bBPB twins.** The other six are ARC and HellaSwag, both scored on acc_norm since 2026-10-08.
- **Part of the highlight block is not current.** The MMLU subject figure (`global_mmlu_full_subjects.png`) and the "median gain by case" line still include the Case 2 and Case 3 rows of 2026-09-21.

**Follow-ups**

- Sort the auto table by `gain_over_null` instead of the raw gain, so that its top rows are the ones that survive the selection null.

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

Every swept cell in one grid (`predictivity` pool). Regenerate with `python analysis/rq08_subset_selection/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate". Each figure's table sits next to it under the same name.

![rq08 in one figure](pretraining/predictivity/highlights.png)

![Gain over the null](pretraining/predictivity/gain_over_null.png)
<!-- END auto:panels -->

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/highlights.csv) ·
GitHub: [gain_over_null.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/gain_over_null.png) · [gain_over_null.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/gain_over_null.csv)

**Key findings** (the two figures above: pool `predictivity`, every case, the above-random gate, and the 227 cells with a null value: 221 from Case 1 and 6 from the Case 2 and 3 tables of 2026-09-21)

- **The Case 1 numbers match the [opening figure](#key-figure).** The Case 2 and 3 columns of `highlights.png` (share 1.0 at 350M and 600M) come from the stale tables.
- **The benchmark ranking in `highlights.png` includes two stale Case 3 rows.** They are `global_mmlu_full_ms` (+0.47) and `global_mmlu_full_lt` (+0.43). On the current rows alone, the top six are HellaSwag, ARC, MultiBLiMP, Global-MMLU RF, INCLUDE v2 (OG) (+0.43) and Belebele RF (+0.42).

**Follow-ups**

- Rerun Cases 2 and 3 on the current pool so both panels read one snapshot.

<!-- BEGIN auto:reference-solved (reference_solved.py --pool predictivity) -->
## Items the reference solves

DA-size, final checkpoints, multi-axis pairs of `predictivity` (grid seed), gate `predictivity` at the proxy and the reference, 560 tasks with per-item outputs. An item is solved when at least 0.5 of the 1.7B runs of the selecting half answer it right. The subset's proxy mean is scored against the 1.7B final of the full task on the other half's pairs, both ways round (held out, rule 11), beside the full set on the same pairs. The in-sample column selects with every 1.7B run, so it has seen the truth. SNR is the relative dispersion of the design means over the relative k-fold noise, read at the proxy alone. The solved set's SNR uses the items that at least half of all the 1.7B runs solve, while the DA it is correlated with is the held-out one. Regenerate with `python analysis/rq08_subset_selection/reference_solved.py --pool predictivity`.

| proxy | tasks | solved share | DA all (held out) | DA solved (held out) | Δ | Wilcoxon p | DA solved (in sample) | ρ(SNR, DA) all | ρ(SNR, DA) solved |
|---|---|---|---|---|---|---|---|---|---|
| 90M | 296 | 0.41 | 0.52 | 0.56 | +0.039 | 0.000 | 0.56 | 0.31 | 0.39 |
| 175M | 332 | 0.41 | 0.53 | 0.56 | +0.031 | 0.000 | 0.57 | 0.31 | 0.46 |
| 350M | 362 | 0.40 | 0.52 | 0.55 | +0.028 | 0.000 | 0.56 | 0.36 | 0.44 |
| 600M | 406 | 0.39 | 0.54 | 0.56 | +0.018 | 0.000 | 0.57 | 0.44 | 0.49 |
| 1B | 452 | 0.38 | 0.56 | 0.56 | +0.006 | 0.253 | 0.57 | 0.37 | 0.39 |

![Items the reference solves](pretraining/predictivity/reference_solved_da_size_multi_axes.png)
<!-- END auto:reference-solved -->

**Key findings** (the block above: pool `predictivity`, ladder report 2026-10-10 04:31, DA-size at the final checkpoint, multi-axis pairs, 238 tasks at 90M to 375 at 1B with paired held-out values, finals read from the per-item store `predictivity` without the 13 pool models the block lists)

- **Held out, the solved items agree better with the reference at 90M–600M.**
  They gain +0.033, +0.030, +0.021 and +0.015 over the full set on the same
  pairs (Wilcoxon p < 0.001 at each size), and nothing at 1B (+0.004,
  p = 0.33).
- **Choosing the items with the reference inflates DA very little.** At
  every proxy 39–42 % of the items are solved, and the in-sample DA
  (0.56–0.58) is at most 0.01 above the held-out DA (0.55–0.57).
- **SNR tracks the held-out DA a little more closely on the solved items.**
  The correlation is ρ = 0.39–0.51, against 0.37–0.45 on all items (equal at
  1B, 0.39). So the higher SNR on the solved items is not only removed noise.
- **The in-sample counterpart is the above-chance items analysis.** It
  keeps, for every task, only the items its 1.7B runs answer above chance
  ([above-chance items](../rq12_above_chance_items/README.md)).

**Follow-ups**

- The same table per benchmark, to see whether the gain comes from a few
  families with many easy items.
- The held-out DA-ckpt of the solved items, now that the store holds every
  checkpoint, to see whether the gain also appears earlier in training.

GitHub: [reference_solved_da_size_multi_axes.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/reference_solved_da_size_multi_axes.png) · [reference_solved_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/reference_solved_da_size_multi_axes.csv) · [reference_solved_summary_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/reference_solved_summary_da_size_multi_axes.csv) · [reference_solved_per_task_da_size_multi_axes.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/reference_solved_per_task_da_size_multi_axes.csv)

<!-- BEGIN auto:per-item-ladder (per_item_ladder.py --pool predictivity) -->
## Per item on the ladder

Per-item subset selection over the `predictivity` design variants, from the per-item store (`build_per_item_store.sbatch`). Regenerate with `python analysis/rq08_subset_selection/per_item_ladder.py --pool predictivity`. Grid-seed (1904) design variants, 623 tasks over 32 runs at most; per-item SNR on the rule-4 window (80-100 % of the run, k/20 points). An item is dead when every run has the same mean outcome on it. Gain = best-prefix SNR minus the 95th percentile of 100 random subsets of the same size. Held-out DA: items chosen on half of the families (stratified over L, arch, scheme, T), scored on the other half's multi-axis pairs (>= 3, proxy final vs the 1.7B final of the full task), both halves averaged. Random = 20 draws of the same size from the items alive on the selecting half (the subset's own pool). Grey = at chance at that size (rule 1). The task set differs across sizes (counts in the cells, rule 13).

| task | size | items | dead | full -> best SNR | best n | null p95 | held-out DA full / subset / random |
|---|---|---|---|---|---|---|---|
| `piqa` | 600M | 1838 | 0.56 | 4.69 -> 8.19 | 3 | 3.74 | 0.62 / 0.20 / 0.35 |
| `toksuite_math` | 1B | 189 | 0.17 | 4.31 -> 7.82 | 2 | 3.60 | 0.67 / 0.22 / 0.35 |
| `rf_mmmlu_zh_cn` | 350M | 14040 | 0.52 | 3.68 -> 8.14 | 9 | 4.10 | 0.70 / 0.38 / 0.44 |
| `winogrande` | 600M | 1267 | 0.07 | 4.55 -> 7.54 | 2 | 3.61 | 0.68 / 0.19 / 0.30 |
| `rf_acp_bench_mcq_just` | 175M | 130 | 0.12 | 3.72 -> 8.00 | 2 | 4.10 | 0.53 / 0.14 / 0.23 |
| `rf_mmmlu_de_de` | 350M | 14040 | 0.56 | 3.08 -> 7.98 | 12 | 4.10 | 0.59 / 0.24 / 0.35 |
| `rf_m_mmlu_ru` | 600M | 13007 | 0.52 | 4.06 -> 7.84 | 5 | 3.98 | 0.67 / 0.23 / 0.34 |
| `rf_m_mmlu_de` | 350M | 13258 | 0.55 | 3.59 -> 8.02 | 12 | 4.16 | 0.64 / 0.43 / 0.52 |
| `rf_mmmlu_ja_jp` | 350M | 14041 | 0.52 | 3.16 -> 8.01 | 17 | 4.15 | 0.50 / 0.33 / 0.33 |
| `rf_m_mmlu_zh` | 350M | 13165 | 0.49 | 3.81 -> 8.09 | 8 | 4.26 | 0.64 / 0.42 / 0.41 |

| size | tasks | DA full set | DA held-out subset | DA random subset |
|---|---|---|---|---|
| 90M | 296 | 0.52 | 0.32 | 0.36 |
| 175M | 332 | 0.53 | 0.33 | 0.37 |
| 350M | 362 | 0.52 | 0.37 | 0.39 |
| 600M | 406 | 0.54 | 0.33 | 0.37 |
| 1B | 452 | 0.56 | 0.31 | 0.35 |

![per-item ladder](pretraining/predictivity/per_item_ladder.png)
<!-- END auto:per-item-ladder -->

**Key findings** (the block above: pool `predictivity`, seed 1904, sizes 90M–1.7B, per-item store `predictivity` (every checkpoint), above-chance tasks only: 316 at 90M to 512 at 1.7B in `per_item_summary.csv`, held-out DA-size on multi-axis pairs, finals)

- **An item subset raises the SNR almost everywhere.** The best prefix beats the null in 97–99 % of the above-chance tasks at every size, with a median gain over the null of +0.76 (350M) to +1.11 (1.7B) SNR.
- **The SNR gain comes from a handful of items.** The winning prefixes are tiny (median 4–8 items out of a median 900). About half of each benchmark's items are dead, meaning that every run has the same mean outcome on them (median share 0.52–0.61 per size).
- **These subsets do not predict the reference, and do worse than random items.** Held out (90M–1B, 238 to 375 tasks with multi-axis pairs), the subset's DA is 0.32–0.37, against 0.53–0.57 for the full set on the same pairs and 0.35–0.39 for a random subset of the same size. The 316–512 above-chance tasks apply only to the SNR bullets.
- **Choosing items by SNR has the opposite effect of keeping the items the reference solves.** That selection keeps about 40 % of the items and gains DA held out ([items the reference solves](#items-the-reference-solves)).

**Follow-ups**

- The same held-out DA with a floor on the subset size (for example 10 % of the items), to see whether the loss comes from the small size of the subset or from choosing by SNR.

GitHub: [per_item_ladder.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/per_item_ladder.png) · [per_item_ladder.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq08_subset_selection/pretraining/predictivity/per_item_ladder.csv) · `per_item_summary.csv` (cluster-only, not in git)

## TODO

- [ ] Recommend a *family* of robust subjects (e.g. `medical_genetics`,
      `human_aging`, `international_law`, world-history), not an exact subset.
      Only subsets that recur in both the train and the test seed pools transfer.
- [ ] Treat best-subset picks as candidates, and prefer subsets that recur in
      both the train and the test seed pools (Case 2 subjects are the safest,
      per-item picks the least transferable).
- [ ] Bootstrap CIs on `snr_gain` per (case, task, size) and check cross-seed
      Jaccard / SNR-rank Spearman of the winning subsets.

## Files

- `per_item_store/<store>/` (git-ignored, cluster-only): one row per (model,
  checkpoint, task, item), built by `build_per_item_store.sbatch`
  (`build_per_item_store.py`, resumable, every checkpoint). Read by
  `per_item_ladder.py`, `reference_solved.py`, the bBPB DA comparison and the
  above-chance items analysis.
- `bench_bpb.csv`: the store reduced to one bits-per-byte value of the gold
  answer per (model, step, task) by `build_per_item_store.py --bench-bpb`
  (the driver's first pass, rewritten only when its content changes).
  `utils.with_bbpb_twins` reads it to add every loader's `bbpb_` twins.
- `pretraining/<pool>/reference_solved_per_task_da_size_multi_axes.csv`: per
  task and proxy, the pairs, the DAs and SNRs of the full and the solved
  items, and `gated`. `reference_solved_summary_da_size_multi_axes.csv` gives
  the same per proxy, and the figure `reference_solved_da_size_multi_axes.png`
  has its CSV beside it.
- `pretraining/<pool>/summary.csv`: every (case, task, size), sorted by `snr_gain`.
- `…/per_benchmark.csv` (Case 1), `global_mmlu_full.csv` (Case 2),
  `global_mmlu_full_per_language.csv` (Case 3) + their `*_plots/`.
- `per_sample/variance_prefilter/analysis/`: the Option-D size distribution,
  cross-size Jaccard/Spearman, `highlights.md`.

## Extensions from other sweeps

Everything below comes from the **36-model sweep** (2026-04…06, 4 sizes × 3
data mixtures × 3 seeds, pools `seeds_28_1797_1904` and `custom_swissai_hf`,
reference **1B**, 12 languages, the 86-task old list) or from the **external
tier** (`all/external`: the public and reference models, 270M–70B,
cross-model dispersion with no mixture axis). The sweep's SNR is computed
over three mixtures with its own checkpoint window, on a task list without
the twins. Its subset gains therefore replicate the finding and are never
rows of the ladder's table.

The per-item (Option D) pass under `per_sample/` is also 36-sweep only. It
needs the per-sample files that live on the cluster, and the ladder's
per-item store (`build_per_item_store.sbatch`, `per_item_ladder.py`) replaces
it. That store is cluster-only too, so `per_item_snr.csv` and
`per_item_summary.csv` exist only where it was built. Off the cluster the
driver's step writes nothing.

## External model-set tier (`all/external`, 36-sweep)

On the `external` tier the SNR is the **cross-model dispersion** over the
270M…70B external ladder (no mixture axis), and only the three subtask cases
run (no per-item Option-D pass). The outputs are in `all/external/`.
Regenerate them with
`python analysis/rq08_subset_selection/smooth_subtasks.py --pool external`.

**A subset still beats the full set on another set of models.** The external
ladder's full-set SNRs are low, because its mix of model families washes out
a benchmark's combined signal. Even so, a small subtask subset recovers that
signal, with gains as large as on the custom pool:

| case | task | size | full → best SNR | +gain | best subset |
|---|---|---|---|---|---|
| case3_global_mmlu_full_per_language | `global_mmlu_full_ar` | 1B | 0.18 → 1.95 | +1.77 | `high_school_chemistry` |
| case1_per_benchmark | `truthfulqa` | 1B | 0.52 → 1.96 | +1.44 | `truthfulqa_hi_mc2` \| `truthfulqa_vi_mc2` |
| case1_per_benchmark | `paws` | 3B | 0.37 → 1.81 | +1.44 | `paws_eu` |
| case1_per_benchmark | `truthfulqa` | 3B | 0.66 → 1.92 | +1.26 | `truthfulqa_es_mc1` |
| case1_per_benchmark | `global_piqa_completions` | 1B | 0.56 → 1.73 | +1.16 | `global_piqa_completions_eng_latn` |
| case3_global_mmlu_full_per_language | `global_mmlu_full_vi` | 1B | 0.87 → 1.93 | +1.06 | `high_school_chemistry` \| `elementary_mathematics` |

![MMLU subject subsets vs full set (external)](all/external/global_mmlu_full_subjects.png)

The recurring choices carry over (MMLU subject subsets, per-language
TruthfulQA `mc` splits, `paws_eu`), but the **exact** winning subjects differ
from the custom pool (`high_school_chemistry` recurs here, world-history
there). Subset selection is therefore a robust *lever*, but the exact subset
does not transfer.

The per-benchmark curves of the sweep (full set against best subset across
the external ladder), for the families with the largest gains:

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
