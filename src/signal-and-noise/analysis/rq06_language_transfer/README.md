# Language transfer: does it generalize to languages we have not measured?

## Research question

> If a language's BPB is measured on only one or two small rungs, can the
> exponent pooled over the other languages predict its BPB at the reference
> size, also for languages the model never trained on? Does a design decision
> a proxy reads on its trained languages carry over to untrained ones? Which
> languages must a developer evaluate to recover the multilingual decision? This is the paper's language-transfer
> analysis
> ([`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex)).
> The folder also holds the detailed per-language BPB curves of every cell.

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_seeds) -->
## Highlighted result

- **Trained languages (106 (L, language) cases).** One 175M rung plus the pooled exponent predicts the reference within 4.4 % (median). With every proxy rung, the median error is 3.9 % for the transferred exponent, 1.5 % for the language's own fit and 7.6 % for the largest proxy's BPB taken as is.
- **Never-trained languages (494 (L, language) cases).** One 175M rung plus the pooled exponent predicts the reference within 5.0 % (median). With every proxy rung, the median error is 3.1 % for the transferred exponent, 2.7 % for the language's own fit and 6.6 % for the largest proxy's BPB taken as is.
- **Pooled exponent α by L.** L1 0.090, L2 0.096, L8 0.096, L15 0.094, L30 0.106, L50 0.117.
<!-- END auto:highlight -->

## Experimental setup

**Setup.**
- **Measured.** We test whether the language-list decision, read on a language's BPB, carries over to languages the lists do not train.
- **Models and pairs.** We read the scheme A versus B decision at L ∈ {8, 15, 30} on single-axis pairs, on the seed-1904 runs and sizes 90M–1.7B.
- **Tasks.** The tasks are the per-language BPB of the 100 evaluation languages, grouped by whether both lists, one or neither trains them, and the untrained ones by whether a list trains their script. BPB has no chance level, so nothing is gated.
- **How.** DA-size compares the final checkpoint of each proxy (90M–1B) with that of the 1.7B run, and DA-ckpt the 1.7B run's checkpoints with its final one. Both are averaged over the three L.
- **Untrained relatives.** Per language family, we also measure how often tasks in untrained languages clear chance, against the English-only cell on the same tasks (figures: app_rq06_family_transfer, app_rq06_family_lift).

**Exponent transfer.** Per-language BPB of the plan grid (pool
`predictivity_seeds` cut to deep, data A, seed 1904) at the final checkpoint,
with proxies 175M–1B (90M is left out of the fit) and reference 1.7B, at all
six L (1, 2, 8, 15, 30, 50). Each of the 100 validation languages is a series,
which gives 600 (L, language) cases: 106 trained (the language is in the
cell's FineWeb-2 list at that L, `launch_trainings.cell_fineweb_subsets`) and
494 never trained.

**Decision transfer.** The mono-axis interventions of the design-decisions
analysis (pool `predictivity_seeds`, reference 1.7B), read on the per-language
BPB of the 100 languages. Depth (deep vs shallow) is read at all six L. Scheme
A vs B is read at L1 (A vs DCLMP), L2 (A vs ZH) and L8–L30 (the list decision),
scheme A vs C at L1 (A vs FWEB) and L2 (A vs ES), temperature (T=1 vs T=3) at
L15–L50, and activation (xIELU vs SwiGLU) at L8–L30. The optimizer has no 1.7B
evaluation yet, so it has no row.

**Language panel.** Pool `predictivity`, seed 1904, 15–17 design variants of
every data build (105–136 multi-axis pairs per benchmark), gate `predictivity`,
no filter, reference 1.7B, and the eight L8 languages. The bBPB twins exist at
every checkpoint of every size since the 2026-10-07 refresh.

## Key figure

![The list decision by language group, by proxy size and by the reference's checkpoints](pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis_paper.png)

Population: pool `predictivity_seeds`, the list decision (scheme A vs B on the L8, L15 and L30 lists, mono-axis pairs, reference 1.7B). Left: DA-size of the proxies 90M–1B at their final checkpoint. Right: DA-ckpt of the reference's own checkpoints at 10–90 % of its run. No gate or task filter (per-language BPB has no chance level), mean over the three L. The items are the 100 evaluation languages, grouped by what the two lists do with each. Per L there are 5–26 languages trained by both lists, 6–18 trained by one, 53–66 untrained with a trained script and 13–23 with an untrained script. Dotted line: 0.75.

**Key finding.** **The list decision does not transfer to untrained languages at a proxy's scale.** Languages whose script a list trains are read at 0.63 at 90M and at 0.72–0.73 at 175M–600M, and reach 0.81 only at 1B. Languages in unseen scripts are read at 0.42–0.73 and never reach 0.75. By contrast, the languages that the lists train are read at 0.77–1.00 (both lists) and 1.00 (one list).

GitHub: [transfer_da_all_lines_mono_axis_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis_paper.png) · [transfer_da_all_lines_mono_axis_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis_paper.csv). Every intervention by language count: [Per benchmark and per language](#per-benchmark-and-per-language).

<!-- BEGIN auto:cross-task-transfer (cross_task_transfer.py --pool predictivity) -->
## Cross-task transfer of the ranking

Pool `predictivity` (seed 1904, every data build), mono-axis pairs, ≥ 3 pairs per (task pair, size), each side gated (proxy task at the proxy size, target task at 1.7B), accuracy-scored parent tasks. Each cell is the DA-size of the proxy task's final ranking at a proxy size (y) against the target task's final ranking at 1.7B (x), averaged over 90M–1B. `trained` reads each family only on the languages it trains (rule 2). `all_languages` adds the scores of the scheme-A deep seed-1904 cells on every other language (and those of the scheme-B L8 deep cells at 350M–1B), rq06's one exception to rule 2. Regenerate with `python analysis/rq06_language_transfer/cross_task_transfer.py --pool predictivity`.

| population | filter | tasks | cells | diagonal | same benchmark, other language | other benchmark, same language | other benchmark and language | off-diagonal share ≥ 0.75 |
|---|---|---|---|---|---|---|---|---|
| trained | `above_chance` | 555 | 308025 | 0.51 | 0.48 | 0.48 | 0.47 | 0.04 |
| trained | `above_66_either` | 111 | 12321 | 0.63 | 0.60 | 0.59 | 0.53 | 0.15 |
| trained | `L8` | 175 | 30625 | 0.51 | 0.50 | 0.48 | 0.48 | 0.00 |
| trained | `hellaswag` | 26 | 676 | 0.68 | 0.61 | – | – | 0.23 |
| all_languages | `above_chance` | 873 | 762129 | 0.56 | 0.51 | 0.50 | 0.47 | 0.01 |
| all_languages | `above_66_either` | 111 | 12321 | 0.65 | 0.58 | 0.58 | 0.52 | 0.02 |
| all_languages | `L8` | 301 | 90601 | 0.55 | 0.52 | 0.49 | 0.45 | 0.00 |
| all_languages | `hellaswag` | 31 | 961 | 0.69 | 0.56 | – | – | 0.03 |

![HellaSwag, language against language](pretraining/predictivity/cross_task_da_size_by_language_hellaswag_all_languages_mono_axis.png)

![By benchmark, the rq2 tasks](pretraining/predictivity/cross_task_da_size_by_benchmark_above_66_either_mono_axis.png)

![By language, the rq2 tasks](pretraining/predictivity/cross_task_da_size_by_language_above_66_either_mono_axis.png)

The other maps (every filter, per task, per benchmark and per language, both populations) sit beside them as `cross_task_da_size_by_<task|benchmark|language>_<filter>[_all_languages]_mono_axis.png`.
<!-- END auto:cross-task-transfer -->

The earlier version of this reading is the decision-accuracy analysis's
[cross-task maps](../rq02_decision_accuracy/README.md#cross-task-predictability)
(`rq02_decision_accuracy/cross_task.py`, 2026-09-18), which draw the smallest
safe proxy size. This script reuses their pair-sign kernel but draws the DA
value, with the proxy on y and the target on x. Snapshot: the ladder report of
2026-10-10 04:31 (the `above_66_either` task list is rq02's table of the same
day).

Key findings (`cross_task_da_size_summary_mono_axis.csv`, the HellaSwag map's CSV):

- **Within one benchmark, the languages that a mixture trains together predict
  each other.** On HellaSwag (all languages), German, Spanish, French and Italian,
  the four European languages of the L8 list, predict each other's 1.7B ranking at
  0.81 on average (figure: app_rq06_cross_task_hellaswag), almost as well as each predicts itself (0.82).
- **English is a poor proxy for the other languages.** English HellaSwag ranks
  itself at 0.78 but the other languages at only 0.43 on average, and they rank
  it at 0.50. No other language predicts Basque, which no list trains (its
  column averages 0.46, at most 0.61).
- **Across benchmarks the transfer is weak.** On every task above chance (trained
  languages), the own-task diagonal averages 0.51 and the three off-diagonal kinds
  0.47–0.49. Only 4 % of the off-diagonal cells reach 0.75. On the rq2 tasks
  (`above_66_either`), the diagonal is 0.63, the same benchmark in another
  language 0.60, another benchmark in the same language 0.58, and a different
  benchmark in a different language 0.52.
- **A cell reads only the pairs that both tasks have.** Two cells can therefore
  average different decisions. In the all-languages maps, a language that no list
  trains is read on the scheme-A cells alone (pairs that move L). The trained
  HellaSwag map is noisier for the L30 and L50 languages, which few families
  train (few pairs, quantized values).

Follow-ups:

- The same HellaSwag map per proxy size, to see whether the L8 block already
  forms at 90M or only at 1B.
- The language map ordered by language family instead of resource rank, to test
  the block structure against the family reading below.

<!-- BEGIN auto:family-transfer (family_transfer.py --pool predictivity) -->
## Above chance on untrained languages, by language family

Pool `predictivity`, the gate's run-level verdicts (rule 1) at the final checkpoint, deep seed-1904 cells at L1, L2 and L8, accuracy-scored parent tasks of `auto` benchmarks (no probe candidates, `utils.auto_tasks`) in languages the cell does not train. Same family means that the language shares the top-level family or the subfamily (`configs/languages.json` `family`) of a trained non-English language. Untrained-language results exist for L1 A every size, L2 A every size, L8 A every size, L8 B every size, L2 ZH every size, L2 ES every size, and for none of L1 DCLMP, L1 FWEB. Left out: 0 languages without a family in languages.json. Regenerate with `python analysis/rq06_language_transfer/family_transfer.py --pool predictivity`.

![Above chance on untrained languages by family](pretraining/predictivity/above_chance_untrained_by_family.png)

![Lift over English only by subfamily](pretraining/predictivity/above_chance_untrained_lift_by_subfamily.png)

Numbers (generated):

- **L2 A, subfamily.** Share above chance, same family 90M 0.36, 175M 0.39, 350M 0.40, 600M 0.38, 1B 0.46, 1.7B 0.49. Other family 90M 0.36, 175M 0.38, 350M 0.39, 600M 0.41, 1B 0.45, 1.7B 0.49. Lift over L1 A on the same tasks: same family -0.01, +0.05, +0.03, -0.01, +0.08, +0.07, other family -0.03, -0.02, -0.01, -0.01, -0.01, -0.04.
- **L2 A, top-level family.** Share above chance, same family 90M 0.39, 175M 0.41, 350M 0.42, 600M 0.43, 1B 0.49, 1.7B 0.54. Other family 90M 0.31, 175M 0.33, 350M 0.35, 600M 0.37, 1B 0.40, 1.7B 0.42. Lift over L1 A on the same tasks: same family -0.01, +0.01, +0.01, -0.01, +0.00, -0.03, other family -0.05, -0.04, -0.03, -0.01, -0.00, -0.03.
- **L2 ES, subfamily.** Share above chance, same family 90M 0.43, 175M 0.46, 350M 0.48, 600M 0.60, 1B 0.63, 1.7B 0.74. Other family 90M 0.35, 175M 0.35, 350M 0.36, 600M 0.40, 1B 0.41, 1.7B 0.45. Lift over L1 A on the same tasks: same family +0.01, -0.01, +0.03, +0.12, +0.06, +0.07, other family -0.03, -0.03, -0.02, +0.01, -0.01, -0.04.
- **L2 ES, top-level family.** Share above chance, same family 90M 0.39, 175M 0.39, 350M 0.41, 600M 0.46, 1B 0.47, 1.7B 0.52. Other family 90M 0.32, 175M 0.32, 350M 0.33, 600M 0.38, 1B 0.40, 1.7B 0.42. Lift over L1 A on the same tasks: same family -0.01, -0.01, +0.01, +0.03, +0.01, -0.03, other family -0.04, -0.05, -0.04, +0.00, -0.01, -0.03.
- **L2 ZH, subfamily.** Share above chance, same family 90M 0.00, 175M 0.00, 350M 0.00, 600M 1.00, 1B 0.00, 1.7B 0.00. Other family 90M 0.34, 175M 0.35, 350M 0.37, 600M 0.40, 1B 0.44, 1.7B 0.47. Lift over L1 A on the same tasks: same family +0.00, -1.00, +0.00, +1.00, +0.00, +0.00, other family -0.04, -0.04, -0.03, -0.01, -0.01, -0.05.
- **L2 ZH, top-level family.** Share above chance, same family 90M 0.00, 175M 0.00, 350M 0.00, 600M 0.17, 1B 0.17, 1.7B 0.00. Other family 90M 0.34, 175M 0.36, 350M 0.37, 600M 0.40, 1B 0.44, 1.7B 0.47. Lift over L1 A on the same tasks: same family +0.00, -0.17, +0.00, +0.00, +0.00, -0.33, other family -0.04, -0.04, -0.03, -0.01, -0.01, -0.05.
- **L8 A, subfamily.** Share above chance, same family 90M 0.39, 175M 0.43, 350M 0.41, 600M 0.48, 1B 0.54, 1.7B 0.60. Other family 90M 0.33, 175M 0.38, 350M 0.38, 600M 0.35, 1B 0.41, 1.7B 0.43. Lift over L1 A on the same tasks: same family -0.04, +0.01, -0.02, +0.04, +0.04, +0.04, other family -0.03, +0.01, +0.01, -0.03, +0.02, -0.02.
- **L8 A, top-level family.** Share above chance, same family 90M 0.37, 175M 0.41, 350M 0.39, 600M 0.41, 1B 0.48, 1.7B 0.53. Other family 90M 0.32, 175M 0.38, 350M 0.39, 600M 0.36, 1B 0.43, 1.7B 0.43. Lift over L1 A on the same tasks: same family -0.02, +0.02, -0.00, +0.00, +0.03, -0.00, other family -0.05, -0.01, +0.01, -0.03, +0.03, -0.01.
- **L8 B, subfamily.** Share above chance, same family 90M 0.43, 175M 0.37, 350M 0.40, 600M 0.47, 1B 0.56, 1.7B 0.59. Other family 90M 0.34, 175M 0.35, 350M 0.37, 600M 0.42, 1B 0.45, 1.7B 0.50. Lift over L1 A on the same tasks: same family +0.00, -0.03, -0.02, +0.03, +0.09, +0.05, other family -0.03, -0.04, -0.02, +0.01, +0.00, -0.01.
- **L8 B, top-level family.** Share above chance, same family 90M 0.39, 175M 0.37, 350M 0.39, 600M 0.47, 1B 0.51, 1.7B 0.56. Other family 90M 0.32, 175M 0.35, 350M 0.35, 600M 0.37, 1B 0.42, 1.7B 0.47. Lift over L1 A on the same tasks: same family +0.00, -0.04, -0.02, +0.03, +0.02, -0.01, other family -0.05, -0.04, -0.03, -0.02, +0.02, +0.03.

GitHub: [above_chance_untrained_by_family.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity/above_chance_untrained_by_family.png) · [above_chance_untrained_by_family.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity/above_chance_untrained_by_family.csv) · [above_chance_untrained_lift_by_subfamily.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity/above_chance_untrained_lift_by_subfamily.png) · [above_chance_untrained_lift_by_subfamily.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity/above_chance_untrained_lift_by_subfamily.csv) · [above_chance_untrained_by_family_per_task.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity/above_chance_untrained_by_family_per_task.csv)
<!-- END auto:family-transfer -->

Key findings (2026-10-10 snapshot, `above_chance_untrained_by_family.csv`, `above_chance_untrained_lift_by_subfamily.csv`). Every evaluated language has a family in `languages.json` since 2026-10-08, and the figures draw a lift only on ≥ 20 verdicts. They read the `auto` benchmarks only (`utils.auto_tasks`, RULES.md, "The probe candidates are not in the populations"): 1,027 tasks of 34 benchmarks and 22,074 verdicts. On 2026-10-08 there were 941 tasks of 27 benchmarks and 15,898 verdicts. The all-languages backlog has since brought in L2 ZH, L2 ES and the other sizes of L8 B, and it is still landing. L8 B has 373 of its 824 tasks at 1.7B, L2 ZH has 67–327 per size, and L2 ES has 131 at 90M–350M only. Their 1.7B points (and ZH's 1B point) therefore rest on a subset, and the dips there are not yet a finding:

- **Most subfamilies that a cell trains are lifted, and the others are not.** Five of the seven (cell, trained subfamily) lifts over English only are positive, +0.03 to +0.11 (the +0.11 is L8 B's Iranian, on 37 verdicts). L8 B's Germanic is at 0.00 and L8 A's Germanic at −0.02 (450 verdicts). Weighted by verdicts, the mean lift is +0.02. The 83 (cell, other subfamily) lifts centre on zero (median 0.00, 37 % positive, −0.01 weighted, range −0.12 to +0.12).
- **Romance is lifted only where it is trained, but untrained Bantu is lifted as much.** L8 A (Spanish, French, Italian) puts 0.58 of the Catalan, Galician, Portuguese and Romanian tasks above chance, against 0.53 for English only (+0.05). L2 A, L2 ZH and L8 B, whose lists have no Romance language, sit at −0.03 to −0.02 on Romance. L8 A also lifts Bantu, which it does not train, by as much (+0.06), so the lift of one subfamily alone is within the scatter. L2 ES, the direct test of Spanish alone, has 3 untrained Romance tasks per size (French, Italian, Portuguese) at three sizes. That gives 9 verdicts, under the 20 that the figure needs.
- **Slavic is lifted by Russian, not by Chinese.** Over the eleven untrained Slavic languages, the lift is +0.04 for L2 A (Russian alone), +0.04 for L8 A and +0.03 for L8 B. L2 ZH, which swaps Russian for Chinese, is at −0.01 on the same subfamily (131 verdicts).
- **The effect is small but present from 600M.** L8 A's lift on its own subfamilies is −0.04 at 90M, +0.01 at 175M, and +0.04 at 600M, 1B and 1.7B. There is no 350M point, because the L1 cell's evaluation is incomplete there. On the other subfamilies the lift is −0.03 to +0.02. The raw same-family share sits above the other-family share in most panels. However, English only shows most of that gap too, so the gap mostly reflects which languages and benchmarks fall in each group.
- **ZH cannot test its own subfamily.** `languages.json` gives Chinese the family "Sino-Tibetan" with no subfamily, so no untrained language shares it. At the top level its relatives are Tibetan and Burmese (2 tasks at 90M and 175M, none above chance for either cell). DCLMP and FWEB are still evaluated on their own languages only.

Follow-ups:

- Re-read L2 ZH, L2 ES and L8 B at 1.7B once the all-languages backlog finishes.
  L2 ES at 600M–1.7B would put the Spanish-alone Romance test over the
  20-verdict bar.

## Methodology

- **Own exponent.** log BPB = a − α log N per language over the proxy rungs.
- **Leave-one-language-out.** For each language, the exponent is the median
  of the other languages' α at that L. The intercept comes from the held-out
  language's k smallest rungs (k = 1 … all). The prediction at the reference
  is compared with the language's own fit (k ≥ 3) and with the largest proxy's
  BPB taken as is. Errors are relative, with medians per (trained, k).
- **Decision transfer.** The interventions of the design-decisions analysis
  (its `intervention_da`: depth, scheme A vs B, scheme A vs C, temperature and
  activation), read on the BPB of every evaluation language. This is the one
  population that rule 2 lets this folder use. Per (intervention, L, proxy size,
  checkpoint), it is the share of a language group's languages on which the
  proxy ranks the two levels as the reference does at its final checkpoint.
  `analyze.py` writes it (`transfer_da_all_by_group_mono_axis.csv`) and
  `panels.py` draws it.
- **BPB curves.** One panel per cell of the pool, with per-language BPB against
  the fraction of the run: trained languages in blue, unseen ones in grey, and
  macro BPB dashed (the progress report's `plot_bpb` on the analysis' cells).

Hand-written numbers in this README come from the ladder report of
**2026-10-10 04:31** and the outputs regenerated from it on 2026-10-10 (the
2026-10-08 refresh brought acc_norm on the cloze-format originals and the
detrended checkpoint noise). The bBPB twins are scored at every checkpoint of
every size and seed in this report.

<!-- BEGIN auto:results (analyze.py --pool predictivity_seeds) -->
## Results

Numbers from the `predictivity_seeds` pool. Regenerate with `python analysis/rq06_language_transfer/analyze.py --pool predictivity_seeds`.

**Median |relative error| of the reference's BPB** (k = rungs measured for the held-out language):

| languages | k | transferred α | own fit | largest proxy | n |
|---|---|---|---|---|---|
| never trained | 1 | 0.050 |  | 0.248 | 494 |
| never trained | 2 | 0.043 |  | 0.188 | 494 |
| never trained | 3 | 0.037 | 0.048 | 0.121 | 494 |
| never trained | 4 | 0.031 | 0.027 | 0.066 | 494 |
| trained | 1 | 0.044 |  | 0.326 | 106 |
| trained | 2 | 0.053 |  | 0.257 | 106 |
| trained | 3 | 0.047 | 0.039 | 0.161 | 106 |
| trained | 4 | 0.039 | 0.015 | 0.076 | 106 |

![Transfer](pretraining/predictivity_seeds/rq5_transfer.png)

**Decision transfer** (DA-size of rq05's interventions on per-language BPB: the share of a group's languages on which the proxy's final ranking of the two levels matches the reference's, mean over interventions and L. The group says what the two levels' lists do with the language):

| language group | proxy | DA-size | cells |
|---|---|---|---|
| trained by both levels | 90M | 0.86 | 22 |
| trained by both levels | 175M | 0.85 | 22 |
| trained by both levels | 350M | 0.76 | 22 |
| trained by both levels | 600M | 0.59 | 22 |
| trained by both levels | 1B | 0.82 | 22 |
| trained by one level | 90M | 1.00 | 5 |
| trained by one level | 175M | 1.00 | 5 |
| trained by one level | 350M | 1.00 | 5 |
| trained by one level | 600M | 1.00 | 5 |
| trained by one level | 1B | 1.00 | 5 |
| script trained | 90M | 0.66 | 22 |
| script trained | 175M | 0.71 | 22 |
| script trained | 350M | 0.70 | 22 |
| script trained | 600M | 0.62 | 22 |
| script trained | 1B | 0.70 | 22 |
| script not trained | 90M | 0.52 | 22 |
| script not trained | 175M | 0.58 | 22 |
| script not trained | 350M | 0.63 | 22 |
| script not trained | 600M | 0.63 | 22 |
| script not trained | 1B | 0.68 | 22 |

![BPB curves](pretraining/predictivity_seeds/bpb_curves.png)
<!-- END auto:results -->

GitHub: [rq5_transfer.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/rq5_transfer.png) · [rq5_transfer.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/rq5_transfer.csv) ·
[rq5_transfer_summary.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/rq5_transfer_summary.csv) ·
GitHub: [bpb_curves.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/bpb_curves.png) · [bpb_curves.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/bpb_curves.csv) ·
[facts.json](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/facts.json)

**Key findings** (`rq5_transfer`: deep, data A, seed 1904, proxies 175M–1B,
reference 1.7B, six L, 106 trained and 494 never-trained (L, language) cases)

- **One small rung plus the pooled exponent beats the largest proxy read as
  is.** With the 175M rung alone, the median error on the 1.7B BPB is 4.4 %
  (trained) and 5.0 % (never trained). Taking the 1B proxy's own BPB as the
  reference's misses by 7.6 % and 6.6 %.
- **The pooled exponent serves untrained languages better than trained ones.**
  With all four proxies, on never-trained languages the transferred exponent
  errs by 3.1 % against 2.7 % for the language's own fit (it wins on 44 % of
  cases). On trained languages it errs by 3.9 % against 1.5 % (it wins on 13 %).
  Trained languages scale more steeply (median own α 0.12 against 0.09 for
  never-trained ones). The pooled α (0.090–0.117 by L) is set mostly by the
  untrained languages, which are 70–99 of the 100 at L1–L30. At L50 the split
  is even (50/50) and the pooled α is highest (0.117).
- **The error grows with L.** From one 175M rung, never-trained languages are
  predicted within 3.6 % at L1 and 7.6 % at L50, and trained ones within 2.2 %
  (English alone, L1) to 5.1 % (L50).

**Follow-ups**

- For a trained held-out language, pool the exponent over the trained
  languages only, to test whether the gap of 3.9 % against 1.5 % comes from
  the exponent mismatch above.

<!-- BEGIN auto:panels (panels.py --pool predictivity_seeds) -->
## Per benchmark and per language

The summary above, per language (`predictivity_seeds` pool). Regenerate with `python analysis/rq06_language_transfer/panels.py --pool predictivity_seeds`. In every grid, white means "no value" and grey "filtered out by the gate". Each figure's table sits next to it under the same name.

![rq06 in one figure](pretraining/predictivity_seeds/highlights.png)

![Transfer error per language and L](pretraining/predictivity_seeds/transfer_error_by_L.png)

![The list decision (scheme A vs B at L8–L30) on untrained languages, by proxy size and checkpoint](pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis.png)

![The decisions on untrained languages, by language count](pretraining/predictivity_seeds/transfer_da_all_by_L_mono_axis.png)
<!-- END auto:panels -->

The two decision figures read the mono-axis interventions of the
design-decisions analysis (pool `predictivity_seeds`, reference 1.7B). The
lines figure shows the list decision (scheme A vs B at L8–L30) alone, and the
by-L figure every intervention at every L it has. Scheme B is DCLMP at L1, ZH
at L2 and the B lists at L8–L30. Scheme C is FWEB at L1 and ES at L2.

**Key findings** (`transfer_da_all_lines_mono_axis`, `transfer_da_all_by_L_mono_axis`: the per-item
agreement of the design decisions on the per-language BPB of every evaluation
language. Per (intervention, L), the languages are grouped by what the two
levels' lists do with them: both train the language, only one does, neither
does but one trains its script, or neither trains even the script. The last
two groups are the transfer test.)

- **The list decision (A vs B) transfers only partly.** On languages whose
  script a list trains, DA-size is 0.63 at 90M and 0.72–0.73 from 175M to
  600M, and clears 0.75 only at 1B (0.81). On unseen scripts it is 0.42–0.73
  and never clears 0.75.
- **The reference's own early checkpoints do better.** DA-ckpt on
  script-trained languages rises from 0.65 at 10 % of the run to 0.90 at
  90 %. On unseen scripts it goes from 0.50 at 10 % to 0.85 at 90 %, but not
  steadily (0.77 at 50 %, back to 0.63 at 80 %).
- **Per language count, script-trained languages read most data decisions but
  never depth or activation.** At the final checkpoint (DA-size, mean over
  proxy sizes), they read scheme A vs B at L1 (0.84), L2 (0.96), L8 (0.82) and
  L15 (0.77) but not at L30 (0.58). They also read scheme A vs C at L1 (0.98)
  and L2 (0.90), and temperature at L15–L50 (0.78–0.79). Depth is not read on
  untrained languages at any L (0.41–0.72), and neither is activation
  (0.40–0.72 at L8–L30).
- **Unseen scripts read only the L1 English-source swap.** Scheme A vs C at
  L1 (A vs FWEB, where both levels train English alone) reads 0.99 on unseen
  scripts. Every other (intervention, L) reads them at 0.40–0.71.
- **The 1.00 on trained languages is the inclusion decision, not transfer.**
  It comes from the languages that only one list trains (1.00 wherever the
  group exists), where the model that saw the language wins.
- **Script is a coarse proxy for relatedness.** The Latin script covers Welsh
  and Vietnamese alike. At L1 only English is trained, so "script trained"
  means every other Latin-script language (50 at L1).

**Follow-ups**

- Group the untrained languages by the trained tokens of their language
  family or genus (from the data manifest), or by tokenizer overlap with the
  trained languages, and plot DA against that exposure instead of three bins.
- Scheme A vs C and temperature have only two and three L's (A vs C: FWEB at
  L1, ES at L2), so fold them into a table. The "mean over proxy sizes" of
  `transfer_da_all_by_L_mono_axis` averages a proxy that flips with one that
  agrees into 0.5. For the paper version, draw one line per proxy size, or
  only the largest proxy below the reference.
- The same caveats as in the design-decisions analysis apply. Items are
  correlated, so bootstrap over L. The reference is 1.7B for every intervention.

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/highlights.csv) ·
GitHub: [transfer_error_by_L.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_error_by_L.png) · [transfer_error_by_L.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_error_by_L.csv) ·
GitHub: [transfer_da_all_lines_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis.png) · [transfer_da_all_lines_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis.csv) ·
GitHub: [transfer_da_all_by_L_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_by_L_mono_axis.png) · [transfer_da_all_by_L_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_by_L_mono_axis.csv)

**The minimal language panel · DA-size against the 1.7B MACRO ranking
(mean score over the L8 panel languages above chance at 1.7B) · no filter ·
multi-axis pairs from `predictivity` at seed 1904 · gate
`predictivity`.** Which languages must a developer evaluate to recover the
multilingual decision? We compare three options: one language at a time,
English alone, and the proxy's own macro over the languages it can read at
its size.

Population: 15–17 design variants (105–136 pairs per benchmark) and the eight
L8 languages (en, ru, zh, de, ja, es, fr, it). The pooled row covers 18
benchmarks plus 19 bBPB twins at 600M and 1B (37 tasks), and one task fewer at
90M–350M, which lack `global_piqa_parallel_cloze`. Its populations therefore
differ by one task across sizes.

<!-- BEGIN auto:language-panel (language_panel.py --pool predictivity) -->
## The minimal language panel

Per benchmark family, decision accuracy of a proxy size against the 1.7B MACRO ranking of the design variants (mean over the L8 panel languages above chance at 1.7B), when the proxy reads one language, English, or its own macro over the languages readable at that size. Grid-seed pairs of every data build (multi-axis), ≥ 3 pairs, gate `predictivity`. A macro above every single language says that the languages' errors are independent and that the panel is worth evaluating. A single language as good as the macro says that this language suffices. Regenerate with `python analysis/rq06_language_transfer/language_panel.py --pool predictivity`.

| benchmark | languages readable at 1.7B | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| arc | 7 | macro 0.59 / en 0.59 / best lang 0.59 (en) | macro 0.48 / en 0.48 / best lang 0.48 (en) | macro 0.64 / en 0.59 / best lang 0.63 (fr) | macro 0.72 / en 0.59 / best lang 0.72 (zh) | macro 0.69 / en 0.62 / best lang 0.75 (ru) |
| bbpb_arc | 7 | macro 0.51 / en 0.61 / best lang 0.61 (en) | macro 0.68 / en 0.63 / best lang 0.71 (ru) | macro 0.67 / en 0.55 / best lang 0.72 (fr) | macro 0.63 / en 0.60 / best lang 0.69 (de) | macro 0.53 / en 0.41 / best lang 0.59 (zh) |
| bbpb_belebele | 8 | macro 0.59 / en 0.47 / best lang 0.64 (it) | macro 0.56 / en 0.54 / best lang 0.66 (ru) | macro 0.59 / en 0.53 / best lang 0.63 (ja) | macro 0.65 / en 0.57 / best lang 0.66 (ru) | macro 0.59 / en 0.59 / best lang 0.62 (de) |
| bbpb_cultural_bench_easy | 5 | macro 0.61 / en 0.62 / best lang 0.66 (zh) | macro 0.54 / en 0.57 / best lang 0.57 (en) | macro 0.52 / en 0.51 / best lang 0.56 (zh) | macro 0.42 / en 0.41 / best lang 0.49 (zh) | macro 0.53 / en 0.52 / best lang 0.56 (ru) |
| bbpb_global_mmlu_full | 8 | macro 0.61 / en 0.59 / best lang 0.64 (es) | macro 0.64 / en 0.63 / best lang 0.71 (ru) | macro 0.65 / en 0.62 / best lang 0.72 (es) | macro 0.63 / en 0.61 / best lang 0.66 (fr) | macro 0.72 / en 0.68 / best lang 0.74 (it) |
| bbpb_global_piqa_parallel_cloze | 8 | macro 0.71 / en 0.49 / best lang 0.73 (de) | macro 0.66 / en 0.49 / best lang 0.69 (de) | macro 0.67 / en 0.62 / best lang 0.70 (de) | macro 0.63 / en 0.54 / best lang 0.66 (fr) | macro 0.65 / en 0.45 / best lang 0.67 (es) |
| bbpb_hellaswag | 6 | macro 0.83 / en 0.76 / best lang 0.84 (es) | macro 0.81 / en 0.75 / best lang 0.82 (es) | macro 0.80 / en 0.64 / best lang 0.81 (es) | macro 0.76 / en 0.61 / best lang 0.78 (es) | macro 0.81 / en 0.73 / best lang 0.81 (es) |
| bbpb_include_base_44 | 7 | macro 0.62 / best lang 0.67 (de) | macro 0.58 / best lang 0.59 (ja) | macro 0.62 / best lang 0.64 (es) | macro 0.61 / best lang 0.65 (fr) | macro 0.63 / best lang 0.68 (zh) |
| bbpb_include_v2_en | 7 | macro 0.72 / best lang 0.76 (es) | macro 0.41 / best lang 0.54 (ja) | macro 0.58 / best lang 0.68 (de) | macro 0.53 / best lang 0.63 (ja) | macro 0.61 / best lang 0.67 (ja) |
| bbpb_include_v2_og | 7 | macro 0.63 / best lang 0.65 (ru) | macro 0.57 / best lang 0.61 (de) | macro 0.65 / best lang 0.68 (ru) | macro 0.68 / best lang 0.69 (de) | macro 0.61 / best lang 0.66 (ru) |
| bbpb_multiblimp | 6 | macro 0.74 / en 0.70 / best lang 0.75 (fr) | macro 0.60 / en 0.29 / best lang 0.64 (fr) | macro 0.71 / en 0.56 / best lang 0.73 (fr) | macro 0.76 / en 0.71 / best lang 0.78 (it) | macro 0.61 / en 0.57 / best lang 0.66 (fr) |
| bbpb_paws | 6 | macro 0.72 / en 0.56 / best lang 0.74 (zh) | macro 0.74 / en 0.45 / best lang 0.77 (zh) | macro 0.73 / en 0.51 / best lang 0.77 (de) | macro 0.73 / en 0.57 / best lang 0.71 (zh) | macro 0.68 / en 0.67 / best lang 0.73 (zh) |
| bbpb_rf_belebele | 8 | macro 0.63 / en 0.59 / best lang 0.71 (it) | macro 0.61 / en 0.56 / best lang 0.66 (ja) | macro 0.67 / en 0.61 / best lang 0.68 (it) | macro 0.65 / en 0.63 / best lang 0.66 (de) | macro 0.67 / en 0.57 / best lang 0.65 (it) |
| bbpb_rf_cultural_bench_easy | 5 | macro 0.59 / en 0.62 / best lang 0.62 (en) | macro 0.55 / en 0.59 / best lang 0.59 (en) | macro 0.64 / en 0.63 / best lang 0.68 (ja) | macro 0.55 / en 0.38 / best lang 0.58 (ru) | macro 0.71 / en 0.67 / best lang 0.72 (es) |
| bbpb_rf_global_mmlu_full | 8 | macro 0.76 / en 0.56 / best lang 0.79 (de) | macro 0.66 / en 0.56 / best lang 0.70 (de) | macro 0.65 / en 0.60 / best lang 0.66 (ja) | macro 0.75 / en 0.55 / best lang 0.73 (ja) | macro 0.66 / en 0.57 / best lang 0.67 (ja) |
| bbpb_rf_include_base_44 | 7 | macro 0.56 / best lang 0.62 (zh) | macro 0.74 / best lang 0.74 (ja) | macro 0.64 / best lang 0.68 (de) | macro 0.76 / best lang 0.76 (fr) | macro 0.64 / best lang 0.68 (zh) |
| bbpb_rf_m_mmlu | 6 | macro 0.58 / best lang 0.65 (zh) | macro 0.53 / best lang 0.57 (zh) | macro 0.66 / best lang 0.76 (zh) | macro 0.68 / best lang 0.71 (zh) | macro 0.63 / best lang 0.75 (ru) |
| bbpb_rf_mmmlu | 6 | macro 0.62 / best lang 0.68 (zh) | macro 0.61 / best lang 0.64 (zh) | macro 0.69 / best lang 0.77 (zh) | macro 0.72 / best lang 0.77 (zh) | macro 0.69 / best lang 0.68 (fr) |
| bbpb_rfgm_belebele | 8 | macro 0.75 / en 0.76 / best lang 0.77 (es) | macro 0.71 / en 0.62 / best lang 0.74 (fr) | macro 0.77 / en 0.72 / best lang 0.81 (it) | macro 0.69 / en 0.54 / best lang 0.72 (ru) | macro 0.71 / en 0.47 / best lang 0.73 (ru) |
| bbpb_rfgm_include_base_44 | 7 | macro 0.81 / best lang 0.80 (zh) | macro 0.78 / best lang 0.82 (es) | macro 0.84 / best lang 0.86 (ja) | macro 0.81 / best lang 0.80 (es) | macro 0.83 / best lang 0.83 (de) |
| bbpb_xnli | 6 | macro 0.66 / en 0.52 / best lang 0.71 (ru) | macro 0.70 / en 0.46 / best lang 0.77 (zh) | macro 0.73 / en 0.49 / best lang 0.78 (ru) | macro 0.77 / en 0.69 / best lang 0.77 (ru) | macro 0.70 / en 0.55 / best lang 0.72 (ru) |
| bbpb_xstorycloze | 4 | macro 0.87 / en 0.63 / best lang 0.91 (es) | macro 0.89 / en 0.64 / best lang 0.90 (es) | macro 0.79 / en 0.56 / best lang 0.81 (zh) | macro 0.81 / en 0.63 / best lang 0.86 (ru) | macro 0.86 / en 0.71 / best lang 0.89 (es) |
| global_piqa_parallel_cloze | 6 | — | — | — | macro 0.53 / en 0.44 / best lang 0.52 (ja) | macro 0.54 / en 0.49 / best lang 0.63 (ja) |
| hellaswag | 6 | macro 0.82 / en 0.68 / best lang 0.81 (es) | macro 0.82 / en 0.79 / best lang 0.82 (it) | macro 0.76 / en 0.69 / best lang 0.79 (it) | macro 0.75 / en 0.69 / best lang 0.77 (de) | macro 0.86 / en 0.84 / best lang 0.87 (es) |
| include_v2_en | 7 | macro 0.62 / best lang 0.66 (ja) | macro 0.60 / best lang 0.58 (fr) | macro 0.53 / best lang 0.54 (ru) | macro 0.56 / best lang 0.66 (zh) | macro 0.63 / best lang 0.63 (ru) |
| include_v2_og | 7 | macro 0.57 / best lang 0.62 (ru) | macro 0.60 / best lang 0.65 (zh) | macro 0.73 / best lang 0.67 (ru) | macro 0.72 / best lang 0.69 (it) | macro 0.75 / best lang 0.72 (ru) |
| lambada_openai_mt | 5 | macro 0.67 / en 0.54 / best lang 0.77 (de) | macro 0.68 / en 0.62 / best lang 0.71 (de) | macro 0.60 / en 0.52 / best lang 0.62 (es) | macro 0.66 / en 0.66 / best lang 0.70 (it) | macro 0.86 / en 0.80 / best lang 0.83 (de) |
| multiblimp | 6 | macro 0.73 / en 0.42 / best lang 0.73 (ru) | macro 0.68 / en 0.40 / best lang 0.72 (fr) | macro 0.78 / en 0.57 / best lang 0.76 (de) | macro 0.71 / en 0.48 / best lang 0.74 (ru) | macro 0.70 / en 0.52 / best lang 0.68 (es) |
| paws | 4 | macro 0.46 / best lang 0.46 (es) | macro 0.63 / en 0.63 / best lang 0.63 (en) | macro 0.62 / en 0.62 / best lang 0.62 (en) | macro 0.45 / en 0.55 / best lang 0.55 (en) | macro 0.51 / en 0.45 / best lang 0.59 (es) |
| rf_belebele | 8 | macro 0.66 / en 0.45 / best lang 0.66 (ru) | macro 0.68 / en 0.65 / best lang 0.65 (en) | macro 0.57 / en 0.56 / best lang 0.64 (fr) | macro 0.61 / en 0.48 / best lang 0.67 (zh) | macro 0.61 / en 0.53 / best lang 0.63 (ru) |
| rf_global_mmlu_full | 8 | macro 0.48 / en 0.40 / best lang 0.59 (es) | macro 0.71 / en 0.62 / best lang 0.75 (de) | macro 0.69 / en 0.56 / best lang 0.75 (it) | macro 0.72 / en 0.62 / best lang 0.76 (ja) | macro 0.79 / en 0.65 / best lang 0.80 (de) |
| rf_include_base_44 | 6 | macro 0.71 / best lang 0.64 (fr) | macro 0.66 / best lang 0.65 (fr) | macro 0.65 / best lang 0.59 (zh) | macro 0.72 / best lang 0.65 (es) | macro 0.65 / best lang 0.65 (ru) |
| rf_m_mmlu | 6 | macro 0.52 / best lang 0.57 (fr) | macro 0.61 / best lang 0.63 (fr) | macro 0.64 / best lang 0.71 (fr) | macro 0.82 / best lang 0.79 (de) | macro 0.75 / best lang 0.77 (ru) |
| rf_mmmlu | 6 | macro 0.54 / best lang 0.59 (es) | macro 0.69 / best lang 0.76 (es) | macro 0.72 / best lang 0.69 (it) | macro 0.72 / best lang 0.69 (de) | macro 0.83 / best lang 0.79 (de) |
| rfgm_belebele | 8 | macro 0.80 / en 0.59 / best lang 0.75 (zh) | macro 0.75 / en 0.55 / best lang 0.68 (ru) | macro 0.66 / en 0.54 / best lang 0.64 (ru) | macro 0.68 / en 0.42 / best lang 0.77 (fr) | macro 0.75 / en 0.58 / best lang 0.69 (de) |
| rfgm_include_base_44 | 6 | macro 0.69 / best lang 0.61 (zh) | macro 0.73 / best lang 0.69 (es) | macro 0.65 / best lang 0.71 (zh) | macro 0.80 / best lang 0.73 (fr) | macro 0.86 / best lang 0.70 (zh) |
| xnli | 6 | macro 0.53 / en 0.62 / best lang 0.62 (en) | macro 0.59 / en 0.41 / best lang 0.63 (fr) | macro 0.52 / en 0.44 / best lang 0.61 (de) | macro 0.59 / en 0.56 / best lang 0.62 (ru) | macro 0.43 / en 0.48 / best lang 0.54 (de) |
| xstorycloze | 4 | macro 0.76 / en 0.65 / best lang 0.68 (es) | macro 0.73 / en 0.69 / best lang 0.69 (en) | macro 0.79 / en 0.65 / best lang 0.74 (ru) | macro 0.73 / en 0.69 / best lang 0.75 (es) | macro 0.71 / en 0.59 / best lang 0.70 (es) |
| xwinograd | 5 | macro 0.44 / en 0.64 / best lang 0.64 (en) | macro 0.53 / en 0.35 / best lang 0.60 (zh) | macro 0.65 / en 0.63 / best lang 0.63 (en) | macro 0.65 / en 0.57 / best lang 0.62 (ja) | macro 0.52 / en 0.63 / best lang 0.63 (en) |
| all benchmarks | 8 | macro 0.65 / en 0.59 / one lang 0.59 | macro 0.65 / en 0.56 / one lang 0.56 | macro 0.67 / en 0.58 / one lang 0.58 | macro 0.68 / en 0.57 / one lang 0.57 | macro 0.68 / en 0.59 / one lang 0.59 |

![The minimal language panel](pretraining/predictivity/language_panel.png)
<!-- END auto:language-panel -->

**Key findings**

- **English alone is the weakest single-language proxy at 1B.** Pooled over
  the panel's benchmarks, English reads 0.58, against 0.60–0.66 for the other
  seven languages (de 0.66, zh 0.65, ru 0.64, it 0.64, es 0.63, fr 0.62,
  ja 0.60) and 0.67 for the panel macro. At 600M English (0.55) is again the
  lowest, below Japanese (0.60) and the macro (0.66).
- **English is the lowest single language at every size, and Japanese the
  second lowest from 175M up.** English reads 0.54–0.58 and Japanese 0.57–0.62:
  90M (en 0.57, ja 0.62, with Chinese second lowest at 0.61), 175M (en 0.55,
  ja 0.58), 350M (en 0.54, ja 0.57), 600M (en 0.55, ja 0.60) and 1B (en 0.58,
  ja 0.60).
- **Per benchmark, the macro beats English where it matters.** At 1B the
  proxy's macro reaches 0.88 on hellaswag and LAMBADA, 0.87 on INCLUDE-rfgm
  and 0.76 on Global-MMLU-rf. English alone reads 0.84, 0.80 and 0.65 on
  hellaswag, LAMBADA and Global-MMLU-rf (INCLUDE-rfgm has no English cell). On
  `multiblimp` English reads 0.50 against 0.68 for the macro.
- **Where the panel is worth evaluating.** At 1B the macro lies above every
  single language on 4 of the 18 benchmarks (INCLUDE-og, multiblimp,
  Belebele-rfgm, INCLUDE-rfgm) and ties the best one on hellaswag, LAMBADA
  and `paws`. There the languages' errors are partly independent. INCLUDE-rf
  trails Russian, 0.62 to 0.67. On `xnli` (macro 0.36) and `xwinograd` (0.40)
  no single language reads the reference above 0.59.
- **The macro is a safe proxy but not always the best one.** Pooled, it
  reads 0.66 at 600M and 0.67 at 1B, above every single language (French
  0.66 at 600M, German 0.66 and Chinese 0.65 at 1B). German beats it at 90M
  (0.672 against 0.657), 175M (0.653 against 0.650) and 350M (0.657 against
  0.654), and so does Italian at 90M (0.663).

**Follow-ups** (`plan/next_analyses.md` §6)

- Greedy k-language panels: which two or three languages recover the macro.
- The same against the macro over ALL trained languages per L (requires
  per-L pairs, rule 2).
- The seed null of the decision-accuracy analysis ([its seed-uncertainty figure](../rq02_decision_accuracy/README.md#7-seed-uncertainty-and-the-da-ckpt-null))
  as the floor of every panel.
- Drop `global_piqa_parallel_cloze` from the pooled row at 600M and 1B
  (90M–350M have no cell for it), so that the five sizes pool exactly the
  same 36 tasks.

GitHub: [language_panel.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity/language_panel.png) · [language_panel.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity/language_panel.csv)

## Extensions from other sweeps

None. The language-transfer analysis exists on the ladder only (`predictivity_seeds`, `predictivity`).
The 36-model sweep evaluated 12 languages, with no per-language BPB and no
never-trained language, so neither the leave-one-language-out fit nor the panel
question could be asked of it. Its numbers would not be pooled with the
ladder's in any case (a different harness, task set and reference size).

## Files

- `pretraining/<pool>/rq5_transfer.csv`, `rq5_transfer_summary.csv`,
  `rq5_transfer.png`: the paper's language-transfer table and figure.
- `…/bpb_curves.png`: per-cell per-language BPB curves.
- `…/facts.json`: the numbers the paper quotes.
- `pretraining/predictivity/language_panel.{png,csv}`: the minimal language
  panel (`language_panel.py`).
