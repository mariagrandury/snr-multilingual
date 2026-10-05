# RQ6 — Does it generalize to languages we have not measured? (paper RQ5)

## Research question

> If a language's BPB was measured on one or two small rungs only, can the
> exponent pooled over the other languages predict its BPB at the reference
> size — and does that hold for languages the model never trained on? The
> paper's RQ5
> ([`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex)).
> The folder also holds the detailed per-language BPB curves of every cell.

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_all) -->
## Highlighted result

- **trained languages (106 (L, language) cases)** — one 175M rung plus the pooled exponent predicts the reference within 4.4 % (median); with every proxy rung, transferred 3.9 % vs own fit 1.5 % vs largest proxy as is 7.6 %.
- **never-trained languages (494 (L, language) cases)** — one 175M rung plus the pooled exponent predicts the reference within 5.0 % (median); with every proxy rung, transferred 3.1 % vs own fit 2.7 % vs largest proxy as is 6.6 %.
- **Pooled exponent α by L**: L1 0.090, L2 0.096, L8 0.096, L15 0.094, L30 0.106, L50 0.117.
<!-- END auto:highlight -->

## Experimental setup

Per-language BPB of the plan grid (deep, scheme A, seed 1904) at the final
checkpoint, from 90M up, at every L with at least four sizes (three proxy
rungs and a reference, the largest size at that L). Every one of the 100
validation languages is a series; "trained" means the language is in the
cell's FineWeb-2 list at that L (`launch_trainings.cell_fineweb_subsets`).

## Methodology

- **Own exponent.** log BPB = a − α log N per language over the proxy rungs.
- **Leave-one-language-out.** For each language the exponent is the median
  of the other languages' α at that L; the intercept comes from the held-out
  language's k smallest rungs (k = 1 … all); the prediction at the reference
  is compared with the language's own fit (k ≥ 3) and with the largest proxy's
  BPB taken as is. Relative errors, medians per (trained, k).
- **Decision transfer.** rq05's five interventions (its `intervention_da`)
  read on every evaluation language's BPB, the one population rule 2 lets
  this folder use: per (intervention, L, proxy size, checkpoint) the share of
  a language group's languages on which the proxy ranks the two levels as the
  reference does at its final checkpoint. `analyze.py` writes it
  (`transfer_da_all_by_group_mono_axis.csv`), `panels.py` draws it.
- **BPB curves.** One panel per cell of the pool, per-language BPB against
  fraction of run: trained languages blue, unseen grey, macro BPB dashed
  (the progress report's `plot_bpb` on the analysis' cells).

Hand-written numbers in this README are from the ladder-report snapshot
**2026-09-30 23:54**.

<!-- BEGIN auto:results (analyze.py --pool predictivity_all) -->
## Results

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq06_language_transfer/analyze.py --pool predictivity_all`.

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

![Transfer](pretraining/predictivity_all/rq5_transfer.png)

**Decision transfer** (DA-size of rq05's interventions on per-language BPB: the share of a group's languages on which the proxy's final ranking of the two levels matches the reference's, mean over interventions and L; the group says what the two levels' lists do with the language):

| language group | proxy | DA-size | cells |
|---|---|---|---|
| trained by both levels | 90M | 0.95 | 14 |
| trained by both levels | 175M | 0.96 | 14 |
| trained by both levels | 350M | 0.81 | 14 |
| trained by both levels | 600M | 0.66 | 14 |
| trained by both levels | 1B | 0.89 | 14 |
| trained by one level | 90M | 1.00 | 5 |
| trained by one level | 175M | 1.00 | 5 |
| trained by one level | 350M | 1.00 | 5 |
| trained by one level | 600M | 1.00 | 5 |
| trained by one level | 1B | 1.00 | 5 |
| script trained | 90M | 0.67 | 14 |
| script trained | 175M | 0.71 | 14 |
| script trained | 350M | 0.65 | 14 |
| script trained | 600M | 0.66 | 14 |
| script trained | 1B | 0.71 | 14 |
| script not trained | 90M | 0.52 | 14 |
| script not trained | 175M | 0.59 | 14 |
| script not trained | 350M | 0.58 | 14 |
| script not trained | 600M | 0.58 | 14 |
| script not trained | 1B | 0.64 | 14 |

![BPB curves](pretraining/predictivity_all/bpb_curves.png)
<!-- END auto:results -->

GitHub: [rq5_transfer.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/rq5_transfer.png) · [rq5_transfer.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/rq5_transfer.csv) ·
[rq5_transfer_summary.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/rq5_transfer_summary.csv) ·
GitHub: [bpb_curves.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/bpb_curves.png) · [bpb_curves.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/bpb_curves.csv) ·
[facts.json](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/facts.json)

<!-- BEGIN auto:panels (panels.py --pool predictivity_all) -->
## Per benchmark and per language

The summary above, per language (`predictivity_all` pool). Regenerate with `python analysis/rq06_language_transfer/panels.py --pool predictivity_all`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq06 in one figure](pretraining/predictivity_all/highlights.png)

![Transfer error per language and L](pretraining/predictivity_all/transfer_error_by_L.png)

![The list decision on untrained languages, by proxy size and checkpoint](pretraining/predictivity_all/transfer_da_all_lines_mono_axis.png)

![The decisions on untrained languages, by language count](pretraining/predictivity_all/transfer_da_all_by_L_mono_axis.png)
<!-- END auto:panels -->

**Key findings** (`transfer_da_all_lines_mono_axis`, `transfer_da_all_by_L_mono_axis`: rq05's per-item
agreement on the per-language BPB of every evaluation language, grouped
per (intervention, L) by what the two levels' lists do with the language —
both train it, only one does, neither does but one trains its script, or
neither trains even the script; the last two groups are the transfer test)

- **The list decision (A vs B) transfers only partly.** DA-size on languages
  whose script a list trains is 0.63 at 90M and 0.72–0.73 from 175M to 600M,
  and clears 0.75 only at 1B (0.81). On unseen scripts it is 0.42–0.73 and
  never clears 0.75. The reference's own early checkpoints do better: DA-ckpt
  on script-trained languages climbs from 0.65 at 10 % to 0.90 at 90 %, and
  on unseen scripts from 0.50 to 0.85.
- **Per language count** (final checkpoint, mean over proxy sizes):
  script-trained languages read the list decision at L8 (0.82) and L15
  (0.77) but not L30 (0.58), temperature at every L (0.78–0.79) and the
  second-language swaps clearly (zh 0.96, es 0.90). Depth is not read on
  untrained languages at any L (0.41–0.72), and unseen scripts stay at
  0.43–0.68 for every intervention.
- The "trained languages read the decision at 1.0" reading is the languages
  only one list trains, where the model that saw the language wins at every
  size; it is the inclusion decision, not transfer.
- Script is a coarse proxy for relatedness (Latin covers Welsh and Vietnamese
  alike), and at L1 only English is trained, so "script trained" is every
  Latin-script language.

**Follow-ups**

- Group the untrained languages by the trained tokens of their language
  family or genus (from the data manifest), or by tokenizer overlap with the
  trained languages, and plot DA against that exposure instead of three bins.
- Temperature, ZH and ES have one L each (single points); fold them into a
  table. The "mean over proxy sizes" of `transfer_da_all_by_L_mono_axis` averages a proxy
  that flips with one that agrees into 0.5; one line per proxy size, or the
  largest proxy below the reference, for the paper version.
- The same caveats as rq05: items are correlated; bootstrap over L. The
  reference is 1.7B for every intervention since 2026-09-26.

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/highlights.csv) ·
GitHub: [transfer_error_by_L.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_error_by_L.png) · [transfer_error_by_L.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_error_by_L.csv) ·
GitHub: [transfer_da_all_lines_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_da_all_lines_mono_axis.png) · [transfer_da_all_lines_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_da_all_lines_mono_axis.csv) ·
GitHub: [transfer_da_all_by_L_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_da_all_by_L_mono_axis.png) · [transfer_da_all_by_L_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_da_all_by_L_mono_axis.csv)

**The minimal language panel · DA-size against the 1.7B MACRO ranking
(mean score over the L8 panel languages above chance at 1.7B) · no filter ·
multi-axis pairs from `predictivity_schemes` at seed 1904 · gate
`predictivity`.** Which languages does a developer have to evaluate to
recover the multilingual decision: one language at a time, English alone, or
the proxy's own macro over the languages readable at its size.

<!-- BEGIN auto:language-panel (language_panel.py --pool predictivity) -->
## The minimal language panel

Per benchmark family, decision accuracy of a proxy size against the 1.7B MACRO ranking of the design variants (mean over the L8 panel languages above chance at 1.7B), when the proxy reads one language, English, or its own macro over the languages readable at that size. Grid-seed pairs of every scheme (multi-axis), ≥ 3 pairs, gate `predictivity`. A macro above every single language says the languages' errors are independent and the panel is worth evaluating; a single language at the macro says it suffices. Regenerate with `python analysis/rq06_language_transfer/language_panel.py --pool predictivity`.

| benchmark | languages readable at 1.7B | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| arc | 7 | macro 0.55 / en 0.55 / best lang 0.55 (en) | macro 0.43 / en 0.43 / best lang 0.43 (en) | macro 0.52 / en 0.52 / best lang 0.52 (en) | macro 0.68 / en 0.68 / best lang 0.68 (en) | macro 0.62 / en 0.52 / best lang 0.66 (it) |
| hellaswag | 6 | macro 0.82 / en 0.67 / best lang 0.84 (fr) | macro 0.89 / en 0.85 / best lang 0.89 (de) | macro 0.88 / en 0.69 / best lang 0.92 (es) | macro 0.84 / en 0.69 / best lang 0.87 (fr) | macro 0.93 / en 0.87 / best lang 0.92 (es) |
| include_v2_en | 6 | macro 0.66 / best lang 0.64 (ja) | macro 0.53 / best lang 0.53 (fr) | macro 0.35 / best lang 0.66 (fr) | macro 0.55 / best lang 0.60 (ru) | macro 0.53 / best lang 0.67 (de) |
| include_v2_og | 6 | macro 0.51 / best lang 0.52 (ru) | macro 0.54 / best lang 0.56 (ru) | macro 0.57 / best lang 0.60 (ru) | macro 0.64 / best lang 0.74 (it) | macro 0.66 / best lang 0.68 (it) |
| lambada_openai_mt | 5 | macro 0.78 / en 0.59 / best lang 0.85 (de) | macro 0.81 / en 0.66 / best lang 0.84 (de) | macro 0.70 / en 0.59 / best lang 0.70 (es) | macro 0.78 / en 0.75 / best lang 0.84 (it) | macro 0.89 / en 0.78 / best lang 0.89 (de) |
| multiblimp | 6 | macro 0.71 / en 0.43 / best lang 0.74 (it) | macro 0.73 / en 0.36 / best lang 0.74 (es) | macro 0.80 / en 0.53 / best lang 0.79 (de) | macro 0.70 / en 0.58 / best lang 0.71 (ru) | macro 0.68 / en 0.40 / best lang 0.67 (de) |
| paws | 4 | — | — | macro 0.69 / en 0.69 / best lang 0.69 (en) | macro 0.41 / en 0.60 / best lang 0.60 (en) | macro 0.52 / en 0.48 / best lang 0.56 (de) |
| rf_belebele | 8 | macro 0.66 / en 0.44 / best lang 0.65 (es) | macro 0.62 / en 0.64 / best lang 0.64 (en) | macro 0.57 / en 0.48 / best lang 0.63 (zh) | macro 0.55 / en 0.53 / best lang 0.66 (fr) | macro 0.62 / en 0.59 / best lang 0.64 (ru) |
| rf_global_mmlu_full | 8 | macro 0.47 / en 0.42 / best lang 0.55 (es) | macro 0.73 / en 0.54 / best lang 0.73 (de) | macro 0.71 / en 0.52 / best lang 0.77 (it) | macro 0.75 / en 0.55 / best lang 0.79 (ja) | macro 0.81 / en 0.69 / best lang 0.81 (de) |
| rf_include_base_44 | 7 | macro 0.80 / best lang 0.71 (ru) | macro 0.73 / best lang 0.75 (fr) | macro 0.70 / best lang 0.66 (es) | macro 0.79 / best lang 0.74 (es) | macro 0.69 / best lang 0.73 (ru) |
| rfgm_belebele | 8 | macro 0.82 / en 0.55 / best lang 0.75 (zh) | macro 0.78 / en 0.59 / best lang 0.74 (fr) | macro 0.69 / en 0.54 / best lang 0.68 (ru) | macro 0.60 / en 0.53 / best lang 0.79 (fr) | macro 0.86 / en 0.76 / best lang 0.80 (es) |
| rfgm_include_base_44 | 6 | macro 0.66 / best lang 0.64 (fr) | macro 0.73 / best lang 0.66 (zh) | macro 0.70 / best lang 0.70 (ru) | macro 0.82 / best lang 0.76 (fr) | macro 0.89 / best lang 0.75 (zh) |
| xnli | 6 | macro 0.54 / en 0.63 / best lang 0.63 (en) | macro 0.62 / en 0.36 / best lang 0.69 (fr) | macro 0.48 / en 0.44 / best lang 0.68 (de) | macro 0.55 / en 0.55 / best lang 0.67 (ru) | macro 0.33 / en 0.45 / best lang 0.54 (de) |
| xstorycloze | 4 | macro 0.74 / en 0.58 / best lang 0.69 (es) | macro 0.77 / en 0.67 / best lang 0.67 (en) | macro 0.90 / en 0.68 / best lang 0.80 (es) | macro 0.76 / en 0.68 / best lang 0.77 (es) | macro 0.77 / en 0.64 / best lang 0.75 (es) |
| xwinograd | 5 | macro 0.45 / en 0.59 / best lang 0.59 (en) | macro 0.35 / en 0.26 / best lang 0.49 (ja) | macro 0.67 / en 0.63 / best lang 0.63 (en) | macro 0.53 / en 0.42 / best lang 0.56 (ja) | macro 0.26 / en 0.48 / best lang 0.51 (zh) |
| all benchmarks | 8 | macro 0.66 / en 0.55 / one lang 0.55 | macro 0.66 / en 0.54 / one lang 0.54 | macro 0.66 / en 0.57 / one lang 0.57 | macro 0.66 / en 0.60 / one lang 0.60 | macro 0.67 / en 0.61 / one lang 0.61 |

![The minimal language panel](pretraining/predictivity/language_panel.png)
<!-- END auto:language-panel -->

**Key findings**

- English alone is a weak single-language proxy of the multilingual
  decision, beaten by every panel language but Japanese: pooled over the panel's benchmarks at 1B English reads 0.61
  against 0.62–0.67 for the other panel languages (de 0.67, it 0.65, zh 0.65,
  ru 0.64, es 0.63, fr 0.62; ja 0.55) and 0.67 for the panel macro; at 600M
  English (0.60) sits with Japanese and Chinese (0.56) at the bottom, below
  the macro's 0.66.
- Per benchmark the proxy's macro reaches 0.93 on hellaswag, 0.89 on LAMBADA
  and INCLUDE-rfgm and 0.81 on Global-MMLU-rf at 1B, where English alone reads
  0.87, 0.78, — and 0.69; on `multiblimp` English reads 0.40 against 0.68 for
  the macro. Where the macro lies above every single language at 1B
  (hellaswag, multiblimp, Belebele-rfgm, INCLUDE-rfgm, xstorycloze;
  Global-MMLU-rf and LAMBADA tie the best language, INCLUDE-rf trails
  Russian 0.69 to 0.73) the languages' errors are
  partly independent and the panel is worth evaluating; on `xnli` and
  `xwinograd` nothing reads the reference at 1B.
- A developer who evaluates a multilingual recipe on English benchmarks
  alone misreads the multilingual decision more often than one who evaluates
  any other single panel language except Japanese, which reads lowest at
  every size; the macro over the readable panel is the safest proxy from
  350M up (tied with German at 1B, 0.67), while Italian at 90M (0.66) and
  German at 175M (0.69) read the decision better than it.

**Follow-ups** (`plan/next_analyses.md` §6)

- Greedy k-language panels: which two or three languages recover the macro.
- The same against the macro over ALL trained languages per L (requires
  per-L pairs, rule 2).
- The seed null of [rq02 figure 7](../rq02_decision_accuracy/README.md#7-seed-uncertainty-and-the-da-ckpt-null)
  as the floor of every panel.

GitHub: [language_panel.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity/language_panel.png) · [language_panel.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity/language_panel.csv)

## Extensions from other sweeps

None. rq06 exists on the ladder only (`predictivity_all`, `predictivity`):
the 36-model sweep evaluated 12 languages with no per-language BPB and no
never-trained language, so neither the leave-language-out fit nor the panel
question could be asked of it, and its numbers would not be pooled with the
ladder's in any case (a different harness, task set and reference size).

## Files

- `pretraining/<pool>/rq5_transfer.csv`, `rq5_transfer_summary.csv`,
  `rq5_transfer.png/.pdf` — the paper's RQ5 table and figure.
- `…/bpb_curves.png` — per-cell per-language BPB curves.
- `…/facts.json` — the numbers the paper quotes.
- `pretraining/predictivity/language_panel.{png,csv}` — the minimal language
  panel (`language_panel.py`).
