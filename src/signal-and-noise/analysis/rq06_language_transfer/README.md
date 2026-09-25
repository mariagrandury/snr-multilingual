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

- **trained languages (106 (L, language) cases)** — one 175M rung plus the pooled exponent predicts the reference within 4.0 % (median); with every proxy rung, transferred 2.1 % vs own fit 4.6 % vs largest proxy as is 7.6 %.
- **never-trained languages (494 (L, language) cases)** — one 175M rung plus the pooled exponent predicts the reference within 7.0 % (median); with every proxy rung, transferred 5.2 % vs own fit 3.0 % vs largest proxy as is 6.6 %.
- **Pooled exponent α by L**: L1 0.141, L2 0.163, L8 0.163, L15 0.153, L30 0.160, L50 0.181.
<!-- END auto:highlight -->

## Experimental setup

Per-language BPB of the plan grid (deep, scheme A, seed 1904) at the final
checkpoint, from 175M up, at every L with at least four sizes (three proxy
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
- **Decision transfer.** rq05's final-checkpoint agreement on the
  `bpb_untrained` population against `bpb_trained` — whether a decision read
  on the trained languages also holds on the ones neither level trains.
- **BPB curves.** One panel per cell of the pool, per-language BPB against
  fraction of run: trained languages blue, unseen grey, macro BPB dashed
  (the progress report's `plot_bpb` on the analysis' cells).

Hand-written numbers in this README are from the ladder-report snapshot
**2026-09-23 06:16**.

<!-- BEGIN auto:results (analyze.py --pool predictivity_all) -->
## Results

Numbers from the `predictivity_all` pool. Regenerate with `python analysis/rq06_language_transfer/analyze.py --pool predictivity_all`.

**Median |relative error| of the reference's BPB** (k = rungs measured for the held-out language):

| languages | k | transferred α | own fit | largest proxy | n |
|---|---|---|---|---|---|
| never trained | 1 | 0.070 |  | 0.384 | 494 |
| never trained | 2 | 0.068 |  | 0.188 | 494 |
| never trained | 3 | 0.061 | 0.063 | 0.121 | 494 |
| never trained | 4 | 0.052 | 0.030 | 0.066 | 494 |
| trained | 1 | 0.040 |  | 0.497 | 106 |
| trained | 2 | 0.027 |  | 0.257 | 106 |
| trained | 3 | 0.024 | 0.088 | 0.161 | 106 |
| trained | 4 | 0.021 | 0.046 | 0.076 | 106 |

![Transfer](pretraining/predictivity_all/rq5_transfer.png)

**Decision transfer** (rq05's final-checkpoint agreement, mean over interventions and L, on the languages both levels train vs the languages neither does):

| population | proxy | agreement | cells |
|---|---|---|---|
| bpb_trained | 175M | 0.79 | 10 |
| bpb_trained | 350M | 0.78 | 10 |
| bpb_trained | 600M | 0.53 | 9 |
| bpb_trained | 1B | 0.88 | 8 |

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

![The list decision on untrained languages, by proxy size and checkpoint](pretraining/predictivity_all/transfer_da_lines.png)

![The decisions on untrained languages, by language count](pretraining/predictivity_all/transfer_da_by_L.png)
<!-- END auto:panels -->

**Key findings** (`transfer_da_lines`, `transfer_da_by_L`: rq05's per-item
agreement on the per-language BPB of all 100 evaluation languages, grouped
per (intervention, L) by what the two levels' lists do with the language —
both train it, only one does, neither does but one trains its script, or
neither trains even the script; the last two groups are the transfer test)

- The "trained languages read the decision at 1.0" reading is the languages
  only one list trains, where the model that saw the language wins at every
  size; it is the inclusion decision, not transfer.
- Script is a coarse proxy for relatedness (Latin covers Welsh and Vietnamese
  alike), and at L1 only English is trained, so "script trained" is every
  Latin-script language.
- `transfer_da_lines.csv` is empty on the 2026-09-23 06:16 tables (the L8 and
  L30 list decisions against 1.7B do not reach the three items a panel needs
  at this snapshot), so the per-size numbers of the earlier snapshot
  (script-trained languages read at 0.61–0.77 from 175M to 1B, unseen scripts
  0.33–0.73, neither clearing 0.75 below 1B) are not re-read here; they are
  quoted again once the figure draws.

**Follow-ups**

- Group the untrained languages by the trained tokens of their language
  family or genus (from the data manifest), or by tokenizer overlap with the
  trained languages, and plot DA against that exposure instead of three bins.
- Temperature, ZH and ES have one L each (single points); fold them into a
  table. The "mean over proxy sizes" of `transfer_da_by_L` averages a proxy
  that flips with one that agrees into 0.5; one line per proxy size, or the
  largest proxy below the reference, for the paper version.
- The same caveats as rq05: items are correlated and the reference is 1.7B
  only for the lists; bootstrap over L and re-read the other interventions
  against 1.7B when their cells exist.

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/highlights.csv) ·
GitHub: [transfer_error_by_L.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_error_by_L.png) · [transfer_error_by_L.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_error_by_L.csv) ·
GitHub: [transfer_da_lines.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_da_lines.png) · [transfer_da_lines.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_da_lines.csv) ·
GitHub: [transfer_da_by_L.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_da_by_L.png) · [transfer_da_by_L.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_all/transfer_da_by_L.csv)

**The minimal language panel · DA-size against the 1.7B MACRO ranking
(mean score over the L8 panel languages above chance at 1.7B) · no filter ·
multi-axis pairs from `predictivity_schemes` at seed 1904 · gate
`predictivity`.** Which languages does a developer have to evaluate to
recover the multilingual decision: one language at a time, English alone, or
the proxy's own macro over the languages readable at its size.

<!-- BEGIN auto:language-panel (language_panel.py --pool predictivity) -->
## The minimal language panel

Per benchmark family, decision accuracy of a proxy size against the 1.7B MACRO ranking of the design variants (mean over the L8 panel languages above chance at 1.7B), when the proxy reads one language, English, or its own macro over the languages readable at that size. Grid-seed pairs of every scheme (multi-axis), ≥ 3 pairs, gate `predictivity`. A macro above every single language says the languages' errors are independent and the panel is worth evaluating; a single language at the macro says it suffices. Regenerate with `python analysis/rq06_language_transfer/language_panel.py --pool predictivity`.

| benchmark | languages readable at 1.7B | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| arc | 7 | macro 0.43 / en 0.43 / best lang 0.43 (en) | macro 0.52 / en 0.52 / best lang 0.52 (en) | macro 0.68 / en 0.68 / best lang 0.68 (en) | macro 0.62 / en 0.52 / best lang 0.66 (it) |
| hellaswag | 6 | macro 0.75 / en 0.65 / best lang 0.75 (es) | macro 0.88 / en 0.69 / best lang 0.92 (es) | macro 0.84 / en 0.69 / best lang 0.87 (fr) | macro 0.93 / en 0.87 / best lang 0.92 (es) |
| include_v2_en | 6 | macro 0.46 / best lang 0.60 (ja) | macro 0.35 / best lang 0.66 (fr) | macro 0.55 / best lang 0.60 (ru) | macro 0.53 / best lang 0.67 (de) |
| include_v2_og | 6 | macro 0.69 / best lang 0.69 (ru) | macro 0.57 / best lang 0.60 (ru) | macro 0.64 / best lang 0.74 (it) | macro 0.66 / best lang 0.68 (it) |
| lambada_openai_mt | 5 | macro 0.62 / en 0.53 / best lang 0.76 (de) | macro 0.70 / en 0.59 / best lang 0.70 (es) | macro 0.78 / en 0.75 / best lang 0.84 (it) | macro 0.89 / en 0.78 / best lang 0.89 (de) |
| multiblimp | 6 | macro 0.67 / en 0.51 / best lang 0.68 (es) | macro 0.80 / en 0.53 / best lang 0.79 (de) | macro 0.70 / en 0.58 / best lang 0.71 (ru) | macro 0.68 / en 0.40 / best lang 0.67 (de) |
| paws | 4 | — | macro 0.69 / en 0.69 / best lang 0.69 (en) | macro 0.41 / en 0.60 / best lang 0.60 (en) | macro 0.52 / en 0.48 / best lang 0.56 (de) |
| rf_belebele | 8 | macro 0.64 / en 0.46 / best lang 0.70 (fr) | macro 0.57 / en 0.48 / best lang 0.63 (zh) | macro 0.55 / en 0.53 / best lang 0.66 (fr) | macro 0.62 / en 0.59 / best lang 0.64 (ru) |
| rf_global_mmlu_full | 8 | macro 0.58 / en 0.58 / best lang 0.68 (es) | macro 0.71 / en 0.52 / best lang 0.77 (it) | macro 0.75 / en 0.55 / best lang 0.79 (ja) | macro 0.81 / en 0.69 / best lang 0.81 (de) |
| rf_include_base_44 | 7 | macro 0.48 / best lang 0.45 (fr) | macro 0.70 / best lang 0.66 (es) | macro 0.79 / best lang 0.74 (es) | macro 0.69 / best lang 0.73 (ru) |
| rfgm_include_base_44 | 6 | macro 0.33 / best lang 0.33 (fr) | macro 0.70 / best lang 0.70 (ru) | macro 0.82 / best lang 0.76 (fr) | macro 0.89 / best lang 0.75 (zh) |
| xnli | 6 | macro 0.60 / en 0.63 / best lang 0.63 (en) | macro 0.48 / en 0.44 / best lang 0.68 (de) | macro 0.55 / en 0.55 / best lang 0.67 (ru) | macro 0.33 / en 0.45 / best lang 0.54 (de) |
| xstorycloze | 4 | macro 0.62 / en 0.62 / best lang 0.62 (en) | macro 0.90 / en 0.68 / best lang 0.80 (es) | macro 0.76 / en 0.68 / best lang 0.77 (es) | macro 0.77 / en 0.64 / best lang 0.75 (es) |
| xwinograd | 5 | macro 0.48 / en 0.58 / best lang 0.58 (en) | macro 0.67 / en 0.63 / best lang 0.63 (en) | macro 0.53 / en 0.42 / best lang 0.56 (ja) | macro 0.26 / en 0.48 / best lang 0.51 (zh) |
| all benchmarks | 8 | macro 0.57 / en 0.55 / one lang 0.55 | macro 0.66 / en 0.58 / one lang 0.58 | macro 0.67 / en 0.60 / one lang 0.60 | macro 0.66 / en 0.59 / one lang 0.59 |

![The minimal language panel](pretraining/predictivity/language_panel.png)
<!-- END auto:language-panel -->

**Key findings**

- English alone is the worst single-language proxy of the multilingual
  decision: pooled over twelve benchmarks at 1B English reads 0.59 against
  0.61–0.67 for the other panel languages (de 0.67, ru 0.65, zh 0.64, it 0.64,
  es 0.62, fr 0.61; ja 0.54) and 0.66 for the panel macro; at 600M English is
  again the lowest at 0.60.
- Per benchmark the proxy's macro reaches 0.93 on hellaswag, 0.89 on LAMBADA
  and INCLUDE-rfgm and 0.81 on Global-MMLU-rf at 1B, where English alone reads
  0.87, 0.78, — and 0.69; on `multiblimp` English reads 0.40 against 0.68 for
  the macro. Where the macro lies above every single language (hellaswag,
  Global-MMLU-rf, INCLUDE-rf/rfgm, xstorycloze) the languages' errors are
  partly independent and the panel is worth evaluating; on `xnli` and
  `xwinograd` nothing reads the reference at 1B.
- A developer who evaluates a multilingual recipe on English benchmarks
  alone misreads the multilingual decision more often than one who evaluates
  any other single panel language; the macro over the readable panel is the
  safest proxy at every size.

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
