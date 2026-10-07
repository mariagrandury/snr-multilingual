# Language transfer — does it generalize to languages we have not measured?

## Research question

> If a language's BPB was measured on one or two small rungs only, can the
> exponent pooled over the other languages predict its BPB at the reference
> size, and does that hold for languages the model never trained on? Does a
> design decision a proxy reads on its trained languages carry over to the
> languages it did not train, and which languages must a developer evaluate
> to recover the multilingual decision? The paper's language-transfer
> analysis
> ([`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex)).
> The folder also holds the detailed per-language BPB curves of every cell.

<!-- BEGIN auto:highlight (analyze.py --pool predictivity_seeds) -->
## Highlighted result

- **trained languages (106 (L, language) cases)** — one 175M rung plus the pooled exponent predicts the reference within 4.4 % (median); with every proxy rung, transferred 3.9 % vs own fit 1.5 % vs largest proxy as is 7.6 %.
- **never-trained languages (494 (L, language) cases)** — one 175M rung plus the pooled exponent predicts the reference within 5.0 % (median); with every proxy rung, transferred 3.1 % vs own fit 2.7 % vs largest proxy as is 6.6 %.
- **Pooled exponent α by L**: L1 0.090, L2 0.096, L8 0.096, L15 0.094, L30 0.106, L50 0.117.
<!-- END auto:highlight -->

## Experimental setup

Exponent transfer: per-language BPB of the plan grid (pool
`predictivity_seeds` cut to deep, data A, seed 1904) at the final
checkpoint, proxies 175M–1B (90M is left out of the fit) and reference 1.7B,
at all six L (1, 2, 8, 15, 30, 50). Every one of the 100 validation
languages is a series, 600 (L, language) cases: 106 trained (the language is
in the cell's FineWeb-2 list at that L,
`launch_trainings.cell_fineweb_subsets`) and 494 never trained.

Decision transfer: the design-decisions analysis' four mono-axis
interventions (pool `predictivity_seeds`, reference 1.7B) read on the
per-language BPB of the 100 languages. Depth (deep vs shallow) at all six L;
scheme A vs B at L1 (A vs DCLMP), L2 (A vs ZH) and L8–L30 (the list
decision); scheme A vs C at L1 (A vs FWEB) and L2 (A vs ES); temperature
(T=1 vs T=3) at L15–L50.

Language panel: pool `predictivity`, seed 1904, 14 design variants of every
data build (91 multi-axis pairs per benchmark), gate `predictivity`, no
filter, reference 1.7B, the eight L8 languages; the bBPB twins exist at the
final checkpoint only in this refresh.

## Key figure

![The list decision by language group, by proxy size and by the reference's checkpoints](pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis_paper.png)

Population: pool `predictivity_seeds`, the list decision (scheme A vs B on the L8, L15 and L30 lists, mono-axis pairs, reference 1.7B), DA-size (left: proxies 90M–1B at their final checkpoint) and DA-ckpt (right: the reference's own checkpoints at 10–90 % of its run), no gate or task filter (per-language BPB has no chance level), mean over the three L. Items are the 100 evaluation languages grouped by what the two lists do with each: per L, 5–26 trained by both, 6–18 by one, 53–66 untrained with a trained script, 13–23 with an untrained script; dotted line = 0.75.

**Key finding.** The list decision does not transfer to untrained languages at a proxy's scale: languages whose script a list trains are read at 0.63 (90M) and 0.72–0.73 (175M–600M), 0.81 only at 1B, and unseen scripts at 0.42–0.73, never 0.75. The languages the lists train are read at 0.77–1.00 (both lists) and 1.00 (one list).

GitHub: [transfer_da_all_lines_mono_axis_paper.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis_paper.png) · [transfer_da_all_lines_mono_axis_paper.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis_paper.csv). Every intervention by language count: [Per benchmark and per language](#per-benchmark-and-per-language).

## Methodology

- **Own exponent.** log BPB = a − α log N per language over the proxy rungs.
- **Leave-one-language-out.** For each language the exponent is the median
  of the other languages' α at that L; the intercept comes from the held-out
  language's k smallest rungs (k = 1 … all); the prediction at the reference
  is compared with the language's own fit (k ≥ 3) and with the largest proxy's
  BPB taken as is. Relative errors, medians per (trained, k).
- **Decision transfer.** The design-decisions analysis' four interventions
  (its `intervention_da`: depth, scheme A vs B, scheme A vs C, temperature)
  read on every evaluation language's BPB, the one population rule 2 lets
  this folder use: per (intervention, L, proxy size, checkpoint) the share of
  a language group's languages on which the proxy ranks the two levels as the
  reference does at its final checkpoint. `analyze.py` writes it
  (`transfer_da_all_by_group_mono_axis.csv`), `panels.py` draws it.
- **BPB curves.** One panel per cell of the pool, per-language BPB against
  fraction of run: trained languages blue, unseen grey, macro BPB dashed
  (the progress report's `plot_bpb` on the analysis' cells).

Hand-written numbers in this README are from the ladder-report snapshot
**2026-10-06 04:26** (refresh at commit b316f53b, read 2026-10-07). The bBPB
twins are scored at final checkpoints only in this refresh.

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

**Decision transfer** (DA-size of rq05's interventions on per-language BPB: the share of a group's languages on which the proxy's final ranking of the two levels matches the reference's, mean over interventions and L; the group says what the two levels' lists do with the language):

| language group | proxy | DA-size | cells |
|---|---|---|---|
| trained by both levels | 90M | 0.95 | 16 |
| trained by both levels | 175M | 0.96 | 16 |
| trained by both levels | 350M | 0.83 | 16 |
| trained by both levels | 600M | 0.70 | 16 |
| trained by both levels | 1B | 0.91 | 16 |
| trained by one level | 90M | 1.00 | 5 |
| trained by one level | 175M | 1.00 | 5 |
| trained by one level | 350M | 1.00 | 5 |
| trained by one level | 600M | 1.00 | 5 |
| trained by one level | 1B | 1.00 | 5 |
| script trained | 90M | 0.70 | 16 |
| script trained | 175M | 0.73 | 16 |
| script trained | 350M | 0.69 | 16 |
| script trained | 600M | 0.69 | 16 |
| script trained | 1B | 0.74 | 16 |
| script not trained | 90M | 0.56 | 16 |
| script not trained | 175M | 0.62 | 16 |
| script not trained | 350M | 0.61 | 16 |
| script not trained | 600M | 0.61 | 16 |
| script not trained | 1B | 0.68 | 16 |

![BPB curves](pretraining/predictivity_seeds/bpb_curves.png)
<!-- END auto:results -->

GitHub: [rq5_transfer.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/rq5_transfer.png) · [rq5_transfer.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/rq5_transfer.csv) ·
[rq5_transfer_summary.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/rq5_transfer_summary.csv) ·
GitHub: [bpb_curves.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/bpb_curves.png) · [bpb_curves.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/bpb_curves.csv) ·
[facts.json](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/facts.json)

**Key findings** (`rq5_transfer`: deep, data A, seed 1904, proxies 175M–1B,
reference 1.7B, six L; 106 trained and 494 never-trained (L, language) cases)

- **One small rung plus the pooled exponent beats the largest proxy read as
  is.** From the 175M rung alone the 1.7B BPB is predicted within 4.4 %
  (trained) and 5.0 % (never trained), median; the 1B proxy's own BPB taken as
  the reference's misses by 7.6 % and 6.6 %.
- **The pooled exponent serves untrained languages better than trained ones.**
  With all four proxies the transferred exponent errs 3.1 % against the
  language's own fit's 2.7 % on never-trained languages (it wins on 44 % of
  cases), but 3.9 % against 1.5 % on trained ones (13 %). Trained languages
  scale more steeply (median own α 0.12 against 0.09 for never-trained), while
  the pooled α (0.090–0.117 by L) is set mostly by the untrained languages at
  L1–L30 (70–99 of 100). At L50 the split is even (50/50) and the pooled α is
  highest (0.117).
- **The error grows with L.** From one 175M rung, never-trained languages are
  predicted within 3.6 % at L1 and 7.6 % at L50, and trained ones within 2.2 %
  (English alone, L1) to 5.1 % (L50).

**Follow-ups**

- Pool the exponent over the trained languages only for the trained
  held-out language, to test whether the 3.9 % against 1.5 % gap is the
  exponent mismatch above.

<!-- BEGIN auto:panels (panels.py --pool predictivity_seeds) -->
## Per benchmark and per language

The summary above, per language (`predictivity_seeds` pool). Regenerate with `python analysis/rq06_language_transfer/panels.py --pool predictivity_seeds`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq06 in one figure](pretraining/predictivity_seeds/highlights.png)

![Transfer error per language and L](pretraining/predictivity_seeds/transfer_error_by_L.png)

![The list decision (scheme A vs B at L8–L30) on untrained languages, by proxy size and checkpoint](pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis.png)

![The decisions on untrained languages, by language count](pretraining/predictivity_seeds/transfer_da_all_by_L_mono_axis.png)
<!-- END auto:panels -->

The two decision figures read the four mono-axis interventions of the
design-decisions analysis (pool `predictivity_seeds`, reference 1.7B): the
lines figure the list decision (scheme A vs B at L8–L30) alone, the by-L
figure every intervention at every L it has. Scheme B is DCLMP at L1, ZH at
L2 and the B lists at L8–L30; scheme C is FWEB at L1 and ES at L2.

**Key findings** (`transfer_da_all_lines_mono_axis`, `transfer_da_all_by_L_mono_axis`: the design-decisions per-item
agreement on the per-language BPB of every evaluation language, grouped
per (intervention, L) by what the two levels' lists do with the language —
both train it, only one does, neither does but one trains its script, or
neither trains even the script; the last two groups are the transfer test)

- **The list decision (A vs B) transfers only partly.** DA-size on languages
  whose script a list trains is 0.63 at 90M and 0.72–0.73 from 175M to 600M,
  clearing 0.75 only at 1B (0.81); on unseen scripts it is 0.42–0.73 and
  never clears 0.75.
- **The reference's own early checkpoints do better.** DA-ckpt on
  script-trained languages rises from 0.65 at 10 % of the run to 0.90 at
  90 %. On unseen scripts it is 0.50 at 10 % and 0.85 at 90 %, but not
  steadily (0.77 at 50 %, back to 0.63 at 80 %).
- **Per language count** (DA-size at the final checkpoint, mean over proxy
  sizes): script-trained languages read scheme A vs B at L1 (0.84), L2 (0.96),
  L8 (0.82) and L15 (0.77) but not L30 (0.58), scheme A vs C at L1 (0.98) and
  L2 (0.90), and temperature at L15–L50 (0.78–0.79). Depth is not read on
  untrained languages at any L (0.41–0.72).
- **Unseen scripts read only the L1 English-source swap.** Scheme A vs C at
  L1 (A vs FWEB, both levels train English alone) reads 0.99 on unseen
  scripts; every other (intervention, L) reads them at 0.43–0.71.
- The "trained languages read the decision at 1.0" reading is the languages
  only one list trains (1.00 wherever the group exists), where the model that saw
  the language wins; it is the inclusion decision, not transfer.
- Script is a coarse proxy for relatedness (Latin covers Welsh and Vietnamese
  alike), and at L1 only English is trained, so "script trained" is every
  other Latin-script language (50 at L1).

**Follow-ups**

- Group the untrained languages by the trained tokens of their language
  family or genus (from the data manifest), or by tokenizer overlap with the
  trained languages, and plot DA against that exposure instead of three bins.
- Scheme A vs C and temperature have two and three L's each (A vs C: FWEB at
  L1, ES at L2); fold them into a table. The "mean over proxy sizes" of `transfer_da_all_by_L_mono_axis` averages a proxy
  that flips with one that agrees into 0.5; one line per proxy size, or the
  largest proxy below the reference, for the paper version.
- The same caveats as the design-decisions analysis: items are correlated;
  bootstrap over L. The reference is 1.7B for every intervention.

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/highlights.csv) ·
GitHub: [transfer_error_by_L.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_error_by_L.png) · [transfer_error_by_L.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_error_by_L.csv) ·
GitHub: [transfer_da_all_lines_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis.png) · [transfer_da_all_lines_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_lines_mono_axis.csv) ·
GitHub: [transfer_da_all_by_L_mono_axis.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_by_L_mono_axis.png) · [transfer_da_all_by_L_mono_axis.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity_seeds/transfer_da_all_by_L_mono_axis.csv)

**The minimal language panel · DA-size against the 1.7B MACRO ranking
(mean score over the L8 panel languages above chance at 1.7B) · no filter ·
multi-axis pairs from `predictivity` at seed 1904 · gate
`predictivity`.** Which languages does a developer have to evaluate to
recover the multilingual decision: one language at a time, English alone, or
the proxy's own macro over the languages readable at its size.

Population: 14 design variants (91 pairs per benchmark), the eight L8
languages (en, ru, zh, de, ja, es, fr, it). The pooled row covers 15
benchmarks plus their 19 bBPB twins at 175M, 350M and 1B but only the 14–15
benchmarks at 90M and 600M (no twin there), so its populations differ by size.

<!-- BEGIN auto:language-panel (language_panel.py --pool predictivity) -->
## The minimal language panel

Per benchmark family, decision accuracy of a proxy size against the 1.7B MACRO ranking of the design variants (mean over the L8 panel languages above chance at 1.7B), when the proxy reads one language, English, or its own macro over the languages readable at that size. Grid-seed pairs of every data build (multi-axis), ≥ 3 pairs, gate `predictivity`. A macro above every single language says the languages' errors are independent and the panel is worth evaluating; a single language at the macro says it suffices. Regenerate with `python analysis/rq06_language_transfer/language_panel.py --pool predictivity`.

| benchmark | languages readable at 1.7B | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| arc | 7 | macro 0.55 / en 0.55 / best lang 0.55 (en) | macro 0.43 / en 0.43 / best lang 0.43 (en) | macro 0.52 / en 0.52 / best lang 0.52 (en) | macro 0.68 / en 0.68 / best lang 0.68 (en) | macro 0.58 / en 0.52 / best lang 0.64 (es) |
| bbpb_arc | 7 | — | macro 0.65 / en 0.55 / best lang 0.66 (it) | macro 1.00 / en 1.00 / best lang 1.00 (en) | — | macro 0.56 / en 0.51 / best lang 0.62 (de) |
| bbpb_belebele | 8 | — | macro 0.55 / en 0.51 / best lang 0.64 (ru) | macro 0.67 / en 0.33 / best lang 1.00 (zh) | — | macro 0.62 / en 0.49 / best lang 0.63 (zh) |
| bbpb_cultural_bench_easy | 5 | — | macro 0.54 / en 0.53 / best lang 0.54 (ru) | macro 0.33 / en 0.33 / best lang 0.33 (en) | — | macro 0.56 / en 0.52 / best lang 0.60 (zh) |
| bbpb_global_mmlu_full | 8 | — | macro 0.73 / en 0.68 / best lang 0.76 (ru) | macro 0.33 / en 0.33 / best lang 1.00 (ru) | — | macro 0.80 / en 0.70 / best lang 0.81 (fr) |
| bbpb_global_piqa_parallel_cloze | 8 | — | macro 0.82 / en 0.53 / best lang 0.81 (fr) | macro 0.33 / en 0.00 / best lang 1.00 (it) | — | macro 0.68 / en 0.46 / best lang 0.71 (es) |
| bbpb_hellaswag | 6 | — | macro 0.88 / en 0.78 / best lang 0.89 (es) | macro 0.67 / en 1.00 / best lang 1.00 (en) | — | macro 0.88 / en 0.78 / best lang 0.87 (es) |
| bbpb_include_base_44 | 7 | — | macro 0.65 / best lang 0.66 (fr) | macro 0.70 / best lang 0.70 (fr) | — | macro 0.64 / best lang 0.76 (ja) |
| bbpb_include_v2_en | 7 | — | macro 0.16 / best lang 0.41 (zh) | macro 0.60 / best lang 0.80 (de) | — | macro 0.57 / best lang 0.65 (ru) |
| bbpb_include_v2_og | 7 | — | macro 0.37 / best lang 0.52 (fr) | macro 0.30 / best lang 0.90 (ru) | — | macro 0.48 / best lang 0.63 (ru) |
| bbpb_multiblimp | 6 | — | macro 0.62 / en 0.36 / best lang 0.64 (fr) | macro 0.60 / en 0.33 / best lang 0.80 (fr) | — | macro 0.43 / en 0.42 / best lang 0.55 (ru) |
| bbpb_paws | 6 | — | macro 0.74 / en 0.53 / best lang 0.79 (zh) | macro 0.47 / en 0.20 / best lang 0.67 (de) | — | macro 0.64 / en 0.58 / best lang 0.74 (zh) |
| bbpb_rf_belebele | 8 | — | macro 0.62 / en 0.52 / best lang 0.68 (ja) | macro 0.68 / en 0.57 / best lang 0.79 (fr) | — | macro 0.64 / en 0.55 / best lang 0.65 (zh) |
| bbpb_rf_cultural_bench_easy | 5 | — | macro 0.38 / en 0.45 / best lang 0.45 (en) | macro 0.57 / en 0.68 / best lang 0.68 (en) | — | macro 0.64 / en 0.44 / best lang 0.69 (ru) |
| bbpb_rf_global_mmlu_full | 8 | — | macro 0.67 / en 0.52 / best lang 0.69 (zh) | macro 0.46 / en 0.46 / best lang 0.61 (ja) | — | macro 0.73 / en 0.62 / best lang 0.71 (ja) |
| bbpb_rf_include_base_44 | 7 | — | macro 0.77 / best lang 0.77 (es) | macro 0.36 / best lang 0.57 (de) | — | macro 0.66 / best lang 0.68 (zh) |
| bbpb_rfgm_belebele | 8 | — | macro 0.67 / en 0.58 / best lang 0.69 (it) | macro 0.72 / en 0.50 / best lang 0.78 (fr) | — | macro 0.70 / en 0.52 / best lang 0.79 (ru) |
| bbpb_rfgm_include_base_44 | 7 | — | macro 0.79 / best lang 0.82 (es) | macro 0.67 / best lang 0.75 (ja) | — | macro 0.85 / best lang 0.85 (es) |
| bbpb_xnli | 6 | — | macro 0.71 / en 0.57 / best lang 0.76 (ru) | macro 0.67 / en 0.36 / best lang 0.72 (fr) | — | macro 0.69 / en 0.57 / best lang 0.70 (ru) |
| bbpb_xstorycloze | 4 | — | macro 0.96 / en 0.67 / best lang 0.93 (es) | macro 0.72 / en 0.39 / best lang 0.83 (ru) | — | macro 0.92 / en 0.77 / best lang 0.91 (es) |
| hellaswag | 6 | macro 0.82 / en 0.67 / best lang 0.84 (fr) | macro 0.89 / en 0.85 / best lang 0.89 (de) | macro 0.88 / en 0.69 / best lang 0.92 (es) | macro 0.84 / en 0.69 / best lang 0.87 (fr) | macro 0.93 / en 0.87 / best lang 0.92 (es) |
| include_v2_en | 6 | macro 0.66 / best lang 0.64 (ja) | macro 0.53 / best lang 0.53 (fr) | macro 0.35 / best lang 0.66 (fr) | macro 0.55 / best lang 0.60 (ru) | macro 0.53 / best lang 0.67 (de) |
| include_v2_og | 6 | macro 0.51 / best lang 0.52 (ru) | macro 0.54 / best lang 0.56 (ru) | macro 0.57 / best lang 0.60 (ru) | macro 0.64 / best lang 0.74 (it) | macro 0.66 / best lang 0.68 (it) |
| lambada_openai_mt | 5 | macro 0.78 / en 0.59 / best lang 0.85 (de) | macro 0.81 / en 0.66 / best lang 0.84 (de) | macro 0.70 / en 0.59 / best lang 0.70 (es) | macro 0.78 / en 0.75 / best lang 0.84 (it) | macro 0.89 / en 0.78 / best lang 0.89 (de) |
| multiblimp | 6 | macro 0.71 / en 0.43 / best lang 0.74 (it) | macro 0.73 / en 0.36 / best lang 0.74 (es) | macro 0.80 / en 0.53 / best lang 0.79 (de) | macro 0.70 / en 0.58 / best lang 0.71 (ru) | macro 0.68 / en 0.40 / best lang 0.67 (de) |
| paws | 4 | — | macro 0.69 / en 0.69 / best lang 0.69 (en) | macro 0.69 / en 0.69 / best lang 0.69 (en) | macro 0.60 / en 0.60 / best lang 0.60 (en) | macro 0.54 / en 0.48 / best lang 0.56 (de) |
| rf_belebele | 8 | macro 0.66 / en 0.44 / best lang 0.65 (es) | macro 0.62 / en 0.64 / best lang 0.64 (en) | macro 0.57 / en 0.48 / best lang 0.63 (zh) | macro 0.55 / en 0.53 / best lang 0.66 (fr) | macro 0.62 / en 0.59 / best lang 0.64 (ru) |
| rf_global_mmlu_full | 8 | macro 0.47 / en 0.42 / best lang 0.55 (es) | macro 0.73 / en 0.54 / best lang 0.73 (de) | macro 0.71 / en 0.52 / best lang 0.77 (it) | macro 0.75 / en 0.55 / best lang 0.79 (ja) | macro 0.81 / en 0.69 / best lang 0.81 (de) |
| rf_include_base_44 | 6 | macro 0.80 / best lang 0.68 (ru) | macro 0.68 / best lang 0.74 (fr) | macro 0.70 / best lang 0.64 (es) | macro 0.86 / best lang 0.76 (es) | macro 0.67 / best lang 0.73 (ru) |
| rfgm_belebele | 8 | macro 0.82 / en 0.55 / best lang 0.75 (zh) | macro 0.78 / en 0.59 / best lang 0.74 (fr) | macro 0.69 / en 0.54 / best lang 0.68 (ru) | macro 0.60 / en 0.53 / best lang 0.79 (fr) | macro 0.86 / en 0.76 / best lang 0.80 (es) |
| rfgm_include_base_44 | 6 | macro 0.66 / best lang 0.64 (fr) | macro 0.73 / best lang 0.66 (zh) | macro 0.70 / best lang 0.70 (ru) | macro 0.82 / best lang 0.76 (fr) | macro 0.89 / best lang 0.75 (zh) |
| xnli | 6 | macro 0.54 / en 0.63 / best lang 0.63 (en) | macro 0.62 / en 0.36 / best lang 0.69 (fr) | macro 0.48 / en 0.44 / best lang 0.68 (de) | macro 0.55 / en 0.55 / best lang 0.67 (ru) | macro 0.33 / en 0.45 / best lang 0.54 (de) |
| xstorycloze | 4 | macro 0.74 / en 0.58 / best lang 0.69 (es) | macro 0.77 / en 0.67 / best lang 0.67 (en) | macro 0.90 / en 0.68 / best lang 0.80 (es) | macro 0.76 / en 0.68 / best lang 0.77 (es) | macro 0.77 / en 0.64 / best lang 0.75 (es) |
| xwinograd | 5 | macro 0.45 / en 0.59 / best lang 0.59 (en) | macro 0.42 / en 0.26 / best lang 0.49 (ja) | macro 0.67 / en 0.63 / best lang 0.63 (en) | macro 0.53 / en 0.42 / best lang 0.56 (ja) | macro 0.26 / en 0.48 / best lang 0.51 (zh) |
| all benchmarks | 8 | macro 0.66 / en 0.55 / one lang 0.55 | macro 0.65 / en 0.55 / one lang 0.55 | macro 0.65 / en 0.55 / one lang 0.55 | macro 0.68 / en 0.60 / one lang 0.60 | macro 0.67 / en 0.58 / one lang 0.58 |

![The minimal language panel](pretraining/predictivity/language_panel.png)
<!-- END auto:language-panel -->

**Key findings**

- **English alone is the weakest single-language proxy at 1B.** Pooled over
  the panel's benchmarks, English reads 0.58 against 0.61–0.66 for the other
  seven languages (zh 0.66, ru 0.66, de 0.64, es 0.62, fr 0.62, it 0.62,
  ja 0.61) and 0.67 for the panel macro; at 600M English (0.60) sits with
  Japanese (0.56) and Chinese (0.57) at the bottom, below the macro's 0.68.
- **English or Japanese is the lowest single language at every size.** They
  are the two lowest at 90M (ja 0.54, en 0.55), 175M (en 0.55, ja 0.57), 350M
  (ja 0.53, en 0.55) and 1B (en 0.58, ja 0.61). At 600M the bottom two are
  Japanese 0.56 and Chinese 0.57, with English third at 0.60.
- **Per benchmark the macro beats English where it matters.** At 1B the
  proxy's macro reaches 0.93 on hellaswag, 0.89 on LAMBADA and INCLUDE-rfgm
  and 0.81 on Global-MMLU-rf, where English alone reads 0.87, 0.78, — and
  0.69; on `multiblimp` English reads 0.40 against 0.68 for the macro.
- **Where the panel is worth evaluating.** The macro lies above every single
  language at 1B on 5 of the 15 benchmarks (hellaswag, multiblimp,
  Belebele-rfgm, INCLUDE-rfgm, xstorycloze) and ties the best one on
  Global-MMLU-rf and LAMBADA, so there the languages' errors are partly
  independent; INCLUDE-rf trails Russian 0.67 to 0.73, and on `xnli` (macro
  0.33) and `xwinograd` (0.26) nothing reads the reference above 0.54.
- **The macro is the safest proxy from 350M up.** Pooled, it reads 0.65,
  0.68 and 0.67 at 350M, 600M and 1B, above every single language, while
  Italian edges it at 90M (0.659 against 0.655, both 0.66 at two decimals)
  and German beats it at 175M (0.66 against 0.65).

**Follow-ups** (`plan/next_analyses.md` §6)

- Greedy k-language panels: which two or three languages recover the macro.
- The same against the macro over ALL trained languages per L (requires
  per-L pairs, rule 2).
- The seed null of the decision-accuracy analysis ([its seed-uncertainty figure](../rq02_decision_accuracy/README.md#7-seed-uncertainty-and-the-da-ckpt-null))
  as the floor of every panel.
- The pooled row once the bBPB twins exist at every size, so that 90M and
  600M pool the same 34 benchmarks as the other sizes.

GitHub: [language_panel.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity/language_panel.png) · [language_panel.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq06_language_transfer/pretraining/predictivity/language_panel.csv)

## Extensions from other sweeps

None. The language-transfer analysis exists on the ladder only (`predictivity_seeds`, `predictivity`):
the 36-model sweep evaluated 12 languages with no per-language BPB and no
never-trained language, so neither the leave-language-out fit nor the panel
question could be asked of it, and its numbers would not be pooled with the
ladder's in any case (a different harness, task set and reference size).

## Files

- `pretraining/<pool>/rq5_transfer.csv`, `rq5_transfer_summary.csv`,
  `rq5_transfer.png/.pdf` — the paper's language-transfer table and figure.
- `…/bpb_curves.png` — per-cell per-language BPB curves.
- `…/facts.json` — the numbers the paper quotes.
- `pretraining/predictivity/language_panel.{png,csv}` — the minimal language
  panel (`language_panel.py`).
