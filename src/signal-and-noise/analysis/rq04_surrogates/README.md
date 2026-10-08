# Surrogates: which cheap statistic predicts decision accuracy?

## Research question

> Before training the reference, can a statistic computed on the proxy alone
> tell us that the proxy's decision will match the reference's? First among
> the 22 SNR definitions of the noise-and-SNR analysis: which best correlates
> with **decision accuracy** across the ladder's languages, and does the
> answer survive a change of training seed?
>
> Then beyond SNR (the paper's
> surrogate analysis,
> [`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex)):
> its signal or noise part alone, the proxy's early-checkpoint agreement, how
> well its scores fit a scaling trend, its margin above chance. Finally, 211
> statistics from the literature and the AllenAI signal × noise grid
> (`catalogue.py`, `search.py`, [`literature.md`](literature.md)), per
> language, language count and benchmark.

<!-- BEGIN auto:highlight (snr_definition_postprocess.py --pool predictivity) -->
## Highlighted result

- **Global-best SNR definition (`predictivity`): `rel_std`** — mean Pearson r of log₁₀(SNR) vs decision accuracy **0.56** (DA-size, proxy → 1.7B, 50 languages), **0.58** (DA-ckpt, proxy sizes pooled, 50 languages), 0.57 overall. DA-ckpt is led by `rel_std`/`rel_mpd`/`rel_dispersion` (≈ 0.58; one family: rel_spread) — recommend the *family*, not an exact variant.
- **Per-language anchor: `bbpb_rfgm_include_base_44`** — the highest-SNR above-random benchmark in **12 of 50** languages (`rel_std` SNR @ 1.7B; `train_loss` and `bpb_macro` are not a language's and are left out); the language's own BPB, ungated and on its own noise scale, outranks that benchmark in 26 of the 50 languages that have both. Weakest variants overall: `projection`, `tukey`.
- **Seed holdout (predictivity_seeds_train → predictivity_seeds_test)**: Spearman ρ of the global variant ranking **0.70** (DA-ckpt), **0.55** (DA-size); family-level per-language agreement 100% / 100%. A ranking that does not survive the seed swap is noise-dominated — only the *family* recommendation transfers.
<!-- END auto:highlight -->

## Headlines (ladder report 2026-10-08 12:06)

*Pool `predictivity`: seed 1904, every cell (L ∈ {1, 2, 8, 15, 30, 50}; the
deep, shallow and swiglu ladders; every data build), sizes 90M–1.7B, the gate
per (task, size) (rule 1), parent tasks, trained languages. Per-language
Pearson r of log₁₀(SNR) with DA over the 50 languages with ≥ 3 tasks
(rule 8), then averaged.*

![The surrogate question in one figure](pretraining/predictivity/highlights.png)

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/highlights.csv)

**Key findings**

- **The relative-spread SNR definitions track decision accuracy best, as a
  block:** `rel_std`, `rel_mpd` and `rel_dispersion` read mean r 0.54–0.56
  with DA-size and 0.57–0.58 with DA-ckpt (0.56–0.57 overall,
  `top_variants_overall.csv`), ahead of `iqr` (0.53) and the eight
  dispersion and robust definitions (`aad`, `rms_deviation`, `dist_std`, `mpd`,
  `range`, `dispersion`, `quartile_deviation`, `mad`: 0.49–0.51 overall);
  the figure's all-task panels read 0.50–0.62 and 0.59–0.64 for their top
  six, both led by `rel_mpd`. The leader `rel_std` (0.56 / 0.58) wins by
  less than 0.01, and the depth definitions are null or anti-signals
  (`tukey` −0.01 / −0.11, `projection` −0.06 / −0.14).
- **No exact definition is best everywhere:** per language the winner is
  one of 11 definitions under DA-size and one of 12 under DA-ckpt. The
  relative-spread family wins 32 of 50 languages under both DA kinds;
  dispersion wins 13 under DA-ckpt and 7 under DA-size, discrepancy 6 under
  DA-size.
- **The seed holdout agrees under both DA kinds, on one or two
  languages:** the global ranking of the 22 definitions correlates across
  the seed split at Spearman ρ 0.70 under DA-ckpt and 0.55 under DA-size.
  The per-language agreement quoted in the highlight rests on English alone
  (DA-size, 1 of 1) and on English and Russian (DA-ckpt, 2 of 2).
- **Beyond SNR, the early-checkpoint agreement and SNR relative std share
  the lead, and on the accuracy tasks alone SNR leads:** over 1118–1286
  benchmark tasks (816 of them twins at every rung) the early agreement
  reads Spearman ρ 0.55–0.62 with DA-size and SNR relative std 0.54–0.61,
  the early agreement ahead at 90M, 175M and 1B. On the 302–470 accuracy
  tasks alone SNR relative std reads 0.35–0.43, against 0.23–0.40 for the
  early agreement and 0.19–0.40 for the inverted noise; margin above chance
  (0.01 to 0.11) and signal alone (−0.09 to 0.05) carry nothing on the
  benchmark tasks.
- **The proxy's own rank stability beats every SNR in the catalogue:** over
  420 (benchmark, language) clusters, the window sign consistency reads ρ
  0.93 against DA-ckpt, consecutive-checkpoint Kendall τ 0.86 against
  DA-goal and the proxy's mean DA-ckpt 0.83 against DA-size (ceilings 0.99 /
  0.95 / 0.92). AllenAI's relative-std / checkpoint-noise SNR reads
  0.79–0.85.
- **FineTasks' criteria separate reliable tasks only modestly:** on the 846
  accuracy benchmark tasks the three together pass 16 % (90M) to 44 % (1B),
  against 37–56 % for our gate. Their passers read DA-size 0.56–0.58
  against 0.46–0.50 for the rest.

## Experimental setup

- **Pools** (outputs under `pretraining/<pool>/`; [RULES.md](../RULES.md),
  Definitions). `predictivity` is the headline: seed 1904, every cell of the
  grid (L ∈ {1, 2, 8, 15, 30, 50}; deep, shallow and swiglu; every data
  build), 90M–1.7B; `predictivity_seeds` adds every replicate seed as a
  separate model of the signal pool.
- **One data-scheme axis.** A data build is read as scheme A/B/C ×
  temperature T, never by its label: AT3 is scheme A at T = 3, ZH and DCLMP
  are scheme B, ES and FWEB scheme C.
- **Seed holdout.** `predictivity_seeds_train` (replicate seeds 64/313 at
  the 175M/600M ×3 cells, 28/1797 at the 1B ×3 cells) →
  `predictivity_seeds_test` (seed 1904 on exactly those cells). Among the ×3
  cells (L ∈ {1, 2, 50} at 175M/600M, L ∈ {1, 2, 30} at 1B) a non-English
  harness task is trained only at L30/L50 (Russian also at L2), so only
  English (DA-size) and English and Russian (DA-ckpt) have a best
  definition on both splits.
- **Signal, noise, DA.** The signal at a size is the spread over every design
  variant trained there; the noise is the late-checkpoint noise over the noise
  window, 80/85/90/95/100 % of the run: the residual std around a line through
  it since 2026-10-08 (rule 4). DA-size is proxy final →
  1.7B final (rule 9); DA-ckpt is a proxy's early checkpoints → its final,
  the proxy sizes pooled and never the 1.7B run's own checkpoints (rule 11).
- **Tasks.** Benchmarks, their `rf_` twins and the per-language BPB
  (`bpb_<subset>`) take part like any task; `bpb_macro` and `train_loss` are
  not a language's and are left out (rule 7). The 22 definitions are grouped
  into families (dispersion / relative spread / discrepancy / robust /
  depth).
- **bBPB twins (populations move, rule 13).** The per-item store was rebuilt
  over every checkpoint of every seed-1904 ladder cell, so the 816 `bbpb_*`
  twins have an SNR, a DA-size and a DA-ckpt at every rung from 90M to 1.7B,
  and every pooled or per-proxy number below mixes them in. The seed
  replicates (seeds 64/313/28/1797, 20 models) and the five 3B cells still
  have no twin, so the seed holdout holds no bBPB row.

## Methodology

- **SNR per (task, size bucket)** is the noise-and-SNR table
  (`../rq03_noise_and_snr/pretraining/<pool>/snr_variants_per_task.csv`:
  22 variants, gated cells NaN, coverage per variant recorded there).
- **Decision accuracy** is joined from the decision-accuracy tables (`decision_acc_size_*`,
  `decision_acc_ckpt_*`; pair counts in `da_all_n_pairs_per_task_both_axes.csv`).
- **Ranking.** Per language, Pearson r of log₁₀(SNR) against DA over the
  (task, bucket) cells; the global variant is the highest mean r over
  languages and over both DA kinds (`top_variants_overall.csv`), and the
  per-language "most reliable benchmark" table reads that variant at the
  reference size.
- **Seed holdout.** The same per-language table on the replicate seeds
  (64/313/28/1797) and on seed 1904 of the same cells
  (`../rq03_noise_and_snr/pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/headline_metrics.csv`);
  agreement is counted over the languages that have a best variant on both
  splits, today 1 (DA-size) and 2 (DA-ckpt).

Hand-written numbers in this README are from the ladder-report snapshot
**2026-10-08 12:06** (regenerated locally: acc_norm on the cloze-format
originals) and the outputs of the 2026-10-08 refresh (detrended checkpoint
noise), prose re-read 2026-10-08.

<!-- BEGIN auto:results (snr_definition_postprocess.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq04_surrogates/snr_definition_postprocess.py --pool predictivity`.

**Global variant ranking** — mean Pearson r of log₁₀(SNR) vs DA across the trained languages with ≥ 3 tasks (rule 8; 50 languages under DA-size, 50 under DA-ckpt). DA-size = proxy final → 1.7B final only (the proxy-to-proxy scaling pairs are not DA-size, rule 9); DA-ckpt = a proxy size's early checkpoints → its final, the proxy sizes 90M, 175M, 350M, 600M, 1B pooled, never the 1.7B run's own checkpoints (rule 11):

| variant | DA-size r | DA-ckpt r | overall |
|---|---|---|---|
| `rel_std` | 0.56 | 0.58 | 0.57 |
| `rel_mpd` | 0.55 | 0.58 | 0.57 |
| `rel_dispersion` | 0.54 | 0.57 | 0.56 |
| `iqr` | 0.52 | 0.54 | 0.53 |
| `aad` | 0.47 | 0.56 | 0.51 |
| `rms_deviation` | 0.47 | 0.55 | 0.51 |
| `dist_std` | 0.47 | 0.55 | 0.51 |
| … |  |  |  |
| `tukey` | -0.01 | -0.11 | -0.06 |
| `projection` | -0.06 | -0.14 | -0.10 |

![SNR variants ranked by correlation with DA](pretraining/predictivity/top_variants_overall.png)

**Statistical power by pool** — each pool's best variant (mean r over both DA kinds):

| pool | best variant (overall) | DA-size r | DA-ckpt r |
|---|---|---|---|
| `predictivity` (grid, seed 1904, every design) | `rel_std` | 0.56 | 0.58 |
| `predictivity_seeds` (every seed) | `rel_std` | 0.56 | 0.58 |

**Most reliable benchmark per language** — `rel_std` SNR @ 1.7B over the above-random benchmarks, with the language's own BPB SNR alongside (ungated, on its own noise scale; DA-size is undefined at the reference size itself, so DA-ckpt@1.7B is shown; `train_loss` and `bpb_macro` measure the whole mixture and are not a row, rule 7):

| lang | top benchmark | SNR | DA-ckpt@1.7B | BPB SNR |
|---|---|---|---|---|
| ar | `bbpb_rfgm_belebele_arz_Arab` | 5.14 | 0.82 | 5.37 |
| az | `bbpb_rfgm_include_base_44_azerbaijani` | 10.98 | 0.93 | 7.67 |
| bg | `bbpb_rfgm_include_base_44_bulgarian` | 4.48 | 0.86 | 5.92 |
| bn | `bbpb_rfgm_include_base_44_bengali` | 5.75 | 0.89 | 7.69 |
| bs | `bbpb_global_piqa_nonparallel_cloze_bos_latn` | 6.00 | 0.87 | 8.56 |
| ca | `bbpb_hellaswag_ca` | 7.67 | 0.98 | 7.24 |
| cs | `bbpb_multiblimp_ces` | 3.38 | 0.75 | 2.37 |
| da | `multiblimp_dan` | 5.70 | 0.48 | 4.51 |
| de | `bbpb_hellaswag_de` | 6.62 | 0.89 | 3.91 |
| el | `bbpb_multiblimp_ell` | 2.76 | 0.72 | 2.55 |
| en | `hellaswag` | 5.96 | 0.90 | 17.08 |
| es | `include_v2_og_spanish_espa_a` | 8.44 | 0.74 | 8.33 |
| et | `bbpb_xcopa_et` | 12.99 | 0.91 | 8.93 |
| fa | `bbpb_rfgm_include_base_44_persian` | 6.87 | 0.87 | 5.22 |
| fi | `bbpb_rfgm_include_base_44_finnish` | 4.37 | 0.87 | 5.51 |
| fr | `bbpb_hellaswag_fr` | 6.06 | 0.93 | 3.26 |
| he | `bbpb_rfgm_include_base_44_hebrew` | 6.78 | 0.91 | 6.94 |
| hi | `bbpb_hellaswag_hi` | 5.80 | 0.91 | 7.32 |
| hr | `bbpb_rfgm_include_base_44_croatian` | 8.68 | 0.94 | 7.02 |
| hu | `bbpb_multiblimp_hun` | 4.16 | 0.83 | 3.41 |
| id | `bbpb_rfgm_include_base_44_indonesian` | 4.36 | 0.74 | 3.74 |
| it | `hellaswag_it` | 4.60 | 0.85 | 3.30 |
| ja | `bbpb_rf_global_mmlu_full_ja` | 4.56 | 0.73 | 5.07 |
| ka | `bbpb_rfgm_belebele_kat_Geor` | 6.19 | 0.88 | 7.51 |
| kk | `bbpb_global_piqa_nonparallel_cloze_kaz_cyrl` | 9.03 | 0.98 | 9.52 |
| ko | `bbpb_paws_ko` | 3.08 | 0.79 | 3.71 |
| lt | `bbpb_rfgm_include_base_44_lithuanian` | 6.43 | 0.89 | 6.90 |
| lv | `bbpb_rfgm_belebele_lvs_Latn` | 8.94 | 0.98 | 8.46 |
| ml | `bbpb_global_piqa_parallel_cloze_mal_mlym` | 6.06 | 0.89 | 7.76 |
| mr | `bbpb_rfgm_belebele_mar_Deva` | 6.49 | 1.00 | 10.31 |
| ms | `bbpb_include_v2_og_malay_malaysia` | 7.73 | 0.93 | 7.32 |
| ne | `bbpb_hellaswag_ne` | 6.85 | 0.85 | 10.88 |
| nl | `bbpb_hellaswag_nl` | 4.18 | 0.91 | 3.83 |
| no | `global_piqa_parallel_cloze_nob_latn` | 2.42 | 0.45 | 4.24 |
| pl | `bbpb_rf_global_mmlu_full_pl` | 3.24 | 0.66 | 3.21 |
| pt | `hellaswag_pt` | 4.27 | 0.92 | 3.19 |
| ro | `bbpb_multiblimp_ron` | 3.95 | 0.70 | 3.69 |
| ru | `bbpb_hellaswag_ru` | 7.74 | 0.91 | 9.55 |
| sk | `bbpb_hellaswag_sk` | 4.19 | 0.96 | 5.67 |
| sl | `bbpb_multiblimp_slv` | 7.43 | 0.91 | 8.81 |
| sq | `bbpb_rfgm_include_base_44_albanian` | 10.10 | 0.91 | 9.27 |
| sr | `bbpb_hellaswag_sr` | 8.22 | 0.96 | 9.55 |
| sv | `bbpb_hellaswag_sv` | 4.93 | 0.95 | 4.97 |
| ta | `bbpb_rfgm_include_base_44_tamil` | 5.39 | 0.93 | 7.23 |
| th | `bbpb_xnli_th` | 3.06 | 0.76 | 3.77 |
| tr | `bbpb_rf_include_base_44_turkish` | 5.48 | 0.74 | 3.30 |
| uk | `bbpb_multiblimp_ukr` | 3.11 | 0.76 | 1.95 |
| ur | `bbpb_rfgm_belebele_urd_Arab` | 7.26 | 0.98 | 10.14 |
| vi | `bbpb_rfgm_include_base_44_vietnamese` | 5.75 | 0.83 | 5.62 |
| zh | `bbpb_xstorycloze_zh` | 7.01 | 0.92 | 8.31 |

![Top-5 benchmarks per language by SNR](pretraining/predictivity/top_benchmarks_per_language.png)

**Seed generalization** — holdout `predictivity_seeds_train` → `predictivity_seeds_test` (the ×3 cells only). A variant ranking whose Spearman ρ is low here is noise-dominated; recommend the family that transfers, not the argmax:

| metric | DA-size | DA-ckpt |
|---|---|---|
| Spearman ρ on global variant ranking | 0.55 | 0.70 |
| Pearson r between splits (all cells) | 0.72 | 0.34 |
| Exact-variant agreement (per lang) | 0% | 0% |
| Family-level agreement (per lang) | 100% | 100% |
| Retention of train-best r on test | 85% | 97% |
<!-- END auto:results -->

GitHub: [top_variants_overall.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_variants_overall.png) · [top_variants_overall.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_variants_overall.csv) ·
GitHub: [top_benchmarks_per_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_benchmarks_per_language.png) · [top_benchmarks_per_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_benchmarks_per_language.csv) ·
[snr_variant_ranking.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_ranking.csv) ·
[best_variant_per_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/best_variant_per_language.csv)

**Key findings** (pool `predictivity`, 90M–1.7B, gated, 50 languages under
each DA kind; the holdout rows on the ×3 cells only)

- **A block, not a winner:** the three relative-spread definitions span
  0.56–0.57 overall, and `rel_std` leads `rel_mpd` by 0.002. With every
  replicate seed in the signal pool (`predictivity_seeds`) the leader stays
  `rel_std`, at 0.56 / 0.58, with `rel_mpd` second by 0.006.
- **The most reliable benchmark per language is a bBPB twin:** a `bbpb_*`
  task has the highest `rel_std` SNR at 1.7B in 44 of 50 languages, and an
  accuracy task in six (Danish MultiBLiMP, English, Italian and Portuguese
  HellaSwag, Spanish INCLUDE, Norwegian Global-PIQA); in 26 languages the
  language's own BPB ranks above every benchmark. INCLUDE (either version)
  and HellaSwag, in either scoring, are the top benchmark in 15 and 13
  languages and MultiBLiMP in 7, and the twins now carry every checkpoint, so their rows are no
  longer provisional.
- **The seed holdout is too thin to read:** the per-language agreement
  (100 % / 100 %) and the 85 % / 97 % retention are over 1 language
  (DA-size) and 2 (DA-ckpt). The cell-level Pearson r between the splits,
  0.72 over 22 (definition, language) cells and 0.34 over 44, carries the
  same caveat.

**Follow-ups**

- The top-benchmark table split by scoring (accuracy vs bBPB), so the
  accuracy anchor per language is visible while the twins take 44 of 50 rows.
- A bootstrap over languages for the mean r of the top three definitions:
  they sit within 0.02 of each other, so only the family can be claimed.
- The language count beside every holdout percentage in the generated
  block (see the TODO below): the two 100 % are 1 of 1 and 2 of 2.

### Statistics beyond SNR — setup

The truth is DA-size from the [decision-accuracy tables](../rq02_decision_accuracy/):
per task, the agreement between the ranking of the design variants at a proxy
size and at the 1.7B reference, as carried in the noise-and-SNR per-task table
(`snr_variants_per_task.csv`, `decision_acc_size_<proxy>`). The candidates are
read from the same table (`snr_*`, `signal_*`, `noise_*`, `decision_acc_ckpt_*`),
from the gate's above-random scores (margin above chance) and from the
scaling fits (log-N R², refitted on the proxy rungs only).

The population per
proxy size is the tasks with an SNR at that size (the above-random survivors),
benchmarks and per-language BPB scored separately. Spearman ρ between the
candidate and DA-size per proxy size (90M–1B), with the p-value and n; a cell
needs at least 8 tasks, and the noise part is inverted so that "higher is
better" holds for every candidate.

Population: pool `predictivity`. The benchmark n carries the bBPB twins
(rule 13): 1118, 1158, 1193, 1245 and 1286 tasks at 90M–1B, of which 816 at
every proxy are twins and 302–470 accuracy tasks; the BPB rows are the 50
trained languages at every proxy.

<!-- BEGIN auto:surrogates (analyze.py --pool predictivity) -->
## Statistics beyond SNR (paper RQ3)

Numbers from the `predictivity` pool's rq03 table. Regenerate with `python analysis/rq04_surrogates/analyze.py --pool predictivity`. The population at a proxy size is the tasks above chance at the proxy and at 1.7B (rule 1); each candidate is scored on the tasks of it where the candidate has a value, so n differs per candidate and is given next to every ρ. The scaling-fit R² is rq01's log-N fit refitted on the rungs up to the proxy only (rule 11; it needs 3 rungs, so it starts at 350M). `bpb_macro` and `train_loss` are in neither population (rule 7).

- **benchmark tasks** — strongest surrogate of DA-size (mean ρ over proxies): `early-checkpoint agreement (10 %)` 0.58; weakest: `signal alone (relative std)` -0.01.
- **per-language bits per byte** — strongest surrogate of DA-size (mean ρ over proxies): `SNR, dist_std` 0.31; weakest: `scaling-fit R² (proxy rungs only)` 0.00.

**benchmark tasks** (Spearman ρ of the statistic with DA-size, per proxy size; n = the tasks behind the ρ):

| metric | 90M ρ (n) | 175M ρ (n) | 350M ρ (n) | 600M ρ (n) | 1B ρ (n) |
|---|---|---|---|---|---|
| early-checkpoint agreement (10 %) | 0.59 (1118) | 0.62 (1158) | 0.58 (1193) | 0.55 (1245) | 0.57 (1286) |
| SNR, relative std | 0.56 (1118) | 0.55 (1158) | 0.61 (1193) | 0.57 (1245) | 0.54 (1286) |
| noise alone (relative std, inverted) | 0.46 (1118) | 0.44 (1158) | 0.56 (1193) | 0.43 (1245) | 0.49 (1286) |
| scaling-fit R² (proxy rungs only) |  |  | 0.46 (1115) | 0.42 (1156) | 0.46 (1194) |
| SNR, dist_std | 0.46 (1118) | 0.38 (1158) | 0.47 (1193) | 0.46 (1245) | 0.41 (1286) |
| SNR, discrepancy | 0.20 (302) | 0.28 (342) | 0.33 (377) | 0.31 (429) | 0.38 (470) |
| margin above chance | 0.11 (297) | 0.02 (337) | 0.06 (372) | 0.08 (424) | 0.01 (465) |
| signal alone (relative std) | 0.05 (1118) | 0.03 (1158) | -0.02 (1193) | 0.00 (1245) | -0.09 (1286) |

**per-language bits per byte** (Spearman ρ of the statistic with DA-size, per proxy size; n = the tasks behind the ρ):

| metric | 90M ρ (n) | 175M ρ (n) | 350M ρ (n) | 600M ρ (n) | 1B ρ (n) |
|---|---|---|---|---|---|
| SNR, dist_std | 0.49 (50) | 0.55 (50) | 0.52 (50) | -0.46 (50) | 0.45 (50) |
| SNR, relative std | 0.30 (50) | 0.45 (50) | 0.29 (50) | -0.42 (50) | 0.23 (50) |
| signal alone (relative std) | 0.32 (50) | 0.40 (50) | 0.28 (50) | -0.41 (50) | 0.25 (50) |
| noise alone (relative std, inverted) | 0.04 (50) | 0.00 (50) | -0.08 (50) | 0.37 (50) | -0.11 (50) |
| early-checkpoint agreement (10 %) | 0.48 (50) | 0.41 (50) | 0.24 (50) | -0.49 (50) | -0.52 (50) |
| scaling-fit R² (proxy rungs only) |  |  | 0.31 (50) | -0.64 (50) | 0.34 (50) |

![Surrogates](pretraining/predictivity/rq3_surrogates.png)
<!-- END auto:surrogates -->

GitHub: [rq3_surrogates.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/rq3_surrogates.png) · [rq3_surrogates.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/rq3_surrogates.csv) ·
[facts.json](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/facts.json)

**Key findings** (pool `predictivity`, DA-size against 1.7B, gated; the
benchmark n per cell in the table; the accuracy-only and twin-only ρ are
recomputed by hand from the noise-and-SNR per-task table with this
script's gate, without the `bbpb_` rows or with them alone)

- **With the twins, the early-checkpoint agreement and SNR relative std are
  level:** over 1118–1286 tasks they read ρ 0.55–0.62 (mean 0.58) and
  0.54–0.61 (mean 0.57), the early agreement ahead at 90M, 175M and 1B and
  SNR relative std at 350M and 600M, with the scaling-fit R² at 0.42–0.46
  from 350M. On the 816 twins alone the early agreement reads 0.57–0.70 and
  SNR relative std 0.48–0.61.
- **On the accuracy tasks alone SNR relative std leads at four of the five
  proxies:** over 302–470 tasks it reads 0.35–0.43, the early-checkpoint
  agreement 0.23–0.40 and the inverted noise 0.19–0.40. Only at 1B does the
  early agreement (0.40) edge ahead, level with the noise (0.40).
- **On the benchmark tasks, margin above chance and signal alone predict
  nothing:** ρ 0.01 to 0.11 (297–465 accuracy tasks) and −0.09 to 0.05 at
  every proxy. There the SNR's predictive part is its noise, which reads
  0.43–0.56 inverted.
- **Per-language BPB flips sign at 600M:** SNR `dist_std` reads 0.45–0.55 at
  90M, 175M, 350M and 1B but −0.46 at 600M (50 languages). SNR relative
  std, signal alone and the scaling-fit R² flip there too, and the inverted
  noise rises from −0.11–0.04 to 0.37.

**Follow-ups**

- The benchmark rows split by scoring (accuracy vs bBPB) in the generated
  table, so the accuracy-only reading above comes from the pipeline rather
  than by hand.
- The 600M BPB row: which cells and languages drive the sign flip, and
  whether the 600M DA-size against 1.7B is itself an outlier in the
  decision-accuracy tables.

### FineTasks' criteria — setup

A selection rule computed on one size, with no larger model in sight, is the
practical alternative to SNR: FineTasks (Kydlíček, Penedo et al., 2024)
selects tasks per language with four such statistics.

`finetasks_criteria.py`
(moved here from the benchmark-design folder on 2026-09-23; outputs
`pretraining/predictivity/finetasks_*`) computes them per (task, size) on the
`predictivity` pool: every design variant at each size at seed 1904 (every L,
ladder and data build; up to 32 runs per task and size), on the ten evaluated
tenths, which every task row now has. It judges them by DA-size against the
1.7B reference, which the criteria never see.

The gate is not applied (non-random is one of the criteria under test) but
the mask is carried, and every candidate is scored as Spearman ρ with DA-size
across tasks, the way the SNR definitions above are. Their "noise" is the
spread across recipes at one step, which this framework calls signal, so
their SNR is the inverse of ours and the two should not be expected to agree.

Population: 846 accuracy benchmark tasks plus the 816 bBPB twins at every
size (1662 rows; rule 13). The twins pass neither monotonicity nor
non-random by construction (lower is better, no chance level), so they
lower the generated table's pass shares at every size and sit in its
"fail" column.

<!-- BEGIN auto:finetasks-criteria (finetasks_criteria.py --pool predictivity) -->
## FineTasks' criteria on the ladder

FineTasks selects tasks with four statistics computed on single-seed runs at one size (monotonicity, a cross-run SNR, a non-random margin, consecutive-step ordering). Computed here per (task, size) on the `predictivity` variants and judged by DA-size against 1.7B, which the criteria never see. Columns: share passing each criterion, all three, and our gate; mean DA-size of passers vs failers; Spearman of each statistic with DA-size (monotonicity / SNR / non-random / ordering). Regenerate with `python analysis/rq04_surrogates/finetasks_criteria.py --pool predictivity`.

| size | tasks | monotone | SNR > 20 | non-random | all three | our gate | DA-size pass vs fail | ρ with DA-size |
|---|---|---|---|---|---|---|---|---|
| 90M | 1662 | 9% | 63% | 24% | 8% | 37% | 0.57 [135] vs 0.57 [1527] | -0.49 / +0.11 / +0.12 / +0.61 |
| 175M | 1662 | 13% | 62% | 25% | 11% | 40% | 0.57 [187] vs 0.56 [1475] | -0.37 / +0.13 / +0.20 / +0.50 |
| 350M | 1662 | 18% | 64% | 26% | 16% | 45% | 0.56 [258] vs 0.57 [1404] | -0.45 / +0.22 / +0.24 / +0.57 |
| 600M | 1662 | 22% | 62% | 28% | 19% | 50% | 0.56 [316] vs 0.58 [1346] | -0.37 / +0.17 / +0.22 / +0.52 |
| 1B | 1662 | 26% | 62% | 31% | 23% | 56% | 0.58 [376] vs 0.59 [1286] | -0.33 / +0.26 / +0.25 / +0.54 |
| 1.7B | 1662 | 28% | 62% | 33% | 24% | 60% | — | +nan / +nan / +nan / +nan |

![FineTasks criteria](pretraining/predictivity/finetasks_criteria.png)

![DA against each surrogate](pretraining/predictivity/finetasks_surrogates_scatter.png)
<!-- END auto:finetasks-criteria -->

*`finetasks_criteria.png`: (a) the share of tasks passing each criterion and
all three, with our gate; (b) each criterion's Spearman correlation with
DA-size across tasks; (c) DA-size of the tasks passing all three against the
rest; (d) panel (a) with the share of tasks clearing the two decision-accuracy
cuts at that size (DA-size against the reference ≥ 0.66; median DA-ckpt of
the size's own run ≥ 0.66); (e) panel (b) with the 22 SNR definitions behind
it and, for scale, the proxy ranking's own ρ and τ_b against the reference's
(an oracle) and the scaling analysis's ρ of score with size; (f) the
FineTasks picks with a counterpart in our registry, originals and cloze twins.
`finetasks_surrogates_scatter.png`: DA-size against each surrogate, one point
per (task, proxy size) coloured by the proxy — FineTasks' four criteria, our
gate, the ρ of score with size, the three SNR definitions with the largest
|Spearman| here, the proxy ranking's own ρ (the ceiling), and ten further
reference-free candidates (ordering in the noise window, the 10 % and 50 %
rankings against the final, the proxy's own DA-ckpt, DA between the two rungs
below the proxy, relative signal and noise, cross-variant std, margin over
chance, item count); Pearson r, Spearman ρ and the cell count in each corner.
Pooled ranking in `finetasks_surrogates_rank.csv`, the FineTasks picks in
`finetasks_overlap.csv` (folder root, written 2026-09-23).*

**Key findings** (pool `predictivity`, DA-size against 1.7B, ungated; the
accuracy-only numbers are read from `finetasks_criteria.csv` without the
`bbpb_` rows)

- **The composite selects better-than-average tasks but not reliable ones:**
  on the 846 accuracy tasks, monotonicity ≥ 0.5, cross-run SNR > 20 and best
  margin > 3 std together pass 16 / 22 / 30 / 37 / 44 % of tasks at
  90M / 175M / 350M / 600M / 1B, below our gate's 37 / 40 / 45 / 50 / 56 %.
  Their passers read DA-size 0.56–0.58 against 0.46–0.50 for the rest.
- **Their SNR criterion does not discriminate, monotonicity does the work:**
  the SNR > 20 cut passes 61–63 % of the accuracy tasks at every proxy, while
  monotonicity rises from 18 % (90M) to 50 % (1B).
- **The four statistics correlate weakly with DA-size on the accuracy
  tasks:** Spearman 0.12–0.41 per proxy (0.12–0.24 at 90M, 0.25–0.41 at 1B).
  The negative monotonicity of the generated table at every size (−0.33 to
  −0.49) is the bBPB twins (−0.51 to −0.63 on them alone): pooled over the
  proxies on the accuracy tasks alone it reads +0.31.
- **Pooled, the proxy's own run is the best reference-free signal:**
  `finetasks_surrogates_rank.csv` (twins included) puts the proxy's median
  DA-ckpt at 0.61, the DA between the two rungs below at 0.56, its own
  trajectory R² at 0.56 and its DA-ckpt at 50 % of the run at 0.55, against
  at most 0.53 for an SNR definition (`rel_mpd`), 0.34 for the item count,
  0.22 for our gate and 0.18 for FineTasks' SNR. On the accuracy tasks alone
  the median DA-ckpt drops to 0.38, level with SNR `rel_mpd` (0.35), the
  item count (0.34) and the DA between the two rungs below (0.32).
- **Spread across variants is not agreement with the reference:** the
  relative signal reads −0.01 and the cross-variant std 0.14 pooled, while
  the relative noise reads −0.45. The statistics that see the reference
  remain far above (the proxy ranking's own τ_b 0.99, the full-trajectory R²
  0.55).
- **The format FineTasks chose matters more than its thresholds:** of their
  96 picks, 55 exist in our registry and 35 have results at 1B, where 54 % of
  the originals pass their criteria on our ladder against all 13 cloze twins.
  The twins also read the higher mean DA-size (0.63 against 0.60).

**Follow-ups**

- Bootstrap the ρ over tasks (a 90 % interval per statistic): on the
  accuracy tasks the DA-ckpt median, the lower-rung DA and the item count sit
  within 0.05 of each other, so claim the family ("persistence of the
  proxy's own ranking"), not a winner.
- A per-size and per-scoring version of the pooled ranking (accuracy vs
  bBPB), as the `_b` panels do for the SNR definitions: the pooled Spearman
  mixes the twins, whose monotonicity and DA-ckpt behave differently, with
  the accuracy tasks.
- The cross-variant spread against DA-size per family, to see whether the
  `rf_` twins (large spread, re-sorting with scale;
  [the gate's figure 3](../rq00_gate_and_curves/README.md#3-the-reformulated-twins-move-whole-families-across-the-gate))
  explain why spread at the proxy does not mean agreement with the reference.

GitHub: [finetasks_criteria.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_criteria.png) · [finetasks_criteria.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_criteria.csv) ·
GitHub: [finetasks_surrogates_scatter.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_surrogates_scatter.png) · [finetasks_surrogates_scatter.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_surrogates_scatter.csv) ·
[finetasks_surrogates_rank.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_surrogates_rank.csv) ·
[finetasks_overlap.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/finetasks_overlap.csv)

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

The rankings above, without the aggregation (`predictivity` pool); a surrogate subplot needs 8 tasks at a proxy size. Regenerate with `python analysis/rq04_surrogates/panels.py --pool predictivity`. In every grid white is "no value" and grey "filtered out by the gate"; each figure's table sits next to it under the same name.

![rq04 in one figure](pretraining/predictivity/highlights.png)

![SNR definition per language](pretraining/predictivity/snr_definition_by_language.png)

![Surrogates per benchmark](pretraining/predictivity/surrogates_by_benchmark.png)

![Surrogates per language](pretraining/predictivity/surrogates_by_language.png)

**Per language count** (rq02's `da_all_by_L_per_task_multi_axes.csv`: pairs of design variants sharing the L, on the tasks in the languages every variant at the L trains on (the intersection of the L's lists, English always); a level counts when it holds at every larger level with a value; DA ≥ 0.75, an SNR definition tracks DA at ρ ≥ 0.3; an L with fewer than 3 pairs against 1.7B is named in the figure's caption):

![Smallest safe level per measurement and L](pretraining/predictivity/min_level_by_L.png)

![the same as lines](pretraining/predictivity/min_level_by_L_lines.png)

![Smallest size at which an SNR definition tracks DA, per L](pretraining/predictivity/snr_variant_min_size_by_L.png)

![the same as lines](pretraining/predictivity/snr_variant_min_size_by_L_lines.png)

**Version B — every pair pooled, the size axis instead of the language count** (`da_all_pooled_per_task_multi_axes.csv`, ten checkpoints; the population is every design-variant pair of the pool on the tasks of the languages each cell trains (rule 2), parent tasks only (rule 6), gated at the proxy and at 1.7B; the benchmark mean's task count per size differs with the gate (rule 13) and is in each figure's caption and on the DA-at-1C panel):

![Earliest checkpoint per proxy size and DA at 1C](pretraining/predictivity/min_level_by_L_b.png)

![the same as lines](pretraining/predictivity/min_level_by_L_lines_b.png)

![Spearman rho of each SNR definition with DA per proxy size](pretraining/predictivity/snr_variant_min_size_by_L_b.png)

![the same as lines](pretraining/predictivity/snr_variant_min_size_by_L_lines_b.png)

**Version per FLOPs** — every (proxy size, checkpoint) cell at its training compute, the same population as version B:

![DA of every cell against compute](pretraining/predictivity/min_level_by_L_flops.png)

![rho of each SNR definition against compute](pretraining/predictivity/snr_variant_min_size_by_L_flops.png)
<!-- END auto:panels -->

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/highlights.csv) ·
[snr_definition_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_definition_by_language.png) ·
[snr_definition.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_definition.csv) ·
[surrogates_by_benchmark.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/surrogates_by_benchmark.png) ·
[surrogates_by_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/surrogates_by_language.png) ·
[surrogates.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/surrogates.csv) ·
GitHub: [min_level_by_L.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L.png) · [min_level_by_L.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L.csv) ·
GitHub: [min_level_by_L_lines.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_lines.png) · [min_level_by_L_lines.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_lines.csv) ·
GitHub: [snr_variant_min_size_by_L.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L.png) · [snr_variant_min_size_by_L.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L.csv) ·
GitHub: [snr_variant_min_size_by_L_lines.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_lines.png) · [snr_variant_min_size_by_L_lines.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_lines.csv) ·
GitHub: [min_level_by_L_b.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_b.png) · [min_level_by_L_b.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_b.csv) ·
GitHub: [min_level_by_L_lines_b.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_lines_b.png) · [min_level_by_L_lines_b.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_lines_b.csv) ·
GitHub: [snr_variant_min_size_by_L_b.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_b.png) · [snr_variant_min_size_by_L_b.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_b.csv) ·
GitHub: [snr_variant_min_size_by_L_lines_b.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_lines_b.png) · [snr_variant_min_size_by_L_lines_b.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_lines_b.csv) ·
GitHub: [min_level_by_L_flops.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_flops.png) · [min_level_by_L_flops.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/min_level_by_L_flops.csv) ·
GitHub: [snr_variant_min_size_by_L_flops.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_flops.png) · [snr_variant_min_size_by_L_flops.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_min_size_by_L_flops.csv)

**Key findings** (pool `predictivity`, multi-axis pairs, gated at the proxy
and at 1.7B; populations per figure as stated in the generated block)

- **Per L, an SNR definition rarely tracks DA-size:** 97 of the 132
  (definition, L) cells of `snr_variant_min_size_by_L` never reach ρ ≥ 0.3
  under DA-size, but only 38 under DA-ckpt. Per-L DA rests on few pairs, so
  this reads the lattice of possible DA values as much as the definitions.
- **Pooled over every pair (version B), the best definition reads ρ
  0.55–0.61 with DA-size on the benchmarks at every proxy**, the median
  definition 0.37–0.46; with DA-ckpt the best reads 0.76–0.84, both bests
  peaking at 350M (`rel_mpd` is the DA-size best at every proxy).
- **The per-language BPB flip at 600M shows here too:** the median
  definition's ρ with DA-size on BPB is 0.39–0.51 at 90M–350M and 0.44 at
  1B, but −0.36 at 600M.
- **The per-FLOPs line zigzags by construction:** it joins a bigger model's
  first (peak-LR) checkpoint to a smaller model's annealed final.

**Follow-ups**

- The L-dependence of SNR with SNR computed over the L's own variants
  (degenerate below three variants), instead of the pooled signal.
- One per-FLOPs line per proxy size with the annealed points marked, plus
  the compute frontier (the best DA reachable at or below each compute), for
  the paper.

## Surrogates from the literature — summary

`catalogue.py` computes 211 statistics read on the proxy alone, save
`pseudo_ref_da`, which reads the 1B rung below the reference (labelled so): the literature catalogue
([`literature.md`](literature.md); every definition, its source and how our port
differs from it are in the paper's appendix,
`documents/paper/sections/app_snr_new.tex`), every AllenAI signal over every
noise (22 × 6, the k-fold benchmark noise included), and each signal and noise
alone.

`search.py` scores them against 48 truths from the decision-accuracy
tables (DA / τ-b / ρ for DA-size, DA-goal and DA-ckpt, both pair sets, every
pair or the pairs sharing one L) on all the data: Spearman over one point per
(benchmark, language) cluster, permutation p below 30 clusters,
Benjamini–Hochberg over every configuration, and a split-half confirmation as
an extra. The numbers are in the generated block below.

Population: pool `predictivity`, proxies 90M–1B, cells above chance
(rule 1), 420 (benchmark, language) clusters over every benchmark. A cluster
holds a benchmark's original, its `rf_` twins and their bBPB twins, which
now carry every checkpoint at every proxy, so the DA-goal and DA-ckpt rows
include them in full.

**Key findings**

- **The ceiling.** A truth re-read at 90 % keeps ρ 0.92 (DA-size), 0.95
  (DA-goal) and 0.99 (DA-ckpt) with itself on the benchmarks (multi-axis).
- **No surrogate reaches it, the proxy's rank stability comes closest:** the
  window sign consistency reads 0.93 against DA-ckpt (the early-vs-final
  agreements, consecutive-checkpoint Kendall τ included, are left out of
  DA-ckpt as circular), the consecutive τ 0.86 against DA-goal and the
  proxy's mean DA-ckpt 0.83 against DA-size (90–94 % of the ceiling; the
  mean DA-ckpt, tied with `sign_persistence`, also leads DA-goal at 0.87).
  On mono-axis pairs the best read 0.84, 0.73 and 0.69.
- **SNR is close behind.** AllenAI's `rel_std` / checkpoint noise reads
  0.79, 0.82 and 0.85 on DA-size, DA-goal and DA-ckpt (14th–15th of the
  203–210 surrogates); the best grid combination, `mpd / ckpt_abs`, reads
  0.79, 0.83 and 0.86, within 0.02 of it.
- **As the SNR's denominator, the detrended checkpoint noise beats the
  k-fold noise:** relative k-fold noise beats relative checkpoint noise for
  only 6, 6 and 6 of the 22 signals (DA-size, DA-goal, DA-ckpt), although
  alone it still reads slightly stronger (−0.46, −0.51, −0.64 against
  −0.44, −0.45, −0.58). `n_items` alone reads 0.37, 0.40 and 0.52.
- **Signal alone carries almost nothing:** range, `dist_std` and `mpd` across
  variants read −0.02 to 0.13 against the three DA truths, and the depth
  signals are negative (−0.17 to −0.38).
- **Particular cases reach |ρ| 0.95–1.00, on single languages or
  benchmarks:** each holds 10–26 clusters. The `reliable (on the truth)`
  subsets are selected on the truth, so their ρ is conditional on it.
- **Language count.** Over pairs of variants sharing L, mean DA-size is 0.61
  / 0.50 / 0.50 / 0.55 / 0.56 / 0.61 at L1 / 2 / 8 / 15 / 30 / 50 (131–1289
  tasks): no monotone trend.
- **Split-half check.** 1846 of 5743 configurations chosen on one half hold
  on the other at q < 0.05; the strongest is the window sign consistency on
  the L8 tier against DA-ckpt (0.75 on 152 held-out tasks).

**Follow-ups**

- The catalogue rows split by scoring (accuracy vs bBPB), so the
  rank-stability lead can be read on accuracy alone; the twins now carry
  every checkpoint, so nothing blocks it.
- A per-proxy version of the strongest surrogates: the split-half winners
  concentrate on the L8 tier, the late checkpoints and the 90M–350M
  proxies.

<!-- BEGIN auto:catalogue (search.py --pool predictivity) -->
## A catalogue of surrogates beyond SNR

Numbers from the `predictivity` pool. Regenerate with `python analysis/rq04_surrogates/catalogue.py --pool predictivity` then `search.py --pool predictivity`. Surrogates are read on the proxy alone (rule 11), except `pseudo_ref_da`, which reads the 1B rung; the truths are rq02's DA (DA-size; DA-goal and DA-ckpt at every early checkpoint of the proxy; both pair sets; every pair or the pairs sharing L) and Kendall τ-b / Spearman ρ on the same rankings, over cells above chance (rule 1). ρ is a Spearman over one point per (benchmark, language) cluster, pooled over proxies and checkpoints; populations differ per configuration (rule 13) and n is in the CSVs. Definitions and sources: [`literature.md`](literature.md); method: `search.py`'s docstring.

**Main analysis (all the data).** 771,864 configurations (truths × subsets × surrogates, the threshold filters included); 471,760 have a p-value (≥ 10 units); **131,897 hold at BH q < 0.05**. A best-of-many ρ is optimistic even when it is significant; the extra below measures by how much.

**The ceiling** — each truth against itself re-read at 90 % (benchmarks):

| truth | ρ | ρ over cells | units |
|---|---|---|---|
| DA-size da (multi-axis) | 0.92 | 0.84 | 420 |
| DA-goal da (multi-axis) | 0.95 | 0.84 | 420 |
| DA-ckpt da (multi-axis) | 0.99 | 0.89 | 420 |
| DA-size tau_b (multi-axis) | 0.92 | 0.82 | 420 |
| DA-size rho (multi-axis) | 0.93 | 0.84 | 420 |
| DA-goal tau_b (multi-axis) | 0.95 | 0.82 | 420 |
| DA-goal rho (multi-axis) | 0.95 | 0.84 | 420 |
| DA-ckpt tau_b (multi-axis) | 0.98 | 0.88 | 420 |
| DA-ckpt rho (multi-axis) | 0.99 | 0.89 | 420 |

**Strongest surrogate per truth**, every benchmark:

| truth | surrogate | ρ | q | ρ over cells | ρ within stratum | units |
|---|---|---|---|---|---|---|
| DA-ckpt da (mono-axis, pairs within L1) | `snr__rel_dispersion__tukey_depth` | 0.86 | 2.7e-04 | 0.03 | 0.01 | 24 |
| DA-ckpt da (mono-axis, pairs within L15) | `snr__star_discrepancy_shifted__kfold_rel` | 0.53 | 2.0e-11 | 0.16 | 0.16 | 154 |
| DA-ckpt da (mono-axis, pairs within L2) | `n_items` | 0.81 | 2.7e-04 | 0.31 | 0.31 | 18 |
| DA-ckpt da (mono-axis, pairs within L30) | `sign_consistency_window` | 0.63 | 2.4e-32 | 0.25 | 0.24 | 298 |
| DA-ckpt da (mono-axis, pairs within L50) | `window_kendall` | 0.80 | 9.7e-91 | 0.36 | 0.35 | 420 |
| DA-ckpt da (mono-axis, pairs within L8) | `window_kendall` | 0.67 | 2.7e-13 | 0.25 | 0.22 | 102 |
| DA-ckpt da (mono-axis) | `window_kendall` | 0.84 | 7.6e-112 | 0.49 | 0.50 | 420 |
| DA-ckpt da (multi-axis, pairs within L1) | `snr__rel_dispersion__ckpt_abs` | 0.87 | 2.7e-04 | 0.10 | 0.09 | 24 |
| DA-ckpt da (multi-axis, pairs within L15) | `sign_consistency_window` | 0.59 | 3.0e-17 | 0.35 | 0.34 | 185 |
| DA-ckpt da (multi-axis, pairs within L2) | `snr__star_discrepancy_shifted__ckpt_abs` | 0.82 | 2.7e-04 | 0.32 | 0.31 | 19 |
| DA-ckpt da (multi-axis, pairs within L30) | `sign_consistency_window` | 0.75 | 4.6e-53 | 0.40 | 0.41 | 298 |
| DA-ckpt da (multi-axis, pairs within L50) | `mde_resolved` | 0.86 | 4.9e-122 | 0.44 | 0.45 | 420 |
| DA-ckpt da (multi-axis, pairs within L8) | `sign_consistency_window` | 0.60 | 1.4e-11 | 0.26 | 0.24 | 116 |
| DA-ckpt da (multi-axis) | `sign_consistency_window` | 0.93 | 5.1e-178 | 0.64 | 0.68 | 420 |
| DA-ckpt rho (multi-axis) | `sign_consistency_window` | 0.91 | 1.2e-160 | 0.61 | 0.65 | 420 |
| DA-ckpt tau_b (multi-axis) | `mde_resolved` | 0.92 | 8.5e-167 | 0.56 | 0.59 | 420 |
| DA-goal da (mono-axis, pairs within L1) | `rung_gap_corr` | 0.74 | 2.7e-04 | 0.31 | 0.31 | 24 |
| DA-goal da (mono-axis, pairs within L15) | `benchmark_consensus` | 0.48 | 1.3e-10 | 0.13 | 0.12 | 176 |
| DA-goal da (mono-axis, pairs within L2) | `snr__star_discrepancy_shifted__ckpt_abs` | 0.78 | 1.8e-03 | 0.30 | 0.30 | 18 |
| DA-goal da (mono-axis, pairs within L30) | `n_items` | 0.37 | 5.3e-08 | 0.13 | 0.13 | 230 |
| DA-goal da (mono-axis, pairs within L50) | `consecutive_kendall` | 0.75 | 3.8e-74 | 0.34 | 0.34 | 420 |
| DA-goal da (mono-axis, pairs within L8) | `snr__iqr__kfold_abs` | 0.49 | 6.8e-04 | 0.10 | 0.10 | 56 |
| DA-goal da (mono-axis) | `snr__mpd__ckpt_abs` | 0.73 | 3.4e-69 | 0.36 | 0.39 | 420 |
| DA-goal da (multi-axis, pairs within L1) | `snr__rel_dispersion__ckpt_abs` | 0.73 | 5.1e-04 | 0.17 | 0.17 | 24 |
| DA-goal da (multi-axis, pairs within L15) | `tie_rate_items` | -0.52 | 2.2e-10 | -0.22 | -0.22 | 144 |
| DA-goal da (multi-axis, pairs within L2) | `snr__star_discrepancy_shifted__ckpt_abs` | 0.66 | 1.4e-02 | 0.30 | 0.30 | 18 |
| DA-goal da (multi-axis, pairs within L30) | `pseudo_ref_da` | 0.63 | 2.4e-32 | 0.30 | 0.30 | 298 |
| DA-goal da (multi-axis, pairs within L50) | `consecutive_kendall` | 0.82 | 5.4e-102 | 0.45 | 0.46 | 420 |
| DA-goal da (multi-axis, pairs within L8) | `gain_over_noise` | 0.46 | 3.0e-06 | 0.07 | 0.07 | 112 |
| DA-goal da (multi-axis) | `sign_persistence` | 0.87 | 9.3e-127 | 0.63 | 0.64 | 420 |
| DA-goal rho (multi-axis) | `sign_persistence` | 0.86 | 8.8e-120 | 0.60 | 0.61 | 420 |
| DA-goal tau_b (multi-axis) | `sign_persistence` | 0.85 | 5.7e-117 | 0.58 | 0.59 | 420 |
| DA-size da (mono-axis, pairs within L1) | `snr__rel_dispersion__tukey_depth` | 0.73 | 5.1e-04 | 0.15 | 0.14 | 24 |
| DA-size da (mono-axis, pairs within L15) | `benchmark_consensus` | 0.36 | 5.1e-06 | 0.13 | 0.13 | 176 |
| DA-size da (mono-axis, pairs within L2) | `signal__star_discrepancy_shifted` | 0.71 | 5.7e-03 | 0.33 | 0.34 | 18 |
| DA-size da (mono-axis, pairs within L30) | `pseudo_ref_da` | 0.32 | 3.0e-07 | 0.15 | 0.15 | 293 |
| DA-size da (mono-axis, pairs within L50) | `consecutive_kendall` | 0.66 | 1.9e-52 | 0.35 | 0.35 | 420 |
| DA-size da (mono-axis, pairs within L8) | `signal__gini` | 0.33 | 1.4e-02 | 0.02 | 0.02 | 74 |
| DA-size da (mono-axis) | `bpb_rank_agreement` | 0.69 | 2.0e-57 | 0.42 | 0.42 | 420 |
| DA-size da (multi-axis, pairs within L1) | `snr__rel_dispersion__ckpt_abs` | 0.75 | 2.7e-04 | 0.19 | 0.19 | 24 |
| DA-size da (multi-axis, pairs within L15) | `tie_rate_items` | -0.47 | 1.8e-08 | -0.27 | -0.27 | 144 |
| DA-size da (multi-axis, pairs within L2) | `snr__star_discrepancy_shifted__projection_depth` | 0.69 | 6.6e-03 | 0.39 | 0.41 | 18 |
| DA-size da (multi-axis, pairs within L30) | `pseudo_ref_da` | 0.62 | 4.1e-31 | 0.35 | 0.35 | 298 |
| DA-size da (multi-axis, pairs within L50) | `consecutive_kendall` | 0.77 | 1.6e-82 | 0.46 | 0.47 | 420 |
| DA-size da (multi-axis, pairs within L8) | `snr__quartile_deviation__kfold_rel` | 0.35 | 2.2e-03 | 0.11 | 0.11 | 94 |
| DA-size da (multi-axis) | `sign_persistence` | 0.83 | 6.7e-107 | 0.64 | 0.65 | 420 |
| DA-size rho (multi-axis) | `sign_persistence` | 0.83 | 2.6e-105 | 0.61 | 0.61 | 420 |
| DA-size tau_b (multi-axis) | `sign_persistence` | 0.82 | 3.6e-99 | 0.59 | 0.60 | 420 |

**The AllenAI grid** (22 signals × 6 noises, DA, every pair, benchmarks): best combination against AllenAI's own `rel_std` / checkpoint noise:

| truth | best signal / noise | ρ | rel_std / ckpt_rel ρ |
|---|---|---|---|
| DA-ckpt | `mpd / ckpt_abs` | 0.86 | 0.85 |
| DA-goal | `mpd / ckpt_abs` | 0.83 | 0.82 |
| DA-size | `mpd / ckpt_abs` | 0.79 | 0.79 |

**Particular cases** — the strongest significant configurations outside "every benchmark", two per (truth, subset type); `reliable (on the truth)` subsets are selected on the truth itself, so their ρ is conditional on it; all of them in `surrogate_correlations.csv` (`q` column):

| truth | subset | surrogate | ρ | q | ρ over cells | units |
|---|---|---|---|---|---|---|
| DA-goal da (multi-axis) | language: vi | `snr__rel_std__ckpt_rel` | 1.00 | 2.7e-04 | 0.58 | 12 |
| DA-goal da (multi-axis) | language: vi | `snr__rms_deviation__ckpt_abs` | 1.00 | 2.7e-04 | 0.57 | 12 |
| DA-ckpt da (mono-axis) | language: it | `snr__mad__ckpt_abs` | 0.99 | 2.7e-04 | 0.43 | 12 |
| DA-ckpt da (mono-axis) | language: it | `snr__mad__projection_depth` | 0.99 | 2.7e-04 | 0.37 | 12 |
| DA-goal rho (multi-axis) | language: vi | `snr__rms_deviation__ckpt_abs` | 0.99 | 2.7e-04 | 0.54 | 12 |
| DA-goal rho (multi-axis) | language: vi | `snr__rel_std__ckpt_rel` | 0.99 | 2.7e-04 | 0.54 | 12 |
| DA-size da (multi-axis, pairs within L15) | language: pt | `bpb_corr_training` | 0.98 | 2.7e-04 | 0.51 | 11 |
| DA-goal tau_b (multi-axis) | language: vi | `snr__rms_deviation__ckpt_abs` | 0.98 | 2.7e-04 | 0.54 | 12 |
| DA-goal tau_b (multi-axis) | language: vi | `snr__rel_std__ckpt_rel` | 0.98 | 2.7e-04 | 0.54 | 12 |
| DA-ckpt rho (multi-axis) | language: id | `window_kendall` | 0.98 | 2.7e-04 | 0.60 | 10 |
| DA-ckpt rho (multi-axis) | language: pt | `total_variation` | -0.97 | 2.7e-04 | -0.50 | 11 |
| DA-ckpt tau_b (multi-axis) | language: pt | `total_variation` | -0.97 | 2.7e-04 | -0.49 | 11 |
| DA-size rho (multi-axis) | benchmark: bbpb_arc_mt | `retest_agreement` | 0.97 | 2.7e-04 | 0.46 | 11 |
| DA-ckpt tau_b (multi-axis) | language: ar | `prev_rung_da` | 0.97 | 2.7e-04 | 0.52 | 13 |
| DA-size rho (multi-axis) | language: vi | `da_ckpt_mean` | 0.97 | 2.7e-04 | 0.62 | 12 |
| DA-size rho (multi-axis) | language: vi | `sign_persistence` | 0.97 | 2.7e-04 | 0.62 | 12 |
| DA-ckpt rho (multi-axis) | benchmark: hellaswag | `snr__rel_std__kfold_rel` | 0.97 | 2.7e-04 | 0.63 | 26 |
| DA-ckpt da (multi-axis) | benchmark: hellaswag | `snr__quartile_deviation__ckpt_rel` | 0.97 | 2.7e-04 | 0.55 | 26 |
| DA-ckpt rho (multi-axis) | benchmark: hellaswag | `snr__rms_deviation__kfold_abs` | 0.97 | 2.7e-04 | 0.63 | 26 |
| DA-ckpt tau_b (multi-axis) | benchmark: hellaswag | `snr__quartile_deviation__ckpt_rel` | 0.97 | 2.7e-04 | 0.55 | 26 |
| DA-ckpt da (multi-axis) | benchmark: hellaswag | `signal__aad` | 0.97 | 2.7e-04 | 0.64 | 26 |
| DA-size da (multi-axis) | language: vi | `da_ckpt_mean` | 0.97 | 2.7e-04 | 0.68 | 12 |
| DA-ckpt tau_b (multi-axis) | benchmark: hellaswag | `snr__mpsd__projection_depth` | 0.96 | 2.7e-04 | 0.58 | 26 |
| DA-goal da (multi-axis) | benchmark: hellaswag | `resolved_pairs_binomial` | 0.96 | 2.7e-04 | 0.57 | 26 |
| DA-size tau_b (multi-axis) | language: vi | `da_ckpt_mean` | 0.96 | 2.7e-04 | 0.61 | 12 |
| DA-size tau_b (multi-axis) | language: vi | `sign_persistence` | 0.96 | 2.7e-04 | 0.61 | 12 |
| DA-goal tau_b (multi-axis) | benchmark: hellaswag | `resolved_pairs_binomial` | 0.96 | 2.7e-04 | 0.56 | 26 |
| DA-goal rho (multi-axis) | benchmark: hellaswag | `kendall_w_window` | 0.95 | 2.7e-04 | 0.57 | 26 |
| DA-goal rho (multi-axis) | benchmark: hellaswag | `signal__rms_deviation` | 0.95 | 2.7e-04 | 0.59 | 26 |
| DA-goal da (multi-axis) | benchmark: hellaswag | `sign_consistency_window` | 0.95 | 2.7e-04 | 0.56 | 26 |

**Per language count** (pairs of variants sharing L; benchmarks):

| truth | L | mean DA | sd | tasks | best surrogate | ρ |
|---|---|---|---|---|---|---|
| DA-ckpt | 1 | 0.70 | 0.25 | 138 | `snr__rel_dispersion__ckpt_abs` | 0.87 |
| DA-ckpt | 2 | 0.61 | 0.25 | 138 | `snr__star_discrepancy_shifted__ckpt_abs` | 0.82 |
| DA-ckpt | 8 | 0.60 | 0.26 | 453 | `sign_consistency_window` | 0.60 |
| DA-ckpt | 15 | 0.63 | 0.22 | 684 | `sign_consistency_window` | 0.59 |
| DA-ckpt | 30 | 0.63 | 0.19 | 979 | `sign_consistency_window` | 0.75 |
| DA-ckpt | 50 | 0.68 | 0.25 | 1310 | `mde_resolved` | 0.86 |
| DA-goal | 1 | 0.59 | 0.27 | 131 | `snr__rel_dispersion__ckpt_abs` | 0.73 |
| DA-goal | 2 | 0.51 | 0.24 | 131 | `snr__star_discrepancy_shifted__ckpt_abs` | 0.66 |
| DA-goal | 8 | 0.49 | 0.25 | 334 | `gain_over_noise` | 0.46 |
| DA-goal | 15 | 0.55 | 0.23 | 671 | `tie_rate_items` | -0.52 |
| DA-goal | 30 | 0.55 | 0.20 | 964 | `pseudo_ref_da` | 0.63 |
| DA-goal | 50 | 0.60 | 0.26 | 1289 | `consecutive_kendall` | 0.82 |
| DA-size | 1 | 0.61 | 0.25 | 131 | `snr__rel_dispersion__ckpt_abs` | 0.75 |
| DA-size | 2 | 0.50 | 0.25 | 131 | `snr__star_discrepancy_shifted__projection_depth` | 0.69 |
| DA-size | 8 | 0.50 | 0.25 | 334 | `snr__quartile_deviation__kfold_rel` | 0.35 |
| DA-size | 15 | 0.55 | 0.23 | 671 | `tie_rate_items` | -0.47 |
| DA-size | 30 | 0.56 | 0.20 | 964 | `pseudo_ref_da` | 0.62 |
| DA-size | 50 | 0.61 | 0.26 | 1289 | `consecutive_kendall` | 0.77 |

**Extra: held-out confirmation.** The (benchmark, language) clusters split in two halves; 5743 configurations ranked best on one half (within-stratum ρ) were tested once on the other (one-sided cluster permutation test, BH): **1846 hold at q < 0.05**. The strongest:

| truth | subset | surrogate | ρ disc. | ρ val. [90 %] | q | tasks val. |
|---|---|---|---|---|---|---|
| DA-ckpt da (multi-axis) | tier: L8 | `sign_consistency_window` | 0.72 | 0.75 [0.70, 0.79] | 2.4e-03 | 152 |
| DA-ckpt da (multi-axis) | tier: L8 | `kendall_w_window` | 0.70 | 0.74 [0.69, 0.77] | 2.4e-03 | 152 |
| DA-ckpt da (multi-axis) | stage: late (60-70%) | `sign_consistency_window` | 0.73 | 0.72 [0.70, 0.74] | 2.4e-03 | 560 |
| DA-size da (multi-axis) | benchmark: multiblimp | `ladder_da` | 0.76 | 0.72 [0.46, 0.85] | 2.4e-03 | 17 |
| DA-ckpt da (multi-axis) | tier: L8 | `ladder_da` | 0.71 | 0.72 [0.67, 0.76] | 2.4e-03 | 144 |
| DA-ckpt da (multi-axis) | tier: L30 | `cronbach_alpha` | 0.71 | 0.71 [0.67, 0.74] | 2.4e-03 | 151 |
| DA-ckpt da (multi-axis) | proxy: 90M | `sign_consistency_window` | 0.71 | 0.71 [0.67, 0.74] | 2.4e-03 | 491 |
| DA-ckpt da (multi-axis) | proxy: 175M | `cronbach_alpha` | 0.71 | 0.71 [0.67, 0.74] | 2.4e-03 | 501 |
| DA-ckpt da (multi-axis) | stage: late (60-70%) | `cronbach_alpha` | 0.72 | 0.71 [0.68, 0.73] | 2.4e-03 | 560 |
| DA-ckpt da (multi-axis) | stage: late (60-70%) | `window_kendall` | 0.70 | 0.70 [0.68, 0.73] | 2.4e-03 | 560 |
| DA-size rho (multi-axis) | benchmark: hellaswag | `bpb_rank_agreement` | 0.74 | 0.70 [0.46, 0.83] | 2.4e-03 | 13 |
| DA-ckpt rho (multi-axis) | tier: L8 | `sign_consistency_window` | 0.65 | 0.70 [0.65, 0.74] | 2.4e-03 | 152 |
| DA-ckpt da (multi-axis) | proxy: 350M | `cronbach_alpha` | 0.70 | 0.70 [0.68, 0.72] | 2.4e-03 | 514 |
| DA-ckpt rho (multi-axis) | tier: L8 | `kendall_w_window` | 0.64 | 0.70 [0.65, 0.74] | 2.4e-03 | 152 |
| DA-ckpt da (multi-axis) | proxy: 90M | `bpb_corr_training` | 0.71 | 0.70 [0.66, 0.73] | 2.4e-03 | 491 |

![surrogates_significant](pretraining/predictivity/surrogates_significant.png)

![surrogates_snr_grid](pretraining/predictivity/surrogates_snr_grid.png)

![surrogates_catalogue](pretraining/predictivity/surrogates_catalogue.png)

![surrogates_by_L](pretraining/predictivity/surrogates_by_L.png)

![surrogates_catalogue_by_language](pretraining/predictivity/surrogates_catalogue_by_language.png)

![surrogates_validated](pretraining/predictivity/surrogates_validated.png)
<!-- END auto:catalogue -->

## TODO

- [ ] **The seed holdout ranks most tasks on a single model pair (open, 2026-09-17).**
      *What it is.* This analysis claims one SNR definition tracks decision
      accuracy best; the holdout (`rq03_noise_and_snr/compare_seed_splits.py`,
      reported in the highlight and "Seed generalization" table) asks whether that
      ranking of definitions survives a change of seed: it is computed on
      `predictivity_seeds_train` (the replicate seeds 64, 313, 28, 1797) and again on
      `predictivity_seeds_test` (seed 1904 of the same cells) and the two are
      compared.
      *The problem.* (2026-10-05: the train split is now every replicate
      seed, so the 1B ×3 cells, seeds 28/1797, join both splits and the
      text below predates that.) Replicate seeds exist only at 175M and 600M, so both
      splits hold those two sizes and DA-size there is 175M → 600M (the 1.7B
      reference never enters). English tasks and BPB rest on 15 model pairs.
      A non-English benchmark is trained only in the L50 cell, so its train
      split holds one pair — the two seeds of the same cell — and its
      "decision accuracy" is 0 or 1 and measures seed noise, not a ranking.
      The median task has one pair.
      *Implications (2026-10-08 12:06 report).* Only English has a best
      definition on both splits under DA-size and only English and Russian
      under DA-ckpt, so the per-language agreement (100 % / 100 %) is 1 of 1
      and 2 of 2 languages. The global ranking ρ (0.55 DA-size, 0.70 DA-ckpt) is
      over 22 definitions on splits that share only those 1–2 languages, so "the ranking
      survives a seed swap" says as much about the holdout as about the
      definitions; nothing in the main `predictivity` tables is affected.
      *Options.* (A) keep only tasks with ≥ 6 pairs on both splits: an honest
      check, but English + BPB only. (B) treat replicate seeds as replicates,
      not as models: average them before ranking, and use the holdout only
      for the noise estimate — changes what the holdout means. (C) report the
      holdout for DA-ckpt only, whose pairs are checkpoints × variants within
      a size and do not shrink to one. (D) drop the holdout numbers from the
      highlight until more sizes have replicate seeds.
      *Recommendation.* A now (small change, honest scope, say "English and
      BPB" in the highlight), and revisit once the 90M / 1.7B-L15 cells and
      any further replicate seeds exist. Not implemented yet.
- [ ] Bootstrap CIs on per-language Pearson r and cross-pool Spearman ρ.
- [ ] Recommend a *family* (dispersion / relative-spread), not an exact variant
      — only the family transfers across seeds.
- [ ] Use a larger DA-size target (e.g. Apertus-8B) instead of the
      not-fully-converged 1B custom model.

## Extensions from other sweeps

Everything below comes from the **36-model sweep** (2026-04…06, 4 sizes × 3
data mixtures × 3 seeds, pools `seeds_1904`, `seeds_28_1797`,
`seeds_28_1797_1904`, `custom_swissai_hf`; 12 languages, reference **1B**)
or from the **external tier** (`all/external`: the public and reference
models, 270M–70B, no data-mixture axis). Its harness, task set, checkpoint
window and reference differ from the ladder's, and its 36-sweep DA-size is
a 3-mixture ranking against 1B, so its numbers are replications of the
family-level recommendation above, never rows of the same table. The dated
comparison note that once tracked these tiers (`ANALYSIS_new_vs_previous.md`, removed 2026-09-23:
`rel_mpd` on top in every tier, DA-ckpt r 0.379 → 0.519 with the externals
folded in, holdout ρ 0.80 / 0.93) predates the tie fix in `decision_acc_fast`
and the parent-only filter and is superseded by the blocks below.

## External model-set tier (`all/external`, 36-sweep)

The `external` tier pools every non-custom model (reference HF, a06,
distillation, posttraining; sizes 270M…70B) with **no data-mixture axis**, so
"Signal" here is **cross-model dispersion** of final-checkpoint scores across the
external ladder — *how far different model families separate on a benchmark*,
rather than how far the three FineWeb mixtures separate. SNR magnitudes are
consequently an order larger than the custom pool's (e.g. `multiblimp_rus` SNR
≈ 119) and DA is the within-family scaling-DA over that ladder. The headline —
**recommend the dispersion *family*, not an exact variant** — holds on this
disjoint model set, the strongest robustness check we have. Regenerate with
`python analysis/rq04_surrogates/snr_definition_postprocess.py --pool external`.

**Global variant ranking** — mean Pearson r of log₁₀(SNR) vs DA across languages.
The dispersion and relative-spread clusters lead overall and on DA-ckpt; the
discrepancy cluster leads DA-size only; depth metrics fail; the custom-pool winner
`dist_std` is undefined on this tier:

| variant (family) | DA-size r | DA-ckpt r | overall |
|---|---|---|---|
| `dispersion` / `mpd` / `range` / `mad` (dispersion) | 0.16 | **0.44** | **0.30** |
| `rel_std` / `rel_mpd` / `iqr` (relative-spread) | 0.16 | 0.43 | 0.30 |
| `mpsd` (dispersion) | 0.10 | **0.45** | 0.27 |
| `star_discrepancy` (discrepancy) | **0.20** | 0.05 | 0.12 |
| `discrepancy` / `gini` / `dispersion_shifted` (discrepancy) | 0.19 | 0.07 | 0.13 |
| `projection` (depth) | 0.06 | −0.21 | −0.07 |
| `dist_std`, `tukey` | — | — | — |

![SNR variants ranked by correlation with DA (external)](all/external/top_variants_overall.png)

DA-size is sparse on the external ladder (most cross-bucket pairs lack ≥2
spanning families), so **DA-ckpt is the more trustworthy axis here** — and on it
the dispersion/relative-spread families win cleanly, agreeing with the custom
pool's family-level recommendation.

**Most reliable benchmark per language** — rank-1 benchmark by `dist_std` SNR @ 1B
over above-random tasks. With capable models clearing the gate, the
long-completion 4-option `hellaswag_<lang>` joins MultiBLiMP at the top:

| lang | top benchmark | SNR | | lang | top benchmark | SNR |
|---|---|---|---|---|---|---|
| ar | `hellaswag_ar` | 78.9 | | ru | `multiblimp_rus` | 119.5 |
| en | `multiblimp_eng` | 82.1 | | th | `xcopa_th` | 10.1 |
| es | `multiblimp_spa` | 87.0 | | tr | `multiblimp_tur` | 40.5 |
| eu | `multiblimp_eus` | 38.4 | | vi | `hellaswag_vi` | 38.5 |
| hi | `multiblimp_hin` | 73.7 | | zh | `xstorycloze_zh` | 25.4 |
| ja | `xwinograd_jp` | 20.5 | | | | |

![Top benchmarks per language by SNR (external)](all/external/top_benchmarks_per_language.png)

The exact per-language argmax still does not transfer across tiers — only the
dispersion/relative-spread *family* does — so the paper-level claim is the family
recommendation, with HellaSwag and MultiBLiMP as the durable multilingual anchors.

[top_variants_overall.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/all/external/top_variants_overall.png) ·
[top_benchmarks_per_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/all/external/top_benchmarks_per_language.png)

## Results from the 36-model sweep (2026-06, superseded)

The numbers below were generated on the 36-model sweep (4 sizes × 3 data mixtures × 3 seeds, 12 languages, pool `custom_swissai_hf` unless stated) and are kept as history; the predictivity ladder regenerates the blocks above.

### Highlighted result

- **Global-best SNR definition (`custom_swissai_hf`): `dist_std`** — mean Pearson r of log₁₀(SNR) vs decision accuracy **0.32** (DA-size), **0.43** (DA-ckpt), 0.38 overall. DA-ckpt is led by the mean-pairwise-distance / relative-spread cluster (`mpsd`/`rel_mpd`/`rel_mpsd` ≈ 0.51) — all dispersion-family, so recommend the *family*, not an exact variant.
- **Per-language anchor: `multiblimp`** — the highest-SNR above-random benchmark in **6 of 11** languages (`dist_std` SNR @ 1B).
- **Variant ranking generalizes across seeds under DA-ckpt** (holdout Spearman ρ **0.81**) **but not under DA-size** (ρ **-0.07**): the DA-size per-variant correlations are small and near-tied, so their ranking is noise-dominated and does not survive a seed swap — treat DA-size variant ranking as unreliable. The exact per-language argmax never transfers. **Never `tukey` / `projection`** (r ≤ 0).

### Results

Headline numbers from the `custom_swissai_hf` pool. Regenerate with `python analysis/rq04_surrogates/snr_definition_postprocess.py --pool custom_swissai_hf`.

**Global variant ranking** — mean Pearson r of log₁₀(SNR) vs DA across languages (top of a tight dispersion block; depth metrics collapse):

| variant | DA-size r | DA-ckpt r | overall |
|---|---|---|---|
| `dist_std` | 0.32 | 0.43 | 0.38 |
| `star_discrepancy_shifted` | 0.14 | 0.15 | 0.15 |
| `gini` | 0.13 | 0.12 | 0.13 |
| `discrepancy` | 0.13 | -0.04 | 0.05 |
| `star_discrepancy` | 0.12 | -0.00 | 0.06 |
| `rel_mpd` | 0.11 | 0.51 | 0.31 |
| `rel_std` | 0.11 | 0.50 | 0.31 |
| … |  |  |  |
| `tukey` | 0.05 | 0.22 | 0.14 |
| `projection` | -0.03 | -0.27 | -0.15 |

![SNR variants ranked by correlation with DA](pretraining/custom_swissai_hf/top_variants_overall.png)

**Statistical power by pool** — each pool's best DA-size variant:

| pool | best variant (DA-size) | DA-size r | DA-ckpt r |
|---|---|---|---|
| `seeds_1904` (1 seed) | `mad` | 0.50 | 0.28 |
| `seeds_28_1797` (2 seeds) | `rel_mpd` | 0.31 | 0.37 |
| `seeds_28_1797_1904` (3 seeds) | `rel_std` | 0.43 | 0.48 |
| `custom_swissai_hf` (3 seeds + externals) | `dist_std` | 0.32 | 0.43 |

**Most reliable benchmark per language** — `dist_std` SNR @ 1B over above-random tasks (DA-size is NaN at the 1B target, so DA-ckpt@1B is shown):

| lang | top benchmark | SNR | DA-ckpt@1B |
|---|---|---|---|
| ar | `multiblimp_arb` | 2.65 | 0.87 |
| en | `xwinograd_en` | 2.40 | 0.83 |
| es | `multiblimp_spa` | 3.37 | 0.85 |
| eu | `multiblimp_eus` | 1.28 | 0.64 |
| hi | `multiblimp_hin` | 4.95 | 0.85 |
| ja | `xwinograd_jp` | 2.28 | 0.76 |
| ru | `multiblimp_rus` | 7.08 | 0.86 |
| th | `xnli_th` | 1.28 | 0.75 |
| tr | `multiblimp_tur` | 2.75 | 0.79 |
| vi | `xcopa_vi` | 1.61 | 0.76 |
| zh | `xcopa_zh` | 1.57 | 0.61 |

![Top-5 benchmarks per language by SNR](pretraining/custom_swissai_hf/top_benchmarks_per_language.png)

**Seed generalization** — holdout `seeds_28_1797` → `seeds_1904`. DA-ckpt ranking transfers; **DA-size does not** — its per-variant correlations are small and clustered (top variants within ~0.1), so the global ranking is noise-dominated and its Spearman ρ is unstable run-to-run (don't read it as a real effect):

| metric | DA-size | DA-ckpt |
|---|---|---|
| Spearman ρ on global variant ranking | -0.07 | 0.81 |
| Pearson r between splits (all cells) | 0.48 | 0.60 |
| Exact-variant agreement (per lang) | 7% | 7% |
| Family-level agreement (per lang) | 21% | 29% |
| Retention of train-best r on test | 61% | 79% |

[top_variants_overall.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/custom_swissai_hf/top_variants_overall.png) ·
[top_benchmarks_per_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/custom_swissai_hf/top_benchmarks_per_language.png)

## Files

- `pretraining/<pool>/snr_variant_ranking.csv` — per-(variant, DA-def, scope)
  Pearson r (`analyze_snr_variants.py`).
- `…/top_variants_overall.csv`, `best_variant_per_language.csv`,
  `best_variant_family_per_language.csv`, `variant_clusters.csv`,
  `top_benchmarks_per_language.csv` — the variant ranking and the per-language
  anchor (`snr_definition_postprocess.py`).
- `…/top_variants_overall.png`, `best_variant_per_language.png`,
  `best_variant_family_per_language.png`, `top_benchmarks_per_language.png`,
  `variant_correlation_matrix.png`, `da_size_vs_da_ckpt_multi_axes.png`,
  `{da_size,da_ckpt}/…` — supporting figures.
- `…/rq3_surrogates.csv`, `rq3_surrogates.png`, `facts.json` — the
  statistics beyond SNR (`analyze.py`; the paper's surrogate figure).
- `…/surrogate_values.csv`, `surrogate_targets.csv`, `surrogate_definitions.csv` — the
  surrogates per (task, proxy, pair set) and the truths, long (`catalogue.py`; sources in
  `literature.md`); `surrogate_correlations.csv`, `surrogate_filters.csv`,
  `surrogate_validated.csv`, `da_all_retest_multi_axes.csv` and
  `surrogates_{validated,snr_grid,catalogue,by_L,catalogue_by_language}.png` — the
  validated search (`search.py`).
- Inputs: the noise-and-SNR `snr_variants_per_task.csv` and holdout
  `headline_metrics.csv`, the gate's `above_random_scores.csv`, the scaling
  analysis's `rq1_fits.csv`.
- `…/finetasks_criteria.{csv,png}`, `finetasks_surrogates_scatter.{csv,png}`,
  `finetasks_surrogates_rank.csv`, `../finetasks_overlap.csv` — FineTasks'
  criteria on the ladder and every surrogate against DA-size
  (`finetasks_criteria.py`; moved from the benchmark-design folder on
  2026-09-23, the copies left
  under `rq09_benchmark_design/pretraining/predictivity/` are stale).
