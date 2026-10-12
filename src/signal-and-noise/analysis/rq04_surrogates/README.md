# Surrogates: which cheap statistic predicts decision accuracy?

## Research question

> Before the reference is trained, can a statistic computed on the proxy
> alone tell us whether the proxy's decision will match the reference's?
> First, which of the 22 SNR definitions of the noise-and-SNR analysis
> correlates best with **decision accuracy** across the ladder's languages,
> and does the answer survive a change of training seed?
>
> Then we look beyond SNR (the paper's surrogate analysis,
> [`documents/paper/sections/04_analysis.tex`](../../../../documents/paper/sections/04_analysis.tex)):
> the signal or the noise part alone, the proxy's early-checkpoint agreement,
> how well its scores fit a scaling trend and its margin above chance.
> Finally, we test 211 statistics from the literature and the AllenAI
> signal × noise grid (`catalogue.py`, `search.py`,
> [`literature.md`](literature.md)), per language, language count and
> benchmark.

<!-- BEGIN auto:highlight (snr_definition_postprocess.py --pool predictivity) -->
## Highlighted result

- **Global-best SNR definition (`predictivity`): `rel_std`.** Its mean Pearson r of log₁₀(SNR) with decision accuracy is **0.53** under DA-size (proxy → 1.7B, 50 languages), **0.54** under DA-ckpt (proxy sizes pooled, 50 languages) and 0.53 overall. DA-ckpt is led by `rel_std`/`rel_mpd`/`rel_dispersion` (≈ 0.54, one family: rel_spread), so we recommend the *family*, not an exact variant.
- **Per-language anchor: `bbpb_rfgm_include_base_44`.** It is the highest-SNR above-random benchmark in **9 of 50** languages (`rel_std` SNR @ 1.7B, leaving out `train_loss` and `bpb_macro`, which belong to no language). The language's own BPB, ungated and on its own noise scale, outranks that benchmark in 21 of the 50 languages that have both. The weakest variants overall are `projection`, `tukey`.
- **Seed holdout (predictivity_seeds_train → predictivity_seeds_test).** The Spearman ρ of the global variant ranking is **0.73** under DA-ckpt and **0.55** under DA-size, and the family-level per-language agreement is 50% / 100%. A ranking that does not survive the seed swap is dominated by noise, so only the *family* recommendation transfers.
<!-- END auto:highlight -->

## Headlines (ladder report 2026-10-10 04:31)

*Pool `predictivity`: seed 1904, every cell (L ∈ {1, 2, 8, 15, 30, 50}, the
deep, shallow and swiglu ladders, every data build), sizes 90M–1.7B, the gate
per (task, size) (rule 1), parent tasks, trained languages. The Pearson r of
log₁₀(SNR) with DA is computed per language over the 50 languages with ≥ 3
tasks (rule 8), then averaged.*

![The surrogate question in one figure](pretraining/predictivity/highlights.png)

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/highlights.csv)

**Key findings**

- **The relative-spread SNR definitions track decision accuracy best, as a
  block.** `rel_std`, `rel_mpd` and `rel_dispersion` read mean r 0.51–0.52
  with DA-size and 0.55–0.56 with DA-ckpt (0.53–0.54 overall,
  `top_variants_overall.csv`), ahead of `iqr` (0.51) and the eight
  dispersion and robust definitions (`aad`, `rms_deviation`, `dist_std`, `mpd`,
  `range`, `dispersion`, `quartile_deviation`, `mad`: 0.48–0.50 overall).
  The figure's all-task panels read 0.48–0.59 and 0.58–0.62 for their top
  six, both led by `rel_mpd`. The leader `rel_std` (0.52 / 0.56) wins by
  less than 0.01, and the depth definitions are null or anti-signals
  (`tukey` −0.02 / −0.12, `projection` −0.05 / −0.15).
- **No exact definition is best everywhere.** Per language the winner is
  one of 11 definitions under DA-size and one of 12 under DA-ckpt. The
  relative-spread family wins 26 of 50 languages under DA-size and 31 under
  DA-ckpt. Dispersion wins 7 under DA-size and 13 under DA-ckpt, and
  discrepancy 11 under DA-size.
- **On the few languages the seed holdout can read (one under DA-size, two
  under DA-ckpt), it keeps the global ranking, but the per-language picks
  agree only under DA-size.** The global ranking of the 22 definitions
  correlates across the seed split at Spearman ρ 0.72 under DA-ckpt and 0.55
  under DA-size (on the holdout DA-size is 175M → 600M). The per-language
  picks rest on English alone under DA-size, where both splits pick
  `dispersion_shifted`, and on English and Russian under DA-ckpt, where only
  Russian keeps the family (relative spread) and neither keeps the variant.
- **Beyond SNR, the early-checkpoint agreement and SNR relative std share
  the lead, and on the accuracy tasks alone SNR leads up to 350M.** Over
  1175–1360 benchmark tasks (816 of them twins at every rung) the early
  agreement reads Spearman ρ 0.53–0.60 with DA-size and SNR relative std
  0.53–0.59, the early agreement ahead at 90M, 175M and 1B. On the 359–544
  accuracy tasks alone SNR relative std reads 0.31–0.36, against 0.20–0.37
  for the early agreement and 0.20–0.38 for the inverted noise, which leads
  at 1B. Margin above chance (−0.04 to 0.04) and signal alone (−0.09 to 0.05)
  carry nothing on the benchmark tasks.
- **The proxy's own rank stability beats every SNR in the catalogue.** Over
  457 (benchmark, language) clusters, the window sign consistency reads ρ
  0.92 against DA-ckpt, and the proxy's mean DA-ckpt reads 0.85 against
  DA-goal and 0.81 against DA-size (ceilings 0.98 / 0.94 / 0.89). AllenAI's
  relative-std / checkpoint-noise SNR reads 0.75–0.86.
- **FineTasks' criteria separate reliable tasks only modestly.** On the
  846–913 accuracy benchmark tasks the three together pass 16 % (90M) to
  44 % (1B), against 39–56 % for our gate. Their passers read DA-size
  0.56–0.58 against 0.46–0.50 for the rest.

## Experimental setup

**Setup.**
- **Measured.** For each of 22 SNR definitions, we correlate log₁₀ SNR with decision accuracy (Pearson r) across the tasks of one language, on every design pair (multi-axis) and pooling the proxy sizes 90M–1B.
- **Models.** We use the seed-1904 runs of every cell, 90M–1.7B.
- **Tasks.** The tasks are those above chance at the proxy and, for DA-size, at 1.7B, plus the bBPB twins and the per-language BPB, which have no chance level. The four discrepancy definitions are undefined off [0, 1], so they read the accuracy tasks alone, a smaller population than the other 18. We also read the accuracy tasks and the bBPB twins apart (figure: app_rq04_surrogates_by_scoring), and the single-axis pairs alone (figure: app_rq04_surrogates_mono_axis). A language needs at least three tasks.
- **SNR.** Each definition divides a spread of the final scores over the design variants at a size by the checkpoint noise of the last 20 % of the runs, relative to the mean window score (to the mean depth for `tukey` and `projection`).
- **How.** We average the per-language correlation over the languages with at least three tasks (all 50 trained languages, 48–49 for the discrepancy definitions), for DA-size (proxy final against 1.7B final) and DA-ckpt (a proxy's nine early checkpoints, 10–90 %, against its own final).

- **Pools** (outputs under `pretraining/<pool>/`, see [RULES.md](../RULES.md),
  Definitions). `predictivity` is the headline: seed 1904, every cell of the
  grid (L ∈ {1, 2, 8, 15, 30, 50}, deep, shallow and swiglu, every data
  build), 90M–1.7B. `predictivity_seeds` adds every replicate seed as a
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
  variant trained there. The noise is the late-checkpoint noise over the noise
  window, 80/85/90/95/100 % of the run, measured since 2026-10-08 as the
  residual std around a line through it (rule 4). DA-size is proxy final →
  1.7B final (rule 9). DA-ckpt is a proxy's early checkpoints → its final,
  with the proxy sizes pooled and never the 1.7B run's own checkpoints (rule 11).
- **Tasks.** Benchmarks, their `rf_` twins and the per-language BPB
  (`bpb_<subset>`) take part like any task. `bpb_macro` and `train_loss`
  belong to no language and are left out (rule 7). The 22 definitions are grouped
  into families (dispersion / relative spread / discrepancy / robust /
  depth).
- **bBPB twins (populations move, rule 13).** The per-item store was rebuilt
  over every checkpoint of every seed-1904 ladder cell, so the 816 `bbpb_*`
  twins have an SNR, a DA-size and a DA-ckpt at every rung from 90M to 1.7B,
  and every pooled or per-proxy number below mixes them in. The seed
  replicates (seeds 64/313/28/1797, 20 models) now carry twins too, so the
  seed holdout mixes them in (98 English twins with a DA-size
  175M → 600M, and 116 twins with a DA-ckpt: 98 English, 18 Russian). The 3B cells sit outside these pools.

## Methodology

- **SNR per (task, size bucket)** is the noise-and-SNR table
  (`../rq03_noise_and_snr/pretraining/<pool>/snr_variants_per_task.csv`:
  22 variants, gated cells NaN, coverage per variant recorded there).
- **Decision accuracy** is joined from the decision-accuracy tables (`decision_acc_size_*`,
  `decision_acc_ckpt_*`, with the pair counts in `da_all_n_pairs_per_task_both_axes.csv`).
- **Ranking.** Per language, Pearson r of log₁₀(SNR) against DA over the
  (task, bucket) cells. The global variant is the highest mean r over
  languages and over both DA kinds (`top_variants_overall.csv`), and the
  per-language "most reliable benchmark" table reads that variant at the
  reference size.
- **Seed holdout.** The same per-language table on the replicate seeds
  (64/313/28/1797) and on seed 1904 of the same cells
  (`../rq03_noise_and_snr/pretraining/predictivity_seeds_train__vs__predictivity_seeds_test/headline_metrics.csv`).
  Agreement is counted over the languages that have a best variant on both
  splits, today 1 (DA-size) and 2 (DA-ckpt).

Hand-written numbers in this README are from the ladder-report snapshot
**2026-10-10 04:31** and the outputs of the 2026-10-10 refresh (detrended
checkpoint noise), prose re-read 2026-10-10.

<!-- BEGIN auto:results (snr_definition_postprocess.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq04_surrogates/snr_definition_postprocess.py --pool predictivity`.

**Global variant ranking.** The mean Pearson r of log₁₀(SNR) with DA across the trained languages with ≥ 3 tasks (rule 8, 50 languages under DA-size, 50 under DA-ckpt). DA-size is proxy final → 1.7B final only (the proxy-to-proxy scaling pairs are not DA-size, rule 9). DA-ckpt is a proxy size's early checkpoints → its final, with the proxy sizes 90M, 175M, 350M, 600M, 1B pooled and never the 1.7B run's own checkpoints (rule 11):

| variant | DA-size r | DA-ckpt r | overall |
|---|---|---|---|
| `rel_std` | 0.53 | 0.54 | 0.53 |
| `rel_mpd` | 0.53 | 0.54 | 0.53 |
| `rel_dispersion` | 0.52 | 0.52 | 0.52 |
| `iqr` | 0.51 | 0.50 | 0.50 |
| `aad` | 0.46 | 0.52 | 0.49 |
| `mpd` | 0.46 | 0.52 | 0.49 |
| `rms_deviation` | 0.46 | 0.52 | 0.49 |
| … |  |  |  |
| `tukey` | -0.03 | -0.14 | -0.09 |
| `projection` | -0.08 | -0.17 | -0.12 |

![SNR variants ranked by correlation with DA](pretraining/predictivity/top_variants_overall.png)

**Statistical power by pool.** Each pool's best variant (mean r over both DA kinds):

| pool | best variant (overall) | DA-size r | DA-ckpt r |
|---|---|---|---|
| `predictivity` (grid, seed 1904, every design) | `rel_std` | 0.53 | 0.54 |
| `predictivity_seeds` (every seed) | `rel_std` | 0.54 | 0.53 |

**Most reliable benchmark per language.** `rel_std` SNR @ 1.7B over the above-random benchmarks, with the language's own BPB SNR alongside (ungated, on its own noise scale). DA-size is undefined at the reference size itself, so DA-ckpt@1.7B is shown. `train_loss` and `bpb_macro` measure the whole mixture and are not a row (rule 7):

| lang | top benchmark | SNR | DA-ckpt@1.7B | BPB SNR |
|---|---|---|---|---|
| ar | `bbpb_global_piqa_parallel_cloze_apc_arab_leba` | 4.41 | 0.76 | 3.83 |
| az | `bbpb_rfgm_include_base_44_azerbaijani` | 10.98 | 0.93 | 7.67 |
| bg | `multiblimp_bul` | 3.43 | 0.76 | 4.51 |
| bn | `bbpb_rfgm_include_base_44_bengali` | 4.42 | 0.85 | 6.13 |
| bs | `bbpb_global_piqa_nonparallel_cloze_bos_latn` | 6.00 | 0.87 | 8.56 |
| ca | `bbpb_hellaswag_ca` | 7.67 | 0.98 | 7.24 |
| cs | `bbpb_multiblimp_ces` | 2.73 | 0.77 | 1.89 |
| da | `bbpb_hellaswag_da` | 3.95 | 0.85 | 3.51 |
| de | `bbpb_hellaswag_de` | 5.10 | 0.82 | 2.90 |
| el | `bbpb_rf_include_base_44_greek` | 2.54 | 0.56 | 2.21 |
| en | `hellaswag` | 5.00 | 0.80 | 13.60 |
| es | `include_v2_og_spanish_espa_a` | 7.27 | 0.72 | 5.53 |
| et | `bbpb_xcopa_et` | 12.99 | 0.91 | 8.93 |
| fa | `bbpb_rfgm_include_base_44_persian` | 5.71 | 0.80 | 4.07 |
| fi | `bbpb_include_v2_og_finnish_finland` | 3.36 | 0.73 | 4.15 |
| fr | `bbpb_hellaswag_fr` | 4.38 | 0.82 | 2.37 |
| he | `bbpb_rfgm_include_base_44_hebrew` | 6.78 | 0.91 | 6.94 |
| hi | `bbpb_hellaswag_hi` | 4.78 | 0.83 | 5.92 |
| hr | `bbpb_rfgm_include_base_44_croatian` | 8.68 | 0.94 | 7.02 |
| hu | `bbpb_include_v2_og_hungarian_hungary` | 3.88 | 0.70 | 2.65 |
| id | `bbpb_rf_m_mmlu_id` | 3.62 | 0.61 | 2.68 |
| it | `bbpb_rf_m_mmlu_it` | 3.34 | 0.56 | 2.33 |
| ja | `bbpb_rf_global_mmlu_full_ja` | 3.80 | 0.70 | 3.75 |
| ka | `bbpb_rfgm_belebele_kat_Geor` | 6.19 | 0.88 | 7.51 |
| kk | `bbpb_global_piqa_nonparallel_cloze_kaz_cyrl` | 9.03 | 0.98 | 9.52 |
| ko | `bbpb_rf_global_mmlu_full_ko` | 3.98 | 0.74 | 3.05 |
| lt | `bbpb_rfgm_include_base_44_lithuanian` | 6.43 | 0.89 | 6.90 |
| lv | `bbpb_rfgm_belebele_lvs_Latn` | 8.94 | 0.98 | 8.46 |
| ml | `bbpb_global_piqa_parallel_cloze_mal_mlym` | 6.06 | 0.89 | 7.76 |
| mr | `bbpb_rfgm_belebele_mar_Deva` | 6.49 | 1.00 | 10.31 |
| ms | `bbpb_include_v2_og_malay_malaysia` | 7.73 | 0.93 | 7.32 |
| ne | `bbpb_hellaswag_ne` | 6.85 | 0.85 | 10.88 |
| nl | `rf_m_mmlu_nl` | 3.82 | 0.66 | 2.75 |
| no | `norec_document_p2` | 4.50 | 0.62 | 3.10 |
| pl | `bbpb_multiblimp_pol` | 3.06 | 0.67 | 2.27 |
| pt | `bbpb_rf_include_base_44_portuguese` | 3.84 | 0.61 | 2.30 |
| ro | `bbpb_rf_m_mmlu_ro` | 3.27 | 0.63 | 2.87 |
| ru | `bbpb_hellaswag_ru` | 5.91 | 0.85 | 6.94 |
| sk | `bbpb_hellaswag_sk` | 4.19 | 0.96 | 5.67 |
| sl | `bbpb_multiblimp_slv` | 7.43 | 0.91 | 8.81 |
| sq | `bbpb_rfgm_include_base_44_albanian` | 10.10 | 0.91 | 9.27 |
| sr | `bbpb_hellaswag_sr` | 8.22 | 0.96 | 9.55 |
| sv | `bbpb_include_v2_og_swedish_sweden` | 4.14 | 0.63 | 3.85 |
| ta | `bbpb_rfgm_include_base_44_tamil` | 5.39 | 0.93 | 7.23 |
| th | `bbpb_xnli_th` | 2.60 | 0.77 | 3.29 |
| tr | `bbpb_rf_include_base_44_turkish` | 4.92 | 0.72 | 2.59 |
| uk | `bbpb_rf_m_mmlu_uk` | 2.91 | 0.48 | 1.51 |
| ur | `bbpb_rfgm_belebele_urd_Arab` | 7.26 | 0.98 | 10.14 |
| vi | `bbpb_rfgm_include_base_44_vietnamese` | 3.77 | 0.75 | 3.92 |
| zh | `bbpb_xstorycloze_zh` | 6.05 | 0.87 | 5.92 |

![Top-5 benchmarks per language by SNR](pretraining/predictivity/top_benchmarks_per_language.png)

**Seed generalization.** Holdout `predictivity_seeds_train` → `predictivity_seeds_test` (the ×3 cells only). A variant ranking with a low Spearman ρ here is dominated by noise, so we recommend the family that transfers, not the argmax:

| metric | DA-size | DA-ckpt |
|---|---|---|
| Spearman ρ on global variant ranking | 0.55 | 0.73 |
| Pearson r between splits (all cells) | 0.62 | 0.33 |
| Exact-variant agreement (per lang) | 0% | 0% |
| Family-level agreement (per lang) | 100% | 50% |
| Retention of train-best r on test | 95% | 99% |
<!-- END auto:results -->

GitHub: [top_variants_overall.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_variants_overall.png) · [top_variants_overall.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_variants_overall.csv) ·
GitHub: [top_benchmarks_per_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_benchmarks_per_language.png) · [top_benchmarks_per_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_benchmarks_per_language.csv) ·
[snr_variant_ranking.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_ranking.csv) ·
[best_variant_per_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/best_variant_per_language.csv)

**Key findings** (pool `predictivity`, 90M–1.7B, gated, 50 languages under
each DA kind, with the holdout rows on the ×3 cells only)

- **A block, not a winner.** The three relative-spread definitions span
  0.53–0.54 overall, and `rel_std` leads `rel_mpd` by 0.0002. With every
  replicate seed in the signal pool (`predictivity_seeds`) the leader stays
  `rel_std`, at 0.53 / 0.56, with `rel_mpd` second by 0.005.
- **The most reliable benchmark per language is a bBPB twin.** A `bbpb_*`
  task has the highest `rel_std` SNR at 1.7B in 44 of 50 languages, and an
  accuracy task in six (Danish MultiBLiMP, English, Italian and Portuguese
  HellaSwag, Spanish INCLUDE, Norwegian Nynorsk OpenBookQA). In 23 languages
  the language's own BPB ranks above every benchmark. INCLUDE (either
  version) and HellaSwag, in either scoring, are the top benchmark in 15 and
  13 languages and MultiBLiMP in 7. The twins now carry every checkpoint, so
  their rows are no longer provisional.
- **The seed holdout is too thin to read.** The family-level agreement
  (100 % under DA-size, 50 % under DA-ckpt) and the 100 % / 98 % retention
  are over 1 language (DA-size) and 2 (DA-ckpt). The cell-level Pearson r
  between the splits, 0.62 over 22 (definition, language) cells under
  DA-size and 0.29 over 44 under DA-ckpt, carries the same caveat.

**Follow-ups**

- **The top-benchmark table by scoring.** Split by scoring (accuracy vs
  bBPB), so that the accuracy anchor per language is visible while the twins
  take 44 of 50 rows.
- **A bootstrap over languages.** For the mean r of the top three
  definitions: they sit within 0.02 of each other, so only the family can be
  claimed.
- **Language counts beside the holdout percentages.** In the generated block
  (see the TODO below): the 100 % is 1 of 1 and the 50 % is 1 of 2.

### Statistics beyond SNR: setup

The truth is DA-size from the [decision-accuracy tables](../rq02_decision_accuracy/):
per task, the agreement between the ranking of the design variants at a proxy
size and at the 1.7B reference, as carried in the noise-and-SNR per-task table
(`snr_variants_per_task.csv`, `decision_acc_size_<proxy>`). The candidates come
from the same table (`snr_*`, `signal_*`, `noise_*`, `decision_acc_ckpt_*`),
from the gate's above-random scores (margin above chance) and from the
scaling fits (log-N R², refitted on the proxy rungs only).

The population per proxy size is the tasks with an SNR at that size (the
above-random survivors), with benchmarks and per-language BPB scored
separately. We compute the Spearman ρ between the candidate and DA-size per
proxy size (90M–1B), with the p-value and n. A cell needs at least 8 tasks,
and the noise part is inverted so that "higher is better" holds for every
candidate.

Population: pool `predictivity`. The benchmark n carries the bBPB twins
(rule 13): 1175, 1218, 1248, 1312 and 1360 tasks at 90M–1B, of which 816 at
every proxy are twins and 359–544 are accuracy tasks. The BPB rows are the 50
trained languages at every proxy.

<!-- BEGIN auto:surrogates (analyze.py --pool predictivity) -->
## Statistics beyond SNR (paper RQ3)

Numbers from the `predictivity` pool's rq03 table. Regenerate with `python analysis/rq04_surrogates/analyze.py --pool predictivity`. The population at a proxy size is the tasks above chance at the proxy and at 1.7B (rule 1). Each candidate is scored on the tasks where it has a value, so n differs per candidate and is given next to every ρ. The scaling-fit R² is rq01's log-N fit refitted on the rungs up to the proxy only (rule 11). It needs 3 rungs, so it starts at 350M. `bpb_macro` and `train_loss` are in neither population (rule 7).

- **benchmark tasks.** The strongest surrogate of DA-size (mean ρ over proxies) is `early-checkpoint agreement (10 %)` at 0.57, and the weakest is `signal alone (relative std)` at -0.06.
- **per-language bits per byte.** The strongest surrogate of DA-size (mean ρ over proxies) is `SNR, dist_std` at 0.36, and the weakest is `early-checkpoint agreement (10 %)` at 0.04.

**benchmark tasks.** Spearman ρ of the statistic with DA-size per proxy size, with n the number of tasks behind the ρ:

| metric | 90M ρ (n) | 175M ρ (n) | 350M ρ (n) | 600M ρ (n) | 1B ρ (n) |
|---|---|---|---|---|---|
| early-checkpoint agreement (10 %) | 0.58 (1274) | 0.62 (1316) | 0.59 (1351) | 0.54 (1407) | 0.56 (1461) |
| SNR, relative std | 0.55 (1274) | 0.53 (1316) | 0.60 (1351) | 0.53 (1407) | 0.53 (1461) |
| noise alone (relative std, inverted) | 0.46 (1274) | 0.44 (1316) | 0.56 (1351) | 0.44 (1407) | 0.52 (1461) |
| scaling-fit R² (proxy rungs only) |  |  | 0.49 (1269) | 0.45 (1313) | 0.49 (1350) |
| SNR, dist_std | 0.46 (1274) | 0.38 (1316) | 0.46 (1351) | 0.43 (1407) | 0.39 (1461) |
| SNR, discrepancy | 0.22 (368) | 0.28 (410) | 0.34 (445) | 0.31 (501) | 0.42 (555) |
| margin above chance | 0.09 (363) | 0.00 (405) | -0.03 (440) | 0.05 (496) | -0.02 (550) |
| signal alone (relative std) | -0.01 (1274) | 0.00 (1316) | -0.08 (1351) | -0.05 (1407) | -0.16 (1461) |

**per-language bits per byte.** Spearman ρ of the statistic with DA-size per proxy size, with n the number of tasks behind the ρ:

| metric | 90M ρ (n) | 175M ρ (n) | 350M ρ (n) | 600M ρ (n) | 1B ρ (n) |
|---|---|---|---|---|---|
| SNR, dist_std | 0.52 (50) | 0.62 (50) | 0.62 (50) | -0.40 (50) | 0.45 (50) |
| SNR, relative std | 0.39 (50) | 0.59 (50) | 0.51 (50) | -0.37 (50) | 0.29 (50) |
| noise alone (relative std, inverted) | 0.37 (50) | 0.31 (50) | 0.46 (50) | 0.13 (50) | 0.03 (50) |
| signal alone (relative std) | 0.37 (50) | 0.53 (50) | 0.46 (50) | -0.35 (50) | 0.25 (50) |
| scaling-fit R² (proxy rungs only) |  |  | 0.37 (50) | -0.60 (50) | 0.35 (50) |
| early-checkpoint agreement (10 %) | 0.48 (50) | 0.51 (50) | 0.12 (50) | -0.40 (50) | -0.53 (50) |

![Surrogates](pretraining/predictivity/rq3_surrogates.png)
<!-- END auto:surrogates -->

GitHub: [rq3_surrogates.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/rq3_surrogates.png) · [rq3_surrogates.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/rq3_surrogates.csv) ·
[facts.json](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/facts.json)

**Key findings** (pool `predictivity`, DA-size against 1.7B, gated, with the
benchmark n per cell in the table. The accuracy-only and twin-only ρ are
recomputed by hand from the noise-and-SNR per-task table with this script's
gate, without the `bbpb_` rows or with them alone.)

- **With the twins, the early-checkpoint agreement and SNR relative std are
  level.** Over 1175–1360 tasks they read ρ 0.53–0.60 (mean 0.56) and
  0.53–0.59 (mean 0.55). The early agreement is ahead at 90M, 175M and 1B and
  SNR relative std at 350M and 600M, with the scaling-fit R² at 0.42–0.46
  from 350M. On the 816 twins alone the early agreement reads 0.57–0.70 and
  SNR relative std 0.48–0.61.
- **On the accuracy tasks alone, SNR relative std leads up to 350M.** Over
  359–544 tasks it reads 0.31–0.36, the early-checkpoint agreement 0.20–0.37
  and the inverted noise 0.20–0.38. SNR relative std ties the inverted noise
  at 600M (0.32). At 1B the inverted noise leads (0.38), ahead of the early
  agreement (0.37) and SNR discrepancy (0.35), with SNR relative std at 0.31.
- **On the benchmark tasks, margin above chance and signal alone predict
  nothing.** They read ρ −0.04 to 0.04 (354–539 accuracy tasks) and −0.09 to
  0.05 at every proxy. There the SNR's predictive part is its noise, which
  reads 0.41–0.53 inverted.
- **Per-language BPB flips sign at 600M.** SNR `dist_std` reads 0.44–0.60 at
  90M, 175M, 350M and 1B but −0.36 at 600M (50 languages). SNR relative std,
  signal alone and the scaling-fit R² flip there too. The inverted noise
  instead falls from 0.39 (90M–350M) to 0.15 at 600M and −0.07 at 1B, and the
  early agreement is negative at both 600M and 1B (−0.39, −0.56).

**Follow-ups**

- **Benchmark rows by scoring.** The benchmark rows split by scoring
  (accuracy vs bBPB) in the generated table, so that the accuracy-only
  reading above comes from the pipeline rather than by hand.
- **The 600M BPB row.** Which cells and languages drive the sign flip, and
  whether the 600M DA-size against 1.7B is itself an outlier in the
  decision-accuracy tables.

### Statistics beyond SNR, split by scoring (added 2026-10-09)

The same statistics, gate and x axis as `rq3_surrogates.png`, with the
benchmark panel split into the accuracy tasks (359–544 per proxy) and their
816 bBPB twins. The per-language BPB panel is unchanged
(`analyze.py --pool predictivity` writes it beside the pooled figure).

![Surrogates by scoring](pretraining/predictivity/rq3_surrogates_by_scoring.png)

GitHub: [rq3_surrogates_by_scoring.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/rq3_surrogates_by_scoring.png) · [rq3_surrogates_by_scoring.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/rq3_surrogates_by_scoring.csv)

**Key findings** (pool `predictivity`, DA-size against 1.7B, gated, the
2026-10-10 per-task table)

- **The ρ ≈ 0.6 level and the early-agreement lead come from the bBPB
  twins.** The early-checkpoint agreement reads 0.57–0.70 on the twins and
  0.20–0.37 on the accuracy tasks. SNR relative std reads 0.48–0.61 against
  0.31–0.36, the inverted noise 0.45–0.57 against 0.20–0.38, and the
  scaling-fit R² 0.37–0.50 against 0.28–0.32. On the accuracy tasks SNR
  relative std leads at 90M–350M and ties the inverted noise at 600M. At 1B
  the inverted noise leads (0.38), with the early agreement second (0.37).
- **Signal alone and the margin carry nothing in either population.** Signal
  reads −0.21 to 0.04 on accuracy and −0.14 to −0.04 on bBPB. The margin
  reads −0.04 to 0.04 (accuracy only, since the twins have no chance level).
- **Against the earlier write-ups.** Before the twins (2026-10-05 table) the
  benchmark panel was accuracy only and read SNR relative std 0.05–0.15,
  early agreement 0.03–0.23 and inverted noise 0.20–0.34 (the leader at every
  proxy). On the 2026-10-08 table the accuracy tasks alone read 0.35–0.43 /
  0.23–0.40 / 0.19–0.40 without any twin, and on the current one 0.31–0.36 /
  0.20–0.37 / 0.20–0.38. On the same 231–355 tasks, the 2026-10-07 pool
  redefinition (every seed-1904 cell, schemes and swiglu included) lifted SNR
  relative std to 0.23–0.40, and the detrended noise (2026-10-08) to
  0.40–0.49. The switch to acc_norm, the ungated non-accuracy tasks and the
  new evals moved it by about 0.01. The detrended noise is also what makes
  the twins' SNR strong: with the raw SD the twins read 0.20–0.46 and the
  pooled panel 0.25–0.52. The corrected chance levels of the 2026-10-08 mask
  change no ρ.
- **The per-language BPB flip at 600M is not a bBPB effect.** That panel has
  no twins. On the 2026-10-05 table (34 languages) SNR relative std read
  +0.85 at 600M, and it reads −0.34 since the 50-language, every-cell pool.

**Follow-ups**

- **By-scoring tables in the generated block.** The numbers above would then
  be read from the pipeline rather than recomputed for this note.

### Statistics beyond SNR on single-axis pairs (added 2026-10-10)

The same statistics, gate and tasks as `rq3_surrogates.png`, with DA-size and
the early-checkpoint agreement read over the mono-axis pairs (rq02's
`da_all_per_task_both_axes.csv`) instead of the multi-axis set that the
noise-and-SNR table carries. The paper's rq3 keeps every pair, and this
figure is its appendix twin. Regenerate with
`python analysis/rq04_surrogates/analyze.py --pool predictivity --axes mono-axis`
(written to `pretraining/predictivity/mono_axis/`, no auto block).

![Surrogates on mono-axis pairs](pretraining/predictivity/mono_axis/rq3_surrogates.png)

GitHub: [rq3_surrogates.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/mono_axis/rq3_surrogates.png) · [rq3_surrogates.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/mono_axis/rq3_surrogates.csv) ·
[rq3_surrogates_by_scoring.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/mono_axis/rq3_surrogates_by_scoring.csv)

**Key findings** (pool `predictivity`, DA-size against 1.7B over mono-axis
pairs, gated, the 2026-10-10 tables)

- **Every statistic with a signal is weaker on the single-axis pairs, and SNR
  relative std leads at every proxy.** Over 1174–1360 benchmark tasks (816 of
  them twins) SNR relative std reads ρ 0.38–0.46 against 0.53–0.59 over every
  pair. The early-checkpoint agreement falls to 0.24–0.36 (0.53–0.60), the
  inverted noise to 0.26–0.39 (0.41–0.53) and the scaling-fit R² to
  0.24–0.35 (0.42–0.46).
- **On the twins the early agreement no longer leads.** On the 816 bBPB
  twins it reads 0.21–0.40 and leads only at 175M, while SNR relative std
  reads 0.31–0.44 and leads from 350M on. On the 358–544
  accuracy tasks SNR relative std reads 0.22–0.27 and leads at 90M and 175M,
  and the inverted noise leads from 350M (0.26–0.31).
- **The per-language BPB flip at 600M stays.** Every statistic but the
  inverted noise is negative at 600M (−0.44 to −0.68 over 50 languages), and
  the early agreement is negative from 350M on.

**Follow-ups**

- **A mono-axis auto block.** The numbers above are read by hand from
  `mono_axis/rq3_surrogates.csv` and `mono_axis/rq3_surrogates_by_scoring.csv`.
  A generated block would keep them in step with each refresh.

### FineTasks' criteria: setup

A selection rule computed on one size, with no larger model in sight, is the
practical alternative to SNR. FineTasks (Kydlíček, Penedo et al., 2024)
selects tasks per language with four such statistics.

`finetasks_criteria.py`
(moved here from the benchmark-design folder on 2026-09-23, outputs
`pretraining/predictivity/finetasks_*`) computes them per (task, size) on the
`predictivity` pool: every design variant at each size at seed 1904 (every L,
ladder and data build, up to 32 runs per task and size), on the ten evaluated
tenths, which every task row now has. It judges them by DA-size against the
1.7B reference, which the criteria never see.

The gate is not applied, because non-random is one of the criteria under
test, but the mask is carried. Every candidate is scored as Spearman ρ with
DA-size across tasks, the way the SNR definitions above are. Their "noise" is
the spread across recipes at one step, which this framework calls signal, so
their SNR is the inverse of ours and the two should not be expected to agree.

Population: 846–913 accuracy benchmark tasks per size plus the 816 bBPB twins
at every size (1662–1729 rows, rule 13). The twins are lower-is-better and
have no chance level: none of their rows passes non-random and one in 4896
passes monotonicity. They therefore lower the generated table's pass shares
at every size and sit in its "fail" column.

<!-- BEGIN auto:finetasks-criteria (finetasks_criteria.py --pool predictivity) -->
## FineTasks' criteria on the ladder

FineTasks selects tasks with four statistics computed on single-seed runs at one size (monotonicity, a cross-run SNR, a non-random margin, consecutive-step ordering). We compute them per (task, size) on the `predictivity` variants and judge them by DA-size against 1.7B, which the criteria never see. The columns give the share of tasks passing each criterion, all three, and our gate, then the mean DA-size of passers vs failers, then the Spearman ρ of each statistic with DA-size (monotonicity / SNR / non-random / ordering). Regenerate with `python analysis/rq04_surrogates/finetasks_criteria.py --pool predictivity`.

| size | tasks | monotone | SNR > 20 | non-random | all three | our gate | DA-size pass vs fail | ρ with DA-size |
|---|---|---|---|---|---|---|---|---|
| 90M | 1844 | 10% | 63% | 25% | 8% | 40% | 0.57 [148] vs 0.57 [1696] | -0.50 / +0.12 / +0.11 / +0.60 |
| 175M | 1844 | 14% | 63% | 26% | 11% | 44% | 0.56 [209] vs 0.56 [1635] | -0.39 / +0.14 / +0.21 / +0.51 |
| 350M | 1844 | 18% | 64% | 28% | 16% | 48% | 0.56 [288] vs 0.57 [1556] | -0.46 / +0.25 / +0.25 / +0.56 |
| 600M | 1844 | 23% | 63% | 30% | 20% | 53% | 0.56 [360] vs 0.57 [1484] | -0.35 / +0.19 / +0.26 / +0.51 |
| 1B | 1844 | 26% | 63% | 33% | 23% | 59% | 0.58 [424] vs 0.59 [1420] | -0.32 / +0.29 / +0.27 / +0.55 |
| 1.7B | 1844 | 29% | 62% | 35% | 25% | 64% | — | — / — / — / — |

![FineTasks criteria](pretraining/predictivity/finetasks_criteria.png)

![DA against each surrogate](pretraining/predictivity/finetasks_surrogates_scatter.png)
<!-- END auto:finetasks-criteria -->

*`finetasks_criteria.png`: (a) the share of tasks passing each criterion and
all three, with our gate. (b) Each criterion's Spearman correlation with
DA-size across tasks. (c) DA-size of the tasks passing all three against the
rest. (d) Panel (a) with the share of tasks clearing the two decision-accuracy
cuts at that size (DA-size against the reference ≥ 0.66, median DA-ckpt of
the size's own run ≥ 0.66). (e) Panel (b) with the 22 SNR definitions behind
it and, for scale, the proxy ranking's own ρ and τ_b against the reference's
(an oracle) and the scaling analysis's ρ of score with size. (f) The
FineTasks picks with a counterpart in our registry, originals and cloze twins.
`finetasks_surrogates_scatter.png`: DA-size against each surrogate, one point
per (task, proxy size) coloured by the proxy. The panels are FineTasks' four
criteria, our gate, the ρ of score with size, the three SNR definitions with
the largest |Spearman| here, the proxy ranking's own ρ (the ceiling), ten
further reference-free candidates (ordering in the noise window, the 10 % and
50 % rankings against the final, the proxy's own DA-ckpt, DA between the two
rungs below the proxy, relative signal and noise, cross-variant std, margin
over chance, item count) and ten trajectory and checkpoint statistics
(`r2_trajectory`, `r2_traj_own`, `gain_half`, `gain_last`, `tau_stable_from`,
`window_tau`, `da_ckpt_f50`, `da_ckpt_f90`, `late_slope`, `rank_var`), with
Pearson r, Spearman ρ and the cell count in each corner. The pooled ranking is
in `finetasks_surrogates_rank.csv`, the FineTasks picks in
`finetasks_overlap.csv` (folder root, written 2026-09-23).*

**Key findings** (pool `predictivity`, DA-size against 1.7B, ungated. The
accuracy-only numbers are read from `finetasks_criteria.csv` without the
`bbpb_` rows.)

- **The composite selects better-than-average tasks but not reliable ones.**
  On the 846–913 accuracy tasks, monotonicity ≥ 0.5, cross-run SNR > 20 and
  best margin > 3 std together pass 16 / 23 / 31 / 37 / 44 % of tasks at
  90M / 175M / 350M / 600M / 1B, below our gate's 39 / 43 / 47 / 50 / 56 %.
  Their passers read DA-size 0.56–0.58 against 0.46–0.50 for the rest.
- **Their SNR criterion does not discriminate, and monotonicity does the
  work.** The SNR > 20 cut passes 62–64 % of the accuracy tasks at every
  proxy, while monotonicity rises from 19 % (90M) to 50 % (1B).
- **The four statistics correlate weakly with DA-size on the accuracy
  tasks.** Spearman ρ is 0.13–0.41 per proxy (0.13–0.25 at 90M, 0.25–0.41 at
  1B). The negative monotonicity of the generated table at every size (−0.33
  to −0.47) comes from the bBPB twins (−0.51 to −0.62 on them alone). Pooled
  over the proxies on the accuracy tasks alone it reads +0.30.
- **Pooled, the proxy's own run is the best reference-free signal.**
  `finetasks_surrogates_rank.csv` (twins included) puts the proxy's median
  DA-ckpt at 0.60, the DA between the two rungs below at 0.56, its own
  trajectory R² at 0.55 and its DA-ckpt at 50 % of the run at 0.54. An SNR
  definition reaches at most 0.53 (`rel_mpd`), the item count 0.32, our gate
  0.21 and FineTasks' SNR 0.17. On the accuracy tasks alone the median
  DA-ckpt drops to 0.37, level with SNR `rel_mpd` (0.34), the item count
  (0.32) and the DA between the two rungs below (0.32).
- **Spread across variants is not agreement with the reference.** The
  relative signal reads −0.00 and the cross-variant std 0.15 pooled, while
  the relative noise reads −0.43. The statistics that see the reference
  remain far above (the proxy ranking's own τ_b 0.99, the full-trajectory R²
  0.54).
- **The format FineTasks chose matters more than its thresholds.** Of their
  96 picks, 55 exist in our registry and 35 have results at 1B. There 54 % of
  the originals pass their criteria on our ladder, against all 13 cloze
  twins, and the twins also read the higher mean DA-size (0.63 against 0.60).

**Follow-ups**

- **Bootstrap the ρ over tasks** (a 90 % interval per statistic). On the
  accuracy tasks the DA-ckpt median, the lower-rung DA and the item count sit
  within 0.05 of each other, so we claim the family ("persistence of the
  proxy's own ranking"), not a winner.
- **A per-size and per-scoring pooled ranking** (accuracy vs bBPB), as the
  `_b` panels do for the SNR definitions. The pooled Spearman mixes the
  twins, whose monotonicity and DA-ckpt behave differently, with the
  accuracy tasks.
- **Cross-variant spread per family.** The spread against DA-size per family
  would show whether the `rf_` twins (large spread, re-sorting with scale,
  [the gate's figure 3](../rq00_gate_and_curves/README.md#3-the-reformulated-twins-move-whole-families-across-the-gate))
  explain why spread at the proxy does not mean agreement with the reference.

GitHub: [finetasks_criteria.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_criteria.png) · [finetasks_criteria.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_criteria.csv) ·
GitHub: [finetasks_surrogates_scatter.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_surrogates_scatter.png) · [finetasks_surrogates_scatter.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_surrogates_scatter.csv) ·
[finetasks_surrogates_rank.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/finetasks_surrogates_rank.csv) ·
[finetasks_overlap.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/finetasks_overlap.csv)

<!-- BEGIN auto:panels (panels.py --pool predictivity) -->
## Per benchmark and per language

The rankings above, without the aggregation (`predictivity` pool). A surrogate subplot needs 8 tasks at a proxy size. Regenerate with `python analysis/rq04_surrogates/panels.py --pool predictivity`. In every grid white means "no value" and grey "filtered out by the gate". Each figure's table sits next to it under the same name.

![rq04 in one figure](pretraining/predictivity/highlights.png)

![SNR definition per language](pretraining/predictivity/snr_definition_by_language.png)

![Surrogates per benchmark](pretraining/predictivity/surrogates_by_benchmark.png)

![Surrogates per language](pretraining/predictivity/surrogates_by_language.png)

**Per language count.** From rq02's `da_all_by_L_per_task_multi_axes.csv`: pairs of design variants sharing the L, on the tasks in the languages every variant at the L trains on (the intersection of the L's lists, English always). A level counts when it holds at every larger level with a value. The thresholds are DA ≥ 0.75 and, for an SNR definition to track DA, ρ ≥ 0.3. An L with fewer than 3 pairs against 1.7B is named in the figure's caption:

![Smallest safe level per measurement and L](pretraining/predictivity/min_level_by_L.png)

![the same as lines](pretraining/predictivity/min_level_by_L_lines.png)

![Smallest size at which an SNR definition tracks DA, per L](pretraining/predictivity/snr_variant_min_size_by_L.png)

![the same as lines](pretraining/predictivity/snr_variant_min_size_by_L_lines.png)

**Version B: every pair pooled, with the size axis instead of the language count.** From `da_all_pooled_per_task_multi_axes.csv` at ten checkpoints. The population is every design-variant pair of the pool on the tasks of the languages each cell trains (rule 2), parent tasks only (rule 6), gated at the proxy and at 1.7B. The benchmark mean's task count per size differs with the gate (rule 13) and is given in each figure's caption and on the DA-at-1C panel:

![Earliest checkpoint per proxy size and DA at 1C](pretraining/predictivity/min_level_by_L_b.png)

![the same as lines](pretraining/predictivity/min_level_by_L_lines_b.png)

![Spearman rho of each SNR definition with DA per proxy size](pretraining/predictivity/snr_variant_min_size_by_L_b.png)

![the same as lines](pretraining/predictivity/snr_variant_min_size_by_L_lines_b.png)

**Version per FLOPs.** Every (proxy size, checkpoint) cell at its training compute, with the same population as version B:

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
and at 1.7B, with the populations per figure as stated in the generated block)

- **Per L, an SNR definition rarely tracks DA-size.** 111 of the 132
  (definition, L) cells of `snr_variant_min_size_by_L` never reach ρ ≥ 0.3
  under DA-size, against only 40 under DA-ckpt. Per-L DA rests on few pairs,
  so this reads the lattice of possible DA values as much as the definitions.
- **Pooled over every pair (version B), the best definition reads ρ
  0.53–0.59 with DA-size on the benchmarks at every proxy.** The median
  definition reads 0.37–0.44, and with DA-ckpt the best reads 0.76–0.83. Both
  bests peak at 350M, and `rel_mpd` is the DA-size best at every proxy.
- **The per-language BPB flip at 600M shows here too.** The median
  definition's ρ with DA-size on BPB is 0.47–0.57 at 90M–350M and 0.42 at
  1B, but −0.28 at 600M.
- **The per-FLOPs line zigzags by construction.** It joins a bigger model's
  first (peak-LR) checkpoint to a smaller model's annealed final.

**Follow-ups**

- **SNR over the L's own variants.** The L-dependence of SNR with SNR
  computed over the L's own variants (degenerate below three variants),
  instead of the pooled signal.
- **One per-FLOPs line per proxy size.** With the annealed points marked,
  plus the compute frontier (the best DA reachable at or below each compute),
  for the paper.

## Surrogates from the literature: summary

`catalogue.py` computes 211 statistics read on the proxy alone, except
`pseudo_ref_da`, which reads the 1B rung below the reference (labelled so).
They are the literature catalogue ([`literature.md`](literature.md), with
every definition, its source and how our port differs from it in the paper's
appendix, `documents/paper/sections/app_06_snr_new.tex`), every AllenAI signal over
every noise (22 × 6, the k-fold benchmark noise included), and each signal
and noise alone.

`search.py` scores them against 48 truths from the decision-accuracy
tables (DA / τ-b / ρ for DA-size, DA-goal and DA-ckpt, both pair sets, every
pair or the pairs sharing one L) on all the data. It uses a Spearman ρ over
one point per (benchmark, language) cluster, a permutation p below 30
clusters, Benjamini–Hochberg over every configuration, and a split-half
confirmation as an extra. The numbers are in the generated block below.

Population: pool `predictivity`, proxies 90M–1B, cells above chance
(rule 1), up to 472 (benchmark, language) clusters over every benchmark (457
for DA-ckpt). A cluster holds a benchmark's original, its `rf_` twins and
their bBPB twins, which now carry every checkpoint at every proxy, so the
DA-goal and DA-ckpt rows include them in full.

**Key findings**

- **The ceiling.** A truth re-read at 90 % keeps ρ 0.89 (DA-size), 0.94
  (DA-goal) and 0.98 (DA-ckpt) with itself on the benchmarks (multi-axis).
- **No surrogate reaches it, and the proxy's rank stability comes closest.**
  The window sign consistency reads 0.92 against DA-ckpt (the early-vs-final
  agreements, consecutive-checkpoint Kendall τ included, are left out of
  DA-ckpt as circular). The proxy's mean DA-ckpt reads 0.81 against DA-size,
  just ahead of `bpb_rank_agreement` (0.80), and 0.85 against DA-goal, ahead
  of `sign_persistence` (0.84) and the consecutive τ (0.83). That is 91–94 %
  of the ceiling. On mono-axis pairs the best read 0.83, 0.70 and 0.65
  (DA-ckpt, DA-goal, DA-size).
- **SNR is close behind.** AllenAI's `rel_std` / checkpoint noise reads
  0.75, 0.79 and 0.86 on DA-size, DA-goal and DA-ckpt (16th, 12th and 11th
  of the 210, 210 and 203 surrogates). The best grid combination,
  `mpd / ckpt_abs` (tied with `rel_mpd / ckpt_rel`), reads 0.76, 0.80 and
  0.87, about 0.01 above it.
- **As the SNR's denominator, the detrended checkpoint noise beats the
  k-fold noise.** Relative k-fold noise beats relative checkpoint noise for
  only 6, 8 and 8 of the 22 signals (DA-size, DA-goal, DA-ckpt), although
  alone it still reads clearly stronger (−0.47, −0.54, −0.60 against −0.40,
  −0.40, −0.46). `n_items` alone reads 0.35, 0.38 and 0.47.
- **Within a stratum the SNR tracks DA-size less well than pooled, with three
  exceptions** (`surrogate_correlations.csv`, DA-size, multi-axis, every L,
  the paper's Table `tab:sur-corr-strata`, written by
  `documents/paper/sections/make_surrogate_tables.py`). AllenAI's `rel_std` /
  checkpoint-noise SNR reads 0.75 over 472 clusters and 0.53–0.61 within one
  proxy size (0.53 at 90M). It reads 0.62 / 0.61 / 0.77 / 0.75 within the L8
  / L15 / L30 / L50 language tiers, and from −0.18 (INCLUDE v2 English, 41
  clusters) to 0.82 (HellaSwag, 26) within one of the table's twelve
  benchmarks. The three exceptions are HellaSwag (0.82), MultiBLiMP (0.78)
  and the L30 tier (0.77). On per-language BPB it reads 0.39 (50). The best
  of the 132 SNR ratios per stratum reaches 0.93 on HellaSwag, 0.66 on the
  L8 tier and 0.68 on BPB (where 72 ratios have a value), best-of-many values
  on small strata.
- **Signal alone carries almost nothing.** Range, `dist_std` and `mpd` across
  variants read 0.06 to 0.21 against the three DA truths, and the depth
  signals are negative (−0.18 to −0.45).
- **Particular cases reach |ρ| 0.95–0.99, mostly on single languages or
  benchmarks.** All but one hold 10–26 clusters (the exception, sign
  consistency on the L30 tier against DA-ckpt at 0.95, holds 121). The
  `reliable (on the truth)` subsets are selected on the truth, so their ρ is
  conditional on it.
- **Language count.** Over pairs of variants sharing L, mean DA-size is 0.61
  / 0.50 / 0.49 / 0.55 / 0.56 / 0.60 at L1 / 2 / 8 / 15 / 30 / 50 (134–1371
  tasks), with no monotone trend.
- **Split-half check.** 1872 of 5609 configurations chosen on one half hold
  on the other at q < 0.05. The strongest is the window sign consistency on
  the L8 tier against DA-ckpt (0.73 on 169 held-out tasks).

**Follow-ups**

- **Catalogue rows by scoring.** The catalogue rows split by scoring
  (accuracy vs bBPB), so that the rank-stability lead can be read on accuracy
  alone. The twins now carry every checkpoint, so nothing blocks it.
- **A per-proxy version of the strongest surrogates.** The split-half
  winners concentrate on the L8 tier, the late checkpoints and the 90M–350M
  proxies.

<!-- BEGIN auto:catalogue (search.py --pool predictivity) -->
## A catalogue of surrogates beyond SNR

Numbers from the `predictivity` pool. Regenerate with `python analysis/rq04_surrogates/catalogue.py --pool predictivity` then `search.py --pool predictivity`. Surrogates are read on the proxy alone (rule 11), except `pseudo_ref_da`, which reads the 1B rung. The truths are rq02's DA (DA-size, DA-goal and DA-ckpt at every early checkpoint of the proxy, both pair sets, every pair or the pairs sharing L) and Kendall τ-b / Spearman ρ on the same rankings, over cells above chance (rule 1). ρ is a Spearman over one point per (benchmark, language) cluster, pooled over proxies and checkpoints. Populations differ per configuration (rule 13), and n is in the CSVs. Definitions and sources are in [`literature.md`](literature.md), the method in `search.py`'s docstring.

**Main analysis (all the data).** 856,301 configurations (truths × subsets × surrogates, the threshold filters included). 555,049 of them have a p-value (≥ 10 units), and **120,684 hold at BH q < 0.05**. A best-of-many ρ is optimistic even when it is significant. The extra below measures by how much.

**The ceiling.** Each truth against itself re-read at 90 % (benchmarks):

| truth | ρ | ρ over cells | units |
|---|---|---|---|
| DA-size da (multi-axis) | 0.90 | 0.83 | 475 |
| DA-goal da (multi-axis) | 0.95 | 0.83 | 475 |
| DA-ckpt da (multi-axis) | 0.98 | 0.88 | 475 |
| DA-size tau_b (multi-axis) | 0.89 | 0.81 | 475 |
| DA-size rho (multi-axis) | 0.90 | 0.83 | 475 |
| DA-goal tau_b (multi-axis) | 0.94 | 0.81 | 475 |
| DA-goal rho (multi-axis) | 0.94 | 0.82 | 475 |
| DA-ckpt tau_b (multi-axis) | 0.97 | 0.86 | 475 |
| DA-ckpt rho (multi-axis) | 0.97 | 0.87 | 475 |

**Strongest surrogate per truth**, every benchmark:

| truth | surrogate | ρ | q | ρ over cells | ρ within stratum | units |
|---|---|---|---|---|---|---|
| DA-ckpt da (mono-axis, pairs within L1) | `snr__rel_dispersion__ckpt_abs` | 0.81 | 7.7e-07 | 0.07 | 0.07 | 30 |
| DA-ckpt da (mono-axis, pairs within L15) | `snr__star_discrepancy_shifted__kfold_rel` | 0.51 | 3.5e-12 | 0.15 | 0.15 | 184 |
| DA-ckpt da (mono-axis, pairs within L2) | `n_items` | 0.83 | 3.8e-04 | 0.30 | 0.31 | 24 |
| DA-ckpt da (mono-axis, pairs within L30) | `sign_consistency_window` | 0.55 | 2.8e-26 | 0.22 | 0.22 | 342 |
| DA-ckpt da (mono-axis, pairs within L50) | `window_kendall` | 0.73 | 1.2e-78 | 0.32 | 0.32 | 475 |
| DA-ckpt da (mono-axis, pairs within L8) | `tie_rate_items` | -0.53 | 1.5e-08 | -0.14 | -0.15 | 115 |
| DA-ckpt da (mono-axis) | `window_kendall` | 0.82 | 2.6e-112 | 0.45 | 0.46 | 475 |
| DA-ckpt da (multi-axis, pairs within L1) | `snr__rel_dispersion__ckpt_abs` | 0.77 | 5.8e-06 | 0.09 | 0.09 | 30 |
| DA-ckpt da (multi-axis, pairs within L15) | `snr__star_discrepancy_shifted__kfold_rel` | 0.54 | 6.1e-14 | 0.21 | 0.21 | 184 |
| DA-ckpt da (multi-axis, pairs within L2) | `n_items` | 0.83 | 3.8e-04 | 0.36 | 0.37 | 24 |
| DA-ckpt da (multi-axis, pairs within L30) | `sign_consistency_window` | 0.65 | 5.1e-40 | 0.32 | 0.31 | 342 |
| DA-ckpt da (multi-axis, pairs within L50) | `bpb_rank_agreement` | 0.83 | 2.2e-118 | 0.43 | 0.44 | 475 |
| DA-ckpt da (multi-axis, pairs within L8) | `tie_rate_items` | -0.59 | 6.4e-11 | -0.20 | -0.20 | 115 |
| DA-ckpt da (multi-axis) | `sign_consistency_window` | 0.92 | 1.3e-191 | 0.62 | 0.65 | 475 |
| DA-ckpt rho (multi-axis) | `sign_consistency_window` | 0.90 | 1.4e-165 | 0.59 | 0.61 | 475 |
| DA-ckpt tau_b (multi-axis) | `mde_resolved` | 0.91 | 4.3e-176 | 0.54 | 0.56 | 475 |
| DA-goal da (mono-axis, pairs within L1) | `snr__rel_dispersion__projection_depth` | 0.71 | 1.1e-04 | 0.13 | 0.13 | 30 |
| DA-goal da (mono-axis, pairs within L15) | `snr__star_discrepancy_shifted__kfold_abs` | 0.42 | 9.6e-08 | 0.13 | 0.13 | 179 |
| DA-goal da (mono-axis, pairs within L2) | `snr__star_discrepancy_shifted__tukey_depth` | 0.63 | 6.3e-03 | 0.23 | 0.22 | 24 |
| DA-goal da (mono-axis, pairs within L30) | `gain_over_noise` | 0.35 | 4.8e-10 | 0.14 | 0.13 | 342 |
| DA-goal da (mono-axis, pairs within L50) | `consecutive_kendall` | 0.73 | 1.5e-77 | 0.33 | 0.33 | 475 |
| DA-goal da (mono-axis, pairs within L8) | `tie_rate_items` | -0.33 | 2.8e-03 | -0.09 | -0.09 | 110 |
| DA-goal da (mono-axis) | `consecutive_kendall` | 0.72 | 1.4e-75 | 0.40 | 0.40 | 475 |
| DA-goal da (multi-axis, pairs within L1) | `snr__rel_dispersion__tukey_depth` | 0.69 | 2.2e-04 | 0.12 | 0.13 | 30 |
| DA-goal da (multi-axis, pairs within L15) | `tie_rate_items` | -0.50 | 4.4e-11 | -0.22 | -0.22 | 174 |
| DA-goal da (multi-axis, pairs within L2) | `gain_over_noise` | 0.65 | 7.7e-04 | 0.18 | 0.17 | 30 |
| DA-goal da (multi-axis, pairs within L30) | `pseudo_ref_da` | 0.56 | 2.6e-28 | 0.25 | 0.26 | 342 |
| DA-goal da (multi-axis, pairs within L50) | `consecutive_kendall` | 0.80 | 6.8e-106 | 0.44 | 0.45 | 475 |
| DA-goal da (multi-axis, pairs within L8) | `n_pairs` | 0.35 | 2.4e-04 | 0.07 | 0.07 | 136 |
| DA-goal da (multi-axis) | `da_ckpt_mean` | 0.88 | 2.5e-150 | 0.61 | 0.61 | 475 |
| DA-goal rho (multi-axis) | `da_ckpt_mean` | 0.86 | 9.7e-136 | 0.57 | 0.57 | 475 |
| DA-goal tau_b (multi-axis) | `da_ckpt_mean` | 0.86 | 4.0e-137 | 0.55 | 0.56 | 475 |
| DA-size da (mono-axis, pairs within L1) | `snr__rel_dispersion__projection_depth` | 0.68 | 3.0e-04 | 0.12 | 0.12 | 30 |
| DA-size da (mono-axis, pairs within L15) | `n_pairs` | -0.31 | 3.7e-05 | -0.11 | -0.11 | 214 |
| DA-size da (mono-axis, pairs within L2) | `rung_sign_consistency` | 0.66 | 5.9e-04 | 0.18 | 0.18 | 30 |
| DA-size da (mono-axis, pairs within L30) | `pseudo_ref_da` | 0.30 | 1.2e-07 | 0.15 | 0.15 | 342 |
| DA-size da (mono-axis, pairs within L50) | `consecutive_kendall_late` | 0.65 | 2.8e-55 | 0.33 | 0.33 | 475 |
| DA-size da (mono-axis, pairs within L8) | `projected_flip_rate` | -0.23 | 3.0e-02 | -0.05 | -0.05 | 136 |
| DA-size da (mono-axis) | `bpb_rank_agreement` | 0.65 | 2.0e-56 | 0.41 | 0.41 | 475 |
| DA-size da (multi-axis, pairs within L1) | `snr__rel_dispersion__tukey_depth` | 0.64 | 1.1e-03 | 0.12 | 0.12 | 30 |
| DA-size da (multi-axis, pairs within L15) | `tie_rate_items` | -0.45 | 6.5e-09 | -0.31 | -0.31 | 174 |
| DA-size da (multi-axis, pairs within L2) | `rung_gap_corr` | 0.65 | 6.9e-04 | 0.20 | 0.19 | 30 |
| DA-size da (multi-axis, pairs within L30) | `pseudo_ref_da` | 0.56 | 1.1e-27 | 0.36 | 0.37 | 342 |
| DA-size da (multi-axis, pairs within L50) | `consecutive_kendall` | 0.75 | 1.1e-85 | 0.45 | 0.46 | 475 |
| DA-size da (multi-axis, pairs within L8) | `tie_rate_items` | -0.34 | 1.8e-03 | -0.19 | -0.18 | 110 |
| DA-size da (multi-axis) | `da_ckpt_mean` | 0.81 | 5.6e-108 | 0.63 | 0.63 | 475 |
| DA-size rho (multi-axis) | `da_ckpt_mean` | 0.80 | 6.7e-102 | 0.59 | 0.59 | 475 |
| DA-size tau_b (multi-axis) | `da_ckpt_mean` | 0.79 | 3.2e-100 | 0.57 | 0.58 | 475 |

**The AllenAI grid** (22 signals × 6 noises, DA, every pair, benchmarks). The best combination against AllenAI's own `rel_std` / checkpoint noise:

| truth | best signal / noise | ρ | rel_std / ckpt_rel ρ |
|---|---|---|---|
| DA-ckpt | `mpd / ckpt_abs` | 0.88 | 0.86 |
| DA-goal | `aad / ckpt_abs` | 0.82 | 0.81 |
| DA-size | `rms_deviation / ckpt_abs` | 0.75 | 0.75 |

**Particular cases.** The strongest significant configurations outside "every benchmark", two per (truth, subset type). `reliable (on the truth)` subsets are selected on the truth itself, so their ρ is conditional on it. All of them are in `surrogate_correlations.csv` (`q` column):

| truth | subset | surrogate | ρ | q | ρ over cells | units |
|---|---|---|---|---|---|---|
| DA-goal da (multi-axis, pairs within L50) | language: bn | `bpb_rank_agreement` | 1.00 | 3.8e-04 | 0.54 | 11 |
| DA-ckpt da (multi-axis) | language: sv | `window_kendall` | 1.00 | 3.8e-04 | 0.63 | 10 |
| DA-ckpt da (multi-axis, pairs within L50) | language: bn | `bpb_rank_agreement` | 0.99 | 3.8e-04 | 0.55 | 11 |
| DA-size da (multi-axis, pairs within L50) | language: bn | `language_consensus` | 0.99 | 3.8e-04 | 0.65 | 11 |
| DA-ckpt tau_b (multi-axis) | language: ta | `window_kendall` | 0.99 | 3.8e-04 | 0.70 | 10 |
| DA-goal rho (multi-axis) | language: hu | `rung_gap_corr` | 0.99 | 3.8e-04 | 0.50 | 10 |
| DA-ckpt rho (multi-axis) | language: ta | `sign_consistency_window` | 0.99 | 3.8e-04 | 0.70 | 10 |
| DA-goal da (multi-axis) | language: vi | `snr__dispersion__projection_depth` | 0.98 | 3.8e-04 | 0.45 | 13 |
| DA-size da (multi-axis, pairs within L50) | language: bn | `bpb_rank_agreement` | 0.98 | 3.8e-04 | 0.61 | 11 |
| DA-size rho (multi-axis) | language: bn | `language_consensus` | 0.98 | 3.8e-04 | 0.77 | 11 |
| DA-ckpt da (multi-axis) | benchmark: hellaswag | `snr__mad__kfold_rel` | 0.98 | 3.8e-04 | 0.60 | 26 |
| DA-ckpt tau_b (multi-axis) | benchmark: bbpb_rfgm_include_base_44 | `window_kendall` | 0.98 | 9.6e-23 | 0.67 | 36 |
| DA-ckpt da (multi-axis) | benchmark: bbpb_rfgm_include_base_44 | `window_kendall` | 0.98 | 1.3e-22 | 0.67 | 36 |
| DA-size rho (multi-axis) | language: hu | `ladder_da` | 0.98 | 7.2e-04 | 0.75 | 10 |
| DA-size da (multi-axis) | benchmark: rf_mmmlu | `prev_rung_da` | 0.98 | 7.2e-04 | 0.55 | 10 |
| DA-ckpt tau_b (multi-axis) | language: ta | `prev_rung_da` | 0.98 | 3.8e-04 | 0.71 | 10 |
| DA-goal rho (multi-axis) | language: hu | `ladder_da` | 0.98 | 3.8e-04 | 0.56 | 10 |
| DA-goal tau_b (multi-axis) | language: hu | `rung_gap_corr` | 0.98 | 3.8e-04 | 0.48 | 10 |
| DA-ckpt rho (multi-axis) | language: sv | `window_kendall` | 0.98 | 7.2e-04 | 0.61 | 10 |
| DA-size rho (multi-axis) | benchmark: rf_mmmlu | `prev_rung_da` | 0.98 | 7.2e-04 | 0.57 | 10 |
| DA-ckpt tau_b (multi-axis) | benchmark: bbpb_rfgm_include_base_44 | `sign_consistency_window` | 0.97 | 3.9e-22 | 0.68 | 36 |
| DA-ckpt rho (multi-axis) | benchmark: bbpb_rfgm_include_base_44 | `window_kendall` | 0.97 | 4.6e-21 | 0.69 | 36 |
| DA-ckpt rho (multi-axis) | benchmark: bbpb_rfgm_include_base_44 | `kendall_w_window` | 0.97 | 7.5e-21 | 0.69 | 36 |
| DA-goal tau_b (multi-axis) | language: vi | `snr__range__projection_depth` | 0.97 | 3.8e-04 | 0.43 | 13 |
| DA-size da (multi-axis, pairs within L30) | benchmark: bbpb_arc_mt | `snr__dispersion_shifted__ckpt_rel` | 0.96 | 3.8e-04 | 0.36 | 11 |
| DA-goal tau_b (multi-axis) | benchmark: rf_mmmlu | `snr__aad__kfold_rel` | 0.96 | 1.0e-03 | 0.49 | 10 |
| DA-size tau_b (multi-axis) | benchmark: rf_mmmlu | `prev_rung_da` | 0.96 | 7.2e-04 | 0.53 | 10 |
| DA-goal rho (multi-axis) | benchmark: rf_mmmlu | `snr__aad__kfold_rel` | 0.96 | 1.0e-03 | 0.49 | 10 |
| DA-goal rho (multi-axis) | benchmark: rf_mmmlu | `signal__aad` | 0.96 | 1.0e-03 | 0.47 | 10 |
| DA-goal tau_b (multi-axis) | benchmark: rf_mmmlu | `signal__aad` | 0.96 | 1.0e-03 | 0.47 | 10 |

**Per language count** (pairs of variants sharing L; benchmarks):

| truth | L | mean DA | sd | tasks | best surrogate | ρ |
|---|---|---|---|---|---|---|
| DA-ckpt | 1 | 0.70 | 0.23 | 150 | `snr__rel_dispersion__ckpt_abs` | 0.77 |
| DA-ckpt | 2 | 0.61 | 0.25 | 150 | `n_items` | 0.83 |
| DA-ckpt | 8 | 0.58 | 0.21 | 494 | `tie_rate_items` | -0.59 |
| DA-ckpt | 15 | 0.60 | 0.19 | 745 | `snr__star_discrepancy_shifted__kfold_rel` | 0.54 |
| DA-ckpt | 30 | 0.60 | 0.17 | 1136 | `sign_consistency_window` | 0.65 |
| DA-ckpt | 50 | 0.67 | 0.25 | 1489 | `bpb_rank_agreement` | 0.83 |
| DA-goal | 1 | 0.59 | 0.26 | 143 | `snr__rel_dispersion__tukey_depth` | 0.69 |
| DA-goal | 2 | 0.51 | 0.25 | 143 | `gain_over_noise` | 0.65 |
| DA-goal | 8 | 0.47 | 0.21 | 480 | `n_pairs` | 0.35 |
| DA-goal | 15 | 0.52 | 0.19 | 731 | `tie_rate_items` | -0.50 |
| DA-goal | 30 | 0.53 | 0.17 | 1119 | `pseudo_ref_da` | 0.56 |
| DA-goal | 50 | 0.60 | 0.26 | 1466 | `consecutive_kendall` | 0.80 |
| DA-size | 1 | 0.61 | 0.25 | 143 | `snr__rel_dispersion__tukey_depth` | 0.64 |
| DA-size | 2 | 0.51 | 0.26 | 143 | `rung_gap_corr` | 0.65 |
| DA-size | 8 | 0.51 | 0.21 | 480 | `tie_rate_items` | -0.34 |
| DA-size | 15 | 0.55 | 0.19 | 731 | `tie_rate_items` | -0.45 |
| DA-size | 30 | 0.55 | 0.18 | 1119 | `pseudo_ref_da` | 0.56 |
| DA-size | 50 | 0.60 | 0.26 | 1466 | `consecutive_kendall` | 0.75 |

**Extra: held-out confirmation.** The (benchmark, language) clusters are split in two halves. 5838 configurations ranked best on one half (within-stratum ρ) were tested once on the other (one-sided cluster permutation test, BH), and **1959 hold at q < 0.05**. The strongest:

| truth | subset | surrogate | ρ disc. | ρ val. [90 %] | q | tasks val. |
|---|---|---|---|---|---|---|
| DA-ckpt rho (multi-axis) | benchmark: bbpb_arc | `rung_sign_consistency` | 0.61 | 0.78 [0.64, 0.82] | 2.3e-03 | 14 |
| DA-ckpt rho (multi-axis) | benchmark: bbpb_arc | `ladder_da` | 0.60 | 0.77 [0.58, 0.81] | 2.3e-03 | 14 |
| DA-ckpt tau_b (multi-axis) | benchmark: bbpb_arc | `rung_sign_consistency` | 0.59 | 0.74 [0.59, 0.78] | 2.3e-03 | 14 |
| DA-ckpt da (multi-axis) | benchmark: bbpb_arc | `rung_sign_consistency` | 0.59 | 0.74 [0.61, 0.79] | 2.3e-03 | 14 |
| DA-size rho (multi-axis) | benchmark: bbpb_rf_belebele | `cronbach_alpha` | 0.71 | 0.74 [0.59, 0.82] | 2.3e-03 | 25 |
| DA-goal rho (multi-axis) | benchmark: bbpb_rf_include_base_44 | `rung_sign_consistency` | 0.66 | 0.74 [0.59, 0.81] | 2.3e-03 | 18 |
| DA-ckpt tau_b (multi-axis) | benchmark: bbpb_arc | `ladder_da` | 0.58 | 0.74 [0.62, 0.78] | 2.3e-03 | 14 |
| DA-ckpt da (multi-axis) | benchmark: bbpb_arc | `ladder_da` | 0.58 | 0.74 [0.60, 0.77] | 2.3e-03 | 14 |
| DA-ckpt da (multi-axis) | tier: L8 | `sign_consistency_window` | 0.68 | 0.72 [0.69, 0.76] | 2.3e-03 | 178 |
| DA-size rho (multi-axis) | benchmark: bbpb_rf_belebele | `da_ckpt_mean` | 0.70 | 0.72 [0.56, 0.78] | 2.3e-03 | 25 |
| DA-goal rho (multi-axis) | benchmark: bbpb_rfgm_include_base_44 | `da_ckpt_mean` | 0.75 | 0.71 [0.57, 0.78] | 2.3e-03 | 18 |
| DA-goal rho (multi-axis) | benchmark: bbpb_rfgm_include_base_44 | `sign_persistence` | 0.75 | 0.71 [0.56, 0.78] | 2.3e-03 | 18 |
| DA-ckpt rho (multi-axis) | all: bpb | `kendall_w_window` | 0.73 | 0.70 [0.59, 0.78] | 2.3e-03 | 25 |
| DA-ckpt rho (multi-axis) | benchmark: bbpb_rfgm_belebele | `cronbach_alpha` | 0.69 | 0.70 [0.61, 0.75] | 2.3e-03 | 25 |
| DA-size da (multi-axis) | tier: L8 | `bpb_rank_agreement` | 0.65 | 0.70 [0.64, 0.75] | 2.3e-03 | 169 |

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
      accuracy best. The holdout (`rq03_noise_and_snr/compare_seed_splits.py`,
      reported in the highlight and "Seed generalization" table) asks whether that
      ranking of definitions survives a change of seed: it is computed on
      `predictivity_seeds_train` (the replicate seeds 64, 313, 28, 1797) and again on
      `predictivity_seeds_test` (seed 1904 of the same cells), and the two are
      compared.
      *The problem (re-read 2026-10-10).* Replicate seeds exist at 175M, 600M
      and 1B (the 1B ×3 cells, seeds 28/1797, joined both splits on
      2026-10-05), but DA-size on the holdout is 175M → 600M only, so the 1.7B
      reference never enters. On that pair the English tasks rest on 6 model
      pairs in the train split and 3 in the test split, and so does the
      English BPB (`bpb_dclm`). A non-English benchmark is trained only at L30/L50 (Russian also at
      L2), so only the Russian tasks have pairs (2 on train, 1 on test). With
      one or two pairs a task's "decision accuracy" is close to 0 or 1 and
      measures seed noise, not a ranking. The median task has no DA-size pair
      at all (1552 of the 1806 train-split tasks).
      *Implications (2026-10-10 04:31 report).* Only English has a best
      definition on both splits under DA-size, and only English and Russian
      under DA-ckpt. The family-level agreement is therefore 100 % (1 of 1)
      under DA-size and 50 % (1 of 2) under DA-ckpt, and the exact-variant
      agreement 100 % / 0 %. The global ranking ρ (0.55 DA-size, 0.72 DA-ckpt)
      is over 22 definitions on splits that share only those 1–2 languages, so
      "the ranking survives a seed swap" says as much about the holdout as
      about the definitions. Nothing in the main `predictivity` tables is
      affected.
      *Options.* (A) keep only tasks with ≥ 6 pairs on both splits: an honest
      check, but English + BPB only. (B) treat replicate seeds as replicates,
      not as models: average them before ranking, and use the holdout only
      for the noise estimate. This changes what the holdout means. (C) report the
      holdout for DA-ckpt only, whose pairs are checkpoints × variants within
      a size and do not shrink to one. (D) drop the holdout numbers from the
      highlight until more sizes have replicate seeds.
      *Recommendation.* A now (small change, honest scope, say "English and
      BPB" in the highlight), and revisit once the 90M / 1.7B-L15 cells and
      any further replicate seeds exist. Not implemented yet.
- [ ] Bootstrap CIs on per-language Pearson r and cross-pool Spearman ρ.
- [ ] Recommend a *family* (dispersion / relative-spread), not an exact variant,
      because only the family transfers across seeds.
- [ ] Use a larger DA-size target (e.g. Apertus-8B) instead of the
      not-fully-converged 1B custom model.

## Extensions from other sweeps

Everything below comes from the **36-model sweep** (2026-04…06, 4 sizes × 3
data mixtures × 3 seeds, pools `seeds_1904`, `seeds_28_1797`,
`seeds_28_1797_1904`, `custom_swissai_hf`, with 12 languages and reference **1B**)
or from the **external tier** (`all/external`: the public and reference
models, 270M–70B, no data-mixture axis). Its harness, task set, checkpoint
window and reference differ from the ladder's, and its 36-sweep DA-size is
a 3-mixture ranking against 1B. Its numbers are therefore replications of the
family-level recommendation above, never rows of the same table. The dated
comparison note that once tracked these tiers (`ANALYSIS_new_vs_previous.md`, removed 2026-09-23:
`rel_mpd` on top in every tier, DA-ckpt r 0.379 → 0.519 with the externals
folded in, holdout ρ 0.80 / 0.93) predates the tie fix in `decision_acc_fast`
and the parent-only filter. The blocks below supersede it.

## External model-set tier (`all/external`, 36-sweep)

The `external` tier pools every non-custom model (reference HF, a06,
distillation, posttraining, at sizes 270M…70B) with **no data-mixture axis**. The
"signal" here is therefore the **cross-model dispersion** of final-checkpoint scores across the
external ladder: *how far different model families separate on a benchmark*,
rather than how far the three FineWeb mixtures separate. SNR magnitudes are
an order larger than the custom pool's (e.g. `multiblimp_rus` SNR
≈ 119), and DA is the within-family scaling-DA over that ladder. The headline,
**recommend the dispersion *family*, not an exact variant**, holds on this
disjoint model set, which is the strongest robustness check we have. Regenerate with
`python analysis/rq04_surrogates/snr_definition_postprocess.py --pool external`.

**Global variant ranking.** Mean Pearson r of log₁₀(SNR) with DA across languages.
The dispersion and relative-spread clusters lead overall and on DA-ckpt. The
discrepancy cluster leads DA-size only, the depth metrics fail, and the custom-pool winner
`dist_std` is undefined on this tier:

| variant (family) | DA-size r | DA-ckpt r | overall |
|---|---|---|---|
| `dispersion` / `mpd` / `range` / `mad` (dispersion) | 0.16 | **0.44** | **0.30** |
| `rel_std` / `rel_mpd` / `iqr` (relative-spread) | 0.16 | 0.43 | 0.30 |
| `mpsd` (dispersion) | 0.10 | **0.45** | 0.27 |
| `star_discrepancy` (discrepancy) | **0.20** | 0.05 | 0.12 |
| `gini` / `dispersion_shifted` / `star_discrepancy_shifted` (discrepancy) | 0.19 | 0.07 | 0.13 |
| `discrepancy` (discrepancy) | 0.19 | −0.03 | 0.08 |
| `projection` (depth) | 0.06 | −0.21 | −0.07 |
| `dist_std`, `tukey` | — | — | — |

![SNR variants ranked by correlation with DA (external)](all/external/top_variants_overall.png)

DA-size is sparse on the external ladder (most cross-bucket pairs lack ≥2
spanning families), so **DA-ckpt is the more trustworthy axis here**. On it
the dispersion/relative-spread families win cleanly, in line with the custom
pool's family-level recommendation.

**Most reliable benchmark per language.** The rank-1 benchmark by `dist_std` SNR @ 1B
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

The exact per-language argmax still does not transfer across tiers. Only the
dispersion/relative-spread *family* does, so the paper-level claim is the family
recommendation, with HellaSwag and MultiBLiMP as the durable multilingual anchors.

[top_variants_overall.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/all/external/top_variants_overall.png) ·
[top_benchmarks_per_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/all/external/top_benchmarks_per_language.png)

## Results from the 36-model sweep (2026-06, superseded)

The numbers below were generated on the 36-model sweep (4 sizes × 3 data mixtures × 3 seeds, 12 languages, pool `custom_swissai_hf` unless stated) and are kept as history. The predictivity ladder regenerates the blocks above.

### Highlighted result

- **Global-best SNR definition (`custom_swissai_hf`): `dist_std`.** Its mean Pearson r of log₁₀(SNR) with decision accuracy is **0.32** (DA-size), **0.43** (DA-ckpt) and 0.38 overall. DA-ckpt is led by the mean-pairwise-distance / relative-spread cluster (`mpsd`/`rel_mpd`/`rel_mpsd` ≈ 0.51). All of them are in the dispersion family, so we recommend the *family*, not an exact variant.
- **Per-language anchor: `multiblimp`.** It is the highest-SNR above-random benchmark in **6 of 11** languages (`dist_std` SNR @ 1B).
- **Variant ranking generalizes across seeds under DA-ckpt** (holdout Spearman ρ **0.81**) **but not under DA-size** (ρ **-0.07**). The DA-size per-variant correlations are small and near-tied, so their ranking is dominated by noise and does not survive a seed swap. Treat the DA-size variant ranking as unreliable. The exact per-language argmax never transfers. **Never `tukey` / `projection`** (r ≤ 0).

### Results

Headline numbers from the `custom_swissai_hf` pool. Regenerate with `python analysis/rq04_surrogates/snr_definition_postprocess.py --pool custom_swissai_hf`.

**Global variant ranking.** Mean Pearson r of log₁₀(SNR) with DA across languages (the top of a tight dispersion block, with the depth metrics collapsing):

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

**Statistical power by pool.** Each pool's best DA-size variant:

| pool | best variant (DA-size) | DA-size r | DA-ckpt r |
|---|---|---|---|
| `seeds_1904` (1 seed) | `mad` | 0.50 | 0.28 |
| `seeds_28_1797` (2 seeds) | `rel_mpd` | 0.31 | 0.37 |
| `seeds_28_1797_1904` (3 seeds) | `rel_std` | 0.43 | 0.48 |
| `custom_swissai_hf` (3 seeds + externals) | `dist_std` | 0.32 | 0.43 |

**Most reliable benchmark per language.** `dist_std` SNR @ 1B over above-random tasks (DA-size is NaN at the 1B target, so DA-ckpt@1B is shown):

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

**Seed generalization.** Holdout `seeds_28_1797` → `seeds_1904`. The DA-ckpt ranking transfers, but **DA-size does not**. Its per-variant correlations are small and clustered (top variants within ~0.1), so the global ranking is dominated by noise and its Spearman ρ is unstable from run to run (do not read it as a real effect):

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

- `pretraining/<pool>/snr_variant_ranking.csv`: per-(variant, DA-def, scope)
  Pearson r (`analyze_snr_variants.py`).
- `…/top_variants_overall.csv`, `best_variant_per_language.csv`,
  `best_variant_family_per_language.csv`, `variant_clusters.csv`,
  `top_benchmarks_per_language.csv`: the variant ranking and the per-language
  anchor (`snr_definition_postprocess.py`).
- `…/top_variants_overall.png`, `best_variant_per_language.png`,
  `best_variant_family_per_language.png`, `top_benchmarks_per_language.png`,
  `variant_correlation_matrix.png`, `da_size_vs_da_ckpt_multi_axes.png`,
  `{da_size,da_ckpt}/…`: supporting figures.
- `…/rq3_surrogates.csv`, `rq3_surrogates.png`, `facts.json`: the
  statistics beyond SNR (`analyze.py`, the paper's surrogate figure).
- `…/surrogate_values.csv`, `surrogate_targets.csv`, `surrogate_definitions.csv`: the
  surrogates per (task, proxy, pair set) and the truths, long (`catalogue.py`, sources in
  `literature.md`). `surrogate_correlations.csv`, `surrogate_filters.csv`,
  `surrogate_validated.csv`, `da_all_retest_multi_axes.csv` and
  `surrogates_{validated,snr_grid,catalogue,by_L,catalogue_by_language}.png`: the
  validated search (`search.py`).
- Inputs: the noise-and-SNR `snr_variants_per_task.csv` and holdout
  `headline_metrics.csv`, the gate's `above_random_scores.csv`, the scaling
  analysis's `rq1_fits.csv`.
- `…/finetasks_criteria.{csv,png}`, `finetasks_surrogates_scatter.{csv,png}`,
  `finetasks_surrogates_rank.csv`, `../finetasks_overlap.csv`: FineTasks'
  criteria on the ladder and every surrogate against DA-size
  (`finetasks_criteria.py`, moved from the benchmark-design folder on
  2026-09-23). The copies left
  under `rq09_benchmark_design/pretraining/predictivity/` are stale.
