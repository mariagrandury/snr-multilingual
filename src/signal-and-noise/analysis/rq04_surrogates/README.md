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

- **Global-best SNR definition (`predictivity`): `aad`** — mean Pearson r of log₁₀(SNR) vs decision accuracy **0.30** (DA-size, proxy → 1.7B, 50 languages), **0.42** (DA-ckpt, proxy sizes pooled, 50 languages), 0.36 overall. DA-ckpt is led by `aad`/`rms_deviation`/`dist_std` (≈ 0.42; one family: dispersion) — recommend the *family*, not an exact variant.
- **Per-language anchor: `bbpb_multiblimp`** — the highest-SNR above-random benchmark in **12 of 50** languages (`aad` SNR @ 1.7B; `train_loss` and `bpb_macro` are not a language's and are left out); the language's own BPB, ungated and on its own noise scale, outranks that benchmark in 1 of the 50 languages that have both. Weakest variants overall: `projection`, `tukey`.
- **Seed holdout (predictivity_seeds_train → predictivity_seeds_test)**: Spearman ρ of the global variant ranking **0.84** (DA-ckpt), **0.81** (DA-size); family-level per-language agreement 100% / 100%. A ranking that does not survive the seed swap is noise-dominated — only the *family* recommendation transfers.
<!-- END auto:highlight -->

## Headlines (ladder report 2026-10-07 15:51)

*Pool `predictivity`: seed 1904, every cell (L ∈ {1, 2, 8, 15, 30, 50}; the
deep, shallow and swiglu ladders; every data build), sizes 90M–1.7B, the gate
per (task, size) (rule 1), parent tasks, trained languages. Per-language
Pearson r of log₁₀(SNR) with DA over the 50 languages with ≥ 3 tasks
(rule 8), then averaged.*

![The surrogate question in one figure](pretraining/predictivity/highlights.png)

GitHub: [highlights.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/highlights.png) · [highlights.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/highlights.csv)

**Key findings**

- **The dispersion SNR definitions track decision accuracy best, as a
  block:** the top eight (`aad`, `dist_std`, `quartile_deviation`,
  `rms_deviation`, `mpd`, `range`, `dispersion`, `mad`) read mean r
  0.27–0.30 with DA-size and 0.40–0.42 with DA-ckpt (0.34–0.36 overall,
  `top_variants_overall.csv`; the figure's all-task panels read 0.31–0.34
  and 0.41–0.44 for their top six, its DA-size panel led by the
  relative-spread `rel_mpd`, `iqr` and `rel_std`). The leader `aad` (0.30 /
  0.42) wins by less than 0.01, and the depth definitions are anti-signals
  (`tukey` −0.12 / −0.21, `projection` −0.15 / −0.23).
- **No exact definition is best everywhere:** per language the winner is
  one of 12 definitions under DA-size and one of 11 under DA-ckpt. The
  dispersion family wins 23 of 50 languages under DA-ckpt but 10 under
  DA-size, where discrepancy wins 24.
- **The seed holdout now agrees under both DA kinds, on one or two
  languages:** the global ranking of the 22 definitions correlates across
  the seed split at Spearman ρ 0.84 under DA-ckpt and 0.81 under DA-size.
  The per-language agreement quoted in the highlight rests on English alone
  (DA-size, 1 of 1) and on English and Russian (DA-ckpt, 2 of 2).
- **Beyond SNR, the early-checkpoint agreement leads at every proxy, but on
  the bBPB twins:** it reads Spearman ρ 0.54–0.64 with DA-size over
  1117–1266 tasks (816 of them twins at every rung) against 0.18–0.40 on
  the 301–450 accuracy tasks alone, where SNR relative std (0.20–0.39) and
  the inverted noise (0.22–0.35) read as much. Margin above chance (−0.07
  to 0.04) and signal alone (−0.11 to 0.04) carry nothing on the benchmark
  tasks.
- **The proxy's own rank stability beats every SNR in the catalogue:** over
  420 (benchmark, language) clusters, consecutive-checkpoint Kendall τ reads
  ρ 0.97 against DA-ckpt and 0.86 against DA-goal, and the proxy's mean
  DA-ckpt 0.84 against DA-size (ceilings 0.99 / 0.95 / 0.92). AllenAI's
  relative-std / checkpoint-noise SNR reads 0.59–0.65.
- **FineTasks' criteria separate reliable tasks only modestly:** on the 846
  accuracy benchmark tasks the three together pass 16 % (90M) to 42 % (1B),
  against 37–54 % for our gate. Their passers read DA-size 0.56–0.59
  against 0.46–0.49 for the rest.

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
  variant trained there; the noise is the late-checkpoint std over the noise
  window, 80/85/90/95/100 % of the run (rule 4). DA-size is proxy final →
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
**2026-10-07 15:51** (outputs regenerated in `7966367c`, prose re-read
2026-10-07).

<!-- BEGIN auto:results (snr_definition_postprocess.py --pool predictivity) -->
## Results

Headline numbers from the `predictivity` pool. Regenerate with `python analysis/rq04_surrogates/snr_definition_postprocess.py --pool predictivity`.

**Global variant ranking** — mean Pearson r of log₁₀(SNR) vs DA across the trained languages with ≥ 3 tasks (rule 8; 50 languages under DA-size, 50 under DA-ckpt). DA-size = proxy final → 1.7B final only (the proxy-to-proxy scaling pairs are not DA-size, rule 9); DA-ckpt = a proxy size's early checkpoints → its final, the proxy sizes 90M, 175M, 350M, 600M, 1B pooled, never the 1.7B run's own checkpoints (rule 11):

| variant | DA-size r | DA-ckpt r | overall |
|---|---|---|---|
| `aad` | 0.30 | 0.42 | 0.36 |
| `dist_std` | 0.30 | 0.42 | 0.36 |
| `quartile_deviation` | 0.30 | 0.42 | 0.36 |
| `rms_deviation` | 0.30 | 0.42 | 0.36 |
| `mpd` | 0.29 | 0.42 | 0.35 |
| `range` | 0.28 | 0.41 | 0.35 |
| `dispersion` | 0.28 | 0.41 | 0.35 |
| … |  |  |  |
| `tukey` | -0.12 | -0.21 | -0.17 |
| `projection` | -0.15 | -0.23 | -0.19 |

![SNR variants ranked by correlation with DA](pretraining/predictivity/top_variants_overall.png)

**Statistical power by pool** — each pool's best variant (mean r over both DA kinds):

| pool | best variant (overall) | DA-size r | DA-ckpt r |
|---|---|---|---|
| `predictivity` (grid, seed 1904, every design) | `aad` | 0.30 | 0.42 |
| `predictivity_seeds` (every seed) | `dist_std` | 0.30 | 0.42 |

**Most reliable benchmark per language** — `aad` SNR @ 1.7B over the above-random benchmarks, with the language's own BPB SNR alongside (ungated, on its own noise scale; DA-size is undefined at the reference size itself, so DA-ckpt@1.7B is shown; `train_loss` and `bpb_macro` measure the whole mixture and are not a row, rule 7):

| lang | top benchmark | SNR | DA-ckpt@1.7B | BPB SNR |
|---|---|---|---|---|
| ar | `bbpb_rf_belebele_arb_Latn` | 3.33 | 0.60 | 0.75 |
| az | `bbpb_include_v2_og_azerbaijani_azerbaijan` | 12.65 | 0.98 | 1.71 |
| bg | `multiblimp_bul` | 2.62 | 0.81 | 0.80 |
| bn | `bbpb_global_piqa_parallel_cloze_ben_latn` | 3.09 | 0.67 | 0.74 |
| bs | `bbpb_global_piqa_nonparallel_cloze_bos_latn` | 2.49 | 0.87 | 2.06 |
| ca | `bbpb_xnli_ca` | 5.88 | 0.89 | 1.65 |
| cs | `bbpb_multiblimp_ces` | 1.95 | 0.76 | 0.44 |
| da | `multiblimp_dan` | 3.21 | 0.46 | 0.97 |
| de | `bbpb_multiblimp_deu` | 2.07 | 0.73 | 0.63 |
| el | `bbpb_belebele_ell_Grek` | 2.56 | 0.67 | 0.27 |
| en | `bbpb_acp_bench_cloze_val` | 4.33 | 0.59 | 1.41 |
| es | `bbpb_xnli_es` | 3.04 | 0.79 | 0.83 |
| et | `bbpb_multiblimp_est` | 7.11 | 0.91 | 2.36 |
| fa | `bbpb_belebele_pes_Arab` | 2.00 | 0.71 | 0.65 |
| fi | `bbpb_multiblimp_fin` | 3.01 | 0.80 | 1.29 |
| fr | `bbpb_include_v2_og_french_canada` | 3.68 | 0.65 | 0.48 |
| he | `bbpb_multiblimp_heb` | 5.18 | 0.79 | 1.21 |
| hi | `bbpb_include_base_44_hindi` | 3.09 | 0.66 | 0.74 |
| hr | `bbpb_global_piqa_parallel_cloze_hrv_latn` | 2.53 | 0.87 | 1.72 |
| hu | `bbpb_include_v2_og_hungarian_hungary` | 2.60 | 0.72 | 0.70 |
| id | `bbpb_include_base_44_indonesian` | 2.17 | 0.61 | 0.55 |
| it | `bbpb_multiblimp_ita` | 2.47 | 0.76 | 0.53 |
| ja | `bbpb_include_v2_og_japanese_japan` | 2.46 | 0.63 | 0.79 |
| ka | `bbpb_belebele_kat_Geor` | 1.71 | 0.75 | 0.65 |
| kk | `bbpb_rf_include_base_44_kazakh` | 4.61 | 0.83 | 1.37 |
| ko | `bbpb_belebele_kor_Hang` | 1.82 | 0.63 | 0.73 |
| lt | `bbpb_include_v2_og_lithuanian_lithuania` | 3.63 | 0.85 | 1.67 |
| lv | `bbpb_rfgm_belebele_lvs_Latn` | 1.59 | 0.98 | 2.09 |
| ml | `bbpb_include_base_44_malayalam` | 2.80 | 0.72 | 0.72 |
| mr | `bbpb_belebele_mar_Deva` | 2.70 | 0.69 | 1.23 |
| ms | `bbpb_include_v2_og_malay_singapore` | 5.90 | 0.67 | 1.65 |
| ne | `bbpb_rf_belebele_npi_Latn` | 2.61 | 0.81 | 1.17 |
| nl | `bbpb_multiblimp_nld` | 2.24 | 0.73 | 0.61 |
| no | `bbpb_global_piqa_parallel_cloze_nob_latn` | 1.49 | 0.63 | 0.89 |
| pl | `bbpb_multiblimp_pol` | 3.32 | 0.70 | 0.49 |
| pt | `bbpb_multiblimp_por` | 3.08 | 0.64 | 0.46 |
| ro | `bbpb_multiblimp_ron` | 2.28 | 0.71 | 0.71 |
| ru | `bbpb_include_base_44_russian` | 2.73 | 0.72 | 0.82 |
| sk | `bbpb_global_piqa_parallel_cloze_slk_latn_sari` | 3.46 | 0.89 | 1.32 |
| sl | `bbpb_global_piqa_parallel_cloze_slv_latn_cerk` | 7.65 | 0.85 | 2.44 |
| sq | `bbpb_include_v2_og_albanian_albania` | 5.81 | 0.85 | 2.37 |
| sr | `bbpb_global_piqa_parallel_cloze_srp_latn` | 3.11 | 0.83 | 1.37 |
| sv | `bbpb_multiblimp_swe` | 2.56 | 0.81 | 1.03 |
| ta | `bbpb_include_base_44_tamil` | 3.05 | 0.72 | 0.64 |
| th | `bbpb_belebele_tha_Thai` | 1.65 | 0.66 | 0.40 |
| tr | `bbpb_multiblimp_tur` | 3.46 | 0.55 | 0.60 |
| uk | `bbpb_belebele_ukr_Cyrl` | 2.27 | 0.73 | 0.18 |
| ur | `bbpb_rf_include_base_44_urdu` | 6.22 | 0.81 | 1.79 |
| vi | `bbpb_include_v2_og_vietnamese_vietnam` | 1.87 | 0.70 | 0.88 |
| zh | `bbpb_paws_zh` | 4.87 | 0.86 | 1.52 |

![Top-5 benchmarks per language by SNR](pretraining/predictivity/top_benchmarks_per_language.png)

**Seed generalization** — holdout `predictivity_seeds_train` → `predictivity_seeds_test` (the ×3 cells only). A variant ranking whose Spearman ρ is low here is noise-dominated; recommend the family that transfers, not the argmax:

| metric | DA-size | DA-ckpt |
|---|---|---|
| Spearman ρ on global variant ranking | 0.81 | 0.84 |
| Pearson r between splits (all cells) | 0.85 | 0.67 |
| Exact-variant agreement (per lang) | 0% | 0% |
| Family-level agreement (per lang) | 100% | 100% |
| Retention of train-best r on test | 48% | 96% |
<!-- END auto:results -->

GitHub: [top_variants_overall.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_variants_overall.png) · [top_variants_overall.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_variants_overall.csv) ·
GitHub: [top_benchmarks_per_language.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_benchmarks_per_language.png) · [top_benchmarks_per_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/top_benchmarks_per_language.csv) ·
[snr_variant_ranking.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/snr_variant_ranking.csv) ·
[best_variant_per_language.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/best_variant_per_language.csv)

**Key findings** (pool `predictivity`, 90M–1.7B, gated, 50 languages under
each DA kind; the holdout rows on the ×3 cells only)

- **A block, not a winner:** the eight leading definitions span 0.34–0.36
  overall, and `aad` leads `dist_std` by 0.002. With every replicate seed in
  the signal pool (`predictivity_seeds`) the leader becomes `dist_std`, at
  0.30 / 0.42, with `aad` second by 0.002.
- **The most reliable benchmark per language is a bBPB twin:** a `bbpb_*`
  task has the highest `aad` SNR at 1.7B in 47 of 50 languages, and an
  accuracy task only in Bulgarian and Danish (MultiBLiMP); in Latvian the
  language's own BPB ranks first. MultiBLiMP, in either scoring, is the top
  benchmark in 14 languages, and the twins now carry every checkpoint, so
  their rows are no longer provisional.
- **The seed holdout is too thin to read:** the per-language agreement
  (100 % / 100 %) and the 48 % / 96 % retention are over 1 language
  (DA-size) and 2 (DA-ckpt). The cell-level Pearson r between the splits,
  0.85 over 22 (definition, language) cells and 0.67 over 44, carries the
  same caveat.

**Follow-ups**

- The top-benchmark table split by scoring (accuracy vs bBPB), so the
  accuracy anchor per language is visible while the twins take 47 of 50 rows.
- A bootstrap over languages for the mean r of the top eight definitions:
  they sit within 0.03 of each other, so only the family can be claimed.
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
(rule 13): 1117, 1152, 1188, 1232 and 1266 tasks at 90M–1B, of which 816 at
every proxy are twins and 301–450 accuracy tasks; the BPB rows are the 50
trained languages at every proxy.

<!-- BEGIN auto:surrogates (analyze.py --pool predictivity) -->
## Statistics beyond SNR (paper RQ3)

Numbers from the `predictivity` pool's rq03 table. Regenerate with `python analysis/rq04_surrogates/analyze.py --pool predictivity`. The population at a proxy size is the tasks above chance at the proxy and at 1.7B (rule 1); each candidate is scored on the tasks of it where the candidate has a value, so n differs per candidate and is given next to every ρ. The scaling-fit R² is rq01's log-N fit refitted on the rungs up to the proxy only (rule 11; it needs 3 rungs, so it starts at 350M). `bpb_macro` and `train_loss` are in neither population (rule 7).

- **benchmark tasks** — strongest surrogate of DA-size (mean ρ over proxies): `early-checkpoint agreement (10 %)` 0.58; weakest: `signal alone (relative std)` -0.02.
- **per-language bits per byte** — strongest surrogate of DA-size (mean ρ over proxies): `SNR, dist_std` 0.31; weakest: `scaling-fit R² (proxy rungs only)` 0.00.

**benchmark tasks** (Spearman ρ of the statistic with DA-size, per proxy size; n = the tasks behind the ρ):

| metric | 90M ρ (n) | 175M ρ (n) | 350M ρ (n) | 600M ρ (n) | 1B ρ (n) |
|---|---|---|---|---|---|
| early-checkpoint agreement (10 %) | 0.59 (1117) | 0.64 (1152) | 0.56 (1188) | 0.54 (1232) | 0.56 (1266) |
| scaling-fit R² (proxy rungs only) |  |  | 0.44 (1115) | 0.40 (1149) | 0.45 (1189) |
| SNR, relative std | 0.32 (1117) | 0.36 (1152) | 0.51 (1188) | 0.38 (1232) | 0.24 (1266) |
| noise alone (relative std, inverted) | 0.24 (1117) | 0.30 (1152) | 0.46 (1188) | 0.30 (1232) | 0.31 (1266) |
| SNR, discrepancy | 0.24 (301) | 0.29 (336) | 0.33 (372) | 0.29 (416) | 0.33 (450) |
| SNR, dist_std | 0.30 (1117) | 0.21 (1152) | 0.36 (1188) | 0.31 (1232) | 0.19 (1266) |
| margin above chance | 0.03 (296) | -0.07 (331) | -0.02 (367) | 0.04 (411) | -0.05 (445) |
| signal alone (relative std) | 0.04 (1117) | -0.00 (1152) | -0.03 (1188) | -0.02 (1232) | -0.11 (1266) |

**per-language bits per byte** (Spearman ρ of the statistic with DA-size, per proxy size; n = the tasks behind the ρ):

| metric | 90M ρ (n) | 175M ρ (n) | 350M ρ (n) | 600M ρ (n) | 1B ρ (n) |
|---|---|---|---|---|---|
| SNR, dist_std | 0.45 (50) | 0.53 (50) | 0.52 (50) | -0.44 (50) | 0.47 (50) |
| SNR, relative std | 0.32 (50) | 0.43 (50) | 0.30 (50) | -0.39 (50) | 0.25 (50) |
| signal alone (relative std) | 0.32 (50) | 0.40 (50) | 0.27 (50) | -0.40 (50) | 0.25 (50) |
| noise alone (relative std, inverted) | -0.01 (50) | -0.02 (50) | 0.00 (50) | 0.41 (50) | -0.02 (50) |
| early-checkpoint agreement (10 %) | 0.48 (50) | 0.41 (50) | 0.21 (50) | -0.45 (50) | -0.56 (50) |
| scaling-fit R² (proxy rungs only) |  |  | 0.31 (50) | -0.64 (50) | 0.34 (50) |

![Surrogates](pretraining/predictivity/rq3_surrogates.png)
<!-- END auto:surrogates -->

GitHub: [rq3_surrogates.png](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/rq3_surrogates.png) · [rq3_surrogates.csv](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/rq3_surrogates.csv) ·
[facts.json](https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/rq04_surrogates/pretraining/predictivity/facts.json)

**Key findings** (pool `predictivity`, DA-size against 1.7B, gated; the
benchmark n per cell in the table; the accuracy-only and twin-only ρ are
recomputed by hand from the noise-and-SNR per-task table with this
script's gate, without the `bbpb_` rows or with them alone)

- **On the twins the early-checkpoint agreement leads at every proxy:** ρ
  0.54–0.64 over 1117–1266 tasks (mean 0.58), ahead of the scaling-fit R²
  (0.40–0.45 from 350M) and SNR relative std (0.24–0.51). On the 816 twins
  alone it reads 0.59–0.72.
- **On the accuracy tasks alone no candidate stands out:** over 301–450
  tasks the early-checkpoint agreement reads 0.18–0.40, SNR relative std
  0.20–0.39 and the inverted noise 0.22–0.35. The leader among the three
  changes with the proxy (SNR relative std at 90M, 175M and 600M, the noise
  at 350M, the early agreement at 1B).
- **On the benchmark tasks, margin above chance and signal alone predict
  nothing:** ρ −0.07 to 0.04 (296–445 accuracy tasks) and −0.11 to 0.04 at
  every proxy. There the SNR's predictive part is its noise, which reads
  0.24–0.46 inverted.
- **Per-language BPB flips sign at 600M:** SNR `dist_std` reads 0.45–0.53 at
  90M, 175M, 350M and 1B but −0.44 at 600M (50 languages). SNR relative
  std, signal alone and the scaling-fit R² flip there too, and the inverted
  noise rises from about 0 to 0.41.

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
ladder and data build; up to 29 runs per task and size), on the ten evaluated
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
| 90M | 1662 | 9% | 63% | 23% | 8% | 37% | 0.58 [134] vs 0.58 [1528] | -0.49 / +0.11 / +0.12 / +0.59 |
| 175M | 1662 | 13% | 62% | 24% | 11% | 40% | 0.57 [183] vs 0.56 [1479] | -0.37 / +0.14 / +0.21 / +0.51 |
| 350M | 1662 | 18% | 64% | 25% | 15% | 44% | 0.56 [250] vs 0.57 [1412] | -0.44 / +0.22 / +0.25 / +0.56 |
| 600M | 1662 | 23% | 62% | 27% | 18% | 49% | 0.57 [304] vs 0.58 [1358] | -0.35 / +0.17 / +0.24 / +0.52 |
| 1B | 1662 | 26% | 62% | 29% | 22% | 54% | 0.59 [359] vs 0.59 [1303] | -0.33 / +0.28 / +0.29 / +0.55 |
| 1.7B | 1662 | 27% | 62% | 31% | 24% | 58% | — | +nan / +nan / +nan / +nan |

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
  margin > 3 std together pass 16 / 22 / 30 / 36 / 42 % of tasks at
  90M / 175M / 350M / 600M / 1B, below our gate's 37 / 40 / 44 / 49 / 54 %.
  Their passers read DA-size 0.56–0.59 against 0.46–0.49 for the rest.
- **Their SNR criterion does not discriminate, monotonicity does the work:**
  the SNR > 20 cut passes 61–63 % of the accuracy tasks at every proxy, while
  monotonicity rises from 18 % (90M) to 50 % (1B).
- **The four statistics correlate weakly with DA-size on the accuracy
  tasks:** Spearman 0.12–0.42 per proxy (0.12–0.25 at 90M, 0.29–0.42 at 1B).
  The negative monotonicity of the generated table at every size (−0.33 to
  −0.49) is the bBPB twins (−0.50 to −0.63 on them alone): pooled over the
  proxies on the accuracy tasks alone it reads +0.31.
- **Pooled, the proxy's own run is the best reference-free signal:**
  `finetasks_surrogates_rank.csv` (twins included) puts the proxy's median
  DA-ckpt at 0.61, the DA between the two rungs below at 0.57, its own
  trajectory R² at 0.56 and its DA-ckpt at 50 % of the run at 0.54, against
  0.33 for the item count, at most 0.31 for an SNR definition, 0.23 for our
  gate and 0.19 for FineTasks' SNR. On the accuracy tasks alone the median
  DA-ckpt drops to 0.38, level with the DA between the two rungs below
  (0.33) and the item count (0.33).
- **Spread across variants is not agreement with the reference:** the
  relative signal reads −0.03 and the cross-variant std 0.14 pooled, while
  the relative noise reads −0.29. The statistics that see the reference
  remain far above (the proxy ranking's own τ_b 0.99, the full-trajectory R²
  0.55).
- **The format FineTasks chose matters more than its thresholds:** of their
  96 picks, 55 exist in our registry and 35 have results at 1B, where 49 % of
  the originals pass their criteria on our ladder against all 13 cloze twins.
  The twins also read the higher mean DA-size (0.64 against 0.60).

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

- **Per L, an SNR definition almost never tracks DA:** 115 of the 132
  (definition, L) cells of `snr_variant_min_size_by_L` never reach ρ ≥ 0.3
  under DA-size, 84 under DA-ckpt. Per-L DA rests on few pairs, so this
  reads the lattice of possible DA values as much as the definitions.
- **Pooled over every pair (version B), the best definition reads ρ
  0.34–0.51 with DA-size on the benchmarks at every proxy**, the median
  definition 0.21–0.35; with DA-ckpt the best reads 0.49–0.76, both peaking
  at 350M.
- **The per-language BPB flip at 600M shows here too:** the median
  definition's ρ with DA-size on BPB is 0.39–0.51 at 90M–350M and 0.46 at
  1B, but −0.35 at 600M.
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
  consecutive-checkpoint Kendall τ reads 0.97 against DA-ckpt and 0.86
  against DA-goal, the proxy's mean DA-ckpt 0.84 against DA-size (90–98 % of
  the ceiling; the mean DA-ckpt, tied with `sign_persistence`, also leads
  DA-goal at 0.87). On mono-axis pairs the best read 0.91, 0.73 and 0.68.
- **SNR is mid-table.** AllenAI's `rel_std` / checkpoint noise reads 0.59,
  0.62 and 0.65 on DA-size, DA-goal and DA-ckpt; the best grid combinations
  read 0.59, 0.62 (`aad / ckpt_abs`) and 0.67
  (`star_discrepancy_shifted / kfold_rel`), within 0.02 of it.
- **The k-fold noise helps, and the item count is part of it:** relative
  k-fold noise beats relative checkpoint noise for 16, 18 and 16 of the 22
  signals (DA-size, DA-goal, DA-ckpt). `n_items` alone reads 0.38, 0.40 and
  0.49.
- **Signal alone carries almost nothing:** range, `dist_std` and `mpd` across
  variants read −0.04 to 0.08 against the three DA truths, and the depth
  signals are negative (−0.14 to −0.40).
- **Particular cases** reach |ρ| 0.96–0.99 on single languages, benchmarks
  or L tiers (11–116 clusters). The `reliable (on the truth)` subsets are
  selected on the truth, so their ρ is conditional on it.
- **Language count.** Over pairs of variants sharing L, mean DA-size is 0.61
  / 0.51 / 0.51 / 0.56 / 0.56 / 0.61 at L1 / 2 / 8 / 15 / 30 / 50 (129–1269
  tasks): no monotone trend.
- **Split-half check.** 2052 of 5446 configurations chosen on one half hold
  on the other at q < 0.05; the strongest is the consecutive-checkpoint
  Kendall τ on the L8 tier against DA-ckpt (0.79 on 151 held-out tasks).

**Follow-ups**

- The catalogue rows split by scoring (accuracy vs bBPB), so the
  rank-stability lead can be read on accuracy alone; the twins now carry
  every checkpoint, so nothing blocks it.
- A per-proxy version of the strongest surrogates: the split-half winners
  concentrate at 90M and 175M and on the L8 tier.

<!-- BEGIN auto:catalogue (search.py --pool predictivity) -->
## A catalogue of surrogates beyond SNR

Numbers from the `predictivity` pool. Regenerate with `python analysis/rq04_surrogates/catalogue.py --pool predictivity` then `search.py --pool predictivity`. Surrogates are read on the proxy alone (rule 11), except `pseudo_ref_da`, which reads the 1B rung; the truths are rq02's DA (DA-size; DA-goal and DA-ckpt at every early checkpoint of the proxy; both pair sets; every pair or the pairs sharing L) and Kendall τ-b / Spearman ρ on the same rankings, over cells above chance (rule 1). ρ is a Spearman over one point per (benchmark, language) cluster, pooled over proxies and checkpoints; populations differ per configuration (rule 13) and n is in the CSVs. Definitions and sources: [`literature.md`](literature.md); method: `search.py`'s docstring.

**Main analysis (all the data).** 753,763 configurations (truths × subsets × surrogates, the threshold filters included); 447,934 have a p-value (≥ 10 units); **108,232 hold at BH q < 0.05**. A best-of-many ρ is optimistic even when it is significant; the extra below measures by how much.

**The ceiling** — each truth against itself re-read at 90 % (benchmarks):

| truth | ρ | ρ over cells | units |
|---|---|---|---|
| DA-size da (multi-axis) | 0.92 | 0.83 | 420 |
| DA-goal da (multi-axis) | 0.95 | 0.83 | 420 |
| DA-ckpt da (multi-axis) | 0.99 | 0.89 | 420 |
| DA-size tau_b (multi-axis) | 0.91 | 0.82 | 420 |
| DA-size rho (multi-axis) | 0.92 | 0.84 | 420 |
| DA-goal tau_b (multi-axis) | 0.95 | 0.82 | 420 |
| DA-goal rho (multi-axis) | 0.95 | 0.84 | 420 |
| DA-ckpt tau_b (multi-axis) | 0.98 | 0.88 | 420 |
| DA-ckpt rho (multi-axis) | 0.99 | 0.89 | 420 |

**Strongest surrogate per truth**, every benchmark:

| truth | surrogate | ρ | q | ρ over cells | ρ within stratum | units |
|---|---|---|---|---|---|---|
| DA-ckpt da (mono-axis, pairs within L1) | `item_total_corr` | 0.78 | 6.2e-04 | 0.20 | 0.19 | 24 |
| DA-ckpt da (mono-axis, pairs within L15) | `crossings` | -0.54 | 7.3e-14 | -0.21 | -0.20 | 185 |
| DA-ckpt da (mono-axis, pairs within L2) | `n_items` | 0.81 | 1.9e-03 | 0.30 | 0.30 | 16 |
| DA-ckpt da (mono-axis, pairs within L30) | `window_kendall` | 0.60 | 5.8e-29 | 0.25 | 0.24 | 298 |
| DA-ckpt da (mono-axis, pairs within L50) | `consecutive_kendall` | 0.87 | 8.9e-128 | 0.41 | 0.41 | 420 |
| DA-ckpt da (mono-axis, pairs within L8) | `sign_consistency_window` | 0.65 | 4.6e-09 | 0.23 | 0.21 | 74 |
| DA-ckpt da (mono-axis) | `consecutive_kendall` | 0.91 | 5.5e-155 | 0.54 | 0.55 | 420 |
| DA-ckpt da (multi-axis, pairs within L1) | `language_consensus` | 0.82 | 3.3e-04 | 0.27 | 0.27 | 24 |
| DA-ckpt da (multi-axis, pairs within L15) | `consecutive_kendall` | 0.62 | 7.5e-20 | 0.34 | 0.34 | 185 |
| DA-ckpt da (multi-axis, pairs within L2) | `snr__star_discrepancy_shifted__ckpt_abs` | 0.79 | 6.2e-04 | 0.31 | 0.31 | 17 |
| DA-ckpt da (multi-axis, pairs within L30) | `consecutive_kendall` | 0.73 | 2.7e-49 | 0.41 | 0.41 | 298 |
| DA-ckpt da (multi-axis, pairs within L50) | `consecutive_kendall` | 0.91 | 1.1e-156 | 0.52 | 0.52 | 420 |
| DA-ckpt da (multi-axis, pairs within L8) | `window_kendall` | 0.51 | 6.3e-08 | 0.26 | 0.24 | 115 |
| DA-ckpt da (multi-axis) | `consecutive_kendall` | 0.97 | 8.7e-244 | 0.71 | 0.74 | 420 |
| DA-ckpt rho (multi-axis) | `consecutive_kendall` | 0.96 | 4.1e-232 | 0.69 | 0.72 | 420 |
| DA-ckpt tau_b (multi-axis) | `consecutive_kendall` | 0.97 | 9.3e-245 | 0.68 | 0.71 | 420 |
| DA-goal da (mono-axis, pairs within L1) | `rung_gap_corr` | 0.77 | 3.3e-04 | 0.32 | 0.32 | 24 |
| DA-goal da (mono-axis, pairs within L15) | `tie_rate_items` | -0.26 | 1.5e-01 | -0.02 | -0.01 | 53 |
| DA-goal da (mono-axis, pairs within L2) | `snr__star_discrepancy_shifted__ckpt_abs` | 0.78 | 3.5e-03 | 0.28 | 0.28 | 16 |
| DA-goal da (mono-axis, pairs within L30) | `snr__star_discrepancy_shifted__kfold_abs` | 0.38 | 3.8e-07 | 0.09 | 0.09 | 201 |
| DA-goal da (mono-axis, pairs within L50) | `consecutive_kendall` | 0.74 | 4.8e-71 | 0.33 | 0.34 | 420 |
| DA-goal da (mono-axis, pairs within L8) | `snr__iqr__kfold_abs` | 0.47 | 3.3e-03 | 0.10 | 0.09 | 50 |
| DA-goal da (mono-axis) | `consecutive_kendall` | 0.73 | 6.7e-69 | 0.41 | 0.42 | 420 |
| DA-goal da (multi-axis, pairs within L1) | `n_items` | 0.75 | 9.2e-03 | 0.47 | 0.47 | 15 |
| DA-goal da (multi-axis, pairs within L15) | `n_pairs` | -0.51 | 3.3e-12 | -0.18 | -0.20 | 185 |
| DA-goal da (multi-axis, pairs within L2) | `snr__star_discrepancy_shifted__ckpt_abs` | 0.78 | 2.4e-03 | 0.30 | 0.30 | 16 |
| DA-goal da (multi-axis, pairs within L30) | `pseudo_ref_da` | 0.63 | 5.9e-33 | 0.29 | 0.29 | 298 |
| DA-goal da (multi-axis, pairs within L50) | `consecutive_kendall` | 0.82 | 1.8e-99 | 0.45 | 0.46 | 420 |
| DA-goal da (multi-axis, pairs within L8) | `snr__iqr__kfold_abs` | 0.42 | 9.8e-03 | 0.09 | 0.09 | 50 |
| DA-goal da (multi-axis) | `sign_persistence` | 0.87 | 8.7e-130 | 0.63 | 0.63 | 420 |
| DA-goal rho (multi-axis) | `sign_persistence` | 0.87 | 3.0e-124 | 0.60 | 0.61 | 420 |
| DA-goal tau_b (multi-axis) | `sign_persistence` | 0.86 | 1.4e-120 | 0.58 | 0.59 | 420 |
| DA-size da (mono-axis, pairs within L1) | `bpb_corr_training` | 0.73 | 3.3e-04 | 0.35 | 0.34 | 24 |
| DA-size da (mono-axis, pairs within L15) | `projected_flip_rate` | -0.36 | 5.3e-03 | 0.02 | 0.02 | 81 |
| DA-size da (mono-axis, pairs within L2) | `snr__rel_star_discrepancy__ckpt_abs` | 0.79 | 2.8e-03 | 0.32 | 0.33 | 16 |
| DA-size da (mono-axis, pairs within L30) | `pseudo_ref_da` | 0.32 | 6.8e-07 | 0.12 | 0.13 | 274 |
| DA-size da (mono-axis, pairs within L50) | `consecutive_kendall` | 0.65 | 2.4e-50 | 0.35 | 0.35 | 420 |
| DA-size da (mono-axis, pairs within L8) | `signal__gini` | 0.35 | 9.7e-03 | 0.02 | 0.02 | 74 |
| DA-size da (mono-axis) | `bpb_rank_agreement` | 0.68 | 1.4e-56 | 0.41 | 0.42 | 420 |
| DA-size da (multi-axis, pairs within L1) | `snr__rel_std__kfold_abs` | 0.71 | 1.5e-02 | 0.37 | 0.37 | 16 |
| DA-size da (multi-axis, pairs within L15) | `n_pairs` | -0.46 | 5.4e-10 | -0.21 | -0.23 | 185 |
| DA-size da (multi-axis, pairs within L2) | `snr__rel_star_discrepancy__projection_depth` | 0.79 | 2.4e-03 | 0.33 | 0.34 | 16 |
| DA-size da (multi-axis, pairs within L30) | `pseudo_ref_da` | 0.60 | 5.2e-29 | 0.33 | 0.32 | 298 |
| DA-size da (multi-axis, pairs within L50) | `consecutive_kendall` | 0.77 | 1.6e-80 | 0.46 | 0.47 | 420 |
| DA-size da (multi-axis, pairs within L8) | `signal__gini` | 0.32 | 2.3e-02 | -0.01 | -0.01 | 74 |
| DA-size da (multi-axis) | `sign_persistence` | 0.84 | 9.1e-109 | 0.63 | 0.64 | 420 |
| DA-size rho (multi-axis) | `sign_persistence` | 0.84 | 3.1e-108 | 0.60 | 0.61 | 420 |
| DA-size tau_b (multi-axis) | `sign_persistence` | 0.82 | 3.4e-101 | 0.59 | 0.60 | 420 |

**The AllenAI grid** (22 signals × 6 noises, DA, every pair, benchmarks): best combination against AllenAI's own `rel_std` / checkpoint noise:

| truth | best signal / noise | ρ | rel_std / ckpt_rel ρ |
|---|---|---|---|
| DA-ckpt | `star_discrepancy_shifted / kfold_rel` | 0.67 | 0.65 |
| DA-goal | `aad / ckpt_abs` | 0.62 | 0.62 |
| DA-size | `aad / ckpt_abs` | 0.59 | 0.59 |

**Particular cases** — the strongest significant configurations outside "every benchmark", two per (truth, subset type); `reliable (on the truth)` subsets are selected on the truth itself, so their ρ is conditional on it; all of them in `surrogate_correlations.csv` (`q` column):

| truth | subset | surrogate | ρ | q | ρ over cells | units |
|---|---|---|---|---|---|---|
| DA-ckpt rho (multi-axis) | language: vi | `crossings` | -0.99 | 3.3e-04 | -0.65 | 12 |
| DA-ckpt rho (multi-axis) | language: ru | `consecutive_kendall_late` | 0.99 | 3.3e-04 | 0.84 | 13 |
| DA-goal da (multi-axis) | language: vi | `sign_persistence` | 0.99 | 3.3e-04 | 0.70 | 12 |
| DA-goal da (multi-axis) | language: vi | `da_ckpt_mean` | 0.99 | 3.3e-04 | 0.70 | 12 |
| DA-ckpt da (multi-axis) | language: vi | `consecutive_kendall` | 0.99 | 3.3e-04 | 0.70 | 12 |
| DA-ckpt da (multi-axis) | language: pt | `monotonicity_steps` | 0.98 | 3.3e-04 | 0.49 | 11 |
| DA-ckpt tau_b (multi-axis) | language: pt | `monotonicity_steps` | 0.98 | 3.3e-04 | 0.44 | 11 |
| DA-ckpt tau_b (multi-axis) | language: vi | `crossings` | -0.98 | 3.3e-04 | -0.65 | 12 |
| DA-size tau_b (multi-axis) | language: hi | `settling_time` | -0.98 | 3.3e-04 | -0.52 | 12 |
| DA-size rho (multi-axis) | language: hi | `settling_time` | -0.98 | 3.3e-04 | -0.54 | 12 |
| DA-ckpt rho (multi-axis) | benchmark: bbpb_rf_global_mmlu_full | `crossings` | -0.98 | 3.3e-04 | -0.60 | 29 |
| DA-ckpt rho (multi-axis) | benchmark: bbpb_rf_global_mmlu_full | `consecutive_kendall` | 0.98 | 3.3e-04 | 0.60 | 29 |
| DA-ckpt da (mono-axis, pairs within L15) | benchmark: multiblimp | `signal__quartile_deviation` | 0.97 | 3.3e-04 | 0.15 | 11 |
| DA-size da (multi-axis) | language: hi | `settling_time` | -0.97 | 3.3e-04 | -0.53 | 12 |
| DA-goal rho (multi-axis) | language: zh | `pseudo_ref_da` | 0.97 | 3.3e-04 | 0.75 | 15 |
| DA-ckpt tau_b (multi-axis) | benchmark: bbpb_xnli | `crossings` | -0.97 | 3.3e-04 | -0.58 | 15 |
| DA-ckpt da (multi-axis) | benchmark: bbpb_xnli | `consecutive_kendall` | 0.97 | 3.3e-04 | 0.58 | 15 |
| DA-ckpt tau_b (multi-axis) | benchmark: bbpb_xnli | `consecutive_kendall` | 0.97 | 3.3e-04 | 0.58 | 15 |
| DA-size da (mono-axis) | language: fr | `da_ckpt_half` | 0.97 | 3.3e-04 | 0.40 | 13 |
| DA-ckpt da (multi-axis) | tier: L8 | `consecutive_kendall` | 0.97 | 1.0e-65 | 0.71 | 116 |
| DA-goal tau_b (multi-axis) | language: zh | `pseudo_ref_da` | 0.96 | 3.3e-04 | 0.74 | 15 |
| DA-ckpt tau_b (multi-axis) | tier: L8 | `crossings` | -0.96 | 5.0e-65 | -0.66 | 116 |
| DA-ckpt tau_b (multi-axis) | tier: L8 | `consecutive_kendall` | 0.96 | 8.2e-65 | 0.67 | 116 |
| DA-ckpt da (multi-axis) | stage: early (<= 50 %) | `consecutive_kendall` | 0.96 | 1.0e-233 | 0.71 | 420 |
| DA-size rho (multi-axis) | language: ar | `monotonicity_steps` | 0.96 | 3.3e-04 | 0.53 | 13 |
| DA-ckpt da (multi-axis) | reliable (on the truth): 66_either | `consecutive_kendall` | 0.96 | 5.4e-214 | 0.67 | 390 |
| DA-ckpt da (multi-axis) | tier: L8 | `crossings` | -0.96 | 1.0e-62 | -0.66 | 116 |
| DA-ckpt rho (multi-axis) | tier: L8 | `consecutive_kendall` | 0.96 | 1.2e-62 | 0.67 | 116 |
| DA-ckpt tau_b (multi-axis) | reliable (on the truth): 66_either | `crossings` | -0.96 | 2.0e-212 | -0.66 | 390 |
| DA-ckpt da (multi-axis) | reliable (on the truth): 66_either | `crossings` | -0.96 | 2.0e-212 | -0.67 | 390 |

**Per language count** (pairs of variants sharing L; benchmarks):

| truth | L | mean DA | sd | tasks | best surrogate | ρ |
|---|---|---|---|---|---|---|
| DA-ckpt | 1 | 0.70 | 0.25 | 136 | `language_consensus` | 0.82 |
| DA-ckpt | 2 | 0.61 | 0.25 | 136 | `snr__star_discrepancy_shifted__ckpt_abs` | 0.79 |
| DA-ckpt | 8 | 0.60 | 0.26 | 427 | `window_kendall` | 0.51 |
| DA-ckpt | 15 | 0.64 | 0.24 | 675 | `consecutive_kendall` | 0.62 |
| DA-ckpt | 30 | 0.63 | 0.20 | 966 | `consecutive_kendall` | 0.73 |
| DA-ckpt | 50 | 0.68 | 0.25 | 1294 | `consecutive_kendall` | 0.91 |
| DA-goal | 1 | 0.59 | 0.27 | 129 | `n_items` | 0.75 |
| DA-goal | 2 | 0.51 | 0.24 | 129 | `snr__star_discrepancy_shifted__ckpt_abs` | 0.78 |
| DA-goal | 8 | 0.49 | 0.24 | 250 | `snr__iqr__kfold_abs` | 0.42 |
| DA-goal | 15 | 0.57 | 0.28 | 658 | `n_pairs` | -0.51 |
| DA-goal | 30 | 0.55 | 0.22 | 947 | `pseudo_ref_da` | 0.63 |
| DA-goal | 50 | 0.60 | 0.26 | 1269 | `consecutive_kendall` | 0.82 |
| DA-size | 1 | 0.61 | 0.26 | 129 | `snr__rel_std__kfold_abs` | 0.71 |
| DA-size | 2 | 0.51 | 0.25 | 129 | `snr__rel_star_discrepancy__projection_depth` | 0.79 |
| DA-size | 8 | 0.51 | 0.24 | 250 | `signal__gini` | 0.32 |
| DA-size | 15 | 0.56 | 0.28 | 658 | `n_pairs` | -0.46 |
| DA-size | 30 | 0.56 | 0.22 | 947 | `pseudo_ref_da` | 0.60 |
| DA-size | 50 | 0.61 | 0.26 | 1269 | `consecutive_kendall` | 0.77 |

**Extra: held-out confirmation.** The (benchmark, language) clusters split in two halves; 5446 configurations ranked best on one half (within-stratum ρ) were tested once on the other (one-sided cluster permutation test, BH): **2052 hold at q < 0.05**. The strongest:

| truth | subset | surrogate | ρ disc. | ρ val. [90 %] | q | tasks val. |
|---|---|---|---|---|---|---|
| DA-ckpt da (multi-axis) | tier: L8 | `consecutive_kendall` | 0.75 | 0.79 [0.75, 0.82] | 2.0e-03 | 151 |
| DA-ckpt da (multi-axis) | proxy: 90M | `consecutive_kendall` | 0.79 | 0.78 [0.76, 0.80] | 2.0e-03 | 494 |
| DA-ckpt rho (multi-axis) | proxy: 90M | `consecutive_kendall` | 0.76 | 0.76 [0.74, 0.78] | 2.0e-03 | 494 |
| DA-ckpt da (multi-axis) | tier: L30 | `consecutive_kendall` | 0.77 | 0.76 [0.73, 0.78] | 2.0e-03 | 151 |
| DA-ckpt tau_b (multi-axis) | tier: L8 | `consecutive_kendall` | 0.70 | 0.76 [0.72, 0.79] | 2.0e-03 | 151 |
| DA-ckpt da (multi-axis) | proxy: 175M | `consecutive_kendall` | 0.77 | 0.76 [0.73, 0.78] | 2.0e-03 | 500 |
| DA-ckpt rho (multi-axis) | tier: L8 | `consecutive_kendall` | 0.71 | 0.76 [0.71, 0.79] | 2.0e-03 | 151 |
| DA-ckpt da (multi-axis) | proxy: 90M | `crossings` | -0.77 | -0.76 [-0.78, -0.73] | 2.0e-03 | 494 |
| DA-ckpt tau_b (multi-axis) | proxy: 90M | `consecutive_kendall` | 0.76 | 0.76 [0.73, 0.78] | 2.0e-03 | 494 |
| DA-ckpt da (multi-axis) | tier: L8 | `consecutive_kendall_late` | 0.72 | 0.75 [0.71, 0.79] | 2.0e-03 | 151 |
| DA-ckpt da (multi-axis) | stage: late (60-90 %) | `consecutive_kendall_late` | 0.77 | 0.75 [0.73, 0.77] | 2.0e-03 | 555 |
| DA-ckpt da (multi-axis) | stage: late (60-90 %) | `consecutive_kendall` | 0.76 | 0.75 [0.73, 0.77] | 2.0e-03 | 555 |
| DA-ckpt da (multi-axis) | tier: L8 | `sign_consistency_window` | 0.71 | 0.75 [0.70, 0.78] | 2.0e-03 | 151 |
| DA-ckpt da (multi-axis) | proxy: 90M | `consecutive_kendall_late` | 0.76 | 0.75 [0.72, 0.77] | 2.0e-03 | 494 |
| DA-ckpt rho (multi-axis) | proxy: 175M | `consecutive_kendall` | 0.75 | 0.75 [0.72, 0.77] | 2.0e-03 | 500 |

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
      *Implications (2026-10-07 15:51 report).* Only English has a best
      definition on both splits under DA-size and only English and Russian
      under DA-ckpt, so the per-language agreement (100 % / 100 %) is 1 of 1
      and 2 of 2 languages. The global ranking ρ (0.81 DA-size, 0.84 DA-ckpt) is
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
