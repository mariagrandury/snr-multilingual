# NUMBERS2 — hand-written numbers audit (rq03, rq05, rq06, rq07, rq08, rq09, analysis/README.md, src/signal-and-noise/CLAUDE.md)

Snapshot: the committed tables (2026-09-30 23:54 report). Paths are relative to `src/signal-and-noise/analysis/` unless they start with `src/`.
Only hand-written prose was checked; auto blocks were skipped. Verdicts: MATCH / MISMATCH / CANNOT-LOCATE. STALE = the number is right for the 36-sweep but the sentence now reads as being about the ladder.

## rq03_noise_and_snr/README.md

| # | line | claim | CSV + computation | got | verdict |
|---|---|---|---|---|---|
| 1 | 7, 28 (analysis/README) | 22 SNR definitions | `rq03/pretraining/predictivity/snr_variants_definitions.csv` rows | 22 | MATCH |
| 2 | 15-16, 28-30 | holdout = seeds 64/313 vs 1904 on deep scheme-A 175M/600M, L ∈ {1,2,50} | `configs/models.json` pools `predictivity_seeds_train/_test` | same | MATCH |
| 3 | 137-142 | 2026-09-01: ckpt noise median 0.004; 600M across-L range 0.022 ≈ 4.4×; 56/60 > 2×; xwinograd_en 17×, hellaswag_ru 13× | `plan/todos/status-09-01.md` l.228-230 (git) | same numbers | MATCH (but the cited path `plan/status-09-01.md` moved to `plan/todos/status-09-01.md`) |
| 4 | 143-147 | 14-Sept seed effect Δ loss −0.061…+0.046 etc.; depth −0.009…+0.014 | no committed table or status doc carries these | — | CANNOT-LOCATE (dated, frozen) |
| 5 | 203-204 | "Replicate seeds exist only at 175M and 600M" | `rq03/pretraining/predictivity_all/effect_vs_noise.csv` rows with `seed_noise`: 175M (L1,L2,L50), 600M (L1,L2,L50), **1B (L1,L2,L30)**; models.json has `lm-1B-L{1,2,30}-deep-seed{28,1797}` | 1B has replicates too | MISMATCH |
| 6 | 205 | English tasks and BPB rest on 15 model pairs | `rq02/pretraining/predictivity_seeds_train/da_all_n_pairs_per_task_both_axes.csv`, `axes=multi-axis`, col `decision_acc_size_175M_to_600M` (English tasks, bpb_dclm = 6); DA values in `rq03/.../predictivity_seeds_train/snr_variants_per_task.csv` are multiples of 1/6 | 6 | MISMATCH |
| 7 | 206-208 | non-English benchmark: train split holds one pair, DA is 0 or 1 | same table: non-English tasks have 0 multi-axis pairs (1 only on the `seed` axes); `snr_variants_per_task.csv` has DA-size for 110 of 898 tasks, all English + bpb_dclm/bpb_macro/train_loss | 0 pairs, no DA | MISMATCH |
| 8 | 209 | median task has one pair | same table, multi-axis: 767 of 898 tasks have 0 pairs | median 0 | MISMATCH |
| 9 | 210-213 | 0 % / 0 % family agreement; ρ −0.28 (DA-size), 0.76 (DA-ckpt) | `predictivity_seeds_train__vs__predictivity_seeds_test/headline_metrics.csv` | 0/0 (n=1), −0.278, 0.762 | MATCH (the attribution "dominated by those one-pair tasks" no longer holds: one-pair tasks carry no DA; the 1-language n is the reason) |
| 10 | 217 | option A: ≥ 6 pairs = English + BPB only | same n_pairs table | English + bpb_dclm have exactly 6 | MATCH |
| 11 | 237 | 36-sweep "86-task old list" | not in a committed CSV | — | CANNOT-LOCATE |

## rq05_design_decisions/README.md (beyond 296-298)

| # | line | claim | CSV + computation | got | verdict |
|---|---|---|---|---|---|
| 12 | 26 | L ∈ {1, 2, 8, 15, 30, 50, 100} | `intervention_da_all_mono_axis.csv` L values; L100 dropped 2026-09-20 | {1,2,8,15,30,50} | MISMATCH |
| 13 | 27 | width/depth ≈ 64 deep vs ≈ 128 shallow | `src/pretrain/hyperparams/hyperparams_{deep,shallow}.json` hidden/layers | deep 51–77 (64 at 175M–1B), shallow 110–146 | MATCH (approx.) |
| 14 | 31, 262-264 | reference 1.7B at every L / every intervention | `intervention_da_all_mono_axis.csv` `reference_size` | 1.7B in all 3068 rows | MATCH |
| 15 | 37-42 | populations include `bpb_all` (all 100 languages) | `intervention_da_all_mono_axis.csv` `population` | benchmark, bpb_macro, loss, bpb_trained — no bpb_all; `intervention_da_*_by_group_*.csv` are header-only | CANNOT-LOCATE (population absent from every rq05 table on disk; flag) |
| 16 | 54, 182 | a benchmark cell needs ≥ 3 items | code constant | — | not a data number |
| 17 | 258-261 | 5 late ckpts = 25 % of a 20-ckpt run, 12.5 % of a 40-ckpt run | arithmetic | correct; but contradicts rule 4 (one window: 80/85/90/95/100 %) — stale caveat | MATCH (arithmetic) / stale |
| 18 | 299-303 | decided restriction does not help on benchmarks; bpb_macro and loss leave one item per L | `da_all_lines_mono_axis.csv` vs `da_all_lines_decided_mono_axis.csv` (DA-size, benchmark rows: within ±0.03 except zh/1B); n_items=1 per L for bpb_macro/loss | consistent | MATCH |
| 19 | 305-308 | deep wins by 1.9–2.1 sds at 90M, 1.1–1.6 at 175M; \|z\| ≤ 0.51 at 350M; shallow ahead by ≤ 1.46 at 600M; deep 0.48–1.03 at 1B, 0.81–0.95 at 1.7B | `depth_crossover.csv` | 1.94–2.10; 1.13–1.60; max \|z\| 0.513; 0.28–1.46; 0.48–1.03; 0.81–0.95 | MATCH |
| 20 | 308-310 | depth DA on trained BPB: 0.88–1.00 at 90M, 0.82–1.00 at 175M, 0.02–0.85 at 350M, 0.00–0.08 at 600M, 0.47–0.92 at 1B (L8–L50) | `intervention_da_all_mono_axis.csv`, arch / bpb_trained / frac 1.0, `decision_acc` over L8–L50 | 0.92–1.00, 0.94–1.00, 0.00–0.97, 0.00, 0.78–1.00 | MISMATCH |
| 21 | 312-315 | `bpb_all`: decided items read well | no bpb_all rows on disk | — | CANNOT-LOCATE |
| 22 | 319-321 | "37 languages agreeing" | illustrative; bpb_trained has 8/15/30/50 items per L | — | CANNOT-LOCATE |
| 23 | 323-325 | reference changes between lines (600M or 1B for some interventions) | `reference_size` column | 1.7B everywhere | MISMATCH |
| 24 | 326-329 | median-of-sd biased low by ~17 % | no table | — | CANNOT-LOCATE |
| 25 | 331-332 | `da_all_lines_flops_mono_axis` = 5 × 3 lines of 60 cells | `da_all_lines_flops_mono_axis.csv`: 13 `row`s × 59 points | 13 × 59 | MISMATCH |

## rq06_language_transfer/README.md (beyond 171-177)

| # | line | claim | CSV + computation | got | verdict |
|---|---|---|---|---|---|
| 26 | 24-25, 99 | 100 validation languages | `rq5_transfer.csv` distinct `task` per L | 100 at every L | MATCH |
| 27 | 34 | own fit needs k ≥ 3 | `rq5_transfer_summary.csv` own NaN for k<3 | yes | MATCH |
| 28 | 37 | decision transfer on `bpb_untrained` | rq05 table has no bpb_untrained population (block shows bpb_trained only) | — | CANNOT-LOCATE (flag) |
| 29 | 110-112 | `transfer_da_all_lines_mono_axis.csv` still empty | file = header only | empty | MATCH (the stated reason — list decisions below three items — CANNOT-LOCATE; `transfer_da_all_by_L_mono_axis.csv` has values for the lists at L8/15/30) |
| 30 | 177-179 | at 1B macro 0.93 hellaswag, 0.89 LAMBADA and INCLUDE-rfgm, 0.81 Global-MMLU-rf; English 0.87, 0.78, —, 0.69; multiblimp en 0.40 vs macro 0.68 | `pretraining/predictivity/language_panel.csv` (`reliability`, size 1B) | 0.934, 0.890, 0.890, 0.813; en 0.868, 0.780, n/a, 0.692; 0.396 vs 0.681 | MATCH |
| 31 | 180-182 | macro above every single language: hellaswag, Global-MMLU-rf, INCLUDE-rf/rfgm, xstorycloze (at 1B) | same, macro vs max single language at 1B | above: hellaswag, multiblimp, rfgm_belebele, rfgm_include_base_44, xstorycloze; **tie**: rf_global_mmlu_full (0.813 = de), lambada (0.890 = de); **below**: rf_include_base_44 (0.692 < ru 0.725) | MISMATCH |
| 32 | 182-183 | xnli, xwinograd: nothing reads the reference at 1B | same: best 0.54 (xnli de), 0.51 (xwinograd zh) | MATCH |
| 33 | 184-187 | English misreads more often than ANY other single panel language; macro safest at EVERY size | same, `all benchmarks` row: ja < en at every size (0.538/0.489/0.518/0.560/0.551 vs en 0.545/0.536/0.573/0.596/0.605); macro beaten by it at 90M (0.659 vs 0.655) and de at 175M (0.694 vs 0.659), tied with de at 1B (0.670) | MISMATCH |
| 34 | 122-123 | temperature, ZH, ES have one L each | `transfer_da_all_by_L_mono_axis.csv` | temperature L50 only; zh, es L2 | MATCH |
| 35 | 126-127 | reference is 1.7B only for the lists | rq05 `reference_size` | 1.7B for every intervention | MISMATCH |
| 36 | 202 | 36-sweep evaluated 12 languages | `configs/languages.json` `main` | 12 | MATCH |

## rq07_external_frameworks/README.md

| # | line | claim | CSV + computation | got | verdict |
|---|---|---|---|---|---|
| 37 | 9-12 | ladder overlap = arc_easy, arc_challenge, hellaswag, MMLU; 36-sweep ALSO csqa, openbookqa, piqa | `pretraining/predictivity/task_overlap.csv` `shared` | ladder: arc_challenge, arc_easy, csqa, hellaswag, mmlu, openbookqa (6; 3 after gate: arc_easy, arc_challenge, hellaswag per `top_apertus.csv`); 36-sweep: those + piqa (7) | MISMATCH |
| 38 | 31-32 | size pairs 175M↔150M, 350M↔300M, 600M↔750M, 1B↔1B, 1.7B↔1B | `pearson_r_size_sweep.csv` | same | MATCH |
| 39 | 35, 58 | DataDecide 25 recipes, sizes 150M/300M/750M/1B | external fact; size_sweep sizes match | — | MATCH (sizes) |
| 40 | 74, 78 | post-alias shared set = 7 standalone English tasks | 36-sweep `task_overlap.csv` (custom_swissai_hf, seeds_*, external) | 7 | MATCH for the 36-sweep (methodology section describes it) |
| 41 | 95, 154 | "the 7-task overlap is the binding constraint" | ladder `task_overlap.csv`: 6 shared, 3 after gate | 6/3 | STALE (true for 36-sweep only) |
| 42 | 96 | AllenAI `core` has ~178 tasks Apertus does not evaluate | `task_overlap.csv`: 234 AllenAI-only (241 total) | 234 | CANNOT-LOCATE ("core" split not in repo) |
| 43 | 100-109 | missing counts: mmlu:mc 53, mmlu_pro 19, BBH 27, AGI 19, Math 14, core 10, gen QA 8, arc/hs:mc + code 7 | 36-sweep `task_overlap.csv` AllenAI-only: mmlu_*:mc 57, mmlu_pro 15 (1 + 14), bbh 27, agi_eval 19, gsm*/minerva* 14 | 3 of 5 checkable agree | CANNOT-LOCATE (source "core" split ambiguous; note 57 vs 53 and 15 vs 19) |
| 44 | 116-117, 168 | 3 shared today (after the gate) | `shared_task_agreement.csv` n_shared | 3 | MATCH |
| 45 | 175-176 | external clears gate on 6 of 7 vs 4 custom | `all/external` and `custom_swissai_hf` `shared_task_agreement.csv` | 6, 4 | MATCH |
| 46 | 188-190 | 0.98/1.00/4, 1.00/1.00/4, +0.892/+0.829/6 | `shared_task_agreement.csv` (3 pools) | 0.981/1.0, 0.996/1.0, 0.892/0.829 | MATCH |
| 47 | 192-195 | hellaswag/piqa top, arc_challenge/csqa bottom; top-5 Jaccard 0.67; discrepancy 0.80, dispersion_shifted 0.78, rel_std ≈ 0.28 | `all/external/top_apertus.csv`, `top_allenai.csv`, `agreement.csv`, `pearson_r_per_variant.csv` | yes; 0.667; 0.804, 0.784, 0.284 | MATCH |
| 48 | 201-203 | 7 tasks, n_shared 6, 1 aliased MMLU | external `task_overlap.csv` | 7 / 6 | MATCH |
| 49 | 216-219, 229-232 | 0.98/1.00/4; 0.90 → 1.00 → 0.98; projection 0.90/0.80; mpsd 1.00/1.00 | `seeds_1904`, `seeds_28_1797`, `seeds_28_1797_1904`, `custom_swissai_hf` `shared_task_agreement.csv` | 0.901/0.80, 0.998/1.0, 0.981/1.0, 0.996/1.0 | MATCH |

## rq08_subset_selection/README.md

| # | line | claim | CSV + computation | got | verdict |
|---|---|---|---|---|---|
| 50 | 26 | ladder multilingual families span up to 100 languages | `pretraining/predictivity/per_benchmark.csv` `n_subtasks` | max 69 (include_v2_en country subsets), rfgm_belebele 59, bpb family 8 | CANNOT-LOCATE (100 = BPB validation languages, but the swept `bpb` family has 8) |
| 51 | 37 | global_mmlu languages the pool carries: 37 | `rq00/.../above_random_mask.csv` family global_mmlu_full | 37 (0 above chance at 1.7B; Case 2/3 sweep 3–4 gated languages) | MATCH |
| 52 | 153-158 | external table (6 rows) | `all/external/summary.csv` top 6 by gain | identical | MATCH |
| 53 | 164 | high_school_chemistry recurs here | same | yes | MATCH |
| 54 | 183 | 36-sweep full ~48-subject set | `pretraining/custom_swissai_hf/global_mmlu_full.csv` `n_subtasks` = 61 (57 subjects + 4 roll-ups) | 57 | CANNOT-LOCATE (no 48 in any table; likely should read ~57) |
| 55 | 184 | per-item Jaccard ≈ 0.03, Spearman ≈ 0.05 | `per_sample/variance_prefilter/analysis/cross_size_subset_jaccard.csv` median, `cross_size_snr_spearman.csv` median | 0.025, 0.050 | MATCH |
| 56 | 194-205 | 36-sweep top-12 table | `pretraining/custom_swissai_hf/summary.csv` | identical | MATCH |

## rq09_benchmark_design/README.md

| # | line | claim | CSV + computation | got | verdict |
|---|---|---|---|---|---|
| 57 | 28, 164 | twelve 36-sweep families | `analyze.py` FAMILY_META non-rf, non-ladder keys | 12 | MATCH |
| 58 | 55-58 | PAWS low despite binary; MultiBLiMP sharpest; HellaSwag escapes the 4-option penalty that sinks ARC | `pretraining/predictivity/per_family_snr.csv` (= auto table): xwinograd 1.41 > multiblimp 1.24 > xstorycloze 1.20 > paws 1.19; hellaswag 0.40 < arc 0.47 | contradicted on the ladder (and PAWS is 2nd in the 36-sweep, 2nd in external) | MISMATCH |
| 59 | 58-60 | XStoryCloze high, Belebele low | same: 1.20 vs 0.47 | MATCH |
| 60 | 52-54 | longer options are sharper | `group_stats.csv`: option_len ρ = −0.13 (p 0.73), context_len ρ = −0.25 (p 0.52) | no support (wrong sign, n.s.) | MISMATCH (qualitative; flag, no number to replace) |
| 61 | 159 | MMLU ~57 subjects | 57 | MATCH |
| 62 | 180 | external passes 11 families incl. hellaswag, global_mmlu_full, arc, belebele | `all/external/per_family_snr.csv` | 11, yes | MATCH |
| 63 | 192-202 | external family table | same | identical | MATCH |
| 64 | 213, 218-222 | H 1.78 → 0.05; external H/p | `all/external/group_stats.csv`, `custom_swissai_hf/group_stats.csv` | 1.78 → 0.045; 0.05/0.83, 0.00/1.00, 0.17/0.68, 1.44/0.49, 0.38/0.54 | MATCH |
| 65 | 241-243, 253-273 | 9 families; H 1.78 p 0.18; format 0.00/1.00; curation 0.50/0.78; family table | `custom_swissai_hf/per_family_snr.csv`, `group_stats.csv` | identical | MATCH |

## analysis/README.md

| # | line | claim | CSV + computation | got | verdict |
|---|---|---|---|---|---|
| 66 | 7-8 | Ten research questions in four themes, one folder each | the table itself: themes A–E (5), rq00–rq10 (11 questions) in 12 folders | 11 / 5 / 12 | MISMATCH |
| 67 | 24-35 | 12 main-figure links | file existence | all 12 exist | MATCH |
| 68 | 35 | rq10 "waiting for the 3B evaluations" | `rq10/pretraining/predictivity/above_reference_3B.csv` filled (reference 3B, 120 rows); rq10 README: the four 3B L8/L15 cells are in | filled | MISMATCH |
| 69 | 50-51 | sizes 175M–1.7B | rule 10, `models.json` snr.small_sizes starts at 90M | 90M–1.7B | MISMATCH |
| 70 | 81-82 | one reference (1.7B; L2 ES stops at 1B) | RULES.md rule 9; rq05 `reference_size` = 1.7B for es | ES reaches 1.7B (2026-09-26) | MISMATCH |
| 71 | 86-87 | twenty promoted probe families since 2026-09-23 | RULES.md "Promoted 2026-09-23" | twenty of twenty-one | MATCH |
| 72 | 132 | ~140 accuracy-vs-FLOPs grids | rq00 PNG counts: per_language 122, per_benchmark 42, score_curves 50 | ambiguous | CANNOT-LOCATE |

## src/signal-and-noise/CLAUDE.md

| # | line | claim | CSV + computation | got | verdict |
|---|---|---|---|---|---|
| 73 | 134 | ~210 proxy-only surrogates | `rq04/pretraining/predictivity/surrogate_definitions.csv` | 211 | MATCH |
| 74 | 158 | DA-size 0.54 (90M) → 0.56 (1B) | `rq02/.../scale_convergence_da_size_multi_axes.csv` all benchmarks / all pairs | 0.540 → 0.563 | MATCH |
| 75 | 159 | jackknife ±0.02 | same, `lo`/`hi` | ±0.022–0.024 at 90M–600M, **±0.034 at 1B** | MISMATCH (minor) |
| 76 | 159 | 0.64 → 0.76 of the above_66 figures | rq02 README auto block l.443 (0.644 → 0.759) | MATCH |
| 77 | 162-163 | DA-ckpt 0.74–0.82 at 90 %; seed null up to 0.75 | `early_small_da_ckpt_by_L_multi_axes.csv` L=all, frac 0.9: 0.736 (1.7B)–0.818 (350M); `seed_uncertainty_da_all_seed_null.csv` view seed null max 0.754 | MATCH |
| 78 | 166 | r ≥ 0.96 over 1,883 cells | `agreement_da_size_correlation_multi_axes.csv` size=all | 1883 cells, min r 0.964 | MATCH |
| 79 | 168 | tie convention moves verdict on 2–7 % | `agreement_da_size_cut_sensitivity_multi_axes.csv` flip_share | 2.3 %–6.8 % | MATCH |
| 80 | 170 | token share ρ 0.02–0.27 | `reliability_da_size_vs_language_share_multi_axes.csv`, Spearman(reliability, share_L50) per size | 0.024–0.270 | MATCH |
| 81 | 226 | predictivity = lm-{175M…1.7B} | rule 10 / models.json small_sizes | 90M…1.7B | MISMATCH |
| 82 | 227 | 64/313 at 175M/600M, 28/1797 at 1B ×3 cells | models.json + effect_vs_noise.csv seed rows | yes | MATCH |
| 83 | 228 | seeds_train = "the only cells with replicates" | 1B L1/L2/L30 cells have seeds 28/1797 | not the only ones | MISMATCH |
| 84 | 245 | small_sizes 175M–1B | `configs/models.json` snr.small_sizes | 90M–1B | MISMATCH |
| 85 | 245-246 | target_size 1.7B, da_early_fracs nine tenths, noise_window 0.2, noise_grid 20, min_pairs 3, min_lang_tasks 3 | `configs/models.json` snr | same | MATCH |
| 86 | 515-516 | passers 0.80 vs 0.55 at 1B | recomputed above_66 on gated multi-axis 1B DA-size: 0.75 vs 0.45 (late-reduction definition not reproducible from one table) | — | CANNOT-LOCATE |
| 87 | 249-250, 279-286, 393-399 | historical bug-history numbers (6 vs 17 languages; 649 vs 775; −0.017 → +0.170, etc.) | historical | — | not checked (history) |

## MISMATCH corrections (ready to paste)

M1. rq03 README:203-209 — replace
"Replicate seeds exist only at 175M and 600M, so both splits hold those two sizes and DA-size there is 175M → 600M (the 1.7B reference never enters). English tasks and BPB rest on 15 model pairs. A non-English benchmark is trained only in the L50 cell, so its train split holds one pair — the two seeds of the same cell — and its "decision accuracy" is 0 or 1 and measures seed noise, not a ranking. The median task has one pair."
with
"The holdout's replicate seeds (64/313) exist only at 175M and 600M (the 1B ×3 cells carry seeds 28/1797 and are in neither split), so both splits hold those two sizes and DA-size there is 175M → 600M (the 1.7B reference never enters). English tasks and English BPB rest on 6 model pairs (same seed, L ∈ {1, 2, 50}). A non-English benchmark is trained only in the L50 cell (Russian also at L2), so its train split holds no multi-axis pair — its two seeds differ in nothing but the seed — and it carries no DA at all: 110 of the 898 tasks have a DA-size value, all of them English, so the holdout ranks the variants for English alone."
(and in 213-215 "are dominated by those one-pair tasks" → "rest on English alone (one language with a best variant on both splits)")

M2. rq05 README:26 — "settings L ∈ {1, 2, 8, 15, 30, 50, 100}" → "settings L ∈ {1, 2, 8, 15, 30, 50}"

M3. rq05 README:309-310 — "— 0.88–1.00 at 90M, 0.82–1.00 at 175M, 0.02–0.85 at 350M, 0.00–0.08 at 600M, 0.47–0.92 at 1B over L8–L50 —" → "— 0.92–1.00 at 90M, 0.94–1.00 at 175M, 0.00–0.97 at 350M, 0.00 at 600M, 0.78–1.00 at 1B over L8–L50 —"

M4. rq05 README:323-325 — "The reference changes between lines (600M or 1B for some interventions, 1.7B for others; `refs` in the table): name it in the legend, and re-read every line against 1.7B once the missing 1.7B cells finish." → "The reference is 1.7B on every line since 2026-09-26 (`refs` in the table); keep naming it in the legend."

M5. rq05 README:331-332 — "(5 × 3 lines of 60 cells)" → "(13 lines of 59 points)"

M6. rq06 README:180-182 — "Where the macro lies above every single language (hellaswag, Global-MMLU-rf, INCLUDE-rf/rfgm, xstorycloze) the languages' errors are partly independent" → "Where the macro lies above every single language at 1B (hellaswag, multiblimp, Belebele-rfgm, INCLUDE-rfgm, xstorycloze; Global-MMLU-rf and LAMBADA tie the best language, INCLUDE-rf trails Russian 0.69 to 0.73) the languages' errors are partly independent"

M7. rq06 README:184-187 — "A developer who evaluates a multilingual recipe on English benchmarks alone misreads the multilingual decision more often than one who evaluates any other single panel language; the macro over the readable panel is the safest proxy at every size." → "A developer who evaluates a multilingual recipe on English benchmarks alone misreads the multilingual decision more often than one who evaluates any other single panel language except Japanese, which reads lowest at every size; the macro over the readable panel is the safest proxy from 350M up (tied with German at 1B, 0.67), while Italian at 90M (0.66) and German at 175M (0.69) read the decision better than it."
(Note: the same Japanese fact contradicts "English alone is the worst single-language proxy" at line 171 — ja 0.55 < en 0.61 at 1B, as line 174 itself says.)

M8. rq06 README:126-128 — "The same caveats as rq05: items are correlated and the reference is 1.7B only for the lists; bootstrap over L and re-read the other interventions against 1.7B when their cells exist." → "The same caveats as rq05: items are correlated; bootstrap over L. The reference is 1.7B for every intervention since 2026-09-26."

M9. rq07 README:10-12 — "On the ladder's auto task set they are `arc_easy`, `arc_challenge`, `hellaswag` and MMLU (via the Global-MMLU English split); the 36-sweep also shared `csqa`, `openbookqa` and `piqa`." → "On the ladder's auto task set they are `arc_easy`, `arc_challenge`, `hellaswag`, `csqa` (via `commonsense_qa`), `openbookqa` and MMLU (via the Global-MMLU English split), of which the above-random gate keeps `arc_easy`, `arc_challenge` and `hellaswag`; the 36-sweep also shared `piqa`."
(STALE, optional: l.95 "The 7-task overlap is the binding constraint" → "The 6-task overlap (3 after the gate; 7 on the 36-sweep) is the binding constraint"; l.154 "widen the 7-task shared universe" → "widen the 6-task shared universe".)

M10. rq09 README:55-58 — "Illustrations at fixed option count: PAWS (options `Yes`/`No`, ~2 chars) is low despite being binary; MultiBLiMP (full-sentence minimal pairs) is the sharpest; HellaSwag (long 4-option completions) escapes the 4-option penalty that sinks ARC (short noun-phrase options)." → "On the ladder the illustrations do not hold: PAWS (options `Yes`/`No`, ~2 chars) is among the sharpest families (1.19 at 1.7B, behind xwinograd 1.41, MultiBLiMP 1.24 and XStoryCloze 1.20), and HellaSwag (long 4-option completions, 0.40) sits below ARC (short noun-phrase options, 0.47), inside the 4-option band (0.30–0.56)."
(Also: l.52-54 "longer options concentrate more discriminating tokens" is not supported by `group_stats.csv` — option length ρ = −0.13, p = 0.73; context length ρ = −0.25, p = 0.52, over 16 families.)

M11. analysis/README.md:7-8 — "Ten research questions in four themes, one folder each;" → "Eleven research questions in five themes, one folder each (rq00 has two, the gate and the task reformulation);"

M12. analysis/README.md:35 — "(the only reader of `above_reference=True`; waiting for the 3B evaluations)?" → "(the only reader of `above_reference=True`; the four 3B L8/L15 cells are evaluated, L30/L50 still training)?"

M13. analysis/README.md:50-51 — "sizes 175M–1.7B × L ∈ {1, 2, 8, 15, 30, 50}" → "sizes 90M–1.7B × L ∈ {1, 2, 8, 15, 30, 50}"

M14. analysis/README.md:81-82 — "one reference (1.7B; L2 ES stops at 1B)" → "one reference (1.7B at every L, L2 ZH and ES included since 2026-09-26)"

M15. CLAUDE.md:159 — "jackknife ±0.02" → "jackknife ±0.02 (±0.03 at 1B)"  (same phrase in rq02 README:27)

M16. CLAUDE.md:226 — "lm-{175M…1.7B}-L{1…50}[-schemeB]-{deep,shallow}-seed1904" → "lm-{90M…1.7B}-L{1…50}[-schemeB]-{deep,shallow}-seed1904"

M17. CLAUDE.md:228 — "(the only cells with replicates)" → "(the 64/313 replicates; the 1B ×3 cells' seeds 28/1797 are not in the holdout)"

M18. CLAUDE.md:245 — "`small_sizes` 175M–1B" → "`small_sizes` 90M–1B"

## Counts

87 rows: MATCH 49 (incl. 1 approx. and 1 arithmetic-only), MISMATCH 22 rows → 18 corrections (M1 covers 4 rows; the rq09 option-length mismatch, row 60, has no number to replace), STALE 1, CANNOT-LOCATE 13 (incl. the bpb_all / bpb_untrained populations, absent on disk), not checked (history/constants) 2.
Missing figure embeds: none. Every `![](…)` embed and every GitHub `blob/main` link in the rq03/05/06/07/08/09 READMEs resolves to a file on disk, and so do analysis/README's 12 main-figure links. One stale path: rq03:138 cites `plan/status-09-01.md`, now `plan/todos/status-09-01.md`.
