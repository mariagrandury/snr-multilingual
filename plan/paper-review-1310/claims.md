# Claim audit: "Small Scale, Big Decisions" (Overleaf 13-10 12h)

Source audited: `documents/overleaf/13-10-12h-full-overleaf/` (main.tex and the
sections it `\input`s; commented text ignored), compiled PDF
`documents/overleaf/build-1310/main.pdf` (43 pages, no `??` references).
Evidence: the committed analysis outputs of the 2026-10-10/11 refresh.
`A/` = `src/signal-and-noise/analysis/`. "macro" = mean over tasks
(`reliability_macro`), "pair-weighted" = pooled matching/comparable pairs
(`reliability`). Nothing was re-run; every number below was read from a CSV
or a generated .tex with pandas.

Status: **VERIFIED** (matches) · **MISMATCH** (number or wording disagrees
with the evidence; correct value given) · **UNSUPPORTED** (no evidence, or
the evidence does not support the claim) · **STALE** (number from an older
snapshot; the repo has a newer one).

---

## (A) Blocking problems (headline claims and wrong numbers)

1. **"Per-language training tokens seen ... impact decision reliability more
   than proxy model size" (intro 01:30; abstract main.tex:192 lists tokens
   as a factor; contribution 01:38 "token counts ... at which we observe
   decision reliability"; §3 intro 04:186–188; §5.4 04:252).**
   UNSUPPORTED, and contradicted by the repo. No section or appendix shows a
   token effect on DA for benchmarks. The language-tier table
   (`A/rq02_decision_accuracy/pretraining/predictivity/reliability_da_size_by_language_tier_multi_axes.csv`,
   variant `all`, macro DA-size 90M–1B) gives L8 tier 0.57–0.59, L15
   0.57–0.59, L30 0.57–0.60, L50 tier (fewest tokens) **0.65–0.69**. The
   languages with the fewest tokens rank designs best. On accuracy alone the
   tiers sit within 0.03–0.04 of each other from 600M on (rq02 README), and
   ρ(token share, DA) is +0.07 to +0.22 on accuracy and −0.42 to −0.53 with
   the bBPB twins. The only DA-vs-tokens appendix (App. D,
   `A/rq02_da_vs_train_tokens/.../da_goal_multi_axes_across_langs_bpb_paper.csv`)
   shows the opposite for BPB: 90M reaches 0.92 at its first tenth on
   ~0.008B tokens of a language, while 600M stays at 0.69–0.74 up to 0.50B.
   The paper still makes the claim in the abstract, the intro finding, the
   second contribution and the §3 intro. **Fix:** keep tokens as a factor of
   *measurability* (§4, supported), and state the negative result for DA, or
   drop it. Also drop "and at which we observe decision reliability" from
   contribution 2: no table gives token counts for DA.
2. **"... expected answer format ... impact decision reliability" (01:30)
   and "Reformulating tasks ... can improve evaluations' decision
   reliability" (conclusion 05:11).** UNSUPPORTED for DA. Reformulation is
   shown to change *measurability* only (§4.2, App. B.2). No DA comparison of
   original against RF/LLM-RF is in the paper. The letter originals are gated
   out, so no such comparison can exist on the analysed gate. **Fix:** say
   "makes the benchmark measurable, which is a prerequisite for decision
   reliability".
3. **"... more than changes to proxy model size" (01:30) vs the abstract
   (main.tex:192), which lists "proxy model size" first among the factors
   that *set* informativeness.** The two contradict each other. §5.2 says
   agreement "shows no consistent improvement with increasing model size".
   The only factor with a measured DA effect is scoring: bBPB beats accuracy
   against the 1.7B accuracy decision by +0.10/+0.06/+0.06/+0.05/+0.03
   (90M–1B) (`.../bench_bpb_da_size_summary_multi_axes.csv`, `gain`). That
   is about the size of the accuracy-only size trend (0.51 → 0.56). **Fix:**
   make the abstract and intro say the same thing. The supported version is
   "scoring and the design axis matter, proxy size (within 90M–1B) and tokens
   do not".
4. **"Statistics ... used to predict ranking agreement in monolingual
   settings, do not transfer to our multilingual ladder" (01:32).**
   UNSUPPORTED *in the paper*. §6 reports moderate correlations (ρ 0.53–0.62,
   Figure 4) and never shows a monolingual baseline. The evidence exists in
   the repo but is not `\input`:
   `A/rq13_english_only/pretraining/predictivity/english_only_allenai_snr_da_summary.csv`
   gives ρ(log SNR, DA-size) of 0.89 on DataDecide, 0.46 at K=1, 0.43 at K=2,
   0.04 at K=15 and −0.10 at K=30. `tables/app_snr_monolingual_table.tex`
   exists in the Overleaf folder but is not included. **Fix:** include that
   table (or cite those numbers in §6), and add that on accuracy alone the
   correlations are 0.28–0.39 (App. F).
5. **§4.3 "Across 431 tasks retained ... median R² 0.87" (04:124).**
   MISMATCH/STALE. `A/rq01_scaling_predictability/pretraining/predictivity_seeds/scaling_regimes.csv`
   (non-bBPB rows) has **419** accuracy+BPB tasks with a fit (README: 420),
   median size-fit R² **0.88** (0.878). "0.87" is the median over the 1,417
   (task, K) fits of 509 benchmark tasks (rq01 README). The 311 (74 %) in
   04:131 is 311/419, so the paragraph mixes two denominators. **Fix:** "Across
   the 419 accuracy and BPB tasks ..., the median R² of the model-size fits is
   0.88".
6. **§5.2 "agreement remains modest (59–61% across all pairs)" (04:229) vs
   Figure 3a.** The figure plots `reliability_macro`
   (`paper_rq2.py: Y_SIZE = "reliability_macro"`), which reads **0.60–0.64**
   for "all pairs"
   (`A/rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_above_66_either_transformation_mono_axis.csv`,
   panel DA-size: 90M 0.642, 175M 0.609, 350M 0.633, 600M 0.602, 1B 0.640).
   59–61 % is the pair-weighted `reliability` column (0.588–0.614), which
   the figure does not show. MISMATCH. **Fix:** "60–64 %", or plot
   pair-weighted. Temperature's 72–75 % holds under both columns. The
   caption's "pooled over decisions" (04:103) also describes the wrong
   aggregate.
7. **Summary box §5 (04:270–272): "On accuracy-scored benchmarks ...
   0.51–0.56 ... On the design-variant pairs that the 1.7B reference
   separates by more than two seed SDs, the share ordered correctly is 0.69
   to 0.73."** The first number is VERIFIED (accuracy only, multi-axis,
   unfiltered: 0.506–0.560, `.../da_size_definition_sensitivity_multi_axes.csv`,
   `Benchmark accuracy | current`). The second number is on **accuracy + bBPB
   twins** (`seed_sd_2x`: 0.695–0.735). On accuracy alone the same filter
   gives **0.59–0.67** (0.592/0.598/0.621/0.647/0.674). MISMATCH of
   population. The box also uses a third population: multi-axis and
   unfiltered, while Figure 3 directly above is single-axis, filtered and
   includes bBPB. **Fix:** "0.59 to 0.67" on accuracy, or say "accuracy and
   bBPB" for 0.69–0.73. Name the pair set.
8. **Training grid table (app_01:87, :93) is STALE.** It says 3B at
   K ∈ {8, 15, 30} for both scheme A and scheme B, with 57 and 23 runs. The
   repo's generated table (`documents/paper/sections/app_01_model_ladder.tex`,
   `make_appendix_tables.py`) and `configs/models.json` (3B cells: A-L8,
   L15, L30, L50; B-L8, L15) give **A: K ∈ {8, 15, 30, 50}, 58 runs; B:
   K ∈ {8, 15}, 22 runs**. App. E (app_05_da_analyses:78) already uses the
   correct set, so the paper contradicts itself. **Fix:** re-copy the
   generated block.
9. **Benchmark counts disagree in four places.** Abstract: "\nbenchmarks =
   20 multilingual benchmark families". §3.2 and App. C: "\nbenchanalysed =
   52 benchmarks", "\ntasksanalysed = 938 tasks". Table 6 header: "Analysed
   (40 benchmarks, 695 tasks)", with only **34** rows printed in the Overleaf
   version (rows merged by hand). Screened: "26 benchmarks, 485 tasks" in the
   header, against \nbenchscreened = 43 and \ntasksscreened = 550. App. B 3B
   paragraph: "of **933**". The gate's trained population is 933
   (`A/rq00_gate_and_curves/pretraining/predictivity/gate_pass_counts_all_considered.csv`:
   `auto/all` 52 benchmarks, 933 tasks), and the 34 %/38 % of §4.1 are on
   933. **Fix:** one definition, stated once ("52 benchmarks counting RF/LLM-RF
   variants separately, 40 without, 20 of them multilingual"). Use 933 (tasks
   with a chance level) or explain the 5 extra tasks of 938.
10. **The App. C task-selection sentence misdescribes the screen
    (app_03:58).** "We selected 938 tasks from the total 1,488 we considered,
    removing tasks for which the final checkpoints of the 600M, 1B, 1.7B and
    3B models did not achieve above-chance performance." Only the 229
    discarded tasks failed the gate. The rule is "at chance at 1B and at
    1.7B", which app_03:54 states correctly. Of the screened, not analysed
    tasks, 286/292 **pass** at 1.7B (`gate_pass_counts_all_considered.csv`).
    They were left out for cost or redundancy (rq00 README §1a). The
    committed probe gate also has no 3B verdict (`passes_3B` is False on all
    521 rows of `gate_screened_verdicts_all_considered.csv`). 3B reads exist
    only in the local copy `src/signal-and-noise/data/considered/`, which the
    paper figure uses. **Fix:** "We discarded the 229 tasks at chance at both
    1B and 1.7B, and did not adopt a further 321 screened tasks for cost or
    redundancy".
11. **"Stronger associations emerge for individual benchmarks, reaching
    ρ=0.93 for HellaSwag, compared with a maximum of ρ=0.75 across all
    benchmarks combined" (04:331).** Misleading comparison. 0.93 is the
    *best of 132 SNR ratios* within the HellaSwag stratum (26 clusters).
    0.75 is *one fixed* SNR (AllenAI rel_std / ckpt noise) over 472
    clusters, pooled over proxies (`A/rq04_surrogates/README.md`, catalogue
    and strata bullets). The same fixed SNR gives 0.82 on HellaSwag. Neither
    number comes from Figure 4, whose maximum is 0.62. **Fix:** compare like
    with like ("the same SNR reads 0.82 within HellaSwag against 0.53–0.61
    within one proxy size"), or flag 0.93 as best-of-many.
12. **Near-tie sensitivity (App. E.2: Table `tab:da_tolerance`,
    Figures `da_sensitivity.png` and `da_categories.png`).** UNSUPPORTED by
    the repo. No generator, CSV or README in `src/` or `documents/paper/`
    produces these numbers (357/982/1,949/3,268/4,301 ...). The PNGs exist
    only as files. They cannot be checked against the 2026-10-10 populations,
    and the pair counts cannot be reconciled with the 406 reference pairs.
    The near-tie *DA* effect is covered by the generated
    `da_size_definition_sensitivity_*.csv` (`ckpt_noise_1x/2x`, `ties_half`).
    **Fix:** either commit the script and CSV, or replace the subsection with
    the generated table (`tables/app_rq02_definition_sensitivity.tex`, already
    in the Overleaf folder but not included).
13. **Compute is STALE (app_01:214).** "Pretraining ... consumed 40,500 node
    hours". The repo constants block (`documents/paper/sections/main.tex`,
    from `plan/compute-costs.json` dated 2026-10-11) gives **42,900**
    (other 11,000, GPU-hours 215,600 against 9,300 / 199,200 in the Overleaf
    macros).

---

## (B) Claim table

Locations: `NN:line` = `sections/NN_*.tex`, `app_XX:line` =
`sections/app_XX_*.tex`, `main:line` = main.tex.

### Abstract and introduction

| location | claim | status | evidence | fix |
|---|---|---|---|---|
| main:189, 01:29, 03:12, app_01:108, 05:11 | ladder of \nmodels = 213 models | MISMATCH (internal) | `verified_results_provenance.json` `healthy_175M_to_1_7B` 213 = models at 90M–1.7B only. Grid table total 218 (with six 3B runs). Grid rows at 90M–1.7B sum to 212, and the provenance has 33 at 90M against 32 cells in models.json | say "213 models from 90M to 1.7B plus six at 3B" or use 218. Reconcile the 90M count (33 vs 32) |
| main:189 | "multilingual models" | MISMATCH (wording) | K=1 runs (A, DCLMP, FWEB, replicates) are English-only | "models with 1 to 50 languages" |
| main:190 | "20 multilingual benchmark families" | VERIFIED (definition) / see (A)9 | \nbenchmarks = rows of the multilingual block | state relation to the 52 of §3.2 |
| main:191 | quantify "how well the performance of smaller models predicts larger model performance" | UNSUPPORTED | No performance-prediction (scaling-law error) result in the paper. R² fits include 1.7B, and the rq01 README says R² "is not a statement about predicting 1.7B from below". The forecast numbers (7.7/4.4/2.3 % BPB error) are in the rq01 README, not the paper | rephrase as "rank" or add the scaling-law-error result |
| main:191 | how early the correspondence emerges, and whether small-scale evals recover design effects | VERIFIED | Fig. 3b/c, §5.1–5.3 | – |
| main:192 | informativeness set by proxy size, tokens per language, item format, scoring rule | UNSUPPORTED (tokens, format) / contradicts 01:30 | see (A)1–3 | – |
| main:193 | concrete per-benchmark guidance | VERIFIED (measurability only) | Table 7 (token thresholds), website | say it covers the above-chance thresholds |
| 01:11 | prior decision-reliability work only in English | not checked (citation) | see `plan/citation-review.md` | – |
| 01:16 | many multilingual tasks near chance at small scale | VERIFIED (own data) | 313/933 at chance at every size (rq00 README) | – |
| 01:29 | evaluate 213 models on up to 1,488 tasks | VERIFIED with caveat | 550 of the 1,488 were scored only on 600M–1.7B finals | – |
| 01:30 | tokens, answer format and scoring impact DR more than proxy size | UNSUPPORTED | (A)1–3 | – |
| 01:31 | cloze reformatting makes Belebele, Global-MMLU, INCLUDE measurable where letter format stays at chance | VERIFIED | 1.7B: RF 86/105, 35/37, 31/43 against 5, 1, 3 (`A/rq00_task_reformulation/README.md`, `A/rq00_gate_and_curves/README.md`) | – |
| 01:31 | once above chance, accuracy scales predictably across size and checkpoints | VERIFIED (74 %) | 311/419 predictable across both (`scaling_regimes.csv`) | "most tasks (74 %)" |
| 01:31 | log-likelihood scoring on gold answers improves ranking agreement | VERIFIED (appendix only) | bBPB → 1.7B acc beats acc → 1.7B acc by +0.03 to +0.10, better on 57–71 % of tasks, Wilcoxon p ≤ 4.9e-8 (`bench_bpb_da_size_summary_multi_axes.csv`) | mention the size of the gain (largest at 90M, +0.03 at 1B) |
| 01:32 | proxy statistics from monolingual settings do not transfer | UNSUPPORTED in paper | (A)4 | – |
| 01:37 | open ladder and measurements | promise (not checkable) | – | – |
| 01:38 | guide: per-benchmark token counts to clear chance **and at which we observe decision reliability** | first half VERIFIED, second half UNSUPPORTED | Table 7 gives above-chance thresholds only | drop the DR half or add a DR table |
| 01:42 | recommendations on filtering, reformatting, scoring | partially UNSUPPORTED | The only explicit recommendation is §4.3 ("more sizes rather than longer runs"). Filtering/reformatting/scoring advice appears only implicitly (the draft recommendations in 05:6 are commented out) | restore an explicit recommendations paragraph |

### §3 Experimental setup

| location | claim | status | evidence | fix |
|---|---|---|---|---|
| 03:14 | seven sizes 90M–3B. 90M–1B predict 1.7B, 3B for generalization | VERIFIED | `configs/models.json`, RULES rule 9/10 | – |
| 03:17 | trained on DCLM-Edu and FineWeb-2 | MISMATCH (minor) | DCLMP (DCLM w/o edu filter) and FWEB (FineWeb) builds also train | add "plus two alternative English corpora at K=1" |
| 03:47–70 | six design axes | VERIFIED | RULES Definitions | – |
| 03:64 | Muon at K∈{8,15,30} | VERIFIED (deep only, stated in app_01:159) | CLAUDE.md | add "deep" |
| 03:78 | evaluated at every 10 % + 85 %, 95 % | VERIFIED | RULES rule 3/4 | – |
| 03:83 | 52 downstream benchmarks, 50 languages | VERIFIED (macro) / inconsistent with Table 6 header | (A)9 | – |

### §4 Measurability

| location | claim | status | evidence | fix |
|---|---|---|---|---|
| 04:39 | we analyse 938 tasks | MISMATCH (internal) | gate population 933 (`gate_pass_counts_all_considered.csv`) | (A)9 |
| 04:39 | only 38 % pass at every size, 34 % never up to 1.7B | VERIFIED | 356/933 = 38.2 %, 313/933 = 33.5 % (rq00 README) | – |
| 04:40 | failing tasks predominantly letter-format. Belebele, Global-MMLU, INCLUDE: at most three language subsets pass | VERIFIED on trained languages (3/59, 1/29, 3/36 at 1.7B) / conflicts with 04:90 | 04:90 quotes 5/105 for Belebele (all languages) | say "on trained languages" here, or use one population in both places |
| 04:40 | HellaSwag (Okapi) and MultiBLiMP above chance in all trained languages from 90M | VERIFIED | `data/considered/app_rq00_chance_share_considered_counts_paper.csv`: hellaswag 26/26, multiblimp 37/37 at 90M | – |
| 04:48 | non-English: 42 % of tasks above chance at ≤100M tokens, 64 % at 10B–100B | VERIFIED (cells, not tasks) | `A/rq00_chance_vs_train_tokens/README.md`: 42 % at 10^7–10^8, 64 % at 10^10 (4,945 cells) | "of evaluations", not "of tasks" |
| 04:49 | at chance on a third of tasks after 10B tokens = Chinchilla budget of a 500M model | VERIFIED | 36 % at chance. 20 × 500M = 10B | – |
| 04:50–52 | HellaSwag, MultiBLiMP, TokSuite above chance in nearly all configs <100M tokens | VERIFIED | Table 7 rows "≤" at every size | – |
| 04:53 | at 100M–1B tokens: 43 % (90M) vs 62 % (1.7B) | VERIFIED | rq00_chance_vs_train_tokens README | – |
| 04:60 (Fig. 1 caption) | INCLUDE letter vs answer-text, share vs tokens per size | VERIFIED | source `pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical` | – |
| 04:87 | reformulation with gemini-3.8-flash | VERIFIED | rq00_task_reformulation README l.271 | – |
| 04:90 | 1.7B: 5→86/105, 1→35/37, 3→31/43 | VERIFIED | RULES "reformulated twins", rq00 README | – |
| 04:92 | LLM-RF Belebele passes in all trained languages at every size | VERIFIED | rfgm_belebele 49/49 from 90M | – |
| 04:124 | 431 tasks, median R² 0.87 | MISMATCH | (A)5: 419 tasks, 0.88 | – |
| 04:124 | R² 1.00 for BPB across 50 languages | VERIFIED | 106 BPB fits over 50 languages at 1.00. Figure 1 draws 30 BPB tasks (languages at ≥2 settings) | – |
| 04:125 | reformulated letter benchmarks median R² 0.84–0.92 | VERIFIED | `scaling_regimes_families.csv`: rf_belebele 0.85, rf_global_mmlu 0.84, rf_include 0.84, rfgm_belebele 0.90, rfgm_include 0.92 | – |
| 04:126 | TruthfulQA R² 0.97, ρ=−0.75, "decrease in predictability" | MISMATCH (wording) | one task, `truthfulqa-multi_mc1_es` (R² 0.972, ρ −0.75), fit on 90M–350M only (rq01 README) | "TruthfulQA-Multi (Spanish) ... a consistent decrease in *accuracy*". Note the 90M–350M fit |
| 04:131 | 74 % (311 tasks) R²≥0.5 on both | VERIFIED | 311/419 | – |
| 04:133 | no task predictable across training without size | VERIFIED | 0 "training only" in `scaling_regimes.csv` (non-bBPB). 9 bBPB tasks are training-only (rq01 README) | add "accuracy or BPB" (already there) |
| 04:134 | same pattern for monolingual English | MISMATCH | English L1 has 1–2 "predictable during training only" tasks per corpus (`english_only_scaling_regimes_l1_accuracy_tagged_paper.csv`: DCLM-Edu 1, DCLM 2, FineWeb 2) | "a similar pattern" or name the exceptions |
| 04:137 | recommend more model sizes over longer runs | weakly supported | Rests on R² fits that include 1.7B (rq01 README: "R² inflated by construction"). Ranking results (§5) show size does not improve DA | soften or justify |
| 04:143 | Table 7 gives 50 %/90 % token thresholds | VERIFIED | identical to `A/rq00_chance_vs_train_tokens/.../pass_threshold_tokens_by_benchmark_1904_ckpts.tex` apart from the dropped † marks | re-copy (daggers) |
| 04:153 (box) | 34 % of the 938 tasks never pass. Above chance depends on tokens and size. Accuracy scales predictably | VERIFIED (34 % is of 933) | as above | use 933 |
| Fig. 2 caption 04:70–73 | median R² vs ρ per benchmark | VERIFIED | `scaling_regimes_outliers_paper` | note that the panel also draws bBPB twins (1,029 points) while the text discusses accuracy + BPB |

### §5 Agreement

| location | claim | status | evidence | fix |
|---|---|---|---|---|
| 04:192 | DA only for ≥3 pairs, "since comparisons between only two pairs yield binary agreement" | MISMATCH (wording) | RULES rule 5: pair counts are triangular, never 2. Two variants give one pair | "at least three pairs (three variants)" |
| 04:194 | focus on tasks with median DA ≥0.66 | VERIFIED (selection on outcome) | Selection adds 0.01–0.02 (cross-fit, `plan/paper-review-followups.md` W2). Unfiltered single-axis 0.55–0.59 | report unfiltered/cross-fitted beside it |
| 04:213 | DA-ckpt 62–68 % early → 83–93 % late | VERIFIED | Fig. 3 CSV DA-ckpt: 0.5C 0.619–0.682, 4.5C 0.830–0.925 | – |
| 04:215 | 350M generally highest, no consistent size relation | VERIFIED (late only) | 350M highest from 2.5C (0.755→0.925). Lowest at 0.5C (0.627) | "highest in the second half of training" |
| 04:229 | DA-size 59–61 % all pairs, no consistent size improvement | MISMATCH | (A)6: figure shows 0.60–0.64 | – |
| 04:229 | temperature 72–75 % | VERIFIED (4 pairs) | macro 0.727–0.753 | add "on four pairs". Activation and optimizer rest on 3 pairs, without a band (followups W4) |
| 04:232 | DR depends more on intervention type than proxy size | VERIFIED with caveat | per-axis spread 0.43–0.75 vs size spread ≤0.04 on all pairs. Temperature has 4 pairs | caveat pair counts |
| 04:239 | smaller proxies' DA-goal ~58–64 %, little improvement. 1.7B rises | VERIFIED | DA-goal 90M–1B 0.575–0.643. 1.7B 0.648 → 0.830 | – |
| 04:241 | at 1.7B accuracy agrees with 3B at 0.51–0.52 | VERIFIED (acc + bBPB) | `A/rq10_size_generalisation/README.md` table: 1.7B → 3B 0.51 mono, 0.52 multi (twins pooled). Accuracy alone 0.52/0.53 | say "accuracy and bBPB" |
| 04:253 | near-ties raise DA without changing trends | UNSUPPORTED (no repo source) | (A)12 | – |
| 04:253 | BPB preserves rankings better than accuracy, especially small and early | VERIFIED | BPB DA-size 0.94/0.94/0.89/0.74/0.92 against accuracy 0.51–0.56. BPB DA-goal 0.92 at 90M first tenth | – |
| 04:270 (box) | accuracy: 0.51–0.56 at every proxy | VERIFIED | `da_size_definition_sensitivity_multi_axes.csv` 0.506–0.560 (multi-axis, unfiltered) | name the pair set and that this is unfiltered, unlike Fig. 3 |
| 04:271 (box) | BPB 0.74–0.94 | VERIFIED | `scale_convergence_da_size_multi_axes.csv` population bpb, macro | – |
| 04:272 (box) | >2 seed SD: 0.69–0.73 | MISMATCH (population) | (A)7: accuracy alone 0.59–0.67 | – |
| Fig. 3 caption 04:100–105 | tasks with median DA ≥0.66, (a) pooled over decisions, (b) averaged over tasks | MISMATCH (minor) | (a) is the macro mean over tasks | "averaged over tasks". Add "single-axis pairs" (the phrase is commented out) |

### §6 Anticipation

| location | claim | status | evidence | fix |
|---|---|---|---|---|
| 04:309 | Fig. 4 = correlations on tasks above chance | VERIFIED (+ bBPB) | `A/rq04_surrogates/pretraining/predictivity/rq3_surrogates_heatmap_paper.csv`, n = 1,274–1,461 = accuracy + bBPB twins | say the bBPB twins are included |
| 04:327 | early ckpt ρ 0.54–0.62. rel-std SNR 0.53–0.60. R² 0.45–0.49 | VERIFIED | 0.536–0.616, 0.527–0.598, 0.445–0.487 | – |
| 04:328 | signal alone and margin show little association | VERIFIED | −0.16 to 0.00 and −0.03 to 0.09 | – |
| 04:329 | SNR depends on the noise definition | VERIFIED | rq04 README (22 definitions, rel-spread block 0.51–0.52 vs depth −0.02 to −0.05) | – |
| 04:331 | HellaSwag 0.93 vs max 0.75 all benchmarks | MISMATCH (misleading) | (A)11 | – |
| 04:338–340 | early ckpt 0.62 at 175M. rel-std noise 0.56 at 350M. R² 0.45–0.49 from 350M | VERIFIED | heatmap CSV (noise alone 0.561 at 350M) | – |
| 04:343 | 600M BPB: ckpt noise 0.4 % vs 1.3 % | VERIFIED | `A/rq03_noise_and_snr/.../snr_variants_per_task.csv` `noise_rel_std_600M` median: BPB 0.0036, gated accuracy 0.0126 | – |
| 04:343 | 600M BPB median R² 1.00 | VERIFIED | rq01 | – |
| 04:343 | median rel-std SNR 8.7 vs 2.0 | VERIFIED | `snr.csv` 600M: BPB 10^median = 8.71, gated accuracy 2.04 | – |
| 04:343 | DA at 600M 0.70–0.74 throughout training | MISMATCH (minor) | `A/rq02_da_vs_train_tokens/pretraining/predictivity_seeds/da_goal_multi_axes_across_langs_bpb_paper.csv` 600M: **0.694–0.741** | "0.69–0.74". Also note that the 600M miss is the non-size-matched depth pairs at one seed (followups W6) |
| 04:350–351 (box) | no metric predicts DA consistently. Early ckpt 0.54–0.62, rel-std 0.53–0.60 | VERIFIED | as above | grammar ("No metric ... predicts") |
| Fig. 4 caption 04:316–321 | ρ and n per cell, R² from 350M | VERIFIED | heatmap CSV | – |

### §7 Conclusion, Limitations, Ethics

| location | claim | status | evidence | fix |
|---|---|---|---|---|
| 05:11 | measurable scores, smooth curves and stable proxy rankings do not guarantee cross-scale agreement | VERIFIED | 600M BPB case. DA-ckpt 0.83–0.93 against DA-goal 0.58–0.64 | – |
| 05:11 | increasing proxy size does not consistently resolve it | VERIFIED | DA-size flat. 3B reference 0.49–0.54 | – |
| 05:11 | reformulating tasks and likelihood scores can improve DR | half UNSUPPORTED | (A)2. The likelihood half is VERIFIED | – |
| 05:13 | we release the per-benchmark guide | promise | – | – |
| 06:11 | 3B covers a subset | VERIFIED | 6 cells | – |
| 06:13 | most configurations single seed | VERIFIED | 20 replicate runs of 213 | – |
| 06:13 | "we report fixed-population comparisons" | UNSUPPORTED | No `_fixed_*` figure or table is `\input` (the files `app_rq02_fixed_*.png` etc. sit unused in `figures/`). Only the 3B comparison holds tasks fixed | include one or drop the clause |
| 06:15 | manual review of generations in six languages | not checkable from repo | – | – |
| 06:20 | release models and measurements | promise | – | – |

### Appendix A (model ladder)

| location | claim | status | evidence | fix |
|---|---|---|---|---|
| app_01:27 (Tab. notation) | seeds 1904, 313, 64 | MISMATCH | 28 and 1797 also train at 1B (app_01:183, rq03 README) | list all five |
| app_01:48–68 (Tab. ladder) | hyperparameters | VERIFIED | identical to the generated block in `documents/paper/sections/app_01_model_ladder.tex` (the repo has one more column, saves) | – |
| app_01:87–100 (Tab. grid) | 3B sets and run counts | STALE | (A)8 | – |
| app_01:117 | tied embedding 101M at 90M, 302M at 1.7B | VERIFIED | 768×131,072 = 100.7M. 2,304×131,072 = 302M | – |
| app_01:133 | batch 504, except 84 and 168 | VERIFIED | Tab. ladder | – |
| app_01:155–156 | deep ratio 64 (175M–1B), 51 at 90M, 77/78. Shallow 110–146. Counts differ 0.4–5.2 % | VERIFIED | Tab. ladder (−5.2 % at 90M, −0.4 % at 1.7B and +0.4 % at 1B) | – |
| app_01:165 | SwiGLU FFN ≈2.67 d, within 0.6 % | VERIFIED | max +0.57 % (175M) | – |
| app_01:183 | replicate seeds and cells | VERIFIED | rq03 README "Seed cells" | – |
| app_01:210 | 20/40/60 saves, twelve evaluated | VERIFIED | repo table "saves" column | – |
| app_01:213 | 3 to 21 nodes per run | not checked | – | – |
| app_01:214 | 40,500 node hours | STALE | (A)13 → 42,900 | – |

### Appendix B (data) and C (evaluation)

| location | claim | status | evidence | fix |
|---|---|---|---|---|
| app_02:6 | K=8 A list = ru, zh, de, ja, es, fr, it, nested lists | VERIFIED | `app_02_table_languages.tex` K_A column | – |
| app_02:8 | K=8: B spans 7 scripts vs 4 for A. K=30: 14 scripts/12 families vs 10/10 | VERIFIED | computed from the languages table (A: 4 scripts at K=8, 10/10 at K=30. B: 7, 14/12) | – |
| app_02:10, :26 | FWEB crawl window, validation-set construction (5M tokens, 30 %) | not checked | – | – |
| app_03:52 | 95 benchmarks, 1,488 tasks considered | MISMATCH (internal, repo) | Macro value. The rq00 README counts 93 benchmarks / 1,454 tasks with a chance level, or 97 / 1,539 with all tasks | pick one and footnote the definition |
| app_03:54 | discarded the 229 tasks at chance at both 1B and 1.7B | VERIFIED | `gate_pass_counts_all_considered.csv` discarded: 0/229 at 1B and 1.7B | – |
| app_03:54 | screened by evaluating 600M, 1B, 1.7B **and 3B** finals | UNSUPPORTED (committed gate) | `passes_3B` all False. 3B is only in the uncommitted copy | (A)10 |
| app_03:58 | selected 938 of 1,488, removing tasks not above chance | MISMATCH | (A)10 | – |
| app_03:64 | tasks per K: 113/134/326/485/720/938 | VERIFIED (macro) | `make_appendix_tables.py` | – |
| app_03:64 | which runs are evaluated on every language | not checked | – | – |
| app_03:66 | English covered by 31 benchmarks (17 multilingual, 14 English-only) | MISMATCH | languages table "Fam." column: English **25** | align the text with the table, or explain the different unit (twins?) |
| app_03:66 | other 49 languages: median 8 | MISMATCH | table: median **7** | – |
| app_03:66 | Spanish 18, Mandarin 17, Russian and Arabic 14, Hindi/Italian/Vietnamese 13 | MISMATCH (partly) | Spanish 18 ✓, Mandarin **16**, Russian/Arabic 14 ✓, Hindi/Vietnamese 13 ✓, Italian **12**. German and French have 13 | – |
| app_03:66 | Bosnian only Global PIQA. Latvian Belebele + MultiBLiMP. Slovenian 3 | MISMATCH (Latvian) | table: Bosnian 1 ✓, Slovenian 3 ✓, Latvian **1** | – |
| app_03:66 | most low-resource benchmarks did not pass at 1B or 1.7B | MISMATCH | 286/292 screened, not analysed tasks pass at 1.7B. Only the 229 discarded fail | rephrase |
| Tab. 6 (app_03_table_benchmarks) | header 40 benchmarks / 695 tasks, 26 / 485 | MISMATCH | 34 rows printed. Macros 52/938, 43/550 | (A)9 |
| Tab. 6 | ACP-Bench 1 lang / 2 trained, BBH 1/2, CulturalBench 8/16 | MISMATCH | "Trained" cannot exceed "Langs". The hand-merge summed two rows (repo: 1/1 each, 8/8) | restore the generated rows or take the max |
| Tab. 6 caption | "Mid-Resource Language Examples ... ranks 30 to 50" | MISMATCH (wording) | repo generator says "Low-Resource" | align header and caption |
| app_03:84 | acc_norm on cloze originals and RF versions, acc otherwise | VERIFIED | rq02 README snapshot "acc_norm on the cloze-format originals" | – |

### Appendix D (measurability)

| location | claim | status | evidence | fix |
|---|---|---|---|---|
| app_04:11 | chance 0.2253 for TruthfulQA MC1 | VERIFIED | RULES rule 1 | – |
| app_04:19–22 | gate definition (seed 1904, ≥50 %, trained runs) | VERIFIED | RULES rule 1, rq00 README | – |
| app_04:27 | margins 9.00 / 2.44 / 0.74 / 0.60 pp | VERIFIED | rq00 README: +0.09, +0.024, +0.007, +0.006 | – |
| Fig. app_rq00_chance_shared caption | letter-format benchmarks pass only at 3B. Cloze from 90M | VERIFIED (local data) | `data/considered/...counts_paper.csv`: belebele 3 at 1.7B, 45 at 3B. global_mmlu 1/28. include 2/31. hellaswag/multiblimp all at 90M | note that the 3B column reads six 3B runs only |
| app_04:52 | majority chance lowers the share by 1.5–2.8 pp | VERIFIED | `majority_pass_counts.csv` (`majority_fixed` vs `uniform`, trained): −2.8, −1.6, −1.8, −1.5, −1.8, −1.9 | – |
| app_04:52 | removed tasks: Belebele/GMMLU/INCLUDE near-passes, CB-easy, PAWS-X, BLEnD | MISMATCH (incomplete) | `majority_flips.csv` (fixed options): noreval_extra 55 of 107 flips, paws 15, cb_easy 10, include 7, belebele 5, **boolq 5**, gmmlu 4, bbh_mcq 3, noreval 2, blend 1 | name NorEval (the majority) and BoolQ |
| app_04:71 | McNemar p ≤ 5.8e-8. Significant at every size for Belebele/GMMLU, from 175M for INCLUDE | VERIFIED | rq00_task_reformulation README l.85 | – |
| app_04:73 | LLM-RF INCLUDE 32/43 | VERIFIED | RULES | – |
| app_04:73 | gold longest 22.2 % (original) vs 21.9 % (rewritten) | VERIFIED (INCLUDE only) | reformulation README l.745–747: rf_include 22.2 %, rfgm_include 21.9 % | say "for INCLUDE" |
| app_04:95 | 43 % (10^5, 82), 42 %, 42 %, 52 %, 62 %, 64 % (4,945) | VERIFIED | rq00_chance_vs_train_tokens README l.30 | – |
| app_04:97 | 43 % vs 62 %. Pooled peak 52 %, 41 % at 10B–100B, 80 % English of 25,217 | VERIFIED | README l.29 | – |
| app_04:97 | "ten of the 30 benchmark formats never reach 50 %" | MISMATCH | Table 7 has 28 rows, of which 7 never reach 50 % at any size (Belebele, Global-MMLU, INCLUDE originals, BLEnD, CB-easy RF, CB-hard, TruthfulQA MC2). README (pooled bins): 9 of 31 families | "seven of the 28" (as tabled) |
| app_04:97 | 1.7B needs ~10× fewer tokens than 90M. ARC (Okapi) 3.3B at 600M vs 320M at 1.7B | MISMATCH (wording) | numbers ✓ (Table 7), but the example compares 600M, not 90M (90M never reaches 50 %) | "than a 600M one" |
| Fig. app_rq00_chance_vs_train_tokens caption | rises with every decade from 10M on | VERIFIED | README key finding | – |
| app_04:141 | cloze versions scale predictably | VERIFIED | (A)5 family R² | – |
| app_04:149 | 17 of 39 English accuracy tasks predictable in all three corpora | VERIFIED | `english_only_scaling_regimes_l1_accuracy_tagged_paper.csv`: 17 | – |
| app_04:149 | Global PIQA only popular benchmark above chance not scaling | VERIFIED | rq13 README l.348/368 | – |
| Fig. app_rq01_english_scaling caption | lists 12 tasks predictable in all three corpora | VERIFIED (incomplete) | The 17 also include BBH date RF, Belebele LLM-RF, PIQA, TokSuite STEM, WinoGrande | say "among them" |

### Appendix E (agreement)

| location | claim | status | evidence | fix |
|---|---|---|---|---|
| app_05_ranking:45 | DA 0.5 for random tie-free rankings, ≥3 pairs | VERIFIED | RULES rule 5 | – |
| app_05_ranking:58 | seed replicates never ranked as variants | VERIFIED | RULES rule 15 | – |
| app_05_ranking:69 + Fig. caption | 6,803 values. r = 0.99 (τ_b), 0.97 (ρ) | VERIFIED | `agreement_da_size_correlation_multi_axes.csv`: 6,803, 0.986, 0.970 (the rq02 README's "6,320 / 0.986" is stale) | – |
| app_05_da_sensitivity Tab. + Figs. | tolerance counts. Category trends | UNSUPPORTED | (A)12 | – |
| app_05_da_analyses:8 | acc + bBPB DA-size 0.58, 0.57, 0.58, 0.58, 0.58 | VERIFIED (pair-weighted) | `scale_convergence_da_size_multi_axes.csv` `reliability` 0.584/0.571/0.584/0.580/0.582 | – |
| app_05_da_analyses:8 | BPB 0.94, 0.94, 0.89, 0.74, 0.92. bpb_macro 0.95–0.98 | VERIFIED (macro) | same CSV `reliability_macro`. rq02 highlight block | aggregation differs from the preceding numbers (pair-weighted). Use one |
| app_05_da_analyses:18 | at chance at the proxy: bBPB→bBPB 0.60–0.63 vs accuracy 0.46–0.50 | VERIFIED | `bench_bpb_da_size_by_acc_gate_multi_axes_paper.csv` | – |
| app_05_da_analyses:18 | reliable tasks: accuracy 0.56 → 0.69, bBPB 0.60–0.65 | VERIFIED | `rq2_da_all_above_66_either_transformation_by_scoring_mono_axis.csv` (accuracy rests on 81–110 tasks) | give the task counts |
| app_05_da_analyses:20 | +0.10/+0.06/+0.06/+0.05/+0.03, 360–542 tasks, 57–71 %, p<1e-7 | VERIFIED | `bench_bpb_da_size_summary_multi_axes.csv` | – |
| Fig. app_rq02_bbpb_acc_gate caption | 406 pairs | VERIFIED | rq02 README population | – |
| app_05_da_analyses:47 + Fig. caption | DA-goal of BPB vs tokens | VERIFIED (figure) / no conclusion drawn | The appendix states no finding. The README finding is "tokens do not set the level" | add the sentence. It is the evidence against (A)1 |
| app_05_da_analyses:65 | HellaSwag: de/es/fr/it predict each other at 0.78, self 0.79. English → others 0.45 | VERIFIED | `cross_task_da_size_by_language_hellaswag_all_languages_mono_axis_paper.csv`: 0.776, 0.792, 0.446 (rq06 README prose 0.81/0.82/0.43 is stale) | – |
| Fig. rq2_3b_compare caption | nine single-axis pairs, 553 tasks (394 bBPB). Only K has ≥3 pairs | VERIFIED | `data/rq2_3b/rq2_3b{,_twin_ref1.7B}_reliability.csv`: 553 ∩ kept, 394 bbpb | – |
| app_05_da_analyses:78 | six 3B configs A{8,15,30,50}, B{8,15}. bBPB on 5 of 6 | VERIFIED | rq10 README Setup | conflicts with Tab. grid (A)8 |
| app_05_da_analyses:80 | 612 → 778 of 933 (+27 %) | VERIFIED | rq10 auto block l.243 (the README key finding's 607/772/927 is stale) | – |
| app_05_da_analyses:80 | DA-size to 3B 0.49–0.54 (acc), 0.78–0.91 (BPB). 1.7B itself 0.51 / 0.95 | VERIFIED (mono-axis, acc+bBPB) | rq10 README table | say "accuracy and bBPB" |
| app_05_da_analyses:80 | proxy checkpoints 0.48–0.58 against either reference. Reference's run 0.75–0.77 at 4.5C | VERIFIED (≈) | `data/rq2_3b/rq2_3b_compare.csv` DA-goal: proxies 0.478–0.582. 3B run 0.749, 1.7B run 0.775 | "0.75–0.78" |

### Appendix F (anticipation)

| location | claim | status | evidence | fix |
|---|---|---|---|---|
| app_06_snr:101–103 | checkpoint noise = σ of the last k checkpoints | STALE | RULES rule 4: since 2026-10-08 the noise is the residual SD around a least-squares line over 80–100 % (detrended). All Fig. 4 numbers use it | define the detrended noise |
| app_06_snr:105 | "§6 reports every intervention's effect against the seed noise" | UNSUPPORTED | §6 has no such result (it is in `A/rq03_noise_and_snr`, not in the paper) | drop or add |
| Fig. app_rq04_surrogates caption, app_06_surrogates:8–10 | 22 definitions. Relative-spread block best | VERIFIED | rq04 README (rel_std, rel_mpd, rel_dispersion 0.53–0.54 overall) | – |
| app_06_surrogates:12 | bBPB: early 0.57–0.68, rel-std 0.46–0.58. Accuracy: 0.28–0.39, 0.33–0.40 | VERIFIED except rel-std on accuracy **0.33–0.39** | `rq3_surrogates_by_scoring_paper.csv` | – |
| app_06_surrogates:12 | BPB: five of six statistics negative at 600M | VERIFIED | same CSV (only inverted noise +0.13) | – |
| app_06_surrogates:12 | single-axis: weaker, rel-std leads at every proxy | VERIFIED | `mono_axis/rq3_surrogates_heatmap_paper.csv` (0.37–0.46 vs early 0.24–0.36) | – |
| Fig. app_rq04_surrogates_by_scoring caption | on accuracy alone rel-std SNR "leads or ties" | MISMATCH | at 1B: inverted noise 0.44, discrepancy 0.42, early 0.39, rel-std 0.33 | "leads from 90M to 600M" |

**Counts (rows of this table):** VERIFIED 102 (some with a caveat) ·
MISMATCH 28 · UNSUPPORTED 12 · STALE 3 · not checked, promise or weakly
supported 9. A row with two halves is counted once, under the worse status.
The 13 blocking items of (A) overlap these rows. Counted as distinct
problems, there are about 30 mismatches, 14 unsupported claims and 4 stale
numbers.

---

## (C) Internal inconsistencies

1. **Model count.** 213 in the abstract, intro, §3, App. A and conclusion,
   with "seven sizes (…, 3B)" in the same paragraph. The training grid totals
   218. 213 is the 90M–1.7B count, but the grid's 90M–1.7B rows sum to 212
   (the provenance counts 33 runs at 90M against 32 cells in models.json).
2. **Task and benchmark counts.** 938 (\ntasksanalysed, §4.1, box) vs 933
   (App. E 3B, the gate population behind the 34 %/38 %). 52 benchmarks
   (§3.2) vs 20 families (abstract) vs Table 6 "40 benchmarks, 695 tasks"
   with 34 printed rows. Screened "26 / 485" (Table 6) vs 43 / 550 (macros).
   1,488 considered (macro) vs 1,454 (gate) vs 1,539 (README "paper counts").
3. **DA-size level of a proxy.** 59–61 % (§5.2 text, pair-weighted,
   single-axis, filtered) vs 0.60–0.64 (Figure 3a as plotted) vs 0.51–0.56
   (§5 box, accuracy only, multi-axis, unfiltered) vs 0.58 (App. E,
   accuracy + bBPB, multi-axis, unfiltered) vs 0.49–0.54 (3B). Each is
   correct for its population, but the paper never names the population next
   to the number, and §5.2 contradicts its own figure.
4. **The >2-seed-SD number** (0.69–0.73) is on accuracy + bBPB, inside a box
   whose first sentence is about accuracy (0.59–0.67 on accuracy).
5. **Pair set.** Figure 3 and §5.1–5.3 are single-axis. The §5 box, App. E
   (DA-size, bBPB gate, agreement correlation) and Figure 4 are multi-axis.
   The box does not say which it uses.
6. **Letter-format passes.** §4.1 "at most three language subsets" (trained
   languages) vs §4.2 and App. D "5 of 105" Belebele languages (all
   languages).
7. **Factors of reliability.** The abstract lists proxy size as a factor, the
   intro says tokens/format/scoring matter *more* than size, and §5.2 says
   size has no consistent effect. The abstract and intro name tokens and
   format as DR factors, while §4 shows them for measurability only.
8. **3B configurations.** The training grid (A, B at K ∈ {8, 15, 30}) vs
   App. E and the rq10 setup (A at {8, 15, 30, 50}, B at {8, 15}).
9. **Screening criterion.** App. C:54 "at chance at both 1B and 1.7B" vs
   App. C:58 "600M, 1B, 1.7B and 3B did not achieve above-chance" vs App.
   C:66 "most low-resource benchmarks didn't pass", against 286/292 screened
   tasks passing at 1.7B.
10. **Benchmark coverage text vs languages table.** English 31 vs 25,
    Mandarin 17 vs 16, median 8 vs 7, Italian 13 vs 12, Latvian 2 vs 1.
11. **"Ten of the 30 formats"** (App. D) vs Table 7 directly below (7 of 28)
    vs the README (9 of 31).
12. **600M BPB DA** 0.70–0.74 (§6) vs 0.69–0.74 (App. D figure CSV).
    0.75–0.77 (App. E) vs 0.749/0.775 (3B compare CSV).
13. **Seeds.** The notation table lists 1904, 313, 64. The text adds 28 and
    1797.
14. **Noise definition.** App. F defines checkpoint noise as a raw SD. Every
    SNR number uses the detrended residual SD (RULES rule 4).
15. **DA names.** Table `tab:ra_defs` and §5 use RA/DA-target. The captions
    of `app_rq02_bbpb`, `rq2_3b_compare` and App. E text use DA-goal. App. E
    uses "DA-checkpoint" beside DA-ckpt.
16. **Aggregation within one sentence.** App. E:8 gives accuracy/bBPB
    pair-weighted (0.58) and BPB macro (0.94 …).
17. **Monolingual pattern.** §4.3 says the "no training-only task" pattern
    holds for English. The English L1 regimes have 1–2 training-only tasks per
    corpus.
18. **Limitations** promise fixed-population comparisons that are not in the
    paper.
19. **Typesetting.** The compiled PDF drops two Unicode characters
    (`build-1310/build.out`: missing U+2013 at `04_analysis:212`
    "benchmark–language", and U+2019 in `06_statements:22` "datasets’"). Use
    `--` and `'`.
20. **Stale copies of generated artifacts in the Overleaf folder.** Training
    grid (57/23 runs), compute (40,500), Table 7 († marks dropped), Table 6
    (hand-merged rows). The repo's README prose is also stale in places the
    paper got right (rq06 0.81/0.82/0.43, rq10 607/772/927, rq02 6,320 cells,
    rq02_da_vs_train_tokens 0.69–0.73). Do not "fix" the paper from those
    README sentences.
