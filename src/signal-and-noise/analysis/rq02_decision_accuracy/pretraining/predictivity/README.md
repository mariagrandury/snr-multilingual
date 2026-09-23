# Is the multilingual evaluation predictable? RQ0–RQ2 on the predictivity ladder

Analysis of the `predictivity` pool (seed 1904, 175M–1.7B, L ∈ {1, 2, 8, 15, 30, 50}, deep/shallow, schemes A and B; the decision tables pair over every scheme at the grid seed and the seed pools add the replicates) from the ladder report of 2026-09-23. Every number below is read from the CSV beside the figure it describes; the figures of RQ0 and RQ1 live in their own folders and are linked from here. The statistic conventions are those of `analysis/RULES.md`: the above-random gate at the proxy and at the reference (rule 1), trained languages only (rule 2), at least three design-variant pairs per cell (rule 5), one reference size, 1.7B (rule 9), and the pair sets of rule 15 (multi-axis: every pair of design variants; mono-axis: pairs that differ on one design axis). Where a filtered population is quoted we use the `above_66_size` / `above_66_ckpt` / `above_66_either` cuts (median DA over the proxy cells ≥ 0.66 on that panel's own axis), never the `above_66_both` intersection.

**In one paragraph.** The gate removes half of the benchmark-language tasks and whole four-option families with them; the reformulated twins bring those families back. On the tasks that survive, score against size fits a log-linear law with median R² 0.94, but the fit is made on the rungs that already cleared chance and the reference sits inside it. Ranking design variants from a smaller model is close to a coin flip on the full gated population (0.55 at 175M, 0.61 at 1B); the 0.64 → 0.78 rise of the paper's filtered figure is a conditional statement on the tasks whose DA-size cleared 0.66. Ranking from an early checkpoint of the same run reaches 0.85 at 90 % of training, but two seeds of one design reach 0.82 on the same axis: most of DA-ckpt's rise is within-run persistence, not design signal. Restricting to the eight high-resource languages, to a common task set, or to one language at a time does not change the picture, the per-L lines are unordered and within noise of each other, and a language's token share does not predict how reliably its benchmarks rank (Spearman ρ ≤ 0.19 over 26–43 languages). Decision accuracy, Kendall's τ and Spearman's ρ are one statistic (r ≥ 0.965 over 820 cells; DA and τ_a are an exact identity up to the tie term), and the tie convention alone moves the reliable-task verdict on 3–7 % of cells.

---

## 1. RQ0 — the above-random gate

**Definition.** A run is above chance on a task when the one-sided 95 % Wilson lower bound of its accuracy over the task's items clears 1/n_options; a (task, size) cell passes when at least half of the size's runs that train the language pass. BPB and the loss have no chance level and are never gated. The gate is computed once on the `predictivity` pool and every later table reads its mask (`above_random_mask.csv`).

![First size above random](../../../rq00_gate_and_curves/pretraining/predictivity/first_size_above_random.png)

*Figure 1. Benchmark family × language: the smallest size from which the gate holds at every larger evaluated size; "never" when the largest evaluated size is at chance. 775 cells, 38 families; the twenty probe families (`bbh_*`, `include_v2_*`, `cultural_bench_*`, …) are evaluated from 600M only, so their floor is 600M by construction. CSV: `first_size_above_random.csv`.*

![Gate highlights](../../../rq00_gate_and_curves/pretraining/predictivity/highlights.png)

*Figure 2. Per family and size: the share of the family's tasks above the gate, the median margin over chance, and the level stack. CSV: `highlights.csv`.*

**Half of the evaluation is at chance somewhere on the ladder.** Of the 975 benchmark-language tasks with a chance level, 203 clear the gate at 175M, 263 at 350M, 408 at 600M, 448 at 1B and 499 at 1.7B; 525 clear it at at least one size and 450 nowhere. Of the 775 (family, language) cells of Figure 1, 180 are above chance from 175M on, 61 from 350M, 89 from 600M, 39 from 1B, 55 only at 1.7B, and 351 never. This is the population every later figure averages over, and it is the first result of the study.

**The gate is an option-count effect, and the reformulation undoes it.** At 1.7B the letter-answer four-option families are absent: `global_mmlu_full` clears the gate in 0 of 37 languages, `global_piqa_parallel_cloze` in 1 of 73, `include_base_44` in 4 of 43 and `belebele` in 11 of 94, against `multiblimp` 57/57, `xwinograd` 6/6, `hellaswag` 26/31 and `xnli` 15/18. Their reformulated twins pass where the originals fail: `rf_global_mmlu_full` 35/37, `rf_belebele` 77/94, `rf_include_base_44` 31/43, `rfgm_include_base_44` 33/43. The reformulation is therefore not a cosmetic change of prompt; it decides whether a family exists for the rest of the analysis.

**Interpretation.** A four-option letter task at a few hundred items needs roughly +0.04 to +0.09 over chance for a single run to clear the Wilson bound, which a model below 1B rarely delivers in a non-English language. We read the gate as measuring the format's floor rather than the languages' difficulty; the same languages pass on the cloze twins.

**Caveats.** (i) The gate's power varies nine-fold across cells: a cell rests on 2 to 18 runs (the twenty languages only the L50 mixture trains have two runs per size, so one run's lower bound decides), and no figure shows the run count. (ii) "Never" in Figure 1 also codes the 26 tasks that are above chance at some smaller size but at chance at 1.7B, which is a different fact. (iii) The rq00 README's "Top benchmarks by Signal" ranks tasks without the gate; its top families are exactly the gate's casualties, which is a selection on noise. (iv) The rq00 panels are drawn from the mask at the time the driver reaches them; on 2026-09-23 the mask was rewritten after the panels and they had to be regenerated by hand, so the driver now draws them right after the gate.

---

## 2. RQ1 — scaling predictability

**Definition.** Per (task, L), a log-linear fit of the final score against non-embedding parameters over the rungs where the task is above chance (deep, scheme A, seed 1904; 3 to 5 rungs including the 1.7B reference), and per task a fit of score against log tokens along each run. A task's regime is the quadrant its median R² over L settings falls in, split at 0.5, with "declines with size" for a negative median Spearman ρ; a task needs at least two L settings with a fit to have a median.

![Scaling regimes](../../../rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_outliers_paper.png)

*Figure 3. One point per task with a regime (193): median R² of the size fit against median R² of the trajectory fit, family labels at the family's median, the named outliers being tasks in another quadrant than their family and at least 0.25 away from its point. CSV: `scaling_regimes_outliers_paper.csv`.*

![Survivorship](../../../rq01_scaling_predictability/pretraining/predictivity_all/scaling_regimes_survivorship.png)

*Figure 4. (a) Per family, the tasks Figure 3 draws against the two ways a task loses its point: at chance at every size where a fit was possible (rule 1) or fitted at fewer than two language settings. (b) Figure 3's panel with kept/total in every label. CSV: `scaling_regimes_survivorship.csv`.*

**Scaling is predictable on the tasks that survive the gate.** Of the 193 tasks with a regime, 173 are predictable across both size and training, 14 across size only, 5 weak in both and 1 declines with size (`include_base_44_ukrainian`, on two rungs). The medians are R² 0.941 (IQR 0.888–0.966) for the size fit and 0.842 (0.692–0.903) for the trajectory fit; the median Spearman ρ with size is 1.0.

**The population is 193 of 539 tasks, and the figure did not say so.** 263 tasks have no fit at any L because they are at chance wherever a fit was possible, and a further 83 have a fit at a single L and so no median. Three families lose every task (`global_piqa_parallel_cloze` 0/63, `belebele` 0/59, `global_mmlu_full` 0/29) and `arc` keeps 2 of 28; the reformulated twins are what put those families back (`rf_belebele` 35/59, `rf_global_mmlu_full` 21/29, `rf_include_base_44` 13/36, `rfgm_include_base_44` 14/36). Figure 4 is the version of Figure 3 that carries this fact; the 20 BPB tasks it lists as below the fit minimum are the languages only the L50 mixture trains, which have one L setting by construction.

**The 1.7B BPB is under-predicted from every ladder top.** A power law fitted on 175M–600M or 175M–1B predicts the 1.7B per-language BPB with a median absolute relative error of 11.5 % (three rungs) and 5.7 % (four rungs), and every one of the 836 fits over-predicts the improvement (the signed error is negative throughout). We hypothesize that the exponent fitted on the small rungs is too steep for the reference's regime, which is a curvature finding worth a sentence in the paper rather than a symmetric error bar. CSV: `scaling_law_error.csv`.

**Caveats.** (i) The fit is made on the rungs that already cleared chance: 56 of the 60 three-rung fits lost their two smallest rungs to the gate, so a line through the rising top of a sigmoid is fitted, and R² is inflated by construction. (ii) The reference is inside every fit; R² is a property of the ladder including 1.7B, not a statement about predicting 1.7B from below. (iii) Spearman ρ over five points saturates (ρ = 1 in 448 of 592 benchmark fits); the paper figure should carry the fit's slope per decade beside R², or drop that panel. (iv) The trajectory fit runs over all saved points (about twenty for BPB, ten for benchmarks) and the WSD decay is not log-linear in tokens, so part of panel (b)'s spread is schedule. (v) The paper text still describes weak fits for `global_mmlu_full`, `belebele` and `truthfulqa`; on the current tables those families have no point at all, and their twins fit with median R² 0.91–0.95. That paragraph and Figure 3's caption need the population statement.

---

## 3. RQ2 — decision accuracy

### 3.1 Definitions

Decision accuracy (DA) is the share of design-variant pairs that a proxy orders the same way as the reference; a pair tied on both sides counts as agreement and a pair tied on one side as a miss (the kernel's one departure from upstream, which compared ties by listing order). Three references: **DA-size** compares the proxy size's final checkpoint with the 1.7B final; **DA-ckpt** an early checkpoint of a run with that run's own final; **DA-goal** an early checkpoint of the proxy with the 1.7B final. The tables carry two identities that we verify on every run: DA-size equals DA-goal at 100 % of the proxy's run, and DA-ckpt of the 1.7B run equals DA-goal of the 1.7B run (both to 0.0e+00). Pooled figures report matching decisions over comparable decisions summed over tasks; the `by_L` and `early_small` families report the mean over tasks. The two agree to two decimals on today's tables (0.782 against 0.792 at 1B on the `above_66_size` population) but they are different estimands, and the paper's three-panel figure mixes them on one y axis.

### 3.2 The full population: a smaller model is close to a coin flip

![Scale convergence, pooled](scale_convergence.png)

*Figure 5. Every grid-seed design-variant pair of every scheme, read at both models' final checkpoint against the 1.7B final, pooled over the gated benchmark tasks with at least three pairs (149 tasks at 175M, 246 at 1B; the BPB panel over the trained per-language BPB tasks). The band is the 90 % leave-one-family-out jackknife. Dotted: τ = 0.9. CSV: `scale_convergence.csv`; the mono-axis twin is `scale_convergence_one_axis.*`.*

**On every gated task, DA-size is 0.55 at 175M and 0.61 at 1B.** The pooled multi-axis reliability reads 0.552 [0.523, 0.581], 0.578 [0.548, 0.609], 0.592 [0.559, 0.625] and 0.609 [0.570, 0.649] at 175M, 350M, 600M and 1B; the mono-axis pairs read 0.530, 0.552, 0.563 and 0.567 with bands of the same width. The mean-over-tasks reading of `rq2_ten_checkpoints.csv` (A/B pool, 113–188 tasks) is 0.47–0.55 at every proxy size and checkpoint. Per-language BPB does better (0.58–0.82 on Figure 5's right panel) and the two aggregates best of all: `bpb_macro` ranks the variants at 0.91–0.97 and `train_loss` at 0.81–0.86 from any size.

**Interpretation.** A fully trained proxy below the reference orders the ladder's design variants on a benchmark barely better than chance, and the jackknife band says the result is not an artefact of which variants are in the grid. The design differences the ladder measures (language count, list, depth, temperature) move most benchmark scores by less than their noise at any one size, whereas BPB, which sums over every token, sees them.

### 3.3 The paper figure is a conditional statement

![Paper RQ2 figure, per-panel filters, mono-axis](rq2_above_66_either_transformation_one_axis.png)

*Figure 6. Left: DA-size per design axis (mono-axis pairs) and pooled, over the tasks whose median DA-size over the proxy sizes is ≥ 0.66 (37 tasks at 175M, 55 at 1.7B). Middle: DA-ckpt over the tasks whose median DA-ckpt is ≥ 0.66 (73–100 tasks). Right: DA-goal over the tasks passing either cut (74–113 tasks). The middle and right x axes are the proxy's own run in Chinchilla multiples. CSVs: `rq2_above_66_either_transformation_one_axis.csv`, `early_small_by_L_ckpt_above_66_ckpt_one_axis.csv`, `early_small_by_L_goal_above_66_either_one_axis.csv`.*

| line (mono-axis, `above_66_size` tasks) | 175M | 350M | 600M | 1B | tasks |
|---|---|---|---|---|---|
| all pairs | 0.600 | 0.725 | 0.747 | 0.764 | 37–54 |
| temperature (T = 1 vs 3) | 0.585 | 0.806 | 0.822 | 0.817 | 23–32 |
| language count | 0.579 | 0.771 | 0.834 | 0.798 | 23–32 |
| depth (deep vs shallow) | 0.680 | 0.704 | 0.550 | 0.747 | 23–33 |
| language list (A vs B) | 0.500 | 0.533 | 0.583 | 0.571 | 4–7 |

**On the filtered population the pooled DA-size rises from 0.60 to 0.76, and the list decision stays at chance.** The temperature and language-count decisions are the ones a 350M model already reads at 0.77–0.81; the depth decision dips to 0.55 at 600M and the language-list decision never leaves 0.50–0.58 on 4–7 tasks. **DA-ckpt** on its own filtered population rises from 0.57 at 10 % of the run to 0.87 at 90 % for the 175M run and to 0.83–0.85 for the 1B and 1.7B runs. **DA-goal** separates by size and stays separated: the 175M run reads 0.48 at 10 % and 0.54–0.61 thereafter, the 1B run 0.58 → 0.70.

**The rise is partly the cut.** The filter keeps the tasks whose DA-size cleared 0.66, so the left panel plots the quantity it selected on; on `predictivity_schemes` the passers read 0.795 at 1B against 0.549 for the rest. Every filtered rq2 figure, including the `above_80` "late" one whose 1B point is truncated at 0.80 by construction, is therefore a statement of the form "on the cells that rank reliably, this is how the reliability scales", and Figure 5 is the unconditional one. We keep Figure 6 because the paper's three claims hold on it (reading early is cheap, reading small is not, the pooled line hides which decisions recover), but §3.7 qualifies the first.

**Which tasks rank reliably.** On the multi-axis tables 68 of 298 gated tasks pass the DA-size cut, 140 the DA-ckpt cut, 146 either and 62 both; on mono-axis pairs 55, 108, 121 and 42. The DA-ckpt cut is the permissive one because its median runs over up to 45 cells (nine fractions × five sizes, the reference's own run included). CSV: `da_reliable_tasks.csv`; the per-language counts are `da_reliable_tasks_66_median.png`.

### 3.4 Language count: unordered, on any task set

![Per-L lines on the L8 languages](scale_convergence_L8.png)

*Figure 7. One line per language-count regime (pairs of design variants sharing the L; a regime pools depth, list and temperature decisions, `share_*` in the CSV), on the tasks in the eight L8 languages, every gated task. CSV: `scale_convergence_L8.csv`; `scale_convergence_L8common.*` keeps the 22 tasks with a cell in every regime at every proxy size, `scale_convergence_L_one_axis.*` the mono-axis pairs on every language.*

| regime, L8-language tasks (unfiltered) | 175M | 350M | 600M | 1B | tasks | decision mix at 1B |
|---|---|---|---|---|---|---|
| L8 | 0.462 | 0.518 | 0.491 | 0.592 | 22–40 | depth ⅓, list ⅓, two-axis ⅓ |
| L15 | 0.620 | 0.550 | 0.522 | 0.541 | 39–70 | depth 0.22, list 0.16, T 0.14, two-axis 0.47 |
| L30 | 0.549 | 0.554 | 0.553 | 0.506 | 39–70 | depth 0.20, list 0.20, T 0.10, two-axis 0.50 |
| L50 | 0.568 | 0.590 | 0.545 | 0.614 | 39–70 | depth ⅓, T ⅓, two-axis ⅓ |
| all pairs | 0.569 | 0.607 | 0.614 | 0.621 | 39–70 | L 0.16, two-axis 0.73 |

**The per-L lines do not order by L, and they do not move when the task set is fixed.** On the L8 languages every regime line lies between 0.46 and 0.62 with no monotone relation to L; on the 22 tasks common to every regime and size the lines read L8 0.46 → 0.61, L15 0.61 → 0.53, L30 0.55 → 0.53, L50 0.53 → 0.61. Restricting the languages leaves L8 and L15 unchanged (their tasks already sit in those languages) and cuts L30 and L50 by about three quarters; the earlier version of this figure, on all languages, differed from the L8 version only in those two lines. Filtered on `above_66_size` the same lines read 0.44–0.80 on 9–22 tasks and are again unordered. The mono-axis pairs keep every L line (L8 6 → 4 pairs, L30 10 → 5), against what the script's docstring claimed until this revision.

**Interpretation.** A regime line is not the effect of language count: within a regime the decisions are depth, list and temperature, in a mix that differs by regime and cannot be held fixed without falling below three pairs. What the figure shows is that the decisions within any regime are read equally poorly from a smaller model, on 4–5 families per regime.

### 3.5 One language at a time, and the tokens axis

![Per-language panels](scale_convergence_lang_all.png)

*Figure 8. One panel per L8 language, one line per regime that trains it plus the pooled line, on every gated task of the language; each point carries its task count, the legend the language's share of the regime's tokens and the tokens at 1.7B. The panel title carries the collapse test: R² of one log-linear line through every regime's points with x = size, then with x = tokens of the language. CSV: `scale_convergence_lang_all.csv`; the tokens-axis figure is `scale_convergence_lang_all_tokens.png`, the coverage table `scale_convergence_lang_all_coverage.csv`.*

**Every one of the eight languages draws a panel and none is ordered by L.** Per language the pooled line at 1B reads en 0.61, ru 0.69, zh 0.58, de 0.67, ja 0.57, es 0.61, fr 0.59, it 0.59, on 5–12 tasks; the regime lines within a panel cross each other at every size. Spanish, French and Italian have no L8 line: scheme B's L8 list does not contain them, so their L8 cells have two families and one pair.

**Exposure does not collapse the lines.** Re-expressing x as the tokens of the language the two members of a decision trained on (share × D(N), a relabelling of the (L, size) grid) does not bring the regime lines onto one curve: the collapse R² is 0.00–0.30 under size and 0.00–0.17 under tokens, and it rises under tokens for one language only (ru, 0.00 → 0.17). If reliability on a language's benchmarks were a function of how many of its tokens the models saw, the tokens axis would order the points; it does not. We read this as the same result as §3.2: the decisions are below the benchmarks' resolution at these sizes, whatever the exposure.

### 3.6 High-resource languages are not easier to read

![Reliability by language tier](reliability_by_language_tier.png)

*Figure 9. One line per language tier (the smallest scheme-A regime that trains the language: L8 = the eight high-resource languages, L50 = the twenty only the L50 mixture trains), the pooled multi-axis reliability over the tier's gated tasks, task counts under the points, 90 % jackknife bands. (a) every gated task; (b) the `above_66_size` tasks. CSV: `reliability_by_language_tier.csv`.*

![Reliability against language share](reliability_vs_language_share.png)

*Figure 10. One point per language with at least three gated tasks, per proxy size: pooled reliability against the language's share of all tokens in the L50 mixture. CSV: `reliability_vs_language_share.csv`.*

**The L8 tier leads at 175M by 0.04–0.07 and the tiers coincide by 1B.** Unfiltered, the tiers read L8 0.569 → 0.621, L15 0.534 → 0.608, L30 0.528 → 0.570 and L50 0.498 → 0.602 from 175M to 1B, on 39–70, 26–43, 41–63 and 43–70 tasks; at 1B the four jackknife intervals overlap (L8 [0.58, 0.66], L50 [0.49, 0.71]). Filtered on `above_66_size` every tier reads 0.60–0.66 at 175M and 0.77–0.84 at 1B with no order. Per language, the Spearman correlation between a language's share of the mixture and the reliability of its benchmarks is 0.18 (175M, 26 languages), 0.19 (350M, 40), 0.13 (600M, 41) and 0.03 (1B, 43).

**Interpretation.** We expected the high-resource languages, which every regime trains at the largest share, to rank the variants more reliably. They do so marginally at the smallest proxy and not at all by 1B, and across 43 languages the share explains nothing of the per-language reliability. One possible explanation is that the low-resource tier reads its decisions from the L50 cells alone (6 pairs per task, mostly the temperature and depth decisions at L50), which are the decisions Figure 6 reads best; the tiers therefore differ in decision mix as well as in resource level, and the two pull in opposite directions. A cleaner test needs the same decision set in every tier, which rule 5 forbids on this grid.

### 3.7 Seed uncertainty and the DA-ckpt null

![Seed uncertainty](seed_uncertainty.png)

*Figure 11. Left: DA of the proxy at each of its three seeds against the fixed 1.7B final, on the gated tasks the three seeds share (three markers, not a bar: three pairs put a DA on {0, ⅓, ⅔, 1}). Middle: the same designs ranked from seed s1 against seed s2 at one size, the test-retest ceiling. Right: DA-ckpt over pairs of two seeds of one design (the seed null, dashed) against the real design pairs (solid), mean over the gated tasks of `predictivity_seeds`. CSV: `seed_uncertainty.csv`.*

**Proxy-seed noise is ±0.05 to ±0.06 on English.** The three seeds read 0.72 / 0.78 / 0.83 at 175M (18 decisions on 6 tasks), 0.85 at all three seeds at 600M (33 decisions, verified per task to be a coincidence) and 0.68 / 0.73 / 0.80 at 1B (66 decisions, 11 tasks); Russian at 1B reads 0.78 / 0.82 / 0.85. The test-retest ceiling, what any proxy could reach if the reference itself were redrawn, is 0.83–0.89 at 175M, 0.82–0.85 at 600M and 0.77–0.80 at 1B on English, 0.78–0.89 on Russian. These replicated cells are three or four cross-L deep designs on one or two languages and say nothing about the pooled lines' populations; they give the scale of seed noise, not an interval.

**DA-ckpt is mostly within-run persistence.** Two seeds of one design have nothing to decide, so their DA-ckpt should be 0.5; it is 0.52 at 10 % of the 175M run and 0.82 at 90 %, against 0.54 and 0.84 for the real design pairs, and 0.51 → 0.72 against 0.49 → 0.75 at 1B. At every size and fraction the real pairs exceed the null by 0.02–0.03. The "reading early is cheap" reading of Figure 6's middle panel therefore measures how much a run's ranking at 90 % resembles its ranking at 100 %, which is high because the two checkpoints are the same run, not because the benchmark has resolved the design decision. The null exists for DA-ckpt only (no seed replicate at 1.7B), so the DA-size and DA-goal panels carry no such floor. No Wilson or binomial interval is drawn anywhere: the pairs share models and are not independent, and such an interval would be too narrow by an unknown factor.

### 3.8 Decision accuracy is Kendall's τ under another tie convention

![Three statistics](agreement_correlation.png)

*Figure 12. One point per (task, proxy size) cell (820): decision accuracy, Kendall τ_b and Spearman ρ of the proxy's final ranking of the design variants against the 1.7B final's, every pair, coloured by proxy size; Pearson and Spearman correlations over cells in the corner; dotted, τ = 2·DA − 1. Median 11 models per cell. CSV: `agreement_correlation.csv`; the per-cell values are `agreement_per_cell.csv`.*

![Cut sensitivity](agreement_cut_sensitivity.png)

*Figure 13. Per proxy size, the share of cells whose reliability verdict (DA ≥ 0.66, mapped to τ ≥ 0.32 through the identity) changes under each tie convention: τ_a, τ_b, Goodman–Kruskal γ (tied pairs dropped), and DA with the pairs the reference ties dropped. CSV: `agreement_cut_sensitivity.csv`; `agreement_identity.png` shows the tie term cell by cell.*

**The three statistics are one statistic.** Over the 820 cells r(DA, τ_b) = 0.976, r(DA, ρ) = 0.965 and r(τ_b, ρ) = 0.986; the Spearman versions are 0.970, 0.967 and 0.991. The relation to τ_a is exact: 2·DA − 1 = τ_a + (T_both − T_one)/n holds to 2.5e−16 on every cell, so DA and τ_a differ only in how tied pairs are counted. Ties are 5.0 % of the pairs, they touch 70 % of the cells, and 95 % of them are one-sided (the proxy ties a pair the reference decides, or vice versa), which our kernel counts as a miss and τ_a as neither.

**The convention moves the verdict on 3–7 % of cells.** Against the DA ≥ 0.66 set, τ_a flips 2.9–5.4 % of cells, τ_b 4.5–6.8 %, γ 5.9–7.2 % and DA with the reference's ties dropped 2.4–4.0 %, per proxy size. The high correlations are guaranteed by the identity and carry no information; the flip rate is the number that matters, and it says a published reliable-task list depends on the tie convention at the margin. Spearman ρ is not a rescaling of DA and has no mapped cut, so it is left out of Figure 13.

### 3.9 Multi-axis against mono-axis pairs

`rq2_above_66_both_axes.png` reads the three definitions on the same 62 cells under both pair sets. DA-ckpt is unchanged by the pair set (the 1.7B run reads 0.70 → 0.85 from 10 % to 90 % under mono-axis); DA-size and DA-goal move by a few hundredths in either direction on today's tables, and the earlier finding that multi-axis pairs read 0.04–0.05 higher does not survive the refreshed population. The mono-axis pairs are a third fewer, and `paper_rq2.py`'s left panel already draws them; the pooled black line of Figure 6 is drawn on the same set.

---

## 4. Key findings

First, the above-random gate is the largest single effect in the study: 450 of 975 benchmark-language tasks are at chance at every size and whole four-option families vanish, while their reformulated twins pass at 0.72–0.95 of their languages. Second, scaling is log-linear with median R² 0.94 on the 193 tasks that survive, but the fit is made on the surviving rungs with the reference inside it, and the 1.7B BPB is under-predicted from every ladder top (median 5.7 % from four rungs). Third, a smaller fully trained model orders the ladder's design variants at 0.55–0.61 on the full gated population and at 0.60–0.76 on the tasks selected for it; the language-list decision stays at chance on both. Fourth, an early checkpoint of a run predicts its own final ranking at 0.85, but two seeds of one design reach 0.82 on the same axis, so DA-ckpt measures persistence within a run. Fifth, no restriction of the task population (L8 languages, a common task set, one language, one language tier) orders the regime lines or lifts them out of noise, and a language's share of the mixture does not predict how reliably its benchmarks rank. Sixth, DA, Kendall's τ and Spearman's ρ agree at r ≥ 0.965 by construction, and the tie convention alone moves 3–7 % of reliable-task verdicts.

## 5. Limitations and room for improvement

The seed replicates cover three or four designs on English at three sizes and on Russian at 1B; every interval elsewhere is a jackknife over families, which measures dependence on the grid, not seed noise. The per-L and per-tier comparisons confound language count and resource level with the decision mix, and the grid cannot hold the mix fixed above three pairs; an L8-only replicate of the depth and list decisions in every regime would settle it. The DA-ckpt null should be drawn on every checkpoint-axis figure the paper shows, and the paper's middle panel read against it. The `above_*` filters select on the quantity drawn; the paper should quote Figure 5 beside Figure 6 or restate Figure 6 as conditional. The zh and ja panels of the per-language figure rest on one to two families below 1B because the reformulated twins were evaluated on a subset of cells; an evaluation top-up, not analysis code, would fill them. The gate's run count per cell (2 to 18) should appear on the gate figures, and the rq00 signal ranking should apply the gate. The paper's RQ1 paragraph and the `rq2.*` figure name still describe earlier tables (`documents/paper/sections/04_analysis.tex`, `make_rq_figures.py`), and the hand-written "The RQ2 figure" section of `analysis/rq02_decision_accuracy/README.md` quotes the 23-cell population of 2026-09-21; both need the numbers above.
