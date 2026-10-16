# Changes applied to `documents/overleaf/13-10-review/` (2026-10-13)

Sources: `language.md` (L), `claims.md` (C), `methodology.md` (M), `../citation-review.md` (CR).
Item ids follow those files (e.g. L-A3, C-(A)6, M-M6, CR-A5). Nothing was committed. The
working copy is identical to `13-10-12h-full-overleaf` (the reviewed snapshot), so the reports'
line numbers apply. Review with `git -C documents/overleaf/13-10-review diff`.

Numbers that the reports disagreed on, or that I re-checked in the repo CSVs:

| Value | Paper had | Now | Evidence |
|---|---|---|---|
| Fig. 3a all pairs (macro, as plotted) | 59--61% | 0.60--0.64 | `rq2_da_all_above_66_either_transformation_mono_axis.csv`: 0.642/0.609/0.633/0.602/0.640 |
| Temperature line (macro) | 72--75% | 0.73--0.75 | same CSV: 0.727--0.753 |
| Scaling fits | 431 tasks, R² 0.87 | 419 tasks, 0.88 | `scaling_regimes.csv` non-bBPB: 419 rows, median 0.878; 311/419 both |
| BPB size fits at R² 1.00 | "50 languages" | 30 languages (2+ settings) | same CSV: 30 BPB rows, median 0.996 (= Fig. 2b "bpb (30 tasks)") |
| TruthfulQA example | "TruthfulQA" | Spanish TruthfulQA-Multi (MC1) | `truthfulqa-multi_mc1_es`: R² 0.972, ρ −0.75 |
| §5 box, >2 seed SD | 0.69--0.73 | 0.59--0.67 (accuracy) | `da_size_definition_sensitivity_multi_axes.csv` `seed_sd_2x` accuracy 0.592--0.674 (0.69--0.73 is accuracy+bBPB) |
| §5 box, accuracy | 0.51--0.56 | unchanged, population named | same CSV `current` accuracy 0.506--0.560 |
| 600M BPB DA-target | 0.70--0.74 | 0.69--0.74 | `da_goal_multi_axes_across_langs_bpb_paper.csv` 600M 0.694--0.741 |
| App. E.3 acc+bBPB DA-size | 0.58 ×5 (pair-weighted) next to BPB macro | 0.62/0.59/0.61/0.60/0.61 (macro, as BPB) | `scale_convergence_da_size_multi_axes.csv` `reliability_macro` |
| E.4 result sentence | none | 90M 0.90 at 1st tenth (~8M tokens); 600M 0.69--0.74 to 0.5B; 1B 0.75→0.92 | `da_goal_multi_axes_across_langs_bpb_paper.csv` |
| Language coverage (App. C) | English 31, median 8, zh 17, it 13, lv 2 | 25, 7, 16, 12, 1 | read from Table 4 itself (median over the 49 trained non-English rows = 7) |
| Training grid 3B | A,B at K∈{8,15,30}; 57/23 runs | A K∈{8,15,30,50} 58; B K∈{8,15} 22 | copied from `documents/paper/sections/app_01_model_ladder.tex` |
| Compute macros | 40,500 / 9,300 / 199,200 | 42,900 / 11,000 / 215,600 | `documents/paper/sections/main.tex` constants block |
| Threshold-table formats | "ten of 30" | "seven of 28" | counted in Table 7 as printed |

## (1) Applied changes, by file

### main.tex (8)
- Macros `\npretrainnodehours`, `\nothernodehours`, `\ngpuhours` → 42,900 / 11,000 / 215,600 (C-(A)13).
- Abstract: "ladder of \nmodels multilingual models" → "\nruns models" (213 vs Table 3's 218; C-(C)1, L-A3, M minor 1). `\nmodels` is now unused in compiled text (still defined).
- Abstract: "study this question for multilingual LLMs, and pretrain" → two sentences (L-C2).
- Abstract: "language modelling" → "modeling" (L-C1).
- Abstract: "how well the performance of smaller models predicts larger model performance" → "how predictably performance scales from smaller to larger models" (no scaling-law-error result in the paper; C-(B) main:191).
- Abstract: factors sentence rewritten so tokens and item format govern measurability, scoring affects ranking, proxy size has no consistent effect; "per-language" → "per language" (C-(A)1–3, L-C2, M-M4).

### sections/01_introduction.tex (12)
- Added `heineman_signal_2025` to the "previously studied" cite so "these works" has a plural antecedent (L-C2 01:11).
- "language modelling" → "modeling"; "conclusions of" → "conclusions about" (L-C1, L-C2).
- "scaling behaviour" → "behavior" (L-C1).
- "language--benchmark pairs" → "benchmark--language tasks" (L-B unit of analysis).
- Second question: "per-language training token counts" → "training checkpoints" (§5 studies size and checkpoint, not tokens; C-(A)1, M-M4).
- "\nmodels models" → "\nruns models"; "benchmark-language" → "benchmark--language" (L-A3, L-B).
- "across model size and language training tokens" → "across model size and training progress" (C-(A)1).
- Findings sentence: tokens and answer format → measurability ("a prerequisite for decision reliability"); scoring → decision reliability over proxy size (C-(A)1–3).
- Reformulation sentence: "reformatting" → "reformulation"; grammar fixed ("measurable at sizes where their original letter format stays at chance"); Belebele, Global-MMLU, INCLUDE cited at first mention (L-C2 01:31, L-D).
- "task accuracy scales predictably" → "the accuracy of most tasks" (74%; C-(B) 01:31).
- "do not transfer to our multilingual ladder" → "predict it only moderately on our multilingual ladder" (§6 shows ρ≈0.53--0.62; C-(A)4, M-M6.8).
- Contribution 2: dropped "and at which we observe decision reliability" (Table 7 gives above-chance thresholds only; C-(A)1).
- Contribution 3: "reformatting" → "reformulation" (L-B).

### sections/02_related_work.tex (3)
- Paragraph headers end with a period (L-B paragraph headings).
- Added `polo_tinybenchmarks_2024` to the "small, informative subsets" claim (L-D, CR-C12).

### sections/03_methodology.tex (10)
- "\nmodels" → "\nruns"; "between 90M to 1B" → "from 90M to 1B" (L-A3, L-C2).
- Cited DCLM-Edu, FineWeb-2 and WSD in the main text (CR-C15, L-D).
- `Appendix~\ref{app:model-ladder}` → `\ref{app:models}` (pointed to A.1, not the grid; L-A4).
- First bullet defines $K \in \{1,2,8,15,30,50\}$ (L-B number of languages).
- Scheme bullet: "(and a third, C, at $K \le 2$)" (L-B scheme M).
- Temperature bullet cites Arivazhagan et al. 2019 and Conneau et al. 2020 (CR-C9).
- "a deep with a shallow" → "a deep and a shallow" (L-C2).
- Muon: "on the deep architecture" (C-(B) 03:64).
- "scaling behaviour" → "behavior" (L-C1).

### sections/04_analysis.tex (~47)
§4 Measurability
- "which determines whether a score is" → "that is, whether its score is" (L-C2 04:23).
- Wilson bound cited (CR-C1).
- "model-size passes the filter" → "model size passes the above-chance threshold"; "don't" → "do not"; "filtering"/"filter" → "threshold" throughout §4 prose and box (L-B above-chance test).
- "Section \ref" → `\S\ref` in §4 and §5 running text (L-B section references).
- "analyse/analysed" → "analyze/analyzed" (L-C1).
- 38%/34% pointer: Appendix D.1 first, Figure 5 labelled as the per-benchmark breakdown (L-A4).
- "at most three languages subsets" → "at most three trained-language subsets" (true on trained languages, C-(B) 04:40; L-A3 suggested five, which is the all-language count used in §4.2).
- Cited HellaSwag (Zellers + Okapi), MultiBLiMP, TokSuite at first mention (L-D, CR-A5).
- "42% of tasks", "a third of tasks", "43% of training configurations" → "evaluations" (these are (task, K, checkpoint) evaluations; C-(B) 04:48, L-B evaluation unit). Double space removed (L-C2).
- Chinchilla budget cited to Hoffmann et al. (CR-C3).
- Fig. 1 caption: "above chance threshold" → "above-chance threshold"; "(language, K, checkpoint) evaluations" (L-B).
- Fig. 2 caption: missing final period (L-C2).
- §4.2 title "Factors Influencing the Above-Chance Threshold" (L-B).
- Letter-format mapping claim: added `robinson_leveraging_2023` (author list fixed in bib) and `wiegreffe_answer_2025` (CR-C11).
- RF and LLM-RF defined at first main-text use (L-B reformulation).
- "letter-based format" → "letter format" (L-B answer formats).
- Fig. 3 caption: "on single-axis pairs"; "the results are conditional on this selection" (M-M3, methodology rule 7, L-A6); "1C is $20N$ tokens (20% of a run)" (L-A7); panel names DA-size / DA-ckpt / DA-target (L-A7); "(a) pooled over decisions" → "averaged over tasks" (the plotted column is the task macro; C-(A)6).
- §4.3: 431 → 419 accuracy and BPB tasks, 0.87 → 0.88; BPB 1.00 "of the 30 languages trained at two or more language settings" (C-(A)5, L-A3).
- TruthfulQA example: "the Spanish TruthfulQA-Multi (MC1) task", cited; "decrease in predictability" → "decrease in accuracy"; "model size-fit" → "model-size-fit" (L-A3, C-(B), M minor 9).
- Spearman's ρ cited to Spearman 1904 (CR-C4).
- "The same pattern holds for monolingual English models" → "A similar pattern" (English L1 has 1--2 training-only tasks per corpus; C-(B) 04:134).
- Box 4: "does not perform measurably on these benchmarks" → "is not measurable at these scales"; "On the tasks the filter retains" → "On most tasks the threshold retains" (L-C2, C-(B)).
§5 Agreement
- "summarised" → "summarized" (L-C1).
- DA names introduced: "we write DA-ckpt, DA-size and DA-target for DA in the three settings" (L-A7, L-B).
- Three-pair rule: "comparisons between only two pairs yield binary agreement scores" → "fewer pairs give only coarse agreement scores (0, 0.5 or 1)" (L-C2, M minor 5, methodology rule 7).
- "benchmark–language" (Unicode, dropped by the font) → `--` (L-C2, C-(C)19).
- DA percentages → fractions: 0.62--0.68, 0.83--0.93, 0.60--0.64 (was 59--61%, wrong for the plotted figure), 0.73--0.75 (was 72--75), 0.58--0.64 (L-C3, C-(A)6).
- "350M models generally exhibit the highest agreement" → "in the second half of training" (C-(B) 04:215, M minor 13).
- 3B sentence: "benchmark accuracy shows ... DA of 0.51--0.52" → "benchmark accuracy and bBPB show ... a DA of 0.51" (the population is accuracy+bBPB; one value, matching App. E.5; L-A3, C-(B) 04:241). Double space removed.
- §5.4: "(i.e., model variants with very close rankings)" → "(\ie variants whose reference scores are nearly equal)"; "Accounting for near-tied scores as expected" → "As expected, accounting for near ties" (L-C2).
- Box 5: names the population of 0.51--0.56 (accuracy, every pair of variants (multi-axis), every task above chance); "proxy point" → "final proxy checkpoint"; ranges with `--`; seed-SD number 0.69--0.73 → 0.59--0.67 "of accuracy pairs" (C-(A)7, L-A3, L-C3).
§6 Anticipation
- "In this analysis section, we investigate" → "We investigate" (L-C2).
- Fig. 4 population: "and their bBPB versions, on every pair of variants (multi-axis)" (C-(B) 04:309, L-B single/multi-axis).
- Fig. 4 caption: "at the 1.7B." → "with the 1.7B reference." (L-C2).
- HellaSwag sentence: ρ=0.93 (best of 132 ratios) vs 0.75 (one fixed SNR) → like with like: the relative-std SNR reaches 0.82 within HellaSwag vs 0.75 pooled (`rq04_surrogates/README.md` strata bullet; C-(A)11, L-A3, M-M6.8).
- "relative-standard-deviation noise reaches ρ=0.56" → "noise alone" (matches Fig. 4 row; L-B SNR naming).
- 600M example: 0.70--0.74 → 0.69--0.74, plus "(its depth pair is not size-matched; Appendix A.2)" (C-(B) 04:343, M-M9, methodology rule 7).
- Box 6: "No metrics ... predicts" → "No metric ... predicts" (L-C2).

### sections/05_conclusions.tex (4)
- "\nmodels" → "\nruns" (L-A3).
- "Reformulating tasks and using likelihood-based scores can improve evaluations' decision reliability" → reformulation makes letter-format benchmarks measurable (a prerequisite), likelihood-based scores can improve decision reliability (C-(A)2).
- "favour" → "favor"; double space removed (L-C1, L-C2).

### sections/06_statements.tex (5)
- AI Use: "as described in the methodology" → "as described in §4.2" (L-A4).
- Limitations: dropped "and we report fixed-population comparisons" (no fixed-population result is compiled; L-A4, C-(B) 06:13, M minor 16).
- "Ethical considerations" → "Ethical Considerations" (L-C2).
- "datasets’" (U+2019, dropped by the font) → "datasets'" (C-(C)19).
- "favour" → "favor" (L-C1).

### sections/app_01_model_ladder.tex (26)
- Table 1: depth symbol $A$ → $\mathcal{D}$ (clashed with activation and token budget $D$); seeds row "1904 (default); replicates 64, 313, 28, 1797" (L-B symbols/seeds, C-(B) app_01:27).
- Table 3 (training grid): rows A-deep and B-deep copied from the generated table: 3B at A K∈{8,15,30,50} (58 runs), B K∈{8,15} (22 runs) (C-(A)8, L-A3). FWEB row cites FineWeb (CR-C5).
- "\nmodels" → "\nruns" (L-A3).
- "All sizes are non-embedding parameters" → "count non-embedding parameters"; "smallest models sizes" → "model sizes" (L-C2).
- Chinchilla convention cited to Hoffmann et al. (new cite; the author's commented cite is left as is) (CR-C3).
- RoPE scaling cited to Llama 3 (CR-C7); WSD co-cites MiniCPM (CR-A6); Muon/AdamW cites Loshchilov & Hutter (CR-C8); temperature sampling cites Arivazhagan + Conneau (CR-C9); FineWeb cited in the scheme paragraph (CR-C5).
- "DCLM-edu" → "DCLM-Edu" (L-B names).
- Depth paragraph: "($D$)" → "($\mathcal{D}$)"; "a deep with" → "a deep and"; "0.4% to 5.2%" → "0.4--5.2%"; added that the 600M pair is not size-matched (616.6M vs 594.5M, +3.7%, from Table 2) (L-C2, L-C3, M-M9, methodology rule 7).
- Activation paragraph "($A$)" → "($A_f$)" to match Table 1; double space removed (L-B, L-C2).
- "optimiser" → "optimizer"; "training grid data" → "training data" (L-C1, L-C2).
- Inline formulas $D(N)=100N$, $\tilde U$, FLOPs, $q_i$ → display `$$` (appendix formula style).
- "shared points $k/20$" → "shared fractions" (no "point" for data units).
- Compute: "3 to 21" → "3--21"; "node hours" → "node-hours" (L-C2, L-C3).

### sections/app_02_data_mixtures.tex (6)
- "prioritises" ×2 → "prioritizes"; "towards" → "toward" (L-C1).
- "The lists are nested, \ie the" → "The lists are nested: the" (L-C2).
- "against the 10 and 10 from scheme A" → "against 10 scripts and 10 families for scheme A" (L-C2).
- FineWeb cited (CR-C5); temperature formula cited (CR-C9).

### sections/app_02_table_languages.tex (2, caption only)
- Caption title "Training and validation languages"; "FineWeb2" → "FineWeb-2" (L-C2, L-B names).
- ``val" → ``val''; comma splice → semicolon (L-C2).

### sections/app_03_evaluation.tex (19)
- Formats "MCQA-style letters, cloze, and generative continuation" → "letter, cloze, and completion" (L-B answer formats).
- Subsection "Evaluation Configuration" holding only BPB → `\subsection{Bits-per-Byte (BPB)}`, subsubsection level dropped (L-B evaluation configuration).
- "behaviour" → "behavior"; leading space removed; bytes symbol $b_\ell$ → $\beta_\ell$ (clashed with the language task $b_\ell$; $B$ is taken by the suite) (L-C1, L-C2, L-B symbols).
- Deleted the repeated "BPB connects the analysis to loss-based scaling laws" sentence (L-C2 app_03:46).
- "12 checkpoints per model size" → "per run" (L-C2).
- Task-selection sentence: replaced "removing tasks for which the final checkpoints of the 600M, 1B, 1.7B and 3B models did not achieve above-chance performance" with the rule actually used (the 229 discarded tasks at chance at 1B and 1.7B; the remaining screened tasks left out for cost or redundancy) (C-(A)10, L-A3, M minor 3).
- Language coverage: counts aligned with Table 4 and its basis stated (English 25, median seven, Mandarin 16, German/French/Hindi/Vietnamese 13, Latvian one family); removed the duplicated screening sentence and the wrong claim that most low-resource benchmarks failed at 1B/1.7B (286/292 pass) (L-A3, C-(B) app_03:66, L-C2).
- "reformat ... (Section~\ref{sec:rq1})" → "reformulate ... (RF) ... (LLM-RF) (§4.2)" (L-A4, L-B).
- LAMBADA cited to Paperno et al. (CR-A1).
- Harness sentence; "normalised" ×2 → "normalized"; paragraph headers end with a period; "the Harness" → "the harness"; "We propose it as a measurement for tasks whose accuracy is at chance" (L-C1, L-C2).

### sections/app_03_table_benchmarks.tex (17)
- Trained > Langs fixed from the generated rows: ACP-Bench 1/1, BBH 1/1, CulturalBench 8/8 (the hand-merge summed two rows) (L-A3, C-(B) Tab. 6, CR-D).
- Citations: ARC (MT)/(Okapi) + Clark et al. 2018; HellaSwag (Okapi) + Zellers; MMLU (Okapi) + Hendrycks; TruthfulQA (Okapi, MC1/MC2) + Lin; LAMBADA → Paperno + Radford (was the harness); AlGhafa translated → AlGhafa + Okapi + OALL; Arabic MMLU (OALL) → AceGPT + OALL (CR-A1, A2, A3, A5).
- "Analysed" → "Analyzed"; "Language modelling" → "modeling" (L-C1).
- Caption: "Langs.\ is"; quotes ``…''; RF/LLM-RF defined; "Mid-Resource Language Examples" → "Mid-Resource Langs." to match the header; "ranks 30 to 50 ... from the least resourced up" → "ranks 30--50 of our 50-language list, which orders FineWeb-2 languages by available data" (L-C2).
- First column made a wrapping `p{3.6cm}` column so the added citations fit (also removes the table's pre-existing 8pt overfull box).

### sections/app_04_chance_threshold.tex (30)
- Wilson cited (CR-C1); McNemar cited (CR-C2); Spearman cited (CR-C4).
- "The resulting mask is the gate that every analysis reads" → "Every later analysis uses only the $(b_\ell,N)$ pairs that pass" (L-B, L-C2).
- Margins in fractions (0.090, 0.024, 0.007, 0.006) to match Table 7 (L-C2 app_04:27).
- Fig. 5 caption: "above-chance threshold"; "colored"; 3B reads the six 3B configurations (L-B, L-C1, C-(B) app_04 Fig).
- Majority-class paragraph: "labelled" → "labeled"; "1.5 to 2.8" → "1.5--2.8"; list of removed tasks completed (mostly NorEval; also BoolQ, BBH) (C-(B) app_04:52).
- Length-cue statistic scoped "for INCLUDE" (C-(B) app_04:73).
- Fig. 7: "across the gate" → "above the threshold"; "twin" → "RF version"; "p $<$ 0.05" → "$p<0.05$" (L-C2).
- D.3 title in Title Case; heading/body/caption made consistent ("rises from 10M tokens", "near 40% up to 10M"); "$10^5$" → "100K"; "(Figure.)" punctuation (L-C2 app_04:95).
- "ten of the 30 benchmark formats" → "seven of the 28" (L-A3, C-(B)); "than a 90M one" → "than a 600M one" (the example compares 600M; C-(B) app_04:97).
- Fig. 8 caption: "Each point ... cells" → "Each marker ... evaluations".
- Table 7 caption quotes ``Lang.'' / ``Margin''.
- D.4: "behaviour"; "number of model parameters" → "non-embedding parameters"; "above-chance filtering criterion" → "pass the above-chance threshold"; "MCQ" → "multiple-choice"; parenthetical moved before the full stop; paragraph period inside braces; "filter" → "threshold"; "single-cell fits" → "fits on a single language setting" (L-C1, L-C2, L-B).
- Fig. 9/10 captions: comma splice; "colour" → "color" (L-C2).
- Fig. 11 caption: "are among the tasks predictable" (the list is 12 of the 17); "(mc2)" → "(MC2)"; "filter" → "threshold"; "colour" → "color" (C-(B), L-B).
- "not filtered by this procedure" → "not subject to this procedure"; "analyse" → "analyze".

### sections/app_05_ranking_agreement.tex (18)
- Table 8 column "RA" → "DA" with DA-size / DA-ckpt / DA-target; removed the RA-* naming sentence; "When computing RA" → "DA" (L-B ranking-agreement settings).
- "proxy/reference point" → "proxy/reference checkpoint"; "at a point" → "at a checkpoint"; "A cell $(b_\ell,N,f)$" → "A value" (rule: no "cell"/"point").
- "pairs of model configurations" → "variant pairs" ×2 (L-B).
- "we call" → "we write" (L-C2).
- Example $\theta'$ attached to the sentence with "\eg", depth symbol $\mathcal{D}$ (L-C2, L-B symbols).
- "every table is also computed over the multi-axis pairs" → names Fig. 4, Fig. 12 and App. E.3 (L-C2 app_05_ranking:57).
- Comma splices → semicolons ×2; "the number of the pairs tied" → "the numbers of pairs tied" (L-C2).
- Kendall and Spearman cited (CR-C4).
- Fig. 12 caption: "One point ... (6,803 points)" → "One marker ... (6,803 values)"; "Spearman's correlation over the points" → "Spearman's ρ over the values"; "Colour" → "Color" (L-C2).

### sections/app_05_da_sensitivity.tex (8)
- "proxy/reference point(s)", "at both points" → checkpoint(s) throughout.
- "DA-checkpoint" → "DA-ckpt" ×2 (L-B).
- "The 1.7B point" → "The 1.7B value".
- Dropped "Overall," opener; rewrote the closing "Thus, even when exact ranking recovery remains weak, ..." sentence (L-E app_05_da_sensitivity:67).

### sections/app_05_da_analyses.tex (13)
- "design pairs" → "variant pairs"; "every pair of configurations" → "every pair of variants" (L-B).
- E.3 heading: "predictably" removed (L-C2); acc+bBPB DA-size switched from pair-weighted (0.58 ×5) to the task macro (0.62/0.59/0.61/0.60/0.61) so the sentence uses one aggregation with its BPB numbers and with Fig. 3 / the §5 box (C-(B) app_05_da_analyses:8, C-(C)16, M-M7).
- Fig. 13 caption: "Colour centred at 0.5." → "The color scale is centered at 0.5." (L-C2).
- Headings: "bBPB still ranks the variant pairs where accuracy is at chance"; "... better than the proxy's own accuracy does" (L-C2).
- DA-goal → DA-target in E.3 Fig. 16, E.4 text and Fig. 18, E.5 Fig. 19 (L-B).
- E.4: added a result paragraph citing Fig. 18 (90M 0.90 at its first tenth after ~8M tokens; 600M 0.69--0.74 up to 0.5B; 1B 0.75→0.92; tokens do not set the level) (L-A4, C-(B) app_05_da_analyses:47, M-M4).
- Fig. 18 caption: "checkpoint.}." double period; comma splice (L-A5, L-C2).
- E.5: "Generalisation" → "Generalization"; "above chance" → "above-chance"; "it raises" → "the 3B models raise"; "for benchmark accuracy" → "accuracy and bBPB" (twice); "as against" → "the same as against"; 0.75--0.77 → 0.75--0.78 (L-C1, L-C2, C-(B)).

### sections/app_06_snr.tex (14)
- "and analyses of" → "and details the analyses of" (L-C2).
- Noise window: "population standard deviation $s_a$" → residual SD around a least-squares line (detrended), the definition the code uses (RULES rule 4; C-(B) app_06_snr:101, M-M6.7, methodology rule 7).
- Fig. 21 caption: "leads or ties" → "leads from 90M to 600M" (at 1B it does not lead); comma splices (C-(B), L-C2).
- "three candidate signal definitions to capture" → "three signal definitions, each capturing" (L-C2).
- Discrepancy: broken sentence ending in "with $u_a = …$," (equation commented out) now ends after the definition, cites Hickernell 1998, keeps the SciPy note (L-A5, CR-C13). The commented equation is untouched.
- Seed noise: formula rewritten with defined symbols ($U_r(t_j)$, $k$, $J$; no $M$, $n$, $T$) (L-A5).
- Total variation: $T$ → $J$ (clashed with temperature) (L-A5).
- Checkpoint noise: "when the first three are too expensive" → "when seed noise and total variation are too expensive"; formula changed to the detrended residual SD actually used, stating that Heineman et al. use the plain SD (L-A5, M-M6.7, methodology rule 7). Split over two aligned lines to avoid an overfull box.
- "over the cells trained with three seeds" → "configurations"; deleted "and §6 reports every intervention's effect against the seed noise" (not in §6; L-A4, C-(B) app_06_snr:105).
- Benchmark noise: model set $M$ → $\mathcal{M}$, folds $k$ → $F$ (L-A5 symbol clashes).
- Margin symbol $M$ → "Margin" (clashed with scheme $M$).

### sections/app_06_surrogates.tex (3)
- Broken heading "…better with decision accuracy.}(Figure 20)." → a heading plus one sentence citing Figure 20 (L-A5).
- Accuracy rel-std range 0.33--0.40 → 0.33--0.39 (C-(B)).
- Removed "On single-axis pairs alone, all correlations are weaker, and the relative-standard-deviation SNR leads at every proxy size" (its figure is commented out; restoring it is the author's call, see decisions); added that 600M's depth pair is not size-matched (L-A4, M-M9).

### custom.bib (15)
- Fixed: `heineman_signal_2025` pages 19286--19327 (CR-B1); `li_adaptive_2025` PMLR series/volume/pages/publisher (CR-B2); `messmer_enhancing_2026` volume/pages/note/DOI (CR-B3); `magnini_leaderboard_2025` dropped the wrong eprint (CR-B4); `robinson_leveraging_2023` adds Rytting (CR-C11).
- Added, verbatim from citation-review.md section C: `spearman_proof_1904`, `penedo_fineweb_2024`, `zellers_hellaswag_2019`, `grattafiori_llama3_2024`, `loshchilov_decoupled_2019`, `arivazhagan_massively_2019`, `conneau_unsupervised_2020`, `polo_tinybenchmarks_2024`, `hickernell_generalized_1998`.

### Build check
- `rsync -a --exclude .git 13-10-review/ build-1310/` and `tectonic main.tex` in `build-1310` (log in `build-1310/build-after.out`; the reviewer's `build.out` is kept).
- Result: success, **44 pages** (was 43), no undefined references or citations, no missing-character warnings (the two Unicode drops are fixed).
- Overfull boxes: the two pre-existing ones remain (`app_02_table_languages` 3.2pt, `app_05_da_sensitivity` 13.1pt); the benchmark table's 8pt one is gone; none introduced.
- **Page limit:** the Conclusion now starts at the bottom of page 9, left column, and ends at the top of page 9, right column (before "AI Use Statement"). In the baseline it started at the bottom of page 8, right column, and ended on page 9. The main text grew by about 21 typeset lines, mostly from the citations added at first mention (Belebele, Global-MMLU, INCLUDE, HellaSwag, MultiBLiMP, TokSuite, DCLM-Edu, FineWeb-2, WSD, temperature, Wilson, Chinchilla, Robinson/Wiegreffe, Spearman, TruthfulQA-Multi) and the corrected claims. The paper was already about 30 lines over; it is now about 50 lines (one column) over.

## (2) Decisions for the author (not applied), ranked

### Anonymity and page limit
1. **Anonymity leak, guide footnote** (`01_introduction.tex:38`, L-A1, CR-D1). `https://scaling-reliability.netlify.app` serves the Slidev deck whose JS bundle names the authors and EPFL. Fix: remove the footnote for review, or point it to an anonymized mirror whose built JS has no names, EPFL, swiss-ai or W&B/HF links; also reconcile it with footnote 1 ("released upon publication"). Not touched per instructions.
2. **Over the 8-page limit by about one column** (L-A2; see Build check). Cut candidates: (a) §4.2's duplicated reformulation detail (the 5→86/105 sentence repeats App. D.2); (b) the §5 intro paragraph that restates Table 8 (`04:184–188`); (c) §5.4, which only points to the appendix (`04:252–254`); (d) §6's "Overall, surrogate signal emerges…" and "These findings suggest that reliable evaluation at smaller scales does not necessarily imply…" (both restate `04:338`; L-E); (e) the duplicated cross-lingual-transfer sentence (`01:13` = `02:7`; L-C2); (f) the three teal Summary boxes; (g) move some first-mention benchmark citations to Table 6 only if the venue allows.
3. **Apertus footnote** (`app_01:130`, `huggingface.co/swiss-ai/Apertus-70B-2509`) plus "nodes of four NVIDIA GH200 GPUs" point to the authors' ecosystem; allowed, but the citation is enough (L "not blocking", CR-D4).
4. **Source comments** (`% Catherine:`, `\clara{}`, `\AR{Fixed}`, `% MARIA`, `% TODO`) must be stripped before any source upload; `[review]` must stay (L, CR-D5).

### Headline claims
5. **Seed-noise sentence in the §5 box** ("Seed-to-seed noise … accounts for part of the disagreement"; now 0.59--0.67 on accuracy) rests on `tables/app_rq02_definition_sensitivity.tex`, which is not `\input`, and the seed paragraph in E.3 is commented out. Either input that table (or uncomment the seed paragraph) or drop the sentence (L-A3, C-(A)7, M-M1).
6. **0.51--0.56 in the §5 box** is now labelled (accuracy, multi-axis, every task above chance) but no compiled figure or table shows it; the same table as item 5 would. The paper reports DA-size five ways (Fig. 3 macro single-axis filtered; box multi-axis unfiltered accuracy; E.3 multi-axis acc+bBPB; 3B; reliable-task accuracy). Suggested: one canonical population with the rest as labelled sensitivities in one appendix table (M-M7).
7. **Selection on the outcome in Fig. 3** (median DA ≥ 0.66). The caption now says the results are conditional on the selection. The cross-fit control ("selection adds 0.01--0.02") is commented out in E.3 (`app_05_da_analyses.tex:22`); restoring it, or making the unfiltered population the main Fig. 3, is the author's call (L-A6, M-M3).
8. **"Sampling temperature consistently higher" rests on four pairs; activation and optimizer on three** — no uncertainty is shown. Suggested: pair counts per axis line and the leave-one-family-out jackknife bands already in the Fig. 3 CSV; the commented E.3 paragraph on temperature covers this (M-M2, C-(B) 04:229).
9. **"Spend ablation compute on more sizes rather than longer runs"** (`04:137`). R² over 3--6 size points vs 10 checkpoints is biased; intermediate WSD checkpoints are not shorter runs; §5 shows DA-size flat in size. Suggested: drop or restate as "within one 100N run, accuracy trajectories are less often log-linear than size curves", and add a WSD sentence (M-M5, C-(B) 04:137).
10. **Monolingual SNR comparison** that would support the intro's original "do not transfer" (ρ 0.89 on DataDecide vs 0.46/0.43/0.04/−0.10 at K = 1/2/15/30) is in `tables/app_snr_monolingual_table.tex`, not `\input`. I softened the intro instead; including the table is the author's decision (C-(A)4, M-M6.8).
11. **Recommendations promised in contribution 3** ("task filtering, reformulation, and scoring") have no explicit recommendations paragraph; the draft one in the Conclusion is commented out (C-(B) 01:42).
12. **"Sensitivity of ranking agreement to … per-language training-token budgets"** (`04:188, 252`) is now backed only by the new E.4 result sentence (tokens do not set BPB DA-target). If the author wants the negative token result stated for benchmarks too, the language-tier table (`reliability_da_size_by_language_tier_multi_axes.csv`: K=50 tier highest) exists but is not in the paper (C-(A)1, M-M4).
13. **HellaSwag 0.82 vs 0.75** now compares like with like, but neither number is in a compiled figure or table (they come from `rq04_surrogates/README.md`, clustered, pooled over proxies; Fig. 4's pooled max is 0.62). Either add the strata table or rephrase against Fig. 4 (L-A3, C-(A)11).
14. **600M per-language BPB numbers** (checkpoint noise 0.4% vs 1.3%, rel-std SNR 8.7 vs 2.0) have no pointer to a table or appendix (L-A3). The example's DA dip is the non-size-matched depth pair plus one seed draw (with depth held fixed, 600M BPB DA ≈ 0.86; the commented E.4 paragraph has this). Consider replacing the example (M-M9).

### Methodology majors (need new analysis, restructuring, or restored content)
15. **Reference ceiling (M-M1):** add the seed test–retest DA (0.53--0.65 on English), decisive-pairs DA, and the permutation-null share (`app_rq02_seed_uncertainty.png`, `app_rq02_permutation_null.png`, `app_rq02_decisive_pairs.png` are in `figures/` but not input); reword the 3B paragraph as a reproducibility statement; ideally train a second 1.7B seed on two or three families.
16. **Surrogate analysis (M-M6):** Fig. 4 pools accuracy and bBPB (bBPB drives the level; accuracy alone ρ≈0.28--0.39); report accuracy-only, clustered bootstrap CIs, the noise-alone baseline, a cross-fitted choice among the 22 SNR definitions, and the DA split-half ceiling; the main text uses Spearman across tasks per size and App. F uses mean within-language Pearson pooled over sizes.
17. **Single-axis surrogate figure** (`app_06_surrogates.tex:16–21`, commented out): I removed the sentence that relied on it; restore figure and sentence if wanted (L-A4).
18. **Construct validity (M-M8):** untrained-language above-chance rates for RF/LLM-RF Belebele as a surface-cue baseline; exclude letter-format bBPB or report it separately; report bBPB→accuracy (+0.03 to +0.10) as the bBPB headline.
19. **Design confounds to state (M-M9, M-M10):** batch/step/AdEMAMix-window differences of the 90M/175M rungs; scheme axis = three different interventions; K couples language count with per-language dose; L2 repetition epochs (ES 3.5 at 1.7B); replicates only for deep at 175M/600M/1B. Suggested Limitations additions: single-seed reference and its measured ceiling, per-axis pair counts, WSD intermediate checkpoints, the 600M depth mismatch, 1.7B as largest reference, 100N training only.
20. **DA for the suite-level score** (macro per language / over languages; BPB macro 0.95--0.98) is likely the most decision-relevant number and is not reported (M-M10).
21. **Near-tie sensitivity (App. E.2)**: Table 9 and `da_sensitivity.png` / `da_categories.png` have no generator or CSV in the repo, and ε in percentage points has no meaning for bBPB. Either commit the script and CSV, or replace E.2 with `tables/app_rq02_definition_sensitivity.tex` (seed-SD based) (C-(A)12, M minor 10). Not edited beyond terminology.

### Counts, tables and definitions still pending (need the author or a regeneration)
22. **938 vs 933 analysed tasks.** The 34%/38% shares and the 3B paragraph use the gate population of 933 (313/933 = 33.5%; 313/938 = 33.4% would round to 33%). `\ntasksanalysed` (938) counts five more tasks; explain or switch (C-(A)9).
23. **Benchmark counts**: abstract 20 families vs §3 52 benchmarks vs Table 6 header "Analyzed (40 benchmarks, 695 tasks)" with 34 printed rows (rows were merged by hand) vs screened "26 / 485" vs `\nbenchscreened` 43 / `\ntasksscreened` 550; 1,488 − 229 ≠ 938. Needs one definition stated once (with/without RF and language-specific benchmarks) (L-A3, C-(A)9).
24. **"5 to 86 of 105 languages for Belebele"** (`04:90`, `app_04:71`): Table 6 gives 94 harness / 49 trained languages; 105 matches neither and may include untrained languages (contradicts the trained-languages rule). Clarify the denominator (L-A3, M minor 4).
25. **Table 7 (thresholds)**: the repo version marks CulturalBench-easy/-hard and INCLUDE v2 RF (EN) with a dagger that the Overleaf copy drops, and "(EN)/(OG)" is never explained; meaning of both not determinable from the repo text. Re-copy with a caption note (C-(B) 04:143, L-B names).
26. **INCLUDE v2** is cited to the INCLUDE paper, which does not describe it; needs either a `@misc` for the `include-results/include-128` release (check the org page for author names first) or a relabel plus a caption note (CR-A4). No verified entry exists.
27. **Missing citations without a verified entry**: Wilcoxon 1945 (`app_05_da_analyses:20`), SciPy (Virtanen et al. 2020, `app_06_snr` discrepancy). Pending.
28. **Related Work**: one sentence on DataDecide (`magnusson_datadecide_2025`) and Bhagia et al. 2025, the closest prior work (L-D). Adds main-text length.
29. **OLMES list co-cite** `gu_olmes_2025` (CR-C14) no longer applies (sentence removed). Arabic Open LLM Leaderboard / FrenchBench / IberoBench inline cites in App. C are optional (CR-C16; cited in Table 6).
30. **App. E.6 language transfer (Fig. 15) and the majority-class paragraph** are never referenced from the main text; cite them or cut (L unused labels). E.6 is one benchmark, one list, four languages: present it as illustrative (M minor 17).
31. **"Ladder" means both the grid and a recipe** (Table 2 column "Ladder"); renaming to "recipe" touches generated tables (L-B).
32. **Fig. 3 legend "data scheme"** → "language scheme", and figure-internal lowercase benchmark names ("hellaswag", "bpb") need figure regeneration (L-B).
33. **Table 4 header "Fam."** sits next to "Family" with a different meaning; "#Bench." was suggested (L-C2). Left to avoid widening a table that is already 3pt overfull.
34. **AI-style wording in the main text** left as written (main-text rule): `main.tex:193` guidance sentence; `01:12`; `02:5`; `04:133` "Notably…"; `04:214` "Nevertheless…"; `04:229` "Overall,"; `04:232`; `04:240`; `04:302`; `04:341–342, 344`; `05:11` "challenge the assumption"; `05:13` "suggests the issue may be more prevalent than currently believed" (no evidence; M-M10 also flags the overreach); `06:11` "However, the controlled model ladder allows us…". Rewrites are in L-E.
35. **Other main-text wording not applied** (correct as written): `01:29` "multilingual language scaling ladder … architecture design axes"; DCLMP/FWEB as additional English corpora in §3 (C-(B) 03:17); "multilingual models" for K=1 runs (now just "models" in the abstract); "most configurations use a single seed" could add the replicate count.
36. **Minor appendix items not applied**: give task counts (81--110) next to the reliable-task accuracy trend in E.3; "pp" vs "percentage points" in E.2; ε for bBPB in E.2 (unknown how bBPB is scaled); report the McNemar/length-cue statistics per language and the reformulation error rate (M minor 18); token thresholds are often bounds (≤/>) from extrapolated logistic fits, worth a caption sentence (M minor 8); tokenizer-token vs byte thresholds (M minor 7); the per-run Wilson false-pass floor (M minor 6).
37. **macros with trailing spaces** (`main.tex:78–101`) are safe today but break before punctuation; switch to `\xspace` (L-C3). Not changed because the block is generated.
