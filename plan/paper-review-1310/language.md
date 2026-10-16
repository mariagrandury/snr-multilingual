# Language, terminology and reference review: `13-10-12h-full-overleaf` (2026-10-13)

Scope: `main.tex` plus the files it `\input`s: `sections/01_introduction`, `02_related_work`,
`03_methodology`, `04_analysis`, `05_conclusions`, `06_statements`, `app_01_model_ladder`,
`app_02_data_mixtures` (+ `app_02_table_languages`), `app_03_evaluation` (+ `app_03_table_benchmarks`),
`app_04_chance_threshold` (+ `tables/app_rq00_thresholds`), `app_05_ranking_agreement`,
`app_05_da_sensitivity`, `app_05_da_analyses`, `app_06_snr` and `app_06_surrogates`. Text inside `%`
comments was ignored. The rendered numbering comes from `build-1310/main.pdf` (43 pages). The figures
`rq0`–`rq3.png` were opened and compared with the text. No paper file was edited.

Paths are relative to `documents/overleaf/13-10-12h-full-overleaf/`. Line numbers are source lines.
PDF numbering: Fig 1 = rq0, Fig 2 = rq1, Fig 3 = rq2, Fig 4 = rq3; Tables 1–3 = App. A, Table 4 =
languages, Table 5 = evaluation notation, Table 6 = benchmarks, Table 7 = token thresholds,
Table 8 = RA settings, Table 9 = DA tolerance. Appendices: A Model Ladder, B Data Mixtures,
C Evaluation, D Measurability, E Agreement, F Anticipation.

---

## (A) Blocking issues

These would embarrass the paper at submission: anonymity leaks, page limits, references or numbers
that contradict the content, and broken text.

### A1. Anonymity leak: the guide footnote
`sections/01_introduction.tex:38`: `\footnote{\url{https://scaling-reliability.netlify.app}}`.
The URL is live (HTTP 200). It serves the Slidev deck: the `<title>` is "Signal-Aware Framework for
Multilingual LM Evaluation - Slidev". Its bundle `/assets/index-cIatT3qN.js` (the same file the earlier
review flagged under the old `snr-multilingual` domain) contains "Grandury", "Romanou", "Hasler",
"Meister", "Bosselut" and "EPFL", three times each. The page is also not the "guide" that the bullet
promises. **Fix:** remove the footnote for review. Otherwise point it to an anonymized mirror whose
built JS has no names, EPFL, swiss-ai or W&B/HF org links. Also reconcile it with footnote 1 on line 37
("All artifacts will be publicly released upon publication"): the second footnote says the guide is
already public.

### A2. Over the 8-page limit
In the PDF, Section 7 Conclusion starts at the bottom of page 8 and runs onto page 9 (PDF lines
626–653). The Conclusion counts toward the 8 pages of an ARR long paper. **Fix:** cut about 30 lines
from §4–§6. Candidates: the duplicated reformulation detail in §4.2, the §5 intro paragraph, which
restates Table 8, and the §5.4 paragraph, which only points to the appendix.

### A3. Numbers in the text that contradict the figures and appendix they cite
| file:line | Current text | Problem | Fix |
|---|---|---|---|
| `04_analysis.tex:40` vs `:90` and `app_04_chance_threshold.tex:71` | "for which at most three languages subsets pass the filter" | Belebele passes in 5 languages at 1.7B (line 90 and App. D.2: "against 5, 1 and 3"). | "for which at most five language subsets pass the threshold" |
| `04_analysis.tex:90`, `app_04:71` | "from 5 to 86 of 105 languages for Belebele" | Table 6 gives Belebele 94 harness languages and 49 trained. Table 7 gives 48 non-English. "105" matches neither. | State what the 105 counts (tasks? language variants?), or use the trained-language denominator. |
| `04_analysis.tex:124` | "reaching 1.00 for the BPB across 50 languages" | Fig 2b shows "bpb (30 tasks)". The commented line 127 still says "30 languages trained at two or more language settings". | "…for the BPB of the 30 languages trained at two or more language settings" |
| `04_analysis.tex:124–125, 136` | Median R² 0.87, reformulated versions 0.84–0.92, "same pattern holds for monolingual English models (Appendix D.4)" | App. D.4 (`app_04:141–151`) reports none of these numbers and does not test the "no training-only predictable task" pattern on English. It only says 17/39 tasks are predictable on both axes. | Add the numbers to D.4, or point to where they are. |
| `04_analysis.tex:126` | "TruthfulQA has a median model size-fit $R^2$ of 0.97 but $\rho=-0.75$, indicating a strong and consistent decrease in **predictability** with model size" | ρ is the direction of the **score**, not of predictability. TruthfulQA is not in Fig 2a, and the language is not given (the commented original says "the Spanish TruthfulQA task"). | "…the Spanish TruthfulQA (MC2) task has … indicating that accuracy decreases consistently with model size." |
| `04_analysis.tex:229` | "agreement remains modest (59--61\% across all pairs)" | The black "all pairs" line in Fig 3a runs from about 0.60 to 0.64 (0.64, 0.61, 0.63, 0.60, 0.64). | "(60--64\%)" |
| `04_analysis.tex:270` (Summary box) | "On accuracy-scored benchmarks … close to pairwise chance: 0.51-0.56 at every proxy size." | No compiled figure or table gives 0.51–0.56. Fig 3 (acc+bBPB, filtered, single-axis) is 0.60–0.64. App. E.3 (multi-axis, acc+bBPB) is 0.57–0.58. Fig 17 (accuracy only, reliable tasks) is 0.56–0.69. | Quote one of the shown numbers and name its population (single-axis or multi-axis, filtered or not). |
| `04_analysis.tex:272` | "pairs that the 1.7B reference model separates by more than two seed standard deviations … 0.69 to 0.73" | No compiled analysis supports this. The seed paragraph is commented out (`app_05_da_analyses.tex:26`). | Add the analysis to App. E, or drop the sentence. |
| `04_analysis.tex:271` vs `:343` | Summary: "per-language BPB … ranging from 0.74 to 0.94". §6: the 600M BPB DA "remains at $0.70$--$0.74$ throughout training". | 0.70 lies below the summary's range. The two numbers come from different settings (multi-axis DA-size vs DA-target over training), and neither sentence says so. | Name the setting in both places. |
| `04_analysis.tex:331` | "reaching \(\rho=0.93\) for HellaSwag, compared with a maximum of $\rho=0.75$ across all benchmarks combined" | Fig 4 peaks at 0.62. The per-benchmark analysis that would give 0.93 is not in the compiled appendix. | Point to the per-benchmark result, or drop it. The pooled maximum in Fig 4 is 0.62. |
| `04_analysis.tex:343` | 600M BPB: "checkpoint noise of 0.4\% against 1.3\% … median relative-standard-deviation SNR of 8.7 against 2.0" | These numbers appear nowhere else and have no pointer. | Add a pointer, or add the numbers to App. F. |
| `04_analysis.tex:241` vs `app_05_da_analyses.tex:80` | "with DA of 0.51--0.52" | The appendix says "the 1.7B models themselves reach only 0.51 on accuracy". | Use one value. |
| `app_05_da_analyses.tex:78` vs `app_01_model_ladder.tex:87,93` | E.5: 3B at "scheme A at $K \in \{8, 15, 30, 50\}$ and scheme B at $K \in \{8, 15\}$" | Table 3 gives 3B at $K\in\{8,15,30\}$ for **both** A and B. The repo `CLAUDE.md` agrees with E.5: B-L30 3B was removed and A-L50 added. Table 3 also counts A deep = 57 and B deep = 23 runs, which follows the wrong split. | Regenerate Table 3 (`make_appendix_tables.py`) and check the run counts. |
| `main.tex:96` vs Table 3 | "213 models" (`\nmodels`) in the abstract, intro, §3, App. A and the conclusion | Table 3 totals **218** runs (`\nruns` = 218). | Use one number. If 5 runs are excluded, say so in Table 3. |
| `app_03_evaluation.tex:52–58`, `app_03_table_benchmarks.tex:10,52`, abstract | 95 benchmarks considered, 1,488 tasks; 229 discarded; 52 evaluated; 938 selected. Table 6: "Analysed (40 benchmarks, 695 tasks)", "Screened (26 benchmarks, 485 tasks)". Abstract: "20 multilingual benchmark families". | 1,488 − 229 = 1,259 ≠ 938. Table 6 lists 34 analysed rows, not 40. 34 + 26 = 60 ≠ 95, yet the caption says "Every benchmark considered". `\nbenchscreened` (43/550) is defined and never used. | Reconcile the counts: families, benchmarks incl. language-specific, RF variants. Give each number its unit once in §3. |
| `app_03_evaluation.tex:54` vs `:58` | Discard rule: "stayed at chance at both 1B and 1.7B" vs "final checkpoints of the 600M, 1B, 1.7B and 3B models did not achieve above-chance performance" | Two different screening criteria. | Use one. |
| `app_03_table_benchmarks.tex:11,15,21` | ACP-Bench Langs 1 / Trained 2; BBH 1 / 2; CulturalBench 8 / 16 | "Trained" cannot exceed "Langs" under the caption's definitions. The earlier review flagged this and it is not fixed. | Regenerate the table. These are probably RF rows double-counted. |
| `app_03_evaluation.tex:66` vs Table 4 "Fam." | "English is covered by 31 benchmarks"; "Mandarin Chinese (17)"; "Hindi, Italian and Vietnamese (13 each)"; "Standard Latvian has Belebele and MultiBLiMP" | Table 4 gives English 25, Chinese 16, Italian 12 and Latvian 1. | Count on one basis, with or without RF and language-specific benchmarks, and say which. |
| `app_04_chance_threshold.tex:97` | "ten of the 30 benchmark formats never reach 50\%" | Table 7 has 28 rows. Seven of them never reach 50% at any size: Belebele, BLEnD, CulturalBench-easy RF, CulturalBench-hard, Global-MMLU, INCLUDE and TruthfulQA. | "seven of the 28" |

### A4. Cross-references whose target does not show what the sentence claims
- `app_06_snr.tex:105`: "\S\ref{sec:rq3} reports every intervention's effect against the seed noise."
  §6 does not. Seed noise is mentioned only in the §5 summary box, and that claim has no support
  (see A3). Fix: delete the clause.
- `04_analysis.tex:254`: "These additional analyses are presented in Appendices E.2, E.3, and
  \ref{app:rq02_da_vs_train_tokens}". E.4 has a setup paragraph and Fig 18, but it reports no result,
  because its results paragraph is commented out (`app_05_da_analyses.tex:49`). Fig 18
  (`fig:app_rq02_da_vs_train_tokens`) is never referenced. Fix: add one results sentence to E.4 that
  cites Fig 18.
- `04_analysis.tex:39`: "For only 38\% of tasks … 34\% … (Figure 5)". Fig 5 shows per-benchmark
  counts of languages, coloured by the smallest passing size (including 3B). It does not show the
  38%/34% shares. Fix: point to where the shares are computed, or say "per-benchmark breakdown in
  Figure 5".
- `03_methodology.tex:18`: "Appendix~\ref{app:model-ladder} provides the complete training grid and
  hyperparameters". The label is on A.1 "Shared Configuration". The grid is in A.3 and Tables 2–3.
  Fix: `\ref{app:models}`, which is currently unused.
- `06_statements.tex:3`: the reformulation is "described in the methodology". It is described in
  §4.2. Fix: "(\S\ref{subsec:rq1_factors})".
- `06_statements.tex:13`: "we report fixed-population comparisons". No fixed-population figure or
  table is in the compiled paper; the `app_rq*_fixed_*` figures exist but are not input. Fix: add one
  or drop the clause.
- `app_06_surrogates.tex:12`: "On single-axis pairs alone, all correlations are weaker, and the
  relative-standard-deviation SNR leads at every proxy size." The supporting figure is commented out
  (`% TODO?`). Fix: restore the figure or drop the sentence.
- `app_03_evaluation.tex:71`: "(Section~\ref{sec:rq1})" should be `\S\ref{subsec:rq1_factors}`.

### A5. Broken text in the rendered PDF
- `app_06_snr.tex:56–72`: "In the squared form computed by SciPy's \texttt{qmc.discrepancy}, with
  $u_a = |x_a - \tfrac{1}{2}|$," is followed by nothing, because the equation is commented out. The
  next sentence is "It is defined for accuracy only." Fix: restore the equation (the two-line
  `aligned` version on lines 60–70 fits a column), or end the sentence after "evenly spread".
- `app_06_surrogates.tex:10`: `\paragraph{The relative-spread group is the one that correlates better
  with decision accuracy.}(Figure~\ref{fig:app_rq04_surrogates}).` This renders a heading followed
  only by "(Figure 20)." Fix: "\paragraph{…} The relative-spread definitions correlate most with DA
  as a group (Figure 20)."
- `app_06_snr.tex:101`: "Proposed by \citet{heineman_signal_2025} as a proxy when **the first three**
  are too expensive". Only two noise types precede it (seed noise and total variation). Data-order
  noise is commented out, and the phrase comes from Heineman et al.'s list. The formulas on lines
  79–111 also use undefined symbols: $U(t)$, $M_i$, $n$ vs $k$, $T$ for the checkpoint count (it
  clashes with temperature $T$) and $M$ for the model set (it clashes with scheme $M$). Fix: "when
  seed noise and total variation are too expensive", and define $U(t_j)$ = score at checkpoint $j$
  with one index letter.
- `app_05_da_analyses.tex:54`: "…seen by the checkpoint.}." renders as "checkpoint.." (PDF line 6909).
- `app_04_chance_threshold.tex:141`: "…originally cloze benchmarks.  (Figures~\ref{…} and \ref{…})."
  This is a parenthetical after the full stop. Fix: move it before the full stop.

### A6. Selection on the outcome, stated in the main text (wording)
`04_analysis.tex:194–195`: "We focus on accuracy and bBPB tasks whose median DA … reaches at least
0.66 … This heuristic threshold selects tasks with higher observed ranking agreement." Fig 3 then
reports DA on these tasks without saying that the selection inflates it. The appendix says so
(`app_05_ranking_agreement.tex:47`: "a result on these tasks is conditional on that selection"), but
the cross-fit control is commented out (`app_05_da_analyses.tex:22`). A reviewer will read Fig 3's
60–64% as optimistic. Fix: restore the one-line cross-fit result ("selection adds 0.01–0.02") to E.3
and cite it from §5, or state the conditionality in the Fig 3 caption.

### A7. Fig 3 and Fig 4 use names and units the main text never defines
Fig 3's y-axes read DA-size, DA-ckpt and DA-target. Its x-axis is "Proxy training tokens" in
1C–5C. Fig 4's colour bar reads "Spearman ρ with DA-size". The main text names the settings only as
"Scale / Within-model training / Target agreement" and gives training progress in % (`04:213`,
`03:78`). 1C is defined only in Table 1's caption (App. A). Fix: in §5's intro (`04:186`), write
"Scale agreement (DA-size)…", and add "(1C = the 20N Chinchilla budget, 20\% of a run)" to the Fig 3
caption. Catherine's commented note at `04:105` raises the same point.

### Not blocking, but check before submission
- No compiled `\TODO`, `\clara`, `\AR`, `\MA`, `\CL`, `\checknum`, `\citeneeded`, PENDING or red text
  survives. The macros are still defined in `main.tex:55–63`, which is harmless.
- `main.tex:164–181`: the real author block is hidden by `[review]`, so it must stay in review mode.
  Strip the comments before any source upload (`% Catherine:`, `\clara{}`, `\AR{Fixed}`, `% MARIA`,
  `% TODO: resolve appendix reference` at `04_analysis.tex:241`).
- `app_01_model_ladder.tex:130`: the footnote `https://huggingface.co/swiss-ai/Apertus-70B-2509`,
  together with "nodes of four NVIDIA GH200 GPUs" (`:213`), points toward the authors' ecosystem. This
  is allowed, but the citation `apertus_apertus_2025` is enough and the footnote can go.
- The `rqfinding` boxes ("Summary.") are teal tcolorboxes. The comment at `04:9` calls them "for
  internal alignment". If you keep them, keep all three the same length and make sure their numbers
  match the text (A3).
- Page 23 of the PDF holds only the bBPB equation, because the `[p]` table and the `\clearpage` at the
  start of App. D leave a mostly empty page.

---

## (B) Terminology inconsistencies

| Concept | Variants found (file:line) | Recommendation |
|---|---|---|
| Ranking-agreement settings | Main text: "Scale agreement", "Within-model training agreement", "Target agreement" (`04:186, 103–105`). Table 8 / E.1: **RA-size, RA-ckpt, RA-target** (`app_05_ranking_agreement.tex:12,16,20,35`), also "When computing RA" (`:50`). DA names: **DA-size** (everywhere), **DA-ckpt** (`app_05_ranking_agreement:46`, `app_05_da_analyses:38,70`, `app_06_snr:15,124`, `app_06_surrogates:8`) vs **DA-checkpoint** (`app_05_da_sensitivity:40,48`). **DA-target** (`app_05_ranking_agreement:46`, Fig 3 y-axis) vs **DA-goal** (`app_05_da_analyses:38,47,54,70`; Fig 18 title; Fig 19 caption). | Use DA-size / DA-ckpt / DA-target (the brief's "DA-goal" survives only in E.3–E.5 and its file names). Drop the RA-* column of Table 8 and use DA-* throughout. Introduce the DA-* names in §5 (A7). |
| Number of languages | $K$ is used everywhere it is symbolised (no stray $L$). The main body never defines $K$, yet Fig 1's caption uses "language count $K$" (`04:60`). Prose: "number of training languages" (`03:49`), "language count" (`03:54`, `04:60`, `app_03:64`, `app_05_ranking:52`), "language setting" (`app_02:12,16,26`, `app_05_da_analyses:6`), "number of languages" (`app_05_da_analyses:70`). | Define $K$ in §3's first bullet ("$K \in \{1,…,50\}$ languages"). Use "language count $K$" in prose. |
| Language-selection scheme $M$ | §3 says "two approaches", scheme A and B (`03:51`). Table 1 gives "A, B, C" (`app_01:23`), and A.2/B add scheme C (`app_01:145`, `app_02:10`). Table 3's caption says "B and C are the alternatives at that $K$". Fig 3 legend: "data scheme". Spelling: "scheme A" / "scheme-A lists" / "Scheme A" are all used, which is fine. | In §3 add "(and a third, C, at $K\le2$)". Make the Fig 3 legend "language scheme". |
| Symbols clash | Depth: $A$ in Table 1 (`app_01:15`), "Depth ($D$)" (`app_01:154`), $D_\mathrm{deep}$ (`app_05_ranking:53`). Activation: $A_f$ (Table 1) vs "Activation ($A$)" (`app_01:163`). $D$ is also the token budget (`app_01:121`, Table 2) and the discordant pairs (`app_05_ranking:62`). $M$ is the scheme and the model set (`app_06_snr:83,109`). $T$ is the temperature and the checkpoint count (`app_06_snr:84,96`). $b_\ell$ is the language task and the byte count (`app_03:41`). $N$ is the parameter count, with "number of model parameters" (`app_04:133`, which should say non-embedding). | Depth $d$ or $\mathcal{D}$, activation $\alpha$ or $A_f$ consistently. Bytes $B_\ell$, checkpoints $J$, models $\mathcal{M}$. |
| Seeds | Table 1: "1904 (default), 313, 64" (`app_01:27`). A.3 adds 28 and 1797 (`app_01:183`). | Table 1: "1904 (default); replicates 64, 313, 28, 1797". |
| Above-chance test | "above-chance threshold" (intro:38, `04:89,143,309`, statements:13, App D), "above chance threshold" without the hyphen (`04:60`, `app_03:66`, `app_04:34`, `app_05_da_analyses:80`), "above-chance filter" / "the filter" (`04:31,39,40,47,124,153`, `app_04:133,175`), "gate" (`app_04:22,79`), "mask" (`app_04:22`), "chance test" (commented). | "above-chance threshold" and "passes the threshold" throughout. Remove "gate" and "mask". |
| Unit of analysis | "benchmark--language task" (`04:30,31,39,72,102,153,307,308,317`, `app_04:133`), "benchmark-language" with a single hyphen (`intro:29`), "benchmark–language" as a Unicode en dash (`04:212`), "language--benchmark pairs" (`intro:16`), "benchmark--language--model-size combinations" (`04:31`), "language task" (`app_03:12,52`; `app_04:10`), "benchmark task" (`app_05_ranking:76`; `app_05_da_analyses:6`), "(task, language) cell" (commented). | "benchmark--language task" in the main text. "language task $b_\ell$" is fine once defined in Table 5. |
| "cell" (should not appear) | `app_04_chance_threshold.tex:104` ("(language, $K$, checkpoint) cells"), `:151` ("single-cell fits"), `app_05_ranking_agreement.tex:45` ("A cell $(b_\ell,N,f)$ is reported only if…"), `app_06_snr.tex:105` ("the cells trained with three seeds"). The Fig 4 caption "Cells report…" (`04:318`) refers to heatmap cells and is fine. | "evaluations", "fits on one language setting", "A value $(b_\ell,N,f)$", "the configurations trained with three seeds". |
| What is ranked | "model variants" (`03:5,6`; `04:185,193,252,302,330`; `app_05_da_sensitivity:7,51`; `app_05_ranking:26,35`; `app_06_snr:25`), "design variants" (`app_05_da_analyses:31,63`; `app_05_ranking:58`; `app_06_surrogates:8`), "variant pairs" (`04:270,272`), "design pairs" (`app_05_da_analyses:6,8,18,47`), "pairs of model configurations" (`app_05_ranking:37,62`), "training configuration" (`03:15`; `04:23,25,27,30,31,53,89,154`; `app_04:133`), "configurations" (`app_05_da_analyses:78`). | "model variant" for the object ranked, "variant pair" for the pair, "training configuration" only in §4 (measurability). |
| "evaluation configuration" | Defined in the intro (`01:8`) as benchmark + format + scoring, used `01:27,38`. App C has a subsection titled "Evaluation Configuration" (`app_03:34`) that contains only BPB. | Rename the subsection "Bits-per-Byte" and drop the empty level. |
| Answer formats | §3: "multiple-choice letters, cloze, and completion" (`03:89`). App C: "MCQA-style letters, cloze, and generative continuation" (`app_03:31`), "letter / cloze / completion" (`app_03:71`). "MCQ benchmarks" (`app_04:141`). "letter-format" / "letter-based format" (`04:91`). | letter / cloze / completion everywhere. Spell "multiple-choice" once and drop "MCQ"/"MCQA". |
| Reformulation | §4.2: "programmatic reformulation", "model-based reformulation" (`04:87`). App D.2 and tables: **RF / LLM-RF** (`app_04:67`; Table 6; Table 7; Fig 17 caption). "LLM rewrite" (`app_04:73`), "twin" (`app_04:79`), "reformatting" (`intro:31,42`; `app_03:71`; App D.2 title "Task Reformatting and Reformulation"), "cloze versions" (`app_04:141`), "reformulated versions" (`04:125`). RF and LLM-RF are never defined in the main text or in Table 6's caption. | Define "(RF)" and "(LLM-RF)" in §4.2 at `04:87`. Use "reformulation" (not "reformatting") throughout, and add "RF: programmatic reformulation; LLM-RF: model-based" to the Table 6 caption. Replace "twin" with "original". |
| Ladder / rung / run | "ladder" means both the whole model ladder (`03:12`, `app_01:108`) and an architecture recipe ("deep ladder", "SwiGLU ladder", "per ladder and model size", Table 2 column "Ladder", `app_01:180`, `app_02:10`). "rung" does not occur (good). "run" vs "model": "213 models" vs "218 runs" (A3). | Call the recipe "model recipe" or "architecture variant" (the column "Ladder" → "Recipe"), and keep "ladder" for the whole grid. |
| Chinchilla multiples | 1C/5C defined in Table 1's caption and A.3 (`app_01:32,210`). Used in the Fig 3 x-axis, `app_05_da_analyses:80` ("4.5C") and the Fig 10 caption ("Chinchilla multiples"). The main text uses % (`03:78`, `04:213`) and the Summary uses "10\% checkpoint" (`04:351`). | Pick one for the main text. If the figures keep C, define it in §3. |
| Single- vs multi-axis | Defined only in App. E.1 (`app_05_ranking:51,55`). The main text never says which one Fig 3, Fig 4 and the Summary boxes use. Fig 3 is single-axis. The Fig 4 surrogate values and the E.3 numbers quoted in the §5 summary are multi-axis (`app_05_da_analyses:6`; the n-counts in Fig 4 match the multi-axis pool). "mono-axis" appears only in file names. | State "single-axis pairs" in the Fig 3 caption (it was commented out at `04:101`). State "every pair (multi-axis)" in the Fig 4 caption. Define both terms in one sentence in §5. |
| SNR variant naming | "relative-standard-deviation SNR" (`04:327,351`), "relative-standard-deviation noise" (`04:339`, which is Fig 4's "Noise alone" row), "relative-spread SNR definitions" (`app_06_snr:15`, `app_06_surrogates:10`), "SNR, relative std." (Fig 4), "relative standard deviation" signal $S_\mathrm{rel}$ (`app_06_snr:52`). | `04:339`: "noise alone reaches ρ=0.56 at 350M". Use one name for the SNR row. |
| Proxy/reference point | "proxy point", "reference point" are consistent. "proxy checkpoint" once (`app_05_da_analyses:80`), fine. "reference scale/size" also appears, acceptable. | OK. |
| BPB / bBPB | "bits-per-byte (BPB)" is defined twice (`03:83`; `app_03:31,36`). "benchmark bits-per-byte (bBPB)" is defined twice (`03:90`; `app_03:31,86`). "per-language BPB" vs "BPB" vs "the BPB of the 50 trained languages" is consistent enough. | Fine, but the definitions in App C repeat §3. |
| Benchmark names | "TruthfulQA (MC2)" vs "TruthfulQA (mc2)" (`app_04:175`) vs bare "TruthfulQA" (`04:126`). "Global PIQA" vs "Global-MMLU" (both are the official spellings, OK). "CulturalBench" (Table 6) vs "CulturalBench-easy/-hard" (`app_04:52`, Table 7). "INCLUDE v2 RF (EN)" / "(OG)" (Table 7) are never explained, and Table 6 lists only "INCLUDE v2". Benchmark names are lowercase inside Figs 2 and 4 ("hellaswag", "bpb"). "DCLM-Edu" (`03:17`, `app_01:128`, `app_02:10`) vs "DCLM-edu" (`app_01:145`). "FineWeb-2" vs "FineWeb2" (Table 4 caption, `app_02_table_languages:122`). "AdEMAMix" is consistent. | Normalize the names and explain (EN)/(OG) in the Table 7 caption. |
| "evaluation" (as a count unit) | §4.1 reports shares "of tasks" (`04:48`). App D.3 reports shares "of evaluations" (`app_04:95`) and "configurations" (`04:53`) for the same numbers (42%, 64%, 43% vs 62%). | Use "evaluations" in §4.1, or say "(task, $K$, checkpoint) evaluations". |
| Section references | Mix of `\S\ref`, "Section \ref" without a tilde (`04:33` ×3), "Section~\ref" (`04:186` ×3, `:193`; `app_03:71`) and "Appendix~". | `\S` in running text everywhere, or `\Cref`. |
| Paragraph headings | Some end with a period ("Intervention axes."), some do not (`02:3,7` "Benchmark Reliability", "Multilingual Evaluation"; `app_03:84,86` "Accuracy", "Benchmark bits-per-byte (bBPB)"). `app_04:147` puts the period outside the braces. Case is mixed: sentence-case findings vs Title Case. `app_04:88` subsection "Above chance against the training tokens of the language" is sentence case, while every other subsection title is Title Case. | Normalize. |

---

## (C) Typos, grammar and formatting

### C1. Spelling (US vs UK)
The paper mostly uses **-ize** (normalized ×5, prioritizes ×3, initialization ×2, optimization,
generalize, stabilize ×2, characterize ×2, categorize ×2, minimize, visualization, summarizes,
tokenization). For the other word families it mostly uses UK forms. I recommend US throughout, as is
standard for ACL. The deviations from US are:
- behaviour: `01:13`, `03:85`, `app_03:39`, `app_04:133` (but "Behavior" in the `04:111` heading)
- modelling: abstract `main.tex:190`, `01:12`, Table 6 "Language modelling" (but "modeling" in `03:16`)
- analyse / analysed / Analysed: `04:39,153`; `app_04:34,43`; Table 6 header and caption
- prioritises: `app_02:6,8` (but "prioritizes" in `03:51`, `app_01:144`)
- normalised: `app_03:75,84` (but "normalized" in `03:84,91`, `app_03:38,88`)
- optimiser: `app_01:180` (but "optimizer" everywhere else)
- Generalisation: `app_05_da_analyses:75`
- summarised: `04:187`
- favour: `05:13`, `06:24`
- colour / coloured / Colour: `app_04:34,158,166`; `app_05_da_analyses:14`; `app_05_ranking:76`
- centred: `app_05_da_analyses:14`, `app_06_snr:56`
- labelled: `app_04:52`
- towards: `app_02:16`, `app_03:66`

### C2. Grammar and wording
| file:line | Current | Fix |
|---|---|---|
| `main.tex:189` | "In this work, we study this question for multilingual LLMs, and pretrain a controlled ladder" | "…for multilingual LLMs. We pretrain a controlled ladder" |
| `main.tex:190` | "evaluate language modelling and downstream performance on \nbenchmarks multilingual benchmark families" | "evaluate held-out language modelling and downstream performance on N benchmark families" (BPB is not a benchmark family) |
| `main.tex:192` | "the training tokens seen per-language" | "the training tokens seen per language" |
| `01:11` | "While decision reliability has been previously studied \citep{magnusson}, these works only conducted experiments in English settings." | "Decision reliability has been studied before \citep{…}, but only in English." (the citation is to one work) |
| `01:12` | "factors that influence conclusions of decision reliability" | "factors that affect conclusions about decision reliability" |
| `01:13` vs `02:7` | The same sentence twice: "cross-lingual transfer … can help or hurt others depending on model capacity and language similarity" | Keep it in one place. |
| `01:16` | "multilingual language--benchmark pairs" | "benchmark--language tasks" |
| `01:29` | "a controlled multilingual language scaling ladder, spanning … architecture design axes" | "a controlled multilingual scaling ladder spanning model sizes, language mixtures, and architecture and optimizer choices" |
| `01:29` | "We evaluate \nmodels models on up to \ntasksconsidered benchmark-language tasks" | 1,488 is the considered set and 938 were analysed (A3). Use the analysed number. |
| `01:31` | "makes Belebele, Global-MMLU and INCLUDE scores measurable where accuracy when using their original letter formats stays at chance" | "makes Belebele, Global-MMLU and INCLUDE measurable at sizes where their original letter format stays at chance" |
| `01:32` | "statistics … used to predict ranking agreement in monolingual settings, do not transfer to our multilingual ladder" | §6 reports ρ ≈ 0.55–0.62, so "transfer only partially: none predicts ranking agreement consistently" |
| `03:14` | "between 90M to 1B parameters" | "from 90M to 1B parameters" |
| `03:60`, `app_01:154` | "We compare a deep with a shallow architecture" | "We compare a deep and a shallow(-wide) architecture" |
| `04:23` | "the measurability of a training configuration on a task, which determines whether a score is …" | "…, that is, whether its score is …" |
| `04:31` | "a model-size passes"; "combinations that don't pass" | "a model size passes"; "do not" |
| `04:48` | "score above chance  on 42\%" (double space) | single space |
| `04:126` | "median model size-fit $R^2$" | "median model-size-fit $R^2$" |
| `04:153` | "A large share of the suite does not perform measurably on these benchmarks." | "A large share of the suite is not measurable at these scales." |
| `04:192` | "since comparisons between only two pairs yield binary agreement scores" | With two pairs DA ∈ {0, 0.5, 1}; with one pair it is binary. Write "since fewer pairs give only coarse (0, 0.5 or 1) scores". |
| `04:212` | "benchmark–language" (Unicode en dash) | `benchmark--language` |
| `04:252` | "near-tied model scores (i.e., model variants with very close rankings)" | "(\ie, variants whose reference scores are nearly equal)". Use the `\ie` macro. |
| `04:253` | "Accounting for near-tied scores as expected increases DA" | "As expected, accounting for near ties increases DA" |
| `04:270` | "0.51-0.56" (hyphen) | `0.51--0.56` |
| `04:271–272` | "from 0.74 to 0.94", "0.69 to 0.73" | `0.74--0.94`, `0.69--0.73` (the paper uses en-dash ranges elsewhere) |
| `04:304` | "In this analysis section, we investigate" | "We investigate" |
| `04:319` | "ranking agreement at the 1.7B." | "…with the 1.7B reference." |
| `04:350` | "No metrics computed from proxy evaluations predicts DA" | "No metric computed from proxy evaluations predicts DA" |
| `04:73`, `04:74` | Fig 2 caption ends without a period | add one |
| `05:13` | "two questions:  how" (double space) | single space |
| `06:18` | "Ethical considerations" | "Ethical Considerations" (match "AI Use Statement") |
| `app_01:117` | "All sizes are non-embedding parameters, that is, the transformer blocks alone" | "All sizes count non-embedding parameters, that is, the transformer blocks alone" |
| `app_01:117` | "the spacing of the smallest models sizes" | "model sizes" |
| `app_01:165` | "constant.  This width" (double space) | single space |
| `app_01:180` | "a data build specifies the training grid data" | "a data build specifies the training data" |
| `app_01:213–214` | "using 3 to 21 nodes per run"; "\npretrainnodehours node hours" | `3--21`; "node-hours" |
| `app_02:6` | "The lists are nested, \ie the $K=2$ list" | "The lists are nested: the $K=2$ list …" |
| `app_02:8` | "against the 10 and 10 from scheme A" | "against 10 scripts and 10 families for scheme A" |
| Table 4 caption (`app_02_table_languages:122`) | "Training languages of the model ladder" (half the rows are validation-only); ``` ``val" ``` (wrong closing quote; renders as “val")); "are in no training mixture, they are only included" (comma splice); "FineWeb2" | "Training and validation languages"; ``` ``val'' ```; "…in no training mixture; they are only in the 100-language validation set"; "FineWeb-2". The column header "Fam." sits next to "Family" with a different meaning: rename it "#Bench." |
| `app_03:41` | Leading space; $b_\ell$ "the number of bytes" | Rename to $B_\ell$ (B row). |
| `app_03:39` and `:46` | BPB "connecting our analysis to established loss-based scaling laws" and again "BPB connects the analysis to loss-based scaling laws" | Delete line 46's repeat. |
| `app_03:53` and `:66` | The sentence "To try to increase … we screened language-specific benchmarks like the Arabic Open LLM Leaderboard, FrenchBench, and IberoBench" appears twice. The second copy is missing "and". | Keep one. |
| `app_03:54` | "the evaluation of 12 checkpoints per model size" | "per run" |
| `app_03:66` | "didn't pass the above chance threshold … and weren't included" | "did not pass the above-chance threshold … and were not included" |
| `app_03:73` | "We use the LM Evaluation Harness … and use the original prompt" | "…, with the original prompts and few-shot examples." |
| `app_03:89` | "We propose it as measurement where benchmark accuracy is not above chance." | "We propose it as a measurement for tasks whose accuracy is at chance." |
| `app_03:90` | "the Harness" | "the harness" |
| Table 6 caption (`app_03_table_benchmarks:81`) | "Langs is the number" (missing period); ``` ``letter" ```, ``` ``cloze" ```, ``` ``completion" ``` (wrong closing quotes); "Mid-Resource Language Examples lists …" (header says "Mid-Resource Langs."); "among ranks 30 to 50 … from the least resourced up" (contradictory: ranks 30–50 of a resource-ranked list are the *least* resourced, so "from the least resourced up" makes no sense) | "Langs.\ is …"; `''`; match the header; "the benchmark's languages among ranks 30–50 (the lowest-resourced) of our 50-language list (ISO 639-1)". Also define RF/LLM-RF here. |
| `app_04:22` | "The resulting mask is the gate that every analysis reads." | "Every later analysis uses only the (task, size) pairs that pass." |
| `app_04:27` vs Table 7 "Margin" | Margins given in percentage points ("9.00 … 2.44") in the text vs fractions (0.024) in Table 7 | One unit. |
| `app_04:52` | "Applied to the labelled tasks" | "Applied to the letter-format and labelled-option tasks" (say which) |
| `app_04:79` | Fig 7 title "across the gate"; "twin"; "p $<$ 0.05" | "above the threshold"; "original"; "$p<0.05$" |
| `app_04:95` | Title "rises from 100M tokens of the language" vs Fig 8 title "rises … from 10M on" vs body "stays near 40\% up to 100M … 52\% at 100M"; "43\% at $10^{5}$ … 42\% at 1M" (mixed notation); "(Figure~\ref{…}.)" | Make the three statements agree; "100K"; "(Figure 8)." |
| `app_04:133` | "$N$ denotes the number of model parameters" | "non-embedding parameters" |
| `app_04:147` | `\paragraph{…predictably}.` | Put the period inside the braces. |
| Fig 9, 10 captions (`app_04:158,166`) | "one line per (…) tuple, the colour indicates $K$" (comma splice) | "…tuple; colour indicates $K$" |
| `app_05_ranking:46` | "we call DA-size, DA-ckpt and DA-target for decision accuracy in …" | "we write DA-size, DA-ckpt and DA-target for decision accuracy in …" |
| `app_05_ranking:53` | "($\theta' = …$)." stands alone as a sentence | Attach it to the previous sentence with "e.g.," |
| `app_05_ranking:57` | "every table is also computed over the multi-axis pairs" | Which tables? Name them or drop the clause. |
| `app_05_ranking:58`, `:66` | Comma splices: "…as design variants, every ranking compares…"; "…of the $V$ variants, it is the Pearson…" | semicolons |
| `app_05_ranking:65` | "the number of the pairs tied" | "the numbers of pairs tied" |
| `app_05_ranking:76` | Fig 12 caption: "Spearman's correlation over the points" | "Spearman's $\rho$" |
| `app_05_da_sensitivity:8` vs Table 9 | ε given in percentage points while the analysed tasks include bBPB (no pp) | Say ε applies to accuracy tasks only, or how bBPB is scaled. |
| `app_05_da_analyses:8` | "Accuracy and bBPB predictably rank the design pairs close to chance." | "Accuracy and bBPB rank the design pairs close to chance." |
| `app_05_da_analyses:18` | "bBPB predictably ranks design pairs where accuracy is at chance." | "bBPB still ranks design pairs where accuracy is at chance." |
| `app_05_da_analyses:20` | "bBPB predicts the reference's accuracy better than accuracy." | "…better than the proxy's own accuracy does." |
| `app_05_da_analyses:14` | "Colour centred at 0.5." (fragment) | "The colour scale is centred at 0.5." |
| `app_05_da_analyses:54` | "…checkpoint.}. One line per proxy size and one marker per evaluated tenth, bars give…" | Remove the extra period; "…tenth; bars give…" |
| `app_05_da_analyses:80` | "On the six configurations, it raises the number of tasks" (no antecedent); "as against 1.7B" | "the 3B reference raises …"; "the same as against 1.7B" |
| `app_06_snr:9` | "defines the surrogates … and analyses of \S6" | "…and details the analyses of \S6" |
| `app_06_snr:40` | Fig 21 title and body comma splices: "…in Figure 4, on accuracy alone …"; "…for accuracy only, the scaling-fit $R^2$ …" | semicolons |
| `app_06_snr:52` | "three candidate signal definitions to capture a different notion" | "three signal definitions, each capturing a different notion" |
| `app_06_snr:101` | "…as a proxy when … and depends only on the final $n$ training checkpoints" (no subject; $n$ vs $k$ in the formula) | "It depends only on the final $k$ checkpoints." |

### C3. Number and unit formatting
- DA is reported as percentages in §5.1–5.3 (`04:213,229,239`: "62--68\%", "59--61\%", "72--75\%",
  "58--64\%") but as fractions in the Summary boxes, §5.3 (`04:241`) and every appendix. Use fractions
  (0.62–0.68) throughout.
- Ranges: `--` is the norm. Deviations are a hyphen at `04:270`, "to" at `04:271,272` and
  `app_01:156` ("0.4\% to 5.2\%"), and "3 to 21" at `app_01:213`.
- Token counts: "100M", "$10^{5}$", "$10^9$ = 1B" (Fig 1 caption), "8$\times$1M" (commented). Use the
  K/M/B suffixes everywhere.
- "percentage points" and "pp" are both used in App. E.2. Pick one.
- `app_01:214`: "node hours". Elsewhere "node-hours" (commented).
- Thousands separators and the `1{,}172` style are consistent.
- Macros with a trailing space (`main.tex:78–101`): every use is followed by a word, so none renders
  "213 ." — but a future use before punctuation will break. Remove the trailing spaces and use
  `\xspace`.

---

## (D) Missing citations

Earlier items that are still missing are marked [prev Cn] (see F). No `\citeneeded` is compiled.

| file:line | Claim / item | Add |
|---|---|---|
| `04:25`, `app_04:12` | "one-sided 95\% Wilson lower confidence bound" | `wilson_probable_1927` (in custom.bib, unused) [prev C1] |
| `app_04:69,71,79` | McNemar test | `mcnemar_note_1947` (in bib, unused) [prev C2] |
| `app_05_da_analyses:20` | "Wilcoxon $p < 10^{-7}$" | Wilcoxon 1945 (new) |
| `04:49`, `app_01:122`, Table 1/2 captions | "Chinchilla budget", "$20N$ Chinchilla token-budget convention" | `hoffmann_empirical_2022` (in bib, unused; the cite is commented out at `app_01:122`) [prev C3] |
| `04:119`, `04:308`, `app_05_ranking:63–66` | Spearman's ρ, Kendall's τ_a/τ_b | `kendall_new_1938` (in bib) and Spearman 1904 (new; not `spearman_correlation_1910`) [prev C4] |
| `app_01:98` (Table 3 FWEB), `app_02:10` | FineWeb | Penedo et al. 2024 (new) [prev C5] |
| Table 6 rows ARC (MT), ARC (Okapi), HellaSwag (Okapi), MMLU (Okapi), TruthfulQA (Okapi); main text `04:40,50` | Source benchmarks HellaSwag and ARC are never cited | `zellers_hellaswag_2019` (new), `clark_think_2018` (in bib), `hendrycks_measuring_2021`, `lin_truthfulqa_2022` [prev A5/C6] |
| `04:40,50,90,125`; `03:83` | Belebele, Global-MMLU, INCLUDE, HellaSwag, MultiBLiMP, TokSuite and TruthfulQA are named in the main text without citations (they are cited only in App. Table 6) | Cite at first mention in §4.1 (the keys already exist). |
| Table 6 row LAMBADA (`app_03_table_benchmarks:30`), `app_03:71` | LAMBADA cited to the harness | `paperno_lambada_2016` (+ `radford_language_2019`), both in bib, unused [prev A1] |
| `app_01:130` | "Llama-3-style scaling factor 8" | Grattafiori et al. 2024 (new) [prev C7] |
| `app_01:161`, `03:65` (commented) | AdamW | Loshchilov & Hutter 2019 (new) [prev C8] |
| `03:57`, `app_01:150`, `app_02:16` | Temperature sampling $q_i \propto p_i^{1/T}$ | Arivazhagan et al. 2019 and/or Conneau et al. 2020 (new) [prev C9] |
| `04:83` | Letter answers "require mapping answer labels to their content" | add `robinson_leveraging_2023` (fix its author list) and `wiegreffe_answer_2025` (both in bib) [prev C11] |
| `02:3` | "small, informative subsets can often recover the ranking of the full benchmark" | Polo et al. 2024 tinyBenchmarks (new) [prev C12] |
| `app_06_snr:56` | "centred $L_2$ discrepancy" and "SciPy's qmc.discrepancy" | Hickernell 1998 (new) [prev C13]; Virtanen et al. 2020 for SciPy (new) |
| `03:17` (main text) | "DCLM-Edu and FineWeb-2 … warmup-stable-decay" are uncited in the main text (cited only in App. A) | `allal_smollm2_2025`, `penedo_fineweb2_2025`, `hu_minicpm_2024`, `hagele_scaling_2024` [prev C15] |
| `app_01:133` | WSD is cited only to Hägele | add `hu_minicpm_2024` [prev A6, 3rd bullet] |
| `app_03:53/66` | Arabic Open LLM Leaderboard, FrenchBench, IberoBench (optional; they are cited in Table 6) | `elfilali_open_2024`, `faysse_croissantllm_2025`, `baucells_iberobench_2025` [prev C16] |
| `02` (Related Work) | DataDecide, the closest prior work (decision accuracy, small-to-large prediction), is cited in the intro and §5 but not discussed in Related Work. Downstream scaling-prediction work is also absent (Bhagia et al. 2025 is in the bib; `chen_scaling_2025` sits in a commented paragraph). | One sentence in "Benchmark Reliability" on `magnusson_datadecide_2025` and `bhagia_establishing_2025`. |
| `06:15` | "web-derived training data may not represent the full range of varieties" | Optional. The old "curse of multilinguality" sentence (prev C10) is gone, so C10 no longer applies. |

---

## (E) AI-style wording

The paper is light on LLM lingo. I found no "delve", "crucial", "pivotal", "showcase", "leverage",
"landscape", "nuanced", "it is worth noting" or "plays a key role". It has one em dash (`04:33`, which
is fine) and no "not only … but". The patterns that remain are hedge-and-restate sentences, trailing
"-ing" clauses that state a significance, and "Overall/Notably/This highlights" openers.

| file:line | Current | Proposed rewrite |
|---|---|---|
| `main.tex:193` | "We convert these findings into concrete per-benchmark guidance to support the identification of evaluations that can lead to cheaper, better-informed multilingual model development." | "We turn these findings into per-benchmark guidance on which evaluations to trust at small scale." |
| `01:12` | "multilingual language modelling introduces a host of interacting (and potentially confounding) factors that influence conclusions of decision reliability." | "multilingual pretraining adds factors that can change these conclusions." |
| `02:5` | "Together, these papers motivate our focus on identifying which multilingual measurements remain reliable across model sizes and nearby pretraining checkpoints, rather than treating all benchmark averages as equally meaningful." | "We ask which multilingual measurements stay reliable across model sizes and checkpoints." |
| `04:133` | "Notably, no accuracy or BPB task exhibits predictable scaling across training checkpoints without also scaling predictably across model sizes, although the reverse is not always true." | "Every task that is predictable along training is also predictable across sizes; the reverse does not hold." |
| `04:214` | "Nevertheless, disagreement persists even near the end of training, indicating that intermediate evaluations do not always reproduce final model-selection decisions." | "Even near the end of training, 7–17\% of decisions differ from the final ones." |
| `04:229` and `04:341` | "Overall, agreement remains modest …" / "Overall, surrogate signal emerges at small scales but does not consistently strengthen with model size." | Drop "Overall,". The second sentence repeats `04:338`, so delete it. |
| `04:232` | "These results suggest that the reliability of cross-scale decisions depends more on the type of model intervention than on the size of the proxy model alone." | "Which intervention is tested matters more than the proxy size." |
| `04:240` | "This contrast indicates that improvements in ranking agreement during training do not necessarily transfer across model scales, limiting the ability of smaller, partially trained models to predict decisions at the reference size." | "So a smaller proxy's ranking converges to its own final ranking, not to the 1.7B one." |
| `04:302` | "Prior work has proposed several surrogate metrics, most notably signal-to-noise ratio (SNR; …)" | "The best-known surrogate is the signal-to-noise ratio (SNR; …)" |
| `04:331` | "…compared with a maximum of ρ=0.75 across all benchmarks combined, demonstrating that surrogate effectiveness is benchmark-dependent." | "…against at most 0.62 pooled over benchmarks (Figure 4)." (this also fixes the number; see A3) |
| `04:342` | "These findings suggest that reliable evaluation at smaller scales does not necessarily imply consistent model rankings at larger scales." | Delete. The next sentence's example makes the point. |
| `04:344` | "This highlights the distinction between reliably measuring performance differences at the proxy scale and preserving model rankings across scales." | "Measuring a difference reliably at the proxy is not the same as predicting its sign at the reference." |
| `05:11` | "Our findings challenge the assumption that ablations at smaller scales reliably identify the best choices for larger multilingual models." | "In our ladder, small-scale ablations often do not pick the choice that is best at 1.7B." |
| `05:13` | "…but suggests the issue may be more prevalent than currently believed." | Delete. It is a flourish without evidence. |
| `06:11` | "However, the controlled model ladder allows us to study these factors systematically within the available budget." | Delete. It is defensive and adds nothing to a Limitations section. |
| `app_05_da_sensitivity:67` | "Thus, even when exact ranking recovery remains weak, larger models and later checkpoints provide a better signal of whether an intervention will matter at the reference point." | "Larger and later proxies are better at telling whether an intervention matters, even when they get its sign wrong." |
| Recurrent | "does not necessarily" ×3 (`04:126,240,342`), "generally" ×4, "substantially/substantial" ×5 (`02:3`, `04:89,123,229,329`) | Cut at least half. Each occurrence hedges a number that the sentence already gives. |

Bold-lead-in bullets: the contributions list (`01:37–42`) and the §3 intervention list (`03:49–69`)
are conventional and fine. The `\paragraph{<finding as a sentence>.}` headings in Appendices D–F are a
deliberate style. Keep them, but drop the adverb "predictably" where it is wrong (C2).

---

## (F) Status of the blocking items from `plan/citation-review.md`

| Item | Status in this version |
|---|---|
| A1 LAMBADA cited to the harness | **Not fixed** (`app_03_table_benchmarks.tex:30`). |
| A2 Arabic MMLU (OALL) cited only to OALL | **Not fixed** (`:57`). |
| A3 AlGhafa translated cited only to OALL | **Not fixed** (`:55`). |
| A4 INCLUDE v2 cited to the INCLUDE paper | **Not fixed** (`:27`). Table 7 now also has "INCLUDE v2 RF (EN)/(OG)", which is unexplained. |
| A5 HellaSwag / ARC never cited; Okapi rows lack source cites | **Not fixed.** `zellers_hellaswag_2019` is not in the bib, and `clark_think_2018` is still unused. |
| A6.1 DataDecide "design choices" overclaim (`01:11`) | **Fixed.** Now "decision reliability has been previously studied". |
| A6.2 Bhagia + Heineman ladder convention | **Fixed.** Only Bhagia is cited (`app_01:117`). |
| A6.3 WSD cited only to Hägele | **Not fixed** (`app_01:133`). |
| B1 `heineman_signal_2025` pages | **Not fixed** (still 17073–17114; should be 19286–19327). |
| B2 `li_adaptive_2025` PMLR volume/pages | **Not fixed.** |
| B3 `messmer_enhancing_2026` volume/pages/DOI | **Not fixed.** |
| B4 `magnini_leaderboard_2025` wrong eprint | **Not fixed** (eprint 2502.02289 is still there). |
| C1–C9, C11–C13, C15 missing citations | **None added** (see D). C14 (OLMES list in App F) no longer applies, because the sentence was removed. C10 no longer applies. |
| D1 Netlify footnote de-anonymizes | **Not fixed.** The URL changed to `scaling-reliability.netlify.app`, but it serves the same Slidev bundle (`index-cIatT3qN.js`) with all author names and EPFL (A1). |
| D2 author block hidden by `[review]` | OK (`main.tex:5`). |
| "Not citation" extras: `app_00_terminology` input | **Fixed** (no longer input). |
| Limitations / Ethics as bullet drafts | **Fixed** (now prose). |
| `\ref{tab:da_tolerance}` undefined | **Fixed** (Table 9). |
| Table 6 "Trained" > "Langs" (ACP-Bench, BBH, CulturalBench) | **Not fixed** (A3). |

---

## Counts

- (A) Blocking: 7 groups. A1 anonymity, A2 page limit, A3 18 number contradictions, A4 8 wrong or
  empty references, A5 5 broken-text spots, A6 selection-on-outcome wording, A7 undefined
  figure terms.
- (B) Terminology: 21 rows.
- (C) Typos and grammar: 13 UK-spelling families (C1), 67 grammar and wording items (C2), 7
  formatting rules (C3).
- (D) Missing citations: 19 rows. 14 are carried over from the earlier review, and 5 are new
  (Wilcoxon, SciPy, main-text benchmark cites, DataDecide/Bhagia in Related Work, the WSD co-cite
  detail).
- (E) AI-style: 17 rows with rewrites.
- (F) Earlier review: 17 items. 6 are fixed or no longer apply, 11 are not fixed (including D1 and
  all of B1–B4).

Unused labels (no action needed unless you want them cited): `sec:introduction`, `sec:methodology`,
`sec:sensitivity`, `sec:conclusions`, `app:models` (use it in A4), `tab:notation-models`, `eq:bpb`,
`eq:decision-accuracy`, `app:rq1_measurability`, `app:agreement`, `app:anticipation`,
`app:measuring-signal`, `app:measuring-noise`, `app:total-variation`, `app:sur-other`,
`app:monolingual-comparison`. Two appendix parts are never referenced from anywhere: **E.4's
Fig 18** (`fig:app_rq02_da_vs_train_tokens`) and **E.? "Language Transfer Example"**
(`app:rq06_language_transfer`, with Fig 15). The **majority-class chance paragraph**
(`app:rq00_majority_chance`) is also unreferenced. Either cite them from §4/§5 or cut them.
`app_04:145` attaches `\label{app:monolingual-comparison}` to a commented-out subsubsection, so it
resolves to D.4.
