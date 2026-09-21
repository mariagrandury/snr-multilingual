# Slides outline

Map of [slides.md](slides.md) — 190 slides. One line per slide:
`N. [layout] title — main idea` (the key claim, the figure's subject, or the
concept being defined). Titles are verbatim, so every line is greppable in
`slides.md`.

Regenerate the numbers after editing the deck: slides are separated by `---`,
so `grep -n '^layout:\|^title:' slides.md` is the quick check.

## Title & framing (1–4)

1. **[section]** Signal-Aware Framework for Multilingual LM Evaluation — title slide; authors, EPFL NLP.
2. **[image-right]** *(no title)* — `motivation.gif`, the cold open.
3. **[agenda]** Agenda — Introduction · Related Work · Methodology · Research questions · Experimental setup · Findings · Open discussion.
4. **[section]** Introduction — training multilingual LMs needs constant evaluation decisions, but evaluation is expensive and often uninformative.

## Motivation (5–7)

5. **[bullets]** Motivation — *Lack of benchmark reliability*: not all benchmarks give informative signal — high variance, redundancy, weak correlation with real progress, cost.
6. **[bullets]** Motivation — *Why multilingual?*: the reliability assumption is **especially fragile** multilingually — low-resource underrepresentation, near-random small models, new variability, tools validated only on English.
7. **[focus]** *(research question)* — Which (subsets of) benchmarks give reliable signal at each stage of multilingual training, and which model sizes can stand in for the large one?

## Related work (8–10)

8. **[section]** Related Work — Heineman et al. (2025), AllenAI, *Signal and Noise*.
9. **[image-left]** Signal-and-Noise Paper — *Figure 1*: examples of signal and noise.
10. **[image-left]** Signal-and-Noise Paper — *Figure 2*: SNR vs decision accuracy ($R=0.791$); + the paper's five takeaways (noise→scaling-law error, SNR-filtered subtasks, checkpoint averaging, BPB > accuracy).

## Methodology (11–18)

11. **[section]** Methodology — the Signal-Aware Framework, multilingual edition.
12. **[default]** Methodology — *Decision Accuracy*: **defines DA** (pairwise rank agreement small→large) and its four variants: DA-size, DA-ckpt, and for post-training DA-stage, DA-ctx.
13. **[focus]** Decision Accuracy is what we ultimately want a benchmark to get right — everything else is a cheap proxy computable during training.
14. **[default]** Methodology — *Signal & Noise*: **defines signal** (relative dispersion across mixtures), **noise** (relative std over the final *n* checkpoints), and SNR = signal / noise.
15. **[default]** Methodology — *Signal Definitions*: 22 candidate signal variants in 5 families (dispersion, relative-spread, discrepancy, robust, depth).
16. **[default]** Methodology — *Signal Definitions, formulas*: the 22 variants written out, two columns.
17. **[compare]** Preliminary: Benchmark Noise — **defines the two noise variants**: checkpoint noise (needs intermediate checkpoints) vs benchmark noise (k-fold on one eval run, more predictive of DA).
18. **[default]** Methodology — *Stage-specific reliability*: pretraining / midtraining / post-training each make different decisions on different models; the goal is per-stage recommendations, and these results are the pretraining stage.

## Research questions (19–20)

19. **[section]** Research questions — seven, one directory each.
20. **[default]** Research questions — RQ0–RQ5 ask which *benchmarks* are reliable, RQ6 asks which *model sizes* are; one entry point, `run_all_predictivity.sh`.

## Experimental setup (21–26)

21. **[section]** Experimental Setup — the predictivity ladder.
22. **[default]** Experimental Setup, the ladder — the grid: 6 sizes × 7 language counts at 5 × Chinchilla, interventions depth and data scheme, seeds; outcome metric is per-language BPB.
23. **[figure]** Experimental Setup — *the planned grid* (`pretrain_progress_plan.png`).
24. **[default]** Experimental Setup, what exists today — 60 of 89 cells finished; L100 empty everywhere, 1.7B empty, 51 models enter the analysis.
25. **[bullets]** Which languages we report on — all 50 trained languages, not the old 12; validation is 100 subsets over 95 languages, 45 of them zero-shot.
26. **[figure]** What exists today — *the grid as a picture* (`grid_status.png`).

## Findings (27–73)

27. **[section]** Findings — what the ladder says so far.

### F1 — scaling holds above 90M

28. **[figure]** Finding 1 — Above 90M, the ladder is a ladder (`ladder_report_scaling.png`; power-law fit per language setting).
29. **[bullets]** Finding 1 — scaling holds and α = 0.16–0.20 regardless of L; residuals ≤ 0.07 nats.
30. **[bullets]** Above 90M the ladder behaves — plain-language version of 29: languages shift the curve up, not its slope.

### F2 — the 90M rung

31. **[figure]** Finding 2 — The 90M rung is not on the ladder (`ladder_report_loss.png`; 9 of 10 90M runs peak early then degrade).
32. **[default]** Finding 2 — an optimizer timescale fixed in steps on runs that differ 18×. ⚠️ **Broken slide**: its body is raw frontmatter for the `optimizer_timescale.png` figure — see *Known deck bugs*.
33. **[default]** *(empty orphan slide, fallout from 32)*.
34. **[bullets]** Why the 90M rung broke — AdEMAMix β₃ fixed at 10,000 steps vs a 4,500-step run; tying it to run length takes final loss 5.762 → 2.778 and drift 1.365 → 0.011.
35. **[default]** *(untitled)* — the same diagnosis as tables: steps ÷ optimizer timescale per rung, and the corrected-90M beats uncorrected-175M comparison.
36. **[bullets]** Finding 3 — One 90M cell slips through the filter — `90M-L2-shallow` misses the divergence test by 0.016 nats, so it is the only 90M model in every pool and anchors 101 of RQ6's 303 fits.
37. **[bullets]** One 90M cell got through — plain-language version of 36; the loader ignores the published `run__off_trend` flag.

### F4 — more languages

38. **[figure]** Finding 4 — More languages is nearly free for English (`english_vs_rest.png`).
39. **[bullets]** Finding 4 — the multilingual tax is paid once, at the first extra language. ⚠️ **Broken slide**: body is raw frontmatter for `bpb_gain_per_language.png` — see *Known deck bugs*.
40. **[default]** *(empty orphan slide, fallout from 39)*.
41. **[figure]** English pays once, everyone else keeps gaining (`english_vs_rest.png`; dotted English vs the median of 99 others).
42. **[default]** *(untitled)* — the numbers behind 41: English +0.038 then +0.007 bits/byte; non-English median 1.650 → 1.139 at 600M; a 30× effect against seed noise.

### F5 — which axes clear the noise floor

43. **[figure]** Finding 5 — Only one of our three axes clears the noise floor (`effect_vs_seed_noise.png`; median |Δ loss| in seed-σ units).
44. **[bullets]** Finding 5 — language count 12.0× seed noise, data scheme 2.2×, depth 1.0×; depth separates per task (1.55×) but not on the headline metric.
45. **[bullets]** Two of our three axes are inside the noise — plain-language version of 44.

### F6 — the above-chance gate

46. **[figure]** Finding 6 — Emerged, or at chance (`fig1_gate.png`; one benchmark learns, one never leaves the line).
47. **[default]** Finding 6 — Three quarters of the suite is at chance: 94 of 324 gated tasks clear chance anywhere; strongly an answer-count effect (45 % of 2-option vs 13 % of 4-option).
48. **[figure]** Which benchmark works, in which language (`first_clearing_size.png`; smallest model beating chance, grey = never).

### F7 — BPB vs benchmarks

49. **[focus]** In 89 of the 95 validation languages the most reliable measurement is **bits-per-byte**, not any benchmark.
50. **[figure]** Finding 7 — Reliability is a (family × language) map (`fig3_reliability_map.png`; DA-ckpt at 1B).
51. **[default]** Finding 7 — BPB out-SNRs the benchmarks, and not narrowly: 89 languages rank a `bpb_` task first (median SNR 5.9), the 6 exceptions are de/en/es/fr/ru/zh at SNR 0.01–0.20.
52. **[figure]** Finding 7, read again (`snr_bpb_vs_benchmark.png`; BPB wins on coverage, not sharpness).
53. **[bullets]** We measure best where our choices matter least — the gate empties 87 of 95 languages; in the 8 that survive every measurement scores < 0.03. The suite covers the languages our choices do not move.
54. **[default]** RQ2 — SNR definition *(auto-generated)*: per-language top benchmark, SNR and DA-ckpt@1B for 50 languages.

### F8–F9 — does SNR predict DA, and does it transfer

55. **[bullets]** Finding 8 — SNR predicts decision accuracy weakly: best variant `rel_mpsd` r = +0.32 (DA-ckpt), +0.06 (DA-size), against R = 0.79 in Heineman et al.
56. **[bullets]** SNR predicts much less here than in English — plain-language version of 55; more evals moved the number, no new models did, so the pool is the binding constraint.
57. **[default]** Finding 9 — No SNR choice survives a seed swap cleanly: train on seeds 64/313, test on 1904; recommend a *family*, never a variant — the per-language argmax never transfers.
58. **[figure]** Does the SNR ranking survive a seed swap (`seed_holdout.png`).

### F10 — the noise definition itself

59. **[bullets]** Finding 10 — Late-checkpoint noise understates the real noise 2× — seed noise ÷ detrended checkpoint noise = 2.04 over 5,546 cells, so every SNR here is ~2× too optimistic.
60. **[bullets]** Our noise estimate is too small — plain-language version of 59.

### F11 — RQ6, the proxy-size question

61. **[default]** Finding 11 — Bits per byte answers the question, benchmarks do not: all 28 intervention comparisons resolve against 600M; 175M gets the scheme decision backwards at L15/L30, 350M is reliable.
62. **[figure]** The benchmark suite can be scored, and it fails (`benchmark_predictivity.png`; 2,227 shared benchmark tasks).
63. **[bullets]** What that means for the benchmark suite — under the strict test 0 of 2,116 benchmark comparisons pass, 369 of 500 BPB comparisons do; the two even disagree on which scheme wins.
64. **[figure]** How small a proxy can we get away with (`min_predictive_size.png`; one dot per language).
65. **[figure]** The same question, language by language (`min_predictive_per_language.png`).
66. **[bullets]** How small a proxy can we get away with — L1 depth is hardest (43/100 subsets have no working proxy); 175M suffices at L8/L15; macro BPB is *not* a shortcut; no benchmark ever qualifies.
67. **[figure]** The figure we actually want (`rq6_whiteboard.png`; smallest model reaching each DA vs language count).
68. **[figure]** All the measurement behind it (`rq6_da_vs_size.png`; two proxy sizes, one reference).
69. **[bullets]** Why the curve is not there yet — only two usable proxy sizes, nothing reaches 0.9; finish 1B with matched pairs and the figure becomes drawable.

### RQ3–RQ5

70. **[bullets]** Not run — RQ3, agreement with DataDecide: needs the AllenAI SNR table (`git lfs pull`); the shared universe is only the English tasks.
71. **[figure]** Finding 12 — A subject subset beats the full Global-MMLU at every size (`fig4_subset_sweep.png`; cumulative SNR in standalone-SNR order).
72. **[default]** RQ4 — Subtask subsets *(auto-generated)*: top full → best-subset SNR gains.
73. **[default]** RQ5 — Benchmark design *(auto-generated)*: median SNR per family against option count and format.

## Open discussion (74–82)

74. **[section]** Open discussion — seven decisions, in the order they block each other.
75. **[focus]** 0. If the suite only covers the languages our choices do not move, what is it for? — recommend BPB, or is that conceding the question?
76. **[focus]** 1. The 90M rung — retrain corrected, drop it, or report both?
77. **[focus]** 2. Sampling temperature — T = 1 is not training 100 languages; recommendation T = 2 sweep-wide.
78. **[focus]** 3. Eval walltime blocks everything above L = 30 — 1,356 min at 1.7B against a 719-min cap; this is why L100 is empty.
79. **[focus]** 4. 1.7B — 40 % of the sweep, zero finished cells; dropping it frees 15,086 of 37,860 node-hours.
80. **[focus]** 5. Is 5 × C the right budget at L ≥ 30? — ATLAS says 137 tokens/param at L50, we train 100.
81. **[focus]** 6. Model depth sits at the seed-noise floor — keep it, swap it for temperature, or spend the compute on the reference rungs?
82. **[focus]** 7. Six benchmark families never clear chance — drop them, or keep them as the gate's negative control?

## Where this leaves us (83–86)

83. **[section]** Where this leaves us.
84. **[bullets]** Summary — established / answered for one decision / uncomfortable.
85. **[bullets]** Where this leaves us — plain-language version: solid, awkward, open.
86. **[bullets]** Next — decide T, fix eval walltime, settle 90M, finish the reference rungs, re-run the analysis.

## Appendix (87–190) — signal & predictability across sizes

*Generated by `analysis/rq02_decision_accuracy/da_per_benchmark.py`; the heatmaps
come from `documents/figures/fig_appendix.py`.*

87. **[section]** Appendix — Signal & Predictability across Sizes.
88. **[default]** Appendix — Above-random signal: mean score per family × size, bold beats chance + 0.05.
89. **[figure]** Appendix — Decision accuracy, all languages at once (`appendix/da_by_benchmark.png`).
90. **[figure]** Appendix — Decision accuracy, all benchmarks at once (`appendix/da_by_language.png`).

91–190. **[default] + [figure]**, two slides per language — the DA table across
size pairs (bold ≥ 0.75), then the same table as a heatmap
(`appendix/da_<lang>.png`). 50 languages in this order:

> en · ar · az · bg · bn · bs · ca · cs · da · de · el · es · et · fa · fi · fr ·
> he · hi · hr · hu · id · it · ja · ka · kk · ko · lt · lv · ml · mr · ms · ne ·
> nl · no · pl · pt · ro · ru · sk · sl · sq · sr · sv · ta · th · tr · uk · ur ·
> vi · zh

So English is 91–92, Arabic 93–94, … Chinese 189–190.

## Known deck bugs

Two slides are malformed in the same way: a slide's frontmatter is closed, the
slide body is empty, and the *next* slide's frontmatter follows without a `---`
separator — so Slidev renders the YAML as body text and the figure never
appears, plus an empty slide after it.

- Slide 32 (`slides.md` line 526) swallows the **`optimizer_timescale.png`** figure slide ("Why the 90M rung broke").
- Slide 39 (`slides.md` line 622) swallows the **`bpb_gain_per_language.png`** figure slide ("What more languages buys, language by language") — and leaves *Finding 4* with no bullets at all.

Fix in both cases: insert a `---` separator line between the closing `---` of
the first frontmatter block and the `layout:` line that follows.
