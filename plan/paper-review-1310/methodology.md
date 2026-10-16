# Methodology review: "Small Scale, Big Decisions" (Overleaf snapshot 13-10-12h)

Reviewer stance: senior ACL/NeurIPS reviewer and methodologist. Sources read:
`documents/overleaf/13-10-12h-full-overleaf/` (main.tex and every `\input`, comments
ignored), the compiled `documents/overleaf/build-1310/main.pdf` (43 pp., figures
inspected as rendered pages in `plan/paper-review-1310/pages/`), and in the repo
`src/signal-and-noise/analysis/RULES.md`, `src/signal-and-noise/CLAUDE.md` (bug
history), `plan/paper-review-followups.md`, `plan/90M-175M-batch-retrain.md`,
`plan/600m-depth-pair-mismatch.md`, `src/pretrain/hyperparams/*.json`, and the
README auto blocks of `rq02_permutation_null` and `rq02_decisive_pairs`. No analysis
was rerun. Line numbers refer to the `.tex` source files, comments included.

---

## 1. Summary verdict

**What the paper does.** It trains a controlled ladder of 213–218 models (90M–1.7B,
plus six 3B runs), crossed with six language counts K, data schemes, temperature,
depth, activation and optimizer. Most runs use one seed. The paper then asks three
questions. (i) When is a multilingual benchmark task distinguishable from chance?
(ii) Do proxy rankings of design variants agree with the 1.7B ranking (decision
accuracy, DA)? (iii) Can proxy-only statistics such as SNR predict DA?

**Scores as a reviewer would give them**

- **Soundness: 2.5 / 5.** The measurability part is solid. The decision-reliability
  part draws its headline conclusions from populations and comparisons that cannot
  support them as stated.
- **Excitement: 3.5 / 5.** The ladder is a valuable, expensive, rare resource. The
  reformulation result and the "proxy ranking ≈ coin flip on accuracy" result matter
  in practice.
- **Overall: borderline, leaning weak reject in its current form.** The ladder and
  the measurements are strong. The way the agreement results are framed, aggregated
  and generalised is the problem. Most fixes use results that already exist in the
  repo, so a revision could plausibly move it to accept.

**Main weaknesses**

1. **The DA headline cannot be told apart from seed noise in the reference.**
   - The 1.7B reference has one seed.
   - On English, the test–retest DA between two seeds of the *same* designs is
     0.53–0.65, the same range as proxy-vs-reference DA.
   - Only 10–12% of task cells beat their own permutation null.
   - 1.7B against 3B reads 0.51.

   So the paper cannot currently separate "small proxies fail" from "the reference
   decision is not reproducible". The key evidence (seed ceiling, decisive pairs,
   permutation null) exists in the repo but is mostly absent from the paper.
2. **The effective sample size is the number of design pairs, not the number of tasks.**
   - Figure 3a pools about 1,300 tasks, but they all rank the same 32 single-axis
     model pairs.
   - Per-axis lines rest on 3–12 pairs (temperature: 4; activation and optimizer:
     3 each).
   - The figures show no uncertainty. The axis comparisons ("temperature
     consistently higher") are therefore underpowered, and they are confounded with
     effect size.
3. **Several headline claims do not follow from, or contradict, the evidence.**
   - "Per-language tokens impact decision reliability more than proxy size" (abstract,
     intro l.30): the repo's own analysis finds no such relation (language-tier DA
     *rises* for the lowest-token tier, and token share ρ = 0.02–0.27).
   - "Spend ablation compute on more sizes rather than longer runs" (§4.3 l.137)
     rests on an R² comparison with a mechanical bias, and on WSD intermediate
     checkpoints that are not shorter runs.
   - "Proxy-only statistics … do not transfer" (intro l.32) contradicts §6's
     ρ ≈ 0.55–0.62.
4. **Population and statistic choices change the headline numbers, and the paper
   reports them inconsistently.**
   - The same quantity, DA-size, appears as 0.59–0.61 (§5.2), 0.51–0.56 (§5 box),
     0.57–0.58 (App. E.3) and 0.58–0.64 (Fig. 3).
   - These come from different task filters (median-DA ≥ 0.66 selection), pair sets
     (single- vs multi-axis), scorings (accuracy vs bBPB pooled) and averages
     (macro vs pair-weighted).
   - The figure the paper leads with is filtered on DA itself (bug #17).
5. **The surrogate analysis pools heterogeneous, non-independent tasks.**
   - Figure 4's ρ ≈ 0.55 is largely a between-scoring-type effect: bBPB drives the
     level, and accuracy alone gives ρ = 0.28–0.40.
   - Its tasks include twins of the same items.
   - The paper selects a noise definition among 22 by its correlation with DA, then
     reports that correlation without correction.
   - "Signal alone" ≈ 0, so the SNR's association is carried entirely by the noise
     term.

---

## 2. Major issues

### M1. The reference decision is single-seed and largely irreproducible; the ceiling is missing from the paper

- **What.** The paper presents DA-size ≈ 0.51–0.61 as proxy failure. It never
  states the attainable ceiling: how often two seeds of the *same* reference designs
  agree on the ranking.
- **Where.**
  - `04_analysis.tex` l.229 ("agreement remains modest (59–61%)").
  - l.241 (3B: "Even at 1.7B … DA of 0.51–0.52 … increasing proxy model size to
    1.7B is still not sufficient").
  - Summary box l.270–272.
  - `06_statements.tex` l.13 (Limitations mentions single seed in one clause).
  - Fig. 3, Fig. 19.
- **Why it matters.**
  - If the reference ranking is a coin flip with respect to seed, then DA ≈ 0.5 is
    *expected for any proxy*, including a second 1.7B seed. The conclusion
    "proxies at 90M–1B are unreliable" is then unidentified.
  - The 3B result shows the same thing: 1.7B → 3B DA = 0.51 on accuracy. That is
    much more naturally read as "the 9 K-axis decisions are within noise at both
    sizes" than as "even 1.7B is too small".
- **Evidence (repo).**
  - `plan/paper-review-followups.md` W5: seed test–retest DA on replicated cells is
    0.59–0.65 (175M en), 0.61–0.62 (600M en), 0.53–0.60 (1B en) and 0.75–0.83
    (1B ru). The proxy-vs-1.7B values on the same cells are in the same range
    (0.55–0.64 en, 0.79–0.83 ru).
  - `tables/app_rq10_reference_choice.tex` (present in the folder, not `\input`):
    swapping only the seed of a 1B reference moves 175M single-axis DA from 0.66 to
    0.53.
  - `rq02_permutation_null/README.md`: only 10–12% of benchmark task cells beat
    their own permutation null at BH q < 0.05 (BPB: 34–68%).
  - `rq02_decisive_pairs/README.md`: restricting to pairs whose 1.7B gap exceeds
    2·√2 seed SD keeps 24% of pairs and raises mono-axis benchmark DA from
    0.54–0.57 to 0.61–0.66.
  - `tables/app_rq02_definition_sensitivity.tex` (also not `\input`) has the same
    picture.
  - The summary box (l.272) quotes "0.69 to 0.73" from these tables, but the table
    it comes from is not in the paper.
- **Additional subtlety.**
  - The seed SD is estimated from replicates at 175M/600M/1B (median over cells of
    SDs from 3 seeds) and applied at 1.7B, where no replicates exist
    (`decisive_pairs.py` docstring).
  - A 3-seed SD is itself very noisy: the 95% CI of σ from n = 3 spans about
    0.5σ–6σ.
- **Fix (existing results).**
  1. Put the seed test–retest ceiling into Fig. 3a as a band, or as a "second-seed
     proxy" line.
  2. Add the decisive-pairs DA and the permutation-null share to the main text
     (`tables/app_rq02_definition_sensitivity.tex`, `app_rq02_seed_uncertainty.png`,
     `app_rq02_permutation_null.png`, `app_rq02_decisive_pairs.png`; all are in the
     Overleaf figures folder but not included).
  3. Reframe §5's headline as "on most tasks the design effects are below seed noise
     at the reference, so neither proxies nor a second reference seed recover the
     ranking; on decisive pairs proxies reach 0.61–0.72".
  4. Reword the 3B paragraph as a reproducibility statement, not as evidence that
     larger proxies would not help.
- **New work.** One or two extra 1.7B seeds on two or three families would give a
  ceiling measured at the reference instead of borrowed from smaller sizes. This is
  cheap relative to the ladder and is the single most valuable addition.

### M2. Unit of analysis: tasks are pseudo-replicates of a few model pairs; no uncertainty; axis comparisons underpowered and confounded with effect size

- **What.**
  - Every DA is averaged over hundreds to thousands of benchmark–language tasks, as
    if tasks were independent.
  - All tasks rank the *same* 32 single-axis model pairs. Twins (original, RF,
    LLM-RF, and each one's bBPB) are the same items scored differently.
  - Main-text figures show no CIs.
- **Where.**
  - Fig. 3a and §5.2 l.229 ("sampling temperature consistently exhibiting higher
    agreement (72–75%)").
  - l.192 (three-pair rule).
  - Fig. 4 (n = 1,274–1,461 "tasks").
- **Why it matters.**
  - Per-axis pair counts (followups W4): language count 27, scheme 12, depth 10,
    temperature 4, activation 3, optimizer 3.
  - With 3–4 pairs, the axis lines in Fig. 3a are determined by 3–4 model
    comparisons; the jackknife leaves no band for activation and optimizer.
  - DA also depends mechanically on the effect size of the intervention. T = 3 and
    K reallocate per-language tokens by large factors (a "dose" intervention), so
    their reference gaps are large and easy to recover. Depth, activation and
    optimizer produce small effects.
  - "Temperature agrees more" is therefore a statement about effect size, not about
    the axis or about proxy reliability.
  - The same confound drives BPB ≫ accuracy: per-language BPB responds monotonically
    to the language's token share.
- **Additional confound: language tier and pairs.**
  - Rule 2 restricts each language's task to the families that train it. So a
    K=50-only language's DA is computed only on K=50 pairs (temperature and depth).
  - The "language tier" pattern (K=50 tier DA 0.64–0.69, followups W1) is therefore
    a composition effect of which axes exist for that language.
- **Fix (existing results).**
  1. Report pair counts beside every axis line.
  2. Use cluster-robust uncertainty: the leave-one-family-out jackknife bands
     already computed in `rq2_da_all_above_66_either_transformation_mono_axis.csv`.
     Put them in Fig. 3a.
  3. Drop or grey the activation and optimizer lines, and soften the temperature
     claim to "the one axis with large reference gaps".
  4. Stratify DA by reference effect size in seed-SD units: decisive-pairs by axis.
  5. In correlations, use one point per (benchmark, language) cluster, as in
     `rq04 search.py`, or a mixed model.

### M3. Selection on the outcome (bug #17) in the paper's lead agreement figure and text

- **What.** Fig. 3 and §5 restrict to tasks whose median DA-size or median DA-ckpt
  is ≥ 0.66, then plot DA-size, DA-ckpt and DA-target on them.
- **Where.**
  - `04_analysis.tex` l.193–195 ("This heuristic threshold selects tasks with higher
    observed ranking agreement").
  - Fig. 3 caption.
  - App. E.1.
- **Why it matters.**
  - Filtering on the plotted quantity inflates the level and can manufacture
    trends.
  - DA-ckpt in Fig. 3b is selected on its own median, which is circular.
  - A reviewer will read Fig. 3 as the population result. The text uses its
    numbers (59–61%) next to unfiltered ones (0.51–0.56 in the box), with no
    statement of which is which.
- **Evidence.**
  - Followups W2/W3: unfiltered single-axis DA-size is 0.55–0.59 (macro) and
    0.54–0.57 (pair-weighted).
  - Fig. 3 in-sample reads 0.60–0.64. Cross-fitted selection gives 0.57–0.60.
  - The selection adds little *once cross-fitted*. That is good news the paper
    should show instead of hiding.
  - The fixed-population twin (`figures/app_rq02_fixed_rq2.png`, not included) is
    almost identical to Fig. 3, so population drift across sizes is not the
    explanation either. That is also worth stating.
- **Fix (existing results).**
  1. Make the unfiltered population the main Fig. 3.
  2. Show the cross-fitted reliable-task line beside it
     (`app_rq02_rq2_crossfit.png`, `app_rq02_decision_accuracy_crossfit.png`).
  3. Move the filtered one to the appendix, labelled as conditional.
  4. Use one canonical number for "DA-size" throughout (see M7).

### M4. "Per-language tokens impact decision reliability more than proxy size" is unsupported

- **Where.**
  - Abstract ("informativeness … set by … the training tokens seen per-language").
  - `01_introduction.tex` l.30.
  - Contribution bullet l.38 ("at which we observe decision reliability").
  - §5 intro l.~185 ("per-language training-token budgets").
- **Why it matters.** This is a headline claim with no supporting analysis in the
  paper.
  - The only token analysis of DA is App. E.4: per-language BPB DA-goal against
    tokens, without the figure's conclusion stated.
  - The repo's analyses contradict it.
- **Evidence.**
  - Followups W1: by language tier, DA-size is highest for the lowest-token tier
    (K=50: 0.64–0.69, against 0.57–0.60).
  - `src/signal-and-noise/CLAUDE.md`: "a language's token share does not predict its
    benchmarks' reliability (ρ 0.02–0.27)".
  - For BPB, 90M reaches 0.90 at 1C on 0.02B tokens while 600M stays at 0.70–0.74
    up to 0.50B.
  - The per-benchmark guide (Table 7) gives token thresholds for *measurability*
    only, not for decision reliability.
- **Fix (text only).**
  - Restrict the token claim to measurability (§4, where it is supported).
  - Either state the negative result for DA, or delete it.
  - Correct contribution bullet 2.

### M5. The "more sizes rather than longer runs" recommendation does not follow

- **Where.** `04_analysis.tex` l.130–137; Fig. 2b; App. D.4.
- **Problems.**
  1. **Mechanical R² asymmetry.**
     - Size fits use 3–6 points (`rq01/analyze.py` `MIN_RUNGS = 3`; gated sizes are
       dropped). Training fits use 10 checkpoints.
     - Under a null of no relation, E[R²] = 1/(n−1): 0.5 at n = 3, 0.2 at n = 6,
       about 0.11 at n = 10.
     - At the R² ≥ 0.5 cut, a 3-point size fit passes about half the time by chance,
       a 10-point training fit rarely.
     - "No task predictable along training without along size" is therefore
       expected even with no real difference.
  2. **Intermediate WSD checkpoints are not shorter runs.**
     - The schedule decays over the last 20% of training
       (`hyperparams_deep.json`: `lr_wsd_decay_iters` ≈ 20% of `train_iters`).
     - Checkpoints at 10–80% are un-annealed and high-LR. Their scores sit below
       those of a fully annealed run of that length, and the decay causes a
       discontinuity.
     - Neither the training-trajectory fits nor DA-ckpt/DA-target evaluate "a
       shorter run". The paper never mentions this.
  3. **Wrong evidence base.**
     - The recommendation concerns *decisions*, but the evidence is fit R² of
       *scores*.
     - On decisions, §5 finds the opposite: DA-size is flat in size (0.55–0.59),
       while DA-ckpt rises with training.
     - The two sections therefore give contradictory practical advice.
  4. **Not a compute-matched comparison.** A practitioner choosing between "more
     sizes" and "a longer run" should compare equal-FLOP alternatives. The ladder
     contains no longer-than-100N runs.
- **Fix.**
  - Drop the recommendation, or restate it as "within one 100N run, accuracy
    trajectories are less often log-linear than size curves".
  - Control for point count: subsample the training trajectory to the same number
    of points as the size fit, or use adjusted R² or a held-out-point error. Both
    can be done from existing tables.
  - Add one sentence on WSD: intermediate checkpoints are stable-phase checkpoints.
  - If a compute-allocation claim is wanted, it needs new runs: short fully-decayed
    runs, e.g. 20N with its own decay, at two sizes.

### M6. Surrogate analysis (§6): pooling and composition effects, non-independence, specification search, attenuation

- **Where.** `04_analysis.tex` l.~300–350; Fig. 4; App. F; intro l.32.
- **Problems.**
  1. **Between-group (Simpson-type) effects.**
     - Fig. 4 pools accuracy and bBPB tasks across 40+ benchmarks and 50 languages
       into one Spearman ρ per size.
     - App. F.2 states that bBPB "drives the level". Accuracy alone gives
       early-checkpoint agreement ρ = 0.28–0.39 and relative-SD SNR 0.33–0.40;
       bBPB gives 0.57–0.68 and 0.46–0.58 (Fig. 21; followups W8).
     - Relative noise (SD/mean) differs systematically between bits and accuracies,
       and so does DA. The pooled ρ partly encodes "is this a bBPB task".
  2. **Noise drives the "SNR" result.**
     - In Fig. 4, "noise alone" (0.44–0.56) is about as good as relative-SD SNR, and
       "signal alone" is about 0 or negative.
     - The finding is therefore about noise, not about SNR, and should say so.
  3. **Non-independence.**
     - Points include twins of the same items (original/RF/LLM-RF and their bBPB),
       the same benchmark across up to 49 languages, and the same model pairs.
     - Nominal n = 1,274–1,461 overstates the information, and no CI or test is
       given.
     - RULES.md itself says "its points are not independent … a nominal p is
       optimistic; rq04's catalogue search (`search.py`) is the clustered version".
  4. **Specification search.**
     - App. F.1: "we evaluate several candidates and select the pair with the
       strongest empirical correlation to DA" (`app_06_snr.tex` l.34), over 22
       definitions.
     - The best-of-22 correlation is reported in the main text without
       out-of-sample validation.
     - The repo has a cross-fit mechanism for reliable tasks; the same idea (choose
       the definition on half the languages or benchmarks, report on the other)
       applies here.
  5. **The outcome is noisy and coarse.**
     - Per-task DA with 3–10 pairs takes a handful of values (with 3 pairs: 0, 1/3,
       2/3, 1). Its test–retest reliability is low (M1).
     - This caps any surrogate's achievable correlation. The paper should report the
       DA split-half reliability (pairs split in half) as the ceiling. Without it,
       "no surrogate works" and "DA is unmeasurable per task" cannot be told apart.
  6. **Two inconsistent methods.**
     - The main text uses Spearman across all tasks at each size. App. F uses the
       Pearson r of log SNR within language, averaged over languages, pooling sizes.
     - RULES.md records the pooled r for the best definition as 0.215 (2026-09-30
       tables), far below the main-text 0.53–0.60.
     - Readers will not know which to believe.
  7. **The definition in the paper does not match the code.**
     - App. F.1 (`app_06_snr.tex` l.25) defines window noise as the *population SD*
       of five checkpoints.
     - Since 2026-10-08 the code uses the residual SD around a fitted line
       (detrended, `utils.checkpoint_noise`, RULES rule 4), because the decay still
       rises and the raw SD overstated noise 1.5–4×.
     - The paper must state which one the numbers use. The seed-noise equation
       (l.~65) and the discrepancy equation (the formula after "with
       u_a = …" is missing in the PDF) are garbled or incomplete.
  8. **Contradictory claims.**
     - Intro l.32 says proxy-only statistics "do not transfer", but §6 reports
       ρ ≈ 0.55–0.62.
     - "Maximum ρ = 0.75 across all benchmarks" (l.331) does not appear in Fig. 4
       (max 0.62).
     - "ρ = 0.93 for HellaSwag" is a best-of-benchmarks value on about 18 tasks.
     - The monolingual comparison that would support "do not transfer" (ρ = 0.89 on
       DataDecide against 0.46/0.43/0.04/−0.10 at K = 1/2/15/30,
       `tables/app_snr_monolingual_table.tex`) is not included. It also has n = 9
       benchmarks per column, with DA from very few pairs.
- **Fix (mostly existing).**
  1. Make the main Fig. 4 accuracy-only, with bBPB as a separate panel.
  2. Report clustered (by benchmark) bootstrap CIs.
  3. Report the noise-alone baseline explicitly.
  4. Cross-fit the choice of SNR definition.
  5. Add the DA reliability ceiling.
  6. Include the monolingual table with its caveats.
  7. Align the noise definition with the code.
  8. Fix intro l.32 to "proxy-only statistics are moderately associated with DA,
     mainly through bBPB and noise; on accuracy the association is weak (ρ ≈ 0.3–0.4)".

### M7. One quantity, many numbers: inconsistent populations and aggregation across sections

- **Where (DA-size of benchmarks, 90M–1B):**

  | Location | Value |
  |---|---|
  | §5.2 l.229 | 59–61% (Fig. 3 filtered, single-axis, macro, acc + bBPB) |
  | §5 box l.270 | 0.51–0.56 (accuracy only, unfiltered) |
  | App. E.3 l.8 | 0.57–0.58 (multi-axis, pooled over pairs) |
  | §5.3 | 58–64% |
  | App. E.3 l.17 | 0.56→0.69 for accuracy "on the tasks that rank reliably" |

  BPB: 0.74–0.94 (box), 0.94/0.94/0.89/0.74/0.92 (App. E.3), 0.95–0.98 (macro).
- **Why it matters.**
  - A reviewer cannot tell which number is the result.
  - The differences are larger than the effects being discussed: an accuracy trend
    of 0.56→0.69 exists only inside the DA-selected set.
  - Macro over tasks and pair-weighted pooling differ by about 0.03–0.05.
- **Fix.**
  1. Declare one canonical population and statistic: unfiltered, single-axis,
     accuracy and bBPB reported separately, pair-weighted, with jackknife CI.
  2. Give every other number as a labelled sensitivity in one appendix table.
  3. Make the summary box and the body agree.

### M8. Construct validity of "measurability" and of bBPB

- **Above-chance is not the same as measuring the capability.**
  - Table 7 shows Belebele RF and LLM-RF reaching 50% above chance at ≤ 440K tokens
    of the language (90M), and LLM-RF passing "in all trained languages at every
    model size" (`04_analysis.tex` l.92). HellaSwag, MultiBLiMP and TokSuite pass
    at ≤ 0.5–6M tokens.
  - A 90M model that has seen less than 1M tokens of a language cannot read that
    language. Passing then reflects cross-lingual transfer (shared scripts and
    named entities), lexical overlap between passage and option in cloze scoring,
    or cues the LLM rewrite introduced.
  - The only leakage check is the "longest option" length cue (App. D.2).
  - Fix with existing data: the deep scheme-A seed-1904 runs are evaluated on
    *untrained* languages (App. C, l.~61; rq06). Report RF/LLM-RF Belebele above-chance
    rates on untrained languages as a surface-cue baseline.
  - A cheap new check: passage-ablated or question-ablated LLM-RF items, and an
    option–passage lexical-overlap baseline.
- **bBPB.**
  - The bBPB of a letter-format task scores the gold *letter* (e.g. `belebele_eng_Latn`
    bBPB = 1.28 bits/byte in `rq08_subset_selection/bench_bpb.csv`), i.e. label-prior
    calibration over 1–2 bytes.
  - On cloze tasks it is largely a language-modelling score of the answer text, so
    its higher DA may be inherited from per-language BPB (data-dose effect, M2)
    rather than from the benchmark's capability.
  - The fairer comparison (bBPB → reference *accuracy*, +0.03 to +0.10) is in
    App. E.3 but is not the headline.
  - Fix:
    1. Exclude letter-format bBPB or report it separately.
    2. Report bBPB→accuracy as the bBPB result.
    3. Report the partial association with per-language BPB DA, or use a
       gold-vs-distractor margin, which the per-item store already holds as
       `margin`.

### M9. Design confounds in the interventions that bear on "scale-dependent" rankings

These may legitimately change rankings across size. The paper must state them so
that reversals are not read as proxy unreliability.

- **Batch, steps and optimizer window vary with size.**
  - 90M and 175M train at batch 84/168 with 27,000/25,620 steps. 350M has 16,660;
    1.7B has 81,000.
  - The 90M/175M rungs were retrained because AdEMAMix's slow EMA (β₃ = 0.9999,
    about a 10k-step window) misbehaves on short runs
    (`plan/90M-175M-batch-retrain.md`).
  - The AdEMAMix-vs-Muon decision is thus intrinsically run-length dependent
    (window/run ratio 0.6× at 350M against 8× at 1.7B). Its proxy-vs-reference flips
    may be real optimizer×scale interactions.
  - Model shape also changes at the top: d/L is 64 at 175M–1B, 77 at 1.7B and 51 at
    90M.
- **Size matching.**
  - Shallow 600M has +3.7% N and +15% compute against deep (`plan/600m-depth-pair-mismatch.md`).
    Shallow 90M has −5.2% N.
  - The paper says the counts "differ by 0.4% to 5.2%" (App. A) but does not flag
    600M.
  - §6 l.343 then uses 600M BPB DA (0.70–0.74) as its showcase "surrogates look
    good but DA is low" example. That example is the size-mismatched depth pair plus
    one seed draw (followups W6: with depth held fixed, 600M BPB DA = 0.86).
  - Remove or reframe the example; flag the pair in Fig. 3a.
- **The scheme axis is three different interventions under one label.**
  - At K=1 it is the English corpus (DCLM-Edu / DCLM / FineWeb), at K=2 the second
    language (ru/zh/es), at K ≥ 8 a diversity-first list.
  - At K=2, the tasks of the swapped language are evaluated only on the family that
    trains it (rule 2), so scheme DA at K=2 is read on English tasks, where the
    effect is second-order and DA ≈ 0.5 is expected.
  - Data repetition varies across the ladder: ES at 0.37 epochs at 175M against
    3.51 at 1.7B; ZH at 1.61 epochs on an undersized build (RULES "Second-language
    repetition"). The rules require stating epochs; the paper does not.
  - Scheme has the lowest DA (0.49–0.53). Split it by K or report it with these
    caveats.
- **K couples language count, per-language token share and English share.**
  - English goes from 100% at K=1 to 50% at K ≥ 2.
  - A K decision for a German task is mostly "fewer German tokens", so it is a dose
    decision, not a "number of languages" decision.
  - State this in §3 and in the interpretation of the K line.
- **Seeds.**
  - Replicates exist only for deep scheme A/B at 175M/600M (K ∈ {1, 2, 50}) and 1B
    (K ∈ {1, 2, 30} + B30). There are none at 90M, 350M or 1.7B, and none for any
    non-deep ladder.
  - Seed noise for depth, activation and optimizer pairs is therefore assumed equal
    to deep's.

### M10. Generalisation and scope of conclusions

- **Scope of claims.**
  - The reference is 1.7B at 100N. The 3B check covers six configurations, 9
    single-axis pairs, and effectively one axis (K) with ≥ 3 pairs.
  - Claims such as "challenge the assumption that ablations at smaller scales
    reliably identify the best choices for larger multilingual models" (Conclusion)
    and "how much performance have current models lost through unreliable proxy
    decisions" overreach.
  - At 1.7B the decisions themselves are mostly within seed noise (M1). That says
    little about production-scale decisions, where effects such as data mixtures are
    larger and models are trained far beyond 100N.
- **DA unit vs the practitioner's decision.**
  - DA is per task. Practitioners decide on an aggregate (suite or language
    average), which averages noise away.
  - The repo shows the BPB macro reaches 0.95–0.98 and a "panel macro is the safest
    from 350M up" (`rq06 language_panel.py`).
  - Report DA for the aggregate benchmark score, macro over tasks within a language
    and over languages. This is likely the most decision-relevant number, and it may
    change the conclusion considerably.
- **Limitations section.** It should state:
  - The single-seed reference and the measured test–retest ceiling.
  - Per-axis pair counts.
  - WSD intermediate checkpoints.
  - The size, batch and step heterogeneity of the 90M/175M rungs.
  - The 600M depth mismatch.
  - L2 repetition.
  - That 1.7B is the largest reference.
  - That conclusions apply to 100N training.

---

## 3. Minor issues

1. **Counts disagree across the paper.**
   - Models: 213 (abstract, §3, conclusion) against 218 runs (Table 2 total,
     `\nruns`).
   - Benchmarks: "20 benchmark families" (abstract) against 52 analysed / 43
     screened / 95 considered (App. C) against "Analysed (40 benchmarks, 695 tasks)"
     and "Screened (26 benchmarks, 485 tasks)" in the PDF's Table (p.22) against
     938 analysed tasks in the text.
   - Reconcile them, and say whether "tasks" includes RF, LLM-RF and bBPB twins.
2. **Training-grid table is wrong for 3B** (`app_01_model_ladder.tex` l.87, l.93).
   It lists 3B at K ∈ {8, 15, 30} for both A and B. The actual cells (and App. E.5)
   are A at {8, 15, 30, 50} and B at {8, 15}.
3. **Task selection is described in two ways** (`app_03_evaluation.tex` l.54 and
   l.58).
   - One says tasks at chance at 1B and 1.7B were discarded; the other says those
     at chance at 600M–3B. Yet §4.1 says 34% of the analysed tasks pass at no size
     up to 1.7B.
   - Explain: presumably the letter originals are kept because their twins pass.
   - Note that the screening used the reference and 3B models, which conditions the
     population on reference-scale measurability. That is acceptable but should be
     said.
4. **"105 languages" for Belebele** (§4.2 l.90). Models train 50 languages and
   Table 6 lists 49 trained Belebele languages. If 105 includes untrained-language
   evaluations, this breaks the paper's own trained-languages-only rule (rule 2).
   Clarify.
5. **Three-pair rule wording** (l.192). "Comparisons between only two pairs yield
   binary agreement scores" is wrong: pair counts are triangular (1, 3, 6, …), so
   the binding case is 2 families giving 1 pair. Also, with 3 pairs DA takes only
   {0, 1/3, 2/3, 1}, which deserves a sentence and affects the 0.66 cut: "≥ 0.66"
   equals "≥ 2/3" equals "at most one flip in three".
6. **Wilson gate details.**
   - One-sided 95% gives a 5% false-pass rate per run on at-chance tasks. The
     "half of runs" vote assumes independence, but runs answer the same items.
   - In the token analysis (App. D.3) each *single evaluation* is gated, so the
     floor of the "share above chance" curves is at least 5%, not 0.
   - Accuracy is reconstructed as round(p̂·n), which is fine. acc_norm under Wilson
     is still a proportion, but chance under length normalisation is not exactly 1/m
     when lengths correlate with gold. The majority-class check (App. D.1) partly
     covers this.
7. **Token axis uses tokenizer tokens.** Tokenizer fertility varies by language
   (cited in the intro), so "tokens of the language" thresholds are not comparable
   across scripts. Give bytes too, or note it.
8. **Threshold table extrapolation.** Many entries are "≤ X" or "> X": the logistic
   fit's 50% point lies outside the observed range. Say these are bounds, and give
   the number of evaluations per fit.
9. **Scaling fits.**
   - The R² median "0.87 across 431 tasks" uses only above-chance sizes. Truncating
     the low end selects for curvature-free segments.
   - TruthfulQA "ρ = −0.75 … a strong and consistent decrease in predictability"
     should read "decrease in score with size" (inverse scaling), not
     predictability.
   - TruthfulQA is not visible in Fig. 2a.
10. **Near-tie sensitivity (App. E.2)** is one-sided: only reference near-ties that
    are reversed get credited. It uses an ε in accuracy percentage points that has
    no meaning for bBPB or BPB. Use the seed-SD-based decisive-pairs version instead,
    since the table already exists.
11. **DA vs Kendall.** The identity 2·DA − 1 = τ_a + (T_both − T_one)/n is fine. The
    r = 0.99 agreement is computed over values that share the same pairs, so it says
    the choice of statistic does not matter. It does not validate DA.
12. **DA-ckpt and "within-model agreement" are partly within-run autocorrelation.**
    - Adjacent checkpoints share seed, data order and the decay phase. DA at 4.5C
      against 5C (90% against 100%, both inside the decay) is high by construction.
    - `src/signal-and-noise/CLAUDE.md` notes DA-ckpt at 90% (0.74–0.82) sits near
      the seed null (up to 0.75).
    - Show the seed null in Fig. 3b. Also, the 1.7B line in Fig. 3c is the 1.7B
      DA-ckpt, not a proxy.
13. **§5.1 "350M highest"** holds only late in the run (followups W4). There is no
    statistical test.
14. **App. E.4 figure** has no stated conclusion in the text, and the caption has a
    double period.
15. **Undefined or garbled equations.** The seed-noise formula (U, M_i, n, T
    undefined), the discrepancy definition (missing equation) and the total
    variation are defined but unused in Fig. 4. Trim to what is used.
16. **Unresolved items.** A TODO comment survives in the source (l.241, "resolve
    appendix reference"). Limitations claims "we report fixed-population
    comparisons", but no fixed-population figure is included. Include
    `app_rq02_fixed_rq2.png` or drop the claim.
17. **Language transfer example (App. E.6).** "English predicts the rankings of the
    other languages at only 0.45" is a single benchmark, a single list (K=8) and
    four target languages, with no CI. Present it as illustrative.
18. **Model-based reformulation.** Manual review covers six languages. Report an
    error rate and whether gold-option meaning is preserved for low-resource
    languages. The "no length cue" statistic (22.2% vs 21.9% longest) should also be
    reported per language.

---

## 4. Questions reviewers will ask (with answers where the repo has them)

1. **What DA would a second seed of the 1.7B reference get against the first?**
   - Not measured at 1.7B.
   - At 175M/600M/1B on English: 0.53–0.65. On Russian at 1B: 0.75–0.83.
     (`rq02_decision_accuracy/README.md` auto block `seed-uncertainty`; followups W5.)
2. **How many model pairs are behind each line in Fig. 3a, and what are the CIs?**
   - 32 single-axis pairs overall: K 27, scheme 12, depth 10, T 4, activation 3,
     optimizer 3.
   - 90% leave-one-family-out jackknife lower bands: all pairs 0.55, T 0.69; none
     for activation or optimizer.
   - Source: `rq2_da_all_above_66_either_transformation_mono_axis.csv`.
3. **What is DA on pairs the reference actually separates?**
   - Mono-axis benchmarks, reference gap > 2√2 seed SD (24% of pairs): 0.61–0.66.
   - Multi-axis: 0.69–0.73.
   - BPB: about 0.94–0.99.
   - Source: `rq02_decisive_pairs/README.md`.
4. **Is the reliable-task filter driving Fig. 3?**
   - Partly. Unfiltered: 0.55–0.59. Cross-fitted: 0.57–0.60. Filtered in-sample:
     0.60–0.64.
   - Source: followups W2/W3.
5. **How many individual task cells beat a random ranking?**
   - Benchmarks: 10–12%. BPB: 34–68%. (BH q < 0.05; `rq02_permutation_null`.)
6. **Does the moving task population explain the flat DA-size?**
   - No. The fixed-population twin is nearly identical
     (`figures/app_rq02_fixed_rq2.png`).
7. **Is the 600M BPB dip real?**
   - It is the non-size-matched depth pair plus a seed draw.
   - With depth held fixed, BPB DA-size is 0.91/0.90/0.85/0.86/0.88. Re-drawing the
     deep 600M seed moves the depth flips from 50/50 to 15/50.
   - Source: followups W6, `rq03_noise_and_snr` `flip-attribution`.
8. **Does bBPB help with the accuracy decision, not just its own?**
   - Yes, modestly: +0.10 (90M) down to +0.03 (1B), better on 57–71% of tasks,
     Wilcoxon p < 1e-7 (but tasks are not independent).
   - Source: `bench_bpb_da_size_summary_multi_axes.csv`.
9. **Do per-language tokens predict DA?**
   - No. Language-tier DA is highest for the K=50 (fewest-token) tier, which is
     confounded with which axes exist there. Token-share ρ = 0.02–0.27.
   - Source: followups W1; `reliability_da_size_by_language_tier_multi_axes.csv`.
10. **Are the surrogate correlations driven by bBPB, and what about accuracy alone?**
    - Yes. Accuracy alone: early-checkpoint agreement ρ = 0.28–0.39, relative-SD SNR
      0.33–0.40 (Fig. 21).
11. **Does your SNR implementation reproduce Heineman et al. on English?**
    - Yes on DataDecide (ρ = 0.89). On the ladder: 0.46 (K=1), 0.43 (K=2), 0.04
      (K=15), −0.10 (K=30), with 9 benchmarks each.
    - Source: `tables/app_snr_monolingual_table.tex`, `rq13_english_only/allenai_snr.py`.
    - The table is not in the paper.
12. **Which noise definition do the reported numbers use, raw or detrended
    window SD?**
    - The code has used detrended since 2026-10-08 (RULES rule 4); the paper
      describes raw. Confirm against the snapshot the figures were built from.
13. **Are 90M/175M comparable rungs given the different batch and step count?**
    - The retrain note explains the choice (AdEMAMix window). The paper's Table 1
      lists the batches but does not discuss the consequence for optimizer-axis
      decisions.
14. **Is the measurability of reformulated Belebele real comprehension?**
    - Not tested. The untrained-language evaluations of the scheme-A deep runs
      (rq06 data) can answer this without new training.
15. **What is DA for the suite-level score a practitioner would use?**
    - BPB macro: 0.95–0.98.
    - Benchmark macro per language or over languages: partially in
      `rq06_language_transfer/language_panel.py` ("panel macro is safest from 350M
      up"). Needs to be surfaced.
16. **Why 0.66 as the reliability cut?**
    - With 3 pairs it means "at most one flip". It is heuristic and is applied to the
      median over proxies. Show sensitivity (0.5 / 0.66 / 0.75 / 0.8; the repo has
      `above_80` and `RELIABLE_DA = 0.75` variants) or drop the filter (M3).

---

## Priority list for the revision

1. **M1.** Seed ceiling, decisive pairs and permutation null into §5 main text;
   reframe.
2. **M3, M7.** One canonical unfiltered population and pair-weighted statistic with
   CIs; filtered figure to the appendix.
3. **M4, M5.** Remove or restate the token→reliability and sizes-over-runs claims.
4. **M6.** Accuracy-only Fig. 4 with clustered CIs, noise-alone baseline, cross-fitted
   definition choice, DA ceiling; fix intro l.32.
5. **M2, M9.** Pair counts per axis; caveats for scheme, optimizer and 600M depth;
   K as a dose intervention.
6. **M8.** Untrained-language control for RF/LLM-RF; bBPB→accuracy as the bBPB
   headline; letter-format bBPB excluded.
7. **Only new compute recommended.** A second 1.7B seed on two or three families.
