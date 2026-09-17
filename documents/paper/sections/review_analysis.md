# Analysis review

The prose now distinguishes agreement with an observed reference from a stable development decision, descriptive scaling fits from extrapolation, and associations from causal explanations. All five questions and figures remain in `04_analysis.tex`. The suggestions below are proposals, not experiments performed in this review. Figure assets and analysis code were not edited.

## Resolve before finalizing the results

1. **Reconcile RQ3 with its figure data.** The draft reports benchmark SNR correlations of 0.52, 0.60, and 0.41 and population sizes of 68, 78, and 93. The checked-in `rq_facts.json` starts with 0.501 and 67 tasks at 175M; discrepancy uses 66 tasks. The prose's numbers have been retained pending selection of one consistent analysis snapshot. Verify every RQ3 value against `rq3_surrogates.csv`, report candidate-specific sample sizes or use a common complete-case population, and check whether the 101 BPB entries include the macro average. An aggregate should not be treated as another independent language.

2. **State exactly where the gate applies.** RQ1 includes near-chance families, while the opening originally said every benchmark analysis was gated. RQ3 counts also differ from the opening's survivor counts. Specify the task universe, aggregation over variants/seeds, treatment of tasks without a chance baseline, and gate application for each RQ. Distinguish a five-percentage-point margin from statistical evidence above chance.

3. **Audit population definitions.** The original RQ4 count of 98 languages trained by neither second-language variant is inconsistent with excluding English and two different second languages from a universe of 100, which would leave 97. The count is now an author comment rather than an asserted result. Check the implementation's exact set difference and the task-to-language mapping. Also verify whether RQ2's original examples of 2,250 and 500 tasks count unique tasks or repeated comparisons.

4. **Report reference sizes by metric and setting.** The figure data use different references for BPB and benchmarks, including 600M versus 1B for temperature and mixtures of 600M and 1.7B for depth. The RQ2 heatmap also changes its number of contributing settings across columns. Avoid interpreting these changes as a pure effect of proxy size or training progress. The complete planned grid in the methodology and the available scored grid in the analysis should remain explicitly distinguished.

5. **Make reference self-comparisons a check.** A reference's final ranking compared with itself should agree perfectly under the stated sign-agreement definition. The original text's claim of near-half agreement in this case was incorrect; the plotted reference comparisons use earlier checkpoints. Preserve the distinction and document tie handling. The RQ2 image also shows agreement slightly below one at 80% training, so the prose now says agreement remains high rather than unchanged.

6. **Keep RQ5's limits visible.** The signed error is `(prediction - observation) / observation`; the reported -3.4% means optimistic BPB prediction, not better-than-predicted reference performance. This interpretation is corrected. Borrowed exponents outperform own-language fits for trained languages, but the checked-in summary shows own-language fits outperforming transfer for untrained languages at four rungs. Populations shrink at four rungs, so improvements across the horizontal axis need a matched-chain comparison. “Held out” refers to exponent fitting; the target language still supplies at least one measurement.

## Experimental improvements, in priority order

1. **Quantify reference uncertainty.** Repeat both intervention levels at the actual reference size for the most consequential comparisons. Estimate the distribution of the paired score difference and its sign stability. The existing median baseline seed standard deviation is a useful descriptive scale, but cannot establish significance of a contrast or variability at an unreplicated 1.7B reference. Keep seed variation separate from evaluation-sample uncertainty.

2. **Use matched populations and paired uncertainty estimates.** Recompute the principal size/checkpoint comparisons using a fixed set of settings, tasks, and reference sizes. Resample benchmark items or validation documents jointly across models, and account for repeated languages/settings when summarizing. Report ties and confidence intervals for agreement; one half is not automatically the appropriate null when ties or preferred intervention levels are imbalanced.

3. **Validate the surrogate on held-out decisions.** Choose the SNR definition and any acceptance threshold on one set of interventions or language families, then test it on others. Compare against signal alone and early-to-final proxy stability. Verify that the scaling-fit R-squared comparator does not use reference-size observations if it is presented as available before training the reference. Report reliability versus coverage for a rule that can abstain when evidence is weak.

4. **Test gate sensitivity and reference availability.** Show ungated results alongside several reasonable gate thresholds, and distinguish insufficient measurement precision from absence of a capability. Complete the missing reference evaluations before making claims about the full planned ladder. These checks would make the negative benchmark result more persuasive.

5. **Strengthen language transfer.** Evaluate pooled exponents with entire language families or scripts held out, as well as individual languages. Compare methods on identical chains at every number of measured rungs. Report the cost of obtaining donor-language scaling curves alongside the single target-language measurement; the transfer method is economical only once donor information is available.

6. **Separate architecture from budget effects.** Report sensitivity to the shallow/deep differences in exact parameter count, token budget, and output-projection compute. For a follow-up, use matched-compute or matched-token comparisons and replicate the smallest stable rungs before attributing their rank reversals to optimizer timescales.

## Figure improvements

| Figure | Proposed change | Purpose |
| --- | --- | --- |
| RQ1 | Show family fit distributions and numbers of unique languages as well as medians; add held-out-size prediction error. | A strong in-sample fit on a few sizes is not an extrapolation test, and pooled medians hide language variation. |
| RQ2 | Center a diverging color scale at one half, with a note that this is descriptive unless the null is justified. Label omitted self-comparisons and missing results separately. Show reference size and item count as well as number of settings; add a matched-setting panel. | The current single-color heatmap obscures systematic reversals and mixes changes in coverage with changes in agreement. |
| RQ3 | Add uncertainty intervals, candidate-specific sample counts, and a separate group for statistics requiring multiple model sizes. Mark values unavailable before reference training. | Makes predictive strength and measurement cost comparable and exposes leakage risks. |
| RQ4 | Plot per-setting signed effects with uncertainty alongside absolute effect/noise ratios; label reference sizes. | Absolute medians conceal reversals, and the ratio-one line is not a significance boundary. |
| RQ5 | Split the crowded six-entry legend into training-exposure panels, show counts at each rung count, and add matched-population curves. Add relative-error bands and a residual panel to the prediction scatter. | The existing log-scale scatter can look strong because languages have very different BPB levels; residuals reveal systematic error and changing populations. |

The RQ2 and RQ5 PNGs were inspected directly. Suggestions for the other figures follow their captions and associated data. No figure was regenerated.

## Methodology follow-through

The main methodology now states approximate architecture matching, the BPB normalization limits, combined initialization/data-order noise, and the difference between the planned and available references. Configuration tables and implementation details are in `app_methodology.tex`, included through `app_languages.tex`. Existing citation placeholders and provisional compute totals remain to be finalized. No bibliography or top-level LaTeX document is present in the paper directory, and no LaTeX compiler is available here; checks were structural rather than a PDF build.
