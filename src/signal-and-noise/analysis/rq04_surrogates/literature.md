# Surrogates of decision accuracy: literature review

This file lists the statistics, besides the SNR variants (rq03) and the
FineTasks criteria, that published work uses or proposes to tell cheaply
whether a benchmark ranks models reliably. `catalogue.py` computes the ones
that our data supports and scores them against DA-size. It reports the
results in the README section "A catalogue of surrogates beyond SNR".

Sources were checked on 2026-09-24. **Full text** means we read the paper
through the Hugging Face papers mirror. **Abstract** means we checked it
against the abstract or index entry only. **Classic** means a textbook
statistic whose reference we did not re-fetch.

## 1. What others found

| paper | what is correlated | number | read |
|---|---|---|---|
| Heineman et al. 2025, *Signal and Noise*, NeurIPS, [arXiv:2508.13144](https://arxiv.org/abs/2508.13144) | SNR vs DA (DataDecide 60M–750M → 1B) | R = 0.79 (R² = 0.63). Signal alone or noise alone: not correlated. Signal variants inside SNR: relative dispersion R² 0.57, relative std 0.57, IQR 0.48, Gini/discrepancy/depth ≈ 0.03 | full text |
| same, App. B.1 | DA resampled over late checkpoints | DA is itself noisy: its variance falls as SNR rises | full text |
| Magnusson et al. 2025, *DataDecide*, ICML, [arXiv:2504.11393](https://arxiv.org/abs/2504.11393) | spread vs seed noise vs DA | qualitative: high DA needs low seed noise or wide spread. Single-scale ranking gives about 80 % DA, and crossovers set the ceiling | full text |
| Bhagia et al. 2024, *Model Ladders*, [arXiv:2412.04403](https://arxiv.org/abs/2412.04403) | late-checkpoint std of task loss vs prediction error | r = 0.82 | abstract / snippet |
| Liu et al. 2024, *RegMix*, ICLR 2025, [arXiv:2407.01492](https://arxiv.org/abs/2407.01492) | 1M-proxy vs 1B mixture ranking (validation loss) | Spearman 0.97 | abstract |
| Yauney et al. 2026, *How reliable is LM micro-benchmarking?*, ICLR, [arXiv:2510.08730](https://arxiv.org/abs/2510.08730) | aggregate rank correlation vs pairwise correctness | a high τ still misorders close pairs. Proposes the minimum detectable ability difference (MDAD) | abstract |
| Patel et al. 2026, [arXiv:2605.18607](https://arxiv.org/abs/2605.18607) | token-level proxies on expert traces vs true ranking | Spearman 0.81 against 0.36 for cross-entropy | full text |
| Lourie, Hu & Cho 2025, EMNLP Findings, [arXiv:2507.00885](https://arxiv.org/abs/2507.00885) | downstream scaling laws | extrapolation works in 39 % of cases | abstract |
| *Can small training runs reliably guide data curation?* 2025, [arXiv:2512.24503](https://arxiv.org/abs/2512.24503) | proxy ranking under hyperparameter changes | rankings reverse with small changes to hyperparameters | abstract |
| Madaan et al. 2024, [arXiv:2406.10229](https://arxiv.org/abs/2406.10229) | seed variance, monotonicity | no DA. Monotonicity is 0.09 for MMLU and 0.95 for MMLU-cloze | full text |

The only paper with a strong correlation (R = 0.79) spans tasks from
near-chance to easy on a 25-recipe ladder. Our above-random gate removes the
low-SNR tail that drives that correlation, so part of the gap is range
restriction.

## 2. The catalogue

We have one aggregate score per checkpoint, the item count and the chance
level, but no per-item data or log-probabilities. The *code* column is the
name in `catalogue.py`. A dash means the statistic needs data we do not have.

| # | family | statistic | code | source |
|---|---|---|---|---|
| 1 | separation | median pairwise gap / checkpoint noise | `gap_over_noise` | Heineman 2025 (pairwise form) |
| 2 | separation | share of pairs resolved at 1.96 σ (discriminative power) | `resolved_pairs_noise` | Sakai 2006, SIGIR |
| 3 | separation | model-implied retest agreement Φ(z)² + Φ(−z)² | `retest_agreement` | Voorhees & Buckley 2002 (swap rate) |
| 4 | separation | ICC(1) over the noise window | `icc_window` | Shrout & Fleiss 1979 (classic) |
| 5 | separation | G-coefficient Eρ² = ICC(1,k) | `g_coefficient` | Bodoff & Li 2007; Urbano et al. 2013, SIGIR |
| 6 | separation | ANOVA η² | `eta2_window` | classic |
| 7 | separation | Cronbach's α (checkpoints as items) | `cronbach_alpha` | Cronbach 1951 (classic) |
| 8 | separation | binomial SNR: spread / √(p(1−p)/n) | `binomial_snr` | Card et al. 2020, EMNLP, [arXiv:2010.06595](https://arxiv.org/abs/2010.06595); Madaan 2024 |
| 9 | separation | share of pairs a two-proportion z-test separates | `resolved_pairs_binomial` | Card 2020; Dror et al. 2018 |
| 10 | separation | share of pairs closer than one item | `tie_rate_items` | Card 2020 |
| 11 | separation | checkpoint noise / binomial se | `noise_to_binomial` | Madaan 2024; Wang 2025, [arXiv:2512.21326](https://arxiv.org/abs/2512.21326) (abstract) |
| 12 | separation | share of pairs above the MDE at 80 % power | `mde_resolved` | Card 2020 |
| 13 | separation | probability of outperforming, mean \|P − ½\| | `prob_outperform` | Bouthillier et al. 2021, MLSys |
| 14 | separation | DIoR: 5th percentile of τ under random window checkpoints | `dior` | Perlitz et al. 2023, [arXiv:2308.11696](https://arxiv.org/abs/2308.11696) (full text) |
| 15 | rank stability | Kendall τ between consecutive tenths | `consecutive_kendall`, `consecutive_kendall_late` | FineTasks (Kydlíček et al. 2024) |
| 16 | rank stability | τ of window checkpoints with the final | `window_kendall` | Heineman 2025 |
| 17 | rank stability | Kendall's W over the window | `kendall_w_window` | Kendall & Babington Smith 1939 (classic) |
| 18 | rank stability | share of pairs with a constant sign over the window | `sign_consistency_window` | Buckley & Voorhees 2000 |
| 19 | rank stability | curve crossings per pair | `crossings` | DataDecide §3.2 |
| 20 | rank stability | sign persistence to the final | `sign_persistence` | Buckley & Voorhees 2000 |
| 21 | rank stability | rank settling time (τ ≥ 0.8 from then on) | `settling_time` | DataDecide |
| 22 | rank stability | DA of early checkpoints vs the proxy final | `da_ckpt_mean`, `da_ckpt_half` | Heineman 2025; DataDecide |
| 23 | rank stability | split-half reliability, Spearman–Brown corrected | `split_half` | Spearman 1910; Brown 1910 (classic) |
| 24 | curve shape | monotonicity (Spearman of score vs step) | `monotonicity` | FineTasks; Hofmann et al. 2025, *Fluid*, [arXiv:2509.11106](https://arxiv.org/abs/2509.11106) |
| 25 | curve shape | step monotonicity, normalised total variation | `monotonicity_steps`, `total_variation` | Heineman 2025 (`snr/stats.py`); Fluid §4.2 |
| 26 | curve shape | lag autocorrelation | `autocorr` | E2LM 2025, [arXiv:2506.07731](https://arxiv.org/abs/2506.07731) (full text) |
| 27 | curve shape | learning gain / noise; late slope / residual | `gain_over_noise`, `late_slope_to_noise` | Madaan 2024 |
| 28 | curve shape | non-randomness, z above chance, share of variants above chance | `nonrandom`, `z_above_chance`, `emerged_share` | FineTasks; Du et al. 2024, [arXiv:2403.15796](https://arxiv.org/abs/2403.15796); Schaeffer et al. 2023, [arXiv:2304.15004](https://arxiv.org/abs/2304.15004) |
| 29 | curve shape | aggregate test information q(1−q) | `item_information` | Rodriguez et al. 2021, ACL; Vania et al. 2021, [arXiv:2106.00840](https://arxiv.org/abs/2106.00840) |
| 30 | small ladder | DA of the previous rung vs the proxy; mean over rung pairs | `prev_rung_da`, `ladder_da` | DataDecide; RegMix |
| 31 | small ladder | pair sign consistent across rungs; correlation of gaps | `rung_sign_consistency`, `rung_gap_corr` | RegMix (rank invariance) |
| 32 | small ladder | projected crossovers by the reference's N | `projected_flip_rate` | DataDecide (crossovers) |
| 33 | small ladder | size monotonicity; scale gain / noise | `size_monotonicity`, `scale_gain_over_noise` | Bhagia 2024; Schaeffer et al. 2024, [arXiv:2406.04391](https://arxiv.org/abs/2406.04391) |
| 34 | pseudo-reference | DA against the 1B rung (the largest below the reference) | `pseudo_ref_da` | RegMix; DataDecide |
| 35 | agreement | ranking vs the language's BPB ranking; score–BPB correlation over training | `bpb_rank_agreement`, `bpb_corr_training` | Thrush et al. 2024, [arXiv:2409.05816](https://arxiv.org/abs/2409.05816); Brandfonbrener et al. 2024, [arXiv:2411.12925](https://arxiv.org/abs/2411.12925) |
| 36 | agreement | DA vs the language's other benchmarks; vs the same benchmark in other languages | `language_consensus`, `benchmark_consensus` | Perlitz et al. 2024, BenchBench, [arXiv:2407.13696](https://arxiv.org/abs/2407.13696) |
| 37 | agreement | corrected item-total correlation (common factor) | `item_total_corr` | Ruan et al. 2024, [arXiv:2405.10938](https://arxiv.org/abs/2405.10938) |
| 38 | control | pairs, items, chance level | `n_pairs`, `n_items`, `chance` | – |
| 39 | SNR grid | each of AllenAI's 22 signals over each of 6 noises (132 ratios), and each signal and noise alone | `snr__<signal>__<noise>`, `signal__*`, `noise__*` | Heineman 2025 (`snr/snr_variants.py`) |
| 40 | noise | benchmark (k-fold) noise, relative and absolute, closed form | `noise__kfold_rel`, `noise__kfold_abs` | the 2026-04 slides (Benchmark noise) |
| – | separation | almost stochastic dominance ε | – (5 samples per variant is too few) | Dror, Shlomov & Reichart 2019, ACL |
| – | needs log-probs | CORRECT_PROB, NORM_CORRECT_PROB, margin | – | DataDecide App. B; Schaeffer 2024 |
| – | needs items | IRT discrimination/difficulty, tinyBenchmarks, metabench | – | Polo et al. 2024, [arXiv:2402.14992](https://arxiv.org/abs/2402.14992); Kipnis et al. 2024, [arXiv:2407.12844](https://arxiv.org/abs/2407.12844) |
| – | needs formats | prompt/format sensitivity | – (rq00's `rf_` twins could serve) | Alzahrani et al. 2024, [arXiv:2402.01781](https://arxiv.org/abs/2402.01781) |
| – | needs extra evals | expert-trace token proxies, rBridge | – | Patel 2026; Koh et al. 2025, [arXiv:2509.21013](https://arxiv.org/abs/2509.21013) |

### The noises of the SNR grid

- `ckpt_rel`: AllenAI's checkpoint noise, the mean over models of the std over
  the noise window, over the mean window score (rule 4's window). `ckpt_abs`:
  the same without the division.
- `tukey_depth`, `projection_depth`: the noises that AllenAI's two depth
  aggregators pair with their signals.
- `kfold_rel`, `kfold_abs`: the benchmark noise, i.e. the relative std of a
  model's accuracy across k folds of the items, averaged over models. The
  ladder report has no per-item outputs, so the noise is taken in closed form.
  For n binary items with accuracy p, a random partition into k folds of n/k
  items gives fold accuracies m_i with E[(1/k) Σ (m_i − p)²] = p(1−p)(k−1)/(n−1)
  exactly (hypergeometric sampling). We use its root with k = 5. Changing k
  multiplies every task's noise by the same factor, so no ranking moves.

### How the search is kept honest

Scoring ~250 statistics against 30 truths in ~50 subsets and ~7,500 filters
would surface large correlations by chance alone. `search.py` therefore
splits the (benchmark, language) clusters into two halves. It ranks every
configuration on one half and tests the best once on the other half: a
one-sided cluster permutation test, with Benjamini–Hochberg over everything
tested. Correlations are taken within each (proxy size, fraction) stratum, so
a statistic that only grows with training or size cannot score.

## 3. Why every surrogate might stay weak

- **DA is itself noisy.** With few pairs per task, DA is coarse, and it moves
  when the reference is read one checkpoint earlier. `catalogue.py` measures
  this ceiling (`da_retest.csv`): the Spearman ρ between DA-size and DA-size
  with the reference read at 90 % of its run. By Spearman's attenuation
  formula, an observed ρ cannot exceed roughly √(reliability of DA).
- **Noise cannot see crossovers.** Statistics of separation, stability and
  curve shape describe the proxy's own noise. A DA miss caused by a real rank
  change between the proxy and the reference is invisible to them. Only the
  small-ladder family (30–34) looks at how rankings move with size.
- **Range restriction.** The gate removes the near-chance tasks, where every
  statistic and DA are low together. That low tail drives the R = 0.79 in
  Heineman 2025.
