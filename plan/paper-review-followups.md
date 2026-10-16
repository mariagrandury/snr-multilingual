# Paper review follow-ups: the numbers behind each fix

The internal review of the 13-12 Overleaf version (W1–W10) asked for analyses
before the text changes. Every number below is read from a committed table of
the 2026-10-10/11 refresh (pool `predictivity`, seed 1904 unless stated);
nothing in the paper has been changed yet. Paths are under
`src/signal-and-noise/analysis/`.

## W2/W3 — one canonical population for §5

The paper's Figure 3 keeps only tasks whose median DA-size or DA-ckpt is at
least 0.66 and plots `reliability_macro` (the mean over tasks). On that same
scale, single-axis DA-size over all pairs:

| proxy | every task above chance | tasks | Fig. 3 (in sample) | same half | cross-fitted |
|---|---|---|---|---|---|
| 90M | 0.59 | 1,274 | 0.64 | 0.61 | 0.60 |
| 175M | 0.55 | 1,316 | 0.61 | 0.58 | 0.57 |
| 350M | 0.57 | 1,351 | 0.63 | 0.61 | 0.60 |
| 600M | 0.55 | 1,407 | 0.60 | 0.58 | 0.58 |
| 1B | 0.57 | 1,461 | 0.64 | 0.60 | 0.59 |

Sources: `rq02_decision_accuracy/pretraining/predictivity/rq2_da_all_mono_axis.csv`
(panel DA-size, group `all pairs`; the pair-weighted `reliability` column reads
0.54–0.57) and `rq2_da_all_above_66_either_crossfit_transformation_mono_axis.csv`
(readings `in-sample`, `in-sample (half)`, `cross-fitted`; 40 splits of the
pairs, stratified by axis, read both ways).

On the unfiltered population, DA-goal of every proxy checkpoint is flat at
0.53–0.59 (all tenths, 90M–1B), while the 1.7B run's own checkpoints rise from
0.57 to 0.77. DA-ckpt at 90 % of the run is 0.81–0.88 (350M highest, 0.88;
90M and 175M 0.84). Same CSV, panels DA-goal and DA-ckpt.

**Implication.** The selection itself (cross-fitted against the same half)
adds only 0.01–0.02; the rest of the gap to Figure 3 is reading DA on half of
the pairs. Report the unfiltered line (0.55–0.59) and the cross-fitted one
(0.57–0.60) beside the filtered figure. Added to the review copy's appendix
(Table `tab:app_rq02_selection`).

## W4 — pairs and uncertainty per design axis (Figure 3a)

| axis | pairs | DA-size, 90M–1B (Fig. 3, macro) | lowest lower band |
|---|---|---|---|
| all pairs | 32 | 0.60–0.64 | 0.55 |
| language count | 27 | 0.60–0.63 | 0.57 |
| data scheme | 12 | 0.49–0.53 | 0.44 |
| depth | 10 | 0.47–0.59 | 0.43 |
| temperature | 4 | 0.73–0.75 | 0.69 |
| activation | 3 | 0.52–0.57 | no band |
| optimizer | 3 | 0.43–0.58 | no band |

Source: `rq2_da_all_above_66_either_transformation_mono_axis.csv` (`n_units` =
design pairs, `lo`/`hi` = 90 % leave-one-family-out jackknife of the pooled,
pair-weighted DA-size `reliability`, not of the macro mean).

**Implication.** The temperature lead is real (its band stays above 0.69 while
all pairs stay at 0.55–0.64) but rests on 4 pairs. Activation and optimizer
(3 pairs each) cannot be ranked against the others; say so or drop their lines
from the claim. "350M highest" holds only late in the run (DA-ckpt at 90 %).
Added to the review copy's appendix (Table `tab:app_rq02_axis_pairs`).

## W5 — the attainable ceiling (single-seed reference)

Two seeds of the same designs, ranked against each other (test-retest), on the
replicated deep cells:

| size | language | test-retest DA | proxy seeds vs 1.7B |
|---|---|---|---|
| 175M | en | 0.59–0.65 | 0.55, 0.55, 0.64 |
| 600M | en | 0.61–0.62 | 0.60, 0.56, 0.61 |
| 1B | en | 0.53–0.60 | 0.60, 0.57, 0.56 |
| 1B | ru | 0.75–0.83 | 0.83, 0.79, 0.81 |

Source: `rq02_decision_accuracy/README.md`, auto block `seed-uncertainty`
(`seed_uncertainty_da_all_seed_null.csv`). With the 1B models as the reference,
the seed of the reference alone moves a proxy's single-axis DA-size by up to
0.13 (175M: 0.66 / 0.53 / 0.56 against the three 1B seeds, 0.56 against 1.7B;
`tables/app_rq10_reference_choice.tex`).

**Implication.** On English, a design's ranking is about as reproducible across
seeds as across sizes: the 51–57 % agreement is close to the test-retest
ceiling, so "proxies agree poorly" partly means "single-seed decisions are
noisy at every size". This is the strongest framing of the RQ2 result and
should be in the main text. A second 1.7B seed on two cells would give
DA-size and DA-goal the same ceiling (README follow-up).

## W6 — the 600M per-language BPB anomaly

Per-language BPB, share of (pair, language) comparisons whose 600M sign differs
from 1.7B, by design axis:

| axis | size-matched | 600M flips | other proxies |
|---|---|---|---|
| depth | no (shallow +3.7 %) | 95 % | 4–27 % |
| activation | yes | 83 % | 4–13 % |
| optimizer | yes | 70 % | 62–70 % (every size) |
| language count | yes | 15 % | 12–17 % |
| temperature | yes | 1 % | 1–3 % |
| multi-axis, depth held fixed | yes | 14 % | 9–15 % |

With depth held fixed, BPB DA-size is 0.91, 0.90, 0.85, 0.86, 0.88 from 90M to
1B: 600M is not an outlier. Re-drawing only the deep model's seed at 600M moves
the depth flips at K=50 from 50/50 (seed 1904) to 43/50 (313) and 15/50 (64).
Source: `rq03_noise_and_snr/README.md`, auto block `flip-attribution`
(`flip_attribution_da_size_by_axis_both_axes.csv`).

**Implication.** The 600M anomaly is the depth (not size-matched) and
activation pairs at one seed draw. In §6, present it as a seed-noise and
size-matching case rather than as a failure of the surrogates, or drop the
example.

## W7 — bBPB against the accuracy decision

DA-size on every gated benchmark task, multi-axis pairs, paired tasks:

Paired over the 360–542 tasks where both readings exist, bBPB → 1.7B accuracy
minus accuracy → 1.7B accuracy is +0.10, +0.06, +0.06, +0.05, +0.03 (90M–1B),
bBPB better on 57–71 % of tasks, Wilcoxon p < 1e-7 (columns `gain`,
`bbpb_better`, `p`). The unpaired means:

| proxy | accuracy → 1.7B accuracy | bBPB → 1.7B accuracy | bBPB → 1.7B bBPB |
|---|---|---|---|
| 90M | 0.50 | 0.59 | 0.69 |
| 175M | 0.52 | 0.58 | 0.64 |
| 350M | 0.52 | 0.58 | 0.68 |
| 600M | 0.54 | 0.58 | 0.65 |
| 1B | 0.56 | 0.59 | 0.67 |

Source: `bench_bpb_da_size_summary_multi_axes.csv` (group `all benchmarks`).
Where accuracy is at chance at the proxy, bBPB → 1.7B bBPB reads 0.60–0.63
against 0.46–0.50 for accuracy (`bench_bpb_da_size_by_acc_gate_multi_axes.csv`).

**Implication.** The fair claim is "+0.03 to +0.09 on the accuracy decision,
largest at the smallest proxy"; the 0.64–0.69 is agreement with an easier
target (the reference's own bBPB) and should be labelled as such.

## W1 — tokens of a language and decision reliability

No analysis supports "per-language tokens impact decision reliability" for the
benchmarks. By language tier (multi-axis DA-size, every task above chance), the
languages that enter only at K=50, which have the fewest tokens, rank the
designs *best*:

| tier (languages) | DA-size, 90M–1B |
|---|---|
| K=8 (8) | 0.57–0.58 |
| K=15 (7) | 0.57–0.60 |
| K=30 (15) | 0.57–0.60 |
| K=50 (20) | 0.64–0.69 |

Source: `reliability_da_size_by_language_tier_multi_axes.csv` (variant `all`).
For per-language BPB, `rq02_da_vs_train_tokens` finds the same: the tokens seen
do not set the level (90M reaches 0.90 at 1C on 0.02B tokens; 600M stays at
0.70–0.74 up to 0.50B).

**Implication.** Keep tokens as a factor of *measurability* (§4, where it is
supported) and drop it as a factor of decision reliability, or state the
negative result.

## W8 — surrogates by scoring and the monolingual check

On accuracy alone, early-checkpoint agreement reaches ρ 0.28–0.39 and the
relative-standard-deviation SNR 0.33–0.40; on the bBPB tasks 0.57–0.68 and
0.46–0.58 (`rq04_surrogates`, figure `app_rq04_surrogates_by_scoring`). The
pooled Figure 4 levels are therefore driven by bBPB.

Monolingual check (`rq13_english_only/allenai_snr.py`): ρ(log SNR, DA-size) on
the nine OLMES benchmarks is 0.89 on DataDecide, 0.46 at K=1, 0.43 at K=2,
0.04 at K=15 and −0.10 at K=30
(`english_only_allenai_snr_da_summary.csv`).

**Implication.** Report the accuracy-only correlations in the main text and
cite the monolingual check as the evidence that the weak surrogates come from
multilingual training, not from the implementation.

## Not needing new analyses

W9 (size vs training recommendations), W10 (limitations) and the minor
inconsistencies are text fixes; the numbers they need are above or in the
existing appendices.
