# Per-sample SNR subset search — design options

For each language-specific benchmark (`arc_es`, `xnli_de`, `belebele_zh`, ...)
we want the subset of *samples* (individual instances, identified by
`doc_id` / `doc_hash` in the lm-eval `samples_*.jsonl` files) that
maximizes the signal-to-noise ratio of the resulting score.

The search space is `2^N` (N ≈ 1.1k for arc_*, ≈ 14k for mmlu_*), so
exhaustive enumeration is out. The SNR primitive is the same as the
rest of the codebase (`snr.metrics.signal_to_noise_ratio` over per-mix
last-5-ckpt arrays). What varies between options is *how subsets are
proposed*.

The currently-implemented option is **D** (variance prefilter +
Option-A sort). The others are documented here so we can revisit them
if D's curves don't plateau or we want a non-greedy sanity check.

## A. Per-sample SNR + sort (greedy by individual rank)

For each sample independently, compute its SNR (signal = range of
per-mix mean acc; noise = std of last-N ckpt scores pooled). Sort
samples by SNR descending; sweep cumulative subset 1..N where the
"combined" score at each step is the per-mix mean of the included
samples; pick argmax.

- Matches `analysis/smooth_subtasks.py` semantics exactly — this is
  what the upstream AllenAI paper does.
- `O(N)` per (task, size); single-pass.
- Misses sample-sample interactions (two individually-mediocre samples
  that together separate mixes well).
- For binary `acc`, many samples have constant scores across mixes →
  per-sample signal=0, SNR is 0/0; treat as -inf and let them sort to
  the bottom.

## B. Forward greedy selection

Start with the best single sample; at each iteration add the sample
whose addition maximally increases combined-subset SNR; stop when SNR
drops or after a budget K.

- Captures interactions A misses; usually finds higher final SNR.
- `O(N · K · cost-of-snr)`. For mmlu_de (N=14k, K=200) that is ~3M SNR
  evaluations per (size, mix); tractable only with the SNR formula
  vectorised over the candidate pool, and probably needs the candidate
  pool capped to the top-1000 by Option A.
- More search-y, harder to interpret the resulting subset (no clean
  per-sample SNR ranking).

## C. IRT discrimination filtering

Fit a 2PL item-response-theory model (sample = item, ckpt = examinee).
Keep samples with high discrimination parameter `a_i`. The upstream
`snr.mask_analysis` already has IRT utilities (`get_subtask_utils`),
and `compute_subtask_snr` is partly built on this.

- Principled, model-based; this is what AllenAI uses internally for
  instance-level analysis.
- Requires an IRT fit per (size, language, benchmark) — adds a
  dependency and a tuning surface.
- Usually paired with Option A on the IRT-filtered survivors to pick
  the actual subset, so it's "C then A," not C alone.

## D. Variance prefilter + Option A  *(implemented)*

Drop "dead" samples — those whose per-mix mean accuracy is constant
across all mixes (signal=0 → no information about which mix is
better). Then run Option A on the survivors.

- Much smaller `N` after filter (typically 30–60% of samples drop).
- Option A's weakness (interactions) matters less because the
  candidate pool is already informative.
- Cheap: one variance computation + one sort per (task, size).
- The prefilter threshold is a knob (`min_signal`); default 0
  (strictly constant) is the most conservative.

## E. Random / black-box search

Sample R random subsets of varying sizes; or simulated annealing /
genetic search.

- Can find non-greedy optima.
- Slow; results depend on R.
- Mostly useful as a *baseline* to validate A/B/D, not as the
  production method. The cumulative plots already include a
  random-order baseline curve, which serves a similar sanity-check
  role at almost zero cost.

## Why D first

D is the fastest path to a usable per-(language, benchmark) ranked
sample list and matches the upstream subset-search semantics. If a
benchmark's cumulative-SNR curve doesn't plateau (i.e., adding
samples keeps helping past some large N, suggesting the greedy rank
missed interactions), B or C are the natural next step on that
specific benchmark — they're not worth running on all 98
language-benchmarks up front.
