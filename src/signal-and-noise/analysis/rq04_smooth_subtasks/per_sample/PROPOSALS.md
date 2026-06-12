# Per-sample SNR subset search — design options

For each language-specific benchmark (`arc_es`, `xnli_de`, `belebele_zh`, ...)
we want the subset of _samples_ (individual instances, identified by
`doc_id` / `doc_hash` in the lm-eval `samples_*.jsonl` files) that
maximizes the signal-to-noise ratio of the resulting score.

The search space is `2^N` (N ≈ 1.1k for `arc_*`, ≈ 14k for `mmlu_*`), so
exhaustive enumeration is out. The SNR primitive is the same as the
rest of the codebase (`snr.metrics.signal_to_noise_ratio` over per-mix
last-5-ckpt arrays). What varies between options is _how subsets are
proposed_.

**A, B, C and D are all implemented** in
[`analysis/rq04_smooth_subtasks/smooth_subtasks_per_sample.py`](../../../analysis/rq04_smooth_subtasks/smooth_subtasks_per_sample.py),
selectable with `--method` and each writing to its own dir under
`per_sample/` (dir names in the headings below). **D is the default
and the recommended starting point**; A is the upstream baseline, B
the non-greedy check, C the IRT-based filter. E (random search) is not
implemented — the random-order baseline curve covers that role.

## A. Per-sample SNR + sort (greedy by individual rank) — `greedy_snr_rank`

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

## B. Forward greedy selection — `forward_greedy`

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
- **Implemented** with the candidate pool and budget exposed as
  `--b-pool` / `--b-budget` (defaults 500 / 100, smaller than the
  top-1000 sketched above to keep per-task runtime modest); raise them
  per-benchmark if the cumulative curve hasn't plateaued at the budget.

## C. IRT discrimination filtering — `irt_discrimination`

Fit a 2PL item-response-theory model (sample = item, ckpt = examinee).
Keep samples with high discrimination parameter `a_i`, then run Option A
on the survivors to order them ("C then A").

- Principled, model-based.
- **Caveat (this corpus):** the examinees are checkpoints — at most
  ~9 (mix, seed) runs × 5 ckpts per size, and highly correlated (one
  training trajectory). 2PL discrimination needs many _independent_
  examinees, so `a_i` estimates here are noisy. Treat C as exploratory,
  not authoritative.
- **Correction:** an earlier draft claimed the upstream
  `snr.mask_analysis` ships IRT utilities and that "AllenAI uses this
  internally." That is not borne out by the released code —
  `get_subtask_utils` only _parses_ subtasks, and the upstream
  subset-selection analysis (`compute_error_by_subtask`) is greedy
  SNR-sort (Option A), with **no IRT anywhere**. C is therefore a
  genuine extension, not a reproduction.
- **Implemented** via the `girth` package (`twopl_mml`), lazily
  imported so A/B/D run without it. Items with no across-checkpoint
  variance are dropped before the fit; `--c-keep-frac` (default 0.5)
  sets the discrimination quantile kept.

## D. Variance prefilter + Option A — `variance_prefilter` _(default)_

Drop "dead" samples — those whose per-mix mean accuracy is constant
across all mixes (signal=0 → no information about which mix is
better). Then run Option A on the survivors.

- Much smaller `N` after filter (typically 30–60% of samples drop).
- Option A's weakness (interactions) matters less because the
  candidate pool is already informative.
- Cheap: one variance computation + one sort per (task, size).
- The prefilter threshold is a knob (`min_signal`); default 0
  (strictly constant) is the most conservative.

## E. Random / black-box search _(not implemented)_

Sample R random subsets of varying sizes; or simulated annealing /
genetic search.

- Can find non-greedy optima.
- Slow; results depend on R.
- Mostly useful as a _baseline_ to validate A/B/D, not as the
  production method. The cumulative plots already include a
  random-order baseline curve, which serves a similar sanity-check
  role at almost zero cost.

## Why D is the default

D is the fastest path to a usable per-(language, benchmark) ranked
sample list and matches the upstream subset-search semantics (A) while
dropping uninformative samples first. If a benchmark's cumulative-SNR
curve doesn't plateau (i.e., adding samples keeps helping past some
large N, suggesting the greedy rank missed interactions), compare
against B (`forward_greedy`) or C (`irt_discrimination`) on that
specific benchmark. All four run from the same entry point:

```bash
# all four methods for a pool
python analysis/rq04_smooth_subtasks/smooth_subtasks_per_sample.py --pool seeds_28_1797
# just one, with a bigger forward-greedy budget
python analysis/rq04_smooth_subtasks/smooth_subtasks_per_sample.py --pool seeds_28_1797 \
    --method forward_greedy --b-pool 1000 --b-budget 200
```

## Why the argmax is so small

The committed Option-D run picks a median best subset of only ~2.5% of
items (e.g. 12 of 500 for `xcopa_sw` 1B). That is a statistical artifact,
not a benchmark anyone would trust to evaluate a model. Two things make the
`signal/noise` argmax collapse to a handful of items:

- **The objective is a ratio that spikes early.** The top few items have
  large cross-mix dispersion; as more are averaged in, the signal
  (numerator) regresses toward the mean faster than the noise (denominator)
  falls, so `signal/noise` peaks at small `N` and then declines slowly. The
  cumulative curve has a _broad near-peak plateau_ — a much larger subset
  sits within a hair of the peak.
- **The per-item ranking is mostly noise.** Each item's SNR rests on only
  ~5 ckpts × 3 mixes; cross-size rank Spearman is ≈ 0.05 (see
  [`variance_prefilter/analysis/highlights.md`](../variance_prefilter/analysis/highlights.md)).
  A noisy ranking makes the greedy top-prefix _overfit to a few lucky
  items_ — exactly the subset that will not replicate.

A larger, trustworthy subset therefore needs two moves — **relax the
selection rule** (stop taking the knife-edge argmax) and **denoise the item
scores** (so the ranking is real) — plus a way to **prove** trust. The
levers below are ordered cheapest-first.

## Getting a larger, trustworthy subset

### Lever 1 — Relax the selection rule _(cheapest; reuses the cumulative curve)_

- **Still-beats-full-set** — report the _largest_ subset whose SNR ≥ the
  full-set SNR. Answers "how many items can I keep and still beat evaluating
  on everything?"; almost always a large fraction, not 2.5%.
- **1-SE / ε-plateau** — take the _largest_ `N` within one standard error
  (or a fixed ε) of the peak SNR, à la the lasso "1-SE rule". Trades a sliver
  of SNR for a much bigger, more stable subset.
- **Target size** — fix a practitioner-friendly size (e.g. 25–50% of items)
  and report the SNR retained there.

These read the `cumulative_snrs` array the sweep already produces, so they
need no extra compute — only a different argmax rule in
[`smooth_subtasks_per_sample.py`](../../../analysis/rq04_smooth_subtasks/smooth_subtasks_per_sample.py).

### Lever 2 — Denoise the per-item informativeness _(fixes the root cause)_

- **Pool across seeds/sizes** — average each item's SNR over the 3 seeds
  (and optionally the 4 sizes) before ranking, so the order reflects signal,
  not luck. A flatter optimum yields a larger subset.
- **Shrinkage / IRT** — shrink each item's SNR toward the benchmark mean
  (empirical Bayes), or use the **IRT discrimination** score (Option C,
  already implemented) as a smoother, model-based item-quality estimate.
- **Group selection** — select at the granularity of subtask / topic /
  difficulty bin instead of individual items. This is exactly Cases 1–3,
  which are _partially_ stable across scale where per-item selection is ≈0;
  the honest trustworthy unit may be item-groups, not items.

### Lever 3 — Make trust measurable _(validation)_

- **Held-out seed-pool CV** — select on `seeds_28_1797`, then measure the
  subset's SNR _and_ decision accuracy on the held-out `seeds_1904`. A 2.5%
  subset collapses out-of-sample; selecting for held-out performance pushes
  toward larger, robust subsets and yields a trust number to report.
- **Stability selection** — bootstrap the ckpts/seeds, run the sweep many
  times, keep items chosen in ≥ X% of runs. The consensus subset is bigger
  and reproducible by construction.

### Lever 4 — Change the objective

- **Decision accuracy, not raw SNR** — decision accuracy (does the subset
  rank model pairs correctly?) is what practitioners want, and it _saturates_
  rather than spiking, so the smallest subset hitting a DA threshold needs
  many more items than the SNR argmax and is far easier to defend.

### Recommendation

Combine one lever per layer: **pool seeds (Lever 2)** → **report the largest
subset within 1-SE that still beats the full set (Lever 1)** → **validate SNR
and decision accuracy on the held-out seed pool (Lever 3)**, with the
decision-accuracy curve shown alongside SNR (Lever 4). That yields a subset
that is large, sits on the SNR plateau, provably beats full-benchmark
evaluation, and carries an out-of-sample trust number. If even the denoised
per-item subset fails to transfer, that is the empirical case for
**group-level selection** (Lever 2) as the trustworthy unit.

Lever 1 is nearly free (a new argmax rule over the existing curve); the rest
require a cluster re-run, since they touch the acc matrix, the seed pools, or
decision accuracy.
