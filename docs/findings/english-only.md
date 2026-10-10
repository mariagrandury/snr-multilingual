---
title: "RQ13 · English-only check"
---

# RQ13 · Do the English-only models do better on English benchmarks?

**The question.** The ladder's L1 models (the paper's K = 1) are trained on
English only; every other model spends half of its tokens on other
languages. If the training runs are sound, the English-only models should do
clearly better on English benchmarks than multilingual models of the same
size: higher scores, above chance earlier, and at least as reliable in their
decisions. This page is a sanity check of the ladder itself, a look at which
popular English benchmarks scale without a multilingual mixture, and a
comparison with AllenAI's DataDecide models on the same English tasks.

**How we measure it.** The English-only baseline model against the baseline
model of the same size and depth at every other language setting (L = 2, 8,
15, 30, 50), final checkpoints, 90M–1.7B, on the English tasks above chance
at each size. Two more English-only models change only the English corpus:
DCLM without its educational filter, and FineWeb.

## Does each expectation hold?

<div class="viz" data-viz="englishVerdict"></div>

The same verdict as a scorecard: one card per expectation in plain words,
green when it holds, amber when it holds only partly, grey when the two
sides cannot be compared fairly.

<div class="viz" data-viz="englishScorecard"></div>

## How large is the English advantage?

<div class="viz" data-viz="englishGap"></div>

The average above, one benchmark at a time. Pick an English benchmark (★ marks
the widely used ones, and English BPB is at the top): the bold line is the
English-only model, the thin lines the multilingual ones of the same size.

<div class="viz" data-viz="englishBench"></div>

## Which English benchmarks scale?

We repeat the [scaling analysis](scaling.md) on the three English-only models,
one per English corpus, under the same filter and the same fits. In each
corpus 16 of the 33 English accuracy tasks above chance (48 %) are predictable
along both size and training. That half includes the benchmarks used most in
model development: ARC (easy and challenge), HellaSwag and OpenBookQA scale in
all three corpora, together with the reformulated (cloze) versions of MMLU,
Global-MMLU, Belebele and CommonsenseQA, whose original letter format never
leaves chance. Global PIQA is the only widely used benchmark above chance that
does not scale. Each cell is a single model's fit, so the levels are not
comparable with the medians over L on the [scaling page](scaling.md).

Each row is a benchmark, each column an English corpus, and each cell says in
words how that benchmark scales there; click a row to see its scores per size.

<div class="viz" data-viz="englishRegimes"></div>

## What we find

- **The English-only models are better at English, by a small margin.** The
  deep English-only model wins more than half of the English accuracy tasks
  against every other L at every size drawn (25 of 25 comparisons), but the
  mean gap at 1.7B is 1.3 points; the shallow one wins 30 of 30, by 0.9 points
  at 1.7B.
- **English bits per byte separates them most cleanly.** The English-only
  model is lower in all 30 comparisons at both depths: 0.782 against 0.810 at
  1.7B (deep).
- **Part of the lead is beyond seed noise.** The deep English-only model is
  ahead by more than two seed standard deviations in 45–55 % of the
  comparisons, and behind by as much in 2–7 %.
- **Clearing chance earlier holds per task, not per share.** It clears chance
  at a smaller size on 98 (task, L) pairs and at a larger one on 30, but its
  share of English tasks above chance is ahead in only 15 of 25 comparisons.
- **Scaling is about as regular.** The 1.7B English BPB forecast from
  90M–1B misses by 1.0 % at L1 against 1.1–1.4 % elsewhere; the median R² of
  the English accuracy fits is 0.86 against 0.81–0.93.
- **Popular benchmarks either scale or never leave chance.** ARC, HellaSwag,
  OpenBookQA and the reformulated MMLU scale in every corpus; letter-format
  MMLU stays at chance; Global PIQA is above chance but does not scale.
- **Decisions are more accurate at L1, but they are different decisions.**
  With four model families per L, a small L1 model ranks them like 1.7B does
  in 0.58–0.63 of the pairs against 0.44–0.53 at L = 2, 15 and 30; on the one
  decision every L shares (deep against shallow), 0.51 against 0.44–0.47.
- **Next to DataDecide.** L1's ranking agreement is below, inside and above
  DataDecide's range in 5, 6 and 2 of 13 format-matched (task, size) cells,
  median ratio 0.90.

!!! success "Takeaway"
    The ladder passes its sanity check: English-only models score higher on
    English, by a point or two at 1.7B, and their English BPB is lower
    everywhere. The widely used English benchmarks behave as expected once
    the gate is applied: they scale in every corpus in their cloze form, and
    the letter-format originals stay at chance at these sizes.

[Full analysis →](../signal-noise/analysis.md)
