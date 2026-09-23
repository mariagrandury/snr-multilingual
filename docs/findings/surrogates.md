---
title: "RQ4 · Cheap predictors"
---

# RQ4 · Which cheap statistic predicts decision accuracy?

**The question.** Decision accuracy needs the large model you want to avoid
training. Can a number computed on the small model alone — an SNR, its
signal or noise part, early agreement, fit quality, the margin above chance —
tell you in advance whether its decision will hold?

**How we measure it.** For each proxy size we correlate each statistic with
the tasks' decision accuracy against the reference (Spearman ρ).

## Explore

<div class="viz" data-viz="surrogates"></div>

The best SNR definition, language by language:

<div class="viz" data-viz="bestVariant"></div>

## What we find

<!-- highlight: rq04_surrogates -->

!!! success "Takeaway"
    Use SNR to *shortlist* benchmarks, not to certify a decision. Prefer a
    discrepancy-based definition, and check that the scaling fit on your
    proxy rungs is good: fit quality predicts decision accuracy at 600M–1B.

[Full analysis →](../signal-noise/surrogates.md)
