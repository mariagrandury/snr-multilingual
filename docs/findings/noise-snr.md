---
title: "RQ3 · Noise and SNR"
---

# RQ3 · How noisy is each benchmark?

**The question.** How much does a score move when only the seed or the
checkpoint changes, and how does that compare with the differences between
recipes?

**How we measure it.** Checkpoint noise is the spread of a run's scores over
its late checkpoints; seed noise is the spread across seed replicates of one
recipe. Signal is the spread across recipes. We compute 22 SNR definitions
from the literature per task and size; this view shows the one rq04 finds
best (`rel_star_discrepancy`).

## Explore

<div class="viz" data-viz="snr"></div>

Try the idea on toy numbers:

<div class="viz" data-viz="snrToy"></div>

## What we find

- Only above-chance tasks receive an SNR. The median task has
  log₁₀ SNR ≈ 0.2: the spread between recipes is only about 1.5× the
  checkpoint noise.
- Seed noise and checkpoint noise are not interchangeable: a ranking of SNR
  definitions that holds across checkpoints does not survive a change of
  seed (see [RQ4](surrogates.md)).

!!! success "Takeaway"
    Estimate noise on your own runs (several late checkpoints, and at least
    one seed replicate) before trusting a difference between two recipes.

[Full analysis →](../signal-noise/noise_and_snr.md)
