---
title: "RQ2 · Decision accuracy"
---

# RQ2 · Does a small model rank recipes like the large one?

**The question.** You train recipes at a small size, pick the best, and scale
it up. How often does the small model pick the recipe the large model would
pick? And how early in a run can you read that answer?

**How we measure it.** Decision accuracy (DA) is the share of recipe pairs
ordered the same way by the proxy and by the 1.7B reference. **DA-size**
compares final checkpoints across sizes; **DA-ckpt** compares an early
checkpoint with the final one of the same run. 0.5 is a coin flip.

## Explore

How early and how small can you go?

<div class="viz" data-viz="da"></div>

Which benchmarks, in which language?

<div class="viz" data-viz="daBench"></div>

## What we find

<!-- highlight: rq02_decision_accuracy -->

!!! success "Takeaway"
    Rank recipes on per-language bits per byte first; it agrees with the
    reference from 350M. Use benchmarks as a second check, one family at a
    time, and never stop a run on a single early checkpoint: early
    agreement can reverse.

[Full analysis →](../signal-noise/decision_accuracy.md)
