---
title: "RQ5 · Design decisions"
---

# RQ5 · Which proxy sizes read each design decision?

**The question.** Do small models rank a design choice — depth, the language
list, the sampling temperature, the second language — the way the reference
does? Does the answer change with the number of languages, and how early in
the run can you read it?

**How we measure it.** For every decision we pair runs that differ only in
it, and compare the proxy's preference with the reference's across benchmarks
and languages. We also measure how large each decision's effect is against
seed noise at the reference.

## Explore

Is there a decision to read? Effect of each decision, in units of seed noise:

<div class="viz" data-viz="effect"></div>

Agreement with the reference, by proxy size:

<div class="viz" data-viz="interventions"></div>

How early can you decide?

<div class="viz" data-viz="early"></div>

## What we find

<!-- highlight: rq05_design_decisions -->

!!! success "Takeaway"
    Before an ablation, ask whether its effect will be larger than seed
    noise. Large effects (temperature on bits per byte) are readable from
    175M; small ones (depth on benchmarks) are coin flips at every proxy size.

[Full analysis →](../signal-noise/design_decisions.md)
