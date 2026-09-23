---
title: "RQ8 · Subsets"
---

# RQ8 · Can a subset beat the full benchmark?

**The question.** Averaging many languages or subjects adds noisy parts to
good ones. Does a subset of a benchmark's languages, subjects or items give a
higher SNR than the full set — beyond what a random subset gives?

**How we measure it.** We rank a benchmark's subtasks by their SNR, add them
one by one, keep the best prefix, and compare it with 100 random subsets of
the same size.

## Explore

<div class="viz" data-viz="subsets"></div>

## What we find

<!-- highlight: rq08_subset_selection -->

!!! success "Takeaway"
    A curated subset can raise a benchmark's SNR and cut its cost, but only
    about a third of the gains beat random subsets: validate a subset on runs
    you did not select it on.

[Full analysis →](../signal-noise/subset_selection.md)
