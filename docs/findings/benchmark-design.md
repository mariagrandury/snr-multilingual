---
title: "RQ9 · Benchmark design"
---

# RQ9 · What makes a benchmark high-SNR?

**The question.** Is it curation (native vs translated), format, answer
count or item length that makes some benchmarks separate models cleanly?

**How we measure it.** We annotate every family with its design features and
test whether they explain the SNR of the families that clear chance.

## Explore

<div class="viz" data-viz="design"></div>

## What we find

<!-- highlight: rq09_benchmark_design -->

!!! success "Takeaway"
    For small-model ablations, answer count matters most, and it acts through
    the chance gate: two-option and minimal-pair formats give signal
    earliest.

[Full analysis →](../signal-noise/benchmark_design.md)
