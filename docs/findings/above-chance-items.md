---
title: "RQ12 · Above-chance items"
---

# RQ12 · Do scores get more reliable on the questions the large model gets right?

**The question.** Many benchmark questions are answered at chance even by the
1.7B model: they only add noise to the score. What happens if each
benchmark keeps only the questions the 1.7B model answers above chance? Do
decision accuracy, the signal-to-noise ratio and the above-chance test
improve, and does it matter whether we filter the benchmarks
([RQ0](gate.md)) before or after choosing the questions?

**A warning on reading it.** The questions are chosen by looking at the
reference model, so the numbers here are an *upper bound* on what this
selection can buy, not an estimate for a new model. [RQ8](subsets.md) holds
out the reference to give the fair version.

## How many questions survive

<div class="viz" data-viz="itemsSurvival"></div>

## Decision accuracy, with and without the selection

<div class="viz" data-viz="itemsDA"></div>

[Full analysis →](../signal-noise/analysis.md)
