---
title: "RQ13 · English-only check"
---

# RQ13 · Do the English-only models do better on English benchmarks?

**The question.** The ladder's L1 models are trained on English only; every
other model spends half of its tokens on other languages. If the training
runs are sound, the English-only models should do clearly better on English
benchmarks than multilingual models of the same size: higher scores, above
chance earlier, and at least as reliable in their decisions. This page is a
sanity check of the ladder itself, and a comparison with AllenAI's
DataDecide models on the same English tasks.

## Does each expectation hold?

<div class="viz" data-viz="englishVerdict"></div>

## How large is the English advantage?

<div class="viz" data-viz="englishGap"></div>

[Full analysis →](../signal-noise/analysis.md)
