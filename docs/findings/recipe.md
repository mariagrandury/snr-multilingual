---
title: "RQ11 · Evaluation recipe"
---

# RQ11 · Which benchmark, posed and scored how, gives a reliable decision?

**The question.** The other pages ask whether a cheap measurement agrees
with the 1.7B model. This one turns the answer into advice. A benchmark can
be posed in three ways: **as published**, **reformatted (RF)** so that the
model is scored on the answer text instead of a letter, or **rewritten by an
LLM (LLM-RF)** (see [RQ0](gate.md#how-does-the-format-affect-the-evaluation)).
And it can be scored in two ways: **accuracy** (did the model pick the right
answer?) or **bits per byte of the correct answer (bBPB)**, how surprised the
model is by the right answer, which moves smoothly even before the model
picks it. For each benchmark, which of these six versions makes the right
decision from the smallest proxy?

**How we measure it.** For every version of every benchmark we compute the
decision accuracy of each proxy size against the 1.7B model (the share of
pairs of training choices both order the same way). A version is
**reliable** at a size when that decision accuracy is at least the threshold
in the table note, and stays so at every larger proxy.

## The recommendation

<div class="viz" data-viz="recipe"></div>

## How each version grows with proxy size

<div class="viz" data-viz="recipeSizes"></div>

!!! success "Takeaway"
    Choose the version of a benchmark, not only the benchmark. Click a column
    of the table to sort by it, and evaluate each benchmark in the version
    listed for it.

[Full analysis →](../signal-noise/analysis.md)
