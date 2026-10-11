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

**How we measure it.** For every version of every benchmark in every
language (1,754 tasks over 43 benchmarks, seed-1904 runs, multi-axis design
pairs) we compute the DA-size of each proxy 90M–1B against the 1.7B model:
the share of pairs of training choices both order the same way. A version is
**reliable** at a size when its DA-size is at least τ = 0.75 there and at
every larger proxy. A bBPB version is checked in two ways: against the 1.7B
model's **accuracy** (the same truth as the accuracy versions) or against its
own **bBPB** at 1.7B (an easier target: the reference's bBPB ranking, not
its decision).

## The recommendation

Each benchmark appears once per way of checking bBPB; the version column
names which.

<div class="viz" data-viz="recipe"></div>

## Your language, your proxy

Choose your language and proxy size: each card is one benchmark, the bars
compare its versions at that size (black tick: reliable, grey tick:
chance), and the bold line is the version to evaluate.

<div class="viz" data-viz="recipeCards"></div>

## How each version grows with proxy size

<div class="viz" data-viz="recipeSizes"></div>

## What we find

- **No version reads the 1.7B model's accuracy decision reliably on
  average at any proxy up to 1B.** The best mean DA-size is 0.63 (LLM-RF
  scored by bBPB, at 90M); only the easier bBPB-against-bBPB target clears
  τ (LLM-RF at 0.76–0.88, on 35 tasks).
- **Against the same truth, bBPB matches or edges accuracy.** Over every
  task, the original benchmarks scored by bBPB read 0.57–0.60 against
  0.53–0.57 for their accuracy, and 0.59–0.64 on the tasks both have.
- **The rewrites help only when scored by bBPB.** RF and LLM-RF scored by
  accuracy read 0.48–0.56, below the original accuracy; scored by bBPB they
  read 0.57–0.63.
- **bBPB is the pick for 20 of the 36 benchmarks** with a value against the
  1.7B accuracy, and accuracy for 16.
- **Few benchmarks are reliable by 1B in most languages.** Against the 1.7B
  accuracy, the share of languages reliable by 1B is 0.69 for HellaSwag,
  0.62 for XStoryCloze and 0.80 for LAMBADA. Every other benchmark with more
  than three ranked languages is at 0.50 or below.

!!! success "Takeaway"
    Choose the version of a benchmark, not only the benchmark: score it by
    the bits per byte of the correct answer, which reads the large model's
    decision at least as well as accuracy and makes the rewritten formats
    useful. Even then, few benchmarks decide reliably below 1B, so confirm a
    small-model decision on the languages and sizes where the cards above
    say it holds.

[Full analysis →](../signal-noise/analysis.md)
