---
title: "RQ0 · Above chance"
---

# RQ0 · Which benchmarks beat chance?

**The question.** A benchmark whose scores sit at chance ranks recipes at
random. Before anything else, which (task, language) pairs clear chance at
each model size?

**How we measure it.** For every run, we test whether its final score is above
the chance level with confidence (a lower confidence bound over the task's
items). The share of runs at a size that pass is the task's gate value. Every
later RQ uses only the tasks that pass at the sizes it compares.

## Explore

<div class="viz" data-viz="gate"></div>

### Scoring the answer, not the letter

Three knowledge families ask the model to answer with a letter (A–D). Small
base models have not learned that convention. Scoring the full answer text as
a continuation measures the knowledge instead:

<div class="viz" data-viz="reformulation"></div>

## What we find

<!-- highlight: rq00_gate_and_curves -->

!!! success "Takeaway"
    Check the gate at *your* proxy size before choosing a benchmark.
    Minimal pairs (MultiBLiMP), HellaSwag, XNLI, XWinograd and XCOPA carry
    signal from 175M–350M; letter-format knowledge tests need a cloze
    reformulation to be usable at these sizes.

[Full analysis →](../signal-noise/gate_and_curves.md)
