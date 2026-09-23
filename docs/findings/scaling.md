---
title: "RQ1 · Scaling"
---

# RQ1 · What scales predictably?

**The question.** A small-model result is only useful if it sits on a trend
you can extrapolate. Which benchmarks, and which per-language measurements,
follow a log-linear trend in model size and along training?

**How we measure it.** For each task and language setting we fit score
against log-compute on the sizes where the task is above chance, and report
the fit quality (R²) across sizes and along each run's checkpoints.

## Explore

<div class="viz" data-viz="scaling"></div>

## What we find

<!-- highlight: rq01_scaling_predictability -->

!!! success "Takeaway"
    Likelihood-based measurements (bits per byte, LAMBADA, HellaSwag,
    cloze-style ARC and XStoryCloze) give trends a small ladder can
    extrapolate. Knowledge multiple choice (INCLUDE, Global-MMLU) does not.

[Full analysis →](../signal-noise/scaling_predictability.md)
