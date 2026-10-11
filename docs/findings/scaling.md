---
title: "RQ1 · Scaling"
---

# RQ1 · What scales predictably?

**The question.** A small-model result is only useful if it sits on a trend
you can extrapolate. Which benchmarks, and which per-language measurements,
follow a log-linear trend in model size and along training?

**How we measure it.** For each task and language setting L we fit the final
score against log model size, on the sizes where the task is above chance (at
least three), and within each run we fit the score against log training
tokens (at least five checkpoints). R² says how well a straight line explains
the trend; Spearman ρ says its direction. A task's point is the median over
its fits (at least two per axis), on the deep baseline models, 90M–1.7B. Per-language
bits per byte (BPB) has no chance level, so it is never gated. A task with
R² ≥ 0.5 on both axes is *predictable across both*; a task whose ρ is negative
*declines with size*, however well it fits.

## Explore

<div class="viz" data-viz="scaling"></div>

The same points with the four regimes drawn and counted. Type a benchmark or
a language to name its points, switch to the bBPB twins (each benchmark's
gold-answer bits per byte), or compare size with direction (ρ) to see tasks
that fit well but get worse. Click a point, or a match below the chart, to
open its curves in the explorer.

<div class="viz" data-viz="scalingQuadrant"></div>

### Pick a benchmark

Top: each model size's final score, one colour per language setting, with the
fitted line through the sizes above chance and its R² in the legend; hollow
dots are at chance and stay out of the fit. Bottom: the same task along each
run's training, one line per size.

<div class="viz" data-viz="scalingCurves"></div>

### Can the small models forecast the 1.7B model?

The R² above includes the 1.7B model in the fit. Here a power law is fitted
on the small models only and extended to 1.7B. Pick a language, a language
setting and how far up the ladder the fit reaches, and compare the forecast
(diamond) with what 1.7B actually reaches.

<div class="viz" data-viz="scalingForecast"></div>

## What we find

<!-- highlight: rq01_scaling_predictability -->

- **Above chance, accuracy scales log-linearly with size.** The 1417 gated
  (task, L) fits over 509 benchmark tasks have a median size-fit R² of 0.87
  (IQR 0.76–0.94). Per-language BPB fits at 1.00 for all 50 trained languages.
  HellaSwag's family median is 0.88, LAMBADA's 0.96 and ARC's 0.98.
- **74 % of the tasks with a fit are predictable along both size and
  training.** Of the 420 accuracy and BPB tasks with a regime, 311 are, 56 are
  predictable across size only, 42 are weak in both and 11 decline with size.
  None is predictable during training but not across size (9 of the 609 bBPB
  twins are), so varying model size gives the signal without long training
  runs.
- **The gate decides which benchmarks have a law at all.** 429 of 988
  accuracy and BPB tasks are at chance wherever a fit was possible, and 16 of
  54 families lose every task. Letter-format Global-MMLU, INCLUDE and
  Belebele are among them. Their reformulated versions fit at family
  medians of 0.84–0.92.
- **R² alone is not enough.** TruthfulQA fits at a family median R² of 0.97
  with ρ = −0.75: it gets worse with size, and does so consistently (its ρ
  rests on 90M–350M fits, the sizes where it is above chance).
- **A power law fitted on the small models over-predicts the 1.7B BPB.** In
  all 318 deep baseline per-language fits the forecast is too high: median
  error 7.7 % fitted up to 350M, 4.4 % up to 600M and 2.3 % up to 1B.

!!! success "Takeaway"
    Likelihood-based measurements (bits per byte, LAMBADA, HellaSwag,
    cloze-style ARC and XStoryCloze) and the reformulated versions of
    letter-format benchmarks give trends a small ladder can extrapolate.
    Letter-format knowledge multiple choice (Global-MMLU, INCLUDE, Belebele)
    stays at chance on this ladder and has no trend at all. Read the direction
    as well as R², and expect a small-model forecast to overshoot.

[Full analysis →](../signal-noise/scaling_predictability.md)
