---
title: "RQ12 · Above-chance items"
---

# RQ12 · Do scores get more reliable on the questions the large model gets right?

**The question.** Many benchmark questions are answered at chance even by the
1.7B model: they only add noise to the score. What happens if each
benchmark keeps only the questions the 1.7B runs answer above chance? Do
the signal-to-noise ratio, decision accuracy and the above-chance test
improve, and does it matter whether we filter the benchmarks
([RQ0](gate.md)) before or after choosing the questions?

**How we measure it.** For every benchmark in every trained language (933 tasks), a question is kept when the 1.7B final runs that trained on its
language answer it above chance on average (4 to 32 runs per task). We
compare three readings: the **full benchmark**; **gate, then items** (the
tasks the above-chance gate passes, cut to their kept questions, so the same
tasks as the full benchmark); and **items, then gate** (every task cut
first, then the gate recomputed on the kept questions).

**A warning on reading it.** The questions are chosen by looking at the
reference model, so the numbers here are an *upper bound* on what this
selection can buy, not an estimate for a new model. [RQ8](subsets.md) holds
out the reference to give the fair version.

## How many questions survive

<div class="viz" data-viz="itemsSurvival"></div>

The same steps as one bar per reading: each bar is every question of the
suite, split into the ones scored, the ones dropped as at chance, and the
ones in tasks the gate drops.

<div class="viz" data-viz="itemsFunnel"></div>

## Per benchmark and language

Pick a benchmark (or all of them): each row is one language, the bar the
share of its questions kept; switch to SNR or DA-size to see each task move
from the full benchmark (grey dot) to its kept questions (arrow head).

<div class="viz" data-viz="itemsExplorer"></div>

## Decision accuracy, with and without the selection

<div class="viz" data-viz="itemsDA"></div>

The same comparison task by task: each dot is one task at the chosen proxy,
placed by how much its SNR changed (left to right) and how much its decision
accuracy changed (down to up).

<div class="viz" data-viz="itemsSnrVsDa"></div>

## What we find

- **Fewer than half the questions are kept:** 43 % of the questions of the 595 tasks the gate passes at 1.7B, 42 % over all 932 tasks with a question
  above chance.
- **Choosing the questions first makes the gate almost vacuous.**
  Recomputed on the kept questions it passes 910 of those 932 tasks at 1.7B and 84–91 % of them at the proxies, against 40 % (90M) to 64 % (1.7B) for
  the full benchmarks.
- **The SNR rises through a lower noise, not a better ranking.** On the same
  tasks the median SNR rises ×1.29 at 90M to ×1.51 at 1.7B, because the
  noise falls (×0.58–0.86) while the signal stays flat (×0.89–1.09).
- **Decision accuracy barely follows, although the selection reads the
  reference it is scored against.** The paired gain is +0.045 at 90M and +0.002 to +0.033 at 175M–1B, inside the full benchmark's band; the tasks
  the recomputed gate admits rank at or below chance (0.47–0.50).
- **The held-out version gains about as much.** The questions half of the
  1.7B runs solve, chosen on other runs ([RQ8](subsets.md)), gain +0.018 to +0.039 at 90M–600M.

!!! success "Takeaway"
    Dropping the questions the large model gets wrong makes a benchmark look
    more reliable (a higher SNR) without making it rank the training choices
    more like the large model. Do not filter the benchmarks after choosing
    the questions: the gate then passes almost every task, including ones
    that rank no better than chance.

[Full analysis →](../signal-noise/analysis.md)
