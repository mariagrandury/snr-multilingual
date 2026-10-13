---
title: "RQ3 · Noise and SNR"
---

# RQ3 · How noisy is each benchmark?

**The question.** How much does a score move when only the seed or the
checkpoint changes, and how does that compare with the differences between
recipes?

**How we measure it.** *Signal* is the spread of the final scores across
every design variant trained at a size (relative standard deviation).
*Checkpoint noise* is how much one run wobbles over the last 20 % of
training (the checkpoints at 80, 85, 90, 95 and 100 %): since 2026-10-08 it
is the residual standard deviation around a straight line through those five
points, so the rise that the learning-rate decay still makes at the end of a
run is not counted as noise; the plain standard deviation is kept as the
*raw* variant. *Seed noise* is the spread of the final score across three
seeds of the same recipe. We compute 22 SNR definitions from the literature
per task and size; the views show `rel_std` (relative std of the finals over
the relative checkpoint noise), the one [RQ4](surrogates.md) finds best.

## Explore

<div class="viz" data-viz="snr"></div>

**Signal against noise, task by task.** Each dot is one task at the chosen
size: up-left is a benchmark whose recipes end far apart and whose runs are
quiet. Hover a dot to see why: its curves across the size's design variants,
with the spread of the final scores (signal) and the wobble inside the shaded
window (noise). Toggle the raw noise to see what the detrending changed.

<div class="viz" data-viz="noiseSnrMap"></div>

Try the idea on toy numbers:

<div class="viz" data-viz="snrToy"></div>

**Is checkpoint noise the right noise?** Read each row as one design decision
measured against two yardsticks: the wobble of one run (blue) and a re-roll
of the seed (orange). The gap between the two dots is how much a
checkpoint-noise SNR flatters the decision.

<div class="viz" data-viz="noiseSnrEffect"></div>

**Where the 600M proxy goes wrong.** Each bar splits the proxy's
comparisons against 1.7B into agreements and flips (orange), and each into
"the proxy could tell the two variants apart" (strong shade) or not (muted shade). Pick
600M and per-language BPB: the flips sit on the depth pairs.

<div class="viz" data-viz="noiseSnrFlips"></div>

## What we find

- **SNR peaks at 350M and is lowest at the reference.** The median log₁₀
  SNR of the benchmark tasks above chance is 0.29 at 90M, 0.41 at 350M and
  0.23 at 1.7B (381 to 588 tasks): at 1.7B the typical benchmark separates
  the design variants by only 1.7× its checkpoint noise. 75 of 588 tasks
  (13 %) are below the noise at 1.7B. The bBPB twins follow the same curve
  one notch higher (0.36 at 1.7B), and the per-language BPB is far above it
  (1.21 at 350M, 0.80 at 1.7B).
- **HellaSwag carries the most signal at 1.7B** (median log₁₀ SNR 0.62 over
  26 tasks, then XStoryCloze 0.49 over 8, among the families with more than
  one task). KoBEST carries the least (−0.09), one of four families below the
  noise. Every trained language but Lithuanian and Norwegian is above the
  noise.
- **Seed noise exceeds checkpoint noise, so the checkpoint SNR is
  optimistic.** A re-roll of the seed moves the final score a median 1.75×
  the detrended checkpoint noise (in 75 % of 4650 cells). Depth's median
  effect is 2.12× the checkpoint noise but 1.30× the seed noise, and no
  design decision clears twice the seed noise in half of its cells (depth
  30 %, data scheme A vs B 29 %, A vs C 43 %, temperature 46 %,
  activation 27 %). The optimizer has a single cell so far.
- **The 600M BPB misses are one seed of one depth pair.** Per-language BPB
  ranks the variants like 1.7B from every proxy (DA-size 0.89–0.94) except
  600M (0.74). There, 95 % of the depth comparisons flip against 4–27 % at
  the other proxies, while language count, data scheme and temperature
  behave as everywhere else. Most of those flips are below the 600M seed
  noise, and re-drawing the deep run's seed flips 15, 43 or 50 of the 50
  languages: the reversal is a draw of the seed (the 600M depth pair is also
  not size-matched, +3.7 % parameters for shallow).
- **The ranking of SNR definitions survives a seed swap only as a
  family.** Across replicate seeds the ranking of the 22 definitions
  correlates at Spearman ρ 0.73 (DA-ckpt) and 0.55 (DA-size), but that rests
  on one or two languages (English, Russian).

!!! success "Takeaway"
    Estimate noise on your own runs: detrend the late checkpoints when the
    learning rate is still decaying, and train at least one seed replicate,
    because the seed moves the score more than the checkpoints do. A design
    difference below one seed standard deviation at the proxy can reverse at
    the reference.

[Full analysis →](../signal-noise/noise_and_snr.md)
