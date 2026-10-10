---
title: "RQ2 · Decision accuracy"
---

# RQ2 · Does a small model rank recipes like the large one?

**The question.** You train recipes at a small size, pick the best, and scale
it up. How often does the small model pick the recipe the large model would
pick? And how early in a run can you read that answer?

**How we measure it.** Decision accuracy (DA) is the share of pairs of design
variants (two training recipes that differ in the number of languages, the
depth, the activation, the data scheme or the sampling temperature) that a
proxy orders the same way as the 1.7B reference. **DA-size** compares the
final checkpoint of a smaller model with the final 1.7B checkpoint;
**DA-ckpt** compares an early checkpoint with the final one of the same run;
**DA-goal** compares an early checkpoint of a smaller model with the final
1.7B checkpoint. The pairs come from the seed-1904 runs of every language
setting, depth and data build from 90M to 1.7B (29 designs with a 1.7B run,
406 pairs). A benchmark task counts only where it is above chance at the
proxy and at 1.7B ([RQ0](gate.md)). A proxy that ranks at random scores about
0.5, which we call **chance**.

!!! note "Key findings"
    - **On benchmark accuracy a fully trained smaller model is close to
      chance.** DA-size is 0.51–0.57 from 90M to 1B. On the per-language bits
      per byte (BPB) of the 50 trained languages it is 0.92–0.96 at every
      proxy except 600M (0.74). At 600M the depth pairs are not size-matched:
      the shallow model has 3.7 % more non-embedding parameters and spends
      about 15 % more compute, and almost all of the drop comes from these
      pairs.
    - **Training a proxy longer does not close the gap.** Against its own
      final checkpoint, an early checkpoint recovers 61–68 % of the decisions,
      and 84–92 % at 90 % of the run. Two seeds of one design, which have
      nothing to decide, already agree on 78–80 % at that point, against
      82–86 % for real design pairs, so most of the rise is persistence
      within a run. Against the 1.7B final the curve stays flat.
    - **Scoring the same items by bits per byte helps, but only a little.**
      The bits per byte of the gold answer (bBPB) of a proxy predicts the
      1.7B accuracy ranking at 0.58–0.60 at every size, against 0.51–0.57 for
      accuracy itself (+0.12 at 90M, +0.03 at 1B). On the tasks whose
      accuracy is still at chance at the proxy, bBPB ranks like the 1.7B bBPB
      at 0.61–0.64, against 0.46–0.50 for accuracy, but far below the
      0.95–0.98 of the macro average of BPB on the same pairs.
    - **Chance is about 0.49, not 0.5, and few benchmark tasks beat it.**
      Shuffling a proxy's scores across the designs gives 0.49 on the
      benchmark tasks and their bBPB twins (a tie on one side counts as a
      miss), so their observed 0.58–0.59 is only 0.08–0.10 above it, and only
      8–10 % of the benchmark cells beat their own null. Per-language BPB
      reads 0.79–0.95 against a null of 0.50.
    - **Part of the gap is the reference's own noise.** On the pairs whose
      1.7B gap exceeds one or two seed standard deviations, benchmark DA-size
      rises to 0.64–0.66 and 0.71–0.73, but these pairs are only 52 % and
      25 % of the comparable ones. BPB reads 0.88–0.99 and 0.95–0.995 on them.
    - **BPB does not need many tokens of a language.** The 90M proxies
      recover 95 % of the 1.7B decisions at the first tenth of their run,
      after about 8 million tokens of a language. 600M stays at 70–74 % over
      its whole run, while 350M and 1B climb to 92 %.

## Explore

How early and how small can you go?

<div class="viz" data-viz="da"></div>

*Drag the slider along the run: each row is a proxy size, and the dots show
how often that checkpoint orders the design pairs like the 1.7B final, like
its own final, and how often two seeds of one design agree with each other.*

<div class="viz" data-viz="daCheckpoints"></div>

Which benchmarks, in which language?

<div class="viz" data-viz="daBench"></div>

*Each row is a way of scoring the proxy, each dot one proxy size: the further
right, the more design decisions it shares with the 1.7B model. Switch the
tasks to see the items a proxy cannot yet answer above chance.*

<div class="viz" data-viz="daChannels"></div>

## Pick two designs

*Choose two training recipes and a score: each bar shows on how many
languages or tasks a proxy of that size picks the same winner as the 1.7B
model. The dashed tick is the same share averaged over all 406 pairs.*

<div class="viz" data-viz="daPairs"></div>

The default pair, deep against shallow at eight languages, shows the 600M
exception: the 600M proxy picks the other depth on every language, while
90M, 175M and 1B pick the reference's winner on every one. Pairs that move
the language count, the sampling temperature or the data scheme are read
right on most languages from 90M; on accuracy every design axis stays near
chance. The SwiGLU runs have no BPB pair yet.

## How far from chance?

*Per proxy size: the dark tick is what a proxy with shuffled scores reaches,
the grey dot is DA-size over every pair, and the blue dot is DA-size over
the pairs the 1.7B reference itself separates beyond seed noise.*

<div class="viz" data-viz="daDecisive"></div>

All of these values compare a proxy with one 1.7B run per design. Two seeds
of the same 1B designs agree with each other on only 53–60 % of the
decisions, about as often as the proxies agree with the reference; the
[next rung](size-generalisation.md) shows how much the choice of reference
moves DA-size.

## What we find

<!-- highlight: rq02_decision_accuracy -->

The highlighted means include the bBPB twins and average over tasks, so they
sit a few points above the accuracy-only 0.51–0.57 quoted above.

!!! success "Takeaway"
    Rank recipes on per-language bits per byte; it agrees with the 1.7B
    reference on more than 90 % of the decisions from 90M, except where the
    pair is not size-matched. Treat benchmark accuracy as a second check,
    one family at a time, and score it by bits per byte where accuracy is
    still at chance. Do not stop a run on an early checkpoint because it
    agrees with its own end: two seeds of the same recipe agree almost as
    much.

[Full analysis →](../signal-noise/decision_accuracy.md)
