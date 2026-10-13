---
title: "RQ5 · Design decisions"
---

# RQ5 · Which proxy sizes read each design decision?

**The question.** Do small models rank a design choice the way the 1.7B
reference does? The decisions are the model's depth (deep vs shallow at
equal non-embedding size), the data scheme (the language list at L8–L30,
the second language at L2, the English source at L1), the sampling
temperature (T = 1 vs T = 3), the activation (xIELU vs SwiGLU) and the
optimizer (AdEMAMix vs Muon). Does the answer change with the number of
languages, and how early in the run can you read it?

**How we measure it.** For every decision we pair runs that differ only in
it, each against the baseline (deep, scheme A, T = 1), and measure the share
of items on which the proxy (90M–1B) prefers the same level as the 1.7B
reference (DA-size), averaged over the language settings L. We read each
decision on two populations: the bits per byte of the languages both levels
train, and the benchmark tasks above chance at the proxy and at 1.7B. We also
measure how large each decision's effect is at the reference, in units of
seed noise (the spread of three seeds of the same recipe). Every seed of
every ladder and data build is in the pool.

The design-decision tables cover depth, data scheme, temperature, activation and optimizer. The SwiGLU ladder is trained at L8–L30 from 90M to 1.7B and is
read on both populations. The Muon ladder (L8–L30) is evaluated from 90M to 1.7B; no proxy reads the optimizer decision (0.19–0.23 on BPB, 0.46–0.55 on the benchmarks).

The 600M depth pair is not size-matched: shallow 600M (14 layers × 2048) has
3.7 % more non-embedding parameters than deep 600M (24 × 1536) and spends
about 15 % more training compute, against +2.6 % to +8.9 % compute at the
other sizes. Its pairs are kept and flagged (†) in every view.

## Explore

Is there a decision to read? Effect of each decision, in units of seed noise:

<div class="viz" data-viz="effect"></div>

Agreement with the reference, by proxy size:

<div class="viz" data-viz="interventions"></div>

**Which decisions can a small model make?** One row per decision, one column
per proxy size: a filled dot means the proxy agrees with the 1.7B reference on
at least the threshold share of items, a hollow one that it does not. The bar
on the right is the decision's effect at 1.7B in seed standard deviations
(dashed lines at 1× and 2×), so you can read a row as "is there a decision,
and who sees it". Move the checkpoint to read the proxies earlier in their run.

<div class="viz" data-viz="ddLadder"></div>

How early can you decide?

<div class="viz" data-viz="early"></div>

**The decision clock.** Each cell is one proxy's run from 0.5C to 5C (C = 20
tokens per parameter), one tick per evaluated checkpoint, filled where the
proxy agrees with the 1.7B final; the label says from when it agrees for good.
The last column is the reference's own run against its final.

<div class="viz" data-viz="ddClock"></div>

## What we find

<!-- highlight: rq05_design_decisions -->

- **Per-language BPB reads the temperature and list decisions at every proxy
  size, the gated benchmarks almost never.** The trained languages' BPB reads
  temperature at 0.95–0.99 (mean over L15–L50) and the language lists at
  0.77–1.00 (L8–L30). On the benchmarks every intervention sits at 0.45–0.63,
  and only the two L1 English-data swaps come near 0.75 (DCLM without the edu
  filter 0.52–0.79, FineWeb 0.57–0.75), on 14–133 tasks.
- **Only temperature on BPB is far outside seed noise at the reference.** Its
  effect is 5.4 seed sds on per-language BPB, against 1.3 for depth and 1.6
  for the language lists; on the benchmarks every decision sits at 1.1–2.1.
  Keeping only the items the reference decides outside seed noise raises the
  benchmark DA from 0.54 to 0.60, still short of 0.75.
- **Depth is a vanishing advantage, not a crossover.** Deep beats shallow by
  1.9–2.1 standard deviations of the two-run difference at 90M and 1.1–1.6 at
  175M, and from 350M on the two are within 2 of each other at every size, so the depth DA on BPB (0.97 at
  90M, 0.49 at 350M, 0.95 at 1B) reads a reference with no real preference.
  At 600M, whose depth pairs are not size-matched, it is 0.00 at all four L.
- **On BPB the smallest proxy is enough, and early is enough.** 90M reads the
  depth decision at 0.93–0.98 and the list decision at 0.96–1.00 from 0.5C
  on. Larger proxies are not safer: the 1B list read falls from 0.91 at 1C
  to 0.72–0.77 from 4C on. On the benchmarks no proxy reaches 0.75 on these
  two decisions at any checkpoint. The reference's own checkpoints reach 0.76
  at 4.5C.
- **SwiGLU follows the same pattern, and no proxy reads Muon.** The
  activation decision is read at 0.48–0.56 on the benchmarks by every proxy,
  and 0.76 by the reference's own run at 4.5C. On per-language BPB the proxies read it at
  0.90–0.98, except 600M (0.34). The Muon decision is read at 0.19–0.23 on BPB and 0.46–0.55 on the benchmarks; its effect at the reference is 0.3 seed standard deviations on BPB.

!!! success "Takeaway"
    Before an ablation, ask whether its effect will be larger than seed
    noise. Large effects (temperature on bits per byte) are readable from
    90M and from the first checkpoints; small ones (depth, or any decision
    read on the benchmarks) are close to chance at every proxy size.

[Full analysis →](../signal-noise/design_decisions.md)
