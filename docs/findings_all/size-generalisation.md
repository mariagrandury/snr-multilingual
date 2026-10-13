---
title: "RQ10 · Past the reference"
---

# RQ10 · Does a ranking that holds at 1.7B still hold at 3B?

**The question.** Every other page judges a small proxy by whether it makes
the decisions the 1.7B model makes. But the 1.7B model is itself a proxy for
the larger models one actually wants to train. We trained six of the
configurations one rung further, to 3B, to ask the one question the rest of
the study cannot: is the reference itself predictive of the next size up?

**How we measure it.** The six 3B models are deep, seed 1904: data scheme A
with 8, 15, 30 and 50 training languages, and scheme B with 8 and 15. On the
decisions between them we compute the decision accuracy of every smaller
model against the 3B model, and on the same decisions against the 1.7B
model. If the two agree, a proxy that reads the 1.7B decision also reads the
3B one. A decision needs at least three pairs per design axis, and among six
configurations only the number of languages clears it: the data scheme
varies in two pairs only.

!!! note "Key findings"
    - **Per-language bits per byte carries over.** The 1.7B models predict
      the 3B decisions at 0.94–0.95, every proxy from 90M reaches at least
      0.78, and the macro average of BPB orders every pair the same way at
      1.7B and at 3B.
    - **Benchmark accuracy stays at chance against either reference.** On
      the same decisions it reads 0.48–0.53 against 1.7B and against 3B, and
      the 1.7B model orders the 3B decisions no better than a smaller proxy.
      The gap between accuracy and BPB does not come from the choice of the
      1.7B reference.
    - **The larger reference measures more of the suite.** The above-chance
      filter keeps 778 of 933 benchmark tasks at 3B against 612 at 1.7B
      (+27 %), mostly the letter-format originals of Belebele, Global-MMLU
      and INCLUDE, which their reformulations already make measurable at
      1.7B.
    - **The reference's seed matters as much as its size.** At 1B, the only
      size above 600M with three seeds, two seeds of the same English designs
      agree on only 53–60 % of the decisions. With the 1B models as the
      reference, the seed of the reference alone moves the single-axis
      DA-size of a proxy by up to 0.13, more than replacing the 1B reference
      with the 1.7B one.

## Explore

<div class="viz" data-viz="aboveReference"></div>

The six 3B models give only 15 pairs (9 that move one design axis), so each
point rests on few decisions, and the BPB lines on 16–21 tasks: the
languages every 3B model with a BPB final trains, English and the macro
average. Read the trend, not single points.

*Each row is a proxy size; the grey dot is its agreement with the 1.7B
model, the blue one with the 3B model, on the same decisions. The 1.7B row
reads the reference itself against 3B.*

<div class="viz" data-viz="sgReference"></div>

## What the 3B rung can measure

*One row per benchmark: the grey dot is the share of its tasks above chance
at 1.7B, the blue one at 3B, on the same six configurations.*

<div class="viz" data-viz="sgGate"></div>

## How much does the choice of reference matter?

*Each dot is a proxy's DA-size against a different reference on the same
decisions: the spread within a row is what the reference alone changes.*

<div class="viz" data-viz="sgSeeds"></div>

!!! success "Takeaway"
    A 1.7B reference is large enough for BPB: its decisions are the 3B
    decisions. For benchmark accuracy, moving the reference up one rung does
    not help the proxies; it makes more benchmarks measurable, but their
    rankings still do not transfer. One run per design at the reference is
    a noisy target: its seed moves decision accuracy as much as its size.

[Full analysis →](../signal-noise/analysis.md)
