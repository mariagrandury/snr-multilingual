---
title: "RQ7 · External frameworks"
---

# RQ7 · Do our results agree with DataDecide?

**The question.** Do the benchmarks that separate models on our ladder also
separate them on AllenAI's DataDecide / OLMo ladder (Heineman et al., 2025),
on the English tasks both evaluate?

**How we measure it.** We compute the SNR of every shared English task on
both ladders and correlate them (Pearson r on log SNR, Spearman ρ on the
ranking). The headline compares our 1B rung (seed 1904, every design variant)
with DataDecide's 1B rung (25 data recipes), on the nine shared tasks that
clear the above-chance gate on our side: ARC Easy, ARC Challenge, HellaSwag,
MMLU, CommonsenseQA, OpenBookQA, PIQA, Winogrande and BoolQ (MMLU and
CommonsenseQA through our cloze versions, which match DataDecide's format). Both sides use the same noise
window, the same detrended checkpoint noise and the same metric. What they
vary differs: 25 pretraining corpora on DataDecide, our design variants on
our ladder.

## Explore

The first view is the external model set of the earlier 36-model sweep (public
models from 270M to 70B, one SNR per task across models), on its own shared
English tasks:

<div class="viz" data-viz="external"></div>

**The ladder's comparison as a slopegraph.** Each line is one shared task,
from its SNR rank on our ladder (left) to its rank on DataDecide (right): flat
lines keep their place, crossing lines swap. Switch the SNR definition or the
size pair, or position the tasks by their SNR on one log axis to see the gap
in levels.

<div class="viz" data-viz="extSlope"></div>

## What we find

<!-- highlight: rq07_external_frameworks -->

- **Both corpora rank the nine gated shared tasks alike** (Spearman ρ 0.78, Pearson r of log₁₀ SNR 0.75). Both put BoolQ last and OpenBookQA
  near the bottom. The top task differs (HellaSwag on ours, ARC Easy on
  DataDecide). Nine points make this indicative, not robust.
- **Our SNR is 1.2–4.2× smaller.** DataDecide's SNR exceeds ours on every
  shared task, by 1.2× on Winogrande up to 4.2× on ARC Challenge: compare
  orders across corpora, never SNR levels.
- **The agreement holds for most SNR definitions.** 16 of the 22
  definitions have r ≥ 0.6 at 1B ↔ 1B, from 0.21 (`gini`) to 0.90
  (`rel_dispersion`). The seed pool raises r for all but one definition and
  moves no r by more than 0.27.
- **It does depend on the size pair.** With seven or nine points, `rel_std`
  spans r 0.36 (175M ↔ 150M) to 0.85 (1.7B ↔ 1B), while `rel_star_discrepancy`
  stays at 0.82–0.95.

[Full analysis →](../signal-noise/external_frameworks.md)
