---
title: Signal and noise in multilingual evaluation
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Which benchmarks can you trust at small scale?

<p class="lead">To choose a data mixture or an architecture, teams train small
models and compare their benchmark scores. That only works if the small-model
ranking is the one the large model would give. In multilingual training it
often is not: most benchmarks sit at chance on small models, and many that do
move rank two recipes one way at 350M and the other way at 1.7B. We trained a
controlled multilingual model ladder to measure, per benchmark and per
language, <b>when a small-scale evaluation predicts a large-scale
decision</b>.</p>

</div>

## The problem in one chart

<div class="viz" data-viz="hero"></div>

A benchmark gives a useful signal for an ablation only if it passes three
tests. It must be **above chance** at the size you train, its **signal**
(how much the recipes differ) must be larger than its **noise** (how much a
score moves from one checkpoint or seed to the next), and its ranking must
**agree** with the ranking at the size you care about. This site shows which
benchmarks pass, at which size, in which language.

## Contributions

<div class="cards" markdown>

<a class="card" href="ladder/"><b>An open model ladder</b><span>163 runs from 175M to 1.7B non-embedding parameters (plus 90M and 3B rungs), 1 to 50 training languages, four design interventions and seed replicates, each trained for 100 tokens per parameter.</span></a>

<a class="card" href="ladder/#per-language-bits-per-byte"><b>Per-checkpoint measurements</b><span>Bits per byte in 100 languages and 14 multilingual benchmark families, on ten aligned fractions of every run.</span></a>

<a class="card" href="findings/"><b>Signal and noise, multilingual</b><span>The signal-to-noise framework of Heineman et al. (2025) extended from English to controlled multilingual ladders: chance gate, scaling, decision accuracy, cheap predictors, language transfer.</span></a>

<a class="card" href="recommend/"><b>A benchmark recommender</b><span>Pick your language, proxy size and decision, or upload your own results, and get the benchmarks that are reliable for your use case.</span></a>

</div>

## Key findings

<div class="cards" markdown>

<a class="card" href="findings/gate/"><b>Half the suite is noise at small scale</b><span>Of 647 (task, language) pairs, only 331 beat chance at any size. Four-option knowledge benchmarks (Global-MMLU, INCLUDE, Belebele) stay at chance up to 1.7B, but scoring the answer text instead of a letter recovers their signal.</span></a>

<a class="card" href="findings/scaling/"><b>Likelihood-based measurements scale</b><span>Per-language bits per byte, HellaSwag, LAMBADA, ARC and XStoryCloze follow log-linear trends (median R² ≥ 0.97) across size and along training.</span></a>

<a class="card" href="findings/decision-accuracy/"><b>Bits per byte decide early</b><span>A 350M proxy ranks design variants like the 1.7B reference on per-language bits per byte (decision accuracy 0.97), while the benchmark average stays near 0.56.</span></a>

<a class="card" href="findings/design-decisions/"><b>Effect size sets reliability</b><span>A decision is easy to read at small scale when its effect is large against seed noise (sampling temperature: 6.6× on BPB) and hard when it is not (depth: about 1×).</span></a>

<a class="card" href="findings/surrogates/"><b>No universal SNR</b><span>The SNR definition that best predicts decision accuracy changes with the language and the decision. Only the family of definitions (discrepancy-based) is stable across seeds.</span></a>

<a class="card" href="findings/language-transfer/"><b>Scaling transfers across languages</b><span>One 175M measurement plus an exponent pooled over other languages predicts a never-trained language's bits per byte at the reference size within 6.7 % (median).</span></a>

</div>

## Where to go next

- **See the models** — the [model ladder](ladder.md): every run, its data, its status and its per-language scores.
- **Understand the results** — the [findings](findings/index.md): one page per research question, each with its method and interactive results.
- **Choose benchmarks for your ablations** — the [recommender](recommend.md), with your own CSV if you have one.
- **Reproduce or extend** — the [repository docs](repo.md): pretraining, evaluation and analysis pipelines.
