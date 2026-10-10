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
often is not: many benchmark tasks sit at chance on small models, and those
that do not often rank two design choices differently at a small size and at
1.7B. We trained a controlled multilingual model ladder to measure, per
benchmark and per language, <b>when a small-scale evaluation predicts a
large-scale decision</b>.</p>

</div>

## The problem in one chart

<div class="viz" data-viz="hero"></div>

The chart counts every benchmark task that has a chance level, in every
evaluated language, whether or not the model trained on it.

A benchmark is useful for an ablation only if it answers three questions
well. Is its score **above chance** at the size and number of training tokens
you can afford? Does a smaller or earlier model **rank the design choices**
the way the fully trained reference model does? And can you tell **from the
small runs alone**, through a statistic such as the signal-to-noise ratio,
whether it will? This site shows which benchmarks pass, at which size, in
which language.

## Contributions

<div class="cards" markdown>

<a class="card" href="ladder/"><b>An open model ladder</b><span>218 runs from 90M to 3B non-embedding parameters, 1 to 50 training languages, seven single-choice interventions (depth, activation, optimizer, language list, sampling temperature, second language, English source) and seed replicates, each trained for 100 tokens per parameter.</span></a>

<a class="card" href="ladder/#evaluation"><b>Per-checkpoint measurements</b><span>Bits per byte in 100 languages and 40 benchmarks (846 benchmark-language tasks, chosen from 92 considered), on ten aligned fractions of every run: 207 models and 4,292 intermediate checkpoints.</span></a>

<a class="card" href="findings/"><b>Signal and noise, multilingual</b><span>The signal-to-noise framework extended from English to a controlled multilingual ladder: chance threshold, scaling, decision accuracy, noise, cheap predictors, language transfer and a check one size above the reference.</span></a>

<a class="card" href="recommend/"><b>A benchmark recommender</b><span>Pick your language, proxy size and decision, or upload your own results, and get the benchmarks that are reliable for your use case.</span></a>

</div>

## Key findings

<div class="cards" markdown>

<a class="card" href="findings/gate/"><b>Much of the suite is at chance</b><span>In the languages the models train on, only 37 % of the 841 analysed benchmark tasks with a chance level are above chance at 90M and 60 % at 1.7B; 37 % never pass at any size.</span></a>

<a class="card" href="findings/gate/"><b>The answer format decides who passes</b><span>Scoring the answer text instead of a letter lifts Belebele from 5 to 86 of 105 languages at 1.7B, Global-MMLU from 1 to 35 of 37 and INCLUDE from 3 to 31 of 43. More tokens of the language do not do it: letter-format INCLUDE stays at 5–11 % of its cells from 10<sup>7</sup> tokens on.</span></a>

<a class="card" href="findings/scaling/"><b>Above chance, scores scale</b><span>Where a task is above chance, a log-linear fit in model size explains most of its variance (median R² = 0.88), and per-language bits per byte fits almost perfectly (median R² ≈ 1.00).</span></a>

<a class="card" href="findings/decision-accuracy/"><b>Accuracy ranks near chance, bits per byte do not</b><span>Against the 1.7B reference, a fully trained smaller model agrees on 51–57 % of the design decisions read on benchmark accuracy (90M to 1B), but on 92–96 % of those read on per-language bits per byte, at every proxy except 600M (74 %, where the depth pair is not size-matched).</span></a>

<a class="card" href="findings/decision-accuracy/"><b>Training a proxy longer does not help</b><span>A proxy's ranking converges to its own final ranking, but its agreement with the 1.7B ranking stays flat over training. Two seeds of the same design already agree on 78–80 % of pairs at 90 % of a run, so early agreement is mostly persistence.</span></a>

<a class="card" href="findings/noise-snr/"><b>Seed noise is the yardstick</b><span>A score moves 1.75× more between seeds than between late checkpoints (median). Against seed noise, no design decision's effect clears 2× in half of its cells; depth's median is 1.31×.</span></a>

<a class="card" href="findings/surrogates/"><b>Cheap predictors reach ρ ≈ 0.6</b><span>Early-checkpoint ranking agreement and the relative-spread SNR predict decision accuracy best (Spearman ρ 0.54–0.62), already at 90M. The signal term alone and the margin above chance carry no information.</span></a>

<a class="card" href="findings/language-transfer/"><b>Scaling transfers across languages</b><span>One 175M measurement plus an exponent pooled over the other languages predicts a language's 1.7B bits per byte within 4.4 % (median), and within 5.0 % for languages the model never trained on.</span></a>

<a class="card" href="findings/size-generalisation/"><b>The same picture at 3B</b><span>One rung above the reference, benchmark accuracy ranks the 3B runs at chance from every proxy (0.47–0.53), while per-language bits per byte ranks them at 0.81–0.96.</span></a>

</div>

## Where to go next

- **See the models** — the [model ladder](ladder.md): every run, its data, its status and its per-language scores, also read one design decision at a time.
- **Understand the results** — the [findings](findings/index.md): one page per research question, each with its method and interactive results.
- **Choose benchmarks for your ablations** — the [evaluation recipe](findings/recipe.md) (on average, no way of posing and scoring a benchmark reads the 1.7B accuracy decision at a 0.75 bar from a proxy up to 1B; the best reaches 0.63) and the [recommender](recommend.md), with your own CSV if you have one.
- **Reproduce or extend** — the [repository docs](repo.md): pretraining, evaluation and analysis pipelines.
