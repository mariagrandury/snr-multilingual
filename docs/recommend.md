---
title: Find reliable benchmarks
---

# Find reliable benchmarks for your ablations

<!-- Our findings inform the following recommendations for multilingual ablations. Developers should check whether benchmark accuracy is distinguishable from chance at the available per-language training budget, reformat tasks to score answer text directly, and consider likelihood-based scores alongside accuracy. Smooth scaling curves and stable proxy rankings do not guarantee cross-scale agreement. Reliability should be assessed for the specific benchmark, language, and design intervention being tested. We release our model ladder, evaluation measurements, and per-benchmark guide to support these checks. -->

Our findings inform the following recommendations for multilingual ablations.

!!! success "Recommendations"
    1. **Check that accuracy is distinguishable from chance** at your
       per-language training budget. A score at chance ranks design choices at
       random ([Measurability](findings/measurability.md)).
    2. **Reformat tasks to score the answer text directly.** Letter-format
       multiple choice stays at chance at these sizes; the same items scored on
       the answer text become measurable
       ([Measurability](findings/measurability.md#answer-format-and-reformulation)).
    3. **Consider likelihood-based scores alongside accuracy.** Per-language
       bits per byte and bBPB preserve rankings more reliably than accuracy,
       above all at small sizes and early checkpoints
       ([Agreement](findings/agreement.md#sensitivity-analysis)).
    4. **Do not take smooth scaling or a stable proxy ranking as agreement.**
       Neither guarantees that the proxy orders design choices like the
       reference model ([Agreement](findings/agreement.md),
       [Anticipation](findings/anticipation.md)).
    5. **Assess reliability for your benchmark, language and design
       intervention.** Agreement depends on all three, so check the one you
       are testing in the tools below
       ([Agreement](findings/agreement.md#scale-agreement)).

Moreover, developers can use the tables below to adapt our findings to their custom use cases and design reliable multilingual ablations.

## 1. Above chance per training tokens

A benchmark can only rank model design decisions reliably once the model scores above chance on it. In our ladder, we find that it depends on model size and **how many tokens of the target language the model has seen** ([Measurability](findings/measurability.md)).

Check your own setup:

1. **Proxy size**: the model size of your ablation runs.
2. **Token budget**: the tokens one proxy run trains on. By default this is our
   budget for that size (100 tokens per parameter, 5× Chinchilla).
3. **Mixture**: the share of each language in the training data. You can start from one
   of our mixtures and edit the shares, or add and remove languages.

We calculate the tokens of each language your proxy sees multiplying share x budget. For each evaluation and language, the table gives the fewest tokens of that language from which the evaluation *stays* above chance in our runs of the same size. Your setup either reaches that number or it does not. A ✓ also says at what point in your run the evaluation becomes usable. Bits per byte has no chance level and is usable at every size.

<div class="viz" data-viz="lookup"></div>

<!--
??? note "How the minimum-tokens table is built"
    The table is a summary of the tokens-seen curves of RQ0: the share of runs above chance against the tokens of the language a checkpoint has seen. We read every evaluated tenth of every seed-1904 run of each size, so one point is one run
    configuration (number of languages K) at one checkpoint. A point is above
    chance when at least half of the benchmark's tasks in that language are
    above chance, using the [same rule](findings/measurability.md) as the rest of the site.
    The threshold is the smallest token count from which **every** point with
    at least as many tokens is above chance. A benchmark that falls back to chance
    later in training therefore gets no threshold. Results outside the token
    range we trained are marked "?", not guessed. Mixtures and token budgets
    are the ones listed under [Model ladder](ladder.md).
-->

## 2. Decision reliability per size

Select the language you care about, the size of the proxy models given your compute budget, and the reliability question you want the proxy to answer.

<div class="viz" data-viz="recommend"></div>
