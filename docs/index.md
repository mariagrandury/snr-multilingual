---
title: Scaling Reliability
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Small Scale, Big Decisions:

# Evaluating Decision Reliability in Multilingual LLM Pretraining

</div>

## Abstract

Design choices for training large multilingual language models are typically
guided by results from ablations done using smaller, proxy models. However,
such results are only useful if evaluations of small-scale models reliably
predict a decision's effects in larger-scale models. In this work, we study
this question for multilingual LLMs, and pretrain a controlled ladder of 218
multilingual models, spanning scales and language mixtures, and evaluate
language modelling and downstream performance on 20 multilingual benchmark
families. For each benchmark, we quantify how well the performance of smaller
models predicts larger model performance, how early in training this
correspondence emerges, and whether small-scale evaluations recover the
relative effects of design choices at full scale. Our results show that the
informativeness of small-scale multilingual results is set by multiple
factors: proxy model size, the training tokens seen per-language, the item
format, and the scoring rule. We convert these findings into concrete
per-benchmark guidance to support the identification of evaluations that can
lead to cheaper, better-informed multilingual model development.

## Findings

!!! success "Measurability: When Do Benchmarks Become Measurable?"
    A large share of the suite does not perform measurably on these
    benchmarks. In 34% of the 938 analysed benchmark–language tasks, no model
    size up to 1.7B passes the above-chance filter; these tasks are
    predominantly those scored using multiple-choice answer letters. Whether a
    training configuration scores above chance on a task depends on both the
    target-language tokens it has seen and its model size. On the tasks the
    filter retains, accuracy scales predictably with model size and with
    training tokens.
    [Read more](findings/measurability.md)

!!! success "Agreement: How Early Can We Trust Proxy Evaluations?"
    On accuracy-scored benchmarks, the share of variant pairs that a proxy
    point orders the same way as the final 1.7B checkpoint is close to
    pairwise chance: 0.51-0.56 at every proxy size. On per-language BPB, the
    share is higher, ranging from 0.74 to 0.94 over the same proxies.
    Seed-to-seed noise in the reference models accounts for part of the
    disagreement. On the design-variant pairs that the 1.7B reference model
    separates by more than two seed standard deviations, the share ordered
    correctly is 0.69 to 0.73.
    [Read more](findings/agreement.md)

!!! success "Anticipation: Can We Predict Ranking Agreement?"
    No metrics computed from proxy evaluations predicts DA consistently across
    benchmarks or proxy sizes. The two metrics with the strongest association
    are early-checkpoint ranking agreement, the DA between a run's 10%
    checkpoint and final checkpoint ($\rho = 0.54$–$0.62$), and the
    relative-standard-deviation SNR, the spread of the variants' final scores
    divided by checkpoint noise ($\rho = 0.53$–$0.60$).
    [Read more](findings/anticipation.md)

## Contributions

<div class="cards" markdown>

<a class="card" href="ladder/"><b>An open multilingual scaling ladder</b><span>An openly-available controlled multilingual scaling ladder together with evaluation measurements.</span></a>

<a class="card" href="recommend/"><b>A guide to evaluation configuration</b><span>Per benchmark, the per-language training-token counts at which scores clear the above-chance threshold and at which we observe decision reliability.</span></a>

<a class="card" href="findings/"><b>Trends in decision reliability</b><span>An analysis of general trends in decision reliability to inform multilingual ablation design, including recommendations on task filtering, reformatting, and scoring.</span></a>

</div>

## Conclusion

We study when small-scale multilingual evaluation supports reliable pretraining decisions using a controlled ladder of 218 models. Our findings challenge the assumption that ablations at smaller scales reliably identify the best choices for larger multilingual models. Measurable scores, smooth scaling curves, and stable proxy rankings do not guarantee cross-scale agreement, and increasing proxy size does not consistently resolve this problem within our ladder. Reformulating tasks can improve evaluations' decision reliability, but reliability remains conditional on the benchmark, language, and intervention axis. 

These results raise a broader question: how reliably do current model-development practices select the best pretraining choices? If proxy experiments are noisy or favour choices that become inferior at larger scales, they may lead developers to discard promising alternatives and limit the performance of the resulting models. Our study does not establish how often this occurs in practice or quantify its impact, but suggests the issue may be more prevalent than currently believed. Future work should therefore investigate two questions:  how much performance have current models lost through unreliable proxy decisions, and how can we better predict the effects of pretraining choices at the reference scale? We release our model ladder, evaluation measurements, and per-benchmark guide to help assess these conditions. 
