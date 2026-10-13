---
title: Scaling Reliability
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Small Scale, Big Decisions: A Guide to Reliable Small-Scale Evaluation for Multilingual LLM Pretraining

</div>

## Abstract

Design choices for training large multilingual language models are typically
guided by results from ablations done using smaller, proxy models. However,
such results are only useful if evaluations of small-scale models reliably
predict a decision's effects in larger-scale models. In this work, we study
this question for multilingual LLMs, and pretrain a controlled ladder of 213
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

!!! success "Finding"
    We find that the per-language training tokens seen, expected answer
    format, and scoring strategy impact decision reliability more than changes
    to proxy model size across the scales we test. Question reformatting as
    cloze items makes Belebele, Global-MMLU and INCLUDE scores measurable where
    their original letter-formats are at chance. Once above chance, task
    accuracy scales predictably across model size and training checkpoints,
    and log-likelihood-based scoring on benchmark gold answers improves ranking
    agreement. However, statistics computed from the proxy runs alone used to
    predict ranking agreement in monolingual settings, do not transfer to our
    multilingual ladder. We convert the measurements into the per-benchmark
    guide released with this paper.

## Contributions

- An **[openly-available controlled multilingual scaling ladder](ladder.md)**
  together with evaluation measurements.[^release]
- A **[guide to evaluation configuration](findings/measurability.md#the-guide)**
  for multilingual ablations: per benchmark, the per-language training-token
  counts at which scores clear the above-chance threshold and at which we
  observe decision reliability.
- An **[analysis of general trends in decision reliability](findings/index.md)**
  to inform multilingual ablation design, including recommendations on task
  filtering, reformatting, and scoring.

[^release]: All artifacts will be publicly released upon publication.
