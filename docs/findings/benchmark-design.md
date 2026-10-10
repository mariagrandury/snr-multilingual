---
title: "RQ9 · Benchmark design"
---

# RQ9 · What makes a benchmark high-SNR?

**The question.** Some benchmarks separate the training choices cleanly and
others do not. Is it how the items were made (written in the language or
translated), the task format, the number of answer options, a reading
passage, the item length, or what the benchmark tests (its domain)?

**How we measure it.** We describe each of the 29 benchmark families that
clear the above-chance gate (512 tasks in 50 languages) by its
characteristics: curation, source, task format, answer options, reading
passage, domain and two item lengths. We then ask whether a characteristic
orders the families, and their tasks, by two measures: the **SNR** at the
1.7B reference, and the **DA-size**, how often the proxies 90M–1B rank the
design pairs as the 1.7B model does. DA-size is read on every task above
chance and on the paper's reliable tasks (those with a median DA of at least
0.66), with a Spearman ρ, a bootstrap interval that resamples whole families,
and a Kruskal–Wallis test.

## Explore

One dot per family: its median SNR at 1.7B against a design feature.

<div class="viz" data-viz="design"></div>

## Every task, grouped by a characteristic

Pick a characteristic and a measure: each dot is one benchmark in one
language, each row a level of the characteristic, the thick tick its median.
A characteristic that mattered would push whole rows apart; overlapping rows
mean it does not.

<div class="viz" data-viz="designStrip"></div>

## What we find

<!-- highlight: rq09_benchmark_design -->

- **The answer count orders the SNR, not the decisions.** The six
  two-option families reach a median SNR of 1.91 against 0.75 for the
  four-option ones (ρ = −0.65, interval −0.84 to −0.33), but against DA-size
  the same code gives ρ = −0.30 on the reliable tasks and −0.03 on every task
  above chance, both intervals across zero. The two-option families are also
  all continuation or minimal-pair formats, so format and option count cannot
  be told apart here.
- **No characteristic orders the benchmarks by DA-size in both readings.**
  On the reliable tasks the reading passage comes closest (ρ = 0.41, interval
  −0.02 to 0.73); on every task above chance only the source does (ρ = −0.39:
  the benchmarks translated from English decide better), and neither holds in
  the other reading.
- **Read task by task, the levels overlap.** On the reliable tasks the
  medians of every two-level characteristic lie between 0.63 and 0.67; on
  every task above chance the median task sits at chance agreement (0.50 to
  0.51) at every proxy size.
- **Knowledge benchmarks have a lower SNR, not a lower DA-size.** Their
  median SNR is 0.73 against 1.77 for the commonsense ones; their tasks
  read a DA-size of 0.50 against 0.59, but the domain's correlation with
  DA-size has an interval across zero in both readings.
- **HellaSwag is the clean example.** A four-option, translated benchmark
  with a passage, it has the highest DA-size on every task above chance (0.75)
  and the second-highest SNR (2.29).
- **The same items read sharper in their own language.** INCLUDE v2 in the
  item's language reaches an SNR of 0.89 against 0.68 for its English
  translation.

!!! success "Takeaway"
    A benchmark's design sheet does not tell you whether it will rank your
    ablations like a larger model. Two-option formats give the cleanest
    separation (the highest SNR), but no characteristic (curation, source,
    format, option count, passage or domain) predicts the decision
    accuracy: choose benchmarks by their measured decision accuracy, not by
    how they were built.

[Full analysis →](../signal-noise/benchmark_design.md)
