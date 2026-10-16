---
title: "Measurability: When Do Benchmarks Become Measurable?"
---

# Measurability: When Do Benchmarks Become Measurable?

A score that is statistically indistinguishable from chance provides no
reliable information about which model is better. We therefore first assess
the *measurability* of a training configuration on a task, which determines
whether a score is statistically distinguishable from chance, as a
prerequisite for examining decision reliability. For each training
configuration, we compute the one-sided 95% Wilson lower confidence bound on
its accuracy $\hat p$ over the task's $n$ items. We say that a training
configuration is *above chance* on the task when this bound exceeds $1/m_b$,
where $m_b$ is the number of answer choices; this test on one
configuration's result is the *above-chance filter*. Since the bound depends
on $n$, tasks with fewer items require a larger margin above chance.

This section examines, for each benchmark–language task, which training
configurations score above chance. We say that a model-size passes the
filter for a given benchmark–language task when at least half of the
training configurations at that model size score above chance. The
benchmark–language–model-size combinations that don't pass are excluded from
later analyses. BPB and bBPB are exempt from the filter because they have no
defined chance level. We report how this pass rate changes with model size
and as a function of *target-language tokens*, the number of training tokens
seen by the evaluated model on the given task's language ([Model Size and
Token Count Thresholds](#model-size-and-token-count-thresholds)). We then
relate it to the answer format ([Answer Format and
Reformulation](#answer-format-and-reformulation)) and ask whether model
performance on the filtered tasks scales predictably ([Benchmark Scaling
Behavior](#benchmark-scaling-behavior)).

## Model Size and Token Count Thresholds

We analyse 938 benchmark–language tasks. For only 38% of tasks does every
model size pass the above-chance filter; for 34% of tasks, no model up to
1.7B passes the filter. The tasks on which no model size passes the filter
are predominantly from benchmarks requiring letter-format answers like
Belebele, Global-MMLU and INCLUDE, for which the filter retains at most three
language tasks at any model size. In contrast, on benchmarks scored on the
answer text, such as HellaSwag (Okapi) and MultiBLiMP, training
configurations score above chance in all trained languages even at our
smallest size (90M). This finding suggests that evaluation format affects
measurability, which we explore below ([Answer Format and
Reformulation](#answer-format-and-reformulation)).

Model size is not the only factor that determines whether a configuration
scores significantly above chance on a task. We also ask how many
target-language tokens a model needs to have seen for it to pass the filter
on a given task. Across non-English benchmarks, training configurations with
≤100M target-language tokens score above chance on 42% of tasks, while the
proportion reaches 64% with 10B–100B tokens. Accuracy therefore stays at
chance on a third of tasks even after training on 10B tokens of the target
language, the full Chinchilla budget of a 500M monolingual model. Some
notable exceptions are HellaSwag (Okapi), MultiBLiMP and TokSuite, on which
nearly all training configurations with fewer than 100M target-language
tokens score above chance. The effect of target-language tokens also varies
with model size, with 43% of training configurations at 90M parameters
scoring above chance after 100M–1B target-language tokens, compared with 62%
at 1.7B. These findings suggest that both the number of target-language
tokens and model size influence measurability ([the guide](#the-guide)).

<div class="viz" data-viz="paperRq0"></div>

*Original and reformulated INCLUDE results across training. Pick any
benchmark to see its plot.*

## Answer Format and Reformulation

Multiple-choice benchmarks, where the answer is a letter, require mapping
answer labels to their content, an ability that may only emerge at larger
scales or with fine-tuning, regardless of whether the model knows the correct
answer. To isolate the effect of task format, we reformulate these
benchmarks to score the text of each answer option directly. We consider two
approaches: *programmatic reformulation*, which removes the answer letters
without changing the question or the options, and *model-based
reformulation*, which uses `gemini-3.8-flash` to rewrite each question and
options as declarative statements, without revealing the correct answer.

These reformulations substantially increase the number of languages for
which training configurations pass the above-chance filter. At the 1.7B
size, programmatic reformulation increases this number from 5 to 86 of 105
languages for Belebele, from 1 to 35 of 37 for Global-MMLU, and from 3 to 31
of 43 for INCLUDE. The figure above illustrates this effect for INCLUDE. In
the original letter format, accuracy rarely exceeds chance, while in the
reformulated version, the share of training configurations that pass the
filter rises as training progresses. Model-based reformulation yields further
improvements: on Belebele, every model size passes the filter in all trained
languages.

## Benchmark Scaling Behavior

Next, we examine how predictably performance scales along both dimensions.
We assess scaling predictability by fitting task performance against log
model size at the final checkpoint and against log training tokens across
checkpoints, using $R^2$ to measure goodness of fit and Spearman's $\rho$ to
measure monotonicity and direction.

<div class="viz" data-viz="paperRq1"></div>

***Benchmark scaling predictability.** (a) Median $R^2$ versus Spearman
$\rho$ across model-size fits for each benchmark, showing scaling
predictability and direction. (b) Median $R^2$ across model-size and
training-trajectory fits for selected benchmarks, with each point
representing a benchmark–language task.*

Panel (a) shows substantial variation in scaling predictability across
benchmarks. Across 431 tasks retained by the above-chance filter, the median
$R^2$ of the model-size fits is 0.87, reaching 1.00 for the BPB across 50
languages. For the letter-format benchmarks (Global-MMLU, INCLUDE and
Belebele), too few model sizes pass the filter for fitting, but the
model-size fits of their reformulated versions achieve a median $R^2$ of
0.84–0.92. A high $R^2$, though, does not necessarily imply improvement
across scales. For example, TruthfulQA has a median model size-fit $R^2$ of
0.97 but $\rho=-0.75$, indicating a strong and consistent decrease in
predictability with model size.

Panel (b) compares scaling predictability across model size and training
checkpoints. Among accuracy and BPB tasks with valid fits along both
dimensions, 74% (311 tasks) achieve $R^2 \geq 0.5$ on both. Notably, no
accuracy or BPB task exhibits predictable scaling across training
checkpoints without also scaling predictably across model sizes, although
the reverse is not always true. The same pattern holds for monolingual
English models. We therefore recommend, within the scales we test, spending
ablation compute on additional model sizes rather than on a longer run of
one size: the model-size fits give a consistent trend on every task the
training fits do, and on more.

## The guide

To make these results practically useful, the table below provides, per
benchmark and answer format, the *above-chance thresholds*: the
target-language token counts at which at least 50% and 90% of the
evaluations pass the filter.

<div class="viz" data-viz="gateThresholdTable"></div>

!!! success "Summary"
    A large share of the suite does not perform measurably on these
    benchmarks. In 34% of the 938 analysed benchmark–language tasks, no model
    size up to 1.7B passes the above-chance filter; these tasks are
    predominantly those scored using multiple-choice answer letters. Whether a
    training configuration scores above chance on a task depends on both the
    target-language tokens it has seen and its model size. On the tasks the
    filter retains, accuracy scales predictably with model size and with
    training tokens.
