---
title: "Measurability: When Do Benchmarks Become Measurable?"
---

# Measurability: When Do Benchmarks Become Measurable?

A score that is statistically indistinguishable from chance provides no
reliable information about which model is better. We therefore first assess
*measurability*, which determines whether a score is statistically
distinguishable from chance, as a prerequisite for examining decision
reliability. For each training configuration, we compute the one-sided 95%
Wilson lower confidence bound on its accuracy $\hat p$ over the task's $n$
items. We say that a training configuration is *above chance* on the task when
this bound exceeds $1/m_b$, where $m_b$ is the number of answer choices. Since
the bound depends on $n$, tasks with fewer items require a larger margin above
chance.

In this section, we determine which training configurations lead to models
that perform better than the task's corresponding chance level for each
benchmark–language task. A benchmark–language–model-size combination passes
this filter when, for that model size, at least half of the training
configurations score above chance. The rest are excluded from later analyses.
BPB and bBPB are exempt from this filtering because they have no defined
chance level. We report how this pass rate changes as a function of the
per-language training tokens ([Model Size and Token Count
Thresholds](#model-size-and-token-count-thresholds)), and relate it to the
answer format ([Factors Influencing Chance
Threshold](#factors-influencing-chance-threshold)). We then ask whether model
performance on the filtered tasks scales predictably ([Benchmark Scaling
Behavior](#benchmark-scaling-behavior)).

## Model Size and Token Count Thresholds

We analyse 938 benchmark–language tasks. Only 38% of these tasks pass the
above-chance filter for every model size, and for 34%, no model size up to
1.7B passes the above-chance filter. The filtered tasks are predominantly from
benchmarks requiring letter-format answers like Belebele, Global-MMLU and
INCLUDE, for which at most three languages subsets pass the filter. In
contrast, on benchmarks scored on the answer text, such as HellaSwag (Okapi)
and MultiBLiMP, models achieve above-chance performance in all trained
languages even at our smallest size (90M). This suggests that evaluation
format also affects measurability, which we explore below in [Factors
Influencing Chance Threshold](#factors-influencing-chance-threshold).

Model size is not the only factor that determines whether a configuration
scores significantly above chance on a task. We also ask how many
target-language tokens a model needs to be trained on for this threshold to be
passed. Across non-English benchmarks, 42% of tasks score above chance when at
most 100M target-language tokens are trained on, while the proportion reaches
64% with 10B–100B tokens. Some notable exceptions are HellaSwag (Okapi),
MultiBLiMP and TokSuite, which evaluate language modelling and grammatical
abilities, and score above chance in nearly all configurations trained with
fewer than 100M target-language tokens. The effect of training tokens also
varies with model size, with 43% of training configurations at 90M parameters
scoring above chance after 100M–1B target-language training tokens, compared
with 62% at 1.7B. These findings suggest that both the number of
target-language training tokens and model size influence measurability.

<div class="viz" data-viz="paperRq0"></div>

*Reformulation helps benchmarks pass the above chance threshold. The share of
(language, $K$, checkpoint) evaluations above chance against the training
tokens of the language that the checkpoint had seen, for the original letter
format and for the version scored on the answer text, one line per model size.
Pick any benchmark; INCLUDE is the example of the paper.*

## Factors Influencing Chance Threshold

Multiple-choice benchmarks, where the answer is a letter, require mapping
answer labels to their content, an ability that may only emerge at larger
scales or with fine-tuning, regardless of whether the model knows the correct
answer. To isolate the effect of task format, we reformulate these benchmarks
to score the text of each answer option directly. We consider two approaches:
*programmatic reformulation*, which removes the answer letters without
changing the question or the options, and *model-based reformulation*, which
uses `gemini-3.8-flash` to rewrite each question and options as declarative
statements, without revealing the correct answer.

These reformulations substantially increase the number of languages for which
training configurations pass the above-chance threshold. At the 1.7B size,
programmatic reformulation increases this number from 5 to 86 of 105 languages
for Belebele, from 1 to 35 of 37 for Global-MMLU, and from 3 to 31 of 43 for
INCLUDE. The figure above illustrates this effect for INCLUDE, where the
original letter-based format rarely exceeds chance, while the reformulated
version increasingly passes the threshold as training progresses. Model-based
reformulation yields further improvements, with Belebele passing in all
trained languages at every model size.

## Benchmark Scaling Behavior

Next, we examine how predictably performance scales along both dimensions. We
assess scaling predictability by fitting task performance against log model
size at the final checkpoint and against log training tokens across
checkpoints, using $R^2$ to measure goodness of fit and Spearman's $\rho$ to
measure monotonicity and direction.

<div class="viz" data-viz="paperRq1"></div>

*Benchmark scaling predictability. (a) Median $R^2$ versus Spearman $\rho$
across model-size fits, showing scaling predictability and direction. (b)
Median $R^2$ across model-size and training-trajectory fits, comparing
predictability across the two dimensions.*

Panel (a) shows substantial variation in scaling predictability across
benchmarks. Across 389 tasks retained by the above-chance filter, the median
$R^2$ of the model-size fits is 0.87, reaching 1.00 for the BPB of the 30
languages trained at two or more language settings. Letter-format benchmarks
(Global-MMLU, INCLUDE and Belebele) lack sufficient above-chance model sizes
for fitting, but their reformulated versions achieve a median $R^2$ of
0.84–0.92. A high $R^2$, though, does not necessarily imply improvement across
scales. For example, the Spanish TruthfulQA task has a median model size-fit
$R^2$ of 0.97 but $\rho=-0.75$, indicating a strong and consistent decrease in
accuracy with model size.

Panel (b) compares scaling predictability across model size and training
checkpoints. Among accuracy and BPB tasks with valid fits along both
dimensions, 74% (311 tasks) achieve $R^2 \geq 0.5$ on both. Notably, no
accuracy or BPB task exhibits predictable scaling across training checkpoints
without also scaling predictably across model sizes, although the reverse is
not always true. The same pattern holds for monolingual English models. We
therefore recommend, within the scales we test, spending ablation compute on
additional model sizes rather than on a longer run of one size: the
model-size fits give a consistent trend on every task the training fits do,
and on more.

## The guide

To turn the results from this section into a practical guide, the table below
provides, per benchmark and answer format, the training tokens of the language
from which at least 50% and 90% of the evaluations pass the above-chance
threshold.

<div class="viz" data-viz="gateThresholdTable"></div>

!!! success "Finding"
    A large share of the suite is not measurable at these scales: in 34% of
    the 938 analyzed tasks, models up to 1.7B never score above chance. These
    tasks are disproportionately those that require multiple-choice
    letter-format answers. Whether performance on a task is measurable depends
    on both the training tokens of its language and on model size. Past the
    above chance threshold, language tasks scale predictably across model size
    and training checkpoints.
