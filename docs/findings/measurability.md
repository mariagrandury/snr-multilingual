---
title: "Measurability: When Do Benchmarks Become Measurable?"
---

# Measurability: When Do Benchmarks Become Measurable?

A score that is statistically indistinguishable from chance provides no
reliable information about which model is better. For each training
configuration, we compute the one-sided 95% Wilson lower confidence bound on
its accuracy $\hat p$ over the task's $n$ items. We say that a training
configuration is *above chance* on the task when this bound exceeds $1/m_b$,
where $m_b$ is the number of answer choices; this test on one
configuration's result is the *above-chance filter*. Since the bound depends
on $n$, tasks with fewer items require a larger margin above chance.

<div class="viz" data-viz="gateMarginCalc"></div>

***Figure 4: Required margin above chance.** The margin above chance an accuracy needs for its one-sided 95% Wilson
lower bound to exceed chance shrinks with the number of items; the median
required margins are 9.00 points for non-parallel Global PIQA (100 items),
2.44 for Belebele (900), 0.74 for HellaSwag (about 9,000) and 0.60 for
Global-MMLU (about 14,000).*
{ #figure-4 }

We say that a model-size passes the
filter for a given benchmark–language task when at least half of the
training configurations at that model size score above chance. The
benchmark–language–model-size combinations that don't pass are excluded from
later analyses. BPB and bBPB are exempt from the filter because they have no
defined chance level. We report how this pass rate changes with model size
and as a function of *target-language tokens*, the number of training tokens
seen by the evaluated model on the given task's language
([§1.1](#11-answer-format-and-reformulation)).

## 1.1 Answer Format and Reformulation

We analyse 938 benchmark–language tasks. For only 38% of tasks does every
model size pass the above-chance filter; for 34% of tasks, no model up to
1.7B passes the filter. The tasks on which no model size passes the filter
are predominantly from benchmarks requiring letter-format answers like
Belebele, Global-MMLU and INCLUDE, for which the filter retains at most three
language tasks at any model size. In contrast, on benchmarks scored on the
answer text, such as HellaSwag (Okapi) and MultiBLiMP, training
configurations score above chance in all trained languages even at our
smallest size (90M).

<div class="viz" data-viz="firstSize"></div>

***Figure 5: Language tasks that pass the above-chance threshold, per benchmark and
model size.** Each cell is one benchmark in one language, coloured by the smallest model
size from which it passes the threshold. The screened benchmarks that we do
not analyse are left out. This figure corresponds to Figure 6 from the paper.*
{ #figure-5 }

Across non-English benchmarks, training configurations with
≤100M target-language tokens score above chance on 42% of tasks, while the
proportion reaches 64% with 10B–100B tokens. Accuracy therefore stays at
chance on a third of tasks even after training on 10B tokens of the target
language, the full Chinchilla budget of a 500M monolingual model. Some
notable exceptions are HellaSwag (Okapi), MultiBLiMP and TokSuite, on which
nearly all training configurations with fewer than 100M target-language
tokens score above chance. The effect of target-language tokens also varies
with model size, with 43% of training configurations at 90M parameters
scoring above chance after 100M–1B target-language tokens, compared with 62%
at 1.7B.

<div class="viz" data-viz="paperRq0"></div>

***Figure 6:** Original and reformulated (RF) INCLUDE results across
training. Pick any benchmark to see its plot. This figure corresponds to Figures 1 and 8 from the paper.*
{ #figure-6 }

## 1.2 Factors Influencing Chance Threshold

Multiple-choice benchmarks, where the answer is a letter, require mapping
answer labels to their content, an ability that may only emerge at larger
scales or with fine-tuning, regardless of whether the model knows the correct
answer. To isolate the effect of task format, we reformulate these
benchmarks to score the text of each answer option directly. We consider two
approaches: *programmatic reformulation*, which removes the answer letters
without changing the question or the options, and *model-based
reformulation*, which uses `gemini-3.8-flash` to rewrite each question and
options as declarative statements, without revealing the correct answer.

<div class="viz" data-viz="llmRewrite"></div>

***Figure 7: Task reformulation.** One item of a letter-format benchmark as published, after programmatic
reformulation (RF) and after model-based reformulation (LLM-RF).*
{ #figure-7 }

These reformulations substantially increase the number of languages for
which training configurations pass the above-chance filter. At the 1.7B
size, programmatic reformulation increases this number from 5 to 86 of 105
languages for Belebele, from 1 to 35 of 37 for Global-MMLU, and from 3 to 31
of 43 for INCLUDE. Figure 6 illustrates this effect for INCLUDE. Model-based reformulation yields further
improvements: on Belebele, every model size passes the filter in all trained
languages.

<div class="viz" data-viz="reformulation"></div>

***Figure 8: Dropping the letters moves whole families across the threshold.** For each letter-format benchmark (its number of languages in parentheses),
the share of tasks above the chance threshold for the original items (grey),
the RF version at each size (blue) and the LLM-RF version (orange diamonds);
a star marks a McNemar test with p < 0.05. This figure corresponds to Figure 7 from the paper.*
{ #figure-8 }

## 1.3 Benchmark Scaling Behavior

We assess scaling predictability by fitting task performance against log
model size at the final checkpoint and against log training tokens across
checkpoints, using $R^2$ to measure goodness of fit and Spearman's $\rho$ to
measure monotonicity and direction.

<div class="viz" data-viz="paperRq1" data-panel="a"></div>

***Figure 9: Benchmark scaling predictability across model size.** Median
$R^2$ versus Spearman $\rho$ across model-size fits for each benchmark,
showing scaling predictability and direction. This figure corresponds to
Figure 2a from the paper.*
{ #figure-9 }

Figure 9 shows substantial variation in scaling predictability across
benchmarks. Across 431 tasks retained by the above-chance filter, the median
$R^2$ of the model-size fits is 0.87, reaching 1.00 for the BPB across 50
languages. For the letter-format benchmarks (Global-MMLU, INCLUDE and
Belebele), too few model sizes pass the filter for fitting, but the
model-size fits of their reformulated versions achieve a median $R^2$ of
0.84–0.92. A high $R^2$, though, does not necessarily imply improvement
across scales. For example, TruthfulQA has a median model size-fit $R^2$ of
0.97 but $\rho=-0.75$, indicating a strong and consistent decrease in
predictability with model size.

<div class="viz" data-viz="paperRq1" data-panel="b"></div>

***Figure 10: Benchmark scaling predictability across model size and
training.** Median $R^2$ across model-size and training-trajectory fits for
selected benchmarks, with each point representing a benchmark–language task.
This figure corresponds to Figure 2b from the paper.*
{ #figure-10 }

Among accuracy and BPB tasks with valid fits along both
dimensions, 74% (311 tasks) achieve $R^2 \geq 0.5$ on both. Notably, no
accuracy or BPB task exhibits predictable scaling across training
checkpoints without also scaling predictably across model sizes, although
the reverse is not always true. The same pattern holds for monolingual
English models (Figure 12). We therefore recommend, within the scales we test, spending
ablation compute on additional model sizes rather than on a longer run of
one size: the model-size fits give a consistent trend on every task the
training fits do, and on more.

<div class="viz" data-viz="scalingCurves"></div>

***Figure 11: The reformulated versions rise with model size while the letter-format
original benchmarks stay at chance up to 1.7B.** Final-checkpoint accuracy against non-embedding parameters, or accuracy at
the ten evaluated tenths of each run against Chinchilla multiples of its token
budget, averaged over the tasks in the languages each configuration trains.
One line per (K, ladder, data build, seed); the dotted red line is chance.
These curves are not filtered by the above-chance threshold. This figure corresponds to Figures 9 and 10 from the paper.*
{ #figure-11 }

<div class="viz" data-viz="englishRegimes"></div>

***Figure 12: The commonly used English benchmarks scale predictably.** For the monolingual English models (K = 1) on DCLM-Edu, DCLM and FineWeb,
the R² of the model-size fit against the median R² of the training fits for
the 39 English accuracy tasks, coloured by regime. This figure corresponds to Figure 11 from the paper.*
{ #figure-12 }

### Above-chance thresholds

To make these results practically useful, Table 2 provides, per benchmark
and answer format, the *above-chance thresholds*: the target-language token
counts at which at least 50% and 90% of the evaluations pass the filter.

<div class="viz" data-viz="gateThresholdTable"></div>

***Table 2:** Training tokens of the language a model of each size needs
before a benchmark is above chance. This table corresponds to Table 7 from the paper.*
{ #table-2 }

!!! success "Summary"
    A large share of the suite does not perform measurably on these
    benchmarks. In 34% of the 938 analysed benchmark–language tasks, no model
    size up to 1.7B passes the above-chance filter; these tasks are
    predominantly those scored using multiple-choice answer letters. Whether a
    training configuration scores above chance on a task depends on both the
    target-language tokens it has seen and its model size. On the tasks the
    filter retains, accuracy scales predictably with model size and with
    training tokens.
