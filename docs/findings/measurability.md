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

??? note "Formal definition of the above-chance threshold"
    For each language task $b_\ell$ and model size $N$, we test whether the
    observed accuracy is distinguishable from random guessing. For a task with
    $m_b$ answer options, chance accuracy is $1/m_b$. Where the option count
    varies across items, chance is the mean of $1/m$ over the items (0.2253 for
    TruthfulQA MC1). For every model of size $N$, we take its final-checkpoint
    accuracy $\hat p$ over the $n$ scored items of the task. We recover the
    number of correct answers as $\mathrm{round}(\hat p n)$ and compute the
    one-sided 95% Wilson lower confidence bound on the proportion,

    $$\mathrm{LCB}_{95}(\hat p) = \frac{\hat p + \frac{z^2}{2n} - z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \qquad z = 1.645.$$

    The model is above chance when

    $$\mathrm{LCB}_{95}(\hat p) > \frac{1}{m_b}.$$

    We then aggregate these decisions over the models of the same size, i.e.,
    over every ladder and every data build at seed 1904. A $(b_\ell, N)$ pair
    passes the threshold if at least 50% of these models are above chance.
    Only the models whose mixture trains the task's language are considered.
    A scaling fit uses only the sizes where the task passes. A ranking against
    the reference also requires the task to pass at the reference size.
    Measurements without a meaningful chance baseline, such as BPB and bBPB,
    are not filtered by this procedure.

    <div class="viz" data-viz="gateMarginCalc" id="figure-4" data-caption="Figure 4: Required margin above chance."></div>


This section examines, for each benchmark–language task, which training
configurations score above chance. We say that a model-size passes the
filter for a given benchmark–language task when at least half of the
training configurations at that model size score above chance. The
benchmark–language–model-size combinations that don't pass are excluded from
later analyses. BPB and bBPB are exempt from the filter because they have no
defined chance level. We report how this pass rate changes with model size
and as a function of *target-language tokens*, the number of training tokens
seen by the evaluated model on the given task's language
([§1.1](#11-answer-format-and-reformulation)). We then relate it to the
answer format ([§1.2](#12-factors-influencing-chance-threshold)) and ask
whether model performance on the filtered tasks scales predictably
([§1.3](#13-benchmark-scaling-behavior)).

## 1.1 Answer Format and Reformulation

We analyse 938 benchmark–language tasks. For only 38% of tasks does every
model size pass the above-chance filter; for 34% of tasks, no model up to
1.7B passes the filter. The tasks on which no model size passes the filter
are predominantly from benchmarks requiring letter-format answers like
Belebele, Global-MMLU and INCLUDE, for which the filter retains at most three
language tasks at any model size. In contrast, on benchmarks scored on the
answer text, such as HellaSwag (Okapi) and MultiBLiMP, training
configurations score above chance in all trained languages even at our
smallest size (90M). This finding suggests that evaluation format affects
measurability, which we explore below
([§1.2](#12-factors-influencing-chance-threshold)).

<div class="viz" data-viz="firstSize" id="figure-5" data-caption="Figure 5: Language tasks that pass the above-chance threshold, per benchmark and model size." data-paper="Figure 6"></div>

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
tokens and model size influence measurability ([Table 2](#table-2)).

<div class="viz" data-viz="paperRq0" id="figure-6" data-caption="Figure 6: Original and reformulated (RF) INCLUDE results across training." data-text="Pick any benchmark to see its plot." data-paper="Figures 1 and 8"></div>

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

<div class="viz" data-viz="llmRewrite" id="figure-7" data-caption="Figure 7: Task reformulation."></div>

These reformulations substantially increase the number of languages for
which training configurations pass the above-chance filter. At the 1.7B
size, programmatic reformulation increases this number from 5 to 86 of 105
languages for Belebele, from 1 to 35 of 37 for Global-MMLU, and from 3 to 31
of 43 for INCLUDE. Figure 6 illustrates this effect for INCLUDE. In
the original letter format, accuracy rarely exceeds chance, while in the
reformulated version, the share of training configurations that pass the
filter rises as training progresses. Model-based reformulation yields further
improvements: on Belebele, every model size passes the filter in all trained
languages.

<div class="viz" data-viz="reformulation" id="figure-8" data-caption="Figure 8: Dropping the letters moves whole families across the threshold." data-paper="Figure 7"></div>

## 1.3 Benchmark Scaling Behavior

Next, we examine how predictably performance scales along both dimensions.
We assess scaling predictability by fitting task performance against log
model size at the final checkpoint and against log training tokens across
checkpoints, using $R^2$ to measure goodness of fit and Spearman's $\rho$ to
measure monotonicity and direction.

<div class="viz" data-viz="paperRq1" data-panel="a" id="figure-9" data-caption="Figure 9: Benchmark scaling predictability across model size." data-paper="Figure 2a"></div>

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

<div class="viz" data-viz="paperRq1" data-panel="b" id="figure-10" data-caption="Figure 10: Benchmark scaling predictability across model size and training." data-paper="Figure 2b"></div>

Figure 10 compares scaling predictability across model size and training
checkpoints. Among accuracy and BPB tasks with valid fits along both
dimensions, 74% (311 tasks) achieve $R^2 \geq 0.5$ on both. Notably, no
accuracy or BPB task exhibits predictable scaling across training
checkpoints without also scaling predictably across model sizes, although
the reverse is not always true. The same pattern holds for monolingual
English models (Figure 12). We therefore recommend, within the scales we test, spending
ablation compute on additional model sizes rather than on a longer run of
one size: the model-size fits give a consistent trend on every task the
training fits do, and on more.

<div class="viz" data-viz="scalingCurves" id="figure-11" data-caption="Figure 11: The reformulated versions rise with model size while the letter-format original benchmarks stay at chance up to 1.7B." data-paper="Figures 9 and 10"></div>

<div class="viz" data-viz="englishRegimes" id="figure-12" data-caption="Figure 12: The commonly used English benchmarks scale predictably." data-paper="Figure 11"></div>

### Above-chance thresholds

To make these results practically useful, Table 2 provides, per benchmark
and answer format, the *above-chance thresholds*: the target-language token
counts at which at least 50% and 90% of the evaluations pass the filter.

<div class="viz" data-viz="gateThresholdTable" id="table-2" data-caption="Table 2: Training tokens of the language a model of each size needs before a benchmark is above chance." data-paper="Table 7"></div>

!!! success "Summary"
    A large share of the suite does not perform measurably on these
    benchmarks. In 34% of the 938 analysed benchmark–language tasks, no model
    size up to 1.7B passes the above-chance filter; these tasks are
    predominantly those scored using multiple-choice answer letters. Whether a
    training configuration scores above chance on a task depends on both the
    target-language tokens it has seen and its model size. On the tasks the
    filter retains, accuracy scales predictably with model size and with
    training tokens.
