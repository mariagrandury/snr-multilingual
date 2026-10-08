---
title: "RQ0 · Above chance"
---

# RQ0 · Which benchmarks are above chance?

**The question.** A multiple-choice benchmark has a floor: a model that
guesses at random still gets some answers right (one in four with four
options). We call that floor **chance**. A small model whose score sits at
chance tells us nothing, and two such models are ranked by luck. So before
any other question, we ask: for each benchmark in each language, from which
model size are the scores above chance?

**How we measure it.** We score every run of the [model ladder](../ladder.md)
at the end of its training. A run *passes* when its score is above chance
with 95 % confidence, which depends on how many questions the benchmark has
(the last section explains the rule). A benchmark is *above chance at a size*
when at least half of the runs of that size that trained on its language
pass. Every later research question only uses the benchmarks and sizes that
are above chance.

## Which benchmarks are above chance?

<div class="viz" data-viz="firstSize"></div>

Most of the map is decided early or never. Reasoning-light tasks such as
grammatical minimal pairs (MultiBLiMP), sentence completion (HellaSwag) and
the reformatted reading test (Belebele RF) are above chance from small
models in most languages. Knowledge tests posed with answer letters
stay at chance in most languages even at 1.7B. Switch to *with rewritten
versions* to see the same knowledge tests once their format is fixed
(see [the format section](#how-does-the-format-affect-the-evaluation)).

## How does training data affect it?

Model size is not the only lever: a model can only answer in a language it
has read. Here every task, run and checkpoint is placed by how many tokens of
that task's language the run had seen at that point.

<div class="viz" data-viz="passTokens"></div>

For a fixed model size, more tokens of a language means more of its
benchmarks above chance; for a fixed number of tokens, larger models get
there more often. A language with a small share of the training mixture can
stay at chance even in a large model, which is why the same benchmark is
above chance in one language and not in another.

## How does the format affect the evaluation?

In our preliminary studies, knowledge benchmarks where the model has to
predict a letter (A, B, C, D) instead of the answer itself (Madrid, Delhi,
Cairo, Buenos Aires) almost never cleared chance at our model sizes. Small
base models have not yet learned the multiple-choice convention, so the
letter format measures the convention, not the knowledge. We therefore
**reformatted the benchmarks mechanically (RF)**: the same question, but the
model is scored on how likely it finds each full answer.

<div class="viz" data-viz="formatExample"></div>

We also tried **rewriting the questions with a large language model
(LLM-RF)**. Some questions do not read naturally once the answers are
attached to them (for example "Which of the following is true?"). We asked
an LLM to rewrite each question so that every answer is its natural
continuation, keeping the answers unchanged. We did this for Belebele and INCLUDE.

<div class="viz" data-viz="llmRewrite"></div>

The figure compares the three versions: for each benchmark and model size,
the share of its languages that are above chance. The number next to each
benchmark is how many languages it covers.

<div class="viz" data-viz="reformulation"></div>

!!! success "Key findings"
    - As published, the letter-format multilingual knowledge benchmarks
      (Belebele, INCLUDE, Global-MMLU) are above chance in almost no language
      at any size up to 1.7B.
    - Scored on the answer text, the same questions are above chance in most
      languages from the smallest sizes, and the change is significant for
      most sizes (stars).
    - The LLM rewrite helps most where the question does not read as a
      sentence to complete: it brings Belebele above chance in every language
      from the smallest model, and INCLUDE further than the mechanical
      reformat.
    - The English-only benchmarks (MMLU, CommonsenseQA: one language) were
      at chance as published and are above chance at every size once
      reformatted.

## Do larger public models pass this threshold?

Some tasks are at chance at every size of our ladder. Is that because our
models are too small, or because the task is too hard, or its language too
rare, for any model? We ran the same test on public models from 270M to 70B
parameters, scored on the same tasks with the same rule.

<div class="viz" data-viz="floors"></div>

Most of what our ladder cannot read is a **training-budget floor**, not a
benchmark floor: public models of 1.7B or less read it, but they were trained
on far more tokens than our ladder (100 tokens per parameter). The tasks that
need much larger models, or that no model reads, are floors of the benchmark
or of the language's data (hover to see which languages they are in).

## How do we decide a score is above chance?

**The intuition.** A score is an average over a finite set of questions, so
it is noisy: a model that guesses at random on a four-option test with 100
questions can easily score 30 % by luck. To call a run "above chance" we ask
that its score be above chance by more than that luck can explain. The
fewer the questions, the larger the margin we ask for. Concretely we compute
a **lower confidence bound** of the run's accuracy: a value the true accuracy
is above with 95 % confidence. A run passes when its lower bound is above
chance. A model size passes when at least half of its runs that trained on
the language pass, so one lucky run is not enough.

??? info "Theoretical definition"
    Let a run answer $k$ of the benchmark's $n$ questions correctly, so its
    accuracy is $\hat p = k / n$, and let the benchmark have $m$ answer
    options, so chance is $c = 1/m$ (uniform guessing). We use the one-sided
    95 % **Wilson score lower bound** of $\hat p$, with $z = 1.645$ (the
    95 % quantile of the standard normal distribution):

    $$
    \mathrm{LCB}(\hat p, n) \;=\;
    \frac{\hat p + \dfrac{z^2}{2n} - z\sqrt{\dfrac{\hat p\,(1-\hat p)}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}}
    $$

    The Wilson bound is preferred to the textbook $\hat p - z\sqrt{\hat p(1-\hat p)/n}$
    because it stays accurate for small $n$ and for accuracies near 0 or 1.

    1. **A run passes** a task when $\mathrm{LCB}(\hat p, n) > c$.
       Equivalently, its accuracy is above the smallest $\hat p$ that
       satisfies this, the *score needed*, which shrinks towards $c$ as $n$
       grows.
    2. **A task is above chance at a model size** when at least 50 % of the
       runs of that size whose training data contain the task's language pass.
    3. A task with no chance level (bits per byte, the training loss,
       open-ended generation) is never filtered.

    Two caveats. Chance is *uniform* guessing: a model that always picks the
    most frequent correct letter can beat $1/m$ without knowing anything,
    and the test does not catch it. And the runs of one size answer the same
    questions, so they are not independent trials; the 50 % rule is a
    majority vote, not a significance test.

Pick a benchmark and a language. Hover over any run to see its score and
lower bound; click a run to draw its bound along training.

<div class="viz" data-viz="gateExample"></div>

**(a) Every run, along training.** Each line is one run that trained on the
language, coloured by model size. The dotted line is chance, the dashed line
the score a run needs for its lower bound to clear chance with this many
questions. The shaded band under the highlighted run reaches down to its
lower bound: the run passes only if the bottom of the band, not the line
itself, is above chance.

**(b) The verdict per model size.** Each dot is one run's final score, and
its whisker reaches down to its lower bound; blue runs pass, grey runs fail.
Above each size: how many runs pass, and the verdict. At least half must
pass for the benchmark to count as above chance at that size.

**(c) What the verdict changes.** The later research questions ask whether
a small model ranks training choices like the 1.7B model (*DA-size*), and
whether a run at 90 % of its training ranks them like at its end
(*DA-ckpt*). A DA-size cell needs the benchmark above chance at the small
size *and* at 1.7B; a DA-ckpt cell needs it at that size. Cells marked
*gated* are left out of every result on this site.

## What we find

<!-- highlight: rq00_gate_and_curves -->

!!! success "Takeaway"
    Check that a benchmark is above chance at *your* proxy size, in *your*
    languages, before using it. Grammatical minimal pairs, sentence
    completion and reformatted reading comprehension carry signal from the
    smallest models; letter-format knowledge tests need to be scored on the
    answer text to be usable at these sizes.

[Full analysis →](../signal-noise/gate_and_curves.md)
