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
at the end of its training: 186 runs from 90M to 1.7B parameters, on 933
tasks (one benchmark in one language) from 52 benchmarks in the 50 languages
the ladder trains. A run *passes* when its score is above chance with 95 %
confidence, which depends on how many questions the benchmark has (the last
section explains the rule). A benchmark is *above chance at a size* when at
least half of the runs of that size that trained on its language pass. Every
later research question only uses the benchmarks and sizes that are above
chance.

## Which benchmarks are above chance?

<div class="viz" data-viz="firstSize"></div>

A large part of the evaluation is at chance on our ladder. Of the 933 tasks,
377 (40 %) are above chance at 90M and 599 (64 %) at 1.7B. The count grows
at every size step, but only 356 tasks pass at all six sizes, and 313 (34 %)
never pass at any size.

Most of the map is decided early or never. Twelve of the 52 benchmarks never
pass in any trained language, among them MMLU and CommonsenseQA in their
original letter format, TruthfulQA, BBH, ACP-Bench and CulturalBench. Fifteen
pass from 90M in every language they cover. Three of them cover many
languages: HellaSwag (26), MultiBLiMP (34) and the rewritten Belebele (49).
The other twelve cover one to five languages, among them MathQA, OpenBookQA
and the reformatted MMLU and CommonsenseQA. Knowledge
tests posed with answer letters stay at chance in most languages even at
1.7B. Switch to *with rewritten versions* to see the same knowledge tests
once their format is fixed (see
[the format section](#how-does-the-format-affect-the-evaluation)).

*The same map with one bar per benchmark: read each bar from left to right,
light segments are languages above chance from the smallest models, red ones
never.*

<div class="viz" data-viz="gateFirstShare"></div>

### Every benchmark we considered

The 52 benchmarks above are the ones the gate reads. To choose them we
considered 93 benchmarks with a chance level, 1,454 tasks in the trained
languages (95 benchmarks and 1,488 tasks with those that have no chance
level). The 33 benchmarks we screened but did not analyse (338 tasks) and
the 229 tasks we discarded were scored only at the end of the 600M, 1B and
1.7B runs. Three of the screened benchmarks, MMLU (Okapi) RF, MMMLU RF and
TokSuite (46 tasks), joined the suite late. They are in the map above, but
we count them as screened until their evaluation along training is
complete, so 887 tasks are analysed.

- **The screened benchmarks pass almost everywhere.** 85 % of their tasks
  are above chance at 600M, 95 % at 1B and 98 % at 1.7B (partly by
  construction: the screened tasks at chance at both 1B and 1.7B were moved
  to the discarded group). We left them out for cost, or because an
  analysed benchmark already covers them, not because they failed the test.
- **The discarded tasks fail everywhere.** All 229 are at chance at 1B and
  1.7B (that is the rule they were discarded by), and 227 at 600M too.
  Thirteen of the screened or discarded benchmarks never pass at 1.7B;
  eight of them are multiple-choice knowledge exams (ArabicMMLU, CareQA,
  C-Eval, MMMLU, TurkishMMLU, HAE-RAE Bench and two Arabic exam sets),
  while the reformatted versions of ArabicMMLU, CareQA and the Okapi MMLU
  pass in every task.
- **Over everything considered, 51 % of the tasks pass at 600M, 57 % at 1B
  and 61 % at 1.7B**, against 52 %, 57 % and 62 % of the 887 analysed ones.

*Pick a group of tasks. Each bar is one model size; dark blue is the share
above chance at that size, light blue the tasks at chance there but above
chance at another size, red the tasks never above chance.*

<div class="viz" data-viz="gateFunnel"></div>

## How does training data affect it?

Model size is not the only lever: a model can only answer in a language it
has read. Here every task, run and checkpoint is placed by how many tokens of
that task's language the run had seen at that point.

<div class="viz" data-viz="passTokens"></div>

Outside English, the share of cells above chance is flat at about 40 % up to
10⁸ tokens of the language and rises with every tenfold increase after, to
64 % at 10¹⁰–10¹¹ tokens. Exposure is necessary but not sufficient: at
10⁸–10⁹ tokens, 43 % of the 90M cells are above chance against 62 % of the
1.7B cells. A language with a small share of the training mixture can stay
at chance even in a large model, which is why the same benchmark is above
chance in one language and not in another. The format matters here too:
the letter-format INCLUDE stays at 3–17 % of its cells above chance from
10⁷ tokens on, while its reformatted version reaches 99–100 % from 10¹⁰ tokens.

**Token thresholds.** Per benchmark and format we read the token count of
the language from which at least half (or 90 %) of its cells are above
chance, with a logistic fit of the cells' verdict on the tokens. Ten of the
30 benchmark formats never reach half, among them the letter-format
Belebele, Global-MMLU and INCLUDE; their reformatted versions reach it at
4.4 × 10⁵ to 4.4 × 10⁸ tokens. The threshold also moves with the model size,
by nearly three orders of magnitude at the 90 % level: Belebele RF never
reaches it at 90M, needs 5.6 × 10⁹ tokens at 175M and 9 × 10⁶ at 1.7B. A
1.7B model needs about ten times fewer tokens of a language than a 90M one
before the same benchmark leaves chance.

*Drag the token count: each row is a benchmark format, its dot the
threshold. A blue dot left of the black line has reached it; red means it
never does. Switch the model size to see the thresholds move.*

<div class="viz" data-viz="gateTokenSlider"></div>

*The same thresholds as a table, per model size: the token count of the
language from which half (and 90 %) of a benchmark's cells are above chance.
Filter by format or name, and click a column to sort.*

<div class="viz" data-viz="gateThresholdTable"></div>

### Early passes come from a constant answer

A few tasks are above chance at very few tokens of their language and at
chance at intermediate ones: eight CulturalBench-easy countries, INCLUDE
Georgian and PAWS-X Japanese. These are not early knowledge.

- **The model gives one answer to almost everything.** Of the 149 runs that
  pass with few tokens, 74 % pick the same option on at least 80 % of the
  items, and in 97 % that option alone would clear chance, because the
  answer keys of these small tasks (17 to 53 items for CulturalBench) are
  unbalanced. The passes stop when the preferred option moves.
- **Not the longest option, not a related language.** A preference for the
  longest answer explains only PAWS-X Japanese. On the other nine tasks,
  runs that never trained the language pass as often at the same size and
  checkpoint (31–88 % against 30–79 %).

*One row per task: the share of runs above chance with few tokens of the
language (blue), with more tokens (grey) and without the language at all
(orange). Blue and orange together mean the language is not what passes.*

<div class="viz" data-viz="gateEarly"></div>

### Does dropping the languages below the threshold help?

A natural rule is to leave out of a benchmark's aggregate score every
language below its token threshold. It does not help the decisions. Over
the 22 benchmarks that keep an aggregate, the aggregate ranks pairs of
training designs like the 1.7B model in 71–77 % of the cases without the
excluded languages, against 75–82 % with every language (one design decision
at a time). The aggregate's standard error also rises, from 0.0088 to 0.0112
at 90M, since only 58 % of the cells are kept there. What the rule does
remove is the nine benchmarks that never reach their threshold, whose
aggregate ranks the designs at only 44–65 %.

*Each line is how often a benchmark's aggregate at a small model ranks two
designs like at 1.7B; higher is better. Switch to "per benchmark" to see
where the exclusion costs most.*

<div class="viz" data-viz="gateExclusion"></div>

## How does the format affect the evaluation?

Most of the tasks at chance ask for the letter of the correct option (A, B,
C, D) instead of the answer itself (Madrid, Delhi, Cairo, Buenos Aires).
Small base models have not yet learned the multiple-choice convention, so
the letter format measures the convention, not the knowledge. We therefore
**reformatted the benchmarks mechanically (RF)**: the same question, but the
model is scored on how likely it finds each full answer.

<div class="viz" data-viz="formatExample"></div>

We also tried **rewriting the questions with a large language model
(LLM-RF)**. Some questions do not read naturally once the answers are
attached to them (for example "Which of the following is true?"). We asked
an LLM to rewrite each question so that every answer is its natural
continuation, keeping the answers unchanged and without revealing the
correct one. We did this for Belebele and INCLUDE.

<div class="viz" data-viz="llmRewrite"></div>

The figure compares the three versions: for each benchmark and model size,
the share of its languages that are above chance. The number next to each
benchmark is how many languages it covers.

<div class="viz" data-viz="reformulation"></div>

At 1.7B the reformatted version raises the languages above chance from 5 to
86 of 105 for Belebele, from 1 to 35 of 37 for Global-MMLU and from 3 to 31
of 43 for INCLUDE (in each case the languages where only the original
passes are 0 or 1; McNemar p < 10⁻⁷). The rewritten Belebele passes in every
trained language at every size; the rewritten INCLUDE in 32 of 43, with an
accuracy gain over the original that widens from 0.015 at 175M to 0.075 at
1.7B. The rewrite adds no length cue: the correct option is the single
longest in 22.2 % of the original items and 21.9 % of the rewritten ones.
Passing, however, is not ranking: the reformatted versions do not rank
training designs better than the other benchmarks (decision accuracy 0.49 to
0.56 from 90M to 1B, against 0.53 to 0.57).

*The same comparison as a before-and-after per benchmark: the grey dot is
the benchmark as published, the line runs to its reformatted (blue) or
rewritten (orange) version. Pick a model size.*

<div class="viz" data-viz="gateReformDumbbell"></div>

!!! success "Key findings"
    - Of the 933 tasks the gate reads, 40 % are above chance at 90M and 64 %
      at 1.7B, and 34 % never are. Over all 1,454 tasks we considered, 61 %
      pass at 1.7B.
    - As published, the letter-format multilingual knowledge benchmarks
      (Belebele, INCLUDE, Global-MMLU) are above chance in almost no language
      at any size up to 1.7B: 5 of 105, 3 of 43 and 1 of 37 languages at 1.7B.
    - Scored on the answer text, the same questions are above chance in 86,
      31 and 35 of those languages at 1.7B, and the change is significant at
      most sizes (stars). Tokens of the language do not rescue the letter
      format: the letter-format INCLUDE stays at 3–17 % above chance however
      much of the language the model has read.
    - The LLM rewrite brings Belebele above chance in every language from the
      smallest model, and INCLUDE slightly further than the mechanical
      reformat.
    - The English-only benchmarks (MMLU, CommonsenseQA: one language) were
      at chance as published and are above chance at every size once
      reformatted.
    - The test is robust to its definition of chance: using the most
      frequent answer instead of uniform guessing changes the share above
      chance by less than three points wherever a constant answer is possible.

## Do larger public models pass this threshold?

Some tasks are at chance at every size of our ladder. Is that because our
models are too small, or because the task is too hard, or its language too
rare, for any model? We ran the same test on public models from 270M to 70B
parameters, scored on the same tasks with the same rule.

<div class="viz" data-viz="floors"></div>

Of the 74 tasks both are scored on, 27 are at chance at every size of our
ladder. Most of them are a **training-budget floor**, not a benchmark floor:
public models of 1.7B or less read 22 of them, but they were trained on far
more tokens than our ladder (100 tokens per parameter). Vietnamese TruthfulQA
needs a 4B model, Hindi ARC one of 12B or more and Arabic XNLI a 70B one.
PAWS-X Japanese and Chinese TruthfulQA are read by no model up to 70B. Those
are floors of the benchmark or of the language's data (hover to see which
languages they are in). Many are near misses: on 24 of
the 27, at least one of our 1.7B runs clears chance, but not half of them.
The public models' verdicts still use an older, fixed-margin version of the
test, so this comparison is provisional.

*The same comparison per model family: each line is a family from its
smallest to its largest model, at the share of the shared tasks it reads.
At the same size the public models, trained much longer, read far more.*

<div class="viz" data-viz="gateFloorsLines"></div>

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
    most frequent correct letter can beat $1/m$ without knowing anything.
    The section below measures how much that matters. And the runs of one
    size answer the same questions, so they are not independent trials; the
    50 % rule is a majority vote, not a significance test.

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

**The item count sets the bar.** The margin a run needs above chance shrinks
with the number of questions: a median 9.0 points for the non-parallel
Global PIQA (100 items per language), 2.4 for Belebele (900), 0.7 for
HellaSwag (about 9,000) and 0.6 for Global-MMLU (about 14,000), up to 15.7
points for CulturalBench-easy (25 items). The bar alone does not decide:
over the six sizes these benchmarks pass in 75.0 %, 1.4 %, 100 % and 2.3 %
of their cells, and the reformatted Belebele and Global-MMLU, with the same
items and so the same bar, in 89.5 % and 78.2 %. A benchmark with few items
cannot detect small departures from chance; that does not make it
unmeasurable.

*Drag the item count to read the bar for any benchmark size; each dot is one
of our tasks (blue: above chance at 1.7B).*

<div class="viz" data-viz="gateMarginCalc"></div>

### Does it matter that chance is uniform guessing?

The test compares every task with uniform guessing (1/m). A model that always
gives the same answer instead scores the share of the most frequent correct
position, which can be higher. We reran the test with that **majority-class**
chance level.

- **Taken at face value, it removes almost half of the passes.** The tasks
  above chance fall from 377 to 146 at 90M and from 599 to 426 at 1.7B.
- **Almost all of that is answer order a model cannot exploit.** 92 % of the
  lost passes are on tasks whose answers change from item to item
  (MultiBLiMP always lists the grammatical sentence first, for example). The
  model scores each answer's text and never sees its position, so it cannot
  answer "always the first one".
- **Where a constant answer is possible, the test barely moves.** On the
  tasks with fixed answer labels (letters, yes/no) the counts are 352 at 90M
  and 579 at 1.7B against 377 and 599, under three points at every size, and
  the later decision-accuracy results move by at most 0.006.

*Each line counts the tasks above chance per model size under one chance
level; the green line, the realistic one, sits on the blue one. Switch to
"per benchmark" to see which benchmarks lose passes.*

<div class="viz" data-viz="gateMajority"></div>

## What we find

<!-- highlight: rq00_gate_and_curves -->

!!! success "Takeaway"
    Check that a benchmark is above chance at *your* proxy size, in *your*
    languages, before using it. Grammatical minimal pairs, sentence
    completion and reformatted reading comprehension carry signal from the
    smallest models; letter-format knowledge tests need to be scored on the
    answer text to be usable at these sizes, however much of the language
    the model has read. And an early pass on a small task is worth a look at
    the answers: it is often one option given every time.

[Full analysis →](../signal-noise/gate_and_curves.md)
