---
theme: scholarly
layout: section
transition:
footerLeft: EPFL NLP
footerMiddle: "Small scale, big decisions"
description: "Small Scale, Big Decisions: Evaluating Decision Reliability in Multilingual LLM Pretraining"
aspectRatio: 16/9
lang: en
themeConfig:
  colorTheme: classic-blue
  fontTheme: contemporary
  colorMode: dark
  sectionMode: dark
authors:
  - name: "María Grandury"
  - name: "Angelika Romanou"
  - name: "Éléonore Hasler"
  - name: Clara Meister
  - name: "Antoine Bosselut"
---

# Small Scale, Big Decisions

## Evaluating Decision Reliability in Multilingual LLM Pretraining

*María Grandury, Angelika Romanou, Éléonore Hasler,*

*Catherine TODO, Clara Meister, Antoine Bosselut*


---
layout: center
---

<div class="opening">

People choose expensive pretraining options from evaluations of small or partially trained models.

**When does that practice make the same decision as the full model in multilingual pretraining?**

</div>

<!--
Design choices such as data mixtures, architecture and optimizer are decided on small proxy models or early checkpoints.
That works only if the proxy ranks the options as the large model would. We call this decision reliability.
Magnusson et al. (2025) studied it in English. Multilingual pretraining adds per-language token budgets, tokenization cost, cross-lingual transfer and interference, and many tasks that stay at chance at small scale.
Our answer: it depends on the proxy size, the tokens seen per language, the item format and the scoring rule.
-->

---
layout: agenda
title: Outline
items:
  - Experimental Setup
  - "RQ1 Measurability: When is a task above chance?"
  - "RQ2 Agreement: How small and early can a proxy be?"
  - "RQ3 Anticipation: Can proxy statistics predict agreement?"
---

<!--
The three questions follow the order in which a developer should ask them.
First, is the score distinguishable from chance at my model size and token budget?
Second, does the proxy rank the variants as the reference does?
Third, can I tell in advance from the proxy runs alone?

TEST
-->

---
layout: section
---

# Experimental Setup

We pretrain 218 models from 90M to 3B,

trained on 1 to 50 languages along six design axes.

---
layout: figure
title: Experimental Setup
subtitle: The Model Ladder
---

::figure::
<Interactive viz="ladder" :zoom="0.85" />

<!--
218 models at seven sizes, from 90M to 3B non-embedding parameters. The 90M to 1B models are the proxies, 1.7B is the reference, and 3B checks whether the conclusions hold one size up.
All models follow the Apertus recipe with a budget of D = 100N tokens (5× Chinchilla) and a warmup-stable-decay schedule. Pretraining used 199,200 GPU hours.
The six design axes are the number of languages K (1, 2, 8, 15, 30 or 50), the language-selection scheme (A ranks by available data, B by diversity, C is the alternative at K ≤ 2), the sampling temperature (1 or 3), depth, the optimizer (AdEMAMix or Muon) and the activation (xIELU or SwiGLU).
Every configuration trains with seed 1904. Replicate seeds at 175M, 600M and 1B estimate seed noise.
Click a square to see the run's data, budget and per-language BPB.
-->

---
layout: figure
image: /tables/table01_notation-models.png
fit: contain
height: 40vh
title: "Experimental Setup"
subtitle: "Table: notation for the model ladder"
---

<!--
N is the non-embedding parameter count. The axes are depth, optimizer, activation, number of languages K, language-selection scheme M and sampling temperature T. Seed S is 1904 by default. Training progress f is the share of a run's token budget, so f = 0.2 is one Chinchilla multiple.
-->

---
layout: figure
image: /tables/table02_ladder.png
fit: contain
height: 50vh
title: "Experimental Setup"
subtitle: "Table: pretraining hyperparameters per ladder and model size"
---

<!--
N counts non-embedding parameters, and the tied embedding adds d_model × 131,072. The budget is D = 100N tokens. Grouped-query attention with four query heads per key-value head. The SwiGLU ladder narrows the FFN so that the three projections match the deep parameter count. 90M and 175M train at a smaller batch with proportionally more steps.
-->

---
layout: figure
image: /tables/table03_training-grid.png
fit: contain
height: 40vh
title: "Experimental Setup"
subtitle: "Table: the training grid, one row per data build and ladder"
---

<!--
A data build fixes the language-selection scheme M and the sampling temperature T. A ladder fixes the model recipe: deep, shallow, deep with SwiGLU or deep with Muon. 218 runs in total, 20 of them replicate seeds.
-->

---
layout: figure
title: Experimental Setup
subtitle: The data mixtures
---

::figure::
<Interactive viz="mixtures" :zoom="0.85" />

<!--
Each mixture is half English (DCLM-Edu) and half FineWeb-2, split over the other K−1 languages.
Scheme A takes the languages with the most data. Scheme B covers as many scripts and families as it can.
At temperature 1 each language gets its share of the corpus. Temperature 3 flattens the split towards the smaller languages.
The lists are nested: the K = 8 list is inside the K = 15 list, and so on up to K = 50.
-->

---
layout: figure
title: "Experimental Setup"
subtitle: "Table: the training languages"
---

::figure::
<div class="scroll-fig"><img src="/tables/table05_app-languages.png"></div>

<!--
For each FineWeb-2 language: its script and family, the smallest K whose scheme-A list and scheme-B list contain it, and the number of benchmark families the harness offers for it. The lists are nested, so a language also trains at every larger K of its scheme. The languages marked val are only in the 100-language validation set.
-->

---
layout: figure
title: Experimental Setup
subtitle: 52 benchmarks in 50 languages, scored at every tenth of training
---

::figure::
<Interactive viz="scoreCurves" :zoom="0.85" />

<!--
We considered 95 benchmarks with 1,488 tasks and analyse 52 benchmarks with 938 tasks, counting the RF and LLM-RF versions separately. These cover 20 multilingual benchmark families.
Every run is evaluated at each tenth of its training, plus 85% and 95% for the noise window, and only on the languages it trains on.
Items come in three formats: letter, cloze and completion. We score them with accuracy and with bBPB, the bits per byte of the gold answer, which has no chance level.
Per-language BPB on a shared 100-language validation set is the loss-based reference signal.
-->


---
layout: figure
title: "Experimental Setup"
subtitle: "Table: every benchmark considered"
---

::figure::
<div class="scroll-fig"><img src="/tables/table07_app_benchmarks_considered.png"></div>

<!--
Analysed: 40 benchmarks with 695 tasks, the reformulated versions folded into the row of their original. Screened and not analysed: 26 benchmarks with 485 tasks.
Langs is the number of languages the harness registers. Trained is how many of them the ladder trains. The last column lists the benchmark's languages among ranks 30 to 50 of our 50-language list.
-->

---
layout: default
title: Experimental Setup
subtitle: Decision accuracy (DA)
---

$$\text{DA}(N,f)=\frac{1}{|\mathcal P|}\sum_{(a,b)\in\mathcal P}\mathbb 1\big[\,s_{ab}(N,f)=s_{ab}(N_\text{ref},1)\,\big]$$

| | Proxy | Reference | Question |
|---|---|---|---|
| DA-size | $(N,\ f{=}1)$ | $(N_\text{ref},\ f{=}1)$ | How small? |
| DA-ckpt | $(N,\ f{<}1)$ | $(N,\ f{=}1)$ | How early? |
| DA-target | $(N,\ f{<}1)$ | $(N_\text{ref},\ f{=}1)$ | How small and how early? |

<!--
DA is the share of pairwise preferences that the proxy keeps from the reference (Magnusson et al. 2025).
A pair tied on one side only counts as a disagreement. A random ranking without ties scores 0.5.
A (task, proxy) cell needs at least three pairs.
Single-axis pairs differ in one design choice, as an ablation does. Multi-axis pairs may differ in several, as on a leaderboard.
The main text calls the third setting DA-target. Some appendix figures still label it DA-goal.
-->

---
layout: figure
image: /tables/table09_ra_defs.png
fit: contain
height: 30vh
title: "Experimental Setup"
subtitle: "Table: the three ranking-agreement settings"
---

<!--
The proxy and the reference rank the same variants but differ in model size, training progress or both. N_ref is the 1.7B reference, or 3B in the size-generalisation appendix.
-->

---
layout: figure
image: /paper/app_rq02_agreement_correlation.png
fit: contain
height: 46vh
title: Experimental Setup
subtitle: DA, Kendall's τ and Spearman's ρ measure nearly the same ranking agreement
---

<!--
Over 6,803 (task, proxy) values of DA-size, DA correlates with Kendall's τ_b at r = 0.99 and with Spearman's ρ at r = 0.97.
2·DA − 1 = τ_a + (T_both − T_one)/n, so DA and τ_a differ only in how they treat ties.
We therefore report DA throughout.
-->

---
layout: section
---

# RQ1: Measurability

At which model size and per-language token count is a task above chance?

---
layout: default
title: "RQ1: Measurability"
subtitle: Definition of the above-chance threshold
---

<div class="definition">

$$\mathrm{LCB}_{95}(\hat p) = \frac{\hat p + \frac{z^2}{2n} - z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \qquad z = 1.645$$

A model is above chance on a task when $\mathrm{LCB}_{95}(\hat p) > 1/m_b$.

A model size passes the task when at least half of its models are above chance.

</div>

<!--
For each language task b_ℓ and model size N, we test whether the observed accuracy can be told apart from random guessing.
Chance is 1/m_b for a task with m_b options. Where the option count varies across items, chance is the mean of 1/m over the items (0.2253 for TruthfulQA MC1).
We take the final-checkpoint accuracy p̂ over the n scored items, recover round(p̂ n) correct answers, and compute the one-sided 95% Wilson lower bound.
The vote is over the models of one size, every ladder and data build at seed 1904, and only over models whose mixture trains the task's language.
A scaling fit uses only the sizes where the task passes. A ranking against the reference also needs the task to pass at the reference size.
BPB and bBPB have no chance level and are not filtered.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: "The above-chance threshold: the one-sided 95% Wilson bound must exceed 1/m"
---

::figure::
<Interactive viz="gateMarginCalc" :zoom="0.9" />

<!--
A configuration is above chance when the one-sided 95% Wilson lower bound on its accuracy exceeds 1/m, where m is the number of options.
A model size passes a task when at least half of its configurations are above chance. The (task, size) cells that fail are dropped from later analyses.
Small benchmarks need a larger margin. The median margin is 9.00 points for non-parallel Global PIQA (100 items), 2.44 for Belebele (900), 0.74 for HellaSwag (about 9,000) and 0.60 for Global-MMLU (about 14,000).
BPB and bBPB have no chance level, so the threshold does not apply to them.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: Letter-format benchmarks pass only at 3B, cloze benchmarks from 90M
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq00_chance_share_considered_counts_3b.png"></div>

<!--
Of the 933 analysed tasks with a chance level, 40% are above chance at 90M and 64% at 1.7B. 34% never pass at any size.
Up to 1.7B, the letter-format Belebele, Global-MMLU and INCLUDE pass in only 3 of 49, 1 of 29 and 2 of 36 trained languages, and MMLU and CommonsenseQA stay at chance in English. Most of these languages pass only at 3B.
Among the screened benchmarks, MMMLU stays at chance in all 12 trained languages and MMLU (Okapi) in 26 of 27, even at 3B.
HellaSwag (Okapi) and MultiBLiMP pass at 90M in every trained language.
TruthfulQA (MC2), ToxiGen, BBH and ACP-Bench stay at chance up to 3B in every format.
Scroll for the full list.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: Interactive version, the smallest size above chance per benchmark and language
---

::figure::
<Interactive viz="firstSize" :zoom="0.85" />

<!--
The buttons switch the layout, add or remove the reformulated versions, and include the screened benchmarks.
Orange marks a task that never passes. White marks a language without items.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: Every benchmark in every trained language
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq00_chance_full.png"></div>

<!--
The full grid behind the previous slides: the smallest size at which each language task passes.
The columns that stay red are the letter-format benchmarks.
-->

---
layout: figure
image: /paper/rq0.png
fit: contain
height: 50vh
title: "RQ1: Measurability"
subtitle: "Figure 1: reformulation moves INCLUDE above chance"
---

<!--
The share of (language, K, checkpoint) evaluations above chance, against the training tokens of the language.
In the original letter format (top), the share stays near zero at every size and token count. Scored on the answer text (bottom), it rises from about 100M tokens and reaches nearly 100% by 10B.
Outside English, 42% of the cells below 100M tokens are above chance, 55% between 100M and 10B, and 64% beyond 10B. A third stay at chance after 10B tokens, the full Chinchilla budget of a 500M monolingual model.
Size matters as well. At 100M to 1B tokens, 43% of the 90M runs are above chance against 62% of the 1.7B runs. Exposure to the language is necessary but not sufficient.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: Interactive version of Figure 1 for any benchmark
---

::figure::
<Interactive viz="paperRq0" :zoom="0.85" />

<!--
Pick a benchmark. The button adds the LLM-RF version.
Good examples are HellaSwag, above chance almost everywhere, ARC (Okapi), which needs about 10B tokens, and Belebele in its three formats.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: Outside English, every decade of tokens from 10⁷ on raises the share above chance
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq00_chance_vs_train_tokens.png"></div>

<!--
One panel per benchmark and one line per model size.
HellaSwag (Okapi), MultiBLiMP and TokSuite pass in nearly every run below 100M tokens.
Both versions of ARC need about 10B tokens.
Pooled over every language, the curve dips in the top bins because those are mostly English.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: "Table: training tokens of the language each size needs before a benchmark is above chance"
---

::figure::
<div class="scroll-fig"><img src="/tables/table08_app_rq0_thresholds.png"></div>

<!--
For each non-English benchmark and answer format, the tokens of the language at which 50% and 90% of its (language, K, checkpoint) cells are above chance. Lang. is the number of non-English trained languages. Margin is the median accuracy above chance a task needs for its Wilson bound to clear chance. Items is the median item count.
The next slide is the interactive version.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: "The guide: tokens of the language needed for 50% and 90% of the cells to pass"
---

::figure::
<Interactive viz="gateThresholdTable" :zoom="0.85" />

<!--
This is the paper's threshold table, per benchmark, answer format and model size.
HellaSwag, MultiBLiMP and the LLM-RF Belebele reach 90% with the fewest tokens.
The letter-format Belebele, Global-MMLU and INCLUDE, BLEnD, CulturalBench-hard and TruthfulQA (MC2) never reach 50%. Their reformatted versions do.
ARC (Okapi) reaches 50% after 3.3B tokens at 600M and after 320M at 1.7B. A 1.7B model needs about ten times fewer tokens of a language than a 90M one.
The later analyses do not apply these token thresholds, because dropping the cells below them lowers the DA of the aggregate scores.
-->

---
layout: focus
color: amber
---

## Most tasks at chance expect the letter of the answer

We score the answer text instead

<!--
Mapping a label to its content is a separate ability that small models may lack, whether or not they know the answer (Khatun and Brown 2024, Madaan et al. 2024).
The programmatic reformatting (RF) drops the letters and scores the text of each option.
The model-based reformulation (LLM-RF) asks gemini-3.8-flash to restate each question as a declarative stem and each option as a statement that completes it, without revealing the answer.
The rewrite adds no length cue: the correct option is the single longest in 22.2% of the original items and in 21.9% of the rewritten ones, against 25% by chance.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: One INCLUDE item as letter, RF and LLM-RF
---

::figure::
<Interactive viz="llmRewrite" :zoom="0.85" />

<!--
The same item in the three formats. RF keeps the question and scores the option texts. LLM-RF turns each option into a natural continuation of the stem.
Pick another benchmark or language to inspect the rewrites. We reviewed the generations by hand in six languages.
-->

---
layout: figure
image: /paper/app_rq00_chance_reformulation.png
fit: contain
height: 46vh
title: "RQ1: Measurability"
subtitle: Dropping the letters moves whole benchmarks across the threshold
---

<!--
At 1.7B, RF raises the number of languages above chance from 5 to 86 of 105 for Belebele, from 1 to 35 of 37 for Global-MMLU and from 3 to 31 of 43 for INCLUDE.
LLM-RF Belebele passes in every trained language at every size. LLM-RF INCLUDE passes in 32 of 43.
The INCLUDE accuracy gain from the original to RF widens with size, from 0.015 at 175M to 0.075 at 1.7B.
A star marks a size where the McNemar test on the paired verdicts gives p < 0.05.
-->

---
layout: figure
image: /paper/rq1.png
fit: contain
height: 46vh
title: "RQ1: Measurability"
subtitle: "Figure 2: benchmark scaling predictability"
---

<!--
We fit the final-checkpoint score against log N (at least three sizes) and the score along a run against log tokens (at least five checkpoints), and report R² and Spearman's ρ.
(a) Above chance, the median size-fit R² is 0.87 over 389 benchmark tasks and 1.00 for the per-language BPB of 30 languages. HellaSwag reaches 0.88 and LAMBADA 0.96. The letter-format Global-MMLU, INCLUDE and Belebele have no fit because they stay at chance. Their RF versions fit at 0.84 to 0.92.
A high R² does not mean improvement: Spanish TruthfulQA-Multi has R² 0.97 with ρ = −0.75.
(b) 311 of the 419 accuracy and BPB tasks (74%) are predictable along both axes. None is predictable along training but not along size. Only 9 of the 688 bBPB tasks are.
So in our setting, comparing larger models earlier tells more than training smaller models longer.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: Interactive version of Figure 2a, one dot per benchmark
---

::figure::
<Interactive viz="paperRq1" panel="a" :zoom="0.85" />

---
layout: figure
title: "RQ1: Measurability"
subtitle: Interactive version of Figure 2b, fits along size and along training
---

::figure::
<Interactive viz="paperRq1" panel="b" :zoom="0.85" />

---
layout: figure
title: "RQ1: Measurability"
subtitle: The reformulated versions rise with model size while the letter originals stay at chance
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq00_benchmark_size_curves.png"></div>

<!--
Final-checkpoint accuracy against non-embedding parameters, one line per design. These curves are not filtered by the threshold.
Belebele, Global-MMLU and INCLUDE stay on the red chance line while their RF and LLM-RF versions climb.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: Along training, the same split
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq00_benchmark_curves.png"></div>

<!--
Accuracy at each evaluated tenth of a run, in Chinchilla multiples.
The letter formats stay at chance for the whole run. Their reformulations increase steadily.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: Interactive version, scaling curves per benchmark
---

::figure::
<Interactive viz="scalingCurves" :zoom="0.85" />

---
layout: figure
title: "RQ1: Measurability"
subtitle: On English-only models, the commonly used English benchmarks scale predictably
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq13_english_scaling_regimes_l1_accuracy_tagged.png"></div>

<!--
The K = 1 models trained on three corpora: DCLM-Edu, DCLM without the educational filter, and FineWeb.
In all three, 17 of the 39 English accuracy tasks are predictable along both size and training. They include ARC-Easy and ARC-Challenge, HellaSwag, OpenBookQA, LAMBADA, XNLI, XStoryCloze, XWinograd and the RF forms of MMLU, Global-MMLU, Belebele and CommonsenseQA, whose letter forms never pass.
Global PIQA is the only popular benchmark above chance that does not scale predictably.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: Interactive version, English scaling regimes
---

::figure::
<Interactive viz="englishRegimes" :zoom="0.85" />

---
layout: default
title: "RQ1: Measurability"
subtitle: Summary
---

<div class="summary">

A large share of the suite is not measurable on these models. On 34% of the 938 analysed benchmark–language tasks, no model size up to 1.7B passes the above-chance threshold, and most of these tasks are scored on the letter of the answer.

Whether a run is above chance depends on both the tokens of the language it has seen and its size.

On the tasks that pass, accuracy scales predictably with model size and along training.

</div>

<!--
The paper's Summary box for RQ1.
34% never pass up to 1.7B. Reformulating the letter-format benchmarks moves most of their languages above chance. Accuracy then scales predictably along both axes.
-->

---
layout: section
---

# RQ2: Agreement

How small and how early can a proxy be while it keeps the reference decision?

---
layout: figure
image: /paper/rq2.png
fit: contain
height: 40vh
title: "RQ2: Agreement"
subtitle: "Figure 3: decision reliability across scale and training"
---

<!--
Single-axis pairs, on the accuracy and bBPB tasks whose median DA-size or DA-ckpt is at least 0.66.
(b) Within one model, early checkpoints recover 62 to 68% of the final decisions, and 83 to 93% at 4.5C.
(a) Against the 1.7B reference, the fully trained smaller models agree at about 0.60 to 0.64 over all pairs, with no steady gain from size. Sampling temperature is the easiest axis, at about 0.72 to 0.75.
(c) The smaller models stay at about 0.58 to 0.64 whatever their training. Only the 1.7B run rises towards its own end.
So convergence within a model does not carry across scale, and training a cheap proxy longer does not close the gap.
The Summary box reads 0.51 to 0.56 for accuracy at every proxy on all pairs, and 0.74 to 0.94 for per-language BPB.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: Interactive version of Figure 3a, scale agreement per design axis
---

::figure::
<Interactive viz="paperRq2" panel="size" :zoom="0.9" />

---
layout: figure
title: "RQ2: Agreement"
subtitle: Interactive version of Figure 3b, agreement within one model along training
---

::figure::
<Interactive viz="paperRq2" panel="ckpt" :zoom="0.9" />

---
layout: figure
title: "RQ2: Agreement"
subtitle: Interactive version of Figure 3c, agreement with the target
---

::figure::
<Interactive viz="paperRq2" panel="target" :zoom="0.9" />

---
layout: figure
image: /paper/da_sensitivity.png
fit: contain
height: 43vh
title: "RQ2: Agreement"
subtitle: Near-tied reference scores raise DA but leave the trends unchanged
---

<!--
With a tolerance ε, reference gaps of at most ε count as ties. At ε = 0.5 points, 755 reversed pairs become agreements, 1,099 at 1 point and 1,308 at 2 points.
The DA-size curves move up and stay parallel, so the proxies do not converge with size. DA-ckpt moves up and still rises with training.
-->

---
layout: figure
image: /tables/table10_da_tolerance.png
fit: contain
height: 36vh
title: "RQ2: Agreement"
subtitle: "Table: sensitivity of DA to the reference-gap tolerance ε"
---

<!--
Within tolerance counts the pairs whose absolute gap at the reference is at most ε. Only reversed pairs are reclassified as agreements. At ε = 0.5 points, 755 reversals become ties, 1,099 at 1 point and 1,308 at 2 points.
-->

---
layout: figure
image: /paper/da_categories.png
fit: contain
height: 43vh
title: "RQ2: Agreement"
subtitle: Larger and later proxies miss fewer of the effects that persist at the reference
---

<!--
Each pair falls in one of four categories at ε = 0.5 points: tie at both points, a difference that disappears, a difference that appears only at the reference, or a difference at both.
With size and training, the effects that appear only at the reference become rarer and the differences at both points more common.
The exact ranking stays hard to predict, but whether an intervention matters becomes clearer.
-->

---
layout: figure
image: /paper/app_rq02_bbpb.png
fit: contain
height: 46vh
title: "RQ2: Agreement"
subtitle: Figure 3 split by scoring, accuracy on top and bBPB below
---

<!--
On accuracy, DA-size grows with the proxy, from 0.56 at 90M to 0.69 at 1B on the reliable tasks.
On bBPB it stays at 0.60 to 0.65, so much of the reference ranking is already visible at the smallest proxy.
bBPB also starts DA-ckpt from a higher early value, which suggests a smoother likelihood signal.
-->

---
layout: figure
image: /paper/app_rq02_bbpb_acc_gate.png
fit: contain
height: 40vh
title: "RQ2: Agreement"
subtitle: Where accuracy is at chance, bBPB still ranks the variants, and BPB ranks them best
---

<!--
On the tasks at chance at the proxy, bBPB ranks the variants as the 1.7B bBPB does at 0.60 to 0.63, against 0.46 to 0.50 for accuracy on the same tasks.
The macro average of per-language BPB on the same pairs reaches 0.95 to 0.98.
Against the 1.7B accuracy, the proxy's bBPB agrees more often than its accuracy on 57 to 71% of the tasks, by 0.10 at 90M down to 0.03 at 1B.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: Interactive version, accuracy, bBPB and BPB on the same pairs
---

::figure::
<Interactive viz="daChannels" :zoom="0.85" />

---
layout: figure
image: /paper/app_rq02_da_goal_multi_axes_bpb.png
fit: contain
height: 40vh
title: "RQ2: Agreement"
subtitle: On per-language BPB, the 600M rung is the outlier, not the smallest proxy
---

<!--
DA-target of per-language BPB against the tokens of the language the checkpoint has seen.
The 90M proxies recover 90% of the 1.7B decisions at the first tenth of their run, after about 8 million tokens of a language. 175M reaches 94% at its final checkpoint.
The 600M proxies stay at 0.69 to 0.74 over their whole run, while 350M and 1B climb to 0.89 and 0.92.
The tokens seen do not set the level, and more training does not close the 600M gap.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: Agreement per benchmark and language on the reliable tasks
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq02_da_by_language.png"></div>

<!--
Averaged over the proxies 90M to 1B, DA-size is above 0.5 (95% interval) on 7 of 9 benchmarks and 32 of 40 languages. DA-target is above 0.5 on 7 of 9 and 31 of 40, DA-ckpt on 11 of 11 and 38 of 40.
The mean cell reads 0.65 for DA-size, 0.71 for DA-ckpt and 0.61 for DA-target.
-->

---
layout: figure
image: /paper/app_rq02_decisive_pairs.png
fit: contain
height: 40vh
title: "RQ2: Agreement"
subtitle: On the pairs the reference separates beyond seed noise, DA rises
---

<!--
A pair is decisive when its 1.7B gap exceeds k·√2 seed standard deviations, for k = 1 or 2.
On benchmarks, DA goes from about 0.57 to 0.59 on every pair to 0.69 to 0.73 at two seed standard deviations. Only about a quarter of the benchmark pairs are that decisive.
Per-language BPB is near perfect on decisive pairs, and its 600M dip shrinks.
Seed noise in the reference explains part of the disagreement.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: Interactive version, decisive pairs at one or two seed standard deviations
---

::figure::
<Interactive viz="daDecisive" :zoom="0.85" />

---
layout: figure
image: /paper/app_rq10_size_generalisation.png
fit: contain
height: 40vh
title: "RQ2: Agreement"
subtitle: A 3B reference measures more tasks but decides no differently
---

<!--
Six configurations trained to 3B: deep, scheme A at K = 8, 15, 30 and 50, scheme B at K = 8 and 15.
At 3B, 778 of 933 tasks pass the threshold against 612 at 1.7B, mostly the letter-format Belebele, Global-MMLU and INCLUDE.
Per-language BPB carries over. The 1.7B models predict the 3B decisions at 0.94 to 0.95, and every proxy from 90M on reaches at least 0.78.
Benchmark accuracy reads 0.48 to 0.53 against either reference, so its gap to BPB does not come from using 1.7B as the reference.
-->

---
layout: figure
image: /paper/rq2_3b_compare.png
fit: contain
height: 46vh
title: "RQ2: Agreement"
subtitle: Figure 3 against the 3B reference on top and the 1.7B reference below
---

<!--
The same six configurations, the same nine single-axis pairs and 553 tasks reliable under both references.
Only the language-count axis has three or more pairs. DA-ckpt does not depend on the reference.
Against 3B, even the 1.7B models reach only about 0.51 on accuracy and 0.95 on BPB.
-->

---
layout: figure
image: /paper/app_rq11_evaluation_recipe.png
fit: contain
height: 43vh
title: "RQ2: Agreement"
subtitle: Which way of evaluating a benchmark recovers the reference decision?
---

<!--
A benchmark can be read with its original, RF or LLM-RF items, scored by accuracy or by bBPB.
No way of reading a benchmark reaches a mean DA-size of 0.75 against the reference accuracy at any proxy up to 1B.
Against the same truth, the 1.7B accuracy, bBPB is the better pick for 20 of the 36 benchmarks and accuracy for 16.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: The best way to evaluate each benchmark, per proxy size
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq11_recipe_by_benchmark.png"></div>

<!--
At 90M, a benchmark's bBPB read against the 1.7B accuracy ranks the pairs better than its best accuracy format on all three top benchmarks where both exist.
Accuracy DA-size goes from 0.51 at 90M to 0.56 at 1B, against 0.62 to 0.66 for the bBPB versions.
-->

---
layout: figure
image: /paper/app_rq06_language_transfer.png
fit: contain
height: 40vh
title: "RQ2: Agreement"
subtitle: The language-list decision does not transfer to untrained languages
---

<!--
The scheme A against scheme B decision at K = 8, 15 and 30, read on the per-language BPB of four groups: languages both lists train, languages one list trains, untrained languages that share a script or family with a trained one, and unrelated languages.
At the proxy sizes, only the trained languages recover the decision.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: Languages trained together predict each other, English does not
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq06_cross_task_hellaswag.png"></div>

<!--
On HellaSwag, each cell is the DA-size of one language's proxy ranking against another language's 1.7B ranking.
German, Spanish, French and Italian, which the K = 8 list trains together, predict each other at 0.78, about as well as each predicts itself (0.79).
English ranks the other languages at 0.45, so English is a poor proxy for multilingual decisions.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: Interactive version, which language predicts which
---

::figure::
<Interactive viz="ltCross" :zoom="0.85" />

---
layout: figure
title: "RQ2: Agreement"
subtitle: Untrained relatives gain little
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq06_family_transfer.png"></div>

<!--
The K = 8 scheme-A run passes more untrained tasks of the same subfamily than the English-only run does, by up to 0.04 at 1B. On the other subfamilies the gain is at most 0.02.
-->

---
layout: default
title: "RQ2: Agreement"
subtitle: Summary
---

<div class="summary">

On accuracy, a proxy orders the variant pairs like the final 1.7B checkpoint close to pairwise chance: 0.51 to 0.56 at every proxy size. On per-language BPB the share is higher, 0.74 to 0.94 over the same proxies.

Seed noise in the reference explains part of the disagreement. On the pairs the 1.7B reference separates by more than two seed standard deviations, the proxies order 0.69 to 0.73 of them correctly.

</div>

<!--
The paper's Summary box for RQ2.
Within one model the ranking stabilises along training, but that does not carry across scale. A 3B reference does not change the picture.
-->

---
layout: section
---

# RQ3: Anticipation

Can statistics of the proxy runs alone predict decision accuracy?

---
layout: figure
title: "RQ3: Anticipation"
subtitle: "Table: the 22 signal definitions"
---

::figure::
<div class="scroll-fig"><img src="/tables/table11_sur-signals.png"></div>

<!--
The signal definitions of the code released with Heineman et al. (2025). The coverage signals read the scores as points of [0, 1] and exist for benchmarks only. Each SNR divides one of these signals by a noise estimate.
-->

---
layout: figure
title: "RQ3: Anticipation"
subtitle: "Table: the catalogue of statistics computed from the proxy runs alone"
---

::figure::
<div class="scroll-fig"><img src="/tables/table12_sur-catalogue.png"></div>

<!--
Grouped by the question each statistic asks. A pair's gap is measured in units of the two variants' checkpoint noise. Statistics that compare an early ranking with the final one are not scored against DA-ckpt, which they would reproduce.
-->

---
layout: figure
image: /paper/rq3.png
fit: contain
height: 45vh
title: "RQ3: Anticipation"
subtitle: "Figure 4: how well each surrogate predicts DA at each proxy size"
---

<!--
DA looks back: it needs the reference. A surrogate is a statistic computed from the proxy runs alone.
The figure gives Spearman's ρ between each surrogate and DA-size on all pairs, over the tasks above chance and their bBPB, at each proxy size.
No surrogate predicts DA consistently.
Early-checkpoint agreement (the 10% checkpoint against the final one) reaches ρ = 0.54 to 0.62, the relative-standard-deviation SNR 0.53 to 0.60, and the scaling-fit R² 0.45 to 0.49 from 350M on.
Signal alone (−0.16 to 0.00) and the margin above chance (−0.03 to 0.09) carry no information.
The noise definition matters. With k-fold benchmark noise instead of checkpoint noise, the pooled correlation of the relative-standard-deviation SNR drops from ρ = 0.75 to 0.21.
The signal is already moderate at 90M and does not grow with size.
-->

---
layout: figure
title: "RQ3: Anticipation"
subtitle: Interactive version of Figure 4
---

::figure::
<Interactive viz="paperRq3" :zoom="0.85" />

---
layout: figure
title: "RQ3: Anticipation"
subtitle: Interactive version, the correlations within each benchmark
---

::figure::
<Interactive viz="surrogatesBench" :zoom="0.85" />

<!--
The strongest pooled correlation is ρ = 0.75. Within HellaSwag, the best SNR definition reaches ρ = 0.93.
How well DA can be predicted depends on the benchmark. The appendix gives the correlations per stratum.
-->

---
layout: figure
title: "RQ3: Anticipation"
subtitle: "Table: correlation of each catalogue statistic with DA"
---

::figure::
<div class="scroll-fig"><img src="/tables/table13_sur-corr-catalogue.png"></div>

<!--
Spearman ρ over the benchmark tasks, one point per (benchmark, language) cluster, on every pair of variants and pooled over the proxy sizes 90M to 1B.
-->

---
layout: figure
title: "RQ3: Anticipation"
subtitle: "Table: the signal and noise grid"
---

::figure::
<div class="scroll-fig"><img src="/tables/table14_sur-corr-snr.png"></div>

<!--
Spearman ρ with DA of each signal alone, of its ratio to the relative checkpoint noise, and of its ratio to the relative benchmark noise. rel_dispersion over the checkpoint noise is the SNR of Heineman et al. (2025) as released.
The noise definition matters: with k-fold benchmark noise instead of checkpoint noise, the relative-standard-deviation SNR drops from ρ = 0.75 to 0.21.
-->

---
layout: figure
title: "RQ3: Anticipation"
subtitle: "Table: correlation with DA-size within one stratum"
---

::figure::
<div class="scroll-fig"><img src="/tables/table15_sur-corr-strata.png"></div>

<!--
Spearman ρ of a surrogate with DA-size within one proxy size, one language-count tier, one benchmark or the per-language BPB. Within HellaSwag the best SNR definition reaches ρ = 0.93.
-->

---
layout: figure
image: /paper/app_snr_monolingual.png
fit: contain
height: 37vh
title: "RQ3: Anticipation"
subtitle: SNR predicts DA on English models and stops doing so as the models become multilingual
---

<!--
Each point is one of the nine OLMES benchmarks at one proxy size: its log SNR against its DA-size.
On DataDecide (25 English recipes, 150M to 750M against 1B), SNR tracks DA as Heineman et al. (2025) report.
On our models, the association weakens as K grows.
-->

---
layout: figure
image: /tables/table16_app_snr_monolingual.png
fit: contain
height: 44vh
title: "RQ3: Anticipation"
subtitle: "Table: SNR and DA-size of the OLMES benchmarks on English and multilingual models"
---

<!--
Per benchmark, the median over the proxy sizes of SNR and of DA-size, for DataDecide and for our models at each K, with Spearman's ρ between log SNR and DA-size over all (benchmark, proxy) points.
-->

---
layout: figure
title: "RQ3: Anticipation"
subtitle: The relative-spread SNR definitions track DA best, as a group
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq04_surrogates.png"></div>

<!--
For each of the 22 SNR definitions, the mean over languages of the Pearson r between log10 SNR and DA (blue for DA-size, orange for DA-ckpt).
The relative-spread definitions (rel_std, rel_mpd, rel_dispersion and others) lead, but no single definition is best everywhere.
A seed holdout keeps the overall ranking.
-->

---
layout: figure
title: "RQ3: Anticipation"
subtitle: Interactive version, the 22 SNR definitions
---

::figure::
<Interactive viz="surrogatesVariants" :zoom="0.85" />

---
layout: figure
image: /paper/app_rq04_surrogates_by_scoring.png
fit: contain
height: 37vh
title: "RQ3: Anticipation"
subtitle: Figure 4 split by scoring into accuracy, bBPB and per-language BPB
---

<!--
The bBPB versions carry most of the pooled correlation and the lead of early-checkpoint agreement (ρ = 0.57 to 0.68 on bBPB).
On accuracy alone, the relative-standard-deviation SNR leads at 90M, 175M and 600M (ρ = 0.33 to 0.39).
On per-language BPB at 600M, the SNR definitions, signal alone and the scaling-fit R² all turn negative (ρ = −0.35 to −0.60).
This is what a surrogate cannot see. BPB scales almost perfectly (R² = 1.00), yet its DA-size at 600M is 0.74, because the sign of the gap at the reference is not in the proxy runs.
-->

---
layout: figure
title: "RQ3: Anticipation"
subtitle: Interactive version, the surrogates by scoring
---

::figure::
<Interactive viz="surrogatesScoring" :zoom="0.85" />

---
layout: figure
image: /paper/app_rq04_surrogates_mono_axis.png
fit: contain
height: 45vh
title: "RQ3: Anticipation"
subtitle: On single-axis pairs the correlations are weaker, and the relative-standard-deviation SNR leads at every size
---

<!--
Figure 4 redrawn with DA-size and early-checkpoint agreement on the single-axis pairs.
The relative-standard-deviation SNR reaches ρ = 0.37 to 0.46, against 0.24 to 0.36 for early-checkpoint agreement.
-->

---
layout: default
title: "RQ3: Anticipation"
subtitle: Summary
---

<div class="summary">

No statistic computed from the proxy runs predicts DA consistently across benchmarks or proxy sizes.

The two with the strongest association are early-checkpoint agreement, the DA between a run's 10% checkpoint and its final checkpoint (ρ = 0.54 to 0.62), and the relative-standard-deviation SNR, the spread of the variants' final scores divided by checkpoint noise (ρ = 0.53 to 0.60).

</div>

<!--
The paper's Summary box for RQ3.
Both are already informative at 90M and do not strengthen with size. Within one benchmark the association can be much stronger, up to ρ = 0.93 for HellaSwag.
-->

---
layout: section
---

# Benchmark design

Which properties make a multilingual benchmark reliable?

---
layout: figure
image: /paper/app_rq09_benchmark_design.png
fit: contain
height: 40vh
title: Benchmark design
subtitle: MultiBLiMP, HellaSwag and XStoryCloze separate the designs most sharply
---

<!--
Median SNR at 1.7B per benchmark family with at least five tasks.
Seven of the nine sharpest families have two answer options.
-->

---
layout: figure
title: Benchmark design
subtitle: Grouped by any single characteristic, the levels overlap
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq09_design_da_by_level.png"></div>

<!--
The 113 reliable tasks have a median DA-size of 0.65. The medians of the levels with at least five tasks lie between 0.55 and 0.72.
-->

---
layout: figure
image: /paper/app_rq09_design_da_correlation.png
fit: contain
height: 37vh
title: Benchmark design
subtitle: Item count orders the benchmarks by DA-size, and skill orders them by both SNR and DA
---

<!--
Item count correlates with DA-size at ρ = 0.63 but with SNR at only 0.10, within and across benchmarks.
Benchmarks close to language modelling have a higher SNR (0.39) and DA-size (0.51).
The number of answer options correlates with SNR (ρ = −0.67) but not with DA.
-->

---
layout: figure
title: "Benchmark design"
subtitle: "Table: the 41 benchmark families and their characteristics"
---

::figure::
<div class="scroll-fig"><img src="/tables/table17_app_rq09_benchmark_characteristics.png"></div>

<!--
17 families are translated from English, 12 are English originals evaluated in English, and 23 have four answer options. The domain is what the benchmark tests, as its own paper describes it.
-->

---
layout: section
---

# The recommender

From the findings to a choice of benchmarks for one ablation

---
layout: figure
title: The recommender
subtitle: Which evaluations are above chance for your proxy size, budget and mixture
---

::figure::
<Interactive viz="lookup" :zoom="0.85" />

<!--
The developer gives the proxy size, the token budget of one run and the share of each language in the mixture. The budget defaults to 100 tokens per parameter.
The tokens each language receives are its share times the budget.
For each evaluation and language, the table gives the fewest tokens of that language from which the evaluation stays above chance in our runs of the same size. A check mark also says at what point of the run the evaluation becomes usable.
This rule is stricter than the paper's thresholds table: it asks every later checkpoint to stay above chance.
BPB has no chance level and is usable at every size.
-->

---
layout: figure
title: The recommender
subtitle: Which evaluations rank the design variants as the 1.7B reference does
---

::figure::
<Interactive viz="recommend" :zoom="0.85" />

<!--
The developer picks the target language, the proxy size, the decision (will the ranking hold at 1.7B, or can I decide before the run ends) and the kind of pairs.
A benchmark gets "use" when it is above chance at that size (at least half of the runs) and its decision accuracy is at least 0.66, the paper's threshold.
The SNR column is the relative-standard-deviation SNR at the proxy size.
-->

---
layout: figure
image: /recommender_example.png
fit: contain
height: 50vh
title: The recommender
subtitle: "Example: German, a 350M proxy, will the one-axis ranking hold at 1.7B?"
---

<!--
For German at 350M, 8 of the 40 measurements pass. Per-language BPB is the most reliable, with a decision accuracy of 0.82.
The other seven are the bBPB of INCLUDE v2 RF (OG), ARC (Okapi), HellaSwag (Okapi), MultiBLiMP and INCLUDE LLM-RF (0.67 to 0.71), and the accuracy of HellaSwag (Okapi) and XNLI (0.67).
Global-MMLU in its letter format is above chance in only 8% of the runs, so it is not usable here. Its RF version is above chance in every run but reaches only 0.61.
BPB at 350M ranks like the 1.7B reference from 10% of the run.
Read it as a prior and confirm it on your own runs.
-->

---
layout: section
---

# Takeaways

---
layout: bullets
title: Takeaways
icon: "▸"
---

- Check that accuracy is above chance at your per-language token budget
- Score the answer text, not the letter
- Add likelihood scores such as per-language BPB and bBPB
- Spend ablation compute on more model sizes rather than longer runs
- Smooth scaling curves do not guarantee agreement across scale
- Check reliability for your benchmark, language and design axis

<!--
We release the 218-model ladder, every per-checkpoint measurement and the per-benchmark guide on the project site.
The broader question: how much performance have current models lost through unreliable proxy decisions?
-->

---
layout: section
---

# Released resources

---
layout: figure
image: /tables/table20_card-ladder.png
fit: contain
height: 40vh
title: "Released resources"
subtitle: "Table: the released model ladder"
---

<!--
The model card of the released ladder: every run as a Hugging Face model with its intermediate checkpoints.
-->

---
layout: figure
image: /tables/table18_card-builds.png
fit: contain
height: 36vh
title: "Released resources"
subtitle: "Table: the data builds"
---

<!--
One row per data build of the released ladder.
-->

---
layout: figure
image: /tables/table19_card-languages.png
fit: contain
height: 46vh
title: "Released resources"
subtitle: "Table: the languages of the released models"
---

<!--
The training languages of each language setting, as listed in the model card.
-->

---
layout: bullets
title: Discussion
icon: "▸"
---

- Is a DA of 0.5 to 0.6 a property of multilingual evaluation or of near-tied designs?
- How much of the gap comes from seed noise in the reference?
- Should ablations be decided on BPB and only reported on benchmarks?
- Which reference is right: 1.7B, 3B or a much larger model?
- Can the token thresholds be predicted for languages we did not train?
- Detecting a difference at the proxy and agreeing across scale are two separate targets

<!--
The 600M BPB case: the scaling is perfect and DA is still 0.74, so the surrogates cannot see the sign of the gap at the reference.
The paper names two follow-ups: a held-out-language test of the token thresholds, and separating detectability at the proxy from agreement across scale.
How often do real development decisions fall among the near ties?
-->

---
layout: bullets
title: Questions reviewers may raise
subtitle: Design and scope
icon: "▸"
---

- Most cells have a single seed: what is the noise floor of DA?
- A 1.7B reference is small, and only six configurations reach 3B
- Each design axis has few pairs, temperature only four
- One recipe: Apertus, 5× Chinchilla and one tokenizer
- No contamination check and no post-training
- Thin coverage of low-resource languages

<!--
Seeds: replicate seeds exist at 175M, 600M and 1B. On the pairs separated by two seed standard deviations, DA reaches 0.69 to 0.73. The choice of reference seed moves DA by up to 0.13.
3B: the conclusions hold. Accuracy reads 0.48 to 0.53 against either reference, and BPB carries over from 1.7B to 3B at 0.94 to 0.95.
Pairs: a cell needs at least three. We report single-axis and multi-axis pairs.
Recipe: the ladder is controlled by design, and the Limitations section says so.
Contamination: stated in the Limitations. The training data is web text.
Coverage: 50 trained languages with a median of eight benchmark families each.
-->

---
layout: bullets
title: Questions reviewers may raise
subtitle: Methodology
icon: "▸"
---

- Is filtering tasks by a median DA of at least 0.66 circular?
- The above-chance threshold changes the task set at each size
- Can the LLM rewrite change an item's meaning or reveal the answer?
- Is it fair to compare bBPB with accuracy when their targets differ?
- Does pooling the surrogates over multi-axis pairs inflate ρ?
- Why a 95% Wilson bound, and why half of the configurations?

<!--
The 0.66 filter is a heuristic that shows the trends on the tasks that rank at all. The appendix reports the unfiltered population, where accuracy DA is 0.51 to 0.56.
The changing task set: the Limitations report comparisons on a fixed population, and the guide gives the thresholds per size.
LLM-RF: we reviewed the rewrites by hand in six languages and iterated on the prompt. The length check gives 22.2% against 21.9%.
bBPB: we read it against its own 1.7B value and against the 1.7B accuracy. Against the accuracy it still agrees more often, by 0.03 to 0.10.
Pooling: the appendix gives the single-axis results, where the correlations are weaker and the relative-standard-deviation SNR leads, and a table per stratum.
Threshold: rerunning it with the majority class as chance lowers the share of tasks above chance by only 1.5 to 2.8 points.
-->

---
layout: bullets
title: Questions reviewers may raise
subtitle: Claims and framing
icon: "▸"
---

- Comparing larger models earlier was tested only up to 1.7B
- Why does 600M break per-language BPB?
- Does the guide hold for other mixtures and tokenizers?
- How does this compare with DataDecide in English?
- What should a team actually run?

<!--
Scope: the paper states the recommendation within the scales we test.
600M: the seed draw of the deep 600M runs, on pairs whose 1.7B gap is about one seed standard deviation. More training does not close it.
Guide: the thresholds are per benchmark, format and size. A held-out-language test is future work.
DataDecide: SNR predicts DA on the English OLMES benchmarks, and the association weakens as K grows (appendix figure).
A practical recipe: RF or LLM-RF formats, BPB and bBPB alongside accuracy, several small sizes, and early-checkpoint agreement and the relative-standard-deviation SNR as checks.
-->

---
layout: end
thankYou: Thank you
subtitle: "scaling-reliability.netlify.app"
---
