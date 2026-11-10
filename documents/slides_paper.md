---
theme: scholarly
layout: section
transition:
htmlAttrs:
  data-deck: paper
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

# Small Scale, Big Decisions:<br>Evaluating Decision Reliability in Multilingual LLM Pretraining

<br>

*María Grandury, Angelika Romanou, Éléonore Hasler,*

*Catherine Arnett, Clara Meister, Antoine Bosselut*


---
layout: figure
image: /motivation.gif
fit: contain
height: 46vh
title: Motivation
subtitle: "Every pretraining run starts with many design decisions"
---

<!--
Modern language model development relies on smaller proxy models and intermediate training checkpoints to guide decisions about data mixtures, architectures, tokenizers, and other training choices.
-->


---
layout: default
title: Motivation
subtitle: "Decision reliability"
---

Development relies on small proxies to decide what the large model gets. The practice assumes that the small run decides like the large one.

<div class="decision">
<div class="decision-col"><div class="decision-head">Proxy, 350M</div><ol><li><b>Option A</b></li><li>Option B</li></ol></div>
<div class="decision-mid">same choice?</div>
<div class="decision-col"><div class="decision-head">Reference, 1.7B</div><ol><li><b>Option A</b></li><li>Option B</li></ol></div>
</div>

<div class="decision-def">

**Decision reliability**: Whether decisions at small scale would be the same as those taken for models of the reference scale.

</div>

<!--
Every large run is the result of many such decisions. Only the winning option is ever trained at full scale.
This practice rests on the assumption that measured performance of smaller training runs is predictive of the performance of larger training runs.
The predictive property of an evaluation configuration (a benchmark task, an answer format and a scoring rule) is its decision reliability.
-->


---
layout: figure
image: /snr_paper_figure_2.png
fit: contain
height: 46vh
title: Motivation
subtitle: "In English, a benchmark's signal-to-noise ratio predicts its decision accuracy (Heineman et al., 2025)"
---

<!--
Heineman et al. (2025, AllenAI) define a benchmark's signal as how well it separates models and its noise as how much its score moves across the last checkpoints of a run.
Across their English benchmarks, SNR correlates with decision accuracy at R = 0.791, while signal or noise alone does not.
Their models are English and their benchmarks are already measurable at small scale. We ask whether this holds when the models and the benchmarks cover many languages.
-->

---
layout: default
title: Motivation
subtitle: "These findings come from English, and multilingual pretraining adds new factors"
---

<div class="summary">

**Why English findings may not transfer.**

- Languages vary in training mixture allocation, tokenization cost, and scaling behaviour.
- Cross-lingual transfer and interference couple decisions across languages: Adding data for one language can help or hurt others depending on model capacity and language similarity.
- Many multilingual language–benchmark pairs exhibit little measurable signal at small scales, with performance often remaining near chance or floor levels.

</div>

<!--
Decision reliability has been studied (Magnusson et al., 2025), but only in English settings.
These challenges suggest that decision reliability findings in monolingual settings cannot be transferred directly to multilingual settings.
-->

---
layout: center
---

<div class="opening">

People choose expensive pretraining options from evaluations of small or partially trained models.

**When does that practice make the same decision as the full model in multilingual pretraining?**

</div>

<!--
-->

---
layout: default
title: Motivation
subtitle: "What we find: Small multilingual runs often cannot make the decision"
---

<div class="findings-grid">
<div class="finding"><b>34%</b><span>of the 938 analysed tasks: No model size up to 1.7B is above chance</span></div>
<div class="finding"><b>0.51 to 0.56</b><span>of variant pairs ordered like the 1.7B reference on accuracy, close to pairwise chance</span></div>
<div class="finding"><b>0.74 to 0.94</b><span>of variant pairs ordered like the reference on per-language BPB</span></div>
<div class="finding"><b>No statistic</b><span>computed from the proxies predicts this agreement consistently</span></div>
</div>

<div class="decision-def">

The per-language training tokens seen, the expected answer format and the scoring strategy impact decision reliability more than changes to proxy model size across the scales we test.

</div>


---
layout: default
title: Motivation
subtitle: Our contributions
---

<div class="summary">

- **An open controlled multilingual scaling ladder**: 218 models from 90M to 3B, with evaluation measurements.
- **A guide to evaluation configuration for multilingual ablations**: For each benchmark, the per-language training-token thresholds at which training configurations pass the above-chance filter, and the counts at which we observe decision reliability.
- **An analysis of general trends in decision reliability** to inform multilingual ablation design, including recommendations on task filtering, reformatting, and scoring.

</div>

<!--
The guide is on the project site, scaling-reliability.netlify.app.
-->

---
layout: agenda
title: Research Questions
items:
  - "RQ1 Measurability: When is a task above chance?"
  - "RQ2 Agreement: How small and early can a proxy be?"
  - "RQ3 Anticipation: Can proxy statistics predict agreement?"
---

<!--
The three questions, in the order a developer should ask them: is the score above chance, does the proxy rank like the reference, can I tell in advance.
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
subtitle: The model ladder
---

::figure::
<Interactive viz="ladderAxes" panel="sizes" :zoom="0.85" />

<!--
N is the non-embedding parameter count. The axes are depth, optimizer, activation, number of languages K, language-selection scheme M and sampling temperature T. Seed S is 1904 by default. 2 is one Chinchilla multiple.
-->

---
layout: figure
image: /tables/table02_ladder.png
fit: contain
height: 50vh
title: "Experimental Setup"
subtitle: "Pretraining hyperparameters"
---

<!--
Every rung has its own hyperparameters. The token budget is D = 100N.

N counts non-embedding parameters, and the tied embedding adds d_model × 131,072. The budget is D = 100N tokens. Grouped-query attention with four query heads per key-value head. The SwiGLU ladder narrows the FFN so that the three projections match the deep parameter count. 90M and 175M train at a smaller batch with proportionally more steps.
-->



---
layout: figure
title: Experimental Setup
subtitle: The data mixtures
---

::figure::
<Interactive viz="mixtures" panel="bars" :zoom="0.85" />

<!--
Each mixture is half English (DCLM-Edu) and half FineWeb-2, split over the other K−1 languages.
Scheme A takes the languages with the most data. Scheme B covers as many scripts and families as it can.
At temperature 1 each language gets its share of the corpus. Temperature 3 flattens the split towards the smaller languages.
The lists are nested: the K = 8 list is inside the K = 15 list, and so on up to K = 50.
-->




---
layout: default
title: Experimental Setup
subtitle: "Every benchmark considered"
---

<div class="bench-tables">
<div>

**Analysed (40)**

| Task | Benchmarks (trained / available languages) |
|---|---|
| **Knowledge** | ARC (MT) 11/11, ARC (Okapi) 27/32, Global-MMLU 29/37, MMLU 1/1, MMLU (Okapi) 27/27, MMMLU 12/12, OpenBookQA 1/1 |
| **Commonsense** | CommonsenseQA 1/1, Global PIQA 48/77, HellaSwag (Okapi) 26/31, PIQA 1/1, WinoGrande 1/1, XCOPA 8/11, XStoryCloze 8/13, XWinograd 6/6 |
| **Mixed suite** | KoBEST 1/1, NorEval 1/1 |
| **Cultural knowledge** | BLEnD 4/4, CulturalBench-easy 8/8, CulturalBench-hard 8/8, INCLUDE 36/43, INCLUDE v2 RF (EN) 42/42, INCLUDE v2 RF (OG) 42/42 |
| **Truthfulness** | TruthfulQA (MC2) 3/3, TruthfulQA-Multi (MC1) 2/3 |
| **Reasoning** | ACP-Bench (cloze) 1/1, ACP-Bench (MCQ) 1/1, BBH (cloze) 1/1, BBH (MCQ) 1/1, MathQA 1/1 |
| **Grammar** | MultiBLiMP 37/60 |
| **Reading** | Belebele 49/94, BoolQ 1/1 |
| **Tokenization** | TokSuite 5/5, TokSuite math 1/1, TokSuite STEM 1/1 |
| **Inference** | PAWS-X 8/10, XNLI 15/18 |
| **Toxicity** | ToxiGen 1/1 |
| **Language modelling** | LAMBADA 5/5 |

</div>
<div>

**Screened, not analysed (26)**

| Task | Benchmarks (trained / available languages) |
|---|---|
| **Knowledge** | Arabic EXAMS (OALL) 1/1, Arabic MMLU (OALL) 1/1, ArabicMMLU 1/1, C-Eval 1/1, CareQA 1/1, HAE-RAE Bench 1/1, HEAD-QA 1/1, KMMLU 1/1, TurkishMMLU 1/1 |
| **Commonsense** | COPAL-ID 1/1 |
| **Mixed suite** | AlGhafa (OALL) 1/1, AlGhafa translated (OALL) 1/1, Bangla 1/1, EVALITA-LLM 1/1, FrenchBench 1/1, IberoBench 3/5 |
| **Cultural knowledge** | ACVA (OALL) 1/1 |
| **Truthfulness** | TruthfulQA (MC1) 1/1, TruthfulQA (Okapi, MC1) 26/27, TruthfulQA (Okapi, MC2) 24/24, TruthfulQA-Multi (MC2) 3/3 |
| **Grammar** | BLiMP-NL 1/1, MELA 9/10, TurBLiMP 1/1, ZhoBLiMP 1/1 |
| **Reading** | XQuAD 1/1 |

</div>
</div>

<!--
-->

---
layout: figure
title: Experimental Setup
subtitle: "Answer formats and scoring settings"
---

::figure::
<Interactive viz="evalFormats" :zoom="0.85" />

<!--
The same question in the letter format, which expects "C", and in the cloze format, which expects "Madrid".
Length-normalised accuracy picks the option with the highest likelihood per byte. In the letter format every option has one character, so it equals plain accuracy. Chance is 1 over the number of options.
bBPB is the bits per byte of the gold answer and has no chance level.
BPB is the bits per byte of held-out text in the language, with no question.
-->

---
layout: figure
title: Experimental Setup
subtitle: We evaluate 10 ckpts per model
---

::figure::
<Interactive viz="scoreCurves" :zoom="0.85" />

<!--
We evaluate every run at each tenth of training, only on the languages it trains.
Live: Pick HellaSwag (Okapi): Every size scores rise above the dashed chance line. Also xnli, xcopa, xstorycloze. Then INCLUDE: Every size stays at chance. Also belebele, global mmlu.
-->


---
layout: section
---

# RQ1: Measurability

At which model size and per-language token count is a task above chance?


---
layout: default
title: "RQ1: Measurability"
subtitle: The above-chance threshold
---

<div class="definition">

A model is above chance on a task when $\mathrm{LCB}_{95}(\hat p) > 1/m_b$, with $m_b$ the number of options of benchmark $b$.

$\mathrm{LCB}_{95}$ is the one-sided 95% Wilson lower confidence bound on the final-ckpt accuracy over the task's n items ($\hat p$).

A model size passes the task when at least half of its configurations are above chance. BPB and bBPB have no chance level and are not filtered.

</div>

<!--
One-sided 95% Wilson lower confidence bound on the accuracy over the task's n items. A model size passes when at least half of its configurations are above chance. BPB and bBPB have no chance level and are not filtered.

For each language task b_ℓ and model size N, we test whether the observed accuracy can be told apart from random guessing.
Chance is 1/m_b for a task with m_b options. Where the option count varies across items, chance is the mean of 1/m over the items (0.2253 for TruthfulQA MC1).
We take the final-checkpoint accuracy p̂ over the n scored items, recover round(p̂ n) correct answers, and compute the one-sided 95% Wilson lower bound.
The vote is over the models of one size, every ladder and data build at seed 1904, and only over models whose mixture trains the task's language.
A scaling fit uses only the sizes where the task passes. A ranking against the reference also needs the task to pass at the reference size.
BPB and bBPB have no chance level and are not filtered.
-->

---
layout: figure
image: /paper/app_rq00_chance_reformulation.png
fit: contain
height: 46vh
title: "RQ1: Measurability"
subtitle: "Cloze benchmark version scores pass the threshold"
---

<!--
At 1.7B, programmatic reformulation increases the languages above chance from 5 to 86 of 105 for Belebele, from 1 to 35 of 37 for Global-MMLU, and from 3 to 31 of 43 for INCLUDE.

LLM-RF: Belebele passes in every trained language at every size. INCLUDE passes in 32 of 43 languages at 1.7B.
-->



---
layout: figure
title: "RQ1: Measurability"
subtitle: "Letter-format benchmarks pass only at 3B, cloze benchmarks from 90M"
---

::figure::
<Interactive viz="firstSize" panel="languages-x" :zoom="0.68" height="500px" />

<!--
For only 38% of tasks does every model size pass the above-chance filter.
For 34% of tasks, no model up to 1.7B passes the filter.
These tasks are predominantly from benchmarks requiring letter-format answers like Belebele, Global-MMLU and INCLUDE.
On benchmarks scored on the answer text, such as HellaSwag (Okapi) and MultiBLiMP, configurations score above chance in all trained languages even at 90M.
-->

---
layout: figure
image: /paper/rq0.png
fit: contain
height: 50vh
title: "RQ1: Measurability"
subtitle: "Language training tokens and format effect: the case of INCLUDE"
---

<!--
The share of (language, K, checkpoint) evaluations above chance, against the training tokens of the language.
Original (top): Accuracy rarely exceeds chance. Reformulated (bottom): The share of configurations that pass the filter rises as training progresses.
At 100M to 1B target-language tokens, 43% of 90M configurations are above chance, against 62% at 1.7B.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: "Language training tokens and format effect"
---

::figure::
<Interactive viz="paperRq0" :zoom="0.85" />

<!--
Live: Pick HellaSwag (above chance from the fewest tokens), ARC (Okapi) (needs about 10B tokens), then Belebele with 'also LLM-RF'.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: "Language training tokens needed to pass the threshold"
---

::figure::
<Interactive viz="gateThresholdTable" :zoom="0.85" />

<!--
For each non-English benchmark and answer format, the tokens of the language at which 50% and 90% of its (language, K, checkpoint) cells are above chance. Lang. is the number of non-English trained languages. Margin is the median accuracy above chance a task needs for its Wilson bound to clear chance. Items is the median item count.

How to read it: A cell gives the tokens of the language a model of that size needs before 50% (top) and 90% (bottom) of its evaluations pass. '≤ X': It passes from the fewest tokens we saw. '> X', orange: never reached in our runs.
Live: HellaSwag passes from the start at every size. ARC (Okapi) reaches 50% at 3.3B tokens at 600M and at 320M at 1.7B: ten times fewer. Click Format 'Original', then 'RF': Belebele goes from orange everywhere to light blue.

Examples: The letter-format Belebele, Global-MMLU and INCLUDE, BLEnD, CulturalBench-hard and TruthfulQA (MC2) never reach 50%. Their reformatted versions do.
ARC (Okapi) reaches 50% after 3.3B tokens at 600M and after 320M at 1.7B. A 1.7B model needs about ten times fewer tokens of a language than a 90M one.
-->



---
layout: default
title: "RQ1: Measurability"
subtitle: "Benchmark scaling predictability across sizes"
---

Most benchmarks above chance scale predictably with size. Across 431 tasks retained by the above-chance filter, the median R² of the model-size fits is 0.87, reaching 1.00 for the BPB across 50 languages.

<Interactive viz="paperRq1" panel="a" :zoom="0.85" />

<!--
We fit the final-checkpoint score against log N (at least three sizes) and the score along a run against log tokens (at least five checkpoints), and report R² and Spearman's ρ.
(a) Above chance, the median size-fit R² is 0.87 over 389 benchmark tasks and 1.00 for the per-language BPB of 30 languages. HellaSwag reaches 0.88 and LAMBADA 0.96. The letter-format Global-MMLU, INCLUDE and Belebele have no fit because they stay at chance. Their RF versions fit at 0.84 to 0.92.
A high R² does not mean improvement: Spanish TruthfulQA-Multi has R² 0.97 with ρ = −0.75.

Live: Click 'one dot per benchmark (paper)' and 'accuracy and BPB (paper)' for the paper's view. Then highlight TruthfulQA-Multi: R² 0.97 but ρ = −0.75, a strong and consistent decrease with size.
-->

---
layout: default
title: "RQ1: Measurability"
subtitle: "Benchmark scaling predictability across sizes and training checkpoints"
---

No accuracy or BPB task exhibits predictable scaling across training checkpoints without also scaling predictably across model sizes. Comparing larger models earlier tells more than training smaller models longer.

<Interactive viz="paperRq1" panel="b" :zoom="0.85" />

<!--
(b) 311 of the 419 accuracy and BPB tasks (74%) are predictable along both axes. None is predictable along training but not along size. Only 9 of the 688 bBPB tasks are.
So in our setting, comparing larger models earlier tells more than training smaller models longer.

Live: Click 'highlighted examples (paper)'. Point at MultiBLiMP: most languages top right, Danish and Swedish weak on both axes. No dot sits top left: none is predictable along training only.
-->



---
layout: figure
title: "RQ1: Measurability"
subtitle: English-only scaling predictability
---


::figure::
<Interactive viz="englishRegimes" panel="r2" :zoom="0.85" />

<!--
The same pattern holds for monolingual English models.
Live: Type 'piqa' in Find a task: Global PIQA is the only popular benchmark above chance that does not scale predictably. The list on the right gives the 17 tasks predictable in all three corpora.
-->

---
layout: default
title: "RQ1: Measurability"
subtitle: Summary
---

<div class="summary">

- A large share of the suite does not perform measurably on these benchmarks.
- In 34% of the 938 analysed benchmark–language tasks, no model size up to 1.7B passes the above-chance filter.
- These tasks are predominantly those scored using multiple-choice answer letters.
- Whether a training configuration scores above chance on a task depends on both the target-language tokens it has seen and its model size.
- On the tasks the filter retains, accuracy scales predictably with model size and with training tokens.

</div>

<!--
The Summary box of RQ1 in the submitted paper (paper_submission_arr.pdf), split into bullets.
34% never pass up to 1.7B. Reformulating the letter-format benchmarks moves most of their languages above chance. Accuracy then scales predictably along both axes.
-->

---
layout: section
---

# RQ2: Agreement

How small and how early can a proxy be while it keeps the reference decision?

---
layout: default
title: Experimental Setup
subtitle: Ranking agreement
---


| | Proxy | Reference | Question |
|---|---|---|---|
| Scale agreement | $(N,\ f{=}1)$ | $(N_\text{ref},\ f{=}1)$ | How small can the proxy model be? |
| Within-model training agreement | $(N,\ f{<}1)$ | $(N,\ f{=}1)$ | How early does the ranking stabilize? |
| Target agreement | $(N,\ f{<}1)$ | $(N_\text{ref},\ f{=}1)$ | How small and how early? |

We quantify ranking agreement using decision accuracy:

$$\text{DA}(N,f)=\frac{1}{|\mathcal P|}\sum_{(a,b)\in\mathcal P}\mathbb 1\big[\,s_{ab}(N,f)=s_{ab}(N_\text{ref},1)\,\big]$$


<!--
We examine three settings.

DA is the fraction of pairwise model preferences preserved between the proxy and the reference points. We compute it only with at least three variant pairs.
-->

---
layout: figure
image: /tables/table09_ra_defs.png
fit: contain
height: 30vh
title: "Experimental Setup"
subtitle: "Table: The three ranking-agreement settings"
---

<!--
The proxy and the reference rank the same variants but differ in model size, training progress or both. N_ref is the 1.7B reference, or 3B in the size-generalisation.
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
DA, Kendall's τ and Spearman's ρ measure nearly the same ranking agreement (r = 0.99 and 0.97), so we report DA.
-->

---
layout: figure
image: /paper/rq2.png
fit: contain
height: 40vh
title: "RQ2: Agreement"
subtitle: "Figure 3: Decision reliability across scale and training"
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
layout: default
title: "RQ2: Agreement"
subtitle: "Figure 3: Decision reliability across scale and training, with its findings"
---

<div class="fig-below">
<div class="fig-below-img"><img src="/paper/rq2.png"></div>
<ul>
<li>Within one model, agreement rises from approximately 62 to 68% at early checkpoints to 83 to 93% in the later stages of training.</li>
<li>Across scales, agreement remains modest (59 to 61% across all pairs) and shows no consistent improvement with model size.</li>
<li>Sampling temperature consistently exhibits higher agreement (72 to 75%) than other design choices.</li>
<li>Against the target, smaller models remain at approximately 58 to 64%, while agreement at 1.7B increases steadily throughout training.</li>
</ul>
</div>

<!--
Same figure as the previous slide. The bullets are the paper's sentences.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: "Figure 3a: Fully trained proxies agree with the 1.7B ranking at about 0.6, whatever their size"
---

::figure::
<Interactive viz="paperRq2" panel="size" :zoom="0.9" />

<!--
Live: Hover the temperature line (0.72 to 0.75) and the black all-pairs line (no gain from 90M to 1B).
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: "Figure 3a, two-column version: scale agreement"
---

::figure::
<div class="two-col">
<div class="two-col-plot"><Interactive viz="paperRq2" panel="size" :zoom="0.8" height="400px" /></div>
<ul class="two-col-points">
<li>Overall, agreement remains modest (59 to 61% across all pairs) and shows no consistent improvement with increasing model size.</li>
<li>Sampling temperature consistently exhibits higher agreement (72 to 75%) than other design choices.</li>
<li>The reliability of cross-scale decisions depends more on the type of model intervention than on the size of the proxy model alone.</li>
</ul>
</div>

<!--
Same figure, findings beside it.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: "Figure 3b: Within one model, agreement with the final checkpoint rises from about 0.65 to about 0.9"
---

::figure::
<Interactive viz="paperRq2" panel="ckpt" :zoom="0.9" />

<!--
Live: Hover the 350M line, which agrees most, and the 1.7B line, which agrees least near the end.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: "Figure 3b, two-column version: agreement within one model"
---

::figure::
<div class="two-col">
<div class="two-col-plot"><Interactive viz="paperRq2" panel="ckpt" :zoom="0.8" height="400px" /></div>
<ul class="two-col-points">
<li>Ranking agreement generally increases with training progress, rising from approximately 62 to 68% at early checkpoints to 83 to 93% in the later stages of training.</li>
<li>Disagreement persists even near the end of training.</li>
<li>We observe no consistent relationship between model size and ranking agreement.</li>
</ul>
</div>

<!--
Same figure, findings beside it.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: "Figure 3c: Smaller proxies stay near 0.6 against the 1.7B target however long they train"
---

::figure::
<Interactive viz="paperRq2" panel="target" :zoom="0.9" />

<!--
Live: The 1.7B line climbs because it converges to itself. Every smaller proxy stays flat.
-->

---
layout: figure
title: "RQ2: Agreement"
subtitle: "Figure 3c, two-column version: agreement with the target"
---

::figure::
<div class="two-col">
<div class="two-col-plot"><Interactive viz="paperRq2" panel="target" :zoom="0.8" height="400px" /></div>
<ul class="two-col-points">
<li>For smaller models, agreement remains modest (approximately 58 to 64%) and shows little systematic improvement with additional training.</li>
<li>Agreement at 1.7B increases steadily throughout training.</li>
<li>Improvements in ranking agreement during training do not necessarily transfer across model scales.</li>
</ul>
</div>

<!--
Same figure, findings beside it.
-->

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
layout: default
title: "RQ2: Agreement"
subtitle: "Where accuracy is at chance, bBPB still ranks the variants, and BPB ranks them best, with its findings"
---

<div class="fig-below">
<div class="fig-below-img"><img src="/paper/app_rq02_bbpb_acc_gate.png"></div>
<ul>
<li>Where accuracy is at chance at the proxy, bBPB still ranks the pairs like the 1.7B bBPB at 0.60 to 0.63.</li>
<li>That is above accuracy on the same tasks (0.46 to 0.50) but far below BPB.</li>
<li>Against the 1.7B accuracy, the proxy's bBPB agrees more often than its accuracy, by +0.10 at 90M down to +0.03 at 1B.</li>
<li>bBPB is the better proxy on 57 to 71% of these tasks.</li>
</ul>
</div>

<!--
The findings version of the previous slide: the bullets are the paper's sentences, as close to its wording as possible.
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
layout: default
title: "RQ2: Agreement"
subtitle: "Figure 3 against the 3B reference on top and the 1.7B reference below, with its findings"
---

<div class="fig-side">
<ul>
<li>At 3B more tasks pass the above-chance threshold but ranking agreement does not improve.</li>
<li>The number of tasks above chance rises from 612 to 778 of 933 (+27%).</li>
<li>DA-size against 3B is 0.49 to 0.54 for benchmark accuracy and 0.78 to 0.91 for per-language BPB.</li>
<li>The 1.7B models themselves reach only 0.51 on accuracy and 0.95 on BPB.</li>
</ul>
<div class="fig-side-img"><img src="/paper/rq2_3b_compare.png"></div>
</div>

<!--
The findings version of the previous slide: the bullets are the paper's sentences, as close to its wording as possible.
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
subtitle: "Languages trained together predict each other, English does not"
---

::figure::
<Interactive viz="ltCross" panel="map" :zoom="0.85" />

<!--
Live: Hover the German–Spanish–French–Italian block (blue), then the English row (red): English is a poor proxy for the other languages.
-->

---
layout: default
title: "RQ2: Agreement"
subtitle: Summary
---

<div class="summary">

- On accuracy-scored benchmarks, the share of variant pairs that a proxy point orders the same way as the final 1.7B checkpoint is close to pairwise chance: 0.51 to 0.56 at every proxy size.
- On per-language BPB, the share is higher, ranging from 0.74 to 0.94 over the same proxies.
- Seed-to-seed noise in the reference models accounts for part of the disagreement.
- On the design-variant pairs that the 1.7B reference model separates by more than two seed standard deviations, the share ordered correctly is 0.69 to 0.73.

</div>

<!--
The Summary box of RQ2 in the submitted paper (paper_submission_arr.pdf), split into bullets.
Within one model the ranking stabilises along training, but that does not carry across scale. A 3B reference does not change the picture.
-->

---
layout: section
---

# RQ3: Anticipation

Can statistics of the proxy runs alone predict decision accuracy?

---
layout: default
title: "RQ3: Anticipation"
subtitle: The proxy statistics we test as surrogates of DA
---

<div class="defs">

| Surrogate | Definition, over the V variants of one proxy size |
|---|---|
| Early-checkpoint agreement | $\mathrm{DA}(N, 0.1)$: The ranking at 10% of the run against the run's final ranking |
| Signal (relative std) | $S_\mathrm{rel} = \frac{1}{\bar c}\sqrt{\frac{1}{V}\sum_{a}(c_a - \bar c)^2}$, on the final scores $c_a$ |
| Checkpoint noise (relative) | the standard deviation of a variant's detrended score over 80 to 100% of its run, averaged over variants, divided by $\bar c$ |
| SNR | signal divided by noise, for each signal and noise definition |
| SNR, dist_std | the standard deviation of the pairwise gaps $\lvert c_a - c_b \rvert$, divided by the noise |
| SNR, discrepancy | the centred $L_2$ discrepancy of the scores in $[0, 1]$, divided by the noise |
| Scaling-fit $R^2$ | the median $R^2$ of the log-linear fits of the final score against $\log_{10} N$, over the sizes up to the proxy |
| Margin above chance | $M = \bar c - 1/m$ |

</div>

<!--
A surrogate is a statistic of a benchmark task computed at the proxy size alone, without the reference.
The noise window is the five checkpoints at 80, 85, 90, 95 and 100% of the run. Its linear trend is removed first, because scores still rise during the learning-rate decay.
Scores are oriented so that higher is better, so BPB and bBPB enter with a flipped sign.
Heineman et al. (2025) use the relative dispersion over the relative checkpoint noise. We test 22 signals and six noises.
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
layout: default
title: "RQ3: Anticipation"
subtitle: "Figure 4: How well each surrogate predicts DA at each proxy size, with its findings"
---

<div class="fig-side">
<div class="fig-side-img"><img src="/paper/rq3.png"></div>
<ul>
<li>Early-checkpoint ranking agreement (ρ = 0.54 to 0.62) and the relative-standard-deviation SNR (ρ = 0.53 to 0.60) show the strongest overall associations.</li>
<li>They are followed by scaling-fit R² (ρ = 0.45 to 0.49).</li>
<li>Signal alone and margin above chance show little association with DA.</li>
<li>Moderate correlations are already present at small scales and do not consistently improve as models grow.</li>
</ul>
</div>

<!--
The findings version of the previous slide: the bullets are the paper's sentences, as close to its wording as possible.
-->

---
layout: figure
title: "RQ3: Anticipation"
subtitle: "Figure 4: Early-checkpoint agreement and the relative-standard-deviation SNR predict DA best"
---

::figure::
<Interactive viz="paperRq3" :zoom="0.85" />

<!--
Live: Hover the top two rows at 175M (0.62) and 350M (0.60), then the bottom two rows, near zero.
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
subtitle: "The relative-spread SNR definitions track DA best, as a group"
---

::figure::
<Interactive viz="surrogatesVariants" :zoom="0.85" />

<!--
Live: The relative-spread definitions (rel_std, rel_mpd, rel_dispersion) lead as a group. The depth ones (tukey, projection) fall below zero.
-->

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
layout: default
title: "RQ3: Anticipation"
subtitle: "Figure 4 split by scoring into accuracy, bBPB and per-language BPB, with its findings"
---

<div class="fig-below">
<div class="fig-below-img"><img src="/paper/app_rq04_surrogates_by_scoring.png"></div>
<ul>
<li>Split by scoring, bBPB raises both the level of the correlations in Figure 4 and the lead of early-checkpoint agreement.</li>
<li>On bBPB, early-checkpoint agreement reaches ρ = 0.57 to 0.68 and the relative-standard-deviation SNR 0.46 to 0.58.</li>
<li>On accuracy, they reach 0.28 to 0.39 and 0.33 to 0.40.</li>
<li>On per-language BPB, five of the six statistics turn negative at 600M.</li>
</ul>
</div>

<!--
Same figure as the previous slide. The bullets are the paper's sentences.
-->

---
layout: figure
title: "RQ3: Anticipation"
subtitle: "Split by scoring, the bBPB versions carry most of the correlation"
---

::figure::
<Interactive viz="surrogatesScoring" :zoom="0.85" />

<!--
Live: Switch to 'one scoring as a grid' and pick per-language BPB: The 600M column turns negative.
-->

---
layout: default
title: "RQ3: Anticipation"
subtitle: Summary
---

<div class="summary">

- No metrics computed from proxy evaluations predicts DA consistently across benchmarks or proxy sizes.
- The two metrics with the strongest association are early-checkpoint ranking agreement and the relative-standard-deviation SNR.
- Early-checkpoint ranking agreement is the DA between a run's 10% checkpoint and final checkpoint (ρ = 0.54 to 0.62).
- The relative-standard-deviation SNR is the spread of the variants' final scores divided by checkpoint noise (ρ = 0.53 to 0.60).

</div>

<!--
The Summary box of RQ3 in the submitted paper (paper_submission_arr.pdf), split into bullets.
Both are already informative at 90M and do not strengthen with size. Within one benchmark the association can be much stronger, up to ρ = 0.93 for HellaSwag.
-->

---
layout: section
---

# The recommender

From the findings to a choice of benchmarks for one ablation

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
A benchmark gets 'use' when it is above chance at that size and its decision accuracy is at least 0.66, the paper's threshold.
Live: German at 350M, then switch the language to Hungarian or Hindi, then the decision to 'Can I decide before the run ends?'.
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
8 of 40 measurements pass. Per-language BPB is the most reliable (0.82). Global-MMLU in its letter format is above chance in only 8% of the runs. Read it as a prior and confirm it on your own runs.
-->

---
layout: section
---

# Appendix

Additinoal details

---
layout: figure
title: "RQ1: Measurability"
subtitle: Final-checkpoint accuracy against non-embedding parameters
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
subtitle: Accuracy across training checkpoints
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq00_benchmark_curves.png"></div>

<!--
Accuracy at each evaluated tenth of a run, in Chinchilla multiples.
The letter formats stay at chance for the whole run. Their reformulations increase steadily.
-->

---
layout: default
title: "RQ1: Measurability"
subtitle: The above-chance threshold
---

<div class="definition">

A model is above chance on a task when $\mathrm{LCB}_{95}(\hat p) > 1/m_b$, with $m_b$ the number of options of benchmark $b$.

$\mathrm{LCB}_{95}$ is the one-sided 95% Wilson lower confidence bound on the final-ckpt accuracy over the task's n items ($\hat p$).

$$\mathrm{LCB}_{95}(\hat p) = \frac{\hat p + \frac{z^2}{2n} - z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \qquad z = 1.645$$

A model size passes the task when at least half of its configurations are above chance. BPB and bBPB have no chance level and are not filtered.

</div>

---
layout: figure
image: /extra/above_random_example.png
fit: contain
height: 44vh
title: "RQ1: Measurability"
subtitle: "The threshold on one task: INCLUDE v2 in Hungarian, from 350M on"
---

<!--
INCLUDE v2 in Hungarian: 4 options, 1,386 items, so a run needs 27.0% (2 points above chance).
Per size, runs passing over runs that train Hungarian: 90M 1 of 11, 175M 5 of 11, 350M 6 of 11, 600M 5 of 11, 1B 7 of 11, 1.7B 9 of 11.
So the task enters the analysis at 350M, 1B and 1.7B. The orange run (350M) scores 25.8%: above the chance line, but its lower bound (23.9%) is not.
-->

---
layout: figure
title: "RQ1: Measurability"
subtitle: "The above-chance threshold: The one-sided 95% Wilson bound must exceed 1/m"
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
layout: section
---

# Limitations

Besides classic compute/data limitations.

---
layout: figure
image: /extra/rq2_da_all_mono_axis.png
fit: contain
height: 40vh
title: "Limitations"
subtitle: "Without the 0.66 filter, Figure 3 reads about 0.55 to 0.59 at every proxy size"
---

<!--
The 0.66 filter is a cut on DA itself. On every task above chance, DA-size stays at 0.55 to 0.59 at every proxy size.

Figure 3 shows the tasks whose median DA-size or DA-ckpt is at least 0.66. That filter is a cut on DA itself, so part of the rise it shows is the cut.
On every task above chance (single-axis pairs), DA-size stays at about 0.55 to 0.59 from 90M to 1B, DA-ckpt still rises along training, and DA-target stays near 0.55 for every proxy below the reference.
The paper reports the unfiltered population in its appendix, with accuracy DA at 0.51 to 0.56.
-->

---
layout: figure
image: /paper/app_rq02_da_goal_multi_axes_bpb.png
fit: contain
height: 40vh
title: "Limitations"
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
image: /paper/app_rq02_decisive_pairs.png
fit: contain
height: 40vh
title: "Limitations"
subtitle: On the pairs the reference separates beyond seed noise, DA rises
---

<!--
On the pairs the 1.7B reference separates by more than two seed standard deviations, the share ordered correctly is 0.69 to 0.73. Only about a quarter of the pairs are that decisive.

A pair is decisive when its 1.7B gap exceeds k·√2 seed standard deviations, for k = 1 or 2.
On benchmarks, DA goes from about 0.57 to 0.59 on every pair to 0.69 to 0.73 at two seed standard deviations. Only about a quarter of the benchmark pairs are that decisive.
Per-language BPB is near perfect on decisive pairs, and its 600M dip shrinks.
Seed noise in the reference explains part of the disagreement.
-->

---
layout: figure
image: /paper/app_rq10_size_generalisation.png
fit: contain
height: 40vh
title: "Limitations"
subtitle: A 3B reference measures more tasks but decides no differently
---

<!--
At 3B more tasks pass the above-chance threshold (612 to 778 of 933) but ranking agreement does not improve.

Six configurations trained to 3B: deep, scheme A at K = 8, 15, 30 and 50, scheme B at K = 8 and 15.
At 3B, 778 of 933 tasks pass the threshold against 612 at 1.7B, mostly the letter-format Belebele, Global-MMLU and INCLUDE.
Per-language BPB carries over. The 1.7B models predict the 3B decisions at 0.94 to 0.95, and every proxy from 90M on reaches at least 0.78.
Benchmark accuracy reads 0.48 to 0.53 against either reference, so its gap to BPB does not come from using 1.7B as the reference.
-->

---
layout: figure
image: /paper/app_rq04_surrogates_mono_axis.png
fit: contain
height: 45vh
title: "Limitations"
subtitle: On single-axis pairs the correlations are weaker, and the relative-standard-deviation SNR leads at every size
---

<!--
Figure 4 redrawn with DA-size and early-checkpoint agreement on the single-axis pairs.
The relative-standard-deviation SNR reaches ρ = 0.37 to 0.46, against 0.24 to 0.36 for early-checkpoint agreement.

On single-axis pairs the correlations are weaker. The relative-standard-deviation SNR leads at every size (ρ = 0.37 to 0.46).
-->

---
layout: section
---

# Other questions

Analyses we ran that are not in the paper

---
layout: figure
title: "Other questions"
subtitle: Agreement per benchmark and language on the reliable tasks
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq02_da_by_language.png"></div>

<!--
Not in the paper's main argument: agreement per benchmark and language on the reliable tasks.

Averaged over the proxies 90M to 1B, DA-size is above 0.5 (95% interval) on 7 of 9 benchmarks and 32 of 40 languages. DA-target is above 0.5 on 7 of 9 and 31 of 40, DA-ckpt on 11 of 11 and 38 of 40.
The mean cell reads 0.65 for DA-size, 0.71 for DA-ckpt and 0.61 for DA-target.
-->

---
layout: figure
image: /paper/app_rq11_evaluation_recipe.png
fit: contain
height: 43vh
title: "Other questions"
subtitle: Which way of evaluating a benchmark recovers the reference decision?
---

<!--
Not in the paper: which format and scoring of a benchmark recovers the reference decision best.

A benchmark can be read with its original, RF or LLM-RF items, scored by accuracy or by bBPB.
No way of reading a benchmark reaches a mean DA-size of 0.75 against the reference accuracy at any proxy up to 1B.
Against the same truth, the 1.7B accuracy, bBPB is the better pick for 20 of the 36 benchmarks and accuracy for 16.
-->

---
layout: figure
title: "Other questions"
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
title: "Other questions"
subtitle: "Within one benchmark the association is stronger, up to ρ = 0.93 for HellaSwag"
---

::figure::
<Interactive viz="surrogatesBench" :zoom="0.85" />

<!--
Stronger associations emerge for individual benchmarks, up to ρ = 0.93 for HellaSwag, a best-of-many value.

The strongest pooled correlation is ρ = 0.75. Within HellaSwag, the best SNR definition reaches ρ = 0.93.
How well DA can be predicted depends on the benchmark. The appendix gives the correlations per stratum.
-->

---
layout: figure
image: /paper/app_rq09_benchmark_design.png
fit: contain
height: 40vh
title: "Other questions"
subtitle: MultiBLiMP, HellaSwag and XStoryCloze separate the designs most sharply
---

<!--
Not in the paper: which benchmark properties go with a high SNR.

Median SNR at 1.7B per benchmark family with at least five tasks.
Seven of the nine sharpest families have two answer options.
-->

---
layout: figure
title: "Other questions"
subtitle: Grouped by any single characteristic, the levels overlap
---

::figure::
<div class="scroll-fig"><img src="/paper/app_rq09_design_da_by_level.png"></div>

<!--
The 113 reliable tasks have a median DA-size of 0.65. The medians of the levels with at least five tasks lie between 0.55 and 0.72.
Grouped by any single characteristic, the levels overlap.
-->



---
layout: figure
title: "Other questions"
subtitle: "On HellaSwag, which untrained relatives does a language predict?"
---

::figure::
<Interactive viz="ltCross" panel="family" :zoom="0.85" />

<!--
Not in the paper. Live: Pick Spanish as the proxy: Its Romance relatives are blue, unrelated families red.
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
The changing task set: The Limitations report comparisons on a fixed population, and the guide gives the thresholds per size.
LLM-RF: We reviewed the rewrites by hand in six languages and iterated on the prompt. The length check gives 22.2% against 21.9%.
bBPB: We read it against its own 1.7B value and against the 1.7B accuracy. Against the accuracy it still agrees more often, by 0.03 to 0.10.
Pooling: The appendix gives the single-axis results, where the correlations are weaker and the relative-standard-deviation SNR leads, and a table per stratum.
Threshold: Rerunning it with the majority class as chance lowers the share of tasks above chance by only 1.5 to 2.8 points.
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
Scope: The paper states the recommendation within the scales we test.
600M: the seed draw of the deep 600M runs, on pairs whose 1.7B gap is about one seed standard deviation. More training does not close it.
Guide: The thresholds are per benchmark, format and size. A held-out-language test is future work.
DataDecide: SNR predicts DA on the English OLMES benchmarks, and the association weakens as K grows (appendix figure).
A practical recipe: RF or LLM-RF formats, BPB and bBPB alongside accuracy, several small sizes, and early-checkpoint agreement and the relative-standard-deviation SNR as checks.
-->

---
layout: section
---

# An ARR review of our own paper

The main reasons a reviewer could give to reject it, and how we could answer

---
layout: default
title: "ARR review"
subtitle: "Overall assessment: a useful resource with conclusions that need more support"
---

<div class="summary">

**Summary of the review.**

- Strengths: a controlled, released ladder of 218 multilingual models, a careful above-chance filter, and a practical per-benchmark guide.
- Main weakness: The central negative result (proxies decide close to chance) is not separated from the noise of a single-seed reference or from the selection of tasks by DA.
- Scale: Proxies and reference are small for claims about pretraining decisions.
- Several findings (letter formats at chance, scoring the answer text) are known in English.
- Likely scores: soundness 3, excitement 3, overall borderline.

</div>
---
layout: default
title: "ARR review"
subtitle: "The headline result may measure the reference's noise, not the proxy's unreliability"
---

<div class="review">
<div class="review-concern">

**Concern.** Accuracy DA is 0.51 to 0.56 against a 1.7B reference trained with one seed per configuration. Two seeds of the same design already disagree with each other often, so the reference ranking is itself noisy. Without dividing DA by that ceiling, a reader cannot tell a bad proxy from an unstable target. The Summary box admits seed noise accounts for part of the disagreement, but the main figures and the conclusion do not separate the two.

</div>
<div class="review-answer">

**Possible answer.** Report DA next to its seed ceiling in the main text: the decisive-pairs analysis (pairs the reference separates by more than two seed standard deviations reach 0.69 to 0.73) and the seed null of two seeds of one design. Make this the main reading of RQ2 rather than an appendix figure, and phrase the conclusion as agreement on the decisions the reference itself makes reliably.

</div>
</div>

---
layout: figure
image: /paper/app_rq02_decisive_pairs.png
fit: contain
height: 42vh
title: "ARR review"
subtitle: "For our answer: On the pairs the reference separates beyond seed noise, DA rises to 0.69 to 0.73"
---

<!--
Decisive pairs: The 1.7B gap exceeds k·√2 seed standard deviations. Benchmark DA rises, BPB is near perfect.
-->

---
layout: figure
title: "ARR review"
subtitle: "For our answer: the same at one or two seed standard deviations"
---

::figure::
<Interactive viz="daDecisive" :zoom="0.85" />

<!--
Show both thresholds live, and the share of pairs that stay decisive (about a quarter of the benchmark pairs at two SD).
-->

---
layout: default
title: "ARR review"
subtitle: "The 0.66 filter selects tasks on the quantity it then plots"
---

<div class="review">
<div class="review-concern">

**Concern.** Figure 3 keeps the tasks whose median DA-size or DA-ckpt is at least 0.66 and then shows their DA. Part of the level and of the rise is the cut itself. The text quotes 59 to 61% for Figure 3a, the figure shows about 0.60 to 0.64, and the Summary box quotes 0.51 to 0.56 on the unfiltered population. A reviewer will find three numbers for one question and a selected figure in the main text.

</div>
<div class="review-answer">

**Possible answer.** Swap the main figure for the unfiltered one (DA-size 0.55 to 0.59 at every proxy size on single-axis pairs) and show the filtered version as a conditional analysis with its selection rule stated, or choose the tasks out of sample (the cross-fitted selection already exists). Align the text with the figure values.

</div>
</div>

---
layout: figure
image: /paper/rq2.png
fit: contain
height: 36vh
title: "ARR review"
subtitle: "For the reviewer: Figure 3 as published, on the tasks kept by the 0.66 filter"
---

<!--
The published figure: single-axis pairs, accuracy and bBPB tasks with median DA-size or DA-ckpt at least 0.66.
-->

---
layout: figure
image: /extra/rq2_da_all_mono_axis.png
fit: contain
height: 36vh
title: "ARR review"
subtitle: "For our answer: the same figure on every task above chance"
---

<!--
Without the filter DA-size reads about 0.55 to 0.59 at every proxy size. The conclusion (no gain with proxy size) holds on both.
-->

---
layout: default
title: "ARR review"
subtitle: "The scales are too small for the claim about pretraining decisions"
---

<div class="review">
<div class="review-concern">

**Concern.** Proxies are 90M to 1B and the reference is 1.7B, with 3B for six configurations only. Real development decisions are taken for 7B to 70B models. The gap between the largest proxy and the reference is small, so low agreement at 1B against 1.7B mostly says that rankings at these sizes are unstable, not how far a small proxy can be trusted for a large model.

</div>
<div class="review-answer">

**Possible answer.** The question is whether proxy evaluation works at all in the regime where ablations are actually run, and the answer is already negative at a 1.7× gap. The 3B check shows the conclusions do not change with a larger reference. The released ladder lets others extend it. State the scale limitation in the abstract, not only in the Limitations section.

</div>
</div>

---
layout: figure
image: /paper/rq2_3b_compare.png
fit: contain
height: 42vh
title: "ARR review"
subtitle: "For our answer: A 3B reference ranks the proxies no better than the 1.7B one"
---

<!--
Figure 21: The six configurations trained to 3B, against the 3B (top) and 1.7B (bottom) final checkpoints.
-->

---
layout: figure
image: /paper/app_rq10_size_generalisation.png
fit: contain
height: 42vh
title: "ARR review"
subtitle: "For our answer: The 3B rung measures more tasks but decides no differently"
---

<!--
612 to 778 of 933 tasks above chance at 3B, with the same agreement.
-->

---
layout: default
title: "ARR review"
subtitle: "High agreement on per-language BPB may be arithmetic, not reliability"
---

<div class="review">
<div class="review-concern">

**Concern.** Several design axes (number of languages, temperature, language list) change how many tokens of each language a model sees. Per-language BPB then ranks the variants by their token share at every size, so proxy and reference agree almost by construction. The contrast between BPB (0.74 to 0.94) and accuracy may be driven by those axes.

</div>
<div class="review-answer">

**Possible answer.** Split BPB agreement by design axis and show it on the axes that do not change the language shares (depth, activation, optimizer). The 600M dip already shows BPB is not trivially reliable. Report the per-axis numbers in the main text so the BPB recommendation rests on decisions where the shares are fixed.

</div>
</div>

---
layout: figure
image: /paper/app_rq02_da_goal_multi_axes_bpb.png
fit: contain
height: 42vh
title: "ARR review"
subtitle: "For our answer: BPB agreement is not automatic, the 600M rung stays at 0.69 to 0.74"
---

<!--
Figure 18: If BPB agreement were only token-share arithmetic, every proxy size would agree. The 600M proxies stay at 0.69 to 0.74 over their whole run.
-->

---
layout: figure
title: "ARR review"
subtitle: "For the reviewer: agreement per design axis, where temperature leads"
---

::figure::
<Interactive viz="paperRq2" panel="size" :zoom="0.85" />

<!--
Figure 3a per axis: Temperature, an axis that changes the language shares, agrees best (0.72 to 0.75). A per-axis split of BPB agreement would answer the concern directly.
-->

---
layout: default
title: "ARR review"
subtitle: "Agreement counts every reversal, even when the decision does not matter"
---

<div class="review">
<div class="review-concern">

**Concern.** DA treats a reversal between two variants that differ by 0.1 points like one between variants that differ by 5 points. Many design pairs are near ties at the reference, so low DA may come from decisions that cost nothing when taken wrong. The paper never measures the cost of a wrong decision.

</div>
<div class="review-answer">

**Possible answer.** The near-ties tolerance analysis and the tie and difference categories already show that larger and later proxies miss fewer effects that persist. Add a regret measure: the reference score lost when choosing the proxy's winner. That turns DA into the quantity a developer cares about.

</div>
</div>

---
layout: figure
image: /paper/da_sensitivity.png
fit: contain
height: 42vh
title: "ARR review"
subtitle: "For our answer: Treating near ties as ties raises DA but not the trends"
---

<!--
Figure 13 and Table 9: 755 reversed pairs become agreements at 0.5 points, 1,099 at 1 point, 1,308 at 2 points.
-->

---
layout: figure
image: /paper/da_categories.png
fit: contain
height: 42vh
title: "ARR review"
subtitle: "For our answer: Larger and later proxies miss fewer effects that persist"
---

<!--
Figure 14: Tie and difference categories at ε = 0.5 points.
-->

---
layout: default
title: "ARR review"
subtitle: "The decisions tested are few and not all comparable"
---

<div class="review">
<div class="review-concern">

**Concern.** Each design axis has only a handful of pairs at the reference (temperature has four), and multi-axis pairs mix decisions no developer would compare. Models with different language counts are evaluated on different sets of languages, so a pair is compared only on the tasks both train, which changes the population across pairs. Confidence intervals are not shown for the per-axis curves.

</div>
<div class="review-answer">

**Possible answer.** Report the number of pairs and tasks per point and add bootstrap intervals over tasks and pairs. Present single-axis results as the main reading, since they match real ablations, and keep multi-axis as a leaderboard-style complement.

</div>
</div>

---
layout: figure
image: /tables/table03_training-grid.png
fit: contain
height: 40vh
title: "ARR review"
subtitle: "For the reviewer: the training grid, one row per data build and ladder"
---

<!--
Table 3: The runs behind each design axis. Temperature has four single-axis pairs at the reference.
-->

---
layout: figure
title: "ARR review"
subtitle: "For our answer: the configurations available at each number of languages"
---

::figure::
<Interactive viz="ladderAxes" panel="sizes" :zoom="0.85" />

<!--
Ladder version C: every configuration and its number of runs per K.
-->

---
layout: default
title: "ARR review"
subtitle: "Measurability and reformulation findings are largely known"
---

<div class="review">
<div class="review-concern">

**Concern.** That small models fail letter-format multiple choice and do better when the answer text is scored is reported by OLMES, by Gu et al. (2025) and by the FineTasks work. The multilingual extension and the token thresholds are useful, but RQ1 alone is not a new finding, and the LLM reformulation changes the benchmark that practitioners use.

</div>
<div class="review-answer">

**Possible answer.** The contribution is the measurement across 50 languages and the per-language token thresholds, which no prior work gives, together with the link to decision reliability. State the overlap with prior work explicitly and present RQ1 as the filter the later questions need, not as a finding on its own.

</div>
</div>

---
layout: figure
image: /paper/app_rq00_chance_share_considered_counts_3b.png
fit: contain
height: 50vh
title: "ARR review"
subtitle: "For the reviewer: Letter formats pass only at 3B, cloze formats from 90M"
---

<!--
Figure 5: The pattern known from English (OLMES, Gu et al. 2025), now across 50 languages.
-->

---
layout: figure
image: /tables/table08_app_rq0_thresholds.png
fit: contain
height: 50vh
title: "ARR review"
subtitle: "For our answer: per-language token thresholds no prior work gives"
---

<!--
Table 7: The tokens of the language each size needs before a benchmark is above chance.
-->

---
layout: default
title: "ARR review"
subtitle: "The surrogate search risks reporting the best of many correlations"
---

<div class="review">
<div class="review-concern">

**Concern.** Twenty-two signal definitions times six noise definitions, pooled across heterogeneous tasks, are compared with DA, and the best value (ρ = 0.93 for HellaSwag) is quoted. Without held-out validation this is a best-of-many estimate. Pooled correlations across benchmarks can also reflect differences between benchmarks rather than a usable per-task signal.

</div>
<div class="review-answer">

**Possible answer.** Fix the surrogates before looking (early-checkpoint agreement and the relative-standard-deviation SNR, as in the prior work), validate on held-out seeds or languages (the seed holdout exists) and apply the multiple-comparison correction already computed. Quote the within-benchmark correlations as exploratory.

</div>
</div>

---
layout: figure
image: /paper/app_rq04_surrogates.png
fit: contain
height: 50vh
title: "ARR review"
subtitle: "For the reviewer: 22 SNR definitions compared with DA"
---

<!--
Figure 19: Many definitions, one best value per language.
-->

---
layout: figure
image: /paper/app_rq04_surrogates_mono_axis.png
fit: contain
height: 42vh
title: "ARR review"
subtitle: "For our answer: The leaders are fixed in advance and hold on single-axis pairs"
---

<!--
The relative-standard-deviation SNR, the definition of Heineman et al. (2025), leads at every size on single-axis pairs (ρ = 0.37 to 0.46).
-->

---
layout: default
title: "ARR review"
subtitle: "Clarity: inconsistent numbers and terms, and a 40-page appendix"
---

<div class="review">
<div class="review-concern">

**Concern.** The paper uses DA-target and DA-goal, filter and gate, 20 families, 40, 52 and 95 benchmarks, and 938, 933 and 1,488 tasks without saying how they relate. Some figures are hard to read (Figure 2a labels overlap, Figures 8 to 10 are tiny), and the main claims depend on appendix analyses.

</div>
<div class="review-answer">

**Possible answer.** Unify the terms, add one table relating the counts, redraw Figure 2 with fewer labels, and move the decisive pairs and the unfiltered agreement into the main text in place of the filtered figure. The project site already reconciles the counts and offers every figure interactively.

</div>
</div>

---
layout: figure
image: /paper/rq1.png
fit: contain
height: 40vh
title: "ARR review"
subtitle: "For the reviewer: Figure 2 as published, with overlapping labels"
---

<!--
The published Figure 2 is hard to read.
-->

---
layout: figure
title: "ARR review"
subtitle: "For our answer: the same figure, readable and interactive on the project site"
---

::figure::
<Interactive viz="paperRq1" panel="a" :zoom="0.85" />

<!--
The site version of Figure 2a, one dot per benchmark.
-->

---
layout: end
thankYou: Thank you!
subtitle: "scaling-reliability.netlify.app"
---
