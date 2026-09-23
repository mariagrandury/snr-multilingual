---
theme: scholarly
layout: section
transition: 
footerLeft: EPFL NLP
footerMiddle: Signal-Aware Multilingual Evaluation
description: Signal-Aware Framework for Multilingual LM Evaluation
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

# Signal-Aware Framework for Multilingual LM Evaluation

*María Grandury, Angelika Romanou, Éléonore Hasler,*

*Clara Meister, Antoine Bosselut*

EPFL NLP 

---
layout: image-right
image: /motivation.gif
ratio: "1:3"
fit: contain
---

---
layout: agenda
title: Agenda
items:
  - Introduction
  - Related Work
  - Methodology
  - Research questions
  - Experimental setup
  - Findings
  - Open discussion
---

<!--
This deck follows the structure of research_proposal_high_level.pdf:
Introduction → Related Work → Methodology → Experimental Setup → Results → Analysis.
Results/Analysis numbers + figures are from the `custom_swissai_hf` pool
(3 seeds + external pretraining models, instruct excluded).
-->

---
layout: section
---

# Introduction

Training multilingual LMs requires constant evaluation decisions,

but evaluation is expensive and often uninformative.

---
layout: bullets
title: Motivation
subtitle: Lack of benchmark reliability
icon: "⚠️"
---

Training LMs has a **high cost** and requires **constant decisions** (data mixtures, hyperparameters, etc.) guided by benchmark evaluations. However,

## Not all benchmarks provide informative signals

- **High variance**: scores fluctuate across runs
- **Redundancy**: multiple benchmarks measure the same thing
- **Weak correlation**: improvements don't reflect real progress
- **Cost**: large suites are expensive to run frequently

---
layout: bullets
title: Motivation
subtitle: Why multilingual?
icon: "🌍"
---

## This assumption is **especially fragile** in multilingual settings

- Low-resource languages are underrepresented in training data
- Small models may perform **near-randomly** on harder tasks
- Linguistic diversity introduces new sources of variability
- Existing reliability tools were validated **only on English**

<!--
Most benchmark reliability research has been done on English benchmarks and English-first models. Yet multilingual models face fundamentally different challenges. The tools developed for English may not transfer. Understanding whether the SNR framework extends to multilingual settings is both urgent and crucial, given the growing importance of multilingual NLP.
-->

---
layout: focus
color: green
icon: 🎯
---

## Which (subsets of) benchmarks provide reliable signal at each stage of multilingual
## model training — and which model sizes can stand in for the large one?

---
layout: section
---

# Related Work

Insights from the Signal-and-Noise Paper, Heineman et al. (2025), AllenAI

---
layout: image-left
image: /snr_paper_figure_1.png
ratio: "25:1"
fit: contain
title: Signal-and-Noise Paper
subtitle: Figure 1. Examples of signal and noise
---

---
layout: image-left
image: /snr_paper_figure_2.png
ratio: "2:3"
fit: contain
title: Signal-and-Noise Paper
subtitle: Figure 2. Correlation between SNR and Decision Accuracy
---

- **SNR predicts decision accuracy** ($R = 0.791$), while signal or noise alone do not
- **Noise predicts scaling-law error** ($R = 0.653$): noisier benchmarks have less reliable extrapolation
- **Filtering subtasks by SNR** yields subsets that outperform full benchmarks (e.g., 16/57 MMLU subtasks → +2.6% DA)
- **Averaging checkpoint scores** reduces noise and improves decision accuracy (+2.4% on 30-task avg.)
- **Bits-per-byte (BPB)** improves DA over accuracy on 90% of benchmarks, especially math and code


---
layout: section
---

# Methodology

The Signal-Aware Framework, multilingual edition

---
title: Methodology
subtitle: Decision Accuracy
---

<Block type="success" title="Decision Accuracy">

For all pairs of small models $(s_a, s_b)$ trained on datasets ($a$, $b$) and their large versions $(m_a, m_b)$, does the ranking for task $B$ hold?

$$\text{DA} = \frac{1}{|\mathcal{P}|} \sum_{(a,b) \in \mathcal{P}} \mathbb{1}\big[\text{sign}(B(s_a) - B(s_b)) = \text{sign}(B(m_a) - B(m_b))\big]$$

</Block>


Pretraining:
- **DA-size**: small-size → large-size rank agreement (cross-scale)
- **DA-ckpt**: within-size early → late checkpoint rank agreement

Posttraining (additional):
- **DA-stage**: cross-posttraining stage (SFT, DPO, RLVR)
- **DA-ctx**: smaller-larger context rank agreement

---
layout: focus
color: blue
icon: 🎯
---

## Decision Accuracy is what we ultimately want a benchmark to get right

We will look for other metrics that are **proxies** we can compute cheaply during training.

---
title: Methodology
subtitle: Signal & Noise
---

<Block type="success" title="Signal (Relative Dispersion / Normalized Max. Difference)">

How well a benchmark $b$ separates a pair of model scores $m_j, m_k$ of similar scale $s$ trained on different settings:

$$\text{Rel. Dispersion}(b, s) = \frac{\max_{j,k} |m_j - m_k|}{\bar{m}}$$

</Block>

<Block type="success" title="Noise (Relative Std. Dev.)">

Benchmark $b$ variability across the final $n$ training checkpoints of a model $m$:

$$\text{Rel. Std.}(b, m) = \frac{\sqrt{\frac{1}{n-1} \sum_{i=1}^{n}(m_i - \bar{m})^2}}{\bar{m}}$$

</Block>

$$\text{SNR} = \frac{\text{Avg. Rel. Dispersion}(\text{final checkpoint})}{\text{Rel. Std.}(\text{final $n$ checkpoints})}$$

<!--
Signal candidates: the 22 variants on the previous slide are all alternative numerators.
Noise candidates: checkpoint noise (final n ckpts) vs benchmark noise (k-fold, single run).
SNR = signal / noise. Higher = more reliable benchmark.
-->


---
title: Methodology
subtitle: Signal Definitions
---

Signal = how much a benchmark separates models. There are **many ways to quantify "spread"**, we consider **22 candidate variants** grouped into 5 mathematical families:

| Family | Members | Idea |
| ------ | ------- | ---- |
| **Dispersion** | `mpd`, `aad`, `rms_deviation`, `quartile_deviation`, `dist_std`, `dispersion`, `range` | Absolute spread of scores |
| **Relative-spread** | `rel_std`, `rel_mpd`, `rel_mpsd`, `iqr`, `rel_dispersion` | Spread normalized by mean (AllenAI default `rel_std`) |
| **Discrepancy** | `discrepancy`, `star_discrepancy`, `star_discrepancy_shifted`, `rel_star_discrepancy`, `dispersion_shifted`, `gini` | Uniformity / inequality of the score distribution |
| **Robust** | `mad`, `mpsd` | Outlier-resistant spread |
| **Depth** | `tukey`, `projection` | Half-space statistical depth |

**Q1**: Which variant best tracks decision accuracy across languages?

---
title: Methodology
subtitle: "Signal Definitions — Formulas"
---

Each variant is an alternative **signal** numerator (SNR = signal / noise). Scores $c_i$ = per-mix final accuracies at one size; mean $\bar c\,(\mu)$, std $\sigma$, quartiles $Q_1, Q_3$, empirical CDF $F_n$.

<div class="grid grid-cols-2 gap-x-8">

<div>

**Dispersion** · absolute spread
| | |
|---|---|
| `dispersion` | $\max_{i,j}\lvert c_i-c_j\rvert$ |
| `range` | $\max c-\min c$ |
| `mpd` (mean pairwise dist.) | $\frac{1}{n^2}\sum_{i,j}\lvert c_i-c_j\rvert$ |
| `aad` (avg abs deviation) | $\frac{1}{n}\sum_i\lvert c_i-\bar c\rvert$ |
| `rms_deviation` | $\sqrt{\tfrac{1}{n}\sum_i(c_i-\bar c)^2}$ |
| `quartile_deviation` | $(Q_3-Q_1)/2$ |
| `dist_std` | $\operatorname{std}\{\lvert c_i-c_j\rvert\}$ |

**Relative-spread** · ÷ mean
| | |
|---|---|
| `rel_std` | $\sigma/\mu$ |
| `rel_dispersion` | $\max_{i,j}\lvert c_i-c_j\rvert/\bar c$ |
| `rel_mpd` | $\frac{1}{n^2}\sum_{i,j}\lvert c_i-c_j\rvert/\bar c$ |
| `rel_mpsd` | $\frac{1}{n^2}\sum_{i,j}(c_i-c_j)^2/\bar c^2$ |
| `iqr` (interquartile range) | $(Q_3-Q_1)/\bar c$ |

</div>

<div>

**Discrepancy** · uniformity / inequality
| | |
|---|---|
| `discrepancy` | $\max_c\lvert F_n(c)-F(c)\rvert$ |
| `star_discrepancy` | $\sup_{[0,c]}\lvert F_n-F\rvert$ |
| `star_discrepancy_shifted` | $\sup_{[0,c]}\lvert F_n-F\rvert$, scaled |
| `rel_star_discrepancy` | $\sup_{[0,c]}\lvert F_n-F\rvert/F$ |
| `dispersion_shifted` | $\max_{i,j}\lvert c_i-c_j\rvert$, scaled |
| `gini` | $\frac{1}{2n^2\mu}\sum_{i,j}\lvert c_i-c_j\rvert$ |

**Robust** · outlier-resistant
| | |
|---|---|
| `mad` (median abs deviation) | $\operatorname{med}\lvert c_i-\operatorname{med}(c)\rvert$ |
| `mpsd` (mean pairwise squared dev) | $\frac{1}{n^2}\sum_{i,j}(c_i-c_j)^2$ |

**Depth** · half-space
| | |
|---|---|
| `tukey` | $\min\!\big(F_n(x),\,1-F_n(x)\big)$ |
| `projection` | $\big(1+\lvert x-\operatorname{med}(c)\rvert/\operatorname{MAD}\big)^{-1}$ |

</div>

</div>

<style>
table { font-size: 0.74em; margin: 0.15em 0; }
thead { display: none; }
.grid td { padding: 0.12em 0.5em; border: none; }
</style>

---
layout: compare
title: "Preliminary: Benchmark Noise"
leftLabel: Original
rightLabel: Éléonore & Clara
leftColor: blue
rightColor: green
---

### Checkpoint noise

Std. dev. over **late training checkpoints**, aggregated across models, normalized by benchmark-level mean 

$$\text{Noise} = \frac{\frac{1}{|M|}\sum_{m} \sigma_{\text{step}}(m)}{\mu(M)}$$

⚠️ Requires intermediate checkpoints (rarely available)

::right::

### Benchmark noise

Relative std. dev. across **$k$-fold splits** of the evaluation set, averaged across models

$$\text{Noise} = \frac{1}{|M|} \sum_{m} \frac{\sqrt{\frac{1}{k}\sum_{i=1}^{k}(m_i - \bar{m})^2}}{\bar{m}} $$

✅ Computable from a **single evaluation run**

✅ More predictive of decision accuracy in preliminary results from Eléonore and Clara.


---
title: Methodology
subtitle: Stage-specific reliability
---

The right benchmark depends on **where in training** you are.

| Stage | Decision being made | Models available |
| ----- | ------------------- | ---------------- |
| **Pretraining** | Language count, architecture, scale | The 90M–1.7B predictivity ladder (controlled) |
| **Midtraining** | Domain/quality up-sampling | Open-source bases (3B–70B) |
| **Post-training** | SFT / DPO / RLVR recipe | Instruct families (3B–70B) |

- We compute signal, noise, SNR, DA and scaling-law error **per stage**
- Goal: **stage-specific evaluation recommendations**, not a single global ranking
- Current results are the **pretraining** stage: the predictivity ladder


---
layout: section
---

# Research questions


---
title: Research questions
---

| | question | reads |
|---|---|---|
| **RQ0** | Score vs compute; which benchmarks clear chance? | the gate every other RQ depends on |
| **RQ1** | Does a benchmark rank the design variants at a small size / early checkpoint like the reference? | decision accuracy |
| **RQ2** | Which of **22 SNR definitions** predicts decision accuracy, per language — and does it survive a seed swap? | SNR × DA |
| **RQ3** | Does our SNR agree with AllenAI DataDecide on the shared English tasks? | cross-corpus |
| **RQ4** | Can a language or subject **subset** beat the full benchmark's SNR? | subtask sweeps |
| **RQ5** | Which **design features** (curation, format, option count, length) predict SNR? | benchmark metadata |
| **RQ6** | Which **proxy size** ranks an intervention like the reference, and how does that move with L? | *new — the plan's question* |


<!--
One entry point: `cd src/signal-and-noise && bash run_all_predictivity.sh`

Source of truth is one file: the wide per-checkpoint ladder_report.csv published to
msnr-data/ladder-report. Every number in this deck regenerates from it.
-->

---
title: Research questions
---

RQs about statistics and general analysis as the first section:
1. Which benchmarks scale predictably? (Simple fit of scaling laws formula and measure R^2 and Spearman rho)
2. Which benchmarks are stable? (Noise is the metric for this)
3. Which languages benefit most from increasing model scale? We have 3 potential patterns: (1) high baseline + strong scaling, (2) low baseline + strong scaling, (3) low baseline + weak scaling.
4. Does the benchmark distinguish the things we want to compare? (Signal is the metric here)

RQs about the decision accuracy and the surrogates of it:
1. How early can we predict final multilingual performance?
2. What is the smallest evaluation suite that reliably predicts full multilingual model quality?
3. Which is the most informative surrogate? (this is the plot that we said we want ot have with the min model size and the number of languages).


---
layout: section
---

# Experimental Setup

The predictivity ladder

---
title: Experimental Setup — The ladder
subtitle: "Two axes, two interventions, one outcome metric"
---

| Axis | Values |
| ---- | ------ |
| **Size** (non-embedding) | 90M · 175M · 350M · 600M · 1B · 1.7B — each at **5 × Chinchilla** (D = 100·N) |
| **Languages** L | 1 · 2 · 8 · 15 · 30 · 50 · 100 — English share fixed at 50 %, only the *count* varies |
| **Intervention 1** | model **depth** — deep vs shallow at equal non-embedding size |
| **Intervention 2** | data **scheme** — A resource-ranked vs B diversity-first (differ at L ∈ {8, 15, 30}) |
| **Seeds** | 1904 everywhere; ×3 (64 / 313 / 1904) on the replicate cells |

**Outcome metric**: per-language bits-per-byte on a fixed 100-language validation set.
Benchmarks are the secondary signal.

<!--
62 cells per intervention level; 162 counting both architectures and scheme B where its
language set differs. Grid + counts: plan/small-to-large-predictivity-training-plan.md,
generated from launch_trainings.py so the table cannot drift.
-->

---
layout: figure
image: /ladder/pretrain_progress_plan.png
fit: contain
height: 56vh
title: Experimental Setup
subtitle: The planned grid
---

---
layout: figure
image: /ladder/pretrain_progress_simple.png
fit: contain
height: 56vh
title: Experimental Setup
subtitle: The planned grid
---

---
layout: figure
image: /ladder/pretrain_progress_detailed.png
fit: contain
height: 56vh
title: Experimental Setup
subtitle: The planned grid
---

---
title: Experimental Setup — What exists today
subtitle: "66 of 106 cells finished, and the top of the ladder has started to arrive"
---

| | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| **complete cells** | 10 | 17 | 15 | 20 | 2 | **2** |

- **L ∈ {1, 2, 8, 15, 30, 50}**, and **L = 100 has no finished cell at any size**
- 1B is finished at **L8 and L50**, 1.7B at **L1 and L30**. All four are deep, scheme A, seed 1904
- After the loader drops diverged and unfinished runs, **57 models** enter the analysis and 54 of them are also on trend
- Seed replicates: 175M and 600M at L ∈ {1, 2, 50}, six ×3 cells, the whole noise estimate

<!--
Counts from ladder_report.csv (msnr-data/ladder-report, branch data/ladder-report).
106 cells in the report, 66 complete, 57 complete-and-not-diverged, 54 also on-trend.
Every cell above 600M is deep and scheme A, so none of them can host an intervention
comparison yet. That is why every RQ6 decision still resolves against 600M.
-->

---
layout: bullets
title: Which languages we report on
subtitle: "All 50 the ladder trains on, not the 12 we started with"
icon: "🌐"
---

- The deck used to show 12 hand-picked languages. Every per-language figure now covers **all 50 languages of the L50 mixture**
- Validation is wider than training: **100 subsets across 95 languages**. All 50 trained languages are in it, and the other **45 are zero-shot**
- We stop at 50 because the 100-language distribution is not decided. `configs/languages.json` has the list as `groups.trained`
- Widening the view changed one reading: with only 12 languages the benchmark coverage looked thin. Across 50 it is worse

<!--
todo: lists a b, l1-l50
groups.main is kept for the historical decks. groups.trained is the 49 FineWeb-2
tags of FW_L50 in src/pretrain/data/language_sets_schemeA.json, plus English,
which comes from DCLM rather than FineWeb-2. Add the L100 list the same way once
the distribution is decided.
-->

---
layout: section
---
layout: section
---

# How good are our instruments?

Four questions about the benchmarks themselves, before any result

---
layout: figure
image: /ladder/rq_a1_scaling.png
fit: contain
height: 70vh
title: "1. Which benchmarks scale predictably?"
subtitle: "Completion tasks do. Knowledge tasks do not move with size at all."
---

<!--
Each task's score is fitted against log parameters from 175M to 1B. R² says how much
of the movement a straight line explains, Spearman ρ says whether it moves the right
way. hellaswag 0.98, xwinograd 0.98, xstorycloze 0.97, bits per byte 0.96. At the
bottom global_mmlu_full and global_piqa sit near 0.07, which means their scores are
essentially unrelated to model size. truthfulqa is the odd one: a tight fit with
ρ = −0.9, so it reliably gets worse as models grow.
Read bits per byte and loss with the sign in mind. They fall as models improve, so
ρ = −1 is the ideal for them, not a failure.
-->
---
layout: figure
image: /ladder/rq_a2_stability.png
fit: contain
height: 70vh
title: "2. Which benchmarks are stable?"
subtitle: "Noise over a run's late checkpoints, relative to the score itself"
---

<!--
This is the denominator of every SNR we quote. Small means you get the same number if
you evaluate a slightly different checkpoint of the same run. multiblimp and bits per
byte are the steadiest. The four-option knowledge benchmarks are the noisiest, which
compounds the previous slide: they neither move with size nor sit still.
-->
---
layout: figure
image: /ladder/rq_a3_languages.png
fit: contain
height: 68vh
title: "3. Which languages benefit most from a bigger model?"
subtitle: "All 100 gain. The ones that start hardest gain the most, ρ = 0.63."
---

<!--
Baseline is bits per byte at 175M, gain is what a 1B model saves, both on the
50-language mixture. Every one of the 100 validation languages improves, median 0.39
bits per byte. The three patterns the team asked about are all present but the split
is by the median, not by an absolute floor: nothing gets worse. The correlation is
the headline. Scale helps hardest where the model is currently worst, which is the
opposite of what a rich-get-richer story would predict.
-->
---
layout: figure
image: /ladder/rq_a4_signal.png
fit: contain
height: 68vh
title: "4. Does the benchmark separate the things we compare?"
subtitle: "Only bits per byte clears its own noise. Every benchmark family but two does not."
---

<!--
Signal is the spread across the models we want to tell apart, noise is the spread
within one run. The ratio is the number on the right. Bits per byte 7.1x, hellaswag
5.1x, multiblimp 2.3x, arc 1.6x, lambada 1.4x, training loss 1.3x, and then
everything else is under 1x, meaning the benchmark varies more between checkpoints
of one model than between the models themselves. This is the clearest gain from the
new evals: hellaswag on 15 tasks and multiblimp on 57 have crossed the line, so bits
per byte is no longer the only well-powered row above it.
-->
---
layout: section
---

# Can a cheap measurement stand in for the real one?

Three questions about decision accuracy and its surrogates

---
layout: figure
image: /ladder/rq_b1_early.png
fit: contain
height: 68vh
title: "5. How early can we predict final multilingual performance?"
subtitle: "Bits per byte calls it at 20 % of training. Benchmarks never settle."
---

<!--
Decision accuracy of an early checkpoint against the same run's final ranking, at 1B.
Bits per byte is at 0.91 by 20 percent of the run and stays there, so four fifths of
each run adds nothing to the ranking. Hellaswag at 0.90 and arc at 0.88 are almost as
early, multiblimp starts at 0.81 and reaches 0.91, and the at-chance families crawl up
from 0.41. Averaged over everything the suite goes 0.70, 0.71, 0.74, 0.84 across the
four fractions. The practical read: we could evaluate at 20 percent and free most of
the eval budget.
-->
---
layout: figure
image: /ladder/rq_b2_suite_size.png
fit: contain
height: 68vh
title: "6. What is the smallest suite that predicts full model quality?"
subtitle: "Four tasks. Adding the other 2,893 makes the answer worse, not better."
---

<!--
Tasks are added best first, by how well each alone tracks the ranking that
per-language bits per byte gives. Four tasks reach rho = 0.98, and one task already
gets within 5 percent of that. The whole suite of 2,897 reaches 0.53. 2,893 of the
2,897 suite sizes beat the whole suite, because the tail is at-chance tasks whose
scores are noise and dilute the average.
This is the strongest practical result in the deck: a small curated suite is not a
compromise, it is better than running everything.
-->
---
layout: figure
image: /ladder/rq_b3_surrogate.png
fit: contain
height: 58vh
title: "7. Which is the most informative surrogate?"
subtitle: "Per-language bits per byte works on 73 % of 500 comparisons. Benchmarks, on 47 % of 8,449."
---

<!--
This is the min-model-size question in aggregate. A surrogate counts as working when
some proxy size picks the same winner as the reference and every larger size keeps
picking it. Per-language bits per byte 73 percent of 500, and 175M alone is enough
for 32 percent of those. Benchmarks 47 percent of 8,449, up from 0 percent on the
thinner ladder, with 175M alone enough for 23 percent. Read the benchmark row against
the coin-flip result later in the deck: agreeing at two rungs by chance is close to
what 47 percent looks like.
Read the top two rows with care: training loss and macro bits per byte are single
measurements, so they rest on 5 comparisons each. The two well-powered rows are
per-language bits per byte and the benchmarks, and they are the two that matter.
The next slides break this out by language count, which is the shape the team drew.
-->

---

# Findings

What the ladder says so far

---
layout: figure
image: /aromanou.png
fit: contain
height: 66vh
title: Benchmark scaling predictability
---

---
layout: figure
image: /ladder/ladder_report_scaling.png
fit: contain
height: 66vh
title: Finding 1 — Above 90M, the ladder is a ladder
subtitle: "Power-law fit per language setting; red ring = off trend"
---

---
layout: bullets
title: Finding 1 — Above 90M, the ladder is a ladder
subtitle: "Scaling holds, and the exponent barely moves with L"
icon: "📐"
---

- Final loss falls monotonically with size at **every** language setting
- Fitted exponent **α = 0.16 – 0.20** across L (deep, scheme A) — the language count does *not* change how the model scales
- Residuals of the healthy rungs stay within **±0.07 nats**
- The fit is deliberately made **without** the smallest rung, then asked to predict it

<!--
alpha per L (deep/A): L1 0.164, L2 0.192, L8 0.163, L15 0.179, L30 0.185, L50 0.156.
run__resid over healthy deep/A/seed1904 cells: max |resid| 0.073.
Fits are per (L, arch, scheme) on the larger rungs — ladder_report.check_scaling.
-->

---
layout: figure
image: /ladder/ladder_report_loss.png
fit: contain
height: 72vh
title: Finding 2 — The 90M rung is not on the ladder
subtitle: "Nine of ten 90M runs peak early and then get worse"
---

---
title: Finding 2 — The 90M rung is not on the ladder
subtitle: "An optimizer timescale fixed in steps, on runs that differ 18×"
layout: figure
image: /ladder/optimizer_timescale.png
fit: contain
height: 52vh
---

---
layout: bullets
title: Why the 90M rung broke
subtitle: "One number explains it"
icon: "🔧"
---

- We train with AdEMAMix, which keeps a slow average of past gradients over 10,000 steps
- Each rung trains on its own budget, so run length spans 18 times across the ladder
- The 90M run finishes at 4,500 steps. It never reaches the regime the optimizer was set up for
- Retraining 90M with the averaging window tied to run length fixes it completely. Final loss goes from 5.762 to 2.778 and the drift falls from 1.365 to 0.011
- The corrected 90M then beats the uncomplete 175M, which suggests 175M is held back too

<!--
Improved version of the two tables. Same numbers, one picture and five sentences.
The lesson for future ladders is in the last two points: write every optimizer
timescale as a fraction of the run, never as a step count. plan/90M-rung-anomaly.md
carries the three learning rates we ruled out and the held out BPB evidence.
-->
---

Nine of ten 90M runs reach their **best loss at 15–19 % of training** and degrade for the rest.
Not overfitting (single epoch), not capacity (they get *worse*), not the LR (three rates spanning 5× all diverge).

AdEMAMix β₃ was held fixed at a **10,000-step** timescale while run length spans 18× across the ladder:

| rung | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| steps ÷ optimizer timescale | **0.45** | 0.85 | 1.7 | 2.9 | 4.6 | 8.1 |

| 90M, 2 languages | best loss | final loss | drift |
| --- | ---: | ---: | ---: |
| β₃ fixed at 10,000 steps | 4.397 | 5.762 | +1.365 |
| β₃ = 0.2 × run (900 steps) | 2.767 | **2.778** | **+0.011** |

The corrected 90M (2.778) beats the *uncorrected* 175M (2.904) — so 175M is likely depressed too.

<!--
plan/90M-rung-anomaly.md. The control changed nothing but beta3. Recommendation for future
ladders: express every optimizer timescale as a fraction of the run, never as a step count.
A 175M control at the same correction was running as of 09-03.
-->

---
layout: bullets
title: Finding 3 — One 90M cell slips through the filter
subtitle: "And it is the only 90M model in every analysis"
icon: "⚠️"
---

- The divergence test is `final > best + 0.25 nats`. **`90M-L2-shallow` ends 0.234 above its best** — 0.016 under the line
- So it is flagged healthy, while its scaling residual is **+1.29 nats**
- It is therefore the **only** 90M model that reaches the pools — every "90M" column is that one run
- It anchors **101 of RQ6's 505** scaling fits, the shallow scheme-A L2 chain, the only ladder that starts below 175M
- `run__off_trend` is already published in the ladder report; the loader ignores it

<!--
Verified on the real report: run__diverged = 0, run__off_trend = 1, run__resid = +1.291.
Fix is one line in snr/download/ladder.py (gate on off_trend alongside diverged) but it
changes what enters every pool, so it is a call for the team, not a patch.
-->

---
layout: figure
image: /ladder/english_vs_rest.png
fit: contain
height: 62vh
title: Finding 4 — More languages is nearly free for English
subtitle: "Held-out bits-per-byte, deep / scheme A / seed 1904"
---

---
layout: figure
image: /ladder/bpb_gain_per_language.png
fit: contain
height: 46vh
title: What more languages buys, language by language
subtitle: "Bits per byte saved by training on 50 languages instead of 2"
---

<!--
Improved version: no aggregate, every validation language is a column. Blue means
the language gained, red means it lost. 84 of the 100 gain on average across
sizes. The gain shrinks as the model grows, which is the top row being darker
than the bottom one. The losers on the right are Belarusian, Dhivehi and
Armenian, all of which sit in the 2 language mixture already.
-->

---
layout: figure
image: /ladder/english_vs_rest.png
fit: contain
height: 56vh
title: English pays once, everyone else keeps gaining
subtitle: "Dotted is English, solid is the median of the other 99 languages"
---

<!--
English costs 0.038 bits per byte going from 1 to 2 languages at 600M, then
0.007 for the next 48. The solid lines never flatten. This is the strongest
result the ladder has so far, and it holds at every size we have finished.
-->
---

- **English**: 600M pays **+0.038 bits/byte** going 1 → 2 languages, then **+0.007** for the next 48
- **Everything else**: non-English median falls **1.650 → 1.139** at 600M, **1.484 → 1.063** at 1B
- L50 beats L2 on **89 / 81 / 78 of the 99** non-English languages at 175M / 350M / 600M,
  median gain **+0.44 / +0.34 / +0.35 bits/byte**
- Against a seed noise of **0.011 bits/byte** on macro BPB — a 30× effect

<!--
From ladder_report.csv final checkpoints. English = bpb__dclm; non-English median over the
99 FineWeb-2 subsets. Seed std from the six x3 cells.
-->

---
layout: figure
image: /ladder/effect_vs_seed_noise.png
fit: contain
height: 58vh
title: Finding 5 — Only one of our three axes clears the noise floor
subtitle: "Median |Δ final loss| in units of the seed standard deviation"
---

---
layout: bullets
title: Finding 5 — Only one of our three axes clears the noise floor
subtitle: "Depth is currently indistinguishable from re-rolling the seed"
icon: "📉"
---

- Seed noise: **0.021 nats** on final loss, measured on the six ×3 cells
- **Language count**: 11.5× the seed noise, a real axis
- **Data scheme** (A vs B): 2.2× on loss, marginal
- **Model depth** (deep vs shallow): **1.0×** on the aggregate loss, the same model measured twice
- Per *individual* task the depth effect is larger, median **1.43×** with 35 % of 853 cells above 2×, so it separates *somewhere*, just not on the headline metric
- Depth is **half the grid**, and it buys a per-task effect we would have to hunt for

<!--
Matched pairs on healthy complete cells only: depth n=9, scheme n=7. Language axis is the
across-L range at fixed size (n=14 cells). Regenerated by documents/figures/fig_setup.py
from ladder_report.csv, so the figure and these numbers move together from now on.
-->


---
layout: figure
image: /ladder/fig1_gate.png
fit: contain
height: 52vh
title: Finding 6 — Emerged, or at chance
subtitle: "Same seven language settings, same sizes; one benchmark learns and one never leaves the line"
---

---
title: Finding 6 — Three quarters of the suite is at chance
subtitle: "A benchmark must beat 1/#options by +0.05 to enter any analysis"
---

Of the **461** tasks that have a chance level, **113 clear it at any size**.

| answer options | chance | clears chance | at 1B |
| :---: | :---: | :---: | :---: |
| **2** (completion, minimal pair) | 0.50 | **80 / 191** | 76 / 191 |
| **3** (XNLI) | 0.33 | **12 / 18** | 9 / 18 |
| **4** (knowledge MCQA) | 0.25 | **21 / 252** | 18 / 252 |

- Clearing chance at 1B: `multiblimp` (57), `hellaswag` (15), `xnli` (9), `xwinograd` (6), `xcopa` (6), `xstorycloze` (5), `arc` (2), `paws` (2), `include_base_44` (1)
- **Never clearing chance anywhere**: `global_mmlu_full`, `global_piqa_parallel_cloze`, `global_piqa_nonparallel_cloze`, `truthfulqa-multi_mc1`
- Strongly an **answer-count effect**: 42 % of 2-option tasks clear chance, **8 % of 4-option** ones do
- On the 36-model sweep the same families cleared chance for external 270M–70B models (122 of 124), a capability floor of the small rungs rather than a property of the benchmarks

<!--
above_random.py --only predictivity on the 14 September report: 568 tasks, 461 with a chance
level. The other 107 are per-language BPB, the training loss and the generative tasks, which
have none and are never gated. Per size: 90M 3/22, 175M 62/461, 350M 71/461, 600M 85/461,
1B 103/461, 1.7B 110/461. The 1.7B rung brings the first belebele task ever to clear chance.
-->

---
layout: figure
image: /ladder/first_clearing_size.png
fit: contain
height: 54vh
title: Which benchmark works, in which language
subtitle: "The smallest model that beats chance. Grey means it never does."
---

<!--
Improved version of the option count table, resolved per language. Read it row by
row. MultiBLiMP works earliest and in the most languages. XNLI, XStoryCloze,
HellaSwag, XWinograd and XCOPA need 350M or 600M and only work in some languages.
The bottom five families are grey everywhere, so they carry no signal at any size
we have trained. Blank means there is no task in that language at all. 90M is
excluded because its one surviving cell is off trend.
-->
---
layout: focus
color: blue
icon: "📏"
---

## In 55 of the 68 languages we can score at 1B, the most reliable measurement is now **MultiBLiMP**, not bits per byte

<!--
top_benchmarks_per_language.csv on the canonical pool, 14 September run: the rank-1 task is
a benchmark in 59 of 68 languages and multiblimp in 55 of them. Bits per byte is rank 1 in
only 9: bn, bo, ja, kn, ml, mr, ne, te, th. This reverses the reading of the previous deck,
and the eval coverage added since is what changed it. Say the caveat out loud: multiblimp is
a minimal-pair grammaticality test, so it separates models on morphosyntax, not on the
knowledge or reasoning the rest of the suite is meant to measure.
-->

---
layout: figure
image: /ladder/fig3_reliability_map.png
fit: contain
height: 70vh
title: Finding 7 — Reliability is a (family × language) map
subtitle: "Checkpoint decision accuracy at 1B; grey = removed by the gate, × = no task"
---

---
title: Finding 7 — One benchmark family now out-SNRs bits per byte
subtitle: "Highest-SNR above-random measurement per language (`discrepancy` @ 1B)"
---

| | rank-1 measurement | typical SNR |
|---|---|---|
| **59 languages** | a harness benchmark, `multiblimp` in 55 of them | median 13.2, up to 200.5 (sv 200.5, fr 164.2, es 122.5, ru 116.8) |
| **9 languages** | `bpb_<language>` | median 6.4 (bn, bo, ja, kn, ml, mr, ne, te, th) |

- The gate is still part of the story: most benchmarks are removed before SNR is computed, bits per byte never is
- The whole result rests on **one family**. Drop `multiblimp` and bits per byte leads again in most languages
- `multiblimp` tests minimal-pair grammaticality, so this says our models separate on morphosyntax, not on knowledge

<!--
This flips the previous deck. It is a coverage result as much as a quality one: multiblimp
covers 57 languages and clears the gate in all of them, while the knowledge benchmarks clear
it almost nowhere. Present it as "the suite now has one instrument that works" rather than
"the benchmarks work".
-->

---
layout: figure
image: /ladder/snr_bpb_vs_benchmark.png
fit: contain
height: 60vh
title: Finding 7, read again
subtitle: "Almost every language keeps a benchmark, and it usually wins the head to head"
---

<!--
Left bar: of the 68 languages with any measurable SNR at 1B, 62 still have a benchmark once the above-random
gate has run and 6 have none. Right: the 14 languages where both a benchmark and a bits per
byte SNR exist at 1B, best against best. The benchmark wins 11 of the 14. The comparison
needs a language to be trained in at least two of the settings finished at 1B, which is why
it is 14 languages and not 62.
-->
---
layout: bullets
title: We measure best where our choices matter least
subtitle: "What that number is really counting"
icon: "🔁"
---

- In **62 of the 68** validation languages a benchmark now survives the gate. Six months of the deck said the opposite, and the new evals are the reason
- Where **both** measurements exist, 14 languages, the benchmark wins **11**. The margins are large, not tiny: `multiblimp_fra` 164 against 13.6 for French bits per byte
- The catch is concentration. **55 of the 62** are won by `multiblimp` alone, so the suite has one working instrument rather than a working suite
- And it is a grammaticality instrument. Nothing here says the knowledge and reasoning benchmarks have started to work

<!--
Numbers from top_benchmarks_per_language.csv on the canonical pool, 14 September run,
variant discrepancy at 1B. SNR here is spread across design variants over the mean, so a
low score means the variants land on top of each other. The honest framing for the room:
coverage improved a lot, the diagnosis did not change much, because one family carries it.
-->

<!-- BEGIN auto:rq4-results (snr_definition_postprocess.py) -->
---
title: RQ4 — Surrogates: SNR definition
subtitle: "Results (auto) — most reliable benchmark per language (`aad` @ 1.7B)"
---

| lang | top benchmark | SNR | DA-ckpt@1.7B | BPB SNR |
|---|---|---|---|---|
| ar | `multiblimp_arb` | 0.9 | 0.42 | 0.2 |
| az | `rf_include_base_44_azerbaijani` | 0.5 |  | 0.2 |
| bg | `multiblimp_bul` | 1.3 | 0.59 | 0.1 |
| bn | `multiblimp_ben` | 1.2 | 0.37 | 0.3 |
| bs | `global_piqa_nonparallel_cloze_bos_latn` | 0.4 |  | 0.2 |
| ca | `xstorycloze_ca` | 2.1 |  | 0.2 |
| cs | `multiblimp_ces` | 0.7 | 0.67 | 0.2 |
| da | `multiblimp_dan` | 3.1 | 0.50 | 0.2 |
| de | `multiblimp_deu` | 1.1 | 0.58 | 0.5 |
| el | `multiblimp_ell` | 1.0 | 0.62 | 0.2 |
| en | `xwinograd_en` | 1.0 | 0.74 | 0.6 |
| es | `paws_es` | 1.1 | 0.75 | 0.3 |
| et | `multiblimp_est` | 1.2 |  | 0.3 |
| fa | `multiblimp_fas` | 0.5 | 0.62 | 0.4 |
| fi | `multiblimp_fin` | 0.9 | 0.72 | 0.2 |
| fr | `xwinograd_fr` | 1.0 | 0.53 | 0.4 |
| he | `multiblimp_heb` | 0.8 | 0.63 | 0.4 |
| hi | `xstorycloze_hi` | 0.7 | 0.52 | 0.2 |
| hr | `belebele_hrv_Latn` | 0.5 |  | 0.2 |
| hu | `multiblimp_hun` | 1.4 | 0.55 | 0.2 |
| id | `rfgm_include_base_44_indonesian` | 0.4 | 0.61 | 0.3 |
| it | `multiblimp_ita` | 0.9 | 0.64 | 0.4 |
| ja | `xwinograd_jp` | 1.0 | 0.69 | 0.7 |
| ka | `multiblimp_kat` | 1.1 | 0.53 | 0.2 |
| kk | `rf_include_base_44_kazakh` | 0.5 |  | 0.1 |
| ko | `rf_belebele_kor_Hang` | 0.4 | 0.65 | 0.6 |
| lt | `rf_include_base_44_lithuanian` | 0.8 |  | 0.2 |
| lv | `rf_belebele_lvs_Latn` | 0.1 |  | 0.3 |
| ml | `rf_belebele_mal_Mlym` | 0.1 | 0.61 | 0.3 |
| mr | `multiblimp_mar` | 0.5 |  | 0.1 |
| ms | `rfgm_include_base_44_malay` | 0.3 |  | 0.2 |
| ne | `rf_belebele_npi_Latn` | 0.6 |  | 0.1 |
| nl | `multiblimp_nld` | 1.3 | 0.63 | 0.2 |
| no | `rf_belebele_nob_Latn` | 0.4 | 0.69 | 0.2 |
| pl | `multiblimp_pol` | 1.0 | 0.60 | 0.2 |
| pt | `xwinograd_pt` | 1.4 | 0.61 | 0.3 |
| ro | `multiblimp_ron` | 0.7 | 0.53 | 0.2 |
| ru | `xnli_ru` | 1.0 | 0.67 | 0.6 |
| sk | `multiblimp_slk` | 0.3 |  | 0.2 |
| sl | `rf_belebele_slv_Latn` | 0.7 |  | 0.4 |
| sq | `rf_include_base_44_albanian` | 0.5 |  | 0.3 |
| sr | `rfgm_include_base_44_serbian` | 0.6 |  | 0.2 |
| sv | `rf_belebele_swe_Latn` | 0.3 | 0.47 | 0.3 |
| ta | `xcopa_ta` | 0.6 | 0.44 | 0.3 |
| th | `xnli_th` | 0.7 | 0.63 | 0.4 |
| tr | `multiblimp_tur` | 1.2 | 0.59 | 0.2 |
| uk | `multiblimp_ukr` | 0.8 | 0.53 | 0.1 |
| ur | `rf_belebele_urd_Arab` | 0.6 |  | 0.1 |
| vi | `xnli_vi` | 0.7 | 0.63 | 0.2 |
| zh | `xstorycloze_zh` | 0.8 | 0.65 | 0.8 |

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq4-results -->

---
layout: bullets
title: Finding 8 — SNR barely predicts decision accuracy here
subtitle: "R = 0.79 in Heineman et al., 0.06 on this ladder"
icon: "❓"
---

- Best of the 22 variants on the canonical pool is `discrepancy`: mean Pearson r of log₁₀(SNR) vs DA
  = **+0.06** (DA-size), **−0.05** (DA-ckpt), **0.00** overall
- Checkpoint decisions have a different leader, `dist_std` at **+0.20**, with `mad` and `rel_mpsd` beside it. No variant is best on both
- More eval coverage made this **worse**, not better. The thinner ladder gave +0.32 on DA-ckpt, this one gives +0.20 at best
- With 41 models in the canonical pool and a two-level intervention, DA is coarse. **The variant question may simply not be answerable at this pool size**

<!--
Compare Heineman et al. (2025): R = 0.791 between SNR and DA on the English DataDecide
ladder. Ours is over 68 languages with 41 models in the canonical pool. Do not oversell the
earlier +0.32: three successive runs of this pipeline have put the leader at +0.20, +0.32
and now +0.20, on different variants each time, which is what an estimate dominated by
noise looks like. The claim to make is the negative one.
-->

---
layout: bullets
title: SNR predicts much less here than in English
subtitle: "Plain version of the slide before"
icon: "❓"
---

- Heineman et al. report a correlation of 0.79 between SNR and decision accuracy on the English ladder
- Our best of 22 definitions reaches 0.20, when decisions are read across checkpoints
- Read across sizes the best is 0.06, and the overall winner sits at 0.00
- No definition is best on both kinds of decision, which is itself a sign the ranking is noise
- More evaluation data made this number smaller, not larger. The pool size is the thing holding us back
---
title: Finding 9 — No SNR choice survives a seed swap cleanly
subtitle: "Train on seeds 64/313, test on seed 1904"
---

| | DA-size | DA-ckpt |
| --- | ---: | ---: |
| Spearman ρ on the global variant ranking | **−0.20** | **+0.01** |
| Pearson r between splits (all cells) | +0.27 (n = 22) | −0.06 (n = 1,104) |
| Per-language family agreement | 1 % | 11 % |
| Per-language exact-variant agreement | 1 % | 5 % |

- Recommend an SNR **family**, never an exact variant
- The **per-language argmax never transfers**. 11 % of the 122 language cells agree even at family level
- Three runs of this holdout have given ρ = +0.70, +0.29 and now +0.01 on DA-ckpt. The estimate is noise

<!--
compare_seed_splits.py, predictivity_seeds_train -> predictivity_seeds_test. The x3 cells
are 175M/600M at L in {1,2,50} only, so this holdout is thinner than the 36-sweep's.
Retention reads 100 % under DA-size but on a single cell, so it is left off the table.
Same qualitative conclusion as the 36-sweep, which is itself reassuring.
-->

---
layout: figure
image: /ladder/seed_holdout.png
fit: contain
height: 52vh
title: Does the SNR ranking survive a seed swap
subtitle: "Pick the best variant on seeds 64 and 313, test it on seed 1904"
---

<!--
Improved version of the table. Blue is the checkpoint based ranking, orange is
the size based one. Only the checkpoint bars are usable. The last row is the one
that matters for the paper: the per language argmax agrees on 11 percent of
languages at family level and 5 percent exactly, so we recommend a family of SNR
definitions and never a single one. Numbers refreshed on the 14 September run.
-->
---
layout: bullets
title: Finding 10 — Late-checkpoint noise understates the real noise 2×
subtitle: "The standard S&N noise definition is optimistic on this ladder"
icon: "🔬"
---

- Signal-and-Noise measures noise as the spread over the **last few checkpoints** of one run
- We can measure it the other way too, over **seed replicates** of the same cell
- Median ratio **seed noise ÷ detrended checkpoint noise = 2.04**, over 5,546 (size, L, task) cells
- So an SNR computed the standard way is roughly **2× too optimistic** here
- An effect that looks like 2× checkpoint noise is about the size of **one seed re-roll**

<!--
rq03 effect_vs_noise.csv. Checkpoint noise is detrended first — under WSD the final window
is still descending, so the raw std would be smaller still. A methodological result about
the framework rather than about our models; worth reporting in the paper.
-->

---
title: Finding 11 — Bits per byte answers the question, benchmarks do not
subtitle: "RQ6 now scores both, and every comparison still resolves against 600M"
---

All **36 intervention comparisons** resolve against **600M**. Every finished cell above 600M is
deep and scheme A, so none of them has a matched twin to compare against.

Data-scheme decision, bits per byte over the languages both schemes train:

| proxy | L8 | L15 | L30 |
|---|---:|---:|---:|
| **175M** | 1.00 | 0.00 | **0.04** |
| **350M** | 1.00 | 0.83 | 1.00 |

175M gets the scheme decision **backwards** at L15 and L30, and 350M is reliable wherever it is measured.

- The encouraging half: the **scaling fits now reach the top of the ladder** (202 predict 1B, 202 predict 1.7B),
  and per-language BPB at the reference is predicted from the proxy rungs to within **4–8 %**

<!--
rq05 (with rq01 and rq03) on predictivity_seeds, 14 September: 36 intervention-DA cells, 505 scaling fits,
reference_size = 600M throughout. The depth rows exist only at L1/L2 and swing 0.13-0.77,
which is what "not enough matched pairs" looks like. Scaling error by reference: 0.076 for
1.7B at L1, 0.051 for 1B at L8, 0.058 for 1.7B at L30, 0.043 for 1B at L50, with L2 the one
outlier at 0.244. This slide is still the argument for finishing a matched pair above 600M.
-->

---
layout: figure
image: /ladder/benchmark_predictivity.png
fit: contain
height: 56vh
title: The benchmark suite can be scored, and it reads as a coin flip
subtitle: "Two schemes, 2,896 shared benchmark tasks per setting, one clear answer"
---

<!--
Left: the benchmark bars sit on the coin-flip line at both proxy rungs, 0.46 and 0.45 at
8 languages, 0.50 and 0.55 at 15, 0.57 and 0.48 at 30. Going from 175M to 350M does not
help and at 30 languages it hurts. Bits per byte behaves completely differently: wrong at
175M, right at 350M. Right: the strict test, a proxy that agrees and keeps agreeing at
every larger size. 365 of 500 bits per byte comparisons pass it, 3,995 of 8,449 benchmark
comparisons do. That 47 percent was 0 percent on the thinner ladder, but it is also about
what agreeing twice by chance would give, so read it as unreliable rather than as working.
-->
---
layout: bullets
title: What that means for the benchmark suite
subtitle: "The first direct answer to the question the ladder was built for"
icon: "🎯"
---

- The suite now has **2,896 shared task comparisons per language setting**. A month ago it had none
- The suite reads the scheme decision as a **coin flip at both proxy rungs**: 0.46 and 0.45 at 8 languages, 0.50 and 0.55 at 15, 0.57 and 0.48 at 30
- A bigger proxy does not help it. Bits per byte, by contrast, goes from wrong at 175M to right at 350M
- Under the strict test, a proxy that agrees **and keeps agreeing** as models grow, **3,995 of 8,449** benchmark comparisons pass against **365 of 500** for bits per byte. Two coin flips would give about the same 47 %
- We are not saying benchmarks cannot work. We are saying that at the sizes we can afford, **they do not carry the decision and bits per byte does**

<!--
Say this plainly to the room. The project's premise is that a cheap small model plus a
benchmark can stand in for an expensive large one. On the one decision we can now test at
scale, the benchmark half of that premise does not hold and the bits per byte half does.
That is a result, not a setback, and it is what makes the per-language BPB outcome metric
the right thing to have built the plan around.
-->

---
layout: figure
image: /ladder/min_predictive_size.png
fit: contain
height: 58vh
title: How small a proxy can we get away with
subtitle: "Every dot is one language. Lines are the aggregate measures."
---

---
layout: figure
image: /ladder/min_predictive_per_language.png
fit: contain
height: 46vh
title: The same question, language by language
subtitle: "Smallest proxy that picks the same winner as 600M"
---

---
layout: bullets
title: How small a proxy can we get away with
subtitle: "What the two plots say"
icon: "🪜"
---

- At **1 language** the depth decision is the hardest. 50 of the 100 validation subsets have no proxy size that gets it right
- At **8 and 15 languages** a 175M model is enough for most languages. The scheme decision is easier to predict than the depth one
- **Macro bits per byte is not a shortcut.** It fails outright at 1 and 8 languages, where the per language answers are mostly fine
- Training loss agrees with 600M at every language count except 1
- Benchmarks now appear at every language count, and **3,995 of 8,449** reach a proxy size that agrees and keeps agreeing. Per setting that is 38 of 73 at L1, 81 of 141 at L2, 1,160 of 2,740 at L8, 1,458 of 2,749 at L15 and 1,258 of 2,746 at L30

<!--
The reference here is 600M, not 1B, because no cell above 600M has a finished matched
shallow or scheme B twin. So this is a small to 600M read, not a small to large one. Two
messages for the room: the aggregate metric we planned to decide on is worse than the per
language ones it averages, and the benchmark line has moved off the floor but sits close to
what chance would give.
-->
---
layout: figure
image: /ladder/rq6_whiteboard.png
fit: contain
height: 58vh
title: The figure we actually want
subtitle: "Smallest model reaching each decision accuracy, against the number of languages"
---

---
layout: figure
image: /ladder/rq6_da_vs_size.png
fit: contain
height: 54vh
title: All the measurement behind it
subtitle: "Two proxy sizes, one reference. That is the whole thing."
---

---
layout: bullets
title: Why the curve is not there yet
subtitle: "Three things have to change before that figure can be drawn"
icon: "🧱"
---

- We have **two** usable proxy sizes, 175M and 350M. 90M diverged and 600M is the reference, so it scores 1.0 by construction
- The expected shape needs a size axis with four or five rungs and a reference above them. Today the whole plot is two columns wide
- At **1 language** the decision accuracy falls from 0.84 to 0.57 as the model grows. That is not a trend, it is the depth effect sitting inside the noise
- The scheme decision behaves as expected. At 8, 15 and 30 languages accuracy rises with size, and 350M clears 0.75 at all three
- Nothing reaches 0.9 anywhere. **Finish 1B with matched pairs and the picture becomes drawable**
---
layout: bullets
title: Not run — RQ3, agreement with DataDecide
subtitle: "The AllenAI SNR table is a git-lfs pointer in this clone"
icon: "⏸️"
---

- Everything else in this deck regenerates from the published ladder report
- RQ3 additionally needs the DataDecide-side SNR table — `git lfs pull` first, then re-run
- The shared universe is small either way: only the English tasks both corpora evaluate

<!-- BEGIN auto:rq7-results (rq07_external_frameworks/analyze.py) -->
---
title: RQ7 — External frameworks
subtitle: "Results (auto) — cross-corpus agreement with AllenAI by pool"
---

| pool | best variant | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `predictivity` | `aad` | 1.00 | 1.00 | 3 |
| `predictivity_seeds` | `dist_std` | 1.00 | 1.00 | 3 |

The shared universe is the English tasks both corpora evaluate, after the above-random gate.

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq7-results -->

---
layout: figure
image: /ladder/fig4_subset_sweep.png
fit: contain
height: 56vh
title: Finding 12 — A subject subset beats the full Global-MMLU at every size
subtitle: "Cumulative SNR as subjects are added in standalone-SNR order; dashed = the full set"
---

<!-- BEGIN auto:rq8-results (smooth_subtasks.py) -->
---
title: RQ8 — Subset selection
subtitle: "Results (auto) — top subset gains (SNR: full → best subset)"
---

| case | task | size | full → best SNR | +gain |
|---|---|---|---|---|
| per_benchmark | `multiblimp` | 350M | 3.16 → 4.66 | +1.50 |
| per_benchmark | `arc` | 1B | 2.57 → 3.97 | +1.40 |
| per_benchmark | `rf_belebele` | 350M | 3.16 → 4.41 | +1.25 |
| global_mmlu_full_subjects | `global_mmlu_full` | 600M | 2.14 → 3.33 | +1.18 |
| per_benchmark | `bpb` | 175M | 3.27 → 4.37 | +1.10 |
| per_benchmark | `multiblimp` | 175M | 3.44 → 4.49 | +1.04 |
| global_mmlu_full_per_language | `global_mmlu_full_sr` | 350M | 2.66 → 3.65 | +0.99 |
| global_mmlu_full_per_language | `global_mmlu_full_ms` | 350M | 2.53 → 3.43 | +0.91 |

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq8-results -->

<!-- BEGIN auto:rq9-results (rq09_benchmark_design/analyze.py) -->
---
title: RQ9 — Benchmark design
subtitle: "Results (auto) — per-family SNR, above-random survivors"
---

| family | median SNR | n_opts | format |
|---|---|---|---|
| `xwinograd` | 1.41 | 2 | completion |
| `multiblimp` | 1.24 | 2 | minimal_pair |
| `xstorycloze` | 1.20 | 2 | completion |
| `paws` | 1.19 | 2 | classification |
| `xcopa` | 0.84 | 2 | completion |
| `xnli` | 0.80 | 3 | classification |
| `rf_include_base_44` | 0.56 | 4 | cloze_completion |
| `include_base_44` | 0.56 | 4 | mcq_question_only |
| `rfgm_include_base_44` | 0.51 | 4 | statement_continuation |
| `belebele` | 0.47 | 4 | mrc_passage |
| `arc` | 0.47 | 4 | mcq_question_only |
| `rf_belebele` | 0.47 | 4 | cloze_completion |
| `global_piqa_parallel_cloze` | 0.42 | 2 | completion |
| `hellaswag` | 0.40 | 4 | completion |
| `rf_global_mmlu_full` | 0.30 | 4 | cloze_completion |

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq9-results -->

---
layout: section
---

# Open discussion

---
layout: default
color: amber
icon: "🔧"
---

## 1. The 90M rung — retrain corrected, drop it, or report both?

Correcting β₃ changes the optimizer at **every** rung, so a corrected 90M is not on the same ladder as the rest.

<!--
plan/90M-rung-anomaly.md settles the diagnosis, not the treatment. Options:
(a) report with and without the rung (current plan, cheapest);
(b) retrain 90M alone with the fraction-of-run beta3 and mark it as a different ladder;
(c) retrain every rung — outside the compute budget.
Whatever we pick, gate the loader on run__off_trend so 90M-L2-shallow stops being the
one 90M model in the pools.
-->

---
layout: default
color: red
icon: "🌡️"
---

## 2. Sampling temperature — T = 1 is not training 100 languages

At L100 with T = 1, **66 of 99 languages get under 10 M tokens** at 90M; the smallest gets 80 K.

**Recommendation: T = 2 sweep-wide.** Lifts the floor 40× (0.8 M → 33 M at 1B) and repeats nothing (≤ 0.3 epochs).

<!--
Head/tail share ratio at L100, T=1: 16,581:1. Eight languages were swapped INTO L100
because they have benchmarks; at T=1 they get ~100K tokens at 90M. A per-language SNR of
zero there measures the mixture, not the benchmark. T must never vary with L — that
confounds the intervention. plan/small-to-large-predictivity-training-plan.md.
-->

---
layout: default
color: amber
icon: "🪜"
---

## 4. 1.7B — 40 % of the sweep, zero finished cells

Dropping it frees **15,086 of 37,860 node-hours** and makes 1B the reference at every L.

But 1B is finished at **two** language settings, so the reference rung is thin either way.

<!--
Corollary from the temperature table: with 1B as the reference, the "66 of 99 languages
under 100M tokens" row IS the reference model, not a small rung. Dropping 1.7B makes the
temperature decision more urgent, not less.
-->

---
layout: default
color: blue
icon: "📊"
---

## 5. Is 5 × C the right budget at L ≥ 30?

ATLAS puts the compute-optimal ratio at **137 tokens/param at L50** and 193 at L100; we train **100**.

Cheap falsification first: check whether the languages bending the loss are the ones with negligible token share.

<!--
Full test needs annealed endpoints, not intermediate checkpoints (WSD leaves the LR
undecayed). 12 WSD cooldown branches (350M/600M x L in {1,30,100} x f in {0.25,0.5}) is
~6% of one level. But the temperature table suggests the tail is data-starved rather than
under-trained — that check is a spreadsheet, not a sweep.
-->

---
layout: default
color: amber
icon: "📐"
---

## 6. Model depth sits at the seed-noise floor

Depth is **half the grid** and its effect on final loss is **1.0× a seed re-roll** (n = 9 matched pairs).

Keep it, replace it with sampling temperature, or spend the compute on the missing reference rungs?

<!--
Finding 5. The scheme axis is 2.2x on loss and 1.2x on macro BPB — also marginal. The
intervention was one of three candidates (tokenizer, depth, temperature); temperature is
the one the plan calls "most likely to show a ranking that flips with scale".
-->

---
layout: default
color: blue
icon: "🎯"
---

## 7. Six benchmark families never clear chance

`belebele`, `global_mmlu_full`, `global_piqa` ×2, `paws`, `truthfulqa-multi` — no size, no language.

Do we drop them from the during-training suite, or keep them as the negative control the gate needs?

<!--
Dropping them buys eval walltime (discussion point 3) at the cost of the gate's evidence.
Related: INCLUDE v2 is the best single addition available (89 languages) but ships as
generative CoT — unusable for base models at these sizes until an MC variant exists.
LAMBADA-MT is marked legacy upstream: switch or drop. plan/benchmark_selection.md.
-->

---
layout: section
---

# Where this leaves us

---
layout: bullets
title: Summary
subtitle: "What the ladder has established, and what it cannot yet answer"
icon: "✅"
---

- **Established**: scaling holds above 90M (α ≈ 0.13–0.19, residuals ≤ 0.10 nats); more languages is nearly free for English and worth ~0.4 bits/byte to everything else; three quarters of the multilingual suite sits at chance at these sizes; no per-language SNR choice survives a seed swap, so recommend a *family*
- **Answered, for one decision**: on data scheme, per-language bits per byte transfers from a 350M proxy and the benchmark suite reads a coin flip at both proxy rungs, on 2,896 shared tasks per setting. Still small-to-600M, because nothing above 600M has a finished matched pair
- **Changed this run**: 62 of 68 languages now keep a benchmark after the gate, and `multiblimp` out-SNRs bits per byte in 55 of them. One family carries the whole improvement
- **Uncomfortable**: two of our three axes, depth and data scheme, are at or near the seed-noise floor, and SNR itself barely predicts decision accuracy (best +0.20 on checkpoint decisions)

---
layout: bullets
title: Where this leaves us
subtitle: "Plain version of the slide before"
icon: "✅"
---

- **Solid.** Scaling works above 90M. More languages costs English once and keeps paying everyone else. Three quarters of the benchmark suite sits at chance
- **Better than we thought.** 62 of 68 languages keep a benchmark after the gate, and it beats bits per byte in 11 of the 14 languages where both can be measured
- **Awkward.** All of that improvement is `multiblimp`, a grammaticality test. Our noise estimate is twice too small. Depth and scheme sit at or near the noise floor. SNR still does not predict decision accuracy
- **Open.** The question the ladder was built for. Every proxy comparison we can make today lands on 600M, because every finished cell above it is deep and scheme A
---
layout: bullets
title: Next
subtitle: "In the order things unblock each other"
icon: "➡️"
---

1. Decide **T** — it gates every remaining data build
2. Fix the **eval walltime** — it is why L = 100 is empty
3. Settle the **90M** treatment and gate the loader on `run__off_trend`
4. Finish a **matched pair above 600M**: a shallow or scheme-B twin of one of the four finished 1B/1.7B cells is what unblocks every small-to-large claim
5. Re-run `run_all_predictivity.sh` — every number in this deck refreshes from the published report


<!-- BEGIN generated signal slides (analysis/rq02_decision_accuracy/da_per_benchmark.py) -->

---
layout: section
---

# Appendix — Signal & Predictability across Sizes


---
title: Appendix — Above-random signal
subtitle: "Predictivity ladder (175M–1.7B, seed 1904) · mean score per family × size (bold = above chance in most of its tasks)"
---

| benchmark | rand | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| `loss` |  | 3.15 | 2.65 | 2.47 | 2.35 | 2.22 |
| `bpb` |  | 1.96 | 1.66 | 1.56 | 1.48 | 1.38 |
| `multiblimp` | 0.50 | **0.77** | **0.82** | **0.84** | **0.85** | **0.87** |
| `xwinograd` | 0.50 | 0.54 | **0.59** | **0.64** | **0.67** | **0.72** |
| `xcopa` | 0.50 | 0.53 | **0.54** | **0.55** | **0.57** | **0.59** |
| `xstorycloze` | 0.50 | 0.49 | **0.52** | **0.54** | **0.56** | **0.58** |
| `global_piqa_nonparallel_cloze` | 0.50 | 0.48 | 0.49 | 0.51 | 0.52 | 0.55 |
| `paws` | 0.50 | 0.49 | 0.50 | 0.50 | **0.52** | **0.54** |
| `truthfulqa_mc2` |  |  |  | 0.43 |  |  |
| `toxigen` | 0.50 |  |  | 0.43 |  |  |
| `xnli` | 0.33 | 0.35 | **0.39** | **0.41** | **0.42** | **0.44** |
| `lambada_openai_mt` |  | 0.16 | 0.27 | 0.33 | 0.37 | 0.42 |
| `cultural_bench_hard` | 0.50 |  |  | 0.30 |  |  |
| `hellaswag` | 0.25 | **0.26** | **0.28** | **0.30** | **0.32** | **0.35** |
| `rf_belebele` | 0.25 | **0.28** | **0.29** | **0.29** | **0.31** | **0.32** |
| `rfgm_include_base_44` | 0.25 | 0.26 | 0.27 | **0.29** | **0.30** | **0.32** |
| `rf_include_base_44` | 0.25 | 0.26 | 0.27 | 0.28 | **0.29** | **0.31** |
| `rf_global_mmlu_full` | 0.25 | 0.26 | **0.26** | **0.26** | **0.27** | **0.29** |
| `mmlu` | 0.25 |  |  | 0.26 |  |  |
| `include_base_44` | 0.25 | 0.25 | 0.25 | 0.25 | 0.26 | 0.25 |
| `cultural_bench_easy` | 0.25 |  |  | 0.25 |  |  |
| `belebele` | 0.25 | 0.25 | 0.24 | 0.24 | 0.24 | 0.25 |
| `global_mmlu_full` | 0.25 | 0.24 | 0.25 | 0.24 | 0.24 | 0.25 |
| `blend_sample` | 0.25 |  |  | 0.24 |  |  |
| `arc` | 0.25 | 0.20 | 0.22 | 0.23 | 0.25 | 0.27 |
| `openbookqa` | 0.25 |  |  | 0.23 |  |  |
| `truthfulqa-multi_mc1` | 0.25 | 0.25 | 0.23 | 0.23 | 0.23 | 0.22 |
| `mathqa` | 0.20 |  |  | 0.23 |  |  |
| `global_piqa_parallel_cloze` | 0.25 | 0.20 | 0.21 | 0.21 | 0.22 | 0.23 |
| `commonsense_qa` | 0.20 |  |  | 0.19 |  |  |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Above-random signal
subtitle: "Custom Apertus pretrains only · mean score per family × size (bold = above chance in most of its tasks)"
---

| benchmark | rand | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| `multiblimp` | 0.50 | **0.90** | **0.91** | **0.92** | **0.93** |
| `piqa` | 0.50 | **0.67** | **0.70** | **0.71** | **0.73** |
| `xwinograd` | 0.50 | **0.60** | **0.64** | **0.67** | **0.70** |
| `xstorycloze` | 0.50 | 0.54 | 0.55 | **0.57** | **0.58** |
| `xcopa` | 0.50 | 0.54 | **0.55** | **0.56** | **0.56** |
| `global_piqa_completions` | 0.50 | 0.50 | 0.52 | 0.53 | 0.54 |
| `paws` | 0.50 | 0.51 | 0.51 | 0.51 | 0.52 |
| `xnli` | 0.33 | 0.38 | **0.39** | **0.40** | **0.40** |
| `hellaswag` | 0.25 | 0.29 | **0.31** | **0.33** | **0.34** |
| `arc` | 0.25 | 0.26 | 0.27 | 0.29 | 0.30 |
| `truthfulqa` | 0.25 | 0.26 | 0.26 | 0.26 | 0.26 |
| `agieval_sat` | 0.25 | 0.26 | 0.25 | 0.26 | 0.24 |
| `mmlu` | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| `belebele` | 0.25 | 0.24 | 0.25 | 0.25 | 0.25 |
| `global_mmlu_full` | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| `global_mmlu` | 0.25 |  | 0.25 |  |  |
| `arabic_leaderboard_alghafa_mcq_exams_test` | 0.25 | 0.24 | 0.25 | 0.24 | 0.24 |
| `agieval_logiqa` | 0.25 | 0.22 | 0.22 | 0.23 | 0.23 |
| `truthfulqa_mc1` | 0.25 | 0.23 | 0.22 | 0.22 | 0.23 |
| `openbookqa` | 0.25 | 0.20 | 0.22 | 0.23 | 0.25 |
| `agieval_lsat` | 0.20 | 0.22 | 0.22 | 0.22 | 0.22 |
| `commonsense_qa` | 0.20 | 0.20 | 0.21 | 0.21 | 0.20 |
| `agieval` | 0.25 | 0.18 | 0.18 | 0.18 | 0.18 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Above-random signal
subtitle: "All models (custom + Swiss-AI/HF refs) · mean score per family × size (bold = above chance in most of its tasks)"
---

| benchmark | rand | 175M | 270M | 350M | 600M | 1B | 1.7B | 3B | 4B | 7-9B | 12-14B | 27-32B | 70B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `multiblimp` | 0.50 | **0.90** | **0.91** | **0.91** | **0.92** | **0.93** | **0.94** | **0.92** | **0.96** | **0.91** | **0.94** | **0.94** | **0.98** |
| `piqa` | 0.50 | **0.67** | **0.68** | **0.70** | **0.71** | **0.73** | **0.76** | **0.77** | **0.79** | **0.79** | **0.81** | **0.81** | **0.83** |
| `xwinograd` | 0.50 | **0.60** | **0.63** | **0.64** | **0.67** | **0.70** | **0.72** | **0.76** | **0.77** | **0.76** | **0.79** | **0.79** | **0.82** |
| `xstorycloze` | 0.50 | 0.54 | 0.54 | 0.55 | **0.56** | **0.58** | **0.59** | **0.62** | **0.64** | **0.63** | **0.67** | **0.68** | **0.71** |
| `xcopa` | 0.50 | 0.54 | **0.56** | **0.55** | **0.56** | **0.57** | **0.59** | **0.60** | **0.64** | **0.60** | **0.65** | **0.64** | **0.70** |
| `global_piqa_completions` | 0.50 | 0.50 | 0.51 | 0.52 | 0.53 | 0.55 | **0.57** | **0.61** | **0.63** | 0.60 | **0.65** | **0.65** | **0.74** |
| `paws` | 0.50 | 0.51 | 0.52 | 0.51 | 0.51 | 0.53 | **0.58** | **0.57** | **0.60** | **0.57** | **0.60** | **0.61** | **0.60** |
| `agieval_sat` | 0.25 | 0.26 | 0.30 | 0.25 | 0.29 | 0.25 | **0.64** | **0.41** | **0.78** | **0.65** | **0.83** | **0.83** | **0.74** |
| `commonsense_qa` | 0.20 | 0.20 | 0.21 | 0.21 | 0.24 | 0.22 | **0.73** | **0.48** | **0.70** | **0.69** | **0.77** | **0.76** | **0.54** |
| `mmlu` | 0.25 | 0.25 | 0.26 | 0.25 | 0.27 | 0.27 | **0.61** | **0.46** | **0.65** | **0.63** | **0.71** | **0.73** | **0.65** |
| `belebele` | 0.25 | 0.24 | 0.24 | 0.25 | 0.27 | 0.26 | **0.60** | **0.43** | **0.66** | **0.55** | **0.70** | **0.69** | **0.68** |
| `xnli` | 0.33 | 0.38 | **0.39** | **0.39** | **0.40** | **0.41** | **0.42** | **0.42** | **0.44** | **0.41** | **0.43** | **0.43** | **0.46** |
| `global_mmlu_full` | 0.25 | 0.25 | 0.26 | 0.25 | 0.26 | 0.25 | **0.46** | **0.38** | **0.52** | **0.46** | **0.56** | **0.56** | **0.53** |
| `hellaswag` | 0.25 | 0.29 | 0.29 | **0.31** | **0.33** | **0.34** | **0.36** | **0.40** | **0.41** | **0.40** | **0.44** | **0.45** | **0.49** |
| `global_mmlu` | 0.25 |  |  | 0.25 |  |  |  | **0.49** |  |  |  |  |  |
| `arc` | 0.25 | 0.26 | 0.25 | 0.27 | 0.29 | 0.30 | **0.36** | **0.37** | **0.43** | **0.40** | **0.46** | **0.46** | **0.47** |
| `truthfulqa` | 0.25 | 0.26 | 0.35 | 0.26 | 0.35 | 0.35 | **0.38** | **0.35** | **0.38** | **0.38** | **0.38** | **0.37** | **0.40** |
| `arabic_leaderboard_alghafa_mcq_exams_test` | 0.25 | 0.24 | 0.25 | 0.25 | 0.24 | 0.24 | **0.37** | **0.36** | **0.43** | **0.36** | **0.45** | **0.45** | **0.48** |
| `agieval_logiqa` | 0.25 | 0.22 | 0.21 | 0.22 | 0.23 | 0.23 | 0.32 | 0.26 | **0.39** | **0.35** | **0.42** | **0.39** | **0.36** |
| `agieval` | 0.25 | 0.18 | 0.18 | 0.18 | 0.19 | 0.18 | **0.35** | 0.25 | **0.41** | **0.36** | **0.46** | **0.46** | **0.36** |
| `openbookqa` | 0.25 | 0.20 | 0.21 | 0.22 | 0.24 | 0.26 | **0.30** | **0.30** | **0.33** | **0.34** | **0.36** | **0.36** | **0.38** |
| `truthfulqa_mc1` | 0.25 | 0.23 | 0.24 | 0.22 | 0.23 | 0.23 | **0.32** | 0.27 | **0.32** | **0.31** | **0.35** | **0.34** | **0.36** |
| `agieval_lsat` | 0.20 | 0.22 | 0.24 | 0.22 | 0.21 | 0.21 | **0.25** | 0.23 | 0.21 | 0.22 | 0.25 | 0.24 | 0.21 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_by_benchmark.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy, all languages at once
subtitle: "Benchmark × size pair, averaged over every language it covers"
---

---
layout: figure
image: /ladder/appendix/da_by_language.png
fit: contain
height: 78vh
title: Appendix — Decision accuracy, all benchmarks at once
subtitle: "Language × size pair, averaged over every benchmark it has"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "English (en) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `hellaswag` | 0.59 | 0.48 | 0.66 | 0.70 | 0.71 | 0.69 | **0.76** | 0.72 | **0.75** | **0.86** |
| `bpb` | 0.67 | 0.39 | 0.75 | **0.78** | 0.61 | 0.71 | 0.70 | 0.59 | 0.54 | **0.84** |
| `xstorycloze` | 0.60 | 0.53 | 0.59 | 0.58 | 0.71 | 0.64 | 0.70 | 0.73 | 0.75 | 0.67 |
| `lambada_openai_mt` | 0.68 | 0.62 | 0.67 | 0.72 | 0.46 | 0.49 | 0.51 | **0.77** | 0.68 | **0.82** |
| `arc_challenge` | 0.58 | 0.56 | 0.48 | 0.58 | **0.84** | 0.61 | 0.65 | 0.65 | 0.73 | 0.66 |
| `xwinograd` | 0.67 | 0.59 | 0.58 | 0.53 | 0.57 | 0.71 | 0.71 | 0.67 | 0.58 | 0.69 |
| `rf_global_mmlu_full` | 0.60 | 0.45 | 0.56 | 0.54 | 0.64 | 0.47 | 0.41 | 0.61 | 0.66 | 0.74 |
| `arc_easy` | 0.47 | 0.42 | 0.59 | 0.67 | 0.71 | 0.52 | 0.60 | 0.53 | 0.52 | 0.50 |
| `rf_belebele` | 0.46 | 0.59 | 0.63 | 0.42 | 0.51 | 0.56 | 0.62 | 0.72 | 0.45 | 0.54 |
| `xnli` | 0.54 | 0.65 | 0.48 | 0.58 | 0.62 | 0.60 | 0.41 | 0.39 | 0.59 | 0.41 |
| `global_mmlu_full` | 0.52 | 0.54 | 0.50 | 0.54 | 0.46 | 0.50 | 0.45 | 0.54 | 0.58 | 0.49 |
| `global_piqa_parallel_cloze` | 0.40 | 0.70 | 0.50 | 0.47 | 0.43 | 0.52 | 0.47 | 0.48 | 0.52 | 0.46 |
| `paws` | 0.63 | 0.49 | 0.57 | 0.38 | 0.43 | 0.46 | 0.63 | 0.43 | 0.42 | 0.48 |
| `belebele` | 0.51 | 0.63 | 0.42 | 0.44 | 0.41 | 0.42 | 0.41 | 0.59 | 0.48 | 0.45 |
| `multiblimp` | 0.66 | 0.52 | 0.35 | 0.58 | 0.39 | 0.42 | 0.52 | 0.43 | 0.46 | 0.31 |
| `truthfulqa-multi_mc1` | 0.40 | 0.17 | 0.58 | 0.42 | 0.46 | 0.54 | 0.32 | 0.44 | 0.61 | 0.35 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_en.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "English (en) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Modern Std. Arabic (ar) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.64 | 0.46 | 0.64 | **0.75** | **0.75** | **0.86** | **0.75** | **0.75** | 0.57 | 0.68 |
| `multiblimp` | **0.79** | 0.39 | 0.68 | 0.50 | 0.39 | 0.54 | 0.50 | 0.54 | **0.75** | 0.50 |
| `hellaswag` | 0.46 | 0.54 | 0.54 | 0.57 | 0.50 | 0.64 | 0.61 | 0.57 | 0.39 | 0.68 |
| `rf_belebele_arz_Arab` | 0.50 | 0.25 | 0.71 | 0.43 | 0.64 | 0.64 | **0.75** | 0.50 | 0.61 | 0.43 |
| `rf_include_base_44` | 0.50 | 0.68 | 0.57 | 0.43 | 0.46 | **0.75** | 0.50 | 0.39 | 0.43 | 0.46 |
| `global_mmlu_full` | 0.54 | 0.46 | 0.54 | 0.61 | 0.57 | 0.50 | 0.43 | 0.43 | 0.50 | 0.57 |
| `belebele_arb_Arab` | 0.25 | 0.36 | 0.61 | 0.36 | 0.54 | 0.46 | 0.43 | 0.71 | 0.71 | 0.57 |
| `rf_global_mmlu_full` | 0.54 | 0.64 | 0.36 | 0.46 | 0.61 | 0.57 | 0.36 | 0.29 | 0.39 | 0.68 |
| `rf_belebele_arb_Latn` | 0.43 | 0.50 | 0.50 | 0.54 | 0.57 | 0.25 | 0.61 | 0.29 | **0.82** | 0.32 |
| `arc` | 0.50 | 0.50 | 0.36 | 0.50 | 0.43 | 0.57 | 0.64 | 0.50 | 0.25 | 0.43 |
| `belebele_apc_Arab` | 0.50 | 0.36 | 0.68 | 0.46 | 0.43 | 0.43 | 0.25 | 0.64 | 0.46 | 0.43 |
| `rfgm_include_base_44` | 0.39 | 0.57 | 0.57 | 0.57 | 0.25 | 0.32 | 0.43 | 0.50 | 0.50 | 0.50 |
| `belebele_ary_Arab` | 0.32 | 0.39 | **0.75** | 0.39 | 0.68 | 0.39 | 0.32 | 0.43 | 0.54 | 0.39 |
| `belebele_ars_Arab` | 0.46 | 0.39 | **0.82** | 0.36 | 0.21 | 0.43 | 0.43 | 0.39 | 0.57 | 0.50 |
| `rf_belebele_ary_Arab` | 0.25 | 0.54 | 0.64 | 0.43 | 0.43 | 0.29 | **0.75** | 0.61 | 0.36 | 0.21 |
| `rf_belebele_ars_Arab` | 0.50 | 0.36 | 0.64 | 0.18 | 0.57 | 0.50 | 0.46 | 0.43 | 0.57 | 0.21 |
| `xstorycloze` | 0.50 | 0.43 | 0.43 | 0.39 | 0.61 | 0.54 | 0.25 | 0.54 | 0.32 | 0.43 |
| `rf_belebele_apc_Arab` | 0.46 | 0.39 | 0.32 | 0.57 | **0.75** | 0.18 | 0.57 | 0.32 | 0.43 | 0.39 |
| `xnli` | 0.50 | 0.39 | 0.64 | 0.54 | 0.46 | 0.43 | 0.25 | 0.32 | 0.21 | 0.61 |
| `include_base_44` | 0.29 | 0.29 | 0.39 | 0.61 | 0.43 | 0.57 | 0.39 | 0.36 | 0.50 | 0.43 |
| `rf_belebele_arb_Arab` | 0.54 | 0.61 | 0.25 | 0.50 | **0.79** | 0.43 | 0.21 | 0.21 | 0.29 | 0.43 |
| `belebele_arb_Latn` | 0.71 | 0.43 | 0.36 | 0.43 | 0.39 | 0.25 | 0.43 | 0.36 | 0.43 | 0.43 |
| `belebele_arz_Arab` | 0.36 | 0.46 | 0.64 | 0.14 | 0.21 | 0.43 | 0.50 | 0.61 | 0.54 | 0.29 |
| `global_piqa_parallel_cloze_apc_arab_leba` | 0.36 | 0.50 | 0.14 | 0.54 | 0.21 | 0.64 | 0.43 | 0.32 | 0.36 | 0.46 |
| `global_piqa_parallel_cloze_arz_arab` | 0.32 | 0.21 | 0.32 | 0.46 | 0.36 | 0.46 | 0.25 | 0.46 | 0.46 | 0.57 |
| `global_piqa_parallel_cloze_apc_arab_pale` | 0.14 | 0.54 | 0.11 | 0.61 | 0.50 | 0.68 | 0.29 | 0.32 | 0.39 | 0.29 |
| `global_piqa_parallel_cloze_ary_arab` | 0.50 | 0.36 | 0.36 | 0.39 | 0.36 | 0.36 | 0.36 | 0.32 | 0.29 | 0.50 |
| `global_piqa_parallel_cloze_arb_arab` | 0.25 | 0.36 | 0.36 | 0.43 | 0.29 | 0.61 | 0.18 | 0.36 | 0.50 | 0.36 |
| `global_piqa_parallel_cloze_ars_arab` | 0.36 | 0.46 | 0.46 | 0.32 | 0.11 | 0.46 | 0.29 | 0.43 | 0.29 | 0.36 |
| `global_piqa_parallel_cloze_apc_arab_syri` | 0.36 | 0.46 | 0.29 | 0.54 | 0.29 | 0.39 | 0.21 | 0.25 | 0.39 | 0.14 |
| `global_piqa_parallel_cloze_apc_arab_jord` | 0.50 | 0.14 | 0.29 | 0.54 | 0.14 | 0.21 | 0.32 | 0.21 | 0.21 | 0.32 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ar.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Modern Std. Arabic (ar) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "bg (bg) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `multiblimp` | **0.83** | 0.50 | 0.33 | 0.50 | 0.67 | 0.50 | 0.67 | 0.50 | **1.00** | 0.50 |
| `bpb` | **1.00** | 0.17 | **0.83** | **0.83** | 0.17 | **0.83** | **0.83** | 0.00 | 0.00 | **1.00** |
| `rf_belebele` | 0.33 | 0.67 | 0.17 | 0.33 | 0.67 | **0.83** | 0.67 | 0.50 | 0.67 | 0.50 |
| `rf_include_base_44` | 0.33 | **1.00** | 0.67 | 0.50 | 0.33 | 0.67 | 0.17 | 0.67 | 0.50 | 0.50 |
| `xnli` | 0.67 | 0.67 | 0.33 | 0.50 | **1.00** | 0.33 | 0.50 | 0.33 | 0.50 | 0.17 |
| `include_base_44` | 0.33 | 0.67 | 0.17 | 0.50 | 0.33 | 0.50 | **0.83** | 0.50 | 0.17 | 0.33 |
| `rfgm_include_base_44` | 0.67 | 0.50 | 0.33 | 0.50 | 0.50 | 0.33 | 0.50 | 0.17 | 0.00 | **0.83** |
| `belebele` | 0.17 | 0.67 | 0.67 | 0.50 | 0.00 | 0.50 | 0.33 | 0.33 | 0.50 | 0.50 |
| `global_piqa_parallel_cloze` | 0.33 | 0.17 | 0.00 | 0.50 | 0.67 | 0.50 | 0.50 | 0.50 | 0.17 | 0.50 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_bg.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "bg (bg) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "bn (bn) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.68 | 0.54 | 0.71 | **0.89** | **0.86** | **0.82** | **0.79** | **0.82** | 0.64 | **0.82** |
| `rfgm_include_base_44` | 0.61 | 0.46 | 0.43 | 0.54 | 0.36 | 0.36 | 0.36 | 0.64 | 0.64 | 0.57 |
| `rf_include_base_44` | 0.50 | 0.64 | 0.43 | 0.39 | 0.61 | 0.32 | 0.32 | 0.54 | 0.61 | 0.57 |
| `rf_belebele_ben_Beng` | 0.57 | 0.54 | 0.68 | 0.54 | 0.21 | 0.71 | 0.21 | 0.25 | **0.79** | 0.39 |
| `rf_belebele_ben_Latn` | 0.36 | 0.46 | 0.21 | **0.75** | 0.36 | 0.64 | 0.50 | 0.57 | 0.57 | 0.46 |
| `rf_global_mmlu_full` | 0.61 | 0.61 | 0.36 | 0.43 | 0.39 | 0.46 | 0.61 | 0.39 | 0.46 | 0.57 |
| `global_mmlu_full` | 0.43 | 0.46 | 0.57 | 0.50 | 0.46 | 0.57 | 0.50 | 0.32 | 0.32 | 0.71 |
| `belebele_ben_Beng` | 0.39 | 0.61 | 0.68 | 0.71 | 0.36 | 0.25 | 0.29 | 0.54 | 0.50 | 0.50 |
| `include_base_44` | 0.32 | **0.79** | 0.46 | 0.46 | 0.29 | 0.54 | 0.29 | 0.46 | 0.50 | 0.64 |
| `belebele_ben_Latn` | 0.36 | 0.54 | 0.54 | 0.36 | 0.32 | 0.50 | 0.64 | 0.29 | 0.36 | **0.75** |
| `global_piqa_parallel_cloze_ben_latn` | 0.46 | 0.64 | 0.64 | 0.43 | 0.32 | 0.46 | 0.64 | 0.43 | 0.36 | 0.21 |
| `arc` | 0.29 | 0.25 | 0.54 | 0.61 | 0.54 | 0.39 | 0.46 | 0.18 | 0.57 | 0.46 |
| `hellaswag` | 0.29 | 0.25 | 0.46 | 0.46 | 0.57 | 0.39 | 0.57 | 0.43 | 0.46 | 0.39 |
| `global_piqa_parallel_cloze_ben_beng` | 0.50 | 0.21 | 0.32 | 0.21 | 0.54 | 0.29 | 0.29 | 0.29 | 0.57 | 0.54 |
| `multiblimp` | 0.29 | 0.57 | 0.36 | 0.32 | 0.29 | 0.21 | 0.18 | 0.25 | 0.43 | 0.25 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_bn.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "bn (bn) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "cs (cs) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.80** | 0.47 | **0.87** | **0.93** | 0.67 | **0.93** | **0.87** | 0.60 | 0.53 | **0.80** |
| `multiblimp` | **0.87** | 0.60 | 0.60 | 0.67 | 0.73 | 0.60 | 0.53 | 0.47 | 0.40 | 0.67 |
| `belebele` | 0.67 | 0.53 | 0.27 | 0.60 | 0.60 | 0.33 | 0.47 | 0.47 | 0.53 | 0.40 |
| `global_mmlu_full` | 0.20 | 0.47 | 0.47 | 0.33 | 0.33 | 0.33 | 0.60 | 0.60 | 0.60 | 0.60 |
| `rf_belebele` | 0.60 | 0.47 | 0.47 | 0.60 | 0.60 | 0.40 | 0.27 | 0.27 | 0.27 | 0.60 |
| `rf_global_mmlu_full` | 0.60 | 0.40 | 0.47 | 0.33 | 0.40 | 0.60 | 0.20 | 0.53 | 0.40 | 0.33 |
| `global_piqa_parallel_cloze` | 0.13 | 0.73 | 0.33 | 0.33 | 0.27 | 0.47 | 0.27 | 0.27 | 0.47 | 0.07 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_cs.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "cs (cs) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "da (da) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `hellaswag` | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | 0.67 | **1.00** | **1.00** | **1.00** |
| `bpb` | 0.67 | 0.17 | 0.67 | **0.83** | 0.50 | **1.00** | **0.83** | 0.50 | 0.33 | **0.83** |
| `belebele` | 0.50 | 0.50 | **0.83** | 0.33 | 0.33 | 0.50 | 0.67 | 0.50 | 0.00 | 0.50 |
| `multiblimp` | 0.50 | 0.33 | **1.00** | 0.50 | 0.17 | 0.50 | 0.17 | 0.33 | 0.50 | 0.50 |
| `rf_belebele` | 0.50 | 0.50 | 0.50 | 0.17 | 0.33 | 0.67 | 0.33 | 0.17 | 0.17 | 0.67 |
| `arc` | 0.00 | 0.17 | 0.50 | **0.83** | 0.67 | 0.50 | 0.00 | 0.17 | 0.33 | 0.50 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_da.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "da (da) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "German (de) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.65 | 0.57 | **0.77** | **0.76** | **0.90** | **0.88** | **0.82** | **0.80** | **0.77** | **0.86** |
| `hellaswag` | 0.52 | 0.46 | 0.56 | 0.56 | **0.81** | 0.65 | 0.68 | **0.77** | **0.77** | **0.80** |
| `lambada_openai_mt` | 0.63 | 0.56 | **0.78** | 0.69 | 0.51 | 0.65 | 0.59 | 0.64 | 0.57 | 0.71 |
| `xnli` | 0.49 | 0.68 | 0.70 | 0.53 | 0.51 | 0.48 | 0.71 | **0.81** | 0.55 | 0.49 |
| `multiblimp` | 0.57 | 0.63 | 0.60 | 0.49 | 0.62 | 0.55 | 0.53 | 0.59 | 0.64 | 0.54 |
| `arc` | 0.47 | 0.54 | 0.59 | 0.52 | 0.52 | 0.49 | 0.52 | 0.51 | 0.58 | 0.69 |
| `paws` | 0.32 | 0.74 | 0.40 | 0.43 | 0.51 | 0.52 | 0.51 | 0.51 | 0.48 | 0.73 |
| `rf_belebele` | 0.59 | 0.65 | 0.44 | 0.38 | 0.47 | 0.49 | 0.40 | 0.59 | 0.43 | 0.52 |
| `include_base_44` | 0.25 | 0.57 | 0.52 | 0.65 | 0.48 | 0.41 | 0.52 | 0.44 | 0.52 | 0.53 |
| `rf_global_mmlu_full` | 0.33 | 0.52 | 0.43 | 0.41 | 0.34 | 0.40 | 0.43 | 0.63 | 0.56 | 0.71 |
| `global_mmlu_full` | 0.30 | 0.42 | 0.56 | 0.33 | 0.38 | 0.56 | 0.59 | 0.42 | 0.70 | 0.42 |
| `belebele` | 0.35 | 0.27 | 0.43 | 0.64 | 0.44 | 0.42 | 0.33 | 0.48 | 0.35 | 0.55 |
| `global_piqa_parallel_cloze` | 0.30 | 0.33 | 0.47 | 0.40 | 0.59 | 0.42 | 0.34 | 0.41 | 0.52 | 0.49 |
| `rf_include_base_44` | 0.21 | 0.48 | 0.59 | 0.25 | 0.47 | 0.30 | 0.56 | 0.35 | 0.60 | 0.40 |
| `rfgm_include_base_44` | 0.37 | 0.44 | 0.49 | 0.51 | 0.27 | 0.44 | 0.32 | 0.30 | 0.16 | 0.44 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_de.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "German (de) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "el (el) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.64 | 0.51 | **0.76** | 0.67 | **0.82** | **0.89** | **0.80** | **0.76** | 0.67 | **0.82** |
| `rf_belebele` | 0.60 | 0.47 | 0.56 | 0.58 | 0.56 | 0.64 | 0.38 | 0.69 | 0.62 | 0.60 |
| `rf_global_mmlu_full` | 0.51 | 0.47 | 0.47 | 0.53 | 0.60 | 0.38 | 0.62 | 0.64 | 0.53 | 0.44 |
| `multiblimp` | 0.56 | 0.53 | 0.29 | 0.53 | 0.42 | 0.42 | 0.56 | 0.53 | 0.49 | 0.60 |
| `belebele` | 0.44 | 0.38 | 0.47 | **0.87** | 0.24 | 0.58 | 0.58 | 0.51 | 0.29 | 0.56 |
| `xnli` | 0.60 | 0.36 | 0.58 | 0.53 | 0.18 | 0.49 | 0.62 | 0.56 | 0.31 | 0.58 |
| `global_mmlu_full` | 0.71 | 0.44 | 0.47 | 0.40 | 0.47 | 0.62 | 0.24 | 0.47 | 0.36 | 0.53 |
| `rfgm_include_base_44` | 0.36 | 0.47 | 0.22 | 0.33 | 0.53 | 0.53 | 0.67 | 0.38 | 0.60 | 0.58 |
| `include_base_44` | 0.47 | 0.38 | 0.62 | 0.38 | 0.53 | 0.47 | 0.44 | 0.49 | 0.44 | 0.42 |
| `global_piqa_parallel_cloze` | 0.60 | 0.51 | 0.33 | 0.36 | 0.49 | 0.29 | 0.33 | 0.51 | 0.40 | 0.31 |
| `rf_include_base_44` | 0.36 | 0.53 | 0.56 | 0.29 | 0.36 | 0.42 | 0.49 | 0.38 | 0.20 | 0.53 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_el.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "el (el) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Spanish (es) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `hellaswag` | 0.62 | 0.62 | **0.76** | 0.62 | **0.87** | **0.84** | **0.89** | 0.73 | **0.78** | **0.87** |
| `bpb` | 0.64 | 0.47 | 0.73 | 0.69 | **0.78** | **0.87** | **0.78** | 0.69 | 0.64 | **0.78** |
| `xstorycloze` | 0.38 | 0.42 | 0.40 | 0.40 | 0.67 | 0.56 | **0.76** | 0.67 | **0.89** | 0.67 |
| `arc` | 0.42 | 0.29 | 0.49 | 0.49 | 0.60 | 0.62 | 0.73 | 0.51 | 0.62 | 0.71 |
| `multiblimp` | 0.56 | 0.49 | 0.51 | **0.84** | 0.24 | 0.60 | 0.69 | 0.49 | 0.40 | 0.60 |
| `rf_global_mmlu_full` | 0.47 | 0.56 | 0.53 | 0.62 | 0.53 | 0.56 | 0.42 | 0.67 | 0.47 | 0.44 |
| `lambada_openai_mt` | 0.36 | 0.49 | 0.42 | 0.60 | 0.27 | 0.49 | 0.62 | 0.51 | 0.47 | 0.73 |
| `xnli` | 0.53 | 0.29 | 0.60 | **0.82** | 0.36 | 0.40 | 0.44 | 0.56 | 0.33 | 0.60 |
| `paws` | 0.33 | 0.36 | 0.42 | 0.38 | 0.58 | 0.53 | **0.76** | 0.69 | 0.33 | 0.42 |
| `global_piqa_parallel_cloze_spa_latn_spai` | 0.47 | 0.16 | 0.49 | 0.56 | 0.49 | 0.44 | 0.60 | 0.53 | 0.42 | 0.62 |
| `rfgm_include_base_44` | 0.58 | 0.44 | 0.47 | 0.49 | 0.22 | 0.64 | 0.49 | 0.44 | 0.42 | 0.56 |
| `truthfulqa-multi_mc1` | 0.40 | 0.56 | 0.33 | 0.69 | 0.53 | 0.53 | 0.53 | 0.27 | 0.60 | 0.27 |
| `rf_belebele` | 0.38 | 0.33 | 0.42 | 0.33 | 0.51 | 0.47 | 0.49 | 0.62 | 0.71 | 0.36 |
| `global_piqa_parallel_cloze_spa_latn_mexi` | 0.47 | 0.33 | 0.40 | 0.33 | 0.47 | 0.29 | 0.47 | 0.40 | 0.71 | 0.58 |
| `belebele` | 0.47 | 0.49 | 0.24 | 0.67 | 0.38 | 0.20 | 0.42 | 0.64 | 0.56 | 0.38 |
| `global_mmlu_full` | 0.40 | 0.64 | 0.22 | 0.44 | 0.51 | 0.60 | 0.47 | 0.29 | 0.40 | 0.42 |
| `include_base_44` | 0.44 | 0.49 | 0.38 | 0.58 | 0.38 | 0.44 | 0.29 | 0.44 | 0.42 | 0.36 |
| `rf_include_base_44` | 0.38 | 0.42 | 0.16 | 0.53 | 0.47 | 0.56 | 0.33 | 0.42 | 0.42 | 0.44 |
| `global_piqa_parallel_cloze_spa_latn_peru` | 0.42 | 0.49 | 0.51 | 0.38 | 0.49 | 0.22 | 0.18 | 0.36 | 0.51 | 0.56 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_es.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Spanish (es) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "fa (fa) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.74 | 0.62 | **0.82** | **0.82** | **0.85** | **0.83** | **0.86** | **0.77** | 0.74 | **0.88** |
| `multiblimp` | 0.74 | 0.58 | 0.55 | 0.56 | 0.71 | 0.58 | 0.55 | 0.64 | 0.58 | 0.55 |
| `global_mmlu_full` | 0.61 | 0.44 | 0.71 | 0.50 | 0.53 | 0.41 | 0.53 | 0.48 | 0.67 | 0.42 |
| `rf_include_base_44` | 0.55 | 0.48 | 0.58 | 0.48 | 0.53 | 0.52 | 0.58 | 0.50 | 0.62 | 0.47 |
| `include_base_44` | 0.48 | 0.56 | 0.59 | 0.56 | 0.45 | 0.42 | 0.30 | 0.67 | 0.58 | 0.65 |
| `rf_global_mmlu_full` | 0.41 | 0.64 | 0.44 | 0.52 | 0.56 | 0.55 | 0.36 | 0.44 | 0.58 | 0.70 |
| `belebele` | 0.27 | 0.50 | 0.55 | 0.36 | 0.44 | 0.36 | 0.58 | 0.58 | 0.56 | 0.56 |
| `rfgm_include_base_44` | 0.36 | 0.41 | 0.45 | 0.53 | 0.48 | 0.47 | 0.39 | 0.59 | 0.44 | 0.53 |
| `rf_belebele` | 0.26 | 0.38 | 0.39 | 0.53 | 0.68 | 0.42 | 0.52 | 0.53 | 0.47 | 0.42 |
| `global_piqa_parallel_cloze` | 0.36 | 0.15 | 0.48 | 0.26 | 0.48 | 0.50 | 0.27 | 0.32 | 0.55 | 0.45 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_fa.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "fa (fa) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "fi (fi) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.83** | 0.33 | **0.83** | **0.83** | 0.17 | **1.00** | **1.00** | 0.17 | 0.17 | **1.00** |
| `multiblimp` | **0.83** | **0.83** | 0.33 | **0.83** | 0.67 | 0.17 | 0.67 | 0.50 | 0.67 | 0.50 |
| `belebele` | 0.33 | **0.83** | 0.33 | 0.50 | 0.50 | **1.00** | 0.50 | 0.50 | 0.67 | 0.50 |
| `include_base_44` | **1.00** | 0.50 | 0.33 | 0.33 | 0.50 | 0.33 | 0.33 | 0.50 | **0.83** | 0.67 |
| `rf_belebele` | 0.50 | 0.50 | 0.17 | 0.67 | 0.33 | 0.33 | 0.17 | 0.33 | **0.83** | 0.50 |
| `rf_include_base_44` | 0.67 | 0.33 | 0.17 | 0.33 | 0.67 | 0.17 | 0.33 | 0.50 | 0.67 | 0.50 |
| `rfgm_include_base_44` | 0.00 | 0.33 | 0.17 | 0.50 | 0.67 | **0.83** | 0.50 | 0.50 | 0.50 | 0.33 |
| `global_piqa_parallel_cloze` | 0.67 | 0.33 | 0.33 | 0.17 | 0.33 | 0.67 | 0.33 | 0.00 | 0.67 | 0.17 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_fi.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "fi (fi) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "French (fr) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `hellaswag` | 0.64 | 0.67 | 0.58 | 0.47 | **0.89** | **0.84** | **0.78** | **0.82** | **0.80** | **0.84** |
| `bpb` | 0.67 | 0.47 | **0.78** | 0.73 | **0.80** | **0.84** | **0.80** | 0.69 | 0.64 | **0.78** |
| `lambada_openai_mt` | 0.69 | 0.67 | 0.69 | **0.80** | 0.49 | 0.56 | 0.58 | **0.76** | 0.73 | **0.76** |
| `rf_belebele` | 0.58 | 0.73 | 0.62 | 0.47 | 0.64 | 0.71 | 0.64 | 0.71 | 0.56 | 0.67 |
| `arc` | **0.76** | 0.56 | 0.51 | 0.69 | 0.40 | 0.69 | 0.67 | 0.49 | 0.53 | 0.64 |
| `rf_global_mmlu_full` | 0.47 | 0.49 | 0.33 | 0.33 | 0.67 | 0.69 | 0.60 | **0.76** | 0.40 | 0.60 |
| `multiblimp` | 0.56 | 0.49 | 0.58 | 0.36 | 0.47 | 0.56 | 0.47 | 0.49 | 0.64 | 0.58 |
| `belebele` | 0.49 | 0.56 | 0.51 | 0.58 | 0.33 | 0.53 | 0.49 | 0.53 | 0.27 | 0.51 |
| `xnli` | 0.38 | 0.40 | 0.67 | 0.64 | 0.53 | 0.38 | 0.47 | 0.40 | 0.40 | 0.53 |
| `paws` | 0.33 | 0.40 | 0.67 | 0.56 | 0.42 | 0.42 | 0.62 | 0.56 | 0.27 | 0.51 |
| `global_mmlu_full` | 0.24 | 0.58 | 0.40 | 0.58 | 0.58 | 0.53 | 0.53 | 0.33 | 0.42 | 0.51 |
| `include_base_44` | 0.44 | 0.44 | 0.58 | 0.31 | 0.60 | 0.42 | 0.42 | 0.36 | 0.64 | 0.42 |
| `rf_include_base_44` | 0.36 | 0.40 | 0.53 | 0.47 | 0.33 | 0.53 | 0.64 | 0.29 | 0.44 | 0.42 |
| `xwinograd` | 0.29 | 0.53 | 0.38 | 0.62 | 0.53 | 0.60 | 0.29 | 0.44 | 0.49 | 0.22 |
| `global_piqa_parallel_cloze_fra_latn_cana` | 0.27 | 0.38 | 0.33 | 0.53 | **0.76** | 0.38 | 0.40 | 0.31 | 0.51 | 0.42 |
| `rfgm_include_base_44` | 0.53 | 0.36 | 0.31 | 0.58 | 0.16 | 0.40 | 0.33 | 0.47 | 0.53 | 0.40 |
| `global_piqa_parallel_cloze_fra_latn_fran` | 0.40 | 0.36 | 0.42 | 0.22 | 0.71 | 0.40 | 0.44 | 0.24 | 0.47 | 0.31 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_fr.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "French (fr) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "he (he) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.67 | 0.60 | 0.60 | **0.80** | **0.80** | **0.80** | **0.87** | **1.00** | **0.80** | **0.80** |
| `rf_global_mmlu_full` | 0.60 | 0.60 | 0.53 | 0.47 | 0.60 | 0.67 | 0.60 | 0.40 | 0.33 | **0.93** |
| `global_mmlu_full` | 0.27 | 0.60 | 0.60 | 0.60 | 0.40 | 0.67 | 0.67 | 0.60 | 0.47 | 0.73 |
| `multiblimp` | 0.67 | 0.53 | 0.40 | 0.27 | 0.20 | 0.47 | 0.60 | 0.73 | 0.47 | 0.73 |
| `rfgm_include_base_44` | 0.60 | 0.27 | 0.60 | 0.33 | 0.40 | 0.40 | 0.47 | 0.47 | **0.93** | 0.53 |
| `rf_belebele` | 0.47 | 0.73 | 0.53 | 0.60 | 0.27 | 0.07 | 0.60 | **0.80** | 0.47 | 0.40 |
| `include_base_44` | 0.27 | 0.53 | 0.53 | 0.47 | 0.20 | 0.60 | 0.53 | 0.27 | 0.67 | 0.47 |
| `belebele` | 0.33 | 0.47 | 0.33 | 0.33 | 0.47 | 0.40 | 0.47 | 0.53 | 0.20 | 0.60 |
| `rf_include_base_44` | 0.40 | 0.27 | 0.07 | 0.33 | 0.53 | 0.47 | 0.40 | 0.53 | 0.27 | 0.47 |
| `global_piqa_parallel_cloze` | 0.20 | 0.40 | 0.53 | 0.47 | 0.13 | 0.33 | 0.20 | 0.27 | 0.33 | 0.73 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_he.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "he (he) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Hindi (hi) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.68 | 0.54 | 0.68 | **0.89** | **0.86** | 0.71 | 0.71 | **0.79** | 0.57 | **0.79** |
| `rf_belebele_hin_Deva` | 0.43 | 0.32 | 0.50 | 0.25 | 0.64 | 0.64 | 0.61 | 0.64 | **0.89** | 0.57 |
| `xnli` | 0.32 | 0.54 | 0.46 | 0.61 | 0.64 | 0.57 | 0.71 | 0.43 | 0.50 | 0.57 |
| `xstorycloze` | 0.46 | 0.64 | 0.54 | 0.57 | 0.39 | 0.57 | 0.71 | 0.64 | 0.36 | 0.43 |
| `rf_belebele_hin_Latn` | 0.46 | 0.54 | 0.36 | 0.50 | 0.61 | 0.71 | 0.43 | 0.64 | 0.54 | 0.43 |
| `hellaswag` | 0.50 | 0.57 | 0.43 | 0.68 | 0.57 | 0.36 | 0.68 | 0.29 | 0.61 | 0.43 |
| `multiblimp` | 0.50 | 0.68 | 0.68 | 0.43 | 0.61 | 0.57 | 0.36 | 0.64 | 0.36 | 0.29 |
| `rf_global_mmlu_full` | 0.57 | **0.79** | 0.46 | 0.39 | 0.39 | 0.64 | 0.57 | 0.25 | 0.32 | 0.68 |
| `include_base_44` | 0.46 | 0.68 | 0.32 | 0.36 | 0.57 | 0.50 | **0.75** | 0.32 | 0.32 | 0.64 |
| `global_mmlu_full` | **0.86** | 0.32 | 0.64 | 0.46 | 0.25 | 0.64 | 0.46 | 0.32 | 0.43 | 0.39 |
| `rfgm_include_base_44` | 0.50 | 0.50 | 0.71 | 0.36 | 0.43 | 0.43 | 0.43 | **0.75** | 0.39 | 0.21 |
| `belebele_hin_Deva` | 0.57 | 0.32 | 0.68 | 0.43 | 0.61 | 0.50 | 0.39 | 0.32 | 0.50 | 0.39 |
| `arc` | 0.61 | 0.54 | 0.39 | 0.39 | 0.36 | 0.39 | 0.32 | **0.75** | 0.50 | 0.32 |
| `global_piqa_parallel_cloze` | 0.32 | 0.46 | 0.54 | 0.39 | 0.57 | 0.29 | 0.36 | 0.50 | 0.39 | 0.57 |
| `rf_include_base_44` | 0.39 | 0.43 | 0.68 | 0.57 | 0.21 | 0.29 | 0.50 | 0.46 | 0.29 | 0.46 |
| `belebele_hin_Latn` | 0.32 | 0.61 | 0.25 | 0.64 | 0.54 | 0.54 | 0.18 | 0.21 | 0.43 | 0.43 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_hi.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Hindi (hi) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "hu (hu) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `rf_belebele` | **0.87** | **0.80** | 0.53 | 0.47 | **0.80** | 0.53 | 0.33 | 0.60 | 0.53 | 0.47 |
| `bpb` | 0.73 | 0.20 | **0.80** | 0.67 | 0.47 | **0.93** | 0.67 | 0.40 | 0.27 | 0.60 |
| `hellaswag` | 0.53 | 0.47 | 0.40 | 0.40 | 0.27 | 0.60 | 0.60 | 0.53 | 0.67 | 0.73 |
| `rfgm_include_base_44` | 0.40 | 0.60 | 0.73 | 0.33 | 0.27 | 0.40 | 0.47 | 0.67 | 0.53 | 0.40 |
| `arc` | 0.13 | 0.53 | 0.53 | 0.67 | 0.13 | 0.27 | 0.13 | **0.80** | 0.53 | 0.67 |
| `include_base_44` | 0.53 | 0.53 | 0.20 | 0.27 | 0.20 | 0.67 | 0.60 | 0.27 | 0.60 | 0.53 |
| `rf_include_base_44` | 0.47 | 0.33 | 0.73 | 0.47 | 0.60 | 0.27 | 0.20 | 0.20 | 0.60 | 0.47 |
| `belebele` | 0.47 | 0.47 | 0.73 | 0.20 | 0.07 | 0.60 | 0.27 | 0.40 | 0.53 | 0.33 |
| `global_piqa_parallel_cloze` | 0.13 | 0.33 | 0.47 | 0.60 | 0.47 | 0.33 | 0.40 | 0.13 | 0.40 | 0.53 |
| `multiblimp` | 0.53 | 0.40 | 0.53 | 0.40 | 0.40 | 0.60 | 0.00 | 0.00 | 0.27 | 0.40 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_hu.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "hu (hu) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "id (id) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `hellaswag` | **0.75** | 0.54 | **0.79** | **0.75** | 0.68 | **0.79** | **0.89** | 0.68 | **0.79** | **0.82** |
| `bpb` | 0.71 | 0.43 | **0.75** | **0.75** | 0.71 | **0.96** | **0.82** | 0.68 | 0.61 | **0.86** |
| `rf_global_mmlu_full` | 0.50 | 0.46 | 0.71 | 0.54 | 0.61 | 0.57 | 0.54 | 0.54 | 0.50 | **0.75** |
| `xcopa` | 0.61 | 0.54 | **0.82** | 0.64 | 0.39 | 0.71 | 0.32 | 0.50 | 0.61 | 0.50 |
| `arc` | 0.54 | 0.46 | 0.57 | 0.71 | 0.57 | 0.46 | **0.82** | 0.32 | 0.50 | 0.57 |
| `belebele` | 0.46 | 0.50 | 0.68 | 0.61 | 0.46 | 0.29 | 0.50 | 0.71 | 0.54 | 0.57 |
| `xstorycloze` | 0.39 | 0.50 | 0.71 | 0.57 | 0.61 | 0.39 | 0.57 | 0.50 | 0.46 | 0.57 |
| `rfgm_include_base_44` | 0.57 | 0.61 | 0.29 | 0.54 | 0.50 | 0.43 | 0.54 | 0.50 | 0.68 | 0.46 |
| `include_base_44` | 0.39 | 0.50 | 0.68 | 0.46 | 0.43 | 0.36 | 0.36 | 0.46 | 0.71 | 0.57 |
| `global_mmlu_full` | 0.57 | 0.54 | 0.43 | 0.50 | 0.46 | 0.50 | 0.32 | 0.46 | 0.54 | 0.50 |
| `rf_include_base_44` | 0.39 | 0.64 | 0.46 | 0.50 | 0.11 | 0.32 | 0.36 | 0.61 | 0.64 | 0.39 |
| `global_piqa_parallel_cloze` | 0.39 | 0.36 | 0.54 | 0.25 | 0.21 | 0.61 | 0.61 | 0.36 | 0.50 | 0.54 |
| `rf_belebele` | 0.61 | 0.61 | 0.46 | 0.18 | 0.61 | 0.36 | 0.36 | 0.39 | 0.36 | 0.32 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_id.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "id (id) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "it (it) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.71 | 0.51 | **0.82** | **0.76** | **0.76** | **0.89** | **0.87** | 0.64 | 0.71 | **0.89** |
| `hellaswag` | 0.64 | 0.56 | 0.67 | 0.73 | 0.73 | 0.71 | **0.76** | **0.76** | 0.67 | **0.82** |
| `lambada_openai_mt` | **0.76** | 0.64 | 0.62 | 0.73 | 0.58 | 0.73 | 0.60 | 0.73 | 0.67 | 0.67 |
| `multiblimp` | 0.53 | 0.40 | 0.56 | 0.44 | 0.69 | **0.76** | 0.71 | 0.62 | 0.69 | 0.47 |
| `rf_global_mmlu_full` | 0.64 | 0.40 | 0.67 | 0.71 | 0.58 | 0.64 | 0.69 | 0.44 | 0.47 | 0.60 |
| `belebele` | 0.62 | 0.36 | 0.31 | 0.60 | 0.53 | 0.53 | 0.44 | 0.69 | 0.42 | 0.47 |
| `arc` | 0.51 | 0.53 | 0.38 | 0.51 | 0.44 | 0.51 | 0.64 | 0.44 | 0.53 | 0.40 |
| `global_mmlu_full` | 0.36 | 0.67 | 0.49 | 0.58 | 0.31 | 0.47 | 0.29 | 0.42 | 0.47 | 0.69 |
| `rf_belebele` | 0.42 | 0.44 | 0.56 | 0.47 | 0.58 | 0.62 | 0.47 | 0.33 | 0.33 | 0.47 |
| `xcopa` | 0.33 | 0.36 | 0.24 | 0.42 | 0.27 | 0.53 | 0.56 | 0.64 | 0.53 | 0.71 |
| `rfgm_include_base_44` | 0.51 | 0.51 | 0.47 | 0.38 | 0.53 | 0.27 | 0.31 | 0.40 | 0.49 | 0.51 |
| `rf_include_base_44` | 0.33 | 0.27 | 0.69 | 0.44 | 0.47 | 0.27 | 0.44 | 0.42 | 0.51 | 0.33 |
| `include_base_44` | 0.36 | 0.40 | 0.60 | 0.44 | 0.31 | 0.33 | 0.49 | 0.27 | 0.42 | 0.49 |
| `global_piqa_parallel_cloze` | 0.51 | 0.38 | 0.38 | 0.38 | 0.20 | 0.13 | 0.29 | 0.58 | 0.47 | 0.44 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_it.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "it (it) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Japanese (ja) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.67 | 0.59 | 0.75 | 0.75 | **0.86** | **0.92** | **0.86** | **0.85** | **0.80** | **0.91** |
| `rf_global_mmlu_full` | 0.54 | 0.69 | 0.56 | 0.56 | 0.54 | 0.55 | 0.59 | 0.60 | 0.75 | 0.64 |
| `xwinograd` | 0.59 | 0.45 | 0.46 | 0.66 | 0.36 | 0.45 | 0.44 | 0.65 | 0.62 | 0.66 |
| `paws` | 0.49 | 0.68 | 0.41 | 0.52 | 0.49 | 0.69 | 0.60 | 0.35 | 0.37 | 0.57 |
| `global_mmlu_full` | 0.54 | 0.48 | 0.48 | 0.73 | 0.40 | 0.29 | 0.52 | 0.63 | 0.40 | 0.38 |
| `belebele` | 0.35 | 0.52 | 0.49 | 0.55 | 0.49 | 0.46 | 0.24 | 0.60 | 0.52 | 0.56 |
| `include_base_44` | 0.46 | 0.44 | 0.55 | 0.55 | 0.59 | 0.48 | 0.31 | 0.38 | 0.44 | 0.55 |
| `rf_include_base_44` | 0.35 | 0.41 | 0.23 | 0.52 | 0.56 | 0.52 | 0.56 | 0.44 | 0.48 | 0.46 |
| `rfgm_include_base_44` | 0.48 | 0.23 | 0.49 | 0.47 | 0.45 | 0.53 | 0.46 | 0.57 | 0.31 | 0.44 |
| `rf_belebele` | 0.44 | 0.35 | 0.29 | 0.43 | 0.31 | 0.51 | 0.44 | 0.52 | 0.37 | 0.54 |
| `global_piqa_parallel_cloze` | 0.42 | 0.34 | 0.30 | 0.33 | 0.74 | 0.32 | 0.27 | 0.43 | 0.38 | 0.56 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ja.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Japanese (ja) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "ka (ka) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.53 | 0.53 | 0.60 | **0.87** | **1.00** | **0.80** | 0.67 | **0.80** | 0.67 | 0.60 |
| `include_base_44` | 0.47 | 0.47 | 0.67 | 0.47 | **0.87** | 0.60 | 0.27 | 0.60 | 0.40 | 0.67 |
| `belebele` | 0.47 | 0.67 | 0.67 | 0.60 | 0.27 | 0.27 | 0.60 | 0.60 | 0.40 | 0.53 |
| `rf_include_base_44` | 0.33 | 0.40 | 0.53 | 0.67 | 0.33 | 0.33 | 0.33 | 0.73 | 0.60 | 0.47 |
| `rf_belebele` | 0.27 | 0.40 | 0.33 | 0.53 | 0.53 | 0.60 | 0.07 | **0.87** | 0.47 | 0.40 |
| `rfgm_include_base_44` | 0.60 | 0.27 | 0.07 | 0.47 | 0.60 | 0.20 | 0.53 | 0.60 | 0.60 | 0.47 |
| `multiblimp` | 0.13 | 0.27 | 0.53 | 0.33 | 0.47 | 0.47 | 0.47 | 0.27 | 0.53 | 0.33 |
| `global_piqa_parallel_cloze` | 0.27 | 0.60 | 0.20 | 0.47 | 0.07 | 0.47 | 0.13 | 0.27 | 0.47 | 0.53 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ka.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "ka (ka) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Korean (ko) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.68 | 0.54 | **0.75** | **0.82** | **0.86** | **0.86** | **0.86** | **0.79** | 0.71 | **0.86** |
| `rf_global_mmlu_full` | 0.54 | 0.71 | **0.79** | 0.57 | 0.71 | 0.54 | 0.43 | 0.64 | 0.50 | 0.71 |
| `include_base_44` | 0.46 | 0.71 | 0.39 | 0.39 | 0.39 | 0.54 | **0.79** | 0.50 | 0.46 | 0.64 |
| `rf_belebele` | 0.46 | 0.32 | 0.64 | 0.64 | 0.57 | 0.39 | 0.36 | 0.68 | 0.36 | 0.64 |
| `rfgm_include_base_44` | 0.54 | 0.64 | 0.57 | 0.39 | 0.36 | 0.43 | 0.64 | 0.64 | 0.43 | 0.39 |
| `paws` | 0.61 | 0.57 | 0.39 | 0.64 | 0.57 | 0.43 | 0.54 | 0.25 | 0.57 | 0.43 |
| `global_piqa_parallel_cloze` | 0.57 | 0.39 | 0.43 | 0.29 | 0.46 | **0.75** | 0.39 | 0.50 | 0.61 | 0.57 |
| `belebele` | 0.25 | 0.36 | 0.61 | **0.75** | 0.61 | 0.25 | 0.43 | 0.54 | 0.50 | 0.54 |
| `global_mmlu_full` | 0.57 | 0.29 | 0.61 | 0.57 | 0.29 | 0.39 | 0.50 | 0.61 | 0.29 | 0.46 |
| `rf_include_base_44` | 0.54 | 0.29 | 0.50 | 0.36 | 0.21 | 0.50 | 0.14 | 0.29 | 0.68 | 0.39 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ko.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Korean (ko) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "ml (ml) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.53 | 0.60 | 0.60 | **0.80** | **0.93** | **0.93** | 0.73 | **0.87** | **0.80** | **0.80** |
| `belebele` | **0.87** | 0.40 | 0.53 | 0.73 | 0.27 | 0.40 | **0.87** | 0.73 | 0.20 | 0.27 |
| `rfgm_include_base_44` | 0.53 | 0.47 | 0.47 | 0.27 | 0.73 | 0.73 | 0.53 | 0.47 | 0.27 | 0.60 |
| `hellaswag` | 0.73 | 0.33 | 0.67 | 0.33 | 0.40 | 0.40 | 0.53 | 0.53 | 0.47 | 0.40 |
| `include_base_44` | 0.67 | 0.53 | 0.27 | 0.40 | 0.47 | 0.20 | 0.33 | 0.33 | 0.67 | 0.60 |
| `arc` | 0.20 | 0.33 | 0.20 | 0.27 | 0.53 | 0.67 | 0.60 | 0.60 | 0.27 | 0.67 |
| `rf_include_base_44` | 0.33 | 0.53 | 0.47 | 0.53 | 0.07 | 0.20 | 0.40 | **0.80** | 0.40 | 0.47 |
| `rf_belebele` | 0.13 | 0.33 | **0.80** | 0.47 | 0.40 | 0.27 | 0.60 | 0.53 | 0.27 | 0.33 |
| `global_piqa_parallel_cloze` | 0.60 | 0.07 | 0.13 | 0.20 | 0.33 | 0.13 | 0.33 | 0.67 | 0.47 | 0.40 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ml.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "ml (ml) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "nl (nl) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `hellaswag` | 0.57 | 0.39 | 0.61 | 0.64 | **0.82** | **0.82** | **0.93** | 0.64 | **0.75** | **0.89** |
| `bpb` | 0.71 | 0.43 | **0.79** | **0.75** | 0.64 | **0.86** | **0.82** | 0.57 | 0.54 | **0.75** |
| `multiblimp` | 0.57 | 0.46 | 0.50 | 0.57 | 0.68 | 0.71 | 0.57 | 0.68 | 0.71 | 0.57 |
| `belebele` | 0.61 | 0.46 | 0.46 | 0.57 | 0.57 | 0.50 | 0.68 | 0.57 | **0.86** | 0.61 |
| `arc` | 0.43 | 0.54 | 0.39 | 0.64 | 0.43 | **0.75** | 0.46 | 0.54 | 0.39 | 0.54 |
| `rf_global_mmlu_full` | 0.50 | 0.61 | 0.25 | 0.46 | 0.25 | 0.61 | 0.54 | 0.36 | 0.36 | 0.71 |
| `global_mmlu_full` | 0.25 | 0.64 | 0.36 | 0.43 | 0.46 | 0.43 | 0.46 | 0.43 | 0.43 | 0.64 |
| `rf_belebele` | 0.39 | 0.39 | **0.75** | 0.29 | 0.46 | 0.43 | 0.54 | 0.61 | 0.32 | 0.25 |
| `global_piqa_parallel_cloze` | 0.50 | 0.39 | 0.46 | 0.43 | 0.43 | 0.43 | 0.29 | 0.29 | 0.43 | 0.46 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_nl.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "nl (nl) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "no (no) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **1.00** | 0.17 | **1.00** | **0.83** | 0.17 | **1.00** | **0.83** | 0.17 | 0.33 | **0.83** |
| `global_piqa_parallel_cloze` | 0.33 | 0.33 | 0.33 | 0.33 | 0.67 | 0.50 | 0.67 | 0.67 | **0.83** | **0.83** |
| `belebele` | 0.33 | 0.00 | 0.00 | 0.50 | 0.67 | 0.67 | **0.83** | **1.00** | 0.50 | 0.50 |
| `rf_belebele` | 0.33 | 0.33 | 0.17 | 0.50 | 0.17 | 0.50 | 0.50 | 0.67 | 0.50 | 0.67 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_no.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "no (no) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "pl (pl) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.68 | 0.39 | 0.71 | 0.64 | 0.64 | **0.89** | **0.82** | 0.61 | 0.54 | **0.79** |
| `rfgm_include_base_44` | 0.68 | 0.54 | **0.86** | 0.54 | 0.64 | 0.68 | 0.71 | 0.50 | 0.71 | 0.61 |
| `rf_global_mmlu_full` | 0.68 | **0.75** | 0.32 | 0.54 | 0.57 | 0.57 | 0.43 | 0.36 | 0.50 | 0.43 |
| `multiblimp` | 0.46 | 0.43 | 0.46 | 0.57 | 0.43 | 0.64 | 0.46 | 0.43 | 0.61 | 0.46 |
| `include_base_44` | 0.46 | 0.50 | 0.57 | 0.14 | 0.46 | 0.57 | 0.54 | 0.61 | 0.50 | 0.50 |
| `rf_belebele` | 0.43 | 0.46 | 0.64 | 0.50 | 0.36 | 0.36 | 0.32 | 0.64 | 0.57 | 0.57 |
| `belebele` | 0.36 | 0.43 | 0.39 | 0.46 | 0.43 | 0.32 | 0.32 | **0.79** | 0.61 | 0.64 |
| `global_mmlu_full` | 0.29 | 0.46 | 0.64 | 0.36 | 0.39 | 0.43 | 0.57 | 0.54 | 0.54 | 0.43 |
| `rf_include_base_44` | 0.43 | 0.32 | 0.57 | 0.57 | 0.29 | 0.64 | 0.57 | 0.25 | 0.21 | **0.79** |
| `global_piqa_parallel_cloze` | 0.46 | 0.50 | 0.39 | 0.39 | 0.50 | 0.21 | 0.21 | 0.32 | 0.25 | 0.71 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_pl.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "pl (pl) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Portuguese (pt) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.79** | 0.43 | **0.86** | **0.75** | 0.64 | **0.86** | **0.89** | 0.50 | 0.61 | **0.82** |
| `hellaswag` | 0.57 | 0.61 | 0.61 | 0.61 | 0.71 | 0.71 | 0.64 | **0.79** | 0.71 | **0.93** |
| `multiblimp` | 0.71 | 0.50 | 0.61 | 0.64 | 0.50 | 0.61 | 0.57 | 0.68 | **0.75** | 0.61 |
| `rf_belebele` | 0.50 | 0.50 | 0.61 | **0.75** | 0.71 | 0.64 | 0.29 | 0.61 | 0.25 | 0.50 |
| `arc` | 0.61 | 0.61 | 0.57 | 0.50 | 0.64 | 0.68 | 0.43 | 0.50 | 0.36 | 0.46 |
| `rf_global_mmlu_full` | 0.57 | 0.43 | 0.43 | 0.43 | 0.36 | 0.36 | 0.64 | 0.64 | 0.43 | 0.64 |
| `belebele` | 0.50 | 0.36 | 0.39 | 0.39 | 0.54 | 0.46 | 0.36 | 0.61 | 0.54 | **0.79** |
| `rf_include_base_44` | 0.46 | 0.64 | 0.36 | 0.29 | 0.64 | 0.64 | 0.39 | 0.36 | 0.61 | 0.32 |
| `rfgm_include_base_44` | 0.32 | 0.61 | 0.14 | 0.61 | 0.54 | 0.36 | 0.50 | 0.36 | **0.79** | 0.36 |
| `global_piqa_parallel_cloze_por_latn_port` | 0.61 | 0.39 | 0.50 | 0.57 | 0.57 | 0.46 | 0.46 | 0.32 | 0.25 | 0.36 |
| `global_piqa_parallel_cloze_por_latn_braz` | 0.68 | 0.29 | 0.50 | 0.46 | 0.25 | 0.57 | 0.46 | 0.50 | 0.32 | 0.43 |
| `global_mmlu_full` | 0.36 | 0.64 | 0.46 | 0.43 | 0.50 | 0.39 | 0.57 | 0.25 | 0.21 | 0.61 |
| `include_base_44` | 0.54 | 0.54 | 0.46 | 0.29 | 0.29 | 0.50 | 0.54 | 0.54 | 0.29 | 0.43 |
| `xwinograd` | 0.32 | 0.61 | 0.43 | 0.61 | 0.29 | 0.32 | 0.39 | 0.57 | 0.43 | 0.39 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_pt.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Portuguese (pt) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "ro (ro) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `hellaswag` | 0.67 | 0.67 | 0.67 | 0.73 | 0.53 | 0.60 | 0.67 | 0.60 | 0.60 | **0.87** |
| `bpb` | 0.73 | 0.27 | **0.93** | 0.73 | 0.53 | **0.80** | **0.87** | 0.33 | 0.40 | **0.80** |
| `rf_belebele` | 0.67 | 0.47 | 0.73 | 0.40 | 0.27 | **0.93** | 0.60 | 0.33 | 0.40 | 0.67 |
| `rf_global_mmlu_full` | 0.20 | **0.80** | **0.93** | 0.40 | 0.27 | 0.27 | 0.67 | **0.87** | 0.33 | 0.33 |
| `belebele` | 0.73 | 0.20 | 0.53 | 0.40 | 0.47 | 0.40 | 0.53 | 0.47 | 0.67 | 0.47 |
| `arc` | 0.33 | 0.27 | 0.47 | 0.47 | 0.33 | 0.47 | **0.80** | 0.53 | 0.40 | 0.33 |
| `global_mmlu_full` | 0.53 | 0.53 | 0.20 | 0.40 | 0.20 | 0.40 | 0.60 | 0.40 | 0.47 | 0.40 |
| `multiblimp` | 0.67 | 0.20 | 0.33 | 0.53 | 0.27 | 0.40 | 0.33 | 0.33 | 0.60 | 0.40 |
| `global_piqa_parallel_cloze` | 0.40 | 0.13 | 0.67 | 0.53 | 0.33 | 0.40 | 0.27 | 0.13 | 0.53 | 0.47 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ro.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "ro (ro) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Russian (ru) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.75** | 0.68 | **0.82** | **0.81** | **0.87** | **0.92** | **0.93** | **0.85** | **0.84** | **0.94** |
| `hellaswag` | 0.69 | 0.59 | 0.69 | 0.64 | 0.73 | **0.91** | **0.82** | **0.78** | **0.81** | **0.85** |
| `xstorycloze` | **0.78** | 0.68 | 0.59 | 0.64 | 0.63 | 0.61 | 0.67 | 0.72 | **0.76** | 0.66 |
| `multiblimp` | 0.72 | 0.71 | 0.59 | 0.60 | **0.86** | 0.60 | 0.69 | 0.68 | 0.62 | 0.57 |
| `rf_global_mmlu_full` | 0.67 | 0.49 | 0.46 | 0.49 | 0.68 | 0.65 | 0.69 | **0.80** | 0.68 | **0.81** |
| `arc` | 0.66 | 0.53 | 0.55 | 0.69 | 0.58 | 0.52 | 0.57 | **0.76** | 0.56 | 0.66 |
| `rf_belebele` | 0.60 | 0.56 | 0.49 | 0.68 | 0.59 | 0.70 | 0.60 | 0.61 | 0.53 | 0.68 |
| `xnli` | 0.47 | 0.54 | 0.53 | 0.57 | 0.42 | 0.53 | 0.33 | 0.62 | **0.76** | 0.57 |
| `rf_include_base_44` | 0.35 | 0.36 | 0.37 | 0.41 | 0.64 | 0.51 | 0.62 | 0.56 | 0.69 | 0.55 |
| `rfgm_include_base_44` | 0.43 | 0.24 | 0.64 | 0.35 | 0.65 | 0.53 | 0.58 | 0.48 | 0.58 | 0.55 |
| `belebele` | 0.41 | 0.58 | 0.56 | 0.53 | 0.47 | 0.50 | 0.42 | 0.60 | 0.60 | 0.34 |
| `include_base_44` | 0.45 | 0.37 | 0.51 | 0.54 | 0.59 | 0.62 | 0.43 | 0.46 | 0.47 | 0.37 |
| `global_mmlu_full` | 0.50 | 0.44 | 0.42 | 0.62 | 0.46 | 0.40 | 0.48 | 0.45 | 0.46 | 0.49 |
| `xwinograd` | 0.37 | 0.37 | 0.56 | 0.38 | 0.42 | 0.56 | 0.56 | 0.34 | 0.53 | 0.60 |
| `global_piqa_parallel_cloze` | 0.33 | 0.30 | 0.46 | 0.47 | 0.38 | 0.51 | 0.37 | 0.51 | 0.47 | 0.57 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ru.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Russian (ru) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "sv (sv) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.67 | 0.47 | 0.67 | 0.67 | 0.67 | **1.00** | **0.87** | 0.67 | 0.53 | **0.87** |
| `global_piqa_parallel_cloze` | 0.67 | 0.67 | 0.60 | 0.60 | 0.67 | 0.67 | 0.60 | 0.53 | 0.47 | 0.67 |
| `hellaswag` | 0.73 | 0.40 | 0.47 | 0.60 | 0.67 | 0.60 | 0.60 | 0.53 | 0.53 | 0.60 |
| `rf_global_mmlu_full` | 0.73 | 0.53 | 0.20 | 0.67 | 0.67 | 0.33 | 0.67 | 0.53 | 0.73 | 0.53 |
| `rf_belebele` | 0.20 | 0.53 | 0.73 | **0.87** | 0.40 | 0.47 | 0.33 | 0.27 | 0.67 | 0.60 |
| `arc` | 0.53 | 0.20 | 0.53 | 0.40 | 0.27 | 0.40 | 0.47 | 0.60 | **0.80** | 0.67 |
| `global_mmlu_full` | 0.33 | 0.53 | 0.53 | 0.33 | 0.67 | 0.27 | 0.47 | 0.47 | 0.53 | 0.40 |
| `belebele` | 0.53 | 0.20 | 0.33 | 0.53 | 0.53 | 0.47 | 0.53 | 0.53 | 0.20 | 0.53 |
| `multiblimp` | 0.27 | 0.47 | 0.27 | 0.40 | 0.67 | 0.20 | 0.40 | 0.40 | 0.67 | 0.67 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_sv.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "sv (sv) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "ta (ta) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.73 | 0.60 | 0.67 | **0.93** | **0.87** | **0.80** | **0.80** | **0.93** | 0.67 | 0.73 |
| `hellaswag` | 0.60 | 0.27 | 0.33 | **0.93** | 0.53 | 0.73 | 0.53 | 0.73 | 0.20 | 0.27 |
| `belebele` | 0.13 | 0.40 | 0.67 | 0.47 | 0.73 | 0.33 | 0.40 | 0.47 | 0.67 | 0.53 |
| `multiblimp` | 0.40 | 0.53 | 0.47 | 0.60 | 0.53 | 0.47 | 0.33 | 0.60 | 0.33 | 0.53 |
| `rfgm_include_base_44` | 0.40 | 0.53 | 0.40 | 0.67 | 0.27 | 0.33 | 0.20 | 0.67 | 0.73 | 0.53 |
| `arc` | 0.53 | 0.60 | 0.47 | 0.53 | 0.53 | 0.33 | 0.47 | 0.13 | 0.73 | 0.13 |
| `rf_belebele` | 0.47 | 0.47 | 0.33 | 0.53 | 0.73 | 0.40 | 0.40 | 0.27 | 0.20 | 0.60 |
| `include_base_44` | 0.47 | 0.60 | 0.47 | 0.20 | 0.53 | 0.47 | 0.40 | 0.13 | 0.13 | 0.67 |
| `xcopa` | 0.20 | 0.27 | 0.47 | 0.33 | 0.73 | 0.27 | 0.40 | 0.33 | 0.67 | 0.33 |
| `rf_include_base_44` | 0.13 | 0.40 | 0.33 | 0.33 | 0.53 | 0.40 | 0.60 | 0.27 | **0.80** | 0.07 |
| `global_piqa_parallel_cloze` | 0.20 | 0.40 | 0.40 | 0.27 | 0.40 | 0.53 | 0.47 | 0.20 | 0.40 | 0.40 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ta.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "ta (ta) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Thai (th) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.64 | 0.60 | **0.80** | **0.78** | **0.78** | **0.84** | **0.87** | **0.80** | **0.78** | **0.93** |
| `belebele` | 0.42 | 0.49 | 0.47 | 0.40 | 0.47 | 0.47 | 0.69 | 0.67 | 0.40 | 0.44 |
| `xnli` | 0.33 | 0.67 | 0.40 | 0.38 | 0.49 | 0.47 | 0.44 | 0.47 | 0.58 | 0.62 |
| `rf_belebele` | 0.42 | 0.44 | 0.38 | 0.64 | 0.42 | 0.51 | 0.49 | 0.36 | 0.49 | 0.33 |
| `xcopa` | 0.29 | 0.69 | 0.38 | 0.29 | 0.31 | 0.29 | 0.42 | 0.60 | 0.38 | 0.42 |
| `global_piqa_parallel_cloze` | 0.47 | 0.51 | 0.38 | 0.33 | 0.33 | 0.40 | 0.44 | 0.33 | 0.38 | 0.33 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_th.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Thai (th) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Turkish (tr) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `belebele` | 0.73 | 0.40 | 0.60 | **0.80** | 0.60 | 0.73 | **0.80** | 0.67 | 0.47 | **0.80** |
| `xnli` | 0.47 | **0.80** | 0.60 | 0.60 | 0.47 | 0.47 | 0.67 | 0.73 | 0.67 | 0.67 |
| `bpb` | 0.73 | 0.20 | **0.80** | 0.73 | 0.47 | **0.93** | 0.73 | 0.40 | 0.33 | 0.67 |
| `global_mmlu_full` | 0.40 | 0.67 | 0.67 | 0.60 | 0.20 | 0.33 | 0.27 | **0.87** | 0.67 | 0.67 |
| `include_base_44` | 0.40 | 0.60 | 0.73 | 0.33 | 0.40 | 0.53 | **0.80** | 0.60 | 0.47 | 0.47 |
| `xcopa` | 0.67 | 0.40 | 0.47 | 0.53 | 0.20 | 0.27 | 0.60 | **0.80** | 0.53 | 0.60 |
| `rf_belebele` | 0.53 | 0.40 | 0.40 | 0.60 | 0.47 | 0.27 | 0.60 | 0.53 | 0.40 | 0.27 |
| `rf_global_mmlu_full` | 0.40 | 0.40 | 0.33 | 0.67 | 0.47 | 0.40 | 0.33 | 0.40 | 0.33 | 0.53 |
| `rf_include_base_44` | 0.53 | 0.53 | 0.27 | 0.27 | 0.67 | 0.40 | 0.33 | 0.33 | 0.40 | 0.40 |
| `rfgm_include_base_44` | 0.27 | 0.40 | 0.47 | 0.07 | 0.40 | 0.40 | 0.73 | 0.53 | 0.47 | 0.40 |
| `multiblimp` | 0.60 | 0.13 | 0.33 | 0.33 | 0.33 | 0.07 | 0.67 | 0.60 | 0.53 | 0.33 |
| `global_piqa_parallel_cloze` | 0.33 | 0.33 | 0.33 | 0.60 | 0.13 | 0.73 | 0.47 | 0.33 | 0.00 | 0.53 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_tr.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Turkish (tr) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Ukrainian (uk) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.87** | 0.27 | **0.93** | **0.80** | 0.40 | **0.80** | **0.93** | 0.20 | 0.47 | 0.73 |
| `hellaswag` | **0.93** | 0.53 | 0.47 | 0.67 | 0.60 | 0.53 | 0.60 | 0.67 | 0.60 | **0.80** |
| `multiblimp` | **0.80** | 0.47 | 0.53 | 0.67 | 0.40 | 0.33 | 0.47 | 0.67 | 0.40 | 0.73 |
| `belebele` | 0.67 | 0.67 | **0.80** | 0.20 | 0.60 | 0.73 | 0.27 | 0.73 | 0.53 | 0.27 |
| `include_base_44` | 0.40 | **0.80** | 0.53 | 0.53 | 0.33 | 0.33 | 0.73 | 0.67 | 0.60 | 0.47 |
| `global_mmlu_full` | 0.73 | 0.33 | **0.87** | 0.47 | 0.20 | 0.73 | 0.47 | 0.47 | 0.33 | 0.47 |
| `rf_include_base_44` | 0.60 | 0.73 | 0.33 | 0.60 | 0.40 | 0.40 | 0.67 | 0.47 | 0.53 | 0.13 |
| `rfgm_include_base_44` | 0.27 | 0.47 | 0.47 | 0.73 | 0.40 | 0.40 | 0.33 | 0.73 | 0.47 | 0.60 |
| `rf_global_mmlu_full` | 0.53 | 0.33 | 0.47 | 0.73 | 0.40 | 0.40 | 0.53 | 0.33 | 0.33 | 0.73 |
| `arc` | 0.67 | 0.53 | 0.33 | 0.47 | 0.33 | 0.40 | 0.73 | 0.53 | 0.27 | 0.33 |
| `global_piqa_parallel_cloze` | 0.20 | 0.47 | 0.47 | **0.87** | 0.13 | 0.33 | 0.20 | **0.80** | 0.60 | 0.53 |
| `rf_belebele` | 0.40 | 0.20 | 0.33 | 0.53 | 0.40 | 0.53 | 0.53 | 0.33 | 0.67 | 0.07 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_uk.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Ukrainian (uk) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Vietnamese (vi) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.64 | 0.46 | **0.89** | **0.75** | **0.82** | **0.75** | **0.82** | 0.57 | 0.71 | **0.86** |
| `xcopa` | 0.57 | **0.75** | 0.54 | 0.71 | 0.36 | 0.54 | 0.68 | 0.36 | 0.54 | 0.71 |
| `hellaswag` | 0.68 | 0.32 | 0.64 | 0.64 | 0.21 | 0.61 | 0.64 | 0.54 | 0.46 | **0.86** |
| `xnli` | 0.68 | 0.25 | 0.46 | 0.36 | 0.54 | 0.57 | 0.50 | 0.50 | 0.54 | 0.46 |
| `global_mmlu_full` | 0.61 | 0.43 | 0.64 | 0.57 | 0.46 | 0.32 | 0.39 | 0.43 | 0.50 | 0.43 |
| `rf_include_base_44` | 0.50 | 0.61 | 0.61 | 0.46 | 0.25 | 0.39 | 0.29 | 0.43 | 0.57 | 0.61 |
| `rf_belebele` | 0.39 | 0.64 | 0.64 | 0.50 | 0.36 | 0.54 | 0.39 | 0.50 | 0.36 | 0.36 |
| `belebele` | 0.21 | 0.57 | 0.39 | 0.68 | 0.36 | **0.79** | 0.36 | 0.39 | 0.32 | 0.43 |
| `include_base_44` | 0.57 | 0.29 | 0.54 | 0.43 | 0.50 | 0.46 | 0.50 | 0.14 | 0.61 | 0.39 |
| `rf_global_mmlu_full` | 0.25 | 0.54 | 0.57 | 0.50 | 0.39 | 0.25 | 0.29 | 0.36 | 0.36 | **0.82** |
| `arc` | 0.36 | 0.36 | 0.50 | 0.61 | 0.46 | 0.54 | 0.25 | 0.32 | 0.29 | 0.39 |
| `rfgm_include_base_44` | 0.54 | 0.32 | 0.50 | 0.32 | 0.54 | 0.50 | 0.32 | 0.54 | 0.21 | 0.29 |
| `global_piqa_parallel_cloze` | 0.57 | 0.29 | 0.36 | 0.29 | 0.25 | 0.39 | 0.32 | 0.21 | 0.21 | 0.46 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_vi.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Vietnamese (vi) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Mandarin Chinese (zh) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | 0.68 | 0.56 | 0.75 | 0.74 | **0.86** | **0.93** | **0.81** | **0.81** | **0.76** | **0.84** |
| `xstorycloze` | 0.66 | 0.53 | 0.68 | 0.67 | 0.58 | 0.68 | 0.63 | 0.62 | 0.58 | 0.57 |
| `rf_include_base_44` | 0.54 | 0.41 | 0.56 | 0.56 | 0.55 | 0.69 | 0.56 | 0.59 | 0.62 | 0.60 |
| `rf_global_mmlu_full` | 0.42 | 0.55 | 0.46 | 0.40 | 0.48 | 0.63 | **0.76** | 0.70 | 0.53 | 0.69 |
| `rfgm_include_base_44` | 0.41 | 0.51 | 0.62 | 0.49 | 0.42 | 0.54 | 0.51 | 0.58 | 0.64 | 0.54 |
| `global_mmlu_full` | 0.48 | 0.54 | 0.52 | 0.64 | 0.53 | 0.42 | 0.52 | 0.47 | 0.51 | 0.55 |
| `rf_belebele_zho_Hant` | 0.51 | 0.48 | 0.54 | 0.47 | 0.57 | 0.49 | 0.52 | 0.56 | 0.49 | 0.51 |
| `arc` | 0.49 | 0.55 | 0.54 | 0.41 | 0.52 | 0.65 | 0.37 | 0.46 | 0.58 | 0.49 |
| `belebele_zho_Hant` | 0.58 | 0.45 | 0.54 | 0.46 | 0.62 | 0.44 | 0.52 | 0.56 | 0.45 | 0.37 |
| `rf_belebele_zho_Hans` | 0.52 | 0.74 | 0.52 | 0.35 | 0.51 | 0.47 | 0.31 | 0.66 | 0.41 | 0.47 |
| `paws` | 0.46 | 0.53 | 0.45 | 0.53 | 0.64 | 0.53 | 0.41 | 0.54 | 0.41 | 0.38 |
| `belebele_zho_Hans` | 0.52 | 0.62 | 0.49 | 0.42 | 0.37 | 0.43 | 0.36 | 0.58 | 0.49 | 0.58 |
| `xnli` | 0.37 | 0.48 | 0.73 | 0.34 | 0.52 | 0.47 | 0.37 | 0.59 | 0.41 | 0.49 |
| `include_base_44` | 0.47 | 0.41 | 0.70 | 0.23 | 0.52 | 0.53 | 0.43 | 0.38 | 0.40 | 0.38 |
| `xwinograd` | 0.34 | 0.44 | 0.32 | 0.30 | 0.32 | 0.49 | 0.46 | 0.69 | 0.53 | 0.54 |
| `xcopa` | 0.49 | 0.38 | 0.48 | 0.53 | 0.31 | 0.48 | 0.41 | 0.31 | 0.47 | 0.47 |
| `global_piqa_parallel_cloze_cmn_hant` | 0.47 | 0.40 | 0.46 | 0.38 | 0.25 | 0.43 | 0.53 | 0.34 | 0.40 | 0.45 |
| `global_piqa_parallel_cloze_cmn_hans` | 0.43 | 0.56 | 0.53 | 0.33 | 0.37 | 0.38 | 0.37 | 0.33 | 0.42 | 0.27 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_zh.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "Mandarin Chinese (zh) · the table before, as a heatmap"
---

<!-- END generated signal slides -->
