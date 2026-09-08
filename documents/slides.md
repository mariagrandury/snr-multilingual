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

Seven, one directory each

---
title: Research questions
subtitle: "RQ0–RQ5 ask which benchmarks are reliable; RQ6 asks which model sizes are"
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

One entry point: `cd src/signal-and-noise && bash run_all_predictivity.sh`

<!--
Source of truth is one file: the wide per-checkpoint ladder_report.csv published to
msnr-data/ladder-report. Every number in this deck regenerates from it.
-->

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
height: 74vh
title: Experimental Setup
subtitle: The planned grid
---

---
title: Experimental Setup — What exists today
subtitle: "60 of 89 cells finished; the top of the ladder is missing"
---

| | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| **complete cells** | 10 | 17 | 14 | 17 | 2 | **0** |

- **L ∈ {1, 2, 8, 15, 30, 50}** — **L = 100 has no finished cell at any size**
- 1B is finished at **L8 and L50 only**; 1.7B has not finished anywhere
- After the loader drops diverged and unfinished runs: **51 models** enter the analysis, 48 of them also on-trend
- Seed replicates: 175M and 600M at L ∈ {1, 2, 50} — six ×3 cells, the whole noise estimate

<!--
Counts from ladder_report.csv (msnr-data/ladder-report, branch data/ladder-report).
89 cells in the report, 60 complete, 51 complete-and-not-diverged, 48 also on-trend.
The reference rung the whole predictivity question needs is the one we do not have.
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
groups.main is kept for the historical decks. groups.trained is the 49 FineWeb-2
tags of FW_L50 in src/pretrain/data/language_sets_schemeA.json, plus English,
which comes from DCLM rather than FineWeb-2. Add the L100 list the same way once
the distribution is decided.
-->
---
layout: figure
image: /ladder/grid_status.png
fit: contain
height: 66vh
title: What exists today
subtitle: "Same numbers as the table before, as a picture"
---

<!--
Improved version of the previous slide. Blue is a finished cell, grey is the 90M
rung that diverged, pale blue is started but not done, near white is not started.
Two things to point at: the bottom row (100 languages) is empty at every size,
and the two right columns are nearly empty. 1B is finished only at 8 and 50
languages. The 90M square at L2 is blue because one shallow cell there missed the
divergence test by 0.016 nats, which is the next slide but one.
-->
---
layout: section
---

# Findings

What the ladder says so far

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
layout: bullets
title: Above 90M the ladder behaves
subtitle: "Plain version of the slide before"
icon: "📐"
---

- Loss falls with size at every language count, with no exceptions
- The slope of that fall is the same everywhere. It sits between 0.16 and 0.20 whether the model sees 1 language or 50
- So adding languages shifts the whole curve up. It does not change how the model scales
- Each fit leaves out the smallest rung, then tries to predict it. The healthy rungs land within 0.07 nats
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
---
layout: figure
image: /ladder/optimizer_timescale.png
fit: contain
height: 52vh
title: Why the 90M rung broke
subtitle: "The optimizer averages over 10,000 steps. The 90M run is 4,500 steps long."
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
- It anchors **101 of RQ6's 303** scaling fits — every fit starting below 175M, all `shallow / scheme A`
- `run__off_trend` is already published in the ladder report; the loader ignores it

<!--
Verified on the real report: run__diverged = 0, run__off_trend = 1, run__resid = +1.291.
Fix is one line in snr/download/ladder.py (gate on off_trend alongside diverged) but it
changes what enters every pool, so it is a call for the team, not a patch.
-->

---
layout: bullets
title: One 90M cell got through
subtitle: "Plain version of the slide before"
icon: "⚠️"
---

- A run counts as diverged if it ends more than 0.25 nats above its own best
- `90M-L2-shallow` ends 0.234 above its best. It misses the line by 0.016
- So it is the only 90M model in any of our tables, and it sits 1.29 nats off its own scaling fit
- Every number we report for 90M is that single run
- It also anchors 101 of the 303 scaling fits in the last research question
- The ladder report already publishes an `off_trend` flag. Our loader does not read it. One line fixes this, but it changes every number, so it is a decision for the team
---
layout: figure
image: /ladder/bpb_vs_languages.png
fit: contain
height: 62vh
title: Finding 4 — More languages is nearly free for English
subtitle: "Held-out bits-per-byte, deep / scheme A / seed 1904"
---

---
layout: bullets
title: Finding 4 — More languages is nearly free for English
subtitle: "The multilingual tax is paid once, at the first extra language"
icon: "🌍"
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

- Seed noise: **0.021 nats** on final loss, **0.011 bits/byte** on macro BPB (six ×3 cells)
- **Language count**: 12.0× the seed noise — a real axis
- **Data scheme** (A vs B): 2.2× on loss, 1.2× on macro BPB — marginal
- **Model depth** (deep vs shallow): **1.0×** on the aggregate loss — the same model measured twice
- Per *individual* task the depth effect is larger — median **1.55×**, 39 % of 556 cells above 2× — so it separates *somewhere*, just not on the headline metric
- Depth is **half the grid**, and it buys a per-task effect we would have to hunt for

<!--
Matched pairs on healthy complete cells only: depth n=9, scheme n=7. Language axis is the
across-L range at fixed size (n=18 cells). This is the transformation table of
ladder_report.md, resolved against a proper seed standard deviation.
-->

---
layout: bullets
title: Two of our three axes are inside the noise
subtitle: "Plain version of the slide before"
icon: "📉"
---

- Train the same cell with three different seeds and the final loss moves by 0.021 nats. That is our noise floor
- Changing the number of languages moves it by 12 times that. This axis is real
- Changing the data scheme moves it by 2.2 times. This axis is marginal
- Changing the depth moves it by 1.0 times. On the headline metric, deep and shallow are the same model measured twice
- Task by task the depth effect is bigger, 1.55 times the noise, and above 2 times on 39 percent of cells. So depth does something, just not to the number we optimise
- Depth is half of the grid
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

Of the **324** tasks that have a chance level, **94 clear it at any size**. Last night's evals added nine.

| answer options | chance | clears chance | at 1B |
| :---: | :---: | :---: | :---: |
| **2** (completion, minimal pair) | 0.50 | **58 / 129** | 53 / 129 |
| **3** (XNLI) | 0.33 | **12 / 15** | 11 / 15 |
| **4** (knowledge MCQA) | 0.25 | **24 / 180** | 23 / 180 |

- Clearing chance at 1B: `multiblimp` (34), `hellaswag` (20), `xnli` (11), `xwinograd` (6), `xcopa` (5), `xstorycloze` (5), `paws` (3), `arc` (2), `include_base_44` (1)
- **Never clearing chance anywhere**: `belebele`, `global_mmlu_full`, `global_piqa_parallel_cloze`, `truthfulqa-multi`
- Strongly an **answer-count effect**: 45 % of 2-option tasks clear chance, **13 % of 4-option** ones do
- On the 36-model sweep the same families cleared chance for external 270M–70B models (122 of 124) — a capability floor of the small rungs, not a property of the benchmarks

<!--
above_random.py --only predictivity on the 8 September report: 431 tasks, 324 with a chance
level. The other 107 are per-language BPB and generative tasks, which have none and are
never gated. Per bucket: 90M 3/22, 175M 41/324, 350M 59/324, 600M 82/324, 1B 87/324.
`paws` and `global_piqa_nonparallel_cloze` crossed the line for the first time last night,
which is why the never-clearing list is four families rather than six.
-->

---
layout: figure
image: /ladder/first_clearing_size.png
fit: contain
height: 74vh
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

## In 89 of the 95 validation languages, the most reliable measurement is **bits-per-byte**, not any benchmark

<!--
top_benchmarks_per_language.csv on the canonical pool, 8 September run: rank-1 task is a
bpb_ task in 90 of 96 rows, which is 89 of the 95 real validation languages plus the `multi`
pseudo language. The six exceptions are high-resource languages with mature benchmarks:
de, en, es, fr, ru, zh. Italian and Japanese moved to bpb in this run.
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
title: Finding 7 — BPB out-SNRs the benchmarks, and not narrowly
subtitle: "Highest-SNR above-random measurement per language (`rel_mpsd` @ 1B)"
---

| | rank-1 measurement | typical SNR |
|---|---|---|
| **89 languages** | `bpb_<language>` | median 5.9, up to 78.9 (sq 78.9, et 78.3, lt 77.3, sk 66.3) |
| **6 languages** | a harness benchmark | 0.01 – 0.20 |

The six exceptions are `de`, `en`, `es`, `fr`, `ru`, `zh` — and their best benchmark
(`arc_challenge` 0.198, `hellaswag_de` 0.021) sits far below a typical BPB task.

- The gate is part of the story: most benchmarks are removed before SNR is computed, BPB never is
- But where a benchmark does survive, **neither measurement has much signal** — see the next slide

<!--
This is the project's research question answered in the negative for the benchmark suite,
and in the positive for BPB. It is also why the plan made per-language BPB the outcome
metric rather than the benchmarks.
-->

---
layout: figure
image: /ladder/snr_bpb_vs_benchmark.png
fit: contain
height: 60vh
title: Finding 7, read again
subtitle: "Bits per byte wins on coverage, not on sharpness"
---

<!--
Improved version of the two slides before. Left bar: of the 95 validation
languages, 87 have no benchmark left once the above-random gate has run, so bits
per byte wins there with nothing to beat. Right: the 8 languages where a
benchmark does survive, best benchmark against best bits per byte at 1B.
-->
---
layout: bullets
title: We measure best where our choices matter least
subtitle: "What that number is really counting"
icon: "🔁"
---

- In **87 of the 95** validation languages the gate removes every benchmark. Bits per byte is the only thing left, so it wins by walkover
- **8 languages** keep a benchmark. The benchmark edges it in 6, bits per byte in 2. The margins are tiny either way
- Here is the real point. In those 8 languages **every** measurement scores under 0.03. In the other 87, bits per byte has a median of **6.8** and clears 10 in 39 of them
- So the suite covers the languages where **no data mixture we tried makes a difference**, and says nothing about the languages where they do

<!--
Numbers from top_benchmarks_per_language.csv on the canonical pool, 8 September run,
variant rel_mpsd at 1B. The 96 in the original headline counts the `multi` pseudo language
(train_loss) alongside the 95 real ones. SNR here is spread across mixtures over the mean,
so a low score means the mixtures land on top of each other. English scores 0.001 because
every mixture serves it about equally well. This is the slide to say out loud: our suite is
silent in most
languages, which is a coverage problem to fix, not a reason to drop benchmarks.
-->

<!-- BEGIN auto:rq1-results (snr_definition_postprocess.py) -->
---
title: RQ2 — SNR definition
subtitle: "Results (auto) — most reliable benchmark per language (`rel_mpsd` @ 1B)"
---

| lang | top benchmark | SNR | DA-ckpt@1B |
|---|---|---|---|
| ar | `bpb_ary_Arab` | 13.7 | 1.00 |
| az | `bpb_azj_Latn` | 48.4 | 1.00 |
| bg | `bpb_bul_Cyrl` | 20.0 | 1.00 |
| bn | `bpb_ben_Beng` | 13.9 | 1.00 |
| bs | `bpb_bos_Latn` | 43.8 | 1.00 |
| ca | `bpb_cat_Latn` | 21.1 | 1.00 |
| cs | `bpb_ces_Latn` | 42.7 | 1.00 |
| da | `bpb_dan_Latn` | 32.5 | 1.00 |
| de | `hellaswag_de` | 0.0 | 1.00 |
| el | `bpb_ell_Grek` | 15.0 | 1.00 |
| en | `arc_challenge` | 0.2 | 1.00 |
| es | `hellaswag_es` | 0.0 | 1.00 |
| et | `bpb_ekk_Latn` | 78.3 | 1.00 |
| fa | `bpb_fas_Arab` | 18.5 | 1.00 |
| fi | `bpb_fin_Latn` | 48.8 | 1.00 |
| fr | `xwinograd_fr` | 0.1 | 0.50 |
| he | `bpb_heb_Hebr` | 23.1 | 1.00 |
| hi | `bpb_hin_Deva` | 7.3 | 1.00 |
| hr | `bpb_hrv_Latn` | 46.1 | 1.00 |
| hu | `bpb_hun_Latn` | 64.6 | 1.00 |
| id | `bpb_ind_Latn` | 13.4 | 1.00 |
| it | `bpb_ita_Latn` | 0.0 | 1.00 |
| ja | `bpb_jpn_Jpan` | 0.0 | 1.00 |
| ka | `bpb_kat_Geor` | 8.8 | 1.00 |
| kk | `bpb_kaz_Cyrl` | 20.2 | 1.00 |
| ko | `bpb_kor_Hang` | 14.2 | 1.00 |
| lt | `bpb_lit_Latn` | 77.3 | 1.00 |
| lv | `bpb_lvs_Latn` | 57.0 | 1.00 |
| ml | `bpb_mal_Mlym` | 15.4 | 1.00 |
| mr | `bpb_mar_Deva` | 17.8 | 1.00 |
| ms | `bpb_zsm_Latn` | 11.3 | 1.00 |
| ne | `bpb_npi_Deva` | 10.8 | 1.00 |
| nl | `bpb_nld_Latn` | 16.2 | 1.00 |
| no | `bpb_nob_Latn` | 34.2 | 1.00 |
| pl | `bpb_pol_Latn` | 33.8 | 1.00 |
| pt | `bpb_por_Latn` | 5.9 | 1.00 |
| ro | `bpb_ron_Latn` | 32.2 | 1.00 |
| ru | `hellaswag_ru` | 0.0 | 1.00 |
| sk | `bpb_slk_Latn` | 66.3 | 1.00 |
| sl | `bpb_slv_Latn` | 40.8 | 1.00 |
| sq | `bpb_als_Latn` | 78.9 | 1.00 |
| sr | `bpb_srp_Latn` | 47.5 | 1.00 |
| sv | `bpb_swe_Latn` | 27.5 | 1.00 |
| ta | `bpb_tam_Taml` | 6.8 | 1.00 |
| th | `bpb_tha_Thai` | 11.4 | 1.00 |
| tr | `bpb_tur_Latn` | 37.2 | 1.00 |
| uk | `bpb_ukr_Cyrl` | 7.8 | 1.00 |
| ur | `bpb_urd_Arab` | 13.2 | 1.00 |
| vi | `bpb_vie_Latn` | 16.7 | 1.00 |
| zh | `xcopa_zh` | 0.0 | 1.00 |

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq1-results -->

---
layout: bullets
title: Finding 8 — SNR predicts decision accuracy weakly, but better than yesterday
subtitle: "R = 0.79 in Heineman et al., 0.32 on this ladder"
icon: "❓"
---

- Best of the 22 variants on the canonical pool (`rel_mpsd`): mean Pearson r of log₁₀(SNR) vs DA
  = **+0.32** (DA-ckpt), **+0.06** (DA-size), **+0.19** overall
- Last night's evals moved this. A day earlier the leader was `iqr` at +0.28, and DA-size was **negative** at −0.10
- The **relative-spread** family leads throughout: `rel_mpsd` 0.32, `iqr` 0.28, `mpsd` 0.27. Plain dispersion trails at 0.03
- With this few models per size and a two-level intervention, DA is coarse — **the variant question may simply not be answerable at this pool size**

<!--
Compare Heineman et al. (2025): R = 0.791 between SNR and DA on the English DataDecide
ladder. Ours is over 96 languages with 39 models in the canonical pool and mostly BPB
tasks surviving the gate. The jump from 0.20 to 0.32 came from eval coverage alone, with
no change to the models, which is the strongest hint we have that the pool is the binding
constraint rather than the framework.
-->

---
layout: bullets
title: SNR predicts much less here than in English
subtitle: "Plain version of the slide before"
icon: "❓"
---

- Heineman et al. report a correlation of 0.79 between SNR and decision accuracy on the English ladder
- Our best of 22 definitions now reaches 0.32, when decisions are read across checkpoints
- Read across sizes it is 0.06. That is weak, but it was negative before last night's evals
- The definitions that do best are the relative spread family, around 0.3
- More evaluation data moved this number and no new models did. That points at the pool size as the thing holding us back
---
title: Finding 9 — No SNR choice survives a seed swap cleanly
subtitle: "Train on seeds 64/313, test on seed 1904"
---

| | DA-size | DA-ckpt |
| --- | ---: | ---: |
| Spearman ρ on the global variant ranking | **−0.09** | **+0.29** |
| Pearson r between splits (all cells) | +0.52 (n = 43) | +0.03 (n = 1094) |
| Retention of the train-picked variant | 58 % | 35 % |
| Per-language family agreement | 1 % | 10 % |

- Recommend an SNR **family**, never an exact variant
- The **per-language argmax never transfers** — 10 of 96 languages agree even at family level
- This is weaker than the same test gave a day earlier (ρ = +0.70). Read that as a noisy estimate, not a regression

<!--
compare_seed_splits.py, predictivity_seeds_train -> predictivity_seeds_test. The x3 cells
are 175M/600M at L in {1,2,50} only, so this holdout is thinner than the 36-sweep's.
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
that matters for the paper: the per language argmax agrees on 4 percent of
languages, so we recommend a family of SNR definitions and never a single one.
Numbers refreshed on the 8 September run: 10 percent at family level, not 4.
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
rq06 effect_vs_noise.csv. Checkpoint noise is detrended first — under WSD the final window
is still descending, so the raw std would be smaller still. A methodological result about
the framework rather than about our models; worth reporting in the paper.
-->

---
layout: bullets
title: Our noise estimate is too small
subtitle: "Plain version of the slide before"
icon: "🔬"
---

- The framework measures noise as the wobble over a run's last few checkpoints
- On the cells with three seeds we can measure it the honest way instead, across seeds
- Seed noise is 2.04 times the checkpoint noise, taken over 5,546 cells
- So every SNR we quote is about twice as flattering as it should be
- An effect that looks like twice the checkpoint noise is really about one seed re-roll
- Last night's evals grew this sample from 752 cells to 5,546, and the ratio settled from 2.54 to 2.04
---
title: Finding 11 — Bits per byte answers the question, benchmarks do not
subtitle: "RQ6 now scores both, and every comparison still resolves against 600M"
---

All **28 intervention comparisons** resolve against **600M** — 1B is finished at two language
settings and its scheme-B twins are started but not done.

Data-scheme decision, bits per byte over the languages both schemes train:

| proxy | L8 | L15 | L30 |
|---|---:|---:|---:|
| **175M** | 1.00 | 0.00 | **0.04** |
| **350M** | 1.00 | 0.83 | 1.00 |

175M gets the scheme decision **backwards** at L15 and L30; 350M is reliable wherever it is measured.

- The encouraging half: the **scaling fits do reach 1B** (202 of 303), and per-language BPB at the
  reference is predicted from the proxy rungs to within **5–6 %** (L8 0.058, L50 0.046)

<!--
rq06 on predictivity_seeds, 8 September: 28 intervention-DA cells, 303 scaling fits,
reference_size = 600M throughout. The depth rows exist only at L1/L2 and swing 0.24-0.84,
which is what "not enough matched pairs" looks like. The benchmark population is no longer
empty — see the next slide. This slide is still the argument for finishing 1B first.
-->

---
layout: figure
image: /ladder/benchmark_predictivity.png
fit: contain
height: 56vh
title: New tonight — the benchmark suite can be scored, and it fails
subtitle: "Two schemes, 2,227 shared benchmark tasks, one clear answer"
---

<!--
This is the headline change from last night's evals. Until this run no intervention had
both of its levels on enough shared benchmark tasks to score, so the benchmark population
was empty. It is not any more. Left: at 175M neither measurement is right, and bits per
byte is confidently wrong. What separates them is 350M, where bits per byte recovers to
0.83 and 1.00 while the benchmark suite has no comparison at all. Right: the strict test,
a proxy that agrees and keeps agreeing at every larger size. 369 of 500 bits per byte
comparisons pass it. None of 2,116 benchmark comparisons do.
-->
---
layout: bullets
title: What that means for the benchmark suite
subtitle: "The first direct answer to the question the ladder was built for"
icon: "🎯"
---

- The suite now has **2,227 task comparisons** with both data schemes on the same tasks. A week ago it had none
- At 175M the suite reads the scheme decision as a **coin flip**: 0.51 at 15 languages, 0.57 at 30
- Under the strict test, a proxy that agrees **and keeps agreeing** as models grow, **0 of 2,116** benchmark comparisons pass. **369 of 500** bits per byte comparisons do
- The two measurements also **disagree on the answer**. Bits per byte prefers scheme B, the benchmark suite prefers scheme A
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

- At **1 language** the depth decision is the hardest. 43 of the 100 validation subsets have no proxy size that gets it right
- At **8 and 15 languages** a 175M model is enough for most languages. The scheme decision is easier to predict than the depth one
- **Macro bits per byte is not a shortcut.** It fails outright at 8 languages, where the per language answers are mostly fine
- Training loss agrees with 600M at every language count except 1
- Benchmarks now appear, at 15 and 30 languages. **Not one of the 2,116** ever reaches a proxy size that agrees and keeps agreeing

<!--
The reference here is 600M, not 1B, because 1B has no finished matched shallow or scheme B
cell. So this is a small to 600M read, not a small to large one. Two messages for the room:
the aggregate metric we planned to decide on is worse than the per language ones it
averages, and the benchmark line sits flat on "no size works" for every task we can test.
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

<!-- BEGIN auto:rq2-results (allenai_comparison/analyze.py) -->
<!-- END auto:rq2-results -->

---
layout: figure
image: /ladder/fig4_subset_sweep.png
fit: contain
height: 56vh
title: Finding 12 — A subject subset beats the full Global-MMLU at every size
subtitle: "Cumulative SNR as subjects are added in standalone-SNR order; dashed = the full set"
---

<!-- BEGIN auto:rq3-results (smooth_subtasks.py) -->
---
title: RQ4 — Subtask subsets
subtitle: "Results (auto) — top subset gains (SNR: full → best subset)"
---

| case | task | size | full → best SNR | +gain |
|---|---|---|---|---|
| global_mmlu_full_per_language | `global_mmlu_full_ro` | 175M | 0.07 → 2.49 | +2.42 |
| global_mmlu_full_per_language | `global_mmlu_full_ja` | 175M | 1.10 → 3.12 | +2.03 |
| global_mmlu_full_per_language | `global_mmlu_full_sv` | 175M | 0.36 → 2.31 | +1.95 |
| per_benchmark | `global_piqa_parallel_cloze` | 350M | 0.75 → 2.66 | +1.91 |
| global_mmlu_full_per_language | `global_mmlu_full_uk` | 350M | 0.15 → 1.95 | +1.80 |
| global_mmlu_full_per_language | `global_mmlu_full_uk` | 175M | 0.47 → 2.24 | +1.77 |
| global_mmlu_full_subjects | `global_mmlu_full` | 175M | 2.13 → 3.87 | +1.75 |
| global_mmlu_full_per_language | `global_mmlu_full_ar` | 350M | 0.54 → 2.26 | +1.72 |

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq3-results -->

<!-- BEGIN auto:rq4-results (benchmark_creation/analyze.py) -->
---
title: RQ5 — Benchmark design
subtitle: "Results (auto) — per-family SNR, above-random survivors"
---

| family | median SNR | n_opts | format |
|---|---|---|---|
| `arc` | 1.03 | 4 | mcq_question_only |
| `multiblimp` | 0.71 | 2 | minimal_pair |
| `hellaswag` | 0.65 | 4 | completion |
| `xcopa` | 0.63 | 2 | completion |
| `xstorycloze` | 0.61 | 2 | completion |
| `xnli` | 0.37 | 3 | classification |
| `xwinograd` | 0.32 | 2 | completion |

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq4-results -->

---
layout: section
---

# Open discussion

Seven decisions, in the order they block each other

---
layout: focus
color: green
icon: "📏"
---

## 0. If the suite only covers the languages our choices do not move, what is it for?

Do we recommend **BPB as the measurement** and use benchmarks only where they clear the gate — or is that
conceding the question the project set out to answer?

<!--
Finding 7. Three readings, and they need different follow-ups:
(a) benchmarks are genuinely unreliable at 90M-1B in most languages — then the contribution
    is the negative result plus the gate, and BPB is the recommendation;
(b) it is a selection artefact of the above-random gate, which never gates BPB — then we
    need an SNR that is comparable across gated and ungated measurements;
(c) it is a scale artefact and benchmarks separate above 1B — then the reference rungs
    settle it, and we cannot answer until they land.
-->

---
layout: focus
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
layout: focus
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
layout: focus
color: red
icon: "⏱️"
---

## 3. Eval walltime blocks everything above L = 30

L100 = 463 tasks → **1,356 min at 1.7B against a 719-min queue cap**. Already over at 1.7B/L30 and 1B/L50.

`BATCH_TASKS=1` writes nothing when a job is killed, so an over-cap job is resubmitted and killed forever.

**This is why L = 100 has zero finished cells.**

<!--
status-09-02, still open. Options: split the task set across jobs, cap the during-training
task list and backfill the rest offline, or raise the cap. Blocks the whole top-right of
the grid, and the predictivity question needs exactly that corner.
-->

---
layout: focus
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
layout: focus
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
layout: focus
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
layout: focus
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

- **Established**: scaling holds above 90M (α ≈ 0.16–0.20, residuals ≤ 0.07 nats); more languages is nearly free for English and worth ~0.3 bits/byte to everything else; three quarters of the multilingual suite sits at chance at these sizes; no per-language SNR choice survives a seed swap, so recommend a *family*
- **Answered, for one decision**: on data scheme, per-language bits per byte transfers from a 350M proxy and the benchmark suite does not, on 2,116 task comparisons. Still small-to-600M, because 1B has no finished matched pair yet
- **Uncomfortable**: two of our three axes — depth and data scheme — are at or near the seed-noise floor

---
layout: bullets
title: Where this leaves us
subtitle: "Plain version of the slide before"
icon: "✅"
---

- **Solid.** Scaling works above 90M. More languages costs English once and keeps paying everyone else. Three quarters of the benchmark suite sits at chance
- **Awkward.** The suite is silent in 87 of 95 languages, and the 8 it covers are the ones where no mixture we tried makes a difference. Our noise estimate is twice too small. Depth and scheme sit at or near the noise floor
- **Open.** The question the ladder was built for. Every proxy comparison we can make today lands on 600M, because 1B has no matched pair and 1.7B has none at all
---
layout: bullets
title: Next
subtitle: "In the order things unblock each other"
icon: "➡️"
---

1. Decide **T** — it gates every remaining data build
2. Fix the **eval walltime** — it is why L = 100 is empty
3. Settle the **90M** treatment and gate the loader on `run__off_trend`
4. Finish the **reference rungs**: 1B at more L, or 1.7B, but not both
5. Re-run `run_all_predictivity.sh` — every number in this deck refreshes from the published report


<!-- BEGIN generated signal slides (analysis/rq01_decision_accuracy/da_per_benchmark.py) -->

---
layout: section
---

# Appendix — Signal & Predictability across Sizes


---
title: Appendix — Above-random signal
subtitle: "Predictivity ladder (90M–1.7B, seed 1904) · mean score per family × size (bold = beats chance + 0.05)"
---

| benchmark | rand | 90M | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|---|
| `loss` |  | 4.85 | 3.10 | 2.65 | 2.46 | 2.37 |
| `bpb` |  | 4.36 | 1.97 | 1.68 | 1.58 | 1.40 |
| `multiblimp` | 0.50 | **0.70** | **0.80** | **0.89** | **0.91** | **0.92** |
| `xwinograd` | 0.50 | 0.51 | 0.53 | **0.59** | **0.64** | **0.68** |
| `xcopa` | 0.50 |  | 0.54 | 0.55 | **0.56** | **0.57** |
| `xstorycloze` | 0.50 | 0.48 | 0.50 | 0.54 | **0.57** | **0.58** |
| `global_piqa_nonparallel_cloze` | 0.50 |  | 0.49 | 0.53 | **0.57** | 0.50 |
| `paws` | 0.50 | 0.50 | 0.49 | 0.50 | 0.49 | 0.52 |
| `xnli` | 0.33 | 0.33 | 0.35 | **0.40** | **0.42** | **0.44** |
| `hellaswag` | 0.25 | 0.26 | 0.26 | 0.28 | **0.30** | **0.33** |
| `include_base_44` | 0.25 | 0.27 | 0.25 | 0.25 | 0.25 | 0.25 |
| `global_mmlu_full` | 0.25 | 0.23 | 0.24 | 0.25 | 0.24 | 0.24 |
| `belebele` | 0.25 | 0.23 | 0.25 | 0.24 | 0.24 | 0.25 |
| `truthfulqa-multi_mc1` | 0.25 | 0.24 | 0.25 | 0.22 | 0.22 | 0.22 |
| `arc` | 0.25 | 0.23 | 0.20 | 0.22 | 0.23 | 0.26 |
| `lambada_openai_mt` |  | 0.00 | 0.14 | 0.27 | 0.33 | 0.31 |
| `global_piqa_parallel_cloze` | 0.50 | 0.19 | 0.20 | 0.21 | 0.21 | 0.21 |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `multiblimp` | **0.80** | 0.71 | **1.00** | 0.60 | **1.00** | **1.00** |
| `hellaswag` | 0.67 | 0.46 | 0.00 | **0.80** | **1.00** | **1.00** |
| `xwinograd` | **0.87** | 0.50 | **1.00** | 0.53 | **1.00** | 0.00 |
| `global_mmlu_full` | 0.47 | **0.75** | 0.00 | 0.67 | **1.00** | **1.00** |
| `arc_challenge` | 0.53 | 0.54 | 0.00 | 0.67 | **1.00** | **1.00** |
| `paws` | 0.33 | 0.60 | **1.00** | 0.30 | 0.00 | **1.00** |
| `belebele` | 0.60 | 0.71 | 0.00 | 0.60 | 0.00 | **1.00** |
| `bpb` | 0.71 | 0.53 | 0.00 | 0.67 | 0.00 | **1.00** |
| `xnli` | 0.47 | 0.54 | **1.00** | **0.87** | 0.00 | 0.00 |
| `arc_easy` | 0.73 | 0.43 | 0.00 | 0.53 | 0.00 | **1.00** |
| `lambada_openai_mt` | 0.67 | 0.60 | 0.00 | 0.30 | 0.00 | **1.00** |
| `global_piqa_parallel_cloze` | 0.33 | 0.67 | **1.00** | 0.40 | 0.00 | 0.00 |
| `xstorycloze` | 0.60 | 0.68 | 0.00 | **0.93** | 0.00 | 0.00 |
| `truthfulqa-multi_mc1` | 0.50 | 0.07 | 0.00 | 0.50 | 0.00 | **1.00** |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb_ary_Arab` | **0.82** | **0.82** | **1.00** | **1.00** | **1.00** | **1.00** |
| `bpb_ars_Arab` | **0.78** | **0.80** | **1.00** | **0.98** | **1.00** | **1.00** |
| `bpb_arb_Arab` | **0.78** | **0.80** | **1.00** | **0.98** | **1.00** | **1.00** |
| `bpb_arz_Arab` | **0.78** | **0.82** | **1.00** | **0.96** | **1.00** | **1.00** |
| `arc` | 0.67 | 0.67 |  | **1.00** |  |  |
| `multiblimp` | **1.00** | 0.67 |  | 0.67 |  |  |
| `global_piqa_parallel_cloze_arb_arab` | **1.00** | 0.67 |  | 0.33 |  |  |
| `global_piqa_parallel_cloze_apc_arab_jord` | 0.33 | 0.50 |  | **1.00** |  |  |
| `global_mmlu_full` | 0.33 | **0.83** |  | 0.33 |  |  |
| `global_piqa_parallel_cloze_arz_arab` | 0.33 | 0.50 |  | 0.67 |  |  |
| `global_piqa_parallel_cloze_ars_arab` | 0.67 | 0.67 |  | 0.00 |  |  |
| `global_piqa_parallel_cloze_ary_arab` | 0.33 | 0.33 |  | 0.67 |  |  |
| `belebele_arz_Arab` | 0.33 | 0.67 |  | 0.33 |  |  |
| `belebele_arb_Latn` | 0.67 | 0.33 |  | 0.33 |  |  |
| `belebele_ars_Arab` | 0.67 | 0.67 |  | 0.00 |  |  |
| `global_piqa_parallel_cloze_apc_arab_leba` | 0.33 | 0.67 |  | 0.33 |  |  |
| `xnli` | 0.33 | **0.83** |  | 0.00 |  |  |
| `belebele_arb_Arab` | 0.33 | 0.17 |  | 0.67 |  |  |
| `belebele_apc_Arab` | 0.33 | 0.00 |  | 0.67 |  |  |
| `belebele_ary_Arab` | 0.33 | 0.00 |  | 0.67 |  |  |
| `global_piqa_parallel_cloze_apc_arab_pale` | 0.00 | **1.00** |  | 0.00 |  |  |
| `global_piqa_parallel_cloze_apc_arab_syri` | 0.33 | 0.33 |  | 0.33 |  |  |
| `hellaswag` | 0.00 | 0.33 |  | 0.67 |  |  |
| `include_base_44` | 0.33 | 0.50 |  | 0.00 |  |  |
| `xstorycloze` | 0.00 | 0.17 |  | 0.67 |  |  |

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
subtitle: "az (az) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.87** | **0.91** | **1.00** | **0.93** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_az.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "az (az) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "bg (bg) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `xnli` | **1.00** | **1.00** |  | **1.00** |  |  |
| `bpb` | **0.95** | **0.95** | **1.00** | **1.00** | **1.00** | **1.00** |
| `belebele` | **1.00** | 0.00 |  | 0.00 |  |  |
| `global_piqa_parallel_cloze` | **1.00** | 0.00 |  | 0.00 |  |  |
| `include_base_44` | 0.00 | **1.00** |  | 0.00 |  |  |
| `multiblimp` | **1.00** | 0.00 |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `global_piqa_parallel_cloze_ben_latn` | **1.00** | **1.00** |  | **1.00** |  |  |
| `bpb` | **0.85** | **0.89** | **1.00** | **0.96** | **1.00** | **1.00** |
| `multiblimp` | **1.00** | 0.67 |  | **1.00** |  |  |
| `global_mmlu_full` | **1.00** | 0.67 |  | 0.00 |  |  |
| `hellaswag` | 0.00 | 0.50 |  | **1.00** |  |  |
| `arc` | 0.00 | 0.50 |  | **1.00** |  |  |
| `global_piqa_parallel_cloze_ben_beng` | 0.00 | 0.50 |  | **1.00** |  |  |
| `belebele_ben_Latn` | 0.00 | **0.83** |  | 0.00 |  |  |
| `include_base_44` | 0.00 | **0.83** |  | 0.00 |  |  |
| `belebele_ben_Beng` | 0.00 | 0.67 |  | 0.00 |  |  |

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
subtitle: "bs (bs) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.78** | 0.75 | **1.00** | **0.89** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_bs.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "bs (bs) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "ca (ca) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.85** | **0.84** | **1.00** | **0.91** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ca.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "ca (ca) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "cs (cs) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `belebele` | **1.00** | **1.00** |  | **1.00** |  |  |
| `bpb` | 0.75 | 0.75 | **1.00** | **0.96** | **1.00** | **1.00** |
| `global_mmlu_full` | 0.00 | 0.67 |  | **1.00** |  |  |
| `global_piqa_parallel_cloze` | 0.00 | 0.33 |  | **1.00** |  |  |
| `multiblimp` | **1.00** | 0.00 |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.87** | **0.80** | **1.00** | **0.85** | **1.00** | **1.00** |
| `arc` | 0.00 | 0.00 |  | **1.00** |  |  |
| `belebele` | 0.00 | 0.00 |  | **1.00** |  |  |
| `hellaswag` | 0.00 | **1.00** |  | 0.00 |  |  |
| `multiblimp` | 0.00 | **1.00** |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `hellaswag` | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | **1.00** |
| `bpb` | **0.82** | **0.80** | **1.00** | **0.98** | **1.00** | **1.00** |
| `multiblimp` | **1.00** | 0.47 | **1.00** | 0.67 | **1.00** | **1.00** |
| `arc` | 0.33 | 0.67 | **1.00** | 0.67 | **1.00** | **1.00** |
| `global_piqa_parallel_cloze` | 0.33 | 0.60 | **1.00** | 0.50 | 0.00 | **1.00** |
| `lambada_openai_mt` | 0.33 | 0.67 | **1.00** | 0.33 | 0.00 | **1.00** |
| `xnli` | 0.67 | 0.40 | **1.00** | 0.50 | 0.00 | 0.00 |
| `belebele` | 0.67 | 0.20 | 0.00 | 0.50 | 0.00 | **1.00** |
| `paws` | 0.17 | **0.87** | 0.00 | 0.33 | **1.00** | 0.00 |
| `global_mmlu_full` | 0.50 | 0.40 | 0.00 | 0.17 | **1.00** | 0.00 |
| `include_base_44` | 0.33 | 0.73 | 0.00 | 0.50 | 0.00 | 0.00 |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `multiblimp` | **1.00** | **1.00** |  | **1.00** |  |  |
| `bpb` | **0.89** | **0.87** | **1.00** | **0.98** | **1.00** | **1.00** |
| `global_mmlu_full` | **1.00** | 0.50 |  | 0.00 |  |  |
| `global_piqa_parallel_cloze` | 0.00 | 0.50 |  | **1.00** |  |  |
| `xnli` | **1.00** | 0.17 |  | 0.00 |  |  |
| `include_base_44` | 0.00 | **0.83** |  | 0.00 |  |  |
| `belebele` | 0.00 | 0.67 |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.91** | **0.91** | **1.00** | **0.93** | **1.00** | **1.00** |
| `hellaswag` | **0.83** | 0.60 | **1.00** | **1.00** | **1.00** | **1.00** |
| `truthfulqa-multi_mc1` | 0.50 | 0.50 | **1.00** | 0.50 | **1.00** | **1.00** |
| `arc` | 0.67 | 0.30 | 0.00 | 0.50 | **1.00** | **1.00** |
| `global_mmlu_full` | 0.17 | **0.80** | 0.00 | 0.50 | **1.00** | **1.00** |
| `lambada_openai_mt` | 0.50 | 0.60 | 0.00 | 0.33 | **1.00** | **1.00** |
| `paws` | 0.17 | 0.30 | 0.00 | **0.83** | **1.00** | **1.00** |
| `xstorycloze` | 0.33 | 0.30 | 0.00 | 0.67 | **1.00** | **1.00** |
| `xnli` | 0.67 | 0.30 | **1.00** | 0.17 | 0.00 | **1.00** |
| `global_piqa_parallel_cloze_spa_latn_spai` | 0.33 | 0.10 | 0.00 | 0.67 | **1.00** | **1.00** |
| `multiblimp` | 0.33 | 0.50 | **1.00** | 0.00 | 0.00 | **1.00** |
| `global_piqa_parallel_cloze_spa_latn_mexi` | 0.50 | 0.60 | 0.00 | 0.33 | **1.00** | 0.00 |
| `global_piqa_parallel_cloze_spa_latn_peru` | 0.67 | 0.60 | 0.00 | **0.83** | 0.00 | 0.00 |
| `include_base_44` | 0.17 | 0.60 | 0.00 | 0.33 | **1.00** | 0.00 |
| `belebele` | 0.33 | 0.40 | 0.00 | 0.17 | 0.00 | **1.00** |

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
subtitle: "et (et) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | 0.60 | 0.60 | **1.00** | **0.96** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_et.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "et (et) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "fa (fa) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.91** | **0.89** | **1.00** | **0.98** | **1.00** | **1.00** |
| `global_mmlu_full` | 0.67 | **0.90** |  | 0.33 |  |  |
| `multiblimp` | 0.00 | 0.50 |  | **1.00** |  |  |
| `belebele` | 0.33 | 0.60 |  | 0.33 |  |  |
| `global_piqa_parallel_cloze` | 0.33 | 0.10 |  | 0.67 |  |  |
| `include_base_44` | 0.33 | 0.50 |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `global_piqa_parallel_cloze` | **1.00** | **1.00** |  | **1.00** |  |  |
| `include_base_44` | **1.00** | **1.00** |  | **1.00** |  |  |
| `bpb` | **0.76** | **0.76** | **1.00** | **0.93** | **1.00** | **1.00** |
| `belebele` | 0.00 | **1.00** |  | 0.00 |  |  |
| `multiblimp` | 0.00 | **1.00** |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.91** | **0.87** | **1.00** | **0.93** | **1.00** | **1.00** |
| `hellaswag` | 0.67 | **0.80** | **1.00** | **0.83** | **1.00** | **1.00** |
| `arc` | **0.83** | 0.30 | **1.00** | 0.33 | **1.00** | **1.00** |
| `xnli` | 0.33 | 0.60 | **1.00** | 0.33 | **1.00** | **1.00** |
| `multiblimp` | 0.33 | 0.40 | **1.00** | 0.50 | **1.00** | **1.00** |
| `global_mmlu_full` | 0.33 | 0.70 | **1.00** | 0.50 | 0.00 | **1.00** |
| `global_piqa_parallel_cloze_fra_latn_cana` | 0.17 | 0.60 | **1.00** | 0.50 | 0.00 | **1.00** |
| `paws` | 0.50 | 0.10 | 0.00 | 0.50 | **1.00** | **1.00** |
| `global_piqa_parallel_cloze_fra_latn_fran` | 0.33 | 0.20 | 0.00 | 0.50 | **1.00** | **1.00** |
| `lambada_openai_mt` | 0.00 | **0.90** | **1.00** | 0.00 | 0.00 | **1.00** |
| `belebele` | **1.00** | 0.50 | 0.00 | 0.33 | 0.00 | **1.00** |
| `xwinograd` | 0.33 | **0.80** | 0.00 | 0.67 | **1.00** | 0.00 |
| `include_base_44` | 0.67 | 0.50 | 0.00 | 0.50 | 0.00 | 0.00 |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `include_base_44` |  | **1.00** |  |  |  |  |
| `global_mmlu_full` |  | **1.00** |  |  |  |  |
| `bpb` | **0.80** | **0.76** | **1.00** | **0.89** | **1.00** | **1.00** |
| `belebele` |  | 0.67 |  |  |  |  |
| `global_piqa_parallel_cloze` |  | 0.33 |  |  |  |  |
| `multiblimp` |  | 0.33 |  |  |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.80** | **0.84** | **1.00** | **0.93** | **1.00** | **1.00** |
| `hellaswag` | **1.00** | 0.50 |  | **1.00** |  |  |
| `xnli` | **1.00** | 0.17 |  | **1.00** |  |  |
| `include_base_44` | 0.00 | 0.67 |  | **1.00** |  |  |
| `global_piqa_parallel_cloze` | 0.00 | 0.50 |  | **1.00** |  |  |
| `belebele_hin_Deva` | 0.00 | 0.33 |  | **1.00** |  |  |
| `multiblimp` | **1.00** | 0.17 |  | 0.00 |  |  |
| `xstorycloze` | **1.00** | 0.17 |  | 0.00 |  |  |
| `global_mmlu_full` | 0.00 | **1.00** |  | 0.00 |  |  |
| `belebele_hin_Latn` | 0.00 | 0.50 |  | 0.00 |  |  |
| `arc` | 0.00 | 0.50 |  | 0.00 |  |  |

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
subtitle: "hr (hr) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.78** | 0.73 | **1.00** | **0.91** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_hr.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "hr (hr) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "hu (hu) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.78** | 0.73 | **1.00** | **0.91** | **1.00** | **1.00** |
| `multiblimp` | **1.00** | 0.33 |  | **1.00** |  |  |
| `hellaswag` | **1.00** | 0.33 |  | **1.00** |  |  |
| `global_piqa_parallel_cloze` | 0.00 | 0.00 |  | **1.00** |  |  |
| `belebele` | **1.00** | 0.00 |  | 0.00 |  |  |
| `arc` | 0.00 | 0.67 |  | 0.00 |  |  |
| `include_base_44` | 0.00 | 0.33 |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.89** | **0.85** | **1.00** | **0.89** | **1.00** | **1.00** |
| `belebele` | **1.00** | **0.83** |  | 0.67 |  |  |
| `arc` | 0.67 | 0.17 |  | 0.67 |  |  |
| `global_mmlu_full` | **1.00** | 0.50 |  | 0.00 |  |  |
| `hellaswag` | 0.33 | 0.33 |  | 0.67 |  |  |
| `xstorycloze` | 0.33 | 0.33 |  | 0.67 |  |  |
| `global_piqa_parallel_cloze` | 0.33 | 0.67 |  | 0.00 |  |  |
| `include_base_44` | 0.00 | 0.67 |  | 0.33 |  |  |
| `xcopa` | 0.67 | 0.00 |  | 0.33 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.85** | **0.87** | **1.00** | **0.91** | **1.00** | **1.00** |
| `belebele` | **0.83** | 0.70 | **1.00** | 0.50 | **1.00** | **1.00** |
| `hellaswag` | 0.67 | 0.50 | **1.00** | **0.83** | **1.00** | **1.00** |
| `arc` | 0.67 | 0.50 | **1.00** | 0.67 | **1.00** | **1.00** |
| `include_base_44` | 0.33 | 0.70 | **1.00** | 0.33 | 0.00 | **1.00** |
| `global_mmlu_full` | 0.50 | 0.50 | **1.00** | 0.33 | 0.00 | **1.00** |
| `lambada_openai_mt` | 0.50 | 0.60 | 0.00 | 0.67 | 0.00 | **1.00** |
| `global_piqa_parallel_cloze` | **0.83** | 0.30 | 0.00 | 0.00 | 0.00 | **1.00** |
| `xcopa` | 0.17 | **0.80** | 0.00 | 0.00 | **1.00** | 0.00 |
| `multiblimp` | 0.17 | 0.60 | 0.00 | 0.67 | 0.00 | 0.00 |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.84** | **0.80** | **1.00** | **0.96** | **1.00** | **1.00** |
| `paws` | 0.50 | 0.33 | **1.00** | **0.83** | **1.00** | **1.00** |
| `global_mmlu_full` | 0.50 | 0.73 | **1.00** | 0.50 | 0.00 | **1.00** |
| `xwinograd` | **1.00** | 0.60 | 0.00 | **0.83** | 0.00 | **1.00** |
| `global_piqa_parallel_cloze` | 0.33 | 0.20 | 0.00 | 0.50 | **1.00** | **1.00** |
| `include_base_44` | 0.33 | 0.53 | **1.00** | 0.17 | 0.00 | 0.00 |
| `belebele` | 0.50 | 0.40 | 0.00 | 0.67 | 0.00 | 0.00 |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.78** | **0.80** | **1.00** | **0.95** | **1.00** | **1.00** |
| `belebele` |  | 0.67 |  |  |  |  |
| `global_piqa_parallel_cloze` |  | 0.67 |  |  |  |  |
| `multiblimp` |  | 0.67 |  |  |  |  |
| `include_base_44` |  | 0.00 |  |  |  |  |

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
subtitle: "kk (kk) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.84** | **0.84** | **1.00** | **1.00** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_kk.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "kk (kk) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Korean (ko) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `global_piqa_parallel_cloze` | **1.00** | **0.83** |  | **1.00** |  |  |
| `bpb` | **0.85** | **0.78** | **1.00** | **0.93** | **1.00** | **1.00** |
| `belebele` | **1.00** | 0.67 |  | **1.00** |  |  |
| `paws` | **1.00** | 0.67 |  | **1.00** |  |  |
| `include_base_44` | **1.00** | 0.50 |  | 0.00 |  |  |
| `global_mmlu_full` | 0.00 | 0.33 |  | 0.00 |  |  |

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
subtitle: "lt (lt) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | 0.62 | 0.64 | **1.00** | **0.95** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_lt.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "lt (lt) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "lv (lv) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | 0.56 | 0.60 | **1.00** | **0.96** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_lv.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "lv (lv) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "ml (ml) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `hellaswag` |  | **1.00** |  |  |  |  |
| `global_piqa_parallel_cloze` |  | **1.00** |  |  |  |  |
| `bpb` | **0.82** | **0.91** | **1.00** | **0.87** | **1.00** | **1.00** |
| `include_base_44` |  | 0.67 |  |  |  |  |
| `arc` |  | 0.33 |  |  |  |  |
| `belebele` |  | 0.00 |  |  |  |  |

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
subtitle: "mr (mr) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.82** | **0.78** | **1.00** | **0.85** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_mr.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "mr (mr) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "ms (ms) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.89** | **0.87** | **1.00** | **0.91** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ms.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "ms (ms) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "ne (ne) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.89** | **0.78** | **1.00** | **0.78** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ne.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "ne (ne) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "nl (nl) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.89** | **0.84** | **1.00** | **0.91** | **1.00** | **1.00** |
| `multiblimp` | 0.67 | **1.00** |  | 0.67 |  |  |
| `global_mmlu_full` | 0.67 | **1.00** |  | 0.67 |  |  |
| `hellaswag` | 0.33 | 0.50 |  | **1.00** |  |  |
| `global_piqa_parallel_cloze` | 0.67 | 0.50 |  | 0.33 |  |  |
| `belebele` | 0.33 | 0.33 |  | 0.67 |  |  |
| `arc` | 0.67 | 0.67 |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `global_piqa_parallel_cloze` | **1.00** | **1.00** |  | **1.00** |  |  |
| `bpb` | **0.93** | **0.89** | **1.00** | **0.93** | **1.00** | **1.00** |
| `belebele` | 0.00 | 0.00 |  | **1.00** |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | 0.75 | **0.78** | **1.00** | **0.93** | **1.00** | **1.00** |
| `multiblimp` | 0.67 | **1.00** |  | 0.67 |  |  |
| `global_piqa_parallel_cloze` | 0.33 | 0.50 |  | 0.67 |  |  |
| `belebele` | 0.67 | 0.33 |  | 0.00 |  |  |
| `global_mmlu_full` | 0.33 | 0.67 |  | 0.00 |  |  |
| `include_base_44` | 0.00 | 0.67 |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.91** | **0.91** | **1.00** | **0.93** | **1.00** | **1.00** |
| `global_piqa_parallel_cloze_por_latn_port` | **1.00** | **0.83** |  | 0.67 |  |  |
| `multiblimp` | **1.00** | 0.67 |  | 0.67 |  |  |
| `global_mmlu_full` | 0.33 | **0.83** |  | 0.67 |  |  |
| `hellaswag` | 0.33 | 0.33 |  | 0.67 |  |  |
| `arc` | 0.67 | 0.67 |  | 0.00 |  |  |
| `belebele` | 0.00 | 0.17 |  | **1.00** |  |  |
| `include_base_44` | 0.67 | 0.50 |  | 0.00 |  |  |
| `global_piqa_parallel_cloze_por_latn_braz` | 0.67 | 0.33 |  | 0.00 |  |  |
| `xwinograd` | 0.33 | 0.50 |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.78** | **0.78** | **1.00** | **0.89** | **1.00** | **1.00** |
| `global_mmlu_full` | **1.00** | 0.33 |  | **1.00** |  |  |
| `global_piqa_parallel_cloze` | **1.00** | 0.67 |  | 0.00 |  |  |
| `arc` | **1.00** | 0.33 |  | 0.00 |  |  |
| `belebele` | 0.00 | 0.00 |  | **1.00** |  |  |
| `hellaswag` | 0.00 | 0.00 |  | **1.00** |  |  |
| `multiblimp` | **1.00** | 0.00 |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.89** | **0.89** | **1.00** | **1.00** | **1.00** | **1.00** |
| `hellaswag` | 0.70 | 0.57 | **1.00** | **1.00** | **1.00** | **1.00** |
| `arc` | 0.70 | 0.48 | 0.00 | **1.00** | **1.00** | **1.00** |
| `include_base_44` | 0.60 | 0.48 | **1.00** | 0.60 | **1.00** | 0.00 |
| `belebele` | 0.50 | 0.57 | 0.00 | 0.30 | **1.00** | **1.00** |
| `global_piqa_parallel_cloze` | 0.50 | 0.27 | **1.00** | 0.60 | 0.00 | **1.00** |
| `multiblimp` | **1.00** | 0.62 | 0.00 | 0.70 | 0.00 | **1.00** |
| `global_mmlu_full` | 0.40 | 0.52 | **1.00** | 0.30 | 0.00 | **1.00** |
| `xnli` | **0.90** | 0.43 | 0.00 | 0.40 | 0.00 | **1.00** |
| `xwinograd` | 0.50 | 0.43 | **1.00** | 0.40 | 0.00 | 0.00 |
| `xstorycloze` | **1.00** | 0.67 | 0.00 | 0.60 | 0.00 | 0.00 |

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
subtitle: "sk (sk) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.76** | 0.71 | **1.00** | **0.91** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_sk.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "sk (sk) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "sl (sl) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | 0.67 | 0.73 | **1.00** | **0.95** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_sl.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "sl (sl) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "sq (sq) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | 0.60 | 0.65 | **1.00** | **0.76** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_sq.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "sq (sq) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "sr (sr) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb_srp_Cyrl` | **0.80** | **0.76** | **1.00** | **0.96** | **1.00** | **1.00** |
| `bpb_srp_Latn` | 0.73 | **0.76** | **1.00** | **0.93** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_sr.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "sr (sr) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "sv (sv) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `hellaswag` | **1.00** | **1.00** |  | **1.00** |  |  |
| `bpb` | 0.75 | **0.87** | **1.00** | **0.80** | **1.00** | **1.00** |
| `multiblimp` | **1.00** | 0.67 |  | **1.00** |  |  |
| `belebele` | 0.00 | 0.33 |  | **1.00** |  |  |
| `arc` | **1.00** | 0.33 |  | 0.00 |  |  |
| `global_piqa_parallel_cloze` | 0.00 | 0.00 |  | **1.00** |  |  |
| `global_mmlu_full` | 0.00 | **1.00** |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.89** | **0.85** | **1.00** | **0.89** | **1.00** | **1.00** |
| `xcopa` |  | 0.67 |  |  |  |  |
| `multiblimp` |  | 0.67 |  |  |  |  |
| `belebele` |  | 0.33 |  |  |  |  |
| `include_base_44` |  | 0.33 |  |  |  |  |
| `global_piqa_parallel_cloze` |  | 0.33 |  |  |  |  |
| `arc` |  | 0.00 |  |  |  |  |
| `hellaswag` |  | 0.00 |  |  |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.87** | **0.87** | **1.00** | **0.96** | **1.00** | **1.00** |
| `global_piqa_parallel_cloze` | **1.00** | 0.67 |  | **1.00** |  |  |
| `xnli` | **1.00** | 0.50 |  | **1.00** |  |  |
| `xcopa` | **1.00** | 0.33 |  | **1.00** |  |  |
| `belebele` | 0.00 | 0.33 |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.82** | **0.82** | **1.00** | **0.93** | **1.00** | **1.00** |
| `xnli` | **1.00** | 0.67 |  | **1.00** |  |  |
| `multiblimp` | 0.00 | 0.33 |  | **1.00** |  |  |
| `global_piqa_parallel_cloze` | **1.00** | 0.33 |  | 0.00 |  |  |
| `global_mmlu_full` | 0.00 | **1.00** |  | 0.00 |  |  |
| `belebele` | **1.00** | 0.00 |  | 0.00 |  |  |
| `include_base_44` | 0.00 | **1.00** |  | 0.00 |  |  |
| `xcopa` | 0.00 | 0.00 |  | **1.00** |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.91** | **0.91** | **1.00** | **1.00** | **1.00** | **1.00** |
| `hellaswag` | **1.00** | 0.67 |  | **1.00** |  |  |
| `belebele` | **1.00** | 0.67 |  | 0.00 |  |  |
| `global_piqa_parallel_cloze` | 0.00 | 0.00 |  | **1.00** |  |  |
| `include_base_44` | 0.00 | **1.00** |  | 0.00 |  |  |
| `arc` | 0.00 | 0.67 |  | 0.00 |  |  |
| `global_mmlu_full` | 0.00 | 0.67 |  | 0.00 |  |  |
| `multiblimp` | 0.00 | 0.67 |  | 0.00 |  |  |

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
subtitle: "ur (ur) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.87** | **0.80** | **1.00** | **0.85** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
layout: figure
image: /ladder/appendix/da_ur.png
fit: contain
height: 72vh
title: Appendix — Decision accuracy across sizes
subtitle: "ur (ur) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Vietnamese (vi) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.87** | **0.84** | **1.00** | **0.89** | **1.00** | **1.00** |
| `include_base_44` | 0.67 | 0.67 |  | **1.00** |  |  |
| `arc` | 0.67 | 0.33 |  | **1.00** |  |  |
| `xnli` | **1.00** | 0.33 |  | 0.33 |  |  |
| `global_piqa_parallel_cloze` | **1.00** | 0.33 |  | 0.33 |  |  |
| `hellaswag` | 0.67 | 0.67 |  | 0.00 |  |  |
| `global_mmlu_full` | 0.67 | 0.67 |  | 0.00 |  |  |
| `belebele` | 0.00 | 0.50 |  | 0.67 |  |  |
| `xcopa` | 0.00 | **0.83** |  | 0.00 |  |  |

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

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.91** | **0.87** | **1.00** | **0.96** | **1.00** | **1.00** |
| `arc` | 0.50 | **0.80** | **1.00** | **0.83** | **1.00** | **1.00** |
| `paws` | 0.50 | 0.53 | **1.00** | 0.67 | **1.00** | **1.00** |
| `xnli` | 0.33 | 0.73 | **1.00** | 0.67 | 0.00 | **1.00** |
| `global_piqa_parallel_cloze_cmn_hans` | 0.33 | 0.53 | **1.00** | 0.33 | **1.00** | 0.00 |
| `belebele_zho_Hans` | 0.50 | 0.47 | **1.00** | 0.17 | 0.00 | **1.00** |
| `include_base_44` | 0.17 | 0.47 | **1.00** | 0.50 | 0.00 | **1.00** |
| `belebele_zho_Hant` | 0.67 | **0.80** | **1.00** | 0.50 | 0.00 | 0.00 |
| `global_mmlu_full` | 0.17 | 0.67 | **1.00** | 0.50 | 0.00 | 0.00 |
| `xstorycloze` | 0.50 | 0.47 | 0.00 | 0.33 | 0.00 | **1.00** |
| `xcopa` | 0.33 | 0.53 | 0.00 | 0.33 | **1.00** | 0.00 |
| `xwinograd` | 0.00 | 0.53 | 0.00 | 0.67 | **1.00** | 0.00 |
| `global_piqa_parallel_cloze_cmn_hant` | 0.67 | 0.27 | 0.00 | 0.00 | 0.00 | **1.00** |

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
