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
- **Language count**: 12.1× the seed noise — a real axis
- **Data scheme** (A vs B): 2.2× on loss, 1.2× on macro BPB — marginal
- **Model depth** (deep vs shallow): **1.0×** on the aggregate loss — the same model measured twice
- Per *individual* task the depth effect is larger — median **1.64×**, 42 % of 408 cells above 2× — so it separates *somewhere*, just not on the headline metric
- Depth is **half the grid**, and it buys a per-task effect we would have to hunt for

<!--
Matched pairs on healthy complete cells only: depth n=9, scheme n=7. Language axis is the
across-L range at fixed size (n=18 cells). This is the transformation table of
ladder_report.md, resolved against a proper seed standard deviation.
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

Of the **324** tasks that have a chance level, **85 clear it at any size**.

| answer options | chance | clears chance | at 1B |
| :---: | :---: | :---: | :---: |
| **2** (completion, minimal pair) | 0.50 | **50 / 129** | 48 / 129 |
| **3** (XNLI) | 0.33 | **12 / 15** | 11 / 15 |
| **4** (knowledge MCQA) | 0.25 | **23 / 180** | 23 / 180 |

- Clearing chance at 1B: `multiblimp` (33), `hellaswag` (20), `xnli` (11), `xwinograd` (6), `xstorycloze` (5), `xcopa` (4)
- **Never clearing chance anywhere**: `belebele`, `global_mmlu_full`, `global_piqa` (both splits), `paws`, `truthfulqa-multi`
- Strongly an **answer-count effect**: 39 % of 2-option tasks clear chance, **13 % of 4-option** ones do
- On the 36-model sweep the same families cleared chance for external 270M–70B models (122 of 124) — a capability floor of the small rungs, not a property of the benchmarks

<!--
above_random.py --only predictivity on the real report: 431 tasks, 324 with a chance level.
The other 107 are per-language BPB and generative tasks, which have none and are never
gated. Per bucket: 90M 3/18, 175M 41/324, 350M 56/219, 600M 75/219, 1B 82/219.
-->

---
layout: focus
color: blue
icon: "📏"
---

## In 89 of 96 languages, the most reliable measurement is **bits-per-byte**, not any benchmark

<!--
top_benchmarks_per_language.csv on the canonical pool: rank-1 task is a bpb_ task in 89 of
96 languages. The seven exceptions are exactly the high-resource languages with mature
benchmarks: de, en, es, fr, it, ru, zh.
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
subtitle: "Highest-SNR above-random measurement per language (`iqr` @ 1B)"
---

| | rank-1 measurement | typical SNR |
|---|---|---|
| **89 languages** | `bpb_<language>` | 15 – 23 (az 22.9, cs 22.2, bg 21.0, ar 15.1) |
| **7 languages** | a harness benchmark | 0.8 – 2.6 |

The seven exceptions are `de`, `en`, `es`, `fr`, `it`, `ru`, `zh` — and their best benchmark
(`hellaswag_de` 0.89, `arc_challenge` 2.58) is still an order of magnitude below a typical BPB task.

- The gate is part of the story: most benchmarks are removed before SNR is computed, BPB never is
- But the surviving benchmarks are also **genuinely noisier** — this is not only a selection effect

<!--
This is the project's research question answered in the negative for the benchmark suite,
and in the positive for BPB. It is also why the plan made per-language BPB the outcome
metric rather than the benchmarks.
-->

<!-- BEGIN auto:rq1-results (snr_definition_postprocess.py) -->
---
title: RQ2 — SNR definition
subtitle: "Results (auto) — most reliable benchmark per language (`iqr` @ 1B)"
---

| lang | top benchmark | SNR | DA-ckpt@1B |
|---|---|---|---|
| ar | `bpb_arb_Arab` | 15.1 | 1.00 |
| en | `arc_challenge` | 2.6 | 1.00 |
| es | `hellaswag_es` | 1.0 | 1.00 |
| eu | `bpb_eus_Latn` | 1.4 | 0.75 |
| hi | `bpb_hin_Deva` | 12.8 | 1.00 |
| ja | `bpb_jpn_Jpan` | 0.7 | 1.00 |
| ru | `multiblimp_rus` | 1.5 | 0.75 |
| sw | `bpb_swh_Latn` | 0.4 | 0.75 |
| th | `bpb_tha_Thai` | 15.1 | 1.00 |
| tr | `bpb_tur_Latn` | 22.1 | 1.00 |
| vi | `bpb_vie_Latn` | 15.5 | 1.00 |
| zh | `xstorycloze_zh` | 0.8 | 0.50 |

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq1-results -->

---
layout: bullets
title: Finding 8 — SNR barely predicts decision accuracy here
subtitle: "R = 0.79 in Heineman et al.; 0.20 on this ladder"
icon: "❓"
---

- Best of the 22 variants on the canonical pool (`iqr`): mean Pearson r of log₁₀(SNR) vs DA
  = **+0.20** (DA-ckpt), **−0.10** (DA-size), **+0.05** overall
- DA-ckpt is led by `mad` / `rel_mpsd` / `rel_mpd` (≈ 0.25) — the robust and relative-spread families
- Weakest: `tukey`, `dispersion_shifted`
- With this few models per size and a two-level intervention, DA is coarse — **the variant question may simply not be answerable at this pool size**

<!--
Compare Heineman et al. (2025): R = 0.791 between SNR and DA on the English DataDecide
ladder. Ours is over 96 languages with 39 models in the canonical pool and mostly BPB
tasks surviving the gate. Either the framework transfers weakly, or we need more models
per size — RQ2 cannot yet tell those apart.
-->

---
title: Finding 9 — Only the checkpoint-based ranking survives a seed swap
subtitle: "Train on seeds 64/313, test on seed 1904"
---

| | DA-size | DA-ckpt |
| --- | ---: | ---: |
| Spearman ρ on the global variant ranking | **+0.02** | **+0.70** |
| Pearson r between splits (all cells) | +0.44 | +0.73 |
| Retention of the train-picked variant | 8 % | 61 % |
| Per-language family agreement | 0 % | 4 % |

- Recommend an SNR **family**, never an exact variant
- The **per-language argmax never transfers** — 4 of 96 languages agree even at family level
- DA-size ranking is noise-dominated at this pool size; do not read it as a result

<!--
compare_seed_splits.py, predictivity_seeds_train -> predictivity_seeds_test. The x3 cells
are 175M/600M at L in {1,2,50} only, so this holdout is thinner than the 36-sweep's.
Same qualitative conclusion as the 36-sweep, which is itself reassuring.
-->

---
layout: bullets
title: Finding 10 — Late-checkpoint noise understates the real noise 2.5×
subtitle: "The standard S&N noise definition is optimistic on this ladder"
icon: "🔬"
---

- Signal-and-Noise measures noise as the spread over the **last few checkpoints** of one run
- We can measure it the other way too, over **seed replicates** of the same cell
- Median ratio **seed noise ÷ detrended checkpoint noise = 2.54**, over 752 (size, L, task) cells
- So an SNR computed the standard way is roughly **2.5× too optimistic** here
- An effect that looks like 2× checkpoint noise is about **0.8×** a seed re-roll

<!--
rq06 effect_vs_noise.csv. Checkpoint noise is detrended first — under WSD the final window
is still descending, so the raw std would be smaller still. A methodological result about
the framework rather than about our models; worth reporting in the paper.
-->

---
title: Finding 11 — The predictivity question is not yet answerable
subtitle: "RQ6 runs, but every comparison resolves against 600M"
---

All **26 intervention comparisons** resolve against **600M** — 1B is finished at two language
settings and has no matched shallow or scheme-B counterpart at either. On **benchmarks** there is
not one usable cell: no intervention has both levels on ≥ 3 shared tasks.

Data-scheme decision over the languages both schemes train:

| proxy | L8 | L15 | L30 |
|---|---:|---:|---:|
| **175M** | 1.00 | 0.00 | **0.04** |
| **350M** | 1.00 | 0.83 | 1.00 |

175M gets the scheme decision **backwards** at L15 and L30; 350M is reliable wherever it is measured.

- The encouraging half: the **scaling fits do reach 1B** (202 of 303), and per-language BPB at the
  reference is predicted from the proxy rungs to within **5–6 %** (L8 0.058, L50 0.046)

<!--
rq06 on predictivity_seeds: 26 intervention-DA cells, 303 scaling fits, reference_size =
600M throughout. The depth rows exist only at L1/L2 and swing 0.24-0.84 — that is what
"not enough matched pairs" looks like. This slide is the argument for finishing 1B first.
-->

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
| global_mmlu_full_per_language | `global_mmlu_full_id` | 600M | 0.35 → 2.21 | +1.86 |
| global_mmlu_full_per_language | `global_mmlu_full_uk` | 350M | 0.15 → 1.95 | +1.80 |
| global_mmlu_full_per_language | `global_mmlu_full_ro` | 175M | 0.21 → 1.94 | +1.74 |
| global_mmlu_full_per_language | `global_mmlu_full_ar` | 350M | 0.54 → 2.26 | +1.72 |
| global_mmlu_full_per_language | `global_mmlu_full_hi` | 350M | 0.25 → 1.95 | +1.70 |
| global_mmlu_full_per_language | `global_mmlu_full_en` | 1B | 0.03 → 1.64 | +1.61 |
| global_mmlu_full_per_language | `global_mmlu_full_zh` | 1B | 0.01 → 1.62 | +1.61 |
| global_mmlu_full_per_language | `global_mmlu_full_cs` | 600M | 0.18 → 1.77 | +1.59 |

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

## 0. If BPB wins in 89 of 96 languages, what is the benchmark suite for?

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

- **Established**: scaling holds above 90M (α ≈ 0.16–0.20, residuals ≤ 0.07 nats); more languages is nearly free for English and worth ~0.3 bits/byte to everything else; three quarters of the multilingual suite sits at chance at these sizes; the SNR *family* transfers across seeds, the per-language argmax does not
- **Not yet answerable**: the predictivity question itself. It needs a reference rung, and 1B is finished at two language settings while 1.7B and L = 100 have none
- **Uncomfortable**: two of our three axes — depth and data scheme — are at or near the seed-noise floor

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
| `multiblimp` | 0.50 | **0.70** | **0.83** | **0.89** | **0.92** | **0.93** |
| `xwinograd` | 0.50 | 0.51 | 0.54 | **0.59** | **0.64** | **0.68** |
| `xcopa` | 0.50 |  | 0.54 | 0.55 | **0.56** | **0.59** |
| `xstorycloze` | 0.50 | 0.48 | 0.50 | 0.55 | **0.57** | **0.60** |
| `paws` | 0.50 |  | 0.51 |  |  |  |
| `global_piqa_nonparallel_cloze` | 0.50 |  | 0.49 |  |  |  |
| `xnli` | 0.33 | 0.33 | 0.36 | **0.40** | **0.42** | **0.44** |
| `hellaswag` | 0.25 | 0.26 | 0.27 | 0.28 | **0.30** | **0.33** |
| `include_base_44` | 0.25 | 0.27 | 0.25 | 0.24 | 0.26 | 0.26 |
| `truthfulqa-multi_mc1` | 0.25 |  | 0.25 |  |  |  |
| `global_mmlu_full` | 0.25 | 0.23 | 0.24 | 0.25 | 0.25 | 0.24 |
| `belebele` | 0.25 | 0.23 | 0.24 | 0.24 | 0.24 | 0.25 |
| `arc` | 0.25 | 0.23 | 0.20 | 0.22 | 0.23 | 0.26 |
| `global_piqa_parallel_cloze` | 0.50 |  | 0.20 |  |  |  |
| `lambada_openai_mt` |  |  | 0.17 |  |  |  |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "English (en) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `multiblimp` | **0.80** | **0.80** | **1.00** | 0.60 | **1.00** | **1.00** |
| `hellaswag` | 0.67 | 0.47 | 0.00 | **0.80** | **1.00** | **1.00** |
| `xwinograd` | **0.87** | 0.53 | **1.00** | 0.53 | **1.00** | 0.00 |
| `arc_challenge` | 0.53 | 0.60 | 0.00 | 0.67 | **1.00** | **1.00** |
| `global_mmlu_full` | 0.47 | 0.67 | 0.00 | 0.67 | **1.00** | **1.00** |
| `xnli` | 0.47 | 0.60 | **1.00** | **0.87** | 0.00 | 0.00 |
| `belebele` | 0.60 | 0.73 | 0.00 | 0.60 | 0.00 | **1.00** |
| `bpb` | 0.71 | 0.53 | 0.00 | 0.67 | 0.00 | **1.00** |
| `arc_easy` | 0.73 | 0.53 | 0.00 | 0.53 | 0.00 | **1.00** |
| `xstorycloze` | 0.60 | 0.53 | 0.00 | **0.93** | 0.00 | 0.00 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Modern Std. Arabic (ar) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb_ary_Arab` | **0.82** | **0.82** | **1.00** | **1.00** | **1.00** | **1.00** |
| `bpb_arb_Arab` | **0.78** | **0.80** | **1.00** | **0.98** | **1.00** | **1.00** |
| `bpb_ars_Arab` | **0.78** | **0.80** | **1.00** | **0.98** | **1.00** | **1.00** |
| `bpb_arz_Arab` | **0.78** | **0.82** | **1.00** | **0.96** | **1.00** | **1.00** |
| `arc` | 0.67 | 0.67 |  | **1.00** |  |  |
| `multiblimp` | **1.00** | 0.67 |  | 0.67 |  |  |
| `global_mmlu_full` | 0.33 | **1.00** |  | 0.33 |  |  |
| `belebele_arb_Latn` | 0.67 | 0.67 |  | 0.33 |  |  |
| `belebele_ary_Arab` | 0.33 | 0.00 |  | 0.67 |  |  |
| `belebele_arz_Arab` | 0.33 | 0.33 |  | 0.33 |  |  |
| `belebele_arb_Arab` | 0.33 | 0.00 |  | 0.67 |  |  |
| `belebele_ars_Arab` | 0.67 | 0.33 |  | 0.00 |  |  |
| `hellaswag` | 0.00 | 0.33 |  | 0.67 |  |  |
| `include_base_44` | 0.33 | 0.67 |  | 0.00 |  |  |
| `xnli` | 0.33 | 0.67 |  | 0.00 |  |  |
| `xstorycloze` | 0.00 | 0.33 |  | 0.67 |  |  |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Spanish (es) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.91** | **0.91** | **1.00** | **0.93** | **1.00** | **1.00** |
| `hellaswag` | **0.83** | **0.83** | **1.00** | **1.00** | **1.00** | **1.00** |
| `arc` | 0.67 | 0.17 | 0.00 | 0.50 | **1.00** | **1.00** |
| `global_mmlu_full` | 0.17 | 0.67 | 0.00 | 0.50 | **1.00** | **1.00** |
| `xstorycloze` | 0.33 | 0.33 | 0.00 | 0.67 | **1.00** | **1.00** |
| `xnli` | 0.67 | 0.50 | **1.00** | 0.17 | 0.00 | **1.00** |
| `multiblimp` | 0.33 | 0.67 | **1.00** | 0.00 | 0.00 | **1.00** |
| `belebele` | 0.33 | 0.50 | 0.00 | 0.17 | 0.00 | **1.00** |
| `include_base_44` | 0.17 | 0.50 | 0.00 | 0.33 | **1.00** | 0.00 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Basque (eu) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.78** | **0.80** | **1.00** | **0.87** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Hindi (hi) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `xnli` | **1.00** | **1.00** |  | **1.00** |  |  |
| `hellaswag` | **1.00** | **1.00** |  | **1.00** |  |  |
| `bpb` | **0.80** | **0.84** | **1.00** | **0.93** | **1.00** | **1.00** |
| `arc` | 0.00 | **1.00** |  | 0.00 |  |  |
| `belebele_hin_Deva` | 0.00 | 0.00 |  | **1.00** |  |  |
| `belebele_hin_Latn` | 0.00 | **1.00** |  | 0.00 |  |  |
| `global_mmlu_full` | 0.00 | **1.00** |  | 0.00 |  |  |
| `include_base_44` | 0.00 | 0.00 |  | **1.00** |  |  |
| `multiblimp` | **1.00** | 0.00 |  | 0.00 |  |  |
| `xstorycloze` | **1.00** | 0.00 |  | 0.00 |  |  |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Japanese (ja) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.84** | **0.80** | **1.00** | **0.96** | **1.00** | **1.00** |
| `xwinograd` | **1.00** | **0.83** | 0.00 | **0.83** | 0.00 | **1.00** |
| `global_mmlu_full` | 0.50 | 0.67 | **1.00** | 0.50 | 0.00 | **1.00** |
| `include_base_44` | 0.33 | 0.50 | **1.00** | 0.17 | 0.00 | 0.00 |
| `belebele` | 0.50 | 0.17 | 0.00 | 0.67 | 0.00 | 0.00 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Russian (ru) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.89** | **0.89** | **1.00** | **1.00** | **1.00** | **1.00** |
| `hellaswag` | 0.70 | 0.70 | **1.00** | **1.00** | **1.00** | **1.00** |
| `arc` | 0.70 | 0.70 | 0.00 | **1.00** | **1.00** | **1.00** |
| `include_base_44` | 0.60 | 0.20 | **1.00** | 0.60 | **1.00** | 0.00 |
| `multiblimp` | **1.00** | 0.70 | 0.00 | 0.70 | 0.00 | **1.00** |
| `belebele` | 0.50 | 0.40 | 0.00 | 0.30 | **1.00** | **1.00** |
| `global_mmlu_full` | 0.40 | 0.30 | **1.00** | 0.30 | 0.00 | **1.00** |
| `xnli` | **0.90** | 0.30 | 0.00 | 0.40 | 0.00 | **1.00** |
| `xstorycloze` | **1.00** | 0.60 | 0.00 | 0.60 | 0.00 | 0.00 |
| `xwinograd` | 0.50 | 0.30 | **1.00** | 0.40 | 0.00 | 0.00 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Swahili (sw) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | 0.49 | 0.45 | 0.00 | **0.93** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Thai (th) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `xnli` | **1.00** | **1.00** |  | **1.00** |  |  |
| `xcopa` | **1.00** | **1.00** |  | **1.00** |  |  |
| `bpb` | **0.87** | **0.87** | **1.00** | **0.96** | **1.00** | **1.00** |
| `belebele` | 0.00 | **1.00** |  | 0.00 |  |  |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Turkish (tr) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `xnli` | **1.00** | **1.00** |  | **1.00** |  |  |
| `bpb` | **0.82** | **0.82** | **1.00** | **0.93** | **1.00** | **1.00** |
| `belebele` | **1.00** | 0.00 |  | 0.00 |  |  |
| `global_mmlu_full` | 0.00 | **1.00** |  | 0.00 |  |  |
| `include_base_44` | 0.00 | **1.00** |  | 0.00 |  |  |
| `multiblimp` | 0.00 | 0.00 |  | **1.00** |  |  |
| `xcopa` | 0.00 | 0.00 |  | **1.00** |  |  |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Vietnamese (vi) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.87** | **0.84** | **1.00** | **0.89** | **1.00** | **1.00** |
| `arc` | 0.67 | 0.67 |  | **1.00** |  |  |
| `include_base_44` | 0.67 | 0.67 |  | **1.00** |  |  |
| `xnli` | **1.00** | 0.33 |  | 0.33 |  |  |
| `global_mmlu_full` | 0.67 | 0.33 |  | 0.00 |  |  |
| `belebele` | 0.00 | 0.33 |  | 0.67 |  |  |
| `hellaswag` | 0.67 | 0.33 |  | 0.00 |  |  |
| `xcopa` | 0.00 | **1.00** |  | 0.00 |  |  |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Mandarin Chinese (zh) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B |
|---|---|---|---|---|---|---|
| `bpb` | **0.91** | **0.87** | **1.00** | **0.96** | **1.00** | **1.00** |
| `arc` | 0.50 | 0.67 | **1.00** | **0.83** | **1.00** | **1.00** |
| `xnli` | 0.33 | 0.67 | **1.00** | 0.67 | 0.00 | **1.00** |
| `belebele_zho_Hans` | 0.50 | 0.67 | **1.00** | 0.17 | 0.00 | **1.00** |
| `include_base_44` | 0.17 | 0.67 | **1.00** | 0.50 | 0.00 | **1.00** |
| `belebele_zho_Hant` | 0.67 | 0.50 | **1.00** | 0.50 | 0.00 | 0.00 |
| `global_mmlu_full` | 0.17 | 0.67 | **1.00** | 0.50 | 0.00 | 0.00 |
| `xcopa` | 0.33 | 0.67 | 0.00 | 0.33 | **1.00** | 0.00 |
| `xstorycloze` | 0.50 | 0.50 | 0.00 | 0.33 | 0.00 | **1.00** |
| `xwinograd` | 0.00 | 0.33 | 0.00 | 0.67 | **1.00** | 0.00 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

<!-- END generated signal slides -->
