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
  - Related work
  - Methodology
  - Research questions
  - Experimental setup
  - Findings
  - Open discussion
---

<!--
This deck follows the structure of research_proposal_high_level.pdf:
Introduction → Related Work → Methodology → Experimental Setup → Results → Analysis.
Results/Analysis numbers + figures are from the ladder report (branch data/ladder-report):
the `predictivity` pool (seed 1904, every ladder and data build, 90M–1.7B against the 1.7B
reference), with `predictivity_seeds` for the seed-noise reads. The 36-model sweep
(`custom_swissai_hf`) is an earlier period of the project and is never pooled with it.
-->

---
layout: section
---

# Introduction

Training a multilingual LM means making decisions all the time,

but the evaluations behind them are expensive and often uninformative.

---
layout: bullets
title: Motivation
subtitle: Lack of benchmark reliability
icon: "⚠️"
---

Training LMs is **expensive** and needs **constant decisions** (data mixture, hyperparameters...), each guided by benchmark scores. But

## Not every benchmark gives a useful signal

- **High variance**: scores change from run to run
- **Redundancy**: several benchmarks measure the same skill
- **Weak correlation**: a higher score does not always mean a better model
- **Cost**: large suites are too expensive to run often

---
layout: bullets
title: Motivation
subtitle: Why multilingual?
icon: "🌍"
---

## The problem is **worse** in multilingual settings

- Low-resource languages have little training data
- Small models score **near chance** on harder tasks
- Each added language brings new sources of variance
- Existing reliability tools were validated **on English only**

<!--
Most work on benchmark reliability uses English benchmarks and English-first models. Multilingual models face different problems, so the tools built for English may not transfer. We test whether the SNR framework still holds when the models and the benchmarks cover many languages.
-->

---
layout: focus
color: green
icon: 🎯
---

## Which (subsets of) benchmarks give a reliable signal at each stage of multilingual
## model training, and which small models can stand in for the large one?

---
layout: section
---

# Related work

What we build on: the Signal-and-Noise paper (Heineman et al., 2025, AllenAI)

---
layout: image-left
image: /snr_paper_figure_1.png
ratio: "25:1"
fit: contain
title: Signal-and-Noise paper
subtitle: Figure 1. Examples of signal and noise
---

---
layout: image-left
image: /snr_paper_figure_2.png
ratio: "2:3"
fit: contain
title: Signal-and-Noise paper
subtitle: Figure 2. Correlation between SNR and decision accuracy
---

- **SNR predicts decision accuracy** ($R = 0.791$). Signal or noise alone does not
- **Noise predicts scaling-law error** ($R = 0.653$): noisier benchmarks extrapolate less reliably
- **Keeping only high-SNR subtasks** gives subsets that beat the full benchmark (16 of 57 MMLU subtasks: +2.6% DA)
- **Averaging scores over checkpoints** lowers noise and raises decision accuracy (+2.4% on the 30-task average)
- **Bits per byte (BPB)** gives higher DA than accuracy on 90% of benchmarks, most of all on math and code


---
layout: section
---

# Methodology

The signal-aware framework, extended to many languages

---
title: Methodology
subtitle: Decision accuracy
---

<Block type="success" title="Decision Accuracy">

Take every pair of small models $(s_a, s_b)$ trained on datasets $a$ and $b$, and their large versions $(m_a, m_b)$. Does benchmark $B$ rank the small pair the same way as the large pair?

$$\text{DA} = \frac{1}{|\mathcal{P}|} \sum_{(a,b) \in \mathcal{P}} \mathbb{1}\big[\text{sign}(B(s_a) - B(s_b)) = \text{sign}(B(m_a) - B(m_b))\big]$$

</Block>


Pretraining:
- **DA-size**: does a small model rank the pair like the large one? (across sizes)
- **DA-ckpt**: does an early checkpoint rank the pair like the final one? (same size)

Post-training (extra):
- **DA-stage**: rank agreement across post-training stages (SFT, DPO, RLVR)
- **DA-ctx**: rank agreement between a shorter and a longer context

---
layout: focus
color: blue
icon: 🎯
---

## Decision accuracy is what a benchmark must get right

It is expensive to measure, so we look for **proxies** that are cheap to compute during training.

---
title: Methodology
subtitle: Signal and noise
---

<Block type="success" title="Signal (Relative Dispersion / Normalized Max. Difference)">

How well benchmark $b$ separates models of the same size $s$ trained on different settings (scores $m_j, m_k$):

$$\text{Rel. Dispersion}(b, s) = \frac{\max_{j,k} |m_j - m_k|}{\bar{m}}$$

</Block>

<Block type="success" title="Noise (Relative Std. Dev.)">

How much the score on benchmark $b$ moves over the last $n$ checkpoints of one model $m$:

$$\text{Rel. Std.}(b, m) = \frac{\sqrt{\frac{1}{n-1} \sum_{i=1}^{n}(m_i - \bar{m})^2}}{\bar{m}}$$

</Block>

$$\text{SNR} = \frac{\text{Avg. Rel. Dispersion}(\text{final checkpoint})}{\text{Rel. Std.}(\text{final $n$ checkpoints})}$$

<!--
Signal candidates: the 22 variants on the next slides are all alternative numerators.
Noise candidates: checkpoint noise (last n checkpoints) or benchmark noise (k-fold splits of a single run).
SNR = signal / noise. Higher means a more reliable benchmark.
-->


---
title: Methodology
subtitle: Signal definitions
---

Signal is how much a benchmark separates models. There are **many ways to measure spread**, so we test **22 candidate variants** in 5 families:

| Family | Members | Idea |
| ------ | ------- | ---- |
| **Dispersion** | `mpd`, `aad`, `rms_deviation`, `quartile_deviation`, `dist_std`, `dispersion`, `range` | Absolute spread of scores |
| **Relative-spread** | `rel_std`, `rel_mpd`, `rel_mpsd`, `iqr`, `rel_dispersion` | Spread divided by the mean (AllenAI default: `rel_std`) |
| **Discrepancy** | `discrepancy`, `star_discrepancy`, `star_discrepancy_shifted`, `rel_star_discrepancy`, `dispersion_shifted`, `gini` | How even or unequal the scores are |
| **Robust** | `mad`, `mpsd` | Outlier-resistant spread |
| **Depth** | `tukey`, `projection` | Half-space statistical depth |

**Q1**: Which variant best tracks decision accuracy across languages?

---
title: Methodology
subtitle: "Signal definitions: formulas"
---

Each variant is a different **signal** numerator (SNR = signal / noise). $c_i$ are the final accuracies of the data mixes at one size, with mean $\bar c\,(\mu)$, std $\sigma$, quartiles $Q_1, Q_3$ and empirical CDF $F_n$.

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
title: "Preliminary: benchmark noise"
leftLabel: Original
rightLabel: Éléonore & Clara
leftColor: blue
rightColor: green
---

### Checkpoint noise

Std. dev. over **late training checkpoints**, averaged over models and divided by the benchmark mean

$$\text{Noise} = \frac{\frac{1}{|M|}\sum_{m} \sigma_{\text{step}}(m)}{\mu(M)}$$

⚠️ Needs intermediate checkpoints, which are rarely released

::right::

### Benchmark noise

Relative std. dev. across **$k$-fold splits** of the evaluation set, averaged over models

$$\text{Noise} = \frac{1}{|M|} \sum_{m} \frac{\sqrt{\frac{1}{k}\sum_{i=1}^{k}(m_i - \bar{m})^2}}{\bar{m}} $$

✅ Needs only a **single evaluation run**

✅ Predicts decision accuracy better in Éléonore and Clara's preliminary results


---
title: Methodology
subtitle: Reliability per training stage
---

The right benchmark depends on **where in training** you are.

| Stage | Decision being made | Models available |
| ----- | ------------------- | ---------------- |
| **Pretraining** | Language count, architecture, scale | The 90M–1.7B predictivity ladder (controlled) |
| **Midtraining** | Domain/quality up-sampling | Open-source bases (3B–70B) |
| **Post-training** | SFT / DPO / RLVR recipe | Instruct families (3B–70B) |

- We compute signal, noise, SNR, DA and scaling-law error **per stage**
- Goal: **one evaluation recommendation per stage**, not a single global ranking
- Results so far cover **pretraining** only: the predictivity ladder


---
layout: section
---

# Research questions


---
title: Research questions
---

| | question | reads |
|---|---|---|
| **RQ0** | Score vs compute: which benchmarks clear chance? | the gate every other RQ depends on |
| **RQ1** | Does a benchmark rank the interventions at a small size or an early checkpoint as it does at the reference? | decision accuracy |
| **RQ2** | Which of **22 SNR definitions** predicts decision accuracy in each language, and does the choice survive a seed swap? | SNR × DA |
| **RQ3** | Does our SNR agree with AllenAI DataDecide on the shared English tasks? | cross-corpus |
| **RQ4** | Can a language or subject **subset** beat the full benchmark's SNR? | subtask sweeps |
| **RQ5** | Which **design features** (curation, format, option count, length) predict SNR? | benchmark metadata |
| **RQ6** | Which **proxy size** ranks an intervention like the reference, and how does that change with L? | *new: the question the ladder was built for* |


<!--
One entry point: `cd src/signal-and-noise && bash run_all_predictivity.sh`

Everything comes from one file: the wide per-checkpoint ladder_report.csv published to
msnr-data/ladder-report. Every number in this deck regenerates from it.
-->

---
title: Research questions
---

First section, statistics and general analysis:
1. Which benchmarks scale predictably? (Fit a scaling law, report $R^2$ and Spearman $\rho$)
2. Which benchmarks are stable? (Measured by noise)
3. Which languages gain most from a larger model? Three possible patterns: (1) high baseline and strong scaling, (2) low baseline and strong scaling, (3) low baseline and weak scaling.
4. Does the benchmark separate the models we want to compare? (Measured by signal)

Second section, decision accuracy and its surrogates:
1. How early can we predict final multilingual performance?
2. What is the smallest evaluation suite that reliably predicts full multilingual model quality?
3. Which surrogate is most informative? (The plot we said we want: minimum model size against the number of languages.)


---
layout: section
---

# Experimental setup

The predictivity ladder

---
title: "Experimental setup: the ladder"
subtitle: "Two axes, five interventions"
---

| Axis | Values |
| ---- | ------ |
| **Size** (non-embedding) | 90M · 175M · 350M · 600M · 1B · 1.7B · 3B, each trained at **5 × Chinchilla** (D = 100·N). 3B is deep only, above the 1.7B reference |
| **Languages** L | 1 · 2 · 8 · 15 · 30 · 50. English stays at 50 % (L1 is English only), only the *count* changes |
| **Intervention 1** | model **depth**: deep vs shallow at the same non-embedding size |
| **Intervention 2** | data **scheme**: A (resource-ranked) vs B (diversity-first) at L ∈ {8, 15, 30}, plus the second language at L2 and the English corpus at L1 |
| **Intervention 3** | sampling **temperature**: T = 1 vs T = 3 (AT3) at L ∈ {15, 30, 50} |
| **Interventions 4, 5** | **activation** (SwiGLU) and **optimizer** (Muon) on the deep model at L ∈ {8, 15, 30} |
| **Seeds** | 1904 everywhere, ×3 on ten replicate cells: 64 / 313 / 1904 at 175M and 600M, 28 / 1797 / 1904 at 1B |

**Outcome**: does a proxy rank the design choices like the 1.7B reference, on benchmark accuracy (gated above chance)
and on bits per byte per language (fixed validation set, 100 subsets, 95 languages).

<!--
219 cells are in the ladder report. Grid and counts: the generated block in
plan/small-to-large-predictivity-training-plan.md, written by pretrain_progress.py --plot
from launch_trainings.py so the table cannot drift. The three progress figures that follow
are drawn on the cluster and were last redrawn on 10 October.
-->

---
layout: figure
image: /ladder/pretrain_progress_plan.png
fit: contain
height: 56vh
title: Experimental setup
subtitle: The planned grid
---

---
layout: figure
image: /ladder/pretrain_progress_simple.png
fit: contain
height: 56vh
title: Experimental setup
subtitle: Progress at a glance
---

---
layout: figure
image: /ladder/pretrain_progress_detailed.png
fit: contain
height: 56vh
title: Experimental setup
subtitle: Progress per cell
---

---
title: "Experimental setup: what exists today"
subtitle: "218 of the 219 report cells finished, every rung to 1.7B at every L, and six 3B cells"
---

| | 90M | 175M | 350M | 600M | 1B | 1.7B | 3B |
|---|---|---|---|---|---|---|---|
| **complete cells** | 33 | 38 | 32 | 38 | 40 | **31** | 6 |

- Finished cells cover **L ∈ {1, 2, 8, 15, 30, 50}** at every size from 90M to 1.7B. **L = 100 was dropped from the grid**
- 1B and 1.7B are finished at all six L, with their shallow, scheme-B, AT3, SwiGLU and Muon twins. The one unfinished cell is 1.7B-L8-Muon
- No finished run is diverged or off trend: **212 models** at 90M–1.7B enter the analysis, 192 of them seed 1904
- Seed replicates: 175M and 600M at L ∈ {1, 2, 50}, 1B at L ∈ {1, 2, 30} and B-L30. These ten ×3 cells are the whole seed-noise estimate

<!--
Counts from ladder_report.csv (branch data/ladder-report, 2026-10-10) via documents/ladder-facts.json:
219 cells, 218 complete, 218 neither diverged nor off trend. The 219 are the 218 runs the grid
plans (pretrain_progress_plan.png) plus one test run outside it, lm-90M-L1-b84-swiglu-seed1904.
The progress heatmap counts something else: checkpoints on disk, planned runs only. It reads
218 of 218, because 1.7B-L8-Muon has finished since the report. 350M-L1-deep counts as done
from its converted copies on capstor: scratch purged its Megatron checkpoints on 2026-10-08,
after it was trained and evaluated. The 3B cells are read only by the
size-generalisation check. Matched pairs are finished at 1B and 1.7B, so every decision now
resolves against the 1.7B reference (RULES.md rule 9).
-->

---
layout: bullets
title: Which languages we report on
subtitle: "All 50 the ladder trains on, instead of the 12 we started with"
icon: "🌐"
---

- The deck used to show 12 hand-picked languages. Every per-language figure now covers **all 50 languages of the L50 mixture**
- Validation is wider than training: **100 subsets across 95 languages**. All 50 trained languages are in it, and the other **45 are zero-shot**
- We stop at 50 because the ladder does: L = 100 was dropped from the grid. `configs/languages.json` has the list as `groups.trained`
- Widening the view changed one reading: at 1.7B every one of the 50 trained languages keeps an above-chance benchmark

<!--
todo: lists a b, l1-l50
groups.main is kept for the historical decks. groups.trained is the 49 FineWeb-2
tags of FW_L50 in src/pretrain/data/language_sets_schemeA.json, plus English,
which comes from DCLM rather than FineWeb-2. L100 was dropped (plan/l100_data_mixture.md),
so there is no L100 list to add.
-->

---
layout: section
---

# How good are our instruments?

Four questions about the benchmarks themselves, before any results

---
layout: figure
image: /ladder/rq_a1_scaling.png
fit: contain
height: 70vh
title: "1. Which benchmarks scale predictably?"
subtitle: "Completion tasks do. Letter-format knowledge tasks do not, their reformulated versions do."
---

<!--
We fit each task's score against log parameters from 90M to 1.7B. R² says how much
of the change a straight line explains. Spearman ρ says whether the score moves the right
way. Bits per byte 0.99, xwinograd 0.92, xstorycloze 0.88, hellaswag 0.88 (median R² per
family). The reformulated knowledge tasks fit too (rf_global_mmlu_full 0.79, rf_mmlu 0.91),
while letter-format global_mmlu_full sits at 0.16. At the bottom, bangla (0.01) and ceval
(0.08) barely move with model size. truthfulqa is the odd one out: truthfulqa_mc2 fits
well (R² 0.81) with ρ = −0.83, so it gets steadily worse as models grow.
Mind the sign for bits per byte and loss. They fall as models improve, so
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
This is the denominator of every SNR we quote. Small means you get the same number when
you evaluate a slightly different checkpoint of the same run. At 1.7B the hellaswag bBPB twin
is the steadiest family, then multiblimp and the xstorycloze bBPB twin. Bits per byte comes
fourth. The BBH bBPB twins and
letter-format INCLUDE are the noisiest. The reformulated knowledge tasks are among the steadiest.
-->
---
layout: figure
image: /ladder/rq_a3_languages.png
fit: contain
height: 68vh
title: "3. Which languages benefit most from a bigger model?"
subtitle: "All 100 gain, and the ones that start hardest gain the most (ρ = 0.71)"
---

<!--
Baseline is bits per byte at 90M, gain is what a 1.7B model saves, both on the
50-language mixture. All 100 validation languages improve, by a median of 0.41
bits per byte. The three patterns the team asked about are all there, but the split
is at the median, not at an absolute floor: no language gets worse. The correlation is
the headline. Scale helps most where the model is weakest now, the
opposite of what a rich-get-richer story would predict.
-->
---
layout: figure
image: /ladder/rq_a4_signal.png
fit: contain
height: 68vh
title: "4. Does the benchmark separate the things we compare?"
subtitle: "Two bBPB twins clear their noise most (7.7× and 6.2×), then bits per byte (5.8×) and loss (5.4×). All but 5 of 80 families clear it."
---

<!--
Signal is the spread across the models we want to tell apart. Noise is the spread
within one run. The ratio is the number on the right. At 1.7B the two highest are bBPB
twins, global_piqa_nonparallel_cloze 7.7x (2 tasks) and xstorycloze 6.2x (8 tasks). Then
come bits per byte 5.8x, training loss 5.4x, hellaswag 4.0x, multiblimp 2.3x, arc 2.1x, lambada 1.9x. Only kobest,
noreval, noreval_extra, toksuite and winogrande stay under 1x: for them the benchmark
varies more between checkpoints of one model than between the models themselves.
Almost every family now clears the line, and hellaswag on 26 tasks and multiblimp on 34
sit well above it.
-->
---
layout: section
---

# Can a cheap measurement stand in for the real one?

Three questions about decision accuracy and its cheaper surrogates

---
layout: figure
image: /ladder/rq_b1_early.png
fit: contain
height: 68vh
title: "5. How early can we predict final multilingual performance?"
subtitle: "Bits per byte calls it at 20 % of training. Benchmarks never settle."
---

<!--
Decision accuracy of an early checkpoint against the same run's final ranking, at 1.7B.
Bits per byte is at 0.96 by 20 percent of the run and reaches 0.99, so four fifths of
each run add little to the ranking. Hellaswag at 0.85 is almost as early, arc starts at
0.51 and reaches 0.68, multiblimp starts at 0.66 and reaches 0.72, and the at-chance
families crawl up from 0.40. Averaged over all tasks, the suite goes 0.60, 0.64, 0.66,
0.69 across the four fractions. In practice, we could evaluate at 20 percent and free most of
the eval budget.
-->
---
layout: figure
image: /ladder/rq_b2_suite_size.png
fit: contain
height: 68vh
title: "6. What is the smallest suite that predicts full model quality?"
subtitle: "One or two tasks. Adding the other 496 makes the answer worse, not better."
---

<!--
Tasks are added best first, by how well each alone tracks the ranking that
per-language bits per byte gives. Two tasks reach rho = 0.93, and one task alone
(lambada_openai_mt_it) is already there. The whole suite of 498 reaches 0.26. 497 of the
498 suite sizes beat the whole suite, because the tail is at-chance tasks whose
scores are noise and dilute the average. Caveat: the ranking pools all 186 models from
90M to 1.7B, so it is largely a size ordering.
This is the strongest practical result in the deck: a small curated suite beats
running everything.
-->
---
layout: figure
image: /ladder/rq_b3_surrogate.png
fit: contain
height: 58vh
title: "7. Which is the most informative surrogate?"
subtitle: "Per-language bits per byte works on 79 % of 143 comparisons. Benchmarks, on 49 % of 5,398."
---

<!--
This is the minimum-model-size question, aggregated. A surrogate works when
some proxy size picks the same winner as the reference and every larger size keeps
picking it. The comparisons follow the analysis rules: trained languages only, one task
per benchmark and language (with the bBPB twins), and a benchmark must be above chance at
1.7B and at every proxy size that counts. Per-language bits per byte works on 79 percent
of 143, and 175M or smaller is already enough for 13 percent of them. Benchmarks work on
49 percent of 5,398, with 175M or smaller enough for 7 percent. Read the benchmark row
against the chance-level result later in the deck: with the reference at 1.7B, "works" only
needs 1B to agree, which a random ranking does half the time, so 49 percent is chance level.
Read the training loss and macro bits per byte rows with care: they are single
measurements, so they rest on 9 comparisons each (44 and 67 percent). The two well-powered rows are
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
title: "Finding 1: every rung, 90M included, sits on the ladder"
subtitle: "Power-law fit per language setting, red ring = off trend"
---

---
layout: bullets
title: "Finding 1: every rung, 90M included, sits on the ladder"
subtitle: "Scaling holds, and the exponent barely moves with L"
icon: "📐"
---

- Final loss falls monotonically with size at **every** language setting
- Fitted exponent **α = 0.09 – 0.11** across L (deep, scheme A): the language count barely changes how the model scales
- Residuals of the healthy rungs stay within **±0.09 nats**
- We fit **without** the smallest rung on purpose, then ask the fit to predict it

<!--
alpha per L (deep/A): L1 0.092, L2 0.098, L8 0.100, L15 0.103, L30 0.108, L50 0.109.
run__resid over healthy deep/A/seed1904 cells: max |resid| 0.087 (0.135 over every healthy cell).
Fits are per (L, arch, scheme) on the larger rungs (ladder_report.check_scaling).
-->

---
layout: figure
image: /ladder/ladder_report_loss.png
fit: contain
height: 72vh
title: "Finding 2: the 90M rung fell off the ladder, and a retrain fixed it"
subtitle: "Every run, 90M included, now trains cleanly to its end"
---

---
title: "Finding 2: why the old 90M runs diverged"
subtitle: "An optimizer timescale fixed in steps, on runs whose lengths differ 18×"
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
- Each rung trains on its own budget, so run lengths differ 18× across the ladder
- The old 90M run finished at 4,500 steps. It never reached the regime the optimizer was set up for
- Tying the averaging window to run length fixed it: final loss 5.762 → 2.778, drift 1.365 → 0.011
- 175M was held back too, +0.26 nats off trend
- So on 2026-09-23 we retrained 90M and 175M at their own batch (84 / 168), which gives them 27,000 / 25,620 steps. They now end at 2.735 and 2.566 (L2 deep, seed 1904)

<!--
Improved version of the two tables: the same numbers, as one picture and a few sentences.
The lesson for future ladders: write every optimizer timescale as a fraction of the run,
never as a step count. plan/90M-rung-anomaly.md has the three learning rates we ruled out
and the held-out BPB evidence. plan/90M-175M-batch-retrain.md has the retrain: it is the
extra steps that fix it, not the smaller batch.
-->
---

Before the retrain, nine of ten 90M runs reached their **best loss at 15–19 % of training** and got worse for the rest of the run.
It is not overfitting (single epoch), not capacity (they get *worse*), and not the learning rate (three rates spanning 5× all diverge).

AdEMAMix β₃ was fixed at a **10,000-step** timescale, while run length varies 18× across the ladder:

| rung | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|
| steps ÷ optimizer timescale | **0.45** | 0.85 | 1.7 | 2.9 | 4.6 | 8.1 |

| 90M, 2 languages | best loss | final loss | drift |
| --- | ---: | ---: | ---: |
| β₃ fixed at 10,000 steps | 4.397 | 5.762 | +1.365 |
| β₃ = 0.2 × run (900 steps) | 2.767 | **2.778** | **+0.011** |

The corrected 90M (2.778) beat the *uncorrected* 175M (2.904): 175M was held back too, and both rungs were retrained at their own batch.

<!--
plan/90M-rung-anomaly.md. The control changed nothing but beta3. Recommendation for future
ladders: express every optimizer timescale as a fraction of the run, never as a step count.
Resolved 2026-09-23: 90M and 175M retrained at batch 84 / 168 (plan/90M-175M-batch-retrain.md).
-->

---
layout: bullets
title: "Finding 3: the 90M cell that slipped through the filter is gone"
subtitle: "The retrain replaced it, and no cell is diverged or off trend today"
icon: "⚠️"
---

- The divergence test is `final > best + 0.25 nats`. The old **`90M-L2-shallow` ended 0.234 above its best**, 0.016 under the line
- So it counted as healthy, although its scaling residual was **+1.29 nats**, and it was the only 90M model in the pools
- The retrain at batch 84 replaced every old 90M run. The loader keeps only the runs at each rung's current batch
- Today **0 of 219** cells are diverged or off trend, and 33 healthy 90M cells enter the analysis

<!--
The old cell had run__diverged = 0, run__off_trend = 1, run__resid = +1.291. Since the
2026-09-23 retrain, ladder_report.on_grid (applied in snr/download/ladder.py) drops the
batch-504 runs, so the question of gating on off_trend no longer arises (RULES.md rule 10).
Counts: documents/ladder-facts.json, grid and paper_population.
-->

---
layout: figure
image: /ladder/english_vs_rest.png
fit: contain
height: 62vh
title: "Finding 4: more languages are nearly free for English"
subtitle: "Held-out bits per byte, deep / scheme A / seed 1904"
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
the language gained, red means it lost. 83 of the 100 gain on average across
sizes. The gain shrinks as the model grows: the top row is darker
than the bottom one. The biggest losers on the right are Armenian, Belarusian and
Dhivehi.
-->

---
layout: figure
image: /ladder/english_vs_rest.png
fit: contain
height: 56vh
title: English pays once, every other language keeps gaining
subtitle: "Dotted is English, solid is the median of the other 99 languages"
---

<!--
English costs 0.039 bits per byte going from 1 to 2 languages at 600M, then
0.007 for the next 48 (at 1.7B: 0.029, then 0.000). The solid lines never flatten.
This is the strongest result the ladder has so far, and it holds at every size we have finished.
-->
---

- **English**: 600M pays **+0.039 bits/byte** going 1 → 2 languages, then **+0.007** for the next 48
- **Other languages**: the non-English median falls **1.650 → 1.139** at 600M, **1.461 → 0.987** at 1.7B
- L50 beats L2 on **81 / 81 / 78 / 78 / 87 of the 99** non-English languages at 175M / 350M / 600M / 1B / 1.7B,
  median gain **+0.36 / +0.34 / +0.35 / +0.31 / +0.30 bits/byte**
- Seed noise on macro BPB is **0.004 bits/byte**, so this is an ~80× effect

<!--
From ladder_report.csv final checkpoints. English = bpb__dclm, non-English median over the
99 FineWeb-2 subsets. Seed std: median over the ten x3 cells of macro_bpb.
-->

---
layout: figure
image: /ladder/effect_vs_seed_noise.png
fit: contain
height: 58vh
title: "Finding 5: only one of our three axes clears the noise floor"
subtitle: "Median |Δ final loss| in units of the seed standard deviation"
---

---
layout: bullets
title: "Finding 5: only one of our three axes clears the noise floor"
subtitle: "Changing depth moves final loss little more than re-rolling the seed"
icon: "📉"
---

- Seed noise: **0.017 nats** on final loss, measured on the ten ×3 cells
- **Language count**: 5.2× the seed noise, a real axis
- **Data scheme** (A vs B): 2.2× on loss, marginal
- **Model depth** (deep vs shallow): **1.2×** on the aggregate loss, close to measuring the same model twice
- On *single* tasks the depth effect is about the same (median **1.3×**, 30 % of 4,562 cells above 2×). It separates models *somewhere*, just not on the headline metric
- Depth takes **half the grid**, and it buys a per-task effect we would have to hunt for

<!--
Matched pairs on healthy complete cells only: depth n=60, scheme n=40. Language axis is the
across-L range at fixed size (n=50). Per-task depth numbers: ladder-facts.json rq6.depth_over_seed_*. Regenerated by documents/figures/fig_setup.py
from ladder_report.csv, so the figure and these numbers move together from now on.
-->


---
layout: figure
image: /ladder/fig1_gate.png
fit: contain
height: 52vh
title: "Finding 6: emerged, or at chance"
subtitle: "Same six language settings and sizes: one benchmark learns, the other never leaves the chance line"
---

---
title: "Finding 6: over a third of the suite never clears chance"
subtitle: "A task enters the analysis only when its one-sided 95 % Wilson bound clears 1/#options in at least half the runs"
---

Of the **933** tasks in trained languages, **377 (40 %)** clear chance at 90M and **599 (64 %)** at 1.7B. **313 (34 %)** never do.

| at 1.7B, trained languages | letter answer | answer text (cloze) |
| :---: | :---: | :---: |
| `belebele` | 3 / 59 | **57 / 59** |
| `global_mmlu_full` | 1 / 29 | **29 / 29** |
| `include_base_44` | 3 / 36 | **29 / 36** |

- An **answer-format effect**, not an option-count one: two- and four-option tasks pass alike at 1.7B (83 / 122 against 488 / 759)
- **Never clear chance in any trained language**: 12 of the 52 benchmarks, among them MMLU and CommonsenseQA in letter format, TruthfulQA, BBH, ACP-Bench and CulturalBench
- From 90M in every language: 15 benchmarks, among them HellaSwag, MultiBLiMP and OpenBookQA
- Of the 74 tasks shared with public models, 27 are at chance at every ladder size. Public models read 24 of them at ≤ 1.7B, so this is mostly a size floor, not a flaw of the benchmarks

<!--
rq00_gate_and_curves README and paper Sec. 4.1, pool predictivity (192 runs, seed 1904):
1,696 tasks with a chance level, 933 of them in trained languages. Per-language BPB, the
bBPB twins and the generative tasks have no chance level and are never gated. Trained tasks
above chance per size: 90M 377, 175M 411, 350M 447, 600M 498, 1B 551, 1.7B 599. 356 pass
at all six sizes. The old fixed +0.05 margin is gone. The public-model comparison is
provisional: their mask still uses the +0.05 rule.
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
Improved version of the option-count table, split by language. Read it row by
row. rfgm_belebele, MultiBLiMP and HellaSwag already beat chance at 90M in every
language they cover, and rf_belebele does in 42 of its 49. Letter-format Belebele, Global-MMLU and INCLUDE, and
CulturalBench stay grey in almost every language, so they carry no signal at any size
we have trained. Blank means there is no task in that language at all. The figure
starts at 90M, since the retrained 90M rung is healthy.
-->
---
layout: focus
color: blue
icon: "📏"
---

## In 27 of the 50 trained languages, the most reliable measurement at 1.7B is now a **benchmark**, not bits per byte

<!--
rq04 top_benchmarks_per_language.csv, pool predictivity, rel_std SNR at 1.7B: the rank-1
measurement is a benchmark in 27 of 50 trained languages and the language's own BPB in 23
(bg, bn, bs, en, fi, he, hi, ja, ka, kk, ko, lt, ml, mr, ne, ru, sk, sl, sr, ta, th, ur, zh).
The 14 September deck read MultiBLiMP in 55 of 68 languages at 1B. Say the caveat out loud:
22 of the 27 winners are bBPB twins, which score the likelihood of the answer, not accuracy.
-->

---
layout: figure
image: /ladder/fig3_reliability_map.png
fit: contain
height: 70vh
title: "Finding 7: reliability is a (family × language) map"
subtitle: "Checkpoint decision accuracy at 1.7B. Grey × = removed by the gate, white = no task"
---

---
title: "Finding 7: a benchmark now beats bits per byte in about half the languages"
subtitle: "Highest-SNR above-random measurement per language (`rel_std` @ 1.7B)"
---

| | rank-1 measurement | typical SNR |
|---|---|---|
| **27 languages** | a harness benchmark, scored as bBPB in 22 of them | median 5.5, up to 13.0 (et 13.0, az 11.0, sq 10.1, lv 8.9) |
| **23 languages** | `bpb_<language>` | median 7.5 (en, ru, zh, ja, ko, hi and 17 more) |

- The gate still matters: it removes many benchmark tasks before SNR is computed, but never bits per byte
- No single family carries it. The top benchmark is INCLUDE in 15 languages, HellaSwag in 13, MultiBLiMP in 7
- 44 of the 50 top benchmarks are bBPB twins: they score the answer's bits per byte, not accuracy

<!--
The 14 September deck had multiblimp alone carrying this. Today the winners are spread over
INCLUDE, HellaSwag and MultiBLiMP, mostly as bBPB twins (rq04 README, Key findings). Present it
as "likelihood scoring makes the benchmarks usable" rather than "the benchmarks work".
-->

---
layout: figure
image: /ladder/snr_bpb_vs_benchmark.png
fit: contain
height: 60vh
title: Finding 7, read again
subtitle: "Every language keeps a benchmark, and it wins the head-to-head in 27 of 50"
---

<!--
Left bar: all 50 trained languages still have a benchmark once the above-random gate has
run. Right: best benchmark against best bits per byte, rel_std SNR at 1.7B, in all 50
languages. The benchmark wins 27 of the 50, mostly by small margins (et 13.0 against 8.9).
-->
---
layout: bullets
title: We measure best where our choices matter least
subtitle: "What that number is really counting"
icon: "🔁"
---

- In **all 50** trained languages a benchmark now survives the gate. For six months the deck said the opposite, and the new evals are what changed
- The benchmark beats bits per byte in **27 of 50**, by small margins: the largest is `bbpb_xcopa_et` at 13.0 against 8.9 for Estonian bits per byte
- No single family carries it: INCLUDE leads in 15 languages, HellaSwag in 13, MultiBLiMP in 7
- The catch is the scoring. 44 of the 50 top benchmarks are bBPB twins, so the win is for likelihood scoring as much as for the benchmarks

<!--
Numbers from rq04 top_benchmarks_per_language.csv and rq03 snr_variants_per_task.csv,
pool predictivity, variant rel_std at 1.7B. SNR here is the spread across design variants divided by the mean, so a
low score means the variants land on top of each other. The honest framing for the room:
coverage improved a lot, and most winners are bBPB twins, which score likelihood rather than accuracy.
-->

<!-- BEGIN auto:rq4-results (snr_definition_postprocess.py) -->
---
title: "RQ4. Surrogates: the SNR definition"
subtitle: "Results (auto): the most reliable benchmark per language (`rel_std` @ 1.7B)"
---

| lang | top benchmark | SNR | DA-ckpt@1.7B | BPB SNR |
|---|---|---|---|---|
| ar | `bbpb_global_piqa_parallel_cloze_apc_arab_leba` | 4.4 | 0.76 | 3.8 |
| az | `bbpb_rfgm_include_base_44_azerbaijani` | 11.0 | 0.93 | 7.7 |
| bg | `multiblimp_bul` | 3.4 | 0.76 | 4.5 |
| bn | `bbpb_rfgm_include_base_44_bengali` | 4.4 | 0.85 | 6.1 |
| bs | `bbpb_global_piqa_nonparallel_cloze_bos_latn` | 6.0 | 0.87 | 8.6 |
| ca | `bbpb_hellaswag_ca` | 7.7 | 0.98 | 7.2 |
| cs | `bbpb_multiblimp_ces` | 2.7 | 0.77 | 1.9 |
| da | `bbpb_hellaswag_da` | 3.9 | 0.85 | 3.5 |
| de | `bbpb_hellaswag_de` | 5.1 | 0.82 | 2.9 |
| el | `bbpb_rf_include_base_44_greek` | 2.5 | 0.56 | 2.2 |
| en | `hellaswag` | 5.0 | 0.80 | 13.6 |
| es | `include_v2_og_spanish_espa_a` | 7.3 | 0.72 | 5.5 |
| et | `bbpb_xcopa_et` | 13.0 | 0.91 | 8.9 |
| fa | `bbpb_rfgm_include_base_44_persian` | 5.7 | 0.80 | 4.1 |
| fi | `bbpb_include_v2_og_finnish_finland` | 3.4 | 0.73 | 4.2 |
| fr | `bbpb_hellaswag_fr` | 4.4 | 0.82 | 2.4 |
| he | `bbpb_rfgm_include_base_44_hebrew` | 6.8 | 0.91 | 6.9 |
| hi | `bbpb_hellaswag_hi` | 4.8 | 0.83 | 5.9 |
| hr | `bbpb_rfgm_include_base_44_croatian` | 8.7 | 0.94 | 7.0 |
| hu | `bbpb_include_v2_og_hungarian_hungary` | 3.9 | 0.70 | 2.6 |
| id | `bbpb_rf_m_mmlu_id` | 3.6 | 0.61 | 2.7 |
| it | `bbpb_rf_m_mmlu_it` | 3.3 | 0.56 | 2.3 |
| ja | `bbpb_rf_global_mmlu_full_ja` | 3.8 | 0.70 | 3.8 |
| ka | `bbpb_rfgm_belebele_kat_Geor` | 6.2 | 0.88 | 7.5 |
| kk | `bbpb_global_piqa_nonparallel_cloze_kaz_cyrl` | 9.0 | 0.98 | 9.5 |
| ko | `bbpb_rf_global_mmlu_full_ko` | 4.0 | 0.74 | 3.1 |
| lt | `bbpb_rfgm_include_base_44_lithuanian` | 6.4 | 0.89 | 6.9 |
| lv | `bbpb_rfgm_belebele_lvs_Latn` | 8.9 | 0.98 | 8.5 |
| ml | `bbpb_global_piqa_parallel_cloze_mal_mlym` | 6.1 | 0.89 | 7.8 |
| mr | `bbpb_rfgm_belebele_mar_Deva` | 6.5 | 1.00 | 10.3 |
| ms | `bbpb_include_v2_og_malay_malaysia` | 7.7 | 0.93 | 7.3 |
| ne | `bbpb_hellaswag_ne` | 6.8 | 0.85 | 10.9 |
| nl | `rf_m_mmlu_nl` | 3.8 | 0.66 | 2.7 |
| no | `norec_document_p2` | 4.5 | 0.62 | 3.1 |
| pl | `bbpb_multiblimp_pol` | 3.1 | 0.67 | 2.3 |
| pt | `bbpb_rf_include_base_44_portuguese` | 3.8 | 0.61 | 2.3 |
| ro | `bbpb_rf_m_mmlu_ro` | 3.3 | 0.63 | 2.9 |
| ru | `bbpb_hellaswag_ru` | 5.9 | 0.85 | 6.9 |
| sk | `bbpb_hellaswag_sk` | 4.2 | 0.96 | 5.7 |
| sl | `bbpb_multiblimp_slv` | 7.4 | 0.91 | 8.8 |
| sq | `bbpb_rfgm_include_base_44_albanian` | 10.1 | 0.91 | 9.3 |
| sr | `bbpb_hellaswag_sr` | 8.2 | 0.96 | 9.5 |
| sv | `bbpb_include_v2_og_swedish_sweden` | 4.1 | 0.63 | 3.8 |
| ta | `bbpb_rfgm_include_base_44_tamil` | 5.4 | 0.93 | 7.2 |
| th | `bbpb_xnli_th` | 2.6 | 0.77 | 3.3 |
| tr | `bbpb_rf_include_base_44_turkish` | 4.9 | 0.72 | 2.6 |
| uk | `bbpb_rf_m_mmlu_uk` | 2.9 | 0.48 | 1.5 |
| ur | `bbpb_rfgm_belebele_urd_Arab` | 7.3 | 0.98 | 10.1 |
| vi | `bbpb_rfgm_include_base_44_vietnamese` | 3.8 | 0.75 | 3.9 |
| zh | `bbpb_xstorycloze_zh` | 6.0 | 0.87 | 5.9 |

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq4-results -->

---
layout: bullets
title: "Finding 8: SNR predicts decision accuracy, but less than in English"
subtitle: "R = 0.79 in Heineman et al., 0.54 on this ladder"
icon: "❓"
---

- The best of the 22 variants is `rel_std`: mean Pearson r of log₁₀(SNR) vs DA
  = **+0.52** (DA-size, against 1.7B), **+0.56** (DA-ckpt), **+0.54** overall, over 50 languages
- The relative-spread family leads both kinds of decision: `rel_std`, `rel_mpd` and `rel_dispersion` sit within 0.02 of each other
- More eval coverage and the 1.7B reference made this **better**. The 14 September ladder gave +0.20 at best
- The top three are too close to rank, so **only the family can be claimed, not an exact variant**

<!--
Compare Heineman et al. (2025): R = 0.791 between SNR and DA on the English DataDecide
ladder. Ours covers the 50 trained languages, pool predictivity (seed 1904, every cell),
DA-size against the 1.7B reference, DA-ckpt with the proxy sizes pooled
(rq04 top_variants_overall.csv: rel_std 0.521 / 0.560 / 0.540, rel_mpd 0.520 / 0.560 / 0.540,
rel_dispersion 0.505 / 0.548 / 0.527). Earlier runs put the leader at +0.20, +0.32 and +0.20,
on a different variant each time. Claim the family, never the variant.
-->

---
layout: bullets
title: SNR predicts less here than in English
subtitle: "Plain version of the slide before"
icon: "❓"
---

- Heineman et al. report a correlation of 0.79 between SNR and decision accuracy on the English ladder
- Our best of 22 definitions reaches 0.56 when decisions are read across checkpoints
- Read across sizes, the best is 0.52, and 0.54 overall
- One family of definitions (relative spread) leads both kinds of decision
- More evaluation data and a larger reference made this number larger. The thinner ladder gave 0.20 at best
---
title: "Finding 9: the variant ranking survives a seed swap, on thin evidence"
subtitle: "Train on the replicate seeds (64/313, 28/1797 at 1B), test on seed 1904"
---

| | DA-size | DA-ckpt |
| --- | ---: | ---: |
| Spearman ρ on the global variant ranking | **+0.55** | **+0.72** |
| Pearson r between splits (all cells) | +0.62 (n = 22) | +0.29 (n = 44) |
| Per-language family agreement | 100 % (1 language) | 50 % (2 languages) |
| Per-language exact-variant agreement | 100 % (1 language) | 0 % (2 languages) |

- Recommend an SNR **family**, never an exact variant
- The per-language rows rest on **one and two languages**, too few to read
- Earlier runs of this holdout gave ρ = +0.70, +0.29 and +0.01 on DA-ckpt. It reads +0.72 now, but the splits are thin

<!--
compare_seed_splits.py, predictivity_seeds_train -> predictivity_seeds_test. The ten x3 cells
are 175M/600M at L in {1,2,50} and 1B at L1, L2 and L30 (data A and B). DA-size on the holdout
is 175M -> 600M only, because the pools stop at 600M. Retention reads 100 % / 98 %, but on
1 and 2 languages, so it is left off the table (rq03 headline_metrics.csv).
-->

---
layout: figure
image: /ladder/seed_holdout.png
fit: contain
height: 52vh
title: Does the SNR ranking survive a seed swap?
subtitle: "Pick the best variant on the replicate seeds, test it on seed 1904"
---

<!--
Improved version of the table. Blue is the checkpoint-based ranking, orange is
the size-based one. Both global rankings now hold (Spearman 0.72 and 0.55). The last
row is the one that matters for the paper: the per-language family agreement rests on
one and two languages, so we recommend a family of SNR definitions, never a single one.
Numbers from the current rq03 headline_metrics.csv.
-->
---
layout: bullets
title: "Finding 10: late-checkpoint noise understates the real noise 1.75×"
subtitle: "The standard S&N noise definition is optimistic on this ladder"
icon: "🔬"
---

- Signal-and-Noise measures noise as the spread over the **last few checkpoints** of one run
- We can also measure it over **seed replicates** of the same cell
- Median ratio **seed noise ÷ detrended checkpoint noise = 1.75**, over 4,650 (size, L, task) cells
- So an SNR computed the standard way is roughly **1.75× too optimistic** here
- Depth's median effect is **2.12×** the checkpoint noise but only **1.30×** the seed noise

<!--
rq03 effect_vs_noise.csv. Checkpoint noise is detrended first. Under WSD the final window
is still descending, so the raw std also counts that descent, and the ratio against it
would be smaller. Per size the ratio is 2.04 at 175M, 1.72 at 600M and 1.53 at 1B. This is a methodological result about
the framework rather than about our models, and worth reporting in the paper.
-->

---
title: "Finding 11: bits per byte answers the question, benchmarks do not"
subtitle: "RQ6 scores both, and every comparison now resolves against 1.7B"
---

All **4,248 intervention cells** resolve against the **1.7B** reference, now that 1B and 1.7B have matched shallow and scheme-B twins.

Data-scheme decision, scored with bits per byte on the languages both schemes train:

| proxy | L8 | L15 | L30 |
|---|---:|---:|---:|
| **90M** | 1.00 | 1.00 | 1.00 |
| **175M** | 1.00 | 1.00 | 1.00 |
| **350M** | 1.00 | 1.00 | 1.00 |
| **600M** | 1.00 | 0.83 | 1.00 |
| **1B** | 1.00 | 1.00 | **0.31** |

Every proxy picks the 1.7B scheme winner on bits per byte, except 1B at L30.

- The good news: **1,566 scaling fits** predict the 1.7B reference, and per-language BPB there is predicted from the proxy rungs to within **1.6–4.1 %** (median by L)

<!--
rq05 intervention_da_all_mono_axis.csv on predictivity_seeds (scheme_B, bpb_trained, final
checkpoint) and documents/ladder-facts.json rq6: 4,248 intervention cells, 1,566 scaling fits,
reference_size = 1.7B throughout. Median |relative error| of the scaling fit by L: 0.016 (L1),
0.027 (L2), 0.031 (L8), 0.034 (L15), 0.037 (L30), 0.041 (L50). The 14 September deck had 36
cells against 600M. The matched pairs above 600M have since finished.
-->

---
layout: figure
image: /ladder/benchmark_predictivity.png
fit: contain
height: 56vh
title: The benchmark suite can be scored, and it ranks at chance
subtitle: "Two schemes, 236–807 shared benchmark tasks per setting, one clear answer"
---

<!--
Left: the benchmark bars sit on the chance line (0.5) at both proxy rungs, 0.55 and 0.44 at
8 languages, 0.48 and 0.50 at 15, 0.40 and 0.42 at 30 (175M and 350M, against 1.7B).
Bits per byte behaves very differently: 1.00 at both rungs and every L. Right: the strict
test, a proxy that agrees and keeps agreeing at every larger size. 113 of 143 bits per byte
comparisons pass it, 2,662 of 5,398 benchmark comparisons do. That 49 percent is about
what chance would give, so read it as unreliable rather than as working.
-->
---
layout: bullets
title: What that means for the benchmark suite
subtitle: "The first direct answer to the question the ladder was built for"
icon: "🎯"
---

- The suite now has **236–807 shared task comparisons per language setting**. In August it had none
- The suite reads the scheme decision **at chance at both proxy rungs**: 0.55 and 0.44 at 8 languages, 0.48 and 0.50 at 15, 0.40 and 0.42 at 30
- A bigger proxy does not help it. Bits per byte, by contrast, is right at every proxy size from 90M to 600M, except 600M at 15 languages (0.83)
- Under the strict test (a proxy that agrees **and keeps agreeing** as models grow), **2,662 of 5,398** benchmark comparisons pass, against **113 of 143** for bits per byte. That 49 % is about chance level
- This does not mean benchmarks cannot work. At the sizes we can afford, **bits per byte carries the decision and the benchmarks do not**

<!--
Say this plainly to the room. The project's premise is that a cheap small model plus a
benchmark can stand in for an expensive large one. On the one decision we can now test at
scale, the benchmark half of that premise fails and the bits per byte half holds.
That is a result, not a setback. It is why per-language BPB was the right outcome metric
to build the plan around.
-->

---
layout: figure
image: /ladder/min_predictive_size.png
fit: contain
height: 58vh
title: How small can the proxy be?
subtitle: "Every dot is one language. Lines are the aggregate measures."
---

---
layout: figure
image: /ladder/min_predictive_per_language.png
fit: contain
height: 46vh
title: The same question, language by language
subtitle: "Smallest proxy that picks the same winner as 1.7B"
---

---
layout: bullets
title: How small can the proxy be?
subtitle: "What the two plots say"
icon: "🪜"
---

- On **depth**, no proxy below 1B keeps agreeing once there are 8 or more languages: 600M picks the other winner in every trained language. At 50 languages, 11 of the 50 have no proxy size that gets it right
- On the **scheme**, a 90M model is enough for all 5 shared languages at L8 and for 5 of 6 at L15. At L30 the 1B proxy flips, so only 8 of 26 languages keep agreeing. The scheme decision is easier to predict than the depth one
- **Macro bits per byte is not a shortcut.** On depth it fails outright at 1, 8 and 50 languages
- Training loss agrees with 1.7B on only 4 of the 9 decisions
- Benchmarks appear at every language count, and **2,662 of 5,398** reach a proxy size that agrees and keeps agreeing. Per L that is 65 of 134 at L1, 72 of 165 at L2, 337 of 721 at L8, 495 of 985 at L15, 1,029 of 1,999 at L30 and 664 of 1,394 at L50

<!--
The reference here is 1.7B at every L (documents/figures/predictivity.py). It reads the
rq pool (build_snr_pool) with the above-chance gate: 90M to 1.7B, trained languages only,
one task per benchmark and language, so a language counts only where both runs train it.
Each decision at each L still rests on one pair of runs. Depth is compared at L 1, 2, 8,
15, 30 and 50, the scheme at L 8, 15 and 30.
Two messages for the room. First, the aggregate metric we planned to decide on is no better
than the per-language ones it averages. Second, the benchmark line sits close to what chance
would give.
-->
---
layout: figure
image: /ladder/rq6_whiteboard.png
fit: contain
height: 58vh
title: The figure we actually want
subtitle: "Smallest proxy that reaches each decision accuracy and keeps it, against the number of languages"
---

---
layout: figure
image: /ladder/rq6_da_vs_size.png
fit: contain
height: 54vh
title: All the measurement behind it
subtitle: "Five proxy sizes against the 1.7B reference"
---

---
layout: bullets
title: What the curve shows now
subtitle: "Five proxy sizes against the 1.7B reference"
icon: "🧱"
---

- We have **five** proxy sizes, 90M to 1B, and the 1.7B reference above them. The 1B matched pairs that blocked the figure have finished
- Depth is the hard decision. For L ≥ 8, per-language agreement with 1.7B is 0.92–1.00 at 90M and 175M, falls to 0.00 at 600M at every L, and recovers to 0.78–1.00 at 1B. At the middle rungs the depth effect sits inside the noise
- The scheme decision behaves better: 1.00 from 90M to 350M at 8, 15 and 30 languages. It dips twice: 0.83 at 600M for 15 languages and 0.31 at 1B for 30
- Read as "reaches it and keeps it", depth needs 1B from 8 languages on. The scheme needs only 90M at 8 languages, while at 30 languages the 1B dip means no proxy keeps even 0.5
---
layout: bullets
title: "RQ3: agreement with DataDecide"
subtitle: "Our SNR against AllenAI's, on the English tasks both corpora evaluate"
icon: "🔗"
---

- The DataDecide SNR table is now pulled, so this comparison runs with the rest of the analysis
- The shared universe is small: only the English tasks both corpora evaluate, 9 after the above-random gate
- Results on the next slide (rq07_external_frameworks)

<!-- BEGIN auto:rq7-results (rq07_external_frameworks/analyze.py) -->
---
title: RQ7 — External frameworks
subtitle: "Results (auto) — cross-corpus agreement with AllenAI by pool"
---

| pool | best variant | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `predictivity` | `rel_std` | 0.75 | 0.78 | 9 |
| `predictivity_seeds` | `rel_std` | 0.84 | 0.85 | 9 |

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
title: "Finding 12: a subject subset beats the full Global-MMLU where it can be measured"
subtitle: "Cumulative SNR as subjects are added in order of their own SNR. Dashed = the full set"
---

<!-- BEGIN auto:rq8-results (smooth_subtasks.py) -->
---
title: RQ8 — Subset selection
subtitle: "Results (auto) — top subset gains (SNR: full → best subset)"
---

| case | task | size | full → best SNR | +gain |
|---|---|---|---|---|
| per_benchmark | `bbpb_paws` | 175M | 3.59 → 6.55 | +2.96 |
| per_benchmark | `bbpb_rf_belebele` | 1B | 3.17 → 5.99 | +2.82 |
| per_benchmark | `bbpb_rfgm_belebele` | 350M | 3.17 → 5.96 | +2.80 |
| per_benchmark | `bbpb_rfgm_belebele` | 175M | 3.14 → 5.86 | +2.73 |
| per_benchmark | `bbpb_paws` | 350M | 3.67 → 6.28 | +2.60 |
| per_benchmark | `bbpb_rfgm_belebele` | 1.7B | 3.48 → 6.02 | +2.54 |
| per_benchmark | `arc` | 350M | 2.86 → 5.36 | +2.51 |
| per_benchmark | `bbpb_rfgm_belebele` | 1B | 3.35 → 5.83 | +2.49 |

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
| `multiblimp-extra` | 5.86 | 2 | minimal_pair |
| `toksuite_math` | 2.35 | 4 | completion |
| `multiblimp` | 2.21 | 2 | minimal_pair |
| `global_piqa_nonparallel_cloze` | 2.06 | 2 | completion |
| `boolq` | 2.01 | 2 | classification |
| `xstorycloze` | 1.97 | 2 | completion |
| `hellaswag` | 1.84 | 4 | completion |
| `xwinograd` | 1.79 | 2 | completion |
| `piqa` | 1.61 | 2 | completion |
| `toksuite` | 1.59 | 4 | completion |
| `winogrande` | 1.43 | 2 | completion |
| `paws` | 1.41 | 2 | classification |
| `xcopa` | 1.33 | 2 | completion |
| `global_mmlu_full` | 1.26 | 4 | mcq_question_only |
| `rf_mmlu` | 1.06 | 4 | cloze_completion |
| `kobest` | 1.03 |  | nan |
| `toksuite_stem` | 1.00 | 4 | completion |
| `xnli` | 0.99 | 3 | classification |
| `rf_commonsense_qa` | 0.97 | 5 | cloze_completion |
| `rf_acp_bench_mcq` | 0.96 | 4 | cloze_completion |
| `rf_mmmlu` | 0.93 | 4 | cloze_completion |
| `lambada_openai_mt` | 0.90 |  | last_word |
| `rf_bbh_mcq` | 0.89 |  | cloze_completion |
| `include_v2_og` | 0.83 | 4 | cloze_completion |
| `rfgm_belebele` | 0.82 | 4 | statement_continuation |
| `rf_global_mmlu_full` | 0.82 | 4 | cloze_completion |
| `rf_cultural_bench_easy` | 0.81 | 4 | cloze_completion |
| `noreval_extra` | 0.75 |  | nan |
| `openbookqa` | 0.73 | 4 | completion |
| `rfgm_include_base_44` | 0.72 | 4 | statement_continuation |
| `arc` | 0.71 | 4 | cloze_completion |
| `include_v2_en` | 0.70 | 4 | cloze_completion |
| `rf_m_mmlu` | 0.68 | 4 | cloze_completion |
| `include_base_44` | 0.67 | 4 | mcq_question_only |
| `arc_mt` | 0.62 | 4 | cloze_completion |
| `global_piqa_parallel_cloze` | 0.59 | 4 | completion |
| `rf_belebele` | 0.59 | 4 | cloze_completion |
| `rf_include_base_44` | 0.58 | 4 | cloze_completion |
| `mathqa` | 0.55 | 5 | cloze_completion |
| `noreval` | 0.53 |  | nan |
| `belebele` | 0.51 | 4 | mrc_passage |

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

## 1. The 90M rung: fixed by its own batch

Retrained at batch 84 (175M at 168) with **every optimizer constant unchanged**: no run diverges, and both rungs are back in the analysis.

<!--
plan/90M-rung-anomaly.md: option 2 chosen 2026-09-23, trained and verified 2026-09-30.
The diverged batch-504 runs are dropped by ladder_report.on_grid (RULES.md rule 10).
90M now sits at most 0.14 nats from the line the larger rungs fit.
-->

---
layout: default
color: red
icon: "🌡️"
---

## 2. Sampling temperature: now an axis, not a sweep-wide choice

L100 was dropped: at T = 1, **51 of its 99 languages get under 100M tokens**, and at T = 3 the build falls 8.2 B short of the 1.7B draw.

**T = 1 vs T = 3 (AT3) is trained at L15, L30 and L50.** Its effect is 1.8× the seed noise.

<!--
The old recommendation was T = 2 sweep-wide (head/tail share ratio at L100, T=1: 16,581:1).
L100 decision and token tables: plan/l100_data_mixture.md. The A vs AT3 pair keeps the
language lists fixed, so T does not vary with L. Effect: rq03 effect_vs_noise.csv,
effect_temperature_over_seed median 1.80 (n = 3,705).
-->

---
layout: default
color: amber
icon: "🪜"
---

## 3. 1.7B: the reference, finished at every L

**31 cells**, the largest compute share of any rung: **21,114 of 53,741 node-hours** charged on CSCS (39 %).

3B adds six deep cells above it (8,017 node-hours) to check that its rankings hold one rung up.

<!--
Reference = TARGET_SIZE 1.7B (RULES.md rule 9). Node-hours: plan/compute-costs.md, "By size",
as of 2026-10-10, Azure excluded.
-->

---
layout: default
color: blue
icon: "📊"
---

## 4. Is 5 × C the right budget at L ≥ 30?

ATLAS puts the compute-optimal ratio at **137 tokens/param at L50** and 107 at L30. We train **100**.

A cheap test first: check whether the languages that bend the loss curve are the ones with a negligible token share.

<!--
Full test needs annealed endpoints, not intermediate checkpoints (WSD leaves the LR
undecayed). The plan sizes this at 12 WSD cooldown branches (350M/600M x L in {1,30,100}
x f in {0.25,0.5}), ~6% of one level. L100 is gone, so the L set needs revisiting. But the temperature table suggests the tail is data-starved rather than
under-trained. That check is a spreadsheet, not a sweep.
-->

---
layout: default
color: amber
icon: "📐"
---

## 5. Model depth sits near the seed-noise floor

Depth is **half the grid** and its effect on final loss is **1.2× a seed re-roll** (n = 60 matched pairs).

Keep it, or spend the compute on the axes that clear the noise?

<!--
Finding 5. The scheme axis is 2.2x on loss, also marginal (rq03 effect_vs_noise.csv: 1.22x
per task). The intervention was one of three candidates (tokenizer, depth, temperature).
Temperature, now trained as AT3, is the one the plan calls "most likely to show a ranking
that flips with scale".
-->

---
layout: default
color: blue
icon: "🎯"
---

## 6. Twelve benchmarks never clear chance in a trained language

MMLU and CommonsenseQA in letter format, TruthfulQA, BBH, ACP-Bench, CulturalBench and six more: at no size. Letter-format Belebele, Global-MMLU and INCLUDE pass in only a few languages.

Do we drop them from the during-training suite, or keep them as the negative control the gate needs?

<!--
rq00_gate_and_curves README: 12 of the 52 benchmarks never pass in any
trained language. Dropping them buys eval walltime at the cost of the gate's evidence.
Related: INCLUDE v2 is now scored in cloze form (include_v2_og, include_v2_en). By 175M
include_v2_en passes in 79 % of its languages, include_v2_og in 33 %. LAMBADA-MT is marked legacy upstream: switch or drop.
plan/benchmark_selection.md.
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

- **Established**: scaling holds at every rung (α ≈ 0.09–0.11 across L, residuals ≤ 0.09 nats on deep, scheme A). More languages are nearly free for English and worth ~0.3 bits/byte (median) to the other 99 validation languages. 34 % of the trained-language benchmark tasks never clear chance. The seed holdout leaves only one or two languages to compare, so we recommend an SNR *family*
- **Answered, against 1.7B**: per-language bits per byte reads the design decisions from every proxy (DA-size 0.91–0.95, 0.73 at 600M), while benchmark accuracy reads them barely above chance (0.51–0.57, chance ≈ 0.47)
- **Changed this run**: all 50 trained languages keep a benchmark after the gate, and a benchmark beats bits per byte on SNR in 27 of them. The bBPB twins carry it (22 of the 27)
- **Uncomfortable**: depth and data scheme sit near the seed-noise floor (1.2× and 2.2× on final loss), and SNR predicts decision accuracy only moderately (best `rel_std`, r = 0.52 on size decisions, 0.56 on checkpoint decisions)

---
layout: bullets
title: Where this leaves us
subtitle: "Plain version of the slide before"
icon: "✅"
---

- **Solid.** Scaling works at every rung. Adding languages costs English once and keeps helping every other language. Over a third of the benchmark suite never leaves chance
- **Better than we thought.** All 50 trained languages keep a benchmark after the gate, and one beats bits per byte in 27 of 50
- **Awkward.** Most of that improvement is the bBPB twins, not accuracy. Seed noise is 1.75× the standard checkpoint noise. Depth and scheme sit near the noise floor. SNR predicts decision accuracy only moderately
- **Open.** Why benchmarks stay near chance on decisions that bits per byte reads easily. Every comparison now resolves against 1.7B
---
layout: bullets
title: Next
subtitle: "Open items on 2026-10-10, in the order things unblock each other"
icon: "➡️"
---

1. Finish the **Muon 1B / 1.7B** evals: `lm-1.7B-L8-muon` is the one unfinished cell, so the optimizer decision does not resolve against 1.7B yet
2. Fill the **350M L1 deep** eval gap and score the promoted benchmarks (piqa, winogrande, boolq)
3. Publish, rebuild the **bBPB store**, then re-run `scripts/refresh_analysis.sh` with `FORCE=1`
4. Figures and generated blocks refresh, hand-written numbers do not: `documents/figures/facts.py` lists the ones that moved
5. Redraw the **progress figures** on the cluster (`pretrain_progress.py --plot`)


<!-- BEGIN generated signal slides (analysis/rq02_decision_accuracy/da_per_benchmark.py) -->

---
layout: section
---

# Appendix — Signal & Predictability across Sizes


---
title: Appendix — Above-random signal
subtitle: "Predictivity ladder (90M–1.7B, seed 1904) · mean score per family × size (bold = above chance in most of its tasks)"
---

| benchmark | rand | 90M | 175M | 350M | 600M | 1B | 1.7B |
|---|---|---|---|---|---|---|---|
| `bbpb_mathqa` |  | 3.08 | 2.91 | 2.82 | 2.68 | 2.54 | 2.47 |
| `bbpb_toksuite_math` |  | 3.02 | 2.88 | 2.79 | 2.59 | 2.46 | 2.23 |
| `loss` |  | 2.97 | 2.79 | 2.66 | 2.47 | 2.37 | 2.23 |
| `bbpb_toksuite_stem` |  | 2.70 | 2.60 | 2.52 | 2.33 | 2.20 | 2.05 |
| `bbpb_toxigen` |  | 2.43 | 2.37 | 2.30 | 2.12 | 2.05 | 2.04 |
| `bbpb_bbh_mcq` |  | 2.32 | 2.07 | 1.96 | 2.07 | 1.84 | 2.10 |
| `bbpb_rf_commonsense_qa` |  | 2.26 | 2.15 | 2.06 | 1.95 | 1.88 | 1.76 |
| `bbpb_belebele` |  | 2.91 | 2.33 | 2.04 | 1.65 | 1.41 | 1.25 |
| `bbpb_acp_bench_cloze` |  | 1.93 | 1.88 | 1.90 | 1.86 | 1.84 | 1.75 |
| `bbpb_multiblimp-extra` |  | 2.12 | 1.95 | 1.88 | 1.77 | 1.68 | 1.63 |
| `bbpb_noreval` |  | 1.95 | 1.83 | 1.71 | 1.57 | 1.46 | 1.34 |
| `bbpb_openbookqa` |  | 1.77 | 1.70 | 1.65 | 1.60 | 1.57 | 1.53 |
| `bpb` |  | 1.86 | 1.75 | 1.66 | 1.56 | 1.48 | 1.38 |
| `bbpb_toksuite` |  | 2.02 | 1.81 | 1.70 | 1.49 | 1.36 | 1.22 |
| `bbpb_include_v2_og` |  | 1.80 | 1.68 | 1.60 | 1.47 | 1.39 | 1.32 |
| `bbpb_rf_include_base_44` |  | 1.79 | 1.69 | 1.59 | 1.46 | 1.36 | 1.25 |
| `bbpb_multiblimp` |  | 1.63 | 1.54 | 1.49 | 1.41 | 1.36 | 1.31 |
| `bbpb_global_piqa_parallel_cloze` |  | 1.67 | 1.57 | 1.50 | 1.40 | 1.34 | 1.26 |
| `bbpb_commonsense_qa` |  | 1.65 | 1.52 | 1.40 | 1.36 | 1.37 | 1.35 |
| `bbpb_rf_m_mmlu` |  | 1.64 | 1.54 | 1.48 | 1.38 | 1.31 | 1.25 |
| `bbpb_arc_mt` |  | 1.73 | 1.61 | 1.49 | 1.36 | 1.25 | 1.14 |
| `bbpb_include_v2_en` |  | 1.68 | 1.55 | 1.47 | 1.36 | 1.28 | 1.22 |
| `bbpb_rf_mmmlu` |  | 1.60 | 1.50 | 1.44 | 1.34 | 1.29 | 1.24 |
| `bbpb_rf_global_mmlu_full` |  | 1.62 | 1.52 | 1.45 | 1.35 | 1.26 | 1.17 |
| `bbpb_bbh_cloze` |  | 1.36 | 1.47 | 1.54 | 1.57 | 1.21 | 1.09 |
| `bbpb_noreval_extra` |  | 1.46 | 1.43 | 1.35 | 1.30 | 1.22 | 1.15 |
| `bbpb_rf_cultural_bench_easy` |  | 1.65 | 1.48 | 1.37 | 1.22 | 1.13 | 1.05 |
| `bbpb_rfgm_include_base_44` |  | 1.56 | 1.44 | 1.36 | 1.25 | 1.16 | 1.07 |
| `bbpb_rf_bbh_mcq` |  | 1.49 | 1.40 | 1.35 | 1.26 | 1.18 | 1.11 |
| `bbpb_rf_mmlu` |  | 1.52 | 1.41 | 1.34 | 1.23 | 1.15 | 1.09 |
| `bbpb_arc` |  | 1.50 | 1.40 | 1.32 | 1.20 | 1.11 | 1.02 |
| `bbpb_boolq` |  | 1.47 | 1.30 | 1.20 | 1.14 | 1.21 | 1.18 |
| `bbpb_acp_bench_mcq` |  | 1.40 | 1.32 | 1.22 | 1.17 | 1.16 | 1.14 |
| `bbpb_global_mmlu_full` |  | 1.44 | 1.27 | 1.21 | 1.17 | 1.17 | 1.13 |
| `bbpb_include_base_44` |  | 1.30 | 1.26 | 1.20 | 1.19 | 1.21 | 1.20 |
| `bbpb_xcopa` |  | 1.44 | 1.34 | 1.27 | 1.17 | 1.11 | 1.03 |
| `bbpb_blend_sample` |  | 1.27 | 1.24 | 1.19 | 1.21 | 1.21 | 1.23 |
| `bbpb_xnli` |  | 1.32 | 1.26 | 1.23 | 1.19 | 1.16 | 1.13 |
| `bbpb_paws` |  | 1.28 | 1.21 | 1.18 | 1.13 | 1.10 | 1.06 |
| `bbpb_piqa` |  | 1.30 | 1.23 | 1.18 | 1.12 | 1.07 | 1.04 |
| `bbpb_cultural_bench_easy` |  | 1.15 | 1.20 | 1.15 | 1.13 | 1.16 | 1.16 |
| `bbpb_mmlu` |  | 1.29 | 1.20 | 1.14 | 1.11 | 1.10 | 1.10 |
| `bbpb_kobest` |  | 1.44 | 1.30 | 1.27 | 1.07 | 1.01 | 0.84 |
| `bbpb_global_piqa_nonparallel_cloze` |  | 1.35 | 1.24 | 1.15 | 1.03 | 0.94 | 0.84 |
| `bbpb_rf_belebele` |  | 1.23 | 1.15 | 1.10 | 1.02 | 0.95 | 0.87 |
| `bbpb_rfgm_belebele` |  | 1.16 | 1.09 | 1.04 | 0.97 | 0.92 | 0.87 |
| `bbpb_hellaswag` |  | 1.01 | 0.95 | 0.90 | 0.83 | 0.78 | 0.74 |
| `multiblimp-extra` | 0.50 | **0.79** | **0.83** | **0.86** | **0.89** | **0.91** | **0.92** |
| `bbpb_xstorycloze` |  | 0.98 | 0.92 | 0.87 | 0.80 | 0.75 | 0.70 |
| `multiblimp` | 0.50 | **0.80** | **0.81** | **0.83** | **0.84** | **0.85** | **0.87** |
| `bbpb_rf_acp_bench_mcq` |  | 0.95 | 0.89 | 0.85 | 0.78 | 0.73 | 0.69 |
| `zhoblimp` | 0.50 | **0.76** | **0.79** | **0.80** | **0.81** | **0.83** | **0.84** |
| `bbpb_truthfulqa_mc2` |  | 0.86 | 0.80 | 0.77 | 0.72 | 0.69 | 0.64 |
| `blimp_nl` | 0.50 | **0.67** | **0.70** | **0.74** | **0.77** | **0.79** | **0.80** |
| `piqa` | 0.50 | **0.63** | **0.65** | **0.67** | **0.70** | **0.72** | **0.75** |
| `turblimp` | 0.50 | **0.60** | **0.62** | **0.64** | **0.67** | **0.68** | **0.70** |
| `bbpb_truthfulqa-multi_mc1` |  | 0.73 | 0.68 | 0.65 | 0.62 | 0.59 | 0.57 |
| `toksuite` | 0.25 | **0.51** | **0.56** | **0.59** | **0.67** | **0.71** | **0.79** |
| `xwinograd` | 0.50 | 0.56 | **0.57** | **0.59** | **0.64** | **0.67** | **0.72** |
| `ibero_copa` | 0.50 |  |  |  | 0.58 | 0.61 | 0.65 |
| `boolq` | 0.50 | **0.58** | **0.59** | **0.59** | **0.60** | **0.62** | **0.65** |
| `ibero_paraphrase` | 0.50 |  |  |  | **0.54** | **0.57** | **0.60** |
| `global_piqa_nonparallel_cloze` | 0.50 | 0.53 | 0.52 | 0.54 | 0.57 | 0.59 | 0.61 |
| `xcopa` | 0.50 | 0.53 | **0.53** | **0.54** | **0.55** | **0.57** | **0.59** |
| `ibero_piqa` | 0.50 | **0.54** | **0.54** | **0.54** | **0.55** | **0.56** | **0.57** |
| `winogrande` | 0.50 | 0.51 | 0.51 | 0.51 | **0.54** | **0.58** | **0.62** |
| `xstorycloze` | 0.50 | 0.51 | 0.51 | **0.52** | **0.54** | **0.56** | **0.58** |
| `copal_id` | 0.50 |  |  |  | 0.52 | 0.52 | 0.53 |
| `kobest` | 0.50 | 0.48 | **0.50** | **0.51** | **0.51** | 0.52 | 0.54 |
| `paws` | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | **0.52** | **0.54** |
| `toksuite_stem` | 0.25 | **0.43** | **0.47** | **0.48** | **0.52** | **0.55** | **0.58** |
| `rf_french_bench_extra` | 0.50 |  |  |  | 0.50 | 0.49 | 0.50 |
| `french_bench` | 0.25 | **0.42** | **0.45** | **0.49** | **0.51** | **0.54** | **0.56** |
| `acp_bench_cloze` | 0.50 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 | 0.49 |
| `bbh_cloze` | 0.50 | 0.47 | 0.48 | 0.47 | 0.47 | 0.47 | 0.47 |
| `ibero_nli` | 0.33 |  |  |  | **0.45** | **0.46** | **0.47** |
| `toksuite_math` | 0.25 | **0.36** | **0.39** | **0.41** | **0.46** | **0.53** | **0.62** |
| `french_bench_extra` | 0.50 |  |  |  | 0.44 | 0.46 | 0.48 |
| `ibero_cola` | 0.50 | 0.47 | 0.41 | 0.43 | 0.45 | 0.48 | 0.48 |
| `toxigen` | 0.50 | 0.49 | 0.48 | 0.46 | 0.43 | 0.41 | 0.42 |
| `noreval_extra` | 0.50 | **0.45** | **0.45** | **0.46** | **0.42** | **0.42** | **0.43** |
| `m_truthfulqa_mc2` | 0.45 |  |  |  | 0.44 | 0.43 | 0.43 |
| `bbq` | 0.08 |  |  |  | **0.40** | **0.41** | **0.44** |
| `evalita_llm_extra` | 0.25 |  |  |  | 0.42 | 0.42 | 0.41 |
| `truthfulqa_mc2` | 0.45 | 0.43 | 0.42 | 0.41 | 0.40 | 0.40 | 0.40 |
| `oall_acva` | 0.50 |  |  |  | 0.41 | 0.41 | 0.41 |
| `xnli` | 0.33 | **0.37** | **0.38** | **0.39** | **0.41** | **0.42** | **0.44** |
| `truthfulqa-multi_mc2` | 0.45 |  |  |  | 0.40 | 0.39 | 0.39 |
| `evalita_llm` | 0.20 | 0.40 | 0.39 | 0.39 | 0.39 | **0.39** | **0.39** |
| `oall_mt` | 0.25 |  |  |  | **0.38** | **0.38** | **0.39** |
| `rf_acp_bench_mcq` | 0.25 | 0.36 | 0.37 | **0.38** | **0.39** | **0.40** | **0.40** |
| `rf_commonsense_qa` | 0.20 | **0.30** | **0.32** | **0.35** | **0.39** | **0.41** | **0.46** |
| `cultural_bench_hard` | 0.50 | 0.33 | 0.35 | 0.34 | 0.35 | 0.38 | 0.42 |
| `rfgm_belebele` | 0.25 | **0.32** | **0.33** | **0.34** | **0.36** | **0.38** | **0.41** |
| `ibero_siqa` | 0.33 | 0.35 | 0.35 | 0.35 | **0.35** | **0.36** | **0.37** |
| `hellaswag` | 0.25 | **0.29** | **0.30** | **0.31** | **0.34** | **0.38** | **0.43** |
| `rf_arabicmmlu` | 0.29 |  |  |  | **0.33** | **0.34** | **0.35** |
| `oall_alghafa` | 0.25 |  |  |  | 0.33 | 0.33 | 0.34 |
| `openbookqa` | 0.25 | **0.29** | **0.30** | **0.31** | **0.33** | **0.36** | **0.39** |
| `include_v2_en` | 0.25 | **0.29** | **0.30** | **0.31** | **0.33** | **0.35** | **0.37** |
| `headqa` | 0.25 |  |  |  | **0.29** | **0.31** | **0.33** |
| `rf_bbh_mcq` | 0.17 | 0.29 | 0.30 | 0.30 | **0.31** | **0.32** | **0.33** |
| `ibero_openbookqa` | 0.25 | **0.29** | **0.29** | **0.30** | **0.31** | **0.32** | **0.34** |
| `lambada_openai_mt` |  | 0.21 | 0.24 | 0.27 | 0.33 | 0.37 | 0.42 |
| `rf_mmlu` | 0.25 | **0.27** | **0.28** | **0.29** | **0.31** | **0.33** | **0.36** |
| `bangla` | 0.20 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 | 0.30 |
| `rf_belebele` | 0.25 | **0.28** | **0.28** | **0.29** | **0.29** | **0.31** | **0.32** |
| `arabicmmlu` | 0.29 |  |  |  | 0.29 | 0.30 | 0.30 |
| `rf_cultural_bench_easy` | 0.25 | 0.24 | 0.26 | 0.27 | 0.30 | 0.33 | 0.37 |
| `include_v2_og` | 0.25 | 0.27 | 0.27 | 0.28 | **0.29** | **0.31** | **0.33** |
| `cultural_bench_easy` | 0.25 | **0.38** | 0.26 | 0.29 | 0.29 | 0.25 | 0.26 |
| `rfgm_include_base_44` | 0.25 | 0.26 | 0.27 | 0.27 | 0.29 | **0.31** | **0.33** |
| `rf_oall_exams` | 0.25 |  |  |  | 0.27 | **0.28** | **0.30** |
| `rf_include_base_44` | 0.25 | 0.26 | 0.26 | 0.27 | 0.28 | **0.29** | **0.31** |
| `ibero_arc` | 0.25 | 0.26 | 0.27 | 0.27 | 0.28 | 0.29 | **0.30** |
| `arc` | 0.25 | 0.25 | 0.25 | 0.26 | 0.27 | **0.28** | **0.31** |
| `rf_mmmlu` | 0.25 | **0.26** | **0.26** | **0.26** | **0.27** | **0.28** | **0.29** |
| `rf_careqa` | 0.25 |  |  |  | 0.26 | **0.27** | **0.29** |
| `rf_m_mmlu` | 0.25 | **0.26** | **0.26** | **0.26** | **0.27** | **0.28** | **0.29** |
| `arc_mt` | 0.25 | 0.24 | 0.24 | 0.25 | 0.26 | **0.29** | **0.33** |
| `noreval` | 0.50 | 0.25 | 0.26 | 0.26 | 0.27 | 0.27 | 0.29 |
| `rf_global_mmlu_full` | 0.25 | 0.26 | **0.26** | **0.26** | **0.27** | **0.27** | **0.29** |
| `rf_oall_arabic_mmlu` | 0.25 |  |  |  | 0.25 | **0.26** | **0.27** |
| `m_truthfulqa_mc1` | 0.23 |  |  |  | **0.26** | **0.26** | **0.26** |
| `acp_bench_mcq` | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.26 |
| `careqa` | 0.25 |  |  |  | 0.25 | 0.25 | 0.26 |
| `include_base_44` | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| `oall_exams` | 0.25 |  |  |  | 0.26 | 0.25 | 0.25 |
| `mmlu` | 0.25 | 0.24 | 0.25 | 0.25 | 0.25 | 0.25 | 0.26 |
| `oall_arabic_mmlu` | 0.25 |  |  |  | 0.25 | 0.25 | 0.25 |
| `blend_sample` | 0.25 | 0.24 | 0.25 | 0.25 | 0.24 | 0.25 | 0.25 |
| `bbh_mcq` | 0.17 | 0.24 | 0.25 | 0.25 | 0.25 | 0.25 | 0.25 |
| `global_mmlu_full` | 0.25 | 0.24 | 0.25 | 0.25 | 0.25 | 0.24 | 0.25 |
| `ceval` | 0.25 | 0.24 | 0.24 | 0.25 | 0.25 | 0.24 | 0.25 |
| `m_mmlu` | 0.25 |  |  |  | 0.24 | 0.24 | 0.25 |
| `mmmlu` | 0.25 |  |  |  | 0.24 | 0.24 | 0.25 |
| `belebele` | 0.25 | 0.24 | 0.24 | 0.24 | 0.24 | 0.24 | 0.25 |
| `global_piqa_parallel_cloze` | 0.25 | 0.23 | 0.23 | 0.23 | 0.24 | 0.24 | 0.26 |
| `mathqa` | 0.20 | **0.22** | **0.22** | **0.23** | **0.23** | **0.24** | **0.25** |
| `truthfulqa-multi_mc1` | 0.23 | 0.24 | 0.24 | 0.23 | 0.23 | 0.22 | 0.22 |
| `truthfulqa_mc1` | 0.23 |  |  |  | 0.22 | 0.22 | 0.23 |
| `commonsense_qa` | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 |
| `ibero_truthfulqa` | 0.23 | 0.19 | 0.19 | 0.18 | 0.21 | 0.21 | 0.21 |
| `haerae` | 0.20 | 0.19 | 0.19 | 0.19 | 0.19 | 0.19 | 0.20 |
| `kmmlu` | 0.25 | 0.13 | 0.15 | 0.16 | 0.20 | 0.23 | 0.25 |
| `turkishmmlu` | 0.20 | 0.18 | 0.18 | 0.18 | 0.18 | 0.19 | 0.19 |
| `xquad` |  | 0.00 | 0.00 | 0.00 | 0.01 | 0.03 | 0.07 |
| `mela` |  | -0.00 | -0.01 | 0.00 | 0.00 | 0.01 | -0.00 |

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

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `acp_bench_cloze_areach` | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `bbh_cloze_sports_understanding` | **1.00** | **1.00** | **1.00** | **1.00** | **0.94** | **1.00** | **1.00** | **1.00** | **0.94** | **1.00** | **1.00** | **0.94** | **1.00** | **0.94** | **0.94** |
| `acp_bench_cloze_val` | **1.00** | **1.00** | **1.00** | **1.00** | **0.94** | **1.00** | **1.00** | **1.00** | **0.94** | **1.00** | **1.00** | **0.94** | **1.00** | **0.94** | **0.94** |
| `acp_bench_cloze_reach` | **0.94** | **0.94** | **0.94** | **0.94** | **0.94** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `acp_bench_cloze_prog` | **0.88** | **1.00** | **1.00** | **1.00** | **1.00** | **0.88** | **0.88** | **0.88** | **0.88** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `acp_bench_cloze_land` | **0.94** | **1.00** | **1.00** | **1.00** | **0.88** | **0.94** | **0.94** | **0.94** | **0.82** | **1.00** | **1.00** | **0.88** | **1.00** | **0.88** | **0.88** |
| `acp_bench_cloze_app` | **0.82** | **0.82** | **0.82** | **0.88** | **0.88** | **0.88** | **0.88** | **0.94** | **0.94** | **0.88** | **0.94** | **0.94** | **0.94** | **0.94** | **1.00** |
| `acp_bench_cloze_just` | **0.77** | **0.77** | **0.82** | **0.88** | **0.88** | **0.77** | **0.82** | **0.88** | **0.88** | **0.82** | **0.88** | **0.88** | **0.94** | **0.94** | **1.00** |
| `rf_bbh_mcq_geometric_shapes` | **0.94** | **0.82** | **0.82** | **0.88** | **0.88** | **0.88** | **0.88** | **0.94** | **0.94** | **0.76** | **0.82** | **0.82** | **0.82** | **0.82** | **0.88** |
| `bbpb_toksuite_math` | **0.82** | **0.85** | **0.83** | **0.81** | **0.80** | **0.85** | **0.82** | **0.82** | **0.82** | **0.86** | **0.87** | **0.86** | **0.89** | **0.87** | **0.87** |
| `hellaswag` | **0.86** | **0.79** | 0.73 | **0.81** | **0.80** | **0.82** | **0.80** | **0.89** | **0.86** | **0.84** | **0.81** | **0.81** | **0.80** | **0.79** | **0.92** |
| `bpb` | **0.95** | **0.77** | 0.71 | **0.89** | **0.87** | **0.79** | **0.75** | **0.91** | **0.87** | **0.79** | **0.79** | **0.77** | **0.75** | 0.70 | **0.90** |
| `bbpb_hellaswag` | **0.89** | **0.76** | **0.75** | **0.80** | **0.75** | **0.81** | **0.82** | **0.84** | **0.78** | **0.81** | **0.80** | 0.73 | **0.78** | 0.72 | **0.83** |
| `bbpb_xstorycloze` | 0.74 | 0.64 | 0.74 | **0.78** | 0.66 | **0.76** | 0.68 | **0.80** | 0.67 | 0.64 | 0.65 | 0.62 | **0.75** | 0.62 | 0.73 |
| `xstorycloze` | 0.68 | 0.69 | 0.67 | 0.69 | 0.66 | 0.71 | 0.74 | 0.72 | 0.64 | 0.74 | 0.72 | 0.69 | 0.72 | 0.68 | 0.71 |
| `lambada_openai_mt` | 0.67 | 0.70 | 0.67 | 0.68 | 0.63 | 0.67 | 0.74 | 0.72 | 0.74 | 0.68 | 0.60 | 0.57 | 0.74 | 0.67 | **0.82** |
| `bbpb_piqa` | 0.55 | 0.64 | 0.75 | **0.77** | 0.69 | 0.73 | 0.61 | 0.64 | 0.60 | 0.67 | 0.64 | 0.63 | 0.70 | 0.66 | 0.66 |
| `bbpb_rf_cultural_bench_easy_south_africa` | 0.68 | 0.69 | 0.67 | 0.69 | 0.66 | 0.64 | 0.69 | 0.71 | 0.60 | 0.68 | 0.69 | 0.65 | 0.58 | 0.55 | **0.76** |
| `rf_mmlu` | 0.70 | 0.61 | 0.57 | 0.66 | 0.67 | 0.67 | 0.57 | 0.65 | 0.61 | 0.72 | 0.72 | 0.61 | **0.81** | 0.66 | 0.67 |
| `arc_easy` | 0.60 | 0.68 | 0.71 | 0.66 | 0.72 | 0.66 | 0.63 | 0.66 | 0.60 | 0.71 | 0.71 | 0.61 | 0.65 | 0.66 | 0.62 |
| `bbpb_rf_global_mmlu_full` | 0.71 | 0.69 | 0.69 | 0.71 | 0.58 | 0.67 | 0.64 | 0.66 | 0.59 | **0.75** | 0.61 | 0.63 | 0.61 | 0.69 | 0.64 |
| `xwinograd` | 0.64 | 0.69 | 0.61 | 0.70 | 0.67 | 0.66 | 0.64 | 0.64 | 0.60 | 0.62 | 0.65 | 0.73 | 0.70 | 0.66 | 0.69 |
| `bbpb_rf_cultural_bench_easy_united_kingdom` | 0.64 | 0.71 | 0.65 | 0.67 | 0.58 | 0.67 | 0.67 | 0.64 | 0.69 | 0.73 | 0.64 | 0.67 | 0.64 | 0.59 | 0.65 |
| `rf_global_mmlu_full` | 0.64 | 0.58 | 0.61 | 0.62 | 0.54 | 0.60 | 0.63 | 0.71 | 0.69 | 0.74 | 0.64 | 0.62 | 0.71 | 0.70 | 0.70 |
| `bbpb_rf_mmlu` | 0.62 | 0.70 | 0.64 | 0.71 | 0.70 | 0.68 | 0.62 | 0.58 | 0.56 | 0.64 | 0.61 | 0.68 | 0.68 | 0.63 | 0.60 |
| `bbpb_rf_cultural_bench_easy_australia` | 0.68 | 0.74 | 0.56 | 0.68 | 0.65 | 0.66 | 0.61 | 0.60 | 0.61 | 0.61 | 0.71 | 0.68 | 0.58 | 0.67 | 0.56 |
| `bbpb_global_mmlu_full` | 0.60 | 0.66 | 0.72 | 0.64 | 0.66 | 0.68 | 0.68 | 0.62 | 0.56 | 0.72 | 0.67 | 0.55 | 0.69 | 0.58 | 0.57 |
| `piqa` | 0.61 | 0.61 | 0.60 | 0.60 | 0.60 | **0.76** | 0.55 | 0.70 | 0.62 | 0.72 | 0.69 | 0.68 | 0.56 | 0.64 | 0.64 |
| `bbh_cloze_formal_fallacies` | 0.47 | 0.71 | **0.76** | **0.88** | **0.94** | 0.37 | 0.39 | 0.40 | 0.44 | 0.62 | 0.67 | 0.72 | 0.66 | 0.71 | **0.83** |
| `bbpb_arc_challenge` | 0.61 | 0.68 | **0.76** | 0.63 | 0.65 | 0.59 | 0.63 | 0.55 | 0.62 | 0.66 | 0.64 | 0.54 | 0.62 | 0.62 | 0.60 |
| `bbpb_rf_belebele` | **0.75** | 0.62 | 0.70 | 0.67 | 0.53 | 0.62 | 0.62 | 0.69 | 0.53 | 0.67 | 0.67 | 0.57 | 0.64 | 0.55 | 0.55 |
| `bbpb_rf_cultural_bench_easy_nigeria` | 0.71 | 0.68 | 0.52 | 0.67 | 0.66 | 0.65 | 0.55 | 0.63 | 0.64 | 0.48 | 0.69 | 0.69 | 0.55 | 0.55 | 0.65 |
| `bbpb_arc_easy` | 0.64 | 0.68 | 0.73 | 0.58 | 0.62 | 0.57 | 0.62 | 0.56 | 0.61 | 0.65 | 0.65 | 0.55 | 0.63 | 0.62 | 0.57 |
| `bbpb_toksuite_stem` | 0.66 | 0.60 | 0.63 | 0.64 | 0.62 | 0.68 | 0.62 | 0.60 | 0.60 | 0.72 | 0.57 | 0.55 | 0.49 | 0.51 | 0.74 |
| `arc_challenge` | 0.59 | 0.53 | 0.52 | 0.62 | 0.62 | 0.56 | 0.67 | 0.61 | 0.60 | 0.64 | 0.65 | 0.63 | 0.63 | 0.62 | 0.71 |
| `bbpb_belebele` | 0.61 | 0.68 | 0.60 | 0.57 | 0.59 | 0.65 | 0.60 | 0.63 | 0.55 | 0.68 | 0.61 | 0.61 | 0.58 | 0.68 | 0.55 |
| `bbpb_openbookqa` | 0.57 | 0.68 | 0.57 | 0.49 | 0.61 | 0.70 | 0.70 | 0.67 | 0.55 | 0.67 | 0.56 | 0.60 | 0.59 | 0.66 | 0.46 |
| `bbpb_xnli` | 0.67 | 0.63 | 0.67 | 0.57 | 0.56 | 0.66 | 0.58 | 0.59 | 0.49 | 0.67 | 0.53 | 0.60 | 0.54 | 0.65 | 0.65 |
| `bbpb_truthfulqa-multi_mc1` | 0.69 | 0.67 | 0.62 | 0.61 | 0.54 | 0.66 | 0.54 | 0.55 | 0.52 | 0.68 | 0.60 | 0.59 | 0.68 | 0.53 | 0.50 |
| `bbpb_mmlu` | 0.52 | 0.64 | 0.68 | 0.55 | 0.56 | 0.58 | 0.65 | 0.63 | 0.59 | 0.61 | 0.60 | 0.62 | 0.63 | 0.56 | 0.54 |
| `bbpb_multiblimp` | 0.50 | 0.61 | 0.69 | 0.71 | 0.69 | 0.59 | 0.51 | 0.55 | 0.44 | 0.56 | 0.53 | 0.57 | 0.65 | 0.66 | 0.62 |
| `bbpb_truthfulqa_mc2` | 0.60 | 0.67 | 0.56 | 0.54 | 0.55 | 0.64 | 0.64 | 0.56 | 0.51 | 0.60 | 0.58 | 0.64 | 0.73 | 0.52 | 0.53 |
| `bbpb_rf_commonsense_qa` | 0.57 | 0.59 | 0.61 | 0.57 | 0.57 | 0.63 | 0.62 | 0.63 | 0.60 | 0.55 | 0.65 | 0.57 | 0.50 | 0.64 | 0.57 |
| `bbpb_paws` | 0.59 | 0.64 | 0.59 | 0.65 | 0.61 | 0.56 | 0.50 | 0.50 | 0.54 | 0.52 | 0.59 | 0.55 | 0.60 | 0.67 | 0.66 |
| `winogrande` | 0.58 | 0.53 | 0.44 | 0.61 | 0.46 | 0.51 | 0.57 | 0.65 | 0.58 | 0.58 | 0.69 | 0.58 | 0.64 | 0.69 | 0.62 |
| `bbpb_rf_cultural_bench_easy_zimbabwe` | 0.55 | 0.59 | 0.65 | 0.55 | 0.65 | 0.56 | 0.54 | 0.64 | 0.58 | 0.47 | 0.52 | 0.65 | 0.57 | 0.57 | 0.61 |
| `rf_commonsense_qa` | 0.59 | 0.69 | 0.58 | 0.64 | 0.52 | 0.54 | 0.61 | 0.51 | 0.49 | 0.65 | 0.62 | 0.47 | 0.56 | 0.58 | 0.57 |
| `bbpb_rfgm_belebele` | 0.75 | 0.67 | 0.63 | 0.48 | 0.52 | 0.61 | 0.64 | 0.49 | 0.52 | 0.69 | 0.49 | 0.47 | 0.57 | 0.49 | 0.59 |
| `bbpb_global_piqa_parallel_cloze` | 0.69 | 0.54 | 0.64 | 0.55 | 0.52 | 0.65 | 0.59 | 0.63 | 0.54 | 0.53 | 0.53 | 0.57 | 0.57 | 0.54 | 0.51 |
| `bbpb_rf_cultural_bench_easy_united_states` | 0.60 | 0.65 | 0.53 | 0.60 | 0.59 | 0.58 | 0.53 | 0.57 | 0.61 | 0.52 | 0.58 | 0.55 | 0.48 | 0.55 | 0.62 |
| `bbpb_rf_bbh_mcq_snarks` | 0.64 | 0.49 | 0.69 | 0.65 | 0.43 | 0.51 | 0.68 | 0.64 | 0.56 | 0.56 | 0.55 | 0.66 | 0.56 | 0.41 | 0.52 |
| `bbpb_rf_acp_bench_mcq_app` | 0.70 | 0.61 | 0.56 | 0.53 | 0.53 | 0.66 | 0.52 | 0.56 | 0.52 | 0.52 | 0.60 | 0.51 | 0.55 | 0.54 | 0.57 |
| `bbpb_cultural_bench_easy_zimbabwe` | 0.51 | 0.58 | 0.56 | 0.56 | 0.62 | 0.64 | 0.65 | 0.56 | 0.51 | 0.65 | 0.57 | 0.53 | 0.58 | 0.46 | 0.47 |
| `rf_belebele` | 0.45 | 0.47 | 0.55 | 0.52 | 0.47 | 0.61 | 0.64 | 0.64 | 0.59 | 0.53 | 0.60 | 0.60 | 0.64 | 0.49 | 0.56 |
| `bbpb_toksuite` | 0.65 | 0.56 | 0.53 | 0.60 | 0.42 | 0.66 | 0.46 | 0.64 | 0.54 | 0.55 | 0.54 | 0.59 | 0.54 | 0.56 | 0.49 |
| `bbpb_rf_bbh_mcq_movie_recommendation` | 0.59 | 0.47 | 0.70 | 0.49 | 0.53 | 0.60 | 0.66 | 0.56 | 0.54 | 0.46 | 0.60 | 0.49 | 0.53 | 0.58 | 0.54 |
| `bbpb_cultural_bench_easy_canada` | 0.46 | 0.56 | 0.49 | 0.51 | 0.58 | 0.58 | 0.65 | 0.66 | 0.55 | 0.63 | 0.60 | 0.53 | 0.66 | 0.40 | 0.44 |
| `bbpb_cultural_bench_easy_united_states` | 0.49 | 0.53 | 0.57 | 0.52 | 0.63 | 0.62 | 0.55 | 0.56 | 0.56 | 0.52 | 0.59 | 0.64 | 0.61 | 0.43 | 0.49 |
| `bbpb_bbh_cloze_web_of_lies` | 0.68 | 0.61 | 0.46 | 0.56 | 0.46 | 0.62 | 0.48 | 0.58 | 0.53 | 0.52 | 0.62 | 0.50 | 0.53 | 0.54 | 0.55 |
| `bbpb_rf_bbh_mcq_disambiguation_qa` | 0.59 | 0.52 | 0.59 | 0.51 | 0.55 | 0.56 | 0.53 | 0.57 | 0.52 | 0.49 | 0.61 | 0.54 | 0.64 | 0.48 | 0.51 |
| `bbpb_rf_bbh_mcq_date_understanding` | 0.61 | 0.65 | 0.55 | 0.46 | 0.45 | 0.58 | 0.54 | 0.48 | 0.53 | 0.54 | 0.56 | 0.52 | 0.51 | 0.61 | 0.62 |
| `bbpb_cultural_bench_easy_united_kingdom` | 0.50 | 0.63 | 0.56 | 0.51 | 0.60 | 0.59 | 0.61 | 0.52 | 0.50 | 0.57 | 0.54 | 0.62 | 0.53 | 0.44 | 0.49 |
| `bbpb_cultural_bench_easy_south_africa` | 0.47 | 0.53 | 0.51 | 0.57 | 0.60 | 0.60 | 0.61 | 0.57 | 0.51 | 0.59 | 0.54 | 0.56 | 0.59 | 0.47 | 0.50 |
| `bbpb_cultural_bench_easy_australia` | 0.55 | 0.54 | 0.56 | 0.50 | 0.63 | 0.61 | 0.69 | 0.51 | 0.50 | 0.56 | 0.51 | 0.59 | 0.54 | 0.41 | 0.50 |
| `bbpb_acp_bench_cloze_land` | 0.54 | 0.56 | 0.57 | 0.61 | 0.50 | 0.56 | 0.44 | 0.53 | 0.54 | 0.56 | 0.53 | 0.52 | 0.60 | 0.57 | 0.57 |
| `openbookqa` | 0.55 | 0.53 | 0.60 | 0.50 | 0.53 | 0.62 | 0.46 | 0.51 | 0.44 | 0.55 | 0.61 | 0.61 | 0.59 | 0.58 | 0.49 |
| `bbpb_rf_bbh_mcq_ruin_names` | 0.57 | 0.45 | 0.49 | 0.61 | 0.49 | 0.62 | 0.55 | 0.54 | 0.55 | 0.54 | 0.45 | 0.54 | 0.61 | 0.59 | 0.55 |
| `bbpb_commonsense_qa` | 0.59 | 0.51 | 0.60 | 0.56 | 0.48 | 0.50 | 0.51 | 0.46 | 0.62 | 0.61 | 0.61 | 0.52 | 0.54 | 0.55 | 0.48 |
| `bbpb_rf_acp_bench_mcq_areach` | 0.68 | 0.57 | 0.52 | 0.56 | 0.51 | 0.56 | 0.48 | 0.53 | 0.52 | 0.50 | 0.63 | 0.50 | 0.46 | 0.54 | 0.54 |
| `bbpb_rf_bbh_mcq_logical_deduction_five_objects` | 0.48 | 0.55 | 0.59 | 0.46 | 0.56 | 0.59 | 0.55 | 0.49 | 0.60 | 0.51 | 0.51 | 0.65 | 0.53 | 0.51 | 0.49 |
| `bbpb_blend_sample` | 0.57 | 0.52 | 0.52 | 0.43 | 0.59 | 0.48 | 0.54 | 0.57 | 0.59 | 0.60 | 0.43 | 0.56 | 0.49 | 0.62 | 0.55 |
| `bbpb_cultural_bench_easy_nigeria` | 0.55 | 0.53 | 0.57 | 0.48 | 0.59 | 0.52 | 0.67 | 0.53 | 0.50 | 0.52 | 0.52 | 0.50 | 0.62 | 0.46 | 0.49 |
| `bbpb_rf_acp_bench_mcq_prog` | 0.62 | 0.55 | 0.55 | 0.48 | 0.54 | 0.55 | 0.54 | 0.53 | 0.42 | 0.53 | 0.60 | 0.49 | 0.59 | 0.59 | 0.47 |
| `bbpb_acp_bench_mcq_val` | 0.68 | 0.57 | 0.62 | 0.42 | 0.51 | 0.59 | 0.64 | 0.49 | 0.51 | 0.65 | 0.43 | 0.40 | 0.50 | 0.46 | 0.56 |
| `mmlu` | 0.58 | 0.60 | 0.61 | 0.51 | 0.66 | 0.51 | 0.55 | 0.48 | 0.49 | 0.53 | 0.51 | 0.45 | 0.49 | 0.60 | 0.42 |
| `bbpb_bbh_mcq_movie_recommendation` | 0.49 | 0.53 | 0.50 | 0.52 | 0.46 | 0.49 | 0.55 | 0.61 | 0.59 | 0.57 | 0.50 | 0.50 | 0.54 | 0.53 | 0.63 |
| `bbpb_acp_bench_cloze_app` | 0.52 | 0.44 | 0.68 | 0.54 | 0.54 | 0.73 | 0.44 | 0.52 | 0.54 | 0.33 | 0.46 | 0.57 | 0.61 | 0.54 | 0.55 |
| `bbpb_acp_bench_mcq_areach` | 0.58 | 0.55 | 0.52 | 0.47 | 0.63 | 0.55 | 0.52 | 0.47 | 0.59 | 0.57 | 0.49 | 0.51 | 0.54 | 0.52 | 0.48 |
| `bbpb_rf_bbh_mcq_logical_deduction_seven_objects` | 0.46 | 0.56 | 0.55 | 0.47 | 0.55 | 0.54 | 0.47 | 0.49 | 0.56 | 0.56 | 0.50 | 0.69 | 0.53 | 0.55 | 0.52 |
| `bbpb_rf_bbh_mcq_logical_deduction_three_objects` | 0.52 | 0.52 | 0.63 | 0.44 | 0.53 | 0.64 | 0.49 | 0.46 | 0.63 | 0.53 | 0.48 | 0.64 | 0.48 | 0.47 | 0.49 |
| `bbpb_acp_bench_mcq_land` | 0.60 | 0.53 | 0.56 | 0.57 | 0.61 | 0.56 | 0.54 | 0.45 | 0.58 | 0.44 | 0.48 | 0.47 | 0.50 | 0.50 | 0.55 |
| `bbpb_acp_bench_mcq_just` | 0.58 | 0.60 | 0.54 | 0.51 | 0.67 | 0.52 | 0.55 | 0.48 | 0.57 | 0.49 | 0.49 | 0.50 | 0.45 | 0.50 | 0.50 |
| `rfgm_belebele` | 0.47 | 0.49 | 0.52 | 0.49 | 0.61 | 0.53 | 0.49 | 0.58 | 0.48 | 0.58 | 0.56 | 0.51 | 0.55 | 0.48 | 0.58 |
| `bbpb_acp_bench_mcq_app` | 0.54 | 0.51 | 0.51 | 0.49 | 0.64 | 0.56 | 0.57 | 0.53 | 0.53 | 0.53 | 0.49 | 0.52 | 0.50 | 0.48 | 0.51 |
| `bbpb_acp_bench_cloze_prog` | 0.64 | 0.53 | 0.49 | 0.61 | 0.49 | 0.69 | 0.44 | 0.58 | 0.59 | 0.47 | 0.47 | 0.55 | 0.44 | 0.51 | 0.42 |
| `bbpb_boolq` | 0.59 | 0.55 | 0.55 | 0.48 | 0.48 | 0.52 | 0.54 | 0.59 | 0.48 | 0.55 | 0.59 | 0.36 | 0.56 | 0.52 | 0.55 |
| `xnli` | 0.48 | 0.63 | 0.63 | 0.64 | 0.44 | 0.56 | 0.53 | 0.47 | 0.44 | 0.60 | 0.57 | 0.45 | 0.52 | 0.52 | 0.42 |
| `bbpb_acp_bench_cloze_just` | 0.52 | 0.51 | 0.52 | 0.51 | 0.50 | 0.68 | 0.44 | 0.53 | 0.56 | 0.40 | 0.53 | 0.60 | 0.50 | 0.57 | 0.53 |
| `bbpb_acp_bench_cloze_areach` | 0.53 | 0.43 | 0.62 | 0.55 | 0.59 | 0.63 | 0.43 | 0.60 | 0.57 | 0.44 | 0.46 | 0.52 | 0.51 | 0.52 | 0.47 |
| `bbpb_rf_acp_bench_mcq_just` | 0.61 | 0.53 | 0.54 | 0.45 | 0.41 | 0.56 | 0.51 | 0.49 | 0.49 | 0.48 | 0.52 | 0.64 | 0.54 | 0.54 | 0.55 |
| `bbpb_rf_bbh_mcq_penguins_in_a_table` | 0.56 | 0.52 | 0.53 | 0.62 | 0.53 | 0.43 | 0.51 | 0.44 | 0.47 | 0.64 | 0.55 | 0.54 | 0.50 | 0.56 | 0.48 |
| `bbpb_bbh_mcq_ruin_names` | 0.57 | 0.49 | 0.53 | 0.47 | 0.40 | 0.50 | 0.60 | 0.52 | 0.59 | 0.44 | 0.45 | 0.51 | 0.61 | 0.62 | 0.54 |
| `bbpb_acp_bench_mcq_prog` | 0.60 | 0.53 | 0.56 | 0.51 | 0.59 | 0.53 | 0.54 | 0.49 | 0.57 | 0.50 | 0.45 | 0.45 | 0.52 | 0.48 | 0.54 |
| `bbpb_rf_acp_bench_mcq_reach` | 0.65 | 0.56 | 0.57 | 0.57 | 0.36 | 0.60 | 0.48 | 0.55 | 0.39 | 0.45 | 0.63 | 0.47 | 0.58 | 0.50 | 0.49 |
| `paws` | 0.53 | 0.49 | 0.48 | 0.56 | 0.55 | 0.54 | 0.45 | 0.55 | 0.59 | 0.44 | 0.53 | 0.66 | 0.46 | 0.43 | 0.56 |
| `bbpb_rf_bbh_mcq_salient_translation_error_detection` | 0.58 | 0.49 | 0.53 | 0.45 | 0.56 | 0.54 | 0.41 | 0.51 | 0.58 | 0.59 | 0.49 | 0.58 | 0.46 | 0.55 | 0.51 |
| `bbpb_mathqa` | 0.46 | 0.59 | 0.65 | 0.42 | 0.63 | 0.61 | 0.44 | 0.54 | 0.51 | 0.50 | 0.38 | 0.57 | 0.52 | 0.52 | 0.46 |
| `bbpb_acp_bench_mcq_reach` | 0.59 | 0.47 | 0.56 | 0.55 | 0.62 | 0.52 | 0.51 | 0.51 | 0.57 | 0.44 | 0.46 | 0.47 | 0.52 | 0.51 | 0.54 |
| `bbpb_acp_bench_cloze_val` | 0.60 | 0.50 | 0.50 | 0.50 | 0.57 | 0.64 | 0.47 | 0.47 | 0.55 | 0.45 | 0.52 | 0.58 | 0.47 | 0.47 | 0.53 |
| `bbpb_bbh_mcq_hyperbaton` | 0.48 | 0.55 | 0.53 | 0.57 | 0.47 | 0.52 | 0.53 | 0.60 | 0.48 | 0.45 | 0.58 | 0.54 | 0.58 | 0.46 | 0.46 |
| `bbpb_rf_cultural_bench_easy_canada` | 0.54 | 0.46 | 0.47 | 0.65 | 0.43 | 0.56 | 0.55 | 0.55 | 0.53 | 0.50 | 0.56 | 0.55 | 0.40 | 0.46 | 0.61 |
| `bbpb_bbh_mcq_salient_translation_error_detection` | 0.51 | 0.40 | 0.45 | 0.46 | 0.49 | 0.59 | 0.60 | 0.47 | 0.52 | 0.62 | 0.58 | 0.60 | 0.43 | 0.54 | 0.53 |
| `toksuite_math` | 0.37 | 0.45 | 0.57 | 0.42 | 0.47 | 0.46 | 0.40 | 0.46 | 0.45 | 0.64 | 0.66 | 0.62 | 0.54 | 0.62 | 0.67 |
| `bbpb_rf_acp_bench_mcq_land` | 0.58 | 0.43 | 0.53 | 0.51 | 0.53 | 0.54 | 0.59 | 0.46 | 0.50 | 0.54 | 0.46 | 0.52 | 0.52 | 0.59 | 0.45 |
| `bbpb_bbh_mcq_date_understanding` | 0.57 | 0.61 | 0.44 | 0.48 | 0.42 | 0.56 | 0.57 | 0.46 | 0.48 | 0.48 | 0.51 | 0.56 | 0.51 | 0.58 | 0.51 |
| `bbpb_toxigen` | 0.60 | 0.52 | 0.55 | 0.52 | 0.44 | 0.64 | 0.53 | 0.57 | 0.48 | 0.47 | 0.45 | 0.51 | 0.52 | 0.53 | 0.42 |
| `blend_sample` | 0.58 | 0.47 | 0.41 | 0.55 | 0.58 | 0.48 | 0.43 | 0.59 | 0.51 | 0.46 | 0.54 | 0.50 | 0.49 | 0.56 | 0.59 |
| `bbpb_rf_bbh_mcq_tracking_shuffled_objects_seven_objects` | 0.53 | 0.48 | 0.51 | 0.62 | 0.42 | 0.62 | 0.58 | 0.55 | 0.54 | 0.57 | 0.42 | 0.56 | 0.44 | 0.36 | 0.52 |
| `bbpb_bbh_cloze_boolean_expressions` | 0.53 | 0.61 | 0.50 | 0.54 | 0.54 | 0.54 | 0.53 | 0.53 | 0.46 | 0.42 | 0.53 | 0.50 | 0.49 | 0.55 | 0.48 |
| `bbpb_rf_bbh_mcq_tracking_shuffled_objects_five_objects` | 0.49 | 0.51 | 0.52 | 0.62 | 0.41 | 0.65 | 0.57 | 0.54 | 0.49 | 0.56 | 0.46 | 0.58 | 0.44 | 0.40 | 0.49 |
| `bbpb_bbh_cloze_sports_understanding` | 0.52 | 0.50 | 0.48 | 0.55 | 0.50 | 0.56 | 0.55 | 0.51 | 0.37 | 0.63 | 0.51 | 0.54 | 0.47 | 0.53 | 0.51 |
| `global_mmlu_full` | 0.50 | 0.61 | 0.55 | 0.49 | 0.61 | 0.59 | 0.47 | 0.47 | 0.48 | 0.47 | 0.44 | 0.49 | 0.55 | 0.56 | 0.45 |
| `bbpb_bbh_cloze_causal_judgement` | 0.58 | 0.56 | 0.42 | 0.50 | 0.59 | 0.60 | 0.40 | 0.49 | 0.49 | 0.47 | 0.58 | 0.46 | 0.56 | 0.47 | 0.53 |
| `bbpb_acp_bench_cloze_reach` | 0.56 | 0.41 | 0.65 | 0.46 | 0.60 | 0.69 | 0.43 | 0.47 | 0.57 | 0.36 | 0.44 | 0.49 | 0.52 | 0.54 | 0.53 |
| `bbpb_bbh_cloze_formal_fallacies` | 0.52 | 0.55 | 0.50 | 0.48 | 0.64 | 0.62 | 0.37 | 0.58 | 0.46 | 0.44 | 0.58 | 0.42 | 0.62 | 0.46 | 0.45 |
| `toksuite_stem` | 0.53 | 0.47 | 0.58 | 0.46 | 0.56 | 0.55 | 0.59 | 0.52 | 0.50 | 0.51 | 0.42 | 0.45 | 0.48 | 0.52 | 0.55 |
| `bbpb_rf_acp_bench_mcq_val` | 0.58 | 0.48 | 0.41 | 0.48 | 0.51 | 0.54 | 0.49 | 0.48 | 0.56 | 0.60 | 0.48 | 0.56 | 0.52 | 0.47 | 0.52 |
| `bbpb_rf_bbh_mcq_temporal_sequences` | 0.52 | 0.42 | 0.59 | 0.64 | 0.49 | 0.42 | 0.60 | 0.49 | 0.60 | 0.46 | 0.54 | 0.45 | 0.53 | 0.48 | 0.44 |
| `rf_bbh_mcq_disambiguation_qa` | 0.46 | 0.56 | 0.41 | 0.54 | 0.42 | 0.61 | 0.62 | 0.43 | 0.58 | 0.53 | 0.40 | 0.54 | 0.45 | 0.60 | 0.54 |
| `bbpb_bbh_mcq_penguins_in_a_table` | 0.50 | 0.52 | 0.44 | 0.37 | 0.38 | 0.67 | 0.54 | 0.44 | 0.59 | 0.55 | 0.47 | 0.60 | 0.58 | 0.52 | 0.45 |
| `bbpb_bbh_mcq_logical_deduction_five_objects` | 0.50 | 0.44 | 0.47 | 0.45 | 0.39 | 0.51 | 0.58 | 0.49 | 0.52 | 0.49 | 0.55 | 0.54 | 0.53 | 0.61 | 0.55 |
| `bbpb_bbh_mcq_logical_deduction_seven_objects` | 0.54 | 0.44 | 0.43 | 0.44 | 0.44 | 0.48 | 0.60 | 0.41 | 0.48 | 0.51 | 0.56 | 0.57 | 0.56 | 0.53 | 0.62 |
| `bbpb_rf_bbh_mcq_geometric_shapes` | 0.50 | 0.57 | 0.42 | 0.40 | 0.54 | 0.48 | 0.56 | 0.41 | 0.47 | 0.51 | 0.53 | 0.61 | 0.49 | 0.52 | 0.54 |
| `bbpb_bbh_mcq_tracking_shuffled_objects_seven_objects` | 0.52 | 0.53 | 0.41 | 0.43 | 0.48 | 0.55 | 0.52 | 0.48 | 0.47 | 0.55 | 0.48 | 0.65 | 0.48 | 0.46 | 0.55 |
| `bbpb_rf_bbh_mcq_hyperbaton` | 0.50 | 0.54 | 0.51 | 0.51 | 0.49 | 0.51 | 0.45 | 0.50 | 0.58 | 0.42 | 0.42 | 0.59 | 0.54 | 0.50 | 0.47 |
| `bbpb_bbh_mcq_temporal_sequences` | 0.55 | 0.52 | 0.46 | 0.57 | 0.40 | 0.59 | 0.55 | 0.44 | 0.48 | 0.52 | 0.49 | 0.41 | 0.56 | 0.51 | 0.47 |
| `bbpb_bbh_mcq_tracking_shuffled_objects_five_objects` | 0.51 | 0.49 | 0.48 | 0.44 | 0.40 | 0.62 | 0.46 | 0.50 | 0.49 | 0.56 | 0.49 | 0.55 | 0.50 | 0.49 | 0.55 |
| `bbpb_rf_bbh_mcq_tracking_shuffled_objects_three_objects` | 0.47 | 0.50 | 0.52 | 0.57 | 0.42 | 0.63 | 0.52 | 0.51 | 0.53 | 0.50 | 0.47 | 0.58 | 0.47 | 0.39 | 0.45 |
| `mathqa` | 0.64 | 0.48 | 0.47 | 0.47 | 0.45 | 0.56 | 0.43 | 0.52 | 0.46 | 0.53 | 0.54 | 0.45 | 0.44 | 0.58 | 0.48 |
| `truthfulqa_mc2` | 0.49 | 0.61 | 0.53 | 0.59 | 0.48 | 0.43 | 0.48 | 0.33 | 0.37 | 0.55 | 0.54 | 0.52 | 0.55 | 0.46 | 0.52 |
| `bbpb_bbh_mcq_geometric_shapes` | 0.54 | 0.51 | 0.47 | 0.38 | 0.49 | 0.49 | 0.47 | 0.43 | 0.53 | 0.50 | 0.52 | 0.64 | 0.54 | 0.43 | 0.50 |
| `bbpb_bbh_cloze_navigate` | 0.55 | 0.50 | 0.34 | 0.49 | 0.51 | 0.60 | 0.43 | 0.49 | 0.47 | 0.50 | 0.53 | 0.45 | 0.51 | 0.54 | 0.54 |
| `toxigen` | 0.49 | 0.50 | 0.32 | 0.44 | 0.48 | 0.59 | 0.41 | 0.49 | 0.45 | 0.63 | 0.55 | 0.46 | 0.54 | 0.59 | 0.46 |
| `cultural_bench_hard_united_states` | 0.56 | 0.53 | 0.53 | 0.46 | 0.55 | 0.41 | 0.44 | 0.56 | 0.50 | 0.39 | 0.51 | 0.57 | 0.50 | 0.52 | 0.39 |
| `belebele` | 0.49 | 0.54 | 0.49 | 0.52 | 0.51 | 0.36 | 0.44 | 0.44 | 0.55 | 0.51 | 0.55 | 0.42 | 0.60 | 0.52 | 0.47 |
| `rf_bbh_mcq_movie_recommendation` | 0.54 | 0.49 | 0.56 | 0.38 | 0.48 | 0.53 | 0.53 | 0.45 | 0.58 | 0.47 | 0.52 | 0.48 | 0.47 | 0.45 | 0.45 |
| `commonsense_qa` | 0.50 | 0.54 | 0.40 | 0.43 | 0.53 | 0.55 | 0.46 | 0.47 | 0.60 | 0.46 | 0.55 | 0.52 | 0.43 | 0.49 | 0.46 |
| `bbpb_bbh_mcq_disambiguation_qa` | 0.54 | 0.41 | 0.50 | 0.46 | 0.41 | 0.49 | 0.55 | 0.45 | 0.53 | 0.45 | 0.51 | 0.43 | 0.55 | 0.55 | 0.53 |
| `bbpb_rf_bbh_mcq_reasoning_about_colored_objects` | 0.51 | 0.43 | 0.47 | 0.57 | 0.51 | 0.43 | 0.56 | 0.39 | 0.45 | 0.50 | 0.59 | 0.47 | 0.58 | 0.43 | 0.45 |
| `bbpb_bbh_mcq_reasoning_about_colored_objects` | 0.54 | 0.47 | 0.41 | 0.43 | 0.53 | 0.50 | 0.49 | 0.48 | 0.50 | 0.50 | 0.50 | 0.53 | 0.59 | 0.40 | 0.44 |
| `boolq` | 0.45 | 0.49 | 0.46 | 0.50 | 0.56 | 0.50 | 0.32 | 0.53 | 0.48 | 0.50 | 0.53 | 0.53 | 0.49 | 0.54 | 0.44 |
| `bbpb_bbh_mcq_logical_deduction_three_objects` | 0.45 | 0.48 | 0.39 | 0.45 | 0.42 | 0.48 | 0.61 | 0.48 | 0.53 | 0.30 | 0.49 | 0.53 | 0.57 | 0.56 | 0.52 |
| `cultural_bench_hard_south_africa` | 0.42 | 0.48 | 0.49 | 0.46 | 0.57 | 0.45 | 0.44 | 0.53 | 0.45 | 0.45 | 0.43 | 0.48 | 0.57 | 0.53 | 0.45 |
| `bbpb_bbh_mcq_tracking_shuffled_objects_three_objects` | 0.46 | 0.48 | 0.42 | 0.47 | 0.42 | 0.54 | 0.48 | 0.52 | 0.48 | 0.43 | 0.47 | 0.43 | 0.51 | 0.50 | 0.54 |
| `rf_bbh_mcq_hyperbaton` | 0.55 | 0.45 | 0.47 | 0.45 | 0.63 | 0.49 | 0.42 | 0.39 | 0.49 | 0.47 | 0.38 | 0.46 | 0.55 | 0.50 | 0.42 |
| `bbpb_bbh_mcq_snarks` | 0.45 | 0.43 | 0.49 | 0.55 | 0.42 | 0.49 | 0.53 | 0.50 | 0.46 | 0.39 | 0.53 | 0.53 | 0.51 | 0.40 | 0.43 |
| `global_piqa_parallel_cloze` | 0.39 | 0.45 | 0.50 | 0.41 | 0.50 | 0.47 | 0.42 | 0.40 | 0.58 | 0.54 | 0.43 | 0.47 | 0.50 | 0.49 | 0.47 |
| `cultural_bench_hard_nigeria` | 0.40 | 0.45 | 0.41 | 0.38 | 0.57 | 0.48 | 0.44 | 0.48 | 0.51 | 0.42 | 0.43 | 0.50 | 0.64 | 0.46 | 0.45 |
| `bbh_mcq_movie_recommendation` | 0.44 | 0.43 | 0.54 | 0.43 | 0.42 | 0.52 | 0.50 | 0.43 | 0.47 | 0.47 | 0.47 | 0.59 | 0.35 | 0.51 | 0.43 |
| `rf_bbh_mcq_tracking_shuffled_objects_five_objects` | 0.50 | 0.48 | 0.50 | 0.44 | 0.53 | 0.45 | 0.45 | 0.46 | 0.47 | 0.40 | 0.41 | 0.53 | 0.37 | 0.51 | 0.45 |
| `cultural_bench_easy_south_africa` | 0.44 | 0.58 | 0.43 | 0.53 | 0.44 | 0.58 | 0.45 | 0.43 | 0.41 | 0.47 | 0.45 | 0.39 | 0.50 | 0.42 | 0.41 |
| `rf_bbh_mcq_snarks` | 0.39 | 0.53 | 0.40 | 0.39 | 0.43 | 0.49 | 0.55 | 0.44 | 0.49 | 0.54 | 0.44 | 0.47 | 0.50 | 0.43 | 0.40 |
| `truthfulqa-multi_mc1` | 0.43 | 0.36 | 0.49 | 0.51 | 0.55 | 0.60 | 0.48 | 0.45 | 0.38 | 0.40 | 0.43 | 0.35 | 0.48 | 0.54 | 0.42 |
| `bbh_mcq_ruin_names` | 0.46 | 0.40 | 0.39 | 0.49 | 0.40 | 0.49 | 0.35 | 0.47 | 0.58 | 0.42 | 0.45 | 0.51 | 0.50 | 0.43 | 0.52 |
| `rf_bbh_mcq_date_understanding` | 0.48 | 0.45 | 0.46 | 0.54 | 0.48 | 0.45 | 0.43 | 0.44 | 0.35 | 0.43 | 0.44 | 0.58 | 0.45 | 0.42 | 0.42 |
| `rf_bbh_mcq_logical_deduction_three_objects` | 0.41 | 0.44 | 0.44 | 0.50 | 0.53 | 0.46 | 0.45 | 0.43 | 0.45 | 0.51 | 0.41 | 0.51 | 0.56 | 0.41 | 0.31 |
| `rf_bbh_mcq_ruin_names` | 0.51 | 0.45 | 0.46 | 0.40 | 0.42 | 0.52 | 0.41 | 0.44 | 0.41 | 0.37 | 0.44 | 0.53 | 0.47 | 0.44 | 0.48 |
| `rf_bbh_mcq_logical_deduction_seven_objects` | 0.44 | 0.35 | 0.42 | 0.41 | 0.39 | 0.46 | 0.50 | 0.45 | 0.47 | 0.46 | 0.46 | 0.46 | 0.46 | 0.50 | 0.50 |
| `acp_bench_mcq_areach` | 0.42 | 0.43 | 0.40 | 0.43 | 0.43 | 0.39 | 0.43 | 0.53 | 0.42 | 0.47 | 0.51 | 0.40 | 0.55 | 0.43 | 0.48 |
| `rf_bbh_mcq_reasoning_about_colored_objects` | 0.58 | 0.43 | 0.44 | 0.47 | 0.50 | 0.36 | 0.51 | 0.33 | 0.39 | 0.64 | 0.55 | 0.35 | 0.50 | 0.21 | 0.42 |
| `rf_acp_bench_mcq_areach` | 0.49 | 0.51 | 0.49 | 0.48 | 0.48 | 0.39 | 0.42 | 0.33 | 0.43 | 0.44 | 0.53 | 0.42 | 0.35 | 0.51 | 0.38 |
| `rf_acp_bench_mcq_just` | 0.30 | 0.47 | 0.44 | 0.44 | 0.38 | 0.49 | 0.43 | 0.41 | 0.53 | 0.47 | 0.44 | 0.38 | 0.52 | 0.40 | 0.52 |
| `bbh_mcq_snarks` | 0.48 | 0.35 | 0.49 | 0.46 | 0.41 | 0.40 | 0.48 | 0.40 | 0.43 | 0.42 | 0.40 | 0.55 | 0.39 | 0.48 | 0.49 |
| `bbh_mcq_date_understanding` | 0.35 | 0.39 | 0.45 | 0.29 | 0.40 | 0.43 | 0.45 | 0.53 | 0.57 | 0.43 | 0.56 | 0.42 | 0.46 | 0.44 | 0.44 |
| `rf_bbh_mcq_tracking_shuffled_objects_seven_objects` | 0.49 | 0.46 | 0.46 | 0.40 | 0.36 | 0.41 | 0.45 | 0.48 | 0.33 | 0.43 | 0.47 | 0.50 | 0.33 | 0.51 | 0.50 |
| `acp_bench_mcq_val` | 0.42 | 0.46 | 0.31 | 0.49 | 0.39 | 0.46 | 0.44 | 0.55 | 0.48 | 0.55 | 0.40 | 0.41 | 0.39 | 0.40 | 0.44 |
| `rf_cultural_bench_easy_canada` | 0.42 | 0.35 | 0.30 | 0.33 | 0.41 | 0.62 | 0.60 | 0.43 | 0.43 | 0.50 | 0.46 | 0.59 | 0.36 | 0.34 | 0.44 |
| `cultural_bench_easy_united_kingdom` | 0.38 | 0.52 | 0.39 | 0.53 | 0.44 | 0.58 | 0.42 | 0.40 | 0.43 | 0.44 | 0.49 | 0.49 | 0.37 | 0.33 | 0.36 |
| `bbh_mcq_logical_deduction_five_objects` | 0.39 | 0.41 | 0.45 | 0.40 | 0.39 | 0.37 | 0.45 | 0.35 | 0.39 | 0.48 | 0.50 | 0.40 | 0.49 | 0.55 | 0.53 |
| `multiblimp` | 0.53 | 0.47 | 0.43 | 0.40 | 0.42 | 0.45 | 0.47 | 0.46 | 0.39 | 0.44 | 0.46 | 0.46 | 0.38 | 0.43 | 0.37 |
| `acp_bench_mcq_app` | 0.48 | 0.37 | 0.43 | 0.44 | 0.41 | 0.52 | 0.40 | 0.50 | 0.42 | 0.42 | 0.41 | 0.46 | 0.39 | 0.42 | 0.47 |
| `cultural_bench_hard_united_kingdom` | 0.30 | 0.37 | 0.42 | 0.43 | 0.55 | 0.39 | 0.45 | 0.53 | 0.43 | 0.39 | 0.39 | 0.39 | 0.52 | 0.53 | 0.43 |
| `cultural_bench_hard_canada` | 0.45 | 0.50 | 0.45 | 0.38 | 0.50 | 0.43 | 0.46 | 0.36 | 0.44 | 0.47 | 0.33 | 0.50 | 0.38 | 0.52 | 0.32 |
| `rf_bbh_mcq_tracking_shuffled_objects_three_objects` | 0.38 | 0.56 | 0.41 | 0.50 | 0.41 | 0.40 | 0.48 | 0.46 | 0.40 | 0.34 | 0.45 | 0.44 | 0.41 | 0.36 | 0.51 |
| `acp_bench_mcq_just` | 0.51 | 0.55 | 0.32 | 0.50 | 0.40 | 0.47 | 0.32 | 0.47 | 0.44 | 0.44 | 0.42 | 0.44 | 0.47 | 0.35 | 0.38 |
| `rf_bbh_mcq_logical_deduction_five_objects` | 0.38 | 0.42 | 0.48 | 0.59 | 0.36 | 0.33 | 0.47 | 0.27 | 0.46 | 0.43 | 0.48 | 0.49 | 0.47 | 0.45 | 0.37 |
| `bbh_mcq_penguins_in_a_table` | 0.40 | 0.36 | 0.37 | 0.33 | 0.28 | 0.50 | 0.44 | 0.54 | 0.50 | 0.48 | 0.55 | 0.41 | 0.49 | 0.35 | 0.42 |
| `acp_bench_mcq_land` | 0.45 | 0.38 | 0.35 | 0.47 | 0.50 | 0.40 | 0.42 | 0.48 | 0.45 | 0.43 | 0.43 | 0.34 | 0.43 | 0.47 | 0.39 |
| `rf_acp_bench_mcq_prog` | 0.44 | 0.37 | 0.30 | 0.35 | 0.49 | 0.43 | 0.51 | 0.36 | 0.41 | 0.42 | 0.44 | 0.43 | 0.50 | 0.47 | 0.45 |
| `bbh_mcq_temporal_sequences` | 0.39 | 0.43 | 0.45 | 0.42 | 0.27 | 0.39 | 0.38 | 0.47 | 0.51 | 0.48 | 0.49 | 0.46 | 0.32 | 0.37 | 0.50 |
| `rf_acp_bench_mcq_app` | 0.38 | 0.47 | 0.42 | 0.44 | 0.50 | 0.30 | 0.44 | 0.44 | 0.44 | 0.33 | 0.44 | 0.38 | 0.49 | 0.39 | 0.48 |
| `acp_bench_mcq_prog` | 0.51 | 0.48 | 0.35 | 0.38 | 0.39 | 0.36 | 0.53 | 0.36 | 0.39 | 0.31 | 0.50 | 0.41 | 0.35 | 0.52 | 0.50 |
| `rf_acp_bench_mcq_reach` | 0.39 | 0.47 | 0.47 | 0.39 | 0.39 | 0.31 | 0.50 | 0.47 | 0.48 | 0.42 | 0.41 | 0.34 | 0.47 | 0.41 | 0.39 |
| `bbh_mcq_disambiguation_qa` | 0.39 | 0.42 | 0.39 | 0.48 | 0.34 | 0.31 | 0.38 | 0.39 | 0.41 | 0.54 | 0.44 | 0.42 | 0.47 | 0.47 | 0.45 |
| `cultural_bench_hard_australia` | 0.42 | 0.44 | 0.45 | 0.26 | 0.48 | 0.41 | 0.44 | 0.44 | 0.43 | 0.47 | 0.41 | 0.47 | 0.47 | 0.40 | 0.32 |
| `toksuite` | 0.47 | 0.46 | 0.44 | 0.48 | 0.50 | 0.33 | 0.38 | 0.29 | 0.45 | 0.42 | 0.44 | 0.39 | 0.40 | 0.45 | 0.40 |
| `cultural_bench_easy_zimbabwe` | 0.40 | 0.55 | 0.36 | 0.34 | 0.40 | 0.52 | 0.44 | 0.37 | 0.43 | 0.44 | 0.39 | 0.43 | 0.42 | 0.45 | 0.31 |
| `rf_bbh_mcq_salient_translation_error_detection` | 0.35 | 0.34 | 0.36 | 0.44 | 0.53 | 0.45 | 0.41 | 0.30 | 0.44 | 0.40 | 0.48 | 0.41 | 0.45 | 0.46 | 0.43 |
| `rf_bbh_mcq_penguins_in_a_table` | 0.44 | 0.36 | 0.45 | 0.36 | 0.36 | 0.34 | 0.45 | 0.28 | 0.57 | 0.45 | 0.52 | 0.38 | 0.43 | 0.44 | 0.38 |
| `cultural_bench_hard_zimbabwe` | 0.42 | 0.29 | 0.40 | 0.32 | 0.52 | 0.41 | 0.42 | 0.45 | 0.49 | 0.34 | 0.45 | 0.40 | 0.51 | 0.49 | 0.31 |
| `bbh_mcq_reasoning_about_colored_objects` | 0.44 | 0.34 | 0.23 | 0.30 | 0.47 | 0.39 | 0.31 | 0.38 | 0.48 | 0.44 | 0.56 | 0.55 | 0.46 | 0.32 | 0.49 |
| `acp_bench_mcq_reach` | 0.46 | 0.36 | 0.39 | 0.38 | 0.44 | 0.42 | 0.44 | 0.42 | 0.36 | 0.45 | 0.41 | 0.31 | 0.47 | 0.36 | 0.40 |
| `cultural_bench_easy_canada` | 0.39 | 0.41 | 0.33 | 0.43 | 0.41 | 0.47 | 0.44 | 0.49 | 0.31 | 0.47 | 0.47 | 0.42 | 0.44 | 0.31 | 0.31 |
| `rf_acp_bench_mcq_land` | 0.44 | 0.35 | 0.43 | 0.46 | 0.45 | 0.36 | 0.38 | 0.40 | 0.46 | 0.49 | 0.40 | 0.45 | 0.29 | 0.35 | 0.40 |
| `cultural_bench_easy_australia` | 0.38 | 0.41 | 0.35 | 0.36 | 0.34 | 0.54 | 0.47 | 0.43 | 0.44 | 0.30 | 0.41 | 0.41 | 0.37 | 0.40 | 0.42 |
| `bbh_mcq_tracking_shuffled_objects_five_objects` | 0.23 | 0.34 | 0.40 | 0.41 | 0.29 | 0.40 | 0.45 | 0.34 | 0.48 | 0.52 | 0.46 | 0.44 | 0.48 | 0.38 | 0.45 |
| `cultural_bench_easy_nigeria` | 0.42 | 0.33 | 0.29 | 0.40 | 0.35 | 0.45 | 0.46 | 0.47 | 0.35 | 0.42 | 0.44 | 0.36 | 0.43 | 0.40 | 0.47 |
| `cultural_bench_easy_united_states` | 0.38 | 0.43 | 0.36 | 0.50 | 0.38 | 0.54 | 0.40 | 0.33 | 0.47 | 0.29 | 0.37 | 0.48 | 0.43 | 0.34 | 0.33 |
| `bbh_mcq_tracking_shuffled_objects_three_objects` | 0.32 | 0.42 | 0.34 | 0.37 | 0.38 | 0.42 | 0.34 | 0.43 | 0.42 | 0.42 | 0.47 | 0.43 | 0.41 | 0.34 | 0.44 |
| `rf_acp_bench_mcq_val` | 0.41 | 0.44 | 0.36 | 0.34 | 0.38 | 0.45 | 0.53 | 0.46 | 0.25 | 0.36 | 0.34 | 0.33 | 0.47 | 0.37 | 0.43 |
| `rf_bbh_mcq_temporal_sequences` | 0.45 | 0.31 | 0.37 | 0.33 | 0.45 | 0.41 | 0.29 | 0.30 | 0.49 | 0.38 | 0.35 | 0.28 | 0.51 | 0.40 | 0.52 |
| `bbh_mcq_logical_deduction_three_objects` | 0.35 | 0.39 | 0.34 | 0.21 | 0.31 | 0.34 | 0.43 | 0.42 | 0.48 | 0.42 | 0.38 | 0.44 | 0.36 | 0.45 | 0.47 |
| `bbh_mcq_salient_translation_error_detection` | 0.33 | 0.35 | 0.26 | 0.39 | 0.26 | 0.33 | 0.28 | 0.39 | 0.43 | 0.48 | 0.45 | 0.46 | 0.41 | 0.39 | 0.54 |
| `rf_cultural_bench_easy_australia` | 0.26 | 0.32 | 0.37 | 0.23 | 0.36 | 0.41 | 0.39 | 0.36 | 0.41 | 0.48 | 0.41 | 0.47 | 0.49 | 0.38 | 0.32 |
| `bbh_mcq_logical_deduction_seven_objects` | 0.38 | 0.27 | 0.28 | 0.26 | 0.28 | 0.36 | 0.34 | 0.38 | 0.39 | 0.50 | 0.36 | 0.44 | 0.42 | 0.53 | 0.39 |
| `bbh_cloze_causal_judgement` | 0.39 | 0.49 | 0.42 | 0.39 | 0.32 | 0.35 | 0.35 | 0.31 | 0.31 | 0.31 | 0.35 | 0.40 | 0.42 | 0.36 | 0.30 |
| `bbh_mcq_tracking_shuffled_objects_seven_objects` | 0.27 | 0.32 | 0.25 | 0.36 | 0.30 | 0.41 | 0.33 | 0.32 | 0.37 | 0.45 | 0.30 | 0.50 | 0.39 | 0.51 | 0.35 |
| `bbh_cloze_web_of_lies` | 0.36 | 0.46 | 0.36 | 0.30 | 0.29 | 0.37 | 0.33 | 0.38 | 0.30 | 0.46 | 0.40 | 0.30 | 0.38 | 0.35 | 0.38 |
| `bbh_cloze_boolean_expressions` | 0.34 | 0.41 | 0.43 | 0.39 | 0.30 | 0.24 | 0.44 | 0.29 | 0.28 | 0.35 | 0.36 | 0.45 | 0.54 | 0.27 | 0.33 |
| `bbh_cloze_navigate` | 0.32 | 0.55 | 0.21 | 0.29 | 0.38 | 0.49 | 0.31 | 0.39 | 0.29 | 0.32 | 0.47 | 0.41 | 0.36 | 0.31 | 0.30 |
| `bbh_mcq_hyperbaton` | 0.41 | 0.37 | 0.27 | 0.31 | 0.32 | 0.37 | 0.36 | 0.26 | 0.32 | 0.31 | 0.45 | 0.37 | 0.42 | 0.35 | 0.43 |
| `rf_cultural_bench_easy_united_kingdom` | 0.26 | 0.41 | 0.33 | 0.23 | 0.29 | 0.35 | 0.36 | 0.39 | 0.38 | 0.34 | 0.38 | 0.38 | 0.42 | 0.28 | 0.38 |
| `rf_cultural_bench_easy_south_africa` | 0.37 | 0.37 | 0.35 | 0.39 | 0.21 | 0.38 | 0.31 | 0.24 | 0.29 | 0.42 | 0.42 | 0.36 | 0.43 | 0.29 | 0.36 |
| `bbh_mcq_geometric_shapes` | 0.32 | 0.20 | 0.21 | 0.23 | 0.27 | 0.32 | 0.26 | 0.28 | 0.20 | 0.48 | 0.57 | 0.45 | 0.41 | 0.47 | 0.40 |
| `rf_cultural_bench_easy_zimbabwe` | 0.39 | 0.38 | 0.36 | 0.41 | 0.34 | 0.40 | 0.28 | 0.23 | 0.30 | 0.42 | 0.26 | 0.36 | 0.25 | 0.32 | 0.34 |
| `rf_cultural_bench_easy_nigeria` | 0.38 | 0.27 | 0.29 | 0.31 | 0.35 | 0.30 | 0.36 | 0.35 | 0.34 | 0.39 | 0.30 | 0.34 | 0.26 | 0.35 | 0.29 |
| `rf_cultural_bench_easy_united_states` | 0.38 | 0.39 | 0.31 | 0.26 | 0.42 | 0.25 | 0.27 | 0.40 | 0.31 | 0.34 | 0.30 | 0.29 | 0.23 | 0.37 | 0.33 |

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

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.93** | **0.84** | **0.81** | **0.87** | **0.97** | **0.91** | **0.84** | **0.90** | **0.92** | **0.87** | **0.89** | **0.84** | **0.88** | **0.81** | **0.87** |
| `bbpb_hellaswag` | **0.93** | **0.86** | **0.83** | **0.84** | **0.91** | **0.89** | **0.87** | **0.89** | **0.86** | **0.86** | **0.87** | **0.85** | **0.84** | **0.81** | **0.80** |
| `bbpb_xstorycloze` | **0.88** | **0.84** | **0.85** | **0.86** | **0.90** | **0.93** | **0.85** | **0.86** | **0.88** | **0.84** | **0.85** | **0.84** | **0.84** | **0.83** | **0.82** |
| `bbpb_rfgm_belebele_ars_Arab` | **0.87** | **0.82** | **0.83** | **0.79** | **0.77** | **0.88** | **0.77** | **0.81** | **0.78** | 0.73 | **0.79** | **0.78** | 0.72 | 0.72 | **0.82** |
| `bbpb_rfgm_belebele_arz_Arab` | **0.81** | **0.82** | **0.90** | 0.71 | **0.78** | **0.83** | **0.82** | **0.75** | **0.75** | **0.76** | 0.72 | 0.72 | **0.76** | **0.81** | **0.80** |
| `bbpb_rfgm_belebele_apc_Arab` | **0.81** | **0.79** | **0.88** | **0.77** | 0.71 | **0.75** | **0.80** | **0.78** | 0.68 | **0.77** | **0.81** | **0.75** | **0.78** | 0.70 | **0.79** |
| `bbpb_rfgm_belebele_arb_Arab` | **0.90** | **0.82** | 0.72 | **0.84** | 0.69 | **0.83** | 0.68 | **0.79** | 0.71 | 0.72 | **0.82** | 0.69 | 0.72 | 0.68 | 0.70 |
| `bbpb_rf_belebele_ary_Arab` | **0.79** | **0.78** | **0.80** | 0.68 | 0.70 | 0.72 | **0.79** | 0.66 | 0.54 | 0.72 | 0.71 | 0.74 | **0.75** | 0.62 | 0.67 |
| `bbpb_rfgm_belebele_ary_Arab` | 0.68 | 0.68 | 0.74 | 0.72 | 0.72 | 0.72 | 0.68 | 0.69 | 0.64 | 0.65 | 0.71 | 0.66 | **0.78** | **0.79** | **0.77** |
| `bbpb_rfgm_include_base_44` | 0.65 | **0.80** | 0.63 | 0.62 | 0.68 | **0.75** | 0.72 | 0.71 | 0.71 | 0.68 | 0.74 | 0.68 | **0.82** | 0.72 | 0.68 |
| `bbpb_rf_belebele_ars_Arab` | **0.79** | 0.74 | 0.72 | **0.78** | 0.65 | 0.72 | 0.72 | **0.80** | 0.68 | 0.62 | 0.70 | 0.62 | 0.68 | 0.64 | 0.69 |
| `bbpb_xnli` | 0.67 | 0.72 | **0.78** | **0.79** | 0.60 | **0.78** | 0.67 | 0.66 | 0.67 | **0.78** | **0.82** | 0.57 | **0.78** | 0.58 | 0.57 |
| `bbpb_rf_belebele_arz_Arab` | **0.81** | **0.76** | **0.77** | 0.72 | 0.55 | **0.75** | **0.76** | 0.74 | 0.59 | **0.76** | 0.61 | 0.56 | 0.65 | 0.63 | 0.67 |
| `bbpb_rf_belebele_apc_Arab` | **0.81** | **0.83** | 0.68 | 0.72 | 0.59 | **0.79** | 0.74 | **0.78** | 0.58 | 0.70 | 0.64 | 0.59 | 0.62 | 0.56 | 0.62 |
| `bbpb_global_piqa_parallel_cloze_apc_arab_pale` | 0.68 | 0.70 | 0.71 | 0.72 | **0.79** | 0.67 | 0.57 | 0.64 | 0.62 | 0.54 | 0.72 | 0.68 | 0.60 | 0.72 | **0.80** |
| `bbpb_global_piqa_parallel_cloze_apc_arab_jord` | 0.65 | 0.73 | 0.66 | 0.65 | 0.72 | 0.65 | 0.62 | 0.70 | 0.57 | 0.61 | **0.77** | 0.68 | 0.68 | 0.69 | **0.75** |
| `hellaswag` | 0.60 | 0.47 | 0.52 | 0.47 | 0.44 | **0.78** | 0.72 | 0.74 | 0.71 | 0.73 | **0.75** | **0.77** | **0.78** | 0.71 | **0.76** |
| `bbpb_global_piqa_parallel_cloze_arb_arab` | 0.73 | 0.63 | 0.72 | 0.57 | 0.58 | **0.75** | **0.85** | 0.68 | 0.57 | **0.75** | 0.62 | 0.62 | 0.62 | 0.60 | 0.59 |
| `bbpb_global_piqa_parallel_cloze_apc_arab_leba` | **0.77** | **0.75** | 0.68 | 0.62 | 0.57 | 0.72 | 0.64 | 0.65 | 0.56 | 0.66 | 0.65 | 0.68 | 0.59 | 0.67 | 0.69 |
| `bbpb_global_piqa_parallel_cloze_apc_arab_syri` | 0.70 | 0.72 | 0.61 | 0.56 | 0.58 | 0.70 | 0.71 | 0.64 | 0.60 | 0.57 | 0.68 | 0.65 | 0.65 | 0.64 | **0.84** |
| `bbpb_rf_mmmlu` | 0.56 | **0.82** | 0.59 | 0.57 | 0.65 | 0.66 | 0.72 | **0.76** | 0.53 | 0.68 | 0.63 | 0.62 | **0.76** | 0.71 | 0.62 |
| `bbpb_global_piqa_parallel_cloze_ars_arab` | 0.69 | 0.66 | 0.59 | 0.60 | 0.65 | **0.83** | 0.58 | 0.66 | 0.61 | 0.68 | 0.71 | 0.59 | 0.59 | 0.64 | 0.62 |
| `bbpb_multiblimp` | 0.61 | 0.65 | 0.68 | 0.74 | 0.63 | 0.74 | 0.51 | 0.72 | 0.62 | 0.48 | 0.72 | 0.67 | 0.61 | 0.58 | 0.62 |
| `bbpb_global_piqa_parallel_cloze_arz_arab` | 0.58 | 0.68 | 0.61 | 0.57 | 0.58 | 0.64 | 0.53 | 0.67 | 0.63 | 0.72 | 0.71 | 0.61 | 0.66 | 0.68 | 0.63 |
| `bbpb_rf_m_mmlu` | 0.57 | **0.82** | 0.56 | 0.54 | 0.63 | 0.65 | 0.66 | **0.76** | 0.53 | 0.62 | 0.56 | 0.63 | 0.68 | 0.66 | 0.61 |
| `bbpb_arc` | 0.61 | 0.68 | 0.56 | 0.62 | 0.66 | 0.70 | 0.58 | 0.68 | 0.70 | 0.58 | 0.50 | 0.65 | 0.65 | 0.60 | 0.63 |
| `bbpb_rf_global_mmlu_full` | 0.66 | 0.56 | **0.75** | 0.59 | 0.70 | 0.63 | 0.61 | **0.75** | 0.72 | 0.61 | 0.45 | 0.54 | 0.49 | 0.63 | 0.68 |
| `bbpb_rf_include_base_44` | 0.68 | **0.81** | 0.51 | 0.46 | 0.62 | **0.75** | 0.62 | 0.58 | 0.62 | 0.55 | 0.48 | 0.64 | 0.65 | 0.69 | 0.62 |
| `bbpb_rf_belebele_arb_Arab` | 0.68 | 0.61 | 0.73 | **0.77** | 0.53 | 0.71 | 0.65 | 0.65 | 0.51 | 0.49 | 0.49 | 0.48 | **0.75** | 0.49 | 0.57 |
| `bbpb_global_piqa_parallel_cloze_ary_arab` | 0.57 | 0.49 | 0.58 | 0.54 | 0.59 | 0.68 | 0.69 | 0.67 | 0.62 | 0.64 | 0.73 | 0.58 | 0.62 | 0.49 | 0.55 |
| `bbpb_include_v2_og_arabic_uae` | 0.69 | 0.71 | 0.51 | 0.44 | 0.54 | 0.63 | 0.63 | 0.57 | 0.57 | 0.53 | 0.48 | 0.53 | 0.62 | 0.70 | **0.77** |
| `bbpb_include_v2_og_arabic_qatar` | 0.61 | 0.68 | 0.58 | **0.75** | 0.67 | 0.53 | 0.62 | 0.66 | 0.44 | 0.40 | 0.52 | 0.60 | 0.55 | 0.57 | 0.68 |
| `bbpb_include_v2_og_arabic_jordan` | 0.57 | **0.78** | 0.56 | 0.61 | 0.66 | 0.53 | 0.64 | 0.74 | 0.36 | 0.49 | 0.53 | 0.68 | 0.62 | 0.48 | 0.48 |
| `bbpb_cultural_bench_easy_saudi_arabia` | 0.57 | 0.47 | 0.60 | 0.71 | 0.56 | 0.60 | 0.63 | 0.64 | 0.61 | 0.60 | 0.56 | 0.47 | 0.59 | 0.42 | 0.63 |
| `bbpb_belebele_arz_Arab` | 0.49 | 0.47 | 0.72 | 0.50 | 0.62 | 0.63 | 0.68 | 0.54 | 0.60 | 0.53 | 0.62 | 0.52 | 0.65 | 0.59 | 0.49 |
| `bbpb_belebele_apc_Arab` | 0.52 | 0.46 | 0.72 | 0.50 | 0.64 | 0.66 | 0.66 | 0.53 | 0.57 | 0.53 | 0.59 | 0.47 | 0.66 | 0.57 | 0.53 |
| `bbpb_belebele_ars_Arab` | 0.50 | 0.49 | 0.71 | 0.54 | 0.62 | 0.61 | 0.68 | 0.51 | 0.57 | 0.52 | 0.62 | 0.46 | 0.65 | 0.59 | 0.54 |
| `multiblimp` | 0.67 | 0.47 | 0.57 | **0.75** | 0.57 | 0.51 | 0.56 | 0.61 | 0.48 | 0.65 | 0.50 | 0.38 | 0.63 | 0.54 | 0.70 |
| `bbpb_global_mmlu_full` | 0.59 | 0.42 | 0.62 | 0.62 | 0.40 | 0.50 | 0.66 | 0.68 | 0.62 | 0.44 | 0.61 | 0.54 | 0.63 | 0.50 | 0.72 |
| `bbpb_cultural_bench_easy_morocco` | 0.58 | 0.47 | 0.53 | 0.61 | 0.68 | 0.60 | 0.51 | 0.66 | 0.55 | 0.56 | 0.56 | 0.53 | 0.65 | 0.47 | 0.56 |
| `bbpb_belebele_ary_Arab` | 0.49 | 0.44 | 0.66 | 0.49 | 0.57 | 0.67 | 0.62 | 0.58 | 0.55 | 0.57 | 0.58 | 0.53 | 0.67 | 0.55 | 0.48 |
| `bbpb_belebele_arb_Arab` | 0.47 | 0.49 | 0.68 | 0.59 | 0.58 | 0.60 | 0.64 | 0.50 | 0.59 | 0.51 | 0.55 | 0.42 | 0.69 | 0.55 | 0.54 |
| `arc` | 0.35 | 0.57 | 0.60 | 0.55 | 0.59 | 0.44 | 0.44 | 0.52 | 0.50 | 0.62 | 0.64 | 0.62 | 0.61 | 0.64 | 0.70 |
| `bbpb_include_v2_og_arabic_egypt` | 0.67 | 0.72 | 0.44 | 0.58 | 0.53 | 0.54 | 0.59 | 0.70 | 0.45 | 0.45 | 0.53 | 0.53 | 0.66 | 0.49 | 0.43 |
| `bbpb_include_v2_en_arabic_ksa` | 0.47 | 0.62 | 0.48 | 0.63 | 0.66 | 0.46 | 0.49 | 0.46 | 0.55 | 0.53 | 0.62 | 0.61 | 0.52 | 0.61 | 0.61 |
| `bbpb_include_v2_og_arabic_ksa` | 0.62 | 0.62 | 0.62 | 0.49 | 0.51 | 0.54 | 0.69 | 0.67 | 0.45 | 0.55 | 0.46 | 0.71 | 0.49 | 0.49 | 0.40 |
| `xstorycloze` | 0.47 | 0.48 | 0.57 | 0.47 | 0.62 | 0.44 | 0.57 | **0.80** | 0.47 | 0.67 | 0.53 | 0.47 | 0.66 | 0.51 | 0.53 |
| `bbpb_include_v2_og_arabic_kuwait` | 0.68 | 0.67 | 0.49 | 0.56 | 0.49 | 0.64 | 0.55 | 0.65 | 0.43 | 0.42 | 0.54 | 0.53 | 0.73 | 0.50 | 0.37 |
| `bbpb_include_v2_en_arabic_egypt` | 0.42 | 0.57 | 0.45 | 0.42 | 0.62 | 0.49 | 0.45 | 0.56 | 0.35 | 0.62 | 0.70 | 0.68 | 0.68 | 0.65 | 0.57 |
| `global_mmlu_full` | 0.46 | 0.52 | 0.49 | 0.57 | 0.58 | 0.64 | 0.57 | 0.50 | 0.59 | 0.49 | 0.42 | 0.53 | 0.55 | 0.57 | 0.71 |
| `bbpb_include_v2_en_arabic_palestine` | 0.38 | 0.64 | 0.55 | 0.53 | 0.65 | 0.48 | 0.49 | 0.49 | 0.34 | 0.59 | 0.66 | 0.64 | 0.63 | 0.55 | 0.55 |
| `bbpb_include_v2_og_arabic_palestine` | 0.60 | **0.77** | 0.38 | 0.47 | 0.57 | 0.55 | 0.53 | 0.71 | 0.38 | 0.57 | 0.46 | 0.53 | 0.62 | 0.55 | 0.44 |
| `bbpb_include_v2_en_arabic_uae` | 0.52 | 0.60 | 0.48 | 0.57 | 0.68 | 0.47 | 0.47 | 0.67 | 0.58 | 0.65 | 0.52 | 0.42 | 0.53 | 0.48 | 0.50 |
| `bbpb_include_v2_en_arabic_jordan` | 0.29 | 0.68 | 0.53 | 0.54 | 0.67 | 0.38 | 0.46 | 0.60 | 0.34 | 0.60 | 0.64 | 0.65 | 0.56 | 0.65 | 0.54 |
| `bbpb_include_v2_en_arabic_qatar` | 0.36 | 0.63 | 0.49 | 0.51 | 0.73 | 0.42 | 0.47 | 0.65 | 0.44 | 0.59 | 0.62 | 0.57 | 0.58 | 0.51 | 0.54 |
| `bbpb_include_base_44` | 0.62 | 0.44 | 0.43 | 0.49 | 0.45 | 0.62 | 0.57 | 0.59 | 0.57 | 0.46 | 0.57 | 0.54 | 0.56 | 0.62 | 0.59 |
| `bbpb_rfgm_belebele_arb_Latn` | 0.54 | 0.59 | 0.44 | 0.67 | 0.47 | 0.55 | 0.67 | 0.69 | 0.53 | 0.37 | 0.53 | 0.61 | 0.53 | 0.54 | 0.38 |
| `bbpb_rf_belebele_arb_Latn` | 0.68 | 0.53 | 0.44 | 0.74 | 0.50 | 0.50 | 0.49 | 0.72 | 0.47 | 0.71 | 0.49 | 0.42 | 0.50 | 0.36 | 0.47 |
| `include_v2_en_arabic_jordan` | 0.60 | 0.68 | 0.57 | 0.29 | 0.51 | 0.47 | 0.52 | 0.48 | 0.62 | 0.65 | 0.48 | 0.47 | 0.55 | 0.56 | 0.57 |
| `include_v2_og_arabic_jordan` | 0.53 | 0.53 | 0.59 | 0.64 | 0.38 | 0.52 | 0.60 | 0.59 | 0.47 | 0.62 | 0.56 | 0.47 | 0.48 | 0.58 | 0.40 |
| `rf_global_mmlu_full` | 0.53 | 0.50 | 0.53 | 0.53 | 0.59 | 0.56 | 0.49 | 0.52 | 0.50 | 0.54 | 0.65 | 0.42 | 0.46 | 0.50 | 0.62 |
| `bbpb_belebele_arb_Latn` | 0.56 | 0.51 | 0.51 | 0.57 | 0.49 | 0.42 | 0.50 | 0.53 | 0.55 | 0.47 | 0.63 | 0.55 | 0.52 | 0.60 | 0.53 |
| `belebele_arb_Arab` | 0.36 | 0.59 | 0.53 | 0.64 | 0.53 | 0.52 | 0.52 | 0.29 | 0.37 | 0.47 | 0.56 | 0.53 | 0.64 | 0.60 | 0.72 |
| `bbpb_rf_cultural_bench_easy_saudi_arabia` | 0.53 | 0.63 | 0.57 | 0.31 | 0.47 | 0.45 | 0.62 | 0.51 | 0.46 | 0.56 | 0.53 | 0.61 | 0.47 | 0.57 | 0.43 |
| `rf_m_mmlu` | 0.51 | 0.52 | 0.41 | 0.31 | 0.39 | 0.56 | 0.53 | 0.46 | 0.48 | 0.61 | 0.50 | 0.56 | 0.72 | 0.65 | 0.50 |
| `bbpb_include_v2_og_arabic_morocco` | 0.48 | 0.58 | 0.47 | 0.55 | 0.38 | 0.58 | 0.67 | 0.57 | 0.39 | 0.50 | 0.38 | 0.49 | 0.63 | 0.54 | 0.42 |
| `bbpb_rf_cultural_bench_easy_morocco` | 0.31 | 0.43 | 0.56 | 0.46 | 0.46 | 0.59 | 0.52 | 0.57 | 0.57 | 0.51 | 0.47 | 0.59 | 0.38 | 0.50 | 0.72 |
| `include_base_44` | 0.41 | 0.46 | 0.55 | 0.33 | 0.63 | 0.57 | 0.57 | 0.49 | 0.54 | 0.57 | 0.53 | 0.49 | 0.52 | 0.53 | 0.40 |
| `rfgm_include_base_44` | 0.50 | 0.43 | 0.50 | 0.62 | 0.55 | 0.57 | 0.47 | 0.51 | 0.53 | 0.37 | 0.49 | 0.51 | 0.50 | 0.43 | 0.56 |
| `include_v2_en_arabic_qatar` | 0.56 | 0.42 | 0.57 | 0.52 | 0.49 | 0.58 | 0.44 | 0.50 | 0.42 | 0.48 | 0.56 | 0.47 | 0.49 | 0.56 | 0.45 |
| `bbpb_include_v2_en_arabic_morocco` | 0.70 | 0.58 | 0.33 | 0.54 | 0.38 | 0.52 | 0.42 | 0.42 | 0.40 | 0.50 | 0.39 | 0.62 | 0.62 | 0.63 | 0.44 |
| `bbpb_include_v2_en_arabic_kuwait` | 0.34 | 0.57 | 0.48 | 0.47 | 0.53 | 0.54 | 0.53 | 0.57 | 0.38 | 0.33 | 0.68 | 0.57 | 0.36 | 0.52 | 0.59 |
| `rf_belebele_ars_Arab` | 0.53 | 0.55 | 0.50 | 0.58 | 0.52 | 0.47 | 0.38 | 0.57 | 0.52 | 0.58 | 0.48 | 0.42 | 0.44 | 0.50 | 0.41 |
| `blend_sample` | 0.44 | 0.46 | 0.28 | 0.44 | 0.55 | 0.66 | 0.54 | 0.42 | 0.58 | 0.52 | 0.52 | 0.41 | 0.51 | 0.53 | 0.53 |
| `rfgm_belebele_ars_Arab` | 0.45 | 0.62 | 0.42 | 0.40 | 0.26 | 0.68 | 0.48 | 0.54 | 0.68 | 0.43 | 0.44 | 0.47 | 0.38 | 0.42 | 0.63 |
| `include_v2_og_arabic_morocco` | 0.40 | 0.59 | 0.47 | 0.50 | 0.30 | 0.58 | 0.62 | 0.48 | 0.43 | 0.55 | 0.43 | 0.33 | 0.50 | 0.64 | 0.48 |
| `rf_belebele_arz_Arab` | 0.32 | 0.42 | 0.34 | 0.53 | 0.44 | 0.60 | 0.64 | 0.53 | 0.53 | 0.51 | 0.55 | 0.55 | 0.42 | 0.53 | 0.38 |
| `bbpb_blend_sample` | 0.58 | 0.53 | 0.54 | 0.39 | 0.62 | 0.34 | 0.39 | 0.56 | 0.56 | 0.65 | 0.33 | 0.57 | 0.28 | 0.62 | 0.32 |
| `rf_mmmlu` | 0.41 | 0.47 | 0.52 | 0.42 | 0.42 | 0.42 | 0.53 | 0.63 | 0.43 | 0.29 | 0.61 | 0.62 | 0.47 | 0.45 | 0.57 |
| `belebele_apc_Arab` | 0.49 | 0.61 | 0.47 | 0.51 | 0.45 | 0.42 | 0.49 | 0.28 | 0.42 | 0.40 | 0.64 | 0.59 | 0.47 | 0.41 | 0.61 |
| `belebele_ary_Arab` | 0.44 | 0.44 | 0.58 | 0.55 | 0.50 | 0.49 | 0.62 | 0.18 | 0.50 | 0.49 | 0.49 | 0.49 | 0.38 | 0.47 | 0.50 |
| `include_v2_en_arabic_egypt` | 0.42 | 0.55 | 0.52 | 0.25 | 0.60 | 0.53 | 0.51 | 0.47 | 0.47 | 0.53 | 0.37 | 0.62 | 0.53 | 0.40 | 0.35 |
| `include_v2_og_arabic_egypt` | 0.25 | 0.38 | 0.57 | 0.40 | 0.46 | 0.44 | 0.35 | 0.51 | 0.62 | 0.46 | 0.57 | 0.40 | 0.52 | 0.64 | 0.53 |
| `include_v2_og_arabic_palestine` | 0.43 | 0.42 | 0.47 | 0.54 | 0.35 | 0.55 | 0.57 | 0.57 | 0.44 | 0.34 | 0.62 | 0.47 | 0.47 | 0.36 | 0.47 |
| `belebele_ars_Arab` | 0.39 | 0.61 | 0.43 | 0.54 | 0.51 | 0.46 | 0.52 | 0.27 | 0.57 | 0.28 | 0.55 | 0.51 | 0.49 | 0.49 | 0.45 |
| `include_v2_en_arabic_palestine` | 0.58 | 0.41 | 0.53 | 0.48 | 0.57 | 0.44 | 0.49 | 0.30 | 0.55 | 0.56 | 0.34 | 0.38 | 0.46 | 0.42 | 0.56 |
| `rf_belebele_arb_Latn` | 0.36 | 0.27 | 0.40 | 0.49 | 0.49 | 0.53 | 0.50 | 0.57 | 0.54 | 0.59 | 0.42 | 0.51 | 0.35 | 0.57 | 0.47 |
| `rfgm_belebele_arz_Arab` | 0.51 | 0.27 | 0.62 | 0.29 | 0.53 | 0.59 | 0.36 | 0.47 | 0.38 | 0.41 | 0.63 | 0.42 | 0.51 | 0.67 | 0.38 |
| `include_v2_en_arabic_morocco` | 0.58 | 0.47 | 0.38 | 0.43 | 0.26 | 0.54 | 0.39 | 0.33 | 0.40 | 0.53 | 0.34 | 0.59 | 0.62 | 0.62 | 0.54 |
| `rfgm_belebele_apc_Arab` | 0.43 | 0.64 | 0.48 | 0.43 | 0.50 | 0.48 | 0.45 | 0.47 | 0.52 | 0.42 | 0.45 | 0.59 | 0.45 | 0.38 | 0.32 |
| `rfgm_belebele_arb_Latn` | 0.46 | 0.35 | 0.62 | 0.50 | 0.54 | 0.48 | 0.53 | 0.32 | 0.38 | 0.47 | 0.35 | 0.38 | 0.61 | 0.57 | 0.47 |
| `global_piqa_parallel_cloze_ars_arab` | 0.28 | 0.51 | 0.44 | 0.47 | 0.47 | 0.46 | 0.33 | 0.47 | 0.62 | 0.41 | 0.65 | 0.47 | 0.49 | 0.49 | 0.43 |
| `belebele_arz_Arab` | 0.23 | 0.40 | 0.39 | 0.51 | 0.62 | 0.52 | 0.62 | 0.39 | 0.35 | 0.40 | 0.50 | 0.47 | 0.63 | 0.47 | 0.51 |
| `include_v2_en_arabic_ksa` | 0.53 | 0.50 | 0.42 | 0.63 | 0.48 | 0.53 | 0.53 | 0.43 | 0.49 | 0.37 | 0.29 | 0.49 | 0.46 | 0.42 | 0.42 |
| `rf_belebele_apc_Arab` | 0.65 | 0.52 | 0.32 | 0.69 | 0.41 | 0.47 | 0.37 | 0.58 | 0.36 | 0.51 | 0.47 | 0.51 | 0.24 | 0.46 | 0.45 |
| `include_v2_og_arabic_qatar` | 0.41 | 0.47 | 0.42 | 0.38 | 0.52 | 0.44 | 0.48 | 0.34 | 0.48 | 0.42 | 0.55 | 0.52 | 0.46 | 0.46 | 0.56 |
| `xnli` | 0.33 | 0.40 | 0.56 | 0.63 | 0.42 | 0.49 | 0.56 | 0.38 | 0.38 | 0.53 | 0.47 | 0.36 | 0.57 | 0.31 | 0.53 |
| `rf_belebele_ary_Arab` | 0.37 | 0.26 | 0.63 | 0.46 | 0.39 | 0.42 | 0.49 | 0.56 | 0.29 | 0.41 | 0.51 | 0.60 | 0.57 | 0.47 | 0.45 |
| `rfgm_belebele_arb_Arab` | 0.48 | 0.47 | 0.50 | 0.42 | 0.29 | 0.61 | 0.52 | 0.55 | 0.48 | 0.33 | 0.47 | 0.45 | 0.40 | 0.43 | 0.45 |
| `rfgm_belebele_ary_Arab` | 0.55 | 0.48 | 0.33 | 0.65 | 0.35 | 0.45 | 0.41 | 0.47 | 0.31 | 0.38 | 0.46 | 0.47 | 0.42 | 0.62 | 0.49 |
| `rf_belebele_arb_Arab` | 0.41 | 0.34 | 0.40 | 0.45 | 0.57 | 0.54 | 0.53 | 0.59 | 0.32 | 0.71 | 0.48 | 0.23 | 0.44 | 0.35 | 0.45 |
| `belebele_arb_Latn` | 0.47 | 0.40 | 0.44 | 0.47 | 0.59 | 0.42 | 0.42 | 0.33 | 0.65 | 0.42 | 0.39 | 0.49 | 0.35 | 0.42 | 0.48 |
| `cultural_bench_hard_morocco` | 0.54 | 0.57 | 0.44 | 0.33 | 0.61 | 0.54 | 0.27 | 0.41 | 0.48 | 0.34 | 0.28 | 0.50 | 0.50 | 0.52 | 0.43 |
| `rf_include_base_44` | 0.45 | 0.47 | 0.56 | 0.33 | 0.35 | 0.47 | 0.45 | 0.37 | 0.53 | 0.40 | 0.53 | 0.47 | 0.38 | 0.56 | 0.42 |
| `cultural_bench_hard_saudi_arabia` | 0.30 | 0.46 | 0.42 | 0.33 | 0.51 | 0.55 | 0.44 | 0.51 | 0.41 | 0.42 | 0.51 | 0.27 | 0.54 | 0.49 | 0.31 |
| `cultural_bench_easy_morocco` | 0.45 | 0.53 | 0.31 | 0.29 | 0.52 | 0.59 | 0.32 | 0.42 | 0.42 | 0.38 | 0.46 | 0.31 | 0.62 | 0.40 | 0.42 |
| `global_piqa_parallel_cloze_arb_arab` | 0.51 | 0.58 | 0.37 | 0.54 | 0.48 | 0.25 | 0.23 | 0.53 | 0.40 | 0.32 | 0.52 | 0.52 | 0.27 | 0.29 | 0.59 |
| `global_piqa_parallel_cloze_apc_arab_pale` | 0.70 | 0.55 | 0.34 | 0.44 | 0.33 | 0.58 | 0.34 | 0.33 | 0.34 | 0.41 | 0.38 | 0.41 | 0.40 | 0.32 | 0.52 |
| `include_v2_og_arabic_uae` | 0.53 | 0.31 | 0.36 | 0.42 | 0.39 | 0.26 | 0.38 | 0.26 | 0.39 | 0.42 | 0.52 | 0.49 | 0.49 | 0.49 | 0.50 |
| `global_piqa_parallel_cloze_apc_arab_jord` | 0.46 | 0.56 | 0.29 | 0.36 | 0.40 | 0.43 | 0.29 | 0.47 | 0.47 | 0.39 | 0.40 | 0.47 | 0.40 | 0.28 | 0.40 |
| `include_v2_en_arabic_uae` | 0.40 | 0.29 | 0.53 | 0.43 | 0.43 | 0.45 | 0.58 | 0.35 | 0.30 | 0.36 | 0.47 | 0.38 | 0.30 | 0.26 | 0.49 |
| `global_piqa_parallel_cloze_apc_arab_leba` | 0.40 | 0.58 | 0.45 | 0.20 | 0.48 | 0.19 | 0.37 | 0.42 | 0.57 | 0.37 | 0.37 | 0.37 | 0.32 | 0.42 | 0.38 |
| `cultural_bench_easy_saudi_arabia` | 0.44 | 0.40 | 0.41 | 0.40 | 0.38 | 0.50 | 0.27 | 0.42 | 0.26 | 0.24 | 0.43 | 0.42 | 0.53 | 0.41 | 0.38 |
| `global_piqa_parallel_cloze_apc_arab_syri` | 0.54 | 0.53 | 0.17 | 0.42 | 0.40 | 0.38 | 0.38 | 0.61 | 0.33 | 0.23 | 0.27 | 0.44 | 0.38 | 0.35 | 0.33 |
| `global_piqa_parallel_cloze_arz_arab` | 0.28 | 0.58 | 0.49 | 0.29 | 0.34 | 0.46 | 0.30 | 0.31 | 0.46 | 0.47 | 0.29 | 0.43 | 0.33 | 0.33 | 0.33 |
| `global_piqa_parallel_cloze_ary_arab` | 0.52 | 0.57 | 0.25 | 0.40 | 0.27 | 0.38 | 0.43 | 0.32 | 0.22 | 0.23 | 0.39 | 0.28 | 0.33 | 0.53 | 0.57 |
| `include_v2_og_arabic_ksa` | 0.23 | 0.46 | 0.23 | 0.53 | 0.28 | 0.33 | 0.46 | 0.24 | 0.42 | 0.27 | 0.47 | 0.26 | 0.26 | 0.57 | 0.26 |
| `include_v2_og_arabic_kuwait` | 0.44 | 0.37 | 0.23 | 0.32 | 0.38 | 0.28 | 0.31 | 0.23 | 0.43 | 0.38 | 0.47 | 0.32 | 0.34 | 0.38 | 0.20 |
| `include_v2_en_arabic_kuwait` | 0.34 | 0.35 | 0.40 | 0.24 | 0.38 | 0.43 | 0.33 | 0.18 | 0.19 | 0.56 | 0.31 | 0.29 | 0.37 | 0.38 | 0.30 |
| `rf_cultural_bench_easy_saudi_arabia` | 0.32 | 0.31 | 0.42 | 0.23 | 0.23 | 0.43 | 0.25 | 0.36 | 0.28 | 0.36 | 0.21 | 0.41 | 0.30 | 0.36 | 0.41 |
| `rf_cultural_bench_easy_morocco` | 0.33 | 0.18 | 0.42 | 0.32 | 0.44 | 0.26 | 0.31 | 0.32 | 0.30 | 0.26 | 0.21 | 0.28 | 0.31 | 0.27 | 0.35 |

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
subtitle: "Azerbaijani (az) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_rfgm_belebele` | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** |
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_rfgm_include_base_44` | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** |
| `bbpb_rf_belebele` | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** |
| `bbpb_global_piqa_parallel_cloze` | 0.67 | **0.83** | 0.67 | **1.00** | **1.00** | **0.83** | **1.00** | 0.67 | 0.67 | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** |
| `bbpb_include_v2_og` | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | 0.67 | 0.67 |
| `bbpb_rf_include_base_44` | **1.00** | 0.67 | 0.67 | 0.67 | **1.00** | 0.67 | 0.67 | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | 0.67 | 0.67 |
| `rfgm_include_base_44` | **1.00** | **0.83** | **0.83** | 0.50 | 0.50 | **0.83** | **0.83** | 0.50 | 0.50 | 0.67 | 0.33 | 0.67 | 0.50 | 0.50 | 0.67 |
| `include_v2_en` | 0.33 | 0.50 | 0.67 | **0.83** | 0.17 | **0.83** | 0.67 | 0.50 | **0.83** | **0.83** | 0.67 | 0.67 | 0.50 | 0.50 | 0.33 |
| `bbpb_belebele` | 0.67 | 0.33 | 0.50 | 0.67 | 0.67 | 0.67 | **0.83** | 0.33 | 0.33 | **0.83** | 0.33 | 0.33 | 0.50 | 0.50 | **1.00** |
| `include_base_44` | 0.67 | 0.67 | 0.50 | 0.67 | 0.33 | 0.33 | 0.33 | 0.33 | 0.00 | **0.83** | **1.00** | 0.67 | **0.83** | 0.50 | 0.67 |
| `rfgm_belebele` | 0.50 | 0.50 | 0.50 | 0.50 | 0.00 | 0.67 | 0.67 | 0.33 | 0.50 | **1.00** | 0.67 | 0.50 | 0.67 | 0.50 | 0.50 |
| `bbpb_include_base_44` | 0.33 | 0.33 | 0.67 | 0.50 | 0.17 | 0.67 | 0.33 | 0.50 | 0.50 | 0.67 | **0.83** | 0.50 | **0.83** | 0.50 | 0.33 |
| `include_v2_og` | 0.67 | 0.67 | 0.17 | 0.17 | 0.17 | **1.00** | 0.50 | 0.50 | 0.17 | 0.50 | 0.50 | 0.17 | 0.67 | 0.67 | 0.67 |
| `bbpb_include_v2_en` | 0.17 | 0.67 | 0.17 | 0.50 | 0.67 | 0.50 | 0.67 | 0.67 | 0.17 | 0.50 | **0.83** | 0.33 | 0.67 | 0.17 | 0.17 |
| `rf_belebele` | 0.67 | 0.50 | 0.17 | 0.50 | 0.67 | 0.33 | 0.33 | 0.67 | 0.67 | 0.33 | 0.33 | 0.67 | 0.67 | 0.00 | 0.33 |
| `belebele` | 0.50 | 0.33 | **0.83** | 0.17 | 0.33 | **0.83** | 0.33 | 0.50 | 0.17 | 0.17 | 0.67 | 0.33 | 0.33 | 0.50 | 0.50 |
| `global_piqa_parallel_cloze` | 0.67 | 0.00 | 0.33 | 0.17 | 0.50 | 0.33 | 0.50 | 0.33 | **0.83** | 0.33 | 0.33 | 0.33 | **0.83** | 0.33 | 0.17 |
| `rf_include_base_44` | 0.50 | 0.17 | 0.33 | 0.17 | 0.67 | 0.33 | 0.33 | 0.17 | 0.50 | 0.33 | 0.50 | 0.50 | **0.83** | 0.00 | 0.17 |

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
subtitle: "Azerbaijani (az) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Bulgarian (bg) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **1.00** | **0.89** | **0.83** | **0.89** | **0.86** | **0.89** | **0.83** | **0.89** | **0.86** | 0.72 | **0.94** | **0.92** | 0.72 | 0.69 | **0.97** |
| `bbpb_rfgm_belebele` | **0.94** | **0.78** | **0.92** | **0.94** | **0.81** | **0.78** | **0.92** | **0.94** | **0.81** | **0.81** | 0.72 | **0.86** | **0.86** | **0.83** | **0.75** |
| `bbpb_rfgm_include_base_44` | **0.78** | **0.83** | **0.75** | **0.83** | **0.86** | **0.89** | **0.86** | **0.89** | **0.81** | **0.86** | **0.83** | **0.81** | **0.75** | 0.72 | **0.86** |
| `bbpb_rf_belebele` | **0.81** | **0.89** | **0.78** | **0.75** | **0.89** | **0.81** | 0.69 | **0.94** | **0.75** | 0.67 | **0.75** | **0.78** | 0.64 | **0.78** | 0.69 |
| `multiblimp` | 0.67 | 0.64 | 0.72 | **0.86** | 0.69 | **0.89** | 0.72 | 0.69 | **0.92** | **0.81** | **0.78** | **0.81** | **0.81** | **0.81** | 0.72 |
| `bbpb_xnli` | **0.92** | **0.75** | **0.92** | 0.72 | **0.75** | 0.67 | **0.89** | 0.64 | 0.72 | **0.78** | 0.64 | **0.89** | 0.64 | **0.83** | 0.64 |
| `bbpb_global_piqa_parallel_cloze` | **0.94** | **0.86** | **0.86** | 0.67 | 0.72 | **0.86** | **0.81** | 0.67 | 0.72 | **0.89** | 0.69 | 0.69 | 0.69 | 0.58 | 0.67 |
| `bbpb_multiblimp` | 0.72 | **0.78** | **0.83** | 0.72 | **0.86** | **0.83** | 0.67 | **0.83** | 0.69 | 0.72 | **0.78** | 0.69 | **0.83** | **0.75** | 0.58 |
| `bbpb_rf_include_base_44` | 0.72 | 0.64 | 0.42 | 0.61 | 0.47 | **0.75** | 0.58 | **0.83** | 0.58 | 0.61 | **0.81** | **0.78** | **0.75** | 0.67 | 0.64 |
| `bbpb_include_v2_og` | 0.64 | 0.56 | 0.47 | 0.61 | 0.53 | 0.69 | 0.56 | **0.75** | 0.67 | **0.75** | 0.72 | 0.69 | 0.53 | 0.56 | **0.75** |
| `bbpb_belebele` | 0.56 | 0.69 | 0.67 | 0.64 | 0.58 | 0.58 | 0.72 | **0.75** | 0.53 | 0.64 | 0.61 | 0.44 | **0.75** | 0.64 | 0.56 |
| `xnli` | 0.64 | 0.42 | 0.25 | 0.44 | 0.44 | 0.44 | 0.42 | 0.50 | 0.58 | **0.78** | 0.56 | 0.58 | 0.47 | 0.61 | 0.53 |
| `include_v2_og` | 0.56 | 0.33 | 0.50 | 0.61 | 0.47 | 0.28 | 0.61 | 0.69 | 0.53 | 0.44 | 0.33 | 0.47 | 0.64 | 0.56 | 0.53 |
| `include_v2_en` | 0.31 | 0.56 | 0.67 | 0.44 | 0.61 | 0.33 | 0.53 | 0.44 | 0.17 | 0.56 | **0.81** | 0.58 | 0.64 | 0.39 | 0.50 |
| `bbpb_include_v2_en` | 0.33 | 0.61 | 0.39 | 0.44 | **0.75** | 0.44 | 0.44 | 0.39 | 0.42 | 0.56 | 0.61 | 0.58 | 0.61 | 0.36 | 0.58 |
| `rfgm_belebele` | 0.25 | 0.33 | 0.47 | 0.33 | 0.61 | 0.64 | 0.58 | 0.50 | 0.33 | 0.64 | 0.61 | 0.58 | 0.69 | 0.47 | 0.33 |
| `bbpb_include_base_44` | 0.64 | 0.50 | 0.50 | 0.36 | 0.67 | 0.64 | 0.25 | 0.44 | 0.53 | 0.28 | 0.53 | 0.61 | 0.42 | 0.50 | 0.47 |
| `rf_belebele` | 0.28 | 0.39 | 0.47 | 0.22 | 0.47 | 0.72 | 0.58 | 0.67 | 0.50 | 0.72 | 0.53 | 0.47 | 0.47 | 0.42 | 0.42 |
| `rf_include_base_44` | 0.61 | 0.39 | 0.56 | 0.56 | 0.56 | 0.36 | 0.58 | 0.61 | 0.47 | 0.11 | 0.64 | 0.47 | 0.42 | 0.33 | 0.53 |
| `belebele` | 0.28 | 0.56 | 0.61 | 0.47 | 0.42 | 0.56 | 0.22 | 0.44 | 0.67 | 0.36 | 0.56 | 0.50 | 0.36 | 0.47 | 0.56 |
| `rfgm_include_base_44` | 0.36 | 0.64 | 0.31 | 0.44 | 0.72 | 0.28 | 0.56 | 0.64 | 0.50 | 0.33 | 0.22 | 0.56 | 0.50 | 0.19 | 0.53 |
| `include_base_44` | 0.33 | 0.72 | 0.64 | 0.61 | 0.42 | 0.47 | 0.61 | 0.14 | 0.28 | 0.61 | 0.33 | 0.44 | 0.36 | 0.19 | 0.47 |
| `global_piqa_parallel_cloze` | 0.25 | 0.67 | 0.67 | 0.50 | 0.17 | 0.25 | 0.22 | 0.53 | 0.39 | 0.69 | 0.44 | 0.31 | 0.56 | 0.11 | 0.19 |

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
subtitle: "Bulgarian (bg) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Bengali (bn) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.91** | **0.86** | **0.76** | **0.83** | **0.88** | **0.90** | **0.85** | **0.87** | **0.92** | **0.87** | **0.87** | **0.85** | **0.85** | **0.77** | **0.87** |
| `bbpb_rfgm_include_base_44` | **0.86** | **0.85** | **0.95** | **0.78** | **0.86** | **0.86** | **0.83** | **0.77** | **0.82** | **0.82** | **0.81** | 0.71 | **0.81** | **0.83** | **0.79** |
| `bbpb_hellaswag` | **0.87** | **0.85** | **0.79** | **0.79** | **0.82** | **0.79** | **0.77** | **0.87** | **0.87** | 0.74 | **0.82** | **0.79** | **0.82** | 0.72 | **0.82** |
| `bbpb_rf_include_base_44` | **0.79** | **0.87** | **0.78** | **0.94** | 0.71 | **0.82** | 0.71 | **0.78** | **0.78** | 0.68 | **0.88** | **0.76** | 0.74 | **0.79** | 0.72 |
| `bbpb_rfgm_belebele_ben_Beng` | **0.83** | **0.76** | **0.86** | **0.78** | **0.79** | **0.77** | **0.77** | **0.82** | **0.76** | 0.72 | 0.72 | **0.78** | **0.79** | 0.73 | **0.81** |
| `bbpb_rf_global_mmlu_full` | 0.71 | **0.81** | **0.78** | **0.83** | 0.68 | **0.82** | 0.72 | 0.72 | **0.79** | **0.79** | **0.77** | 0.72 | **0.79** | 0.69 | **0.85** |
| `bbpb_global_piqa_parallel_cloze_ben_beng` | **0.78** | 0.71 | 0.74 | **0.78** | 0.71 | 0.74 | 0.73 | **0.90** | 0.74 | **0.76** | 0.72 | 0.74 | **0.78** | 0.73 | 0.69 |
| `bbpb_rf_mmmlu` | 0.71 | **0.82** | 0.63 | **0.77** | **0.88** | **0.86** | **0.77** | 0.68 | **0.77** | 0.71 | 0.72 | **0.81** | 0.63 | 0.67 | 0.73 |
| `bbpb_arc` | 0.65 | 0.73 | 0.69 | **0.81** | 0.73 | 0.69 | **0.81** | 0.69 | **0.82** | 0.60 | 0.72 | **0.77** | **0.78** | **0.78** | 0.74 |
| `bbpb_rf_belebele_ben_Beng` | 0.74 | 0.71 | **0.79** | **0.76** | 0.62 | 0.73 | **0.85** | 0.65 | 0.72 | **0.76** | **0.77** | 0.71 | 0.68 | **0.77** | 0.73 |
| `bbpb_include_v2_og_bengali_bangladesh` | 0.55 | 0.68 | 0.62 | **0.77** | 0.69 | **0.82** | **0.76** | 0.68 | **0.78** | 0.73 | **0.81** | **0.86** | 0.64 | 0.72 | **0.82** |
| `bbpb_rf_m_mmlu` | 0.68 | **0.82** | 0.60 | 0.71 | **0.87** | **0.83** | 0.72 | 0.69 | **0.81** | 0.65 | 0.73 | **0.79** | 0.62 | 0.63 | **0.76** |
| `bbpb_include_v2_og_bengali_india` | 0.72 | 0.60 | 0.50 | 0.73 | 0.69 | 0.55 | 0.71 | 0.60 | 0.67 | 0.51 | 0.56 | 0.60 | 0.56 | 0.58 | 0.68 |
| `bbpb_rf_belebele_ben_Latn` | 0.59 | 0.59 | 0.55 | 0.69 | 0.64 | 0.67 | 0.55 | 0.64 | 0.67 | 0.73 | 0.64 | 0.59 | 0.58 | 0.45 | 0.67 |
| `bbpb_rfgm_belebele_ben_Latn` | 0.64 | 0.51 | 0.53 | 0.62 | 0.47 | **0.77** | 0.71 | 0.56 | 0.63 | **0.83** | 0.56 | 0.58 | 0.58 | 0.49 | 0.50 |
| `include_v2_og_bengali_bangladesh` | 0.53 | 0.53 | 0.69 | 0.53 | 0.59 | 0.47 | 0.53 | 0.50 | 0.47 | 0.60 | 0.62 | 0.55 | **0.76** | 0.73 | **0.82** |
| `bbpb_multiblimp` | 0.56 | **0.77** | 0.54 | 0.65 | 0.54 | 0.56 | 0.59 | 0.71 | 0.54 | 0.59 | 0.65 | 0.62 | 0.50 | 0.54 | 0.45 |
| `bbpb_include_v2_en_bengali_bangladesh` | 0.51 | 0.72 | 0.41 | 0.60 | **0.76** | 0.49 | 0.46 | 0.55 | 0.42 | 0.56 | 0.60 | 0.68 | 0.58 | 0.45 | 0.59 |
| `hellaswag` | 0.58 | 0.55 | 0.53 | 0.56 | 0.49 | 0.47 | 0.59 | 0.55 | 0.56 | 0.41 | 0.37 | 0.37 | 0.73 | 0.68 | **0.77** |
| `belebele_ben_Latn` | 0.54 | 0.63 | 0.46 | 0.50 | 0.58 | **0.81** | 0.36 | 0.44 | 0.53 | 0.41 | 0.50 | 0.60 | 0.47 | 0.62 | 0.69 |
| `bbpb_include_base_44` | 0.64 | 0.55 | 0.58 | 0.46 | 0.53 | 0.37 | 0.58 | 0.59 | 0.37 | 0.51 | 0.63 | 0.51 | 0.55 | 0.74 | 0.37 |
| `bbpb_global_piqa_parallel_cloze_ben_latn` | 0.38 | **0.76** | 0.51 | 0.35 | 0.27 | 0.58 | 0.67 | 0.55 | 0.50 | 0.65 | 0.56 | 0.36 | 0.55 | 0.65 | 0.64 |
| `bbpb_belebele_ben_Latn` | 0.53 | 0.53 | 0.53 | 0.63 | 0.38 | 0.56 | 0.41 | 0.59 | 0.65 | 0.54 | 0.51 | 0.35 | 0.56 | 0.42 | 0.55 |
| `include_v2_en_bengali_bangladesh` | 0.55 | 0.36 | 0.62 | 0.54 | 0.45 | 0.49 | 0.59 | 0.40 | 0.45 | 0.40 | 0.51 | 0.65 | 0.58 | 0.44 | 0.73 |
| `bbpb_belebele_ben_Beng` | 0.50 | 0.54 | 0.71 | 0.47 | 0.62 | 0.55 | 0.33 | 0.51 | 0.47 | 0.60 | 0.47 | 0.54 | 0.41 | 0.50 | 0.45 |
| `bbpb_include_v2_en_bengali_india` | 0.35 | 0.60 | 0.44 | 0.42 | 0.51 | 0.59 | 0.47 | 0.46 | 0.40 | 0.58 | 0.67 | 0.45 | **0.76** | 0.54 | 0.45 |
| `global_mmlu_full` | 0.51 | 0.41 | 0.40 | 0.50 | 0.60 | 0.46 | 0.42 | 0.60 | 0.63 | 0.50 | 0.45 | 0.40 | 0.51 | 0.49 | **0.79** |
| `rf_mmmlu` | 0.47 | 0.54 | 0.51 | 0.33 | 0.26 | 0.71 | 0.62 | 0.45 | 0.63 | 0.60 | 0.51 | 0.46 | 0.50 | 0.41 | 0.63 |
| `rf_global_mmlu_full` | 0.54 | 0.54 | 0.45 | 0.51 | 0.27 | 0.59 | 0.36 | 0.45 | 0.51 | 0.51 | 0.41 | 0.60 | 0.53 | 0.64 | 0.63 |
| `bbpb_global_mmlu_full` | **0.77** | 0.42 | 0.46 | 0.44 | 0.60 | 0.22 | 0.49 | 0.54 | 0.47 | 0.50 | 0.40 | 0.49 | 0.49 | 0.63 | 0.53 |
| `rf_include_base_44` | 0.50 | 0.56 | 0.49 | 0.32 | 0.42 | 0.46 | 0.46 | 0.40 | 0.44 | 0.42 | 0.50 | 0.50 | 0.49 | 0.55 | 0.64 |
| `include_base_44` | 0.33 | 0.40 | 0.49 | 0.69 | 0.58 | 0.45 | 0.37 | 0.35 | 0.44 | 0.50 | 0.50 | 0.45 | 0.49 | 0.51 | 0.47 |
| `rf_m_mmlu` | 0.67 | 0.58 | 0.38 | 0.21 | 0.46 | 0.46 | 0.33 | 0.37 | 0.63 | 0.53 | 0.37 | 0.44 | 0.72 | 0.36 | 0.47 |
| `rfgm_belebele_ben_Latn` | 0.38 | 0.35 | 0.56 | 0.29 | 0.51 | 0.47 | 0.54 | 0.58 | 0.40 | 0.46 | 0.55 | 0.51 | 0.27 | 0.49 | 0.58 |
| `rfgm_belebele_ben_Beng` | 0.58 | 0.68 | 0.36 | 0.42 | 0.41 | 0.53 | 0.40 | 0.27 | 0.53 | 0.36 | 0.33 | 0.33 | 0.58 | 0.64 | 0.50 |
| `rf_belebele_ben_Beng` | 0.56 | 0.50 | 0.54 | 0.31 | 0.47 | 0.41 | 0.44 | 0.32 | 0.55 | 0.51 | 0.54 | 0.35 | 0.35 | 0.65 | 0.35 |
| `arc` | 0.45 | 0.63 | 0.36 | 0.33 | 0.51 | 0.59 | 0.42 | 0.40 | 0.41 | 0.44 | 0.35 | 0.38 | 0.65 | 0.33 | 0.58 |
| `rf_belebele_ben_Latn` | 0.36 | 0.50 | 0.40 | 0.27 | 0.40 | 0.44 | 0.32 | 0.42 | 0.46 | 0.38 | 0.47 | 0.46 | 0.42 | 0.50 | 0.62 |
| `include_v2_en_bengali_india` | 0.46 | 0.45 | 0.38 | 0.58 | 0.38 | 0.42 | 0.51 | 0.35 | 0.27 | 0.40 | 0.54 | 0.40 | 0.50 | 0.36 | 0.41 |
| `belebele_ben_Beng` | 0.31 | 0.50 | 0.36 | 0.40 | 0.44 | 0.33 | 0.47 | 0.50 | 0.55 | 0.37 | 0.44 | 0.35 | 0.41 | 0.41 | 0.44 |
| `rfgm_include_base_44` | 0.53 | 0.50 | 0.26 | 0.33 | 0.38 | 0.47 | 0.27 | 0.26 | 0.29 | 0.32 | 0.41 | 0.33 | 0.54 | 0.68 | 0.59 |
| `include_v2_og_bengali_india` | 0.42 | 0.51 | 0.46 | 0.32 | 0.47 | 0.41 | 0.32 | 0.32 | 0.44 | 0.51 | 0.36 | 0.45 | 0.41 | 0.33 | 0.18 |
| `global_piqa_parallel_cloze_ben_latn` | 0.47 | 0.51 | 0.35 | 0.32 | 0.49 | 0.47 | 0.49 | 0.33 | 0.27 | 0.41 | 0.32 | 0.45 | 0.21 | 0.41 | 0.35 |
| `global_piqa_parallel_cloze_ben_beng` | 0.44 | 0.27 | 0.45 | 0.27 | 0.45 | 0.28 | 0.22 | 0.40 | 0.28 | 0.41 | 0.47 | 0.40 | 0.36 | 0.41 | 0.42 |
| `multiblimp` | 0.33 | 0.27 | 0.40 | 0.42 | 0.36 | 0.35 | 0.35 | 0.50 | 0.46 | 0.35 | 0.21 | 0.28 | 0.26 | 0.47 | 0.28 |

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
subtitle: "Bengali (bn) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Bosnian (bs) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_global_piqa_nonparallel_cloze` | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | 0.67 | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** |
| `global_piqa_nonparallel_cloze` | **0.83** | 0.50 | 0.50 | **0.83** | 0.33 | 0.33 | 0.67 | **0.83** | 0.50 | 0.00 | 0.33 | 0.50 | 0.50 | 0.50 | 0.33 |

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
subtitle: "Bosnian (bs) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Catalan (ca) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_paws` | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** |
| `bbpb_global_piqa_parallel_cloze` | **0.83** | **0.83** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** |
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_hellaswag` | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** |
| `bbpb_rf_m_mmlu` | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | 0.67 |
| `multiblimp` | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 |
| `bbpb_multiblimp` | **1.00** | **0.83** | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** | **1.00** | 0.67 | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 |
| `bbpb_rf_belebele` | **0.83** | **1.00** | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** | 0.67 |
| `bbpb_arc` | **1.00** | **0.83** | **0.83** | 0.67 | 0.67 | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** |
| `bbpb_xstorycloze` | **0.83** | **0.83** | 0.67 | **1.00** | **1.00** | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** |
| `bbpb_xnli` | **1.00** | **0.83** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** |
| `bbpb_rfgm_belebele` | **1.00** | **1.00** | 0.67 | **1.00** | 0.67 | **1.00** | 0.67 | **1.00** | 0.67 | 0.67 | **1.00** | 0.67 | 0.67 | 0.67 | 0.67 |
| `hellaswag` | **1.00** | **0.83** | 0.50 | 0.50 | 0.50 | **0.83** | 0.50 | 0.50 | 0.50 | 0.67 | 0.67 | 0.67 | **1.00** | **1.00** | **1.00** |
| `belebele` | 0.33 | 0.67 | 0.50 | 0.50 | 0.67 | 0.67 | 0.67 | **0.83** | 0.67 | 0.50 | **0.83** | 0.67 | 0.67 | **0.83** | **0.83** |
| `bbpb_belebele` | 0.67 | 0.50 | 0.50 | 0.67 | **0.83** | **0.83** | **0.83** | 0.33 | 0.50 | **1.00** | 0.17 | 0.33 | 0.17 | 0.33 | **0.83** |
| `paws` | 0.17 | 0.67 | 0.17 | 0.17 | 0.17 | 0.50 | 0.67 | **1.00** | **1.00** | 0.17 | 0.50 | 0.50 | 0.67 | 0.67 | **1.00** |
| `rfgm_belebele` | 0.67 | 0.50 | 0.67 | 0.67 | 0.50 | 0.33 | **0.83** | 0.50 | 0.50 | 0.50 | **0.83** | 0.00 | 0.67 | 0.50 | 0.17 |
| `xstorycloze` | 0.33 | 0.67 | 0.33 | 0.50 | 0.67 | 0.00 | 0.17 | 0.00 | 0.00 | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** |
| `arc` | 0.67 | 0.33 | **0.83** | 0.50 | 0.50 | 0.67 | 0.50 | 0.17 | 0.50 | 0.17 | 0.50 | 0.50 | 0.33 | 0.33 | 0.67 |
| `rf_belebele` | 0.50 | 0.33 | 0.67 | 0.67 | 0.50 | 0.17 | 0.50 | 0.17 | 0.67 | 0.33 | 0.67 | 0.17 | 0.33 | 0.50 | 0.50 |
| `rf_m_mmlu` | 0.50 | **0.83** | 0.67 | 0.67 | 0.00 | 0.67 | 0.17 | 0.17 | 0.50 | 0.50 | 0.50 | 0.17 | 0.67 | 0.33 | 0.33 |
| `xnli` | 0.33 | 0.50 | 0.00 | 0.67 | 0.00 | 0.33 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.33 | **1.00** | 0.33 |
| `global_piqa_parallel_cloze` | 0.50 | 0.17 | 0.33 | 0.67 | 0.50 | 0.17 | 0.00 | 0.50 | 0.17 | 0.50 | 0.00 | 0.50 | 0.33 | **0.83** | 0.17 |

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
subtitle: "Catalan (ca) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Czech (cs) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.93** | **0.84** | 0.75 | **0.91** | **0.95** | **0.91** | **0.78** | **0.91** | **0.91** | **0.80** | **0.85** | **0.85** | **0.76** | **0.76** | **0.89** |
| `bbpb_rfgm_belebele` | **0.82** | **0.84** | 0.75 | **0.84** | **0.78** | **0.87** | 0.75 | **0.76** | 0.67 | **0.84** | **0.85** | 0.73 | **0.76** | 0.64 | 0.73 |
| `multiblimp` | 0.73 | **0.80** | 0.75 | **0.76** | 0.69 | 0.75 | 0.73 | 0.65 | **0.85** | **0.84** | 0.73 | **0.78** | 0.75 | 0.65 | 0.65 |
| `bbpb_global_piqa_parallel_cloze` | **0.89** | **0.78** | 0.64 | 0.73 | 0.75 | 0.75 | 0.64 | 0.69 | 0.67 | 0.67 | 0.62 | 0.67 | **0.80** | 0.71 | 0.69 |
| `bbpb_rf_belebele` | **0.89** | **0.85** | 0.64 | **0.85** | 0.55 | **0.85** | 0.71 | **0.89** | 0.55 | 0.67 | **0.78** | 0.47 | 0.67 | 0.47 | 0.62 |
| `bbpb_multiblimp` | **0.78** | 0.69 | **0.89** | 0.73 | 0.60 | **0.84** | 0.71 | **0.80** | 0.53 | 0.62 | 0.71 | 0.55 | 0.73 | 0.56 | 0.58 |
| `bbpb_global_mmlu_full` | 0.75 | 0.71 | 0.45 | **0.78** | 0.53 | **0.82** | 0.56 | 0.64 | 0.45 | 0.60 | 0.60 | 0.49 | 0.56 | 0.56 | 0.71 |
| `bbpb_include_v2_en` | 0.53 | 0.65 | 0.64 | 0.45 | 0.73 | 0.62 | 0.49 | 0.45 | **0.76** | 0.69 | 0.51 | **0.78** | 0.60 | 0.62 | 0.47 |
| `bbpb_rf_global_mmlu_full` | 0.56 | 0.60 | 0.45 | **0.78** | 0.53 | **0.78** | 0.71 | 0.67 | 0.56 | 0.56 | 0.67 | 0.49 | 0.38 | 0.56 | 0.42 |
| `bbpb_belebele` | 0.53 | 0.44 | 0.64 | 0.71 | 0.55 | 0.73 | 0.64 | 0.56 | 0.55 | 0.55 | 0.55 | 0.38 | 0.67 | 0.69 | 0.47 |
| `bbpb_include_v2_og` | 0.60 | 0.53 | 0.36 | 0.65 | 0.42 | 0.56 | 0.55 | 0.69 | 0.64 | 0.69 | 0.65 | 0.67 | 0.45 | 0.51 | 0.65 |
| `rfgm_belebele` | 0.53 | 0.49 | 0.58 | 0.56 | 0.56 | 0.38 | 0.49 | 0.45 | 0.45 | 0.62 | 0.42 | 0.58 | 0.71 | 0.65 | 0.55 |
| `rf_global_mmlu_full` | 0.55 | 0.55 | 0.45 | 0.51 | 0.47 | 0.62 | 0.45 | 0.38 | 0.67 | 0.53 | 0.53 | 0.36 | 0.55 | 0.51 | 0.38 |
| `rf_belebele` | 0.62 | 0.55 | 0.55 | 0.51 | 0.45 | 0.31 | 0.33 | 0.45 | 0.64 | 0.60 | 0.53 | 0.33 | 0.42 | 0.47 | 0.55 |
| `global_mmlu_full` | 0.47 | 0.64 | 0.36 | 0.33 | 0.51 | 0.47 | 0.42 | 0.53 | 0.38 | 0.40 | 0.33 | 0.51 | 0.56 | 0.49 | 0.67 |
| `include_v2_en` | 0.64 | 0.42 | 0.53 | 0.29 | 0.44 | 0.55 | 0.55 | 0.38 | 0.51 | 0.53 | 0.53 | 0.35 | 0.53 | 0.49 | 0.35 |
| `belebele` | 0.45 | 0.31 | 0.25 | 0.47 | 0.40 | 0.47 | 0.53 | 0.49 | 0.45 | 0.51 | 0.45 | 0.58 | 0.65 | 0.53 | 0.38 |
| `include_v2_og` | 0.55 | 0.56 | 0.44 | 0.53 | 0.31 | 0.58 | 0.33 | 0.45 | 0.31 | 0.38 | 0.49 | 0.42 | 0.36 | 0.56 | 0.56 |
| `global_piqa_parallel_cloze` | 0.49 | 0.73 | 0.40 | 0.36 | 0.35 | 0.60 | 0.58 | 0.33 | 0.40 | 0.44 | 0.36 | 0.38 | 0.07 | 0.58 | 0.35 |

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
subtitle: "Czech (cs) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Danish (da) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_hellaswag` | **0.97** | **0.86** | **0.86** | **0.97** | **0.86** | **0.89** | **0.89** | **1.00** | **0.89** | **0.89** | **0.89** | **0.83** | **0.89** | **0.89** | **0.89** |
| `bpb` | **1.00** | **0.94** | **0.78** | **0.89** | **0.94** | **0.94** | **0.78** | **0.89** | **0.94** | **0.83** | **0.94** | **0.89** | **0.78** | 0.72 | **0.89** |
| `bbpb_rfgm_belebele` | **0.89** | **0.89** | **0.83** | **0.97** | **0.81** | **0.83** | **0.83** | **0.86** | **0.86** | 0.72 | **0.92** | **0.86** | **0.81** | 0.69 | **0.83** |
| `bbpb_rf_m_mmlu` | **0.78** | **0.81** | 0.64 | **0.89** | 0.72 | **0.81** | **0.81** | **0.83** | 0.72 | **0.78** | **0.75** | **0.81** | **0.75** | **0.86** | **0.78** |
| `bbpb_multiblimp` | **0.89** | **0.78** | **0.83** | **0.78** | **0.75** | **0.89** | **0.78** | **0.83** | 0.64 | 0.67 | **0.83** | 0.58 | 0.61 | **0.75** | 0.69 |
| `hellaswag` | 0.64 | 0.64 | 0.72 | **0.78** | 0.69 | **0.75** | 0.72 | **0.81** | 0.72 | **0.81** | 0.67 | 0.69 | **0.81** | **0.83** | **0.92** |
| `bbpb_rf_belebele` | **0.89** | **0.81** | 0.61 | **0.81** | 0.72 | **0.81** | 0.61 | **0.81** | 0.72 | **0.75** | 0.72 | **0.75** | 0.64 | 0.61 | 0.69 |
| `bbpb_arc_mt` | 0.69 | **0.75** | 0.56 | 0.72 | 0.64 | 0.72 | **0.75** | 0.69 | 0.56 | **0.75** | 0.69 | 0.72 | **0.78** | 0.58 | 0.64 |
| `bbpb_include_v2_og` | **0.75** | 0.69 | 0.56 | **0.78** | 0.72 | 0.67 | 0.64 | 0.69 | 0.69 | 0.47 | **0.81** | **0.75** | 0.50 | 0.50 | **0.78** |
| `bbpb_belebele` | 0.64 | 0.67 | 0.72 | 0.67 | 0.58 | 0.69 | **0.86** | **0.75** | 0.67 | 0.72 | 0.67 | 0.42 | 0.67 | 0.69 | 0.58 |
| `bbpb_arc` | 0.69 | 0.67 | 0.53 | 0.67 | 0.58 | 0.58 | 0.67 | 0.64 | 0.44 | **0.75** | 0.72 | 0.69 | **0.81** | 0.56 | 0.64 |
| `bbpb_include_v2_en` | 0.61 | **0.78** | 0.67 | 0.61 | 0.61 | **0.78** | 0.67 | 0.61 | 0.50 | 0.72 | 0.61 | 0.56 | 0.61 | 0.33 | 0.50 |
| `include_v2_og` | 0.58 | 0.58 | 0.58 | 0.64 | 0.56 | 0.53 | **0.81** | **0.75** | 0.56 | 0.61 | 0.61 | 0.44 | 0.67 | 0.56 | 0.50 |
| `rfgm_belebele` | 0.44 | 0.69 | 0.42 | 0.72 | 0.50 | 0.39 | 0.39 | 0.36 | 0.36 | 0.69 | **0.81** | 0.72 | 0.56 | 0.64 | 0.64 |
| `include_v2_en` | 0.47 | 0.44 | 0.56 | 0.69 | 0.64 | 0.42 | 0.64 | 0.28 | 0.58 | 0.56 | 0.47 | 0.44 | 0.47 | 0.64 | 0.44 |
| `arc_mt` | 0.47 | 0.64 | 0.53 | 0.28 | 0.33 | 0.36 | 0.33 | 0.36 | 0.50 | **0.75** | 0.56 | 0.53 | 0.67 | 0.69 | 0.69 |
| `rf_m_mmlu` | 0.58 | 0.47 | 0.47 | 0.39 | 0.28 | 0.44 | 0.47 | 0.58 | 0.36 | 0.67 | 0.47 | 0.64 | 0.61 | 0.67 | 0.58 |
| `arc` | 0.58 | 0.56 | 0.53 | 0.22 | 0.36 | 0.39 | 0.50 | 0.42 | 0.69 | 0.53 | 0.56 | 0.58 | 0.47 | 0.44 | 0.67 |
| `belebele` | 0.39 | 0.53 | 0.36 | 0.50 | 0.53 | **0.78** | 0.53 | 0.36 | 0.47 | 0.42 | 0.39 | 0.50 | 0.64 | 0.28 | 0.56 |
| `rf_belebele` | 0.36 | 0.28 | 0.31 | 0.39 | 0.47 | 0.53 | 0.61 | 0.42 | 0.42 | 0.50 | 0.72 | 0.42 | 0.44 | 0.25 | 0.56 |
| `multiblimp` | 0.28 | 0.17 | 0.47 | 0.28 | 0.31 | 0.53 | 0.19 | 0.33 | 0.17 | 0.25 | 0.44 | 0.28 | 0.44 | 0.44 | 0.50 |

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
subtitle: "Danish (da) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "German (de) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.97** | **0.88** | **0.87** | **0.93** | **0.93** | **0.89** | **0.88** | **0.93** | **0.91** | **0.93** | **0.92** | **0.85** | **0.90** | **0.83** | **0.89** |
| `bbpb_hellaswag` | **0.95** | **0.86** | **0.89** | **0.90** | **0.79** | **0.88** | **0.89** | **0.88** | **0.78** | **0.89** | **0.87** | 0.74 | **0.88** | **0.75** | **0.76** |
| `bbpb_rfgm_belebele` | **0.86** | **0.86** | **0.79** | **0.81** | 0.70 | **0.91** | **0.84** | **0.82** | 0.73 | **0.83** | **0.80** | 0.67 | **0.78** | 0.74 | 0.72 |
| `hellaswag` | **0.80** | **0.79** | **0.80** | **0.79** | 0.72 | **0.78** | **0.84** | **0.87** | **0.78** | **0.82** | **0.79** | 0.71 | **0.84** | 0.73 | **0.80** |
| `bbpb_rfgm_include_base_44` | **0.78** | 0.74 | **0.78** | **0.78** | 0.72 | 0.69 | 0.70 | 0.74 | 0.70 | **0.82** | **0.75** | 0.71 | **0.80** | 0.74 | **0.78** |
| `bbpb_include_v2_og` | **0.75** | **0.76** | 0.73 | **0.82** | 0.67 | 0.71 | 0.71 | **0.76** | 0.65 | 0.67 | **0.77** | 0.75 | 0.73 | 0.72 | 0.71 |
| `bbpb_xnli` | **0.81** | **0.78** | **0.82** | 0.71 | 0.67 | **0.78** | 0.74 | 0.68 | 0.62 | **0.81** | 0.69 | 0.71 | 0.72 | 0.73 | 0.66 |
| `bbpb_multiblimp` | **0.81** | **0.78** | **0.77** | 0.73 | 0.71 | **0.82** | 0.70 | **0.79** | 0.66 | 0.71 | 0.73 | 0.71 | 0.72 | 0.67 | 0.61 |
| `lambada_openai_mt` | **0.75** | 0.72 | 0.65 | **0.80** | **0.80** | **0.79** | **0.78** | **0.77** | 0.67 | 0.70 | 0.67 | 0.64 | 0.65 | 0.63 | 0.75 |
| `bbpb_global_piqa_parallel_cloze` | **0.81** | **0.79** | **0.79** | 0.65 | 0.65 | **0.81** | 0.72 | 0.68 | 0.64 | **0.80** | 0.71 | 0.66 | 0.74 | 0.62 | 0.66 |
| `bbpb_paws` | **0.78** | **0.76** | 0.72 | 0.72 | 0.68 | **0.75** | 0.73 | 0.74 | 0.68 | 0.69 | 0.71 | 0.70 | 0.67 | 0.65 | 0.64 |
| `bbpb_global_mmlu_full` | **0.76** | 0.68 | 0.68 | 0.72 | 0.62 | 0.70 | 0.74 | 0.70 | 0.62 | 0.74 | 0.68 | 0.69 | 0.75 | 0.62 | 0.60 |
| `bbpb_rf_global_mmlu_full` | **0.79** | 0.70 | 0.72 | 0.65 | 0.71 | **0.75** | 0.74 | 0.71 | 0.65 | **0.78** | 0.60 | 0.65 | 0.57 | 0.66 | 0.59 |
| `bbpb_arc` | 0.72 | 0.71 | 0.54 | 0.74 | 0.64 | **0.75** | 0.67 | **0.79** | 0.68 | 0.58 | 0.70 | **0.75** | 0.58 | 0.64 | 0.67 |
| `bbpb_arc_mt` | 0.73 | 0.71 | 0.54 | 0.73 | 0.63 | **0.75** | 0.67 | **0.80** | 0.71 | 0.56 | 0.70 | 0.73 | 0.58 | 0.62 | 0.67 |
| `bbpb_rf_belebele` | **0.81** | 0.71 | 0.71 | **0.77** | 0.61 | **0.76** | 0.73 | 0.67 | 0.50 | 0.63 | 0.67 | 0.57 | 0.66 | 0.63 | 0.60 |
| `bbpb_rf_include_base_44` | 0.74 | 0.68 | 0.64 | 0.67 | 0.61 | 0.72 | 0.66 | 0.71 | 0.60 | 0.63 | 0.64 | 0.66 | 0.63 | 0.73 | 0.59 |
| `multiblimp` | 0.68 | 0.68 | 0.68 | 0.66 | 0.61 | 0.71 | 0.70 | 0.62 | 0.55 | 0.68 | 0.58 | 0.66 | 0.66 | 0.65 | 0.58 |
| `bbpb_rf_m_mmlu` | 0.74 | 0.71 | 0.65 | 0.71 | 0.57 | 0.74 | 0.69 | 0.62 | 0.51 | 0.57 | 0.62 | 0.58 | 0.62 | 0.63 | 0.61 |
| `bbpb_rf_mmmlu` | 0.74 | 0.71 | 0.67 | 0.70 | 0.58 | 0.71 | 0.70 | 0.64 | 0.49 | 0.59 | 0.64 | 0.55 | 0.61 | 0.62 | 0.57 |
| `bbpb_include_base_44` | 0.71 | 0.68 | 0.63 | 0.67 | 0.64 | 0.64 | **0.75** | 0.64 | 0.53 | 0.59 | 0.59 | 0.60 | 0.70 | 0.53 | 0.55 |
| `bbpb_belebele` | 0.68 | 0.63 | 0.68 | 0.68 | 0.58 | 0.64 | 0.71 | 0.60 | 0.56 | 0.64 | 0.55 | 0.56 | 0.64 | 0.59 | 0.59 |
| `rf_global_mmlu_full` | 0.60 | 0.56 | 0.49 | 0.57 | 0.54 | 0.60 | 0.59 | 0.67 | 0.68 | 0.53 | 0.56 | 0.61 | 0.67 | 0.67 | **0.76** |
| `bbpb_include_v2_en` | 0.51 | 0.70 | 0.59 | 0.64 | 0.68 | 0.57 | 0.56 | 0.55 | 0.50 | 0.60 | 0.62 | 0.64 | 0.64 | 0.49 | 0.56 |
| `arc_mt` | 0.62 | 0.58 | 0.67 | 0.64 | 0.62 | 0.48 | 0.53 | 0.59 | 0.51 | 0.61 | 0.62 | 0.49 | 0.59 | 0.60 | 0.58 |
| `include_v2_og` | 0.50 | 0.56 | 0.53 | 0.63 | 0.53 | 0.48 | 0.63 | 0.55 | 0.60 | 0.55 | 0.62 | 0.56 | 0.66 | 0.67 | 0.61 |
| `rf_mmmlu` | 0.39 | 0.49 | 0.57 | 0.52 | 0.44 | 0.66 | 0.50 | 0.68 | 0.51 | 0.67 | 0.64 | 0.59 | 0.64 | 0.63 | 0.68 |
| `rf_m_mmlu` | 0.47 | 0.50 | 0.50 | 0.55 | 0.49 | 0.62 | 0.42 | 0.60 | 0.52 | 0.56 | 0.51 | 0.63 | 0.62 | 0.68 | 0.62 |
| `arc` | 0.57 | 0.42 | 0.39 | 0.45 | 0.45 | 0.59 | 0.57 | 0.59 | 0.68 | 0.54 | 0.60 | 0.53 | 0.63 | 0.59 | 0.59 |
| `rfgm_belebele` | 0.51 | 0.63 | 0.59 | 0.54 | 0.44 | 0.52 | 0.57 | 0.56 | 0.41 | 0.53 | 0.56 | 0.42 | 0.57 | 0.59 | 0.59 |
| `xnli` | 0.41 | 0.54 | 0.66 | 0.55 | 0.57 | 0.26 | 0.48 | 0.54 | 0.50 | 0.57 | 0.45 | 0.64 | 0.55 | 0.48 | 0.50 |
| `paws` | 0.54 | 0.52 | 0.49 | 0.54 | 0.57 | 0.46 | 0.51 | 0.37 | 0.47 | 0.53 | 0.65 | 0.46 | 0.46 | 0.43 | 0.58 |
| `global_mmlu_full` | 0.50 | 0.51 | 0.50 | 0.47 | 0.45 | 0.59 | 0.40 | 0.41 | 0.38 | 0.46 | 0.53 | 0.52 | 0.52 | 0.66 | 0.57 |
| `include_v2_en` | 0.48 | 0.47 | 0.50 | 0.52 | 0.53 | 0.57 | 0.54 | 0.38 | 0.37 | 0.55 | 0.49 | 0.43 | 0.59 | 0.51 | 0.53 |
| `belebele` | 0.46 | 0.57 | 0.54 | 0.56 | 0.51 | 0.55 | 0.47 | 0.43 | 0.40 | 0.57 | 0.51 | 0.39 | 0.59 | 0.38 | 0.53 |
| `rf_belebele` | 0.44 | 0.30 | 0.44 | 0.40 | 0.49 | 0.57 | 0.53 | 0.51 | 0.54 | 0.54 | 0.48 | 0.45 | 0.61 | 0.44 | 0.57 |
| `include_base_44` | 0.36 | 0.45 | 0.30 | 0.43 | 0.41 | 0.57 | 0.45 | 0.41 | 0.46 | 0.49 | 0.42 | 0.51 | 0.42 | 0.45 | 0.59 |
| `rfgm_include_base_44` | 0.57 | 0.50 | 0.63 | 0.41 | 0.32 | 0.48 | 0.51 | 0.50 | 0.48 | 0.46 | 0.38 | 0.37 | 0.29 | 0.29 | 0.45 |
| `rf_include_base_44` | 0.43 | 0.42 | 0.50 | 0.32 | 0.39 | 0.51 | 0.52 | 0.34 | 0.47 | 0.45 | 0.34 | 0.51 | 0.36 | 0.56 | 0.38 |
| `global_piqa_parallel_cloze` | 0.44 | 0.40 | 0.38 | 0.47 | 0.43 | 0.55 | 0.45 | 0.44 | 0.41 | 0.38 | 0.45 | 0.34 | 0.43 | 0.46 | 0.42 |

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
subtitle: "Greek (el) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.95** | **0.79** | 0.73 | **0.89** | **0.89** | **0.84** | **0.78** | **0.90** | **0.90** | **0.77** | **0.85** | **0.81** | **0.77** | 0.73 | **0.87** |
| `bbpb_rfgm_include_base_44` | **0.87** | **0.83** | **0.82** | **0.87** | **0.81** | **0.85** | **0.76** | **0.85** | **0.79** | **0.76** | **0.81** | **0.85** | **0.84** | **0.82** | **0.87** |
| `bbpb_rfgm_belebele` | **0.83** | **0.75** | 0.57 | **0.89** | 0.72 | **0.81** | 0.67 | **0.81** | 0.74 | 0.53 | **0.75** | 0.65 | 0.67 | 0.71 | 0.72 |
| `bbpb_multiblimp` | 0.72 | **0.79** | 0.73 | 0.70 | 0.60 | 0.72 | 0.70 | **0.83** | 0.53 | **0.77** | 0.69 | 0.58 | 0.63 | 0.58 | 0.50 |
| `bbpb_global_piqa_parallel_cloze` | 0.73 | 0.72 | **0.83** | 0.41 | 0.58 | **0.86** | **0.77** | 0.60 | 0.64 | **0.76** | 0.65 | 0.69 | 0.47 | 0.56 | 0.64 |
| `bbpb_rf_belebele` | **0.80** | 0.72 | 0.71 | 0.70 | 0.50 | 0.73 | 0.70 | **0.80** | 0.53 | 0.63 | 0.74 | 0.48 | 0.71 | 0.49 | 0.60 |
| `bbpb_xnli` | **0.75** | 0.63 | 0.71 | 0.73 | 0.61 | 0.63 | 0.56 | 0.66 | 0.63 | 0.55 | 0.67 | 0.60 | 0.64 | 0.50 | 0.59 |
| `bbpb_include_v2_og_greek_greece` | 0.63 | 0.65 | 0.42 | 0.62 | 0.64 | **0.79** | 0.58 | 0.67 | 0.59 | 0.56 | 0.63 | 0.65 | 0.57 | 0.61 | 0.64 |
| `bbpb_belebele` | **0.75** | 0.65 | **0.75** | 0.73 | 0.53 | 0.55 | 0.62 | 0.62 | 0.59 | 0.59 | 0.48 | 0.62 | 0.70 | 0.61 | 0.36 |
| `bbpb_rf_global_mmlu_full` | 0.72 | 0.61 | 0.53 | 0.57 | 0.69 | 0.64 | 0.71 | 0.60 | 0.60 | 0.66 | 0.49 | 0.50 | 0.50 | 0.58 | 0.60 |
| `bbpb_rf_include_base_44` | **0.81** | 0.69 | 0.39 | 0.67 | 0.56 | 0.74 | 0.39 | 0.74 | 0.60 | 0.38 | **0.75** | 0.61 | 0.55 | 0.60 | 0.51 |
| `bbpb_include_v2_og_greek_cyprus` | 0.64 | 0.73 | 0.54 | 0.65 | 0.51 | 0.70 | 0.52 | 0.50 | 0.63 | 0.49 | 0.55 | 0.63 | 0.69 | 0.59 | 0.58 |
| `bbpb_global_mmlu_full` | **0.77** | 0.52 | 0.72 | 0.65 | 0.68 | 0.62 | 0.61 | 0.57 | 0.64 | 0.30 | 0.53 | 0.56 | 0.64 | 0.55 | 0.55 |
| `bbpb_include_base_44` | **0.81** | 0.51 | 0.56 | 0.53 | 0.67 | 0.59 | 0.52 | 0.46 | 0.69 | 0.29 | 0.50 | 0.62 | 0.53 | 0.51 | 0.49 |
| `bbpb_arc_mt` | 0.62 | 0.65 | 0.50 | 0.57 | 0.70 | 0.59 | 0.48 | **0.78** | 0.62 | 0.37 | 0.60 | 0.48 | 0.47 | 0.36 | 0.50 |
| `multiblimp` | 0.60 | 0.58 | 0.37 | 0.67 | 0.53 | 0.56 | 0.57 | 0.55 | 0.61 | 0.39 | 0.48 | 0.57 | 0.46 | 0.36 | 0.65 |
| `bbpb_include_v2_en_greek_cyprus` | 0.41 | 0.43 | 0.50 | 0.40 | 0.49 | 0.52 | 0.63 | 0.67 | 0.58 | 0.55 | 0.51 | 0.62 | 0.71 | 0.40 | 0.40 |
| `include_v2_en_greek_greece` | 0.50 | 0.61 | 0.57 | 0.49 | 0.54 | 0.47 | 0.53 | 0.46 | 0.43 | 0.55 | 0.45 | 0.56 | 0.62 | 0.47 | 0.46 |
| `rfgm_include_base_44` | 0.55 | 0.50 | 0.63 | 0.46 | 0.40 | 0.60 | 0.55 | 0.48 | 0.40 | 0.51 | 0.67 | 0.56 | 0.45 | 0.49 | 0.44 |
| `rfgm_belebele` | 0.40 | 0.38 | 0.40 | 0.57 | 0.61 | 0.54 | 0.53 | 0.59 | 0.47 | 0.42 | 0.42 | 0.45 | 0.61 | 0.58 | 0.64 |
| `bbpb_include_v2_en_greek_greece` | 0.29 | 0.52 | 0.51 | 0.41 | 0.53 | 0.46 | 0.49 | 0.50 | 0.43 | 0.69 | 0.56 | 0.65 | 0.55 | 0.50 | 0.46 |
| `xnli` | 0.44 | 0.52 | 0.56 | 0.59 | 0.43 | 0.48 | 0.41 | 0.60 | 0.52 | 0.30 | 0.50 | 0.64 | 0.61 | 0.37 | 0.56 |
| `rf_belebele` | 0.30 | 0.45 | 0.43 | 0.36 | 0.43 | 0.50 | 0.51 | 0.50 | 0.44 | 0.50 | 0.57 | 0.34 | 0.73 | 0.68 | 0.67 |
| `include_v2_og_greek_greece` | 0.51 | 0.67 | 0.47 | 0.47 | 0.50 | 0.62 | 0.50 | 0.46 | 0.55 | 0.47 | 0.37 | 0.41 | 0.40 | 0.52 | 0.43 |
| `rf_global_mmlu_full` | 0.31 | 0.62 | 0.44 | 0.67 | 0.59 | 0.37 | 0.52 | 0.31 | 0.40 | 0.47 | 0.52 | 0.63 | 0.49 | 0.51 | 0.48 |
| `rf_include_base_44` | 0.35 | 0.55 | 0.30 | 0.73 | 0.55 | 0.48 | 0.50 | 0.44 | 0.48 | 0.48 | 0.44 | 0.60 | 0.47 | 0.39 | 0.47 |
| `global_mmlu_full` | 0.29 | 0.45 | 0.54 | 0.63 | 0.65 | 0.33 | 0.32 | 0.46 | 0.50 | 0.52 | 0.48 | 0.33 | 0.54 | 0.46 | 0.70 |
| `belebele` | 0.32 | 0.47 | 0.52 | 0.59 | 0.41 | 0.44 | 0.50 | 0.27 | 0.39 | 0.33 | 0.52 | 0.57 | 0.59 | 0.41 | 0.65 |
| `include_base_44` | 0.38 | 0.65 | 0.39 | 0.42 | 0.50 | 0.49 | 0.61 | 0.49 | 0.34 | 0.57 | 0.41 | 0.45 | 0.42 | 0.40 | 0.46 |
| `arc_mt` | 0.60 | 0.38 | 0.38 | 0.51 | 0.60 | 0.50 | 0.28 | 0.55 | 0.48 | 0.33 | 0.49 | 0.30 | 0.53 | 0.43 | 0.51 |
| `global_piqa_parallel_cloze` | 0.58 | 0.52 | 0.42 | 0.34 | 0.39 | 0.54 | 0.49 | 0.36 | 0.53 | 0.47 | 0.30 | 0.47 | 0.30 | 0.47 | 0.43 |
| `include_v2_en_greek_cyprus` | 0.39 | 0.43 | 0.26 | 0.37 | 0.48 | 0.46 | 0.41 | 0.44 | 0.34 | 0.60 | 0.39 | 0.55 | 0.36 | 0.31 | 0.45 |
| `include_v2_og_greek_cyprus` | 0.22 | 0.36 | 0.39 | 0.47 | 0.24 | 0.36 | 0.41 | 0.39 | 0.65 | 0.50 | 0.23 | 0.50 | 0.27 | 0.38 | 0.33 |

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
subtitle: "Greek (el) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Spanish (es) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.99** | **0.87** | **0.84** | **0.90** | **0.92** | **0.87** | **0.85** | **0.90** | **0.92** | **0.90** | **0.90** | **0.84** | **0.88** | **0.83** | **0.88** |
| `bbpb_hellaswag` | **0.97** | **0.87** | **0.89** | **0.90** | **0.84** | **0.90** | **0.90** | **0.91** | **0.84** | **0.90** | **0.90** | **0.84** | **0.86** | **0.79** | **0.82** |
| `bbpb_xstorycloze` | **0.88** | **0.78** | **0.81** | **0.89** | **0.89** | **0.88** | **0.86** | **0.90** | **0.88** | **0.88** | **0.85** | **0.80** | **0.90** | **0.81** | **0.88** |
| `hellaswag` | **0.83** | **0.79** | **0.82** | **0.86** | **0.83** | **0.89** | **0.91** | **0.88** | **0.84** | **0.89** | **0.84** | **0.82** | **0.84** | **0.80** | **0.88** |
| `bbpb_rfgm_include_base_44` | **0.82** | **0.79** | **0.76** | 0.74 | **0.78** | **0.82** | **0.79** | **0.81** | 0.74 | **0.81** | 0.74 | **0.77** | 0.72 | 0.72 | **0.80** |
| `bbpb_rfgm_belebele` | **0.86** | **0.86** | **0.79** | **0.83** | 0.74 | **0.79** | **0.83** | **0.79** | 0.67 | 0.73 | **0.88** | 0.72 | 0.74 | 0.64 | 0.68 |
| `bbpb_include_v2_og_spanish_argentina` | **0.77** | **0.80** | **0.75** | **0.76** | 0.65 | 0.70 | 0.74 | **0.76** | 0.61 | 0.67 | 0.75 | 0.71 | 0.68 | 0.70 | 0.65 |
| `bbpb_truthfulqa-multi_mc1` | **0.76** | **0.80** | **0.76** | 0.67 | 0.67 | 0.72 | 0.69 | 0.60 | 0.67 | 0.74 | 0.70 | **0.78** | **0.77** | 0.65 | 0.63 |
| `bbpb_global_piqa_parallel_cloze_spa_latn_spai` | **0.76** | **0.79** | **0.76** | 0.61 | 0.63 | **0.80** | **0.79** | 0.65 | 0.60 | **0.77** | 0.75 | 0.67 | 0.70 | 0.66 | 0.65 |
| `bbpb_rf_belebele` | **0.76** | **0.76** | 0.72 | 0.75 | 0.63 | 0.74 | **0.78** | 0.74 | 0.66 | 0.70 | 0.66 | 0.64 | 0.72 | 0.69 | 0.60 |
| `bbpb_include_v2_og_spanish_ecuador` | 0.66 | **0.78** | 0.74 | 0.66 | 0.67 | 0.70 | 0.73 | 0.74 | 0.61 | 0.64 | 0.65 | 0.73 | 0.68 | 0.75 | 0.72 |
| `bbpb_include_v2_og_spanish_espa_a` | 0.71 | **0.76** | 0.69 | 0.69 | 0.61 | 0.67 | **0.77** | **0.81** | 0.59 | 0.66 | 0.69 | 0.67 | 0.75 | 0.68 | 0.67 |
| `bbpb_include_v2_en_spanish_ecuador` | 0.67 | 0.74 | 0.66 | 0.69 | **0.80** | 0.65 | 0.64 | 0.70 | 0.67 | 0.60 | 0.71 | 0.72 | 0.70 | 0.66 | 0.73 |
| `include_v2_og_spanish_espa_a` | 0.65 | 0.71 | 0.70 | 0.70 | 0.72 | **0.76** | 0.66 | 0.54 | 0.65 | 0.68 | 0.63 | 0.71 | 0.70 | **0.78** | 0.72 |
| `bbpb_global_piqa_parallel_cloze_spa_latn_mexi` | **0.75** | 0.75 | 0.70 | 0.59 | 0.64 | 0.73 | 0.74 | 0.60 | 0.61 | 0.70 | 0.68 | 0.70 | 0.72 | 0.72 | 0.67 |
| `bbpb_paws` | 0.72 | **0.77** | 0.74 | 0.65 | 0.71 | **0.76** | 0.63 | 0.71 | 0.65 | 0.63 | 0.69 | 0.66 | 0.60 | 0.70 | 0.66 |
| `bbpb_global_mmlu_full` | 0.70 | 0.66 | 0.75 | 0.72 | 0.67 | 0.63 | 0.68 | 0.66 | 0.61 | 0.70 | 0.74 | 0.71 | 0.73 | 0.63 | 0.65 |
| `bbpb_include_v2_og_spanish_uruguay` | 0.73 | **0.79** | 0.70 | **0.75** | 0.68 | 0.68 | 0.73 | 0.70 | 0.58 | 0.60 | 0.69 | 0.66 | 0.63 | 0.65 | 0.62 |
| `bbpb_rf_mmmlu` | 0.71 | **0.75** | 0.66 | 0.57 | 0.63 | **0.76** | 0.74 | 0.71 | 0.62 | 0.69 | 0.69 | 0.65 | 0.66 | 0.73 | 0.57 |
| `bbpb_global_piqa_parallel_cloze_spa_latn_peru` | 0.74 | 0.75 | 0.68 | 0.54 | 0.63 | **0.77** | 0.75 | 0.59 | 0.58 | **0.83** | 0.70 | 0.63 | 0.69 | 0.60 | 0.65 |
| `bbpb_rf_global_mmlu_full` | 0.69 | 0.61 | 0.62 | 0.71 | 0.70 | 0.70 | 0.72 | 0.75 | 0.63 | 0.68 | 0.56 | 0.70 | 0.61 | 0.74 | 0.67 |
| `bbpb_include_base_44` | **0.78** | 0.70 | **0.76** | 0.64 | 0.70 | 0.58 | 0.69 | 0.63 | 0.57 | 0.68 | 0.69 | 0.68 | 0.71 | 0.66 | 0.61 |
| `bbpb_include_v2_og_spanish_chile` | 0.70 | **0.76** | **0.82** | 0.62 | 0.64 | **0.77** | 0.71 | 0.65 | 0.58 | 0.69 | 0.67 | 0.73 | 0.56 | 0.64 | 0.50 |
| `bbpb_rf_m_mmlu` | 0.66 | 0.73 | 0.61 | 0.62 | 0.62 | 0.71 | 0.74 | 0.72 | 0.58 | 0.65 | 0.74 | 0.66 | 0.65 | 0.72 | 0.58 |
| `bbpb_multiblimp` | 0.69 | 0.69 | 0.73 | 0.62 | 0.66 | **0.75** | 0.66 | 0.67 | 0.58 | 0.61 | 0.61 | 0.63 | 0.62 | 0.74 | 0.65 |
| `bbpb_arc_mt` | 0.70 | 0.68 | 0.65 | 0.61 | 0.56 | 0.67 | **0.80** | 0.71 | 0.67 | 0.66 | 0.64 | 0.68 | 0.62 | 0.61 | 0.60 |
| `bbpb_include_v2_og_spanish_cuba` | 0.67 | 0.73 | 0.55 | 0.68 | 0.65 | 0.62 | 0.61 | 0.60 | 0.58 | 0.55 | 0.71 | **0.77** | 0.65 | 0.70 | 0.69 |
| `rf_mmmlu` | 0.60 | 0.56 | 0.53 | 0.60 | 0.64 | 0.68 | 0.69 | 0.74 | **0.79** | 0.66 | 0.57 | 0.61 | 0.65 | 0.67 | **0.76** |
| `bbpb_include_v2_og_spanish_el_salvador` | 0.60 | 0.73 | 0.66 | 0.69 | 0.64 | 0.61 | 0.61 | 0.73 | 0.54 | 0.60 | 0.71 | 0.64 | 0.66 | 0.69 | 0.61 |
| `bbpb_include_v2_og_spanish_colombia` | 0.69 | 0.62 | 0.72 | 0.63 | 0.65 | 0.64 | 0.68 | **0.83** | 0.66 | 0.48 | 0.58 | 0.63 | 0.69 | 0.54 | 0.65 |
| `xstorycloze` | 0.60 | 0.63 | 0.73 | 0.59 | 0.64 | 0.66 | 0.58 | 0.51 | 0.58 | 0.70 | 0.65 | 0.65 | 0.71 | **0.77** | 0.70 |
| `lambada_openai_mt` | **0.81** | 0.66 | 0.66 | 0.66 | 0.66 | 0.60 | 0.71 | 0.61 | 0.62 | 0.55 | 0.62 | 0.64 | 0.59 | 0.57 | 0.70 |
| `bbpb_arc` | 0.69 | 0.70 | 0.64 | 0.60 | 0.54 | 0.66 | **0.82** | 0.69 | 0.63 | 0.64 | 0.62 | 0.65 | 0.62 | 0.60 | 0.55 |
| `bbpb_include_v2_og_spanish_costa_rica` | 0.70 | 0.69 | 0.64 | 0.59 | 0.53 | 0.70 | 0.69 | 0.72 | 0.59 | 0.60 | 0.67 | 0.60 | 0.57 | 0.71 | 0.61 |
| `bbpb_xnli` | 0.71 | 0.68 | 0.70 | 0.54 | 0.64 | 0.66 | 0.58 | 0.61 | 0.65 | 0.72 | 0.60 | 0.69 | 0.53 | 0.71 | 0.57 |
| `bbpb_rf_include_base_44` | 0.53 | 0.64 | 0.69 | 0.69 | 0.58 | 0.59 | 0.74 | 0.61 | 0.62 | 0.62 | 0.63 | 0.68 | 0.61 | 0.74 | 0.60 |
| `bbpb_include_v2_og_spanish_per_` | 0.68 | 0.72 | 0.62 | 0.59 | 0.56 | 0.60 | 0.67 | 0.68 | 0.50 | 0.57 | 0.66 | 0.64 | 0.66 | 0.69 | 0.68 |
| `bbpb_include_v2_og_spanish_bolivia` | 0.70 | 0.70 | 0.65 | 0.61 | 0.64 | 0.59 | 0.69 | 0.66 | 0.55 | 0.57 | 0.65 | 0.66 | 0.61 | 0.65 | 0.61 |
| `bbpb_include_v2_og_spanish_m_xico` | 0.70 | 0.61 | 0.59 | 0.64 | 0.59 | 0.56 | 0.70 | **0.77** | 0.50 | 0.60 | 0.61 | 0.68 | 0.68 | 0.69 | 0.56 |
| `bbpb_include_v2_og_spanish_rep_blica_dominicana` | 0.62 | 0.70 | 0.50 | 0.67 | 0.60 | 0.67 | 0.60 | 0.67 | 0.68 | 0.55 | 0.59 | 0.65 | 0.57 | 0.63 | 0.74 |
| `bbpb_include_v2_og_spanish_panam_` | 0.68 | 0.60 | 0.68 | 0.67 | 0.57 | 0.64 | 0.69 | 0.59 | 0.66 | 0.61 | 0.61 | 0.65 | 0.55 | 0.60 | 0.57 |
| `multiblimp` | 0.75 | 0.62 | 0.53 | 0.60 | 0.61 | 0.65 | 0.62 | 0.64 | 0.70 | 0.52 | 0.63 | 0.63 | 0.55 | 0.64 | 0.67 |
| `bbpb_include_v2_og_spanish_nicaragua` | 0.58 | 0.70 | 0.64 | 0.60 | 0.59 | 0.51 | 0.60 | 0.56 | 0.46 | 0.67 | 0.59 | 0.69 | **0.75** | 0.73 | 0.68 |
| `rf_global_mmlu_full` | 0.55 | 0.41 | 0.47 | 0.66 | 0.56 | 0.64 | 0.69 | 0.69 | 0.74 | 0.65 | 0.64 | 0.61 | 0.67 | 0.63 | 0.70 |
| `arc_mt` | 0.56 | 0.67 | 0.60 | 0.47 | 0.49 | 0.61 | 0.68 | 0.57 | 0.66 | 0.64 | 0.60 | 0.67 | 0.65 | 0.70 | 0.70 |
| `bbpb_include_v2_og_spanish_jamaica` | 0.69 | 0.66 | 0.51 | 0.63 | 0.53 | 0.51 | 0.71 | 0.65 | 0.57 | 0.54 | 0.71 | 0.57 | 0.60 | 0.73 | 0.62 |
| `bbpb_belebele` | 0.64 | 0.63 | 0.68 | 0.71 | 0.58 | 0.61 | 0.70 | 0.56 | 0.52 | 0.67 | 0.54 | 0.50 | 0.58 | 0.57 | 0.59 |
| `bbpb_rf_cultural_bench_easy_argentina` | 0.60 | 0.57 | 0.60 | 0.56 | 0.58 | 0.57 | 0.52 | 0.64 | 0.56 | 0.69 | 0.60 | 0.57 | 0.66 | 0.66 | 0.72 |
| `bbpb_include_v2_en_spanish_per_` | 0.39 | 0.59 | 0.56 | 0.60 | **0.79** | 0.56 | 0.60 | 0.62 | 0.44 | 0.55 | 0.71 | 0.70 | 0.65 | 0.61 | 0.68 |
| `rfgm_belebele` | 0.61 | 0.64 | 0.65 | 0.47 | 0.58 | 0.59 | 0.61 | 0.59 | 0.64 | 0.65 | 0.60 | 0.65 | 0.60 | 0.50 | 0.64 |
| `bbpb_include_v2_og_spanish_paraguay` | 0.63 | 0.66 | 0.66 | 0.58 | 0.60 | 0.50 | 0.62 | 0.62 | 0.43 | 0.59 | 0.74 | 0.61 | 0.64 | 0.59 | 0.51 |
| `rf_belebele` | 0.61 | 0.54 | 0.60 | 0.53 | 0.60 | 0.62 | 0.59 | 0.61 | 0.70 | 0.60 | 0.63 | 0.58 | 0.67 | 0.57 | 0.51 |
| `arc` | 0.60 | 0.55 | 0.62 | 0.61 | 0.65 | 0.55 | 0.61 | 0.61 | 0.61 | 0.54 | 0.44 | 0.54 | 0.60 | **0.75** | 0.64 |
| `bbpb_include_v2_og_spanish_venezuela` | 0.65 | 0.70 | 0.50 | 0.56 | 0.58 | 0.58 | 0.60 | **0.75** | 0.58 | 0.50 | 0.56 | 0.54 | 0.56 | 0.68 | 0.57 |
| `rf_m_mmlu` | 0.60 | 0.50 | 0.54 | 0.53 | 0.55 | 0.58 | 0.51 | 0.64 | 0.59 | 0.67 | 0.57 | 0.63 | 0.58 | **0.77** | 0.64 |
| `include_v2_og_spanish_m_xico` | 0.63 | 0.45 | 0.51 | 0.47 | 0.55 | 0.56 | 0.60 | 0.52 | 0.60 | 0.67 | 0.65 | 0.65 | 0.63 | 0.71 | 0.60 |
| `bbpb_include_v2_en_spanish_espa_a` | 0.47 | 0.62 | 0.69 | 0.58 | 0.65 | 0.56 | 0.57 | 0.53 | 0.51 | 0.60 | 0.59 | 0.64 | 0.57 | 0.54 | 0.66 |
| `rf_include_base_44` | 0.63 | 0.66 | 0.64 | 0.62 | 0.58 | 0.59 | 0.50 | 0.59 | 0.54 | 0.54 | 0.54 | 0.53 | 0.52 | 0.64 | 0.62 |
| `bbpb_include_v2_en_spanish_m_xico` | 0.42 | 0.61 | 0.56 | 0.51 | 0.68 | 0.56 | 0.65 | 0.74 | 0.45 | 0.58 | 0.54 | 0.69 | **0.76** | 0.50 | 0.47 |
| `bbpb_cultural_bench_easy_chile` | 0.56 | 0.54 | 0.64 | 0.56 | 0.54 | 0.62 | 0.67 | 0.69 | 0.49 | 0.59 | 0.67 | 0.58 | 0.61 | 0.41 | 0.51 |
| `bbpb_cultural_bench_easy_argentina` | 0.58 | 0.55 | 0.61 | 0.55 | 0.70 | 0.57 | 0.51 | 0.62 | 0.61 | 0.58 | 0.60 | 0.54 | 0.60 | 0.49 | 0.51 |
| `bbpb_include_v2_en_spanish_cuba` | 0.44 | 0.55 | 0.48 | 0.53 | 0.65 | 0.62 | 0.60 | 0.55 | 0.58 | 0.70 | 0.46 | 0.64 | 0.61 | 0.61 | 0.58 |
| `bbpb_include_v2_en_spanish_rep_blica_dominicana` | 0.63 | 0.67 | 0.64 | 0.57 | 0.64 | 0.50 | 0.61 | 0.56 | 0.52 | 0.54 | 0.50 | 0.63 | 0.55 | 0.50 | 0.52 |
| `bbpb_include_v2_og_spanish_belice` | 0.69 | 0.66 | 0.52 | 0.62 | 0.50 | 0.50 | 0.63 | 0.50 | 0.58 | 0.44 | 0.68 | 0.56 | 0.57 | 0.68 | 0.48 |
| `bbpb_rf_cultural_bench_easy_mexico` | 0.58 | 0.63 | 0.52 | 0.51 | 0.60 | 0.60 | 0.52 | 0.64 | 0.55 | 0.60 | 0.67 | 0.60 | 0.44 | 0.59 | 0.49 |
| `bbpb_cultural_bench_easy_mexico` | 0.53 | 0.49 | 0.66 | 0.46 | 0.68 | 0.63 | 0.60 | 0.60 | 0.58 | 0.60 | 0.66 | 0.54 | 0.62 | 0.42 | 0.46 |
| `bbpb_include_v2_en_spanish_costa_rica` | 0.50 | 0.58 | 0.62 | 0.53 | 0.63 | 0.54 | 0.62 | 0.60 | 0.40 | 0.61 | 0.54 | 0.56 | 0.59 | 0.55 | 0.64 |
| `bbpb_include_v2_en_spanish_belice` | 0.52 | 0.65 | 0.55 | 0.57 | 0.65 | 0.45 | 0.49 | 0.43 | 0.48 | 0.60 | 0.53 | 0.67 | 0.72 | 0.60 | 0.53 |
| `include_v2_og_spanish_bolivia` | 0.64 | 0.50 | 0.58 | 0.58 | 0.48 | 0.57 | 0.53 | 0.73 | 0.44 | 0.46 | 0.67 | 0.58 | 0.54 | 0.54 | 0.58 |
| `rfgm_include_base_44` | 0.59 | 0.60 | 0.55 | 0.56 | 0.50 | 0.54 | 0.63 | 0.55 | 0.58 | 0.47 | 0.55 | 0.60 | 0.57 | 0.57 | 0.52 |
| `bbpb_include_v2_en_spanish_chile` | 0.51 | 0.47 | 0.62 | 0.52 | 0.60 | 0.59 | 0.57 | 0.67 | 0.50 | 0.57 | 0.56 | 0.50 | 0.50 | 0.61 | 0.57 |
| `include_v2_og_spanish_ecuador` | 0.66 | 0.51 | 0.62 | 0.62 | 0.63 | 0.62 | 0.65 | 0.46 | 0.53 | 0.59 | 0.56 | 0.49 | 0.43 | 0.44 | 0.54 |
| `bbpb_rf_cultural_bench_easy_spain` | 0.69 | 0.69 | 0.51 | 0.58 | 0.50 | 0.59 | 0.62 | 0.51 | 0.48 | 0.51 | 0.53 | 0.48 | 0.56 | 0.45 | 0.66 |
| `bbpb_include_v2_en_spanish_argentina` | 0.50 | 0.59 | 0.61 | 0.52 | 0.58 | 0.56 | 0.45 | 0.60 | 0.47 | 0.54 | 0.61 | 0.59 | 0.48 | 0.52 | 0.68 |
| `include_v2_en_spanish_colombia` | 0.42 | 0.55 | 0.56 | 0.58 | 0.41 | 0.51 | 0.61 | 0.61 | 0.54 | 0.59 | 0.59 | 0.50 | 0.64 | 0.63 | 0.55 |
| `bbpb_include_v2_en_spanish_colombia` | 0.47 | 0.57 | 0.67 | 0.48 | 0.65 | 0.60 | 0.60 | 0.60 | 0.52 | 0.47 | 0.57 | 0.53 | 0.53 | 0.50 | 0.51 |
| `include_v2_og_spanish_el_salvador` | 0.57 | 0.53 | 0.47 | 0.60 | 0.71 | 0.57 | 0.52 | 0.57 | 0.59 | 0.43 | 0.43 | 0.54 | 0.57 | 0.55 | 0.61 |
| `bbpb_include_v2_en_spanish_nicaragua` | 0.43 | 0.64 | 0.48 | 0.64 | 0.60 | 0.41 | 0.53 | 0.49 | 0.44 | 0.54 | 0.60 | 0.72 | 0.63 | 0.60 | 0.50 |
| `include_v2_en_spanish_m_xico` | 0.47 | 0.64 | 0.63 | 0.57 | 0.49 | 0.45 | 0.54 | 0.56 | 0.47 | 0.67 | 0.46 | 0.60 | 0.59 | 0.60 | 0.51 |
| `bbpb_include_v2_og_spanish_puerto_rico` | 0.51 | 0.60 | 0.45 | 0.53 | 0.48 | 0.49 | 0.59 | 0.64 | 0.52 | 0.42 | 0.52 | 0.57 | 0.59 | 0.68 | 0.64 |
| `bbpb_cultural_bench_easy_peru` | 0.50 | 0.45 | 0.57 | 0.46 | 0.59 | 0.60 | 0.59 | 0.63 | 0.62 | 0.63 | 0.62 | 0.49 | 0.60 | 0.41 | 0.48 |
| `bbpb_include_v2_en_spanish_venezuela` | 0.43 | 0.54 | 0.46 | 0.50 | 0.60 | 0.45 | 0.63 | 0.73 | 0.51 | 0.51 | 0.52 | 0.59 | 0.61 | 0.60 | 0.54 |
| `bbpb_rf_cultural_bench_easy_peru` | 0.69 | 0.71 | 0.58 | 0.42 | 0.43 | 0.61 | 0.55 | 0.43 | 0.45 | 0.58 | 0.40 | 0.57 | 0.61 | 0.50 | 0.60 |
| `bbpb_cultural_bench_easy_spain` | 0.50 | 0.54 | 0.57 | 0.51 | 0.61 | 0.58 | 0.52 | 0.60 | 0.52 | 0.56 | 0.61 | 0.51 | 0.57 | 0.44 | 0.45 |
| `include_v2_og_spanish_chile` | 0.48 | 0.40 | 0.48 | 0.52 | 0.50 | 0.54 | 0.57 | 0.63 | 0.43 | 0.60 | 0.57 | 0.63 | 0.66 | 0.56 | 0.53 |
| `bbpb_include_v2_en_spanish_el_salvador` | 0.41 | 0.53 | 0.56 | 0.56 | 0.62 | 0.44 | 0.51 | 0.60 | 0.42 | 0.54 | 0.59 | 0.60 | 0.59 | 0.54 | 0.51 |
| `include_v2_en_spanish_espa_a` | 0.57 | 0.54 | 0.49 | 0.51 | 0.51 | 0.40 | 0.47 | 0.45 | 0.45 | 0.62 | 0.61 | 0.62 | 0.45 | 0.59 | 0.66 |
| `bbpb_include_v2_en_spanish_bolivia` | 0.45 | 0.47 | 0.57 | 0.51 | 0.59 | 0.42 | 0.62 | 0.69 | 0.52 | 0.45 | 0.45 | 0.64 | 0.51 | 0.51 | 0.52 |
| `xnli` | 0.50 | 0.48 | 0.43 | 0.51 | 0.50 | 0.55 | 0.53 | 0.51 | 0.60 | 0.50 | 0.46 | 0.56 | 0.65 | 0.53 | 0.57 |
| `bbpb_rf_cultural_bench_easy_chile` | 0.57 | 0.59 | 0.58 | 0.43 | 0.57 | 0.55 | 0.59 | 0.50 | 0.44 | 0.51 | 0.53 | 0.56 | 0.33 | 0.63 | 0.50 |
| `include_v2_og_spanish_per_` | 0.59 | 0.55 | 0.56 | 0.57 | 0.50 | 0.50 | 0.63 | 0.56 | 0.50 | 0.63 | 0.42 | 0.42 | 0.38 | 0.38 | 0.69 |
| `include_v2_og_spanish_colombia` | 0.41 | 0.43 | 0.40 | 0.59 | 0.36 | 0.57 | 0.53 | 0.45 | 0.64 | 0.58 | 0.57 | 0.56 | 0.53 | 0.64 | 0.57 |
| `bbpb_include_v2_en_spanish_uruguay` | 0.51 | 0.55 | 0.57 | 0.58 | 0.66 | 0.46 | 0.39 | 0.61 | 0.41 | 0.42 | 0.52 | 0.56 | 0.49 | 0.49 | 0.57 |
| `include_v2_en_spanish_ecuador` | 0.52 | 0.49 | 0.56 | 0.57 | 0.56 | 0.54 | 0.53 | 0.50 | 0.57 | 0.55 | 0.45 | 0.43 | 0.58 | 0.42 | 0.52 |
| `include_v2_en_spanish_per_` | 0.41 | 0.52 | 0.47 | 0.61 | 0.62 | 0.53 | 0.54 | 0.49 | 0.42 | 0.42 | 0.61 | 0.55 | 0.55 | 0.38 | 0.51 |
| `paws` | 0.54 | 0.49 | 0.35 | 0.54 | 0.49 | 0.56 | 0.39 | 0.41 | 0.67 | 0.57 | 0.49 | 0.57 | 0.51 | 0.40 | 0.58 |
| `bbpb_include_v2_en_spanish_paraguay` | 0.37 | 0.45 | 0.58 | 0.50 | 0.56 | 0.35 | 0.58 | 0.56 | 0.39 | 0.45 | 0.44 | 0.70 | 0.60 | 0.48 | 0.52 |
| `bbpb_include_v2_en_spanish_jamaica` | 0.51 | 0.62 | 0.49 | 0.46 | 0.64 | 0.39 | 0.60 | 0.33 | 0.52 | 0.41 | 0.43 | 0.57 | 0.53 | 0.56 | 0.50 |
| `include_v2_en_spanish_argentina` | 0.60 | 0.47 | 0.50 | 0.65 | 0.43 | 0.38 | 0.52 | 0.52 | 0.52 | 0.47 | 0.45 | 0.40 | 0.51 | 0.61 | 0.43 |
| `include_v2_en_spanish_uruguay` | 0.48 | 0.42 | 0.54 | 0.51 | 0.48 | 0.43 | 0.58 | 0.50 | 0.55 | 0.47 | 0.42 | 0.42 | 0.49 | 0.62 | 0.50 |
| `bbpb_include_v2_en_spanish_puerto_rico` | 0.32 | 0.54 | 0.48 | 0.50 | 0.66 | 0.43 | 0.62 | 0.60 | 0.41 | 0.44 | 0.37 | 0.54 | 0.54 | 0.42 | 0.55 |
| `bbpb_blend_spain_sample` | 0.51 | 0.50 | 0.51 | 0.43 | 0.65 | 0.40 | 0.42 | 0.50 | 0.55 | 0.67 | 0.40 | 0.50 | 0.42 | 0.55 | 0.38 |
| `include_v2_en_spanish_bolivia` | 0.49 | 0.35 | 0.57 | 0.47 | 0.44 | 0.46 | 0.45 | 0.45 | 0.70 | 0.58 | 0.53 | 0.47 | 0.61 | 0.42 | 0.39 |
| `include_base_44` | 0.50 | 0.53 | 0.37 | 0.48 | 0.37 | 0.52 | 0.51 | 0.62 | 0.55 | 0.51 | 0.50 | 0.43 | 0.48 | 0.47 | 0.52 |
| `cultural_bench_hard_argentina` | 0.50 | 0.45 | 0.56 | 0.35 | 0.70 | 0.57 | 0.34 | 0.44 | 0.53 | 0.35 | 0.44 | 0.43 | 0.61 | 0.60 | 0.48 |
| `cultural_bench_hard_mexico` | 0.53 | 0.50 | 0.53 | 0.34 | 0.64 | 0.61 | 0.37 | 0.43 | 0.54 | 0.43 | 0.49 | 0.53 | 0.45 | 0.59 | 0.37 |
| `include_v2_en_spanish_cuba` | 0.43 | 0.53 | 0.43 | 0.43 | 0.48 | 0.48 | 0.59 | 0.61 | 0.46 | 0.58 | 0.31 | 0.45 | 0.52 | 0.51 | 0.52 |
| `bbpb_blend_mexico_sample` | 0.49 | 0.51 | 0.52 | 0.46 | 0.71 | 0.36 | 0.42 | 0.47 | 0.55 | 0.60 | 0.38 | 0.50 | 0.37 | 0.56 | 0.41 |
| `include_v2_og_spanish_cuba` | 0.59 | 0.53 | 0.46 | 0.43 | 0.50 | 0.62 | 0.45 | 0.49 | 0.38 | 0.55 | 0.55 | 0.39 | 0.45 | 0.46 | 0.48 |
| `blend_spain_sample` | 0.50 | 0.50 | 0.48 | 0.42 | 0.40 | 0.44 | 0.60 | 0.51 | 0.51 | 0.52 | 0.45 | 0.51 | 0.42 | 0.48 | 0.48 |
| `global_mmlu_full` | 0.34 | 0.50 | 0.45 | 0.60 | 0.41 | 0.58 | 0.48 | 0.47 | 0.41 | 0.51 | 0.53 | 0.42 | 0.50 | 0.57 | 0.50 |
| `include_v2_og_spanish_uruguay` | 0.50 | 0.46 | 0.45 | 0.65 | 0.53 | 0.37 | 0.50 | 0.41 | 0.57 | 0.37 | 0.59 | 0.46 | 0.36 | 0.51 | 0.50 |
| `include_v2_en_spanish_el_salvador` | 0.50 | 0.44 | 0.46 | 0.43 | 0.56 | 0.43 | 0.50 | 0.43 | 0.52 | 0.59 | 0.46 | 0.42 | 0.41 | 0.49 | 0.59 |
| `bbpb_include_v2_en_spanish_panam_` | 0.50 | 0.62 | 0.42 | 0.51 | 0.61 | 0.37 | 0.40 | 0.60 | 0.59 | 0.48 | 0.35 | 0.50 | 0.33 | 0.38 | 0.53 |
| `include_v2_en_spanish_chile` | 0.62 | 0.29 | 0.61 | 0.44 | 0.36 | 0.38 | 0.55 | 0.50 | 0.43 | 0.36 | 0.48 | 0.60 | 0.56 | 0.47 | 0.55 |
| `include_v2_og_spanish_paraguay` | 0.48 | 0.52 | 0.51 | 0.31 | 0.42 | 0.53 | 0.48 | 0.50 | 0.62 | 0.39 | 0.43 | 0.45 | 0.52 | 0.48 | 0.52 |
| `include_v2_og_spanish_argentina` | 0.52 | 0.39 | 0.40 | 0.57 | 0.62 | 0.53 | 0.53 | 0.46 | 0.39 | 0.46 | 0.63 | 0.37 | 0.32 | 0.37 | 0.55 |
| `truthfulqa-multi_mc1` | 0.47 | 0.55 | 0.50 | 0.38 | 0.50 | 0.40 | 0.43 | 0.48 | 0.38 | 0.50 | 0.47 | 0.52 | 0.49 | 0.55 | 0.44 |
| `include_v2_og_spanish_venezuela` | 0.54 | 0.44 | 0.56 | 0.47 | 0.41 | 0.41 | 0.52 | 0.51 | 0.41 | 0.41 | 0.52 | 0.38 | 0.48 | 0.56 | 0.43 |
| `cultural_bench_easy_mexico` | 0.44 | 0.43 | 0.50 | 0.43 | 0.50 | 0.56 | 0.52 | 0.37 | 0.58 | 0.43 | 0.43 | 0.47 | 0.50 | 0.48 | 0.39 |
| `cultural_bench_hard_chile` | 0.49 | 0.37 | 0.50 | 0.40 | 0.59 | 0.59 | 0.42 | 0.47 | 0.46 | 0.31 | 0.46 | 0.28 | 0.51 | 0.57 | 0.57 |
| `blend_mexico_sample` | 0.46 | 0.64 | 0.50 | 0.43 | 0.56 | 0.40 | 0.38 | 0.47 | 0.42 | 0.56 | 0.35 | 0.63 | 0.53 | 0.37 | 0.28 |
| `cultural_bench_hard_spain` | 0.49 | 0.46 | 0.51 | 0.30 | 0.57 | 0.45 | 0.43 | 0.52 | 0.58 | 0.52 | 0.37 | 0.42 | 0.46 | 0.43 | 0.45 |
| `belebele` | 0.49 | 0.37 | 0.46 | 0.36 | 0.48 | 0.52 | 0.43 | 0.39 | 0.40 | 0.59 | 0.42 | 0.46 | 0.56 | 0.48 | 0.52 |
| `include_v2_og_spanish_costa_rica` | 0.46 | 0.43 | 0.32 | 0.43 | 0.52 | 0.48 | 0.54 | 0.50 | 0.40 | 0.49 | 0.55 | 0.48 | 0.40 | 0.36 | 0.50 |
| `include_v2_en_spanish_paraguay` | 0.52 | 0.41 | 0.44 | 0.39 | 0.48 | 0.45 | 0.44 | 0.44 | 0.51 | 0.55 | 0.29 | 0.54 | 0.56 | 0.47 | 0.35 |
| `include_v2_en_spanish_costa_rica` | 0.57 | 0.42 | 0.42 | 0.49 | 0.33 | 0.33 | 0.41 | 0.44 | 0.45 | 0.48 | 0.44 | 0.39 | 0.57 | 0.50 | 0.51 |
| `cultural_bench_easy_spain` | 0.50 | 0.40 | 0.48 | 0.47 | 0.46 | 0.60 | 0.40 | 0.46 | 0.38 | 0.37 | 0.38 | 0.44 | 0.54 | 0.42 | 0.38 |
| `include_v2_en_spanish_puerto_rico` | 0.40 | 0.52 | 0.41 | 0.30 | 0.27 | 0.51 | 0.60 | 0.36 | 0.47 | 0.37 | 0.45 | 0.50 | 0.44 | 0.47 | 0.59 |
| `cultural_bench_hard_peru` | 0.54 | 0.50 | 0.37 | 0.39 | 0.55 | 0.50 | 0.32 | 0.60 | 0.38 | 0.36 | 0.37 | 0.34 | 0.50 | 0.44 | 0.40 |
| `include_v2_og_spanish_nicaragua` | 0.43 | 0.38 | 0.47 | 0.36 | 0.34 | 0.43 | 0.53 | 0.39 | 0.50 | 0.42 | 0.48 | 0.52 | 0.36 | 0.47 | 0.46 |
| `global_piqa_parallel_cloze_spa_latn_peru` | 0.54 | 0.42 | 0.39 | 0.31 | 0.31 | 0.39 | 0.41 | 0.33 | 0.36 | 0.55 | 0.52 | 0.46 | 0.45 | 0.60 | 0.44 |
| `include_v2_en_spanish_venezuela` | 0.40 | 0.32 | 0.50 | 0.41 | 0.51 | 0.44 | 0.37 | 0.43 | 0.32 | 0.36 | 0.52 | 0.34 | 0.52 | 0.57 | 0.44 |
| `cultural_bench_easy_peru` | 0.33 | 0.39 | 0.42 | 0.31 | 0.39 | 0.59 | 0.39 | 0.38 | 0.52 | 0.47 | 0.49 | 0.45 | 0.48 | 0.43 | 0.42 |
| `cultural_bench_easy_argentina` | 0.48 | 0.40 | 0.40 | 0.30 | 0.56 | 0.48 | 0.28 | 0.46 | 0.42 | 0.45 | 0.40 | 0.53 | 0.43 | 0.54 | 0.31 |
| `global_piqa_parallel_cloze_spa_latn_mexi` | 0.45 | 0.31 | 0.40 | 0.49 | 0.42 | 0.51 | 0.35 | 0.43 | 0.42 | 0.49 | 0.50 | 0.39 | 0.36 | 0.42 | 0.45 |
| `global_piqa_parallel_cloze_spa_latn_spai` | 0.34 | 0.32 | 0.43 | 0.34 | 0.45 | 0.47 | 0.30 | 0.57 | 0.36 | 0.40 | 0.36 | 0.46 | 0.46 | 0.59 | 0.51 |
| `cultural_bench_easy_chile` | 0.51 | 0.31 | 0.28 | 0.40 | 0.43 | 0.52 | 0.39 | 0.48 | 0.45 | 0.42 | 0.41 | 0.47 | 0.36 | 0.41 | 0.45 |
| `include_v2_og_spanish_puerto_rico` | 0.41 | 0.41 | 0.57 | 0.35 | 0.39 | 0.45 | 0.36 | 0.44 | 0.35 | 0.36 | 0.31 | 0.56 | 0.45 | 0.35 | 0.41 |
| `include_v2_og_spanish_rep_blica_dominicana` | 0.48 | 0.42 | 0.50 | 0.38 | 0.29 | 0.54 | 0.45 | 0.37 | 0.42 | 0.36 | 0.42 | 0.26 | 0.39 | 0.37 | 0.32 |
| `include_v2_en_spanish_nicaragua` | 0.30 | 0.35 | 0.48 | 0.40 | 0.42 | 0.37 | 0.42 | 0.33 | 0.30 | 0.37 | 0.42 | 0.35 | 0.49 | 0.45 | 0.42 |
| `include_v2_og_spanish_panam_` | 0.42 | 0.40 | 0.43 | 0.50 | 0.44 | 0.21 | 0.48 | 0.33 | 0.37 | 0.33 | 0.37 | 0.40 | 0.46 | 0.32 | 0.37 |
| `rf_cultural_bench_easy_mexico` | 0.45 | 0.24 | 0.34 | 0.43 | 0.46 | 0.30 | 0.43 | 0.45 | 0.44 | 0.40 | 0.31 | 0.29 | 0.50 | 0.40 | 0.32 |
| `rf_cultural_bench_easy_peru` | 0.47 | 0.36 | 0.43 | 0.29 | 0.45 | 0.28 | 0.31 | 0.41 | 0.53 | 0.40 | 0.38 | 0.23 | 0.27 | 0.31 | 0.40 |
| `rf_cultural_bench_easy_spain` | 0.39 | 0.32 | 0.47 | 0.30 | 0.27 | 0.51 | 0.35 | 0.30 | 0.38 | 0.36 | 0.36 | 0.47 | 0.31 | 0.27 | 0.38 |
| `include_v2_en_spanish_panam_` | 0.40 | 0.28 | 0.30 | 0.23 | 0.37 | 0.18 | 0.37 | 0.39 | 0.25 | 0.44 | 0.40 | 0.52 | 0.48 | 0.47 | 0.34 |
| `include_v2_og_spanish_jamaica` | 0.34 | 0.40 | 0.32 | 0.39 | 0.30 | 0.43 | 0.45 | 0.32 | 0.34 | 0.37 | 0.33 | 0.34 | 0.40 | 0.29 | 0.39 |
| `include_v2_og_spanish_belice` | 0.36 | 0.21 | 0.41 | 0.27 | 0.32 | 0.37 | 0.30 | 0.33 | 0.56 | 0.40 | 0.47 | 0.40 | 0.31 | 0.30 | 0.30 |
| `rf_cultural_bench_easy_chile` | 0.37 | 0.45 | 0.39 | 0.32 | 0.32 | 0.32 | 0.47 | 0.21 | 0.43 | 0.39 | 0.25 | 0.31 | 0.30 | 0.42 | 0.38 |
| `include_v2_en_spanish_rep_blica_dominicana` | 0.44 | 0.42 | 0.20 | 0.30 | 0.20 | 0.46 | 0.28 | 0.38 | 0.21 | 0.23 | 0.40 | 0.30 | 0.35 | 0.44 | 0.30 |
| `rf_cultural_bench_easy_argentina` | 0.28 | 0.26 | 0.22 | 0.20 | 0.43 | 0.38 | 0.40 | 0.36 | 0.26 | 0.30 | 0.40 | 0.23 | 0.46 | 0.35 | 0.37 |
| `include_v2_en_spanish_jamaica` | 0.44 | 0.32 | 0.37 | 0.26 | 0.40 | 0.42 | 0.22 | 0.33 | 0.27 | 0.29 | 0.33 | 0.27 | 0.21 | 0.26 | 0.37 |
| `include_v2_en_spanish_belice` | 0.29 | 0.25 | 0.30 | 0.30 | 0.29 | 0.32 | 0.24 | 0.30 | 0.34 | 0.26 | 0.35 | 0.30 | 0.41 | 0.33 | 0.29 |

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
subtitle: "Estonian (et) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_multiblimp` | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** |
| `bbpb_rf_belebele` | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** |
| `bbpb_rfgm_belebele` | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** |
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_rfgm_include_base_44` | 0.67 | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | 0.67 | 0.67 | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** |
| `bbpb_xcopa` | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | 0.67 | **1.00** | 0.67 | 0.67 |
| `bbpb_global_piqa_parallel_cloze` | **0.83** | **0.83** | **1.00** | 0.67 | 0.67 | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** |
| `multiblimp` | **0.83** | **1.00** | 0.67 | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | 0.67 | **0.83** | **1.00** | **0.83** | **0.83** |
| `bbpb_include_v2_og` | **0.83** | **1.00** | 0.50 | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** | 0.67 | 0.50 | **1.00** | **0.83** | 0.50 | 0.67 | **0.83** |
| `bbpb_belebele` | **0.83** | **0.83** | **0.83** | 0.67 | **0.83** | **1.00** | **1.00** | 0.50 | 0.67 | **1.00** | 0.50 | 0.67 | 0.50 | 0.67 | **0.83** |
| `bbpb_rf_include_base_44` | **1.00** | 0.67 | **1.00** | **0.83** | 0.50 | 0.67 | **1.00** | **0.83** | 0.50 | 0.67 | 0.50 | 0.17 | **0.83** | 0.50 | 0.67 |
| `belebele` | **0.83** | 0.67 | 0.33 | **0.83** | **1.00** | **0.83** | 0.17 | 0.67 | **0.83** | 0.33 | 0.50 | 0.67 | 0.50 | 0.33 | **0.83** |
| `include_v2_og` | **0.83** | 0.67 | 0.67 | 0.67 | 0.67 | **0.83** | 0.50 | 0.50 | 0.50 | 0.33 | 0.33 | 0.33 | 0.67 | **1.00** | 0.67 |
| `rf_include_base_44` | 0.67 | 0.17 | **0.83** | 0.67 | 0.50 | 0.50 | 0.50 | 0.67 | **0.83** | 0.33 | 0.50 | 0.67 | **0.83** | 0.67 | **0.83** |
| `xcopa` | 0.67 | 0.50 | 0.17 | 0.67 | 0.67 | 0.50 | 0.50 | 0.67 | 0.67 | 0.00 | 0.33 | 0.33 | 0.50 | 0.50 | **1.00** |
| `bbpb_include_v2_en` | 0.33 | **1.00** | 0.67 | 0.33 | 0.67 | 0.33 | 0.33 | 0.67 | 0.33 | 0.67 | 0.33 | 0.67 | 0.67 | 0.33 | 0.00 |
| `rf_belebele` | **0.83** | 0.50 | 0.17 | 0.33 | 0.00 | 0.67 | 0.33 | 0.50 | 0.00 | 0.67 | **0.83** | 0.33 | **0.83** | 0.67 | 0.50 |
| `rfgm_include_base_44` | 0.67 | 0.33 | 0.67 | 0.50 | 0.50 | 0.33 | 0.67 | 0.33 | **0.83** | 0.17 | 0.17 | 0.50 | 0.50 | 0.50 | 0.17 |
| `global_piqa_parallel_cloze` | 0.67 | 0.50 | 0.67 | 0.67 | 0.33 | 0.67 | 0.67 | 0.33 | 0.17 | 0.50 | 0.17 | 0.33 | 0.67 | 0.00 | 0.33 |
| `bbpb_include_base_44` | **0.83** | 0.33 | 0.17 | **0.83** | 0.17 | 0.17 | 0.00 | 0.67 | 0.33 | **0.83** | 0.50 | 0.50 | 0.33 | 0.67 | 0.33 |
| `include_base_44` | 0.17 | 0.50 | **1.00** | 0.50 | 0.33 | 0.17 | 0.17 | 0.67 | **0.83** | 0.50 | 0.00 | 0.00 | 0.50 | 0.33 | **0.83** |
| `rfgm_belebele` | 0.33 | 0.33 | **1.00** | 0.17 | 0.33 | 0.67 | 0.33 | 0.33 | 0.17 | 0.33 | 0.33 | 0.50 | 0.17 | 0.33 | **0.83** |
| `include_v2_en` | 0.17 | 0.17 | **0.83** | 0.17 | 0.50 | 0.33 | 0.17 | **1.00** | 0.50 | 0.33 | 0.33 | 0.17 | 0.17 | 0.67 | 0.50 |

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
subtitle: "Estonian (et) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Persian (fa) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.97** | **0.84** | **0.83** | **0.88** | **0.93** | **0.86** | **0.84** | **0.90** | **0.93** | **0.86** | **0.87** | **0.86** | **0.87** | **0.84** | **0.94** |
| `bbpb_rfgm_include_base_44` | **0.83** | **0.78** | **0.77** | **0.76** | **0.80** | **0.79** | **0.82** | **0.77** | **0.79** | **0.83** | **0.87** | **0.82** | **0.85** | **0.82** | **0.78** |
| `bbpb_rf_include_base_44` | 0.68 | **0.79** | 0.68 | **0.76** | 0.74 | 0.74 | **0.81** | **0.76** | 0.73 | 0.72 | 0.74 | **0.79** | **0.76** | **0.79** | 0.71 |
| `bbpb_rfgm_belebele` | **0.80** | 0.75 | 0.72 | 0.75 | 0.69 | **0.78** | 0.74 | **0.77** | 0.73 | 0.69 | **0.76** | **0.80** | **0.78** | 0.69 | 0.74 |
| `bbpb_global_piqa_parallel_cloze` | **0.87** | 0.74 | 0.74 | 0.74 | **0.78** | **0.78** | 0.74 | 0.72 | **0.76** | 0.68 | 0.66 | 0.71 | **0.75** | 0.69 | 0.66 |
| `bbpb_rf_belebele` | 0.75 | 0.70 | 0.72 | 0.74 | 0.61 | **0.77** | 0.69 | 0.63 | 0.57 | 0.72 | 0.67 | 0.56 | 0.67 | 0.62 | 0.63 |
| `bbpb_multiblimp` | 0.73 | 0.74 | **0.84** | 0.74 | 0.59 | 0.72 | 0.67 | 0.66 | 0.51 | 0.73 | **0.83** | 0.48 | 0.68 | 0.53 | 0.50 |
| `bbpb_rf_global_mmlu_full` | 0.73 | 0.56 | 0.74 | **0.78** | 0.69 | 0.63 | 0.63 | **0.78** | 0.68 | 0.52 | 0.56 | 0.52 | 0.59 | 0.64 | **0.78** |
| `multiblimp` | **0.76** | 0.71 | 0.64 | 0.62 | 0.64 | 0.71 | 0.61 | 0.62 | 0.63 | 0.65 | 0.60 | 0.69 | 0.66 | 0.61 | 0.66 |
| `bbpb_global_mmlu_full` | 0.73 | 0.66 | 0.62 | 0.64 | 0.66 | 0.66 | 0.54 | 0.64 | 0.67 | 0.52 | 0.56 | 0.63 | 0.64 | 0.66 | 0.69 |
| `bbpb_toksuite` | 0.56 | 0.62 | 0.66 | 0.58 | 0.49 | 0.64 | 0.63 | 0.59 | 0.54 | 0.56 | 0.59 | 0.63 | 0.58 | 0.61 | 0.52 |
| `bbpb_include_base_44` | 0.64 | 0.47 | 0.55 | 0.54 | 0.64 | 0.53 | 0.61 | 0.66 | 0.65 | 0.53 | 0.50 | 0.59 | 0.66 | 0.62 | 0.56 |
| `bbpb_belebele` | 0.60 | 0.56 | 0.70 | 0.70 | 0.52 | 0.55 | 0.59 | 0.54 | 0.51 | 0.63 | 0.58 | 0.49 | 0.65 | 0.56 | 0.46 |
| `rfgm_include_base_44` | 0.53 | 0.45 | 0.55 | 0.62 | 0.50 | 0.43 | 0.61 | 0.61 | **0.76** | 0.41 | 0.43 | 0.31 | 0.60 | 0.58 | 0.58 |
| `bbpb_include_v2_en` | 0.61 | 0.53 | 0.53 | 0.59 | 0.64 | 0.52 | 0.57 | 0.49 | 0.42 | 0.54 | 0.51 | 0.48 | 0.54 | 0.43 | 0.56 |
| `include_base_44` | 0.42 | 0.55 | 0.48 | 0.36 | 0.45 | 0.63 | 0.42 | 0.56 | 0.52 | 0.58 | 0.52 | 0.39 | 0.65 | 0.58 | 0.63 |
| `rf_include_base_44` | 0.51 | 0.40 | 0.39 | 0.43 | 0.41 | 0.59 | 0.55 | 0.56 | 0.49 | 0.53 | 0.51 | 0.61 | 0.55 | 0.61 | 0.53 |
| `global_mmlu_full` | 0.44 | 0.59 | 0.56 | 0.40 | 0.52 | 0.52 | 0.47 | 0.42 | 0.49 | 0.57 | 0.45 | 0.56 | 0.51 | 0.59 | 0.55 |
| `bbpb_include_v2_og` | 0.55 | 0.51 | 0.41 | 0.47 | 0.59 | 0.45 | 0.53 | 0.51 | 0.57 | 0.59 | 0.32 | 0.54 | 0.42 | 0.53 | 0.65 |
| `belebele` | 0.46 | 0.46 | 0.54 | 0.50 | 0.48 | 0.51 | 0.42 | 0.47 | 0.58 | 0.38 | 0.44 | 0.52 | 0.58 | 0.55 | 0.69 |
| `rf_global_mmlu_full` | 0.38 | 0.51 | 0.51 | 0.50 | 0.37 | 0.52 | 0.44 | 0.52 | 0.50 | 0.52 | 0.54 | 0.55 | 0.51 | 0.56 | 0.65 |
| `include_v2_og` | 0.48 | 0.51 | 0.67 | 0.45 | 0.61 | 0.53 | 0.44 | 0.41 | 0.54 | 0.49 | 0.37 | 0.45 | 0.48 | 0.60 | 0.49 |
| `rfgm_belebele` | 0.51 | 0.50 | 0.55 | 0.56 | 0.56 | 0.38 | 0.38 | 0.49 | 0.48 | 0.55 | 0.43 | 0.49 | 0.49 | 0.41 | 0.66 |
| `rf_belebele` | 0.30 | 0.38 | 0.44 | 0.37 | 0.41 | 0.49 | 0.51 | 0.54 | 0.65 | 0.63 | 0.51 | 0.50 | 0.59 | 0.56 | 0.51 |
| `include_v2_en` | 0.55 | 0.38 | 0.36 | 0.36 | 0.48 | 0.58 | 0.51 | 0.40 | 0.52 | 0.57 | 0.57 | 0.42 | 0.58 | 0.46 | 0.55 |
| `toksuite` | 0.41 | 0.47 | 0.34 | 0.45 | 0.31 | 0.47 | 0.46 | 0.38 | 0.39 | 0.44 | 0.38 | 0.45 | 0.43 | 0.44 | 0.35 |
| `global_piqa_parallel_cloze` | 0.44 | 0.42 | 0.56 | 0.35 | 0.44 | 0.40 | 0.54 | 0.26 | 0.48 | 0.55 | 0.34 | 0.27 | 0.35 | 0.36 | 0.39 |

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
subtitle: "Persian (fa) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Finnish (fi) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **1.00** | **0.92** | **0.83** | **0.92** | **0.86** | **0.92** | **0.83** | **0.92** | **0.86** | **0.81** | **1.00** | **0.94** | **0.81** | **0.75** | **0.94** |
| `bbpb_rfgm_belebele` | **0.97** | **0.92** | **0.83** | **0.94** | **0.86** | **0.94** | **0.86** | **0.92** | **0.89** | **0.81** | **0.86** | **0.94** | **0.83** | **0.75** | **0.86** |
| `bbpb_rf_belebele` | **0.92** | **0.94** | **0.81** | **0.94** | **0.89** | **0.92** | **0.83** | **0.92** | **0.92** | **0.81** | **0.89** | **0.83** | **0.81** | **0.81** | **0.94** |
| `bbpb_rf_include_base_44` | **0.83** | **0.81** | **0.81** | **0.83** | **0.75** | **0.92** | **0.86** | **0.94** | **0.86** | **0.89** | **0.97** | **0.94** | **0.86** | **0.83** | **0.92** |
| `bbpb_rfgm_include_base_44` | **0.92** | **0.89** | **0.89** | **0.89** | **0.75** | **0.92** | **0.86** | **0.86** | 0.72 | **0.83** | **0.94** | 0.69 | **0.78** | **0.75** | 0.69 |
| `bbpb_include_v2_og` | **0.81** | **0.81** | 0.72 | **0.83** | 0.72 | **0.83** | **0.81** | **0.81** | **0.81** | **0.86** | **0.92** | **0.86** | **0.89** | **0.78** | **0.78** |
| `bbpb_multiblimp` | **0.83** | 0.72 | **0.89** | **0.89** | **0.83** | **0.78** | **0.78** | **0.83** | 0.72 | 0.67 | 0.61 | 0.56 | **0.94** | **0.89** | **0.89** |
| `multiblimp` | **0.78** | 0.69 | **0.86** | **0.81** | 0.69 | **0.75** | **0.92** | 0.69 | **0.75** | 0.72 | 0.72 | **0.78** | **0.78** | **0.83** | **0.78** |
| `bbpb_arc_mt` | **0.78** | 0.67 | **0.75** | 0.58 | 0.61 | 0.67 | **0.86** | 0.69 | 0.72 | **0.81** | **0.86** | **0.83** | **0.83** | **0.81** | **0.92** |
| `bbpb_global_piqa_parallel_cloze` | **0.92** | **0.78** | **0.75** | 0.72 | 0.72 | **0.86** | 0.72 | **0.75** | 0.69 | 0.69 | 0.72 | 0.67 | **0.86** | 0.64 | 0.67 |
| `bbpb_belebele` | 0.72 | 0.69 | 0.67 | 0.69 | 0.56 | 0.64 | **0.83** | **0.75** | 0.61 | **0.81** | 0.67 | 0.47 | 0.69 | 0.56 | 0.58 |
| `rfgm_belebele` | 0.69 | 0.69 | 0.53 | 0.58 | 0.50 | **0.81** | 0.50 | 0.47 | 0.64 | 0.56 | 0.56 | 0.61 | 0.67 | 0.72 | 0.53 |
| `arc_mt` | 0.50 | 0.50 | 0.69 | **0.75** | 0.33 | 0.58 | 0.42 | 0.61 | 0.61 | 0.44 | 0.64 | 0.72 | **0.78** | 0.31 | 0.42 |
| `include_v2_en` | 0.36 | 0.64 | 0.69 | 0.33 | 0.61 | 0.67 | 0.25 | 0.53 | 0.47 | 0.58 | 0.58 | 0.64 | 0.56 | 0.69 | 0.67 |
| `bbpb_include_v2_en` | 0.33 | 0.69 | 0.50 | 0.44 | 0.53 | 0.47 | 0.67 | 0.61 | 0.36 | 0.69 | 0.69 | 0.56 | 0.67 | 0.25 | 0.42 |
| `bbpb_include_base_44` | 0.67 | 0.44 | 0.50 | 0.61 | 0.64 | 0.28 | 0.50 | 0.56 | 0.69 | 0.61 | 0.39 | 0.47 | 0.56 | 0.42 | 0.42 |
| `rfgm_include_base_44` | **0.78** | 0.42 | 0.67 | 0.39 | 0.47 | 0.47 | 0.67 | 0.39 | 0.58 | 0.58 | 0.53 | 0.44 | 0.42 | 0.39 | 0.56 |
| `rf_belebele` | 0.47 | 0.72 | 0.64 | 0.36 | 0.44 | 0.58 | 0.36 | 0.42 | 0.17 | 0.69 | 0.42 | 0.39 | 0.33 | 0.58 | 0.61 |
| `belebele` | 0.47 | 0.44 | 0.31 | 0.44 | 0.42 | 0.61 | 0.22 | 0.47 | 0.19 | 0.47 | 0.61 | 0.56 | 0.58 | 0.72 | 0.56 |
| `include_v2_og` | 0.50 | 0.64 | 0.33 | 0.61 | 0.44 | 0.42 | 0.56 | 0.42 | 0.47 | 0.31 | 0.72 | 0.33 | 0.53 | 0.42 | 0.36 |
| `rf_include_base_44` | 0.44 | 0.22 | 0.42 | 0.58 | 0.36 | 0.33 | 0.56 | 0.28 | 0.67 | 0.53 | 0.36 | 0.50 | 0.47 | 0.56 | 0.19 |
| `include_base_44` | 0.53 | 0.25 | 0.17 | 0.50 | 0.36 | 0.44 | 0.31 | 0.53 | 0.25 | 0.56 | 0.53 | 0.31 | 0.53 | 0.67 | 0.33 |
| `global_piqa_parallel_cloze` | 0.19 | 0.58 | 0.72 | 0.19 | 0.17 | 0.39 | 0.31 | 0.42 | 0.61 | 0.64 | 0.33 | 0.39 | 0.22 | 0.31 | 0.64 |

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
subtitle: "Finnish (fi) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "French (fr) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.98** | **0.85** | **0.83** | **0.89** | **0.92** | **0.87** | **0.85** | **0.92** | **0.89** | **0.88** | **0.89** | **0.82** | **0.85** | **0.79** | **0.86** |
| `bbpb_hellaswag` | **0.95** | **0.88** | **0.89** | **0.88** | **0.82** | **0.91** | **0.92** | **0.92** | **0.78** | **0.89** | **0.89** | **0.79** | **0.89** | **0.76** | **0.80** |
| `hellaswag` | **0.81** | 0.71 | **0.77** | **0.81** | **0.81** | **0.83** | **0.89** | **0.81** | **0.77** | **0.90** | 0.72 | 0.74 | **0.77** | **0.78** | **0.83** |
| `bbpb_rfgm_include_base_44` | 0.71 | 0.69 | **0.77** | 0.75 | **0.81** | 0.71 | 0.71 | 0.72 | 0.66 | 0.65 | 0.75 | 0.68 | 0.71 | 0.73 | **0.85** |
| `bbpb_rfgm_belebele` | **0.91** | **0.82** | **0.80** | 0.73 | 0.61 | **0.79** | **0.79** | 0.68 | 0.56 | **0.78** | 0.68 | 0.68 | 0.68 | 0.54 | 0.61 |
| `bbpb_global_mmlu_full` | 0.72 | 0.69 | 0.74 | **0.76** | 0.61 | 0.72 | 0.74 | **0.79** | 0.62 | **0.76** | **0.81** | 0.63 | **0.78** | 0.61 | 0.64 |
| `lambada_openai_mt` | **0.77** | **0.76** | 0.62 | 0.72 | 0.62 | 0.73 | **0.76** | 0.75 | 0.66 | 0.68 | 0.68 | 0.55 | **0.78** | 0.65 | **0.75** |
| `bbpb_global_piqa_parallel_cloze_fra_latn_fran` | 0.75 | 0.74 | 0.72 | 0.54 | 0.66 | **0.89** | 0.72 | 0.61 | 0.65 | 0.73 | 0.63 | 0.63 | 0.60 | 0.66 | 0.61 |
| `bbpb_paws` | 0.70 | 0.70 | 0.69 | **0.76** | 0.59 | 0.73 | 0.68 | **0.75** | 0.55 | 0.65 | 0.65 | 0.71 | 0.65 | 0.74 | 0.58 |
| `bbpb_include_base_44` | **0.80** | 0.75 | **0.76** | 0.68 | 0.64 | 0.65 | 0.72 | 0.67 | 0.62 | 0.67 | 0.62 | 0.65 | 0.64 | 0.64 | 0.62 |
| `bbpb_rf_global_mmlu_full` | 0.66 | 0.57 | 0.63 | **0.79** | 0.69 | 0.75 | 0.74 | 0.66 | 0.69 | **0.76** | 0.54 | 0.66 | 0.55 | 0.74 | 0.64 |
| `bbpb_rf_belebele` | **0.82** | 0.72 | **0.76** | 0.72 | 0.62 | 0.74 | 0.74 | 0.72 | 0.58 | 0.65 | 0.65 | 0.63 | 0.65 | 0.55 | 0.52 |
| `bbpb_multiblimp` | 0.68 | 0.66 | 0.69 | 0.67 | 0.72 | 0.71 | 0.65 | 0.71 | 0.58 | 0.63 | 0.62 | 0.71 | 0.69 | 0.72 | 0.60 |
| `bbpb_global_piqa_parallel_cloze_fra_latn_cana` | **0.82** | 0.73 | 0.74 | 0.54 | 0.64 | **0.89** | **0.77** | 0.63 | 0.58 | 0.74 | 0.64 | 0.56 | 0.60 | 0.60 | 0.52 |
| `multiblimp` | 0.75 | 0.73 | 0.71 | 0.58 | 0.69 | **0.78** | 0.69 | 0.59 | 0.67 | 0.71 | 0.55 | 0.66 | 0.55 | 0.72 | 0.56 |
| `bbpb_rf_include_base_44` | 0.65 | 0.64 | 0.69 | 0.59 | 0.61 | 0.68 | 0.68 | 0.69 | 0.71 | 0.57 | 0.50 | 0.63 | 0.66 | 0.73 | 0.60 |
| `bbpb_include_v2_og_french_france` | 0.64 | 0.68 | 0.65 | 0.63 | 0.65 | 0.67 | 0.67 | 0.64 | 0.53 | 0.59 | 0.57 | 0.69 | 0.68 | 0.74 | 0.59 |
| `bbpb_arc` | 0.62 | 0.58 | 0.65 | 0.66 | 0.51 | 0.64 | **0.82** | 0.64 | 0.64 | 0.61 | 0.59 | 0.73 | 0.57 | 0.68 | 0.53 |
| `bbpb_xnli` | 0.66 | 0.67 | 0.66 | 0.59 | 0.62 | 0.64 | 0.65 | 0.51 | 0.55 | 0.75 | 0.56 | 0.66 | 0.59 | 0.66 | 0.57 |
| `bbpb_rf_m_mmlu` | 0.62 | 0.71 | 0.59 | 0.64 | 0.56 | 0.71 | 0.68 | 0.64 | 0.47 | 0.62 | 0.60 | 0.58 | 0.62 | 0.65 | 0.54 |
| `bbpb_rf_mmmlu` | 0.62 | 0.68 | 0.65 | 0.54 | 0.56 | 0.64 | 0.66 | 0.55 | 0.56 | 0.59 | 0.59 | 0.64 | 0.60 | 0.67 | 0.65 |
| `bbpb_belebele` | 0.63 | 0.59 | 0.67 | 0.69 | 0.56 | 0.68 | 0.71 | 0.55 | 0.48 | 0.72 | 0.55 | 0.57 | 0.57 | 0.54 | 0.59 |
| `rf_belebele` | 0.59 | 0.57 | 0.63 | 0.56 | 0.55 | 0.49 | 0.63 | 0.63 | 0.56 | 0.57 | 0.59 | 0.66 | **0.75** | 0.62 | 0.67 |
| `rf_m_mmlu` | 0.52 | 0.52 | 0.56 | 0.57 | 0.49 | 0.53 | 0.70 | 0.59 | 0.64 | 0.67 | 0.58 | 0.69 | 0.62 | 0.73 | 0.63 |
| `rf_mmmlu` | 0.53 | 0.39 | 0.44 | 0.41 | 0.40 | 0.66 | 0.65 | 0.58 | 0.65 | 0.67 | 0.61 | 0.69 | 0.69 | 0.75 | **0.81** |
| `rf_include_base_44` | 0.63 | 0.62 | 0.54 | 0.65 | 0.63 | 0.61 | 0.50 | 0.61 | 0.56 | 0.42 | 0.54 | 0.55 | 0.53 | 0.51 | 0.65 |
| `rf_global_mmlu_full` | 0.51 | 0.47 | 0.47 | 0.43 | 0.43 | 0.53 | 0.49 | 0.58 | 0.64 | 0.66 | 0.71 | 0.66 | 0.57 | 0.57 | 0.67 |
| `rfgm_belebele` | 0.43 | 0.50 | 0.43 | 0.52 | 0.41 | 0.66 | 0.68 | **0.78** | 0.55 | 0.58 | 0.54 | 0.45 | 0.69 | 0.58 | 0.58 |
| `bbpb_include_v2_en_french_france` | 0.34 | 0.68 | 0.63 | 0.55 | 0.66 | 0.52 | 0.46 | 0.53 | 0.36 | 0.58 | 0.64 | 0.58 | 0.61 | 0.56 | 0.59 |
| `arc` | 0.35 | 0.61 | 0.56 | 0.65 | 0.47 | 0.48 | 0.48 | 0.47 | 0.50 | 0.59 | 0.69 | 0.55 | 0.61 | 0.58 | 0.49 |
| `global_mmlu_full` | 0.54 | 0.56 | 0.39 | 0.46 | 0.61 | 0.60 | 0.48 | 0.53 | 0.61 | 0.59 | 0.52 | 0.52 | 0.53 | 0.51 | 0.57 |
| `paws` | 0.53 | 0.53 | 0.48 | 0.39 | 0.47 | 0.62 | 0.61 | 0.55 | 0.55 | 0.58 | 0.55 | 0.54 | 0.72 | 0.32 | 0.37 |
| `rfgm_include_base_44` | 0.54 | 0.47 | 0.56 | 0.43 | 0.54 | 0.62 | 0.46 | 0.54 | 0.51 | 0.43 | 0.58 | 0.44 | 0.58 | 0.58 | 0.54 |
| `bbpb_include_v2_og_french_canada` | 0.51 | 0.55 | 0.47 | 0.51 | 0.52 | 0.40 | 0.60 | 0.47 | 0.46 | 0.51 | 0.44 | 0.62 | 0.47 | 0.48 | **0.77** |
| `xnli` | 0.46 | 0.47 | 0.57 | 0.42 | 0.43 | 0.51 | 0.54 | 0.62 | 0.64 | 0.59 | 0.49 | 0.45 | 0.45 | 0.53 | 0.51 |
| `include_v2_og_french_france` | 0.44 | 0.42 | 0.42 | 0.39 | 0.54 | 0.41 | 0.38 | 0.56 | 0.57 | 0.62 | 0.51 | 0.58 | 0.56 | 0.59 | 0.62 |
| `bbpb_include_v2_en_french_canada` | 0.49 | 0.42 | 0.42 | 0.49 | 0.63 | 0.54 | 0.50 | 0.51 | 0.33 | 0.64 | 0.45 | 0.29 | 0.59 | 0.42 | 0.61 |
| `include_v2_og_french_canada` | 0.42 | 0.45 | 0.42 | 0.47 | 0.54 | 0.35 | 0.47 | 0.46 | 0.38 | 0.66 | 0.58 | 0.43 | 0.53 | 0.45 | 0.51 |
| `include_base_44` | 0.58 | 0.46 | 0.40 | 0.42 | 0.51 | 0.31 | 0.52 | 0.57 | 0.55 | 0.52 | 0.27 | 0.47 | 0.43 | 0.56 | 0.51 |
| `belebele` | 0.38 | 0.58 | 0.48 | 0.40 | 0.46 | 0.41 | 0.44 | 0.45 | 0.52 | 0.42 | 0.41 | 0.52 | 0.67 | 0.41 | 0.51 |
| `include_v2_en_french_france` | 0.41 | 0.44 | 0.53 | 0.43 | 0.50 | 0.46 | 0.52 | 0.68 | 0.46 | 0.30 | 0.30 | 0.52 | 0.53 | 0.43 | 0.39 |
| `global_piqa_parallel_cloze_fra_latn_cana` | 0.47 | 0.42 | 0.47 | 0.35 | 0.56 | 0.49 | 0.54 | 0.42 | 0.42 | 0.62 | 0.39 | 0.39 | 0.37 | 0.43 | 0.50 |
| `xwinograd` | 0.51 | 0.49 | 0.51 | 0.58 | 0.41 | 0.41 | 0.31 | 0.44 | 0.35 | 0.51 | 0.43 | 0.33 | 0.58 | 0.57 | 0.40 |
| `global_piqa_parallel_cloze_fra_latn_fran` | 0.52 | 0.35 | 0.35 | 0.34 | 0.43 | 0.44 | 0.45 | 0.37 | 0.44 | 0.64 | 0.38 | 0.47 | 0.55 | 0.49 | 0.48 |
| `include_v2_en_french_canada` | 0.39 | 0.53 | 0.44 | 0.47 | 0.49 | 0.38 | 0.56 | 0.46 | 0.43 | 0.41 | 0.45 | 0.35 | 0.33 | 0.55 | 0.42 |

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
subtitle: "Hebrew (he) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.96** | **0.93** | **0.86** | **0.89** | **1.00** | **0.96** | **0.89** | **0.93** | **0.96** | **0.86** | **0.89** | **0.93** | **0.96** | **0.86** | **0.89** |
| `bbpb_rfgm_include_base_44` | **0.96** | **0.93** | **0.93** | **0.96** | **0.86** | **0.96** | **0.96** | **0.93** | **0.82** | **0.93** | **0.89** | **0.86** | **0.89** | **0.79** | **0.89** |
| `bbpb_rf_include_base_44` | **0.96** | **0.89** | **0.89** | **0.93** | **0.89** | **0.86** | **0.93** | **0.89** | **0.86** | **0.79** | **0.89** | **0.93** | **0.89** | **0.79** | **0.82** |
| `bbpb_global_piqa_parallel_cloze` | **0.79** | **0.93** | **0.86** | **0.89** | **0.96** | **0.79** | **0.86** | **0.82** | **0.75** | **0.79** | **0.96** | **0.89** | **0.82** | **0.89** | **0.86** |
| `bbpb_rf_belebele` | **0.86** | **0.82** | **0.93** | **0.89** | 0.71 | **0.75** | **0.93** | **0.89** | **0.79** | **0.75** | **0.86** | **0.75** | **0.89** | **0.79** | **0.82** |
| `bbpb_rfgm_belebele` | **0.82** | **0.89** | **0.82** | **0.93** | 0.71 | **0.79** | 0.71 | **0.82** | **0.75** | **0.86** | **0.89** | 0.68 | **0.89** | **0.75** | 0.64 |
| `bbpb_include_v2_og` | **0.82** | **0.86** | **0.75** | 0.68 | **0.82** | 0.68 | 0.71 | 0.64 | **0.79** | **0.82** | **0.82** | **0.89** | **0.93** | **0.79** | **0.79** |
| `bbpb_rf_global_mmlu_full` | **0.79** | 0.71 | **0.75** | **0.93** | **0.75** | 0.71 | **0.82** | **0.79** | **0.75** | 0.68 | 0.71 | 0.61 | 0.68 | 0.57 | **0.82** |
| `bbpb_multiblimp` | 0.64 | 0.68 | **0.82** | 0.71 | **0.75** | **0.75** | **0.82** | **0.79** | 0.46 | 0.71 | **0.82** | 0.71 | **0.82** | 0.64 | 0.68 |
| `multiblimp` | 0.68 | 0.71 | **0.79** | **0.86** | **0.75** | 0.57 | **0.75** | 0.68 | 0.61 | 0.57 | 0.71 | **0.75** | **0.86** | 0.68 | **0.82** |
| `bbpb_include_base_44` | **0.96** | 0.36 | 0.68 | 0.68 | 0.50 | 0.39 | 0.71 | 0.71 | 0.54 | 0.68 | 0.32 | **0.79** | 0.57 | **0.82** | 0.54 |
| `bbpb_global_mmlu_full` | **0.82** | 0.54 | 0.71 | 0.68 | 0.61 | 0.50 | 0.54 | 0.57 | 0.50 | 0.61 | 0.36 | 0.71 | **0.75** | 0.61 | 0.57 |
| `rf_global_mmlu_full` | 0.64 | 0.57 | 0.43 | 0.54 | 0.54 | 0.64 | 0.61 | 0.39 | 0.39 | 0.71 | **0.75** | 0.61 | 0.54 | 0.39 | **0.86** |
| `global_mmlu_full` | 0.43 | 0.46 | 0.57 | 0.64 | 0.68 | 0.68 | 0.29 | 0.71 | 0.68 | 0.32 | 0.54 | 0.57 | 0.57 | 0.46 | **0.82** |
| `bbpb_belebele` | 0.46 | 0.54 | **0.79** | 0.50 | 0.68 | 0.64 | 0.39 | 0.61 | 0.43 | 0.61 | 0.46 | 0.64 | 0.57 | 0.68 | 0.32 |
| `bbpb_include_v2_en` | 0.43 | 0.46 | 0.68 | 0.54 | **0.86** | 0.46 | 0.54 | 0.46 | 0.50 | 0.57 | 0.43 | 0.61 | 0.43 | 0.61 | 0.54 |
| `rf_include_base_44` | 0.64 | 0.36 | 0.54 | 0.57 | **0.82** | 0.36 | 0.57 | 0.71 | 0.57 | 0.39 | 0.57 | 0.39 | 0.46 | 0.43 | 0.46 |
| `rfgm_include_base_44` | 0.54 | 0.39 | 0.39 | 0.68 | 0.46 | 0.61 | 0.57 | 0.43 | 0.64 | 0.46 | 0.46 | 0.50 | 0.36 | **0.86** | 0.43 |
| `belebele` | 0.57 | 0.46 | 0.54 | 0.32 | 0.46 | **0.75** | 0.54 | 0.61 | 0.32 | 0.43 | 0.50 | 0.50 | 0.50 | 0.50 | 0.61 |
| `include_v2_en` | 0.46 | 0.54 | 0.21 | 0.46 | **0.75** | 0.11 | 0.29 | 0.64 | 0.36 | 0.64 | 0.46 | 0.71 | 0.61 | 0.46 | 0.50 |
| `global_piqa_parallel_cloze` | 0.50 | 0.36 | 0.61 | 0.57 | 0.64 | 0.50 | 0.50 | 0.32 | 0.18 | 0.57 | 0.29 | 0.36 | 0.64 | 0.46 | 0.54 |
| `include_v2_og` | 0.29 | 0.68 | 0.43 | 0.61 | 0.50 | 0.07 | 0.46 | 0.39 | 0.71 | 0.39 | 0.57 | 0.32 | 0.57 | 0.43 | 0.54 |
| `include_base_44` | 0.36 | 0.32 | 0.68 | 0.32 | 0.57 | 0.61 | 0.43 | 0.43 | 0.43 | 0.21 | 0.71 | 0.54 | 0.18 | 0.61 | 0.50 |
| `rf_belebele` | 0.39 | 0.36 | **0.79** | 0.61 | 0.32 | 0.25 | 0.46 | 0.54 | 0.64 | 0.29 | 0.18 | 0.50 | 0.64 | 0.36 | 0.57 |
| `rfgm_belebele` | 0.43 | 0.14 | 0.50 | 0.43 | 0.57 | 0.50 | 0.61 | 0.54 | 0.61 | 0.39 | 0.43 | 0.43 | 0.36 | 0.25 | 0.71 |

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
subtitle: "Hebrew (he) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Hindi (hi) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_hellaswag` | **0.91** | **0.92** | **0.92** | **0.91** | **0.87** | **0.91** | **0.88** | **0.87** | **0.94** | **0.92** | **0.86** | **0.90** | **0.88** | **0.85** | **0.83** |
| `bbpb_xstorycloze` | **0.88** | **0.82** | **0.88** | **0.88** | **0.83** | **0.88** | **0.87** | **0.85** | **0.92** | **0.86** | **0.81** | **0.91** | **0.85** | **0.85** | **0.85** |
| `bpb` | **0.94** | **0.91** | **0.79** | **0.79** | **0.82** | **0.95** | **0.83** | **0.83** | **0.86** | **0.88** | **0.83** | **0.81** | **0.85** | 0.72 | **0.85** |
| `bbpb_arc` | **0.79** | **0.86** | 0.72 | **0.78** | 0.72 | **0.78** | **0.82** | **0.81** | **0.82** | **0.78** | **0.77** | **0.78** | **0.86** | **0.77** | 0.73 |
| `bbpb_rfgm_include_base_44` | **0.83** | 0.68 | **0.85** | **0.79** | **0.82** | **0.79** | **0.94** | **0.83** | 0.73 | **0.78** | 0.68 | 0.73 | **0.82** | 0.72 | 0.74 |
| `bbpb_rf_belebele_hin_Deva` | **0.76** | **0.79** | **0.81** | **0.78** | **0.77** | **0.81** | 0.67 | **0.79** | **0.76** | 0.73 | 0.73 | 0.74 | **0.82** | **0.81** | **0.88** |
| `bbpb_multiblimp` | 0.74 | **0.76** | **0.88** | 0.73 | 0.72 | **0.86** | **0.81** | **0.86** | 0.67 | **0.79** | **0.85** | 0.65 | **0.77** | 0.63 | 0.63 |
| `bbpb_rfgm_belebele_hin_Deva` | **0.85** | 0.74 | 0.72 | **0.77** | **0.79** | **0.82** | 0.69 | 0.74 | 0.74 | **0.79** | **0.79** | 0.67 | 0.74 | 0.69 | 0.72 |
| `bbpb_rf_global_mmlu_full` | 0.68 | 0.69 | 0.67 | **0.77** | 0.67 | 0.73 | **0.81** | **0.81** | **0.81** | 0.69 | **0.77** | 0.72 | **0.77** | 0.74 | **0.85** |
| `bbpb_global_piqa_parallel_cloze` | 0.74 | **0.81** | **0.76** | **0.77** | 0.69 | **0.76** | **0.83** | **0.90** | 0.62 | 0.72 | **0.78** | 0.65 | **0.83** | 0.58 | 0.59 |
| `bbpb_rf_include_base_44` | 0.73 | **0.81** | 0.71 | 0.73 | 0.73 | 0.62 | 0.74 | **0.79** | **0.77** | 0.59 | 0.72 | 0.72 | 0.69 | **0.82** | **0.79** |
| `bbpb_rf_mmmlu` | 0.71 | **0.78** | 0.62 | 0.72 | **0.81** | 0.74 | 0.73 | 0.73 | **0.79** | 0.63 | 0.68 | 0.72 | 0.64 | 0.68 | **0.78** |
| `multiblimp` | **0.79** | **0.77** | 0.72 | 0.74 | 0.71 | 0.65 | 0.73 | **0.79** | 0.71 | 0.67 | **0.77** | 0.64 | 0.71 | 0.59 | 0.64 |
| `bbpb_xnli` | **0.77** | **0.76** | **0.85** | 0.68 | 0.63 | **0.88** | 0.74 | **0.78** | 0.58 | 0.71 | 0.74 | 0.59 | 0.65 | 0.53 | 0.69 |
| `bbpb_rf_m_mmlu` | 0.71 | 0.74 | 0.62 | 0.72 | 0.72 | **0.76** | 0.71 | **0.76** | 0.73 | 0.56 | 0.74 | 0.64 | 0.62 | 0.59 | **0.79** |
| `bbpb_rf_belebele_hin_Latn` | 0.55 | 0.53 | 0.69 | 0.73 | 0.62 | 0.69 | 0.47 | 0.69 | 0.53 | 0.71 | 0.72 | 0.60 | 0.58 | 0.67 | 0.63 |
| `hellaswag` | **0.82** | 0.38 | 0.63 | 0.58 | 0.49 | 0.33 | 0.62 | 0.59 | 0.46 | 0.58 | 0.62 | 0.67 | **0.88** | 0.72 | **0.77** |
| `bbpb_cultural_bench_easy` | 0.56 | 0.64 | 0.60 | 0.58 | 0.54 | 0.64 | 0.65 | 0.65 | 0.67 | 0.55 | 0.50 | 0.64 | 0.56 | 0.45 | 0.68 |
| `bbpb_rfgm_belebele_hin_Latn` | 0.47 | 0.59 | 0.53 | 0.51 | 0.46 | 0.53 | 0.64 | 0.60 | 0.63 | 0.68 | 0.56 | 0.49 | 0.58 | 0.71 | 0.59 |
| `bbpb_include_v2_og` | 0.69 | 0.54 | 0.56 | 0.68 | 0.50 | 0.59 | 0.59 | 0.68 | 0.58 | 0.44 | 0.55 | 0.60 | 0.47 | 0.40 | 0.62 |
| `bbpb_global_mmlu_full` | **0.77** | 0.53 | 0.55 | 0.49 | 0.64 | 0.55 | 0.55 | 0.59 | 0.56 | 0.51 | 0.37 | 0.40 | 0.58 | 0.65 | 0.59 |
| `rf_global_mmlu_full` | 0.60 | 0.37 | 0.55 | 0.53 | 0.56 | 0.71 | 0.56 | 0.67 | 0.55 | 0.42 | 0.56 | 0.65 | 0.42 | 0.42 | 0.68 |
| `xstorycloze` | 0.40 | 0.64 | 0.54 | 0.46 | 0.64 | 0.58 | 0.54 | 0.45 | 0.51 | 0.45 | 0.59 | 0.71 | 0.54 | 0.53 | 0.54 |
| `include_v2_og` | 0.53 | 0.56 | 0.62 | 0.55 | 0.49 | 0.45 | 0.37 | 0.41 | 0.41 | 0.55 | 0.62 | 0.71 | 0.65 | 0.63 | 0.51 |
| `bbpb_belebele_hin_Latn` | 0.56 | 0.54 | 0.51 | 0.55 | 0.33 | 0.56 | 0.44 | 0.58 | 0.67 | 0.64 | 0.68 | 0.44 | 0.65 | 0.38 | 0.50 |
| `bbpb_belebele_hin_Deva` | 0.58 | 0.54 | 0.67 | 0.54 | 0.63 | 0.58 | 0.50 | 0.55 | 0.56 | 0.54 | 0.44 | 0.47 | 0.36 | 0.53 | 0.50 |
| `bbpb_include_base_44` | 0.63 | 0.54 | 0.58 | 0.45 | 0.41 | 0.50 | 0.72 | 0.62 | 0.45 | 0.45 | 0.53 | 0.41 | 0.54 | 0.60 | 0.45 |
| `global_mmlu_full` | 0.56 | 0.53 | 0.58 | 0.37 | 0.69 | 0.58 | 0.58 | 0.53 | 0.62 | 0.46 | 0.59 | 0.42 | 0.38 | 0.45 | 0.50 |
| `xnli` | 0.38 | 0.54 | 0.38 | 0.60 | 0.51 | 0.49 | 0.51 | 0.47 | 0.38 | 0.69 | 0.64 | 0.56 | 0.54 | 0.44 | 0.60 |
| `rf_belebele_hin_Deva` | 0.46 | 0.44 | 0.47 | 0.69 | 0.47 | 0.63 | 0.47 | 0.42 | 0.45 | 0.45 | 0.45 | 0.37 | 0.49 | **0.77** | 0.58 |
| `bbpb_include_v2_en` | 0.21 | 0.64 | 0.45 | 0.45 | 0.64 | 0.44 | 0.60 | 0.50 | 0.36 | 0.60 | 0.63 | 0.49 | 0.64 | 0.45 | 0.50 |
| `rf_mmmlu` | 0.56 | 0.47 | 0.46 | 0.46 | 0.51 | 0.54 | 0.49 | 0.68 | 0.41 | 0.63 | 0.47 | 0.36 | 0.59 | 0.31 | 0.56 |
| `bbpb_rf_cultural_bench_easy` | 0.46 | 0.49 | 0.58 | 0.38 | 0.54 | 0.67 | 0.45 | 0.33 | 0.38 | 0.40 | 0.64 | 0.44 | 0.37 | 0.50 | 0.62 |
| `cultural_bench_easy` | 0.55 | 0.50 | 0.38 | 0.35 | 0.53 | 0.56 | 0.54 | 0.38 | 0.54 | 0.44 | 0.50 | 0.53 | 0.42 | 0.56 | 0.44 |
| `belebele_hin_Deva` | 0.46 | 0.31 | 0.51 | 0.49 | 0.58 | 0.55 | 0.68 | 0.33 | 0.51 | 0.59 | 0.46 | 0.36 | 0.37 | 0.50 | 0.47 |
| `include_base_44` | 0.45 | 0.45 | 0.56 | 0.47 | 0.47 | 0.49 | 0.42 | 0.51 | 0.46 | 0.56 | 0.41 | 0.65 | 0.27 | 0.44 | 0.46 |
| `include_v2_en` | 0.41 | 0.44 | 0.56 | 0.46 | 0.69 | 0.49 | 0.40 | 0.35 | 0.46 | 0.65 | 0.38 | 0.37 | 0.62 | 0.45 | 0.33 |
| `global_piqa_parallel_cloze` | 0.42 | 0.41 | 0.46 | 0.44 | 0.55 | 0.53 | 0.28 | 0.50 | 0.56 | 0.44 | 0.46 | 0.37 | 0.45 | 0.44 | 0.72 |
| `rf_m_mmlu` | 0.59 | 0.46 | 0.47 | 0.41 | 0.59 | 0.41 | 0.58 | 0.41 | 0.44 | 0.45 | 0.49 | 0.38 | 0.38 | 0.38 | 0.53 |
| `rfgm_include_base_44` | 0.38 | 0.53 | 0.49 | 0.47 | 0.50 | 0.59 | 0.38 | 0.36 | 0.41 | 0.41 | 0.51 | 0.51 | 0.63 | 0.33 | 0.35 |
| `rfgm_belebele_hin_Latn` | 0.41 | 0.63 | 0.55 | 0.54 | 0.37 | 0.41 | 0.41 | 0.55 | 0.49 | 0.56 | 0.56 | 0.31 | 0.36 | 0.21 | 0.50 |
| `cultural_bench_hard` | 0.40 | 0.32 | 0.46 | 0.38 | 0.49 | 0.60 | 0.33 | 0.50 | 0.35 | 0.35 | **0.81** | 0.44 | 0.50 | 0.54 | 0.38 |
| `rf_include_base_44` | 0.45 | 0.44 | 0.64 | 0.49 | 0.37 | 0.21 | 0.36 | 0.35 | 0.73 | 0.40 | 0.51 | 0.42 | 0.50 | 0.35 | 0.47 |
| `rfgm_belebele_hin_Deva` | 0.47 | 0.35 | 0.59 | 0.45 | 0.38 | 0.26 | 0.38 | 0.46 | 0.38 | 0.44 | 0.50 | 0.45 | 0.38 | 0.49 | 0.54 |
| `arc` | 0.41 | 0.44 | 0.31 | 0.41 | 0.55 | 0.46 | 0.45 | 0.35 | 0.51 | 0.45 | 0.42 | 0.46 | 0.46 | 0.38 | 0.40 |
| `belebele_hin_Latn` | 0.37 | 0.59 | 0.44 | 0.64 | 0.38 | 0.35 | 0.42 | 0.33 | 0.67 | 0.55 | 0.41 | 0.18 | 0.24 | 0.37 | 0.47 |
| `rf_belebele_hin_Latn` | 0.44 | 0.45 | 0.35 | 0.41 | 0.44 | 0.41 | 0.38 | 0.41 | 0.24 | 0.56 | 0.47 | 0.29 | 0.55 | 0.42 | 0.46 |
| `rf_cultural_bench_easy` | 0.46 | 0.40 | 0.41 | 0.44 | 0.62 | 0.45 | 0.22 | 0.53 | 0.47 | 0.35 | 0.35 | 0.33 | 0.35 | 0.29 | 0.37 |

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
subtitle: "Croatian (hr) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_multiblimp-extra` | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** |
| `bbpb_rfgm_belebele` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_rf_belebele` | **1.00** | **1.00** | **0.83** | **1.00** | 0.67 | **1.00** | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** | 0.67 |
| `bbpb_rfgm_include_base_44` | **1.00** | **1.00** | 0.67 | **1.00** | **0.83** | **1.00** | 0.67 | **1.00** | **0.83** | 0.67 | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** |
| `bbpb_hellaswag` | **1.00** | **0.83** | 0.67 | **1.00** | **1.00** | **0.83** | 0.67 | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** |
| `bbpb_include_v2_og` | **1.00** | **1.00** | **0.83** | **1.00** | 0.67 | **1.00** | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | 0.67 | **0.83** | 0.50 | 0.67 |
| `multiblimp-extra` | **0.83** | **1.00** | 0.67 | **1.00** | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 | **1.00** | 0.67 | 0.67 | **1.00** | 0.67 |
| `hellaswag` | **1.00** | 0.67 | **0.83** | 0.67 | 0.67 | 0.67 | **0.83** | 0.67 | 0.67 | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** |
| `bbpb_rf_m_mmlu` | **1.00** | **1.00** | **0.83** | **0.83** | 0.67 | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | **1.00** | 0.50 | 0.50 |
| `bbpb_arc` | **1.00** | **0.83** | **0.83** | **0.83** | 0.50 | **0.83** | **0.83** | **0.83** | 0.50 | **1.00** | **1.00** | 0.67 | **1.00** | 0.67 | 0.67 |
| `bbpb_global_piqa_parallel_cloze` | 0.67 | 0.67 | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | **0.83** |
| `rfgm_belebele` | **0.83** | 0.67 | 0.67 | 0.67 | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | **0.83** | 0.67 | 0.67 |
| `bbpb_include_base_44` | **0.83** | **0.83** | 0.67 | 0.50 | 0.67 | 0.67 | 0.50 | 0.33 | 0.50 | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | 0.50 |
| `bbpb_belebele` | 0.67 | 0.67 | **1.00** | 0.67 | 0.50 | **1.00** | 0.67 | 0.33 | 0.50 | 0.67 | 0.33 | 0.50 | 0.67 | 0.50 | **0.83** |
| `bbpb_rf_include_base_44` | 0.67 | **0.83** | **0.83** | 0.67 | 0.33 | 0.50 | 0.50 | 0.33 | 0.33 | **1.00** | **0.83** | 0.50 | **0.83** | 0.50 | 0.67 |
| `include_base_44` | 0.67 | 0.33 | 0.50 | 0.67 | 0.17 | 0.67 | **0.83** | **1.00** | 0.50 | 0.50 | 0.67 | 0.67 | **0.83** | 0.67 | 0.50 |
| `bbpb_include_v2_en` | 0.33 | 0.67 | 0.50 | **0.83** | **0.83** | 0.67 | 0.50 | 0.50 | 0.17 | 0.50 | **0.83** | 0.50 | 0.33 | 0.67 | 0.67 |
| `include_v2_og` | 0.17 | 0.50 | 0.17 | 0.67 | 0.33 | 0.67 | **1.00** | 0.50 | **0.83** | 0.67 | 0.50 | **0.83** | 0.50 | **0.83** | 0.33 |
| `include_v2_en` | 0.67 | 0.67 | 0.33 | **0.83** | 0.50 | 0.67 | 0.67 | **0.83** | 0.50 | 0.67 | 0.50 | 0.17 | 0.50 | 0.17 | 0.67 |
| `rf_belebele` | **0.83** | 0.33 | 0.50 | **1.00** | 0.67 | 0.50 | 0.33 | **0.83** | 0.50 | 0.17 | 0.33 | 0.33 | 0.50 | **0.83** | 0.67 |
| `rf_m_mmlu` | 0.17 | 0.33 | 0.33 | 0.17 | 0.50 | 0.50 | **0.83** | 0.67 | 0.67 | 0.33 | **0.83** | 0.50 | 0.50 | 0.50 | 0.67 |
| `rf_include_base_44` | 0.50 | 0.67 | 0.50 | 0.33 | 0.50 | **0.83** | **1.00** | 0.00 | 0.67 | **0.83** | 0.17 | 0.50 | 0.00 | 0.67 | 0.17 |
| `belebele` | 0.67 | 0.50 | 0.33 | 0.50 | **0.83** | **0.83** | 0.33 | 0.17 | 0.50 | 0.50 | 0.33 | 0.33 | 0.50 | 0.17 | 0.67 |
| `arc` | 0.50 | 0.67 | 0.17 | 0.50 | **0.83** | 0.33 | 0.67 | 0.33 | 0.67 | 0.33 | 0.17 | 0.50 | 0.33 | 0.33 | 0.33 |
| `rfgm_include_base_44` | 0.67 | 0.33 | 0.67 | 0.00 | 0.00 | 0.33 | 0.50 | 0.33 | 0.17 | 0.17 | 0.67 | 0.50 | 0.17 | 0.33 | **0.83** |
| `global_piqa_parallel_cloze` | 0.50 | 0.00 | 0.67 | 0.17 | 0.50 | 0.50 | 0.50 | 0.33 | 0.17 | 0.33 | **0.83** | 0.33 | 0.17 | 0.33 | 0.33 |

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
subtitle: "Croatian (hr) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Hungarian (hu) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_hellaswag` | **0.96** | **0.89** | **0.84** | **0.85** | **0.80** | **0.89** | **0.84** | **0.89** | **0.84** | **0.84** | **0.82** | **0.76** | **0.84** | 0.71 | **0.80** |
| `bpb` | **0.95** | **0.84** | 0.69 | **0.87** | **0.93** | **0.89** | 0.71 | **0.89** | **0.91** | 0.75 | **0.93** | **0.84** | 0.71 | 0.65 | **0.84** |
| `bbpb_rfgm_belebele` | **0.89** | **0.78** | 0.73 | **0.93** | **0.82** | **0.78** | **0.80** | **0.82** | **0.89** | 0.69 | **0.85** | **0.82** | 0.69 | 0.69 | **0.78** |
| `bbpb_rfgm_include_base_44` | **0.91** | **0.76** | **0.84** | **0.80** | 0.73 | **0.85** | **0.78** | **0.82** | 0.71 | **0.78** | 0.67 | 0.64 | 0.75 | 0.75 | **0.82** |
| `bbpb_global_piqa_parallel_cloze` | **0.87** | **0.80** | 0.67 | 0.65 | 0.75 | **0.78** | **0.80** | 0.67 | **0.84** | **0.76** | 0.71 | 0.73 | 0.69 | 0.67 | 0.65 |
| `bbpb_include_v2_og` | 0.67 | 0.69 | 0.58 | **0.80** | 0.56 | **0.76** | **0.84** | **0.87** | 0.75 | 0.67 | **0.82** | 0.65 | 0.75 | 0.73 | 0.69 |
| `bbpb_rf_belebele` | **0.76** | **0.80** | 0.69 | **0.84** | 0.69 | **0.89** | 0.64 | **0.82** | 0.67 | 0.64 | **0.82** | 0.67 | 0.64 | 0.53 | 0.71 |
| `bbpb_multiblimp` | 0.75 | **0.80** | **0.85** | 0.67 | **0.76** | **0.76** | **0.78** | 0.75 | 0.51 | 0.73 | 0.51 | 0.67 | **0.78** | 0.73 | 0.62 |
| `bbpb_rf_include_base_44` | 0.67 | 0.55 | 0.60 | 0.69 | 0.58 | 0.73 | 0.75 | 0.73 | 0.69 | 0.73 | **0.78** | **0.78** | 0.69 | 0.62 | 0.64 |
| `hellaswag` | 0.73 | **0.82** | 0.51 | 0.64 | 0.60 | **0.80** | 0.56 | 0.65 | 0.65 | 0.55 | 0.67 | 0.60 | **0.78** | 0.71 | **0.82** |
| `bbpb_rf_m_mmlu` | 0.69 | 0.73 | 0.45 | 0.75 | 0.47 | 0.75 | 0.69 | **0.84** | 0.67 | 0.58 | **0.87** | 0.60 | 0.60 | 0.73 | 0.58 |
| `bbpb_arc` | 0.58 | 0.60 | 0.47 | 0.65 | 0.55 | 0.73 | **0.82** | **0.82** | 0.53 | 0.58 | 0.73 | 0.62 | 0.67 | 0.45 | 0.64 |
| `bbpb_include_base_44` | 0.65 | 0.56 | 0.67 | 0.65 | 0.75 | 0.47 | 0.51 | **0.82** | 0.73 | 0.56 | 0.47 | 0.42 | 0.62 | 0.71 | **0.76** |
| `multiblimp` | **0.82** | 0.53 | 0.65 | 0.73 | 0.73 | 0.64 | 0.67 | 0.67 | 0.65 | 0.51 | 0.62 | 0.33 | 0.58 | 0.55 | 0.62 |
| `bbpb_arc_mt` | 0.58 | 0.64 | 0.45 | 0.67 | 0.56 | **0.76** | 0.73 | **0.76** | 0.44 | 0.60 | 0.71 | 0.56 | 0.64 | 0.49 | 0.53 |
| `bbpb_belebele` | 0.64 | 0.47 | 0.56 | **0.84** | 0.49 | 0.62 | 0.64 | 0.65 | 0.67 | 0.51 | 0.42 | 0.44 | 0.58 | 0.64 | 0.51 |
| `include_v2_og` | 0.64 | 0.51 | 0.45 | 0.67 | 0.56 | 0.45 | 0.62 | 0.60 | 0.69 | 0.47 | 0.44 | 0.40 | 0.42 | 0.64 | 0.51 |
| `include_v2_en` | 0.47 | 0.40 | 0.53 | 0.40 | 0.67 | 0.71 | 0.60 | 0.56 | 0.56 | 0.55 | 0.64 | 0.49 | 0.33 | 0.55 | 0.47 |
| `arc_mt` | 0.65 | 0.36 | 0.67 | 0.58 | 0.55 | 0.45 | 0.49 | 0.60 | 0.67 | 0.27 | 0.58 | 0.55 | 0.40 | 0.40 | 0.62 |
| `rfgm_belebele` | 0.56 | 0.47 | 0.56 | 0.64 | 0.49 | 0.56 | 0.58 | 0.51 | 0.62 | 0.44 | 0.45 | 0.42 | 0.51 | 0.40 | 0.60 |
| `rf_belebele` | 0.36 | 0.64 | 0.58 | 0.58 | 0.38 | 0.53 | 0.55 | 0.35 | 0.49 | 0.69 | 0.45 | 0.36 | 0.71 | 0.56 | 0.55 |
| `rf_m_mmlu` | 0.42 | 0.56 | 0.56 | 0.55 | 0.64 | 0.45 | 0.33 | 0.36 | 0.38 | 0.56 | 0.58 | 0.42 | 0.65 | 0.38 | 0.55 |
| `bbpb_include_v2_en` | 0.25 | 0.53 | 0.42 | 0.47 | 0.73 | 0.51 | 0.55 | 0.53 | 0.24 | 0.60 | 0.62 | 0.47 | 0.58 | 0.33 | 0.42 |
| `arc` | 0.40 | 0.36 | 0.62 | 0.56 | 0.42 | 0.42 | 0.44 | 0.44 | 0.56 | 0.38 | 0.31 | 0.65 | **0.76** | 0.49 | 0.33 |
| `include_base_44` | 0.49 | 0.47 | 0.45 | 0.47 | 0.38 | 0.36 | 0.44 | 0.40 | 0.45 | 0.36 | 0.51 | 0.69 | 0.51 | 0.56 | 0.47 |
| `rf_include_base_44` | 0.45 | 0.51 | 0.47 | 0.36 | 0.42 | 0.36 | 0.25 | 0.62 | 0.40 | 0.67 | 0.44 | 0.53 | 0.49 | 0.62 | 0.42 |
| `rfgm_include_base_44` | 0.35 | 0.40 | 0.58 | 0.53 | 0.44 | 0.27 | 0.38 | 0.38 | 0.45 | 0.45 | 0.38 | 0.53 | 0.55 | 0.56 | 0.38 |
| `global_piqa_parallel_cloze` | 0.35 | 0.45 | 0.33 | 0.36 | 0.33 | 0.51 | 0.38 | 0.40 | 0.62 | 0.55 | 0.42 | 0.31 | 0.60 | 0.40 | 0.45 |
| `belebele` | 0.16 | 0.18 | 0.27 | 0.07 | 0.15 | 0.36 | 0.53 | 0.35 | 0.44 | 0.31 | 0.55 | 0.49 | 0.38 | 0.60 | 0.40 |

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
subtitle: "Hungarian (hu) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Indonesian (id) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.97** | **0.87** | **0.78** | **0.93** | **0.92** | **0.89** | **0.81** | **0.93** | **0.89** | **0.83** | **0.92** | **0.80** | **0.83** | **0.75** | **0.88** |
| `bbpb_hellaswag` | **0.90** | **0.84** | **0.82** | **0.81** | **0.84** | **0.89** | **0.88** | **0.84** | **0.81** | **0.84** | **0.88** | **0.80** | **0.79** | **0.76** | **0.83** |
| `bbpb_xstorycloze` | **0.82** | **0.77** | 0.74 | **0.82** | 0.74 | **0.82** | 0.72 | **0.88** | **0.84** | 0.74 | **0.81** | **0.78** | **0.75** | 0.72 | **0.78** |
| `hellaswag` | **0.78** | **0.82** | **0.80** | **0.80** | **0.78** | **0.76** | **0.76** | 0.74 | **0.76** | **0.78** | **0.82** | **0.75** | 0.74 | 0.71 | **0.83** |
| `bbpb_rfgm_include_base_44` | **0.81** | **0.75** | 0.56 | 0.69 | 0.72 | **0.76** | 0.68 | **0.80** | 0.72 | 0.66 | 0.68 | 0.60 | 0.63 | 0.51 | **0.78** |
| `bbpb_rf_global_mmlu_full` | 0.55 | 0.65 | **0.78** | 0.74 | 0.72 | 0.68 | 0.71 | 0.72 | 0.64 | 0.66 | 0.68 | 0.57 | 0.68 | 0.70 | 0.72 |
| `bbpb_include_v2_og` | **0.75** | 0.69 | 0.69 | 0.68 | 0.60 | 0.57 | 0.69 | **0.79** | 0.62 | 0.55 | 0.63 | 0.66 | 0.62 | 0.69 | 0.59 |
| `bbpb_rf_m_mmlu` | 0.68 | 0.73 | **0.75** | 0.60 | 0.68 | 0.53 | **0.75** | 0.62 | 0.57 | 0.70 | 0.62 | 0.72 | 0.53 | **0.76** | 0.53 |
| `bbpb_xcopa` | **0.78** | 0.57 | 0.57 | 0.69 | 0.72 | 0.65 | 0.65 | 0.72 | 0.64 | 0.53 | 0.59 | 0.59 | 0.69 | 0.59 | 0.72 |
| `include_v2_og` | 0.62 | **0.77** | 0.66 | 0.58 | 0.68 | 0.68 | 0.59 | 0.69 | 0.56 | 0.65 | 0.66 | 0.74 | 0.59 | 0.62 | 0.60 |
| `bbpb_rf_mmmlu` | 0.62 | 0.73 | **0.76** | 0.61 | 0.65 | 0.49 | 0.72 | 0.60 | 0.56 | 0.69 | 0.62 | 0.68 | 0.57 | **0.76** | 0.57 |
| `bbpb_arc` | 0.53 | **0.77** | 0.68 | 0.68 | 0.62 | 0.48 | 0.72 | 0.71 | 0.55 | 0.67 | 0.62 | 0.62 | 0.68 | 0.60 | 0.64 |
| `bbpb_rfgm_belebele` | **0.85** | **0.79** | 0.65 | 0.68 | 0.57 | **0.79** | 0.73 | 0.65 | 0.52 | 0.69 | 0.54 | 0.49 | 0.57 | 0.45 | 0.45 |
| `bbpb_global_piqa_parallel_cloze` | **0.76** | 0.59 | 0.72 | 0.59 | 0.56 | 0.70 | **0.80** | 0.67 | 0.50 | 0.63 | 0.63 | 0.38 | 0.65 | 0.48 | 0.55 |
| `bbpb_rf_belebele` | **0.77** | 0.69 | **0.77** | 0.63 | 0.54 | 0.62 | 0.73 | 0.62 | 0.49 | 0.59 | 0.46 | 0.58 | 0.63 | 0.46 | 0.46 |
| `bbpb_belebele` | 0.53 | 0.56 | 0.72 | 0.64 | 0.57 | 0.69 | 0.63 | 0.47 | 0.48 | 0.64 | 0.52 | 0.51 | 0.57 | 0.70 | 0.57 |
| `bbpb_rf_include_base_44` | 0.57 | 0.69 | 0.60 | 0.46 | 0.64 | 0.44 | 0.55 | 0.69 | 0.62 | 0.61 | 0.48 | 0.60 | 0.56 | **0.76** | 0.52 |
| `bbpb_include_base_44` | 0.74 | 0.67 | 0.72 | 0.47 | 0.61 | 0.62 | 0.55 | 0.64 | 0.58 | 0.59 | 0.45 | 0.53 | 0.53 | 0.50 | 0.51 |
| `bbpb_global_mmlu_full` | 0.68 | 0.55 | 0.58 | 0.63 | 0.58 | 0.53 | 0.47 | 0.50 | 0.62 | 0.63 | 0.58 | 0.52 | 0.52 | 0.57 | 0.60 |
| `xstorycloze` | 0.61 | 0.39 | 0.56 | 0.53 | 0.53 | 0.57 | 0.63 | 0.60 | 0.54 | 0.59 | 0.51 | 0.73 | 0.51 | 0.58 | 0.68 |
| `bbpb_include_v2_en` | 0.47 | 0.56 | 0.42 | 0.57 | 0.58 | 0.50 | 0.47 | 0.57 | 0.54 | 0.67 | 0.57 | 0.69 | 0.59 | 0.64 | 0.60 |
| `rfgm_belebele` | 0.65 | 0.55 | 0.57 | 0.63 | 0.53 | 0.47 | 0.64 | 0.61 | 0.51 | 0.53 | 0.47 | 0.46 | 0.55 | 0.54 | 0.68 |
| `include_v2_en` | 0.45 | 0.49 | 0.42 | 0.53 | 0.62 | 0.54 | 0.47 | 0.62 | 0.50 | 0.65 | 0.50 | 0.69 | 0.57 | 0.62 | 0.58 |
| `rf_m_mmlu` | 0.67 | 0.33 | 0.57 | 0.50 | 0.40 | 0.47 | 0.47 | 0.65 | 0.49 | 0.47 | 0.56 | 0.66 | 0.48 | 0.47 | 0.72 |
| `arc` | 0.48 | 0.56 | 0.64 | 0.47 | 0.46 | 0.51 | 0.54 | 0.54 | 0.56 | 0.48 | 0.55 | 0.42 | 0.50 | 0.46 | 0.57 |
| `xcopa` | 0.36 | 0.64 | 0.47 | 0.62 | 0.58 | 0.40 | 0.62 | 0.53 | 0.47 | 0.47 | 0.62 | 0.42 | 0.52 | 0.49 | 0.53 |
| `belebele` | 0.43 | 0.44 | 0.60 | 0.38 | 0.53 | 0.68 | 0.59 | 0.55 | 0.51 | 0.57 | 0.40 | 0.47 | 0.49 | 0.61 | 0.46 |
| `rf_global_mmlu_full` | 0.62 | 0.57 | 0.36 | 0.42 | 0.43 | 0.54 | 0.48 | 0.38 | 0.50 | 0.51 | 0.41 | 0.49 | 0.58 | 0.57 | 0.74 |
| `rfgm_include_base_44` | 0.40 | 0.33 | 0.57 | 0.43 | 0.51 | 0.43 | 0.59 | 0.56 | 0.74 | 0.47 | 0.39 | 0.44 | 0.48 | 0.58 | 0.59 |
| `rf_include_base_44` | 0.43 | 0.59 | 0.47 | 0.45 | 0.48 | 0.49 | 0.41 | 0.53 | 0.53 | 0.40 | 0.44 | 0.49 | 0.60 | 0.60 | 0.59 |
| `include_base_44` | 0.55 | 0.59 | 0.38 | 0.54 | 0.42 | 0.48 | 0.65 | 0.58 | 0.54 | 0.42 | 0.46 | 0.32 | 0.56 | 0.56 | 0.45 |
| `global_mmlu_full` | 0.56 | 0.52 | 0.36 | 0.47 | 0.42 | 0.62 | 0.53 | 0.47 | 0.47 | 0.48 | 0.52 | 0.34 | 0.57 | 0.58 | 0.55 |
| `rf_mmmlu` | 0.44 | 0.57 | 0.53 | 0.53 | 0.38 | 0.52 | 0.41 | 0.56 | 0.52 | 0.37 | 0.41 | 0.46 | 0.46 | 0.49 | 0.65 |
| `rf_belebele` | 0.39 | 0.34 | 0.42 | 0.50 | 0.46 | 0.48 | 0.57 | 0.62 | 0.53 | 0.54 | 0.32 | 0.43 | 0.45 | 0.42 | 0.39 |
| `global_piqa_parallel_cloze` | 0.29 | 0.51 | 0.44 | 0.53 | 0.29 | 0.38 | 0.51 | 0.46 | 0.47 | 0.45 | 0.53 | 0.30 | 0.51 | 0.56 | 0.40 |

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
subtitle: "Indonesian (id) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Italian (it) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.99** | **0.88** | **0.84** | **0.94** | **0.94** | **0.88** | **0.85** | **0.94** | **0.94** | **0.86** | **0.91** | **0.86** | **0.86** | **0.83** | **0.92** |
| `bbpb_hellaswag` | **0.96** | **0.88** | **0.85** | **0.91** | **0.85** | **0.90** | **0.87** | **0.92** | **0.85** | **0.88** | **0.86** | **0.82** | **0.85** | **0.78** | **0.82** |
| `bbpb_rfgm_belebele` | **0.94** | **0.81** | **0.88** | **0.85** | 0.73 | **0.79** | **0.87** | **0.82** | 0.73 | **0.76** | **0.81** | **0.76** | **0.82** | 0.74 | 0.66 |
| `hellaswag` | **0.76** | 0.75 | **0.83** | **0.76** | 0.66 | **0.81** | **0.85** | **0.87** | 0.74 | **0.78** | **0.78** | 0.71 | **0.76** | 0.69 | **0.81** |
| `bbpb_xcopa` | **0.79** | 0.68 | 0.66 | 0.67 | **0.77** | **0.86** | **0.75** | **0.81** | 0.73 | **0.82** | **0.77** | 0.74 | 0.73 | 0.69 | 0.73 |
| `bbpb_rfgm_include_base_44` | 0.72 | 0.72 | 0.73 | 0.55 | 0.69 | **0.78** | 0.68 | 0.63 | 0.64 | 0.75 | 0.65 | 0.68 | 0.64 | **0.79** | 0.69 |
| `bbpb_multiblimp` | 0.72 | **0.77** | **0.79** | 0.58 | **0.78** | 0.73 | 0.61 | 0.73 | 0.58 | 0.68 | 0.61 | 0.71 | 0.61 | **0.80** | 0.59 |
| `bbpb_global_mmlu_full` | **0.82** | 0.74 | 0.75 | 0.70 | 0.63 | 0.65 | 0.68 | 0.70 | 0.56 | 0.72 | 0.65 | 0.66 | 0.68 | 0.61 | 0.73 |
| `multiblimp` | 0.67 | **0.77** | 0.73 | 0.72 | 0.68 | 0.67 | 0.65 | 0.62 | 0.66 | 0.75 | 0.65 | 0.71 | 0.62 | 0.75 | 0.63 |
| `lambada_openai_mt` | 0.67 | 0.67 | 0.65 | 0.62 | 0.64 | **0.76** | **0.79** | 0.61 | 0.66 | 0.70 | 0.66 | 0.62 | 0.68 | 0.69 | 0.74 |
| `bbpb_rf_belebele` | 0.69 | 0.68 | 0.72 | 0.73 | 0.67 | 0.67 | 0.69 | 0.71 | 0.55 | 0.61 | 0.66 | 0.66 | 0.72 | 0.58 | 0.66 |
| `bbpb_rf_global_mmlu_full` | 0.61 | 0.59 | 0.67 | 0.73 | 0.74 | 0.69 | 0.67 | **0.82** | 0.59 | 0.65 | 0.59 | 0.65 | 0.63 | 0.72 | 0.59 |
| `bbpb_include_base_44` | **0.84** | 0.69 | 0.73 | 0.64 | 0.64 | 0.64 | 0.69 | 0.58 | 0.58 | 0.57 | 0.66 | 0.65 | 0.56 | 0.57 | 0.63 |
| `bbpb_include_v2_og` | 0.65 | 0.72 | 0.66 | 0.71 | 0.57 | 0.63 | 0.66 | 0.73 | 0.57 | 0.58 | 0.61 | 0.64 | 0.66 | 0.67 | 0.63 |
| `bbpb_global_piqa_parallel_cloze` | **0.83** | 0.72 | **0.78** | 0.53 | 0.55 | **0.81** | 0.72 | 0.59 | 0.49 | 0.69 | 0.56 | 0.49 | 0.65 | 0.53 | 0.52 |
| `bbpb_toksuite` | 0.62 | 0.61 | 0.62 | 0.66 | 0.56 | 0.64 | 0.61 | 0.55 | 0.61 | 0.65 | 0.59 | 0.68 | 0.74 | 0.72 | 0.60 |
| `bbpb_rf_mmmlu` | 0.62 | 0.74 | 0.63 | 0.65 | 0.55 | 0.62 | 0.65 | 0.69 | 0.54 | 0.61 | 0.62 | 0.63 | 0.66 | 0.65 | 0.57 |
| `bbpb_rf_m_mmlu` | 0.63 | 0.72 | 0.57 | 0.67 | 0.57 | 0.64 | 0.66 | 0.69 | 0.51 | 0.57 | 0.61 | 0.60 | 0.66 | 0.64 | 0.57 |
| `bbpb_rf_include_base_44` | 0.56 | 0.70 | 0.62 | 0.50 | 0.61 | 0.62 | 0.64 | 0.64 | 0.66 | 0.59 | 0.62 | 0.68 | 0.50 | 0.68 | 0.51 |
| `bbpb_arc_mt` | 0.55 | 0.58 | 0.56 | 0.55 | 0.54 | 0.65 | **0.77** | 0.65 | 0.64 | 0.59 | 0.64 | 0.64 | 0.57 | 0.65 | 0.52 |
| `bbpb_arc` | 0.57 | 0.60 | 0.58 | 0.55 | 0.52 | 0.67 | **0.75** | 0.65 | 0.66 | 0.58 | 0.67 | 0.64 | 0.54 | 0.61 | 0.50 |
| `bbpb_belebele` | 0.59 | 0.62 | 0.67 | 0.70 | 0.60 | 0.67 | 0.71 | 0.53 | 0.52 | 0.64 | 0.55 | 0.53 | 0.58 | 0.52 | 0.53 |
| `rf_global_mmlu_full` | 0.55 | 0.44 | 0.50 | 0.49 | 0.56 | 0.64 | 0.53 | 0.56 | 0.65 | 0.72 | 0.67 | 0.71 | 0.57 | 0.61 | 0.71 |
| `bbpb_include_v2_en` | 0.49 | 0.64 | 0.62 | 0.58 | 0.62 | 0.57 | 0.48 | 0.59 | 0.38 | 0.66 | 0.68 | 0.59 | 0.57 | 0.51 | 0.63 |
| `include_v2_og` | 0.47 | 0.57 | 0.37 | 0.49 | 0.51 | 0.51 | 0.53 | 0.67 | 0.68 | 0.55 | 0.48 | 0.58 | 0.67 | 0.65 | 0.61 |
| `rf_m_mmlu` | 0.52 | 0.45 | 0.51 | 0.60 | 0.45 | 0.54 | 0.61 | 0.47 | 0.43 | 0.61 | 0.55 | 0.67 | 0.59 | 0.63 | 0.72 |
| `arc` | 0.45 | 0.52 | 0.41 | 0.52 | 0.54 | 0.48 | 0.65 | 0.54 | 0.63 | 0.46 | 0.59 | 0.61 | 0.66 | 0.59 | 0.57 |
| `rf_mmmlu` | 0.66 | 0.43 | 0.35 | 0.42 | 0.42 | 0.44 | 0.37 | 0.53 | 0.51 | 0.66 | 0.61 | 0.66 | 0.61 | 0.58 | 0.70 |
| `include_v2_en` | 0.50 | 0.49 | 0.61 | 0.67 | 0.55 | 0.39 | 0.42 | 0.54 | 0.51 | 0.47 | 0.59 | 0.43 | 0.63 | 0.48 | 0.57 |
| `arc_mt` | 0.51 | 0.64 | 0.42 | 0.49 | 0.56 | 0.54 | 0.59 | 0.42 | 0.51 | 0.45 | 0.71 | 0.55 | 0.54 | 0.41 | 0.48 |
| `rf_belebele` | 0.48 | 0.50 | 0.54 | 0.51 | 0.47 | 0.47 | 0.45 | 0.59 | 0.56 | 0.63 | 0.54 | 0.41 | 0.40 | 0.44 | 0.59 |
| `belebele` | 0.39 | 0.32 | 0.49 | 0.46 | 0.44 | 0.58 | 0.48 | 0.46 | 0.47 | 0.56 | 0.51 | 0.59 | **0.78** | 0.43 | 0.45 |
| `rfgm_belebele` | 0.50 | 0.41 | 0.47 | 0.55 | 0.54 | 0.47 | 0.47 | 0.46 | 0.50 | 0.46 | 0.59 | 0.36 | 0.47 | 0.51 | 0.61 |
| `global_mmlu_full` | 0.57 | 0.55 | 0.37 | 0.44 | 0.45 | 0.56 | 0.45 | 0.54 | 0.45 | 0.50 | 0.52 | 0.30 | 0.56 | 0.52 | 0.57 |
| `xcopa` | 0.53 | 0.37 | 0.56 | 0.34 | 0.46 | 0.48 | 0.42 | 0.46 | 0.43 | 0.39 | 0.54 | 0.47 | 0.60 | 0.49 | 0.65 |
| `rfgm_include_base_44` | 0.33 | 0.43 | 0.41 | 0.41 | 0.58 | 0.58 | 0.51 | 0.58 | 0.40 | 0.56 | 0.39 | 0.45 | 0.53 | 0.56 | 0.44 |
| `rf_include_base_44` | 0.46 | 0.54 | 0.61 | 0.44 | 0.38 | 0.58 | 0.45 | 0.52 | 0.51 | 0.49 | 0.36 | 0.45 | 0.44 | 0.49 | 0.39 |
| `global_piqa_parallel_cloze` | 0.54 | 0.61 | 0.54 | 0.42 | 0.56 | 0.58 | 0.38 | 0.47 | 0.40 | 0.41 | 0.41 | 0.50 | 0.45 | 0.44 | 0.39 |
| `include_base_44` | 0.49 | 0.55 | 0.29 | 0.52 | 0.49 | 0.65 | 0.55 | 0.31 | 0.55 | 0.45 | 0.44 | 0.45 | 0.33 | 0.42 | 0.46 |
| `toksuite` | 0.37 | 0.45 | 0.34 | 0.36 | 0.27 | 0.41 | 0.48 | 0.44 | 0.48 | 0.36 | 0.35 | 0.41 | 0.61 | 0.54 | 0.55 |

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
subtitle: "Italian (it) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Japanese (ja) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.97** | **0.86** | **0.86** | **0.93** | **0.94** | **0.89** | **0.89** | **0.95** | **0.95** | **0.89** | **0.90** | **0.85** | **0.91** | **0.86** | **0.92** |
| `bbpb_rf_belebele` | **0.86** | **0.84** | **0.80** | **0.79** | 0.68 | **0.77** | **0.83** | **0.78** | 0.71 | **0.76** | 0.75 | 0.70 | **0.78** | 0.74 | 0.71 |
| `bbpb_rf_global_mmlu_full` | **0.83** | 0.68 | 0.75 | **0.78** | 0.72 | **0.75** | **0.75** | **0.82** | **0.76** | **0.75** | 0.65 | 0.68 | 0.74 | **0.75** | 0.69 |
| `bbpb_paws` | **0.78** | **0.82** | **0.81** | **0.82** | 0.62 | **0.77** | **0.78** | **0.79** | 0.61 | **0.79** | **0.82** | 0.59 | **0.80** | 0.62 | 0.61 |
| `bbpb_global_mmlu_full` | **0.84** | 0.72 | **0.76** | 0.66 | 0.59 | 0.74 | 0.74 | 0.62 | 0.63 | 0.69 | 0.55 | 0.62 | 0.64 | 0.59 | 0.66 |
| `bbpb_global_piqa_parallel_cloze` | 0.66 | 0.68 | **0.78** | 0.66 | 0.57 | 0.74 | 0.68 | **0.75** | 0.60 | **0.75** | 0.68 | 0.54 | **0.75** | 0.53 | 0.56 |
| `bbpb_rf_mmmlu` | 0.57 | 0.72 | 0.65 | 0.55 | 0.66 | 0.60 | 0.66 | 0.68 | 0.57 | 0.72 | 0.67 | 0.69 | 0.65 | 0.74 | 0.63 |
| `bbpb_rf_include_base_44` | 0.59 | 0.73 | 0.71 | 0.68 | 0.57 | 0.63 | 0.70 | 0.52 | 0.62 | 0.71 | 0.69 | 0.64 | 0.62 | 0.63 | 0.61 |
| `bbpb_rfgm_belebele` | 0.68 | 0.69 | 0.67 | 0.57 | 0.67 | 0.70 | 0.63 | 0.71 | 0.61 | 0.58 | 0.64 | 0.56 | 0.64 | 0.62 | 0.57 |
| `bbpb_rfgm_include_base_44` | **0.84** | **0.80** | 0.67 | 0.52 | 0.57 | **0.77** | 0.72 | 0.42 | 0.59 | 0.72 | 0.55 | 0.62 | 0.45 | 0.57 | 0.44 |
| `bbpb_include_base_44` | **0.81** | 0.70 | 0.69 | 0.58 | 0.49 | 0.66 | 0.64 | 0.58 | 0.49 | 0.57 | 0.58 | 0.54 | 0.66 | 0.53 | 0.66 |
| `bbpb_rf_cultural_bench_easy` | 0.53 | 0.68 | 0.64 | 0.58 | 0.52 | 0.64 | 0.60 | 0.58 | 0.60 | 0.60 | 0.69 | 0.62 | 0.60 | 0.55 | 0.67 |
| `bbpb_belebele` | 0.72 | 0.59 | 0.72 | 0.57 | 0.58 | 0.66 | 0.64 | 0.60 | 0.66 | 0.59 | 0.46 | 0.60 | 0.63 | 0.56 | 0.53 |
| `bbpb_include_v2_en` | 0.74 | 0.60 | 0.58 | 0.68 | 0.62 | 0.59 | 0.59 | 0.65 | 0.54 | 0.59 | 0.51 | 0.52 | 0.60 | 0.55 | 0.53 |
| `rf_mmmlu` | 0.53 | 0.46 | 0.54 | 0.58 | 0.50 | 0.55 | 0.54 | 0.56 | 0.57 | 0.68 | 0.59 | 0.54 | 0.66 | 0.67 | 0.68 |
| `include_v2_en` | 0.63 | 0.59 | 0.53 | 0.57 | 0.64 | 0.53 | 0.50 | 0.62 | 0.49 | 0.56 | 0.65 | 0.56 | 0.63 | 0.54 | 0.53 |
| `bbpb_include_v2_og` | 0.45 | 0.70 | 0.62 | 0.49 | 0.61 | 0.51 | 0.48 | 0.56 | 0.52 | 0.59 | 0.51 | 0.58 | 0.49 | 0.57 | 0.57 |
| `rf_global_mmlu_full` | 0.43 | 0.47 | 0.49 | 0.52 | 0.50 | 0.51 | 0.38 | 0.60 | 0.45 | 0.55 | 0.62 | 0.54 | 0.66 | 0.73 | 0.66 |
| `bbpb_cultural_bench_easy` | 0.46 | 0.53 | 0.65 | 0.55 | 0.53 | 0.59 | 0.49 | 0.59 | 0.51 | 0.62 | 0.54 | 0.51 | 0.56 | 0.41 | 0.55 |
| `xwinograd` | 0.47 | 0.45 | 0.49 | 0.48 | 0.49 | 0.51 | 0.57 | 0.65 | 0.63 | 0.45 | 0.51 | 0.45 | 0.57 | 0.64 | 0.64 |
| `paws` | 0.58 | 0.54 | 0.53 | 0.48 | 0.50 | 0.39 | 0.62 | 0.52 | 0.51 | 0.51 | 0.55 | 0.54 | 0.44 | 0.35 | 0.57 |
| `global_mmlu_full` | 0.53 | 0.50 | 0.46 | 0.52 | 0.59 | 0.64 | 0.51 | 0.42 | 0.61 | 0.44 | 0.38 | 0.57 | 0.50 | 0.50 | 0.46 |
| `include_v2_og` | 0.46 | 0.49 | 0.35 | 0.45 | 0.51 | 0.38 | 0.40 | 0.50 | 0.37 | 0.53 | 0.56 | 0.64 | 0.54 | 0.62 | 0.61 |
| `include_base_44` | 0.40 | 0.47 | 0.54 | 0.43 | 0.45 | 0.44 | 0.60 | 0.52 | 0.51 | 0.51 | 0.46 | 0.41 | 0.50 | 0.54 | 0.61 |
| `belebele` | 0.51 | 0.45 | 0.62 | 0.61 | 0.47 | 0.57 | 0.49 | 0.41 | 0.32 | 0.39 | 0.54 | 0.29 | 0.57 | 0.55 | 0.55 |
| `rf_belebele` | 0.41 | 0.61 | 0.37 | 0.45 | 0.43 | 0.49 | 0.52 | 0.47 | 0.47 | 0.41 | 0.51 | 0.49 | 0.52 | 0.51 | 0.54 |
| `rf_include_base_44` | 0.51 | 0.47 | 0.55 | 0.49 | 0.48 | 0.45 | 0.36 | 0.53 | 0.48 | 0.44 | 0.41 | 0.53 | 0.45 | 0.56 | 0.47 |
| `rfgm_include_base_44` | 0.55 | 0.49 | 0.38 | 0.50 | 0.36 | 0.53 | 0.54 | 0.54 | 0.48 | 0.50 | 0.40 | 0.42 | 0.57 | 0.40 | 0.45 |
| `cultural_bench_easy` | 0.50 | 0.53 | 0.45 | 0.40 | 0.51 | 0.61 | 0.47 | 0.47 | 0.46 | 0.47 | 0.49 | 0.45 | 0.45 | 0.35 | 0.48 |
| `rfgm_belebele` | 0.34 | 0.42 | 0.41 | 0.39 | 0.42 | 0.41 | 0.45 | 0.42 | 0.42 | 0.55 | 0.52 | 0.47 | 0.54 | 0.41 | 0.59 |
| `cultural_bench_hard` | 0.38 | 0.42 | 0.43 | 0.36 | 0.55 | 0.43 | 0.52 | 0.68 | 0.43 | 0.46 | 0.38 | 0.35 | 0.53 | 0.46 | 0.37 |
| `global_piqa_parallel_cloze` | 0.63 | 0.38 | 0.38 | 0.31 | 0.41 | 0.44 | 0.44 | 0.36 | 0.49 | 0.45 | 0.59 | 0.43 | 0.40 | 0.49 | 0.51 |
| `rf_cultural_bench_easy` | 0.31 | 0.47 | 0.34 | 0.29 | 0.32 | 0.33 | 0.46 | 0.51 | 0.33 | 0.36 | 0.33 | 0.38 | 0.59 | 0.45 | 0.42 |

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
subtitle: "Georgian (ka) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_rfgm_include_base_44` | **0.93** | **0.79** | **0.89** | **0.86** | **0.82** | **0.86** | **0.89** | **0.86** | **0.82** | **0.89** | **0.93** | **0.82** | **0.96** | **0.93** | **0.89** |
| `bpb` | **0.89** | **0.89** | **0.86** | **0.79** | **0.93** | **0.93** | **0.89** | **0.89** | **0.89** | **0.96** | **0.89** | **0.82** | **0.86** | **0.79** | **0.79** |
| `bbpb_global_piqa_parallel_cloze` | **0.82** | **0.82** | **0.86** | **0.86** | **0.86** | **0.79** | **0.89** | **0.75** | **0.82** | **0.75** | **0.82** | **0.82** | **0.86** | **0.93** | **0.93** |
| `bbpb_rfgm_belebele` | **0.89** | 0.64 | **0.79** | **0.86** | **0.82** | **0.75** | **0.82** | **0.89** | **0.79** | **0.79** | **0.79** | **0.75** | **0.86** | **0.89** | **0.89** |
| `bbpb_multiblimp` | **0.82** | **0.93** | **0.86** | **0.86** | **0.79** | **0.89** | **0.75** | **0.82** | 0.61 | **0.79** | **0.79** | 0.71 | **0.79** | 0.71 | **0.79** |
| `bbpb_rf_belebele` | **0.75** | **0.89** | **0.79** | **0.86** | 0.71 | 0.71 | **0.75** | **0.82** | **0.75** | **0.82** | **0.82** | **0.75** | **0.79** | **0.86** | 0.64 |
| `bbpb_include_v2_og` | **0.79** | 0.71 | **0.82** | **0.89** | 0.68 | **0.79** | **0.82** | **0.89** | 0.54 | **0.75** | **0.75** | 0.61 | **0.86** | 0.71 | 0.57 |
| `bbpb_rf_include_base_44` | **0.82** | 0.61 | **0.86** | **0.82** | **0.75** | 0.64 | **0.75** | **1.00** | 0.71 | 0.54 | 0.64 | **0.79** | **0.75** | **0.75** | 0.71 |
| `bbpb_include_base_44` | 0.68 | **0.93** | **0.79** | 0.71 | 0.57 | 0.61 | **0.89** | 0.68 | 0.46 | 0.71 | 0.64 | 0.64 | 0.71 | 0.57 | 0.43 |
| `bbpb_belebele` | 0.64 | 0.54 | **0.82** | 0.61 | **0.75** | 0.61 | 0.46 | 0.46 | 0.46 | 0.50 | 0.57 | 0.64 | 0.71 | **0.79** | 0.50 |
| `bbpb_include_v2_en` | 0.68 | 0.68 | 0.57 | 0.54 | 0.64 | 0.71 | 0.61 | 0.57 | 0.46 | 0.32 | 0.50 | 0.46 | 0.61 | 0.36 | 0.39 |
| `rf_include_base_44` | 0.36 | 0.50 | 0.32 | 0.21 | 0.21 | 0.64 | 0.71 | 0.50 | 0.71 | 0.57 | 0.50 | 0.64 | 0.71 | 0.71 | 0.57 |
| `rfgm_include_base_44` | 0.18 | 0.29 | 0.57 | 0.71 | 0.61 | 0.57 | 0.43 | 0.43 | 0.36 | 0.64 | 0.32 | 0.57 | 0.61 | 0.68 | 0.64 |
| `rfgm_belebele` | 0.68 | 0.61 | 0.43 | 0.29 | 0.50 | 0.57 | 0.32 | 0.32 | **0.75** | 0.57 | 0.54 | 0.50 | 0.71 | 0.32 | 0.46 |
| `include_base_44` | 0.46 | 0.57 | 0.68 | **0.75** | 0.50 | 0.25 | 0.18 | 0.57 | 0.54 | 0.68 | 0.43 | 0.25 | 0.61 | 0.50 | 0.61 |
| `global_piqa_parallel_cloze` | 0.64 | 0.54 | 0.43 | 0.54 | **0.75** | 0.68 | 0.36 | 0.57 | 0.50 | 0.43 | 0.39 | 0.39 | 0.32 | 0.46 | 0.46 |
| `include_v2_og` | 0.36 | 0.25 | 0.21 | 0.29 | 0.36 | 0.54 | 0.61 | 0.29 | 0.61 | 0.64 | 0.61 | 0.57 | 0.61 | **0.79** | 0.54 |
| `rf_belebele` | 0.64 | 0.29 | 0.71 | 0.50 | 0.43 | 0.43 | 0.54 | 0.43 | 0.57 | 0.46 | 0.46 | 0.18 | 0.71 | 0.36 | 0.46 |
| `multiblimp` | 0.68 | 0.46 | 0.68 | 0.32 | 0.43 | 0.39 | 0.68 | 0.46 | 0.54 | 0.46 | 0.32 | 0.39 | 0.32 | 0.64 | 0.36 |
| `belebele` | 0.64 | 0.57 | 0.36 | 0.32 | 0.50 | **0.75** | 0.46 | 0.14 | 0.43 | 0.32 | 0.25 | 0.54 | 0.50 | 0.32 | 0.54 |
| `include_v2_en` | 0.25 | 0.43 | 0.39 | 0.32 | 0.64 | 0.57 | 0.43 | 0.64 | 0.39 | 0.36 | 0.68 | 0.18 | 0.57 | 0.46 | 0.14 |

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
subtitle: "Georgian (ka) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Kazakh (kk) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_rf_belebele` | **1.00** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** |
| `bbpb_rf_include_base_44` | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** | **1.00** | **0.83** |
| `bpb` | **1.00** | **0.83** | 0.67 | **1.00** | **1.00** | **0.83** | 0.67 | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** |
| `bbpb_global_piqa_nonparallel_cloze` | **0.83** | **0.83** | 0.67 | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** |
| `bbpb_include_v2_og` | **1.00** | **0.83** | **0.83** | 0.67 | **1.00** | **0.83** | **0.83** | 0.67 | **1.00** | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 |
| `bbpb_multiblimp` | **1.00** | **1.00** | **0.83** | **0.83** | 0.67 | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | 0.67 | **0.83** | **0.83** |
| `bbpb_rfgm_belebele` | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** | 0.67 |
| `bbpb_rfgm_include_base_44` | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 |
| `multiblimp` | 0.67 | 0.67 | 0.67 | **0.83** | 0.67 | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **0.83** | 0.67 |
| `global_piqa_nonparallel_cloze` | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 | **0.83** | 0.50 | 0.67 | **0.83** | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 |
| `bbpb_belebele` | 0.67 | 0.67 | **1.00** | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.33 | 0.67 | **1.00** | 0.33 | 0.67 | 0.67 | 0.33 |
| `rf_belebele` | 0.17 | 0.33 | 0.33 | 0.17 | 0.33 | **0.83** | **0.83** | 0.67 | 0.50 | **1.00** | **0.83** | 0.67 | **0.83** | 0.67 | **0.83** |
| `rfgm_belebele` | 0.50 | 0.33 | **0.83** | 0.33 | **0.83** | 0.17 | 0.33 | 0.00 | 0.33 | 0.50 | **0.83** | 0.50 | 0.50 | **1.00** | 0.50 |
| `bbpb_include_base_44` | **0.83** | **1.00** | 0.67 | 0.17 | 0.33 | **0.83** | 0.50 | 0.33 | 0.17 | 0.67 | 0.17 | 0.33 | 0.17 | 0.67 | 0.50 |
| `bbpb_include_v2_en` | 0.17 | 0.67 | 0.17 | **0.83** | **0.83** | 0.50 | **1.00** | 0.33 | 0.00 | 0.50 | **0.83** | 0.50 | 0.33 | 0.00 | 0.67 |
| `include_v2_en` | 0.50 | **0.83** | 0.33 | 0.50 | 0.50 | 0.67 | **0.83** | 0.67 | 0.00 | 0.50 | 0.67 | 0.33 | 0.50 | 0.17 | 0.33 |
| `include_v2_og` | 0.50 | 0.17 | 0.17 | 0.33 | 0.33 | 0.17 | 0.17 | 0.00 | 0.33 | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 |
| `include_base_44` | 0.17 | 0.67 | 0.33 | 0.50 | 0.50 | 0.17 | 0.50 | 0.67 | 0.67 | 0.33 | 0.17 | 0.17 | 0.50 | 0.50 | **1.00** |
| `belebele` | **0.83** | 0.67 | 0.17 | 0.50 | 0.17 | 0.50 | 0.33 | 0.33 | 0.33 | 0.17 | 0.67 | 0.17 | 0.33 | **1.00** | 0.33 |
| `rfgm_include_base_44` | 0.17 | 0.17 | 0.67 | 0.50 | 0.50 | 0.33 | 0.50 | 0.50 | 0.17 | 0.17 | 0.33 | 0.50 | **0.83** | 0.50 | 0.33 |
| `rf_include_base_44` | 0.67 | 0.33 | 0.17 | 0.33 | 0.50 | 0.67 | 0.33 | 0.00 | 0.17 | 0.33 | 0.33 | 0.17 | 0.50 | 0.67 | **0.83** |

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
subtitle: "Kazakh (kk) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Korean (ko) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.96** | **0.90** | **0.83** | **0.92** | **0.96** | **0.94** | **0.87** | **0.94** | **0.92** | **0.88** | **0.90** | **0.86** | **0.86** | **0.79** | **0.91** |
| `bbpb_kobest_hellaswag` | **0.88** | **0.87** | **0.85** | **0.91** | **0.94** | **0.91** | **0.88** | **0.87** | **0.87** | **0.87** | **0.83** | **0.88** | **0.81** | **0.81** | **0.92** |
| `bbpb_rfgm_include_base_44` | 0.69 | **0.86** | **0.79** | **0.83** | **0.79** | 0.71 | **0.77** | **0.76** | **0.79** | **0.81** | **0.85** | **0.81** | **0.76** | **0.85** | 0.73 |
| `bbpb_rf_include_base_44` | **0.87** | **0.77** | **0.77** | **0.76** | 0.65 | **0.82** | **0.82** | 0.73 | 0.65 | 0.69 | 0.68 | 0.73 | **0.86** | 0.73 | **0.85** |
| `bbpb_paws` | **0.78** | **0.76** | **0.76** | **0.76** | 0.71 | **0.79** | 0.69 | **0.82** | 0.69 | **0.77** | **0.87** | 0.72 | 0.74 | 0.67 | **0.77** |
| `bbpb_include_v2_og` | 0.72 | 0.71 | **0.76** | 0.73 | 0.69 | **0.76** | **0.76** | 0.65 | 0.64 | 0.69 | **0.77** | 0.73 | 0.64 | 0.63 | 0.68 |
| `bbpb_global_mmlu_full` | **0.86** | 0.58 | 0.73 | **0.77** | 0.73 | 0.69 | 0.69 | 0.73 | **0.85** | 0.54 | 0.55 | 0.72 | 0.65 | 0.64 | 0.68 |
| `bbpb_rf_belebele` | 0.73 | **0.78** | 0.71 | 0.72 | 0.68 | 0.64 | 0.64 | **0.78** | 0.54 | 0.64 | 0.68 | 0.72 | 0.73 | 0.69 | 0.60 |
| `bbpb_rf_mmmlu` | 0.67 | 0.72 | 0.63 | 0.50 | 0.62 | **0.79** | 0.71 | 0.68 | 0.72 | 0.58 | 0.58 | 0.62 | 0.69 | 0.71 | **0.76** |
| `bbpb_rfgm_belebele` | 0.69 | 0.69 | 0.68 | 0.58 | 0.58 | 0.72 | 0.71 | 0.68 | **0.76** | 0.68 | **0.76** | 0.53 | 0.67 | 0.67 | 0.56 |
| `bbpb_rf_global_mmlu_full` | 0.73 | 0.68 | 0.62 | 0.71 | 0.74 | 0.59 | 0.55 | **0.79** | 0.65 | 0.53 | 0.49 | 0.53 | 0.55 | 0.49 | **0.76** |
| `bbpb_include_base_44` | 0.67 | 0.60 | 0.64 | 0.64 | 0.68 | 0.55 | 0.64 | 0.64 | 0.58 | 0.42 | 0.50 | 0.64 | 0.72 | 0.50 | 0.65 |
| `bbpb_global_piqa_parallel_cloze` | 0.63 | 0.67 | 0.54 | 0.60 | 0.64 | 0.73 | 0.68 | 0.56 | 0.45 | **0.77** | 0.63 | 0.51 | 0.65 | 0.46 | 0.45 |
| `bbpb_belebele` | 0.63 | 0.56 | **0.78** | 0.64 | 0.54 | 0.55 | 0.62 | 0.58 | 0.65 | 0.60 | 0.33 | 0.62 | 0.58 | 0.60 | 0.54 |
| `kobest_copa` | 0.53 | 0.50 | 0.58 | 0.64 | 0.63 | 0.54 | 0.46 | 0.62 | 0.49 | 0.51 | 0.54 | 0.64 | 0.59 | 0.67 | 0.59 |
| `bbpb_include_v2_en` | 0.35 | 0.56 | 0.47 | 0.67 | 0.72 | 0.65 | 0.62 | 0.37 | 0.35 | 0.71 | 0.64 | 0.49 | 0.63 | 0.47 | 0.69 |
| `bbpb_kobest_sentineg` | 0.53 | 0.40 | 0.42 | 0.40 | 0.45 | 0.59 | 0.49 | 0.49 | 0.72 | 0.64 | 0.69 | 0.54 | 0.69 | 0.54 | 0.49 |
| `include_v2_og` | 0.72 | 0.60 | **0.79** | 0.44 | 0.45 | 0.49 | 0.74 | 0.51 | 0.56 | 0.62 | 0.27 | 0.32 | 0.40 | 0.49 | 0.60 |
| `rf_global_mmlu_full` | 0.40 | 0.62 | 0.54 | 0.42 | 0.44 | 0.44 | 0.50 | 0.53 | 0.49 | 0.63 | 0.53 | 0.37 | 0.58 | 0.46 | **0.76** |
| `include_base_44` | 0.44 | 0.45 | 0.50 | 0.41 | 0.41 | 0.45 | 0.40 | 0.53 | 0.50 | 0.56 | 0.56 | 0.72 | 0.50 | 0.56 | 0.51 |
| `global_mmlu_full` | 0.56 | 0.42 | 0.42 | 0.37 | 0.64 | 0.60 | 0.45 | 0.68 | 0.58 | 0.28 | 0.54 | 0.44 | 0.51 | 0.38 | 0.56 |
| `rfgm_include_base_44` | 0.59 | 0.40 | 0.54 | 0.45 | 0.50 | 0.36 | 0.69 | 0.59 | 0.53 | 0.33 | 0.49 | 0.58 | 0.55 | 0.44 | 0.40 |
| `include_v2_en` | 0.46 | 0.33 | 0.37 | 0.65 | 0.47 | 0.44 | 0.60 | 0.35 | 0.40 | 0.62 | 0.54 | 0.58 | 0.42 | 0.55 | 0.58 |
| `paws` | 0.63 | 0.36 | 0.56 | 0.49 | 0.50 | 0.55 | 0.59 | 0.37 | 0.46 | 0.46 | 0.42 | 0.40 | 0.41 | 0.64 | 0.50 |
| `kobest_sentineg` | 0.36 | 0.41 | 0.49 | 0.54 | 0.54 | 0.45 | 0.38 | 0.28 | 0.46 | 0.38 | 0.47 | 0.44 | 0.67 | 0.73 | 0.62 |
| `belebele` | 0.42 | 0.44 | 0.47 | 0.40 | 0.54 | 0.51 | 0.45 | 0.24 | 0.36 | 0.56 | 0.41 | 0.45 | 0.51 | 0.59 | 0.64 |
| `rfgm_belebele` | 0.53 | 0.42 | 0.42 | 0.36 | 0.32 | 0.45 | 0.46 | 0.36 | 0.62 | 0.40 | 0.45 | 0.49 | 0.65 | 0.55 | 0.47 |
| `rf_mmmlu` | 0.27 | 0.53 | 0.62 | 0.42 | 0.45 | 0.42 | 0.29 | 0.50 | 0.67 | 0.51 | 0.29 | 0.46 | 0.46 | 0.38 | 0.51 |
| `rf_belebele` | 0.32 | 0.33 | 0.41 | 0.49 | 0.45 | 0.49 | 0.32 | 0.31 | 0.44 | 0.49 | 0.47 | 0.41 | 0.58 | 0.44 | 0.62 |
| `global_piqa_parallel_cloze` | 0.37 | 0.33 | 0.46 | 0.49 | 0.58 | 0.47 | 0.45 | 0.51 | 0.37 | 0.49 | 0.42 | 0.37 | 0.37 | 0.18 | 0.63 |
| `rf_include_base_44` | 0.37 | 0.44 | 0.41 | 0.41 | 0.31 | 0.32 | 0.72 | 0.49 | 0.54 | 0.35 | 0.49 | 0.13 | 0.37 | 0.64 | 0.46 |
| `kobest_hellaswag` | 0.37 | 0.42 | 0.44 | 0.40 | 0.38 | 0.42 | 0.51 | 0.29 | 0.50 | 0.32 | 0.37 | 0.22 | 0.44 | 0.40 | 0.54 |

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
subtitle: "Lithuanian (lt) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_multiblimp` | **1.00** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | **1.00** |
| `bbpb_rfgm_include_base_44` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_global_piqa_parallel_cloze` | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** |
| `bbpb_rf_include_base_44` | **1.00** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 |
| `bbpb_rf_global_mmlu_full` | **1.00** | **0.83** | **1.00** | **1.00** | 0.67 | **0.83** | **1.00** | **1.00** | 0.67 | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | 0.67 |
| `bbpb_include_v2_og` | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | 0.67 | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | 0.67 |
| `bbpb_rf_belebele` | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | **1.00** | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** | 0.67 |
| `bbpb_rfgm_belebele` | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** |
| `multiblimp` | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 | **1.00** | 0.67 | **0.83** | 0.67 | **0.83** | 0.67 |
| `bbpb_belebele` | 0.67 | **0.83** | **0.83** | 0.50 | 0.67 | **0.83** | **0.83** | 0.50 | 0.33 | 0.67 | 0.33 | 0.50 | 0.67 | 0.50 | **0.83** |
| `bbpb_include_base_44` | **0.83** | 0.33 | 0.33 | 0.67 | 0.50 | 0.50 | 0.50 | **0.83** | 0.33 | **1.00** | 0.67 | **0.83** | 0.67 | **0.83** | 0.50 |
| `bbpb_global_mmlu_full` | **0.83** | 0.33 | 0.50 | 0.67 | 0.50 | 0.50 | 0.33 | **0.83** | 0.33 | 0.50 | 0.67 | **0.83** | 0.50 | 0.67 | 0.50 |
| `rfgm_belebele` | **0.83** | **0.83** | 0.50 | 0.50 | 0.33 | **1.00** | 0.33 | 0.67 | 0.50 | 0.33 | 0.67 | 0.50 | 0.33 | 0.17 | **0.83** |
| `rf_belebele` | **1.00** | 0.33 | 0.50 | 0.50 | 0.67 | 0.33 | 0.50 | 0.50 | 0.67 | 0.50 | **0.83** | 0.50 | 0.67 | 0.17 | 0.50 |
| `include_v2_en` | 0.67 | 0.17 | **0.83** | 0.33 | 0.33 | 0.50 | **0.83** | 0.67 | 0.67 | 0.33 | 0.50 | 0.50 | 0.50 | 0.50 | **0.83** |
| `belebele` | **0.83** | 0.17 | 0.50 | 0.33 | **1.00** | 0.33 | 0.67 | 0.17 | **0.83** | 0.67 | 0.50 | 0.17 | 0.50 | 0.50 | 0.33 |
| `global_mmlu_full` | 0.33 | **0.83** | 0.17 | 0.17 | 0.50 | 0.17 | 0.50 | 0.50 | **0.83** | 0.33 | 0.33 | 0.33 | **1.00** | 0.67 | 0.67 |
| `include_base_44` | 0.50 | 0.50 | 0.17 | 0.17 | 0.33 | 0.33 | 0.33 | 0.67 | 0.17 | 0.67 | 0.67 | 0.50 | 0.67 | **0.83** | 0.50 |
| `rfgm_include_base_44` | 0.33 | 0.17 | 0.50 | 0.17 | 0.17 | **0.83** | 0.67 | 0.67 | 0.50 | 0.50 | 0.67 | 0.50 | 0.33 | 0.50 | 0.50 |
| `include_v2_og` | 0.33 | 0.33 | 0.50 | 0.50 | 0.33 | 0.67 | **0.83** | 0.33 | 0.17 | 0.50 | 0.67 | 0.50 | 0.17 | 0.00 | **0.83** |
| `rf_global_mmlu_full` | 0.67 | 0.33 | 0.17 | 0.17 | **0.83** | 0.33 | 0.17 | 0.00 | 0.50 | **0.83** | 0.67 | 0.50 | **0.83** | 0.33 | 0.33 |
| `global_piqa_parallel_cloze` | 0.50 | 0.33 | **1.00** | 0.33 | **0.83** | 0.00 | 0.50 | 0.50 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | **0.83** | 0.17 |
| `bbpb_include_v2_en` | 0.17 | 0.17 | 0.17 | 0.67 | 0.67 | 0.67 | 0.67 | 0.17 | 0.50 | 0.67 | 0.17 | 0.50 | 0.50 | 0.17 | 0.33 |
| `rf_include_base_44` | 0.17 | 0.67 | 0.33 | 0.17 | 0.00 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.33 | 0.17 | 0.17 | 0.67 | 0.50 |

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
subtitle: "Lithuanian (lt) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Latvian (lv) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_multiblimp-extra` | **1.00** | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** |
| `multiblimp-extra` | **1.00** | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** |
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_rf_belebele` | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** |
| `bbpb_rfgm_belebele` | **1.00** | **1.00** | 0.67 | **0.83** | **1.00** | **1.00** | 0.67 | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** |
| `bbpb_belebele` | **0.83** | 0.50 | 0.50 | 0.67 | **0.83** | 0.67 | 0.67 | 0.50 | 0.67 | 0.67 | 0.17 | 0.33 | 0.50 | 0.33 | **0.83** |
| `rf_belebele` | 0.50 | 0.33 | 0.50 | 0.33 | 0.17 | 0.50 | **1.00** | 0.50 | 0.50 | 0.50 | **1.00** | 0.67 | 0.50 | 0.50 | 0.67 |
| `rfgm_belebele` | 0.00 | 0.50 | 0.50 | 0.33 | 0.33 | 0.33 | 0.33 | 0.33 | 0.67 | **1.00** | 0.67 | 0.50 | 0.67 | 0.50 | 0.50 |
| `belebele` | 0.33 | **0.83** | 0.17 | 0.33 | 0.33 | 0.50 | 0.33 | **1.00** | 0.50 | 0.00 | 0.50 | 0.50 | 0.33 | 0.50 | 0.50 |

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
subtitle: "Latvian (lv) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Malayalam (ml) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.89** | **0.86** | **0.86** | **0.82** | **0.86** | **0.96** | **0.96** | **0.93** | **0.89** | **0.93** | **0.96** | **0.86** | **0.89** | **0.86** | **0.89** |
| `bbpb_rfgm_include_base_44` | **0.89** | **0.75** | **0.89** | **0.86** | **0.82** | **0.86** | **0.86** | **0.89** | **0.93** | **0.86** | **0.89** | **0.93** | **0.89** | **0.79** | **0.89** |
| `bbpb_global_piqa_parallel_cloze` | **0.79** | **0.93** | **0.93** | **0.93** | **0.82** | **0.79** | **0.86** | **0.79** | **0.75** | **0.93** | **0.93** | **0.82** | **0.93** | **0.82** | **0.89** |
| `bbpb_hellaswag` | **0.75** | **0.75** | **0.79** | **0.82** | **0.86** | **0.93** | **0.96** | **0.86** | **0.89** | **0.89** | **0.86** | **0.82** | **0.89** | **0.93** | **0.82** |
| `bbpb_rfgm_belebele` | **0.89** | **0.93** | **0.86** | **0.89** | **0.79** | **0.82** | **0.89** | **0.79** | **0.89** | **0.93** | **0.89** | **0.79** | **0.82** | **0.86** | 0.68 |
| `bbpb_rf_belebele` | 0.71 | **0.79** | **0.79** | **0.89** | **0.86** | **0.79** | **0.79** | **0.82** | **0.79** | **0.86** | **0.89** | **0.93** | **0.82** | **0.86** | **0.89** |
| `bbpb_arc` | **0.82** | **0.82** | 0.64 | **0.86** | 0.68 | **0.79** | **0.75** | **0.89** | **0.79** | **0.82** | **0.89** | **0.79** | **0.79** | **0.89** | **0.82** |
| `bbpb_include_v2_og` | **0.79** | 0.68 | 0.71 | **0.75** | **0.82** | **0.82** | 0.64 | 0.68 | 0.68 | **0.75** | 0.64 | 0.71 | **0.89** | **0.89** | **0.86** |
| `bbpb_rf_m_mmlu` | **0.79** | 0.71 | 0.64 | **0.79** | **0.89** | **0.86** | 0.64 | **0.79** | **0.75** | **0.79** | 0.71 | **0.75** | 0.64 | 0.68 | **0.89** |
| `bbpb_rf_include_base_44` | 0.64 | 0.71 | 0.54 | **0.75** | **0.82** | **0.79** | **0.82** | **0.75** | **0.75** | **0.75** | 0.61 | 0.54 | **0.79** | 0.64 | **0.86** |
| `bbpb_include_base_44` | 0.64 | 0.57 | 0.71 | 0.57 | 0.57 | 0.57 | 0.71 | 0.71 | 0.43 | 0.64 | 0.64 | 0.71 | **0.86** | 0.71 | 0.57 |
| `bbpb_belebele` | 0.50 | 0.64 | **0.89** | 0.50 | 0.71 | 0.57 | 0.39 | 0.43 | 0.43 | 0.61 | 0.43 | 0.64 | 0.61 | **0.75** | 0.50 |
| `include_v2_en` | 0.57 | 0.64 | 0.54 | 0.43 | 0.54 | 0.50 | 0.54 | **0.75** | 0.61 | 0.36 | 0.54 | 0.54 | 0.39 | 0.54 | 0.61 |
| `rfgm_include_base_44` | 0.32 | 0.50 | 0.64 | 0.71 | 0.57 | 0.50 | 0.43 | 0.39 | 0.32 | 0.64 | 0.54 | 0.61 | 0.61 | 0.54 | 0.61 |
| `bbpb_include_v2_en` | 0.32 | 0.68 | 0.46 | 0.68 | 0.61 | 0.29 | 0.36 | 0.36 | 0.29 | 0.64 | 0.71 | 0.57 | 0.57 | 0.50 | 0.64 |
| `rf_m_mmlu` | 0.43 | 0.54 | 0.46 | 0.39 | 0.61 | 0.57 | 0.57 | 0.36 | 0.68 | 0.71 | 0.43 | 0.50 | 0.64 | 0.36 | 0.39 |
| `arc` | 0.61 | 0.57 | 0.57 | 0.43 | 0.43 | 0.57 | 0.36 | 0.46 | 0.39 | 0.43 | 0.64 | 0.39 | 0.46 | 0.43 | 0.61 |
| `include_base_44` | 0.46 | 0.50 | 0.32 | 0.50 | 0.46 | 0.36 | 0.61 | 0.68 | 0.61 | 0.50 | 0.50 | 0.32 | 0.36 | 0.64 | 0.46 |
| `rf_belebele` | 0.57 | 0.21 | 0.32 | **0.75** | 0.50 | 0.50 | 0.61 | 0.36 | 0.61 | 0.54 | 0.36 | 0.68 | 0.43 | 0.36 | 0.43 |
| `rfgm_belebele` | 0.21 | 0.21 | 0.61 | 0.39 | 0.43 | 0.61 | 0.50 | 0.50 | 0.68 | 0.43 | 0.64 | 0.39 | 0.46 | 0.61 | 0.36 |
| `belebele` | 0.50 | 0.36 | 0.54 | 0.71 | 0.29 | 0.39 | 0.46 | 0.32 | 0.39 | 0.32 | 0.36 | **0.89** | **0.75** | 0.25 | 0.25 |
| `include_v2_og` | 0.32 | 0.07 | 0.39 | 0.57 | 0.39 | 0.46 | 0.43 | 0.36 | 0.54 | 0.57 | 0.36 | 0.46 | 0.57 | 0.57 | 0.68 |
| `rf_include_base_44` | 0.32 | 0.32 | 0.36 | 0.32 | 0.43 | 0.46 | 0.46 | 0.61 | 0.57 | 0.14 | 0.25 | 0.39 | **0.75** | 0.61 | 0.61 |
| `hellaswag` | 0.25 | 0.43 | 0.46 | 0.43 | 0.39 | 0.39 | 0.54 | 0.57 | 0.71 | 0.64 | 0.07 | 0.25 | 0.39 | 0.39 | 0.64 |
| `global_piqa_parallel_cloze` | 0.36 | 0.61 | 0.54 | 0.36 | 0.29 | 0.43 | 0.21 | 0.18 | 0.18 | 0.43 | 0.29 | 0.25 | 0.43 | 0.46 | 0.57 |

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
subtitle: "Malayalam (ml) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Marathi (mr) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_rf_belebele` | **1.00** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** |
| `bbpb_arc` | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** |
| `bbpb_hellaswag` | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** |
| `bpb` | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** |
| `bbpb_rfgm_belebele` | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** | 0.67 | **0.83** |
| `bbpb_rf_m_mmlu` | **1.00** | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** | 0.67 | 0.67 | 0.67 | 0.67 | **1.00** |
| `bbpb_global_piqa_parallel_cloze` | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | 0.67 | 0.67 | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 |
| `bbpb_multiblimp` | 0.67 | **0.83** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** |
| `multiblimp` | **0.83** | 0.67 | 0.67 | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** |
| `bbpb_include_v2_en` | 0.33 | **0.83** | **1.00** | **0.83** | **0.83** | 0.50 | 0.33 | 0.50 | 0.17 | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** | 0.67 |
| `bbpb_include_v2_og` | **0.83** | 0.33 | **0.83** | **0.83** | 0.50 | 0.50 | **1.00** | 0.67 | 0.67 | 0.50 | 0.50 | **0.83** | 0.67 | 0.67 | 0.67 |
| `bbpb_belebele` | 0.67 | **0.83** | 0.67 | 0.67 | 0.50 | **0.83** | 0.33 | 0.67 | 0.50 | 0.50 | **0.83** | 0.67 | 0.33 | 0.50 | **0.83** |
| `rf_belebele` | 0.17 | 0.50 | **0.83** | 0.67 | 0.33 | 0.67 | 0.33 | 0.50 | 0.67 | 0.67 | **0.83** | 0.67 | **0.83** | 0.50 | 0.50 |
| `hellaswag` | 0.67 | 0.50 | 0.67 | 0.50 | **0.83** | 0.50 | 0.67 | 0.17 | **0.83** | 0.17 | 0.67 | 0.33 | 0.50 | **0.83** | 0.33 |
| `rf_m_mmlu` | 0.50 | 0.50 | 0.67 | 0.17 | 0.67 | 0.67 | **0.83** | 0.67 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.33 | 0.17 |
| `belebele` | **0.83** | 0.50 | 0.50 | 0.00 | 0.50 | 0.67 | 0.33 | 0.17 | 0.67 | 0.00 | 0.50 | 0.33 | 0.50 | 0.67 | 0.50 |
| `arc` | 0.33 | 0.67 | 0.17 | 0.33 | 0.50 | 0.33 | 0.50 | 0.67 | 0.50 | 0.50 | 0.00 | 0.17 | 0.50 | 0.17 | 0.67 |
| `rfgm_belebele` | 0.33 | 0.50 | 0.50 | 0.17 | 0.50 | 0.17 | 0.50 | **0.83** | 0.50 | 0.33 | 0.17 | 0.67 | 0.50 | 0.00 | 0.33 |
| `include_v2_og` | 0.33 | 0.17 | 0.67 | **0.83** | 0.17 | 0.67 | 0.17 | 0.50 | 0.17 | 0.00 | 0.33 | 0.50 | 0.67 | 0.50 | 0.17 |
| `global_piqa_parallel_cloze` | 0.33 | 0.50 | 0.00 | 0.00 | 0.17 | 0.50 | 0.67 | 0.67 | 0.17 | 0.33 | 0.50 | 0.33 | **0.83** | 0.33 | 0.50 |
| `include_v2_en` | 0.33 | 0.67 | 0.33 | 0.50 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 | 0.33 | 0.50 | 0.33 | **0.83** | 0.67 | **0.83** |

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
subtitle: "Marathi (mr) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Malay (ms) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_rfgm_include_base_44` | **0.83** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** |
| `bbpb_rf_global_mmlu_full` | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** |
| `bbpb_rf_include_base_44` | 0.67 | **1.00** | **0.83** | **1.00** | **1.00** | 0.67 | **0.83** | 0.67 | 0.67 | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** |
| `bbpb_include_v2_og_malay_malaysia` | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** |
| `bbpb_global_piqa_parallel_cloze` | **0.83** | **0.83** | **0.83** | **1.00** | **0.83** | 0.67 | 0.67 | **0.83** | **1.00** | **1.00** | **0.83** | 0.67 | **0.83** | 0.67 | **0.83** |
| `bbpb_rfgm_belebele` | **0.83** | **0.83** | 0.67 | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** |
| `bbpb_rf_belebele` | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | **0.83** | **1.00** | 0.67 | **0.83** | 0.67 | **1.00** | **0.83** | **0.83** | 0.67 |
| `bbpb_belebele` | 0.50 | 0.50 | 0.67 | **0.83** | **1.00** | **1.00** | **0.83** | 0.67 | 0.50 | **0.83** | 0.67 | 0.50 | **0.83** | 0.67 | **0.83** |
| `bbpb_include_v2_en_malay_singapore` | 0.67 | 0.33 | **0.83** | **0.83** | **0.83** | 0.67 | 0.50 | **0.83** | **0.83** | 0.17 | 0.50 | 0.50 | 0.67 | 0.67 | **1.00** |
| `rfgm_include_base_44` | **0.83** | 0.50 | 0.67 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.17 | **1.00** | **1.00** | 0.17 | 0.17 | **1.00** |
| `bbpb_include_v2_og_malay_singapore` | 0.50 | 0.50 | 0.17 | 0.33 | 0.33 | 0.67 | 0.67 | **0.83** | **0.83** | 0.33 | 0.50 | **0.83** | **0.83** | 0.50 | 0.67 |
| `include_base_44` | 0.50 | 0.67 | 0.50 | 0.67 | **0.83** | **0.83** | 0.67 | 0.17 | 0.33 | **0.83** | 0.33 | 0.50 | 0.50 | 0.33 | **0.83** |
| `belebele` | 0.67 | 0.17 | **0.83** | **0.83** | **0.83** | 0.17 | 0.50 | **0.83** | 0.50 | 0.33 | 0.00 | 0.33 | 0.67 | **1.00** | 0.67 |
| `bbpb_include_base_44` | 0.67 | 0.67 | 0.17 | 0.67 | **0.83** | 0.67 | 0.17 | 0.33 | 0.50 | 0.50 | 0.33 | 0.50 | 0.50 | 0.33 | **0.83** |
| `bbpb_global_mmlu_full` | 0.67 | 0.33 | 0.33 | **0.83** | **0.83** | 0.67 | 0.33 | 0.50 | 0.50 | 0.33 | 0.17 | 0.17 | 0.50 | 0.50 | **1.00** |
| `include_v2_en_malay_malaysia` | 0.33 | 0.50 | 0.67 | 0.67 | 0.33 | 0.17 | 0.33 | 0.00 | 0.67 | 0.50 | 0.67 | 0.50 | 0.67 | 0.67 | 0.33 |
| `global_mmlu_full` | 0.50 | 0.50 | 0.17 | 0.00 | 0.17 | 0.33 | 0.33 | 0.50 | 0.67 | 0.67 | 0.50 | 0.33 | **0.83** | 0.67 | **0.83** |
| `rf_belebele` | 0.33 | 0.67 | 0.33 | **1.00** | 0.17 | 0.33 | 0.33 | 0.33 | 0.50 | 0.33 | 0.67 | 0.50 | 0.33 | **0.83** | 0.17 |
| `bbpb_include_v2_en_malay_malaysia` | 0.33 | 0.67 | 0.17 | 0.67 | **0.83** | 0.67 | 0.50 | 0.33 | 0.17 | 0.50 | 0.33 | 0.50 | 0.17 | 0.33 | 0.50 |
| `rf_global_mmlu_full` | **0.83** | 0.50 | 0.50 | 0.17 | 0.00 | 0.67 | 0.33 | 0.33 | 0.17 | 0.67 | 0.33 | 0.50 | 0.33 | 0.50 | **0.83** |
| `global_piqa_parallel_cloze` | 0.00 | 0.67 | 0.67 | 0.33 | 0.33 | 0.33 | 0.00 | 0.50 | 0.50 | 0.50 | 0.67 | 0.67 | 0.17 | 0.17 | **1.00** |
| `rf_include_base_44` | 0.00 | 0.67 | 0.33 | 0.33 | 0.50 | 0.17 | 0.67 | 0.67 | 0.33 | 0.17 | 0.50 | 0.33 | 0.33 | 0.67 | 0.33 |
| `include_v2_og_malay_malaysia` | 0.17 | 0.50 | 0.50 | 0.33 | 0.33 | 0.17 | 0.17 | 0.67 | 0.50 | 0.50 | 0.50 | 0.50 | 0.00 | 0.67 | 0.33 |
| `include_v2_og_malay_singapore` | 0.50 | 0.33 | 0.50 | 0.17 | 0.50 | 0.50 | 0.33 | 0.17 | **1.00** | 0.50 | 0.00 | 0.50 | 0.17 | 0.33 | 0.17 |
| `rfgm_belebele` | 0.33 | 0.33 | 0.50 | 0.67 | 0.17 | 0.33 | 0.50 | 0.67 | 0.17 | 0.00 | 0.50 | 0.33 | 0.50 | 0.17 | 0.00 |
| `include_v2_en_malay_singapore` | 0.33 | 0.17 | 0.17 | 0.50 | 0.33 | 0.33 | 0.33 | 0.17 | 0.50 | 0.17 | 0.17 | 0.17 | 0.17 | 0.33 | 0.33 |

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
subtitle: "Malay (ms) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Nepali (ne) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_hellaswag` | **1.00** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | **1.00** |
| `bbpb_rf_belebele_npi_Deva` | **1.00** | 0.67 | **0.83** | **1.00** | **1.00** | 0.67 | **0.83** | **1.00** | **1.00** | **0.83** | 0.67 | 0.67 | **0.83** | **0.83** | **1.00** |
| `bbpb_rf_m_mmlu` | **1.00** | **1.00** | **1.00** | 0.67 | **0.83** | **1.00** | **1.00** | 0.67 | **0.83** | **1.00** | 0.67 | **0.83** | 0.67 | **0.83** | **0.83** |
| `bbpb_rf_include_base_44` | **1.00** | **1.00** | **1.00** | **0.83** | 0.67 | **1.00** | **1.00** | **0.83** | 0.67 | **1.00** | **0.83** | 0.67 | **0.83** | 0.67 | **0.83** |
| `bbpb_global_piqa_parallel_cloze` | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | 0.67 | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** |
| `bpb` | **1.00** | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** |
| `bbpb_rfgm_belebele_npi_Deva` | **1.00** | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** |
| `bbpb_rfgm_include_base_44` | 0.67 | **1.00** | **1.00** | **0.83** | **0.83** | 0.67 | 0.67 | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** |
| `bbpb_include_v2_og` | **0.83** | **1.00** | **0.83** | **1.00** | 0.67 | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** | 0.67 |
| `bbpb_rf_global_mmlu_full` | **1.00** | **1.00** | **0.83** | **0.83** | 0.67 | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | 0.67 | **0.83** | **0.83** |
| `bbpb_arc` | **0.83** | 0.67 | 0.67 | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | 0.67 | **0.83** | 0.67 | **0.83** | **0.83** |
| `bbpb_belebele_npi_Latn` | **1.00** | 0.67 | 0.33 | **0.83** | 0.50 | 0.67 | 0.33 | **0.83** | 0.50 | 0.33 | **0.83** | 0.50 | 0.50 | **0.83** | 0.67 |
| `rf_m_mmlu` | **0.83** | **1.00** | 0.17 | **1.00** | 0.50 | **0.83** | 0.33 | **0.83** | 0.33 | 0.17 | **1.00** | 0.50 | 0.17 | 0.33 | 0.50 |
| `bbpb_belebele_npi_Deva` | 0.50 | **0.83** | **0.83** | 0.67 | 0.50 | 0.67 | 0.33 | 0.50 | 0.00 | 0.67 | **0.83** | 0.33 | 0.50 | 0.67 | 0.50 |
| `include_base_44` | 0.50 | 0.67 | **0.83** | 0.33 | 0.33 | 0.50 | 0.33 | **0.83** | **0.83** | 0.50 | 0.67 | 0.67 | 0.17 | 0.17 | **1.00** |
| `global_mmlu_full` | 0.67 | 0.50 | 0.67 | 0.33 | 0.67 | 0.17 | 0.33 | 0.67 | 0.67 | **0.83** | 0.50 | 0.50 | 0.33 | 0.67 | 0.67 |
| `bbpb_rf_belebele_npi_Latn` | 0.17 | 0.17 | 0.50 | 0.00 | 0.33 | **1.00** | 0.33 | **0.83** | **0.83** | 0.33 | **0.83** | **0.83** | 0.50 | 0.50 | 0.67 |
| `rf_include_base_44` | 0.67 | **0.83** | **1.00** | 0.00 | 0.50 | 0.50 | 0.67 | 0.33 | 0.50 | **0.83** | 0.17 | 0.67 | 0.00 | 0.50 | 0.50 |
| `bbpb_include_v2_en` | 0.33 | 0.33 | 0.17 | 0.50 | 0.67 | **1.00** | 0.50 | **0.83** | 0.33 | 0.50 | **0.83** | 0.33 | 0.33 | 0.50 | 0.50 |
| `hellaswag` | **0.83** | **0.83** | 0.33 | 0.50 | 0.17 | 0.67 | 0.50 | 0.67 | 0.33 | 0.17 | 0.50 | 0.17 | 0.50 | 0.50 | 0.67 |
| `arc` | 0.67 | 0.50 | **0.83** | 0.33 | 0.00 | 0.50 | **0.83** | 0.67 | 0.33 | 0.67 | 0.17 | 0.50 | 0.50 | 0.17 | 0.67 |
| `bbpb_global_mmlu_full` | 0.67 | 0.67 | 0.50 | 0.17 | 0.17 | 0.67 | 0.50 | 0.50 | 0.50 | **0.83** | 0.17 | 0.17 | 0.33 | 0.33 | **1.00** |
| `rf_global_mmlu_full` | 0.67 | 0.33 | 0.33 | 0.50 | 0.33 | 0.50 | 0.17 | 0.33 | 0.17 | 0.33 | 0.50 | 0.33 | **0.83** | **1.00** | **0.83** |
| `belebele_npi_Deva` | **0.83** | 0.33 | **1.00** | 0.17 | 0.50 | 0.50 | **0.83** | 0.00 | 0.33 | 0.33 | 0.50 | 0.17 | 0.17 | 0.50 | 0.67 |
| `bbpb_include_base_44` | 0.17 | 0.00 | 0.67 | 0.33 | 0.17 | **0.83** | 0.17 | **0.83** | 0.67 | 0.33 | 0.67 | **0.83** | 0.00 | 0.50 | 0.50 |
| `rf_belebele_npi_Latn` | 0.50 | 0.17 | 0.33 | 0.33 | 0.33 | 0.33 | 0.50 | 0.17 | **0.83** | 0.50 | **0.83** | 0.50 | 0.67 | 0.33 | 0.33 |
| `belebele_npi_Latn` | **0.83** | 0.17 | 0.17 | 0.33 | 0.17 | 0.00 | 0.00 | 0.33 | 0.17 | **1.00** | 0.50 | 0.67 | 0.50 | 0.67 | **0.83** |
| `include_v2_og` | 0.00 | 0.67 | 0.67 | 0.17 | 0.17 | 0.33 | 0.33 | 0.67 | **0.83** | **1.00** | 0.00 | 0.50 | 0.00 | 0.50 | 0.50 |
| `global_piqa_parallel_cloze` | 0.50 | 0.50 | 0.67 | 0.33 | 0.17 | 0.50 | 0.33 | 0.50 | 0.67 | 0.50 | 0.50 | 0.33 | 0.00 | 0.00 | 0.67 |
| `bbpb_rfgm_belebele_npi_Latn` | 0.50 | 0.50 | 0.00 | 0.33 | **0.83** | 0.00 | 0.50 | 0.17 | 0.67 | 0.50 | **0.83** | 0.33 | 0.67 | 0.17 | 0.17 |
| `include_v2_en` | 0.33 | 0.33 | 0.33 | 0.67 | 0.67 | 0.67 | 0.33 | 0.67 | 0.33 | 0.17 | 0.67 | 0.00 | 0.00 | 0.67 | 0.33 |
| `rf_belebele_npi_Deva` | 0.50 | 0.50 | 0.33 | 0.50 | 0.67 | 0.33 | 0.67 | 0.33 | 0.17 | 0.50 | 0.33 | 0.50 | 0.00 | 0.00 | **0.83** |
| `rfgm_belebele_npi_Deva` | 0.50 | 0.33 | 0.17 | 0.50 | 0.17 | 0.17 | 0.50 | 0.67 | 0.50 | 0.33 | 0.17 | 0.33 | 0.67 | 0.67 | 0.33 |
| `rfgm_include_base_44` | 0.50 | 0.50 | 0.50 | 0.00 | 0.67 | 0.00 | 0.67 | 0.33 | 0.17 | 0.17 | 0.50 | **0.83** | 0.33 | 0.17 | 0.33 |
| `rfgm_belebele_npi_Latn` | 0.33 | **1.00** | 0.33 | 0.17 | 0.33 | 0.33 | 0.17 | 0.33 | **0.83** | 0.33 | 0.17 | 0.33 | 0.50 | 0.00 | 0.17 |

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
subtitle: "Nepali (ne) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Dutch (nl) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_hellaswag` | **0.91** | **0.82** | **0.80** | **0.82** | **0.87** | **0.92** | **0.89** | **0.88** | **0.86** | **0.88** | **0.92** | **0.82** | **0.88** | **0.78** | **0.81** |
| `bpb` | **0.97** | **0.86** | **0.76** | **0.90** | **0.90** | **0.89** | **0.79** | **0.90** | **0.90** | **0.85** | **0.88** | **0.82** | **0.79** | 0.72 | **0.85** |
| `bbpb_rfgm_belebele` | **0.86** | **0.88** | **0.79** | **0.84** | **0.86** | **0.86** | **0.83** | **0.82** | **0.78** | **0.79** | **0.86** | **0.88** | **0.78** | 0.72 | **0.80** |
| `hellaswag` | 0.65 | 0.67 | **0.79** | 0.69 | 0.73 | 0.69 | 0.73 | 0.66 | 0.72 | 0.68 | **0.80** | **0.77** | 0.64 | 0.67 | **0.87** |
| `bbpb_include_v2_og_dutch_netherlands` | 0.59 | 0.56 | 0.57 | 0.69 | 0.63 | 0.73 | **0.80** | **0.77** | 0.74 | 0.63 | 0.62 | 0.72 | 0.70 | **0.78** | 0.74 |
| `bbpb_include_v2_og_dutch_belgium` | 0.74 | **0.83** | 0.68 | 0.67 | 0.60 | 0.68 | 0.68 | 0.72 | 0.59 | 0.69 | 0.60 | 0.58 | 0.62 | 0.74 | 0.68 |
| `bbpb_rf_belebele` | 0.71 | 0.69 | 0.62 | 0.72 | 0.71 | **0.75** | 0.70 | 0.67 | 0.67 | 0.60 | 0.60 | 0.52 | 0.63 | 0.65 | 0.72 |
| `bbpb_rf_m_mmlu` | 0.72 | **0.80** | 0.71 | 0.55 | 0.62 | 0.72 | 0.72 | 0.64 | 0.62 | 0.68 | 0.52 | 0.67 | 0.59 | 0.74 | 0.57 |
| `bbpb_rf_global_mmlu_full` | 0.67 | 0.66 | 0.68 | 0.68 | 0.71 | **0.76** | 0.67 | **0.79** | 0.61 | 0.74 | 0.62 | 0.53 | 0.51 | 0.64 | 0.58 |
| `bbpb_multiblimp` | 0.66 | 0.64 | **0.75** | 0.63 | 0.73 | 0.63 | 0.64 | 0.69 | 0.56 | 0.59 | 0.51 | 0.57 | 0.68 | **0.75** | 0.68 |
| `bbpb_global_mmlu_full` | 0.74 | 0.57 | 0.64 | 0.72 | 0.60 | 0.62 | 0.62 | 0.66 | 0.57 | 0.72 | 0.62 | 0.62 | 0.71 | 0.66 | 0.62 |
| `multiblimp` | 0.56 | 0.60 | 0.68 | 0.63 | 0.51 | 0.61 | 0.59 | 0.68 | 0.62 | 0.62 | 0.62 | 0.66 | **0.76** | 0.68 | 0.62 |
| `bbpb_global_piqa_parallel_cloze` | **0.76** | 0.72 | 0.69 | 0.57 | 0.44 | **0.80** | **0.75** | 0.65 | 0.45 | **0.75** | 0.68 | 0.53 | 0.58 | 0.42 | 0.60 |
| `bbpb_include_v2_en_dutch_netherlands` | 0.62 | **0.76** | 0.61 | 0.54 | 0.66 | 0.59 | 0.59 | 0.66 | 0.34 | 0.62 | 0.60 | 0.58 | 0.60 | 0.50 | 0.45 |
| `bbpb_arc` | 0.48 | 0.64 | 0.50 | 0.61 | 0.53 | 0.61 | **0.75** | 0.66 | 0.52 | 0.56 | 0.60 | 0.64 | 0.57 | 0.52 | 0.49 |
| `rf_m_mmlu` | 0.53 | 0.50 | 0.65 | 0.54 | 0.53 | 0.56 | 0.54 | 0.62 | 0.57 | 0.37 | 0.72 | 0.57 | 0.57 | 0.72 | 0.60 |
| `bbpb_belebele` | 0.47 | 0.42 | 0.60 | 0.70 | 0.52 | 0.59 | 0.63 | 0.52 | 0.55 | 0.59 | 0.56 | 0.59 | 0.60 | 0.62 | 0.57 |
| `rfgm_belebele` | 0.42 | 0.53 | 0.60 | 0.56 | 0.58 | 0.57 | 0.58 | 0.51 | 0.59 | 0.58 | 0.57 | 0.55 | 0.48 | 0.50 | 0.62 |
| `rf_belebele` | 0.53 | 0.48 | 0.59 | 0.62 | 0.50 | 0.48 | 0.53 | 0.42 | 0.47 | 0.53 | 0.47 | 0.57 | 0.68 | 0.44 | 0.53 |
| `include_v2_og_dutch_netherlands` | 0.44 | 0.53 | 0.33 | 0.36 | 0.29 | 0.54 | 0.61 | 0.57 | 0.57 | 0.52 | 0.60 | 0.64 | 0.68 | 0.58 | 0.53 |
| `global_mmlu_full` | 0.58 | 0.57 | 0.47 | 0.50 | 0.48 | 0.46 | 0.53 | 0.58 | 0.48 | 0.44 | 0.48 | 0.42 | 0.55 | 0.50 | 0.69 |
| `belebele` | 0.38 | 0.42 | 0.43 | 0.62 | 0.52 | 0.57 | 0.51 | 0.38 | 0.56 | 0.53 | 0.54 | 0.55 | 0.53 | 0.58 | 0.59 |
| `rf_global_mmlu_full` | 0.53 | 0.44 | 0.50 | 0.48 | 0.47 | 0.40 | 0.57 | 0.37 | 0.60 | 0.42 | 0.56 | 0.58 | 0.48 | 0.55 | 0.63 |
| `bbpb_include_v2_en_dutch_belgium` | 0.38 | 0.62 | 0.55 | 0.47 | 0.58 | 0.50 | 0.53 | 0.57 | 0.38 | 0.54 | 0.46 | 0.39 | 0.62 | 0.47 | 0.48 |
| `arc` | 0.57 | 0.58 | 0.47 | 0.47 | 0.44 | 0.51 | 0.47 | 0.50 | 0.33 | 0.47 | 0.57 | 0.33 | 0.64 | 0.45 | 0.62 |
| `include_v2_en_dutch_netherlands` | 0.35 | 0.52 | 0.33 | 0.44 | 0.37 | 0.55 | 0.48 | 0.33 | 0.54 | 0.42 | 0.52 | 0.43 | 0.54 | 0.67 | 0.52 |
| `include_v2_og_dutch_belgium` | 0.53 | 0.52 | 0.28 | 0.42 | 0.48 | 0.33 | 0.47 | 0.50 | 0.27 | 0.48 | 0.33 | 0.57 | 0.40 | 0.38 | 0.51 |
| `include_v2_en_dutch_belgium` | 0.38 | 0.51 | 0.43 | 0.40 | 0.36 | 0.18 | 0.61 | 0.53 | 0.35 | 0.24 | 0.36 | 0.50 | 0.45 | 0.31 | 0.33 |
| `global_piqa_parallel_cloze` | 0.13 | 0.54 | 0.36 | 0.26 | 0.55 | 0.33 | 0.46 | 0.48 | 0.37 | 0.51 | 0.25 | 0.50 | 0.37 | 0.35 | 0.41 |

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
subtitle: "Dutch (nl) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Norwegian (Bokmål) (no) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **1.00** | **0.89** | **0.81** | **0.94** | **0.94** | **0.89** | **0.81** | **0.94** | **0.94** | 0.69 | **0.94** | **0.89** | **0.75** | **0.75** | **0.94** |
| `bbpb_noropenbookqa_nob_p0` | **0.89** | **0.75** | **0.81** | 0.72 | 0.72 | **0.81** | **0.92** | **0.83** | **0.83** | **0.83** | **0.75** | **0.92** | **0.86** | **0.86** | **0.83** |
| `bbpb_rfgm_belebele` | **0.94** | **0.81** | **0.75** | **0.86** | **0.89** | **0.86** | **0.81** | **0.86** | **0.89** | 0.72 | **0.83** | **0.81** | **0.78** | 0.69 | **0.75** |
| `bbpb_noropenbookqa_nno_p0` | **0.83** | **0.75** | **0.81** | **0.83** | **0.83** | **0.75** | **0.75** | 0.67 | **0.83** | **0.78** | **0.92** | **0.81** | **0.81** | 0.69 | **0.78** |
| `bbpb_nrk_quiz_qa_nno_p0` | **0.75** | **0.75** | 0.69 | **0.75** | 0.64 | **0.89** | 0.72 | **0.78** | **0.78** | 0.72 | **0.78** | 0.72 | **0.78** | 0.72 | **0.78** |
| `bbpb_ncb` | **0.92** | **0.81** | **0.81** | 0.58 | **0.75** | **0.83** | 0.72 | 0.61 | **0.78** | **0.78** | 0.56 | 0.72 | 0.56 | 0.72 | **0.83** |
| `bbpb_rf_m_mmlu` | 0.72 | **0.78** | 0.53 | **0.81** | 0.61 | **0.83** | 0.64 | **0.81** | 0.72 | 0.69 | **0.75** | **0.78** | 0.61 | **0.81** | **0.75** |
| `bbpb_norcommonsenseqa_nno_p0` | **0.75** | 0.69 | 0.64 | **0.83** | 0.64 | 0.61 | 0.72 | **0.81** | **0.83** | 0.67 | **0.75** | 0.72 | **0.81** | 0.61 | 0.69 |
| `bbpb_rf_belebele` | **0.83** | **0.81** | 0.47 | **0.86** | 0.67 | **0.86** | 0.58 | **0.86** | **0.78** | 0.56 | **0.78** | 0.69 | 0.56 | 0.53 | 0.69 |
| `bbpb_nrk_quiz_qa_nob_p0` | 0.53 | 0.61 | 0.58 | 0.69 | 0.69 | **0.92** | **0.78** | 0.67 | 0.61 | **0.86** | **0.75** | 0.64 | **0.78** | 0.61 | **0.78** |
| `norcommonsenseqa_nob_p0` | 0.56 | 0.69 | **0.78** | 0.69 | 0.69 | 0.72 | 0.67 | 0.58 | 0.67 | **0.78** | 0.69 | 0.72 | **0.75** | 0.72 | 0.67 |
| `bbpb_norcommonsenseqa_nob_p0` | 0.58 | **0.75** | 0.53 | 0.67 | 0.61 | **0.78** | **0.83** | **0.81** | 0.58 | 0.67 | 0.69 | **0.75** | **0.81** | 0.53 | 0.67 |
| `bbpb_global_piqa_parallel_cloze` | **0.92** | **0.92** | **0.81** | 0.64 | 0.50 | **0.89** | **0.78** | 0.56 | 0.53 | **0.78** | 0.67 | 0.53 | 0.67 | 0.47 | 0.58 |
| `bbpb_nortruthfulqa_mc_nob_p0` | **0.81** | **0.75** | 0.72 | 0.61 | **0.78** | **0.78** | **0.75** | 0.69 | 0.64 | 0.53 | 0.53 | 0.64 | **0.78** | 0.56 | 0.67 |
| `bbpb_belebele` | 0.67 | 0.61 | 0.58 | 0.64 | 0.67 | 0.67 | **0.81** | **0.75** | 0.67 | 0.69 | 0.64 | 0.50 | 0.56 | 0.58 | 0.58 |
| `bbpb_arc_mt` | 0.44 | 0.67 | 0.53 | 0.61 | 0.58 | 0.61 | 0.69 | 0.61 | 0.36 | **0.75** | **0.78** | 0.69 | **0.75** | 0.50 | **0.75** |
| `bbpb_norcommonsenseqa_nob_p3` | 0.39 | 0.47 | 0.47 | 0.42 | 0.67 | 0.64 | 0.69 | 0.58 | 0.67 | 0.72 | 0.67 | 0.64 | **0.78** | 0.69 | 0.53 |
| `ncb` | 0.72 | 0.72 | 0.42 | 0.64 | 0.64 | 0.69 | 0.56 | 0.53 | **0.75** | 0.53 | 0.42 | **0.78** | 0.42 | 0.58 | 0.44 |
| `bbpb_nrk_quiz_qa_nno_p1` | 0.47 | 0.39 | 0.44 | 0.39 | 0.44 | 0.58 | 0.58 | 0.58 | 0.53 | **0.78** | **0.78** | 0.61 | **0.83** | 0.61 | 0.61 |
| `bbpb_norec_document_p1` | 0.69 | 0.61 | 0.72 | 0.39 | 0.72 | 0.47 | 0.58 | 0.53 | 0.64 | 0.56 | 0.56 | 0.56 | 0.44 | 0.61 | 0.44 |
| `bbpb_noropenbookqa_nno_p4` | 0.53 | 0.53 | 0.47 | 0.42 | 0.58 | 0.67 | 0.67 | 0.56 | 0.56 | **0.83** | 0.39 | 0.72 | 0.39 | 0.56 | 0.61 |
| `nrk_quiz_qa_nno_p0` | 0.72 | 0.50 | 0.61 | 0.64 | 0.47 | 0.42 | 0.56 | 0.50 | 0.50 | 0.39 | 0.56 | 0.58 | **0.75** | 0.53 | 0.58 |
| `bbpb_norcommonsenseqa_nob_p2` | 0.56 | 0.47 | 0.25 | 0.36 | 0.61 | **0.81** | 0.53 | 0.58 | 0.61 | 0.67 | 0.67 | 0.58 | 0.67 | 0.36 | 0.58 |
| `bbpb_norbelebele_p1` | 0.36 | 0.72 | 0.69 | 0.42 | 0.44 | 0.64 | 0.61 | 0.50 | 0.58 | **0.81** | 0.53 | 0.61 | 0.33 | 0.64 | 0.42 |
| `bbpb_norec_document_p3` | 0.72 | 0.56 | 0.47 | 0.39 | 0.33 | 0.50 | 0.64 | 0.44 | 0.33 | 0.58 | 0.72 | 0.61 | 0.69 | 0.64 | 0.61 |
| `bbpb_norec_document_p2` | 0.61 | 0.53 | 0.69 | 0.64 | 0.69 | 0.64 | 0.36 | 0.47 | 0.69 | 0.44 | 0.22 | 0.67 | 0.72 | 0.50 | 0.33 |
| `bbpb_noropenbookqa_nno_p1` | 0.56 | 0.31 | **0.75** | 0.53 | 0.44 | 0.25 | 0.58 | 0.64 | 0.56 | 0.44 | 0.50 | 0.47 | 0.72 | 0.53 | **0.75** |
| `nortruthfulqa_mc_nob_p3` | 0.72 | 0.36 | 0.39 | 0.36 | 0.58 | 0.47 | 0.61 | 0.47 | 0.69 | 0.64 | 0.67 | 0.58 | 0.53 | 0.53 | 0.42 |
| `bbpb_nrk_quiz_qa_nob_p4` | 0.47 | 0.19 | 0.47 | 0.31 | 0.53 | 0.50 | 0.56 | 0.50 | 0.67 | 0.61 | 0.72 | 0.50 | 0.72 | 0.61 | 0.61 |
| `bbpb_nortruthfulqa_mc_nob_p3` | 0.39 | 0.25 | 0.44 | 0.47 | 0.58 | 0.58 | 0.67 | 0.53 | 0.69 | 0.42 | 0.50 | 0.39 | 0.58 | 0.69 | 0.72 |
| `bbpb_nortruthfulqa_mc_nno_p4` | 0.47 | 0.42 | 0.58 | 0.56 | 0.56 | **0.78** | 0.39 | 0.42 | **0.75** | 0.39 | 0.31 | 0.69 | 0.53 | 0.53 | 0.50 |
| `bbpb_nortruthfulqa_mc_nno_p2` | 0.53 | 0.42 | 0.36 | 0.47 | 0.50 | 0.72 | 0.67 | 0.44 | 0.53 | 0.56 | 0.50 | 0.47 | 0.39 | 0.64 | 0.64 |
| `bbpb_nortruthfulqa_mc_nno_p3` | 0.44 | 0.39 | 0.42 | 0.50 | 0.47 | 0.61 | 0.69 | 0.56 | 0.53 | 0.47 | 0.50 | 0.58 | 0.47 | 0.56 | 0.64 |
| `bbpb_norcommonsenseqa_nno_p2` | 0.72 | 0.33 | 0.44 | 0.72 | 0.53 | 0.39 | 0.33 | **0.78** | 0.69 | 0.50 | 0.56 | 0.47 | 0.28 | 0.47 | 0.58 |
| `belebele` | 0.36 | 0.47 | 0.58 | 0.61 | 0.58 | 0.50 | 0.58 | 0.33 | 0.50 | 0.50 | 0.50 | 0.72 | 0.50 | 0.44 | 0.61 |
| `noropenbookqa_nob_p0` | 0.39 | 0.53 | 0.42 | 0.42 | 0.56 | 0.53 | 0.39 | 0.47 | 0.64 | 0.44 | 0.53 | 0.50 | 0.72 | 0.61 | 0.64 |
| `bbpb_norec_sentence_p3` | 0.61 | 0.50 | 0.61 | 0.39 | 0.64 | 0.50 | 0.56 | 0.22 | 0.53 | 0.61 | 0.44 | 0.53 | 0.56 | 0.47 | 0.58 |
| `rfgm_belebele` | 0.72 | 0.53 | 0.58 | 0.56 | 0.47 | 0.39 | 0.53 | 0.47 | 0.47 | 0.42 | 0.58 | 0.64 | 0.61 | 0.33 | 0.44 |
| `bbpb_nortruthfulqa_mc_nob_p2` | 0.33 | 0.42 | 0.28 | 0.56 | 0.53 | 0.47 | **0.78** | 0.50 | 0.64 | 0.47 | 0.53 | 0.33 | 0.56 | **0.75** | 0.53 |
| `noropenbookqa_nob_p4` | 0.44 | 0.61 | 0.39 | 0.28 | 0.50 | 0.44 | 0.50 | 0.42 | 0.44 | 0.67 | 0.56 | 0.58 | 0.58 | 0.64 | 0.56 |
| `norcommonsenseqa_nob_p3` | 0.61 | 0.42 | 0.58 | 0.64 | 0.44 | 0.31 | 0.50 | 0.50 | 0.58 | 0.50 | 0.67 | 0.28 | 0.64 | 0.56 | 0.36 |
| `bbpb_norec_sentence_p2` | 0.50 | 0.47 | 0.44 | 0.44 | 0.53 | 0.31 | 0.33 | 0.33 | 0.47 | 0.64 | 0.64 | **0.78** | 0.56 | 0.47 | 0.64 |
| `nrk_quiz_qa_nno_p1` | 0.44 | 0.47 | 0.50 | 0.50 | 0.50 | 0.72 | 0.64 | 0.44 | 0.53 | 0.44 | 0.39 | 0.64 | 0.56 | 0.42 | 0.36 |
| `bbpb_noropenbookqa_nob_p1` | 0.64 | 0.25 | 0.50 | 0.33 | 0.69 | 0.44 | 0.47 | 0.53 | 0.67 | 0.58 | 0.53 | 0.28 | 0.50 | 0.47 | 0.64 |
| `arc_mt` | 0.67 | 0.58 | 0.47 | 0.47 | 0.58 | 0.44 | 0.42 | 0.44 | 0.64 | 0.36 | 0.72 | 0.36 | 0.39 | 0.61 | 0.36 |
| `bbpb_nrk_quiz_qa_nob_p1` | 0.47 | 0.44 | 0.47 | 0.33 | 0.56 | **0.75** | 0.67 | 0.42 | 0.58 | **0.75** | 0.33 | 0.61 | 0.47 | 0.47 | 0.17 |
| `norbelebele_p1` | 0.53 | **0.75** | 0.36 | 0.36 | **0.83** | 0.64 | 0.56 | 0.39 | 0.44 | 0.36 | 0.50 | 0.72 | 0.44 | 0.31 | 0.28 |
| `bbpb_norec_document_p4` | 0.67 | 0.36 | 0.56 | 0.64 | 0.56 | 0.25 | 0.61 | 0.53 | 0.50 | 0.36 | 0.44 | 0.58 | 0.64 | 0.39 | 0.36 |
| `rf_m_mmlu` | 0.58 | 0.47 | 0.53 | 0.33 | 0.42 | 0.64 | 0.42 | 0.47 | 0.33 | 0.44 | 0.44 | 0.53 | 0.61 | 0.69 | 0.53 |
| `bbpb_nortruthfulqa_mc_nno_p1` | 0.53 | 0.64 | 0.50 | 0.28 | 0.69 | 0.67 | 0.42 | 0.58 | 0.61 | 0.31 | 0.36 | **0.78** | 0.50 | 0.25 | 0.31 |
| `nortruthfulqa_mc_nob_p4` | **0.75** | **0.78** | 0.36 | 0.39 | 0.33 | **0.75** | 0.50 | 0.42 | 0.47 | 0.47 | 0.28 | 0.56 | 0.36 | 0.58 | 0.39 |
| `bbpb_noropenbookqa_nob_p4` | 0.53 | 0.39 | 0.31 | 0.39 | 0.56 | **0.75** | 0.44 | 0.58 | 0.36 | 0.64 | 0.50 | 0.50 | 0.58 | 0.42 | 0.33 |
| `bbpb_norec_document_p0` | 0.61 | 0.33 | 0.58 | 0.58 | 0.56 | 0.50 | 0.69 | 0.19 | 0.67 | 0.42 | 0.53 | 0.39 | 0.33 | 0.58 | 0.31 |
| `norec_sentence_p0` | 0.47 | 0.56 | 0.64 | 0.67 | 0.44 | 0.50 | 0.61 | 0.47 | 0.42 | 0.56 | 0.50 | 0.22 | 0.50 | 0.39 | 0.31 |
| `bbpb_nortruthfulqa_mc_nob_p4` | 0.47 | 0.64 | 0.47 | 0.39 | 0.69 | 0.56 | 0.28 | 0.42 | 0.61 | 0.33 | 0.42 | 0.56 | 0.42 | 0.61 | 0.36 |
| `bbpb_nortruthfulqa_mc_nob_p1` | 0.72 | 0.61 | 0.50 | 0.42 | 0.36 | 0.72 | 0.67 | 0.47 | 0.31 | 0.39 | 0.36 | 0.58 | 0.42 | 0.31 | 0.33 |
| `bbpb_norec_sentence_p1` | 0.58 | 0.53 | 0.42 | 0.17 | 0.67 | 0.39 | 0.61 | 0.31 | 0.64 | 0.56 | 0.47 | 0.42 | 0.47 | 0.53 | 0.39 |
| `bbpb_norec_sentence_p0` | **0.83** | 0.64 | 0.47 | 0.39 | 0.53 | 0.69 | 0.53 | 0.28 | 0.53 | 0.50 | 0.14 | 0.56 | 0.42 | 0.22 | 0.42 |
| `nortruthfulqa_mc_nob_p2` | 0.61 | 0.25 | 0.22 | 0.56 | 0.47 | 0.47 | 0.61 | 0.50 | 0.56 | **0.75** | 0.31 | 0.47 | 0.39 | 0.56 | 0.39 |
| `nrk_quiz_qa_nob_p0` | 0.36 | 0.53 | 0.33 | 0.44 | 0.58 | 0.36 | 0.50 | **0.86** | 0.36 | 0.53 | 0.33 | 0.47 | 0.53 | 0.47 | 0.44 |
| `norec_document_p4` | 0.56 | 0.42 | 0.64 | 0.58 | 0.56 | 0.22 | 0.39 | 0.39 | 0.44 | 0.50 | 0.50 | 0.58 | 0.56 | 0.42 | 0.31 |
| `bbpb_norec_sentence_p4` | 0.69 | 0.44 | 0.50 | 0.33 | 0.58 | 0.42 | 0.53 | 0.36 | 0.50 | 0.61 | 0.50 | 0.31 | 0.39 | 0.58 | 0.31 |
| `norec_sentence_p1` | 0.53 | 0.17 | 0.42 | 0.61 | 0.58 | 0.19 | 0.61 | 0.50 | 0.42 | 0.28 | 0.50 | 0.42 | 0.42 | **0.78** | 0.58 |
| `nortruthfulqa_mc_nob_p0` | 0.39 | 0.56 | 0.58 | 0.36 | 0.44 | 0.56 | 0.47 | 0.44 | 0.50 | 0.64 | 0.33 | **0.75** | 0.17 | 0.39 | 0.42 |
| `noropenbookqa_nob_p1` | 0.36 | 0.39 | 0.58 | 0.58 | 0.50 | 0.44 | 0.31 | 0.42 | 0.56 | 0.33 | 0.42 | 0.58 | 0.53 | 0.42 | 0.58 |
| `rf_belebele` | 0.58 | 0.36 | 0.72 | 0.44 | 0.44 | 0.33 | 0.58 | 0.28 | 0.47 | 0.42 | 0.58 | 0.39 | 0.39 | 0.31 | 0.67 |
| `nortruthfulqa_mc_nob_p1` | 0.56 | 0.44 | 0.22 | 0.44 | 0.42 | 0.50 | 0.44 | 0.61 | 0.56 | 0.56 | 0.56 | 0.28 | 0.28 | 0.56 | 0.50 |
| `nrk_quiz_qa_nob_p4` | 0.53 | 0.67 | 0.53 | 0.33 | 0.36 | 0.36 | 0.69 | 0.47 | 0.50 | 0.33 | 0.22 | 0.36 | 0.47 | 0.39 | 0.69 |
| `norec_document_p0` | 0.39 | 0.31 | 0.53 | 0.64 | 0.69 | 0.31 | 0.39 | 0.61 | 0.36 | 0.36 | 0.47 | 0.47 | 0.39 | 0.47 | 0.50 |
| `norcommonsenseqa_nob_p2` | 0.25 | 0.28 | 0.50 | 0.47 | 0.56 | 0.47 | 0.53 | 0.42 | 0.44 | 0.44 | 0.44 | 0.56 | 0.44 | 0.64 | 0.42 |
| `nrk_quiz_qa_nob_p1` | 0.53 | 0.44 | 0.69 | 0.50 | 0.39 | 0.50 | 0.50 | 0.36 | 0.47 | 0.31 | 0.36 | 0.44 | 0.47 | 0.53 | 0.36 |
| `global_piqa_parallel_cloze` | 0.36 | 0.47 | 0.33 | 0.58 | 0.47 | 0.56 | 0.31 | 0.39 | 0.31 | 0.31 | 0.58 | 0.67 | 0.42 | 0.42 | 0.67 |
| `noropenbookqa_nno_p0` | 0.64 | 0.53 | 0.39 | 0.28 | 0.61 | 0.42 | 0.39 | 0.44 | 0.47 | 0.44 | 0.17 | 0.56 | 0.28 | 0.64 | 0.42 |
| `norcommonsenseqa_nno_p2` | 0.50 | 0.56 | 0.50 | 0.31 | 0.58 | 0.36 | 0.53 | 0.56 | 0.39 | 0.61 | 0.36 | 0.36 | 0.39 | 0.19 | 0.33 |
| `nortruthfulqa_mc_nno_p4` | 0.61 | 0.33 | 0.47 | 0.53 | 0.47 | 0.67 | 0.44 | 0.31 | 0.44 | 0.42 | 0.11 | 0.42 | 0.28 | 0.36 | 0.58 |
| `norec_document_p1` | 0.44 | 0.36 | 0.61 | 0.42 | 0.47 | 0.56 | 0.28 | 0.61 | 0.31 | 0.28 | 0.67 | 0.22 | 0.22 | 0.67 | 0.22 |
| `nortruthfulqa_mc_nno_p2` | 0.44 | 0.36 | 0.42 | 0.47 | 0.47 | 0.36 | 0.36 | 0.28 | 0.33 | 0.39 | 0.33 | 0.53 | 0.42 | 0.47 | 0.61 |
| `nortruthfulqa_mc_nno_p3` | 0.50 | 0.50 | 0.25 | 0.36 | 0.36 | 0.56 | 0.39 | 0.36 | 0.44 | 0.42 | 0.28 | 0.44 | 0.61 | 0.50 | 0.22 |
| `norcommonsenseqa_nno_p0` | 0.42 | 0.28 | 0.42 | 0.47 | 0.42 | 0.14 | 0.47 | 0.44 | 0.36 | 0.53 | 0.44 | 0.44 | 0.56 | 0.22 | 0.56 |
| `noropenbookqa_nno_p1` | 0.42 | 0.31 | 0.64 | 0.61 | 0.39 | 0.72 | 0.25 | 0.31 | 0.28 | 0.11 | 0.25 | 0.28 | 0.72 | 0.47 | 0.31 |
| `nortruthfulqa_mc_nno_p1` | 0.31 | 0.33 | 0.47 | 0.19 | 0.44 | 0.39 | 0.44 | 0.47 | 0.56 | 0.53 | 0.36 | 0.36 | 0.31 | 0.47 | 0.33 |
| `norec_document_p3` | 0.44 | 0.28 | 0.42 | 0.53 | 0.42 | 0.33 | 0.33 | 0.56 | 0.33 | 0.25 | 0.47 | 0.50 | 0.36 | 0.22 | 0.42 |
| `noropenbookqa_nno_p4` | 0.53 | 0.28 | 0.53 | 0.50 | 0.47 | 0.33 | 0.31 | 0.47 | 0.39 | 0.39 | 0.11 | 0.44 | 0.56 | 0.22 | 0.25 |
| `norec_document_p2` | 0.11 | 0.42 | 0.17 | 0.25 | 0.25 | 0.31 | 0.58 | 0.42 | 0.53 | 0.25 | 0.39 | 0.39 | 0.50 | 0.58 | 0.58 |
| `norec_sentence_p2` | 0.53 | 0.25 | 0.28 | 0.44 | 0.31 | 0.44 | 0.33 | 0.69 | 0.36 | 0.33 | 0.22 | 0.36 | 0.19 | 0.31 | 0.39 |
| `norec_sentence_p4` | 0.64 | 0.31 | 0.25 | 0.33 | 0.19 | 0.33 | 0.44 | 0.44 | 0.31 | 0.33 | 0.61 | 0.14 | 0.19 | 0.33 | 0.33 |
| `norec_sentence_p3` | 0.56 | 0.11 | 0.17 | 0.31 | 0.36 | 0.33 | 0.19 | 0.53 | 0.44 | 0.42 | 0.56 | 0.17 | 0.11 | 0.58 | 0.08 |

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
subtitle: "Norwegian (Bokmål) (no) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Polish (pl) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.95** | **0.84** | **0.75** | **0.89** | **0.93** | **0.88** | **0.75** | **0.89** | **0.88** | **0.78** | **0.88** | **0.81** | **0.78** | 0.72 | **0.84** |
| `bbpb_rfgm_belebele` | **0.82** | **0.77** | 0.67 | 0.69 | **0.84** | **0.78** | **0.77** | 0.72 | 0.71 | 0.63 | **0.79** | 0.69 | 0.69 | 0.61 | 0.65 |
| `bbpb_rfgm_include_base_44` | 0.71 | 0.68 | 0.70 | 0.64 | **0.77** | **0.78** | 0.59 | 0.72 | 0.72 | 0.72 | 0.67 | 0.64 | 0.61 | 0.58 | **0.76** |
| `bbpb_include_base_44` | 0.71 | **0.75** | 0.72 | 0.72 | **0.76** | 0.64 | 0.63 | 0.67 | 0.58 | 0.53 | 0.69 | 0.68 | 0.67 | 0.65 | 0.73 |
| `bbpb_rf_global_mmlu_full` | 0.64 | 0.58 | 0.67 | 0.71 | **0.77** | 0.72 | 0.69 | **0.78** | 0.62 | 0.57 | 0.64 | 0.55 | 0.66 | **0.77** | 0.71 |
| `multiblimp` | 0.67 | 0.55 | 0.73 | 0.62 | 0.56 | 0.61 | 0.66 | 0.74 | 0.56 | 0.59 | **0.76** | 0.58 | 0.66 | 0.71 | 0.62 |
| `bbpb_rf_belebele` | **0.81** | 0.72 | **0.76** | 0.72 | 0.57 | 0.73 | 0.70 | 0.67 | 0.60 | 0.58 | 0.58 | 0.52 | 0.58 | 0.50 | 0.55 |
| `bbpb_arc_mt` | 0.62 | 0.63 | 0.54 | 0.68 | 0.65 | 0.63 | **0.88** | 0.67 | 0.50 | 0.66 | 0.50 | **0.75** | 0.62 | 0.56 | 0.57 |
| `bbpb_global_piqa_parallel_cloze` | **0.80** | 0.53 | 0.72 | 0.50 | 0.68 | 0.50 | 0.67 | 0.53 | 0.70 | 0.73 | 0.63 | 0.52 | 0.65 | 0.65 | 0.60 |
| `bbpb_global_mmlu_full` | 0.57 | 0.64 | 0.69 | 0.72 | 0.66 | 0.57 | 0.58 | 0.59 | 0.62 | 0.62 | 0.59 | 0.53 | 0.64 | 0.62 | 0.69 |
| `bbpb_include_v2_og` | 0.67 | 0.62 | 0.62 | 0.53 | 0.47 | 0.63 | 0.65 | 0.65 | 0.45 | 0.62 | 0.57 | 0.55 | 0.68 | 0.55 | 0.58 |
| `bbpb_multiblimp` | 0.56 | 0.54 | **0.75** | 0.52 | 0.53 | 0.60 | 0.54 | 0.66 | 0.52 | 0.59 | 0.41 | 0.65 | 0.57 | 0.66 | 0.57 |
| `bbpb_belebele` | 0.56 | 0.42 | 0.63 | 0.59 | 0.57 | 0.61 | 0.59 | 0.53 | 0.46 | 0.57 | 0.56 | 0.42 | 0.59 | 0.53 | 0.61 |
| `arc_mt` | 0.50 | 0.50 | 0.53 | 0.35 | 0.62 | 0.62 | 0.56 | 0.46 | 0.67 | 0.59 | 0.55 | 0.60 | 0.57 | 0.56 | 0.53 |
| `bbpb_rf_include_base_44` | 0.64 | 0.56 | 0.60 | 0.56 | 0.42 | 0.50 | 0.64 | 0.72 | 0.53 | 0.41 | 0.37 | 0.62 | 0.49 | 0.57 | 0.45 |
| `bbpb_include_v2_en` | 0.35 | 0.57 | 0.56 | 0.53 | 0.62 | 0.49 | 0.49 | 0.39 | 0.38 | 0.67 | 0.63 | 0.62 | 0.57 | 0.57 | 0.57 |
| `rf_belebele` | 0.41 | 0.58 | 0.69 | 0.61 | 0.42 | 0.41 | 0.49 | 0.54 | 0.65 | 0.48 | 0.46 | 0.41 | 0.62 | 0.43 | 0.55 |
| `include_v2_en` | 0.55 | 0.43 | 0.52 | 0.50 | 0.53 | 0.38 | 0.56 | 0.41 | 0.62 | 0.66 | 0.62 | 0.47 | 0.50 | 0.47 | 0.39 |
| `rfgm_include_base_44` | 0.41 | 0.44 | 0.51 | 0.55 | 0.43 | 0.45 | 0.62 | 0.43 | 0.47 | 0.54 | 0.61 | 0.55 | 0.47 | 0.53 | 0.57 |
| `belebele` | 0.62 | 0.53 | 0.53 | 0.50 | 0.52 | 0.42 | 0.35 | 0.40 | 0.47 | 0.55 | 0.40 | 0.36 | 0.58 | 0.47 | **0.82** |
| `global_mmlu_full` | 0.68 | 0.51 | 0.45 | 0.54 | 0.42 | 0.45 | 0.47 | 0.53 | 0.30 | 0.42 | 0.45 | 0.47 | 0.61 | 0.62 | 0.54 |
| `rf_global_mmlu_full` | 0.47 | 0.43 | 0.35 | 0.39 | 0.40 | 0.44 | 0.47 | 0.53 | 0.48 | 0.71 | 0.60 | 0.38 | 0.53 | 0.51 | 0.56 |
| `rf_include_base_44` | 0.47 | 0.30 | 0.54 | 0.49 | 0.44 | 0.46 | 0.42 | 0.66 | 0.57 | 0.53 | 0.52 | 0.44 | 0.38 | 0.34 | 0.67 |
| `rfgm_belebele` | 0.50 | 0.62 | 0.53 | 0.36 | 0.44 | 0.68 | 0.45 | 0.40 | 0.46 | 0.44 | 0.38 | 0.48 | 0.38 | 0.68 | 0.35 |
| `include_v2_og` | 0.48 | 0.59 | 0.52 | 0.51 | 0.36 | 0.53 | 0.37 | 0.56 | 0.50 | 0.40 | 0.64 | 0.33 | 0.48 | 0.42 | 0.43 |
| `include_base_44` | 0.50 | 0.60 | 0.40 | 0.31 | 0.51 | 0.47 | 0.39 | 0.50 | 0.39 | 0.46 | 0.50 | 0.41 | 0.51 | 0.41 | 0.44 |
| `global_piqa_parallel_cloze` | 0.39 | 0.38 | 0.45 | 0.41 | 0.37 | 0.54 | 0.24 | 0.50 | 0.53 | 0.41 | 0.47 | 0.42 | 0.46 | 0.35 | 0.44 |

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
subtitle: "Polish (pl) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Portuguese (pt) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.97** | **0.88** | **0.81** | **0.92** | **0.94** | **0.91** | **0.82** | **0.91** | **0.93** | **0.84** | **0.90** | **0.84** | **0.81** | **0.77** | **0.89** |
| `bbpb_hellaswag` | **0.94** | **0.83** | **0.78** | **0.82** | **0.85** | **0.88** | **0.82** | **0.86** | **0.86** | **0.93** | **0.83** | **0.77** | **0.78** | 0.72 | **0.80** |
| `hellaswag` | **0.78** | 0.66 | 0.68 | 0.72 | **0.78** | **0.80** | **0.78** | **0.81** | **0.84** | **0.80** | **0.78** | 0.74 | **0.80** | **0.78** | **0.90** |
| `bbpb_rf_belebele` | **0.78** | **0.75** | 0.67 | 0.68 | 0.58 | 0.74 | 0.66 | **0.79** | 0.64 | 0.57 | 0.67 | 0.67 | 0.63 | 0.58 | 0.65 |
| `bbpb_rf_global_mmlu_full` | 0.64 | 0.53 | 0.63 | **0.78** | 0.68 | 0.68 | 0.69 | **0.78** | 0.68 | 0.64 | 0.56 | 0.64 | 0.65 | 0.72 | 0.70 |
| `bbpb_rfgm_belebele` | **0.84** | **0.81** | 0.71 | 0.65 | 0.65 | 0.72 | 0.72 | 0.68 | 0.61 | 0.72 | 0.53 | 0.62 | 0.68 | 0.46 | 0.53 |
| `bbpb_rf_m_mmlu` | 0.69 | 0.73 | 0.70 | 0.68 | 0.62 | 0.59 | 0.66 | 0.69 | 0.59 | 0.63 | 0.57 | 0.57 | 0.60 | **0.78** | 0.62 |
| `bbpb_rf_mmmlu` | 0.68 | **0.77** | 0.70 | 0.63 | 0.61 | 0.58 | 0.67 | 0.70 | 0.57 | 0.62 | 0.57 | 0.54 | 0.62 | **0.79** | 0.62 |
| `bbpb_arc_mt` | 0.62 | 0.67 | 0.74 | 0.62 | 0.65 | 0.64 | 0.72 | 0.70 | 0.64 | 0.69 | 0.57 | 0.68 | 0.57 | 0.62 | 0.51 |
| `bbpb_arc` | 0.62 | 0.67 | 0.74 | 0.61 | 0.67 | 0.65 | 0.72 | 0.62 | 0.70 | 0.68 | 0.53 | 0.70 | 0.60 | 0.62 | 0.49 |
| `bbpb_global_piqa_parallel_cloze_por_latn_port` | **0.82** | **0.78** | 0.67 | 0.57 | 0.57 | **0.83** | 0.64 | 0.54 | 0.51 | 0.64 | 0.59 | 0.57 | 0.57 | 0.48 | **0.77** |
| `bbpb_global_piqa_parallel_cloze_por_latn_braz` | **0.79** | 0.74 | 0.68 | 0.53 | 0.66 | **0.80** | 0.68 | 0.59 | 0.58 | 0.62 | 0.56 | 0.57 | 0.57 | 0.47 | 0.69 |
| `bbpb_include_v2_og_portuguese_portugal` | 0.72 | 0.74 | 0.72 | 0.57 | 0.53 | 0.63 | 0.66 | 0.68 | 0.53 | 0.68 | 0.62 | 0.56 | 0.50 | 0.65 | 0.58 |
| `bbpb_rf_include_base_44` | 0.43 | 0.74 | 0.61 | 0.49 | 0.52 | 0.54 | 0.71 | 0.53 | 0.70 | 0.68 | 0.63 | 0.64 | 0.60 | **0.81** | 0.49 |
| `bbpb_global_mmlu_full` | 0.58 | 0.61 | 0.64 | 0.70 | 0.67 | 0.57 | 0.47 | 0.47 | 0.52 | 0.55 | 0.68 | 0.54 | **0.76** | 0.64 | 0.65 |
| `bbpb_belebele` | 0.62 | 0.49 | 0.66 | 0.72 | 0.59 | 0.66 | 0.68 | 0.57 | 0.49 | 0.63 | 0.49 | 0.47 | 0.62 | 0.60 | 0.64 |
| `rf_global_mmlu_full` | 0.61 | 0.57 | 0.53 | 0.73 | 0.55 | 0.48 | 0.57 | 0.62 | 0.68 | 0.55 | 0.54 | 0.67 | 0.66 | 0.56 | 0.57 |
| `multiblimp` | 0.47 | 0.60 | 0.68 | 0.64 | 0.62 | 0.39 | 0.55 | 0.52 | 0.52 | 0.60 | 0.57 | 0.61 | 0.68 | **0.75** | 0.67 |
| `bbpb_include_base_44` | **0.76** | 0.51 | 0.67 | 0.50 | 0.67 | 0.48 | 0.62 | 0.46 | 0.53 | 0.54 | 0.68 | 0.54 | 0.63 | 0.57 | 0.55 |
| `bbpb_rfgm_include_base_44` | 0.70 | 0.67 | 0.60 | 0.60 | 0.68 | 0.63 | 0.50 | 0.70 | 0.57 | 0.47 | 0.48 | 0.56 | 0.43 | 0.53 | 0.49 |
| `bbpb_include_v2_og_portuguese_brazil` | 0.55 | 0.68 | 0.59 | 0.38 | 0.52 | 0.56 | 0.64 | 0.45 | 0.62 | 0.62 | 0.49 | 0.66 | 0.57 | 0.68 | 0.52 |
| `arc` | 0.51 | 0.43 | 0.55 | 0.48 | 0.65 | 0.54 | 0.62 | 0.54 | 0.57 | 0.57 | 0.50 | 0.50 | 0.53 | 0.64 | 0.69 |
| `rf_m_mmlu` | 0.53 | 0.42 | 0.47 | 0.57 | 0.41 | 0.50 | 0.53 | 0.70 | 0.55 | 0.52 | 0.46 | 0.66 | 0.69 | 0.62 | 0.68 |
| `rf_mmmlu` | 0.50 | 0.54 | 0.42 | 0.64 | 0.56 | 0.45 | 0.47 | 0.60 | 0.56 | 0.57 | 0.53 | 0.49 | 0.61 | 0.65 | 0.68 |
| `bbpb_include_v2_en_portuguese_portugal` | 0.47 | 0.57 | 0.52 | 0.52 | 0.67 | 0.50 | 0.48 | 0.37 | 0.35 | 0.70 | 0.65 | 0.68 | 0.48 | 0.57 | 0.55 |
| `rf_belebele` | 0.49 | 0.46 | 0.47 | 0.47 | 0.37 | 0.62 | 0.55 | 0.47 | 0.38 | **0.78** | 0.60 | 0.52 | 0.65 | 0.54 | 0.63 |
| `rf_include_base_44` | 0.54 | 0.22 | 0.50 | 0.32 | 0.50 | 0.48 | 0.72 | 0.55 | 0.67 | 0.54 | 0.62 | 0.48 | 0.41 | 0.53 | 0.60 |
| `bbpb_multiblimp` | 0.52 | 0.47 | 0.56 | 0.42 | 0.53 | 0.42 | 0.39 | 0.53 | 0.40 | 0.61 | 0.41 | 0.60 | 0.45 | **0.79** | 0.54 |
| `global_mmlu_full` | 0.58 | 0.53 | 0.47 | 0.57 | 0.37 | 0.53 | 0.54 | 0.39 | 0.43 | 0.61 | 0.44 | 0.55 | 0.45 | 0.49 | 0.57 |
| `rfgm_belebele` | 0.42 | 0.48 | 0.35 | 0.35 | 0.42 | 0.50 | 0.64 | 0.55 | 0.54 | 0.42 | 0.40 | 0.45 | 0.64 | 0.65 | 0.64 |
| `include_v2_og_portuguese_portugal` | 0.42 | 0.32 | 0.48 | 0.42 | 0.48 | 0.56 | 0.38 | 0.46 | 0.38 | 0.55 | 0.61 | 0.57 | 0.52 | 0.67 | 0.57 |
| `include_v2_en_portuguese_portugal` | 0.36 | 0.49 | 0.45 | 0.55 | 0.41 | 0.64 | 0.53 | 0.43 | 0.49 | 0.56 | 0.47 | 0.39 | 0.62 | 0.45 | 0.39 |
| `rfgm_include_base_44` | 0.45 | 0.43 | 0.50 | 0.42 | 0.34 | 0.54 | 0.47 | 0.58 | 0.44 | 0.51 | 0.44 | 0.55 | 0.42 | 0.64 | 0.45 |
| `include_base_44` | 0.46 | 0.74 | 0.38 | 0.57 | 0.33 | 0.40 | 0.62 | 0.57 | 0.51 | 0.38 | 0.47 | 0.39 | 0.55 | 0.40 | 0.40 |
| `global_piqa_parallel_cloze_por_latn_braz` | 0.58 | 0.58 | 0.55 | 0.49 | 0.43 | 0.53 | 0.46 | 0.39 | 0.46 | 0.48 | 0.44 | 0.38 | 0.48 | 0.52 | 0.33 |
| `arc_mt` | 0.32 | 0.33 | 0.47 | 0.36 | 0.42 | 0.50 | 0.33 | 0.48 | 0.38 | 0.53 | 0.52 | 0.53 | 0.63 | 0.65 | 0.66 |
| `bbpb_include_v2_en_portuguese_brazil` | 0.50 | 0.42 | 0.48 | 0.31 | 0.43 | 0.45 | 0.43 | 0.42 | 0.50 | 0.68 | 0.57 | 0.45 | 0.49 | 0.42 | 0.47 |
| `xwinograd` | 0.53 | 0.38 | 0.54 | 0.41 | 0.42 | 0.38 | 0.55 | 0.41 | 0.54 | 0.46 | 0.47 | 0.45 | 0.52 | 0.53 | 0.40 |
| `include_v2_og_portuguese_brazil` | 0.40 | 0.44 | 0.31 | 0.31 | 0.21 | 0.37 | 0.50 | 0.51 | 0.47 | 0.39 | 0.39 | 0.49 | 0.65 | 0.53 | 0.63 |
| `global_piqa_parallel_cloze_por_latn_port` | 0.53 | 0.48 | 0.39 | 0.24 | 0.50 | 0.41 | 0.48 | 0.46 | 0.41 | 0.33 | 0.31 | 0.42 | 0.40 | 0.43 | 0.38 |
| `belebele` | 0.27 | 0.36 | 0.31 | 0.33 | 0.33 | 0.57 | 0.57 | 0.28 | 0.24 | 0.51 | 0.47 | 0.35 | 0.47 | 0.42 | 0.66 |
| `include_v2_en_portuguese_brazil` | 0.54 | 0.28 | 0.34 | 0.47 | 0.48 | 0.23 | 0.33 | 0.38 | 0.34 | 0.37 | 0.60 | 0.44 | 0.27 | 0.58 | 0.42 |

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
subtitle: "Romanian (ro) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **1.00** | **0.93** | **0.78** | **0.95** | **0.93** | **0.93** | **0.78** | **0.95** | **0.93** | **0.78** | **0.87** | **0.89** | 0.73 | 0.75 | **0.91** |
| `bbpb_hellaswag` | **0.98** | **0.84** | **0.87** | **0.84** | **0.89** | **0.85** | **0.89** | **0.82** | **0.87** | **0.93** | **0.85** | **0.80** | **0.82** | **0.76** | **0.87** |
| `bbpb_rfgm_belebele` | **0.85** | **0.82** | 0.64 | **0.84** | **0.80** | **0.82** | 0.67 | **0.87** | 0.73 | 0.75 | **0.80** | 0.69 | 0.69 | 0.51 | 0.67 |
| `bbpb_global_piqa_parallel_cloze` | **0.91** | **0.82** | **0.78** | 0.65 | 0.64 | **0.87** | **0.76** | 0.75 | 0.62 | 0.64 | 0.69 | 0.60 | **0.80** | 0.56 | 0.55 |
| `hellaswag` | **0.76** | 0.58 | 0.58 | 0.71 | 0.71 | 0.65 | 0.65 | 0.67 | 0.65 | 0.75 | **0.84** | 0.73 | 0.73 | 0.65 | **0.85** |
| `bbpb_multiblimp` | **0.82** | **0.76** | **0.84** | **0.80** | 0.60 | **0.76** | **0.80** | 0.73 | 0.53 | 0.75 | 0.64 | 0.47 | 0.75 | 0.62 | 0.55 |
| `bbpb_global_mmlu_full` | **0.80** | 0.53 | 0.55 | 0.53 | 0.62 | 0.62 | 0.56 | 0.51 | 0.64 | 0.73 | 0.71 | 0.69 | 0.73 | 0.71 | **0.80** |
| `multiblimp` | 0.73 | **0.82** | 0.67 | 0.65 | 0.56 | **0.80** | 0.60 | 0.69 | 0.47 | 0.71 | 0.69 | 0.53 | 0.65 | 0.55 | 0.58 |
| `bbpb_rf_belebele` | **0.76** | **0.82** | 0.67 | 0.67 | 0.56 | 0.69 | **0.80** | 0.69 | 0.55 | 0.67 | 0.67 | 0.53 | 0.64 | 0.42 | 0.49 |
| `bbpb_rf_m_mmlu` | 0.62 | 0.67 | 0.53 | **0.78** | 0.55 | 0.47 | **0.76** | 0.65 | 0.56 | 0.49 | 0.64 | 0.58 | 0.64 | **0.80** | 0.62 |
| `bbpb_rf_global_mmlu_full` | **0.76** | 0.64 | 0.36 | **0.76** | 0.58 | 0.65 | 0.53 | **0.78** | 0.45 | 0.62 | 0.55 | 0.36 | 0.38 | 0.38 | 0.53 |
| `bbpb_belebele` | 0.65 | 0.38 | 0.64 | 0.65 | 0.65 | 0.58 | 0.55 | 0.53 | 0.56 | 0.45 | 0.44 | 0.36 | 0.58 | 0.73 | 0.49 |
| `bbpb_arc` | 0.62 | 0.62 | 0.53 | 0.47 | 0.56 | 0.60 | 0.69 | 0.56 | 0.29 | 0.58 | 0.71 | 0.55 | 0.62 | 0.31 | 0.47 |
| `arc` | 0.62 | 0.55 | 0.49 | 0.67 | 0.53 | **0.76** | 0.47 | 0.42 | 0.42 | 0.58 | 0.35 | 0.53 | 0.45 | 0.56 | 0.38 |
| `belebele` | 0.31 | 0.42 | 0.49 | 0.73 | 0.44 | 0.36 | 0.51 | 0.49 | 0.58 | 0.51 | 0.45 | 0.51 | 0.40 | 0.62 | 0.47 |
| `global_mmlu_full` | 0.40 | 0.62 | 0.25 | 0.51 | 0.40 | 0.56 | 0.56 | 0.42 | 0.53 | 0.53 | 0.45 | 0.45 | 0.56 | 0.53 | 0.45 |
| `rf_m_mmlu` | 0.49 | 0.71 | 0.33 | 0.44 | 0.42 | 0.31 | 0.73 | 0.47 | 0.42 | 0.22 | 0.38 | 0.45 | 0.60 | 0.65 | 0.44 |
| `rfgm_belebele` | 0.36 | 0.35 | 0.65 | 0.53 | 0.65 | 0.25 | 0.42 | 0.45 | 0.29 | 0.55 | 0.51 | 0.47 | 0.53 | 0.55 | 0.42 |
| `rf_global_mmlu_full` | 0.55 | 0.47 | 0.31 | 0.38 | 0.27 | 0.24 | 0.45 | 0.44 | 0.55 | 0.42 | 0.51 | 0.53 | 0.62 | 0.53 | 0.38 |
| `rf_belebele` | 0.44 | 0.24 | 0.29 | 0.36 | 0.56 | 0.27 | 0.65 | 0.29 | 0.38 | 0.51 | 0.67 | 0.38 | 0.45 | 0.40 | 0.62 |
| `global_piqa_parallel_cloze` | 0.49 | 0.51 | 0.44 | 0.27 | 0.51 | 0.24 | 0.33 | 0.35 | 0.25 | 0.53 | 0.33 | 0.49 | 0.67 | 0.49 | 0.38 |

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
subtitle: "Romanian (ro) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Russian (ru) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.98** | **0.90** | **0.89** | **0.95** | **0.96** | **0.91** | **0.89** | **0.94** | **0.94** | **0.91** | **0.92** | **0.90** | **0.90** | **0.88** | **0.95** |
| `bbpb_xstorycloze` | **0.91** | **0.84** | **0.90** | **0.92** | **0.91** | **0.88** | **0.92** | **0.93** | **0.91** | **0.86** | **0.88** | **0.86** | **0.92** | **0.91** | **0.89** |
| `bbpb_hellaswag` | **0.97** | **0.90** | **0.90** | **0.91** | **0.89** | **0.93** | **0.90** | **0.92** | **0.87** | **0.90** | **0.89** | **0.85** | **0.88** | **0.86** | **0.85** |
| `hellaswag` | **0.88** | **0.82** | **0.87** | **0.86** | **0.80** | **0.85** | **0.90** | **0.88** | **0.84** | **0.89** | **0.78** | **0.84** | **0.85** | **0.82** | **0.82** |
| `bbpb_rfgm_include_base_44` | **0.89** | **0.79** | **0.83** | **0.91** | **0.81** | **0.82** | **0.83** | **0.94** | **0.80** | **0.83** | **0.84** | **0.77** | **0.85** | **0.80** | **0.80** |
| `bbpb_rfgm_belebele` | **0.92** | **0.90** | **0.85** | **0.84** | **0.80** | **0.91** | **0.83** | **0.82** | **0.77** | **0.82** | **0.79** | **0.79** | **0.85** | **0.82** | **0.79** |
| `bbpb_rf_belebele` | **0.86** | **0.83** | **0.85** | **0.76** | 0.74 | **0.81** | **0.82** | 0.73 | 0.73 | **0.78** | 0.72 | **0.76** | **0.82** | 0.75 | 0.70 |
| `bbpb_xnli` | **0.78** | **0.82** | **0.83** | 0.75 | 0.69 | **0.78** | **0.81** | **0.76** | 0.69 | **0.81** | **0.78** | **0.76** | **0.79** | 0.74 | 0.68 |
| `bbpb_multiblimp` | **0.75** | **0.82** | **0.88** | **0.79** | 0.67 | **0.80** | **0.78** | **0.83** | 0.65 | **0.78** | **0.82** | 0.70 | **0.82** | 0.67 | 0.64 |
| `multiblimp` | **0.76** | **0.75** | **0.82** | 0.73 | 0.67 | **0.79** | **0.81** | 0.75 | 0.66 | **0.81** | 0.72 | **0.76** | 0.74 | 0.70 | 0.67 |
| `bbpb_global_piqa_parallel_cloze` | **0.87** | **0.79** | **0.76** | 0.69 | 0.64 | **0.87** | **0.76** | 0.71 | 0.62 | **0.79** | **0.76** | 0.64 | **0.76** | 0.65 | 0.69 |
| `include_v2_og` | 0.65 | 0.66 | 0.66 | 0.72 | 0.70 | 0.71 | **0.77** | **0.80** | **0.78** | **0.76** | 0.73 | **0.77** | 0.75 | 0.71 | **0.81** |
| `bbpb_rf_global_mmlu_full` | **0.78** | 0.73 | **0.77** | 0.71 | **0.76** | 0.71 | **0.77** | **0.82** | 0.68 | **0.77** | 0.60 | 0.74 | 0.68 | 0.73 | 0.65 |
| `bbpb_rf_include_base_44` | **0.81** | 0.74 | 0.64 | 0.74 | 0.67 | **0.79** | 0.66 | **0.84** | **0.76** | 0.61 | 0.75 | 0.74 | 0.67 | 0.75 | 0.70 |
| `xstorycloze` | 0.70 | **0.80** | 0.73 | 0.72 | **0.76** | 0.69 | 0.66 | 0.64 | **0.78** | 0.71 | 0.71 | 0.72 | 0.74 | **0.77** | 0.68 |
| `bbpb_rf_m_mmlu` | 0.70 | 0.73 | 0.70 | 0.70 | 0.70 | 0.74 | 0.74 | 0.73 | 0.59 | 0.62 | 0.72 | 0.64 | **0.76** | 0.75 | 0.75 |
| `bbpb_include_v2_og` | 0.65 | **0.77** | 0.68 | 0.73 | 0.71 | 0.69 | 0.72 | 0.66 | 0.62 | 0.58 | **0.79** | 0.70 | 0.64 | 0.68 | 0.67 |
| `bbpb_arc` | 0.64 | 0.68 | 0.62 | 0.58 | 0.61 | 0.74 | 0.71 | 0.65 | 0.70 | 0.74 | 0.62 | 0.72 | 0.64 | 0.57 | 0.60 |
| `rf_global_mmlu_full` | 0.53 | 0.46 | 0.53 | 0.60 | 0.50 | 0.50 | 0.57 | 0.64 | 0.64 | **0.77** | 0.70 | 0.73 | **0.79** | 0.71 | **0.82** |
| `rfgm_belebele` | 0.69 | 0.61 | 0.64 | 0.60 | 0.57 | 0.68 | 0.68 | 0.57 | 0.67 | 0.61 | 0.63 | 0.62 | 0.69 | 0.62 | 0.58 |
| `bbpb_belebele` | 0.67 | 0.65 | 0.66 | 0.66 | 0.56 | 0.64 | 0.68 | 0.61 | 0.61 | 0.55 | 0.55 | 0.54 | 0.70 | 0.65 | 0.51 |
| `rf_m_mmlu` | 0.52 | 0.60 | 0.50 | 0.63 | 0.54 | 0.54 | 0.56 | 0.59 | 0.57 | 0.53 | 0.65 | 0.67 | 0.67 | 0.68 | **0.75** |
| `bbpb_rf_cultural_bench_easy` | 0.62 | 0.62 | 0.63 | 0.62 | 0.55 | 0.63 | 0.58 | 0.50 | 0.58 | 0.53 | 0.63 | 0.54 | 0.66 | 0.65 | 0.62 |
| `bbpb_global_mmlu_full` | 0.70 | 0.61 | 0.69 | 0.58 | 0.61 | 0.64 | 0.58 | 0.54 | 0.61 | 0.51 | 0.51 | 0.55 | 0.57 | 0.57 | 0.61 |
| `rfgm_include_base_44` | 0.48 | 0.54 | 0.58 | 0.56 | 0.56 | 0.51 | 0.57 | 0.56 | 0.59 | 0.69 | 0.67 | 0.66 | 0.59 | 0.64 | 0.62 |
| `rf_belebele` | 0.55 | 0.54 | 0.52 | 0.53 | 0.66 | 0.51 | 0.58 | 0.57 | 0.57 | 0.50 | 0.65 | 0.53 | 0.61 | 0.62 | 0.62 |
| `rf_include_base_44` | 0.55 | 0.46 | 0.50 | 0.61 | 0.63 | 0.54 | 0.63 | 0.60 | 0.47 | 0.55 | 0.48 | 0.54 | 0.56 | 0.57 | 0.66 |
| `xnli` | 0.56 | 0.49 | 0.61 | 0.52 | 0.58 | 0.64 | 0.53 | 0.50 | 0.61 | 0.50 | 0.52 | 0.43 | 0.60 | 0.72 | 0.52 |
| `bbpb_include_base_44` | 0.68 | 0.57 | 0.58 | 0.47 | 0.64 | 0.56 | 0.58 | 0.50 | 0.56 | 0.48 | 0.56 | 0.55 | 0.49 | 0.56 | 0.51 |
| `include_v2_en` | 0.53 | 0.51 | 0.54 | 0.62 | 0.50 | 0.50 | 0.50 | 0.47 | 0.57 | 0.61 | 0.64 | 0.52 | 0.58 | 0.55 | 0.57 |
| `bbpb_include_v2_en` | 0.42 | 0.58 | 0.59 | 0.54 | 0.67 | 0.57 | 0.48 | 0.49 | 0.43 | 0.45 | 0.58 | 0.64 | 0.49 | 0.43 | 0.69 |
| `bbpb_cultural_bench_easy` | 0.40 | 0.48 | 0.52 | 0.53 | 0.57 | 0.63 | 0.49 | 0.61 | 0.54 | 0.55 | 0.59 | 0.56 | 0.48 | 0.55 | 0.53 |
| `arc` | 0.56 | 0.35 | 0.56 | 0.64 | 0.69 | 0.41 | 0.56 | 0.47 | 0.50 | 0.55 | 0.47 | 0.50 | 0.54 | 0.59 | 0.61 |
| `global_mmlu_full` | 0.43 | 0.50 | 0.46 | 0.53 | 0.49 | 0.49 | 0.54 | 0.42 | 0.55 | 0.51 | 0.41 | 0.45 | 0.53 | 0.46 | 0.58 |
| `include_base_44` | 0.47 | 0.57 | 0.42 | 0.57 | 0.39 | 0.32 | 0.48 | 0.46 | 0.50 | 0.53 | 0.53 | 0.49 | 0.53 | 0.59 | 0.43 |
| `xwinograd` | 0.51 | 0.39 | 0.49 | 0.38 | 0.36 | 0.37 | 0.61 | 0.47 | 0.46 | 0.50 | 0.48 | 0.57 | 0.44 | 0.54 | 0.58 |
| `belebele` | 0.43 | 0.36 | 0.42 | 0.38 | 0.43 | 0.54 | 0.43 | 0.54 | 0.54 | 0.47 | 0.50 | 0.43 | 0.59 | 0.57 | 0.48 |
| `cultural_bench_easy` | 0.38 | 0.47 | 0.40 | 0.45 | 0.43 | 0.56 | 0.49 | 0.37 | 0.56 | 0.49 | 0.40 | 0.36 | 0.39 | 0.49 | 0.52 |
| `cultural_bench_hard` | 0.45 | 0.40 | 0.57 | 0.33 | 0.56 | 0.45 | 0.49 | 0.53 | 0.46 | 0.41 | 0.40 | 0.45 | 0.45 | 0.47 | 0.33 |
| `global_piqa_parallel_cloze` | 0.50 | 0.32 | 0.25 | 0.41 | 0.49 | 0.48 | 0.40 | 0.38 | 0.39 | 0.58 | 0.56 | 0.45 | 0.50 | 0.47 | 0.47 |
| `rf_cultural_bench_easy` | 0.30 | 0.33 | 0.45 | 0.42 | 0.39 | 0.35 | 0.36 | 0.38 | 0.29 | 0.38 | 0.33 | 0.36 | 0.26 | 0.26 | 0.41 |

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
subtitle: "Slovak (sk) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_hellaswag` | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** |
| `multiblimp` | **1.00** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** |
| `bbpb_multiblimp` | **1.00** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** |
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_rfgm_belebele` | **1.00** | **1.00** | 0.67 | **1.00** | **0.83** | **1.00** | 0.67 | **1.00** | **0.83** | 0.67 | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** |
| `bbpb_global_piqa_parallel_cloze_slk_latn` | 0.67 | **0.83** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | 0.67 | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** |
| `bbpb_rf_belebele` | **1.00** | **1.00** | **0.83** | **0.83** | 0.67 | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | 0.67 | **0.83** | 0.50 |
| `bbpb_arc` | **0.83** | **0.83** | 0.67 | **0.83** | 0.50 | **1.00** | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | 0.67 | **0.83** | 0.50 | 0.67 |
| `bbpb_global_piqa_parallel_cloze_slk_latn_sari` | 0.67 | 0.67 | **0.83** | 0.67 | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** | 0.50 | 0.67 | **0.83** | **0.83** | 0.67 | **0.83** |
| `bbpb_include_v2_og` | **1.00** | **0.83** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | 0.67 | 0.50 | **1.00** | 0.50 | 0.50 |
| `bbpb_rf_m_mmlu` | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | 0.50 | 0.67 | 0.67 | **0.83** | 0.50 | 0.67 |
| `hellaswag` | 0.50 | 0.33 | 0.67 | 0.50 | 0.33 | 0.67 | **0.83** | **1.00** | **0.83** | 0.50 | 0.67 | **0.83** | **0.83** | 0.67 | **0.83** |
| `bbpb_belebele` | 0.67 | 0.67 | 0.50 | 0.50 | **0.83** | 0.67 | **0.83** | **0.83** | 0.50 | **0.83** | 0.50 | 0.50 | 0.67 | 0.33 | 0.67 |
| `rfgm_belebele` | 0.33 | 0.67 | 0.50 | 0.67 | 0.67 | 0.33 | **0.83** | 0.33 | 0.33 | 0.50 | **1.00** | **1.00** | 0.50 | 0.50 | **1.00** |
| `arc` | **0.83** | 0.33 | **0.83** | 0.33 | 0.67 | 0.50 | 0.67 | 0.50 | **0.83** | 0.17 | 0.67 | 0.67 | 0.50 | 0.50 | 0.67 |
| `bbpb_include_v2_en` | 0.17 | 0.50 | **0.83** | 0.50 | **0.83** | 0.33 | 0.00 | 0.67 | 0.00 | 0.67 | 0.67 | 0.67 | 0.33 | **1.00** | 0.33 |
| `rf_belebele` | 0.50 | 0.33 | 0.50 | 0.50 | 0.67 | 0.17 | **0.83** | 0.17 | 0.33 | 0.00 | 0.67 | 0.50 | 0.33 | 0.50 | **0.83** |
| `global_piqa_parallel_cloze_slk_latn` | 0.67 | 0.50 | 0.33 | 0.17 | **1.00** | 0.33 | 0.17 | 0.00 | 0.67 | 0.67 | 0.33 | 0.50 | 0.17 | 0.33 | 0.17 |
| `belebele` | 0.17 | 0.33 | 0.33 | 0.67 | **0.83** | 0.33 | 0.67 | 0.17 | 0.17 | 0.00 | 0.67 | 0.33 | 0.17 | 0.50 | 0.67 |
| `global_piqa_parallel_cloze_slk_latn_sari` | 0.50 | 0.17 | 0.50 | 0.00 | 0.67 | 0.17 | 0.00 | 0.50 | 0.50 | 0.50 | 0.67 | 0.17 | 0.33 | 0.50 | 0.17 |
| `include_v2_en` | 0.50 | 0.00 | 0.67 | 0.50 | 0.17 | 0.00 | 0.67 | 0.50 | 0.50 | 0.17 | 0.17 | 0.50 | 0.33 | 0.50 | 0.17 |
| `include_v2_og` | 0.50 | 0.33 | 0.50 | 0.00 | 0.17 | **0.83** | 0.33 | 0.17 | 0.50 | 0.50 | 0.17 | 0.50 | 0.00 | 0.50 | 0.17 |
| `rf_m_mmlu` | 0.50 | 0.67 | 0.00 | 0.33 | 0.33 | 0.33 | 0.00 | 0.00 | 0.17 | 0.33 | 0.67 | 0.00 | 0.67 | 0.67 | 0.33 |

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
subtitle: "Slovak (sk) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Slovene (sl) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `multiblimp` | **1.00** | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** |
| `bbpb_multiblimp` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_rfgm_belebele` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_rf_belebele` | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** |
| `bbpb_global_piqa_parallel_cloze_slv_latn` | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** |
| `bbpb_global_piqa_parallel_cloze_slv_latn_cerk` | **1.00** | 0.67 | 0.67 | **0.83** | 0.67 | 0.67 | 0.67 | **0.83** | 0.67 | 0.67 | 0.50 | 0.67 | **0.83** | **1.00** | **0.83** |
| `bbpb_belebele` | **0.83** | 0.50 | **0.83** | **1.00** | 0.67 | 0.67 | **1.00** | **0.83** | 0.50 | 0.67 | 0.50 | 0.50 | **0.83** | 0.50 | 0.67 |
| `rfgm_belebele` | 0.67 | 0.33 | 0.67 | **1.00** | 0.33 | 0.67 | 0.67 | 0.67 | 0.67 | 0.50 | 0.33 | 0.67 | 0.67 | 0.67 | 0.33 |
| `global_piqa_parallel_cloze_slv_latn` | 0.33 | 0.50 | 0.33 | 0.33 | 0.50 | 0.33 | 0.17 | 0.17 | 0.00 | **0.83** | **0.83** | 0.67 | **1.00** | 0.67 | 0.67 |
| `belebele` | 0.50 | 0.33 | 0.50 | **1.00** | 0.67 | 0.50 | 0.33 | 0.50 | 0.50 | 0.67 | 0.33 | 0.00 | 0.50 | 0.17 | 0.67 |
| `rf_belebele` | 0.33 | 0.33 | 0.50 | 0.67 | **1.00** | 0.67 | **0.83** | 0.00 | 0.33 | 0.50 | 0.33 | 0.33 | 0.17 | 0.50 | 0.67 |
| `global_piqa_parallel_cloze_slv_latn_cerk` | 0.67 | 0.17 | 0.00 | 0.50 | 0.67 | 0.00 | 0.33 | **0.83** | 0.33 | 0.50 | 0.00 | 0.33 | 0.50 | 0.17 | 0.17 |

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
subtitle: "Slovene (sl) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Albanian (sq) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_rfgm_include_base_44` | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `bbpb_multiblimp-extra` | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** |
| `bbpb_rf_include_base_44` | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** | **1.00** | **0.83** |
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_global_piqa_parallel_cloze` | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | **1.00** | 0.67 | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** |
| `bbpb_include_v2_og` | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | 0.67 | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | 0.67 |
| `bbpb_rfgm_belebele` | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** |
| `bbpb_rf_belebele` | **0.83** | 0.67 | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 |
| `multiblimp-extra` | **0.83** | **0.83** | 0.67 | **1.00** | **0.83** | 0.67 | 0.50 | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | 0.67 | **0.83** | **0.83** |
| `include_v2_og` | 0.50 | **0.83** | 0.67 | 0.33 | **0.83** | 0.67 | **0.83** | **0.83** | 0.50 | 0.50 | 0.50 | **0.83** | 0.67 | 0.50 | 0.33 |
| `bbpb_belebele` | 0.67 | 0.50 | **0.83** | 0.67 | 0.50 | **0.83** | **0.83** | 0.33 | 0.50 | 0.67 | 0.17 | 0.33 | 0.50 | 0.33 | **0.83** |
| `bbpb_include_base_44` | **1.00** | 0.33 | 0.50 | 0.33 | 0.67 | 0.33 | 0.50 | 0.33 | 0.67 | **0.83** | 0.67 | 0.33 | **0.83** | 0.50 | 0.67 |
| `rfgm_include_base_44` | 0.17 | 0.17 | 0.17 | 0.33 | 0.17 | 0.67 | 0.67 | **0.83** | 0.67 | 0.67 | 0.50 | 0.67 | **0.83** | **1.00** | **0.83** |
| `rf_include_base_44` | 0.17 | 0.67 | 0.67 | **0.83** | **0.83** | 0.50 | 0.00 | 0.00 | 0.00 | 0.33 | 0.50 | 0.50 | **0.83** | **0.83** | **1.00** |
| `rf_belebele` | 0.50 | 0.33 | 0.67 | **0.83** | 0.33 | 0.17 | 0.17 | 0.67 | 0.50 | 0.67 | 0.50 | 0.67 | 0.50 | 0.33 | 0.50 |
| `global_piqa_parallel_cloze` | 0.33 | 0.33 | 0.33 | 0.50 | 0.33 | 0.50 | 0.67 | 0.17 | 0.33 | **0.83** | 0.50 | 0.67 | 0.50 | 0.67 | 0.67 |
| `bbpb_include_v2_en` | 0.33 | 0.17 | 0.50 | 0.33 | 0.17 | 0.50 | 0.17 | 0.33 | 0.50 | 0.67 | **0.83** | 0.67 | **0.83** | 0.33 | 0.50 |
| `include_base_44` | 0.67 | 0.50 | 0.67 | 0.17 | 0.50 | 0.50 | 0.33 | 0.33 | 0.50 | 0.17 | 0.67 | **0.83** | 0.17 | 0.17 | 0.50 |
| `rfgm_belebele` | 0.67 | 0.50 | 0.33 | 0.17 | 0.33 | 0.17 | 0.33 | 0.17 | 0.00 | 0.33 | 0.67 | **0.83** | 0.67 | 0.50 | **0.83** |
| `include_v2_en` | **0.83** | 0.00 | 0.50 | 0.00 | 0.33 | 0.00 | 0.50 | 0.00 | 0.33 | 0.50 | **1.00** | 0.67 | 0.50 | 0.50 | 0.67 |
| `belebele` | 0.17 | 0.67 | 0.17 | 0.67 | 0.67 | 0.50 | 0.50 | 0.00 | 0.50 | 0.00 | 0.33 | 0.33 | 0.33 | 0.50 | 0.50 |

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
subtitle: "Albanian (sq) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Serbian (sr) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_rf_global_mmlu_full` | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `bbpb_global_piqa_parallel_cloze_srp_latn` | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** |
| `bbpb_rfgm_include_base_44` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bpb` | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_rfgm_belebele` | **1.00** | **1.00** | 0.67 | **0.83** | **1.00** | **1.00** | 0.67 | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** |
| `bbpb_global_piqa_parallel_cloze_srp_cyrl` | 0.67 | **1.00** | **0.83** | **1.00** | **1.00** | 0.67 | **0.83** | 0.67 | 0.67 | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** |
| `bbpb_hellaswag` | **1.00** | **0.83** | 0.67 | **1.00** | **1.00** | **0.83** | 0.67 | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** |
| `bbpb_arc` | **1.00** | **0.83** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | **0.83** | 0.67 | **1.00** | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** |
| `bbpb_rf_belebele` | **0.83** | **0.83** | 0.67 | **0.83** | 0.67 | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** |
| `bbpb_rf_include_base_44` | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** |
| `bbpb_rf_m_mmlu` | **1.00** | **1.00** | **0.83** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | 0.67 |
| `bbpb_include_v2_og` | 0.67 | **0.83** | 0.67 | **1.00** | **1.00** | **0.83** | **1.00** | 0.67 | 0.67 | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** |
| `hellaswag` | 0.33 | 0.33 | 0.50 | 0.33 | 0.33 | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** |
| `bbpb_belebele` | 0.50 | **0.83** | **1.00** | 0.67 | **0.83** | 0.67 | 0.50 | **0.83** | 0.33 | **0.83** | **0.83** | 0.67 | 0.67 | **0.83** | 0.50 |
| `arc` | **0.83** | 0.33 | 0.17 | 0.33 | 0.50 | 0.50 | 0.33 | 0.50 | 0.67 | **0.83** | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** |
| `rfgm_belebele` | 0.50 | 0.50 | 0.67 | 0.67 | 0.33 | 0.67 | 0.50 | 0.50 | 0.50 | **0.83** | 0.67 | 0.67 | **0.83** | 0.50 | 0.67 |
| `belebele` | **0.83** | **0.83** | 0.50 | 0.50 | 0.50 | 0.67 | 0.67 | 0.33 | 0.67 | 0.67 | 0.67 | 0.33 | 0.67 | 0.67 | 0.33 |
| `bbpb_include_v2_en` | 0.33 | **0.83** | 0.50 | 0.67 | **0.83** | 0.50 | 0.50 | 0.67 | 0.17 | 0.33 | **0.83** | 0.67 | 0.50 | 0.67 | 0.50 |
| `include_v2_en` | 0.50 | **1.00** | 0.67 | 0.67 | 0.50 | 0.50 | **0.83** | 0.50 | 0.00 | 0.67 | 0.67 | 0.50 | 0.33 | 0.17 | 0.50 |
| `global_mmlu_full` | 0.50 | 0.50 | 0.50 | 0.67 | 0.50 | 0.67 | 0.33 | **0.83** | 0.67 | 0.00 | 0.50 | 0.33 | 0.50 | 0.67 | **0.83** |
| `rf_global_mmlu_full` | **1.00** | 0.50 | 0.33 | 0.33 | 0.17 | 0.50 | 0.33 | 0.33 | 0.17 | 0.50 | 0.50 | 0.33 | **1.00** | **0.83** | **0.83** |
| `include_v2_og` | **0.83** | 0.50 | 0.67 | 0.33 | 0.50 | 0.50 | 0.67 | 0.33 | 0.33 | 0.50 | 0.67 | 0.00 | 0.67 | 0.50 | 0.17 |
| `bbpb_include_base_44` | 0.67 | 0.33 | 0.67 | 0.50 | 0.50 | 0.67 | 0.33 | 0.17 | 0.50 | 0.33 | 0.17 | 0.50 | 0.50 | **0.83** | 0.33 |
| `bbpb_global_mmlu_full` | **1.00** | 0.50 | 0.00 | 0.17 | 0.33 | 0.50 | 0.00 | 0.17 | 0.33 | 0.50 | 0.67 | 0.50 | **0.83** | 0.67 | **0.83** |
| `rf_m_mmlu` | 0.33 | 0.17 | 0.50 | **0.83** | 0.50 | 0.50 | **0.83** | 0.17 | 0.17 | 0.67 | 0.33 | 0.67 | 0.33 | 0.33 | 0.67 |
| `global_piqa_parallel_cloze_srp_latn` | 0.17 | 0.50 | 0.00 | 0.17 | 0.17 | 0.67 | **0.83** | **1.00** | 0.33 | 0.50 | 0.67 | 0.00 | **0.83** | 0.50 | 0.33 |
| `rf_include_base_44` | 0.33 | 0.33 | 0.67 | **0.83** | **0.83** | 0.17 | 0.17 | 0.50 | 0.17 | 0.33 | 0.17 | 0.33 | 0.50 | 0.67 | 0.67 |
| `include_base_44` | 0.33 | 0.50 | 0.17 | 0.33 | 0.17 | 0.50 | 0.50 | 0.33 | 0.50 | 0.67 | 0.17 | 0.33 | 0.50 | 0.67 | **0.83** |
| `rf_belebele` | 0.50 | 0.67 | 0.33 | 0.33 | 0.00 | **0.83** | 0.17 | 0.33 | 0.17 | 0.17 | 0.33 | 0.17 | 0.67 | 0.67 | 0.50 |
| `global_piqa_parallel_cloze_srp_cyrl` | 0.17 | 0.17 | 0.67 | 0.00 | 0.00 | 0.50 | 0.50 | **0.83** | 0.33 | 0.33 | 0.67 | 0.50 | 0.33 | 0.00 | 0.50 |
| `rfgm_include_base_44` | 0.17 | 0.17 | **0.83** | 0.17 | 0.17 | 0.50 | 0.00 | 0.50 | 0.33 | 0.17 | 0.67 | 0.50 | 0.17 | 0.17 | **0.83** |

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
subtitle: "Serbian (sr) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Swedish (sv) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_hellaswag` | **0.96** | **0.93** | **0.85** | **0.93** | **0.87** | **0.96** | **0.89** | **0.96** | **0.91** | **0.93** | **0.93** | **0.91** | **0.89** | **0.91** | **0.91** |
| `bpb` | **1.00** | **0.95** | **0.82** | **0.93** | **0.96** | **0.95** | **0.82** | **0.93** | **0.96** | **0.84** | **0.98** | **0.95** | **0.82** | **0.78** | **0.93** |
| `bbpb_rfgm_belebele` | **0.95** | **0.95** | **0.82** | **0.93** | **0.76** | **0.93** | **0.80** | **0.91** | 0.75 | **0.80** | **0.87** | 0.75 | **0.89** | 0.73 | **0.76** |
| `hellaswag` | **0.78** | **0.82** | 0.67 | **0.78** | **0.82** | **0.89** | **0.78** | **0.85** | **0.82** | 0.75 | **0.85** | **0.82** | 0.75 | 0.75 | **0.96** |
| `bbpb_rf_m_mmlu` | **0.82** | **0.78** | 0.60 | **0.78** | 0.60 | **0.93** | **0.78** | **0.82** | 0.75 | **0.82** | **0.89** | 0.75 | **0.78** | **0.82** | **0.78** |
| `bbpb_rf_belebele` | **0.96** | **0.93** | **0.76** | **0.93** | 0.62 | **0.93** | **0.76** | **0.96** | 0.62 | 0.73 | **0.93** | 0.58 | **0.76** | 0.56 | 0.62 |
| `bbpb_global_piqa_parallel_cloze` | **0.82** | **0.76** | 0.71 | 0.67 | **0.82** | **0.84** | **0.78** | 0.67 | 0.67 | **0.84** | 0.69 | **0.76** | **0.78** | 0.71 | 0.71 |
| `bbpb_multiblimp` | **0.87** | **0.84** | **0.80** | **0.78** | 0.71 | **0.78** | 0.75 | **0.91** | 0.62 | **0.78** | 0.73 | 0.65 | 0.69 | 0.65 | 0.64 |
| `bbpb_include_v2_og` | 0.67 | 0.71 | 0.60 | 0.60 | 0.64 | **0.85** | 0.75 | 0.75 | **0.78** | 0.71 | **0.78** | **0.78** | 0.56 | 0.64 | 0.75 |
| `bbpb_arc` | 0.67 | 0.75 | 0.65 | 0.73 | 0.55 | 0.75 | **0.80** | **0.87** | 0.58 | 0.73 | 0.73 | 0.65 | **0.78** | 0.56 | 0.71 |
| `bbpb_arc_mt` | 0.71 | 0.75 | 0.62 | 0.71 | 0.56 | 0.71 | **0.80** | **0.82** | 0.56 | 0.73 | 0.71 | 0.67 | **0.80** | 0.62 | 0.71 |
| `bbpb_rf_global_mmlu_full` | **0.76** | **0.80** | 0.56 | **0.89** | 0.60 | **0.85** | 0.69 | **0.80** | 0.51 | 0.65 | **0.84** | 0.55 | 0.56 | 0.38 | 0.56 |
| `bbpb_belebele` | 0.62 | 0.56 | 0.58 | 0.69 | 0.56 | 0.58 | 0.56 | 0.71 | 0.62 | 0.65 | 0.55 | 0.53 | 0.67 | 0.62 | 0.58 |
| `bbpb_global_mmlu_full` | **0.82** | 0.55 | 0.45 | 0.58 | 0.53 | 0.65 | 0.53 | 0.62 | 0.60 | 0.58 | 0.60 | 0.73 | 0.62 | 0.56 | 0.65 |
| `rf_global_mmlu_full` | 0.42 | 0.55 | 0.55 | 0.49 | 0.44 | 0.53 | 0.60 | 0.60 | **0.76** | 0.71 | 0.47 | 0.55 | 0.65 | 0.65 | 0.69 |
| `bbpb_include_v2_en` | 0.49 | **0.82** | 0.62 | 0.56 | 0.65 | 0.49 | 0.47 | 0.67 | 0.40 | 0.47 | 0.49 | 0.62 | 0.65 | 0.49 | 0.55 |
| `rf_m_mmlu` | 0.53 | 0.55 | 0.55 | 0.25 | 0.44 | 0.49 | 0.51 | 0.36 | 0.44 | 0.64 | 0.53 | 0.67 | 0.56 | 0.71 | 0.60 |
| `include_v2_og` | 0.49 | 0.20 | 0.35 | 0.35 | 0.29 | 0.60 | 0.45 | 0.65 | 0.73 | 0.51 | 0.55 | 0.71 | 0.45 | 0.65 | **0.76** |
| `global_piqa_parallel_cloze` | 0.51 | 0.69 | 0.44 | 0.53 | 0.51 | 0.44 | 0.47 | 0.51 | 0.65 | 0.51 | 0.51 | 0.40 | 0.45 | 0.47 | 0.62 |
| `rfgm_belebele` | 0.64 | 0.58 | 0.51 | 0.40 | 0.44 | 0.56 | 0.53 | 0.58 | 0.49 | 0.55 | 0.44 | 0.31 | 0.53 | 0.55 | 0.49 |
| `include_v2_en` | 0.44 | 0.53 | 0.45 | 0.51 | 0.36 | 0.42 | 0.51 | 0.51 | 0.67 | 0.53 | 0.49 | 0.35 | 0.67 | 0.51 | 0.55 |
| `rf_belebele` | 0.36 | 0.40 | 0.44 | 0.51 | 0.58 | 0.44 | 0.51 | 0.62 | 0.40 | 0.55 | 0.62 | 0.47 | 0.45 | 0.55 | 0.51 |
| `belebele` | 0.36 | 0.29 | 0.38 | 0.53 | 0.42 | 0.60 | 0.51 | 0.56 | 0.45 | 0.53 | 0.45 | 0.75 | 0.51 | 0.40 | 0.51 |
| `arc_mt` | 0.45 | 0.44 | 0.73 | 0.44 | 0.20 | 0.56 | 0.55 | 0.40 | 0.27 | 0.64 | 0.58 | 0.45 | 0.64 | 0.31 | 0.58 |
| `global_mmlu_full` | 0.40 | 0.58 | 0.42 | 0.44 | 0.65 | 0.56 | 0.33 | 0.45 | 0.27 | 0.55 | 0.31 | 0.31 | 0.69 | 0.62 | 0.64 |
| `multiblimp` | 0.65 | 0.56 | 0.38 | 0.25 | 0.29 | 0.45 | 0.38 | 0.31 | 0.38 | **0.82** | 0.40 | 0.51 | 0.53 | 0.65 | 0.53 |
| `arc` | 0.25 | 0.49 | 0.45 | 0.38 | 0.44 | 0.40 | 0.45 | 0.44 | 0.25 | 0.49 | 0.71 | 0.55 | 0.51 | 0.49 | 0.56 |

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
subtitle: "Swedish (sv) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Tamil (ta) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.93** | **0.96** | **0.86** | **0.93** | **0.93** | **0.89** | **0.86** | **0.93** | **0.93** | **0.89** | **0.89** | **0.89** | **0.93** | **0.79** | **0.86** |
| `bbpb_rfgm_belebele` | **0.79** | **0.86** | **0.93** | **0.89** | **0.86** | **0.86** | **0.79** | **0.89** | **0.93** | **0.86** | **0.89** | **0.86** | **0.89** | **0.86** | **0.96** |
| `bbpb_global_piqa_parallel_cloze` | **0.82** | **0.96** | **0.93** | **0.89** | **0.89** | **0.86** | **0.89** | **0.79** | 0.71 | **0.96** | **0.86** | **0.86** | **0.82** | **0.82** | **0.86** |
| `bbpb_xcopa` | **0.93** | **0.96** | **0.93** | **0.79** | **0.82** | **0.89** | **0.86** | **0.79** | **0.89** | **0.89** | **0.82** | **0.79** | **0.86** | **0.82** | **0.89** |
| `bbpb_rfgm_include_base_44` | **0.86** | **0.89** | **0.96** | **0.75** | **0.75** | **0.89** | **0.89** | **0.82** | **0.82** | **0.93** | **0.79** | **0.79** | **0.79** | **0.79** | **0.93** |
| `bbpb_rf_belebele` | **0.86** | **0.86** | 0.71 | **0.93** | **0.89** | **0.86** | **0.79** | **0.79** | **0.89** | **0.79** | **0.86** | **0.82** | 0.64 | **0.82** | **0.82** |
| `bbpb_rf_include_base_44` | **0.93** | 0.68 | **0.89** | **0.89** | **0.82** | **0.75** | **0.82** | **0.96** | **0.75** | **0.79** | 0.71 | **0.79** | **0.79** | **0.93** | 0.71 |
| `bbpb_hellaswag` | **0.89** | **0.86** | **0.82** | **0.75** | 0.71 | **0.75** | **0.86** | 0.71 | 0.61 | **0.89** | **0.89** | **0.86** | **0.86** | **0.75** | **0.82** |
| `bbpb_include_v2_og` | **0.75** | 0.71 | **0.86** | **0.82** | **0.89** | **0.75** | **0.75** | **0.79** | 0.64 | 0.64 | **0.75** | **0.75** | **0.89** | **0.75** | 0.71 |
| `bbpb_multiblimp` | **0.86** | **0.75** | **0.82** | **0.86** | 0.57 | **0.82** | **0.75** | **0.93** | 0.64 | 0.64 | **0.75** | 0.68 | **0.82** | 0.46 | 0.57 |
| `bbpb_arc` | **0.89** | 0.61 | 0.64 | **0.86** | **0.82** | 0.64 | **0.75** | **0.82** | **0.79** | 0.68 | 0.54 | 0.71 | 0.64 | 0.68 | **0.75** |
| `bbpb_rf_m_mmlu` | **0.86** | 0.64 | 0.64 | **0.82** | **0.79** | 0.64 | 0.57 | **0.82** | 0.64 | 0.50 | 0.61 | 0.64 | 0.68 | 0.64 | 0.61 |
| `multiblimp` | 0.68 | 0.64 | **0.79** | 0.61 | 0.64 | 0.54 | **0.75** | 0.68 | 0.57 | 0.68 | 0.54 | 0.46 | 0.61 | 0.46 | 0.61 |
| `arc` | 0.71 | 0.50 | **0.82** | 0.64 | **0.79** | 0.68 | 0.61 | 0.43 | **0.93** | 0.46 | 0.50 | 0.61 | 0.54 | 0.61 | 0.43 |
| `bbpb_belebele` | 0.43 | 0.68 | **0.89** | 0.54 | 0.68 | 0.54 | 0.32 | 0.54 | 0.39 | 0.57 | 0.64 | 0.71 | 0.50 | 0.64 | 0.43 |
| `bbpb_include_base_44` | 0.68 | 0.32 | 0.61 | 0.61 | 0.25 | 0.64 | 0.57 | 0.57 | 0.57 | 0.57 | 0.64 | **0.79** | 0.50 | 0.64 | 0.43 |
| `rf_m_mmlu` | 0.46 | 0.39 | 0.71 | 0.46 | 0.57 | 0.21 | 0.39 | 0.50 | 0.25 | 0.46 | 0.64 | **0.75** | 0.54 | 0.57 | 0.68 |
| `rfgm_belebele` | 0.61 | 0.50 | 0.36 | 0.32 | 0.36 | 0.39 | 0.61 | 0.57 | 0.50 | 0.43 | 0.61 | 0.32 | 0.64 | **0.79** | 0.54 |
| `rf_belebele` | 0.71 | **0.82** | 0.46 | 0.32 | 0.46 | 0.68 | 0.32 | 0.39 | 0.61 | 0.54 | 0.36 | 0.39 | 0.43 | 0.36 | 0.68 |
| `belebele` | 0.43 | 0.43 | 0.46 | 0.54 | 0.46 | 0.46 | 0.57 | 0.32 | **0.82** | 0.71 | 0.32 | 0.54 | 0.46 | 0.57 | 0.36 |
| `bbpb_include_v2_en` | 0.29 | 0.50 | 0.50 | 0.57 | 0.61 | 0.36 | 0.29 | 0.36 | 0.25 | 0.57 | 0.57 | 0.46 | 0.57 | 0.61 | 0.61 |
| `include_v2_en` | 0.39 | 0.43 | 0.50 | 0.64 | 0.32 | 0.29 | 0.68 | 0.57 | 0.54 | 0.18 | 0.43 | 0.36 | 0.57 | 0.43 | 0.50 |
| `include_base_44` | 0.50 | 0.61 | 0.50 | 0.39 | 0.54 | 0.68 | 0.39 | 0.29 | 0.50 | 0.46 | 0.39 | 0.39 | 0.21 | 0.18 | 0.71 |
| `include_v2_og` | 0.54 | 0.43 | 0.43 | 0.61 | 0.21 | 0.25 | 0.29 | 0.54 | 0.43 | 0.54 | 0.36 | 0.54 | 0.61 | 0.50 | 0.39 |
| `hellaswag` | 0.54 | 0.54 | 0.32 | 0.43 | 0.54 | 0.57 | 0.36 | 0.25 | 0.54 | 0.50 | 0.32 | 0.36 | 0.39 | 0.54 | 0.39 |
| `xcopa` | 0.54 | 0.32 | 0.46 | 0.50 | 0.57 | 0.43 | 0.32 | 0.36 | 0.43 | 0.68 | 0.21 | 0.39 | 0.25 | 0.57 | 0.32 |
| `rf_include_base_44` | 0.46 | 0.21 | 0.54 | 0.54 | 0.32 | 0.36 | 0.39 | 0.36 | 0.43 | 0.39 | 0.46 | 0.64 | 0.18 | 0.57 | 0.36 |
| `rfgm_include_base_44` | 0.50 | 0.68 | 0.18 | 0.46 | 0.29 | 0.39 | 0.29 | 0.61 | 0.14 | 0.25 | 0.36 | 0.36 | 0.57 | 0.71 | 0.43 |
| `global_piqa_parallel_cloze` | 0.57 | 0.57 | 0.46 | 0.46 | 0.18 | 0.36 | 0.21 | 0.25 | 0.36 | **0.75** | 0.43 | 0.29 | 0.50 | 0.25 | 0.21 |

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
subtitle: "Tamil (ta) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Thai (th) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.93** | **0.87** | **0.83** | **0.93** | **0.93** | **0.93** | **0.88** | **0.96** | **0.94** | **0.87** | **0.91** | **0.90** | **0.88** | **0.84** | **0.94** |
| `bbpb_global_piqa_parallel_cloze` | 0.68 | 0.58 | 0.69 | 0.67 | 0.70 | 0.71 | **0.78** | 0.69 | **0.82** | 0.65 | **0.76** | 0.63 | 0.70 | 0.71 | 0.70 |
| `bbpb_xcopa` | **0.76** | 0.73 | **0.78** | **0.76** | 0.64 | 0.61 | 0.70 | **0.79** | 0.65 | 0.72 | 0.59 | **0.79** | 0.68 | 0.74 | 0.51 |
| `bbpb_rfgm_belebele` | **0.77** | 0.64 | 0.66 | 0.64 | **0.80** | 0.64 | **0.75** | **0.75** | 0.74 | 0.66 | 0.60 | 0.61 | 0.71 | 0.69 | 0.72 |
| `bbpb_xnli` | **0.76** | 0.66 | 0.65 | 0.67 | **0.76** | **0.82** | 0.66 | **0.83** | 0.70 | 0.53 | **0.82** | 0.61 | 0.58 | 0.62 | 0.64 |
| `bbpb_rf_belebele` | 0.69 | **0.75** | 0.70 | 0.64 | 0.49 | 0.74 | 0.62 | 0.72 | 0.59 | 0.70 | 0.66 | 0.56 | 0.59 | 0.32 | 0.56 |
| `bbpb_belebele` | 0.66 | 0.55 | 0.70 | 0.65 | 0.64 | 0.63 | 0.49 | 0.53 | 0.60 | 0.40 | 0.47 | 0.57 | 0.69 | 0.52 | 0.55 |
| `xnli` | 0.64 | 0.61 | 0.49 | 0.56 | 0.60 | 0.59 | 0.59 | 0.54 | 0.65 | 0.60 | 0.48 | 0.36 | 0.51 | 0.53 | 0.57 |
| `belebele` | 0.45 | 0.51 | 0.58 | 0.59 | 0.46 | 0.37 | 0.56 | 0.66 | 0.53 | 0.41 | 0.43 | 0.62 | 0.65 | 0.49 | 0.61 |
| `rf_belebele` | 0.54 | 0.67 | 0.50 | 0.44 | 0.35 | 0.39 | 0.63 | 0.41 | 0.42 | 0.35 | 0.43 | 0.44 | 0.50 | 0.50 | 0.41 |
| `rfgm_belebele` | 0.32 | 0.38 | 0.39 | 0.43 | 0.18 | 0.50 | 0.50 | 0.52 | 0.60 | 0.55 | 0.51 | 0.52 | 0.35 | 0.56 | 0.58 |
| `xcopa` | 0.50 | 0.57 | 0.41 | 0.36 | 0.32 | 0.44 | 0.41 | 0.36 | 0.22 | 0.35 | 0.27 | 0.37 | 0.46 | 0.39 | 0.58 |
| `global_piqa_parallel_cloze` | 0.55 | 0.30 | 0.32 | 0.47 | 0.35 | 0.39 | 0.30 | 0.53 | 0.29 | 0.37 | 0.52 | 0.45 | 0.15 | 0.35 | 0.29 |

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

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.96** | **0.84** | 0.65 | **0.89** | **0.96** | **0.87** | 0.69 | **0.89** | **0.93** | 0.75 | **0.87** | **0.80** | 0.73 | 0.65 | **0.85** |
| `bbpb_rfgm_include_base_44` | **0.89** | 0.71 | **0.78** | **0.78** | 0.75 | 0.67 | 0.71 | 0.75 | 0.75 | 0.75 | **0.85** | **0.78** | **0.82** | **0.85** | **0.85** |
| `bbpb_xcopa` | **0.76** | 0.62 | **0.80** | **0.84** | **0.85** | 0.71 | 0.64 | 0.75 | 0.73 | 0.67 | **0.78** | 0.69 | **0.78** | 0.73 | **0.80** |
| `bbpb_rfgm_belebele` | **0.89** | **0.78** | 0.65 | 0.67 | 0.71 | 0.75 | **0.76** | 0.71 | 0.67 | 0.73 | 0.71 | **0.82** | **0.80** | 0.65 | 0.67 |
| `bbpb_include_v2_og` | 0.73 | 0.62 | **0.82** | **0.76** | 0.67 | 0.53 | **0.84** | 0.71 | 0.73 | 0.65 | 0.60 | 0.58 | **0.76** | **0.78** | 0.69 |
| `bbpb_rf_belebele` | **0.84** | 0.73 | **0.82** | 0.60 | 0.62 | **0.78** | **0.84** | 0.69 | 0.64 | 0.69 | 0.65 | 0.64 | 0.64 | 0.51 | **0.76** |
| `bbpb_global_piqa_parallel_cloze` | 0.69 | 0.67 | 0.64 | 0.60 | 0.71 | 0.73 | **0.84** | 0.69 | 0.58 | 0.67 | 0.67 | 0.60 | **0.82** | 0.60 | 0.71 |
| `bbpb_xnli` | **0.85** | 0.62 | 0.62 | 0.75 | 0.60 | **0.76** | 0.69 | 0.71 | 0.64 | 0.56 | 0.73 | 0.62 | 0.51 | 0.51 | 0.64 |
| `bbpb_include_base_44` | 0.69 | 0.53 | **0.85** | 0.69 | 0.64 | 0.44 | **0.76** | **0.82** | 0.65 | 0.53 | 0.40 | 0.42 | **0.80** | 0.64 | 0.65 |
| `bbpb_toksuite` | **0.78** | **0.76** | 0.64 | 0.62 | 0.55 | 0.69 | 0.67 | 0.65 | 0.55 | 0.51 | 0.45 | 0.45 | 0.65 | 0.58 | 0.60 |
| `bbpb_rf_global_mmlu_full` | 0.65 | 0.65 | 0.65 | **0.82** | 0.65 | 0.64 | 0.60 | 0.62 | 0.38 | 0.64 | 0.73 | 0.49 | 0.58 | 0.42 | 0.62 |
| `bbpb_global_mmlu_full` | 0.69 | 0.42 | 0.55 | 0.65 | 0.58 | 0.62 | 0.75 | 0.67 | 0.67 | 0.58 | 0.40 | 0.44 | 0.71 | 0.56 | 0.67 |
| `bbpb_belebele` | 0.62 | 0.49 | 0.65 | **0.76** | 0.47 | 0.62 | 0.71 | 0.53 | 0.60 | 0.51 | 0.36 | 0.40 | 0.71 | 0.64 | 0.53 |
| `multiblimp` | 0.53 | 0.47 | 0.55 | 0.73 | 0.35 | **0.78** | 0.51 | 0.51 | 0.56 | 0.58 | 0.55 | 0.64 | 0.75 | 0.55 | 0.45 |
| `bbpb_rf_include_base_44` | 0.49 | 0.42 | 0.65 | 0.60 | 0.40 | 0.56 | 0.69 | 0.53 | 0.44 | 0.58 | 0.53 | 0.47 | 0.58 | 0.53 | **0.76** |
| `bbpb_multiblimp` | 0.60 | 0.53 | 0.64 | 0.71 | 0.51 | 0.71 | 0.53 | 0.56 | 0.44 | 0.53 | 0.49 | 0.51 | 0.49 | 0.47 | 0.51 |
| `include_v2_og` | 0.45 | 0.56 | 0.51 | 0.53 | 0.60 | 0.45 | 0.51 | 0.67 | 0.58 | 0.47 | 0.45 | 0.53 | 0.56 | 0.69 | 0.47 |
| `rf_belebele` | 0.47 | 0.67 | 0.35 | 0.60 | **0.76** | 0.33 | 0.64 | 0.71 | 0.56 | 0.38 | 0.38 | 0.55 | 0.56 | 0.45 | 0.60 |
| `global_mmlu_full` | 0.58 | 0.75 | 0.35 | 0.24 | 0.44 | 0.56 | 0.58 | 0.55 | 0.60 | 0.35 | 0.35 | 0.36 | 0.75 | 0.69 | 0.65 |
| `xnli` | 0.49 | 0.51 | 0.31 | 0.49 | 0.42 | 0.65 | 0.45 | 0.53 | 0.53 | 0.49 | 0.44 | 0.62 | 0.55 | 0.71 | 0.56 |
| `global_piqa_parallel_cloze` | 0.51 | 0.60 | 0.44 | 0.65 | 0.56 | 0.53 | 0.33 | 0.42 | 0.56 | 0.49 | 0.51 | 0.56 | 0.38 | 0.55 | 0.64 |
| `rf_global_mmlu_full` | 0.56 | 0.60 | 0.62 | 0.51 | 0.53 | 0.53 | 0.49 | 0.38 | 0.40 | 0.44 | 0.58 | 0.44 | 0.67 | 0.38 | 0.60 |
| `rfgm_include_base_44` | 0.53 | 0.47 | 0.51 | 0.71 | 0.71 | 0.58 | 0.15 | 0.55 | 0.60 | 0.27 | 0.55 | 0.60 | 0.40 | 0.42 | 0.65 |
| `include_base_44` | 0.55 | 0.35 | 0.53 | 0.64 | 0.49 | 0.33 | 0.56 | 0.62 | 0.49 | 0.40 | 0.40 | 0.64 | 0.62 | 0.60 | 0.44 |
| `rfgm_belebele` | 0.51 | 0.64 | 0.38 | 0.60 | 0.64 | 0.40 | 0.35 | 0.62 | 0.60 | 0.51 | 0.58 | 0.65 | 0.24 | 0.36 | 0.56 |
| `bbpb_include_v2_en` | 0.35 | 0.47 | 0.36 | 0.42 | 0.73 | 0.58 | 0.51 | 0.60 | 0.40 | 0.67 | 0.62 | 0.42 | 0.58 | 0.31 | 0.51 |
| `belebele` | 0.47 | 0.42 | 0.51 | 0.44 | 0.44 | 0.55 | 0.49 | 0.45 | 0.33 | 0.60 | 0.49 | 0.42 | 0.45 | 0.53 | 0.75 |
| `include_v2_en` | 0.55 | 0.55 | 0.47 | 0.49 | 0.47 | 0.51 | 0.51 | 0.51 | 0.60 | 0.51 | 0.27 | 0.33 | 0.47 | 0.18 | 0.47 |
| `rf_include_base_44` | 0.38 | 0.33 | 0.25 | 0.44 | 0.55 | 0.55 | 0.45 | 0.33 | 0.60 | 0.49 | 0.45 | 0.38 | 0.55 | 0.51 | 0.42 |
| `xcopa` | 0.33 | 0.31 | 0.33 | 0.42 | 0.38 | 0.62 | 0.51 | 0.38 | 0.31 | 0.33 | 0.42 | 0.44 | 0.67 | 0.44 | 0.65 |
| `toksuite` | 0.36 | 0.35 | 0.42 | 0.33 | 0.29 | 0.20 | 0.33 | 0.42 | 0.51 | 0.51 | 0.38 | 0.44 | 0.35 | 0.29 | 0.55 |

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

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.96** | **0.93** | 0.75 | **0.89** | **0.96** | **0.93** | **0.78** | **0.85** | **0.93** | 0.75 | **0.93** | **0.93** | 0.67 | 0.71 | **0.89** |
| `bbpb_hellaswag` | **0.93** | **0.85** | **0.85** | **0.82** | **0.80** | **0.89** | **0.85** | **0.78** | **0.84** | **0.89** | **0.82** | **0.80** | **0.85** | 0.73 | **0.76** |
| `bbpb_rfgm_include_base_44` | **0.89** | 0.75 | 0.67 | 0.75 | **0.76** | **0.85** | 0.64 | 0.75 | **0.76** | 0.64 | 0.71 | 0.69 | 0.64 | **0.76** | 0.58 |
| `multiblimp` | **0.76** | **0.78** | 0.75 | 0.71 | 0.67 | 0.73 | 0.73 | **0.78** | 0.58 | 0.71 | 0.60 | 0.65 | **0.78** | 0.69 | 0.65 |
| `bbpb_rfgm_belebele` | **0.76** | **0.78** | 0.65 | 0.67 | **0.76** | 0.65 | **0.82** | **0.84** | 0.71 | 0.58 | 0.60 | **0.80** | 0.73 | 0.56 | 0.62 |
| `bbpb_global_piqa_parallel_cloze` | **0.85** | 0.62 | 0.67 | 0.67 | 0.73 | 0.69 | **0.78** | 0.71 | 0.65 | 0.58 | 0.62 | 0.53 | **0.78** | **0.76** | **0.80** |
| `hellaswag` | 0.56 | 0.67 | 0.60 | 0.56 | 0.51 | 0.75 | 0.75 | **0.82** | 0.69 | 0.67 | **0.82** | **0.76** | 0.71 | 0.55 | **0.84** |
| `bbpb_multiblimp` | **0.78** | 0.67 | **0.84** | **0.78** | 0.60 | 0.64 | 0.73 | **0.78** | 0.49 | 0.69 | 0.60 | 0.64 | **0.80** | 0.58 | 0.49 |
| `bbpb_rf_belebele` | **0.84** | 0.73 | 0.55 | 0.71 | 0.69 | **0.78** | 0.53 | **0.80** | 0.64 | 0.64 | 0.69 | 0.45 | 0.58 | 0.45 | 0.58 |
| `bbpb_include_v2_og` | 0.58 | 0.67 | 0.47 | 0.44 | 0.60 | 0.69 | **0.85** | 0.67 | 0.58 | 0.58 | 0.65 | 0.67 | 0.64 | 0.58 | 0.51 |
| `bbpb_rf_include_base_44` | 0.62 | 0.56 | 0.38 | 0.60 | 0.62 | 0.51 | 0.73 | 0.69 | 0.56 | 0.49 | 0.60 | 0.69 | 0.67 | 0.44 | 0.55 |
| `include_v2_og` | 0.55 | 0.75 | 0.62 | 0.51 | 0.65 | 0.42 | 0.58 | 0.55 | 0.38 | 0.56 | 0.58 | 0.69 | 0.56 | 0.62 | 0.67 |
| `bbpb_belebele` | 0.53 | 0.49 | **0.76** | 0.60 | 0.55 | 0.56 | 0.62 | 0.60 | 0.62 | 0.47 | 0.35 | 0.55 | 0.65 | 0.71 | 0.47 |
| `bbpb_rf_global_mmlu_full` | 0.75 | 0.47 | 0.42 | **0.84** | 0.58 | 0.58 | 0.56 | **0.80** | 0.47 | 0.65 | 0.56 | 0.49 | 0.44 | 0.29 | 0.53 |
| `bbpb_rf_m_mmlu` | 0.62 | 0.55 | 0.31 | 0.44 | 0.60 | 0.67 | 0.58 | 0.64 | 0.58 | 0.58 | 0.53 | 0.55 | 0.55 | 0.53 | 0.65 |
| `bbpb_arc` | 0.73 | 0.55 | 0.51 | 0.73 | 0.71 | 0.35 | 0.56 | 0.67 | 0.44 | 0.45 | 0.56 | 0.51 | 0.64 | 0.40 | 0.55 |
| `rfgm_belebele` | **0.85** | 0.49 | 0.56 | 0.42 | 0.56 | 0.56 | 0.56 | 0.49 | 0.51 | 0.60 | 0.60 | 0.51 | 0.31 | 0.60 | 0.51 |
| `bbpb_include_base_44` | 0.64 | 0.65 | 0.45 | 0.36 | 0.56 | 0.51 | 0.45 | 0.62 | 0.64 | 0.33 | 0.42 | 0.65 | 0.69 | 0.45 | 0.51 |
| `bbpb_global_mmlu_full` | **0.78** | 0.40 | 0.42 | 0.69 | 0.53 | 0.47 | 0.38 | 0.58 | 0.56 | 0.40 | 0.42 | 0.44 | 0.62 | 0.53 | 0.62 |
| `rfgm_include_base_44` | 0.71 | 0.36 | 0.40 | 0.64 | 0.60 | 0.40 | 0.53 | 0.55 | 0.58 | 0.56 | 0.35 | 0.49 | 0.56 | 0.44 | 0.55 |
| `bbpb_include_v2_en` | 0.16 | 0.62 | 0.56 | 0.55 | **0.78** | 0.44 | 0.45 | 0.47 | 0.20 | 0.65 | 0.67 | 0.51 | 0.73 | 0.42 | 0.47 |
| `arc` | 0.40 | 0.45 | 0.36 | 0.45 | 0.62 | 0.73 | 0.55 | 0.62 | 0.38 | 0.42 | 0.60 | 0.47 | **0.80** | 0.36 | 0.42 |
| `include_v2_en` | 0.62 | 0.58 | 0.58 | 0.40 | 0.45 | 0.73 | 0.49 | 0.58 | 0.36 | 0.58 | 0.49 | 0.47 | 0.38 | 0.47 | 0.42 |
| `rf_belebele` | 0.65 | 0.62 | 0.47 | 0.51 | 0.55 | 0.45 | 0.62 | 0.53 | 0.40 | 0.55 | 0.55 | 0.51 | 0.58 | 0.36 | 0.16 |
| `include_base_44` | 0.44 | 0.45 | 0.49 | 0.51 | 0.53 | 0.44 | 0.42 | 0.65 | 0.65 | 0.42 | 0.20 | 0.71 | 0.60 | 0.51 | 0.40 |
| `global_mmlu_full` | 0.42 | 0.51 | 0.42 | 0.44 | 0.40 | 0.47 | 0.45 | 0.62 | **0.80** | 0.55 | 0.53 | 0.35 | 0.51 | 0.33 | 0.60 |
| `belebele` | 0.56 | 0.40 | 0.45 | 0.40 | 0.65 | 0.47 | 0.25 | 0.35 | 0.27 | 0.53 | 0.75 | 0.38 | 0.73 | 0.58 | 0.42 |
| `rf_m_mmlu` | 0.49 | 0.60 | 0.47 | 0.55 | 0.40 | 0.36 | 0.36 | 0.45 | 0.45 | 0.33 | 0.55 | 0.56 | 0.47 | 0.51 | 0.62 |
| `rf_include_base_44` | 0.38 | 0.49 | 0.51 | 0.64 | 0.36 | 0.47 | 0.25 | 0.42 | 0.58 | 0.45 | 0.53 | 0.53 | 0.58 | 0.55 | 0.42 |
| `rf_global_mmlu_full` | 0.44 | 0.58 | 0.44 | 0.47 | 0.40 | 0.44 | 0.56 | 0.24 | 0.35 | 0.44 | 0.45 | 0.42 | 0.38 | 0.45 | 0.67 |
| `global_piqa_parallel_cloze` | 0.58 | 0.38 | 0.42 | 0.22 | 0.58 | 0.58 | 0.22 | 0.40 | 0.60 | 0.35 | 0.60 | 0.42 | 0.64 | 0.29 | 0.24 |

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
subtitle: "Urdu (ur) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bbpb_global_piqa_parallel_cloze_urd_arab` | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** |
| `bbpb_rf_belebele_urd_Arab` | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | **1.00** | **1.00** | 0.67 | 0.67 | **1.00** |
| `bbpb_rfgm_include_base_44` | **0.83** | 0.67 | **1.00** | **0.83** | **1.00** | **0.83** | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** |
| `bpb` | **0.83** | **0.83** | **0.83** | **1.00** | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** |
| `multiblimp` | **1.00** | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | **0.83** |
| `bbpb_multiblimp` | **0.83** | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** | 0.67 | **1.00** | **0.83** | **0.83** | **0.83** | 0.67 | 0.67 | **0.83** | **0.83** |
| `bbpb_xnli` | 0.67 | 0.67 | **0.83** | **0.83** | **1.00** | **1.00** | **0.83** | **0.83** | 0.67 | **0.83** | **0.83** | 0.67 | 0.67 | **0.83** | **0.83** |
| `bbpb_rfgm_belebele_urd_Arab` | **0.83** | **0.83** | 0.67 | 0.67 | **1.00** | 0.67 | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **0.83** | **1.00** | 0.67 | 0.67 |
| `bbpb_include_v2_og_urdu_pakistan` | 0.50 | **0.83** | 0.50 | **0.83** | 0.67 | 0.67 | **1.00** | 0.67 | **0.83** | 0.67 | **1.00** | **0.83** | 0.67 | **0.83** | **0.83** |
| `bbpb_rf_belebele_urd_Latn` | 0.67 | 0.67 | **0.83** | 0.67 | **1.00** | 0.33 | 0.50 | 0.33 | 0.67 | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** | 0.67 |
| `bbpb_belebele_urd_Latn` | **1.00** | 0.67 | 0.50 | **0.83** | 0.67 | 0.67 | 0.50 | **0.83** | 0.67 | 0.50 | **0.83** | 0.33 | 0.67 | 0.50 | 0.50 |
| `bbpb_belebele_urd_Arab` | 0.67 | **0.83** | **0.83** | 0.67 | 0.50 | **0.83** | 0.50 | 0.67 | 0.17 | 0.67 | **0.83** | 0.33 | 0.50 | 0.67 | 0.50 |
| `bbpb_rfgm_belebele_urd_Latn` | 0.17 | **0.83** | **1.00** | **0.83** | **0.83** | 0.00 | 0.17 | 0.00 | 0.33 | **0.83** | **1.00** | 0.67 | **0.83** | **0.83** | 0.67 |
| `bbpb_include_base_44` | 0.50 | 0.17 | 0.33 | 0.50 | 0.33 | 0.67 | 0.50 | 0.67 | 0.50 | **0.83** | 0.67 | **0.83** | 0.50 | **1.00** | 0.50 |
| `belebele_urd_Latn` | 0.33 | **0.83** | 0.67 | 0.67 | 0.50 | 0.50 | 0.33 | 0.33 | **0.83** | 0.50 | **0.83** | 0.67 | 0.33 | 0.50 | 0.50 |
| `rf_belebele_urd_Latn` | 0.50 | 0.33 | 0.50 | 0.33 | 0.33 | 0.67 | **1.00** | 0.33 | 0.33 | 0.67 | 0.67 | 0.67 | 0.33 | 0.33 | **1.00** |
| `include_base_44` | 0.67 | 0.50 | 0.50 | 0.67 | 0.67 | **0.83** | 0.17 | **1.00** | 0.33 | 0.00 | **0.83** | 0.17 | 0.17 | 0.67 | 0.33 |
| `bbpb_include_v2_en_urdu_pakistan` | 0.50 | 0.67 | 0.00 | 0.50 | **1.00** | 0.50 | 0.50 | 0.33 | 0.50 | 0.33 | **0.83** | 0.67 | 0.50 | 0.00 | 0.50 |
| `bbpb_rf_include_base_44` | 0.50 | **0.83** | 0.17 | 0.33 | **0.83** | 0.33 | 0.33 | 0.50 | 0.67 | 0.33 | 0.50 | 0.67 | **0.83** | 0.00 | 0.17 |
| `belebele_urd_Arab` | **0.83** | **0.83** | 0.33 | 0.50 | 0.17 | **1.00** | 0.17 | 0.33 | 0.33 | 0.17 | 0.33 | 0.33 | **0.83** | 0.50 | 0.33 |
| `bbpb_global_piqa_parallel_cloze_urd_latn` | 0.33 | **0.83** | 0.33 | 0.50 | 0.33 | 0.50 | 0.33 | 0.50 | **1.00** | 0.17 | 0.33 | 0.50 | 0.50 | 0.33 | 0.50 |
| `bbpb_include_v2_og_urdu_india` | 0.50 | 0.67 | 0.33 | 0.33 | 0.67 | 0.17 | **0.83** | 0.17 | **0.83** | 0.00 | 0.67 | 0.33 | 0.33 | 0.67 | 0.33 |
| `rfgm_include_base_44` | 0.33 | 0.17 | 0.67 | 0.17 | 0.67 | 0.50 | 0.67 | 0.67 | 0.33 | 0.33 | **0.83** | 0.33 | 0.50 | 0.33 | 0.33 |
| `xnli` | 0.00 | 0.67 | 0.50 | 0.33 | 0.67 | 0.33 | 0.50 | 0.67 | 0.33 | 0.50 | 0.33 | 0.33 | **0.83** | 0.50 | 0.33 |
| `rfgm_belebele_urd_Arab` | 0.67 | 0.33 | 0.67 | 0.67 | 0.50 | 0.17 | 0.67 | 0.33 | 0.17 | 0.00 | 0.67 | 0.50 | 0.33 | 0.50 | 0.50 |
| `bbpb_include_v2_en_urdu_india` | 0.00 | 0.67 | 0.17 | 0.67 | 0.50 | 0.33 | **0.83** | 0.33 | 0.50 | 0.50 | **1.00** | 0.17 | 0.50 | 0.33 | 0.17 |
| `rfgm_belebele_urd_Latn` | 0.33 | **0.83** | 0.33 | 0.50 | 0.67 | 0.17 | 0.33 | 0.00 | 0.17 | 0.50 | 0.67 | 0.50 | 0.50 | 0.33 | **0.83** |
| `global_piqa_parallel_cloze_urd_latn` | 0.50 | 0.00 | 0.00 | 0.50 | 0.33 | 0.17 | 0.33 | **1.00** | **0.83** | **0.83** | 0.17 | 0.33 | 0.33 | 0.50 | **0.83** |
| `include_v2_en_urdu_pakistan` | 0.33 | **0.83** | 0.17 | 0.33 | 0.00 | 0.17 | **0.83** | 0.67 | 0.50 | 0.00 | 0.50 | 0.17 | 0.50 | 0.67 | 0.50 |
| `include_v2_og_urdu_pakistan` | 0.67 | 0.17 | 0.50 | 0.00 | 0.17 | 0.17 | 0.50 | 0.33 | 0.50 | 0.50 | 0.67 | 0.67 | 0.17 | 0.17 | **0.83** |
| `rf_belebele_urd_Arab` | 0.33 | 0.50 | 0.67 | 0.67 | 0.33 | 0.50 | 0.00 | 0.33 | 0.33 | 0.50 | 0.17 | 0.00 | 0.67 | 0.33 | 0.67 |
| `include_v2_og_urdu_india` | 0.17 | 0.17 | 0.17 | 0.33 | 0.00 | 0.50 | **1.00** | 0.17 | 0.50 | 0.50 | 0.17 | 0.67 | 0.17 | 0.50 | 0.50 |
| `include_v2_en_urdu_india` | 0.67 | 0.17 | 0.33 | 0.17 | 0.67 | 0.17 | 0.00 | 0.17 | 0.50 | 0.50 | 0.33 | 0.33 | **0.83** | 0.33 | 0.17 |
| `rf_include_base_44` | 0.17 | 0.17 | 0.17 | 0.67 | 0.17 | 0.50 | 0.50 | 0.17 | 0.50 | 0.67 | 0.33 | 0.33 | 0.33 | 0.50 | 0.00 |
| `global_piqa_parallel_cloze_urd_arab` | 0.50 | 0.00 | 0.17 | 0.33 | 0.33 | 0.00 | 0.00 | 0.00 | 0.17 | **0.83** | 0.67 | 0.50 | 0.50 | 0.33 | 0.50 |

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
subtitle: "Urdu (ur) · the table before, as a heatmap"
---

---
title: Appendix — Decision accuracy across sizes
subtitle: "Vietnamese (vi) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.94** | **0.87** | **0.85** | **0.95** | **0.97** | **0.91** | **0.89** | **0.93** | **0.95** | **0.88** | **0.88** | **0.88** | **0.85** | **0.86** | **0.94** |
| `bbpb_hellaswag` | **0.95** | **0.86** | **0.83** | **0.82** | **0.88** | **0.88** | **0.85** | **0.85** | **0.90** | **0.84** | **0.88** | **0.86** | **0.87** | **0.80** | **0.85** |
| `bbpb_rfgm_include_base_44` | 0.73 | 0.73 | **0.82** | **0.84** | **0.82** | **0.75** | 0.68 | 0.72 | 0.74 | **0.80** | **0.82** | 0.69 | **0.89** | 0.71 | 0.73 |
| `bbpb_xcopa` | **0.78** | 0.72 | 0.72 | 0.74 | **0.81** | **0.85** | 0.70 | 0.74 | **0.78** | 0.68 | 0.71 | **0.79** | **0.79** | **0.78** | **0.78** |
| `bbpb_rfgm_belebele` | **0.85** | **0.76** | **0.77** | 0.67 | **0.82** | 0.72 | **0.78** | 0.62 | 0.73 | 0.71 | 0.64 | 0.72 | 0.70 | 0.70 | 0.65 |
| `hellaswag` | 0.70 | **0.75** | 0.60 | 0.65 | **0.78** | **0.79** | **0.77** | **0.75** | **0.75** | 0.74 | 0.71 | 0.70 | 0.73 | 0.62 | **0.79** |
| `bbpb_global_piqa_parallel_cloze` | 0.68 | **0.76** | **0.78** | 0.63 | 0.67 | 0.73 | **0.75** | 0.71 | 0.69 | 0.70 | 0.57 | 0.68 | **0.84** | 0.66 | 0.72 |
| `bbpb_xnli` | 0.68 | 0.70 | **0.80** | 0.66 | 0.67 | 0.70 | 0.65 | 0.69 | 0.65 | **0.78** | 0.61 | 0.72 | 0.64 | 0.68 | 0.66 |
| `bbpb_rf_belebele` | 0.73 | 0.72 | **0.75** | **0.77** | 0.68 | 0.63 | 0.72 | 0.63 | 0.62 | 0.72 | 0.62 | 0.67 | 0.70 | 0.60 | 0.70 |
| `bbpb_include_v2_og` | 0.65 | 0.67 | 0.66 | 0.65 | 0.63 | 0.58 | 0.74 | **0.75** | 0.67 | 0.66 | 0.65 | **0.75** | 0.71 | **0.78** | 0.65 |
| `bbpb_rf_m_mmlu` | 0.68 | 0.65 | 0.69 | 0.71 | 0.63 | 0.63 | 0.74 | 0.71 | 0.60 | 0.62 | 0.54 | 0.70 | 0.67 | 0.71 | 0.54 |
| `bbpb_rf_include_base_44` | 0.63 | 0.55 | 0.67 | 0.70 | 0.62 | 0.48 | 0.72 | 0.62 | 0.63 | 0.68 | 0.65 | 0.73 | 0.67 | 0.70 | **0.77** |
| `bbpb_rf_global_mmlu_full` | 0.62 | 0.61 | 0.67 | 0.57 | 0.68 | 0.69 | 0.63 | **0.75** | 0.67 | 0.59 | 0.57 | 0.57 | 0.63 | **0.80** | 0.70 |
| `bbpb_arc` | 0.65 | 0.71 | 0.57 | 0.68 | 0.65 | 0.66 | 0.68 | 0.64 | 0.58 | 0.65 | 0.65 | 0.62 | 0.67 | 0.54 | 0.68 |
| `include_v2_og` | 0.54 | 0.64 | 0.55 | 0.60 | 0.68 | 0.65 | 0.68 | **0.81** | 0.55 | 0.49 | 0.62 | 0.66 | 0.68 | 0.65 | 0.65 |
| `bbpb_truthfulqa_mc2` | 0.57 | 0.54 | **0.75** | 0.74 | 0.56 | 0.72 | 0.53 | 0.52 | 0.65 | 0.54 | 0.43 | 0.58 | 0.64 | 0.69 | 0.62 |
| `bbpb_global_mmlu_full` | 0.70 | 0.67 | 0.58 | 0.65 | 0.53 | 0.68 | 0.45 | 0.48 | 0.60 | 0.58 | 0.55 | 0.52 | 0.65 | 0.53 | 0.62 |
| `bbpb_include_base_44` | **0.75** | 0.68 | 0.60 | 0.57 | 0.52 | 0.63 | 0.47 | 0.60 | 0.57 | 0.53 | 0.63 | 0.43 | 0.62 | 0.53 | 0.60 |
| `arc` | 0.51 | 0.41 | 0.46 | 0.57 | 0.47 | 0.64 | 0.62 | 0.66 | 0.66 | 0.47 | 0.68 | 0.66 | 0.51 | 0.51 | 0.73 |
| `bbpb_belebele` | 0.47 | 0.60 | 0.61 | 0.58 | 0.43 | 0.65 | 0.51 | 0.55 | 0.52 | 0.68 | 0.57 | 0.48 | 0.66 | 0.61 | 0.58 |
| `truthfulqa_mc2` | 0.74 | 0.63 | 0.47 | 0.48 | 0.44 | 0.71 | 0.45 | 0.62 | 0.40 | 0.51 | 0.67 | 0.53 | 0.51 | 0.55 | 0.61 |
| `bbpb_rf_cultural_bench_easy` | 0.62 | 0.60 | 0.37 | 0.48 | 0.66 | 0.65 | 0.48 | 0.63 | 0.47 | 0.47 | 0.57 | 0.66 | 0.47 | 0.36 | 0.56 |
| `include_v2_en` | 0.33 | 0.42 | 0.58 | 0.51 | 0.57 | 0.57 | 0.61 | 0.59 | 0.60 | 0.46 | 0.44 | 0.56 | 0.67 | 0.61 | 0.53 |
| `bbpb_cultural_bench_easy` | 0.62 | 0.57 | 0.42 | 0.59 | 0.55 | 0.49 | 0.53 | 0.49 | 0.62 | 0.63 | 0.65 | 0.41 | 0.60 | 0.39 | 0.46 |
| `xcopa` | 0.53 | 0.33 | 0.47 | 0.53 | 0.31 | 0.56 | 0.68 | 0.58 | 0.58 | 0.53 | 0.47 | 0.63 | 0.47 | 0.49 | 0.70 |
| `bbpb_include_v2_en` | 0.47 | 0.63 | 0.49 | 0.55 | 0.69 | 0.52 | 0.51 | 0.47 | 0.44 | 0.61 | 0.52 | 0.53 | 0.39 | 0.42 | 0.59 |
| `xnli` | 0.55 | 0.41 | 0.62 | 0.50 | 0.56 | 0.46 | 0.59 | 0.40 | 0.38 | 0.60 | 0.62 | 0.51 | 0.47 | 0.51 | 0.40 |
| `rf_m_mmlu` | 0.35 | 0.38 | 0.56 | 0.45 | 0.48 | 0.51 | 0.50 | 0.55 | 0.53 | 0.43 | 0.57 | 0.59 | 0.43 | 0.52 | 0.68 |
| `global_mmlu_full` | 0.49 | 0.45 | 0.40 | 0.51 | 0.40 | 0.62 | 0.68 | 0.42 | 0.51 | 0.55 | 0.28 | 0.47 | 0.50 | 0.58 | 0.55 |
| `rf_global_mmlu_full` | 0.57 | 0.41 | 0.54 | 0.44 | 0.49 | 0.24 | 0.31 | 0.47 | 0.58 | 0.71 | 0.55 | 0.47 | 0.49 | 0.38 | 0.72 |
| `rfgm_belebele` | 0.49 | 0.42 | 0.45 | 0.38 | 0.61 | 0.55 | 0.55 | 0.55 | 0.53 | 0.57 | 0.38 | 0.47 | 0.46 | 0.50 | 0.41 |
| `include_base_44` | 0.39 | 0.50 | 0.68 | 0.31 | 0.45 | 0.68 | 0.44 | 0.56 | 0.43 | 0.50 | 0.46 | 0.44 | 0.41 | 0.55 | 0.49 |
| `rf_belebele` | 0.37 | 0.45 | 0.62 | 0.44 | 0.38 | 0.62 | 0.47 | 0.62 | 0.54 | 0.52 | 0.49 | 0.42 | 0.51 | 0.38 | 0.43 |
| `belebele` | 0.39 | 0.59 | 0.44 | 0.53 | 0.57 | 0.50 | 0.54 | 0.50 | 0.27 | 0.48 | 0.63 | 0.39 | 0.47 | 0.46 | 0.44 |
| `rf_include_base_44` | 0.38 | 0.37 | 0.42 | 0.42 | 0.42 | 0.35 | 0.46 | 0.62 | 0.62 | 0.43 | 0.49 | 0.35 | 0.63 | 0.59 | 0.65 |
| `rfgm_include_base_44` | 0.42 | 0.41 | 0.46 | 0.41 | 0.45 | 0.31 | 0.45 | 0.51 | 0.38 | 0.51 | 0.53 | 0.46 | 0.66 | 0.57 | 0.59 |
| `global_piqa_parallel_cloze` | 0.49 | 0.53 | 0.47 | 0.42 | 0.32 | 0.47 | 0.59 | 0.60 | 0.28 | 0.43 | 0.38 | 0.23 | 0.62 | 0.39 | 0.42 |
| `cultural_bench_hard` | 0.35 | 0.47 | 0.44 | 0.38 | 0.52 | 0.44 | 0.42 | 0.38 | 0.47 | 0.35 | 0.33 | 0.28 | 0.36 | 0.53 | 0.53 |
| `cultural_bench_easy` | 0.38 | 0.27 | 0.37 | 0.46 | 0.50 | 0.55 | 0.48 | 0.51 | 0.30 | 0.36 | 0.41 | 0.24 | 0.37 | 0.32 | 0.39 |
| `rf_cultural_bench_easy` | 0.32 | 0.45 | 0.35 | 0.38 | 0.34 | 0.33 | 0.29 | 0.61 | 0.26 | 0.51 | 0.30 | 0.42 | 0.43 | 0.38 | 0.27 |

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

| benchmark | 90M→175M | 90M→350M | 90M→600M | 90M→1B | 90M→1.7B | 175M→350M | 175M→600M | 175M→1B | 175M→1.7B | 350M→600M | 350M→1B | 350M→1.7B | 600M→1B | 600M→1.7B | 1B→1.7B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `bpb` | **0.98** | **0.91** | **0.86** | **0.94** | **0.94** | **0.91** | **0.88** | **0.94** | **0.92** | **0.90** | **0.92** | **0.86** | **0.90** | **0.85** | **0.89** |
| `bbpb_xstorycloze` | **0.91** | **0.86** | **0.88** | **0.89** | **0.90** | **0.88** | **0.89** | **0.94** | **0.92** | **0.87** | **0.83** | **0.85** | **0.89** | **0.87** | **0.87** |
| `bbpb_rfgm_belebele_zho_Hant` | **0.89** | **0.84** | **0.81** | **0.81** | **0.80** | **0.86** | **0.83** | **0.83** | **0.83** | **0.83** | **0.86** | **0.84** | **0.87** | **0.86** | **0.82** |
| `bbpb_rfgm_include_base_44` | **0.86** | **0.88** | **0.90** | **0.84** | **0.79** | **0.84** | **0.87** | **0.84** | 0.75 | **0.86** | **0.81** | **0.78** | **0.88** | **0.81** | **0.80** |
| `bbpb_paws` | **0.82** | **0.79** | **0.82** | **0.83** | **0.77** | **0.87** | **0.85** | **0.83** | **0.81** | **0.80** | **0.79** | **0.82** | **0.86** | **0.81** | **0.81** |
| `bbpb_rfgm_belebele_zho_Hans` | **0.88** | **0.89** | **0.87** | **0.81** | **0.80** | **0.83** | **0.87** | **0.82** | 0.74 | **0.87** | **0.86** | 0.73 | **0.83** | 0.73 | 0.68 |
| `bbpb_rf_belebele_zho_Hant` | **0.84** | **0.83** | **0.87** | **0.77** | **0.75** | **0.88** | **0.88** | **0.80** | **0.76** | **0.83** | **0.76** | **0.77** | **0.75** | **0.76** | 0.68 |
| `bbpb_rf_global_mmlu_full` | **0.80** | 0.73 | **0.79** | **0.77** | 0.72 | **0.79** | **0.82** | **0.83** | **0.78** | **0.84** | 0.68 | 0.73 | 0.74 | **0.79** | 0.71 |
| `bbpb_global_piqa_parallel_cloze_cmn_hant` | **0.79** | **0.78** | **0.80** | 0.74 | 0.74 | **0.84** | 0.73 | **0.81** | 0.74 | **0.80** | **0.81** | 0.72 | **0.78** | 0.69 | 0.70 |
| `bbpb_arc` | 0.64 | 0.72 | 0.66 | 0.71 | 0.64 | **0.81** | **0.78** | **0.80** | **0.78** | **0.76** | **0.76** | 0.74 | **0.77** | 0.74 | 0.75 |
| `bbpb_xcopa` | **0.79** | **0.77** | 0.67 | **0.78** | 0.72 | **0.77** | 0.67 | **0.86** | 0.63 | **0.75** | **0.79** | 0.71 | 0.67 | 0.73 | 0.67 |
| `bbpb_rf_include_base_44` | 0.72 | 0.71 | 0.67 | 0.72 | 0.62 | 0.72 | **0.75** | **0.76** | **0.78** | 0.68 | 0.72 | 0.70 | 0.74 | **0.80** | 0.71 |
| `bbpb_xnli` | 0.71 | **0.79** | 0.75 | 0.69 | 0.73 | **0.75** | 0.70 | **0.75** | **0.76** | 0.70 | 0.69 | **0.78** | 0.64 | 0.67 | 0.65 |
| `include_v2_og_chinese_china` | 0.68 | 0.71 | 0.66 | 0.69 | 0.69 | 0.65 | 0.66 | 0.66 | 0.70 | **0.76** | **0.76** | **0.79** | **0.76** | 0.71 | **0.77** |
| `bbpb_global_piqa_parallel_cloze_cmn_hans` | **0.80** | **0.79** | **0.77** | 0.65 | 0.68 | **0.80** | **0.76** | 0.65 | 0.70 | **0.78** | 0.64 | 0.65 | 0.64 | 0.59 | 0.62 |
| `bbpb_rf_belebele_zho_Hans` | **0.79** | **0.80** | **0.81** | 0.69 | 0.55 | **0.82** | **0.85** | 0.70 | 0.59 | **0.79** | 0.64 | 0.63 | 0.68 | 0.56 | 0.53 |
| `bbpb_rf_mmmlu` | 0.63 | 0.68 | 0.68 | 0.64 | 0.55 | 0.63 | 0.73 | **0.76** | 0.64 | 0.68 | 0.66 | 0.65 | 0.74 | 0.72 | 0.64 |
| `bbpb_rf_m_mmlu` | 0.62 | 0.69 | 0.68 | 0.66 | 0.56 | 0.59 | 0.69 | **0.76** | 0.61 | 0.66 | 0.67 | 0.65 | 0.70 | 0.70 | 0.63 |
| `rf_global_mmlu_full` | 0.57 | 0.57 | 0.71 | 0.64 | 0.55 | 0.58 | 0.67 | 0.58 | 0.61 | 0.59 | 0.73 | 0.72 | **0.76** | 0.68 | **0.75** |
| `bbpb_global_mmlu_full` | **0.77** | 0.72 | 0.72 | 0.61 | 0.59 | 0.62 | 0.69 | 0.57 | 0.56 | 0.72 | 0.64 | 0.59 | 0.68 | 0.58 | 0.65 |
| `rfgm_include_base_44` | 0.67 | 0.62 | 0.66 | 0.61 | 0.67 | 0.67 | 0.59 | 0.64 | 0.67 | 0.57 | 0.66 | 0.65 | 0.66 | 0.67 | 0.66 |
| `bbpb_truthfulqa_mc2` | **0.75** | 0.66 | 0.68 | 0.63 | 0.48 | 0.58 | **0.80** | 0.68 | 0.57 | 0.58 | 0.60 | 0.54 | 0.69 | 0.69 | 0.59 |
| `bbpb_include_v2_og_chinese_china` | 0.59 | 0.65 | 0.61 | 0.55 | 0.59 | 0.52 | 0.66 | 0.66 | 0.55 | 0.62 | 0.64 | 0.65 | 0.65 | 0.71 | 0.66 |
| `rf_m_mmlu` | 0.49 | 0.49 | 0.52 | 0.52 | 0.48 | 0.52 | 0.70 | 0.70 | 0.59 | 0.68 | 0.68 | 0.62 | **0.83** | 0.74 | 0.70 |
| `bbpb_include_v2_og_chinese_hong_kong` | 0.57 | **0.75** | 0.67 | 0.59 | 0.56 | 0.63 | 0.60 | 0.70 | 0.53 | 0.64 | 0.65 | 0.56 | 0.62 | 0.62 | 0.57 |
| `bbpb_belebele_zho_Hant` | 0.72 | 0.65 | 0.74 | 0.58 | 0.59 | 0.66 | 0.69 | 0.59 | 0.60 | 0.61 | 0.44 | 0.63 | 0.62 | 0.59 | 0.52 |
| `rf_mmmlu` | 0.61 | 0.47 | 0.51 | 0.54 | 0.53 | 0.54 | 0.65 | 0.70 | 0.61 | 0.58 | 0.66 | 0.73 | 0.68 | 0.65 | 0.69 |
| `bbpb_belebele_zho_Hans` | 0.70 | 0.61 | 0.73 | 0.55 | 0.52 | 0.68 | 0.71 | 0.58 | 0.59 | 0.58 | 0.45 | 0.58 | 0.62 | 0.58 | 0.62 |
| `rf_include_base_44` | 0.53 | 0.59 | 0.63 | 0.61 | 0.68 | 0.56 | 0.55 | 0.63 | 0.49 | 0.67 | 0.64 | 0.66 | 0.56 | 0.70 | 0.58 |
| `bbpb_toksuite` | 0.69 | 0.63 | 0.68 | 0.47 | 0.47 | 0.70 | 0.67 | 0.58 | 0.51 | 0.75 | 0.62 | 0.50 | 0.61 | 0.50 | 0.53 |
| `arc` | 0.51 | 0.67 | 0.42 | 0.55 | 0.47 | 0.58 | 0.58 | 0.65 | 0.69 | 0.57 | 0.66 | 0.55 | 0.60 | 0.66 | 0.64 |
| `bbpb_include_v2_en_chinese_china` | 0.44 | 0.60 | 0.54 | 0.64 | 0.63 | 0.58 | 0.54 | 0.48 | 0.46 | 0.61 | 0.66 | 0.60 | 0.71 | 0.65 | 0.64 |
| `xstorycloze` | 0.52 | 0.48 | 0.53 | 0.60 | 0.49 | 0.69 | 0.56 | 0.64 | 0.60 | 0.59 | 0.71 | 0.64 | 0.57 | 0.51 | 0.63 |
| `bbpb_rf_cultural_bench_easy` | 0.59 | 0.60 | 0.52 | 0.66 | 0.67 | 0.61 | 0.40 | 0.55 | 0.61 | 0.54 | 0.57 | 0.64 | 0.60 | 0.47 | 0.61 |
| `bbpb_include_v2_og_chinese_singapore` | 0.57 | 0.61 | 0.50 | 0.57 | 0.60 | 0.61 | 0.60 | 0.60 | 0.66 | 0.57 | 0.60 | 0.48 | 0.59 | 0.57 | 0.47 |
| `bbpb_include_base_44` | 0.71 | 0.61 | 0.63 | 0.52 | 0.46 | 0.61 | 0.60 | 0.51 | 0.45 | 0.65 | 0.51 | 0.54 | 0.60 | 0.49 | 0.63 |
| `bbpb_include_v2_en_chinese_singapore` | 0.53 | 0.50 | 0.56 | 0.65 | 0.72 | 0.44 | 0.48 | 0.55 | 0.54 | 0.57 | 0.58 | 0.53 | 0.65 | 0.51 | 0.59 |
| `truthfulqa_mc2` | 0.66 | 0.72 | 0.54 | 0.58 | 0.53 | 0.59 | 0.70 | 0.52 | 0.45 | 0.45 | 0.50 | 0.55 | 0.62 | 0.46 | 0.41 |
| `bbpb_cultural_bench_easy` | 0.54 | 0.50 | 0.56 | 0.56 | 0.63 | 0.56 | 0.48 | 0.62 | 0.54 | 0.59 | 0.61 | 0.49 | 0.62 | 0.44 | 0.50 |
| `rfgm_belebele_zho_Hant` | 0.64 | 0.47 | 0.49 | 0.64 | 0.63 | 0.52 | 0.55 | 0.64 | 0.55 | 0.39 | 0.54 | 0.47 | 0.53 | 0.59 | 0.55 |
| `rfgm_belebele_zho_Hans` | 0.67 | 0.60 | 0.49 | 0.59 | 0.61 | 0.47 | 0.31 | 0.50 | 0.57 | 0.40 | 0.47 | 0.55 | 0.67 | 0.58 | 0.63 |
| `bbpb_include_v2_en_chinese_hong_kong` | 0.54 | 0.45 | 0.53 | 0.60 | 0.48 | 0.61 | 0.53 | 0.60 | 0.39 | 0.54 | 0.58 | 0.51 | 0.55 | 0.61 | 0.44 |
| `paws` | 0.60 | 0.57 | 0.46 | 0.48 | 0.54 | 0.58 | 0.56 | 0.57 | 0.57 | 0.61 | 0.47 | 0.53 | 0.53 | 0.42 | 0.39 |
| `include_v2_en_chinese_china` | 0.45 | 0.46 | 0.41 | 0.49 | 0.53 | 0.55 | 0.54 | 0.50 | 0.49 | 0.49 | 0.48 | 0.50 | 0.62 | 0.74 | 0.61 |
| `xwinograd` | 0.43 | 0.46 | 0.51 | 0.59 | 0.38 | 0.49 | 0.65 | 0.51 | 0.56 | 0.55 | 0.44 | 0.50 | 0.56 | 0.58 | 0.46 |
| `rf_belebele_zho_Hant` | 0.41 | 0.50 | 0.46 | 0.48 | 0.56 | 0.38 | 0.54 | 0.56 | 0.52 | 0.55 | 0.47 | 0.53 | 0.61 | 0.55 | 0.50 |
| `bbpb_blend_sample` | 0.59 | 0.49 | 0.50 | 0.43 | 0.60 | 0.42 | 0.51 | 0.46 | 0.53 | 0.63 | 0.42 | 0.46 | 0.38 | 0.59 | 0.50 |
| `global_mmlu_full` | 0.50 | 0.58 | 0.51 | 0.52 | 0.41 | 0.48 | 0.50 | 0.42 | 0.47 | 0.49 | 0.42 | 0.49 | 0.56 | 0.54 | 0.58 |
| `xnli` | 0.42 | 0.52 | 0.67 | 0.47 | 0.51 | 0.44 | 0.46 | 0.52 | 0.51 | 0.49 | 0.46 | 0.46 | 0.63 | 0.41 | 0.40 |
| `blend_sample` | 0.55 | 0.39 | 0.41 | 0.55 | 0.47 | 0.47 | 0.42 | 0.60 | 0.62 | 0.40 | 0.45 | 0.67 | 0.38 | 0.49 | 0.48 |
| `include_v2_en_chinese_hong_kong` | 0.38 | 0.51 | 0.43 | 0.54 | 0.39 | 0.53 | 0.41 | 0.40 | 0.61 | 0.58 | 0.42 | 0.58 | 0.51 | 0.53 | 0.48 |
| `belebele_zho_Hans` | 0.32 | 0.44 | 0.44 | 0.53 | 0.39 | 0.49 | 0.47 | 0.49 | 0.38 | 0.45 | 0.53 | 0.49 | 0.60 | 0.55 | 0.65 |
| `xcopa` | 0.40 | 0.49 | 0.36 | 0.41 | 0.41 | 0.42 | 0.51 | 0.54 | 0.63 | 0.44 | 0.51 | 0.43 | 0.49 | 0.58 | 0.60 |
| `cultural_bench_hard` | 0.54 | 0.44 | 0.48 | 0.41 | 0.66 | 0.40 | 0.47 | 0.53 | 0.49 | 0.44 | 0.32 | 0.46 | 0.60 | 0.49 | 0.45 |
| `belebele_zho_Hant` | 0.37 | 0.56 | 0.42 | 0.46 | 0.48 | 0.41 | 0.41 | 0.40 | 0.42 | 0.50 | 0.53 | 0.54 | 0.57 | 0.51 | 0.54 |
| `rf_belebele_zho_Hans` | 0.33 | 0.52 | 0.47 | 0.52 | 0.49 | 0.42 | 0.46 | 0.44 | 0.41 | 0.53 | 0.45 | 0.39 | 0.56 | 0.47 | 0.59 |
| `include_base_44` | 0.58 | 0.39 | 0.54 | 0.54 | 0.41 | 0.33 | 0.51 | 0.44 | 0.46 | 0.45 | 0.52 | 0.47 | 0.44 | 0.42 | 0.51 |
| `cultural_bench_easy` | 0.43 | 0.46 | 0.43 | 0.36 | 0.50 | 0.62 | 0.53 | 0.35 | 0.54 | 0.45 | 0.33 | 0.44 | 0.55 | 0.46 | 0.38 |
| `include_v2_og_chinese_hong_kong` | 0.49 | 0.58 | 0.42 | 0.50 | 0.38 | 0.33 | 0.42 | 0.45 | 0.46 | 0.44 | 0.44 | 0.39 | 0.52 | 0.34 | 0.33 |
| `global_piqa_parallel_cloze_cmn_hans` | 0.34 | 0.43 | 0.35 | 0.48 | 0.50 | 0.36 | 0.46 | 0.40 | 0.33 | 0.44 | 0.45 | 0.46 | 0.50 | 0.49 | 0.41 |
| `global_piqa_parallel_cloze_cmn_hant` | 0.46 | 0.43 | 0.31 | 0.38 | 0.25 | 0.49 | 0.34 | 0.32 | 0.37 | 0.34 | 0.50 | 0.56 | 0.54 | 0.38 | 0.58 |
| `toksuite` | 0.38 | 0.35 | 0.51 | 0.32 | 0.36 | 0.41 | 0.33 | 0.34 | 0.39 | 0.40 | 0.38 | 0.35 | 0.42 | 0.41 | 0.66 |
| `rf_cultural_bench_easy` | 0.36 | 0.44 | 0.27 | 0.35 | 0.41 | 0.37 | 0.44 | 0.35 | 0.39 | 0.29 | 0.43 | 0.32 | 0.38 | 0.39 | 0.38 |
| `include_v2_og_chinese_singapore` | 0.27 | 0.36 | 0.25 | 0.38 | 0.33 | 0.37 | 0.44 | 0.39 | 0.23 | 0.34 | 0.35 | 0.31 | 0.48 | 0.39 | 0.37 |
| `include_v2_en_chinese_singapore` | 0.21 | 0.30 | 0.43 | 0.30 | 0.35 | 0.27 | 0.32 | 0.29 | 0.27 | 0.29 | 0.34 | 0.32 | 0.47 | 0.23 | 0.31 |

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
