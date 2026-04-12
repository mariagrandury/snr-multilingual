---
theme: scholarly
layout: cover
transition: slide-left
footerMiddle: Signal-Aware Multilingual Evaluation
description: Signal-Aware Framework for Multilingual Language Model Evaluation
aspectRatio: 4/3
lang: en
themeConfig:
  colorTheme: classic-blue
  fontTheme: contemporary
  colorMode: dark
  sectionMode: dark
authors:
  - name: María Grandury
    institution: SomosNLP / EPFL
---

# Signal-Aware Framework for Multilingual LM Evaluation

<!--
Training multilingual language models costs hundreds of thousands of euros. Throughout training, we constantly make decisions — data mixtures, hyperparameters — guided by benchmark evaluations. But here's the problem: not all benchmarks give us equally useful feedback. Some are noisy, some are redundant, and some just don't correlate with what we actually care about. This project asks: which benchmarks should we actually trust, and when?
-->

---

## layout: section

# The Problem

## Evaluation is expensive and often uninformative

<!--
Let me frame the core problem we're tackling.
-->

---

title: The Evaluation Dilemma
subtitle: Coverage vs. Efficiency

---

## The Evaluation Dilemma

<br/>

### Training multilingual LMs requires constant evaluation decisions

<br/>

**But benchmarks vary wildly in diagnostic value:**

- Some tasks are **sensitive** to meaningful changes in training
- Others show **high variance**, **redundancy**, or **weak correlation** with downstream goals
- Improvements on certain benchmarks **may not reflect real progress**

<br/>

<Block type="warning" title="The status quo">

Practitioners evaluate on large benchmark suites with diminishing returns — or use small subsets with noisy, misaligned signals.

</Block>

<!--
When you're training a multilingual model, you evaluate constantly — after every data mixture change, every hyperparameter tweak. But running 40 benchmarks is expensive, and many of those benchmarks are telling you the same thing, or worse, telling you nothing useful at all. You end up spending compute on evaluations that don't help your decisions.
-->

---

layout: focus
color: blue
icon: 🎯

---

# Which benchmarks provide reliable signal at each stage of multilingual model training?

<!--
This is our core research question. Not just "which benchmarks are good" but specifically which ones give reliable, actionable signal at pre-training, mid-training, and post-training — because the answer changes depending on where you are in the pipeline.
-->

---

## layout: section

# The Framework

## Signal-to-Noise Ratio for Benchmarks

<!--
Let me show you the framework we're building on and extending.
-->

---

title: Signal-to-Noise Ratio
subtitle: Quantifying benchmark reliability

---

## Signal-to-Noise Ratio (SNR)

<br/>

**Signal** — How well a benchmark separates models of similar scale:

$$\text{Signal}(b) = \text{Var}_{\text{models}}\big[\bar{s}_b(m)\big]$$

<br/>

**Noise** — Variability across checkpoints or stochastic runs:

$$\text{Noise}(b) = \mathbb{E}_m\big[\text{Var}_{\text{runs}}[s_b(m)]\big]$$

<br/>

**SNR** — The ratio that tells us if a benchmark is worth running:

$$\text{SNR}(b) = \frac{\text{Signal}(b)}{\text{Noise}(b)}$$

<br/>

<Block type="success" title="Takeaway">

High SNR → benchmark reliably distinguishes models → better training decisions

</Block>

<!--
The key insight from Heineman et al. is simple but powerful. Signal measures how much benchmark scores vary across different models — you want benchmarks that can tell models apart. Noise measures how much scores fluctuate due to randomness — training seeds, checkpoint selection. The ratio gives you a single number: is this benchmark telling you something real, or just showing you noise? We're extending this framework from English-only to multilingual settings.
-->

---

title: Beyond SNR
subtitle: Decision-theoretic metrics

---

## Decision Accuracy and Scaling-Law Error

<br/>

### Decision Accuracy

Does the benchmark correctly identify which model is better?

$$\text{DA}(b) = P\big(\text{rank}_b(m_1, m_2) = \text{rank}_{\text{true}}(m_1, m_2)\big)$$

<br/>

### Scaling-Law Error

Can we extrapolate performance from small to large models?

$$\text{SLE}(b) = \left| \hat{s}_b^{\text{large}} - s_b^{\text{large}} \right|$$

<br/>

<Block type="success" title="The trifecta">

High SNR + high decision accuracy + low scaling-law error = a benchmark you can trust to guide training.

</Block>

<!--
SNR alone isn't enough. We also measure decision accuracy — does the benchmark actually get the ranking right when comparing two models? And scaling-law error — can you use small model evaluations to predict how a large model will perform? Together, these three metrics tell you if a benchmark is reliable, accurate, and predictive. That's the trifecta for practical evaluation during training.
-->

---

## layout: section

# Research Plan

## From models to recommendations

<!--
Here's how we're executing this.
-->

---

layout: timeline
title: Research Timeline
items:

- year: "Phase 1"
  title: "Model Training"
  description: "36 small models (100M–1B), 3 data mixtures, 3 seeds"
- year: "Phase 2"
  title: "Evaluation"
  description: "40 multilingual benchmarks on custom + open-source models"
- year: "Phase 3"
  title: "SNR Analysis"
  description: "Compute signal, noise, SNR, decision accuracy per stage"
- year: "Phase 4"
  title: "INCLUDE Analysis"
  description: "Optimal subsets for 100+ countries across training stages"
- year: "Phase 5"
  title: "Dissemination"
  description: "NeurIPS paper, HF benchmark subsets, open-source toolkit"

---

<!--
Five phases in nine weeks. First, we pretrain 36 small models across four sizes, three multilingual data mixtures from FineWeb and FineWeb2, and three seeds. Then we evaluate these plus open-source model families — Apertus, OLMo, SmolLM — on 40 multilingual benchmarks covering understanding, QA, reasoning, and regional knowledge. In Phase 3, we compute all our metrics. Phase 4 focuses specifically on INCLUDE, our benchmark for regional knowledge across 100+ countries. And we wrap up with a NeurIPS submission and open-source deliverables.
-->

---

title: Model and Benchmark Suites
subtitle: Scale and coverage

---

## Experimental Setup

<br/>

### Model Suite

| Component         | Details                                                  |
| ----------------- | -------------------------------------------------------- |
| **Custom models** | 4 sizes (100M–1B) x 3 mixtures x 3 seeds = **36 models** |
| **Data sources**  | FineWeb (EN) + FineWeb2 (200+ languages)                 |
| **Open-source**   | Apertus, OLMo, SmolLM (1B–70B, intermediate checkpoints) |

<br/>

### Benchmark Suite — 40 multilingual benchmarks

| Category                  | Benchmarks                            |
| ------------------------- | ------------------------------------- |
| **Cross-lingual**         | XNLI, XCOPA, XStoryCloze, Belebele    |
| **QA & Reasoning**        | XQuAD, MGSM, XLSum                    |
| **Regional knowledge**    | INCLUDE (100+ countries), Global MMLU |
| **Instruction-following** | IFEval                                |

<!--
The experimental design is comprehensive. On the model side, 36 custom-trained models give us controlled comparisons — same architecture, different data and seeds — plus open-source families that give us scale diversity up to 70B parameters with intermediate checkpoints. On the benchmark side, 40 tasks spanning cross-lingual understanding, QA, reasoning, regional knowledge, and instruction following. We evaluate with logprobs in both 0-shot and 5-shot setups to maximize comparability.
-->

---

title: "Phase 1: Model Training"
subtitle: "36 models across 4 scales"

---

## Phase 1: Model Training

<br/>

### Architecture — 4 scales, LLaMA-style

| Label    | Layers | d_model | Heads | KV Heads | Non-emb params |
| -------- | ------ | ------- | ----- | -------- | -------------- |
| **175M** | 16     | 1024    | 16    | 4        | 0.176B         |
| **350M** | 20     | 1280    | 20    | 5        | 0.344B         |
| **600M** | 24     | 1536    | 24    | 6        | 0.595B         |
| **1B**   | 28     | 1792    | 28    | 7        | 0.944B         |

<br/>

### Data Mixtures — FineEdu2-DCLM + FineWeb2

| Mixture   | FineEdu2-DCLM (EN) | FineWeb2 (multilingual) |
| --------- | ------------------ | ----------------------- |
| **Mix A** | 30%                | 70%                     |
| **Mix B** | 60%                | 40%                     |
| **Mix C** | 90%                | 10%                     |

<br/>

<Block type="info" title="Total">

4 sizes x 3 mixtures x 3 seeds = **36 models** with full checkpoint history

</Block>

<!--
In Phase 1, we pretrain 36 small models. Four sizes from 175M to 1B parameters, all using grouped query attention with a GQA ratio of 4. Each size is trained on three data mixtures that vary the ratio of English data from FineEdu2-DCLM to multilingual data from FineWeb2 — from mostly multilingual at 30-70, to balanced at 60-40, to mostly English at 90-10. Each configuration runs with three seeds, giving us 36 models total with full checkpoint histories for our SNR analysis.
-->

---

title: "Phase 2: Model Evaluation"
subtitle: "From 100M to 70B across training stages"

---

## Phase 2: Model Evaluation

<br/>

| Stage             | Models                                                |
| ----------------- | ----------------------------------------------------- |
| **Pretraining**   | Custom 100M, 300M, 500M, 1B (3 mixtures each)         |
|                   | Apertus 1B, Apertus 3B                                |
| **Midtraining**   | SmolLM3 3B Base                                       |
|                   | Apertus 8B Base, Apertus 70B Base                     |
|                   | OLMo3 7B Base                                         |
| **Post-training** | Apertus 0.6B / 1.7B Distilled                         |
|                   | Apertus 1.7B Distilled SFT, 8B Instruct, 70B Instruct |
|                   | SmolLM3 3B                                            |
|                   | OLMo3 7B SFT, 7B DPO, 7B Instruct (RLVR)              |

<br/>

<Block type="info" title="Coverage">

Custom models for controlled pretraining analysis + open-source families for mid/post-training coverage up to **70B**

</Block>

<!--
In Phase 2, we evaluate across all three training stages. For pretraining, our 36 custom models give us controlled comparisons — same architecture, different data and seeds — plus Apertus 1B and 3B. For midtraining, we use base checkpoints from SmolLM3, Apertus, and OLMo3, ranging from 3B to 70B. For post-training, we compare distilled, SFT, DPO, and RLVR variants across the same families. This lets us analyze which benchmarks are informative at each specific stage.
-->

---

title: "Phase 3: SNR Analysis"
subtitle: "Signal, noise, and decision metrics"

---

## Phase 3: SNR Computation and Analysis

<br/>

For each of the **40 benchmarks**, compute:

<br/>

| Metric                | Question it answers                                      |
| --------------------- | -------------------------------------------------------- |
| **Signal**            | Does this benchmark separate models meaningfully?        |
| **Noise**             | How much do scores fluctuate due to randomness?          |
| **SNR**               | Is the signal worth the evaluation cost?                 |
| **Decision Accuracy** | Does it correctly rank model pairs?                      |
| **Scaling-Law Error** | Can small model results predict large model performance? |

<br/>

### Key analysis dimensions

- **Stage-specific:** Which benchmarks have high SNR at pre/mid/post-training?
- **Subtask-level:** Which subtasks within a benchmark carry the signal?
- **Efficiency frontier:** Minimum benchmark subset for reliable decisions

<!--
In Phase 3, we compute our five metrics for every benchmark at every training stage. The goal is to identify stage-specific evaluation strategies. A benchmark that's highly informative during pretraining may be useless at post-training, and vice versa. We also go beyond benchmark-level analysis to examine individual subtasks — because within a benchmark like XNLI, some language pairs may be far more informative than others.
-->

---

title: "Phase 4: INCLUDE Analysis"
subtitle: "Regional knowledge across 100+ countries"

---

## Phase 4: Multilingual Benchmark Analysis — INCLUDE

<br/>

<Block type="info" title="INCLUDE Benchmark">

Internally developed benchmark evaluating **regional knowledge** across **100+ countries** in local languages and scripts.

</Block>

<br/>

### Analysis goals

- Apply SNR framework to **country-level subtasks**
- Identify which country subsets are most diagnostic at each training stage
- Produce **optimal evaluation subsets** that maximize signal while minimizing cost

<br/>

### Deliverable

Publish recommended INCLUDE subsets on Hugging Face:

- **Pretraining subset** — countries that track early capability development
- **Midtraining subset** — countries sensitive to data mixture changes
- **Post-training subset** — countries that reflect instruction-following quality

<!--
Phase 4 focuses on INCLUDE, our benchmark for regional knowledge. With over 100 countries, running the full benchmark is expensive. By applying SNR analysis at the country level, we can identify which country subsets are most informative at each training stage. The practical output: published subsets on Hugging Face so practitioners can evaluate efficiently without sacrificing diagnostic quality.
-->

---

title: "Phase 5: Dissemination"
subtitle: "Open science deliverables"

---

## Phase 5: Dissemination

<br/>

**Paper** — NeurIPS submission (May 7th)

- SNR scores and stage-specific recommendations for 40 multilingual benchmarks
- Analysis of benchmark reliability across pre/mid/post-training

<br/>

**INCLUDE Subset Recommendations** — on Hugging Face

- Optimal country subsets for each training stage
- Coverage of 100+ countries

<br/>

**Open-Source Toolkit** — Python package

- Compute SNR on your own benchmarks and models
- Fork and extension of the original SNR codebase

<br/>

<Block type="success" title="Open science">

Everything open-source — enabling the community to make better evaluation decisions for multilingual models.

</Block>

<!--
Three concrete deliverables. The NeurIPS paper with comprehensive SNR analysis. On Hugging Face, recommended INCLUDE subsets per training stage. And an open-source Python toolkit so anyone can compute SNR for their own benchmarks and models.
-->

---

## layout: end
