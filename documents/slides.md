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
  - Motivation
  - Key Concepts & Previous Results
  - Proposed Methodology
  - Open Questions
---

---
layout: section
---

# The Problem

Training multilingual LMs requires constant evaluation decisions,

but evaluation is expensive and often uninformative.

---
layout: bullets
title: The Problem
subtitle: Lack of Benchmark reliability
icon: "⚠️"
---

Training LMs has a **high cost** and requires **constant decisions** (data mixtures, hyperparameters, etc.) guided by benchmark evaluations. However,

## Not all benchmarks provide informative signals

- **High variance**: Scores fluctuate across runs
- **Redundancy**: Multiple benchmarks measure the same thing
- **Weak correlation**: Improvements don't reflect real progress
- **Cost**: Large suites are expensive to run frequently

---
layout: bullets
title: The Problem
subtitle: Why Multilingual?
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

## Which (subsets of) benchmarks provide reliable signal at each stage of multilingual model training?

---
layout: section
---

# Methodology

The Signal-and-Noise Framework

---
title: Key concepts
subtitle: Signal and Noise
---

<Block type="success" title="Signal (Relative Dispersion / Normalized Max. Difference)">

How well a benchmark $b$ separates a pair of models $m_j, m_k$ of similar scale trained on different data:

$$\text{Rel. Dispersion}(M) = \frac{\max_{j,k} |m_j - m_k|}{\bar{m}}$$

</Block>

<Block type="success" title="Noise (Relative Std. Dev.)">

Variability across the final $n$ training checkpoints of a model $m$:

$$\text{Rel. Std.}(m) = \frac{\sqrt{\frac{1}{n-1} \sum_{i=1}^{n}(m_i - \bar{m})^2}}{\bar{m}}$$

</Block>

$$\text{SNR} = \frac{\text{Rel. Dispersion}(\text{final train checkpoint})}{\text{Rel. Std.}(\text{final $n$ train checkpoint})}$$

<!--
Signal:
- Signal = normalized maximum difference between any pair of models
- Measured as **relative dispersion** of scores across training mixtures
- For a fixed benchmark and model size
- Higher signal = benchmark separates mixtures more clearly

Signal captures how much a benchmark's scores vary when you change the training data mixture but keep the model size fixed. If all mixtures produce the same score on a benchmark, then the benchmark has no signal — it can't help you choose between mixtures. We use relative dispersion, which the preliminary results confirm is better than relative spread (R = 0.811 vs R = 0.791).

The key insight from Heineman et al. is simple but powerful. Signal measures how much benchmark scores vary across different models — you want benchmarks that can tell models apart. Noise measures how much scores fluctuate due to randomness — training seeds, checkpoint selection. The ratio gives you a single number: is this benchmark telling you something real, or just showing you noise? We're extending this framework from English-only to multilingual settings.
-->

---
title: Key concepts
subtitle: Decision Accuracy and Scaling-Law Error
---

<Block type="success" title="Decision Accuracy">

For all pairs of small models $(s_a, s_b)$ trained on datasets ($a$, $b$) and their large versions $(m_a, m_b)$, does the ranking for task $B$ hold?

$$\text{DA} = \frac{1}{|\mathcal{P}|} \sum_{(a,b) \in \mathcal{P}} \mathbb{1}\big[\text{sign}(B(s_a) - B(s_b)) = \text{sign}(B(m_a) - B(m_b))\big]$$

</Block>


<Block type="success" title="Scaling-Law Prediction Error">

Can we extrapolate performance from small to large models?

$$\text{Prediction Error} = \frac{|\text{Measured Value} - \text{True Value}|}{|\text{True Value}|}$$

</Block>

- **Heineman et al. (2025):** SNR is strongly correlated with decision accuracy
- SNR predicts how reliably a benchmark will transfer rankings across scales

---
layout: section
---

# Insights from Signal-and-Noise Paper

by Heineman et al. from AllenAI

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
- **Noise predicts scaling law error** ($R = 0.653$), noisier benchmarks have less reliable extrapolation
- **Filtering subtasks by SNR** yields subsets that outperform full benchmarks (e.g., 16/57 MMLU subtasks → +2.6% DA)
- **Averaging checkpoint scores** reduces noise and improves decision accuracy (+2.4% on 30-task avg.)
- **Bits-per-byte (BPB)** improves DA over accuracy in 90% benchmarks, especially for math and code tasks

---
layout: section
---

# Preliminary Results

by Éléonore, Clara, Antoine

<!--
layout: timeline
title: Experiments
items:
  - year: "Exp 1"
    title: "Reproduction of the English SNR Framework"
    description: ""
  - year: "Exp 2"
    title: "Benchmark Noise, A More Practical Noise Metric"
    description: ""
  - year: "Exp 3"
    title: "Extension to Multilingual Downstream Tasks"
    description: ""
  - year: "Exp 4"
    title: "BPB on Raw Text Corpora"
    description: ""
-->

---
layout: default
title: Experimental Setup
subtitle: Models & Benchmarks
---

### Models

- Allen AI DataDecide models, 4M-1B params
- All mixtures are **English-centric** (inc. DCLM, Dolma, Falcon, FineWeb)

### Benchmarks

| Experiment                                 | Benchmarks                                                                                   |
| ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| **Reproduction experiment** (English)      | ARC, HellaSwag, MMLU, WinoGrande, PIQA, OpenBookQA, BoolQ, SocialIQA, CSQA                   |
| **Multilingual downstream tasks**          | BanglaMMLU, BertaQA, Belebele,  BlimpNL, CaBBQ, Click, CrowsPairs, DarijaHellaswag, EgyHellaswag, EUS Proficiency, EUS Reading, EUS Trivia, TruthfulQA-multi, TurBlimp-Core, TurkishMMLU, xCOPA, XStoryCloze, XNLI-eu, xWinograd |
| **BPB evaluation sets** (raw text corpora) | Flores+ (10 subsets, ~220 languages) and Wiki40B (5 subsets, ~40 languages)                  |

<!--
Screenshot with benchmark details in documents/public.
-->

---
layout: compare
title: "Experiment 2: Benchmark Noise"
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


---
layout: image-left
image: /snr_preliminary_figure_3.png
ratio: "2:2"
fit: contain
title: "Experiment 2: Benchmark Noise"
subtitle: Correlation between SNR and Decision Accuracy
---

- Benchmark noise correlates with checkpoint noise ($R = 0.854$).
- Using it in SNR **improves** prediction of decision accuracy:

| SNR noise metric             | R         | R²        |
| ---------------------------- | --------- | --------- |
| Checkpoint noise             | 0.760     | 0.578     |
| **Benchmark noise (k=5)**    | **0.808** | **0.653** |

- Computable from a **single evaluation run** of any model (no checkpoints needed)
- Robust across model sizes (150M–1B)

> Not just easier to compute, also more predictive of decision accuracy

<!--
Benchmark noise (k=5 folds, relative std dev across folds) correlates with checkpoint noise at R = 0.854 (Figure 11), confirming both capture the same underlying instability. Crucially, using benchmark noise in SNR yields a STRONGER correlation with decision accuracy (R = 0.808 vs R = 0.760, Figure 15) — not just a convenient approximation but a better metric. Computable from a single evaluation run on any model, no intermediate checkpoints needed.
-->

---
layout: image-left
image: /snr_preliminary_figure_7.png
ratio: "2:3"
fit: contain
title: "Experiment 3: Multilingual Downstream Tasks"
subtitle: "Framework reliability depends on model competence"
---

| Task subset                 | R     | R²    |
| --------------------------- | ----- | ----- |
| English-only tasks          | 0.594 | 0.353 |
| All non-English tasks       | 0.045 | 0.002 |
| Non-English (excl. 3 worst) | 0.293 | 0.086 |

Small English-first models perform **near-randomly** on underrepresented languages → uninformative rankings.

> Framework reliability is **conditional on model competence**. The SNR framework doesn't fail for multilingual settings in general, it fails when proxy models lack linguistic competence. → Need multilingual models.

<!--
When we apply the framework to multilingual downstream tasks, the correlation essentially disappears for non-English tasks (R = 0.045). The 3 most unstable benchmarks — click, bangla_mmlu, belebele — cover languages most underrepresented in training data. Removing them raises the non-English correlation to R = 0.293.

The DataDecide models are trained on English web data. When evaluated on tasks in Bangla, Swahili, or other underrepresented languages, the small models essentially guess randomly. Random guessing produces unstable, meaningless rankings.

This is perhaps the most important finding: the framework is sound, but its applicability depends on models having at least some competence in the target languages. The next step is to apply this to models trained on multilingual data — which is exactly what the research proposal plans.
-->

---
layout: default
title: "Experiment 4: BPB on Raw Text Corpora"
subtitle: "Bypassing the instruction-following bottleneck"
---

**Accuracy** is discrete and unreliable when models don't understand English-centric prompts. **Bits-per-byte (BPB)** is continuous, requires no instruction-following, and is more stable.

| Metric                       | R         | R²        | Decision Accuracy |
| ---------------------------- | --------- | --------- | ----------------- |
| Non-English tasks (accuracy) | 0.045     | 0.002     | Near-random       |
| **BPB on raw corpora**       | **0.307** | **0.094** | **0.77 – 0.96**   |

- Wiki40B reaches decision accuracy of **0.96**
- Low signal (0.05–0.16) expected: all DataDecide mixtures are English web variants
- With diverse multilingual mixtures → expect much higher signal

> Even small proxy models produce rankings that closely agree with 1B model rankings

<!--
Given the limitations of accuracy on downstream tasks, we explore bits-per-byte on raw multilingual text (Flores+ and Wiki40B). BPB measures how well a model predicts the next byte, without needing to understand a prompt format. This bypasses the instruction-following bottleneck that makes accuracy unreliable for non-English tasks.

The results are striking. While multilingual downstream tasks showed near-random decision accuracy, BPB on raw corpora shows that even small models produce highly reliable rankings (0.77–0.96). The absolute SNR correlation R = 0.307 is still modest, but it's a major improvement over R = 0.045. The low signal is expected since all DataDecide mixtures are English web crawl variants. With genuinely multilingual mixtures, we expect much higher signal and stronger correlations.
-->

---
layout: bullets
title: Preliminary Analysis
subtitle: Key Takeaways
icon: "→"
---

- **Framework reproduces** on English benchmarks ✅
- **Benchmark noise** is more practical AND more predictive than checkpoint noise ✅
- **Multilingual extension weakens** with English-first models (model limitation, not framework limitation) ⚠️
- **BPB on raw corpora** yields higher decision accuracy and better SNR correlation than accuracy on downstream tasks ✅

---
layout: section
---

# Proposed Methodology

---
layout: timeline
title: Research Timeline
items:
  - year: "Phase 1"
    title: "Model Training ⏳"
    description: "36 small models: 4 sizes (100M–1B), 3 data mixtures, 3 seeds"
  - year: "Phase 2"
    title: "Model Evaluation ⏳"
    description: "40 multilingual benchmarks for pre-/mid-/post-training"
  - year: "Phase 3"
    title: "SNR Framework ✅"
    description: "Framework implemented, validated on original AllenAI models"
  - year: "Phase 4"
    title: "INCLUDE Analysis"
    description: "Optimal subsets for 100+ countries across training stages"
  - year: "Phase 5"
    title: "Dissemination"
    <!-- description: "Paper, HF benchmark subsets, open-source toolkit" -->
---

---
title: "Phase 1: Model Training"
subtitle: "36 small models across 4 scales with 3 data mixtures"
---

### Architecture

| Label    | Layers | d_model | Head dim | Heads | KV Heads | FFW Mult | Non-emb Params |
| -------- | ------ | ------- | -------- | ----- | -------- | ---- | --------- |
| **175M** | 16     | 1024    | 64 | 16    | 4        | 4 | 0.176B         |
| **350M** | 20     | 1280    | 64 | 20    | 5        | 4 | 0.344B         |
| **600M** | 24     | 1536    | 64 | 24    | 6        | 4 | 0.595B         |
| **1B**   | 28     | 1792    | 64 | 28    | 7        | 4 | 0.944B         |

Differences with Apertus:
- Tied embeddings (128k vocab, many params dedicated to embeddings)
- No goldfish loss (memorization not an issue at these scales)
- No cross-document attention masking

---
title: "Phase 1: Model Training"
subtitle: "36 small models across 4 scales with 3 data mixtures"
---

### Data Mixtures

- English data: FineEdu2-DCLM
- Multilingual data: FineWeb2
  - Apertus' high-quality filter
  - Use original naturally occurring language distribution
- Mixtures: 30%-70%, 60%-40%, 90%-10%
- Tokens: 100B in total

<!-- Include languages_estimates.json? -->

---
title: "Phase 2: Model Evaluation"
subtitle: "Model Suite"
---

| Stage             | < 1B | 3B | 7-8B | 70 B  |
| ----------------- | ------------------------------------ | --- | --- | --- |
| **Pretraining**   | Custom 175M, 350M, 600M, 1B (3 mixtures each)  | | | | |
|                   | Apertus 1B | Apertus 3B                            | | |
| **Midtraining**   |  | SmolLM3 3B Base                                 | | |
|                   | | | Apertus 8B Base |  Apertus 70B Base                   | | 
|                   | | | OLMo3 7B Base                                         |
| **Post-training** | Apertus 0.6B / 1.7B Distilled                       |  |
|                   | Apertus 1.7B Distilled SFT | | 8B Instruct | 70B Instruct |
|                   | | SmolLM3 3B                                        | | |
|                   | | | OLMo3 7B SFT, 7B DPO, 7B Instruct (RLVR)    |  |

Custom models for controlled pretraining analysis + open-source families for mid/post-training coverage up to 70B.

---
title: "Phase 2: Model Evaluation"
subtitle: "Evaluation Suite"
---

| Category                  | Benchmarks                            |
| ------------------------- | ------------------------------------- |
| **Cross-lingual**         | XNLI, XCOPA, XStoryCloze, Belebele    |
| **QA & Reasoning**        | XQuAD, MGSM, XLSum                    |
| **Regional knowledge**    | INCLUDE (100+ countries), Global MMLU |
| **Instruction-following** | IFEval                                |

<!-- Update the list of benchmarks, including languages and training stage -->

---
title: "Phase 3: SNR Computation and Analysis"
subtitle: "Signal, noise, and decision metrics"
---

| Metric                | Question it answers                                      |
| --------------------- | -------------------------------------------------------- |
| **Signal**            | Does this benchmark separate models meaningfully?        |
| **Noise**             | How much do scores fluctuate due to randomness?          |
| **SNR**               | Is the signal worth the evaluation cost?                 |
| **Decision Accuracy** | Does it correctly rank model pairs?                      |
| **Scaling-Law Error** | Can small model results predict large model performance? |

### Key analysis dimensions

- **Stage-specific:** Which benchmarks have high SNR at pre/mid/post-training?
- **Sample-level:** Which samples within a benchmark carry the signal?
- **Efficiency frontier:** Minimum benchmark subset for reliable decisions

---
title: "Phase 4: Thorough Analysis on INCLUDE"
subtitle: "Regional knowledge across 100+ countries"
---

<Block type="info" title="INCLUDE v2">

Benchmark evaluating **regional knowledge** across **120 country-language** pairs.

</Block>

### Analysis goals

- Apply SNR framework to country-level subsets
- Identify which sample clusters have higher SNR at each training stage
- Produce **optimal evaluation subsets** that maximize signal while minimizing cost

---
title: "Phase 5: Dissemination"
subtitle: "Open science deliverables"
---

### **Paper** contributions

- SNR scores and stage-specific recommendations for 40 multilingual benchmarks
- Analysis of benchmark reliability across pre/mid/post-training

<br/>

### **INCLUDE Subsets** on Hugging Face

- High-signal language-country subsets for each training stage

<br/>

### Open-Source Toolkit

- Modular Python package: metrics, data loading, visualization
- Compute SNR on your own benchmarks and models
- Stage-specific automated analysis and recommendations

---
layout: section
---

# Open Questions

Please share your wisdom!

---
layout: bullets
title: Open Questions
icon: "💡"
---

- **Sub-benchmark selection**: How to approach methodologically sub-benchmark selection?
- **Custom models' architecture**: Any comments or improvements on the custom models?
- **Noise**: Do we use checkpoint noise or benchmark noise? Do we extend benchmark noise analysis?
- **Language coverage**

---
layout: default
title: Open Questions
subtitle: Sub-benchmark Selection
---

### How to approach methodologically sub-benchmark selection?

- SMART
- How to Select Datapoints for Efficient Human Evaluation of NLG Models? (https://arxiv.org/abs/2501.18251)

Other related work:
- Chen et al.: scaling behavior of downstream tasks
- Gupta et al.: SMART filtering of benchmark items
- Zhou et al.: item response theory for benchmark reliability

<!--
Chen et al. found that only a subset of tasks follow predictable scaling trends. Gupta et al. showed that filtering low-quality benchmark items reduces cost while preserving signal. Zhou et al. used item response theory to show many items have weak discriminative power.
-->

---
layout: default
title: Open Questions
subtitle: Custom models architecture
---

### Architecture

| Label    | Layers | d_model | Head dim | Heads | KV Heads | FFW Mult | Non-emb Params |
| -------- | ------ | ------- | -------- | ----- | -------- | ---- | --------- |
| **175M** | 16     | 1024    | 64 | 16    | 4        | 4 | 0.176B         |
| **350M** | 20     | 1280    | 64 | 20    | 5        | 4 | 0.344B         |
| **600M** | 24     | 1536    | 64 | 24    | 6        | 4 | 0.595B         |
| **1B**   | 28     | 1792    | 64 | 28    | 7        | 4 | 0.944B         |

Differences with Apertus:
- Tied embeddings (128k vocab, many params dedicated to embeddings)
- No goldfish loss (memorization not an issue at these scales)
- No cross-document attention masking

---
layout: default
title: Open Questions
subtitle: Custom models architecture
---

### Comments or improvements on the custom model training?

- ATLAS: Adaptive Transfer Scaling Laws for Multilingual Pretraining, Fine-tuning, and Decoding the Curse of Multilinguality (https://arxiv.org/pdf/2510.22037)
  - For similar losses to a 1B model on 100B English tokens, for $n$ languages
  - Model size = 1B x $n$^0.243
  - Dataset size = 100B x $n$^0.728
