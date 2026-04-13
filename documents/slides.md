---
theme: scholarly
layout: cover
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
    institution: "EPFL NLP"
  - name: "Angelika Romanou"
    institution: "EPFL NLP"
  - name: "Éléonore Hasler"
    institution: "EPFL NLP"
  - name: Clara Meister
    institution: "EPFL NLP"
  - name: "Antoine Bosselut"
    institution: "EPFL NLP"
---

# Signal-Aware Framework for Multilingual LM Evaluation

---
layout: agenda
title: Agenda
items:
  - Motivation
  - Key concepts & previous work
  - Proposed methodology
  - Open questions
---

---
layout: section
---

# The Problem

Evaluation is expensive and often uninformative

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

# The Cost of Training Multilingual LLMs

Training requires **billions of tokens** and **hundreds of thousands of euros** in compute

Throughout training, practitioners must choose:

- Data mixtures
- Hyperparameters
- Evaluation strategies

These decisions are guided by **benchmark evaluations**

<!--
At every stage of training — pre-training, mid-training, and post-training — practitioners rely on benchmark scores to decide whether to keep a data mixture, change a hyperparameter, or continue training. But if the benchmark itself is noisy or uninformative, these decisions can be misleading, wasting enormous amounts of compute.
-->

---
layout: bullets
title: The Benchmark Problem
icon: "⚠️"
---

## Not all benchmarks provide equally informative signals

- **High variance**: Scores fluctuate across runs
- **Redundancy**: Multiple benchmarks measure the same thing
- **Weak correlation**: Improvements don't reflect real progress
- **Cost**: Large suites are expensive to run frequently

<!--
Benchmarks vary widely in their diagnostic value. Some are sensitive to meaningful changes, while others exhibit high variance, are redundant with each other, or correlate weakly with downstream objectives. At the same time, running many benchmarks is expensive, especially during early training when evaluations happen frequently. This creates a tradeoff between evaluation coverage and evaluation efficiency.
-->

---
layout: focus
color: green
icon: 🎯
---

## Which (subsets of) benchmarks provide reliable signal at each stage of multilingual model training?

---
layout: bullets
title: Why Multilingual?
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

# Prior Work

**Heineman et al. (2025)**: Signal-and-Noise framework

- Quantifies benchmark reliability via SNR
- Higher SNR → more consistent model rankings across scales
- Validated on English benchmarks only

**Other related work:**

- Chen et al. — scaling behavior of downstream tasks
- Gupta et al. — SMART filtering of benchmark items
- Zhou et al. — item response theory for benchmark reliability

<!--
Heineman et al. introduced the SNR framework showing that benchmarks with higher signal-to-noise ratio more reliably preserve model rankings. Chen et al. found that only a subset of tasks follow predictable scaling trends. Gupta et al. showed that filtering low-quality benchmark items reduces cost while preserving signal. Zhou et al. used item response theory to show many items have weak discriminative power.
-->

---
title: Signal-to-Noise Ratio
subtitle: Quantifying benchmark reliability
---

<Block type="success" title="Signal">

How well a benchmark $b$ separates models $m$ of similar scale:

$$\text{Signal}(b) = \text{Var}_{\text{models}}\big[\bar{s}_b(m)\big]$$

</Block>

<Block type="success" title="Noise">

Variability across checkpoints of a model $m$:

$$\text{Noise}(b) = \mathbb{E}_m\big[\text{Var}_{\text{runs}}[s_b(m)]\big]$$

</Block>

<Block type="success" title="SNR">

$$\text{SNR}(b) = \frac{\text{Signal}(b)}{\text{Noise}(b)}$$

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
layout: section
---

# Methodology

The Signal-and-Noise Framework

---

# Signal

How well does a benchmark **separate** models trained on different data?

$$\text{Signal}(M) = \frac{\max_{m \in M} s(m) - \min_{m \in M} s(m)}{\frac{1}{|M|} \sum_{m \in M} s(m)}$$

- Measured as **relative dispersion** of scores across training mixtures
- For a fixed benchmark and model size
- Higher signal = benchmark separates mixtures more clearly

<!--
Signal captures how much a benchmark's scores vary when you change the training data mixture but keep the model size fixed. If all mixtures produce the same score on a benchmark, then the benchmark has no signal — it can't help you choose between mixtures. We use relative dispersion, which the preliminary results confirm is better than relative spread (R = 0.811 vs R = 0.791).
-->

---
layout: compare
title: "Noise: Two Approaches"
leftLabel: Original
rightLabel: Ours
leftColor: amber
rightColor: green
---

### Checkpoint noise

Score variability across **late training checkpoints**

$$\text{Noise} = \frac{\frac{1}{|M|}\sum_{m} \sigma_{\text{step}}(m)}{\mu(M)}$$

⚠️ Requires intermediate checkpoints (rarely available)

::right::

### Benchmark noise

Score variability across **k-fold splits** of the evaluation set

$$\text{Noise} = \text{Var}_{\text{folds}}[s_b(m)]$$

✅ Computable from a **single evaluation run**

<!--
The original noise metric requires many intermediate training checkpoints for each model, which are rarely publicly available. The DataDecide suite is essentially the only open model family with this granularity. Our key contribution is proposing benchmark noise as an alternative: partition the evaluation set into k=5 folds, compute scores on each fold, and measure the standard deviation. This can be done from a single evaluation run on any model, making the framework much more practical.
-->

---

# Signal-to-Noise Ratio & Decision Accuracy

$$\text{SNR} = \frac{\text{Signal}}{\text{Noise}}$$

<br>

**Decision Accuracy**: if small model A > B on a benchmark, does large model A > B hold?

<br>

**Key insight**: SNR is strongly correlated with decision accuracy

→ SNR predicts how reliably a benchmark will transfer rankings across scales

<!--
The SNR combines both metrics into a single score. A benchmark with high signal and low noise will have high SNR, and Heineman et al. showed this correlates strongly with decision accuracy. In other words, if you want to know which benchmarks to trust when comparing small proxy models, look at their SNR. This is the foundation of the framework.
-->

---
layout: section
---

# Preliminary Results

by Élenorore, Clara, Antoine

---
layout: agenda
title: Preliminary Results
items:
  - "Reproduction of the English SNR Framework"
  - "Benchmark Noise, A More Practical Noise Metric"
  - "Extension to Multilingual Downstream Tasks"
  - "BPB on Raw Text Corpora"
---

---

# Models: DataDecide Suite

| Parameter             | Detail                                |
| --------------------- | ------------------------------------- |
| **Source**            | Allen AI DataDecide models            |
| **Sizes**             | 4M to 1B parameters                   |
| **Training mixtures** | 10–25 English-centric web corpora     |
| **Seeds**             | Up to 3 per configuration             |
| **Mixtures include**  | DCLM, Dolma, Falcon, FineWeb variants |

All mixtures are **English-centric** (Common Crawl derivatives)

<!--
The DataDecide suite provides the model diversity needed for this analysis. The models vary in training data mixture and random seed, but share the same architecture and tokenizer. Importantly, all training mixtures are English-centric — they are filtered/mixed versions of Common Crawl data. This is a key limitation for the multilingual experiments, as the models have minimal exposure to non-English languages.
-->

---

# Benchmarks

| Experiment                                 | Benchmarks                                                                                   |
| ------------------------------------------ | -------------------------------------------------------------------------------------------- |
| **Reproduction experiment** (English)      | ARC, HellaSwag, MMLU, WinoGrande, PIQA, OpenBookQA, BoolQ, SocialIQA, CSQA                   |
| **Multilingual downstream tasks**          | Belebele, XStoryCloze, XNLI, XWinograd, XCOPA, BanglaMMlu, Click, TruthfulQA-multi, and more |
| **BPB evaluation sets** (raw text corpora) | Flores+ (10 subsets, ~220 languages) and Wiki40B (5 subsets, ~40 languages)                  |

<!--
We use three sets of benchmarks across our experiments. The reproduction experiment uses the same English benchmarks as Heineman et al. The multilingual experiment adds a broad set of multilingual downstream tasks. Finally, we evaluate on raw text corpora using bits-per-byte (BPB), which bypasses the instruction-following bottleneck that makes accuracy unreliable for non-English tasks.
-->

---
layout: results
title: "Experiment 1: Reproduction"
subtitle: "Validating the English SNR Framework"
---

## Reproduction of the English SNR Framework

We compared two signal metrics: **relative dispersion** vs. **relative spread**

| Metric              | R         | R²        |
| ------------------- | --------- | --------- |
| Relative dispersion | **0.811** | **0.658** |
| Relative spread     | 0.791     | 0.626     |

**Relative dispersion** gives a stronger SNR–decision accuracy correlation

→ Adopted as the signal metric for all subsequent experiments

<!--
Heineman et al. claimed that relative dispersion is a better signal metric than relative spread, but their published plots actually used relative spread. Our re-analysis confirms their claim: relative dispersion yields a stronger correlation between SNR and decision accuracy. This is an important validation step — it shows the framework produces consistent results and gives us confidence to build on it.
-->

---
layout: results
title: "Experiment 2: Benchmark Noise"
subtitle: "A more practical noise metric"
---

## Benchmark Noise: More Practical AND More Predictive

Checkpoint noise requires intermediate checkpoints (rarely available). We propose **benchmark noise**: computable from a **single evaluation run** via k-fold splits.

| Noise metric        | R         | R²        |
| ------------------- | --------- | --------- |
| Checkpoint noise    | 0.811     | 0.658     |
| **Benchmark noise** | **0.854** | **0.730** |

- Benchmark noise (150M models) correlates strongly with checkpoint noise: **R = 0.854**
- Robust across all model sizes (150M–1B)
- SNR with benchmark noise yields **stronger** correlation with decision accuracy

<Highlight type="success">Not just easier to compute — also more predictive of decision accuracy</Highlight>

<!--
This is a central result. Checkpoint noise requires dozens of training checkpoints per model, which are rarely publicly available. We propose benchmark noise: partition the evaluation set into k=5 folds, compute scores on each fold, and measure the standard deviation. This can be done from a single evaluation run of a small model.

The benchmark noise computed from just the 150M models correlates at R = 0.854 with checkpoint-to-checkpoint noise. This means both metrics capture the same underlying instability — the tendency of a benchmark to produce fluctuating scores.

Not only is benchmark noise easier to compute, it actually produces a stronger SNR-decision accuracy correlation than the original checkpoint noise. This is likely because benchmark noise captures a complementary source of instability — the finite sample nature of the evaluation set — that checkpoint noise misses.
-->

---
layout: results
title: "Experiment 3: Multilingual Downstream Tasks"
subtitle: "Framework reliability depends on model competence"
---

## Multilingual Extension: Framework Weakens

| Task subset                 | R     | R²    |
| --------------------------- | ----- | ----- |
| English-only tasks          | 0.594 | 0.353 |
| All non-English tasks       | 0.045 | 0.002 |
| Non-English (excl. 3 worst) | 0.293 | 0.086 |

**Why?** Small English-first models perform **near-randomly** on underrepresented languages → uninformative rankings. No framework can recover signal from random scores.

<Block type="info" title="Key insight">

Framework reliability is **conditional on model competence**. The SNR framework doesn't fail for multilingual settings in general — it fails when proxy models lack linguistic competence. → Need genuinely multilingual models.

</Block>

<!--
When we apply the framework to multilingual downstream tasks, the correlation essentially disappears for non-English tasks (R = 0.045). The 3 most unstable benchmarks — click, bangla_mmlu, belebele — cover languages most underrepresented in training data. Removing them raises the non-English correlation to R = 0.293.

The DataDecide models are trained on English web data. When evaluated on tasks in Bangla, Swahili, or other underrepresented languages, the small models essentially guess randomly. Random guessing produces unstable, meaningless rankings.

This is perhaps the most important finding: the framework is sound, but its applicability depends on models having at least some competence in the target languages. The next step is to apply this to models trained on multilingual data — which is exactly what the research proposal plans.
-->

---
layout: results
title: "Experiment 4: BPB on Raw Text Corpora"
subtitle: "Bypassing the instruction-following bottleneck"
---

## BPB: A Better Multilingual Evaluation Signal

**Accuracy** is discrete and unreliable when models don't understand English-centric prompts. **Bits-per-byte (BPB)** is continuous, requires no instruction-following, and is more stable.

| Metric                       | R         | R²        | Decision Accuracy |
| ---------------------------- | --------- | --------- | ----------------- |
| Non-English tasks (accuracy) | 0.045     | 0.002     | Near-random       |
| **BPB on raw corpora**       | **0.307** | **0.094** | **0.77 – 0.96**   |

- Wiki40B reaches decision accuracy of **0.96**
- Low signal (0.05–0.16) expected: all DataDecide mixtures are English web variants
- With diverse multilingual mixtures → expect much higher signal

<Highlight type="success">Even small proxy models produce rankings that closely agree with 1B model rankings</Highlight>

<!--
Given the limitations of accuracy on downstream tasks, we explore bits-per-byte on raw multilingual text (Flores+ and Wiki40B). BPB measures how well a model predicts the next byte, without needing to understand a prompt format. This bypasses the instruction-following bottleneck that makes accuracy unreliable for non-English tasks.

The results are striking. While multilingual downstream tasks showed near-random decision accuracy, BPB on raw corpora shows that even small models produce highly reliable rankings (0.77–0.96). The absolute SNR correlation R = 0.307 is still modest, but it's a major improvement over R = 0.045. The low signal is expected since all DataDecide mixtures are English web crawl variants. With genuinely multilingual mixtures, we expect much higher signal and stronger correlations.
-->

---
layout: bullets
title: Key Takeaways
icon: "→"
---

## Key Takeaways

- **Framework reproduces** on English benchmarks ✅
- **Benchmark noise** is more practical AND more predictive than checkpoint noise ✅
- **Multilingual extension weakens** with English-first models — model limitation, not framework limitation ⚠️
- **BPB on raw corpora** yields higher decision accuracy and better SNR correlation than accuracy on downstream tasks ✅

<!--
These four results paint a coherent picture. The framework is sound and reproducible. Our new noise metric is an improvement. The challenges in multilingual settings come from the English-first nature of the models, not from a fundamental flaw in the approach. And BPB offers a promising alternative evaluation strategy that bypasses the instruction-following bottleneck.
-->

---
layout: bullets
title: Limitations
icon: "⚠️"
---

## Limitations

- **English-only training data**: all DataDecide mixtures are English web crawl derivatives
- **No architectural variation**: models differ only in data mixture and seed
- **Small benchmark sizes**: some folds contain fewer items than recommended
- **Inverse scaling**: tasks like TruthfulQA violate the "bigger = better" assumption
- **Single model family**: results constrained to DataDecide suite

<!--
The most significant limitation is the English-centric nature of the training data. This directly explains the weak multilingual results and limits how much we can conclude about the framework's applicability to multilingual models. The lack of architectural variation means we haven't tested whether the framework helps choose between model architectures. And the small fold sizes for some benchmarks (e.g., MMLU with only 57 items per fold) are a theoretical concern, even though the correlations are empirically strong.
-->

---
layout: section
---

# New Results

SNR Analysis on DataDecide and Apertus QAT Models

---
layout: agenda
title: New Results
items:
  - "SNR on DataDecide (AllenAI): English benchmark reliability at scale"
  - "SNR on Apertus QAT: Multilingual benchmarks across training variants"
  - "Cross-dataset SNR comparison"
  - "Implications for our custom model experiments"
---

---
layout: results
title: "DataDecide SNR: 202 English Benchmarks"
subtitle: "389K evaluations across 28 data mixtures"
---

## DataDecide SNR: Framework Validated at Scale

SNR computed over **202 tasks**, **105 models** (150M–1B), **28 data mixtures**

<br/>

| Metric                 | Value     |
| ---------------------- | --------- |
| **R (log-SNR vs DA)**  | **0.617** |
| **R²**                 | **0.380** |
| Mean Decision Accuracy | 59.6%     |

<br/>

**Top benchmarks by SNR + DA:**

| Benchmark     | SNR  | Decision Accuracy |
| ------------- | ---- | ----------------- |
| ARC Easy      | 74.4 | 94.7%             |
| MMLU          | 41.2 | 90.7%             |
| MMLU Pro      | 40.7 | 87.3%             |
| HellaSwag     | 41.5 | 80.3%             |
| ARC Challenge | 32.7 | 85.2%             |

<!--
We ran our SNR toolkit on the full Allen AI signal-and-noise dataset — 389K evaluations of DataDecide models across 28 English web data mixtures. The log-SNR vs decision accuracy correlation is R=0.617, confirming that benchmarks with higher SNR more reliably preserve model rankings from small to large scale. ARC Easy, MMLU, and HellaSwag emerge as the most reliable English benchmarks, with both high SNR and high decision accuracy.
-->

---

## DataDecide: SNR vs Decision Accuracy

<br/>

<img src="" style="max-height: 420px; margin: 0 auto; display: block;" />

<br/>

R = **0.617** with 95% confidence band. Benchmarks in the upper-right quadrant are the most reliable: high SNR and high decision accuracy.

<!--
This scatter plot is the central visualization of the SNR framework. Each point is a benchmark. The x-axis shows SNR (signal-to-noise ratio) and the y-axis shows decision accuracy (how often small model rankings agree with large model rankings). The log-linear fit with R=0.617 confirms that SNR is a good predictor of decision accuracy. ARC Easy, MMLU, and AutoBencher are standout benchmarks in the upper-right quadrant.
-->

---
layout: results
title: "Apertus QAT: Multilingual SNR"
subtitle: "572 tasks across training variants"
---

## Apertus QAT: Multilingual Benchmark Reliability

SNR computed over **572 tasks** (including multilingual), **22 Apertus models** (0.6B–8B)

Signal measures separation across **training variants** (base, SFT, long, longctx)

<br/>

| Metric                 | Value     |
| ---------------------- | --------- |
| **R (log-SNR vs DA)**  | **0.126** |
| Mean Decision Accuracy | 54.9%     |

<br/>

**Top multilingual benchmarks by SNR:**

| Benchmark              | SNR  | Decision Accuracy |
| ---------------------- | ---- | ----------------- |
| XCOPA (Indonesian)     | 16.7 | 100%              |
| HellaSwag (Vietnamese) | 10.3 | 100%              |
| HellaSwag (Bengali)    | 8.3  | 100%              |
| XCOPA (Vietnamese)     | 8.3  | 100%              |
| XNLI (Thai)            | 8.2  | 100%              |

<!--
For the QAT analysis, signal measures how much scores change across training variants — base distillation, SFT, long-context, etc. — rather than data mixtures. The weaker R=0.126 correlation is expected because training variants differ less than data mixtures. However, the multilingual results are striking: cross-lingual benchmarks like XCOPA and HellaSwag in non-English languages show very high SNR and perfect decision accuracy. This suggests that multilingual benchmarks CAN be highly reliable — provided the models have multilingual competence, which Apertus (distilled from an 8B multilingual model) does.
-->

---

## QAT: Top Multilingual Benchmarks by SNR

<br/>

<img src="" style="max-height: 440px; margin: 0 auto; display: block;" />

<br/>

Multilingual benchmarks (XCOPA, HellaSwag, XNLI in non-English) dominate the top SNR rankings — the first evidence that the SNR framework **works for multilingual evaluation**.

<!--
This ranking chart shows the top benchmarks by SNR from the QAT analysis. Strikingly, the top positions are dominated by multilingual benchmarks — XCOPA in Indonesian, HellaSwag in Vietnamese, XNLI variants. This is a fundamental advance over the preliminary results, where multilingual tasks showed near-zero correlation. The difference: Apertus models have genuine multilingual competence from their 8B parent, while DataDecide models were English-only.
-->

---
layout: compare
title: "Cross-Dataset Comparison"
leftLabel: DataDecide (AllenAI)
rightLabel: Apertus (QAT)
leftColor: blue
rightColor: green
---

### English benchmarks

- **R = 0.617** (log-SNR vs DA)
- 28 data mixtures as signal source
- Top: ARC Easy, MMLU, HellaSwag
- Strong correlation confirms framework

::right::

### Multilingual benchmarks

- **R = 0.126** (log-SNR vs DA)
- Training variants as signal source
- Top: XCOPA_id, HellaSwag_vi, XNLI_th
- Weaker R but **multilingual benchmarks now informative**

<!--
The side-by-side comparison reveals important insights. DataDecide gives stronger overall correlation because data mixtures create more variance than training recipe variants. But the QAT analysis provides the first evidence that the SNR framework extends to multilingual settings when models have multilingual competence. The cross-dataset SNR correlation of R=0.408 across 64 common tasks shows moderate agreement — benchmarks that are reliable in one setting tend to be reliable in the other.
-->

---

## Cross-Dataset SNR Correlation

<br/>

<img src="" style="max-height: 420px; margin: 0 auto; display: block;" />

<br/>

**R = 0.408** across 64 common tasks. Benchmarks reliable in the DataDecide setting tend to be reliable in the QAT setting too — partial transfer of SNR rankings across model families.

<!--
This plot shows the SNR of each common benchmark in the AllenAI setting (x-axis) vs the QAT setting (y-axis). The positive correlation of R=0.408 suggests that benchmark reliability is partially an intrinsic property of the benchmark, not just an artifact of the specific model family. However, the scatter shows significant divergence for some tasks — indicating that model family and training setup also matter.
-->

---

## Top Benchmarks: DataDecide vs QAT

<br/>

<img src="" style="max-height: 420px; margin: 0 auto; display: block;" />

<br/>

ARC Easy, HellaSwag, MMLU, and ARC Challenge appear in top 20 for **both** model families — robust benchmark choices regardless of training setting.

<!--
The side-by-side top-20 ranking shows which benchmarks are consistently reliable across both model families. ARC Easy, HellaSwag, MMLU, and ARC Challenge rank highly in both, making them robust choices for evaluation. This is a practical takeaway: these benchmarks can be trusted across different model architectures and training procedures.
-->

---
layout: bullets
title: Key Insights from New Analysis
icon: "→"
---

## Key Insights

- **SNR framework scales**: R=0.617 across 202 tasks on DataDecide confirms Heineman et al.'s findings at larger scale
- **Multilingual SNR works** when models have multilingual competence: XCOPA, HellaSwag, XNLI in non-English show high SNR with Apertus
- **Cross-dataset transfer**: R=0.408 shows benchmark reliability partially transfers across model families
- **Robust benchmarks**: ARC Easy, MMLU, HellaSwag are reliable regardless of model family or training setup
- **Training variant signal** is weaker than data mix signal (R=0.126 vs 0.617) but still informative

<Block type="success" title="Milestone">

First quantitative evidence that the SNR framework extends to multilingual evaluation — resolving the R=0.045 limitation from the preliminary analysis.

</Block>

<!--
The most important finding is that the multilingual weakness of the preliminary analysis was a model competence issue, not a framework issue. With Apertus models that have genuine multilingual capabilities, multilingual benchmarks like XCOPA-Indonesian and HellaSwag-Vietnamese show the highest SNR values. This validates our research hypothesis and motivates the custom multilingual model training in Phase 1.
-->

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
    description: "Paper, HF benchmark subsets, open-source toolkit"
---

---
title: "Phase 1: Model Training"
subtitle: "36 small models across 4 scales with 3 data mixtures"
---

### Architecture

| Label    | Layers | d_model | Heads | KV Heads | Non-emb params |
| -------- | ------ | ------- | ----- | -------- | -------------- |
| **175M** | 16     | 1024    | 16    | 4        | 0.176B         |
| **350M** | 20     | 1280    | 20    | 5        | 0.344B         |
| **600M** | 24     | 1536    | 24    | 6        | 0.595B         |
| **1B**   | 28     | 1792    | 28    | 7        | 0.944B         |

### Data Mixtures

- English data: FineEdu2-DCLM
- Multilingual data: FineWeb2, high-quality filter
- Mixtures: 30%-70%, 60%-40%, 90%-10%
- 100B tokens

---
title: "Phase 2: Model Evaluation"
subtitle: "Model Suite"
---

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

Internally developed benchmark evaluating **regional knowledge** across **120 country-language** pairs.

</Block>

### Analysis goals

- Apply SNR framework to country-level subsets
- Identify which sample clusters have higher SNR at each training stage
- Produce **optimal evaluation subsets** that maximize signal while minimizing cost

---
title: "Phase 5: Dissemination"
subtitle: "Open science deliverables"
---

**Paper** contributions:

- SNR scores and stage-specific recommendations for 40 multilingual benchmarks
- Analysis of benchmark reliability across pre/mid/post-training

<br/>

**INCLUDE Subsets** on Hugging Face:

- High-signal language-country subsets for each training stage

<br/>

Open-Source Toolkit:

- Modular Python package: metrics, data loading, visualization
- Compute SNR on your own benchmarks and models
- Stage-specific automated analysis and recommendations

---
layout: section
---

# Open Questions

---
layout: bullets
title: Open Questions
icon: "💡"
---

- **Data mix signal vs training recipe signal**: Our custom models will provide true data mix signal (EN/multilingual ratios) — will R improve beyond 0.617?
- **Benchmark noise calibration**: How many folds (k) and samples are needed for stable noise estimates?
- **Sub-benchmark selection**: Given a budget of N evaluations, which subset maximizes decision accuracy?
- **Language coverage**: Do SNR-optimal subsets cover all language families, or cluster around high-resource languages?
- **Stage transitions**: Which benchmarks are reliably informative across pre/mid/post-training vs stage-specific?
- **Cross-architecture SNR transfer**: R=0.408 between DataDecide and Apertus — will this hold for our custom models?

<!--
These questions now build on concrete data. The data mix signal question is sharpened by the QAT finding that training recipe signal is weaker (R=0.126 vs 0.617) — our custom models with explicit EN/multilingual mix ratios should produce much stronger signal. The cross-architecture transfer question at R=0.408 suggests benchmark reliability partially transfers, but we need to verify this with our own model family.
-->
