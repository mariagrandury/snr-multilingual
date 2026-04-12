---
theme: scholarly
layout: cover
transition: slide-left
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
  - name: "María Grandury, Angelika Romanou, Eléonore, Clara Meister, Antoine Bosselut"
    institution: "EPFL NLP"
---

# Signal-Aware Benchmark Evaluation for Multilingual LLMs

---
layout: agenda
---


---
layout: section
---

# Motivation

Why do we need signal-aware evaluation?

<!--
Training multilingual LLMs is extremely expensive. Practitioners need to make decisions about data mixtures and hyperparameters early on, using small proxy models. But not all benchmarks are equally useful for guiding those decisions. We need to understand which benchmarks provide reliable signals.
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
icon: 🔬
---

# Can we predict, from small proxy models, which data mixture will produce the best large model?

This relies on rankings being **preserved across scales**. If not → wrong decisions → wasted compute.

<!--
The whole idea behind using proxy models is that if mixture A beats mixture B at small scale, it should also win at large scale. This is what Heineman et al. call "decision accuracy". If a benchmark doesn't preserve rankings across scales, then evaluating on it is worse than useless — it's misleading.
-->

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
layout: section
---

# Methodology

The Signal-and-Noise Framework

<!--
We now explain the core metrics of the framework: signal, noise, SNR, and decision accuracy. Understanding these is essential to interpret the results.
-->

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

| Parameter | Detail |
|---|---|
| **Source** | Allen AI DataDecide models |
| **Sizes** | 4M to 1B parameters |
| **Training mixtures** | 10–25 English-centric web corpora |
| **Seeds** | Up to 3 per configuration |
| **Mixtures include** | DCLM, Dolma, Falcon, FineWeb variants |

All mixtures are **English-centric** (Common Crawl derivatives)

<!--
The DataDecide suite provides the model diversity needed for this analysis. The models vary in training data mixture and random seed, but share the same architecture and tokenizer. Importantly, all training mixtures are English-centric — they are filtered/mixed versions of Common Crawl data. This is a key limitation for the multilingual experiments, as the models have minimal exposure to non-English languages.
-->

---

# Benchmarks

| Experiment | Benchmarks |
|---|---|
| **Reproduction experiment** (English) | ARC, HellaSwag, MMLU, WinoGrande, PIQA, OpenBookQA, BoolQ, SocialIQA, CSQA |
| **Multilingual downstream tasks** | Belebele, XStoryCloze, XNLI, XWinograd, XCOPA, BanglaMMlu, Click, TruthfulQA-multi, and more |
| **BPB evaluation sets** (raw text corpora) | Flores+ (10 subsets, ~220 languages) and Wiki40B (5 subsets, ~40 languages) |

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

| Metric | R | R² |
|---|---|---|
| Relative dispersion | **0.811** | **0.658** |
| Relative spread | 0.791 | 0.626 |

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

| Noise metric | R | R² |
|---|---|---|
| Checkpoint noise | 0.811 | 0.658 |
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

| Task subset | R | R² |
|---|---|---|
| English-only tasks | 0.594 | 0.353 |
| All non-English tasks | 0.045 | 0.002 |
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

| Metric | R | R² | Decision Accuracy |
|---|---|---|---|
| Non-English tasks (accuracy) | 0.045 | 0.002 | Near-random |
| **BPB on raw corpora** | **0.307** | **0.094** | **0.77 – 0.96** |

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

# Current & Next Steps

The Full Research Plan

---
layout: agenda
title: Next Steps
items:
  - "Train custom multilingual models"
  - "Comprehensive evaluation"
  - "INCLUDE Benchmark Analysis"
---

---
layout: methodology
title: "1. Custom Multilingual Models"
subtitle: "Addressing the core limitation"
---

## Pretrain 36 Small Models

- **4 sizes:** 175M, 350M, 600M, 1B params
- **3 multilingual data mixtures** (FineWeb + FineWeb2, 200+ languages)
- **3 seeds** per configuration

<Block type="info" title="Key difference from preliminary work">

Genuinely multilingual training data, not English-only. Addresses the core limitation that caused weak multilingual results.

</Block>

<!--
Unlike the DataDecide models which are all English-centric, these custom models will be trained on genuinely multilingual data from FineWeb and FineWeb2. This addresses the core limitation of the preliminary experiments. With 3 meaningfully different multilingual mixtures, we expect much higher signal on multilingual benchmarks, and can properly test whether the framework extends to this setting.
-->

---
layout: methodology
title: "2. Comprehensive Evaluation"
subtitle: "40 benchmarks across training stages"
---

## Comprehensive Evaluation

**Models:** custom models + 3 open-source families (Apertus, OLMo, SmolLM), 1B–70B

**40 multilingual benchmarks:**
- Cross-lingual understanding: XNLI, XCOPA, XStoryCloze, Belebele
- QA & Reasoning: XQuAD, MGSM, XLSum
- Regional knowledge: INCLUDE, Global MMLU
- Instruction-following: IFEval

Compute SNR, decision accuracy, and scaling-law error for **each benchmark at each training stage**

<!--
The evaluation will be much more comprehensive than the preliminary work. We add significantly more benchmarks and more model families. Critically, we will evaluate at different training stages to understand which benchmarks are most informative at each point during training. This enables stage-specific evaluation strategies.
-->

---
layout: methodology
title: "3. INCLUDE Benchmark Analysis"
subtitle: "Regional knowledge across 100+ countries"
---

## INCLUDE Benchmark Analysis

Deep analysis of **INCLUDE**: a multicultural regional knowledge benchmark covering **100+ countries**

- Identify optimal **country subsets** for evaluation at each training stage
- Provide subset recommendations on Hugging Face

<!--
INCLUDE is an internally developed benchmark designed to evaluate regional knowledge across diverse countries. By applying the SNR framework to INCLUDE, we can identify which country subsets provide the most reliable evaluation signal at different training stages. This will be published as concrete, actionable recommendations for practitioners.
-->

---
layout: section
---

# Open Questions

---

# Open Questions

1. 
