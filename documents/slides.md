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
layout: image-right
image: /cubes.png
ratio: "1:6"
fit: contain
title: Key concepts
subtitle: Signal and Noise
---



---
title: Key concepts
subtitle: Signal and Noise
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

4.1. Paper
To calculate signal we
use the final checkpoint of each of the 25 small models, and to calculate noise, we use the standard
deviation around the final 5 checkpoints of the small-scale models. Since we have a measure of
noise for each model, we use the average of the noise across the small models.


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
    title: "SNR Framework ⏳"
    description: "Framework implemented, currently designing extension"
  - year: "Phase 4"
    title: "INCLUDE Analysis"
    description: "Optimal subsets for 120 countries across training stages"
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
layout: image-right
image: /fineweb2_languages_cropped.png
ratio: "3:1"
fit: contain
title: "Phase 1: Model Training"
subtitle: "36 small models across 4 scales with 3 data mixtures"
---

### Data Mixtures

- English data: FineEdu2-DCLM
- Multilingual data: FineWeb2
  - Apertus' high-quality filter
  - Top 200 languages
  - Use original naturally occurring language distribution
- Mixtures: 30%-70%, 60%-40%, 90%-10%
- Tokens: 100B in total

---
title: "Phase 2: Model Evaluation"
subtitle: "Model Suite"
---

| Size | Pretraining | Midtraining | Post-training |
|------|-------------|-------------|---------------|
| **< 1B** | Custom 175M, 350M, 600M, 1B (3 mixtures each); Apertus 1B | — | Apertus 0.6B / 1.7B Distilled, Apertus 1.7B Distilled SFT |
| **3B** | Apertus 3B | SmolLM3 3B Base | SmolLM3 3B |
| **8B** | — | Apertus 8B Base; OLMo3 7B Base | Apertus 8B Instruct; OLMo3 7B SFT, 7B DPO, 7B Instruct (RLVR) |
| **70B** | — | Apertus 70B Base | 70B Instruct |

Model selection:
- Custom models for controlled pretraining analysis (as AllenAI)
- Open-source families for mid/post-training coverage up to 70B

<!-- 

title: "Phase 2: Model Evaluation"
subtitle: "Evaluation Suite"


Benchmark selection
- Benchmarks commonly used in literature
- Benchmarks with high SNR in AllenAI and Éléonore's analyses 
- Bechmarks covering underserved languages 
- Include V2 for in depth analysis 
-->

---
title: "Phase 2: Model Evaluation"
subtitle: "Evaluation Suite"
---

| Category | Pretraining | Midtraining | Post-training |
|----------|-------------|-------------|---------------|
| **Language Modeling & Completion** |  hellaswag, piqa, xnli, xcopa, pawsx, xwinograd, m_arc, wikitext, lambada, | hellaswag, piqa, xnli, xcopa, pawsx, xwinograd, m_arc, wikitext, lambada | hellaswag |
| **Commonsense & Reasoning** | commonsense_qa, openbookqa, ai2_arc, winogrande, gsm8k | commonsense_qa, openbookqa, ai2_arc, winogrande, gsm8k | bbh, drop, gsm8k_cot, hendrycks_math, mathqa |
| **Knowledge & QA** | mmlu, global_mmlu, squadv2, include_base_44 | mmlu, global_mmlu, squadv2, include_base_44 | mmlu_flan_cot_zeroshot, global_mmlu, truthfulqa, truthfulqa, blend, cultural_bench  |
| **Code** | — | — | humaneval_instruct, mbpp_instruct |
| **Instruction-Following & Safety** | — | — | ifeval, acp_bench, harmbench, toxigen, bbq |

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

# Results — Multilingual SNR Framework

Four analyses on 36 Apertus pretrains (4 sizes × 3 mixes × 3 seeds)
---
layout: focus
color: blue
icon: 🔬
---

## Which SNR definition best correlates with decision accuracy across 12 languages?

---
layout: figure
image: /results/top_variants_overall.png
title: SNR Definition
subtitle: "Q3 — Top variants across languages (DA-size & DA-ckpt)"
---

---
layout: bullets
title: SNR Definition
subtitle: Findings
icon: "✅"
---

- **`quartile_deviation` is the global default**: mean Pearson r ≈ **+0.343** (DA-size)
- 6-way dispersion-cluster tie: `aad`, `mpd`, `rms_deviation`, `dist_std`, `dispersion`, `range` all within ~0.04
- Inter-variant correlation in the cluster: **r ≥ 0.999** — algebraically redundant
- `tukey`, `projection` (depth-based): **r ≈ 0** with DA → avoid at this pool size

---
layout: figure
image: /results/top_benchmarks_per_language.png
title: SNR Definition
subtitle: Top-5 benchmarks per language under `quartile_deviation` @ 1B
---

---
layout: bullets
title: SNR Definition
subtitle: Best benchmarks per language
icon: "🌐"
---

- **`multiblimp_<lang>` ranks #1** wherever it exists (ar, en, es, eu, hi, ru, tr) — by 3–5× the runner-up
- Without multiblimp: **`xstorycloze_<lang>`** (sw, th, vi, zh) or **`hellaswag_<lang>`** (ja)
- **Drop `xnli_<lang>`** rows where DA-size = 0 — high SNR but mis-ranks the 1B target

---
layout: figure
image: /results/variant_r_train_vs_test.png
title: SNR Definition
subtitle: Train vs test split — per-(lang, variant) Pearson r
---

---
layout: bullets
title: SNR Definition
subtitle: "Generalization across seed splits"
icon: "🔁"
---

- Global ranking is **highly stable**: Spearman ρ = **+0.83** (DA-size), **+0.91** (DA-ckpt)
- Per-language picks are **not** stable: 1/14 keep variant, 2/14 keep family
- DA-ckpt cluster lies tightly on the diagonal; DA-size scatter is wider → DA-ckpt is the portable metric
- Retention of train-best on test: 53% (DA-size), **77% (DA-ckpt)**

---
layout: bullets
title: SNR Definition
subtitle: Methodology
icon: "⚙️"
---

- **22 SNR variants** × 4 sizes × 115 multilingual parent tasks
- Three pools: `seeds_28_1797` (train, 6 model_families/size), `seeds_1904` (test, 3), `seeds_28_1797_1904` (pooled, 9)
- **DA-size**: small-last-ckpt → 1B-last-ckpt rank agreement (3 small sizes)
- **DA-ckpt**: within-size early → late ckpt agreement (3 early ckpts × 4 sizes)
- Per-language correlation: Pearson r between log10(SNR) and DA

---
layout: bullets
title: SNR Definition
subtitle: Possible logical bugs
icon: "🐛"
---

- **Dispersion cluster's r ≥ 0.999** → Spearman ρ on global ranking is artificially stabilised by redundancy
- DA-size uses the 1B model as the rank target, but 1B itself isn't fully converged at iter 50K → measures rank-agreement with a small-ish proxy
- Train/test split: per-language argmax is computed on only **6 (mix, seed) units** in train; argmax-of-noisy-vector is biased
- Languages with `de`, `fr` ≤4 valid cells are dropped silently — agreement denominators effectively shrink to 12

---
layout: bullets
title: SNR Definition
subtitle: Methodology improvements
icon: "💡"
---

- **Bootstrap CIs** on the per-language Pearson r and the cross-pool Spearman ρ
- Collapse the dispersion cluster to one representative before computing ranking statistics
- Use a **larger target** (e.g. Apertus 8B) for DA-size instead of the 1B custom pretrain
- Add a third independent seed pool to test generalization without re-using the same 1904 split

---
layout: section
---

# Benchmark Creation

What makes a benchmark high-SNR? Curation vs task design.
---
layout: focus
color: blue
icon: 🔬
---

## What benchmark design features (curation, format, option count, item length) predict SNR?

---
layout: figure
image: /results/snr_per_family_ranked.png
title: Benchmark Creation
subtitle: Per-family median SNR (pooled pool)
---

---
layout: bullets
title: Benchmark Creation
subtitle: Findings
icon: "✅"
---

- **Number of answer options is the strongest predictor**: Spearman ρ = **+0.77** (p = 0.006); KW H = 5.5 (p = 0.019)
- **Curation method explains <1% variance**: KW H = 1.44 (p = 0.49) — once task design is held constant, curation doesn't matter
- Top SNR families: **multiblimp** (3.8), **xwinograd** (2.1), **xstorycloze** (2.0)
- Bottom: **belebele** (0.7), **global_mmlu_full** (0.7), **arc** (0.6) — all 4-option MCQ

---
layout: figure
image: /results/snr_vs_random_baseline.png
title: Benchmark Creation
subtitle: SNR vs random baseline (1 / n_options)
---

---
layout: bullets
title: Benchmark Creation
subtitle: "Why fewer options → higher SNR"
icon: "🎯"
---

- Each option adds another noisy log-likelihood estimate that must be ranked correctly
- 2-option tasks: log-likelihood comparison is sharper
- Multiblimp's minimal-pair format = single-token contrast → uniquely sharp signal
- Length features show the same direction qualitatively but don't reach significance at n=11

---
layout: bullets
title: Benchmark Creation
subtitle: Methodology
icon: "⚙️"
---

- **11 benchmark families**, SNR signal = `snr_mpd_1B` from snr_definition pooled pool
- Per-family aggregate: median across per-language aggregate tasks in the family
- Three phases: curation (Phase 0), task format (Phase A), item lengths (Phase B)
- Statistical tests: Kruskal-Wallis (categorical), Spearman ρ (continuous)
- Outputs in `seeds_<pool>/group_stats.csv`, `per_family_snr.csv`

---
layout: bullets
title: Benchmark Creation
subtitle: Possible logical bugs
icon: "🐛"
---

- **n = 11 families** → every test is underpowered; Spearman on a 3-valued discrete variable (n_options ∈ {2,3,4}) is fragile
- `template_generated` is only used by multiblimp — confounded with `minimal_pair` format
- `snr_median` per family weights families with more language-aggregates more (xnli has 11 langs, multiblimp 7)
- `random_baseline` is `1 / n_options` — perfectly co-linear with n_options, not an independent axis

---
layout: bullets
title: Benchmark Creation
subtitle: Methodology improvements
icon: "💡"
---

- **Controlled comparison**: hold format constant, vary curation (HellaSwag MT vs XStoryCloze human)
- Replace marginal KW tests with a single regression on (format + n_options + curation)
- Add more benchmarks (truthfulqa, mgsm, agieval, …) — the families needed for the curation question
- Re-sample length features from the full multilingual splits, not just English (current Phase B)

---
layout: section
---

# AllenAI Cross-Corpus Comparison

Do our Apertus-derived SNR rankings transfer to AllenAI DataDecide?

---
layout: focus
color: blue
icon: 🔬
---

## Do the SNR variants we recommend on Apertus also correlate with AllenAI DataDecide on the shared English benchmarks?

---
layout: figure
image: /results/snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png
title: AllenAI Comparison
subtitle: "Apertus vs AllenAI SNR — best variant (pooled pool)"
---

---
layout: bullets
title: AllenAI Comparison
subtitle: Findings
icon: "✅"
---

- **Pooled-pool cross-corpus Pearson r = 0.935** (`star_discrepancy_shifted` variant)
- Top-10 reliable benchmarks **agree by Jaccard 1.0** on both corpora
- Cross-corpus-reliable: `arc_challenge`, `arc_easy`, `csqa`, `hellaswag`, `mmlu`, `openbookqa`, `piqa`
- **Discrepancy + dispersion families transfer**; relative-spread family does not (incl. upstream AllenAI default `rel_std`)

---
layout: figure
image: /results/snr_apertus_vs_snr_allenai_grid.png
title: AllenAI Comparison
subtitle: Variant × size scatter grid
---

---
layout: bullets
title: AllenAI Comparison
subtitle: Methodology
icon: "⚙️"
---

- **Apertus side**: 22 SNR variants × 4 sizes × 7 shared tasks (pooled pool, 9 model_families/size)
- **AllenAI side**: same 22 variants on the DataDecide ladder (25 mixes × 5 ckpts)
- Task-name reconciliation: `global_mmlu_full_en[_<subj>] → mmlu[_<subj>]` (alias)
- **Headline axis**: log10(SNR_1B) on each corpus, Pearson r over 7 tasks
- Top-K agreement: intersection / Jaccard at K ∈ {5, 10, 20}

---
layout: bullets
title: AllenAI Comparison
subtitle: Possible logical bugs
icon: "🐛"
---

- **Only 7 shared tasks** → 95% CI on r=0.935 is very wide (Pearson on n=7 is fragile)
- MMLU alias: Apertus runs **Cohere Full** translation, AllenAI runs original Hendrycks MMLU → not the same content
- Variant ranking shifts across pools (`dispersion`→`discrepancy`→`star_discrepancy_shifted`) — could be argmax-of-noise rather than signal
- Reference HF models (single-mix) skipped from SNR pool, but they're part of AllenAI's DataDecide universe — apples-to-oranges

---
layout: bullets
title: AllenAI Comparison
subtitle: Methodology improvements
icon: "💡"
---

- Re-run **original `mmlu`** (not `global_mmlu_full_en`) on Apertus → drop the alias, compare like-for-like
- Add the 53 `mmlu_<subject>:mc` rows to Apertus → expand shared universe from 7 to ~60
- **Bootstrap CIs** on the cross-corpus Pearson r
- Report **Spearman ρ** alongside Pearson r (rank-based is more robust at n=7)

---
layout: section
---

# Smooth Subtasks — Per-Subtask

Selecting languages or MMLU subjects inside a benchmark

---
layout: focus
color: blue
icon: 🔬
---

## Per benchmark, can a subset of **subtasks** (languages / MMLU subjects) give higher SNR than the full set?

---
layout: figure
image: /results/global_mmlu_full_subjects.png
title: Per-subtask
subtitle: "Case 2 — MMLU subject subset curves per size"
---

---
layout: bullets
title: Per-subtask
subtitle: Findings
icon: "✅"
---

- **Best subsets substantially beat full sets**, across all three logical cases
- **Case 1** (language subset): Belebele 350M `+0.89` (full 1.97 → 4-lang 2.86); GMF 175M `+0.87`
- **Case 2** (MMLU subject, mean over 10 langs): 175M `+0.96` (`international_law` alone); 1B `+0.88` (`virology | human_aging`)
- **Case 3** (subject × language): up to `+1.28` (Spanish 175M, `formal_logic`); up to `+1.27` (Swahili 350M)

---
layout: figure
image: /results/belebele.png
title: Per-subtask
subtitle: "Case 1 — Belebele subset SNR per size"
---

---
layout: bullets
title: Per-subtask
subtitle: "Stability across seed pools"
icon: "🔁"
---

- **Case 2 (subjects) is highly stable**: `world_religions`, `international_law`, `human_aging`, `marketing` recur across pools
- **Case 1 (languages) is partially stable**: `xcopa` always collapses to one language; the winning language can flip
- **Case 3 (subject × language) is the most pool-sensitive** — argmax rarely repeats
- **Recipe**: keep subsets that recur in BOTH train (`seeds_28_1797`) and test (`seeds_1904`) pools

---
layout: bullets
title: Per-subtask
subtitle: Methodology
icon: "⚙️"
---

- 36 Apertus models, last-5 ckpts per (size, mix, seed) → 9 model_units per size
- Subset search: rank subtasks by standalone SNR, add greedily, record cumulative SNR
- Best subset = prefix that maximises the cumulative curve; random-order baseline alongside
- Three cases × 3 seed pools × 4 sizes → 100-row `summary.csv` ranked by `snr_gain`
- Combined-subset score = mean across (model, step) of included subtasks (relaxed inner-join)

---
layout: bullets
title: Per-subtask
subtitle: Possible logical bugs
icon: "🐛"
---

- **Selection and evaluation on the same data** → reported gains are optimistic; no held-out fold
- **Greedy argmax** of a noisy cumulative curve is biased upward (the prefix that won this run won't win the next)
- **Relaxed inner-join**: combined-subset denominator changes from cell to cell with coverage gaps → inflates/deflates both signal and noise
- `mgsm_direct` silently dropped (NaN parquet); English `truthfulqa_mc1` dropped by singleton filter

---
layout: bullets
title: Per-subtask
subtitle: Methodology improvements
icon: "💡"
---

- **K-fold over (mix, seed) units**: pick subset on K-1 folds, score on the held-out fold
- **Strict inner-join** with an explicit imputation rule (mean / EM) instead of variable coverage
- **Bootstrap CIs** on every `snr_gain` so we know when the gap is real
- **Group selection at higher granularity** (subject family / topic / difficulty bin) where Case 3 is unstable

---
layout: section
---

# Smooth Subtasks — Per-Sample

Selecting individual items inside a benchmark

---
layout: focus
color: blue
icon: 🔬
---

## Per benchmark, which **individual items** (doc-ids) maximise SNR? Does the per-item ranking transfer across model sizes?

---
layout: figure
image: /results/per_sample_xcopa_sw.png
title: Per-sample
subtitle: "xcopa_sw — cumulative SNR sweep over ranked items (1B)"
---

---
layout: bullets
title: Per-sample
subtitle: Findings
icon: "✅"
---

- A per-sample subset **beats the full set in 100%** of (lang, task, size) cells (320 cells, 80 benchmarks)
- Median `snr_gain` = **+0.641**, max **+2.68** (xcopa_sw 1B: 500 items → best 12 items)
- Only **40% (median)** of items carry any cross-mix signal; the rest are "dead" and dropped
- **Winning subset is tiny** — median **2%** of items
- **Neither subset nor ranking transfers across scale** — median Jaccard `0.03`, Spearman `0.05`

---
layout: figure
image: /results/per_sample_paws_eu.png
title: Per-sample
subtitle: "paws_eu — cumulative SNR sweep over ranked items (175M)"
---

---
layout: bullets
title: Per-sample
subtitle: Four selection methods (A/B/C/D)
icon: "🔀"
---

- **A — `greedy_snr_rank`**: rank items by individual SNR, sweep cumulative. The AllenAI baseline.
- **B — `forward_greedy`**: start from best item, add the one that maximally lifts joint SNR. Catches interactions; `O(N·K)`.
- **C — `irt_discrimination`**: 2PL IRT model on items × ckpts, keep high-`a_i` items, then A. Noisy at our ~5-ckpt scale.
- **D — `variance_prefilter`** *(default)*: drop dead items first, then A. Cheap, matches AllenAI semantics on informative items.

---
layout: bullets
title: Per-sample
subtitle: Why the argmax is so small
icon: "🔍"
---

- **Objective spikes early**: SNR = signal / noise; numerator regresses toward the mean faster than the denominator falls
- **Per-item ranking is mostly noise**: each item's SNR rests on ~5 ckpts × 3 mixes → cross-size Spearman ≈ **0.05**
- A noisy ranking + a knife-edge argmax → over-fit to a few lucky items
- The cumulative curve has a **broad near-peak plateau** — a much larger subset sits within a hair of the peak

---
layout: bullets
title: Per-sample
subtitle: Methodology
icon: "⚙️"
---

- 36 Apertus models, samples from `samples_*.jsonl` (cluster-only)
- SNR primitive: `signal_to_noise_ratio` over per-mix last-5-ckpt arrays — same as the subtask track
- Search spaces: ~1.1k items per `arc_*`, ~14k per `mmlu_*` → exhaustive enumeration impossible
- Four interchangeable methods via `smooth_subtasks_per_sample.py --method {A,B,C,D}`
- Each method writes to its own dir under `per_sample/<method>/<lang>/<task>/`

---
layout: bullets
title: Per-sample
subtitle: Possible logical bugs
icon: "🐛"
---

- **Selection and evaluation on the same samples** → 2.5% subsets that beat full-set in-sample but collapse out-of-sample
- **Per-item SNR estimate noise**: ~5 ckpts × 3 mixes → tiny n; the "best" item ranking is dominated by sampling noise (Spearman ≈ 0.05)
- **Binary-acc divide-by-zero**: a sample with all-correct ckpts has noise = 0 → SNR = ∞, guarded to NaN — drops informative cases
- **IRT (Option C) examinees** are checkpoints from one trajectory — non-independent; `a_i` estimates are unreliable
- Per-sample SNR scale is **not comparable** to per-subtask SNR (binary acc vs aggregate)

---
layout: bullets
title: Per-sample
subtitle: "Improvements — relax the selection rule (Lever 1)"
icon: "💡"
---

Cheapest lever — re-reads the `cumulative_snrs` array we already produce, no extra compute.

- **Still-beats-full-set rule**: report the *largest* subset with SNR ≥ full-set SNR (instead of the knife-edge argmax)
- **1-SE / ε-plateau rule**: take the largest `N` within one SE of the peak — analogous to the lasso 1-SE rule
- **Target-size rule**: fix a practitioner size (e.g. 25–50% of items), report retained SNR there

---
layout: bullets
title: Per-sample
subtitle: "Improvements — denoise the per-item informativeness (Lever 2)"
icon: "💡"
---

Fixes the root cause — the ranking is mostly noise at ~5 ckpts × 3 mixes.

- **Pool across seeds (and sizes)**: average each item's SNR over the 3 seeds before ranking
- **Shrinkage**: empirical-Bayes shrink per-item SNR toward the benchmark mean; or use IRT `a_i` as a smoother quality proxy
- **Group selection**: select at subtask / topic / difficulty bin granularity — exactly the Case 1–3 unit that *is* partially stable

---
layout: bullets
title: Per-sample
subtitle: "Improvements — make trust measurable (Lever 3)"
icon: "💡"
---

The reported `snr_gain` has no out-of-sample number today.

- **Held-out seed-pool CV**: select on `seeds_28_1797`, measure SNR *and* decision accuracy on `seeds_1904`
- **Stability selection**: bootstrap ckpts/seeds, run the sweep many times, keep items chosen in ≥ X% of runs
- Both yield larger consensus subsets *with* an out-of-sample trust number to report

---
layout: bullets
title: Per-sample
subtitle: "Improvements — change the objective (Lever 4)"
icon: "💡"
---

The SNR ratio spikes; decision accuracy saturates instead.

- **Optimise decision accuracy**, not raw SNR: smallest subset hitting a DA threshold is much larger and easier to defend
- DA is what practitioners actually want — "does the subset rank model pairs correctly?"
- Plot the DA curve alongside the SNR curve; report both

---
layout: bullets
title: Per-sample
subtitle: Recommended recipe
icon: "🎯"
---

Stack one lever per layer:

1. **Pool seeds (Lever 2)** → denoise the per-item ranking
2. **1-SE rule that still beats the full set (Lever 1)** → larger subset on the plateau
3. **Held-out seed-pool CV (Lever 3)** → out-of-sample trust number
4. **DA curve alongside SNR (Lever 4)** → practitioner-facing objective

If the denoised per-item subset still fails to transfer → empirical case for **group-level selection** as the trustworthy unit.

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

- **Signal definition**: How to calculate signal? Across which dimensions?
- **Noise definition**: Do we use checkpoint noise or benchmark noise? Do we extend benchmark noise analysis?
- **Sub-benchmark selection**: How to approach methodologically sub-benchmark selection?
- **Custom models' architecture**: Any comments or improvements on the custom models?
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
