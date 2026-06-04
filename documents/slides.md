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
  - Analysis
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

## Which (subsets of) benchmarks provide reliable signal at each stage of multilingual model training?

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
| **Pretraining** | Data mixture, architecture, scale | Custom 175M–1B suite (controlled) |
| **Midtraining** | Domain/quality up-sampling | Open-source bases (3B–70B) |
| **Post-training** | SFT / DPO / RLVR recipe | Instruct families (3B–70B) |

- We compute signal, noise, SNR, DA and scaling-law error **per stage**
- Goal: **stage-specific evaluation recommendations**, not a single global ranking
- Current results focus on the **pretraining** stage (the controlled custom suite)


---
layout: section
---

# Experimental Setup

Models and benchmarks

---
title: Experimental Setup — Models
subtitle: "Controlled custom pretraining suite (36 models)"
---

**36 Apertus pretrains** = 4 sizes × 3 data mixtures × 3 seeds, each trained on **100B tokens** (50k iterations).

| Axis | Values |
| ---- | ------ |
| **Sizes** | 175M, 350M, 600M, 1B |
| **Mixtures** (FineWeb-Edu / FineWeb2) | 30 / 70, 60 / 40, 90 / 10 |
| **Seeds** | 28, 1797, 1904 |

| Label    | Layers | d_model | Head dim | Heads | KV Heads | FFW Mult | Non-emb Params |
| -------- | ------ | ------- | -------- | ----- | -------- | ---- | --------- |
| **175M** | 16     | 1024    | 64 | 16    | 4        | 4 | 0.176B         |
| **350M** | 20     | 1280    | 64 | 20    | 5        | 4 | 0.344B         |
| **600M** | 24     | 1536    | 64 | 24    | 6        | 4 | 0.595B         |
| **1B**   | 28     | 1792    | 64 | 28    | 7        | 4 | 0.944B         |

---
layout: image-right
image: /fineweb2_languages_cropped.png
ratio: "3:1"
fit: contain
title: Experimental Setup — Data Mixtures
subtitle: "FineWeb-Edu (EN) + FineWeb2 (multilingual)"
---

### Data Mixtures

- **English**: FineWeb-Edu (DCLM)
- **Multilingual**: FineWeb2
  - Apertus' high-quality filter
  - Top 200 languages
  - Original naturally-occurring language distribution
- Mixtures: **30/70, 60/40, 90/10**
- Tokens: **100B** in total

Differences with Apertus: tied embeddings (128k vocab), no goldfish loss, no cross-document attention masking.

---
title: Experimental Setup — Models
subtitle: "Scaling beyond 1B: open-source families"
---

The custom suite caps at 1B. Open-source families extend the compute axis to **70B** and cover the mid-/post-training stages.

| Size | Pretraining | Midtraining | Post-training |
|------|-------------|-------------|---------------|
| **≤ 1B** | Custom 175M–1B (3 mixes × 3 seeds); Apertus3 1B; distilled 0.6B/1B | — | distilled Apertus 1.7B SFT |
| **3B** | Apertus3 3B | SmolLM3-3B base | SmolLM3-3B |
| **7–9B** | — | Apertus-8B base; OLMo-3-7B base | Apertus-8B Instruct; OLMo-3-7B SFT/DPO/Instruct |
| **≥ 12B** | — | gemma-3 12–27B; OLMo-3 13–32B; Qwen3 14B+ | Apertus 70B Instruct |

---
title: Experimental Setup — Models
subtitle: "Families spanning >=2 buckets"
---

Families spanning >=2 buckets:
  Apertus-2509                                  ['7-9B', '70B']
  OLMo-2-base                                   ['1B', '7-9B', '12-14B', '27-32B']
  Olmo-3-base                                   ['7-9B', '27-32B']
  Qwen3-Base                                    ['600M', '1.7B', '4B', '7-9B', '12-14B']
  ap-from8b-TOP256                              ['600M', '1B']
  apertus-fwEdu30-fw270-seed1797                ['175M', '350M', '600M', '1B']
  apertus-fwEdu30-fw270-seed1904                ['175M', '350M', '600M', '1B']
  apertus-fwEdu30-fw270-seed28                  ['175M', '350M', '600M', '1B']
  apertus-fwEdu60-fw240-seed1797                ['175M', '350M', '600M', '1B']
  apertus-fwEdu60-fw240-seed1904                ['175M', '350M', '600M', '1B']
  apertus-fwEdu60-fw240-seed28                  ['175M', '350M', '600M', '1B']
  apertus-fwEdu90-fw210-seed1797                ['175M', '350M', '600M', '1B']
  apertus-fwEdu90-fw210-seed1904                ['175M', '350M', '600M', '1B']
  apertus-fwEdu90-fw210-seed28                  ['175M', '350M', '600M', '1B']
  apertus3-a06                                  ['1B', '3B']
  gemma-3-pt                                    ['270M', '1B', '4B', '12-14B', '27-32B']

Buckets present: ['175M', '270M', '350M', '600M', '1B', '1.7B', '3B', '4B', '7-9B', '12-14B', '27-32B', '70B']


---
title: Experimental Setup — Benchmarks
subtitle: "Pretraining suite: 22 benchmark families, 12+ languages"
---

| Category | Pretraining (base suite) | + Midtraining | + Post-training |
|----------|--------------------------|---------------|-----------------|
| **LM & Completion** | hellaswag, piqa, global_piqa, xstorycloze, xwinograd, xcopa, xnli, paws, multiblimp | — | — |
| **Commonsense & Reasoning** | arc, commonsense_qa, openbookqa, winogrande | mgsm_direct | bbh, drop, gsm8k_cot, hendrycks_math, mathqa, mgsm |
| **Knowledge & QA** | mmlu, global_mmlu_full, belebele, triviaqa, squad, include_base_44, agieval, arabic_leaderboard, truthfulqa | — | global_mmlu (gen), blend, cultural_bench, mmlu_flan_cot |
| **Code** | — | — | humaneval, mbpp |
| **Instruction & Safety** | — | — | ifeval, multi-if, acp_bench, bbq, toxigen, harmbench, aya_redteaming, polyglotoxicity |

- **Languages**: en, es, ar, zh, ru, hi, vi, eu, ja, sw, tr, th, te
- **Evaluation**: log-prob 0-shot (pretraining), generative (post-training)

---
layout: section
---

# Results

Accuracy vs. compute

---
layout: bullets
title: Results
subtitle: Accuracy vs Compute curves
icon: "📈"
---

For each benchmark we plot **accuracy vs. compute (FLOPs)**, one training curve per data mixture, across the four model sizes.

- **Signal** = $(\max-\min)/\text{mean}$ of the per-mixture final scores at 1B
- **Noise** = variance of each curve over its late checkpoints
- The benchmarks that **separate the mixtures most** (top Signal at 1B): `agieval_sat` (0.27), `belebele` (0.24), `arabic_leaderboard`

A benchmark is only useful if its mixtures separate by **more than the noise**.

---
layout: figure
image: /results/acc_vs_flops_regimes.png
fit: contain
height: 80vh
title: Results
subtitle: Accuracy vs Compute curves
---

---
layout: figure
image: /results/acc_vs_flops_belebele.png
fit: contain
height: 80vh
title: Results
subtitle: Accuracy vs Compute (Belebele)
---

---
title: Results
subtitle: Are the models even above chance?
---

A benchmark only carries signal if the models clear **chance** (`1 / #options`) by a margin. Per (benchmark, size) we average the score over all models; **above random** iff `mean > chance + 0.05`. Only **44 of 118 benchmarks** clear chance at any size — and it's an **answer-count** effect:

| MCQA options | Random chance | Benchmarks above random |
| :----------: | :-----------: | :---------------------: |
| **2** (completion / minimal pair) | 0.50 | **28 / 42** |
| **3** (`xnli`) | 0.33 | **7 / 11** |
| **4** (knowledge MCQA) | 0.25 | **9 / 63** |
| **5** | 0.20 | **0 / 2** |

- The only 4-option survivors: **`hellaswag`** (contentful completions) and English **`arc_easy` / `arc_challenge`**
- Translated knowledge MCQA sit **at chance**: `belebele` 0/12, `global_mmlu(_full)` 0/16, `truthfulqa` 0/8, `arc_<lang>` 0/9
- **The gate is enforced**: random (benchmark, size) cells are dropped from SNR and every downstream RQ

<!--
Reproducible: src/signal-and-noise/multilingual/above_random.py --pool custom_swissai_hf
→ above_random_scores.csv (mean score per benchmark×size) + above_random_mask.csv (0/1).
Per (benchmark, size): mean primary_score over all models at that size (final ckpt) >
1/n_options + 0.05. n_options is intrinsic benchmark metadata (above_random.N_OPTIONS) — the
gate reads NO RQ output, so RQs depend on it, not the reverse. run_apertus_snr_variants.py
imports scores_and_mask and NaN-s random SNR cells. Above chance ≠ reliable (xnli clears
chance yet has DA-size = 0). Foreshadows RQ3: fewer options → higher SNR.
-->

---
layout: bullets
title: Results
subtitle: Signal ≠ Reliability
icon: "⚠️"
---

The top-Signal families (`belebele`, `agieval_sat`, `arabic_leaderboard`) are exactly the ones the **above-random gate removes** — they sit *at chance*, so they never enter the SNR analysis at all.

- They swing a lot with the data mixture (**high signal**) **but are also high-noise**, and the models can't even clear chance on them
- Raw mixture sensitivity **is not** reliability → high Signal ≠ usable SNR
- SNR (signal **÷** noise) instead ranks `multiblimp` / `hellaswag` (low absolute swing, very low noise) at the **top**


---
layout: default
title: Results - Improvement
subtitle: Pretrain models with fewer languages
---

- Currently: 200 languages
- **ATLAS**: Adaptive Transfer Scaling Laws for Multilingual Pretraining (arXiv 2510.22037)
  - For similar loss to a 1B model on 100B English tokens, over $n$ languages:
  - Model size $= 1\text{B} \times n^{0.243}$
  - Dataset size $= 100\text{B} \times n^{0.728}$


---
layout: section
---

# Analysis

Four research questions

---
layout: default
title: Analysis
subtitle: Research Questions
---

<Block type="info" title="RQ1 — SNR Definition">

Which SNR definition best correlates with decision accuracy across languages? Does it hold across seeds?

</Block>

<Block type="info" title="RQ2 — Framework Generalization">

Do our Apertus-derived SNR rankings transfer to the AllenAI DataDecide corpus on shared benchmarks?

</Block>

<Block type="info" title="RQ3 — Subsampling">

Can subsets of subtasks or individual items give higher SNR than the full benchmark?

</Block>

<Block type="info" title="RQ4 — Benchmark Creation">

What benchmark design features (curation, format, option count, length) predict high SNR?

</Block>



<!-- The four sections below each follow: research question → methodology → highlighted
results → proposed methodology improvements. -->

---
layout: section
---

# RQ1 — SNR Definition

Which SNR variant best predicts decision accuracy? Does it hold across seeds?

<!-- 
layout: bullets
title: RQ1 — SNR Definition
subtitle: Methodology
icon: "⚙️"

- **22 SNR variants** × sizes × ~115 multilingual parent tasks
- Pools: pure custom 1 / 2 / 3 seeds, and **`custom_swissai_hf`** (3 seeds + external pretraining models, spanning 175M → 32B)
- **DA-size**: small-last-ckpt → 1B-last-ckpt rank agreement
- **DA-ckpt**: within-size early → late ckpt agreement (relative-fraction early ckpts let external trajectories enter)
- Per-language correlation: Pearson r between $\log_{10}(\text{SNR})$ and DA
- Generalization check: pick best variant on 2 seeds, evaluate on held-out seed 1904
-->

---
layout: figure
image: /results/top_variants_overall.png
fit: contain
height: 33vh
title: RQ1 — SNR Definition
subtitle: "Top variants across languages (DA-size & DA-ckpt)"
---

---
layout: bullets
title: RQ1 — SNR Definition
subtitle: Highlighted results
icon: "✅"
---

- **The dispersion family wins.** On `custom_swissai_hf`, `dist_std` is the global best — DA-size r **0.32** (far ahead of the next variant at 0.14), DA-ckpt r **0.43**, overall **0.38**
- **DA-ckpt is led by the mean-pairwise-distance / relative-spread cluster** (`rel_mpd`, `mpd`, `mpsd` ≈ **0.51**) — all dispersion-family members
- **Better results with more seeds**: top DA-size r climbs **0.31 → 0.33 → 0.39** (1 → 2 → 3 seeds)
- `tukey`, `projection` (depth): **r ≤ 0** with DA → useless at this pool size
- **Variant ranking transfers to a held-out seed** — Spearman ρ **+0.80** (DA-size), **+0.93** (DA-ckpt); the exact per-language argmax does **not** (family-level agreement 14% / 36%)

<!-- BEGIN auto:rq1-results (snr_definition_postprocess.py) -->
---
title: RQ1 — SNR Definition
subtitle: "Results (auto) — most reliable benchmark per language (`dist_std` @ 1B)"
---

| lang | top benchmark | SNR | DA-ckpt@1B |
|---|---|---|---|
| ar | `multiblimp_arb` | 2.6 | 0.87 |
| en | `xwinograd_en` | 2.4 | 0.83 |
| es | `multiblimp_spa` | 3.4 | 0.85 |
| eu | `multiblimp_eus` | 1.3 | 0.64 |
| hi | `multiblimp_hin` | 4.9 | 0.85 |
| ja | `xwinograd_jp` | 2.3 | 0.76 |
| ru | `multiblimp_rus` | 7.1 | 0.86 |
| th | `xnli_th` | 1.3 | 0.75 |
| tr | `multiblimp_tur` | 2.7 | 0.79 |
| vi | `xcopa_vi` | 1.6 | 0.76 |
| zh | `xcopa_zh` | 1.6 | 0.61 |

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq1-results -->

---
layout: figure
image: /results/top_benchmarks_per_language.png
fit: contain
height: 80vh
title: RQ1 — SNR Definition
subtitle: "Top-5 benchmarks per language by SNR (dist_std @ 1B)"
---

---
layout: bullets
title: RQ1 — SNR Definition
subtitle: Proposed methodology improvements
icon: "💡"
---

- **Bootstrap CIs** on per-language Pearson r and cross-pool Spearman ρ
- Recommend a **family** (dispersion / relative-spread), not an exact variant — only the family transfers
- Use a **larger DA-size target** (e.g. Apertus-8B) instead of the not-fully-converged 1B custom model

---
layout: section
---

# RQ2 — Framework Generalization

Do the SNR variants we recommend correlate with other benchmark reliability frameworks? In particular with AllenAI and FineTasks?


<!--
layout: bullets
title: RQ2 — Framework Generalization
subtitle: Methodology
icon: "⚙️"


- **Our side**: 22 SNR variants × sizes × 7 shared English tasks (`custom_swissai_hf` pool)
- **AllenAI side**: same 22 variants on the DataDecide ladder (25 mixes × 5 ckpts, 150M–1B)
- **Headline axis**: $\log_{10}(\text{SNR}_{1B})$ on each corpus, Pearson r over the 7 shared tasks
- Top-K agreement: intersection / Jaccard at K ∈ {5, 10, 20}
-->

---
layout: figure
image: /results/snr_apertus_vs_snr_allenai_star_discrepancy_shifted.png
fit: contain
height: 30vh
title: RQ2 — Framework Generalization
subtitle: "Apertus vs AllenAI SNR — pure 3-seed pool, best variant (star_discrepancy_shifted)"
---

---
layout: bullets
title: RQ2 — Framework Generalization
subtitle: Highlighted results
icon: "✅"
---

- On the pure 3-seed pool the SNR **values *and* rank order agree** across corpora over the 7 shared English tasks: **Pearson r = 0.92, Spearman ρ = 0.93** (best variant `star_discrepancy_shifted`)
- Value correlation rises with seeds: **0.75** (1 seed) → **0.92** (3 seeds)
- **Dispersion + discrepancy families transfer**; the relative-spread family (incl. AllenAI's own default `rel_std`) does **not**
- Only **7** English tasks overlap → top-K set overlap is uninformative (K ≥ 7 = whole universe ⇒ Jaccard trivially 1.0); **the correlation, not the overlap, is the result**
<!--
- n = 7 shared (arc_challenge, arc_easy, csqa, hellaswag, mmlu, openbookqa, piqa): report Pearson r (values) + Spearman ρ (rank), NOT top-K Jaccard. The pure 3-seed pool is the like-for-like comparison; on the externals pool the above-random gate drops the at-chance MCQA, shrinking the shared set to 4 (mpsd r=0.996, ρ=1.0 over those 4) — use the pure pool for the cross-corpus claim.
-->


<!-- BEGIN auto:rq2-results (allenai_comparison/analyze.py) -->
---
title: RQ2 — Framework Generalization
subtitle: "Results (auto) — cross-corpus agreement with AllenAI by pool"
---

| pool | best variant | Pearson r | Spearman ρ | n_shared |
|---|---|---|---|---|
| `seeds_1904` | `dispersion` | 0.75 | 0.79 | 7 |
| `seeds_28_1797` | `discrepancy` | 0.84 | 0.64 | 7 |
| `seeds_28_1797_1904` | `star_discrepancy_shifted` | 0.92 | 0.93 | 7 |
| `custom_swissai_hf` | `mpsd` | 1.00 | 1.00 | 4 |

Pure pools share all 7 English tasks; `custom_swissai_hf` shares fewer after the above-random gate — the pure 3-seed pool is the like-for-like fit.

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq2-results -->

---
layout: bullets
title: RQ2 — Framework Generalization
subtitle: Proposed improvements
icon: "💡"
---

- **Increase the shared tasks**: add `mmlu` + `mmlu_pro`, BBH, AGI-Eval
- **Bootstrap CIs** on the cross-corpus r (n=7 is fragile)
- Compare with other frameworks:
  - **FineTasks**
  - **SMART** filtering (Gupta et al., 2024)
  - How to Select Datapoints for Efficient Human Evaluation of NLG Models? (arXiv 2501.18251)
  - **Chen et al. (2024)** — scaling behavior of downstream tasks
  - **Zhou et al. (2025)** — item response theory for benchmark reliability


---
layout: section
---

# RQ3 — Subsampling

Can a subset of subtasks (languages, subjects) or individual items give higher SNR than the full benchmark?

---
layout: bullets
title: RQ3 — Subsampling
subtitle: Methodology
icon: "⚙️"
---

- Model pool: 36 Apertus + external pretraining models (`custom_swissai_hf`), last-5 ckpts per model
- **Subtask level,** three cases: language subset, MMLU subject subset, subject × language
- **Per-sample level:** rank by sample SNR, remove the ones with signal = 0, add samples greedily, record cumulative SNR; best subset = argmax of the curve (random-order baseline alongside)

---
layout: figure
image: /results/subtask_belebele_languages.png
fit: contain
height: 120vh
title: RQ3 — Subsampling
subtitle: "Subtask · per language — language subset within a family (Belebele)"
---

---
layout: figure
image: /results/global_mmlu_full_subjects.png
fit: contain
height: 30vh
title: RQ3 — Subsampling
subtitle: "Subtask · per subject — MMLU subject subset curves per size"
---

---
layout: figure
image: /results/per_sample_xcopa_sw.png
fit: contain
height: 30vh
title: RQ3 — Subsampling
subtitle: "Per-sample — cumulative SNR over ranked items (xcopa_sw)"
---

---
layout: bullets
title: RQ3 — Subsampling
subtitle: Highlighted results
icon: "✅"
---

- **Best subset usually beats the full set substantially** — Global-MMLU 175M `+1.52` SNR (`medical_genetics` alone, full 2.12 → 3.65); per-language GMF-tr 1B `+1.56`; Belebele 350M `+1.16`
- **Subject subsets beat language subsets** — MMLU subject (mean over 10 langs) gives the most reliable gains
- **Stability is uneven**: MMLU **subject** picks recur across pools; **language** and **subject × language** picks often flip
- **Per-item (per-sample) ranking is mostly noise** (cross-size Spearman ≈ 0.05) → tiny argmax (2.5%) subsets overfit and collapse out-of-sample

<!-- BEGIN auto:rq3-results (smooth_subtasks.py) -->
---
title: RQ3 — Subsampling
subtitle: "Results (auto) — top subset gains (SNR: full → best subset)"
---

| case | task | size | full → best SNR | +gain |
|---|---|---|---|---|
| global_mmlu_full_per_language | `global_mmlu_full_tr` | 1B | 1.85 → 3.41 | +1.56 |
| global_mmlu_full_subjects | `global_mmlu_full` | 175M | 2.12 → 3.65 | +1.52 |
| per_benchmark | `paws` | 3B | 0.37 → 1.81 | +1.44 |
| global_mmlu_full_per_language | `global_mmlu_full_sw` | 600M | 1.68 → 3.02 | +1.34 |
| global_mmlu_full_per_language | `global_mmlu_full_vi` | 350M | 1.97 → 3.31 | +1.34 |
| global_mmlu_full_per_language | `global_mmlu_full_zh` | 175M | 2.15 → 3.46 | +1.31 |
| global_mmlu_full_subjects | `global_mmlu_full` | 600M | 2.18 → 3.45 | +1.27 |
| per_benchmark | `truthfulqa` | 3B | 0.66 → 1.92 | +1.26 |

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq3-results -->

---
layout: bullets
title: RQ3 — Subsampling
subtitle: Proposed improvements to make subset selection trustworthy

icon: "💡"
---

- **Pick a safer subset.** The single best subset is often a fluke. Instead, take the *largest* subset that ties with the peak and still beats the full benchmark. Bigger subsets are more stable.
- **Average out the noise.** Per-item scores are too noisy to trust. Pool them across seeds and sizes — or pick whole topics, not single items. Coarser units are stabler.
- **Measure the trust.** Test each pick on a held-out seed, and report how often it survives.

We will do a thorough subsampling analysis on INCLUDE-v2.

---
layout: section
---

# RQ4 — Benchmark Creation

What makes a benchmark high-SNR? What benchmark design features (curation, format, option count, item length) predict SNR?

---
layout: bullets
title: RQ4 — Benchmark Creation
subtitle: Methodology
icon: "⚙️"
---

- **9 above-random benchmark families** (the gate already drops every at-chance 4-option MCQA); SNR signal = per-family median SNR @ 1B from the `custom_swissai_hf` pool
- Per-family aggregate: median across the family's per-language aggregate tasks
- Three phases: **curation** (Phase 0), **task format** (Phase A), **item lengths** (Phase B)
- Statistical tests: Kruskal-Wallis (categorical), Spearman ρ (continuous)
- Length features: 100 items/family sampled from each HF dataset

---
layout: figure
image: /results/snr_per_family_ranked.png
height: 30vh
title: RQ4 — Benchmark Creation
subtitle: Per-family median SNR — 9 above-random families (custom_swissai_hf)
---

---
layout: bullets
title: RQ4 — Benchmark Creation
subtitle: Highlighted results
icon: "✅"
---

- **The answer-count penalty is now upstream, in the gate.** It drops every 4-option translated knowledge MCQA (`belebele`, `global_mmlu_full`, `truthfulqa`) before SNR is even computed
- Top SNR (all 2-option): **multiblimp** (3.9), **paws** (2.5), **xwinograd** (2.5), **xstorycloze** (2.3), **xcopa** (2.1)
- The only 4-option survivors are **`hellaswag`** (2.1) and English **`arc`** (2.0) — contentful; bottom: **global_piqa** (1.5), **xnli** (1.2)
- **Among the 9 survivors no single design feature is significant** (n_options KW H = 1.8, p = 0.18; format H = 0, p = 1.0) — and **curation still explains nothing** (H = 0.5, **p = 0.78**)
- Mechanism: each option adds another noisy log-likelihood estimate to rank → 2-option comparisons are sharper

<!-- BEGIN auto:rq4-results (benchmark_creation/analyze.py) -->
---
title: RQ4 — Benchmark Creation
subtitle: "Results (auto) — per-family SNR, above-random survivors"
---

| family | median SNR | n_opts | format |
|---|---|---|---|
| `multiblimp` | 3.85 | 2 | minimal_pair |
| `paws` | 2.55 | 2 | classification |
| `xwinograd` | 2.48 | 2 | completion |
| `xstorycloze` | 2.27 | 2 | completion |
| `xcopa` | 2.06 | 2 | completion |
| `hellaswag` | 2.05 | 4 | completion |
| `arc` | 2.05 | 4 | mcq_question_only |
| `global_piqa_completions` | 1.45 | 2 | completion |
| `xnli` | 1.15 | 3 | classification |

<style>
.slidev-layout table { font-size: 0.7em; }
</style>
<!-- END auto:rq4-results -->

---
layout: bullets
title: RQ4 — Benchmark Creation
subtitle: Proposed methodology improvements
icon: "💡"
---

- **Controlled comparison**: hold format constant, vary curation (HellaSwag MT vs XStoryCloze human translation)
- Replace marginal KW tests with a **single regression** on (format + n_options + curation)
- **Add more families** (truthfulqa, mgsm, agieval, …) — n=9 survivors is underpowered
- Re-sample **length features from the full multilingual splits**, not just English

---
layout: section
---

# Conclusions

---
layout: bullets
title: Conclusions
subtitle: "Answer to the research question"
icon: "→"
---

- **RQ1:** The **dispersion** family (`dist_std`) tracks decision accuracy best (overall r **≈ 0.38**). The *family* ranking holds across seeds (Spearman ρ **+0.80 / +0.93**); the per-language argmax does not.
- **RQ2:** SNR **transfers to AllenAI DataDecide** on the 7 shared English benchmarks — cross-corpus Pearson r **0.92**, Spearman ρ **0.93** (pure 3-seed pool); dispersion + discrepancy families transfer, relative-spread does not.
- **RQ3:** **Subtask subsets beat full benchmarks** (MMLU subjects most stable); per-item selection overfits across scale.
- **RQ4:** The **above-random gate encodes the answer-count penalty** (drops 4-option knowledge MCQA); among survivors **curation has no measurable effect**.

<!-- BEGIN generated signal slides (multilingual/da_per_benchmark.py) -->

---
layout: section
---

# Appendix — Signal & Predictability across Sizes


---
title: Appendix — Above-random signal
subtitle: "Custom Apertus pretrains only · mean score per family × size (bold = beats chance + 0.05)"
---

| benchmark | rand | 175M | 350M | 600M | 1B |
|---|---|---|---|---|---|
| `multiblimp` | 0.50 | **0.90** | **0.91** | **0.92** | **0.93** |
| `piqa` | 0.50 | **0.67** | **0.70** | **0.71** | **0.73** |
| `xwinograd` | 0.50 | **0.60** | **0.64** | **0.67** | **0.70** |
| `xstorycloze` | 0.50 | 0.54 | **0.55** | **0.57** | **0.58** |
| `xcopa` | 0.50 | 0.54 | **0.55** | **0.56** | **0.56** |
| `global_piqa_completions` | 0.50 | 0.50 | 0.52 | 0.53 | 0.54 |
| `paws` | 0.50 | 0.51 | 0.51 | 0.51 | 0.52 |
| `xnli` | 0.33 | 0.38 | **0.39** | **0.40** | **0.40** |
| `hellaswag` | 0.25 | 0.29 | **0.31** | **0.33** | **0.34** |
| `arc` | 0.25 | 0.26 | 0.27 | 0.29 | **0.30** |
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
subtitle: "All models (custom + Swiss-AI/HF refs) · mean score per family × size (bold = beats chance + 0.05)"
---

| benchmark | rand | 175M | 270M | 350M | 600M | 1B | 1.7B | 3B | 4B | 7-9B | 12-14B | 27-32B | 70B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `multiblimp` | 0.50 | **0.90** | **0.91** | **0.91** | **0.92** | **0.93** | **0.94** | **0.92** | **0.96** | **0.91** | **0.94** | **0.94** | **0.98** |
| `piqa` | 0.50 | **0.67** | **0.68** | **0.70** | **0.71** | **0.73** | **0.76** | **0.77** | **0.79** | **0.79** | **0.81** | **0.81** | **0.83** |
| `xwinograd` | 0.50 | **0.60** | **0.63** | **0.64** | **0.67** | **0.70** | **0.72** | **0.76** | **0.77** | **0.76** | **0.79** | **0.79** | **0.82** |
| `xstorycloze` | 0.50 | 0.54 | 0.54 | **0.55** | **0.56** | **0.58** | **0.59** | **0.62** | **0.64** | **0.63** | **0.67** | **0.68** | **0.71** |
| `xcopa` | 0.50 | 0.54 | **0.56** | **0.55** | **0.56** | **0.57** | **0.59** | **0.60** | **0.64** | **0.60** | **0.65** | **0.64** | **0.70** |
| `global_piqa_completions` | 0.50 | 0.50 | 0.51 | 0.52 | 0.53 | 0.55 | **0.57** | **0.61** | **0.63** | **0.60** | **0.65** | **0.65** | **0.74** |
| `paws` | 0.50 | 0.51 | 0.52 | 0.51 | 0.51 | 0.53 | **0.58** | **0.57** | **0.60** | **0.57** | **0.60** | **0.61** | **0.60** |
| `agieval_sat` | 0.25 | 0.26 | 0.30 | 0.25 | 0.29 | 0.25 | **0.64** | **0.41** | **0.78** | **0.65** | **0.83** | **0.83** | **0.74** |
| `commonsense_qa` | 0.20 | 0.20 | 0.21 | 0.21 | 0.24 | 0.22 | **0.73** | **0.48** | **0.70** | **0.69** | **0.77** | **0.76** | **0.54** |
| `mmlu` | 0.25 | 0.25 | 0.26 | 0.25 | 0.27 | 0.27 | **0.61** | **0.46** | **0.65** | **0.63** | **0.71** | **0.73** | **0.65** |
| `belebele` | 0.25 | 0.24 | 0.24 | 0.25 | 0.27 | 0.26 | **0.60** | **0.43** | **0.66** | **0.55** | **0.70** | **0.69** | **0.68** |
| `xnli` | 0.33 | 0.38 | **0.39** | **0.39** | **0.40** | **0.41** | **0.42** | **0.42** | **0.44** | **0.41** | **0.43** | **0.43** | **0.46** |
| `global_mmlu_full` | 0.25 | 0.25 | 0.26 | 0.25 | 0.26 | 0.25 | **0.46** | **0.38** | **0.52** | **0.46** | **0.56** | **0.56** | **0.53** |
| `hellaswag` | 0.25 | 0.29 | 0.29 | **0.31** | **0.33** | **0.34** | **0.36** | **0.40** | **0.41** | **0.40** | **0.44** | **0.45** | **0.49** |
| `global_mmlu` | 0.25 |  |  | 0.25 |  |  |  | **0.49** |  |  |  |  |  |
| `arc` | 0.25 | 0.26 | 0.25 | 0.27 | 0.29 | **0.30** | **0.36** | **0.37** | **0.43** | **0.40** | **0.46** | **0.46** | **0.47** |
| `truthfulqa` | 0.25 | 0.26 | **0.35** | 0.26 | **0.35** | **0.35** | **0.38** | **0.35** | **0.38** | **0.38** | **0.38** | **0.37** | **0.40** |
| `arabic_leaderboard_alghafa_mcq_exams_test` | 0.25 | 0.24 | 0.25 | 0.25 | 0.24 | 0.24 | **0.37** | **0.36** | **0.43** | **0.36** | **0.45** | **0.45** | **0.48** |
| `agieval_logiqa` | 0.25 | 0.22 | 0.21 | 0.22 | 0.23 | 0.23 | **0.32** | 0.26 | **0.39** | **0.35** | **0.42** | **0.39** | **0.36** |
| `agieval` | 0.25 | 0.18 | 0.18 | 0.18 | 0.19 | 0.18 | **0.35** | 0.25 | **0.41** | **0.36** | **0.46** | **0.46** | **0.36** |
| `openbookqa` | 0.25 | 0.20 | 0.21 | 0.22 | 0.24 | 0.26 | **0.30** | **0.30** | **0.33** | **0.34** | **0.36** | **0.36** | **0.38** |
| `truthfulqa_mc1` | 0.25 | 0.23 | 0.24 | 0.22 | 0.23 | 0.23 | **0.32** | 0.27 | **0.32** | **0.31** | **0.35** | **0.34** | **0.36** |
| `agieval_lsat` | 0.20 | 0.22 | 0.24 | 0.22 | 0.21 | 0.21 | **0.25** | 0.23 | 0.21 | 0.22 | 0.25 | 0.24 | 0.21 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "English (en) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hellaswag` | **0.92** | **0.86** | **0.83** | **0.89** | **0.86** | **0.80** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `xstorycloze` | **0.75** | **0.83** | **0.81** | **0.81** | **0.83** | **0.76** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `openbookqa` | **0.78** | **0.75** | 0.67 | **0.81** | 0.72 | **0.84** | **1.00** | **1.00** | 0.00 | **1.00** | **1.00** | **1.00** |
| `piqa` | **0.86** | **0.89** | **0.89** | **0.86** | **0.86** | **0.76** | **1.00** | 0.00 | **1.00** | **1.00** | **1.00** | 0.00 |
| `xwinograd` | **0.86** | **0.81** | 0.72 | **0.78** | **0.81** | 0.73 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `commonsense_qa` | 0.58 | 0.42 | 0.36 | 0.33 | 0.44 | 0.53 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `agieval_logiqa` | 0.56 | 0.61 | 0.67 | 0.44 | 0.56 | 0.44 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `arc_easy` | **0.78** | **0.81** | **0.89** | **0.86** | **0.83** | **0.78** | 0.00 | 0.00 | **1.00** | 0.00 | **1.00** | **1.00** |
| `arc_challenge` | 0.69 | 0.72 | **0.83** | **0.92** | **0.86** | **0.80** | 0.00 | 0.00 | **1.00** | 0.00 | **1.00** | **1.00** |
| `belebele` | 0.64 | 0.69 | **0.75** | 0.72 | **0.78** | **0.78** | 0.00 | 0.00 | **1.00** | **1.00** | 0.00 | **1.00** |
| `paws` | 0.50 | 0.47 | 0.42 | 0.58 | 0.53 | **0.78** | **1.00** | **1.00** | 0.00 | 0.00 | **1.00** | **1.00** |
| `xnli` | 0.64 | 0.53 | 0.33 | **0.78** | 0.42 | 0.51 | **1.00** | **1.00** | 0.00 | 0.00 | **1.00** | **1.00** |
| `agieval_sat` | 0.31 | 0.36 | 0.42 | 0.61 | 0.50 | 0.73 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_mmlu_full` | 0.39 | 0.33 | 0.64 | 0.44 | 0.36 | 0.58 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `truthfulqa_mc1` | 0.33 | 0.53 | 0.33 | 0.36 | **0.78** | 0.33 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 |
| `mmlu` | 0.50 | 0.06 | 0.58 | 0.50 | 0.47 | 0.42 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_piqa_completions` | 0.61 | 0.50 | 0.67 | 0.72 | 0.28 | 0.40 | 0.00 | 0.00 | **1.00** | 0.00 | **1.00** | **1.00** |
| `multiblimp` | 0.61 | 0.50 | **0.78** | **0.83** | 0.61 | 0.71 | **1.00** | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 |
| `agieval` | 0.25 | 0.44 | 0.61 | 0.36 | 0.47 | 0.53 | 0.00 | 0.00 | **1.00** | **1.00** | 0.00 | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Arabic (ar) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hellaswag` | **0.78** | **0.78** | 0.72 | **0.83** | **0.89** | **0.82** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `xstorycloze` | **0.75** | **0.89** | **0.83** | 0.69 | **0.75** | **0.78** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `multiblimp` | 0.69 | **0.83** | **0.83** | **0.86** | **0.75** | **0.89** | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `truthfulqa_ar_mc1` | **0.86** | 0.58 | 0.50 | 0.61 | 0.58 | 0.67 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `belebele` | 0.69 | 0.42 | 0.64 | 0.61 | 0.56 | 0.53 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `truthfulqa_ar_mc2` |  |  |  |  |  |  | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_piqa_completions` | 0.58 | 0.39 | 0.44 | 0.47 | 0.53 | 0.49 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `xnli` | 0.69 | 0.17 | 0.42 | 0.36 | 0.50 | 0.62 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `arabic_leaderboard_alghafa_mcq_exams_test` | 0.33 | 0.61 | 0.36 | 0.44 | 0.69 | 0.31 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `global_mmlu_full` | 0.47 | 0.61 | 0.50 | 0.53 | 0.53 | **0.80** | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `arc` | 0.31 | 0.72 | 0.69 | 0.42 | 0.44 | 0.64 | **1.00** | **1.00** | 0.00 | **1.00** | 0.00 | **1.00** |
| `agieval_lsat` | 0.61 | 0.36 | 0.50 | 0.53 | 0.61 | 0.64 | 0.00 | **1.00** | **1.00** | **1.00** | 0.00 | 0.00 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Spanish (es) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hellaswag` | **0.97** | **0.81** | **0.86** | **0.83** | **0.83** | **0.82** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `multiblimp` | **0.78** | **0.81** | **0.89** | 0.64 | 0.67 | **0.82** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `xstorycloze` | 0.58 | **0.75** | 0.72 | 0.72 | **0.86** | 0.71 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_piqa_completions` | **0.81** | 0.64 | 0.67 | 0.61 | 0.69 | 0.38 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `arc` | 0.53 | 0.67 | 0.67 | 0.47 | 0.47 | 0.58 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `paws` | 0.56 | 0.33 | 0.42 | 0.56 | 0.53 | 0.60 | **1.00** | **1.00** | 0.00 | **1.00** | **1.00** | **1.00** |
| `global_mmlu_full` | 0.50 | 0.36 | 0.47 | 0.47 | 0.64 | 0.73 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `xnli` | 0.39 | 0.47 | 0.44 | 0.64 | 0.61 | 0.51 | **1.00** | **1.00** | **1.00** | 0.00 | 0.00 | **1.00** |
| `truthfulqa_es_mc1` | **0.75** | 0.42 | 0.53 | 0.28 | 0.56 | 0.53 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `belebele` | 0.50 | 0.22 | 0.53 | 0.56 | 0.36 | 0.62 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `truthfulqa_es_mc2` |  |  |  |  |  |  | **1.00** | 0.00 | **1.00** | **1.00** | 0.00 | 0.00 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Basque (eu) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `truthfulqa_eu_mc2` |  |  |  |  |  |  | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `xstorycloze` | **0.92** | 0.69 | **0.75** | 0.72 | 0.72 | **0.82** | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `hellaswag` | **0.75** | 0.56 | 0.61 | 0.47 | 0.58 | 0.58 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `multiblimp` | 0.64 | 0.44 | 0.69 | 0.53 | 0.61 | 0.60 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `arc` | 0.69 | **0.75** | 0.72 | **0.78** | 0.69 | **0.76** | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `xnli` | 0.47 | 0.44 | 0.56 | **0.75** | 0.64 | **0.76** | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `paws` | 0.39 | 0.50 | 0.36 | 0.67 | 0.58 | 0.56 | **1.00** | **1.00** | 0.00 | 0.00 | **1.00** | **1.00** |
| `belebele` | 0.50 | 0.56 | 0.53 | 0.56 | 0.47 | 0.67 | 0.00 | 0.00 | **1.00** | **1.00** | 0.00 | **1.00** |
| `truthfulqa_eu_mc1` | 0.36 | 0.44 | 0.56 | 0.42 | 0.64 | 0.60 | 0.00 | 0.00 | **1.00** | **1.00** | 0.00 | **1.00** |
| `xcopa` | 0.39 | 0.44 | 0.44 | 0.50 | 0.50 | **0.76** | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Hindi (hi) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `multiblimp` | **0.89** | **0.86** | **0.83** | **0.86** | **0.78** | **0.89** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `hellaswag` | **0.75** | **0.81** | **0.78** | **0.83** | **0.81** | **0.89** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `xstorycloze` | 0.61 | 0.58 | 0.64 | 0.64 | **0.81** | 0.71 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_mmlu_full` | 0.39 | 0.53 | 0.53 | 0.64 | 0.53 | 0.38 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `arc` | 0.33 | 0.58 | 0.31 | 0.42 | 0.58 | 0.67 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_piqa_completions` | 0.39 | 0.53 | 0.42 | 0.36 | 0.36 | **0.76** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `belebele` | 0.53 | 0.39 | 0.61 | 0.69 | 0.53 | 0.51 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `truthfulqa_hi_mc2` |  |  |  |  |  |  | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `xnli` | **0.78** | 0.56 | 0.72 | 0.56 | 0.72 | 0.64 | **1.00** | **1.00** | 0.00 | 0.00 | **1.00** | **1.00** |
| `truthfulqa_hi_mc1` | 0.58 | 0.67 | 0.47 | 0.53 | 0.61 | 0.49 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Japanese (ja) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `xwinograd` | 0.61 | **0.78** | 0.67 | 0.61 | 0.67 | 0.56 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `belebele` | 0.42 | 0.72 | 0.64 | 0.53 | 0.56 | **0.84** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `paws` | 0.44 | **0.89** | 0.56 | 0.50 | 0.56 | 0.64 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_mmlu_full` | 0.61 | 0.58 | 0.58 | 0.53 | 0.53 | 0.53 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_piqa_completions` | 0.56 | 0.58 | 0.56 | 0.69 | **0.78** | 0.62 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Russian (ru) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `truthfulqa_ru_mc2` |  |  |  |  |  |  | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `hellaswag` | **0.92** | **0.92** | **0.92** | **0.89** | **0.89** | 0.73 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `xstorycloze` | 0.72 | **0.81** | **0.81** | **0.86** | **0.81** | **0.84** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `multiblimp` | **0.89** | **0.89** | **0.81** | **0.89** | **0.86** | **0.82** | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `global_piqa_completions` | **0.81** | 0.72 | **0.75** | **0.86** | **0.83** | **0.84** | **1.00** | **1.00** | 0.00 | **1.00** | **1.00** | **1.00** |
| `belebele` | 0.61 | 0.47 | **0.75** | 0.58 | 0.64 | 0.47 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `xwinograd` | 0.61 | 0.69 | 0.53 | 0.58 | 0.58 | 0.42 | **1.00** | **1.00** | 0.00 | **1.00** | **1.00** | **1.00** |
| `xnli` | 0.47 | 0.50 | 0.25 | 0.36 | 0.44 | 0.64 | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** | **1.00** |
| `global_mmlu_full` | 0.39 | 0.56 | 0.69 | 0.56 | 0.47 | 0.44 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `arc` | 0.33 | 0.58 | 0.44 | 0.47 | 0.72 | 0.49 | **1.00** | **1.00** | 0.00 | **1.00** | 0.00 | **1.00** |
| `truthfulqa_ru_mc1` | 0.33 | 0.47 | 0.31 | 0.69 | 0.47 | 0.36 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Swahili (sw) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `xcopa` | 0.56 | 0.64 | **0.78** | 0.42 | 0.67 | 0.56 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `xstorycloze` | 0.58 | 0.72 | 0.67 | 0.53 | **0.75** | 0.69 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `xnli` | 0.39 | 0.39 | 0.33 | 0.50 | 0.67 | 0.64 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_mmlu_full` | 0.47 | 0.47 | 0.56 | **0.78** | 0.58 | 0.49 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_piqa_completions` | 0.39 | 0.50 | 0.72 | 0.44 | 0.50 | 0.40 | **1.00** | **1.00** | **1.00** | 0.00 | 0.00 | **1.00** |
| `belebele` | 0.47 | 0.50 | 0.64 | 0.69 | 0.44 | 0.51 | 0.00 | 0.00 | **1.00** | **1.00** | 0.00 | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Thai (th) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `xnli` | 0.72 | 0.67 | **0.81** | 0.56 | 0.69 | 0.69 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `belebele` | 0.61 | 0.67 | 0.64 | 0.72 | 0.64 | **0.78** | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `xcopa` | 0.31 | 0.64 | 0.69 | 0.39 | 0.11 | 0.53 | **1.00** | **1.00** | 0.00 | **1.00** | **1.00** | **1.00** |
| `global_piqa_completions` | 0.56 | 0.67 | 0.47 | 0.44 | 0.42 | 0.67 | **1.00** | **1.00** | 0.00 | **1.00** | 0.00 | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Turkish (tr) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `multiblimp` | **0.89** | 0.69 | **0.89** | **0.75** | **0.94** | **0.80** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_piqa_completions` | **0.78** | 0.69 | 0.64 | 0.64 | 0.64 | 0.69 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_mmlu_full` | 0.56 | 0.39 | 0.64 | 0.50 | 0.42 | 0.44 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `belebele` | **0.86** | 0.47 | 0.61 | 0.61 | 0.58 | 0.62 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `xnli` | 0.72 | 0.61 | 0.64 | 0.72 | 0.53 | 0.44 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `xcopa` | 0.42 | 0.53 | 0.25 | 0.33 | 0.50 | 0.56 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Vietnamese (vi) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `hellaswag` | **0.83** | **0.81** | **0.92** | **0.86** | **0.75** | **0.78** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `truthfulqa_vi_mc1` | 0.72 | **0.86** | 0.33 | 0.58 | 0.56 | 0.24 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `arc` | 0.64 | 0.69 | 0.53 | 0.72 | 0.56 | 0.71 | **1.00** | **1.00** | 0.00 | **1.00** | **1.00** | **1.00** |
| `xnli` | 0.39 | 0.61 | 0.53 | 0.61 | 0.64 | 0.71 | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** | **1.00** |
| `global_piqa_completions` | 0.53 | 0.47 | **0.75** | 0.50 | 0.50 | 0.62 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `xcopa` | 0.53 | 0.69 | 0.67 | **0.78** | **0.81** | **0.84** | **1.00** | **1.00** | 0.00 | **1.00** | 0.00 | **1.00** |
| `global_mmlu_full` | **0.86** | 0.58 | 0.72 | 0.50 | 0.58 | 0.40 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `belebele` | 0.58 | 0.56 | 0.50 | 0.47 | 0.42 | 0.64 | 0.00 | 0.00 | **1.00** | **1.00** | 0.00 | **1.00** |
| `truthfulqa_vi_mc2` |  |  |  |  |  |  | 0.00 | **1.00** | **1.00** | **1.00** | 0.00 | 0.00 |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

---
title: Appendix — Decision accuracy across sizes
subtitle: "Chinese (zh) · small→large size pair (bold ≥ 0.75)"
---

| benchmark | 175M→350M | 175M→600M | 175M→1B | 350M→600M | 350M→1B | 600M→1B | 1B→12-14B | 1B→27-32B | 4B→12-14B | 7-9B→12-14B | 7-9B→27-32B | 12-14B→27-32B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `truthfulqa_zh_mc2` |  |  |  |  |  |  | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `agieval_logiqa` | 0.47 | 0.64 | 0.39 | 0.44 | 0.58 | 0.67 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `global_piqa_completions` | 0.67 | 0.67 | 0.39 | 0.50 | 0.56 | 0.40 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `xwinograd` | 0.36 | 0.56 | 0.42 | 0.64 | 0.50 | 0.36 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |
| `xstorycloze` | 0.44 | 0.50 | 0.67 | 0.61 | 0.61 | 0.71 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `xcopa` | 0.50 | 0.33 | 0.42 | 0.67 | 0.64 | 0.67 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 | **1.00** |
| `arc` | 0.44 | 0.47 | 0.39 | 0.42 | 0.50 | 0.60 | **1.00** | **1.00** | 0.00 | **1.00** | **1.00** | **1.00** |
| `paws` | 0.39 | 0.56 | 0.58 | 0.44 | 0.58 | 0.51 | **1.00** | 0.00 | **1.00** | **1.00** | **1.00** | 0.00 |
| `global_mmlu_full` | 0.36 | 0.56 | 0.64 | 0.42 | 0.28 | 0.51 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** |
| `truthfulqa_zh_mc1` | 0.28 | 0.58 | 0.53 | 0.42 | 0.36 | 0.47 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** | 0.00 |
| `belebele` | 0.67 | 0.25 | 0.50 | 0.58 | 0.72 | 0.60 | 0.00 | 0.00 | **1.00** | **1.00** | 0.00 | **1.00** |
| `xnli` | 0.47 | 0.58 | 0.67 | 0.44 | 0.47 | 0.67 | **1.00** | **1.00** | 0.00 | 0.00 | 0.00 | **1.00** |

<style>
.slidev-layout table { font-size: 0.52em; line-height: 1.15; }
.slidev-layout th, .slidev-layout td { padding: 1px 6px; }
</style>

<!-- END generated signal slides -->
