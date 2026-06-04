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

Of **118 task–language pairs**, **65 (55%)** are usually above chance:

| MCQA options | Random chance | Pairs usually above random |
| :----------: | :-----------: | :------------------------: |
| **2** (completion / minimal pair) | 0.50 | **33 / 42** |
| **3** (`xnli`) | 0.33 | **11 / 11** |
| **4** (knowledge MCQA) | 0.25 | **19 / 63** |
| **5** | 0.20 | **2 / 2** |

- Random: `belebele` 2/12, multilingual `arc` 2/11, `global_mmlu` 0/6
- **Exception: `hellaswag`** (4-option but *contentful* completions) → 9/9 above chance
- A pair **below chance carries no usable signal**

<!--
Reproducible: src/signal-and-noise/multilingual/above_random.py --pool custom_swissai_hf
→ results/acc_vs_flops/pretraining/custom_swissai_hf/above_random{,_summary}.csv.
Custom pretraining suite (primary_score across all sizes×mixes×seeds); "usually above" =
median over all evals > 1/n_options. Baselines from RQ3 per_family_snr.csv (canonical) +
flagged supplementary counts. truthfulqa/agieval option counts are approximate (variable per
item). Above chance ≠ reliable (xnli clears chance everywhere yet has DA-size = 0).
Foreshadows RQ3: fewer options → higher SNR.
-->

---
layout: bullets
title: Results
subtitle: Signal ≠ Reliability
icon: "⚠️"
---

The top-Signal families (`belebele`, `agieval_sat`, `arabic_leaderboard`) are exactly the **lowest-SNR families** in the analysis (RQ3) — `belebele` is *last* by SNR.

- They swing a lot with the data mixture (**high signal**) **but are also high-noise**
- Signal-to-noise stays **low** → raw mixture sensitivity **is not** reliability
- This is why SNR (signal **÷** noise) ranks `multiblimp` / `hellaswag` (low absolute swing, very low noise) **above** `belebele`


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

- **`rel_mpd` and the dispersion / relative-spread family win** — top variant on `custom_swissai_hf`: DA-size r **0.400**, DA-ckpt r **0.519**
- **Better results with more seeds**: top DA-size r climbs 0.31 → 0.33 → 0.39 (1 → 2 → 3 seeds)
- **Adding external models mainly lifts DA-ckpt** (0.379 → 0.519 at 3 seeds)
- `tukey`, `projection` (depth): **r ≈ 0** with DA → useless at this pool size
- **Variant ranking transfers to a held-out seed** — Spearman ρ **+0.80** (DA-size), **+0.93** (DA-ckpt); exact per-language argmax does not (family-level agreement 14% / 36%)

---
title: RQ1 — SNR Definition
subtitle: "Highest-SNR benchmark per language (rel_mpd)"
---

Most reliable benchmark in each language under the chosen definition (`rel_mpd` SNR @ 1B):

<div class="grid grid-cols-2 gap-x-8">

<div>

| lang | top benchmark | SNR |
|---|---|---:|
| ar | **`hellaswag_ar`** | 6.9 |
| en | **`mmlu`** | 6.0 |
| es | **`hellaswag_es`** | 4.9 |
| eu | **`xnli_eu`** | 8.5 |
| hi | **`hellaswag_hi`** | 7.8 |
| ja | **`xwinograd_jp`** | 5.2 |

</div>

<div>

| lang | top benchmark | SNR |
|---|---|---:|
| ru | **`multiblimp_rus`** | 8.8 |
| sw | **`xstorycloze_sw`** | 5.2 |
| th | **`xnli_th`** | 4.7 |
| tr | **`multiblimp_tur`** | 4.6 |
| vi | **`hellaswag_vi`** | 5.1 |
| zh | **`belebele_zho_Hans`** | 3.7 |

</div>

</div>

**`hellaswag` (5 languages), `multiblimp`, and `xstorycloze`** recur as the most reliable formats; `xnli` / `belebele` lead only where those are absent.

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
image: /results/snr_apertus_vs_snr_allenai_rms_deviation.png
fit: contain
height: 30vh
title: RQ2 — Framework Generalization
subtitle: "Apertus vs AllenAI SNR — best variant (rms_deviation)"
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
- n = 7 shared (arc_challenge, arc_easy, csqa, hellaswag, mmlu, openbookqa, piqa): report Pearson r (values) + Spearman ρ (rank), NOT top-K Jaccard. Spearman noisy at n=7 / variant-dependent (comprehensive-pool rms_deviation gives ρ=0.64).
-->


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

- **11 benchmark families**; SNR signal = `snr_mpd_1B` from the `custom_swissai_hf` pool
- Per-family aggregate: median across the family's per-language aggregate tasks
- Three phases: **curation** (Phase 0), **task format** (Phase A), **item lengths** (Phase B)
- Statistical tests: Kruskal-Wallis (categorical), Spearman ρ (continuous)
- Length features: 100 items/family sampled from each HF dataset

---
layout: figure
image: /results/snr_per_family_ranked.png
height: 30vh
title: RQ3 — Benchmark Creation
subtitle: Per-family median SNR (custom_swissai_hf)
---

---
layout: bullets
title: RQ4 — Benchmark Creation
subtitle: Highlighted results
icon: "✅"
---

- **Task design beats curation.** Fewer answer options → higher SNR is the strongest design signal (KW H = 3.7, **p = 0.055**), with task format close behind (H = 5.0, p = 0.080)
- **Curation method explains nothing** (family/curation KW H = 0.77, **p = 0.68**) — once task design is fixed, curation doesn't matter
- Top SNR: **multiblimp** (3.9), **xwinograd** (2.5), **xstorycloze** (2.4)
- Bottom: **global_mmlu_full** (0.8), **arc** (0.8), **belebele** (0.7) — all 4-option MCQ
- Mechanism: each option adds another noisy log-likelihood estimate to rank → 2-option comparisons are sharper

---
layout: bullets
title: RQ4 — Benchmark Creation
subtitle: Proposed methodology improvements
icon: "💡"
---

- **Controlled comparison**: hold format constant, vary curation (HellaSwag MT vs XStoryCloze human translation)
- Replace marginal KW tests with a **single regression** on (format + n_options + curation)
- **Add more families** (truthfulqa, mgsm, agieval, …) — n=11 is underpowered
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

- **RQ1:** The **dispersion / relative-spread** family (`rel_mpd`) tracks decision accuracy best (overall r **≈ 0.46**). The *family* ranking holds across seeds (Spearman ρ **+0.80 / +0.93**); the per-language argmax does not.
- **RQ2:.** The top-10 reliable English benchmarks are **identical** on Apertus and AllenAI DataDecide (Jaccard **1.0**, cross-corpus r **0.84–0.92**).
- **RQ3:** **Fewer answer options ⇒ higher SNR**. Task design beats curation, which has no measurable effect.
- **RQ4:** **Subtask subsets beat full benchmarks** (MMLU subjects most stable); per-item selection overfits across scale.

