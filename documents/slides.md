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
  - Introduction & Motivation
  - Related Work
  - Methodology
  - Experimental Setup
  - Results
  - Analysis — Four Research Questions
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

# Introduction & Motivation

Training multilingual LMs requires constant evaluation decisions,

but evaluation is expensive and often uninformative.

---
layout: bullets
title: Motivation
subtitle: Lack of benchmark reliability
icon: "⚠️"
---

Training LMs has a **high cost** and requires **constant decisions** (data mixtures, hyperparameters, …) guided by benchmark evaluations. However,

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

Two foundations we build on

---
layout: section
---

# Insights from the Signal-and-Noise Paper

Heineman et al. (2025), AllenAI

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

# Preliminary Multilingual SNR Analysis

Éléonore, Clara, Antoine — on AllenAI DataDecide models

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


---
layout: image-left
image: /snr_preliminary_figure_3.png
ratio: "2:2"
fit: contain
title: "Preliminary: Benchmark Noise"
subtitle: A more practical noise metric
---

- Benchmark noise correlates with checkpoint noise ($R = 0.854$)
- Using it in SNR **improves** prediction of decision accuracy:

| SNR noise metric             | R         | R²        |
| ---------------------------- | --------- | --------- |
| Checkpoint noise             | 0.760     | 0.578     |
| **Benchmark noise (k=5)**    | **0.808** | **0.653** |

- Computable from a **single evaluation run** of any model (no checkpoints needed)
- Robust across model sizes (150M–1B)

> Not just easier to compute — also more predictive of decision accuracy

---
layout: image-left
image: /snr_preliminary_figure_7.png
ratio: "2:3"
fit: contain
title: "Preliminary: Multilingual Tasks"
subtitle: "Framework reliability depends on model competence"
---

| Task subset                 | R     | R²    |
| --------------------------- | ----- | ----- |
| English-only tasks          | 0.594 | 0.353 |
| All non-English tasks       | 0.045 | 0.002 |
| Non-English (excl. 3 worst) | 0.293 | 0.086 |

Small English-first models perform **near-randomly** on underrepresented languages → uninformative rankings.

> The framework doesn't fail for multilingual settings in general — it fails when proxy models lack linguistic competence. **→ Need multilingual models.**

---
layout: bullets
title: Preliminary Analysis
subtitle: Key takeaways
icon: "→"
---

- **Framework reproduces** on English benchmarks ✅
- **Benchmark noise** is more practical AND more predictive than checkpoint noise ✅
- **Multilingual extension weakens** with English-first models — a model limitation, not a framework limitation ⚠️
- **BPB on raw corpora** yields higher decision accuracy and better SNR correlation than accuracy on downstream tasks ✅

> This motivates the present work: re-run the framework on **genuinely multilingual** models.

---
layout: section
---

# Methodology

The Signal-Aware Framework, multilingual edition

---
layout: focus
color: blue
icon: 🎯
---

## Decision Accuracy is what we ultimately want a benchmark to get right

The other metrics (signal, noise, SNR) are **proxies** we can compute cheaply during training.

---
title: Methodology — Decision Accuracy
subtitle: The target metric
---

<Block type="success" title="Decision Accuracy">

For all pairs of small models $(s_a, s_b)$ trained on datasets ($a$, $b$) and their large versions $(m_a, m_b)$, does the ranking for task $B$ hold?

$$\text{DA} = \frac{1}{|\mathcal{P}|} \sum_{(a,b) \in \mathcal{P}} \mathbb{1}\big[\text{sign}(B(s_a) - B(s_b)) = \text{sign}(B(m_a) - B(m_b))\big]$$

</Block>


<Block type="success" title="Scaling-Law Prediction Error">

Can we extrapolate performance from small to large models?

$$\text{Prediction Error} = \frac{|\text{Measured Value} - \text{True Value}|}{|\text{True Value}|}$$

</Block>

- **DA-size**: small-size → large-size rank agreement (cross-scale)
- **DA-ckpt**: within-size early → late checkpoint rank agreement

---
title: Methodology — Per Stage
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
title: Methodology — SNR Variants
subtitle: The space of candidate signal definitions
---

Signal = how much a benchmark separates models. There are **many ways to quantify "spread"** — we sweep **22 candidate variants** grouped into 5 mathematical families:

| Family | Members | Idea |
| ------ | ------- | ---- |
| **Dispersion** | `mpd`, `aad`, `rms_deviation`, `quartile_deviation`, `dist_std`, `dispersion`, `range` | Absolute spread of scores |
| **Relative-spread** | `rel_std`, `rel_mpd`, `rel_mpsd`, `iqr`, `rel_dispersion` | Spread normalized by mean (AllenAI default `rel_std`) |
| **Discrepancy** | `discrepancy`, `star_discrepancy`, `star_discrepancy_shifted`, `dispersion_shifted`, `gini` | Uniformity / inequality of the score distribution |
| **Robust** | `mad`, `mpsd` | Outlier-resistant spread |
| **Depth** | `tukey`, `projection` | Half-space statistical depth |

> Which variant best tracks decision accuracy across languages? → **Analysis RQ1**

---
layout: focus
color: blue
icon: 📐
---

## Signal and Noise candidates

The canonical formulas we instantiate per benchmark, per stage.

---
title: Methodology — Signal & Noise Candidates
subtitle: Definitions
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
| **≥ 12B** | — | gemma-3 12–27B; OLMo-3 13–32B; Qwen3 14B+ | 70B Instruct families |

- **Custom models** → controlled pretraining analysis (à la AllenAI DataDecide)
- **Open-source families** → mid/post-training coverage and scaling to 70B

---
title: Experimental Setup — Benchmarks
subtitle: "~40 multilingual benchmarks, 12+ languages"
---

| Category | Pretraining | Midtraining | Post-training |
|----------|-------------|-------------|---------------|
| **LM & Completion** | hellaswag, piqa, xnli, xcopa, paws-x, xwinograd, xstorycloze, m_arc, multiblimp, lambada | (same) | hellaswag |
| **Commonsense & Reasoning** | commonsense_qa, openbookqa, ai2_arc, winogrande, gsm8k | (same) | bbh, drop, gsm8k_cot, hendrycks_math, mgsm |
| **Knowledge & QA** | mmlu, global_mmlu, belebele, squad, triviaqa, include_base_44 | (same) | global_mmlu, truthfulqa, blend, cultural_bench |
| **Code** | — | — | humaneval, mbpp |
| **Instruction & Safety** | — | — | ifeval, acp_bench, harmbench, toxigen, bbq |

- **Languages**: en, es, ar, zh, ru, hi, vi, eu, ja, sw, tr, th, … (12+ aggregates)
- **Evaluation**: log-prob (rank-classification), **0-shot vs 5-shot**
- **INCLUDE**: regional knowledge across **120 country-language pairs** (deep-dive analysis)

---
layout: section
---

# Results

Accuracy vs. compute — the raw material for signal

---
layout: bullets
title: Results — Accuracy vs Compute
subtitle: What the curves show
icon: "📈"
---

For each benchmark we plot **accuracy vs. compute (FLOPs)**, one training curve per data mixture, across the four model sizes. Tasks are parent-aggregated (**950 → 118**); external pretraining models (a06, distillation, swiss-ai/HF) overlay final-checkpoint markers out to **70B**.

- **Signal** = $(\max-\min)/\text{mean}$ of the per-mixture final scores at 1B
- **Noise** = wobble of each curve over its late checkpoints
- The benchmarks that **separate the mixtures most** (top Signal at 1B): `agieval_sat` (0.27), `belebele` (0.24), `arabic_leaderboard`

> A benchmark is only useful if its mixtures separate by **more than the noise**.

---
layout: figure
image: /results/acc_vs_flops_belebele.png
title: Results — Accuracy vs Compute
subtitle: "Belebele — per language, external models overlaid to 70B"
---

---
layout: bullets
title: Results — Signal ≠ Reliability
subtitle: The key caveat
icon: "⚠️"
---

The top-Signal families (`belebele`, `agieval_sat`, `arabic_leaderboard`) are exactly the **lowest-SNR families** in the analysis (RQ3) — `belebele` is *last* by SNR.

- They swing a lot with the data mixture (**high signal**) **but are also high-noise**
- Signal-to-noise stays **low** → raw mixture sensitivity **is not** reliability
- This is why SNR (signal **÷** noise) ranks `multiblimp` / `hellaswag` (low absolute swing, very low noise) **above** `belebele`
- Scaling overlay (→ 70B) is **visual**: cross-size decision accuracy above 1B stays family-coverage-limited

> The whole point of SNR: divide the swing by the noise. → Motivates the four analysis questions.

---
layout: section
---

# Analysis

Four research questions

---
layout: default
title: Analysis — Four Research Questions
subtitle: "From metric design to benchmark design"
---

<Block type="info" title="RQ1 — SNR Definition">

Which SNR definition best correlates with decision accuracy across languages? Does it hold across seeds?

</Block>

<Block type="info" title="RQ2 — Framework Generalization">

Do our Apertus-derived SNR rankings transfer to the AllenAI DataDecide corpus on shared benchmarks?

</Block>

<Block type="info" title="RQ3 — Benchmark Creation">

What benchmark design features (curation, format, option count, length) predict high SNR?

</Block>

<Block type="info" title="RQ4 — Subsampling">

Can subsets of subtasks or individual items give higher SNR than the full benchmark?

</Block>

<!-- The four sections below each follow: research question → methodology → highlighted
results → proposed methodology improvements. -->

---
layout: section
---

# RQ1 — SNR Definition

Which SNR variant best predicts decision accuracy?

---
layout: focus
color: blue
icon: ❓
---

## Of 22 candidate SNR definitions, which best correlates with decision accuracy across 12 languages — and does it hold across seeds?

---
layout: bullets
title: RQ1 — SNR Definition
subtitle: Methodology
icon: "⚙️"
---

- **22 SNR variants** × sizes × ~115 multilingual parent tasks
- Pools: pure custom 1 / 2 / 3 seeds, and **`custom_swissai_hf`** (3 seeds + external pretraining models, instruct excluded) — the recommended comprehensive tier, spanning **175M → 32B**
- **DA-size**: small-last-ckpt → 1B-last-ckpt rank agreement
- **DA-ckpt**: within-size early → late ckpt agreement (relative-fraction early ckpts let external trajectories enter)
- Per-language correlation: Pearson r between $\log_{10}(\text{SNR})$ and DA
- Generalization check: pick best variant on 2 seeds, evaluate on held-out seed 1904

---
layout: figure
image: /results/top_variants_overall.png
title: RQ1 — SNR Definition
subtitle: "Top variants across languages (DA-size & DA-ckpt)"
---

---
layout: bullets
title: RQ1 — SNR Definition
subtitle: Highlighted results
icon: "✅"
---

- **`rel_mpd` and the dispersion / relative-spread family win** — top variant on `custom_swissai_hf`: DA-size r **0.400**, DA-ckpt r **0.519**, overall **0.460**
- **Adding external models mainly lifts DA-ckpt** (0.379 → 0.519 at 3 seeds): the relative-fraction ckpt-DA now draws on external multi-checkpoint trajectories
- **Dose-response in seeds**: top DA-size r climbs 0.31 → 0.33 → 0.39 (1 → 2 → 3 seeds)
- `tukey`, `projection` (depth): **r ≈ 0** with DA → useless at this pool size
- **Variant ranking transfers to a held-out seed** — Spearman ρ **+0.80** (DA-size), **+0.93** (DA-ckpt); exact per-language argmax does not (family-level agreement 14% / 36%)

---
layout: bullets
title: RQ1 — SNR Definition
subtitle: Proposed methodology improvements
icon: "💡"
---

- **Collapse the dispersion cluster** to one representative before ranking (r ≥ 0.999 inflates stability)
- Use a **larger DA-size target** (e.g. Apertus-8B) instead of the not-fully-converged 1B custom model
- **Bootstrap CIs** on per-language Pearson r and cross-pool Spearman ρ
- Add a **third independent seed pool** to test generalization without re-using the 1904 split
- Recommend a **family** (dispersion / relative-spread), not an exact variant — only the family transfers

---
layout: section
---

# RQ2 — Framework Generalization

Does it transfer to AllenAI DataDecide?

---
layout: focus
color: blue
icon: ❓
---

## Do the SNR variants we recommend on Apertus also correlate with AllenAI DataDecide on the shared English benchmarks?

---
layout: bullets
title: RQ2 — Framework Generalization
subtitle: Methodology
icon: "⚙️"
---

- **Apertus side**: 22 SNR variants × sizes × 7 shared English tasks (`custom_swissai_hf` pool)
- **AllenAI side**: same 22 variants on the DataDecide ladder (25 mixes × 5 ckpts, 150M–1B)
- Task-name reconciliation: `global_mmlu_full_en[_<subj>] → mmlu[_<subj>]`
- **Headline axis**: $\log_{10}(\text{SNR}_{1B})$ on each corpus, Pearson r over the 7 shared tasks
- Top-K agreement: intersection / Jaccard at K ∈ {5, 10, 20}

---
layout: figure
image: /results/snr_apertus_vs_snr_allenai_rms_deviation.png
title: RQ2 — Framework Generalization
subtitle: "Apertus vs AllenAI SNR — best variant (rms_deviation)"
---

---
layout: bullets
title: RQ2 — Framework Generalization
subtitle: Highlighted results
icon: "✅"
---

- **Cross-corpus Pearson r = 0.84** on the comprehensive pool (`rms_deviation`; `dispersion`/`range` tie at 0.84) — climbs with seeds (0.75 single-seed → 0.94 pure 3-seed)
- Top-10 most-reliable English benchmarks are **identical** on both corpora (**Jaccard 1.0**)
- Reliable on **both**: `arc_challenge`, `arc_easy`, `csqa`, `hellaswag`, `mmlu`, `openbookqa`, `piqa`
- **Dispersion + discrepancy families transfer**; the relative-spread family (incl. AllenAI's own default `rel_std`) does **not**

---
layout: bullets
title: RQ2 — Framework Generalization
subtitle: Proposed methodology improvements
icon: "💡"
---

- **Enlarge the shared universe** from 7 tasks: add the 53 `mmlu_<subject>:mc` rows + `mmlu_pro`, BBH, AGI-Eval (→ up to ~185 tasks)
- Re-run **original `mmlu`** (not the `global_mmlu_full_en` alias) on Apertus → compare like-for-like content
- **Bootstrap CIs** on the cross-corpus r (n=7 is fragile)
- Report **Spearman ρ** alongside Pearson r (rank-based is more robust at small n)

---
layout: section
---

# RQ3 — Benchmark Creation

What makes a benchmark high-SNR?

---
layout: focus
color: blue
icon: ❓
---

## What benchmark design features (curation, format, option count, item length) predict SNR?

---
layout: bullets
title: RQ3 — Benchmark Creation
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
title: RQ3 — Benchmark Creation
subtitle: Per-family median SNR (custom_swissai_hf)
---

---
layout: bullets
title: RQ3 — Benchmark Creation
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
title: RQ3 — Benchmark Creation
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

# RQ4 — Subsampling

Subsets that beat the full benchmark

---
layout: focus
color: blue
icon: ❓
---

## Can a subset of subtasks (languages, subjects) or individual items give higher SNR than the full benchmark?

---
layout: bullets
title: RQ4 — Subsampling
subtitle: Methodology
icon: "⚙️"
---

- `custom_swissai_hf` pool (36 Apertus + external pretraining models), last-5 ckpts per model
- **Subtask level** — three cases: language subset, MMLU subject subset, subject × language
- **Per-sample level** — four proposers: `greedy_snr_rank` (A), `forward_greedy` (B), `irt_discrimination` (C), `variance_prefilter` (D, default)
- Sweep: rank by standalone SNR, add greedily, record cumulative SNR; best subset = argmax of the curve (random-order baseline alongside)
- Per-sample is **cluster-only** (needs `samples_*.jsonl`)

---
layout: figure
image: /results/global_mmlu_full_subjects.png
title: RQ4 — Subsampling
subtitle: "Subtask level — MMLU subject subset curves per size"
---

---
layout: bullets
title: RQ4 — Subsampling
subtitle: Highlighted results
icon: "✅"
---

- **Best subset usually beats the full set substantially** — Global-MMLU 175M `+1.52` SNR (`international_law` alone, full 2.12 → 3.65); per-language GMF-tr 1B `+1.56`; Belebele 350M `+1.16`
- **Subject subsets beat language subsets** — Case 2 (MMLU subject, mean over 10 langs) gives the most reliable gains
- **Stability is uneven**: MMLU **subject** picks recur across pools; **language** and **subject × language** picks often flip
- **Per-item (per-sample) ranking is mostly noise** (cross-size Spearman ≈ 0.05) → tiny argmax subsets overfit and collapse out-of-sample

---
layout: bullets
title: RQ4 — Subsampling
subtitle: Proposed methodology improvements
icon: "💡"
---

- **Relax the selection rule**: report the largest subset within 1-SE of the peak that still beats the full set (not the knife-edge argmax)
- **Denoise the ranking**: pool item SNR across seeds/sizes; or select at subtask/topic granularity (the unit that *is* partially stable)
- **Make trust measurable**: held-out seed-pool CV + stability selection → out-of-sample trust number
- **Change the objective**: optimize **decision accuracy** (saturates) rather than raw SNR (spikes)

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

- **Signal definition**: which variant / family, across which dimensions?
- **Noise definition**: checkpoint vs benchmark noise — do we extend the benchmark-noise analysis?
- **Sub-benchmark selection**: how to approach subset selection methodologically?
- **Custom models' architecture**: comments or improvements?
- **Language coverage**: how far down the resource ladder can we trust the signal?

---
layout: default
title: Open Questions
subtitle: Sub-benchmark selection
---

### How to approach sub-benchmark selection methodologically?

- SMART filtering (Gupta et al., 2024)
- How to Select Datapoints for Efficient Human Evaluation of NLG Models? (arXiv 2501.18251)

Related work:
- **Chen et al. (2024)** — scaling behavior of downstream tasks
- **Gupta et al. (2024)** — SMART filtering of benchmark items
- **Zhou et al. (2025)** — item response theory for benchmark reliability

---
layout: default
title: Open Questions
subtitle: Custom models — architecture
---

### Comments or improvements on the custom model training?

- **ATLAS**: Adaptive Transfer Scaling Laws for Multilingual Pretraining (arXiv 2510.22037)
  - For similar loss to a 1B model on 100B English tokens, over $n$ languages:
  - Model size $= 1\text{B} \times n^{0.243}$
  - Dataset size $= 100\text{B} \times n^{0.728}$
