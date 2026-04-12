---
theme: scholarly
layout: cover
transition: slide-left
footerMiddle: Signal-Aware Multilingual Evaluation
description: One-slide pitch — Signal-Aware Framework for Multilingual LM Evaluation
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
---

# Signal-Aware Framework for Multilingual LM Eval

_María Grandury, Angelika Romanou, Eléonore Hasler, Clara Meister, Antoine Bosselut_

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 24px; font-size: 0.85em;">

<div>

### 🔴 Problem

Training multilingual LMs is very expensive and requires many experiments at early stages of training. Practitioners evaluate on large suites, but many benchmarks are **noisy**, **redundant**, or **uncorrelated** with downstream goals.

### 🎯 Research Question

> Which (subsets of) benchmarks provide **early reliable signal** at each stage of mLM training?

### 📐 Method: Signal-to-Noise Ratio

<!-- $$\text{SNR}(b) = \frac{\text{Var}_{\text{models}}[\bar{s}_b]}{\mathbb{E}_m[\text{Var}_{\text{runs}}[s_b]]}$$

$$\text{SNR}(b) = \frac{\text{Signal (model separation)}}{\text{Noise (run variance)}} \quad \rightarrow \quad \text{High SNR = trustworthy benchmark}$$ -->

$$\text{SNR} = \frac{\text{Signal}}{\text{Noise}} = \frac{\text{Ability to separate models}}{\text{Sensitivity to random var between training steps}}$$

High SNR → better reliability and improved scaling law error

<!-- **High SNR** → model separation / run-to-run variance benchmark → reliably distinguishes models → better decisions -->

<p font-size: 0.85em>
Heineman, D. et al., Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation, 2025
</p>

</div>

<div>

### 🧪 Experimental Setup

|                   |                                                     |
| ----------------- | --------------------------------------------------- |
| 🤖 **Models**     | Custom (175M–1B) + Apertus (1B-70B) + SmolLM + OLMo |
| 🤖 **PT Models**  | 4 sizes x 3 data mixtures x 3 seeds                 |
| 🌍 **PT Data**    | DCLM (EN) + FineWeb2 (200+ languages)               |
| 📊 **Benchmarks** | 40 multilingual covering pre/mid/post-training      |
| 📏 **Metrics**    | SNR, Decision Accuracy                              |

### 🎁 Contributions

- 📄 **Paper:** SNR scores + stage-specific recommendations
- 🗂️ **INCLUDE:** optimal country subsets per training stage
- 🛠️ **OS toolkit:** compute SNR on your own benchmarks

</div>

</div>
