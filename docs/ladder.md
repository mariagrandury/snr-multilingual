---
title: Model ladder
---

# Model ladder

We release a controlled grid of 218 small multilingual language models. This page
describes the data the models were trained on, the models themselves and how
they were evaluated.

| Symbol | Meaning | Values |
|---|---|---|
| $N$ | Non-embedding parameter count | 90M, 175M, 350M, 600M, 1B, 1.7B, 3B |
| $D$ | Depth | Deep or shallow |
| $O$ | Optimizer | AdEMAMix or Muon |
| $A$ | Activation function | xIELU or SwiGLU |
| $K$ | Number of training languages | 1 (EN), 2, 8, 15, 30, 50 |
| $M$ | Language-selection scheme | A (by availability), B (by diversity), C |
| $T$ | Sampling temperature | 1 (proportional) or 3 (flattened) |
| $S$ | Random seed | 1904 (default), 313, 64 |


## Data mixtures

Every run trains on a mixture of English and FineWeb-2 web text.
Every run trains on $5 \times 20N$ tokens.

??? note "How each mixture is defined"
    A mixture is defined by three choices:

    - **Scheme M**: which languages enter. Scheme A adds languages in order of
      how much data FineWeb-2 has for them; scheme B picks them for diversity of
      family and script. At K = 2, the second language is Russian (A), Chinese (B)
      or Spanish (C); at K = 1, the English corpus is DCLM-Edu (A), DCLM without
      its educational filter (B) or FineWeb (C).

    - **Language count K**: how many languages, English included.

    - **Temperature T**: how the non-English half is split. At T = 1 each
      language gets a share proportional to its available data, so a few
      languages dominate. T = 3 flattens the split (shares ∝ data<sup>1/3</sup>),
      giving the smaller languages more tokens.

Pick a mixture to see how its tokens are spread over languages. Colour shows
the language family and texture the writing script, hover a language for
its details.

<div class="viz" data-viz="mixtures"></div>

## Model ladder

The grid consists of **218 models**.
Each square in the figure below is one model: model size $N$ on the horizontal axis, number of training languages $K$ on the vertical one. Click a square
to see its data, its training budget and its per-language results.

<div class="viz" data-viz="ladder"></div>

<!--
### Per-language bits per byte
Bits per byte (BPB) on a fixed validation set in 100 languages is the ladder's main outcome. Pick a language to see how it scales with model size for each number of training languages.
<div class="viz" data-viz="bpb"></div>
-->

## Evaluation

Every run is scored on the benchmarks of the languages it trained on, at ten
checkpoints spread evenly over the run, plus the ones at 85% and 95% of training. 

!!! note "How the benchmark counts relate"
    - **95 benchmarks, 1,488 benchmark–language tasks**: every benchmark we
      considered, on the 50 trained languages (Appendix C.2).
    - **52 benchmarks, 938 tasks**: the benchmarks evaluated throughout
      training and analysed (§3.2, §4), each reformulated version (RF,
      LLM-RF) of a letter-format benchmark counted as its own benchmark and
      task. A model is evaluated on the tasks of the languages it trains: 113
      tasks at K = 1, 134 at K = 2, 326 at K = 8, 485 at K = 15, 720 at
      K = 30 and 938 at K = 50 (Appendix C.2).
    - **20 multilingual benchmark families** (abstract): the analysed
      benchmarks with more than one trained language, reformulated versions
      not counted separately.
    - **Table 6** lists each benchmark once, its reformulated versions folded
      into the row of the original, and counts each set of items once: 40
      analysed benchmarks (695 tasks) and 26 screened ones (485 tasks).

Pick a language, a benchmark and the x axis. Each line is one model size. The card under the chart describes the selected benchmark.
<!-- its number of answer options, the languages it covers, where its data comes from, and one item in the selected language and in English, exactly as the model reads it (long passages are cut at the start). -->

<div class="viz" data-viz="scoreCurves"></div>


## Access the models and data

We release the ladder and the per-checkpoint measurements to support cheaper, better-informed multilingual model development.

| What | Where |
|---|---|
| Checkpoints (Hugging Face model format) | One repository per run, named like the run (`lm-<size>-L<K>[-<data build>][-b<batch>]-<deep|shallow|swiglu|muon>-seed<seed>`), 218 models from 90M to 3B and 4,412 intermediate checkpoints. *{{ anonymity_notice }}*. |
| Per-checkpoint measurements (Hugging Face dataset format) | Ladder evaluation report. *{{ anonymity_notice }}.* |
| How the models were trained (Github repository) | Pretraining docs. *{{ anonymity_notice }}.* |
