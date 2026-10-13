---
title: The model ladder
---

# The model ladder

We release a controlled grid of small multilingual language models. Each run
differs from a neighbour in one design choice only, so the effect of that
choice can be read off directly and compared across model sizes. This page
describes the data the models were trained on, the models themselves and how
they were evaluated.

## Data mixtures

Every run trains on a mixture of English and FineWeb-2 web text. A mixture is
defined by three choices:

- **Scheme M**: which languages enter. Scheme A adds languages in order of
  how much data FineWeb-2 has for them; scheme B picks them for diversity of
  family and script. At K = 2, the second language is Russian (A), Chinese (B)
  or Spanish (C); at K = 1, the English corpus is DCLM-Edu (A), DCLM without
  its educational filter (B) or FineWeb (C).
- **K**: how many languages, English included.
- **Temperature T**: how the non-English half is split. At T = 1 each
  language gets a share proportional to its available data, so a few
  languages dominate. T = 3 flattens the split (shares ∝ data<sup>1/3</sup>),
  giving the smaller languages more tokens.

Pick a mixture to see how its tokens are spread over languages. Colour shows
the language family and texture the writing script; hover a language for
its details.

<div class="viz" data-viz="mixtures"></div>


## Model ladder

The grid holds **218 runs**: 122 deep and 60 shallow models, 18 with the
SwiGLU activation and 18 trained with the Muon optimizer; 198 at seed 1904
and 20 seed replicates.

- **Size.** Seven rungs: 90M, 175M, 350M, 600M, 1B, 1.7B and 3B
  non-embedding parameters. Each run trains on 100 tokens per parameter
  (5× Chinchilla) with a warmup-stable-decay schedule and evenly spaced
  checkpoints.
- **Languages.** The number of training languages K is 1, 2, 8, 15, 30 or
  50. K = 1 is English only. From K = 2 on, half of the training tokens are
  English and the other half are split over the other K − 1 languages, so
  the mixture changes the *number* of languages, never the English share.
- **Interventions.** Each one changes a single choice with respect to the
  baseline (deep model, xIELU activation, AdEMAMix optimizer, language list
  A, temperature T = 1):
    - *Depth*: deep or shallow at the same non-embedding size, at every
      size from 90M to 1.7B and every L. The 600M shallow model is not
      exactly size-matched (3.7 % more non-embedding parameters, about
      15 % more compute); the analysis keeps and flags it.
    - *Activation*: xIELU or SwiGLU, on the deep model at K = 8, 15 and 30.
    - *Optimizer*: AdEMAMix or Muon, on the deep model at K = 8, 15 and 30.
    - *Language list*: A (languages ranked by available data) or B (chosen
      for diversity), at K = 8, 15 and 30, deep and shallow.
    - *Sampling temperature*: T = 1 or T = 3, at K = 15 and 30 (deep) and
      K = 50 (deep and shallow); see [Data mixtures](#data-mixtures).
    - *Second language*: Russian, Chinese or Spanish, at K = 2.
    - *English data*: at K = 1, DCLM with its educational-quality filter
      (the baseline), DCLM without that filter, or FineWeb.
- **Seeds.** Two more seeds at ten cells (175M and 600M at K = 1, 2 and 50;
  1B at K = 1, 2 and 30, and at K = 30 with list B) measure seed noise, the
  yardstick every design decision is compared with.
- **The 3B rung.** Six deep runs at seed 1904: K = 8 and 15 with lists A
  and B, and K = 30 and 50 with list A.

The analysis reads 90M–1.7B, with 1.7B as the reference model; 3B sits
above the reference and serves as the extrapolation check.

### Explore the grid

Each square is one run: model size on the horizontal axis, number of
training languages K on the vertical one. Colours and filters follow the
paper's training grid: a data build (A, AT3, B, ZH, ES, DCLMP, FWEB) sets the
language-selection scheme M and the sampling temperature T, and a ladder
(deep, shallow, deep + SwiGLU, deep + Muon) the model recipe. Click a square
to see its data, its training budget and its per-language results.

<div class="viz" data-viz="ladder"></div>

### Per-language bits per byte

Bits per byte (BPB) on a fixed validation set in 100 languages is the
ladder's main outcome. Pick a language to see how it scales with model size
for each number of training languages.

<div class="viz" data-viz="bpb"></div>

## Evaluation

Every run is scored on the benchmarks of the languages it trained on, at ten
checkpoints spread evenly over the run (plus a few more near the end). Most
benchmarks are multiple choice: the model sees a prompt, we measure how likely
it finds each answer option as the continuation, and the most likely option is
its answer. The score is the share of items answered correctly. Guessing at
random gives the **chance** level: one over the number of options.

We considered 95 benchmarks (1,488 benchmark-language tasks in the trained
languages). The analysis follows 52 of them (938 tasks) through training,
together with the per-language bits per byte. The other 43 (550 tasks) were screened on
the final checkpoints of the 600M, 1B and 1.7B models only, and 229 of their
tasks, at chance at both 1B and 1.7B, were discarded. Several benchmarks are
also scored in a reformulated version, with the answer text in place of an
answer letter (see [Measurability](findings/measurability.md)).

Pick a language, a benchmark and the x axis. Each line is one model size. A
useful benchmark separates the sizes and rises above chance early. A benchmark
whose lines stay on the chance line tells a small-scale experiment nothing.

<div class="viz" data-viz="scoreCurves"></div>

The card under the chart describes the selected benchmark: its number of answer
options, the languages it covers, where its data comes from, and one item in
the selected language and in English, exactly as the model reads it (long
passages are cut at the start).

## Get the models and the data

| What | Where |
|---|---|
| Checkpoints (Hugging Face format) | *{{ anonymity_notice }}* — one repository per run, named like the run (`lm-<size>-L<L>[-<data build>][-b<batch>]-<deep|shallow|swiglu|muon>-seed<seed>`); 213 models from 90M to 1.7B and 4,412 intermediate checkpoints |
| Per-checkpoint measurements | the ladder report (*{{ anonymity_notice }}*) |
| Training logs | W&B project (*{{ anonymity_notice }}*) |
| How the runs were trained | [Pretraining docs](pretraining.md), `plan/` (*{{ anonymity_notice }}*) |
