---
title: The model ladder
---

# The model ladder

We release a controlled grid of small multilingual language models. Each run
differs from a neighbour in one design choice only, so the effect of that
choice can be read off directly and compared across model sizes. This page
describes the models, the data they were trained on and how they were
evaluated.

## Model ladder

- **Size.** Seven rungs: 90M, 175M, 350M, 600M, 1B, 1.7B and 3B
  non-embedding parameters. Each run trains on 100 tokens per parameter
  (5× Chinchilla) with a warmup-stable-decay schedule and evenly spaced
  checkpoints.
- **Languages.** The number of training languages L is 1, 2, 8, 15, 30 or
  50. L = 1 is English only. From L = 2 on, half of the training tokens are
  English and the other half are split over the other L − 1 languages, so
  the mixture changes the *number* of languages, never the English share.
  The 3B rung trains at L = 8, 15, 30 and 50 only.
- **Interventions.** Each one changes a single choice with respect to the
  baseline (deep model, xIELU activation, language list A, temperature
  T = 1):
    - *Depth*: deep or shallow at the same size.
    - *Activation*: xIELU or SwiGLU, at L = 8, 15 and 30.
    - *Language list*: A (languages ranked by available data) or B (chosen
      for diversity), at L = 8, 15 and 30.
    - *Sampling temperature*: T = 1 or T = 3, at L = 15, 30 and 50 (see
      [Data mixtures](#data-mixtures)).
    - *Second language*: Russian, Chinese or Spanish, at L = 2.
    - *English data*: DCLM with or without its educational-quality filter, or
      FineWeb, at L = 1.
- **Seeds.** Three seeds at selected cells (175M and 600M at L = 1, 2 and 50;
  1B at L = 1, 2 and 30) measure seed noise, the yardstick every design
  decision is compared with.

The analysis reads 90M–1.7B, with 1.7B as the reference model; 3B sits
above the reference and serves as the extrapolation check.

### Explore the grid

Each square is one run: model size on the horizontal axis, number of
training languages on the vertical one. Use the dropdowns to keep one level
of an intervention, and click a square to see its data, its training budget
and its per-language results.

<div class="viz" data-viz="ladder"></div>

### Per-language bits per byte

Bits per byte (BPB) on a fixed validation set in 100 languages is the
ladder's main outcome. Pick a language to see how it scales with model size
for each number of training languages.

<div class="viz" data-viz="bpb"></div>

## Data mixtures

Every run trains on a mixture of English and FineWeb-2 web text. A mixture is
defined by three choices:

- **Scheme**: which languages enter. List A adds languages in order of how
  much data FineWeb-2 has for them; list B picks them for diversity of
  family and script. At L = 2, the second language is Russian in list A, or
  Chinese or Spanish instead.
- **L**: how many languages, English included.
- **Temperature T**: how the non-English half is split. At T = 1 each
  language gets a share proportional to its available data, so a few
  languages dominate. T = 3 flattens the split (shares ∝ data<sup>1/3</sup>),
  giving the smaller languages more tokens.

Pick a mixture to see how its tokens are spread over languages. Colour shows
the language family and texture the writing script; hover a language for
its details.

<div class="viz" data-viz="mixtures"></div>

## Evaluation

Every run is scored on the benchmarks of the languages it trained on, at ten
checkpoints spread evenly over the run (plus a few more near the end). Most
benchmarks are multiple choice: the model sees a prompt, we measure how likely
it finds each answer option as the continuation, and the most likely option is
its answer. The score is the share of items answered correctly. Guessing at
random gives the **chance** level: one over the number of options.

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
| Checkpoints (Hugging Face format) | *{{ anonymity_notice }}* — one repository per run, named like the run (`lm-<size>-L<L>[-scheme]-<arch>-seed<seed>`) |
| Per-checkpoint measurements | the ladder report (*{{ anonymity_notice }}*) |
| Training logs | W&B project (*{{ anonymity_notice }}*) |
| How the runs were trained | [Pretraining docs](pretraining.md), `plan/` (*{{ anonymity_notice }}*) |
