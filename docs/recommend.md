---
title: Find reliable benchmarks
---

# Find reliable benchmarks for your ablations

Three ways to use our results. **Start from our ladder** if you are about to
run ablations and want to know which benchmarks to trust. **Look up your
setup** (proxy size, mixture, token budget) to see which evaluations will be
above chance in each of your languages. **Upload your own results** if you
already have scores from several recipes and want to know which of your
benchmarks carry signal.

## 1. What our ladder says about your setup

Pick the language you care about, the size of the proxy models you can
afford, and the question you want the proxy to answer.

<div class="viz" data-viz="recommend"></div>

!!! tip "Rules of thumb from the ladder"
    1. **Drop what is at chance.** A benchmark below chance at your proxy size
       ranks recipes at random. Check first; [RQ0](findings/gate.md) lists them.
    2. **Prefer likelihood-based measurements.** Per-language bits per byte on a
       held-out set, and cloze-style benchmarks (HellaSwag, XStoryCloze,
       MultiBLiMP), scale predictably and decide early.
       Letter-format multiple choice does not, at these sizes.
    3. **Measure your seed noise once.** Train one recipe with two or three
       seeds. A decision whose effect is not clearly larger than that spread
       cannot be read at any proxy size ([RQ5](findings/design-decisions.md)).
    4. **Look at the full curve.** Early agreement can reverse later in
       training; compare several checkpoints before you stop a run
       ([RQ2](findings/decision-accuracy.md)).
    5. **Pick per language.** The most reliable benchmark changes from one
       language to the next; MultiBLiMP is the most reliable above-chance
       benchmark in 27 of 46 languages ([RQ4](findings/surrogates.md)).

## 2. Which evaluations can your proxy read? A look-up table

A benchmark only ranks recipes once the model scores above chance on it, and
in our ladder that depends mostly on **how many tokens of the language the
model has seen** ([RQ0](findings/gate.md)). So you can check your own setup
before you train:

1. **Proxy size** – the model size of your ablation runs.
2. **Token budget** – the tokens one proxy run trains on. By default this is our
   budget for that size (100 tokens per parameter, 5× Chinchilla).
3. **Mixture** – the share of each language in the training data. Start from one
   of our mixtures and edit the shares, or add and remove languages.

Share × budget gives the tokens of each language your proxy sees. For each
evaluation and language, the table gives the fewest tokens of that language
from which the evaluation *stays* above chance in our runs of the same size.
Your setup either reaches that number or it does not. A ✓ also says at what
point in your run the evaluation becomes usable.

<div class="viz" data-viz="lookup"></div>

??? note "How the minimum-tokens table is built"
    The table is a summary of the tokens-seen curves of RQ0: the share of runs above chance
    against the tokens of the language a checkpoint has seen. We read every
    evaluated tenth of every seed-1904 run of each size, so one point is one run
    configuration (number of languages L) at one checkpoint. A point is above
    chance when at least half of the benchmark's tasks in that language are
    above chance, using the [same rule](findings/gate.md) as the rest of the site.
    The threshold is the smallest token count from which **every** point with
    at least as many tokens is above chance. A benchmark that falls back to chance
    later in training therefore gets no threshold. Results outside the token
    range we trained are marked "?", not guessed. Mixtures and token budgets
    are the ones listed under [Model ladder](ladder.md).

## 3. Rank the benchmarks of your own suite

Export your scores as a CSV with one row per (recipe, size, checkpoint,
task). The analysis runs in your browser; the file is never uploaded.

<div class="viz" data-viz="upload"></div>

| column | required | meaning |
|---|---|---|
| `recipe` | yes | the design variant you are choosing between (a data mix, an architecture, …) |
| `task` | yes | the benchmark (one row per task; per-language tasks are separate tasks) |
| `score` | yes | the metric; higher is better unless `lower_is_better` is set |
| `size` | no | model size label (`150M`, `1B`); with two sizes you get decision accuracy small → large |
| `step` | no | training step; several late checkpoints per recipe give the noise estimate and the early → final decision accuracy |
| `chance` | no | chance level of the task (e.g. `0.25` for four options) |
| `lower_is_better` | no | `1` for losses and bits per byte |

The computation mirrors
`snr/metrics.py` (*{{ anonymity_notice }}*):
signal is `(max − min) / mean` of the recipes' final scores, noise is the
standard deviation over each recipe's last checkpoints divided by their mean,
and decision accuracy is the share of recipe pairs ordered the same way by the
proxy and by the target (ties count as agreement only when both sides tie).
For the full 22 SNR definitions, use the [analysis pipeline](signal-noise/index.md).
