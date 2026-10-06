---
title: The model ladder
---

# The model ladder

We release a controlled grid of small multilingual language models. Every run
differs from its neighbours in one thing only, so the effect of a design
decision can be read off directly — and compared across sizes.

- **Size.** 90M, 175M, 350M, 600M, 1B, 1.7B and 3B non-embedding parameters.
  Each size trains on 100 tokens per parameter (5× Chinchilla) on a
  warmup-stable-decay schedule, with 20 to 60 evenly spaced checkpoints.
- **Languages.** Half English (DCLM), half FineWeb-2 spread over L = 1, 2, 8,
  15, 30 or 50 languages. L = 1 is English only. The mixture changes the
  *number* of languages, never the English share.
- **Interventions.** Depth (deep vs shallow at equal size), the language list
  (A: ranked by data size, B: chosen for diversity), the sampling temperature
  (T = 1 vs T = 3) and the second language at L = 2 (Russian, Chinese or
  Spanish).
- **Seeds.** Replicates at selected cells measure seed noise, the yardstick
  every decision is compared with.

The 90M rung diverges (an optimizer time-scale mismatch) and 3B is the
extrapolation check, so the analysis uses 175M–1.7B with 1.7B as the
reference.

## Explore the grid

Each square is one run. Click it to see its data, its training budget, its
per-language results and where to download it.

<div class="viz" data-viz="ladder"></div>

## Per-language bits per byte

Bits per byte (BPB) on a fixed validation set in 100 languages is the
ladder's main outcome. Pick a language to see how it scales with model size
for each number of training languages.

<div class="viz" data-viz="bpb"></div>

## Get the models and the data

| What | Where |
|---|---|
| Checkpoints (Hugging Face format) | [`huggingface.co/msnr`](https://huggingface.co/msnr) — one repository per run, named like the run (`lm-<size>-L<L>[-scheme]-<arch>-seed<seed>`) |
| Per-checkpoint measurements | the ladder report, [`msnr-data/ladder-report`](https://huggingface.co/datasets/msnr-data/ladder-report) |
| Training logs | W&B project [`mariagrandury-epflnlp/msnr`](https://wandb.ai/mariagrandury-epflnlp/msnr) |
| How the runs were trained | [Pretraining docs](pretraining.md), [`plan/`](https://github.com/swiss-ai/snr-multilingual/tree/main/plan) |
