---
title: Multilingual benchmarks
hide:
  - navigation
  - toc
---

# Multilingual benchmarks

Every multilingual (or non-English) benchmark we found implemented in
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
or [lighteval](https://github.com/huggingface/lighteval), one row per
benchmark. Filter by the language you care about, the framework you run, the
format or how the data was made; hover the language count for the full list
and the framework for the task names to run.

<div class="viz" data-viz="benchmarks"></div>

**Before you pick one for small-model ablations**, check whether it carries
signal at your scale: most four-option benchmarks sit at chance below 1B
([RQ0](findings/gate.md)), and the [recommender](recommend.md) ranks the
benchmarks of our suite per language and proxy size.

## Columns

| column | meaning |
|---|---|
| Languages | ISO 639-1 codes (639-3 where no 639-1 exists) |
| Framework | `harness`, `lighteval` or both; hover for the task or group names |
| Format | `mcqa`: scored by comparing the answer options; `generative`: free generation, exact match, or next-word / perplexity scoring |
| Options | number of answer options (MCQA only) |
| Items | evaluation items summed over languages, on the split the framework evaluates |
| Data source | how the non-English items were produced: `crawled` (collected from existing text, e.g. exams), `manual annotation` (written or translated by humans), `machine translation reviewed` (MT checked or post-edited by humans), `machine translation`, `synthetic` (templates or model-generated) |

The table is read from
[`configs/multilingual_benchmarks.csv`](https://github.com/swiss-ai/snr-multilingual/blob/main/configs/multilingual_benchmarks.csv).
To add or correct a benchmark, edit that file (same columns) and open a pull
request. Counts were collected from the framework code, the Hugging Face
dataset cards and the papers; they can lag behind both frameworks.
