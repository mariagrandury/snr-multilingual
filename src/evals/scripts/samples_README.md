<!--
Pre-release README for the planned `multilingual-snr/samples` hub dataset
(public; the per-instance jsonl files are too large for the team-plan
private quota). **Not yet uploaded** — stays here until the paper
releases, at which point both this file and samples_loader.py get
pushed to the repo root in their own commit. See push_samples.py.
-->
---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
  - multilingual
tags:
  - evaluation
  - per-instance
  - lm-evaluation-harness
  - swiss-ai
  - signal-to-noise
size_categories:
  - 100K<n<1M
---

# SwissAI Evals — Per-instance Samples

Per-instance lm-evaluation-harness outputs (one record per evaluation
example) that back the aggregate scores in
[`multilingual-snr/multilingual-snr-eval-results`](https://huggingface.co/datasets/multilingual-snr/multilingual-snr-eval-results).
Use this dataset when you need to inspect what the model actually
generated for each item, not just the headline metric.

## Layout

```
samples/<NAME>/<task>/<eval_TS>.jsonl
```

- `NAME` — eval-side checkpoint identifier (`<model>-iter<N>` for
  megatron, `<model>-<branch>` for HF / `<model>` for HF `main`).
- `task` — lm-eval task name (e.g. `piqa`, `belebele_eng_Latn`).
- `eval_TS` — wall-clock timestamp of the eval run, format
  `YYYY-MM-DDTHH-MM-SS.NNNNNN`.

## Dropped fields

Two fields are removed from each jsonl record at upload time to keep
the dataset compact (608 GB raw -> 127 GB on hub):

- `doc` — the original example (question / context / choices).
  Re-fetch via `lm_eval`'s task loader: `TaskManager().load_task_or_group(task)`
  then index `task.test_docs()` by `doc_id`.
- `arguments` — the rendered prompt sent to the model (largest field;
  is `doc` inlined into a prompt template). Re-derivable the same way.

Kept: `doc_id`, `target`, `resps`, `filtered_resps`, `filter`,
`metrics`, `acc`, `acc_norm`, `acc_bytes`, `exact_match`, `doc_hash`,
`prompt_hash`, `target_hash`.

## Loader

```python
from samples_loader import load_samples, list_tasks, list_eval_timestamps

for r in load_samples("apertus-175M-fwEdu30-fw270-seed1904-iter50000", "piqa"):
    print(r["doc_id"], r["filtered_resps"], r["acc"])
```

See `samples_loader.py` in the repo root.

## Update policy

Pushes are **additive only** — once a `(NAME, task, eval_TS)` file is
published, it is never overwritten. New evals append; never replace.
