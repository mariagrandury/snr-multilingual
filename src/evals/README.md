# SNR evaluation pipeline

> Evaluate language models at many training checkpoints to feed the SNR
> framework. Built on top of
> [`lm-evaluation-harness`](https://github.com/swiss-ai/lm-evaluation-harness)
> with W&B integration; runs on the CSCS Alps SLURM cluster.

## What this produces

A single source of truth for every (model, checkpoint, task, metric)
score in the project, in three forms:

| Output | Where | Used by |
|---|---|---|
| Per-task JSON results + samples | `eval_logs/<entity>/<project>/<model>/harness/eval_<ts>_<jobid>/` | `build_hf_dataset.py` → HF dataset |
| HF dataset (`multilingual-snr/multilingual-snr-eval-results`) | three parquet splits: `pretraining_custom`, `pretraining_a06`, `reference_hf` | the SNR framework in [`../signal-and-noise/`](../signal-and-noise/) |
| W&B per-model curves | [`mariagrandury-epflnlp/snr-experiments`](https://wandb.ai/mariagrandury-epflnlp/snr-experiments) | live dashboards, one line per benchmark per model |

The HF dataset is the canonical input to every analysis in
[`../signal-and-noise/`](../signal-and-noise/): the SNR-variant CSV, the
benchmark_creation per-family ranking, the AllenAI cross-corpus
comparison, and the subset-search outputs all start from it.

## Models in scope

**36 Apertus pretrains** (the canonical 4 sizes × 3 mixes × 3 seeds from
[`../pretrain/`](../pretrain/)), plus reference HF models (Qwen3, Gemma-3,
SmolLM3, Olmo-3, Apertus-8B/70B) and the a06 main runs (`apertus3-{1b,3b}-*-nodes`).

The full list lives in [`configs/models.json`](configs/models.json) (the
shared source of truth, read via
[`scripts/utils/configs.py`](scripts/utils/configs.py)). Pools group them
for downstream SNR analysis — see
[`../signal-and-noise/README.md`](../signal-and-noise/README.md).

## Tasks in scope

86 tasks per checkpoint — the deduplicated union of
[`configs/signal_to_ratio/tasks_pretraining.txt`](configs/signal_to_ratio/tasks_pretraining.txt)
and `tasks_pretraining_b.txt`, exposed as the launcher mode
`snr-pretraining-full`. Coverage includes per-language benchmarks
(`multiblimp_<lang>`, `xstorycloze_<lang>`, `xwinograd_<lang>`,
`hellaswag_<lang>`, `xnli_<lang>`, `xcopa_<lang>`, `paws_<lang>`,
`belebele_<lang>`, `global_mmlu_full_<lang>_<subject>`, …) for 12+
languages and standalone English benchmarks (`arc_challenge`, `arc_easy`,
`hellaswag`, `piqa`, `openbookqa`, `mmlu`, `commonsense_qa`, …).

Task lists live under [`configs/signal_to_ratio/`](configs/signal_to_ratio/):

| File | Purpose |
|---|---|
| `tasks_pretraining.txt` | 48 tasks — pretraining-stage subset A |
| `tasks_pretraining_b.txt` | 38 tasks — pretraining-stage subset B |
| `tasks_pretraining_full.txt` | 86 tasks — dedup union, used by `snr-pretraining-full` |
| `tasks_posttraining.txt` | post-training tasks (instruct/SFT) |
| `*_main_table.txt` | matching `task/metric` pairs for the W&B summary table |

## How to run

The two-step flow: **generate a runner** from a models file (lists which
checkpoints to evaluate), then **launch** evaluations using the standard
launcher script. See
[`configs/signal_to_ratio/README.md`](configs/signal_to_ratio/README.md)
for the canonical SNR-experiments instructions.

The one-liner that drives the full sweep (idempotent — re-runnable):

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals && git pull && \
  bash scripts/launch_evaluations.sh snr-pretraining-full \
      --script runners/snr_pretraining_all.sh --time 12:00:00
```

Re-running is safe at two layers:

| Layer | Where | When |
|---|---|---|
| Per-checkpoint | [`runners/hf_base_runner.sh`](runners/hf_base_runner.sh) (calls `scripts/_eval_status.py`) | Before each `sbatch` — skips submission entirely if every task already has results for that ckpt |
| Per-task | [`scripts/_run_per_task.sh`](scripts/_run_per_task.sh) (calls `scripts/_eval_status.py`) | Inside a running job — filters `$TASKS` down to remaining; logs skipped to `skipped_tasks.log`; exits cleanly with no work |

Both use the same disk-scan in
[`scripts/_eval_status.py`](scripts/_eval_status.py): a task is "done" iff
a non-empty `eval_*/per_task/<task>/` exists (killed runs) **or** any
`eval_*/results_*.json` lists it under `.results` (clean runs).

Live progress dashboard:

```bash
python3.11 scripts/snr_progress.py                                    # per-ckpt summary
python3.11 scripts/snr_progress.py --status not_submitted             # gaps
python3.11 scripts/snr_progress.py --details --filter <NAME-substr>   # per-task
```

System Python on login nodes is 3.6; use `python3.11` for the dashboard.

When `normal` is backed up, [`scripts/debug_drain.sh`](scripts/debug_drain.sh)
feeds already-pending convert/eval jobs through the idle `debug` partition
(capped at debug-qos' 1 running + 1 queued, each capped to debug's 1:30 wall).
It never submits anything new, so it cannot duplicate work; conversions go
first, since a cell cannot be evaluated before its HF snapshot exists.

```bash
bash scripts/debug_drain.sh --dry-run   # what it would move
bash scripts/debug_drain.sh             # loop until nothing is pending
```

## Bits-per-byte: the second way to evaluate a model

Benchmarks are not the study's outcome metric. The predictivity plan's outcome
is **per-language bits-per-byte on the fixed held-out validation set**, and it
is measured by a different path from everything above — no lm-eval, no vLLM,
no benchmark tasks:

```bash
sbatch --job-name=bpb-<cell> scripts/score_bpb.sbatch <cell> [iters...]
```

One job per cell scores every converted checkpoint it finds and writes
`<LOGS_ROOT>/<entity>/msnr/<cell>-iter<N>/bpb/bpb.json` — per language, the
NLL, the byte count, `bpb` and `ppl`.

How it differs from the harness path, and why:

* **The data is the validation build, not a benchmark.** `build_data_mixtures.py
  --stage validation` wrote one `.bin` per language plus
  `validation.manifest.json`, and every training build skipped exactly those
  rows, so train and validation are disjoint by construction.
* **Nothing is generated.** It is a single teacher-forced forward pass per
  block, summing `-log p` over targets. Blocks overlap by one token so every
  token except the corpus's first is scored with a predecessor.
* **The denominator is measured, not assumed.** BPB divides by UTF-8 bytes, and
  the bytes are obtained by decoding the scored tokens — verified to reproduce
  the manifest byte count exactly on Latin, Cyrillic and CJK. Scaling the
  manifest total by a token fraction would be wrong for a prefix, because
  bytes-per-token varies per document.
* **Documents are concatenated without EOD**, so the numerator and denominator
  describe the same text. An inserted EOD would add likelihood cost that no
  byte in the denominator pays for.
* **It is comparable across tokenizers**, which accuracy is not — the
  denominator is bytes. That is why the plan chose it for the tokenizer
  intervention.

`--max-tokens` (default 1M/language) takes a deterministic leading-document
prefix, so every model is scored on byte-identical text; `--max-tokens 0` uses
the full ~5M. The offline flags in the sbatch are not optional: without
`HF_HUB_OFFLINE=1`, `from_pretrained` on a **local** path still calls the Hub
and blocks for ~25 min per checkpoint on a compute node.

Read the results with
[`../pretrain/sweep_health.py`](../pretrain/sweep_health.py), which also
cross-checks them against the loss curves and the benchmark scores.

## Outputs in detail

### SLURM logs (per job)

`<repo>/logs/<job_name>_<job_id>.{out,err}`. Job name pattern:

- single-node: `eval-<model_name>`
- split-K: `eval-<model_name>-split<i>` + `eval-<model_name>-aggregate`

Where `<model_name>` is `<base>-iter<N>` (Megatron) or `<base>-<branch>`
(HF).

### Harness + W&B logs (per checkpoint)

`$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/<model_name>/harness/eval_<timestamp>_<jobid>/`,
with SNR defaults:

- `LOGS_ROOT=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs`
- `WANDB_ENTITY=mariagrandury-epflnlp`
- `WANDB_PROJECT=snr-experiments`

Each `eval_*/` contains one `results_<timestamp>.json` per task plus
per-task `samples_<task>_<timestamp>.jsonl` files. The per-task
sub-dirs `eval_*/per_task/<task>/` are written *as each task completes*,
so walltime kills lose only the in-progress task.

### W&B per-model curves

[`mariagrandury-epflnlp/snr-experiments`](https://wandb.ai/mariagrandury-epflnlp/snr-experiments)
— one W&B run per *model* (not per ckpt), with each ckpt logged as one
history step. Pushed by
[`scripts/push_all_results.py`](scripts/push_all_results.py) from inside
the eval job and also runnable on the login node for bulk rescue. FLOPs
is the default x-axis (`define_metric(*, step_metric="flops")`); iter and
tokens are also defined and can be swapped in the W&B UI.

### HF dataset build

`build_hf_dataset.py` walks `eval_logs/.../snr-experiments/` and emits the
three parquet splits (`pretraining_custom`, `pretraining_a06`,
`reference_hf`) consumed by the SNR analysis pipeline. The build resolves
model metadata (size, params, tokens, family, split) from
`configs/models.json` via the shared `configs.py` loader and computes
`tokens` (Megatron iter × 504 × 4096 or HF branch value) and
`compute ≈ 6 × params × tokens` at build time.

## Repository structure (used paths only)

```
evals/
├── configs/
│   ├── models.json                      # shared model registry
│   ├── tasks.json                       # task → stage mapping (where applicable)
│   └── signal_to_ratio/                 # SNR experiment configs
│       ├── README.md                    # canonical SNR-experiments how-to
│       ├── models_pretraining_custom*.txt
│       ├── models_midtraining_hf.txt
│       ├── models_posttraining_hf.txt
│       ├── models_test_{hf,megatron}.txt
│       ├── tasks_pretraining{,_b,_full}.txt
│       └── tasks_posttraining.txt
├── scripts/
│   ├── launch_evaluations.sh            # primary entry point
│   ├── evaluate.sbatch                  # SLURM job script
│   ├── aggregate_splits.sbatch          # split-aggregation job
│   ├── generate_snr_runner.sh           # runner generator
│   ├── list_checkpoints.sh              # ckpt enumerator
│   ├── score_bpb.py                     # per-language BPB + perplexity
│   ├── score_bpb.sbatch                 # BPB job, one per cell
│   ├── mirror_eval_logs.sbatch          # rsync eval_logs -> capstor master
│   ├── snr_progress.py                  # progress dashboard
│   ├── _eval_status.py                  # idempotency disk scan
│   ├── _run_per_task.sh                 # inner per-task loop
│   ├── push_all_results.py              # W&B per-model push
│   ├── build_hf_dataset.py              # HF dataset builder
│   └── utils/configs.py                 # shared config loader
├── runners/
│   ├── hf_base_runner.sh                # submission loop, idempotency gate
│   ├── snr_pretraining_all.sh           # 36 models × 13 canonical iters = 468 cells
│   ├── snr_pretraining_local_hf.sh      # vLLM on converted Megatron ckpts
│   ├── snr_pretraining_hf_top.sh        # SmolLM3-3B / Olmo-3-7B / Apertus-8B (HF)
│   └── snr_pretraining_hf_70b.sh        # Apertus-70B (HF)
├── containers/
│   ├── env.toml                         # standard HF eval container
│   └── env_vllm.toml                    # vLLM container (recommended)
└── logs/                                # SLURM stdout/stderr
```

## Backends

| Backend | When | Notes |
|---|---|---|
| `vllm` | **Recommended.** Faster + dramatically more memory-efficient than the Megatron eval path. Used by `runners/snr_pretraining_local_hf.sh` (vLLM on converted Megatron ckpts) and the HF runners. | Generation tasks (gsm8k, squadv2) may differ slightly between backends — only compare results across models using the same backend. |
| `hf` (accelerate) | Default in `evaluate.sbatch`. | Slower; used when vLLM doesn't fit a particular task. |
| `megatron_lm` | Direct evaluation of Megatron ckpts (no HF conversion). | Has known memory pressure issues; prefer the local-HF (vLLM) runner instead. |

## Secrets

`evaluate.sbatch` reads env first, files in `scripts/` as fallback:

- `WANDB_API_KEY` — for the W&B push. Needs membership in
  `mariagrandury-epflnlp` entity.
- `HF_TOKEN` — for HF Hub fetches (reference HF models).
- `CSCS_SERVING_API` — for LLM-as-judge evals (e.g. AlpacaEval). Key at
  https://serving.swissai.cscs.ch.

`HF_HOME` and `HF_HUB_CACHE` are forwarded into the container by
`evaluate.sbatch` via `INNER_EXPORTS`; the populated cache lives at
`/capstor/store/cscs/swissai/infra01/users/$USER/hf_models` (~258 GB,
mounted by both `containers/env.toml` and `containers/env_vllm.toml`).

## See also

- [`configs/signal_to_ratio/README.md`](configs/signal_to_ratio/README.md)
  — canonical SNR-experiments how-to: generate runners, launch, smoke
  tests, rescue procedures.
- [`CLAUDE.md`](CLAUDE.md) — back-of-house notes: bug history (vLLM
  tokenizer revisions, Megatron container choice, idempotency edge cases,
  HF Hub rate limits, …), cluster gotchas, the W&B layout.
