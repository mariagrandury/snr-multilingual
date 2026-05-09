# Signal-to-Ratio experiments

Evaluate a set of models at multiple checkpoints to study how benchmark signal
emerges across training.

## How it works (one-screen overview)

Two steps, cleanly separated:

1. **Generate** a stage runner from a models file. A runner is a plain bash file in
   `runners/` that declares `MODEL_CHECKPOINTS` and sources
   `runners/hf_base_runner.sh` — identical idiom to the existing
   `runners/hf_eval_multiple_*.sh` files. Megatron keys embed the iter as
   `<base>-iter<N>`; the shared runner parses it directly. Commit the runner so
   the evaluated-checkpoint snapshot is version-controlled.
2. **Launch** with the standard `launch_evaluations.sh --script …` entry point.
   The new `snr-pretraining` / `snr-midtraining` / `snr-posttraining` modes set
   `TASKS` to the right `tasks_*.txt` (pretraining and midtraining share
   `tasks_pretraining.txt` — both are base-model evaluations) and switch the
   W&B defaults to `mariagrandury-epflnlp/snr-experiments`. `snr-posttraining`
   additionally defaults `APPLY_CHAT_TEMPLATE=true` because the task set
   (ifeval, humaneval_instruct, …) requires a chat template on Instruct/SFT
   models; pass `--no-chat-template` to override.

Scripts:

| Script                                                                 | Purpose                                                                                                               |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| [scripts/list_checkpoints.sh](../../scripts/list_checkpoints.sh)       | Enumerate Megatron iters (multiples of 2000) or HF non-main branches for one model. Pure selection, prints to stdout. |
| [scripts/generate_snr_runner.sh](../../scripts/generate_snr_runner.sh) | Read a models file, enumerate each, emit a stage runner to stdout.                                                    |
| [scripts/launch_evaluations.sh](../../scripts/launch_evaluations.sh)   | Unchanged entry point. Use `--script runners/snr_<stage>.sh`.                                                         |

## Generate runners

On the cluster (so Megatron paths resolve):

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals && git pull

# Pretraining (Megatron) - N evenly-spaced iters across each run
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_pretraining_custom.txt --total 10 \
    > runners/snr_pretraining.sh

# Midtraining (HuggingFace) - evenly-spaced branches
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_midtraining_hf.txt --total 5 \
    > runners/snr_midtraining.sh

# Posttraining (HuggingFace) - last few branches (most repos have only 0-8)
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_posttraining_hf.txt --last 3 \
    > runners/snr_posttraining.sh
```

Selection flags:

- `--last N` — last N checkpoints (iter order / branch name order)
- `--total T` — T evenly-spaced checkpoints across all available

Regenerate whenever new checkpoints appear or the models file changes, then
commit. The runner's comment header records the exact regenerate command.

Inspect what's available before generating:

```bash
# Megatron iters (% 2000 == 0) under a checkpoints dir
bash scripts/list_checkpoints.sh /iopsstor/.../checkpoints --last 20

# HF non-main branches for a repo
bash scripts/list_checkpoints.sh https://huggingface.co/allenai/Olmo-3-7B-Instruct --last 999

# High-level status for every model in a file
bash configs/signal_to_ratio/check_checkpoints.sh configs/signal_to_ratio/models_pretraining_custom.txt
```

## Launch evaluations

```bash
# Pretraining
bash scripts/launch_evaluations.sh snr-pretraining \
    --script runners/snr_pretraining.sh --splits 10 --time 04:00:00

# Midtraining
bash scripts/launch_evaluations.sh snr-midtraining \
    --script runners/snr_midtraining.sh --splits 5 --time 04:00:00

# Posttraining
bash scripts/launch_evaluations.sh snr-posttraining \
    --script runners/snr_posttraining.sh --splits 2 --time 04:00:00
```

Useful flags (all forwarded verbatim to the normal eval path — nothing SNR-specific):

- `--splits K` — split tasks across K parallel nodes per checkpoint (an aggregation job auto-chains with `afterok`).
- `--limit N` — cap samples per task (quick smoke tests).
- `--harness-branch B` — install `lm-evaluation-harness` from a specific branch.
- `--num-fewshot N` — override few-shot count.
- `--time HH:MM:SS` — override the Slurm `--time` limit on eval jobs (the aggregation job keeps its own short default).

### One command, idempotent across re-launches and collaborators

Launching the full pretraining sweep (12 custom models × 10 canonical
iters × ~85 tasks) is one command, and that **same command can be re-run
any number of times** — by you, by a colleague, after a partial timeout —
without redoing finished work:

```bash
bash scripts/launch_evaluations.sh snr-pretraining-full \
    --script runners/snr_pretraining_all.sh --time 04:00:00
```

`snr-pretraining-full` uses [tasks_pretraining_full.txt](tasks_pretraining_full.txt)
(the dedup union of `tasks_pretraining.txt` + `tasks_pretraining_b.txt`)
and `runners/snr_pretraining_all.sh` enumerates the same 10 iters
(2000, 6000, 12000, 18000, 22000, 28000, 34000, 38000, 44000, 50000)
across every `apertus-{175M,350M,600M,1B}-fwEdu{30,60,90}` checkpoint.

Two layers of idempotency keep re-launches and concurrent submissions
from doing duplicate work:

| Layer | Where | Behavior |
| ----- | ----- | -------- |
| Per-checkpoint | [`runners/hf_base_runner.sh`](../../runners/hf_base_runner.sh) | Before each `sbatch`, calls `scripts/_eval_status.py` and **skips submission entirely** if every task in `$TASKS` already has results on disk for that ckpt. |
| Per-task | [`scripts/_run_per_task.sh`](../../scripts/_run_per_task.sh) | Inside a running job, filters `$TASKS` down to remaining and exits cleanly with no work if everything is already done. Logs skipped tasks to `skipped_tasks.log`. |

A task counts as "done" if a non-empty `eval_*/per_task/<task>/` exists
(saved by killed-mid-run jobs) or if some `eval_*/results_*.json` lists
it under `.results` (saved by clean merged runs). Same logic in both
the launcher gate and the runner gate, so they always agree.

Race window: if two collaborators launch simultaneously, both see the
same set of "missing" ckpts and may submit duplicate sbatch jobs for
them. Once those jobs are running, the per-task layer kicks in — the
second job's `_run_per_task.sh` will skip whatever the first one
already produced. Stagger launches by a few minutes if you want to
tighten this further.

Sanity check before launching:

```bash
# Dashboard — per-ckpt progress bars + pending job IDs across all
# models_pretraining_*.txt files.
python3.11 scripts/snr_progress.py

# Restrict to one models file
python3.11 scripts/snr_progress.py --models configs/signal_to_ratio/models_pretraining_custom_all.txt

# Find gaps not yet submitted
python3.11 scripts/snr_progress.py --status not_submitted

# Per-task detail for one ckpt
python3.11 scripts/snr_progress.py --details \
    --filter apertus-350M-fwEdu30-fw270-seed1904-iter2000
```

Note: the system `python` on login nodes is 3.6; the dashboard needs
`python3.11`.

### Smoke tests

Tiny end-to-end runs through the existing `single` mode (one or
comma-separated `--task` values, `--limit 4` to cap samples per task). The
`WANDB_*` prefix routes results into the SNR W&B family (`single` mode
appends `-single` to the project name, so they land in
`snr-experiments-single` and stay separate from real sweep data).

`runners/snr_test.sh` (committed) covers the simplest HF case; for the
others, generate a runner from `models_test_hf.txt` /
`models_test_megatron.txt` first.

```bash
# 1) 1 HF checkpoint, 1 task — committed runner
bash scripts/launch_evaluations.sh single \
    --script runners/snr_test.sh --task hellaswag --limit 4 --time 00:15:00

# 2) 2 HF checkpoints of the same model, 2 tasks
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_test_hf.txt --last 2 \
    > runners/snr_test_hf_2.sh
bash scripts/launch_evaluations.sh single \
    --script runners/snr_test_hf_2.sh --task hellaswag,piqa --limit 4 --time 00:15:00

# 3) 1 Megatron checkpoint (last iter), 1 task
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_test_megatron.txt --last 1 \
    > runners/snr_test_meg_1.sh
bash scripts/launch_evaluations.sh single \
    --script runners/snr_test_meg_1.sh --task hellaswag --limit 4 --time 00:15:00

# 4) 2 Megatron checkpoints (last two iters), 2 tasks
bash scripts/generate_snr_runner.sh \
    --models configs/signal_to_ratio/models_test_megatron.txt --last 2 \
    > runners/snr_test_meg_2.sh
bash scripts/launch_evaluations.sh single \
    --script runners/snr_test_meg_2.sh --task hellaswag,piqa --limit 4 --time 00:15:00
```

`models_test_megatron.txt` already points at
`/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-350M-fwEdu60-fw240-seed1904/checkpoints/`,
so cases 3 and 4 just work. Sanity-check a generated runner once with
`bash -n runners/snr_test_meg_2.sh` before the first launch.

## Where outputs go

### Slurm stdout/stderr (per job)

`<repo>/logs/<job_name>_<job_id>.{out,err}`, written relative to the directory
`sbatch` was invoked from (see
[scripts/evaluate.sbatch:9](../../scripts/evaluate.sbatch#L9)).

Job name pattern:

- single-node: `eval-<model_name>`
- split (of K): `eval-<model_name>-split<i>` + `eval-<model_name>-aggregate`

Where `<model_name>` is:

- Megatron: `<base>-iter<N>` e.g. `apertus-175M-fwEdu60-fw240-seed28-iter10000`
- HuggingFace: `<base>-<branch>` e.g. `Apertus-8B-2509-step1750000-tokens7652B`

```bash
# Tail the latest log across all eval jobs
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/logs/ && ls -t eval-*.out | head -1 | xargs tail -100

# Tail a specific model
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/logs/ && tail -100 eval-SmolLM3-3B-checkpoints-stage3-step-4720000_1831506.out

# Errors only — latest .err across all eval jobs
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/logs/ && ls -t eval-*.err | head -1 | xargs tail -100
```

### Harness + W&B logs (per checkpoint)

`$LOGS_ROOT/$WANDB_ENTITY/$WANDB_PROJECT/<model_name>/harness/eval_<timestamp>_<jobid>/`
where defaults for SNR are:

- `LOGS_ROOT=/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs`
  (see [scripts/evaluate.sbatch:69](../../scripts/evaluate.sbatch#L69))
- `WANDB_ENTITY=mariagrandury-epflnlp`
- `WANDB_PROJECT=snr-experiments`

The harness writes one `results_<timestamp>.json` per run plus per-task
`samples_<task>_<timestamp>.jsonl` files alongside it.

```bash
# List the most recent checkpoint dirs
cd /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/snr-experiments/ && ls -t | head -20

# Latest eval dir for a specific checkpoint
cd /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/snr-experiments/<model_name>/harness/ && ls -t | head -5

# Cat the aggregated results JSON from the latest eval of a checkpoint
cd /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/snr-experiments/<model_name>/harness/ && cat "$(ls -td eval_*/ | head -1)"results_*.json

# Tail the per-task samples file (e.g. hellaswag) from the latest eval
cd /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/snr-experiments/<model_name>/harness/ && ls -t "$(ls -td eval_*/ | head -1)"samples_hellaswag_*.jsonl | head -1 | xargs tail -20
```

### Weights & Biases

https://wandb.ai/mariagrandury-epflnlp/snr-experiments

## Debug playbook

Check running SNR jobs:

```bash
squeue --me --noheader -o "%.10i %.30j %.2t %.10M %.6D %R" | grep eval-
```

Tail the latest failing job:

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/logs/ && ls -t eval-*.err | head -1 | xargs tail -60
```

Sanity-check a generated runner before submitting:

```bash
bash -n runners/snr_posttraining.sh
cat runners/snr_posttraining.sh
```

Cancel all SNR jobs:

```bash
# Preview
squeue --me --noheader -o "%i %j" | grep eval-
# Cancel
squeue --me --noheader -o "%i %j" | grep eval- | awk '{print $1}' | xargs scancel
```

Re-run a single checkpoint without regenerating the runner (use the main
launcher's single-model path):

```bash
# Megatron
bash scripts/launch_evaluations.sh snr-pretraining \
    --model /iopsstor/.../apertus-175M-.../checkpoints \
    --megatron-iter 10000 --splits 2 --time 04:00:00
```

### Rescue an eval that timed out mid-run

[scripts/_run_per_task.sh](../../scripts/_run_per_task.sh) writes one
sub-directory per finished task to
`$EVAL_DIR/per_task/<task_name>/` *as each task completes* — they live on
persistent storage, not scratch. If Slurm kills the job before the final
merge + W&B upload, only the in-progress task is lost; everything already
finished can be merged and pushed afterwards.

Locate the eval directory (timestamp + jobid):

```bash
RUN=mariagrandury-epflnlp/snr-experiments
NAME=apertus-350M-fwEdu60-fw240-seed1904-iter50000
EVAL_DIR=$(ls -td /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/$RUN/$NAME/harness/eval_*_<jobid> | head -1)
ls "$EVAL_DIR/per_task/"     # surviving per-task dirs
```

Merge the surviving dirs and re-upload to W&B (one-shot, runs anywhere
the repo is checked out — no Slurm needed):

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals

python -m scripts.alignment.merge_split_results \
    --split_dirs "$EVAL_DIR"/per_task/*/ \
    --output_dir "$EVAL_DIR"

python -m scripts.alignment.update_wandb_alignment \
    --entity mariagrandury-epflnlp \
    --project snr-experiments \
    --logs_root "$EVAL_DIR" \
    --name "$NAME" \
    --main_metrics "" --eval_duration 0
```

To finish the missing tasks too, re-launch the same checkpoint with
`launch_evaluations.sh` and a `TASKS=...` override that lists only the
missing ones (skip what's already in `$EVAL_DIR/per_task/`).
