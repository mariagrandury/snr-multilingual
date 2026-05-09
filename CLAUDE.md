# Context for Claude — SNR (signal-to-ratio) experiments

This repo evaluates language models at many training checkpoints to study how
benchmark signal emerges over the course of pretraining. Most of the action is
under [configs/signal_to_ratio/](configs/signal_to_ratio/) and [runners/](runners/).

If you're picking this up cold, **read [configs/signal_to_ratio/README.md](configs/signal_to_ratio/README.md) first**.
This file is the back-of-house Claude memo: the gotchas and the why. The README is
the user-facing how.

---

## What's actually running

Two parallel tracks at any given time:

1. **Eval submissions on the cluster** (this repo). Slurm jobs invoke
   [`scripts/evaluate.sbatch`](scripts/evaluate.sbatch) which calls
   [`scripts/_run_per_task.sh`](scripts/_run_per_task.sh) inside an enroot/pyxis
   container, runs `lm_eval` once per task, merges results, and pushes to W&B
   via [`scripts/push_all_results.py --name $NAME --eval-duration $DURATION`](scripts/push_all_results.py)
   (per-model curve, one step per ckpt — see "W&B layout" below).

2. **Pretraining of half-finished custom models** (separate repo at
   `/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain`).
   The full sweep now spans **3 seeds** (28, 1797, 1904) — 36 models total.
   Resume jobs are managed via
   [`/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/launch_resumes.sh`](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/launch_resumes.sh),
   which is idempotent (skips cells that already have a queued/running job by
   matching the canonical Slurm name). Live status comes from
   [`scripts/pretrain_progress.py`](scripts/pretrain_progress.py) — see
   "Pretraining infrastructure" below.

Status snapshot anytime:

```bash
python3.11 scripts/snr_progress.py        # per-ckpt progress + pending job IDs
squeue --me -o "%.10i %.40j %.2t %.10M %.10L %.10P"
sacct -u $USER -S <since-date> -X -o "JobID,JobName%50,State,Elapsed"
```

---

## Models in scope

### Custom megatron checkpoints (36 models, file [models_pretraining_custom_all.txt](configs/signal_to_ratio/models_pretraining_custom_all.txt))

```
apertus-{175M,350M,600M,1B}-fwEdu{30,60,90}-fw{270,240,210}-seed{28,1797,1904}
```

That's the canonical 4 sizes × 3 mixes × 3 seeds. The seed-1904 column landed
first; seed-28 and seed-1797 are the later additions and are still in flight
on the reservation as of 2026-05-04.

**Canonical iter set** (10 evenly spaced over the full run + up to 5 dense
picks from the last 10% of training, deduplicated):

```
2000, 6000, 12000, 18000, 22000, 28000, 34000, 38000, 42000, 44000, 46000, 48000, 50000
```

The dense tail is what makes the late-training plateau readable — without it
we'd have to rely on the gap between iter 44000 and 50000 alone.
[`scripts/list_checkpoints.sh`](scripts/list_checkpoints.sh) generates this via
`--total 10 --dense-tail 5` (`--tail-pct` defaults to 10%).
[`scripts/generate_snr_runner.sh`](scripts/generate_snr_runner.sh) forwards both
flags. [`scripts/snr_progress.py`](scripts/snr_progress.py) anchors all small
Megatron models in a given models file to the LONGEST per-model iter list (so
fully-trained and half-trained models share the same x-axis grid in W&B —
half-trained ones list iters they don't have on disk yet, marked
`not_submitted` until resume training catches up).

The runner [`runners/snr_pretraining_all.sh`](runners/snr_pretraining_all.sh) is the
single-source-of-truth combo: 36 models × 13 iters = 468 cells.

**Half-trained models** — the live picture lives in
[`scripts/pretrain_progress.py`](scripts/pretrain_progress.py); don't hardcode
a snapshot here. As of 2026-05-04 the seed-1904 column is fully done at iter
50000, most of seed-1797 is done, and seed-28 is mid-resume. Eval jobs that
target iters not yet on disk fail at Megatron's `--exit-on-missing-checkpoint`
— that's expected.
[`scripts/launch_ckpts_in_progress.sh`](scripts/launch_ckpts_in_progress.sh)
short-circuits before submitting (skips with "iter dir missing"), and the
per-task idempotency layer reschedules them next launch once the iter exists.

### Reference HF models (4 models, two runners)

[`runners/snr_pretraining_hf_top.sh`](runners/snr_pretraining_hf_top.sh) — 6 jobs:

| Model | Revision |
|---|---|
| HuggingFaceTB/SmolLM3-3B-checkpoints | stage1-step-3440000, stage2-step-4200000, stage3-step-4720000 |
| swiss-ai/Apertus-8B-2509 | main |
| allenai/Olmo-3-1025-7B | stage2-step47684-mix-round5-from-2T-ckpt, stage3-step11921 |

[`scripts/debug_loop.sh`](scripts/debug_loop.sh) is a single-ckpt
debug-partition loop: it keeps resubmitting 1:30 jobs (narrowing each
submission's `TASKS` env to the still-missing set) until every task in
`tasks_pretraining_full.txt` has results, or until two consecutive cycles
make zero progress. Edit the `NAME`/`CKPT_PATH`/`CKPT_ITER` (or the
HF `MODEL`/`REVISION`, switching `LM_EVAL_BACKEND` back to `vllm`) at the
top of the script to retarget. It currently points at
`apertus-350M-fwEdu30-fw270-seed1904-iter50000`. If you also have it
covering a ckpt that's in one of the top runners, you'll duplicate work —
prune one or the other.

[`runners/snr_pretraining_hf_70b.sh`](runners/snr_pretraining_hf_70b.sh) — 1 job:

| Model | Revision |
|---|---|
| swiss-ai/Apertus-70B-2509 | main |

Apertus 8B/70B intentionally use `main` (HF latest) because the user wants the
shipped model rather than a specific intermediate step. Never change to
`step<N>-tokens<M>B` without confirming.

### Local-HF (vLLM-on-converted-megatron) runner

[`runners/snr_pretraining_local_hf.sh`](runners/snr_pretraining_local_hf.sh)
— evaluates the **converted** SNR ckpts (Megatron → HF) staged at
`/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints/<cell>/iter_NNNNNNN/`.
Walks that tree, builds `MODEL_CHECKPOINTS` with one entry per (cell, iter),
then sources `hf_base_runner.sh`. Why prefer this over the megatron path:

- vLLM is faster and dramatically more memory-efficient than the megatron
  eval path, which had ~196 jobs FAIL with cgroup memory pressure on the
  May-3 / May-4 megatron sweep (per `sacct` audit).
- Local paths skip the HF Hub round-trip — no 429 risk during eval (see
  bug history #12), no GB-scale model downloads.
- Sets `BATCH_TASKS=1` (see bug history #13) and lets `TP` default to
  `GPUS_PER_NODE` (see bug history #11 — DO NOT override `TP=1`).
- `SEEDS_FILTER` env var (default `"1904 1797 28"`) restricts which seeds
  to evaluate. Set `SEEDS_FILTER="1797 28"` to skip seed1904 (most of which
  already has megatron eval results from the May-3 megatron sweep).

Submit through the standard launcher:
```bash
bash scripts/launch_evaluations.sh snr-pretraining-full \
    --script runners/snr_pretraining_local_hf.sh --time 02:00:00 --splits 4
```

The `hf_base_runner.sh` skip logic now does TWO checks per ckpt: (a) disk
results.json lists all tasks → skip; (b) `squeue --me -n eval-$MODEL,…-split0,…`
returns any pending/running job → skip. The latter prevents duplicate
submissions when re-launching after a partial run.

---

## Task list

[`tasks_pretraining_full.txt`](configs/signal_to_ratio/tasks_pretraining_full.txt)
— 86 tasks, the dedup union of `tasks_pretraining.txt` (48) and
`tasks_pretraining_b.txt` (38). The launcher mode `snr-pretraining-full` sets
this. **Always launch with `snr-pretraining-full`** unless the user explicitly
wants a subset.

If a task fails, the per-task fault-tolerance in `_run_per_task.sh` logs them and moves on.

---

## The one-liner workflow (idempotent, multi-collaborator-safe)

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals && git pull && \
bash scripts/launch_evaluations.sh snr-pretraining-full \
    --script runners/snr_pretraining_all.sh --time 12:00:00
```

Plus separately for the HF top-of-stage and 70B (which need longer wall times):

```bash
bash scripts/launch_evaluations.sh snr-pretraining-full \
    --script runners/snr_pretraining_hf_top.sh --time 12:00:00
bash scripts/launch_evaluations.sh snr-pretraining-full \
    --script runners/snr_pretraining_hf_70b.sh --time 12:00:00
```

Re-running these is **safe and idempotent at two layers**:

| Layer | Where | When it fires |
|---|---|---|
| Per-checkpoint | [`runners/hf_base_runner.sh`](runners/hf_base_runner.sh) (calls `_eval_status.py`) | At launch, before each `sbatch` — `continue`s if every task in `$TASKS` already has results for that ckpt. Saves the cold-start of a redundant job. |
| Per-task | [`scripts/_run_per_task.sh`](scripts/_run_per_task.sh) (calls `_eval_status.py`) | Inside a running job — filters `$TASKS` down to remaining; logs skipped to `skipped_tasks.log`; exits 0 cleanly if nothing left. |

Both use the same disk-scan in [`scripts/_eval_status.py`](scripts/_eval_status.py): a
task is "done" iff a non-empty `eval_*/per_task/<task>/` exists (killed runs)
**or** any `eval_*/results_*.json` lists it under `.results` (clean runs).

**Race window:** if two collaborators launch simultaneously, both see the same
"missing" set and may submit duplicate sbatch jobs for the same ckpt. The per-
task layer dedups inside the second job — no compute waste, just queue churn.
Stagger by a few minutes when you can.

**Filling gaps with right-sized walltime.** After the canonical-runner sweep
TIMEOUTs leave a long tail of "1–10 tasks remaining" ckpts, re-running the
runner queues fresh 12h jobs for each — wasteful, and the queue priority
suffers. [`scripts/launch_ckpts_in_progress.sh`](scripts/launch_ckpts_in_progress.sh)
is the targeted alternative: it reads `snr_progress.py`, finds every ckpt
that is `[in_progress]` with no active job (stuck) or `[not_submitted]`
(never run), and submits one sbatch per ckpt with walltime sized to the
*remaining* work — `max(30 min, remaining * per_task + cold_start + buffer)`,
with `--time 11:59:00` reserved for fresh ckpts that have no progress yet.
Per-task minute estimates live at the top of the script; adjust if the task
mix shifts again.

---

## Collaborators with shared access

Both `aromanou` and `cmeister747` have POSIX ACLs granting `rwx` on this repo
and `eval_logs` (recursive + default), plus `r-x` traverse on the parent dirs.
Their jobs write to the **same** `eval_logs` tree, so the idempotency check
sees their results too. They share `--account=infra01`.

To grant more colleagues:

```bash
USER_TO_ADD=...
setfacl -m u:$USER_TO_ADD:rx /iopsstor/scratch/cscs/mariagrandury \
    /iopsstor/scratch/cscs/mariagrandury/data-mix-small \
    /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM \
    /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs
setfacl -R -m u:$USER_TO_ADD:rwx /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals
setfacl -R -d -m u:$USER_TO_ADD:rwx /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals
setfacl -R -m u:$USER_TO_ADD:rwx /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs
setfacl -R -d -m u:$USER_TO_ADD:rwx /iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs
```

Each colleague needs their **own** `HF_TOKEN` and `WANDB_API_KEY` exported
(`evaluate.sbatch` reads env first, files in `scripts/` as fallback). Their
W&B key needs membership in `mariagrandury-epflnlp` entity.

```bash
export HF_TOKEN=<your_hf_token> WANDB_API_KEY=<your_wandb_key> && cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals && git pull && bash scripts/launch_evaluations.sh snr-pretraining-full --script runners/snr_pretraining_all.sh --time 12:00:00
```

---

## Cluster gotchas

- **Login node:** `clariden-ln003`. System Python is **3.6** — anything in
  `scripts/snr_progress.py` or `_eval_status.py` that uses 3.7+ syntax must be
  invoked via `python3.11`.
- **Slurm accounts:** `infra01` (default, in `evaluate.sbatch`) and `a139`.
  Group `a139` (NOT `infra01`) is what the unix files belong to.
- **Partition `normal`:** 12 h max wall, hundreds of nodes, but currently
  fully booked (504 reserved for some other project). Fair-share is the
  scheduling lever. Recent heavy usage (mariagrandury > 3× their share) drops
  priority; expect long pending times.
- **Partition `debug`:** 1:30 max wall, 12-ish idle nodes, but
  **`debug-qos` allows max 1 running + 1 queued per user**. `debug_loop.sh`
  exploits exactly that: it submits, blocks, submits the next.
- **`#SBATCH --gpus-per-node=4` is hardcoded** in `evaluate.sbatch`. The 4-GPU
  node is what the eval container expects.
- **No outbound internet from compute nodes.** Pip installs run via the
  cluster's mirror; W&B uploads work because they go through a pre-configured
  proxy.
- **`/iopsstor/scratch` is the work tree.** `/capstor/store` is shared
  long-term storage (read-mostly).

---

## Hard-won bug history (don't reintroduce)

If a future Claude is editing `evaluate.sbatch` / `_run_per_task.sh` /
`_eval_status.py`, these are the things that broke before. Each one cost
hours to track down:

### 1. vLLM tokenizer revision
Some HF repos (notably `HuggingFaceTB/SmolLM3-3B-checkpoints`) leave `main`
empty and put all files on revision branches. lm-eval's vLLM wrapper has a
**separate** `tokenizer_revision` parameter that defaults to `None` — passing
just `revision=...` makes `AutoTokenizer.from_pretrained` fetch `main`, find
no `tokenizer.json`, and die with a misleading "need sentencepiece or
tiktoken" error. Fix: always pin `tokenizer_revision=$REVISION` next to
`revision=$REVISION` (`evaluate.sbatch` already does this).

### 2. Megatron container vs nvidia-modelopt/peft tf-plugin
The `ngc-nemo` container ships `nvidia-modelopt` and PEFT registers
transformers entry-points declaring the deprecated `tf` / `tensorflow_text`
backends. transformers 5.1.0's `BACKENDS_MAPPING` no longer includes those, so
**`import transformers` fatally raises** `ValueError: Backend should be
defined in the BACKENDS_MAPPING. Offending backend: tf` — not just a warning.
Fix: route the megatron path through `containers/env_vllm.toml` (the original
fork's container) instead. **Don't switch back to ngc-nemo** without solving
the plugin discovery problem.

### 3. Megatron checkpoint `_extra_state` strictness
Older megatron checkpoints (everything pre-resume) don't carry the TE
bookkeeping fields that current Megatron-LM expects. The default
`--dist-ckpt-strictness=assume_ok_unexpected` delegates to the underlying
strategy, which fatally raises `Missing key in checkpoint state_dict:
decoder.layers.self_attention.q_layernorm._extra_state/...`. Fix: pass
`extra_args=--dist-ckpt-strictness=log_unexpected` in the megatron model_args
(see `evaluate.sbatch`). Real model weights still load — only TE quantization
metadata is skipped, which is irrelevant for bf16 eval.

The same fix is in the **training** repo's
`/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/submit-apertus-data-mix.sh` (we patched it
when launching the resume jobs). If you regenerate that file from upstream,
re-apply.

### 4. `bash ./scripts/_run_per_task.sh` resolution
The pyxis container's WORKDIR is `/workspace/vllm`, **not** the repo root. A
relative path to `_run_per_task.sh` won't resolve. Fix: capture
`REPO_DIR=$PWD` on the host shell and emit `cd '$REPO_DIR'` as the first line
of every inner `bash -lc` heredoc.

### 5. Pyxis `--environment=` doesn't forward host env vars
`srun --environment=...` runs the command in the container — and contrary to
naive expectation, host env vars exported on the outer shell are **not**
visible inside. `evaluate.sbatch` builds a single `INNER_EXPORTS` string from
the host shell's values (single-quoted) and inlines it into every inner
`bash -lc` heredoc:

```bash
INNER_EXPORTS="export NAME='$NAME' WANDB_ENTITY='$WANDB_ENTITY' WANDB_PROJECT='$WANDB_PROJECT' \
    LOGS_ROOT='$LOGS_ROOT' TASKS='$TASKS' HARNESS_EVAL_DIR='$HARNESS_EVAL_DIR' CMD_BASE='$CMD_BASE'"
```

If you add a new variable that the runner needs, add it here too.

### 6. `Path("comma,list").is_file()` raises ENAMETOOLONG
`debug_loop.sh` builds the next job's `TASKS` env from the remaining-task set —
68+ task names ≈ 1500 chars. `_eval_status.py:parse_tasks_input` was probing
`Path(tasks_arg).is_file()`, which calls `os.stat()`, which on Linux with
`NAME_MAX=255` raises `OSError(36)` for any string that long. Fix: short-
circuit the file probe when the input contains `,` or exceeds 4000 chars, and
catch `OSError` defensively.

### 7. `|| true` masking a filter crash
`_run_per_task.sh` used to do `REMAINING=$(python _eval_status.py ... || true)`
and then `if [[ -z "$REMAINING" ]]; then exit 0; fi`. When `_eval_status.py`
crashed (rc≥2), the `|| true` swallowed it, `REMAINING` was empty, the script
took the "all done" branch and exited 0. **Then `evaluate.sbatch` ran the W&B
upload step unconditionally** and died with `Expected exactly one results
file, found 0`, ending in Slurm state `FAILED`. Fix: capture the rc; rc=1
means "all done", rc≥2 means crash → fall through and run all tasks
unfiltered. **Never silently equate "filter failure" with "nothing to do".**

### 8. Olmo `stage2-step47684-mix-round5-from-2T-ckpt` revision: unrecognised architecture
The Olmo-3-1025-7B repo's stage2 mid-round revision uses model_type
`olmo2-retrofit`, which is not in transformers 5.1.0's model registry. vLLM
fails inside `ModelConfig.__init__` with:

```
pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, The checkpoint you are trying to load has model type
  `olmo2-retrofit` but Transformers does not recognize this architecture.
```

This is **not** caught usefully by the per-task fault tolerance — every
task fails identically (rc=1) for ~12 h until Slurm wall, then the W&B upload
step finds 0 results and the job ends FAILED. Symptom: 86 / 86
`=== FAILED (rc=1)` markers in the log with the same `ValidationError` block
before each. First observed on job 1947613.

Workarounds: skip that revision, install `transformers` from a branch that
ships the `olmo2-retrofit` mapping, or pin the eval container's transformers
to a version that supports it. The non-mid-round Olmo revisions
(`stage1-step1413814`, `stage3-step11921`) and the `main` revision load fine
— the issue is specific to the `mix-round5-from-2T-ckpt` retrofit checkpoints.

### 9. W&B blacklists run IDs after deletion
When you delete a W&B run via the API, its `id` is permanently reserved.
A subsequent `wandb.init(id=<same>, resume="allow")` returns HTTP 409
(`run X was previously created and deleted; try a new run id`). The wandb
client retries 21 times before giving up — symptoms look like a slow
`init_timeout` to the caller. Fix: **don't delete runs** — they're cheap.
If you really need to replace, bump `RUN_ID_SUFFIX` in
[`scripts/push_all_results.py`](scripts/push_all_results.py) (currently
`-v6`) so new pushes use fresh IDs. The display `name` stays clean (it's
just the model name). Also pass `settings=wandb.Settings(init_timeout=300)`
on `wandb.init` from the login node — the default 90 s occasionally trips
under bursty traffic.

### 10. Single-history-point metrics render as bar charts in W&B
W&B's auto-workspace picks panel type per metric: ≥ 2 history points →
line plot; exactly 1 → bar chart with run name on the y-axis. Tasks that
only land on single-ckpt runs (e.g. `xnli_zh`, `multiblimp_arb`,
`global_mmlu_full_zh` evaluated only on Apertus-8B/Olmo-stage1) therefore
render as bars while same-task multi-ckpt runs (175M-fwEdu*) render as
lines. Not a bug — just data sparsity. The custom workspace view built
by the `wandb-workspaces` script forces `LinePlot` with `range_y=(0,1)`
regardless of point count, side-stepping the auto-pick.

### 11. vLLM rejects offline `data_parallel_size > 1` for dense models
Setting `TP=1` so `DP=GPUS_PER_NODE/TP=4` (the comment in `evaluate.sbatch:177`
recommends this for small models) fails on every dense Apertus model with:

```
RuntimeError: Worker 0 failed during generation:
ValidationError: 1 validation error for ParallelConfig
  Value error, Offline data parallel mode is not supported/useful for dense models.
```

vLLM only allows offline DP for MoE models. On 2026-05-04 this killed
599/810 splits in the first vLLM SNR sweep. **Use the default `TP=$GPUS_PER_NODE`
(=4) for dense models** — all 4 GPUs do tensor-parallel inference instead.
Per-node throughput is similar; parallelism comes from `--splits K` (one
sbatch per task chunk), not from offline DP. The comment in `evaluate.sbatch`
about overriding `TP=1` is **stale for vLLM dense** — if you want to revisit,
test with one ckpt before fanning out.

### 12. HF Hub team plan caps at 3000 API requests / 5 minutes
The huggingface_hub team plan (current state on this account) imposes a
hard rate limit. Burst pushes / queries blow through it fast:
- `push-snr.py` pushing 240 iter dirs back-to-back hit it after ~6 cells
  and the per-call retry exhausted (each `upload_folder` does multiple
  internal API calls).
- `convert-snr.sh` fetching `alehc/swissai-tokenizer` for every iter
  triggered it during a 63-iter conversion (Megatron's `HuggingFaceTokenizer`
  + the saver each fetch the tokenizer once per iter).

Mitigations in place:
- `push-snr.py` has 429-aware backoff (`_retry_on_429` parses `Retry-After`
  + has exponential fallback) — works for moderate bursts, exhausts on
  long sweeps.
- `convert-snr.sh` exports `HF_HUB_OFFLINE=1` + `HF_HUB_CACHE` pointing at
  `/capstor/store/cscs/swissai/infra01/users/mariagrandury/hf_models/`
  (where the tokenizer is pre-cached) so per-iter conversion never hits
  the API.

For a long push sweep, expect ~6-10 cells per 5-min window — wait for the
window to clear before retrying. Or split the push across two rate-limit
windows by sleeping 5 min between batches.

### 13. `_run_per_task.sh` reloads vLLM 86 times per ckpt (BATCH_TASKS toggle)
The default per-task loop calls `lm_eval --tasks "$t"` once per task. vLLM
model load is the dominant cost (~30-90s per load on small models),
dwarfing actual inference time. For a 86-task `snr-pretraining-full` sweep,
this is ~2 hours of pure model-load overhead per ckpt.

Set `BATCH_TASKS=1` (env var threaded through `evaluate.sbatch` →
`_run_per_task.sh`) to run ALL remaining tasks in ONE `lm_eval` call. Model
loads once, all tasks share it. ~10x per-ckpt speedup. Trade-off: a mid-
run crash loses unwritten task results in that call (vs the per-task
loop's "lose only the in-progress task"); the per-task idempotency on re-
run still works because the single `results_*.json` lists all tasks under
`.results`.

The vLLM/HF SNR runner ([`runners/snr_pretraining_local_hf.sh`](runners/snr_pretraining_local_hf.sh))
sets `BATCH_TASKS=1` by default. Megatron path keeps the per-task loop
(safer given megatron eval's known cgroup/memory fragility — see bug 11
context).

**`BATCH_TASKS=1` rescue gotcha (2026-05-04):** in batched mode, lm_eval
instantiates EVERY task in `--tasks` upfront — calling `download()` for each
dataset — BEFORE any token is generated. If any one of those `download()`
calls fails, the whole call dies with zero tasks evaluated and zero
`samples_*.jsonl` written, even though many tasks "loaded successfully"
beforehand. So "the loop got past task X" is not a signal that any task
ran — it's only a signal that those datasets exist and parse. We hit
exactly this on 2026-05-04 with the `mlqa.py` dataset-script removal (bug
15 below); 36 jobs spent 22 minutes loading 109 datasets, then `mlqa`
crashed the call → 0 generations across all 36.

For partial-result rescue:
- per-task mode (`BATCH_TASKS=0`, default) — each completed task's
  `eval_*/per_task/<task>/results.json` survives walltime kills; only the
  in-flight task is lost. `push_all_results.py` aggregates these.
- batched mode (`BATCH_TASKS=1`) — `samples_*.jsonl` per task lands as each
  task finishes (because of `--log_samples`), but the unifying
  `results_*.json` is only written at the very end. Walltime mid-batch
  ⇒ the samples files exist but no aggregated metrics; you'd need a custom
  script to compute scores from samples (none in repo today). For
  long-running sweeps where walltime is tight, prefer `BATCH_TASKS=0`.

---

### 14. vLLM `tensor_parallel_size` must divide `num_key_value_heads`
vLLM rejects model load with:

```
assert self.total_num_kv_heads % tp_size == 0
AssertionError
```

This kills the worker process before any inference. For the SNR Apertus
sizes (HF-converted checkpoints under
`/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints/`), the GQA
counts are:

| Size | num_attention_heads | num_key_value_heads | Valid TP |
|---|---:|---:|---|
| 175M | 16 | 4 | 1, 2, 4 |
| 350M | 20 | 5 | **1 only** |
| 600M | 24 | 6 | 1, 2 |
| 1B   | 28 | 7 | **1 only** |

`evaluate.sbatch:177` defaults `MP=$GPUS_PER_NODE=4`, which **only works
for 175M**. We hit this on 2026-05-04 — 31 of 36 eval jobs (the 350M /
600M-with-TP=4 / 1B cases) all failed at vLLM `WorkerProc` init.

The fix at launch time is to set `TP` per-size in the `--export=` line of
`sbatch`, e.g.:
```bash
case $size in
  175M) tp=4 ;;
  350M) tp=1 ;;
  600M) tp=2 ;;
  1B)   tp=1 ;;
esac
sbatch ... --export=ALL,TP=$tp,...
```

`runners/snr_pretraining_local_hf.sh` does NOT do this today (it lets
`MP` default), so launching the SNR sweep through that runner fails for
all non-175M cells. Either add a per-cell TP map to that runner, or use
the inline submission loop from this bug entry's launch pattern. (For
larger models that don't fit on one GPU, TP must be ≥ ceil(model_size /
single_gpu_mem) anyway, but for SNR sizes a single GPU is plenty.)

---

### 15. `datasets` v3+ removed support for script-based dataset loaders
The eval container's `datasets` library version no longer loads HF
datasets that ship a `<repo>.py` script (the legacy "dataset script"
format). Symptom inside lm_eval task instantiation:

```
RuntimeError: Dataset scripts are no longer supported, but found mlqa.py
```

This kills the entire `lm_eval` call (in batched mode, all 36 jobs
in flight; in per-task mode, just that one task). On 2026-05-04 this hit
during the `tasks_pretraining_full.txt` load of `mlqa*` and several other
script-only datasets.

**Fix**: prune any remaining script-format tasks from
`configs/signal_to_ratio/tasks_pretraining_full.txt`. To audit, look up
each task's HF dataset repo and check whether it contains `*.py` (legacy)
vs the parquet/json data-only layout (modern). A datasets-side pin to
`<3.0` could also work but conflicts with other deps; pruning is cleaner.
On 2026-05-04 the user trimmed every entry from `mlqa` onward (109 → 83
tasks).

---

### 16. `HF_HUB_CACHE` doesn't propagate into the eval container
`HF_HOME` and `HF_HUB_CACHE` set on the host shell (e.g. by the user's
conda env init) are NOT visible inside `srun --environment=...` (per bug
5). If `evaluate.sbatch` doesn't explicitly thread them through
`INNER_EXPORTS`, the container starts with the default in-image cache
(`/root/.cache/huggingface`), forcing every job to re-download tokenizer
+ datasets — which is what triggered the 2026-05-04 HF 429 cascade
(bug 12).

The populated cache lives at
`/capstor/store/cscs/swissai/infra01/users/$USER/hf_models` (~258 GB,
already mounted in both `containers/env.toml` and
`containers/env_vllm.toml`). `evaluate.sbatch` now defaults to it and
forwards both vars via `INNER_EXPORTS`:

```bash
export HF_HOME=${HF_HOME:-/iopsstor/scratch/cscs/$USER/hf_home}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/capstor/store/cscs/swissai/infra01/users/$USER/hf_models}
INNER_EXPORTS="export ... HF_HOME='$HF_HOME' HF_HUB_CACHE='$HF_HUB_CACHE'"
```

Without these two lines, expect a 429 cascade the next time you launch
>10 concurrent eval jobs. To pre-warm a missing model/dataset, run from
the `snr` conda env on the login node:
```bash
hf download <repo>          # respects $HF_HUB_CACHE from the env
```

---

### 17. Long-sleeping `nohup` submitters capture the OLD repo path
`scripts/submit_175M_topup_delayed.sh` (and any similar one-shot delayed
submitter) does `cd "$(dirname "$0")/.."` then sets
`TASKS_FILE=$PWD/configs/...` early, then `sleep 28800` (8 h) before
submitting. If the repo is moved or deleted during the sleep window, the
captured `$PWD` (and therefore `TASKS_FILE`) still points at the old
location — the submitted jobs then fail in two flavors:

- `srun: error: pyxis: --environment: failed to open './containers/env_vllm.toml': No such file or directory`
  (the slurm CWD is the dead old repo path, so the relative `./containers/...`
  in `evaluate.sbatch` fails to resolve; jobs die in ~15-25 s).
- `ValueError: Tasks not found: /old/path/.../tasks_pretraining_full.txt`
  (the absolute `TASKS` env points at the dead path; lm-eval can't find
  the file → fails in ~1:30 after container startup).

We hit both on 2026-05-06 when the repo moved from
`/iopsstor/scratch/cscs/mariagrandury/swissai-evals-post-train` to
`/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals`.
Roughly 30 jobs (2047059–2050368) failed for these reasons even though
all the code in the new tree was correct — the failing jobs were
submitted by a `nohup`'d delayed-launcher process that had captured the
old path before the move.

Mitigations:
- **Before moving/renaming the repo**, kill any long-sleeping submitters:
  `pgrep -af "submit_.*delayed|nohup.*launch_eval" | awk '{print $1}' | xargs -r kill`.
- **Defensive guard in delayed scripts**: re-validate `[[ -f "$TASKS_FILE" ]]`
  right before each `sbatch` call, and abort the loop with a clear error
  if missing. This is cheap (one stat) and turns a silent 30-job failure
  into a single explicit message.
- Prefer absolute paths derived from a *resolved* canonical location
  (e.g. `realpath`) rather than the script's `$PWD`, so a moved repo is
  caught at `realpath` time rather than at job-submit time.

---

## Smoke tests (when you change `evaluate.sbatch` / `_run_per_task.sh`)

Use the four committed smoke tests in
[`configs/signal_to_ratio/README.md`](configs/signal_to_ratio/README.md#smoke-tests).
They run through the `single` mode with `--limit 4`, route results to the
`-single` W&B project to keep them out of the real sweep, and exercise both
the HF (vLLM) and Megatron paths. Re-run them before any merge that touches
`evaluate.sbatch`.

---

## Progress dashboard

[`scripts/snr_progress.py`](scripts/snr_progress.py) is the truth source:

```bash
python3.11 scripts/snr_progress.py                                    # per-ckpt summary
python3.11 scripts/snr_progress.py --status not_submitted             # gaps
python3.11 scripts/snr_progress.py --details --filter <NAME-substr>   # per-task
python3.11 scripts/snr_progress.py --models configs/signal_to_ratio/models_pretraining_custom_all.txt
```

It enumerates the same `(model, ckpt)` cells that the runners would and
cross-references `eval_logs` for completion + `squeue --me` for in-flight
jobs (matching by `eval-<NAME>-*` prefix so suffixes like `-b`, `-srun-debug`
don't break recognition).

---

## W&B layout (per-model curves, single metric per benchmark) {#wb-layout}

Project: [`mariagrandury-epflnlp/snr-experiments`](https://wandb.ai/mariagrandury-epflnlp/snr-experiments).
The previous incarnation of this project (with per-subtopic charts and
default x=tokens) was renamed to `snr-experiments-legacy` on 2026-04-28
when the schema below landed; the live project is a clean rebuild.

[`scripts/push_all_results.py`](scripts/push_all_results.py) is the single
entry point. It does **two** jobs:

- **Single-NAME** (called from `evaluate.sbatch` / `aggregate_splits.sbatch`
  inside the pyxis container after a successful eval):
  `python scripts/push_all_results.py --name $NAME --eval-duration $DURATION`.
  Appends one step to the model's W&B run.
- **Bulk rescue** (login node, with the `snr` conda env that has wandb 0.25):
  `/users/mariagrandury/miniconda3/envs/snr/bin/python scripts/push_all_results.py`.
  Walks every NAME under `eval_logs/.../snr-experiments/`, groups by model,
  pushes the union of disk results, then saves the workspace view (below).

**Run shape**:
- One W&B run per *model* (not per ckpt). Run id = `<sanitised-model-name><RUN_ID_SUFFIX>`.
  The id suffix exists because deleting a wandb run blacklists its id (bug 9).
- Each ckpt = one `run.log({...})` call → one history step.
- Logged per ckpt: bare `iter`, `tokens`, `flops`, optional `eval_duration_seconds`,
  plus exactly one `<task>/acc` (or `<task>/exact_match`) per benchmark.
- `define_metric("*", step_metric="flops")` makes flops the default x-axis.
  `iter` and `tokens` are also defined and can be swapped in via the W&B UI
  (Edit panel → X-axis).

**Per-benchmark metric** — `flatten()`: prefer `acc`; fall back to
`exact_match` (mgsm-style); skip the task otherwise. `acc_norm`,
`acc_bytes`, `degeneration`, all `*_stderr` are intentionally dropped.

**Subtopic aggregation** — `aggregate_parents()`: a task `T` is collapsed
into its parent `P` if `P` is an underscore-prefix of `T` and `P` is also
in the results. So `mmlu_anatomy`, `mmlu_humanities`, … all fold into
`mmlu`; `global_mmlu_full_zh_stem`, `global_mmlu_full_zh_humanities`, …
fold into `global_mmlu_full_zh`. The merge step strips the `groups` field
that holds aggregates like `mmlu`, so `collect()` reads both `results`
and `groups` from per-task fragments to recover them.

**Tokens/FLOPs** — computed at push time; FLOPs ≈ `6 × params × tokens`:
- Megatron iter→tokens: `iter × 504 × 4096` (`MEG_TOKENS_PER_ITER`)
- HF stage→cumulative tokens: `HF_STAGE_TOKENS` lookup table at the top of the script
  (Olmo-3 stages 1/2/3, SmolLM3-3B-checkpoints stages 1/2/3)
- HF main→tokens: `HF_MAIN_TOKENS` (Apertus-8B/70B-2509 → 15 T)
- Params: parsed from the model name (`apertus-NB-…`, `<repo>-NB-…`)
- If we don't know params for some new model, FLOPs is skipped — the iter and
  tokens charts still work. Add it to the lookup before pushing.

**Workspace view** "flops vs metric (y in [0,1])" — one `LinePlot` per
benchmark with `x="flops"`, `range_y=(0, 1)`. Built automatically by
`setup_workspace()` at the end of every bulk push (uses
`wandb-workspaces`, installed in the `snr` conda env; gracefully skipped
in the pyxis container which lacks it). The default project view is
wandb's auto-workspace and will mix bar/line per bug 10 — point
colleagues at the named view instead.

---

## Rescue procedure (when a job hits Slurm wall before merging)

`_run_per_task.sh` writes per-task results to `eval_*/per_task/<task>/` **as
each task finishes**. If Slurm kills the job mid-loop, only the in-progress
task is lost; everything already finished survives on disk. The next bulk
push picks them up automatically (idempotent), so the simplest "rescue" is:

```bash
/users/mariagrandury/miniconda3/envs/snr/bin/python scripts/push_all_results.py
```

That walks every NAME on disk (incl. unfinished `eval_*/per_task/` dirs) and
appends any new ckpts to their model's W&B run. Same effect as resubmitting
the eval job, without the cluster cost. The Megatron-side merge of partial
results still happens inside `_run_per_task.sh` for any future job — no
manual `merge_split_results` step required.

---

## Outsourcing to Colab

[`notebooks/snr_eval_colab.ipynb`](notebooks/snr_eval_colab.ipynb) mirrors the
cluster pipeline on a single A100. It writes results in the same directory
shape so an `rsync` back to the cluster's `eval_logs` triggers cluster-side
idempotency (skips Colab-evaluated tasks). 70B doesn't fit on A100 — keep it
on the cluster.

---

## Pretraining infrastructure (separate repo)

`/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/` is
the training submitter. There's a back-of-house memo over there at
[`/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/CLAUDE.md`](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/CLAUDE.md);
read it for the training-side gotchas. Quick orientation:

- `submit-apertus-data-mix.sh` — the sbatch template. Patched with
  `--dist-ckpt-strictness log_unexpected` (Apr 26) so resumes tolerate older
  checkpoints' missing TE keys.
  **Don't revert** the strictness flag without solving the underlying
  TE/Megatron version skew; **do** drop the reservation line once the
  reservation expires (currently runs until 11 May 12:00).
- `launch_trainings.py` — wraps `sbatch --export=…` from `hyperparams_deep.json`.
  Default `SEEDS = [28, 1797, 1904]` (the canonical SNR set). One sbatch per
  (size × mix × seed); supports `--dry-run`, `--test`, and the usual filters.
- `launch_resumes.sh` — **the right entry point for filling gaps**. Reads
  `pretrain_progress.py`, iterates the canonical 4×3×3 cells, and dispatches
  per cell: `[done]` → skip · already in `squeue` → skip · `[in_progress]` →
  resume with auto-computed walltime · `[corrupt]` → wipe `checkpoints/` and
  submit fresh · `[no_ckpts]` / no exp dir → submit fresh. Idempotent —
  re-running is safe.
- `pretrain_progress.py` (this repo at [`scripts/pretrain_progress.py`](scripts/pretrain_progress.py))
  — also the truth source for the training side. Validates that each
  `iter_NNNNNNN/` actually contains a `.metadata` file *and* ≥ 1 `.distcp`
  shard before counting it as resumable. A marker pointing at an iter dir
  with only `common.pt`/`metadata.json`/`.metadata` (no shards) is flagged
  `[corrupt] (latest valid: …)` so launchers skip it instead of submitting
  a doomed resume. We hit this on `175M-fwEdu60-fw240-seed28` on 2026-05-04.

The pretraining checkpoint dir lives at
`/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/<EXP_NAME>/checkpoints/`.
EXP_NAME format: `apertus-${MODEL_SIZE}-fwEdu${FW_EDU_RATIO}-fw2${FW2_RATIO}-seed${SEED}`.
Slurm job-name format (used by `launch_resumes.sh` for dedup):
`apertus-${size_lc}-edu${FW_EDU_RATIO}-fw2${FW2_RATIO}-seed${SEED}`.

---

## Key recent commits (for context when reviewing changes)

```
ab4daf4 push eval results to W&B as one curve per model, not one run per ckpt
46b3852 fix three bugs that surfaced once real eval jobs started running
2cc58cf make SNR eval launches idempotent + add progress dashboard + canonical runner
aec82e0 fix smoke tests 1-4: vllm tokenizer_revision + megatron container/cwd/strictness
```

Read those commit messages — they document the failure→fix story end to end.
