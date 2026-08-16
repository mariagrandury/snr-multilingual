# Context for Claude — `data-mix-small` pretraining

This directory submits the small-multilingual Apertus pretraining sweep.
Companion to [README.md](README.md) (user-facing) — this file is the back-of-
house memo: what's wired to what, and the failure modes worth remembering.

The eval side lives in `/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/`,
with its own [`CLAUDE.md`](../evals/CLAUDE.md). Both
sides share the same checkpoint tree (this side writes, that side reads).

---

## What we're building

The canonical sweep is **4 sizes × 3 mixes × 3 seeds = 36 models**, each
trained to **iter 50000** (≈ 100B tokens at 504 × 4096 tokens/iter).

| Axis | Values |
|---|---|
| Size | 175M, 350M, 600M, 1B |
| Mix (FW-Edu / FW2-HQ) | 30/70, 60/40, 90/10 |
| Seed | 28, 1797, 1904 |

EXP_NAME: `apertus-${MODEL_SIZE}-fwEdu${FW_EDU_RATIO}-fw2${FW2_RATIO}-seed${SEED}`
→ checkpoint dir at
`/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/<EXP_NAME>/checkpoints/`.

Slurm job name (used by `launch_resumes.sh` for dedup): same skeleton but
size lowercased and dashed, e.g. `apertus-175m-edu60-fw240-seed28`.

---

## The four scripts and how they fit

| File | Role |
|---|---|
| [`submit-apertus-data-mix.sh`](submit-apertus-data-mix.sh) | The sbatch template. Reads env vars (MODEL_SIZE, NUM_LAYERS, …, FW_EDU_RATIO, FW2_RATIO, SEED, TRAINING_STEPS, LR, MBS) injected by the launcher. `--save` and `--load` both point at the experiment's checkpoint dir, so the same script handles fresh and resume runs. Pinned to `--use-checkpoint-opt_param-scheduler` (see failure mode #6). |
| [`launch_trainings.py`](launch_trainings.py) | Wraps `sbatch --export=…` from [`hyperparams/hyperparams_deep.json`](hyperparams/hyperparams_deep.json). One sbatch per (size, mix, seed). Default `SEEDS = [28, 1797, 1904]`. Supports `--size`, `--mix_en`, `--seed` filters, `--dry-run`, `--test`, `--training-steps N` (cap an early exit), and pass-throughs (`--time`, `--account`, `--dependency`). |
| [`pretrain_progress.py`](pretrain_progress.py) | Status. Three modes: text dashboard (default); `--plot PATH` writes `PATH` (canonical-stage 3-panel heatmap with HF/Hub stages, queries the Hub) plus a companion `PATH_all` (every 2000-step iter, megatron-presence only); `--actions` emits one machine-readable line per cell — `done` / `fresh\t<target>` / `corrupt\t<n_iters>` / `resume\t<load_iter>\t<target>` — consumed by `launch_resumes.sh`. |
| [`launch_resumes.sh`](launch_resumes.sh) | **The right entry point for "fill every canonical iter ≤ target"**. Reads `pretrain_progress.py --actions`, dispatches per cell: `done` → skip · in `squeue` → skip · `fresh` → submit a from-scratch run · `resume <load_iter> <target>` → if `<load_iter>` is below the current `latest_checkpointed_iteration.txt` marker (mid-gap backfill), rewind the marker first, then submit with `--training-steps <target>` · `corrupt` → **skip with a warning** (we never auto-rm). Re-runnable. |

`pretrain_progress.py` validates each `iter_NNNNNNN/` has both `.metadata`
and ≥ 1 `.distcp` shard before counting it as valid. Mid-gap canonicals
(missing iter X with X+ canonicals present) are filled one-at-a-time — the
launcher targets the *earliest* missing canonical per cell per call, and
re-running picks up the next gap once the previous job finishes (jobs on
the same cell would race the checkpoint dir, so chaining is left to the
operator).

The standard one-liner to drive everything to 50000:

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain && \
  bash launch_resumes.sh                # add --dry-run first when in doubt
```

---

## Hard rules

- **Never delete checkpoints, eval results, or force-push.** The previous
  `launch_resumes.sh` had a `wipe_checkpoints()` step for `[corrupt]` models
  that ran `rm -rf checkpoints/`. That path is gone (removed 2026-05-09).
  When the disk state for a model is unrecoverable (iter dirs exist but none
  is a valid `.metadata + .distcp` checkpoint), the launcher now **skips the
  model with a warning** so the user can inspect and decide manually. The
  cost of a corrupt dir sitting on disk is near-zero compared to losing
  recoverable training time.

---

## Hard-won failure modes

### 1. Megatron `_extra_state` strictness on resumes
Older checkpoints don't carry the TE bookkeeping fields current Megatron-LM
expects. The default `--dist-ckpt-strictness=assume_ok_unexpected` raises
`Missing key in checkpoint state_dict: decoder.layers.self_attention.q_layernorm._extra_state`.
The script pins `--dist-ckpt-strictness log_unexpected` so weights load and
only the TE quantization metadata is skipped (irrelevant for bf16 training).
**Don't revert** without solving the underlying TE/Megatron version skew —
the eval-side container has the same fix; see the eval CLAUDE.md bug 3.

### 2. Async-save shell directories (the "corrupt" case)
With `--async-save`, Megatron creates `iter_NNNNNNN/` and writes
`.metadata` + `common.pt` + `metadata.json` quickly, then streams in the
`.distcp` weight shards in the background. If the process is killed
between the metadata write and the shard write, the dir survives as a
**3-file shell** with no shards — and `latest_checkpointed_iteration.txt`
may already point at it. Symptom on the next resume:

```
FileNotFoundError: ... /checkpoints/iter_NNNNNNN/__35_0.distcp
```

`pretrain_progress.py` flags any iter that fails the `.metadata + .distcp`
check as not valid. The launcher behaves differently depending on whether
*any* valid iter remains:

- **Some valid iter remains** (most cases): the launcher resumes from the
  latest valid iter — by rewinding `latest_checkpointed_iteration.txt`
  before sbatch if needed — leaving the corrupt iter dirs in place. They
  get overwritten when training next traverses that step, or are otherwise
  harmless.
- **No valid iter at all** (e.g. `175M-fwEdu60-fw240-seed28` on 2026-05-04,
  iter dirs `0002000`–`0014000` all shell): the launcher **skips with a
  warning** and waits for manual cleanup. We do not auto-`rm -rf` the
  checkpoints/ tree.

### 3. Slurm reports `COMPLETED` even when the inner step crashed
The wrapper `submit-apertus-data-mix.sh` exits cleanly after `srun` returns,
so `sacct -j <id> --format=State` shows `COMPLETED 0:0` for the parent job
even when the inner training step was killed. Always check the **`.0` step**
state too:

```bash
sacct -j <id> --format=JobID,State,ExitCode
# Look for "<id>.0  CANCELLED  0:15" — that's training, killed by SIGTERM.
```

The training log (`logs/slurm/training/<jobname>-<id>.err`) has the real
story.

### 4. The 1h SIGUSR2 grace window
`#SBATCH --signal=SIGUSR2@3600` is what triggers the in-Megatron
`--exit-signal-handler` to checkpoint and exit cleanly before walltime.
That's why `launch_resumes.sh`'s `auto_time()` adds a 2h30m margin on top
of the iter-rate estimate (1h grace + cold-start + buffer) and rounds up to
the nearest 15 min, capped at 11:59:59 (the normal queue ceiling).

### 5. seed 1904 used to be the odd one out
Pre-2026-05-04, `launch_trainings.py` had `SEEDS = [28, 64, 1797]` — the SNR
canonical seed `1904` had to be passed explicitly with `--seed 1904`, and
`64` was never actually run. The default is now the canonical
`[28, 1797, 1904]`. If you see seed `64` referenced anywhere, it's stale.

### 6. `OptimizerParamScheduler` train_iters mismatch on mid-gap fills
Megatron's `OptimizerParamScheduler.load_state_dict` runs an exact-match
assertion on the schedule total (`train_iters × GBS` = `train_samples`)
between the CLI args and the checkpoint. For the canonical sweep the saved
value is **25_200_000** samples (50000 iters × 504 GBS). When `launch_resumes.sh`
fills a mid-gap (e.g. canonical 22000 missing → submit with `--train-iters
22000`), the CLI value becomes 11_088_000 samples and the load aborts with:

```
AssertionError: OptimizerParamScheduler: class input value 11088000 and
checkpointvalue 25200000 for total number of iterations do not match
```

Fix (pinned in [`submit-apertus-data-mix.sh`](submit-apertus-data-mix.sh)
on 2026-05-10): add `--use-checkpoint-opt_param-scheduler`. Megatron then
keeps the schedule values **from the saved checkpoint** (peak LR 9.79e-4 for
175M, the original 50000-iter WSD curve, etc.) and the assertion is
bypassed. The training loop still exits at `--train-iters`, so the
mid-gap window stays in the saved schedule's correct phase (peak constant
through canonical 22000 since WSD decay only starts at 40000) and the LR
that lands on the recovered canonical is identical to the original
trajectory's LR at that step.

**Don't reach for `--override-opt_param-scheduler` instead** — that would
recompute the scheduler from CLI args (warmup/decay-iters relative to the
new `train_iters`) and put the model deep in WSD decay at iter 22000,
which is exactly what we do *not* want.

Verified end-to-end on 2026-05-10 against three canonicals (175M-edu90-seed28
→ 22000, 600M-edu30-seed1797 → 34000, 175M-edu60-seed1797 → 18000): LR /
loss / sample-count / token-count all continuous across the resume
boundary, no spike on the recovered canonical.

---

## Per-size cluster cost

Sampled from 1.26M iter lines across all training logs (2026-05-10):

| size | nodes | MBS | median ms/iter | hours to 50000 (steady) |
|---|---|---|---|---|
| 175M | 6 | 7 | **800** | ~11.1 h |
| 350M | 14 | 3 | **565** | ~7.8 h |
| 600M | 21 | 6 | **520** | ~7.2 h |
| 1B | 21 | 6 | **715** | ~9.9 h |

These match the `ITER_MS` table in `launch_resumes.sh` and feed `auto_time()`,
which adds a 2h30m margin for SIGUSR2 grace + cold-start + buffer. p20–p80
spread is ~2 ms for 600M/1B; for 175M and 350M, p80 is inflated by save-iter
overhead (less amortized at smaller node counts) — true per-iter is right
at the median.

---

## Live state (read, don't trust this file's snapshots)

```bash
# Per-model progress + corrupt detection
python3.11 /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/pretrain_progress.py --target 50000

# Drive the gap to zero (idempotent)
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain && \
  bash launch_resumes.sh --dry-run        # first
  bash launch_resumes.sh                  # then for real
```

`launch_resumes.sh --filter <SUBSTR>` scopes by model name substring (e.g.
`--filter seed28`, `--filter 175M`).
