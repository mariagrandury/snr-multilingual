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

## The three scripts and how they fit

| File | Role |
|---|---|
| [`submit-apertus-data-mix.sh`](submit-apertus-data-mix.sh) | The sbatch template. Reads env vars (MODEL_SIZE, NUM_LAYERS, …, FW_EDU_RATIO, FW2_RATIO, SEED, TRAINING_STEPS, LR, MBS) injected by the launcher. `--save` and `--load` both point at the experiment's checkpoint dir, so the same script handles fresh and resume runs. |
| [`launch_trainings.py`](launch_trainings.py) | Wraps `sbatch --export=…` from [`hyperparams_deep.json`](hyperparams_deep.json). One sbatch per (size, mix, seed). Default `SEEDS = [28, 1797, 1904]`. Supports `--size`, `--mix_en`, `--seed` filters, `--dry-run`, `--test`, and pass-throughs (`--time`, `--account`, `--dependency`). |
| [`launch_resumes.sh`](launch_resumes.sh) | **The right entry point for "make all 36 cells reach 50000"**. Iterates the canonical cross-product, parses the progress dashboard, and per cell dispatches: `[done]` → skip · already in `squeue` → skip · `[in_progress]` → resume (auto-time) · `[corrupt]` → `rm -rf checkpoints/` then submit fresh · `[no_ckpts]`/no dir → submit fresh. Re-runnable. |

Live state is read from
`/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/pretrain_progress.py`
(this is the same script the eval repo uses; it lives over there because
both sides depend on it). It validates each `iter_NNNNNNN/` has both
`.metadata` and ≥ 1 `.distcp` shard before counting it as resumable, and
emits `[corrupt] (latest valid: …)` when the marker file points at a dir
that's missing shards — so `launch_resumes.sh` knows to wipe instead of
submit a doomed resume.

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

---

## Live state (read, don't trust this file's snapshots)

```bash
# Per-model progress + corrupt detection
python3.11 /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/pretrain_progress.py --target 50000

# Drive the gap to zero (idempotent)
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain && \
  bash launch_resumes.sh --dry-run        # first
  bash launch_resumes.sh                  # then for real
```

`launch_resumes.sh --filter <SUBSTR>` scopes by model name substring (e.g.
`--filter seed28`, `--filter 175M`).
