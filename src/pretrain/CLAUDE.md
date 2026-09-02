# Context for Claude — predictivity-sweep pretraining

This directory trains the small-to-large predictivity sweep on **two
platforms at once** — the CSCS cluster (SLURM) and Azure ML — from one code
path. Companion to [README.md](README.md) (user-facing) and
[azure/README.md](azure/README.md) (Azure walkthrough) — this file is the
back-of-house memo: what's wired to what, and the failure modes worth
remembering.

The eval side lives in `../evals/` (cluster) with its own CLAUDE.md. Both
platforms auto-evaluate during pretraining: `auto_evals_cscs.py` (cluster)
and `auto_evals_azure.py` (blob storage), same due rule, same W&B project.

---

## The architecture invariant (don't break it)

**Every Megatron training argument lives in `megatron_args.sh` and nowhere
else.** The two wrappers — `launch_pretraining_cscs.sh` (sbatch/srun/pyxis)
and `launch_pretraining_azure.sh` (torchrun) — only add platform machinery
and call `build_megatron_cmd`. The single intentional platform delta is the
SLURM graceful-exit pair (`--exit-signal-handler --trigger-path`), appended
iff `TRIGGER_PATH` is set. If you ever need a new training flag, add it to
`megatron_args.sh` so both platforms get it; adding it to one wrapper
reintroduces the drift this design removed.

`launch_trainings.py` is the single submitter for both platforms
(`cscs`/`azure` positional arg). It builds one env-var dict per cell
(`cell_env`) — that dict IS the run definition; sbatch `--export` and
`az ml job create --set environment_variables.*` are just transports.

| File | Role |
|---|---|
| `megatron_args.sh` | all Megatron args + W&B block; `WANDB_ENTITY` constant lives here |
| `launch_pretraining_cscs.sh` | SBATCH header, Meg-Runs dirs, SIGUSR2 trigger, srun+pyxis, debug log |
| `launch_pretraining_azure.sh` | `azure/get_megatron.sh` checkout, MBS auto-shrink to GPU count, torchrun |
| `launch_trainings.py` | grid (56 cells) + filters + both submit backends; **idempotent** — per cell it skips done/active, warns on corrupt, resumes partial (marker rewind + auto-sized walltime). There is no separate resume script. |
| `pretrain_progress.py` | CSCS per-cell actions (`done/fresh/resume/corrupt` — the same `cell_action` the launcher uses) + `--is-valid` CLI + the plan table and three heatmaps (`--plot`): planned runs, finished models, and eval work outstanding. `--plot` also rewrites the generated grid block in README.md and the plan doc, so the figures and counts cannot drift from the constants in `launch_trainings.py`. |
| `auto_evals_cscs.py` | CSCS watcher: per due ckpt (every 2nd + final + the FLOPs milestones of `configs.milestone_iters`) submits convert-snr then evaluate.sbatch; needs models.json entries (`sync_models_json.py`) |
| `sync_models_json.py` | upserts one models.json entry per grid cell — conversion + W&B push resolve through it |
|  `auto_evals_azure.py` | Azure watcher: same due rule against blob storage |

Cell name everywhere (checkpoint dir, W&B run id/name, models.json key,
parsed by `pretrain_progress.py`):
`lm-<size>-L<L>[-schemeB]-<deep|shallow>-seed<seed>` — `lm`, not `apertus`:
the architecture has diverged from Apertus (renamed 2026-08-21). Job display
names drop the `lm-` for a kind prefix instead
(`launch_trainings.job_name`): `pretrain-90M-L8-deep-seed1904`,
`eval-90M-L8-deep-seed1904-iter425`, `convert-...`. CSCS checkpoints:
`/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/msnr/<cell>/checkpoints/`.
Azure: `predictivity/runs/<cell>/checkpoints` in each workspace's blob store.

Per-size schedule (iters/warmup/decay for D(N) = 100 × N) comes from the
`predictivity` block in `hyperparams/hyperparams_{deep,shallow}.json` — the
top-level `train_iters: 50000` in those files belongs to the finished
36-model sweep, not this one. Two more knobs are launcher-derived per cell
(not in the JSONs): `ADEMAMIX_WARMUP` = the cell's target iters (alpha/beta3
warm up over the full run — always the target, even on capped resumes, so
every submission runs the same optimizer schedule) and `INIT_STD` =
0.008944 × √(1792/hidden) (width-scaled init anchored at the 1B).

W&B: one continuous run per cell across resumes — `megatron_args.sh` sets a
deterministic `WANDB_RUN_ID` (the cell name, dots sanitized) +
`WANDB_RESUME=allow`, so resubmissions append instead of fragmenting into
one run per job. This replaced the old post-hoc merge tool
(`merge_wandb_experiment.py`, deleted — its companion script never existed
in this repo). Corollary of fixed ids: **never delete a run in msnr** — W&B
blacklists deleted run ids forever (evals CLAUDE bug 9) and the cell could
then never log again without a code-side id suffix.

---

## Hard rules

- **Never delete checkpoints, eval results, or force-push.** When a cell's
  disk state is unrecoverable (iter dirs exist but none valid), the tooling
  skips with a warning — cleanup is always a human decision.
- **Never change a grid cell's training config.** #5 covers not changing the
  optimizer *schedule* on a resume; this is the wider rule, across cells: 24
  cells are trained, and a rung that ran different hyperparameters is not on
  the same ladder as the rest — the scaling fit cannot absorb it, so "fixing"
  one rung means re-running every rung. `launch_trainings.py` has exactly two
  config-perturbing flags, `--lr` and `--ademamix-beta3-factor`; both are
  opt-in, both require a `--size/--langs/--seed` filter, and both **force a
  `diag-` EXP_NAME**. That rename is the enforcement, not a convention:
  `diag-` matches neither `pretrain_progress.NAME_RE` nor
  `ladder_report.LOG_RE`, and `sync_models_json` derives its keys from
  `exp_name()`, so a perturbed run cannot occupy a cell's checkpoint dir,
  models.json entry or W&B run id. Keep it that way — if you add a knob, add
  it to the same mechanism. `megatron_args.sh` must keep every such knob as
  `${VAR:-<the ladder value>}` so an unset variable reproduces the trained
  cells exactly. Background: `plan/90M-rung-anomaly.md`.
- **W&B**: entity is the constant `mariagrandury-epflnlp`
  (`megatron_args.sh`); the project comes from `configs/hf_wandb.json`
  (`msnr`) for BOTH training runs and the predictivity eval pushes
  (`azure/eval.sh` reads it from the repo snapshot; `auto_evals_azure.py`
  keys its done-check on the same config), so loss and benchmark curves
  live in one project. Only the legacy 36-sweep eval infra in `../evals/`
  still points at `snr-experiments`.

---

## Hard-won failure modes (inherited from the 36-model sweep — same stack)

### 1. Megatron `_extra_state` strictness on resumes
Default `--dist-ckpt-strictness=assume_ok_unexpected` fatally raises on
checkpoints saved with a different TE version. `megatron_args.sh` pins
`--dist-ckpt-strictness log_unexpected`: weights load, only TE bookkeeping
is skipped (irrelevant for bf16). **Don't revert.**

### 2. Async-save shell directories (the "corrupt" case)
With `--async-save`, a killed job can leave `iter_N/` holding `.metadata` +
`common.pt` but **no `.distcp` shards** — and the
`latest_checkpointed_iteration.txt` marker may point at it. Next resume dies
with `FileNotFoundError: ...__35_0.distcp`. `pretrain_progress.py` counts an
iter valid only with `.metadata` + ≥1 `.distcp` (the `--is-valid` CLI is the
single source of truth; `conversion/convert-snr.sh` uses it too), and
`launch_trainings.py` rewinds the marker to the latest valid iter before
resubmitting. Note the check is deliberately loose — a tighter byte-level
parse over-rejected good iters (2026-05-14).

### 3. Slurm reports `COMPLETED` even when the inner step crashed
The wrapper exits cleanly after `srun` returns. Check the `.0` step:
`sacct -j <id> --format=JobID,State,ExitCode` — and read the training log
under `.../logs/slurm/training/<jobname>-<id>.err`.

### 4. The 1h SIGUSR2 grace window
`#SBATCH --signal=SIGUSR2@3600` + `--exit-signal-handler` checkpoint-and-exit
before walltime. `launch_trainings.py::auto_time()` adds a 2h30m margin
(grace + cold-start + buffer), rounds up to 15 min, caps at 11:59:59.

### 5. `OptimizerParamScheduler` train_iters mismatch on capped resumes
Megatron asserts the CLI schedule total equals the checkpoint's. When a
resume is submitted with a reduced `--train-iters` (mid-gap backfill), the
assertion fires. `megatron_args.sh` pins
`--use-checkpoint-opt_param-scheduler`: the saved schedule wins, the loop
still exits at the CLI iters, and the LR trajectory stays exactly on the
original curve. **Don't switch to `--override-opt_param-scheduler`** — that
recomputes the schedule against the reduced iters and puts the run deep into
WSD decay at the wrong step. Verified end-to-end 2026-05-10.

### 6. Platform parity beyond the arguments
- Azure checks out Megatron at the pinned `MEGATRON_COMMIT`
  (`azure/get_megatron.sh`) and copies `patches/` over it — the same
  dist-checkpointing patch the CSCS checkout carries (README "Before the
  first CSCS run"); the CSCS wrapper uses the on-disk checkout at
  `/iopsstor/.../data-mix-small/Megatron-LM`. Identical args don't guarantee
  identical code — verify the cluster checkout is at the same commit
  (`c92402e`) before cross-platform comparisons.
- CSCS compute nodes have no internet: the tokenizer
  (`swiss-ai/Apertus-70B-2509`) must be pre-downloaded into the HF cache on
  the login node (the old sweep's `alehc/swissai-tokenizer` was already
  cached; the new one is not).
- The shallow ladder has no `nodes`/cluster-valid MBS in its hyperparams
  file — `launch_trainings.py` resolves both at submit time
  (`NODES_BY_SIZE` fallback + `cscs_mbs`, the largest memory-safe
  micro-batch that divides the layout). Don't submit shallow cells by hand
  with the raw JSON values: 4 of 6 would fail Megatron's
  GBS % (DP x MBS) == 0 assertion.

### 7. Azure-specific
- Data inputs must stay `mode: download` — `.bin` memmaps over a blob mount
  are pathologically slow.
- Each cell's outputs are pinned to `predictivity/runs/<cell>/`; reusing
  another cell's checkpoint dir makes `TRAINING_STEPS` and the saved
  schedule disagree (see #5).
- `azure/jobs/convert.yml` pins `transformers==4.57.6` inside its own
  container; don't "fix" the version elsewhere.
- The MBS auto-shrink in `launch_pretraining_azure.sh` keeps
  `GBS % (NPROC × MBS) == 0`; the training math (GBS 504 × seq 4096) is
  identical to the cluster — only gradient accumulation differs.

### 8. The capstor dataloader stall (2026-08-20)
**Train off `/iopsstor`, never off the `/capstor` master copy.** Megatron
memmaps the `.bin` token files and, because samples are shuffled, reads
effectively *random* windows out of them — capstor's worst case. It is
bandwidth-optimised shared storage, not IOPS. This is the same failure the
Azure note in #7 describes; the cluster is not exempt.

Measured on byte-identical copies of the same file, 112 KB random reads
(= MBS 7 × seq 4096 tokens), both stores probed alternately with the same
offsets so cluster load hits both arms equally:

| store | median | p99 | max | MB/s |
| ----- | -----: | --: | --: | ---: |
| capstor  | 13.5 ms | 166 ms | 433 ms | 8 |
| iopsstor | 0.5 ms  | 5 ms   | 13 ms  | 235 |

~28× on the median, up to 200× on the tail — **single process**, before the
contention of 12–84 ranks × 4 dataloader workers all seeking at once.

**How it presented (and why it fooled us for a day):** iterations swung
wildly — deep-175M ran `14890 → 976 → 4424 → 832 ms` inside one job. Averaging
the last 20 iterations made some configs look uniformly slow, and the two
worst offenders (deep-175M, shallow-90M) happened to share `hidden=1024,
ffn=4096`, so it read as a power-of-2 GEMM-aliasing effect. It was not:

- A standalone GEMM benchmark at every ladder shape came out **flat**
  (553–633 TFLOP/s); the pow2 shapes were fine and *padded* controls
  (ffn 4224) were slightly **slower**. Hypothesis dead.
- The giveaway was the **distribution, not the mean**: the "slow" configs hit
  **838 ms at p10** — as fast as the healthy ones — then blew out to 3000–6000.
  A fixed geometry cannot do that. Wide spread ⇒ stall, not arithmetic.

**Rules that follow.** Diagnose ms/iter with median + p10/p90 over a
*mid-run* window; never a trailing average (the tail catches async-checkpoint
saves and end-of-run flushes — it inflated shallow-350M from 507 to 1487 ms).
A tight distribution means compute-bound and the number is trustworthy; a wide
one means you are measuring the filesystem. `ITER_MS` fitted during the
capstor period is contaminated — re-measure from a clean iopsstor run.

**The layout:** capstor is the durable master (iopsstor scratch is purged
~30 days) and `data/launch_builds.sh` writes there; the training copy is
staged on iopsstor and `CSCS_DEFAULT_DATA_DIR` points at it. After a purge,
re-stage from capstor before launching (README "Before the first CSCS run").
Checkpoints were always on iopsstor and stay there — same reasoning.

### 9. AML expands `${{...}}` only in `command` (2026-08-26)
`${{inputs.*}}` / `${{outputs.*}}` are substituted **only** inside a job
yml's `command`. In `environment_variables` — and in any
`--set environment_variables.X=...` the launcher writes — they pass through
verbatim. The job then mounts its outputs correctly and writes every
artifact to a local directory literally named `${{outputs.checkpoints}}`,
which vanishes with the node. **Nothing errors**: the job succeeds, the
mount exists, and blob storage holds 0 bytes.

Bind the paths in `command` and export them into the wrapper
(`CKPT_DIR=${{outputs.checkpoints}} ... bash launch_pretraining_azure.sh`).
`DATA_BLEND` has the same constraint, which is why `launch_trainings.py`
emits `$ENGLISH_DIR/$FINEWEB_DIR` and the wrapper eval-expands it — bash
does not re-expand variables found inside a variable's value.

The check that catches it: after any job that should write, `az storage blob
list --prefix <outputs path> --query "sum([].properties.contentLength)"`.
A 0 there is the bug; the job status will not tell you.

### 10. Megatron touches `$trigger_path/exit` when training finishes
Its default `trigger_path` is `/dev/null`, so the final touch raises
`NotADirectoryError: '/dev/null/exit'` **after** the run has trained and
checkpointed successfully — turning a good run into a `Failed` job. CSCS
never sees it because `TRIGGER_PATH` is a real directory there;
`launch_pretraining_azure.sh` now sets one on Azure too (which also enables
`--exit-signal-handler`, harmless where nothing sends SIGUSR2, and gives
Azure the same manual `touch $TRIGGER_PATH/{save,exit}` controls).

### 11. Azure GPU access has four gates, and three signals lie
Full table in [azure/README.md](azure/README.md) ("The four gates"). The
short version: allow-list → tier policy → quota → capacity, each with a
different remedy. **H100 cannot use low-priority on this subscription at
all** (`UnsupportedVMSizeForLowPriority`); A100 can. A *dedicated* cluster
is quota-checked at create; a *low-priority* one is not, and creates as
`Succeeded` regardless — so a cluster existing proves nothing. Never trust
`LowPriorityCapable`, the retail meter list, or the AML quota counters; the
only reliable test is creating the cluster at `min_instances: 0` (costs $0)
and reading `properties.errors` via `az rest`.

Corollary for planning: Azure jobs are **single-node**
(`torchrun --standalone`), so nodes buy concurrency, not per-run speed, and
GPUs-per-node is the binding constraint. A 1.7B run is 7.2 d on one 8×H100
node but 29 d on a 2×H100 node — and since a run cannot span nodes, no
quantity of small nodes fixes that.

---

## Live state (read, don't trust snapshots)

```bash
# CSCS: per-cell status + idempotent (re-)launch
python3.11 pretrain_progress.py            # what a re-launch would do; --plot for heatmaps
python launch_trainings.py cscs --dry-run  # first
python launch_trainings.py cscs            # then for real (skips done/active, resumes partial)

# Azure: watcher submits convert+eval as checkpoints land (one per workspace)
source azure/env.sh && python auto_evals_azure.py --watch 600
python auto_evals_azure.py --workspace ca --watch 600   # 1B/1.7B cells
```

---

## The finished 36-model sweep

4 sizes × 3 mixes × 3 seeds to iter 50000, checkpoints under
`.../Meg-Runs/data-mix-small/`, evaluated via `../evals/`. Its tooling
evolved in place into the predictivity scripts (see git history for the
sweep-era versions). The per-size cost table in README.md and failure modes
above come from that sweep's 1.26M logged iterations.
