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
| `launch_pretraining_cscs.sh` | SBATCH header, Meg-Runs dirs, SIGUSR2 trigger, srun+pyxis, debug log; falls back to `container/ngc_nemo_iopsstor.toml` when capstor is unreadable (`CONTAINER_TOML` overrides) |
| `launch_pretraining_azure.sh` | `azure/get_megatron.sh` checkout, MBS auto-shrink to GPU count, torchrun |
| `launch_trainings.py` | **the grid** (`LADDER`, `LANG_SETTINGS`, `DATA_SCHEMES`, `SEED_TRIPLES` — every other tool imports them from here) + filters + both submit backends; **idempotent** — per cell it skips done/active, warns on corrupt, resumes partial (marker rewind + auto-sized walltime). There is no separate resume script. |
| `pretrain_progress.py` | CSCS per-cell actions (`done/fresh/resume/corrupt` — the same `cell_action` the launcher uses) + `--is-valid` CLI + the plan table and three heatmaps (`--plot`): planned runs, finished models, and eval work outstanding. `--plot` also rewrites the generated grid block in README.md and the plan doc, so the figures and counts cannot drift from the constants in `launch_trainings.py`. `--plot`, every launch and every watcher pass also write `pretrain_progress_1b_17b.md` (per 1B/1.7B run of any account: data read, checkpoint, job, jobs left; then the launch commands). |
| `auto_evals_cscs.py` | CSCS watcher: per due ckpt (every 2nd of the size's grid, on the run's own save grid — `due_iters` — + final; the FLOPs-milestone add-on decided 09-02 is not implemented yet) submits convert-snr then evaluate.sbatch (one eval worker per GPU, results per task, so a killed job resumes on the next pass with only the missing tasks; `--max-attempts` holds back tasks that keep failing, `--retry-held` frees them for one pass after you fix the cause — the reason shown is the task's own, or its job log's only when that log belongs to the same run and family (../evals/CLAUDE.md "A second opinion from the wrong job's log"); `--all-languages` swaps a cell's trained languages for every language any task is tagged with; `--reformulated` evaluates the `auto_rf` group — belebele / global_mmlu_full / include_base_44 as cloze `rf_*` tasks, see ../evals/CLAUDE.md "Reformulated twins"; `--size` filters, default `EVAL_SIZES` = the ladder without 90M — the diverged rung is trained but never evaluated (2026-09-18), and `eval_progress` leaves it out of its grid); needs models.json entries (`sync_models_json.py`) |
| `sync_models_json.py` | upserts one models.json entry per grid cell — conversion + W&B push resolve through it |
|  `auto_evals_azure.py` | Azure watcher: same due rule against blob storage |

A rung is trained in an architecture only if that architecture's hyperparams
file defines it (`launch_trainings.arches_for`): the 3B exists in
`hyperparams_deep.json` only, so every fan-out over architectures reads
`arches_for(scheme, size)`, never a scheme's `arches` list directly.

Cell name everywhere (checkpoint dir, W&B run id/name, models.json key,
parsed by `pretrain_progress.py`):
`lm-<size>-L<L>[-AT3|-schemeB|-ZH|-ES]-<deep|shallow>-seed<seed>` — `lm`, not `apertus`:
the architecture has diverged from Apertus (renamed 2026-08-21). Job display
names drop the `lm-` for a kind prefix instead
(`launch_trainings.job_name`): `pretrain-90M-L8-deep-seed1904`,
`eval-90M-L8-deep-seed1904-iter425`, `convert-...`. CSCS checkpoints:
`/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/msnr/<cell>/checkpoints/`.
Azure: `predictivity/runs/<cell>/checkpoints` in each workspace's blob store.

**The data axis is `DATA_SCHEMES`** (2026-09-10, five entries, `--scheme` on
`launch_trainings.py`, `pretrain_progress.py`, `sync_models_json.py`,
`auto_evals_cscs.py` and `data/build_data_mixtures.py`): **A** resource-ranked
at T=1 — the baseline, and the only one with no name label; **AT3** the same
language lists at T=3; **B** diversity-first at L ∈ {8, 15, 30}; **ZH**/**ES**
L2 with Chinese / Spanish instead of Russian, deep only. Each entry owns its
label, its data subdir, the settings it defines and its per-setting size cap,
so a cell's scheme *is* its data and two schemes can never collide in a
checkpoint dir, a W&B run id or a models.json key (which keeps the key
`scheme`, now holding the scheme name, plus a `temperature` field).

**L=100 exists ONLY flattened (AT3)**: at T=1, measured on the filtered
subset the builds read, the median of the 99 languages gets 90M tokens and
the smallest 3.5M, so most per-language BPB would measure a language the
model never saw; flattening lifts the median to 373M (floor 14.5M —
data-limited, identical at T=2 and T=3). The plan's re-measured table
recommends T=2: at T=3 the L100 build realizes 75.4B, under the 83.6B a 1.7B
draws. The registry still codes T=3 and the name label / subdir spell it, so
a switch renames the scheme — free only while nothing is built or trained as
AT3. L50 is built both ways so the temperature change is calibrated against
the T=1 curve. AT3 runs the whole ladder at both settings: a 92B L50 build at
T=3 realizes 87.1B on the filtered subset, enough for the 83.6B a 1.7B draws
(0.96 epochs). ZH/ES stop at the 1B rung — no L2 source can feed a 1.7B — so
their reference is 1B.

Seed triples are **per size** (`SEED_TRIPLES`): 175M and 600M run
(64, 313, 1904) at L ∈ {1, 2, 50}, 1B runs (28, 1797, 1904) at
L ∈ {1, 2, 30, 50} — the 1B column is aromanou's already-trained runs (L50
added 2026-09-10 to match the other ×3 columns; those two cells are new), and
naming the wrong triple would submit two more runs per cell while the watcher
ignored the ones on disk. **Her 1B runs follow the old 20-checkpoint regime**
(every 2287 iters to 45740, a 45,740-iter schedule — L1, L2, L15, L30 and
schemeB L8/L15/L30 at seed 1904, plus 28/1797 at L1, L2, L30 and schemeB L30;
15 cells), while `n_checkpoints` gives the 1B rung 40 saves
every 1143 iters to 45720 (60 at 1.7B, unchanged). None of their saves lands
on that grid, so until 2026-09-10 the watcher reported `eval DONE (0/0)` for
all of them — nothing ever fell due — and the launcher counts them done
(45740 ≥ 45720). **Due checkpoints are now read on the run's own grid**
(`launch_trainings.due_iters`, used by both watchers, `eval_counts` and the
ladder report): `run_interval()` takes the modal gap between saves, and every
Nth point of the SIZE's grid is mapped onto it, so a 20-save 1B run yields
every save and a 40-save one every 2nd — the same k/20 fractions, comparable
checkpoint for checkpoint; the final save is due whatever its iter. models.json
still lists the size grid under `checkpoints.all`, which is wrong for her
2287-spaced cells: the watcher is unaffected (it passes `--iters`), but
`convert-snr.sh --models` without `--iters`, `snr_progress.py` and
`azure/launch_evals.py --ckpts final` plan iters those runs never saved — pass
the iters explicitly for them. The 1.7B row now trains at every setting, so no (size, L) cell of scheme A is
empty. Terminology: **scheme** is this data axis; **variant** keeps its older,
looser sense (any run configuration — seed × arch × scheme).

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
- **Never rebuild a data mixture in place.** Build token targets are derived
  from the grid (`data/build_data_mixtures.py` imports `DATA_SCHEMES` /
  `scheme_sizes`): 92B where a 1.7B trains, 52B where the largest rung is 1B.
  When the 1.7B row gained L15 and L50, the finished 52B builds for A L15/L50
  and B L15 became undersized — but cells have already trained on them, so
  they are rebuilt at 92B into a **parallel root** (`data/launch_builds.sh`,
  the `REBUILD` array), never overwritten, and staged to
  `/iopsstor/scratch/cscs/mariagrandury/data-92B`. The launcher reads a cell's
  FineWeb-2 half from there only when the stage copy is too small for it (the
  six 1.7B cells at A-L15/A-L50/B-L15); every other rung stays on 52B. The 3B
  rung (2026-09-19, A/B at L8/L15, deep only — `plan/3b_models.md`) repeats
  the pattern as a second tier: those four builds are sized 165B, built into
  `rebuild-165B`, staged to `data-165B`, and read only by cells the 92B copies
  cannot feed (`CSCS_REBUILD_DATA_DIRS`, smallest fit first). **Do not
  swap the 92B files into the training stage.** Each language section is a
  byte-exact extension of the 52B one, but Megatron shuffles over the whole
  file (a different sample order) and the extra documents are newer crawls
  (Russian 2021–24 share 4% → 14%, Chinese 0% → 32%), so a shallow or new-seed
  cell moved onto it would stop seeing what its trained counterparts saw
  (verified 2026-09-13). The launcher also refuses to resume a run saved on
  another checkpoint grid (`skip [foreign schedule]`) — but only when at least
  three of its saves sit on the inferred interval, because `run_interval`
  breaks ties toward the larger gap and would otherwise read a bogus grid off
  two or three saves and park the cell for good.
- **Never change a grid cell's training config.** #5 covers not changing the
  optimizer *schedule* on a resume; this is the wider rule, across cells: dozens of
  cells are trained, and a rung that ran different hyperparameters is not on
  the same ladder as the rest — the scaling fit cannot absorb it, so "fixing"
  one rung means re-running every rung. `launch_trainings.py` has exactly three
  config-perturbing flags, `--lr`, `--ademamix-beta3-factor` and `--gbs`; all
  are opt-in, all require a `--size/--langs/--seed` filter, and all **force a
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
under `.../logs/slurm/training/<jobname>-<id>.err`. The 2026-09-16 capstor
outage is the worst shape of this: with the container toml unreadable, pyxis
failed before any rank started and 10 job-wide relaunches were recorded
`COMPLETED 0:0` after ~3 min, having trained nothing. An elapsed time far
under the walltime with no new checkpoint is the tell.

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
A scheme subdir on the stage also needs its `english_dclm.*` symlinks —
`submit_build_one.sh` stages them with the mixture, and the launcher skips a
cell whose blend files are missing (2026-09-11: six ZH/ES trainings died at
dataset build because neither existed). Also: Slurm runs the script copy
taken at SUBMIT time, so a job queued before a rename in
`launch_trainings.py` fails on the old name (the ZH/ES builds died on
`DATA_VARIANTS`); resubmit rather than wait for it.
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
