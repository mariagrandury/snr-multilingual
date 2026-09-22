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
| `launch_pretraining_cscs.sh` | SBATCH header, Meg-Runs dirs, SIGUSR2 trigger, `MEGATRON_EXIT_ON_SIGTERM` on the srun line so preemption checkpoints and the singleton self-chain that brings the run back (#4), srun+pyxis, debug log; falls back to `container/ngc_nemo_iopsstor.toml` when capstor is unreadable (`CONTAINER_TOML` overrides) |
| `launch_pretraining_azure.sh` | `azure/get_megatron.sh` checkout, MBS auto-shrink to GPU count, torchrun |
| `launch_trainings.py` | **the grid** (`LADDER`, `LANG_SETTINGS`, `DATA_SCHEMES`, `SEED_TRIPLES` — every other tool imports them from here) + filters + both submit backends; **idempotent** — per cell it skips done/active, warns on corrupt, resumes partial (marker rewind + auto-sized walltime). There is no separate resume script. |
| `pretrain_progress.py` | CSCS per-cell actions (`done/fresh/resume/corrupt` — the same `cell_action` the launcher uses) + `--is-valid` CLI + the plan table and three heatmaps (`--plot`): planned runs, finished models, and eval work outstanding. `--plot` also rewrites the generated grid block in README.md and the plan doc, so the figures and counts cannot drift from the constants in `launch_trainings.py`. `--plot`, every launch and every watcher pass also write `pretrain_progress_1b_17b.md` (per 1B/1.7B run of any account: data read, checkpoint, job, jobs left; then the launch commands). |
| `auto_evals_cscs.py` | CSCS watcher: per due ckpt (every 2nd of the size's grid, on the run's own save grid — `due_iters` — plus the k/20 points of the noise window, 85 % and 95 %, which `every` alone misses at every size but the 40-save one, and + final; the FLOPs-milestone add-on decided 09-02 is not implemented yet) submits convert-snr then evaluate.sbatch (one eval worker per GPU, results per task, so a killed job resumes on the next pass with only the missing tasks; `--max-attempts` holds back tasks that keep failing, `--retry-held` frees them for one pass after you fix the cause — the reason shown is the task's own, or its job log's only when that log belongs to the same run and family (../evals/CLAUDE.md "A second opinion from the wrong job's log"); `--all-languages` swaps a cell's trained languages for every language any task is tagged with; `--reformulated [rf|rfgm]` evaluates the `auto_rf` group — belebele / global_mmlu_full / include_base_44 as cloze `rf_*` tasks — or `auto_rfgm`, the Gemini-rewritten `rfgm_*` twins (`-rfgm` job names), see ../evals/CLAUDE.md "Reformulated twins"; `--size` filters, default `EVAL_SIZES` = the ladder without 90M — the diverged rung is trained but never evaluated (2026-09-18), and `eval_progress` leaves it out of its grid); needs models.json entries (`sync_models_json.py`) |
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

**L=100 is not trained** (planned as AT3 only, dropped 2026-09-20 —
`plan/l100_data_mixture.md`): at T=1, measured on the filtered subset the
builds read, the median of the 99 languages gets 90M tokens and the smallest
3.5M; flattening lifts that to 373M / 14.5M but the tail is data-limited, so
the T=3 build realizes 75.4B, under the 83.6B a 1.7B draws (1.11 epochs).
On the L50 pair, tripling a tail language's tokens moved its share of
above-chance tasks by ~4 points, so neither temperature makes the L100 tail
measurable on benchmarks. The ladder ends at L50. L50 is built both ways so
the temperature change is calibrated against the T=1 curve, and AT3 adds
L15 and L30 (deep only, 1B and 1.7B launched first) where no language is
starved (T=1 floors 1.2B and 343M tokens). AT3 runs the whole ladder at L50: a 92B L50 build at
T=3 realizes 87.1B on the filtered subset, enough for the 83.6B a 1.7B draws
(0.96 epochs). ES stops at the 1B rung: Spanish would repeat 2.0x at 1B and
3.6x at 1.7B, a swing across the ladder that would confound a rank flip with
the repetition, so its reference is 1B. ZH runs to the 1.7B reference
(2026-09-20): that third family is what takes L2 at 1.7B from one DA pair to
three, the minimum rule 5 accepts (`signal-and-noise/analysis/RULES.md`).
**Its build holds 52.0B, not the 59.9B of Chinese there is** — it was sized
when ZH stopped at 1B — so `undersized_build` refuses the cell and it is
launched with `--allow-undersized lm-1.7B-L2-ZH-deep-seed1904` (the flag
names the single cell it applies to, never a blanket opt-out), repeating
1.61x against scheme A's own
1.15x. That is deliberate and is the better of the two options: no rebuild
root holds ZH, so every other ZH rung reads that same 52.0B file, and
rebuilding at 59.9B for the top rung alone would put it on data the rest of
its own ladder never saw — the hazard `fineweb_source` documents. ZH and ES
are one axis, the second language at L=2 (English + Russian / Chinese /
Spanish), deep only.

Seed triples are **per size** (`SEED_TRIPLES`): 175M and 600M run
(64, 313, 1904) at L ∈ {1, 2, 50}, 1B runs (28, 1797, 1904) at
L ∈ {1, 2, 30} — the 1B column is aromanou's already-trained runs (L50 was
added 2026-09-10 to match the other ×3 columns and dropped again 2026-09-22,
never having been launched), and naming the wrong triple would submit two
more runs per cell while the watcher ignored the ones on disk. Replicates are
**deep only**: no analysis reads a shallow seed std. **Her 1B runs follow the old 20-checkpoint regime**
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
checkpoint for checkpoint; the final save is due whatever its iter. Since
2026-09-20 the k/20 points inside the noise window (the last 20 %: 85 % and
95 %) are due at every size as well, so the analysis reads checkpoint noise
over five points instead of three — two extra evals per run on the 20-save
and 60-save sizes, none on the 40-save 1B. models.json
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

**Stock Megatron listens for SIGUSR2 only** —
`DistributedSignalHandler(sig=signal.SIGUSR2)` — while Slurm preempts with
SIGTERM, so a preemption dropped everything back to the last save.

**A batch-shell trap cannot fix that** (tried 2026-09-20, wrong, corrected
09-21). `launch_pretraining_cscs.sh` trapped TERM and re-emitted USR2 to the
step; it never once reached a training loop. Slurm signals the step's *tasks*
directly, so the ranks — which have no SIGTERM handler — are gone before a
shell trap runs. In the logs the trap only ever fired while a checkpoint was
still loading, and a 1.7B lost 334 iterations x 21 nodes to a preemption at
19:43:50 that shows `srun exited 143` with no save. The trap is still there,
for its log line, but it is not the mechanism.

**The mechanism is a patched handler** (`patches/training_dist_signal_handler.py`,
copied into the shared checkout and onto Azure by `azure/get_megatron.sh`):
with `MEGATRON_EXIT_ON_SIGTERM=1` in the rank's environment,
`DistributedSignalHandler` catches SIGTERM as well as SIGUSR2, so the rank
that owns the state handles its own signal and the existing
`--exit-signal-handler` path saves and exits. `launch_trainings.py
--partition preemptable` exports `EXIT_ON_SIGTERM=1`, which
`launch_pretraining_cscs.sh` puts on the srun line beside `RANK`/`LOCAL_RANK`
(host env does not cross into the container — evals CLAUDE #5). It is opt-in
because it also makes `scancel` save before stopping, and `KillWait` is 30 s
against 240 s of preemption grace, so a killed job may not get its async save
down. Rank 0 prints `[exit-signal-handler] catching SIGUSR2, SIGTERM` —
**grep for that line before trusting a preemption to checkpoint**, since an
unpatched checkout ignores the variable silently. The save is off the
checkpoint grid, and that is already expected: `run_interval()` takes the
modal gap and `due_iters()` never marks an off-grid save due, so preemption
saves add disk, not eval work.

**Coming back is a chain, not a requeue** (2026-09-21). `--requeue` was the
first answer and it does not survive a busy day: Clariden sets
`MaxBatchRequeue=5`, a preemption spends one, and on the sixth Slurm holds the
job — reporting `launch_failure_limit_exceeded_requeued_held` even when all
five attempts ended in clean preemptions. Two 3B and one 1B run stopped that
way after 4h43m at 96.7% node-time efficiency; nothing had failed to launch.
`scontrol update Restarts=` is refused, and `MaxBatchRequeue` is cluster-wide
and admin-only.

So `--partition preemptable` sets `PRETRAIN_CHAIN=1` instead, and the wrapper
queues a singleton successor (`--dependency=singleton`, same job name) BEFORE
it starts training — exactly what `data/submit_build_one.sh` has always done,
and what carried `build-a-L8-165b` to completion across 29 links and 36 hours
(24 PREEMPTED, 3 FAILED, 1 CANCELLED). Each link is a new jobid, so no requeue
cap applies, and queuing it up front is what makes it preemption-safe: a killed
attempt leaves its successor already pending, holding a queue position a
resubmit would forfeit.

**What stops a chain**, in the order the wrapper checks it:

- `done` — `pretrain_progress.py --cell-action` on entry or after training;
  the pending successor is cancelled. The check reads the printed WORD, not an
  exit status: an ImportError and a legitimate "not done" both exit non-zero.
  State unreadable → do NOT chain, and say so.
- `corrupt` — iter dirs on disk, none loadable. `launch_trainings.py` refuses
  these for manual review; the chain must too, or it re-runs on 21 nodes
  exactly what the launcher will not touch.
- **no progress four times running** (`CHAIN_MAX_STALLS`). Each link records
  the iteration it started from in `.chain-<jobname>.progress` beside its
  Slurm logs; unchanged means the previous link achieved nothing. Progress
  resets the counter, so ordinary preemptions do not cap a chain — though a
  run preempted before its first save does count as a stall, which is the
  intended conservatism at 21 nodes a link. This is the
  real bound — 2026-09-22, `lm-1B-L2-shallow-seed1904` burned 14 links x 21
  nodes because its `logging/` and `debug/` belonged to a collaborator: rank 83
  (Megatron's tensorboard writer is the LAST rank) died of `PermissionError`
  ~70 s in, Slurm killed the step for TASK FAILURE, and the wrapper still
  exited 0, so nothing downstream could tell a spin from a preemption. The
  counter file is per-user for the same reason the incident happened: these
  trees are shared, and a file a collaborator creates is one this user cannot
  rewrite. A counter that cannot be recorded also stops the chain.
- attempt count (`CHAIN_MAX_ATTEMPTS`, default 200) — only a backstop now, so
  it is generous: a 3-day 3B at ~1 preemption/hour needs 30-70 links, and 29
  were needed for a one-node build. Counted in the directory the controller
  says this job's `StdOut` is in, never a hard-coded path — a collaborator's
  logs land in their own scratch, and counting where there are none reads 0
  forever and never caps. Uncountable → do NOT chain.

A chained job advertises itself with `--comment=selfchain`, because
`PRETRAIN_CHAIN` lives in the job environment and neither `squeue` nor
`scontrol` exposes that — `scripts/preempt_drain.sh` has to know before it
moves a 21-node run onto a partition that will preempt it.

Two constraints this works around, both verified: a job's time limit can only
be LOWERED after submission (so no drainer can stretch a job it moves — ask for
24h at submit, and a successor inherits the wall of the attempt that queued
it), and `normal` refuses a 23:59:00 request outright ("Requested time limit is
invalid"), so the long wall and the partition have to travel together.
`#SBATCH --open-mode=append` is now belt-and-braces: a requeue reopened the
same `%x-%j` log, and a chain link never does.

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

### 12. `preemptable` packs jobs; `normal` and `debug` do not
`normal` and `debug` are `OverSubscribe=EXCLUSIVE`, so every job there gets a
whole node whatever it asks for — a data build requesting 32 CPUs is recorded
`AllocCPUS=288`, and `submit_build_one.sh`'s "~9 builds pack per node" has
never actually happened. `preemptable` is `OverSubscribe=FORCE:1`, which does
pack: on 2026-09-20 six builds moved there landed on ONE node that already ran
another user's job and were cancelled by uid 0 sixteen seconds after starting,
before writing a line of output — so none reached `submit_build_one.sh`'s
self-chain line and six chains died silently, with no job left in the queue to
notice. `launch_builds.sh` now passes `--exclusive` always (reproducing the
allocation every finished build has had) and queues `BUILD_SEGMENTS` segments
up front, 2 on `preemptable`, so a kill before the script runs cannot end a
chain. `scontrol` cannot add `--exclusive` to a queued job, so builds are
submitted to `preemptable`, never moved there — `scripts/preempt_drain.sh`
skips `build-*` for exactly that reason.

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
