# Predictivity-sweep pretraining (CSCS + Azure)

> Pretraining infrastructure for the small-to-large predictivity sweep: a
> 6-rung size ladder (90M–1.7B non-embedding) × 7 language settings, fixed
> 50/50 English/multilingual data, each size trained to its own
> 5×Chinchilla budget. Runs split across the CSCS cluster and Azure ML —
> **both platforms execute the exact same training logic.**
> Full design: [`../../plan/small-to-large-predictivity-training-plan.md`](../../plan/small-to-large-predictivity-training-plan.md).

## TL;DR — pretrain a model

**On CSCS** (data on `/capstor`, jobs via sbatch; run from the login node):

```bash
cd src/pretrain

# 0. First time only (see "Before the first CSCS run" below):
hf download swiss-ai/Apertus-70B-2509      # tokenizer into the HF cache

# 1. Build the data mixtures (once; idempotent self-chaining Slurm jobs)
cd data && ./launch_builds.sh --dry-run && ./launch_builds.sh && cd ..

# 2. Launch — the launcher is IDEMPOTENT: done cells are skipped, active
#    cells are skipped, partial cells are resumed with auto-sized walltime.
#    Re-run the same command any time to drive the sweep to completion.
python3.11 launch_trainings.py cscs --dry-run      # always preview first
python3.11 launch_trainings.py cscs                # whole sweep (or filter)
python3.11 launch_trainings.py cscs --size 90M --langs 2   # one cell

# 3. Auto-evals while training (tmux; converts + evals every 2nd checkpoint
#    + each run's final one, pushes to W&B mariagrandury-epflnlp/msnr;
#    the W&B key comes from your env or src/evals/scripts/wandb_api_key.txt)
python3.11 auto_evals_cscs.py --watch 600

# 4. Monitor
python3.11 pretrain_progress.py                    # per-cell status lines
python3.11 pretrain_progress.py --plot             # + the two heatmaps
```

**On Azure** (one-time setup in [`azure/README.md`](azure/README.md) §1–§4,
data shipped per its §5):

```bash
cd src/pretrain
source azure/env.sh                                # Azure names (edit once)

bash azure/setup.sh                                # once per workspace
az ml job create --file azure/jobs/smoke.yml $AZ_ML_ARGS   # once: smoke test

python launch_trainings.py azure --dry-run         # same launcher, same flags
python launch_trainings.py azure --langs 2 --size 90M     # first real cell
python launch_trainings.py azure                   # the rest

python auto_evals_azure.py --watch 600                   # ES watcher (<=600M cells)
python auto_evals_azure.py --workspace uk --watch 600    # UK watcher (1B/1.7B cells)
```

There is **no separate resume script**: `--save`/`--load` point at the same
checkpoint dir, so a resubmitted cell always continues from its latest valid
checkpoint. On CSCS the launcher additionally checks the disk before
submitting (skip done/active/corrupt, rewind a stale marker, size the
walltime to the remaining iters); on Azure resubmitting is the resume.

## The sweep

<!-- BEGIN generated: pretrain_progress.py --plot -->
| Axis | Values |
| ---- | ------ |
| Size (non-embedding) | 90M, 175M, 350M, 600M, 1B, 1.7B (1.7B at L ∈ {1, 2, 8, 30, 100}) |
| Language setting L | 1, 2, 8, 15, 30, 50, 100 (English + L−1 FineWeb-2 languages; L=1 is 100% English) |
| Seed | 1904; ×3 seeds (28, 1797, 1904) on the 175M, 1B columns at L ∈ {1, 2, 30, 100} |
| Data scheme | A everywhere; B only where its language set differs — L ∈ {8, 15, 30} |
| Architecture | deep (baseline) and shallow (the model-depth intervention) |

**56 runs** at one intervention level (scheme A, deep — the plan grid).
Counting both architectures and scheme B where it differs: **154 runs**.

![Planned runs per grid cell](./pretrain_progress_plan.png)

![Finished models per grid cell](./pretrain_progress_simple.png)
<!-- END generated -->

Variants multiply the
grid and are suffix-marked in the run name: `--arch shallow` (width/depth
128, the model-depth intervention)
and `--scheme B` (diversity-first language sets — B differs from A only at
L ∈ {8, 15, 30}, derived from `data/language_sets_scheme{A,B}.json`; at every
other setting a `--scheme B` sweep runs the scheme-A cell, deduped by the
idempotency check). Each size trains D(N) = 100 × N tokens (5×C); the
per-size schedule lives in the `predictivity` block of the hyperparams files.

Run name = Slurm job name = Azure display name = checkpoint dir = W&B run
name: `lm-<size>-L<L>[-schemeB]-<deep|shallow>-seed<seed>`. Runs log to
W&B under `mariagrandury-epflnlp/msnr` — the entity is a hardcoded constant
(`megatron_args.sh`) and the project comes from
[`configs/hf_wandb.json`](../../configs/hf_wandb.json) (`wandb.project`).
Each cell is **one continuous W&B run across resumes** (deterministic run
id + `resume=allow` in `megatron_args.sh`), so multi-window trainings don't
fragment into one run per job.

Two schedule knobs are derived per cell by the launcher so the optimizer
behaves identically at every ladder size: the AdEMAMix alpha/beta3 warmup
spans the cell's full schedule (`ADEMAMIX_WARMUP` = target iters, replacing
the old fixed 100 000 that short runs never finished), and the init std is
width-scaled (`INIT_STD` = 0.008944 × √(1792/hidden), anchored so the 1B
keeps the reviewed value exactly).

## What's in this folder

**Layout rule:** the top level holds shared code and platform *pairs*
(`*_cscs` / `*_azure` side by side: the training wrappers, the auto-eval
watchers); [`azure/`](azure/) holds everything only Azure needs (infra,
job specs, in-job entrypoints); [`conversion/`](conversion/) is the CSCS
conversion toolbox; [`data/`](data/) and [`hyperparams/`](hyperparams/)
are pipeline stages.

**The training path** (one shared arguments file, two thin wrappers, one
launcher — the core design):

| File | Role |
| ---- | ---- |
| [`megatron_args.sh`](megatron_args.sh) | **The single source of the training logic.** Builds every Megatron argument (architecture, AdEMAMix, WSD schedule, torch_dist checkpointing, data blend, W&B) from env vars. Both platforms produce an identical command; the only delta is the SLURM graceful-exit trigger, added when `TRIGGER_PATH` is set. |
| [`launch_pretraining_cscs.sh`](launch_pretraining_cscs.sh) | CSCS wrapper: SBATCH header, directories under `Meg-Runs/msnr/`, SIGUSR2 trigger, srun + pyxis container, debug logging. |
| [`launch_pretraining_azure.sh`](launch_pretraining_azure.sh) | Azure wrapper: pinned Megatron checkout, GPU-count-aware micro-batch, torchrun. Run through `azure/jobs/pretrain.yml`. |
| [`launch_trainings.py`](launch_trainings.py) | The idempotent launcher for **both** platforms: enumerates the grid, decides skip/fresh/resume per cell, builds one env-var dict, submits via `sbatch --export` (cscs) or `az ml job create --set` (azure). |
| [`pretrain_progress.py`](pretrain_progress.py) | CSCS status: per-cell action lines (the same `cell_action` decision the launcher uses), the `--is-valid` checkpoint check (also used by `conversion/`), and the two progress heatmaps (`--plot`). |
| [`auto_evals_cscs.py`](auto_evals_cscs.py) | CSCS auto-eval watcher (twin of `auto_evals_azure.py`): per due checkpoint submits convert (`conversion/convert-snr.sh --models`) then eval (`../evals/` `evaluate.sbatch`), pushing to W&B msnr. Idempotent. |
| [`sync_models_json.py`](sync_models_json.py) | Upserts one `configs/models.json` entry per grid cell (paths + schedule) — the W&B push refuses cells without one. Both watchers run it automatically each pass; the CLI exists for explicit use. |
| [`auto_evals_azure.py`](auto_evals_azure.py) | Azure auto-eval watcher — same due rule against blob storage (`source azure/env.sh` first). |

**Subfolders:**

| Dir | Contents |
| --- | -------- |
| [`azure/`](azure/) | Everything only Azure needs (guide: [`azure/README.md`](azure/README.md)): [`env.sh`](azure/env.sh) (names — edit once, `source azure/env.sh` before any az command), [`setup.sh`](azure/setup.sh) (one-time workspace/compute setup, consumes the `compute-*.yml` / `environment-*.yml` specs), [`get_megatron.sh`](azure/get_megatron.sh) (pinned Megatron checkout), [`jobs/`](azure/jobs/) (AML job specs: `pretrain.yml`, `smoke.yml`, `convert.yml`, `eval.yml`), [`convert.sh`](azure/convert.sh) / [`eval.sh`](azure/eval.sh) (job entrypoints), [`launch_evals.py`](azure/launch_evals.py) (eval launcher). |
| [`data/`](data/) | Data-mixture pipeline: [`create_data_mixture.py`](data/create_data_mixture.py) (tokenize-and-blend worker), [`build_data_mixtures.py`](data/build_data_mixtures.py) (per-sweep driver), [`language_sets_scheme{A,B}.json`](data/language_sets_schemeA.json) (the nested language lists), [`launch_builds.sh`](data/launch_builds.sh) + [`submit_build_one.sh`](data/submit_build_one.sh) (one idempotent self-chaining Slurm job per mixture — L2 goes through the same path, sized for its 1.7B run). |
| [`hyperparams/`](hyperparams/) | The reviewed architecture ladders: [`hyperparams_deep.json`](hyperparams/hyperparams_deep.json) (baseline) / [`hyperparams_shallow.json`](hyperparams/hyperparams_shallow.json) (depth variant), each with the per-size `predictivity` schedule block; their generators and shared helpers. |
| [`conversion/`](conversion/) | CSCS Megatron → HF conversion ([`convert-snr.sh`](conversion/convert-snr.sh)) and HF-Hub push ([`push-snr.py`](conversion/push-snr.py)). |

Plus [`env.toml`](env.toml) (pyxis container env file, CSCS),
[`.amlignore`](.amlignore) (keeps the AML code snapshot small) and
[`CLAUDE.md`](CLAUDE.md) (back-of-house notes and failure modes).

## 1. Build the data mixtures (once, on CSCS)

[`data/build_data_mixtures.py`](data/build_data_mixtures.py) drives
[`data/create_data_mixture.py`](data/create_data_mixture.py) to build one fixed
validation set, one English (DCLM) dataset, and one FineWeb-2 dataset per
language setting; they are blended 50/50 at train time, so each is built once.
Validation must be built first (english/fineweb read its manifest to hold out
the same rows).

Launch **one self-chaining Slurm job per mixture** so they build in parallel and
each `.bin`/`.idx` is ready to start its training run independently:

```bash
cd data
./launch_builds.sh --dry-run   # print the sbatch commands, submit nothing
./launch_builds.sh
```

[`data/launch_builds.sh`](data/launch_builds.sh) fans out to
[`data/submit_build_one.sh`](data/submit_build_one.sh) (one mixture per job —
english, scheme A {2,8,15,30,50,100}, scheme B {8,15,30}). Each job caps
the tokenizer to ~32 cores — it peaks at ~16–32 threads, so ~9 builds pack per
node — and self-chains a `--dependency=singleton` successor to resume past the
12h wall. Builds are idempotent: a finished mixture (`.idx` present) is skipped,
a preempted one resumes from its checkpoint, so re-running is always safe.
Scheme B lands in `<DATA_DIR>/schemeB/` (its own `--data_dir`, with the shared
english build and validation manifest symlinked in).

Everything runs on the cluster's curated corpora (DCLM-edu + FineWeb-2-HQ on
`/capstor`) — nothing is downloaded from the HF Hub. To train on Azure, ship
the finished `.bin`/`.idx` builds with azcopy: see the Azure guide's §5
([`azure/README.md`](azure/README.md)).

`--scheme {A,B}` picks the language lists
(`data/language_sets_scheme{A,B}.json` — A is resource-ranked, B diversity-first).
Targets: 184.0 B English, 92.0 B FineWeb-2 where the 1.7B trains
(L ∈ {2, 8, 30, 100}) and 52 B
elsewhere (half the largest run's budget + 10% headroom).

## 2. Launch the trainings

[`launch_trainings.py`](launch_trainings.py) takes the platform as its first
argument and the same filters everywhere. CSCS submits against the data built
in §1 on `/capstor`; Azure needs the one-time setup and data upload from
[`azure/README.md`](azure/README.md) first (`source azure/env.sh` before
launching).

```bash
# Whole sweep — always dry-run first (shows the per-cell skip/fresh/resume
# decision without submitting anything):
python launch_trainings.py cscs --dry-run
python launch_trainings.py cscs

# Azure: same grid, same flags (placement by size — Spain <=600M, UK 1B/1.7B):
python launch_trainings.py azure --dry-run
```

**`--size` takes one size or a comma-separated list:**

```bash
python launch_trainings.py cscs --size 600M
python launch_trainings.py cscs --size 350M,175M

# "all sizes up to 1B" = every rung except the 1.7B top:
python launch_trainings.py cscs --size 90M,175M,350M,600M,1B
```

Variant axes and filters compose:

```bash
python launch_trainings.py cscs --arch shallow         # depth-intervention variant
python launch_trainings.py cscs --scheme B --langs 8   # scheme-B data variant
python launch_trainings.py cscs --size 600M --langs 8 --seed 1904
python launch_trainings.py azure --langs 1             # monolingual anchors
python launch_trainings.py cscs --test --dry-run       # smoke: 90M, L8, 50 steps
```

The training plan:

```bash
python launch_trainings.py cscs --size 175M --langs 1 --arch deep --seed 313
python launch_trainings.py cscs --size 90M,175M,350M,600M --langs 30 --arch deep --seed 1904
python launch_trainings.py cscs --size 90M,175M,350M,600M --langs 15 --seed 1904
```

CSCS-only knobs: `--data_dir`, `--time` (override the auto-sized walltime),
`--account`, `--dependency`, `--training-steps` (cap `--train-iters`
manually), `--test`.

**Before the first CSCS run** (once):

- Pre-warm the tokenizer into the HF cache — compute nodes have no
  internet: `hf download swiss-ai/Apertus-70B-2509` on the login node.
- Use the **`swiss-ai/Megatron-LM` fork** — it carries the Apertus kernels
  (xIELU, apex qk-norm); a vanilla `nvidia/Megatron-LM` clone will not run.
  Keep it at `/iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM` and check
  its checkout matches the commit Azure pins
  (`azure/get_megatron.sh::MEGATRON_COMMIT`), so both platforms run the same
  training code, not just the same arguments:
  `git -C /iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM rev-parse HEAD`.
- **Re-apply the legacy-checkpoint load patch after any fresh clone.** A scratch
  cleaning sweep can wipe the checkout, and a re-clone reverts the fix — then
  **every resume** dies in `get_reformulation_metadata` with `AttributeError:
  'Metadata' object has no attribute 'mcore_data'` (our checkpoints predate
  `mcore_data`; the patch synthesizes the reformulation metadata from the saved
  tensor sizes). Copy the tracked, patched file over the fork's:
  ```bash
  cp patches/dist_checkpointing_strategies_torch.py \
     /iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM/megatron/core/dist_checkpointing/strategies/torch.py
  ```
- **Train off the iopsstor copy of the data, not the capstor master.** Megatron
  memmaps the `.bin` files and reads them shuffled (random access): capstor is
  ~28× slower per read and up to 200× on the tail, which stalls training
  unpredictably (CLAUDE.md #8). `launch_builds.sh` writes the durable master to
  capstor; `--data_dir` defaults to the iopsstor stage. iopsstor is purged
  ~30 days, so after a purge re-stage before launching:
  ```bash
  cp /capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data/{english_dclm,fineweb_L*}.{bin,idx} \
     /iopsstor/scratch/cscs/$USER/data/
  ```
- **Pre-build the eval datasets into the iopsstor HF cache.** Compute nodes have
  no internet and the harness runs all tasks in one batched call, so a single
  uncached dataset aborts the whole eval. They live in `$HF_HOME/datasets` on
  iopsstor: the sweep touches them on every eval, and the cleaning policy is
  last-access based, so an active sweep keeps them alive. (After a long idle
  gap they can still be purged — the symptom is an empty directory tree and
  `FileNotFoundError: .../dataset_info.json`.) Re-run any time; it only
  fetches what's missing:
  ```bash
  python3.11 ../evals/scripts/download_eval_datasets.py
  ```

**Idempotency** (why re-running is always safe): per cell the launcher

1. skips it when its latest valid checkpoint has reached its target ("done");
2. skips it when a job with its name is already queued/running;
3. warns and skips `corrupt` cells (iter dirs on disk but none loadable) —
   nothing is ever auto-deleted;
4. otherwise submits fresh or resumes — rewinding a
   `latest_checkpointed_iteration.txt` that points at an invalid dir (the
   async-save failure mode) and sizing the walltime to the *remaining*
   iters (`ITER_MS` table + 2h30m margin for the SIGUSR2 grace).

On Azure only checks 2–4-lite apply (done-detection would need a blob
listing per cell); resubmitting a finished cell is a no-op run — Megatron
loads the final checkpoint and exits immediately.

## 3. Monitor

```bash
python3.11 pretrain_progress.py                 # what a re-launch would do, per cell
python3.11 pretrain_progress.py --filter 1.7B   # subset by name substring
python3.11 pretrain_progress.py --arch shallow --scheme B   # a variant's cells
python3.11 pretrain_progress.py --plot          # + the two heatmaps
```

`--plot` writes two grid heatmaps (x = model size, y = number of languages),
aggregated over **every** run found on disk regardless of variant:

- **`pretrain_progress_simple.png`** — cell = how many finished models exist
  at (size, L), across seeds, deep/shallow, scheme A/B, tokenizers.
- **`pretrain_progress_detailed.png`** — one row of binary (yellow 0 /
  blue 1) heatmaps per transformation: SEED (28 / 1797 / 1904),
  ARCH (deep / shallow), SCHEME (A / B), TOKENIZER (v1 for now).

Both PNGs are also refreshed automatically at the end of every
`launch_trainings.py cscs` invocation, so they're always up to date.

**Benchmark evals while pretraining** — automated on both platforms with
the same rule (**every 2nd saved checkpoint plus each run's final one**,
whatever its iter) and the same destination (W&B
**`mariagrandury-epflnlp/msnr`** — the project the training loss logs to,
so loss and benchmark curves live side by side). The `auto` group in
[`configs/tasks.json`](../../configs/tasks.json) lists **benchmark names**
(arc, belebele, global_mmlu, hellaswag, include_base_44, multiblimp,
xcopa, xnli, xstorycloze, xwinograd); each cell is evaluated on every
listed benchmark's tasks **in the languages it trains on** (English + its
setting's FineWeb-2 languages, mapped via
[`configs/languages.json`](../../configs/languages.json)) — e.g. the L2
cells get `hellaswag` + `hellaswag_ru` + … (18 tasks); L30 cells get 164,
L100 cells 290 — the task languages cover the full 100-language set.
Both watchers are idempotent: stop them, restart them, run them twice —
nothing duplicates.

- **CSCS** — [`auto_evals_cscs.py`](auto_evals_cscs.py) scans the checkpoint
  tree each pass and, per due checkpoint: submits a
  [`conversion/convert-snr.sh`](conversion/convert-snr.sh) job (Megatron →
  HF, models.json-driven — the watcher keeps `configs/models.json` in
  sync with the grid automatically via
  [`sync_models_json.py`](sync_models_json.py); commit the diff it makes),
  then on a later pass an [`../evals/`](../evals/) `evaluate.sbatch` job
  (vLLM, TP=1, `BATCH_TASKS=1`), which pushes to W&B from inside the job.

  ```bash
  python3.11 auto_evals_cscs.py --dry-run       # preview one pass
  python3.11 auto_evals_cscs.py --watch 600     # tmux: a pass every 10 min
  ```

- **Azure** — [`auto_evals_azure.py`](auto_evals_azure.py) does the same
  against blob storage, one watcher per workspace (each has its own blob
  store and compute; the UK one overrides the job YAMLs' Spain-only
  compute):
  `source azure/env.sh && python auto_evals_azure.py --watch 600` and
  `python auto_evals_azure.py --workspace uk --watch 600`.

## Per-size cluster cost (steady state)

350M–1B sampled from 1.26M iter log lines of the completed 36-model sweep
(same architectures and node counts); 90M and 175M measured on this sweep
(2026-08-21, flash attention, data on iopsstor); 1.7B is still an estimate:

| Size  | Nodes | MBS | Median ms/iter | Predictivity iters (5×C) | h (steady) |
| ----- | ----: | --: | -------------: | -----------------------: | ---------: |
| 90M   |     3 |   7 |       **1240** |                    4 500 |     ~1.6 h |
| 175M  |     6 |   7 |        **800** |                    8 540 |     ~1.9 h |
| 350M  |    14 |   3 |        **565** |                   16 660 |     ~2.6 h |
| 600M  |    21 |   6 |        **520** |                   28 800 |     ~4.2 h |
| 1B    |    21 |   6 |        **715** |                   45 740 |     ~9.1 h |
| 1.7B  |    21 |   2 |    ~1200 (est) |                   81 000 |      ~27 h |

deep and shallow cost the same per iteration at equal size (175M: 800 vs 807),
as expected once both ladders pin ffw = 4 and gqa = 4 — one table covers both.

The medians feed `launch_trainings.py::ITER_MS` and `auto_time()` (walltime =
remaining iters × rate + 2h30m margin for the 1h SIGUSR2 grace + cold-start).
For the small sizes that margin, not the rate, dominates the request. Runs
longer than the 12h wall (1B, 1.7B) chain through resumes: each job
checkpoints out at the SIGUSR2 signal and the next `launch_trainings.py cscs`
invocation resumes it.

**Always read ms/iter as a median with its p10/p90 band.** A wide spread means
the job was I/O-bound and the number is not a cost estimate (CLAUDE.md #8);
a tight band (e.g. 90M 1234/1240/1247) means it is compute-bound and usable.


## Possible future improvements

- **Azure micro-batch tuning** — the cluster-tuned per-GPU MBS values are
  memory-safe on Azure but leave throughput on the table with few GPUs
  (e.g. 350M on the 2-GPU node runs MBS 3 × accum 84; memory likely allows
  MBS 21). GBS 504 also forces odd divisors on 8-GPU nodes (600M/1B shrink
  6 → 3; 7 would fit better if memory allows). Measure on a pilot, then
  override per job with `--set environment_variables.MBS=…`.
- **`--chain N` for long CSCS runs** — the 1.7B (~27 h) spans ~3 walltime
  windows; today each window needs a fresh `launch_trainings.py cscs`
  invocation (the active-job skip blocks pre-queued singleton chains). A
  `--chain N` flag submitting N dependent jobs would make it hands-off.
- **Bake eval/convert deps into the AML images** — every eval job
  `pip install`s the lm-eval fork and every convert job pins transformers
  at startup (~minutes each); baking them into `apertus-eval` /
  `apertus-nemo` derivatives would shave that off all ~20 evals per cell.

## Completed: the 36-model data-mix sweep (history)

The first experiment trained **36 small multilingual Apertus models** — 4
sizes (175M–1B) × 3 FineWeb-Edu/FineWeb2-HQ mixtures (30/70, 60/40, 90/10) ×
3 seeds — each to iter 50 000 (~100B tokens). Checkpoints live under
`.../Meg-Runs/data-mix-small/apertus-<size>-fwEdu<edu>-fw2<fw2>-seed<seed>/`
and feed the `pretraining_custom` eval split ([`../evals/`](../evals/)) and
the `acc_vs_flops` curves. Its scripts evolved in place into the predictivity
tooling above (`launch_trainings.py`, `pretrain_progress.py`, and
`submit-apertus-data-mix.sh` → `launch_pretraining_cscs.sh` +
`megatron_args.sh`; the old `launch_resumes.sh` was folded into the
launcher's idempotency) — see git history for the sweep-era versions.

## See also

[`CLAUDE.md`](CLAUDE.md) — back-of-house notes: hard rules, failure modes
(TE `_extra_state` strictness, async-save shell dirs, the
`OptimizerParamScheduler` mismatch on capped resumes, …).