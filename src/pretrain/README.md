# Small multilingual pretrained models

> Pretraining infrastructure for the canonical 36-model Apertus sweep that
> feeds the SNR analysis: 4 sizes × 3 data mixtures × 3 seeds, each trained
> to iter 50 000 (~100 B tokens).

## What we trained

**36 small multilingual Apertus models**, organised on three axes:

| Axis                                     | Values                   |
| ---------------------------------------- | ------------------------ |
| Model size                               | 175 M, 350 M, 600 M, 1 B |
| Data mixture (FineWeb-Edu : FineWeb2-HQ) | 30/70, 60/40, 90/10      |
| Random seed                              | 28, 1797, 1904           |

Each combination is one independent training run. Run name:
`apertus-${MODEL_SIZE}-fwEdu${FW_EDU_RATIO}-fw2${FW2_RATIO}-seed${SEED}`,
with checkpoints under
`/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/<EXP_NAME>/checkpoints/`.

All runs target **iter 50 000** ≈ 100 B tokens (global batch size 504,
sequence length 4096). Canonical checkpoints saved at iters 2000, 6000,
12000, 18000, 22000, 28000, 34000, 38000, 42000, 44000, 46000, 48000,
50000 — 10 evenly spaced + 4 dense picks from the final 10% so the
late-training plateau is readable.

This sweep feeds two evaluation pools:

- **`pretraining_custom`** parquet split, consumed by the SNR
  framework's `signal_to_ratio` configs in [`../evals/`](../evals/).
- **`acc_vs_flops`** training curves in
  [`../signal-and-noise/results/acc_vs_flops/`](../signal-and-noise/results/acc_vs_flops/).

## Per-size cluster cost (steady state)

Sampled from 1.26 M iter log lines across all training runs:

| Size  | Nodes | MBS | Tokens / iter (504 × 4096) | Median ms/iter | Iters / h | h to 50 000 |
| ----- | ----: | --: | -------------------------: | -------------: | --------: | ----------: |
| 175 M |     6 |   7 |                     2.06 M |        **800** |    ~4 500 |       ~11 h |
| 350 M |    14 |   3 |                     2.06 M |        **565** |    ~6 400 |      ~7.8 h |
| 600 M |    21 |   6 |                     2.06 M |        **520** |    ~6 900 |      ~7.2 h |
| 1 B   |    21 |   6 |                     2.06 M |        **715** |    ~5 000 |      ~9.9 h |

The medians are encoded in `launch_resumes.sh::ITER_MS` and used to size
`--time` per submission (with a 2h30m margin for SIGUSR2 grace +
cold-start). Save iters add ~30–60% to that single iter's wall time at
175 M/350 M (less amortised than at larger sizes); the medians above
already include this.

## Active flow — the one thing you should run

`launch_resumes.sh` is the canonical entry point. It reads
`pretrain_progress.py --actions`, then per cell submits `done` / `fresh` /
`resume` or skips on `corrupt`. **Idempotent** — re-runnable. Handles
end-gap (resume to 50 000) and mid-gap (rewind marker, train to a specific
canonical) automatically.

```bash
# The standard one-liner — drive the sweep to 100% canonical
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain && \
  bash launch_resumes.sh --dry-run            # always sanity-check first
  bash launch_resumes.sh                      # then for real
```

Filter to a subset:

```bash
bash launch_resumes.sh --filter seed28
bash launch_resumes.sh --filter apertus-175M-fwEdu30   # both 30/70 seeds you care about
```

Live status + plots:

```bash
python3.11 pretrain_progress.py --target 50000
python3.11 pretrain_progress.py --plot pretrain_progress.png            # +HF/Hub stages
python3.11 pretrain_progress.py --plot pretrain_progress.png --no-hub   # offline / rate-limited
```

`pretrain_progress.py` writes both `progress.png` (canonical-stage 3-panel
heatmap with HF/Hub stages) and a companion `progress_all.png` (every
2000-step iter, Megatron-presence only).

## The four active scripts

| File                                                       | Role                                                                                                                                                                                                                                                                                                                                  |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`launch_resumes.sh`](launch_resumes.sh)                   | The right entry point. Reads `pretrain_progress.py --actions`, dispatches per cell: `done` → skip · in `squeue` → skip · `fresh` → submit from-scratch · `resume <load_iter> <target>` → resume (rewinds `latest_checkpointed_iteration.txt` first if mid-gap) · `corrupt` → skip with a warning.                                     |
| [`launch_trainings.py`](launch_trainings.py)               | Wraps `sbatch --export=…` from [`hyperparams_deep.json`](hyperparams_deep.json). Default `SEEDS = [28, 1797, 1904]`. Supports `--size`, `--mix_en`, `--seed` filters, `--dry-run`, `--test`, `--training-steps N` (cap early exit), pass-throughs (`--time`, `--account`, `--dependency`).                                            |
| [`submit-apertus-data-mix.sh`](submit-apertus-data-mix.sh) | The sbatch template. Reads env vars (`MODEL_SIZE`, …, `FW_EDU_RATIO`, `FW2_RATIO`, `SEED`, `TRAINING_STEPS`, `LR`, `MBS`) from the launcher. `--save` and `--load` both point at the same checkpoint dir, so the same script handles fresh and resume runs. Pinned to `--use-checkpoint-opt_param-scheduler` (mid-gap resume safety). |
| [`pretrain_progress.py`](pretrain_progress.py)             | Status. Three modes: text dashboard (default); `--plot PATH` writes the heatmap + companion `progress_all`; `--actions` emits one machine-readable line per cell consumed by `launch_resumes.sh`. Validates `iter_NNNNNNN/` has both `.metadata` and ≥1 `.distcp` shard before counting it as valid.                                  |

`pretrain_progress.py` anchors all small Megatron models to the longest
per-model iter list, so fully-trained and half-trained models share the
same x-axis grid. Mid-gap canonicals (missing iter X with X+ canonicals
present) are filled one-at-a-time — the launcher targets the _earliest_
missing canonical per cell per call, and re-running picks up the next gap.

## Files (kept for reference)

### Hyperparameters

- [`hyperparams_deep.json`](hyperparams_deep.json) — **active** config consumed by `launch_trainings.py`.
- [`find_hyperparams_deep.py`](find_hyperparams_deep.py) — one-shot generator for `hyperparams_deep.json`.
- `hyperparams.json` / `find_hyperparams.py` / `calculate_params_lr_bs.py` / `fetch_hf_model_hyperparams.py` / `hf_models.txt` / `hf_model_hyperparams.csv` — exploratory artefacts kept for reference.

### Conversion + Hub push (under [`conversion/`](conversion/))

- [`conversion/convert-snr.sh`](conversion/convert-snr.sh) — Megatron `torch_dist` → `torch` → HuggingFace conversion for the sweep. Three modes: per-iter, sbatch wrapper (loops a plan file inside the container), launcher (walks the 36 cells, writes per-size plans, optionally `--submit`s). Walltime auto-set per size to **02:00:00** (full 117-iter sweep is ~78 min wall + overhead; per-iter cost is ~35-40 s and uniform across sizes since the bottleneck is shard I/O, not GPU compute).
- [`conversion/push-snr.py`](conversion/push-snr.py) — push converted iter dirs to the per-seed `snr-models-<seed>` HF orgs as `stage1-step-<NNNNN>` branches; mirrors the latest iter to `main`. 429-aware backoff. Run from the login node (only needs `HF_TOKEN`).

`convert-snr.sh` still depends on Megatron internals
(`torchdist_2_torch.py`, `tools/checkpoint/{convert,loader_core,saver_swissai_hf}.py`)
— those can't move because they `import megatron.core` and `pretrain_gpt`.
The script reaches them via `$MEGATRON_LM_DIR` (default
`/iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM`). See
[`conversion/README.md`](conversion/README.md) for the conversion details.

### Misc

- [`create_data_mixture.py`](create_data_mixture.py) — one-shot data-mixture builder (the data is already built and frozen at `/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/mix_100B_<edu>_<fw2>`).
- [`merge_wandb_experiment.py`](merge_wandb_experiment.py) — post-hoc W&B run merging across resumes.
- [`env.toml`](env.toml) — pyxis container env file.

## UPDATED PLAN: Predictivity sweep (small-to-large)

A **separate** experiment from the 36-model sweep above (own scripts and
configs, so that pipeline is untouched): a 6-rung size ladder × 8 language
settings, fixed **50/50** English/multilingual data, one run per
(size, language-setting, seed) — ~57 runs. Full design in
[`../../.claude-shared/plans/small-to-large-predictivity-training-plan.md`](../../.claude-shared/plans/small-to-large-predictivity-training-plan.md).
The design-choice intervention (tokenizer / depth / temperature) is not yet
wired — see the plan's open question 4.

### 1. Build the data mixtures (once)

[`build_data_mixtures.py`](build_data_mixtures.py) drives
[`create_data_mixture.py`](create_data_mixture.py) over the whole sweep: one
fixed validation set, one English (DCLM) dataset, and one FineWeb-2 dataset per
language setting. They are blended 50/50 at train time, so each is built once.

```bash
# Dry-run first — prints the per-build token targets and the exact commands:
python build_data_mixtures.py --scheme A --output_dir <DATA_DIR> --dry_run

# Build everything (validation + English + all 7 FineWeb-2 settings):
python build_data_mixtures.py --scheme A --output_dir <DATA_DIR>

# Or build a subset (each create_data_mixture.py run is resumable):
python build_data_mixtures.py --scheme A --output_dir <DATA_DIR> --stage validation
python build_data_mixtures.py --scheme A --output_dir <DATA_DIR> --stage english
python build_data_mixtures.py --scheme A --output_dir <DATA_DIR> --stage fineweb --settings 8,30
```

`--scheme {A,B}` picks the language lists
(`language_sets_scheme{A,B}.json` — A is resource-ranked, B diversity-first).
Targets: 184.5 B English (bounds the L=1 run), 52 B / 92.5 B FineWeb-2 per
setting (half the largest run's budget + 10% headroom).

### 2. Launch the trainings

[`launch_trainings_predictivity.py`](launch_trainings_predictivity.py)
enumerates the grid and submits one `sbatch` per cell, reusing
`submit-apertus-data-mix.sh` via env hooks. Per-size architecture, LR, and
micro-batch come from the reviewed hyperparams files — `--arch deep` (default,
[`hyperparams_deep.json`](hyperparams_deep.json)) or `--arch shallow`
([`hyperparams.json`](hyperparams.json), the model-depth intervention variant
at the same non-embedding sizes). The budget D(N) = 100 × N and the WSD
schedule (~4% warmup, ~20% decay) are derived per size by the launcher.
Checkpoints land under `.../Meg-Runs/predictivity/<EXP_NAME>/`.

```bash
# Whole sweep — always dry-run first:
python launch_trainings_predictivity.py --data_dir <DATA_DIR> --dry-run
python launch_trainings_predictivity.py --data_dir <DATA_DIR>
```

**Passing sizes — `--size` takes one size or a comma-separated list:**

```bash
python launch_trainings_predictivity.py --data_dir <DATA_DIR> --size 600M
python launch_trainings_predictivity.py --data_dir <DATA_DIR> --size 350M,175M

# "all sizes up to 1B" = every rung except the 1.7B top:
python launch_trainings_predictivity.py --data_dir <DATA_DIR> --size 90M,175M,350M,600M,1B

# The shallow depth-intervention variant of the same grid:
python launch_trainings_predictivity.py --data_dir <DATA_DIR> --arch shallow
```

Filters compose with `--langs L` (one language setting) and `--seed`:

```bash
python launch_trainings_predictivity.py --data_dir <DATA_DIR> --size 600M --langs 8
python launch_trainings_predictivity.py --data_dir <DATA_DIR> --langs 1        # monolingual anchors
python launch_trainings_predictivity.py --test --data_dir <DATA_DIR>          # 90M, L8, 50 steps
```

Sizes: **90M, 175M, 350M, 600M, 1B, 1.7B** (1.7B trains only at L ∈
{1,8,30,100}). Language settings L ∈ {1, 2, 8, 15, 30, 50, 100}. Three seeds
(28, 1797, 1904) on the 175M and 1B columns at L ∈ {1, 30, 100}; one seed (1904)
elsewhere.

## See also

[`CLAUDE.md`](CLAUDE.md) — back-of-house notes: hard rules, failure modes
(TE `_extra_state` strictness, async-save shell dirs, the
`OptimizerParamScheduler` mismatch on mid-gap fills, …) and the live
status snapshot.
