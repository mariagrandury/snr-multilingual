# Small Multilingual Pretrained Models

Pretraining of small-scale multilingual Apertus models. The canonical sweep:

- **4 model sizes**: 175M, 350M, 600M, 1B
- **3 data mixtures** of FineWeb-Edu (English) and FineWeb2-HQ (non-English),
  ratios 30/70, 60/40, 90/10
- **3 seeds** per data mixture: 28, 1797, 1904
- **Total**: 4 sizes × 3 mixtures × 3 seeds = **36 models**, each trained to
  iter 50000 (~100B tokens at GBS 504 × seq 4096)

## Active flow (the only thing you should run)

| File | Role |
|---|---|
| [`launch_resumes.sh`](launch_resumes.sh) | The right entry point. Reads `pretrain_progress.py --actions`, then per cell submits `done`/`fresh`/`resume`/skip-on-`corrupt`. **Idempotent** — re-runnable. Handles end-gap (resume to 50000) and mid-gap (rewind marker, train to a specific canonical) automatically. |
| [`launch_trainings.py`](launch_trainings.py) | Wraps `sbatch --export=…` from [`hyperparams_deep.json`](hyperparams_deep.json). Used by `launch_resumes.sh`; can also be invoked directly with `--size`/`--mix_en`/`--seed` filters or `--training-steps N` to cap an early exit. |
| [`submit-apertus-data-mix.sh`](submit-apertus-data-mix.sh) | The sbatch template. `--save` and `--load` both point at the experiment's checkpoint dir, so the same script handles fresh and resume runs. |
| [`pretrain_progress.py`](pretrain_progress.py) | Status. Text dashboard (default), `--plot` (writes both `progress.png` canonical-stage heatmap + companion `progress_all.png` for every 2000-step iter), `--actions` (machine-readable per-cell action consumed by `launch_resumes.sh`). |

```bash
# The standard one-liner to drive the sweep to 100% canonical:
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain && \
  bash launch_resumes.sh --dry-run            # always sanity-check first
  bash launch_resumes.sh                      # then for real

# Filter to a subset:
bash launch_resumes.sh --filter seed28
bash launch_resumes.sh --filter apertus-175M-fwEdu30   # both 30/70 seeds you care about

# Status + plots:
python3.11 pretrain_progress.py --target 50000
python3.11 pretrain_progress.py --plot pretrain_progress.png         # +HF/Hub stages
python3.11 pretrain_progress.py --plot pretrain_progress.png --no-hub  # offline / rate-limited
```

## Per-size cluster cost (steady-state, sampled from 1.26M iter logs)

| size | nodes | MBS | tokens / iter (504×4096) | ms / iter (median) | iters / hour | hours to 50000 (steady) |
|---|---|---|---|---|---|---|
| 175M | 6 | 7 | 2.06M | **800** | ~4500 | ~11 h |
| 350M | 14 | 3 | 2.06M | **565** | ~6400 | ~7.8 h |
| 600M | 21 | 6 | 2.06M | **520** | ~6900 | ~7.2 h |
| 1B | 21 | 6 | 2.06M | **715** | ~5000 | ~9.9 h |

These numbers are encoded in `launch_resumes.sh::ITER_MS` and used to size
`--time` per submission (with a 2h30m margin for SIGUSR2 grace + cold-start).
Save iters add ~30–60% to that single iter's wall time at 175M/350M (less
amortized than the larger sizes); the medians above already reflect this.

## Files (kept for reference)

### Hyperparameters
- [`hyperparams_deep.json`](hyperparams_deep.json) — **active** config consumed by `launch_trainings.py`.
- [`find_hyperparams_deep.py`](find_hyperparams_deep.py) — one-shot generator for `hyperparams_deep.json`.
- `hyperparams.json` / `find_hyperparams.py` / `calculate_params_lr_bs.py` / `fetch_hf_model_hyperparams.py` / `hf_models.txt` / `hf_model_hyperparams.csv` — exploratory artifacts kept for reference.

### Conversion + Hub push (under [`conversion/`](conversion/))

- [`conversion/convert-snr.sh`](conversion/convert-snr.sh) — Megatron `torch_dist` → `torch` → HuggingFace conversion for the sweep. Three modes: per-iter, sbatch wrapper (loops a plan file inside the container), launcher (walks the 36 cells, writes per-size plans, optionally `--submit`s). Walltime auto-set per size to **02:00:00** (full 117-iter sweep is ~78 min wall + overhead; per-iter cost is ~35-40s and uniform across sizes since the bottleneck is shard I/O, not GPU compute).
- [`conversion/push-snr.py`](conversion/push-snr.py) — push converted iter dirs to the per-seed `snr-models-<seed>` HF orgs as `stage1-step-<NNNNN>` branches; mirrors the latest iter to `main`. 429-aware backoff. Run from the login node (only needs `HF_TOKEN`).

> `convert-snr.sh` still depends on Megatron internals (`torchdist_2_torch.py`, `tools/checkpoint/{convert,loader_core,saver_swissai_hf}.py`) — those can't move because they `import megatron.core` and `pretrain_gpt`. The script reaches them via `$MEGATRON_LM_DIR` (default `/iopsstor/scratch/cscs/$USER/data-mix-small/Megatron-LM`).

### Misc
- `create_data_mixture.py` — one-shot data-mixture builder (the data is already built and frozen at `/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/mix_100B_<edu>_<fw2>`).
- `merge_wandb_experiment.py` — post-hoc W&B run merging across resumes.
- `env.toml` — pyxis container env file.

## See also

[`CLAUDE.md`](CLAUDE.md) — back-of-house notes, hard rules, and the failure
modes worth remembering (TE `_extra_state` strictness, async-save shell
dirs, the `OptimizerParamScheduler` mismatch on mid-gap fills, etc.).
