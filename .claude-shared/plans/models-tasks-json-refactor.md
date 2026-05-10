# Plan: One JSON source of truth for models + benchmarks

## Context

`models_pretraining_a06.txt` lists two more checkpoint sources to evaluate (`apertus3-1b-21-nodes` and `apertus3-3b-64-nodes`), but the per-path explicit iter lists in that file have no consumer today. More importantly, model and benchmark metadata is spread across:

- 7 `models_*.txt` files in [src/evals/configs/signal_to_ratio/](src/evals/configs/signal_to_ratio/) (paths/URLs only, no per-cell metadata)
- 8 `tasks_*.txt` files (just task names)
- Hardcoded constants in 4+ scripts: `MEG_TOKENS_PER_ITER` and `HF_STAGE_TOKENS` and `HF_MAIN_TOKENS` and `model_params()` regex in [push_all_results.py](src/evals/scripts/push_all_results.py); `_PARAMS` and `_TOKENS_PER_ITER` in [snr/download/apertus.py](src/signal-and-noise/snr/download/apertus.py); `DATA_RATIOS` and `SEEDS` in [pretrain/launch_trainings.py](src/pretrain/launch_trainings.py); `--seed-iters seed28=…` in both [launch_pretraining_hf.sh](src/evals/scripts/launch_pretraining_hf.sh) and [launch_pretraining_megatron.sh](src/evals/scripts/launch_pretraining_megatron.sh); `SMALL_SIZES`/`TARGET_SIZE`/`PLOTTED_MIXES`/`SEED` in [run_apertus.py](src/signal-and-noise/multilingual/run_apertus.py).
- A new conceptual axis the user wants — **four checkpoint subsets per model** (`all`, `dense_tail`, `10_ckpts`, `da_ckpts`) — that the current per-cell tooling doesn't express.

Goal: land `configs/models.json` + `configs/tasks.json` at the repo root, plus a small shared loader, and migrate all three pipelines (pretrain → eval → SNR) to read from it. Same data, declared once, consumed everywhere.

## Schema

### `configs/models.json` (top-level keys = canonical model name)

```json
{
  "apertus-175M-fwEdu30-fw270-seed1904": {
    "/* same for all seed1904 */": "",
    "family": "snr-pretraining-custom",
    "stage": "pretraining",
    "size": "175M",
    "params": 175000000,
    "hyperparams_key": "175M",
    "mix_en": 30,
    "mix_fw2": 70,
    "seed": 1904,
    "checkpoint_kind": "megatron_iter",
    "backends": {
      "megatron": "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1904/checkpoints/",
      "hf_local": "/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints/apertus-175M-fwEdu30-fw270-seed1904/"
    },
    "checkpoints": {
      "final": 50000,
      "all": [
        2000, 6000, 12000, 18000, 22000, 28000, 34000, 38000, 42000, 44000,
        46000, 48000, 50000
      ],
      "dense_tail": [42000, 44000, 46000, 48000, 50000],
      "10_ckpts": [
        6000, 12000, 18000, 22000, 28000, 34000, 38000, 42000, 46000, 50000
      ],
      "da_ckpts": [6000, 12000, 22000, 28000, 50000]
    }
  },
  "apertus-175M-fwEdu30-fw270-seed1797": {
    "/* same for all seed1797, with these checkpoints */": "",
    "checkpoints": {
      "final": 50000,
      "all": [
        2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000,
        24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000, 40000, 42000,
        44000, 46000, 48000, 50000
      ],
      "dense_tail": [42000, 44000, 46000, 48000, 50000],
      "10_ckpts": [
        6000, 10000, 16000, 20000, 26000, 30000, 36000, 40000, 46000, 50000
      ],
      "da_ckpts": [6000, 10000, 20000, 30000, 50000]
    }
  },
  "apertus-175M-fwEdu30-fw270-seed28": {
    "/* same for all seed28, with these checkpoints */": "",
    "checkpoints": {
      "final": 50000,
      "all": [
        2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 22000,
        24000, 26000, 28000, 30000, 32000, 34000, 36000, 38000, 40000, 42000,
        44000, 46000, 48000, 50000
      ],
      "dense_tail": [42000, 44000, 46000, 48000, 50000],
      "10_ckpts": [
        6000, 10000, 16000, 20000, 26000, 30000, 36000, 40000, 46000, 50000
      ],
      "da_ckpts": [6000, 10000, 20000, 30000, 50000]
    }
  },
  "apertus3-1b-21-nodes": {
    "family": "snr-pretraining-a06",
    "stage": "pretraining",
    "size": "1B",
    "params": 1000000000,
    "checkpoint_kind": "megatron_iter",
    "backends": {
      "megatron": "/capstor/store/cscs/swissai/infra01/main_run_megatron/main_run_megatron_ahgele/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-1b-21-nodes/checkpoints/"
    },
    "checkpoints": {
      "final": 0380000,
      "all": [
        0020000, 0040000, 0060000, 0080000, 0100000, 0120000, 0140000, 0160000,
        0180000, 0200000, 0220000, 0240000, 0260000, 0280000, 0300000, 0320000,
        0340000, 0360000, 0370000, 0380000
      ],
      "dense_tail": [0320000, 0340000, 0360000, 0370000, 0380000],
      "10_ckpts": [
        0020000, 0060000, 0100000, 0140000, 0180000, 0220000, 0260000, 0300000,
        0340000, 0380000
      ],
      "da_ckpts": [
        0020000, 0060000, 0100000, 0140000, 0180000, 0220000, 0260000, 0300000,
        0340000, 0380000
      ]
    }
  },
  "apertus3-3b-64-nodes": {
    "family": "snr-pretraining-a06",
    "stage": "pretraining",
    "size": "3B",
    "params": 3000000000,
    "checkpoint_kind": "megatron_iter",
    "backends": {
      "megatron": "/capstor/store/cscs/swissai/infra01/main_run_megatron/main_run_megatron_ahgele/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-3b-64-nodes/checkpoints/"
    },
    "checkpoints": {
      "final": 0165000,
      "all": [
        0015000, 0030000, 0045000, 0060000, 0075000, 0090000, 0105000, 0120000,
        0135000, 0150000, 0155000, 0160000, 0165000
      ],
      "dense_tail": [0135000, 0150000, 0155000, 0160000, 0165000],
      "10_ckpts": [
        0015000, 0030000, 0045000, 0060000, 0075000, 0090000, 0105000, 0120000,
        0135000, 0150000, 0165000
      ],
      "da_ckpts": [
        0015000, 0030000, 0045000, 0060000, 0075000, 0090000, 0105000, 0120000,
        0135000, 0150000, 0165000
      ]
    }
  },
  "Apertus-8B-2509": {
    "family": "swiss-ai-reference",
    "stage": "pretraining",
    "size": "8B",
    "params": 8000000000,
    "checkpoint_kind": "hf_branch",
    "backends": { "hf": "https://huggingface.co/swiss-ai/Apertus-8B-2509" },
    "checkpoints": {
      "all": [{ "branch": "main", "tokens": 15000000000000 }]
    }
  },
  "SmolLM3-3B-checkpoints": {
    "family": "huggingface-reference",
    "stage": "pretraining",
    "size": "3B",
    "params": 3000000000,
    "checkpoint_kind": "hf_branch",
    "backends": {
      "hf": "https://huggingface.co/HuggingFaceTB/SmolLM3-3B-checkpoints"
    },
    "checkpoints": {
      "all": [
        { "branch": "stage1-step-3440000", "tokens": 7200000000000 },
        { "branch": "stage2-step-4200000", "tokens": 8800000000000 },
        { "branch": "stage3-step-4720000", "tokens": 9900000000000 }
      ]
    }
  }
}
```

Notes on the schema:

- `checkpoint_kind` discriminator lets consumers branch cleanly between `megatron_iter` (list of ints; tokens derived via `iter × tokens_per_step`) and `hf_branch` (list of `{branch, tokens}` dicts).
- `hyperparams_key` back-links to [src/pretrain/hyperparams_deep.json](src/pretrain/hyperparams_deep.json) (kept separate, not folded — see R6 below). a06/HF reference models simply omit it.
- `tokens_per_step` is **not** stored per-model — derived from `hyperparams_deep.json["global"]["global_batch_size"] × ["seq_len"]` once.
- `params` is the rounded total used for FLOPs (preserves historical W&B values exactly); `hyperparams_deep.json` keeps the precise `n_non_emb_params` for SNR consumers that want it.

### `configs/tasks.json`

```json
{
  "tasks": {
    "arc_easy": {
      "language": "en",
      "benchmark": "arc",
      "stages": ["pretraining"]
    },
    "arc_zh": {
      "language": "zh",
      "benchmark": "arc",
      "stages": ["pretraining"]
    },
    "global_mmlu_full_zh": {
      "language": "zh",
      "benchmark": "global_mmlu",
      "stages": ["pretraining", "midtraining"]
    },
    "ifeval": {
      "language": "en",
      "benchmark": "ifeval",
      "stages": ["posttraining"],
      "metric": "exact_match"
    }
  },
  "groups": {
    "pretraining_full": ["arc_easy", "arc_zh", "hellaswag", "..."],
    "pretraining_allenai": ["..."],
    "pretraining_finetasks": ["..."],
    "midtraining": ["..."],
    "posttraining": ["..."]
  }
}
```

- `stages` is a list (some tasks span pretraining + midtraining — see [tasks_include.txt](src/evals/configs/signal_to_ratio/tasks_include.txt)).
- Optional `metric` per task — when declared, [push_all_results.py](src/evals/scripts/push_all_results.py)'s `flatten()` uses exactly that key; otherwise it falls back to the existing `acc` → `exact_match` heuristic.
- `groups` mirror today's `tasks_*.txt` files 1-to-1 so the migration is a string-list swap.

## Migration order

Each phase is independently committable and leaves both pretrain resumes and eval submissions running.

### Phase 0 — additive, no behavior change

1. Author `configs/models.json` and `configs/tasks.json` at the repo root (matches the layout already documented in [snr-multilingual/CLAUDE.md](CLAUDE.md)). Generate from the 7 model txt files + 8 task txt files + the user's a06 spec.
2. Add a thin loader at `src/evals/scripts/utils/configs.py` (functions: `load_models()`, `load_tasks(group=…)`, `iters_for(model, subset='full_eval')`, `tokens_for(model, ckpt_id)`). All three pipelines import from here.
3. Lift `collect()` and `aggregate_parents()` out of [push_all_results.py](src/evals/scripts/push_all_results.py) into the same `utils/` module so [snr/download/apertus.py](src/signal-and-noise/snr/download/apertus.py)'s implicit `sys.path` import has a stable home (mitigates R2).
4. Parity test: enumerated `(model, ckpt)` cells from JSON must equal the cells produced by today's [snr_progress.py](src/evals/scripts/snr_progress.py) on every existing `models_*.txt` — byte-identical `snr_progress.csv` against a snapshotted reference.

### Phase 1 — eval pipeline reads JSON

5. [snr_progress.py](src/evals/scripts/snr_progress.py): accept `--models-json <path>` (new) or `--models <txt>` (existing). The JSON path also retires `--seed-iters` (its job is now per-cell `full_eval`). Run dual-mode for ~2 days; `snr_progress.csv` must remain byte-identical for the existing 36 cells.
6. [\_eval_status.py](src/evals/scripts/_eval_status.py): same dual-mode treatment for tasks (file path or JSON group name).
7. Once parity is confirmed, flip [launch_pretraining_hf.sh](src/evals/scripts/launch_pretraining_hf.sh), [launch_pretraining_megatron.sh](src/evals/scripts/launch_pretraining_megatron.sh), and [launch_ckpts_in_progress.sh](src/evals/scripts/launch_ckpts_in_progress.sh) to JSON. Keep `--seed-iters` as a no-op-warning for one cycle (R5).
8. [push_all_results.py](src/evals/scripts/push_all_results.py): replace `MEG_TOKENS_PER_ITER`, `HF_STAGE_TOKENS`, `HF_MAIN_TOKENS`, `model_params()` with JSON lookups. Dry-run a W&B push for 5 representative runs (HF stage / HF main / Megatron iter / a06 / Apertus-8B) and diff the `iter`/`tokens`/`flops`/`<task>/acc` payloads against the live runs — must be identical (R7: wandb run id must remain stable).

### Phase 2 — pretrain pipeline

9. Rewrite [pretrain/launch_trainings.py](src/pretrain/launch_trainings.py) to iterate `models.json` filtered by `family=snr-pretraining-custom` (replacing hardcoded `DATA_RATIOS × SEEDS`). Architecture data still comes from `hyperparams_deep.json` via `hyperparams_key`.
10. Cross-check assertion at start: `--dry-run` must print exactly the canonical 36 cells. Keep `DATA_RATIOS`/`SEEDS` constants as a guard (error if JSON disagrees).

### Phase 3 — SNR analysis + a06 extension

11. [snr/download/apertus.py](src/signal-and-noise/snr/download/apertus.py): replace `_PARAMS` and `_TOKENS_PER_ITER` with JSON lookups via the shared loader.
12. Add a06 support to [multilingual/run_apertus.py](src/signal-and-noise/multilingual/run_apertus.py): include `family=snr-pretraining-a06` models in a separate plot track (their params + token-per-iter come from JSON; they read directly from `eval_logs/`, no parquet). Configure `SMALL_SIZES`/`TARGET_SIZE`/`PLOTTED_MIXES`/`SEED` from JSON or accept them as CLI args.
13. Parquet build: when regenerating the `pretraining_custom-*.parquet` and `reference_hf-*.parquet` splits, build the `mix` column as `f"fwEdu{mix_en}-fw{mix_fw2}"` from JSON (preserves the long form `_normalise_mix` strips on load — R3).

### Phase 4 — retire txt files

14. After some days of stable Phase 1+2+3, delete the txt files. Leave a stub note pointing colleagues to `configs/models.json` and `configs/tasks.json`.

## Risks

- **R1 — implicit `sys.path` coupling** from [snr/download/apertus.py](src/signal-and-noise/snr/download/apertus.py) into [push_all_results.py](src/evals/scripts/push_all_results.py) (`collect`, `aggregate_parents`). Phase 0 lifts these into `utils/` first; downstream imports from the new module.
- **R2 — parquet schema preservation.** Columns `model, size, mix, seed, step, task, primary_score, model_tokens, flops` and `mix` long-form `fwEdu30-fw270` must be preserved; HF reference parquet keeps `seed=NaN`. Snapshot dtypes/value distributions before Phase 3 step 13 and assert post.
- **R3 — longest-iter-list anchor for W&B x-axes.** [snr_progress.py](src/evals/scripts/snr_progress.py) anchors all small Megatron models to the longest per-cell iter list so seed-28's narrower 7-iter list and seed-1904's 13-iter list share the same W&B step grid for the same (size, mix). Preserve this in the new loader: when computing W&B step grids, take the union of `full_eval` across seeds for the same (size, mix). Document the intent in `configs.py`.
- **R4 — `--seed-iters` CLI removal.** Used by collaborators (`aromanou`, `cmeister747`) and the canonical one-liner in [src/evals/CLAUDE.md](src/evals/CLAUDE.md). Keep as no-op-warning for one cycle and update both CLAUDE.md files + [src/evals/configs/signal_to_ratio/README.md](src/evals/configs/signal_to_ratio/README.md).
- **R5 — `params` (total) vs `n_non_emb_params` precision.** Keep them as separate fields. `push_all_results.py` reads `params` (rounded total) to preserve historical FLOPs values; SNR pipelines may opt into the more accurate `n_non_emb_params` from `hyperparams_deep.json` if they want.
- **R6 — wandb run id stability.** Top-level keys in `models.json` must equal exactly the strings today's `parse_name()` produces (e.g. `apertus-175M-fwEdu30-fw270-seed1904`). Add a parity test: for every existing W&B run, derive the wandb id from JSON and confirm it matches.

## Verification

End-to-end test plan:

1. **Phase 0 parity**: `python3.11 src/evals/scripts/snr_progress.py --models-json configs/models.json` produces a `snr_progress.csv` byte-identical to the txt-driven baseline for all 36 canonical cells. Hash both with `sha256sum`.
2. **a06 enumeration**: `python3.11 src/evals/scripts/snr_progress.py --models-json configs/models.json --filter apertus3-1b-21-nodes` lists 13 cells (one per `full_eval` iter). Same for 3b-64-nodes.
3. **Phase 1 dry-run**: `bash src/evals/scripts/launch_pretraining_hf.sh --dry-run` prints the same set of (cell, iter, walltime, TP, PP) tuples as the txt-driven baseline; flip `--filter apertus3` to confirm the a06 cells appear with the right TP/PP for 1B/3B (note: 3B's KV-head count needs verification before adding to `tp_for()`/`pp_for()` — see CLAUDE.md bug 14).
4. **Phase 1 W&B parity**: dry-run [push_all_results.py](src/evals/scripts/push_all_results.py) for 5 representative model runs (HF stage / HF main / Megatron iter / a06 / Apertus-8B) and diff the resulting `iter`/`tokens`/`flops`/`<task>/acc` payloads against today's W&B runs — must match exactly.
5. **Phase 2 cells**: `python src/pretrain/launch_trainings.py --dry-run` prints exactly the canonical 36 cells (4 sizes × 3 mixes × 3 seeds) — no more, no fewer. Job-name and `--export=` strings byte-identical to today's output for one sample cell.
6. **Phase 3 SNR**: `python src/signal-and-noise/multilingual/run_apertus.py` produces the same `snr_per_task.csv` as today; new a06 plots land in a separate `acc_vs_flops_a06/` subdirectory.
7. **a06 evaluation**: `bash src/evals/scripts/launch_pretraining_megatron.sh --filter apertus3 --dry-run` queues 26 jobs (13 iters × 2 sizes), each pointing at the right capstor checkpoint dir. After a real launch, `python3.11 src/evals/scripts/snr_progress.py --filter apertus3` reports completion progress per iter.

## Critical files to modify

- [src/evals/scripts/snr_progress.py](src/evals/scripts/snr_progress.py) — JSON model enumeration
- [src/evals/scripts/\_eval_status.py](src/evals/scripts/_eval_status.py) — JSON task groups
- [src/evals/scripts/launch_pretraining_hf.sh](src/evals/scripts/launch_pretraining_hf.sh) — drop `--seed-iters`, add a06 TP/PP
- [src/evals/scripts/launch_pretraining_megatron.sh](src/evals/scripts/launch_pretraining_megatron.sh) — drop `--seed-iters`, support `family=snr-pretraining-a06`
- [src/evals/scripts/launch_ckpts_in_progress.sh](src/evals/scripts/launch_ckpts_in_progress.sh) — read CSV unchanged, but the upstream JSON drives what's enumerated
- [src/evals/scripts/push_all_results.py](src/evals/scripts/push_all_results.py) — JSON lookups for tokens/params
- [src/pretrain/launch_trainings.py](src/pretrain/launch_trainings.py) — iterate JSON instead of `DATA_RATIOS × SEEDS`
- [src/signal-and-noise/snr/download/apertus.py](src/signal-and-noise/snr/download/apertus.py) — JSON for `_PARAMS`/`_TOKENS_PER_ITER`
- [src/signal-and-noise/multilingual/run_apertus.py](src/signal-and-noise/multilingual/run_apertus.py) — config from JSON, a06 track

New files:

- `configs/models.json` (repo root)
- `configs/tasks.json` (repo root)
- `src/evals/scripts/utils/configs.py` (shared loader: `load_models`, `load_tasks`, `iters_for`, `tokens_for`)
- `src/evals/scripts/utils/results_io.py` (lifted `collect`, `aggregate_parents`, `flatten`)

Files to retire (Phase 4): the 7 `models_*.txt` and 8 `tasks_*.txt` files in [src/evals/configs/signal_to_ratio/](src/evals/configs/signal_to_ratio/).
