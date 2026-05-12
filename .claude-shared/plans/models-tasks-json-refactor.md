# Plan: One JSON source of truth for models + benchmarks (v2)

> v2 — adapted to the **new model unit definition**: every training run
> is a `model`; the **cross-size identity** for decision accuracy is a
> separate `family` field (the model name with only the size token
> stripped). Seed information is incidental — the framework runs on
> any HF / Megatron model with `model` + `size`, no seed required.
>
> **Backwards compatibility is not a constraint.** We're the main
> contributors and we're refactoring. Same numerical results across the
> migration matter (parity tests); same CLI commands do not.

## Path conventions

Throughout this plan, file references use the runtime path of the
consumer:

- **evals + pretrain code** runs on the CSCS cluster — referenced under
  `/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/`.
- **signal-and-noise code** runs on the local Mac — referenced under
  `/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/`.
- **Cluster-only paths** (checkpoint dirs, eval logs, `/capstor/...`
  / `/iopsstor/...` infra) appear in JSON `backends` and in shell-side
  references where the consumer is on the cluster.
- **HF dataset references** stay as `https://huggingface.co/...` URLs
  (loader-agnostic).

`configs/models.json` and `configs/tasks.json` themselves live at the
**shared repo root** — same content readable from either machine.
Paths *inside* the JSON `backends` field are always full cluster paths
(or HF URLs); the loader's resolvers handle which backend is available
on the current host.

## Context

`models_pretraining_a06.txt` lists two more checkpoint sources to evaluate (`apertus3-1b-21-nodes` and `apertus3-3b-64-nodes`), but per-path explicit iter lists in that file have no consumer today. More importantly, model and benchmark metadata is spread across:

- 7 `models_*.txt` files in [src/evals/configs/signal_to_ratio/](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/configs/signal_to_ratio/) (paths/URLs only, no per-cell metadata)
- 8 `tasks_*.txt` files (just task names)
- Hardcoded constants in 4+ scripts: `MEG_TOKENS_PER_ITER` / `HF_STAGE_TOKENS` / `HF_MAIN_TOKENS` / `model_params()` regex in [push_all_results.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/push_all_results.py); `_PARAMS` and `_TOKENS_PER_ITER` in [snr/download/apertus.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/snr/download/apertus.py); `DATA_RATIOS` and `SEEDS` in [pretrain/launch_trainings.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/launch_trainings.py); `--seed-iters seed28=…` in both launch_pretraining_*.sh; `SMALL_SIZES`/`TARGET_SIZE`/`PLOTTED_MIXES`/`SEED` in [run_apertus.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/run_apertus.py); `_strip_size_from_name` + `_SIZE_TOKENS` in [run_apertus_snr_variants.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/run_apertus_snr_variants.py).
- A new conceptual axis the user wants — **four checkpoint subsets per model** (`all`, `dense_tail`, `10_ckpts`, `da_ckpts`) — that the current per-cell tooling doesn't express.
- **The new (v2) `family` / `model_family` notion** that the multilingual SNR pipeline now relies on: cross-size DA groups by `family`, the SNR signal pool groups by `model`. Both must work for HF/external runs that have no seed.

Goal: land `configs/models.json` + `configs/tasks.json` at the repo root, plus a small shared loader, and migrate all three pipelines (pretrain → eval → SNR) to read from it. Same data, declared once, consumed everywhere — and seed-agnostic at the API boundary.

## Schema

### `configs/models.json`

Top-level keys = canonical model name (matches today's W&B run id `parse_name()` output). Two sections: `models` and `pools`.

```jsonc
{
  "models": {
    // ---------- Custom Apertus pretrains (36 total: 4 sizes × 3 mixes × 3 seeds) ----------
    // Per-seed checkpoint lists differ — seed1904 has the original 13-ckpt
    // sparse schedule, seed1797 and seed28 have the dense 25-ckpt schedule.
    // The `family` field is the cross-size identity: the model name with
    // only the size token stripped (mix + fw complement + seed preserved).

    "apertus-175M-fwEdu30-fw270-seed1904": {
      "source": "snr-pretraining-custom",
      "family": "apertus-fwEdu30-fw270-seed1904",
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
    // ... entries for the other sizes (350M, 600M, 1B) at the same mix/seed
    // share family="apertus-fwEdu30-fw270-seed1904" — that's the cross-size
    // pair DA uses. Same checkpoint list as seed1904 above.

    "apertus-175M-fwEdu30-fw270-seed1797": {
      "source": "snr-pretraining-custom",
      "family": "apertus-fwEdu30-fw270-seed1797",
      "stage": "pretraining",
      "size": "175M",
      "params": 175000000,
      "hyperparams_key": "175M",
      "mix_en": 30,
      "mix_fw2": 70,
      "seed": 1797,
      "checkpoint_kind": "megatron_iter",
      "backends": {
        "megatron": "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed1797/checkpoints/",
        "hf_local": "/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints/apertus-175M-fwEdu30-fw270-seed1797/"
      },
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
      "source": "snr-pretraining-custom",
      "family": "apertus-fwEdu30-fw270-seed28",
      "stage": "pretraining",
      "size": "175M",
      "params": 175000000,
      "hyperparams_key": "175M",
      "mix_en": 30,
      "mix_fw2": 70,
      "seed": 28,
      "checkpoint_kind": "megatron_iter",
      "backends": {
        "megatron": "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/Meg-Runs/data-mix-small/apertus-175M-fwEdu30-fw270-seed28/checkpoints/",
        "hf_local": "/iopsstor/scratch/cscs/mariagrandury/snr-hf-checkpoints/apertus-175M-fwEdu30-fw270-seed28/"
      },
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

    // ---------- a06 pretrains (2 total: 1B-21-nodes, 3B-64-nodes) ----------

    "apertus3-1b-21-nodes": {
      "source": "snr-pretraining-a06",
      "family": "apertus3-a06",
      "stage": "pretraining",
      "size": "1B",
      "params": 1000000000,
      "checkpoint_kind": "megatron_iter",
      "backends": {
        "megatron": "/capstor/store/cscs/swissai/infra01/main_run_megatron/main_run_megatron_ahgele/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-1b-21-nodes/checkpoints/"
      },
      "checkpoints": {
        "final": 380000,
        "all": [
          20000, 40000, 60000, 80000, 100000, 120000, 140000, 160000,
          180000, 200000, 220000, 240000, 260000, 280000, 300000, 320000,
          340000, 360000, 370000, 380000
        ],
        "dense_tail": [320000, 340000, 360000, 370000, 380000],
        "10_ckpts": [
          20000, 60000, 100000, 140000, 180000, 220000, 260000, 300000,
          340000, 380000
        ],
        "da_ckpts": [
          20000, 60000, 100000, 140000, 180000, 220000, 260000, 300000,
          340000, 380000
        ]
      }
    },

    "apertus3-3b-64-nodes": {
      "source": "snr-pretraining-a06",
      "family": "apertus3-a06",
      "stage": "pretraining",
      "size": "3B",
      "params": 3000000000,
      "checkpoint_kind": "megatron_iter",
      "backends": {
        "megatron": "/capstor/store/cscs/swissai/infra01/main_run_megatron/main_run_megatron_ahgele/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-3b-64-nodes/checkpoints/"
      },
      "checkpoints": {
        "final": 165000,
        "all": [
          15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000,
          135000, 150000, 155000, 160000, 165000
        ],
        "dense_tail": [135000, 150000, 155000, 160000, 165000],
        "10_ckpts": [
          15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000,
          135000, 150000, 165000
        ],
        "da_ckpts": [
          15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000,
          135000, 150000, 165000
        ]
      }
    },

    // ---------- AllenAI DataDecide ----------
    // The 25 mixes × 4 sizes = 100 DataDecide entries are NOT enumerated
    // here. The build_allenai_variants pipeline pulls them dynamically
    // from the `allenai/signal-and-noise` HF dataset at run time; the
    // loader's `add_family_column` auto-derives `family` for AllenAI rows
    // via _strip_size_from_name (AllenAI naming follows the convention
    // <recipe>-<size>, so the auto-derivation is reliable for them).

    // ---------- Swiss-AI reference (pretraining) ----------

    "Apertus-8B-2509": {
      "source": "swiss-ai-reference",
      "family": "Apertus-2509",
      "stage": "pretraining",
      "size": "8B",
      "params": 8000000000,
      "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/swiss-ai/Apertus-8B-2509" },
      "checkpoints": {
        "all": [{ "branch": "main", "tokens": 15000000000000 }]
      }
    },
    "Apertus-70B-2509": {
      "source": "swiss-ai-reference",
      "family": "Apertus-2509",
      "stage": "pretraining",
      "size": "70B",
      "params": 70000000000,
      "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/swiss-ai/Apertus-70B-2509" },
      "checkpoints": {
        "all": [{ "branch": "main", "tokens": 15000000000000 }]
      }
    },

    // ---------- HF reference: pretraining ----------

    "Qwen3-0.6B-Base": {
      "source": "huggingface-reference",
      "family": "Qwen3-Base",
      "stage": "pretraining",
      "size": "0.6B",
      "params": 600000000,
      "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/Qwen/Qwen3-0.6B-Base" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Qwen3-1.7B-Base": {
      "source": "huggingface-reference", "family": "Qwen3-Base", "stage": "pretraining",
      "size": "1.7B", "params": 1700000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/Qwen/Qwen3-1.7B-Base" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Qwen3-4B-Base": {
      "source": "huggingface-reference", "family": "Qwen3-Base", "stage": "pretraining",
      "size": "4B", "params": 4000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/Qwen/Qwen3.5-4B-Base" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Qwen3-8B-Base": {
      "source": "huggingface-reference", "family": "Qwen3-Base", "stage": "pretraining",
      "size": "8B", "params": 8000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/Qwen/Qwen3-8B-Base" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Qwen3-14B-Base": {
      "source": "huggingface-reference", "family": "Qwen3-Base", "stage": "pretraining",
      "size": "14B", "params": 14000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/Qwen/Qwen3-14B-Base" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },

    "gemma-3-1b-pt": {
      "source": "huggingface-reference", "family": "gemma-3-pt", "stage": "pretraining",
      "size": "1B", "params": 1000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/google/gemma-3-1b-pt" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "gemma-3-4b-pt": {
      "source": "huggingface-reference", "family": "gemma-3-pt", "stage": "pretraining",
      "size": "4B", "params": 4000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/google/gemma-3-4b-pt" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "gemma-3-12b-pt": {
      "source": "huggingface-reference", "family": "gemma-3-pt", "stage": "pretraining",
      "size": "12B", "params": 12000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/google/gemma-3-12b-pt" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "gemma-3-27b-pt": {
      "source": "huggingface-reference", "family": "gemma-3-pt", "stage": "pretraining",
      "size": "27B", "params": 27000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/google/gemma-3-27b-pt" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },

    "SmolLM3-3B-Base": {
      "source": "huggingface-reference",
      "family": "SmolLM3-Base",
      "stage": "pretraining",
      "size": "3B",
      "params": 3000000000,
      "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/HuggingFaceTB/SmolLM3-3B-Base" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "SmolLM3-3B-checkpoints": {
      "source": "huggingface-reference",
      "family": "SmolLM3-checkpoints",
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
    },

    // ---------- HF reference: midtraining ----------

    "Olmo-3-1025-7B": {
      "source": "huggingface-reference",
      "family": "Olmo-3-1025",
      "stage": "midtraining",
      "size": "7B",
      "params": 7000000000,
      "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/allenai/Olmo-3-1025-7B" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },

    // ---------- HF reference: posttraining ----------

    "Qwen3-0.6B": {
      "source": "huggingface-reference", "family": "Qwen3-it", "stage": "posttraining",
      "size": "0.6B", "params": 600000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/Qwen/Qwen3-0.6B" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Qwen3-1.7B": {
      "source": "huggingface-reference", "family": "Qwen3-it", "stage": "posttraining",
      "size": "1.7B", "params": 1700000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/Qwen/Qwen3-1.7B" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Qwen3-4B": {
      "source": "huggingface-reference", "family": "Qwen3-it", "stage": "posttraining",
      "size": "4B", "params": 4000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/Qwen/Qwen3.5-4B" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Qwen3-8B": {
      "source": "huggingface-reference", "family": "Qwen3-it", "stage": "posttraining",
      "size": "8B", "params": 8000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/Qwen/Qwen3-8B" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Qwen3-14B": {
      "source": "huggingface-reference", "family": "Qwen3-it", "stage": "posttraining",
      "size": "14B", "params": 14000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/Qwen/Qwen3-14B" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "gemma-3-1b-it": {
      "source": "huggingface-reference", "family": "gemma-3-it", "stage": "posttraining",
      "size": "1B", "params": 1000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/google/gemma-3-1b-it" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "gemma-3-4b-it": {
      "source": "huggingface-reference", "family": "gemma-3-it", "stage": "posttraining",
      "size": "4B", "params": 4000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/google/gemma-3-4b-it" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "gemma-3-12b-it": {
      "source": "huggingface-reference", "family": "gemma-3-it", "stage": "posttraining",
      "size": "12B", "params": 12000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/google/gemma-3-12b-it" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "gemma-3-27b-it": {
      "source": "huggingface-reference", "family": "gemma-3-it", "stage": "posttraining",
      "size": "27B", "params": 27000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/google/gemma-3-27b-it" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "SmolLM3-3B": {
      "source": "huggingface-reference", "family": "SmolLM3-it", "stage": "posttraining",
      "size": "3B", "params": 3000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/HuggingFaceTB/SmolLM3-3B" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Apertus-1.7B-it800000-SFT": {
      "source": "daslab-testing", "family": "Apertus-it-SFT-1", "stage": "posttraining",
      "size": "1.7B", "params": 1700000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/daslab-testing/Apertus-1.7B-it800000-SFT" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Olmo-3-7B-Instruct-SFT": {
      "source": "huggingface-reference", "family": "Olmo-3-Instruct-SFT", "stage": "posttraining",
      "size": "7B", "params": 7000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/allenai/Olmo-3-7B-Instruct-SFT" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Olmo-3-7B-Instruct-DPO": {
      "source": "huggingface-reference", "family": "Olmo-3-Instruct-DPO", "stage": "posttraining",
      "size": "7B", "params": 7000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/allenai/Olmo-3-7B-Instruct-DPO" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Olmo-3-7B-Instruct": {
      "source": "huggingface-reference", "family": "Olmo-3-Instruct", "stage": "posttraining",
      "size": "7B", "params": 7000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/allenai/Olmo-3-7B-Instruct" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Apertus-8B-Instruct-2509": {
      "source": "swiss-ai-reference", "family": "Apertus-Instruct-2509", "stage": "posttraining",
      "size": "8B", "params": 8000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "Apertus-70B-Instruct-2509": {
      "source": "swiss-ai-reference", "family": "Apertus-Instruct-2509", "stage": "posttraining",
      "size": "70B", "params": 70000000000, "checkpoint_kind": "hf_branch",
      "backends": { "hf": "https://huggingface.co/swiss-ai/Apertus-70B-Instruct-2509" },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },

    // ---------- Posttraining Megatron (distillation) ----------

    "apertus-0.6b-from8b-TOP256-long": {
      "source": "distillation",
      "family": "ap-from8b-TOP256",
      "stage": "posttraining",
      "size": "0.6B",
      "params": 600000000,
      "checkpoint_kind": "megatron_iter",
      "backends": {
        "megatron": "/capstor/store/cscs/swissai/infra01/distillation/checkpoints/distill/ap0.6b-from8b-TOP256-long/checkpoints/"
      },
      "checkpoints": { "final": null, "all": [] }
    },
    "apertus-1b-from8b-TOP256-long": {
      "source": "distillation",
      "family": "ap-from8b-TOP256",
      "stage": "posttraining",
      "size": "1B",
      "params": 1000000000,
      "checkpoint_kind": "megatron_iter",
      "backends": {
        "megatron": "/capstor/store/cscs/swissai/infra01/distillation/checkpoints/distill/ap1b-from8b-TOP256-long/checkpoints/"
      },
      "checkpoints": { "final": null, "all": [] }
    },

    // ---------- Tokenizer-lm experiments (Clara, capstor-local paths) ----------
    // Sizes "TBD" — lint flags them until filled in by the user.

    "tokenizer-lm-tiny-128k-apertus": {
      "source": "tokenizer-lm",
      "family": "tokenizer-lm-128k-apertus",
      "stage": "pretraining",
      "size": "TBD",
      "params": null,
      "checkpoint_kind": "hf_local",
      "backends": {
        "hf_local": "/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/hf-checkpoints/tiny-128k-apertus"
      },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "tokenizer-lm-small-128k-apertus": {
      "source": "tokenizer-lm",
      "family": "tokenizer-lm-128k-apertus",
      "stage": "pretraining",
      "size": "TBD",
      "params": null,
      "checkpoint_kind": "hf_local",
      "backends": {
        "hf_local": "/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/hf-checkpoints/small-128k-apertus"
      },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "tokenizer-lm-pilot-128k-apertus": {
      "source": "tokenizer-lm",
      "family": "tokenizer-lm-128k-apertus",
      "stage": "pretraining",
      "size": "TBD",
      "params": null,
      "checkpoint_kind": "hf_local",
      "backends": {
        "hf_local": "/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/hf-checkpoints/pilot-128k-apertus"
      },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    },
    "tokenizer-lm-full-128k-apertus": {
      "source": "tokenizer-lm",
      "family": "tokenizer-lm-128k-apertus",
      "stage": "pretraining",
      "size": "TBD",
      "params": null,
      "checkpoint_kind": "hf_local",
      "backends": {
        "hf_local": "/capstor/store/cscs/swissai/a139/cmeister/tokenizer-lm/hf-checkpoints/full-128k-apertus"
      },
      "checkpoints": { "all": [{ "branch": "main", "tokens": null }] }
    }
  },

  // ---------- POOLS (new in v2) ----------
  // Named SNR/DA pools the analysis scripts consume.
  // The loader expands a pool to a list of model names via `members`.
  "pools": {
    "seeds_1904": {
      "description": "Single-seed Apertus baseline (3 mixes × 1 seed). Held-out test for the framework-generalization story.",
      "stage": "pretraining",
      "members": [
        { "source": "snr-pretraining-custom", "seeds": [1904] }
      ],
      "include_external": true
    },
    "seeds_28_1797": {
      "description": "Train pool for framework generalization (3 mixes × 2 seeds).",
      "stage": "pretraining",
      "members": [
        { "source": "snr-pretraining-custom", "seeds": [28, 1797] }
      ],
      "include_external": true
    },
    "seeds_28_1797_1904": {
      "description": "Pooled all Apertus seeds (3 mixes × 3 seeds = 9 model_families per size). Recommended for downstream allenai_comparison / benchmark_creation work.",
      "stage": "pretraining",
      "members": [
        { "source": "snr-pretraining-custom", "seeds": [28, 1797, 1904] }
      ],
      "include_external": true
    },
    "pretraining_hf_reference": {
      "description": "All HF reference pretrains. Useful for cross-architecture SNR.",
      "stage": "pretraining",
      "members": [
        { "source": "huggingface-reference" },
        { "source": "swiss-ai-reference" }
      ]
    }
    // "midtraining" / "posttraining" pools can be added the same way.
  }
}
```

Notes on the schema:

- **`source`** (was `family` in v1): provenance / data-source category. Used for grouping in pools and for loader branch logic (e.g. how to read `backends`).
- **`family`** (new in v2): **cross-size identity** for DA. Two models share a `family` iff DA-size between them is meaningful (e.g. same training recipe at different scales). Explicit per-row in JSON — avoids fragile regex stripping for HF model names with non-standard size tokens.
- `checkpoint_kind` discriminator lets consumers branch cleanly between `megatron_iter` (list of ints; tokens derived via `iter × tokens_per_step`) and `hf_branch` (list of `{branch, tokens}` dicts).
- `hyperparams_key` back-links to [src/pretrain/hyperparams_deep.json](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/hyperparams_deep.json) (kept separate, not folded — see R6).
- `tokens_per_step` is **not** stored per-model — derived from `hyperparams_deep.json["global"]["global_batch_size"] × ["seq_len"]` once.
- `params` is the rounded total used for FLOPs (preserves historical W&B values exactly); `hyperparams_deep.json` keeps the precise `n_non_emb_params` for SNR consumers that want it.
- `seed` is **only** set for Apertus pretrains; HF / a06 / distill entries omit it. The new framework no longer requires a seed field for DA.
- For HF posttraining models I included `size` even though their checkpoints list is single-branch; this lets SNR scripts pool models by size and lets DA-size identify cross-size releases that share a `family`.
- Placeholder sizes (`TBD` for Clara's tokenizer-lm models) are flagged for the user to fill in once the actual params are known.

### `configs/tasks.json`

Unchanged from v1:

```json
{
  "tasks": {
    "arc_easy": { "language": "en", "benchmark": "arc", "stages": ["pretraining"] },
    "arc_zh":   { "language": "zh", "benchmark": "arc", "stages": ["pretraining"] },
    "global_mmlu_full_zh": { "language": "zh", "benchmark": "global_mmlu", "stages": ["pretraining", "midtraining"] },
    "ifeval":   { "language": "en", "benchmark": "ifeval", "stages": ["posttraining"], "metric": "exact_match" }
  },
  "groups": {
    "pretraining_full":     ["arc_easy", "arc_zh", "hellaswag", "..."],
    "pretraining_allenai":  ["..."],
    "pretraining_finetasks":["..."],
    "midtraining":          ["..."],
    "posttraining":         ["..."]
  }
}
```

`stages` is a list (some tasks span pretraining + midtraining); `metric` is per-task override for `flatten()`; `groups` mirror today's `tasks_*.txt` files 1-to-1.

## Shared loader API: `/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/utils/configs.py`

```python
# --- Models ---
load_models(path="configs/models.json") -> dict[str, ModelEntry]
get_model(name: str) -> ModelEntry            # KeyError if missing
filter_models(source=None, stage=None, family=None, size=None,
              seeds=None) -> list[str]        # returns model names

# --- Pools ---
load_pools() -> dict[str, PoolDef]
expand_pool(name: str) -> list[str]           # pool → list of model names
expand_pool_to_df_filter(name: str) -> dict   # {"models": [...],
                                              #  "include_external": bool}

# --- Checkpoints ---
iters_for(model_name: str, subset="all") -> list[int]
                                              # subset ∈ {"all", "dense_tail",
                                              #          "10_ckpts", "da_ckpts",
                                              #          "full_eval"}
tokens_for(model_name: str, ckpt_id: int | str) -> int
                                              # int for megatron_iter, str for
                                              # hf_branch; reads tokens_per_step
                                              # from hyperparams_deep.json

# --- family helpers (NEW in v2) ---
family_of(model_name: str, size: str = None) -> str
                                              # 1. If models.json has the model,
                                              #    return its declared `family`.
                                              # 2. Otherwise, auto-derive via
                                              #    _strip_size_from_name(model, size)
                                              #    — used for AllenAI rows (and
                                              #    any dynamically-loaded model
                                              #    not in models.json).
add_family_column(df: pd.DataFrame) -> pd.DataFrame
                                              # apply family_of() per row.
                                              # Idempotent. Used everywhere DA
                                              # is computed.

# --- Tasks ---
load_tasks(path="configs/tasks.json") -> dict[str, TaskEntry]
tasks_for_group(group: str) -> list[str]
metric_for(task: str) -> str | None
```

`add_family_column` is the **single hook** the SNR pipeline uses to attach the cross-size identity onto any df it operates on. Callers (Apertus + AllenAI + HF references) all get the same code path.

## Migration order

Each phase is independently committable. Phases are ordered by the
priority you set: **multilingual SNR adoption first, then the
eval-pipeline migration that unlocks a06 + HF evaluation**, then the
parquet/analysis touchups, then pretrain pipeline last (least urgent —
the 36 custom cells are already trained). Phases 0 and 5 (JSON + retire
txts) bracket the rest.

### Phase 0 — additive, no behavior change

1. Author `configs/models.json` and `configs/tasks.json` at the repo root (matches the layout documented in [snr-multilingual/CLAUDE.md](CLAUDE.md)). Generate from the model txt files + task txt files, preserving the checkpoint lists. **AllenAI DataDecide rows are intentionally absent** — fetched dynamically at parquet-build time.
2. Add the shared loader at `/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/utils/configs.py` (see API above). All three pipelines import from here.
3. Lift `collect()` and `aggregate_parents()` out of [push_all_results.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/push_all_results.py) into `/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/utils/results_io.py` so [snr/download/apertus.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/snr/download/apertus.py)'s implicit `sys.path` import has a stable home (mitigates R1).
4. **Parity tests**:
   - Enumerated `(model, ckpt)` cells from JSON must equal cells from today's [snr_progress.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/snr_progress.py) on every existing `models_*.txt` — byte-identical `snr_progress.csv` against a snapshotted reference.
   - `family_of(name)` for every existing Apertus name must equal today's `_strip_size_from_name(name, size)`.
   - For each AllenAI DataDecide model encountered during a build_allenai_variants dry-run, `family_of()` (auto-derive branch) must produce the same family as the auto-derivation we run today.

### Phase 1 — multilingual SNR adopts `family`

(Cheapest payoff: the three pools already exist on disk; we only need to swap CLI + remove the local stripping helper.)

5. Wire `add_family_column` into both Apertus loaders:
   - [snr/download/apertus.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/snr/download/apertus.py): `load_apertus_eval_results` and `load_reference_hf_eval_results` call `add_family_column` after `_read_parquet`. Removes the in-place `_strip_size_from_name` in `run_apertus_snr_variants.py`.
   - [results/allenai_comparison/build_allenai_variants.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/results/allenai_comparison/build_allenai_variants.py): same — but its AllenAI rows aren't in models.json, so they hit the auto-derive branch in `family_of()`.
6. Update [run_apertus_snr_variants.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/run_apertus_snr_variants.py):
   - Remove the local `_strip_size_from_name`, `_SIZE_TOKENS`, `add_model_family` helpers (moved to `utils/configs.py`).
   - **Replace `--seeds` with `--pool <name>`** (no backwards-compat alias). Output dir derived from the pool name (`results/snr_definition/<pool>/`).
7. Same `--pool` migration for [analyze_snr_variants.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/analyze_snr_variants.py), [snr_definition_postprocess.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/snr_definition_postprocess.py), [smooth_subtasks.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/smooth_subtasks.py), [smooth_subtasks_per_sample.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/smooth_subtasks_per_sample.py), [run_apertus.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/run_apertus.py), [compare_seed_splits.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/compare_seed_splits.py), [results/allenai_comparison/analyze.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/results/allenai_comparison/analyze.py), [results/benchmark_creation/analyze.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/results/benchmark_creation/analyze.py).
8. **Parity test**: `python run_apertus_snr_variants.py --pool seeds_1904` must produce a byte-identical CSV to today's `results/snr_definition/seeds_1904/snr_variants_per_task.csv`. Same for `seeds_28_1797` and `seeds_28_1797_1904`.

### Phase 2 — eval pipeline reads JSON (unlocks a06 + HF evaluation)

This is the priority phase after Phase 1: we want to start launching a06 + HF reference evals through the same pipeline that today only handles the Apertus custom 36-cell grid.

9. [snr_progress.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/snr_progress.py): replace the txt-file model enumeration with JSON. CLI becomes `--models-json <path>` and `--filter <pattern>`. **No txt-file dual-mode.** Parity: `snr_progress.csv` byte-identical for the existing 36 cells.
10. [\_eval_status.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/_eval_status.py): same — task groups are read from `tasks.json["groups"]` only.
11. [launch_pretraining_hf.sh](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_pretraining_hf.sh), [launch_pretraining_megatron.sh](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_pretraining_megatron.sh), [launch_ckpts_in_progress.sh](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_ckpts_in_progress.sh) flip to JSON. `--seed-iters` removed.
12. **Add a06 + HF launchability**:
    - launch_pretraining_megatron.sh: filter by `source IN {snr-pretraining-custom, snr-pretraining-a06}` and the new `distillation` source. TP/PP for the a06 1B/3B configs verified (see CLAUDE.md bug 14 on 3B's KV-head count).
    - launch_pretraining_hf.sh: filter by `source IN {huggingface-reference, swiss-ai-reference, daslab-testing}` and `stage IN {pretraining, midtraining, posttraining}` via `--stage`.
13. [push_all_results.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/push_all_results.py): replace `MEG_TOKENS_PER_ITER`, `HF_STAGE_TOKENS`, `HF_MAIN_TOKENS`, `model_params()` with JSON lookups. Dry-run a W&B push for 7 representative runs (HF stage / HF main / Megatron iter / a06 / Apertus-8B / Qwen3-Base / gemma-3-pt) and diff `iter`/`tokens`/`flops`/`<task>/acc` payloads against live runs — must be identical (R6: wandb run id stable).

After this phase, launching evals for the new HF + a06 models is one CLI invocation:

```bash
bash /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_pretraining_megatron.sh --filter apertus3
bash /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_pretraining_hf.sh --stage pretraining --filter Qwen3
```

### Phase 3 — bring new model results into the multilingual SNR pipeline

14. [snr/download/apertus.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/snr/download/apertus.py): replace `_PARAMS` and `_TOKENS_PER_ITER` with JSON lookups via the shared loader.
15. Parquet rebuild (`build_hf_dataset.py`): when regenerating the `pretraining_custom-*.parquet` and `reference_hf-*.parquet` splits, build the `mix` column as `f"fwEdu{mix_en}-fw{mix_fw2}"` from JSON (preserves the long form `_normalise_mix` strips on load — R2). Also write the `family` column directly into the parquet, so downstream loaders can skip the `add_family_column` step.
16. Add a06 to [multilingual/run_apertus.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/run_apertus.py): include `source=snr-pretraining-a06` models on a separate plot track (params + tokens-per-iter from JSON; they read directly from `eval_logs/`, no parquet round-trip). Configure `SMALL_SIZES` / `TARGET_SIZE` / `PLOTTED_MIXES` from JSON or accept as CLI args.
17. New pools in `models.json` for the freshly-evaluated models — e.g. `pretraining_hf_reference` (already drafted), `pretraining_a06`, `pretraining_all` (cross-source). Each becomes a `--pool` argument the multilingual SNR scripts accept.

### Phase 4 — pretrain pipeline (lowest priority)

18. Rewrite [pretrain/launch_trainings.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/launch_trainings.py) to iterate `models.json` filtered by `source=snr-pretraining-custom` (replacing hardcoded `DATA_RATIOS × SEEDS`). Architecture data still comes from `hyperparams_deep.json` via `hyperparams_key`. The 36 custom cells are already trained, so this is a quality-of-life refactor; no parity test against a future launch (it's against past launches).
19. Cross-check assertion at start: `--dry-run` must print exactly the canonical 36 cells. Job-name and `--export=` strings byte-identical to past launch logs for one sample cell.

### Phase 5 — retire txt files

20. After some days of stable Phases 1–3, delete the 7 `models_*.txt` and 8 `tasks_*.txt`. Leave a stub note pointing to `configs/models.json` and `configs/tasks.json`.

## Risks

- **R1 — implicit `sys.path` coupling** from [snr/download/apertus.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/snr/download/apertus.py) into [push_all_results.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/push_all_results.py) (`collect`, `aggregate_parents`). Phase 0 lifts these into `utils/results_io.py` first; downstream imports from the new module.
- **R2 — parquet schema preservation.** Columns `model, family, size, mix, seed, step, task, primary_score, model_tokens, flops` and `mix` long-form `fwEdu30-fw270` must be preserved (with `family` added in Phase 3 step 15); HF reference parquet keeps `seed=NaN`. Snapshot dtypes/value distributions before Phase 3 step 15 and assert post.
- **R3 — longest-iter-list anchor for W&B x-axes.** [snr_progress.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/snr_progress.py) anchors all small Megatron models to the longest per-cell iter list so seed-28's narrower iter list and seed-1904's 13-iter list share the same W&B step grid for the same (size, mix). Preserve this in the new loader: when computing W&B step grids, take the union of `full_eval` across seeds for the same (size, mix). Document in `configs.py`.
- **R4 — `params` (total) vs `n_non_emb_params` precision.** Keep them as separate fields. `push_all_results.py` reads `params` (rounded total) to preserve historical FLOPs values; SNR pipelines may opt into the more accurate `n_non_emb_params` from `hyperparams_deep.json` if they want.
- **R5 — `family` field hand-curation is error-prone.** For 100+ rows, typos in `family` produce silent DA bugs (wrong cross-size grouping). Mitigation: Phase 0 includes a script `scripts/lint_models_json.py` that asserts (a) every Apertus row's `family` is consistent with `_strip_size_from_name(model, size)` (HF rows are allowed to diverge — that's the whole point of an explicit field), and (b) every `family` either appears at ≥2 sizes (DA-size meaningful) or has only one size (DA-size will be NaN — flagged but not fatal).
- **R6 — wandb run id stability.** Top-level keys in `models.json` must equal exactly the strings today's `parse_name()` produces (e.g. `apertus-175M-fwEdu30-fw270-seed1904`). Add a parity test: for every existing W&B run, derive the wandb id from JSON and confirm it matches.
- **R7 — Apertus-8B and Apertus-70B share `family="Apertus-2509"`.** Stripping their size produces the same family but they're independent training runs. DA-size between 8B and 70B is technically computable but not statistically meaningful (n=1 family covering 2 sizes). The lint script flags families with only one model per size; DA-size with only 1 family is NaN by the existing `<2 models` guard, so the case is harmless — flagged for awareness, not blocked.
- **R8 — `tokenizer-lm-*` models have unknown sizes** (Clara's experiments). Filed under `size: "TBD"`. Pipelines must skip TBD-size rows in SNR / DA computation. Phase 0 lints them; user fills in real sizes when known.

## Verification

End-to-end test plan, ordered to match the phase ordering above:

**Phase 0 — JSON + loader**
1. `python3.11 /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/snr_progress.py --models-json configs/models.json` produces a `snr_progress.csv` byte-identical to the txt-driven baseline for all 36 canonical Apertus cells. Hash both with `sha256sum`.
2. `python scripts/lint_models_json.py` exits 0; reports the expected single-size families (Apertus-2509 at 8B + 70B; daslab-testing rows; tokenizer-lm TBD-size rows).

**Phase 1 — multilingual SNR adopts `family`**
3. For each pool ∈ {`seeds_1904`, `seeds_28_1797`, `seeds_28_1797_1904`}: `python multilingual/run_apertus_snr_variants.py --pool <pool>` produces a CSV byte-identical to today's `results/snr_definition/<pool>/snr_variants_per_task.csv`. The loader-side `add_family_column` must produce the same `family` values as today's local stripping (pre-parquet-rebuild).
4. `python results/allenai_comparison/analyze.py --pool seeds_28_1797_1904` reproduces today's pooled-pool top-K Jaccard = 1.0 across all K ∈ {5, 10, 20}.
5. `python results/benchmark_creation/analyze.py --pool seeds_28_1797_1904` reproduces today's per-family SNR ranking (multiblimp 3.81 → arc 0.62).

**Phase 2 — eval pipeline + a06/HF launchability**
6. `python3.11 /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/snr_progress.py --models-json configs/models.json --filter apertus3-1b-21-nodes` lists 13 cells (one per `full_eval` iter). Same for `apertus3-3b-64-nodes`.
7. `python3.11 /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/snr_progress.py --filter Qwen3 --stage pretraining` lists all 5 Qwen3 pretrain entries (0.6B, 1.7B, 4B, 8B, 14B), each with a single `main` ckpt.
8. `bash /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_pretraining_megatron.sh --filter apertus3 --dry-run` queues 26 jobs (13 iters × 2 sizes), each pointing at the right capstor checkpoint dir with the right TP/PP for 1B vs 3B.
9. `bash /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_pretraining_hf.sh --stage pretraining --filter Qwen3 --dry-run` queues 5 jobs (one per Qwen3 size).
10. **W&B parity**: dry-run [push_all_results.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/push_all_results.py) for 7 representative model runs (HF pretrain / HF post / Megatron iter custom / a06 / Apertus-8B / Qwen3-Base / gemma-3-pt) and diff `iter`/`tokens`/`flops`/`<task>/acc` payloads against today's W&B runs — must match exactly for the 36 Apertus cells (the new model types are write-only at this phase).

**Phase 3 — parquet + multilingual SNR for new models**
11. After the parquet rebuild, `python /Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/run_apertus.py --pool pretraining_hf_reference` produces curves for the HF reference models. a06 plots land in `acc_vs_flops/<pool>_a06/`.
12. `python multilingual/run_apertus_snr_variants.py --pool pretraining_all` produces a CSV that pools Apertus custom + a06 + HF reference at each size where ≥2 model_families coexist.

**Phase 4 — pretrain pipeline**
13. `python /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/launch_trainings.py --dry-run` prints exactly the canonical 36 cells (4 sizes × 3 mixes × 3 seeds) — no more, no fewer. Job-name and `--export=` strings byte-identical to past launch logs for one sample cell.

## Critical files to modify

- [src/evals/scripts/snr_progress.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/snr_progress.py) — JSON model enumeration
- [src/evals/scripts/\_eval_status.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/_eval_status.py) — JSON task groups
- [src/evals/scripts/launch_pretraining_hf.sh](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_pretraining_hf.sh) — drop `--seed-iters`, add a06 TP/PP
- [src/evals/scripts/launch_pretraining_megatron.sh](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_pretraining_megatron.sh) — drop `--seed-iters`, support `source=snr-pretraining-a06`
- [src/evals/scripts/launch_ckpts_in_progress.sh](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/launch_ckpts_in_progress.sh) — read CSV unchanged, upstream JSON drives what's enumerated
- [src/evals/scripts/push_all_results.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/push_all_results.py) — JSON lookups for tokens/params
- [src/pretrain/launch_trainings.py](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/pretrain/launch_trainings.py) — iterate JSON instead of `DATA_RATIOS × SEEDS`
- [src/signal-and-noise/snr/download/apertus.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/snr/download/apertus.py) — JSON for `_PARAMS`/`_TOKENS_PER_ITER`; call `add_family_column` after parquet read
- [src/signal-and-noise/multilingual/run_apertus_snr_variants.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/run_apertus_snr_variants.py) — remove local `_strip_size_from_name`/`_SIZE_TOKENS`/`add_model_family`; import from `utils/configs.py`; `--pool` CLI
- [src/signal-and-noise/multilingual/run_apertus.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/run_apertus.py) — config from JSON, a06 track, `--pool` CLI
- [src/signal-and-noise/multilingual/analyze_snr_variants.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/analyze_snr_variants.py),
  [snr_definition_postprocess.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/snr_definition_postprocess.py),
  [smooth_subtasks.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/smooth_subtasks.py),
  [smooth_subtasks_per_sample.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/smooth_subtasks_per_sample.py),
  [compare_seed_splits.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/multilingual/compare_seed_splits.py),
  [results/allenai_comparison/analyze.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/results/allenai_comparison/analyze.py),
  [results/allenai_comparison/build_allenai_variants.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/results/allenai_comparison/build_allenai_variants.py),
  [results/benchmark_creation/analyze.py](/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/results/benchmark_creation/analyze.py)
   — `--pool <name>` CLI; output dirs derived from pool name; import `add_family_column` from `utils/configs.py`.

New files:

- `configs/models.json` (repo root) — including the `pools` section
- `configs/tasks.json` (repo root)
- `/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/utils/configs.py` (shared loader: `load_models`, `load_tasks`, `iters_for`, `tokens_for`, `family_of`, `add_family_column`, `expand_pool`)
- `/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/utils/results_io.py` (lifted `collect`, `aggregate_parents`, `flatten`)
- `scripts/lint_models_json.py` (parity / consistency lint — see R7)

Files to retire (Phase 5): the 7 `models_*.txt` and 8 `tasks_*.txt` files in [src/evals/configs/signal_to_ratio/](/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/configs/signal_to_ratio/).

## What's different from v1

- **`source` vs `family`**: existing `family` field renamed to `source` (data-provenance category). New `family` field is the cross-size identity, declared explicitly per row (instead of computed from regex stripping). The lint script (R5) guards against typos.
- **`pools` section** added to `models.json` so `seeds_1904`, `seeds_28_1797`, `seeds_28_1797_1904`, `pretraining_hf_reference`, etc. live in config rather than CLI conventions. Scripts adopt `--pool <name>` and the loader expands it into a list of model names.
- **All HF / Megatron entries from `models_*.txt`** are placeholders in v2 (Qwen3 pretrain + post; Gemma-3 pretrain + post; SmolLM3 family; Apertus 1.7B/8B/70B post; Olmo-3 mid + post; Clara's tokenizer-lm experiments; distillation runs). v1 only listed Apertus-8B and SmolLM3-3B as sketches.
- **AllenAI DataDecide rows are NOT enumerated** in models.json. `build_allenai_variants.py` continues to pull them dynamically from the `allenai/signal-and-noise` HF dataset; `family_of()` auto-derives their `family` via size-stripping (their naming convention works perfectly with the auto-derive branch).
- **`add_family_column` is the single hook** for cross-size identity. Replaces the local `_strip_size_from_name` regex in `run_apertus_snr_variants.py`. The Apertus / AllenAI / HF reference loaders all flow through it. The helper has two branches: lookup-from-JSON for declared models, auto-derive via size-stripping for dynamic rows.
- **`scripts/lint_models_json.py`** (new) catches family typos and TBD-size rows before they corrupt downstream DA.
- **Phase ordering**: Phase 1 (multilingual SNR adoption) is first — cheapest, parity already validated by this session's runs. Phase 2 (eval pipeline) is next — enables a06 + HF reference evaluations. Phase 3 brings their results into the multilingual SNR pipeline via a parquet rebuild. Phase 4 (pretrain pipeline) is last — the 36 custom cells are already trained, so it's just a quality-of-life refactor.
- **No CLI backwards compatibility.** `--seeds`, `--seed-iters`, and `--models <txt>` go away in their respective phases without aliases or deprecation cycles. Parity tests still guarantee the *numerical* output stays identical for the existing 36-cell corpus.
