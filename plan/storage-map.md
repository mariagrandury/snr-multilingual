# Storage map — what lives where

One line per artifact. The pattern everywhere: **iopsstor = working copy**
(fast, swept ~30 days), **capstor = durable master**, **Hub = what a
collaborator without cluster access can reach**. If a thing matters and this
map says it lives only on iopsstor, that is a gap, not a design.

## HF Hub — org policy

- **`msnr`** — model repos ONLY: one public repo per cell, one branch per
  iter (`step-<NNNNNN>`, `main` = latest), mirrored **manually** from the
  capstor staging tree by `src/pretrain/conversion/push-snr.py`.
- **`msnr-data`** — every published *data* artifact: the ladder report
  (`msnr-data/ladder-report`, pushed by `ladder_report.py --push-hf`), and
  any future CSVs / eval-results datasets. Nothing else goes in other orgs.
- Legacy, frozen: `multilingual-snr/multilingual-snr-eval-results` (36-sweep
  eval dataset), `multilingual-snr/msnr-ladder-report` (pre-policy report
  pushes, superseded), `snr-models-{28,1797,1904}` (36-sweep model repos).

## CSCS — iopsstor scratch (`/iopsstor/scratch/cscs/mariagrandury/`)

| Path | Holds | Durable copy |
| ---- | ----- | ------------ |
| `data/` | training-stage data mixtures Megatron reads (`stage_to_iopsstor.sh` restages after a purge) | capstor `multilingual_data_mixtures/predictivity-data/` |
| `data-mix-small/Megatron-LM/logs/Meg-Runs/msnr/<cell>/checkpoints/` | live Megatron torch_dist checkpoints | HF snapshots on capstor (every saved iter is converted) |
| `data-mix-small/Megatron-LM/logs/eval_logs/<entity>/msnr/` | eval results tree (harness JSONs, per_task) — the watcher's gate | capstor `msnr-eval-logs/` via **manual** `mirror_eval_logs.sbatch` |
| `data-mix-small/Megatron-LM/logs/slurm/training/` | training job stdout/stderr | capstor `msnr-train-logs/` (same manual mirror) |
| `data-mix-small/Megatron-LM/logs/auto_evals/` | auto-eval watcher logs | none (disposable) |
| `hf_home/datasets/` | offline HF dataset cache for eval jobs | rebuildable: `download_eval_datasets.py` |
| `Projects/snr-multilingual/` | the repo — the sweeper eats loose git objects, push often | GitHub `mariagrandury/snr-multilingual` |
| `snr-hf-checkpoints/` | LEGACY 36-sweep HF conversions (apertus-*) — nothing writes here | none (one sweep from gone) |

## CSCS — capstor store (`/capstor/store/cscs/swissai/infra01/`)

| Path | Holds |
| ---- | ----- |
| `multilingual_data_mixtures/predictivity-data/` | data-mixture master (+ `logs/` with build/stage logs and per-language `*.plan.json`) |
| `msnr-hf-models/<cell>/iter_<N>/` | every converted HF snapshot — the durability step, and the source `push-snr.py` mirrors to the Hub |
| `msnr-eval-logs/` | rsync mirror of the eval-results tree (manual, run `mirror_eval_logs.sbatch`) |
| `msnr-train-logs/` · `msnr-run-logging/` | mirror targets for training logs / Meg-Runs metadata (same sbatch) |
| `msnr-ladder-report/` | ladder_report.csv / _curve.csv / .md (written by `ladder_report.py`) |

## Azure blob (one store per workspace, ES + UK)

| Path | Holds |
| ---- | ----- |
| `predictivity/data/…` | uploaded data mixtures (scheme A + `schemeB/`, `validation/`) |
| `predictivity/runs/<cell>/checkpoints` | Azure training checkpoints |
| `models/<cell>/iter_<N>/` | Azure-side HF conversions (`.hf_complete` markers) |
| `eval_logs/<entity>/msnr/…` | Azure eval results (uploaded by `azure/eval.sh`) |

## W&B (`mariagrandury-epflnlp`)

- **`msnr`** — predictivity training loss + benchmark evals, one continuous
  run per cell across resumes.
- **`snr-experiments`** — legacy 36-sweep only; nothing new lands here.

## Known gaps (in priority order)

1. The capstor eval-log mirror is **manual** — anything evaluated since the
   last `mirror_eval_logs.sbatch` run exists only on swept scratch.
2. Hub model pushes are **manual** (`push-snr.py`); until run, capstor holds
   the only durable copy of the converted checkpoints — and the `msnr` /
   `msnr-data` orgs must exist on the Hub before the first push (org
   creation is a web-UI step).
3. `snr-hf-checkpoints/` (36-sweep conversions) sits on scratch with no
   master — rescue to capstor or accept losing it.
