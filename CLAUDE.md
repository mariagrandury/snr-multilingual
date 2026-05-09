# Context for Claude — signal-and-noise (Apertus fork)

This is a local fork of [allenai/signal-and-noise](https://github.com/allenai/signal-and-noise),
augmented to run the SNR / decision-accuracy pipeline on the 12 **custom Apertus
pretraining checkpoints** evaluated by the sister `evals/` package
(`/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/`). The upstream
README still applies for the AllenAI DataDecide / OLMo path; this file documents
the Apertus extension.

If you're picking this up cold, **read the upstream [README.md](README.md) first**
to understand what signal, noise, decision accuracy, and scaling-law error mean.
This file is the back-of-house Claude memo for the local additions.

---

## What's actually running

The Apertus pipeline reuses the upstream signal-and-noise compute + plotting
helpers, but loads scores from the cluster's eval_logs tree instead of the
HF parquet dataset.

```bash
cd /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/signal-and-noise
python multilingual/run_apertus.py
```

That single entry point:
1. Walks `<EVAL_ROOT>/<model>-iter<N>/` and parses every `eval_*/results_*.json`
   into a long-form DataFrame (one row per `(model, ckpt, task)`).
2. Feeds it into `snr.snr_simple.main` with Apertus-specific sizes:
   `small_sizes=["175M", "350M", "600M"]`, `large_sizes_snr=["1B"]`,
   `target_size="1B"`, `target_step=None` (latest available step per mix).
3. Drops scaling-law error (`large_sizes_scaling=[]`) — we don't have a
   ladder of Apertus models, so prediction error is not computed.
4. Writes outputs under `results/` (the fork's repurposed `PLOT_DIR`):
   - `snr_per_task.csv` — one row per task with `decision_acc_<size>` /
     `snr_<size>` columns
   - `acc_vs_flops/per_benchmark/<family>.png` — one figure per benchmark
     family with subplots per language; built from
     `analysis.plotting.datadecide.plot_task_curves`
   - `acc_vs_flops/per_language/<lang>.png` — one figure per language
     with subplots per benchmark family
   - `snr_vs_decision_accuracy.png` — multi-panel scatter (one panel per
     small size, 175M/350M/600M → 1B), via `snr.plot.plot_snr_da_grid`

The two `acc_vs_flops/` views render the same per-task panels grouped two
ways, so re-running the pipeline produces both at once.

`multilingual/run_apertus_snr_variants.py` is a sibling entry point that
writes `results/snr_definition/snr_variants_per_task.csv`, with one
column per (variant, size) using every aggregator in
[snr/snr_variants.py](snr/snr_variants.py)'s `AGGREGATION_FUNCTIONS`.

`multilingual/analyze_snr_variants.py` reads that CSV and renders
`results/snr_definition/snr_vs_decision_accuracy.png` (one row of
size-panels per SNR variant, ordered top-to-bottom by overall R² with
decision accuracy) plus per-language counterparts
`snr_vs_decision_accuracy_<lang>.png`. No CSVs are emitted by the
analysis step — the per-task CSV is the only persisted table.

---

## Apertus models in scope (12 models)

```
apertus-{175M,350M,600M,1B}-fwEdu{30,60,90}-fw{270,240,210}-seed1904
```

- **mix** = `fwEdu{30,60,90}` (the parser strips the `fw270/240/210`
  complement, so a model's `mix` field carries only the FW-Edu ratio)
- **seed** = 1904 (constant; not the upstream DataDecide seed set)
- **size** = `175M`, `350M`, `600M`, `1B`

In `multilingual/run_apertus.py`:
- `SMALL_SIZES = ["175M", "350M", "600M"]`
- `TARGET_SIZE = "1B"`
- `PLOTTED_MIXES = ["fwEdu30", "fwEdu60", "fwEdu90"]`
- `SEED = 1904`

Half-trained models (600M-fwEdu90, 1B-fwEdu90, the three 175M-fwEdu*) may
have fewer than 5 ckpts on some mixes. `compute_snr_small_scale` in
`snr/snr_simple.py` already accommodates this — it keeps per-mix score
arrays as a jagged list rather than forcing a 2-D ndarray. Don't
"refactor" that back to a square array without re-handling the missing
ckpts, or SNR computation will crash on those mixes.

---

## Eval-results layout (read-only input)

`snr/download/apertus.py` reads from:

```
/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/
    mariagrandury-epflnlp/snr-experiments/
        <model>-iter<N>/
            harness/eval_*/results_*.json     (clean lm-eval output)
            harness/eval_*/per_task/<task>/   (partial, written per-task)
```

This tree is **populated by `src/evals/` (the eval submitter package)**, not
by this one. Every Slurm eval job writes there. We just read it.

To avoid duplicating the parser, `snr/download/apertus.py` does:

```python
sys.path.insert(0, "/iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals")
from scripts.push_all_results import collect, aggregate_parents
```

If `src/evals/scripts/push_all_results.py` moves or its `collect` /
`aggregate_parents` API changes, this import breaks. Both packages live
under the same `snr-multilingual/src/` parent, so they should usually be
in sync — but the coupling is implicit, not declared anywhere.

`collect` reads both `results` and `groups` from per-task fragments so
that aggregates like `mmlu` (which only live under `groups`) are
recovered when results are merged from sharded per-task runs.
`aggregate_parents` then folds e.g. `mmlu_anatomy`, `mmlu_humanities`, …
into `mmlu` when `mmlu` is present in the same ckpt's results.

The metric extracted per task is `acc,none` if present, else
`exact_match,none`. `acc_norm`, `acc_bytes`, `*_stderr`, `degeneration`
are intentionally dropped (matches the W&B push schema in the sister repo).

---

## Tokens / FLOPs (for the curves)

Computed inside `load_apertus_eval_results`:

- Tokens per iter: `_TOKENS_PER_ITER = 504 * 4096` (Megatron training config:
  `micro_batch_size * seq_len`).
- Tokens at iter N: `step * _TOKENS_PER_ITER`.
- Compute (FLOPs) ≈ `6 * params * tokens`, with `_PARAMS = {175M: 175e6,
  350M: 350e6, 600M: 600e6, 1B: 1.0e9}`.

These approximations are the same ones used by `src/evals/`'s W&B push
(`MEG_TOKENS_PER_ITER = 504 * 4096`) so axes line up across the two
pipelines.

---

## Outputs

`results/` is the destination. `PLOT_DIR` is set to `<repo>/results/` in
this fork (upstream points it at `img/`); the directory is committed,
unlike `img/` which is gitignored. Existing artifacts there are
overwritten on each run.

- `snr_per_task.csv` is the table version of the upstream Rich-print.
- `snr_definition/snr_variants_per_task.csv` is the wide-format
  counterpart from `run_apertus_snr_variants.py`: one column per
  (variant, size) for every aggregator in `AGGREGATION_FUNCTIONS`.
- `snr_definition/snr_vs_decision_accuracy.png` and
  `snr_definition/snr_vs_decision_accuracy_<lang>.png` are the variant×
  size scatter grids emitted by `analyze_snr_variants.py`.
- `acc_vs_flops/per_benchmark/` and `acc_vs_flops/per_language/`
  contain combined-grid PNGs (subplots per language and per benchmark
  family respectively). Tasks with incomplete `(size, mix)` coverage are
  silently skipped within a grid (the per-subplot try/except in
  `_plot_grid` swallows the exception — if you expect a task panel and
  it isn't drawn, that's why).
- `snr_vs_decision_accuracy.png` is a 1×3 grid (175M / 350M / 600M → 1B).

---

## Cluster gotchas

- **Login node:** `clariden-ln003`. System Python is 3.6 — use a recent
  env (`miniconda3/envs/snr` or similar) when running this; the upstream
  `pyproject.toml` requires Python ≥ 3.9.
- **No outbound internet from compute nodes.** If you need to call
  `pull_predictions_from_hf` (only used for the upstream DataDecide /
  OLMo path, not Apertus), do it from the login node.
- **`/iopsstor/scratch` is the work tree.** This repo and the eval_logs
  it reads both live there.

---

## Relationship to other local repos

| Repo | Path | Role |
|---|---|---|
| `signal-and-noise` (this) | `snr-multilingual/src/signal-and-noise` | SNR / decision-accuracy compute + plotting |
| `evals` | `snr-multilingual/src/evals` | Submits eval jobs, writes `eval_logs`, pushes to W&B |
| `pretrain` | `snr-multilingual/src/pretrain` | Pretraining submitter (sbatch wrappers) |
| `data-mix-small` (Megatron-LM) | `/iopsstor/scratch/cscs/mariagrandury/data-mix-small` | Pretraining; checkpoints under `Megatron-LM/logs/Meg-Runs/...` |

Flow: `pretrain/` → checkpoints → `evals/` submits `lm_eval` jobs →
`eval_logs/.../snr-experiments/<model>-iter<N>/` →
`signal-and-noise/multilingual/run_apertus.py` reads those and produces
SNR tables + plots.

---

## When upstream changes

This repo tracks `allenai/signal-and-noise`. When pulling upstream, the
local additions to watch for are:

- `multilingual/run_apertus.py` (local-only entry point)
- `snr/download/apertus.py` (local-only loader)
- The lazy import of `run_ladder` inside `compute_scaling_law_error`
  (`snr/snr_simple.py`) — done so the Apertus path doesn't need
  `olmo-ladder` installed. If upstream re-imports it at module top, the
  Apertus run will fail with a missing-dependency error.
- The jagged-array tolerance in `compute_snr_small_scale`
  (`snr/snr_simple.py`) for half-trained mixes.

Re-apply those if a merge undoes them.
