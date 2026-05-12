# Context for Claude — signal-and-noise (Apertus fork)

## Working rules

- **Never `rm -rf`, `mv`, or otherwise delete/relocate files or
  directories without first asking the user for permission.** Even
  files that look orphaned, redundant, or "obviously stale" may be
  intentionally kept or referenced elsewhere — ask first. This
  applies to all destructive shell operations (`rm`, `mv` to a
  different parent, `git rm`, etc.).


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

## Hard-won bug history (don't reintroduce)

These are non-obvious gotchas that broke something in the past and are
not derivable by reading the code; some are cross-file invariants. If
you're modifying the multilingual / variant pipelines, scan this list
before editing.

### 1. `_is_language_aggregate` filter must accept 3- and 4-trailing-token forms
`multilingual/smooth_subtasks.py:_is_language_aggregate` was originally
`<family>_<lang>` or `<family>_<lang>_<script>` only, with a hard-coded
ISO 15924 script list. That silently dropped:

- `<family>_<lang>_<format>` — every `truthfulqa_<lang>_mc1`. The whole
  multilingual TruthfulQA family (7 langs) disappeared from Case 1
  before the fix.
- `<family>_<lang>_<script>_<extra>` —
  `global_piqa_completions_spa_latn_spai` (3 trailing tokens). PIQA had
  10 langs instead of 11 before the fix.

The fix replaces the script-only allowlist with `_TRAILING_OK = {
script codes ∪ lm-eval format suffixes (mc1, mc2) }` and requires
*every* trailing token to be in that set. Subject facets like
`global_mmlu_ar_anatomy` are still correctly rejected.

If you add a new family with non-standard trailing tokens (e.g. a new
`*_mc3` format), extend `_TRAILING_OK`, not the parser.

### 2. Numeric size sort, not lexicographic
`snr/download/apertus.py:_size_key` maps `175M → 0.175`, `1B → 1.0`,
`7B → 7.0`. Both `_read_parquet` and `load_all_eval_results` sort on
this. Lexicographic string sort gives `['175M', '1B', '350M', '600M']`,
which silently breaks any plot that lays sizes left-to-right by index.
If you add a sort elsewhere, use `_size_key`.

### 3. Parent-only task filter in variant analysis
The HF parquet ships per-(lang, subject) facets like
`global_mmlu_full_ar_anatomy` alongside the per-language aggregate
`global_mmlu_full_ar`. The original cluster pipeline used
`aggregate_parents` to fold children into parents and drop them, so
its variant CSV was 104 rows. After the parquet migration,
`run_apertus_snr_variants.py` was iterating over both forms and
emitted 819 rows.

The downstream impact was on per-language Pearson r: ~57 subject
facets per language are statistical replicates of the same parent
ckpt-trajectory, so they ~6×-weighted whichever variant tracked the
subject decomposition (mpd) over the variant that tracked the
across-benchmarks signal — flipping the global "best variant" answer.

The fix is `_is_parent_task` in `run_apertus_snr_variants.py`,
mirroring `aggregate_parents` semantics:
- keep tasks in `_ENGLISH_ONLY_TASKS` (mmlu, hellaswag, …)
- keep tasks where `_is_language_aggregate(task, family)` is True
- drop everything else

The helper imports `_is_language_aggregate` from `smooth_subtasks.py`,
so the filter is shared. **Whenever you load the parquet for variant
analysis, apply this filter** — never iterate raw parquet rows.

### 4. `data_noise` is cross-mix, not per-mix step std
`run_apertus_snr_variants.py:data_noise` was originally per-mix step
std. It must be the cross-mix std of per-mix FINAL scores (the same
quantity used elsewhere as "across-corpora dispersion at the latest
ckpt"). Only `rel_std_snr` reads `data_noise`, but it's the canonical
SNR variant — under the bad definition every `rel_std` cell collapsed
to ~1.0 (signal == noise) and `rel_std`'s rank-correlation vs DA
moved from -0.017 (rank 22) to +0.170 (rank 14, alongside
`rel_dispersion` and `iqr`).

### 5. Rank variants by signed Pearson r, not R²
`analyze_snr_variants.py` ranks variants for the per-language tables.
**Use signed Pearson r descending**, not R². R² conflates +0.5 with
-0.5; ranking by R² placed anti-signals (`projection r=-0.08`,
`tukey r=-0.12`) high in the per-language tables. Now those correctly
sit at the bottom; `mpd` / `dispersion` / `range` / `quartile_deviation`
tie at the top with r=+0.244.

### 6. Skip per-language fits with n < 5 valid points
`config_snr_ax` divides by `n-2` for the confidence band. With n=2..4,
quantized decision accuracy produces meaningless fits. `analyze_snr_
variants.py` skips per-language figures when every size has <5 valid
`(snr, decision_acc)` points and raises the in-panel fit-line
threshold to `n>=5`. Affected langs in the current corpus: de, fr, th
(skipped with a log message).

### 7. Sparse-mix gate: drop sparse mixes, not whole tasks
`compute_snr_small_scale`, `_per_sample_snr`, and
`_variance_prefilter_mask` all share a "drop mixes with <2 ckpts"
gate (`_drop_sparse_mixes` in `smooth_subtasks_per_sample.py`).
Without the gate, a single sparse mix would reject the entire
(task, size) cell. Recovered 8 tasks (truthfulqa_{ar,es,eu,hi,ru,vi,zh}
_mc1 and global_piqa_completions_spa_latn_spai) plus 395 newly-finite
cells in pre-existing tasks. **No values changed on previously-finite
cells** — the gate only affects what gets emitted, not how it's
computed.

### 8. `compute_ckpt_decision_accuracy` requires exact-step matching
The function compares (early_step, late_step) pairs across mixes; if
any mix is missing the exact early step, that pair is silently
NaN-filled. Behavior is documented in the docstring; a one-shot
warning per `(size, early_step, mix)` fires when an early row is
missing for a mix. Currently 35 unique missing-cell warnings on the
real corpus, all from task-specific eval-cadence gaps (paws_*, etc).

If you change the eval cadence, expect more warnings and possibly
more NaN cells in `da_ckpt` views.

### 9. `top_benchmarks_per_language` size column is parametric
`multilingual/snr_definition_postprocess.py:top_benchmarks_per_language`
used to hardcode `da_size_col = "decision_acc_size_600M"`. Now it
follows the `size` arg (`f"decision_acc_size_{size}"`). At the default
`size=1B` the column is NaN by definition — DA-size is `small_size →
target`, so no DA-size exists for the target itself. Don't "fix" the
NaN by reverting to a hardcoded smaller size; it would mix two
different definitions in the same table.

### 10. AllenAI driver `_canonical_seed` for multi-seed corpora
For datasets with multiple seeds (DataDecide), `_canonical_seed` in
`build_allenai_variants.py` picks the most-common seed before the
SNR computation. The Apertus driver pins `SEED=1904` (single-seed,
no-op). The seed filter is defensive against multi-seed step
interleaving in the trailing-N window — currently inert on the
AllenAI `core` split (no seed column, one model per (size, mix, step,
task)) but still active code; don't remove it without re-checking
the random-seeds split.

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
