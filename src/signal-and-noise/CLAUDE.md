# Context for Claude — signal-and-noise (Apertus fork)

## Working rules

- **Never `rm -rf`, `mv`, or otherwise delete/relocate files or
  directories without first asking the user for permission.** Even
  files that look orphaned, redundant, or "obviously stale" may be
  intentionally kept or referenced elsewhere — ask first. This
  applies to all destructive shell operations (`rm`, `mv` to a
  different parent, `git rm`, etc.).

- **Reuse existing code aggressively; keep new code simple and
  boilerplate-free.** Before writing a new helper, grep the repo for
  one that already does the job (e.g. `get_slice`,
  `signal_to_noise_ratio`, `decision_acc_fast`, `_is_language_aggregate`,
  the loader helpers in `snr/download/`, the shared CLI patterns in
  `analysis/`). Don't reimplement, don't wrap-for-wrap's-sake, and
  don't add defensive scaffolding ("just in case" config flags,
  pre-validation of arguments that won't be wrong, try/except around
  pure-Python logic). When extending a script, the new diff should
  read like a small addition, not a rewrite. Prefer one direct call
  over a chain of pass-through helpers.


This is a local fork of [allenai/signal-and-noise](https://github.com/allenai/signal-and-noise),
augmented to run the SNR / decision-accuracy pipeline on our own pretraining
ladders. Two generations of models flow through it:

- the **predictivity ladder** (current): `lm-<size>-L<L>[-schemeB]-<deep|shallow>-seed<seed>`,
  90M–1.7B × L ∈ {1, 2, 8, 15, 30, 50, 100} × deep/shallow × scheme A/B ×
  seeds, evaluated during training by `src/pretrain/auto_evals_*.py` and
  summarised by `src/pretrain/ladder_report.py` into **one wide CSV** published
  as the HF dataset `msnr-data/ladder-report`. That CSV is the source of truth.
- the **36-model sweep** (2026-04…06, superseded): `apertus-<size>-fwEdu<N>-fw<M>-seed<S>`
  plus the a06 main runs and the HF reference models, shipped as the
  `multilingual-snr/multilingual-snr-eval-results` parquet. Its pools and
  committed results remain runnable and are the history behind the READMEs'
  "36-sweep" sections.

If you're picking this up cold, **read [README.md](README.md) first** for what
signal, noise, decision accuracy and scaling-law error mean and how the RQs are
laid out. This file is the back-of-house memo: what is wired to what, and the
failure modes worth remembering.

---

## What's actually running

```bash
cd src/signal-and-noise
bash run_all_predictivity.sh                  # everything, from the published ladder report
SNR_LADDER_DIR=/capstor/store/cscs/swissai/infra01/msnr-ladder-report bash run_all_predictivity.sh   # cluster copy
python analysis/rq06_proxy_predictivity/analyze.py --pool predictivity_seeds   # one RQ
```

**Loaders.** `snr/download/ladder.py::load_predictivity_eval_results` pulls
`ladder_report.csv` (`hf_hub_download` into `<DATA_DIR>/ladder-report`, or
`$SNR_LADDER_DIR`) and melts it to the long schema (`model, size, L, arch,
scheme, seed, mix, step, task, kind, primary_score, tokens, compute, family`).
`snr/download/apertus.py` is the parquet loader of the 36-sweep and the
external models. `analysis/utils.build_snr_pool(pool)` picks the loader from
the pool members' `source` (`configs/models.json` → `sources.<source>.loader`),
so a script never decides by model name.

**What the ladder loader does that the parquet one never had to:**

- Per-language BPB becomes tasks: `bpb_<fineweb subset>` (`bpb_rus_Cyrl`,
  `bpb_dclm` = English), `bpb_macro`; the training loss is `train_loss`. Lower
  is better, which decision accuracy (a rank agreement) and the dispersion-based
  SNR variants do not care about. The above-random gate skips them (no chance
  level). `assign_language` maps subsets through `configs/languages.json`
  (`fineweb_iso2`).
- `mix` is the cell's design variant (`L8-schemeB-deep`, `launch_trainings.mix_label`)
  — the role the data mixture played in the 36-sweep — and `family`
  (`lm-L8-schemeB-deep-seed1904`) is the cross-size identity DA groups on.
- Diverged runs (`run__diverged`, the 90M rung) and runs short of their target
  are dropped by default; the pool flag `include_diverged` keeps them.
- **Shared checkpoint grid** (`shared_grid=True`): benchmark rows on the k/10
  grid every size was evaluated on, BPB rows on the k/20 save grid, plus the
  final checkpoint. Without it the late-window noise (`last_n = 5`) spans 50 %
  of a 20-checkpoint run and 12.5 % of a 40-checkpoint one
  (plan/1b-models.md). rq06 reads the seed replicates for the noise that does
  not depend on the window at all.
- Pool member filters apply to the frame's columns (`seeds`, `sizes`, `L`,
  `arch`, `scheme`), not to models.json names, so scheme-B cells and adopted
  off-grid seeds count whether or not the registry lists them.
- `tokens = iter × 2,064,384`; `compute = 6 × (N_non_emb + d·V) × tokens` from
  the reviewed hyperparams files (`configs.flops_params` convention).

**Pipeline order** (`run_all_predictivity.sh`): rq01 `compute_da.py` (the
truth) → rq02 `run_apertus_snr_variants.py` (22 SNR variants × bucket, joined
to the DA table) → `compare_seed_splits.py` (holdout) → per pool
`analyze_snr_variants.py`, `snr_definition_postprocess.py`,
`da_per_benchmark.py`, rq05 `analyze.py`, rq03 `analyze.py` → rq06, rq04,
the above-random gate, the rq00 curves, `report_figures/make_figures.py`. The
canonical pool (`analysis/autodoc.CANONICAL_POOL = predictivity`) runs last so
its README generators see every other pool's CSVs; generators no-op on other
pools. Outputs: `analysis/<rq>/<stage>/<pool>/`.

**Task metadata** comes from `configs/tasks.json` first: `assign_language`,
`benchmark_family`, `_is_parent_task` and the gate's option counts
(`above_random.task_n_options`) read the registered task's `language`,
`benchmark` and `n_options`; the token-parsing fallbacks only serve names the
36-sweep parquet carries unregistered (subject facets, standalone English
tasks). Registered tasks are per-language evaluations by construction, except
the `multi`-tagged aggregates (`include_base_44`), which are not parents.

---

## Models in scope

`configs/models.json` is the registry (read via `src/evals/scripts/utils/configs.py`).
`sync_models_json.py` writes one entry per grid cell — both archs, both
schemes, every seed — with `params`, `n_non_emb`, `d_model`, `vocab_size`
(the FLOPs convention) and the per-size save grid. The pools:

```
predictivity               lm-{90M…1.7B}-L{1…100}[-schemeB]-{deep,shallow}-seed1904
predictivity_seeds         … every seed (64/313 at the 175M/600M ×3 cells, 28/1797 at the 1B ×3 cells)
predictivity_seeds_train   seeds 64, 313 at 175M/600M, L ∈ {1, 2, 50, 100}
predictivity_seeds_test    seed 1904 on the same cells
seeds_*, custom_swissai_hf, external   the 36-sweep + externals (parquet loader)
```

The `snr` section of models.json is global: `small_sizes` 90M–600M,
`target_size` 1B, `da_early_fracs` 0.2/0.4/0.6/0.8, `last_n` 5,
`size_buckets` (singleton buckets for our sizes, pooled buckets for the
external models). The 36-sweep pools run with these values too — their 90M
columns are simply empty. **Do not drop `da_early_fracs` / `size_buckets`
again** (commit 56c806d did, and `analysis/utils.py` fails at import without
them).

- **size** = `90M`…`1.7B` (the ladder), `175M`…`1B` (36-sweep), native sizes
  for externals; **bucket** = `size_bucket(size)`.
- **family** = cross-size identity (`lm-L8-deep-seed1904` /
  `apertus-fwEdu30-fw270-seed1904`), attached at load; DA groups on it so the
  same design variant at two sizes is one pair.
- **seed** is a separate model in the signal pool of `predictivity_seeds`
  ("the same model measured twice" — ladder_report.md); the headline pool is
  seed 1904 only, and rq06 turns the replicates into a seed-noise column.

---

## Legacy inputs (36-sweep and external models)

`snr/download/apertus.py` reads the **local parquet** of the
`multilingual-snr/multilingual-snr-eval-results` HF dataset
(`$SNR_MULTILINGUAL_DATA_DIR`, default `<DATA_DIR>/multilingual_snr/data/`;
splits `pretraining_custom`, `pretraining_a06`, `reference_hf`, `posttraining`,
`distillation`), built on the cluster by `src/evals/scripts/build_hf_dataset.py`
from the eval_logs tree with `configs/models.json` for size/params/family/tokens.
`tokens` and `compute` are columns there (`compute = 6 × flops_params × tokens`,
with a `flops_basis` column saying whether the count is the ladder convention or
an external model's declared total). `_read_parquet` keeps the SNR columns,
strips the `-fwY` mix complement and numeric-size-sorts.

---

## Outputs

Each RQ writes next to its script: `analysis/<rq>/<stage>/<pool>/`. The
per-task tables are the persisted truth (`rq01/.../da_per_task.csv`,
`rq02/.../snr_variants_per_task.csv`); every figure and README block is
derived from them. `*.csv` / `*.png` under this directory are git-LFS
tracked (`.gitattributes`): commit regenerated results with `git lfs`
installed, and never commit outputs produced from a fixture.

---

## Cluster gotchas

- **Login node:** `clariden-ln003`. System Python is 3.6 — use a recent
  env (`miniconda3/envs/snr` or similar) when running this; the upstream
  `pyproject.toml` requires Python ≥ 3.9.
- **No outbound internet from compute nodes.** If you need to call
  `pull_predictions_from_hf` (only used for the upstream DataDecide /
  OLMo path, not Apertus), do it from the login node.
- **`/iopsstor/scratch` is the work tree.** On the cluster, this repo and
  the eval_logs that `build_hf_dataset.py` reads both live there. The
  signal-and-noise analysis itself usually runs locally on the Mac off
  the downloaded parquet.

---

## Relationship to the other packages

| Package | Path | Role |
|---|---|---|
| `signal-and-noise` (this) | `src/signal-and-noise` | SNR / decision-accuracy compute + plotting |
| `evals` | `src/evals` | eval jobs, `eval_logs`, `score_bpb.py`, W&B push, the published parquet |
| `pretrain` | `src/pretrain` | the launcher, the auto-eval watchers, `ladder_report.py` |

Flow (ladder): `pretrain/launch_trainings.py` → checkpoints →
`auto_evals_*.py` converts + evaluates every 2nd saved checkpoint, `score_bpb.py`
scores every one → `ladder_report.py --plot --publish --push-hf` →
`msnr-data/ladder-report` → `snr/download/ladder.py` → `run_all_predictivity.sh`.

---

## Hard-won bug history (don't reintroduce)

These are non-obvious gotchas that broke something in the past and are
not derivable by reading the code; some are cross-file invariants. If
you're modifying the multilingual / variant pipelines, scan this list
before editing.

### 1. `_is_language_aggregate` filter must accept 3- and 4-trailing-token forms
`analysis/rq04_smooth_subtasks/smooth_subtasks.py:_is_language_aggregate` was originally
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
`analysis/rq02_snr_definition/snr_definition_postprocess.py:top_benchmarks_per_language`
used to hardcode `da_size_col = "decision_acc_size_600M"`. Now it
follows the `size` arg (`f"decision_acc_size_{size}"`). At the default
`size=1B` the column is NaN by definition — DA-size is `small_size →
target`, so no DA-size exists for the target itself. Don't "fix" the
NaN by reverting to a hardcoded smaller size; it would mix two
different definitions in the same table.

### 10. AllenAI driver `_canonical_seed` for multi-seed corpora
For datasets with multiple seeds (DataDecide), `_canonical_seed` in
`build_allenai_variants.py` picks the most-common seed before the
SNR computation. The Apertus driver no longer needs this: each Apertus
seed is a **distinct model** in `configs/models.json`, and DA groups
on the `family` column (which strips only the size token, not the
seed). So `apertus-…-seed28` and `apertus-…-seed1797` are separate
families and the multi-seed interleaving this guard protects against
can't happen on the Apertus side. The guard is still live for the
AllenAI `random-seeds` split — don't remove it without re-checking
that path.

### 11. The gate must not NaN cells that have no chance level
`run_apertus_snr_variants.py` used to keep only `(task, size)` cells whose
above-random mask was 1; a task with no option count (per-language BPB, the
generative LAMBADA) has a NaN mask and was silently dropped from every SNR
column. The gate now removes only cells whose mask is **0** (at chance).
`mask == 0` on an Int64 mask yields `<NA>` where the mask is NA — index with
`.fillna(False)`, or pandas refuses the boolean indexer.

### 12. Reference size while the big rungs train
`snr_definition_postprocess.top_benchmarks_per_language`, rq05's `SNR_COL`
and `make_figures._ref_cols` read the configured `target_size` when its SNR
columns exist and fall back to the largest bucket in the CSV otherwise —
the ladder is analysed while 1B/1.7B are still training. The fallback is
printed / carried in a `size` column; don't hardcode `_1B` again.

### 13. `pretrain.ladder_report` is importable, but only through `src/`
rq06 reuses `_fit` and `NON_EMB` from `src/pretrain/ladder_report.py`. The
module inserts its own directory on `sys.path` to import `pretrain_progress`
→ `launch_trainings`, which read `configs/hf_wandb.json` at import; nothing
heavier happens at import time (matplotlib is lazy). Import it as
`pretrain.ladder_report` with `src/` on the path, as `analysis/utils.py`
already arranges.

---

## Legacy code (upstream DataDecide / OLMo path)

Imported by the multilingual pipeline: `snr/constants/__init__.py`
(`DATA_DIR`, `PLOT_DIR`; it pulls `constants/plot.py` and `constants/tasks.py`
with it), `snr/dataloader.py` (`get_slice`), `snr/metrics.py`,
`snr/snr_variants.py`, `snr/plot.py::config_snr_ax`,
`snr/download/{apertus,ladder,hf}.py`, `allenai_analysis/plotting/datadecide.py::plot_task_curves`.

Not imported by anything we run: `snr/ladder_wrapper.py` (needs the
`olmo-ladder` `scaling`/`fitting` packages — the paper's two-step scaling
law), `snr/metaanalysis.py`, `snr/mask_analysis.py` (instance-level IRT
masks), `snr/stats.py` (total variation / monotonicity of training curves),
`snr/snr_simple.py` (the paper's table driver; `compute_snr_small_scale`
is quoted by rq04's docstring only), `snr/autobencher/`, `snr/scripts/`,
`snr/constants/{datadecide,ladder,ladder_config.json,models,signal,smooth}.py`,
`allenai_analysis/*.ipynb` (LFS pointers) and `allenai_analysis/plotting/scaling.py`,
`analysis/PARALLEL_SESSIONS.md`, the `INSTRUCTIONS.md` files (pre-refactor
`results/` layout), `analysis/ANALYSIS_new_vs_previous.md`,
`analysis/rq04_smooth_subtasks/per_sample/` (cluster-only per-item outputs
of the 36-sweep), `posttraining.ipynb`, `notebook_guidelines.md`,
`run_all_pretraining.sh` (36-sweep driver). See the root README's
"Legacy code" for the removal proposal; nothing is deleted without the
owner's call.

---

## When upstream changes

This repo tracks `allenai/signal-and-noise`. When pulling upstream, the
local additions to watch for are:

- `analysis/` (the whole per-RQ layout, `utils.py`, `autodoc.py`, `paths.py`)
- `snr/download/apertus.py` and `snr/download/ladder.py` (local-only loaders)
- The lazy import of `run_ladder` inside `compute_scaling_law_error`
  (`snr/snr_simple.py`) — done so the Apertus path doesn't need
  `olmo-ladder` installed. If upstream re-imports it at module top, the
  Apertus run will fail with a missing-dependency error.
- The jagged-array tolerance in `compute_snr_small_scale`
  (`snr/snr_simple.py`) for half-trained mixes.

Re-apply those if a merge undoes them.
