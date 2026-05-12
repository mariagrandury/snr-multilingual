"""Generate the per-task table consumed by analyze_snr_variants.py.

For every aggregator in snr.snr_variants.AGGREGATION_FUNCTIONS, store the
three return values (signal, noise, snr) at every model size. Also store
two definitions of decision accuracy:

  size DA  — (mix, seed) ranking at <small>'s last ckpt vs the ranking at
             TARGET_SIZE's last ckpt (the upstream allenai definition,
             generalised to (mix, seed) "models").
             3 cols: decision_acc_size_<175M|350M|600M>.

  ckpt DA  — within a single size, (mix, seed) ranking at an early ckpt vs the
             same size's last ckpt. 3 early ckpts × 4 sizes = 12 cols:
             decision_acc_ckpt_<early>_<size> for early in
             CKPT_DA_EARLY_STEPS and size in ALL_SIZES.

**Signal/noise pool per size = every unique training run at that size**:
Apertus (mix, seed) runs whose ``seed`` is in ``--seeds`` *and* any
external reference models loaded from ``reference_hf`` at that size
(e.g. Qwen3-0.6B at 600M, Apertus-v1.5 at 1B). Groupby is ``model``,
so each unique training run contributes one signal datapoint and one
trailing-N noise sample. This matches AllenAI's >1B SNR_MODELS recipe
(heterogeneous open-source models at a fixed size); we extend it to
small scales by pooling Apertus seeds alongside externals.

**DA stays on the (mix, seed) Apertus axis** — DA needs models that
span both sizes for a cross-size rank comparison, and only the
controlled Apertus (mix, seed) tuples span all four sizes consistently.

``--seeds`` selects the Apertus seed pool (default: 1904);
``--include-external`` (default on) folds the reference_hf parquet
rows into the SNR signal pool; ``--out-subdir`` routes outputs to a
seed-specific folder so multiple seed pools can coexist under
``results/snr_definition/``.

The CSV is the single source of truth for analyze_snr_variants.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Put `src/` on sys.path so `evals.scripts.utils.configs` imports
# resolve via implicit namespace packages.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
from tqdm import tqdm

from evals.scripts.utils.configs import (  # noqa: E402
    add_family_column,
    expand_pool,
    load_pools,
    pool_include_external,
)
from multilingual.analyze_snr_variants import (
    _ENGLISH_ONLY_TASKS,
    assign_language,
    benchmark_family,
)
from multilingual.smooth_subtasks import _is_language_aggregate
from snr.constants import PLOT_DIR
from snr.dataloader import get_slice
from snr.download.apertus import (
    load_apertus_eval_results,
    load_reference_hf_eval_results,
)
from snr.metrics import decision_acc_fast
from snr.snr_variants import AGGREGATION_FUNCTIONS

SMALL_SIZES = ["175M", "350M", "600M"]
TARGET_SIZE = "1B"
ALL_SIZES = SMALL_SIZES + [TARGET_SIZE]
LAST_N = 5
CKPT_DA_EARLY_STEPS = [6000, 18000, 28000]
OUT_ROOT = PLOT_DIR / "snr_definition"

# Dedup set for missing-ckpt warnings — log each (size, early_step, family)
# combination at most once across the whole run, regardless of task.
_LOGGED_MISSING_CKPTS: set = set()


def _safe(fn, *args, **kwargs):
    try:
        v = fn(*args, **kwargs)
        return float(v) if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


# --- decision accuracy ------------------------------------------------------


def compute_size_decision_accuracy(
    df, task, small_size, target_size=TARGET_SIZE, model_filter=None
):
    """DA across model sizes: small_size@last vs target_size@last.

    The cross-size identity is ``family`` — the model name with only
    the size token stripped (so different mixes / seeds remain separate,
    but ``apertus-175M-fwEdu30-…-seed1904`` and the corresponding 1B
    model collapse to the same family).

    ``model_filter`` (optional) restricts the rows to a set of model
    names — passed by `run()` from `expand_pool(<pool>)`. Returns NaN if
    fewer than 2 common families survive across both sizes.
    """
    df = add_family_column(df)
    if model_filter is not None:
        df = df[df["model"].isin(model_filter)]
    scores_small = df[(df["size"] == small_size) & (df["task"] == task)]
    scores_target = df[(df["size"] == target_size) & (df["task"] == task)]
    if scores_small.empty or scores_target.empty:
        return float("nan")
    scores_small = scores_small.loc[scores_small.groupby("family")["step"].idxmax()]
    scores_target = scores_target.loc[scores_target.groupby("family")["step"].idxmax()]
    keys_small = set(scores_small["family"])
    keys_target = set(scores_target["family"])
    common = sorted(keys_small & keys_target)
    if len(common) < 2:
        return float("nan")
    s = scores_small.set_index("family").loc[common, "primary_score"]
    t = scores_target.set_index("family").loc[common, "primary_score"]
    return decision_acc_fast(s.to_numpy(), t.to_numpy())


def compute_ckpt_decision_accuracy(df, task, size, early_step, model_filter=None):
    """DA within a single size: ``family`` ranking at exactly
    ``early_step`` vs the same family's max-step ckpt.

    Within a single size, each family has exactly one model — so
    the grouping is essentially per-model. A missing ckpt is logged
    once per ``(size, early_step, family)`` combination across
    the whole run. If fewer than 2 families survive, returns NaN.

    ``model_filter`` (optional) restricts the rows to a set of model
    names — passed by `run()` from `expand_pool(<pool>)`.
    """
    df = add_family_column(df)
    if model_filter is not None:
        df = df[df["model"].isin(model_filter)]
    scores = df[(df["size"] == size) & (df["task"] == task)]
    if scores.empty:
        return float("nan")
    early, late, keys = [], [], []
    for fam, g in scores.groupby("family"):
        g_early = g[g["step"] == early_step]
        if g_early.empty:
            key = (size, early_step, fam)
            if key not in _LOGGED_MISSING_CKPTS:
                _LOGGED_MISSING_CKPTS.add(key)
                print(
                    f"  ckpt-DA: no row at step={early_step} for "
                    f"size={size} family={fam} "
                    f"(first seen on task={task}) — skipped"
                )
            continue
        max_row = g.loc[g["step"].idxmax()]
        early.append(float(g_early["primary_score"].iloc[0]))
        late.append(float(max_row["primary_score"]))
        keys.append(fam)
    if len(keys) < 2:
        return float("nan")
    return decision_acc_fast(np.asarray(early), np.asarray(late))


# --- per-model arrays for snr_variants --------------------------------------


def per_model_inputs(df, task, size, last_n=LAST_N):
    """Build the four per-model arrays expected by snr_variants aggregators.

    Each unique value of the ``model`` column is a separate training
    run — so the signal pool naturally combines Apertus (mix, seed) runs
    (one model name per tuple) with external reference models (one
    model name per HF release). ``df`` is assumed to already be
    filtered to the desired seed pool / external inclusion.

    Mirrors analysis/snr_variants.ipynb cells 5+7:
      step_noise         = per-model std of the last `last_n` ckpts
      data_scores        = per-model final-ckpt score
      data_noise         = cross-model std of `data_scores`, broadcast as
                           a constant array of the same length
      data_scores_last_n = per-model mean of the last `last_n` ckpts

    Models with fewer than 2 ckpts are dropped (they can't contribute
    step_noise — e.g. a single-revision external like Apertus-70B-2509).
    We require ≥ 2 surviving models overall.
    """
    sub = df[(df["size"] == size) & (df["task"] == task)]
    if sub.empty:
        return None
    sub = sub.sort_values("step")
    grouped = sub.groupby("model")["primary_score"].apply(list)
    last_arrays = [np.asarray(s[-last_n:], dtype=float) for s in grouped]
    last_arrays = [a for a in last_arrays if len(a) >= 2]
    if len(last_arrays) < 2:
        return None
    step_noise = np.array([np.std(a) for a in last_arrays])
    data_scores = np.array([a[-1] for a in last_arrays])
    data_scores_last_n = np.array([a.mean() for a in last_arrays])
    data_noise = np.full_like(data_scores, np.std(data_scores))
    return step_noise, data_scores, data_noise, data_scores_last_n


def variant_signal_noise_snr(inputs, agg_func):
    if inputs is None:
        return float("nan"), float("nan"), float("nan")
    try:
        signal, noise, snr = agg_func(*inputs)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    out = []
    for v in (signal, noise, snr):
        try:
            f = float(v)
            out.append(f if np.isfinite(f) else float("nan"))
        except Exception:
            out.append(float("nan"))
    return out[0], out[1], out[2]


def variant_key(func_dict):
    """Stable column-name token, e.g. 'rel_std' from 'rel_std_snr'."""
    name = func_dict["func"].__name__
    return name[:-4] if name.endswith("_snr") else name


# --- variants metadata table ------------------------------------------------


def variants_definitions_df() -> pd.DataFrame:
    """One row per aggregator describing what its signal/noise/snr mean."""
    rows = []
    for fd in AGGREGATION_FUNCTIONS:
        rows.append(
            {
                "variant": variant_key(fd),
                "title": fd["title"],
                "latex": fd["latex"],
                "signal_label": fd["signal_xlabel"],
                "noise_label": fd["noise_xlabel"],
                "snr_label": fd["snr_xlabel"],
            }
        )
    return pd.DataFrame(rows).set_index("variant")


def write_variants_definitions(out_dir: Path) -> Path:
    """Write the metadata table to CSV and print a readable version."""
    df_def = variants_definitions_df()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "snr_variants_definitions.csv"
    df_def.to_csv(csv_path)
    # Readable view (markdown table; falls back to .to_string() if tabulate is missing).
    try:
        printable = df_def[["title", "latex", "snr_label"]].to_markdown()
    except Exception:
        printable = df_def[["title", "latex", "snr_label"]].to_string()
    print("\nSNR variant definitions:")
    print(printable)
    return csv_path


# --- driver -----------------------------------------------------------------


def _is_parent_task(task: str) -> bool:
    """Match the cluster's ``aggregate_parents`` semantics: keep one row per
    "real" evaluation, dropping the per-(lang, subject) facets that the
    parquet ships alongside their language-aggregate parents.

    Two branches:
      - English standalone tasks (``mmlu``, ``hellaswag``, …): the explicit
        list in ``_ENGLISH_ONLY_TASKS``.
      - Multilingual per-language aggregates: the same
        ``_is_language_aggregate`` rule used in
        ``multilingual.smooth_subtasks.collect_multilingual_families``.
    """
    if task in _ENGLISH_ONLY_TASKS:
        return True
    return _is_language_aggregate(task, benchmark_family(task))


def build_snr_pool(pool: str) -> pd.DataFrame:
    """SNR signal-pool dataframe for the named pool. Apertus rows are
    filtered to the pool's `members` (resolved via configs/models.json
    via expand_pool). When the pool sets `include_external=true`, every
    reference_hf row joins the pool (HF refs have no `seed` and only live
    at their native sizes — the per_model groupby handles them via the
    model name)."""
    pool_models = set(expand_pool(pool))
    df_a = load_apertus_eval_results()
    df_a = df_a[df_a["model"].isin(pool_models)].copy()
    frames = [df_a]
    if pool_include_external(pool):
        try:
            df_e = load_reference_hf_eval_results()
            if not df_e.empty:
                frames.append(df_e)
        except FileNotFoundError:
            print("  (no reference_hf parquet found — SNR pool is Apertus only)")
    return pd.concat(frames, ignore_index=True)


def run(pool: str, out_dir: Path):
    df_apertus = load_apertus_eval_results()  # for DA (cross-size axis)
    df_pool = build_snr_pool(pool)  # for SNR signal/noise
    all_tasks = sorted(df_pool["task"].unique())
    tasks = [t for t in all_tasks if _is_parent_task(t)]

    pool_models = set(expand_pool(pool))
    da_models = df_apertus["model"].isin(pool_models)
    pool_n_models = df_pool.groupby("size")["model"].nunique().to_dict()
    apertus_n_families = df_apertus[da_models].groupby("size")["family"].nunique().to_dict()
    print(
        f"Pool '{pool}': {len(pool_models)} Apertus model(s); "
        f"include_external={pool_include_external(pool)}"
    )
    print(
        f"Loaded {len(df_apertus):,} Apertus rows + "
        f"{len(df_pool) - da_models.sum():,} external rows | "
        f"{len(tasks)} parent tasks (filtered from {len(all_tasks)} total)"
    )
    print(f"  Total SNR-pool models per size: {pool_n_models}")
    print(f"  Apertus DA-axis families per size: {apertus_n_families}")

    write_variants_definitions(out_dir)

    rows = []
    for task in tqdm(tasks, desc="Tasks"):
        row = {"task": task}

        # DA stays on the Apertus axis (cross-size families). Pool's model
        # list restricts which families participate; `model_filter` below
        # passes the names to the DA helpers' internal filter.
        for s in SMALL_SIZES:
            row[f"decision_acc_size_{s}"] = _safe(
                compute_size_decision_accuracy,
                df_apertus,
                task,
                s,
                model_filter=pool_models,
            )
        for early in CKPT_DA_EARLY_STEPS:
            for s in ALL_SIZES:
                row[f"decision_acc_ckpt_{early}_{s}"] = _safe(
                    compute_ckpt_decision_accuracy,
                    df_apertus,
                    task,
                    s,
                    early,
                    model_filter=pool_models,
                )

        size_inputs = {s: per_model_inputs(df_pool, task, s) for s in ALL_SIZES}
        for fd in AGGREGATION_FUNCTIONS:
            key = variant_key(fd)
            for s in ALL_SIZES:
                sig, noi, snr = variant_signal_noise_snr(size_inputs[s], fd["func"])
                row[f"signal_{key}_{s}"] = sig
                row[f"noise_{key}_{s}"] = noi
                row[f"snr_{key}_{s}"] = snr
        rows.append(row)

    out = pd.DataFrame(rows).set_index("task").sort_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "snr_variants_per_task.csv"
    out.to_csv(csv_path)
    n_size = len(SMALL_SIZES)
    n_ckpt = len(CKPT_DA_EARLY_STEPS) * len(ALL_SIZES)
    print(f"\nWrote CSV → {csv_path}")
    print(
        f"  {len(out)} tasks × {len(out.columns)} columns "
        f"({len(AGGREGATION_FUNCTIONS)} variants × {len(ALL_SIZES)} sizes × 3 stats "
        f"+ {n_size} size-DA + {n_ckpt} ckpt-DA)"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pool",
        required=True,
        help="Pool name from configs/models.json (e.g. "
        "seeds_1904, seeds_28_1797, seeds_28_1797_1904, "
        "pretraining_a06).",
    )
    p.add_argument(
        "--out-subdir",
        default=None,
        help="Subdir under results/snr_definition/ " "(default: <pool>).",
    )
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; " f"available: {sorted(load_pools().keys())}")
    out_dir = OUT_ROOT / (args.out_subdir or args.pool)
    run(pool=args.pool, out_dir=out_dir)


if __name__ == "__main__":
    main()
