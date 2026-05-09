"""Generate the per-task table consumed by analyze_snr_variants.py.

For every aggregator in snr.snr_variants.AGGREGATION_FUNCTIONS, store the
three return values (signal, noise, snr) at every model size. Also store
two definitions of decision accuracy:

  size DA  — mix ranking at <small>'s last ckpt vs the ranking at
             TARGET_SIZE's last ckpt (the upstream allenai definition).
             3 cols: decision_acc_size_<175M|350M|600M>.

  ckpt DA  — within a single size, mix ranking at an early ckpt vs the
             same size's last ckpt. 3 early ckpts × 4 sizes = 12 cols:
             decision_acc_ckpt_<early>_<size> for early in
             CKPT_DA_EARLY_STEPS and size in ALL_SIZES. The "late" point
             is each mix's max step (not a hard 50000) so half-trained
             mixes (e.g. 1B-fwEdu90 with max=38000) still contribute.

The CSV is the single source of truth for analyze_snr_variants.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pandas as pd
from tqdm import tqdm

from multilingual.analyze_snr_variants import (
    _ENGLISH_ONLY_TASKS, assign_language, benchmark_family,
)
from multilingual.smooth_subtasks import _is_language_aggregate
from snr.constants import PLOT_DIR
from snr.dataloader import get_slice
from snr.download.apertus import load_apertus_eval_results
from snr.metrics import decision_acc_fast
from snr.snr_variants import AGGREGATION_FUNCTIONS

SMALL_SIZES = ["175M", "350M", "600M"]
TARGET_SIZE = "1B"
ALL_SIZES = SMALL_SIZES + [TARGET_SIZE]
LAST_N = 5
CKPT_DA_EARLY_STEPS = [6000, 18000, 28000]
SEED = 1904  # Apertus pretrains all share this seed.
OUT_DIR = PLOT_DIR / "snr_definition"

# Dedup set for missing-ckpt warnings — log each (size, early_step, mix)
# combination at most once across the whole run, regardless of task.
_LOGGED_MISSING_CKPTS: set = set()


def _safe(fn, *args, **kwargs):
    try:
        v = fn(*args, **kwargs)
        return float(v) if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


# --- decision accuracy ------------------------------------------------------

def compute_size_decision_accuracy(df, task, small_size, target_size=TARGET_SIZE,
                                   seed=None):
    """DA across model sizes: small_size@last vs target_size@last (upstream).

    Pass ``seed`` to pin the slice to a single training seed — required
    on multi-seed corpora (e.g. AllenAI DataDecide); harmless on the
    single-seed Apertus corpus.
    """
    scores_small = get_slice(df, size=small_size, task=task, seed=seed)
    scores_target = get_slice(df, size=target_size, task=task, seed=seed)
    if scores_small.empty or scores_target.empty:
        return float("nan")
    scores_small = scores_small.loc[scores_small.groupby("mix")["step"].idxmax()]
    scores_target = scores_target.loc[scores_target.groupby("mix")["step"].idxmax()]
    common = sorted(set(scores_small["mix"]) & set(scores_target["mix"]))
    if len(common) < 2:
        return float("nan")
    s = scores_small.set_index("mix").loc[common, "primary_score"]
    t = scores_target.set_index("mix").loc[common, "primary_score"]
    return decision_acc_fast(s.to_numpy(), t.to_numpy())


def compute_ckpt_decision_accuracy(df, task, size, early_step, seed=None):
    """DA within a single size: mix ranking at exactly ``early_step`` vs the
    same size's max-step ckpt.

    Requires the *exact* early-step row per mix — mixes lacking that
    step are skipped silently. Set ``CKPT_DA_EARLY_STEPS`` to values
    that match the eval cadence (every 6000 iters for the current
    Apertus runs); a missing ckpt is logged once per
    ``(size, early_step, mix)`` combination across the whole run. If
    fewer than 2 mixes survive, returns NaN.

    Pass ``seed`` to pin the slice to a single training seed.
    """
    scores = get_slice(df, size=size, task=task, seed=seed)
    if scores.empty:
        return float("nan")
    early, late, mixes = [], [], []
    for mix, g in scores.groupby("mix"):
        g_early = g[g["step"] == early_step]
        if g_early.empty:
            key = (size, early_step, mix)
            if key not in _LOGGED_MISSING_CKPTS:
                _LOGGED_MISSING_CKPTS.add(key)
                print(f"  ckpt-DA: no row at step={early_step} for "
                      f"size={size} mix={mix} (first seen on task={task}) — skipped")
            continue
        max_row = g.loc[g["step"].idxmax()]
        early.append(float(g_early["primary_score"].iloc[0]))
        late.append(float(max_row["primary_score"]))
        mixes.append(mix)
    if len(mixes) < 2:
        return float("nan")
    return decision_acc_fast(np.asarray(early), np.asarray(late))


# --- per-mix arrays for snr_variants ----------------------------------------

def per_mix_inputs(df, task, size, last_n=LAST_N, seed=None):
    """Build the four per-mix arrays expected by snr_variants aggregators.

    Mirrors analysis/snr_variants.ipynb cells 5+7:
      step_noise         = per-mix std of the last `last_n` ckpts
      data_scores        = per-mix final-ckpt score
      data_noise         = cross-mix std of `data_scores`, broadcast as a
                           constant array of the same length
      data_scores_last_n = per-mix mean of the last `last_n` ckpts

    Pass ``seed`` to pin the slice to a single training seed — required
    on multi-seed corpora (else multi-seed rows interleave per step and
    the trailing-``last_n`` window mixes seeds). Mixes with fewer than
    2 ckpts are dropped (they can't contribute step_noise) and we
    require ≥ 2 surviving mixes overall.
    """
    scores_df = get_slice(df, size=size, task=task, seed=seed).sort_values("step")
    if scores_df.empty:
        return None
    grouped = scores_df.groupby("mix")["primary_score"].apply(list)
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
        rows.append({
            "variant": variant_key(fd),
            "title": fd["title"],
            "latex": fd["latex"],
            "signal_label": fd["signal_xlabel"],
            "noise_label": fd["noise_xlabel"],
            "snr_label": fd["snr_xlabel"],
        })
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


def run():
    df = load_apertus_eval_results()
    all_tasks = sorted(df["task"].unique())
    tasks = [t for t in all_tasks if _is_parent_task(t)]
    print(f"Loaded {len(df):,} rows | {df['model'].nunique()} models | "
          f"{len(tasks)} parent tasks (filtered from {len(all_tasks)} total)")

    write_variants_definitions(OUT_DIR)

    rows = []
    for task in tqdm(tasks, desc="Tasks"):
        row = {"task": task}

        for s in SMALL_SIZES:
            row[f"decision_acc_size_{s}"] = _safe(
                compute_size_decision_accuracy, df, task, s, seed=SEED
            )
        for early in CKPT_DA_EARLY_STEPS:
            for s in ALL_SIZES:
                row[f"decision_acc_ckpt_{early}_{s}"] = _safe(
                    compute_ckpt_decision_accuracy, df, task, s, early, seed=SEED
                )

        size_inputs = {s: per_mix_inputs(df, task, s, seed=SEED) for s in ALL_SIZES}
        for fd in AGGREGATION_FUNCTIONS:
            key = variant_key(fd)
            for s in ALL_SIZES:
                sig, noi, snr = variant_signal_noise_snr(size_inputs[s], fd["func"])
                row[f"signal_{key}_{s}"] = sig
                row[f"noise_{key}_{s}"] = noi
                row[f"snr_{key}_{s}"] = snr
        rows.append(row)

    out = pd.DataFrame(rows).set_index("task").sort_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "snr_variants_per_task.csv"
    out.to_csv(csv_path)
    n_size = len(SMALL_SIZES)
    n_ckpt = len(CKPT_DA_EARLY_STEPS) * len(ALL_SIZES)
    print(f"\nWrote CSV → {csv_path}")
    print(f"  {len(out)} tasks × {len(out.columns)} columns "
          f"({len(AGGREGATION_FUNCTIONS)} variants × {len(ALL_SIZES)} sizes × 3 stats "
          f"+ {n_size} size-DA + {n_ckpt} ckpt-DA)")


if __name__ == "__main__":
    run()
