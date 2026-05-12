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
import re
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
from snr.download.apertus import (
    load_apertus_eval_results, load_reference_hf_eval_results,
)
from snr.metrics import decision_acc_fast
from snr.snr_variants import AGGREGATION_FUNCTIONS

SMALL_SIZES = ["175M", "350M", "600M"]
TARGET_SIZE = "1B"
ALL_SIZES = SMALL_SIZES + [TARGET_SIZE]
LAST_N = 5
CKPT_DA_EARLY_STEPS = [6000, 18000, 28000]
DEFAULT_SEEDS = [1904]
OUT_ROOT = PLOT_DIR / "snr_definition"

# Dedup set for missing-ckpt warnings — log each (size, early_step, mix)
# combination at most once across the whole run, regardless of task.
_LOGGED_MISSING_CKPTS: set = set()


# --- model_family: cross-size identity helper -------------------------------

# Sizes we know about. The size token is stripped from a model name to
# produce a `model_family` that crosses sizes. We deliberately keep seed
# (and any other axis like mix) IN the family ID — they encode different
# training runs that we want to track separately across sizes.
_SIZE_TOKENS = (
    "175M", "350M", "600M", "1B",  # Apertus
    "150M", "300M", "750M",        # AllenAI DataDecide
    "190M",                         # AllenAI ladder
    "3B", "7B", "8B", "13B", "32B", "70B",  # external HF
)


def _strip_size_from_name(model: str, size: str) -> str:
    """Drop the ``size`` token (with adjacent dashes) from a model name.

    Returns a cross-size identity for the same training run. We do **not**
    strip seeds, mixes, or other recipe tokens — so `apertus-175M-fwEdu30-
    fw270-seed28` and `...-seed1797` produce distinct families. Examples:
        apertus-175M-fwEdu30-fw270-seed1904 → apertus-fwEdu30-fw270-seed1904
        allenai/DataDecide-c4-150M          → allenai/DataDecide-c4
        SmolLM3-3B-Base                     → SmolLM3-Base
        Olmo-3-1025-7B                      → Olmo-3-1025
    """
    if not isinstance(model, str) or not isinstance(size, str):
        return model
    if size not in _SIZE_TOKENS:
        return model
    # Match `-<size>-`, `-<size>$`, `<size>-`, or `<size>$` as a token
    # (require a non-word boundary on each side to avoid partial matches
    # like `100M` swallowing `0M`).
    pattern = re.compile(rf"(?:^|(?<=[-/])){re.escape(size)}(?=[-/]|$)")
    new = pattern.sub("", model)
    # Clean up resulting `--` or trailing/leading `-`.
    new = re.sub(r"-{2,}", "-", new).strip("-/")
    return new or model


def add_model_family(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``model_family`` column to ``df``. Idempotent."""
    if "model_family" in df.columns:
        return df
    df = df.copy()
    df["model_family"] = [
        _strip_size_from_name(m, s) for m, s in zip(df["model"], df["size"])
    ]
    return df


def _safe(fn, *args, **kwargs):
    try:
        v = fn(*args, **kwargs)
        return float(v) if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


# --- decision accuracy ------------------------------------------------------

def compute_size_decision_accuracy(df, task, small_size, target_size=TARGET_SIZE,
                                   seeds=None):
    """DA across model sizes: small_size@last vs target_size@last.

    The cross-size identity is ``model_family`` — the model name with
    only the size token stripped (so different mixes / seeds remain
    separate, but ``apertus-175M-fwEdu30-…-seed1904`` and the
    corresponding 1B model collapse to the same family).

    ``seeds`` restricts the Apertus side of the pool to a list of
    training seeds (rows whose ``seed`` is in the list, plus rows whose
    seed is NaN — those are external models with no seed column).
    Returns NaN if fewer than 2 common model families survive across
    both sizes.
    """
    df = add_model_family(df)
    if seeds is not None:
        seeds_set = set(int(s) for s in seeds)
        keep = df["seed"].isna() | df["seed"].isin(seeds_set)
        df = df[keep]
    scores_small = df[(df["size"] == small_size) & (df["task"] == task)]
    scores_target = df[(df["size"] == target_size) & (df["task"] == task)]
    if scores_small.empty or scores_target.empty:
        return float("nan")
    scores_small = scores_small.loc[
        scores_small.groupby("model_family")["step"].idxmax()
    ]
    scores_target = scores_target.loc[
        scores_target.groupby("model_family")["step"].idxmax()
    ]
    keys_small = set(scores_small["model_family"])
    keys_target = set(scores_target["model_family"])
    common = sorted(keys_small & keys_target)
    if len(common) < 2:
        return float("nan")
    s = scores_small.set_index("model_family").loc[common, "primary_score"]
    t = scores_target.set_index("model_family").loc[common, "primary_score"]
    return decision_acc_fast(s.to_numpy(), t.to_numpy())


def compute_ckpt_decision_accuracy(df, task, size, early_step, seeds=None):
    """DA within a single size: ``model_family`` ranking at exactly
    ``early_step`` vs the same family's max-step ckpt.

    Within a single size, each model_family has exactly one model — so
    the grouping is essentially per-model. A missing ckpt is logged
    once per ``(size, early_step, model_family)`` combination across
    the whole run. If fewer than 2 families survive, returns NaN.
    """
    df = add_model_family(df)
    if seeds is not None:
        seeds_set = set(int(s) for s in seeds)
        keep = df["seed"].isna() | df["seed"].isin(seeds_set)
        df = df[keep]
    scores = df[(df["size"] == size) & (df["task"] == task)]
    if scores.empty:
        return float("nan")
    early, late, keys = [], [], []
    for fam, g in scores.groupby("model_family"):
        g_early = g[g["step"] == early_step]
        if g_early.empty:
            key = (size, early_step, fam)
            if key not in _LOGGED_MISSING_CKPTS:
                _LOGGED_MISSING_CKPTS.add(key)
                print(f"  ckpt-DA: no row at step={early_step} for "
                      f"size={size} family={fam} "
                      f"(first seen on task={task}) — skipped")
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


def _seeds_subdir(seeds: list[int]) -> str:
    """``[1904] → 'seeds_1904'``; ``[28, 1797] → 'seeds_28_1797'``."""
    return "seeds_" + "_".join(str(s) for s in sorted(seeds))


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


def build_snr_pool(seeds: list[int], include_external: bool) -> pd.DataFrame:
    """SNR signal-pool dataframe: Apertus rows filtered to ``seeds`` plus
    (optionally) every reference_hf row. Externals don't have a ``seed``
    so they are gated only by ``include_external``."""
    df_a = load_apertus_eval_results()
    df_a = df_a[df_a["seed"].isin(seeds)].copy()
    frames = [df_a]
    if include_external:
        try:
            df_e = load_reference_hf_eval_results()
            if not df_e.empty:
                frames.append(df_e)
        except FileNotFoundError:
            print("  (no reference_hf parquet found — SNR pool is Apertus only)")
    return pd.concat(frames, ignore_index=True)


def run(seeds: list[int], out_dir: Path, include_external: bool = True):
    df_apertus = load_apertus_eval_results()  # for DA (cross-size axis)
    df_pool = build_snr_pool(seeds, include_external)  # for SNR signal/noise
    all_tasks = sorted(df_pool["task"].unique())
    tasks = [t for t in all_tasks if _is_parent_task(t)]

    pool_n_models = df_pool.groupby("size")["model"].nunique().to_dict()
    apertus_n_units = df_apertus[df_apertus["seed"].isin(seeds)].groupby("size")[
        ["mix", "seed"]
    ].apply(lambda g: g.drop_duplicates().shape[0]).to_dict()
    print(f"Loaded {len(df_apertus):,} Apertus rows + "
          f"{len(df_pool) - len(df_apertus[df_apertus['seed'].isin(seeds)]):,} "
          f"external rows | {len(tasks)} parent tasks "
          f"(filtered from {len(all_tasks)} total)")
    print(f"Seeds: {seeds}  include_external: {include_external}")
    print(f"  Total SNR-pool models per size: {pool_n_models}")
    print(f"  Apertus (mix, seed) DA-axis units per size: {apertus_n_units}")

    write_variants_definitions(out_dir)

    rows = []
    for task in tqdm(tasks, desc="Tasks"):
        row = {"task": task}

        for s in SMALL_SIZES:
            row[f"decision_acc_size_{s}"] = _safe(
                compute_size_decision_accuracy, df_apertus, task, s, seeds=seeds
            )
        for early in CKPT_DA_EARLY_STEPS:
            for s in ALL_SIZES:
                row[f"decision_acc_ckpt_{early}_{s}"] = _safe(
                    compute_ckpt_decision_accuracy,
                    df_apertus, task, s, early, seeds=seeds
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
    print(f"  {len(out)} tasks × {len(out.columns)} columns "
          f"({len(AGGREGATION_FUNCTIONS)} variants × {len(ALL_SIZES)} sizes × 3 stats "
          f"+ {n_size} size-DA + {n_ckpt} ckpt-DA)")


def _parse_seeds(value: str) -> list[int]:
    return sorted({int(x) for x in value.split(",") if x.strip()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=_parse_seeds, default=DEFAULT_SEEDS,
                   help="Comma-separated list of training seeds to include "
                        "in the Apertus SNR/DA pool (default: 1904).")
    p.add_argument("--no-external", action="store_true",
                   help="Disable the external reference_hf parquet "
                        "rows in the SNR signal pool.")
    p.add_argument("--out-subdir", default=None,
                   help="Subdir under results/snr_definition/ (default: "
                        "'seeds_<a>_<b>_...' derived from --seeds).")
    args = p.parse_args()
    out_subdir = args.out_subdir or _seeds_subdir(args.seeds)
    out_dir = OUT_ROOT / out_subdir
    run(seeds=args.seeds, out_dir=out_dir,
        include_external=not args.no_external)


if __name__ == "__main__":
    main()
