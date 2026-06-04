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
    bucket_order,
    expand_pool,
    load_pools,
    load_snr_params,
    pool_include_external,
    size_bucket,
    stage_external_models,
)
from multilingual.analyze_snr_variants import (
    _ENGLISH_ONLY_TASKS,
    assign_language,
    benchmark_family,
)
from multilingual.above_random import SIZES as AR_SIZES, scores_and_mask
from multilingual.smooth_subtasks import _is_language_aggregate
from snr.constants import PLOT_DIR
from snr.dataloader import get_slice
from snr.download.apertus import (
    load_a06_eval_results,
    load_apertus_eval_results,
    load_distillation_eval_results,
    load_reference_hf_eval_results,
)
from snr.metrics import decision_acc_fast
from snr.snr_variants import AGGREGATION_FUNCTIONS

# SNR analysis params — single source of truth in configs/models.json.
# The signal/noise + DA size axis is the *bucket* (size_bucket()), so nearby
# large sizes (7B/8B → "7-9B") pool to ≥2 models. The custom small sizes are
# singleton buckets, so SMALL_SIZES / TARGET_SIZE double as bucket labels for
# the core holdout size-DA.
_SNR = load_snr_params()
SMALL_SIZES = _SNR["small_sizes"]
TARGET_SIZE = _SNR["target_size"]
LAST_N = _SNR["last_n"]
# ckpt-DA early checkpoints are picked per-model at these fractions of each
# model's own max step (absolute custom iters wouldn't match external/a06/
# distill trajectories).
CKPT_DA_EARLY_FRACS = _SNR["da_early_fracs"]
OUT_ROOT = PLOT_DIR / "snr_definition"


def _frac_label(frac: float) -> str:
    """0.12 → 'f12' (stable ckpt-DA column token)."""
    return f"f{int(round(frac * 100))}"


# Dedup set for missing-ckpt warnings — log each (bucket, frac, family)
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
    """DA across size buckets: small_bucket@last vs target_bucket@last.

    Operates on the ``bucket`` column (set by `run()` via `size_bucket`),
    so ``small_size`` / ``target_size`` are bucket labels. The cross-size
    identity is ``family`` (the model name with only the size token
    stripped), so a family present at both buckets contributes one pair.

    ``model_filter`` (optional) restricts the rows to a set of model
    names. Returns NaN if fewer than 2 common families survive across
    both buckets.
    """
    df = add_family_column(df)
    if model_filter is not None:
        df = df[df["model"].isin(model_filter)]
    scores_small = df[(df["bucket"] == small_size) & (df["task"] == task)]
    scores_target = df[(df["bucket"] == target_size) & (df["task"] == task)]
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


def compute_ckpt_decision_accuracy(df, task, bucket, early_frac, model_filter=None):
    """DA within a size bucket: ``family`` ranking at an *early* ckpt vs the
    same family's max-step ckpt.

    The early ckpt is chosen per model as the checkpoint whose step is
    closest to ``early_frac × (that model's own max step)`` — a relative
    fraction rather than an absolute iter, so external / a06 / distillation
    trajectories (whose step scales differ from the custom megatron iters)
    participate. A family with only one checkpoint (single-ckpt HF refs)
    has no distinct early ckpt and is logged once per ``(bucket, frac,
    family)`` and skipped. If fewer than 2 families survive, returns NaN.

    ``model_filter`` (optional) restricts the rows to a set of model names.
    """
    df = add_family_column(df)
    if model_filter is not None:
        df = df[df["model"].isin(model_filter)]
    scores = df[(df["bucket"] == bucket) & (df["task"] == task)]
    if scores.empty:
        return float("nan")
    early, late, keys = [], [], []
    for fam, g in scores.groupby("family"):
        g = g.sort_values("step")
        max_step = g["step"].max()
        g_pre = g[g["step"] < max_step]  # candidate early ckpts (exclude the last)
        if g_pre.empty:
            key = (bucket, early_frac, fam)
            if key not in _LOGGED_MISSING_CKPTS:
                _LOGGED_MISSING_CKPTS.add(key)
                print(
                    f"  ckpt-DA: only one ckpt for bucket={bucket} "
                    f"family={fam} (first seen on task={task}) — skipped"
                )
            continue
        target_step = early_frac * max_step
        early_row = g_pre.iloc[(g_pre["step"] - target_step).abs().argmin()]
        max_row = g.loc[g["step"].idxmax()]
        early.append(float(early_row["primary_score"]))
        late.append(float(max_row["primary_score"]))
        keys.append(fam)
    if len(keys) < 2:
        return float("nan")
    return decision_acc_fast(np.asarray(early), np.asarray(late))


# --- per-model arrays for snr_variants --------------------------------------


def per_model_inputs(df, task, size, last_n=LAST_N):
    """Build the four per-model arrays expected by snr_variants aggregators.

    ``size`` is a *bucket* label and rows are selected on the ``bucket``
    column, so nearby large sizes (e.g. 7B + 8B → "7-9B") pool together.
    Each unique value of the ``model`` column is a separate training run —
    so the signal pool combines Apertus (mix, seed) runs with external /
    a06 / distillation models present in the bucket. ``df`` is assumed to
    already be filtered to the desired pool / external inclusion and to
    carry a ``bucket`` column.

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
    sub = df[(df["bucket"] == size) & (df["task"] == task)]
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
    external pretraining row joins the pool — reference_hf (HF/Swiss-AI
    refs), a06 (apertus3 main runs) and distillation (ap-from8b). These
    have no `seed` and only live at their native sizes (so they never
    join the custom cross-size DA axis), but per_model_inputs groups by
    model name, so each adds a fresh signal/noise data point at its size.

    External rows are restricted to models declared at the pool's stage, so
    instruct/posttraining checkpoints in the reference_hf parquet never leak
    into the pretraining pool."""
    pool_models = set(expand_pool(pool))
    df_a = load_apertus_eval_results()
    df_a = df_a[df_a["model"].isin(pool_models)].copy()
    frames = [df_a]
    if pool_include_external(pool):
        stage = load_pools()[pool].get("stage", "pretraining")
        allowed = stage_external_models(stage)
        for loader in (
            load_reference_hf_eval_results,
            load_a06_eval_results,
            load_distillation_eval_results,
        ):
            try:
                df_e = loader()
            except FileNotFoundError:
                continue
            df_e = df_e[df_e["model"].isin(allowed)]
            if not df_e.empty:
                frames.append(df_e)
    return pd.concat(frames, ignore_index=True)


def _scaling_da_pairs(df_pool) -> list[tuple[str, str]]:
    """Ordered (small_bucket, target_bucket) pairs (small < target by bucket
    order) with ≥2 families present at both buckets — the cross-size pairs
    where decision accuracy is computable. Excludes the canonical
    small→TARGET_SIZE pairs (emitted separately as decision_acc_size_<small>)."""
    present = [b for b in bucket_order() if b in set(df_pool["bucket"].dropna())]
    fams = {b: set(df_pool[df_pool["bucket"] == b]["family"]) for b in present}
    pairs = []
    for i, sb in enumerate(present):
        for tb in present[i + 1:]:
            if sb in SMALL_SIZES and tb == TARGET_SIZE:
                continue
            if len(fams[sb] & fams[tb]) >= 2:
                pairs.append((sb, tb))
    return pairs


def run(pool: str, out_dir: Path):
    # One pooled dataframe drives both SNR signal/noise AND decision accuracy:
    # custom-in-pool rows + folded-in externals (reference_hf / a06 /
    # distillation when include_external). The size axis is the *bucket*
    # (size_bucket): custom small sizes are singletons; larger sizes pool so
    # each bucket has ≥2 models. Cross-bucket DA groups on `family`.
    df_pool = build_snr_pool(pool)
    df_pool["bucket"] = df_pool["size"].map(size_bucket)
    all_tasks = sorted(df_pool["task"].unique())
    tasks = [t for t in all_tasks if _is_parent_task(t)]

    pool_buckets = [b for b in bucket_order() if b in set(df_pool["bucket"].dropna())]
    scaling_pairs = _scaling_da_pairs(df_pool)

    pool_models = set(expand_pool(pool))
    all_models = set(df_pool["model"])
    is_external = ~df_pool["model"].isin(pool_models)

    # Above-random gate (raw-metric competence check): keep a (task, custom-size)
    # SNR cell only if its mean score beats chance by the margin. Random cells
    # carry no usable signal, so their signal/noise/snr are NaN'd here — the gate
    # propagates to every analysis that reads this CSV. External-size buckets
    # (3B, 7-9B, …) aren't gated (no custom random baseline at those sizes).
    _, _ar_mask, _ = scores_and_mask(df_pool[df_pool["model"].isin(pool_models)])
    above_random = {(t, s) for s in AR_SIZES if s in _ar_mask.columns
                    for t in _ar_mask.index[_ar_mask[s] == 1]}
    pool_n_models = df_pool.groupby("bucket")["model"].nunique().to_dict()
    da_n_families = df_pool.groupby("bucket")["family"].nunique().to_dict()
    print(
        f"Pool '{pool}': {len(all_models & pool_models)} custom + "
        f"{len(all_models - pool_models)} external model(s); "
        f"include_external={pool_include_external(pool)}"
    )
    print(
        f"Loaded {len(df_pool):,} rows ({int(is_external.sum()):,} external) | "
        f"{len(tasks)} parent tasks (filtered from {len(all_tasks)} total)"
    )
    print(f"  Buckets: {pool_buckets}")
    print(f"  Pool models per bucket: {pool_n_models}")
    print(f"  DA-axis families per bucket: {da_n_families}")
    if scaling_pairs:
        print(f"  Scaling-DA pairs (≥2 shared families): {scaling_pairs}")

    write_variants_definitions(out_dir)

    rows = []
    for task in tqdm(tasks, desc="Tasks"):
        row = {"task": task}

        # Core size-DA: small custom bucket@last → 1B target@last. Families
        # present at both buckets contribute; the distilled 600M↔1B family
        # joins the 600M column.
        for s in SMALL_SIZES:
            row[f"decision_acc_size_{s}"] = _safe(
                compute_size_decision_accuracy, df_pool, task, s,
            )
        # Scaling-DA: every other cross-bucket pair with ≥2 shared families
        # (lights up the larger ladder where multi-size families exist).
        for sb, tb in scaling_pairs:
            row[f"decision_acc_size_{sb}_to_{tb}"] = _safe(
                compute_size_decision_accuracy, df_pool, task, sb, tb,
            )
        # ckpt-DA: relative-fraction early ckpt vs max, per bucket — multi-ckpt
        # externals (a06, distill, SmolLM3-checkpoints, Olmo-3, Apertus-8B)
        # now participate.
        for frac in CKPT_DA_EARLY_FRACS:
            fl = _frac_label(frac)
            for b in pool_buckets:
                row[f"decision_acc_ckpt_{fl}_{b}"] = _safe(
                    compute_ckpt_decision_accuracy, df_pool, task, b, frac,
                )

        size_inputs = {b: per_model_inputs(df_pool, task, b) for b in pool_buckets}
        for fd in AGGREGATION_FUNCTIONS:
            key = variant_key(fd)
            for b in pool_buckets:
                # Gate custom-size cells: random benchmarks carry no signal.
                if b in AR_SIZES and (task, b) not in above_random:
                    sig = noi = snr = np.nan
                else:
                    sig, noi, snr = variant_signal_noise_snr(size_inputs[b], fd["func"])
                row[f"signal_{key}_{b}"] = sig
                row[f"noise_{key}_{b}"] = noi
                row[f"snr_{key}_{b}"] = snr
        rows.append(row)

    out = pd.DataFrame(rows).set_index("task").sort_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "snr_variants_per_task.csv"
    out.to_csv(csv_path)
    n_size = len(SMALL_SIZES) + len(scaling_pairs)
    n_ckpt = len(CKPT_DA_EARLY_FRACS) * len(pool_buckets)
    print(f"\nWrote CSV → {csv_path}")
    print(
        f"  {len(out)} tasks × {len(out.columns)} columns "
        f"({len(AGGREGATION_FUNCTIONS)} variants × {len(pool_buckets)} buckets × 3 stats "
        f"+ {n_size} size-DA + {n_ckpt} ckpt-DA)"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pool",
        required=True,
        help="Pool name from configs/models.json. Model-set tiers: "
        "seeds_1904, seeds_28_1797, seeds_28_1797_1904, custom_swissai_hf.",
    )
    p.add_argument(
        "--out-subdir",
        default=None,
        help="Subdir under results/<stage>/snr_definition/ (default: <pool>).",
    )
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; " f"available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    out_dir = PLOT_DIR / "snr_definition" / stage / (args.out_subdir or args.pool)
    run(pool=args.pool, out_dir=out_dir)


if __name__ == "__main__":
    main()
