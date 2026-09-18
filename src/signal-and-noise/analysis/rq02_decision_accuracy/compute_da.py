"""Compute decision accuracy (DA) per task → da_per_task.csv.

DA is the ground truth this project ultimately cares about: does a benchmark
rank a pair of models the way a larger-model evaluation would? It's costly, so
the rest of the pipeline (rq03_noise_and_snr) searches for cheap *proxies* —
SNR variants — that correlate with DA. So DA is computed first, here, and the
SNR step reads this CSV and appends its variant columns.

Two DA definitions (the cluster's two flavours):

  DA-size — (mix, seed) ranking at <small>'s last ckpt vs the ranking at
            TARGET_SIZE's last ckpt. ``decision_acc_size_<small>`` for every ``snr.small_sizes`` entry
            plus the cross-bucket scaling pairs ``decision_acc_size_<a>_to_<b>``.
  DA-ckpt — within a bucket, ranking at an early ckpt (a relative fraction of
            each model's own max step) vs the bucket's last ckpt.
            ``decision_acc_ckpt_<frac>_<bucket>``.

DA is computed on every above-random-or-not (task, size) cell — it is the truth,
not a proxy, so it is NOT gated (the above-random gate only NaN-s SNR cells).
``da_n_pairs_per_task.csv`` has the same shape and carries the number of model
pairs behind every value; an early checkpoint counts only within CKPT_TOL of
the fraction asked for.

    python analysis/rq02_decision_accuracy/compute_da.py --pool custom_swissai_hf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from evals.scripts.utils.configs import (  # noqa: E402
    add_family_column, bucket_order, load_pools, pool_include_external,
    size_bucket,
)
from snr.metrics import decision_acc_fast  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.utils import (  # noqa: E402
    CKPT_DA_EARLY_FRACS, SMALL_SIZES, TARGET_SIZE, _is_parent_task,
    build_snr_pool, pool_models,
)

OUT_ROOT = DECISION_ACCURACY


def _frac_label(frac: float) -> str:
    """0.12 → 'f12' (stable ckpt-DA column token)."""
    return f"f{int(round(frac * 100))}"


# Dedup set for missing-ckpt warnings — log each (bucket, frac, family)
# combination at most once across the whole run, regardless of task.
_LOGGED_MISSING_CKPTS: set = set()


def _safe(fn, *args, **kwargs):
    """(DA, number of pairs) — NaN and 0 when the cell is not computable."""
    try:
        v, n = fn(*args, return_n=True, **kwargs)
        return (float(v) if np.isfinite(v) else float("nan")), int(n)
    except Exception:
        return float("nan"), 0


# An early checkpoint counts only within this share of the run of the
# fraction asked for (half the k/10 benchmark grid, as analysis.utils.at_fraction).
CKPT_TOL = 0.06


def compute_size_decision_accuracy(
    df, task, small_size, target_size=TARGET_SIZE, model_filter=None, return_n=False
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
    n_pairs = len(common) * (len(common) - 1) // 2
    if len(common) < 2:
        return (float("nan"), 0) if return_n else float("nan")
    s = scores_small.set_index("family").loc[common, "primary_score"]
    t = scores_target.set_index("family").loc[common, "primary_score"]
    da = decision_acc_fast(s.to_numpy(), t.to_numpy())
    return (da, n_pairs) if return_n else da


def compute_ckpt_decision_accuracy(df, task, bucket, early_frac, model_filter=None,
                                   return_n=False):
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
        if abs(early_row["step"] - target_step) > CKPT_TOL * max_step:
            continue                    # no checkpoint near that fraction
        max_row = g.loc[g["step"].idxmax()]
        early.append(float(early_row["primary_score"]))
        late.append(float(max_row["primary_score"]))
        keys.append(fam)
    n_pairs = len(keys) * (len(keys) - 1) // 2
    if len(keys) < 2:
        return (float("nan"), 0) if return_n else float("nan")
    da = decision_acc_fast(np.asarray(early), np.asarray(late))
    return (da, n_pairs) if return_n else da


EARLY_SMALL_FRACS = list(CKPT_DA_EARLY_FRACS) + [1.0]


def _scores_at(dft, bucket, frac) -> dict:
    """family -> (score, compute) at the checkpoint nearest ``frac`` of the
    family's own run (its final checkpoint at 1.0), under the CKPT_TOL rule of
    ``compute_ckpt_decision_accuracy``."""
    out = {}
    for fam, g in dft[dft["bucket"] == bucket].groupby("family"):
        g = g.sort_values("step")
        max_step = g["step"].max()
        if frac >= 1.0:
            r = g.loc[g["step"].idxmax()]
        else:
            pre = g[g["step"] < max_step]
            if pre.empty:
                continue
            r = pre.iloc[(pre["step"] - frac * max_step).abs().argmin()]
            if abs(r["step"] - frac * max_step) > CKPT_TOL * max_step:
                continue
        out[fam] = (float(r["primary_score"]), float(r.get("compute", np.nan)))
    return out


def compute_early_small_decision_accuracy(dft, target_size=TARGET_SIZE, fracs=EARLY_SMALL_FRACS) -> list[dict]:
    """Early AND small, as a ranking: every design variant at a proxy bucket,
    read at 20-100 % of its own run, ranked against the same variants at the
    reference bucket's final checkpoint. The cross of DA-size (the 100 %
    column) and DA-ckpt (the reference's own row), one task at a time.

    One row per (proxy bucket, fraction) with >= 2 shared families: ``da``,
    ``n_pairs``, the mean training compute of the proxy checkpoints and of the
    reference finals (the cost axis of "how cheaply can we call it")."""
    dft = add_family_column(dft)
    ref = _scores_at(dft, target_size, 1.0)
    if len(ref) < 2:
        return []
    order = bucket_order()
    rows = []
    for b in [b for b in order[: order.index(target_size) + 1] if b in set(dft["bucket"])]:
        for frac in fracs:
            if b == target_size and frac >= 1.0:
                continue                    # the reference against itself
            got = _scores_at(dft, b, frac)
            common = sorted(set(got) & set(ref))
            if len(common) < 2:
                continue
            da = decision_acc_fast(np.asarray([got[f][0] for f in common]),
                                   np.asarray([ref[f][0] for f in common]))
            rows.append({"proxy_size": b, "frac": frac, "da": float(da),
                         "n_pairs": len(common) * (len(common) - 1) // 2,
                         "compute": float(np.mean([got[f][1] for f in common])),
                         "ref_compute": float(np.mean([ref[f][1] for f in common]))})
    return rows


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
    df_pool = build_snr_pool(pool)
    df_pool["bucket"] = df_pool["size"].map(size_bucket)
    all_tasks = sorted(df_pool["task"].unique())
    tasks = [t for t in all_tasks if _is_parent_task(t)]

    pool_buckets = [b for b in bucket_order() if b in set(df_pool["bucket"].dropna())]
    scaling_pairs = _scaling_da_pairs(df_pool)

    own_models = pool_models(pool, df_pool)
    all_models = set(df_pool["model"])
    print(
        f"Pool '{pool}': {len(all_models & own_models)} custom + "
        f"{len(all_models - own_models)} external model(s); "
        f"include_external={pool_include_external(pool)}"
    )
    print(f"  Buckets: {pool_buckets} | {len(tasks)} parent tasks")
    if scaling_pairs:
        print(f"  Scaling-DA pairs (≥2 shared families): {scaling_pairs}")

    # Pre-slice by task once — each compute_*_decision_accuracy call filters by
    # task, so grouping avoids re-scanning the whole pool per call.
    df_by_task = {t: g for t, g in df_pool.groupby("task", sort=False)}

    rows, n_rows, early_rows = [], [], []
    for task in tqdm(tasks, desc="DA tasks"):
        row, nrow = {"task": task}, {"task": task}
        dft = df_by_task[task]
        # Core size-DA: small bucket@last → reference bucket@last.
        for s in SMALL_SIZES:
            row[f"decision_acc_size_{s}"], nrow[f"decision_acc_size_{s}"] = _safe(
                compute_size_decision_accuracy, dft, task, s,
            )
        # Scaling-DA: every other cross-bucket pair with ≥2 shared families.
        for sb, tb in scaling_pairs:
            row[f"decision_acc_size_{sb}_to_{tb}"], nrow[f"decision_acc_size_{sb}_to_{tb}"] = _safe(
                compute_size_decision_accuracy, dft, task, sb, tb,
            )
        # ckpt-DA: relative-fraction early ckpt vs max, per bucket.
        for frac in CKPT_DA_EARLY_FRACS:
            fl = _frac_label(frac)
            for b in pool_buckets:
                row[f"decision_acc_ckpt_{fl}_{b}"], nrow[f"decision_acc_ckpt_{fl}_{b}"] = _safe(
                    compute_ckpt_decision_accuracy, dft, task, b, frac,
                )
        rows.append(row)
        n_rows.append(nrow)
        if TARGET_SIZE in pool_buckets:
            early_rows += [{"task": task, **r} for r in compute_early_small_decision_accuracy(dft)]

    out = pd.DataFrame(rows).set_index("task").sort_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "da_per_task.csv"
    out.to_csv(csv_path)
    # The same shape, holding the number of model pairs behind every cell: a
    # one-pair 1.00 and a 28-pair 0.96 must not read alike downstream.
    pd.DataFrame(n_rows).set_index("task").sort_index().to_csv(out_dir / "da_n_pairs_per_task.csv")
    # Early and small, as a ranking: long, one row per (task, proxy bucket, fraction).
    pd.DataFrame(early_rows, columns=["task", "proxy_size", "frac", "da", "n_pairs", "compute", "ref_compute"]
                 ).to_csv(out_dir / "da_early_small_per_task.csv", index=False)
    n_size = len(SMALL_SIZES) + len(scaling_pairs)
    n_ckpt = len(CKPT_DA_EARLY_FRACS) * len(pool_buckets)
    print(f"\nWrote DA CSV → {csv_path}")
    print(f"  {len(out)} tasks × {len(out.columns)} DA columns "
          f"({n_size} size-DA + {n_ckpt} ckpt-DA)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True,
                   help="Pool name from configs/models.json (e.g. seeds_1904, "
                        "custom_swissai_hf).")
    p.add_argument("--out-subdir", default=None,
                   help="Subdir under <stage>/ (default: <pool>).")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    run(pool=args.pool, out_dir=OUT_ROOT / stage / (args.out_subdir or args.pool))


if __name__ == "__main__":
    main()
