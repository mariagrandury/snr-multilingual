"""Compute decision accuracy (DA) per task → da_per_task.csv.

DA is the ground truth this project ultimately cares about: does a benchmark
rank a pair of models the way a larger-model evaluation would? It's costly, so
the rest of the pipeline (rq03_noise_and_snr) searches for cheap *proxies* —
SNR variants — that correlate with DA. So DA is computed first, here, and the
SNR step reads this CSV and appends its variant columns.

Three DA definitions, by what the ranking is compared against:

  DA-size — ranking at <small>'s last ckpt vs the ranking at TARGET_SIZE's last
            ckpt. ``decision_acc_size_<small>`` for every ``snr.small_sizes``
            entry plus the cross-bucket scaling pairs
            ``decision_acc_size_<a>_to_<b>``.
  DA-ckpt — within a bucket, ranking at an early ckpt (a relative fraction of
            each model's own max step) vs the bucket's last ckpt.
            ``decision_acc_ckpt_<frac>_<bucket>``.
  DA-goal — ranking at an early ckpt of <bucket> vs TARGET_SIZE's last ckpt:
            early AND small at once. ``decision_acc_goal_<frac>_<bucket>``, the
            wide form of ``da_early_small_per_task.csv``. DA-size is its
            ``f100`` column by construction, which is a free consistency check.

...and TWO PAIR SETS, the `axes` column (rule 15). Every table here carries one
row per (task, axes):

  multi-axis — every pair of design variants: the convention to 2026-09-22, in
               which two thirds of the pairs move more than one axis at once.
  mono-axis  — the pairs that move exactly one of L, arch, list, T, lang2, the
               seed held. The decision a practitioner makes, and what upstream's
               "every pair" is by construction (DataDecide's recipes differ in
               the data mix alone).
  seed       — the null: two draws of ONE design, which decide nothing, so a
               benchmark's DA over them is what no signal reads like. Emitted
               only where the pool has replicate seeds.

Consumers that do not ask read `multi-axis`, so no number moved when this was
added. `analysis/utils.py` owns the axis decomposition and the pair sets
(`design_axes`, `pair_sets`, `pair_agreement`); see plan/decision_accuracy.md.

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
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.utils import (  # noqa: E402
    CKPT_DA_EARLY_FRACS, MIN_PAIRS, PAIR_AXES, SMALL_SIZES, TARGET_SIZE, _is_parent_task,
    build_snr_pool, design_axes, pair_agreement, pair_sets, pool_models,
)

OUT_ROOT = DECISION_ACCURACY


def _frac_label(frac: float) -> str:
    """0.12 → 'f12' (stable ckpt-DA column token)."""
    return f"f{int(round(frac * 100))}"


# Dedup set for missing-ckpt warnings — log each (bucket, frac, family)
# combination at most once across the whole run, regardless of task.
_LOGGED_MISSING_CKPTS: set = set()
# Every cell the pair minimum emptied, (task, proxy, target, pairs): summarised
# loudly at the end of a run so a thin population is never silent (rule 5).
_FEW_PAIRS: list = []


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
    df, task, small_size, target_size=TARGET_SIZE, model_filter=None, return_n=False, pairs=None
):
    """DA across size buckets: small_bucket@last vs target_bucket@last.

    Operates on the ``bucket`` column (set by `run()` via `size_bucket`),
    so ``small_size`` / ``target_size`` are bucket labels. The cross-size
    identity is ``family`` (the model name with only the size token
    stripped), so a family present at both buckets contributes one pair.

    ``model_filter`` (optional) restricts the rows to a set of model
    names; ``pairs`` (optional) restricts the DECISIONS to an explicit list of
    family pairs (the mono-axis reading) instead of every pair. Returns NaN
    below MIN_PAIRS design-variant pairs (rule 5); with `return_n` the pair
    count comes back either way.
    """
    df = add_family_column(df)
    if model_filter is not None:
        df = df[df["model"].isin(model_filter)]
    scores_small = df[(df["bucket"] == small_size) & (df["task"] == task)]
    scores_target = df[(df["bucket"] == target_size) & (df["task"] == task)]
    if scores_small.empty or scores_target.empty:
        return (float("nan"), 0) if return_n else float("nan")
    scores_small = scores_small.loc[scores_small.groupby("family")["step"].idxmax()]
    scores_target = scores_target.loc[scores_target.groupby("family")["step"].idxmax()]
    da, n_pairs = pair_agreement(dict(zip(scores_small["family"], scores_small["primary_score"])),
                                 dict(zip(scores_target["family"], scores_target["primary_score"])), pairs)
    if n_pairs < MIN_PAIRS:          # rule 5: fewer pairs than this is not a ranking; the count is still reported
        _FEW_PAIRS.append((task, small_size, target_size, n_pairs))
    return (da, n_pairs) if return_n else da


def compute_ckpt_decision_accuracy(df, task, bucket, early_frac, model_filter=None,
                                   return_n=False, pairs=None):
    """DA within a size bucket: ``family`` ranking at an *early* ckpt vs the
    same family's max-step ckpt.

    The early ckpt is chosen per model as the checkpoint whose step is
    closest to ``early_frac × (that model's own max step)`` — a relative
    fraction rather than an absolute iter, so external / a06 / distillation
    trajectories (whose step scales differ from the custom megatron iters)
    participate. A family with only one checkpoint (single-ckpt HF refs)
    has no distinct early ckpt and is logged once per ``(bucket, frac,
    family)`` and skipped. Below MIN_PAIRS pairs the cell is NaN (rule 5).

    ``model_filter`` (optional) restricts the rows to a set of model names;
    ``pairs`` (optional) restricts the decisions to an explicit family-pair list.
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
    da, n_pairs = pair_agreement(dict(zip(keys, early)), dict(zip(keys, late)), pairs)
    if n_pairs < MIN_PAIRS:          # rule 5
        _FEW_PAIRS.append((task, bucket, f"ckpt@{early_frac}", n_pairs))
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


def compute_early_small_decision_accuracy(dft, target_size=TARGET_SIZE, fracs=EARLY_SMALL_FRACS,
                                          pairs=None) -> list[dict]:
    """Early AND small, as a ranking: every design variant at a proxy bucket,
    read at 20-100 % of its own run, ranked against the same variants at the
    reference bucket's final checkpoint. The cross of DA-size (the 100 %
    column) and DA-ckpt (the reference's own row), one task at a time.

    One row per (proxy bucket, fraction) with >= MIN_PAIRS pairs of shared families (rule 5; the cells below it are
    left out and counted in the run's RULE 5 report): ``da``,
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
            da, n_pairs = pair_agreement({f: v[0] for f, v in got.items()},
                                         {f: v[0] for f, v in ref.items()}, pairs)
            if n_pairs < MIN_PAIRS:                 # rule 5, as the two kernels above
                _FEW_PAIRS.append((None, b, f"early-small@{frac}", n_pairs))
                continue
            common = sorted(set(got) & set(ref))
            rows.append({"proxy_size": b, "frac": frac, "da": float(da), "n_pairs": n_pairs,
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

    # The pair sets the `axes` column names (rule 15). `pair_sets` holds the
    # design pairs at the grid seed, so a replicate never enters a decision set;
    # the `seed` set is exactly those replicate pairs and is the null. A set the
    # pool cannot populate is dropped rather than written as a row of NaN.
    psets = pair_sets(design_axes(df_pool))
    axes_sets = [a for a in PAIR_AXES if psets[a]]
    print("  Pair sets: " + ", ".join(f"{a} {len(psets[a])}" for a in axes_sets))

    # Pre-slice by task once — each compute_*_decision_accuracy call filters by
    # task, so grouping avoids re-scanning the whole pool per call.
    df_by_task = {t: g for t, g in df_pool.groupby("task", sort=False)}

    rows, n_rows, early_rows = [], [], []
    for axes in axes_sets:
        pairs = psets[axes]
        for task in tqdm(tasks, desc=f"DA tasks ({axes})"):
            row, nrow = {"task": task, "axes": axes}, {"task": task, "axes": axes}
            dft = df_by_task[task]
            # Core size-DA: small bucket@last → reference bucket@last.
            for s in SMALL_SIZES:
                row[f"decision_acc_size_{s}"], nrow[f"decision_acc_size_{s}"] = _safe(
                    compute_size_decision_accuracy, dft, task, s, pairs=pairs,
                )
            # Scaling-DA: every other cross-bucket pair with ≥2 shared families.
            for sb, tb in scaling_pairs:
                row[f"decision_acc_size_{sb}_to_{tb}"], nrow[f"decision_acc_size_{sb}_to_{tb}"] = _safe(
                    compute_size_decision_accuracy, dft, task, sb, tb, pairs=pairs,
                )
            # ckpt-DA: relative-fraction early ckpt vs max, per bucket.
            for frac in CKPT_DA_EARLY_FRACS:
                fl = _frac_label(frac)
                for b in pool_buckets:
                    row[f"decision_acc_ckpt_{fl}_{b}"], nrow[f"decision_acc_ckpt_{fl}_{b}"] = _safe(
                        compute_ckpt_decision_accuracy, dft, task, b, frac, pairs=pairs,
                    )
            # goal-DA: early AND small, the long table and its wide columns from
            # ONE computation, so the two shapes cannot disagree.
            if TARGET_SIZE in pool_buckets:
                early = compute_early_small_decision_accuracy(dft, pairs=pairs)
                for r in early:
                    col = f"decision_acc_goal_{_frac_label(r['frac'])}_{r['proxy_size']}"
                    row[col], nrow[col] = r["da"], r["n_pairs"]
                early_rows += [{"task": task, "axes": axes, **r} for r in early]
            rows.append(row)
            n_rows.append(nrow)

    out = pd.DataFrame(rows).set_index(["task", "axes"]).sort_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "da_per_task.csv"
    out.to_csv(csv_path)
    # The same shape, holding the number of model pairs behind every cell: a
    # one-pair 1.00 and a 28-pair 0.96 must not read alike downstream.
    pd.DataFrame(n_rows).set_index(["task", "axes"]).sort_index().to_csv(out_dir / "da_n_pairs_per_task.csv")
    # Early and small, as a ranking: long, one row per (task, axes, proxy bucket, fraction).
    pd.DataFrame(early_rows, columns=["task", "axes", "proxy_size", "frac", "da", "n_pairs", "compute", "ref_compute"]
                 ).to_csv(out_dir / "da_early_small_per_task.csv", index=False)
    n_size = len(SMALL_SIZES) + len(scaling_pairs)
    n_ckpt = len(CKPT_DA_EARLY_FRACS) * len(pool_buckets)
    n_goal = sum(c.startswith("decision_acc_goal_") for c in out.columns)
    if _FEW_PAIRS:
        few = pd.DataFrame(_FEW_PAIRS, columns=["task", "proxy", "target", "pairs"])
        by = few.groupby(["proxy", "target"]).size().sort_values(ascending=False)
        print(f"\n!!! RULE 5: {len(few)} decision-accuracy cells over {few['task'].nunique()} tasks had fewer than "
              f"{MIN_PAIRS} pairs and are NaN (their pair counts are in da_n_pairs_per_task.csv). By comparison:\n"
              + by.head(12).to_string())
    print(f"\nWrote DA CSV → {csv_path}")
    print(f"  {out.index.get_level_values('task').nunique()} tasks × {len(axes_sets)} pair sets "
          f"× {len(out.columns)} DA columns ({n_size} size-DA + {n_ckpt} ckpt-DA + {n_goal} goal-DA)")
    # DA-size is DA-goal read at the proxy's own final checkpoint: the same
    # decisions, computed by two different code paths. They must agree.
    same = [(f"decision_acc_size_{s}", f"decision_acc_goal_f100_{s}") for s in SMALL_SIZES
            if f"decision_acc_goal_f100_{s}" in out.columns]
    # ... and DA-ckpt at the reference is DA-goal at the reference: the
    # reference's own trajectory read against its own final, by both paths.
    same_ref = [(f"decision_acc_ckpt_{_frac_label(f)}_{TARGET_SIZE}", f"decision_acc_goal_{_frac_label(f)}_{TARGET_SIZE}")
                for f in CKPT_DA_EARLY_FRACS if f"decision_acc_goal_{_frac_label(f)}_{TARGET_SIZE}" in out.columns]
    for label, cols in (("DA-size == DA-goal@f100", same), (f"DA-ckpt@{TARGET_SIZE} == DA-goal@{TARGET_SIZE}", same_ref)):
        if not cols:
            # a pool without the reference bucket has no goal-DA, so there is nothing
            # to check — "0.00e+00 over 0 columns" would read as a check that passed
            print(f"  {label}: not checked (no goal-DA in this pool)")
            continue
        bad = {a: float((out[a] - out[b]).abs().max()) for a, b in cols}
        worst = max(bad.values())
        print(f"  {label}: max |diff| = {worst:.2e} over {len(cols)} columns"
              + ("" if worst < 1e-9 else f"  !!! {bad}"))


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
