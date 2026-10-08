"""Compute decision accuracy (DA) per task → da_all_per_task_both_axes.csv.

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
            wide form of ``da_goal_early_small_per_task_both_axes.csv``. DA-size is its
            ``f100`` column by construction, which is a free consistency check.

...and TWO PAIR SETS, the `axes` column (rule 15). Every table here carries one
row per (task, axes):

  multi-axis — every pair of design variants: the convention to 2026-09-22, in
               which two thirds of the pairs move more than one axis at once.
  mono-axis  — the pairs that move exactly one of utils.DESIGN_AXES, the
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
``da_all_n_pairs_per_task_both_axes.csv`` has the same shape and carries the number of model
pairs behind every value; an early checkpoint counts only within CKPT_TOL of
the fraction asked for.

Every column is read from one score cube per task (`_ckpt_cube`): the
checkpoint each (bucket, family) is read at, for every fraction, picked once
under the rule of `_scores_at`, and the pair signs taken once per pair set.
The per-call kernels below (`compute_size_decision_accuracy`, ...) state the
same rule one cell at a time; tests/test_early_small.py holds the two equal,
and a task with two rows at one (bucket, family, step) is computed by them.
`--workers` (or COMPUTE_DA_WORKERS) spreads the tasks over processes.

    python analysis/rq02_decision_accuracy/compute_da.py --pool custom_swissai_hf
"""

from __future__ import annotations

import argparse
import itertools
import multiprocessing as mp
import os
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
                                   return_n=False, pairs=None, runs=None):
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
    ``runs`` (optional) is ``family_runs(df, bucket)`` of this task's rows, built
    once by a caller that asks for many fractions (``df`` and ``model_filter``
    are then not read).
    """
    if runs is None:
        df = add_family_column(df)
        if model_filter is not None:
            df = df[df["model"].isin(model_filter)]
        runs = family_runs(df[df["task"] == task], bucket)
    if not runs:
        return float("nan")
    early, late = {}, {}
    for fam, (steps, score, _) in runs.items():
        if not (steps < steps.max()).any():     # no checkpoint before the last
            key = (bucket, early_frac, fam)
            if key not in _LOGGED_MISSING_CKPTS:
                _LOGGED_MISSING_CKPTS.add(key)
                print(
                    f"  ckpt-DA: only one ckpt for bucket={bucket} "
                    f"family={fam} (first seen on task={task}) — skipped"
                )
            continue
        i = _pick(steps, early_frac)
        if i is None:
            continue                    # no checkpoint near that fraction
        early[fam], late[fam] = float(score[i]), float(score[_pick(steps, 1.0)])
    da, n_pairs = pair_agreement(early, late, pairs)
    if n_pairs < MIN_PAIRS:          # rule 5
        _FEW_PAIRS.append((task, bucket, f"ckpt@{early_frac}", n_pairs))
    return (da, n_pairs) if return_n else da


def family_runs(dft, bucket) -> dict:
    """family -> (steps, primary_score, compute) arrays of its rows at
    ``bucket``, in step order: what ``_pick`` reads. A caller asking for many
    fractions builds it once per (task, bucket); filtering and sorting the
    frame anew for every fraction was the whole cost of the checkpoint grids."""
    out = {}
    for fam, g in dft[dft["bucket"] == bucket].groupby("family"):
        g = g.sort_values("step")
        out[fam] = (g["step"].to_numpy(), g["primary_score"].to_numpy(dtype=float),
                    g["compute"].to_numpy(dtype=float) if "compute" in g.columns else np.full(len(g), np.nan))
    return out


def _pick(steps, frac):
    """Index into a run's step-ordered ``steps`` of the checkpoint nearest
    ``frac`` of the run: the last checkpoint at 1.0, else the nearest one before
    it, or None when that one is further than CKPT_TOL of the run away."""
    max_step = steps.max()
    if frac >= 1.0:
        return int(np.argmax(steps))
    pre = np.flatnonzero(steps < max_step)
    if not len(pre):
        return None
    i = pre[int(np.argmin(np.abs(steps[pre] - frac * max_step)))]
    if abs(steps[i] - frac * max_step) > CKPT_TOL * max_step:
        return None
    return i


EARLY_SMALL_FRACS = list(CKPT_DA_EARLY_FRACS) + [1.0]


def _scores_at(dft, bucket, frac, runs=None) -> dict:
    """family -> (score, compute) at the checkpoint nearest ``frac`` of the
    family's own run (its final checkpoint at 1.0), under the CKPT_TOL rule of
    ``compute_ckpt_decision_accuracy``. ``runs`` = ``family_runs(dft, bucket)``
    when the caller already has it."""
    out = {}
    for fam, (steps, score, compute) in (family_runs(dft, bucket) if runs is None else runs).items():
        i = _pick(steps, frac)
        if i is not None:
            out[fam] = (float(score[i]), float(compute[i]))
    return out


def compute_early_small_decision_accuracy(dft, target_size=TARGET_SIZE, fracs=EARLY_SMALL_FRACS,
                                          pairs=None, runs=None) -> list[dict]:
    """Early AND small, as a ranking: every design variant at a proxy bucket,
    read at 20-100 % of its own run, ranked against the same variants at the
    reference bucket's final checkpoint. The cross of DA-size (the 100 %
    column) and DA-ckpt (the reference's own row), one task at a time.

    One row per (proxy bucket, fraction) with >= MIN_PAIRS pairs of shared families (rule 5; the cells below it are
    left out and counted in the run's RULE 5 report): ``da``,
    ``n_pairs``, the mean training compute of the proxy checkpoints and of the
    reference finals (the cost axis of "how cheaply can we call it")."""
    dft = add_family_column(dft)
    order = bucket_order()
    buckets = [b for b in order[: order.index(target_size) + 1] if b in set(dft["bucket"])]
    # one family_runs per bucket (``runs`` = bucket -> it, when the caller has them), read at every fraction
    runs = {b: (runs or {}).get(b) or family_runs(dft, b) for b in {*buckets, target_size}}
    ref = _scores_at(dft, target_size, 1.0, runs[target_size])
    if len(ref) < 2:
        return []
    rows = []
    for b in buckets:
        for frac in fracs:
            if b == target_size and frac >= 1.0:
                continue                    # the reference against itself
            got = _scores_at(dft, b, frac, runs[b])
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


def _ckpt_cube(df: pd.DataFrame, buckets, fracs):
    """The scores every DA column reads, picked once: (tasks, families, cols,
    S, C, P, single). Column (bucket, frac) holds, per task and family, the
    score S and compute C at the checkpoint `_scores_at(bucket, frac)` reads
    (frac 1.0: the final checkpoint); P marks a family read there, so a NaN
    score stays apart from an absent family (it is counted, and is a miss, as
    in `pair_agreement`). `single` lists the (task, bucket, family) runs with
    one checkpoint, which DA-ckpt logs and skips.

    The rule of `_scores_at` and `compute_ckpt_decision_accuracy`: the
    non-final checkpoint nearest frac × the run's max step, the smaller step
    on a tie, within CKPT_TOL of the run. It assumes one row per (task,
    bucket, family, step); `run` sends a task that breaks that to the per-call
    kernels."""
    d = df.loc[df["bucket"].isin(buckets) & df["family"].notna(), ["task", "bucket", "family", "step", "primary_score"]]
    d = d.assign(compute=df["compute"] if "compute" in df.columns else np.nan)
    d = d.sort_values(["task", "bucket", "family", "step"], kind="mergesort").reset_index(drop=True)
    gid = d.groupby(["task", "bucket", "family"], sort=False).ngroup().to_numpy()
    step = d["step"]
    mx = step.groupby(gid).transform("max")
    tasks = sorted(d["task"].unique())
    fams = sorted(d["family"].unique())
    cols = [(b, f) for b in buckets for f in list(fracs) + [1.0]]
    S = np.full((len(tasks), len(fams), len(cols)), np.nan)
    C = np.full_like(S, np.nan)
    P = np.zeros(S.shape, dtype=bool)
    ti = d["task"].map({t: i for i, t in enumerate(tasks)}).to_numpy(dtype=int)
    fi = d["family"].map({f: i for i, f in enumerate(fams)}).to_numpy(dtype=int)
    bpos = {b: i for i, b in enumerate(buckets)}
    bi = d["bucket"].map(bpos).to_numpy(dtype=int)
    nf = len(fracs) + 1
    score, comp = d["primary_score"].to_numpy(), d["compute"].to_numpy(dtype=float)

    def put(rows, k):
        c = bi[rows] * nf + k
        S[ti[rows], fi[rows], c], C[ti[rows], fi[rows], c], P[ti[rows], fi[rows], c] = score[rows], comp[rows], True

    put(np.flatnonzero((step == mx).to_numpy()), nf - 1)          # the final checkpoint
    is_pre = (step < mx).to_numpy()
    pre = np.flatnonzero(is_pre)
    for k, frac in enumerate(fracs):
        target = frac * mx.iloc[pre]
        dist = (step.iloc[pre] - target).abs()
        best = dist.groupby(gid[pre]).idxmin().to_numpy()       # first minimum: the smaller step on a tie
        ok = (dist.loc[best] <= CKPT_TOL * mx.loc[best]).to_numpy()
        put(best[ok], k)
    has_pre = np.zeros(gid.max() + 1 if len(gid) else 0, dtype=bool)
    has_pre[gid[pre]] = True
    first = np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]]) if len(gid) else np.array([], int)
    single = [(tasks[ti[r]], d["bucket"].iat[r], d["family"].iat[r]) for r in first if not has_pre[gid[r]]]
    return tasks, fams, cols, S, C, P, single


# Shared with forked workers (set in `run` before the pool starts).
_CUBE: dict = {}


def _da(D, V, pc, rc):
    """(DA, pairs) of proxy column `pc` against reference column `rc`, over
    the pairs both columns read (`pair_agreement`'s rule; NaN below MIN_PAIRS)."""
    m = V[:, pc] & V[:, rc]
    n = int(m.sum())
    if n < MIN_PAIRS:
        return float("nan"), n
    agree = (D[m, pc] == D[m, rc]).sum()
    return float(agree / n), n


def _task_rows(k: int):
    """Every pair set's (row, n_row, early rows) for the k-th task of the cube,
    plus its rule-5 log — the rows `run` used to build one call at a time."""
    c = _CUBE
    task, S, Cm, P = c["tasks"][k], c["S"][k], c["C"][k], c["P"][k]
    col = c["col"]
    present = {b: bool(P[:, col[(b, 1.0)]].any()) for b in c["buckets"]}
    few, out = [], {}
    for axes, (I, J) in c["pairs"].items():
        D = np.sign(S[I] - S[J])
        V = P[I] & P[J]
        row, nrow = {"task": task, "axes": axes}, {"task": task, "axes": axes}

        def size(sb, tb, name):
            if not (present.get(sb) and present.get(tb)):
                row[name], nrow[name] = float("nan"), 0
                return
            row[name], nrow[name] = _da(D, V, col[(sb, 1.0)], col[(tb, 1.0)])
            if nrow[name] < MIN_PAIRS:
                few.append((task, sb, tb, nrow[name]))

        for s in SMALL_SIZES:
            size(s, TARGET_SIZE, f"decision_acc_size_{s}")
        for sb, tb in c["scaling_pairs"]:
            size(sb, tb, f"decision_acc_size_{sb}_to_{tb}")
        for frac in CKPT_DA_EARLY_FRACS:
            fl = _frac_label(frac)
            for b in c["buckets"]:
                name = f"decision_acc_ckpt_{fl}_{b}"
                if not present[b]:
                    row[name], nrow[name] = float("nan"), 0
                    continue
                row[name], nrow[name] = _da(D, V, col[(b, frac)], col[(b, 1.0)])
                if nrow[name] < MIN_PAIRS:
                    few.append((task, b, f"ckpt@{frac}", nrow[name]))
        early = []
        if TARGET_SIZE in c["buckets"]:
            rc = col[(TARGET_SIZE, 1.0)]
            if P[:, rc].sum() >= 2:
                order = bucket_order()
                for b in [b for b in order[: order.index(TARGET_SIZE) + 1] if present.get(b)]:
                    for frac in EARLY_SMALL_FRACS:
                        if b == TARGET_SIZE and frac >= 1.0:
                            continue
                        pc = col[(b, frac)]
                        da, n_pairs = _da(D, V, pc, rc)
                        if n_pairs < MIN_PAIRS:
                            few.append((None, b, f"early-small@{frac}", n_pairs))
                            continue
                        common = P[:, pc] & P[:, rc]
                        early.append({"proxy_size": b, "frac": frac, "da": da, "n_pairs": n_pairs,
                                      "compute": float(np.mean(Cm[common, pc])),
                                      "ref_compute": float(np.mean(Cm[common, rc]))})
            for r in early:
                name = f"decision_acc_goal_{_frac_label(r['frac'])}_{r['proxy_size']}"
                row[name], nrow[name] = r["da"], r["n_pairs"]
        out[axes] = (row, nrow, [{"task": task, "axes": axes, **r} for r in early])
    return out, few


def _task_rows_per_call(dft, task, axes_sets, psets, scaling_pairs, pool_buckets):
    """`_task_rows` by the per-call kernels, for a task the cube cannot hold."""
    out = {}
    for axes in axes_sets:
        pairs = psets[axes]
        row, nrow = {"task": task, "axes": axes}, {"task": task, "axes": axes}
        for s in SMALL_SIZES:
            row[f"decision_acc_size_{s}"], nrow[f"decision_acc_size_{s}"] = _safe(
                compute_size_decision_accuracy, dft, task, s, pairs=pairs)
        for sb, tb in scaling_pairs:
            row[f"decision_acc_size_{sb}_to_{tb}"], nrow[f"decision_acc_size_{sb}_to_{tb}"] = _safe(
                compute_size_decision_accuracy, dft, task, sb, tb, pairs=pairs)
        for frac in CKPT_DA_EARLY_FRACS:
            fl = _frac_label(frac)
            for b in pool_buckets:
                row[f"decision_acc_ckpt_{fl}_{b}"], nrow[f"decision_acc_ckpt_{fl}_{b}"] = _safe(
                    compute_ckpt_decision_accuracy, dft, task, b, frac, pairs=pairs)
        early = []
        if TARGET_SIZE in pool_buckets:
            early = compute_early_small_decision_accuracy(dft, pairs=pairs)
            for r in early:
                col = f"decision_acc_goal_{_frac_label(r['frac'])}_{r['proxy_size']}"
                row[col], nrow[col] = r["da"], r["n_pairs"]
        out[axes] = (row, nrow, [{"task": task, "axes": axes, **r} for r in early])
    return out


def _n_workers(cli: int | None) -> int:
    """--workers, else COMPUTE_DA_WORKERS, else 1 (the login-node default)."""
    return max(1, cli if cli is not None else int(os.environ.get("COMPUTE_DA_WORKERS", "1")))


def da_by_task(dfp, tasks, axes_sets, psets, scaling_pairs, pool_buckets, workers: int = 1,
               cube: bool = True) -> dict:
    """{task: {axes: (row, n_row, early rows)}} over `dfp`, the pool's rows of
    `tasks`. `cube=False` sends every task to the per-call kernels, which the
    cube reproduces bit for bit (tests/test_early_small.py)."""
    # One cube per task, read by every pair set. A task with two rows at one
    # (bucket, family, step) has no single "nearest checkpoint" the cube can
    # hold; it goes to the per-call kernels, which pick as they always did.
    dup = dfp.duplicated(["task", "bucket", "family", "step"]) & dfp["bucket"].notna()
    per_call = sorted(set(dfp.loc[dup, "task"])) if cube else list(tasks)
    if per_call and cube:
        print(f"  {len(per_call)} task(s) with duplicate (bucket, family, step) rows → per-call kernels")
    ctasks, fams, cols, S, Cm, P, single = _ckpt_cube(dfp[~dfp["task"].isin(per_call)], pool_buckets,
                                                     CKPT_DA_EARLY_FRACS)
    fidx = {f: i for i, f in enumerate(fams)}
    pairs_ij = {}
    for a in axes_sets:
        pl = [(fidx[x], fidx[y]) for x, y in psets[a] if x in fidx and y in fidx]
        pairs_ij[a] = (np.array([i for i, _ in pl], dtype=int), np.array([j for _, j in pl], dtype=int))
    _CUBE.update(tasks=ctasks, S=S, C=Cm, P=P, col={c: i for i, c in enumerate(cols)}, buckets=pool_buckets,
                 scaling_pairs=scaling_pairs, pairs=pairs_ij)
    # DA-ckpt skips a run with one checkpoint; say so once per (bucket, frac, family).
    for t, grp in itertools.groupby(single, key=lambda x: x[0]):      # `single` is in task order
        grp = sorted(grp, key=lambda x: (pool_buckets.index(x[1]), x[2]))
        for frac, (_, b, f) in itertools.product(CKPT_DA_EARLY_FRACS, grp):
            if (b, frac, f) not in _LOGGED_MISSING_CKPTS:
                _LOGGED_MISSING_CKPTS.add((b, frac, f))
                print(f"  ckpt-DA: only one ckpt for bucket={b} family={f} (first seen on task={t}) — skipped")

    by_task = {}
    n_workers = min(workers, max(1, len(ctasks)))
    if n_workers > 1:
        print(f"  {n_workers} worker processes")
        with mp.get_context("fork").Pool(n_workers) as ex:
            res = ex.imap(_task_rows, range(len(ctasks)), chunksize=max(1, len(ctasks) // (8 * n_workers)))
            for t, (out, few) in zip(ctasks, tqdm(res, total=len(ctasks), desc="DA tasks")):
                by_task[t] = out
                _FEW_PAIRS.extend(few)
    else:
        for k, t in enumerate(tqdm(ctasks, desc="DA tasks")):
            by_task[t], few = _task_rows(k)
            _FEW_PAIRS.extend(few)
    for t in [t for t in tasks if t not in by_task]:       # duplicate rows, or no row in any bucket
        by_task[t] = _task_rows_per_call(dfp[dfp["task"] == t], t, axes_sets, psets, scaling_pairs, pool_buckets)

    return by_task


def score_cube(pool: str):
    """The pool's parent tasks in one cube, read with this module's checkpoint
    choice (`_ckpt_cube`), for the analyses that score DA on pair lists of
    their own (rq02's cross-fitted filter, the permutation null, the decisive
    pairs): (frame, tasks, families, {(bucket, frac): column}, S, P). A task
    whose rows the cube cannot hold (two rows at one checkpoint) is left out
    and printed."""
    df = add_family_column(build_snr_pool(pool).assign(bucket=lambda x: x["size"].map(size_bucket)))
    df = df[df["task"].map(_is_parent_task)]
    dup = set(df.loc[df.duplicated(["task", "bucket", "family", "step"]) & df["bucket"].notna(), "task"])
    if dup:
        print(f"  score_cube: {len(dup)} task(s) with duplicate (bucket, family, step) rows left out")
    buckets = [b for b in bucket_order() if b in set(df["bucket"].dropna())]
    tasks, fams, cols, S, _, P, _ = _ckpt_cube(df[~df["task"].isin(dup)], buckets, CKPT_DA_EARLY_FRACS)
    return df, tasks, fams, {c: i for i, c in enumerate(cols)}, S, P


def pair_agree(S, P, I, J, pc: int, rc: int):
    """Per (task, pair) of the cube: whether pair (I[k], J[k]) is decided at the
    proxy column `pc` as at the reference column `rc` (the sign rule of
    `_da`), and whether both sides read it. Boolean arrays of shape
    (tasks, pairs); their row sums are a cell's matching and comparable pairs."""
    V = P[:, I, pc] & P[:, J, pc] & P[:, I, rc] & P[:, J, rc]
    with np.errstate(invalid="ignore"):
        A = (np.sign(S[:, I, pc] - S[:, J, pc]) == np.sign(S[:, I, rc] - S[:, J, rc])) & V
    return A, V


def run(pool: str, out_dir: Path, workers: int | None = None):
    df_pool = build_snr_pool(pool)
    df_pool["bucket"] = df_pool["size"].map(size_bucket)
    df_pool = add_family_column(df_pool)
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

    by_task = da_by_task(df_pool[df_pool["task"].isin(tasks)], tasks, axes_sets, psets, scaling_pairs,
                         pool_buckets, _n_workers(workers))
    rows, n_rows, early_rows = [], [], []
    for axes in axes_sets:
        for task in tasks:
            row, nrow, early = by_task[task][axes]
            rows.append(row)
            n_rows.append(nrow)
            early_rows += early

    out = pd.DataFrame(rows).set_index(["task", "axes"]).sort_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "da_all_per_task_both_axes.csv"
    out.to_csv(csv_path)
    # The same shape, holding the number of model pairs behind every cell: a
    # one-pair 1.00 and a 28-pair 0.96 must not read alike downstream.
    pd.DataFrame(n_rows).set_index(["task", "axes"]).sort_index().to_csv(out_dir / "da_all_n_pairs_per_task_both_axes.csv")
    # Early and small, as a ranking: long, one row per (task, axes, proxy bucket, fraction).
    pd.DataFrame(early_rows, columns=["task", "axes", "proxy_size", "frac", "da", "n_pairs", "compute", "ref_compute"]
                 ).to_csv(out_dir / "da_goal_early_small_per_task_both_axes.csv", index=False)
    n_size = len(SMALL_SIZES) + len(scaling_pairs)
    n_ckpt = len(CKPT_DA_EARLY_FRACS) * len(pool_buckets)
    n_goal = sum(c.startswith("decision_acc_goal_") for c in out.columns)
    if _FEW_PAIRS:
        few = pd.DataFrame(_FEW_PAIRS, columns=["task", "proxy", "target", "pairs"])
        by = few.groupby(["proxy", "target"]).size().sort_values(ascending=False)
        print(f"\n!!! RULE 5: {len(few)} decision-accuracy cells over {few['task'].nunique()} tasks had fewer than "
              f"{MIN_PAIRS} pairs and are NaN (their pair counts are in da_all_n_pairs_per_task_both_axes.csv). By comparison:\n"
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
    p.add_argument("--workers", type=int, default=None,
                   help="Processes over the tasks (default: COMPUTE_DA_WORKERS, else 1).")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    run(pool=args.pool, out_dir=OUT_ROOT / stage / (args.out_subdir or args.pool), workers=args.workers)


if __name__ == "__main__":
    main()
