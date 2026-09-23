"""rq08, per-item view on the ladder: can a subset of a benchmark's ITEMS give a
higher SNR than the full set, and does the subset still rank the designs like
the reference when it is chosen on other designs?

Reads the per-item store (`build_per_item_store.py`, built by the sbatch, not
here) and, per (task, size) over the grid-seed design variants:

  * the items x (run, checkpoint) matrix on the rule-4 noise window
    (`noise_checkpoints`), the per-item SNR of `smooth_subtasks_per_sample.py`
    (signal = range of the run means / their mean, noise = pooled std / pooled
    mean, vectorised over items) read on that window rather than the last five
    saves, and the DEAD share: items whose mean outcome is the same in every
    run of the window (no cross-run signal at all);
  * the cumulative SNR sweep over items ranked by SNR, against one random
    order and the selection null of `smooth_subtasks.py` (NULL_DRAWS random
    subsets of the best size, 95th percentile);
  * the leakage-safe decision accuracy (rule 11): the families are split in two
    halves stratified over L / arch / scheme, the subset is chosen on one half
    and its mean at the proxy's final checkpoint is scored against the 1.7B
    final of the FULL task (`pair_agreement`, multi-axis pairs within the
    held-out half, rule 5) beside the full set and RANDOM_DRAWS random subsets
    of the same size; then the halves swap and both readings are reported.

Outputs under pretraining/<pool>/: per_item_snr.csv (task, size, item),
per_item_summary.csv (task, size), per_item_ladder.png + .csv, and the
`per-item-ladder` README block. Rule 1: the gate blanks the cells at chance
(kept, `gated`); rule 6: the store holds the parents' items, subject files
folded in; rule 13: the captions carry the populations.

    python analysis/rq08_subset_selection/per_item_ladder.py --pool predictivity [--store predictivity_seeds]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evals.scripts.utils.configs import load_pools, metric_for  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, fmt, md_table, replace_block  # noqa: E402
from analysis.paths import SUBSET_SELECTION  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq08_subset_selection.build_per_item_store import STORE  # noqa: E402
from analysis.rq08_subset_selection.smooth_subtasks import NULL_DRAWS  # noqa: E402
from analysis.utils import (GRID_SEED, MIN_PAIRS, TARGET_SIZE, benchmark_family, design_axes, finals,  # noqa: E402
                            ladder_frame, noise_checkpoints, pair_agreement, pair_sets, passes_gate, size_order)

RANDOM_DRAWS = 20        # random subsets of the best size scored on the held-out half
mpl.rcParams.update(S.RC)


def snr_cols(A: np.ndarray, runs: np.ndarray) -> np.ndarray:
    """SNR of every column of `A` (rows = the window checkpoints of `runs`):
    `signal_to_noise_ratio` of smooth_subtasks — range of the run means over
    their mean, pooled std over pooled mean — vectorised; a run with one
    checkpoint is dropped, fewer than two runs is NaN."""
    r = pd.Series(runs)
    keep = r.map(r.value_counts()).to_numpy() >= 2
    A, r = A[keep], r[keep]
    if r.nunique() < 2:
        return np.full(A.shape[1], np.nan)
    means = np.stack([A[(r == k).to_numpy()].mean(0) for k in r.unique()])
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = ((means.max(0) - means.min(0)) / means.mean(0)) / (A.std(0) / A.mean(0))
    return np.where(np.isfinite(snr), snr, np.nan)


def sweep(A: np.ndarray, runs: np.ndarray, rng) -> dict:
    """Per-item SNR, the dead mask, and the cumulative sweep by SNR rank with
    its random-order twin and the selection null."""
    snr = snr_cols(A, runs)
    means = np.stack([A[runs == k].mean(0) for k in np.unique(runs)])
    dead = means.std(0) == 0
    order = np.argsort(-np.nan_to_num(snr[~dead], nan=-np.inf))
    alive = np.flatnonzero(~dead)[order]
    out = {"snr": snr, "dead": dead, "snr_full": snr_cols(A.mean(1, keepdims=True), runs)[0],
           "snr_best": np.nan, "best_n": 0, "snr_random_order": np.nan, "null_p95": np.nan}
    if alive.size == 0 or np.isnan(snr[alive]).all():
        return out
    k = np.arange(1, alive.size + 1)
    curve = snr_cols(A[:, alive].cumsum(1) / k, runs)
    best = int(np.nanargmax(curve))
    shuffled = rng.permutation(alive)
    out.update(snr_best=curve[best], best_n=best + 1,
               snr_random_order=snr_cols(A[:, shuffled].cumsum(1) / k, runs)[best], subset=alive[:best + 1])
    if 0 < best + 1 < alive.size:
        draws = np.stack([A[:, rng.choice(alive, best + 1, replace=False)].mean(1) for _ in range(NULL_DRAWS)], 1)
        out["null_p95"] = float(np.nanpercentile(snr_cols(draws, runs), 95))
    return out


def heldout_da(A: np.ndarray, runs: np.ndarray, P: dict, ref: dict, pairs: list, attrs: pd.DataFrame, rng) -> dict:
    """Choose the subset on one half of the families, score it on the other
    half's pairs (proxy final vs the reference's final of the full task), swap.
    `P` = family -> item vector at the proxy's final; `ref` = family -> score."""
    fams = attrs.loc[sorted(set(runs) & set(P) & set(ref))].sort_values(["L", "arch", "scheme"]).index.to_numpy()
    halves = fams[::2], fams[1::2]                          # alternate along the sorted axes: stratified
    out = {"da_heldout_ab": np.nan, "da_heldout_ba": np.nan, "da_full": [], "da_random_subset": [],
           "n_pairs_heldout": 0, "best_n_heldout": []}
    for tag, (sel, held) in zip(("ab", "ba"), (halves, halves[::-1])):
        rows = np.isin(runs, sel)
        sw = sweep(A[rows], runs[rows], rng)
        hp = [(a, b) for a, b in pairs if a in set(held) and b in set(held)]
        if "subset" not in sw or len(hp) < MIN_PAIRS:                   # rule 5
            out["n_pairs_heldout"] = max(out["n_pairs_heldout"], len(hp))
            continue
        r = {f: ref[f] for f in held}
        da, n = pair_agreement({f: P[f][sw["subset"]].mean() for f in held}, r, hp)
        out[f"da_heldout_{tag}"] = da
        out["n_pairs_heldout"] = n
        out["da_full"].append(pair_agreement({f: P[f].mean() for f in held}, r, hp)[0])
        out["best_n_heldout"].append(sw["best_n"])
        out["da_random_subset"].append(np.mean([
            pair_agreement({f: P[f][idx].mean() for f in held}, r, hp)[0]
            for idx in (rng.choice(A.shape[1], sw["best_n"], replace=False) for _ in range(RANDOM_DRAWS))]))
    for k in ("da_full", "da_random_subset", "best_n_heldout"):
        out[k] = float(np.mean(out[k])) if out[k] else np.nan
    both = [v for v in (out["da_heldout_ab"], out["da_heldout_ba"]) if not np.isnan(v)]
    out["da_subset_heldout"] = float(np.mean(both)) if both else np.nan
    return out


def figure(summary: pd.DataFrame, out_dir: Path, pool: str, note: str) -> None:
    ok = summary[~summary["gated"]]
    sizes = size_order(summary["size"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1.1, 1]})
    gated = summary.groupby(["family", "size"])["gated"].all().unstack().reindex(columns=sizes)
    by = ok.groupby(["family", "size"])
    tables = [
        G.matrix_ax(axes[0], by["dead_share"].mean().unstack().reindex(columns=sizes), "Dead items (share of the benchmark)",
                    cnt=by["dead_share"].count().unstack().reindex(columns=sizes), gated=gated, xlabel="model size", ylabel="benchmark family"),
        G.matrix_ax(axes[1], by["gain_over_null"].median().unstack().reindex(columns=sizes), "SNR gain of the best subset over the null (median)",
                    cnt=by["gain_over_null"].count().unstack().reindex(columns=sizes), gated=gated, vmin=-2, vmax=2, center=0.0,
                    cmap=S.DIV, fmt="{:+.2f}", xlabel="model size")]
    ax = axes[2]
    da = ok[ok["size"] != TARGET_SIZE]
    rows = []
    for col, label, c in (("da_full", "full set", S.INK), ("da_subset_heldout", "subset chosen on the other half", S.SERIES[0]),
                          ("da_random_subset", "random subset, same size", S.SERIES[1])):
        m = da.groupby("size")[col].mean().reindex([s for s in sizes if s != TARGET_SIZE])
        ax.plot(range(len(m)), m.to_numpy(), "-o", color=c, lw=1.4, ms=4, label=label)
        rows += [{"panel": "held-out DA", "row": s, "col": label, "value": v} for s, v in m.items()]
    n = da.groupby("size")["da_subset_heldout"].count().reindex(m.index)
    ax.set_xticks(range(len(m))); ax.set_xticklabels([f"{s}\n({int(k)} tasks)" for s, k in zip(m.index, n.fillna(0))])
    ax.set_ylim(0.4, 1.0); ax.axhline(0.5, color=S.MUTED, lw=.8, ls=":"); S.clean(ax)
    ax.set_title(f"Held-out decision accuracy vs the {TARGET_SIZE} final", loc="left", fontsize=8.5)
    ax.set_ylabel("decision accuracy (mean over tasks)", fontsize=7.5); ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    tables.append(pd.DataFrame(rows, columns=["panel", "row", "col", "value"]))
    G.save_highlights(fig, out_dir, f"rq08 per item: a subset of items against the full benchmark ({pool})", note, tables,
                      name="per_item_ladder")


def main(pool: str, store: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = SUBSET_SELECTION / stage / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    df = ladder_frame(pool)                                   # rules 2, 6 and 10 applied at load
    df = df[df["kind"] == "benchmark"]
    ref = finals(df[df["size"] == TARGET_SIZE])               # the reference: the full task's final, per family
    ref = {t: dict(zip(g["family"], g["primary_score"])) for t, g in ref.groupby("task")}
    grid = df[df["seed"] == GRID_SEED]                        # the design variants the per-item matrices are over
    attrs = design_axes(grid)
    pairs = pair_sets(attrs)["multi-axis"]
    win = noise_checkpoints(grid)[["model", "step", "task"]].assign(is_win=True)          # rule 4
    fin = finals(grid)[["model", "step", "task"]].assign(is_fin=True)
    keys = win.merge(fin, how="outer").fillna(False).astype({"is_win": bool, "is_fin": bool})
    keys = keys.merge(grid[["model", "size", "family"]].drop_duplicates(), on="model")
    mask = load_mask(pool)
    parts = sorted((STORE / store).glob("*.parquet"))
    print(f"{pool}: store {STORE / store}: {len(parts)} families; {keys['task'].nunique()} tasks x {keys['model'].nunique()} "
          f"grid-seed runs in scope, {int(keys['is_win'].sum())} window rows, {len(pairs)} multi-axis pairs")
    rows, items = [], []
    for part in parts:
        s = pd.read_parquet(part, columns=["model", "step", "task", "doc_id", "acc", "acc_norm"],
                            filters=[("model", "in", sorted(set(keys["model"]))), ("step", "in", sorted(set(keys["step"])))])
        s = s.merge(keys, on=["model", "step", "task"])
        for (task, size), g in s.groupby(["task", "size"], sort=True):
            metric = metric_for(task) or "acc"
            M = g.pivot_table(index="doc_id", columns=["model", "step"], values=metric).dropna()
            cols = pd.DataFrame(M.columns.tolist(), columns=["model", "step"]).merge(keys[keys["task"] == task], how="left")
            w = cols["is_win"].to_numpy(dtype=bool)
            A, runs = M.to_numpy(dtype=np.float32)[:, w].T, cols.loc[w, "family"].to_numpy()
            if M.empty or pd.Series(runs).value_counts().ge(2).sum() < 2:
                print(f"  {task} {size}: {len(M)} items, {len(set(runs))} runs in the window: too few runs with >= 2 window checkpoints")
                continue
            sw = sweep(A, runs, rng)
            P = {f: M.to_numpy(dtype=np.float32)[:, i] for i, f in zip(np.flatnonzero(cols["is_fin"]), cols.loc[cols["is_fin"], "family"])}
            da = heldout_da(A, runs, P, ref.get(task, {}), pairs, attrs, rng)
            gated = not passes_gate(mask, [task], size).iloc[0]
            gated_ref = not passes_gate(mask, [task], size, TARGET_SIZE).iloc[0]
            row = {"task": task, "family": benchmark_family(task), "size": size, "n_items": len(M), "n_runs": len(set(runs)),
                   "n_window": int(w.sum()), "dead_share": float(sw["dead"].mean()), "snr_full": sw["snr_full"],
                   "snr_best": sw["snr_best"], "best_n": sw["best_n"], "snr_random_order": sw["snr_random_order"],
                   "null_p95": sw["null_p95"], "gain_over_null": sw["snr_best"] - sw["null_p95"], **da,
                   "gated": gated, "gated_ref": gated_ref}
            if gated:                                          # rule 1: at chance at this size, kept and blanked
                row.update({k: np.nan for k in ("dead_share", "snr_full", "snr_best", "snr_random_order", "null_p95", "gain_over_null")})
            if gated or gated_ref:
                row.update({k: np.nan for k in ("da_full", "da_subset_heldout", "da_random_subset", "da_heldout_ab", "da_heldout_ba")})
            rows.append(row)
            items.append(pd.DataFrame({"task": task, "size": size, "doc_id": M.index, "snr": np.round(sw["snr"], 4),
                                       "dead": sw["dead"], "gated": gated}))
            print(f"  {task} {size}: {len(M)} items, {len(set(runs))} runs, dead {row['dead_share'] if not gated else float('nan'):.2f}, "
                  f"SNR {fmt(row['snr_full'])} -> {fmt(row['snr_best'])} (n={sw['best_n']}, null p95 {fmt(row['null_p95'])}), "
                  f"held-out DA full {fmt(row['da_full'])} subset {fmt(row['da_subset_heldout'])} random {fmt(row['da_random_subset'])} "
                  f"over {da['n_pairs_heldout']} pairs{' [gated]' if gated else ''}")
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "per_item_summary.csv", index=False)
    (pd.concat(items) if items else pd.DataFrame(columns=["task", "size", "doc_id", "snr", "dead", "gated"])).to_csv(
        out_dir / "per_item_snr.csv", index=False)
    if summary.empty:
        print("!!! no (task, size) cell has two runs with two window checkpoints in the store: no figure, no README block")
        return
    ok = summary[~summary["gated"]]
    note = (f"Grid-seed ({GRID_SEED}) design variants, {ok['task'].nunique()} tasks over {ok['n_runs'].max()} runs at most; "
            f"per-item SNR on the rule-4 window (80-100 % of the run, k/20 points); dead = the same mean outcome in every run. "
            f"Gain = best-prefix SNR minus the 95th percentile of {NULL_DRAWS} random subsets of the same size. Held-out DA: items "
            f"chosen on half of the families (stratified over L, arch, scheme), scored on the other half's multi-axis pairs "
            f"(>= {MIN_PAIRS}, proxy final vs the {TARGET_SIZE} final of the full task), both halves averaged; random = {RANDOM_DRAWS} "
            f"draws. Grey = at chance at that size (rule 1); the task set differs across sizes (counts in the cells, rule 13).")
    figure(summary, out_dir, pool, note)
    if pool != CANONICAL_POOL:
        return
    top = ok.dropna(subset=["gain_over_null"]).sort_values("gain_over_null", ascending=False).head(10)
    da = ok[ok["size"] != TARGET_SIZE].groupby("size")[["da_full", "da_subset_heldout", "da_random_subset"]].agg(["mean", "count"])
    body = "\n\n".join([
        "## Per item on the ladder",
        f"Per-item subset selection over the `{pool}` design variants, from the per-item store (`build_per_item_store.sbatch`). "
        f"Regenerate with `python analysis/rq08_subset_selection/per_item_ladder.py --pool {pool}`. {note}",
        md_table(["task", "size", "items", "dead", "full -> best SNR", "best n", "null p95", "held-out DA full / subset / random"],
                 [[f"`{r.task}`", r.size, r.n_items, fmt(r.dead_share), f"{fmt(r.snr_full)} -> {fmt(r.snr_best)}", r.best_n,
                   fmt(r.null_p95), f"{fmt(r.da_full)} / {fmt(r.da_subset_heldout)} / {fmt(r.da_random_subset)}"] for r in top.itertuples()]),
        md_table(["size", "tasks", "DA full set", "DA held-out subset", "DA random subset"],
                 [[s, int(v[("da_subset_heldout", "count")]), fmt(v[("da_full", "mean")]), fmt(v[("da_subset_heldout", "mean")]),
                   fmt(v[("da_random_subset", "mean")])] for s, v in da.reindex(size_order(da.index)).iterrows()]),
        f"![per-item ladder]({stage}/{pool}/per_item_ladder.png)"])
    replace_block(SUBSET_SELECTION / "README.md", "per-item-ladder", body, f"per_item_ladder.py --pool {pool}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL)
    ap.add_argument("--store", default="predictivity_seeds", help="the per_item_store/<store> folder to read (a superset pool)")
    a = ap.parse_args()
    main(a.pool, a.store)
