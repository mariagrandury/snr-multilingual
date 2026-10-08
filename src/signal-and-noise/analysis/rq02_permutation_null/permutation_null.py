"""Each DA-size cell against its own no-signal null.

Decision accuracy is read against a flat 0.5, but that is not what a proxy with
no signal scores in a given cell. A pair the proxy ties is a miss unless the
reference ties it too, which pulls an uninformative proxy below 0.5, and the
cell's pair structure (how many families it holds, how many pairs share one)
sets how far a lucky ordering can reach. So every (task, proxy size, pair set)
cell gets its own null: the proxy's final scores are shuffled across the
cell's families PERMUTATIONS times, and each shuffle is scored with the
pipeline's kernel (`compute_da.pair_agree`: the sign rule, ties included)
against the unchanged 1.7B ranking.

Per cell: the observed DA-size, the null mean and 95th percentile, the
one-sided permutation p (how often a shuffle agrees at least as well), and its
Benjamini-Hochberg q over the gated cells of one pair set. Pooled per (pair
set, population, size): the pooled DA-size (matching pairs over comparable
pairs, as `scale_convergence.py` pools) against the pooled null, whose
distribution is the sum of the cells' independent shuffles (the band is its
2.5-97.5 % range). Rule 1 at the proxy and at the reference, rule 5 (cells
under MIN_PAIRS pairs are dropped), the populations of rq02 (every benchmark
variant, the bBPB twins included, and the per-language BPB).

    permutation_null_da_size_per_task_both_axes.csv  per (task, axes, size): DA-size, pairs, null mean,
                                                      null p95, p, q, gated
    permutation_null_da_size_both_axes.png / .csv    per (axes, population, size): pooled DA-size, the pooled
                                                      null and its band, the share of cells above their null
    permutation_null_da_size_paper.png / .csv the multi-axis reading for the paper (rule 18)
    python analysis/rq02_permutation_null/permutation_null.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import PERMUTATION_NULL  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq02_decision_accuracy.compute_da import pair_agree, score_cube  # noqa: E402
from analysis.utils import (  # noqa: E402
    AXES_SUFFIX, MIN_PAIRS, SMALL_SIZES, TARGET_SIZE, design_axes, pair_sets, passes_gate)

PERMUTATIONS = 1000
SEED = 1904
Q = 0.05
AXES = ("multi-axis", "mono-axis")
POPULATIONS = {"all benchmarks": "Benchmarks", "bpb": "BPB"}
NAME = "permutation_null_da_size"
mpl.rcParams.update(S.RC)


def cell_nulls(S_, P, I, J, pc: int, rc: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per task of the cube: (matching pairs, comparable pairs, the matching
    pairs of each of PERMUTATIONS shuffles of the proxy's scores across the
    families the cell reads). The reference side is never shuffled."""
    A, V = pair_agree(S_, P, I, J, pc, rc)
    null = np.zeros((len(A), PERMUTATIONS), dtype=np.int32)
    for t in np.flatnonzero(V.sum(1) >= MIN_PAIRS):
        k = np.flatnonzero(V[t])
        fams = np.unique(np.r_[I[k], J[k]])
        pos = {f: i for i, f in enumerate(fams)}
        a, b = np.array([pos[x] for x in I[k]]), np.array([pos[x] for x in J[k]])
        x = rng.permuted(np.tile(S_[t, fams, pc], (PERMUTATIONS, 1)), axis=1)
        ref = np.sign(S_[t, I[k], rc] - S_[t, J[k], rc])
        null[t] = (np.sign(x[:, a] - x[:, b]) == ref).sum(1)
    return A.sum(1), V.sum(1), null


def run(pool: str):
    df, tasks, fams, col, S_, P = score_cube(pool)
    mask = load_mask(pool)
    psets = pair_sets(design_axes(df))
    fi = {f: i for i, f in enumerate(fams)}
    rng = np.random.default_rng(SEED)
    cells, pooled = [], []
    for axes in AXES:
        I = np.array([fi[a] for a, b in psets[axes]])
        J = np.array([fi[b] for a, b in psets[axes]])
        for size in SMALL_SIZES:
            m, n, null = cell_nulls(S_, P, I, J, col[(size, 1.0)], col[(TARGET_SIZE, 1.0)], rng)
            keep = n >= MIN_PAIRS                                                     # rule 5
            c = pd.DataFrame({"task": np.array(tasks)[keep], "axes": axes, "size": size, "n_matching": m[keep],
                              "n_pairs": n[keep]})
            nk = null[keep]
            c["da"] = c["n_matching"] / c["n_pairs"]
            c["null_mean"] = nk.mean(1) / c["n_pairs"]
            c["null_p95"] = np.percentile(nk, 95, axis=1) / c["n_pairs"]
            c["p"] = (1 + (nk >= c["n_matching"].to_numpy()[:, None]).sum(1)) / (PERMUTATIONS + 1)
            c["gated"] = ~passes_gate(mask, c["task"], size, TARGET_SIZE).to_numpy()   # rule 1
            c = G.add_meta(c)                                                         # drops the aggregates (rule 7)
            c = c[c["family"] != "loss"]
            c["population"] = np.where(c["family"] == "bpb", "bpb", "all benchmarks")
            cells.append(c)
            nk = null[keep][c.index]
            for pop, g in c[~c["gated"]].groupby("population"):
                draws = nk[g.index.map(c.index.get_loc)].sum(0) / g["n_pairs"].sum()
                pooled.append({"axes": axes, "population": pop, "size": size, "n_tasks": len(g),
                               "n_pairs": int(g["n_pairs"].sum()), "da": g["n_matching"].sum() / g["n_pairs"].sum(),
                               "null_mean": draws.mean(), "null_lo": np.percentile(draws, 2.5),
                               "null_hi": np.percentile(draws, 97.5), "da_cell_mean": g["da"].mean(),
                               "null_cell_mean": g["null_mean"].mean()})
    cells = pd.concat(cells, ignore_index=True)
    cells["q"] = np.nan
    for axes, g in cells[~cells["gated"]].groupby("axes"):
        cells.loc[g.index, "q"] = multipletests(g["p"].to_numpy(), method="fdr_bh")[1]
    pooled = pd.DataFrame(pooled)
    sig = (cells[~cells["gated"]].assign(above=lambda x: x["q"] < Q).groupby(["axes", "population", "size"])["above"]
           .mean().rename("share_above_null").reset_index())
    return cells, pooled.merge(sig, on=["axes", "population", "size"], how="left")


def draw(axs, pooled: pd.DataFrame, axes: str, paper: bool) -> None:
    x = np.arange(len(SMALL_SIZES))
    for ax, (pop, lab) in zip(axs, POPULATIONS.items()):
        g = pooled[(pooled["axes"] == axes) & (pooled["population"] == pop)].set_index("size").reindex(SMALL_SIZES)
        ax.fill_between(x, g["null_lo"], g["null_hi"], color=S.GRID, lw=0, label="No signal null, 95% band" if paper
                        else "null: proxy scores shuffled, 95 % band")
        ax.plot(x, g["null_mean"], color=S.MUTED, lw=1.1, ls="--", label="Null mean" if paper else "null mean")
        ax.plot(x, g["da"], "-o", color=S.INK, ms=4, lw=1.6, label="Observed" if paper else "observed")
        ax.axhline(0.5, color=S.MUTED, lw=.7, ls=":")
        ax.set_title(lab, loc="left", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(SMALL_SIZES)
        ax.set_xlabel("Proxy size" if paper else "proxy size")
        ax.set_ylim(0.3, 1.0); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    axs[0].set_ylabel(f"DA-size against the {TARGET_SIZE} final" if not paper else f"Decision accuracy against {TARGET_SIZE}")
    axs[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    ax = axs[-1]
    for pop, lab in POPULATIONS.items():
        g = pooled[(pooled["axes"] == axes) & (pooled["population"] == pop)].set_index("size").reindex(SMALL_SIZES)
        ax.plot(x, g["share_above_null"], "-o", ms=4, lw=1.4, color=S.SERIES[0] if pop == "bpb" else S.INK, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(SMALL_SIZES); ax.set_ylim(0, 1)
    ax.set_xlabel("Proxy size" if paper else "proxy size")
    ax.set_ylabel(f"Share of cells above their null (q < {Q:g})" if paper else f"share of cells above their null, BH q < {Q:g}")
    ax.legend(fontsize=6.5, frameon=False); ax.grid(color=S.GRID, lw=.6); S.clean(ax)


def figures(pooled: pd.DataFrame, out_dir: Path, pool: str) -> None:
    fig, grid = plt.subplots(len(AXES), 3, figsize=(12, 3.4 * len(AXES)), squeeze=False)
    for row, axes in zip(grid, AXES):
        draw(row, pooled, axes, paper=False)
        row[0].set_title(f"{POPULATIONS['all benchmarks']}, {axes} pairs", loc="left", fontsize=9)
        row[1].set_title(f"{POPULATIONS['bpb']}, {axes} pairs", loc="left", fontsize=9)
    top = G._header(fig, "Decision accuracy against its own no-signal null",
                    f"pooled DA-size (matching over comparable pairs) of the gated cells with >= {MIN_PAIRS} pairs, pool {pool}, "
                    f"against the same cells with the proxy's final scores shuffled across their families ({PERMUTATIONS} shuffles, "
                    "the pipeline's sign rule with ties); band = 2.5-97.5 % of the pooled null; right: share of cells whose one-sided "
                    f"permutation p clears BH q < {Q:g}; the task count moves along a line (rule 13, in the CSV)")
    fig.tight_layout(rect=(0, 0, 1, top))
    pooled.to_csv(out_dir / f"{NAME}_both_axes.csv", index=False)
    S.save(fig, out_dir / f"{NAME}_both_axes.png", dpi=150)
    fig, row = plt.subplots(1, 3, figsize=(11, 3.3))
    draw(row, pooled, "multi-axis", paper=True)
    fig.tight_layout()
    pooled[pooled["axes"] == "multi-axis"].to_csv(out_dir / f"{NAME}_paper.csv", index=False)
    S.save_paper(fig, out_dir / f"{NAME}_paper")


def generate_readme(pool: str, cells: pd.DataFrame, pooled: pd.DataFrame, out_dir: Path) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    rows = []
    for (axes, pop), g in pooled.groupby(["axes", "population"], sort=False):
        g = g.set_index("size").reindex(SMALL_SIZES)
        rows.append([axes, POPULATIONS[pop], *[f"{r.da:.3f} / {r.null_mean:.3f} [{r.null_lo:.3f}, {r.null_hi:.3f}] "
                                               f"/ {r.share_above_null:.0%} ({int(r.n_tasks)})" if np.isfinite(r.da) else ""
                                               for r in g.itertuples()]])
    body = "\n\n".join([
        "## Results",
        f"Pool `{pool}`; per cell (task, proxy size, pair set) {PERMUTATIONS} shuffles of the proxy's final scores across "
        f"the cell's families, scored by the pipeline's sign rule against the unchanged {TARGET_SIZE} ranking. Cells are "
        f"gated at the proxy and at {TARGET_SIZE} (rule 1) and need {MIN_PAIRS} pairs (rule 5). Regenerate with "
        f"`python analysis/rq02_permutation_null/permutation_null.py --pool {pool}`.",
        f"![DA-size against its null]({stage}/{pool}/{NAME}_both_axes.png)",
        "Per cell: observed pooled DA-size / pooled null mean [2.5 %, 97.5 %] / share of cells above their own null at "
        f"BH q < {Q:g} (tasks):",
        md_table(["pairs", "population", *SMALL_SIZES], rows),
        f"Tables: `{NAME}_per_task_both_axes.csv` (every cell's DA, null mean, null p95, p and q) and "
        f"`{NAME}_both_axes.csv` (the pooled lines)."])
    replace_block(PERMUTATION_NULL / "README.md", "permutation-null", body, f"permutation_null.py --pool {pool}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL)
    args = ap.parse_args()
    out_dir = PERMUTATION_NULL / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    out_dir.mkdir(parents=True, exist_ok=True)
    cells, pooled = run(args.pool)
    cells.to_csv(out_dir / f"{NAME}_per_task_both_axes.csv", index=False)
    print(pooled.round(3).to_string(index=False))
    figures(pooled, out_dir, args.pool)
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, cells, pooled, out_dir)
