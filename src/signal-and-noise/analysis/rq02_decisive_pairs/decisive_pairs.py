"""DA-size on the decisions the reference actually makes.

Decision accuracy counts every pair of design variants, including pairs the
1.7B reference itself cannot separate: when two variants' reference scores
differ by less than seed noise, the reference's order is a coin flip, and a
proxy that "misses" it has missed nothing. Those pairs pull every DA toward
0.5. This script keeps, per task, the pairs whose reference gap exceeds K seed
standard deviations of a difference of two runs (sqrt(2) x the task's seed sd
of one run's final score), for K in (1, 2), and reads DA-size on them beside
the DA over every pair.

The seed sd is rq05's (`rq05_design_decisions.analyze.seed_sd`): per task, the
median over the baseline (deep, data A) cells with replicate seeds of the sd of
the final score. Replicates exist at 175M, 600M and 1B (none at the 1.7B
reference), so the threshold assumes the seed noise of a run's final score
does not grow with size (rq03 measures it near size-invariant). A task without
replicates has no sd and no decisive pair (the bBPB twins: their store holds
seed 1904 only). Every line, the all-pairs one included, reads the same tasks:
those with an sd and at least MIN_PAIRS decisive pairs at the largest K (rule
5 at the strictest threshold), so the lines differ by their pairs alone (rule
13). Rule 1 at the proxy and at the reference, the populations of rq02.

    decisive_pairs_da_size_both_axes.png / .csv   per (axes, population, size, K): pooled DA-size over the
                                                   kept pairs, the share of comparable pairs kept, the tasks
    decisive_pairs_da_size_paper.png / .csv the multi-axis reading for the paper (rule 18)
    python analysis/rq02_decisive_pairs/decisive_pairs.py --pool predictivity
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
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import DECISIVE_PAIRS  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq02_decision_accuracy.compute_da import pair_agree, score_cube  # noqa: E402
from analysis.rq05_design_decisions.analyze import seed_sd  # noqa: E402
from analysis.utils import (  # noqa: E402
    MIN_PAIRS, SMALL_SIZES, TARGET_SIZE, design_axes, finals, ladder_frame, pair_sets, passes_gate)

KS = (0, 1, 2)                       # 0 = every comparable pair
SEED_POOL = "predictivity_seeds"     # the pool with the replicate seeds the sd is read from
AXES = ("multi-axis", "mono-axis")
POPULATIONS = {"all benchmarks": "Benchmarks", "bpb": "BPB"}
NAME = "decisive_pairs_da_size"
mpl.rcParams.update(S.RC)


def label(k: int, paper: bool = False) -> str:
    if k == 0:
        return "All pairs" if paper else "every pair"
    return f"Gap > {k} seed sd of a difference" if paper else f"reference gap > {k} x sqrt(2) seed sd"


def run(pool: str) -> pd.DataFrame:
    df, tasks, fams, col, S_, P = score_cube(pool)
    sd = seed_sd(finals(ladder_frame(SEED_POOL))).reindex(tasks).to_numpy()
    has_sd = np.isfinite(sd) & (sd > 0)      # every line reads the same tasks: the ones a threshold exists for
    print(f"seed sd: {has_sd.sum()} of {len(tasks)} tasks have replicate seeds")
    mask = load_mask(pool)
    meta = G.add_meta(pd.DataFrame({"task": tasks}))
    meta = meta[meta["family"] != "loss"]
    pop = pd.Series(np.where(meta["family"] == "bpb", "bpb", "all benchmarks"), index=meta.index)
    psets = pair_sets(design_axes(df))
    fi = {f: i for i, f in enumerate(fams)}
    rows = []
    for axes in AXES:
        I = np.array([fi[a] for a, b in psets[axes]])
        J = np.array([fi[b] for a, b in psets[axes]])
        rc = col[(TARGET_SIZE, 1.0)]
        gap = np.abs(S_[:, I, rc] - S_[:, J, rc])
        for size in SMALL_SIZES:
            A, V = pair_agree(S_, P, I, J, col[(size, 1.0)], rc)
            ok = passes_gate(mask, tasks, size, TARGET_SIZE).to_numpy()                   # rule 1
            n_all = V.sum(1)
            keeps = {k: V & (gap > k * np.sqrt(2) * sd[:, None]) if k else V for k in KS}
            cell = ok & has_sd & (keeps[max(KS)].sum(1) >= MIN_PAIRS)    # rule 5 at the strictest K: one task set per line
            for k, keep in keeps.items():
                m, n = (A & keep).sum(1), keep.sum(1)
                for p, idx in pop.groupby(pop).groups.items():
                    t = np.intersect1d(idx, np.flatnonzero(cell))
                    rows.append({"axes": axes, "population": p, "size": size, "k": k, "n_tasks": len(t),
                                 "n_pairs": int(n[t].sum()), "da": m[t].sum() / n[t].sum() if n[t].sum() else np.nan,
                                 "da_cell_mean": float(np.mean(m[t] / n[t])) if len(t) else np.nan,
                                 "share_kept": n[t].sum() / n_all[t].sum() if n_all[t].sum() else np.nan})
    return pd.DataFrame(rows)


def draw(axs, t: pd.DataFrame, axes: str, paper: bool) -> None:
    x = np.arange(len(SMALL_SIZES))
    shades = {0: S.INK, 1: S.RAMP[2], 2: S.RAMP[0]}
    for ax, (p, lab) in zip(axs, POPULATIONS.items()):
        for k in KS:
            g = t[(t["axes"] == axes) & (t["population"] == p) & (t["k"] == k)].set_index("size").reindex(SMALL_SIZES)
            ax.plot(x, g["da"], "-o", ms=4, lw=1.5, color=shades[k], label=label(k, paper))
        ax.axhline(0.5, color=S.MUTED, lw=.7, ls=":")
        ax.set_title(lab, loc="left", fontsize=9); ax.set_ylim(0.3, 1.0)
        ax.set_xticks(x); ax.set_xticklabels(SMALL_SIZES); ax.set_xlabel("Proxy size" if paper else "proxy size")
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    axs[0].set_ylabel(f"Decision accuracy against {TARGET_SIZE}" if paper else f"DA-size against the {TARGET_SIZE} final (pooled)")
    axs[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    ax = axs[-1]
    for p, lab in POPULATIONS.items():
        for k in KS[1:]:
            g = t[(t["axes"] == axes) & (t["population"] == p) & (t["k"] == k)].set_index("size").reindex(SMALL_SIZES)
            ax.plot(x, g["share_kept"], "-o" if p == "bpb" else "--o", ms=3.5, lw=1.3, color=shades[k],
                    label=f"{lab}, {label(k, paper).lower() if paper else label(k)}")
    ax.set_ylim(0, 1); ax.set_xticks(x); ax.set_xticklabels(SMALL_SIZES); ax.set_xlabel("Proxy size" if paper else "proxy size")
    ax.set_ylabel("Share of comparable pairs kept" if paper else "share of the comparable pairs kept")
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="lower left"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)


def figures(t: pd.DataFrame, out_dir: Path, pool: str) -> None:
    fig, grid = plt.subplots(len(AXES), 3, figsize=(12, 3.4 * len(AXES)), squeeze=False)
    for row, axes in zip(grid, AXES):
        draw(row, t, axes, paper=False)
        for ax, lab in zip(row[:2], POPULATIONS.values()):
            ax.set_title(f"{lab}, {axes} pairs", loc="left", fontsize=9)
    top = G._header(fig, "Decision accuracy on the decisions the reference makes",
                    f"pooled DA-size (matching over comparable pairs) over the gated cells, pool {pool}, on every pair and on the "
                    f"pairs whose {TARGET_SIZE} gap exceeds K x sqrt(2) x the task's seed sd (rq05's: median over the replicated "
                    f"baseline cells, 175M-1B); every line reads the tasks with >= {MIN_PAIRS} pairs at the largest K; right: share "
                    "of the comparable pairs kept; task counts in the CSV")
    fig.tight_layout(rect=(0, 0, 1, top))
    t.to_csv(out_dir / f"{NAME}_both_axes.csv", index=False)
    S.save(fig, out_dir / f"{NAME}_both_axes.png", dpi=150)
    fig, row = plt.subplots(1, 3, figsize=(11, 3.3))
    draw(row, t, "multi-axis", paper=True)
    fig.tight_layout()
    t[t["axes"] == "multi-axis"].to_csv(out_dir / f"{NAME}_paper.csv", index=False)
    S.save_paper(fig, out_dir / f"{NAME}_paper")


def generate_readme(pool: str, t: pd.DataFrame) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    rows = []
    for (axes, p, k), g in t.groupby(["axes", "population", "k"], sort=False):
        g = g.set_index("size").reindex(SMALL_SIZES)
        rows.append([axes, POPULATIONS[p], label(k), *[f"{r.da:.3f} ({int(r.n_tasks)}; {r.share_kept:.0%})"
                                                       if np.isfinite(r.da) else "" for r in g.itertuples()]])
    body = "\n\n".join([
        "## Results",
        f"Pool `{pool}`, seed sd from `{SEED_POOL}`. A pair counts when both proxies and both reference runs have a "
        f"score; a cell is gated at the proxy and at {TARGET_SIZE} (rule 1), and every line reads the tasks with {MIN_PAIRS} "
        f"decisive pairs at the largest K (rule 5), so the lines differ by their pairs alone. "
        f"Regenerate with `python analysis/rq02_decisive_pairs/decisive_pairs.py --pool {pool}`.",
        f"![DA-size on decisive pairs]({stage}/{pool}/{NAME}_both_axes.png)",
        "Pooled DA-size (tasks; share of the comparable pairs kept):",
        md_table(["pairs", "population", "pairs kept", *SMALL_SIZES], rows),
        f"Table: `{NAME}_both_axes.csv`."])
    replace_block(DECISIVE_PAIRS / "README.md", "decisive-pairs", body, f"decisive_pairs.py --pool {pool}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL)
    args = ap.parse_args()
    out_dir = DECISIVE_PAIRS / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    out_dir.mkdir(parents=True, exist_ok=True)
    t = run(args.pool)
    print(t.round(3).to_string(index=False))
    figures(t, out_dir, args.pool)
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, t)
