"""Does a benchmark that scales cleanly also rank the design variants like
the reference? Two Spearman correlations that are easy to confuse.

rq01's ρ is the correlation between MODEL SIZE and a task's score along the
ladder (the size fit; `scaling_regimes.csv`, `rho_size`, the median over the
L settings, and `r2_size` its R²). rq02's ρ is the correlation between the
PROXY'S RANKING of the design variants and the 1.7B ranking on the same task
(`agreement_per_cell.csv`, `rho`, one value per proxy size; `da` is the
same agreement as a pair share). The first says the benchmark moves with
scale, the second that it moves with the design differences the way the
reference does. This script puts the two side by side per task and reports
the Spearman correlation ACROSS TASKS between them, per proxy size: a
benchmark can climb smoothly with size and still order the variants at
chance, and the number here says how often.

Population: the tasks in both tables — gated at the proxy and at the
reference (rule 1) with ≥ MIN_PAIRS pairs (rule 5) on the rq02 side, a
regime (≥ 2 L settings with a size fit) on the rq01 side.

    scaling_vs_ranking.png / .csv   (a) rq01 ρ against rq02 ρ at each proxy size, one point
                                    per task; (b) rq01 R² against rq02 DA-size; the Spearman
                                    across tasks in each corner
    python analysis/rq02_decision_accuracy/scaling_vs_ranking.py --pool predictivity
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

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY, SCALING_PREDICTABILITY  # noqa: E402
from analysis.utils import TARGET_SIZE, size_order  # noqa: E402

OUT_ROOT = DECISION_ACCURACY
mpl.rcParams.update(S.RC)


def joined(pool: str) -> pd.DataFrame:
    stage = load_pools()[pool].get("stage", "pretraining")
    reg = pd.read_csv(SCALING_PREDICTABILITY / "pretraining" / "predictivity_all" / "scaling_regimes.csv")
    cells = pd.read_csv(OUT_ROOT / stage / pool / "agreement_per_cell.csv")
    t = cells[["task", "size", "family", "language", "rho", "da", "n_models"]].merge(
        reg[["task", "rho_size", "r2_size", "r2_trajectory", "regime"]], on="task")
    return t.rename(columns={"rho": "rho_ranking", "da": "da_size"})


def summary(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for size, g in t.groupby("size"):
        rows.append({"size": size, "tasks": len(g),
                     "spearman_rho_size_vs_rho_ranking": g["rho_size"].corr(g["rho_ranking"], method="spearman"),
                     "spearman_r2_size_vs_da_size": g["r2_size"].corr(g["da_size"], method="spearman"),
                     "spearman_r2_trajectory_vs_da_size": g["r2_trajectory"].corr(g["da_size"], method="spearman"),
                     "da_size_mean_predictable_both": g.loc[g["regime"] == "predictable across both", "da_size"].mean(),
                     "da_size_mean_other_regimes": g.loc[g["regime"] != "predictable across both", "da_size"].mean()})
    out = pd.DataFrame(rows)
    return out.set_index("size").reindex(size_order(out["size"].unique())).reset_index()


def figure(t: pd.DataFrame, sm: pd.DataFrame, path: Path, pool: str) -> None:
    sizes = [s for s in size_order(t["size"].unique()) if s != TARGET_SIZE]
    fig, axes = plt.subplots(2, len(sizes), figsize=(3.4 * len(sizes), 6.6), sharey="row")
    rng = np.random.default_rng(0)
    for j, size in enumerate(sizes):
        g = t[t["size"] == size]
        s = sm.set_index("size").loc[size]
        a, b = axes[0, j], axes[1, j]
        a.scatter(g["rho_size"] + rng.uniform(-.02, .02, len(g)), g["rho_ranking"], s=9, color=S.RAMP[2], alpha=.55, lw=0)
        a.axhline(0, color=S.MUTED, lw=.8, ls=":")
        a.text(0.03, 0.97, f"{size} proxy\nSpearman across tasks = {s['spearman_rho_size_vs_rho_ranking']:.2f}\n{int(s['tasks'])} tasks",
               transform=a.transAxes, va="top", fontsize=7, bbox=dict(boxstyle="round,pad=0.3", fc=S.SURFACE, ec=S.GRID, lw=.6))
        a.set_xlabel("rq01 ρ: score vs model size")
        a.set_xlim(-1.05, 1.05); a.grid(color=S.GRID, lw=.6); S.clean(a)
        b.scatter(g["r2_size"], g["da_size"], s=9, color=S.RAMP[3], alpha=.55, lw=0)
        b.axhline(0.5, color=S.MUTED, lw=.8, ls=":")
        b.text(0.03, 0.97, f"Spearman across tasks = {s['spearman_r2_size_vs_da_size']:.2f}\n"
                            f"DA-size, 'predictable across both' {s['da_size_mean_predictable_both']:.2f}\nother regimes {s['da_size_mean_other_regimes']:.2f}",
               transform=b.transAxes, va="top", fontsize=7, bbox=dict(boxstyle="round,pad=0.3", fc=S.SURFACE, ec=S.GRID, lw=.6))
        b.set_xlabel("rq01 R²: log-N size fit")
        b.set_xlim(-0.02, 1.02); b.grid(color=S.GRID, lw=.6); S.clean(b)
    axes[0, 0].set_ylabel(f"rq02: Spearman ρ, proxy ranking vs {TARGET_SIZE} ranking")
    axes[1, 0].set_ylabel(f"rq02: DA-size, proxy → {TARGET_SIZE}")
    top = G._header(fig, "Scaling cleanly and ranking the variants like the reference are different properties",
                    f"One point per benchmark task with both a scaling regime (rq01: ≥ 2 language settings with a size "
                    f"fit on the gated rungs, deep scheme-A seed-1904 cells) and a decision-accuracy cell at the proxy "
                    f"(rq02: every grid-seed design-variant pair of every scheme, ≥ 3 pairs, above chance at the proxy "
                    f"and at {TARGET_SIZE}). Top: rq01's ρ (score against model size along the ladder, jittered: it "
                    f"lives on a lattice) against rq02's ρ (the proxy's ranking of the variants against the reference's). "
                    f"Bottom: rq01's R² against DA-size. The corner number is the Spearman correlation across tasks "
                    f"between the two properties. Gate: `{pool}`.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, out_dir: Path, sm: pd.DataFrame) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    rows = [[r["size"], int(r["tasks"]), f"{r['spearman_rho_size_vs_rho_ranking']:+.2f}", f"{r['spearman_r2_size_vs_da_size']:+.2f}",
             f"{r['spearman_r2_trajectory_vs_da_size']:+.2f}", f"{r['da_size_mean_predictable_both']:.2f}", f"{r['da_size_mean_other_regimes']:.2f}"]
            for _, r in sm.iterrows() if r["size"] != TARGET_SIZE]
    body = "\n\n".join([
        "## Scaling utility against ranking utility",
        f"Per proxy size, the Spearman correlation ACROSS TASKS between rq01's scaling statistics (ρ of score with "
        f"model size; R² of the size fit; R² of the trajectory fit) and rq02's ranking statistics (ρ of the proxy's "
        f"ranking of the design variants with the {TARGET_SIZE} ranking; DA-size), and the mean DA-size of the tasks "
        f"rq01 calls 'predictable across both' against the rest. Regenerate with "
        f"`python analysis/rq02_decision_accuracy/scaling_vs_ranking.py --pool {pool}`.",
        md_table(["proxy", "tasks", "ρ_size vs ρ_ranking", "R²_size vs DA-size", "R²_traj vs DA-size",
                  "DA-size, predictable both", "DA-size, other regimes"], rows),
        f"![Scaling against ranking]({stage}/{pool}/scaling_vs_ranking.png)"])
    replace_block(OUT_ROOT / "README.md", "scaling-vs-ranking", body, f"scaling_vs_ranking.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    args = p.parse_args()
    out_dir = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    t = joined(args.pool)
    t.to_csv(out_dir / "scaling_vs_ranking.csv", index=False)
    sm = summary(t)
    print(sm.round(3).to_string(index=False))
    figure(t, sm, out_dir / "scaling_vs_ranking.png", args.pool)
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, out_dir, sm)
