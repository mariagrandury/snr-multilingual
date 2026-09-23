"""Decision accuracy and its relatives: Kendall's tau, Goodman-Kruskal's gamma,
Spearman's rho and Pearson's r on the same rankings — and the closed form that
ties DA to tau, so the table shows WHY they differ rather than that they do.

Per (task, proxy size) the proxy's final scores of every design variant at the
grid seed are ranked against the reference's finals of the same variants
(DA-size's population, every pair). `utils.agreement_measures` returns the pair
counts C (concordant), D (discordant), T_both (tied on both sides) and T_one
(tied on one), and every statistic is a function of them:

    DA          = (C + T_both) / n       the pipeline's kernel (a both-tied pair agrees)
    tau_a       = (C - D) / n            ties count for neither
    gamma       = (C - D) / (C + D)      ties dropped
    DA_drop_ref = C / (C + D + T_proxy)  rq05's convention: a pair the REFERENCE ties is no decision
    tau_b       scipy's, tie-corrected denominator;  rho, r: rank / raw correlation

    2·DA − 1 = tau_a + (T_both − T_one) / n      exactly, ties included

So DA and Kendall's tau are ONE statistic under two tie conventions, and the
only question with content is whether the convention changes a verdict:
the share of cells that clear the reliability cut under one statistic and not
another (DA ≥ 0.66 maps to tau ≥ 0.32 through the identity).

    agreement_per_cell.csv          one row per (task, size): the counts and every statistic
    agreement_identity.png / .csv   DA against tau_a; the tie-free cells lie on the line, the
                                    tied cells sit off it by exactly (T_both − T_one)/n
    agreement_cut_sensitivity.png / .csv   per size, the share of cells whose reliability verdict
                                    depends on the statistic

    python analysis/rq02_decision_accuracy/agreement.py --pool predictivity
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
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import FILTERS  # noqa: E402
from analysis.utils import (  # noqa: E402
    GRID_SEED, MIN_PAIRS, TARGET_SIZE, agreement_measures, design_axes, finals, ladder_frame, size_order)

OUT_ROOT = DECISION_ACCURACY
POOL = "predictivity_all"          # every scheme at the grid seed, as by_L and scale_convergence pair over
STATS = ["da", "tau_a", "tau_b", "gamma", "da_drop_ref_ties", "rho", "pearson_r"]
# The reliability cut every rq02 `above_66_*` figure uses, and its image under
# the identity for the tau-scaled statistics: DA ≥ 0.66 <=> 2·DA − 1 ≥ 0.32.
CUT = FILTERS["above_66_size"][1]
CUT_OF = {"da": CUT, "da_drop_ref_ties": CUT, "tau_a": 2 * CUT - 1, "tau_b": 2 * CUT - 1,
          "gamma": 2 * CUT - 1, "rho": 2 * CUT - 1}
mpl.rcParams.update(S.RC)


def per_cell(fin: pd.DataFrame, pool: str) -> pd.DataFrame:
    """Every statistic per (task, proxy size), over the grid-seed families with
    a final score at both the proxy and the reference; `pool` names the gate."""
    fams = set(design_axes(fin).query("seed == @GRID_SEED").index)
    fin = fin[fin["family"].isin(fams)]
    ref = fin[fin["size"] == TARGET_SIZE].set_index(["task", "family"])["primary_score"]
    rows = []
    for (task, size), g in fin[fin["size"] != TARGET_SIZE].groupby(["task", "size"], sort=False):
        g = g.set_index("family")["primary_score"]
        both = [f for f in g.index if (task, f) in ref.index]
        if len(both) * (len(both) - 1) // 2 < MIN_PAIRS:               # rule 5
            continue
        rows.append({"task": task, "size": size, **agreement_measures(g.loc[both].to_numpy(), ref.loc[[(task, f) for f in both]].to_numpy())})
    d = G.add_meta(pd.DataFrame(rows))
    d = d[d["family"] != "bpb"]                                         # a chance level is what the cut is read against
    d["gap_pairs"] = (d["tied_both"] - d["tied_one"]) / d["n_pairs"]    # the identity's tie term
    return G.mark_gated(d, pool, "size", "da", TARGET_SIZE).dropna(subset=["da"])   # rule 1, at the proxy and the reference


def cut_sensitivity(d: pd.DataFrame) -> pd.DataFrame:
    """Per size and statistic: how many cells clear the cut, and how many change
    verdict against DA — the number a reliable-task list's author needs."""
    rows = []
    for size, g in d.groupby("size", sort=False):
        base = g["da"] >= CUT
        for st in STATS:
            if st not in CUT_OF:
                continue
            pas = g[st] >= CUT_OF[st]
            rows.append({"size": size, "statistic": st, "cut": CUT_OF[st], "cells": len(g),
                         "pass": int(pas.sum()), "pass_da": int(base.sum()),
                         "flip_vs_da": int((pas != base).sum()), "flip_share": float((pas != base).mean())})
    return pd.DataFrame(rows)


def figure_identity(d: pd.DataFrame, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    free, tied = d[d["tied_both"] + d["tied_one"] == 0], d[d["tied_both"] + d["tied_one"] > 0]
    a.plot([0, 1], [-1, 1], color=S.MUTED, lw=.8, ls=":", zorder=1, label="τ = 2·DA − 1")
    a.scatter(free["da"], free["tau_a"], s=9, color=S.RAMP[3], alpha=.6, lw=0, label=f"{len(free)} cells without a tie", zorder=3)
    a.scatter(tied["da"], tied["tau_a"], s=9, color=S.SERIES[1], alpha=.6, lw=0, label=f"{len(tied)} cells with ≥ 1 tied pair", zorder=2)
    a.set_xlabel("decision accuracy (a both-tied pair agrees, a one-tied pair misses)")
    a.set_ylabel("Kendall τ_a (a tied pair counts for neither)")
    a.legend(fontsize=6.5, frameon=False, loc="upper left"); a.grid(color=S.GRID, lw=.6); S.clean(a)
    a.set_title("the identity", loc="left", fontsize=8.5)
    resid = (2 * d["da"] - 1) - d["tau_a"]
    b.scatter(d["gap_pairs"], resid, s=9, color=S.INK, alpha=.5, lw=0)
    b.plot([-1, 1], [-1, 1], color=S.MUTED, lw=.8, ls=":")
    b.set_xlim(d["gap_pairs"].min() - .02, d["gap_pairs"].max() + .02); b.set_ylim(*b.get_xlim())
    b.set_xlabel("(T_both − T_one) / n_pairs, the tie term"); b.set_ylabel("(2·DA − 1) − τ_a")
    b.set_title(f"the residual is the tie term to {np.abs(resid - d['gap_pairs']).max():.0e}", loc="left", fontsize=8.5)
    b.grid(color=S.GRID, lw=.6); S.clean(b)
    top = G._header(fig, "Decision accuracy IS Kendall's τ under another tie convention",
                    f"One point per (benchmark task, proxy size): the proxy's final ranking of the {GRID_SEED}-seed design "
                    f"variants against the {TARGET_SIZE} final's, every pair, ≥ {MIN_PAIRS} pairs (rule 5), above chance at "
                    f"both sizes (rule 1). Left: DA against τ_a; without ties the two are the same number rescaled. "
                    f"Right: the whole difference is the tie term, cell by cell. Ties are "
                    f"{d['tied_both'].sum() + d['tied_one'].sum():,} of {d['n_pairs'].sum():,} pairs "
                    f"({(d['tied_both'].sum() + d['tied_one'].sum()) / d['n_pairs'].sum():.1%}), "
                    f"{(d['tied_one'].sum() / max(d['tied_both'].sum() + d['tied_one'].sum(), 1)):.0%} of them one-sided, and they touch "
                    f"{(d['tied_both'] + d['tied_one'] > 0).mean():.0%} of the cells. Median {int(d['n_models'].median())} models per cell.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def figure_cuts(sens: pd.DataFrame, d: pd.DataFrame, path: Path) -> None:
    sizes = size_order(sens["size"].unique())
    stats = [s for s in STATS if s in set(sens["statistic"]) and s != "da"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.2, 4.4))
    w = .8 / len(stats)
    # the tau family on the ramp (four tie conventions of one statistic), rho in
    # the categorical slot: it is a different kind of statistic, not a fifth convention
    colour = {"tau_a": S.RAMP[0], "tau_b": S.RAMP[1], "gamma": S.RAMP[2], "da_drop_ref_ties": S.RAMP[3], "rho": S.SERIES[1]}
    for k, st in enumerate(stats):
        g = sens[sens["statistic"] == st].set_index("size").reindex(sizes)
        a.bar(np.arange(len(sizes)) + (k - len(stats) / 2 + .5) * w, g["flip_share"], w, color=colour[st], label=st)
    a.set_xticks(range(len(sizes))); a.set_xticklabels(sizes); a.set_ylabel("share of cells whose verdict differs from DA's")
    a.set_xlabel("proxy size"); a.legend(fontsize=6.5, frameon=False); a.grid(color=S.GRID, lw=.6, axis="y"); S.clean(a)
    a.set_title(f"cells that pass one cut and fail the other (DA ≥ {CUT:g} ↔ τ ≥ {2 * CUT - 1:.2f})", loc="left", fontsize=8.5)
    # the two statistics that are not rescalings: rho and r against DA
    b.scatter(d["da"], d["rho"], s=8, color=S.RAMP[2], alpha=.5, lw=0, label=f"Spearman ρ  (r = {d['da'].corr(d['rho']):.3f})")
    b.scatter(d["da"], d["pearson_r"], s=8, color=S.SERIES[1], alpha=.4, lw=0, label=f"Pearson r on scores  (r = {d['da'].corr(d['pearson_r']):.3f})")
    b.axvline(CUT, color=S.MUTED, lw=.8, ls=":"); b.axhline(2 * CUT - 1, color=S.MUTED, lw=.8, ls=":")
    b.set_xlabel("decision accuracy"); b.set_ylabel("correlation of the two rankings")
    b.legend(fontsize=6.5, frameon=False, loc="upper left"); b.grid(color=S.GRID, lw=.6); S.clean(b)
    b.set_title("the magnitude-weighted statistics against DA", loc="left", fontsize=8.5)
    top = G._header(fig, "Does the choice of statistic change which cells count as reliable?",
                    f"Same cells as agreement_identity. Left: each tau-scaled statistic's cut is DA's cut through the "
                    f"identity, so a flip is the tie convention alone — τ_b corrects the denominator for ties, γ drops "
                    f"tied pairs, DA_drop_ref drops the pairs the reference ties (rq05's convention). Right: ρ and r weight a "
                    f"pair by how far it is displaced, which DA does not; a cell far from the diagonal is one where a few "
                    f"large swaps or many small ones tell different stories. Median {int(d['n_models'].median())} models per "
                    f"cell, so per-cell ρ and r are coarse; the pooled reading is the one to trust.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, out_dir: Path, d: pd.DataFrame, sens: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    ties = d["tied_both"].sum() + d["tied_one"].sum()
    t = (sens.pivot_table(index="statistic", columns="size", values="flip_share")
         .reindex(columns=size_order(sens["size"].unique())).mul(100).round(1).reset_index())
    body = "\n\n".join([
        "## Decision accuracy is Kendall's τ under another tie convention",
        f"Over the {len(d):,} (benchmark task, proxy size) cells of DA-size's population (every pair of the grid-seed "
        f"variants, ≥ {MIN_PAIRS} pairs, above chance at the proxy and at {TARGET_SIZE}): 2·DA − 1 = τ_a + (T_both − T_one)/n "
        f"exactly, where T_both / T_one are the pairs tied on both / one side. Ties are {ties:,} of {d['n_pairs'].sum():,} pairs "
        f"({ties / d['n_pairs'].sum():.1%}), {d['tied_one'].sum() / max(ties, 1):.0%} of them one-sided, and touch "
        f"{(d['tied_both'] + d['tied_one'] > 0).mean():.0%} of the cells — so the two statistics correlate at "
        f"r = {d['da'].corr(d['tau_b']):.3f} by construction, and the number with content is how often the tie "
        f"convention changes a reliability verdict (DA ≥ {CUT:g}, i.e. τ ≥ {2 * CUT - 1:.2f}), in % of cells per proxy size:",
        md_table(list(t.columns), t.values.tolist()),
        f"Spearman ρ and Pearson r on the raw scores are the two statistics that are NOT a rescaling — they weight a pair "
        f"by its displacement — and sit at r = {d['da'].corr(d['rho']):.3f} and {d['da'].corr(d['pearson_r']):.3f} against DA. "
        f"Median {int(d['n_models'].median())} models per cell. Values in `agreement_per_cell.csv`; regenerate with "
        f"`python analysis/rq02_decision_accuracy/agreement.py --pool {pool}`.",
        f"![DA against Kendall's tau]({stage}/{pool}/agreement_identity.png)",
        f"![Cut sensitivity]({stage}/{pool}/agreement_cut_sensitivity.png)"])
    replace_block(OUT_ROOT / "README.md", "agreement-measures", body, f"agreement.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose gate applies and whose folder receives the outputs")
    args = p.parse_args()
    out_dir = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    d = per_cell(finals(ladder_frame(POOL)), args.pool)
    d.to_csv(out_dir / "agreement_per_cell.csv", index=False)
    resid = ((2 * d["da"] - 1) - d["tau_a"] - d["gap_pairs"]).abs().max()
    print(f"{len(d)} cells; identity 2·DA − 1 = τ_a + (T_both − T_one)/n to {resid:.1e}; "
          f"r(DA, τ_b) = {d['da'].corr(d['tau_b']):.4f}, r(DA, ρ) = {d['da'].corr(d['rho']):.4f}")
    sens = cut_sensitivity(d)
    sens.to_csv(out_dir / "agreement_cut_sensitivity.csv", index=False)
    print(sens.pivot_table(index="statistic", columns="size", values="flip_share").round(3).to_string())
    d[["task", "size", "da", "tau_a", "gap_pairs", "tied_both", "tied_one", "n_pairs"]].to_csv(out_dir / "agreement_identity.csv", index=False)
    figure_identity(d, out_dir / "agreement_identity.png")
    figure_cuts(sens, d, out_dir / "agreement_cut_sensitivity.png")
    generate_readme(args.pool, out_dir, d, sens)
