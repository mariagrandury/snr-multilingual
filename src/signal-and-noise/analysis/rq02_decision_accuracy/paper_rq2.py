"""The paper's RQ2 figure: the three decision accuracies side by side.

One horizontal row, one panel per definition, drawn from the tables the three
rq02 scripts already write — this module only composes, it computes nothing,
so the panels can never disagree with the figures they are taken from:

    left    DA-size   `scale_convergence.csv`, the `all benchmarks` population:
                      how small a FULLY TRAINED model may be and still decide like
                      the reference. The PLAIN grouping — the single pooled line
                      over every pair at the grid seed, every data scheme (NOT the
                      A/B-only `predictivity` headline pool), which is the claim the
                      question is written as; `scale_convergence_transformation.csv`
                      breaks the same decisions down by design axis and is the
                      figure to read next to it, not inside it.
                      x = non-embedding parameters (log).
    middle  DA-ckpt   `early_small_by_L_ckpt.csv`, the `all pairs` panel: agreement
                      of an earlier checkpoint with the SAME size's final ranking,
                      one line per proxy size. x = Chinchilla multiples.
    right   DA-goal   `early_small_by_L_goal.csv`, the `all pairs` panel: agreement
                      with the reference's final ranking, same lines and x axis, so
                      the middle and right panels overlay directly — the distance
                      between them is what the proxy SIZE costs, on top of reading
                      the proxy early.

Paper conventions: no figure title, no note, no panel titles. Each panel names
its definition in its own y label, the two line legends sit inside the axes
(design axes on the left, proxy sizes in the middle, shared with the right),
and the y axis is shared so the three panels are read against one scale.

    rq2.png / .svg / .csv        the figure the paper embeds (the SVG is the vector
                                 copy the paper build prefers)
    rq2_above_80.*               the same three panels over the cells reliable on
                                 BOTH axes at 0.80 (the `late` reduction)
    rq2_above_66_both.*          the same at 0.66 on the `median` reduction: one
                                 population, applied to all three panels
    rq2_above_66_both_transformation.*
                                 the same population as rq2_above_66_both, with the
                                 DA-size panel broken out by the design axis each pair
                                 differs on instead of pooled into one line: the middle
                                 and right panels are unchanged, the left one says WHICH
                                 design decisions a small fully trained model gets right
                                 rather than how many.
    rq2_above_66_one.*           each panel over the cells reliable on ITS OWN axis
                                 at 0.66: DA-size over the DA-size passers, DA-ckpt
                                 over the DA-ckpt passers, DA-goal over either. The
                                 populations differ BETWEEN panels here, so the
                                 panels are three separate claims rather than one
                                 population seen three ways — read it as "how well
                                 does each definition do on the tasks it is
                                 trustworthy for", not as a like-for-like comparison.

This module reads CSVs and draws them. It derives nothing, so a change to how a
panel's own figure is computed reaches rq2 the moment that script reruns — the
two can never disagree. Keep it that way: new logic belongs in the script that
owns the table, not here.

Runs after `scale_convergence.py` and `by_L.py`, whose CSVs it reads.

    python analysis/rq02_decision_accuracy/paper_rq2.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
from analysis.autodoc import CANONICAL_POOL  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq02_decision_accuracy.early_small import SAFE_DA  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import GROUP_COLOURS, OVERALL, TAU  # noqa: E402
from analysis.utils import NON_EMB, SMALL_SIZES, TARGET_SIZE, size_order  # noqa: E402

OUT_ROOT = DECISION_ACCURACY
YLIM = (0.25, 1.0)            # one scale for the three panels
# rq2 variant -> the tables the three panels read: the DA-size panel's stem in
# full (so the GROUPING is data too, not only the filter), then the DA-ckpt and
# DA-goal suffixes. One entry per figure, so both the per-panel filter and the
# choice of scale-convergence grouping are data rather than a branch.
RQ2_VARIANTS = {
    "": ("scale_convergence", "", ""),
    "above_80": ("scale_convergence_above_80", "_above_80", "_above_80"),
    "above_66_both": ("scale_convergence_above_66_both", "_above_66_both", "_above_66_both"),
    "above_66_one": ("scale_convergence_above_66_size", "_above_66_ckpt", "_above_66_either"),
    "above_66_both_transformation": ("scale_convergence_transformation_above_66_both",
                                     "_above_66_both", "_above_66_both"),
}
mpl.rcParams.update(S.RC)


def _scale_panel(ax, out_dir: Path, stem: str = "scale_convergence") -> pd.DataFrame:
    """DA-size: reliability vs non-embedding parameters. `stem` picks which
    scale-convergence table, and so whether this is the single pooled line or
    one line per design axis."""
    d = pd.read_csv(out_dir / f"{stem}.csv")
    d = d[d["population"] == "all benchmarks"].sort_values("non_emb")
    rest = [g for g in dict.fromkeys(d["group"]) if g != OVERALL]
    groups = [OVERALL] + rest           # OVERALL heads the legend and sits on top
    colours = dict(zip(rest, GROUP_COLOURS * 3))
    for grp in groups:
        g = d[d["group"] == grp]
        real, ref = g[g["size"] != TARGET_SIZE], g[g["size"] == TARGET_SIZE]
        c, lw, z = (S.INK, 2.0, 6) if grp == OVERALL else (colours[grp], 1.4, 3)
        ax.plot(real["non_emb"], real["reliability"], color=c, marker="o", ms=4, lw=lw, label=grp, zorder=z)
        if len(ref) and len(real):      # the reference is 1.0 by self-comparison: faint, hollow
            ax.plot([real["non_emb"].iloc[-1], ref["non_emb"].iloc[0]],
                    [real["reliability"].iloc[-1], ref["reliability"].iloc[0]],
                    color=c, lw=1.0, ls=(0, (2, 2)), alpha=.45, zorder=2)
            ax.plot(ref["non_emb"], ref["reliability"], marker="o", ms=4.5, mfc=S.SURFACE, mec=c, mew=1.2, ls="none", zorder=z)
    ax.axhline(TAU, color=S.MUTED, lw=.8, ls=":")
    ax.set_xscale("log")
    sizes = size_order(d["size"].unique())
    ax.set_xticks([NON_EMB[s] for s in sizes]); ax.set_xticklabels(sizes)
    ax.set_xlabel("model size (non-embedding parameters)")
    ax.set_ylabel(f"DA-size — vs {TARGET_SIZE} final, fully trained")
    if len(groups) > 1:               # the pooled grouping is one line; it needs no key
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")
    return d.assign(panel="DA-size")


def _run_panel(ax, out_dir: Path, name: str, ylabel: str, legend: bool, suffix: str = "") -> pd.DataFrame:
    """DA-ckpt / DA-goal: the `all pairs` panel of by_L, one line per proxy size."""
    d = pd.read_csv(out_dir / f"early_small_by_L_{name}{suffix}.csv")
    d = d[(d["L"].astype(str) == "all") & (d["group"] == "all benchmarks")].sort_values("chinchilla")
    sizes = [s for s in SMALL_SIZES + [TARGET_SIZE] if s in set(d["proxy_size"])]
    for s_ in sizes:
        g = d[d["proxy_size"] == s_]
        ax.plot(g["chinchilla"], g["da"], color=S.SIZE_COLOR.get(s_, S.MUTED), marker="o", ms=3.5, lw=1.4, label=s_)
    ax.axhline(SAFE_DA, color=S.MUTED, lw=.8, ls=":")
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels([G.chinchilla(f) for f in (.2, .4, .6, .8, 1.0)])
    ax.set_xlim(0.3, 5.2)
    ax.set_xlabel("proxy's training tokens (× Chinchilla)")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2, title="proxy size", title_fontsize=6.5)
    return d.assign(panel=ylabel.split(" —")[0])


def figure(out_dir: Path, variant: str = "") -> None:
    suffix = f"_{variant}" if variant else ""
    sz, ck, gl = RQ2_VARIANTS[variant]
    need = [f"{sz}.csv", f"early_small_by_L_ckpt{ck}.csv",
            f"early_small_by_L_goal{gl}.csv"]
    missing = [f for f in need if not (out_dir / f).is_file()]
    if missing:
        print(f"  (rq2{suffix}: missing {', '.join(missing)} — skipped)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    rows = [_scale_panel(axes[0], out_dir, sz),
            _run_panel(axes[1], out_dir, "ckpt", "DA-ckpt — vs its own size's final", True, ck),
            _run_panel(axes[2], out_dir, "goal", f"DA-goal — vs {TARGET_SIZE} final", False, gl)]
    for ax in axes:
        ax.set_ylim(*YLIM); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    fig.tight_layout()
    pd.concat(rows, ignore_index=True).to_csv(out_dir / f"rq2{suffix}.csv", index=False)
    fig.savefig(out_dir / f"rq2{suffix}.svg", bbox_inches="tight", facecolor=S.SURFACE)
    S.save(fig, out_dir / f"rq2{suffix}.png", dpi=200)          # closes the figure


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    args = p.parse_args()
    out = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    for variant in RQ2_VARIANTS:
        figure(out, variant)
