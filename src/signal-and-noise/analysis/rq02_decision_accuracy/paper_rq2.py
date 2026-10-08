"""The paper's RQ2 figure: the three decision accuracies side by side.

One horizontal row, one panel per definition, drawn from the tables the three
rq02 scripts already write — this module only composes, it computes nothing,
so the panels can never disagree with the figures they are taken from:

    left    DA-size   `scale_convergence_da_size<axes>.csv`, the `all benchmarks` population:
                      how small a FULLY TRAINED model may be and still decide like
                      the reference. Drawn as `reliability_macro`, the mean over
                      tasks, which is how the other two panels average, so a
                      size's DA-size is exactly its 5C point in the DA-goal panel
                      (the table's pooled ratio `reliability` weights a task by its
                      pair count and is not that number). The PLAIN grouping — the single pooled line
                      over every pair at the grid seed, every data build (the
                      `predictivity` pool), which is the claim the
                      question is written as; `scale_convergence_da_size_transformation<axes>.csv`
                      breaks the same decisions down by design axis and is the
                      figure to read next to it, not inside it.
                      x = non-embedding parameters (log).
    middle  DA-ckpt   `early_small_da_ckpt_by_L<axes>.csv`, the `all pairs` panel: agreement
                      of an earlier checkpoint with the SAME size's final ranking,
                      one line per proxy size. x = Chinchilla multiples.
    right   DA-goal   `early_small_da_goal_by_L<axes>.csv`, the `all pairs` panel: agreement
                      with the reference's final ranking, same lines and x axis, so
                      the middle and right panels overlay directly — the distance
                      between them is what the proxy SIZE costs, on top of reading
                      the proxy early.

Paper conventions: no figure title, no note, no panel titles, no reference
lines. Each panel names its definition and its reference in its own y label,
the two line legends sit inside the axes (design axes on the left, in shades
of green beside the black pooled line, so they are not read as the blue
proxy sizes of the middle and right panels, whose legend is shared), and the y
axis is shared so the three panels are read against one scale. A point that is
1.0 by comparing a ranking with itself (the 1.7B final on the left, every
size's own final in the middle, the 1.7B final on the right) is drawn hollow,
joined by a dashed segment.

The three panels of one figure read ONE task population, so the 1.7B line of
the DA-ckpt panel is the 1.7B line of the DA-goal panel, and the pooled DA-size
line is the 5C points of the DA-goal panel; `rq2_da_all_above_66_own` is the one
exception, and says so below.

    rq2_da_all.png / .csv               the three panels over every task
    rq2_da_all_above_80.*               the same three panels over the cells reliable on
                                 BOTH axes at 0.80 (the `late` reduction)
    rq2_da_all_above_66_both.*          the same at 0.66 on the `median` reduction: one
                                 population, applied to all three panels
    rq2_da_all_above_66_both_transformation.*
                                 the same population as rq2_da_all_above_66_both, with the
                                 DA-size panel broken out by the design axis each pair
                                 differs on instead of pooled into one line: the middle
                                 and right panels are unchanged, the left one says WHICH
                                 design decisions a small fully trained model gets right
                                 rather than how many.
    rq2_da_all_above_66_own.*           each panel over the cells reliable on ITS OWN axis
                                 at 0.66: DA-size over the DA-size passers, DA-ckpt
                                 over the DA-ckpt passers, DA-goal over either. The
                                 populations differ BETWEEN panels here, so the
                                 panels are three separate claims rather than one
                                 population seen three ways — read it as "how well
                                 does each definition do on the tasks it is
                                 trustworthy for", not as a like-for-like comparison.
    rq2_da_all_above_66_either_transformation.*
                                 the paper's figure: the three panels over the cells
                                 reliable on EITHER axis at 0.66 (`median`), one
                                 population for all three, with the DA-size panel
                                 broken out by design axis as
                                 rq2_da_all_above_66_both_transformation does for the
                                 `both` population.
    rq2_da_all_above_66_own_transformation.*
                                 rq2_da_all_above_66_own with the DA-size panel broken
                                 out by design axis: DA-size over the DA-size passers,
                                 DA-ckpt over the DA-ckpt passers, DA-goal over either
                                 (the paper figure's populations before 2026-10-07).
                                 Three populations, so its 1.7B lines and its 5C points
                                 need not match across panels.

    rq2_da_all_above_66_either_transformation_by_scoring.*
                                 the paper figure in three rows, one per reading
                                 (rq11's keys, `READINGS` of bench_bpb_da.py):
                                   acc_acc    the accuracy tasks (originals and their
                                              `rf_`/`rfgm_` twins) -> their 1.7B accuracy
                                   bbpb_acc   the `bbpb_` twins -> their ORIGINAL's 1.7B
                                              accuracy, gated on that accuracy at the
                                              reference only (rq11's gate); rq11's
                                              `bbpb_to_acc` computes it, per design axis
                                              too. No DA-ckpt: its reference would be the
                                              proxy's own final on another score, so the
                                              panel is left empty with a note
                                   bbpb_bbpb  the twins -> their own 1.7B bBPB (no chance
                                              level, so rule 1 passes them)
                                 Same filter, pairs, pool and gate as the paper figure:
                                 rows 1 and 3 partition its tasks, row 2 is row 3's twins
                                 whose original is above chance at the reference.

Every name above carries the pair set's AXES_SUFFIX (rule 15), `rq2_da_all_multi_axes.*`
or `rq2_da_all_mono_axis.*` with --axes mono-axis, and reads the tables with the same one.

This module reads CSVs and draws them. It derives nothing, so a change to how a
panel's own figure is computed reaches rq2 the moment that script reruns — the
two can never disagree. Keep it that way: new logic belongs in the script that
owns the table, not here. The one exception is the `_by_scoring` split, which
the summary tables cannot give: it reduces by_L's per-task tables
(`da_all_pooled_per_task<axes>.csv`, `da_all_by_transformation_per_task_mono_axis.csv`)
with by_L's own `_summary`, the reduction behind the three tables above. Over
both scorings at once that reproduces the paper figure's CSV (to 1e-12, same
task counts), so rows 1 and 3 are the paper figure's population, split. No
rq02 table holds bbpb_acc, so row 2 loads the pool and runs rq11's
`bbpb_to_acc` (compute_da's early-small kernel) on it, then the same `_summary`.

Runs after `scale_convergence.py` and `by_L.py`, whose CSVs it reads.

    python analysis/rq02_decision_accuracy/paper_rq2.py --pool predictivity
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

from evals.scripts.utils.configs import load_pools, size_bucket  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq02_decision_accuracy.by_L import READINGS, _summary  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import load_reliable  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import OVERALL, pairs_by_group  # noqa: E402
from analysis.rq11_evaluation_recipe.recipe import bbpb_to_acc  # noqa: E402
from analysis.utils import (AXES_SUFFIX, BBPB, NON_EMB, PAIR_AXES, SMALL_SIZES, TARGET_SIZE,  # noqa: E402
                            build_snr_pool, design_axes, pair_sets, passes_gate, size_order, variant)

OUT_ROOT = DECISION_ACCURACY
YLIM = (0.25, 1.02)           # one scale for the three panels; 1.0 (the hollow self-references) inside it
Y_SIZE = "reliability_macro"  # DA-size as a mean over tasks, the other two panels' average
# rq2 variant -> the tables the three panels read: the DA-size panel's stem in
# full (so the GROUPING is data too, not only the filter), then the DA-ckpt and
# DA-goal suffixes. One entry per figure, so both the per-panel filter and the
# choice of scale-convergence grouping are data rather than a branch.
RQ2_VARIANTS = {
    "": ("scale_convergence_da_size", "", ""),
    "above_80": ("scale_convergence_da_size_above_80", "_above_80", "_above_80"),
    "above_66_both": ("scale_convergence_da_size_above_66_both", "_above_66_both", "_above_66_both"),
    "above_66_own": ("scale_convergence_da_size_above_66_size", "_above_66_ckpt", "_above_66_either"),
    "above_66_both_transformation": ("scale_convergence_da_size_transformation_above_66_both",
                                     "_above_66_both", "_above_66_both"),
    "above_66_either_transformation": ("scale_convergence_da_size_transformation_above_66_either",
                                       "_above_66_either", "_above_66_either"),
    "above_66_own_transformation": ("scale_convergence_da_size_transformation_above_66_size",
                                    "_above_66_ckpt", "_above_66_either"),
}
# the `_by_scoring` split: the paper variant it splits, its reliability filter, and one row per reading
# (<scoring at the proxy>_<score at the reference>, rq11's keys) with its row title
SCORING_VARIANT, SCORING_FILTER = "above_66_either_transformation", "above_66_either"
SCORING_ROWS = {"acc_acc": f"Accuracy → {TARGET_SIZE} accuracy", "bbpb_acc": f"bBPB → {TARGET_SIZE} accuracy",
                "bbpb_bbpb": f"bBPB → {TARGET_SIZE} bBPB"}
mpl.rcParams.update(S.RC)


def _ncol(n: int) -> int:
    """Legend columns of equal height (rule 18): two once a single column gets
    tall and the entries split evenly, one otherwise."""
    return 2 if n > 3 and n % 2 == 0 else 1


def _self_reference(ax, x0, y0, x1, c, z=3) -> None:
    """The dashed segment to a point that is 1.0 by comparing a ranking with
    itself, and that point, hollow: it closes the line without counting as one
    of its measurements."""
    ax.plot([x0, x1], [y0, 1.0], color=c, lw=1.0, ls=(0, (2, 2)), alpha=.45, zorder=2)
    ax.plot([x1], [1.0], marker="o", ms=4.5, mfc=S.SURFACE, mec=c, mew=1.2, ls="none", zorder=z)


def _scale_panel(ax, out_dir: Path, stem: str = "scale_convergence_da_size") -> pd.DataFrame:
    """DA-size: the mean over tasks vs non-embedding parameters. `stem` picks
    which scale-convergence table, and so whether this is the single pooled
    line or one line per design axis."""
    d = pd.read_csv(out_dir / f"{stem}.csv")
    return _scale_lines(ax, d[d["population"] == "all benchmarks"])


def _scale_lines(ax, d: pd.DataFrame, order: list | None = None) -> pd.DataFrame:
    """The DA-size lines of `d` (group, size, non_emb, reliability_macro).
    `order` fixes the design axes' order, and so their shades of green."""
    d = d.sort_values("non_emb")
    rest = order or [g for g in dict.fromkeys(d["group"]) if g != OVERALL]
    groups = [OVERALL] + rest           # OVERALL heads the legend and sits on top
    shades = mpl.colormaps["Greens"](np.linspace(.95, .55, len(rest))) if rest else []
    colours = dict(zip(rest, shades))
    for grp in groups:
        g = d[d["group"] == grp]
        real, ref = g[g["size"] != TARGET_SIZE], g[g["size"] == TARGET_SIZE]
        c, lw, z = (S.INK, 2.0, 6) if grp == OVERALL else (colours[grp], 1.4, 3)
        ax.plot(real["non_emb"], real[Y_SIZE], color=c, marker="o", ms=4, lw=lw, label=grp, zorder=z)
        if len(ref) and len(real):      # the reference is 1.0 by self-comparison
            _self_reference(ax, real["non_emb"].iloc[-1], real[Y_SIZE].iloc[-1], ref["non_emb"].iloc[0], c, z)
    ax.set_xscale("log")
    sizes = size_order(d["size"].unique())
    ax.set_xticks([NON_EMB[s] for s in sizes]); ax.set_xticklabels(sizes); ax.minorticks_off()
    ax.set_xlabel("Model size (non-embedding parameters)")
    ax.set_ylabel(f"DA-size (reference is the final checkpoint of {TARGET_SIZE})")
    if len(groups) > 1:               # the pooled grouping is one line; it needs no key
        ncol = _ncol(len(groups))     # one tall column sits in the empty top left, under the 1.0 line
        ax.legend(fontsize=6.5, frameon=False, loc="lower left" if ncol > 1 else "upper left", ncol=ncol)
    return d.assign(panel="DA-size")


def _run_panel(ax, out_dir: Path, name: str, ylabel: str, legend: bool, suffix: str = "") -> pd.DataFrame:
    """DA-ckpt / DA-goal: the `all pairs` panel of by_L, one line per proxy size.
    A line with no 5C point is ranked against its own final there (every size
    under DA-ckpt, the reference under DA-goal), which is 1.0 and drawn hollow."""
    d = pd.read_csv(out_dir / f"early_small_da_{name}_by_L{suffix}.csv")
    return _run_lines(ax, d[(d["L"].astype(str) == "all") & (d["group"] == "all benchmarks")], name, ylabel, legend)


def _run_lines(ax, d: pd.DataFrame, name: str, ylabel: str, legend: bool) -> pd.DataFrame:
    """The DA-ckpt / DA-goal lines of `d` (proxy_size, chinchilla, da), one per proxy size."""
    d = d.sort_values("chinchilla")
    sizes = [s for s in SMALL_SIZES + [TARGET_SIZE] if s in set(d["proxy_size"])]
    x_final = G.CHINCHILLA_AT_FULL
    for s_ in sizes:
        g = d[d["proxy_size"] == s_]
        c = S.SIZE_COLOR.get(s_, S.MUTED)
        ax.plot(g["chinchilla"], g["da"], color=c, marker="o", ms=3.5, lw=1.4, label=s_)
        if g["chinchilla"].max() < x_final:
            _self_reference(ax, g["chinchilla"].iloc[-1], g["da"].iloc[-1], x_final, c)
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels([G.chinchilla(f) for f in (.2, .4, .6, .8, 1.0)])
    ax.set_xlim(0.3, 5.2)
    ax.set_xlabel("Proxy's training tokens (× Chinchilla)")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=_ncol(len(sizes)),
                  title="Proxy size", title_fontsize=6.5)
    return d.assign(panel=f"DA-{name}")


def figure(out_dir: Path, variant: str = "", axes: str = "multi-axis") -> None:
    """`axes` picks the pair set (rule 15): every panel reads the table drawn
    over it, and the figure carries the same suffix, so `rq2_da_all_multi_axes.png` and
    `rq2_da_all_mono_axis.png` sit side by side over the same three definitions."""
    a = AXES_SUFFIX[axes]
    suffix = (f"_{variant}" if variant else "") + a
    sz, ck, gl = RQ2_VARIANTS[variant]
    need = [f"{sz}{a}.csv", f"early_small_da_ckpt_by_L{ck}{a}.csv",
            f"early_small_da_goal_by_L{gl}{a}.csv"]
    missing = [f for f in need if not (out_dir / f).is_file()]
    if missing:
        print(f"  (rq2_da_all{suffix}: missing {', '.join(missing)} — skipped)")
        return
    fig, ax3 = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    rows = [_scale_panel(ax3[0], out_dir, sz + a),
            _run_panel(ax3[1], out_dir, "ckpt", "DA-ckpt (reference is the final checkpoint of the same size)", True, ck + a),
            _run_panel(ax3[2], out_dir, "goal", f"DA-goal (reference is the final checkpoint of {TARGET_SIZE})", False, gl + a)]
    for ax in ax3:
        ax.set_ylim(*YLIM); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    fig.tight_layout()
    pd.concat(rows, ignore_index=True).to_csv(out_dir / f"rq2_da_all{suffix}.csv", index=False)
    S.save_paper(fig, out_dir / f"rq2_da_all{suffix}")       # rule 18: every rq2 figure is a bare paper figure


def bbpb_acc_tables(pool: str, twins: set, axes: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """by_L-shaped per-task tables (pooled over `axes`, per mono-axis design axis)
    of the bBPB twins `twins` read against their original's 1.7B ACCURACY: rq11's
    `bbpb_to_acc` on the pool, one call per pair set. DA-goal only (`da_ref`): a
    proxy cell is the twin's bBPB, the reference the original's accuracy."""
    df = build_snr_pool(pool)
    df["bucket"] = df["size"].map(size_bucket)
    attrs = design_axes(df)
    df = df[df["task"].isin(twins | {t[len(BBPB):] for t in twins})]
    none = {a: [] for a in PAIR_AXES}           # bbpb_to_acc runs every non-empty pair set it is given

    def run(key: str, pairs: list) -> pd.DataFrame:
        return bbpb_to_acc(df, none | {key: pairs}).drop(columns="axes").rename(
            columns={"da": "da_ref", "n_pairs": "n_pairs_ref"})
    groups = pairs_by_group(attrs, "transformation", "mono-axis")
    trans = pd.concat([run("mono-axis", pl).assign(axis=g) for g, pl in groups.items() if g != OVERALL], ignore_index=True)
    return run(axes, pair_sets(attrs)[axes]), trans


def scoring_panels(out_dir: Path, pool: str, axes: str, reading: str) -> tuple[pd.DataFrame, ...]:
    """The three panels of the paper variant over the tasks of one reading, from
    by_L's per-task tables reduced by its `_summary` (gate, pair minimum, mean
    over tasks): DA-size per design axis and pooled (the 5C column of DA-goal),
    DA-ckpt and DA-goal over every pair of `axes`. As in scale_convergence, the
    per-axis lines keep the cells reliable on the mono-axis pairs and the pooled
    lines those reliable on `axes`'. `bbpb_acc` has no DA-ckpt (None): its
    reference would be the proxy's own final on another score, so its 5C point
    is not a ranking compared with itself."""
    keep = set(load_reliable(out_dir, SCORING_FILTER, axes)["task"])
    mono = set(load_reliable(out_dir, SCORING_FILTER, "mono-axis")["task"])
    if reading == "bbpb_acc":
        # the twins of the bBPB row whose original is above chance at the reference (rq11's gate for this reading)
        twins = sorted(t for t in keep | mono if t.startswith(BBPB))
        ok = passes_gate(load_mask(pool), [t[len(BBPB):] for t in twins], TARGET_SIZE)
        pooled, trans = bbpb_acc_tables(pool, {t for t in twins if ok[t[len(BBPB):]]}, axes)
    else:
        pooled = pd.read_csv(out_dir / f"da_all_pooled_per_task{AXES_SUFFIX[axes]}.csv")
        trans = pd.read_csv(out_dir / "da_all_by_transformation_per_task_mono_axis.csv")
        scoring = reading.split("_")[0]
        pooled, trans = (t[[variant(x)[1] == scoring for x in t["task"]]] for t in (pooled, trans))
    pooled = pooled[pooled["task"].isin(keep)]
    trans = trans[trans["task"].isin(mono) & (trans["frac"] == 1.0)]

    def summary(t, keys, name):
        d = _summary(pool, t, keys, READINGS[name])
        return d[d["group"] == "all benchmarks"].drop(columns="group")
    goal = summary(pooled, [], "goal")
    ckpt = None if reading == "bbpb_acc" else summary(pooled, [], "ckpt")
    size = pd.concat([goal[goal["frac"] == 1.0].assign(axis=OVERALL), summary(trans, ["axis"], "goal")])
    size = size.rename(columns={"axis": "group", "proxy_size": "size", "da": "reliability_macro", "tasks": "n_tasks"})
    size = size[size["size"] != TARGET_SIZE]
    # the reference is 1.0 by comparing its ranking with itself (drawn hollow), as in scale_convergence's table
    ref = pd.DataFrame({"group": size["group"].unique(), "size": TARGET_SIZE, "frac": 1.0, "reliability_macro": 1.0})
    size = pd.concat([size, ref], ignore_index=True).assign(non_emb=lambda d: d["size"].map(NON_EMB))
    return size, ckpt, goal


def _no_ckpt(ax, ylabel: str) -> None:
    """The DA-ckpt panel of the bBPB -> accuracy row: empty, with the reason."""
    ax.text(0.5, 0.5, "Not defined for this reading.\nIts reference would be the proxy's own\n"
            "final accuracy, another score than bBPB,\nso it is not a checkpoint comparison.",
            transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color=S.MUTED)
    ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels([G.chinchilla(f) for f in (.2, .4, .6, .8, 1.0)])
    ax.set_xlim(0.3, 5.2)
    ax.set_xlabel("Proxy's training tokens (× Chinchilla)")
    ax.set_ylabel(ylabel)


def scoring_figure(out_dir: Path, pool: str, axes: str = "multi-axis") -> None:
    """`rq2_da_all_<SCORING_VARIANT>_by_scoring<axes>`: the paper figure's three
    panels, one row per reading (accuracy, bBPB against the 1.7B accuracy, bBPB
    against the 1.7B bBPB), on one y axis."""
    a = AXES_SUFFIX[axes]
    # the design axes in the order (and so the colours) of the paper figure's DA-size panel
    sz = pd.read_csv(out_dir / f"{RQ2_VARIANTS[SCORING_VARIANT][0]}{a}.csv")
    sz = sz[sz["population"] == "all benchmarks"].sort_values("non_emb")
    order = [g for g in dict.fromkeys(sz["group"]) if g != OVERALL]
    fig, grid = plt.subplots(len(SCORING_ROWS), 3, figsize=(13.5, 11.2), sharey=True, sharex="col")
    rows = []
    ck_label = "DA-ckpt (reference is the final checkpoint of the same size)"
    for i, (ax3, (reading, label)) in enumerate(zip(grid, SCORING_ROWS.items())):
        size, ckpt, goal = scoring_panels(out_dir, pool, axes, reading)
        drawn = [_scale_lines(ax3[0], size, order),
                 _run_lines(ax3[2], goal, "goal", f"DA-goal (reference is the final checkpoint of {TARGET_SIZE})", False)]
        if ckpt is None:
            _no_ckpt(ax3[1], ck_label)
        else:
            drawn.append(_run_lines(ax3[1], ckpt, "ckpt", ck_label, i == 0))
        rows += [d.assign(row=label) for d in drawn]
        ax3[0].set_title(label, loc="left", fontsize=9.5, fontweight="bold")
        if i:                             # the design-axis legend once, in the top row
            ax3[0].get_legend().remove()
    for ax in grid.ravel():
        ax.set_ylim(*YLIM); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    for ax in grid[:-1].ravel():
        ax.set_xlabel("")
    fig.tight_layout()
    stem = f"rq2_da_all_{SCORING_VARIANT}_by_scoring{a}"
    pd.concat(rows, ignore_index=True).to_csv(out_dir / f"{stem}.csv", index=False)
    S.save_paper(fig, out_dir / stem)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    p.add_argument("--axes", default="multi-axis", choices=["multi-axis", "mono-axis"],
                   help="the pair set (rule 15); mono-axis writes the `_mono_axis` twins")
    args = p.parse_args()
    out = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    for v in RQ2_VARIANTS:
        figure(out, v, args.axes)
    scoring_figure(out, args.pool, args.axes)
