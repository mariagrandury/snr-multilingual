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
                                 the paper figure in two rows, split by how a task is
                                 scored (`utils.variant`): accuracy on top (originals and
                                 their `rf_`/`rfgm_` twins), the `bbpb_` twins below. A
                                 twin is read as bBPB -> 1.7B bBPB (its own bBPB at the
                                 reference and at its own final, as rq02's tables hold
                                 it), so it has no chance level and rule 1 passes it.
                                 Same filter, pairs, pool and gate as the paper figure,
                                 whose two populations these rows partition.

    rq2_da_all_by_benchmark_and_language.*
                                 per (benchmark, language) of the ORIGINAL accuracy tasks,
                                 one map per DA, each averaged over the proxies 90M-1B
                                 (DA-ckpt also over 1.7B, and over the tenths for DA-ckpt
                                 and DA-goal), with the row and column means and their
                                 95 % bootstrap bands (`language_figures`); unfiltered,
                                 and over the `above_66_either` tasks (`_above_66_either`).
    rq2_da_size_by_benchmark_and_language_per_proxy.*
                                 DA-size of the same cells, one map per proxy (both
                                 populations too).

Every name above carries the pair set's AXES_SUFFIX (rule 15), `rq2_da_all_multi_axes.*`
or `rq2_da_all_mono_axis.*` with --axes mono-axis, and reads the tables with the same one.

This module reads CSVs and draws them. It derives nothing, so a change to how a
panel's own figure is computed reaches rq2 the moment that script reruns — the
two can never disagree. Keep it that way: new logic belongs in the script that
owns the table, not here. The per-language maps average rq02's per-task table
over its cells and nothing else. The other exception is the `_by_scoring` split, which
the summary tables cannot give: it reduces by_L's per-task tables
(`da_all_pooled_per_task<axes>.csv`, `da_all_by_transformation_per_task_mono_axis.csv`)
with by_L's own `_summary`, the reduction behind the three tables above. Over
both scorings at once that reproduces the paper figure's CSV (to 1e-12, same
task counts), so the two rows are the paper figure's population, split.

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

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq02_decision_accuracy.by_L import READINGS, _summary  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import load_reliable  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import OVERALL  # noqa: E402
from analysis.utils import (AXES_SUFFIX, NON_EMB, SMALL_SIZES, TARGET_SIZE, bootstrap_band, one_axes,  # noqa: E402
                            size_order, variant)

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
# the `_by_scoring` split: the paper variant it splits, its reliability filter, and one row per scoring
SCORING_VARIANT, SCORING_FILTER = "above_66_either_transformation", "above_66_either"
SCORING_ROWS = {"acc": "Accuracy", "bbpb": "bBPB"}
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


def scoring_panels(out_dir: Path, pool: str, axes: str, scoring: str) -> tuple[pd.DataFrame, ...]:
    """The three panels of the paper variant over the tasks of one scoring, from
    by_L's per-task tables reduced by its `_summary` (gate, pair minimum, mean
    over tasks): DA-size per design axis and pooled (the 5C column of DA-goal),
    DA-ckpt and DA-goal over every pair of `axes`. As in scale_convergence, the
    per-axis lines keep the cells reliable on the mono-axis pairs and the pooled
    lines those reliable on `axes`'."""
    keep = set(load_reliable(out_dir, SCORING_FILTER, axes)["task"])
    mono = set(load_reliable(out_dir, SCORING_FILTER, "mono-axis")["task"])
    pooled = pd.read_csv(out_dir / f"da_all_pooled_per_task{AXES_SUFFIX[axes]}.csv")
    trans = pd.read_csv(out_dir / "da_all_by_transformation_per_task_mono_axis.csv")
    pooled, trans = (t[[variant(x)[1] == scoring for x in t["task"]]] for t in (pooled, trans))
    pooled = pooled[pooled["task"].isin(keep)]
    trans = trans[trans["task"].isin(mono) & (trans["frac"] == 1.0)]

    def summary(t, keys, name):
        d = _summary(pool, t, keys, READINGS[name])
        return d[d["group"] == "all benchmarks"].drop(columns="group")
    goal, ckpt = summary(pooled, [], "goal"), summary(pooled, [], "ckpt")
    size = pd.concat([goal[goal["frac"] == 1.0].assign(axis=OVERALL), summary(trans, ["axis"], "goal")])
    size = size.rename(columns={"axis": "group", "proxy_size": "size", "da": "reliability_macro", "tasks": "n_tasks"})
    size = size[size["size"] != TARGET_SIZE]
    # the reference is 1.0 by comparing its ranking with itself (drawn hollow), as in scale_convergence's table
    ref = pd.DataFrame({"group": size["group"].unique(), "size": TARGET_SIZE, "frac": 1.0, "reliability_macro": 1.0})
    size = pd.concat([size, ref], ignore_index=True).assign(non_emb=lambda d: d["size"].map(NON_EMB))
    return size, ckpt, goal


def scoring_figure(out_dir: Path, pool: str, axes: str = "multi-axis") -> None:
    """`rq2_da_all_<SCORING_VARIANT>_by_scoring<axes>`: the paper figure's three
    panels, one row per scoring (accuracy above, the bBPB twins below), on one y axis."""
    a = AXES_SUFFIX[axes]
    # the design axes in the order (and so the colours) of the paper figure's DA-size panel
    sz = pd.read_csv(out_dir / f"{RQ2_VARIANTS[SCORING_VARIANT][0]}{a}.csv")
    sz = sz[sz["population"] == "all benchmarks"].sort_values("non_emb")
    order = [g for g in dict.fromkeys(sz["group"]) if g != OVERALL]
    fig, grid = plt.subplots(len(SCORING_ROWS), 3, figsize=(13.5, 7.6), sharey=True, sharex="col")
    rows = []
    for i, (ax3, (scoring, label)) in enumerate(zip(grid, SCORING_ROWS.items())):
        size, ckpt, goal = scoring_panels(out_dir, pool, axes, scoring)
        rows += [d.assign(row=label) for d in (
            _scale_lines(ax3[0], size, order),
            _run_lines(ax3[1], ckpt, "ckpt", "DA-ckpt (reference is the final checkpoint of the same size)", i == 0),
            _run_lines(ax3[2], goal, "goal", f"DA-goal (reference is the final checkpoint of {TARGET_SIZE})", False))]
        ax3[0].set_title(label, loc="left", fontsize=9.5, fontweight="bold")
        if i:                             # the design-axis legend once, in the top row
            ax3[0].get_legend().remove()
    for ax in grid.ravel():
        ax.set_ylim(*YLIM); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    for ax in grid[0]:
        ax.set_xlabel("")
    fig.tight_layout()
    stem = f"rq2_da_all_{SCORING_VARIANT}_by_scoring{a}"
    pd.concat(rows, ignore_index=True).to_csv(out_dir / f"{stem}.csv", index=False)
    S.save_paper(fig, out_dir / stem)


# The per-language figures: each decision accuracy of the ORIGINAL accuracy tasks per (benchmark, language)
HEAT_FRACS = [k / 10 for k in range(1, 11)]          # the ten evaluated tenths (rule 3)
HEAT_KINDS = {  # DA kind -> (its (proxy, tenth, column) cells of da_all_per_task_both_axes, the reference the gate checks)
    "DA-size": ([(s, 1.0, f"decision_acc_size_{s}") for s in SMALL_SIZES], TARGET_SIZE),
    # DA-ckpt is defined at the reference too (its own run), so it reads 1.7B beside the proxies
    "DA-ckpt": ([(s, f, f"decision_acc_ckpt_f{round(f * 100)}_{s}") for s in SMALL_SIZES + [TARGET_SIZE]
                 for f in HEAT_FRACS[:-1]], None),
    "DA-goal": ([(s, f, f"decision_acc_goal_f{round(f * 100)}_{s}") for s in SMALL_SIZES for f in HEAT_FRACS], TARGET_SIZE),
}
MEAN = "Mean"


def language_cells(out_dir: Path, pool: str, axes: str) -> pd.DataFrame:
    """One row per (DA kind, task, proxy, tenth) of the original accuracy tasks
    (no rf_/rfgm_/bbpb_ twin) from rq02's per-task table, gated as every rq02
    reading (rule 1: the proxy, and 1.7B where it is the reference; a gated
    cell keeps `gated` and loses its value). No reliability filter: a heat map
    shows every task's value, and an above_66 cut would select on it."""
    d = one_axes(pd.read_csv(out_dir / "da_all_per_task_both_axes.csv"), axes)
    d = G.add_meta(d[[variant(t) == ("original", "acc") and not t.startswith("bpb_") and t != "train_loss"
                      for t in d["task"]]])
    out = []
    for kind, (cols, ref) in HEAT_KINDS.items():
        x = d.melt(id_vars=["task", "family", "language"], value_vars=[c for _, _, c in cols], var_name="col",
                   value_name="da")
        where = {c: (s_, f) for s_, f, c in cols}
        x["proxy_size"], x["frac"] = x["col"].map(lambda c: where[c][0]), x["col"].map(lambda c: where[c][1])
        out.append(G.mark_gated(x.drop(columns="col"), pool, "proxy_size", "da", ref).assign(kind=kind))
    return pd.concat(out, ignore_index=True)


def _cells(x: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Per `keys` + (benchmark, language): the mean over the benchmark's tasks
    in the language of each task's mean over its cells with a value, the task
    count, and `gated` where every cell was gated (grey) rather than empty."""
    t = x.groupby(keys + ["family", "language", "task"]).agg(da=("da", "mean"), gated=("gated", "all")).reset_index()
    c = t.groupby(keys + ["family", "language"]).agg(da=("da", "mean"), tasks=("task", "nunique"),
                                                     gated=("gated", "all")).reset_index()
    return c.assign(gated=c["gated"] & c["da"].isna())


def _heat(ax, val: pd.DataFrame, gat: pd.DataFrame, mean_col: bool, cuts: list) -> mpl.image.AxesImage:
    """A benchmark x language map: grey where gated, white where empty, the
    diverging scale centred at 0.5 (chance agreement), a line where a language
    list ends; `mean_col` appends the row means after a blank column."""
    from matplotlib.colors import ListedColormap, TwoSlopeNorm
    mat = val.copy()
    if mean_col:
        mat[" "], mat[MEAN] = np.nan, val.mean(axis=1)
    g = gat.reindex(index=mat.index, columns=mat.columns).fillna(False).astype(bool).to_numpy()
    ax.imshow(np.where(g, 1.0, 0.0), cmap=ListedColormap([S.SURFACE, GATED_GREY]), vmin=0, vmax=1, aspect="auto",
              interpolation="none")
    cmap = S.DIV.copy(); cmap.set_bad(alpha=0)
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy(dtype=float)), cmap=cmap, aspect="auto", interpolation="none",
                   norm=TwoSlopeNorm(vcenter=0.5, vmin=0.0, vmax=1.0))
    for x in cuts:
        ax.axvline(x - 0.5, color=S.INK, lw=.5)
    S.clean(ax, spines=())
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels([G.paper_name(r) for r in mat.index])
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=90)
    ax.tick_params(length=0, labelsize=6, labelbottom=False, labeltop=False)
    return im


GATED_GREY = mpl.colors.to_rgba(S.MUTED, .45)    # darker than the scale's centre, so a gated cell never reads as 0.5


def _key(fig, im, y: float) -> None:
    """The colour bar and the grey / white key above the maps (no figure text: rule 18)."""
    from matplotlib.patches import Patch
    cb = fig.colorbar(im, cax=fig.add_axes([0.30, y, 0.36, 0.010]), orientation="horizontal", ticks=[0, .25, .5, .75, 1])
    cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_visible(False)
    cb.set_label("Decision accuracy (0.5 is chance agreement)", fontsize=6.5); cb.ax.xaxis.set_label_position("top")
    fig.legend(handles=[Patch(color=GATED_GREY, label="Gated (at chance)"),
                        Patch(facecolor=S.SURFACE, edgecolor=S.GRID, label="No value")],
               loc="lower left", bbox_to_anchor=(0.70, y - 0.01), ncol=1, fontsize=6, frameon=False)


def language_figures(out_dir: Path, pool: str, axes: str = "multi-axis", filt: str | None = None) -> None:
    """Two benchmark x language figures of the original accuracy tasks
    (`language_cells`), the rows the benchmarks with at least two trained
    languages and a value somewhere, ordered by their mean DA-size; the columns
    the trained languages by resource rank, a line where a language list ends.
    Each CSV holds every cell, the benchmarks not drawn too (`drawn`). `filt`
    (a reliable_tasks FILTERS name) keeps only the tasks that pass it on the same
    pair set and adds `_<filt>` to the names; the tasks it drops are white.

    rq2_da_all_by_benchmark_and_language[_<filt>]<axes>   one map per DA kind, a cell the
        mean over the proxies 90M-1B (DA-size), over the proxies, 1.7B and the nine
        tenths before the final (DA-ckpt, defined at 1.7B too), over the proxies
        and the ten tenths (DA-goal), each task averaged before the tasks of a cell; beside and
        under each map the row and column means with a 95 % bootstrap band over
        the cells (`utils.bootstrap_band`).
    rq2_da_size_by_benchmark_and_language_per_proxy[_<filt>]<axes>   DA-size, one map per
        proxy 90M-1B, the row means in a last column; the benchmark with the most
        languages above 0.5 at every proxy is outlined (`highlight`, with its
        `steady_languages` of `complete_languages`)."""
    from analysis.rq02_decision_accuracy.cross_task import l_boundaries, resource_order
    x = language_cells(out_dir, pool, axes)
    if filt:                         # the reliability filter, reached on the same pair set (rule 15); dropped = white
        x = x[x["task"].isin(set(load_reliable(out_dir, filt, axes)["task"]))]
    a = (f"_{filt}" if filt else "") + AXES_SUFFIX[axes]
    mean = _cells(x, ["kind"])
    n_lang = mean.groupby("family")["language"].nunique()
    n_val = mean.groupby("family")["da"].count()
    drawn = set(n_lang.index[(n_lang >= 2) & (n_val.reindex(n_lang.index) > 0)])
    size = mean[(mean["kind"] == "DA-size") & mean["family"].isin(drawn)]
    rows = list(size.groupby("family")["da"].mean().sort_values(ascending=False, na_position="last").index)
    langs = resource_order(mean.loc[mean["family"].isin(drawn), "language"].unique())
    cuts = l_boundaries(langs)

    def grid(c: pd.DataFrame):
        return (c.pivot_table(index="family", columns="language", values="da").reindex(index=rows, columns=langs),
                c.pivot_table(index="family", columns="language", values="gated", aggfunc="all").reindex(index=rows, columns=langs))

    # 1. one map per DA kind, with the marginal means and their bands; heights in units of one map row
    unit, top, foot = 0.085, 0.42, 0.32                  # inches: a map row, the key and top labels, the bottom labels
    heights = [r for i in range(len(HEAT_KINDS)) for r in ((2.6 if i else 2.4), len(rows), 0.5, 4.5)]
    fig = plt.figure(figsize=(6.15, sum(heights) * unit + top + foot))      # 6.3 in once the labels are in
    gs = fig.add_gridspec(len(heights), 2, width_ratios=[1, 0.13], wspace=0.04, hspace=0, height_ratios=heights,
                          top=1 - top / fig.get_figheight(), bottom=foot / fig.get_figheight(), left=0.19, right=0.99)
    margins = []
    for i, kind in enumerate(HEAT_KINDS):
        val, gat = grid(mean[mean["kind"] == kind])
        ax = fig.add_subplot(gs[4 * i + 1, 0])
        im = _heat(ax, val, gat, False, cuts)
        ax.set_title(kind, loc="left", fontsize=7.5, pad=3)
        ax.tick_params(labeltop=i == 0)
        right, below = fig.add_subplot(gs[4 * i + 1, 1], sharey=ax), fig.add_subplot(gs[4 * i + 3, 0], sharex=ax)
        for side, axis, keys in ((right, 1, rows), (below, 0, langs)):
            m = val.apply(lambda v: pd.Series([v.mean(), *bootstrap_band(v.dropna())] if v.notna().any()
                                              else [np.nan] * 3, index=["mean", "lo", "hi"]), axis=axis)
            m = m if axis == 1 else m.T
            pos = np.arange(len(keys))
            err = [m["mean"] - m["lo"], m["hi"] - m["mean"]]
            S.clean(side)
            if side is right:
                side.errorbar(m["mean"], pos, xerr=err, fmt="o", ms=2, color=S.INK, elinewidth=.8, capsize=0)
                side.axvline(0.5, color=S.MUTED, lw=.6, ls=":"); side.set_xlim(0.2, 1.0); side.set_xticks([.5, 1])
                side.tick_params(labelleft=False, labelsize=6, length=2)
            else:
                side.errorbar(pos, m["mean"], yerr=err, fmt="o", ms=2, color=S.INK, elinewidth=.8, capsize=0)
                side.axhline(0.5, color=S.MUTED, lw=.6, ls=":"); side.set_ylim(0.0, 1.0); side.set_yticks([0, .5, 1])
                side.set_ylabel(MEAN, fontsize=6)
                side.tick_params(labelbottom=i == len(HEAT_KINDS) - 1, labelsize=6, length=2)
                for x_ in cuts:
                    side.axvline(x_ - 0.5, color=S.INK, lw=.5)
            margins.append(m.reset_index(names="key").assign(kind=kind, margin="benchmark" if side is right else "language"))
        if i == 0:
            right.set_title(f"{MEAN} (95% CI)", fontsize=6, pad=3)
    plt.setp(below.get_xticklabels(), rotation=90)
    _key(fig, im, 1 - 0.12 / fig.get_figheight())
    m = pd.concat(margins, ignore_index=True)
    bench, lang = m[m["margin"] == "benchmark"], m[m["margin"] == "language"]
    table = (mean.merge(bench.rename(columns={"key": "family", "mean": "benchmark_mean", "lo": "benchmark_lo",
                                              "hi": "benchmark_hi"}).drop(columns="margin"), how="left")
                 .merge(lang.rename(columns={"key": "language", "mean": "language_mean", "lo": "language_lo",
                                             "hi": "language_hi"}).drop(columns="margin"), how="left"))
    table.assign(benchmark=table["family"].map(G.paper_name), drawn=table["family"].isin(drawn), axes=axes,
                 filter=filt or "none",
                 n_languages=table["family"].map(n_lang)).to_csv(out_dir / f"rq2_da_all_by_benchmark_and_language{a}.csv",
                                                                 index=False)
    S.save_paper(fig, out_dir / f"rq2_da_all_by_benchmark_and_language{a}")

    # 2. DA-size, one map per proxy, over the benchmarks with a DA-size somewhere (the rest is grey at every proxy)
    per = _cells(x[x["kind"] == "DA-size"], ["proxy_size"])
    sized = [r for r in rows if per.loc[per["family"] == r, "da"].notna().any()]
    # the highlighted row: the benchmark with the most languages above 0.5 at every proxy (a value at all five)
    wide = per.pivot_table(index=["family", "language"], columns="proxy_size", values="da").reindex(columns=SMALL_SIZES)
    steady = wide.dropna().gt(0.5).all(axis=1).groupby(level="family").sum()
    best = steady.idxmax()
    heights = [h for i in range(len(SMALL_SIZES)) for h in ((2.4 if i == 0 else 1.3), len(sized))]
    fig = plt.figure(figsize=(6.05, sum(heights) * unit + top + foot))
    gs = fig.add_gridspec(len(heights), 1, hspace=0, height_ratios=heights,
                          top=1 - top / fig.get_figheight(), bottom=foot / fig.get_figheight(), left=0.19, right=0.99)
    for i, s_ in enumerate(SMALL_SIZES):
        ax = fig.add_subplot(gs[2 * i + 1, 0])
        val, gat = grid(per[per["proxy_size"] == s_])
        im = _heat(ax, val.reindex(sized), gat.reindex(sized), True, cuts)
        ax.set_ylabel(f"{s_} proxy", fontsize=7)
        y = sized.index(best)                    # the callout: an outline around the highlighted row
        ax.add_patch(mpl.patches.Rectangle((-0.5, y - 0.5), len(langs), 1, fill=False, ec=S.INK, lw=1.0, clip_on=False))
        ax.tick_params(labeltop=i == 0, labelbottom=i == len(SMALL_SIZES) - 1)
    _key(fig, im, 1 - 0.12 / fig.get_figheight())
    rmean = per[per["family"].isin(sized)].groupby(["proxy_size", "family"])["da"].mean().rename("benchmark_mean")
    per = per.merge(rmean.reset_index(), how="left")
    per.assign(benchmark=per["family"].map(G.paper_name), drawn=per["family"].isin(sized), axes=axes,
               filter=filt or "none", highlight=per["family"] == best,
               steady_languages=per["family"].map(steady), complete_languages=per["family"].map(
                   wide.dropna().groupby(level="family").size()),
               n_languages=per["family"].map(n_lang)).to_csv(
        out_dir / f"rq2_da_size_by_benchmark_and_language_per_proxy{a}.csv", index=False)
    S.save_paper(fig, out_dir / f"rq2_da_size_by_benchmark_and_language_per_proxy{a}")


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
    for filt in (None, SCORING_FILTER):
        language_figures(out, args.pool, args.axes, filt)
