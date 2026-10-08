"""The reliable-task figures with the tasks chosen out of sample.

Every `above_*` figure keeps the tasks whose decision accuracy cleared a cut
and then draws decision accuracy on them, so part of what it shows is the cut
(src/signal-and-noise/CLAUDE.md, bug history 17). Here the verdict and the
figure read different decisions: the pairs of design variants are split in two
halves (at random, stratified by the design axes a pair moves, so every axis
sits in both halves), `above_66_either` (`reliable_tasks.FILTERS`: the median
over a task's cells of DA-size or of DA-ckpt >= 0.66) is decided on one half,
the panels are read on the other, then the halves swap; SPLITS random splits,
each read both ways. Two in-sample readings are recomputed beside it by the
same code: on every pair (what the committed figures draw) and on the half
that is read (`in-sample (half)`: verdict and panels on the same half). A half
empties more cells under MIN_PAIRS than every pair does, so only the gap
between the cross-fitted line and the half in-sample one is the selection
effect alone; the figures draw that one dashed.

Two figures:

  rq2_da_all_above_66_either_crossfit_transformation_mono_axis
      the paper's rq2 figure (mono-axis pairs): DA-size by design axis, DA-ckpt
      and DA-goal by proxy size, each a mean over the reliable tasks of its
      cells, as `paper_rq2.py` draws them; cross-fitted lines solid with the
      5-95 % range over the splits, in sample on the same half dashed
  scale_convergence_da_size_above_66_either_crossfit_multi_axes_paper
      the reliable twin of the appendix figure `scale_convergence_da_size_multi_axes_paper`:
      pooled DA-size per proxy size over every task, over the reliable tasks
      chosen in sample on the same half and over the cross-fitted ones
      (benchmarks; BPB is never filtered)

A half holds about half of a cell's pairs, so rule 5 (MIN_PAIRS) empties more
cells than on every pair, and the pairs of the two halves share families (a
family's runs enter both), so the halves are not independent draws of models:
what the split removes is choosing the tasks on the very decisions that are
then scored. Rule 1 per DA kind (`reliable_tasks.GATE_REF`), rule 5 per cell.

    python analysis/rq02_decision_accuracy/crossfit_reliable.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
import warnings
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
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq02_decision_accuracy.compute_da import pair_agree, score_cube  # noqa: E402
from analysis.rq02_decision_accuracy.paper_rq2 import YLIM, _self_reference  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import FILTERS  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import AXIS_LABEL, OVERALL  # noqa: E402
from analysis.utils import (  # noqa: E402
    CKPT_DA_EARLY_FRACS, MIN_PAIRS, NON_EMB, SMALL_SIZES, TARGET_SIZE, design_axes, moved_axes, pair_sets,
    passes_gate)

SPLITS = 20
SEED = 1904
VARIANT = "above_66_either"
RED, CUT, CRIT = FILTERS[VARIANT]
SIZES = SMALL_SIZES + [TARGET_SIZE]
mpl.rcParams.update(S.RC)


class Cube:
    """The pool's score cube with its gates, and the per-task DA of any pair list."""

    def __init__(self, pool: str):
        df, self.tasks, fams, self.col, self.S, self.P = score_cube(pool)
        mask = load_mask(pool)
        self.fi = {f: i for i, f in enumerate(fams)}
        self.attrs = design_axes(df)
        meta = G.add_meta(pd.DataFrame({"task": self.tasks}))
        self.bench = np.isin(self.tasks, meta.loc[~meta["family"].isin(["bpb", "loss"]), "task"])
        self.bpb = np.isin(self.tasks, meta.loc[meta["family"] == "bpb", "task"])
        self.gate = {s: passes_gate(mask, self.tasks, s).to_numpy() for s in SIZES}         # rule 1, DA-ckpt
        self.gate_ref = {s: self.gate[s] & self.gate[TARGET_SIZE] for s in SIZES}          # DA-size, DA-goal

    def idx(self, pairs: list) -> tuple[np.ndarray, np.ndarray]:
        return np.array([self.fi[a] for a, _ in pairs], int), np.array([self.fi[b] for _, b in pairs], int)

    def cell(self, pairs: list, pc: tuple, rc: tuple, gate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(DA per task, NaN under rule 5 or the gate; comparable pairs per task)."""
        if not pairs:
            return np.full(len(self.tasks), np.nan), np.zeros(len(self.tasks), int)
        A, V = pair_agree(self.S, self.P, *self.idx(pairs), self.col[pc], self.col[rc])
        n = V.sum(1)
        ok = (n >= MIN_PAIRS) & gate
        return np.where(ok, A.sum(1) / np.maximum(n, 1), np.nan), np.where(ok, n, 0)


def verdict(c: Cube, pairs: list) -> np.ndarray:
    """`above_66_either` on these pairs: per benchmark task the median of its
    DA-size cells or of its DA-ckpt cells (every size, the reference's own run
    included, as `reliable_tasks.per_task` reduces them) clears the cut."""
    size = np.stack([c.cell(pairs, (s, 1.0), (TARGET_SIZE, 1.0), c.gate_ref[s])[0] for s in SMALL_SIZES], 1)
    ckpt = np.stack([c.cell(pairs, (b, f), (b, 1.0), c.gate[b])[0] for b in SIZES for f in CKPT_DA_EARLY_FRACS], 1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # an all-NaN task has no median
        med = {"size": np.nanmedian(size, 1), "ckpt": np.nanmedian(ckpt, 1)}
    keep = {"size": med["size"] >= CUT, "ckpt": med["ckpt"] >= CUT}
    keep |= {"both": keep["size"] & keep["ckpt"], "either": keep["size"] | keep["ckpt"]}
    return keep[CRIT] & c.bench


def rq2_lines(c: Cube, pairs: list, axis_of: dict, reliable: np.ndarray, fixed: bool = False) -> list[dict]:
    """The three panels of rq2 over the reliable tasks: per point, the mean of
    the tasks' cell DA and how many tasks it averages. `fixed` keeps, per line,
    only the tasks with a value at every point of it (`utils.fixed_population`'s
    population, the fixed-task twin)."""
    groups = {OVERALL: pairs} | {g: [p for p in pairs if axis_of[p] == g] for g in dict.fromkeys(axis_of[p] for p in pairs)}
    lines = [("DA-size", grp, [(s, c.cell(pl, (s, 1.0), (TARGET_SIZE, 1.0), c.gate_ref[s])[0]) for s in SMALL_SIZES])
             for grp, pl in groups.items()]
    for b in SIZES:
        lines.append(("DA-ckpt", b, [(f, c.cell(pairs, (b, f), (b, 1.0), c.gate[b])[0]) for f in CKPT_DA_EARLY_FRACS]))
        lines.append(("DA-goal", b, [(f, c.cell(pairs, (b, f), (TARGET_SIZE, 1.0), c.gate_ref[b])[0])
                                     for f in CKPT_DA_EARLY_FRACS + ([1.0] if b != TARGET_SIZE else [])]))
    rows = []
    for panel, line, pts in lines:
        keep = reliable & np.all([np.isfinite(d) for _, d in pts], 0) if fixed else reliable
        for x, d in pts:
            k = keep & np.isfinite(d)
            rows.append({"panel": panel, "line": line, "x": x, "da": d[k].mean() if k.any() else np.nan, "n_tasks": int(k.sum())})
    return rows


def pooled_size(c: Cube, pairs: list, tasks: np.ndarray) -> list[dict]:
    """Pooled DA-size (matching over comparable pairs) per proxy size over `tasks`."""
    out = []
    for s in SMALL_SIZES:
        da, n = c.cell(pairs, (s, 1.0), (TARGET_SIZE, 1.0), c.gate_ref[s])
        k = tasks & np.isfinite(da)
        out.append({"x": s, "da": (da[k] * n[k]).sum() / n[k].sum() if n[k].sum() else np.nan, "n_tasks": int(k.sum())})
    return out


def halves(pairs: list, stratum: dict, rng) -> tuple[list, list]:
    """Two halves of `pairs`, every stratum (the axes a pair moves) split between them."""
    a, b = [], []
    for st in dict.fromkeys(stratum[p] for p in pairs):
        group = [p for p in pairs if stratum[p] == st]
        group = [group[i] for i in rng.permutation(len(group))]
        a += group[::2]
        b += group[1::2]
    return a, b


def run(pool: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = Cube(pool)
    psets = pair_sets(c.attrs)
    moved = {p: tuple(moved_axes(c.attrs.loc[p[0]], c.attrs.loc[p[1]])) for p in psets["multi-axis"]}
    axis_of = {p: AXIS_LABEL[moved[p][0]] for p in psets["mono-axis"]}
    rng = np.random.default_rng(SEED)
    rq2, twin = [], []
    mono, multi = psets["mono-axis"], psets["multi-axis"]
    rq2 += [r | {"reading": "in-sample", "run": 0} for r in rq2_lines(c, mono, axis_of, verdict(c, mono))]
    twin += [r | {"reading": "every task", "run": 0} for r in pooled_size(c, multi, c.bench)]
    twin += [r | {"reading": "in-sample", "run": 0} for r in pooled_size(c, multi, verdict(c, multi))]
    for k in range(SPLITS):
        for pl, out in ((mono, rq2), (multi, twin)):
            h = halves(pl, moved, rng)
            for i, (sel, read) in enumerate((h, h[::-1])):
                run_id = 2 * k + i + 1
                lines = (lambda ps, rel: rq2_lines(c, ps, axis_of, rel)) if pl is mono else \
                    (lambda ps, rel: pooled_size(c, ps, rel))
                out += [r | {"reading": "cross-fitted", "run": run_id} for r in lines(read, verdict(c, sel))]
                out += [r | {"reading": "in-sample (half)", "run": run_id} for r in lines(read, verdict(c, read))]
    print(f"{SPLITS} splits x 2 directions; mono-axis pairs {len(mono)}, multi-axis {len(multi)}; "
          f"in-sample reliable tasks: mono {int(verdict(c, mono).sum())}, multi {int(verdict(c, multi).sum())}")
    bpb = pd.DataFrame(pooled_size(c, multi, c.bpb)).assign(reading="every task", population="bpb", run=0)
    return summarise(pd.DataFrame(rq2), ["panel", "line", "x"]), \
        pd.concat([summarise(pd.DataFrame(twin), ["x"]).assign(population="all benchmarks"),
                   bpb.assign(lo=np.nan, hi=np.nan)], ignore_index=True)


def summarise(t: pd.DataFrame, keys: list) -> pd.DataFrame:
    """One row per (reading, point): the value (the mean over the cross-fitted runs), its 5-95 % range, the tasks."""
    return (t.groupby(["reading"] + keys, sort=False)
            .agg(da=("da", "mean"), lo=("da", lambda v: v.quantile(.05)), hi=("da", lambda v: v.quantile(.95)),
                 n_tasks=("n_tasks", "mean"), runs=("run", "nunique")).reset_index())


def figure_rq2(t: pd.DataFrame, path: Path, solid: str = "cross-fitted", dashed: str = "in-sample (half)",
               dashed_label: str = "In sample, same half") -> None:
    fig, ax3 = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    d = t[t["panel"] == "DA-size"]
    rest = [g for g in dict.fromkeys(d["line"]) if g != OVERALL]
    colours = {OVERALL: S.INK} | dict(zip(rest, mpl.colormaps["Greens"](np.linspace(.95, .55, len(rest)))))
    x = {s: NON_EMB[s] for s in SIZES}
    for grp in [OVERALL] + rest:
        if d.loc[d["line"] == grp, "da"].isna().all():
            continue                                  # an axis too thin for any cell to keep MIN_PAIRS pairs
        for reading, ls, lw in ((solid, "-", 1.8), (dashed, (0, (3, 2)), 1.0)):
            g = d[(d["line"] == grp) & (d["reading"] == reading)].set_index("x").reindex(SMALL_SIZES)
            xs = [x[s] for s in SMALL_SIZES]
            ax3[0].plot(xs, g["da"], color=colours[grp], ls=ls, lw=lw, marker="o" if reading == solid else None,
                        ms=3.5, label=grp if reading == solid else None)
            if reading == solid and g["da"].notna().any():
                if "lo" in g:
                    ax3[0].fill_between(xs, g["lo"], g["hi"], color=colours[grp], alpha=.10, lw=0)
                _self_reference(ax3[0], xs[-1], g["da"].iloc[-1], x[TARGET_SIZE], colours[grp])
    ax3[0].set_xscale("log"); ax3[0].set_xticks([x[s] for s in SIZES]); ax3[0].set_xticklabels(SIZES); ax3[0].minorticks_off()
    ax3[0].set_xlabel("Model size (non-embedding parameters)")
    ax3[0].set_ylabel(f"DA-size (reference is the final checkpoint of {TARGET_SIZE})")
    ax3[0].plot([], [], color=S.MUTED, ls=(0, (3, 2)), lw=1.0, label=dashed_label)
    ax3[0].legend(fontsize=6.3, frameon=False, loc="upper left", ncol=2)
    for ax, panel, ylab in ((ax3[1], "DA-ckpt", "DA-ckpt (reference is the final checkpoint of the same size)"),
                            (ax3[2], "DA-goal", f"DA-goal (reference is the final checkpoint of {TARGET_SIZE})")):
        d = t[t["panel"] == panel]
        for s in SIZES:
            col = S.SIZE_COLOR.get(s, S.MUTED)
            for reading, ls, lw in ((solid, "-", 1.5), (dashed, (0, (3, 2)), .9)):
                g = d[(d["line"] == s) & (d["reading"] == reading)].sort_values("x")
                ax.plot(g["x"] * G.CHINCHILLA_AT_FULL, g["da"], color=col, ls=ls, lw=lw,
                        marker="o" if reading == solid else None, ms=3,
                        label=s if reading == solid and panel == "DA-ckpt" else None)
                if reading == solid and g["da"].notna().any() and g["x"].max() < 1:
                    _self_reference(ax, g["x"].iloc[-1] * G.CHINCHILLA_AT_FULL, g["da"].iloc[-1], G.CHINCHILLA_AT_FULL, col)
        ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels([G.chinchilla(f) for f in (.2, .4, .6, .8, 1.0)])
        ax.set_xlim(0.3, 5.2); ax.set_xlabel("Proxy's training tokens (× Chinchilla)"); ax.set_ylabel(ylab)
    ax3[1].legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2, title="Proxy size", title_fontsize=6.5)
    for ax in ax3:
        ax.set_ylim(*YLIM); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    fig.tight_layout()
    t.to_csv(path.with_suffix(".csv"), index=False)
    S.save_paper(fig, path)


def figure_twin(t: pd.DataFrame, path: Path) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    xs = [NON_EMB[s] for s in SMALL_SIZES]
    styles = {"every task": (S.INK, "-", "All tasks"), "in-sample (half)": (S.MUTED, (0, (3, 2)), "Reliable in sample"),
              "cross-fitted": (S.SERIES[0], "-", "Reliable, cross fitted")}
    for ax, (pop, lab) in zip(axs, (("all benchmarks", "Benchmarks"), ("bpb", "BPB"))):
        for reading, (col, ls, name) in styles.items():
            g = t[(t["population"] == pop) & (t["reading"] == reading)].set_index("x").reindex(SMALL_SIZES)
            if g["da"].notna().any():
                ax.plot(xs, g["da"], color=col, ls=ls, marker="o", ms=3.5, lw=1.5, label=name)
                if reading == "cross-fitted":
                    ax.fill_between(xs, g["lo"], g["hi"], color=col, alpha=.15, lw=0)
        ax.set_title(lab, loc="left", fontsize=9)
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels(SMALL_SIZES); ax.minorticks_off()
        ax.set_xlabel("Non-embedding parameters (log)"); ax.set_ylim(0, 1.02); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    axs[0].set_ylabel(f"Decision reliability vs {TARGET_SIZE} final")
    axs[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    t.to_csv(path.with_suffix(".csv"), index=False)
    S.save_paper(fig, path)


def generate_readme(pool: str, rq2: pd.DataFrame, twin: pd.DataFrame, out_dir: Path) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    size = rq2[(rq2["panel"] == "DA-size") & (rq2["line"] == OVERALL)]
    rows = [[r, *[f"{v.da:.3f} ({v.n_tasks:.0f})" for v in size[size["reading"] == r].set_index("x").reindex(SMALL_SIZES).itertuples()]]
            for r in ("in-sample", "in-sample (half)", "cross-fitted")]
    tw = twin[twin["population"] == "all benchmarks"]
    rows_t = [[r, *[f"{v.da:.3f} ({v.n_tasks:.0f})" for v in tw[tw["reading"] == r].set_index("x").reindex(SMALL_SIZES).itertuples()]]
              for r in ("every task", "in-sample", "in-sample (half)", "cross-fitted")]
    body = "\n\n".join([
        "### The reliable tasks chosen out of sample",
        f"`{VARIANT}` decided on one half of the pairs and read on the other ({SPLITS} random splits stratified by the "
        "design axes a pair moves, both directions; the value is the mean over the 40 readings, the band its 5-95 % range). "
        "`in-sample` is the same code with the verdict and the figure on every pair, i.e. the committed figures' "
        "numbers; `in-sample (half)` decides and reads on the same half, so it differs from the cross-fitted reading "
        "by the selection alone (the figures draw it dashed). "
        f"Regenerate with `python analysis/rq02_decision_accuracy/crossfit_reliable.py --pool {pool}`.",
        f"![rq2 cross-fitted]({stage}/{pool}/rq2_da_all_{VARIANT}_crossfit_transformation_mono_axis.png)",
        "DA-size over every mono-axis pair, mean over the reliable tasks (mean number of tasks):",
        md_table(["reading", *SMALL_SIZES], rows),
        f"![DA-size, reliable twin]({stage}/{pool}/scale_convergence_da_size_{VARIANT}_crossfit_multi_axes_paper.png)",
        "Pooled DA-size over the multi-axis pairs, benchmarks (tasks):",
        md_table(["reading", *SMALL_SIZES], rows_t)])
    replace_block(DECISION_ACCURACY / "README.md", "crossfit-reliable", body, f"crossfit_reliable.py --pool {pool}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL)
    args = ap.parse_args()
    out = DECISION_ACCURACY / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    rq2, twin = run(args.pool)
    print(rq2[rq2["panel"] == "DA-size"].round(3).to_string(index=False))
    print(twin.round(3).to_string(index=False))
    figure_rq2(rq2, out / f"rq2_da_all_{VARIANT}_crossfit_transformation_mono_axis")
    figure_twin(twin, out / f"scale_convergence_da_size_{VARIANT}_crossfit_multi_axes_paper")
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, rq2, twin, out)
