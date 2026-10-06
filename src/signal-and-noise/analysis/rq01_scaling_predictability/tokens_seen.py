"""Tokens seen: is a language's evaluation a function of how much of that
language the proxy has trained on?

The two figure families live in the folders of the questions they answer:
B (above chance) in `rq00_chance_vs_train_tokens/`, A (decision accuracy) in
`rq02_da_vs_train_tokens/`, each `<stage>/<pool>/` with its README block there.

Both figures share one x axis: the training tokens of ONE language a checkpoint
has seen — the language's share of the mixture (L, data build), from the build's own
plan (`utils.language_token_share`), times the cell's budget D(N), times the
checkpoint's share of the run. The share is a relabelling of the (L, size,
checkpoint) grid, not a new measurement, and the plan files live on capstor: a
cell whose share is unreachable carries NaN tokens, is left out of the figure
and is counted in the `!!!` line and the README.

A) da_goal_multi_axes_across_langs_bpb.png/.pdf/.csv, _cells.csv  (and `_mono_axis_` twins of goal and ckpt)
   DA-goal of a language's BPB against the tokens of that language the proxies
   had seen. DA-goal = the share of pairs of design variants the proxy
   checkpoint orders like the reference's final (rq02's kernel; rule 15's
   multi-axis set: every pair at the grid seed of every data build, the pair set of
   rq02's pooled panel; the `da_{goal,ckpt}_mono_axis_` twins read the mono-axis
   pairs, those that move exactly one design axis). A BPB task is read only on the variants that train its
   language (rule 2, the loader), so the variants behind one cell differ in L,
   scheme and temperature and saw different amounts of the language: a cell's x is
   the checkpoint's share of the run times the MEAN over the pair set's proxy
   variants of their tokens of the language (the cells table keeps the min and
   max). A cell needs MIN_PAIRS pairs (rule 5). Per (proxy size, tenth of the
   run): the mean DA over languages, its standard error over languages and the
   geometric mean of their x — one line per proxy size, the reference's own line
   being its early checkpoints against its final. Two more references, the same
   cells otherwise: da_ckpt_* ranks a proxy checkpoint against the PROXY SIZE's
   own final (rq02's DA-ckpt: is the run's early ranking its own final ranking?)
   and da_size_* the FINAL of every size against the reference's (DA-size, one
   point per size — the 100 % end of the DA-goal lines). The `_vs_frac` twins of
   the goal and ckpt figures put the same points against the share of the run
   the checkpoint sits at instead of the tokens: if the lines that are apart on
   the token axis fall together there, it is the schedule, not the exposure,
   that decides when a language's BPB ranks the variants.

B) pass_prob_vs_train_tokens_by_benchmark_<population>.png/.pdf/.csv, _points.csv
   Per benchmark and size: the share of its (language, L) cells that are above
   chance, against the tokens of the language the cell trained on, one line per
   size, the cells binned on log10 tokens (BINS_PER_DECADE per decade, the cell
   count on every point). The `.csv` is the cell table, `_points.csv` the binned
   values drawn. Two populations of runs behind a (task, size, L) cell:
     _deep_A_1904   the plan grid (deep, data A, seed 1904): one run per cell,
                    above chance = its one-sided 95 % Wilson lower bound clears
                    chance (rule 1's per-run test, `above_random.above_chance`)
     _1904          every seed-1904 run at the (size, L) that trains the language
                    (every data build and ladder): above chance when at least
                    MIN_SHARE of them are (rule 1's cell rule); the score and the
                    tokens are their means
   Only benchmarks with a chance level appear (BPB and the generative tasks
   cannot be gated). A `_ckpts` twin of each population reads the cells at
   every evaluated tenth of the run instead of the final alone: a cell is
   (task, size, L, tenth), above chance by the same rule on that checkpoint's
   scores, and its x the tokens of the language seen BY THAT CHECKPOINT (the
   tenth × the run's tokens) — ten times the cells and a token axis that runs
   through every training run, so the bins fill where the final-only version
   has one cell per (language, L, size). Every cell table is drawn three ways:
   `_by_benchmark_` (a panel per benchmark, the cells of all its languages),
   `_by_language_` (a panel per language, the cells of all its benchmarks,
   panels in order of the language's tokens) and `_all_` (one panel, every cell);
   the cell table is the `_by_benchmark_` stem's `.csv`, the two others' `.csv`
   hold the binned points they draw.
   Two paper copies of the `_by_benchmark_1904_ckpts` figure, PNG and SVG, no
   header, the language count alone in a panel's title: `_paper` (every
   benchmark) and `_include_rf_paper_vertical` (INCLUDE over its `rf_` twin,
   the two panels stacked in the height of one), and `da_goal_..._paper`, the
   DA-goal figure without its header. Every paper copy goes through
   `style.save_paper` (rule 18).

    python analysis/rq01_scaling_predictability/tokens_seen.py --pool predictivity_seeds
    python analysis/rq01_scaling_predictability/tokens_seen.py --paper     # the paper copies alone, from the tables on disk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools, size_bucket  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import CHANCE_VS_TRAIN_TOKENS, DA_VS_TRAIN_TOKENS  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import MIN_SHARE, above_chance  # noqa: E402
from analysis.rq02_decision_accuracy.compute_da import _scores_at, compute_early_small_decision_accuracy  # noqa: E402
from analysis.rq02_decision_accuracy.early_small import SAFE_DA  # noqa: E402
from analysis.utils import (  # noqa: E402
    AXES_SUFFIX, GRID_SEED, MIN_PAIRS, SHARED_FRACS, TARGET_SIZE, assign_language, at_fraction, benchmark_family,
    design_axes, finals, ladder_frame, language_tokens, languages_only, pair_sets, size_order)

CANONICAL = "predictivity_seeds"
BINS_PER_DECADE = 3
DA_STEM = "da_{mode}{axes}_across_langs_bpb"
DA_NAME = DA_STEM.format(mode="goal", axes=AXES_SUFFIX["multi-axis"])
# the pair sets each DA mode is drawn over (rule 15): mono-axis twins of the two checkpoint readings
DA_AXES = {"goal": ("multi-axis", "mono-axis"), "ckpt": ("multi-axis", "mono-axis"), "size": ("multi-axis",)}
# mode -> (whose final a proxy checkpoint is ranked against, the fractions of the run it is read at)
DA_MODES = {"goal": (f"the {TARGET_SIZE} final", SHARED_FRACS),
            "ckpt": ("the proxy size's own final", SHARED_FRACS),
            "size": (f"the {TARGET_SIZE} final, the proxy at its own final", (1.0,))}
PASS_NAME = "pass_prob_vs_train_tokens"
PANELS = ("benchmark", "language", "all")         # what a panel of the pass figure holds
# population -> (the runs of a (task, size, L) cell, how the cell is called above chance)
POPULATIONS = {
    "deep_A_1904": ("the plan grid, deep / data A / seed 1904: one run per cell",
                    "the run's one-sided 95 % Wilson lower bound clears chance"),
    "1904": (f"every seed-{GRID_SEED} run at the (size, L) that trains the language, every data build and ladder",
             f"at least {MIN_SHARE:.0%} of the cell's runs clear chance (Wilson lower bound); score and tokens are their means"),
}
mpl.rcParams.update(S.RC)


def _tokens(L, data, size, ladder, lang) -> float:
    """Tokens of `lang` a full run of the cell trains on; NaN when the build's
    plan is unreachable. A trained language is always in the share, so a
    missing key is a bug and raises."""
    t = language_tokens(int(L), data, size, ladder)
    return np.nan if t is None else t[lang]


# --- A: DA-goal of a language's BPB against the tokens of the language seen -------------

def da_cells(df: pd.DataFrame, mode: str = "goal", pairs: list | None = None) -> pd.DataFrame:
    """One row per (BPB task, proxy size, tenth) with >= MIN_PAIRS pairs: the DA
    of the proxy checkpoint against the reference of `mode` (DA_MODES) over
    every design pair (`pairs`: only these, the mono-axis set), and the tokens of the task's language the pair set's
    proxies had seen at that checkpoint (mean, min and max over the variants)."""
    bpb = df[df["kind"] == "bpb"].assign(language=lambda d: d["task"].map(assign_language))
    bpb = languages_only(bpb)
    attrs = bpb[["family", "L", "data", "ladder"]].drop_duplicates().set_index("family")
    fracs = DA_MODES[mode][1]
    rows = []
    for task, dft in bpb.groupby("task", sort=False):
        lang = dft["language"].iloc[0]
        # ckpt: every size is its own reference; the kernel also ranks the smaller sizes against it, which is not DA-ckpt
        targets = size_order(dft["bucket"].dropna().unique()) if mode == "ckpt" else [TARGET_SIZE]
        for target in targets:
            ref = set(_scores_at(dft, target, 1.0))
            for c in compute_early_small_decision_accuracy(dft, target_size=target, fracs=fracs, pairs=pairs):
                if mode == "ckpt" and c["proxy_size"] != target:
                    continue
                fams = sorted(set(_scores_at(dft, c["proxy_size"], c["frac"])) & ref)
                tok = np.array([c["frac"] * _tokens(attrs.at[f, "L"], attrs.at[f, "data"], c["proxy_size"], attrs.at[f, "ladder"], lang)
                                for f in fams])
                rows.append({"task": task, "language": lang, "proxy_size": c["proxy_size"], "frac": c["frac"], "da": c["da"],
                             "n_pairs": c["n_pairs"], "n_variants": len(fams), "tokens": tok.mean(),
                             "tokens_min": tok.min(), "tokens_max": tok.max()})
    return pd.DataFrame(rows)


def da_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """Per (proxy size, tenth): mean DA over the languages with a token count,
    its standard error over languages, the geometric mean of their tokens."""
    ok = cells.dropna(subset=["tokens"])
    return (ok.groupby(["proxy_size", "frac"])
            .agg(da=("da", "mean"), da_se=("da", lambda v: v.std(ddof=1) / np.sqrt(len(v))),
                 n_languages=("task", "nunique"), median_pairs=("n_pairs", "median"),
                 tokens=("tokens", lambda v: 10 ** np.log10(v).mean()))
            .reset_index())


def plot_da(summary: pd.DataFrame, cells: pd.DataFrame, out_dir: Path, mode: str = "goal", x: str = "tokens",
            paper: bool = False, axes: str = "multi-axis") -> None:
    """`x` = "tokens" (the language's tokens the proxy checkpoint had seen) or
    "frac" (the share of its run the checkpoint sits at: the `_vs_frac` twin);
    `paper` writes the `_paper` copy, PNG and SVG, without the header (rule 18);
    `axes` the pair set the DA is over."""
    ref = DA_MODES[mode][0]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for s in size_order(summary["proxy_size"].unique()):
        g = summary[summary["proxy_size"] == s].sort_values("frac")
        n = g["n_languages"]
        langs = f"{n.min()} languages" if n.min() == n.max() else f"{n.min()} to {n.max()} languages"
        ax.errorbar(g[x], g["da"], yerr=g["da_se"].fillna(0), color=S.SIZE_COLOR.get(s, S.MUTED), marker="o",
                    ms=3.5, lw=1.4, capsize=2, label=f"{s}  ({langs})")
    ax.axhline(SAFE_DA, color=S.MUTED, lw=.8, ls=":")
    ax.set_ylim(0.25, 1.0)
    if x == "tokens":
        ax.set_xscale("log"); ax.set_xlabel("Training tokens of the language seen by the proxy checkpoint (log)")
    else:
        ax.set_xlim(0, 1.02); ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.set_xlabel("Share of the proxy's run at the checkpoint")
    ax.set_ylabel(f"Mean DA-{mode} of a language's BPB")
    ax.legend(frameon=False, loc="lower right", title="Proxy size"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    stem = DA_STEM.format(mode=mode, axes=AXES_SUFFIX[axes]) + ("_vs_frac" if x == "frac" else "")
    if paper:
        fig.tight_layout()
        summary.to_csv(out_dir / f"{stem}_paper.csv", index=False)
        S.save_paper(fig, out_dir / f"{stem}_paper")
        return
    what = {"goal": "like the reference once the proxy has seen enough of it",
            "ckpt": "like its own final once it has seen enough of it",
            "size": "like the reference at its final"}[mode]
    unit = "one proxy size at its final" if mode == "size" else "one proxy size at one tenth of its run"
    xdesc = ("geometric mean over languages of the tokens of the language the pair set's proxies had seen (mean over variants)"
             if x == "tokens" else "the share of the run the proxy checkpoint sits at")
    top = G._header(fig, f"Does a language's BPB rank the design variants {what}?",
                    f"point = {unit}: mean over languages of DA-{mode} (share of design-variant pairs the proxy's BPB of that "
                    f"language orders like {ref}; {'every pair' if axes == 'multi-axis' else 'the pairs moving one design axis'} at seed {GRID_SEED} of every data build, on the variants that train the "
                    f"language, ≥ {MIN_PAIRS} pairs), bar = standard error over languages; x = {xdesc}; "
                    f"{cells.dropna(subset=['tokens'])['task'].nunique()} languages, dotted = {SAFE_DA}"
                    + (f"; the {TARGET_SIZE} line is its own early checkpoints against its final" if mode == "goal" else ""))
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save_figure(fig, out_dir, stem)


# --- B: share of a benchmark's cells above chance against the tokens of the language -------------

def gate_cells(fin: pd.DataFrame, ckpts: bool = False) -> pd.DataFrame:
    """One row per (task, size, L) of the gateable benchmarks — per (task,
    size, L, tenth) with `ckpts`, `fin` then holding the scores at the tenths:
    the runs' mean score, the share of them above chance (rule 1's per-run
    test), the verdict at MIN_SHARE, and the tokens of the task's language they
    had trained on (the tenth's share of the run's tokens with `ckpts`; mean
    over the runs; NaN when any run's build plan is unreachable)."""
    fin = fin[fin["kind"] == "benchmark"].copy()
    fin["above"] = above_chance(fin["primary_score"].to_numpy(), fin["task"].to_numpy()).to_numpy()
    fin = fin.dropna(subset=["above"])            # no chance level or item count: cannot be gated
    fin["language"] = fin["task"].map(assign_language)
    fin = languages_only(fin)
    fin["tokens"] = [_tokens(L, d, size, a, lang)
                     for L, d, size, a, lang in zip(fin["L"], fin["data"], fin["size"], fin["ladder"], fin["language"])]
    keys = ["task", "language", "size", "L"]
    if ckpts:
        fin["tokens"] *= fin["frac"]
        keys.append("frac")
    cells = (fin.groupby(keys)
             .agg(task_score=("primary_score", "mean"), share_above=("above", "mean"), n_runs=("model", "nunique"),
                  train_tokens=("tokens", "mean"), missing=("tokens", lambda v: v.isna().any()))
             .reset_index())
    cells.loc[cells["missing"], "train_tokens"] = np.nan
    cells["above_chance"] = cells["share_above"] >= MIN_SHARE
    cells["benchmark"] = cells["task"].map(benchmark_family)
    cells["language_scheme"] = "L" + cells["L"].astype(int).astype(str)
    return cells.rename(columns={"size": "model_size"})[
        ["benchmark", "task", "language", "model_size", "language_scheme"] + (["frac"] if ckpts else [])
        + ["train_tokens", "task_score", "above_chance", "share_above", "n_runs"]]


def bin_cells(cells: pd.DataFrame, by: str = "benchmark") -> pd.DataFrame:
    """Per (panel, size, log-token bin): the share of the cells above chance,
    their count and the geometric mean of their tokens. `by` names the panel
    column (PANELS; "all" pools every cell into one panel)."""
    c = cells.dropna(subset=["train_tokens"]).copy()
    if by == "all":
        c["all"] = "all benchmarks, all languages"
    c["bin"] = np.floor(np.log10(c["train_tokens"]) * BINS_PER_DECADE).astype(int)
    out = (c.groupby([by, "model_size", "bin"])
           .agg(share_above=("above_chance", "mean"), n_cells=("task", "size"),
                tokens=("train_tokens", lambda v: 10 ** np.log10(v).mean()))
           .reset_index())
    out["bin_lo"], out["bin_hi"] = 10 ** (out["bin"] / BINS_PER_DECADE), 10 ** ((out["bin"] + 1) / BINS_PER_DECADE)
    return out


def pass_stem(by: str, population: str, ckpts: bool) -> str:
    return f"{PASS_NAME}_{'all' if by == 'all' else 'by_' + by}_{population}" + ("_ckpts" if ckpts else "")


def _pass_panel(ax, g0: pd.DataFrame, sizes: list, counts: bool = True) -> None:
    """One panel: the share of cells above chance against the tokens of the
    language, one line per size, the cell count on every point (`counts`)."""
    for s in sizes:
        g = g0[g0["model_size"] == s].sort_values("tokens")
        ax.plot(g["tokens"], g["share_above"], color=S.SIZE_COLOR.get(s, S.MUTED), marker="o", ms=3, lw=1.2)
        for x, y, n in zip(g["tokens"], g["share_above"], g["n_cells"]) if counts else ():
            ax.annotate(str(n), (x, y), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=4.5, color=S.MUTED)
    ax.set_xscale("log"); ax.set_ylim(-0.04, 1.12); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(color=S.GRID, lw=.6); S.clean(ax)


def _size_legend(ax, sizes: list) -> None:
    ax.legend(handles=[plt.Line2D([], [], color=S.SIZE_COLOR.get(s, S.MUTED), lw=2, label=s) for s in sizes],
              fontsize=6.5, frameon=False, loc="upper left", title="model size", title_fontsize=6.5)


def _cell_label(ckpts: bool) -> str:
    return "Share of (lang., K, ckpt) above chance" if ckpts else "Share of (lang., K) above chance"


def plot_pass(points: pd.DataFrame, cells: pd.DataFrame, out_dir: Path, population: str, ckpts: bool = False,
              by: str = "benchmark", paper: bool = False) -> None:
    """`paper`: no header, the language count alone in a panel's title, the
    cell named on the y axis, written as `<stem>_paper.png/.svg`."""
    if by == "all":
        cells = cells.assign(all="all benchmarks, all languages")
    if by == "benchmark":
        fams = G.panel_order(points["benchmark"].unique())
    elif by == "language":                      # the best-resourced language first
        fams = list(cells.groupby("language")["train_tokens"].median().sort_values(ascending=False).index)
    else:
        fams = [cells["all"].iloc[0]]
    sizes = size_order(points["model_size"].unique())
    ncols = {"benchmark": 5, "language": 10, "all": 1}[by]
    nrows = -(-len(fams) // ncols)
    w, h = {"benchmark": (3.3, 2.7), "language": (2.2, 2.0), "all": (6.4, 4.2)}[by]
    fig, axes = plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows + 1.0), sharex=True, sharey=True, squeeze=False)
    flat = axes.ravel()
    for i, (ax, fam) in enumerate(zip(flat, fams)):
        _pass_panel(ax, points[points[by] == fam], sizes, counts=not paper)
        nc = cells[cells[by] == fam]
        head = {"benchmark": G.display(fam) if by == "benchmark" else fam,
                "language": f"{fam}  ({nc['benchmark'].nunique()} benchmarks, {len(nc)} cells)",
                "all": f"{fam}  ({nc['benchmark'].nunique()} benchmarks, {nc['language'].nunique()} languages, {len(nc)} cells)"}
        if by == "benchmark" and paper:
            title = f"{G.paper_name(fam)} ({nc['language'].nunique()})"     # the language count, as the other paper figures
        elif by == "benchmark":
            title = f"{head['benchmark']}  ({nc['language'].nunique()} languages, {len(nc)} cells)"
        else:
            title = head[by]
        ax.set_title(title, loc="center" if paper else "left", fontsize=8 if by != "language" else 7)
        if i + ncols >= len(fams):                # the last panel of its column shows the ticks
            ax.tick_params(labelbottom=True)
    for ax in flat[len(fams):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(_cell_label(ckpts) if paper else "cells above chance")
    fig.supxlabel("Training tokens of the task's language (log)", fontsize=9)
    _size_legend(flat[0], sizes)
    runs, verdict = POPULATIONS[population]
    stem = pass_stem(by, population, ckpts)
    if paper:
        fig.tight_layout()
        points.to_csv(out_dir / f"{stem}_paper.csv", index=False)
        S.save_paper(fig, out_dir / f"{stem}_paper")
        return
    unit, where = (("(task, size, L, tenth of the run)", "that checkpoint had seen (the tenth × the run's tokens)") if ckpts
                   else ("(task, size, L)", "a full run of the cell trains on"))
    whose, of = {"benchmark": ("a benchmark's", "language, L"), "language": ("a language's", "benchmark, L"),
                 "all": ("all", "benchmark, language, L")}[by]
    top = G._header(fig, f"Share of {whose} cells above chance against the tokens of the language seen — {population}"
                    f"{', ten checkpoints' if ckpts else ''}",
                    f"cell = one ({of}{', checkpoint' if ckpts else ''}) = one {unit}: {runs}; above chance = {verdict}; "
                    f"x = the tokens of the task's language {where} "
                    f"(its share of the mixture × D(N)), cells binned {BINS_PER_DECADE} per decade, point = share "
                    f"of the bin's cells above chance, small number = cells in the bin, x = their geometric mean; one line per size"
                    f"{'; panels from the best- to the least-resourced language (median tokens of its cells)' if by == 'language' else ''}; "
                    f"benchmarks with a chance level only, trained languages (rule 2), parent tasks (rule 6); "
                    f"{cells.dropna(subset=['train_tokens'])['task'].nunique()} tasks, {cells['language'].nunique()} languages")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save_figure(fig, out_dir, stem)


def plot_pass_include_rf(points: pd.DataFrame, cells: pd.DataFrame, out_dir: Path, population: str, ckpts: bool = True) -> None:
    """The paper's INCLUDE pair: the original over its `rf_` twin, the two
    panels stacked in the height of one panel of the full figure."""
    fams = ["include_base_44", "rf_include_base_44"]
    sizes = size_order(points["model_size"].unique())
    fig, axes = plt.subplots(2, 1, figsize=(3.3, 3.4), sharex=True, sharey=True)
    for ax, fam in zip(axes, fams):
        _pass_panel(ax, points[points["benchmark"] == fam], sizes, counts=False)
        ax.set_yticks([0, 0.5, 1.0]); ax.set_title(G.paper_name(fam), loc="center", fontsize=8)
    axes[-1].set_xlabel("Training tokens of the task's language")
    fig.supylabel(_cell_label(ckpts), fontsize=8.5)
    axes[0].legend(handles=[plt.Line2D([], [], color=S.SIZE_COLOR.get(s, S.MUTED), lw=2, label=s) for s in sizes],
                   fontsize=6, frameon=False, loc="upper right", ncol=1)
    fig.tight_layout()
    stem = pass_stem("benchmark", population, ckpts) + "_include_rf_paper_vertical"
    points[points["benchmark"].isin(fams)].to_csv(out_dir / f"{stem}.csv", index=False)
    S.save_paper(fig, out_dir / stem)


def paper(pass_dir: Path, da_dir: Path, population: str = "1904", ckpts: bool = True) -> None:
    """The paper copies, from the `_by_benchmark_` tables on disk."""
    stem = pass_stem("benchmark", population, ckpts)
    cells, points = pd.read_csv(pass_dir / f"{stem}.csv"), pd.read_csv(pass_dir / f"{stem}_points.csv")
    plot_pass(points, cells, pass_dir, population, ckpts, "benchmark", paper=True)
    plot_pass_include_rf(points, cells, pass_dir, population, ckpts)
    summary = pd.read_csv(da_dir / f"{DA_NAME}.csv")
    if summary.empty:
        print(f"!!! {DA_NAME}.csv is empty (no cell had a token count): its paper copy is not redrawn")
    else:
        plot_da(summary, pd.read_csv(da_dir / f"{DA_NAME}_cells.csv"), da_dir, paper=True)


# --- README ----------------------------------------------------------------------------

# (mode, pair set, x axis, alt text): every DA figure the README embeds, in reading order
DA_FIGURES = [("goal", "multi-axis", "tokens", "DA-goal of BPB vs tokens seen"),
              ("goal", "mono-axis", "tokens", "DA-goal of BPB vs tokens seen, mono-axis pairs"),
              ("goal", "multi-axis", "frac", "DA-goal of BPB vs share of the run"),
              ("ckpt", "multi-axis", "tokens", "DA-ckpt of BPB vs tokens seen"),
              ("ckpt", "mono-axis", "tokens", "DA-ckpt of BPB vs tokens seen, mono-axis pairs"),
              ("ckpt", "multi-axis", "frac", "DA-ckpt of BPB vs share of the run"),
              ("size", "multi-axis", "tokens", "DA-size of BPB vs tokens seen")]
GITHUB = "https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis"


def _gh_links(folder: Path, rel: str, name: str) -> str:
    """README rule 6: the figure and its CSV on the main branch (internal only: rule 19)."""
    base = f"{GITHUB}/{folder.name}/{rel}/{name}"
    return f"[PNG on GitHub]({base}.png) · [CSV on GitHub]({base}.csv)"


def _da_figure(rel: str, mode: str, axes: str, x: str, alt: str) -> list[str]:
    """One DA figure: the image, its GitHub links and Key findings read from its
    table on disk (README rule 5); nothing when the table is missing."""
    name = DA_STEM.format(mode=mode, axes=AXES_SUFFIX[axes]) + ("_vs_frac" if x == "frac" else "")
    path = DA_VS_TRAIN_TOKENS / rel / f"{name}.csv"
    if not path.exists():
        return []
    s = pd.read_csv(path)
    proxies = [z for z in size_order(s["proxy_size"].unique()) if z != TARGET_SIZE]
    last = s[s["proxy_size"].isin(proxies)].sort_values("frac").groupby("proxy_size").last()["da"]
    lo, hi = last.idxmin(), last.idxmax()
    bullets = [f"- Over {int(s['n_languages'].max())} languages, the proxies' last point ranges from {fmt(last[lo])} "
               f"({lo}) to {fmt(last[hi])} ({hi})."]
    if mode != "size":
        first = {z: g.sort_values("frac").loc[lambda d: d["da"] >= SAFE_DA, "frac"].min()
                 for z, g in s[s["proxy_size"].isin(proxies)].groupby("proxy_size")}
        bullets.append(f"- First tenth of the run at DA ≥ {SAFE_DA}: " + ", ".join(
            f"{z} {'never' if pd.isna(first[z]) else f'{first[z]:.0%}'}" for z in proxies) + ".")
    if axes == "mono-axis":
        multi = pd.read_csv(DA_VS_TRAIN_TOKENS / rel / f"{DA_STEM.format(mode=mode, axes=AXES_SUFFIX['multi-axis'])}.csv")
        gap = s.merge(multi, on=["proxy_size", "frac"], suffixes=("", "_multi"))
        gap = gap[gap["proxy_size"].isin(proxies)]
        gap = (gap["da_multi"] - gap["da"])
        bullets.append(f"- The mono-axis pairs read lower than the multi-axis set at {int((gap > 0).sum())} of {len(gap)} "
                       f"(size, tenth) points, by {fmt(gap.mean())} on average.")
    return [f"![{alt}]({rel}/{name}.png)", _gh_links(DA_VS_TRAIN_TOKENS, rel, name), "Key findings:\n" + "\n".join(bullets)]


def _pass_finding(population: str, c: pd.DataFrame) -> str:
    """One Key-findings bullet per population: the share of its cells above chance, overall and per size."""
    by = c.groupby("model_size")["above_chance"].mean()
    by = by.reindex(size_order(by.index))
    return (f"- `{population}`: {c['above_chance'].mean():.0%} of {len(c)} cells above chance; per size "
            + ", ".join(f"{z} {v:.0%}" for z, v in by.items()) + ".")


def generate_readme(pool: str, summary: pd.DataFrame, cells_a: pd.DataFrame,
                    cells_b: dict[str, pd.DataFrame]) -> None:
    """Two blocks: the DA figures' in rq02_da_vs_train_tokens/README.md, the
    above-chance figures' in rq00_chance_vs_train_tokens/README.md."""
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    sizes = size_order(summary["proxy_size"].unique())
    at = lambda s, f: summary[(summary["proxy_size"] == s) & (np.isclose(summary["frac"], f))].iloc[0]  # noqa: E731
    rows = [[s, f"{at(s, .2)['tokens'] / 1e9:.2f} B", fmt(at(s, .2)["da"]), f"{at(s, 1.0)['tokens'] / 1e9:.2f} B",
             fmt(at(s, 1.0)["da"]), int(at(s, 1.0)["n_languages"])]
            for s in sizes if s != TARGET_SIZE and not summary[(summary["proxy_size"] == s) & np.isclose(summary["frac"], 1.0)].empty]
    missing_a = int(cells_a["tokens"].isna().sum())
    missing_b = {p: int(c["train_tokens"].isna().sum()) for p, c in cells_b.items()}
    unreachable = ("" if not missing_a and not any(missing_b.values()) else
                   f"\n\n!!! Token counts unreachable (the build's plan is not readable from where this ran): {missing_a} cells of "
                   f"the DA figure, " + ", ".join(f"{n} of `{p}`" for p, n in missing_b.items()) + " — those cells are left out of "
                   "the figures and carry NaN in the tables.")
    intro = (f"`tokens_seen.py`, `{pool}` pool: a language's evaluation against the training tokens of that language "
             f"the model had seen (its share of the mixture from the build's plan × the cell's budget × the checkpoint's share of the "
             f"run). Regenerate with `python analysis/rq01_scaling_predictability/tokens_seen.py --pool {pool}`." + unreachable)
    da_body = "\n\n".join([
        "## Decision accuracy against the tokens of the language seen", intro,
        f"**DA-goal of a language's BPB against the tokens seen.** Per proxy size and tenth of the run, the mean over languages "
        f"of the share of design-variant pairs the proxy's BPB orders like the {TARGET_SIZE} final (rq02's kernel, every pair at "
        f"seed {GRID_SEED} of every data build on the variants that train the language, ≥ {MIN_PAIRS} pairs; rule 15's multi-axis "
        f"set), with its standard error over languages; the x of a cell is the mean over the pair set's proxies of the tokens of "
        f"the language they had seen, and a point's x the geometric mean over languages "
        f"({cells_a.dropna(subset=['tokens'])['task'].nunique()} languages; `{DA_NAME}_cells.csv` has the per-language cells "
        f"with the min and max over variants). The {TARGET_SIZE} line is its own early checkpoints against its final. "
        f"`{DA_STEM.format(mode='ckpt', axes=AXES_SUFFIX['multi-axis'])}` ranks each checkpoint against the proxy size's OWN final (DA-ckpt) and "
        f"`{DA_STEM.format(mode='size', axes=AXES_SUFFIX['multi-axis'])}` the final of every size against the {TARGET_SIZE} final (DA-size, one point per size: "
        f"the 100 % end of the DA-goal lines); the `_vs_frac` twins of the goal and ckpt figures put the same points against the "
        f"share of the run the checkpoint sits at, where lines that are apart on the token axis falling together says the "
        f"schedule, not the exposure, decides. The `{AXES_SUFFIX['mono-axis']}` twins of the goal and ckpt figures read only the "
        f"pairs that move one design axis (rule 15).",
        md_table(["proxy size", "tokens of a language at 1C", "DA at 1C", "tokens at 5C", "DA at 5C", "languages"], rows)]
        + [part for mode, axes, x, alt in DA_FIGURES for part in _da_figure(rel, mode, axes, x, alt)])
    replace_block(DA_VS_TRAIN_TOKENS / "README.md", "da-vs-train-tokens", da_body, f"tokens_seen.py --pool {pool}")
    pass_body = "\n\n".join([
        "## Share of cells above chance against the tokens of the language seen", intro,
        f"**Share of a benchmark's cells above chance against the tokens seen.** One (task, size, L) cell per benchmark, "
        f"language and language setting; above chance by rule 1's Wilson test on the cell's runs; cells binned {BINS_PER_DECADE} "
        f"per decade of tokens, a point = the share of the bin's cells above chance with the cell count, one line per size. Two "
        f"populations: `deep_A_1904`, {POPULATIONS['deep_A_1904'][0]}; `1904`, {POPULATIONS['1904'][0]}. The `.csv` next to each "
        f"figure is the cell table (benchmark, task, language, model_size, language_scheme, train_tokens, task_score, "
        f"above_chance, share_above, n_runs), `_points.csv` the binned values drawn. The `_ckpts` twins read the same runs at "
        f"every evaluated tenth (a cell = (task, size, L, tenth), x = the tokens seen by that checkpoint): ten times the cells, "
        f"and a token axis that runs through every training run. Each cell table is drawn per benchmark (`_by_benchmark_`), "
        f"per language (`_by_language_`, panels from the best- to the least-resourced language) and pooled (`_all_`, one panel; "
        f"their `.csv` is the binned points drawn). "
        f"Cells above chance: "
        + ", ".join(f"`{p}` {int(c['above_chance'].sum())} of {len(c)} ({c['task'].nunique()} tasks)" for p, c in cells_b.items()) + ".",
    ] + [f"![Share above chance vs tokens seen, {p}]({rel}/{PASS_NAME}_by_benchmark_{p}.png)\n\n"
         + _gh_links(CHANCE_VS_TRAIN_TOKENS, rel, f"{PASS_NAME}_by_benchmark_{p}") for p in cells_b]
      + ["Key findings:\n" + "\n".join(_pass_finding(p, c) for p, c in cells_b.items())])
    replace_block(CHANCE_VS_TRAIN_TOKENS / "README.md", "chance-vs-train-tokens", pass_body, f"tokens_seen.py --pool {pool}")
    print(f"Wrote auto README blocks → {DA_VS_TRAIN_TOKENS / 'README.md'}, {CHANCE_VS_TRAIN_TOKENS / 'README.md'}")


# --- driver ------------------------------------------------------------------------------

def main(pool: str, pass_dir: Path, da_dir: Path) -> None:
    df = ladder_frame(pool)
    df = df[df["seed"] == GRID_SEED].copy()       # the grid seed: a replicate is a draw of one design, not a second design
    df["bucket"] = df["size"].map(size_bucket)
    for d in (pass_dir, da_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"Pool '{pool}' at seed {GRID_SEED}: {df['model'].nunique()} cells")

    mono = pair_sets(design_axes(df))["mono-axis"]
    for mode, axes in ((m, a) for m in DA_MODES for a in DA_AXES[m]):
        stem, out_dir = DA_STEM.format(mode=mode, axes=AXES_SUFFIX[axes]), da_dir
        c = da_cells(df, mode, pairs=mono if axes == "mono-axis" else None)
        s = da_summary(c)
        c.to_csv(out_dir / f"{stem}_cells.csv", index=False)
        s.to_csv(out_dir / f"{stem}.csv", index=False)
        missing = c[c["tokens"].isna()]
        print(f"A [{mode}, {axes}]: {len(c)} (language, size, tenth) cells over {c['task'].nunique()} languages with ≥ {MIN_PAIRS} pairs "
              f"(rule 5; the kernel leaves the others out), {c['n_variants'].min()}–{c['n_variants'].max()} variants per cell")
        if len(missing):
            print(f"!!! A [{mode}, {axes}]: {len(missing)} cells over {missing['task'].nunique()} languages have no token count "
                  f"(build plan unreachable) and are left out of the figure")
        if not s.empty:
            plot_da(s, c, out_dir, mode, axes=axes)
            if mode != "size":
                s.to_csv(out_dir / f"{stem}_vs_frac.csv", index=False)     # the same points, drawn against the run share (rule 12)
                plot_da(s, c, out_dir, mode, x="frac", axes=axes)
        if mode == "goal" and axes == "multi-axis":
            cells, summary = c, s

    fin = finals(df)
    # the same runs at every evaluated tenth: the `_ckpts` twins (`frac` = the tenth asked for)
    tenths = pd.concat([at_fraction(df, f).assign(frac=f) for f in SHARED_FRACS], ignore_index=True)
    cells_b, out_dir = {}, pass_dir
    for population in POPULATIONS:
        for ckpts, frame in ((False, fin), (True, tenths)):
            sub = frame[(frame["ladder"] == "deep") & (frame["data"] == "A")] if population == "deep_A_1904" else frame
            c = gate_cells(sub, ckpts)
            c.to_csv(out_dir / f"{pass_stem('benchmark', population, ckpts)}.csv", index=False)
            cells_b[population + ("_ckpts" if ckpts else "")] = c
            n_missing = int(c["train_tokens"].isna().sum())
            print(f"B [{population}{', tenths' if ckpts else ''}]: {len(c)} cells over {c['task'].nunique()} tasks, "
                  f"{c['benchmark'].nunique()} benchmarks, {c['language'].nunique()} languages; {int(c['above_chance'].sum())} above chance"
                  + (f"; !!! {n_missing} cells without a token count (build plan unreachable), left out of the figure" if n_missing else ""))
            for by in PANELS:                 # the cell table is the by_benchmark stem's .csv; the others' .csv is the binned points
                points = bin_cells(c, by)
                points.to_csv(out_dir / (f"{pass_stem(by, population, ckpts)}_points.csv" if by == "benchmark"
                                         else f"{pass_stem(by, population, ckpts)}.csv"), index=False)
                if not points.empty:
                    plot_pass(points, c, out_dir, population, ckpts, by)
    paper(pass_dir, da_dir)
    if pool == CANONICAL:
        generate_readme(pool, summary, cells, cells_b)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL, help=f"Ladder pool from configs/models.json (default: {CANONICAL})")
    p.add_argument("--paper", action="store_true", help="only the paper copies of the pass figure, from the tables on disk")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    sub = Path(load_pools()[args.pool].get("stage", "pretraining")) / args.pool
    dirs = (CHANCE_VS_TRAIN_TOKENS / sub, DA_VS_TRAIN_TOKENS / sub)
    paper(*dirs) if args.paper else main(args.pool, *dirs)
