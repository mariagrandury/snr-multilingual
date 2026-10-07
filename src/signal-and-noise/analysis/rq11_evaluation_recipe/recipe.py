"""RQ11 — which benchmark, posed how and scored how, reads the reference's
decision from the smallest proxy?

Every benchmark task is one VARIANT of a benchmark: a `format` (the original
items, their `rf_` cloze rewrite, their `rfgm_` LLM rewrite) and a `scoring`
(accuracy, or the gold answer's bits per byte, `bbpb_`): `utils.variant`.
This RQ puts the variants of each benchmark side by side and recommends one.

Every variant is read against a 1.7B score, and every output says which (the
`reading` column, bench_bpb_da.py's keys):

  acc_acc     an accuracy variant at the proxy  vs its accuracy at the reference
  bbpb_acc    -bBPB of a twin at the proxy      vs its original's ACCURACY at the reference
  bbpb_bbpb   bBPB of a twin at the proxy       vs its bBPB at the reference

acc_acc and bbpb_bbpb are the rows of the decision-accuracy table
`rq02_decision_accuracy/<stage>/<pool>/da_goal_early_small_per_task_both_axes.csv`
(per task, pair set, proxy size and evaluated tenth of the proxy's run, the DA
against the 1.7B final). bbpb_acc is computed here by that table's own kernel
(`compute_da.compute_early_small_decision_accuracy`) on the same frame, pairs
and grid, oriented with `utils.lower_is_better`: bench_bpb_da.py's bbpb_acc
reading. The original keeps its `rf_`/`rfgm_` prefix, so an RF twin is read
against the RF accuracy. Every table, figure and recommendation exists twice,
once per reading of the bBPB variants (`bbpb_reading`): with bbpb_acc every
variant predicts the same truth, the reference's accuracy decision; with
bbpb_bbpb the bBPB variants predict the reference's bBPB ranking, an easier
target. The accuracy variants are acc_acc in both. Per task, pair set and
reading:

  da_size_<s>    DA-size: the proxy s at its final checkpoint
  safe_size      the smallest proxy from which DA-size >= tau holds at every
                 larger proxy with a value (rq02's rule, grids.smallest_safe)
  safe_compute   the cheapest (proxy, checkpoint) cell, as a share of the
                 reference run's compute, from which every costlier cell clears
                 tau (rq02's safe-FLOPs rule, along the DA-goal grid); -1 = no
                 cell from which every costlier one clears tau; empty when only
                 final checkpoints exist (bBPB for now)

with tau = utils.RELIABLE_DA = 0.75 and rule 5 (n_pairs >= MIN_PAIRS). Rule 1
(`passes_gate`, mask `predictivity`): acc_acc at the proxy and the reference;
bbpb_bbpb passes (no chance level); bbpb_acc on the original's accuracy at the
reference only, as bench_bpb_da.py gates it (gating the proxy would discard
the regime bBPB is for). The bBPB twins exist at each run's final checkpoint
only until the per-item store covers every checkpoint. The rebuild fills their
DA-goal and DA-ckpt cells in the decision-accuracy tables; here it changes
only `safe_compute`. Every figure and table here is DA-size. A twin's cell
with fewer pairs than its original's (a family the store lacks) is blank
(`same_pairs`), so every variant of a task is read over the same decisions.

Per benchmark x variant: the languages (all of them, and those ranked: a task
with no value at any proxy, gated or under the pair minimum, has no safe rank),
the tasks ranked, the share of them reliable (DA >= tau) at each proxy, the
mean DA-size, the median safe size. The recommendation per benchmark and bBPB
reading: the variant with the smallest MEAN safe rank over its ranked tasks
(0 = 90M ... 4 = 1B, 5 = never; a median collapses to "never" whenever half the
languages never get there); a tie goes to the higher mean DA-size over the
(task, proxy) cells every tied variant has (`decided_by` says which rule picked
it); per benchmark x language: the variant with the smallest safe size, ties
to the higher mean DA-size.

Combined and per variant: mean DA-size and share of reliable tasks per proxy,
for each variant and for all variants together, over every task and over the
PAIRED set (the tasks whose own original, every `bbpb_`/`rf_`/`rfgm_` prefix
stripped, has an accuracy value at that proxy: the only head-to-head in which
the gate treats the variants alike, since bBPB passes where accuracy is at
chance).

    recipe_da_all_per_task{_multi_axes,_mono_axis}.csv          per task and reading
    recipe_da_size_by_variant{...}.csv                          benchmark x variant, per bBPB reading
    recipe_da_size_recommendation{...}.csv                      one row per benchmark and bBPB reading
    recipe_da_size_heatmap{...}.png/.csv                        benchmark x variant and reading, one panel per proxy
    recipe_da_size_ladder{...}.png/.csv                         the cheapest reliable proxy per variant, a panel per bBPB reading
    recipe_da_size_profiles{...}.png/.csv                       share of languages reliable per proxy, per benchmark
    recipe_da_size_variants{...}.png/.csv                       combined vs per variant and reading
    recipe_da_size_variants_multi_axes_paper.png/.svg/.csv      its first panel for the paper, every line against the 1.7B accuracy

    python analysis/rq11_evaluation_recipe/recipe.py --pool predictivity
    python analysis/rq11_evaluation_recipe/recipe.py --paper    # the paper figure alone, from the overview table on disk
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
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
from analysis.autodoc import CANONICAL_POOL, fmt, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY, EVALUATION_RECIPE  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask, task_n_items  # noqa: E402
from analysis.rq02_decision_accuracy import compute_da  # noqa: E402
from analysis.utils import (AXES_SUFFIX, BBPB, FORMATS, MIN_PAIRS, PAIR_AXES, RELIABLE_DA, SMALL_SIZES,  # noqa: E402
                            TARGET_SIZE, assign_language, benchmark_family, build_snr_pool, design_axes,
                            lower_is_better, pair_sets, passes_gate, variant)

TAU = RELIABLE_DA
FORMAT_NAME = {"original": "original", "rf": "RF", "rfgm": "LLM-RF"}
SCORING_NAME = {"acc": "accuracy", "bbpb": "bBPB"}
READINGS = ("acc_acc", "bbpb_acc", "bbpb_bbpb")             # <scoring at the proxy>_<score at the reference>
BBPB_READINGS = {"bbpb_acc": ("acc_acc", "bbpb_acc"), "bbpb_bbpb": ("acc_acc", "bbpb_bbpb")}   # the tasks each recommendation ranks
VARIANTS = [(f, r.split("_")[0], r) for r in READINGS for f in FORMATS]   # (format, scoring, reading)
ALL = "all variants"
LEVELS = list(SMALL_SIZES)                       # the safe-size levels; NEVER_CODE past the last
MIN_DRAWN = 5                                    # a mean over fewer tasks is kept in the CSV but not drawn
mpl.rcParams.update(S.RC)


def against(r: str) -> str:
    """`bbpb_acc` -> '→ 1.7B accuracy': the reference score a reading predicts."""
    return f"→ {TARGET_SIZE} {SCORING_NAME[r.split('_')[1]]}"


def vname(f: str, s: str, r: str) -> str:
    return f"{FORMAT_NAME[f]} · {SCORING_NAME[s]} {against(r)}"


def bbpb_name(r: str) -> str:
    """The name of a bBPB reading: 'bBPB → 1.7B accuracy'."""
    return f"bBPB {against(r)}"


def same_population(early: pd.DataFrame, df: pd.DataFrame, psets: dict) -> None:
    """Stop when rq02's table was computed on another population than the pool
    loads now (a table left from an earlier pool or report): its accuracy rows
    would be compared with bbpb_acc rows computed live on this pool. The test:
    the largest pair count of its accuracy rows is the number of the pool's
    pairs whose two families both reach the reference on one of its benchmark
    tasks (a family whose reference run has only its BPB scored, e.g. a 1.7B
    cell whose benchmark evals have not landed, cannot be in that table)."""
    acc = set(early.loc[~early["task"].str.startswith(BBPB), "task"])
    ref = set(df.loc[(df["bucket"] == TARGET_SIZE) & df["task"].isin(acc), "family"])
    for axes, g in early[~early["task"].str.startswith(BBPB)].groupby("axes"):
        want = sum(a in ref and b in ref for a, b in psets[axes])
        if g["n_pairs"].max() != want:
            sys.exit(f"rq02's early-small table has at most {g['n_pairs'].max()} {axes} pairs where the pool has {want}: "
                     "it was computed on another population; re-run compute_da.py on this pool first")


def bbpb_to_acc(df: pd.DataFrame, psets: dict) -> pd.DataFrame:
    """The early-small table's rows for every bBPB twin read against its
    original's ACCURACY at the reference (bbpb_acc): rq02's kernel, frame and
    pair sets, with the twin's score at the proxy cells and the original's
    accuracy at the reference's final."""
    twins = sorted(t for t in set(df["task"]) if t.startswith(BBPB))
    keep = df["task"].isin(set(twins) | {t[len(BBPB):] for t in twins})
    by = dict(tuple(df.loc[keep, ["model", "family", "bucket", "step", "task", "primary_score", "compute"]].groupby("task")))
    n_few, rows = len(compute_da._FEW_PAIRS), []
    for t in twins:
        acc = by[t[len(BBPB):]]
        acc = acc[acc["bucket"] == TARGET_SIZE]
        ref = acc[acc["step"] == acc.groupby("model")["step"].transform("max")]
        tw = by[t].assign(primary_score=-by[t]["primary_score"] if lower_is_better(t) else by[t]["primary_score"])
        # the kernel's reference is each family's last row at the reference size: keep the twin there only before
        # the original's final, so the reference is the accuracy and every proxy cell the twin
        tw = tw[(tw["bucket"] != TARGET_SIZE) | (tw["step"] < tw["model"].map(ref.set_index("model")["step"]))]
        fracs = compute_da.EARLY_SMALL_FRACS if tw.groupby("model")["step"].nunique().gt(1).any() else [1.0]   # finals only: no grid to read
        for axes in [a for a in PAIR_AXES if psets[a]]:
            rows += [{"task": t, "axes": axes, **r}
                     for r in compute_da.compute_early_small_decision_accuracy(pd.concat([tw, ref]), fracs=fracs, pairs=psets[axes])]
    print(f"bBPB → {TARGET_SIZE} accuracy: {len(twins)} twins; rule 5 left out "
          f"{len(compute_da._FEW_PAIRS) - n_few} cells under {MIN_PAIRS} pairs")
    return pd.DataFrame(rows, columns=["task", "axes", "proxy_size", "frac", "da", "n_pairs", "compute", "ref_compute"])


def same_pairs(early: pd.DataFrame) -> pd.DataFrame:
    """Blank a bBPB twin's cell whose pair count is below its original's at the
    same (pair set, proxy, tenth): a twin exists only where the per-item store
    holds the run, so fewer pairs means fewer families, and the variants would be
    compared over different decisions. Printed; it lifts once the store covers
    the pool."""
    key = ["axes", "proxy_size", "frac"]
    orig = early[early["reading"] == "acc_acc"].set_index(["task", *key])["n_pairs"]
    tw = early["task"].str.startswith(BBPB)
    idx = pd.MultiIndex.from_frame(early.loc[tw, ["task", *key]].assign(task=lambda d: d["task"].str[len(BBPB):]))
    short = early.loc[tw, "n_pairs"].to_numpy() < orig.reindex(idx).to_numpy()   # NaN (no original row): kept
    drop = early.loc[tw].index[short]
    if len(drop):
        print(f"!!! {len(drop)} of {int(tw.sum())} bBPB cells have fewer pairs than their original (families missing from "
              f"the per-item store): left blank until build_per_item_store.sbatch covers the pool")
    return early.assign(da=early["da"].where(~early.index.isin(drop)))


def per_task(early: pd.DataFrame, mask: pd.DataFrame | None, reading: str) -> pd.DataFrame:
    """One row per task, read one way: its variant, DA-size per proxy (gated
    and rule-5 cells NaN), the safe size and rank and the safe compute share."""
    e = early[(early["n_pairs"] >= MIN_PAIRS) & early["da"].notna()].copy()            # rule 5

    def gate(t: str, s: str) -> pd.Series:      # rule 1; bbpb_acc: the original's accuracy at the reference only
        return (passes_gate(mask, [t[len(BBPB):]], TARGET_SIZE) if reading == "bbpb_acc"
                else passes_gate(mask, [t], s, TARGET_SIZE))
    ok = {(t, s): bool(gate(t, s).iloc[0]) for t, s in set(zip(e["task"], e["proxy_size"]))}
    e["gated"] = [not ok[(t, s)] for t, s in zip(e["task"], e["proxy_size"])]
    fin = e[e["frac"] == 1.0]
    wide = fin[~fin["gated"]].pivot_table(index="task", columns="proxy_size", values="da").reindex(columns=LEVELS)
    tasks = sorted(set(e["task"]))
    out = pd.DataFrame(index=pd.Index(tasks, name="task"))
    for s in LEVELS:
        out[f"da_size_{s}"] = wide[s] if s in wide else np.nan
    gated = fin.pivot_table(index="task", columns="proxy_size", values="gated", aggfunc="all").reindex(index=tasks, columns=LEVELS)
    for s in LEVELS:
        out[f"gated_{s}"] = gated[s].fillna(False).astype(bool) if s in gated else False
    out["gated_everywhere"] = fin.groupby("task")["gated"].all().reindex(tasks).fillna(True)
    level = G.smallest_safe(wide.ge(TAU).where(wide.notna()))
    out["safe_size"] = level.reindex(tasks)
    known = out["safe_size"].notna()                                                    # NaN: no proxy had a value
    out["safe_rank"] = out["safe_size"].where(~known | (out["safe_size"] >= 0), len(LEVELS))   # never ranks after every size
    out["mean_da_size"] = out[[f"da_size_{s}" for s in LEVELS]].mean(axis=1)
    # the cheapest cell from which every costlier one clears tau, along compute (rq02's safe-FLOPs rule); a task
    # with final checkpoints only (bBPB until the store has the grid) has no compute ladder to read it on
    g = e[~e["gated"]].assign(share=lambda d: d["compute"] / d["ref_compute"]).sort_values("share")
    on_grid = e.groupby("task")["frac"].nunique().reindex(tasks).gt(1)
    out["safe_compute"] = pd.Series({t: _safe_share(x) for t, x in g.groupby("task")}).reindex(tasks).where(on_grid)
    out = out.reset_index()
    v = out["task"].map(variant)
    out["format"], out["scoring"], out["reading"] = v.str[0], v.str[1], reading
    original = out["task"].str.replace(f"^{BBPB}", "", regex=True)
    out["src"] = original.str.replace("^(rfgm|rf)_", "", regex=True)                  # the original-accuracy task it rewrites
    out["benchmark"] = original.map(lambda t: G.base(benchmark_family(t)))
    out["language"] = out["task"].map(assign_language)
    out["n_items"] = original.map(task_n_items)
    return out


def _safe_share(g: pd.DataFrame) -> float:
    ok = (g["da"] >= TAU).to_numpy()
    i = G.smallest_safe(pd.DataFrame([ok])).iloc[0]
    return np.nan if i != i else (-1.0 if i < 0 else float(min(g["share"].iloc[int(i)], 1.0)))


def _tie_da_size(t: pd.DataFrame, by: pd.DataFrame) -> pd.Series:
    """For the variants tied at a benchmark's smallest mean safe rank: the mean
    DA-size over the (task, proxy) cells every one of them has, per proxy and
    then over proxies as mean_da_size, so a tie is broken on the same languages."""
    out = pd.Series(np.nan, index=by.index)
    cols = [f"da_size_{s}" for s in LEVELS]
    for b, g in by.groupby("benchmark"):
        tied = g[g["mean_safe_rank"] == g["mean_safe_rank"].min()]
        if len(tied) < 2:
            continue
        x = t[t["benchmark"] == b].melt(id_vars=["src", "format", "scoring"], value_vars=cols, var_name="size", value_name="da")
        cells = x.pivot_table(index=["src", "size"], columns=["format", "scoring"], values="da")
        cells = cells.reindex(columns=list(zip(tied["format"], tied["scoring"]))).dropna()     # the cells every tied variant has
        m = cells.groupby(level="size").mean().mean()
        out[tied.index] = [m.get(k, np.nan) for k in zip(tied["format"], tied["scoring"])]
    return out


def recommend(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """benchmark x variant summary, and the recommended variant per benchmark,
    over one bBPB reading's tasks; `t` gains `recommended` per (benchmark, language)."""
    never = len(LEVELS)
    rows = []
    for (b, f, s, rd), g in t.groupby(["benchmark", "format", "scoring", "reading"]):
        ranked = g["safe_rank"].notna()               # the population of the rank columns (rule 13)
        r = {"benchmark": b, "format": f, "scoring": s, "reading": rd, "variant": vname(f, s, rd),
             "languages": g["language"].nunique(), "languages_ranked": g.loc[ranked, "language"].nunique(),
             "n_tasks": int(ranked.sum()), "mean_safe_rank": g["safe_rank"].mean(), "median_safe_rank": g["safe_rank"].median(),
             "safe_by_1B": (g["safe_size"].dropna() >= 0).mean() if g["safe_size"].notna().any() else np.nan}
        for size in LEVELS:
            d = g[f"da_size_{size}"]
            r[f"mean_da_size_{size}"] = d.mean()
            r[f"n_tasks_{size}"] = int(d.notna().sum())
            r[f"reliable_share_{size}"] = (d >= TAU).sum() / d.notna().sum() if d.notna().any() else np.nan
            r[f"all_gated_{size}"] = bool(g[f"gated_{size}"].all())
        rows.append(r)
    by = pd.DataFrame(rows)
    by["median_safe_size"] = by["median_safe_rank"].map(lambda x: np.nan if x != x else
                                                        ("never" if x >= never else LEVELS[int(np.floor(x))]))
    by = by.dropna(subset=["median_safe_rank"]).reset_index(drop=True)
    by["mean_da_size"] = by[[f"mean_da_size_{s}" for s in LEVELS]].mean(axis=1)
    by["tie_da_size"] = _tie_da_size(t, by)
    best = (by.sort_values(["benchmark", "mean_safe_rank", "tie_da_size", "mean_da_size"], ascending=[True, True, False, False],
                           na_position="last").groupby("benchmark").head(1))
    rank = by.groupby("benchmark")["mean_safe_rank"]
    rec = best[["benchmark", "variant", "format", "scoring", "reading", "mean_safe_rank", "median_safe_size", "safe_by_1B",
                "languages", "languages_ranked", "n_tasks", "mean_da_size", "tie_da_size"]].set_index("benchmark")
    rec["n_variants"] = rank.size()
    rec["accuracy_competes"] = by.groupby("benchmark")["scoring"].apply(lambda x: (x == "acc").any())
    tied = (by["mean_safe_rank"] == rank.transform("min")).groupby(by["benchmark"]).sum().reindex(rec.index)
    rec["decided_by"] = np.where(rec["n_variants"] == 1, "only variant", np.where(tied > 1, "DA-size tie-break", "safe rank"))
    order = t.sort_values(["benchmark", "language", "safe_rank", "mean_da_size"], ascending=[True, True, True, False])
    first = order.dropna(subset=["safe_size"]).groupby(["benchmark", "language"]).head(1).index
    t["recommended"] = t.index.isin(first)
    return by, rec.reset_index()


def overview(t: pd.DataFrame) -> pd.DataFrame:
    """Per proxy: mean DA-size and the share of reliable tasks, for every
    variant and reading and for all variants together per bBPB reading (its
    `reading` is that bBPB reading), over every task and the paired set."""
    rows = []
    for s in LEVELS:
        c = f"da_size_{s}"
        have = t.dropna(subset=[c])
        orig = set(have.loc[(have["reading"] == "acc_acc") & (have["format"] == "original"), "task"])
        paired = have[have["src"].isin(orig)]                 # the task's own original has an accuracy value here
        for pop, d in (("every task", have), ("paired", paired)):
            groups = [(vname(f, sc, r), r, d[(d["format"] == f) & (d["reading"] == r)]) for f, sc, r in VARIANTS]
            groups += [(f"{ALL}, {bbpb_name(r)}", r, d[d["reading"].isin(rs)]) for r, rs in BBPB_READINGS.items()]
            for name, r, g in groups:
                if len(g):
                    rows.append({"population": pop, "variant": name, "reading": r, "size": s, "mean_da_size": g[c].mean(),
                                 "reliable_share": (g[c] >= TAU).mean(), "n_tasks": len(g),
                                 "n_benchmarks": g["benchmark"].nunique()})
    return pd.DataFrame(rows)


def _columns(names) -> list[str]:
    """The variant labels present, in VARIANTS order."""
    return [vname(f, s, r) for f, s, r in VARIANTS if vname(f, s, r) in set(names)]


LS = {"acc_acc": "-", "bbpb_acc": ":", "bbpb_bbpb": "--"}      # a reading's line style
FORMAT_COLOUR = {"original": S.INK, "rf": S.SERIES[0], "rfgm": S.SERIES[2]}


def heatmap(by: pd.DataFrame, path: Path, note: str) -> None:
    by = by.drop_duplicates(["benchmark", "variant"])            # an accuracy variant is the same row in both readings
    cols = _columns(by["variant"])
    rows = sorted(set(by["benchmark"]), key=G.paper_name)
    fig, axes = plt.subplots(1, len(LEVELS), figsize=((0.38 * len(cols) + 0.6) * len(LEVELS) + 2.5, 0.28 * len(rows) + 3.6),
                             sharey=True)
    tables = []
    for ax, s in zip(axes, LEVELS):
        piv = by.pivot_table(index="benchmark", columns="variant", values=f"mean_da_size_{s}").reindex(index=rows, columns=cols)
        cnt = by.pivot_table(index="benchmark", columns="variant", values=f"n_tasks_{s}").reindex(index=rows, columns=cols)
        gated = by.pivot_table(index="benchmark", columns="variant", values=f"all_gated_{s}", aggfunc="all").reindex(index=rows, columns=cols)
        tables.append(G.matrix_ax(ax, piv.rename(index=G.paper_name), f"{s} proxy", cnt=cnt.where(piv.notna()).rename(index=G.paper_name),
                                  vmin=0.3, vmax=1.0, center=TAU, cmap=S.DIV, fmt="{:.2f}",
                                  gated=gated.rename(index=G.paper_name).astype(float).fillna(0).astype(bool)))
        ax.tick_params(axis="x", labelrotation=70, labelsize=6)
    G.save_highlights(fig, path.parent, f"Which variant of each benchmark reads the {TARGET_SIZE} decision? (τ = {TAU:g}; "
                      f"bBPB read against the {TARGET_SIZE} accuracy and against the {TARGET_SIZE} bBPB)",
                      note + f" Cell = mean DA-size over the benchmark's tasks with a value at that proxy; colour centred on "
                      f"τ = {TAU:g}; small number = those tasks; grey = every task gated (rule 1), white = no value.",
                      tables, name=path.stem)


def ladder(by: pd.DataFrame, rec: pd.DataFrame, path: Path, note: str) -> None:
    """Per benchmark, a marker per variant at its mean safe rank over the
    ranked tasks (0 = 90M ... 4 = 1B, 5 = never): how small a proxy each way
    of evaluating the benchmark needs, on average; a panel per bBPB reading.
    Marker size = share of tasks safe by 1B; the recommended variant is ringed."""
    first = by[by["bbpb_reading"] == "bbpb_acc"].groupby("benchmark")["mean_safe_rank"].min()
    rows = sorted(set(by["benchmark"]), key=lambda b: (first.get(b, len(LEVELS) + 1), G.paper_name(b)))
    marker = {"acc": "o", "bbpb": "D"}
    fig, axes = plt.subplots(1, len(BBPB_READINGS), figsize=(13, 0.3 * len(rows) + 1.8), sharey=True)
    offset = {(f, s): (i - (len(FORMATS) * 2 - 1) / 2) * 0.11 for i, (f, s) in enumerate((f, s) for s in ("acc", "bbpb") for f in FORMATS)}
    tab = []
    for ax, (rd, readings) in zip(axes, BBPB_READINGS.items()):
        b_rd, x = by[by["bbpb_reading"] == rd], rec[rec["bbpb_reading"] == rd]
        picked = set(zip(x["benchmark"], x["format"], x["scoring"]))
        for y, b in enumerate(rows):
            for r in b_rd[b_rd["benchmark"] == b].itertuples():
                ax.scatter(r.mean_safe_rank, y + offset[(r.format, r.scoring)], s=12 + 70 * r.safe_by_1B, marker=marker[r.scoring],
                           color=FORMAT_COLOUR[r.format], zorder=3, linewidth=1.4,
                           edgecolor=S.SERIES[1] if (b, r.format, r.scoring) in picked else "none")
                tab.append({"panel": bbpb_name(rd), "row": G.paper_name(b), "col": r.variant, "value": r.mean_safe_rank})
        ax.set_yticks(range(len(rows))); ax.set_yticklabels([G.paper_name(b) for b in rows], fontsize=7)
        ax.set_xticks(range(len(LEVELS) + 1)); ax.set_xticklabels(LEVELS + ["never"])
        ax.set_xlim(-0.5, len(LEVELS) + 0.5); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
        ax.set_title(f"accuracy {against('acc_acc')}, {bbpb_name(rd)}", loc="left", fontsize=8.5)
        ax.set_xlabel(f"smallest proxy whose DA-size stays ≥ τ = {TAU:g}, mean over tasks ('never' = one step past 1B)",
                      fontsize=7.5)
    axes[0].invert_yaxis()
    handles = ([plt.Line2D([], [], ls="", marker="o", color=FORMAT_COLOUR[f], label=FORMAT_NAME[f]) for f in FORMATS]
               + [plt.Line2D([], [], ls="", marker=marker[s], color=S.MUTED, label=SCORING_NAME[s]) for s in ("acc", "bbpb")]
               + [plt.Line2D([], [], ls="", marker="o", color="white", markeredgecolor=S.SERIES[1], label="recommended")])
    axes[-1].legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    G.save_highlights(fig, path.parent, "The cheapest reliable proxy, per benchmark and way of evaluating it, per bBPB reading",
                      note + " Marker size = share of the benchmark's ranked tasks safe by 1B; ringed = the recommended "
                      "variant of that panel (the smallest mean safe rank, then the higher mean DA-size on the cells the "
                      "tied variants share).", [pd.DataFrame(tab)], name=path.stem)


def profiles(by: pd.DataFrame, rec: pd.DataFrame, path: Path, note: str) -> None:
    """One small panel per benchmark: the share of its tasks reliable
    (DA-size >= tau) at each proxy, one line per variant and reading, the
    recommended ones thick: where each way of evaluating it starts to read the
    reference."""
    by = by.drop_duplicates(["benchmark", "variant"])
    rows = sorted(set(by["benchmark"]), key=G.paper_name)
    ncol = 6
    fig, axes = plt.subplots(int(np.ceil(len(rows) / ncol)), ncol, figsize=(2.6 * ncol, 2.0 * np.ceil(len(rows) / ncol) + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    picked = set(zip(rec["benchmark"], rec["format"], rec["reading"]))
    tab = []
    for ax, b in zip(axes.ravel(), rows):
        for r in by[by["benchmark"] == b].itertuples():
            y = [getattr(r, f"reliable_share_{s}") for s in LEVELS]
            ax.plot(range(len(LEVELS)), y, color=FORMAT_COLOUR[r.format], ls=LS[r.reading],
                    lw=2.4 if (b, r.format, r.reading) in picked else 1.1, marker="o", ms=2.5)
            tab += [{"panel": G.paper_name(b), "row": r.variant, "col": s, "value": v} for s, v in zip(LEVELS, y)]
        n = int(by.loc[by["benchmark"] == b, "languages"].max())
        ax.set_title(f"{G.paper_name(b)} ({n} lang.)", loc="left", fontsize=7.5)
        ax.set_ylim(-0.03, 1.03); ax.grid(color=S.GRID, lw=.5); S.clean(ax)
    spare = axes.ravel()[len(rows):]
    for ax in spare:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS, fontsize=6.5, rotation=45)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of tasks\nreliable", fontsize=7)
    handles = ([plt.Line2D([], [], color=FORMAT_COLOUR[f], label=FORMAT_NAME[f]) for f in FORMATS]
               + [plt.Line2D([], [], color=S.MUTED, ls=LS[r], label=f"{SCORING_NAME[r.split('_')[0]]} {against(r)}") for r in READINGS]
               + [plt.Line2D([], [], color=S.MUTED, lw=2.4, label="recommended")])
    (spare[0] if len(spare) else axes.ravel()[-1]).legend(handles=handles, fontsize=7, frameon=False, loc="center")
    G.save_highlights(fig, path.parent, f"Where each way of evaluating a benchmark starts to read the {TARGET_SIZE} decision",
                      note + f" Line = share of the benchmark's tasks with a value whose DA-size is ≥ τ = {TAU:g} at that proxy; "
                      f"thick = the recommended variant of each bBPB reading (a thick dotted line is the pick with bBPB "
                      f"{against('bbpb_acc')}, a thick dashed one with bBPB {against('bbpb_bbpb')}; a thick solid accuracy line "
                      "is the pick of every reading whose bBPB line is not thick).", [pd.DataFrame(tab)], name=path.stem)


def variants_figure(ov: pd.DataFrame, t: pd.DataFrame, path: Path, note: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    tables = []
    for ax, pop in zip(axes[:2], ("every task", "paired")):
        d = ov[ov["population"] == pop]
        lines = [(vname(f, s, r), dict(color={"original": S.MUTED, "rf": S.SERIES[0], "rfgm": S.SERIES[2]}[f], ls=LS[r], lw=1.4))
                 for f, s, r in VARIANTS]
        lines += [(f"{ALL}, {bbpb_name(r)}", dict(color=S.INK, ls=LS[r], lw=2.4)) for r in BBPB_READINGS]
        for name, style in lines:
            g = d[d["variant"] == name].set_index("size").reindex(LEVELS)
            g["mean_da_size"] = g["mean_da_size"].where(g["n_tasks"] >= MIN_DRAWN)
            if g["mean_da_size"].isna().all():
                continue
            ax.plot(range(len(LEVELS)), g["mean_da_size"], marker="o", ms=3.5, label=name, **style)
            tables.append(g.reset_index().assign(panel=pop, row=name, col=lambda x: x["size"], value=lambda x: x["mean_da_size"])
                          [["panel", "row", "col", "value"]])
        ax.axhline(TAU, color=S.MUTED, lw=.8, ls=":"); ax.axhline(0.5, color=S.GRID, lw=.8)
        ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS); ax.set_ylim(0.3, 1.0)
        ax.set_title(f"Mean DA-size, {pop}" + (" (tasks whose own original has an accuracy value)" if pop == "paired" else ""),
                     loc="left", fontsize=8.5)
        ax.set_xlabel("proxy size"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    axes[0].set_ylabel(f"DA-size (mean over tasks; dotted = τ = {TAU:g})")
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=3, handlelength=3.2)
    lv = t.assign(variant=[vname(f, s, r) for f, s, r in zip(t["format"], t["scoring"], t["reading"])],
                  level=t["safe_size"].where(~t["gated_everywhere"], G.GATED))           # grey = the gate, blank = no value
    level = lv.pivot_table(index="variant", columns="task", values="level", aggfunc="first")
    tables.append(G.stack_ax(axes[2], level, f"Smallest safe proxy (DA-size ≥ τ = {TAU:g}), share of each variant's tasks",
                             levels=LEVELS, xlabel="share of tasks", rows=_columns(level.index), name=str, legend_cols=3))
    G.save_highlights(fig, path.parent, f"Every way of evaluating a benchmark, together and apart (τ = {TAU:g})",
                      note + f" A point resting on fewer than {MIN_DRAWN} tasks is not drawn (it is in the CSV).",
                      tables, name=path.stem)


def variants_paper(ov: pd.DataFrame, path: Path) -> None:
    """The first panel of `variants_figure` for the paper: mean DA-size over
    every task per variant and for all variants together, no title. One target
    only: bBPB is read against the original's accuracy (`bbpb_acc`), so every
    line predicts the 1.7B accuracy, and the y label says so."""
    d = ov[ov["population"] == "every task"]
    reading = "bbpb_acc"
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    rows = []
    for f, s, r in [(f, s, r) for f in FORMATS for s, r in (("acc", "acc_acc"), ("bbpb", reading))] + [(None, None, reading)]:
        name = f"{ALL}, {bbpb_name(r)}" if f is None else vname(f, s, r)     # legend columns: one per format
        g = d[d["variant"] == name].set_index("size").reindex(LEVELS)
        g["mean_da_size"] = g["mean_da_size"].where(g["n_tasks"] >= MIN_DRAWN)
        label = "All variants" if f is None else f"{FORMAT_NAME[f][0].upper()}{FORMAT_NAME[f][1:]} {SCORING_NAME[s]}"
        style = dict(color=S.INK, ls="--", lw=2) if f is None else dict(
            color={"original": S.MUTED, "rf": S.SERIES[0], "rfgm": S.SERIES[2]}[f], ls="-" if s == "acc" else ":", lw=1.4)
        ax.plot(range(len(LEVELS)), g["mean_da_size"], marker="o", ms=3.5, label=label, **style)
        rows.append(g.reset_index()[["size", "mean_da_size", "n_tasks"]].assign(variant=label, reading=r))
    ax.axhline(TAU, color=S.MUTED, lw=.8, ls=":", label=f"Reliable ({TAU:g})")
    ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS); ax.set_ylim(0.4, 0.8)
    ax.set_xlabel("Proxy size"); ax.set_ylabel(f"Decision accuracy against {TARGET_SIZE} accuracy")
    ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    ax.legend(fontsize=6.5, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    pd.concat(rows)[["variant", "reading", "size", "mean_da_size", "n_tasks"]].to_csv(path.with_suffix(".csv"), index=False)
    S.save_paper(fig, path.with_suffix(""))


def readme(pool: str, rec: pd.DataFrame, ov: pd.DataFrame, t: pd.DataFrame) -> None:
    rel = "pretraining/" + pool
    never = len(LEVELS)
    # a mean over fewer than MIN_DRAWN tasks is blanked here as in the figures (it stays in the CSV)
    ov = ov.assign(cell=[f"{fmt(m)} ({n})" if n >= MIN_DRAWN else f"n = {n}" for m, n in zip(ov["mean_da_size"], ov["n_tasks"])])
    order = _columns(ov["variant"]) + [f"{ALL}, {bbpb_name(r)}" for r in BBPB_READINGS]

    def table(pop: str) -> str:
        p = ov[ov["population"] == pop].pivot_table(index="variant", columns="size", values="cell", aggfunc="first")
        return md_table(["variant"] + LEVELS, [[v] + list(p.loc[v].reindex(LEVELS).fillna("")) for v in order if v in p.index])

    # how each pick is made: reliable somewhere (a language safe at some proxy) or never, and against what
    split = []
    for rd in BBPB_READINGS:
        r = rec[rec["bbpb_reading"] == rd]
        for s in ("bbpb", "acc"):
            x = r[r["scoring"] == s]
            split.append([bbpb_name(rd), SCORING_NAME[s], len(x), int((x["mean_safe_rank"] < never).sum()),
                          int((x["mean_safe_rank"] >= never).sum()), int((x["decided_by"] == "DA-size tie-break").sum()),
                          int((x["decided_by"] == "only variant").sum()), int((~x["accuracy_competes"]).sum())])
    first = rec[rec["bbpb_reading"] == "bbpb_acc"].set_index("benchmark")
    benchmarks = sorted(set(t["benchmark"]), key=lambda b: (first["mean_safe_rank"].get(b, never + 1),
                                                           -first["mean_da_size"].get(b, 0), G.paper_name(b)))
    rec_rows = []
    for b in benchmarks:
        for rd in BBPB_READINGS:
            x = rec[(rec["benchmark"] == b) & (rec["bbpb_reading"] == rd)]
            if not len(x):
                rec_rows.append([G.paper_name(b), bbpb_name(rd), "— (no variant has a value: every cell at chance or under "
                                 "the pair minimum)", "", "", "", "", f"0 / {t.loc[t['benchmark'] == b, 'language'].nunique()}", "0", ""])
                continue
            r = x.iloc[0]
            thin = r.n_tasks < MIN_DRAWN
            rec_rows.append([G.paper_name(b), bbpb_name(rd), r.variant, r.decided_by,
                             f"n < {MIN_DRAWN}" if thin else fmt(r.mean_safe_rank), r.median_safe_size, fmt(r.safe_by_1B),
                             f"{int(r.languages_ranked)} / {int(r.languages)}", int(r.n_tasks),
                             f"n < {MIN_DRAWN}" if thin else fmt(r.mean_da_size)
                             + (f" (shared cells: {fmt(r.tie_da_size)})" if r.tie_da_size == r.tie_da_size else "")])
    gen = f"recipe.py --pool {pool}"
    # the snapshot of the input this run read: rq02's table (the ladder report's mtime would be the run's, not the table's)
    snap = datetime.fromtimestamp(early_table(pool).stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    readme_path = EVALUATION_RECIPE / "README.md"
    # one block per figure, so each figure's hand-written findings, follow-ups and links sit right after it (README rules 4-6)
    replace_block(readme_path, "results", "\n\n".join([
        f"DA-size against the {TARGET_SIZE} final, multi-axis pairs (rule 15), pool `{pool}`, rq02's early-small table "
        f"written {snap}, "
        f"≥ {MIN_PAIRS} pairs; reliable = "
        f"DA-size ≥ τ = {TAU:g} (`utils.RELIABLE_DA`); safe size = the smallest proxy from which it stays ≥ τ at every "
        f"larger proxy. Every row says what it predicts: an accuracy variant its accuracy at {TARGET_SIZE} (gate "
        f"`predictivity` at the proxy and the reference); a bBPB variant either its original's accuracy at {TARGET_SIZE} "
        f"(**bBPB {against('bbpb_acc')}**, gated on that accuracy at the reference only, the reading of "
        f"`bench_bpb_da.py`) or its own bBPB at {TARGET_SIZE} (**bBPB {against('bbpb_bbpb')}**, no chance level, never "
        f"gated). {t['task'].nunique()} tasks over {t['benchmark'].nunique()} benchmarks. The bBPB twins are read at final "
        f"checkpoints only for now. A mean over fewer than {MIN_DRAWN} tasks is left blank (`n < {MIN_DRAWN}`) in the tables "
        f"below and the figures; it stays in the CSVs. Regenerate with "
        f"`python analysis/rq11_evaluation_recipe/recipe.py --pool {pool}`.",
        f"**How the pick is made**, per bBPB reading and scoring of the pick (benchmarks with a value; reliable somewhere = "
        f"mean safe rank below {never}, a task of the pick safe from some proxy on; never reliable = {never}.00, no task of the "
        f"pick ever safe):",
        md_table(["bBPB read", "pick", "benchmarks", "reliable somewhere", "never reliable", "won on the DA-size tie-break",
                  "only variant with a value", "no accuracy variant has a value"], split),
        f"**The recommendation** (per benchmark and bBPB reading: the variant with the smallest mean safe rank over its "
        f"ranked tasks — 0 = safe from 90M, 4 = from 1B, {never} = never; a task gated or under the pair minimum at every "
        f"proxy has no rank, so the ranked languages and tasks are counted beside it (rule 13) — then the higher mean "
        f"DA-size over the (task, proxy) cells every tied variant has, given in brackets; τ = {TAU:g}), ordered by the "
        f"reading with bBPB {against('bbpb_acc')}:",
        md_table(["benchmark", "bBPB read", "evaluate it as", "decided by", "mean safe rank", "median safe size",
                  "share safe by 1B", "languages ranked / all", "tasks ranked", "mean DA-size"], rec_rows),
        f"![The cheapest reliable proxy]({rel}/recipe_da_size_ladder_multi_axes.png)"]), gen)
    replace_block(readme_path, "variants", "\n\n".join([
        f"**Every variant together and apart** — mean DA-size per proxy over every task with a value (τ = {TAU:g}; "
        f"in brackets the tasks behind each mean, which differ across proxies and variants because the gate keeps "
        f"different tasks at different sizes, rule 13):",
        table("every task"),
        "The same over the paired tasks (those whose own original, every `bbpb_`/`rf_`/`rfgm_` prefix stripped, has an "
        "accuracy value at that proxy, so the gate treats every variant alike):",
        table("paired"),
        f"![Variants together and apart]({rel}/recipe_da_size_variants_multi_axes.png)"]), gen)
    replace_block(readme_path, "heatmap", f"![Benchmark x variant heat map]({rel}/recipe_da_size_heatmap_multi_axes.png)", gen)
    replace_block(readme_path, "profiles", f"![Reliability profiles per benchmark]({rel}/recipe_da_size_profiles_multi_axes.png)", gen)


def early_table(pool: str) -> Path:
    """rq02's per-task early-small table, the input every reading but bbpb_acc comes from."""
    return DECISION_ACCURACY / load_pools()[pool].get("stage", "pretraining") / pool / "da_goal_early_small_per_task_both_axes.csv"


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = EVALUATION_RECIPE / stage / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    early = pd.read_csv(early_table(pool))
    early = early[~early["task"].str.startswith("bpb_") & (early["task"] != "train_loss")]          # benchmarks only
    df = build_snr_pool(pool)
    df["bucket"] = df["size"].map(size_bucket)
    psets = pair_sets(design_axes(df))
    same_population(early, df, psets)
    early = pd.concat([early.assign(reading=np.where(early["task"].str.startswith(BBPB), "bbpb_bbpb", "acc_acc")),
                       bbpb_to_acc(df, psets).assign(reading="bbpb_acc")], ignore_index=True)
    early = same_pairs(early)
    mask = load_mask(pool)
    for axes, g in early.groupby("axes"):
        sfx = AXES_SUFFIX[axes]
        t = pd.concat([per_task(x, mask, r) for r, x in g.groupby("reading")], ignore_index=True)
        bys, recs = [], []
        for rd, readings in BBPB_READINGS.items():
            sub = t[t["reading"].isin(readings)].copy()
            by, rec = recommend(sub)
            t[f"recommended_{rd}"] = t.index.isin(sub.index[sub["recommended"]])     # per (benchmark, language), this reading
            bys.append(by.assign(bbpb_reading=rd)); recs.append(rec.assign(bbpb_reading=rd))
        by, rec = pd.concat(bys, ignore_index=True), pd.concat(recs, ignore_index=True)
        ov = overview(t)
        t.assign(axes=axes).to_csv(out_dir / f"recipe_da_all_per_task{sfx}.csv", index=False)
        by.assign(axes=axes).to_csv(out_dir / f"recipe_da_size_by_variant{sfx}.csv", index=False)
        rec.assign(axes=axes).to_csv(out_dir / f"recipe_da_size_recommendation{sfx}.csv", index=False)
        ov.assign(axes=axes).to_csv(out_dir / f"recipe_da_size_overview{sfx}.csv", index=False)
        note = (f"DA-size against the {TARGET_SIZE} final, {axes} pairs of `{pool}`, ≥ {MIN_PAIRS} pairs; τ = {TAU:g}. "
                f"Accuracy {against('acc_acc')}, gate `predictivity` at the proxy and the reference; bBPB {against('bbpb_acc')} "
                f"(the original's accuracy, gated at the reference only) or bBPB {against('bbpb_bbpb')} (no chance level: "
                "never gated); bBPB at final checkpoints only.")
        heatmap(by, out_dir / f"recipe_da_size_heatmap{sfx}.png", note)
        ladder(by, rec, out_dir / f"recipe_da_size_ladder{sfx}.png", note)
        profiles(by, rec, out_dir / f"recipe_da_size_profiles{sfx}.png", note)
        variants_figure(ov, t, out_dir / f"recipe_da_size_variants{sfx}.png", note)
        if axes == "multi-axis":
            variants_paper(ov, out_dir / "recipe_da_size_variants_multi_axes_paper.png")
        print(f"[{axes}] {t['task'].nunique()} tasks over {t['benchmark'].nunique()} benchmarks; recommendation:")
        print(rec.pivot(index="benchmark", columns="bbpb_reading", values="variant").to_string())
        if axes == "multi-axis" and pool == CANONICAL_POOL:
            readme(pool, rec, ov, t)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    p.add_argument("--paper", action="store_true", help="only the paper figure, from the multi-axis overview table on disk")
    args = p.parse_args()
    if args.paper:
        d = EVALUATION_RECIPE / load_pools()[args.pool].get("stage", "pretraining") / args.pool
        variants_paper(pd.read_csv(d / "recipe_da_size_overview_multi_axes.csv"), d / "recipe_da_size_variants_multi_axes_paper.png")
    else:
        main(args.pool)
