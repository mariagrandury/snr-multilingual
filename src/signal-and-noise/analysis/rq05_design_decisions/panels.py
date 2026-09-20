"""rq05 per benchmark and per language: the decision table and the
early-decision read, without pooling the benchmarks.

    intervention_da_by_benchmark.png   agreement with the reference, proxy size x L; one subplot per (intervention, benchmark), intervention by intervention, BPB first
    intervention_da_by_language.png    the same, one subplot per (intervention, language)
    intervention_da_early_by_benchmark.png   the two planned decisions, proxy size x share of the run (mean over L), per benchmark
    intervention_da_early_by_language.png    the same per language
    da_lines.png                       DA-size (x = proxy size) and DA-ckpt (x = the reference's checkpoint), one line per
                                       intervention, mean over L; solid per-language BPB, dashed benchmarks, dotted training loss.
                                       Read on the ten evaluated checkpoints of every run (`intervention_da_ckpt10.csv` and
                                       `intervention_da_by_group_ckpt10.csv`, the decision table of analyze.py recomputed at
                                       every k/10 checkpoint; the rest of the folder stays on 20-100 %)
    da_lines_decided.png               da_lines on the items whose reference |Δ| is >= DECIDED seed sds (analyze.py)
    depth_crossover.png                deep − shallow final BPB per size x L in seed sds: which depth wins, and by more than noise?
    da_lines_flops.png                 the same with every (proxy size, checkpoint) cell at its training compute

Reads `intervention_da_by_benchmark.csv` and `intervention_da_by_language.csv`
(the per-item agreement `analyze.py` aggregates: per-language BPB on the
languages both levels train, and the benchmark tasks). Unlike the pooled
`benchmark` population of the decision table, these tables leave out the
language aggregates and the per-subject facets (`global_mmlu_full_<lang>_<subject>`),
which have no single language: about 60 % of the pooled benchmark items
remain, so a benchmark row here is not a slice of the pooled number. The
per-language table averages a language's BPB item with its benchmark items.

    python analysis/rq05_design_decisions/panels.py --pool predictivity_all
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
from analysis.autodoc import replace_block  # noqa: E402
from analysis.paths import DESIGN_DECISIONS  # noqa: E402
from analysis.rq05_design_decisions.analyze import (  # noqa: E402
    CANONICAL, COLOUR, DECIDED, INTERVENTIONS, MIN_ITEMS, intervention_da, seed_sd)
from analysis.rq05_design_decisions.early_decision import DECISIONS  # noqa: E402
from analysis.utils import CKPT_DA_EARLY_FRACS, GRID_SEED, LADDER_SIZES, TARGET_SIZE, finals, ladder_frame, trained_bpb_tasks  # noqa: E402

OUT_ROOT = DESIGN_DECISIONS
FRACS10 = list(CKPT_DA_EARLY_FRACS) + [1.0]   # every evaluated checkpoint (rule 3)
LINE_POPULATIONS = (("bpb_trained", "-", "per-language BPB (trained languages)"), ("benchmark", "--", "benchmark tasks"),
                    ("loss", ":", "training loss"))
mpl.rcParams.update(S.RC)


def one_reference(da: pd.DataFrame, series: str) -> tuple[pd.DataFrame, str]:
    """Keep, per (series, population) line, only the L's whose reference is
    the size most of that line's L's share (ties: the largest), so a line
    never averages decisions read against different references. Returns the
    frame and a caption line naming each line's reference and L's."""
    keep, words = [], []
    for (key, pop), g in da.groupby([series, "population"], sort=False):
        per_L = g[g["frac"] == 1.0].groupby("L")["reference_size"].first()
        if per_L.empty:
            continue
        counts = per_L.value_counts()
        ref = sorted(counts[counts == counts.max()].index, key=LADDER_SIZES.index)[-1]
        Ls = sorted(per_L.index[per_L == ref])
        keep.append(g[g["L"].isin(Ls)])
        words.append(f"{key}/{pop}: vs {ref} on L{','.join(map(str, Ls))}")
    return pd.concat(keep) if keep else da.iloc[0:0], "; ".join(words)


def da_lines(da: pd.DataFrame, out_dir: Path, *, name: str = "da_lines", series: str = "intervention", colours: dict = COLOUR,
             labels: dict | None = None, populations=LINE_POPULATIONS, title: str, note: str) -> None:
    """Two panels: DA-size (x = proxy size at its final checkpoint) and DA-ckpt
    (x = the reference's own checkpoints), one line per value of `series`
    (mean over the L's that share the line's reference size), one line style
    per population."""
    da, refs = one_reference(da[da[series].notna()], series)
    note = note + ". Each line keeps the L's that share one reference size (the largest size trained at both levels): " + refs
    size = (da[da["frac"] == 1.0].groupby([series, "population", "proxy_size"])["decision_acc"].mean().reset_index())
    ckpt = (da[(da["proxy_size"] == da["reference_size"]) & (da["frac"] < 1.0)]
            .groupby([series, "population", "frac"])["decision_acc"].mean().reset_index())
    sizes = [s_ for s_ in LADDER_SIZES if s_ in set(size["proxy_size"])]
    fracs = sorted(ckpt["frac"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6), sharey=True)
    tables = []
    for ax, t, x, xs, xlab, ttl in ((axes[0], size, "proxy_size", sizes, "proxy size (final checkpoint)",
                                     "DA-size: the proxy's final ranking vs the reference's"),
                                    (axes[1], ckpt, "frac", fracs, "reference's training tokens (× Chinchilla)",
                                     "DA-ckpt: the reference's own early checkpoints vs its final ranking")):
        for key in colours:
            for pop, ls, _ in populations:
                g = t[(t[series] == key) & (t["population"] == pop)].set_index(x).reindex(xs)
                if g["decision_acc"].notna().any():
                    ax.plot(range(len(xs)), g["decision_acc"], color=colours[key], ls=ls, marker="o", ms=3.5, lw=1.3)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels([str(v) if x != "frac" else G.chinchilla(v) for v in xs])
        ax.axhline(0.75, color=S.MUTED, lw=.8, ls=":"); ax.set_ylim(0.0, 1.02)
        ax.set_xlabel(xlab); ax.set_title(ttl, loc="left", fontsize=8.5); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
        tables.append(t.rename(columns={series: "row", x: "col", "decision_acc": "value"}).assign(panel=ttl)
                      .assign(row=lambda d: d["row"].astype(str) + " / " + d["population"])[["panel", "row", "col", "value"]])
    axes[0].set_ylabel("decision accuracy (mean over L)")
    axes[1].legend(handles=[plt.Line2D([], [], color=c, lw=2, label=(labels or {}).get(k, k)) for k, c in colours.items()]
                   + [plt.Line2D([], [], color=S.INK, ls=ls, label=lab) for _, ls, lab in populations],
                   fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    G.save_highlights(fig, out_dir, title, note, tables, name=name)


def da_lines_flops(da: pd.DataFrame, full_compute: pd.Series, out_dir: Path, *, name: str = "da_lines_flops",
                   series: str = "intervention", colours: dict = COLOUR, labels: dict | None = None,
                   populations=LINE_POPULATIONS, title: str, note: str) -> None:
    """One panel: every (proxy size, fraction) cell at its training compute
    (fraction x the size's mean full run, as a share of the reference's), y = DA
    (mean over the L's sharing the line's reference), one line per value of
    `series`, one style per population."""
    da, refs = one_reference(da[da[series].notna() & da["proxy_size"].isin(full_compute.index)], series)
    note = note + ". Each line keeps the L's that share one reference size: " + refs
    da = da.copy()
    da["compute_share"] = da["frac"] * da["proxy_size"].map(full_compute) / full_compute[TARGET_SIZE]
    t = da.groupby([series, "population", "proxy_size", "frac", "compute_share"])["decision_acc"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    tables = []
    for key in colours:
        for pop, ls, _ in populations:
            g = t[(t[series] == key) & (t["population"] == pop)].sort_values("compute_share")
            if len(g):
                ax.plot(g["compute_share"], g["decision_acc"], color=colours[key], ls=ls, marker="o", ms=2.8, lw=1.1)
                tables.append(g.assign(panel="flops", row=f"{key} / {pop}").rename(columns={"compute_share": "col", "decision_acc": "value"})
                              [["panel", "row", "col", "value"]])
    ax.set_xscale("log"); ax.axhline(0.75, color=S.MUTED, lw=.8, ls=":"); ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(f"training compute of the (proxy size, checkpoint) cell, share of the {TARGET_SIZE} run")
    ax.set_ylabel("decision accuracy (mean over L)"); ax.grid(color=S.GRID, lw=.6, which="both"); S.clean(ax)
    ax.legend(handles=[plt.Line2D([], [], color=c, lw=2, label=(labels or {}).get(k, k)) for k, c in colours.items()]
              + [plt.Line2D([], [], color=S.INK, ls=ls, label=lab) for _, ls, lab in populations],
              fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    G.save_highlights(fig, out_dir, title, note, tables, name=name)


def depth_crossover(frame: pd.DataFrame, out_dir: Path) -> None:
    """Which depth wins, per (size, L), and is it outside seed noise: deep −
    shallow final BPB on the languages the scheme-A list trains, in sds of
    that difference (sqrt(2) x the per-run seed sd), median over the
    languages (< 0: deep wins)."""
    fin = finals(frame)
    sd = seed_sd(fin)
    g = fin[(fin["seed"] == GRID_SEED) & (fin["scheme"] == "A") & (fin["kind"] == "bpb") & (fin["task"] != "bpb_macro")]
    piv = g.pivot_table(index=["size", "L", "task"], columns="arch", values="primary_score")
    if not {"deep", "shallow"} <= set(piv.columns):
        return
    piv = piv.dropna(subset=["deep", "shallow"])
    rows = []
    for (size, L), p in piv.groupby(level=["size", "L"]):
        tasks = p.index.get_level_values("task")
        p = p[tasks.isin(trained_bpb_tasks(int(L), "A") or set())]
        z = ((p["deep"] - p["shallow"]) / (np.sqrt(2) * p.index.get_level_values("task").map(sd).to_numpy(float))).dropna()
        if len(z) >= MIN_ITEMS:
            rows.append({"size": size, "L": int(L), "value": z.median(), "n": len(z)})
    if not rows:
        return
    t = pd.DataFrame(rows)
    sizes = [s_ for s_ in LADDER_SIZES if s_ in set(t["size"])]
    mat, cnt = (t.pivot(index="size", columns="L", values=v).reindex(sizes).rename(columns=lambda L: f"L{L}") for v in ("value", "n"))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    tables = [G.matrix_ax(ax, mat, "deep − shallow final BPB, in difference sds (< 0: deep wins)", cnt=cnt, vmin=-6, vmax=6, center=0.0,
                          cmap=S.DIV, fmt="{:+.1f}", xlabel="language setting", ylabel="model size")]
    G.save_highlights(fig, out_dir, "Depth: which architecture wins at each size, and is it outside seed noise?",
                      f"cell = median over the languages the scheme-A list trains of (deep − shallow final BPB) / sqrt(2) x the "
                      f"language's seed sd (a median over the baseline cells with 3 replicates), seed {GRID_SEED}; small number = "
                      f"languages; |cell| < 2 is the two-run difference inside seed noise", tables, name="depth_crossover")


def _panels(t: pd.DataFrame, by: str, path: Path, *, keys: list, ncols: int, **kw) -> None:
    units = G.panel_order(t[by].unique())
    t = t.assign(panel=[f"{INTERVENTIONS[k][0]} — {u}" for k, u in zip(t["intervention"], t[by])])
    order = [f"{INTERVENTIONS[k][0]} — {u}" for k in keys for u in units]
    G.panel_grid(t, path, by="panel", value="decision_acc", order=order, ncols=ncols, counts=False, csv=False,
                 row="proxy_size", row_order=[s for s in LADDER_SIZES if s in set(t["proxy_size"])],
                 cbar="agreement with the reference's final decision", ylabel="proxy size", **kw)


def highlights(out_dir: Path, fin: pd.DataFrame, keys: list) -> None:
    """rq05 on one page, final checkpoints: per intervention, how the
    agreement grows with the proxy size (BPB, benchmarks) and which
    benchmarks carry it."""
    sizes = [s for s in LADDER_SIZES if s in set(fin["proxy_size"])]
    label = {k: INTERVENTIONS[k][0] for k in keys}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    tables = []
    for ax, (name, g) in zip(axes, (("bits per byte", fin[fin["family"] == "bpb"]), ("benchmarks", fin[fin["family"] != "bpb"]))):
        m = g.groupby(["intervention", "proxy_size"])["decision_acc"].mean().unstack().reindex(index=keys, columns=sizes)
        tables.append(G.matrix_ax(ax, m.rename(index=label), f"Agreement with the reference's decision, {name}", xlabel="proxy size"))
    m = fin.groupby(["family", "intervention"])["decision_acc"].mean().unstack().reindex(columns=keys)
    tables.append(G.matrix_ax(axes[2], m.reindex(G.panel_order(m.index)).rename(columns=label), "Per benchmark, mean over proxy sizes",
                              xlabel="intervention"))
    for lab in axes[2].get_xticklabels():
        lab.set_rotation(30); lab.set_ha("right")
    G.save_highlights(fig, out_dir, "rq05 in one figure: would a small proxy have made the reference's design decision?",
                      "cell = share of items on which the proxy prefers the same level of the intervention as the reference, final "
                      "checkpoints, mean over language settings (and over benchmarks in the first two panels)", tables)


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    note = ("cell = share of the unit's items (a benchmark task or a language's BPB) on which the proxy prefers the same level "
            "of the intervention as the reference at its final checkpoint")
    for by, name, ncols in (("family", "benchmark", 6), ("language", "language", 10)):
        t = pd.read_csv(out_dir / f"intervention_da_by_{name}.csv")
        keys = [k for k in INTERVENTIONS if k in set(t["intervention"])]
        fin = t[t["frac"] == 1.0]
        _panels(fin, by, out_dir / f"intervention_da_by_{name}.png", keys=keys, ncols=ncols, col="L",
                col_order=sorted(fin["L"].unique()), col_label=lambda L: f"L{L}", xlabel="language setting",
                title=f"Does the proxy prefer the level the reference prefers? Final checkpoints, per {name}", note=note)
        early = t[t["intervention"].isin(DECISIONS)]
        early.to_csv(out_dir / "intervention_da_early.csv", index=False)      # rule 12: one table for both _early panels
        _panels(early, by, out_dir / f"intervention_da_early_by_{name}.png", keys=[k for k in DECISIONS if k in keys],
                ncols=ncols, col="frac", col_order=sorted(early["frac"].unique()), col_label=G.chinchilla,
                xlabel="proxy's training tokens (C = Chinchilla-optimal; 5C = the full run)", note=note,
                title=f"How small and how early, per {name} (mean over language settings)")
        if by == "family":
            highlights(out_dir, fin, keys)
    frame = ladder_frame(pool)
    da, _, groups = intervention_da(frame, fracs=FRACS10)
    da.to_csv(out_dir / "intervention_da_ckpt10.csv", index=False)
    gcols = ["intervention", "label", "L", "proxy_size", "frac", "reference_size", "group"]
    (groups.groupby(gcols).agg(decision_acc=("agree", "mean"), n_items=("agree", "size")).reset_index()
     if not groups.empty else pd.DataFrame(columns=gcols + ["decision_acc", "n_items"])
     ).to_csv(out_dir / "intervention_da_by_group_ckpt10.csv", index=False)
    if groups.empty:
        print("!!! RULE 2: no per-group BPB items — the never-trained groups are rq06's; only trained languages reach this pool")
    da_lines(da, out_dir, labels={k: v[0] for k, v in INTERVENTIONS.items()},
             title="How small and how early each design decision can be read",
             note="DA = share of items on which the proxy prefers the level of the intervention the reference prefers at its final "
                  "checkpoint, mean over the language settings; dotted line = 0.75")
    # a proxy cell's compute: the mean full-run compute of the size's families (deep and shallow differ by up to 13 %)
    depth_crossover(frame, out_dir)
    decided = da.assign(decision_acc=da["decision_acc_decided"])
    da_lines(decided, out_dir, name="da_lines_decided", labels={k: v[0] for k, v in INTERVENTIONS.items()},
             populations=LINE_POPULATIONS[:2],                  # the loss is one item: a 0/1 step, not a share
             title="The same, on the items the reference decides outside seed noise",
             note=f"DA as in da_lines, restricted to the items (languages' BPB, benchmark tasks) whose reference |Δ| between the "
                  f"two levels is at least {DECIDED:g} sds of that difference (sqrt(2) x the per-run seed sd, a median over the "
                  f"baseline cells with 3 replicates); a cell needs {MIN_ITEMS} such items; missing points = the reference "
                  f"decides too few items")
    da_lines_flops(da, frame.groupby(["size", "model"])["compute"].max().groupby("size").mean(), out_dir,
                   labels={k: v[0] for k, v in INTERVENTIONS.items()},
                   title="How much compute reads each design decision",
                   note="point = one (proxy size, checkpoint) cell at the compute spent up to that checkpoint; DA = share of items on "
                        "which the cell prefers the level the reference prefers at its final checkpoint, mean over L; dotted line = 0.75")
    if pool != CANONICAL:
        return
    rel = f"{stage}/{pool}"
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"The decision table and the early read above, without pooling the benchmarks (`{pool}` pool). Language "
        "aggregates and per-subject facets are left out here (about 60 % of the pooled benchmark items remain), "
        "and a language's subplot averages its BPB item with its benchmark items. Regenerate with "
        f"`python analysis/rq05_design_decisions/panels.py --pool {pool}`. White cells have no value; each figure's table "
        "sits next to it under the same name (`intervention_da_by_<unit>.csv`).",
        f"![rq05 in one figure]({rel}/highlights.png)",
        f"![Decisions by proxy size and checkpoint]({rel}/da_lines.png)",
        f"![The same on the items decided outside seed noise]({rel}/da_lines_decided.png)",
        f"![Which depth wins, in seed sds]({rel}/depth_crossover.png)",
        f"![Decisions by compute]({rel}/da_lines_flops.png)",
        f"![Decisions per benchmark]({rel}/intervention_da_by_benchmark.png)",
        f"![Early and small per benchmark]({rel}/intervention_da_early_by_benchmark.png)",
        f"![Decisions per language]({rel}/intervention_da_by_language.png)",
        f"![Early and small per language]({rel}/intervention_da_early_by_language.png)"])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
