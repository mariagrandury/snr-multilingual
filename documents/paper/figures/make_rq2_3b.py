"""rq2.png extended to the 3B rung, for the paper (not part of the pipeline: it
reads cached tables, see Inputs).

The paper's rq2 figure (paper_rq2.py, variant above_66_either_transformation,
mono-axis) read on the six families trained at 3B (deep, L in {8, 15} in builds
A and B, plus A-L30 and A-L50, seed 1904) against the 3B final, and its twin on
the same six families against the 1.7B final, so the two differ in nothing but
the reference.

Computes nothing new: the per-task DA tables come from rq02's own kernel
(`by_L._da_table`, the code behind da_all_pooled_per_task / by_transformation),
reduced with by_L's `_summary`, filtered with reliable_tasks' `above_66_either`
rule (median DA-size or median DA-ckpt >= 0.66) recomputed on these families at
each reference, and drawn with paper_rq2's `_scale_lines` / `_run_lines`. The
gate at 3B is above_reference.gate_mask (the Wilson rule on the 3B runs).

Inputs (git-ignored, under src/signal-and-noise/data/rq2_3b/): frame6.pkl (the
ladder frame of the six families, 2026-10-11 report), bench_bpb_6fam.csv (the
bBPB twins of those families), the per-task DA caches per_task_ref{3B,1.7B}.pkl
and rq10's committed per-task tables for the known-answer check. Outputs:
documents/paper/figures/rq2_3b.png (reference 3B), rq2_3b_twin_ref1.7B.png (the
same six families against 1.7B), rq2_3b_compare.png (the two stacked, on the
tasks reliable under both references, so they differ in the reference alone),
and the _unfiltered versions; the plotted values next to the caches.

    cd src/signal-and-noise && PYTHONPATH=$PWD:$PWD/../../src python ../../documents/paper/figures/make_rq2_3b.py
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
HERE = REPO / "documents" / "paper" / "figures"
DATA = REPO / "src" / "signal-and-noise" / "data" / "rq2_3b"

import analysis.utils as U  # noqa: E402
U.BENCH_BPB = DATA / "bench_bpb_6fam.csv"           # the twins; the worktree has no bench_bpb.csv
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.rq00_gate_and_curves import above_random  # noqa: E402
from analysis.rq02_decision_accuracy import by_L, paper_rq2 as P  # noqa: E402
from analysis.rq02_decision_accuracy.compute_da import compute_early_small_decision_accuracy  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import FILTERS  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import OVERALL, pairs_by_group  # noqa: E402
from analysis.rq10_size_generalisation.above_reference import gate_mask  # noqa: E402
from evals.scripts.utils.configs import size_bucket  # noqa: E402

POOL = "predictivity"
FILTER = "above_66_either"
SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B", "3B"]


def frame() -> pd.DataFrame:
    f = DATA / "frame6.pkl"
    df = pd.read_pickle(f) if f.exists() else U.ladder_frame("predictivity_seeds", above_reference=True)
    if not f.exists():
        df.to_pickle(f)
    df = df[df["seed"] == U.GRID_SEED].copy()
    df["bucket"] = df["size"].map(size_bucket)
    return df


def per_task(df: pd.DataFrame, ref: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """by_L's (per design axis, pooled) per-task tables with `ref` as the reference."""
    out = DATA / f"per_task_ref{ref}.pkl"
    groups = pairs_by_group(U.design_axes(U.finals(df)), "transformation", "mono-axis")
    if out.exists():
        by_axis, pooled = pd.read_pickle(out)
        return by_axis, pooled, groups
    df = df[df["size"].map(U.NON_EMB) <= U.NON_EMB[ref]]
    by_L.TARGET_SIZE = ref                     # the kernel's reference (its `ref_full` and DA-goal target)
    by_L.compute_early_small_decision_accuracy = functools.partial(compute_early_small_decision_accuracy, target_size=ref)
    tables = {g: by_L._da_table(df[df["family"].isin({f for pr in pl for f in pr})], g, pl) for g, pl in groups.items()}
    by_axis = pd.DataFrame([r for g, rows in tables.items() if g != OVERALL for r in rows],
                           columns=by_L.COLS).rename(columns={"L": "axis"})
    pooled = pd.DataFrame(tables[OVERALL], columns=by_L.COLS).drop(columns="L")
    pd.to_pickle((by_axis, pooled), out)
    return by_axis, pooled, groups


def reliable(pooled: pd.DataFrame, ref: str) -> tuple[set, pd.DataFrame]:
    """reliable_tasks' `above_66_either` on these families: per benchmark task,
    median DA-size over the proxies' finals and median DA-ckpt over every size
    and the nine earlier tenths, each cell with >= MIN_PAIRS pairs and gated as
    rq02 gates it (DA-size at the proxy and the reference, DA-ckpt at the size)."""
    red, thresh, crit = FILTERS[FILTER]
    t = pooled
    size = t[(t["frac"] == 1.0) & (t["proxy_size"] != ref) & (t["n_pairs_ref"] >= U.MIN_PAIRS)].rename(columns={"da_ref": "da"})
    ckpt = t[(t["frac"] < 1.0) & (t["n_pairs_own"] >= U.MIN_PAIRS)].rename(columns={"da_own": "da"})
    cells = pd.concat([G.mark_gated(G.add_meta(size.dropna(subset=["da"])), POOL, "proxy_size", "da", ref).assign(kind="size"),
                       G.mark_gated(G.add_meta(ckpt.dropna(subset=["da"])), POOL, "proxy_size", "da", None).assign(kind="ckpt")])
    cells = cells[~cells["family"].isin(["bpb", "loss"])].dropna(subset=["da"])
    med = cells.pivot_table(index="task", columns="kind", values="da", aggfunc=red)
    ok = (med.get("size", pd.Series(np.nan, index=med.index)) >= thresh) | (med.get("ckpt", pd.Series(np.nan, index=med.index)) >= thresh)
    return set(med.index[ok]), med


def panels(by_axis, pooled, keep, ref):
    """paper_rq2.scoring_panels without the scoring split: DA-size per design
    axis and pooled, DA-ckpt and DA-goal over the mono-axis pairs."""
    reading = {k: {**v, "reference": ref if v["reference"] else None} for k, v in by_L.READINGS.items()}
    pooled, trans = pooled[pooled["task"].isin(keep)], by_axis[by_axis["task"].isin(keep) & (by_axis["frac"] == 1.0)]

    def summary(t, keys, name):
        d = by_L._summary(POOL, t, keys, reading[name])
        return d[d["group"] == "all benchmarks"].drop(columns="group")
    goal, ckpt = summary(pooled, [], "goal"), summary(pooled, [], "ckpt")
    size = pd.concat([goal[goal["frac"] == 1.0].assign(axis=OVERALL), summary(trans, ["axis"], "goal")])
    size = size.rename(columns={"axis": "group", "proxy_size": "size", "da": "reliability_macro", "tasks": "n_tasks"})
    size = size[size["size"] != ref]
    refrow = pd.DataFrame({"group": size["group"].unique(), "size": ref, "frac": 1.0, "reliability_macro": 1.0})
    size = pd.concat([size, refrow], ignore_index=True).assign(non_emb=lambda d: d["size"].map(U.NON_EMB))
    return size, ckpt, goal


def draw_row(ax3, size, ckpt, goal, ref, legend=True):
    P.TARGET_SIZE = ref                                   # the hollow self-reference point and the line set
    P.SMALL_SIZES = [s for s in SIZES if U.NON_EMB[s] < U.NON_EMB[ref]]
    rows = [P._scale_lines(ax3[0], size),
            P._run_lines(ax3[1], ckpt, "ckpt", "DA-ckpt (reference is the final checkpoint of the same size)", legend),
            P._run_lines(ax3[2], goal, "goal", f"DA-goal (reference is the final checkpoint of {ref})", False)]
    ax3[0].set_ylabel(f"DA-size (reference is the final checkpoint of {ref})")
    for ax in ax3:
        ax.set_ylim(*P.YLIM); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    return pd.concat(rows, ignore_index=True).assign(reference=ref)


def draw(size, ckpt, goal, ref, path):
    fig, ax3 = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    draw_row(ax3, size, ckpt, goal, ref).to_csv(DATA / f"{path.name}.csv", index=False)
    fig.tight_layout()
    fig.savefig(path.parent / f"{path.name}.png", dpi=200)   # not with_suffix: "ref1.7B" has a dot
    plt.close(fig)


def check(pooled, ref, committed):
    """Known answer: the mono-axis DA-size per task equals rq10's per-task table."""
    a = pooled[(pooled["frac"] == 1.0) & (pooled["n_pairs_ref"] >= U.MIN_PAIRS)][["task", "proxy_size", "da_ref", "n_pairs_ref"]]
    b = pd.read_csv(DATA / committed)
    b = b[(b["axes"] == "mono-axis") & (b["frac"] == 1.0) & b["da"].notna()][["task", "size", "da", "n_pairs"]]
    m = a.merge(b, left_on=["task", "proxy_size"], right_on=["task", "size"], how="outer", indicator=True)
    both = m[m["_merge"] == "both"]
    print(f"check vs {committed}: {len(both)} shared cells, {int(((both['da_ref'] - both['da']).abs() < 1e-9).sum())} exact, "
          f"pairs equal {int((both['n_pairs_ref'] == both['n_pairs']).sum())}; only mine {int((m['_merge'] == 'left_only').sum())}, "
          f"only rq10 {int((m['_merge'] == 'right_only').sum())}")


if __name__ == "__main__":
    df = frame()
    mask = gate_mask(df, POOL, "3B")          # the committed mask plus a 3B column (the Wilson rule on the 3B runs)
    above_random.load_mask = lambda pool: mask   # what G.mark_gated reads, so every gate below sees the 3B column
    kept, tabs = {}, {}
    for ref, name, committed in (("3B", "rq2_3b", "above_reference_3B_per_task.csv"),
                                 ("1.7B", "rq2_3b_twin_ref1.7B", "above_reference_1.7B_design3B_per_task.csv")):
        by_axis, pooled, groups = per_task(df, ref)
        check(pooled, ref, committed)
        keep, med = reliable(pooled, ref)
        med.assign(kept=med.index.isin(keep)).to_csv(DATA / f"{name}_reliability.csv")
        print(f"--- {name}: pairs per group {({g: len(p) for g, p in groups.items()})}; {len(keep)} tasks pass {FILTER} "
              f"({sum(t.startswith('bbpb_') for t in keep)} bBPB twins) of {len(med)} benchmark tasks with a cell")
        kept[ref], tabs[ref] = keep, (by_axis, pooled)
        size, ckpt, goal = panels(by_axis, pooled, keep, ref)
        draw(size, ckpt, goal, ref, HERE / name)
        # the unfiltered twin: every gated benchmark task
        allt = set(pooled["task"])
        draw(*panels(by_axis, pooled, allt, ref), ref, HERE / f"{name}_unfiltered")

    # the two references stacked, on the tasks reliable under both: they then differ in the reference alone
    both = kept["3B"] & kept["1.7B"]
    print(f"--- rq2_3b_compare: {len(both)} tasks reliable under both references "
          f"({sum(t.startswith('bbpb_') for t in both)} bBPB twins)")
    fig, ax = plt.subplots(2, 3, figsize=(13.5, 8.0), sharey=True)
    out = [draw_row(ax[i], *panels(*tabs[ref], both, ref), ref, legend=(i == 0)) for i, ref in enumerate(("3B", "1.7B"))]
    fig.tight_layout()
    pd.concat(out, ignore_index=True).to_csv(DATA / "rq2_3b_compare.csv", index=False)
    fig.savefig(HERE / "rq2_3b_compare.png", dpi=200)
    plt.close(fig)
