"""DA-size of benchmark BPB (bBPB): bits-per-byte of the gold answer, against accuracy.

For each parent task in the rq08 per-item store (built with
`build_per_item_store.py --finals-only`), a model's bBPB is the item mean of
``-ll_gold / ln2 / bytes_gold`` at its final checkpoint (the OLMES / Heineman
et al. 2025 ``correct_bpb``). Per proxy size in SMALL_SIZES, against
TARGET_SIZE, over the multi-axis design pairs (rule 15), three readings, all
`pair_agreement` on the same families and pairs (rule 5's NaN below MIN_PAIRS):

  acc_acc     accuracy at the proxy  vs accuracy at the reference
  bbpb_acc    -bBPB at the proxy     vs accuracy at the reference
  bbpb_bbpb   bBPB at the proxy      vs bBPB at the reference

Rule 1 applies to the accuracy side only (bBPB has no chance level), with the
`predictivity` gate mask as every rq02 reader uses it. The head-to-head
population is the tasks whose accuracy is above chance at the reference: there
bbpb_acc and bbpb_bbpb count, and acc_acc also needs its usual gate at the
proxy, so its task count is smaller and is printed beside it. Gating bBPB at the
proxy would discard exactly the regime it is for (a proxy at chance on accuracy
whose bBPB still separates the designs). The paired test (bBPB -> acc minus
acc -> acc, Wilcoxon signed-rank over tasks) runs on the tasks where both are
defined. The raw ungated values and both gate flags stay in bench_bpb_da_size_multi_axes.csv.

A task whose `target` is not a choice index has no bBPB and drops out: xwinograd
(the answer string; the choices vary the context, not the continuation) and
lambada (one continuation).

A lettered task's continuation is " A": its bBPB is the letter's surprisal, not
the answer text's. Such tasks are flagged `letter` (mean gold length <= 2.5
bytes) and summarised apart from the cloze ones.

Outputs under pretraining/<pool>/: bench_bpb_da_size_multi_axes.csv (task x size),
bench_bpb_da_size_summary_multi_axes.csv (group x size), bench_bpb_da_size_heatmap_multi_axes.png,
bench_bpb_da_size_bars_multi_axes.png (overall), bench_bpb_da_size_bars_benchmarks_multi_axes.png (per
benchmark), each with its CSV, and the README's `bench-bpb` block. `--store`
names the per_item_store/<store> folder when it differs from the pool; the
merge keeps the pool's models, and the pool models the store lacks are printed
and named in the README block.

    python analysis/rq02_decision_accuracy/bench_bpb_da.py --pool predictivity [--store predictivity_schemes]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq08_subset_selection.build_per_item_store import STORE, bench_bpb, store_gap  # noqa: E402
from analysis.utils import (BBPB, SMALL_SIZES, TARGET_SIZE, benchmark_family, design_axes, finals,  # noqa: E402
                            ladder_frame, pair_agreement, pair_sets, passes_gate)

GATE_POOL = "predictivity"           # the gate every rq02 reader uses, whatever pool the DA comes from
DOC_POOL = "predictivity"             # the pool rq02's DA verdicts come from; only it writes the README
READINGS = {"acc_acc": "acc → 1.7B acc", "bbpb_acc": "bBPB → 1.7B acc", "bbpb_bbpb": "bBPB → 1.7B bBPB"}
ALL, CLOZE, LETTER = "all benchmarks", "all cloze", "all lettered"
mpl.rcParams.update(S.RC)


def da_table(pool: str, store_dir: Path) -> tuple[pd.DataFrame, list, int, list[str], dict]:
    """task x size: the three raw DAs, the gate flags and the three gated readings;
    the multi-axis pairs they are over (those between families the store holds),
    the pool's pair count, the pool models the store lacks, and the FineWeb2 val
    BPB tick (`bpb_macro`'s DA-size on the same pairs)."""
    per = bench_bpb(sorted(store_dir.glob("*.parquet"))).rename(columns={"bbpb": "bpb"})
    letter = per.groupby("task")["bytes_gold"].mean() <= 2.5

    df = ladder_frame(pool)
    pairs = pair_sets(design_axes(df))["multi-axis"]
    missing = store_gap(df["model"], store_dir)
    macro = finals(df[df["task"] == "bpb_macro"])
    df = finals(df[(df["kind"] == "benchmark") & ~df["task"].str.startswith(BBPB)]).merge(per[["model", "step", "task", "bpb"]], on=["model", "step", "task"])
    df = df.dropna(subset=["bpb"])   # a NaN score would count as a disagreeing pair in pair_agreement
    fams = set(df["family"])         # a family the store lacks has no bBPB: its pairs are not compared
    used = [(a, b) for a, b in pairs if a in fams and b in fams]
    print(f"{pool}: {df['model'].nunique()} models, {df['task'].nunique()} tasks with bBPB, {len(used)} of "
          f"{len(pairs)} multi-axis pairs (between families the store holds), finals only, gate {GATE_POOL}")
    at = {s: macro[macro["size"] == s].set_index("family")["primary_score"].to_dict() for s in [*SMALL_SIZES, TARGET_SIZE]}
    tick = {s: pair_agreement(at[s], at[TARGET_SIZE], used)[0] for s in SMALL_SIZES} if len(macro) else {}

    rows = []
    for task, g in df.groupby("task"):
        at = {s: g[g["size"] == s].set_index("family") for s in [*SMALL_SIZES, TARGET_SIZE]}
        ref = at[TARGET_SIZE]
        for s in SMALL_SIZES:
            p = at[s]
            da_acc, n = pair_agreement(p["primary_score"].to_dict(), ref["primary_score"].to_dict(), used)
            da_bpb, _ = pair_agreement(p["bpb"].to_dict(), ref["bpb"].to_dict(), used)
            da_x, _ = pair_agreement((-p["bpb"]).to_dict(), ref["primary_score"].to_dict(), used)
            rows.append({"task": task, "family": benchmark_family(task), "size": s, "da_acc": da_acc,
                         "da_bpb": da_bpb, "da_bpb_to_acc": da_x, "n_pairs": n})
    out = pd.DataFrame(rows)
    out.insert(1, "axes", "multi-axis")      # rule 15: the pair set every DA here is over
    out["letter"] = out["task"].map(letter)
    mask, tasks = load_mask(GATE_POOL), sorted(set(out["task"]))
    gate = {s: passes_gate(mask, tasks, s) for s in [*SMALL_SIZES, TARGET_SIZE]}
    out["above_chance_proxy"] = [bool(gate[s][t]) for t, s in zip(out["task"], out["size"])]
    out["above_chance_ref"] = out["task"].map(gate[TARGET_SIZE]).astype(bool)
    ref_ok = out["above_chance_ref"]
    out["acc_acc"] = out["da_acc"].where(ref_ok & out["above_chance_proxy"])
    out["bbpb_acc"] = out["da_bpb_to_acc"].where(ref_ok)
    out["bbpb_bbpb"] = out["da_bpb"].where(ref_ok)
    return out, used, len(pairs), missing, tick


def summarise(g: pd.DataFrame) -> pd.Series:
    """Mean, sd and task count of each reading; the paired test where both are defined."""
    both = g.dropna(subset=["acc_acc", "bbpb_acc"])
    d = both["bbpb_acc"] - both["acc_acc"]
    s = {"n_tasks": g["task"].nunique()}
    for c in READINGS:
        s |= {c: g[c].mean(), f"{c}_sd": g[c].std(), f"{c}_n": g[c].notna().sum()}
    return pd.Series(s | {"n_paired": len(d), "gain": d.mean(), "bbpb_better": (d > 0).mean(),
                          "acc_better": (d < 0).mean(),
                          "p": wilcoxon(d).pvalue if len(d) and (d != 0).any() else np.nan})


def summary_table(out: pd.DataFrame) -> pd.DataFrame:
    groups = [(ALL, out), (CLOZE, out[~out["letter"]]), (LETTER, out[out["letter"]])]
    fams = out.drop_duplicates("family").sort_values(["letter", "family"])
    groups += [(f, out[out["family"] == f]) for f in fams["family"]]
    return pd.concat({name: g.groupby("size").apply(summarise).reindex(SMALL_SIZES) for name, g in groups},
                     names=["group", "size"])


def label(group: str, letter: dict) -> str:
    return f"{G.display(group)} (letter)" if letter.get(group) else G.display(group)


def heatmap(summ: pd.DataFrame, letter: dict, out_dir: Path, note: str) -> None:
    groups = list(dict.fromkeys(summ.index.get_level_values("group")))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 1.6 + 0.22 * len(groups)), sharey=True)
    long = []
    for ax, (col, title) in zip(axes, READINGS.items()):
        mat = summ[col].unstack("size").reindex(index=groups, columns=SMALL_SIZES)
        cnt = summ[f"{col}_n"].unstack("size").reindex(index=groups, columns=SMALL_SIZES)
        gated = mat.isna() & (summ["n_tasks"].unstack("size").reindex(index=groups, columns=SMALL_SIZES) > 0)
        mat.index = cnt.index = gated.index = [label(g, letter) for g in groups]
        long.append(G.matrix_ax(ax, mat, title, cnt=cnt, vmin=0, vmax=1, cmap=S.DIV, center=0.5,
                                xlabel="proxy size", gated=gated))
    top = G._header(fig, "Decision accuracy of benchmark BPB (bBPB) against accuracy, proxy → 1.7B", note)
    fig.tight_layout(rect=(0, 0, 1, top))
    pd.concat(long).to_csv(out_dir / "bench_bpb_da_size_heatmap_multi_axes.csv", index=False)
    S.save_figure(fig, out_dir, "bench_bpb_da_size_heatmap_multi_axes")


def bars_ax(ax, s: pd.DataFrame, title: str, ref: dict | None = None) -> None:
    """Grouped bars, one group per proxy size, one bar per reading (mean DA over tasks)."""
    x, w = np.arange(len(SMALL_SIZES)), 0.26
    for k, (col, name) in enumerate(READINGS.items()):
        ax.bar(x + (k - 1) * w, s[col], w, color=S.SERIES[k], edgecolor=S.SURFACE, linewidth=1, label=name)
    if ref:
        ax.scatter(x, [ref.get(z, np.nan) for z in SMALL_SIZES], marker="_", s=260, color=S.INK, linewidths=1.5,
                   label="FineWeb2 val BPB (bpb_macro)", zorder=3)
    ax.axhline(0.5, color=S.MUTED, linewidth=0.8, linestyle="--")
    ax.set_xticks(x, SMALL_SIZES); ax.set_ylim(0, 1); ax.set_title(title, loc="left", fontsize=8.5)
    S.clean(ax); ax.grid(axis="y", color=S.GRID, linewidth=0.5); ax.set_axisbelow(True)


def legend_top(fig, ax, top: float) -> float:
    """One legend in the figure's upper right, in a band of its own under the
    header, clear of every bar and tick; returns the top left for the axes."""
    fig.legend(*ax.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.99, top), ncol=4,
               frameon=False, fontsize=7.5)
    return top - 0.35 / fig.get_figheight()


def bars(summ: pd.DataFrame, letter: dict, out_dir: Path, note: str, ref: dict) -> None:
    keep = [c for c in summ.columns if c != "p"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, g in zip(axes, [ALL, CLOZE, LETTER]):
        s = summ.loc[g]
        n = s["n_tasks"].max()
        if pd.isna(n):           # a store without lettered (or cloze) tasks: the group is all NaN rows
            ax.set_title(f"{g} (0 tasks)", loc="left", fontsize=8.5); S.clean(ax)
            continue
        bars_ax(ax, s, f"{g} ({int(n)} tasks)", ref)
    axes[0].set_ylabel("decision accuracy vs 1.7B")
    top = legend_top(fig, axes[0], G._header(fig, "Benchmark BPB against accuracy as the proxy for the 1.7B ranking", note))
    fig.tight_layout(rect=(0, 0, 1, top))
    summ.loc[[ALL, CLOZE, LETTER], keep].to_csv(out_dir / "bench_bpb_da_size_bars_multi_axes.csv")
    S.save_figure(fig, out_dir, "bench_bpb_da_size_bars_multi_axes")

    fams = [g for g in dict.fromkeys(summ.index.get_level_values("group")) if g not in (ALL, CLOZE, LETTER)]
    ncols = 6
    nrows = -(-len(fams) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.1 * ncols, 1.9 * nrows + 0.8), sharey=True, squeeze=False)
    for ax, g in zip(axes.flat, fams):
        s = summ.loc[g]
        bars_ax(ax, s, f"{label(g, letter)} ({int(s['n_tasks'].max())})")
        ax.tick_params(labelsize=6.5)
    for ax in axes.flat[len(fams):]:
        ax.set_visible(False)
    top = legend_top(fig, axes[0, 0], G._header(fig, "Benchmark BPB against accuracy, per benchmark (proxy → 1.7B)", note))
    fig.tight_layout(rect=(0, 0, 1, top))
    summ.loc[fams, keep].to_csv(out_dir / "bench_bpb_da_size_bars_benchmarks_multi_axes.csv")
    S.save_figure(fig, out_dir, "bench_bpb_da_size_bars_benchmarks_multi_axes")


def readme(summ: pd.DataFrame, letter: dict, ref: dict, n_pairs: int, n_pool_pairs: int, pool: str, store: str,
           missing: list[str]) -> None:
    def size_rows(g):
        return [[z, f"{fmt(r.acc_acc)} ({int(r.acc_acc_n)})", f"{fmt(r.bbpb_acc)} ({int(r.bbpb_acc_n)})",
                 fmt(r.bbpb_bbpb), f"{r.gain:+.2f} ({int(r.n_paired)})",
                 f"{r.bbpb_better:.0%} / {r.acc_better:.0%}", "<0.001" if r.p < 1e-3 else fmt(r.p, 3),
                 fmt(ref.get(z))] for z, r in summ.loc[g].iterrows()]
    head = ["proxy", "acc → acc (tasks)", "bBPB → acc (tasks)", "bBPB → bBPB", "paired gain (tasks)",
            "bBPB better / worse", "Wilcoxon p", "FineWeb2 val BPB"]
    fams = [g for g in dict.fromkeys(summ.index.get_level_values("group")) if g not in (ALL, CLOZE, LETTER)]
    mean = summ.groupby(level="group")[list(READINGS) + ["n_tasks"]].mean()
    fam_rows = [[label(f, letter), int(summ.loc[f, "n_tasks"].max())] + [fmt(mean.loc[f, c]) for c in READINGS]
                for f in fams]
    body = [
        "## Benchmark BPB against accuracy",
        f"DA-size, final checkpoints, multi-axis pairs of `{pool}` ({n_pairs} of its {n_pool_pairs} pairs: those between "
        f"families the store holds, for every reading and the FineWeb2 tick), gate `{GATE_POOL}` on the "
        "accuracy side only: every reading counts the tasks above chance at 1.7B, and acc → acc also needs the "
        "task above chance at the proxy (its task count is the smaller one). The paired gain is bBPB → acc "
        "minus acc → acc on the tasks where both are defined. FineWeb2 val BPB is `bpb_macro`'s DA-size on the same "
        "pairs. Regenerate with "
        f"`python analysis/rq02_decision_accuracy/bench_bpb_da.py --pool {pool}"
        f"{'' if store == pool else ' --store ' + store}` (after "
        f"`build_per_item_store.py --pool {store} --finals-only`)."
        + (f" The store `{store}` lacks {len(missing)} of the pool's models ({', '.join(missing)}); "
           "they are left out until the store is rebuilt for the pool." if missing else ""),
        *[x for g, cap in [(ALL, ""), (CLOZE, " (the answer text is the continuation)"),
                           (LETTER, " (the continuation is the letter: bBPB is the letter's surprisal)")]
          if summ.loc[g, "n_tasks"].notna().any() for x in (f"**{g}**{cap}", md_table(head, size_rows(g)))],
        "**Per benchmark**, mean over the five proxy sizes (tasks: the parent tasks with bBPB; cells: task-mean DA over those above chance at 1.7B, blank = all gated):",
        md_table(["benchmark", "tasks", *READINGS.values()], fam_rows),
        f"![bBPB DA, overall](pretraining/{pool}/bench_bpb_da_size_bars_multi_axes.png)",
        f"![bBPB DA, per benchmark](pretraining/{pool}/bench_bpb_da_size_bars_benchmarks_multi_axes.png)",
        f"![bBPB DA, heat map](pretraining/{pool}/bench_bpb_da_size_heatmap_multi_axes.png)",
    ]
    replace_block(DECISION_ACCURACY / "README.md", "bench-bpb", "\n\n".join(body), f"bench_bpb_da.py --pool {pool}")


def main(pool: str, store: str | None = None) -> None:
    store = store or pool
    if not any((STORE / store).glob("*.parquet")):
        # the store lives on the cluster only: an empty one would overwrite the
        # committed tables with column-less files (per_item_ladder.py does the same)
        print(f"no per-item store at {STORE / store}: nothing written (build_per_item_store.sbatch builds it)")
        return
    out, used, n_pool_pairs, missing, ref = da_table(pool, STORE / store)
    out_dir = DECISION_ACCURACY / "pretraining" / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "bench_bpb_da_size_multi_axes.csv", index=False)
    summ = summary_table(out)
    summ.assign(axes="multi-axis").to_csv(out_dir / "bench_bpb_da_size_summary_multi_axes.csv")
    print(summ.loc[[ALL, CLOZE, LETTER], ["n_tasks", *READINGS, "acc_acc_n", "bbpb_acc_n", "n_paired",
                                          "gain", "bbpb_better", "acc_better", "p"]].round(3).to_string())

    letter = out.groupby("family")["letter"].all().to_dict()
    note = (f"DA-size, final checkpoints, multi-axis pairs of {pool}; gate {GATE_POOL} on the accuracy side only "
            "(tasks above chance at 1.7B; acc → acc also above chance at the proxy). bBPB = item mean of "
            "-log2 p(gold answer) / UTF-8 bytes of the answer.")
    heatmap(summ, letter, out_dir, note + " Cells: mean DA over tasks, small number = tasks; colour centred on 0.5 "
            "(coin flip); grey = every task gated.")
    bars(summ, letter, out_dir, note + " Bars: mean DA over tasks (n in the panel title); dashed line = coin flip; "
         "black tick = FineWeb2 val BPB's DA; an empty panel = every task gated.", ref)
    if pool == DOC_POOL:
        readme(summ, letter, ref, len(used), n_pool_pairs, pool, store, missing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=DOC_POOL)
    ap.add_argument("--store", default=None, help="the per_item_store/<store> folder to read (default: the pool's)")
    a = ap.parse_args()
    main(a.pool, a.store)
