"""RQ10 — Which benchmarks the 3B rung can measure that the 1.7B reference cannot.

`above_reference.py` asks whether a RANKING survives one rung up. This asks the
prior question: whether the benchmark is above chance at all. Rule 1's gate is a
property of (task, size), not of the task — a one-sided 95 % Wilson lower bound
over the task's items must clear chance for at least half the runs at that size
— so a benchmark the reference rung cannot resolve may become measurable one
rung above it, and the reference's own mask would never say so.

Both columns are recomputed HERE on the same families that have a scored 3B final
(`above_reference.gate_mask` does it for the reference rung alone). The
committed mask's 1.7B column is pooled over every 1.7B run in the pool, so
reading it against a few-run 3B column would confound the rung with the run
count; this way the only thing that differs between the two columns is the rung.

    gate_crossover.png / .csv       the 2x2, and the crossings per benchmark and per language
    gate_crossover_by_benchmark.png / .csv
                                    per benchmark, the share of its tasks above chance at each
                                    rung (the paired view: which benchmarks the rung lifts)
    gate_crossover_per_task.csv     per task, the share of runs above chance at each rung, the
                                    gate verdict at each and whether it newly passes

A (family, task) cell enters only where it is scored at BOTH rungs, so a family
whose 3B evaluations have not landed (or a task one rung lacks) cannot put runs
in one column alone.

    python analysis/rq10_size_generalisation/gate_crossover.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
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
from analysis.paths import SIZE_GENERALISATION  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import MIN_SHARE, scores_and_mask  # noqa: E402
from analysis.rq10_size_generalisation.above_reference import REFERENCE, on_shared_grid  # noqa: E402
from analysis.utils import TARGET_SIZE, assign_language, benchmark_family, finals, ladder_frame  # noqa: E402

mpl.use("Agg")
OUT_ROOT = SIZE_GENERALISATION
KEY = "gate-crossover"


def crossover(pool: str, reference: str) -> tuple[pd.DataFrame, list]:
    """One row per task with a chance level at BOTH rungs: its gate at the
    reference, at `reference`, the share of its runs above chance at each
    (rq00's one-sided Wilson rule; the gate is that share >= MIN_SHARE), the
    runs behind each share, and the benchmark family / language it belongs
    to. Tasks with no chance level (BPB, the generative tasks) have no gate to
    cross and are absent by construction."""
    df = ladder_frame(pool, above_reference=True)
    df = df[on_shared_grid(df)]
    fams = sorted(set(finals(df).loc[finals(df)["size"] == reference, "family"]))
    at_final = df[df["family"].isin(fams) & (df["frac"] >= .99) & df["size"].isin([TARGET_SIZE, reference])]
    # the same cells at both rungs: a (family, task) scored at one rung only would
    # move one column's population, not its rung
    at_final = at_final[at_final.groupby(["family", "task"])["size"].transform("nunique") == 2]
    fams = sorted(at_final.loc[at_final["kind"] == "benchmark", "family"].unique())
    cols, shares, runs = {}, {}, {}
    for size in (TARGET_SIZE, reference):
        _, m, _, r, share, pop = scores_and_mask(at_final[at_final["size"] == size], sizes=[size], runs=True)
        cols[size], shares[size] = m[size], share[size]
        # the runs the share is over: the trained ones where any trained the task (rule 2), else all
        trained = r[r["trained"]] if "trained" in r else r
        n_tr, n_all = trained.groupby("task").size(), r.groupby("task").size()
        runs[size] = n_tr.reindex(m.index).where(pop[size] == "trained", n_all.reindex(m.index))
    both = pd.DataFrame(cols).dropna().astype(int)
    both.columns = ["ref", "above"]
    both["share_ref"], both["share_above"] = shares[TARGET_SIZE].reindex(both.index), shares[reference].reindex(both.index)
    both["runs_ref"], both["runs_above"] = runs[TARGET_SIZE].reindex(both.index), runs[reference].reindex(both.index)
    both["benchmark"] = [benchmark_family(t) for t in both.index]
    both["language"] = [assign_language(t) for t in both.index]
    return both, fams


def status(both: pd.DataFrame) -> pd.Series:
    return pd.Series([{(1, 1): "above at both", (0, 1): "newly above", (1, 0): "lost", (0, 0): "at chance at both"}[k]
                      for k in zip(both["ref"], both["above"])], index=both.index)


def by_benchmark(both: pd.DataFrame) -> pd.DataFrame:
    """Per benchmark: its tasks with a chance level at both rungs, how many the
    gate admits at each, the share that is, and the crossings."""
    st = status(both)
    g = both.assign(gained=st == "newly above", lost=st == "lost").groupby("benchmark")
    out = g.agg(n_tasks=("ref", "size"), above_ref=("ref", "sum"), above=("above", "sum"),
                gained=("gained", "sum"), lost=("lost", "sum")).reset_index()
    out["share_ref"], out["share_above"] = out["above_ref"] / out["n_tasks"], out["above"] / out["n_tasks"]
    out["delta"] = out["share_above"] - out["share_ref"]
    return out.sort_values(["delta", "share_above", "n_tasks"], ascending=[False, False, False]).reset_index(drop=True)


def dumbbell(ax, b: pd.DataFrame, reference: str, *, label_n: bool = True, paper: bool = False) -> None:
    """One row per benchmark: the share of its tasks above chance at the
    reference (open) and one rung up (filled), joined; the largest gain on top."""
    b = b.iloc[::-1].reset_index(drop=True)
    y = range(len(b))
    ax.hlines(y, b["share_ref"], b["share_above"], color=S.MUTED, lw=1)
    ax.scatter(b["share_ref"], y, s=22, facecolor=S.SURFACE, edgecolor=S.INK, lw=1, zorder=3, label=TARGET_SIZE)
    ax.scatter(b["share_above"], y, s=22, color=S.SERIES[0], zorder=3, label=reference)
    names = b["benchmark"].map(G.paper_name if paper else G.display)
    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=6.5 if paper else 6)
    if label_n:
        for i, r in b.iterrows():
            ax.text(1.02, i, f"{int(r['above_ref'])}→{int(r['above'])} of {int(r['n_tasks'])}", va="center",
                    fontsize=5.5, color=S.MUTED, transform=ax.get_yaxis_transform())
    ax.set_xlim(-.02, 1.02); ax.set_ylim(-.7, len(b) - .3)
    ax.grid(color=S.GRID, lw=.6, axis="x"); S.clean(ax)


def figure_by_benchmark(both: pd.DataFrame, path: Path, reference: str, fams: list) -> pd.DataFrame:
    b = by_benchmark(both)
    fig, ax = mpl.pyplot.subplots(figsize=(7.5, max(3.5, .17 * len(b) + 1.4)))
    dumbbell(ax, b, reference)
    ax.set_xlabel(f"share of the benchmark's tasks above chance (gate: ≥ {MIN_SHARE:.0%} of the runs, Wilson 95 %)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    top = G._header(fig, f"Which benchmarks the {reference} rung lifts above chance",
                    f"Per benchmark, the share of its benchmark-language tasks the above-random gate (rule 1) admits at {TARGET_SIZE} "
                    f"(open) and at {reference} (filled), on the same {len(fams)} families at both rungs ({', '.join(fams)}), each "
                    f"(family, task) scored at both; {len(both)} tasks over {len(b)} benchmarks. The text gives the tasks "
                    f"admitted at {TARGET_SIZE} → {reference} of the benchmark's tasks. Sorted by the gain.")
    fig.tight_layout(rect=(0, 0, 1, top))
    b.to_csv(path.with_suffix(".csv"), index=False)
    S.save(fig, path, dpi=150)
    return b


def figure(both: pd.DataFrame, path: Path, pool: str, reference: str, fams: list) -> None:
    fig, axes = mpl.pyplot.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.5, 1.5]})
    ax_a, ax_b, ax_c = axes
    tables = []

    # (a) the 2x2 itself, as counts
    ct = (pd.crosstab(both["ref"], both["above"])
            .reindex(index=[1, 0], columns=[0, 1], fill_value=0)
            .rename(index={1: f"above at {TARGET_SIZE}", 0: f"at chance at {TARGET_SIZE}"},
                    columns={1: f"above at {reference}", 0: f"at chance at {reference}"}))
    tables.append(G.matrix_ax(ax_a, ct / len(both), f"Where the {TARGET_SIZE} and {reference} gates disagree",
                              cnt=ct, fmt="{:.0%}", xlabel=f"the {reference} rung", ylabel=f"the {TARGET_SIZE} reference"))

    # (b)/(c) the crossings that matter: at chance at the reference, above it one rung up
    new = both[(both["ref"] == 0) & (both["above"] == 1)]
    lost = both[(both["ref"] == 1) & (both["above"] == 0)]
    for ax, col, lab in ((ax_b, "benchmark", "benchmark"), (ax_c, "language", "language")):
        gained = new[col].value_counts()
        net = gained.subtract(lost[col].value_counts(), fill_value=0)
        net = net[net != 0]
        # rank_ax splits its palette at k and colours the tail as the "worst"
        # end. Every crossing here is a gain, so widen k until the series fits
        # on one side and the figure stops implying a distinction it has not got.
        tables.append(G.rank_ax(ax, net, f"Tasks gained at {reference}, net of those lost, per {lab}",
                                k=max(10, (len(net) + 1) // 2),
                                xlabel=f"tasks above chance at {reference} but not at {TARGET_SIZE} (net)", fmt="{:+.0f}"))

    top = G._header(fig, f"What the {reference} rung can measure that the {TARGET_SIZE} reference cannot",
                    f"The above-random gate (rule 1) is a property of (task, size): a one-sided 95 % Wilson lower bound over the task's "
                    f"items clears chance for at least half the runs at that size. Both columns are recomputed on the SAME "
                    f"{len(fams)} families that have a {reference} final ({', '.join(fams)}), so the rung is the only thing that differs — "
                    f"the committed mask's {TARGET_SIZE} column pools every {TARGET_SIZE} run and would confound the rung with the run count. "
                    f"{len(both)} tasks have a chance level at both rungs; BPB and the generative tasks have none and are absent. "
                    f"(b) and (c) count tasks that cross INTO the gate at {reference}, net of those that drop out.")
    fig.tight_layout(rect=(0, 0, 1, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables)[["panel", "row", "col", "value"]].to_csv(path.with_suffix(".csv"), index=False)
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, reference: str, both: pd.DataFrame, fams: list, rel: str, gh: str, b: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    new = both[(both["ref"] == 0) & (both["above"] == 1)]
    lost = both[(both["ref"] == 1) & (both["above"] == 0)]
    n_ref, n_above = int(both["ref"].sum()), int(both["above"].sum())
    counts = md_table(["", f"at chance at {reference}", f"above chance at {reference}"],
                      [[f"**above chance at {TARGET_SIZE}**", int(((both["ref"] == 1) & (both["above"] == 0)).sum()),
                        int(((both["ref"] == 1) & (both["above"] == 1)).sum())],
                       [f"**at chance at {TARGET_SIZE}**", int(((both["ref"] == 0) & (both["above"] == 0)).sum()), len(new)]])
    per_bench = md_table(["benchmark", "gained", "lost", "net"],
                         [[b, int((new["benchmark"] == b).sum()), int((lost["benchmark"] == b).sum()),
                           f"{int((new['benchmark'] == b).sum()) - int((lost['benchmark'] == b).sum()):+d}"]
                          for b in new["benchmark"].value_counts().index[:12]])
    body = "\n\n".join([
        f"## What the {reference} rung can measure that the {TARGET_SIZE} reference cannot",
        f"**The above-random gate at two rungs · the {len(fams)} families with a {reference} final · both columns recomputed on those "
        f"families (rule 1) so the rung is the only difference.** Regenerate with "
        f"`python analysis/rq10_size_generalisation/gate_crossover.py --pool {pool}`.",
        f"**Population.** {len(both)} tasks carry a chance level at both rungs; BPB and the generative tasks have none, so they never "
        f"enter the gate. The gate admits **{n_ref}** of them at {TARGET_SIZE} and **{n_above}** at {reference} "
        f"(**{n_above - n_ref:+d}**, {(n_above / n_ref - 1):+.0%}).",
        f"![Gate crossover]({rel}/gate_crossover.png)",
        counts,
        f"**{len(new)} tasks cross into the gate at {reference}** and **{len(lost)}** drop out of it. Where they come from:",
        per_bench,
        f"Files: [`gate_crossover.png`]({gh}/gate_crossover.png), [`gate_crossover.csv`]({gh}/gate_crossover.csv), "
        f"[`gate_crossover_per_task.csv`]({gh}/gate_crossover_per_task.csv) (every task's run shares and verdicts, "
        f"`status` = `newly above` for the tasks that cross in).",
        f"### The share of each benchmark above chance at {TARGET_SIZE} and at {reference}",
        f"Per benchmark, the share of its tasks the gate admits at each rung, on the same (family, task) cells; "
        f"{len(b)} benchmarks, {int((b['delta'] > 0).sum())} gain, {int((b['delta'] < 0).sum())} lose, "
        f"{int(((b['above_ref'] == 0) & (b['above'] > 0)).sum())} have no task above chance at {TARGET_SIZE} and at least one at {reference}.",
        f"![Gate share by benchmark]({rel}/gate_crossover_by_benchmark.png)",
        md_table(["benchmark", "tasks", f"above at {TARGET_SIZE}", f"above at {reference}", "share change"],
                 [[r["benchmark"], int(r["n_tasks"]), f"{int(r['above_ref'])} ({r['share_ref']:.0%})",
                   f"{int(r['above'])} ({r['share_above']:.0%})", f"{r['delta']:+.0%}"]
                  for _, r in b[b["delta"] != 0].iterrows()]),
        f"Files: [`gate_crossover_by_benchmark.png`]({gh}/gate_crossover_by_benchmark.png), "
        f"[`gate_crossover_by_benchmark.csv`]({gh}/gate_crossover_by_benchmark.csv)."])
    replace_block(OUT_ROOT / "README.md", KEY, body, f"gate_crossover.py --pool {pool}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL, help="the gate's pool and the output folder")
    ap.add_argument("--reference", default=REFERENCE, help="the rung read ABOVE the reference (default: %(default)s)")
    args = ap.parse_args()
    stage = load_pools()[args.pool].get("stage", "pretraining")
    out = OUT_ROOT / stage / args.pool
    both, fams = crossover(args.pool, args.reference)
    figure(both, out / "gate_crossover.png", args.pool, args.reference, fams)
    b = figure_by_benchmark(both, out / "gate_crossover_by_benchmark.png", args.reference, fams)
    both.assign(status=status(both)).rename_axis("task").reset_index() \
        .rename(columns={"ref": f"above_{TARGET_SIZE}", "above": f"above_{args.reference}",
                         "share_ref": f"share_runs_{TARGET_SIZE}", "share_above": f"share_runs_{args.reference}",
                         "runs_ref": f"runs_{TARGET_SIZE}", "runs_above": f"runs_{args.reference}"}) \
        .to_csv(out / "gate_crossover_per_task.csv", index=False)
    rel = f"{stage}/{args.pool}"
    gh = ("https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/"
          f"rq10_size_generalisation/{rel}")
    generate_readme(args.pool, args.reference, both, fams, rel, gh, b)
    print(f"--- gate_crossover: {len(both)} tasks gated at both rungs, {len(fams)} families ---")
    print(pd.crosstab(both["ref"], both["above"]).to_string())
    print(b[b["delta"] != 0].head(15).to_string(index=False))
