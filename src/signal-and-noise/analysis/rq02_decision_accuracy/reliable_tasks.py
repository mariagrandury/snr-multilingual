"""RQ2 — Which (benchmark, language) cells rank reliably at all, and in which
languages are there any?

rq02's tables answer "how well does task t rank" one size or one checkpoint at
a time. To *select* tasks — for a figure, or for the paper's claim that some
benchmarks carry a usable signal in some languages — the two axes have to be
reduced to one number each:

    DA-size(t)   agreement of a proxy size's final ranking with the reference's
                 (`decision_acc_size_<s>`)
    DA-ckpt(t)   agreement of an earlier checkpoint with the same size's final
                 (`decision_acc_ckpt_f<f>_<s>`)

Four reductions, because they disagree by an order of magnitude about how many
tasks clear a cut:

    late     ONE fixed cell per axis — DA-size at the largest proxy (the last
             rung below the reference) and DA-ckpt at the last early checkpoint
             (90 %) of that same size. No cell is chosen by its value, so the
             reduction adds no selection of its own — the cut on that cell's
             DA still selects tasks by the quantity the figures then draw, so
             every filtered figure is a conditional reading. The default.
    mean     the average cell — "does this task rank reliably across the ladder".
    median   the typical cell; between `mean` and `max`.
    max      the best cell — "is there ANY size/checkpoint where it ranks
             reliably". Reported for contrast, not to be used as a filter: a task
             has ~4 DA-size cells but ~45 DA-ckpt cells, so the max over the
             checkpoint axis clears 0.8 for almost every task and the `both` test
             silently collapses to DA-size alone.

A cell counts only with >= MIN_PAIRS pairs (rule 5) and only where the
above-random gate keeps it (rule 1): a benchmark at chance has no ranking to
agree with, so it can never be reliable. BPB and the loss have no chance level
and are excluded — this script is about benchmarks (rule 7 drops the language
aggregates).

ONE table, many views. The per-task DA values do not depend on the cut, so they
are written once and every threshold is a filter over them rather than a file of
its own (the alternative was 12 near-identical CSVs):

    da_reliable_tasks.csv        per task: language, benchmark, and da_{size,ckpt}_<reduction>
                                 for every reduction. The pass flags are NOT stored —
                                 they are `da_size_<red> >= t and da_ckpt_<red> >= t`.
    da_reliable_by_language.csv  long: one row per (threshold, reduction, language)
                                 with how many benchmarks pass on DA-size, on
                                 DA-ckpt, on either and on both.
    da_reliable_tasks_<t>_<red>.png   one per (THRESHOLDS x REDUCTIONS): left, per
                                 language, benchmarks passing both / size only /
                                 ckpt only; right, the benchmark x language grid of
                                 which test each cell passes.

`FILTERS` is the one registry of what a variant NAME means — its reduction, its
cut and which test it applies. `by_L.py`, `scale_convergence.py` and
`paper_rq2.py` all resolve names through it, so a figure called `above_66_both`
means the same thing wherever it is drawn, and a new cut is one line here rather
than a parser in three modules.

    python analysis/rq02_decision_accuracy/reliable_tasks.py --pool predictivity
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
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq02_decision_accuracy.cross_task import resource_order  # noqa: E402
from analysis.utils import (  # noqa: E402
    CKPT_DA_EARLY_FRACS, MIN_PAIRS, PAIR_AXES, SMALL_SIZES, TARGET_SIZE, assign_language,
    benchmark_family)

OUT_ROOT = DECISION_ACCURACY
THRESHOLDS = (0.80, 0.75, 0.66)        # the cuts drawn; the first is the default
THRESH = THRESHOLDS[0]
DEFAULT_AXES = PAIR_AXES[0]            # "multi-axis": what a caller that does not ask gets
KINDS = ("size", "ckpt", "goal")       # the three references a DA column can rank against
# The DA table this reads by default. `predictivity_schemes` is every data
# scheme at the grid seed — the population `by_L` and `scale_convergence`
# actually pair over — while the gate and the outputs stay with the headline
# pool, whose A/B filter exists to protect an SNR *signal* and has no business
# selecting decisions. Reading `predictivity`'s table instead made a cell
# "reliable" on A/B decisions and then used that verdict to filter all-scheme
# ones (plan/decision_accuracy.md §3.2).
DA_POOL = "predictivity_schemes"
REDUCTIONS = ("late", "mean", "median", "max")
DEFAULT_REDUCTION = "late"             # what the `above_80` filters use
LATE_SIZE = SMALL_SIZES[-1]            # the largest proxy: the rung just below the reference
LATE_FRAC = CKPT_DA_EARLY_FRACS[-1]    # the last checkpoint before the final (90 %)
LATE_COL = {"size": f"decision_acc_size_{LATE_SIZE}",
            "ckpt": f"decision_acc_ckpt_f{int(round(LATE_FRAC * 100))}_{LATE_SIZE}",
            "goal": f"decision_acc_goal_f{int(round(LATE_FRAC * 100))}_{LATE_SIZE}"}
# variant suffix -> (reduction, threshold, which test a task must pass).
# `size` / `ckpt` keep the tasks reliable on that ONE axis, which is what a
# per-panel filter wants: the DA-size panel should be read over the tasks whose
# DA-size is trustworthy, not over the tasks that also happen to pass DA-ckpt.
FILTERS = {
    "above_80":        ("late",   0.80, "both"),
    "above_66_both":   ("median", 0.66, "both"),
    "above_66_size":   ("median", 0.66, "size"),
    "above_66_ckpt":   ("median", 0.66, "ckpt"),
    "above_66_either": ("median", 0.66, "either"),
}
mpl.rcParams.update(S.RC)


def pct(thresh: float) -> int:
    """0.66 -> 66: the suffix a threshold's figures carry."""
    return int(round(thresh * 100))


def passes(tasks: pd.DataFrame, red: str, thresh: float) -> pd.DataFrame:
    """The four pass flags at one (reduction, threshold) — derived, never stored."""
    s, c = tasks[f"da_size_{red}"] >= thresh, tasks[f"da_ckpt_{red}"] >= thresh
    return pd.DataFrame({"size": s, "ckpt": c, "both": s & c, "either": s | c}, index=tasks.index)


def load_reliable(out_dir: Path, variant: str, axes: str = DEFAULT_AXES) -> pd.DataFrame | None:
    """The (benchmark, language) cells a named variant keeps, with their DA
    values — the population that figure averages over. None when this script has
    not been run for the pool yet, so a caller can skip the variant instead of
    drawing an empty panel. `axes` selects the pair set the verdict was reached
    on, so a figure drawn over mono-axis decisions is filtered by a mono-axis
    reliability."""
    red, thresh, crit = FILTERS[variant]
    f = out_dir / "da_reliable_tasks.csv"
    if not f.exists():
        print(f"  (no {f.name}: run reliable_tasks.py first — skipping {variant})")
        return None
    d = pd.read_csv(f)
    if "axes" in d.columns:
        d = d[d["axes"] == axes]
    return d[passes(d, red, thresh)[crit]]


# Which reference each DA kind ranks against, and so which gate rule 1 asks for:
# DA-size and DA-goal rank a proxy against the REFERENCE, so the task must be
# above chance there as well; DA-ckpt ranks a proxy against its OWN final, so the
# proxy's gate is the whole gate. `early_small.py` splits them the same way
# (lines 89/90); gating all of them without the reference let three cells that
# are at chance at 1.7B into the reliable population (fixed 2026-09-22).
GATE_REF = {"size": TARGET_SIZE, "goal": TARGET_SIZE, "ckpt": None}


def long_da(out_dir: Path, pool: str, axes: str = DEFAULT_AXES) -> pd.DataFrame:
    """da_per_task.csv melted to (task, kind, level, da), with the pair
    minimum and the above-random gate applied. `kind` is `size`, `ckpt` or
    `goal`; `axes` picks the pair set (`compute_da`'s `axes` column, rule 15)
    and defaults to the multi-axis reading, so this table did not move when the
    mono-axis one was added."""
    da = pd.read_csv(out_dir / "da_per_task.csv")
    n = pd.read_csv(out_dir / "da_n_pairs_per_task.csv")
    if "axes" in da.columns:                 # a table written before rule 15 has one pair set
        da, n = da[da["axes"] == axes], n[n["axes"] == axes]
    da = da.set_index("task").drop(columns="axes", errors="ignore")
    n = n.set_index("task").drop(columns="axes", errors="ignore").reindex_like(da)
    # `decision_acc_size_<a>_to_<b>` is agreement with a NON-reference target, so it
    # is not DA-size in rule 9's sense and is left out; DA-size here is proxy -> reference.
    keep = [c for c in da.columns
            if (c.startswith("decision_acc_size_") and "_to_" not in c)
            or c.startswith(("decision_acc_ckpt_", "decision_acc_goal_"))]
    long = (da[keep].stack(future_stack=True).rename("da").reset_index()
            .rename(columns={"level_1": "col"}))
    long["n_pairs"] = n[keep].stack(future_stack=True).to_numpy()
    long["kind"] = long["col"].str.split("_").str[2]                # size | ckpt | goal
    # the size a cell is READ at: the proxy throughout (DA-size names it last too)
    long["size"] = np.where(long["kind"] == "size",
                            long["col"].str.replace("decision_acc_size_", "", regex=False),
                            long["col"].str.rsplit("_", n=1).str[-1])
    long = long[long["n_pairs"] >= MIN_PAIRS]                       # rule 5
    long = pd.concat([G.mark_gated(G.add_meta(long[long["kind"] == k]), pool, "size", "da", ref)
                      for k, ref in GATE_REF.items() if (long["kind"] == k).any()], ignore_index=True)
    return long.dropna(subset=["da"])


def per_task(long: pd.DataFrame) -> pd.DataFrame:
    """One row per benchmark task: each DA kind under each reduction.
    Threshold-free — the cuts are applied by `passes`. DA-goal is carried for
    information; the FILTERS cut on DA-size and DA-ckpt, which are the two
    independent questions (how small, how early)."""
    long = long[~long["family"].isin(["bpb", "loss"])]              # benchmarks only
    t = long.pivot_table(index="task", columns="kind", values="da", aggfunc=list)
    late = {k: long[long["col"] == c].set_index("task")["da"] for k, c in LATE_COL.items()}
    rows = []
    for task, r in t.iterrows():
        row = {"task": task, "language": assign_language(task), "benchmark": benchmark_family(task)}
        for kind in KINDS:
            v = r.get(kind)
            v = np.asarray(v, dtype=float) if isinstance(v, list) else np.array([])
            row[f"n_cells_{kind}"] = len(v)
            row[f"da_{kind}_late"] = late.get(kind, pd.Series(dtype=float)).get(task, np.nan)
            for red in ("mean", "median", "max"):
                row[f"da_{kind}_{red}"] = getattr(np, red)(v) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["language", "benchmark"]).reset_index(drop=True)


def per_language(tasks: pd.DataFrame, red: str, thresh: float) -> pd.DataFrame:
    """Per language, how many benchmarks pass each test at one (reduction, cut)."""
    p = passes(tasks, red, thresh)
    d = tasks[["language", "benchmark"]].join(p)
    out = (d.groupby("language").agg(**{"benchmarks evaluated": ("benchmark", "size"),
                                        "DA-size": ("size", "sum"), "DA-ckpt": ("ckpt", "sum"),
                                        "either": ("either", "sum"), "both": ("both", "sum")}).reset_index())
    out = out.set_index("language").reindex(resource_order(out["language"])).reset_index()
    return out.assign(threshold=thresh, reduction=red)


def figure(tasks: pd.DataFrame, path: Path, red: str, thresh: float) -> pd.DataFrame:
    by_lang = per_language(tasks, red, thresh)
    langs = list(by_lang["language"])
    both = by_lang["both"].to_numpy()
    s_only = by_lang["DA-size"].to_numpy() - both
    c_only = by_lang["DA-ckpt"].to_numpy() - both
    none = by_lang["benchmarks evaluated"].to_numpy() - both - s_only - c_only
    fig, axes = plt.subplots(1, 2, figsize=(14.5, max(4.5, .28 * len(langs) + 2)),
                             gridspec_kw={"width_ratios": [1, 1.25]})
    y = np.arange(len(langs))
    left = np.zeros(len(langs))
    for vals, c, lab in ((both, S.RAMP[3], f"both (DA-size and DA-ckpt ≥ {thresh:g})"),
                         (s_only, S.RAMP[1], "DA-size only"), (c_only, S.SERIES[1], "DA-ckpt only"),
                         (none, S.GRID, "neither")):
        axes[0].barh(y, vals, left=left, color=c, label=lab, height=.72)
        left = left + vals
    axes[0].set_yticks(y); axes[0].set_yticklabels(langs, fontsize=7); axes[0].invert_yaxis()
    axes[0].set_xlabel("benchmarks evaluated in that language")
    axes[0].set_title(f"how many benchmarks rank reliably ({red} over the cells)", loc="left", fontsize=8.5)
    axes[0].legend(fontsize=6.5, frameon=False, loc="lower right")
    axes[0].grid(color=S.GRID, lw=.6, axis="x"); S.clean(axes[0])

    p = passes(tasks, red, thresh)
    code = p["size"].astype(int) + 2 * p["ckpt"].astype(int)        # 0 none 1 size 2 ckpt 3 both
    grid = (tasks.assign(code=code).pivot_table(index="benchmark", columns="language", values="code", aggfunc="max")
            .reindex(columns=[l for l in langs if l in set(tasks["language"])]))
    cmap = mpl.colors.ListedColormap([S.GRID, S.RAMP[1], S.SERIES[1], S.RAMP[3]]); cmap.set_bad(S.SURFACE)
    axes[1].imshow(np.ma.masked_invalid(grid.to_numpy(dtype=float)), cmap=cmap,
                   norm=mpl.colors.BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4), aspect="auto", interpolation="nearest")
    axes[1].set_xticks(range(grid.shape[1])); axes[1].set_xticklabels(grid.columns, fontsize=6, rotation=90)
    axes[1].set_yticks(range(grid.shape[0])); axes[1].set_yticklabels(grid.index, fontsize=6)
    axes[1].set_xlabel("language (resource order)"); axes[1].set_ylabel("benchmark")
    axes[1].set_title("per cell: which test it passes (white = not evaluated)", loc="left", fontsize=8.5)
    S.clean(axes[1], spines=()); axes[1].tick_params(length=0)
    top = G._header(fig, f"Which (benchmark, language) cells rank reliably: DA ≥ {thresh:g} on both axes",
                    f"DA-size = agreement of a proxy size's final ranking with the reference's; DA-ckpt = agreement of an "
                    f"earlier checkpoint with the same size's final. Each is the {red} over that task's cells with "
                    f"≥ {MIN_PAIRS} pairs (rule 5) that the above-random gate keeps (rule 1); a benchmark at chance can "
                    f"never pass. Benchmarks only — BPB and the loss have no chance level. Decisions from the "
                    f"`{DA_POOL}` pool over its {DEFAULT_AXES} pairs (rule 15); the gate is the figure's own pool. "
                    f"Values in `da_reliable_tasks.csv`, which carries every pair set; this figure is one "
                    f"(threshold, reduction) view of it.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)
    return by_lang


def generate_readme(pool: str, out_dir: Path, tasks: pd.DataFrame, by_lang: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    head = by_lang[(by_lang["threshold"] == THRESH) & (by_lang["reduction"] == DEFAULT_REDUCTION)]
    t = head[["language", "benchmarks evaluated", "DA-size", "DA-ckpt", "either", "both"]]
    sweep = [[f"{th:g}", red, int(p["both"].sum()), int(tasks.loc[p["both"], "language"].nunique()),
              ", ".join(sorted(tasks.loc[p["both"], "benchmark"].unique())) or "—"]
             for th in THRESHOLDS for red in REDUCTIONS for p in [passes(tasks, red, th)]]
    body = "\n\n".join([
        "## Which benchmark-language cells rank reliably",
        f"Per language, how many benchmarks clear DA ≥ {THRESH:g} on DA-size (a proxy size's final ranking vs the "
        f"reference's) and on DA-ckpt (an earlier checkpoint vs the same size's final), reducing each task's cells "
        f"with `{DEFAULT_REDUCTION}` (one fixed cell per axis, so no cell is chosen by its value). Cells need ≥ "
        f"{MIN_PAIRS} pairs (rule 5) and must survive the above-random gate (rule 1). The decisions come from the "
        f"`{DA_POOL}` pool — every data scheme at the grid seed, which is the population `by_L` and "
        f"`scale_convergence` pair over — while the gate and this folder stay with `{pool}`; the table carries one "
        f"row per pair set (rule 15) and the figures show `{DEFAULT_AXES}`. "
        f"`da_reliable_tasks.csv` holds the per-task values for every reduction and is threshold-free — each figure "
        f"is one view of it. Regenerate with `python analysis/rq02_decision_accuracy/reliable_tasks.py --pool {pool}`.",
        md_table(list(t.columns), t.values.tolist()),
        "**How many cells pass, by cut and reduction** — the cut is a choice, and this is its whole sensitivity:",
        md_table(["threshold", "reduction", "tasks passing both", "languages", "benchmarks"], sweep),
        f"![Reliable benchmark-language cells]({stage}/{pool}/da_reliable_tasks_{pct(THRESH)}_{DEFAULT_REDUCTION}.png)"])
    replace_block(OUT_ROOT / "README.md", "reliable-tasks", body, f"reliable_tasks.py --pool {pool}")


def available_axes(out_dir: Path) -> list[str]:
    """The pair sets `compute_da` wrote for this pool, in PAIR_AXES order."""
    da = pd.read_csv(out_dir / "da_per_task.csv", nrows=2000)
    got = set(da["axes"]) if "axes" in da.columns else {DEFAULT_AXES}
    return [a for a in PAIR_AXES if a in got]


def run(pool: str, out_dir: Path, da_dir: Path | None = None) -> pd.DataFrame:
    """One table for every pair set (the `axes` column), and figures for the
    default one only: the figures are a view of the table, and drawing each of
    the twelve twice would double the folder to say the same thing.

    `da_dir` is where the DA table is read from (DA_POOL's folder by default);
    `pool` still names the gate and `out_dir` the folder written to.
    """
    da_dir = da_dir or out_dir
    axes_sets = available_axes(da_dir)
    print(f"DA table: {da_dir.name} | gate: {pool} | pair sets: {', '.join(axes_sets)}")
    tasks = pd.concat([per_task(long_da(da_dir, pool, axes=a)).assign(axes=a) for a in axes_sets],
                      ignore_index=True)
    tasks = tasks[["axes"] + [c for c in tasks.columns if c != "axes"]]
    tasks.to_csv(out_dir / "da_reliable_tasks.csv", index=False)
    for a in axes_sets:
        d = tasks[tasks["axes"] == a]
        print(f"\n[{a}] {len(d)} benchmark tasks over {d['language'].nunique()} languages "
              f"({d['benchmark'].nunique()} benchmarks)")
        for thresh in THRESHOLDS:
            for red in REDUCTIONS:
                q = passes(d, red, thresh)
                print(f"  cut {thresh:g} / {red:6}: DA-size {int(q['size'].sum()):3d} | DA-ckpt {int(q['ckpt'].sum()):3d} | "
                      f"either {int(q['either'].sum()):3d} | both {int(q['both'].sum()):3d} tasks over "
                      f"{d.loc[q['both'], 'language'].nunique():2d} languages "
                      f"[{', '.join(sorted(d.loc[q['both'], 'benchmark'].unique())) or '—'}]")
    head = tasks[tasks["axes"] == axes_sets[0]].reset_index(drop=True)
    rows = [figure(head, out_dir / f"da_reliable_tasks_{pct(thresh)}_{red}.png", red, thresh)
            for thresh in THRESHOLDS for red in REDUCTIONS]
    by_lang = pd.concat(rows, ignore_index=True).assign(axes=axes_sets[0])
    by_lang.to_csv(out_dir / "da_reliable_by_language.csv", index=False)
    generate_readme(pool, out_dir, head, by_lang)
    return tasks


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose gate applies and whose folder is written")
    p.add_argument("--da-pool", default=DA_POOL,
                   help="the pool whose da_per_task.csv the decisions come from (default: %(default)s)")
    args = p.parse_args()
    stage = load_pools()[args.pool].get("stage", "pretraining")
    run(args.pool, OUT_ROOT / stage / args.pool,
        OUT_ROOT / load_pools()[args.da_pool].get("stage", "pretraining") / args.da_pool)
