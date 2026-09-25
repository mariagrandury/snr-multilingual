"""rq00 per benchmark and per language: the gate as cell grids, and the score
curves of every language.

    highlights.png / .csv                   rq00 on one page
    gate_margin.csv                         the cells behind the two gate_margin figures
    gate_margin_by_benchmark.png            score minus chance, language x size, one subplot per benchmark
    gate_margin_by_language.png             score minus chance, benchmark x size, one subplot per language
    first_size_above_random.png / .csv      language x benchmark: smallest size from which the score stays above chance
    score_curves.csv                        the curves below
    score_curves/<language>.png             score vs training tokens (Chinchilla multiples, 5C = the full run),
                                            one line per size, one subplot per benchmark

The margin reads `above_random_scores.csv` (the gate's bucket means: a
benchmark averaged over the cells that trained its language). The curves
average, per size, the cells that train the language, on the shared
checkpoint grid; the dotted line is chance.

    python analysis/rq00_gate_and_curves/panels.py --pool predictivity
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

from evals.scripts.utils.configs import bucket_order, load_pools  # noqa: E402
from pretrain.ladder_report import _trained_tasks  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, replace_block  # noqa: E402
from analysis.paths import GATE_AND_CURVES  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import ALPHA, MIN_SHARE, task_chance, wilson_lcb  # noqa: E402
from analysis.utils import LADDER_SIZES, ladder_frame  # noqa: E402

OUT_ROOT = GATE_AND_CURVES
mpl.rcParams.update(S.RC)


def gate_panels(out_dir: Path) -> None:
    scores = pd.read_csv(out_dir / "above_random_scores.csv")
    sizes = [b for b in bucket_order() if b in scores.columns]
    long = scores.melt(id_vars=["task", "family", "language", "random_baseline"], value_vars=sizes,
                       var_name="size", value_name="score").dropna(subset=["score", "random_baseline"])
    long = long[~long["language"].isin(["??", "multi"])]
    long["margin"] = long["score"] - long["random_baseline"]
    mask = pd.read_csv(out_dir / "above_random_mask.csv").melt(id_vars=["task"], value_vars=sizes, var_name="size", value_name="above")
    long = long.merge(mask, on=["task", "size"], how="left")
    gate = (f"the gate keeps a cell when at least {MIN_SHARE:.0%} of the size's runs are confidently above chance "
            f"(one-sided {1 - ALPHA / 2:.0%} Wilson lower bound over the task's items > chance)")
    note = ("cell = mean final-checkpoint score of the size's models that trained the language (every model of the size where none did), minus chance (1 / number of "
            f"options); {gate}")
    kw = dict(value="margin", vmin=-0.1, vmax=0.3, center=0.0, cmap=S.DIV, fmt="{:+.2f}", note=note,
              cbar="score − chance (neutral colour = chance)")
    G.panel_grid(long, out_dir / "gate_margin_by_benchmark.png", by="family", row="size", row_order=sizes,
                 col="language", ncols=1, cell_w=0.3, counts=False, xlabel="language", ylabel="model size",
                 title="Margin above chance per benchmark", **kw)
    G.panel_grid(long, out_dir / "gate_margin_by_language.png", by="language", row="family", ylabel="benchmark",
                 col="size", col_order=sizes, xlabel="model size", ncols=6, csv=False,
                 title="Margin above chance per language", **kw)
    ok = long.groupby(["family", "language", "size"])["above"].mean().unstack("size").reindex(columns=sizes)
    ok = ok.ge(MIN_SHARE).where(ok.notna())
    level = G.smallest_safe(ok).rename("level").rename_axis(["family", "language"]).reset_index() \
             .pivot(index="family", columns="language", values="level")
    G.level_heatmap(level, out_dir / "first_size_above_random.png", levels=sizes, cbar="smallest size above chance",
                    title="Smallest model size from which the score stays above chance",
                    note=f"cell = smallest size that clears the gate ({gate}), and so does every larger size with a value")
    highlights(out_dir, long, level, sizes, gate)
    threshold_panels(out_dir)


def threshold_panels(out_dir: Path) -> None:
    """above_random_thresholds.png: what the Wilson gate asks of each benchmark
    and how many (task, size) cells it keeps, against fixed margins."""
    runs = pd.read_csv(out_dir / "above_random_runs.csv")
    runs = runs[runs["trained"]] if "trained" in runs else runs
    scores = pd.read_csv(out_dir / "above_random_scores.csv")
    sizes = [b for b in bucket_order() if b in scores.columns]
    scores = scores[scores["language"].isin(["??", "multi"]) == False]
    runs = runs.merge(scores[["task", "family", "random_baseline"]], on="task")
    # the smallest accuracy whose lower bound clears chance, per task: the margin the test implies
    tasks = runs.drop_duplicates("task").set_index("task")
    thr = {}
    for t, r in tasks.iterrows():
        n = int(r["n_items"]); ks = np.arange(0, n + 1)
        ok = wilson_lcb(ks / n, np.full(n + 1, n)) > r["random_baseline"]
        thr[t] = ks[ok][0] / n - r["random_baseline"] if ok.any() else np.nan
    tasks["threshold"] = pd.Series(thr)
    fam = G.panel_order(tasks["family"].unique())
    # cells above random under each rule
    cell = runs.groupby(["task", "bucket"]).agg(share=("above", "mean"), lcb=("lcb", "median"), n=("score", "size")).reset_index()
    cell = cell.merge(scores.melt(id_vars=["task", "random_baseline", "family"], value_vars=sizes, var_name="bucket", value_name="score"),
                      on=["task", "bucket"])
    cell["margin"] = cell["score"] - cell["random_baseline"]
    cell["share_002"] = runs.assign(a=(runs["lcb"] > runs["random_baseline"] + 0.02).astype(float)).groupby(["task", "bucket"])["a"].mean() \
                            .reindex(pd.MultiIndex.from_frame(cell[["task", "bucket"]])).to_numpy()
    rules = {"Wilson LCB > chance": cell["share"] >= MIN_SHARE, "Wilson LCB > chance + 0.02": cell["share_002"] >= MIN_SHARE,
             "mean > chance + 0.01": cell["margin"] > 0.01, "mean > chance + 0.03": cell["margin"] > 0.03,
             "mean > chance + 0.05": cell["margin"] > 0.05}
    counts = pd.DataFrame({k: v.groupby(cell["family"]).sum() for k, v in rules.items()}).reindex(fam)
    total = cell.groupby("family").size().reindex(fam)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [1, 1.5]})
    med = tasks.groupby("family")["threshold"].median().reindex(fam)
    ni = tasks.groupby("family")["n_items"].median().reindex(fam)
    tables = [G.rank_ax(axes[0], med.rename(index=lambda f: f"{f}  (n = {int(ni[f]):,})"),
                        "Margin over chance the Wilson gate implies per benchmark (median task)", k=len(fam),
                        xlabel="smallest (accuracy − chance) with LCB > chance", fmt="{:+.3f}")]
    tables.append(G.matrix_ax(axes[1], counts.div(total, axis=0), "Share of a benchmark's (task, size) cells above random, per rule",
                              cnt=counts, xlabel="rule", ylabel="benchmark"))
    for lab in axes[1].get_xticklabels():
        lab.set_rotation(20); lab.set_ha("right")
    G.save_highlights(fig, out_dir, "What the Wilson gate asks, and what each rule keeps",
                      f"left: per task the smallest accuracy whose Wilson {1 - ALPHA:.0%} lower bound over its n items clears chance, "
                      "minus chance (median over the benchmark's tasks; n = median items); right: (task, size) cells above random "
                      f"under the gate (≥ {MIN_SHARE:.0%} of the size's runs), the gate with a +0.02 floor, and fixed margins on the "
                      "mean score; small number = cells; runs of the size that trained the language", tables,
                      name="above_random_thresholds")


def highlights(out_dir: Path, long: pd.DataFrame, level: pd.DataFrame, sizes: list, gate: str) -> None:
    """rq00 on one page: which benchmarks clear chance, at which size, and in how many languages."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.3]})
    fam = G.panel_order(long["family"].unique())
    share = long.groupby(["family", "size"])["above"].mean().unstack().reindex(index=fam, columns=sizes)
    n = long.groupby(["family", "size"])["task"].nunique().unstack().reindex(index=fam, columns=sizes)
    margin = long.groupby(["family", "size"])["margin"].median().unstack().reindex(index=fam, columns=sizes)
    tables = [G.matrix_ax(axes[0], share, "Share of a benchmark's languages above the gate", cnt=n, xlabel="model size"),
              G.matrix_ax(axes[1], margin, "Median margin above chance", vmin=-0.1, vmax=0.3, center=0.0, cmap=S.DIV,
                          fmt="{:+.2f}", xlabel="model size"),
              G.stack_ax(axes[2], level, "Smallest size that stays above chance, share of languages", levels=sizes)]
    G.save_highlights(fig, out_dir, "rq00 in one figure: which benchmarks carry information, and from which size?",
                      f"{gate}; small number = language tasks behind the cell; "
                      "right: per benchmark, how its languages split by the smallest size from which the gate holds", tables)


def score_curves(pool: str, out_dir: Path) -> int:
    df = ladder_frame(pool)
    df = df[df["kind"] == "benchmark"]
    df = G.add_meta(df[[t in _trained_tasks(L, s) for t, L, s in zip(df["task"], df["L"], df["scheme"])]])
    df["chance"] = df["task"].map(task_chance)
    df["chinchilla"] = df["frac"] * G.CHINCHILLA_AT_FULL
    curve_dir = out_dir / "score_curves"
    curves = df.groupby(["language", "family", "size", "chinchilla"]).agg(
        score=("primary_score", "mean"), chance=("chance", "mean"), tasks=("task", "nunique"), models=("model", "nunique")).reset_index()
    curves.to_csv(out_dir / "score_curves.csv", index=False)
    n = 0
    for lang, g in curves.groupby("language"):
        fams = G.panel_order(g["family"].unique())
        cols = min(5, len(fams)); rows = (len(fams) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.4 * rows + 0.5), squeeze=False)
        flat = [a for r in axes for a in r]
        for ax in flat[len(fams):]:
            ax.axis("off")
        for ax, fam in zip(flat, fams):
            gf = g[g["family"] == fam]
            for size in [s for s in LADDER_SIZES if s in set(gf["size"])]:
                m = gf[gf["size"] == size].sort_values("chinchilla")
                ax.plot(m["chinchilla"], m["score"], color=S.SIZE_COLOR.get(size, S.MUTED), lw=1.4, marker="o", ms=2.5,
                        label=size)
            if gf["chance"].notna().any():
                ax.axhline(gf["chance"].mean(), color=S.ALERT, lw=.9, ls=":")
            nt = int(gf["tasks"].max())
            ax.set_title(f"{fam} ({nt} task{'s' if nt > 1 else ''})", loc="left", fontsize=8)
            ax.set_xlabel("training tokens (× Chinchilla)"); ax.set_ylabel("score"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
        handles = [plt.Line2D([], [], color=S.SIZE_COLOR[s], lw=2, label=s) for s in LADDER_SIZES if s in set(g["size"])]
        fig.legend(handles=handles, ncol=len(handles), loc="lower center", frameon=False)
        fig.suptitle(f"{lang}: benchmark score along the run, mean over the cells that train it "
                     "(colour = size, dotted = chance; 5C = the full run)", x=0.01, ha="left", y=1.0, fontsize=10)
        fig.tight_layout(rect=(0, 0.04, 1, 0.98))
        curve_dir.mkdir(parents=True, exist_ok=True)
        S.save(fig, curve_dir / f"{lang}.png", dpi=110)
        n += 1
    print(f"Wrote {n} per-language score-curve figures → {curve_dir}")
    return n


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    gate_panels(out_dir)
    n = score_curves(pool, out_dir)
    if pool != CANONICAL_POOL:
        return
    rel = f"{stage}/{pool}"
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"The gate and the curves without the aggregation (`{pool}` pool). Regenerate with "
        f"`python analysis/rq00_gate_and_curves/panels.py --pool {pool}`. In every grid white is \"no value\" and grey "
        "\"filtered out by the gate\"; each figure's table sits next to it under the same name.",
        f"![rq00 in one figure]({rel}/highlights.png)",
        f"![Smallest size above chance]({rel}/first_size_above_random.png)",
        f"![Margin above chance per benchmark]({rel}/gate_margin_by_benchmark.png)",
        f"![Margin above chance per language]({rel}/gate_margin_by_language.png)",
        f"![What the gate asks, and what each rule keeps]({rel}/above_random_thresholds.png)",
        f"Score along the run, one figure per language ({n} languages, one subplot per benchmark, one line per "
        f"size): `{rel}/score_curves/<language>.png`, e.g.",
        f"![Score curves, German]({rel}/score_curves/de.png)"])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    main(p.parse_args().pool)
