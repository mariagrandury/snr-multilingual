"""Scaling regimes per benchmark-language pair: how predictable a task is
across model size and along a training run.

One point per task (a benchmark in one language, or one language's BPB),
aggregated over the deep, scheme-A, seed-1904 cells that train the task's
language (the same cells rq01's fits use):

  (a) the median R² of the log-N fit of the final score (rq1_fits.csv, one
      fit per L) against the median Spearman ρ of the same fits, ρ oriented
      so that improving with size is positive (BPB and the loss decrease);
  (b) the median R² of the log-N fit against the median R² of the
      training-trajectory fit: per (L, size) cell the run's evaluated
      checkpoints fitted as score ~ a + b log10(tokens), then the median
      over the cells. Shaded quadrants split each axis at R2_SPLIT — a
      heuristic reading, not a fitted boundary.

    scaling_regimes.png / .csv             the figure and its per-task table (medians, counts, the quadrant)
    scaling_regimes_families.png / .csv    the same, one label per benchmark family at its median point (no legend)
    scaling_regimes_outliers.png / .csv    the family labels plus the tasks that break their family's regime
                                           (another quadrant than the family's majority and > OUTLIER_DIST from
                                           its median point), named family:language
    scaling_regimes_outliers_paper.png/pdf the outliers figure for the paper: no header, panel titles or quadrant
                                           labels, square panels, the caption's axis labels
    scaling_regimes_by_family.png          panel (b) per family: its tasks labelled with the language code, the
                                           other tasks in grey behind (table: scaling_regimes_families.csv)
    scaling_regimes.html                   the two panels with hover names and a click-to-highlight legend
                                           (Vega-Lite from a CDN, for the project site; not for the paper)

    python analysis/rq01_scaling_predictability/regimes.py --pool predictivity_all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

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
from analysis.paths import SCALING_PREDICTABILITY  # noqa: E402
from analysis.rq01_scaling_predictability.analyze import _grid  # noqa: E402
from analysis.utils import _is_parent_task, assign_language, benchmark_family, ladder_frame, trained_bpb_tasks  # noqa: E402
from pretrain.ladder_report import _trained_tasks  # noqa: E402

OUT_ROOT = SCALING_PREDICTABILITY
CANONICAL = "predictivity_all"
R2_SPLIT = 0.5          # the heuristic boundary between "predictable" and "weak" on both R² axes
MIN_POINTS = 5          # checkpoints a trajectory fit needs
MIN_FITS = 2            # fits (L's, or cells) a task needs for a median
OUTLIER_DIST = 0.25     # a task in another quadrant than its family AND this far from its median point (either panel) gets named
REGIME_SHORT = {"predictable across both": "both", "predictable during training only": "training only",
                "predictable across size only": "size only", "weak / unpredictable in both": "weak"}
SHORT = {"global_piqa_parallel_cloze": "piqa", "global_mmlu_full": "gmmlu", "include_base_44": "include",
         "lambada_openai_mt": "lambada", "truthfulqa-multi_mc1": "truthfulqa"}
COLS = ["r2_size", "rho_size", "r2_trajectory"]
mpl.rcParams.update(S.RC)
QUADRANTS = {(True, True): ("predictable across both", "#b9cfe8"), (False, True): ("predictable during training only", "#cfe3c5"),
             (True, False): ("predictable across size only", "#b8ccd0"), (False, False): ("weak / unpredictable in both", "#f1d8bd")}


def trained(task: str, L: int) -> bool:
    return task in _trained_tasks(L, "A") or task in (trained_bpb_tasks(L, "A") or set()) or task in ("train_loss", "bpb_macro")


def size_medians(fits: pd.DataFrame) -> pd.DataFrame:
    """Per task: median R² and oriented median ρ of the log-N fits over the
    L's that train the task's language."""
    f = fits[[trained(t, L) for t, L in zip(fits["task"], fits["L"])]].copy()
    f["rho"] = np.where(f["kind"] == "benchmark", f["rho"], -f["rho"])
    g = f.groupby("task").agg(r2_size=("r2", "median"), rho_size=("rho", "median"), n_size_fits=("r2", "size"))
    return g[g["n_size_fits"] >= MIN_FITS]


def trajectory_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Per task: median R² of score ~ a + b log10(tokens) over the run's
    checkpoints, one fit per (L, size) grid cell that trains the language."""
    g0 = _grid(df)
    g0 = g0[g0["task"].map(_is_parent_task) & (g0["tokens"] > 0)]
    g0 = g0[[trained(t, int(L)) for t, L in zip(g0["task"], g0["L"])]]
    rows = []
    for (task, L, size), g in g0.groupby(["task", "L", "size"]):
        if len(g) < MIN_POINTS:
            continue
        x, y = np.log10(g["tokens"].to_numpy(float)), g["primary_score"].to_numpy(float)
        b, a = np.polyfit(x, y, 1)
        ss_res, ss_tot = ((y - (b * x + a)) ** 2).sum(), ((y - y.mean()) ** 2).sum()
        rows.append({"task": task, "L": int(L), "size": size, "n_points": len(g), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan})
    t = pd.DataFrame(rows).dropna(subset=["r2"])
    g = t.groupby("task").agg(r2_trajectory=("r2", "median"), n_trajectory_fits=("r2", "size"))
    return g[g["n_trajectory_fits"] >= MIN_FITS]


def short(family: str) -> str:
    return SHORT.get(family, family)


def point_label(t: pd.DataFrame) -> pd.Series:
    """The language code, or the task's suffix where a family has several
    tasks in one language (arc_challenge / arc_easy)."""
    dup = t.groupby(["family", "language"])["task"].transform("size") > 1
    return pd.Series(np.where(dup, [k.removeprefix(f + "_") for k, f in zip(t["task"], t["family"])], t["language"]), index=t.index)


def family_table(t: pd.DataFrame) -> pd.DataFrame:
    """Per family: task count, the median point of each panel and the
    majority regime (with its share)."""
    g = t.groupby("family")
    out = g[COLS].median().join(g.size().rename("n_tasks"))
    out["regime_majority"] = g["regime"].agg(lambda s: s.value_counts().index[0])
    out["share_majority"] = g["regime"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    return out.reset_index()[["family", "n_tasks"] + COLS + ["regime_majority", "share_majority"]]


def outliers(t: pd.DataFrame, fam: pd.DataFrame) -> pd.DataFrame:
    """The tasks that break their family's pattern: another quadrant than the
    family's majority and farther than OUTLIER_DIST from the family's median
    point in either panel (the distance keeps out the tasks just across the
    split line)."""
    m = t.merge(fam, on="family", suffixes=("", "_family"))
    m["dist_a"] = np.hypot(m["r2_size"] - m["r2_size_family"], m["rho_size"] - m["rho_size_family"])
    m["dist_b"] = np.hypot(m["r2_size"] - m["r2_size_family"], m["r2_trajectory"] - m["r2_trajectory_family"])
    m["other_regime"] = m["regime"] != m["regime_majority"]
    m = m[m["other_regime"] & (m[["dist_a", "dist_b"]].max(axis=1) > OUTLIER_DIST)]
    m["label"] = [f"{short(f)}:{l}" for f, l in zip(m["family"], point_label(m))]
    return m[["task", "family", "language", "label"] + COLS + ["regime", "regime_majority", "other_regime", "dist_a", "dist_b"]]


def _colours(t: pd.DataFrame) -> dict:
    fams = G.panel_order(t["family"].unique())
    return {f: (S.RAMP[3] if f == "bpb" else S.MUTED if f == "loss" else plt.cm.tab20(i % 20)) for i, f in enumerate(fams)}


def _quadrants(ax, labels: bool = True, alpha: float = .55) -> None:
    for (px, py), (label, col) in QUADRANTS.items():
        x0, x1 = (R2_SPLIT, 1.02) if px else (-0.02, R2_SPLIT)
        y0, y1 = (R2_SPLIT, 1.02) if py else (-0.02, R2_SPLIT)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=col, alpha=alpha, lw=0, zorder=0))
        if labels:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, label, ha="center", va="center", fontsize=7.5, color=S.INK, alpha=.8, zorder=1)


def _panels(t: pd.DataFrame, a, b, colours: dict, legend: bool, paper: bool = False) -> None:
    """The two panels: (a) R² vs oriented ρ, (b) R² vs trajectory R² over the
    quadrants; `paper` = no panel titles or quadrant labels, square panels,
    the paper caption's axis labels."""
    _quadrants(b, labels=not paper)
    for f in colours:
        g = t[t["family"] == f]
        kw = dict(s=11, color=colours[f], alpha=.8, lw=.3, edgecolor="white", label=f"{f} ({len(g)})", zorder=2)
        a.scatter(g["r2_size"], g["rho_size"], **kw)
        b.scatter(g["r2_size"], g["r2_trajectory"], **kw)
    xlabel = "Median R² across model-size scaling fits" if paper else "median R² across model-size fits"
    a.set_xlabel(xlabel); a.set_ylabel("Median Spearman ρ with model size" if paper else "median Spearman ρ with model size (improving = +)")
    a.set_xlim(-0.02, 1.02); a.set_ylim(-1.05, 1.05); a.axhline(0, color=S.GRID, lw=.6)
    b.set_xlabel(xlabel); b.set_ylabel("Median R² across training-trajectory fits" if paper else "median R² across training-trajectory fits")
    b.set_xlim(-0.02, 1.02); b.set_ylim(-0.02, 1.02)
    if not paper:
        a.set_title("(a) fit quality and direction across size", loc="left", fontsize=9)
        b.set_title("(b) predictability across size and along training", loc="left", fontsize=9)
    for ax in (a, b):
        ax.grid(color=S.GRID, lw=.5); S.clean(ax)
        if paper:
            ax.set_box_aspect(1)
    if legend:
        b.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="benchmark (tasks)", title_fontsize=6.5)


def _labels(ax, xs, ys, names, colours, *, fontsize, weight="normal", avoid=None) -> None:
    """Texts at the points, pushed apart by adjustText, a thin leader back to
    the point; `avoid` = (x, y) of the other points to keep clear of."""
    texts = [ax.text(x, y, n, fontsize=fontsize, color=c, weight=weight, zorder=4, ha="center", va="center")
             for x, y, n, c in zip(xs, ys, names, colours)]
    ax_, ay_ = (avoid if avoid is not None else (xs, ys))
    adjust_text(texts, x=np.asarray(ax_, float), y=np.asarray(ay_, float), ax=ax, expand=(1.3, 1.6),
                arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=.4, alpha=.7), min_arrow_len=3)


NOTE = ("point = one task, medians over the deep scheme-A seed-1904 cells that train its language: (a) R² and Spearman ρ of "
        "the final score ~ log N fit, one fit per L (ρ of BPB and the loss negated so improving is positive); (b) the same R² "
        "against the R² of score ~ log tokens over each run's checkpoints, one fit per (L, size); shaded quadrants split at "
        f"R² = {R2_SPLIT}, a heuristic reading")


def figure(t: pd.DataFrame, out_dir: Path, fam: pd.DataFrame | None = None, out: pd.DataFrame | None = None, name: str = "scaling_regimes",
           paper: bool = False) -> None:
    """The two panels; with `fam` one label per family at its median point
    instead of the legend, with `out` also the named outliers; `paper` = the
    bare version for the paper (PNG + PDF), see _panels."""
    colours = _colours(t)
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 4.6) if paper else (10.4, 4.3))
    _panels(t, a, b, colours, legend=fam is None, paper=paper)
    note = NOTE
    if fam is not None:
        names = [f"{short(f)} ({n})" for f, n in zip(fam["family"], fam["n_tasks"])]
        cols = [colours[f] for f in fam["family"]]
        a.set_ylim(-1.05, 1.5); a.set_yticks(np.arange(-1, 1.01, .5))    # headroom for the labels of the families at ρ = 1
        for ax, y in ((a, "rho_size"), (b, "r2_trajectory")):
            ax.scatter(fam["r2_size"], fam[y], s=26, color=cols, lw=.8, edgecolor=S.INK, zorder=3)
            _labels(ax, fam["r2_size"], fam[y], names, cols, fontsize=6.5, weight="bold", avoid=(t["r2_size"], t[y]))
        note += "; label = the family at its median point (tasks)"
    if out is not None:
        cols = [colours[f] for f in out["family"]]
        for ax, y in ((a, "rho_size"), (b, "r2_trajectory")):
            _labels(ax, out["r2_size"], out[y], out["label"], cols, fontsize=5, avoid=(t["r2_size"], t[y]))
        note += (f"; small labels = tasks in another quadrant than their family's majority and > {OUTLIER_DIST} from its "
                 f"median point in either panel")
    if paper:
        fig.tight_layout()
        S.save_figure(fig, out_dir, name)
        return
    top = G._header(fig, "Benchmark scaling predictability per benchmark-language pair", note)
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, out_dir / f"{name}.png"); plt.close(fig)


def figure_by_family(t: pd.DataFrame, fam: pd.DataFrame, out_dir: Path) -> None:
    """Panel (b) once per family, its tasks labelled with the language code,
    every other task in grey behind."""
    colours = _colours(t)
    labels = point_label(t)
    n = len(colours); ncol = 4; nrow = -(-n // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.9 * ncol, 2.8 * nrow + 0.6), sharex=True, sharey=True)
    for ax, f in zip(axes.flat, colours):
        g = t["family"] == f
        _quadrants(ax, labels=False, alpha=.35)
        ax.scatter(t.loc[~g, "r2_size"], t.loc[~g, "r2_trajectory"], s=5, color=S.NODATA, lw=0, zorder=1)
        ax.scatter(t.loc[g, "r2_size"], t.loc[g, "r2_trajectory"], s=14, color=colours[f], lw=.3, edgecolor="white", zorder=2)
        _labels(ax, t.loc[g, "r2_size"], t.loc[g, "r2_trajectory"], labels[g], [colours[f]] * int(g.sum()), fontsize=5.5)
        row = fam[fam["family"] == f].iloc[0]
        ax.set_title(f"{short(f)} ({int(row['n_tasks'])}, {row['share_majority']:.0%} {REGIME_SHORT[row['regime_majority']]})", loc="left", fontsize=7)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.grid(color=S.GRID, lw=.5); S.clean(ax)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("median R² across size", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("median R² along training", fontsize=7)
    top = G._header(fig, "Panel (b) per benchmark family, tasks named by language",
                    "the family's tasks coloured and labelled (language code; arc: the task suffix), the other tasks in grey; "
                    "title = tasks, share of them in the family's majority regime and that regime (both = predictable across size and "
                    "training, weak = neither); quadrants as in scaling_regimes.png")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, out_dir / "scaling_regimes_by_family.png"); plt.close(fig)


def html(t: pd.DataFrame, out_dir: Path) -> None:
    """The two panels as a Vega-Lite page: hover shows the task and its
    numbers, clicking a legend entry highlights the family."""
    colours = _colours(t)
    fams = list(colours)
    rng = [mpl.colors.to_hex(colours[f]) for f in fams]
    rows = t.round(3).to_dict("records")
    quads = [{"x": R2_SPLIT if px else 0, "x2": 1 if px else R2_SPLIT, "y": R2_SPLIT if py else 0, "y2": 1 if py else R2_SPLIT,
              "regime": lab, "colour": col} for (px, py), (lab, col) in QUADRANTS.items()]
    tooltip = [{"field": "task"}, {"field": "family"}, {"field": "language"}, {"field": "regime"},
               {"field": "r2_size", "title": "R² size"}, {"field": "rho_size", "title": "ρ size"},
               {"field": "r2_trajectory", "title": "R² trajectory"}, {"field": "n_size_fits"}, {"field": "n_trajectory_fits"}]
    def points(y, ytitle, legend):
        return {"mark": {"type": "circle", "size": 55, "stroke": "white", "strokeWidth": .5},
                "encoding": {"x": {"field": "r2_size", "type": "quantitative", "title": "median R² across model-size fits",
                                   "scale": {"domain": [0, 1]}},
                             "y": {"field": y, "type": "quantitative", "title": ytitle, "scale": {"domain": [-1, 1] if y == "rho_size" else [0, 1]}},
                             "color": {"field": "family", "type": "nominal", "scale": {"domain": fams, "range": rng},
                                       "legend": {"title": "benchmark (click to highlight)"} if legend else None},
                             "opacity": {"condition": {"param": "fam", "value": .9}, "value": .12},
                             "tooltip": tooltip},
                **({"params": [{"name": "fam", "select": {"type": "point", "fields": ["family"]}, "bind": "legend"}]} if legend else {})}
    panel_a = {"width": 360, "height": 320, "title": {"text": "(a) fit quality and direction across size", "anchor": "start"},
               "data": {"values": rows}, "layer": [{"mark": {"type": "rule", "color": "#e3e2de"}, "encoding": {"y": {"datum": 0}}},
                                                   points("rho_size", "median Spearman ρ with model size (improving = +)", False)]}
    panel_b = {"width": 360, "height": 320, "title": {"text": "(b) predictability across size and along training", "anchor": "start"},
               "layer": [{"data": {"values": quads}, "mark": {"type": "rect", "opacity": .55},
                          "encoding": {"x": {"field": "x", "type": "quantitative"}, "x2": {"field": "x2"},
                                       "y": {"field": "y", "type": "quantitative"}, "y2": {"field": "y2"},
                                       "color": {"field": "colour", "type": "nominal", "scale": None, "legend": None}}},
                         {"data": {"values": quads}, "mark": {"type": "text", "fontSize": 10, "opacity": .7},
                          "encoding": {"x": {"field": "x", "type": "quantitative"}, "y": {"field": "y", "type": "quantitative"},
                                       "text": {"field": "regime"}},
                          "transform": [{"calculate": "(datum.x + datum.x2) / 2", "as": "x"}, {"calculate": "(datum.y + datum.y2) / 2", "as": "y"}]},
                         {"data": {"values": rows}, **points("r2_trajectory", "median R² across training-trajectory fits", True)}]}
    spec = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", "hconcat": [panel_a, panel_b],
            "resolve": {"scale": {"color": "shared"}}, "config": {"font": "Helvetica, Arial, sans-serif", "axis": {"grid": True, "gridColor": "#e3e2de"}}}
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Benchmark scaling predictability</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script><script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<style>body{{font-family:Helvetica,Arial,sans-serif;color:#1b1b19;margin:24px}} p{{max-width:900px;font-size:13px;color:#5c5c57}}</style></head>
<body><h3>Benchmark scaling predictability per benchmark–language pair</h3><p>{NOTE}. Hover a point for the task; click a legend entry to highlight a family (shift-click adds, click the background to reset).</p>
<div id="vis"></div><script>vegaEmbed("#vis", {json.dumps(spec)}, {{actions: false}});</script></body></html>
"""
    (out_dir / "scaling_regimes.html").write_text(page)
    print(f"wrote {out_dir / 'scaling_regimes.html'}")


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    fits = pd.read_csv(out_dir / "rq1_fits.csv")
    df = ladder_frame(pool)
    t = size_medians(fits).join(trajectory_medians(df), how="inner").reset_index()
    t["family"] = t["task"].map(benchmark_family)
    t["language"] = t["task"].map(assign_language)
    t["regime"] = [QUADRANTS[(x >= R2_SPLIT, y >= R2_SPLIT)][0] for x, y in zip(t["r2_size"], t["r2_trajectory"])]
    t = t[["task", "family", "language", "n_size_fits", "r2_size", "rho_size", "n_trajectory_fits", "r2_trajectory", "regime"]]
    t.to_csv(out_dir / "scaling_regimes.csv", index=False)
    fam, out = family_table(t), outliers(t, family_table(t))
    fam.to_csv(out_dir / "scaling_regimes_families.csv", index=False)
    out.to_csv(out_dir / "scaling_regimes_outliers.csv", index=False)
    figure(t, out_dir)
    figure(t, out_dir, fam=fam, name="scaling_regimes_families")
    figure(t, out_dir, fam=fam, out=out, name="scaling_regimes_outliers")
    figure(t, out_dir, fam=fam, out=out, name="scaling_regimes_outliers_paper", paper=True)
    figure_by_family(t, fam, out_dir)
    html(t, out_dir)
    print(f"Wrote scaling_regimes*.png/.csv/.html ({len(t)} tasks, {len(out)} named outliers)")
    print(t.groupby("regime").size().rename("tasks").to_string())
    print(t.groupby("family")[["r2_size", "rho_size", "r2_trajectory"]].median().round(2).to_string())
    if pool == CANONICAL:
        body = "\n\n".join([
            "## Scaling regimes per benchmark-language pair",
            f"`regimes.py`: one point per task, medians over the deep scheme-A seed-1904 cells that train its language — the R² and "
            f"(oriented) Spearman ρ of the log-N fits of `rq1_fits.csv`, one per L, and the R² of the training-trajectory fit "
            f"(score ~ log tokens over a run's checkpoints, ≥ {MIN_POINTS} points), one per (L, size). The quadrants of (b) split at "
            f"R² = {R2_SPLIT}, a heuristic. Table: `scaling_regimes.csv`. Regenerate with `python analysis/rq01_scaling_predictability/regimes.py --pool {pool}`.",
            f"![Scaling regimes]({stage}/{pool}/scaling_regimes.png)",
            f"Named variants of the same points: `scaling_regimes_families.png` (one label per family at its median point, "
            f"`scaling_regimes_families.csv`), `scaling_regimes_outliers.png` (plus the tasks in another quadrant than their family's "
            f"majority and > {OUTLIER_DIST} from its median point, `scaling_regimes_outliers.csv`; `scaling_regimes_outliers_paper.png/.pdf` is its bare, square-panel version "
            f"for the paper), `scaling_regimes_by_family.png` "
            f"(panel (b) per family, tasks named by language) and `scaling_regimes.html` (hover names, click-to-highlight legend; "
            f"for the project site).",
            f"![Scaling regimes, outliers named]({stage}/{pool}/scaling_regimes_outliers.png)",
            f"![Scaling regimes per family]({stage}/{pool}/scaling_regimes_by_family.png)"])
        replace_block(OUT_ROOT / "README.md", "regimes", body, f"regimes.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
