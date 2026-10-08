"""The ladder's curves — training loss and benchmark accuracy against the
fraction of the run — drawn from the analysis loader's frame so they carry
every assumption of the other RQs (diverged and unfinished runs dropped,
shared checkpoint grid, `require_final`). The detailed counterpart of the
progress report's figures, on the same cells.

    loss_curves.png        training loss vs fraction of run, per L: full run and last 10 %
    benchmark_curves.png   benchmark accuracy vs the run in Chinchilla multiples per family, chance line;
                           a family's `rf_` / `rfgm_` twin sits next to it, titled "<name> (rf)"
    benchmark_curves_paper.png/.svg   the same without the header, a legend instead (rule 18)
    benchmark_size_curves_paper.png/.svg/.csv   its size twin: each design's final accuracy against
                           non-embedding parameters per family, colour = L (the rq01 appendix figure)

    python analysis/rq00_gate_and_curves/curves.py --pool predictivity_seeds
    python analysis/rq00_gate_and_curves/curves.py --paper    # the benchmark figures alone, from benchmark_curves.csv
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
from pretrain.ladder_report import CELL_RE, NON_EMB, SCHEME_OF, _trained_tasks  # noqa: E402
from snr.download.ladder import ladder_dir  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import replace_block  # noqa: E402
from analysis.paths import GATE_AND_CURVES  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import task_chance  # noqa: E402
from analysis.utils import LADDER_SIZES, SHARED_FRACS, benchmark_family, ladder_frame, on_shared_grid  # noqa: E402

OUT_ROOT = GATE_AND_CURVES
CANONICAL = "predictivity_seeds"      # every cell: all seeds and data builds
CLOSEUP_YMAX = 3.5                  # ceiling of the last-10 % loss panels, as in the report
TWINS = ("rf", "rfgm")             # a twin's prefix, written as a suffix in the panel title
Y_TICKS = 6                         # every benchmark panel carries this many y ticks
mpl.rcParams.update(S.RC)


def _line_style(size, ladder, data) -> dict:
    return dict(color=S.SIZE_COLOR.get(size, S.MUTED), lw=S.LADDER_WIDTH.get(ladder, 1.0),
                ls=S.DATA_DASH.get(data, "-"))


def plot_loss_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """The report's loss figure on the analysis' cells: per L, the whole run
    and its last 10 % (the close-up is where ladder and data build separate)."""
    # the report's `scheme` is the build label, the frame's `data`
    curve = pd.read_csv(ladder_dir() / "ladder_report_curve.csv", low_memory=False).rename(columns={"scheme": "data"})
    # the cell's ladder from the frame: a report from before 2026-10-05 has no `ladder` column
    ladder_of = dict(zip(df["model"], df["ladder"]))
    curve = curve[curve["cell"].isin(ladder_of)].assign(ladder=lambda c: c["cell"].map(ladder_of))
    if curve.empty:
        return
    Ls = sorted(curve["L"].unique())
    fig, axes = plt.subplots(len(Ls), 2, figsize=(9.5, 2.6 * len(Ls)), squeeze=False)
    for row, L in enumerate(Ls):
        sub = curve[curve["L"] == L]
        for col, (lo, title) in enumerate(((0.0, "full run"), (0.9, "last 10 %"))):
            ax = axes[row][col]
            win = sub[sub["frac"] >= lo]
            for _cell, g in win.groupby("cell"):
                g = g.sort_values("frac")
                ax.plot(g["frac"], g["loss"], **_line_style(*g.iloc[0][["size", "ladder", "data"]]))
            if col == 0:
                ax.set_ylim(2, 8)
            else:
                v = win["loss"]
                ax.set_ylim(v.min() * 0.995, min(v.max() * 1.005, CLOSEUP_YMAX))
            ax.set_title(f"L = {L} — {title}", loc="left"); ax.set_xlabel("fraction of run")
            ax.set_ylabel("lm loss"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    handles = ([plt.Line2D([], [], color=S.SIZE_COLOR[s], lw=2, label=s) for s in LADDER_SIZES if s in set(curve["size"])]
               + [plt.Line2D([], [], color=S.INK, lw=S.LADDER_WIDTH[a], label=a) for a in S.LADDER_WIDTH if a in set(curve["ladder"])]
               + [plt.Line2D([], [], color=S.INK, ls=S.DATA_DASH[v], label=f"data {v}") for v in S.DATA_DASH if v in set(curve["data"])])
    fig.legend(handles=handles, ncol=min(8, len(handles)), loc="lower center", frameon=False)
    fig.suptitle("Training loss — colour = size, width = ladder, dash = data build", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    curve.to_csv(out_dir / "loss_curves.csv", index=False)      # rule 12
    S.save(fig, out_dir / "loss_curves.png", dpi=150)



def curve_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The benchmark rows the curves draw: the ten tenths (rule 3: the 85 % /
    95 % evals exist only for the noise window and only on some runs, and would
    draw those lines at a different density from the rest), the tasks in the
    languages the cell trains on (the watcher's list)."""
    b = df[(df["kind"] == "benchmark") & on_shared_grid(df)]
    b = b[[t in _trained_tasks(L, d) for t, L, d in zip(b["task"], b["L"], b["data"])]]
    return b[["model", "size", "ladder", "data", "task", "frac", "primary_score"]].copy()


def _twin_key(fam: str) -> tuple:
    """(base family, 0 original / 1 rf / 2 rfgm): sorts a twin next to its original."""
    head, _, rest = fam.partition("_")
    return (rest, 1 + TWINS.index(head)) if head in TWINS and rest else (fam, 0)


def panel_title(fam: str) -> str:
    """`rf_global_mmlu_full` -> `global mmlu full (rf)`."""
    base, k = _twin_key(fam)
    return base.replace("_", " ") + (f" ({TWINS[k - 1]})" if k else "")


def _six_ticks(ax, values: pd.Series) -> None:
    """Y_TICKS evenly spaced ticks on a round step that covers the panel's values."""
    lo, hi = float(values.min()), float(values.max())
    for step in (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2):
        start = np.floor(lo / step) * step
        if start + (Y_TICKS - 1) * step >= hi:
            break
    ticks = start + step * np.arange(Y_TICKS)
    ax.set_yticks(ticks); ax.set_ylim(ticks[0], ticks[-1])


def plot_benchmark_curves(b: pd.DataFrame, out_dir: Path, paper: bool = False) -> None:
    """Benchmark accuracy along the run (Chinchilla multiples), one panel per
    family, a twin next to its original, one line per cell over the tasks in the
    languages that cell trains on, with the chance line from the option count.
    `paper`: no header, a legend for the line encoding (rule 18)."""
    if b.empty:
        return
    b = b.assign(family=b["task"].map(benchmark_family), chance=b["task"].map(task_chance))
    fams = sorted(b["family"].unique(), key=_twin_key)
    cols = min(4, len(fams)); rows = (len(fams) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    flat = [a for r in axes for a in r]
    for ax in flat[len(fams):]:
        ax.axis("off")
    ticks = [f for f in SHARED_FRACS if round(f * G.CHINCHILLA_AT_FULL, 6) == int(round(f * G.CHINCHILLA_AT_FULL))]
    for i, (ax, fam) in enumerate(zip(flat, fams)):
        g = b[b["family"] == fam]
        for _cell, gc in g.groupby("model"):
            m = gc.groupby("frac")["primary_score"].mean().sort_index()
            ax.plot(m.index, m.values, **_line_style(*gc.iloc[0][["size", "ladder", "data"]]))
        ch = g["chance"].dropna()
        if not ch.empty:
            ax.axhline(ch.mean(), color=S.ALERT, lw=.9, ls=":")
        _six_ticks(ax, pd.concat([g.groupby(["model", "frac"])["primary_score"].mean(), ch]))
        ax.set_xticks(ticks); ax.set_xticklabels([G.chinchilla(f) for f in ticks])
        ax.set_title(panel_title(fam), loc="left")
        if i + cols >= len(fams):                  # the lowest panel of its column
            ax.set_xlabel("Training progress (Chinchilla multiples)")
        if i % cols == 0:
            ax.set_ylabel("Mean accuracy over\ntrained-language tasks")
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    if paper:
        present = lambda col, keys: [k for k in keys if k in set(b[col])]   # noqa: E731
        handles = ([plt.Line2D([], [], color=S.SIZE_COLOR[s], lw=2, label=s) for s in present("size", LADDER_SIZES)]
                   # ladder samples grey and solid, data-build samples black and thin, so `Deep` and `Data A` differ
                   + [plt.Line2D([], [], color=S.MUTED, lw=S.LADDER_WIDTH[a] + 0.6, label=a.capitalize()) for a in present("ladder", S.LADDER_WIDTH)]
                   + [plt.Line2D([], [], color=S.INK, lw=0.9, ls=S.DATA_DASH[v], label=f"Data {v}") for v in present("data", S.DATA_DASH)]
                   + [plt.Line2D([], [], color=S.ALERT, lw=.9, ls=":", label="Chance")])
        ncol = next((k for k in (8, 9, 6, 7, 5) if len(handles) % k == 0), 8)
        fig.legend(handles=handles, ncol=ncol, loc="lower center", frameon=False, fontsize=7.5, handlelength=3.2)
        fig.tight_layout(rect=(0, 0.5 / fig.get_figheight() * (-(-len(handles) // ncol)), 1, 1))
        S.save_paper(fig, out_dir / "benchmark_curves_paper")
        return
    fig.suptitle("Benchmark accuracy along the run, mean over the cell's trained-language tasks\n"
                 "colour = size, width = ladder, dash = data build, dotted red = chance", y=1.0)
    fig.tight_layout()
    b[["model", "size", "ladder", "data", "task", "frac", "primary_score"]].to_csv(out_dir / "benchmark_curves.csv", index=False)   # rule 12
    S.save(fig, out_dir / "benchmark_curves.png", dpi=150)


def plot_benchmark_size_curves(b: pd.DataFrame, out_dir: Path) -> None:
    """The paper figure's size twin: each design's final-checkpoint accuracy
    (mean over the tasks in the languages it trains on) against non-embedding
    parameters, one panel per family as in `plot_benchmark_curves`, one line per
    design (L, ladder, data build, seed) across its rungs, colour = L, width =
    ladder, dash = data build. The plotted values go to the CSV of the same name
    (rule 12)."""
    keys = {m: CELL_RE.match(m) for m in b["model"].unique()}
    fin = b[b["frac"] == b.groupby("model")["frac"].transform("max")]
    fin = fin.assign(family=fin["task"].map(benchmark_family), chance=fin["task"].map(task_chance),
                     L=fin["model"].map(lambda m: int(keys[m]["L"])), seed=fin["model"].map(lambda m: int(keys[m]["seed"])))
    t = fin.groupby(["family", "L", "ladder", "data", "seed", "size"], as_index=False).agg(
        primary_score=("primary_score", "mean"), chance=("chance", "mean"))
    t["n_non_emb"] = t["size"].map(NON_EMB)
    Ls = sorted(int(L) for L in t["L"].unique())
    colour = dict(zip(Ls, S.SEQ(np.linspace(0.3, 1, len(Ls)))))
    sizes = [s for s in LADDER_SIZES if s in set(t["size"])]
    fams = sorted(t["family"].unique(), key=_twin_key)
    cols = min(4, len(fams)); rows = (len(fams) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.6 * rows), squeeze=False)
    flat = [a for r in axes for a in r]
    for ax in flat[len(fams):]:
        ax.axis("off")
    for i, (ax, fam) in enumerate(zip(flat, fams)):
        g = t[t["family"] == fam]
        for (L, ladder, data, _seed), gd in g.groupby(["L", "ladder", "data", "seed"]):
            gd = gd.sort_values("n_non_emb")
            ax.plot(gd["n_non_emb"], gd["primary_score"], color=colour[L], lw=S.LADDER_WIDTH.get(ladder, 1.0),
                    ls=S.DATA_DASH.get(data, "-"), marker="o", ms=1.8)
        ch = g["chance"].dropna()
        if not ch.empty:
            ax.axhline(ch.mean(), color=S.ALERT, lw=.9, ls=":")
        _six_ticks(ax, pd.concat([g["primary_score"], ch]))
        ax.set_xscale("log"); ax.minorticks_off()
        ax.set_xticks([NON_EMB[s] for s in sizes]); ax.set_xticklabels(sizes)
        ax.set_title(panel_title(fam), loc="left")
        if i + cols >= len(fams):                  # the lowest panel of its column
            ax.set_xlabel("Non-embedding parameters")
        if i % cols == 0:
            ax.set_ylabel("Final accuracy over\ntrained-language tasks")
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    present = lambda col, keys: [k for k in keys if k in set(t[col])]   # noqa: E731
    handles = ([plt.Line2D([], [], color=colour[L], lw=2, label=f"{L} language" + "s" * (L > 1)) for L in Ls]
               + [plt.Line2D([], [], color=S.MUTED, lw=S.LADDER_WIDTH[a] + 0.6, label=a.capitalize()) for a in present("ladder", S.LADDER_WIDTH)]
               + [plt.Line2D([], [], color=S.INK, lw=0.9, ls=S.DATA_DASH[v], label=f"Data {v}") for v in present("data", S.DATA_DASH)]
               + [plt.Line2D([], [], color=S.ALERT, lw=.9, ls=":", label="Chance")])
    ncol = next((k for k in (8, 9, 6, 7, 5) if len(handles) % k == 0), 8)
    fig.legend(handles=handles, ncol=ncol, loc="lower center", frameon=False, fontsize=7.5, handlelength=3.2)
    fig.tight_layout(rect=(0, 0.5 / fig.get_figheight() * (-(-len(handles) // ncol)), 1, 1))
    t.to_csv(out_dir / "benchmark_size_curves_paper.csv", index=False)
    S.save_paper(fig, out_dir / "benchmark_size_curves_paper")


def benchmark_figures(out_dir: Path, b: pd.DataFrame | None = None) -> None:
    """Both benchmark figures; without `b`, from `benchmark_curves.csv` (a table
    from before it carried the cell's size, ladder and data build reads them off the
    name; one that called the build `scheme` is renamed)."""
    if b is None:
        b = pd.read_csv(out_dir / "benchmark_curves.csv")
        if "data" not in b.columns and "scheme" in b.columns:
            b = b.rename(columns={"scheme": "data"})
        if "ladder" not in b.columns:
            keys = {m: CELL_RE.match(m) for m in b["model"].unique()}
            b["size"], b["ladder"] = b["model"].map(lambda m: keys[m]["size"]), b["model"].map(lambda m: keys[m]["ladder"])
            b["data"] = b["model"].map(lambda m: SCHEME_OF[keys[m]["scheme"] or ""])
    plot_benchmark_curves(b, out_dir)
    plot_benchmark_curves(b, out_dir, paper=True)
    plot_benchmark_size_curves(b, out_dir)


def generate_readme(pool: str) -> None:
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    body = ("## Curves on the analysis' cells\n\n"
            f"Every cell the `{pool}` pool holds (all seeds and data builds), after the loader has dropped "
            "diverged and unfinished runs and restricted checkpoints to the shared grid: the detailed "
            "counterpart of the progress report's figures. Loss per L, the whole run and its last 10 % "
            f"(capped at {CLOSEUP_YMAX} nats, where ladder and data build separate); benchmark accuracy as the mean "
            "over the tasks in the languages the cell trains on, one line per cell, chance from the option "
            f"count, along the run in Chinchilla multiples; a family's cloze twin sits next to it, titled `(rf)`. Regenerate with `python analysis/rq00_gate_and_curves/curves.py --pool {pool}`.\n\n"
            f"![Loss curves]({rel}/loss_curves.png)\n\n"
            f"![Benchmark curves]({rel}/benchmark_curves.png)\n\n"
            "The paper version, `benchmark_curves_paper.png` (`--paper`, redrawn from `benchmark_curves.csv`), "
            "drops the header for a legend of the line encoding; its size twin, `benchmark_size_curves_paper.png`, draws "
            "each design's final accuracy against non-embedding parameters, colour = L (the scaling-predictability "
            "appendix's size figure).")
    readme = OUT_ROOT / "README.md"
    replace_block(readme, "curves", body, f"curves.py --pool {pool}")
    print(f"Wrote auto README block → {readme}")


def main(pool: str, out_dir: Path) -> None:
    df = ladder_frame(pool)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pool '{pool}': {df['model'].nunique()} cells")
    plot_loss_curves(df, out_dir)
    benchmark_figures(out_dir, curve_frame(df))
    generate_readme(pool)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Ladder pool from configs/models.json (default: {CANONICAL})")
    p.add_argument("--paper", action="store_true", help="only the benchmark figures, from benchmark_curves.csv")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    out = OUT_ROOT / stage / args.pool
    if args.paper:
        benchmark_figures(out)
        generate_readme(args.pool)
    else:
        main(args.pool, out)
