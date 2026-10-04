"""The ladder's curves — training loss and benchmark accuracy against the
fraction of the run — drawn from the analysis loader's frame so they carry
every assumption of the other RQs (diverged and unfinished runs dropped,
shared checkpoint grid, `require_final`). The detailed counterpart of the
progress report's figures, on the same cells.

    loss_curves.png        training loss vs fraction of run, per L: full run and last 10 %
    benchmark_curves.png   benchmark accuracy vs fraction of run per family, chance line

    python analysis/rq00_gate_and_curves/curves.py --pool predictivity_all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from pretrain.ladder_report import _trained_tasks  # noqa: E402
from snr.download.ladder import ladder_dir  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import replace_block  # noqa: E402
from analysis.paths import GATE_AND_CURVES  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import task_chance  # noqa: E402
from analysis.utils import LADDER_SIZES, benchmark_family, ladder_frame, on_shared_grid  # noqa: E402

OUT_ROOT = GATE_AND_CURVES
CANONICAL = "predictivity_all"      # every cell: all seeds and schemes
CLOSEUP_YMAX = 3.5                  # ceiling of the last-10 % loss panels, as in the report
mpl.rcParams.update(S.RC)


def _line_style(size, arch, scheme) -> dict:
    return dict(color=S.SIZE_COLOR.get(size, S.MUTED), lw=S.ARCH_WIDTH.get(arch, 1.0),
                ls=S.SCHEME_DASH.get(scheme, "-"))


def plot_loss_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """The report's loss figure on the analysis' cells: per L, the whole run
    and its last 10 % (the close-up is where arch and scheme separate)."""
    curve = pd.read_csv(ladder_dir() / "ladder_report_curve.csv", low_memory=False)
    curve = curve[curve["cell"].isin(set(df["model"]))]
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
                ax.plot(g["frac"], g["loss"], **_line_style(*g.iloc[0][["size", "arch", "scheme"]]))
            if col == 0:
                ax.set_ylim(2, 8)
            else:
                v = win["loss"]
                ax.set_ylim(v.min() * 0.995, min(v.max() * 1.005, CLOSEUP_YMAX))
            ax.set_title(f"L = {L} — {title}", loc="left"); ax.set_xlabel("fraction of run")
            ax.set_ylabel("lm loss"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    handles = ([plt.Line2D([], [], color=S.SIZE_COLOR[s], lw=2, label=s) for s in LADDER_SIZES if s in set(curve["size"])]
               + [plt.Line2D([], [], color=S.INK, lw=S.ARCH_WIDTH[a], label=a) for a in ("deep", "shallow") if a in set(curve["arch"])]
               + [plt.Line2D([], [], color=S.INK, ls=S.SCHEME_DASH[v], label=f"scheme {v}") for v in S.SCHEME_DASH if v in set(curve["scheme"])])
    fig.legend(handles=handles, ncol=min(8, len(handles)), loc="lower center", frameon=False)
    fig.suptitle("Training loss — colour = size, width = arch, dash = data scheme", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    curve.to_csv(out_dir / "loss_curves.csv", index=False)      # rule 12
    S.save(fig, out_dir / "loss_curves.png", dpi=150)



def plot_benchmark_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """Benchmark accuracy against fraction of run, one panel per family, one
    line per cell over the tasks in the languages that cell trains on (the
    watcher's list), with the chance line from the option count."""
    b = df[(df["kind"] == "benchmark") & on_shared_grid(df)]   # a checkpoint axis is the ten tenths (rule 3):
    # the 85 % / 95 % evals exist only for the noise window and only on some
    # runs, and would draw those lines at a different density from the rest
    b = b[[t in _trained_tasks(L, s) for t, L, s in zip(b["task"], b["L"], b["scheme"])]].copy()
    if b.empty:
        return
    b["family"] = b["task"].map(benchmark_family)
    b["chance"] = b["task"].map(task_chance)
    fams = sorted(b["family"].unique())
    cols = min(4, len(fams)); rows = (len(fams) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False)
    flat = [a for r in axes for a in r]
    for ax in flat[len(fams):]:
        ax.axis("off")
    for ax, fam in zip(flat, fams):
        g = b[b["family"] == fam]
        for _cell, gc in g.groupby("model"):
            m = gc.groupby("frac")["primary_score"].mean().sort_index()
            ax.plot(m.index, m.values, **_line_style(*gc.iloc[0][["size", "arch", "scheme"]]))
        ch = g["chance"].dropna()
        if not ch.empty:
            ax.axhline(ch.mean(), color=S.ALERT, lw=.9, ls=":")
        ax.set_title(fam, loc="left"); ax.set_xlabel("fraction of run"); ax.set_ylabel("accuracy")
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    fig.suptitle("Benchmark accuracy vs fraction of run, mean over the cell's trained-language tasks\n"
                 "colour = size, width = arch, dash = scheme, dotted red = chance", y=1.0)
    fig.tight_layout()
    b[["model", "task", "frac", "primary_score"]].to_csv(out_dir / "benchmark_curves.csv", index=False)   # rule 12
    S.save(fig, out_dir / "benchmark_curves.png", dpi=150)



def generate_readme(pool: str) -> None:
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    body = ("## Curves on the analysis' cells\n\n"
            f"Every cell the `{pool}` pool holds (all seeds and schemes), after the loader has dropped "
            "diverged and unfinished runs and restricted checkpoints to the shared grid: the detailed "
            "counterpart of the progress report's figures. Loss per L, the whole run and its last 10 % "
            f"(capped at {CLOSEUP_YMAX} nats, where arch and scheme separate); benchmark accuracy as the mean "
            "over the tasks in the languages the cell trains on, one line per cell, chance from the option "
            f"count. Regenerate with `python analysis/rq00_gate_and_curves/curves.py --pool {pool}`.\n\n"
            f"![Loss curves]({rel}/loss_curves.png)\n\n"
            f"![Benchmark curves]({rel}/benchmark_curves.png)")
    readme = OUT_ROOT / "README.md"
    replace_block(readme, "curves", body, f"curves.py --pool {pool}")
    print(f"Wrote auto README block → {readme}")


def main(pool: str, out_dir: Path) -> None:
    df = ladder_frame(pool)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pool '{pool}': {df['model'].nunique()} cells")
    plot_loss_curves(df, out_dir)
    plot_benchmark_curves(df, out_dir)
    generate_readme(pool)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Ladder pool from configs/models.json (default: {CANONICAL})")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    main(args.pool, OUT_ROOT / stage / args.pool)
