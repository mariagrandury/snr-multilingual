"""RQ6 in the shape the team drew on the whiteboard.

Two views of the same numbers, one panel per decision (depth, data scheme).
First, what we can measure: the share of languages whose bits per byte picks
the reference's winner, against proxy size, one line per language count.
Second, the whiteboard figure itself: the smallest proxy that reaches a given
decision accuracy, against the number of languages, one line per level.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S
from predictivity import agreement_by_size

OUT = Path(__file__).resolve().parents[1] / "public" / "ladder"
GUTTER = 1.35e9         # x position of the "no proxy size reaches this" column
NPARAMS = {"90M": 9.0e7, "175M": 1.75e8, "350M": 3.5e8, "600M": 6.0e8,
           "1B": 1.0e9, "1.7B": 1.7e9}
# predictivity.INTERVENTION's two decisions: depth is paired at every L, the
# data scheme (A vs B) at L8-L30 only.
DECISION = {"arch": "depth", "scheme": "data scheme"}
_RAMP = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#154a8a", "#0b2d55"]


def lshade(L, present) -> str:
    """Colour for a language count, ramped over the counts the data has."""
    order = sorted(present)
    return _RAMP[min(order.index(L), len(_RAMP) - 1)] if L in order else _RAMP[-1]


def _proxies(g):
    g = g[g["size"] != g["reference_size"]]
    return g.sort_values("size", key=lambda s: s.map(NPARAMS))


def da_vs_size(d, out):
    """What we can measure: agreement against proxy size, one panel per
    decision, one line per language count."""
    ref = d["reference_size"].iloc[0]
    present = sorted(d["L"].unique())
    sizes = [s for s in NPARAMS if s in set(d["size"]) and s != ref]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    for ax, (axis, what) in zip(axes, DECISION.items()):
        for L, g in d[d["intervention"] == axis].groupby("L"):
            g = _proxies(g)
            ax.plot([NPARAMS[s] for s in g["size"]], g["agree"], marker="o", ms=7, lw=2.0,
                    color=lshade(L, present), label=f"{L} language{'s' if L > 1 else ''}")
        for t, lab in ((0.5, "chance"), (0.75, "usable")):
            ax.axhline(t, color=S.MUTED, ls="--", lw=1)
            ax.annotate(lab, (NPARAMS[sizes[0]] * .9, t), textcoords="offset points",
                        xytext=(0, 4), fontsize=8, color=S.MUTED)
        ax.set_xscale("log")
        ax.set_xticks([NPARAMS[s] for s in sizes]); ax.set_xticklabels(sizes, fontsize=9)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlim(NPARAMS[sizes[0]] * .8, NPARAMS[sizes[-1]] * 1.25)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{what} decision", fontsize=10.5, color=S.INK)
        ax.set_xlabel("proxy model size", fontsize=9.5, color=S.MUTED)
        ax.grid(axis="y", color=S.GRID, lw=.8); ax.set_axisbelow(True)
        S.clean(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=S.MUTED, loc="lower left")
    axes[0].set_ylabel(f"share of languages that pick the {ref} winner",
                       fontsize=9.5, color=S.MUTED)
    S.title(fig, f"{len(sizes)} proxy sizes against the {ref} reference, on per-language bits per byte")
    S.save(fig, out / "rq6_da_vs_size.png")


def whiteboard(d, out, levels=(0.5, 0.75, 0.9)):
    """The sketch: smallest proxy reaching each accuracy level, against L, one
    panel per decision."""
    ref = d["reference_size"].iloc[0]
    sizes = [s for s in NPARAMS if s in set(d["size"]) and s != ref]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    for ax, (axis, what) in zip(axes, DECISION.items()):
        sub = d[d["intervention"] == axis]
        Ls = sorted(sub["L"].unique())
        ax.axvline(GUTTER, color=S.GRID, lw=1)
        ax.annotate("never reached", (GUTTER, 70), fontsize=8, color=S.MUTED, ha="center",
                    va="bottom", annotation_clip=False)
        for lvl, colour, marker in zip(levels, S.SERIES, ("o", "s", "^")):
            xs, ys, misses = [], [], []
            for L in Ls:
                g = _proxies(sub[sub["L"] == L])
                # the smallest proxy from which every larger proxy reaches lvl,
                # as predictivity.min_predictive_size reads a single language
                held = (g["agree"] >= lvl)[::-1].cummin()[::-1]
                ok = g[held.to_numpy()]
                if len(ok):
                    xs.append(NPARAMS[ok["size"].iloc[0]]); ys.append(L)
                else:
                    misses.append(L)
            ax.plot(xs, ys, marker=marker, ms=8, lw=2.0, color=colour,
                    label=f"decision accuracy ≥ {lvl:g}")
            for L in misses:
                ax.plot(GUTTER, L, marker=marker, ms=8, mfc="white", mec=colour, mew=1.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([NPARAMS[s] for s in sizes]); ax.set_xticklabels(sizes, fontsize=9)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks([1, 2, 8, 15, 30, 50])
        ax.set_yticklabels(["1", "2", "8", "15", "30", "50"], fontsize=9)
        ax.set_xlim(NPARAMS[sizes[0]] * .8, 1.6e9); ax.set_ylim(0.8, 65)
        ax.set_title(f"{what} decision", fontsize=10.5, color=S.INK)
        ax.set_xlabel("smallest proxy that reaches it and keeps it",
                      fontsize=9.5, color=S.MUTED)
        ax.grid(color=S.GRID, lw=.8); ax.set_axisbelow(True)
        S.clean(ax)
    axes[0].plot([], [], marker="o", ms=8, mfc="white", mec=S.MUTED, mew=1.8, ls="none",
                 label="not reached by any proxy size")
    axes[0].set_ylabel("languages in the pretraining data", fontsize=9.5, color=S.MUTED)
    axes[0].legend(frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="upper left")
    S.title(fig, f"Smallest proxy from which every larger proxy reaches each accuracy ({ref} reference)")
    S.save(fig, out / "rq6_whiteboard.png")


if __name__ == "__main__":
    d = agreement_by_size()
    OUT.mkdir(parents=True, exist_ok=True)
    da_vs_size(d, OUT)
    whiteboard(d, OUT)
