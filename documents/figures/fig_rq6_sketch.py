"""RQ6 in the shape the team drew on the whiteboard.

Two views of the same numbers. First, what we can measure: decision accuracy
against model size, one line per language count. Second, the whiteboard figure
itself: the smallest model that reaches a given decision accuracy, against the
number of languages, one line per accuracy level, with the part of the grid
today's runs cannot answer greyed out.
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
GUTTER = 5.3e8          # x position of the "no trained size reaches this" column
NPARAMS = {"90M": 9.0e7, "175M": 1.75e8, "350M": 3.5e8, "600M": 6.0e8,
           "1B": 1.0e9, "1.7B": 1.7e9}
DECISION = {1: "depth", 2: "depth", 8: "scheme", 15: "scheme", 30: "scheme"}
LSHADE = {1: "#cde2fb", 2: "#9ec5f4", 8: "#5598e7", 15: "#2a78d6", 30: "#154a8a"}


def da_vs_size(d, out):
    """What we can measure today: accuracy against size, one line per L."""
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for L, g in d.groupby("L"):
        g = g[g["size"] != g["reference_size"]].sort_values(
            "size", key=lambda s: s.map(NPARAMS))
        ax.plot([NPARAMS[s] for s in g["size"]], g["agree"], marker="o", ms=8, lw=2.2,
                color=LSHADE[L], label=f"{L} language{'s' if L > 1 else ''}  ({DECISION[L]})")
    for t, lab in ((0.5, "coin flip"), (0.75, "usable")):
        ax.axhline(t, color=S.MUTED, ls="--", lw=1)
        ax.annotate(lab, (1.02e8, t), textcoords="offset points", xytext=(0, 4),
                    fontsize=8, color=S.MUTED)
    ax.set_xscale("log")
    ax.set_xticks([NPARAMS[s] for s in ("175M", "350M")]); ax.set_xticklabels(["175M", "350M"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlim(1.0e8, 4.6e8)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("proxy model size", fontsize=9.5, color=S.MUTED)
    ax.set_ylabel("share of languages that pick the same winner as 600M",
                  fontsize=9.5, color=S.MUTED)
    ax.grid(axis="y", color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="upper left",
              bbox_to_anchor=(1.01, 1.0))
    S.title(fig, "Only two proxy sizes exist, so this is the whole measurement")
    S.save(fig, out / "rq6_da_vs_size.png")


def whiteboard(d, out, levels=(0.5, 0.75, 0.9)):
    """The sketch: smallest model reaching each accuracy level, against L."""
    Ls = sorted(d["L"].unique())
    fig, ax = plt.subplots(figsize=(7.8, 4.4))

    # Grey is what today's grid cannot answer: 600M is the reference, so no
    # answer can land at or past it, and above 30 languages nothing is paired.
    ax.axvspan(4.3e8, 2.4e9, color="#f6f5f3", zorder=0)
    ax.axhspan(34, 240, color="#f6f5f3", zorder=0)
    ax.annotate("no paired runs above 30 languages", (1.35e8, 40), fontsize=9,
                color="#9a978f", ha="left")
    ax.annotate("", (1.75e9, 90), (7.2e8, 4), zorder=1,
                arrowprops=dict(arrowstyle="->", color="#c9c6bf", lw=1.6))
    ax.annotate("the shape we expect:\nmore languages needs a bigger proxy",
                (1.15e9, 16), fontsize=8.5, color="#9a978f", ha="center", rotation=31)
    # a gutter for the levels no trained size reaches
    ax.axvline(GUTTER, color=S.GRID, lw=1)
    ax.annotate("never reached", (GUTTER, 250), fontsize=8, color=S.MUTED, ha="center",
                va="bottom", annotation_clip=False)

    for lvl, colour, marker in zip(levels, S.SERIES, ("o", "s", "^")):
        xs, ys, misses = [], [], []
        for L in Ls:
            g = d[(d["L"] == L) & (d["size"] != d["reference_size"])]
            ok = g[g["agree"] >= lvl].sort_values("size", key=lambda s: s.map(NPARAMS))
            if len(ok):
                xs.append(NPARAMS[ok["size"].iloc[0]]); ys.append(L)
            else:
                misses.append(L)
        ax.plot(xs, ys, marker=marker, ms=8, lw=2.2, color=colour,
                label=f"decision accuracy ≥ {lvl:g}")
        for L in misses:
            ax.plot(GUTTER, L, marker=marker, ms=8, mfc="white", mec=colour, mew=1.8)
    ax.plot([], [], marker="o", ms=8, mfc="white", mec=S.MUTED, mew=1.8, ls="none",
            label="not reached at any size we have")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks([NPARAMS[s] for s in ("90M", "175M", "350M", "600M", "1B", "1.7B")])
    ax.set_xticklabels(["90M", "175M", "350M", "600M", "1B", "1.7B"], fontsize=9)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([1, 2, 8, 15, 30, 50, 100, 200])
    ax.set_yticklabels(["1", "2", "8", "15", "30", "50", "100", "200"], fontsize=9)
    ax.set_xlim(7.5e7, 2.4e9); ax.set_ylim(0.8, 240)
    ax.set_xlabel("smallest model that reaches that accuracy (600M is the reference)",
                  fontsize=9.5, color=S.MUTED)
    ax.set_ylabel("languages in the pretraining data", fontsize=9.5, color=S.MUTED)
    ax.grid(color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="upper left")
    S.title(fig, "The figure we want, and the corner of it we can fill in")
    S.save(fig, out / "rq6_whiteboard.png")


if __name__ == "__main__":
    d = agreement_by_size()
    OUT.mkdir(parents=True, exist_ok=True)
    da_vs_size(d, OUT)
    whiteboard(d, OUT)
