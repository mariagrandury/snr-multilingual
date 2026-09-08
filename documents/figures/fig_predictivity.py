"""RQ6 figures: how big a proxy has to be before it picks the same winner."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S, data
from predictivity import min_predictive_size, kind

OUT = Path(__file__).resolve().parents[1] / "public" / "ladder"
PROXIES = ["175M", "350M"]
NEVER = len(PROXIES)                       # the "no size works" column
DECISION = {1: "depth", 2: "depth", 8: "scheme", 15: "scheme", 30: "scheme"}


def _slot(v):
    return PROXIES.index(v) if isinstance(v, str) and v in PROXIES else NEVER


def line_plot(d, out):
    """x = smallest predictive size, y = languages in the data. Every language is a
    dot, so the spread is visible; the aggregate measures are lines on top."""
    d = d.assign(kind=d["task"].map(kind), slot=d["min_size"].map(_slot))
    Ls = sorted(d["L"].unique())
    ypos = {L: i for i, L in enumerate(Ls)}
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    per_lang = d[d["kind"] == "one language's bits per byte"]
    for L, g in per_lang.groupby("L"):
        y = ypos[L] + rng.uniform(-.26, .26, len(g))
        x = g["slot"] + rng.uniform(-.16, .16, len(g))
        ax.scatter(x, y, s=9, color=S.RAMP[0], alpha=.55, lw=0, zorder=2)
    med = per_lang.groupby("L")["slot"].median()
    # three different KINDS of measurement, so categorical slots in fixed order,
    # each with its own marker shape so identity never rests on colour alone
    ax.plot([med[L] for L in Ls], [ypos[L] for L in Ls], marker="o", ms=9, lw=2.4,
            color=S.SERIES[0], zorder=5, label="one language's bits per byte (median)")
    for name, colour, marker in (("macro bits per byte", S.SERIES[1], "s"),
                                 ("training loss", S.SERIES[2], "^")):
        g = d[d["kind"] == name].set_index("L")["slot"]
        ax.plot([g.get(L, np.nan) for L in Ls], [ypos[L] for L in Ls], marker=marker,
                ms=7, lw=1.8, color=colour, zorder=4, label=name)

    ax.set_xticks(range(NEVER + 1)); ax.set_xticklabels(PROXIES + ["no size\nworks"], fontsize=9)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([f"{L} language{'s' if L > 1 else ''}   ·   {DECISION[L]}" for L in Ls],
                       fontsize=9, color=S.INK)
    ax.set_xlabel("smallest model that picks the same winner as 600M", fontsize=9.5, color=S.MUTED)
    ax.set_xlim(-.45, NEVER + .45); ax.set_ylim(-.6, len(Ls) - .4)
    ax.grid(axis="x", color=S.GRID, lw=0.8); ax.set_axisbelow(True)
    S.clean(ax, spines=("bottom",)); ax.tick_params(length=0)
    ax.scatter([], [], s=9, color=S.RAMP[0], alpha=.55, label="each dot is one language")
    ax.legend(frameon=False, fontsize=8, labelcolor=S.MUTED, loc="upper left",
              bbox_to_anchor=(0, -0.14), ncol=2)
    S.title(fig, "How big must a proxy be before it agrees with the reference?")
    S.save(fig, out / "min_predictive_size.png")


def heatmap(d, out):
    """Every language, every language count, the size it starts agreeing at."""
    b = d[d["task"].str.startswith("bpb_") & (d["task"] != "bpb_macro")].copy()
    b["lang"] = b["task"].str[len("bpb_"):].map(data.subset_label)
    piv = b.pivot_table(index="lang", columns="L", values="min_size",
                        aggfunc="first").map(_slot, na_action=None)
    piv = piv.fillna(NEVER)
    piv = piv.loc[piv.mean(axis=1).sort_values().index]
    Ls = list(piv.columns)

    fig, ax = plt.subplots(figsize=(13.6, 2.6))
    colours = [S.SIZE_COLOR["175M"], S.SIZE_COLOR["350M"], S.NODATA]
    m = piv.to_numpy().T
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=colours[int(m[i, j])],
                                       edgecolor="white", lw=.35))
    ax.set_xlim(0, m.shape[1]); ax.set_ylim(m.shape[0], 0)
    ax.set_yticks(np.arange(len(Ls)) + .5)
    ax.set_yticklabels([f"L{L}  ({DECISION[L]})" for L in Ls], fontsize=8.5, color=S.INK)
    ax.set_xticks(np.arange(len(piv.index)) + .5)
    ax.set_xticklabels(piv.index, fontsize=4.4, rotation=90, color=S.MUTED)
    ax.set_xlabel("validation language, ordered by how small a proxy it tolerates",
                  fontsize=9, color=S.MUTED)
    S.clean(ax, spines=()); ax.tick_params(length=0)
    ax.legend(handles=[Patch(facecolor=colours[0], label="175M is enough"),
                       Patch(facecolor=colours[1], label="needs 350M"),
                       Patch(facecolor=colours[2], label="no proxy works")],
              frameon=False, fontsize=8.5, labelcolor=S.MUTED, ncol=3,
              loc="upper center", bbox_to_anchor=(.5, -0.42))
    S.title(fig, "Smallest proxy that gets the decision right, language by language", y=1.08)
    S.save(fig, out / "min_predictive_per_language.png")


if __name__ == "__main__":
    d = min_predictive_size()
    OUT.mkdir(parents=True, exist_ok=True)
    line_plot(d, OUT)
    heatmap(d, OUT)
