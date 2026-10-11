"""RQ6 figures: how big a proxy has to be before it picks the same winner.

Both read predictivity.min_predictive_size(): one row per (decision, L,
measurement), on the rule-filtered pool, against the 1.7B reference. A row of a
figure is one decision at one language count: depth (deep vs shallow) is paired
at every L, the data scheme (A vs B) at L8-L30.
"""
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
DECISION = {"arch": "depth", "scheme": "scheme"}


def _proxies(d):
    """Every proxy size the rows were read at: the ladder below the reference."""
    ref = d["reference_size"].iloc[0]
    return [s for s in S.SIZES if s != ref], ref


def _rows(d):
    """(intervention, L) in display order: depth first, then the scheme."""
    keys = d[["intervention", "L"]].drop_duplicates()
    keys = keys.assign(o=keys["intervention"].map({"arch": 0, "scheme": 1}))
    return [(a, int(L)) for a, L in keys.sort_values(["o", "L"])[["intervention", "L"]].itertuples(index=False)]


def _label(a, L):
    return f"{L} language{'s' if L > 1 else ''}   ·   {DECISION.get(a, a)}"


def line_plot(d, out):
    """x = smallest predictive size, y = (decision, L). Every language is a dot,
    so the spread is visible; the other measures are lines on top."""
    proxies, ref = _proxies(d)
    never = len(proxies)
    slot = lambda v: proxies.index(v) if isinstance(v, str) and v in proxies else never  # noqa: E731
    d = d.assign(kind=d["task"].map(kind), slot=d["min_size"].map(slot))
    rows = _rows(d)
    ypos = {r: i for i, r in enumerate(rows)}
    rng = np.random.default_rng(0)

    h = 0.62 * len(rows) + 1.6
    fig, ax = plt.subplots(figsize=(8.6, h))
    fig.subplots_adjust(top=1 - 0.55 / h)
    per_lang = d[d["kind"] == "one language's bits per byte"]
    for (a, L), g in per_lang.groupby(["intervention", "L"]):
        y = ypos[(a, int(L))] + rng.uniform(-.24, .24, len(g))
        x = g["slot"] + rng.uniform(-.16, .16, len(g))
        ax.scatter(x, y, s=12, color=S.RAMP[0], alpha=.6, lw=0, zorder=2)

    def line(sub, name, colour, marker, ms, lw, ls="-", stat="median"):
        """One line per decision, so depth and scheme rows are never joined."""
        g = sub.groupby(["intervention", "L"])["slot"].agg(stat)
        for k, a in enumerate(dict.fromkeys(r[0] for r in rows)):
            rr = [r for r in rows if r[0] == a]
            ax.plot([g.get(r, np.nan) for r in rr], [ypos[r] for r in rr], marker=marker,
                    ms=ms, lw=lw, ls=ls, color=colour, zorder=4, label=None if k else name)

    line(per_lang, "one language's bits per byte (median)", S.SERIES[0], "o", 9, 2.4)
    line(d[d["kind"] == "macro bits per byte"], "macro bits per byte", S.SERIES[1], "s", 7, 1.8)
    line(d[d["kind"] == "training loss"], "training loss", S.SERIES[2], "^", 7, 1.8)
    bench = d[d["kind"] == "a benchmark task"]
    share = bench.assign(ok=bench["min_size"].notna()).groupby(["intervention", "L"])["ok"].mean()
    line(bench, f"benchmark tasks (median of {len(bench):,})", S.MUTED, "x", 8, 1.8, ls="--")
    for r in rows:
        if r in share.index:
            ax.annotate(f"{share[r]:.0%} of benchmarks agree", (never + .5, ypos[r]),
                        fontsize=7.5, color=S.MUTED, va="center", annotation_clip=False)

    ax.set_xticks(range(never + 1)); ax.set_xticklabels(proxies + ["no size\nworks"], fontsize=9)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([_label(*r) for r in rows], fontsize=9, color=S.INK)
    ax.set_xlabel(f"smallest proxy that picks the {ref} winner, and every larger proxy too",
                  fontsize=9.5, color=S.MUTED)
    ax.set_xlim(-.45, never + .45); ax.set_ylim(-.6, len(rows) - .4)
    ax.grid(axis="x", color=S.GRID, lw=0.8); ax.set_axisbelow(True)
    S.clean(ax, spines=("bottom",)); ax.tick_params(length=0)
    ax.scatter([], [], s=12, color=S.RAMP[0], alpha=.6, label="each dot is one trained language")
    ax.legend(frameon=False, fontsize=8, labelcolor=S.MUTED, loc="upper left",
              bbox_to_anchor=(0, -0.11 * 9 / h), ncol=2)
    S.title(fig, "How big must a proxy be before it agrees with the reference?")
    S.save(fig, out / "min_predictive_size.png")


def heatmap(d, out):
    """Every trained language, every (decision, L), the size it starts agreeing at.
    Blank where the language is not trained at that L."""
    proxies, ref = _proxies(d)
    never = len(proxies)
    b = d[d["task"].str.startswith("bpb_") & (d["task"] != "bpb_macro")].copy()
    b["lang"] = b["task"].str[len("bpb_"):].map(data.subset_label)
    b["slot"] = b["min_size"].map(lambda v: proxies.index(v) if isinstance(v, str) else never)
    rows = [r for r in _rows(d) if r in set(zip(b["intervention"], b["L"].astype(int)))]
    piv = b.pivot_table(index="lang", columns=["intervention", "L"], values="slot", aggfunc="first")
    piv = piv[[r for r in rows if r in piv.columns]]
    piv = piv.loc[piv.mean(axis=1).sort_values().index]

    colours = [S.SIZE_COLOR[s] for s in proxies] + [S.NODATA]
    h = 0.42 * len(rows) + 1.6
    fig, ax = plt.subplots(figsize=(max(8.0, 0.24 * len(piv.index) + 2.4), h))
    fig.subplots_adjust(top=1 - 0.5 / h)
    m = piv.to_numpy().T
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if np.isfinite(m[i, j]):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=colours[int(m[i, j])],
                                           edgecolor="white", lw=.5))
    ax.set_xlim(0, m.shape[1]); ax.set_ylim(m.shape[0], 0)
    ax.set_yticks(np.arange(len(piv.columns)) + .5)
    ax.set_yticklabels([f"L{L}  ({DECISION.get(a, a)})" for a, L in piv.columns], fontsize=8.5,
                       color=S.INK)
    ax.set_xticks(np.arange(len(piv.index)) + .5)
    ax.set_xticklabels(piv.index, fontsize=7, rotation=90, color=S.MUTED)
    ax.set_xlabel("trained language (blank: not trained at that L), ordered by how small a proxy it tolerates",
                  fontsize=9, color=S.MUTED)
    S.clean(ax, spines=()); ax.tick_params(length=0)
    ax.legend(handles=[Patch(facecolor=c, label=f"{s} is enough") for s, c in zip(proxies, colours)]
              + [Patch(facecolor=S.NODATA, label="no proxy works")],
              frameon=False, fontsize=8.5, labelcolor=S.MUTED, ncol=len(proxies) + 1,
              loc="upper center", bbox_to_anchor=(.5, -0.9 / h - .12))
    S.title(fig, f"Smallest proxy that picks the {ref} winner, language by language", y=1 - 0.15 / h)
    S.save(fig, out / "min_predictive_per_language.png")


if __name__ == "__main__":
    d = min_predictive_size()
    OUT.mkdir(parents=True, exist_ok=True)
    line_plot(d, OUT)
    heatmap(d, OUT)
