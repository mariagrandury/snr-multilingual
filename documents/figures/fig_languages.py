"""Per-language figures: what more languages buys, language by language."""
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data, style as S

OUT = Path(__file__).resolve().parents[1] / "public" / "ladder"


def bpb_gain_per_language(m, out):
    """Every validation language on the x axis, one row per size, cell = bits per
    byte saved by training on 50 languages instead of 2."""
    sizes = [s for s in S.SIZES if (s, 2) in m.index and (s, 50) in m.index]
    gain = {s: m.loc[(s, 2)] - m.loc[(s, 50)] for s in sizes}
    g = pd.DataFrame(gain).T
    order = g.mean().sort_values(ascending=False).index          # best-helped first
    g = g[order]
    lim = float(np.nanmax(np.abs(g.to_numpy())))

    # 100 languages in one strip forces a 4pt label that no projector resolves,
    # so the ordering is cut in half and stacked. Best helped on the top row.
    half = (len(order) + 1) // 2
    blocks = [order[:half], order[half:]]
    fig, axes = plt.subplots(len(blocks), 1, figsize=(13.4, 2.35 * len(blocks)))
    for ax, cols in zip(axes, blocks):
        sub = g[cols]
        im = ax.imshow(np.ma.masked_invalid(sub.to_numpy()), cmap=S.DIV,
                       vmin=-lim, vmax=lim, aspect="auto")
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels(sub.index, fontsize=9, color=S.INK)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([data.subset_label(c) for c in cols], fontsize=7.6,
                           rotation=90, color=S.MUTED)
        S.clean(ax, spines=())
    axes[-1].set_xlabel("validation language, ordered by how much it gains",
                        fontsize=9, color=S.MUTED)
    cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.008)
    cb.set_label("bits per byte saved", fontsize=8.5, color=S.MUTED)
    cb.ax.tick_params(labelsize=7.5, colors=S.MUTED)
    n_worse = int((g.mean() < 0).sum())
    S.title(fig, f"Going from 2 to 50 languages: every language, every size "
                 f"({len(order) - n_worse} of {len(order)} improve)", y=1.03, size=12)
    S.save(fig, out / "bpb_gain_per_language.png")


def english_vs_rest(m, out):
    """The one language that pays, against the ones that gain."""
    sizes = [s for s in S.SIZES if any(k[0] == s for k in m.index)]
    Ls = sorted({k[1] for k in m.index})
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for s in sizes:
        xs = [L for L in Ls if (s, L) in m.index]
        if len(xs) < 2:
            continue
        eng = [m.loc[(s, L), "dclm"] for L in xs]
        rest = [m.loc[(s, L)].drop("dclm").median() for L in xs]
        ax.plot(xs, eng, marker="o", ms=5, lw=2, color=S.SIZE_COLOR[s], ls=":")
        ax.plot(xs, rest, marker="o", ms=5, lw=2, color=S.SIZE_COLOR[s])
        ax.annotate(s, (xs[-1], rest[-1]), textcoords="offset points", xytext=(7, 0),
                    fontsize=8.5, color=S.MUTED, va="center")
    ax.set_xscale("log"); ax.set_xticks(Ls)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("languages in the pretraining data", fontsize=9, color=S.MUTED)
    ax.set_ylabel("bits per byte", fontsize=9, color=S.MUTED)
    ax.grid(axis="y", color=S.GRID, lw=0.8); ax.set_axisbelow(True)
    S.clean(ax)
    # sizes are labelled on the lines already, so the legend only explains the
    # two line styles
    ax.plot([], [], color=S.MUTED, ls="-", lw=2, label="median of the other 99")
    ax.plot([], [], color=S.MUTED, ls=":", lw=2, label="English")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="upper right",
              handlelength=2.4, borderaxespad=0.2)
    S.title(fig, "English flattens out. Everything else keeps improving.", y=1.02)
    S.save(fig, out / "english_vs_rest.png")


if __name__ == "__main__":
    import pandas as pd
    globals()["pd"] = pd
    m = data.bpb_matrix(data.wide())
    OUT.mkdir(parents=True, exist_ok=True)
    bpb_gain_per_language(m, OUT)
    english_vs_rest(m, OUT)
