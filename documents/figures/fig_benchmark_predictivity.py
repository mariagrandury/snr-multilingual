"""The new RQ6 result: benchmarks can be scored now, and they read as a coin flip.

Left, how often a 175M proxy picks the same data scheme as the 600M reference,
for the benchmark suite against per-language bits per byte. Right, how many
measurements of each kind ever reach a proxy size that agrees and keeps
agreeing. Both from the same intervention run.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S
from predictivity import min_predictive_size, kind

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "documents" / "public" / "ladder"
IV = (REPO / "src" / "signal-and-noise" / "analysis" / "rq06_proxy_predictivity"
      / "pretraining" / "predictivity_seeds" / "intervention_da.csv")
BPB, BENCH = S.RAMP[3], S.SERIES[1]


def agreement(ax):
    """Both proxy rungs, so the recovery at 350M is visible and the missing
    benchmark rung is visible too."""
    iv = pd.read_csv(IV)
    iv = iv[iv.intervention == "scheme"]
    b = iv[iv.population == "benchmark"].set_index(["L", "proxy_size"])
    p = iv[iv.population == "bpb_trained"].set_index(["L", "proxy_size"])
    Ls = sorted({L for L, _ in b.index})
    bars = [("bits per byte, 175M", p, "175M", BPB, .45),
            ("bits per byte, 350M", p, "350M", BPB, 1.0),
            ("benchmark tasks, 175M", b, "175M", BENCH, .45),
            ("benchmark tasks, 350M", b, "350M", BENCH, 1.0)]
    x = np.arange(len(Ls))
    for j, (label, src, size, colour, alpha) in enumerate(bars):
        off = (j - 1.5) * .2
        vals, miss = [], []
        for i, L in enumerate(Ls):
            if (L, size) in src.index:
                vals.append(float(src.loc[(L, size), "decision_acc"]))
            else:
                vals.append(np.nan); miss.append(i)
        ax.bar(x + off, vals, .19, color=colour, alpha=alpha, label=label)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.annotate(f"{v:.2f}", (i + off, v), xytext=(0, 3), ha="center",
                            textcoords="offset points", fontsize=8, color=S.INK)
        for i in miss:
            ax.annotate("no\ndata", (i + off, 0.02), ha="center", va="bottom",
                        fontsize=7.5, color=S.MUTED)
    ax.axhline(.5, color=S.MUTED, ls="--", lw=1)
    ax.annotate("coin flip", (len(Ls) - .45, .5), xytext=(0, 4), ha="right",
                textcoords="offset points", fontsize=8, color=S.MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{L} languages\n{int(b.loc[(L,'175M'),'n_items']):,} benchmark tasks"
                        for L in Ls], fontsize=9)
    ax.set_ylim(0, 1.2); ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xlim(-.6, len(Ls) - .4)
    ax.set_ylabel("agrees with the 600M reference", fontsize=9, color=S.MUTED)
    ax.set_title("Does the proxy pick the same data scheme?", fontsize=10.5,
                 color=S.INK, pad=8)
    ax.grid(axis="y", color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax); ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=S.MUTED, loc="upper left", ncol=2)


def ever_predictive(ax):
    d = min_predictive_size()
    d = d.assign(kind=d["task"].map(kind))
    rows = [("per-language bits per byte", "one language's bits per byte", BPB),
            ("benchmark tasks", "a benchmark task", BENCH)]
    y = np.arange(len(rows))
    for i, (label, k, colour) in enumerate(rows):
        g = d[d["kind"] == k]
        share = g["min_size"].notna().mean() if len(g) else 0.0
        ax.barh(i, share, .5, color=colour)
        ax.annotate(f"{share:.0%}  ({int(g['min_size'].notna().sum()):,} of {len(g):,})",
                    (share, i), xytext=(6, 0), va="center", textcoords="offset points",
                    fontsize=9, color=S.INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, 1.0); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel("share with a proxy size that agrees, and keeps agreeing",
                  fontsize=9, color=S.MUTED)
    ax.set_title("Does any proxy size work at all?", fontsize=10.5, color=S.INK, pad=8)
    ax.grid(axis="x", color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax); ax.tick_params(length=0)
    ax.legend(handles=[Patch(facecolor=BPB, label="bits per byte"),
                       Patch(facecolor=BENCH, label="benchmarks")],
              frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="upper center",
              bbox_to_anchor=(.5, -.18), ncol=2)


if __name__ == "__main__":
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.4, 4.0),
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    agreement(a0)
    ever_predictive(a1)
    fig.subplots_adjust(wspace=.32)
    S.title(fig, "The benchmark suite can finally be scored on the proxy question, and it fails", y=1.04)
    S.save(fig, OUT / "benchmark_predictivity.png")
