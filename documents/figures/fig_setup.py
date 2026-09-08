"""Setup and diagnostic figures that replace tables in the deck."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data, style as S

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "documents" / "public" / "ladder"
ANALYSIS = REPO / "src" / "signal-and-noise" / "analysis"


def grid_status(out):
    """What the ladder actually contains: one square per planned cell."""
    w = data.wide()
    c = data.cells(w)
    Ls = [1, 2, 8, 15, 30, 50, 100]
    sizes = S.SIZES
    state = np.zeros((len(Ls), len(sizes)))          # 0 nothing, 1 running, 2 done, 3 broken
    for (L, s), g in c.groupby([c["L"].astype(int), "size"]):
        if L not in Ls or s not in sizes:
            continue
        i, j = Ls.index(L), sizes.index(s)
        done = g[(g["run__complete"] == 1) & (g["run__diverged"] != 1)]
        if (g["run__diverged"] == 1).all():
            state[i, j] = 3
        elif len(done):
            state[i, j] = 2
        else:
            state[i, j] = 1
    colours = {0: "#f6f5f3", 1: "#cde2fb", 2: S.RAMP[2], 3: "#ded9d3"}

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for i in range(len(Ls)):
        for j in range(len(sizes)):
            ax.add_patch(plt.Rectangle((j, i), .92, .92, facecolor=colours[state[i, j]],
                                       edgecolor="white", lw=1.4))
    ax.set_xlim(-.1, len(sizes)); ax.set_ylim(len(Ls), -.1)
    ax.set_xticks(np.arange(len(sizes)) + .46); ax.set_xticklabels(sizes, fontsize=9.5)
    ax.set_yticks(np.arange(len(Ls)) + .46)
    ax.set_yticklabels([f"{L}" for L in Ls], fontsize=9.5)
    ax.set_ylabel("languages", fontsize=9.5, color=S.MUTED)
    ax.xaxis.set_ticks_position("top")
    S.clean(ax, spines=()); ax.tick_params(length=0, colors=S.INK)
    ax.legend(handles=[Patch(facecolor=colours[2], label="finished"),
                       Patch(facecolor=colours[3], label="diverged (90M)"),
                       Patch(facecolor=colours[1], label="started"),
                       Patch(facecolor=colours[0], label="not started")],
              frameon=False, fontsize=8.5, labelcolor=S.MUTED, ncol=4,
              loc="upper center", bbox_to_anchor=(.5, -.03))
    S.title(fig, "The ladder as it stands: the top right corner is missing", y=1.05)
    S.save(fig, out / "grid_status.png")


def optimizer_timescale(out):
    """Why 90M diverged: run length against the optimizer's memory."""
    steps = {"90M": 4500, "175M": 8540, "350M": 16660, "600M": 28800,
             "1B": 45740, "1.7B": 81000}
    ratio = {k: v / 10000 for k, v in steps.items()}
    names = list(steps)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bars = ax.bar(names, [ratio[n] for n in names],
                  color=[S.SERIES[1] if n == "90M" else S.RAMP[1] for n in names], width=.62)
    ax.axhline(1, color=S.MUTED, ls="--", lw=1.2)
    for n, b in zip(names, bars):
        ax.annotate(f"{ratio[n]:.2f}×", (b.get_x() + b.get_width() / 2, b.get_height()),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5, color=S.INK)
    ax.set_ylabel("run length ÷ 10,000 steps", fontsize=9, color=S.MUTED)
    ax.set_ylim(0, 9.4)
    ax.grid(axis="y", color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax); ax.tick_params(length=0, labelsize=9.5)
    ax.legend(handles=[Patch(facecolor=S.SERIES[1], label="90M, 9 of 10 runs diverged"),
                       Patch(facecolor=S.RAMP[1], label="trains cleanly"),
                       Line2D([], [], color=S.MUTED, ls="--", lw=1.2,
                              label="the optimizer's memory, 10,000 steps")],
              frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="upper left")
    S.title(fig, "The 90M run is shorter than the optimizer it was given")
    S.save(fig, out / "optimizer_timescale.png")


def seed_holdout(out):
    """Does the SNR variant ranking survive a seed swap?"""
    h = pd.read_csv(ANALYSIS / "rq02_snr_definition/pretraining/"
                    "predictivity_seeds_train__vs__predictivity_seeds_test/headline_metrics.csv")
    v = {(r.metric, r.da_kind): r.value for r in h.itertuples()}
    labels = ["ranking of the 22 variants\n(Spearman)", "values, cell by cell\n(Pearson)",
              "how much of the best\ncorrelation is kept", "same family picked,\nper language"]
    keys = ["spearman_rank_global", "pearson_r_cells", "retention", "family_agreement"]
    size = [v[(k, "size")] for k in keys]
    ckpt = [v[(k, "ckpt")] for k in keys]

    y = np.arange(len(keys))[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    ax.barh(y + .19, ckpt, height=.34, color=S.SERIES[0], label="across checkpoints")
    ax.barh(y - .19, size, height=.34, color=S.SERIES[1], label="across sizes")
    for yi, a, b in zip(y, size, ckpt):
        ax.annotate(f"{b:.2f}", (b, yi + .19), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=S.INK)
        ax.annotate(f"{a:.2f}", (a, yi - .19), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=S.INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.8, color=S.INK)
    ax.set_xlim(0, 1.02); ax.set_xlabel("agreement between the two seed splits", fontsize=9,
                                        color=S.MUTED)
    ax.grid(axis="x", color=S.GRID, lw=.8); ax.set_axisbelow(True)
    S.clean(ax, spines=("bottom",)); ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="lower right")
    S.title(fig, "Swap the seeds and only the checkpoint-based ranking survives")
    S.save(fig, out / "seed_holdout.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    grid_status(OUT); optimizer_timescale(OUT); seed_holdout(OUT)
