"""Per-language benchmark figures: which benchmark works, in which language."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import data, style as S

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "src" / "signal-and-noise" / "analysis"
OUT = REPO / "documents" / "public" / "ladder"
sys.path.insert(0, str(REPO / "src" / "signal-and-noise"))
from analysis.utils import assign_language, benchmark_family  # noqa: E402


def first_clearing_size(out):
    """Rows = benchmark family, columns = language, cell = the smallest model that
    beats chance. Grey where the family never does, blank where there is no task."""
    m = pd.read_csv(ANALYSIS / "rq00_acc_vs_flops/pretraining/predictivity/above_random_mask.csv")
    m = m[m["n_options"].notna()]
    m["lang"] = m["task"].map(assign_language)
    m["fam"] = m["task"].map(benchmark_family)
    # 90M is excluded: one cell survives the divergence filter and it is off
    # trend, so "clears chance at 90M" would rest on a single broken run.
    sizes = [s for s in S.SIZES if s in m.columns and s != "90M"]
    langs = [l for l in data.TRAINED_LANGS if l in set(m["lang"])]

    grid = np.full((0, len(langs)), np.nan)
    fams, rows = [], []
    for fam, g in m.groupby("fam"):
        row = np.full(len(langs), np.nan)
        for j, lang in enumerate(langs):
            cell = g[g["lang"] == lang]
            if cell.empty:
                continue
            row[j] = -1                                  # a task exists, never clears
            for i, s in enumerate(sizes):
                if (cell[s] == 1).any():
                    row[j] = i
                    break
        if np.isfinite(row).any():
            fams.append(fam); rows.append(row)
    grid = np.array(rows)
    order = np.argsort([np.nansum(r >= 0) for r in grid])[::-1]
    grid, fams = grid[order], [fams[i] for i in order]

    # Past ~20 columns the size no longer fits inside a cell, so the legend
    # carries it alone and the figure grows with the column count instead.
    ncols = grid.shape[1]
    annotate = ncols <= 20
    fig, ax = plt.subplots(figsize=(max(9.2, 0.26 * ncols + 2.6), 0.42 * len(fams) + 1.8))
    colors = [S.SIZE_COLOR[s] for s in sizes]
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if not np.isfinite(v):
                continue
            face = S.NODATA if v < 0 else colors[int(v)]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=face, edgecolor="white", lw=1.2))
            if annotate and v >= 0:
                ax.text(j + .5, i + .5, sizes[int(v)], ha="center", va="center",
                        fontsize=7.2, color="white" if v >= 2 else S.INK)
    ax.set_xlim(0, grid.shape[1]); ax.set_ylim(grid.shape[0], 0)
    ax.set_xticks(np.arange(grid.shape[1]) + .5); ax.set_xticklabels(langs, fontsize=9)
    ax.set_yticks(np.arange(grid.shape[0]) + .5); ax.set_yticklabels(fams, fontsize=9)
    ax.xaxis.set_ticks_position("top"); ax.xaxis.set_label_position("top")
    S.clean(ax, spines=())
    ax.tick_params(length=0, colors=S.INK)
    handles = [Patch(facecolor=c, label=s) for s, c in zip(sizes, colors)]
    handles.append(Patch(facecolor=S.NODATA, label="never beats chance"))
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=S.MUTED, ncol=len(handles),
              loc="upper center", bbox_to_anchor=(0.5, -0.02 - 0.5 / len(fams)))
    S.title(fig, "Smallest model that beats chance, per benchmark and language",
            y=0.99)
    S.save(fig, OUT / "first_clearing_size.png")


def snr_bpb_vs_benchmark(out):
    """Two panels. Left: how many languages still have a benchmark after the gate.
    Right: where one survives, does it beat that language's bits per byte?"""
    t = pd.read_csv(ANALYSIS / "rq02_snr_definition/pretraining/predictivity/"
                    "top_benchmarks_per_language.csv")
    t = t[t["language"] != "multi"]
    rows = []
    for lang, d in t.groupby("language"):
        bench = d[~d["task"].str.startswith("bpb_")]
        bpb = d[d["task"].str.startswith("bpb_")]["snr"].max()
        rows.append({"lang": lang, "bpb": bpb,
                     "bench": bench["snr"].max() if not bench.empty else np.nan,
                     "task": (bench.sort_values("snr", ascending=False)["task"].iloc[0]
                              if not bench.empty else "")})
    d = pd.DataFrame(rows)
    head = d.dropna(subset=["bench"]).sort_values("bench", ascending=True)
    n_none = int(d["bench"].isna().sum())

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.0, 3.6),
                                 gridspec_kw={"width_ratios": [1, 2.1]})

    # left: the coverage fact
    a0.barh([0], [n_none], color=S.NODATA, height=.55)
    a0.barh([0], [len(head)], left=[n_none], color=S.RAMP[1], height=.55)
    a0.text(n_none / 2, 0, f"{n_none}", ha="center", va="center", fontsize=16, color=S.INK)
    a0.text(n_none + len(head) / 2, 0, f"{len(head)}", ha="center", va="center",
            fontsize=13, color="white")
    a0.text(n_none / 2, .42, "no benchmark left\nafter the gate", ha="center", va="bottom",
            fontsize=8.5, color=S.MUTED)
    a0.text(n_none + len(head) / 2, .42, "at least one\nsurvives", ha="center", va="bottom",
            fontsize=8.5, color=S.MUTED)
    a0.set_xlim(0, len(d)); a0.set_ylim(-.5, 1.2); a0.axis("off")
    a0.set_title(f"{len(d)} validation languages", fontsize=10, color=S.INK, pad=2)

    # right: the head to head
    y = np.arange(len(head))
    for yi, r in zip(y, head.itertuples()):
        lo, hi = sorted((r.bpb, r.bench))
        a1.plot([lo, hi], [yi, yi], color="#dfe8f5", lw=3, solid_capstyle="round", zorder=2)
        a1.plot(r.bpb, yi, "o", ms=9, color=S.RAMP[3], zorder=4)
        a1.plot(r.bench, yi, "o", ms=9, color=S.RAMP[0], zorder=3)
        a1.annotate(r.task, (max(r.bpb, r.bench), yi), textcoords="offset points",
                    xytext=(10, 0), fontsize=7.5, color=S.MUTED, va="center")
    a1.set_yticks(y); a1.set_yticklabels(head["lang"], fontsize=10, color=S.INK)
    a1.set_xlabel("signal to noise ratio at 1B", fontsize=9, color=S.MUTED)
    a1.set_xlim(0, float(head[["bpb", "bench"]].to_numpy().max()) * 1.55)
    a1.grid(axis="x", color=S.GRID, lw=0.8); a1.set_axisbelow(True)
    S.clean(a1, spines=("bottom",)); a1.tick_params(length=0)
    n_bench_wins = int((head["bench"] > head["bpb"]).sum())
    a1.set_title(f"and where one survives it wins, in {n_bench_wins} of {len(head)}",
                 fontsize=10, color=S.INK, pad=2)
    a1.legend(handles=[Patch(color=S.RAMP[3], label="bits per byte"),
                       Patch(color=S.RAMP[0], label="best benchmark")],
              frameon=False, fontsize=8.5, labelcolor=S.MUTED, loc="lower right")
    S.title(fig, "Bits per byte wins almost everywhere because almost nothing else is left",
            y=1.04)
    S.save(fig, OUT / "snr_bpb_vs_benchmark.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    first_clearing_size(OUT)
    snr_bpb_vs_benchmark(OUT)
