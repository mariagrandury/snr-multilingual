"""The deck's appendix heatmaps, from the decision-accuracy table.

Two overviews first: one collapsed across languages (benchmark by size pair)
and one collapsed across benchmarks (language by size pair). Then the same
numbers as the appendix tables `da_per_benchmark.py` writes, one heatmap per
language: rows are benchmarks ordered most predictive first, columns are the
small→large size pairs, cells are decision accuracy. 0.5 is a coin flip, so
every scale diverges there: red below chance, blue above.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import style as S

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "signal-and-noise"))
from evals.scripts.utils.configs import bucket_order, load_languages  # noqa: E402

ANALYSIS = REPO / "src" / "signal-and-noise" / "analysis"
SRC = ANALYSIS / "rq01_decision_accuracy" / "pretraining" / "predictivity" / "da_per_benchmark.csv"
OUT = REPO / "documents" / "public" / "ladder" / "appendix"
BOLD = 0.75


def _comparison_key(comp):
    order = bucket_order()
    a, b = comp.split("→")
    bi = lambda x: order.index(x) if x in order else 99
    return bi(a), bi(b)


def _draw(grid, rows, comps, subtitle, path, note=None, cellw=1.25, collab=None):
    """Shared renderer: DA grid, diverging at the 0.5 coin flip."""
    nrow, ncol = grid.shape
    fig, ax = plt.subplots(figsize=(cellw * ncol + 3.6, 0.42 * nrow + 1.9))
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=S.DIV, vmin=0, vmax=1, aspect="auto")

    for i in range(nrow):
        for j in range(ncol):
            v = grid[i, j]
            if not np.isfinite(v):
                continue
            # White text only on the dark ends of the diverging ramp.
            shade = "white" if (v <= 0.18 or v >= 0.82) else S.INK
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=shade, fontweight="bold" if v >= BOLD else "normal")

    ax.set_xticks(range(ncol)); ax.set_xticklabels(collab or comps, fontsize=9)
    ax.set_yticks(range(nrow)); ax.set_yticklabels(rows, fontsize=9)
    ax.set_xticks(np.arange(-.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-.5, nrow, 1), minor=True)
    ax.grid(which="minor", color=S.SURFACE, lw=2)
    ax.tick_params(which="both", length=0, colors=S.INK)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                      ticks=[0, 0.25, 0.5, BOLD, 1])
    cb.ax.set_yticklabels(["0", "0.25", "0.5\ncoin flip", "0.75", "1"], fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=S.MUTED)
    if note:
        ax.set_xlabel(note, fontsize=8.5, color=S.MUTED, labelpad=8)
    # On a tall figure a suptitle at y=1 leaves a band of empty space, so the
    # title sits on the axes instead.
    ax.set_title(subtitle, fontsize=12, color=S.INK, pad=14)
    S.save(fig, path)


def _order(sub):
    return sorted(sub["comparison"].unique(), key=_comparison_key)


def shared_families():
    """How many model families each size pair actually compares. Two families
    means one pairwise call, so decision accuracy can only come out 0 or 1 —
    the column looks decisive and carries almost nothing."""
    from analysis.utils import build_snr_pool
    from evals.scripts.utils.configs import size_bucket
    pool = build_snr_pool("predictivity")
    pool["bucket"] = pool["size"].map(size_bucket)
    fams = {b: set(g["family"]) for b, g in pool.groupby("bucket")}
    return {f"{a}→{b}": len(fams.get(a, set()) & fams.get(b, set()))
            for a in fams for b in fams}


def _col_labels(comps, nfam):
    return [f"{c}\n{nfam.get(c, 0)} models" for c in comps]


def across_languages(size, out, nfam):
    """Benchmark by size pair, averaged over every language it covers."""
    comps = _order(size)
    piv = size.pivot_table(index="benchmark", columns="comparison",
                           values="decision_acc", aggfunc="mean").reindex(columns=comps)
    piv = (piv.assign(_m=piv.mean(axis=1))
           .sort_values("_m", ascending=False).drop(columns="_m"))
    n = size.groupby("benchmark")["language"].nunique()
    rows = [f"{b}  ({n[b]} lang)" for b in piv.index]
    _draw(piv.to_numpy(dtype=float), rows, comps,
          "Which benchmark predicts best, averaged over languages",
          out / "da_by_benchmark.png",
          "mean decision accuracy over every language the benchmark covers. "
          "A pair with 2 models can only read 0 or 1.",
          collab=_col_labels(comps, nfam))


def across_benchmarks(size, out, langs, names, nfam):
    """Language by size pair, averaged over every benchmark it has."""
    comps = _order(size)
    sub = size[size["language"].isin(langs)]
    piv = sub.pivot_table(index="language", columns="comparison",
                          values="decision_acc", aggfunc="mean").reindex(columns=comps)
    piv = piv.reindex([l for l in langs if l in piv.index])
    n = sub.groupby("language")["benchmark"].nunique()
    rows = [f"{names.get(l, l)} ({l})  ({n[l]})" for l in piv.index]
    _draw(piv.to_numpy(dtype=float), rows, comps,
          "Which language is easiest to predict, averaged over benchmarks",
          out / "da_by_language.png",
          "mean decision accuracy over every benchmark the language has "
          "(count in brackets). A pair with 2 models can only read 0 or 1.",
          collab=_col_labels(comps, nfam))


def language_heatmap(sub, lang, name, out, nfam):
    comps = _order(sub)
    wide = sub.pivot_table(index=["benchmark", "task"], columns="comparison",
                           values="decision_acc").reindex(columns=comps)
    wide = (wide.assign(_m=wide.mean(axis=1))
            .sort_values("_m", ascending=False).drop(columns="_m"))
    counts = wide.index.get_level_values("benchmark").value_counts()
    labels = [fam if counts[fam] == 1 else task for fam, task in wide.index]

    _draw(wide.to_numpy(dtype=float), labels, comps,
          f"{name} ({lang}): does the small model rank like the large one?",
          out / f"da_{lang}.png", collab=_col_labels(comps, nfam))


if __name__ == "__main__":
    long = pd.read_csv(SRC)
    size = long[long["da_def"] == "DA-size"]
    names = {c: e["language"] for c, e in load_languages()["languages"].items()}
    langs = [l for l in load_languages()["groups"]["trained"] if l in set(size["language"])]
    langs = sorted(langs, key=lambda l: (l != "en", l))
    OUT.mkdir(parents=True, exist_ok=True)
    nfam = shared_families()
    across_languages(size, OUT, nfam)
    across_benchmarks(size, OUT, langs, names, nfam)
    for lang in langs:
        language_heatmap(size[size["language"] == lang], lang,
                         names.get(lang, lang), OUT, nfam)
    print(f"2 overview + {len(langs)} per-language appendix heatmaps -> {OUT}")
