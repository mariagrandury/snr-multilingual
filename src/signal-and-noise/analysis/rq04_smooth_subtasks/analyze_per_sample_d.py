"""Reuse the committed Option-D per-sample outputs — no raw samples needed.

The raw ``samples_*.jsonl`` (hence the per-checkpoint acc matrix) are
cluster-only, so the cumulative-SNR sweep — and therefore methods **B**
(``forward_greedy``) and **C** (``irt_discrimination``) — cannot be recomputed
locally. Method **A** (``greedy_snr_rank``) is provably identical to **D**: the
variance prefilter only drops samples with zero cross-mix signal, which carry
~0 per-sample SNR and sort to the tail of the sweep, so they never enter the
argmax subset.

What the D intermediates under
``results/smooth_subtasks/per_sample/variance_prefilter/`` *do* support, all
computed here:

  1. **Informative-sample fraction** — ``n_candidates / n_total``: how many
     items carry any cross-mix signal at all.
  2. **SNR gain distribution** — best subset vs full set (``snr_gain``).
  3. **Cross-size subset stability** — Jaccard of the selected ``best_subset``
     doc-ids between model sizes: are the most-discriminating items shared as
     the model scales?
  4. **Cross-size SNR-rank stability** — Spearman of the per-sample ``snr_<size>``
     columns between sizes: does an item keep its SNR ranking across scale?

Outputs land under ``variance_prefilter/analysis/``.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from snr.constants import PLOT_DIR
from analysis.paths import SMOOTH_SUBTASKS

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from evals.scripts.utils.configs import load_snr_params  # noqa: E402

# Per-sample committed intermediates are custom-only (the 4-size ladder).
_SNR = load_snr_params()
D_ROOT = SMOOTH_SUBTASKS / "per_sample" / "variance_prefilter"
SIZES = _SNR["small_sizes"] + [_SNR["target_size"]]


def _read_subset(task_dir: Path, size: str):
    p = task_dir / f"best_subset_{size}.txt"
    if not p.exists():
        return None
    toks = p.read_text().split()
    return {int(x) for x in toks} if toks else None


def load_summary(root: Path) -> pd.DataFrame:
    # A fresh run writes the roll-up to variance_prefilter/summary_all.csv; the
    # committed legacy tree left it one level up at per_sample/summary_all.csv
    # (task dirs were moved into variance_prefilter/ but the roll-up wasn't).
    summary = root / "summary_all.csv"
    if not summary.exists():
        summary = root.parent / "summary_all.csv"
    df = pd.read_csv(summary)
    # The writer renamed this column n_after_prefilter → n_candidates; the
    # committed legacy roll-up still carries the old name.
    df = df.rename(columns={"n_after_prefilter": "n_candidates"})
    df = df[df["status"] == "ok"].copy()
    df["informative_frac"] = df["n_candidates"] / df["n_total"]
    df["subset_frac"] = df["best_n"] / df["n_total"]
    return df


def per_size_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for size in SIZES + ["ALL"]:
        s = df if size == "ALL" else df[df["size"] == size]
        if s.empty:
            continue
        rows.append({
            "size": size, "n_cells": len(s),
            "median_informative_pct": round(float(s["informative_frac"].median()) * 100, 1),
            "median_subset_pct": round(float(s["subset_frac"].median()) * 100, 1),
            "median_gain": round(float(s["snr_gain"].median()), 3),
            "max_gain": round(float(s["snr_gain"].max()), 3),
            "pct_gain_pos": round(float((s["snr_gain"] > 0).mean()) * 100, 1),
        })
    return pd.DataFrame(rows)


def cross_size_subset_stability(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for (lang, task), _ in df.groupby(["language", "task"]):
        subsets = {s: _read_subset(root / lang / task, s) for s in SIZES}
        subsets = {s: v for s, v in subsets.items() if v}
        for a, b in combinations([s for s in SIZES if s in subsets], 2):
            sa, sb = subsets[a], subsets[b]
            inter, union = len(sa & sb), len(sa | sb)
            rows.append({
                "language": lang, "task": task, "pair": f"{a}|{b}",
                "n_a": len(sa), "n_b": len(sb), "inter": inter,
                "jaccard": inter / union if union else np.nan,
                "overlap_coef": inter / min(len(sa), len(sb)),
            })
    return pd.DataFrame(rows)


def cross_size_snr_corr(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows = []
    for (lang, task), _ in df.groupby(["language", "task"]):
        p = root / lang / task / "ranked_samples.csv"
        if not p.exists():
            continue
        rs = pd.read_csv(p)
        for a, b in combinations(SIZES, 2):
            ca, cb = f"snr_{a}", f"snr_{b}"
            if ca not in rs or cb not in rs:
                continue
            m = rs[ca].notna() & rs[cb].notna()
            if m.sum() < 5:
                continue
            rows.append({
                "language": lang, "task": task, "pair": f"{a}|{b}",
                "n": int(m.sum()),
                "spearman": round(float(rs.loc[m, ca].corr(rs.loc[m, cb], method="spearman")), 3),
            })
    return pd.DataFrame(rows)


def _pair_median(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty:
        return df
    g = (df.groupby("pair")[col].median().round(3)
         .reindex([f"{a}|{b}" for a, b in combinations(SIZES, 2)]).dropna())
    return g.rename(f"median_{col}").reset_index()


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_highlights(path: Path, dist, stab, corr, df):
    overall = dist[dist["size"] == "ALL"].iloc[0]
    top = (df.sort_values("snr_gain", ascending=False).head(10)
           [["language", "task", "size", "n_total", "n_candidates",
             "best_n", "full_set_snr", "best_snr", "snr_gain"]]
           .round({"full_set_snr": 2, "best_snr": 2, "snr_gain": 2}))
    stab_med = _pair_median(stab, "jaccard")
    corr_med = _pair_median(corr, "spearman")
    j_all = float(stab["jaccard"].median()) if not stab.empty else float("nan")
    sp_all = float(corr["spearman"].median()) if not corr.empty else float("nan")

    L = []
    L.append("# Per-sample (Option D) — highlights from the committed intermediates\n")
    L.append(f"_{len(df)} (language, benchmark, size) cells across "
             f"{df['task'].nunique()} benchmarks, reusing "
             f"`variance_prefilter/` outputs (no raw samples)._\n")
    L.append("> **Scope.** Methods **B** and **C** need the per-checkpoint acc "
             "matrix (cluster-only) and are not reconstructable here; **A** is "
             "provably identical to **D**. Per-sample SNR is on binary item "
             "accuracy — not comparable to subtask-level (Case 1–3) SNR.\n")

    L.append("## Headline\n")
    L.append(f"- A per-sample subset beats the full set in "
             f"**{overall['pct_gain_pos']:.0f}%** of cells; median gain "
             f"**+{overall['median_gain']}**, max **+{overall['max_gain']}**.")
    L.append(f"- Only a median of **{overall['median_informative_pct']:.0f}%** of "
             f"items carry any cross-mix signal (the rest are dead and dropped).")
    L.append(f"- The winning subset is tiny — a median of "
             f"**{overall['median_subset_pct']:.0f}%** of all items.")
    L.append(f"- **Neither the subset nor the ranking transfers across scale.** "
             f"Median best-subset Jaccard across sizes is **{j_all:.2f}** and "
             f"median per-sample SNR-rank Spearman is **{sp_all:.2f}** — both "
             f"≈0. Per-sample SNR is estimated from only ~5 ckpts × 3 mixes per "
             f"item, so this near-zero transfer likely reflects estimation noise "
             f"as much as true scale-dependence, and argues for the subtask-level "
             f"(Case 1–3) approach over per-item selection.\n")

    L.append("## Distribution by size\n")
    L.append(_md(dist) + "\n")

    L.append("## Biggest gains (top 10 cells)\n")
    L.append(_md(top) + "\n")

    L.append("## Cross-size stability of the selected subset (Jaccard of best_subset doc-ids)\n")
    if not stab_med.empty:
        L.append(_md(stab_med) + "\n")
        L.append("Low Jaccard = the items that best separate data mixtures differ "
                 "by model size — a subset tuned at a small size won't transfer "
                 "verbatim to a larger one.\n")

    L.append("## Cross-size stability of per-sample SNR ranking (Spearman of snr_<size>)\n")
    if not corr_med.empty:
        L.append(_md(corr_med) + "\n")
        L.append("All ≈0: an item's SNR ranking does not survive a change in model "
                 "size. With only ~5 ckpts × 3 mixes behind each item's SNR, the "
                 "per-sample estimate is itself very noisy, so this is a lower "
                 "bound on true scale-dependence rather than proof of it — but "
                 "either way per-item subsets don't transfer across scale.\n")

    L.append("## Files\n")
    L.append("- `size_distribution.csv` — the per-size table above.")
    L.append("- `cross_size_subset_jaccard.csv` — per (task, size-pair) subset overlap.")
    L.append("- `cross_size_snr_spearman.csv` — per (task, size-pair) SNR-rank correlation.")
    path.write_text("\n".join(L) + "\n")


def main(root: Path):
    df = load_summary(root)
    dist = per_size_distribution(df)
    stab = cross_size_subset_stability(df, root)
    corr = cross_size_snr_corr(df, root)

    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    dist.to_csv(out / "size_distribution.csv", index=False)
    stab.to_csv(out / "cross_size_subset_jaccard.csv", index=False)
    corr.to_csv(out / "cross_size_snr_spearman.csv", index=False)
    write_highlights(out / "highlights.md", dist, stab, corr, df)

    print(f"Analyzed {len(df)} cells / {df['task'].nunique()} benchmarks.")
    print(f"Wrote → {out}/(size_distribution.csv, cross_size_subset_jaccard.csv, "
          f"cross_size_snr_spearman.csv, highlights.md)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-root", type=Path, default=D_ROOT,
                   help="variance_prefilter/ dir holding summary_all.csv + "
                        "<lang>/<task>/ subdirs (default: the committed one).")
    args = p.parse_args()
    main(root=args.d_root)
