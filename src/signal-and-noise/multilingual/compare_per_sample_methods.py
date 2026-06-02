"""Compare the per-sample subset-search proposers (A/B/C/D) and extract
paper-ready highlights.

Reads each method's roll-up
``results/smooth_subtasks/per_sample/<method>/summary_all.csv`` (produced
by ``smooth_subtasks_per_sample.py``) and writes, under
``per_sample/comparison/``:

  - ``method_comparison.csv`` — one row per (language, task, size) with each
    method's best_snr / snr_gain / best_n / n_candidates side by side, plus
    the winning method and its gain.
  - ``method_summary.csv`` — per-method aggregates (win rate, median gain,
    subset size, candidate count).
  - ``highlights.md`` — the narrative + tables worth lifting into a paper.

Subset-overlap stats (A vs D, A vs C) read the per-(lang, task, size)
``best_subset_<size>.txt`` files when present, to show whether the methods
pick the *same* samples or just reach similar SNR by different routes.

Usage:
    python multilingual/compare_per_sample_methods.py
    python multilingual/compare_per_sample_methods.py --methods A,D
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from snr.constants import PLOT_DIR

PER_SAMPLE_ROOT = PLOT_DIR / "smooth_subtasks" / "per_sample"

# dir name → short label; order is the canonical display / tie-break order.
METHODS_ALL = {
    "variance_prefilter": "D",
    "greedy_snr_rank": "A",
    "forward_greedy": "B",
    "irt_discrimination": "C",
}
LABEL_TO_DIR = {v: k for k, v in METHODS_ALL.items()}
KEYS = ["language", "task", "size"]
METRICS = ["best_snr", "snr_gain", "best_n", "n_candidates"]
SHARED = ["full_set_snr", "n_total"]  # method-independent
TIE = 1e-6


def load_summaries(root: Path, method_dirs: list[str]) -> dict[str, pd.DataFrame]:
    """{method_dir: ok-rows of its summary_all.csv} for methods that ran."""
    out = {}
    for m in method_dirs:
        p = root / m / "summary_all.csv"
        if p.exists():
            df = pd.read_csv(p)
            out[m] = df[df["status"] == "ok"].copy()
    return out


def build_comparison(summaries: dict[str, pd.DataFrame]):
    """Wide table: one row per (language, task, size), each method's metrics
    side by side, plus the winning method (by snr_gain) and its stats."""
    methods = list(summaries)
    merged = None
    for m in methods:
        cols = KEYS + METRICS + SHARED
        df = summaries[m][cols].rename(columns={c: f"{c}__{m}" for c in METRICS + SHARED})
        merged = df if merged is None else merged.merge(df, on=KEYS, how="outer")

    # full_set_snr / n_total are method-independent — coalesce to one column.
    for c in SHARED:
        merged[c] = merged[[f"{c}__{m}" for m in methods]].bfill(axis=1).iloc[:, 0]
        merged = merged.drop(columns=[f"{c}__{m}" for m in methods])

    gains = merged[[f"snr_gain__{m}" for m in methods]].to_numpy(float)
    finite = np.isfinite(gains)
    any_finite = finite.any(axis=1)
    best_idx = np.where(finite, gains, -np.inf).argmax(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN rows → NaN, not a crash
        best_gain = np.nanmax(np.where(finite, gains, np.nan), axis=1)
    best_gain = np.where(any_finite, best_gain, np.nan)

    def pick(colbase):
        return [merged.iloc[r][f"{colbase}__{methods[i]}"] if ok else np.nan
                for r, (i, ok) in enumerate(zip(best_idx, any_finite))]

    merged["best_method"] = [METHODS_ALL[methods[i]] if ok else "—"
                             for i, ok in zip(best_idx, any_finite)]
    merged["best_gain"] = best_gain
    merged["best_snr"] = pick("best_snr")
    merged["best_subset_n"] = pick("best_n")
    return merged.sort_values("best_gain", ascending=False), methods


def method_aggregates(merged: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    """Per-method win rate, gain distribution, and subset-size stats."""
    gmat = merged[[f"snr_gain__{m}" for m in methods]].to_numpy(float)
    rowmax = np.where(np.isfinite(gmat), gmat, -np.inf).max(axis=1)
    valid = np.isfinite(rowmax)
    denom = max(int(valid.sum()), 1)

    rows = []
    for j, m in enumerate(methods):
        g = merged[f"snr_gain__{m}"]
        bn = merged[f"best_n__{m}"]
        frac = bn / merged["n_total"]
        wins = np.isfinite(gmat[:, j]) & (gmat[:, j] >= rowmax - TIE)
        rows.append({
            "method": METHODS_ALL[m], "dir": m,
            "n_cells": int(g.notna().sum()),
            "median_gain": round(float(g.median()), 3),
            "mean_gain": round(float(g.mean()), 3),
            "pct_gain_pos": round(float((g > 0).mean() * 100), 1),
            "win_rate_pct": round(float(wins[valid].sum()) / denom * 100, 1),
            "median_best_n": round(float(bn.median()), 1),
            "median_subset_frac_pct": round(float(frac.median() * 100), 1),
            "mean_candidates": round(float(merged[f"n_candidates__{m}"].mean()), 0),
        })
    # Display in canonical A,B,C,D order.
    order = {lab: i for i, lab in enumerate(["A", "B", "C", "D"])}
    return pd.DataFrame(rows).sort_values("method", key=lambda s: s.map(order))


def _read_best_subset(root, method_dir, language, task, size):
    p = root / method_dir / language / task / f"best_subset_{size}.txt"
    if not p.exists():
        return None
    txt = p.read_text().split()
    return {int(x) for x in txt} if txt else None


def overlap_stats(merged, root, methods, pairs):
    """Median Jaccard / overlap-coefficient of the best subsets per method pair."""
    rows = []
    for a, b in pairs:
        if a not in methods or b not in methods:
            continue
        jac, ovc = [], []
        for _, r in merged.iterrows():
            sa = _read_best_subset(root, a, r["language"], r["task"], r["size"])
            sb = _read_best_subset(root, b, r["language"], r["task"], r["size"])
            if not sa or not sb:
                continue
            inter = len(sa & sb)
            jac.append(inter / len(sa | sb))
            ovc.append(inter / min(len(sa), len(sb)))
        if jac:
            rows.append({
                "pair": f"{METHODS_ALL[a]} vs {METHODS_ALL[b]}",
                "n_cells": len(jac),
                "median_jaccard": round(float(np.median(jac)), 3),
                "median_overlap_coef": round(float(np.median(ovc)), 3),
            })
    return pd.DataFrame(rows)


def forward_greedy_extra(merged, methods):
    """How much B beats the best per-rank method (A/C/D) per cell."""
    if "forward_greedy" not in methods:
        return None
    others = [m for m in methods if m != "forward_greedy"]
    if not others:
        return None
    b = merged["snr_gain__forward_greedy"].to_numpy(float)
    omat = merged[[f"snr_gain__{m}" for m in others]].to_numpy(float)
    obest = np.where(np.isfinite(omat), omat, -np.inf).max(axis=1)
    valid = np.isfinite(b) & np.isfinite(obest)
    if not valid.any():
        return None
    delta = b[valid] - obest[valid]
    return {"n": int(valid.sum()),
            "median_delta": round(float(np.median(delta)), 3),
            "mean_delta": round(float(np.mean(delta)), 3),
            "pct_b_strictly_best": round(float((delta > TIE).mean() * 100), 1)}


def gain_corr(merged, a_dir, c_dir):
    """Correlation of per-cell snr_gain between two methods."""
    a = merged.get(f"snr_gain__{a_dir}")
    c = merged.get(f"snr_gain__{c_dir}")
    if a is None or c is None:
        return None
    m = a.notna() & c.notna()
    if m.sum() < 3:
        return None
    return {"n": int(m.sum()),
            "pearson": round(float(a[m].corr(c[m])), 3),
            "spearman": round(float(a[m].corr(c[m], method="spearman")), 3)}


def _md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_highlights(path, merged, agg, ov, bextra, cacorr, methods):
    labels = [METHODS_ALL[m] for m in methods]
    n_cells = int(merged["best_gain"].notna().sum())
    pct_pos = float((merged["best_gain"] > 0).mean() * 100)

    rec = "variance_prefilter" if "variance_prefilter" in methods else methods[0]
    rec_label = METHODS_ALL[rec]
    rg = merged[f"snr_gain__{rec}"]
    rec_row = merged.loc[rg.idxmax()] if rg.notna().any() else None
    rec_med = float(rg.median())
    rec_frac = float((merged[f"best_n__{rec}"] / merged["n_total"]).median() * 100)

    top = merged.head(15)[[
        "language", "task", "size", "n_total", "full_set_snr",
        "best_method", "best_snr", "best_gain", "best_subset_n",
    ]].round({"full_set_snr": 2, "best_snr": 2, "best_gain": 2})

    L = []
    L.append("# Per-sample subset selection — method comparison & paper highlights\n")
    L.append(f"_{n_cells} (language, benchmark, size) cells; methods compared: "
             f"{', '.join(labels)}._\n")
    L.append("> Generated by `multilingual/compare_per_sample_methods.py`. Per-sample "
             "SNR is on binary item accuracy, so absolute values are not comparable "
             "to the subtask-level (Case 1–3) SNR.\n")

    L.append("## Headline\n")
    L.append(f"- A per-sample subset **beats the full set** in **{pct_pos:.0f}%** "
             f"of cells (best-of-methods `snr_gain > 0`).")
    if rec_row is not None:
        L.append(f"- Recommended cheap method **{rec_label} (`{rec}`)**: median SNR "
                 f"gain **{rec_med:+.2f}**, max **{float(rec_row[f'snr_gain__{rec}']):+.2f}** "
                 f"on `{rec_row['task']}` {rec_row['size']} "
                 f"(full {float(rec_row['full_set_snr']):.2f} → "
                 f"{float(rec_row[f'best_snr__{rec}']):.2f}).")
    L.append(f"- The best subset uses a median of **{rec_frac:.0f}%** of the benchmark's "
             f"samples ({rec_label}).\n")

    L.append("## Per-method summary\n")
    L.append(_md(agg) + "\n")
    L.append("- `win_rate_pct` — share of cells where the method's gain is the best "
             "(ties counted). `median_subset_frac_pct` — best subset size as a "
             "fraction of all samples. `mean_candidates` — samples fed to the sweep "
             "(after each method's prefilter/cap).\n")

    L.append("## Biggest gains (top 15 cells)\n")
    L.append(_md(top) + "\n")

    L.append("## Method trade-offs\n")
    if bextra:
        if bextra["median_delta"] > 0.05:
            verdict = "interactions matter — keep B in the loop"
        elif bextra["pct_b_strictly_best"] > 50:
            verdict = ("B is nominally best almost everywhere but by a negligible "
                       "median margin — A/D are within ε, so prefer them for cost")
        else:
            verdict = "interactions add little — the cheap rank methods suffice"
        L.append(f"- **B (`forward_greedy`)** beats the best per-rank method by a median "
                 f"of **{bextra['median_delta']:+.2f}** SNR (mean {bextra['mean_delta']:+.2f}); "
                 f"strictly best in **{bextra['pct_b_strictly_best']:.0f}%** of cells "
                 f"(n={bextra['n']}). → {verdict}.")
    if cacorr:
        verdict = ("IRT discrimination tracks per-sample SNR"
                   if cacorr["spearman"] > 0.5
                   else "IRT discrimination and per-sample SNR disagree — expected, "
                        "given the thin/correlated checkpoint examinee pool")
        L.append(f"- **C (IRT) vs A**: per-cell gain Pearson r=**{cacorr['pearson']}**, "
                 f"Spearman=**{cacorr['spearman']}** (n={cacorr['n']}). → {verdict}.")
    if ov is not None and not ov.empty:
        L.append("- **Best-subset overlap** (do methods pick the *same* samples?):\n")
        L.append(_md(ov) + "\n")
        ad = ov[ov["pair"] == "A vs D"]
        if not ad.empty:
            j = float(ad.iloc[0]["median_jaccard"])
            L.append(f"  - A vs D median Jaccard **{j:.2f}** — the variance prefilter "
                     f"{'leaves picks essentially unchanged (pure speed-up)' if j > 0.8 else 'shifts the selected subset'}.")
    L.append("")

    L.append("## Files\n")
    L.append("- `method_comparison.csv` — full per-cell table (sorted by best gain).")
    L.append("- `method_summary.csv` — the per-method aggregates above.")
    path.write_text("\n".join(L) + "\n")


def main(root: Path, method_dirs: list[str]):
    summaries = load_summaries(root, method_dirs)
    if not summaries:
        raise SystemExit(
            f"no summary_all.csv found under {root}/<method>/ for {method_dirs}. "
            f"Run smooth_subtasks_per_sample.py first."
        )
    if len(summaries) < 2:
        print(f"warning: only one method present ({list(summaries)}); "
              f"cross-method stats will be trivial.")
    merged, methods = build_comparison(summaries)
    agg = method_aggregates(merged, methods)

    out = root / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out / "method_comparison.csv", index=False)
    agg.to_csv(out / "method_summary.csv", index=False)

    ov = overlap_stats(merged, root, methods,
                       [("greedy_snr_rank", "variance_prefilter"),
                        ("greedy_snr_rank", "irt_discrimination")])
    bextra = forward_greedy_extra(merged, methods)
    cacorr = gain_corr(merged, "greedy_snr_rank", "irt_discrimination")
    write_highlights(out / "highlights.md", merged, agg, ov, bextra, cacorr, methods)

    print(f"Compared {len(methods)} methods over {len(merged)} cells.")
    print(f"Wrote → {out}/(method_comparison.csv, method_summary.csv, highlights.md)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-sample-root", type=Path, default=PER_SAMPLE_ROOT,
                   help="Dir holding the <method>/ subdirs (default: "
                        "results/smooth_subtasks/per_sample/).")
    p.add_argument("--methods", default="all",
                   help="'all', or a comma list of letters (A/B/C/D) or dir names.")
    args = p.parse_args()

    if args.methods.strip().lower() == "all":
        dirs = list(METHODS_ALL)
    else:
        dirs = []
        for tok in (t.strip() for t in args.methods.split(",")):
            if tok in METHODS_ALL:
                dirs.append(tok)
            elif tok.upper() in LABEL_TO_DIR:
                dirs.append(LABEL_TO_DIR[tok.upper()])
            else:
                p.error(f"unknown method {tok!r}")
    main(root=args.per_sample_root, method_dirs=dirs)
