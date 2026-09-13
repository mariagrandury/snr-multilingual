"""For each benchmark family, count the languages whose top-5 SNR
benchmarks (at 1B, pooled `seeds_28_1797_1904` pool = 9 model_families
per size) include a task from that family.

Reads:  results/snr_definition/seeds_28_1797_1904/top_benchmarks_per_language.csv
Writes: results/benchmark_creation/seeds_28_1797_1904/top5_family_coverage.{csv,png}
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt
import pandas as pd

from multilingual.analyze_snr_variants import benchmark_family

POOL = "seeds_28_1797_1904"
SRC = _REPO / "results" / "snr_definition" / POOL / "top_benchmarks_per_language.csv"
OUT_DIR = _REPO / "results" / "benchmark_creation" / POOL
TOP_K = 5


def main() -> None:
    df = pd.read_csv(SRC)
    df = df[df["rank"] <= TOP_K].copy()
    df["family"] = df["task"].map(benchmark_family)

    n_langs = df["language"].nunique()
    coverage = (
        df.drop_duplicates(["family", "language"])
        .groupby("family")["language"]
        .agg(lambda s: (len(s), "|".join(sorted(s))))
        .apply(pd.Series)
    )
    coverage.columns = ["n_languages_in_top5", "languages"]
    coverage = coverage.sort_values("n_languages_in_top5", ascending=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(OUT_DIR / "top5_family_coverage.csv")

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(coverage) + 1.5))
    bars = ax.barh(coverage.index, coverage["n_languages_in_top5"],
                   color="#1f77b4", edgecolor="black", linewidth=0.4)
    for bar, (_, row) in zip(bars, coverage.iterrows()):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                row["languages"], va="center", fontsize=8, color="#444")
    ax.set_xlabel(f"# languages with task in top-{TOP_K} (out of {n_langs})")
    ax.set_xlim(0, n_langs + 0.5)
    ax.set_title(
        f"How often does each benchmark family appear in a language's top-{TOP_K}?\n"
        f"Pool: {POOL} (9 model_families per size at 1B)",
        fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top5_family_coverage.png", dpi=140)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'top5_family_coverage.csv'}")
    print(f"Wrote {OUT_DIR / 'top5_family_coverage.png'}")


if __name__ == "__main__":
    main()
