#!/usr/bin/env python3
"""Generate analysis plots from SNR results.

Creates plots for signal, noise, SNR, and DA, saved under results/analysis/.

Usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --input results/snr/snr.csv --output results/analysis/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.plots import (
    benchmark_ranking,
    signal_noise_scatter,
    snr_distribution,
    snr_vs_da_pair,
)


def _save(fig, output_dir: Path, fname: str) -> None:
    if fig is None:
        return
    path = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def main():
    parser = argparse.ArgumentParser(description="Generate SNR analysis plots")
    parser.add_argument("--input", type=str, default="results/snr/snr.csv")
    parser.add_argument("--output", type=str, default="results/analysis/")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(df)} rows")
    print(f"  Categories: {sorted(df['category'].unique())}")
    print(f"  Size groups: {sorted(df['size_group'].unique())}")

    da_cols = [c for c in df.columns if c.startswith("da_")]

    for category in sorted(df["category"].unique()):
        print(f"\n--- {category} ---")
        cat_df = df[df["category"] == category]

        _save(signal_noise_scatter(cat_df, title=f"Signal vs Noise ({category})"),
              output_dir, f"signal_noise_{category}.png")
        _save(snr_distribution(cat_df, title=f"SNR Distribution ({category})"),
              output_dir, f"snr_dist_{category}.png")

        for metric in ("snr", "signal", "noise"):
            _save(benchmark_ranking(cat_df, metric=metric,
                                    title=f"Top tasks by {metric.upper()} ({category})",
                                    top_n=25),
                  output_dir, f"ranking_{metric}_{category}.png")
        for da_col in da_cols:
            _save(benchmark_ranking(cat_df, metric=da_col,
                                    title=f"Top tasks by {da_col} ({category})",
                                    top_n=25),
                  output_dir, f"ranking_{da_col}_{category}.png")

        sizes = sorted(cat_df["size_group"].unique())
        for da_col in da_cols:
            target = da_col.removeprefix("da_")
            for small in sizes:
                if small == target:
                    continue
                fig = snr_vs_da_pair(cat_df, small, target)
                if fig is not None:
                    fname = f"snr_vs_da_{category}_{small}_{target}.png".replace("~", "")
                    _save(fig, output_dir, fname)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
