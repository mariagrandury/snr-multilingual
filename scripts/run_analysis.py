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
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SIZE_GROUP_COLORS = {
    "~0.6B": "#1f77b4",
    "~1.5B": "#2ca02c",
    "~3B":   "#ff7f0e",
    "~8B":   "#d62728",
    "~32B":  "#9467bd",
    "~70B":  "#8c564b",
}


def plot_snr_vs_da(df, small_group, target_group, category, output_dir):
    """SNR vs DA scatter for one (small_group → target_group, category) pair."""
    da_col = f"da_{target_group}"
    if da_col not in df.columns:
        return
    sub = df[(df["size_group"] == small_group) & (df["category"] == category)].copy()
    sub = sub.dropna(subset=["snr", da_col])
    sub = sub[(sub["snr"] > 0) & np.isfinite(sub["snr"])]
    if len(sub) < 5:
        return

    x = sub["snr"].values
    y = sub[da_col].values

    fig, ax = plt.subplots(figsize=(6, 5))
    color = SIZE_GROUP_COLORS.get(small_group, "gray")
    ax.scatter(x, y, s=12, alpha=0.7, color=color, edgecolors="none")

    # Fit in log(SNR) vs DA
    x_log = np.log10(x)
    valid = np.isfinite(x_log) & np.isfinite(y)
    if valid.sum() >= 3:
        x_log_v, y_v = x_log[valid], y[valid]
        z = np.polyfit(x_log_v, y_v, 1)
        p = np.poly1d(z)
        x_line = np.logspace(np.log10(x[valid].min()), np.log10(x[valid].max()), 100)
        y_line = p(np.log10(x_line))

        n = len(x_log_v)
        x_mean = np.mean(x_log_v)
        s_err = np.sqrt(np.sum((y_v - p(x_log_v)) ** 2) / max(n - 2, 1))
        conf = (
            stats.t.ppf(0.975, max(n - 2, 1)) * s_err
            * np.sqrt(1 / n + (np.log10(x_line) - x_mean) ** 2 / np.sum((x_log_v - x_mean) ** 2))
        )
        r = np.corrcoef(x_log_v, y_v)[0, 1]
        r2 = r ** 2
        stderr = s_err * np.sqrt((1 - r2) / max(n - 2, 1))

        ax.plot(x_line, y_line, "--", color="black", alpha=0.5)
        ax.fill_between(x_line, y_line - conf, y_line + conf, color="gray", alpha=0.2)
        ax.text(0.03, 0.97, f"R = {r:.3f} ± {stderr:.3f}\nR² = {r2:.3f}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top")

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel("SNR = Rel. Dispersion / Rel. Std.", fontsize=12)
    ax.set_ylabel("Decision Accuracy", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax.set_title(f"SNR vs DA  ({category}, {small_group} → {target_group})", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")
    fig.tight_layout()

    fname = f"snr_vs_da_{category}_{small_group}_{target_group}.png".replace("~", "")
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}  (n={len(sub)}, R={r:.3f})" if valid.sum() >= 3 else f"  Saved {fname}")


def plot_signal_noise(df, category, output_dir):
    """Signal vs noise scatter colored by size group."""
    sub = df[df["category"] == category].dropna(subset=["signal", "noise"])
    sub = sub[(sub["signal"] > 0) & (sub["noise"] > 0)]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for sg in sorted(sub["size_group"].unique()):
        mask = sub["size_group"] == sg
        ax.scatter(sub.loc[mask, "noise"], sub.loc[mask, "signal"],
                   s=10, alpha=0.6, color=SIZE_GROUP_COLORS.get(sg, "gray"),
                   label=sg, edgecolors="none")

    lims = [min(sub["noise"].min(), sub["signal"].min()) * 0.8,
            max(sub["noise"].max(), sub["signal"].max()) * 1.2]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="SNR = 1")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Noise (rel. std across checkpoints)", fontsize=12)
    ax.set_ylabel("Signal (rel. dispersion across models)", fontsize=12)
    ax.set_title(f"Signal vs Noise ({category})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")
    fig.tight_layout()

    fname = f"signal_noise_{category}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_snr_distribution(df, category, output_dir):
    """SNR histogram per size group."""
    sub = df[(df["category"] == category) & np.isfinite(df["snr"]) & (df["snr"] > 0)]
    sizes = sorted(sub["size_group"].unique())
    if not sizes:
        return

    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 4), sharey=True)
    if len(sizes) == 1:
        axes = [axes]

    for ax, sg in zip(axes, sizes):
        vals = sub[sub["size_group"] == sg]["snr"]
        ax.hist(vals, bins=20, color=SIZE_GROUP_COLORS.get(sg, "steelblue"), alpha=0.7, edgecolor="white")
        ax.axvline(vals.median(), color="red", linestyle="--", alpha=0.7, label=f"median={vals.median():.2f}")
        ax.set_xlabel("SNR", fontsize=11)
        ax.set_title(sg, fontsize=12)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count", fontsize=11)
    fig.suptitle(f"SNR Distribution ({category})", fontsize=13)
    fig.tight_layout()

    fname = f"snr_dist_{category}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_benchmark_ranking(df, metric, category, output_dir, top_n=25):
    """Bar chart ranking tasks by a metric, averaging across size groups."""
    sub = df[df["category"] == category].dropna(subset=[metric])
    if sub.empty:
        return

    avg = sub.groupby("task")[metric].mean().nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.28)))
    ax.barh(avg.index, avg.values, color="steelblue", alpha=0.8)
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(f"Top {top_n} Tasks by {metric.upper()} ({category})", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    fname = f"ranking_{metric}_{category}.png"
    fig.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


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

        # Signal vs noise
        plot_signal_noise(df, category, output_dir)

        # SNR distribution
        plot_snr_distribution(df, category, output_dir)

        # Benchmark rankings
        plot_benchmark_ranking(df, "snr", category, output_dir)
        plot_benchmark_ranking(df, "signal", category, output_dir)
        plot_benchmark_ranking(df, "noise", category, output_dir)

        # SNR vs DA for every (small, target) pair
        sizes = sorted(df[df["category"] == category]["size_group"].unique())
        for da_col in da_cols:
            target = da_col.replace("da_", "")
            for small in sizes:
                if small == target:
                    continue
                plot_snr_vs_da(df, small, target, category, output_dir)

    # DA ranking: average DA columns per task
    for category in sorted(df["category"].unique()):
        for da_col in da_cols:
            plot_benchmark_ranking(df, da_col, category, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
