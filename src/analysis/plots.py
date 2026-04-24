"""Analysis plots for SNR results.

Reproduces the visual style from Heineman et al. (2025) "Signal and Noise",
adapted for our size-grouped multilingual setting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


SIZE_GROUP_COLORS = {
    "~0.6B": "#1f77b4",
    "~1.5B": "#2ca02c",
    "~3B":   "#ff7f0e",
    "~8B":   "#d62728",
    "~32B":  "#9467bd",
    "~70B":  "#8c564b",
}


def snr_vs_decision_accuracy(
    df: pd.DataFrame,
    target_size_group: str,
    title: str | None = None,
) -> plt.Figure:
    """Scatter of SNR vs decision accuracy, colored by size group.

    Reproduces Figure 2 from the paper: log-scale SNR on x-axis,
    decision accuracy on y-axis, fit line with R² annotation,
    points colored by model size.

    `target_size_group` selects which `da_<size>` column to use
    (i.e. which target group the small-scale models are trying to predict).
    """
    da_col = f"da_{target_size_group}"
    if da_col not in df.columns:
        raise KeyError(f"{da_col!r} not in DataFrame; available: "
                       f"{[c for c in df.columns if c.startswith('da_')]}")
    if title is None:
        title = f"SNR vs DA (target {target_size_group})"

    plot_df = df.dropna(subset=["snr", da_col]).copy()
    plot_df = plot_df[(plot_df["snr"] > 0) & np.isfinite(plot_df["snr"])]
    if plot_df.empty:
        fig, ax = plt.subplots()
        ax.set_title(title + " (no data)")
        return fig

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot per size group
    for sg in sorted(plot_df["size_group"].unique()):
        mask = plot_df["size_group"] == sg
        color = SIZE_GROUP_COLORS.get(sg, "gray")
        ax.scatter(
            plot_df.loc[mask, "snr"], plot_df.loc[mask, da_col],
            s=10, alpha=0.7, color=color, label=sg, edgecolors="none",
        )

    # Fit line across all points
    x = plot_df["snr"].values
    y = plot_df[da_col].values
    _add_fit_line(ax, x, y)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.set_xlabel("SNR = Rel. Dispersion / Rel. Std.", fontsize=12)
    ax.set_ylabel("Decision Accuracy", fontsize=12)
    ax.set_ylim(top=1)
    ax.set_title(title, fontsize=13)
    ax.legend(title="Model Size", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")

    fig.tight_layout()
    return fig


def snr_vs_da_pair(
    df: pd.DataFrame,
    small_size_group: str,
    target_size_group: str,
    title: str | None = None,
) -> plt.Figure | None:
    """Per-pair scatter: SNR (small group) vs DA (small → target).

    Returns None when the input has fewer than 5 valid points.
    """
    da_col = f"da_{target_size_group}"
    if da_col not in df.columns:
        return None

    sub = df[df["size_group"] == small_size_group].dropna(subset=["snr", da_col]).copy()
    sub = sub[(sub["snr"] > 0) & np.isfinite(sub["snr"])]
    if len(sub) < 5:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    color = SIZE_GROUP_COLORS.get(small_size_group, "gray")
    ax.scatter(sub["snr"], sub[da_col], s=12, alpha=0.7, color=color, edgecolors="none")

    _add_fit_line(ax, sub["snr"].values, sub[da_col].values)

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_xlabel("SNR = Rel. Dispersion / Rel. Std.", fontsize=12)
    ax.set_ylabel("Decision Accuracy", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax.set_title(title or f"SNR vs DA ({small_size_group} → {target_size_group})", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def snr_distribution(
    df: pd.DataFrame,
    title: str = "SNR Distribution",
) -> plt.Figure | None:
    """Histogram of SNR per size group (one subplot per group)."""
    sub = df[np.isfinite(df["snr"]) & (df["snr"] > 0)]
    sizes = sorted(sub["size_group"].unique())
    if not sizes:
        return None

    fig, axes = plt.subplots(1, len(sizes), figsize=(4 * len(sizes), 4), sharey=True)
    if len(sizes) == 1:
        axes = [axes]

    for ax, sg in zip(axes, sizes):
        vals = sub[sub["size_group"] == sg]["snr"]
        ax.hist(vals, bins=20, color=SIZE_GROUP_COLORS.get(sg, "steelblue"),
                alpha=0.7, edgecolor="white")
        ax.axvline(vals.median(), color="red", linestyle="--", alpha=0.7,
                   label=f"median={vals.median():.2f}")
        ax.set_xlabel("SNR", fontsize=11)
        ax.set_title(sg, fontsize=12)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count", fontsize=11)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def signal_noise_scatter(
    df: pd.DataFrame,
    title: str = "Signal vs Noise",
) -> plt.Figure:
    """Scatter plot of signal vs noise, colored by size group."""
    plot_df = df.dropna(subset=["signal", "noise"]).copy()

    fig, ax = plt.subplots(figsize=(7, 6))

    for sg in sorted(plot_df["size_group"].unique()):
        mask = plot_df["size_group"] == sg
        color = SIZE_GROUP_COLORS.get(sg, "gray")
        ax.scatter(
            plot_df.loc[mask, "noise"], plot_df.loc[mask, "signal"],
            s=8, alpha=0.7, color=color, label=sg, edgecolors="none",
        )

    lims = [
        min(plot_df["noise"].min(), plot_df["signal"].min()) * 0.8,
        max(plot_df["noise"].max(), plot_df["signal"].max()) * 1.2,
    ]
    ax.plot(lims, lims, "--", color="gray", alpha=0.5, label="SNR = 1")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Noise (checkpoint variability)", fontsize=12)
    ax.set_ylabel("Signal (model dispersion)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")

    fig.tight_layout()
    return fig


def benchmark_ranking(
    df: pd.DataFrame,
    metric: str = "snr",
    title: str | None = None,
    top_n: int = 30,
) -> plt.Figure:
    """Horizontal bar chart ranking benchmarks by a metric."""
    plot_df = df.dropna(subset=[metric]).nlargest(top_n, metric).sort_values(metric)
    if title is None:
        title = f"Top {top_n} Benchmarks by {metric.upper()}"

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    labels = plot_df["task"]
    if "size_group" in plot_df.columns:
        labels = plot_df["task"] + " (" + plot_df["size_group"] + ")"
    colors = [SIZE_GROUP_COLORS.get(sg, "steelblue") for sg in plot_df.get("size_group", [""] * len(plot_df))]
    ax.barh(labels, plot_df[metric], color=colors, alpha=0.8)
    ax.set_xlabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    fig.tight_layout()
    return fig


def correlation_matrix(df: pd.DataFrame) -> plt.Figure:
    """Correlation heatmap of SNR metrics (includes every `da_<size>` column)."""
    da_cols = [c for c in df.columns if c.startswith("da_")]
    metric_cols = [c for c in ["signal", "noise", "snr"] if c in df.columns] + da_cols
    corr = df[metric_cols].corr()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(metric_cols)))
    ax.set_yticks(range(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(metric_cols, fontsize=10)

    for i in range(len(metric_cols)):
        for j in range(len(metric_cols)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)

    ax.set_title("Metric Correlations", fontsize=13)
    fig.tight_layout()
    return fig


def stage_comparison(results: dict[str, pd.DataFrame]) -> plt.Figure:
    """Compare SNR distributions across training stages."""
    stages = list(results.keys())
    fig, axes = plt.subplots(1, len(stages), figsize=(5 * len(stages), 5), sharey=True)
    if len(stages) == 1:
        axes = [axes]

    for ax, stage in zip(axes, stages):
        df = results[stage]
        if "snr" in df.columns:
            valid = df["snr"].dropna()
            ax.hist(valid, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
            ax.axvline(valid.median(), color="red", linestyle="--", alpha=0.7,
                       label=f"median={valid.median():.2f}")
            ax.set_xlabel("SNR", fontsize=12)
            ax.set_title(stage, fontsize=13)
            ax.legend(fontsize=9)

    axes[0].set_ylabel("Count", fontsize=12)
    fig.tight_layout()
    return fig


def _add_fit_line(ax, x, y):
    """Add log-scale line of best fit with R² annotation (paper style)."""
    x_log = np.log10(x)
    valid = np.isfinite(x_log) & np.isfinite(y)
    x_log, y_fit, x_raw = x_log[valid], y[valid], x[valid]

    if len(x_log) < 3:
        return

    z = np.polyfit(x_log, y_fit, 1)
    p = np.poly1d(z)

    x_line = np.logspace(np.log10(x_raw.min()), np.log10(x_raw.max()), 100)
    y_line = p(np.log10(x_line))

    n = len(x_log)
    x_mean = np.mean(x_log)
    s_err = np.sqrt(np.sum((y_fit - p(x_log)) ** 2) / (n - 2))
    conf = (
        stats.t.ppf(0.975, n - 2) * s_err
        * np.sqrt(1 / n + (np.log10(x_line) - x_mean) ** 2 / np.sum((x_log - x_mean) ** 2))
    )

    r = np.corrcoef(x_log, y_fit)[0, 1]
    r2 = r ** 2
    stderr = s_err * np.sqrt((1 - r2) / (n - 2)) if n > 2 else 0

    ax.plot(x_line, y_line, "--", color="black", alpha=0.5)
    ax.fill_between(x_line, y_line - conf, y_line + conf, color="gray", alpha=0.2)
    ax.text(
        0.03, 0.97,
        f"R = {r:.3f} ± {stderr:.3f}\nR² = {r2:.3f}",
        transform=ax.transAxes, fontsize=10, verticalalignment="top",
    )
