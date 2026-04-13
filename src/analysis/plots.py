"""Visualization functions for SNR analysis results.

All functions take DataFrames and return matplotlib Figure objects.
No I/O (saving/displaying) is done here — callers decide what to do with figures.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def snr_vs_decision_accuracy(
    results_df: pd.DataFrame,
    *,
    title: str = "SNR vs Decision Accuracy",
    annotate: bool = True,
    log_fit: bool = True,
) -> plt.Figure:
    """Scatter plot of SNR vs decision accuracy with optional log-linear fit.

    This is the central plot of the SNR framework: benchmarks with higher SNR
    should have higher decision accuracy.

    Args:
        results_df: DataFrame with columns: task, snr, decision_accuracy.
        title: Plot title.
        annotate: Whether to label each point with the task name.
        log_fit: Whether to overlay a log-linear regression fit with R/R².
    """
    df = results_df.dropna(subset=["snr", "decision_accuracy"])
    df = df[np.isfinite(df["snr"])]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(df["snr"], df["decision_accuracy"], s=60, alpha=0.7, zorder=3)

    if annotate:
        for _, row in df.iterrows():
            ax.annotate(
                _short_task_name(row["task"]),
                (row["snr"], row["decision_accuracy"]),
                fontsize=7,
                alpha=0.8,
                xytext=(5, 5),
                textcoords="offset points",
            )

    if log_fit and len(df) >= 3:
        _add_log_fit(ax, df["snr"].values, df["decision_accuracy"].values)

    ax.set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax.set_ylabel("Decision Accuracy")
    ax.set_title(title)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def signal_noise_scatter(
    results_df: pd.DataFrame,
    *,
    title: str = "Signal vs Noise",
    annotate: bool = True,
) -> plt.Figure:
    """Scatter plot of signal vs noise, colored by decision accuracy.

    Helps identify benchmarks that are high-signal-low-noise (top-left = best).
    """
    df = results_df.dropna(subset=["signal", "noise", "decision_accuracy"])

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        df["noise"],
        df["signal"],
        c=df["decision_accuracy"],
        cmap="RdYlGn",
        s=60,
        alpha=0.8,
        vmin=0.5,
        vmax=1.0,
        zorder=3,
    )
    plt.colorbar(scatter, ax=ax, label="Decision Accuracy")

    if annotate:
        for _, row in df.iterrows():
            ax.annotate(
                _short_task_name(row["task"]),
                (row["noise"], row["signal"]),
                fontsize=7,
                alpha=0.8,
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.set_xlabel("Noise")
    ax.set_ylabel("Signal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def benchmark_ranking(
    results_df: pd.DataFrame,
    *,
    metric: str = "snr",
    title: str | None = None,
    top_n: int = 30,
) -> plt.Figure:
    """Horizontal bar chart ranking benchmarks by a metric.

    Args:
        results_df: DataFrame with columns: task, <metric>.
        metric: Column to rank by.
        title: Plot title (auto-generated if None).
        top_n: Maximum number of benchmarks to show.
    """
    df = results_df.dropna(subset=[metric]).copy()
    df = df[np.isfinite(df[metric])]
    df = df.nlargest(top_n, metric)
    df = df.sort_values(metric)

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.35)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df)))
    ax.barh(
        [_short_task_name(t) for t in df["task"]],
        df[metric],
        color=colors,
    )
    ax.set_xlabel(metric.upper())
    ax.set_title(title or f"Benchmark Ranking by {metric.upper()}")
    fig.tight_layout()
    return fig


def stage_comparison(
    results_by_stage: dict[str, pd.DataFrame],
    *,
    metric: str = "snr",
    top_n: int = 15,
) -> plt.Figure:
    """Compare benchmark rankings across training stages side by side.

    Args:
        results_by_stage: Dict mapping stage name to results DataFrame.
        metric: Column to compare.
        top_n: Number of top benchmarks per stage.
    """
    n_stages = len(results_by_stage)
    fig, axes = plt.subplots(1, n_stages, figsize=(6 * n_stages, 8), sharey=False)
    if n_stages == 1:
        axes = [axes]

    for ax, (stage, df) in zip(axes, results_by_stage.items()):
        df = df.dropna(subset=[metric])
        df = df[np.isfinite(df[metric])]
        df = df.nlargest(top_n, metric).sort_values(metric)

        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(df)))
        ax.barh(
            [_short_task_name(t) for t in df["task"]],
            df[metric],
            color=colors,
        )
        ax.set_xlabel(metric.upper())
        ax.set_title(f"{stage}")

    fig.suptitle(f"Stage Comparison: {metric.upper()}", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def correlation_matrix(
    results_df: pd.DataFrame,
    metrics_cols: list[str] | None = None,
) -> plt.Figure:
    """Heatmap showing correlations between different SNR metrics.

    Args:
        results_df: DataFrame with metric columns.
        metrics_cols: Columns to include (defaults to signal, noise, snr, decision_accuracy).
    """
    cols = metrics_cols or ["signal", "noise", "snr", "decision_accuracy"]
    available = [c for c in cols if c in results_df.columns]
    corr = results_df[available].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Pearson R")

    ax.set_xticks(range(len(available)))
    ax.set_yticks(range(len(available)))
    ax.set_xticklabels(available, rotation=45, ha="right")
    ax.set_yticklabels(available)

    for i in range(len(available)):
        for j in range(len(available)):
            ax.text(
                j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=10
            )

    ax.set_title("Metric Correlations")
    fig.tight_layout()
    return fig


def cross_dataset_scatter(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    *,
    label_a: str = "Dataset A",
    label_b: str = "Dataset B",
    metric: str = "snr",
) -> plt.Figure:
    """Scatter plot comparing a metric across two datasets on common tasks.

    Args:
        results_a, results_b: DataFrames with at least 'task' and <metric> columns.
        label_a, label_b: Axis labels for each dataset.
        metric: Column to compare.
    """
    from scipy import stats as sp_stats

    agg_a = results_a.groupby("task")[metric].mean().reset_index()
    agg_b = results_b.groupby("task")[metric].mean().reset_index()
    merged = agg_a.merge(agg_b, on="task", suffixes=("_a", "_b"))
    merged = merged.dropna()
    merged = merged[np.isfinite(merged[f"{metric}_a"]) & np.isfinite(merged[f"{metric}_b"])]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(merged[f"{metric}_a"], merged[f"{metric}_b"], s=50, alpha=0.7)
    for _, row in merged.iterrows():
        ax.annotate(
            _short_task_name(row["task"]),
            (row[f"{metric}_a"], row[f"{metric}_b"]),
            fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points",
        )

    if len(merged) >= 3:
        r, _ = sp_stats.pearsonr(merged[f"{metric}_a"], merged[f"{metric}_b"])
        ax.set_title(f"{metric.upper()} Correlation: {label_a} vs {label_b} (R={r:.3f})")
    else:
        ax.set_title(f"{metric.upper()}: {label_a} vs {label_b}")

    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3)
    ax.set_xlabel(f"{metric.upper()} ({label_a})")
    ax.set_ylabel(f"{metric.upper()} ({label_b})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def side_by_side_scatter(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    *,
    label_a: str = "Dataset A",
    label_b: str = "Dataset B",
) -> plt.Figure:
    """Side-by-side SNR vs Decision Accuracy scatter for two datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for ax, df, title in [(ax1, results_a, label_a), (ax2, results_b, label_b)]:
        agg = df.groupby("task").agg({"snr": "mean", "decision_accuracy": "mean"}).reset_index()
        valid = agg.dropna(subset=["snr", "decision_accuracy"])
        valid = valid[np.isfinite(valid["snr"])]
        ax.scatter(valid["snr"], valid["decision_accuracy"], s=50, alpha=0.7)
        for _, row in valid.iterrows():
            ax.annotate(
                _short_task_name(row["task"]),
                (row["snr"], row["decision_accuracy"]),
                fontsize=6, alpha=0.7, xytext=(3, 3), textcoords="offset points",
            )
        ax.set_xlabel("SNR")
        ax.set_ylabel("Decision Accuracy")
        ax.set_title(title)
        ax.set_ylim(0.3, 1.05)
        ax.grid(True, alpha=0.3)

    fig.suptitle("SNR vs Decision Accuracy Comparison", fontsize=14)
    fig.tight_layout()
    return fig


def side_by_side_ranking(
    results_a: pd.DataFrame,
    results_b: pd.DataFrame,
    *,
    label_a: str = "Dataset A",
    label_b: str = "Dataset B",
    metric: str = "snr",
    top_n: int = 20,
) -> plt.Figure:
    """Side-by-side bar chart ranking for two datasets."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    for ax, df, title in [(ax1, results_a, label_a), (ax2, results_b, label_b)]:
        agg = df.groupby("task")[metric].mean().reset_index()
        agg = agg.dropna().nlargest(top_n, metric).sort_values(metric)
        colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(agg)))
        ax.barh([_short_task_name(t) for t in agg["task"]], agg[metric], color=colors)
        ax.set_xlabel(metric.upper())
        ax.set_title(f"{title} Top {top_n}")

    fig.suptitle(f"Top Benchmarks by {metric.upper()}", fontsize=14)
    fig.tight_layout()
    return fig


# --- Internal helpers ---


def _add_log_fit(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    """Add log-linear regression fit line with R/R² annotation and 95% CI band."""
    # Filter to positive x values for log
    mask = x > 0
    x_pos, y_pos = x[mask], y[mask]
    if len(x_pos) < 3:
        return

    log_x = np.log(x_pos)
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, y_pos)

    x_fit = np.linspace(x_pos.min(), x_pos.max(), 100)
    log_x_fit = np.log(x_fit)
    y_fit = slope * log_x_fit + intercept

    # 95% confidence interval band
    n = len(x_pos)
    residuals = y_pos - (slope * log_x + intercept)
    se_residuals = np.sqrt(np.sum(residuals**2) / (n - 2))
    log_x_mean = np.mean(log_x)
    ss_x = np.sum((log_x - log_x_mean) ** 2)
    se_fit = se_residuals * np.sqrt(1 / n + (log_x_fit - log_x_mean) ** 2 / ss_x)
    t_crit = stats.t.ppf(0.975, n - 2)
    ci_upper = y_fit + t_crit * se_fit
    ci_lower = y_fit - t_crit * se_fit

    ax.fill_between(x_fit, ci_lower, ci_upper, alpha=0.15, color="red", label="95% CI")
    ax.plot(x_fit, y_fit, "r--", alpha=0.7, linewidth=1.5)
    ax.text(
        0.05,
        0.95,
        f"R = {r_value:.3f}\nR² = {r_value**2:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )


def _short_task_name(task: str) -> str:
    """Shorten task names for plot labels."""
    # Remove common prefixes
    for prefix in ("leaderboard_", "harness_", "lighteval|", "custom|"):
        if task.startswith(prefix):
            task = task[len(prefix) :]
    # Truncate very long names
    if len(task) > 25:
        task = task[:22] + "..."
    return task
