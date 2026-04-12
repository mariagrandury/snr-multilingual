"""Generate summary tables and recommendations from SNR results."""

import pandas as pd


def results_table(
    results_df: pd.DataFrame,
    *,
    sort_by: str = "snr",
    ascending: bool = False,
) -> pd.DataFrame:
    """Format results as a readable summary table.

    Returns a DataFrame with columns:
        Task, Signal, Noise, SNR, Decision Accuracy, N Models
    """
    df = results_df.copy()
    df = df.sort_values(sort_by, ascending=ascending)

    summary = pd.DataFrame(
        {
            "Task": df["task"],
            "Proxy Size": df["small_size"],
            "Target Size": df["large_size"],
            "Signal": df["signal"].round(4),
            "Noise": df["noise"].round(4),
            "SNR": df["snr"].round(2),
            "Decision Acc.": (df["decision_accuracy"] * 100).round(1).astype(str) + "%",
            "# Models": df["n_models_small"],
        }
    )
    return summary.reset_index(drop=True)


def recommend_benchmarks(
    results_df: pd.DataFrame,
    *,
    snr_threshold: float = 2.0,
    da_threshold: float = 0.7,
    max_benchmarks: int = 10,
) -> pd.DataFrame:
    """Recommend a minimal benchmark suite based on SNR and decision accuracy.

    Selects benchmarks with SNR >= snr_threshold AND decision accuracy >= da_threshold,
    sorted by SNR descending, up to max_benchmarks.
    """
    df = results_df.copy()
    high_quality = df[
        (df["snr"] >= snr_threshold) & (df["decision_accuracy"] >= da_threshold)
    ]
    recommended = high_quality.nlargest(max_benchmarks, "snr")

    if recommended.empty:
        print(
            f"No benchmarks meet thresholds (SNR >= {snr_threshold}, DA >= {da_threshold})"
        )
        # Fall back to top by SNR alone
        recommended = df.nlargest(max_benchmarks, "snr")

    return results_table(recommended)


def stage_recommendations(
    results_by_stage: dict[str, pd.DataFrame],
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Generate benchmark recommendations for each training stage.

    Args:
        results_by_stage: Dict mapping stage name to results DataFrame.
        **kwargs: Passed to recommend_benchmarks.

    Returns:
        Dict mapping stage name to recommended benchmark table.
    """
    return {
        stage: recommend_benchmarks(df, **kwargs)
        for stage, df in results_by_stage.items()
    }
