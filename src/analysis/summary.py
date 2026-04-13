"""Generate summary tables, recommendations, and metadata from SNR results."""

import json
from pathlib import Path

import numpy as np
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


def dataset_metadata(
    df: pd.DataFrame,
    *,
    model_catalog: dict | None = None,
    label: str = "",
) -> dict:
    """Compute and return a metadata dict describing a dataset.

    Captures the kind of information that is useful in logs and W&B artifacts:
    row counts, task/model/size/mix breakdowns, and per-model checkpoint counts.

    Args:
        df: Evaluation data in canonical format.
        model_catalog: Optional pre-computed model metadata from load_imported_results.
        label: Human label for the dataset (e.g., "AllenAI DataDecide").
    """
    valid_snr = None
    if "snr" in df.columns:
        valid = df.dropna(subset=["snr"])
        valid = valid[np.isfinite(valid["snr"]) & (valid["snr"] > 0)]
        valid_snr = len(valid)

    meta = {
        "label": label,
        "n_rows": len(df),
        "n_tasks": int(df["task"].nunique()) if "task" in df else 0,
        "n_models": int(df["model_id"].nunique()) if "model_id" in df else 0,
        "sizes": sorted(df["model_size"].dropna().unique().tolist()) if "model_size" in df else [],
        "mixes": sorted(df["data_mix"].dropna().unique().tolist()) if "data_mix" in df else [],
    }

    if valid_snr is not None:
        meta["n_valid_snr_results"] = valid_snr

    if model_catalog:
        meta["models"] = model_catalog

    return meta


def save_metadata(meta: dict, filepath: str | Path) -> None:
    """Save metadata dict as pretty-printed JSON."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Metadata saved to {filepath}")
