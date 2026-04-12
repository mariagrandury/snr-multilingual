"""Orchestrate SNR metric computation across all benchmarks.

Computes signal, noise, SNR, and decision accuracy for each benchmark/task,
organized by training stage. Results are returned as a DataFrame and optionally
logged to W&B.
"""

import numpy as np
import pandas as pd

from . import metrics
from .data import get_primary_metric


def compute_all_metrics(
    df: pd.DataFrame,
    tasks: list[str],
    *,
    small_sizes: list[str] | None = None,
    large_sizes: list[str] | None = None,
    noise_type: str = "benchmark",
    fold_scores: dict[str, dict[str, np.ndarray]] | None = None,
    n_last_checkpoints: int = 5,
) -> pd.DataFrame:
    """Compute signal, noise, SNR, and decision accuracy for each task.

    Args:
        df: Evaluation results DataFrame (from src.snr.data loaders).
        tasks: List of task names to analyze.
        small_sizes: Model sizes to use as proxy (e.g., ["175M", "350M"]).
        large_sizes: Model sizes to use as target (e.g., ["1B"]).
        noise_type: "benchmark" (k-fold) or "checkpoint" (multi-checkpoint).
        fold_scores: Pre-computed fold scores dict: {task: {model_key: ndarray(k,)}}.
            Required when noise_type="benchmark".
        n_last_checkpoints: Number of late checkpoints to use for checkpoint noise.

    Returns:
        DataFrame with one row per (task, small_size) combination, columns:
            task, small_size, large_size, signal, noise, snr,
            decision_accuracy, n_models, n_mixes
    """
    results = []
    small_sizes = small_sizes or _infer_sizes(df, role="small")
    large_sizes = large_sizes or _infer_sizes(df, role="large")

    for task in tasks:
        task_df = get_primary_metric(df, task)
        if task_df.empty:
            print(f"  Skipping {task}: no data")
            continue

        for small_size in small_sizes:
            small_df = task_df[task_df["model_size"] == small_size]
            if small_df.empty:
                continue

            # Signal: dispersion across data mixes
            sig = _compute_signal(small_df, n_last_checkpoints)

            # Noise
            if noise_type == "benchmark" and fold_scores and task in fold_scores:
                noi = _compute_benchmark_noise(fold_scores[task])
            else:
                noi = _compute_checkpoint_noise(small_df, n_last_checkpoints)

            # SNR
            snr_val = metrics.snr(sig, noi)

            # Decision accuracy (against each large size)
            for large_size in large_sizes:
                large_df = task_df[task_df["model_size"] == large_size]
                if large_df.empty:
                    da = float("nan")
                else:
                    da = _compute_decision_accuracy(small_df, large_df)

                results.append(
                    {
                        "task": task,
                        "small_size": small_size,
                        "large_size": large_size,
                        "signal": sig,
                        "noise": noi,
                        "noise_type": noise_type,
                        "snr": snr_val,
                        "decision_accuracy": da,
                        "n_models_small": small_df["model_id"].nunique(),
                        "n_models_large": (
                            large_df["model_id"].nunique() if not large_df.empty else 0
                        ),
                        "n_mixes": small_df["data_mix"].nunique(),
                    }
                )

    return pd.DataFrame(results)


def compute_fold_scores_from_samples(
    samples_dir_map: dict[str, dict],
    k: int = 5,
    seed: int = 42,
) -> dict[str, dict[str, np.ndarray]]:
    """Compute k-fold scores from per-sample evaluation data.

    Args:
        samples_dir_map: Dict mapping task names to dicts of
            {model_key: list[float]} per-document scores.
        k: Number of folds.
        seed: Random seed for fold assignment.

    Returns:
        Dict: {task: {model_key: ndarray(k,)}} fold scores.
    """
    result = {}
    for task, model_scores in samples_dir_map.items():
        result[task] = {}
        for model_key, doc_scores in model_scores.items():
            result[task][model_key] = metrics.compute_fold_scores(
                doc_scores, k=k, seed=seed
            )
    return result


def log_results_to_wandb(
    results_df: pd.DataFrame,
    entity: str,
    project: str,
    *,
    stage: str = "pretraining",
    run_name: str | None = None,
) -> None:
    """Log SNR analysis results to W&B as a table and summary metrics."""
    import wandb

    run_name = run_name or f"snr-analysis-{stage}"
    wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        job_type="snr-analysis",
        tags=["snr", stage],
        config={"stage": stage},
    )

    # Log full results table
    table = wandb.Table(dataframe=results_df)
    wandb.log({"snr_results": table})

    # Log per-task summary metrics
    for _, row in results_df.iterrows():
        prefix = f"{row['task']}/{row['small_size']}"
        wandb.summary[f"{prefix}/signal"] = row["signal"]
        wandb.summary[f"{prefix}/noise"] = row["noise"]
        wandb.summary[f"{prefix}/snr"] = row["snr"]
        wandb.summary[f"{prefix}/decision_accuracy"] = row["decision_accuracy"]

    wandb.finish()
    print(f"SNR results logged to {entity}/{project}/{run_name}")


# --- Internal helpers ---


def _compute_signal(df: pd.DataFrame, n_last: int) -> float:
    """Compute signal as relative dispersion of mix-level mean scores."""
    mixes = df["data_mix"].dropna().unique()
    if len(mixes) < 2:
        return 0.0

    scores_per_mix = []
    for mix in mixes:
        mix_df = df[df["data_mix"] == mix].sort_values("checkpoint_index")
        # Use last N checkpoints (or all if fewer)
        scores = mix_df["score"].values[-n_last:]
        if len(scores) > 0:
            scores_per_mix.append(scores)

    if len(scores_per_mix) < 2:
        return 0.0

    return metrics.signal(scores_per_mix)


def _compute_checkpoint_noise(df: pd.DataFrame, n_last: int) -> float:
    """Compute checkpoint noise from score variability across late checkpoints."""
    mixes = df["data_mix"].dropna().unique()
    if len(mixes) == 0:
        # No mix info: use all scores
        scores = df.sort_values("checkpoint_index")["score"].values[-n_last:]
        return metrics.relative_spread(scores)

    scores_per_mix = []
    for mix in mixes:
        mix_df = df[df["data_mix"] == mix].sort_values("checkpoint_index")
        scores = mix_df["score"].values[-n_last:]
        if len(scores) > 1:
            scores_per_mix.append(scores)

    if not scores_per_mix:
        return 0.0

    return metrics.checkpoint_noise(scores_per_mix)


def _compute_benchmark_noise(model_folds: dict[str, np.ndarray]) -> float:
    """Compute benchmark noise from pre-computed fold scores."""
    if not model_folds:
        return 0.0
    fold_matrix = np.array(list(model_folds.values()))
    return metrics.benchmark_noise(fold_matrix)


def _compute_decision_accuracy(
    small_df: pd.DataFrame,
    large_df: pd.DataFrame,
) -> float:
    """Compute decision accuracy between small and large model rankings.

    Aligns models by data_mix (or model_id if no mix info).
    """
    # Get one score per model: mean across checkpoints
    small_scores = (
        small_df.groupby(
            "data_mix" if small_df["data_mix"].notna().any() else "model_id"
        )["score"]
        .mean()
        .sort_index()
    )
    large_scores = (
        large_df.groupby(
            "data_mix" if large_df["data_mix"].notna().any() else "model_id"
        )["score"]
        .mean()
        .sort_index()
    )

    # Align on common models/mixes
    common = small_scores.index.intersection(large_scores.index)
    if len(common) < 2:
        return float("nan")

    return metrics.decision_accuracy(
        small_scores.loc[common].values,
        large_scores.loc[common].values,
    )


def _infer_sizes(df: pd.DataFrame, role: str) -> list[str]:
    """Infer model sizes from available data."""
    sizes = df["model_size"].dropna().unique()
    if len(sizes) == 0:
        return []

    # Parse and sort by numeric value
    def _size_to_num(s: str) -> float:
        s = s.upper()
        if s.endswith("B"):
            return float(s[:-1]) * 1000
        if s.endswith("M"):
            return float(s[:-1])
        return float(s)

    sorted_sizes = sorted(sizes, key=_size_to_num)

    if role == "small":
        # Use smaller sizes as proxies
        return sorted_sizes[: max(1, len(sorted_sizes) - 1)]
    else:
        # Use largest as target
        return sorted_sizes[-1:]
