"""Orchestrate SNR metric computation from evaluation results.

Wraps the signal-and-noise reference implementation to work with
lm-evaluation-harness result DataFrames. Each "model" in the DataFrame
corresponds to a data mixture (signal dimension), and each "revision"
corresponds to a checkpoint (noise dimension).
"""

import numpy as np
import pandas as pd

from src.snr.metrics import decision_accuracy, signal_to_noise_ratio


def compute_snr_for_task(
    df: pd.DataFrame,
    task: str,
    models: list[str],
    *,
    last_n_checkpoints: int = 5,
) -> dict | None:
    """Compute signal, noise, and SNR for a single task.

    Uses signal_to_noise_ratio() from the reference implementation:
      - signal_scores: mean score per model (data mixture) over last-N checkpoints
      - noise_scores: all individual last-N checkpoint scores (flattened)

    Args:
        df: DataFrame with columns [model, revision, task, score].
        task: Benchmark task name.
        models: Model names to compare (treated as different mixtures).
        last_n_checkpoints: Number of final checkpoints for noise estimation.

    Returns:
        Dict with task, signal, noise, snr, n_models, n_checkpoints,
        or None if insufficient data.
    """
    task_df = df[(df["task"] == task) & (df["model"].isin(models))]
    if task_df.empty:
        return None

    signal_scores = []
    noise_scores = []

    for model in models:
        model_scores = task_df[task_df["model"] == model]["score"].values
        if len(model_scores) == 0:
            continue
        tail = model_scores[-last_n_checkpoints:]
        signal_scores.append(np.mean(tail))
        noise_scores.extend(tail.tolist())

    if len(signal_scores) < 2:
        return None

    snr_value = signal_to_noise_ratio(
        np.array(signal_scores), np.array(noise_scores)
    )

    # Decompose to get individual signal and noise values
    mean = np.mean(signal_scores)
    sig = (np.max(signal_scores) - np.min(signal_scores)) / mean if mean else float("inf")
    noi = np.std(noise_scores) / np.mean(noise_scores) if np.mean(noise_scores) else float("inf")

    return {
        "task": task,
        "signal": sig,
        "noise": noi,
        "snr": snr_value,
        "n_models": len(signal_scores),
        "n_checkpoints": len(noise_scores),
    }


def compute_decision_accuracy_for_task(
    df: pd.DataFrame,
    task: str,
    small_models: list[str],
    target_models: list[str],
) -> float | None:
    """Compute decision accuracy between small and target models.

    Uses decision_acc_fast() from the reference implementation.
    Takes the latest checkpoint score per model.

    Args:
        df: DataFrame with columns [model, revision, task, score].
        task: Benchmark task name.
        small_models: Model names at small scale (one per mix).
        target_models: Model names at target scale (one per mix,
            same order as small_models).

    Returns:
        Decision accuracy (0 to 1), or None if insufficient data.
    """
    task_df = df[df["task"] == task]

    small_scores = []
    for model in small_models:
        model_df = task_df[task_df["model"] == model]
        if model_df.empty:
            return None
        small_scores.append(model_df["score"].values[-1])

    target_scores = []
    for model in target_models:
        model_df = task_df[task_df["model"] == model]
        if model_df.empty:
            return None
        target_scores.append(model_df["score"].values[-1])

    if len(small_scores) < 2:
        return None

    return float(
        decision_accuracy(np.array(small_scores), np.array(target_scores))
    )


def compute_all_metrics(
    df: pd.DataFrame,
    tasks: list[str] | None = None,
    *,
    models: list[str] | None = None,
    target_models: list[str] | None = None,
    last_n_checkpoints: int = 5,
) -> pd.DataFrame:
    """Compute all SNR metrics for a set of tasks.

    Args:
        df: DataFrame with columns [model, revision, task, score].
        tasks: Tasks to compute for (default: all in DataFrame).
        models: Model names for SNR (default: all in DataFrame).
        target_models: Target models for decision accuracy (same order
            as models). If None, decision accuracy is skipped.
        last_n_checkpoints: Final checkpoints for noise estimation.

    Returns:
        DataFrame with columns: task, signal, noise, snr,
        decision_accuracy (optional).
    """
    if tasks is None:
        tasks = sorted(df["task"].unique())
    if models is None:
        models = sorted(df["model"].unique())

    rows = []
    for task in tasks:
        snr_result = compute_snr_for_task(
            df, task, models, last_n_checkpoints=last_n_checkpoints
        )
        if snr_result is None:
            continue
        row = snr_result

        if target_models is not None:
            row["decision_accuracy"] = compute_decision_accuracy_for_task(
                df, task, models, target_models
            )

        rows.append(row)

    return pd.DataFrame(rows)
