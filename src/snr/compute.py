"""Orchestrate SNR metric computation across all benchmarks.

Computes signal, noise, SNR, and decision accuracy for each benchmark/task,
organized by training stage. Results are returned as a DataFrame and optionally
logged to W&B.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics
from .data import get_primary_metric

# Metrics where lower is better (loss, perplexity, BPB)
LOWER_IS_BETTER_METRICS = {"byte_perplexity", "word_perplexity", "bits_per_byte", "loss"}


def compute_all_metrics(
    df: pd.DataFrame,
    tasks: list[str],
    *,
    small_sizes: list[str] | None = None,
    large_sizes: list[str] | None = None,
    noise_type: str = "benchmark",
    fold_scores: dict[str, dict[str, np.ndarray]] | None = None,
    n_last_checkpoints: int = 5,
    higher_is_better: dict[str, bool] | None = None,
    group_weights: dict[str, dict[str, int]] | None = None,
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
        higher_is_better: Override per-task metric directionality. If None,
            auto-detected from the metric name (e.g., byte_perplexity → lower is better).
        group_weights: Optional weighted subtask aggregation. Dict mapping
            task_group → {subtask: n_samples}. When provided, subtasks are
            aggregated into task_group scores weighted by sample count.

    Returns:
        DataFrame with one row per (task, small_size, large_size) combination.
    """
    results = []
    small_sizes = small_sizes or _infer_sizes(df, role="small")
    large_sizes = large_sizes or _infer_sizes(df, role="large")
    higher_is_better = higher_is_better or {}

    # Optionally aggregate subtasks into weighted group scores
    if group_weights:
        df = _apply_group_weights(df, group_weights)

    for task in tasks:
        task_df = get_primary_metric(df, task)
        if task_df.empty:
            print(f"  Skipping {task}: no data")
            continue

        # Determine metric direction
        metric_name = task_df["metric"].iloc[0] if not task_df.empty else ""
        hib = higher_is_better.get(task, metric_name not in LOWER_IS_BETTER_METRICS)

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
                    da = _compute_decision_accuracy(small_df, large_df, higher_is_better=hib)

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
                        "higher_is_better": hib,
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
    """Compute signal as relative dispersion of mix-level mean scores.

    Matches Allen AI snr_simple.compute_snr_small_scale():
    - Groups by mix, takes last n_last checkpoints per mix
    - Computes mean per mix → relative_dispersion across mix means
    """
    mixes = df["data_mix"].dropna().unique()
    if len(mixes) < 2:
        return 0.0

    scores_per_mix = []
    for mix in mixes:
        mix_df = df[df["data_mix"] == mix].sort_values("checkpoint_index")
        scores = mix_df["score"].dropna().values[-n_last:]
        if len(scores) > 0:
            scores_per_mix.append(scores)

    if len(scores_per_mix) < 2:
        return 0.0

    return metrics.signal(scores_per_mix)


def _compute_checkpoint_noise(df: pd.DataFrame, n_last: int) -> float:
    """Compute checkpoint noise: std/mean over pooled late-checkpoint scores.

    Matches Allen AI snr_simple.compute_snr_small_scale():
        noise_scores = scores_arr.flatten()
        noise = std(noise_scores) / mean(noise_scores)

    All checkpoint scores across all mixes are concatenated, then:
        noise = std(all_flat) / mean(all_flat)

    This pools cross-mix and within-mix variance, which is the correct
    denominator for the SNR ratio (signal also measures cross-mix variance).
    """
    mixes = df["data_mix"].dropna().unique()

    if len(mixes) == 0:
        # No mix info: use all scores directly
        scores = df.sort_values("checkpoint_index")["score"].dropna().values[-n_last:]
        if len(scores) < 2:
            return 0.0
        return metrics.relative_spread(scores)

    # Collect last n_last checkpoints per mix, then pool
    scores_per_mix = []
    for mix in mixes:
        mix_df = df[df["data_mix"] == mix].sort_values("checkpoint_index")
        scores = mix_df["score"].dropna().values[-n_last:]
        if len(scores) > 0:
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
    *,
    higher_is_better: bool = True,
) -> float:
    """Compute decision accuracy between small and large model rankings.

    Matches EPFL multilingual preliminary compute_decision_accuracy_cross_size():
    - Groups by data_mix (base_name in preliminary) to get one score per mix
    - Compares pairwise rankings between small and large scale
    - Supports metric directionality (higher_is_better)

    Falls back to model_id grouping when neither df has mix info.
    """
    # Choose grouping column: prefer data_mix if EITHER df has it,
    # but both must have it for alignment to work
    small_has_mix = small_df["data_mix"].notna().any()
    large_has_mix = large_df["data_mix"].notna().any()

    if small_has_mix and large_has_mix:
        group_col = "data_mix"
    elif not small_has_mix and not large_has_mix:
        group_col = "model_id"
    else:
        # Mismatched: one has mix, other doesn't — can't align
        return float("nan")

    # Get one score per model/mix: mean across checkpoints
    small_scores = small_df.groupby(group_col)["score"].mean().sort_index()
    large_scores = large_df.groupby(group_col)["score"].mean().sort_index()

    # Align on common models/mixes
    common = small_scores.index.intersection(large_scores.index)
    if len(common) < 2:
        return float("nan")

    small_vals = small_scores.loc[common].values
    large_vals = large_scores.loc[common].values

    # Flip for lower-is-better metrics so "better" always means higher
    if not higher_is_better:
        small_vals = -small_vals
        large_vals = -large_vals

    return metrics.decision_accuracy(small_vals, large_vals)


def _apply_group_weights(
    df: pd.DataFrame,
    group_weights: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Aggregate subtasks into weighted group scores.

    For multilingual benchmarks with many subtasks (e.g., XNLI per language),
    this computes weighted means using sample counts.

    Args:
        df: DataFrame with individual subtask rows.
        group_weights: {group_name: {subtask_name: n_samples}}.

    Returns:
        DataFrame with subtask rows replaced by aggregated group rows.
    """
    aggregated_rows = []
    remaining = df.copy()

    for group_name, subtask_weights in group_weights.items():
        subtask_names = list(subtask_weights.keys())
        subtask_df = df[df["task"].isin(subtask_names)]
        remaining = remaining[~remaining["task"].isin(subtask_names)]

        if subtask_df.empty:
            continue

        # Group by model and compute weighted score
        for (model_id, rev, metric), group in subtask_df.groupby(
            ["model_id", "revision", "metric"]
        ):
            total_weight = 0
            weighted_score = 0
            for _, row in group.iterrows():
                w = subtask_weights.get(row["task"], 1)
                weighted_score += row["score"] * w
                total_weight += w
            if total_weight > 0:
                aggregated_rows.append(
                    {
                        **group.iloc[0].to_dict(),
                        "task": group_name,
                        "score": weighted_score / total_weight,
                    }
                )

    if aggregated_rows:
        agg_df = pd.DataFrame(aggregated_rows)
        return pd.concat([remaining, agg_df], ignore_index=True)
    return df


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


# --- Noise caching ---


def save_noise_results(
    fold_scores: dict[str, dict[str, np.ndarray]],
    filepath: str | Path,
) -> None:
    """Cache fold scores to JSON for reuse across runs.

    Args:
        fold_scores: {task: {model_key: ndarray(k,)}} fold scores.
        filepath: Path to save JSON file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for task, models in fold_scores.items():
        serializable[task] = {
            model: scores.tolist() for model, scores in models.items()
        }

    with open(filepath, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Noise results saved to {filepath}")


def load_noise_results(filepath: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load cached fold scores from JSON.

    Args:
        filepath: Path to saved JSON file.

    Returns:
        {task: {model_key: ndarray(k,)}} fold scores.
    """
    filepath = Path(filepath)
    with open(filepath) as f:
        data = json.load(f)

    fold_scores = {}
    for task, models in data.items():
        fold_scores[task] = {
            model: np.array(scores) for model, scores in models.items()
        }
    print(f"Loaded noise results for {len(fold_scores)} tasks from {filepath}")
    return fold_scores
