"""Load evaluation results from W&B or local JSON files into a standardized DataFrame.

The canonical DataFrame has columns:
    model_id, revision, checkpoint_index, task, metric, score, model_size, data_mix, seed
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"


def load_wandb_results(
    entity: str,
    project: str,
    *,
    tags: list[str] | None = None,
    job_type: str | None = None,
) -> pd.DataFrame:
    """Load evaluation results from W&B runs into a standardized DataFrame.

    Args:
        entity: W&B entity (team/user).
        project: W&B project name.
        tags: Optional filter by tags.
        job_type: Optional filter by job type.
    """
    import wandb

    api = wandb.Api()
    filters = {}
    if tags:
        filters["tags"] = {"$in": tags}
    if job_type:
        filters["jobType"] = job_type

    runs = api.runs(f"{entity}/{project}", filters=filters or None)
    print(f"Loading {len(runs)} runs from {entity}/{project}")

    rows = []
    for run in runs:
        config = run.config
        model_id = config.get("model_id", run.group or run.name)
        revision = config.get("revision", "unknown")
        checkpoint_index = config.get("checkpoint_index", 0)

        # Extract model metadata from name/tags
        model_size, data_mix, seed = _parse_model_metadata(model_id, revision, run.tags)

        # Get task metrics from summary (keys like "task_name/metric_name")
        for key, value in run.summary._json_dict.items():
            if "/" not in key or key.startswith("_"):
                continue
            if "stderr" in key:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            task, metric = key.split("/", 1)
            rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "checkpoint_index": checkpoint_index,
                    "task": task,
                    "metric": metric,
                    "score": value,
                    "model_size": model_size,
                    "data_mix": data_mix,
                    "seed": seed,
                    "run_id": run.id,
                    "run_name": run.name,
                }
            )

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} metric rows across {df['task'].nunique()} tasks")
    return df


def load_wandb_history(
    entity: str,
    project: str,
    *,
    tags: list[str] | None = None,
) -> pd.DataFrame:
    """Load full W&B history (multi-step runs) into a standardized DataFrame.

    Use this for runs that log one row per checkpoint (e.g., imported runs).
    """
    import wandb

    api = wandb.Api()
    filters = {"tags": {"$in": tags}} if tags else None
    runs = api.runs(f"{entity}/{project}", filters=filters)
    print(f"Loading history from {len(runs)} runs in {entity}/{project}")

    rows = []
    for run in runs:
        config = run.config
        model_id = config.get("model_name", config.get("model_id", run.name))

        history = run.history(samples=10000, pandas=True)
        for _, row in history.iterrows():
            checkpoint_index = row.get("checkpoint_index", 0)
            model_size, data_mix, seed = _parse_model_metadata(
                model_id, str(checkpoint_index), run.tags
            )

            for key, value in row.items():
                if "/" not in key or key.startswith("_"):
                    continue
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isnan(value):
                    continue

                task, metric = key.split("/", 1)
                rows.append(
                    {
                        "model_id": model_id,
                        "revision": str(checkpoint_index),
                        "checkpoint_index": (
                            int(checkpoint_index)
                            if not np.isnan(checkpoint_index)
                            else 0
                        ),
                        "task": task,
                        "metric": metric,
                        "score": value,
                        "model_size": model_size,
                        "data_mix": data_mix,
                        "seed": seed,
                        "run_id": run.id,
                        "run_name": run.name,
                    }
                )

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} metric rows from history")
    return df


def load_local_results(results_dir: Path | None = None) -> pd.DataFrame:
    """Load evaluation results from local JSON files.

    Scans results/<model>/<revision>/results_*.json files.
    """
    results_dir = results_dir or RESULTS_DIR
    rows = []

    for results_file in sorted(results_dir.rglob("results_*.json")):
        with open(results_file) as f:
            data = json.load(f)

        # Handle both direct results and imported format
        if "checkpoints" in data:
            # Imported multi-checkpoint format
            model_id = data.get("source_run_name", results_file.parent.parent.name)
            for idx, checkpoint in enumerate(data["checkpoints"]):
                _extract_task_rows(
                    rows, checkpoint.get("results", {}), model_id, str(idx), idx
                )
        elif "results" in data:
            # Standard single-checkpoint format
            model_id = data.get("config", {}).get(
                "model", results_file.parent.parent.name
            )
            revision = results_file.parent.name
            checkpoint_index = data.get("config", {}).get("checkpoint_index", 0)
            _extract_task_rows(
                rows, data["results"], model_id, revision, checkpoint_index
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"Loaded {len(df)} metric rows from {results_dir}")
    return df


def load_samples(samples_dir: Path) -> dict[str, list[float]]:
    """Load per-sample scores from JSONL files for noise computation.

    Args:
        samples_dir: Directory containing samples_<task>_*.jsonl files.

    Returns:
        Dict mapping task names to lists of per-document scores.
    """
    task_scores: dict[str, list[float]] = {}
    for jsonl_file in sorted(samples_dir.glob("samples_*.jsonl")):
        # Extract task name: samples_<task>_<timestamp>.jsonl
        parts = jsonl_file.stem.split("_")
        # Rejoin task name parts (task names can contain underscores)
        timestamp = parts[-1]  # Last part is timestamp
        task = "_".join(parts[1:-1])  # Everything between "samples" and timestamp

        scores = []
        with open(jsonl_file) as f:
            for line in f:
                sample = json.loads(line)
                # lm_eval stores the primary metric score in different fields
                score = _extract_sample_score(sample)
                if score is not None:
                    scores.append(score)

        if scores:
            task_scores[task] = scores

    return task_scores


def _extract_sample_score(sample: dict) -> float | None:
    """Extract the primary score from an lm_eval sample dict."""
    # lm_eval uses different fields depending on the task type
    for key in ("acc", "acc_norm", "exact_match", "f1", "byte_perplexity"):
        if key in sample:
            try:
                return float(sample[key])
            except (TypeError, ValueError):
                continue
    # Fallback: check filtered_resps for loglikelihood tasks
    if "filtered_resps" in sample:
        resps = sample["filtered_resps"]
        if resps and isinstance(resps[0], list) and len(resps[0]) >= 1:
            # For multiple-choice: check if the model got it right
            if "target" in sample and "doc" in sample:
                return 1.0 if sample.get("acc") else 0.0
    return None


def _extract_task_rows(
    rows: list[dict],
    results: dict,
    model_id: str,
    revision: str,
    checkpoint_index: int,
) -> None:
    """Extract per-task metric rows from a results dict."""
    model_size, data_mix, seed = _parse_model_metadata(model_id, revision, [])
    for task_name, task_results in results.items():
        for metric_key, value in task_results.items():
            if metric_key == "alias" or "stderr" in metric_key:
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "checkpoint_index": checkpoint_index,
                    "task": task_name,
                    "metric": metric_key,
                    "score": value,
                    "model_size": model_size,
                    "data_mix": data_mix,
                    "seed": seed,
                    "run_id": None,
                    "run_name": None,
                }
            )


def _parse_model_metadata(
    model_id: str, revision: str, tags: list[str]
) -> tuple[str | None, str | None, str | None]:
    """Extract model_size, data_mix, seed from model naming conventions.

    Supports patterns like:
        - "apertus-175M-en30-seed28" (custom models)
        - Tags: ["175M", "en30", "seed28"]
        - HuggingFace repo names with size in the name
    """
    import re

    model_size = None
    data_mix = None
    seed = None

    text = f"{model_id} {revision} {' '.join(tags)}"

    # Size: look for patterns like 175M, 1B, 3B, 7B
    size_match = re.search(r"\b(\d+[BMb])\b", text)
    if size_match:
        model_size = size_match.group(1).upper()

    # Data mix: look for patterns like en30, en60, en90 or mix-A, etc.
    mix_match = re.search(r"\b(?:en|mix[_-]?)(\d+|[A-C])\b", text, re.IGNORECASE)
    if mix_match:
        data_mix = mix_match.group(0).lower()

    # Seed: look for seed patterns
    seed_match = re.search(r"\bseed[_-]?(\d+)\b", text, re.IGNORECASE)
    if seed_match:
        seed = seed_match.group(1)

    return model_size, data_mix, seed


def get_primary_metric(df: pd.DataFrame, task: str) -> pd.DataFrame:
    """Filter DataFrame to the primary metric for a given task.

    Prefers: acc_norm > acc > exact_match > f1 > byte_perplexity.
    """
    task_df = df[df["task"] == task]
    if task_df.empty:
        return task_df

    metric_priority = ["acc_norm", "acc", "exact_match", "f1", "byte_perplexity"]
    available = task_df["metric"].unique()
    for metric in metric_priority:
        if metric in available:
            return task_df[task_df["metric"] == metric]

    # Fall back to first available metric
    return task_df[task_df["metric"] == available[0]]
