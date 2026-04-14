"""Load evaluation results from lm-evaluation-harness JSON files or W&B.

Produces a flat DataFrame with columns:
    model, revision, task, metric, score
suitable for SNR computation.
"""

import json
import re
from pathlib import Path

import pandas as pd


# Metric priority: prefer acc_norm, fall back to acc, then exact_match, etc.
METRIC_PRIORITY = [
    "acc_norm", "acc_norm,none",
    "acc", "acc,none",
    "exact_match", "exact_match,none",
]


def _extract_primary_score(task_results: dict) -> tuple[str, float] | None:
    """Pick the best available metric from a task result dict."""
    for metric in METRIC_PRIORITY:
        if metric in task_results:
            return metric.replace(",none", ""), task_results[metric]
    # Fallback: first numeric non-stderr key
    for key, value in task_results.items():
        if isinstance(value, (int, float)) and "stderr" not in key and key != "alias":
            return key.replace(",none", ""), value
    return None


def load_results_dir(results_dir: str | Path) -> pd.DataFrame:
    """Load all lm-evaluation-harness results from a directory tree.

    Expected structure:
        results_dir/{model_name}/{revision}/results_*.json

    Each JSON has:
        {"results": {"task_name": {"acc": 0.5, "acc_norm": 0.6, ...}}}

    Returns:
        DataFrame with columns: model, revision, task, metric, score
    """
    results_dir = Path(results_dir)
    rows = []

    for results_file in sorted(results_dir.rglob("results_*.json")):
        rel = results_file.relative_to(results_dir)
        parts = rel.parts
        if len(parts) < 3:
            continue
        model_name = parts[0]
        revision = parts[1]

        with open(results_file) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        if "results" not in data:
            continue

        for task_name, task_results in data["results"].items():
            result = _extract_primary_score(task_results)
            if result is None:
                continue
            metric_name, score = result
            rows.append({
                "model": model_name,
                "revision": revision,
                "task": task_name,
                "metric": metric_name,
                "score": score,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # If multiple result files exist per model/revision, keep the latest
    df = df.drop_duplicates(subset=["model", "revision", "task"], keep="last")
    return df.reset_index(drop=True)


def load_wandb_results(
    entity: str, project: str, tags: list[str] | None = None
) -> pd.DataFrame:
    """Download evaluation results from W&B.

    Args:
        entity: W&B entity (user or team).
        project: W&B project name.
        tags: Optional tags to filter runs.

    Returns:
        DataFrame with columns: model, revision, task, metric, score
    """
    import wandb

    api = wandb.Api()
    filters = {}
    if tags:
        filters["tags"] = {"$in": tags}

    runs = api.runs(f"{entity}/{project}", filters=filters or None)
    rows = []

    for run in runs:
        config = run.config
        model_name = config.get("model_id", run.name).split("/")[-1]
        revision = config.get("revision", "unknown")

        summary = dict(run.summary)
        for key, value in summary.items():
            if "/" not in key or not isinstance(value, (int, float)):
                continue
            task, metric = key.rsplit("/", 1)
            if "stderr" in metric:
                continue
            rows.append({
                "model": model_name,
                "revision": revision,
                "task": task,
                "metric": metric,
                "score": value,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["model", "revision", "task"], keep="last")
    return df.reset_index(drop=True)


def extract_step(revision: str) -> int | None:
    """Extract step number from a revision/branch name.

    Examples:
        "stage1-step-400000" -> 400000
        "step-1000" -> 1000
        "main" -> None
    """
    match = re.search(r"step[_-]?(\d+)", revision, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None
