"""Load evaluation results from lm-evaluation-harness JSON files or W&B.

Produces a flat DataFrame with columns:
    model, checkpoint, task, metric, score
where each row is one (model, checkpoint, task) observation.
The checkpoint column carries the training step (int) so that scores
can be ordered chronologically for noise estimation.
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


def _rows_from_results_dict(
    results: dict, model_name: str, checkpoint: int,
) -> list[dict]:
    """Extract rows from a {"task": {"acc": ...}} results dict."""
    rows = []
    for task_name, task_results in results.items():
        result = _extract_primary_score(task_results)
        if result is None:
            continue
        metric_name, score = result
        rows.append({
            "model": model_name,
            "checkpoint": checkpoint,
            "task": task_name,
            "metric": metric_name,
            "score": score,
        })
    return rows


def load_results_dir(results_dir: str | Path) -> pd.DataFrame:
    """Load all lm-evaluation-harness results from a directory tree.

    Handles two formats produced by import_wandb.py:
      1. Flat: {"results": {...}, "OptStep": N}
      2. Multi-checkpoint: {"checkpoints": [{"results": {...}, "OptStep": N}, ...]}

    Also handles native lm_eval output: {"results": {...}} (checkpoint=0).

    Returns:
        DataFrame with columns: model, checkpoint, task, metric, score
    """
    results_dir = Path(results_dir)
    rows = []

    for results_file in sorted(results_dir.rglob("results_*.json")):
        rel = results_file.relative_to(results_dir)
        parts = rel.parts
        if len(parts) < 3:
            continue
        model_name = parts[0]

        with open(results_file) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        if "checkpoints" in data:
            # Multi-checkpoint format from import_wandb.py
            for cp in data["checkpoints"]:
                if "results" not in cp:
                    continue
                step = int(cp.get("OptStep", 0) or 0)
                rows.extend(_rows_from_results_dict(cp["results"], model_name, step))
        elif "results" in data:
            # Flat format (single checkpoint)
            step = int(data.get("OptStep", 0) or 0)
            rows.extend(_rows_from_results_dict(data["results"], model_name, step))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.drop_duplicates(subset=["model", "checkpoint", "task"], keep="last")
    return df.sort_values(["model", "checkpoint"]).reset_index(drop=True)


def _matches_family(name: str, families: list[str]) -> bool:
    """Check if a model name matches any of the family prefixes (case-insensitive)."""
    name_lower = name.lower()
    return any(f in name_lower for f in families)


def load_wandb_results(
    entity_project: str,
    tags: list[str] | None = None,
    model_families: list[str] | None = None,
) -> pd.DataFrame:
    """Download evaluation results with full checkpoint history from W&B.

    Args:
        entity_project: W&B path as "entity/project".
        tags: Optional tags to filter runs.
        model_families: If provided, only fetch runs whose name contains
            one of these strings (case-insensitive).

    Returns:
        DataFrame with columns: model, checkpoint, task, metric, score
    """
    import wandb

    api = wandb.Api(timeout=60)
    filters = {}
    if tags:
        filters["tags"] = {"$in": tags}

    runs = api.runs(entity_project, filters=filters or None)
    frames = []

    for run in runs:
        model_name = run.name
        if model_families and not _matches_family(model_name, model_families):
            continue
        try:
            history = run.history(samples=10000, pandas=True)
        except Exception as e:
            print(f"    WARNING: failed to fetch history for {model_name}: {e}")
            continue
        if history is None or history.empty:
            continue

        frame = _history_to_long(history, model_name)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["model", "checkpoint", "task"], keep="last")
    return df.sort_values(["model", "checkpoint"]).reset_index(drop=True)


def _history_to_long(history: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Vectorized melt of a wandb run.history frame into long form."""
    if "OptStep" in history.columns:
        steps = history["OptStep"]
    elif "OptimizerStep" in history.columns:
        steps = history["OptimizerStep"]
    else:
        steps = pd.Series(0, index=history.index)
    steps = pd.to_numeric(steps, errors="coerce").fillna(0).astype(int)

    metric_cols = [
        c for c in history.columns
        if "/" in c
        and not c.startswith(("_", "system."))
        and "stderr" not in c.rsplit("/", 1)[-1]
    ]
    if not metric_cols:
        return pd.DataFrame()

    sub = history[metric_cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.assign(checkpoint=steps.values)
    long = sub.melt(id_vars="checkpoint", var_name="task_metric", value_name="score")
    long = long.dropna(subset=["score"])
    long = long[long["score"] >= 0]
    if long.empty:
        return long
    task_metric = long["task_metric"].str.rsplit("/", n=1, expand=True)
    long["task"] = task_metric[0]
    long["metric"] = task_metric[1]
    long["model"] = model_name
    return long[["model", "checkpoint", "task", "metric", "score"]]


def load_wandb_projects(
    projects: list[str],
    model_families: list[str] | None = None,
) -> pd.DataFrame:
    """Load and concatenate results from multiple W&B projects.

    Args:
        projects: List of "entity/project" strings.
        model_families: If provided, only fetch runs whose name contains
            one of these strings (case-insensitive).

    Returns:
        Combined DataFrame with an extra 'source' column.
    """
    dfs = []
    for proj in projects:
        print(f"  Pulling {proj}...")
        df = load_wandb_results(proj, model_families=model_families)
        if not df.empty:
            df["source"] = proj
            dfs.append(df)
            print(f"    {len(df)} rows, {df['model'].nunique()} models, "
                  f"{df['task'].nunique()} tasks, "
                  f"{df.groupby('model')['checkpoint'].nunique().mean():.1f} avg checkpoints/model")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


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
