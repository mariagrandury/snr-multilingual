"""SNR metric computation with size-grouped models.

Adapts the signal-and-noise framework (Heineman et al., 2025) for settings
without data mixtures. Different models at the same size play the role of
"data mixtures."

Metrics (all per task):
  - Signal: dispersion of model scores within a size group (rel. dispersion)
  - Noise: checkpoint-to-checkpoint variability within models (rel. std)
  - SNR = signal / noise
  - Decision Accuracy: pairwise ranking agreement between two size groups

Models are split into "base" and "aligned" categories:
  - aligned: name contains sft, instruct, or it (case-insensitive)
  - base: everything else
"""

import re

import numpy as np
import pandas as pd

from src.snr.metrics import decision_accuracy, signal_to_noise_ratio

# ---------------------------------------------------------------------------
# Size extraction and grouping
# ---------------------------------------------------------------------------

DEFAULT_SIZE_BUCKETS: dict[str, tuple[float, float]] = {
    "~0.6B": (0.2, 1.0),
    "~1.5B": (1.0, 2.0),
    "~3B":   (2.0, 5.0),
    "~8B":   (5.0, 15.0),
    "~32B":  (15.0, 50.0),
    "~70B":  (50.0, 100.0),
}

_ALIGNED_PATTERNS = re.compile(
    r"[-_](sft|instruct|aligned|it)(?:[-_]|$)", re.IGNORECASE,
)


def extract_model_size_b(name: str) -> float | None:
    """Extract model size in billions from a model name.

    For names with "fromXB", the *first* size token is the actual model size.
    """
    m = re.search(r"(?:^|-)(\d+\.?\d*)(B|M)(?:-|$)", name)
    if m:
        val = float(m.group(1))
        return val if m.group(2) == "B" else val / 1000
    return None


def classify_model(name: str) -> str:
    """Classify a model as 'base' or 'aligned'."""
    return "aligned" if _ALIGNED_PATTERNS.search(name) else "base"


def assign_size_group(
    name: str,
    buckets: dict[str, tuple[float, float]] | None = None,
) -> str | None:
    """Assign a model name to a size bucket."""
    if buckets is None:
        buckets = DEFAULT_SIZE_BUCKETS
    size = extract_model_size_b(name)
    if size is None:
        return None
    for label, (lo, hi) in buckets.items():
        if lo <= size < hi:
            return label
    return None


def build_size_groups(
    df: pd.DataFrame,
    buckets: dict[str, tuple[float, float]] | None = None,
    min_models: int = 2,
    min_checkpoints: int = 2,
) -> dict[str, list[str]]:
    """Group model names by size bucket.

    Only includes models with >= min_checkpoints and returns groups
    with >= min_models.
    """
    cp_counts = df.groupby("model")["checkpoint"].nunique()
    eligible = set(cp_counts[cp_counts >= min_checkpoints].index)

    groups: dict[str, list[str]] = {}
    for model in eligible:
        label = assign_size_group(model, buckets)
        if label is not None:
            groups.setdefault(label, []).append(model)

    return {k: sorted(v) for k, v in groups.items() if len(v) >= min_models}


# ---------------------------------------------------------------------------
# Add category column
# ---------------------------------------------------------------------------

def add_category_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'category' (base/aligned) and 'size_group' columns."""
    df = df.copy()
    df["category"] = df["model"].apply(classify_model)
    df["size_group"] = df["model"].apply(assign_size_group)
    return df


# ---------------------------------------------------------------------------
# Signal, noise, SNR for one (task, size_group, category)
# ---------------------------------------------------------------------------

def compute_snr_for_task(
    df: pd.DataFrame,
    task: str,
    models: list[str],
    *,
    last_n_checkpoints: int = 5,
) -> dict | None:
    """Compute signal, noise, SNR for one task across a set of models.

    Signal = rel. dispersion of per-model means across models.
    Noise = rel. std across all last-N checkpoint scores (flattened).
    SNR = signal / noise  (via reference signal_to_noise_ratio).

    Returns dict with signal, noise, snr, n_models, n_checkpoints or None.
    """
    task_df = df[(df["task"] == task) & (df["model"].isin(models))]
    if task_df.empty:
        return None

    per_model_tails = []
    for model in models:
        model_df = task_df[task_df["model"] == model].sort_values("checkpoint")
        scores = model_df["score"].values
        if len(scores) == 0:
            continue
        per_model_tails.append(scores[-last_n_checkpoints:])

    if len(per_model_tails) < 2:
        return None

    signal_scores = np.array([np.mean(t) for t in per_model_tails])
    noise_scores = np.concatenate(per_model_tails)

    if np.mean(signal_scores) == 0 or np.mean(noise_scores) == 0:
        return None

    snr_value = signal_to_noise_ratio(signal_scores, noise_scores)
    if not np.isfinite(snr_value):
        return None

    return {
        "signal": (np.max(signal_scores) - np.min(signal_scores)) / np.mean(signal_scores),
        "noise": np.std(noise_scores) / np.mean(noise_scores),
        "snr": snr_value,
        "n_models": len(per_model_tails),
        "n_checkpoints": int(noise_scores.size),
    }


# ---------------------------------------------------------------------------
# Decision Accuracy between two size groups
# ---------------------------------------------------------------------------

def compute_da_for_task(
    df: pd.DataFrame,
    task: str,
    small_models: list[str],
    target_models: list[str],
) -> float | None:
    """Pairwise ranking agreement between two groups of models.

    Takes the latest-checkpoint score for each model. Both lists are used
    as-is (no family matching — all models in each size group participate).
    Truncates to min(len(small), len(target)) by alphabetical sort.

    Returns DA (0-1) or None if < 2 models in either group.
    """
    task_df = df[df["task"] == task]

    def _latest_scores(model_list):
        scores = []
        for model in sorted(model_list):
            model_df = task_df[task_df["model"] == model]
            if model_df.empty:
                return None
            latest = model_df.loc[model_df["checkpoint"].idxmax()]
            scores.append(latest["score"])
        return scores

    small_scores = _latest_scores(small_models)
    target_scores = _latest_scores(target_models)
    if small_scores is None or target_scores is None:
        return None

    n = min(len(small_scores), len(target_scores))
    if n < 2:
        return None

    return float(decision_accuracy(
        np.array(small_scores[:n]),
        np.array(target_scores[:n]),
    ))


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_all_metrics(
    df: pd.DataFrame,
    tasks: list[str] | None = None,
    *,
    last_n_checkpoints: int = 5,
    min_models: int = 2,
    min_checkpoints: int = 2,
) -> pd.DataFrame:
    """Compute signal, noise, SNR, and DA for all (task, size, category) combos.

    DA is computed for every ordered pair of size groups (small → target)
    within the same category.

    Returns DataFrame with columns:
        task, size_group, category, signal, noise, snr, n_models, n_checkpoints,
        da_<target_size> columns for each target size group.
    """
    df = add_category_column(df)

    if tasks is None:
        tasks = sorted(df["task"].unique())

    all_rows = []

    for category in ["base", "aligned"]:
        cat_df = df[df["category"] == category]
        if cat_df.empty:
            continue

        size_groups = build_size_groups(
            cat_df, min_models=min_models, min_checkpoints=min_checkpoints,
        )
        if not size_groups:
            continue

        print(f"\n  [{category}] Size groups:")
        for label, models in sorted(size_groups.items()):
            print(f"    {label}: {models}")

        size_labels = sorted(size_groups.keys())

        for task in tasks:
            for size_label in size_labels:
                models = size_groups[size_label]
                result = compute_snr_for_task(
                    cat_df, task, models, last_n_checkpoints=last_n_checkpoints,
                )
                if result is None:
                    continue

                row = {"task": task, "size_group": size_label, "category": category}
                row.update(result)

                # DA: this size group → every other size group as target
                for target_label in size_labels:
                    if target_label == size_label:
                        continue
                    target_models = size_groups[target_label]
                    da = compute_da_for_task(cat_df, task, models, target_models)
                    row[f"da_{target_label}"] = da

                all_rows.append(row)

    return pd.DataFrame(all_rows)
