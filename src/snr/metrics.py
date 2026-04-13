"""Core SNR metric computations.

All functions operate on numpy arrays of scores. No I/O or data loading here.

Formulas match the Allen AI signal-and-noise implementation (Heineman et al., 2025)
and the EPFL multilingual preliminary analysis.
"""

import numpy as np
from numpy.typing import NDArray


def relative_dispersion(scores: NDArray[np.float64]) -> float:
    """Relative dispersion: (max - min) / mean.

    Measures how much a benchmark separates models, normalized by overall level.
    Preferred over relative spread per preliminary results (R=0.811 vs 0.791).

    Mathematically equivalent to the Allen AI formula:
        max_{j,k} |m_j - m_k| / mean(m)
    since max pairwise |diff| = max - min for real numbers.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0
    mean = np.mean(scores)
    if mean == 0:
        return 0.0
    return float((np.max(scores) - np.min(scores)) / mean)


def relative_spread(scores: NDArray[np.float64]) -> float:
    """Relative spread: std / mean. Alternative signal metric."""
    scores = np.asarray(scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return 0.0
    mean = np.mean(scores)
    if mean == 0:
        return 0.0
    return float(np.std(scores) / mean)


def signal(scores_per_mix: list[NDArray[np.float64]]) -> float:
    """Signal: relative dispersion of mean scores across data mixtures.

    Matches Allen AI snr_simple.compute_snr_small_scale():
        signal_scores = [mean(mix_i_checkpoints) for each mix]
        signal = (max - min) / mean

    Args:
        scores_per_mix: List of score arrays, one per data mixture.
            Each array may contain scores from multiple seeds/checkpoints.
    """
    mix_means = np.array([float(np.mean(s)) for s in scores_per_mix])
    return relative_dispersion(mix_means)


def checkpoint_noise(
    scores_per_mix: list[NDArray[np.float64]],
) -> float:
    """Checkpoint noise: std / mean over all late-checkpoint scores, pooled.

    Matches Allen AI snr_simple.compute_snr_small_scale():
        noise_scores = scores_arr.flatten()
        noise = std(noise_scores) / mean(noise_scores)

    All checkpoint scores across all mixes are concatenated into one array,
    then noise = std(all) / mean(all).

    This captures BOTH cross-mix variance AND within-mix checkpoint variance,
    which is what makes it directly comparable to signal (also cross-mix).
    """
    all_scores = np.concatenate(scores_per_mix)
    return relative_spread(all_scores)


def benchmark_noise(fold_scores: NDArray[np.float64]) -> float:
    """Benchmark noise: mean of per-model sample std across k-fold splits.

    Matches EPFL multilingual preliminary noise_from_fold_scores():
        per_model_noise = std(fold_scores, ddof=1)  # sample std
        noise = mean(per_model_noise)

    Uses ddof=1 (sample std, Bessel's correction) because k is small (typically 5).

    Args:
        fold_scores: Array of shape (n_models, k_folds) with per-fold scores.
    """
    fold_scores = np.asarray(fold_scores, dtype=float)
    if fold_scores.ndim == 1:
        return float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
    # Per-model noise = sample std across folds, then average across models
    per_model_noise = np.std(fold_scores, axis=1, ddof=1)
    return float(np.mean(per_model_noise))


def snr(signal_value: float, noise_value: float) -> float:
    """Signal-to-Noise Ratio. Higher means more reliable benchmark."""
    if noise_value == 0:
        return float("inf") if signal_value > 0 else 0.0
    return signal_value / noise_value


def decision_accuracy(
    scores_small: NDArray[np.float64],
    scores_large: NDArray[np.float64],
) -> float:
    """Fraction of model pairs where small-scale ranking matches large-scale ranking.

    Uses vectorized pairwise comparison. Models must be aligned (same order).
    Identical to Allen AI decision_acc_fast().

    Args:
        scores_small: 1D array of scores from small proxy models.
        scores_large: 1D array of scores from large target models.

    Returns:
        Proportion of concordant pairs (0.0 to 1.0).
    """
    scores_small = np.asarray(scores_small, dtype=float)
    scores_large = np.asarray(scores_large, dtype=float)
    assert len(scores_small) == len(scores_large), "Score arrays must have same length"
    n = len(scores_small)
    if n < 2:
        return float("nan")

    # Pairwise comparison matrices
    small_gt = scores_small[:, np.newaxis] > scores_small[np.newaxis, :]
    large_gt = scores_large[:, np.newaxis] > scores_large[np.newaxis, :]

    # Upper triangular mask (avoid self-comparison and double-counting)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    agreements = (small_gt == large_gt)[mask]
    return float(np.mean(agreements))


def scaling_law_error(
    predicted: float,
    actual: float,
) -> float:
    """Relative error of scaling law prediction.

    Args:
        predicted: Score predicted by scaling law extrapolation.
        actual: Observed score at target scale.
    """
    if actual == 0:
        return float("inf") if predicted != 0 else 0.0
    return abs(predicted - actual) / abs(actual)


def compute_fold_scores(
    doc_scores: list[float],
    k: int = 5,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Split per-document scores into k folds and compute fold-level accuracy.

    Args:
        doc_scores: List of per-document binary scores (0 or 1 for accuracy tasks).
        k: Number of folds.
        seed: Random seed for fold assignment.

    Returns:
        Array of k fold scores (mean accuracy per fold).
    """
    rng = np.random.RandomState(seed)
    scores = np.asarray(doc_scores, dtype=float)
    indices = rng.permutation(len(scores))
    folds = np.array_split(indices, k)
    return np.array([float(np.mean(scores[fold])) for fold in folds])
