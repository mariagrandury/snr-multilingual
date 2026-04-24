"""Core SNR metric primitives.

Formulas match the Allen AI signal-and-noise reference implementation
(Heineman et al., 2025): the same `signal_to_noise_ratio` and the
vectorized `decision_acc_fast` are reproduced here so this repo does
not require the upstream package to be vendored.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def relative_dispersion(scores: ArrayLike) -> float:
    """(max - min) / mean, equivalent to max pairwise |diff| / mean for reals."""
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if mean == 0:
        return 0.0
    return float((np.max(arr) - np.min(arr)) / mean)


def relative_spread(scores: ArrayLike) -> float:
    """std / mean (population std, ddof=0) — matches the reference."""
    arr = np.asarray(scores, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if mean == 0:
        return 0.0
    return float(np.std(arr) / mean)


def signal_to_noise_ratio(
    signal_scores: ArrayLike,
    noise_scores: ArrayLike,
) -> float:
    """SNR = relative_dispersion(signal_scores) / relative_spread(noise_scores).

    Returns +inf when noise is zero and signal is positive, 0.0 otherwise.
    """
    sig = relative_dispersion(signal_scores)
    noi = relative_spread(noise_scores)
    if noi == 0:
        return float("inf") if sig > 0 else 0.0
    return sig / noi


def decision_accuracy(
    scores_small: ArrayLike,
    scores_large: ArrayLike,
) -> float:
    """Vectorized pairwise ranking agreement (Allen AI `decision_acc_fast`).

    Returns the fraction of (i, j) pairs (i < j) where the small-scale and
    large-scale rankings agree.
    """
    s = np.asarray(scores_small, dtype=float)
    l = np.asarray(scores_large, dtype=float)
    if s.shape != l.shape or s.ndim != 1 or s.size < 2:
        return float("nan")
    diff_s = s[:, None] - s[None, :]
    diff_l = l[:, None] - l[None, :]
    agree = np.sign(diff_s) == np.sign(diff_l)
    iu = np.triu_indices(s.size, k=1)
    return float(np.mean(agree[iu]))


__all__ = [
    "relative_dispersion",
    "relative_spread",
    "signal_to_noise_ratio",
    "decision_accuracy",
]
