"""Signal-to-noise ratio and decision accuracy (Heineman et al., 2025).

This file is upstream allenai/signal-and-noise ``snr/metrics.py`` (subtree
commit f70bfcc) with ONE change: the tie handling of ``decision_acc_fast``,
documented on the function. The upstream kernel is kept verbatim as
``decision_acc_fast_upstream`` so the two can be compared side by side and
the effect measured (``tests/test_metrics.py``). Every other departure from
upstream is listed in CLAUDE.md, "When upstream changes".
"""
import numpy as np


def signal_to_noise_ratio(signal_scores: np.ndarray, noise_scores: np.ndarray) -> float:
    """
    signal = max_{j,k} |m_j - m_k| / m̄
    noise = σ_m / m̄
    snr = signal / noise
    """
    dispersion = np.max([np.abs(mj - mk) for mj in signal_scores for mk in signal_scores])
    signal = dispersion / np.mean(signal_scores)
    noise = np.std(noise_scores) / np.mean(noise_scores)
    snr = signal / noise
    return snr


def decision_acc_fast_upstream(scores_small: np.ndarray, scores_target: np.ndarray) -> float:
    """The upstream kernel, verbatim (allenai/signal-and-noise, f70bfcc).

    Kept only to compare against ``decision_acc_fast`` below; nothing in the
    pipeline calls it. See that function for the tie problem it carries.
    """
    scores_small = np.array(scores_small)
    scores_target = np.array(scores_target)
    small_diffs = scores_small[:, np.newaxis] > scores_small[np.newaxis, :]
    target_diffs = scores_target[:, np.newaxis] > scores_target[np.newaxis, :]
    mask = np.triu(np.ones_like(small_diffs), k=1).astype(bool)
    agreements = (small_diffs == target_diffs)[mask]
    return np.mean(agreements)


def decision_acc_fast(scores_small: np.ndarray, scores_target: np.ndarray) -> float:
    """Decision accuracy: the share of unordered model pairs that the proxy
    (``scores_small``) orders the way the target (``scores_target``) does.

    DEPARTURE FROM UPSTREAM — tie handling (2026-09-16).

    Upstream compares ``a > b`` on both vectors and counts a pair as agreeing
    when the two booleans are equal. That is correct only when neither pair
    is tied. When the proxy ties a pair the target decides, the boolean is
    False on the proxy side whichever model is listed first, while on the
    target side it is True or False depending on the listing order — so the
    same two models count as an agreement in one order and a disagreement in
    the other:

        decision_acc_fast_upstream([1, 1], [1, 2]) == 1.0
        decision_acc_fast_upstream([1, 1], [2, 1]) == 0.0

    The same happens for a pair the target ties and the proxy decides. The
    published number therefore depends on how the models happen to be sorted
    (alphabetically by family name in rq02), which is not a property of the
    benchmark. This version compares the SIGN of the score difference on both
    sides: a pair tied in both vectors agrees, a pair tied in one and decided
    in the other disagrees, and the result does not depend on the listing
    order. Without ties the two kernels are identical.

    Measured effect on the predictivity ladder (pool ``predictivity``, final
    checkpoints, 2026-09-16): exact ties occur in 3,301 of 119,456 family
    pairs, 2.76 % overall — 4.05 % of benchmark pairs (accuracy over a fixed
    item count ties easily) and 0 % of per-language BPB and loss pairs — and
    1,225 of 2,845 (task, size) cells contain at least one. A cell's value
    can move by up to (tied pairs) / (all pairs). The per-task tables
    committed on 2026-09-16 (rq02 ``da_per_task.csv``, rq03
    ``snr_variants_per_task.csv``) were produced with the upstream kernel;
    ``run_all_predictivity.sh`` regenerates them with this one.

    An alternative convention would drop the pairs the target cannot decide
    instead of counting them as misses; rq05 does that for its two-model
    items, where the reference's tie leaves no decision to agree with. The
    kernel keeps the one-number signature upstream callers expect.
    """
    scores_small = np.asarray(scores_small, dtype=float)
    scores_target = np.asarray(scores_target, dtype=float)
    small_sign = np.sign(scores_small[:, np.newaxis] - scores_small[np.newaxis, :])
    target_sign = np.sign(scores_target[:, np.newaxis] - scores_target[np.newaxis, :])
    mask = np.triu(np.ones_like(small_sign, dtype=bool), k=1)
    agreements = (small_sign == target_sign)[mask]
    return np.mean(agreements)
