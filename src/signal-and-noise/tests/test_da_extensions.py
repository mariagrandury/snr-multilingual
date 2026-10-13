"""Tests for the shared pieces of the decision-accuracy extensions: the
fixed-task population (rule 13), the cube kernel `compute_da.pair_agree` and
the permutation null built on it.

    python -m unittest discover -s tests -v        # from src/signal-and-noise
"""

from __future__ import annotations

import sys
import unittest
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

_SND = Path(__file__).resolve().parents[1]
if str(_SND) not in sys.path:
    sys.path.insert(0, str(_SND))
if str(_SND.parent) not in sys.path:
    sys.path.insert(0, str(_SND.parent))

from analysis.rq02_decision_accuracy.compute_da import pair_agree  # noqa: E402
from analysis.rq02_permutation_null.permutation_null import PERMUTATIONS, cell_nulls  # noqa: E402
from analysis.utils import fixed_population  # noqa: E402
from snr.metrics import decision_acc_fast  # noqa: E402


def cube(proxy: np.ndarray, ref: np.ndarray):
    """One task, families on axis 1, columns (proxy, reference); every pair of families."""
    S = np.stack([proxy, ref], axis=-1)[None]
    I, J = map(np.array, zip(*combinations(range(len(proxy)), 2)))
    return S, np.isfinite(S), I, J


class FixedPopulation(unittest.TestCase):
    def test_keeps_the_tasks_with_every_point_of_their_line(self):
        cells = pd.DataFrame({"line": ["a"] * 5 + ["b"] * 2, "task": ["t1", "t1", "t2", "t3", "t3", "t2", "t2"],
                              "x": [1, 2, 1, 1, 2, 1, 2], "v": [.5, .6, .7, .4, np.nan, .5, .5]})
        kept = fixed_population(cells, ["line"], "x", "v")
        self.assertEqual(sorted(map(tuple, kept[["line", "task"]].drop_duplicates().to_numpy())),
                         [("a", "t1"), ("b", "t2")])      # t2 misses x=2 on a, t3's x=2 is NaN

    def test_without_line_keys(self):
        cells = pd.DataFrame({"task": ["t1", "t1", "t2"], "x": [1, 2, 1], "v": [1., 1., 1.]})
        self.assertEqual(set(fixed_population(cells, [], "x", "v")["task"]), {"t1"})


class PairAgree(unittest.TestCase):
    def test_matches_the_sign_rule_kernel_with_ties(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            proxy, ref = rng.integers(0, 4, 7).astype(float), rng.integers(0, 4, 7).astype(float)
            A, V = pair_agree(*cube(proxy, ref), 0, 1)
            self.assertAlmostEqual(A.sum() / V.sum(), decision_acc_fast(proxy, ref))

    def test_a_missing_score_drops_its_pairs(self):
        A, V = pair_agree(*cube(np.array([1., np.nan, 3.]), np.array([1., 2., 3.])), 0, 1)
        self.assertEqual(V.sum(), 1)


class PermutationNull(unittest.TestCase):
    def test_null_is_chance_and_the_reference_is_kept(self):
        ref = np.arange(12, dtype=float)
        S, P, I, J = cube(ref.copy(), ref)
        m, n, null = cell_nulls(S, P, I, J, 0, 1, np.random.default_rng(1904))
        self.assertEqual(m[0], n[0])                             # the proxy is the reference
        self.assertAlmostEqual(null[0].mean() / n[0], 0.5, delta=0.02)
        p = (1 + (null[0] >= m[0]).sum()) / (PERMUTATIONS + 1)
        self.assertLess(p, 0.01)

    def test_proxy_ties_pull_the_null_below_half(self):
        S, P, I, J = cube(np.repeat([0., 1.], 6), np.arange(12, dtype=float))
        _, n, null = cell_nulls(S, P, I, J, 0, 1, np.random.default_rng(1904))
        self.assertLess(null[0].mean() / n[0], 0.4)              # tied proxy pairs are misses


if __name__ == "__main__":
    unittest.main()
