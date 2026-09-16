"""Tests for snr.metrics: the SNR definition and the decision-accuracy
kernel, including the one departure from upstream (tie handling).

    python -m unittest discover -s tests -v        # from src/signal-and-noise

The last test reads the ladder report when one is on disk and checks
data-independent invariants only; nothing here pins a number that new cells
would move.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

_SND = Path(__file__).resolve().parents[1]
if str(_SND) not in sys.path:
    sys.path.insert(0, str(_SND))

from snr.metrics import (  # noqa: E402
    decision_acc_fast, decision_acc_fast_upstream, signal_to_noise_ratio)


class DecisionAccuracy(unittest.TestCase):
    def test_perfect_and_reversed(self):
        self.assertEqual(decision_acc_fast([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertEqual(decision_acc_fast([4, 3, 2, 1], [10, 20, 30, 40]), 0.0)

    def test_lower_is_better_needs_no_flip(self):
        self.assertEqual(decision_acc_fast([1.2, 1.1, 1.3], [0.9, 0.8, 1.0]), 1.0)

    def test_identical_to_upstream_without_ties(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            s, t = rng.normal(size=7), rng.normal(size=7)
            self.assertAlmostEqual(decision_acc_fast(s, t), decision_acc_fast_upstream(s, t))

    def test_upstream_tie_handling_depends_on_listing_order(self):
        # the bug: the same two models, listed the other way round
        self.assertEqual(decision_acc_fast_upstream([1, 1], [1, 2]), 1.0)
        self.assertEqual(decision_acc_fast_upstream([1, 1], [2, 1]), 0.0)
        self.assertEqual(decision_acc_fast_upstream([1, 2], [1, 1]), 1.0)
        self.assertEqual(decision_acc_fast_upstream([2, 1], [1, 1]), 0.0)

    def test_fixed_tie_handling_is_order_invariant(self):
        # proxy cannot decide a pair the target decides: a miss, either order
        self.assertEqual(decision_acc_fast([1, 1], [1, 2]), 0.0)
        self.assertEqual(decision_acc_fast([1, 1], [2, 1]), 0.0)
        # target cannot decide a pair the proxy decides: a miss, either order
        self.assertEqual(decision_acc_fast([1, 2], [1, 1]), 0.0)
        self.assertEqual(decision_acc_fast([2, 1], [1, 1]), 0.0)
        # tied in both: agreement
        self.assertEqual(decision_acc_fast([1, 1], [2, 2]), 1.0)
        # any permutation of the models gives the same value
        rng = np.random.default_rng(1)
        s, t = np.array([1, 1, 2, 3, 3, 4.]), np.array([2, 1, 1, 3, 4, 4.])
        vals = set()
        for _ in range(20):
            p = rng.permutation(len(s))
            vals.add(round(decision_acc_fast(s[p], t[p]), 12))
        self.assertEqual(len(vals), 1)


class SNR(unittest.TestCase):
    def test_definition(self):
        snr = signal_to_noise_ratio([1.0, 3.0], [1.0, 3.0])
        self.assertAlmostEqual(snr, (2 / 2) / (1 / 2))


class Ladder(unittest.TestCase):
    """Invariants on the real report, when one is on disk."""

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from snr.download.ladder import ladder_dir, load_predictivity_eval_results
            cls.df = load_predictivity_eval_results(ladder_dir())
        except Exception:                          # no local copy, no network
            raise unittest.SkipTest("no ladder report available")

    def _finals(self, size):
        f = self.df[(self.df["size"] == size) & (self.df["seed"] == 1904)
                    & (self.df["arch"] == "deep") & (self.df["scheme"] == "A")]
        f = f.loc[f.groupby(["family", "task"])["step"].idxmax()]
        return f.pivot(index="family", columns="task", values="primary_score")

    def test_self_da_is_one_and_kernels_agree_without_ties(self):
        small, ref = self._finals("600M"), self._finals("1B")
        fams = small.index.intersection(ref.index)
        self.assertGreaterEqual(len(fams), 3, "fewer than 3 families at both sizes")
        checked = differ = 0
        for task in small.columns.intersection(ref.columns):
            s, t = small.loc[fams, task], ref.loc[fams, task]
            if s.isna().any() or t.isna().any():
                continue
            self.assertEqual(decision_acc_fast(t, t), 1.0)
            fixed, up = decision_acc_fast(s, t), decision_acc_fast_upstream(s, t)
            if s.nunique() == len(s) and t.nunique() == len(t):
                self.assertAlmostEqual(fixed, up)
            elif fixed != up:
                differ += 1
            checked += 1
        self.assertGreater(checked, 0)
        print(f"\n  ladder 600M->1B: {checked} tasks checked, {differ} differ between kernels because of ties")


if __name__ == "__main__":
    unittest.main()
