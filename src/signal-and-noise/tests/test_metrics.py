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


class TestPairAgreement(unittest.TestCase):
    """`utils.pair_agreement` states the kernel's tie rule a second time, over
    an explicit pair list (rule 15's mono-axis reading). `decision_acc_fast`
    cannot take one, so nothing but this test keeps the two definitions equal:
    change one without the other and the mono-axis numbers drift from the
    multi-axis ones with no error anywhere.
    """

    def _pair_agreement(self):
        sys.path.insert(0, str(_SND.parent))          # analysis imports `pretrain`
        from analysis.utils import pair_agreement
        return pair_agreement

    def test_matches_the_kernel_over_every_pair(self):
        pair_agreement = self._pair_agreement()
        rng = np.random.default_rng(1904)
        fams = [f"f{i}" for i in range(6)]
        for trial in range(200):
            # a third of the draws are integers, so ties are common — the whole
            # point of the departure the kernel documents
            s, t = (rng.integers(0, 3, 6).astype(float) if trial % 3 else rng.normal(size=6)
                    for _ in range(2))
            self.assertAlmostEqual(pair_agreement(dict(zip(fams, s)), dict(zip(fams, t)))[0],
                                   decision_acc_fast(s, t),
                                   msg=f"pair_agreement != decision_acc_fast on {s} vs {t}")

    def test_min_pairs_and_restriction(self):
        pair_agreement = self._pair_agreement()
        fams = [f"f{i}" for i in range(4)]
        s = dict(zip(fams, [1.0, 2.0, 3.0, 4.0]))
        t = dict(zip(fams, [4.0, 3.0, 2.0, 1.0]))
        self.assertEqual(pair_agreement(s, t), (0.0, 6))          # every pair reversed
        # a pair list is honoured, and rule 5 NaNs a cell below MIN_PAIRS
        da, n = pair_agreement(s, t, [("f0", "f1"), ("f0", "f2"), ("f1", "f2")])
        self.assertEqual((da, n), (0.0, 3))
        da, n = pair_agreement(s, t, [("f0", "f1"), ("f0", "f2")])
        self.assertTrue(np.isnan(da))
        self.assertEqual(n, 2)
        # pairs naming a family the proxy does not have are dropped, not counted
        self.assertEqual(pair_agreement(s, t, [("f0", "f1"), ("f0", "gone")])[1], 1)


class TestAgreementMeasures(unittest.TestCase):
    """`utils.agreement_measures`: the kernel's DA and every rank statistic it
    is a relative of, with the closed-form relation between them."""

    def _fn(self):
        sys.path.insert(0, str(_SND.parent))
        from analysis.utils import agreement_measures, jackknife_ratio
        return agreement_measures, jackknife_ratio

    def test_da_is_the_kernel_and_tau_is_its_closed_form(self):
        agreement_measures, _ = self._fn()
        rng = np.random.default_rng(7)
        for trial in range(300):
            n = int(rng.integers(4, 12))
            s = rng.integers(0, 3, n).astype(float) if trial % 2 else rng.normal(size=n)
            t = rng.integers(0, 3, n).astype(float) if trial % 3 else rng.normal(size=n)
            m = agreement_measures(s, t)
            if np.isnan(m["da"]):
                self.assertTrue(np.std(s) == 0 or np.std(t) == 0)
                continue
            self.assertAlmostEqual(m["da"], decision_acc_fast(s, t))
            # 2·DA − 1 = τ_a + (T_both − T_one) / n_pairs, ties included
            self.assertAlmostEqual(2 * m["da"] - 1, m["tau_a"] + (m["tied_both"] - m["tied_one"]) / m["n_pairs"], places=12)
            self.assertEqual(m["concordant"] + m["discordant"] + m["tied_both"] + m["tied_one"], m["n_pairs"])
        # without ties every convention is the same number
        m = agreement_measures([1, 2, 3, 4, 5], [1, 3, 2, 5, 4])
        self.assertAlmostEqual(m["tau_a"], m["tau_b"]); self.assertAlmostEqual(m["tau_a"], m["gamma"])
        self.assertAlmostEqual(m["da"], m["da_drop_ref_ties"]); self.assertAlmostEqual(2 * m["da"] - 1, m["tau_a"])

    def test_jackknife_is_zero_width_when_every_family_agrees(self):
        import pandas as pd
        _, jackknife_ratio = self._fn()
        fams = list("abcde")
        pairs = [(x, y) for i, x in enumerate(fams) for y in fams[i + 1:]]
        d = pd.DataFrame({"g": "x", "family_a": [p[0] for p in pairs], "family_b": [p[1] for p in pairs], "match": 1})
        out = jackknife_ratio(d, ["g"]).iloc[0]
        self.assertEqual((out["reliability"], out["se"], out["n_families"]), (1.0, 0.0, 5))
        # one bad family: leaving it out moves the ratio, so the band opens
        d.loc[d["family_a"] == "a", "match"] = 0
        out = jackknife_ratio(d, ["g"]).iloc[0]
        self.assertGreater(out["se"], 0); self.assertLess(out["lo"], out["reliability"])
        # below MIN_PAIRS + 1 families: no band, the count still reported
        out = jackknife_ratio(d[d["family_a"].isin(list("ab")) & d["family_b"].isin(list("bc"))], ["g"]).iloc[0]
        self.assertTrue(np.isnan(out["se"])); self.assertEqual(out["n_families"], 3)


if __name__ == "__main__":
    unittest.main()
