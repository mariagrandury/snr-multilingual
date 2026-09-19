"""Tests for the early-and-small ranking grid of rq02 (compute_da.py), on a
synthetic ladder: three design variants, two sizes, ten checkpoints.

    python -m unittest discover -s tests -v        # from src/signal-and-noise
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

_SND = Path(__file__).resolve().parents[1]
if str(_SND) not in sys.path:
    sys.path.insert(0, str(_SND))

from analysis.rq02_decision_accuracy import compute_da as C  # noqa: E402


def _ladder() -> pd.DataFrame:
    """Variants a < b < c at the reference. The proxy ranks them c < b < a
    early (reversed) and a < b < c from 60 % on."""
    rows = []
    for size, flip_until in ((C.SMALL_SIZES[0], 5), (C.TARGET_SIZE, 0)):
        for i, fam in enumerate("abc"):
            for step in range(1, 11):
                rank = (2 - i) if step <= flip_until else i
                rows.append({"model": f"{fam}-{size}", "family": fam, "bucket": size, "size": size, "task": "t",
                             "step": step * 100, "primary_score": rank + step / 100, "compute": float(step)})
    return pd.DataFrame(rows)


class EarlySmall(unittest.TestCase):
    def setUp(self):
        self.df = _ladder()
        self.grid = pd.DataFrame(C.compute_early_small_decision_accuracy(self.df)).set_index(["proxy_size", "frac"])

    def test_final_column_is_da_size(self):
        small = C.SMALL_SIZES[0]
        self.assertEqual(self.grid.loc[(small, 1.0), "da"],
                         C.compute_size_decision_accuracy(self.df, "t", small))

    def test_reference_row_is_its_own_da_ckpt(self):
        for f in C.CKPT_DA_EARLY_FRACS:
            self.assertEqual(self.grid.loc[(C.TARGET_SIZE, f), "da"],
                             C.compute_ckpt_decision_accuracy(self.df, "t", C.TARGET_SIZE, f))
        self.assertNotIn((C.TARGET_SIZE, 1.0), self.grid.index)      # the reference against itself

    def test_early_reversal_is_seen(self):
        small = C.SMALL_SIZES[0]
        self.assertEqual(self.grid.loc[(small, 0.2), "da"], 0.0)
        self.assertEqual(self.grid.loc[(small, 0.6), "da"], 1.0)
        self.assertEqual(self.grid.loc[(small, 0.6), "n_pairs"], 3)

    def test_compute_is_the_proxy_checkpoint(self):
        small = C.SMALL_SIZES[0]
        self.assertEqual(self.grid.loc[(small, 0.4), "compute"], 4.0)
        self.assertEqual(self.grid.loc[(small, 0.4), "ref_compute"], 10.0)


class ToleranceAndReference(unittest.TestCase):
    def test_a_family_without_a_checkpoint_near_the_fraction_is_dropped(self):
        df = _ladder()
        small = C.SMALL_SIZES[0]
        # variant "c" of the proxy keeps only its first and last checkpoints: nothing within CKPT_TOL of 60 %
        df = df[~((df["bucket"] == small) & (df["family"] == "c") & df["step"].between(200, 900))]
        grid = pd.DataFrame(C.compute_early_small_decision_accuracy(df)).set_index(["proxy_size", "frac"])
        self.assertEqual(grid.loc[(small, 0.6), "n_pairs"], 1)        # a and b only
        self.assertEqual(grid.loc[(small, 1.0), "n_pairs"], 3)        # the final checkpoint is still there

    def test_the_reference_is_read_at_its_final_checkpoint(self):
        df = _ladder()
        ref = df["bucket"] == C.TARGET_SIZE
        # the reference ranks c < b < a until 90 % and a < b < c only at its last checkpoint
        flip = ref & (df["step"] < 1000)
        df.loc[flip, "primary_score"] = (2 - df.loc[flip, "family"].map({"a": 0, "b": 1, "c": 2})) + df.loc[flip, "step"] / 1e4
        grid = pd.DataFrame(C.compute_early_small_decision_accuracy(df)).set_index(["proxy_size", "frac"])
        self.assertEqual(grid.loc[(C.SMALL_SIZES[0], 1.0), "da"], 1.0)   # vs the final ranking, not an earlier one
        self.assertEqual(grid.loc[(C.TARGET_SIZE, 0.8), "da"], 0.0)


if __name__ == "__main__":
    unittest.main()
