"""Tests for the proxy-only item selection (`rq14_proxy_item_selection/proxy_item_selection.py`).

    python -m unittest tests.test_proxy_item_selection -v        # from src/signal-and-noise
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_SND = Path(__file__).resolve().parents[1]
for _p in (_SND, _SND.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analysis.rq14_proxy_item_selection.proxy_item_selection import (  # noqa: E402
    SELECT_SIZES, discrimination, heldout_rows, select_on, top_items)
from analysis.utils import TARGET_SIZE  # noqa: E402

FAMS = [f"f{i}" for i in range(8)]
SIZES = [*SELECT_SIZES, TARGET_SIZE]
N_INFO, N_NOISE = 20, 20


def synthetic(seed: int):
    """8 families x (600M, 1B, 1.7B), 40 items: the first 20 are solved when the
    family's latent quality (plus a size bonus and a little noise) clears the
    item's difficulty, the last 20 are coin flips. Returns the items x runs
    matrix, the runs' family and size, and the truth (family -> latent quality,
    which the 1.7B full task orders the same way)."""
    rng = np.random.default_rng(seed)
    z = dict(zip(FAMS, rng.permutation(np.linspace(-1, 1, len(FAMS)))))
    cols = pd.DataFrame([(f, s) for s in SIZES for f in FAMS], columns=["family", "size"])
    bonus = cols["size"].map(dict(zip(SIZES, (0.0, 0.3, 0.6)))).to_numpy()
    q = cols["family"].map(z).to_numpy() + bonus
    diff = rng.uniform(-1, 1.5, N_INFO)[:, None]
    info = (q[None, :] + rng.normal(0, 0.25, (N_INFO, len(cols))) > diff).astype(float)
    noise = rng.integers(0, 2, (N_NOISE, len(cols))).astype(float)
    return np.vstack([info, noise]), cols, z


class Discrimination(unittest.TestCase):
    def test_informative_items_rank_first(self):
        for seed in range(5):
            A, cols, _ = synthetic(seed)
            disc = select_on(A, cols, set(FAMS))
            kept, _ = top_items(disc, 0.25)
            self.assertGreaterEqual(np.isin(kept, np.arange(N_INFO)).mean(), 0.9)
            self.assertGreater(np.nanmean(disc[:N_INFO]), np.nanmean(disc[N_INFO:]) + 0.3)

    def test_constant_items_are_nan_and_never_kept(self):
        A, cols, _ = synthetic(1)
        A[3], A[25] = 1.0, 0.0
        disc = select_on(A, cols, set(FAMS))
        self.assertTrue(np.isnan(disc[[3, 25]]).all())
        kept, cand = top_items(disc, 0.75)
        self.assertNotIn(3, cand); self.assertNotIn(25, kept)
        self.assertEqual(len(kept), int(np.ceil(0.75 * len(cand))))

    def test_centred_within_size(self):
        # an item solved at 1B and failed at 600M by every family separates sizes, not designs
        size = np.array(["600M"] * 4 + ["1B"] * 4)
        A = np.vstack([np.r_[np.zeros(4), np.ones(4)], [0, 1, 0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 0, 0, 1, 1]])
        d = discrimination(A, size)
        self.assertTrue(np.isnan(d[0]))
        self.assertTrue(np.isfinite(d[1:]).all())


class NoReferenceInTheSelection(unittest.TestCase):
    def test_reference_and_heldout_runs_do_not_move_the_selection(self):
        A, cols, _ = synthetic(2)
        sel = set(FAMS[::2])
        before = select_on(A, cols, sel)
        B = A.copy()
        out = (cols["size"] == TARGET_SIZE).to_numpy() | ~cols["family"].isin(sel).to_numpy()
        B[:, out] = np.random.default_rng(9).integers(0, 2, (len(B), out.sum()))
        np.testing.assert_array_equal(before, select_on(B, cols, sel))

    def test_perturbing_the_reference_items_leaves_the_proxy_reading(self):
        A, cols, z = synthetic(3)
        halves = (set(FAMS[::2]), set(FAMS[1::2]))
        groups = {"multi-axis": [(a, b) for i, a in enumerate(FAMS) for b in FAMS[i + 1:]]}
        B = A.copy()
        ref = (cols["size"] == TARGET_SIZE).to_numpy()
        B[:, ref] = 1 - B[:, ref]
        keys = ["axes", "q", "size", "n_kept", "da_kept", "da_full"]
        a = pd.DataFrame(heldout_rows(A, cols, z, groups, halves, np.array([], int), np.random.default_rng(0)))[keys]
        b = pd.DataFrame(heldout_rows(B, cols, z, groups, halves, np.array([], int), np.random.default_rng(0)))[keys]
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(set(a["size"]) <= set(SELECT_SIZES))       # DA is read at the proxies, never at the reference


class HeldOut(unittest.TestCase):
    def test_kept_items_beat_random_in_expectation(self):
        groups = {"multi-axis": [(a, b) for i, a in enumerate(FAMS) for b in FAMS[i + 1:]]}
        rows = []
        for seed in range(10):
            A, cols, z = synthetic(seed)
            halves = (set(FAMS[::2]), set(FAMS[1::2]))
            rows += heldout_rows(A, cols, z, groups, halves, np.arange(N_INFO), np.random.default_rng(seed))
        t = pd.DataFrame(rows)
        self.assertTrue(((t["n_pairs"] == 12) & t["da_kept"].notna()).all())     # 6 held-out pairs per half
        m = t.groupby("q")[["da_kept", "da_random", "da_full"]].mean()
        self.assertTrue((m["da_kept"] >= m["da_random"]).all(), m)
        self.assertGreater(m.loc[0.25, "da_kept"], m.loc[0.25, "da_random"])
        self.assertTrue(t["da_reference_selected"].notna().all())

    def test_below_min_pairs_is_nan_with_its_count(self):
        A, cols, z = synthetic(4)
        halves = ({"f0", "f1", "f2", "f3", "f4", "f5"}, {"f6", "f7"})       # one pair on the second half
        groups = {"multi-axis": [("f6", "f7")]}
        t = pd.DataFrame(heldout_rows(A, cols, z, groups, halves, np.array([], int), np.random.default_rng(0)))
        self.assertTrue(t["da_kept"].isna().all())
        self.assertTrue((t["n_pairs"] == 1).all())


if __name__ == "__main__":
    unittest.main()
