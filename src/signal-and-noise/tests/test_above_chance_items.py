"""Tests for the above-chance item selection (`above_chance_items.select_and_score`).

    python -m unittest tests.test_above_chance_items -v        # from src/signal-and-noise
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

from analysis.rq12_above_chance_items.above_chance_items import select_and_score  # noqa: E402


def items(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["model", "step", "doc_id", "acc"])


class SelectAndScore(unittest.TestCase):
    # two reference runs (r1, r2 at their finals) and one proxy (p) at two steps, four items, chance 0.5
    G = items([("r1", 10, 0, 1), ("r2", 10, 0, 1),     # item 0: reference mean 1.0  -> kept
               ("r1", 10, 1, 1), ("r2", 10, 1, 0),     # item 1: 0.5 = chance        -> dropped (strictly above)
               ("r1", 10, 2, 0), ("r2", 10, 2, 0),     # item 2: 0.0                 -> dropped
               ("r1", 10, 3, 1), ("r2", 10, 3, 1),     # item 3: 1.0                 -> kept
               ("r1", 5, 2, 1), ("r2", 5, 2, 1),       # an earlier checkpoint of the reference runs does not select
               ("p", 3, 0, 0), ("p", 3, 1, 1), ("p", 3, 2, 1), ("p", 3, 3, 1),
               ("p", 7, 0, 1), ("p", 7, 1, 0), ("p", 7, 2, 0), ("p", 7, 3, 1)])
    REF = pd.MultiIndex.from_tuples([("r1", 10), ("r2", 10)])

    def test_keeps_items_strictly_above_chance_at_the_reference_finals(self):
        keep, _ = select_and_score(self.G, "acc", self.REF, 0.5)
        self.assertEqual(sorted(keep), [0, 3])

    def test_full_and_sub_scores_per_run(self):
        _, out = select_and_score(self.G, "acc", self.REF, 0.5)
        out = out.set_index(["model", "step"])
        self.assertAlmostEqual(out.loc[("p", 3), "full"], 0.75)
        self.assertAlmostEqual(out.loc[("p", 3), "sub"], 0.5)
        self.assertAlmostEqual(out.loc[("p", 7), "sub"], 1.0)
        self.assertEqual(out.loc[("p", 7), "n_sub"], 2)
        self.assertEqual(out.loc[("r1", 10), "n_full"], 4)

    def test_nothing_above_chance_leaves_an_empty_sub_benchmark(self):
        keep, out = select_and_score(self.G, "acc", self.REF, 1.0)
        self.assertEqual(len(keep), 0)
        self.assertTrue(out["sub"].isna().all())
        self.assertTrue((out["n_sub"] == 0).all())

    def test_continuous_metric(self):
        g = self.G.assign(acc=self.G["acc"] * 0.4 + np.where(self.G["doc_id"] == 0, 0.3, 0.0))
        keep, _ = select_and_score(g, "acc", self.REF, 0.449)        # truthfulqa_mc2's chance level
        self.assertEqual(sorted(keep), [0])


if __name__ == "__main__":
    unittest.main()
