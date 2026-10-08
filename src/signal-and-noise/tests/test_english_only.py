"""Tests for the family groups the English-only check compares: which data
build a family name carries, and which families mirror L1's four at each L.

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

from analysis.rq13_english_only import english_only as E  # noqa: E402


class TestFamilyGroups(unittest.TestCase):
    def test_build_token(self):
        self.assertEqual(E.build_token("lm-L1-deep-seed1904"), "")
        self.assertEqual(E.build_token("lm-L1-dclmP-deep-seed1904"), "dclmP")
        self.assertEqual(E.build_token("lm-L8-schemeB-shallow-seed1904"), "schemeB")
        self.assertEqual(E.build_token("lm-L15-swiglu-seed1904"), "")
        self.assertEqual(E.build_token("lm-L2-ZH-deep-seed64"), "ZH")

    def test_mirror_groups(self):
        fams = ["lm-L1-deep-seed1904", "lm-L1-shallow-seed1904", "lm-L1-dclmP-deep-seed1904", "lm-L1-fweb-deep-seed1904",
                "lm-L8-deep-seed1904", "lm-L8-shallow-seed1904", "lm-L8-schemeB-deep-seed1904",
                "lm-L8-schemeB-shallow-seed1904", "lm-L8-swiglu-seed1904"]
        fin = pd.DataFrame({"family": fams, "L": [1] * 4 + [8] * 5, "seed": 1904,
                            "ladder": [f.rsplit("-", 2)[-2] for f in fams]})
        groups = E.mirror_groups(E.cell_table(fin))
        self.assertEqual(groups[1], sorted(fams[:4]))
        # the shallow cell of another build and the swiglu ladder have no L1 counterpart
        self.assertEqual(groups[8], sorted(["lm-L8-deep-seed1904", "lm-L8-shallow-seed1904", "lm-L8-schemeB-deep-seed1904"]))

    def test_replicates_are_not_families(self):
        fin = pd.DataFrame({"family": ["lm-L1-deep-seed1904", "lm-L1-deep-seed64"], "L": 1, "seed": [1904, 64],
                            "ladder": "deep"})
        self.assertEqual(E.mirror_groups(E.cell_table(fin)), {1: ["lm-L1-deep-seed1904"]})


class TestVerdict(unittest.TestCase):
    """The verdict's reading: L1 against every comparator at every size, with a tie margin."""

    def test_lead_counts_every_cell(self):
        wide = pd.DataFrame({"L1": [0.60, 0.50], "L2": [0.50, 0.505], "L8": [0.70, 0.40]}, index=["90M", "1B"])
        self.assertEqual(E.lead(wide), (2, 1, 1))                    # 90M: ahead of L2, behind L8; 1B: tie, ahead
        self.assertEqual(E.lead(wide, higher=False), (1, 1, 2))      # lower is better flips ahead and behind

    def test_grade(self):
        self.assertEqual(E.grade(3, 1, 0), "yes")
        self.assertEqual(E.grade(3, 0, 1), "partly")
        self.assertEqual(E.grade(0, 2, 0), "tie")
        self.assertEqual(E.grade(0, 0, 2), "no")

    def test_thin(self):
        self.assertEqual(E.thin(pd.Series([28, 6, 31])).tolist(), [False, True, False])


class TestSpikes(unittest.TestCase):
    def test_one_cell_spike(self):
        sizes = ["600M", "1B", "1.7B"]
        rows = []
        for task, l1 in (("bbpb_a", [1.0, 2.0, 1.1]), ("bbpb_b", [1.0, 1.1, 1.0])):     # a spikes at 1B, b does not
            for L, other in ((2, [1.2, 1.3, 1.1]), (8, [1.2, 1.9, 1.1])):           # L8 spikes on both tasks
                rows += [{"task": task, "size": s, "L": L, "arch": "deep", "scoring": "bbpb", "score_l1": a,
                          "score_other": o} for s, a, o in zip(sizes, l1, other)]
        at, tasks, n_other, n_all = E.spikes(pd.DataFrame(rows), sizes)
        self.assertEqual((at, list(tasks), n_other, n_all), ("1B", ["bbpb_a"], 2, 4))


if __name__ == "__main__":
    unittest.main()
