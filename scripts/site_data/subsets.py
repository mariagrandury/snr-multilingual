"""Data of the subsets page's additional view (rq08_subset_selection, Case 1):
per (benchmark, size), the languages ranked by their own SNR, the SNR of
every prefix of that ranking and of a random ordering, the full set's SNR and
the random-subset null at the best prefix's size. Rows of per_benchmark.csv
and summary.csv, only split and renamed."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
from analysis import paths  # noqa: E402
from analysis.grids import paper_name  # noqa: E402

PRED = "pretraining/predictivity"


def _floats(s: str) -> list:
    return [round(float(x), 3) for x in str(s).split("|")] if isinstance(s, str) else []


def build() -> dict:
    d = paths.SUBSET_SELECTION / PRED
    pb = pd.read_csv(d / "per_benchmark.csv")
    summ = pd.read_csv(d / "summary.csv")
    summ = summ[summ["case"] == "case1_per_benchmark"].set_index(["task", "size"])
    cells = []
    for r in pb.to_dict("records"):
        task = r["task"]
        subs = str(r["ranked_subtasks"]).split("|")
        # a subtask's label is its name without the benchmark's prefix (arc_easy -> easy)
        labels = [s[len(task) + 1:] if s.startswith(task + "_") else s for s in subs]
        g = summ.loc[(task, r["size"])] if (task, r["size"]) in summ.index else None
        null = r["null_snr_p95"]
        cells.append(dict(
            task=task, size=r["size"], n=int(r["n_subtasks"]), subs=labels,
            own=_floats(r["ranked_subtask_snrs"]), cum=_floats(r["cumulative_snrs"]),
            rand=_floats(r["random_cumulative_snrs"]), full=round(float(r["full_set_snr"]), 3),
            best_n=int(r["best_n"]), best=round(float(r["best_snr"]), 3),
            null_p95=None if pd.isna(null) else round(float(null), 3),
            null_med=None if pd.isna(r["null_snr_median"]) else round(float(r["null_snr_median"]), 3),
            gain_null=None if g is None or pd.isna(g["gain_over_null"]) else round(float(g["gain_over_null"]), 3)))
    names = {t: paper_name(t) for t in sorted(pb["task"].unique())}
    return {"subsets_curves": dict(names=names, cells=cells)}
