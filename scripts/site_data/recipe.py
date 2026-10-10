"""Data of the recipe page's additional view (rq11_evaluation_recipe): one row
per (benchmark, language, variant) task with its DA-size against the 1.7B
final at each proxy 90M-1B and its safe rank (0 = reliable from 90M on,
4 = from 1B, 5 = never), multi-axis pairs. Rows of
recipe_da_all_per_task_multi_axes.csv, only filtered."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
from analysis import paths  # noqa: E402
from analysis.grids import paper_name  # noqa: E402

PRED = "pretraining/predictivity"
SIZES = ["90M", "175M", "350M", "600M", "1B"]


def build() -> dict:
    t = pd.read_csv(paths.EVALUATION_RECIPE / PRED / "recipe_da_all_per_task_multi_axes.csv")
    t = t[t["language"].notna() & (t["language"] != "multi")]
    cols = ["benchmark", "language", "format", "scoring", "reading", "safe_rank", "n_items",
            *[f"da_size_{s}" for s in SIZES]]
    t = t[cols].rename(columns={f"da_size_{s}": s for s in SIZES})
    # a task with no DA-size at any proxy (gated everywhere) tells the reader only that it is at chance
    t["at_chance"] = t[SIZES].isna().all(axis=1)
    t = t.round(3).astype(object).where(t.round(3).notna(), None)
    return {"recipe_tasks": dict(sizes=SIZES, names={b: paper_name(b) for b in sorted(t["benchmark"].unique())},
                                 columns=list(t.columns), data=t.values.tolist())}
