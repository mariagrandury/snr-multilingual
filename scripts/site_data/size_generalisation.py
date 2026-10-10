"""Data of the size-generalisation page's additional views (RQ10, the 3B
paragraph of paper §4.2).

  reference  DA-size of each proxy against the 1.7B and against the 3B final on
             the same decisions (the six families with a 3B final), split into
             benchmark accuracy, the bBPB twins and per-language BPB: the
             matching and comparable pairs of
             reference_consistency_da_size_per_task_both_axes.csv summed over
             the tasks that pass the gate (`gated` is True for a cell at
             chance), as that analysis pools them; plus the 1.7B model itself
             against 3B, summed the same way from above_reference_3B_per_task.csv
             (final checkpoint).
  gate       per benchmark, the tasks above chance at 1.7B and at 3B
             (gate_crossover_by_benchmark.csv).
  seeds      DA-size against each seed of the 1B reference, against 1.7B and
             against 3B at K = 30 (reference_choice_da_size_both_axes.csv).
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "src/signal-and-noise", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from analysis import paths  # noqa: E402

PRED = paths.SIZE_GENERALISATION / "pretraining/predictivity"


def _rows(df: pd.DataFrame) -> list[dict]:
    return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
             if not (isinstance(v, float) and v != v)} for r in df.to_dict("records")]


def _kind(task: pd.Series) -> pd.Series:
    """accuracy / bbpb / bpb; the training loss is left out (one task)."""
    k = task.str.startswith("bbpb_").map({True: "bbpb", False: "accuracy"})
    return k.where(~task.str.startswith("bpb_"), "bpb").where(task != "train_loss", None)


def reference() -> list[dict]:
    rc = pd.read_csv(PRED / "reference_consistency_da_size_per_task_both_axes.csv")
    rc = rc[~rc["gated"] & rc["da_1.7B"].notna()].assign(kind=lambda d: _kind(d["task"])).dropna(subset=["kind"])
    out = []
    for ref in ("1.7B", "3B"):
        g = rc.groupby(["axes", "kind", "size"]).agg(n_tasks=("task", "size"), n_pairs=("n_pairs", "sum"),
                                                      n_matching=(f"n_matching_{ref}", "sum")).reset_index()
        out.append(g.assign(reference=ref))
    top = pd.read_csv(PRED / "above_reference_3B_per_task.csv")
    top = top[(top["size"] == "1.7B") & (top["frac"] == 1.0) & ~top["gated"] & top["da"].notna()]
    top = top.assign(kind=lambda d: _kind(d["task"])).dropna(subset=["kind"])
    out.append(top.groupby(["axes", "kind", "size"]).agg(n_tasks=("task", "size"), n_pairs=("n_pairs", "sum"),
                                                          n_matching=("n_matching", "sum")).reset_index()
               .assign(reference="3B"))
    df = pd.concat(out, ignore_index=True)
    df["n_matching"] = df["n_matching"].astype(int)
    df["n_pairs"] = df["n_pairs"].astype(int)
    return _rows(df.assign(da=df["n_matching"] / df["n_pairs"]))


def build() -> dict:
    return {"size_generalisation_extra": dict(
        reference=reference(),
        gate=_rows(pd.read_csv(PRED / "gate_crossover_by_benchmark.csv")),
        seeds=_rows(pd.read_csv(PRED / "reference_choice_da_size_both_axes.csv")))}
