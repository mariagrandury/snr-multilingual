"""Minimum model size at which a measurement predicts the reference's decision.

At a fixed number of languages the ladder offers one design decision: depth
(deep vs shallow) at L in {1, 2}, data scheme (A vs B) at L in {8, 15, 30}. For
one measurement -- a single language's bits per byte, macro BPB, or a benchmark
task -- a proxy size is predictive when it picks the same winner as the largest
size trained at that L, and every size above it does too.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "signal-and-noise"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from snr.download.ladder import load_predictivity_eval_results  # noqa: E402
import style as S  # noqa: E402

INTERVENTION = {"arch": ("deep", "shallow"), "scheme": ("A", "B")}
HOLD = {"arch": ("scheme", "A"), "scheme": ("arch", "deep")}


def finals(df):
    return df.loc[df.groupby(["model", "task"])["step"].idxmax()]


def min_predictive_size(seed=1904):
    """One row per (L, intervention, task): the smallest predictive size, or NaN."""
    fin = finals(load_predictivity_eval_results())
    fin = fin[fin["seed"] == seed]
    rows = []
    for axis, (a, b) in INTERVENTION.items():
        hold_col, hold_val = HOLD[axis]
        sub = fin[fin[hold_col] == hold_val]
        for L, g in sub.groupby("L"):
            piv = g.pivot_table(index=["task", "size"], columns=axis, values="primary_score")
            if not {a, b} <= set(piv.columns):
                continue
            piv = piv.dropna(subset=[a, b])
            delta = (piv[a] - piv[b]).unstack("size")
            order = [s for s in S.SIZES if s in delta.columns]
            if len(order) < 2:
                continue
            delta = delta[order]
            ref = order[-1]
            for task, r in delta.iterrows():
                if not np.isfinite(r[ref]) or r[ref] == 0:
                    continue
                want = np.sign(r[ref])
                best = np.nan
                # walk down from the reference: keep the run of agreeing sizes
                for s in reversed(order[:-1]):
                    if np.isfinite(r[s]) and np.sign(r[s]) == want:
                        best = s
                    else:
                        break
                rows.append({"L": int(L), "intervention": axis, "task": task,
                             "reference_size": ref, "min_size": best,
                             "n_sizes": int(np.isfinite(r[order[:-1]]).sum())})
    return pd.DataFrame(rows)


def kind(task):
    if task == "bpb_macro":
        return "macro bits per byte"
    if task.startswith("bpb_"):
        return "one language's bits per byte"
    if task == "train_loss":
        return "training loss"
    return "a benchmark task"


if __name__ == "__main__":
    d = min_predictive_size()
    d["kind"] = d["task"].map(kind)
    print(d.groupby(["intervention", "L"]).agg(n=("task", "size"),
                                               ref=("reference_size", "first")).to_string())
    print()
    print(d.groupby(["kind", "L"])["min_size"].apply(
        lambda s: f"{s.notna().sum()}/{len(s)} predictive").unstack().to_string())
    d.to_csv(os.environ.get("OUT_CSV", "/tmp/min_predictive.csv"), index=False)
