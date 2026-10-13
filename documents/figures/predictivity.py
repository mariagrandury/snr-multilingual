"""Minimum model size at which a measurement predicts the reference's decision.

At a fixed number of languages the ladder offers one design decision: depth
(deep vs shallow) at every L, data scheme (A vs B) at L in {8, 15, 30}. For
one measurement -- a single language's bits per byte, macro BPB, or a benchmark
task -- a proxy size is predictive when it picks the same winner as the largest
size trained at that L, and every size above it does too.

The scores come from `analysis.utils.build_snr_pool("predictivity")`, so the
analysis rules apply as in every rq table: sizes 90M to the 1.7B reference
(rule 10), per-language parents only (rule 6), trained languages only (rule 2),
with the bBPB twins the pool adds. A benchmark with a chance level also passes
the above-random gate (rule 1, rq00's mask): it is dropped when it is at chance
at the reference, and a proxy size where it is at chance cannot agree. One
difference stays: each (decision, L) is read on ONE pair of runs, so rule 5's
three-pair minimum is not applied (rq05's intervention tables pool the pairs).
Imported by fig_predictivity, fig_benchmark_predictivity, fig_rq6_sketch,
fig_rqs (b3) and facts.py.
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "signal-and-noise"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from functools import lru_cache  # noqa: E402
from analysis.utils import build_snr_pool, passes_gate  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
import style as S  # noqa: E402


@lru_cache(maxsize=1)
def _pool():
    """The rule-filtered ladder scores (see the module docstring)."""
    return build_snr_pool("predictivity")


INTERVENTION = {"arch": ("deep", "shallow"), "scheme": ("A", "B")}
# The depth pair and the held baseline are read on the ladder, not the depth
# level `arch`: swiglu is deep-shaped and would join deep's cells. The scheme
# pair and the held baseline are read on the data build (`data`), not the
# scheme letter: scheme A also matches AT3 (analysis/RULES.md, Definitions).
COLUMN = {"arch": "ladder", "scheme": "data"}
HOLD = {"arch": ("data", "A"), "scheme": ("ladder", "deep")}


def finals(df):
    return df.loc[df.groupby(["model", "task"])["step"].idxmax()]


def min_predictive_size(seed=1904):
    """One row per (L, intervention, task): the smallest predictive size, or NaN."""
    fin = finals(_pool())
    fin = fin[fin["seed"] == seed]
    mask = load_mask("predictivity")
    rows = []
    for axis, (a, b) in INTERVENTION.items():
        hold_col, hold_val = HOLD[axis]
        sub = fin[fin[hold_col] == hold_val]
        for L, g in sub.groupby("L"):
            piv = g.pivot_table(index=["task", "size"], columns=COLUMN.get(axis, axis), values="primary_score")
            if not {a, b} <= set(piv.columns):
                continue
            piv = piv.dropna(subset=[a, b])
            delta = (piv[a] - piv[b]).unstack("size")
            order = [s for s in S.SIZES if s in delta.columns]
            if len(order) < 2:
                continue
            delta = delta[order]
            ref = order[-1]
            above = {s: passes_gate(mask, delta.index, s) for s in order}
            for task, r in delta.iterrows():
                if not np.isfinite(r[ref]) or r[ref] == 0 or not above[ref][task]:
                    continue
                want = np.sign(r[ref])
                best = np.nan
                # walk down from the reference: keep the run of agreeing sizes
                for s in reversed(order[:-1]):
                    if np.isfinite(r[s]) and np.sign(r[s]) == want and above[s][task]:
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


def agreement_by_size(seed=1904):
    """DA(L, size): the share of languages whose bits per byte picks the same
    winner at `size` as it does at the reference size."""
    fin = finals(_pool())
    fin = fin[(fin["seed"] == seed) & fin["task"].str.startswith("bpb_")
              & (fin["task"] != "bpb_macro")]
    rows = []
    for axis, (a, b) in INTERVENTION.items():
        hold_col, hold_val = HOLD[axis]
        sub = fin[fin[hold_col] == hold_val]
        for L, g in sub.groupby("L"):
            piv = g.pivot_table(index=["task", "size"], columns=COLUMN.get(axis, axis), values="primary_score")
            if not {a, b} <= set(piv.columns):
                continue
            delta = (piv[a] - piv[b]).dropna().unstack("size")
            order = [s for s in S.SIZES if s in delta.columns]
            if len(order) < 2:
                continue
            ref = order[-1]
            want = np.sign(delta[ref])
            for s in order:
                ok = np.sign(delta[s]) == want
                rows.append({"L": int(L), "intervention": axis, "size": s,
                             "reference_size": ref, "n": int(want.notna().sum()),
                             "agree": float(ok[want.notna()].mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = min_predictive_size()
    d["kind"] = d["task"].map(kind)
    print(d.groupby(["intervention", "L"]).agg(n=("task", "size"),
                                               ref=("reference_size", "first")).to_string())
    print()
    print(d.groupby(["kind", "L"])["min_size"].apply(
        lambda s: f"{s.notna().sum()}/{len(s)} predictive").unstack().to_string())
    d.to_csv(os.environ.get("OUT_CSV", "/tmp/min_predictive.csv"), index=False)
