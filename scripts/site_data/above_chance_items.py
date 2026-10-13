"""Data of the above-chance-items page's additional views
(rq12_above_chance_items): per (benchmark, language) task, how many items it
has and how many its 1.7B runs answer above chance, whether the gate passes it
at 1.7B before and after the selection, and per size its final-checkpoint SNR
(full benchmark vs gate-then-items) and DA-size against the 1.7B final
(multi-axis pairs, kept-items truth, as the README's paired gain). Rows of
above_chance_items_selection.csv, _snr_per_task.csv and
_da_size_per_task_both_axes.csv, only joined. None until the analysis is in
the checkout."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
from analysis import paths  # noqa: E402
from analysis.grids import paper_name  # noqa: E402

PRED = "pretraining/predictivity"
KEYS = ["snr_full", "snr_kept", "sig", "noise", "da_full", "da_kept"]


def build() -> dict:
    d = paths.GATE_AND_CURVES.parent / "rq12_above_chance_items" / PRED
    if not d.exists():
        print("above_chance_items_tasks: skipped, its analysis folder is not in this checkout")
        return {}
    sel = pd.read_csv(d / "above_chance_items_selection.csv")
    snr = pd.read_csv(d / "above_chance_items_snr_per_task.csv")
    snr = snr[snr["ordering"].isin(["full", "gate_then_items"]) & ~snr["gated"]]
    snr = snr.pivot_table(index=["task", "size"], columns="ordering", values=["snr", "signal", "noise_kfold"])
    da = pd.read_csv(d / "above_chance_items_da_size_per_task_both_axes.csv")
    da = da[(da["axes"] == "multi-axis") & (((da["truth"] == "full") & (da["ordering"] == "full"))
                                            | ((da["truth"] == "kept") & (da["ordering"] == "gate_then_items")))]
    da = da.pivot_table(index=["task", "size"], columns="ordering", values="da")
    per = {}
    for (task, size), r in snr.iterrows():
        per.setdefault(task, {})[size] = dict(
            snr_full=r.get(("snr", "full")), snr_kept=r.get(("snr", "gate_then_items")),
            sig=r.get(("signal", "gate_then_items")) / r.get(("signal", "full")) if r.get(("signal", "full")) else None,
            noise=r.get(("noise_kfold", "gate_then_items")) / r.get(("noise_kfold", "full")) if r.get(("noise_kfold", "full")) else None)
    for (task, size), r in da.iterrows():
        per.setdefault(task, {}).setdefault(size, {}).update(da_full=r.get("full"), da_kept=r.get("gate_then_items"))
    # per size one array in KEYS order (null where a reading has no value), to keep the file small
    clean = lambda o: [None if o.get(k) is None or o[k] != o[k] else round(float(o[k]), 3) for k in KEYS]
    per = {t: {s: c for s, c in ((s, clean(v)) for s, v in by.items()) if c.count(None) < len(KEYS)}
           for t, by in per.items()}
    rows = []
    for r in sel.to_dict("records"):
        if pd.isna(r["n_kept"]):
            continue
        rows.append(dict(task=r["task"], b=r["benchmark"], lang=r["language"], items=int(r["n_items"]),
                         kept=int(r["n_kept"]), runs=int(r["n_reference_runs"]), chance=round(float(r["chance"]), 3),
                         gate_full=bool(r["gate_full_at_reference"] == 1),
                         gate_items=None if pd.isna(r["gate_items_at_reference"]) else bool(r["gate_items_at_reference"] == 1),
                         sizes=per.get(r["task"], {})))
    return {"above_chance_items_tasks": dict(keys=KEYS, names={b: paper_name(b) for b in sorted(sel["benchmark"].unique())},
                                             tasks=rows)}
