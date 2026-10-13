"""The Model ladder page's run-status view (`ladderPairs` in
docs/interactive/views/ladder.js): per run of the training registry, how far
it trained and how many of its checkpoints are scored, read from the
published ladder report (`ladder_report.csv`, one row per checkpoint), and
whether the committed analysis tables read it (it has a row in rq06's
`bpb_curves.csv`, the table the grid's final BPB already comes from)."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
sys.path.insert(0, str(REPO / "src/pretrain"))
from analysis import paths  # noqa: E402
from snr.download.ladder import ladder_dir  # noqa: E402

TENTHS = 10   # the analysis reads ten evaluated checkpoints per run (RULES.md)


def build() -> dict:
    import launch_trainings as lt
    names = [lt.exp_name(c["size"], c["L"], lad, c["seed"], c["scheme"])
             for lad in lt.LADDERS for c in lt.predictivity_cells(ladder=lad)
             if lad in lt.ladders_for(c["scheme"], c["size"], c["L"])]
    f = ladder_dir() / "ladder_report.csv"
    run = ["run__complete", "run__diverged", "run__off_trend", "run__last_iter", "run__target_iters"]
    w = pd.read_csv(f, usecols=lambda c: c in ("cell", *run) or c.startswith(("bench__", "bpb__")), low_memory=False)
    scored = pd.DataFrame({"cell": w["cell"],
                           "bench": w.filter(like="bench__").notna().any(axis=1),
                           "bpb": w.filter(like="bpb__").notna().any(axis=1)})
    per = scored.groupby("cell")[["bench", "bpb"]].sum().join(w.groupby("cell")[run].first())
    analysed = set(pd.read_csv(paths.LANGUAGE_TRANSFER / "pretraining/predictivity_seeds/bpb_curves.csv",
                               usecols=["model"])["model"])
    out = {}
    for n in names:
        if n not in per.index:
            out[n] = dict(state="queued", progress=0, bench=0, bpb=0, analysed=False)
            continue
        r = per.loc[n]
        progress = min(1.0, float(r.run__last_iter) / float(r.run__target_iters)) if r.run__target_iters else None
        if r.run__diverged == 1 or r.run__off_trend == 1:
            state = "excluded"
        elif r.run__complete != 1:
            state = "training"
        elif r.bench >= TENTHS:
            state = "evaluated"
        else:
            state = "evaluating"
        out[n] = dict(state=state, progress=None if progress is None else round(progress, 3),
                      bench=int(r.bench), bpb=int(r.bpb),
                      # the 3B rung is read only by the size-generalisation question (rq10)
                      analysed=None if n.startswith("lm-3B-") else n in analysed)
    missing = sorted(set(per.index) - set(names))
    if missing:
        print(f"ladder_status: {len(missing)} report cell(s) not in the registry, left out: {', '.join(missing)}")
    return {"ladder_status": dict(runs=out, tenths=TENTHS, report_mtime=pd.Timestamp(f.stat().st_mtime, unit="s").strftime("%Y-%m-%d"))}
