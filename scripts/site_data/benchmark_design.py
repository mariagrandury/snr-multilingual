"""Data of the benchmark-design page's additional view (rq09_benchmark_design):
one row per (benchmark, language) task with its SNR at 1.7B and its DA-size
(the mean over the proxies 90M-1B, every gated task and the paper's
above_66_either tasks), and one row per benchmark family with the
characteristics the analysis groups it by. Rows of per_task_snr.csv,
design_da_size_per_task_mono_axis.csv and benchmark_characteristics.csv."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
from analysis import paths  # noqa: E402

PRED = "pretraining/predictivity"
CHARS = ["benchmark", "domain", "curation_category", "curation", "source_origin", "source",
         "format", "format_type", "n_options", "passage"]


def build() -> dict:
    d = paths.BENCHMARK_DESIGN / PRED
    snr = pd.read_csv(d / "per_task_snr.csv")[["task", "family", "language", "snr_mpd_1.7B"]]
    da = pd.read_csv(d / "design_da_size_per_task_mono_axis.csv")
    # a task's DA-size is its mean over the proxies that have one (the analysis' own reading)
    da = (da.groupby(["population", "family", "task", "language"])["da"].agg(["mean", "count"])
          .reset_index().pivot_table(index=["family", "task", "language"], columns="population",
                                     values=["mean", "count"]))
    da.columns = [f"{'da' if s == 'mean' else 'n'}_{'all' if p == 'unfiltered' else 'rel'}" for s, p in da.columns]
    t = snr.merge(da.reset_index(), on=["family", "task", "language"], how="outer")
    t = t.rename(columns={"snr_mpd_1.7B": "snr"})
    chars = pd.read_csv(d / "benchmark_characteristics.csv").set_index("family")
    chars["passage"] = chars["passage"].map({True: "passage", False: "no passage", "True": "passage", "False": "no passage"})
    chars["n_options"] = chars["n_options"].map(lambda v: None if pd.isna(v) else f"{int(v)} options")
    t = t[t["family"].isin(chars.index)].round(4)
    rows = [{k: v for k, v in r.items() if not (isinstance(v, float) and v != v)} for r in t.to_dict("records")]
    for r in rows:
        for k in ("n_all", "n_rel"):
            if k in r:
                r[k] = int(r[k])
    fams = {f: {k: (None if pd.isna(v) else v) for k, v in r.items()} for f, r in chars[CHARS].to_dict("index").items()}
    return {"benchmark_design_tasks": dict(families=fams, tasks=rows)}
