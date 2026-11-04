"""Data of the scaling page's additional views (docs/interactive/views/scaling.js).

Read from committed tables only, nothing refitted: rq01's (task, L) fits
(`rq1_fits.csv`) and its scaling-law error table, rq00's gate mask and the
per-checkpoint scores behind both (rq00's `benchmark_curves.csv` for the
benchmarks, rq06's `bpb_curves.csv` for the per-language BPB), on the cells
rq01 fits: deep, data A, seed 1904, 90M-1.7B.

    scaling_index           one entry per benchmark or BPB task of the fits: chance level, the gate per size, the fit per L
    scaling_curves/<family> the scores behind them, per task, L and size, at the ten evaluated tenths of each run
    scaling_forecast        per-language BPB per size and the power law fitted up to each ladder top (rq01 figure 3)
    scaling_paper           the benchmarks of the paper's Figures 9 and 10, each with its formats (original, RF, LLM-RF)
    scaling_paper/<base>    per format: each design's final accuracy per size (Figure 9) and each run's accuracy at
                            the ten tenths (Figure 10), both the mean over the tasks in the languages the run trains
                            (rq00's `benchmark_curves.csv`, pool predictivity_seeds: every seed, ladder and data build,
                            90M-1.7B, not filtered), the extra families pooled into their main benchmark
"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis import paths  # noqa: E402
from analysis.grids import paper_name  # noqa: E402
from analysis.utils import benchmark_family  # noqa: E402

ALL = "pretraining/predictivity_seeds"
SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B"]


def _r(x, d=4):
    return None if pd.isna(x) else round(float(x), d)


def _cells(df: pd.DataFrame) -> pd.DataFrame:
    """rq01's grid (`analyze._grid`): deep, data A, seed 1904, with L and size
    read from the cell name; each run's ten evaluated tenths (rule 3)."""
    df = df[df["model"].str.fullmatch(r"lm-[\d.]+[MB]-L\d+(-b\d+)?-deep-seed1904")].copy()
    df["L"] = df["model"].str.extract(r"-L(\d+)-")[0].astype(int)
    df["size"] = df["model"].str.extract(r"^lm-([\d.]+[MB])-")[0]
    df["k"] = (df["frac"] * 10).round()
    return df[((df["k"] / 10 - df["frac"]).abs() < 0.005) & (df["k"] >= 1) & df["size"].isin(SIZES)]


def _paper_curves() -> dict:
    """Figures 9 and 10 (rq00 `curves.py`, `plot_benchmark_size_curves` and
    `plot_benchmark_curves`): per family, the mean over a run's trained-language
    tasks at each of its ten tenths; Figure 9 keeps each run's last tenth and
    pools the runs of one design (K, ladder, data build, seed) per size. The
    extra families (MultiBLiMP extra, NorEval extra) stay apart, as in the paper."""
    b = pd.read_csv(paths.GATE_AND_CURVES / ALL / "benchmark_curves.csv")
    task_chance = pd.read_csv(paths.GATE_AND_CURVES / "pretraining/predictivity/above_random_mask.csv",
                              usecols=["task", "random_baseline"]).set_index("task")["random_baseline"]
    fam = {t: benchmark_family(t) for t in b["task"].unique()}
    b = b.assign(family=b["task"].map(fam), k=(b["frac"] * 10).round().astype(int))
    chance = (b.drop_duplicates(["family", "task"]).assign(c=lambda x: x["task"].map(task_chance))
              .groupby("family")["c"].mean())
    runs = b.groupby(["family", "model", "size", "ladder", "data", "k"], as_index=False)["primary_score"].mean()
    runs["K"] = runs["model"].str.extract(r"-L(\d+)-")[0].astype(int)
    runs["seed"] = runs["model"].str.extract(r"seed(\d+)$")[0].astype(int)
    last = runs[runs["k"] == runs.groupby("model")["k"].transform("max")]
    design = ["K", "ladder", "data", "seed", "size"]
    final = last.groupby(["family", *design], as_index=False)["primary_score"].mean()

    base = lambda f: re.sub(r"^rf(gm)?_", "", f)   # noqa: E731
    fmt = lambda f: "LLM-RF" if f.startswith("rfgm_") else "RF" if f.startswith("rf_") else "Original"   # noqa: E731
    order = {"Original": 0, "RF": 1, "LLM-RF": 2}
    out, index = {}, []
    groups = {}
    for f in sorted(runs["family"].unique()):
        groups.setdefault(base(f), []).append(f)
    for bname, fams in groups.items():
        fams = sorted(fams, key=lambda f: order[fmt(f)])
        formats = {}
        for f in fams:
            r = runs[runs["family"] == f]
            train = [dict(size=size, K=int(K), ladder=ladder, data=data, seed=int(seed),
                          y=[_r(g.set_index("k")["primary_score"].get(k)) for k in range(1, 11)])
                     for (size, K, ladder, data, seed), g in r.groupby(["size", "K", "ladder", "data", "seed"])]
            fin = final[final["family"] == f]
            formats[fmt(f)] = dict(family=f, name=paper_name(f), chance=_r(chance.get(f)),
                                   size=[[int(x.K), x.ladder, x.data, int(x.seed), x.size, _r(x.primary_score)]
                                         for x in fin.itertuples()],
                                   train=train)
        name = paper_name(fams[0])
        out[f"scaling_paper/{bname}"] = formats
        index.append(dict(base=bname, name=name, formats=list(formats)))
    out["scaling_paper"] = dict(benchmarks=sorted(index, key=lambda x: x["name"].lower()))
    return out


def build() -> dict:
    p = paths.SCALING_PREDICTABILITY / ALL
    fits = pd.read_csv(p / "rq1_fits.csv")
    fits = fits[fits["kind"].isin(["benchmark", "bpb"]) & ~fits["task"].str.startswith("bbpb_")]   # the bBPB twins have no curve table
    gate = paths.GATE_AND_CURVES / "pretraining/predictivity"
    mask = pd.read_csv(gate / "above_random_mask.csv").set_index("task")
    chance = mask["random_baseline"]
    language = pd.read_csv(p / "fit_r2.csv", usecols=["task", "language"]).drop_duplicates("task").set_index("task")["language"]

    curves = pd.concat([pd.read_csv(paths.GATE_AND_CURVES / ALL / "benchmark_curves.csv",
                                    usecols=["model", "task", "frac", "primary_score"]),
                        pd.read_csv(paths.LANGUAGE_TRANSFER / ALL / "bpb_curves.csv")])
    curves = _cells(curves[curves["task"].isin(set(fits["task"]))])
    curves = curves.merge(fits[["task", "L", "family"]], on=["task", "L"])   # the (task, L) rq01 fits: trained languages only

    index = []
    for task, f in fits.groupby("task", sort=True):
        row = {"task": task, "family": f["family"].iloc[0], "language": language.get(task), "kind": f["kind"].iloc[0],
               "fits": {int(r.L): [_r(r.r2), _r(r.rho), _r(r.slope, 5), int(r.n_rungs)] for r in f.itertuples()}}
        if task in mask.index:
            row["chance"] = _r(chance[task])
            row["above"] = [None if pd.isna(mask.at[task, s]) else int(mask.at[task, s]) for s in SIZES]
        index.append(row)

    out = {"scaling_index": {"sizes": SIZES, "fields": ["r2", "rho", "slope", "n_rungs"], "tasks": index,
                             "names": {f: paper_name(f) for f in sorted(fits["family"].unique())}}}
    for fam, g in curves.groupby("family"):
        tasks = {}
        for (task, L, size), c in g.groupby(["task", "L", "size"]):
            c = c.sort_values("frac").drop_duplicates("k", keep="last")
            tasks.setdefault(task, {}).setdefault(str(L), {})[size] = [[int(k), _r(v)] for k, v in zip(c["k"], c["primary_score"])]
        out[f"scaling_curves/{fam}"] = tasks

    err = pd.read_csv(p / "scaling_law_error.csv")
    err = err[(err["ladder"] == "deep") & (err["data"] == "A")]
    fin = curves[(curves["family"] == "bpb") & (curves["k"] == 10)]
    out["scaling_forecast"] = {
        "observed": [{"task": t, "L": int(L), "size": s, "bpb": _r(v)} for t, L, s, v in
                     zip(fin["task"], fin["L"], fin["size"], fin["primary_score"])],
        "fits": [{"task": r.task, "language": r.language, "L": int(r.L), "top": r.ladder_top, "n": int(r.n_points),
                  "alpha": _r(r.alpha, 5), "predicted": _r(r.predicted), "observed": _r(r.observed), "error": _r(r.rel_error, 5)}
                 for r in err.itertuples()]}
    return {**out, **_paper_curves()}
