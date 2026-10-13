"""Data of the scaling page's additional views (docs/interactive/views/scaling.js).

Read from committed tables only, nothing refitted: rq01's (task, L) fits
(`rq1_fits.csv`) and its scaling-law error table, rq00's gate mask and the
per-checkpoint scores behind both (rq00's `benchmark_curves.csv` for the
benchmarks, rq06's `bpb_curves.csv` for the per-language BPB), on the cells
rq01 fits: deep, data A, seed 1904, 90M-1.7B.

    scaling_index           one entry per benchmark or BPB task of the fits: chance level, the gate per size, the fit per L
    scaling_curves/<family> the scores behind them, per task, L and size, at the ten evaluated tenths of each run
    scaling_forecast        per-language BPB per size and the power law fitted up to each ladder top (rq01 figure 3)
"""
import pandas as pd

from analysis import paths

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

    out = {"scaling_index": {"sizes": SIZES, "fields": ["r2", "rho", "slope", "n_rungs"], "tasks": index}}
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
    return out
