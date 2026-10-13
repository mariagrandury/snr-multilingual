"""Data of the English-only page's additional views (docs/interactive/views/english-only.js).

Read from rq13's committed tables (pool `predictivity`, seed 1904, finals),
nothing recomputed; empty until rq13 is in the checkout, like
build_site_data.english_only().

    english_only_regimes  the scaling regime of every English accuracy task in each English corpus at L1 (the
                          tagged table, `filtered` = at chance at too many sizes to fit), the L1 scores per size
                          behind it, and the regime shares with the bBPB twins counted (the summary table)
    english_only_bench    per English task and size: the English-only score against each multilingual L, with
                          the seed sd of the English-only score and the gate (True = either side at chance), plus each
                          task's chance level
"""
import pandas as pd

from analysis import paths

D = paths.GATE_AND_CURVES.parent / "rq13_english_only/pretraining/predictivity"
CORPUS = {"lm-L1-deep-seed1904": "DCLM-Edu", "lm-L1-dclmP-deep-seed1904": "DCLM", "lm-L1-fweb-deep-seed1904": "FineWeb"}


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not (isinstance(v, float) and v != v)}
            for r in df.to_dict("records")]


def build() -> dict:
    if not D.exists():
        print("english_only (site_data): skipped, rq13 is not in this checkout")
        return {}
    chance = pd.read_csv(paths.GATE_AND_CURVES / "pretraining/predictivity/above_random_mask.csv",
                         usecols=["task", "random_baseline"]).set_index("task")["random_baseline"]

    tagged = pd.read_csv(D / "english_only_scaling_regimes_l1_accuracy_tagged.csv")
    tagged = tagged[["task", "family", "name", "popular", "format", "corpus", "regime", "r2_size", "rho_size", "r2_trajectory"]]
    above = pd.read_csv(D / "english_only_above_random_per_task.csv")
    above = above[above["family"].isin(CORPUS) & above["task"].isin(set(tagged["task"]))]
    curves = {}
    for r in above.itertuples():
        curves.setdefault(r.task, {}).setdefault(CORPUS[r.family], {})[r.size] = [round(r.primary_score, 4), int(r.above)]
    summary = pd.read_csv(D / "english_only_scaling_regimes_summary.csv")

    s = pd.read_csv(D / "english_only_scores_per_task.csv")
    s = s[s["scoring"].isin(["acc", "bpb"])]
    names, popular = dict(zip(tagged["task"], tagged["name"])), set(tagged.loc[tagged["popular"], "task"])
    others = sorted(s["L"].unique())
    scores = {}
    for (task, arch, size), g in s.groupby(["task", "arch", "size"]):
        g = g.set_index("L").reindex(others)
        sd = g["seed_sd"].dropna()
        scores.setdefault(task, {}).setdefault(arch, {})[size] = {
            "l1": round(g["score_l1"].dropna().iloc[0], 4), "sd": round(sd.iloc[0], 4) if len(sd) else None,
            "other": [None if v != v else round(v, 4) for v in g["score_other"]],
            "gated": [bool(v) for v in g["gated"].fillna(True)]}
    return {
        "english_only_regimes": {"rows": _records(tagged), "curves": curves, "summary": _records(summary),
                                 "chance": {t: round(chance[t], 4) for t in curves if t in chance.index}},
        "english_only_bench": {
            "others": [int(L) for L in others],
            "tasks": [{"task": t, "name": names.get(t), "popular": t in popular, "scoring": g["scoring"].iloc[0],
                       **({"chance": round(chance[t], 4)} if t in chance.index and chance[t] == chance[t] else {})}
                      for t, g in s.groupby("task")],
            "scores": scores},
    }
