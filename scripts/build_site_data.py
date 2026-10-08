"""Export the committed analysis tables as the compact JSON the site's
interactive views read (docs/interactive/data/). No statistic is computed
here — every number is a row of an rqNN_ output, only filtered and renamed —
except the grid and the data mixtures, which are read from the training
registry (src/pretrain/launch_trainings.py) and the builder's mixture plans,
and the recommender's minimum-tokens table (lookup()), a threshold read off
the tokens-seen cell table.

    python3 scripts/build_site_data.py
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
sys.path.insert(0, str(REPO))
from analysis import paths  # noqa: E402

OUT = REPO / "docs/interactive/data"
PRED = "pretraining/predictivity"
ALL = "pretraining/predictivity_seeds"   # every seed and data build


def rows(df: pd.DataFrame) -> list[dict]:
    """Records with floats rounded to 4 digits and NaN dropped."""
    return [{k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in r.items() if not (isinstance(v, float) and v != v)}
            for r in df.to_dict("records")]


def table(df: pd.DataFrame) -> dict:
    """Columnar form for the large tables (keys are most of a record's bytes)."""
    df = df.round(4).astype(object).where(df.notna(), None)
    return dict(columns=list(df.columns), data=df.values.tolist())


def no_byline(text: str) -> str:
    """Drop a citation byline ("(X et al.)", "(X & Y)", "(X 2023)"): no
    author names on the site during double-blind review."""
    return re.sub(r"\s*\([^)]*(?:et al|&|\d{4})[^)]*\)", "", text).strip()


def dump(name: str, obj) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":")))
    print(f"{name}.json  {(OUT / f'{name}.json').stat().st_size // 1024} KB")


def _grid():
    """The current training grid, from the registry the launcher itself reads
    (src/pretrain/launch_trainings.py: DATA_SCHEMES x LADDERS x seeds)."""
    sys.path.insert(0, str(REPO / "src/pretrain"))
    import launch_trainings as lt
    return lt


def ladder() -> dict:
    lt = _grid()
    models = json.loads((REPO / "configs/models.json").read_text())["models"]
    sets = {}
    for f in sorted((REPO / "src/pretrain/data").glob("language_sets_scheme*.json")):
        d = json.loads(f.read_text())
        sets[d["scheme"]] = d["sets"]
    langs = json.loads((REPO / "configs/languages.json").read_text())
    iso2 = langs["fineweb_iso2"]
    english = {"A": "DCLM (edu-filtered)", "DCLMP": "DCLM (no edu filter)", "FWEB": "FineWeb"}
    cells = []
    for lad, levels in lt.LADDERS.items():
        for c in lt.predictivity_cells(ladder=lad):
            scheme, size, L = c["scheme"], c["size"], c["L"]
            if lad not in lt.ladders_for(scheme, size, L):
                continue
            name = lt.exp_name(size, L, lad, c["seed"], scheme)
            cfg, m = lt.DATA_SCHEMES[scheme], models.get(name, {})
            pre = m.get("stages", {}).get("pretraining", {})
            # One field per intervention axis; None where the axis does not
            # apply to the cell (the second language exists only at L = 2,
            # the language list and the temperature only with a FineWeb half).
            second = sets[cfg["sets"]]["FW_L2"][0] if L == 2 else None
            cells.append(dict(
                name=name, size=size, L=L, seed=c["seed"], scheme=scheme, sets=cfg["sets"],
                ladder=lad, arch=levels["arch"], activation=levels["activation"],
                optimizer=levels["optimizer"],
                list=("B" if cfg["sets"] == "B" else "A") if L > 2 else None,
                T=cfg["temp"] if L > 1 else None,
                lang2=iso2.get(second.split("_")[0], second) if second else None,
                english=english.get(scheme, english["A"]),
                params=m.get("params"), n_non_emb=m.get("n_non_emb"), d_model=m.get("d_model"),
                tokens=pre.get("tokens"), n_ckpts=len(pre.get("checkpoints", {}).get("all", []))))
    bpb = pd.read_csv(paths.LANGUAGE_TRANSFER / ALL / "bpb_curves.csv")
    final = bpb[bpb.frac == bpb.groupby("model").frac.transform("max")]
    final_bpb = {m: dict(zip(g.task.str.removeprefix("bpb_"), g.primary_score.round(4)))
                 for m, g in final.groupby("model")}
    return dict(cells=cells, final_bpb=final_bpb, language_sets=sets,
                languages=langs["languages"], fineweb_iso2=iso2)


def mixtures(cells: list[dict]) -> dict:
    """Per-language token shares of every (scheme, L) training mixture: the
    FineWeb-2 half split as the builder's own plan splits it (read through
    src/pretrain/data/data_progress.py, which prefers plan.json and never
    estimates silently), plus the fixed English share. Token counts are the
    share times the training budget of the mixture's 1.7B reference run (its
    largest planned run where it has none)."""
    lt = _grid()
    sys.path.insert(0, str(REPO / "src/pretrain/data"))
    import data_progress as dp
    order = {s: i for i, s in enumerate(lt.LADDER)}
    out = []
    for L, scheme in dp.MIXTURES:
        cfg = lt.DATA_SCHEMES[scheme]
        if cfg.get("english"):   # L = 1 English-source variants: no mixture to show
            continue
        langs = dp.language_sets(scheme).get(L, [])
        col = dp.column(dp.DEFAULT_DATA_DIR, L, scheme, langs, {})
        if langs and col["source"] in ("—", dp.ESTIMATED):
            print(f"mixtures: skip {scheme} L{L} (no exact per-language counts: {col['source']})")
            continue
        fw_total = sum(col["tokens"].values())
        en = 1.0 if L == 1 else lt.EN_SHARE / 100
        share = {"eng_Latn": en, **{k: (1 - en) * v / fw_total for k, v in col["tokens"].items()}}
        runs = [c for c in cells if c["scheme"] == scheme and c["L"] == L and c["tokens"]]
        ref = max(runs, key=lambda c: (c["size"] == "1.7B", order[c["size"]])) if runs else None
        out.append(dict(scheme=cfg["sets"] if scheme != "AT3" else "A", T=cfg["temp"], L=L,
                        name=scheme, ref_size=ref and ref["size"], ref_tokens=ref and ref["tokens"],
                        shares={k: round(v, 6) for k, v in share.items()}))
    return dict(mixtures=out)


def evaluation(languages: list[str]) -> dict:
    """The ladder page's score curves and benchmark cards.

    Curves: rq00's score_curves.csv (per language: benchmark score per size
    along the run, mean over the cells that train the language) and, for
    "all", predictivity_seeds/benchmark_curves.csv averaged the same way (per
    cell over its trained-language tasks, then over the cells of a size).
    The x grid is training tokens in Chinchilla multiples (5 = the full run);
    `sizes` carries each rung's full-run tokens and FLOPs parameters (baseline
    ladder) so the page can show tokens or compute = 6 N D.
    Cards: configs/tasks.json (n_options, languages, the example and dataset
    written by src/evals/scripts/derive_task_options.py --examples)."""
    from analysis.utils import benchmark_family
    from mkdocs_hooks import _ORGS, _PEOPLE
    from snr.download.ladder import cell_params
    blocked = re.compile(rf"{_PEOPLE}|{_ORGS}|epfl|swiss", re.I)
    hp = json.loads((REPO / "src/pretrain/hyperparams/hyperparams_deep.json").read_text())["configs"]

    sc = pd.read_csv(paths.GATE_AND_CURVES / PRED / "score_curves.csv")
    sc["c"] = (sc["chinchilla"] * 4).round() / 4          # the saves drift a few iterations off the grid
    sc["w"] = sc["score"] * sc["models"]
    g = sc.groupby(["language", "family", "size", "c"])
    cur = (g["w"].sum() / g["models"].sum()).rename("score").reset_index()
    chance = sc.groupby(["language", "family"])["chance"].mean()

    bc = pd.read_csv(paths.GATE_AND_CURVES / ALL / "benchmark_curves.csv")
    bc["family"] = bc["task"].map(benchmark_family)
    bc["size"] = bc["model"].str.extract(r"^lm-([\d.]+[MB])-", expand=False)
    bc["c"] = (bc["frac"] * 5 * 4).round() / 4
    per_cell = bc.groupby(["model", "size", "family", "c"])["primary_score"].mean().reset_index()
    allc = per_cell.groupby(["size", "family", "c"])["primary_score"].mean().rename("score").reset_index()
    allc["language"] = "all"
    cur = pd.concat([cur, allc[cur.columns]])
    fam_chance = chance.groupby("family").mean()

    grid = sorted(cur["c"].unique())
    pos = {c: i for i, c in enumerate(grid)}
    curves = {}
    for (lang, fam, size), d in cur.groupby(["language", "family", "size"]):
        ys = [None] * len(grid)
        for c, s in zip(d["c"], d["score"]):
            ys[pos[c]] = round(float(s), 4)
        ch = fam_chance.get(fam) if lang == "all" else chance.get((lang, fam))
        node = curves.setdefault(lang, {}).setdefault(fam, {"chance": None if ch != ch else ch, "sizes": {}})
        node["sizes"][size] = ys
        if node["chance"] is not None:
            node["chance"] = round(float(node["chance"]), 4)
    sizes = [s for s in ["90M", "175M", "350M", "600M", "1B", "1.7B", "3B"] if s in set(cur["size"])]

    tj = json.loads((REPO / "configs/tasks.json").read_text())
    meta = tj["benchmarks"]
    fams = sorted({f for lang in curves.values() for f in lang})
    cards = {}
    for fam in fams:
        tasks = {k: v for k, v in tj["tasks"].items() if v["benchmark"] == fam and "pretraining" in v["stages"]}
        opts = pd.Series([v["n_options"] for v in tasks.values() if "n_options" in v])
        m = meta.get(fam) or meta.get(re.sub(r"^rf(gm)?_", "", fam)) or {}
        examples = {}
        for k in sorted(tasks):                            # first task of each language, in name order
            v = tasks[k]
            if "example" in v and v["language"] not in examples and not fam.startswith("rfgm_"):
                src = v.get("source")
                examples[v["language"]] = dict(task=k, **v["example"], source=src,
                                               anonymized=bool(src and blocked.search(src)))
                if examples[v["language"]]["anonymized"]:
                    examples[v["language"]]["source"] = None
        cards[fam] = dict(
            # display name without its byline ("(X et al.)", "(X & Y)", "(X 2023)"): no author names on the site
            name=no_byline(m.get("name", fam)),
            venue=m.get("venue"), paper=m.get("paper") or m.get("url"),
            format=m.get("format") and re.sub(r"\s*\([^)]*\.jsonl?\)", "", m["format"]),   # no internal file paths
            n_options=int(opts.mode()[0]) if len(opts) else None,
            options_vary=bool(opts.nunique() > 1),
            languages=sorted({v["language"] for v in tasks.values()} - {"multi"}),
            # INCLUDE v2 "og" items have no English task of their own: the
            # English version of the same items is the include_v2_en family
            english_twin="include_v2_en" if fam == "include_v2_og" else None,
            examples=examples)
    missing = [f for f in fams if f not in meta and re.sub(r"^rf(gm)?_", "", f) not in meta]
    if missing:
        print(f"evaluation: no tasks.json 'benchmarks' entry for {missing}")
    return dict(grid=grid, sizes={s: dict(tokens=hp[s]["predictivity"]["train_tokens"], params=cell_params(s, "deep"))
                                  for s in sizes},
                languages=["all"] + [l for l in languages if l in curves], curves=curves, benchmarks=cards)


GATE_SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B"]
# Public-model lines left off the site: internal checkpoints that are not
# released, so naming them would identify the authors (double-blind review).
UNRELEASED_LINES = {"apertus3-a06", "ap-from8b-TOP256"}
UNRELEASED = r"apertus3|a06|from8b|distill"   # guard for a line renamed upstream


def _names(keys) -> dict:
    """Display name per benchmark key (`rf_belebele` -> `Belebele RF`), the
    paper's own naming (analysis.grids.paper_name)."""
    from analysis.grids import paper_name
    return {k: paper_name(k) for k in sorted(set(keys))}


def _public_readers(tasks: set) -> tuple[dict, pd.DataFrame | None]:
    """(readers, floors) over the released public models only. readers: task
    -> the models whose own final run clears the gate on it, smallest first,
    as [model, size, post_trained]. floors: task, external_floor, external_bin
    recomputed the way above_random_external.floors does, but with the
    unreleased lines dropped, so no internal checkpoint sets a floor colour.
    ({}, None) when the external parquet is not in this checkout (set
    SNR_MULTILINGUAL_DATA_DIR)."""
    from analysis.rq00_gate_and_curves.above_random import scores_and_mask
    from analysis.rq00_gate_and_curves.above_random_external import BINS, external_models, floor_of
    from analysis.utils import build_snr_pool
    try:
        df = build_snr_pool("external", untrained=True)
    except ValueError:            # no external parquet: build_snr_pool has nothing to concatenate
        print("gate: no external parquet in this checkout, the floors view lists no public model")
        return {}, None
    m = external_models(df)
    m = m[~m["line"].isin(UNRELEASED_LINES)
          & ~m["line"].str.contains(UNRELEASED, case=False)
          & ~m["model"].str.contains(UNRELEASED, case=False)]
    df = df[df["model"].isin(m["model"])]
    _, mask, _, runs = scores_and_mask(df, runs=True)[:4]
    fl = mask.rename_axis("task").reset_index()
    fl = pd.DataFrame({"task": fl["task"], "external_floor": fl["task"].map(floor_of(fl))})
    fl["external_bin"] = fl["external_floor"].map({lvl: name for name, lv in BINS for lvl in lv})
    runs = runs[runs["above"].eq(1) & runs["task"].isin(tasks)].merge(m, on="model").sort_values(["params", "model"])
    return {t: g[["model", "size", "post_trained"]].values.tolist() for t, g in runs.groupby("task")}, fl


def gate() -> dict:
    """rq00, the page's five views: the gate per (task, size) (share of runs
    above chance), the smallest size above chance per (benchmark, language),
    the share of cells above chance against the tokens a language was trained
    on (rq01 tokens_seen.py, every checkpoint of seed 1904), the format story
    (one INCLUDE item in each format, the reformulation figure's table) and the
    benchmark floors of the public models."""
    g = paths.GATE_AND_CURVES / PRED
    share = pd.read_csv(g / "above_random_share.csv")
    sizes = [s for s in GATE_SIZES if s in share.columns]

    first = pd.read_csv(g / "first_size_above_random_paper.csv")
    langs = list(dict.fromkeys(first["language"]))          # the paper's order: English, then scheme A's resource order
    order = list(first.assign(ok=first["level"].ne("never")).groupby("benchmark")["ok"].sum()
                 .sort_values(ascending=False, kind="stable").index)

    tok = _tokens_dir()
    by_b = pd.read_csv(tok / "pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_points.csv")
    every = pd.read_csv(tok / "pass_prob_vs_train_tokens_all_1904_ckpts.csv").assign(benchmark="all")
    pts = pd.concat([by_b, every])[["benchmark", "model_size", "share_above", "n_cells", "tokens", "bin_lo", "bin_hi"]]

    tj = json.loads((REPO / "configs/tasks.json").read_text())["tasks"]
    fmt = {k: dict(tj[t]["example"], task=t, language=tj[t]["language"])
           for k, t in [("og", "include_v2_og_greek_greece"), ("en", "include_v2_en_greek_greece"),
                        ("letter", "include_base_44_greek"), ("rf", "rf_include_base_44_greek"),
                        ("llm", "rfgm_include_base_44_greek")] if "example" in tj.get(t, {})}

    rf = pd.read_csv(paths.GATE_AND_CURVES.parent / "rq00_task_reformulation/reformulations_gate_paper.csv")

    ext = pd.read_csv(g / "above_random_external.csv")
    floors = ext[["task", "family", "language", "n_options", "ladder_floor", "external_floor", "external_bin"]]
    readers, public = _public_readers(set(floors["task"]))
    if public is not None:        # the colour follows the listed public models only
        floors = (floors.drop(columns=["external_floor", "external_bin"]).merge(public, on="task")
                  .dropna(subset=["external_floor"]))
    else:
        print("gate: WARNING floors keep above_random_external.csv buckets, which include unreleased lines")

    keys = list(share["family"]) + list(first["benchmark"]) + list(by_b["benchmark"]) + list(rf["family"]) + list(floors["family"])
    return dict(
        sizes=sizes, names=_names(keys),
        tasks=table(share[["task", "family", "language", "n_options", "random_baseline", *sizes]]),
        first_size=dict(languages=langs, benchmarks=order, levels=sizes,
                        cells=first[["benchmark", "language", "level"]].values.tolist()),
        pass_tokens=rows(pts),
        format_example=fmt,
        reformulation=rows(rf),
        floors=dict(tasks=rows(floors), readers=readers))


def gate_example() -> dict:
    """rq00's gate explained on any task (above_random_example.py draws one):
    per task its chance level, item count, the score a run needs, the mask per
    size, every run that trains the language with its final score, Wilson
    lower bound and verdict, its score at each tenth of the run, and rq02's
    DA-size (to 1.7B) and DA-ckpt (90 % of the run) cells of the task."""
    import numpy as np
    from analysis.rq00_gate_and_curves.above_random import wilson_lcb

    def needed(n: int, chance: float):
        """The smallest accuracy whose Wilson bound over n items clears chance
        (above_random_example.needed_score), None when even 100 % does not."""
        ks = np.arange(n + 1)
        ok = ks[wilson_lcb(ks / n, np.full(n + 1, n)) > chance]
        return round(float(ok[0] / n), 4) if len(ok) else None

    g = paths.GATE_AND_CURVES / PRED
    meta = pd.read_csv(g / "above_random_scores.csv").set_index("task")
    mask = pd.read_csv(g / "above_random_mask.csv").set_index("task")
    sizes = [s for s in GATE_SIZES if s in mask.columns]
    runs = pd.read_csv(g / "above_random_runs.csv")
    runs = runs[runs["trained"] & runs["above"].notna() & runs["bucket"].isin(sizes)]   # no verdict: no chance level
    runs = runs[runs["task"].map(meta["language"]).notna() & ~runs["task"].map(meta["language"]).isin(["multi", "??"])]
    cur = pd.read_csv(paths.GATE_AND_CURVES / ALL / "benchmark_curves.csv")
    cur["k"] = (cur["frac"] * 10).round(6)
    cur = cur[cur["k"] % 1 == 0].merge(runs[["task", "model"]], on=["task", "model"])
    traj = {key: dict(zip(d["k"].astype(int), d["primary_score"].round(4))) for key, d in cur.groupby(["task", "model"])}
    da = pd.read_csv(paths.DECISION_ACCURACY / PRED / "da_all_per_benchmark_multi_axes.csv")
    da_size = da[(da["da_def"] == "DA-size") & (da["size_to"] == "1.7B")]
    da_ckpt = da[(da["da_def"] == "DA-ckpt") & da["comparison"].str.startswith("f90@")]
    cells = {t: dict(zip(d["size_from"], d["decision_acc"].round(3))) for t, d in da_size.groupby("task")}
    ckpts = {t: dict(zip(d["size_from"], d["decision_acc"].round(3))) for t, d in da_ckpt.groupby("task")}
    models = sorted(runs["model"].unique())
    idx = {m: i for i, m in enumerate(models)}
    out = {}
    for task, d in runs.groupby("task"):
        r = meta.loc[task]
        out[task] = dict(
            family=r["family"], language=r["language"], chance=round(float(r["random_baseline"]), 4),
            n_items=int(r["n_items"]), need=needed(int(r["n_items"]), float(r["random_baseline"])),
            mask=[None if pd.isna(mask.at[task, s]) else int(mask.at[task, s]) for s in sizes],
            runs=[[idx[m], b, round(float(sc), 4), round(float(lcb), 4), int(a),
                   [traj.get((task, m), {}).get(k) for k in range(1, 11)]]
                  for m, b, sc, lcb, a in d[["model", "bucket", "score", "lcb", "above"]].itertuples(index=False)],
            da_size=cells.get(task, {}), da_ckpt=ckpts.get(task, {}))
    # one file per benchmark, fetched when the reader picks it (all of them are ~4 MB)
    fams = {}
    for task, t in out.items():
        fams.setdefault(t["family"], {})[task] = t
    index = dict(sizes=sizes, models=models, names=_names(fams),
                 families={f: {k: v["language"] for k, v in sorted(ts.items())} for f, ts in sorted(fams.items())})
    return index, fams


def recipe() -> dict:
    """rq11 (which benchmark, posed and scored how) and rq10 (does the 1.7B
    ranking hold at 3B): the recommendation table, the DA-size per variant and
    size, and the DA of each proxy against the 3B and the 1.7B reference on the
    same design pairs."""
    from analysis.utils import RELIABLE_DA
    r = paths.EVALUATION_RECIPE / PRED
    ref = paths.SIZE_GENERALISATION / PRED
    rec = pd.read_csv(r / "recipe_da_size_recommendation_multi_axes.csv")
    return dict(
        tau=RELIABLE_DA, names=_names(rec["benchmark"]), recommendation=rows(rec),
        overview=rows(pd.read_csv(r / "recipe_da_size_overview_multi_axes.csv")),
        above_reference=rows(pd.concat([pd.read_csv(ref / "above_reference_3B.csv"),
                                        pd.read_csv(ref / "above_reference_1.7B_design3B.csv")])
                             .drop(columns=["non_emb", "n_matching"])))


def _tokens_dir() -> Path:
    """The tokens-seen tables (share above chance against the tokens of the
    language seen): rq00_chance_vs_train_tokens/ once that folder exists,
    rq01_scaling_predictability/ before the move."""
    d = getattr(paths, "CHANCE_VS_TRAIN_TOKENS", None)
    return (d if d is not None and (d / ALL).exists() else paths.SCALING_PREDICTABILITY) / ALL


def lookup(cells: list[dict]) -> dict:
    """The recommender's minimum-tokens table: per (benchmark, language, proxy
    size), the fewest training tokens of the language from which the benchmark
    stays above chance — the table form of the tokens-seen panels
    (pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.csv, every evaluated
    tenth of every seed-1904 run). A (benchmark, language, size, L, tenth) cell
    is above chance when at least MIN_SHARE of the benchmark's tasks in that
    language are (each task by rule 1's cell rule, as in the table), the same
    family-level rule as rq00's first_size_above_random. The threshold is the
    smallest token count from which EVERY cell with at least as many tokens is
    above chance (rq00's 'stays above' reading, on the token axis instead of the
    size axis); null when the cell with the most tokens is still at chance.
    `lo`/`hi` are the token range the size's cells cover. The default budget of
    a size is the most common budget of its runs in the registry."""
    from analysis.rq00_gate_and_curves.above_random import MIN_SHARE
    c = pd.read_csv(_tokens_dir() / "pass_prob_vs_train_tokens_by_benchmark_1904_ckpts.csv",
                    usecols=["benchmark", "task", "language", "model_size", "language_scheme", "frac",
                             "train_tokens", "above_chance"])
    if c["train_tokens"].isna().any():
        print(f"lookup: {c['train_tokens'].isna().mean():.0%} of the cells have no token count "
              "(the mixture plans were unreachable when rq tokens_seen ran) and are left out")
    c = c.dropna(subset=["train_tokens"])
    key = ["benchmark", "language", "model_size"]
    cell = (c.assign(above=c["above_chance"].astype(bool))
            .groupby([*key, "language_scheme", "frac"])
            .agg(tokens=("train_tokens", "mean"), share=("above", "mean"), n_tasks=("task", "nunique"))
            .reset_index())
    cell["ok"] = cell["share"].ge(MIN_SHARE)

    def threshold(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("tokens")
        bad = g.loc[~g["ok"], "tokens"]
        above = g.loc[g["tokens"] > bad.max(), "tokens"] if len(bad) else g["tokens"]
        return pd.Series(dict(min_tokens=above.min() if len(above) else None, lo=g["tokens"].min(),
                              hi=g["tokens"].max(), n_cells=len(g), n_tasks=g["n_tasks"].max()))

    t = cell.groupby(key).apply(threshold, include_groups=False).reset_index()
    t[["min_tokens", "lo", "hi"]] = t[["min_tokens", "lo", "hi"]].round(-6)
    sizes = [s for s in GATE_SIZES if s in set(t["model_size"])]
    budget = {s: int(pd.Series([r["tokens"] for r in cells if r["size"] == s]).mode().iloc[0]) for s in sizes}
    order = list(t.assign(ok=t["min_tokens"].notna()).groupby("benchmark")["ok"].mean()
                 .sort_values(ascending=False, kind="stable").index)
    return dict(sizes=sizes, budget=budget, min_share=MIN_SHARE, benchmarks=order, names=_names(order),
                table=table(t[[*key, "min_tokens", "lo", "hi", "n_cells", "n_tasks"]]
                            .rename(columns={"model_size": "size"})))


def above_chance_items() -> dict | None:
    """rq12: the benchmarks reduced to the items the 1.7B reference answers
    above chance (None until the analysis is merged into this checkout)."""
    d = paths.GATE_AND_CURVES.parent / "rq12_above_chance_items" / "pretraining/predictivity"
    if not d.exists():
        return None
    return dict(survival=rows(pd.read_csv(d / "above_chance_items_survival.csv")),
                da_size=rows(pd.read_csv(d / "above_chance_items_da_size_both_axes.csv")),
                gate=rows(pd.read_csv(d / "above_chance_items_gate_pass_share.csv")))


def english_only() -> dict | None:
    """rq13: do the English-only (L1) runs read the English benchmarks better
    than the multilingual runs of the same size (None until merged)."""
    d = paths.GATE_AND_CURVES.parent / "rq13_english_only" / "pretraining/predictivity"
    if not d.exists():
        return None
    return dict(verdict=rows(pd.read_csv(d / "english_only_verdict.csv")),
                scores=rows(pd.read_csv(d / "english_only_scores_summary.csv")))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lad = ladder()
    dump("ladder", lad)
    dump("mixtures", mixtures(lad["cells"]))
    dump("lookup", lookup(lad["cells"]))
    dump("evaluation", evaluation(sorted(lad["languages"])))
    dump("facts", json.loads((REPO / "documents/ladder-facts.json").read_text()))

    dump("gate", gate())
    index, fams = gate_example()
    dump("gate_example", index)
    (OUT / "gate_example").mkdir(exist_ok=True)
    for fam, tasks in fams.items():
        (OUT / "gate_example" / f"{fam}.json").write_text(json.dumps(tasks, separators=(",", ":")))
    print(f"gate_example/  {len(fams)} files")
    dump("scaling", dict(
        regimes=rows(pd.read_csv(paths.SCALING_PREDICTABILITY / ALL / "scaling_regimes.csv")),
        families=rows(pd.read_csv(paths.SCALING_PREDICTABILITY / ALL / "rq1_families.csv"))))

    # rq02 names every table by its pair set (rule 15): the site reads the multi-axis pairs
    da = pd.read_csv(paths.DECISION_ACCURACY / PRED / "da_all_per_benchmark_multi_axes.csv")
    dump("decision_accuracy", dict(
        early_small=rows(pd.read_csv(paths.DECISION_ACCURACY / PRED / "early_small_da_goal_summary_multi_axes.csv")),
        per_task=table(da[["language", "benchmark", "task", "da_def", "comparison", "decision_acc"]])))

    dump("snr", dict(
        per_task=table(pd.read_csv(paths.NOISE_AND_SNR / PRED / "snr.csv")),
        effect_over_seed=table(pd.read_csv(paths.NOISE_AND_SNR / ALL / "effect_over_seed.csv"))))

    dump("surrogates", dict(
        rho=rows(pd.read_csv(paths.SURROGATES / PRED / "rq3_surrogates.csv")),
        best_variant=rows(pd.read_csv(paths.SURROGATES / PRED / "best_variant_per_language.csv"))))

    dump("design_decisions", dict(
        # one design decision at a time: rq05's tables are the mono-axis pairs
        da=rows(pd.read_csv(paths.DESIGN_DECISIONS / ALL / "rq4_da_size_by_intervention_mono_axis.csv")),
        effect=rows(pd.read_csv(paths.DESIGN_DECISIONS / ALL / "rq4_effect_vs_seed.csv")),
        early=rows(pd.read_csv(paths.DESIGN_DECISIONS / ALL / "rq2_da_goal_early_small_mono_axis.csv"))))

    dump("transfer", dict(
        summary=rows(pd.read_csv(paths.LANGUAGE_TRANSFER / ALL / "rq5_transfer_summary.csv")),
        per_language=table(pd.read_csv(paths.LANGUAGE_TRANSFER / ALL / "rq5_transfer.csv",
                                      usecols=["L", "task", "trained", "reference_size", "k", "observed",
                                               "pred_transfer", "pred_last", "err_transfer", "err_last"]))))

    ext = paths.EXTERNAL_FRAMEWORKS / "all/external"
    dump("external", dict(
        ours=rows(pd.read_csv(ext / "top_apertus.csv")), allenai=rows(pd.read_csv(ext / "top_allenai.csv")),
        agreement=rows(pd.read_csv(ext / "shared_task_agreement.csv")),
        per_variant=rows(pd.read_csv(ext / "pearson_r_per_variant.csv"))))

    dump("subsets", rows(pd.read_csv(paths.SUBSET_SELECTION / "all/external/summary.csv")))
    design = pd.read_csv(paths.BENCHMARK_DESIGN / PRED / "per_family_snr.csv")
    design["data_source"] = design["data_source"].map(no_byline, na_action="ignore")
    dump("benchmark_design", rows(design))
    dump("recipe", recipe())
    for name, build in [("above_chance_items", above_chance_items), ("english_only", english_only)]:
        obj = build()
        if obj is None:
            print(f"{name}: skipped, its analysis folder is not in this checkout")
        else:
            dump(name, obj)


if __name__ == "__main__":
    main()
