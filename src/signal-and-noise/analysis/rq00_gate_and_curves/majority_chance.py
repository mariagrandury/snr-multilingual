"""The gate with a majority-class chance level: a sensitivity analysis.

The canonical gate (rule 1, `above_random.task_chance`) holds a run against
uniform guessing, 1 / n_options. With unbalanced gold labels a constant answer
scores the majority label's share instead (always "yes" on a yes-heavy binary
task, always "A" when the gold sits in position A more often than not), which
can exceed 1 / n. This script asks how much of the gate, and of the headline
numbers read through it, rests on that difference. It never replaces the
canonical gate: its tables live in `majority_chance/` only.

    --derive   for every accuracy task of configs/tasks.json with an option
               count, the share of its evaluated items whose gold option
               index is the most frequent one (the accuracy of always giving
               that index), read from the harness samples of the final
               checkpoint of the ladder cells (1.7B first). A parent without a
               samples file of its own (Global-MMLU, INCLUDE, mmlu) is the union
               of its subject files. Writes `majority_baselines.csv` (task,
               n_items, majority, majority_index, fixed_options, ...).
    --write-tasks-json   adds that table's `majority` to every task of
               configs/tasks.json (atomic, refused if the file changed
               meanwhile: `configs.write_tasks_json`); nothing reads it but
               this script.
    (default)  re-runs the gate of pool `predictivity` with the chance level
               max(uniform, majority) (`SNR_CHANCE=majority`, read by
               `above_random.task_chance`) into `majority_chance/gate/`
               (`SNR_GATE_DIR`), compares it with the canonical mask and
               writes the flips, the headline pass counts under both chance
               levels, the downstream reads and the figure.

`fixed_options` says whether each answer position carries the same label on
every item: a letter answer (" A".." D"), True / False, or the Yes / Also / No
of a partial-scoring prompt (XNLI, PAWS-X), read as a character bigram every
item shares at that position and not at all the others. Only there can a
model reach the majority baseline by a constant preference. Where every item
has its own answers (cloze options such as TruthfulQA's, whose gold the
harness lists first, or MultiBLiMP's grammatical sentence, always option 0),
log-likelihood scoring never sees the position, so the majority baseline of
those tasks is an artefact of the item layout that no model can exploit, and
a flip there is not a finding about the model. The third gate,
`majority_fixed`, takes the majority level on the labelled tasks alone.

    python analysis/rq00_gate_and_curves/majority_chance.py --derive --write-tasks-json
    python analysis/rq00_gate_and_curves/majority_chance.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import (  # noqa: E402
    DEFAULT_TASKS_JSON, accuracy_metric, load_tasks, read_tasks_json, write_tasks_json)
from analysis.paths import GATE_AND_CURVES  # noqa: E402
from analysis.rq08_subset_selection.build_per_item_store import gold_index  # noqa: E402
from pretrain.ladder_report import EVAL_LOGS  # noqa: E402

OUT = GATE_AND_CURVES / "majority_chance"
BASELINES = OUT / "majority_baselines.csv"
_FILE = re.compile(r"^samples_(.+)_(\d{4}-\d{2}-\d{2}T[\d\-.]+)\.jsonl$")
_DIR = re.compile(r"^lm-(\d+(?:\.\d+)?)(M|B)-.*-iter(\d+)$")
_SIZE_ORDER = ["1.7B", "1B", "600M", "350M", "175M", "90M", "3B"]
WORKERS = 16


# --- derive ----------------------------------------------------------------

def final_dirs() -> list[Path]:
    """The final-checkpoint eval dir of every ladder cell, 1.7B cells first (they
    run every task the site shows), then the smaller sizes."""
    last: dict[str, tuple[int, str]] = {}
    for e in os.scandir(EVAL_LOGS):
        m = _DIR.match(e.name)
        if m:
            cell = e.name.rsplit("-iter", 1)[0]
            if int(m[3]) > last.get(cell, (-1, ""))[0]:
                last[cell] = (int(m[3]), e.name)
    size = lambda c: c.split("-")[1]
    rank = lambda c: _SIZE_ORDER.index(size(c)) if size(c) in _SIZE_ORDER else len(_SIZE_ORDER)
    return [EVAL_LOGS / last[c][1] for c in sorted(last, key=lambda c: (rank(c), c))]


def _resolve(name: str, listed: dict, memo: dict) -> str | None:
    """A samples-file task name -> its tasks.json entry: itself, or the longest
    listed prefix (a subject file of a parent), as derive_task_options does."""
    if name not in memo:
        memo[name] = name if name in listed else max(
            (k for k in listed if name.startswith(k + "_")), key=len, default=None)
    return memo[name]


def _gold(r: dict, opts: list[str]) -> int:
    """The gold option's index: the target as an index (a digit, or a letter,
    `gold_index`), else the one continuation equal to the target text (COPA,
    cultural_bench's True / False, HAE-RAE's "(B)"), else, on a two-option task
    whose options differ in the context alone (the Winograd schemas), the
    model's pick when it scored the item right and the other option when it
    did not; -1 when none applies."""
    g = gold_index(r.get("target"))
    if 0 <= g < len(opts):
        return g
    hits = [i for i, o in enumerate(opts) if o.strip() == str(r.get("target")).strip()]
    if len(hits) == 1:
        return hits[0]
    if len(opts) == 2 and r.get("acc") in (0, 1, 0.0, 1.0):
        pick = int(np.argmax([float(x[0][0]) for x in r["resps"]]))
        return pick if r["acc"] == 1 else 1 - pick
    return -1


def _labelled(common: dict[int, set]) -> bool:
    """Every position carries a mark of its own on every item: a character
    bigram all the items share at that position and not at every other one."""
    if len(common) < 2:
        return False
    everywhere = set.intersection(*common.values())
    return all(c - everywhere for c in common.values())


def _read(files: list[Path]) -> dict:
    """Gold-index counts over the items of `files`, and whether each answer
    position means the same on every item (`fixed_options`): the continuations,
    or the whole scored strings, share a mark per position (a letter, Yes /
    No, True / False, the entailment words of a partial-scoring prompt)."""
    golds, bad = Counter(), 0
    common = {"cont": {}, "full": {}}
    for f in files:
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                args = r.get("arguments") or {}
                keys = sorted(args, key=lambda k: int(k.rsplit("_", 1)[-1]))
                opts = [str(args[k].get("arg_1")) for k in keys]
                g = _gold(r, opts)
                if g < 0:
                    bad += 1
                    continue
                golds[g] += 1
                for kind, strings in (("cont", opts), ("full", [str(args[k].get("arg_0")) + o for k, o in zip(keys, opts)])):
                    for i, o in enumerate(strings):
                        grams = {o[j:j + 2] for j in range(len(o) - 1)}
                        common[kind][i] = common[kind][i] & grams if i in common[kind] else grams
    n = sum(golds.values())
    k, top = golds.most_common(1)[0] if golds else (np.nan, 0)
    return {"n_items": n, "majority": top / n if n else np.nan, "majority_index": k,
            "fixed_options": _labelled(common["cont"]) or _labelled(common["full"]), "n_undecoded": bad,
            "gold_counts": json.dumps(dict(sorted(golds.items())))}


def derive() -> None:
    listed = load_tasks()
    want = {t for t, e in listed.items() if e.get("n_options") and accuracy_metric(t)}
    memo: dict[str, str | None] = {}
    rows: dict[str, dict] = {}
    for d in final_dirs():
        latest: dict[str, tuple[str, Path]] = {}
        for f in d.glob("harness/eval_*/samples_*.jsonl"):
            m = _FILE.match(f.name)
            if m and (m[1] not in latest or m[2] > latest[m[1]][0]):
                latest[m[1]] = (m[2], f)
        groups: dict[str, list[Path]] = {}
        for name, (_, f) in latest.items():
            entry = _resolve(name, listed, memo)
            if entry in want and entry not in rows:
                # a parent's own file, when there is one, is the task; else the union of its subjects
                if name == entry:
                    groups[entry] = [f]
                elif not (groups.get(entry) and groups[entry][0].name.startswith(f"samples_{entry}_2")):
                    groups.setdefault(entry, []).append(f)
        with Pool(WORKERS) as pool:
            read = pool.map(_read, [sorted(f) for f in groups.values()])
        for entry, row in zip(groups, read):
            expect = listed[entry].get("n_items")
            if expect and (row["n_items"] + row["n_undecoded"] != expect or row["n_undecoded"] > 0.01 * expect):
                continue          # a partial subject set at this checkpoint (a later dir may hold them all), or golds it cannot read
            rows[entry] = {"task": entry, **row, "n_options": listed[entry]["n_options"],
                           "source": d.name}
        print(f"  {d.name}: {len(rows)}/{len(want)} tasks", flush=True)
        if want <= set(rows):
            break
    missing = sorted(want - set(rows))
    out = pd.DataFrame(list(rows.values())).sort_values("task")
    print(f"majority read for {len(out)} of {len(want)} accuracy tasks; "
          f"{len(missing)} without a complete samples set: {missing[:20]}")
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(BASELINES, index=False, float_format="%.4f")
    print(f"wrote {BASELINES}")


def write_majority() -> None:
    """Add the `majority` key of every task in BASELINES to configs/tasks.json
    (atomic, and refused if another writer changed the file meanwhile)."""
    data, before = read_tasks_json(DEFAULT_TASKS_JSON)
    base = pd.read_csv(BASELINES)
    for t, v in zip(base["task"], base["majority"]):
        data["tasks"][t]["majority"] = round(float(v), 4)
    write_tasks_json(data, before, DEFAULT_TASKS_JSON)
    print(f"wrote the `majority` key of {len(base)} tasks to {DEFAULT_TASKS_JSON}")


# --- the sensitivity analysis ----------------------------------------------

CHANCES = {"uniform": "Uniform (canonical)", "majority": "Max(uniform, majority)",
           "majority_fixed": "Max(uniform, majority), fixed answer labels only"}
MIN_LABEL = 5          # a benchmark is named in panel (a) from this many flipped tasks


def masks() -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """The three gates over the analysis sizes ({chance: task x size mask}),
    the trained population (task x size, True where the cell is read on runs
    that train the language) and the per-task meta. The majority gate is
    rerun into OUT/gate (`above_random.run` with CHANCE_MODE "majority" and
    GATE_DIR there); `majority_fixed` takes it on the tasks whose options are
    the same strings on every item and the canonical mask elsewhere (a task's
    mask depends on its own chance level alone)."""
    from analysis.autodoc import CANONICAL_POOL
    from analysis.utils import ANALYSIS_SIZES
    from analysis.rq00_gate_and_curves import above_random as AR
    canon = AR.GATE_DIR / "pretraining" / CANONICAL_POOL
    gate = OUT / "gate" / "pretraining" / CANONICAL_POOL
    keep = AR.CHANCE_MODE, AR.GATE_DIR
    AR.CHANCE_MODE, AR.GATE_DIR = "majority", OUT / "gate"
    AR.run(CANONICAL_POOL)
    AR.CHANCE_MODE, AR.GATE_DIR = keep
    read = lambda d, f: pd.read_csv(d / f"above_random_{f}.csv", index_col="task")
    uni, maj, share = read(canon, "mask"), read(gate, "mask"), read(canon, "share")
    sizes = [s for s in ANALYSIS_SIZES if s in uni.columns]
    base = pd.read_csv(BASELINES).set_index("task")
    meta = uni[["family", "language", "n_options", "n_items"]].join(
        base[["majority", "majority_index", "fixed_options"]]).assign(
        uniform=uni["random_baseline"], chance=maj["random_baseline"].reindex(uni.index))
    meta = meta[uni[sizes].notna().any(axis=1)]                     # tasks with a chance level
    fixed = meta["fixed_options"].fillna(False).astype(bool)
    m = {"uniform": uni.loc[meta.index, sizes].astype("Int64"),
         "majority": maj[sizes].reindex(meta.index).astype("Int64")}
    m["majority_fixed"] = m["majority"].where(fixed, m["uniform"], axis=0)
    # same scores, same chance level: the two gates must agree there (else the report moved under the rerun)
    same = (meta["chance"] - meta["uniform"]).abs() < 1e-9
    drift = int((m["uniform"][same].fillna(-1) != m["majority"][same].fillna(-1)).to_numpy().sum())
    print(f"cells that differ where the chance level did not move (report drift): {drift}")
    trained = pd.DataFrame({s: share[f"{s}__population"].reindex(meta.index) == "trained" for s in sizes})
    return m, trained, meta.assign(drift_cells=drift)


def pass_counts(m: dict, trained: pd.DataFrame) -> pd.DataFrame:
    """Per chance level, population (the trained cells, rq00's headline, or
    every chance-level task) and size: cells with a verdict and above chance,
    and the tasks never above chance at any size."""
    rows = []
    for chance, mask in m.items():
        for pop, mk in (("trained", mask.where(trained)), ("all", mask)):
            have = mk.notna().any(axis=1)
            for s in mk.columns:
                rows.append({"chance": chance, "population": pop, "size": s,
                             "n_tasks": int(mk[s].notna().sum()), "n_pass": int((mk[s] == 1).sum())})
            rows.append({"chance": chance, "population": pop, "size": "never",
                         "n_tasks": int(have.sum()), "n_pass": int(((mk[have] == 1).sum(axis=1) == 0).sum())})
    return pd.DataFrame(rows).assign(share=lambda d: d["n_pass"] / d["n_tasks"])


def flips(m: dict, trained: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """The trained (task, size) cells above chance under uniform guessing and at
    chance against the majority baseline."""
    u, j = m["uniform"].where(trained), m["majority"].where(trained)
    f = ((u == 1) & (j == 0)).fillna(False)
    out = f.stack().rename("flip").reset_index().rename(columns={"level_1": "size"})
    out = out[out["flip"]].drop(columns="flip").merge(meta.reset_index(), on="task")
    return out[["task", "family", "language", "size", "n_options", "n_items", "uniform", "majority",
                "majority_index", "fixed_options"]]


def by_benchmark(m: dict, trained: pd.DataFrame, meta: pd.DataFrame, fl: pd.DataFrame) -> pd.DataFrame:
    """Per benchmark family, over its trained chance-level tasks: the chance
    levels, the trained cells above chance under each gate and the flips."""
    keep = trained.any(axis=1)
    t = meta[keep]
    cells = lambda c: (m[c].where(trained) == 1).sum(axis=1)[keep]
    out = t.assign(excess=t["majority"] - t["uniform"], cells_uniform=cells("uniform"),
                   cells_majority=cells("majority"), cells_majority_fixed=cells("majority_fixed")).groupby("family").agg(
        n_tasks=("uniform", "size"), uniform=("uniform", "median"), majority=("majority", "median"),
        max_excess=("excess", "max"), fixed_options=("fixed_options", "mean"),
        cells_uniform=("cells_uniform", "sum"), cells_majority=("cells_majority", "sum"),
        cells_majority_fixed=("cells_majority_fixed", "sum"))
    out["flipped_cells"] = out["cells_uniform"] - out["cells_majority"]
    out["flipped_tasks"] = fl.groupby("family")["task"].nunique().reindex(out.index).fillna(0).astype(int)
    return out.sort_values(["flipped_cells", "max_excess"], ascending=False).reset_index()


def downstream(m: dict, passes: pd.DataFrame) -> pd.DataFrame:
    """The headline numbers under each gate, each from the script that writes
    it: rq00's trained pass shares (above), rq01's fits and regimes
    (`analyze.fit_table`, `regimes.regime_table`, pool predictivity_seeds, the
    deep data-A seed-1904 grid) and rq02's DA-size on accuracy (the per-task
    table of `bench_bpb_da.py`, re-gated with `utils.passes_gate` at the proxy
    and at the reference, multi-axis pairs)."""
    from analysis.paths import DECISION_ACCURACY
    from analysis.rq01_scaling_predictability.analyze import CANONICAL as RQ01_POOL, fit_table
    from analysis.rq01_scaling_predictability.regimes import QUADRANTS, regime_table, size_medians
    from analysis.utils import SMALL_SIZES, TARGET_SIZE, finals, ladder_frame, passes_gate
    rows = []
    tr = passes[passes["population"] == "trained"].set_index(["chance", "size"])
    for c in m:
        for s in ("90M", TARGET_SIZE):
            rows.append({"rq": "rq00", "quantity": f"trained tasks above chance at {s}", "chance": c,
                         "value": tr.at[(c, s), "share"], "n": tr.at[(c, s), "n_tasks"]})
        rows.append({"rq": "rq00", "quantity": "trained tasks never above chance", "chance": c,
                     "value": tr.at[(c, "never"), "share"], "n": tr.at[(c, "never"), "n_tasks"]})
    df = ladder_frame(RQ01_POOL)
    fin = finals(df)
    both = QUADRANTS[(True, True)][0]
    for c, mask in m.items():
        fits, _ = fit_table(fin, RQ01_POOL, mask=mask)
        fits = fits[~fits["family"].str.startswith("bbpb")]          # the gate never touches a bBPB twin
        sm = size_medians(fits[fits["kind"] == "benchmark"])
        rows.append({"rq": "rq01", "quantity": "median size-fit R2 of the accuracy tasks", "chance": c,
                     "value": sm["r2_size"].median(), "n": len(sm)})
        t = regime_table(fits, df, RQ01_POOL, mask=mask)
        rows.append({"rq": "rq01", "quantity": "accuracy and BPB tasks predictable across both", "chance": c,
                     "value": (t["regime"] == both).mean(), "n": len(t)})
    da = pd.read_csv(DECISION_ACCURACY / "pretraining" / "predictivity" / "bench_bpb_da_size_multi_axes.csv")
    tasks = sorted(set(da["task"]))
    for c, mask in m.items():
        gate = {s: passes_gate(mask, tasks, s) for s in [*SMALL_SIZES, TARGET_SIZE]}
        ok = [bool(gate[s][t] and gate[TARGET_SIZE][t]) for t, s in zip(da["task"], da["size"])]
        acc_acc = da["da_acc"].where(ok)
        for s, v in acc_acc.groupby(da["size"]):
            rows.append({"rq": "rq02", "quantity": f"DA-size on accuracy, {s} to {TARGET_SIZE}", "chance": c,
                         "value": v.mean(), "n": int(v.notna().sum())})
    return pd.DataFrame(rows)


def figure(meta: pd.DataFrame, trained: pd.DataFrame, fl: pd.DataFrame, passes: pd.DataFrame, paper: bool) -> None:
    """(a) per trained chance-level task its uniform chance against its majority
    baseline, the tasks whose verdict flips at some size highlighted and their
    benchmarks labelled; (b) the share of trained tasks above chance per size
    under each gate. A CSV of the same name holds both panels."""
    import matplotlib.pyplot as plt
    from analysis import grids as G
    from analysis import style as S
    t = meta[trained.any(axis=1)].copy()
    t["flipped_sizes"] = fl.groupby("task")["size"].agg(",".join).reindex(t.index)
    hit = t["flipped_sizes"].notna()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    for fixed, marker in ((True, "o"), (False, "^")):
        for flip, colour, z in ((False, S.MUTED, 1), (True, S.SERIES[1], 3)):
            g = t[(t["fixed_options"].fillna(False) == fixed) & (hit == flip)]
            a.scatter(g["uniform"], g["majority"], s=14 if flip else 8, marker=marker, color=colour, alpha=.8 if flip else .35,
                      lw=0, zorder=z, label=("Flips" if flip else "Unchanged") + (", fixed answer labels" if fixed
                                                                                   else ", item specific answers"))
    lim = (0, 1.03)
    a.plot(lim, lim, color=S.INK, lw=.6, ls=":")
    a.set(xlim=(0, .65), ylim=lim, xlabel="Uniform chance level", ylabel="Majority baseline")
    lab = t[hit].groupby("family")[["uniform", "majority"]].median()
    lab["n"] = t[hit].groupby("family").size()
    texts = [a.text(r.uniform, r.majority, f"{G.paper_name(f)} ({int(r.n)})", fontsize=6.5, color=S.INK)
             for f, r in lab[lab["n"] >= MIN_LABEL].iterrows()]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=a, arrowprops=dict(arrowstyle="-", color=S.MUTED, lw=.4))
    except ImportError:
        pass
    a.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=1)
    tr = passes[(passes["population"] == "trained") & (passes["size"] != "never")]
    sizes = list(dict.fromkeys(tr["size"]))
    for (c, name), colour, ls in zip(CHANCES.items(), (S.INK, S.SERIES[1], S.SERIES[0]), ("-", "-", "--")):
        g = tr[tr["chance"] == c].set_index("size").reindex(sizes)
        b.plot(range(len(sizes)), g["share"], marker="o", ms=3.5, color=colour, ls=ls, lw=1.4, label=name)
    b.set_xticks(range(len(sizes)), sizes)
    b.set(ylim=(0, 1), xlabel="Model size", ylabel="Share of trained tasks above chance")
    b.legend(fontsize=6.5, frameon=False, loc="lower right")
    for ax in (a, b):
        S.clean(ax)
    name = "majority_chance_paper" if paper else "majority_chance"
    tab = pd.concat([t.reset_index().assign(panel="a")[["panel", "task", "family", "language", "n_options", "n_items",
                                                          "uniform", "majority", "majority_index", "fixed_options",
                                                          "flipped_sizes"]],
                     tr.assign(panel="b")[["panel", "chance", "size", "n_tasks", "n_pass", "share"]]])
    tab.to_csv(OUT / f"{name}.csv", index=False)
    if paper:
        fig.tight_layout()
        S.save_paper(fig, OUT / name)
        return
    top = G._header(fig, "Majority-class chance: which tasks flip, and what the gate keeps",
                    "(a) one point per trained task with a chance level: uniform chance (1/n options, or E[1/n_i]) against the "
                    "accuracy of always giving the most frequent gold option (majority_baselines.csv); orange = above chance at "
                    "some size under the uniform gate and at chance there against max(uniform, majority); circle = every answer "
                    f"position carries a fixed label (a letter, Yes / No), triangle = item specific answers; label = benchmark (tasks "
                    f"flipping, from {MIN_LABEL}). (b) share of the trained tasks above chance per size under the canonical gate, the majority gate, "
                    "and the majority gate applied only where the answer positions carry fixed labels (where a constant answer is "
                    "one a model can give). Pool predictivity, seed 1904, cells read on the runs that train the language.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save_figure(fig, OUT, name)


def analyze() -> None:
    m, trained, meta = masks()
    passes = pass_counts(m, trained)
    fl = flips(m, trained, meta)
    fam = by_benchmark(m, trained, meta, fl)
    passes.to_csv(OUT / "majority_pass_counts.csv", index=False)
    fl.to_csv(OUT / "majority_flips.csv", index=False)
    fam.to_csv(OUT / "majority_by_benchmark.csv", index=False)
    print(passes[passes["population"] == "trained"].pivot(index="size", columns="chance", values="n_pass").to_string())
    print(fam.head(15).to_string())
    for paper in (False, True):
        figure(meta, trained, fl, passes, paper)
    ds = downstream(m, passes)
    ds.to_csv(OUT / "majority_downstream.csv", index=False)
    print(ds.pivot_table(index=["rq", "quantity"], columns="chance", values="value", sort=False).round(3).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--derive", action="store_true", help="read the majority baselines from the samples")
    p.add_argument("--write-tasks-json", action="store_true",
                   help="add the derived `majority` key to configs/tasks.json")
    args = p.parse_args()
    if args.derive:
        derive()
    if args.write_tasks_json:
        write_majority()
    if not (args.derive or args.write_tasks_json):
        analyze()


if __name__ == "__main__":
    main()
