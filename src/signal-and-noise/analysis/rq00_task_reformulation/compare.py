#!/usr/bin/env python3
"""Original vs reformulated (rf_*) scores, from the ladder report.

The rq00 gate, focused on the three letter-format families and read on both
task sets: per (task, size) the mean final-checkpoint score of the size's
models that trained the task's language (the deep scheme-A seed-1904 ladder,
the only cells with rf results), minus chance (0.25). Both sets are read on
the same models: a (model, task) enters only when the original and its rf
twin are both scored, so the difference is not one of model sets while the
rf evals are still landing. The scores are the
report's `primary_score` — `acc` for the originals, `acc_norm` for the rf
twins (tasks.json `metric`; the cloze answers differ in length). The
originals' acc equals their acc_norm — one letter per option — so this IS
the like-for-like comparison. Writes, in this directory:

    rf_gate.png / .csv               family x size: original | reformulated | difference
    rf_gate_by_language.png / .csv   one row per family, language x size, same three panels
    rf_significance.csv              per (task, size, model): the two accuracies, their item counts and the p-value of
                                     the two-proportion z-test (`statsmodels.stats.proportion.proportions_ztest`)

The difference panel says whether a gain is more than noise: per model, the
original and the reformulated accuracy are two binomial proportions over the
task's items (`n_items` in tasks.json; the rf twin drops the items with a
missing option, so the counts differ), and their difference is significant
at P_SIG under the two-proportion z-test. A task's gain is significant at a
size when it is for at least half of the size's models; the small number on
the difference panel is how many of the family's tasks (per family) or of
the models (per language) that is. README.md gets the `auto:rf-compare`
block (the family x size table).

    python3.11 src/signal-and-noise/analysis/rq00_task_reformulation/compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "signal-and-noise"))
from pretrain.ladder_report import _trained_tasks  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import replace_block  # noqa: E402
from analysis.utils import build_snr_pool, size_order  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import scores_and_mask, task_n_items  # noqa: E402
from analysis.utils import finals  # noqa: E402
from statsmodels.stats.proportion import proportions_ztest  # noqa: E402

DOC = HERE / "README.md"
TASKS = json.loads((ROOT / "configs" / "tasks.json").read_text())["tasks"]
FAMILIES = ["belebele", "global_mmlu_full", "include_base_44"]
CHANCE = 0.25
P_SIG = 0.05      # two-proportion z-test level for "the reformulation moved the score"
# orange = the reformulation lost score, blue = gained; white = no change
DIFF = LinearSegmentedColormap.from_list("rf_diff", ["#d9730d", "#f7f4ef", "#0d366b"])
DIFF.set_bad(S.NODATA)
mpl.rcParams.update(S.RC)


def pool() -> pd.DataFrame:
    """The deep scheme-A seed-1904 rows of the three families, on the (model,
    task) pairs with both the original and the rf twin scored."""
    df = build_snr_pool("predictivity")
    df = df[(df["arch"] == "deep") & (df["scheme"] == "A")]
    fam = df["task"].map(lambda t: TASKS.get(t, {}).get("benchmark", "").removeprefix("rf_"))
    trained = [t.removeprefix("rf_") in _trained_tasks(L, s)
               for t, L, s in zip(df["task"], df["L"], df["scheme"])]
    df = df[fam.isin(FAMILIES) & pd.Series(trained, index=df.index)]
    rf = df["task"].str.startswith("rf_")
    df = df[rf.groupby([df["model"], df["task"].str.removeprefix("rf_")]).transform("nunique") == 2]
    if df.empty:
        # a report without the rf columns (the published one, until the rf evals
        # are pushed) would otherwise overwrite the figures with an empty table
        sys.exit("no rf_* results in this ladder report: point SNR_LADDER_DIR at one that has them")
    return df


def significance(df: pd.DataFrame) -> pd.DataFrame:
    """Per (task, size, model): original vs rf accuracy at the final checkpoint
    and the two-proportion z-test p-value; `sig` when p < P_SIG."""
    f = finals(df)[["model", "size", "task", "primary_score"]].copy()
    f["base"] = f["task"].str.removeprefix("rf_")
    f["n"] = f["task"].map(task_n_items)
    f["set"] = np.where(f["task"].str.startswith("rf_"), "rf", "orig")
    w = f.pivot(index=["model", "size", "base"], columns="set", values=["primary_score", "n"]).dropna()
    k = np.rint(w["primary_score"].to_numpy() * w["n"].to_numpy())
    p = np.array([proportions_ztest([ko, kr], [no, nr])[1] for (ko, kr), (no, nr) in zip(k, w["n"].to_numpy())])
    out = pd.DataFrame({"acc_orig": w["primary_score"]["orig"], "acc_rf": w["primary_score"]["rf"],
                        "n_orig": w["n"]["orig"].astype(int), "n_rf": w["n"]["rf"].astype(int), "p": p}, index=w.index).reset_index()
    out["sig"] = out["p"] < P_SIG
    return out.rename(columns={"base": "task"})


def cells(df: pd.DataFrame, sig: pd.DataFrame, by: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The gate's cells, (original, reformulated, significance): per task and
    size the mean final-checkpoint score over the size's models that trained
    the language, minus chance, then the median over the tasks of `by`; and
    per unit of `by` and size how many tasks (or, for one task, models) have
    a significant difference."""
    scores, _, _ = scores_and_mask(df)
    per_task = (scores - CHANCE).stack().dropna().rename("margin").reset_index()   # stack() keeps NaN cells since pandas 2.1
    per_task.columns = ["task", "size", "margin"]
    per_task["family"] = per_task["task"].map(lambda t: TASKS[t]["benchmark"].removeprefix("rf_"))
    per_task["language"] = per_task["task"].map(lambda t: TASKS[t]["language"])
    per_task["set"] = per_task["task"].str.startswith("rf_").map({True: "rf", False: "orig"})
    agg = per_task.groupby(["set"] + by + ["size"]).agg(margin=("margin", "median"), n=("task", "nunique")).reset_index()
    sig = sig.assign(family=sig["task"].map(lambda t: TASKS[t]["benchmark"]), language=sig["task"].map(lambda t: TASKS[t]["language"]))
    n_sig = (sig.groupby(["task", "size"] + by)["sig"].mean().ge(0.5).groupby(by + ["size"]).sum()   # tasks: ≥ half of the models
             if by == ["family"] else sig.groupby(by + ["size"])["sig"].sum()).rename("n_sig").reset_index()
    return agg[agg["set"] == "orig"], agg[agg["set"] == "rf"], n_sig


def panels(axes, orig: pd.DataFrame, rf: pd.DataFrame, sig: pd.DataFrame, index: str, sizes: list, ylabel: str,
           prefix: str = "") -> list:
    piv = lambda t, v="margin": t.pivot(index=index, columns="size", values=v).reindex(columns=sizes)
    a, b = piv(orig), piv(rf)
    a, b = a.reindex(a.index.union(b.index)), b.reindex(a.index.union(b.index))
    kw = dict(vmin=-0.1, vmax=0.3, center=0.0, cmap=S.DIV, fmt="{:+.2f}", xlabel="model size")
    return [G.matrix_ax(axes[0], a, f"{prefix}original (letters A–D)", cnt=piv(orig, "n").reindex(a.index), ylabel=ylabel, **kw),
            G.matrix_ax(axes[1], b, f"{prefix}reformulated (answer strings)", cnt=piv(rf, "n").reindex(b.index), **kw),
            G.matrix_ax(axes[2], b - a, f"{prefix}difference (reformulated − original)", vmin=-0.1, vmax=0.2,
                        cnt=piv(sig, "n_sig").reindex(a.index), center=0.0, cmap=DIFF, fmt="{:+.2f}", xlabel="model size")]


def main() -> None:
    df = pool()
    sig = significance(df)
    sig.to_csv(HERE / "rf_significance.csv", index=False)
    fam_o, fam_r, fam_sig = cells(df, sig, ["family"])
    sizes = size_order(set(fam_o["size"]) | set(fam_r["size"]))
    note = (f"cell = median over the family's tasks of (mean final-checkpoint score of the size's deep scheme-A seed-1904 models "
            f"that trained the language and have both task sets scored) − chance 0.25; small number = tasks, and on the "
            f"difference panel the tasks whose gain is significant (two-proportion z-test over the task's items, p < {P_SIG}, "
            f"for at least half of the size's models); original = acc, reformulated = acc_norm")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    tables = panels(axes, fam_o, fam_r, fam_sig, "family", sizes, "benchmark")
    G.save_highlights(fig, HERE, "The three letter-format families, before and after the reformulation", note, tables, name="rf_gate")

    lang_o, lang_r, lang_sig = cells(df, sig, ["family", "language"])
    fig, axes = plt.subplots(len(FAMILIES), 3, figsize=(13, 2.6 * len(FAMILIES) + 0.8), squeeze=False)
    tables = []
    for row, fam in zip(axes, FAMILIES):
        tables += panels(row, lang_o[lang_o["family"] == fam], lang_r[lang_r["family"] == fam], lang_sig[lang_sig["family"] == fam],
                         "language", sizes, fam, prefix=f"{fam}: ")
    G.save_highlights(fig, HERE, "Per language, before and after the reformulation",
                      note.replace("the tasks whose gain is significant", "the models whose gain is significant"), tables,
                      name="rf_gate_by_language")

    # the markdown block: family x size, original / rf / Δ
    lines = [f"Gate cells (median task margin over chance 0.25, trained languages, deep scheme-A seed-1904 ladder, "
             f"from the ladder report; both sets on the same models, those with the original and the rf twin scored). "
             f"Cell: original acc → rf acc_norm, **Δ** = rf − original; n = rf tasks, sig = tasks whose gain is significant "
             f"(two-proportion z-test, p < {P_SIG}, for at least half of the size's models).", "",
             "| family | " + " | ".join(sizes) + " |", "|---|" + "---:|" * len(sizes)]
    for fam in FAMILIES:
        out = []
        for s in sizes:
            o = fam_o[(fam_o["family"] == fam) & (fam_o["size"] == s)]
            r = fam_r[(fam_r["family"] == fam) & (fam_r["size"] == s)]
            if o.empty or r.empty or o["margin"].isna().all() or r["margin"].isna().all():
                out.append("—"); continue
            om, rm = o["margin"].iloc[0], r["margin"].iloc[0]
            k = fam_sig[(fam_sig["family"] == fam) & (fam_sig["size"] == s)]["n_sig"]
            out.append(f"{om:+.3f} → {rm:+.3f}, **{rm - om:+.3f}**, n={int(r['n'].iloc[0])}, sig={int(k.iloc[0]) if len(k) else 0}")
        lines.append(f"| {fam} | " + " | ".join(out) + " |")
    lines += ["", "![family x size](rf_gate.png)", "", "![per language](rf_gate_by_language.png)"]
    replace_block(DOC, "rf-compare", "\n".join(lines), "analysis/rq00_task_reformulation/compare.py")
    print("\n".join(lines[:4 + len(FAMILIES)]))
    print(f"updated {DOC}")


if __name__ == "__main__":
    main()
