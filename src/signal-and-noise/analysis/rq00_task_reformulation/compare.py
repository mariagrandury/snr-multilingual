#!/usr/bin/env python3
"""Original vs reformulated (rf_*) scores, from the ladder report.

The rq00 gate, focused on the three letter-format families and read on both
task sets: per (task, size) the mean final-checkpoint score of the size's
models that trained the task's language (the deep scheme-A seed-1904 ladder,
the only cells with rf results), minus chance (0.25). The scores are the
report's `primary_score` — `acc` for the originals, `acc_norm` for the rf
twins (tasks.json `metric`; the cloze answers differ in length). The
originals' acc equals their acc_norm — one letter per option — so this IS
the like-for-like comparison. Writes, in this directory:

    rf_gate.png / .csv               family x size: original | reformulated | difference
    rf_gate_by_language.png / .csv   one row per family, language x size, same three panels
    README.md                        the `auto:rf-compare` block (family x size table)

    python3.11 src/signal-and-noise/analysis/rq00_task_reformulation/compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
from analysis.rq00_gate_and_curves.above_random import MARGIN, scores_and_mask  # noqa: E402

DOC = HERE / "README.md"
TASKS = json.loads((ROOT / "configs" / "tasks.json").read_text())["tasks"]
FAMILIES = ["belebele", "global_mmlu_full", "include_base_44"]
CHANCE = 0.25
# orange = the reformulation lost score, blue = gained; white = no change
DIFF = LinearSegmentedColormap.from_list("rf_diff", ["#d9730d", "#f7f4ef", "#0d366b"])
DIFF.set_bad(S.NODATA)
mpl.rcParams.update(S.RC)


def cells(by: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The gate's cells, (original, reformulated): per task and size the mean
    final-checkpoint score over the size's models that trained the language,
    minus chance, then the median over the tasks of `by`."""
    df = build_snr_pool("predictivity")
    df = df[(df["arch"] == "deep") & (df["scheme"] == "A")]
    fam = df["task"].map(lambda t: TASKS.get(t, {}).get("benchmark", "").removeprefix("rf_"))
    trained = [t.removeprefix("rf_") in _trained_tasks(L, s)
               for t, L, s in zip(df["task"], df["L"], df["scheme"])]
    df = df[fam.isin(FAMILIES) & pd.Series(trained, index=df.index)]
    if not df["task"].str.startswith("rf_").any():
        # a report without the rf columns (the published one, until the rf evals
        # are pushed) would otherwise overwrite the figures with an empty table
        sys.exit("no rf_* results in this ladder report: point SNR_LADDER_DIR at one that has them")
    scores, _, _ = scores_and_mask(df)
    per_task = (scores - CHANCE).stack().rename("margin").reset_index()
    per_task.columns = ["task", "size", "margin"]
    per_task["family"] = per_task["task"].map(lambda t: TASKS[t]["benchmark"].removeprefix("rf_"))
    per_task["language"] = per_task["task"].map(lambda t: TASKS[t]["language"])
    per_task["set"] = per_task["task"].str.startswith("rf_").map({True: "rf", False: "orig"})
    agg = per_task.groupby(["set"] + by + ["size"]).agg(margin=("margin", "median"), n=("task", "nunique")).reset_index()
    return agg[agg["set"] == "orig"], agg[agg["set"] == "rf"]


def panels(axes, orig: pd.DataFrame, rf: pd.DataFrame, index: str, sizes: list, ylabel: str, prefix: str = "") -> list:
    piv = lambda t: t.pivot(index=index, columns="size", values="margin").reindex(columns=sizes)
    cnt = lambda t: t.pivot(index=index, columns="size", values="n").reindex(columns=sizes)
    a, b = piv(orig), piv(rf)
    a, b = a.reindex(a.index.union(b.index)), b.reindex(a.index.union(b.index))
    kw = dict(vmin=-0.1, vmax=0.3, center=MARGIN, cmap=S.DIV, fmt="{:+.2f}", xlabel="model size")
    return [G.matrix_ax(axes[0], a, f"{prefix}original (letters A–D)", cnt=cnt(orig).reindex(a.index), ylabel=ylabel, **kw),
            G.matrix_ax(axes[1], b, f"{prefix}reformulated (answer strings)", cnt=cnt(rf).reindex(b.index), **kw),
            G.matrix_ax(axes[2], b - a, f"{prefix}difference (reformulated − original)", vmin=-0.1, vmax=0.2,
                        center=0.0, cmap=DIFF, fmt="{:+.2f}", xlabel="model size")]


def main() -> None:
    fam_o, fam_r = cells(["family"])
    sizes = size_order(set(fam_o["size"]) | set(fam_r["size"]))
    note = (f"cell = median over the family's tasks of (mean final-checkpoint score of the size's deep scheme-A seed-1904 models "
            f"that trained the language) − chance 0.25; small number = tasks; the gate keeps a cell above {MARGIN:+.2f}; "
            f"original = acc, reformulated = acc_norm")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    tables = panels(axes, fam_o, fam_r, "family", sizes, "benchmark")
    G.save_highlights(fig, HERE, "The three letter-format families, before and after the reformulation", note, tables, name="rf_gate")

    lang_o, lang_r = cells(["family", "language"])
    fig, axes = plt.subplots(len(FAMILIES), 3, figsize=(13, 2.6 * len(FAMILIES) + 0.8), squeeze=False)
    tables = []
    for row, fam in zip(axes, FAMILIES):
        tables += panels(row, lang_o[lang_o["family"] == fam], lang_r[lang_r["family"] == fam], "language", sizes, fam, prefix=f"{fam}: ")
    G.save_highlights(fig, HERE, "Per language, before and after the reformulation", note, tables, name="rf_gate_by_language")

    # the markdown block: family x size, original / rf / Δ
    lines = [f"Gate cells (median task margin over chance 0.25, trained languages, deep scheme-A seed-1904 ladder, "
             f"from the ladder report). Cell: original acc → rf acc_norm, **Δ** = rf − original; n = rf tasks.", "",
             "| family | " + " | ".join(sizes) + " |", "|---|" + "---:|" * len(sizes)]
    for fam in FAMILIES:
        out = []
        for s in sizes:
            o = fam_o[(fam_o["family"] == fam) & (fam_o["size"] == s)]
            r = fam_r[(fam_r["family"] == fam) & (fam_r["size"] == s)]
            if o.empty or r.empty or o["margin"].isna().all() or r["margin"].isna().all():
                out.append("—"); continue
            om, rm = o["margin"].iloc[0], r["margin"].iloc[0]
            out.append(f"{om:+.3f} → {rm:+.3f}, **{rm - om:+.3f}**, n={int(r['n'].iloc[0])}")
        lines.append(f"| {fam} | " + " | ".join(out) + " |")
    lines += ["", "![family x size](rf_gate.png)", "", "![per language](rf_gate_by_language.png)"]
    replace_block(DOC, "rf-compare", "\n".join(lines), "analysis/rq00_task_reformulation/compare.py")
    print("\n".join(lines[:4 + len(FAMILIES)]))
    print(f"updated {DOC}")


if __name__ == "__main__":
    main()
