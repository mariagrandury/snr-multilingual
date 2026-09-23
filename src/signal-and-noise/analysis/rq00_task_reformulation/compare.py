#!/usr/bin/env python3
"""Original vs reformulated scores, from the ladder report — the `rf_*` cloze
twins and the `rfgm_*` Gemini-rewritten twins (SETS) side by side.

The rq00 gate, focused on the three letter-format families and read on every
task set: per (task, size) the mean final-checkpoint score of the size's
models that trained the task's language (the deep scheme-A seed-1904 ladder,
the only cells with reformulated results), minus chance (0.25). Each set is
paired with the originals on the same models: a (model, task) enters a set's
comparison only when the original and that twin are both scored, so the
difference is not one of model sets while a set's evals are still landing
(the "original" panel is the first set's pairing; each difference panel uses
its own). The scores are the report's `primary_score` — `acc` for the
originals, `acc_norm` for the twins (tasks.json `metric`; the cloze answers
differ in length). Writes, in this directory:

    rf_gate.png / .csv               family x size: original | rf | rfgm | rf − original | rfgm − original
    rf_gate_by_language.png / .csv   one row per family, language x size, the same five panels
    rf_significance.csv              per (set, task, size, model): both accuracies, the twin run's own `acc`,
                                     the normalisation offset, the item counts and the z-test p-value

The difference panel says whether a gain is more than noise, and the test is
run on ONE metric. The twins are scored with `acc_norm` and the originals
with `acc`, so part of every plotted difference is the metric, not the
reformulation. Measured on the same run and the same items (`norm_offset`,
1 393 rf pairs): the median is +0.005 but it is family-shaped — belebele
−0.016, Global-MMLU +0.016, INCLUDE +0.018 on average, spanning −0.058 to
+0.084 — and it is more than half the size of the plotted gain in 36 % of
the pairs. The significance test therefore compares the ORIGINAL's acc with
the twin run's own `acc` (read from the harness results files, which carry
both metrics) — same metric, only the formulation differs — as two binomial
proportions over the task's items (`n_items` in tasks.json; the twins drop
the items with a missing option or a rejected rewrite, so the counts differ)
under the two-proportion z-test. The panels keep acc_norm for the twins,
because that is the metric a cloze task is scored with, and `norm_offset`
in the CSV is how much of the cell is that choice. A task's gain is
significant at a size when p < P_SIG for at least half of the size's models
(per family); per language the small number counts the significant (task,
model) pairs instead. README.md gets the `auto:rf-compare` block (the
family x size table).

    python3.11 src/signal-and-noise/analysis/rq00_task_reformulation/compare.py
"""
from __future__ import annotations

import argparse
import json
import re
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
from analysis.rq00_gate_and_curves.above_random import scores_and_mask, task_n_items, task_n_options  # noqa: E402
from pretrain.auto_evals_cscs import DEFAULT_LOGS_ROOT  # noqa: E402
from analysis.utils import finals  # noqa: E402
from statsmodels.stats.proportion import proportions_ztest  # noqa: E402

DOC = HERE / "README.md"
TASKS = json.loads((ROOT / "configs" / "tasks.json").read_text())["tasks"]
# the default set; `--families` names another (the probe pairs), `--tag` keeps
# its outputs and README block apart from these
FAMILIES = ["belebele", "global_mmlu_full", "include_base_44"]
# the reformulated sets, prefix -> panel title (make_rf_tasks.py --set)
SETS = {"rf": "reformulated (answer strings)", "rfgm": "rewritten (Gemini statements)"}
PREFIX = re.compile(r"^(rfgm|rf)_")
P_SIG = 0.05      # two-proportion z-test level for "the reformulation moved the score"
# The harness writes both metrics; the report keeps only each task's primary
# one (tasks.json `metric`), so the twins' plain `acc` is read from here.
RESULTS = Path(DEFAULT_LOGS_ROOT) / "mariagrandury-epflnlp" / "msnr"
# orange = the reformulation lost score, blue = gained; white = no change
DIFF = LinearSegmentedColormap.from_list("rf_diff", ["#d9730d", "#f7f4ef", "#0d366b"])
DIFF.set_bad(S.NODATA)
mpl.rcParams.update(S.RC)


def base(task: str) -> str:
    return PREFIX.sub("", task)


def set_of(task: str) -> str:
    m = PREFIX.match(task)
    return m.group(1) if m else "orig"


def pool(families: list[str]) -> pd.DataFrame:
    """The deep scheme-A seed-1904 rows of `families`, originals and every
    twin, trained languages only; `set` and `base` columns."""
    # untrained=True because utils' trained_only() resolves a task against
    # auto_benchmarks(), which knows only the originals — every rf_*/rfgm_*
    # row is "untrained" there and the pool would come back without a single
    # twin. The rule is applied below instead, on the base task name.
    df = build_snr_pool("predictivity", untrained=True)
    df = df[(df["arch"] == "deep") & (df["scheme"] == "A")]
    fam = df["task"].map(lambda t: base(TASKS.get(t, {}).get("benchmark", "")))
    trained = [base(t) in _trained_tasks(L, s) for t, L, s in zip(df["task"], df["L"], df["scheme"])]
    df = df[fam.isin(families) & pd.Series(trained, index=df.index)]
    return df.assign(set=df["task"].map(set_of), base=df["task"].map(base))


def paired(df: pd.DataFrame, s: str) -> pd.DataFrame:
    """The originals and the twins of set `s` on the (model, base) pairs that
    have both scored."""
    d = df[df["set"].isin(["orig", s])]
    return d[d.groupby(["model", "base"])["set"].transform("nunique") == 2]


def twin_plain_acc(fin: pd.DataFrame, s: str) -> pd.Series:
    """The twin runs' own `acc` (not acc_norm), per (model, task), from the
    harness results files — the like-for-like control for the originals' acc."""
    out = {}
    for (model, step), g in fin[fin["set"] == s].groupby(["model", "step"]):
        want = set(g["task"])
        for f in (RESULTS / f"{model}-iter{step}").glob(f"harness/eval_*/per_task/{s}_*/*/results_*.json"):
            try:
                res = json.loads(f.read_text()).get("results", {})
            except (OSError, json.JSONDecodeError):
                continue
            for task, r in res.items():
                if task in want and "acc,none" in r:
                    out[(model, task)] = r["acc,none"]
    return pd.Series(out, name="acc_twin_plain", dtype=float)


SIG_COLS = ["set", "task", "size", "model", "acc_orig", "acc_twin", "acc_twin_plain", "norm_offset",
            "n_orig", "n_twin", "p", "sig"]


def significance(df: pd.DataFrame, s: str) -> pd.DataFrame:
    """Per (task, size, model): original vs twin accuracy at the final
    checkpoint and the two-proportion z-test p-value; `sig` when p < P_SIG."""
    if df.empty:
        return pd.DataFrame(columns=SIG_COLS)
    f = finals(df)[["model", "size", "step", "task", "primary_score", "set", "base"]].copy()
    f["n"] = f["task"].map(task_n_items)
    plain = twin_plain_acc(f, s)
    f["set"] = f["set"].replace({s: "twin"})
    w = f.pivot(index=["model", "size", "base"], columns="set", values=["primary_score", "n"]).dropna()
    # the twin side on the originals' metric; where the results file is gone the
    # pair drops out of the test rather than being compared across metrics
    acc_plain = pd.Series([plain.get((m, f"{s}_{t}"), np.nan) for m, _, t in w.index], index=w.index)
    k_orig = np.rint(w[("primary_score", "orig")].to_numpy() * w[("n", "orig")].to_numpy())
    k_twin = np.rint(acc_plain.to_numpy() * w[("n", "twin")].to_numpy())
    n_orig, n_twin = w[("n", "orig")].to_numpy(), w[("n", "twin")].to_numpy()
    p = np.array([proportions_ztest([ko, kt], [no, nt])[1] if np.isfinite(kt) else np.nan
                  for ko, kt, no, nt in zip(k_orig, k_twin, n_orig, n_twin)])
    out = pd.DataFrame({"acc_orig": w["primary_score"]["orig"], "acc_twin": w["primary_score"]["twin"],
                        "acc_twin_plain": acc_plain, "norm_offset": w["primary_score"]["twin"] - acc_plain,
                        "n_orig": w["n"]["orig"].astype(int), "n_twin": w["n"]["twin"].astype(int), "p": p},
                       index=w.index).reset_index()
    out["sig"] = out["p"] < P_SIG
    missing = out["acc_twin_plain"].isna().sum()
    if missing:
        print(f"({s}: {missing} of {len(out)} pairs have no twin results file: not tested)")
    return out.rename(columns={"base": "task"}).assign(set=s)[SIG_COLS]


def cells(df: pd.DataFrame, sig: pd.DataFrame, s: str, by: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The gate's cells of one pairing, (original, twin, significance): per
    task and size the mean final-checkpoint score over the size's models
    that trained the language, minus chance, then the median over the tasks
    of `by`; and per unit of `by` and size how many tasks (or, for one task,
    models) have a significant difference."""
    empty = pd.DataFrame(columns=by + ["size", "margin", "n"])
    if df.empty:
        return empty, empty, pd.DataFrame(columns=by + ["size", "n_sig"])
    scores, _, _ = scores_and_mask(df)
    chance = pd.Series({t: 1 / task_n_options(t) for t in scores.index})   # per task: the probe pairs are 2- to 10-way
    per_task = scores.sub(chance, axis=0).stack().dropna().rename("margin").reset_index()   # stack() keeps NaN cells since pandas 2.1
    per_task.columns = ["task", "size", "margin"]
    per_task["family"] = per_task["task"].map(lambda t: base(TASKS[t]["benchmark"]))
    per_task["language"] = per_task["task"].map(lambda t: TASKS[t]["language"])
    per_task["set"] = per_task["task"].map(set_of)
    agg = per_task.groupby(["set"] + by + ["size"]).agg(margin=("margin", "median"), n=("task", "nunique")).reset_index()
    sig = sig.assign(family=sig["task"].map(lambda t: TASKS[t]["benchmark"]), language=sig["task"].map(lambda t: TASKS[t]["language"]))
    n_sig = (sig.groupby(["task", "size"] + by)["sig"].mean().ge(0.5).groupby(by + ["size"]).sum()   # tasks: ≥ half of the models
             if by == ["family"] else sig.groupby(by + ["size"])["sig"].sum()).rename("n_sig").reset_index()
    return agg[agg["set"] == "orig"], agg[agg["set"] == s], n_sig


def panels(axes, orig: pd.DataFrame, twins: dict, index: str, sizes: list, ylabel: str, prefix: str = "") -> list:
    """original | one panel per set | one difference panel per set; `twins` is
    set -> (original cells of that pairing, twin cells, significance)."""
    piv = lambda t, v="margin": (t.pivot(index=index, columns="size", values=v) if len(t)
                                 else pd.DataFrame()).reindex(columns=sizes)
    rows = piv(orig).index.union(pd.Index([i for _, t, _ in twins.values() for i in piv(t).index]))
    a = piv(orig).reindex(rows)
    kw = dict(vmin=-0.1, vmax=0.3, center=0.0, cmap=S.DIV, fmt="{:+.2f}", xlabel="model size")
    out = [G.matrix_ax(axes[0], a, f"{prefix}original (letters A–D)", cnt=piv(orig, "n").reindex(rows), ylabel=ylabel, **kw)]
    for i, (s, (o, t, sig)) in enumerate(twins.items()):
        b = piv(t).reindex(rows)
        out.append(G.matrix_ax(axes[1 + i], b, f"{prefix}{s}: {SETS[s]}", cnt=piv(t, "n").reindex(rows), **kw))
        out.append(G.matrix_ax(axes[1 + len(twins) + i], b - piv(o).reindex(rows), f"{prefix}{s} − original",
                               vmin=-0.1, vmax=0.2, cnt=piv(sig, "n_sig").reindex(rows), center=0.0, cmap=DIFF,
                               fmt="{:+.2f}", xlabel="model size"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help="comma-separated benchmark families to pair with their twins (default: the three letter-format families)")
    ap.add_argument("--tag", default="",
                    help="suffix for every output file and the README block, so another family set (the probe) does not overwrite the default one's")
    args = ap.parse_args()
    families = [f for f in args.families.split(",") if f]
    sfx = f"_{args.tag}" if args.tag else ""
    df = pool(families)
    have = [s for s in SETS if (df["set"] == s).any()]
    if not have:
        # a report without the twin columns (the published one, until their evals
        # are pushed) would otherwise overwrite the figures with an empty table
        sys.exit("no rf_* / rfgm_* results in this ladder report: point SNR_LADDER_DIR at one that has them")
    pairs = {s: paired(df, s) for s in SETS}
    sig = pd.concat([significance(pairs[s], s) for s in SETS], ignore_index=True)
    sig.to_csv(HERE / f"rf_significance{sfx}.csv", index=False)
    fam = {s: cells(pairs[s], sig[sig["set"] == s], s, ["family"]) for s in SETS}      # set -> (orig, twin, n_sig)
    fam_o = fam[have[0]][0]
    sizes = size_order({sz for o, t, _ in fam.values() for sz in set(o["size"]) | set(t["size"])})
    cell = ("cell = median over the {unit}'s tasks of (mean final-checkpoint score of the size's deep scheme-A seed-1904 "
            "models that trained the language and have the original and the twin scored) − the task's chance level "
            "(1 / n_options); original = acc, twins = acc_norm, and part of every twin cell is that metric choice "
            f"(`norm_offset` in rf_significance{sfx}.csv"
            + (": rf median +0.005, belebele −0.016 / Global-MMLU +0.016 / INCLUDE +0.018 on average)" if not sfx else ")"))
    sig_note = ("the z-test compares the original's acc with the twin run's own acc, so the formulation is the only "
                f"difference (p < {P_SIG})")
    note = f"{cell.format(unit='family')}; small number = tasks, and on a difference panel the tasks significant for at " \
           f"least half of the size's models — {sig_note}"
    note_lang = f"{cell.format(unit='language')}; small number = tasks, and on a difference panel the significant " \
                f"(task, model) pairs — {sig_note}"
    ncol = 1 + 2 * len(SETS)

    fig, axes = plt.subplots(1, ncol, figsize=(4.3 * ncol, 3.2))
    tables = panels(axes, fam_o, fam, "family", sizes, "benchmark")
    what = "The three letter-format families" if not args.tag else f"The {args.tag} families ({', '.join(families)})"
    G.save_highlights(fig, HERE, f"{what}, before and after the reformulations", note, tables, name=f"rf_gate{sfx}")

    lang = {s: cells(pairs[s], sig[sig["set"] == s], s, ["family", "language"]) for s in SETS}
    fig, axes = plt.subplots(len(families), ncol, figsize=(4.3 * ncol, 2.6 * len(families) + 0.8), squeeze=False)
    tables = []
    for row, f in zip(axes, families):
        sub = lambda t: t[t["family"] == f] if len(t) else t
        tables += panels(row, sub(lang[have[0]][0]), {s: tuple(map(sub, lang[s])) for s in SETS},
                         "language", sizes, f, prefix=f"{f}: ")
    G.save_highlights(fig, HERE, "Per language, before and after the reformulations", note_lang, tables, name=f"rf_gate_by_language{sfx}")

    # the markdown block: family x size, original / per set: twin, Δ, sig
    lines = [f"Gate cells (median task margin over the task's chance level, trained languages, deep scheme-A seed-1904 ladder, "
             f"from the ladder report; each set on the models that have the original and that twin scored — the original "
             f"shown is the {have[0]} pairing). Cell: original acc, then per set `twin acc_norm (**Δ** = twin − original, "
             f"n = tasks, sig)`; sig = tasks whose gain is significant for at least half of the size's models "
             f"(two-proportion z-test of the original's acc against the twin run's own acc, p < {P_SIG}"
             + ("; the acc_norm−acc offset is family-shaped, rf median +0.005, and exceeds half the plotted rf gain in "
                "36 % of the pairs" if not sfx else "") + ").", "",
             "| family | " + " | ".join(sizes) + " |", "|---|" + "---:|" * len(sizes)]
    for f in families:
        out = []
        for sz in sizes:
            o = fam_o[(fam_o["family"] == f) & (fam_o["size"] == sz)]
            if o.empty or o["margin"].isna().all():
                out.append("—"); continue
            parts = [f"{o['margin'].iloc[0]:+.3f}"]
            for s in SETS:
                po, t, k = fam[s]
                r = t[(t["family"] == f) & (t["size"] == sz)]
                oo = po[(po["family"] == f) & (po["size"] == sz)]
                if r.empty or oo.empty or r["margin"].isna().all():
                    parts.append(f"{s} —"); continue
                ks = k[(k["family"] == f) & (k["size"] == sz)]["n_sig"]
                parts.append(f"{s} {r['margin'].iloc[0]:+.3f} (**{r['margin'].iloc[0] - oo['margin'].iloc[0]:+.3f}**, "
                             f"n={int(r['n'].iloc[0])}, sig={int(ks.iloc[0]) if len(ks) else 0})")
            out.append(" · ".join(parts))
        lines.append(f"| {f} | " + " | ".join(out) + " |")
    lines += ["", f"![family x size](rf_gate{sfx}.png)", "", f"![per language](rf_gate_by_language{sfx}.png)"]
    replace_block(DOC, "rf-compare" + (f"-{args.tag}" if args.tag else ""), "\n".join(lines),
                  "analysis/rq00_task_reformulation/compare.py" + (f" --tag {args.tag}" if args.tag else ""))
    print("\n".join(lines[:4 + len(families)]))
    print(f"updated {DOC}")


if __name__ == "__main__":
    main()
