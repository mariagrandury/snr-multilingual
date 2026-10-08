"""What the reformulated twins do to the gate, and to every headline reading.

Two questions the paper has to answer once the `rf_` (letters → cloze) and
`rfgm_` (Gemini-rewritten items) twins are in the pool:

  1. Does a twin clear the above-random gate where its original does not,
     and is that more than chance? Per (family, size) the share of the
     family's languages above the gate for the original and for each twin,
     paired on the language, with McNemar's exact test on the discordant
     languages (original passes / twin fails against the reverse) — the
     honest test, because the same languages are evaluated on the same
     models in both formulations.
  2. Which headline numbers move when the twins are in? The gate's pass
     share, the mean DA-size against 1.7B, the share of reliable tasks and
     rq01's median R², each read on every task, on the originals alone and
     on the twins alone — so a reader of any figure knows what the twins
     contribute without a second copy of the figure.

Everything is read from the tables the other RQs already write (the gate
mask, `scaling_regimes.csv`, `da_all_per_task_both_axes.csv`, `da_all_reliable_tasks_both_axes.csv`), so
this runs in seconds after them.

    reformulations_gate.png / .csv   (a) gate pass share per family and size, original vs
                                     twins, stars where McNemar p < P_SIG; (b)–(d) the
                                     headline readings with and without the twins
    reformulations_gate_mcnemar.csv  per (family, twin set, size): shares, discordant counts, p
    reformulations_gate_paper.png / .csv   panel (a) alone, no header, for the paper
    python analysis/rq00_task_reformulation/reformulations_gate.py --pool predictivity
    python analysis/rq00_task_reformulation/reformulations_gate.py --paper   # the paper panel alone, from the CSV
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY, GATE_AND_CURVES, SCALING_PREDICTABILITY  # noqa: E402
from analysis.utils import BBPB, TARGET_SIZE, passes_gate, size_order  # noqa: E402

HERE = Path(__file__).resolve().parent
SETS = ("rf", "rfgm")
P_SIG = 0.05
# A pair tied on one side counts as a miss in the DA kernel, so a proxy with no
# signal reads 0.5 × (1 − share of one-sided ties), not 0.5: 0.47 on this ladder
# (mean one-sided tie share 6.3 % per cell, `agreement_da_size_per_cell_multi_axes.csv`).
TIE_NULL = 0.47
mpl.rcParams.update(S.RC)


def twin_set(task: str) -> str:
    return "rfgm" if task.startswith("rfgm_") else "rf" if task.startswith("rf_") else "original"


def base_family(family: str) -> str:
    return family.split("_", 1)[1] if family.startswith(("rf_", "rfgm_")) else family


def mcnemar(mask: pd.DataFrame, sizes: list) -> pd.DataFrame:
    """Per (base family, twin set, size): paired gate verdicts over the
    languages both formulations score, and McNemar's exact p."""
    m = mask.copy()
    m["set"] = m["task"].map(twin_set)
    m["base"] = m["family"].map(base_family)
    rows = []
    for (fam, s), g in m.groupby(["base", "set"]):
        if s == "original":
            continue
        # paired on the task a twin rewrites (`rf_<task>` -> `<task>`), not on its language:
        # a family can hold several tasks per language (CulturalBench countries, belebele zh)
        orig = m[(m["base"] == fam) & (m["set"] == "original")].set_index("task")
        tw = g.assign(task=g["task"].str.removeprefix(f"{s}_")).set_index("task")
        common = orig.index.intersection(tw.index)
        for size in sizes:
            o, t = orig.loc[common, size], tw.loc[common, size]
            ok = o.notna() & t.notna()
            o, t = o[ok].astype(int), t[ok].astype(int)
            b, c = int(((o == 1) & (t == 0)).sum()), int(((o == 0) & (t == 1)).sum())
            p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else np.nan
            rows.append({"family": fam, "set": s, "size": size, "languages": int(ok.sum()),   # paired tasks
                         "n_languages": orig.loc[o.index, "language"].nunique(),   # CulturalBench: 19 countries over 8 languages
                         "share_original": o.mean() if len(o) else np.nan, "share_twin": t.mean() if len(t) else np.nan,
                         "twin_only": c, "original_only": b, "p_mcnemar": p})
    return pd.DataFrame(rows)


def headline(pool: str, mask: pd.DataFrame, sizes: list) -> pd.DataFrame:
    """Gate share, mean DA-size, reliable share and rq01 median R² on every
    task, the originals and the twins."""
    stage = load_pools()[pool].get("stage", "pretraining")
    # the question is the accuracy formulations (original, rf, rfgm): a `bbpb_` twin is
    # the same items read as bits per byte, has no gate, and would count as an "original"
    no_bbpb = lambda d: d[~d["task"].str.startswith(BBPB)].copy()
    bench = no_bbpb(mask[~mask["family"].isin(["bpb", "loss"])])
    bench["set"] = bench["task"].map(twin_set)
    da = pd.read_csv(DECISION_ACCURACY / stage / pool / "da_all_per_task_both_axes.csv")
    da = no_bbpb(da[da["axes"] == "multi-axis"] if "axes" in da.columns else da)
    da["set"] = da["task"].map(twin_set)
    rel = pd.read_csv(DECISION_ACCURACY / stage / pool / "da_all_reliable_tasks_both_axes.csv")
    rel = no_bbpb(rel[rel["axes"] == "multi-axis"] if "axes" in rel.columns else rel)
    rel["set"] = rel["task"].map(twin_set)
    reg = pd.read_csv(SCALING_PREDICTABILITY / "pretraining" / "predictivity_seeds" / "scaling_regimes.csv")
    reg = reg[reg["task"].isin(bench["task"])].copy()        # benchmarks only: per-language BPB has no twin and would count as an original
    reg["set"] = reg["task"].map(twin_set)
    passes = mask.set_index("task")
    rows = []
    for pop, sel in (("every task", lambda d: d), ("originals only", lambda d: d[d["set"] == "original"]),
                     ("twins only", lambda d: d[d["set"] != "original"])):
        b, d, r, q = sel(bench), sel(da), sel(rel), sel(reg)
        for size in sizes:
            gate_col = b[size].dropna()
            col = f"decision_acc_size_{size}"
            # rule 1: the DA mean is over the tasks above chance at the proxy AND at the reference
            # (passes_gate: a mask of NA, the generative tasks, passes); benchmarks only, as the gate share
            ok = (d["task"].isin(bench["task"]) & passes_gate(passes, d["task"], size, TARGET_SIZE).to_numpy()) \
                if col in d.columns else pd.Series(False, index=d.index)
            rows.append({"population": pop, "size": size, "tasks_gated": int(gate_col.notna().sum()),
                         "gate_share": gate_col.mean() if len(gate_col) else np.nan,
                         "da_size_mean": d.loc[ok, col].mean() if col in d.columns else np.nan,
                         "da_size_n": int(d.loc[ok, col].notna().sum()) if col in d.columns else 0,
                         "da_size_mean_ungated": d.loc[d["task"].isin(bench["task"]), col].mean() if col in d.columns else np.nan,
                         "reliable_share": (r["da_size_median"] >= 0.66).mean() if len(r) else np.nan,
                         "reliable_n": len(r), "r2_size_median": q["r2_size"].median() if len(q) else np.nan,
                         "regime_tasks": len(q)})
    return pd.DataFrame(rows)


def _twins_ax(a, mc: pd.DataFrame, sizes: list) -> None:
    """Panel (a): per family with a twin, the share of its languages above the
    gate for the original (grey) and the `rf_` twin, per size; diamonds the
    `rfgm_` twin; a star where McNemar's p < P_SIG."""
    fams = sorted(mc["family"].unique(), key=lambda f: -mc[mc["family"] == f]["languages"].max())
    x = np.arange(len(fams))
    w = .8 / (1 + 2 * len(sizes))
    for k, size in enumerate(sizes):
        g = mc[(mc["size"] == size) & (mc["set"] == "rf")].set_index("family").reindex(fams)
        a.bar(x + (k - len(sizes) / 2) * w * 2, g["share_original"], w, color=S.GRID, edgecolor=S.MUTED, lw=.4)
        a.bar(x + (k - len(sizes) / 2) * w * 2 + w, g["share_twin"], w, color=S.SIZE_COLOR[size],
              label=f"rf twin at {size}")
        gm = mc[(mc["size"] == size) & (mc["set"] == "rfgm")].set_index("family").reindex(fams)
        a.scatter(x + (k - len(sizes) / 2) * w * 2 + w, gm["share_twin"], s=14, marker="D", color=S.SERIES[1], zorder=4,
                  label="rfgm twin" if k == 0 else None)
        for i, f in enumerate(fams):
            p = g.loc[f, "p_mcnemar"]
            if p == p and p < P_SIG:
                a.text(x[i] + (k - len(sizes) / 2) * w * 2 + w, g.loc[f, "share_twin"] + .02, "*", ha="center", fontsize=8, color=S.INK)
    a.set_xticks(x); a.set_xticklabels(fams, rotation=30, ha="right", fontsize=7)
    a.set_ylim(0, 1.12); a.set_ylabel("share of the family's languages above the gate")
    a.legend(fontsize=6, frameon=False, ncol=2); a.grid(color=S.GRID, lw=.6, axis="y"); S.clean(a)


def figure_paper(mc: pd.DataFrame, path: Path, sizes: list) -> None:
    """Panel (a) alone, in the paper's words: `rf` is the RF version, `rfgm` the LLM-RF version.
    Each benchmark is named with its language count; the legend is three columns of three:
    the original, the LLM-RF version and the test, then the RF version by size."""
    fig, a = plt.subplots(figsize=(6.4, 3.4))
    _twins_ax(a, mc, sizes)
    h, labels = a.get_legend_handles_labels()
    rf = [(x, l.replace("rf twin", "RF version")) for x, l in zip(h, labels) if l.startswith("rf twin")]
    gm = [(x, "LLM-RF version") for x, l in zip(h, labels) if l == "rfgm twin"]
    handles = [(plt.Rectangle((0, 0), 1, 1, facecolor=S.GRID, edgecolor=S.MUTED, lw=.4), "Original")] + gm \
        + [(plt.Line2D([], [], marker="$*$", ls="none", color=S.INK, ms=7), f"McNemar p < {P_SIG}")] + rf
    a.legend([x for x, _ in handles], [l for _, l in handles], fontsize=6, frameon=False, ncol=3,
             loc="lower center", bbox_to_anchor=(0.5, 1.0))
    a.set_ylabel("Share of tasks above threshold")
    n_lang = mc.groupby("family")["n_languages"].max()      # the label counts languages, the share is over tasks
    a.set_xticklabels([f"{G.paper_name(t.get_text())} ({n_lang[t.get_text()]})" for t in a.get_xticklabels()],
                      rotation=30, ha="right", fontsize=7)
    fig.tight_layout()
    mc.to_csv(path.with_suffix(".csv"), index=False)
    S.save_paper(fig, path.with_suffix(""))


def figure(mc: pd.DataFrame, head: pd.DataFrame, path: Path, sizes: list) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.6), gridspec_kw={"width_ratios": (1.5, 1, 1, 1)})
    _twins_ax(axes[0], mc, sizes)
    axes[0].set_title("(a) originals (grey) against their twins, per size; * McNemar p < 0.05", loc="left", fontsize=8.5)
    pops = [("every task", S.INK, "-"), ("originals only", S.MUTED, "--"), ("twins only", S.RAMP[1], "-")]
    for ax, col, lab, ttl in ((axes[1], "gate_share", "share of tasks above the gate", "(b) the gate"),
                              (axes[2], "da_size_mean", f"mean DA-size, proxy → {TARGET_SIZE}", "(c) DA-size (multi-axis, gated at proxy and reference)"),
                              (axes[3], "reliable_share", "share with median DA-size ≥ 0.66", "(d) reliable tasks")):
        for pop, c, ls in pops:
            g = head[head["population"] == pop].set_index("size").reindex(sizes)
            ax.plot(range(len(sizes)), g[col], color=c, ls=ls, marker="o", ms=4, label=pop)
        ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes); ax.set_ylabel(lab)
        ax.set_title(ttl, loc="left", fontsize=8.5); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
        if col == "da_size_mean":
            ax.axhline(0.5, color=S.MUTED, lw=.8, ls=":")
            ax.axhline(TIE_NULL, color="#b3261e", lw=.8, ls=":")
            ax.text(len(sizes) - 1, TIE_NULL, f" tie null {TIE_NULL:.2f}", fontsize=6, color="#b3261e", va="bottom", ha="right")
    axes[1].legend(fontsize=6.5, frameon=False)
    r2 = head[head["size"] == sizes[0]].set_index("population")["r2_size_median"]
    n = head[head["size"] == sizes[0]].set_index("population")["regime_tasks"]
    top = G._header(fig, "The reformulated twins: what they do to the gate, and to every headline reading",
                    f"(a) per benchmark family with a twin, the share of its languages above the gate (one-sided 95 % "
                    f"Wilson bound, ≥ ½ of the size's runs) for the original letter format and the `rf_` cloze twin, "
                    f"paired on the language; diamonds the `rfgm_` rewritten twin; a star where McNemar's exact test on "
                    f"the discordant languages gives p < {P_SIG}. (b)–(d) the same table read three ways — every task, "
                    f"the originals alone, the twins alone — so any figure of this analysis can be read with and without "
                    f"the twins: the gate's pass share, the mean DA-size against {TARGET_SIZE} over the multi-axis pairs on the tasks above chance at "
                    f"both sizes (rule 1; the red line is the tie null, what a proxy with no signal reads because a pair tied on one side is a miss)"
                    f" "
                    f"pairs, and the share of tasks whose median DA-size clears 0.66. rq01's median size-fit R² is "
                    + ", ".join(f"{p} {r2[p]:.3f} ({int(n[p])} tasks)" for p in r2.index) + ".")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, mc: pd.DataFrame, head: pd.DataFrame, sizes: list) -> None:
    ref = mc[mc["size"] == TARGET_SIZE]
    rows = [[r["family"], r["set"], int(r["languages"]), f"{r['share_original']:.2f}", f"{r['share_twin']:.2f}",
             f"{int(r['twin_only'])} / {int(r['original_only'])}", f"{r['p_mcnemar']:.3g}" if r["p_mcnemar"] == r["p_mcnemar"] else "—"]
            for _, r in ref.sort_values(["family", "set"]).iterrows()]
    h = head.pivot_table(index="population", columns="size", values="gate_share").reindex(columns=sizes).round(2)
    body = "\n\n".join([
        "## The twins and the gate, with significance",
        f"Per family at {TARGET_SIZE}: the share of languages above the gate for the original and the twin, paired on the "
        f"language; `twin only / original only` are the discordant languages McNemar's exact test is run on. Below, the "
        f"gate's pass share per size on every task, the originals alone and the twins alone (`reformulations_gate.csv` carries "
        f"the same three populations for mean DA-size, the reliable share and rq01's median R²). Regenerate with "
        f"`python analysis/rq00_task_reformulation/reformulations_gate.py --pool {pool}`.",
        md_table(["family", "twin", "languages", "original", "twin", "twin only / original only", "p (McNemar)"], rows),
        md_table(["population"] + sizes, [[p] + [f"{v:.2f}" for v in h.loc[p]] for p in h.index]),
        "![The reformulations and the gate](reformulations_gate.png)"])
    replace_block(HERE / "README.md", "reformulations-gate", body, f"reformulations_gate.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    p.add_argument("--paper", action="store_true", help="only the paper panel, from reformulations_gate_mcnemar.csv")
    args = p.parse_args()
    stage = load_pools()[args.pool].get("stage", "pretraining")
    mask = pd.read_csv(GATE_AND_CURVES / stage / args.pool / "above_random_mask.csv")
    sizes = size_order([c for c in mask.columns if c[0].isdigit()])
    if args.paper:
        figure_paper(pd.read_csv(HERE / "reformulations_gate_mcnemar.csv"), HERE / "reformulations_gate_paper.png", sizes)
        sys.exit()
    mc = mcnemar(mask, sizes)
    mc.to_csv(HERE / "reformulations_gate_mcnemar.csv", index=False)
    head = headline(args.pool, mask, sizes)
    head.to_csv(HERE / "reformulations_gate.csv", index=False)
    print(mc[mc["size"] == TARGET_SIZE].round(3).to_string(index=False))
    print(head.round(3).to_string(index=False))
    figure(mc, head, HERE / "reformulations_gate.png", sizes)
    figure_paper(mc, HERE / "reformulations_gate_paper.png", sizes)
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, mc, head, sizes)
