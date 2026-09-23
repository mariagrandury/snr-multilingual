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
mask, `scaling_regimes.csv`, `da_per_task.csv`, `da_reliable_tasks.csv`), so
this runs in seconds after them.

    twins_gate.png / .csv            (a) gate pass share per family and size, original vs
                                     twins, stars where McNemar p < P_SIG; (b)–(d) the
                                     headline readings with and without the twins
    twins_gate_mcnemar.csv           per (family, twin set, size): shares, discordant counts, p
    python analysis/rq00_task_reformulation/twins_gate.py --pool predictivity
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

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY, GATE_AND_CURVES, SCALING_PREDICTABILITY  # noqa: E402
from analysis.utils import TARGET_SIZE, size_order  # noqa: E402

HERE = Path(__file__).resolve().parent
SETS = ("rf", "rfgm")
P_SIG = 0.05
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
        orig = m[(m["base"] == fam) & (m["set"] == "original")].set_index("language")
        tw = g.set_index("language")
        langs = orig.index.intersection(tw.index)
        for size in sizes:
            o, t = orig.loc[langs, size], tw.loc[langs, size]
            ok = o.notna() & t.notna()
            o, t = o[ok].astype(int), t[ok].astype(int)
            b, c = int(((o == 1) & (t == 0)).sum()), int(((o == 0) & (t == 1)).sum())
            p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else np.nan
            rows.append({"family": fam, "set": s, "size": size, "languages": int(ok.sum()),
                         "share_original": o.mean() if len(o) else np.nan, "share_twin": t.mean() if len(t) else np.nan,
                         "twin_only": c, "original_only": b, "p_mcnemar": p})
    return pd.DataFrame(rows)


def headline(pool: str, mask: pd.DataFrame, sizes: list) -> pd.DataFrame:
    """Gate share, mean DA-size, reliable share and rq01 median R² on every
    task, the originals and the twins."""
    stage = load_pools()[pool].get("stage", "pretraining")
    bench = mask[~mask["family"].isin(["bpb", "loss"])].copy()
    bench["set"] = bench["task"].map(twin_set)
    da = pd.read_csv(DECISION_ACCURACY / stage / pool / "da_per_task.csv")
    da = da[da["axes"] == "multi-axis"] if "axes" in da.columns else da
    da["set"] = da["task"].map(twin_set)
    rel = pd.read_csv(DECISION_ACCURACY / stage / pool / "da_reliable_tasks.csv")
    rel = rel[rel["axes"] == "multi-axis"] if "axes" in rel.columns else rel
    rel["set"] = rel["task"].map(twin_set)
    reg = pd.read_csv(SCALING_PREDICTABILITY / "pretraining" / "predictivity_all" / "scaling_regimes.csv")
    reg["set"] = reg["task"].map(twin_set)
    rows = []
    for pop, sel in (("every task", lambda d: d), ("originals only", lambda d: d[d["set"] == "original"]),
                     ("twins only", lambda d: d[d["set"] != "original"])):
        b, d, r, q = sel(bench), sel(da), sel(rel), sel(reg)
        for size in sizes:
            gate_col = b[size].dropna()
            col = f"decision_acc_size_{size}"
            rows.append({"population": pop, "size": size, "tasks_gated": int(gate_col.notna().sum()),
                         "gate_share": gate_col.mean() if len(gate_col) else np.nan,
                         "da_size_mean": d[col].mean() if col in d.columns else np.nan,
                         "da_size_n": int(d[col].notna().sum()) if col in d.columns else 0,
                         "reliable_share": (r["da_size_median"] >= 0.66).mean() if len(r) else np.nan,
                         "reliable_n": len(r), "r2_size_median": q["r2_size"].median() if len(q) else np.nan,
                         "regime_tasks": len(q)})
    return pd.DataFrame(rows)


def figure(mc: pd.DataFrame, head: pd.DataFrame, path: Path, sizes: list) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16.4, 4.6), gridspec_kw={"width_ratios": (1.5, 1, 1, 1)})
    a = axes[0]
    fams = sorted(mc["family"].unique(), key=lambda f: -mc[mc["family"] == f]["languages"].max())
    x = np.arange(len(fams))
    w = .8 / (1 + 2 * len(sizes))
    for k, size in enumerate(sizes):
        g = mc[(mc["size"] == size) & (mc["set"] == "rf")].set_index("family").reindex(fams)
        a.bar(x + (k - len(sizes) / 2) * w * 2, g["share_original"], w, color=S.GRID, edgecolor=S.MUTED, lw=.4)
        a.bar(x + (k - len(sizes) / 2) * w * 2 + w, g["share_twin"], w, color=S.RAMP[min(k, 3)],
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
    a.set_title("(a) originals (grey) against their twins, per size; * McNemar p < 0.05", loc="left", fontsize=8.5)
    a.legend(fontsize=6, frameon=False, ncol=2); a.grid(color=S.GRID, lw=.6, axis="y"); S.clean(a)
    pops = [("every task", S.INK, "-"), ("originals only", S.MUTED, "--"), ("twins only", S.RAMP[1], "-")]
    for ax, col, lab, ttl in ((axes[1], "gate_share", "share of tasks above the gate", "(b) the gate"),
                              (axes[2], "da_size_mean", f"mean DA-size, proxy → {TARGET_SIZE}", "(c) DA-size (multi-axis, ungated mean)"),
                              (axes[3], "reliable_share", "share with median DA-size ≥ 0.66", "(d) reliable tasks")):
        for pop, c, ls in pops:
            g = head[head["population"] == pop].set_index("size").reindex(sizes)
            ax.plot(range(len(sizes)), g[col], color=c, ls=ls, marker="o", ms=4, label=pop)
        ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes); ax.set_ylabel(lab)
        ax.set_title(ttl, loc="left", fontsize=8.5); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
        if col == "da_size_mean":
            ax.axhline(0.5, color=S.MUTED, lw=.8, ls=":")
    axes[1].legend(fontsize=6.5, frameon=False)
    r2 = head[head["size"] == sizes[0]].set_index("population")["r2_size_median"]
    n = head[head["size"] == sizes[0]].set_index("population")["regime_tasks"]
    top = G._header(fig, "The reformulated twins: what they do to the gate, and to every headline reading",
                    f"(a) per benchmark family with a twin, the share of its languages above the gate (one-sided 95 % "
                    f"Wilson bound, ≥ ½ of the size's runs) for the original letter format and the `rf_` cloze twin, "
                    f"paired on the language; diamonds the `rfgm_` rewritten twin; a star where McNemar's exact test on "
                    f"the discordant languages gives p < {P_SIG}. (b)–(d) the same table read three ways — every task, "
                    f"the originals alone, the twins alone — so any figure of this analysis can be read with and without "
                    f"the twins: the gate's pass share, the ungated mean DA-size against {TARGET_SIZE} over the multi-axis "
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
        f"gate's pass share per size on every task, the originals alone and the twins alone (`twins_gate.csv` carries "
        f"the same three populations for mean DA-size, the reliable share and rq01's median R²). Regenerate with "
        f"`python analysis/rq00_task_reformulation/twins_gate.py --pool {pool}`.",
        md_table(["family", "twin", "languages", "original", "twin", "twin only / original only", "p (McNemar)"], rows),
        md_table(["population"] + sizes, [[p] + [f"{v:.2f}" for v in h.loc[p]] for p in h.index]),
        "![The twins and the gate](twins_gate.png)"])
    replace_block(HERE / "README.md", "twins-gate", body, f"twins_gate.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    args = p.parse_args()
    stage = load_pools()[args.pool].get("stage", "pretraining")
    mask = pd.read_csv(GATE_AND_CURVES / stage / args.pool / "above_random_mask.csv")
    sizes = size_order([c for c in mask.columns if c[0].isdigit()])
    mc = mcnemar(mask, sizes)
    mc.to_csv(HERE / "twins_gate_mcnemar.csv", index=False)
    head = headline(args.pool, mask, sizes)
    head.to_csv(HERE / "twins_gate.csv", index=False)
    print(mc[mc["size"] == TARGET_SIZE].round(3).to_string(index=False))
    print(head.round(3).to_string(index=False))
    figure(mc, head, HERE / "twins_gate.png", sizes)
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, mc, head, sizes)
