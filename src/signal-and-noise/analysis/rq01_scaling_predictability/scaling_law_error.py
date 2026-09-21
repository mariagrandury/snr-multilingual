"""Scaling-law error on per-language BPB: how well does a power law fitted on
the proxy rungs predict the reference rung?

Per (L, arch, scheme, language): log BPB = a − α log N is fitted on the proxy
rungs up to a ladder top (≥ 3 points; `pretrain.ladder_report._fit`, the
ladder health check's law) and predicts the reference's BPB. The reference is
TARGET_SIZE (rule 9), never the largest size a chain happens to have: a chain
without a TARGET_SIZE final is dropped and named (`!!! RULE 9:` on stdout, the
README block). The relative error is signed, (predicted − observed) / observed,
so a negative value is the power law under-predicting BPB; it is reported per
ladder top, so the table reads "how far up the ladder must one train before
the reference is predicted within x %". A constant offset between small and
large models shows up here but not in the decision accuracy of rq05, so the
two reads can disagree; the plan's "prediction ability" read. The loader
keeps trained languages only (rule 2), so every chain is a trained language.

    scaling_law_error.csv   per (L, arch, scheme, language, ladder top): α, predicted vs observed, signed relative error
                            (the loader keeps trained languages only, so every chain is a trained one; bpb_macro is left out, rule 7)
    scaling_law_error.png   median |relative error| per L and ladder top (deep, scheme A)

    python analysis/rq01_scaling_predictability/scaling_law_error.py --pool predictivity_all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from pretrain.ladder_report import _fit  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import SCALING_PREDICTABILITY  # noqa: E402
from analysis.utils import (  # noqa: E402
    GRID_SEED, NON_EMB, TARGET_SIZE, assign_language, finals, ladder_frame, languages_only, size_order)

OUT_ROOT = SCALING_PREDICTABILITY
CANONICAL = "predictivity_all"
mpl.rcParams.update(S.RC)


def scaling_law_error(fin: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """The fits, and the (L, arch, scheme) settings dropped for lack of a
    TARGET_SIZE final (rule 9: the reference is never a smaller size)."""
    sub = fin[(fin["seed"] == GRID_SEED) & (fin["kind"] == "bpb")]
    sub = languages_only(sub.assign(language=sub["task"].map(assign_language)))   # rule 7: no bpb_macro chain
    rows, dropped = [], []
    for (L, arch, scheme, task), g in sub.groupby(["L", "arch", "scheme", "task"]):
        g = g.assign(N=g["size"].map(NON_EMB)).dropna(subset=["N"]).sort_values("N")
        if TARGET_SIZE not in set(g["size"]):
            if (int(L), arch, scheme) not in dropped:
                dropped.append((int(L), arch, scheme))
            continue
        ref = g[g["size"] == TARGET_SIZE].iloc[0]
        proxies = g[g["N"] < ref["N"]]
        if len(proxies) < 3:
            continue
        for top in range(3, len(proxies) + 1):
            pts = proxies.iloc[:top]
            fit = _fit(list(zip(pts["N"], pts["primary_score"])))
            if not fit:
                continue
            slope, icpt = fit
            pred = float(np.exp(icpt + slope * np.log(ref["N"])))
            rows.append({
                "L": int(L), "arch": arch, "scheme": scheme, "task": task,
                "language": assign_language(task),
                "ladder_top": pts["size"].iloc[-1], "n_points": top,
                "reference_size": ref["size"], "alpha": -slope,
                "predicted": pred, "observed": float(ref["primary_score"]),
                "rel_error": (pred - float(ref["primary_score"])) / float(ref["primary_score"]),
            })
    if dropped:
        print(f"!!! RULE 9: no {TARGET_SIZE} final, chains dropped (never referenced at a smaller size): "
              + ", ".join(f"L{L} {a}/{s}" for L, a, s in dropped))
    return pd.DataFrame(rows), dropped



def plot_scaling_error(sle: pd.DataFrame, path: Path) -> None:
    sub = sle[(sle["arch"] == "deep") & (sle["scheme"] == "A")]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for L, g in sub.groupby("L"):
        med = g.groupby("ladder_top")["rel_error"].apply(lambda v: np.median(np.abs(v)))
        tops = size_order(med.index)
        ax.plot(tops, [med[t] for t in tops], marker="o", label=f"L{L}")
    ax.set_xlabel("largest proxy rung in the fit")
    ax.set_ylabel("median |relative error| of predicted BPB")
    ax.grid(color=S.GRID, lw=.6); ax.legend(fontsize=7, ncol=2, frameon=False); S.clean(ax)
    neg = int((sub["rel_error"] < 0).sum())
    top = G._header(fig, f"Per-language BPB power-law prediction of the {TARGET_SIZE} rung",
                    f"point = median over the L's trained languages of |(predicted − observed) / observed| of the {TARGET_SIZE} BPB, "
                    f"log BPB = a − α log N fitted on the proxy rungs up to the x rung (deep, scheme A, seed {GRID_SEED}; {sub['task'].nunique()} "
                    f"languages, {len(sub)} fits); the error is signed in the CSV and negative in {neg} of {len(sub)} rows = the law "
                    f"under-predicts BPB; chains without a {TARGET_SIZE} final are dropped (rule 9)")
    fig.tight_layout(rect=(0, 0, 1, top)); S.save(fig, path, dpi=140)



def generate_readme(pool: str, out_dir: Path, sle: pd.DataFrame, dropped: list) -> None:
    if pool != CANONICAL or sle.empty:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    core = sle[(sle["arch"] == "deep") & (sle["scheme"] == "A")]
    if core.empty:
        return
    m = (core.groupby(["L", "ladder_top"])["rel_error"]
         .apply(lambda v: float(np.median(np.abs(v)))).unstack("ladder_top"))
    m = m[size_order(m.columns)]
    signed = core.groupby(["L", "ladder_top"])["rel_error"].median().unstack("ladder_top")[m.columns]
    n_lang = core.groupby("L")["task"].nunique()
    rows = [[f"L{L}", int(n_lang[L])] + [fmt(m.loc[L, c], 3) for c in m.columns] for L in m.index]
    body = "\n\n".join([
        "## Scaling-law error on per-language BPB",
        f"Numbers from the `{pool}` pool (deep, scheme A, seed {GRID_SEED}; the loader keeps trained languages only). The reference is "
        f"{TARGET_SIZE} for every chain (rule 9); a chain without a {TARGET_SIZE} final is dropped, never referenced at a smaller size"
        + (": dropped " + ", ".join(f"L{L} {a}/{s}" for L, a, s in dropped) if dropped else "; none dropped") + ". "
        f"Regenerate with `python analysis/rq01_scaling_predictability/scaling_law_error.py --pool {pool}`.",
        f"- **Scaling-law error** — median |relative error| of the {TARGET_SIZE} per-language BPB "
        "predicted from the proxy ladder: "
        + ", ".join(f"L{L} {fmt(m.loc[L].dropna().iloc[-1], 3)}" for L in m.index)
        + " (largest proxy ladder at that L).",
        f"- **Sign** — `rel_error` in the CSV is signed, (predicted − observed) / observed: {int((core['rel_error'] < 0).sum())} of "
        f"{len(core)} plotted fits are negative, the power law under-predicts the {TARGET_SIZE} BPB; median signed error at the largest "
        "proxy ladder: " + ", ".join(f"L{L} {fmt(signed.loc[L].dropna().iloc[-1], 3)}" for L in signed.index) + ".",
        "**Median |relative error|** (columns: largest proxy rung in the fit; languages = trained languages behind the median):",
        md_table(["L", "languages"] + list(m.columns), rows),
        f"![Scaling-law error]({rel}/scaling_law_error.png)"])
    readme = OUT_ROOT / "README.md"
    replace_block(readme, "scaling-law-error", body, f"scaling_law_error.py --pool {pool}")
    print(f"Wrote auto README block → {readme}")


def main(pool: str, out_dir: Path) -> None:
    df = ladder_frame(pool)
    fin = finals(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    sle, dropped = scaling_law_error(fin)
    sle.to_csv(out_dir / "scaling_law_error.csv", index=False)
    print(f"Wrote → {out_dir / 'scaling_law_error.csv'} ({len(sle)} fits, reference {TARGET_SIZE})")
    if not sle.empty:
        plot_scaling_error(sle, out_dir / "scaling_law_error.png")
    generate_readme(pool, out_dir, sle, dropped)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Ladder pool from configs/models.json (default: {CANONICAL})")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    main(args.pool, OUT_ROOT / stage / args.pool)
