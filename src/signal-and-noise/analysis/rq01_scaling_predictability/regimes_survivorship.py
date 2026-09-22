"""The scaling-regimes figure with what it leaves out put back: how many tasks
the above-random gate removed before a point was drawn, per family.

Exploratory: this feeds plan/decision_accuracy.md and does not replace
`regimes.py`, whose table it reads. The paper figure
(`scaling_regimes_outliers_paper.png`) shows the 110 tasks that survive rule 1
and reads "predictable across both" for 102 of them; it does not show that 224
tasks were gated out first, among them three whole families (global_piqa
parallel, belebele, global_mmlu) with no survivor at all. A reader of the
figure alone concludes that scaling is predictable; the population it is
predictable on is the finding this figure adds.

  (a) per family: tasks with a point in the regimes figure against tasks the
      gate removed (a task with a size fit somewhere but at chance wherever a
      fit was possible, `rq1_fits.csv`), sorted by the total
  (b) regimes.py's panel (b) — median R² of the log-N fit against median R² of
      the training-trajectory fit — with every family label carrying kept/total
      and the families with no survivor listed in the panel

    scaling_regimes_survivorship.png / .csv   the figure and the per-family table

    python analysis/rq01_scaling_predictability/regimes_survivorship.py --pool predictivity_all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.rq01_scaling_predictability.regimes import (  # noqa: E402
    LABEL_DARK, OUT_ROOT, R2_SPLIT, _colours, _labels, _quadrants, darken, family_table, short)

mpl.rcParams.update(S.RC)


def survivorship(t: pd.DataFrame, fits: pd.DataFrame) -> pd.DataFrame:
    """Per family: tasks with any size fit, tasks with a point, and the gap."""
    fits = fits[fits["kind"] != "loss"]                               # rule 7
    total = fits.groupby("family")["task"].nunique().rename("tasks_with_fit")
    kept = t.groupby("family")["task"].nunique().rename("in_figure")
    s = pd.concat([total, kept], axis=1).fillna(0).astype(int)
    s["removed"] = s["tasks_with_fit"] - s["in_figure"]
    s["share_kept"] = (s["in_figure"] / s["tasks_with_fit"]).round(3)
    fam = family_table(t).set_index("family")
    s = s.join(fam[["r2_size", "r2_trajectory", "regime_majority"]])
    return s.sort_values("tasks_with_fit", ascending=False).reset_index()


def figure(t: pd.DataFrame, s: pd.DataFrame, path: Path, n_twins: int) -> None:
    colours = _colours(t)
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.4, 5.2), gridspec_kw={"width_ratios": (1, 1.25)})
    # (a) survivorship, one bar per family
    y = range(len(s))[::-1]
    a.barh(y, s["in_figure"], color=[colours.get(f, S.MUTED) for f in s["family"]], label="has a point (survives rule 1)")
    a.barh(y, s["removed"], left=s["in_figure"], color=S.GRID, hatch="///", edgecolor=S.MUTED, lw=0, label="removed by the gate")
    a.set_yticks(list(y)); a.set_yticklabels([short(f) for f in s["family"]], fontsize=7.5)
    for yi, (k, n) in zip(y, zip(s["in_figure"], s["tasks_with_fit"])):
        a.text(n + 0.8, yi, f"{k}/{n}", va="center", fontsize=6.8, color=S.INK if k else "#8c1d18")
    a.set_xlabel("benchmark-language tasks with a size fit")
    a.set_xlim(0, s["tasks_with_fit"].max() * 1.16)
    a.legend(fontsize=7, frameon=False, loc="lower right")
    a.grid(color=S.GRID, lw=.6, axis="x"); S.clean(a)
    # (b) the regimes panel, family labels carrying kept/total
    _quadrants(b, labels=False)
    b.scatter(t["r2_size"], t["r2_trajectory"], s=9, c=[colours[f] for f in t["family"]], alpha=.7, lw=0, zorder=2)
    fam = s[s["in_figure"] > 0]
    b.scatter(fam["r2_size"], fam["r2_trajectory"], s=46, c=[colours[f] for f in fam["family"]], edgecolor=S.INK, lw=.8, zorder=3)
    _labels(b, fam["r2_size"], fam["r2_trajectory"],
            [f"{short(f)} ({k}/{n})" for f, k, n in zip(fam["family"], fam["in_figure"], fam["tasks_with_fit"])],
            [darken(colours[f], LABEL_DARK) for f in fam["family"]], fontsize=6.8, weight="bold")
    gone = s[s["in_figure"] == 0]
    if len(gone):
        b.text(0.02, 0.03, "no survivor at any size:\n" + "\n".join(f"{short(f)}  (0/{n})" for f, n in zip(gone["family"], gone["tasks_with_fit"])),
               transform=b.transAxes, fontsize=6.8, color="#8c1d18", va="bottom", ha="left",
               bbox=dict(boxstyle="round,pad=0.35", fc=S.SURFACE, ec="#8c1d18", lw=.6, alpha=.92), zorder=5)
    b.set_xlim(-0.02, 1.02); b.set_ylim(-0.02, 1.02); b.set_aspect("equal")
    b.set_xlabel("median R² across model-size scaling fits"); b.set_ylabel("median R² across training-trajectory fits")
    b.grid(color=S.GRID, lw=.6); S.clean(b)
    twins = (f"The {n_twins} rf_/rfgm_ twin tasks now in the pool are not in this table yet — rerun regimes.py first. "
             if n_twins else "")
    top = G._header(fig, "Scaling regimes, with the gate's survivorship",
                    f"(a) per family, the tasks the regimes figure draws against the tasks rule 1 removed before it could "
                    f"(at chance at every size where a log-N fit was possible; `rq1_fits.csv`). (b) `regimes.py`'s panel (b) — "
                    f"one point per surviving task, the family at its median point, labelled kept/total; quadrants split at "
                    f"R² = {R2_SPLIT}. {int(s['in_figure'].sum())} of {int(s['tasks_with_fit'].sum())} tasks survive; "
                    f"{len(gone)} families lose every task. {twins}Same population and medians as `scaling_regimes.csv`.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default="predictivity_all")
    args = p.parse_args()
    out_dir = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    t = pd.read_csv(out_dir / "scaling_regimes.csv")
    fits = pd.read_csv(out_dir / "rq1_fits.csv")
    twin = lambda d: int(d["task"][d["task"].str.startswith(("rf_", "rfgm_"))].nunique())
    n_twins = twin(fits) - twin(t)           # twins with a fit but no point yet
    print(f"rf_/rfgm_ twins: {twin(fits)} in rq1_fits.csv, {twin(t)} in scaling_regimes.csv")
    s = survivorship(t, fits)
    s.to_csv(out_dir / "scaling_regimes_survivorship.csv", index=False)
    print(s.to_string(index=False))
    figure(t, s, out_dir / "scaling_regimes_survivorship.png", n_twins)
