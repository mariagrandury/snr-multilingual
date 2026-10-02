"""Multi-axis vs mono-axis pairs: the same three decision accuracies, on the same
reliable population, with the PAIR SET as the only thing that changes.

Both rows are filtered by the MULTI-AXIS reliable tasks, the one stated exception
to rule 15: filtering each row by its own pair set would change the task
population together with the pair set.

The production form — the `axes` column of `da_per_task.csv` (compute_da.py),
so every rq02 table carries both readings — is in place; this script is the
side-by-side view of the two on one population (plan/decision_accuracy.md).

    multi-axis   every pair of design variants at the grid seed: rq02's convention
                 to date. Two thirds of these pairs move two or three axes at once
                 ("L8-A-deep vs L50-B-shallow"), a comparison nobody makes.
    mono-axis    the pairs that differ on exactly ONE of L, arch, list, T, lang2, en.
                 This is the structure upstream had by construction — DataDecide's
                 recipes differ only in the data mix, so every one of its pairs is
                 a single-axis decision — generalised to a grid with seven axes. The
                 seed is held at the grid seed in both sets, so a seed pair (two
                 draws of one design, which decide nothing) is in neither.

Three definitions, all on the ten evaluated checkpoints where a checkpoint axis
exists (rule 3), gated per rule 1 — DA-size and DA-goal at the proxy AND the
reference, DA-ckpt at the proxy alone, since its reference is the proxy's own
final — with MIN_PAIRS per cell (rule 5), pooled over decisions:

    DA-size   proxy's final vs the reference's final, one point per size
    DA-ckpt   proxy at a checkpoint vs the SAME size's final
    DA-goal   proxy at a checkpoint vs the reference's final

    rq2_above_66_both_axes.png / .csv   2 x 3: rows = pair set, columns = the three
                                        definitions, over the above_66_both cells

    python analysis/rq02_decision_accuracy/pair_axes.py --pool predictivity
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
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq02_decision_accuracy.early_small import SAFE_DA  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import load_reliable  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import (  # noqa: E402
    FRACS, POOL, TAU, decisions, grid_frame, reliability)
from analysis.utils import (  # noqa: E402
    DESIGN_AXES, MIN_PAIRS, NON_EMB, TARGET_SIZE, design_axes, finals, ladder_frame,
    pair_sets, size_order)

OUT_ROOT = DECISION_ACCURACY
VARIANT = "above_66_both"
AXES = ("multi-axis", "mono-axis")
PANELS = ("DA-size", "DA-ckpt", "DA-goal")
mpl.rcParams.update(S.RC)


def decision_sets(attrs: pd.DataFrame) -> dict[str, list]:
    """The two decision sets `utils.pair_sets` defines, without its null."""
    ps = pair_sets(attrs)
    return {a: ps[a] for a in AXES}


def cells(df: pd.DataFrame, fin: pd.DataFrame, pairs: dict, sizes: list) -> pd.DataFrame:
    """Per (panel, axes, task, size, frac): matching and comparable decisions."""
    grid = grid_frame(df, FRACS)
    out = []
    out.append(reliability(decisions(fin.assign(frac=1.0), pairs, sizes, fin)).assign(panel="DA-size"))
    out.append(reliability(decisions(grid, pairs, sizes, fin, FRACS)).assign(panel="DA-goal"))
    for s in sizes:              # DA-ckpt: the reference is the proxy's OWN final
        out.append(reliability(decisions(grid, pairs, [s], fin, FRACS[:-1], ref=s)).assign(panel="DA-ckpt"))
    return pd.concat(out, ignore_index=True).rename(columns={"group": "axes"})


def pooled(c: pd.DataFrame, pool: str, keep: set) -> pd.DataFrame:
    """Rules 1 and 5, the population, then decisions pooled per (panel, axes, size, frac)."""
    c = c[c["task"].isin(keep) & (c["n_comparable"] >= MIN_PAIRS)]
    # The reference's own final against itself is 1.0 by construction: not a
    # reading of DA-goal, so it is not a point of that line (DA-size keeps it as
    # the hollow end point, as every scale-convergence figure does).
    c = c[~((c["panel"] == "DA-goal") & (c["size"] == TARGET_SIZE) & (c["frac"] == 1.0))]
    parts = []
    for panel, ref in (("DA-size", TARGET_SIZE), ("DA-goal", TARGET_SIZE), ("DA-ckpt", None)):
        g = G.mark_gated(G.add_meta(c[c["panel"] == panel]), pool, "size", "da", ref).dropna(subset=["da"])
        parts.append(g[g["family"] != "bpb"])
    c = pd.concat(parts)
    out = (c.groupby(["panel", "axes", "size", "frac"])
           .agg(n_matching=("n_matching", "sum"), n_comparable=("n_comparable", "sum"),
                n_tasks=("task", "nunique")).reset_index())
    out["da"] = out["n_matching"] / out["n_comparable"]
    out["non_emb"] = out["size"].map(NON_EMB)
    out["chinchilla"] = out["frac"] * 5            # every run trains 5C
    return out


def figure(out: pd.DataFrame, path: Path, n_cells: int) -> None:
    sizes = size_order(out["size"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharey=True)
    for r, ax_set in enumerate(AXES):
        for c, panel in enumerate(PANELS):
            ax = axes[r, c]
            d = out[(out["axes"] == ax_set) & (out["panel"] == panel)]
            if panel == "DA-size":
                d = d.sort_values("non_emb")
                real, ref = d[d["size"] != TARGET_SIZE], d[d["size"] == TARGET_SIZE]
                ax.plot(real["non_emb"], real["da"], color=S.INK, marker="o", ms=4, lw=2.0, zorder=3)
                if len(ref):
                    ax.plot(ref["non_emb"], ref["da"], marker="o", ms=4.5, mfc=S.SURFACE, mec=S.INK, mew=1.2, ls="none")
                ax.axhline(TAU, color=S.MUTED, lw=.8, ls=":")
                ax.set_xscale("log"); ax.set_xticks([NON_EMB[s] for s in sizes]); ax.set_xticklabels(sizes)
                if r == 1:
                    ax.set_xlabel("model size (non-embedding parameters)")
            else:
                for s in sizes:
                    g = d[d["size"] == s].sort_values("chinchilla")
                    if len(g):
                        ax.plot(g["chinchilla"], g["da"], color=S.SIZE_COLOR.get(s, S.MUTED), marker="o", ms=3.5, lw=1.4, label=s)
                ax.axhline(SAFE_DA, color=S.MUTED, lw=.8, ls=":")
                ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels([G.chinchilla(f) for f in (.2, .4, .6, .8, 1.0)])
                ax.set_xlim(0.3, 5.2)
                if r == 1:
                    ax.set_xlabel("proxy's training tokens (× Chinchilla)")
            ax.set_ylim(0.25, 1.0); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
            ax.set_title(f"{panel} — {ax_set}", loc="left", fontsize=8.5)
        axes[r, 0].set_ylabel("decision accuracy")
    axes[0, 1].legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2, title="proxy size", title_fontsize=6.5)
    n_pairs = {a: int(out[(out["axes"] == a) & (out["panel"] == "DA-size") & (out["size"] == TARGET_SIZE)]["n_comparable"].iloc[0])
               for a in AXES}
    top = G._header(fig, "Multi-axis vs mono-axis pairs: the three decision accuracies on the reliable cells",
                    f"rows = pair set, columns = definition; the {n_cells} (benchmark, language) cells reliable on both axes "
                    f"at 0.66 (median, reliable_tasks.py). multi-axis = every pair at the grid seed ({n_pairs['multi-axis']} "
                    f"decisions at the reference); mono-axis = those moving exactly one of L, depth, list, temperature, "
                    f"second language ({n_pairs['mono-axis']}). DA-size: final vs the {TARGET_SIZE} final; DA-ckpt: a "
                    f"checkpoint vs its own size's final; DA-goal: a checkpoint vs the {TARGET_SIZE} final. Pooled over "
                    f"decisions; gate and MIN_PAIRS as everywhere in rq02; dotted = τ {TAU:g} / {SAFE_DA}.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def table(out: pd.DataFrame) -> pd.DataFrame:
    """The numbers the plan quotes: level at the ends of each axis, per definition and pair set."""
    rows = []
    for panel in PANELS:
        for a in AXES:
            d = out[(out["panel"] == panel) & (out["axes"] == a)]
            if panel == "DA-size":
                pick = {f"{s} final": d[d["size"] == s]["da"] for s in ("175M", "1B")}
            else:
                last = d["chinchilla"].max()
                pick = {f"175M @0.5C": d[(d["size"] == "175M") & (d["chinchilla"] == 0.5)]["da"],
                        f"175M @{last:g}C": d[(d["size"] == "175M") & (d["chinchilla"] == last)]["da"],
                        f"1B @{last:g}C": d[(d["size"] == "1B") & (d["chinchilla"] == last)]["da"]}
            rows.append({"panel": panel, "axes": a,
                         **{k: round(float(v.iloc[0]), 3) if len(v) else np.nan for k, v in pick.items()},
                         "decisions @175M": int(d[d["size"] == "175M"]["n_comparable"].max())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose gate applies and whose folder receives the outputs")
    args = p.parse_args()
    out_dir = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    keep = load_reliable(out_dir, VARIANT)
    if keep is None:
        sys.exit(1)
    df = ladder_frame(POOL)
    fin = finals(df)
    attrs = design_axes(df)
    pairs = decision_sets(attrs)
    sizes = size_order(fin["size"].unique())
    print({a: len(pl) for a, pl in pairs.items()}, "family pairs")
    out = pooled(cells(df, fin, pairs, sizes), args.pool, set(keep["task"]))
    out.to_csv(out_dir / f"rq2_{VARIANT}_axes.csv", index=False)
    print(table(out).to_string(index=False))
    figure(out, out_dir / f"rq2_{VARIANT}_axes.png", len(keep))
