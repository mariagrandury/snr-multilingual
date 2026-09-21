"""rq06 per language: the leave-one-language-out prediction error.

    transfer_error_by_L.png   |relative error| of the transferred prediction, language x rungs used, one subplot per L
    transfer_da_lines.png     the list decision (scheme A vs B) read on per-language BPB, DA-size (x = proxy size) and
                              DA-ckpt (x = the reference's checkpoint), one line per language group: both levels'
                              lists train the language, only one does, neither does but one trains its script, or
                              neither trains even the script (mean over L); the last two are the transfer test
    transfer_da_by_L.png      the same agreement per language count, one panel per intervention, one line per group
                              (DA-size, mean over the proxy sizes)

Reads `rq5_transfer.csv`. There is no per-benchmark view: the transfer test
is on per-language bits per byte only.
The decision lines read rq05's `intervention_da_by_group_ckpt10.csv` (every
evaluated checkpoint).

    python analysis/rq06_language_transfer/panels.py --pool predictivity_all
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

from evals.scripts.utils.configs import bucket_order, load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import replace_block  # noqa: E402
from analysis.paths import DESIGN_DECISIONS, LANGUAGE_TRANSFER  # noqa: E402
from analysis.rq05_design_decisions.analyze import INTERVENTIONS, LANGUAGE_GROUPS  # noqa: E402
from analysis.rq05_design_decisions.panels import da_lines  # noqa: E402

OUT_ROOT = LANGUAGE_TRANSFER
CANONICAL = "predictivity_all"
GROUP_COLOUR = dict(zip(LANGUAGE_GROUPS, [S.RAMP[3], S.MUTED, S.RAMP[1], S.SERIES[1]]))
mpl.rcParams.update(S.RC)


def decision_lines(out_dir: Path, stage: str) -> None:
    src = DESIGN_DECISIONS / stage / "predictivity_all" / "intervention_da_by_group_ckpt10.csv"
    if not src.is_file():
        return
    g = pd.read_csv(src).assign(population="bpb")
    if g.empty:
        print("!!! RULE 2: rq05's per-group table is empty (its pool holds trained languages only); the transfer decision "
              "lines need the untrained groups, which only this folder's analyze.py may read — skipped")
        return
    da_lines(g[g["intervention"] == "scheme"], out_dir, name="transfer_da_lines", series="group", colours=GROUP_COLOUR,
             populations=(("bpb", "-", "per-language BPB"),),
             title="Does a proxy read the list decision (scheme A vs B) for languages it did not train?",
             note="DA = share of a group's languages on which the proxy prefers the list the reference prefers at its final "
                  "checkpoint (per-language BPB), mean over L; group = what the two levels' lists do with the language: both train "
                  "it, only one does (a decision the language's inclusion makes by itself), neither does but a list trains its "
                  "script, or neither trains even the script; dotted line = 0.75")
    fin = g[g["frac"] == 1.0].groupby(["intervention", "label", "L", "group"])["decision_acc"].mean().reset_index()
    keys = [k for k in INTERVENTIONS if k in set(fin["intervention"])]
    Ls = sorted(fin["L"].unique())
    fig, axes = plt.subplots(1, len(keys), figsize=(3.4 * len(keys) + 1.5, 3.8), sharey=True, squeeze=False)
    tables = []
    for ax, k in zip(axes[0], keys):
        sub = fin[fin["intervention"] == k]
        for grp, c in GROUP_COLOUR.items():
            r = sub[sub["group"] == grp].set_index("L")["decision_acc"].reindex(Ls)
            ax.plot(range(len(Ls)), r, color=c, marker="o", ms=3.5, lw=1.3, label=grp)
        ax.set_xticks(range(len(Ls))); ax.set_xticklabels([f"L{L}" for L in Ls])
        ax.axhline(0.75, color=S.MUTED, lw=.8, ls=":"); ax.set_ylim(0.0, 1.02)
        ax.set_title(INTERVENTIONS[k][0], loc="left", fontsize=8.5); ax.set_xlabel("language count"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
        tables.append(sub.rename(columns={"group": "row", "L": "col", "decision_acc": "value"}).assign(panel=INTERVENTIONS[k][0])
                      [["panel", "row", "col", "value"]])
    axes[0, 0].set_ylabel("DA-size (mean over proxy sizes)")
    axes[0, -1].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    G.save_highlights(fig, out_dir, "Does the decision transfer to untrained languages at every language count?",
                      "DA-size = share of the group's languages on which the proxy's final ranking of the two levels matches the "
                      "reference's (the largest size trained at both levels at that L, so the reference changes along x), mean over "
                      "the proxy sizes with a value (a proxy that flips and one that agrees average to 0.5; the per-size values are "
                      "in the CSV of transfer_da_lines); a group is empty where the lists leave it no language — \"trained by one "
                      "level\" exists only where the two levels' lists differ",
                      tables, name="transfer_da_by_L")


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    t = G.add_meta(pd.read_csv(out_dir / "rq5_transfer.csv"))
    t["abs_err"] = t["err_transfer"].abs()
    t["setting"] = "L" + t["L"].astype(str)
    t["row"] = t["language"] + np.where(t["trained"], "", " (untrained)")
    G.panel_grid(t, out_dir / "transfer_error_by_L.png", by="setting", row="row", col="k", value="abs_err",
                 col_order=sorted(t["k"].unique()), ncols=6, vmin=0, vmax=0.25, fmt="{:.2f}", counts=False,
                 order=[f"L{L}" for L in sorted(t["L"].unique())], cbar="|relative error| of the predicted reference BPB",
                 xlabel="smallest rungs used (k)", ylabel="language",
                 title="Leave-one-language-out: error of the transferred scaling law, per language",
                 note="cell = |predicted − observed| / observed bits per byte at the reference size, the prediction made from the "
                      "language's k smallest rungs and the exponent pooled over the other languages")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    Ls, ks = sorted(t["L"].unique()), sorted(t["k"].unique())
    tables = []
    for ax, (flag, name) in zip(axes, ((True, "trained languages"), (False, "untrained languages"))):
        m = t[t["trained"] == flag].groupby(["L", "k"])["abs_err"].median().unstack().reindex(index=Ls, columns=ks)
        tables.append(G.matrix_ax(ax, m.rename(index=lambda L: f"L{L}"), f"Median |relative error|, {name}", vmin=0, vmax=0.25,
                                  xlabel="smallest rungs used (k)", ylabel="language setting"))
    tables.append(G.rank_ax(axes[2], t[t["k"] == ks[-1]].groupby("row")["abs_err"].median(), ascending=True,
                            title=f"Languages by |relative error| at k = {ks[-1]}, smallest on top", xlabel="|relative error|",
                            fmt="{:.3f}"))
    G.save_highlights(fig, out_dir, "rq06 in one figure: does a scaling law transfer to a language it has not seen?",
                      "error of the reference-size BPB predicted from the k smallest rungs with the exponent of the other languages", tables)
    decision_lines(out_dir, stage)
    if pool != CANONICAL:
        return
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"The summary above, per language (`{pool}` pool). Regenerate with `python analysis/rq06_language_transfer/panels.py --pool {pool}`. In every grid white is \"no value\" and grey \"filtered out by the gate\"; each figure's table sits next to it under the same name.",
        f"![rq06 in one figure]({stage}/{pool}/highlights.png)"]
        + [f"![{alt}]({stage}/{pool}/{name})" for alt, name in [('Transfer error per language and L', 'transfer_error_by_L.png'), ('The list decision on untrained languages, by proxy size and checkpoint', 'transfer_da_lines.png'), ('The decisions on untrained languages, by language count', 'transfer_da_by_L.png')]])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
