"""rq05 per benchmark and per language: the decision table and the
early-decision read, without pooling the benchmarks.

    intervention_da_by_benchmark.png   agreement with the reference, proxy size x L; one subplot per (intervention, benchmark), intervention by intervention, BPB first
    intervention_da_by_language.png    the same, one subplot per (intervention, language)
    intervention_da_by_benchmark_early.png   the two planned decisions, proxy size x share of the run (mean over L), per benchmark
    intervention_da_by_language_early.png    the same per language

Reads `intervention_da_by_benchmark.csv` and `intervention_da_by_language.csv`
(the per-item agreement `analyze.py` aggregates: per-language BPB on the
languages both levels train, and the benchmark tasks). Unlike the pooled
`benchmark` population of the decision table, these tables leave out the
language aggregates and the per-subject facets (`global_mmlu_full_<lang>_<subject>`),
which have no single language: about 60 % of the pooled benchmark items
remain, so a benchmark row here is not a slice of the pooled number. The
per-language table averages a language's BPB item with its benchmark items.

    python analysis/rq05_design_decisions/panels.py --pool predictivity_all
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
from analysis.autodoc import replace_block  # noqa: E402
from analysis.paths import DESIGN_DECISIONS  # noqa: E402
from analysis.rq05_design_decisions.analyze import CANONICAL, INTERVENTIONS  # noqa: E402
from analysis.rq05_design_decisions.early_decision import DECISIONS  # noqa: E402
from analysis.utils import LADDER_SIZES  # noqa: E402

OUT_ROOT = DESIGN_DECISIONS
mpl.rcParams.update(S.RC)


def _panels(t: pd.DataFrame, by: str, path: Path, *, keys: list, ncols: int, **kw) -> None:
    units = G.panel_order(t[by].unique())
    t = t.assign(panel=[f"{INTERVENTIONS[k][0]} — {u}" for k, u in zip(t["intervention"], t[by])])
    order = [f"{INTERVENTIONS[k][0]} — {u}" for k in keys for u in units]
    G.panel_grid(t, path, by="panel", value="decision_acc", order=order, ncols=ncols, counts=False, csv=False,
                 row="proxy_size", row_order=[s for s in LADDER_SIZES if s in set(t["proxy_size"])],
                 cbar="agreement with the reference's final decision", ylabel="proxy size", **kw)


def highlights(out_dir: Path, fin: pd.DataFrame, keys: list) -> None:
    """rq05 on one page, final checkpoints: per intervention, how the
    agreement grows with the proxy size (BPB, benchmarks) and which
    benchmarks carry it."""
    sizes = [s for s in LADDER_SIZES if s in set(fin["proxy_size"])]
    label = {k: INTERVENTIONS[k][0] for k in keys}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    tables = []
    for ax, (name, g) in zip(axes, (("bits per byte", fin[fin["family"] == "bpb"]), ("benchmarks", fin[fin["family"] != "bpb"]))):
        m = g.groupby(["intervention", "proxy_size"])["decision_acc"].mean().unstack().reindex(index=keys, columns=sizes)
        tables.append(G.matrix_ax(ax, m.rename(index=label), f"Agreement with the reference's decision, {name}", xlabel="proxy size"))
    m = fin.groupby(["family", "intervention"])["decision_acc"].mean().unstack().reindex(columns=keys)
    tables.append(G.matrix_ax(axes[2], m.reindex(G.panel_order(m.index)).rename(columns=label), "Per benchmark, mean over proxy sizes",
                              xlabel="intervention"))
    for lab in axes[2].get_xticklabels():
        lab.set_rotation(30); lab.set_ha("right")
    G.save_highlights(fig, out_dir, "rq05 in one figure: would a small proxy have made the reference's design decision?",
                      "cell = share of items on which the proxy prefers the same level of the intervention as the reference, final "
                      "checkpoints, mean over language settings (and over benchmarks in the first two panels)", tables)


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    note = ("cell = share of the unit's items (a benchmark task or a language's BPB) on which the proxy prefers the same level "
            "of the intervention as the reference at its final checkpoint")
    for by, name, ncols in (("family", "benchmark", 6), ("language", "language", 10)):
        t = pd.read_csv(out_dir / f"intervention_da_by_{name}.csv")
        keys = [k for k in INTERVENTIONS if k in set(t["intervention"])]
        fin = t[t["frac"] == 1.0]
        _panels(fin, by, out_dir / f"intervention_da_by_{name}.png", keys=keys, ncols=ncols, col="L",
                col_order=sorted(fin["L"].unique()), col_label=lambda L: f"L{L}", xlabel="language setting",
                title=f"Does the proxy prefer the level the reference prefers? Final checkpoints, per {name}", note=note)
        early = t[t["intervention"].isin(DECISIONS)]
        _panels(early, by, out_dir / f"intervention_da_by_{name}_early.png", keys=[k for k in DECISIONS if k in keys],
                ncols=ncols, col="frac", col_order=sorted(early["frac"].unique()), col_label=G.chinchilla,
                xlabel="proxy's training tokens (C = Chinchilla-optimal; 5C = the full run)", note=note,
                title=f"How small and how early, per {name} (mean over language settings)")
        if by == "family":
            highlights(out_dir, fin, keys)
    if pool != CANONICAL:
        return
    rel = f"{stage}/{pool}"
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"The decision table and the early read above, without pooling the benchmarks (`{pool}` pool). Language "
        "aggregates and per-subject facets are left out here (about 60 % of the pooled benchmark items remain), "
        "and a language's subplot averages its BPB item with its benchmark items. Regenerate with "
        f"`python analysis/rq05_design_decisions/panels.py --pool {pool}`. White cells have no value; each figure's table "
        "sits next to it under the same name (`intervention_da_by_<unit>.csv`).",
        f"![rq05 in one figure]({rel}/highlights.png)",
        f"![Decisions per benchmark]({rel}/intervention_da_by_benchmark.png)",
        f"![Early and small per benchmark]({rel}/intervention_da_by_benchmark_early.png)",
        f"![Decisions per language]({rel}/intervention_da_by_language.png)",
        f"![Early and small per language]({rel}/intervention_da_by_language_early.png)"])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
