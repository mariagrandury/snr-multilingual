"""rq01 per benchmark and per language: how well each (task, L) series follows
the log-N fit.

    fit_r2_by_benchmark.png   R² of the log-N fit, language x L, one subplot per benchmark (BPB first)
    fit_r2_by_language.png    R² of the log-N fit, benchmark x L, one subplot per language

Reads `rq1_fits.csv` (one row per (task, L) fit on the deep, scheme-A,
seed-1904 cells).

    python analysis/rq01_scaling_predictability/panels.py --pool predictivity_all
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
from analysis.paths import SCALING_PREDICTABILITY  # noqa: E402

OUT_ROOT = SCALING_PREDICTABILITY
CANONICAL = "predictivity_all"
mpl.rcParams.update(S.RC)


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    fits = G.add_meta(pd.read_csv(out_dir / "rq1_fits.csv").drop(columns=["family"]))
    Ls = sorted(fits["L"].unique())
    fits["setting"] = "L" + fits["L"].astype(str)
    note = ("cell = R² of a straight line of the final score against log(parameters), over the rungs trained at that language "
            "setting (deep, scheme A, seed 1904); 1 = the score moves with size exactly as the line says")
    G.panel_grid(fits, out_dir / "fit_r2_by_benchmark.png", by="family", row="setting", row_order=[f"L{L}" for L in Ls],
                 col="language", value="r2", ncols=1, cell_w=0.3, counts=False, cbar="R² of score ~ log N", note=note,
                 xlabel="language", ylabel="language setting",
                 title="How predictably each benchmark scales: R² of the log-N fit")
    G.panel_grid(fits, out_dir / "fit_r2_by_language.png", by="language", row="family", ylabel="benchmark", ncols=6,
                 col="L", value="r2", col_order=Ls, col_label=lambda L: f"L{L}", cbar="R² of score ~ log N", note=note, csv=False,
                 xlabel="language setting", title="How predictably each language scales: R² of the log-N fit")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    fam = G.panel_order(fits["family"].unique())
    by = fits.groupby(["family", "L"])["r2"]
    lang = fits[fits["kind"] == "benchmark"].groupby("language")["r2"].median()
    tables = [G.matrix_ax(axes[0], by.median().unstack().reindex(index=fam, columns=Ls).rename(columns=lambda L: f"L{L}"),
                          "Median R² of the log-N fit", cnt=by.count().unstack().reindex(index=fam, columns=Ls), xlabel="language setting"),
              G.rank_ax(axes[1], fits[fits["family"] != "bpb"].groupby("family")["r2"].median(), "Benchmarks by median R²", xlabel="median R²"),
              G.rank_ax(axes[2], lang, "Languages by median R² over their benchmarks", xlabel="median R²")]
    G.save_highlights(fig, out_dir, "rq01 in one figure: what scales predictably with model size?",
                      "R² of final score ~ log(parameters) per (task, language setting); small number = fits behind the cell", tables)
    if pool != CANONICAL:
        return
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"The family medians above, without the aggregation (`{pool}` pool). Regenerate with `python analysis/rq01_scaling_predictability/panels.py --pool {pool}`. In every grid white is \"no value\" and grey \"filtered out by the gate\"; each figure's table sits next to it under the same name.",
        f"![rq01 in one figure]({stage}/{pool}/highlights.png)"]
        + [f"![{alt}]({stage}/{pool}/{name})" for alt, name in [('Fit R² per benchmark', 'fit_r2_by_benchmark.png'), ('Fit R² per language', 'fit_r2_by_language.png')]])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
