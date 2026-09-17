"""rq06 per language: the leave-one-language-out prediction error.

    transfer_error_by_L.png   |relative error| of the transferred prediction, language x rungs used, one subplot per L

Reads `rq5_transfer.csv`. There is no per-benchmark view: the transfer test
is on per-language bits per byte only.

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
from analysis.paths import LANGUAGE_TRANSFER  # noqa: E402

OUT_ROOT = LANGUAGE_TRANSFER
CANONICAL = "predictivity_all"
mpl.rcParams.update(S.RC)


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
    if pool != CANONICAL:
        return
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"The summary above, per language (`{pool}` pool). Regenerate with `python analysis/rq06_language_transfer/panels.py --pool {pool}`. In every grid white is \"no value\" and grey \"filtered out by the gate\"; each figure's table sits next to it under the same name.",
        f"![rq06 in one figure]({stage}/{pool}/highlights.png)"]
        + [f"![{alt}]({stage}/{pool}/{name})" for alt, name in [('Transfer error per language and L', 'transfer_error_by_L.png')]])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
