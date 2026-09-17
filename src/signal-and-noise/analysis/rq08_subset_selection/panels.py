"""rq08 as one grid: where does the best subset beat the random-subset null?

    gain_over_null.png   best-subset SNR minus the null's 95th percentile, task x size, one subplot per case

Reads `summary.csv`. The per-benchmark and per-language sweeps themselves are
`per_benchmark_plots/` and `global_mmlu_full_per_language_plots/`.

    python analysis/rq08_subset_selection/panels.py --pool predictivity
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
from analysis.paths import SUBSET_SELECTION  # noqa: E402

OUT_ROOT = SUBSET_SELECTION
CANONICAL = "predictivity"
mpl.rcParams.update(S.RC)


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    s = pd.read_csv(out_dir / "summary.csv").dropna(subset=["gain_over_null"])
    if s.empty:
        return
    G.panel_grid(s, out_dir / "gain_over_null.png", by="case", row="task", col="size", value="gain_over_null",
                 col_order=[b for b in bucket_order() if b in set(s["size"])], ncols=3, vmin=-2, vmax=2, center=0.0, cmap=S.DIV,
                 fmt="{:+.2f}", counts=False, first=(), cbar="best-subset SNR − null p95", xlabel="model size",
                 ylabel="benchmark", title="Subset selection against the random-subset null (> 0 beats it)",
                 note="cell = SNR of the best subset found minus the 95th percentile of the SNR of random subsets of the same size")
    sizes = [b for b in bucket_order() if b in set(s["size"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    by = s.assign(beats=s["gain_over_null"] > 0).groupby(["case", "size"])["beats"]
    tables = [G.matrix_ax(axes[0], by.mean().unstack().reindex(columns=sizes), "Share of benchmarks whose best subset beats the null",
                          cnt=by.count().unstack().reindex(columns=sizes), xlabel="model size"),
              G.rank_ax(axes[1], s.groupby("task")["gain_over_null"].median(), "Benchmarks by median gain over the null",
                        xlabel="best-subset SNR − null p95", ref=0.0)]
    G.save_highlights(fig, out_dir, "rq08 in one figure: does selecting a subset raise the SNR beyond chance?",
                      "gain = SNR of the best subset − 95th percentile of random subsets of the same size; small number = benchmarks", tables)
    if pool != CANONICAL:
        return
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"Every swept cell in one grid (`{pool}` pool). Regenerate with `python analysis/rq08_subset_selection/panels.py --pool {pool}`. In every grid white is \"no value\" and grey \"filtered out by the gate\"; each figure's table sits next to it under the same name.",
        f"![rq08 in one figure]({stage}/{pool}/highlights.png)"]
        + [f"![{alt}]({stage}/{pool}/{name})" for alt, name in [('Gain over the null', 'gain_over_null.png')]])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
