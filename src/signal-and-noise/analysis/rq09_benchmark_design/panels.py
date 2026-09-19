"""rq09 per language: the SNR behind every family median.

    snr_family_by_language.png   log10 SNR at the reference size, benchmark x language

Reads `per_task_snr.csv` (the per-language tasks the family medians are taken over).

    python analysis/rq09_benchmark_design/panels.py --pool predictivity
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
from analysis.paths import BENCHMARK_DESIGN  # noqa: E402

OUT_ROOT = BENCHMARK_DESIGN
CANONICAL = "predictivity"
mpl.rcParams.update(S.RC)


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    t = pd.read_csv(out_dir / "per_task_snr.csv")
    snr_col = next(c for c in t.columns if c.startswith("snr_"))
    t["log_snr"] = np.log10(t[snr_col].where(t[snr_col] > 0))
    t = t[~t["language"].isin(["??", "multi"])].assign(view=snr_col, size=snr_col.rsplit("_", 1)[1])
    t = G.mark_gated(t, pool, "size", "log_snr")
    G.panel_grid(t, out_dir / "snr_family_by_language.png", by="view", row="family", col="language", value="log_snr",
                 row_order=G.panel_order(t["family"].unique()), ncols=1, vmin=-1, vmax=2, first=(),
                 cbar="log10 SNR", xlabel="language", ylabel="benchmark",
                 title="SNR of every (benchmark, language) task behind the family medians",
                 note=f"cell = log10 of `{snr_col}` of the (benchmark, language) task; a task at chance at that size has no SNR")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    tables = [G.rank_ax(axes[0], t.groupby("family")["log_snr"].median(), "Benchmarks by median log10 SNR", xlabel="log10 SNR", ref=0.0),
              G.rank_ax(axes[1], t.groupby("language")["log_snr"].median(), "Languages by median log10 SNR", xlabel="log10 SNR", ref=0.0),
              G.rank_ax(axes[2], t.groupby("curation_category")["log_snr"].median(), "How the benchmark was built", xlabel="log10 SNR", ref=0.0)]
    G.save_highlights(fig, out_dir, "rq09 in one figure: which benchmark designs give signal?",
                      f"log10 of `{snr_col}` per (benchmark, language) task, medians; 0 = signal equals noise", tables)
    if pool != CANONICAL:
        return
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"The family medians above, per language (`{pool}` pool). Regenerate with `python analysis/rq09_benchmark_design/panels.py --pool {pool}`. In every grid white is \"no value\" and grey \"filtered out by the gate\"; each figure's table sits next to it under the same name.",
        f"![rq09 in one figure]({stage}/{pool}/highlights.png)"]
        + [f"![{alt}]({stage}/{pool}/{name})" for alt, name in [('SNR per benchmark and language', 'snr_family_by_language.png')]])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
