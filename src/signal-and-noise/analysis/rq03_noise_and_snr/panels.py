"""rq03 per benchmark and per language: the SNR of every task and the size of
a design effect against seed noise.

    snr_by_benchmark.png                 log10 SNR (relative std), language x size, one subplot per benchmark
    snr_by_language.png                  log10 SNR (relative std), benchmark x size, one subplot per language
    effect_over_seed_by_benchmark.png    log10 |depth effect| / seed noise, size x L, per benchmark
    effect_over_seed_by_language.png     the same per language

SNR reads `snr_variants_per_task.csv` of the pool (the paper's definition,
`rel_std`; cells the above-random gate puts at chance are grey, rule 1). The
effect reads `effect_vs_noise.csv` of `predictivity_all`, the pool with seed
replicates, gated cells included and drawn grey: 0 means the depth effect
equals the seed noise (zero effects have no log and are left out). The effect
figures are written next to their source, in `predictivity_all`, whatever
`--pool` is. The size axis is EVAL_SIZES (175M–1.7B, rule 10), never a column
the table happens to carry.

    python analysis/rq03_noise_and_snr/panels.py --pool predictivity
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
from analysis.autodoc import replace_block  # noqa: E402
from analysis.paths import NOISE_AND_SNR  # noqa: E402
from analysis.utils import EVAL_SIZES  # noqa: E402

OUT_ROOT = NOISE_AND_SNR
CANONICAL = "predictivity"
NOISE_POOL = "predictivity_all"
VARIANT = "rel_std"
mpl.rcParams.update(S.RC)


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    t = pd.read_csv(out_dir / "snr_variants_per_task.csv")
    sizes = [b for b in EVAL_SIZES if f"snr_{VARIANT}_{b}" in t.columns]   # rule 10: never 90M
    long = t.melt(id_vars="task", value_vars=[f"snr_{VARIANT}_{b}" for b in sizes], var_name="size", value_name="snr")
    long["size"] = long["size"].str.replace(f"snr_{VARIANT}_", "", regex=False)
    long["log_snr"] = np.log10(long["snr"].where(long["snr"] > 0))
    long = G.mark_gated(G.add_meta(long), pool, "size", "log_snr")
    long = long[long["log_snr"].notna() | long["gated"]]
    kw = dict(value="log_snr", vmin=-1.0, vmax=2.0, fmt="{:.1f}", cbar=f"log10 SNR ({VARIANT})",
              note=f"cell = log10 of SNR ({VARIANT}): spread of the final scores across the size's design variants over the "
                   "checkpoint-to-checkpoint noise over the 80/85/90/95/100 % checkpoints; 0 = signal equals noise, 1 = ten times the noise")
    G.panel_grid(long, out_dir / "snr_by_benchmark.png", by="family", row="size", row_order=sizes, col="language",
                 ncols=1, cell_w=0.3, counts=False, xlabel="language", ylabel="model size",
                 title="Signal-to-noise ratio per benchmark", **kw)
    G.panel_grid(long, out_dir / "snr_by_language.png", by="language", row="family", ylabel="benchmark", ncols=6,
                 col="size", col_order=sizes, xlabel="model size", title="Signal-to-noise ratio per language", csv=False, **kw)
    have = long.dropna(subset=["log_snr"])
    fam = G.panel_order(have["family"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    by = have.groupby(["family", "size"])["log_snr"]
    top = have[have["size"] == sizes[-1]]
    tables = [G.matrix_ax(axes[0], by.median().unstack().reindex(index=fam, columns=sizes), "Median log10 SNR", vmin=-1, vmax=2,
                          fmt="{:.1f}", cnt=by.count().unstack().reindex(index=fam, columns=sizes), xlabel="model size"),
              G.rank_ax(axes[1], top.groupby("family")["log_snr"].median(), f"Benchmarks by median log10 SNR at {sizes[-1]}",
                        xlabel="log10 SNR", ref=0.0),
              G.rank_ax(axes[2], top[top["family"] != "bpb"].groupby("language")["log_snr"].median(),
                        f"Languages by median benchmark log10 SNR at {sizes[-1]}", xlabel="log10 SNR", ref=0.0)]
    G.save_highlights(fig, out_dir, "rq03 in one figure: where is the signal above the noise?",
                      f"SNR ({VARIANT}) = spread across design variants / checkpoint noise; 0 on the log scale = signal equals noise; "
                      "cells the above-random gate filtered out are left out; small number = tasks behind the cell", tables)

    src = OUT_ROOT / stage / NOISE_POOL / "effect_vs_noise.csv"
    if src.is_file():
        e = pd.read_csv(src, usecols=["size", "L", "task", "effect_arch_over_seed", "gated"])
        e["log_ratio"] = np.log10(e["effect_arch_over_seed"].where(e["effect_arch_over_seed"] > 0))
        e = G.add_meta(e[e["log_ratio"].notna() | e["gated"]])      # the gated rows are the grey cells
        G.benchmark_and_language_panels(
            e, OUT_ROOT / stage / NOISE_POOL, "effect_over_seed", row="size", col="L", value="log_ratio",
            row_order=[b for b in EVAL_SIZES if b in set(e["size"])], col_order=sorted(e["L"].unique()),
            col_label=lambda L: f"L{L}", vmin=-1.0, vmax=1.0, center=0.0, cmap=S.DIV, fmt="{:+.1f}",
            cbar="log10 |depth effect| / seed noise", xlabel="language setting", ylabel="model size",
            title="Depth effect against seed noise (0 = the effect equals the noise)",
            note="cell = log10 of |final score of the deep model − the shallow one| over the sample std across replicate "
                 "seeds of the same cell, mean over the subplot's tasks; grey = at chance at that size")
    if pool != CANONICAL:
        return
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"Regenerate with `python analysis/rq03_noise_and_snr/panels.py --pool {pool}`. In every grid white is \"no value\" and grey \"filtered out by the gate\" (at chance at that size, rule 1); each figure's table sits next to it under the same name; sizes are 175M–1.7B (rule 10). SNR noise is the std over the 80/85/90/95/100 % checkpoints (rule 4).",
        f"![rq03 in one figure]({stage}/{pool}/highlights.png)",
        f"![SNR per benchmark]({stage}/{pool}/snr_by_benchmark.png)",
        f"![SNR per language]({stage}/{pool}/snr_by_language.png)",
        f"![Depth effect over seed noise per benchmark]({stage}/{NOISE_POOL}/effect_over_seed_by_benchmark.png)",
        f"![Depth effect over seed noise per language]({stage}/{NOISE_POOL}/effect_over_seed_by_language.png)"])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
