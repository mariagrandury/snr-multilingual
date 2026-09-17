"""rq04 per benchmark and per language: which SNR definition, and which cheap
statistic, tracks decision accuracy where.

    snr_definition_by_language.png   Pearson r of log10 SNR with DA, SNR definition x language (DA-size, DA-ckpt)
    surrogates_by_benchmark.png      Spearman rho of each statistic with DA-size, statistic x proxy size, per benchmark
    surrogates_by_language.png       the same per language

The first reads `snr_variant_ranking.csv` (the per-language pooled r that
`analyze_snr_variants.py` writes). The second reruns `analyze.surrogates` on
one benchmark's, or one language's, tasks; a subplot needs MIN_TASKS tasks
at a proxy size, so most languages stay blank.

    python analysis/rq04_surrogates/panels.py --pool predictivity
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
from analysis.paths import GATE_AND_CURVES, NOISE_AND_SNR, SCALING_PREDICTABILITY, SURROGATES  # noqa: E402
from analysis.rq04_surrogates.analyze import CANONICAL, FITS_POOL, KINDS, MIN_TASKS, surrogates  # noqa: E402
from analysis.utils import SMALL_SIZES  # noqa: E402

OUT_ROOT = SURROGATES
MIN_LANG_TASKS = 5
mpl.rcParams.update(S.RC)


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / pool
    rk = pd.read_csv(out_dir / "snr_variant_ranking.csv")
    # a language's r is pooled over its (task, size pair) points; `all` has one r per size pair, read as their mean
    rk = rk[((rk["column"] == "pooled") | ((rk["scope"] == "all") & (rk["column"] == "mean")))
            & rk["da_def"].isin(["da_size", "da_ckpt/da_ckpt_mix"])].copy()
    rk["da"] = rk["da_def"].map({"da_size": "DA-size", "da_ckpt/da_ckpt_mix": "DA-ckpt (all sizes)"})
    order = rk.groupby("variant")["pearson_r"].mean().sort_values(ascending=False).index.tolist()
    # a per-language r over fewer than MIN_LANG_TASKS tasks is ±1 by construction (CLAUDE.md #6)
    n_tasks = G.add_meta(pd.read_csv(NOISE_AND_SNR / stage / pool / "snr_variants_per_task.csv",
                                      usecols=["task"]))["language"].value_counts()
    keep = ["all"] + sorted(l for l, n in n_tasks.items() if n >= MIN_LANG_TASKS)
    rk = rk[rk["scope"].isin(keep)]
    G.panel_grid(rk, out_dir / "snr_definition_by_language.png", by="da", row="variant", col="scope", value="pearson_r",
                 row_order=order, col_order=keep, ncols=1, vmin=-1, vmax=1, center=0.0, cmap=S.DIV, fmt="{:+.1f}",
                 counts=False, cell_w=0.3,
                 order=["DA-size", "DA-ckpt (all sizes)"], cbar="Pearson r of log10 SNR with DA",
                 xlabel=f"language (≥ {MIN_LANG_TASKS} tasks; `all` = pooled)", ylabel="SNR definition",
                 title="Which SNR definition tracks decision accuracy, per language",
                 note="cell = Pearson r, over the language's tasks, between log10 SNR under that definition and the task's decision "
                      "accuracy (pooled over proxy sizes); `all` = every task, mean of the r over the size pairs")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    tables = []
    for ax, da in zip(axes, ["DA-size", "DA-ckpt (all sizes)"]):
        r = rk[(rk["da"] == da) & (rk["scope"] == "all")].set_index("variant")["pearson_r"]
        tables.append(G.rank_ax(ax, r, f"SNR definitions by r with {da}, all tasks", k=6, xlabel="Pearson r", ref=0.0))
    per_lang = rk[(rk["da"] == "DA-size") & (rk["scope"] != "all")].dropna(subset=["pearson_r"])
    wins = per_lang.loc[per_lang.groupby("scope")["pearson_r"].idxmax(), "variant"].value_counts().astype(float)
    tables.append(G.rank_ax(axes[2], wins, f"Languages in which a definition is the best (DA-size, {per_lang['scope'].nunique()} languages)",
                            k=len(wins), xlabel="languages", fmt="{:.0f}"))
    G.save_highlights(fig, out_dir, "rq04 in one figure: which SNR definition predicts decision accuracy?",
                      f"Pearson r between log10 SNR and DA over tasks; a language counts with ≥ {MIN_LANG_TASKS} tasks", tables)

    v = pd.read_csv(NOISE_AND_SNR / stage / pool / "snr_variants_per_task.csv", index_col=0).rename_axis("task").reset_index()
    scores = pd.read_csv(GATE_AND_CURVES / stage / pool / "above_random_scores.csv")
    fits_path = SCALING_PREDICTABILITY / stage / FITS_POOL / "rq1_fits.csv"
    fits_r2 = pd.read_csv(fits_path).groupby("task")["r2"].median() if fits_path.is_file() else pd.Series(dtype=float)
    v = G.add_meta(v)
    both = []
    for by, ncols in (("family", 4), ("language", 6)):
        cells = []
        for key, g in v.groupby(by):
            t = surrogates(g.reset_index(drop=True), scores, fits_r2)
            cells.append(t.assign(**{by: key}))
        cells = pd.concat(cells) if cells else pd.DataFrame()
        if cells.empty:
            continue
        both.append(cells.rename(columns={by: "key"}).assign(unit="benchmark" if by == "family" else "language"))
        metrics = cells.groupby("metric")["rho"].mean().sort_values(ascending=False).index.tolist()
        G.panel_grid(cells, out_dir / f"surrogates_by_{'benchmark' if by == 'family' else 'language'}.png", by=by,
                     row="metric", col="proxy", value="rho", row_order=metrics, csv=False,
                     col_order=[s for s in SMALL_SIZES if s in set(cells["proxy"])], ncols=ncols, vmin=-1, vmax=1,
                     center=0.0, cmap=S.DIV, fmt="{:+.2f}", counts=False, cbar="Spearman ρ with DA-size", xlabel="proxy size",
                     note="cell = Spearman ρ, over the subplot's tasks, between the statistic read at the proxy size and the "
                          f"task's DA-size (needs {MIN_TASKS} tasks)",
                     title="Which statistic predicts DA-size, per " + ("benchmark" if by == "family" else "language"))
    if both:                        # the two figures draw different cells: one table, `unit` says whose
        pd.concat(both).to_csv(out_dir / "surrogates.csv", index=False)
    if pool != CANONICAL:
        return
    body = "\n\n".join([
        "## Per benchmark and per language",
        f"The rankings above, without the aggregation (`{pool}` pool); a surrogate subplot needs 8 tasks at a proxy size. Regenerate with `python analysis/rq04_surrogates/panels.py --pool {pool}`. In every grid white is \"no value\" and grey \"filtered out by the gate\"; each figure's table sits next to it under the same name.",
        f"![rq04 in one figure]({stage}/{pool}/highlights.png)"]
        + [f"![{alt}]({stage}/{pool}/{name})" for alt, name in [('SNR definition per language', 'snr_definition_by_language.png'), ('Surrogates per benchmark', 'surrogates_by_benchmark.png'), ('Surrogates per language', 'surrogates_by_language.png')]])
    replace_block(OUT_ROOT / "README.md", "panels", body, f"panels.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL)
    main(p.parse_args().pool)
