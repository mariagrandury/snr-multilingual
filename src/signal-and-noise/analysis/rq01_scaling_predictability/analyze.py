"""RQ1 (paper RQ1) — What scales predictably?

Per (task, L), the final-checkpoint score of the deep, scheme-A, seed-1904
ladder against log10 N over the rungs from 175M up: the R² of the log-linear
fit and the Spearman ρ with size, then medians per benchmark family. A family
whose scores do not move with size has no trend to extrapolate, whatever its
provenance. The above-random gate (analysis/RULES.md, rule 1) is applied per
(task, size): a rung enters a task's fit only where rq00's mask is not 0
(`grids.mark_gated`, the canonical pool's mask), a fit needs MIN_RUNGS such
rungs, and a (task, L) the gate leaves short is kept in the table with NaN
statistics and `gated` set (grey in panels.py). The loader already keeps only
parent tasks (rule 6) and trained languages (rule 2). The loss scaling fit is
the ladder health check's power law on the seed-1904 cell of every (L, arch,
scheme); `scaling_law_error.py` next to this script asks how well such a fit
on the proxy rungs predicts the reference rung's per-language BPB.

    rq1_scaling.png/.pdf/.csv  above-chance curves at L=30 (ungated, so a task at chance is visible at 0),
                               R²/ρ per family over the gated fits; the CSV holds both panels' plotted values
    rq1_fits.csv           one row per (task, L): n_rungs (fitted), gated_rungs, gated, the fit;  rq1_families.csv  the family medians
    scaling_fit.png/.csv   final loss vs N per L, one power-law fit per (arch, scheme), the rung count in the legend

    python analysis/rq01_scaling_predictability/analyze.py --pool predictivity_all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from pretrain.ladder_report import _fit  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import SCALING_PREDICTABILITY  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import task_n_options  # noqa: E402
from analysis.utils import (  # noqa: E402
    GRID_SEED, NON_EMB, benchmark_family, finals, ladder_frame, size_order)

OUT_ROOT = SCALING_PREDICTABILITY
CANONICAL = "predictivity_all"
MIN_RUNGS = 3           # above-chance rungs a (task, L) fit needs
MIN_FAMILY_FITS = 3     # fits a family needs for a median
EXAMPLE_L = 30
mpl.rcParams.update(S.RC)


def _grid(fin: pd.DataFrame) -> pd.DataFrame:
    return fin[(fin["seed"] == GRID_SEED) & (fin["arch"] == "deep") & (fin["scheme"] == "A")]


def _line_style(size, arch, scheme) -> dict:
    return dict(color=S.SIZE_COLOR.get(size, S.MUTED), lw=S.ARCH_WIDTH.get(arch, 1.0),
                ls=S.SCHEME_DASH.get(scheme, "-"))


# --- the paper's RQ1: log-N fits ------------------------------------------------

def fit_table(fin: pd.DataFrame, pool: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (task, L) of the grid: the log-N fit over the rungs where
    the task is above chance (rule 1, `grids.mark_gated`); `n_rungs` counts
    them, `gated_rungs` the rungs the gate removed, and `gated` marks a series
    the gate left with fewer than MIN_RUNGS (its statistics are NaN)."""
    g0 = G.mark_gated(_grid(fin), pool, "size", "primary_score")
    rows = []
    for (task, L), g_all in g0.groupby(["task", "L"]):
        g_all = g_all.assign(N=g_all["size"].map(NON_EMB)).dropna(subset=["N"]).sort_values("N")
        g = g_all[~g_all["gated"]]
        kind = g_all["kind"].iloc[0]
        row = {"task": task, "L": int(L), "family": benchmark_family(task), "kind": kind,
               "n_rungs": len(g), "gated_rungs": int(g_all["gated"].sum()),
               "gated": len(g) < MIN_RUNGS <= len(g_all), "slope": np.nan, "r2": np.nan, "rho": np.nan,
               "n_options": task_n_options(task) if kind == "benchmark" else np.nan}
        if len(g) >= MIN_RUNGS:
            x, y = np.log10(g["N"].to_numpy()), g["primary_score"].to_numpy(float)
            b, a = np.polyfit(x, y, 1)
            ss_res = float(((y - (b * x + a)) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            row.update(slope=b, r2=1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                       rho=spearmanr(x, y).statistic if len(set(y)) > 1 else np.nan)
        rows.append(row)
    fits = pd.DataFrame(rows)
    fits = fits[fits["task"] != "bpb_macro"]
    fam = fits.groupby("family").agg(
        r2=("r2", "median"), rho=("rho", "median"), n=("r2", "count"),
        n_options=("n_options", lambda s: s.mode().iloc[0] if s.notna().any() else np.nan))
    fam = fam[fam["n"] >= MIN_FAMILY_FITS].sort_values("r2")
    return fits, fam


def plot_rq1(fin: pd.DataFrame, fits: pd.DataFrame, fam: pd.DataFrame, out_dir: Path) -> None:
    g0 = _grid(fin)
    sizes = size_order(g0["size"].unique())
    rungs = g0.groupby("L")["size"].nunique()
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) score above chance against size, family medians at one L (ungated: a task at chance sits at 0)
    ex = g0[(g0["L"] == EXAMPLE_L) & (g0["kind"] == "benchmark")].copy()
    ex["family"] = ex["task"].map(benchmark_family)
    ex["chance"] = ex["task"].map(task_n_options)
    ex = ex.dropna(subset=["chance"])
    ex["above"] = ex["primary_score"] - 1 / ex["chance"]
    show = [("multiblimp", S.RAMP[3]), ("hellaswag", S.RAMP[1]), ("xnli", S.SERIES[2]),
            ("belebele", S.SERIES[1]), ("global_mmlu_full", S.MUTED)]
    for famname, col in show:
        g = ex[ex["family"] == famname]
        if g.empty:
            continue
        for _task, t in g.groupby("task"):
            t = t.assign(N=t["size"].map(NON_EMB)).sort_values("N")
            a0.plot(t["N"], t["above"], color=col, lw=.5, alpha=.25)
        med = g.groupby("size")["above"].median()
        order = size_order(med.index)
        a0.plot([NON_EMB[s] for s in order], med[order], marker="o", ms=4, lw=2, color=col, label=famname)
    a0.axhline(0, color=S.INK, lw=.8, ls="--")
    a0.set_xscale("log")
    a0.set_xticks([NON_EMB[s] for s in sizes]); a0.set_xticklabels(sizes)
    a0.set_xlabel("non-embedding parameters"); a0.set_ylabel("accuracy above chance")
    a0.set_title(f"(a) benchmark families at L = {EXAMPLE_L}", loc="left")
    a0.legend(frameon=False, loc="upper left"); a0.grid(color=S.GRID, lw=.6); a0.set_axisbelow(True)
    S.clean(a0)
    # (b) R² and ρ per family
    y = np.arange(len(fam))
    a1.hlines(y, 0, fam["r2"], color=S.GRID, lw=1.2, zorder=1)
    a1.scatter(fam["r2"], y, s=42, color=S.RAMP[2], zorder=3, label="R² of the log-N fit")
    a1.scatter(fam["rho"], y, s=42, marker="D", color=S.SERIES[1], zorder=3, label="Spearman ρ with N")
    a1.axvline(0, color=S.MUTED, lw=.8)
    labels = [f"{f} (n={int(r['n'])}" + ("" if not np.isfinite(r["n_options"]) else f", {int(r['n_options'])}-way") + ")"
              for f, r in fam.iterrows()]
    a1.set_yticks(y); a1.set_yticklabels(labels, fontsize=7.5)
    a1.set_xlim(-1.05, 1.05)
    a1.set_xlabel(f"median over the family's (task, L) fits, {sizes[0]} to {sizes[-1]}")
    a1.set_title("(b) how predictably each family scales", loc="left")
    a1.legend(frameon=False, loc="lower left"); a1.grid(axis="x", color=S.GRID, lw=.6); a1.set_axisbelow(True)
    S.clean(a1); a1.tick_params(length=0)
    fitted = fits.dropna(subset=["r2"])
    top = G._header(fig, "What scales predictably with model size?",
                    f"(a) final accuracy minus 1/options per task at L = {EXAMPLE_L}, thin = tasks, thick = the family's median per size, "
                    f"every rung (no gate, so a task at chance reads 0); (b) per family the median R² and Spearman ρ of score ~ log N "
                    f"over its (task, L) fits ({len(fitted)} fits, {fitted['task'].nunique()} tasks), each fit on the rungs where the "
                    f"task is above chance (rule 1 gate; ≥ {MIN_RUNGS} rungs); rungs per L: "
                    + ", ".join(f"L{L} {n}" for L, n in rungs.items()) + "; deep, scheme A, seed 1904, parent tasks, trained languages")
    fig.tight_layout(rect=(0, 0, 1, top)); fig.subplots_adjust(wspace=.55)
    S.save_figure(fig, out_dir, "rq1_scaling")
    med_a = ex.groupby(["family", "size"])["above"].median().rename("value").reset_index().rename(columns={"size": "col"}).assign(panel="a")
    med_b = fam[["r2", "rho", "n", "n_options"]].stack().rename("value").rename_axis(["family", "col"]).reset_index().assign(panel="b")
    pd.concat([med_a, med_b])[["panel", "family", "col", "value"]].to_csv(out_dir / "rq1_scaling.csv", index=False)


# --- the loss scaling fit, from the loader's frame -------------------------------

def loss_scaling(fin: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Final training loss of the seed-1904 cells against N per L, one power
    law per (arch, scheme) over every rung present (≥ MIN_RUNGS; `n_rungs` in
    the table and the legend): `pretrain.ladder_report._fit`, the ladder
    health check's law, applied to the analysis' cells."""
    loss = fin[(fin["task"] == "train_loss") & (fin["seed"] == GRID_SEED)]
    loss = loss.assign(N=loss["size"].map(NON_EMB)).dropna(subset=["N"])
    rows = []
    for (L, arch, scheme), g in loss.groupby(["L", "arch", "scheme"]):
        g = g.sort_values("N")
        fit = _fit(list(zip(g["N"], g["primary_score"]))) if len(g) >= MIN_RUNGS else None
        for _, r in g.iterrows():
            pred = float(np.exp(fit[1] + fit[0] * np.log(r["N"]))) if fit else np.nan
            rows.append({"L": int(L), "arch": arch, "scheme": scheme, "size": r["size"], "N": r["N"],
                         "final_loss": r["primary_score"], "alpha": -fit[0] if fit else np.nan,
                         "pred_loss": pred, "n_rungs": len(g)})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "scaling_fit.csv", index=False)
    if out.empty:
        return out
    Ls = sorted(out["L"].unique())
    fig, axes = plt.subplots(1, len(Ls), figsize=(3.1 * len(Ls), 3.2), squeeze=False)
    for ax, L in zip(axes[0], Ls):
        for (arch, scheme), g in out[out["L"] == L].groupby(["arch", "scheme"]):
            g = g.sort_values("N")
            st = dict(color=S.SIZE_COLOR["1B"] if arch == "deep" else S.SERIES[1], ls=S.SCHEME_DASH.get(scheme, "-"))
            lab = f"{arch}/{scheme}" + (f" α={g['alpha'].iloc[0]:.3f}" if g["alpha"].notna().any() else "") + f" (n={len(g)})"
            ax.plot(g["N"], g["final_loss"], "o", ms=4, color=st["color"], label=lab)
            if g["pred_loss"].notna().any():
                ax.plot(g["N"], g["pred_loss"], lw=1, alpha=.7, **st)
        ax.set_xscale("log"); ax.set_yscale("log")
        sizes = size_order(out["size"].unique())
        ax.set_xticks([NON_EMB[s] for s in sizes]); ax.set_xticklabels(sizes, fontsize=6)
        ax.set_xticks([], minor=True); ax.set_title(f"L = {L}", loc="left")
        ax.set_xlabel("non-embedding params"); ax.set_ylabel("final lm loss")
        ax.legend(fontsize=6, frameon=False); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    top = G._header(fig, "Final loss against size, one fit per (arch, scheme) over every rung",
                    f"point = the seed-{GRID_SEED} cell's final training loss; line = log loss = log A − α log N fitted over the (L, arch, "
                    f"scheme)'s rungs present (n in the legend, ≥ {MIN_RUNGS}); no gate (the loss has no chance level)")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, out_dir / "scaling_fit.png", dpi=150)
    return out


# --- README ----------------------------------------------------------------------

def generate_readme(pool: str, out_dir: Path, fits: pd.DataFrame, fam: pd.DataFrame,
                    sf: pd.DataFrame) -> None:
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    fitted = fits.dropna(subset=["r2"])
    bench = fitted[fitted["kind"] == "benchmark"]
    by_opt = bench.groupby("n_options")["r2"].median()
    top = fam.sort_values("r2", ascending=False)
    no_fit = sorted(set(fits.loc[fits["gated"], "task"]) - set(fitted["task"]))
    bullets = [
        f"- **{len(fitted)} (task, L) fits over {fitted['task'].nunique()} tasks**, each on the rungs where the task is above "
        f"chance (rule 1, rq00's mask): the gate removed {int(fitted['gated_rungs'].sum())} rungs from the fitted series and left "
        f"{int(fits['gated'].sum())} (task, L) series with fewer than {MIN_RUNGS} rungs (no fit, `gated` in `rq1_fits.csv`), "
        f"{len(no_fit)} tasks without any fit. Best-scaling families (median R²): "
        + ", ".join(f"`{f}` {fmt(r.r2)}" for f, r in top.head(4).iterrows())
        + "; worst: " + ", ".join(f"`{f}` {fmt(r.r2)}" for f, r in top.tail(3).iterrows()) + ".",
        "- **Median R² by answer count** (over the gated benchmark fits): "
        + ", ".join(f"{fmt(v)} over the {int(k)}-option fits" for k, v in by_opt.items()) + ".",
    ]
    if not sf.empty:
        al = sf.dropna(subset=["alpha"]).groupby(["L", "arch", "scheme"]).agg(alpha=("alpha", "first"), n=("n_rungs", "first"))
        bullets.append(f"- **Loss exponent α per (L, arch, scheme)**, the seed-{GRID_SEED} cells, sizes in the fit in brackets: "
                       + ", ".join(f"L{L} {a}/{s} {fmt(r.alpha, 3)} ({int(r.n)})" for (L, a, s), r in al.iterrows()) + ".")
    rows = [[f, fmt(r.r2), fmt(r.rho), int(r.n), "" if not np.isfinite(r.n_options) else int(r.n_options)]
            for f, r in fam.iterrows()]
    blocks = [f"**Per family** (median over its gated (task, L) fits, ≥ {MIN_FAMILY_FITS} fits; ρ = −1 is the ideal for BPB and loss):",
              md_table(["family", "R²", "ρ", "fits", "options"], rows),
              f"![RQ1 scaling]({rel}/rq1_scaling.png)",
              f"![Scaling fit]({rel}/scaling_fit.png)"]
    readme = OUT_ROOT / "README.md"
    gen = f"analyze.py --pool {pool}"
    replace_block(readme, "highlight", "## Highlighted result\n\n" + "\n".join(bullets), gen)
    replace_block(readme, "results", "## Results\n\n"
                  + f"Numbers from the `{pool}` pool. Regenerate with "
                  f"`python analysis/rq01_scaling_predictability/analyze.py --pool {pool}`.\n\n"
                  + "\n\n".join(blocks), gen)
    print(f"Wrote auto README blocks → {readme}")


# --- driver ------------------------------------------------------------------------

def main(pool: str, out_dir: Path) -> None:
    df = ladder_frame(pool)
    fin = finals(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pool '{pool}': {df['model'].nunique()} cells, {fin['task'].nunique()} tasks")
    fits, fam = fit_table(fin, pool)
    fits.to_csv(out_dir / "rq1_fits.csv", index=False)
    fam.to_csv(out_dir / "rq1_families.csv")
    fitted = fits.dropna(subset=["r2"])
    print(f"Wrote → {out_dir / 'rq1_fits.csv'} ({len(fitted)} fits over {fitted['task'].nunique()} tasks, {len(fam)} families; "
          f"gate: {int(fits['gated'].sum())} (task, L) series without a fit, {int(fitted['gated_rungs'].sum())} rungs removed from the fitted ones)")
    if not fam.empty:
        plot_rq1(fin, fits, fam, out_dir)
    sf = loss_scaling(fin, out_dir)
    facts = {"grid": {"cells": int(df["model"].nunique()),
                      "by_size": df.groupby("size")["model"].nunique().to_dict()},
             "rq1": {"n_fits": int(len(fitted)), "n_tasks": int(fitted["task"].nunique()),
                     "gated_series": int(fits["gated"].sum()), "gated_rungs": int(fitted["gated_rungs"].sum()),
                     "families": fam.round(3).reset_index().to_dict("records"),
                     "by_options": fitted[fitted["kind"] == "benchmark"].groupby("n_options")["r2"].median().round(3).to_dict()}}
    (out_dir / "facts.json").write_text(json.dumps(facts, indent=1, default=str))
    generate_readme(pool, out_dir, fits, fam, sf)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Ladder pool from configs/models.json (default: {CANONICAL})")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    main(args.pool, OUT_ROOT / stage / args.pool)
