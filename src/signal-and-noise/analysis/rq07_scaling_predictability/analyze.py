"""RQ7 (paper RQ1) — What scales predictably?

Per (task, L), the final-checkpoint score of the deep, scheme-A, seed-1904
ladder against log10 N over the rungs from 175M up: the R² of the log-linear
fit and the Spearman ρ with size, then medians per benchmark family. A family
whose scores do not move with size has no trend to extrapolate, whatever its
provenance (rq05's gate result, read as a scaling statement).

The same folder holds the ladder's descriptive curves, drawn from the loader's
frame so they carry the analysis' assumptions (diverged and unfinished runs
dropped, shared checkpoint grid, `require_final`):

    rq1_scaling.png/.pdf   the paper figure: above-chance curves at L=30, R²/ρ per family
    rq1_fits.csv           one row per (task, L) fit;  rq1_families.csv  the family medians
    loss_curves.png        training loss vs fraction of run, per L: full run and last 10 %
    scaling_fit.png/.csv   final loss vs N per L, one power-law fit per (arch, scheme)
    benchmark_curves.png   benchmark accuracy vs fraction of run per family, chance line

    python analysis/rq07_scaling_predictability/analyze.py --pool predictivity_all
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
from pretrain.ladder_report import _fit, _trained_tasks  # noqa: E402
from snr.download.ladder import ladder_dir  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import SCALING_PREDICTABILITY  # noqa: E402
from analysis.rq00_acc_vs_flops.above_random import task_n_options  # noqa: E402
from analysis.utils import (  # noqa: E402
    GRID_SEED, LADDER_SIZES, NON_EMB, _is_parent_task, benchmark_family, finals,
    ladder_frame, size_order)

OUT_ROOT = SCALING_PREDICTABILITY
CANONICAL = "predictivity_all"
MIN_RUNGS = 3
EXAMPLE_L = 30
CLOSEUP_YMAX = 3.5          # ceiling of the last-10 % loss panels, as in the report
mpl.rcParams.update(S.RC)


def _grid(fin: pd.DataFrame) -> pd.DataFrame:
    return fin[(fin["seed"] == GRID_SEED) & (fin["arch"] == "deep") & (fin["scheme"] == "A")]


def _line_style(size, arch, scheme) -> dict:
    return dict(color=S.SIZE_COLOR.get(size, S.MUTED), lw=S.ARCH_WIDTH.get(arch, 1.0),
                ls=S.SCHEME_DASH.get(scheme, "-"))


# --- the paper's RQ1: log-N fits ------------------------------------------------

def fit_table(fin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    g0 = _grid(fin)
    g0 = g0[(g0["size"] != LADDER_SIZES[0]) & g0["task"].map(_is_parent_task)]
    rows = []
    for (task, L), g in g0.groupby(["task", "L"]):
        g = g.assign(N=g["size"].map(NON_EMB)).dropna(subset=["N"]).sort_values("N")
        if len(g) < MIN_RUNGS:
            continue
        x, y = np.log10(g["N"].to_numpy()), g["primary_score"].to_numpy(float)
        b, a = np.polyfit(x, y, 1)
        ss_res = float(((y - (b * x + a)) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        rho = spearmanr(x, y).statistic if len(set(y)) > 1 else np.nan
        kind = g["kind"].iloc[0]
        rows.append({"task": task, "L": int(L), "family": benchmark_family(task), "kind": kind,
                     "n_rungs": len(g), "slope": b,
                     "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan, "rho": rho,
                     "n_options": task_n_options(task) if kind == "benchmark" else np.nan})
    fits = pd.DataFrame(rows)
    fits = fits[fits["task"] != "bpb_macro"]
    fam = fits.groupby("family").agg(
        r2=("r2", "median"), rho=("rho", "median"), n=("task", "size"),
        n_options=("n_options", lambda s: s.mode().iloc[0] if s.notna().any() else np.nan))
    fam = fam[fam["n"] >= MIN_RUNGS].sort_values("r2")
    return fits, fam


def plot_rq1(fin: pd.DataFrame, fits: pd.DataFrame, fam: pd.DataFrame, out_dir: Path) -> None:
    g0 = _grid(fin)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) score above chance against size, family medians at one L
    ex = g0[(g0["L"] == EXAMPLE_L) & (g0["kind"] == "benchmark") & (g0["size"] != LADDER_SIZES[0])].copy()
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
    a0.set_xticks([NON_EMB[s] for s in LADDER_SIZES[1:]]); a0.set_xticklabels(LADDER_SIZES[1:])
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
    a1.set_xlabel(f"median over the family's (task, L) fits, {LADDER_SIZES[1]} to {LADDER_SIZES[-1]}")
    a1.set_title("(b) how predictably each family scales", loc="left")
    a1.legend(frameon=False, loc="lower left"); a1.grid(axis="x", color=S.GRID, lw=.6); a1.set_axisbelow(True)
    S.clean(a1); a1.tick_params(length=0)
    fig.subplots_adjust(wspace=.55)
    S.save_figure(fig, out_dir, "rq1_scaling")


# --- the ladder's curves, from the loader's frame -------------------------------

def plot_loss_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """The report's loss figure on the analysis' cells: per L, the whole run
    and its last 10 % (the close-up is where arch and scheme separate)."""
    curve = pd.read_csv(ladder_dir() / "ladder_report_curve.csv", low_memory=False)
    curve = curve[curve["cell"].isin(set(df["model"]))]
    if curve.empty:
        return
    Ls = sorted(curve["L"].unique())
    fig, axes = plt.subplots(len(Ls), 2, figsize=(9.5, 2.6 * len(Ls)), squeeze=False)
    for row, L in enumerate(Ls):
        sub = curve[curve["L"] == L]
        for col, (lo, title) in enumerate(((0.0, "full run"), (0.9, "last 10 %"))):
            ax = axes[row][col]
            win = sub[sub["frac"] >= lo]
            for _cell, g in win.groupby("cell"):
                g = g.sort_values("frac")
                ax.plot(g["frac"], g["loss"], **_line_style(*g.iloc[0][["size", "arch", "scheme"]]))
            if col == 0:
                ax.set_ylim(2, 8)
            else:
                v = win["loss"]
                ax.set_ylim(v.min() * 0.995, min(v.max() * 1.005, CLOSEUP_YMAX))
            ax.set_title(f"L = {L} — {title}", loc="left"); ax.set_xlabel("fraction of run")
            ax.set_ylabel("lm loss"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    handles = ([plt.Line2D([], [], color=S.SIZE_COLOR[s], lw=2, label=s) for s in LADDER_SIZES if s in set(curve["size"])]
               + [plt.Line2D([], [], color=S.INK, lw=S.ARCH_WIDTH[a], label=a) for a in ("deep", "shallow") if a in set(curve["arch"])]
               + [plt.Line2D([], [], color=S.INK, ls=S.SCHEME_DASH[v], label=f"scheme {v}") for v in S.SCHEME_DASH if v in set(curve["scheme"])])
    fig.legend(handles=handles, ncol=min(8, len(handles)), loc="lower center", frameon=False)
    fig.suptitle("Training loss — colour = size, width = arch, dash = data scheme", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    S.save(fig, out_dir / "loss_curves.png", dpi=150)


def loss_scaling(fin: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Final training loss against N per L, one power law per (arch, scheme)
    over every rung present (≥ 3): `pretrain.ladder_report._fit`, the
    ladder health check's law, applied to the analysis' cells."""
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
            lab = f"{arch}/{scheme}" + (f" α={g['alpha'].iloc[0]:.3f}" if g["alpha"].notna().any() else "")
            ax.plot(g["N"], g["final_loss"], "o", ms=4, color=st["color"], label=lab)
            if g["pred_loss"].notna().any():
                ax.plot(g["N"], g["pred_loss"], lw=1, alpha=.7, **st)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([NON_EMB[s] for s in LADDER_SIZES]); ax.set_xticklabels(LADDER_SIZES, fontsize=6)
        ax.set_xticks([], minor=True); ax.set_title(f"L = {L}", loc="left")
        ax.set_xlabel("non-embedding params"); ax.set_ylabel("final lm loss")
        ax.legend(fontsize=6, frameon=False); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    fig.suptitle("Final loss against size, one fit per (arch, scheme) over every rung", y=1.02)
    fig.tight_layout()
    S.save(fig, out_dir / "scaling_fit.png", dpi=150)
    return out


def plot_benchmark_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """Benchmark accuracy against fraction of run, one panel per family, one
    line per cell over the tasks in the languages that cell trains on (the
    watcher's list), with the chance line from the option count."""
    b = df[df["kind"] == "benchmark"]
    b = b[[t in _trained_tasks(L, s) for t, L, s in zip(b["task"], b["L"], b["scheme"])]].copy()
    if b.empty:
        return
    b["family"] = b["task"].map(benchmark_family)
    b["chance"] = 1 / b["task"].map(task_n_options)
    fams = sorted(b["family"].unique())
    cols = min(4, len(fams)); rows = (len(fams) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False)
    flat = [a for r in axes for a in r]
    for ax in flat[len(fams):]:
        ax.axis("off")
    for ax, fam in zip(flat, fams):
        g = b[b["family"] == fam]
        for _cell, gc in g.groupby("model"):
            m = gc.groupby("frac")["primary_score"].mean().sort_index()
            ax.plot(m.index, m.values, **_line_style(*gc.iloc[0][["size", "arch", "scheme"]]))
        ch = g["chance"].dropna()
        if not ch.empty:
            ax.axhline(ch.mean(), color=S.ALERT, lw=.9, ls=":")
        ax.set_title(fam, loc="left"); ax.set_xlabel("fraction of run"); ax.set_ylabel("accuracy")
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    fig.suptitle("Benchmark accuracy vs fraction of run, mean over the cell's trained-language tasks\n"
                 "colour = size, width = arch, dash = scheme, dotted red = chance", y=1.0)
    fig.tight_layout()
    S.save(fig, out_dir / "benchmark_curves.png", dpi=150)


# --- README ----------------------------------------------------------------------

def generate_readme(pool: str, out_dir: Path, fits: pd.DataFrame, fam: pd.DataFrame,
                    sf: pd.DataFrame) -> None:
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    bench = fits[fits["kind"] == "benchmark"]
    by_opt = bench.groupby("n_options")["r2"].median()
    top = fam.sort_values("r2", ascending=False)
    bullets = [
        f"- **{len(fits)} (task, L) fits over {fits['task'].nunique()} tasks**; "
        f"best-scaling families (median R²): "
        + ", ".join(f"`{f}` {fmt(r.r2)}" for f, r in top.head(4).iterrows())
        + "; worst: " + ", ".join(f"`{f}` {fmt(r.r2)}" for f, r in top.tail(3).iterrows()) + ".",
        "- **The answer count splits the families**: median R² "
        + ", ".join(f"{fmt(v)} over the {int(k)}-option fits" for k, v in by_opt.items()) + ".",
    ]
    if not sf.empty:
        al = sf.dropna(subset=["alpha"]).groupby(["L", "arch", "scheme"])["alpha"].first()
        bullets.append("- **Loss exponent α per (L, arch, scheme)**: "
                       + ", ".join(f"L{L} {a}/{s} {fmt(v, 3)}" for (L, a, s), v in al.items()) + ".")
    rows = [[f, fmt(r.r2), fmt(r.rho), int(r.n), "" if not np.isfinite(r.n_options) else int(r.n_options)]
            for f, r in fam.iterrows()]
    blocks = ["**Per family** (median over its (task, L) fits; ρ = −1 is the ideal for BPB and loss):",
              md_table(["family", "R²", "ρ", "fits", "options"], rows),
              f"![RQ1 scaling]({rel}/rq1_scaling.png)",
              f"![Loss curves]({rel}/loss_curves.png)",
              f"![Scaling fit]({rel}/scaling_fit.png)",
              f"![Benchmark curves]({rel}/benchmark_curves.png)"]
    readme = OUT_ROOT / "README.md"
    gen = f"analyze.py --pool {pool}"
    replace_block(readme, "highlight", "## Highlighted result\n\n" + "\n".join(bullets), gen)
    replace_block(readme, "results", "## Results\n\n"
                  + f"Numbers from the `{pool}` pool. Regenerate with "
                  f"`python analysis/rq07_scaling_predictability/analyze.py --pool {pool}`.\n\n"
                  + "\n\n".join(blocks), gen)
    print(f"Wrote auto README blocks → {readme}")


# --- driver ------------------------------------------------------------------------

def main(pool: str, out_dir: Path) -> None:
    df = ladder_frame(pool)
    fin = finals(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pool '{pool}': {df['model'].nunique()} cells, {fin['task'].nunique()} tasks")
    fits, fam = fit_table(fin)
    fits.to_csv(out_dir / "rq1_fits.csv", index=False)
    fam.to_csv(out_dir / "rq1_families.csv")
    print(f"Wrote → {out_dir / 'rq1_fits.csv'} ({len(fits)} fits, {len(fam)} families)")
    if not fam.empty:
        plot_rq1(fin, fits, fam, out_dir)
    plot_loss_curves(df, out_dir)
    sf = loss_scaling(fin, out_dir)
    plot_benchmark_curves(df, out_dir)
    facts = {"grid": {"cells": int(df["model"].nunique()),
                      "by_size": df.groupby("size")["model"].nunique().to_dict()},
             "rq1": {"n_fits": int(len(fits)),
                     "families": fam.round(3).reset_index().to_dict("records"),
                     "by_options": fits[fits["kind"] == "benchmark"].groupby("n_options")["r2"].median().round(3).to_dict()}}
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
