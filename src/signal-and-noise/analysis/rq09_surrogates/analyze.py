"""RQ9 (paper RQ3) — Can we know cheaply whether the decision will be reliable?

Statistics computable on the proxy alone — an SNR variant, its signal or noise
part, the proxy's own early-checkpoint agreement, the scaling-fit R² and the
margin above chance — against the proxy's decision accuracy versus the
reference (DA-size, rq01's truth as carried in rq02's per-task table). Every
candidate is scored on the same tasks, the above-random survivors at that
proxy size, as a Spearman ρ per proxy size and task kind.

    rq3_surrogates.csv        ρ, p and n per (proxy size, kind, metric)
    rq3_surrogates.png/.pdf

    python analysis/rq09_surrogates/analyze.py --pool predictivity
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
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, fmt, md_table, replace_block  # noqa: E402
from analysis.paths import ACC_VS_FLOPS, SCALING_PREDICTABILITY, SNR_DEFINITION, SURROGATES  # noqa: E402
from analysis.rq00_acc_vs_flops.above_random import task_n_options  # noqa: E402
from analysis.rq01_decision_accuracy.compute_da import _frac_label  # noqa: E402
from analysis.utils import CKPT_DA_EARLY_FRACS, SMALL_SIZES  # noqa: E402

OUT_ROOT = SURROGATES
CANONICAL = CANONICAL_POOL                 # rq02's headline table
FITS_POOL = "predictivity_all"             # where rq07 writes its fits
MIN_TASKS = 8
KINDS = ("benchmark tasks", "per-language bits per byte")
mpl.rcParams.update(S.RC)


def candidates(v: pd.DataFrame, s: str, scores: pd.DataFrame, fits_r2: pd.Series) -> dict:
    """Name -> per-task series of a statistic readable on the proxy alone."""
    early = _frac_label(CKPT_DA_EARLY_FRACS[0])
    c = {
        "SNR, relative std": v.get(f"snr_rel_std_{s}"),
        "SNR, discrepancy": v.get(f"snr_discrepancy_{s}"),
        "SNR, dist_std": v.get(f"snr_dist_std_{s}"),
        "signal alone (relative std)": v.get(f"signal_rel_std_{s}"),
        "noise alone (relative std, inverted)": -v[f"noise_rel_std_{s}"] if f"noise_rel_std_{s}" in v else None,
        f"early-checkpoint agreement ({int(CKPT_DA_EARLY_FRACS[0] * 100)} %)": v.get(f"decision_acc_ckpt_{early}_{s}"),
        "scaling-fit R²": v["task"].map(fits_r2),
    }
    if s in scores.columns:
        margin = scores[s] - 1 / scores["task"].map(task_n_options)
        margin.index = scores["task"]
        c["margin above chance"] = v["task"].map(margin)
    return c


def surrogates(v: pd.DataFrame, scores: pd.DataFrame, fits_r2: pd.Series) -> pd.DataFrame:
    rows = []
    proxies = [s for s in SMALL_SIZES if f"decision_acc_size_{s}" in v.columns and f"snr_rel_std_{s}" in v.columns]
    for s in proxies:
        da = v[f"decision_acc_size_{s}"]
        gated = v[f"snr_rel_std_{s}"].notna()             # the gate survivors at this size
        is_bpb = v["task"].str.startswith("bpb_")
        for kind, mask in ((KINDS[0], gated & ~is_bpb), (KINDS[1], gated & is_bpb)):
            for name, x in candidates(v, s, scores, fits_r2).items():
                if x is None:
                    continue
                x = pd.to_numeric(x, errors="coerce")
                ok = mask & x.notna() & da.notna() & np.isfinite(x)
                if ok.sum() < MIN_TASKS:
                    continue
                r = spearmanr(x[ok], da[ok])
                rows.append({"proxy": s, "kind": kind, "metric": name, "rho": r.statistic,
                             "p": r.pvalue, "n": int(ok.sum())})
    return pd.DataFrame(rows)


def plot(t: pd.DataFrame, out_dir: Path) -> None:
    proxies = [s for s in SMALL_SIZES if s in set(t["proxy"])]
    order = t[t["kind"] == KINDS[0]].groupby("metric")["rho"].mean().sort_values().index.tolist()
    order += [m for m in t["metric"].unique() if m not in order]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, kind in zip(axes, KINDS):
        g = t[t["kind"] == kind]
        y = np.arange(len(order))
        for j, s in enumerate(proxies):
            gg = g[g["proxy"] == s].set_index("metric").reindex(order)
            off = (j - (len(proxies) - 1) / 2) * .22
            ax.scatter(gg["rho"], y + off, s=34, color=S.SIZE_COLOR[s], zorder=3, label=f"{s} proxy")
            for yi, (_m, r) in enumerate(gg.iterrows()):
                if np.isfinite(r["rho"]):
                    ax.annotate(f"n={int(r['n'])}", (r["rho"], yi + off), xytext=(5, -2),
                                textcoords="offset points", fontsize=5.5, color=S.MUTED)
        ax.axvline(0, color=S.MUTED, lw=.8)
        ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7.5)
        ax.set_xlim(-0.6, 0.9); ax.set_xlabel("Spearman ρ with decision accuracy (proxy vs reference)")
        ax.set_title(kind, loc="left"); ax.grid(axis="x", color=S.GRID, lw=.6); ax.set_axisbelow(True)
        S.clean(ax); ax.tick_params(length=0)
    axes[0].legend(frameon=False, loc="lower right")
    S.save_figure(fig, out_dir, "rq3_surrogates")


def generate_readme(pool: str, out_dir: Path, t: pd.DataFrame) -> None:
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    bullets, blocks = [], []
    for kind in KINDS:
        g = t[t["kind"] == kind]
        if g.empty:
            continue
        best = g.groupby("metric")["rho"].mean().sort_values(ascending=False)
        bullets.append(f"- **{kind}** — strongest surrogate of DA-size (mean ρ over proxies): "
                       f"`{best.index[0]}` {fmt(best.iloc[0])}; weakest: `{best.index[-1]}` {fmt(best.iloc[-1])}.")
        piv = g.pivot_table(index="metric", columns="proxy", values="rho").reindex(best.index)
        cols = [s for s in SMALL_SIZES if s in piv.columns]
        blocks += [f"**{kind}** (Spearman ρ of the statistic with DA-size, per proxy size):",
                   md_table(["metric"] + cols, [[m] + [fmt(piv.loc[m, c]) for c in cols] for m in piv.index])]
    blocks.append(f"![Surrogates]({stage}/{pool}/rq3_surrogates.png)")
    readme = OUT_ROOT / "README.md"
    gen = f"analyze.py --pool {pool}"
    replace_block(readme, "highlight", "## Highlighted result\n\n" + "\n".join(bullets), gen)
    replace_block(readme, "results", "## Results\n\n"
                  + f"Numbers from the `{pool}` pool's rq02 table. Regenerate with "
                  f"`python analysis/rq09_surrogates/analyze.py --pool {pool}`.\n\n"
                  + "\n\n".join(blocks), gen)
    print(f"Wrote auto README blocks → {readme}")


def main(pool: str, out_dir: Path) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    src = SNR_DEFINITION / stage / pool / "snr_variants_per_task.csv"
    if not src.is_file():
        sys.exit(f"missing {src} — run rq01 compute_da.py and rq02 run_apertus_snr_variants.py --pool {pool} first")
    v = pd.read_csv(src, index_col=0).rename_axis("task").reset_index()
    scores = pd.read_csv(ACC_VS_FLOPS / stage / pool / "above_random_scores.csv")
    fits_path = SCALING_PREDICTABILITY / stage / FITS_POOL / "rq1_fits.csv"
    fits_r2 = (pd.read_csv(fits_path).groupby("task")["r2"].median() if fits_path.is_file()
               else pd.Series(dtype=float))
    if fits_r2.empty:
        print(f"note: no scaling fits at {fits_path}; the R² candidate is skipped")
    out_dir.mkdir(parents=True, exist_ok=True)
    t = surrogates(v, scores, fits_r2)
    t.to_csv(out_dir / "rq3_surrogates.csv", index=False)
    print(f"Wrote → {out_dir / 'rq3_surrogates.csv'} ({len(t)} rows)")
    if not t.empty:
        plot(t, out_dir)
    (out_dir / "facts.json").write_text(json.dumps({"rq3": t.round(3).to_dict("records")}, indent=1, default=str))
    generate_readme(pool, out_dir, t)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Pool whose rq02 table to read (default: {CANONICAL})")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    main(args.pool, OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool)
