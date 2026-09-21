"""Statistics beyond SNR (paper RQ3) — can we know cheaply whether the decision will be reliable?

Statistics computable on the proxy alone — an SNR variant, its signal or noise
part, the proxy's own early-checkpoint agreement, the R² of the log-N fit over
the rungs up to the proxy (rule 11: never a fit that includes the reference;
it needs MIN_RUNGS rungs, so it exists from 600M) and the margin above chance —
against the proxy's decision accuracy versus the reference (DA-size, rq02's
truth as carried in rq03's per-task table). The population at a proxy size is
the tasks above chance at the proxy and at the reference (rule 1); a
candidate is scored, as a Spearman ρ per proxy size and task kind, on the
tasks of that population where it has a value, so n differs per candidate
and is reported next to every ρ. Benchmarks and the per-language BPB are two
populations; `bpb_macro` and `train_loss` are in neither (rule 7).

    rq3_surrogates.csv        ρ, p and n per (proxy size, kind, metric)
    rq3_surrogates.png/.pdf

    python analysis/rq04_surrogates/analyze.py --pool predictivity
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
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, fmt, md_table, replace_block  # noqa: E402
from analysis.paths import GATE_AND_CURVES, NOISE_AND_SNR, SURROGATES  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask, task_n_options  # noqa: E402
from analysis.rq01_scaling_predictability.analyze import MIN_RUNGS, fit_table  # noqa: E402
from analysis.rq02_decision_accuracy.compute_da import _frac_label  # noqa: E402
from analysis.utils import (  # noqa: E402
    CKPT_DA_EARLY_FRACS, EVAL_SIZES, SMALL_SIZES, TARGET_SIZE, assign_language, finals, ladder_frame,
    languages_only, passes_gate)

OUT_ROOT = SURROGATES
CANONICAL = CANONICAL_POOL                 # rq03's headline table
FITS_POOL = "predictivity_all"             # the pool rq01 fits on
MIN_TASKS = 8
KINDS = ("benchmark tasks", "per-language bits per byte")
R2_NAME = "scaling-fit R² (proxy rungs only)"
mpl.rcParams.update(S.RC)


def proxy_fit_r2(proxies: list[str], pool: str) -> pd.DataFrame:
    """task x proxy: the median over L of rq01's log-N fit R² (`fit_table`:
    the same fit, the same grid cells, the same gate on the rungs) refitted
    on the rungs up to and including the proxy, so the number is available
    at the proxy (rule 11). A proxy below MIN_RUNGS rungs has no column."""
    fin = finals(ladder_frame(FITS_POOL))
    out = {}
    for s in proxies:
        rungs = EVAL_SIZES[:EVAL_SIZES.index(s) + 1]
        if len(rungs) < MIN_RUNGS:
            continue
        fits, _ = fit_table(fin[fin["size"].isin(rungs)], pool)
        out[s] = fits.groupby("task")["r2"].median()
    return pd.DataFrame(out)


def candidates(v: pd.DataFrame, s: str, scores: pd.DataFrame, fits_r2: pd.DataFrame) -> dict:
    """Name -> per-task series of a statistic readable on the proxy alone
    (`fits_r2`: `proxy_fit_r2`'s frame)."""
    early = _frac_label(CKPT_DA_EARLY_FRACS[0])
    c = {
        "SNR, relative std": v.get(f"snr_rel_std_{s}"),
        "SNR, discrepancy": v.get(f"snr_discrepancy_{s}"),
        "SNR, dist_std": v.get(f"snr_dist_std_{s}"),
        "signal alone (relative std)": v.get(f"signal_rel_std_{s}"),
        "noise alone (relative std, inverted)": -v[f"noise_rel_std_{s}"] if f"noise_rel_std_{s}" in v else None,
        f"early-checkpoint agreement ({int(CKPT_DA_EARLY_FRACS[0] * 100)} %)": v.get(f"decision_acc_ckpt_{early}_{s}"),
        R2_NAME: v["task"].map(fits_r2[s]) if s in fits_r2.columns else None,
    }
    if s in scores.columns:
        margin = scores[s] - 1 / scores["task"].map(task_n_options)
        margin.index = scores["task"]
        c["margin above chance"] = v["task"].map(margin)
    return c


def surrogates(v: pd.DataFrame, scores: pd.DataFrame, fits_r2: pd.DataFrame,
               gate: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (proxy, kind, candidate): Spearman ρ with DA-size over the
    gated tasks (above chance at the proxy and at TARGET_SIZE, `gate` = rq00's
    mask) where the candidate has a value; `n` is that count."""
    rows = []
    proxies = [s for s in SMALL_SIZES if f"decision_acc_size_{s}" in v.columns and f"snr_rel_std_{s}" in v.columns]
    for s in proxies:
        da = v[f"decision_acc_size_{s}"]
        gated = (v[f"snr_rel_std_{s}"].notna()             # rq03's survivors at this size
                 & passes_gate(gate, v["task"], s, TARGET_SIZE).to_numpy())   # and above chance at the reference (rule 1)
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
        ax.set_xlim(min(-0.6, t["rho"].min() - 0.1), max(0.9, t["rho"].max() + 0.1))   # from the data: no point clipped
        ax.set_xlabel("Spearman ρ with decision accuracy (proxy vs reference)")
        ax.set_title(kind, loc="left"); ax.grid(axis="x", color=S.GRID, lw=.6); ax.set_axisbelow(True)
        S.clean(ax); ax.tick_params(length=0)
    axes[0].legend(frameon=False, loc="lower right")
    top = G._header(fig, "Which statistic of the proxy alone predicts decision accuracy?",
                    f"point = Spearman ρ, over the tasks above chance at the proxy and at {TARGET_SIZE} where the statistic has a "
                    f"value (n next to it; ≥ {MIN_TASKS}), between the statistic read on the proxy and the task's DA-size "
                    f"(proxy final → {TARGET_SIZE} final); the R² is fitted on the rungs up to the proxy only; `bpb_macro` and "
                    "`train_loss` are in neither population")
    fig.tight_layout(rect=(0, 0, 1, top))
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
        n = g.pivot_table(index="metric", columns="proxy", values="n").reindex(best.index)
        cols = [s for s in SMALL_SIZES if s in piv.columns]
        blocks += [f"**{kind}** (Spearman ρ of the statistic with DA-size, per proxy size; n = the tasks behind the ρ):",
                   md_table(["metric"] + [f"{c} ρ (n)" for c in cols],
                            [[m] + [f"{fmt(piv.loc[m, c])} ({int(n.loc[m, c])})" if np.isfinite(piv.loc[m, c]) else ""
                                    for c in cols] for m in piv.index])]
    blocks.append(f"![Surrogates]({stage}/{pool}/rq3_surrogates.png)")
    readme = OUT_ROOT / "README.md"
    body = "\n\n".join([
        "## Statistics beyond SNR (paper RQ3)",
        f"Numbers from the `{pool}` pool's rq03 table. Regenerate with "
        f"`python analysis/rq04_surrogates/analyze.py --pool {pool}`. The population at a proxy size is the tasks "
        f"above chance at the proxy and at {TARGET_SIZE} (rule 1); each candidate is scored on the tasks of it where "
        f"the candidate has a value, so n differs per candidate and is given next to every ρ. The scaling-fit R² is "
        f"rq01's log-N fit refitted on the rungs up to the proxy only (rule 11; it needs {MIN_RUNGS} rungs, so it "
        f"starts at {EVAL_SIZES[MIN_RUNGS - 1]}). `bpb_macro` and `train_loss` are in neither population (rule 7).",
        "\n".join(bullets)] + blocks)
    replace_block(readme, "surrogates", body, f"analyze.py --pool {pool}")
    print(f"Wrote auto README block → {readme}")


def main(pool: str, out_dir: Path) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    src = NOISE_AND_SNR / stage / pool / "snr_variants_per_task.csv"
    if not src.is_file():
        sys.exit(f"missing {src} — run rq02 compute_da.py and rq03 run_apertus_snr_variants.py --pool {pool} first")
    v = pd.read_csv(src, index_col=0).rename_axis("task").reset_index()
    v = languages_only(v.assign(language=v["task"].map(assign_language)))   # rule 7: no bpb_macro / train_loss
    scores = pd.read_csv(GATE_AND_CURVES / stage / pool / "above_random_scores.csv")
    fits_r2 = proxy_fit_r2(SMALL_SIZES, pool)
    print(f"proxy-only scaling-fit R² at {list(fits_r2.columns)} (≥ {MIN_RUNGS} rungs; NaN below)")
    out_dir.mkdir(parents=True, exist_ok=True)
    t = surrogates(v, scores, fits_r2, load_mask(pool))
    t.to_csv(out_dir / "rq3_surrogates.csv", index=False)
    print(f"Wrote → {out_dir / 'rq3_surrogates.csv'} ({len(t)} rows)")
    if not t.empty:
        plot(t, out_dir)
    (out_dir / "facts.json").write_text(json.dumps({"rq3": t.round(3).to_dict("records")}, indent=1, default=str))
    generate_readme(pool, out_dir, t)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Pool whose rq03 table to read (default: {CANONICAL})")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    main(args.pool, OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool)
