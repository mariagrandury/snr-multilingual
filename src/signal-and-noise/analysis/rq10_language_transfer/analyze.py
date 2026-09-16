"""RQ10 (paper RQ5) — Does it generalize to languages we have not measured?

Per-language BPB scaling on the plan grid (deep, scheme A, seed 1904): each
language's own exponent over the proxy rungs, and a leave-one-language-out
test — the held-out language's reference BPB predicted from its k smallest
rungs and the exponent pooled over the other languages, against its own fit
and against "the largest proxy as is". Trained and never-trained languages
are reported apart. The folder also holds the detailed per-language BPB
curves of every cell (the progress report's BPB figure on the analysis' cells).

    rq5_transfer.csv          one row per (L, language, k)
    rq5_transfer_summary.csv  median |relative error| per (trained, k) and method
    rq5_transfer.png/.pdf     the paper figure
    bpb_curves.png            per cell: per-language BPB vs fraction of run

    python analysis/rq10_language_transfer/analyze.py --pool predictivity_all
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

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import LANGUAGE_TRANSFER, PROXY_PREDICTIVITY  # noqa: E402
from analysis.utils import (  # noqa: E402
    GRID_SEED, LADDER_SIZES, NON_EMB, finals, ladder_frame, size_order, trained_bpb_tasks)

OUT_ROOT = LANGUAGE_TRANSFER
CANONICAL = "predictivity_all"
MIN_SIZES = 4                       # three proxy rungs + the reference
BPB, UNTR = S.RAMP[3], S.SERIES[2]
mpl.rcParams.update(S.RC)


def transfer(fin: pd.DataFrame) -> pd.DataFrame:
    g0 = fin[(fin["seed"] == GRID_SEED) & (fin["arch"] == "deep") & (fin["scheme"] == "A")
             & (fin["kind"] == "bpb") & (fin["task"] != "bpb_macro") & (fin["size"] != LADDER_SIZES[0])]
    rows = []
    for L, g in g0.groupby("L"):
        piv = g.pivot_table(index="task", columns="size", values="primary_score")
        sizes = size_order(piv.columns)
        if len(sizes) < MIN_SIZES:
            continue
        piv = piv[sizes].dropna()
        ref, proxies = sizes[-1], sizes[:-1]
        x = np.log(np.array([NON_EMB[s] for s in proxies]))
        Y = np.log(piv[proxies].to_numpy())
        alpha = np.array([-np.polyfit(x, yy, 1)[0] for yy in Y])       # own exponent, all proxy rungs
        trained = trained_bpb_tasks(int(L), "A") or set()
        obs = piv[ref].to_numpy()
        for i, task in enumerate(piv.index):
            a_pool = float(np.median(np.delete(alpha, i)))
            for k in range(1, len(proxies) + 1):
                xs, ys = x[:k], Y[i, :k]
                icpt = float(np.mean(ys + a_pool * xs))               # intercept from the k smallest rungs
                pred_t = np.exp(icpt - a_pool * np.log(NON_EMB[ref]))
                out = {"L": int(L), "task": task, "trained": task in trained, "reference_size": ref,
                       "k": k, "observed": obs[i], "alpha_own": alpha[i], "alpha_pooled": a_pool,
                       "pred_transfer": pred_t, "err_transfer": (pred_t - obs[i]) / obs[i],
                       "pred_last": float(np.exp(ys[-1])), "err_last": (np.exp(ys[-1]) - obs[i]) / obs[i]}
                if k >= 3:
                    b, a = np.polyfit(xs, ys, 1)
                    pred_o = np.exp(a + b * np.log(NON_EMB[ref]))
                    out.update({"pred_own": pred_o, "err_own": (pred_o - obs[i]) / obs[i]})
                rows.append(out)
    return pd.DataFrame(rows)


def summarise(t: pd.DataFrame) -> pd.DataFrame:
    med = lambda e: float(np.median(np.abs(e.dropna()))) if e.notna().any() else np.nan  # noqa: E731
    return (t.groupby(["trained", "k"])
            .agg(transfer=("err_transfer", med), own=("err_own", med), last=("err_last", med),
                 n=("task", "size")).reset_index())


def plot_transfer(t: pd.DataFrame, agg: pd.DataFrame, out_dir: Path) -> None:
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    for trained, col, lab in ((True, BPB, "trained languages"), (False, UNTR, "never-trained languages")):
        g = agg[agg["trained"] == trained].sort_values("k")
        if g.empty:
            continue
        a0.plot(g["k"], g["transfer"], marker="o", ms=5, lw=2, color=col, label=f"transferred exponent, {lab}")
        a0.plot(g["k"], g["own"], marker="D", ms=4.5, lw=1.4, ls="--", color=col, label=f"own fit, {lab}")
        a0.plot(g["k"], g["last"], marker="x", ms=5, lw=1, ls=":", color=col, label=f"largest proxy as is, {lab}")
    a0.set_xticks(sorted(agg["k"].unique())); a0.set_xlabel("rungs measured for the held-out language (smallest first)")
    a0.set_ylabel("median |relative error| at the reference"); a0.set_yscale("log")
    a0.set_title("(a) predicting the reference's bits per byte", loc="left")
    a0.legend(frameon=False, fontsize=6.6, loc="upper right"); a0.grid(color=S.GRID, lw=.6); a0.set_axisbelow(True)
    S.clean(a0)
    k1 = t[t["k"] == 1]
    for trained, col, lab in ((True, BPB, "trained"), (False, UNTR, "never trained")):
        g = k1[k1["trained"] == trained]
        a1.scatter(g["observed"], g["pred_transfer"], s=12, color=col, alpha=.7, lw=0, label=lab)
    lim = [k1["observed"].min() * .9, k1["observed"].max() * 1.1]
    a1.plot(lim, lim, color=S.MUTED, lw=.8, ls="--")
    a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(lim); a1.set_ylim(lim)
    a1.set_xlabel("observed bits per byte at the reference"); a1.set_ylabel("predicted from one rung + pooled exponent")
    a1.set_title(f"(b) one {LADDER_SIZES[1]} measurement, every language", loc="left")
    a1.legend(frameon=False, loc="upper left"); a1.grid(color=S.GRID, lw=.6); a1.set_axisbelow(True); S.clean(a1)
    fig.subplots_adjust(wspace=.4)
    S.save_figure(fig, out_dir, "rq5_transfer")


def plot_bpb_curves(df: pd.DataFrame, out_dir: Path) -> None:
    """The report's BPB figure on the analysis' cells: one panel per cell,
    per-language BPB against fraction of run — trained languages in blue,
    unseen ones grey, the macro average dashed."""
    b = df[df["kind"] == "bpb"]
    cells = sorted(b["model"].unique())
    if not cells:
        return
    cols = 4; rows = (len(cells) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.0 * rows), squeeze=False)
    flat = [a for r in axes for a in r]
    for ax in flat[len(cells):]:
        ax.axis("off")
    for ax, cell in zip(flat, cells):
        g = b[b["model"] == cell]
        r0 = g.iloc[0]
        trained = trained_bpb_tasks(int(r0["L"]), r0["scheme"]) or {"bpb_dclm"}
        for task, gt in g.groupby("task"):
            gt = gt.sort_values("frac")
            if task == "bpb_macro":
                ax.plot(gt["frac"], gt["primary_score"], "--", lw=1.4, color=S.ALERT, label="macro (all 100)")
                continue
            tr = task in trained
            ax.plot(gt["frac"], gt["primary_score"], lw=1.2 if tr else 0.5,
                    color=S.RAMP[2] if tr else S.NODATA, zorder=3 if tr else 1)
            if tr:
                ax.annotate(task[len("bpb_"):], (gt["frac"].iloc[-1], gt["primary_score"].iloc[-1]),
                            fontsize=5.5, xytext=(3, 0), textcoords="offset points", color=S.MUTED)
        ax.set_yscale("log"); ax.set_title(cell.replace("lm-", ""), loc="left", fontsize=8)
        ax.set_xlabel("fraction of run"); ax.set_ylabel("bits per byte")
        ax.grid(color=S.GRID, lw=.6, which="both"); S.clean(ax)
        if "bpb_macro" in set(g["task"]):
            ax.legend(fontsize=6, loc="upper right", frameon=False)
    fig.suptitle("Per-language BPB vs fraction of run — blue = languages the cell trains on, grey = unseen",
                 y=1.0)
    fig.tight_layout()
    S.save(fig, out_dir / "bpb_curves.png", dpi=120)


def generate_readme(pool: str, out_dir: Path, t: pd.DataFrame, agg: pd.DataFrame, dd: pd.DataFrame) -> None:
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    bullets, blocks = [], []
    if not agg.empty:
        for trained, lab in ((True, "trained"), (False, "never-trained")):
            g = agg[agg["trained"] == trained].sort_values("k")
            if g.empty:
                continue
            k1, kmax = g.iloc[0], g.iloc[-1]
            bullets.append(
                f"- **{lab} languages ({int(k1.n)} (L, language) cases)** — one {LADDER_SIZES[1]} rung plus the "
                f"pooled exponent predicts the reference within {fmt(100 * k1.transfer, 1)} % (median); "
                f"with every proxy rung, transferred {fmt(100 * kmax.transfer, 1)} % vs own fit "
                f"{fmt(100 * kmax.own, 1)} % vs largest proxy as is {fmt(100 * kmax.last, 1)} %.")
        ap = t.groupby("L")["alpha_pooled"].first()
        bullets.append("- **Pooled exponent α by L**: " + ", ".join(f"L{L} {fmt(v, 3)}" for L, v in ap.items()) + ".")
        rows = [[("trained" if r.trained else "never trained"), int(r.k), fmt(r.transfer, 3), fmt(r.own, 3),
                 fmt(r.last, 3), int(r.n)] for r in agg.itertuples()]
        blocks += ["**Median |relative error| of the reference's BPB** (k = rungs measured for the held-out language):",
                   md_table(["languages", "k", "transferred α", "own fit", "largest proxy", "n"], rows),
                   f"![Transfer]({stage}/{pool}/rq5_transfer.png)"]
    if not dd.empty:
        rows = [[r.population, r.proxy_size, fmt(r.da), int(r.cells)] for r in dd.itertuples()]
        blocks += ["**Decision transfer** (rq06's final-checkpoint agreement, mean over interventions and L, "
                   "on the languages both levels train vs the languages neither does):",
                   md_table(["population", "proxy", "agreement", "cells"], rows)]
    blocks.append(f"![BPB curves]({stage}/{pool}/bpb_curves.png)")
    readme = OUT_ROOT / "README.md"
    gen = f"analyze.py --pool {pool}"
    replace_block(readme, "highlight", "## Highlighted result\n\n" + "\n".join(bullets), gen)
    replace_block(readme, "results", "## Results\n\n"
                  + f"Numbers from the `{pool}` pool. Regenerate with "
                  f"`python analysis/rq10_language_transfer/analyze.py --pool {pool}`.\n\n"
                  + "\n\n".join(blocks), gen)
    print(f"Wrote auto README blocks → {readme}")


def main(pool: str, out_dir: Path) -> None:
    df = ladder_frame(pool)
    fin = finals(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pool '{pool}': {df['model'].nunique()} cells")
    t = transfer(fin)
    t.to_csv(out_dir / "rq5_transfer.csv", index=False)
    agg = summarise(t) if not t.empty else pd.DataFrame()
    agg.to_csv(out_dir / "rq5_transfer_summary.csv", index=False)
    print(f"Wrote → {out_dir / 'rq5_transfer.csv'} ({len(t)} rows, {t['task'].nunique() if not t.empty else 0} languages)")
    if not t.empty:
        plot_transfer(t, agg, out_dir)
    plot_bpb_curves(df, out_dir)
    # decision transfer: rq06's agreement on never-trained languages
    stage = load_pools()[pool].get("stage", "pretraining")
    src = PROXY_PREDICTIVITY / stage / pool / "intervention_da.csv"
    dd = pd.DataFrame()
    if src.is_file():
        d = pd.read_csv(src)
        d = d[(d["frac"] == 1.0) & d["population"].isin(["bpb_trained", "bpb_untrained"])]
        dd = (d.groupby(["population", "proxy_size"]).agg(da=("decision_acc", "mean"), cells=("decision_acc", "size"))
              .reset_index())
        dd = dd.iloc[[i for s in size_order(dd["proxy_size"]) for i in dd.index[dd["proxy_size"] == s]]]
    facts = {"rq5": {"summary": agg.round(3).to_dict("records"),
                     "alpha_pooled_by_L": t.groupby("L")["alpha_pooled"].first().round(3).to_dict() if not t.empty else {},
                     "alpha_own_spread_by_L": {int(k): v for k, v in t[t.k == 1].groupby("L")["alpha_own"]
                                               .agg(["median", "std"]).round(3).to_dict("index").items()} if not t.empty else {},
                     "n_languages": int(t["task"].nunique()) if not t.empty else 0,
                     "references": t.groupby("L")["reference_size"].first().to_dict() if not t.empty else {},
                     "decision_transfer": dd.round(3).to_dict("records")}}
    (out_dir / "facts.json").write_text(json.dumps(facts, indent=1, default=str))
    generate_readme(pool, out_dir, t, agg, dd)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Ladder pool from configs/models.json (default: {CANONICAL})")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    main(args.pool, OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool)
