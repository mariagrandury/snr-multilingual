"""FineTasks' four selection criteria, computed on our ladder and scored
against decision accuracy.

FineTasks (Kydlíček, Penedo et al., 2024; the FineWeb-2 paper) selects tasks
per language from 49 single-seed 1.5B recipes evaluated along training with
four criteria: monotonicity (mean over runs of Spearman ρ between step and
score, ≥ 0.5), low noise (mean final score over the runs divided by the mean
over steps of the score's std ACROSS runs — their "SNR" — > 20), non-random
performance (best run's margin over the random baseline > 3 × the final
cross-run std) and ordering consistency (Kendall τ_a of the run ranking
between consecutive steps after half the run; no threshold). None of the
four looks at a larger model, so the criteria can be computed on any one
size of the ladder and then judged by what they cannot see: DA-size against
the 1.7B reference.

Two things to hold in mind. Their "noise" is the spread across recipes at
one step, which the SNR framework (and rq03) calls SIGNAL; so their SNR is
1 / our relative signal, averaged over the run. And their runs are single
seeds, so consecutive-step τ carries within-run persistence (rq02's seed
null); ours does too, and is reported as the same number for comparability.

Population: the `predictivity` pool (the 18 design variants at each size,
grid seed, schemes A/B), benchmark tasks on the ten evaluated tenths (rule 3),
trained languages only (rule 2), parents only (rule 6). The gate is NOT
applied — non-random is one of the criteria under test — but the mask is
carried so the two verdicts can be compared.

    finetasks_criteria.csv   one row per (task, size): the four statistics, their
                             verdicts at FineTasks' thresholds, our gate, DA-size
    finetasks_criteria.png   (a) share of tasks passing each criterion per size;
                             (b) DA-size at each proxy for tasks passing / failing
                             the composite; (c) Spearman of each statistic with
                             DA-size per size, the way rq04 scores SNR variants;
                             (d) the 45 FineTasks picks present in our registry:
                             their verdict under our criteria and our gate
    python analysis/rq09_benchmark_design/finetasks_criteria.py --pool predictivity
"""

from __future__ import annotations

import argparse
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
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import BENCHMARK_DESIGN, DECISION_ACCURACY  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask, task_n_options  # noqa: E402
from analysis.utils import (  # noqa: E402
    MIN_PAIRS, TARGET_SIZE, agreement_measures, ladder_frame, on_shared_grid, size_order)

OUT_ROOT = BENCHMARK_DESIGN
# FineTasks' thresholds, as published: ρ ≥ 0.5, SNR > 20, margin > 3 × std; the
# ordering statistic is reported without one. Consecutive-step τ "starting
# after 15B tokens" of 30B = the second half of the run.
MONO_MIN, SNR_MIN, NSTD_MIN, ORDER_FROM = 0.5, 20.0, 3.0, 0.5
STATS = ["monotonicity", "snr_finetasks", "n_std", "ordering"]
LABEL = {"monotonicity": "monotonicity (mean Spearman ρ step–score)",
         "snr_finetasks": "FineTasks SNR (mean final / mean cross-run std)",
         "n_std": "non-random (best margin / final std)",
         "ordering": "ordering (mean τ between consecutive steps)"}
mpl.rcParams.update(S.RC)


def criteria(df: pd.DataFrame) -> pd.DataFrame:
    """The four statistics per (task, size) over the runs at that size."""
    b = df[(df["kind"] == "benchmark") & on_shared_grid(df)].copy()
    b["tenth"] = (b["frac"] * 10).round().astype(int)
    rows = []
    for (task, size), g in b.groupby(["task", "size"]):
        wide = g.pivot_table(index="model", columns="tenth", values="primary_score")
        wide = wide.dropna(axis=1, thresh=max(3, int(0.8 * len(wide))))     # a tenth most runs reached
        if wide.shape[0] < MIN_PAIRS or wide.shape[1] < 3:
            continue
        final = wide.iloc[:, -1].dropna()
        # monotonicity: per run, Spearman between the tenth and the score
        mono = np.nanmean([spearmanr(wide.columns, r.values, nan_policy="omit").statistic
                           for _, r in wide.iterrows() if r.notna().sum() >= 3])
        # their noise: std across runs at each tenth, averaged over the tenths
        std_steps = wide.std(axis=0, ddof=1).mean()
        snr = final.mean() / std_steps if std_steps > 0 else np.inf
        base = 1.0 / task_n_options(task)
        n_std = (final.max() - base) / final.std(ddof=1) if final.std(ddof=1) > 0 else np.inf
        # ordering: τ_a between the run rankings at consecutive tenths in the second half
        late = [c for c in wide.columns if c / 10 >= ORDER_FROM]
        taus = []
        for c0, c1 in zip(late, late[1:]):
            both = wide[[c0, c1]].dropna()
            if len(both) >= MIN_PAIRS:
                taus.append(agreement_measures(both[c0].to_numpy(), both[c1].to_numpy())["tau_a"])  # ties count for neither, as theirs
        rows.append({"task": task, "size": size, "runs": int(len(final)), "tenths": int(wide.shape[1]),
                     "monotonicity": mono, "snr_finetasks": snr, "n_std": n_std,
                     "ordering": float(np.nanmean(taus)) if taus else np.nan,
                     "final_mean": final.mean(), "random_baseline": base})
    out = pd.DataFrame(rows)
    out["pass_mono"] = out["monotonicity"] >= MONO_MIN
    out["pass_snr"] = out["snr_finetasks"] > SNR_MIN
    out["pass_nonrandom"] = out["n_std"] > NSTD_MIN
    out["pass_finetasks"] = out["pass_mono"] & out["pass_snr"] & out["pass_nonrandom"]
    return out


def join_ours(out: pd.DataFrame, pool: str) -> pd.DataFrame:
    """Our gate at the size and DA-size (proxy → reference, multi-axis) beside the criteria."""
    mask = load_mask(pool)
    out["gate"] = [float(mask.loc[t, s]) if mask is not None and t in mask.index and s in mask.columns
                   and pd.notna(mask.loc[t, s]) else np.nan for t, s in zip(out["task"], out["size"])]
    stage = load_pools()[pool].get("stage", "pretraining")
    da = pd.read_csv(DECISION_ACCURACY / stage / pool / "da_per_task.csv")
    if "axes" in da.columns:
        da = da[da["axes"] == "multi-axis"]
    da = da.set_index("task")
    out["da_size"] = [da[f"decision_acc_size_{s}"].get(t, np.nan) if f"decision_acc_size_{s}" in da.columns else np.nan
                      for t, s in zip(out["task"], out["size"])]
    return out


def figure(out: pd.DataFrame, overlap: pd.DataFrame | None, path: Path, pool: str) -> None:
    sizes = size_order(out["size"].unique())
    proxies = [s for s in sizes if s != TARGET_SIZE]
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.2))
    # (a) pass shares per size
    a = axes[0]
    share = out.groupby("size")[["pass_mono", "pass_snr", "pass_nonrandom", "pass_finetasks"]].mean().reindex(sizes)
    gate = out.groupby("size")["gate"].mean().reindex(sizes)
    x = np.arange(len(sizes))
    for k, (col, lab, c) in enumerate([("pass_mono", "monotone (ρ ≥ 0.5)", S.RAMP[0]), ("pass_snr", "SNR > 20", S.RAMP[1]),
                                       ("pass_nonrandom", "non-random (> 3 std)", S.RAMP[2]), ("pass_finetasks", "all three", S.INK)]):
        a.bar(x + (k - 1.5) * .19, share[col], .18, color=c, label=lab)
    a.plot(x, gate, color=S.SERIES[1], marker="o", ms=4, lw=1.2, label="our gate (Wilson, ≥ ½ of runs)")
    a.set_xticks(x); a.set_xticklabels(sizes); a.set_ylim(0, 1); a.set_ylabel("share of benchmark tasks")
    a.set_title("(a) who passes FineTasks' criteria, per size", loc="left", fontsize=8.5)
    a.legend(fontsize=6, frameon=False); a.grid(color=S.GRID, lw=.6, axis="y"); S.clean(a)
    # (b) DA-size for passers / failers of the composite
    b = axes[1]
    for k, (flag, lab, c) in enumerate([(True, "passes all three", S.INK), (False, "fails one", S.MUTED)]):
        vals = [out[(out["size"] == s) & (out["pass_finetasks"] == flag)]["da_size"].dropna() for s in proxies]
        pos = np.arange(len(proxies)) + (k - .5) * .32
        bp = b.boxplot(vals, positions=pos, widths=.28, patch_artist=True, showfliers=False,
                       medianprops=dict(color=S.SURFACE, lw=1.2), boxprops=dict(fc=c, ec=c, alpha=.85), whiskerprops=dict(color=c), capprops=dict(color=c))
        for p, v in zip(pos, vals):
            b.text(p, 0.12, str(len(v)), ha="center", fontsize=6, color=c)
        b.plot([], [], color=c, lw=6, label=lab)
    b.axhline(0.5, color=S.MUTED, lw=.8, ls=":")
    b.set_xticks(np.arange(len(proxies))); b.set_xticklabels(proxies); b.set_ylim(0.1, 1)
    b.set_ylabel(f"DA-size, proxy → {TARGET_SIZE} (multi-axis)")
    b.set_title("(b) does passing predict the reference ranking?", loc="left", fontsize=8.5)
    b.legend(fontsize=6, frameon=False, loc="upper left"); b.grid(color=S.GRID, lw=.6, axis="y"); S.clean(b)
    # (c) Spearman of each statistic with DA-size, per size
    c = axes[2]
    for st, col in zip(STATS, S.RAMP):
        rho = [out[(out["size"] == s)][[st, "da_size"]].replace([np.inf, -np.inf], np.nan).dropna().corr(method="spearman").iloc[0, 1]
               for s in proxies]
        c.plot(np.arange(len(proxies)), rho, marker="o", ms=4, color=col, label=LABEL[st].split(" (")[0])
    c.axhline(0, color=S.MUTED, lw=.8, ls=":")
    c.set_xticks(np.arange(len(proxies))); c.set_xticklabels(proxies); c.set_ylim(-0.5, 0.8)
    c.set_ylabel("Spearman ρ with DA-size, across tasks")
    c.set_title("(c) each criterion as a surrogate of DA-size", loc="left", fontsize=8.5)
    c.legend(fontsize=6, frameon=False); c.grid(color=S.GRID, lw=.6); S.clean(c)
    # (d) FineTasks' own picks under our criteria at 1B
    d = axes[3]
    if overlap is not None and len(overlap):
        at = out[out["size"] == "1B"].set_index("task")
        rows = []
        for _, r in overlap.dropna(subset=["our_task"]).iterrows():
            for which, t in (("original", r["our_task"]), ("rf twin", r.get("rf_task"))):
                if isinstance(t, str) and t in at.index:
                    rows.append({"which": which, "pass": bool(at.loc[t, "pass_finetasks"]), "gate": at.loc[t, "gate"] == 1,
                                 "da": at.loc[t, "da_size"]})
        o = pd.DataFrame(rows)
        if len(o):
            summ = o.groupby("which").agg(n=("pass", "size"), finetasks=("pass", "mean"), gate=("gate", "mean"),
                                          da=("da", "mean")).reindex(["original", "rf twin"]).dropna(how="all")
            x = np.arange(len(summ))
            d.bar(x - .2, summ["finetasks"], .38, color=S.INK, label="pass FineTasks' three criteria (ours, 1B)")
            d.bar(x + .2, summ["gate"], .38, color=S.SERIES[1], label="pass our gate at 1B")
            for i, (n, v) in enumerate(zip(summ["n"], summ["da"])):
                d.text(i, 0.95, f"{int(n)} tasks\nmean DA-size {v:.2f}", ha="center", va="top", fontsize=6.5)
            d.set_xticks(x); d.set_xticklabels([f"FineTasks picks,\n{w}" for w in summ.index], fontsize=7.5)
    d.set_ylim(0, 1); d.set_ylabel("share")
    d.set_title("(d) FineTasks' 96 picks that exist here, at 1B", loc="left", fontsize=8.5)
    d.legend(fontsize=6, frameon=False, loc="lower left"); d.grid(color=S.GRID, lw=.6, axis="y"); S.clean(d)
    top = G._header(fig, "FineTasks' selection criteria on the ladder, judged by the reference they cannot see",
                    f"Per (benchmark task, size), the {out['runs'].max()} design variants of `{pool}` along the ten "
                    f"evaluated tenths: monotonicity = mean over runs of Spearman ρ(tenth, score); SNR = mean final "
                    f"score / mean over tenths of the score's std ACROSS runs (their noise is our signal); non-random = "
                    f"(best final − 1/n_options) / final std; ordering = mean Kendall τ_a between consecutive tenths "
                    f"from {ORDER_FROM:.0%} on. Thresholds ρ ≥ {MONO_MIN}, SNR > {SNR_MIN:.0f}, margin > {NSTD_MIN:.0f} std, "
                    f"as published. (b)–(c) judge the criteria by DA-size against the {TARGET_SIZE} final, which needs "
                    f"the reference; (d) the FineTasks picks with a counterpart in our registry (`finetasks_overlap.csv`), "
                    f"originals and their cloze twins. No gate applied except where named.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, out_dir: Path, out: pd.DataFrame) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    sizes = size_order(out["size"].unique())
    rows = []
    for s in sizes:
        g = out[out["size"] == s]
        p, f = g[g["pass_finetasks"]]["da_size"].dropna(), g[~g["pass_finetasks"]]["da_size"].dropna()
        rho = {st: g[[st, "da_size"]].replace([np.inf, -np.inf], np.nan).dropna().corr(method="spearman").iloc[0, 1] for st in STATS}
        rows.append([s, len(g), f"{g['pass_mono'].mean():.0%}", f"{g['pass_snr'].mean():.0%}", f"{g['pass_nonrandom'].mean():.0%}",
                     f"{g['pass_finetasks'].mean():.0%}", f"{g['gate'].mean():.0%}",
                     f"{p.mean():.2f} [{len(p)}] vs {f.mean():.2f} [{len(f)}]" if len(p) and len(f) else "—",
                     " / ".join(f"{rho[st]:+.2f}" for st in STATS)])
    body = "\n\n".join([
        "## FineTasks' criteria on the ladder",
        f"FineTasks selects tasks with four statistics computed on single-seed runs at one size (monotonicity, a "
        f"cross-run SNR, a non-random margin, consecutive-step ordering). Computed here per (task, size) on the "
        f"`{pool}` variants and judged by DA-size against {TARGET_SIZE}, which the criteria never see. Columns: share "
        f"passing each criterion, all three, and our gate; mean DA-size of passers vs failers; Spearman of each "
        f"statistic with DA-size (monotonicity / SNR / non-random / ordering). Regenerate with "
        f"`python analysis/rq09_benchmark_design/finetasks_criteria.py --pool {pool}`.",
        md_table(["size", "tasks", "monotone", "SNR > 20", "non-random", "all three", "our gate",
                  "DA-size pass vs fail", "ρ with DA-size"], rows),
        f"![FineTasks criteria]({stage}/{pool}/finetasks_criteria.png)"])
    replace_block(OUT_ROOT / "README.md", "finetasks-criteria", body, f"finetasks_criteria.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    args = p.parse_args()
    out_dir = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    out_dir.mkdir(parents=True, exist_ok=True)
    out = join_ours(criteria(ladder_frame(args.pool)), args.pool)
    out.to_csv(out_dir / "finetasks_criteria.csv", index=False)
    ov_path = OUT_ROOT / "finetasks_overlap.csv"
    overlap = pd.read_csv(ov_path) if ov_path.exists() else None
    print(out.groupby("size")[["pass_mono", "pass_snr", "pass_nonrandom", "pass_finetasks", "gate"]].mean().round(2).to_string())
    for s in [x for x in size_order(out["size"].unique()) if x != TARGET_SIZE]:
        g = out[out["size"] == s]
        print(f"{s}: DA-size passers {g[g.pass_finetasks]['da_size'].mean():.3f} [{g.pass_finetasks.sum()}] "
              f"vs failers {g[~g.pass_finetasks]['da_size'].mean():.3f} [{(~g.pass_finetasks).sum()}]")
    figure(out, overlap, out_dir / "finetasks_criteria.png", args.pool)
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, out_dir, out)
