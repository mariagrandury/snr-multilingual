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
the 1.7B reference. rq02's own ρ / τ_b of the proxy ranking against the
reference's are drawn beside the surrogates for scale only: they are computed
against the reference, so they are not surrogates but the ceiling.

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
    finetasks_surrogates_scatter.png / .csv   DA-size against every surrogate, one panel each, coloured by proxy size:
                             FineTasks' four, our gate, rq01's ρ and R², rq03's SNRs, and the
                             reference-free trajectory statistics of the proxy's own run (gain_half,
                             gain_last, tau_stable_from, window_tau, late_slope, rank_var, DA-ckpt at 50 / 90 %)
    finetasks_surrogates_rank.csv   every surrogate ranked by its Spearman ρ with DA-size over the cells
    finetasks_criteria.png   (a) share of tasks passing each criterion per size;
                             (b) Spearman of each statistic with DA-size per size,
                             the way rq04 scores SNR variants; (c) DA-size at each
                             proxy for tasks passing / failing the composite;
                             (d) = (a) with our above_66_either / above_66_both
                             shares beside it; (e) = (b) with rq03's 22 SNR
                             definitions behind it; (f) the FineTasks picks present
                             in our registry: their verdict under our criteria and
                             our gate, originals and cloze twins
    python analysis/rq04_surrogates/finetasks_criteria.py --pool predictivity
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
from analysis.paths import DECISION_ACCURACY, SURROGATES  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask, task_chance, task_n_items  # noqa: E402
from analysis.utils import (  # noqa: E402
    MIN_PAIRS, TARGET_SIZE, agreement_measures, ladder_frame, on_shared_grid, size_order)

OUT_ROOT = SURROGATES
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
        base = task_chance(task)
        n_std = (final.max() - base) / final.std(ddof=1) if final.std(ddof=1) > 0 else np.inf
        # ordering: τ_a between the run rankings at consecutive tenths in the second half
        def tau(c0, c1):
            both = wide[[c0, c1]].dropna()
            return agreement_measures(both[c0].to_numpy(), both[c1].to_numpy())["tau_a"] if len(both) >= MIN_PAIRS else np.nan

        def consecutive(start):
            cs = [c for c in wide.columns if c / 10 >= start]
            vals = [tau(c0, c1) for c0, c1 in zip(cs, cs[1:])]
            return float(np.nanmean(vals)) if vals and not all(np.isnan(vals)) else np.nan
        last = wide.columns[-1]
        half = min(wide.columns, key=lambda c: abs(c - 5))
        c90 = min(wide.columns[:-1], key=lambda c: abs(c - 9))
        means = wide.mean(axis=0)
        # the earliest tenth from which every later tenth's ranking is within τ_a ≥ 0.8 of the final (1 = never)
        ok = [tau(c, last) >= 0.8 for c in wide.columns[:-1]]
        stable = next((i for i in range(len(ok)) if all(ok[i:])), None)
        late = [c for c in wide.columns if c >= 8]
        win = [tau(c0, c1) for i, c0 in enumerate(late) for c1 in late[i + 1:]]

        def r2(r):     # rq01's trajectory fit, score ~ a + b log10(tokens), on one run of the proxy alone
            r = r.dropna(); x, y = np.log10(r.index.to_numpy(float)), r.to_numpy(float)
            b_, a_ = np.polyfit(x, y, 1); ss = ((y - y.mean()) ** 2).sum()
            return 1 - ((y - b_ * x - a_) ** 2).sum() / ss if ss > 0 else np.nan
        rows.append({"task": task, "size": size, "runs": int(len(final)), "tenths": int(wide.shape[1]),
                     "monotonicity": mono, "snr_finetasks": snr, "n_std": n_std,
                     "ordering": consecutive(ORDER_FROM),                       # theirs: consecutive tenths from 50 %
                     "ordering_late": consecutive(0.8),                         # consecutive tenths in the noise window
                     "tau_half_final": tau(half, last) if half != last else np.nan,   # the 50 % ranking against the final
                     "tau_first_final": tau(wide.columns[0], last),             # the first tenth against the final
                     "final_mean": final.mean(), "random_baseline": base, "margin": final.mean() - base,
                     "cross_run_std": final.std(ddof=1),
                     "gain_half": means[last] - means[half] if half != last else np.nan,   # how much the task still moves late
                     "gain_last": means[last] - means[c90],
                     "tau_stable_from": wide.columns[stable] / 10 if stable is not None else 1.0,
                     "window_tau": float(np.nanmean(win)) if win and not all(np.isnan(win)) else np.nan,
                     "late_slope": np.polyfit(wide.columns[-3:], means.iloc[-3:], 1)[0],   # per tenth of the run
                     "r2_traj_own": np.nanmean([r2(r) for _, r in wide.iterrows() if r.notna().sum() >= 3]),
                     "rank_var": wide[[c for c in wide.columns if c >= 5]].rank(axis=0).var(axis=1, ddof=1).mean()})
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


def our_snr(pool: str, sizes: list) -> pd.DataFrame:
    """rq03's 22 SNR definitions per (task, size), long: task, size, variant, snr."""
    from analysis.paths import NOISE_AND_SNR
    stage = load_pools()[pool].get("stage", "pretraining")
    t = pd.read_csv(NOISE_AND_SNR / stage / pool / "snr_variants_per_task.csv")
    rows = []
    for s in sizes:
        cols = [c for c in t.columns if c.startswith("snr_") and c.endswith(f"_{s}")]
        for c in cols:
            rows.append(pd.DataFrame({"task": t["task"], "size": s, "variant": c[len("snr_"):-len(f"_{s}")], "snr": t[c]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["task", "size", "variant", "snr"])


def reliable_shares(pool: str, sizes: list, out: pd.DataFrame) -> dict:
    """Per size, the share of the tasks in `out` that clear our cuts AT THAT
    PROXY: DA-size(proxy → reference) ≥ 0.66 or the median DA-ckpt of the
    proxy's run ≥ 0.66 (`either`), and both (`both`) — the per-size reading of
    `reliable_tasks.py`'s cuts, so the lines can move with the proxy."""
    stage = load_pools()[pool].get("stage", "pretraining")
    da = pd.read_csv(DECISION_ACCURACY / stage / pool / "da_per_task.csv")
    da = (da[da["axes"] == "multi-axis"] if "axes" in da.columns else da).set_index("task")
    shares = {}
    for s in sizes:
        tasks = out.loc[out["size"] == s, "task"]
        size_col = f"decision_acc_size_{s}"
        ckpt_cols = [c for c in da.columns if c.startswith("decision_acc_ckpt_f") and c.endswith(f"_{s}")]
        d = da.reindex(tasks)
        ok_size = (d[size_col] >= 0.66).mean() if size_col in d.columns else np.nan     # no DA-size at the reference
        ok_ckpt = (d[ckpt_cols].median(axis=1) >= 0.66).mean() if ckpt_cols else np.nan
        shares[s] = (ok_size, ok_ckpt)
    return shares


def reference_statistics(pool: str, out: pd.DataFrame) -> pd.DataFrame:
    """Per (task, size): rq02's Spearman ρ and Kendall τ_b of the proxy's ranking against the
    reference's (`agreement_per_cell.csv`) and rq01's ρ of score with size (`scaling_regimes.csv`),
    beside DA-size — the statistics a selection rule cannot compute without the reference (rq02)
    or without the ladder (rq01), drawn for scale."""
    from analysis.paths import SCALING_PREDICTABILITY
    stage = load_pools()[pool].get("stage", "pretraining")
    cells = pd.read_csv(DECISION_ACCURACY / stage / pool / "agreement_per_cell.csv")[["task", "size", "rho", "tau_b"]]
    reg = pd.read_csv(SCALING_PREDICTABILITY / "pretraining" / "predictivity_all" / "scaling_regimes.csv")[["task", "rho_size", "r2_trajectory"]]
    return (out[["task", "size", "da_size"]].merge(cells, on=["task", "size"], how="left").merge(reg, on="task", how="left")
            .rename(columns={"rho": "rho_ranking", "tau_b": "tau_ranking"}))


def rho_by_size(out: pd.DataFrame, st: str, proxies: list) -> list:
    return [out[out["size"] == s][[st, "da_size"]].replace([np.inf, -np.inf], np.nan).dropna().corr(method="spearman").iloc[0, 1]
            for s in proxies]


def figure(out: pd.DataFrame, overlap: pd.DataFrame | None, path: Path, pool: str) -> None:
    sizes = size_order(out["size"].unique())
    proxies = [s for s in sizes if s != TARGET_SIZE]
    xs, xp = np.arange(len(sizes)), np.arange(len(proxies))
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.4))
    crit = [("pass_mono", "monotone (ρ ≥ 0.5)", S.RAMP[0]), ("pass_snr", "SNR > 20", S.RAMP[1]),
            ("pass_nonrandom", "non-random (> 3 std)", S.RAMP[2]), ("pass_finetasks", "all three", S.INK)]
    share = out.groupby("size")[[c for c, _, _ in crit]].mean().reindex(sizes)
    gate = out.groupby("size")["gate"].mean().reindex(sizes)

    def pass_bars(ax, title):
        for k, (col, lab, c) in enumerate(crit):
            ax.bar(xs + (k - 1.5) * .19, share[col], .18, color=c, label=lab)
        ax.plot(xs, gate, color=S.SERIES[1], marker="o", ms=4, lw=1.2, label="our gate (Wilson, ≥ ½ of runs)")
        ax.set_xticks(xs); ax.set_xticklabels(sizes); ax.set_ylim(0, 1); ax.set_ylabel("share of benchmark tasks")
        ax.set_title(title, loc="left", fontsize=8.5); ax.grid(color=S.GRID, lw=.6, axis="y"); S.clean(ax)

    # (a) pass shares per size
    pass_bars(axes[0, 0], "(a) who passes FineTasks' criteria, per size")
    axes[0, 0].legend(fontsize=6, frameon=False)
    # (b) each criterion as a surrogate of DA-size
    b = axes[0, 1]
    for st, col in zip(STATS, S.RAMP):
        b.plot(xp, rho_by_size(out, st, proxies), marker="o", ms=4, color=col, label=LABEL[st].split(" (")[0])
    b.axhline(0, color=S.MUTED, lw=.8, ls=":")
    b.set_xticks(xp); b.set_xticklabels(proxies); b.set_ylim(-0.5, 0.8); b.set_ylabel("Spearman ρ with DA-size, across tasks")
    b.set_title("(b) each criterion as a surrogate of DA-size", loc="left", fontsize=8.5)
    b.legend(fontsize=6, frameon=False); b.grid(color=S.GRID, lw=.6); S.clean(b)
    # (c) DA-size for passers / failers of the composite
    c = axes[0, 2]
    for k, (flag, lab, col) in enumerate([(True, "passes all three", S.INK), (False, "fails one", S.MUTED)]):
        vals = [out[(out["size"] == s) & (out["pass_finetasks"] == flag)]["da_size"].dropna() for s in proxies]
        pos = xp + (k - .5) * .32
        c.boxplot(vals, positions=pos, widths=.28, patch_artist=True, showfliers=False, medianprops=dict(color=S.SURFACE, lw=1.2),
                  boxprops=dict(fc=col, ec=col, alpha=.85), whiskerprops=dict(color=col), capprops=dict(color=col))
        for p_, v in zip(pos, vals):
            c.text(p_, 0.12, str(len(v)), ha="center", fontsize=6, color=col)
        c.plot([], [], color=col, lw=6, label=lab)
    c.axhline(0.5, color=S.MUTED, lw=.8, ls=":")
    c.set_xticks(xp); c.set_xticklabels(proxies); c.set_ylim(0.1, 1); c.set_ylabel(f"DA-size, proxy → {TARGET_SIZE} (multi-axis)")
    c.set_title("(c) does passing predict the reference ranking?", loc="left", fontsize=8.5)
    c.legend(fontsize=6, frameon=False, loc="upper left"); c.grid(color=S.GRID, lw=.6, axis="y"); S.clean(c)
    # (d) the pass shares again, with our reliable-task cuts beside them
    d = axes[1, 0]
    pass_bars(d, "(d) FineTasks' criteria against our two DA cuts, per size")
    rs = reliable_shares(pool, sizes, out)
    d.plot(xs, [rs[s][0] for s in sizes], color="#b3261e", marker="s", ms=4, lw=1.4, label=f"DA-size(proxy → {TARGET_SIZE}) ≥ 0.66")
    d.plot(xs, [rs[s][1] for s in sizes], color="#b3261e", marker="s", ms=4, lw=1.4, ls="--", label="median DA-ckpt of the size's own run ≥ 0.66")
    d.legend(fontsize=6, frameon=False)
    # (e) the surrogates again, with rq03's 22 SNR definitions behind them
    e = axes[1, 1]
    snr = our_snr(pool, proxies)
    if len(snr):
        m = snr.merge(out[["task", "size", "da_size"]], on=["task", "size"])
        best = {}
        for v, g in m.groupby("variant"):
            rho = [g[g["size"] == s][["snr", "da_size"]].replace([np.inf, -np.inf], np.nan).dropna().corr(method="spearman").iloc[0, 1]
                   for s in proxies]
            e.plot(xp, rho, color=S.GRID, lw=.8, zorder=1)
            best[v] = np.nanmean(rho)
        top = max(best, key=best.get)
        g = m[m["variant"] == top]
        e.plot(xp, [g[g["size"] == s][["snr", "da_size"]].replace([np.inf, -np.inf], np.nan).dropna().corr(method="spearman").iloc[0, 1]
                    for s in proxies], color=S.SERIES[2], lw=1.6, marker="o", ms=4, zorder=4, label=f"best of our 22 SNR definitions ({top})")
        e.plot([], [], color=S.GRID, lw=.8, label=f"the other {len(best) - 1} SNR definitions (rq03)")
    for st, col in zip(STATS, S.RAMP):
        e.plot(xp, rho_by_size(out, st, proxies), marker="o", ms=4, color=col, label=LABEL[st].split(" (")[0], zorder=3)
    # the reference-side statistics, for scale: rq02's own rank correlations of the proxy ranking with the
    # reference's (computed against the reference: the ceiling, not a surrogate) and rq01's ρ of score with size
    ref = reference_statistics(pool, out)
    for st, col, ls, lab in (("rho_ranking", "#b3261e", "-", "rq02 ρ vs the reference (needs the reference: ceiling)"),
                             ("tau_ranking", "#b3261e", "--", "rq02 τ_b vs the reference (needs the reference: ceiling)"),
                             ("rho_size", "#7a5195", "-", "rq01 Spearman ρ, score vs size")):
        e.plot(xp, [ref[(ref["size"] == s)][[st, "da_size"]].dropna().corr(method="spearman").iloc[0, 1] for s in proxies],
               marker="s", ms=3.5, color=col, ls=ls, lw=1.2, label=lab, zorder=3)
    e.axhline(0, color=S.MUTED, lw=.8, ls=":")
    e.set_xticks(xp); e.set_xticklabels(proxies); e.set_ylim(-0.45, 1.0); e.set_ylabel("Spearman ρ with DA-size, across tasks")
    e.set_title("(e) FineTasks' criteria and our SNR definitions as surrogates", loc="left", fontsize=8.5)
    e.legend(fontsize=6, frameon=False, ncol=2); e.grid(color=S.GRID, lw=.6); S.clean(e)
    # (f) FineTasks' own picks under our criteria at 1B
    f = axes[1, 2]
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
            xf = np.arange(len(summ))
            f.bar(xf - .2, summ["finetasks"], .38, color=S.RAMP[3], label="pass FineTasks' three criteria (our ladder, 1B)")
            f.bar(xf + .2, summ["gate"], .38, color=S.SERIES[1], label="pass our gate at 1B")
            for i, (n, v, fs, gs) in enumerate(zip(summ["n"], summ["da"], summ["finetasks"], summ["gate"])):
                f.text(i - .2, fs + .02, f"{fs:.0%}", ha="center", va="bottom", fontsize=7, color=S.RAMP[3])
                f.text(i + .2, gs + .02, f"{gs:.0%}", ha="center", va="bottom", fontsize=7, color=S.SERIES[1])
            f.set_xticks(xf)
            f.set_xticklabels([f"FineTasks picks, {w}\n{int(n)} tasks · mean DA-size {v:.2f}"
                               for w, n, v in zip(summ.index, summ["n"], summ["da"])], fontsize=7.5)
    f.set_ylim(0, 1.3); f.set_yticks([0, .25, .5, .75, 1]); f.set_ylabel("share")
    f.set_title("(f) FineTasks' 96 picks that exist here, at 1B", loc="left", fontsize=8.5)
    f.legend(fontsize=7, frameon=False, loc="upper left"); f.grid(color=S.GRID, lw=.6, axis="y"); S.clean(f)
    top = G._header(fig, "FineTasks' selection criteria on the ladder, judged by the reference they cannot see",
                    f"Per (benchmark task, size), the {out['runs'].max()} design variants of `{pool}` along the ten "
                    f"evaluated tenths: monotonicity = mean over runs of Spearman ρ(tenth, score); SNR = mean final "
                    f"score / mean over tenths of the score's std ACROSS runs (their noise is our signal); non-random = "
                    f"(best final − 1/n_options) / final std; ordering = mean Kendall τ_a between consecutive tenths "
                    f"from {ORDER_FROM:.0%} on. Thresholds ρ ≥ {MONO_MIN}, SNR > {SNR_MIN:.0f}, margin > {NSTD_MIN:.0f} std, "
                    f"as published. (b), (c), (e) judge the criteria by DA-size against the {TARGET_SIZE} final, which needs "
                    f"the reference; (d) adds the share of tasks clearing our DA cuts at that size (DA-size against the reference ≥ 0.66, solid; median DA-ckpt of the size's own run ≥ 0.66, dashed); (e) adds "
                    f"rq03's 22 SNR definitions scored the same way and, for scale, rq02's own ρ and τ_b of the proxy ranking against "
                    f"the reference's (computed against the reference, so not surrogates but the ceiling) and rq01's ρ of score with size; (f) the FineTasks picks with a counterpart in our "
                    f"registry (`finetasks_overlap.csv`), originals and their cloze twins. No gate applied except where named.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def surrogate_table(pool: str, out: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, proxy size): DA-size and every surrogate — FineTasks'
    four, our gate (0/1), rq03's SNR definitions, rq01's ρ of score with size and
    trajectory R² (pooled over every size, the reference included), the same fit on
    the proxy's runs alone, the trajectory statistics of the proxy's own run, rq02's DA-ckpt
    at 50 / 90 % and rq02's own ρ (the ceiling: it needs the reference) — for the
    scatter figure and its CSV."""
    proxies = [s for s in size_order(out["size"].unique()) if s != TARGET_SIZE]
    run_side = STATS + ["ordering_late", "tau_half_final", "tau_first_final", "margin", "cross_run_std",
                        "gain_half", "gain_last", "tau_stable_from", "window_tau", "late_slope", "rank_var", "r2_traj_own"]
    t = out[out["size"].isin(proxies)][["task", "size", "da_size", "gate"] + run_side].copy()
    snr = our_snr(pool, proxies)
    if len(snr):
        wide = snr.pivot_table(index=["task", "size"], columns="variant", values="snr").add_prefix("snr_").reset_index()
        t = t.merge(wide, on=["task", "size"], how="left")
    ref = reference_statistics(pool, out)[["task", "size", "rho_ranking", "tau_ranking", "rho_size", "r2_trajectory"]]
    t = t.merge(ref, on=["task", "size"], how="left")
    # rq03's relative signal and noise at the proxy; rq02's DA between the two rungs below the proxy and the
    # median DA-ckpt of the proxy's own run: consistency a developer can measure without the reference
    stage = load_pools()[pool].get("stage", "pretraining")
    from analysis.paths import NOISE_AND_SNR
    snr_tab = pd.read_csv(NOISE_AND_SNR / stage / pool / "snr_variants_per_task.csv").set_index("task")
    da = pd.read_csv(DECISION_ACCURACY / stage / pool / "da_per_task.csv")
    da = (da[da["axes"] == "multi-axis"] if "axes" in da.columns else da).set_index("task")
    order = size_order(list(proxies))
    extra = []
    for s in proxies:
        prev = order[order.index(s) - 1] if order.index(s) else None
        sub = t[t["size"] == s][["task"]].copy()
        sub["signal_rel_std"] = sub["task"].map(snr_tab.get(f"signal_rel_std_{s}", pd.Series(dtype=float)))
        sub["noise_rel_std"] = sub["task"].map(snr_tab.get(f"noise_rel_std_{s}", pd.Series(dtype=float)))
        sub["da_prev_rung"] = sub["task"].map(da.get(f"decision_acc_size_{prev}_to_{s}", pd.Series(dtype=float))) if prev else np.nan
        ck = [c for c in da.columns if c.startswith("decision_acc_ckpt_f") and c.endswith(f"_{s}")]
        sub["da_ckpt_median"] = sub["task"].map(da[ck].median(axis=1)) if ck else np.nan
        for f in ("f50", "f90"):
            sub[f"da_ckpt_{f}"] = sub["task"].map(da.get(f"decision_acc_ckpt_{f}_{s}", pd.Series(dtype=float)))
        sub["size"] = s
        extra.append(sub)
    t = t.merge(pd.concat(extra, ignore_index=True), on=["task", "size"], how="left")
    t["n_items"] = t["task"].map(task_n_items)
    return t


def figure_scatter(t: pd.DataFrame, path: Path, pool: str) -> None:
    """DA-size against each surrogate in the style of `agreement_correlation.png`:
    one point per (task, proxy size) coloured by the proxy, Pearson r, Spearman
    ρ and the cell count in the corner."""
    sizes = size_order(t["size"].unique())
    colour = dict(zip(sizes, S.RAMP))

    def spearman(c):
        g = t[[c, "da_size"]].replace([np.inf, -np.inf], np.nan).dropna()
        return abs(g[c].corr(g["da_size"], method="spearman")) if len(g) > 2 else 0

    snr_cols = [c for c in t.columns if c.startswith("snr_") and c != "snr_finetasks"]
    best = sorted(snr_cols, key=lambda c: -spearman(c))[:3]
    panels = [("monotonicity", "FineTasks monotonicity (mean ρ step–score)", "linear"),
              ("snr_finetasks", "FineTasks SNR (mean final / cross-run std)", "log"),
              ("n_std", "FineTasks non-random (best margin / final std)", "log"),
              ("ordering", "FineTasks ordering (mean τ_a, consecutive tenths)", "linear"),
              ("gate", "our gate at the proxy (0 at chance, 1 above)", "linear"),
              ("rho_size", "rq01 Spearman ρ, score vs model size", "linear")] + \
             [(c, f"rq03 SNR {c[len('snr_'):]}", "log") for c in best] + \
             [("rho_ranking", "rq02 ρ vs the reference (needs the reference: ceiling)", "linear"),
              ("ordering_late", "ordering in the noise window (mean τ_a, tenths ≥ 80 %)", "linear"),
              ("tau_half_final", "τ_a: ranking at 50 % of the run vs the final", "linear"),
              ("tau_first_final", "τ_a: ranking at 10 % of the run vs the final", "linear"),
              ("da_ckpt_median", "median DA-ckpt of the proxy's own run", "linear"),
              ("da_prev_rung", "DA between the two rungs below (prev → proxy)", "linear"),
              ("signal_rel_std", "rq03 relative signal at the proxy (std / mean across variants)", "log"),
              ("noise_rel_std", "rq03 relative checkpoint noise at the proxy", "log"),
              ("cross_run_std", "cross-variant std of the final score", "log"),
              ("margin", "final mean score − 1/n_options", "linear"),
              ("n_items", "items in the task", "log"),
              ("r2_trajectory", "rq01 median trajectory R², all sizes (incl. ref.)", "linear"),
              ("r2_traj_own", "mean trajectory R² of the proxy's runs", "linear"),
              ("gain_half", "mean final − mean at 50 %", "linear"),
              ("gain_last", "mean final − mean at 90 %", "linear"),
              ("tau_stable_from", "run fraction from which τ_a(t, final) ≥ 0.8", "linear"),
              ("window_tau", "mean pairwise τ_a, tenths ≥ 80 %", "linear"),
              ("da_ckpt_f50", "rq02 DA-ckpt, 50 % vs final", "linear"),
              ("da_ckpt_f90", "rq02 DA-ckpt, 90 % vs final", "linear"),
              ("late_slope", "slope of mean score, last three tenths", "linear"),
              ("rank_var", "rank variance per run, tenths ≥ 50 %", "linear")]
    ncol = 5
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.1 * nrow + 0.9), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (col, lab, scale) in zip(axes.ravel(), panels):
        g = t[["size", col, "da_size"]].replace([np.inf, -np.inf], np.nan).dropna()
        if scale == "log":
            g = g[g[col] > 0]
        for sz in sizes:
            h = g[g["size"] == sz]
            x = h[col] + (rng.uniform(-.04, .04, len(h)) if col == "gate" else 0)
            ax.scatter(x, h["da_size"], s=8, color=colour[sz], alpha=.55, lw=0, label=f"{sz} proxy ({len(h)} cells)")
        if scale == "log":
            ax.set_xscale("log")
            ax.set_xlim(g[col].quantile(.01) / 1.5, g[col].quantile(.99) * 1.5)   # a few extreme values would flatten the cloud
        if col == "r2_traj_own":
            ax.set_xlim(g[col].quantile(.01), 1.02)                                # a few flat runs have R² ≪ 0
        r, rho = g[col].corr(g["da_size"]), g[col].corr(g["da_size"], method="spearman")
        ax.text(0.03, 0.97, f"Pearson r = {r:.3f}\nSpearman ρ = {rho:.3f}\n{len(g)} cells", transform=ax.transAxes,
                va="top", ha="left", fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", fc=S.SURFACE, ec=S.GRID, lw=.6))
        ax.axhline(0.5, color=S.MUTED, lw=.8, ls=":")
        ax.set_xlabel(lab, fontsize=7); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"DA-size, proxy → {TARGET_SIZE}")
    axes.ravel()[9].legend(fontsize=6, frameon=False, loc="lower right")
    top = G._header(fig, "Decision accuracy against each surrogate, one point per (task, proxy size)",
                    f"y: DA-size of the proxy against the {TARGET_SIZE} final over the multi-axis pairs (no gate applied: "
                    f"the gate is one of the surrogates), coloured by proxy size. x: FineTasks' four criteria on the "
                    f"proxy's runs; our gate at the proxy; rq01's ρ of score with size; the three of rq03's 22 SNR "
                    f"definitions with the largest |Spearman| here; rq02's own ρ of the proxy ranking with the "
                    f"reference's, computed against the reference, so not a surrogate but the ceiling. Rows 3–6: further candidates a developer can "
                    f"measure without the reference — the ordering statistic in the noise window, the ranking at 10 % and at "
                    f"50 % of the run against the final, the median DA-ckpt of the proxy's own run, DA between the two rungs "
                    f"below the proxy (the cross-size consistency already in hand), rq03's relative signal and noise, the "
                    f"cross-variant std and mean margin of the final score, the item count; then rq01's median R² of the trajectory fits "
                    f"(pooled over every size, the reference included — a reference-size quantity, rule 11) and the same fit on the "
                    f"proxy's own runs, "
                    f"how much the mean score still moves after 50 % and after 90 % of the run, the run fraction from which the "
                    f"ranking stays within τ_a ≥ 0.8 of the final (1 = never), the mean pairwise τ_a among the noise-window tenths, "
                    f"rq02's DA-ckpt at 50 % and 90 %, the slope of the mean score over the last three tenths and the mean variance "
                    f"of a run's rank over the tenths ≥ 50 %. Pearson r and Spearman ρ across cells "
                    f"in each corner (over every cell; the log axes are cut at the 1st and 99th percentiles for legibility, "
                    f"non-positive values dropped). Gate: `{pool}`.")
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
        f"`python analysis/rq04_surrogates/finetasks_criteria.py --pool {pool}`.",
        md_table(["size", "tasks", "monotone", "SNR > 20", "non-random", "all three", "our gate",
                  "DA-size pass vs fail", "ρ with DA-size"], rows),
        f"![FineTasks criteria]({stage}/{pool}/finetasks_criteria.png)",
        f"![DA against each surrogate]({stage}/{pool}/finetasks_surrogates_scatter.png)"])
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
    st = surrogate_table(args.pool, out)
    st.to_csv(out_dir / "finetasks_surrogates_scatter.csv", index=False)
    cand = [c for c in st.columns if c not in ("task", "size", "da_size")]
    rank = pd.DataFrame([{"surrogate": c, "cells": int(st[[c, "da_size"]].replace([np.inf, -np.inf], np.nan).dropna().shape[0]),
                          "spearman": st[[c, "da_size"]].replace([np.inf, -np.inf], np.nan).dropna().corr(method="spearman").iloc[0, 1]}
                         for c in cand]).sort_values("spearman", ascending=False)
    rank.to_csv(out_dir / "finetasks_surrogates_rank.csv", index=False)
    print(rank.round(3).to_string(index=False))
    figure_scatter(st, out_dir / "finetasks_surrogates_scatter.png", args.pool)
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, out_dir, out)
