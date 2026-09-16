"""The paper's five research-question figures, one per RQ, from the ladder report.

Every number comes from the published per-checkpoint table (ladder_report.csv,
loaded through snr.download.ladder) and, for RQ3, from the rq02 SNR-variant
table that run_all_predictivity.sh writes. Nothing reads the cluster.

    RQ1 scaling        rq1_scaling.pdf       score vs N per family; R² / ρ of a log-N fit
    RQ2 prediction     rq2_early_small.pdf   agreement with the reference by proxy size x fraction of training
    RQ3 surrogates     rq3_surrogates.pdf    Spearman ρ of small-scale metrics with decision accuracy
    RQ4 interventions  rq4_interventions.pdf effect vs seed noise, and DA by proxy size, per intervention
    RQ5 transfer       rq5_transfer.pdf      leave-one-language-out scaling prediction

    python make_rq_figures.py            # all five + rq_facts.json
    python make_rq_figures.py rq2 rq5    # a subset
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (REPO / "src", REPO / "src" / "signal-and-noise", REPO / "documents" / "figures"):
    sys.path.insert(0, str(p))
import style as S  # noqa: E402  (documents/figures/style.py: the deck palette)
from snr.download.ladder import load_predictivity_eval_results  # noqa: E402
from analysis.utils import _is_parent_task, assign_language, benchmark_family  # noqa: E402
from analysis.rq00_acc_vs_flops.above_random import task_n_options  # noqa: E402
from pretrain.launch_trainings import cell_fineweb_subsets  # noqa: E402

OUT = HERE
ANALYSIS = REPO / "src" / "signal-and-noise" / "analysis"
SIZES = ["90M", "175M", "350M", "600M", "1B", "1.7B"]
NON_EMB = {"90M": 9.0e7, "175M": 1.75e8, "350M": 3.5e8, "600M": 6.0e8, "1B": 1.0e9, "1.7B": 1.7e9}
GRID_SEED = 1904
FRACS = [0.2, 0.4, 0.6, 0.8, 1.0]
MIN_ITEMS = 3
# One design decision per axis, read with the other axes at their baseline.
INTERVENTIONS = {
    "depth (deep vs shallow)":        dict(axis="arch",   levels=("deep", "shallow"), hold=("scheme", "A")),
    "language lists (A vs B)":        dict(axis="scheme", levels=("A", "B"),          hold=("arch", "deep")),
    "temperature (T=1 vs T=3)":       dict(axis="scheme", levels=("A", "AT3"),        hold=("arch", "deep")),
    "2nd language (ru vs zh)":        dict(axis="scheme", levels=("A", "ZH"),         hold=("arch", "deep")),
    "2nd language (ru vs es)":        dict(axis="scheme", levels=("A", "ES"),         hold=("arch", "deep")),
}
BPB, BENCH, UNTR = S.RAMP[3], S.SERIES[1], S.SERIES[2]
FACTS = {}

mpl_rc = {"font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 9.5, "legend.fontsize": 7.5,
          "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42}
matplotlib.rcParams.update(mpl_rc)


def size_order(sizes):
    return [s for s in SIZES if s in set(sizes)]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight", facecolor=S.SURFACE)
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf")


# --- data --------------------------------------------------------------------

def load():
    """Long frame of every healthy complete cell, with the fraction of the run
    each checkpoint sits at. Diverged and unfinished runs are dropped by the
    loader; off-trend cells are kept, as in every other analysis."""
    df = load_predictivity_eval_results(shared_grid=True)
    target = df.groupby("model")["step"].transform("max")
    df["frac"] = df["step"] / target
    return df


def finals(df):
    return df.loc[df.groupby(["model", "task"])["step"].idxmax()]


def at_fraction(df, f, tol=0.06):
    """Each (model, task)'s score at the saved checkpoint nearest fraction f."""
    d = df.assign(dist=(df["frac"] - f).abs())
    d = d.loc[d.groupby(["model", "task"])["dist"].idxmin()]
    return d[d["dist"] <= tol].drop(columns="dist")


def trained_tasks(L, scheme):
    try:
        return {"bpb_dclm"} | {f"bpb_{s}" for s in cell_fineweb_subsets(L, scheme)}
    except KeyError:                      # the scheme defines no list at this L
        return None


def population(sub, name, L, levels, axis):
    if name == "benchmark":
        return sub[sub["kind"] == "benchmark"]
    bpb = sub[(sub["kind"] == "bpb") & (sub["task"] != "bpb_macro")]
    schemes = levels if axis == "scheme" else ("A",)
    tr = [trained_tasks(L, s) for s in schemes]
    if any(t is None for t in tr):
        return bpb.iloc[0:0]
    if name == "bpb_trained":
        return bpb[bpb["task"].isin(set.intersection(*tr))]
    if name == "bpb_untrained":
        return bpb[~bpb["task"].isin(set.union(*tr))]
    raise ValueError(name)


def decision_table(df):
    """One row per (intervention, L, population, proxy size, fraction):
    the share of items on which the proxy's decision matches the reference's
    (largest size trained at both levels, final checkpoint)."""
    grid = df[df["seed"] == GRID_SEED]
    fin = finals(grid)
    at_f = {f: at_fraction(grid, f) for f in FRACS}
    rows = []
    for name, iv in INTERVENTIONS.items():
        axis, levels, (hcol, hval) = iv["axis"], iv["levels"], iv["hold"]
        sub_fin = fin[fin[hcol] == hval]
        for L in sorted(sub_fin["L"].unique()):
            for pop in ("bpb_trained", "bpb_untrained", "benchmark"):
                ref_piv = (population(sub_fin[sub_fin["L"] == L], pop, L, levels, axis)
                           .pivot_table(index=["size", "task"], columns=axis, values="primary_score"))
                if not set(levels) <= set(ref_piv.columns):
                    continue
                ref_piv = ref_piv.dropna(subset=list(levels))
                sizes = size_order(ref_piv.index.get_level_values("size"))
                if len(sizes) < 2:
                    continue
                ref = sizes[-1]
                ref_sign = np.sign(ref_piv.xs(ref, level="size")[levels[0]]
                                   - ref_piv.xs(ref, level="size")[levels[1]])
                ref_sign = ref_sign[ref_sign != 0]
                for f in FRACS:
                    sub = at_f[f]
                    sub = sub[(sub[hcol] == hval) & (sub["L"] == L)]
                    piv = (population(sub, pop, L, levels, axis)
                           .pivot_table(index=["size", "task"], columns=axis, values="primary_score"))
                    if not set(levels) <= set(piv.columns):
                        continue
                    piv = piv.dropna(subset=list(levels))
                    for s in sizes:
                        if s == ref and f == 1.0:
                            continue
                        if s not in piv.index.get_level_values("size"):
                            continue
                        p = piv.xs(s, level="size")
                        items = p.index.intersection(ref_sign.index)
                        if len(items) < MIN_ITEMS:
                            continue
                        sign = np.sign(p.loc[items, levels[0]] - p.loc[items, levels[1]])
                        rows.append({"intervention": name, "L": int(L), "population": pop,
                                     "proxy_size": s, "frac": f, "reference_size": ref,
                                     "n_items": int(len(items)),
                                     "da": float((sign == ref_sign.loc[items]).mean())})
    return pd.DataFrame(rows)


# --- RQ1: what scales predictably --------------------------------------------

def rq1(df):
    fin = finals(df[(df["seed"] == GRID_SEED) & (df["arch"] == "deep") & (df["scheme"] == "A")])
    fin = fin[(fin["size"] != "90M") & fin["task"].map(_is_parent_task)]
    rows = []
    for (task, L), g in fin.groupby(["task", "L"]):
        g = g.assign(N=g["size"].map(NON_EMB)).dropna(subset=["N"]).sort_values("N")
        if len(g) < 3:
            continue
        x, y = np.log10(g["N"].to_numpy()), g["primary_score"].to_numpy(float)
        b, a = np.polyfit(x, y, 1)
        ss_res = float(((y - (b * x + a)) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        rho = spearmanr(x, y).statistic if len(set(y)) > 1 else np.nan
        rows.append({"task": task, "L": L, "family": benchmark_family(task), "kind": g["kind"].iloc[0],
                     "n_rungs": len(g), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                     "rho": rho, "n_options": task_n_options(task) if g["kind"].iloc[0] == "benchmark" else np.nan})
    fits = pd.DataFrame(rows)
    fits = fits[fits["task"] != "bpb_macro"]
    fits.to_csv(OUT / "rq1_fits.csv", index=False)
    fam = fits.groupby("family").agg(r2=("r2", "median"), rho=("rho", "median"), n=("task", "size"),
                                     n_options=("n_options", lambda s: s.mode().iloc[0] if s.notna().any() else np.nan))
    fam = fam[fam["n"] >= 3].sort_values("r2")
    fam.to_csv(OUT / "rq1_families.csv")

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw={"width_ratios": [1, 1.25]})
    # (a) score above chance against size, family medians at L = 30
    ex = fin[(fin["L"] == 30) & (fin["kind"] == "benchmark")].copy()
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
        for task, t in g.groupby("task"):
            t = t.assign(N=t["size"].map(NON_EMB)).sort_values("N")
            a0.plot(t["N"], t["above"], color=col, lw=.5, alpha=.25)
        med = g.groupby("size")["above"].median()
        order = size_order(med.index)
        a0.plot([NON_EMB[s] for s in order], med[order], marker="o", ms=4, lw=2, color=col, label=famname)
    a0.axhline(0, color=S.INK, lw=.8, ls="--")
    a0.set_xscale("log"); a0.set_xticks([NON_EMB[s] for s in SIZES[1:]]); a0.set_xticklabels(SIZES[1:])
    a0.set_xlabel("non-embedding parameters"); a0.set_ylabel("accuracy above chance")
    a0.set_title("(a) benchmark families at L = 30", loc="left")
    a0.legend(frameon=False, loc="upper left"); a0.grid(color=S.GRID, lw=.6); a0.set_axisbelow(True)
    S.clean(a0)
    # (b) R² and ρ per family
    y = np.arange(len(fam))
    a1.hlines(y, 0, fam["r2"], color=S.GRID, lw=1.2, zorder=1)
    a1.scatter(fam["r2"], y, s=42, color=S.RAMP[2], zorder=3, label="R² of the log-N fit")
    a1.scatter(fam["rho"], y, s=42, marker="D", color=S.SERIES[1], zorder=3, label="Spearman ρ with N")
    a1.axvline(0, color=S.MUTED, lw=.8)
    labels = []
    for f, r in fam.iterrows():
        opt = "" if not np.isfinite(r["n_options"]) else f", {int(r['n_options'])}-way"
        labels.append(f"{f} (n={int(r['n'])}{opt})")
    a1.set_yticks(y); a1.set_yticklabels(labels, fontsize=7.5)
    a1.set_xlim(-1.05, 1.05); a1.set_xlabel("median over the family's (task, L) fits, 175M to 1.7B")
    a1.set_title("(b) how predictably each family scales", loc="left")
    a1.legend(frameon=False, loc="lower left"); a1.grid(axis="x", color=S.GRID, lw=.6); a1.set_axisbelow(True)
    S.clean(a1); a1.tick_params(length=0)
    fig.subplots_adjust(wspace=.55)
    save(fig, "rq1_scaling")
    FACTS["rq1"] = {"n_fits": int(len(fits)), "families": fam.round(3).reset_index().to_dict("records"),
                    "by_options": fits[fits.kind == "benchmark"].groupby("n_options")["r2"].median().round(3).to_dict()}
    return fits


# --- RQ2: how early and how small ---------------------------------------------

def rq2(dt):
    decisions = list(INTERVENTIONS)[:2]                      # the two planned axes
    core = dt[dt["intervention"].isin(decisions)]
    core.to_csv(OUT / "rq2_decisions.csv", index=False)
    agg = (core.groupby(["intervention", "population", "proxy_size", "frac"])
           .agg(da=("da", "mean"), cells=("da", "size"), items=("n_items", "sum"),
                refs=("reference_size", lambda s: "/".join(sorted(set(s)))))
           .reset_index())
    agg.to_csv(OUT / "rq2_early_small.csv", index=False)
    pops = [("bpb_trained", "per-language bits per byte"), ("benchmark", "benchmark tasks")]
    fig, axes = plt.subplots(len(decisions), 2, figsize=(9.2, 6.4), sharex=True)
    for i, dec in enumerate(decisions):
        for j, (pop, title) in enumerate(pops):
            ax = axes[i][j]
            g = agg[(agg["intervention"] == dec) & (agg["population"] == pop)]
            sizes = size_order(g["proxy_size"])
            mat = np.full((len(sizes), len(FRACS)), np.nan)
            cnt = np.zeros_like(mat)
            for _, r in g.iterrows():
                mat[sizes.index(r["proxy_size"]), FRACS.index(r["frac"])] = r["da"]
                cnt[sizes.index(r["proxy_size"]), FRACS.index(r["frac"])] = r["cells"]
            im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap=S.SEQ, aspect="auto")
            for a in range(len(sizes)):
                for b in range(len(FRACS)):
                    if np.isfinite(mat[a, b]):
                        ax.text(b, a, f"{mat[a, b]:.2f}\n({int(cnt[a, b])})", ha="center", va="center",
                                fontsize=6.8, color="white" if mat[a, b] > 0.7 else S.INK)
            ax.set_xticks(range(len(FRACS))); ax.set_xticklabels([f"{int(f*100)} %" for f in FRACS])
            ax.set_yticks(range(len(sizes))); ax.set_yticklabels(sizes)
            if i == 0:
                ax.set_title(title, loc="left")
            if j == 0:
                ax.set_ylabel(f"{dec}\nproxy size")
            if i == len(decisions) - 1:
                ax.set_xlabel("checkpoint, as a share of the proxy's own run")
            for sp in ("top", "right", "left", "bottom"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.025, pad=.02)
    cb.set_label("agreement with the reference's final decision (mean over L; cells in brackets)")
    cb.outline.set_visible(False)
    save(fig, "rq2_early_small")
    FACTS["rq2"] = {"table": agg.round(3).to_dict("records"),
                    "references": {f"{d}|{p}|L{L}": r for (d, p, L), r in
                                   core[core.frac == 1.0].groupby(["intervention", "population", "L"])["reference_size"].first().items()}}
    return agg


# --- RQ3: surrogates for decision accuracy ------------------------------------

def rq3(df, fits):
    """Metrics computable on the proxy alone against the proxy's decision accuracy
    versus the reference (DA-size, from rq01 through the rq02 table)."""
    v = pd.read_csv(ANALYSIS / "rq02_snr_definition/pretraining/predictivity/snr_variants_per_task.csv",
                    index_col=0)
    v = v.rename_axis("task").reset_index()
    da_cols = {c: c for c in v.columns if c.startswith("decision_acc_size_")}
    print("DA-size columns:", sorted(da_cols)[:12])
    fits_r2 = fits.groupby("task")["r2"].median()
    scores = pd.read_csv(ANALYSIS / "rq00_acc_vs_flops/pretraining/predictivity/above_random_scores.csv")
    rows = []
    proxies = [s for s in ("175M", "350M", "600M") if f"decision_acc_size_{s}" in v.columns]
    for s in proxies:
        da = v[f"decision_acc_size_{s}"]
        cands = {
            "SNR, relative std": v.get(f"snr_rel_std_{s}"),
            "SNR, discrepancy": v.get(f"snr_discrepancy_{s}"),
            "SNR, dist_std": v.get(f"snr_dist_std_{s}"),
            "signal alone (relative std)": v.get(f"signal_rel_std_{s}"),
            "noise alone (relative std, inverted)": -v.get(f"noise_rel_std_{s}") if f"noise_rel_std_{s}" in v else None,
            "early-checkpoint agreement (20 %)": v.get(f"decision_acc_ckpt_f20_{s}"),
            "scaling-fit R²": v["task"].map(fits_r2),
        }
        if s in scores.columns:
            chance = scores["task"].map(task_n_options)
            margin = (scores[s] - 1 / chance)
            margin.index = scores["task"]
            cands["margin above chance"] = v["task"].map(margin)
        # one population per panel: the tasks with an SNR at this size, i.e. the
        # gate survivors, so every candidate is scored on the same tasks
        gated = v[f"snr_rel_std_{s}"].notna()
        for kind_name, mask in (("benchmark tasks", gated & ~v["task"].str.startswith("bpb_")),
                                ("per-language bits per byte", gated & v["task"].str.startswith("bpb_"))):
            for name, x in cands.items():
                if x is None:
                    continue
                ok = mask & x.notna() & da.notna() & np.isfinite(x)
                if ok.sum() < 8:
                    continue
                r = spearmanr(x[ok], da[ok])
                rows.append({"proxy": s, "kind": kind_name, "metric": name, "rho": r.statistic,
                             "p": r.pvalue, "n": int(ok.sum())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "rq3_surrogates.csv", index=False)
    order = (t[t["kind"] == "benchmark tasks"].groupby("metric")["rho"].mean().sort_values().index.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    for ax, kind_name in zip(axes, ("benchmark tasks", "per-language bits per byte")):
        g = t[t["kind"] == kind_name]
        y = np.arange(len(order))
        for j, s in enumerate(proxies):
            gg = g[g["proxy"] == s].set_index("metric").reindex(order)
            off = (j - (len(proxies) - 1) / 2) * .22
            ax.scatter(gg["rho"], y + off, s=34, color=S.SIZE_COLOR[s], zorder=3, label=f"{s} proxy")
            for yi, (m, r) in enumerate(gg.iterrows()):
                if np.isfinite(r["rho"]):
                    ax.annotate(f"n={int(r['n'])}", (r["rho"], yi + off), xytext=(5, -2),
                                textcoords="offset points", fontsize=5.5, color=S.MUTED)
        ax.axvline(0, color=S.MUTED, lw=.8)
        ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7.5)
        ax.set_xlim(-0.6, 0.9); ax.set_xlabel("Spearman ρ with decision accuracy (proxy vs reference)")
        ax.set_title(kind_name, loc="left"); ax.grid(axis="x", color=S.GRID, lw=.6); ax.set_axisbelow(True)
        S.clean(ax); ax.tick_params(length=0)
    axes[0].legend(frameon=False, loc="lower right")
    save(fig, "rq3_surrogates")
    FACTS["rq3"] = t.round(3).to_dict("records")
    return t


# --- RQ4: does it depend on the intervention ---------------------------------

def rq4(df, dt):
    grid = finals(df[df["seed"] == GRID_SEED])
    # per-task seed noise floor: std over seed replicates of the baseline cells,
    # median over the (size, L) cells that have replicates
    base = finals(df[(df["arch"] == "deep") & (df["scheme"] == "A")])
    sd = base.groupby(["size", "L", "task"])["primary_score"].agg(["std", "count"])
    sd = sd[sd["count"] >= 2]["std"].groupby("task").median()
    rows = []
    for name, iv in INTERVENTIONS.items():
        axis, levels, (hcol, hval) = iv["axis"], iv["levels"], iv["hold"]
        sub = grid[grid[hcol] == hval]
        for L, g in sub.groupby("L"):
            piv = g.pivot_table(index=["size", "task"], columns=axis, values="primary_score")
            if not set(levels) <= set(piv.columns):
                continue
            piv = piv.dropna(subset=list(levels))
            is_bpb = piv.index.get_level_values("task").str.startswith("bpb_")
            for pop, mask in (("bits per byte", is_bpb & (piv.index.get_level_values("task") != "bpb_macro")),
                              ("benchmarks", ~is_bpb)):
                pp = piv[mask]
                # the reference for this population: the largest size scored on it at both levels
                counts = pp.groupby(level="size").size()
                sizes = size_order(counts[counts >= MIN_ITEMS].index)
                if not sizes:
                    continue
                ref = sizes[-1]
                p = pp.xs(ref, level="size")
                eff = (p[levels[0]] - p[levels[1]]).abs()
                ratio = eff / eff.index.map(sd)
                r = ratio.dropna()
                if len(r) >= MIN_ITEMS:
                    rows.append({"intervention": name, "L": int(L), "reference_size": ref, "population": pop,
                                 "median_effect_over_seed_sd": float(r.median()), "share_above_2": float((r > 2).mean()),
                                 "n": int(len(r))})
    ev = pd.DataFrame(rows)
    ev.to_csv(OUT / "rq4_effect_vs_seed.csv", index=False)
    da = dt[dt["frac"] == 1.0]
    dag = (da.groupby(["intervention", "population", "proxy_size"])
           .agg(da=("da", "mean"), cells=("da", "size"), refs=("reference_size", lambda s: ",".join(sorted(set(s)))))
           .reset_index())
    dag.to_csv(OUT / "rq4_da_by_intervention.csv", index=False)

    names = list(INTERVENTIONS)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1, 1.15]})
    y = np.arange(len(names))
    for j, (pop, col) in enumerate((("bits per byte", BPB), ("benchmarks", BENCH))):
        g = ev[ev["population"] == pop].groupby("intervention")["median_effect_over_seed_sd"].median().reindex(names)
        a0.barh(y + (j - .5) * .36, g.values, .34, color=col, label=pop)
        for yi, val in zip(y, g.values):
            if np.isfinite(val):
                a0.annotate(f"{val:.1f}×", (val, yi + (j - .5) * .36), xytext=(4, 0), va="center",
                            textcoords="offset points", fontsize=7, color=S.INK)
    a0.axvline(1, color=S.MUTED, ls="--", lw=1); a0.axvline(2, color=S.GRID, lw=.8)
    a0.set_yticks(y); a0.set_yticklabels(names, fontsize=7.5); a0.invert_yaxis()
    a0.set_xscale("log"); a0.set_xlabel("|effect| at the reference, in seed standard deviations (median)")
    a0.set_title("(a) is there a decision to make?", loc="left")
    a0.legend(frameon=False, loc="lower right"); a0.grid(axis="x", color=S.GRID, lw=.6); a0.set_axisbelow(True)
    S.clean(a0); a0.tick_params(length=0)
    styles = {"bpb_trained": ("-", "o"), "benchmark": ("--", "s")}
    cols = dict(zip(names, [S.RAMP[3], S.RAMP[1], S.SERIES[2], S.SERIES[1], "#8c1d18"]))
    for name in names:
        for pop, (ls, mk) in styles.items():
            g = dag[(dag["intervention"] == name) & (dag["population"] == pop)]
            if g.empty:
                continue
            order = size_order(g["proxy_size"])
            g = g.set_index("proxy_size").reindex(order)
            a1.plot(range(len(order)), g["da"], ls=ls, marker=mk, ms=4.5, lw=1.6, color=cols[name],
                    label=name if pop == "bpb_trained" else None)
    a1.axhline(.5, color=S.MUTED, ls=":", lw=1)
    a1.set_xticks(range(len(SIZES[1:5]))); a1.set_xticklabels(SIZES[1:5])
    a1.set_ylim(0, 1.05); a1.set_xlabel("proxy size"); a1.set_ylabel("agreement with the reference (mean over L)")
    a1.set_title("(b) does the proxy agree, per intervention", loc="left")
    h = [Line2D([], [], color=cols[n], lw=1.6, label=n) for n in names]
    h += [Line2D([], [], color=S.INK, ls="-", marker="o", ms=4, label="per-language bits per byte"),
          Line2D([], [], color=S.INK, ls="--", marker="s", ms=4, label="benchmark tasks")]
    a1.legend(handles=h, frameon=False, loc="lower right", ncol=1, fontsize=6.8)
    a1.grid(color=S.GRID, lw=.6); a1.set_axisbelow(True); S.clean(a1)
    fig.subplots_adjust(wspace=.5)
    save(fig, "rq4_interventions")
    FACTS["rq4"] = {"effect": ev.round(3).to_dict("records"), "da": dag.round(3).to_dict("records")}
    return ev, dag


# --- RQ5: transfer to unmeasured languages ------------------------------------

def rq5(df, dt):
    fin = finals(df[(df["seed"] == GRID_SEED) & (df["arch"] == "deep") & (df["scheme"] == "A")
                    & (df["kind"] == "bpb") & (df["task"] != "bpb_macro")])
    fin = fin[fin["size"] != "90M"]
    rows, points = [], []
    for L, g in fin.groupby("L"):
        piv = g.pivot_table(index="task", columns="size", values="primary_score")
        sizes = size_order(piv.columns)
        if len(sizes) < 4:
            continue
        piv = piv[sizes].dropna()
        ref, proxies = sizes[-1], sizes[:-1]
        x = np.log(np.array([NON_EMB[s] for s in proxies]))
        Y = np.log(piv[proxies].to_numpy())
        # own-language exponent on all proxy rungs
        alpha = np.array([-np.polyfit(x, yy, 1)[0] for yy in Y])
        trained = trained_tasks(int(L), "A")
        obs = piv[ref].to_numpy()
        for i, task in enumerate(piv.index):
            others = np.delete(alpha, i)
            a_pool = float(np.median(others))
            for k in range(1, len(proxies) + 1):
                xs, ys = x[:k], Y[i, :k]
                # transferred slope, intercept from the held-out language's k smallest rungs
                icpt = float(np.mean(ys + a_pool * xs))
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
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "rq5_transfer.csv", index=False)
    agg = (t.groupby(["trained", "k"])
           .agg(transfer=("err_transfer", lambda e: np.median(np.abs(e))),
                own=("err_own", lambda e: np.median(np.abs(e.dropna())) if e.notna().any() else np.nan),
                last=("err_last", lambda e: np.median(np.abs(e))), n=("task", "size")).reset_index())
    agg.to_csv(OUT / "rq5_transfer_summary.csv", index=False)

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    for trained, col, lab in ((True, BPB, "trained languages"), (False, UNTR, "never-trained languages")):
        g = agg[agg["trained"] == trained].sort_values("k")
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
    a1.set_title("(b) one 175M measurement, every language", loc="left")
    a1.legend(frameon=False, loc="upper left"); a1.grid(color=S.GRID, lw=.6); a1.set_axisbelow(True); S.clean(a1)
    fig.subplots_adjust(wspace=.4)
    save(fig, "rq5_transfer")
    # decision transfer: agreement on never-trained languages, from the decision table
    d = dt[(dt["frac"] == 1.0) & dt["population"].isin(["bpb_trained", "bpb_untrained"])]
    dd = d.groupby(["population", "proxy_size"]).agg(da=("da", "mean"), cells=("da", "size")).reset_index()
    FACTS["rq5"] = {"summary": agg.round(3).to_dict("records"),
                    "alpha_pooled_by_L": t.groupby("L")["alpha_pooled"].first().round(3).to_dict(),
                    "alpha_own_spread_by_L": {int(k): v for k, v in t[t.k == 1].groupby("L")["alpha_own"].agg(["median", "std"]).round(3).to_dict("index").items()},
                    "n_languages": int(t["task"].nunique()), "references": t.groupby("L")["reference_size"].first().to_dict(),
                    "decision_transfer": dd.round(3).to_dict("records")}
    return t


if __name__ == "__main__":
    which = set(sys.argv[1:]) or {"rq1", "rq2", "rq3", "rq4", "rq5"}
    df = load()
    FACTS["grid"] = {"cells": int(df["model"].nunique()),
                     "by_size": df.groupby("size")["model"].nunique().to_dict()}
    dt = decision_table(df) if which & {"rq2", "rq4", "rq5"} else None
    fits = rq1(df) if which & {"rq1", "rq3"} else None
    if "rq2" in which:
        rq2(dt)
    if "rq3" in which:
        rq3(df, fits)
    if "rq4" in which:
        rq4(df, dt)
    if "rq5" in which:
        rq5(df, dt)
    facts_path = OUT / "rq_facts.json"
    old = json.loads(facts_path.read_text()) if facts_path.is_file() else {}
    old.update(FACTS)
    facts_path.write_text(json.dumps(old, indent=1, default=str))
    print(f"wrote {facts_path}")
