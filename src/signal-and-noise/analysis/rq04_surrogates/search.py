"""Search the surrogates of `catalogue.py` for any that track a truth, and keep
only what survives held-out validation.

A cell is (task, proxy size, fraction) — one truth value and the surrogates of
that (task, proxy, pair set). Every correlation is a WITHIN-STRATUM Spearman:
ranks are taken within each (proxy size, fraction) stratum and the percentile
ranks correlated over the pooled cells (Pearson of within-stratum ranks; one
stratum = Spearman). A statistic that grows with training or with size
therefore cannot "predict" a DA that also grows with them.

The truths (`catalogue.py`): DA-size, DA-goal, DA-ckpt x DA, Kendall tau_b,
Spearman rho x the multi-axis and mono-axis pair sets x every pair or the
pairs sharing one language count L. Against DA-ckpt the surrogates that ARE an
early-vs-final agreement of the proxy (`CIRCULAR`) are left out.

The search space, per truth and task kind (benchmarks, per-language BPB):

  subsets   every cell; one language tier; one benchmark; one language
            (descriptive, rule 8); one proxy size; early (<= 50 %) or late
            fractions; rq02's reliability filters (`reliable_tasks.passes`,
            median reduction, 0.66 / 0.75, DA-size / DA-ckpt / both / either,
            on the same pair set) — SELECTED ON THE TRUTH, so a correlation
            there is conditional on it and labelled so
  filters   A >= q_A and B >= q_B for two of the strongest surrogates,
            thresholds at the 25/50/75 % quantiles of the discovery half:
            the correlation of each strong surrogate inside the filter, and
            the filter's own indicator against the truth

Held-out validation, so that a search over ~10^5 correlations does not report
its own noise. The (benchmark, language) clusters — a task and its `rf_`
twins together — are split in two halves within each benchmark. Everything
is ranked on the discovery half; the top N_VALIDATE configurations, and the
best TOP_PER_SUBSET surrogates of every (truth, subset), are then scored once
on the validation half with a one-sided cluster permutation test (clusters with
the same cell pattern exchange their truths; the direction is the discovery
sign) and Benjamini-Hochberg over every validated configuration. A finding is
a configuration with q < Q on the validation half.

    surrogate_correlations.csv    rho_w per (truth, subset, surrogate): full data, discovery, validation, n
    surrogate_validated.csv       the validated configurations: rho, permutation p, BH q, 90 % cluster-bootstrap band
    surrogate_filters.csv         the threshold filters: rho inside, and the indicator's rho
    surrogate_by_language.csv     per-language rho_w (descriptive, rule 8)
    da_retest.csv                 the ceiling: each truth against itself re-read at 90 %
    surrogates_*.png (+ .csv)

    python analysis/rq04_surrogates/search.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

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
from analysis.paths import DECISION_ACCURACY, SURROGATES  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import long_da, passes, per_task  # noqa: E402
from analysis.rq04_surrogates.catalogue import NOISES, SIGNALS  # noqa: E402
from analysis.utils import MIN_LANG_TASKS, PAIR_AXES  # noqa: E402

OUT_ROOT = SURROGATES
KEYS = ["task", "language", "benchmark", "kind", "tier", "proxy_size", "axes"]
TRUTH = ["t_kind", "metric", "axes", "L"]
CIRCULAR = {"da_ckpt_mean", "da_ckpt_half", "sign_persistence", "settling_time"}   # early-vs-final agreements
RELIABLE_CUTS = (0.66, 0.75)
RELIABLE_RED = "median"                 # the reduction of rq02's above_66_* filters
MIN_CELLS, MIN_TASKS = 20, 8            # a correlation needs this many cells and tasks
MIN_VALID_TASKS = 10                    # ... and this many tasks in EACH half to be validated
TOP_K = 8                               # surrogates a threshold filter is built from
QUANTILES = (0.25, 0.5, 0.75)
N_VALIDATE, N_PERM, N_BOOT = 400, 2000, 500
TOP_PER_SUBSET = 3                      # ... plus the discovery-best three of every (truth, subset)
Q = 0.05
mpl.rcParams.update(S.RC)


# --- statistics ---------------------------------------------------------------


def strat_ranks(v: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Percentile ranks of `v` within each stratum, ties averaged: rank globally,
    then rank (stratum, global rank), then subtract each stratum's offset."""
    r = rankdata(strata * (len(v) + 1) + rankdata(v))
    _, inv, cnt = np.unique(strata, return_inverse=True, return_counts=True)
    offset = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    return (r - offset[inv] - 0.5) / cnt[inv]


def rho_w(x: np.ndarray, y: np.ndarray, strata: np.ndarray, tasks: np.ndarray) -> tuple[float, int, int]:
    """(within-stratum Spearman, cells, tasks) over the cells where both are finite."""
    ok = np.isfinite(x) & np.isfinite(y)
    n, nt = int(ok.sum()), len(np.unique(tasks[ok]))
    if n < MIN_CELLS or nt < MIN_TASKS:
        return np.nan, n, nt
    px, py = strat_ranks(x[ok], strata[ok]), strat_ranks(y[ok], strata[ok])
    if px.std() == 0 or py.std() == 0:
        return np.nan, n, nt
    return float(np.corrcoef(px, py)[0, 1]), n, nt


def _blocks(cluster: np.ndarray, key: np.ndarray) -> list[np.ndarray]:
    """Groups of clusters with the same set of cell keys, each as a
    (clusters x keys) matrix of cell indices: the units a permutation or a
    bootstrap may exchange without moving a truth to another stratum."""
    d = pd.DataFrame({"c": cluster, "k": key, "i": np.arange(len(key))})
    piv = d.pivot_table(index="c", columns="k", values="i", aggfunc="first")
    pattern = piv.notna().apply(lambda r: tuple(r), axis=1)
    return [piv.loc[g].dropna(axis=1).to_numpy(int) for _, g in piv.groupby(pattern).groups.items()]


def perm_test(x, y, strata, cluster, key, sign: float, rng) -> tuple[float, float, float, float]:
    """(rho_w, one-sided permutation p in the direction `sign`, 5 %, 95 %
    cluster-bootstrap band) on the finite cells."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y, strata, cluster, key = x[ok], y[ok], strata[ok], cluster[ok], key[ok]
    px, py = strat_ranks(x, strata), strat_ranks(y, strata)
    obs = np.corrcoef(px, py)[0, 1]
    blocks = _blocks(cluster, key)
    src = np.concatenate([b.ravel() for b in blocks])
    hits = 0
    for _ in range(N_PERM):
        perm = np.concatenate([b[rng.permutation(len(b))].ravel() for b in blocks])
        yy = py.copy()
        yy[src] = py[perm]
        hits += sign * np.corrcoef(px, yy)[0, 1] >= sign * obs - 1e-12
    p = (hits + 1) / (N_PERM + 1)
    clusters = np.unique(cluster)
    idx = {c: np.flatnonzero(cluster == c) for c in clusters}
    boot = []
    for _ in range(N_BOOT):
        sel = np.concatenate([idx[c] for c in rng.choice(clusters, len(clusters))])
        bx, by = strat_ranks(x[sel], strata[sel]), strat_ranks(y[sel], strata[sel])
        if bx.std() and by.std():
            boot.append(np.corrcoef(bx, by)[0, 1])
    lo, hi = np.percentile(boot, [5, 95]) if boot else (np.nan, np.nan)
    return float(obs), float(p), float(lo), float(hi)


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values."""
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


# --- data -----------------------------------------------------------------------


def load(out_dir: Path, pool: str) -> tuple[pd.DataFrame, list[str]]:
    """Cells: every truth row joined to its surrogates, with the stratum, the
    (benchmark, language) cluster, the discovery / validation half and the
    reliability flags."""
    v = pd.read_csv(out_dir / "surrogate_values.csv", low_memory=False)
    t = pd.read_csv(out_dir / "surrogate_targets.csv", dtype={"L": str}).rename(columns={"kind": "t_kind", "n_pairs": "t_n_pairs"})
    names = [c for c in v.columns if c not in KEYS]
    v[names] = v[names].replace([np.inf, -np.inf], np.nan)
    d = t.drop(columns="language").merge(v, on=["task", "proxy_size", "axes"], how="inner")
    d["stratum"] = pd.factorize(d["proxy_size"] + "@" + d["frac"].astype(str))[0]
    base = d["benchmark"].str.replace(G._TWIN, r"\2", regex=True)
    d["cluster"] = base + "|" + d["language"]
    tix = d.groupby(["cluster", "benchmark"])["task"].transform(lambda t: pd.factorize(t)[0])
    d["key"] = pd.factorize(d["benchmark"] + "@" + tix.astype(str) + "@" + d["stratum"].astype(str))[0]
    rng = np.random.default_rng(0)
    half = {}
    for b, cl in d.groupby(base)["cluster"]:
        cl = rng.permutation(sorted(cl.unique()))
        half |= {c: i % 2 for i, c in enumerate(cl)}
    d["half"] = d["cluster"].map(half)                                  # 0 discovery, 1 validation
    stage = load_pools()[pool].get("stage", "pretraining")
    for a in PAIR_AXES[:2]:
        rel = per_task(long_da(DECISION_ACCURACY / stage / pool, pool, axes=a)).set_index("task")
        for cut in RELIABLE_CUTS:
            p = passes(rel, RELIABLE_RED, cut)
            for crit in p.columns:
                col = f"reliable_{int(round(cut * 100))}_{crit}"
                m = d["axes"] == a
                d.loc[m, col] = d.loc[m, "task"].map(p[crit]).fillna(False).astype(bool)
    return d, names


def truths(d: pd.DataFrame):
    for key, g in d.groupby(TRUTH, sort=False):
        yield dict(zip(TRUTH, key)), g


def subsets(g: pd.DataFrame):
    """(type, name, mask) over one truth's benchmark cells, plus BPB as its own population."""
    b = g["kind"] == "benchmark"
    yield "all", "benchmarks", b.to_numpy()
    yield "all", "bpb", (g["kind"] == "bpb").to_numpy()
    for col, typ in (("tier", "tier"), ("benchmark", "benchmark"), ("proxy_size", "proxy")):
        for k in g.loc[b, col].dropna().unique():
            yield typ, str(k), (b & (g[col] == k)).to_numpy()
    if (g["frac"] < 1).any():
        yield "stage", "early (<= 50 %)", (b & (g["frac"] <= .5)).to_numpy()
        yield "stage", "late (60-90 %)", (b & (g["frac"] > .5)).to_numpy()
    for col in [c for c in g.columns if c.startswith("reliable_")]:
        yield "reliable (on the truth)", col.removeprefix("reliable_"), (b & g[col].fillna(False).astype(bool)).to_numpy()


def surrogate_names(names: list[str], t_kind: str) -> list[str]:
    return [m for m in names if not (t_kind == "ckpt" and m in CIRCULAR)]


# --- the search -------------------------------------------------------------------


def _three(x, y, st, tk, half, m) -> dict:
    """rho_w on the full data, the discovery and the validation half of mask `m`."""
    out = {}
    for tag, mm in (("", m), ("_disc", m & (half == 0)), ("_val", m & (half == 1))):
        r, n, nt = rho_w(x[mm], y[mm], st[mm], tk[mm])
        out |= {f"rho{tag}": r, f"n{tag}": n, f"n_tasks{tag}": nt}
    return out


def screen_truth(job) -> list[dict]:
    """rho_w of every surrogate per subset of one truth, on the full data and on each half."""
    tr, g, names = job
    y, st, tk, half = g["value"].to_numpy(float), g["stratum"].to_numpy(), g["task"].to_numpy(), g["half"].to_numpy()
    rows = []
    for typ, name, m in subsets(g):
        if m.sum() < MIN_CELLS:
            continue
        for s in surrogate_names(names, tr["t_kind"]):
            r = _three(g[s].to_numpy(float), y, st, tk, half, m)
            if np.isfinite(r["rho"]):
                rows.append({**tr, "subset_type": typ, "subset": name, "surrogate": s, **r})
    return rows


def filters_truth(job) -> list[dict]:
    """A >= q_A and B >= q_B over the benchmarks, A and B among the TOP_K
    surrogates of the truth on the discovery half, each oriented by its
    discovery sign, thresholds at discovery quantiles. Per filter: the rho_w
    of each of the TOP_K inside it, and the rho_w of the filter's own
    indicator over every benchmark cell. The thresholds travel with the row."""
    tr, g, top = job
    g = g[g["kind"] == "benchmark"]
    y, st, tk, half = g["value"].to_numpy(float), g["stratum"].to_numpy(), g["task"].to_numpy(), g["half"].to_numpy()
    sign = dict(zip(top["surrogate"], np.sign(top["rho_disc"])))
    rows = []
    for a, b in combinations(top["surrogate"], 2):
        xa, xb = g[a].to_numpy(float) * sign[a], g[b].to_numpy(float) * sign[b]
        for qa in QUANTILES:
            for qb in QUANTILES:
                f = {"f_a": a, "f_sa": sign[a], "f_ta": np.nanquantile(xa[half == 0], qa),
                     "f_b": b, "f_sb": sign[b], "f_tb": np.nanquantile(xb[half == 0], qb)}
                name = f"{a} ≥ q{int(qa * 100)} & {b} ≥ q{int(qb * 100)}"
                keep = filter_keep(g, f)
                ind = np.where(np.isfinite(xa) & np.isfinite(xb), keep.astype(float), np.nan)
                for s, x, m in [("[filter indicator]", ind, np.ones(len(g), bool))] + \
                               [(s, g[s].to_numpy(float), keep) for s in top["surrogate"]]:
                    r = _three(x, y, st, tk, half, m)
                    if np.isfinite(r["rho"]):
                        rows.append({**tr, "subset_type": "filter", "subset": name, "surrogate": s, **f, **r})
    return rows


def filter_keep(g: pd.DataFrame, f) -> np.ndarray:
    return ((g[f["f_a"]].to_numpy(float) * f["f_sa"] >= f["f_ta"])
            & (g[f["f_b"]].to_numpy(float) * f["f_sb"] >= f["f_tb"]))


def cells(g: pd.DataFrame, r) -> tuple[np.ndarray, np.ndarray]:
    """(mask, surrogate values) of the configuration in row `r`, over the truth's cells `g`."""
    b = (g["kind"] == "benchmark").to_numpy()
    typ, name, s = r["subset_type"], r["subset"], r["surrogate"]
    if typ == "filter":
        keep = filter_keep(g, r)
        if s == "[filter indicator]":
            fin = np.isfinite(g[r["f_a"]].to_numpy(float)) & np.isfinite(g[r["f_b"]].to_numpy(float))
            return b, np.where(fin, keep.astype(float), np.nan)
        return b & keep, g[s].to_numpy(float)
    m = {"all": lambda: b if name == "benchmarks" else (g["kind"] == "bpb").to_numpy(),
         "tier": lambda: b & (g["tier"].astype(str) == name).to_numpy(),
         "benchmark": lambda: b & (g["benchmark"] == name).to_numpy(),
         "proxy": lambda: b & (g["proxy_size"] == name).to_numpy(),
         "stage": lambda: b & ((g["frac"] <= .5) if name.startswith("early") else (g["frac"] > .5)).to_numpy(),
         }.get(typ, lambda: b & g[f"reliable_{name}"].fillna(False).astype(bool).to_numpy())()
    return m, g[s].to_numpy(float)


def validate_job(job) -> list[dict]:
    """The permutation test and bootstrap band on the validation half, per configuration."""
    g, rows, seed = job
    g = g[g["half"] == 1]
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        m, x = cells(g, r)
        st = perm_test(x[m], g["value"].to_numpy(float)[m], g["stratum"].to_numpy()[m],
                       g["cluster"].to_numpy()[m], g["key"].to_numpy()[m], np.sign(r["rho_disc"]), rng)
        out.append({**r, "rho_val_check": st[0], "p_val": st[1], "val_lo": st[2], "val_hi": st[3]})
    return out




def ceiling(d: pd.DataFrame) -> pd.DataFrame:
    """rho_w of each truth with itself re-read at 90 % (`retest`), per kind of task."""
    rows = []
    for tr, g in truths(d[d["retest"].notna()]):
        for kind, h in g.groupby("kind"):
            r, n, nt = rho_w(h["value"].to_numpy(float), h["retest"].to_numpy(float),
                             h["stratum"].to_numpy(), h["task"].to_numpy())
            rows.append({**tr, "kind": kind, "rho": r, "n": n, "n_tasks": nt})
    return pd.DataFrame(rows)


def by_language(d: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """rho_w per language over its benchmark cells (every pair, L all),
    descriptive: >= MIN_LANG_TASKS tasks (rule 8), no validation."""
    rows = []
    for tr, g in truths(d[(d["kind"] == "benchmark") & (d["L"] == "all")]):
        for lang, h in g.groupby("language"):
            if h["task"].nunique() < MIN_LANG_TASKS:
                continue
            y, st, tk = h["value"].to_numpy(float), h["stratum"].to_numpy(), h["task"].to_numpy()
            for s in surrogate_names(names, tr["t_kind"]):
                x = h[s].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if len(np.unique(tk[ok])) < MIN_LANG_TASKS:
                    continue
                px, py = strat_ranks(x[ok], st[ok]), strat_ranks(y[ok], st[ok])
                if px.std() and py.std():
                    rows.append({**tr, "language": lang, "surrogate": s, "rho": np.corrcoef(px, py)[0, 1],
                                 "n": int(ok.sum()), "n_tasks": len(np.unique(tk[ok]))})
    return pd.DataFrame(rows)


def label(r) -> str:
    L = "" if str(r["L"]) == "all" else f", pairs within L{r['L']}"
    return f"DA-{r['t_kind']} {r['metric']} ({r['axes']}{L})"


# --- figures ----------------------------------------------------------------------


def fig_snr_grid(corr: pd.DataFrame, out_dir: Path) -> None:
    """The AllenAI signal x noise grid against each DA kind (benchmarks, every pair)."""
    c = corr[(corr["subset"] == "benchmarks") & (corr["axes"] == PAIR_AXES[0]) & (corr["L"].astype(str) == "all")
             & (corr["metric"] == "da") & corr["surrogate"].str.startswith("snr__")]
    kinds = [k for k in ("size", "goal", "ckpt") if k in set(c["t_kind"])]
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds) + 1, 7.5), sharey=True)
    tabs = []
    for ax, k in zip(np.atleast_1d(axes), kinds):
        g = c[c["t_kind"] == k].assign(signal=lambda x: x["surrogate"].str.split("__").str[1],
                                       noise=lambda x: x["surrogate"].str.split("__").str[2])
        mat = g.pivot_table(index="signal", columns="noise", values="rho").reindex(index=SIGNALS, columns=list(NOISES))
        tabs.append(mat.assign(truth=f"DA-{k}"))
        G._draw(ax, mat, None, vmin=-.5, vmax=.5, cmap=S.DIV, fmt="{:.2f}", fontsize=6.5, center=0)
        ax.set_xticklabels(mat.columns, rotation=35, ha="right")
        ax.set_title(f"DA-{k}", loc="left")
    top = G._header(fig, "Every AllenAI signal against every noise",
                    "cell = within-(proxy, fraction) Spearman ρ of signal / noise with DA over the benchmark cells above "
                    "chance (every pair); kfold = benchmark noise in closed form; white = undefined (discrepancy off benchmarks)")
    fig.tight_layout(rect=(0, 0, 1, top))
    pd.concat(tabs).to_csv(out_dir / "surrogates_snr_grid.csv")
    S.save_figure(fig, out_dir, "surrogates_snr_grid")


def fig_truths(corr: pd.DataFrame, out_dir: Path) -> None:
    """The strongest surrogates against every truth (benchmarks, full data)."""
    c = corr[corr["subset"] == "benchmarks"].copy()
    c["truth"] = c.apply(label, axis=1)
    mat = c.pivot_table(index="surrogate", columns="truth", values="rho")
    rows = mat.abs().mean(axis=1).sort_values(ascending=False).head(35).index
    mat = mat.loc[rows]
    fig, ax = plt.subplots(figsize=(.36 * mat.shape[1] + 4, .26 * len(mat) + 2.5))
    G._draw(ax, mat, None, vmin=-.5, vmax=.5, cmap=S.DIV, fmt="{:.2f}", fontsize=5.5, center=0)
    ax.set_xticklabels(mat.columns, rotation=60, ha="right", fontsize=6.5)
    top = G._header(fig, "The 35 strongest surrogates against every truth",
                    "cell = within-(proxy, fraction) Spearman ρ over every benchmark cell above chance, full data "
                    "(not validated: see surrogates_validated); rows = largest mean |ρ| over the truths")
    fig.tight_layout(rect=(0, 0, 1, top))
    mat.to_csv(out_dir / "surrogates_catalogue.csv")
    S.save_figure(fig, out_dir, "surrogates_catalogue")


def fig_validated(val: pd.DataFrame, out_dir: Path) -> None:
    """The findings: discovery rho, validation rho with its 90 % band, q."""
    f = val[val["q_val"] < Q].copy()
    f = f.reindex(f["rho_val"].abs().sort_values(ascending=False).index).head(30)
    f["config"] = f.apply(lambda r: f"{label(r)} | {r['subset_type']}: {r['subset']} | {r['surrogate']}", axis=1)
    f.to_csv(out_dir / "surrogates_validated.csv", index=False)
    if f.empty:
        return
    fig, ax = plt.subplots(figsize=(11, .3 * len(f) + 2))
    y = np.arange(len(f))[::-1]
    ax.scatter(f["rho_disc"], y + .15, color=S.MUTED, s=16, label="discovery half")
    ax.errorbar(f["rho_val"], y - .15, xerr=[f["rho_val"] - f["val_lo"], f["val_hi"] - f["rho_val"]], fmt="o",
                ms=4, color=S.SERIES[0], lw=.9, label="validation half (90 % cluster bootstrap)")
    for yi, (_, r) in zip(y, f.iterrows()):
        ax.annotate(f"q={r['q_val']:.1e}, {int(r['n_tasks_val'])} tasks", (max(r["val_hi"], r["rho_disc"]), yi),
                    xytext=(4, -2), textcoords="offset points", fontsize=5.5, color=S.MUTED)
    ax.axvline(0, color=S.MUTED, lw=.8)
    ax.set_yticks(y); ax.set_yticklabels(f["config"], fontsize=5.8)
    ax.set_xlabel("within-(proxy, fraction) Spearman ρ with the truth"); ax.legend(frameon=False, fontsize=6.5)
    S.clean(ax); ax.grid(axis="x", color=S.GRID, lw=.6)
    top = G._header(fig, "Validated surrogates: found on one half of the benchmarks, confirmed on the other",
                    f"the {len(f)} strongest configurations with BH q < {Q} on the held-out half (one-sided cluster "
                    "permutation test); a (benchmark, language) cluster and its rf twins sit in one half")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save_figure(fig, out_dir, "surrogates_validated")


def fig_by_L(d: pd.DataFrame, corr: pd.DataFrame, out_dir: Path) -> None:
    """Decision accuracy over the pairs that share L, and how well the best surrogate tracks it."""
    t = d[(d["L"].astype(str) != "all") & (d["kind"] == "benchmark")].drop_duplicates(TRUTH + ["task", "proxy_size", "frac"])
    t = t.assign(L=t["L"].astype(int))
    summ = t.groupby(["t_kind", "axes", "L"]).agg(mean=("value", "mean"), sd=("value", "std"),
                                                   tasks=("task", "nunique"), cells=("value", "size")).reset_index()
    c = corr[(corr["L"].astype(str) != "all") & (corr["subset"] == "benchmarks")]
    best = c.loc[c.groupby(TRUTH)["rho"].apply(lambda r: r.abs().idxmax())]
    best = best.assign(L=best["L"].astype(int))
    summ = summ.merge(best[["t_kind", "axes", "L", "surrogate", "rho"]], on=["t_kind", "axes", "L"], how="left")
    summ.to_csv(out_dir / "surrogates_by_L.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for (kind, axes_name), g in summ.groupby(["t_kind", "axes"]):
        ls = "-" if axes_name == PAIR_AXES[0] else "--"
        col = {"size": S.SERIES[0], "goal": S.SERIES[1], "ckpt": S.SERIES[2]}[kind]
        axes[0].errorbar(g["L"], g["mean"], yerr=g["sd"], color=col, ls=ls, marker="o", ms=3, capsize=2,
                         label=f"DA-{kind}, {axes_name}")
        axes[1].plot(g["L"], g["rho"].abs(), color=col, ls=ls, marker="o", ms=3)
    axes[0].set_ylabel("DA over the pairs sharing L (mean ± sd over cells)")
    axes[1].set_ylabel("|ρ| of the best surrogate (full data)")
    for ax in axes:
        ax.set_xlabel("language count L"); ax.set_xticks(sorted(summ["L"].unique())); S.clean(ax)
        ax.grid(color=S.GRID, lw=.6)
    axes[0].legend(frameon=False, fontsize=6)
    top = G._header(fig, "Does rank agreement get noisier with more languages in the mix?",
                    "pairs of variants that share L (L with ≥ 3 such pairs in this pool); benchmark cells above chance; "
                    "the right panel is not validated")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save_figure(fig, out_dir, "surrogates_by_L")


def fig_languages(bl: pd.DataFrame, out_dir: Path) -> None:
    g = bl[(bl["t_kind"] == "size") & (bl["metric"] == "da") & (bl["axes"] == PAIR_AXES[0])]
    mat = g.pivot_table(index="surrogate", columns="language", values="rho")
    mat = mat.loc[mat.abs().mean(axis=1).sort_values(ascending=False).head(30).index]
    cnt = g.pivot_table(index="surrogate", columns="language", values="n_tasks").reindex(mat.index)
    mat = mat[mat.mean().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(.32 * mat.shape[1] + 4, .26 * len(mat) + 2.5))
    G._draw(ax, mat, None, vmin=-1, vmax=1, cmap=S.DIV, fmt="{:.1f}", fontsize=5, center=0)
    top = G._header(fig, "Surrogates of DA-size per language (descriptive)",
                    f"cell = within-proxy Spearman ρ over the language's benchmark cells above chance, every pair; "
                    f"white = fewer than {MIN_LANG_TASKS} tasks (rule 8); none is validated, task counts in the CSV")
    fig.tight_layout(rect=(0, 0, 1, top))
    mat.join(cnt.add_prefix("n_tasks_")).to_csv(out_dir / "surrogates_catalogue_by_language.csv")
    S.save_figure(fig, out_dir, "surrogates_catalogue_by_language")


# --- README -------------------------------------------------------------------------


def generate_readme(pool: str, corr, val, ceil, byL) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    n_tests = len(corr)
    found = val[val["q_val"] < Q]
    blocks = [f"**Search size.** {n_tests:,} correlations screened (truths × subsets × surrogates, plus the threshold "
              f"filters); {len(val)} of the strongest on the discovery half were tested on the validation half; "
              f"**{len(found)} hold at BH q < {Q}** ({found['subset_type'].value_counts().to_dict()})."]
    ce = ceil[(ceil["kind"] == "benchmark")]
    blocks += ["**The ceiling** — each truth against itself re-read at 90 % (within-stratum ρ, benchmarks):",
               md_table(["truth", "ρ", "tasks"], [[label(r), fmt(r["rho"]), int(r["n_tasks"])] for _, r in ce.iterrows()])]
    b = corr[corr["subset"] == "benchmarks"]
    best = b.loc[b.groupby(TRUTH)["rho"].apply(lambda r: r.abs().idxmax())]
    blocks += ["**Strongest surrogate per truth** (every benchmark cell, full data; the discovery and validation "
               "halves beside it):",
               md_table(["truth", "surrogate", "ρ", "ρ discovery", "ρ validation", "tasks"],
                        [[label(r), f"`{r['surrogate']}`", fmt(r["rho"]), fmt(r["rho_disc"]), fmt(r["rho_val"]),
                          int(r["n_tasks"])] for _, r in best.iterrows()])]
    snr = b[b["surrogate"].str.startswith("snr__") & (b["metric"] == "da") & (b["axes"] == PAIR_AXES[0])
            & (b["L"].astype(str) == "all")]
    sb = snr.loc[snr.groupby("t_kind")["rho"].idxmax()]
    std = snr[snr["surrogate"] == "snr__rel_std__ckpt_rel"].set_index("t_kind")["rho"]
    blocks += ["**The AllenAI grid** (22 signals × 6 noises, DA, every pair, benchmarks): best combination against "
               "AllenAI's own `rel_std` / checkpoint noise:",
               md_table(["truth", "best signal / noise", "ρ", "rel_std / ckpt_rel ρ"],
                        [[f"DA-{r['t_kind']}", f"`{r['surrogate'].removeprefix('snr__').replace('__', ' / ')}`",
                          fmt(r["rho"]), fmt(std.get(r["t_kind"], np.nan))] for _, r in sb.iterrows()])]
    top = found.reindex(found["rho_val"].abs().sort_values(ascending=False).index).head(25)
    blocks += [f"**Validated findings** (the 25 strongest by validation ρ; all {len(found)} in "
               "`surrogate_validated.csv`). `reliable (on the truth)` subsets are selected on the truth itself, so "
               "their ρ is conditional on it:",
               md_table(["truth", "subset", "surrogate", "ρ disc.", "ρ val. [90 %]", "q", "tasks val."],
                        [[label(r), f"{r['subset_type']}: {r['subset']}", f"`{r['surrogate']}`", fmt(r["rho_disc"]),
                          f"{fmt(r['rho_val'])} [{fmt(r['val_lo'])}, {fmt(r['val_hi'])}]", f"{r['q_val']:.1e}",
                          int(r["n_tasks_val"])] for _, r in top.iterrows()])]
    bl = byL[byL["axes"] == PAIR_AXES[0]]
    blocks += ["**Per language count** (pairs of variants sharing L; benchmarks):",
               md_table(["truth", "L", "mean DA", "sd", "tasks", "best surrogate", "ρ"],
                        [[f"DA-{r['t_kind']}", int(r["L"]), fmt(r["mean"]), fmt(r["sd"]), int(r["tasks"]),
                          f"`{r['surrogate']}`", fmt(r["rho"])] for _, r in bl.iterrows()])]
    blocks += [f"![{n}]({stage}/{pool}/{n}.png)" for n in ("surrogates_validated", "surrogates_snr_grid",
                                                             "surrogates_catalogue", "surrogates_by_L",
                                                             "surrogates_catalogue_by_language")]
    body = "\n\n".join([
        "## A catalogue of surrogates beyond SNR",
        f"Numbers from the `{pool}` pool. Regenerate with `python analysis/rq04_surrogates/catalogue.py --pool {pool}` "
        f"then `search.py --pool {pool}`. Surrogates are read on the proxy alone (rule 11); the truths are rq02's DA "
        f"(DA-size, DA-goal, DA-ckpt; both pair sets; every pair or the pairs sharing L) and Kendall τ-b / Spearman ρ "
        f"on the same rankings. Every ρ is a within-(proxy size, fraction) Spearman over cells above chance (rule 1), "
        f"so a statistic that only grows with training or size cannot score. Populations differ per cell (rule 13); n "
        f"is in the CSVs. Definitions and sources: [`literature.md`](literature.md); method: `search.py`'s docstring.",
        *blocks])
    replace_block(OUT_ROOT / "README.md", "catalogue", body, f"search.py --pool {pool}")


def main(pool: str, out_dir: Path) -> None:
    from multiprocessing import Pool
    d, names = load(out_dir, pool)
    print(f"{len(d):,} cells, {d['task'].nunique()} tasks, {len(names)} surrogates, "
          f"{d.groupby(TRUTH).ngroups} truths")
    jobs = [(tr, g, names) for tr, g in truths(d)]
    with Pool(4) as p:
        corr = pd.DataFrame([r for rows in p.map(screen_truth, jobs) for r in rows])
    print(f"  screened: {len(corr):,} correlations")
    tops = []
    for tr, g in truths(d):
        c = corr[(corr[TRUTH].astype(str) == pd.Series(tr).astype(str)).all(axis=1) & (corr["subset"] == "benchmarks")]
        c = c.dropna(subset=["rho_disc"])
        top = c.reindex(c["rho_disc"].abs().sort_values(ascending=False).index).head(TOP_K)
        if len(top) >= 2:
            tops.append((tr, g, top))
    with Pool(4) as p:
        filt = pd.DataFrame([r for rows in p.map(filters_truth, tops) for r in rows])
    filt.to_csv(out_dir / "surrogate_filters.csv", index=False)
    print(f"  filters: {len(filt):,} correlations")
    allres = pd.concat([corr, filt], ignore_index=True)
    allres.drop(columns=[c for c in allres.columns if c.startswith("f_")]).to_csv(
        out_dir / "surrogate_correlations.csv", index=False)
    ok = allres[(allres["n_tasks_disc"] >= MIN_VALID_TASKS) & (allres["n_tasks_val"] >= MIN_VALID_TASKS)].dropna(
        subset=["rho_disc", "rho_val"]).copy()                         # both halves above the size minimums
    ok["abs_disc"] = ok["rho_disc"].abs()
    best = ok[ok["subset_type"] != "filter"].sort_values("abs_disc").groupby(TRUTH + ["subset_type", "subset"]).tail(TOP_PER_SUBSET)
    pick = pd.concat([ok.nlargest(N_VALIDATE, "abs_disc"), best])
    pick = pick.loc[~pick.index.duplicated()].drop(columns="abs_disc")
    vjobs = []
    for i, (tr, g) in enumerate(truths(d)):
        rows = pick[(pick[TRUTH].astype(str) == pd.Series(tr).astype(str)).all(axis=1)]
        if len(rows):
            vjobs.append((g, rows.to_dict("records"), i))
    with Pool(4) as p:
        val = pd.DataFrame([r for rows in p.map(validate_job, vjobs) for r in rows])
    assert np.allclose(val["rho_val"], val["rho_val_check"], equal_nan=True), "validation cells differ from the screen"
    val["q_val"] = bh(val["p_val"].to_numpy())
    val = val.drop(columns="rho_val_check").sort_values("q_val")
    val.to_csv(out_dir / "surrogate_validated.csv", index=False)
    print(f"  validated: {len(val)} tested, {(val['q_val'] < Q).sum()} at q < {Q}")
    ceil = ceiling(d)
    ceil.to_csv(out_dir / "da_retest.csv", index=False)
    bl = by_language(d, names)
    bl.to_csv(out_dir / "surrogate_by_language.csv", index=False)
    fig_snr_grid(corr, out_dir)
    fig_truths(corr, out_dir)
    fig_validated(val, out_dir)
    fig_by_L(d, corr, out_dir)
    fig_languages(bl, out_dir)
    byL = pd.read_csv(out_dir / "surrogates_by_L.csv")
    generate_readme(pool, corr, val, ceil, byL)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}")
    main(args.pool, OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool)
