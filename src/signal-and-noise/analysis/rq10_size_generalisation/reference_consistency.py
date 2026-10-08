"""Size generalisation: does the DA/SNR framework read the same when the
reference moves one rung up, from 1.7B to 3B?

`above_reference.py` asks whether a proxy predicts the 3B ranking. This asks
whether the FRAMEWORK is consistent across the two references, on the same
families, pairs and tasks, so the reference is the only thing that differs:

  DA   (a) DA-size of every proxy (90M–1B) to the 3B final and to the 1.7B
           final, over the decisions both references can score (the same
           pairs of the same families on the same task), pooled per proxy with
           the leave-one-family-out jackknife band (`utils.jackknife_ratio`);
           benchmark accuracy and per-language BPB apart, as everywhere in rq10;
       (b) per task, DA-size to 1.7B against DA-size to 3B;
       (c) per proxy, Spearman over tasks (and over benchmarks, the mean DA of
           their tasks) between the two references' DA: does the ranking of
           benchmarks by DA survive the change of reference?
  SNR  rq03's headline SNR (`rel_std`: the relative std of the finals across
       the families over the relative checkpoint noise of rule 4's window) at
       1.7B and at 3B on the same (family, task) cells, and the Spearman over
       tasks and over benchmarks between the two.

Rules: the gate is `above_reference.gate_mask` (`predictivity`'s mask, plus
the Wilson rule on the 3B runs); a DA cell needs the task above chance at the
proxy, at 1.7B and at 3B (rule 1, both readings on one task set), and
MIN_PAIRS pairs (rule 5: below it the cell is NaN and its pair count stays in
the per-task table); an SNR cell needs the task above chance at its size.
Rule 10: the 3B rung is read here only, through `above_reference=True`.

    reference_consistency_da_size_<pair set>.png / .csv   (a)–(c) per pair set
    reference_consistency_da_size_per_task_both_axes.csv  every (task, pair set, proxy) cell
    reference_consistency_snr.png / .csv                  SNR at 1.7B against 3B, per task and per benchmark
    gate_share_and_da_size_mono_axis_paper.png / .svg / .csv
        the paper figure: the share of each benchmark above chance at 1.7B and 3B
        (from gate_crossover_by_benchmark.csv) beside panel (a), mono-axis
    gate_share_and_da_size_mono_axis_fixed_tasks_paper.png / .svg / .csv
        its DA panel over one task set per line (rule 13): the tasks kept at
        every proxy size, solid, over the paper's reading, dashed; a panel per
        reference, no band (from the per-task table)

    python analysis/rq10_size_generalisation/reference_consistency.py --pool predictivity
    python analysis/rq10_size_generalisation/reference_consistency.py --paper   # the paper figure from the CSVs on disk
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
from snr.snr_variants import rel_std_snr  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import SIZE_GENERALISATION  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import decisions  # noqa: E402
from analysis.rq03_noise_and_snr.run_apertus_snr_variants import per_model_inputs, variant_signal_noise_snr  # noqa: E402
from analysis.rq10_size_generalisation.above_reference import GITHUB, REFERENCE, gate_mask  # noqa: E402
from analysis.rq10_size_generalisation.gate_crossover import dumbbell  # noqa: E402
from analysis.utils import (  # noqa: E402
    AXES_SUFFIX, MIN_PAIRS, NON_EMB, PAIR_AXES, TARGET_SIZE, benchmark_family, design_axes, finals, fixed_population,
    jackknife_ratio, ladder_frame, on_noise_grid, on_shared_grid, pair_sets, passes_gate, size_order)

OUT_ROOT = SIZE_GENERALISATION
KEY = "reference-consistency"
AXES = PAIR_AXES[:2]                      # the design pair sets; the pool has no seed replicates at 3B
PAIR_KEYS = ["task", "group", "size", "frac", "family_a", "family_b"]
REFS = (TARGET_SIZE, REFERENCE)
STYLE = {REFERENCE: dict(ls="-", lw=1.6), TARGET_SIZE: dict(ls=(0, (3, 2)), lw=1.2)}
CHANNEL_COLOR = {"benchmarks": S.INK, "bpb": S.SERIES[1]}
PAPER_STEM = f"gate_share_and_da_size{AXES_SUFFIX['mono-axis']}_paper"   # no size in the stem: "1.7B" reads as a suffix to Path
FIXED_STEM = PAPER_STEM.removesuffix("_paper") + "_fixed_tasks_paper"
mpl.rcParams.update(S.RC)


def channel(task: str) -> str:
    """The measurement channel rq10 pools apart: benchmark accuracy, per-language BPB, the loss."""
    return {"bpb": "bpb", "loss": "loss"}.get(benchmark_family(task), "benchmarks")


def load(pool: str) -> tuple[pd.DataFrame, list]:
    """The pool with the rung above the reference, on the families scored at
    both 1.7B and 3B (a family with only its training loss at 3B is not one)."""
    df = ladder_frame(pool, above_reference=True)
    df = df[on_shared_grid(df) | on_noise_grid(df)]       # the tenths, and the five k/20 points of the noise window (rule 4)
    fin = finals(df)
    scored = fin[fin["kind"] != "loss"]
    fams = sorted(set(scored.loc[scored["size"] == REFERENCE, "family"]) & set(fin.loc[fin["size"] == TARGET_SIZE, "family"]))
    return df[df["family"].isin(fams)], fams


# --- decision accuracy -----------------------------------------------------------------------------------

def da_cells(df: pd.DataFrame, mask: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The decisions both references score (the same pair on the same task at
    the same proxy) and, per (task, pair set, proxy), DA-size to each reference
    with its pair count, NaN under rule 5 or rule 1."""
    fin = finals(df).assign(frac=1.0)
    sets = pair_sets(design_axes(fin))
    proxies = [s for s in size_order(fin["size"].unique()) if NON_EMB[s] < NON_EMB[TARGET_SIZE]]
    dec = {r: decisions(fin, {a: sets[a] for a in AXES}, proxies, fin, ref=r).astype({c: str for c in PAIR_KEYS if c != "frac"})
           for r in REFS}
    d = dec[REFERENCE].merge(dec[TARGET_SIZE][PAIR_KEYS + ["match"]], on=PAIR_KEYS, suffixes=(f"_{REFERENCE}", f"_{TARGET_SIZE}"))
    d = d.rename(columns={"group": "axes"})
    ok = {s: passes_gate(mask, d["task"].unique(), s, *REFS) for s in proxies}
    d["gated"] = [not ok[s][t] for t, s in zip(d["task"], d["size"])]
    d["channel"] = d["task"].map(channel)
    agg = {"n_pairs": (f"match_{REFERENCE}", "size"), "gated": ("gated", "first"), "channel": ("channel", "first")}
    agg |= {f"n_matching_{r}": (f"match_{r}", "sum") for r in REFS}
    cells = d.groupby(["task", "axes", "size"]).agg(**agg).reset_index()
    keep = (cells["n_pairs"] >= MIN_PAIRS) & ~cells["gated"]
    for r in REFS:
        cells[f"da_{r}"] = (cells[f"n_matching_{r}"] / cells["n_pairs"]).where(keep)
    cells["benchmark"] = cells["task"].map(benchmark_family)
    few = cells[cells["n_pairs"] < MIN_PAIRS]
    if len(few):
        print(f"!!! RULE 5: {len(few)} (task, pair set, proxy) cells over {few['task'].nunique()} tasks have fewer than "
              f"{MIN_PAIRS} pairs and are NaN (their pair counts are in the per-task table)")
    return d, cells


def pooled(d: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Per (channel, pair set, proxy) and reference: the pooled ratio over the
    kept cells' decisions with its jackknife band, and the tasks behind it."""
    kept = cells.dropna(subset=[f"da_{REFERENCE}"])
    kept = kept[kept["channel"] != "loss"]
    dk = d.merge(kept[["task", "axes", "size"]], on=["task", "axes", "size"])
    keys = ["channel", "axes", "size"]
    n = kept.groupby(keys).agg(n_tasks=("task", "nunique"), n_pairs=("n_pairs", "sum")).reset_index()
    out = [jackknife_ratio(dk.assign(match=dk[f"match_{r}"]), keys).merge(n, on=keys).assign(reference=r) for r in REFS]
    out = pd.concat(out, ignore_index=True).rename(columns={"reliability": "da"})
    out["non_emb"] = out["size"].map(NON_EMB)
    return out.sort_values(["channel", "axes", "reference", "non_emb"]).reset_index(drop=True)


def spearman(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    ok = x.notna() & y.notna()
    n = int(ok.sum())
    if n < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan, n
    return float(spearmanr(x[ok], y[ok]).statistic), n


def rank_agreement(cells: pd.DataFrame) -> pd.DataFrame:
    """Per pair set and proxy: Spearman between the two references' DA over the
    benchmark tasks, and over the benchmarks (the mean DA of each one's tasks)."""
    rows = []
    b = cells[cells["channel"] == "benchmarks"]
    for (axes, size), g in b.groupby(["axes", "size"]):
        rho_t, n_t = spearman(g[f"da_{TARGET_SIZE}"], g[f"da_{REFERENCE}"])
        m = g.groupby("benchmark")[[f"da_{TARGET_SIZE}", f"da_{REFERENCE}"]].mean()
        rho_b, n_b = spearman(m[f"da_{TARGET_SIZE}"], m[f"da_{REFERENCE}"])
        rows.append({"axes": axes, "size": size, "rho_tasks": rho_t, "n_tasks": n_t, "rho_benchmarks": rho_b, "n_benchmarks": n_b})
    out = pd.DataFrame(rows)
    out["non_emb"] = out["size"].map(NON_EMB)
    return out.sort_values(["axes", "non_emb"]).reset_index(drop=True)


def figure_da(axes_: str, pool_: pd.DataFrame, cells: pd.DataFrame, ranks: pd.DataFrame, fams: list, path: Path) -> None:
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(15, 4.6))
    p = pool_[pool_["axes"] == axes_]
    sizes = size_order(p["size"].unique())
    for (ch, ref), g in p.groupby(["channel", "reference"]):
        g = g.sort_values("non_emb")
        ax_a.plot(g["non_emb"], g["da"], color=CHANNEL_COLOR[ch], marker="o" if ch == "benchmarks" else "s", ms=4, **STYLE[ref],
                  label=f"{'benchmark accuracy' if ch == 'benchmarks' else 'BPB'} → {ref} ({int(g['n_tasks'].median())} tasks)")
        ax_a.fill_between(g["non_emb"], g["lo"], g["hi"], color=CHANNEL_COLOR[ch], alpha=.10 if ref == REFERENCE else .06, lw=0)
    ax_a.axhline(.5, color=S.MUTED, lw=.8, ls=":")
    ax_a.set_xscale("log"); ax_a.set_xticks([NON_EMB[s] for s in sizes]); ax_a.set_xticklabels(sizes); ax_a.minorticks_off()
    ax_a.set_ylim(0, 1); ax_a.set_xlabel("proxy size"); ax_a.set_ylabel("DA-size (pooled, 90 % jackknife band)")
    ax_a.set_title(f"(a) DA-size to {REFERENCE} (solid) and to {TARGET_SIZE} (dashed), same pairs", loc="left", fontsize=8.5)
    ax_a.legend(fontsize=6.3, frameon=False, loc="lower right"); ax_a.grid(color=S.GRID, lw=.6); S.clean(ax_a)

    c = cells[(cells["axes"] == axes_) & (cells["channel"] == "benchmarks")].dropna(subset=[f"da_{REFERENCE}"])
    rng = np.random.default_rng(0)            # a jitter: DA sits on a k/n lattice and the points would stack
    for s in sizes:
        g = c[c["size"] == s]
        ax_b.scatter(g[f"da_{TARGET_SIZE}"] + rng.uniform(-.015, .015, len(g)), g[f"da_{REFERENCE}"] + rng.uniform(-.015, .015, len(g)),
                     s=9, alpha=.6, color=S.SIZE_COLOR.get(s, S.INK), label=s)
    ax_b.plot([0, 1], [0, 1], color=S.MUTED, lw=.8, ls=":")
    r = c[f"da_{TARGET_SIZE}"].corr(c[f"da_{REFERENCE}"]) if len(c) > 2 else np.nan
    ax_b.text(.02, .97, f"{len(c)} cells over {c['task'].nunique()} tasks, Pearson r = {r:.2f}", transform=ax_b.transAxes,
              va="top", fontsize=7)
    ax_b.set_xlim(-.05, 1.05); ax_b.set_ylim(-.05, 1.05)
    ax_b.set_xlabel(f"DA-size to {TARGET_SIZE}"); ax_b.set_ylabel(f"DA-size to {REFERENCE}")
    ax_b.set_title("(b) per benchmark task and proxy (jittered)", loc="left", fontsize=8.5)
    ax_b.legend(fontsize=6, frameon=False, loc="lower right", title="proxy", title_fontsize=6)
    ax_b.grid(color=S.GRID, lw=.6); S.clean(ax_b)

    rk = ranks[ranks["axes"] == axes_]
    x = np.arange(len(rk))
    for off, col, lab, n in ((-.18, "rho_tasks", "over tasks", "n_tasks"), (.18, "rho_benchmarks", "over benchmarks", "n_benchmarks")):
        ax_c.bar(x + off, rk[col], width=.34, color=S.RAMP[1] if col == "rho_tasks" else S.RAMP[3], label=lab)
        for xi, (v, k) in enumerate(zip(rk[col], rk[n])):
            ax_c.text(xi + off, (0 if pd.isna(v) else v) + (.02 if (pd.isna(v) or v >= 0) else -.02), f"{k}", ha="center",
                      va="bottom" if (pd.isna(v) or v >= 0) else "top", fontsize=5.5, color=S.MUTED)
    ax_c.axhline(0, color=S.MUTED, lw=.8)
    ax_c.set_xticks(x); ax_c.set_xticklabels(rk["size"]); ax_c.set_ylim(-1, 1)
    ax_c.set_xlabel("proxy size"); ax_c.set_ylabel(f"Spearman ρ, DA to {TARGET_SIZE} vs DA to {REFERENCE}")
    ax_c.set_title("(c) does the DA ranking of benchmarks survive the reference?", loc="left", fontsize=8.5)
    ax_c.legend(fontsize=6.5, frameon=False, loc="lower right"); ax_c.grid(color=S.GRID, lw=.6, axis="y"); S.clean(ax_c)

    top = G._header(fig, f"Is decision accuracy consistent when the reference moves from {TARGET_SIZE} to {REFERENCE}? ({axes_} pairs)",
                    f"The {len(fams)} families scored at both rungs ({', '.join(fams)}), {axes_} pairs at the grid seed (rule 15). "
                    f"Only the decisions both references score enter: the same pair of families on the same task at the same proxy, "
                    f"the task above chance at the proxy, at {TARGET_SIZE} and at {REFERENCE} (rule 1), {MIN_PAIRS} pairs or more "
                    f"(rule 5). (a) pools them per proxy, benchmark accuracy and per-language BPB apart, with the 90 % "
                    f"leave-one-family-out jackknife band; (c) is the Spearman over the benchmark tasks (and over the "
                    f"benchmarks' mean DA) between the two references, with the count above each bar.")
    fig.tight_layout(rect=(0, 0, 1, top))
    pd.concat([p.assign(panel="a"), rk.assign(panel="c")], ignore_index=True).to_csv(path.with_suffix(".csv"), index=False)
    S.save(fig, path, dpi=150)


# --- SNR -------------------------------------------------------------------------------------------------

def snr_cells(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """rq03's `rel_std` SNR per (task, rung) on the (family, task) cells scored
    at both rungs, gated at its own size (rule 1)."""
    sub = df[df["size"].isin(REFS)]
    sub = sub[sub.groupby(["family", "task"])["size"].transform("nunique") == 2].assign(bucket=lambda x: x["size"])
    rows = []
    for (task, size), g in sub.groupby(["task", "size"]):
        with np.errstate(divide="ignore", invalid="ignore"):          # a zero noise or a zero mean is NaN, not a warning
            signal, noise, snr = variant_signal_noise_snr(per_model_inputs(g, task, size), rel_std_snr)
        rows.append({"task": task, "size": size, "n_families": g["family"].nunique(), "signal": signal, "noise": noise, "snr": snr})
    out = pd.DataFrame(rows)
    out["gated"] = [not passes_gate(mask, [t], s).iloc[0] for t, s in zip(out["task"], out["size"])]
    out["log_snr"] = np.log10(out["snr"].where((out["snr"] > 0) & ~out["gated"]))
    w = out.pivot(index="task", columns="size", values=["log_snr", "n_families", "gated"])
    w.columns = [f"{v}_{s}" for v, s in w.columns]
    w = w.reset_index()
    w["benchmark"], w["channel"] = w["task"].map(benchmark_family), w["task"].map(channel)
    return w


def snr_agreement(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ch, g in w[w["channel"] != "loss"].groupby("channel"):
        x, y = g[f"log_snr_{TARGET_SIZE}"].astype(float), g[f"log_snr_{REFERENCE}"].astype(float)
        rho_t, n_t = spearman(x, y)
        m = g.assign(x=x, y=y).dropna(subset=["x", "y"]).groupby("benchmark")[["x", "y"]].median()
        rho_b, n_b = spearman(m["x"], m["y"])
        both = x.notna() & y.notna()
        rows.append({"channel": ch, "rho_tasks": rho_t, "n_tasks": n_t, "rho_benchmarks": rho_b, "n_benchmarks": n_b,
                     f"median_log_snr_{TARGET_SIZE}": x[both].median(), f"median_log_snr_{REFERENCE}": y[both].median(),
                     "share_higher_at_ref": float((y[both] > x[both]).mean()) if both.any() else np.nan})
    return pd.DataFrame(rows)


def figure_snr(w: pd.DataFrame, agree: pd.DataFrame, fams: list, path: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.8))
    x, y = f"log_snr_{TARGET_SIZE}", f"log_snr_{REFERENCE}"
    w = w.astype({x: float, y: float})
    for ch, g in w[w["channel"] != "loss"].dropna(subset=[x, y]).groupby("channel"):
        a = agree[agree["channel"] == ch].iloc[0]
        ax_a.scatter(g[x], g[y], s=10, alpha=.6, color=CHANNEL_COLOR[ch],
                     label=f"{'benchmark tasks' if ch == 'benchmarks' else 'per-language BPB'}: {len(g)}, ρ = {a['rho_tasks']:.2f}")
    m = w[w["channel"] == "benchmarks"].dropna(subset=[x, y]).groupby("benchmark").agg(x=(x, "median"), y=(y, "median"), n=("task", "size"))
    ax_b.scatter(m["x"], m["y"], s=12 + 2 * m["n"], color=S.RAMP[2], alpha=.7)
    for b, r in m.iterrows():
        ax_b.annotate(G.display(b), (r["x"], r["y"]), fontsize=5.5, xytext=(3, 2), textcoords="offset points", color=S.MUTED)
    for ax in (ax_a, ax_b):
        lim = [np.nanmin([ax.get_xlim()[0], ax.get_ylim()[0]]), np.nanmax([ax.get_xlim()[1], ax.get_ylim()[1]])]
        ax.plot(lim, lim, color=S.MUTED, lw=.8, ls=":"); ax.axhline(0, color=S.GRID, lw=.8); ax.axvline(0, color=S.GRID, lw=.8)
        ax.set_xlabel(f"log10 SNR at {TARGET_SIZE}"); ax.set_ylabel(f"log10 SNR at {REFERENCE}"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    ax_a.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax_a.set_title("(a) per task", loc="left", fontsize=8.5)
    ab = agree[agree["channel"] == "benchmarks"]
    ax_b.set_title(f"(b) per benchmark, the median over its tasks (size = tasks)"
                   + (f", ρ = {ab['rho_benchmarks'].iloc[0]:.2f} over {int(ab['n_benchmarks'].iloc[0])}" if len(ab) else ""),
                   loc="left", fontsize=8.5)
    top = G._header(fig, f"Is the SNR consistent from {TARGET_SIZE} to {REFERENCE}?",
                    f"rq03's headline SNR (`rel_std`: the relative std of the final scores across the families over the relative "
                    f"checkpoint noise in the last 20 % of each run, rule 4) at each rung, on the same {len(fams)} families "
                    f"({', '.join(fams)}), each (family, task) scored at both rungs; a task enters where it is above chance at both "
                    f"(rule 1) and its SNR is positive. ρ is the Spearman over the points; the signal rests on as few families as "
                    f"train the task's language (rule 2).")
    fig.tight_layout(rect=(0, 0, 1, top))
    agree.to_csv(path.with_suffix(".csv"), index=False)
    S.save(fig, path, dpi=150)


# --- paper -----------------------------------------------------------------------------------------------

def figure_paper(by_bench: pd.DataFrame, pool_: pd.DataFrame, path: Path, min_tasks: int = 5) -> None:
    """The key finding, bare (rule 18): the benchmarks the 3B rung lifts above
    chance (left; the benchmarks with at least `min_tasks` tasks whose share
    moves), and DA-size to 3B beside DA-size to 1.7B on the same decisions
    (right, mono-axis, the paper's pair set)."""
    b = by_bench[(by_bench["n_tasks"] >= min_tasks) & (by_bench["delta"] != 0)]
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(9.6, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
    dumbbell(ax_l, b, REFERENCE, label_n=False, paper=True)
    ax_l.set_xlabel("Share of tasks above chance")
    ax_l.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.0))
    p = pool_[pool_["axes"] == "mono-axis"]
    for (ch, ref), g in p.groupby(["channel", "reference"]):
        g = g.sort_values("non_emb")
        ax_r.plot(g["non_emb"], g["da"], color=CHANNEL_COLOR[ch], marker="o" if ch == "benchmarks" else "s", ms=4, **STYLE[ref],
                  label=f"{'Benchmark accuracy' if ch == 'benchmarks' else 'BPB'} to {ref}")
        ax_r.fill_between(g["non_emb"], g["lo"], g["hi"], color=CHANNEL_COLOR[ch], alpha=.10, lw=0)
    sizes = size_order(p["size"].unique())
    ax_r.axhline(.5, color=S.MUTED, lw=.8, ls=":")
    ax_r.set_xscale("log"); ax_r.set_xticks([NON_EMB[s] for s in sizes]); ax_r.set_xticklabels(sizes); ax_r.minorticks_off()
    ax_r.set_ylim(0, 1); ax_r.set_xlabel("Proxy size"); ax_r.set_ylabel("Decision accuracy")
    ax_r.legend(fontsize=6.5, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 1.0))
    ax_r.grid(color=S.GRID, lw=.6); S.clean(ax_r)
    fig.tight_layout()
    pd.concat([b.assign(panel="share above chance"), p.assign(panel="decision accuracy")], ignore_index=True) \
      .to_csv(path.parent / f"{path.name}.csv", index=False)
    S.save_paper(fig, path)


def figure_paper_fixed(cells: pd.DataFrame, path: Path) -> None:
    """The paper figure's DA panel over one task set per line (rule 13): per
    channel, the tasks with a kept mono-axis cell at EVERY proxy size
    (`utils.fixed_population`; a cell is kept for both references at once, so
    one set serves both), solid; dashed and muted behind it, the same pooled
    ratio over every kept cell, which is the paper panel's reading. One panel
    per reference; the per-task table carries no pairs, so no band."""
    kept = cells[(cells["axes"] == "mono-axis") & (cells["channel"] != "loss")].dropna(subset=[f"da_{REFERENCE}"])
    rows = []
    for reading, c in (("moving", kept), ("fixed", fixed_population(kept, ["channel"], "size", f"da_{REFERENCE}"))):
        g = c.groupby(["channel", "size"]).agg(n_tasks=("task", "nunique"), n_pairs=("n_pairs", "sum"),
                                               **{r: (f"n_matching_{r}", "sum") for r in REFS}).reset_index()
        rows += [g.assign(reading=reading, panel=r, value=g[r] / g["n_pairs"]) for r in REFS]
    t = pd.concat(rows, ignore_index=True).rename(columns={"size": "x"})
    t["line"] = t["channel"] + " to " + t["panel"]
    sizes = size_order(t["x"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, ref in zip(axes, REFS):
        for (ch, reading), g in t[t["panel"] == ref].groupby(["channel", "reading"]):
            g = g.set_index("x").reindex(sizes)
            kw = dict(ls="-", lw=1.6, marker="o" if ch == "benchmarks" else "s", ms=4) if reading == "fixed" else dict(ls="--", lw=1.0, alpha=.45)
            ax.plot([NON_EMB[s] for s in sizes], g["value"], color=CHANNEL_COLOR[ch], **kw)
        ax.axhline(.5, color=S.MUTED, lw=.8, ls=":")
        ax.set_xscale("log"); ax.set_xticks([NON_EMB[s] for s in sizes]); ax.set_xticklabels(sizes); ax.minorticks_off()
        ax.set_ylim(0, 1); ax.set_xlabel("Proxy size"); ax.set_title(f"Reference {ref}", loc="left", fontsize=8.5)
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    axes[0].set_ylabel("Decision accuracy")
    handles = [plt.Line2D([], [], color=CHANNEL_COLOR["benchmarks"], marker="o", lw=1.6, label="Benchmark accuracy"),
               plt.Line2D([], [], color=CHANNEL_COLOR["bpb"], marker="s", lw=1.6, label="BPB"),
               plt.Line2D([], [], color=S.INK, lw=1.6, label="Same tasks at every size"),
               plt.Line2D([], [], color=S.INK, lw=1.0, ls="--", alpha=.45, label="Tasks above chance at each size")]
    fig.legend(handles=handles, fontsize=6.5, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5, 0.0))
    fig.tight_layout()
    t[["reading", "panel", "line", "x", "value", "n_tasks"]].to_csv(path.parent / f"{path.name}.csv", index=False)
    S.save_paper(fig, path)


# --- README ----------------------------------------------------------------------------------------------

def generate_readme(pool: str, fams: list, n_pairs: dict, pool_: pd.DataFrame, cells: pd.DataFrame, ranks: pd.DataFrame,
                    agree: pd.DataFrame, rel: str, gh: str) -> None:
    if pool != CANONICAL_POOL:
        return
    few = cells[cells["n_pairs"] < MIN_PAIRS]
    fmt = lambda v: "—" if pd.isna(v) else f"{v:.2f}"
    da = pool_[pool_["channel"] == "benchmarks"].pivot_table(index=["axes", "size", "non_emb"], columns="reference",
                                                              values=["da", "lo", "hi", "n_tasks"]).reset_index().sort_values(["axes", "non_emb"])
    rows = [[r[("axes", "")], r[("size", "")], int(r[("n_tasks", REFERENCE)]),
             f"{fmt(r[('da', TARGET_SIZE)])} [{fmt(r[('lo', TARGET_SIZE)])}, {fmt(r[('hi', TARGET_SIZE)])}]",
             f"{fmt(r[('da', REFERENCE)])} [{fmt(r[('lo', REFERENCE)])}, {fmt(r[('hi', REFERENCE)])}]"] for _, r in da.iterrows()]
    rk = [[r["axes"], r["size"], fmt(r["rho_tasks"]), int(r["n_tasks"]), fmt(r["rho_benchmarks"]), int(r["n_benchmarks"])]
          for _, r in ranks.iterrows()]
    sn = [[r["channel"], fmt(r["rho_tasks"]), int(r["n_tasks"]), fmt(r["rho_benchmarks"]), int(r["n_benchmarks"]),
           fmt(r[f"median_log_snr_{TARGET_SIZE}"]), fmt(r[f"median_log_snr_{REFERENCE}"]), fmt(r["share_higher_at_ref"])]
          for _, r in agree.iterrows()]
    body = "\n\n".join([
        f"## Is the framework consistent when the reference moves from {TARGET_SIZE} to {REFERENCE}?",
        f"**DA-size to two references and the SNR at two rungs · {len(fams)} families scored at both ({', '.join(fams)}) · "
        f"multi-axis and mono-axis pairs at the grid seed · gate `{pool}` at the proxy and at {TARGET_SIZE}, the Wilson rule on "
        f"the {REFERENCE} runs · no filter.** Regenerate with "
        f"`python analysis/rq10_size_generalisation/reference_consistency.py --pool {pool}`.",
        f"**Population.** {n_pairs['multi-axis']} multi-axis and {n_pairs['mono-axis']} mono-axis pairs over the "
        f"{len(fams)} families (fewer on a task whose language not every family trains, rule 2). "
        f"A DA cell is a (task, pair set, proxy) whose decisions both references score, the task above chance at "
        f"the proxy, at {TARGET_SIZE} and at {REFERENCE}; {len(few)} cells over {few['task'].nunique()} tasks have fewer than "
        f"{MIN_PAIRS} pairs and are NaN (rule 5, pair counts in the per-task table). Bands are the 90 % leave-one-family-out "
        f"jackknife.",
        f"![DA-size to two references, multi-axis]({rel}/reference_consistency_da_size_multi_axes.png)",
        f"![DA-size to two references, mono-axis]({rel}/reference_consistency_da_size_mono_axis.png)",
        "Benchmark accuracy, pooled DA-size [90 % band]:",
        md_table(["axes", "proxy", "tasks", f"→ {TARGET_SIZE}", f"→ {REFERENCE}"], rows),
        f"Spearman between the two references' DA, over the benchmark tasks and over the benchmarks (mean DA of their tasks):",
        md_table(["axes", "proxy", "ρ tasks", "tasks", "ρ benchmarks", "benchmarks"], rk),
        f"![SNR at two rungs]({rel}/reference_consistency_snr.png)",
        f"SNR (`rel_std`) at {TARGET_SIZE} against {REFERENCE}, tasks above chance at both:",
        md_table(["channel", "ρ tasks", "tasks", "ρ benchmarks", "benchmarks", f"median log10 SNR {TARGET_SIZE}",
                  f"median log10 SNR {REFERENCE}", f"share higher at {REFERENCE}"], sn),
        f"Files: [`reference_consistency_da_size_multi_axes.png`]({gh}/reference_consistency_da_size_multi_axes.png) / "
        f"[`.csv`]({gh}/reference_consistency_da_size_multi_axes.csv), "
        f"[`reference_consistency_da_size_mono_axis.png`]({gh}/reference_consistency_da_size_mono_axis.png) / "
        f"[`.csv`]({gh}/reference_consistency_da_size_mono_axis.csv), "
        f"[`reference_consistency_da_size_per_task_both_axes.csv`]({gh}/reference_consistency_da_size_per_task_both_axes.csv), "
        f"[`reference_consistency_snr.png`]({gh}/reference_consistency_snr.png) / [`.csv`]({gh}/reference_consistency_snr.csv), "
        f"[`reference_consistency_snr_per_task.csv`]({gh}/reference_consistency_snr_per_task.csv)."])
    replace_block(OUT_ROOT / "README.md", KEY, body, f"reference_consistency.py --pool {pool}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL, help="the gate's pool and the output folder")
    ap.add_argument("--paper", action="store_true", help="only the paper figure, from the CSVs on disk")
    args = ap.parse_args()
    stage = load_pools()[args.pool].get("stage", "pretraining")
    out = OUT_ROOT / stage / args.pool
    out.mkdir(parents=True, exist_ok=True)
    if not args.paper:
        df, fams = load(args.pool)
        mask = gate_mask(df, args.pool, REFERENCE)
        d, cells = da_cells(df, mask)
        pool_, ranks = pooled(d, cells), rank_agreement(cells)
        cells.to_csv(out / "reference_consistency_da_size_per_task_both_axes.csv", index=False)
        for a in AXES:
            figure_da(a, pool_, cells, ranks, fams, out / f"reference_consistency_da_size{AXES_SUFFIX[a]}.png")
        w = snr_cells(df, mask)
        agree = snr_agreement(w)
        w.to_csv(out / "reference_consistency_snr_per_task.csv", index=False)
        figure_snr(w, agree, fams, out / "reference_consistency_snr.png")
        rel = f"{stage}/{args.pool}"
        n_pairs = d.groupby("axes").apply(lambda g: len(set(zip(g["family_a"], g["family_b"])))).to_dict()
        generate_readme(args.pool, fams, n_pairs, pool_, cells, ranks, agree, rel, f"{GITHUB}/rq10_size_generalisation/{rel}")
        print(f"--- reference_consistency: {len(fams)} families scored at {TARGET_SIZE} and {REFERENCE}, pairs {n_pairs} ---")
        print(pool_[["channel", "axes", "size", "reference", "da", "lo", "hi", "n_tasks", "n_units"]].to_string(index=False))
        print(ranks.to_string(index=False))
        print(agree.to_string(index=False))
    else:
        pool_ = pd.concat([pd.read_csv(out / f"reference_consistency_da_size{AXES_SUFFIX[a]}.csv") for a in AXES])
        # panel (a)'s rows and columns only, so the paper CSV matches the one a full run writes
        pool_ = pool_[pool_["panel"] == "a"].drop(columns=["panel", "rho_tasks", "rho_benchmarks", "n_benchmarks"])
        cells = pd.read_csv(out / "reference_consistency_da_size_per_task_both_axes.csv")
    figure_paper_fixed(cells, out / FIXED_STEM)
    by_bench = out / "gate_crossover_by_benchmark.csv"
    if by_bench.exists():
        figure_paper(pd.read_csv(by_bench), pool_, out / PAPER_STEM)
    else:
        print(f"!!! {by_bench.name} is missing: run gate_crossover.py first; the paper figure was not drawn")
