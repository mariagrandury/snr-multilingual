"""RQ11 — which benchmark, posed how and scored how, reads the reference's
decision from the smallest proxy?

Every benchmark task is one VARIANT of a benchmark: a `format` (the original
items, their `rf_` cloze rewrite, their `rfgm_` LLM rewrite) and a `scoring`
(accuracy, or the gold answer's bits per byte, `bbpb_`): `utils.variant`.
This RQ puts the variants of each benchmark side by side and recommends one.

Reads rq02's `da_goal_early_small_per_task_both_axes.csv` (pool `predictivity`:
per task, pair set, proxy size and evaluated tenth of the proxy's run, the DA
against the 1.7B final). Per task and pair set:

  da_size_<s>    DA-size: the proxy s at its final checkpoint
  safe_size      the smallest proxy from which DA-size >= tau holds at every
                 larger proxy with a value (rq02's rule, grids.smallest_safe)
  safe_compute   the cheapest (proxy, checkpoint) cell, as a share of the
                 reference run's compute, from which every costlier cell clears
                 tau (rq02's safe-FLOPs rule, along the DA-goal grid)

with tau = utils.RELIABLE_DA = 0.75, rule 1 at the proxy and the reference
(`passes_gate`, mask `predictivity`; a bBPB task has no chance level and
passes) and rule 5 (n_pairs >= MIN_PAIRS). The bBPB twins are evaluated at
each run's final checkpoint only until the per-item store covers every
checkpoint, so their DA-goal cells below 100 % and their `safe_compute`
are empty for now and fill in on the next refresh after the store does.

Per benchmark x variant: the languages, the share of them reliable (DA >= tau)
at each proxy, the mean DA-size, the median safe size. The recommendation per
benchmark: the variant with the smallest MEAN safe rank over its languages
(0 = 90M ... 4 = 1B, 5 = never; a median collapses to "never" whenever half
the languages never get there), ties to the higher mean DA-size; per
benchmark x language: the variant with the smallest safe size, ties to the
higher mean DA-size.

Combined and per variant: mean DA-size and share of reliable tasks per proxy,
for each variant and for all variants together, over every task and over the
PAIRED set (the (benchmark, language) cells that also have an original
accuracy value at that proxy: the only head-to-head in which the gate treats
the variants alike, since bBPB passes where accuracy is at chance).

    recipe_da_all_per_task{_multi_axes,_mono_axis}.csv          per task
    recipe_da_size_by_variant{...}.csv                          benchmark x variant
    recipe_da_size_recommendation{...}.csv                      one row per benchmark
    recipe_da_size_heatmap{...}.png/.csv                        benchmark x variant, one panel per proxy
    recipe_da_size_ladder{...}.png/.csv                         the cheapest reliable proxy per variant
    recipe_da_size_profiles{...}.png/.csv                       share of languages reliable per proxy, per benchmark
    recipe_da_size_variants{...}.png/.csv                       combined vs per variant

    python analysis/rq11_evaluation_recipe/recipe.py --pool predictivity
"""

from __future__ import annotations

import argparse
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
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, fmt, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY, EVALUATION_RECIPE  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask, task_n_items  # noqa: E402
from analysis.utils import (AXES_SUFFIX, BBPB, FORMATS, MIN_PAIRS, RELIABLE_DA, SCORINGS, SMALL_SIZES,  # noqa: E402
                            TARGET_SIZE, assign_language, benchmark_family, passes_gate, variant)

TAU = RELIABLE_DA
FORMAT_NAME = {"original": "original", "rf": "RF", "rfgm": "LLM-RF"}
SCORING_NAME = {"acc": "accuracy", "bbpb": "bBPB"}
VARIANTS = [(f, s) for s in SCORINGS for f in FORMATS]
ALL = "all variants"
LEVELS = list(SMALL_SIZES)                       # the safe-size levels; NEVER_CODE past the last
MIN_DRAWN = 5                                    # a mean over fewer tasks is kept in the CSV but not drawn
mpl.rcParams.update(S.RC)


def vname(f: str, s: str) -> str:
    return f"{FORMAT_NAME[f]} · {SCORING_NAME[s]}"


def per_task(early: pd.DataFrame, mask: pd.DataFrame | None) -> pd.DataFrame:
    """One row per task: its variant, DA-size per proxy (gated and rule-5 cells
    NaN), the safe size and the safe compute share."""
    e = early[(early["n_pairs"] >= MIN_PAIRS) & early["da"].notna()].copy()            # rule 5
    ok = {(t, s): bool(passes_gate(mask, [t], s, TARGET_SIZE).iloc[0]) for t, s in set(zip(e["task"], e["proxy_size"]))}
    e["gated"] = [not ok[(t, s)] for t, s in zip(e["task"], e["proxy_size"])]              # rule 1, proxy and reference
    fin = e[e["frac"] == 1.0]
    wide = fin[~fin["gated"]].pivot_table(index="task", columns="proxy_size", values="da").reindex(columns=LEVELS)
    tasks = sorted(set(e["task"]))
    out = pd.DataFrame(index=pd.Index(tasks, name="task"))
    for s in LEVELS:
        out[f"da_size_{s}"] = wide[s] if s in wide else np.nan
    gated = fin.pivot_table(index="task", columns="proxy_size", values="gated", aggfunc="all").reindex(index=tasks, columns=LEVELS)
    for s in LEVELS:
        out[f"gated_{s}"] = gated[s].fillna(False).astype(bool) if s in gated else False
    out["gated_everywhere"] = fin.groupby("task")["gated"].all().reindex(tasks).fillna(True)
    level = G.smallest_safe(wide.ge(TAU).where(wide.notna()))
    out["safe_size"] = level.reindex(tasks)
    # the cheapest cell from which every costlier one clears tau, along compute (rq02's safe-FLOPs rule)
    g = e[~e["gated"]].assign(share=lambda d: d["compute"] / d["ref_compute"]).sort_values("share")
    out["safe_compute"] = pd.Series({t: _safe_share(x) for t, x in g.groupby("task")}).reindex(tasks)
    out = out.reset_index()
    v = out["task"].map(variant)
    out["format"], out["scoring"] = v.str[0], v.str[1]
    original = out["task"].str.replace(f"^{BBPB}", "", regex=True)
    out["benchmark"] = original.map(lambda t: G.base(benchmark_family(t)))
    out["language"] = out["task"].map(assign_language)
    out["n_items"] = original.map(task_n_items)
    return out


def _safe_share(g: pd.DataFrame) -> float:
    ok = (g["da"] >= TAU).to_numpy()
    i = G.smallest_safe(pd.DataFrame([ok])).iloc[0]
    return np.nan if i != i else (-1.0 if i < 0 else float(min(g["share"].iloc[int(i)], 1.0)))


def recommend(t: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """benchmark x variant summary, and the recommended variant per benchmark;
    `t` gains `recommended` per (benchmark, language)."""
    never = len(LEVELS)
    known = t["safe_size"].notna()                                            # NaN: no proxy had a value
    t["safe_rank"] = t["safe_size"].where(~known | (t["safe_size"] >= 0), never)   # never ranks after every size
    rows = []
    for (b, f, s), g in t.groupby(["benchmark", "format", "scoring"]):
        r = {"benchmark": b, "format": f, "scoring": s, "variant": vname(f, s), "languages": g["language"].nunique(),
             "mean_safe_rank": g["safe_rank"].mean(), "median_safe_rank": g["safe_rank"].median(),
             "safe_by_1B": (g["safe_size"].dropna() >= 0).mean() if g["safe_size"].notna().any() else np.nan}
        for size in LEVELS:
            d = g[f"da_size_{size}"]
            r[f"mean_da_size_{size}"] = d.mean()
            r[f"reliable_share_{size}"] = (d >= TAU).sum() / d.notna().sum() if d.notna().any() else np.nan
            r[f"all_gated_{size}"] = bool(g[f"gated_{size}"].all())
        rows.append(r)
    by = pd.DataFrame(rows)
    by["median_safe_size"] = by["median_safe_rank"].map(lambda x: np.nan if x != x else
                                                        ("never" if x >= never else LEVELS[int(np.floor(x))]))
    by = by.dropna(subset=["median_safe_rank"])
    by["mean_da_size"] = by[[f"mean_da_size_{s}" for s in LEVELS]].mean(axis=1)
    best = (by.sort_values(["benchmark", "mean_safe_rank", "mean_da_size"], ascending=[True, True, False])
              .groupby("benchmark").head(1))
    rec = best[["benchmark", "variant", "format", "scoring", "mean_safe_rank", "median_safe_size", "safe_by_1B", "languages",
                "mean_da_size"]]
    t["mean_da_size"] = t[[f"da_size_{s}" for s in LEVELS]].mean(axis=1)
    order = t.sort_values(["benchmark", "language", "safe_rank", "mean_da_size"], ascending=[True, True, True, False])
    first = order.dropna(subset=["safe_size"]).groupby(["benchmark", "language"]).head(1).index
    t["recommended"] = t.index.isin(first)
    return by, rec.reset_index(drop=True)


def overview(t: pd.DataFrame) -> pd.DataFrame:
    """Per proxy: mean DA-size and the share of reliable tasks, for every
    variant and for all variants together, over every task and the paired set."""
    rows = []
    for s in LEVELS:
        c = f"da_size_{s}"
        have = t.dropna(subset=[c])
        orig = set(zip(*have.loc[(have["format"] == "original") & (have["scoring"] == "acc"), ["benchmark", "language"]].T.values))
        paired = have[[(b, l) in orig for b, l in zip(have["benchmark"], have["language"])]]
        for pop, d in (("every task", have), ("paired", paired)):
            groups = [(vname(f, sc), d[(d["format"] == f) & (d["scoring"] == sc)]) for f, sc in VARIANTS] + [(ALL, d)]
            for name, g in groups:
                if len(g):
                    rows.append({"population": pop, "variant": name, "size": s, "mean_da_size": g[c].mean(),
                                 "reliable_share": (g[c] >= TAU).mean(), "n_tasks": len(g),
                                 "n_benchmarks": g["benchmark"].nunique()})
    return pd.DataFrame(rows)


def heatmap(by: pd.DataFrame, path: Path, note: str) -> None:
    cols = [vname(f, s) for f, s in VARIANTS if vname(f, s) in set(by["variant"])]
    rows = sorted(set(by["benchmark"]), key=G.paper_name)
    fig, axes = plt.subplots(1, len(LEVELS), figsize=(2.3 * len(LEVELS) + 2.5, 0.28 * len(rows) + 2.2), sharey=True)
    tables = []
    for ax, s in zip(axes, LEVELS):
        piv = by.pivot_table(index="benchmark", columns="variant", values=f"mean_da_size_{s}").reindex(index=rows, columns=cols)
        cnt = by.pivot_table(index="benchmark", columns="variant", values="languages").reindex(index=rows, columns=cols)
        gated = by.pivot_table(index="benchmark", columns="variant", values=f"all_gated_{s}", aggfunc="all").reindex(index=rows, columns=cols)
        tables.append(G.matrix_ax(ax, piv.rename(index=G.paper_name), f"{s} proxy", cnt=cnt.rename(index=G.paper_name),
                                  vmin=0.3, vmax=1.0, center=TAU, cmap=S.DIV, fmt="{:.2f}",
                                  gated=gated.rename(index=G.paper_name).astype(float).fillna(0).astype(bool)))
        ax.tick_params(axis="x", labelrotation=60)
    G.save_highlights(fig, path.parent, f"Which variant of each benchmark reads the {TARGET_SIZE} decision? (τ = {TAU:g})",
                      note + f" Cell = mean DA-size over the benchmark's languages; colour centred on τ = {TAU:g}; "
                      "small number = languages; grey = every language at chance (rule 1), white = no value.", tables, name=path.stem)


def ladder(by: pd.DataFrame, rec: pd.DataFrame, path: Path, note: str) -> None:
    """Per benchmark, a marker per variant at its mean safe rank over the
    languages (0 = 90M ... 4 = 1B, 5 = never): how small a proxy each way of
    evaluating the benchmark needs, on average. Marker size = share of
    languages safe by 1B; the recommended variant is ringed."""
    rows = sorted(set(by["benchmark"]), key=lambda b: (by.loc[by["benchmark"] == b, "mean_safe_rank"].min(), G.paper_name(b)))
    colour = {"original": S.INK, "rf": S.SERIES[0], "rfgm": S.SERIES[2]}
    marker = {"acc": "o", "bbpb": "D"}
    fig, ax = plt.subplots(figsize=(8.5, 0.3 * len(rows) + 1.8))
    offset = {v: (i - (len(VARIANTS) - 1) / 2) * 0.11 for i, v in enumerate(VARIANTS)}
    picked = set(zip(rec["benchmark"], rec["format"], rec["scoring"]))
    tab = []
    for y, b in enumerate(rows):
        for r in by[by["benchmark"] == b].itertuples():
            x = r.mean_safe_rank
            dy = offset[(r.format, r.scoring)]
            ax.scatter(x, y + dy, s=12 + 70 * r.safe_by_1B, marker=marker[r.scoring], color=colour[r.format], zorder=3,
                       edgecolor=S.SERIES[1] if (b, r.format, r.scoring) in picked else "none", linewidth=1.4)
            tab.append({"panel": "ladder", "row": G.paper_name(b), "col": r.variant, "value": r.mean_safe_rank})
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([G.paper_name(b) for b in rows], fontsize=7)
    ax.set_xticks(range(len(LEVELS) + 1)); ax.set_xticklabels(LEVELS + ["never"])
    ax.set_xlim(-0.5, len(LEVELS) + 0.5); ax.invert_yaxis(); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    ax.set_xlabel(f"smallest proxy whose DA-size stays ≥ τ = {TAU:g}, mean over languages ('never' counts as one step "
                  "past 1B)", fontsize=7.5)
    handles = ([plt.Line2D([], [], ls="", marker="o", color=colour[f], label=FORMAT_NAME[f]) for f in FORMATS]
               + [plt.Line2D([], [], ls="", marker=marker[s], color=S.MUTED, label=SCORING_NAME[s]) for s in SCORINGS]
               + [plt.Line2D([], [], ls="", marker="o", color="white", markeredgecolor=S.SERIES[1], label="recommended")])
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    G.save_highlights(fig, path.parent, "The cheapest reliable proxy, per benchmark and way of evaluating it",
                      note + " Marker size = share of the benchmark's languages safe by 1B; ringed = the recommended "
                      "variant (the smallest mean safe rank, then the higher mean DA-size).",
                      [pd.DataFrame(tab)], name=path.stem)


def profiles(by: pd.DataFrame, rec: pd.DataFrame, path: Path, note: str) -> None:
    """One small panel per benchmark: the share of its languages reliable
    (DA-size >= tau) at each proxy, one line per variant, the recommended one
    thick: where each way of evaluating it starts to read the reference."""
    rows = sorted(set(by["benchmark"]), key=G.paper_name)
    ncol = 6
    fig, axes = plt.subplots(int(np.ceil(len(rows) / ncol)), ncol, figsize=(2.6 * ncol, 2.0 * np.ceil(len(rows) / ncol) + 0.8),
                             sharex=True, sharey=True, squeeze=False)
    colour = {"original": S.INK, "rf": S.SERIES[0], "rfgm": S.SERIES[2]}
    picked = dict(zip(rec["benchmark"], zip(rec["format"], rec["scoring"])))
    tab = []
    for ax, b in zip(axes.ravel(), rows):
        for r in by[by["benchmark"] == b].itertuples():
            y = [getattr(r, f"reliable_share_{s}") for s in LEVELS]
            best = picked.get(b) == (r.format, r.scoring)
            ax.plot(range(len(LEVELS)), y, color=colour[r.format], ls="-" if r.scoring == "acc" else ":",
                    lw=2.4 if best else 1.1, marker="o", ms=2.5)
            tab += [{"panel": G.paper_name(b), "row": r.variant, "col": s, "value": v} for s, v in zip(LEVELS, y)]
        n = int(by.loc[by["benchmark"] == b, "languages"].max())
        ax.set_title(f"{G.paper_name(b)} ({n} lang.)", loc="left", fontsize=7.5)
        ax.set_ylim(-0.03, 1.03); ax.grid(color=S.GRID, lw=.5); S.clean(ax)
    spare = axes.ravel()[len(rows):]
    for ax in spare:
        ax.set_axis_off()
    for ax in axes[-1]:
        ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS, fontsize=6.5, rotation=45)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of languages\nreliable", fontsize=7)
    handles = ([plt.Line2D([], [], color=colour[f], label=FORMAT_NAME[f]) for f in FORMATS]
               + [plt.Line2D([], [], color=S.MUTED, ls="-" if sc == "acc" else ":", label=SCORING_NAME[sc]) for sc in SCORINGS]
               + [plt.Line2D([], [], color=S.MUTED, lw=2.4, label="recommended")])
    (spare[0] if len(spare) else axes.ravel()[-1]).legend(handles=handles, fontsize=7, frameon=False, loc="center")
    G.save_highlights(fig, path.parent, f"Where each way of evaluating a benchmark starts to read the {TARGET_SIZE} decision",
                      note + f" Line = share of the benchmark's languages whose DA-size is ≥ τ = {TAU:g} at that proxy; thick "
                      "= the recommended variant.", [pd.DataFrame(tab)], name=path.stem)


def variants_figure(ov: pd.DataFrame, t: pd.DataFrame, path: Path, note: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    tables = []
    for ax, pop in zip(axes[:2], ("every task", "paired")):
        d = ov[ov["population"] == pop]
        for i, name in enumerate([vname(f, s) for f, s in VARIANTS] + [ALL]):
            g = d[d["variant"] == name].set_index("size").reindex(LEVELS)
            g["mean_da_size"] = g["mean_da_size"].where(g["n_tasks"] >= MIN_DRAWN)
            if g["mean_da_size"].isna().all():
                continue
            f, s = next(((f, s) for f, s in VARIANTS if vname(f, s) == name), (None, None))
            style = dict(color=S.INK, ls="--", lw=2) if name == ALL else dict(
                color={"original": S.MUTED, "rf": S.SERIES[0], "rfgm": S.SERIES[2]}[f], ls="-" if s == "acc" else ":", lw=1.4)
            ax.plot(range(len(LEVELS)), g["mean_da_size"], marker="o", ms=3.5, label=name, **style)
            tables.append(g.reset_index().assign(panel=pop, row=name, col=lambda x: x["size"], value=lambda x: x["mean_da_size"])
                          [["panel", "row", "col", "value"]])
        ax.axhline(TAU, color=S.MUTED, lw=.8, ls=":"); ax.axhline(0.5, color=S.GRID, lw=.8)
        ax.set_xticks(range(len(LEVELS))); ax.set_xticklabels(LEVELS); ax.set_ylim(0.3, 1.0)
        ax.set_title(f"Mean DA-size, {pop}" + (" (cells with an original-accuracy value)" if pop == "paired" else ""),
                     loc="left", fontsize=8.5)
        ax.set_xlabel("proxy size"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    axes[0].set_ylabel(f"DA-size (mean over tasks; dotted = τ = {TAU:g})")
    axes[1].legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=4)
    lv = t.assign(variant=[vname(f, s) for f, s in zip(t["format"], t["scoring"])],
                  level=t["safe_size"].where(~t["gated_everywhere"], G.GATED))           # grey = the gate, blank = no value
    level = lv.pivot_table(index="variant", columns="task", values="level", aggfunc="first")
    tables.append(G.stack_ax(axes[2], level, f"Smallest safe proxy (DA-size ≥ τ = {TAU:g}), share of each variant's tasks",
                             levels=LEVELS, xlabel="share of tasks", rows=[v for v in (vname(f, s) for f, s in VARIANTS)
                                                                         if v in level.index], name=str, legend_cols=3))
    G.save_highlights(fig, path.parent, f"Every way of evaluating a benchmark, together and apart (τ = {TAU:g})",
                      note + f" A point resting on fewer than {MIN_DRAWN} tasks is not drawn (it is in the CSV).",
                      tables, name=path.stem)


def readme(pool: str, rec: pd.DataFrame, ov: pd.DataFrame, t: pd.DataFrame) -> None:
    rel = "pretraining/" + pool
    # a mean over fewer than MIN_DRAWN tasks is blanked here as in the figures (it stays in the CSV)
    ov = ov.assign(mean_da_size=ov["mean_da_size"].where(ov["n_tasks"] >= MIN_DRAWN))
    head = ov[(ov["population"] == "every task")].pivot_table(index="variant", columns="size", values="mean_da_size")
    head = head.reindex(index=[v for v in [vname(f, s) for f, s in VARIANTS] + [ALL] if v in head.index], columns=LEVELS)
    pr = ov[(ov["population"] == "paired")].pivot_table(index="variant", columns="size", values="mean_da_size")
    pr = pr.reindex(index=[v for v in head.index if v in pr.index], columns=LEVELS)
    silent = sorted(set(t["benchmark"]) - set(rec["benchmark"]), key=G.paper_name)    # no variant has any value
    rec_rows = [[G.paper_name(r.benchmark), r.variant, fmt(r.mean_safe_rank, 2), r.median_safe_size, fmt(r.safe_by_1B),
                 int(r.languages), fmt(r.mean_da_size)]
                for r in rec.sort_values(["mean_safe_rank", "mean_da_size"], ascending=[True, False]).itertuples()] \
        + [[G.paper_name(b), "— (no variant has a value: every cell at chance or under the pair minimum)", "", "", "",
            str(t.loc[t["benchmark"] == b, "language"].nunique()), ""] for b in silent]
    body = "\n\n".join([
        "## Results",
        f"DA-size against the {TARGET_SIZE} final, multi-axis pairs (rule 15), pool `{pool}`, gate `predictivity` at the "
        f"proxy and the reference (a bBPB task has no chance level and passes), ≥ {MIN_PAIRS} pairs; reliable = DA-size "
        f"≥ τ = {TAU:g} (`utils.RELIABLE_DA`); safe size = the smallest proxy from which it stays ≥ τ at every larger "
        f"proxy. {len(t)} tasks over {t['benchmark'].nunique()} benchmarks. The bBPB twins are read at final "
        f"checkpoints only for now; a mean over fewer than {MIN_DRAWN} tasks is left blank. Regenerate with "
        f"`python analysis/rq11_evaluation_recipe/recipe.py --pool {pool}`.",
        f"**The recommendation** (per benchmark: the variant with the smallest mean safe rank over its languages — 0 = "
        f"safe from 90M, 4 = from 1B, 5 = never — then the higher mean DA-size; τ = {TAU:g}), best first:",
        md_table(["benchmark", "evaluate it as", "mean safe rank", "median safe size", "languages safe by 1B", "languages",
                  "mean DA-size"], rec_rows),
        f"**Every variant together and apart** — mean DA-size per proxy over every task (τ = {TAU:g}):",
        md_table(["variant"] + LEVELS, [[v] + [fmt(x) for x in r] for v, r in head.iterrows()]),
        "The same over the paired cells (a (benchmark, language) that also has an original-accuracy value at that "
        "proxy, so the gate treats every variant alike):",
        md_table(["variant"] + LEVELS, [[v] + [fmt(x) for x in r] for v, r in pr.iterrows()]),
        f"![Variants together and apart]({rel}/recipe_da_size_variants_multi_axes.png)",
        f"![Benchmark x variant heat map]({rel}/recipe_da_size_heatmap_multi_axes.png)",
        f"![The cheapest reliable proxy]({rel}/recipe_da_size_ladder_multi_axes.png)",
        f"![Reliability profiles per benchmark]({rel}/recipe_da_size_profiles_multi_axes.png)"])
    replace_block(EVALUATION_RECIPE / "README.md", "results", body, f"recipe.py --pool {pool}")


def main(pool: str) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = EVALUATION_RECIPE / stage / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    early = pd.read_csv(DECISION_ACCURACY / stage / pool / "da_goal_early_small_per_task_both_axes.csv")
    early = early[~early["task"].str.startswith("bpb_") & (early["task"] != "train_loss")]          # benchmarks only
    mask = load_mask(pool)
    for axes, g in early.groupby("axes"):
        sfx = AXES_SUFFIX[axes]
        t = per_task(g, mask)
        by, rec = recommend(t)
        ov = overview(t)
        t.assign(axes=axes).to_csv(out_dir / f"recipe_da_all_per_task{sfx}.csv", index=False)
        by.assign(axes=axes).to_csv(out_dir / f"recipe_da_size_by_variant{sfx}.csv", index=False)
        rec.assign(axes=axes).to_csv(out_dir / f"recipe_da_size_recommendation{sfx}.csv", index=False)
        ov.assign(axes=axes).to_csv(out_dir / f"recipe_da_size_overview{sfx}.csv", index=False)
        note = (f"DA-size against the {TARGET_SIZE} final, {axes} pairs of `{pool}`, gate `predictivity` at the proxy and "
                f"the reference (bBPB passes: no chance level), ≥ {MIN_PAIRS} pairs; τ = {TAU:g}; bBPB at final "
                "checkpoints only.")
        heatmap(by, out_dir / f"recipe_da_size_heatmap{sfx}.png", note)
        ladder(by, rec, out_dir / f"recipe_da_size_ladder{sfx}.png", note)
        profiles(by, rec, out_dir / f"recipe_da_size_profiles{sfx}.png", note)
        variants_figure(ov, t, out_dir / f"recipe_da_size_variants{sfx}.png", note)
        print(f"[{axes}] {len(t)} tasks over {t['benchmark'].nunique()} benchmarks; recommendation:")
        print(rec.to_string(index=False))
        if axes == "multi-axis" and pool == CANONICAL_POOL:
            readme(pool, rec, ov, t)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    main(p.parse_args().pool)
