"""Which languages does a developer have to evaluate? The minimal language
panel that recovers the multilingual decision.

A multilingual benchmark is read per language, but the decision a developer
makes is one ranking of the design variants on the benchmark as a whole. This
script asks, per benchmark family, how well a PROXY at a smaller size recovers
the reference's MACRO ranking — the 1.7B ranking of the design variants by
their mean score over the panel languages — when the proxy reads

  * one language at a time (every panel language; English is one of them),
  * the same macro average at the proxy size.

The panel is the eight languages of the L8 setting (en, ru, zh, de, ja, es,
fr, it), which every regime from L8 up trains, so every family with L ≥ 8 has
a score in each of them (rule 2 keeps the L1/L2 families out by construction:
they lack the rows). Per benchmark the panel is cut to the languages above
chance at 1.7B (rule 1 at the reference; at least MIN_PANEL of them) and, for
each proxy size, a single-language view exists where the language is above
chance at that size too (rule 1 at the proxy); the proxy macro averages the
languages readable at that size. Pairs are every grid-seed pair of design
variants over every scheme (rule 15's multi-axis set), ≥ MIN_PAIRS per view
(rule 5), the band the leave-one-family-out jackknife.

What a result means: if the macro at the proxy recovers the reference macro
better than any single language, the languages' errors are independent and a
developer should evaluate the panel, not a language; if one language matches
the macro, that language alone suffices for the decision; if English trails
the others, an English-only proxy misreads the multilingual decision.

    language_panel.png / .csv   one panel per benchmark family with ≥ MIN_PANEL
                                readable languages, plus every benchmark pooled:
                                per-language lines (faint, labelled), English,
                                the proxy macro (ink, banded); per (benchmark,
                                size, view) rows in the CSV
    python analysis/rq06_language_transfer/language_panel.py --pool predictivity
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

from evals.scripts.utils.configs import fineweb_language, load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import LANGUAGE_TRANSFER  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import OVERALL, POOL, pairs_by_group  # noqa: E402
from analysis.utils import (  # noqa: E402
    GRID_SEED, MIN_PAIRS, NON_EMB, TARGET_SIZE, assign_language, benchmark_family, design_axes,
    finals, jackknife_ratio, ladder_frame, passes_gate, size_order)
from pretrain.launch_trainings import cell_fineweb_subsets  # noqa: E402

OUT_ROOT = LANGUAGE_TRANSFER
PANEL = ["en"] + [fineweb_language(s) for s in cell_fineweb_subsets(8, "A")]
MIN_PANEL = 4          # languages a benchmark needs above chance at the reference to have a macro worth recovering
MACRO, POOLED = "macro", "all benchmarks"
mpl.rcParams.update(S.RC)


def scores_by(fin: pd.DataFrame, size: str) -> dict:
    """task -> {family -> final score} at one size."""
    g = fin[fin["size"] == size]
    out: dict = {}
    for task, fam, v in zip(g["task"], g["family"], g["primary_score"]):
        out.setdefault(task, {})[fam] = v
    return out


def macro(scores: dict, tasks: list) -> dict:
    """family -> mean over `tasks`, families scored on every one of them."""
    fams = set.intersection(*[set(scores.get(t, {})) for t in tasks]) if tasks else set()
    return {f: float(np.mean([scores[t][f] for t in tasks])) for f in fams}


def decisions(proxy: dict, ref: dict, pairs: list) -> list:
    """(family_a, family_b, match) for every pair both sides score."""
    return [(a, b, int(np.sign(proxy[a] - proxy[b]) == np.sign(ref[a] - ref[b])))
            for a, b in pairs if a in proxy and b in proxy and a in ref and b in ref]


def panel_views(fin: pd.DataFrame, pairs: list, mask, sizes: list) -> pd.DataFrame:
    """One row per decision: (benchmark, size, view, family_a, family_b, match).
    `view` is a language code, MACRO, or POOLED (every benchmark's macro
    decisions together, one row per decision again)."""
    bench = fin[fin["kind"] == "benchmark"].copy()
    bench["language"] = bench["task"].map(assign_language)
    bench["benchmark"] = bench["task"].map(benchmark_family)
    bench = bench[bench["language"].isin(PANEL)]
    ref_scores = scores_by(bench, TARGET_SIZE)
    rows, panels = [], {}
    for b, g in bench.groupby("benchmark"):
        task_of = dict(zip(g["language"], g["task"]))          # one parent task per language
        ok_ref = [l for l in PANEL if l in task_of and bool(passes_gate(mask, [task_of[l]], TARGET_SIZE).iloc[0])]
        if len(ok_ref) < MIN_PANEL:
            continue
        ref = macro(ref_scores, [task_of[l] for l in ok_ref])
        panels[b] = ok_ref
        for size in sizes:
            prox = scores_by(bench, size)
            ok = [l for l in ok_ref if bool(passes_gate(mask, [task_of[l]], size).iloc[0])]
            views = {l: prox.get(task_of[l], {}) for l in ok}
            if ok:
                views[MACRO] = macro(prox, [task_of[l] for l in ok])
            for view, p in views.items():
                for a, c, m in decisions(p, ref, pairs):
                    rows.append({"benchmark": b, "size": size, "view": view, "family_a": a, "family_b": c, "match": m,
                                 "n_langs": len(ok) if view == MACRO else 1, "ref_langs": len(ok_ref)})
    d = pd.DataFrame(rows)
    pooled = d[d["view"] != MACRO].assign(benchmark=POOLED)          # every single-language decision, pooled
    pooled_macro = d[d["view"] == MACRO].assign(benchmark=POOLED)
    return pd.concat([d, pooled, pooled_macro], ignore_index=True), panels


def summarise(d: pd.DataFrame) -> pd.DataFrame:
    keys = ["benchmark", "size", "view"]
    out = jackknife_ratio(d, keys)
    n = d.groupby(keys, sort=False).agg(n_pairs=("match", "size"), n_langs=("n_langs", "first"),
                                        ref_langs=("ref_langs", "first")).reset_index()
    out = out.merge(n, on=keys)
    out = out[out["n_pairs"] >= MIN_PAIRS]                                   # rule 5
    out["non_emb"] = out["size"].map(NON_EMB)
    return out


def figure(t: pd.DataFrame, panels: dict, path: Path, pool: str) -> None:
    names = [b for b in panels if b in set(t["benchmark"])] + [POOLED]
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.1 * nrow + 0.6), sharey=True, squeeze=False)
    sizes = [s for s in size_order(t["size"].unique()) if s != TARGET_SIZE]
    for ax, b in zip(axes.ravel(), names):
        g = t[t["benchmark"] == b]
        for view, h in g.groupby("view"):
            h = h.sort_values("non_emb")
            if view == MACRO:
                ax.plot(h["non_emb"], h["reliability"], color=S.INK, lw=2, marker="o", ms=4, zorder=5,
                        label="macro over the panel")
                bb = h.dropna(subset=["lo", "hi"])
                ax.fill_between(bb["non_emb"], bb["lo"].clip(0, 1), bb["hi"].clip(0, 1), color=S.INK, alpha=.08, lw=0)
            else:
                c = S.SERIES[1] if view == "en" else S.MUTED
                ax.plot(h["non_emb"], h["reliability"], color=c, lw=1.2 if view == "en" else .7, marker="o",
                        ms=2.5, alpha=1 if view == "en" else .6, zorder=3 if view == "en" else 2,
                        label="English alone" if view == "en" else None)
                ax.annotate(view, (h["non_emb"].iloc[-1], h["reliability"].iloc[-1]), textcoords="offset points",
                            xytext=(3, 0), fontsize=5.5, color=c, va="center")
        ax.axhline(0.5, color=S.MUTED, lw=.8, ls=":")
        ax.set_xscale("log"); ax.set_ylim(0.2, 1.0)
        ax.set_xticks([NON_EMB[s] for s in sizes]); ax.set_xticklabels(sizes)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        n_ref = g["ref_langs"].iloc[0] if len(g) else 0
        ax.set_title(f"{b}  ({int(n_ref)}/{len(PANEL)} languages readable at {TARGET_SIZE})"
                     if b != POOLED else f"{POOLED}, decisions pooled", loc="left", fontsize=7)
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"DA vs the {TARGET_SIZE} macro ranking")
    axes[0, 0].legend(fontsize=6, frameon=False, loc="upper left")
    top = G._header(fig, "The minimal language panel: which languages recover the multilingual decision?",
                    f"Per benchmark family, the {GRID_SEED}-seed design-variant pairs of every scheme (multi-axis, "
                    f"rule 15) ranked by a proxy against the {TARGET_SIZE} MACRO ranking (mean score over the panel "
                    f"languages above chance at {TARGET_SIZE}, rule 1; panel = the L8 languages {', '.join(PANEL)}, "
                    f"which every family with L ≥ 8 trains). Grey: the proxy reads one language (drawn where that "
                    f"language is above chance at the proxy size); orange: English alone; ink: the proxy's own macro "
                    f"over the languages readable at that size, with its 90 % leave-one-family-out band. ≥ {MIN_PAIRS} "
                    f"pairs per point (rule 5). Last panel: every benchmark's decisions pooled. Dotted: chance. "
                    f"Gate: `{pool}`.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, out_dir: Path, t: pd.DataFrame, panels: dict) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    rows = []
    for b in list(panels) + [POOLED]:
        g = t[(t["benchmark"] == b) & (t["size"] != TARGET_SIZE)]
        if not len(g):
            continue
        rec = {"benchmark": b, "languages readable at 1.7B": len(panels.get(b, PANEL))}
        for size in [s for s in size_order(g["size"].unique())]:
            h = g[g["size"] == size]
            m = h[h["view"] == MACRO]["reliability"]
            e = h[h["view"] == "en"]["reliability"]
            singles = h[~h["view"].isin([MACRO])] if b != POOLED else h[h["view"] != MACRO]
            rec[size] = (f"macro {m.iloc[0]:.2f}" if len(m) else "macro —") + \
                        (f" / en {e.iloc[0]:.2f}" if len(e) else "") + \
                        (f" / best lang {singles['reliability'].max():.2f} ({singles.loc[singles['reliability'].idxmax(), 'view']})"
                         if len(singles) and b != POOLED else (f" / one lang {singles['reliability'].iloc[0]:.2f}" if len(singles) else ""))
        rows.append(rec)
    tab = pd.DataFrame(rows).fillna("—")
    body = "\n\n".join([
        "## The minimal language panel",
        f"Per benchmark family, decision accuracy of a proxy size against the {TARGET_SIZE} MACRO ranking of the "
        f"design variants (mean over the L8 panel languages above chance at {TARGET_SIZE}), when the proxy reads one "
        f"language, English, or its own macro over the languages readable at that size. Grid-seed pairs of every "
        f"scheme (multi-axis), ≥ {MIN_PAIRS} pairs, gate `{pool}`. A macro above every single language says the "
        f"languages' errors are independent and the panel is worth evaluating; a single language at the macro says "
        f"it suffices. Regenerate with `python analysis/rq06_language_transfer/language_panel.py --pool {pool}`.",
        md_table(list(tab.columns), tab.values.tolist()),
        f"![The minimal language panel]({stage}/{pool}/language_panel.png)"])
    replace_block(OUT_ROOT / "README.md", "language-panel", body, f"language_panel.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose gate applies and whose folder receives the outputs")
    args = p.parse_args()
    out_dir = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    out_dir.mkdir(parents=True, exist_ok=True)
    df = ladder_frame(POOL)
    fin = finals(df)
    pairs = pairs_by_group(design_axes(df), "overall", "multi-axis")[OVERALL]
    sizes = [s for s in size_order(fin["size"].unique()) if s != TARGET_SIZE]
    d, panels = panel_views(fin, pairs, load_mask(args.pool), sizes)
    t = summarise(d)
    t.to_csv(out_dir / "language_panel.csv", index=False)
    print(f"{len(panels)} benchmarks with ≥ {MIN_PANEL} panel languages readable at {TARGET_SIZE}: "
          + ", ".join(f"{b} ({len(v)})" for b, v in panels.items()))
    piv = t[t["view"].isin([MACRO, "en"]) | (t["benchmark"] == POOLED)].pivot_table(
        index=["benchmark", "view"], columns="size", values="reliability").reindex(columns=sizes).round(3)
    print(piv.to_string())
    figure(t, panels, out_dir / "language_panel.png", args.pool)
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, out_dir, t, panels)
