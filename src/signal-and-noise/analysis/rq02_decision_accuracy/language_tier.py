"""Do the high-resource languages rank the design variants more reliably?

The L8 figures restrict the TASKS to the eight languages of the L8 setting and
still draw one line per language-count regime, so they answer "does the
regime matter on these benchmarks", not "are these languages easier to read".
This script asks the second question directly, on the pooled decisions:

  * a language's TIER is the smallest scheme-A regime that trains it — the
    lists are nested (L8 ⊂ L15 ⊂ L30 ⊂ L50), so a tier is also how many
    regimes train the language and, within any one regime, how large its
    share is (an L8 language has the largest share at every L that trains it).
    Tier `L8` is the eight high-resource languages, `L50` the twenty the L50
    mixture alone trains.
  * one line per tier over the proxy sizes: the pooled `all pairs` decision
    reliability (every grid-seed pair, rule 15's multi-axis set) read over the
    gated tasks in that tier's languages, with its leave-one-family-out band
    and its task count at every point. Panel (a) is every gated task — the
    inference version, no selection on DA; panel (b) the `above_66_size`
    tasks, a cut on the quantity drawn, kept for continuity with the paper
    figure and read as a conditional.
  * the per-language view: one point per language, its pooled reliability at
    one proxy size against its share of the L50 mixture (the one setting that
    trains every language; the share is of ALL tokens, English being 50 %),
    languages with fewer than MIN_LANG_TASKS gated tasks left out (rule 8),
    Spearman ρ over languages in the corner. That is the direct test of "more
    of a language → its benchmarks decide more reliably".

Caveats the figures carry: a tier pools benchmarks of very different families
(the L50 tier is mostly Belebele and Global-MMLU twins), so a tier gap can be a
family gap; the pairs are the same design variants for every tier, but a task
only counts where both members train its language (rule 2), so the L50 tier
reads fewer pairs per task (its languages are trained by the L50 cells alone).

    reliability_by_language_tier.png / .csv   (a) all gated tasks, (b) above_66_size tasks
    reliability_vs_language_share.png / .csv  per language at each proxy size (rows: size)

    python analysis/rq02_decision_accuracy/language_tier.py --pool predictivity
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
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq02_decision_accuracy.reliable_tasks import load_reliable  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import (  # noqa: E402
    L_COLOUR, OVERALL, POOL, TAU, aggregate, decisions, decorate, keep_cells, pair_axis, pairs_by_group,
    reliability)
from analysis.utils import (  # noqa: E402
    GRID_SEED, MIN_LANG_TASKS, MIN_PAIRS, NON_EMB, TARGET_SIZE, assign_language, design_axes, finals,
    ladder_frame, language_token_share, size_order)
from pretrain.launch_trainings import cell_languages  # noqa: E402

OUT_ROOT = DECISION_ACCURACY
TIER_LS = [8, 15, 30, 50]
TIERS = [f"L{L}" for L in TIER_LS]
TIER_LABEL = {"L8": "L8 languages (in every regime)", "L15": "L15-only languages", "L30": "L30-only languages",
              "L50": "L50-only languages"}
VARIANTS = (("", "(a) every gated task — no selection on DA"),
            ("above_66_size", "(b) tasks with median DA-size ≥ 0.66 — a cut on the quantity drawn"))
mpl.rcParams.update(S.RC)


def tier_of_language() -> dict[str, str]:
    """language -> the smallest scheme-A regime that trains it."""
    out: dict[str, str] = {}
    for L in TIER_LS:
        for lang in cell_languages(L, "A"):
            out.setdefault(lang, f"L{L}")
    return out


def pooled(cells: pd.DataFrame, dec: pd.DataFrame, pool: str, axis_of: dict, label: str) -> pd.DataFrame:
    """The `all pairs` benchmark line over `cells`, decorated, renamed `label`."""
    if not len(keep_cells(cells, pool)):
        return pd.DataFrame()
    out = decorate(aggregate(cells, pool, TAU), dec, cells, pool, ["population", "group", "size"], axis_of)
    out = out[(out["population"] == "all benchmarks") & (out["group"] == OVERALL)].copy()
    out["group"] = label
    return out


def figure_tiers(table: pd.DataFrame, path: Path, pool: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.6), sharey=True)
    sizes = size_order(table["size"].unique())
    for ax, (variant, title) in zip(axes, VARIANTS):
        t = table[table["variant"] == (variant or "all")]
        for tier in TIERS:
            g = t[(t["group"] == tier) & (t["size"] != TARGET_SIZE)].sort_values("non_emb")
            if not len(g):
                continue
            c = L_COLOUR[tier]
            ax.plot(g["non_emb"], g["reliability"], color=c, marker="o", ms=4, lw=1.4, label=TIER_LABEL[tier], zorder=3)
            b = g.dropna(subset=["lo", "hi"])
            if len(b):
                ax.fill_between(b["non_emb"], b["lo"].clip(0, 1), b["hi"].clip(0, 1), color=c, alpha=.10, lw=0, zorder=1)
            for x, y, n in zip(g["non_emb"], g["reliability"], g["n_tasks"]):
                ax.annotate(f"{int(n)}", (x, y), textcoords="offset points", xytext=(0, -9), ha="center",
                            fontsize=5.6, color=c)
        ax.axhline(0.5, color=S.MUTED, lw=.8, ls=":")
        ax.set_xscale("log"); ax.set_ylim(0.3, 1.0)
        ax.set_xticks([NON_EMB[s] for s in sizes if s != TARGET_SIZE])
        ax.set_xticklabels([s for s in sizes if s != TARGET_SIZE])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())      # no 2×10⁸ labels between the sizes
        ax.set_xlabel("proxy size (non-embedding parameters, log)")
        ax.set_title(title, loc="left", fontsize=8.5)
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    axes[0].set_ylabel(f"decision reliability vs {TARGET_SIZE} final (pooled)")
    axes[0].legend(fontsize=6.5, frameon=False, loc="upper left")
    top = G._header(fig, "Decision reliability by language tier: are the high-resource languages easier to read?",
                    f"One line per tier — the smallest scheme-A regime that trains the language (nested lists, so also "
                    f"how many regimes train it and how large its share is) — over the {GRID_SEED}-seed design-variant "
                    f"pairs of every scheme (multi-axis, rule 15), pooled over the gated benchmark tasks in the tier's "
                    f"languages: the proxy's final ranking against the {TARGET_SIZE} final's, ≥ {MIN_PAIRS} pairs per "
                    f"task (rule 5), above chance at both sizes (rule 1), the number under a point its task count, the "
                    f"band the 90 % leave-one-family-out interval. Dotted: chance. (a) supports inference; (b) is "
                    f"conditional on DA-size. A tier also differs in benchmark mix, so a gap between tiers is not a "
                    f"language effect alone. Gate: `{pool}`.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def figure_share(per_lang: pd.DataFrame, path: Path, pool: str) -> None:
    sizes = [s for s in size_order(per_lang["size"].unique()) if s != TARGET_SIZE]
    fig, axes = plt.subplots(1, len(sizes), figsize=(3.4 * len(sizes), 4.2), sharey=True, squeeze=False)
    for ax, size in zip(axes.ravel(), sizes):
        g = per_lang[per_lang["size"] == size]
        for tier in TIERS:
            h = g[g["tier"] == tier]
            if len(h):
                ax.scatter(h["share_L50"], h["reliability"], s=12 + 2 * h["n_tasks"], color=L_COLOUR[tier], alpha=.8,
                           lw=.4, edgecolor=S.INK, label=TIER_LABEL[tier], zorder=3)
        for x, y, lang in zip(g["share_L50"], g["reliability"], g["language"]):
            ax.annotate(lang, (x, y), textcoords="offset points", xytext=(3, 3), fontsize=5.8, color=S.INK)
        rho = g["share_L50"].corr(g["reliability"], method="spearman") if len(g) >= 3 else np.nan
        ax.text(0.03, 0.97, f"{size} proxy\nSpearman ρ = {rho:.2f}\n{len(g)} languages", transform=ax.transAxes,
                va="top", ha="left", fontsize=7.5, bbox=dict(boxstyle="round,pad=0.3", fc=S.SURFACE, ec=S.GRID, lw=.6))
        ax.axhline(0.5, color=S.MUTED, lw=.8, ls=":")
        ax.set_xscale("log"); ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("share of the L50 mixture (log; en = 0.5)")
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    axes[0, 0].set_ylabel(f"decision reliability vs {TARGET_SIZE} final (pooled)")
    axes[0, -1].legend(fontsize=6, frameon=False, loc="lower right")
    top = G._header(fig, "Per-language decision reliability against the language's share of training tokens",
                    f"One point per language with ≥ {MIN_LANG_TASKS} gated benchmark tasks (rule 8), at each proxy "
                    f"size: the pooled reliability of every {GRID_SEED}-seed design-variant pair (multi-axis) on the "
                    f"language's tasks, against the language's share of ALL tokens in the L50 scheme-A mixture, the one "
                    f"setting that trains every language. Marker area grows with the task count; colour is the tier of "
                    f"`reliability_by_language_tier.png`. No selection on DA. The share is a proxy for resource level, "
                    f"not the exposure of any one pair's members (those differ by regime). Gate: `{pool}`.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, out_dir: Path, table: pd.DataFrame, per_lang: pd.DataFrame) -> None:
    stage = load_pools()[pool].get("stage", "pretraining")
    rows = []
    for tier in TIERS:
        rec = {"tier": TIER_LABEL[tier]}
        for variant, _ in VARIANTS:
            g = table[(table["variant"] == (variant or "all")) & (table["group"] == tier) & (table["size"] != TARGET_SIZE)]
            g = g.sort_values("non_emb")
            rec[variant or "all gated tasks"] = ("—" if not len(g) else
                                                 f"{g['reliability'].iloc[0]:.2f} → {g['reliability'].iloc[-1]:.2f} "
                                                 f"[{int(g['n_tasks'].iloc[0])}–{int(g['n_tasks'].iloc[-1])} tasks]")
        rows.append(rec)
    t = pd.DataFrame(rows)
    rho = {s: g["share_L50"].corr(g["reliability"], method="spearman")
           for s, g in per_lang.groupby("size") if s != TARGET_SIZE and len(g) >= 3}
    body = "\n\n".join([
        "## Decision reliability by language tier",
        f"The pooled `all pairs` line of `scale_convergence.py` read over the gated tasks of one language TIER — the "
        f"smallest scheme-A regime that trains the language (L8: the eight high-resource languages every regime trains; "
        f"L50: the twenty only the L50 mixture trains). Reliability at the smallest → largest proxy [task count]. "
        f"`reliability_vs_language_share.png` is the per-language version: reliability against the language's share of "
        f"the L50 mixture, Spearman ρ over languages "
        + ", ".join(f"{s} {v:.2f}" for s, v in rho.items()) + ". "
        f"Both are unfiltered; a tier also differs in benchmark mix. Regenerate with "
        f"`python analysis/rq02_decision_accuracy/language_tier.py --pool {pool}`.",
        md_table(list(t.columns), t.values.tolist()),
        f"![Reliability by language tier]({stage}/{pool}/reliability_by_language_tier.png)",
        f"![Reliability against language share]({stage}/{pool}/reliability_vs_language_share.png)"])
    replace_block(OUT_ROOT / "README.md", "language-tier", body, f"language_tier.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose gate applies and whose folder receives the outputs")
    args = p.parse_args()
    out_dir = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    df = ladder_frame(POOL)
    fin = finals(df)
    attrs = design_axes(df)
    groups, axis_of = pairs_by_group(attrs, "overall", "multi-axis"), pair_axis(attrs)
    sizes = size_order(fin["size"].unique())
    dec = decisions(fin.assign(frac=1.0), groups, sizes, fin)
    cells_all = reliability(dec)
    lang_of = pd.Series({t: assign_language(t) for t in cells_all["task"].unique()})
    tier = tier_of_language()
    tables = []
    for variant, _ in VARIANTS:
        cells = cells_all
        if variant:
            keep = load_reliable(out_dir, variant, "multi-axis")
            if keep is None:
                continue
            cells = cells[cells["task"].isin(set(keep["task"]))]
        for name in TIERS:
            langs = {l for l, t in tier.items() if t == name}
            c = cells[cells["task"].map(lang_of).isin(langs)]
            out = pooled(c, dec, args.pool, axis_of, name)
            if len(out):
                tables.append(out.assign(variant=variant or "all", languages=len(langs)))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(out_dir / "reliability_by_language_tier.csv", index=False)
    figure_tiers(table, out_dir / "reliability_by_language_tier.png", args.pool)
    print(table[table["size"] != TARGET_SIZE].pivot_table(index=["variant", "group"], columns="size", values="reliability")
          .reindex(columns=[s for s in sizes if s != TARGET_SIZE]).round(3).to_string())

    # per language, unfiltered, against the share of the L50 mixture
    share = language_token_share(50, "A") or {}
    rows = []
    for lang in sorted(lang_of.unique()):
        c = cells_all[cells_all["task"].map(lang_of) == lang]
        out = pooled(c, dec, args.pool, axis_of, lang)
        if not len(out):
            continue
        out = out[(out["size"] != TARGET_SIZE) & (out["n_tasks"] >= MIN_LANG_TASKS)]
        if len(out) and lang in share:
            rows.append(out.assign(language=lang, tier=tier.get(lang, "—"), share_L50=share[lang]))
    per_lang = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    per_lang.to_csv(out_dir / "reliability_vs_language_share.csv", index=False)
    if len(per_lang):
        figure_share(per_lang, out_dir / "reliability_vs_language_share.png", args.pool)
        for s, g in per_lang.groupby("size"):
            print(f"{s}: {len(g)} languages, Spearman ρ(share, reliability) = "
                  f"{g['share_L50'].corr(g['reliability'], method='spearman'):.3f}")
    if args.pool == CANONICAL_POOL:
        generate_readme(args.pool, out_dir, table, per_lang)
