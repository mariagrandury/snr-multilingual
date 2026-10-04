"""Early and small, as a ranking — how early in a run, and how small a proxy,
still ranks the design variants like the reference at its final checkpoint?

Reads the long table ``compute_da.py`` writes (`da_early_small_per_task.csv`:
per task, proxy size and fraction of the proxy's own run, the decision
accuracy against the reference's final ranking, with the pair count and the
training compute) and the rq00 gate. A fraction of the run is shown as a
multiple of Chinchilla: every run trains 5C, so 20 % = 1C and 100 % = 5C.
Every figure has its table next to it under the same name.

    highlights.png / .csv                  rq02 on one page
    early_small.csv                        the gated cells behind the three early_small figures
    early_small.png                        BPB and all benchmarks: proxy size x Chinchilla multiple
    early_small_by_benchmark.png           the same grid, one subplot per benchmark (BPB first)
    early_small_by_language.png            one subplot per language
    early_small_summary.csv                the means early_small.png prints
    da_size.csv, da_size_by_benchmark.png  DA-size, language x proxy size, one subplot per benchmark
    da_size_by_language.png                DA-size, benchmark x proxy size, one subplot per language
    safe_size.png / .csv                   language x benchmark: smallest size that safely predicts the reference ranking
    safe_checkpoint.png / .csv             the same per size: fewest training tokens that predict its own final ranking
    safe_flops.png / .csv                  language x benchmark: fewest FLOPs that predict the reference ranking

In a grid white is "no value" and grey is "filtered out by the gate".
"Safely" = decision accuracy >= SAFE_DA over >= MIN_PAIRS model pairs, at that
level and at every larger level with information (it must not come undone
further up). For FLOPs the levels are the (size, fraction) cells ordered by
training compute: the cheapest cell from which every costlier cell clears it
too. A benchmark task counts only where it clears chance at the proxy size
AND at the reference: a ranking against a reference at chance is not a truth.

    python analysis/rq02_decision_accuracy/early_small.py --pool predictivity
"""

from __future__ import annotations

import argparse
import re
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

from evals.scripts.utils.configs import bucket_order, load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, fmt, md_table, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.utils import (  # noqa: E402
    MIN_PAIRS, SMALL_SIZES, TARGET_SIZE, assign_language, benchmark_family, one_axes)

OUT_ROOT = DECISION_ACCURACY
SAFE_DA = 0.75          # the agreement rq05 also calls "reads like the reference"
FLOP_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]   # share of the reference's training compute
mpl.rcParams.update(S.RC)


def _meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["family"] = df["task"].map(benchmark_family)
    df["language"] = df["task"].map(assign_language)
    return df[~df["language"].isin(["??", "multi"]) & (df["family"] != "loss")]


def load_cells(pool_dir: Path, pool: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(early-small cells, DA-size cells, DA-ckpt cells) with pair counts and
    family/language. A benchmark row at chance at its proxy size, or at the
    reference it is ranked against, keeps its place with `gated` true and no
    value (the grey cells); DA-ckpt is gated at its own size only."""
    # `one_axes`: these tables carry one row per (task, pair set) since rule 15;
    # the default keeps the multi-axis reading this figure has always shown.
    early = G.mark_gated(_meta(one_axes(pd.read_csv(pool_dir / "da_early_small_per_task.csv"))),
                         pool, "proxy_size", "da", TARGET_SIZE)
    da = one_axes(pd.read_csv(pool_dir / "da_per_task.csv")).set_index("task")
    npairs = one_axes(pd.read_csv(pool_dir / "da_n_pairs_per_task.csv")).set_index("task")
    size_rows, ckpt_rows = [], []
    for c in da.columns:
        if (m := re.fullmatch(r"decision_acc_size_([^_]+)", c)):
            size_rows.append(pd.DataFrame({"task": da.index, "proxy_size": m.group(1), "da": da[c].to_numpy(),
                                           "n_pairs": npairs[c].to_numpy()}))
        elif (m := re.fullmatch(r"decision_acc_ckpt_f(\d+)_(.+)", c)):
            ckpt_rows.append(pd.DataFrame({"task": da.index, "frac": int(m.group(1)) / 100, "size": m.group(2),
                                           "da": da[c].to_numpy(), "n_pairs": npairs[c].to_numpy()}))
    size = G.mark_gated(_meta(pd.concat(size_rows).dropna(subset=["da"])), pool, "proxy_size", "da", TARGET_SIZE)
    ckpt = G.mark_gated(_meta(pd.concat(ckpt_rows).dropna(subset=["da"])), pool, "size", "da")
    return early, size, ckpt


def _cell_table(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Mean DA per (family, language, level) over the cell's tasks that rest
    on >= MIN_PAIRS pairs (the others are left out of the mean)."""
    df = df[df["n_pairs"] >= MIN_PAIRS]
    return df.groupby(["family", "language", level])["da"].mean().reset_index()


def _gated_pairs(df: pd.DataFrame) -> set:
    return set(zip(df.loc[df["gated"], "family"], df.loc[df["gated"], "language"]))


def safe_level_matrix(df: pd.DataFrame, level: str, levels: list) -> pd.DataFrame:
    """benchmark x language: index in `levels` of the smallest safe level
    (G.GATED where the gate left the pair no level at all)."""
    t = _cell_table(df, level)
    if t.empty:
        return G.with_gated(pd.DataFrame(), _gated_pairs(df))
    wide = t.pivot_table(index=["family", "language"], columns=level, values="da").reindex(columns=levels)
    ok = wide.ge(SAFE_DA).where(wide.notna())
    m = G.smallest_safe(ok).rename("level").rename_axis(["family", "language"]).reset_index() \
         .pivot(index="family", columns="language", values="level")
    return G.with_gated(m, _gated_pairs(df))


def safe_flops_matrix(early: pd.DataFrame) -> pd.DataFrame:
    """benchmark x language: index in FLOP_LEVELS of the cheapest (size,
    fraction) cell from which every costlier cell also clears SAFE_DA — the
    same rule as the other two maps, along the compute axis."""
    e = early[(early["n_pairs"] >= MIN_PAIRS) & early["da"].notna()].copy()
    e["share"] = e["compute"] / e["ref_compute"]
    c = e.groupby(["family", "language", "proxy_size", "frac"]).agg(da=("da", "mean"), share=("share", "mean")).reset_index()
    out = {}
    for (fam, lang), g in c.groupby(["family", "language"]):
        g = g.sort_values("share")
        ok = pd.DataFrame([(g["da"] >= SAFE_DA).to_numpy()])
        i = G.smallest_safe(ok).iloc[0]
        out[(fam, lang)] = -1.0 if i < 0 else float(np.searchsorted(FLOP_LEVELS, min(g["share"].iloc[int(i)], 1.0) - 1e-12))
    s = pd.Series(out, dtype=float).rename_axis(["family", "language"]).rename("level").reset_index()
    m = s.pivot(index="family", columns="language", values="level") if len(s) else pd.DataFrame()
    return G.with_gated(m, _gated_pairs(early))


def figures(pool: str, out_dir: Path) -> dict | None:
    early, size, ckpt = load_cells(out_dir, pool)
    if early.empty:                 # a pool without the reference size (the seed-holdout splits)
        print(f"{pool}: no {TARGET_SIZE} cells, nothing to rank against")
        return None
    sizes = SMALL_SIZES + [TARGET_SIZE]          # every rung, the ones with no information stay blank
    fracs = sorted(early["frac"].unique())
    early["chinchilla"] = early["frac"] * G.CHINCHILLA_AT_FULL
    rule = (f"safe = mean DA ≥ {SAFE_DA} over the pair's tasks with ≥ {MIN_PAIRS} model pairs, and the same at every "
            "larger level that has a value")
    da_note = ("cell = decision accuracy: share of pairs of design variants (every pair of the cells evaluated on the task) the proxy orders like the "
               f"{TARGET_SIZE} final checkpoint, mean over the subplot's tasks")
    kw = dict(row="proxy_size", col="frac", value="da", row_order=sizes, col_order=fracs, col_label=G.chinchilla,
              cbar=f"decision accuracy vs {TARGET_SIZE} final", xlabel="proxy's training tokens (C = Chinchilla-optimal, 20 tokens per parameter)",
              ylabel="proxy size", note=da_note + f"; 5C is the proxy's final checkpoint (that column is DA-size, the {TARGET_SIZE} row its DA-ckpt)")
    # aggregate: BPB and all benchmarks, the two-panel short version (its numbers: early_small_summary.csv)
    agg = early.assign(group=np.where(early["family"] == "bpb", "bpb", "all benchmarks"))
    G.panel_grid(agg, out_dir / "early_small.png", by="group", ncols=2, first=("bpb",),
                 title=f"Early and small: ranking agreement with the {TARGET_SIZE} final checkpoint", **kw)
    G.panel_grid(early, out_dir / "early_small_by_benchmark.png", by="family", csv=False,
                 title=f"Early and small, per benchmark (vs {TARGET_SIZE} final)", **kw)
    G.panel_grid(early, out_dir / "early_small_by_language.png", by="language", ncols=6, csv=False,
                 title=f"Early and small, per language (vs {TARGET_SIZE} final)", **kw)
    proxies = list(SMALL_SIZES)
    skw = dict(value="da", cbar=f"DA-size (proxy → {TARGET_SIZE})", note=da_note + "; both at their final checkpoint")
    G.panel_grid(size, out_dir / "da_size_by_benchmark.png", by="family", row="proxy_size", row_order=proxies,
                 col="language", ncols=1, cell_w=0.3, counts=False, xlabel="language", ylabel="proxy size",
                 title=f"DA-size per benchmark: does the proxy rank the variants like {TARGET_SIZE}?", **skw)
    G.panel_grid(size, out_dir / "da_size_by_language.png", by="language", row="family", ylabel="benchmark", ncols=6,
                 col="proxy_size", col_order=proxies, xlabel="proxy size", csv=False,
                 title=f"DA-size per language: does the proxy rank the variants like {TARGET_SIZE}?", **skw)

    safe_size = safe_level_matrix(size, "proxy_size", proxies)
    G.level_heatmap(safe_size, out_dir / "safe_size.png", levels=proxies, cbar="smallest safe proxy",
                    title=f"Smallest size whose final ranking safely predicts the {TARGET_SIZE} ranking",
                    note=f"cell = smallest proxy size whose DA-size against {TARGET_SIZE} is safe; {rule}")
    ck_fracs = sorted(ckpt["frac"].unique())
    safe_ckpt = {b: safe_level_matrix(ckpt[ckpt["size"] == b], "frac", ck_fracs)
                 for b in bucket_order() if b in set(ckpt["size"])}
    G.level_heatmap(safe_ckpt, out_dir / "safe_checkpoint.png", levels=ck_fracs, level_label=G.chinchilla,
                    cbar="smallest safe training length",
                    title="Smallest checkpoint that safely predicts the size's own final ranking, one map per size",
                    note="cell = fewest training tokens (in Chinchilla multiples, 5C = the full run) at which the size ranks the "
                         f"design variants like its own final checkpoint; {rule}")
    safe_flops = safe_flops_matrix(early)
    G.level_heatmap(safe_flops, out_dir / "safe_flops.png", levels=FLOP_LEVELS, level_label=lambda v: f"≤{v:.0%}",
                    cbar=f"share of the {TARGET_SIZE} run's FLOPs",
                    title=f"Fewest FLOPs that safely predict the {TARGET_SIZE} final ranking",
                    note=f"cell = compute of the cheapest (proxy size, checkpoint), as a share of the {TARGET_SIZE} run, from which "
                         f"every costlier (size, checkpoint) is also safe — the {TARGET_SIZE} run's own 1C–4C checkpoints count; {rule}")

    summary = (agg.dropna(subset=["da"]).groupby(["group", "proxy_size", "frac"])
               .agg(da=("da", "mean"), tasks=("task", "nunique"), median_pairs=("n_pairs", "median")).reset_index())
    summary["chinchilla"] = summary["frac"] * G.CHINCHILLA_AT_FULL
    summary.to_csv(out_dir / "early_small_summary.csv", index=False)
    highlights(out_dir, summary, size, safe_size, safe_flops, sizes, proxies)
    return {"summary": summary, "sizes": sizes, "fracs": fracs, "safe_size": safe_size, "proxies": proxies}


def highlights(out_dir: Path, summary, size, safe_size, safe_flops, sizes, proxies) -> None:
    """rq02 on one page: how early and how small on average, which benchmarks
    a small proxy ranks like the reference, and how the safe size and the
    safe compute distribute over each benchmark's languages."""
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2), gridspec_kw={"width_ratios": [1.1, 0.9, 1.2, 1.2]})
    tables = []
    ax = axes[0]
    for group, ls in (("bpb", "-"), ("all benchmarks", "--")):
        for s_ in sizes:
            g = summary[(summary["group"] == group) & (summary["proxy_size"] == s_)].sort_values("chinchilla")
            if len(g):
                ax.plot(g["chinchilla"], g["da"], color=S.SIZE_COLOR.get(s_, S.MUTED), ls=ls, marker="o", ms=3, lw=1.3)
        tables.append(summary[summary["group"] == group].rename(columns={"proxy_size": "row", "chinchilla": "col", "da": "value"})
                      .assign(panel=f"early and small: {group}")[["panel", "row", "col", "value"]])
    ax.axhline(SAFE_DA, color=S.MUTED, lw=.8, ls=":")
    ax.set_xlabel("proxy's training tokens (× Chinchilla)"); ax.set_ylabel(f"mean DA vs {TARGET_SIZE} final")
    ax.set_title("How early, how small (solid BPB, dashed benchmarks)", loc="left", fontsize=8.5)
    ax.legend(handles=[plt.Line2D([], [], color=S.SIZE_COLOR[s_], lw=2, label=s_) for s_ in sizes if s_ in S.SIZE_COLOR],
              fontsize=6.5, frameon=False, ncol=2); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    fam = size[size["n_pairs"] >= MIN_PAIRS].groupby(["family", "proxy_size"])["da"].mean().unstack().reindex(columns=proxies)
    fam = fam.reindex(G.panel_order(fam.index))
    tables.append(G.matrix_ax(axes[1], fam, f"DA-size → {TARGET_SIZE}, mean over languages", xlabel="proxy size"))
    tables.append(G.stack_ax(axes[2], safe_size, "Smallest safe size, share of each benchmark's languages", levels=proxies))
    tables.append(G.stack_ax(axes[3], safe_flops, f"Fewest safe FLOPs (share of the {TARGET_SIZE} run)", levels=FLOP_LEVELS,
                             level_label=lambda v: f"≤{v:.0%}"))
    G.save_highlights(fig, out_dir, f"rq02 in one figure: how early and how small can the {TARGET_SIZE} ranking be read?",
                      f"DA = share of design-variant pairs ordered like the {TARGET_SIZE} final checkpoint; safe = DA ≥ {SAFE_DA} "
                      f"over ≥ {MIN_PAIRS} pairs, held at every larger level; a benchmark's bar counts its (language) cells, number in brackets",
                      tables)


def generate_readme(pool: str, r: dict | None) -> None:
    if pool != CANONICAL_POOL or r is None:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    blocks, bullets = [], []
    for group in ("bpb", "all benchmarks"):
        g = r["summary"][r["summary"]["group"] == group]
        if g.empty:
            continue
        piv = g.pivot(index="proxy_size", columns="frac", values="da").reindex(index=r["sizes"], columns=r["fracs"])
        cheapest = g[g["da"] >= SAFE_DA]
        blocks += [f"**{group}** (rows: proxy size; columns: the proxy's training tokens in Chinchilla multiples; mean DA over "
                   f"{int(g['tasks'].max())} tasks):",
                   md_table(["proxy"] + [G.chinchilla(f) for f in piv.columns],
                            [[s] + [fmt(piv.loc[s, f]) for f in piv.columns] for s in piv.index])]
        if not cheapest.empty:
            first = min(cheapest.itertuples(), key=lambda x: (r["sizes"].index(x.proxy_size), x.frac))
            bullets.append(f"- **{group}** — smallest proxy whose mean agreement with the {TARGET_SIZE} final ranking "
                           f"reaches {SAFE_DA}: **{first.proxy_size} at {G.chinchilla(first.frac)}** ({fmt(first.da)}).")
        else:
            bullets.append(f"- **{group}** — no (proxy, checkpoint) reaches a mean agreement of {SAFE_DA}.")
    ss = r["safe_size"]
    known = ss.stack().dropna()
    if len(known):
        known = known[known > G.GATED]
        dist = known.map(lambda i: "never" if i < 0 else r["proxies"][int(i)]).value_counts()
        bullets.append("- **Smallest safe size per (benchmark, language)** — "
                       + ", ".join(f"{k}: {v}" for k, v in dist.items()) + f" of {len(known)} cells.")
    body = "\n\n".join([
        "## Early and small, as a ranking",
        f"Numbers from the `{pool}` pool: every design variant at a proxy size, read at 1C–5C of training (C = the "
        "Chinchilla-optimal 20 tokens per parameter; every run trains 5C, so 1C is 20 % of it), "
        f"ranked against the same variants at the {TARGET_SIZE} final checkpoint (the 5C column is DA-size, the "
        f"{TARGET_SIZE} row is that size's DA-ckpt). A benchmark task counts only where it clears chance at the proxy "
        f"size and at {TARGET_SIZE}. "
        f"Regenerate with `python analysis/rq02_decision_accuracy/early_small.py --pool {pool}`.",
        "\n".join(bullets),
        f"![rq02 in one figure]({rel}/highlights.png)"] + blocks + [
        f"![Early and small]({rel}/early_small.png)",
        f"![Early and small per benchmark]({rel}/early_small_by_benchmark.png)",
        f"![Early and small per language]({rel}/early_small_by_language.png)",
        f"**Smallest safe level per language and benchmark** (DA ≥ {SAFE_DA} over ≥ {MIN_PAIRS} pairs, held at "
        "every larger level — for FLOPs, at every costlier (size, checkpoint) cell; red = never, grey = filtered "
        "out by the above-random gate, white = no value). Each figure's table sits next to it under the same name:",
        f"![Smallest safe size]({rel}/safe_size.png)",
        f"![Smallest safe checkpoint]({rel}/safe_checkpoint.png)",
        f"![Smallest safe FLOPs]({rel}/safe_flops.png)",
        f"![DA-size per benchmark]({rel}/da_size_by_benchmark.png)",
        f"![DA-size per language]({rel}/da_size_by_language.png)"])
    readme = OUT_ROOT / "README.md"
    replace_block(readme, "early-small", body, f"early_small.py --pool {pool}")
    print(f"Wrote auto README block → {readme}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools())}")
    out = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    generate_readme(args.pool, figures(args.pool, out))
