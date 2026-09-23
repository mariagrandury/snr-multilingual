"""What seed noise does to decision accuracy, from the replicate seeds the grid
has — three views, each named for what it measures, none for what it cannot.

The grid replicates a design at three seeds in ten cells only (deep, scheme A
at L ∈ {1, 2, 50} for 175M and 600M with seeds 64/313/1904; deep L1, L2, L30-A
and L30-B at 1B with seeds 28/1797/1904) and never at the 1.7B reference. Rule
2 then decides which languages a cross-L pair can be read on: an L1 model is
scored on English alone, so the {L1, L2, L50} decisions exist only for English,
and at 1B the {L2, L30-A, L30-B} triple adds Russian. Everything below is
therefore English at three proxy sizes and Russian at 1B, and is drawn as such.

  proxy seed    DA of the replicated designs' decisions against the fixed 1.7B
                (seed 1904) final, with the proxy read at each of its three
                seeds: three values per (size, language), drawn as three markers
                — with three pairs a DA lives on {0, ⅓, ⅔, 1}, so a standard
                deviation over them would be a fiction. What it measures: how
                much the proxy's own seed moves a decision, on cross-L deep
                scheme-A pairs — a population unlike any published line's, so it
                is illustrative of scale, not an interval for them.
  test-retest   the same designs ranked from their seed-s1 runs against their
                seed-s2 runs at ONE size: the agreement any proxy could reach if
                the reference itself were redrawn — the ceiling the proxy-seed
                view cannot see, the reference-side noise.
  seed null     DA over pairs of two seeds of ONE design (`axes == "seed"` in
                rq02's tables), where there is nothing to decide: what a
                benchmark reads with no signal. Exists for DA-ckpt only (never
                against the reference, which has no replicate); 0.5 is the
                no-autocorrelation value, and a null above it says a run's early
                noise persists to its final and lifts DA-ckpt for real pairs
                too. Drawn against the real DA-ckpt of the same cells.

    seed_uncertainty.png / .csv     the three panels and their long table

    python analysis/rq02_decision_accuracy/seed_uncertainty.py --pool predictivity
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
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.utils import (  # noqa: E402
    CKPT_DA_EARLY_FRACS, GRID_SEED, MIN_PAIRS, NON_EMB, SMALL_SIZES, TARGET_SIZE, assign_language,
    design_axes, finals, ladder_frame, passes_gate, size_order)

OUT_ROOT = DECISION_ACCURACY
POOL = "predictivity_all"          # every seed: the replicates live here
SEED_POOL = "predictivity_seeds"   # where compute_da writes the `seed` pair set
LANGS = ("en", "ru")               # the languages a replicated triple can be read on (rule 2)
mpl.rcParams.update(S.RC)


def pooled_da(proxy: pd.DataFrame, ref: pd.DataFrame, designs: list, tasks: set) -> tuple[float, int, int]:
    """(DA pooled over tasks, decisions, tasks): every design pair, per task the
    pairs both frames score, a task counting only with ≥ MIN_PAIRS of them.
    `proxy`/`ref` are (task, design) -> score."""
    match = total = n_tasks = 0
    for task in tasks:
        p = proxy.get(task, {}); r = ref.get(task, {})
        pl = [(a, b) for a, b in combinations(designs, 2) if a in p and b in p and a in r and b in r]
        if len(pl) < MIN_PAIRS:                                      # rule 5
            continue
        match += sum(np.sign(p[a] - p[b]) == np.sign(r[a] - r[b]) for a, b in pl)
        total += len(pl); n_tasks += 1
    return (match / total if total else float("nan")), total, n_tasks


def scores(fin: pd.DataFrame, size: str, seed: int) -> dict:
    """task -> {design -> final score} at one (size, seed)."""
    g = fin[(fin["size"] == size) & (fin["seed"] == seed)]
    out: dict = {}
    for task, design, v in zip(g["task"], g["design"], g["primary_score"]):
        out.setdefault(task, {})[design] = v
    return out


def seed_views(fin: pd.DataFrame, pool: str) -> pd.DataFrame:
    """The proxy-seed and test-retest rows, per size and language."""
    mask = load_mask(pool)
    fin = fin.copy()
    fin["language"] = fin["task"].map(assign_language)
    ref = scores(fin, TARGET_SIZE, GRID_SEED)
    rows = []
    for size in SMALL_SIZES:
        at = fin[fin["size"] == size]
        by_design = at.groupby("design")["seed"].nunique()
        designs = sorted(by_design[by_design >= 3].index)
        if len(designs) < 3:                                         # three designs = MIN_PAIRS pairs
            continue
        seeds = sorted(set.intersection(*[set(at.loc[at["design"] == d, "seed"]) for d in designs]))
        for lang in LANGS:
            tasks = set(at.loc[(at["language"] == lang) & ~at["task"].str.startswith("bpb_"), "task"])
            # rule 1: above chance at the proxy and at the reference
            tasks = set(pd.Index(sorted(tasks))[passes_gate(mask, sorted(tasks), size, TARGET_SIZE).to_numpy()]) if tasks else set()
            # one task set for every marker of a cell: a task counts only when
            # EVERY seed's proxy scores >= MIN_PAIRS of its pairs against the
            # reference, so three markers differ by the seed alone (rule 13)
            tasks = set.intersection(*[
                {t for t in tasks if pooled_da(scores(fin, size, s), ref, designs, {t})[1] > 0} for s in seeds]) if tasks else set()
            for s in seeds:
                da, n, nt = pooled_da(scores(fin, size, s), ref, designs, tasks)
                rows.append({"view": "proxy seed", "size": size, "language": lang, "seed": s, "seed_b": np.nan,
                             "da": da, "decisions": n, "tasks": nt, "designs": len(designs)})
            for s1, s2 in combinations(seeds, 2):
                da, n, nt = pooled_da(scores(fin, size, s1), scores(fin, size, s2), designs, tasks)
                rows.append({"view": "test-retest", "size": size, "language": lang, "seed": s1, "seed_b": s2,
                             "da": da, "decisions": n, "tasks": nt, "designs": len(designs)})
    return pd.DataFrame(rows)


def null_view(out_dir_seed: Path, pool: str) -> pd.DataFrame:
    """The seed null against the real DA-ckpt, per (size, fraction), mean over
    the gated benchmark tasks that carry both."""
    da = pd.read_csv(out_dir_seed / "da_per_task.csv")
    if "axes" not in da.columns or "seed" not in set(da["axes"]):
        return pd.DataFrame()
    mask = load_mask(pool)
    rows = []
    for size in SMALL_SIZES:
        for f in CKPT_DA_EARLY_FRACS:
            col = f"decision_acc_ckpt_f{int(round(f * 100))}_{size}"
            if col not in da.columns:
                continue
            wide = da.pivot_table(index="task", columns="axes", values=col)
            if "seed" not in wide or "multi-axis" not in wide:
                continue
            both = wide.dropna(subset=["seed", "multi-axis"])
            both = both[~both.index.str.startswith("bpb_") & (both.index != "train_loss")]
            both = both[passes_gate(mask, both.index, size).to_numpy()]   # rule 1 at the proxy: DA-ckpt's whole gate
            if len(both) < MIN_PAIRS:
                continue
            rows.append({"view": "seed null", "size": size, "frac": f, "da": both["seed"].mean(),
                         "da_real": both["multi-axis"].mean(), "tasks": len(both)})
    return pd.DataFrame(rows)


def figure(sv: pd.DataFrame, nv: pd.DataFrame, path: Path, pool: str) -> None:
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15.5, 4.6))
    sizes = size_order(sv["size"].unique()) if len(sv) else []
    for lang, mk in zip(LANGS, ("o", "s")):
        for view, ax in (("proxy seed", a), ("test-retest", b)):
            g = sv[(sv["view"] == view) & (sv["language"] == lang)].dropna(subset=["da"])
            if not len(g):
                continue
            grid = g[g["seed"] == GRID_SEED] if view == "proxy seed" else g.iloc[0:0]
            nd = sorted(set(g["designs"]))
            ax.scatter(g["size"].map(NON_EMB), g["da"], s=26, marker=mk, facecolor="none",
                       edgecolor=S.SERIES[0] if lang == "en" else S.SERIES[1], lw=1.2, zorder=3,
                       label=f"{lang}: {nd[0] if len(nd) == 1 else f'{nd[0]}–{nd[-1]}'} designs, "
                             f"{int(g['decisions'].max())} decisions at most")
            if len(grid):
                ax.scatter(grid["size"].map(NON_EMB), grid["da"], s=26, marker=mk,
                           color=S.SERIES[0] if lang == "en" else S.SERIES[1], zorder=4)
    for ax, ttl in ((a, f"proxy seed: vs the {TARGET_SIZE} final, proxy at each of its 3 seeds (filled = seed {GRID_SEED})"),
                    (b, "test-retest: the same designs ranked from seed s1 vs seed s2, one size")):
        ax.axhline(.5, color=S.MUTED, lw=.8, ls=":"); ax.set_ylim(0, 1.0); ax.set_xscale("log")
        ax.set_xticks([NON_EMB[s] for s in sizes]); ax.set_xticklabels(sizes)
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())      # three sizes on a log axis: no 2×10⁸ labels
        ax.set_xlim(NON_EMB["175M"] / 1.6, NON_EMB["1B"] * 1.6)
        ax.set_xlabel("non-embedding parameters (log)"); ax.set_title(ttl, loc="left", fontsize=7.5)
        ax.legend(fontsize=6, frameon=False, loc="lower right"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    a.set_ylabel(f"DA vs {TARGET_SIZE} final, pooled over tasks"); b.set_ylabel("DA between the two seeds' rankings")
    for s in size_order(nv["size"].unique()) if len(nv) else []:
        g = nv[nv["size"] == s].sort_values("frac")
        col = S.SIZE_COLOR.get(s, S.MUTED)
        c.plot(g["frac"] * G.CHINCHILLA_AT_FULL, g["da"], color=col, ls="--", marker="o", ms=3, lw=1.2, label=f"{s} seed null")
        c.plot(g["frac"] * G.CHINCHILLA_AT_FULL, g["da_real"], color=col, ls="-", marker="o", ms=3, lw=1.2, label=f"{s} real pairs")
    c.axhline(.5, color=S.MUTED, lw=.8, ls=":"); c.set_ylim(0, 1.0)
    c.set_xticks([1, 2, 3, 4, 5]); c.set_xticklabels([G.chinchilla(f) for f in (.2, .4, .6, .8, 1.0)]); c.set_xlim(0.3, 5.2)
    c.set_xlabel("proxy's training tokens (× Chinchilla)"); c.set_ylabel("mean DA-ckpt over the gated tasks")
    c.set_title("seed null: DA-ckpt over two seeds of ONE design (dashed) vs the real pairs (solid)", loc="left", fontsize=7.5)
    c.legend(fontsize=5.5, frameon=False, ncol=2, loc="lower right"); c.grid(color=S.GRID, lw=.6); S.clean(c)
    top = G._header(fig, "Seed uncertainty in decision accuracy: what the replicate seeds can and cannot say",
                    f"Left and middle: the ten replicated cells (deep scheme A at L ∈ {{1, 2, 50}} for 175M/600M, seeds 64/313/"
                    f"{GRID_SEED}; deep L1, L2, L30-A, L30-B at 1B, seeds 28/1797/{GRID_SEED}); rule 2 leaves their cross-L "
                    f"decisions readable on English (all three sizes) and Russian (1B) alone, on the gated tasks with "
                    f"≥ {MIN_PAIRS} pairs. Three pairs put a DA on {{0, ⅓, ⅔, 1}}, so the three seeds are three markers, "
                    f"not a bar. Left measures proxy-side seed noise only (no replicate at {TARGET_SIZE}); middle is the "
                    f"reference-side ceiling the left cannot see; together they bracket it — for English. Right: the null, "
                    f"0.5 without within-run autocorrelation. None of this transfers to other languages or to the pooled "
                    f"lines, whose population is different; it is the scale of seed noise, not an interval. Gate: {pool}.")
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, out_dir: Path, sv: pd.DataFrame, nv: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rows = []
    seen = sv[(sv["view"] == "proxy seed")].dropna(subset=["da"])     # a cell without a marker has no row
    for (size, lang), g in seen.groupby(["size", "language"], sort=False):
        tr = sv[(sv["view"] == "test-retest") & (sv["size"] == size) & (sv["language"] == lang)]["da"]
        vals = ", ".join(f"{v:.2f}" for v in g.sort_values("seed")["da"])
        rows.append([size, lang, int(g["designs"].iloc[0]), vals,
                     f"{tr.min():.2f}–{tr.max():.2f}" if len(tr.dropna()) else "—", int(g["tasks"].max())])
    null = ("; ".join(f"{s}: {nv[nv['size'] == s]['da'].mean():.2f} vs real {nv[nv['size'] == s]['da_real'].mean():.2f}"
                      for s in size_order(nv["size"].unique())) if len(nv) else "not available")
    body = "\n\n".join([
        "## Seed uncertainty",
        f"What the replicate seeds say about decision accuracy — English at three proxy sizes and Russian at 1B, the only "
        f"(size, language) cells where three replicated designs give ≥ {MIN_PAIRS} cross-L pairs under rule 2. Per row: DA "
        f"of those decisions against the {TARGET_SIZE} final with the proxy at each of its three seeds (proxy-side noise), "
        f"and the DA between two seeds' rankings of the same designs at that size (the reference-side ceiling). Three "
        f"pairs put a DA on {{0, ⅓, ⅔, 1}}: read the spread, not a mean. The seed null (DA-ckpt over pairs of two seeds "
        f"of one design, mean over fractions and gated tasks) is {null}. Regenerate with "
        f"`python analysis/rq02_decision_accuracy/seed_uncertainty.py --pool {pool}`.",
        md_table(["size", "language", "designs", "DA at the 3 proxy seeds", "test-retest (3 seed pairs)", "tasks"], rows),
        f"![Seed uncertainty]({stage}/{pool}/seed_uncertainty.png)"])
    replace_block(OUT_ROOT / "README.md", "seed-uncertainty", body, f"seed_uncertainty.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose gate applies and whose folder receives the outputs")
    args = p.parse_args()
    stage = load_pools()[args.pool].get("stage", "pretraining")
    out_dir = OUT_ROOT / stage / args.pool
    fin = finals(ladder_frame(POOL))
    attrs = design_axes(fin)
    fin["design"] = fin["family"].map(lambda f: f"L{int(attrs.loc[f, 'L'])}-{attrs.loc[f, 'arch']}-{attrs.loc[f, 'scheme']}")
    sv = seed_views(fin, args.pool)
    nv = null_view(OUT_ROOT / stage / SEED_POOL, args.pool)
    pd.concat([sv, nv], ignore_index=True).to_csv(out_dir / "seed_uncertainty.csv", index=False)
    print(sv.to_string(index=False))
    if len(nv):
        print(nv.groupby("size")[["da", "da_real"]].mean().round(3).to_string())
    figure(sv, nv, out_dir / "seed_uncertainty.png", args.pool)
    generate_readme(args.pool, out_dir, sv, nv)
