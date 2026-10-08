"""rq14: can a benchmark be shortened from the proxies alone, so that the short
benchmark still ranks the design variants like the 1.7B reference on the FULL task?

rq12 selects items with the 1.7B runs themselves (circular, an upper bound) and
rq08's reference_solved with the reference's solved items; here the selection
reads the proxies only. Per task (parents, trained languages: the loader's
frame, rules 2 and 6; pool `predictivity`, seed 1904, final checkpoints, from
the per-item store `build_per_item_store.py`):

  split           the families with a TARGET_SIZE final in two halves, alternating
                  along L, arch, scheme, T (stratified, the split of
                  reference_solved.py and per_item_ladder.heldout_da)
  discrimination  on the selecting half, at SELECT_SIZES (600M, 1B) only: the
                  point-biserial correlation of an item's outcome (the task's own
                  metric, acc or acc_norm) with the run's total task score over
                  that half's runs, pooled within size (item and total centred at
                  each size first, so an item that only separates 600M from 1B
                  does not count as separating designs). An item with no variance
                  there is NaN and never kept; the items with a value are the
                  candidates.
  keep            the top q of the candidates, q in QUANTILES
  DA-size         on the held-out half: each proxy size's mean over the kept items
                  against the 1.7B final of the FULL task (`pair_agreement`), over
                  the held-out half's multi-axis and mono-axis pairs (rule 15,
                  NaN below MIN_PAIRS per half, rule 5; `n_pairs` = the held-out
                  pairs of both halves); the halves swap and the two
                  readings are averaged. Rule 1 at the proxy and at the reference
                  (`passes_gate`, the committed `predictivity` mask): gated rows
                  are kept and blanked.
  baselines       on the same held-out pairs: the full task; RANDOM_DRAWS random
                  subsets of the kept size drawn from the same candidates; and
                  rq12's subset (`select_and_score`: items above chance over every
                  1.7B final), which reads the reference and is the circular upper
                  bound, NaN where the task has no chance level or no such item.

Rule 11: the selection reads proxy runs of the other half alone, so its DA is a
genuine held-out estimate, available before the reference is trained; the
truth it is scored against is still the 1.7B full task.

Outputs under pretraining/<pool>/:
  proxy_item_selection_da_size_per_task_both_axes.csv   per task, q, proxy size, pair set: the DAs, pairs, items
  proxy_item_selection_da_size_both_axes.png/.csv       per pair set, q, proxy size: mean over the tasks, counts
  proxy_item_selection_da_size_multi_axes_paper.png/.svg/.csv      the paper's bare figure (rule 18)
and, on the canonical pool, the README's `proxy-item-selection` block.

    python analysis/rq14_proxy_item_selection/proxy_item_selection.py --pool predictivity [--store-pool predictivity]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools, metric_for  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, fmt, md_table, replace_block  # noqa: E402
from analysis.paths import PROXY_ITEM_SELECTION  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask, task_chance  # noqa: E402
from analysis.rq08_subset_selection.build_per_item_store import BBPB_POOL, STORE, readable_parts, store_gap  # noqa: E402
from analysis.rq08_subset_selection.per_item_ladder import RANDOM_DRAWS  # noqa: E402
from analysis.rq08_subset_selection.reference_solved import finite_mean  # noqa: E402
from analysis.rq12_above_chance_items.above_chance_items import save, select_and_score  # noqa: E402
from analysis.utils import (BBPB, MIN_PAIRS, PAIR_AXES, SMALL_SIZES, TARGET_SIZE, benchmark_family,  # noqa: E402
                            design_axes, finals, ladder_frame, pair_agreement, pair_sets, passes_gate)

NAME = "proxy_item_selection"
SELECT_SIZES = ("600M", "1B")            # the proxies the discrimination is read at; never the reference
QUANTILES = (0.25, 0.5, 0.75)            # share of the candidate items kept
PAPER_Q = 0.5                            # the q the paper's left panel draws
AXES = PAIR_AXES[:2]                     # the design pair sets; the seed null decides nothing
KEYS = ["model", "step", "task"]
DA_COLS = ["da_kept", "da_full", "da_random", "da_reference_selected"]
LINES = {"da_full": ("full task", S.INK, "-"), "da_random": ("random items, same count", S.SERIES[1], "-"),
         "da_kept": ("items chosen on the proxies", S.SERIES[0], "-"),
         "da_reference_selected": (f"items above chance at {TARGET_SIZE} (rq12, sees the truth)", S.MUTED, ":")}
assert TARGET_SIZE not in SELECT_SIZES and set(SELECT_SIZES) <= set(SMALL_SIZES)
mpl.rcParams.update(S.RC)


def split_families(attrs: pd.DataFrame, fams) -> tuple[set, set]:
    """Two halves alternating along the sorted axes (stratified), as reference_solved.py splits."""
    f = attrs.loc[sorted(fams)].sort_values(["L", "arch", "scheme", "T"]).index
    return set(f[::2]), set(f[1::2])


def discrimination(A: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Per item (row of `A`, items x runs): the point-biserial correlation of its
    outcome with the runs' total score (the mean over every item), item and
    total centred within each size of `size` first. NaN without variance."""
    X, t = A.astype(float).copy(), A.mean(0).astype(float)
    for s in np.unique(size):
        at = size == s
        X[:, at] -= X[:, at].mean(1, keepdims=True)
        t[at] -= t[at].mean()
    den = np.sqrt((X ** 2).sum(1) * (t ** 2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        r = X @ t / den
    return np.where(den > 1e-12, r, np.nan)


def select_on(A: np.ndarray, cols: pd.DataFrame, sel: set) -> np.ndarray:
    """The discrimination of every item on the selecting families' runs at
    SELECT_SIZES: the only columns of `A` the selection reads (no reference run,
    no held-out run). `cols` = the runs' family and size, aligned with `A`."""
    use = (cols["size"].isin(SELECT_SIZES) & cols["family"].isin(sel)).to_numpy()
    return discrimination(A[:, use], cols["size"].to_numpy()[use])


def top_items(disc: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """(kept, candidates): the top q of the items with a discrimination, and those items."""
    cand = np.flatnonzero(np.isfinite(disc))
    return cand[np.argsort(-disc[cand], kind="stable")[:int(np.ceil(q * len(cand)))]], cand


def heldout_rows(A: np.ndarray, cols: pd.DataFrame, truth: dict, groups: dict, halves: tuple,
                 ref_keep: np.ndarray, rng) -> list[dict]:
    """Per (pair set, q, proxy size): DA-size of the proxy-selected items, the
    full task, random subsets of the same count from the same candidates and the
    reference-selected items `ref_keep`, each on the held-out half's pairs against
    `truth` (family -> the 1.7B full-task final); both halves averaged."""
    fam, size = cols["family"].to_numpy(), cols["size"].to_numpy()
    acc = {}
    for sel, held in (halves, halves[::-1]):
        disc = select_on(A, cols, sel)
        picks = {q: top_items(disc, q) for q in QUANTILES}
        hps = {a: [(x, y) for x, y in pairs if x in held and y in held] for a, pairs in groups.items()}
        for s in SMALL_SIZES:
            at = (size == s) & np.isin(fam, list(held))           # np.isin reads a set as one object
            if not at.any():
                continue
            P = dict(zip(fam[at], A[:, at].T))
            score = lambda idx: {f: v[idx].mean() for f, v in P.items()}
            full = score(slice(None))
            n = {a: pair_agreement(full, truth, hp)[1] for a, hp in hps.items()}
            live = {a: hp for a, hp in hps.items() if n[a] >= MIN_PAIRS}              # rule 5
            by_ref = score(ref_keep) if live and len(ref_keep) else None
            for q, (kept, cand) in picks.items():
                for a in hps:
                    r = acc.setdefault((a, q, s), {c: [] for c in DA_COLS} | {"n_pairs": 0, "n_kept": [], "n_candidates": []})
                    r["n_pairs"] += n[a]
                    r["n_kept"].append(len(kept)); r["n_candidates"].append(len(cand))
                if not live or not len(kept):
                    continue
                kept_s = score(kept)
                draws = [score(rng.choice(cand, len(kept), replace=False)) for _ in range(RANDOM_DRAWS)]
                for a, hp in live.items():
                    r = acc[(a, q, s)]
                    r["da_full"].append(pair_agreement(full, truth, hp)[0])
                    r["da_kept"].append(pair_agreement(kept_s, truth, hp)[0])
                    r["da_random"].append(np.mean([pair_agreement(d, truth, hp)[0] for d in draws]))
                    r["da_reference_selected"].append(pair_agreement(by_ref, truth, hp)[0] if by_ref else np.nan)
    return [{"axes": a, "q": q, "size": s, "n_pairs": r["n_pairs"], "n_kept": float(np.mean(r["n_kept"])),
             "n_candidates": float(np.mean(r["n_candidates"])), **{c: finite_mean(r[c]) for c in DA_COLS}}
            for (a, q, s), r in acc.items()]


def summarise(out: pd.DataFrame) -> pd.DataFrame:
    """Per pair set, q and proxy size: the mean of each DA over the tasks that
    have it (`n_tasks`; the reference-selected baseline over its own count) and
    the paired gains of the proxy-selected items over random and over the full task."""
    ok = out[~out["gated"]].assign(gain_over_random=lambda x: x["da_kept"] - x["da_random"],
                                   gain_over_full=lambda x: x["da_kept"] - x["da_full"],
                                   kept_share=lambda x: x["n_kept"] / x["n_items"])
    g = ok.groupby(["axes", "q", "size"])
    t = g[DA_COLS + ["gain_over_random", "gain_over_full"]].mean().join(
        g.agg(n_tasks=("da_kept", "count"), n_tasks_reference=("da_reference_selected", "count"),
              kept_share=("kept_share", "median"), n_pairs_median=("n_pairs", "median"))).reset_index()
    return t.sort_values(["axes", "q", "size"], key=lambda c: c.map(SMALL_SIZES.index) if c.name == "size" else c,
                         ignore_index=True)


def task_frame(store: Path, keys: pd.DataFrame):
    """Per task of the store the pool keeps: its items x final-runs matrix, the
    runs' family and size, and the long rows (for rq12's selection)."""
    for part in sorted(store.glob("*.parquet")):
        files = readable_parts(part)                 # a truncated part is skipped with a warning
        if not files:
            continue
        s = pq.read_table(files, columns=["model", "step", "task", "doc_id", "acc", "acc_norm"],
                          filters=[("model", "in", sorted(set(keys["model"]))),
                                   ("step", "in", sorted(set(keys["step"])))]).to_pandas()
        s = s.astype({"model": str, "task": str}).merge(keys, on=KEYS)           # the finals the frame keeps
        for task, g in s.groupby("task"):
            metric = metric_for(task) or "acc"
            M = g.pivot_table(index="doc_id", columns="model", values=metric).dropna()
            if M.empty:
                continue
            yield task, metric, M, keys[keys["task"] == task].set_index("model").loc[M.columns, ["family", "size"]], g


def figure(summ: pd.DataFrame, out_dir: Path, note: str) -> None:
    axes_ = [a for a in AXES if a in set(summ["axes"])]
    fig, axs = plt.subplots(len(axes_), len(QUANTILES), figsize=(4 * len(QUANTILES), 3.4 * len(axes_)),
                            sharey=True, squeeze=False)
    for i, a in enumerate(axes_):
        for j, q in enumerate(QUANTILES):
            ax, t = axs[i][j], summ[(summ["axes"] == a) & (summ["q"] == q)].set_index("size").reindex(SMALL_SIZES)
            for c, (lab, col, ls) in LINES.items():
                ax.plot(range(len(t)), t[c], ls, marker="o", ms=3.5, lw=1.4, color=col, label=lab)
            ax.set_xticks(range(len(t)))
            ax.set_xticklabels([f"{s}\n({int(n)} tasks)" if np.isfinite(n) else s for s, n in zip(t.index, t["n_tasks"])])
            ax.axhline(.5, color=S.MUTED, lw=.8, ls=":"); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
            ax.set_title(f"{a} pairs, top {q:.0%} by proxy discrimination", loc="left", fontsize=8.5)
            ax.set_xlabel("proxy size")
        axs[i][0].set_ylabel(f"DA-size against the {TARGET_SIZE} full task\n(mean over tasks)")
    axs[0][0].legend(fontsize=6.5, frameon=False, loc="lower right")
    save(fig, out_dir, f"{NAME}_da_size_both_axes", "Proxy-only item selection: held-out decision accuracy", note, summ)


def paper_figure(summ: pd.DataFrame, out_dir: Path) -> None:
    """Rule 18: no title, no description, capitalized labels, no dash or ';'.
    Left: multi-axis DA-size at q = PAPER_Q; right: the gain over random per q."""
    m = summ[summ["axes"] == AXES[0]]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(8.5, 3.0))
    t = m[m["q"] == PAPER_Q].set_index("size").reindex(SMALL_SIZES)
    for c, lab in (("da_full", "Full task"), ("da_random", "Random items"), ("da_kept", "Items chosen on the proxies")):
        a0.plot(range(len(t)), t[c], "-o", ms=3.5, lw=1.4, color=LINES[c][1], label=lab)
    a0.set_ylabel(f"DA-size against {TARGET_SIZE}")
    for q, col in zip(QUANTILES, S.SERIES):
        g = m[m["q"] == q].set_index("size").reindex(SMALL_SIZES)
        a1.plot(range(len(g)), g["gain_over_random"], "-o", ms=3.5, lw=1.4, color=col, label=f"Top {q:.0%} of items")
    a1.axhline(0, color=S.MUTED, lw=.8); a1.set_ylabel("Gain over random items")
    for ax in (a0, a1):
        ax.set_xticks(range(len(SMALL_SIZES))); ax.set_xticklabels(SMALL_SIZES); ax.set_xlabel("Proxy size")
        ax.legend(fontsize=6.5, frameon=False); ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    fig.tight_layout()
    m.to_csv(out_dir / f"{NAME}_da_size_multi_axes_paper.csv", index=False)
    S.save_paper(fig, out_dir / f"{NAME}_da_size_multi_axes_paper")


def readme(summ: pd.DataFrame, pool: str, store_pool: str, note: str, missing: list[str]) -> None:
    cell = lambda r: (f"{fmt(r.da_kept, 3)} / {fmt(r.da_full, 3)} / {fmt(r.da_random, 3)} / {fmt(r.da_reference_selected, 3)} "
                      f"({int(r.n_tasks)})" if r.n_tasks > 0 else "")
    rows = [[a, f"{q:.0%}", *[cell(r) for r in g.set_index("size").reindex(SMALL_SIZES).itertuples()]]
            for (a, q), g in summ.groupby(["axes", "q"], sort=False)]
    gains = [[a, f"{q:.0%}", *[f"{v:+.3f}" if np.isfinite(v) else "" for v in g.set_index("size").reindex(SMALL_SIZES)["gain_over_random"]]]
             for (a, q), g in summ.groupby(["axes", "q"], sort=False)]
    body = "\n\n".join([
        "## Results",
        f"Pool `{pool}`, per-item store `{store_pool}`. {note} Regenerate with "
        f"`python analysis/rq14_proxy_item_selection/proxy_item_selection.py --pool {pool} --store-pool {store_pool}`."
        + (f" The store lacks {len(missing)} of the pool's models ({', '.join(missing)}); they are left out until the "
           "store is rebuilt for the pool." if missing else ""),
        "**DA-size on the held-out half**, kept / full task / random / rq12's reference-selected items (tasks), "
        "mean over the tasks; the task count moves along a row (rule 13):",
        md_table(["pairs", "kept", *SMALL_SIZES], rows),
        "**Gain over random items** (mean paired difference over the same tasks):",
        md_table(["pairs", "kept", *SMALL_SIZES], gains),
        f"![Proxy-only item selection](pretraining/{pool}/{NAME}_da_size_both_axes.png)"])
    replace_block(PROXY_ITEM_SELECTION / "README.md", "proxy-item-selection", body, f"proxy_item_selection.py --pool {pool}")


def main(pool: str, store_pool: str) -> None:
    store = STORE / store_pool
    if not any(store.glob("*.parquet")):
        # the store lives on the cluster only: write nothing rather than empty tables
        print(f"no per-item store at {store}: nothing written (build_per_item_store.sbatch builds it)")
        return
    mask = load_mask(pool)
    if mask is None:
        print("no committed above-random mask: nothing written (the gate's pass writes it first)")
        return
    rng = np.random.default_rng(0)
    df = ladder_frame(pool)                                              # rules 2, 6 and 10 applied at load
    fin = finals(df[(df["kind"] == "benchmark") & ~df["task"].str.startswith(BBPB)])
    keys = fin[KEYS + ["size", "family"]].astype({"model": str, "task": str})
    missing = store_gap(keys["model"], store)
    ref_fin = fin[fin["size"] == TARGET_SIZE]
    truth = {t: dict(zip(g["family"], g["primary_score"])) for t, g in ref_fin.groupby("task")}
    ref_runs = {t: pd.MultiIndex.from_arrays([g["model"].astype(str), g["step"].astype(int)]) for t, g in ref_fin.groupby("task")}
    attrs = design_axes(df)
    groups = {a: p for a, p in pair_sets(attrs).items() if a in AXES and p}
    halves = split_families(attrs, set(ref_fin["family"]))
    print(f"{pool}: store {store_pool}, {keys['task'].nunique()} tasks, "
          + ", ".join(f"{len(p)} {a} pairs" for a, p in groups.items())
          + f", halves of {len(halves[0])} / {len(halves[1])} families at {TARGET_SIZE}; selection at {', '.join(SELECT_SIZES)} only")
    print(f"!!! RULE 11: the items are chosen on the proxies of the other half alone (held out); the truth is the "
          f"{TARGET_SIZE} full task. The rq12 baseline selects on every {TARGET_SIZE} final and is the circular upper bound.")
    rows = []
    for task, metric, M, cols, g in task_frame(store, keys):
        if task not in truth:
            continue
        keep, _ = select_and_score(g, metric, ref_runs[task], task_chance(task))      # rq12's subset: reads the reference
        ref_keep = np.flatnonzero(M.index.isin(keep))
        r = heldout_rows(M.to_numpy(dtype=float), cols, truth[task], groups, halves, ref_keep, rng)
        rows += [x | {"task": task, "benchmark": benchmark_family(task), "n_items": len(M),
                      "n_reference_selected": len(ref_keep)} for x in r]
    out = pd.DataFrame(rows)
    if out.empty:
        print(f"!!! no task of the pool has items for its proxies and a {TARGET_SIZE} final in the store: nothing written")
        return
    cells = out[["task", "size"]].drop_duplicates()
    ok = dict(zip(zip(cells["task"], cells["size"]),
                  [bool(passes_gate(mask, [t], s, TARGET_SIZE).iloc[0]) for t, s in zip(cells["task"], cells["size"])]))
    out["gated"] = [not ok[c] for c in zip(out["task"], out["size"])]
    out.loc[out["gated"], DA_COLS] = np.nan                               # rule 1: at chance, kept and blanked
    few = out[(out["n_pairs"] < MIN_PAIRS) & ~out["gated"]]
    if len(few):
        print(f"!!! RULE 5: {len(few)} (task, q, size, pair set) cells over {few['task'].nunique()} tasks had fewer than "
              f"{MIN_PAIRS} held-out pairs and are NaN (pair counts in the per-task table)")
    out_dir = PROXY_ITEM_SELECTION / load_pools()[pool].get("stage", "pretraining") / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    lead = ["task", "benchmark", "axes", "q", "size", "n_items", "n_candidates", "n_kept", "n_reference_selected", "n_pairs"]
    out[lead + DA_COLS + ["gated"]].to_csv(out_dir / f"{NAME}_da_size_per_task_both_axes.csv", index=False)
    summ = summarise(out)
    print(summ.round(3).to_string(index=False))
    note = (f"DA-size, finals, pool {pool} (seed 1904), held-out half's pairs (>= {MIN_PAIRS} per half), both halves "
            f"averaged; items ranked by point-biserial discrimination (within size) on the other half's "
            f"{' and '.join(SELECT_SIZES)} runs, the top q kept; scored against the {TARGET_SIZE} final of the full task. "
            f"Random = {RANDOM_DRAWS} draws of the kept count from the items with a discrimination; dotted = rq12's items "
            f"above chance over every {TARGET_SIZE} final (reads the reference, upper bound). Gate {CANONICAL_POOL} at the "
            "proxy and the reference (rule 1); the task count moves across sizes (rule 13).")
    figure(summ, out_dir, note)
    paper_figure(summ, out_dir)
    if pool == CANONICAL_POOL:
        readme(summ, pool, store_pool, note, missing)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", default=CANONICAL_POOL)
    ap.add_argument("--store-pool", default=BBPB_POOL, help="the per_item_store/<store> folder to read")
    a = ap.parse_args()
    main(a.pool, a.store_pool)
