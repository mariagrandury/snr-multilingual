"""Decision accuracy per language count — the early-and-small grid read one
L at a time: pairs of design variants that share the L (deep vs shallow,
scheme A vs B, AT3, the second language), at the grid seed of every scheme
(`predictivity_all`, seed 1904). L100 has one pair (AT3 deep vs shallow) once
its 1.7B cells are final; until then its cells stay blank.

    da_by_L_per_task.csv   per task, L, proxy size and fraction of the proxy's own run, on
                           the ten checkpoints of the shared k/10 grid (0.5C ... 5C): `da_ref`
                           vs the reference's final ranking (the early-and-small reading,
                           `n_pairs_ref` pairs) and `da_own` vs the proxy size's own final
                           ranking (DA-ckpt within the L, `n_pairs_own`); `compute` = the training FLOPs the
                           pair's proxies spent up to that checkpoint (mean over the families in the pair
                           set), `compute_share` = as a share of the reference's full run
    da_pooled_per_task.csv the same with every pair of the headline pool (schemes A and B,
                           seed 1904) pooled: the headline reading on ten checkpoints
    pairs_by_L.csv         per L: the design variants the grid plans at the grid seed and the pairs
                           usable against the reference (both members planned at the proxy size and at
                           the reference), planned vs with data today (README table)
    early_small_by_L.png   the left panel of highlights.png (all pairs of the headline pool)
                           followed by one panel per L: mean DA vs the reference's final
                           checkpoint by Chinchilla multiple, one line per proxy size,
                           solid BPB, dashed benchmarks (`early_small_by_L.csv`)

    python analysis/rq02_decision_accuracy/by_L.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import bucket_order, load_pools, size_bucket  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, md_table, replace_block  # noqa: E402
from pretrain.launch_trainings import DATA_SCHEMES, arches_for, mix_label, scheme_sizes  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq02_decision_accuracy.compute_da import (  # noqa: E402
    compute_ckpt_decision_accuracy, compute_early_small_decision_accuracy)
from analysis.rq02_decision_accuracy.early_small import MIN_PAIRS, SAFE_DA  # noqa: E402
from analysis.utils import (  # noqa: E402
    GRID_SEED, SMALL_SIZES, TARGET_SIZE, _is_parent_task, build_snr_pool)

OUT_ROOT = DECISION_ACCURACY
L_POOL = "predictivity_all"      # every scheme; the grid seed keeps replicate seeds out of the pairs
FRACS = [k / 10 for k in range(1, 11)]     # the ten evaluated checkpoints of every run (0.5C ... 5C)
mpl.rcParams.update(S.RC)


def _da_table(df: pd.DataFrame, label) -> list[dict]:
    """Per task, proxy size and fraction: DA vs the reference's final ranking
    and vs the size's own final ranking, over the pairs of `df`."""
    buckets = [b for b in bucket_order() if b in set(df["bucket"])]
    ref_full = df.loc[df["bucket"] == TARGET_SIZE, "compute"].max()
    rows = []
    for t, dft in tqdm(df.groupby("task", sort=False), desc=str(label)):
        if not _is_parent_task(t):
            continue
        ref = {(r["proxy_size"], r["frac"]): r for r in compute_early_small_decision_accuracy(dft, fracs=FRACS)}
        own = {(b, f): compute_ckpt_decision_accuracy(dft, t, b, f, return_n=True) for b in buckets for f in FRACS[:-1]}
        own = {k: v if isinstance(v, tuple) else (np.nan, 0) for k, v in own.items()}   # a bare NaN when the bucket is absent
        for b in buckets:
            for f in FRACS:
                r, o = ref.get((b, f)), own.get((b, f), (np.nan, 0))
                if r is None and not np.isfinite(o[0]):
                    continue
                # the compute the pair's proxies spent up to that checkpoint (mean over the families in the pair set)
                c = r["compute"] if r else f * dft.loc[dft["bucket"] == b, "compute"].max()
                rows.append({"task": t, "L": label, "proxy_size": b, "frac": f,
                             "da_ref": r["da"] if r else np.nan, "n_pairs_ref": r["n_pairs"] if r else 0,
                             "da_own": o[0], "n_pairs_own": o[1], "compute": c, "compute_share": c / ref_full})
    return rows


COLS = ["task", "L", "proxy_size", "frac", "da_ref", "n_pairs_ref", "da_own", "n_pairs_own", "compute", "compute_share"]


def da_by_L() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per L, pooled): the per-L table over the grid seed of every scheme,
    the pooled one over the headline pool's pairs (schemes A and B)."""
    df = build_snr_pool(L_POOL)
    df = df[df["seed"] == GRID_SEED].copy()
    df["bucket"] = df["size"].map(size_bucket)
    by_L = pd.DataFrame([r for L, dl in df.groupby("L") for r in _da_table(dl, f"L{L}")], columns=COLS)
    by_L["L"] = by_L["L"].str[1:].astype(int)
    pooled = pd.DataFrame(_da_table(df[df["scheme"].isin(["A", "B"])], "pooled"), columns=COLS).drop(columns="L")
    return by_L, pooled


def pairs_by_L(t: pd.DataFrame) -> pd.DataFrame:
    """Per L: the variants the grid plans at the grid seed, and the pairs
    usable for DA-size(proxy -> reference) — both members planned at the
    proxy size and at the reference — against the pairs that have data
    today at the proxy's final checkpoint, on BPB / on the benchmarks / on
    the training loss (a cell whose 1.7B has only the loss scored counts
    for the loss alone)."""
    planned = {}                                        # (L, size) -> set of variants
    for scheme, cfg in DATA_SCHEMES.items():
        for L in cfg["langs"]:
            for size in scheme_sizes(scheme, L):
                # arches_for, never cfg["arches"]: a scheme can be trained in
                # one architecture at some settings (AT3 is deep only at L15
                # and L30) or lack a hyperparams config at a size (the 3B is
                # deep only), and counting those as planned inflates the pair
                # count — 15 where the grid plans 10, at L15 and L30.
                for arch in arches_for(scheme, size, L):
                    planned.setdefault((L, size), set()).add(mix_label(L, arch, scheme))
    fin = t[t["frac"] == 1.0].copy()
    fin["kind"] = np.where(fin["task"].str.startswith("bpb_"), "bpb", np.where(fin["task"] == "train_loss", "loss", "bench"))
    have = fin.groupby(["L", "proxy_size", "kind"])["n_pairs_ref"].max()     # the pairs any task of that kind has
    rows = []
    for L in sorted({L for L, _ in planned}):
        variants = sorted(set.union(*[v for (l, _), v in planned.items() if l == L]))
        ref = planned.get((L, TARGET_SIZE), set())
        row = {"L": L, "variants": ", ".join(variants)}
        for s_ in SMALL_SIZES:
            n = len(planned.get((L, s_), set()) & ref)
            row[f"{s_}"] = f"{n * (n - 1) // 2} / " + "/".join(str(int(have.get((L, s_, k), 0))) for k in ("bpb", "bench", "loss"))
        rows.append(row)
    return pd.DataFrame(rows)


def _lines(ax, summary: pd.DataFrame, sizes: list, title: str) -> None:
    for group, ls in (("bpb", "-"), ("all benchmarks", "--")):
        for s_ in sizes:
            g = summary[(summary["group"] == group) & (summary["proxy_size"] == s_)].sort_values("chinchilla")
            if len(g):
                ax.plot(g["chinchilla"], g["da"], color=S.SIZE_COLOR.get(s_, S.MUTED), ls=ls, marker="o", ms=3, lw=1.3)
    ax.axhline(SAFE_DA, color=S.MUTED, lw=.8, ls=":")
    ax.set_ylim(0.25, 1.0); ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels([G.chinchilla(f) for f in (.2, .4, .6, .8, 1.0)])
    ax.set_xlim(0.3, 5.2)
    ax.set_title(title, loc="left", fontsize=8.5); ax.grid(color=S.GRID, lw=.6); S.clean(ax)


def _summary(pool: str, t: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Mean DA vs the reference over the gated tasks with >= MIN_PAIRS pairs
    (fewer is not a ranking), BPB and all benchmarks apart."""
    t = t[t["n_pairs_ref"] >= MIN_PAIRS]
    cells = G.mark_gated(G.add_meta(t.dropna(subset=["da_ref"])), pool, "proxy_size", "da_ref", TARGET_SIZE)
    cells = cells[cells["family"] != "loss"].dropna(subset=["da_ref"])
    cells["group"] = np.where(cells["family"] == "bpb", "bpb", "all benchmarks")
    summary = (cells.groupby(keys + ["group", "proxy_size", "frac"])
               .agg(da=("da_ref", "mean"), tasks=("task", "nunique"), median_pairs=("n_pairs_ref", "median")).reset_index())
    summary["chinchilla"] = summary["frac"] * G.CHINCHILLA_AT_FULL
    return summary


def figure(pool: str, out_dir: Path, t: pd.DataFrame, pooled: pd.DataFrame) -> pd.DataFrame:
    sizes = SMALL_SIZES + [TARGET_SIZE]
    summary, head = _summary(pool, t, ["L"]), _summary(pool, pooled, [])
    Ls = sorted(set(summary["L"]) | {1, 2, 8, 15, 30, 50, 100})
    few = [L for L in Ls if L not in set(summary["L"]) and L in set(t.loc[t["da_ref"].notna(), "L"])]   # has pairs, too few
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.4), sharey=True)
    flat = axes.ravel()
    _lines(flat[0], head, sizes, "all pairs (schemes A and B)")
    for ax, L in zip(flat[1:], Ls):
        g = summary[summary["L"] == L]
        _lines(ax, g, sizes, f"L{L}" + ("" if len(g) else f"  (< {MIN_PAIRS} pairs)" if L in few else "  (no pairs yet)"))
    for ax in flat[len(Ls) + 1:]:
        ax.axis("off")
    for ax in flat[4:]:
        ax.set_xlabel("proxy's training tokens (× Chinchilla)")
    for ax in axes[:, 0]:
        ax.set_ylabel(f"mean DA vs {TARGET_SIZE} final")
    flat[0].legend(handles=[plt.Line2D([], [], color=S.SIZE_COLOR[s_], lw=2, label=s_) for s_ in sizes if s_ in S.SIZE_COLOR]
                   + [plt.Line2D([], [], color=S.INK, ls="-", label="BPB"), plt.Line2D([], [], color=S.INK, ls="--", label="benchmarks")],
                   fontsize=6.5, frameon=False, ncol=2)
    top = G._header(fig, f"Early and small per language count: agreement with the {TARGET_SIZE} final ranking",
                    f"cell panel = pairs of design variants sharing that L (seed {GRID_SEED}, every scheme); first panel = every pair "
                    f"of the {pool} pool. DA = share of pairs the proxy orders like the {TARGET_SIZE} final checkpoint, mean over "
                    f"the gated tasks with ≥ {MIN_PAIRS} pairs; dotted line = {SAFE_DA}"
                    + (f". Left out for having fewer than {MIN_PAIRS} pairs: L{', L'.join(map(str, few))}" if few else ""))
    fig.tight_layout(rect=(0, 0, 1, top))
    pd.concat([head.assign(L="all"), summary]).to_csv(out_dir / "early_small_by_L.csv", index=False)
    S.save(fig, out_dir / "early_small_by_L.png", dpi=150)
    return summary


def generate_readme(pool: str, out_dir: Path, pairs: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    body = "\n\n".join([
        "## Per language count",
        f"**Pairs per L** — the design variants the grid plans at seed {GRID_SEED}, and per proxy size the pairs usable "
        f"against {TARGET_SIZE} (both members planned at that size and at {TARGET_SIZE}): planned / with data today on "
        "BPB / on the benchmarks / on the training loss. ZH and ES stop at 1B, so they never pair against the reference; "
        "L1, L2 and L100 have one pair, so their per-task DA is 0 or 1; L15 has no 1.7B cell yet.",
        md_table(list(pairs.columns), pairs.values.tolist()),
        f"The early-and-small reading one L at a time: pairs of design variants that share the L (seed {GRID_SEED} of every "
        f"scheme, `{L_POOL}`), against the {TARGET_SIZE} final ranking, on the ten evaluated checkpoints of every run; a cell "
        f"needs ≥ {MIN_PAIRS} pairs (rq02's rule), which today leaves out every L with one pair (the table above); the "
        f"first panel pools every pair of schemes A and B (the `{pool}` pool's, `da_pooled_per_task.csv`). "
        f"`da_by_L_per_task.csv` also carries each size's DA-ckpt within the L (`da_own`); rq04 reads both tables. "
        f"Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool {pool}`.",
        f"![Early and small per L]({stage}/{pool}/early_small_by_L.png)"])
    replace_block(OUT_ROOT / "README.md", "by-L", body, f"by_L.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose early_small_summary.csv is the first panel and whose gate applies")
    args = p.parse_args()
    out = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    table, pooled = da_by_L()
    table.to_csv(out / "da_by_L_per_task.csv", index=False)
    pooled.to_csv(out / "da_pooled_per_task.csv", index=False)
    print(f"Wrote {out / 'da_by_L_per_task.csv'} ({len(table)} rows) and da_pooled_per_task.csv ({len(pooled)} rows)")
    figure(args.pool, out, table, pooled)
    pairs = pairs_by_L(table)
    pairs.to_csv(out / "pairs_by_L.csv", index=False)
    print(pairs.to_string(index=False))
    generate_readme(args.pool, out, pairs)
