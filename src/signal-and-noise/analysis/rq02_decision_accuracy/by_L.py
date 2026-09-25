"""Decision accuracy per language count — the early-and-small grid read one
L at a time: pairs of design variants that share the L (deep vs shallow,
scheme A vs B, AT3, the second language), at the grid seed of every scheme
(`predictivity_all`, seed 1904). L1 and L2 have few pairs once
its 1.7B cells are final; until then its cells stay blank.

    da_by_L_per_task<axes>.csv   per task, L, proxy size and fraction of the proxy's own run, on
                           the ten checkpoints of the shared k/10 grid (0.5C ... 5C): `da_ref`
                           vs the reference's final ranking (the early-and-small reading,
                           `n_pairs_ref` pairs) and `da_own` vs the proxy size's own final
                           ranking (DA-ckpt within the L, `n_pairs_own`); `compute` = the training FLOPs the
                           pair's proxies spent up to that checkpoint (mean over the families in the pair
                           set), `compute_share` = as a share of the reference's full run
    da_pooled_per_task<axes>.csv the same with EVERY pair at the grid seed pooled, schemes
                           A, B, AT3, ZH and ES alike: the headline reading on ten
                           checkpoints. The scheme is not held fixed here because a
                           temperature or a second-language swap is a design decision
                           like any other, and the per-L panels beside it have always
                           pooled the schemes — restricting only this panel made the
                           first panel a different population from the six next to it.
                           See "Why the pooled panel keeps every scheme" in README.md.
    pairs_by_L<axes>.csv   per L: the design variants the grid plans at the grid seed and the pairs
                           usable against the reference (both members planned at the proxy size and at
                           the reference), planned vs with data today (README table)
Three decision accuracies share this table, by what the ranking is compared against:

    DA-size   the reference size's final checkpoint vs each proxy size's OWN final
              checkpoint — one number per (task, size), no checkpoint axis. It is the
              5C column of DA-goal, and rq02's `da_per_task.csv` is its home.
    DA-goal   the reference size's final checkpoint vs the proxy at ANY checkpoint
              (`da_ref`): the early-and-small reading, the whole curve.
    DA-ckpt   the proxy size's OWN final checkpoint vs that same size earlier in its
              run (`da_own`): what the size would have decided early. A run's final is
              its own reference, so the 5C column is trivially 1.0 and is not drawn.

Ten figures per pair set, one per (reading, variant); above_66_ckpt draws DA-ckpt
alone and above_66_either DA-goal alone. All share the y axis, so any two of
them overlay. The variants differ only in which tasks the mean runs over:

    <plain>     every gated benchmark task, dashed, one line per proxy size
    _with_bpb   the same plus the solid per-size BPB lines
    _above_80   only the (benchmark, language) cells that rank reliably on BOTH
                axes — DA-size and DA-ckpt each >= reliable_tasks.THRESH, from
                `da_reliable_tasks.csv`. The plain panel asks "what does the
                average benchmark do"; this one asks "what do the benchmarks
                that work do", which is the population a practitioner would
                actually read. An L whose reliable cells fall below MIN_PAIRS
                is blank, as everywhere else.

    early_small_by_L_goal[_<variant>]<axes>.png
    early_small_by_L_ckpt[_<variant>]<axes>.png

    <axes> is the pair set's AXES_SUFFIX: `_multi_axes`, or `_mono_axis` with
    --axes mono-axis. The README block is written from the multi-axis run only.

                           Each writes the table it draws under the same name. The
                           reference size's line is identical in the goal and ckpt
                           figures — at the reference the two definitions coincide.

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
from analysis.rq02_decision_accuracy.reliable_tasks import FILTERS, load_reliable  # noqa: E402
from analysis.utils import (  # noqa: E402
    AXES_SUFFIX, GRID_SEED, SMALL_SIZES, TARGET_SIZE, _is_parent_task, build_snr_pool, design_axes,
    pair_sets)

OUT_ROOT = DECISION_ACCURACY
L_POOL = "predictivity_all"      # every scheme; the grid seed keeps replicate seeds out of the pairs
FRACS = [k / 10 for k in range(1, 11)]     # the ten evaluated checkpoints of every run (0.5C ... 5C)
mpl.rcParams.update(S.RC)


def _da_table(df: pd.DataFrame, label, pairs=None) -> list[dict]:
    """Per task, proxy size and fraction: DA vs the reference's final ranking
    and vs the size's own final ranking, over the pairs of `df`."""
    buckets = [b for b in bucket_order() if b in set(df["bucket"])]
    ref_full = df.loc[df["bucket"] == TARGET_SIZE, "compute"].max()
    rows = []
    for t, dft in tqdm(df.groupby("task", sort=False), desc=str(label)):
        if not _is_parent_task(t):
            continue
        ref = {(r["proxy_size"], r["frac"]): r for r in compute_early_small_decision_accuracy(dft, fracs=FRACS, pairs=pairs)}
        own = {(b, f): compute_ckpt_decision_accuracy(dft, t, b, f, return_n=True, pairs=pairs) for b in buckets for f in FRACS[:-1]}
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


def da_by_L(axes: str = "multi-axis") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per L, pooled): both over the grid seed of every scheme — the per-L one
    within a language count, the pooled one across every pair."""
    df = build_snr_pool(L_POOL)
    df = df[df["seed"] == GRID_SEED].copy()
    df["bucket"] = df["size"].map(size_bucket)
    # The pair set (rule 15): None is every pair, which is what the multi-axis
    # reading means; mono-axis hands the kernels the explicit one-axis list.
    pairs = None if axes == "multi-axis" else pair_sets(design_axes(df))[axes]
    by_L = pd.DataFrame([r for L, dl in df.groupby("L") for r in _da_table(dl, f"L{L}", pairs)], columns=COLS)
    by_L["L"] = by_L["L"].str[1:].astype(int)
    pooled = pd.DataFrame(_da_table(df, "pooled", pairs), columns=COLS).drop(columns="L")
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


def _lines(ax, summary: pd.DataFrame, sizes: list, title: str, groups: tuple) -> None:
    for group, ls in groups:
        for s_ in sizes:
            g = summary[(summary["group"] == group) & (summary["proxy_size"] == s_)].sort_values("chinchilla")
            if len(g):
                ax.plot(g["chinchilla"], g["da"], color=S.SIZE_COLOR.get(s_, S.MUTED), ls=ls, marker="o", ms=3, lw=1.3)
    ax.axhline(SAFE_DA, color=S.MUTED, lw=.8, ls=":")
    ax.set_ylim(0.25, 1.0); ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels([G.chinchilla(f) for f in (.2, .4, .6, .8, 1.0)])
    ax.set_xlim(0.3, 5.2)
    ax.set_title(title, loc="left", fontsize=8.5); ax.grid(color=S.GRID, lw=.6); S.clean(ax)


# The two readings of the same table (the module docstring defines all three
# decision accuracies). `reference` is the extra size the gate must also clear —
# under DA-ckpt the proxy IS the reference, so the proxy's own column, which
# mark_gated always checks, is the whole gate.
READINGS = {
    "goal": {"value": "da_ref", "n_col": "n_pairs_ref", "reference": TARGET_SIZE,
             "ylabel": f"mean DA-goal vs {TARGET_SIZE} final",
             "title": f"Early and small per language count: DA-goal, agreement with the {TARGET_SIZE} final ranking",
             "what": f"the {TARGET_SIZE} final checkpoint"},
    "ckpt": {"value": "da_own", "n_col": "n_pairs_own", "reference": None,
             "ylabel": "mean DA-ckpt vs its own size's final",
             "title": "Early and small per language count: DA-ckpt, agreement with the proxy size's own final ranking",
             "what": "its own size's final checkpoint"},
}
# The L panels the grid draws. L100 was planned and dropped (plan/l100_data_mixture.md),
# so it is not a cell of this grid; an L with data but too few pairs still gets its panel.
PANEL_LS = [1, 2, 8, 15, 30, 50]
# variant -> (filename suffix, the groups drawn, whether the reliable-task filter applies)
BENCH, WITH_BPB = ((("all benchmarks", "--"),), (("bpb", "-"), ("all benchmarks", "--")))
BOTH_READINGS = ("goal", "ckpt")
# variant -> (the groups drawn, the reliable_tasks FILTERS name or None, which
# readings get it). The one-axis filters exist for rq2_above_66_one, which reads
# each panel over the tasks reliable on THAT panel's own axis: the DA-ckpt panel
# over the DA-ckpt passers, the DA-goal panel over either.
VARIANTS = {
    "": (BENCH, None, BOTH_READINGS),
    "with_bpb": (WITH_BPB, None, BOTH_READINGS),
    "above_80": (BENCH, "above_80", BOTH_READINGS),
    "above_66_both": (BENCH, "above_66_both", BOTH_READINGS),
    "above_66_ckpt": (BENCH, "above_66_ckpt", ("ckpt",)),
    "above_66_either": (BENCH, "above_66_either", ("goal",)),
}


def _summary(pool: str, t: pd.DataFrame, keys: list, reading: dict) -> pd.DataFrame:
    """Mean DA vs the reading's reference over the gated tasks with >= MIN_PAIRS
    pairs (fewer is not a ranking), BPB and all benchmarks apart."""
    value, n_col = reading["value"], reading["n_col"]
    t = t[t[n_col] >= MIN_PAIRS]
    cells = G.mark_gated(G.add_meta(t.dropna(subset=[value])), pool, "proxy_size", value, reading["reference"])
    cells = cells[cells["family"] != "loss"].dropna(subset=[value])
    cells["group"] = np.where(cells["family"] == "bpb", "bpb", "all benchmarks")
    summary = (cells.groupby(keys + ["group", "proxy_size", "frac"])
               .agg(da=(value, "mean"), tasks=("task", "nunique"), median_pairs=(n_col, "median")).reset_index())
    summary["chinchilla"] = summary["frac"] * G.CHINCHILLA_AT_FULL
    return summary


def _reliable_panel(ax, keep: pd.DataFrame, red: str, thresh: float, crit: str) -> None:
    """The spare cell of the grid, in every filtered variant: WHICH cells the
    panels average over, and how far past the threshold each one is. Without it
    the reader has to open a CSV to learn that the figure rests on one
    benchmark family."""
    d = keep.sort_values(["benchmark", "language"], ascending=[True, True])
    y = np.arange(len(d))
    lo = d[[f"da_size_{red}", f"da_ckpt_{red}"]].min(axis=1)
    hi = d[[f"da_size_{red}", f"da_ckpt_{red}"]].max(axis=1)
    ax.hlines(y, lo, hi, color=S.GRID, lw=1.4, zorder=1)
    ax.scatter(d[f"da_size_{red}"], y, s=16, color=S.RAMP[1], label="DA-size", zorder=3)
    ax.scatter(d[f"da_ckpt_{red}"], y, s=16, color=S.SERIES[1], marker="s", label="DA-ckpt", zorder=3)
    ax.axvline(thresh, color=S.MUTED, lw=.8, ls=":")
    ax.set_yticks(y); ax.set_yticklabels(d["task"], fontsize=5.5); ax.invert_yaxis()
    ax.set_ylim(len(d) - 0.4, -0.6)
    ax.set_xlim(thresh - 0.04, 1.02)
    ax.set_xlabel(f"DA at the `{red}` cell", fontsize=7)
    ax.set_title(f"the {len(d)} cells these panels average over — reliable on {crit} "
                 f"({d['benchmark'].nunique()} benchmark(s), {d['language'].nunique()} languages)",
                 loc="left", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.grid(color=S.GRID, lw=.6, axis="x"); S.clean(ax)


def figure(pool: str, out_dir: Path, t: pd.DataFrame, pooled: pd.DataFrame,
           name: str = "goal", variant: str = "", axes: str = "multi-axis") -> pd.DataFrame | None:
    reading, value = READINGS[name], READINGS[name]["value"]
    groups, filt, _ = VARIANTS[variant]
    stem = f"early_small_by_L_{name}" + (f"_{variant}" if variant else "") + AXES_SUFFIX[axes]
    keep = red = crit = None
    thresh = 0.0
    if filt:
        red, thresh, crit = FILTERS[filt]
        keep = load_reliable(out_dir, filt, axes)
        if keep is None:
            return None
        names = set(keep["task"])
        t, pooled = t[t["task"].isin(names)], pooled[pooled["task"].isin(names)]
    sizes = SMALL_SIZES + [TARGET_SIZE]
    drawn = [g for g, _ in groups]
    summary, head = _summary(pool, t, ["L"], reading), _summary(pool, pooled, [], reading)
    summary, head = summary[summary["group"].isin(drawn)], head[head["group"].isin(drawn)]
    Ls = sorted(set(summary["L"]) | set(PANEL_LS))
    # "too few pairs" vs "no pairs yet" is a statement about the tasks this figure
    # draws, so ask it of the kinds in `groups` — an L with only BPB data has no
    # pairs at all in a benchmarks-only figure, it does not have too few.
    drawable = t[t["task"].str.startswith("bpb_") == ("bpb" in drawn)] if len(drawn) == 1 else t
    few = [L for L in Ls if L not in set(summary["L"]) and L in set(drawable.loc[drawable[value].notna(), "L"])]
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.4), sharey=True)
    flat = axes.ravel()
    _lines(flat[0], head, sizes, "all pairs (every scheme)", groups)
    for ax, L in zip(flat[1:], Ls):
        g = summary[summary["L"] == L]
        _lines(ax, g, sizes, f"L{L}" + ("" if len(g) else f"  (< {MIN_PAIRS} pairs)" if L in few else "  (no pairs yet)"), groups)
    spare = list(range(len(Ls) + 1, len(flat)))
    for i in spare:
        flat[i].axis("off")
    for ax in flat[4:]:
        ax.set_xlabel("proxy's training tokens (× Chinchilla)")
    for ax in axes[:, 0]:
        ax.set_ylabel(reading["ylabel"])
    flat[0].legend(handles=[plt.Line2D([], [], color=S.SIZE_COLOR[s_], lw=2, label=s_) for s_ in sizes if s_ in S.SIZE_COLOR]
                   + ([plt.Line2D([], [], color=S.INK, ls="-", label="BPB"),
                       plt.Line2D([], [], color=S.INK, ls="--", label="benchmarks")] if variant == "with_bpb" else
                      [plt.Line2D([], [], color=S.INK, ls="--", label="benchmarks")]),
                   fontsize=6.5, frameon=False, ncol=2)
    top = G._header(fig, reading["title"],
                    f"cell panel = pairs of design variants sharing that L (seed {GRID_SEED}, every scheme); first panel = every pair "
                    f"at that seed, every scheme (A, B, AT3, ZH, ES). DA = share of pairs the proxy orders like {reading['what']}, mean over "
                    f"the gated {'tasks' if variant == 'with_bpb' else 'benchmark tasks'}"
                    + (f" reliable on {crit} (DA ≥ {thresh:g}, {red} reduction, reliable_tasks.py)" if filt else "")
                    + f" with ≥ {MIN_PAIRS} pairs; dotted line = {SAFE_DA}"
                    + (f". Left out for having fewer than {MIN_PAIRS} pairs: L{', L'.join(map(str, few))}" if few else ""))
    # the first empty cell becomes the reliable-cell inventory. It cannot share the
    # grid's y axis (a task index, not a DA), so the placeholder is replaced by a
    # fresh subplot in the same grid slot, which tight_layout still manages.
    if filt and spare:
        flat[spare[0]].remove()
        _reliable_panel(fig.add_subplot(2, 4, spare[0] + 1), keep, red, thresh, crit)
    fig.tight_layout(rect=(0, 0, 1, top))
    pd.concat([head.assign(L="all"), summary]).to_csv(out_dir / f"{stem}.csv", index=False)
    S.save(fig, out_dir / f"{stem}.png", dpi=150)
    return summary


def generate_readme(pool: str, out_dir: Path, pairs: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    m = AXES_SUFFIX["multi-axis"]
    stage = load_pools()[pool].get("stage", "pretraining")
    body = "\n\n".join([
        "## Per language count",
        f"**Pairs per L** — the design variants the grid plans at seed {GRID_SEED}, and per proxy size the pairs usable "
        f"against {TARGET_SIZE} (both members planned at that size and at {TARGET_SIZE}): planned / with data today on "
        f"BPB / on the benchmarks / on the training loss. ZH and ES run to {TARGET_SIZE} and are the second and third L2 "
        "families. A cell below MIN_PAIRS (3) families is left empty (rule 5), "
        "so a thin L shows blanks rather than a 0/1 reading.",
        md_table(list(pairs.columns), pairs.values.tolist()),
        f"The early-and-small reading one L at a time: pairs of design variants that share the L (seed {GRID_SEED} of every "
        f"scheme, `{L_POOL}`), against the {TARGET_SIZE} final ranking, on the ten evaluated checkpoints of every run; a cell "
        f"needs ≥ {MIN_PAIRS} pairs (rq02's rule), which today leaves out every L with one pair (the table above); the "
        f"first panel pools every pair at that seed, every scheme included (`da_pooled_per_task{m}.csv`). "
        f"`da_by_L_per_task{m}.csv` also carries each size's DA-ckpt within the L (`da_own`); rq04 reads both tables. "
        f"The `_mono_axis` twins of every table and figure are the same over the one-axis pairs (rule 15). "
        f"Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool {pool} [--axes mono-axis]`.",
        f"Two of rq02's three decision accuracies have a checkpoint axis and so a figure here. **DA-goal** ranks the "
        f"proxy at any checkpoint against the {TARGET_SIZE} final checkpoint; **DA-ckpt** ranks it against its own "
        f"size's final checkpoint, so the 175M line asks what 175M would have decided early and what it misses is the "
        f"checkpoint alone. The distance between the two is what the proxy *size* costs, and the {TARGET_SIZE} line is "
        f"the same curve in both — at the reference the definitions coincide. DA-ckpt has no 5C column: a run's final "
        f"checkpoint is its own reference. The third, **DA-size**, is DA-goal read at 5C alone and lives in "
        f"`da_per_task.csv`. Each figure comes in a benchmarks-only version and a `_with_bpb` one that adds the solid "
        f"per-size BPB lines; all four share the y axis, so any two overlay.",
        f"![DA-goal per L]({stage}/{pool}/early_small_by_L_goal{m}.png)",
        f"![DA-goal per L, with BPB]({stage}/{pool}/early_small_by_L_goal_with_bpb{m}.png)",
        f"![DA-ckpt per L]({stage}/{pool}/early_small_by_L_ckpt{m}.png)",
        f"![DA-ckpt per L, with BPB]({stage}/{pool}/early_small_by_L_ckpt_with_bpb{m}.png)",
        "A third variant of each restricts the mean to the (benchmark, language) cells that rank reliably on BOTH "
        "axes (DA-size and DA-ckpt each ≥ 0.8, `reliable_tasks.py`): the plain panels average over every gated "
        "benchmark, these average over the benchmarks that work.",
        f"![DA-goal per L, reliable cells only]({stage}/{pool}/early_small_by_L_goal_above_80{m}.png)",
        f"![DA-ckpt per L, reliable cells only]({stage}/{pool}/early_small_by_L_ckpt_above_80{m}.png)"])
    replace_block(OUT_ROOT / "README.md", "by-L", body, f"by_L.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose early_small_summary.csv is the first panel and whose gate applies")
    p.add_argument("--axes", default="multi-axis", choices=["multi-axis", "mono-axis"],
                   help="the pair set (rule 15); mono-axis writes the `_mono_axis` twins")
    args = p.parse_args()
    out = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    table, pooled = da_by_L(args.axes)
    sfx = AXES_SUFFIX[args.axes]
    table.to_csv(out / f"da_by_L_per_task{sfx}.csv", index=False)
    pooled.to_csv(out / f"da_pooled_per_task{sfx}.csv", index=False)
    print(f"[{args.axes}] {len(table)} by-L rows, {len(pooled)} pooled rows")
    for name in READINGS:
        for variant, (_, _, readings) in VARIANTS.items():
            if name in readings:
                figure(args.pool, out, table, pooled, name, variant, args.axes)
    pairs = pairs_by_L(table)
    pairs.to_csv(out / f"pairs_by_L{sfx}.csv", index=False)
    print(pairs.to_string(index=False))
    if args.axes == "multi-axis":
        generate_readme(args.pool, out, pairs)
