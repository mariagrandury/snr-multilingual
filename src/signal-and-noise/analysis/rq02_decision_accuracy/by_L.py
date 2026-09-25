"""Decision accuracy per language count — the early-and-small grid read one
L at a time: pairs of design variants that share the L (deep vs shallow,
scheme A vs B, AT3, the second language), at the grid seed of every scheme
(`predictivity_all`, seed 1904). L1 and L2 have few pairs once
its 1.7B cells are final; until then its cells stay blank.

    da_by_L_per_task.csv   per task, L, proxy size and fraction of the proxy's own run, on
                           the ten checkpoints of the shared k/10 grid (0.5C ... 5C): `da_ref`
                           vs the reference's final ranking (the early-and-small reading,
                           `n_pairs_ref` pairs) and `da_own` vs the proxy size's own final
                           ranking (DA-ckpt within the L, `n_pairs_own`); `compute` = the training FLOPs the
                           pair's proxies spent up to that checkpoint (mean over the families in the pair
                           set), `compute_share` = as a share of the reference's full run
    da_pooled_per_task.csv the same with EVERY pair at the grid seed pooled, schemes
                           A, B, AT3, ZH and ES alike: the headline reading on ten
                           checkpoints. The scheme is not held fixed here because a
                           temperature or a second-language swap is a design decision
                           like any other, and the per-L panels beside it have always
                           pooled the schemes — restricting only this panel made the
                           first panel a different population from the six next to it.
                           See "Why the pooled panel keeps every scheme" in README.md.
    pairs_by_L.csv         per L: the design variants the grid plans at the grid seed and the pairs
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

Six figures, one per (reading, variant). All six share the y axis, so any two of
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

    early_small_by_L_goal.png / _goal_with_bpb.png / _goal_above_80.png
    early_small_by_L_ckpt.png / _ckpt_with_bpb.png / _ckpt_above_80.png

                           Each writes the table it draws under the same name. The
                           reference size's line is identical in the goal and ckpt
                           figures — at the reference the two definitions coincide.

    python analysis/rq02_decision_accuracy/by_L.py --pool predictivity

`--by transformation` reads the same two decision accuracies one DESIGN AXIS
at a time instead of one L at a time: the MONO-AXIS pairs at the grid seed
(`utils.pair_sets`), split by the one axis each pair moves
(`scale_convergence.pairs_by_group`) — language count, depth, language list,
temperature, second language, English corpus — one panel per axis and a first
panel over every mono-axis pair. Same gate, pair minimum and filter variants;
the reliability filter is the mono-axis one (rule 15). Task counts sit at the
end of every line.

    da_by_transformation_per_task.csv                    per task, axis, proxy size and fraction
    early_small_by_transformation_{goal,ckpt}[_<variant>].png / .csv

    python analysis/rq02_decision_accuracy/by_L.py --pool predictivity --by transformation
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
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
from analysis.rq02_decision_accuracy.scale_convergence import AXIS_LABEL, OVERALL, pairs_by_group  # noqa: E402
from analysis.utils import (  # noqa: E402
    AXES_SUFFIX, DESIGN_AXES, GRID_SEED, SMALL_SIZES, TARGET_SIZE, _is_parent_task, build_snr_pool, design_axes,
    pair_sets)

OUT_ROOT = DECISION_ACCURACY
GITHUB = "https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis"
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
        # rule 5 up front: a task with fewer than MIN_PAIRS of the explicit pairs in its rows is NaN at every cell
        # (the kernels return exactly that), so the 45 kernel calls below are skipped
        if pairs is not None and sum(a in set(dft["family"]) and b in set(dft["family"]) for a, b in pairs) < MIN_PAIRS:
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


def da_by_transformation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per design axis, pooled): the mono-axis pairs at the grid seed, split
    by the one axis each pair moves; the pooled table is every mono-axis pair
    (the `_one_axis` pooled panel of the per-L figure)."""
    df = build_snr_pool(L_POOL)
    df = df[df["seed"] == GRID_SEED].copy()
    df["bucket"] = df["size"].map(size_bucket)
    groups = pairs_by_group(design_axes(df), "transformation", "mono-axis")
    # one process per axis, each on the rows of the families its pairs use (the kernels read nothing else)
    with mp.get_context("fork").Pool(len(groups)) as pool:
        tables = pool.starmap(_da_table, [(df[df["family"].isin({f for pr in pl for f in pr})], g, pl) for g, pl in groups.items()])
    by_axis = pd.DataFrame([r for g, rows in zip(groups, tables) if g != OVERALL for r in rows],
                           columns=COLS).rename(columns={"L": "axis"})
    pooled = pd.DataFrame(tables[list(groups).index(OVERALL)], columns=COLS).drop(columns="L")
    return by_axis, pooled


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


def _lines(ax, summary: pd.DataFrame, sizes: list, title: str, groups: tuple, counts: bool = False) -> None:
    for group, ls in groups:
        for s_ in sizes:
            g = summary[(summary["group"] == group) & (summary["proxy_size"] == s_)].sort_values("chinchilla")
            if len(g):
                ax.plot(g["chinchilla"], g["da"], color=S.SIZE_COLOR.get(s_, S.MUTED), ls=ls, marker="o", ms=3, lw=1.3)
                if counts:            # rule 13: the tasks behind the line, at its end (a range when the gate moves it)
                    lo, hi = int(g["tasks"].min()), int(g["tasks"].max())
                    ax.annotate(str(lo) if lo == hi else f"{lo}–{hi}", (g["chinchilla"].iloc[-1], g["da"].iloc[-1]),
                                textcoords="offset points", xytext=(4, 0), va="center", fontsize=5.5,
                                color=S.SIZE_COLOR.get(s_, S.MUTED))
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
# The panels of `--by transformation`: one per design axis (the seed is the null, not a decision).
PANEL_AXES = [AXIS_LABEL[k] for k in DESIGN_AXES if k != "seed"]
# grouping -> (its column in the table, the panel list, how a panel is labelled, the first panel's title)
BY = {"L": ("L", PANEL_LS, "L{}".format, "all pairs (every scheme)"),
      "transformation": ("axis", PANEL_AXES, str, "all mono-axis pairs")}
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
           name: str = "goal", variant: str = "", axes: str = "multi-axis", by: str = "L") -> pd.DataFrame | None:
    reading, value = READINGS[name], READINGS[name]["value"]
    groups, filt, _ = VARIANTS[variant]
    col, panel_keys, lab, first = BY[by]
    # the transformation panels are mono-axis by construction, so the stem carries no pair-set suffix
    stem = f"early_small_by_{by}_{name}" + (f"_{variant}" if variant else "") + (AXES_SUFFIX[axes] if by == "L" else "")
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
    summary, head = _summary(pool, t, [col], reading), _summary(pool, pooled, [], reading)
    summary, head = summary[summary["group"].isin(drawn)], head[head["group"].isin(drawn)]
    Ls = sorted(set(summary[col]) | set(panel_keys)) if by == "L" else panel_keys + [p for p in summary[col].unique() if p not in panel_keys]
    # "too few pairs" vs "no pairs yet" is a statement about the tasks this figure
    # draws, so ask it of the kinds in `groups` — an L with only BPB data has no
    # pairs at all in a benchmarks-only figure, it does not have too few.
    drawable = t[t["task"].str.startswith("bpb_") == ("bpb" in drawn)] if len(drawn) == 1 else t
    few = [L for L in Ls if L not in set(summary[col]) and L in set(drawable.loc[drawable[value].notna(), col])]
    counts = by == "transformation"
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.4), sharey=True)
    flat = axes.ravel()
    n_all = pooled[reading["n_col"]].max()
    _lines(flat[0], head, sizes, first + (f"  ({int(n_all)} pairs)" if pd.notna(n_all) else ""), groups, counts)
    for ax, L in zip(flat[1:], Ls):
        g = summary[summary[col] == L]
        n = int(t.loc[t[col] == L, reading["n_col"]].max()) if (t[col] == L).any() else 0
        _lines(ax, g, sizes, lab(L) + (f"  ({n} pairs)" if len(g) else f"  (< {MIN_PAIRS} pairs)" if L in few else "  (no pairs yet)"), groups, counts)
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
    population = (f"cell panel = pairs of design variants sharing that L (seed {GRID_SEED}, every scheme); first panel = every pair "
                  f"at that seed, every scheme (A, B, AT3, ZH, ES)" if by == "L" else
                  f"cell panel = the mono-axis pairs that move that ONE design axis (seed {GRID_SEED}, every scheme, `{L_POOL}`; "
                  f"the pairs behind it in brackets); first panel = every mono-axis pair; the number at the end of a line = tasks behind it")
    top = G._header(fig, reading["title"].replace("per language count", "per design axis" if by == "transformation" else "per language count"),
                    f"{population}. DA = share of pairs the proxy orders like {reading['what']}, mean over "
                    f"the gated {'tasks' if variant == 'with_bpb' else 'benchmark tasks'}"
                    + (f" reliable on {crit} (DA ≥ {thresh:g}, {red} reduction, reliable_tasks.py, {axes} pairs)" if filt else "")
                    + f" with ≥ {MIN_PAIRS} pairs; dotted line = {SAFE_DA}"
                    + (f". Left out for having fewer than {MIN_PAIRS} pairs: {', '.join(map(lab, few))}" if few else ""))
    # the first empty cell becomes the reliable-cell inventory. It cannot share the
    # grid's y axis (a task index, not a DA), so the placeholder is replaced by a
    # fresh subplot in the same grid slot, which tight_layout still manages.
    if filt and spare:
        flat[spare[0]].remove()
        _reliable_panel(fig.add_subplot(2, 4, spare[0] + 1), keep, red, thresh, crit)
    fig.tight_layout(rect=(0, 0, 1, top))
    summary = pd.concat([head.assign(**{col: "all"}), summary])
    summary.to_csv(out_dir / f"{stem}.csv", index=False)
    S.save(fig, out_dir / f"{stem}.png", dpi=150)
    return summary


def generate_readme(pool: str, out_dir: Path, pairs: pd.DataFrame) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    body = "\n\n".join([
        "## Per language count",
        f"**Pairs per L** — the design variants the grid plans at seed {GRID_SEED}, and per proxy size the pairs usable "
        f"against {TARGET_SIZE} (both members planned at that size and at {TARGET_SIZE}): planned / with data today on "
        "BPB / on the benchmarks / on the training loss. ES stops at 1B, so it never pairs against the reference; ZH runs to "
        f"{TARGET_SIZE} (2026-09-20) and is the third L2 family. A cell below MIN_PAIRS (3) families is left empty (rule 5), "
        "so a thin L shows blanks rather than a 0/1 reading.",
        md_table(list(pairs.columns), pairs.values.tolist()),
        f"The early-and-small reading one L at a time: pairs of design variants that share the L (seed {GRID_SEED} of every "
        f"scheme, `{L_POOL}`), against the {TARGET_SIZE} final ranking, on the ten evaluated checkpoints of every run; a cell "
        f"needs ≥ {MIN_PAIRS} pairs (rq02's rule), which today leaves out every L with one pair (the table above); the "
        f"first panel pools every pair at that seed, every scheme included (`da_pooled_per_task.csv`). "
        f"`da_by_L_per_task.csv` also carries each size's DA-ckpt within the L (`da_own`); rq04 reads both tables. "
        f"Regenerate with `python analysis/rq02_decision_accuracy/by_L.py --pool {pool}`.",
        f"Two of rq02's three decision accuracies have a checkpoint axis and so a figure here. **DA-goal** ranks the "
        f"proxy at any checkpoint against the {TARGET_SIZE} final checkpoint; **DA-ckpt** ranks it against its own "
        f"size's final checkpoint, so the 175M line asks what 175M would have decided early and what it misses is the "
        f"checkpoint alone. The distance between the two is what the proxy *size* costs, and the {TARGET_SIZE} line is "
        f"the same curve in both — at the reference the definitions coincide. DA-ckpt has no 5C column: a run's final "
        f"checkpoint is its own reference. The third, **DA-size**, is DA-goal read at 5C alone and lives in "
        f"`da_per_task.csv`. Each figure comes in a benchmarks-only version and a `_with_bpb` one that adds the solid "
        f"per-size BPB lines; all four share the y axis, so any two overlay.",
        f"![DA-goal per L]({stage}/{pool}/early_small_by_L_goal.png)",
        f"![DA-goal per L, with BPB]({stage}/{pool}/early_small_by_L_goal_with_bpb.png)",
        f"![DA-ckpt per L]({stage}/{pool}/early_small_by_L_ckpt.png)",
        f"![DA-ckpt per L, with BPB]({stage}/{pool}/early_small_by_L_ckpt_with_bpb.png)",
        "A third variant of each restricts the mean to the (benchmark, language) cells that rank reliably on BOTH "
        "axes (DA-size and DA-ckpt each ≥ 0.8, `reliable_tasks.py`): the plain panels average over every gated "
        "benchmark, these average over the benchmarks that work.",
        f"![DA-goal per L, reliable cells only]({stage}/{pool}/early_small_by_L_goal_above_80.png)",
        f"![DA-ckpt per L, reliable cells only]({stage}/{pool}/early_small_by_L_ckpt_above_80.png)"])
    replace_block(OUT_ROOT / "README.md", "by-L", body, f"by_L.py --pool {pool}")


def generate_readme_transformation(pool: str, out_dir: Path, summaries: dict) -> None:
    """`summaries[(reading, variant)]` = the per-axis mean lines the figure drew."""
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel, gh = f"{stage}/{pool}", f"{GITHUB}/rq02_decision_accuracy/{stage}/{pool}"
    stem = lambda r, v: f"early_small_by_transformation_{r}" + (f"_{v}" if v else "")
    sizes = SMALL_SIZES + [TARGET_SIZE]

    def cell(s: pd.DataFrame, axis: str, size: str) -> str:
        g = s[(s["axis"] == axis) & (s["proxy_size"] == size) & (s["group"] == "all benchmarks")].sort_values("chinchilla")
        lo, hi = (int(g["tasks"].min()), int(g["tasks"].max())) if len(g) else (0, 0)
        return f"{g['da'].iloc[0]:.2f} → {g['da'].iloc[-1]:.2f} ({lo if lo == hi else f'{lo}–{hi}'})" if len(g) else "—"

    def table(r: str, v: str) -> str:
        s = summaries[(r, v)]
        axes_ = ["all"] + [a for a in PANEL_AXES if a in set(s["axis"])]
        c1, c2 = (G.chinchilla(s["frac"].min()), G.chinchilla(s["frac"].max())) if len(s) else ("", "")
        return md_table([f"axis (DA-{r}, {c1} → {c2}, tasks)"] + sizes, [[a] + [cell(s, a, z) for z in sizes] for a in axes_])

    def early(r: str) -> list[str]:
        """Per axis at the smallest proxy: the checkpoint from which the line stays ≥ SAFE_DA, or never."""
        s = summaries[(r, "")]
        out = []
        for a in [a for a in PANEL_AXES if a in set(s["axis"])]:
            g = s[(s["axis"] == a) & (s["proxy_size"] == SMALL_SIZES[0]) & (s["group"] == "all benchmarks")].sort_values("chinchilla")
            if not len(g):
                continue
            ok = (g["da"] >= SAFE_DA).to_numpy()[::-1]
            k = int(ok.cumprod().sum())            # trailing run of safe checkpoints
            out.append(f"{a}: " + (f"from {G.chinchilla(g['frac'].iloc[-k])} ({g['da'].iloc[-k]:.2f})" if k else
                                   f"never (max {g['da'].max():.2f})"))
        return out

    axes_ = [a for a in PANEL_AXES if a in set(summaries[("ckpt", "")]["axis"])]
    body = "\n\n".join([
        "## Early and small per design axis",
        f"The per-L reading above pools every design axis inside an L; this one splits the MONO-AXIS pairs at seed {GRID_SEED} "
        f"(`{L_POOL}`, every scheme) by the one axis each pair moves — {', '.join(axes_)} — one panel per axis and a first panel "
        f"over every mono-axis pair (median pairs per cell up to {int(summaries[('ckpt', '')]['median_pairs'].max())}). Same gate (rule 1), "
        f"pair minimum (rule 5) and filter variants as the per-L figures; the `above_66_ckpt` twin filters the DA-ckpt figure and "
        f"`above_66_either` the DA-goal one, both on the mono-axis reliability (rule 15). Task counts sit at the end of every line "
        f"and the populations differ between panels and sizes (rule 13). Regenerate with "
        f"`python analysis/rq02_decision_accuracy/by_L.py --pool {pool} --by transformation`.",
        f"![DA-ckpt per design axis]({rel}/{stem('ckpt', '')}.png)",
        f"**DA-ckpt** (against the proxy size's own final; cell = mean DA at the first → last drawn checkpoint, tasks behind the line in brackets):",
        table("ckpt", ""),
        "Key findings:",
        "\n".join([f"- At {SMALL_SIZES[0]} the DA-ckpt line clears {SAFE_DA} and stays there — " + "; ".join(early("ckpt")) + ".",
                    f"- At {TARGET_SIZE} — " + "; ".join(f"{a}: {cell(summaries[('ckpt', '')], a, TARGET_SIZE)}" for a in axes_) + "."]),
        "Follow-ups:",
        "\n".join(["- A `_with_bpb` variant per axis, to see whether BPB decides the temperature and the second language earlier than the benchmarks do.",
                    "- The same panels on the L8 languages only (`scale_convergence.py --langs L8` does it for DA-size), so the language-count panel is read on one task set.",
                    "- Once BT3 trains, the temperature panel gains the B-vs-BT3 pairs and the list panel AT3-vs-BT3 with no code change."]),
        f"![DA-goal per design axis]({rel}/{stem('goal', '')}.png)",
        f"**DA-goal** (against the {TARGET_SIZE} final):",
        table("goal", ""),
        "Key findings:",
        "\n".join([f"- At {SMALL_SIZES[0]} the DA-goal line clears {SAFE_DA} and stays there — " + "; ".join(early("goal")) + ".",
                    f"- The distance between the DA-goal and DA-ckpt cell of an axis at a proxy size is what the size costs; the "
                    f"{TARGET_SIZE} column is the same line in both."]),
        "Follow-ups:",
        "\n".join(["- The size axis of the same split is `scale_convergence_transformation_panels.png` (DA-size pooled over decisions).",
                    "- A jackknife band per axis line (leave one family out), as the scale-convergence panels carry."]),
        "Filtered twins (reliable cells only): "
        + ", ".join(f"[`{stem(r, v)}.png`]({rel}/{stem(r, v)}.png)" for r, v in summaries if v),
        "Files: " + ", ".join(f"[`{stem(r, v)}.{e}`]({gh}/{stem(r, v)}.{e})" for r, v in summaries for e in ("png", "csv"))
        + f", [`da_by_transformation_per_task.csv`]({gh}/da_by_transformation_per_task.csv)."])
    replace_block(OUT_ROOT / "README.md", "early-small-by-transformation", body, f"by_L.py --pool {pool} --by transformation")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose early_small_summary.csv is the first panel and whose gate applies")
    p.add_argument("--axes", default="multi-axis", choices=["multi-axis", "mono-axis"],
                   help="the pair set (rule 15); mono-axis writes the `_one_axis` twins")
    p.add_argument("--by", default="L", choices=list(BY), help="one panel per language count, or per design axis (mono-axis pairs)")
    args = p.parse_args()
    out = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    if args.by == "transformation":
        table, pooled = da_by_transformation()
        table.to_csv(out / "da_by_transformation_per_task.csv", index=False)
        print(f"[transformation] {len(table)} by-axis rows, {len(pooled)} pooled rows")
        summaries = {(name, v): figure(args.pool, out, table, pooled, name, v, "mono-axis", "transformation")
                     for name in READINGS for v, (_, _, readings) in VARIANTS.items() if name in readings and v != "with_bpb"}
        generate_readme_transformation(args.pool, out, {k: v for k, v in summaries.items() if v is not None})
        sys.exit(0)
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
    pairs.to_csv(out / "pairs_by_L.csv", index=False)
    print(pairs.to_string(index=False))
    generate_readme(args.pool, out, pairs)
