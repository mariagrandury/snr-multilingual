"""Cross-task predictability — from which proxy size, and from which
checkpoint, does the ranking of the design variants on task x read like the
FINAL ranking on task y?

The per-task DA of rq02 is the diagonal of this: a task predicting its own
reference ranking. Here every parent task is tried as the proxy for every
other one (x = proxy task on the columns, y = target task on the rows), over
the pairs of design variants of the pool (`predictivity`: seed 1904, schemes
A and B). A cell holds a *level* (the same code as the other level maps):

    cross_task_size.png / .csv   smallest proxy size at which x's final ranking safely predicts
                                 y's final ranking at the reference (DA-size, levels = SMALL_SIZES)
    cross_task_ckpt.png / .csv   earliest checkpoint of x (fraction of its own run, ten
                                 checkpoints) that safely predicts y's final ranking at the same
                                 size; the within-size pairs of every size are pooled (DA-ckpt)
    cross_task_{size,ckpt}_by_family.png / .csv   benchmark x benchmark: the median level over the
                                 task pairs of the two families, and the share that never reach it
    cross_task_{size,ckpt}_by_language.png / .csv   language x language, over the pairs of tasks of
                                 the SAME benchmark (bpb_x -> bpb_y, arc_x -> arc_y, ...), languages
                                 in the resource order of the scheme-A lists (English first)

"Safely" is rq02's rule: DA >= SAFE_DA over >= MIN_PAIRS pairs at that level
and at every larger level with information. The rq00 gate applies to both
sides: a benchmark task at chance at the proxy size (x) or at the size it
is ranked at (y) contributes no pair; a cell whose every level the gate
emptied is grey, a cell with no data is white. Ties follow
`snr.metrics.decision_acc_fast`. The wide CSVs hold the level index (-1
never, -2 filtered out by the gate) with the target task on the rows.

    python analysis/rq02_decision_accuracy/cross_task.py --pool predictivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import bucket_order, load_pools, size_bucket  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, replace_block  # noqa: E402
from analysis.paths import DECISION_ACCURACY  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq02_decision_accuracy.compute_da import CKPT_TOL, add_family_column  # noqa: E402
from analysis.rq02_decision_accuracy.early_small import MIN_PAIRS, SAFE_DA  # noqa: E402
from analysis.utils import (  # noqa: E402
    SMALL_SIZES, TARGET_SIZE, _is_parent_task, assign_language, benchmark_family, build_snr_pool)
from evals.scripts.utils.configs import fineweb_language  # noqa: E402
from pretrain.launch_trainings import cell_fineweb_subsets  # noqa: E402

OUT_ROOT = DECISION_ACCURACY
FRACS = [k / 10 for k in range(1, 11)]     # the ten evaluated checkpoints of every run (0.5C ... 5C)
mpl.rcParams.update(S.RC)


# --- scores and pair signs ---------------------------------------------------

def scores_at(df: pd.DataFrame, frac: float) -> pd.DataFrame:
    """task x (bucket, family): the score of the checkpoint nearest `frac` of
    the family's own run on that task (its final at 1.0), under the CKPT_TOL
    rule of `compute_ckpt_decision_accuracy`; NaN where no checkpoint is near."""
    key = ["task", "bucket", "family"]
    g = df[key + ["step", "primary_score"]]
    mx = g.groupby(key)["step"].transform("max")
    if frac >= 1.0:
        sel = g[g["step"] == mx]
    else:
        pre = g[g["step"] < mx]
        dist = (pre["step"] - frac * mx[pre.index]).abs()
        idx = dist.groupby([pre["task"], pre["bucket"], pre["family"]]).idxmin()
        sel = pre.loc[idx[dist.loc[idx].to_numpy() <= CKPT_TOL * mx.loc[idx].to_numpy()]]
    return sel.drop_duplicates(key).set_index(key)["primary_score"].unstack(["bucket", "family"])


def pair_signs(mat: pd.DataFrame, at_chance: pd.Series | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For a task x family score frame: (signs, valid, informed), each task x
    unordered pair. `valid` is the pair with both scores and the task not at
    chance; `informed` ignores the gate (to tell "gated" from "no data")."""
    n = mat.shape[1]
    i, j = np.triu_indices(n, 1)
    v = mat.to_numpy(dtype=float)
    d = v[:, i] - v[:, j]
    informed = np.isfinite(d)
    signs = np.sign(np.nan_to_num(d))
    valid = informed.copy()
    if at_chance is not None:
        valid &= ~at_chance.reindex(mat.index).fillna(False).to_numpy(dtype=bool)[:, None]
    return signs, valid, informed


def cross_da(sp, vp, sr, vr) -> tuple[np.ndarray, np.ndarray]:
    """(DA, n_pairs), target task x proxy task, from the pair signs of the
    proxy (sp, vp) and of the reference (sr, vr): the share of the pairs both
    sides have that carry the same sign (a tie on one side only is a miss)."""
    agree = sum(((sr == s) & vr).astype(float) @ ((sp == s) & vp).astype(float).T for s in (-1.0, 0.0, 1.0))
    n = vr.astype(float) @ vp.astype(float).T
    with np.errstate(invalid="ignore", divide="ignore"):
        da = np.where(n >= MIN_PAIRS, agree / n, np.nan)
    return da, n


def smallest_safe_nd(ok: np.ndarray) -> np.ndarray:
    """`G.smallest_safe` over the last axis of a 1/0/NaN array: the index of
    the smallest level from which the condition holds at every larger level
    with information; NEVER_CODE if the largest informed level fails, NaN
    where no level is informed."""
    L = ok.shape[-1]
    idx = np.arange(L)
    known = np.isfinite(ok)
    last_fail = np.where(known & (ok == 0), idx, -1).max(-1)
    cand = np.where(known & (idx > last_fail[..., None]), idx, L).min(-1)
    out = np.where(cand == L, G.NEVER_CODE, cand).astype(float)
    out[~known.any(-1)] = np.nan
    return out


def _levels_to_map(tasks: list, ok: np.ndarray, gated: np.ndarray) -> pd.DataFrame:
    """target x proxy level matrix from per-level pass/fail (`ok`, NaN = no
    value) and per-level "the gate emptied it" flags."""
    level = smallest_safe_nd(ok)
    level[np.isnan(level) & gated.any(-1)] = G.GATED
    return pd.DataFrame(level, index=pd.Index(tasks, name="target_task"), columns=pd.Index(tasks, name="proxy_task"))


# --- the two maps --------------------------------------------------------------

def _at_chance(mask: pd.DataFrame | None, bucket: str, tasks: list) -> pd.Series | None:
    if mask is None or bucket not in mask.columns:
        return None
    return (mask[bucket] == 0).fillna(False).astype(bool).reindex(tasks).fillna(False)


def size_map(scores: dict, mask, tasks: list, sizes: list) -> tuple[pd.DataFrame, dict]:
    """DA-size across tasks: x's final ranking at each proxy size against y's
    final ranking at the reference. Returns the level map and the DA per level."""
    final = scores[1.0]
    ref = final[TARGET_SIZE].reindex(tasks)
    sr, vr, ir = pair_signs(ref, _at_chance(mask, TARGET_SIZE, tasks))
    ok = np.full((len(tasks), len(tasks), len(sizes)), np.nan)
    gated = np.zeros_like(ok, dtype=bool)
    das = {}
    for k, b in enumerate(sizes):
        if b not in final.columns.get_level_values(0):
            continue
        prox = final[b].reindex(tasks)
        prox = prox[[f for f in prox.columns if f in ref.columns]]
        refb = ref[prox.columns]
        sp, vp, ip = pair_signs(prox, _at_chance(mask, b, tasks))
        srb, vrb, irb = pair_signs(refb, _at_chance(mask, TARGET_SIZE, tasks))
        da, n = cross_da(sp, vp, srb, vrb)
        _, n_all = cross_da(sp, ip, srb, irb)
        ok[:, :, k] = np.where(np.isfinite(da), (da >= SAFE_DA).astype(float), np.nan)
        gated[:, :, k] = (n_all >= MIN_PAIRS) & (n < MIN_PAIRS)
        das[b] = da
    return _levels_to_map(tasks, ok, gated), das


def ckpt_map(scores: dict, mask, tasks: list, fracs: list, buckets: list) -> tuple[pd.DataFrame, dict]:
    """DA-ckpt across tasks: x at a fraction of its run against y's final at
    the same size, the within-size pairs of every size pooled."""
    final = scores[1.0]
    ok = np.full((len(tasks), len(tasks), len(fracs)), np.nan)
    gated = np.zeros_like(ok, dtype=bool)
    das = {}
    for k, f in enumerate(fracs):
        parts = []
        for b in buckets:
            if b not in scores[f].columns.get_level_values(0) or b not in final.columns.get_level_values(0):
                continue
            prox = scores[f][b].reindex(tasks)
            fams = [x for x in prox.columns if x in final[b].columns]
            if len(fams) < 2:
                continue
            chance = _at_chance(mask, b, tasks)
            parts.append((pair_signs(prox[fams], chance), pair_signs(final[b][fams].reindex(tasks), chance)))
        if not parts:
            continue
        sp, vp, ip = (np.concatenate([p[0][i] for p in parts], axis=1) for i in range(3))
        sr, vr, ir = (np.concatenate([p[1][i] for p in parts], axis=1) for i in range(3))
        da, n = cross_da(sp, vp, sr, vr)
        _, n_all = cross_da(sp, ip, sr, ir)
        ok[:, :, k] = np.where(np.isfinite(da), (da >= SAFE_DA).astype(float), np.nan)
        gated[:, :, k] = (n_all >= MIN_PAIRS) & (n < MIN_PAIRS)
        das[f] = da
    return _levels_to_map(tasks, ok, gated), das


# --- drawing -------------------------------------------------------------------

def task_order(tasks) -> list:
    """Families in the panel order (BPB first), tasks by language then name."""
    meta = pd.DataFrame({"task": list(tasks)})
    meta["family"] = meta["task"].map(benchmark_family)
    meta["language"] = meta["task"].map(assign_language)
    fams = G.panel_order(meta["family"].unique())
    meta["rank"] = meta["family"].map({f: i for i, f in enumerate(fams)})
    return meta.sort_values(["rank", "language", "task"])["task"].tolist()


def big_map(mat: pd.DataFrame, path: Path, *, levels: list, level_label, title: str, note: str, cbar: str) -> None:
    """The full task x task level map: no cell text, one tick per family."""
    order = task_order(mat.index)
    mat = mat.reindex(index=order, columns=order)
    fams = pd.Series([benchmark_family(t) for t in order])
    edges = np.flatnonzero(fams.ne(fams.shift()).to_numpy())
    centres = (np.r_[edges, len(order)][:-1] + np.r_[edges, len(order)][1:] - 1) / 2
    labels = [f"{fams[e]} ({n})" for e, n in zip(edges, np.diff(np.r_[edges, len(order)]))]
    colours = [S.SEQ(x) for x in np.linspace(0.15, 0.95, len(levels))]
    cmap = ListedColormap([S.NODATA, G.NEVER] + colours); cmap.set_bad(S.SURFACE)
    norm = BoundaryNorm(np.arange(-2.5, len(levels) + 0.5, 1), cmap.N)
    fig, ax = plt.subplots(figsize=(13.5, 12.6))
    ax.imshow(np.ma.masked_invalid(mat.to_numpy(dtype=float)), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    for e in edges[1:]:
        ax.axhline(e - 0.5, color=S.INK, lw=0.4); ax.axvline(e - 0.5, color=S.INK, lw=0.4)
    ax.set_xticks(centres); ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_yticks(centres); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("proxy task x (its ranking is read)", fontsize=8)
    ax.set_ylabel("target task y (its final ranking is predicted)", fontsize=8)
    S.clean(ax, spines=()); ax.tick_params(length=0)
    any_gated = (mat == G.GATED).any().any()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours + [G.NEVER] + ([S.NODATA] if any_gated else [])]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=S.SURFACE, edgecolor=S.GRID))
    ax.legend(handles, [level_label(l) for l in levels] + ["never"] + (["filtered out by the gate"] if any_gated else [])
              + ["no value"], title=cbar, fontsize=7, title_fontsize=7.5, frameon=False, loc="upper left",
              bbox_to_anchor=(1.005, 1.0))
    top = G._header(fig, title, note)
    fig.tight_layout(rect=(0, 0, 1, top))
    mat.to_csv(path.with_suffix(".csv"))
    S.save(fig, path, dpi=200)
    print(f"Wrote {path.name} ({len(order)} x {len(order)} tasks, {len(edges)} families)")


L_LISTS = (1, 2, 8, 15, 30, 50, 100)


def resource_rank() -> dict:
    """language -> position in English + the scheme-A L100 list (resource-
    ranked; every smaller list is its prefix, so rank < L means "trained at L")."""
    return {l: i for i, l in enumerate(["en"] + [fineweb_language(sub) for sub in cell_fineweb_subsets(100, "AT3")])}


def resource_order(languages) -> list:
    rank = resource_rank()
    return sorted(languages, key=lambda l: (rank.get(l, len(rank)), l))


def l_boundaries(languages: list) -> list:
    """Row/column positions where the L-lists end in a resource-ordered list
    (the languages of L2, L8, ... form the blocks between them)."""
    rank = resource_rank()
    return sorted({sum(rank.get(l, len(rank)) < L for l in languages) for L in L_LISTS} - {0, len(languages)})


def group_map(mat: pd.DataFrame, path: Path, *, group, order, same_family: bool, levels: list, level_label,
              title: str, note: str, cbar: str, xlabel: str, ylabel: str, separators=lambda keys: ()) -> pd.DataFrame:
    """A map aggregated by `group` (a task -> label function): the median
    level index over the task pairs of the two groups that reach a level
    (rounded up), and, in the CSV, the share that never do. `same_family`
    keeps only pairs of tasks of one benchmark (the by-language reading)."""
    long = mat.stack(future_stack=True).rename("level").reset_index()
    long["target"] = long["target_task"].map(group)
    long["proxy"] = long["proxy_task"].map(group)
    long = long[long["level"].notna() & ~long["target"].isin(["??", "multi"]) & ~long["proxy"].isin(["??", "multi"])]
    if same_family:
        long = long[long["target_task"].map(benchmark_family) == long["proxy_task"].map(benchmark_family)]
    rows = []
    for (t, p), g in long.groupby(["target", "proxy"]):
        reached = g.loc[g["level"] >= 0, "level"]
        rows.append({"target": t, "proxy": p, "n_cells": len(g),
                     "share_reached": len(reached) / len(g), "share_never": (g["level"] == G.NEVER_CODE).mean(),
                     "share_gated": (g["level"] == G.GATED).mean(),
                     "median_level": level_label(levels[int(np.ceil(reached.median()))]) if len(reached) else "",
                     "level_index": np.ceil(reached.median()) if len(reached) else
                     (G.GATED if (g["level"] == G.GATED).all() else G.NEVER_CODE)})
    t = pd.DataFrame(rows)
    wide = t.pivot(index="target", columns="proxy", values="level_index")
    keys = order(sorted(set(wide.index) | set(wide.columns)))
    wide = wide.reindex(index=keys, columns=keys)
    G.level_heatmap(wide, path, levels=levels, level_label=level_label, title=title, note=note, cbar=cbar,
                    xlabel=xlabel, ylabel=ylabel, rows=keys, cols=keys, separators=separators(keys))
    t.to_csv(path.with_suffix(".csv"), index=False)      # clearer columns than the map's default table
    return t


# --- main ----------------------------------------------------------------------

def run(pool: str, out_dir: Path) -> None:
    df = build_snr_pool(pool)
    df = add_family_column(df)
    df["bucket"] = df["size"].map(size_bucket)
    df = df[df["task"].map(_is_parent_task)]
    tasks = sorted(df["task"].unique())
    order = bucket_order()
    sizes = [b for b in order if b in SMALL_SIZES]
    buckets = [b for b in order if b in set(df["bucket"])]
    mask = load_mask(pool)
    if mask is None:
        print(f"  ({pool}: no above-random mask of its own, gating with {CANONICAL_POOL}'s)")
        mask = load_mask(CANONICAL_POOL)
    scores = {f: scores_at(df, f) for f in FRACS}
    ref_fams = scores[1.0][TARGET_SIZE].shape[1]
    pairs_ref = ref_fams * (ref_fams - 1) // 2
    within = {b: scores[1.0][b].shape[1] for b in buckets}
    pairs_within = sum(n * (n - 1) // 2 for n in within.values())

    size_lv, size_da = size_map(scores, mask, tasks, sizes)
    ckpt_lv, ckpt_da = ckpt_map(scores, mask, tasks, FRACS[:-1], buckets)

    # the diagonal is rq02's per-task DA-size: check it against the committed table
    per_task = out_dir / "da_per_task.csv"
    if per_task.exists():
        da = pd.read_csv(per_task, index_col="task")
        for b in sizes:
            col = f"decision_acc_size_{b}"
            if col in da.columns and b in size_da:
                mine = pd.Series(np.diag(size_da[b]), index=tasks)
                both = pd.concat([mine.rename("cross"), da[col].rename("rq02")], axis=1).dropna()
                if len(both):
                    print(f"  diagonal vs rq02 DA-size at {b}: {len(both)} tasks, max |diff| = {(both['cross'] - both['rq02']).abs().max():.3f}")

    common = (f"Pool `{pool}`, {len(tasks)} parent tasks; a cell holds a level only where the two tasks share >= {MIN_PAIRS} "
              f"pairs of design variants; DA >= {SAFE_DA} at that level and at every larger level with a value. "
              "Grey: the gate (a benchmark at chance) emptied every level; white: no data; the diagonal is rq02's own-task DA.")
    big_map(size_lv, out_dir / "cross_task_size.png", levels=sizes, level_label=str,
            title=f"Cross-task DA-size: smallest size of proxy task x whose final ranking predicts task y's final ranking at {TARGET_SIZE}",
            note=f"{common} {ref_fams} design variants at {TARGET_SIZE} ({pairs_ref} pairs).", cbar="smallest proxy size")
    big_map(ckpt_lv, out_dir / "cross_task_ckpt.png", levels=FRACS[:-1], level_label=G.chinchilla,
            title="Cross-task DA-ckpt: earliest checkpoint of proxy task x that predicts task y's final ranking at the same size",
            note=f"{common} Within-size pairs pooled over {', '.join(b for b, n in within.items() if n >= 2)} "
                 f"({pairs_within} pairs); the level is the fraction of x's own run as a Chinchilla multiple (runs train 5C).",
            cbar="earliest checkpoint")
    fam = dict(group=benchmark_family, order=G.panel_order, same_family=False, xlabel="proxy benchmark x", ylabel="target benchmark y",
               note="Cell: the median level over the task pairs of the two benchmarks (every language) that reach one, rounded up; "
                    "the CSV adds the share that never do.")
    lang = dict(group=assign_language, order=resource_order, same_family=True, separators=l_boundaries,
                xlabel="proxy language x", ylabel="target language y",
                note="Cell: the median level over the pairs of tasks of the SAME benchmark in the two languages (bpb_x -> bpb_y, "
                     "arc_x -> arc_y, ...) that reach one, rounded up; the CSV adds the share that never do. Languages in the "
                     "resource order of the scheme-A lists: English, then the L100 list; the lines mark where the L2, L8, L15, L30, L50 "
                     "and L100 lists end, so the languages between two lines enter the mixture at the same L.")
    for kind, lv, levels, label, cb in [("size", size_lv, sizes, str, "median smallest proxy size"),
                                        ("ckpt", ckpt_lv, FRACS[:-1], G.chinchilla, "median earliest checkpoint")]:
        group_map(lv, out_dir / f"cross_task_{kind}_by_family.png", levels=levels, level_label=label, cbar=cb,
                  title=f"Cross-task DA-{kind} by benchmark", **fam)
        group_map(lv, out_dir / f"cross_task_{kind}_by_language.png", levels=levels, level_label=label, cbar=cb,
                  title=f"Cross-task DA-{kind} by language (same benchmark)", **lang)
    generate_readme(pool, out_dir, len(tasks), ref_fams, pairs_ref, pairs_within)


def generate_readme(pool: str, out_dir: Path, n_tasks: int, ref_fams: int, pairs_ref: int, pairs_within: int) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    body = "\n\n".join([
        "## Cross-task predictability",
        f"Every parent task as the proxy for every other one ({n_tasks} x {n_tasks}): the cell is the smallest proxy size "
        f"(DA-size, {ref_fams} variants at {TARGET_SIZE}, {pairs_ref} pairs) or the earliest checkpoint (DA-ckpt, the within-size "
        f"pairs of every size pooled, {pairs_within} pairs, ten checkpoints) at which the ranking on task x (columns) safely "
        f"predicts the final ranking on task y (rows): DA >= {SAFE_DA} over >= {MIN_PAIRS} pairs there and at every larger "
        "level with a value. The diagonal is rq02's own-task DA; the gate empties a benchmark's pairs at every size where it "
        "is at chance. The `_by_family` maps take the median level over the task pairs of two benchmarks, the `_by_language` "
        "maps over the same-benchmark task pairs of two languages (resource order of the scheme-A lists). "
        f"Regenerate with `python analysis/rq02_decision_accuracy/cross_task.py --pool {pool}`.",
        f"![Cross-task DA-size by benchmark]({stage}/{pool}/cross_task_size_by_family.png)",
        f"![Cross-task DA-ckpt by benchmark]({stage}/{pool}/cross_task_ckpt_by_family.png)",
        f"![Cross-task DA-size by language]({stage}/{pool}/cross_task_size_by_language.png)",
        f"![Cross-task DA-ckpt by language]({stage}/{pool}/cross_task_ckpt_by_language.png)",
        f"Full task-level maps: [`cross_task_size.png`]({stage}/{pool}/cross_task_size.png), "
        f"[`cross_task_ckpt.png`]({stage}/{pool}/cross_task_ckpt.png)."])
    replace_block(OUT_ROOT / "README.md", "cross-task", body, f"cross_task.py --pool {pool}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose design-variant pairs and gate apply")
    args = p.parse_args()
    out = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    out.mkdir(parents=True, exist_ok=True)
    run(args.pool, out)
