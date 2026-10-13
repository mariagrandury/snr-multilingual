"""RQ5 — Which proxy sizes rank a design decision like the reference, and how does that depend on the number of languages?

The plan's question (plan/small-to-large-predictivity-training-plan.md,
"Analysis"): at a given number of languages L, does a proxy size rank a design
choice the way the reference size — TARGET_SIZE, 1.7B (rule 9) — does?
rq00–rq04 ask which *benchmarks* carry reliable signal; this RQ asks which
*model sizes* do.

  intervention DA   — per (intervention, L, population, proxy size, fraction of
                      the proxy's run): the share of population items on which
                      the proxy and the reference (final checkpoint) prefer the
                      same level. Six two-level interventions on the design
                      axes (analysis/RULES.md, Definitions), each read with the
                      other axes at their baseline: depth (deep vs shallow, at
                      scheme A, T=1), data scheme A vs B and A vs C (deep, T=1;
                      read per L, since the recipe a letter names depends on the
                      L: B is DCLMP at L1, ZH at L2 and the diversity-first list
                      at L8-L30, C is FWEB at L1 and ES at L2) and temperature
                      (T=1 vs T=3, deep, scheme A), activation (deep vs
                      swiglu: xIELU vs SwiGLU) and optimizer (deep vs muon:
                      AdEMAMix vs Muon), both at scheme A, T=1, L8-L30. Each
                      is checked to be one of rq02's mono-axis pairs
                      (`mono_axis`, utils.moved_axes). A setting whose 1.7B
                      result has not landed is listed by
                      `pending_references` and reads "not yet". Populations:
                      per-language BPB on the languages both levels train
                      (`bpb_trained`; the languages neither trains are rq06's
                      measurement, RULES.md rule 2), the benchmark
                      tasks (`benchmark`), and the two single-item decisions, the
                      macro BPB (`bpb_macro`) and the training loss (`loss`). With two models per item this is
                      `snr.metrics.decision_acc_fast` per item — sign
                      agreement, items the reference ties dropped.
                      `decision_acc_decided` is the same on the items whose
                      reference |Δ| is at least DECIDED sds of the difference
                      (sqrt(2) x the per-run seed sd, itself a median over the
                      cells with 3 replicates): where the reference's own
                      preference is inside seed noise there is no decision to
                      agree with.
                      `size_matched` is False on the 600M depth cells:
                      shallow 600M is not size-matched to deep 600M
                      (utils.UNMATCHED_PAIRS; kept, flagged; RULES.md
                      "Size matching").
  effect at the reference — per intervention, the |Δ| in seed standard
                      deviations (the paper's "is there a decision to make?").

`early_decision.py` next to this script reads the decision table for "how
small and how early" (paper RQ2); rq06 reads it for the never-trained
languages; rq01's `scaling_law_error.py` and rq03's `effect_vs_noise.py` hold
the two other reads this folder used to carry. The paper's RQ4 figure
(`rq4_interventions`) is drawn here.

    python analysis/rq05_design_decisions/analyze.py --pool predictivity_seeds
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import fmt, md_table, replace_block  # noqa: E402
from analysis.paths import DESIGN_DECISIONS  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.utils import (CKPT_DA_EARLY_FRACS,  # noqa: E402
    GRID_SEED, RELIABLE_DA, TARGET_SIZE, at_fraction, data_build, design_axes, finals, ladder_frame, lower_is_better,
    moved_axes, passes_gate, size_match_note, size_matched, size_order, trained_bpb_tasks)
from pretrain.launch_trainings import DATA_SCHEMES, ladders_for  # noqa: E402

OUT_ROOT = DESIGN_DECISIONS
CANONICAL = "predictivity_seeds"        # every cell: all seeds and data builds
MIN_ITEMS = 3                         # fewest population items for a DA cell
# rq00 computes the above-random gate on the grid-seed pool; the all-seeds
# pool has no mask of its own, so fall back to that one rather than leave the
# benchmarks ungated (a task at chance in both cells decides nothing).
GATE_POOL = "predictivity"


def gate_mask(pool: str) -> pd.DataFrame | None:
    """The above-random mask of `pool`, or of GATE_POOL when the pool has none."""
    mask = load_mask(pool)
    return mask if mask is not None else load_mask(GATE_POOL)
# The reference's |Δ| is a difference of two single runs, so its null sd is
# sqrt(2) x the per-run seed sd; DECIDED counts in those difference sds.
DECIDED = 2.0
FRACS = list(CKPT_DA_EARLY_FRACS) + [1.0]     # where the proxy is read: every evaluated tenth of its run (rule 3)
# key -> (label, axis, levels, held: {column: baseline level}). The first level
# is the baseline; the reference is TARGET_SIZE (1.7B, rule 9), and an
# (intervention, L) without a 1.7B cell at both levels is skipped. Depth and the
# held baseline read the `ladder` column, not the depth level `arch`: swiglu is
# deep-shaped and would join deep's cells. The data build is read as its
# (scheme, T): a scheme decision holds T=1 (scheme A alone would also match
# AT3), the temperature decision holds scheme A.
INTERVENTIONS = {
    "arch":        ("depth (deep vs shallow)",  "ladder", ("deep", "shallow"), {"scheme": "A", "T": 1}),
    "scheme_B":    ("data scheme (A vs B)",     "scheme", ("A", "B"),          {"ladder": "deep", "T": 1}),
    "scheme_C":    ("data scheme (A vs C)",     "scheme", ("A", "C"),          {"ladder": "deep", "T": 1}),
    "temperature": ("temperature (T=1 vs T=3)", "T",      (1, 3),              {"ladder": "deep", "scheme": "A"}),
    # the swiglu and muon ladders differ from deep on one design axis each
    # (launch_trainings.LADDERS), at L8, L15 and L30 of scheme A: rq02's
    # mono-axis activation and optimizer pairs (utils.moved_axes), read here
    # against the 1.7B reference like the four above
    "activation":  ("activation (xIELU vs SwiGLU)",  "ladder", ("deep", "swiglu"), {"scheme": "A", "T": 1}),
    "optimizer":   ("optimizer (AdEMAMix vs Muon)",  "ladder", ("deep", "muon"),   {"scheme": "A", "T": 1}),
}


def at_baseline(df: pd.DataFrame, held: dict) -> pd.DataFrame:
    """The rows of `df` at an intervention's held baseline levels."""
    return df[(df[list(held)] == pd.Series(held)).all(axis=1)]


def _builds(rows: pd.DataFrame, axis: str, levels: tuple) -> list[str]:
    """The data builds behind the two levels in `rows` (one L): A and ZH for
    the scheme decision at L2, A alone for the depth decision."""
    return sorted(set(rows.loc[rows[axis].isin(levels), "data"]))


def mono_axis(rows: pd.DataFrame, axis: str, levels: tuple) -> str | None:
    """The one design axis the two levels' families in `rows` (one L, one
    seed) move, by rq02's pair rule (`utils.moved_axes`, RULES.md rule 15):
    an intervention is a mono-axis pair or it is a bug. None when a level has
    no family here."""
    fams = [rows.loc[rows[axis] == lv, "family"].unique() for lv in levels]
    if any(len(f) == 0 for f in fams):
        return None
    attrs = design_axes(rows[rows["family"].isin(np.concatenate(fams))])
    moved = {tuple(moved_axes(attrs.loc[a], attrs.loc[b])) for a in fams[0] for b in fams[1]}
    if len(moved) != 1 or len(next(iter(moved))) != 1:
        raise ValueError(f"{axis} {levels}: the two levels move {sorted(moved)}, not one design axis")
    return next(iter(moved))[0]


def _planned(key: str, L: int, level) -> bool:
    """Whether the launcher plans the TARGET_SIZE cell of one level of
    intervention `key` at L (`launch_trainings.ladders_for`)."""
    _, axis, _, held = INTERVENTIONS[key]
    at = {"ladder": "deep", "scheme": "A", "T": 1, **held, axis: level}
    build = data_build(int(L), at["scheme"], int(at["T"]))
    return build is not None and at["ladder"] in ladders_for(build, TARGET_SIZE, int(L))


def pending_references(fin: pd.DataFrame) -> pd.DataFrame:
    """The (intervention, L, population) settings whose reference the
    launcher plans but whose TARGET_SIZE result has not landed at both levels
    (the loader drops a run, or a task's series, until its final save is
    scored): rule 9 leaves their DA-size and effect cells empty, and the
    tables and figures say "not yet" there. `proxies` = the sizes both levels
    already have."""
    grid = fin[fin["seed"] == GRID_SEED]
    rows = []
    for key, (label, axis, levels, held) in INTERVENTIONS.items():
        for L, g in at_baseline(grid, held).groupby("L"):
            if not all(_planned(key, L, lv) for lv in levels):
                continue
            for pop in CELL_POPULATIONS:
                piv = _pivot(_population(g, pop, int(L), levels, axis), axis, levels)
                sizes = [] if piv is None else size_order(piv.index.get_level_values("size"))
                if TARGET_SIZE not in sizes:
                    rows.append({"intervention": key, "label": label, "L": int(L), "population": pop,
                                 "proxies": ",".join(sizes)})
    return pd.DataFrame(rows, columns=["intervention", "label", "L", "population", "proxies"])


def pending_note(pending: pd.DataFrame) -> str:
    """One sentence naming the settings `pending_references` lists; empty when none."""
    if pending.empty:
        return ""
    words = [f"{lab} on {pop} at " + ", ".join(f"L{L}" for L in g["L"])
             for (lab, pop), g in pending.groupby(["label", "population"], sort=False)]
    return (f" These settings have no {TARGET_SIZE} result at both levels yet, so they have no DA-size or "
            "effect cell: " + " and ".join([", ".join(words[:-1]), words[-1]] if len(words) > 1 else words) + ".")


# `bpb_untrained` and `bpb_all` are not rq05's: a score on a language the
# mixture does not train is rq06's measurement (RULES.md rule 2), and the
# loader does not deliver those rows here. rq06 passes `populations=("bpb_all",)`
# on its untrained frame for the per-group transfer table.
POPULATIONS = ("bpb_trained", "benchmark", "bpb_macro", "loss")
SINGLE = {"bpb_macro": "bpb_macro", "loss": "train_loss"}   # one-task populations: the aggregates
CELL_POPULATIONS = ("bpb_trained", "benchmark")   # the items behind the per-benchmark / per-language tables
COLOUR = dict(zip(INTERVENTIONS, [S.RAMP[3], S.RAMP[1], S.SERIES[2], S.SERIES[1], "#8c1d18", S.OKABE_ITO["purple"]]))
mpl.rcParams.update(S.RC)


# A scheme decision is a different recipe swap at each L (scheme_B: A vs DCLMP
# at L1, A vs ZH at L2, A vs the diversity-first lists at L8-L30; scheme_C: A vs
# FWEB at L1, A vs ES at L2), so it is read per L and never averaged across
# recipes: a reader that averages a decision table over L re-keys it with
# `by_recipe` first. Depth and temperature are one swap at every L.
def recipe(key: str, L: int) -> str | None:
    """The data build the second level of intervention `key` reads at L."""
    _, axis, levels, held = INTERVENTIONS[key]
    at = {"scheme": "A", "T": 1, **held, axis: levels[1]}
    return data_build(int(L), at["scheme"], int(at["T"]))


def _recipes() -> dict[str, tuple[str, str, str]]:
    """key -> (label, colour, marker) of every key `by_recipe` writes: a
    scheme intervention's colour, one marker per build (its own letter's build
    keeps the plain key and the circle)."""
    out = {}
    for k, (label, axis, levels, held) in INTERVENTIONS.items():
        if axis != "scheme":
            out[k] = (label, COLOUR[k], "o")
            continue
        builds = sorted((d for d, v in DATA_SCHEMES.items() if v["letter"] == levels[1] and int(v["temp"]) == held["T"]),
                        key=lambda d: (d != levels[1], -min(DATA_SCHEMES[d]["langs"])))
        for d, mk in zip(builds, "osD^v"):
            Ls = sorted(DATA_SCHEMES[d]["langs"])
            where = f"L{Ls[0]}" if len(Ls) == 1 else f"L{Ls[0]}–L{Ls[-1]}"
            out[k if d == levels[1] else f"{k}:{d}"] = (f"{label[:-1]}, {d} at {where})", COLOUR[k], mk)
    return out


RECIPES = _recipes()


def by_recipe(da: pd.DataFrame) -> pd.DataFrame:
    """`da` (rows with `intervention` and `L`) re-keyed for a reader that
    averages over L: a scheme intervention's rows take the key of the build
    its second level reads (`scheme_B:ZH`, `scheme_C:FWEB`; the L8-L30 lists
    keep `scheme_B`, the planned list decision), so no line or cell mixes two
    recipes. `RECIPES` names, colours and marks the keys."""
    if da.empty:
        return da
    keys = []
    for k, L in zip(da["intervention"], da["L"]):
        b = recipe(k, L) if INTERVENTIONS[k][1] == "scheme" else None
        keys.append(k if b in (None, INTERVENTIONS[k][2][1]) else f"{k}:{b}")
    out = da.assign(intervention=keys)
    if "label" in out:
        out["label"] = out["intervention"].map(lambda k: RECIPES[k][0])
    return out


def _population(sub: pd.DataFrame, name: str, L: int, levels: tuple, axis: str) -> pd.DataFrame:
    """Rows of `sub` (already at one L) belonging to one population."""
    if name == "benchmark":
        return sub[sub["kind"] == "benchmark"]
    if name in SINGLE:
        return sub[sub["task"] == SINGLE[name]]
    bpb = sub[(sub["kind"] == "bpb") & (sub["task"] != "bpb_macro")]
    if name == "bpb_all":
        return bpb
    tr = [trained_bpb_tasks(L, d) for d in _builds(sub, axis, levels)]
    if not tr or any(t is None for t in tr):    # no level here, or a level defines no list at this L
        return bpb.iloc[0:0]
    if name == "bpb_trained":
        return bpb[bpb["task"].isin(set.intersection(*tr))]
    raise ValueError(f"!!! RULE 2: population {name!r} reads untrained languages, which only rq06 may do")


def _pivot(rows: pd.DataFrame, axis: str, levels: tuple) -> pd.DataFrame | None:
    piv = rows.pivot_table(index=["size", "task"], columns=axis, values="primary_score")
    if not set(levels) <= set(piv.columns):
        return None
    return piv.dropna(subset=list(levels))


# --- 1. intervention decision accuracy ---------------------------------------

def seed_sd(fin: pd.DataFrame) -> pd.Series:
    """Per task, the seed standard deviation of ONE run's final score: the
    median over the baseline (deep, data A) (size, L) cells with replicates
    (3 seeds where they exist, so the estimate itself is coarse)."""
    base = fin[(fin["ladder"] == "deep") & (fin["data"] == "A")]
    sd = base.groupby(["size", "L", "task"])["primary_score"].agg(["std", "count"])
    return sd[sd["count"] >= 2]["std"].groupby("task").median()


def intervention_da(df: pd.DataFrame, fracs: list = FRACS, mask: pd.DataFrame | None = None,
                    populations: tuple = POPULATIONS) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(the decision table, the per-item agreement behind it, the same for
    every language's BPB with its `group`). The second frame has one row per
    (intervention, L, proxy size, fraction, task) of CELL_POPULATIONS with
    its `population` and `agree` in {0, 1}: what the per-benchmark and
    per-language tables and the decision table's DA of those populations
    aggregate; the third, filled only for `bpb_all` (rq06's call), is what
    rq06's transfer lines aggregate. A benchmark
    item counts at a proxy size only where the task is above chance there and
    at the cell's reference (rule 1, `mask`)."""
    sd = seed_sd(finals(df))
    grid = df[df["seed"] == GRID_SEED]
    fin = finals(grid)
    at_f = {f: at_fraction(grid, f) for f in fracs}
    rows, items_rows, group_rows = [], [], []
    for key, (label, axis, levels, held) in INTERVENTIONS.items():
        sub_fin = at_baseline(fin, held)
        for L in sorted(sub_fin["L"].unique()):
            mono_axis(sub_fin[sub_fin["L"] == L], axis, levels)      # raises unless rq02's mono-axis pair
            for pop in populations:
                min_items = 1 if pop in SINGLE else MIN_ITEMS
                ref_piv = _pivot(_population(sub_fin[sub_fin["L"] == L], pop, int(L), levels, axis), axis, levels)
                if ref_piv is None:
                    continue
                # rule 9: the reference is the TARGET_SIZE final, never the
                # largest size both levels happen to reach on a partly trained ladder
                sizes = size_order(ref_piv.index.get_level_values("size"))
                if TARGET_SIZE not in sizes:
                    continue
                sizes = sizes[:sizes.index(TARGET_SIZE) + 1]
                if len(sizes) < 2:
                    continue
                ref = TARGET_SIZE
                r = ref_piv.xs(ref, level="size")
                d_ref = r[levels[0]] - r[levels[1]]
                ref_sign = np.sign(d_ref)[np.sign(d_ref) != 0]      # items the reference decides
                # sqrt(2): d_ref is a difference of two runs, each with sd `sd`
                over_sd = d_ref.abs() / (np.sqrt(2) * d_ref.index.map(sd).to_numpy(float))   # NaN where no seed replicates
                decided = ref_sign.index[over_sd.reindex(ref_sign.index) >= DECIDED]
                for f in fracs:
                    sub = at_baseline(at_f[f], held)
                    sub = sub[sub["L"] == L]
                    piv = _pivot(_population(sub, pop, int(L), levels, axis), axis, levels)
                    if piv is None:
                        continue
                    for s in sizes:
                        if (s == ref and f == 1.0) or s not in piv.index.get_level_values("size"):
                            continue
                        p = piv.xs(s, level="size")
                        items = p.index.intersection(ref_sign.index)
                        if pop == "benchmark":
                            items = items[passes_gate(mask, items, s, ref).to_numpy()]
                        if len(items) < min_items:
                            continue
                        d_proxy = p.loc[items, levels[0]] - p.loc[items, levels[1]]
                        agree = (np.sign(d_proxy) == ref_sign.loc[items]).to_numpy(float)
                        if pop in CELL_POPULATIONS:
                            items_rows.append(pd.DataFrame({
                                "intervention": key, "label": label, "population": pop, "L": int(L), "proxy_size": s,
                                "frac": f, "task": items, "agree": agree}))
                        if pop == "bpb_all":            # every language, grouped by what the cell's lists train
                            group_rows.append(pd.DataFrame({
                                "intervention": key, "label": label, "L": int(L), "proxy_size": s, "frac": f, "reference_size": ref,
                                "group": language_group(items, int(L), _builds(sub, axis, levels)), "agree": agree}))
                        dec = items.intersection(decided)
                        row = {"intervention": key, "label": label, "population": pop, "L": int(L),
                               "proxy_size": s, "frac": f, "reference_size": ref, "n_items": int(len(items)),
                               "decision_acc": float(agree.mean()), "n_decided": int(len(dec)),
                               "decision_acc_decided": float((np.sign(d_proxy.loc[dec]) == ref_sign.loc[dec]).mean())
                               if len(dec) >= min_items else np.nan}
                        if f == 1.0:
                            # the first level wins an item when its score is
                            # higher on a benchmark, lower on any BPB or the loss
                            first = pd.Series(np.where(items.map(lower_is_better), d_ref.loc[items] < 0, d_ref.loc[items] > 0))
                            row.update({"mean_abs_delta_proxy": float(d_proxy.abs().mean()),
                                        "mean_abs_delta_ref": float(d_ref.loc[items].abs().mean()),
                                        "reference_prefers": levels[0] if first.mean() > 0.5 else levels[1]})
                        rows.append(row)
    return (pd.DataFrame(rows), pd.concat(items_rows, ignore_index=True) if items_rows else pd.DataFrame(),
            pd.concat(group_rows, ignore_index=True) if group_rows else pd.DataFrame())


LANGUAGE_GROUPS = ("trained by both levels", "trained by one level", "script trained", "script not trained")


def language_group(tasks, L: int, builds: list) -> list[str]:
    """Per `bpb_<subset>` task, what the lists of the two levels (their data
    `builds` at this L) do with the language: both train it, only one does (then the
    decision is mostly "prefer the model that saw it"), neither does but a
    list trains its script, or neither trains even the script. The last two
    are the transfer test proper."""
    lists = [trained_bpb_tasks(L, d) or set() for d in builds]
    both, either = set.intersection(*lists), set.union(*lists)
    scripts = {t.rsplit("_", 1)[-1] for t in either} | {"Latn"}          # bpb_dclm is English
    return [LANGUAGE_GROUPS[0] if t in both else LANGUAGE_GROUPS[1] if t in either
            else LANGUAGE_GROUPS[2] if t.rsplit("_", 1)[-1] in scripts else LANGUAGE_GROUPS[3] for t in tasks]


def effect_at_reference(fin: pd.DataFrame, mask: pd.DataFrame | None) -> pd.DataFrame:
    """Per (intervention, L, population): at the reference size, the median
    |Δ| in per-task seed standard deviations (the seed sd of the baseline
    cells, median over the (size, L) cells with replicates). Benchmarks count
    where they are above chance at the reference (rule 1): a task at chance
    there has no decision to read, and its |Δ| is noise."""
    sd = seed_sd(fin)
    grid = fin[fin["seed"] == GRID_SEED]
    rows = []
    for key, (label, axis, levels, held) in INTERVENTIONS.items():
        sub = at_baseline(grid, held)
        for L, g in sub.groupby("L"):
            piv = _pivot(g, axis, levels)
            if piv is None:
                continue
            tasks = piv.index.get_level_values("task")
            is_bpb = tasks.str.startswith("bpb_")
            for pop, sel in (("bits per byte", is_bpb & (tasks != "bpb_macro")), ("benchmarks", ~is_bpb & (tasks != "train_loss"))):
                pp = piv[sel]
                counts = pp.groupby(level="size").size()
                sizes = size_order(counts[counts >= MIN_ITEMS].index)
                if TARGET_SIZE not in sizes:       # rule 9, as above
                    continue
                ref = TARGET_SIZE
                p = pp.xs(ref, level="size")
                if pop == "benchmarks":
                    p = p[passes_gate(mask, p.index.get_level_values("task"), ref).to_numpy()]
                ratio = ((p[levels[0]] - p[levels[1]]).abs() / p.index.map(sd)).replace(np.inf, np.nan).dropna()
                if len(ratio) >= MIN_ITEMS:
                    rows.append({"intervention": key, "label": label, "L": int(L), "reference_size": ref,
                                 "population": pop, "median_effect_over_seed_sd": float(ratio.median()),
                                 "share_above_2": float((ratio > 2).mean()), "n": int(len(ratio))})
    return pd.DataFrame(rows)


# --- figures ----------------------------------------------------------------

def not_yet(ax, text: str = f"not yet: no {TARGET_SIZE} result") -> None:
    """An empty panel that says why it is empty (a reference still in flight)."""
    ax.text(.5, .5, text, ha="center", va="center", fontsize=7.5, color=S.MUTED, transform=ax.transAxes)
    ax.set_xticks([]); ax.set_yticks([]); S.clean(ax, spines=())


def plot_da_grid(da: pd.DataFrame, path: Path, pending: pd.DataFrame | None = None) -> None:
    da = da[da["frac"] == 1.0]
    pops = [p for p in ("bpb_trained", "benchmark", "bpb_all") if p in set(da["population"])]
    pending = pending if pending is not None else pd.DataFrame(columns=["intervention", "population", "L"])
    waiting = set(zip(pending["intervention"], pending["population"]))
    keys = [k for k in INTERVENTIONS if k in set(da["intervention"]) | {k for k, _ in waiting}]
    if not pops or not keys:
        return
    fig, axes = plt.subplots(len(keys), len(pops), figsize=(4.2 * len(pops), 3.0 * len(keys)), squeeze=False)
    im = None
    for i, k in enumerate(keys):
        for j, pop in enumerate(pops):
            ax = axes[i][j]
            sub = da[(da["intervention"] == k) & (da["population"] == pop)]
            if sub.empty:
                if (k, pop) in waiting:
                    not_yet(ax); ax.set_title(f"{INTERVENTIONS[k][0]} — {pop}", loc="left", fontsize=8)
                else:
                    ax.set_visible(False)
                continue
            late = set(pending.loc[(pending["intervention"] == k) & (pending["population"] == pop), "L"])
            Ls = sorted(set(sub["L"]) | late)
            sizes = size_order(sub["proxy_size"])
            mat = np.full((len(sizes), len(Ls)), np.nan)
            for _, r in sub.iterrows():
                mat[sizes.index(r["proxy_size"]), Ls.index(r["L"])] = r["decision_acc"]
            im = ax.imshow(mat, vmin=0, vmax=1, cmap=S.SEQ, aspect="auto")
            for L in late:                          # white column: the reference has not landed
                ax.text(Ls.index(L), (len(sizes) - 1) / 2, f"not yet\n(no {TARGET_SIZE})", ha="center", va="center",
                        fontsize=6, color=S.MUTED)
            for a in range(len(sizes)):
                for b in range(len(Ls)):
                    if np.isfinite(mat[a, b]):
                        n = int(sub[(sub["proxy_size"] == sizes[a]) & (sub["L"] == Ls[b])]["n_items"].iloc[0])
                        ax.text(b, a, f"{mat[a, b]:.2f}\n(n={n})", ha="center", va="center",
                                fontsize=6, color="white" if mat[a, b] > 0.7 else S.INK)
            ax.set_xticks(range(len(Ls))); ax.set_xticklabels([f"L{L}" for L in Ls], fontsize=7)
            ax.set_yticks(range(len(sizes))); ax.set_yticklabels(sizes, fontsize=7)
            ax.set_title(f"{INTERVENTIONS[k][0]} — {pop}", loc="left", fontsize=8)
            ax.set_xlabel("languages"); ax.set_ylabel("proxy size"); S.clean(ax, spines=()); ax.tick_params(length=0)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="decision accuracy (proxy vs reference, final checkpoints)",
                     fraction=0.02)
    fig.suptitle("Does a proxy size rank the intervention like the reference size at that L?", y=1.0)
    S.save(fig, path, dpi=140)


def plot_interventions(ev: pd.DataFrame, dag: pd.DataFrame, out_dir: Path, pending: pd.DataFrame | None = None) -> None:
    """The paper's RQ4 figure: (a) |effect| at the reference in seed sds per
    intervention; (b) final-checkpoint agreement by proxy size per intervention.
    Both average over L, so both read `by_recipe` keys: a scheme decision per build."""
    late = set() if pending is None else set(zip(pending["intervention"], pending["population"]))
    keys = [k for k in RECIPES if k in set(ev["intervention"]) | set(dag["intervention"]) | {k for k, _ in late}]
    names = [RECIPES[k][0] for k in keys]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(9.6, 3.7), gridspec_kw={"width_ratios": [1, 1.15]})
    y = np.arange(len(keys))
    for j, (pop, cell_pop, col) in enumerate((("bits per byte", "bpb_trained", S.RAMP[3]), ("benchmarks", "benchmark", S.SERIES[1]))):
        g = (ev[ev["population"] == pop].groupby("intervention")["median_effect_over_seed_sd"].median()
             .reindex(keys))
        a0.barh(y + (j - .5) * .36, g.values, .34, color=col, label=pop)
        for k, yi, val in zip(keys, y, g.values):
            if np.isfinite(val):
                a0.annotate(f"{val:.1f}×", (val, yi + (j - .5) * .36), xytext=(4, 0), va="center",
                            textcoords="offset points", fontsize=7, color=S.INK)
            elif (k, cell_pop) in late:          # the reference has not landed: say so, not a zero
                a0.annotate(f"{pop}: not yet (no {TARGET_SIZE})", (1, yi + (j - .5) * .36), xytext=(4, 0), va="center",
                            textcoords="offset points", fontsize=6.5, color=S.MUTED)
    a0.axvline(1, color=S.MUTED, ls="--", lw=1); a0.axvline(2, color=S.GRID, lw=.8)
    a0.set_yticks(y); a0.set_yticklabels(names, fontsize=7.5); a0.set_ylim(len(keys) - .4, -.6)   # inverted, room for the "not yet" notes
    a0.set_xscale("log"); a0.set_xlabel("|effect| at the reference, in seed standard deviations (median)")
    a0.set_title("(a) is there a decision to make?", loc="left")
    a0.legend(frameon=False, loc="upper right"); a0.grid(axis="x", color=S.GRID, lw=.6); a0.set_axisbelow(True)
    S.clean(a0); a0.tick_params(length=0)
    styles = {"bpb_trained": "-", "benchmark": "--"}       # the marker is the recipe's (RECIPES)
    order = size_order(dag["proxy_size"])
    plotted = []                     # (a) says "not yet" for a pending key; (b) lists only the lines it draws
    for k in keys:
        for pop, ls in styles.items():
            g = dag[(dag["intervention"] == k) & (dag["population"] == pop)]
            if g.empty:
                continue
            g = g.set_index("proxy_size").reindex(order)
            plotted += [k] if k not in plotted else []
            a1.plot(range(len(order)), g["decision_acc"], ls=ls, marker=RECIPES[k][2], ms=4.5, lw=1.6, color=RECIPES[k][1])
    a1.axhline(.5, color=S.MUTED, ls=":", lw=1)
    a1.set_xticks(range(len(order))); a1.set_xticklabels(order)
    a1.set_ylim(0, 1.05); a1.set_xlabel("proxy size"); a1.set_ylabel("agreement with the reference (mean over L)")
    a1.set_title("(b) does the proxy agree, per intervention", loc="left")
    h = [Line2D([], [], color=RECIPES[k][1], marker=RECIPES[k][2], ms=4, lw=1.6, label=RECIPES[k][0]) for k in plotted]
    h += [Line2D([], [], color=S.INK, ls="-", label="per-language bits per byte"),
          Line2D([], [], color=S.INK, ls="--", label="benchmark tasks")]
    a1.legend(handles=h, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), ncol=1, fontsize=6.8)   # one entry per recipe: outside the lines
    a1.grid(color=S.GRID, lw=.6); a1.set_axisbelow(True); S.clean(a1)
    fig.subplots_adjust(wspace=.5)
    ev.to_csv(out_dir / "rq4_interventions.csv", index=False)   # rule 12: the table the paper figure draws
    S.save_figure(fig, out_dir, "rq4_interventions")


# --- README ------------------------------------------------------------------

def generate_readme(pool: str, out_dir: Path, da: pd.DataFrame, ev: pd.DataFrame,
                    dag: pd.DataFrame, pending: pd.DataFrame) -> None:
    if pool != CANONICAL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    rel = f"{stage}/{pool}"
    bullets, blocks = [], []
    fin = da[da["frac"] == 1.0] if not da.empty else da
    if not fin.empty:
        for pop, title in (("bpb_trained", "per-language BPB"), ("benchmark", "benchmarks")):
            for k in INTERVENTIONS:
                core = fin[(fin["intervention"] == k) & (fin["population"] == pop)]
                late = sorted(pending.loc[(pending["intervention"] == k) & (pending["population"] == pop), "L"])
                if core.empty:
                    if late:          # rule 9: the reference has not landed, the cell is empty, never filled
                        blocks.append(f"**{INTERVENTIONS[k][0]}, {title}**: not yet, since there is no {TARGET_SIZE} "
                                      "result at both levels (" + ", ".join(f"L{L}" for L in late) + ").")
                    continue
                grid = core.pivot_table(index="proxy_size", columns="L", values="decision_acc")
                grid = grid.reindex(size_order(grid.index))
                # the smallest proxy from which every larger proxy stays at >= RELIABLE_DA (G.smallest_safe), not the
                # first that touches it: at L8 depth 90M reads 1.00 but 350M 0.25 and 600M 0.00
                safe = G.smallest_safe((grid >= RELIABLE_DA).astype(float).where(grid.notna()).T)
                first = {L: grid.index[int(i)] if i == i and i >= 0 else "—" for L, i in safe.items()}
                if pop == "bpb_trained":
                    bullets.append(f"- **{INTERVENTIONS[k][0][:1].upper()}{INTERVENTIONS[k][0][1:]} on {title}.** Smallest proxy from which every larger proxy keeps DA ≥ {RELIABLE_DA:g} "
                                   "against the reference: " + ", ".join(f"L{L}: {s}" for L, s in first.items())
                                   + "".join(f", L{L}: not yet" for L in late) + ".")
                cols = sorted(set(grid.columns) | set(late))
                rows = [[s] + [fmt(grid.loc[s, L]) if L in grid.columns else "not yet" for L in cols] for s in grid.index]
                refs = ", ".join(f"L{L} → {r}" for L, r in core.groupby("L")["reference_size"].first().items())
                blocks += [f"**{INTERVENTIONS[k][0]}, {title}** (rows: proxy size, columns: L, reference {refs}"
                           + (f", not yet = no {TARGET_SIZE} result at both levels" if late else "") + "):",
                           md_table(["proxy"] + [f"L{L}" for L in cols], rows)]
        bench = fin[(fin["intervention"] == "arch") & (fin["population"] == "benchmark")]
        if not bench.empty:
            m = bench.groupby("proxy_size")["decision_acc"].mean()
            bullets.append("- **Depth decision on benchmarks.** Mean DA over L by proxy: "
                           + ", ".join(f"{s} {fmt(m[s])}" for s in size_order(m.index)) + "."
                           + size_match_note([("arch", s) for s in m.index]))
        blocks.append(f"![Intervention DA grid]({rel}/intervention_da_all_mono_axis.png)")
    if not ev.empty:
        med = (by_recipe(ev).groupby(["intervention", "population"])["median_effect_over_seed_sd"].median()
               .unstack("population"))
        med = med.reindex([k for k in RECIPES if k in med.index])
        bullets.append("- **Is there a decision to make?** Median |Δ| at the reference, in seed sds. "
                       + ". ".join(f"{RECIPES[k][0][:1].upper()}{RECIPES[k][0][1:]}: " + ", ".join(f"{p} {fmt(v, 1)}×" for p, v in r.items() if np.isfinite(v))
                                   for k, r in med.iterrows()) + ".")
        blocks += ["**Effect at the reference in seed standard deviations** (median over items and L, one scheme "
                   "decision per recipe, since a letter names a different build at each L):",
                   md_table(["intervention"] + list(med.columns),
                            [[RECIPES[k][0]] + [fmt(med.loc[k, c], 1) for c in med.columns] for k in med.index]),
                   f"![Interventions]({rel}/rq4_interventions.png)"]
    if not pending.empty:
        bullets.append("- **Pending the 1.7B results.**" + pending_note(pending))
    readme = OUT_ROOT / "README.md"
    gen = f"analyze.py --pool {pool}"
    replace_block(readme, "highlight", "## Highlighted result\n\n" + "\n".join(bullets), gen)
    replace_block(readme, "results", "## Results\n\n"
                  + f"Numbers from the `{pool}` pool. Regenerate with "
                  f"`python analysis/rq05_design_decisions/analyze.py --pool {pool}`.\n\n"
                  + "\n\n".join(blocks), gen)
    print(f"Wrote auto README blocks → {readme}")


# --- driver -------------------------------------------------------------------

def main(pool: str, out_dir: Path) -> None:
    df = ladder_frame(pool)
    fin = finals(df)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pool '{pool}': {df['model'].nunique()} cells, {fin['task'].nunique()} tasks, "
          f"seeds {sorted(df['seed'].unique())}, data builds {sorted(df['data'].unique())}")

    da, items, _ = intervention_da(df, mask=gate_mask(pool))
    pending = pending_references(fin)
    for r in pending.itertuples():
        print(f"!!! RULE 9: {r.label}, {r.population}, L{r.L}: no {TARGET_SIZE} result at both levels yet "
              f"(proxies {r.proxies or 'none'}); its DA-size and effect cells stay empty")
    if not da.empty:                     # the 600M depth pairs are not size-matched: kept, flagged
        da["size_matched"] = [size_matched(k, s) for k, s in zip(da["intervention"], da["proxy_size"])]
    da.to_csv(out_dir / "intervention_da_all_mono_axis.csv", index=False)
    if not items.empty:
        # the same agreement, per benchmark and per language (panels.py draws them);
        # add_meta drops the items with no single language (aggregates, subject facets)
        items = G.add_meta(items)
        keys = ["intervention", "label", "L", "proxy_size", "frac"]
        for by, name in (("family", "benchmark"), ("language", "language")):
            (items.groupby(keys + [by]).agg(decision_acc=("agree", "mean"), n_items=("agree", "size")).reset_index()
             .to_csv(out_dir / f"intervention_da_size_by_{name}_mono_axis.csv", index=False))
    print(f"Wrote → {out_dir / 'intervention_da_all_mono_axis.csv'} ({len(da)} cells)")
    dag = pd.DataFrame()
    if not da.empty:
        plot_da_grid(da, out_dir / "intervention_da_all_mono_axis.png", pending)
        rec = by_recipe(da)                  # averaged over L: a scheme decision per recipe
        dag = (rec[rec["frac"] == 1.0].groupby(["intervention", "label", "population", "proxy_size"])
               .agg(decision_acc=("decision_acc", "mean"), cells=("decision_acc", "size"),
                    refs=("reference_size", lambda s: ",".join(sorted(set(s))))).reset_index())
        dag["size_matched"] = [size_matched(k, s) for k, s in zip(dag["intervention"], dag["proxy_size"])]
        dag.to_csv(out_dir / "rq4_da_size_by_intervention_mono_axis.csv", index=False)

    ev = effect_at_reference(fin, gate_mask(pool))
    ev.to_csv(out_dir / "rq4_effect_vs_seed.csv", index=False)
    if not ev.empty or not dag.empty:
        plot_interventions(by_recipe(ev), dag, out_dir, pending)

    (out_dir / "facts.json").write_text(json.dumps(
        {"rq4": {"effect": ev.round(3).to_dict("records"), "da": dag.round(3).to_dict("records"),
                 "pending_reference": pending.to_dict("records")}},
        indent=1, default=str))
    generate_readme(pool, out_dir, da, ev, dag, pending)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL,
                   help=f"Ladder pool from configs/models.json (default: {CANONICAL}; every seed and "
                        "data build is needed for the six interventions and the seed-noise column).")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    main(args.pool, OUT_ROOT / stage / args.pool)
