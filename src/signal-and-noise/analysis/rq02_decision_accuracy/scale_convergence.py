"""RQ2 — Scale convergence: how small may a FULLY TRAINED model be and still
make the reference's decisions?

`early_small.py` and `by_L.py` read a proxy early AND small; this script holds
the size axis still at each cell's final checkpoint and asks the one question
the paper's "minimum useful scale" claim needs:

    R_size(N) = (# decisions at size N matching the reference's)
                / (# comparable decisions)

    N_min(tau) = min { N : R_size(N) >= tau }        (tau configurable, 0.90)

This is rq02's DA-size — the same kernel, the same gate, the same pair
minimum — pooled over decisions instead of averaged over tasks, so the
numerator and denominator are decision counts the CSV can carry. The
reference has R = 1.0 by construction (it is compared with itself) and its
pair count is still real, so a thin reference is visible rather than implied.

Three groupings of the same decisions, `--by`:

    overall          no grouping at all: the single pooled line, which is the
                     plain figure the question is written as. Every pair at the
                     grid seed, one point per trained size.
    L                the language-count regimes (L1 … L50) plus `all pairs`:
                     within a regime a decision is any pair of design variants
                     that share the L, as `by_L.py` defines it. Under
                     `--axes mono-axis` a regime keeps only its pairs that move
                     ONE other axis, and rule 5 then empties every proxy size
                     but 1B (4-5 pairs per regime, three needed per task): the
                     `_one_axis` L figure shows that fact rather than hiding it.
                     The multi-axis L lines therefore pool arch, list and
                     temperature decisions at once, and their mix is written
                     next to every point (`share_*` columns) because it differs
                     between regimes and cannot be held fixed.
    transformation   the axis the pair differs on, and nothing else: language
                     count, depth, language list (A vs B), temperature (T = 1 vs
                     T = 3), second language (ru vs zh vs es at L = 2), seed. The
                     `scheme` token is unpacked into those three design choices
                     first — AT3 is list A at T=3, ZH is list A with Chinese in
                     the second slot — so a pair differing on two of the axes is
                     a decision about neither and is dropped. This is rq05's
                     intervention set read as a scale-convergence curve; if the
                     two RQs merge, this grouping is the piece that moves.

Every grouping also carries `OVERALL`, the pooled black line: every pair at the
grid seed, every data scheme included (A, B, AT3, ZH, ES), so it is the same
population as by_L's first panel and means the same thing in every rq02 figure.
It carries a leave-one-family-out jackknife band (`lo`/`hi`, 90 %): the unit
resampled is the design variant, since pairs and tasks both share them.

Two restrictions make the L lines comparable with each other, which by
default they are not (an L50 line pools 52 tasks, an L8 line 7):

    --langs L8       only tasks in the eight languages of the L8 setting
                     ({en, ru, zh, de, ja, es, fr, it}), which every L >= 8
                     trains — the lists are nested. Stem token `L8`.
    --common-tasks   only tasks with >= MIN_PAIRS pairs in EVERY drawn regime at
                     EVERY proxy size: one task set for the whole figure, so a
                     gap between two lines is a gap on the same benchmarks.
                     Stem token `common`. What it cannot fix is the decision mix.

    scale_convergence.png        the pooled line alone (`--by overall`)
    scale_convergence_<by>.png   one line per group, x = non-embedding parameters
                                 (log), y = R_size, dotted line at tau, the
                                 smallest size clearing it ringed and labelled;
                                 left panel benchmarks, right panel per-language
                                 BPB, identical axes
    scale_convergence[_<by>]_above_66_*_flops.png   the same decisions on a COMPUTE
                                 axis: each of the ten evaluated checkpoints of every
                                 proxy run against the same reference final, so a line
                                 carries ten points per size and x is the FLOPs spent.
                                 Only the reference's own final is 1.0 by construction;
                                 its earlier checkpoints are ordinary proxies. Drawn for
                                 the reliable populations only.
    scale_convergence_<by>_above_80.png   the same, restricted to the (benchmark,
                                 language) cells that rank reliably on both axes
                                 (`reliable_tasks.py`). One panel, not two: the
                                 reliable set is benchmarks by construction, so a
                                 BPB panel would be empty rather than informative.
    scale_convergence[_<by>][_<variant>][_flops].csv   one row per (population,
                                 group, size[, frac]): population (`all benchmarks` /
                                 `bpb`), group (the line), size, frac, n_matching,
                                 n_comparable, reliability (the pooled ratio),
                                 reliability_macro (rq02's mean over tasks, carried so
                                 the two conventions can be compared), n_tasks,
                                 median_pairs, compute, non_emb, reference_size, tau,
                                 reaches_tau, n_min_size (and n_min_compute on the
                                 compute axis). n_min_* is NA when no real proxy
                                 clears tau.

    python analysis/rq02_decision_accuracy/scale_convergence.py --by L --pool predictivity
    python analysis/rq02_decision_accuracy/scale_convergence.py --by L --langs L8 --common-tasks
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
from analysis.rq02_decision_accuracy.reliable_tasks import FILTERS, load_reliable  # noqa: E402
from analysis.utils import (  # noqa: E402
    AXES_SUFFIX, CKPT_DA_EARLY_FRACS, DESIGN_AXES, GRID_SEED, MIN_PAIRS, NON_EMB, TARGET_SIZE,
    assign_language, at_fraction, design_axes, finals, jackknife_ratio, ladder_frame, pair_sets,
    size_order)
from pretrain.launch_trainings import cell_languages  # noqa: E402

OUT_ROOT = DECISION_ACCURACY
POOL = "predictivity_all"     # the transformation axes need every scheme and seed
TAU = 0.90                    # "recovers 90 % of the full-scale decisions"
BY_LINE = {"overall": "one pooled line over every pair at the grid seed",
           "L": "one line per language-count regime (pairs sharing the L; any other axis may move)",
           "transformation": "one line per design axis the pair differs on"}
# The task restrictions a run can ask for, by name: the language set of an L
# setting (scheme A's, the nested lists), or none.
LANG_SETS = {"all": None, "L8": frozenset(cell_languages(8, "A"))}
# The pair's one differing axis -> its label, one label per axis. `scheme` is
# NOT an axis: it glues the language LIST, the sampling TEMPERATURE and the
# SECOND LANGUAGE together, so an A-vs-AT3 pair moves the temperature alone
# while a B-vs-AT3 pair moves the list AND the temperature and is a decision
# about neither. That split, and the pair sets built on it, live in
# `analysis.utils` (DESIGN_AXES, design_axes, pair_sets) because a rule belongs
# in the shared layer, not in one figure — `compute_da` reads the same
# decomposition for its `axes` column (rule 15). Before the split, a pair whose
# level combination had no label was dropped silently, which is what hid
# B-vs-AT3 and ZH-vs-ES; now B-vs-BT3 joins the temperature axis and AT3-vs-BT3
# the list axis with no code change, once BT3 trains.
AXIS_LABEL = {"L": "language count", "arch": "depth (deep vs shallow)",
              "list": "language list (A vs B)", "T": "temperature (T=1 vs T=3)",
              "lang2": "2nd language (ru vs zh vs es)",
              "en": "English corpus (edu filter on vs off)", "seed": "seed"}
KEYS = DESIGN_AXES
# The pooled line every grouping carries, drawn in black: every pair at the grid
# seed, cross-L and cross-scheme included — the same population as by_L's first
# panel, so the black line means one thing in every rq02 figure. The scheme is NOT
# held to A and B: AT3 (temperature), ZH and ES (the second language) are design
# decisions like the depth or the language count, and the per-regime lines have
# always included them, so restricting the pooled line alone made it a different
# population from the lines drawn beside it. Today this is the 37 pairs the two
# L50 AT3 families reach the reference with; ZH and ES contribute nothing yet
# because their 1.7B cells are not in the report (rule 9), and will contribute
# without a code change once they land.
OVERALL = "all pairs"
# OVERALL is drawn in ink and takes NO palette slot, so a group colours the same
# way whether or not the pooled line is present. That is what kept the black line
# readable: before, OVERALL ate the first slot under `--by transformation` and the
# darkest ramp step landed on L8 in one figure and on the language lists in the
# other, so two different near-black curves each looked like `all pairs`. With
# OVERALL out of the cycle only `--by L` reaches the dark step, on L8.
GROUP_COLOURS = S.RAMP + [S.SERIES[1], S.SERIES[2], "#8c1d18", "#7a5195"]
# The L regimes are ORDINAL, so they take the ramp in L order — light at L8,
# dark at L50 — the same colour in every figure that draws them, whichever
# regimes happen to have a line. L1 and L2 draw no line today (rule 5) and sit
# outside the ramp so that, when they do, they do not shift the other four.
L_COLOUR = {"L1": "#8c1d18", "L2": "#7a5195", "L8": S.RAMP[0], "L15": S.RAMP[1], "L30": S.RAMP[2], "L50": S.RAMP[3]}


def group_order(groups) -> list:
    """L regimes by their number, anything else as it comes."""
    return sorted(groups, key=lambda g: (0, int(g[1:])) if g[1:].isdigit() and g.startswith("L") else (1, 0))
POPULATIONS = ("all benchmarks", "bpb")
# every population, then one per named filter this figure is asked for.
# `above_66_size` exists because rq2_above_66_one reads its DA-size panel over
# the tasks whose DA-size is reliable, not over the `both` intersection.
VARIANTS = ("", "above_80", "above_66_both", "above_66_size")
# The ten evaluated checkpoints of every run, 0.5C ... 5C (rule 3).
FRACS = [*CKPT_DA_EARLY_FRACS, 1.0]
# The same decisions on a COMPUTE axis: every one of those checkpoints against the
# same reference final, so a line has ten points per size instead of one and the x
# axis is the FLOPs actually spent. Drawn only for the reliable populations, where
# the curve is worth reading; `_flops` is appended to the stem.
FLOPS_VARIANTS = ("above_66_both", "above_66_size")
mpl.rcParams.update(S.RC)


def pairs_by_group(attrs: pd.DataFrame, by: str, axes: str = "multi-axis") -> dict[str, list[tuple[str, str]]]:
    """group -> the unordered family pairs that are a decision in it.

    Every grouping carries `OVERALL`, the pooled line: every pair at the grid
    seed, cross-L and cross-scheme included — by_L's first panel, so it is the
    pooled ranking and not the mean of the groups, and it is the same population
    in every rq02 figure. Under `by="L"` a
    regime's own line additionally holds the seed at the grid seed, as
    `by_L.py` does: a replicate pair is a decision about the seed, not about
    the language count, and letting it into an L regime would mix two axes.
    """
    fams = sorted(attrs.index)
    # The pooled line IS `utils.pair_sets`'s set for this `axes`, so the black
    # line of every rq02 figure and the matching rows of `da_per_task.csv` are
    # one population by construction rather than by two definitions agreeing.
    sets = pair_sets(attrs)
    groups: dict[str, list] = {OVERALL: list(sets[axes])}
    if by == "overall":                       # the pooled line is the whole figure
        return groups
    allowed = set(sets[axes])                 # an L regime keeps the pair set it is named for
    for i, a in enumerate(fams):
        for b in fams[i + 1:]:
            ra, rb = attrs.loc[a], attrs.loc[b]
            if by == "L":
                if ra["L"] == rb["L"] and ra["seed"] == rb["seed"] == GRID_SEED and (a, b) in allowed:
                    groups.setdefault(f"L{int(ra['L'])}", []).append((a, b))
                continue
            differ = [k for k in KEYS if ra[k] != rb[k]]
            if len(differ) != 1:              # two axes at once decides neither
                continue
            axis = differ[0]
            groups.setdefault(AXIS_LABEL[axis], []).append((a, b))
    return groups


def pair_axis(attrs: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(family_a, family_b) -> the ONE design axis the pair moves, or `multi`
    when it moves several: the decision mix a pooled line is made of."""
    fams = sorted(attrs.index)
    out = {}
    for i, a in enumerate(fams):
        for b in fams[i + 1:]:
            differ = [k for k in KEYS if attrs.loc[a, k] != attrs.loc[b, k]]
            out[(a, b)] = differ[0] if len(differ) == 1 else "multi"
    return out


def grid_frame(df: pd.DataFrame, fracs: list) -> pd.DataFrame:
    """The pool at each evaluated tenth of every run, `frac` snapped to the grid
    point the row was taken for, so a fraction is an exact dictionary key."""
    return pd.concat([at_fraction(df, f).assign(frac=f) for f in fracs], ignore_index=True)


def stem_for(by: str, variant: str = "", axes: str = "multi-axis", langs: str = "all",
             common: bool = False) -> str:
    """`overall` is the plain figure, so it carries no `_by` token; a language
    restriction replaces the `L` token (`scale_convergence_L8_...`), the
    common-task set appends `common` to it."""
    token = (langs if langs != "all" else by) + ("common" if common else "")
    return ("scale_convergence" + (f"_{token}" if by != "overall" or langs != "all" else "")
            + (f"_{variant}" if variant else "") + AXES_SUFFIX[axes])


DECISION_COLS = ["task", "group", "size", "frac", "family_a", "family_b", "match", "compute"]


def decisions(df: pd.DataFrame, groups: dict, sizes: list, fin: pd.DataFrame,
              fracs: list = (1.0,), ref: str = TARGET_SIZE) -> pd.DataFrame:
    """One row per comparable decision: (task, group, size, frac, the two
    families, whether the proxy ordered them like the reference's FINAL, and
    the mean compute the two proxies spent). `reliability` sums these per cell;
    the jackknife and the decision-mix columns need the rows themselves.

    A decision is comparable when both families have a score at the proxy
    (size, frac) AND a final score at the reference. Agreement is the sign of
    the score difference on both sides — `snr.metrics.decision_acc_fast`'s tie
    convention (tied in both agrees, tied in one is a miss, order-invariant),
    applied to an explicit pair list instead of every pair of a vector.

    `df` carries a `frac` snapped to the evaluated grid and `fin` is the frame
    the reference is read from, the same finals the rest of rq02 uses. The
    default `fracs=(1.0,)` over `fin` itself is the SIZE axis, one point per
    size; `FRACS` over `grid_frame` is the COMPUTE axis, ten points per size
    read against that same reference.
    """
    rows = []
    ref_fin = fin[fin["size"] == ref]
    by_task = {t: dict(zip(g["family"], g["primary_score"])) for t, g in ref_fin.groupby("task", sort=False)}
    for task, g in df.groupby("task", sort=False):
        R = by_task.get(task)
        if not R:
            continue
        sc = dict(zip(zip(g["family"], g["size"], g["frac"]), g["primary_score"]))
        fl = dict(zip(zip(g["family"], g["size"], g["frac"]), g["compute"]))
        for grp, pl in groups.items():
            for size in sizes:
                for fr in fracs:
                    for a, b in pl:
                        ka, kb = (a, size, fr), (b, size, fr)
                        if ka in sc and kb in sc and a in R and b in R:
                            rows.append((task, grp, size, fr, a, b,
                                         int(np.sign(sc[ka] - sc[kb]) == np.sign(R[a] - R[b])),
                                         (fl[ka] + fl[kb]) / 2))
    out = pd.DataFrame(rows, columns=DECISION_COLS)
    for c in ("task", "group", "size", "family_a", "family_b"):
        out[c] = out[c].astype("category")          # ten fracs x every pair: keep the strings small
    return out


def reliability(dec: pd.DataFrame) -> pd.DataFrame:
    """Per (task, group, size, frac): the decisions of `decisions()` summed —
    how many matched, how many were comparable, the mean compute, the ratio."""
    out = (dec.groupby(["task", "group", "size", "frac"], observed=True)
           .agg(n_matching=("match", "sum"), n_comparable=("match", "size"), compute=("compute", "mean"))
           .reset_index())
    for c in ("task", "group", "size"):
        out[c] = out[c].astype(str)
    out["da"] = out["n_matching"] / out["n_comparable"]
    return out


def keep_cells(cells: pd.DataFrame, pool: str) -> pd.DataFrame:
    """The cells a figure may read: rule 5 (the pair minimum) and rule 1 (the
    above-random gate at the proxy and at the reference), with the task's
    family and language attached and the two populations named."""
    cells = cells[cells["n_comparable"] >= MIN_PAIRS]        # rule 5
    cells = G.mark_gated(G.add_meta(cells), pool, "size", "da", TARGET_SIZE).dropna(subset=["da"])
    cells["population"] = np.where(cells["family"] == "bpb", "bpb", "all benchmarks")
    return cells


def aggregate(cells: pd.DataFrame, pool: str, tau: float, x: str = "non_emb") -> pd.DataFrame:
    """The gate, the pair minimum, then one row per (population, group, size).

    `reliability` pools the decisions (the ratio the question is written as);
    `reliability_macro` is rq02's mean over tasks, carried next to it so the
    two conventions can be compared rather than silently swapped.
    """
    cells = keep_cells(cells, pool)
    keys = ["population", "group", "size"] + (["frac"] if x == "compute" else [])
    out = (cells.groupby(keys)
           .agg(n_matching=("n_matching", "sum"), n_comparable=("n_comparable", "sum"),
                reliability_macro=("da", "mean"), n_tasks=("task", "nunique"),
                median_pairs=("n_comparable", "median"), compute=("compute", "mean")).reset_index())
    if "frac" not in out:
        out["frac"] = 1.0             # the size axis IS the final checkpoint of each run
    out["reliability"] = out["n_matching"] / out["n_comparable"]
    out["non_emb"] = out["size"].map(NON_EMB)
    out["reference_size"] = TARGET_SIZE
    out["tau"] = tau
    out["reaches_tau"] = out["reliability"] >= tau
    # N_min(tau) per (population, group): the first REAL proxy along the x axis
    # that clears tau — the smallest size, or on the compute axis the least
    # compute (whose `n_min_size` is then that point's size, not a minimum size).
    # The reference is 1.0 by comparing itself with itself, so it is not evidence
    # of convergence and must not answer this: counting it made every group
    # "reach" tau at the reference, so the column could never say "never" and the
    # CSV disagreed with the figure, which rings real proxies only.
    real_pt = ~((out["size"] == TARGET_SIZE) & (out["frac"] == 1.0))
    hit = out[out["reaches_tau"] & real_pt].sort_values(x).groupby(["population", "group"])
    key = list(zip(out["population"], out["group"]))
    out["n_min_size"] = list(map(hit["size"].first().get, key))
    if x == "compute":
        out["n_min_compute"] = list(map(hit["compute"].first().get, key))
    return out.sort_values(["population", "group", x])


def decorate(out: pd.DataFrame, dec: pd.DataFrame, cells: pd.DataFrame, pool: str, keys: list,
             axis_of: dict | None = None) -> pd.DataFrame:
    """Per point of `out`: which axis the pooled decisions moved (`share_<axis>`,
    the mix a regime line is made of) and the leave-one-family-out band
    (`se`, `n_families`, `lo`, `hi`). Both are read from the decision rows
    behind exactly the cells `aggregate` kept, so they join on the cell key."""
    kept = keep_cells(cells, pool)
    d = dec.astype({c: str for c in ("task", "group", "size", "family_a", "family_b")})
    d = d.merge(kept[["task", "group", "size", "frac", "population"]], on=["task", "group", "size", "frac"])
    if axis_of is not None:
        d["axis"] = [axis_of.get((a, b), "multi") for a, b in zip(d["family_a"], d["family_b"])]
        mix = d.groupby(keys + ["axis"], observed=True).size().unstack("axis", fill_value=0)
        mix = mix.div(mix.sum(axis=1), axis=0).add_prefix("share_").reset_index()
        out = out.merge(mix, on=keys, how="left")
    return out.merge(jackknife_ratio(d, keys)[keys + ["se", "n_families", "lo", "hi"]], on=keys, how="left")


def diagnostics(out: pd.DataFrame, groups: dict, fin: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The tables to read before trusting the figure: which size is the
    reference for each group, that only finals are in play, that the reference
    row is exactly 1.0, and how many decisions sit behind a line — and, second,
    every group that produced NO line, with the reason. A group that silently
    vanishes is the failure mode this table exists to catch."""
    at = set(zip(fin["family"], fin["size"]))
    rows = []
    for (pop, grp), g in out.groupby(["population", "group"]):
        fams = {f for pr in groups[grp] for f in pr}
        largest = size_order(set(fin.loc[fin["family"].isin(fams), "size"]))[-1]
        ref_row = g[(g["size"] == TARGET_SIZE) & (g["frac"] == 1.0)]
        rows.append({"population": pop, "group": grp, "families": len(fams),
                     "largest size trained": largest, "reference used": TARGET_SIZE,
                     "reference is largest": "yes" if largest == TARGET_SIZE else f"NO ({largest})",
                     "R at reference": f"{ref_row['reliability'].iloc[0]:.3f}" if len(ref_row) else "—",
                     "sizes": g["size"].nunique(), "points": len(g), "tasks": int(g["n_tasks"].max()),
                     "decisions at 1.7B": int(ref_row["n_comparable"].iloc[0]) if len(ref_row) else 0,
                     "N_min": g["n_min_size"].iloc[0] or "never"})
    drawn = set(out["group"])
    gone = []
    for grp, pl in sorted(groups.items()):
        if grp in drawn:
            continue
        at_ref = sum((a, TARGET_SIZE) in at and (b, TARGET_SIZE) in at for a, b in pl)
        gone.append({"group": grp, "family pairs": len(pl), "pairs at the reference": at_ref,
                     "why no line": "no pair has both members at the reference (rule 9: no fallback)"
                     if not at_ref else f"{at_ref} pairs at the reference, below MIN_PAIRS ({MIN_PAIRS}, rule 5)"
                     if at_ref < MIN_PAIRS else "every task gated out (rq00 above-random)"})
    return pd.DataFrame(rows), pd.DataFrame(gone)


def draw_lines(ax, out: pd.DataFrame, groups: list, colours: dict, x: str = "non_emb",
               counts: bool = False, band: tuple = (OVERALL,), labels: dict | None = None) -> None:
    """One population's lines into `ax`: solid up to the last real proxy, faint
    dashed into the hollow reference point, the first point clearing tau ringed,
    the jackknife band on the groups in `band`, and (`counts`) the task count
    under every point. `labels` renames a group in the legend."""
    for grp in groups:
        g = out[out["group"] == grp].sort_values(x)
        if not len(g):
            continue
        c, lw, z = (S.INK, 2.0, 6) if grp == OVERALL else (colours[grp], 1.3, 3)
        # The reference point is 1.0 because it is compared with itself, so the
        # segment into it is not evidence of convergence. Solid up to the last
        # real proxy, faint dashed into the reference, which is drawn hollow.
        # Only the reference's OWN FINAL is 1.0 by construction; on the compute
        # axis its earlier checkpoints are ordinary proxies and stay on the line.
        is_ref = (g["size"] == TARGET_SIZE) & (g["frac"] == 1.0)
        real, ref_pt = g[~is_ref], g[is_ref]
        if not len(real):        # only the reference: nothing to draw, so no legend entry
            continue
        ms = 4 if x != "compute" else 2.5
        ax.plot(real[x], real["reliability"], color=c, marker="o", ms=ms, lw=lw,
                label=(labels or {}).get(grp, grp), zorder=z)
        # The jackknife band is drawn on the pooled line alone by default: a regime
        # line rests on 4-5 families, where a leave-one-out interval is four
        # numbers and says nothing. The CSV carries every group's `lo`/`hi`.
        if grp in band and "lo" in real:
            b = real.dropna(subset=["lo", "hi"])
            if len(b):
                ax.fill_between(b[x], b["lo"].clip(0, 1), b["hi"].clip(0, 1), color=c, alpha=.10, lw=0, zorder=1)
        if counts:               # rule 13: the population moves along a line, so say so at every point
            for _, r in real.iterrows():
                ax.annotate(f"{int(r['n_tasks'])}", (r[x], r["reliability"]), textcoords="offset points",
                            xytext=(0, -9), ha="center", va="top", fontsize=5, color=c, alpha=.8)
        if len(ref_pt):
            ax.plot([real[x].iloc[-1], ref_pt[x].iloc[0]],
                    [real["reliability"].iloc[-1], ref_pt["reliability"].iloc[0]],
                    color=c, lw=lw * .7, ls=(0, (2, 2)), alpha=.45, zorder=2)
            ax.plot(ref_pt[x], ref_pt["reliability"], marker="o", ms=4.5,
                    mfc=S.SURFACE, mec=c, mew=1.2, ls="none", zorder=z)
        hit = real[real["reaches_tau"]]
        if len(hit):                      # N_min(tau), the claim the figure exists for
            r = hit.iloc[0]
            lbl = r["size"] if x != "compute" else f"{r['size']} @ {G.chinchilla(r['frac'])}"
            ax.plot(r[x], r["reliability"], marker="o", ms=11, mfc="none", mec=c, mew=1.6, zorder=z + 1)
            high = r["reliability"] > .88
            ax.annotate(lbl, (r[x], r["reliability"]), textcoords="offset points",
                        xytext=(0, -15 if high else 10), ha="center",
                        va="top" if high else "bottom", fontsize=6.5, color=c, zorder=5)


def figure(out: pd.DataFrame, path: Path, by: str, pool: str, tau: float,
           populations: tuple = POPULATIONS, note: str = "", x: str = "non_emb",
           counts: bool = False) -> None:
    # OVERALL first, so it heads the legend; its zorder keeps it above the rest.
    rest = group_order(g for g in out["group"].unique() if g != OVERALL)
    groups = [OVERALL] + rest
    colours = {g: L_COLOUR.get(g, c) for g, c in zip(rest, GROUP_COLOURS * 3)}
    fig, axes = plt.subplots(1, len(populations), figsize=(5.9 * len(populations), 4.6), sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax, pop in zip(axes, populations):
        draw_lines(ax, out[out["population"] == pop], groups, colours, x, counts)
        ax.axhline(tau, color=S.MUTED, lw=.8, ls=":")
        ax.set_xscale("log"); ax.set_ylim(0, 1.0)
        if x == "non_emb":
            ax.set_xticks([NON_EMB[s] for s in size_order(out["size"].unique())])
            ax.set_xticklabels(size_order(out["size"].unique()))
            ax.set_xlabel("non-embedding parameters (log)")
        else:
            ax.set_xlabel("training FLOPs spent by the proxies (log)")
        ax.set_title(pop, loc="left", fontsize=8.5)
        ax.grid(color=S.GRID, lw=.6); S.clean(ax)
    axes[0].set_ylabel(f"decision reliability vs {TARGET_SIZE} final")
    if len(groups) > 1:               # a lone pooled line names itself in the subtitle
        axes[0].legend(fontsize=6.5, frameon=False, ncol=2, loc="lower right")
    top = G._header(fig, f"Scale convergence: how small a fully trained model still decides like {TARGET_SIZE}",
                    f"{BY_LINE[by]}; "
                    + (f"a decision = one pair of design variants, read at both models' FINAL checkpoint "
                       f"({len(FRACS)} evaluated, the last of each). " if x == "non_emb" else
                       f"a decision = one pair of design variants, read at each of the {len(FRACS)} evaluated "
                       f"checkpoints of the proxies' runs (0.5C ... 5C) against the reference's FINAL, so a line "
                       f"carries {len(FRACS)} points per size and x is the compute actually spent. ")
                    + f"R = matching decisions / comparable decisions, pooled over the "
                    f"gated tasks with ≥ {MIN_PAIRS} pairs; dotted line = τ = {tau:g}, the ring = N_min(τ), the smallest "
                    f"size clearing it. The hollow {TARGET_SIZE} point is 1.0 by construction (compared with itself) and "
                    f"the dashed segment into it is not evidence of convergence. The shaded band on `{OVERALL}` is a "
                    f"leave-one-design-variant-out jackknife (90 %): how much the line depends on which variants are "
                    f"in the grid, not on seed noise (no replicate exists at {TARGET_SIZE}). Gate and pair "
                    f"minimum as everywhere in rq02; pairs from the {POOL} pool, gated with {pool}'s mask." + note)
    fig.tight_layout(rect=(0, 0, 1, top))
    S.save(fig, path, dpi=150)


def generate_readme(pool: str, out_dir: Path, tables: dict) -> None:
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    body = ["## Scale convergence — the minimum useful scale",
            f"How small a **fully trained** model may be and still decide the way the {TARGET_SIZE} final checkpoint "
            f"does: R_size(N) = matching decisions / comparable decisions over the gated tasks, and N_min(τ) = the "
            f"smallest size with R ≥ τ (τ = {TAU:g}). Same kernel, gate and pair minimum as the rest of rq02 — this is "
            f"DA-size pooled over decisions rather than averaged over tasks, so the counts behind a point are in the "
            f"CSV. The {TARGET_SIZE} point is 1.0 by construction. Regenerate with "
            f"`python analysis/rq02_decision_accuracy/scale_convergence.py` (all three groupings)."]
    heading = {"overall": "**Pooled over every pair**", "L": "**By language count**",
               "transformation": "**By design axis**"}
    for by, t in tables.items():
        body += [f"{heading[by]} (benchmarks; `{stem_for(by)}.csv` carries BPB and the decision counts):",
                 md_table(list(t.columns), t.values.tolist()),
                 f"![Scale convergence, {by}]({stage}/{pool}/{stem_for(by)}.png)"]
    replace_block(OUT_ROOT / "README.md", "scale-convergence", "\n\n".join(body),
                  "scale_convergence.py")


def common_tasks(kept: pd.DataFrame) -> set[str]:
    """The tasks with a cell in EVERY drawn regime at EVERY proxy size: one task
    set for the whole figure, so both the across-L and the across-size readings
    are paired. Empty when no task manages it, which is itself a result."""
    regimes = [g for g in kept["group"].unique() if g != OVERALL]
    proxies = [s for s in kept["size"].unique() if s != TARGET_SIZE]
    sets = [set(kept.loc[(kept["group"] == g) & (kept["size"] == s), "task"]) for g in regimes for s in proxies]
    return set.intersection(*sets) if sets else set()


def run(by: str, pool: str, tau: float, out_dir: Path, variant: str = "",
        dec: pd.DataFrame | None = None, groups: dict | None = None,
        fin: pd.DataFrame | None = None, x: str = "non_emb", axes: str = "multi-axis",
        langs: str = "all", common: bool = False, axis_of: dict | None = None) -> pd.DataFrame | None:
    """`dec` / `groups` / `fin` are passed in when a second variant reuses the
    first one's decisions: the pair sets and the per-decision rows do not depend
    on which tasks are kept, so the pool is loaded and scored once. `x` picks
    the axis, and with it which `dec` table was handed in. `langs` and `common`
    are the two task restrictions the module docstring describes."""
    stem = stem_for(by, variant, axes, langs, common) + ("_flops" if x == "compute" else "")
    populations, note = POPULATIONS, ""
    cells = reliability(dec)
    if variant:
        red, thresh, crit = FILTERS[variant]
        keep = load_reliable(out_dir, variant, axes)
        if keep is None:
            return None
        cells = cells[cells["task"].isin(set(keep["task"]))]
        populations = ("all benchmarks",)
        note = (f" Restricted to the {len(keep)} (benchmark, language) cells reliable on {crit} "
                f"(DA ≥ {thresh:g}, {red} reduction, reliable_tasks.py), over "
                f"{keep['benchmark'].nunique()} benchmark(s) and {keep['language'].nunique()} languages.")
    if LANG_SETS[langs] is not None:
        cells = cells[cells["task"].map(assign_language).isin(LANG_SETS[langs])]
        note += (f" Tasks in the {len(LANG_SETS[langs])} languages of the {langs} setting only "
                 f"({', '.join(sorted(LANG_SETS[langs]))}), which every regime from {langs} up trains.")
    if common:
        kept = keep_cells(cells, pool)
        shared = common_tasks(kept)
        if not shared:
            print(f"\n--- {stem} --- !!! no task has ≥ {MIN_PAIRS} pairs in every regime at every proxy size: no figure")
            return None
        cells = cells[cells["task"].isin(shared)]
        note += (f" Common-task set: the {len(shared)} tasks with ≥ {MIN_PAIRS} pairs in every regime at every "
                 f"proxy size, so every line is read over the same benchmarks. The decision MIX still differs "
                 f"between regimes (`share_*` in the CSV) and cannot be held fixed under rule 5.")
    keys = ["population", "group", "size"] + (["frac"] if x == "compute" else [])
    out = decorate(aggregate(cells, pool, tau, x), dec, cells, pool, keys, axis_of)
    diag, gone = diagnostics(out, groups, fin)
    print(f"\n--- {stem} ---")
    print(diag.to_string(index=False))
    if len(gone):
        print(f"  {len(gone)} group(s) with no line: {', '.join(gone['group'])}")
    out.to_csv(out_dir / f"{stem}.csv", index=False)
    # Per-point counts where they vary and fit: the one-panel restricted figures.
    # The common-task figure has one count everywhere (the caption says it) and
    # the two-panel one has ring labels in the same place.
    figure(out, out_dir / f"{stem}.png", by, pool, tau, populations, note, x,
           counts=langs != "all" and not common and bool(variant))
    return out


def run_all(by: str, pool: str, tau: float, out_dir: Path, axes: str = "multi-axis",
            langs: str = "all", common: bool = False) -> dict:
    df = ladder_frame(POOL)
    fin = finals(df)
    attrs = design_axes(df)
    groups, axis_of = pairs_by_group(attrs, by, axes), pair_axis(attrs)
    sizes = size_order(fin["size"].unique())
    print(f"\n=== --by {by} [{axes}, langs {langs}{', common tasks' if common else ''}]: {len(groups)} groups, "
          f"{sum(map(len, groups.values()))} family pairs, sizes {sizes} ===")
    dec = decisions(fin.assign(frac=1.0), groups, sizes, fin)
    kw = dict(axes=axes, langs=langs, common=common, axis_of=axis_of)
    out = {v: run(by, pool, tau, out_dir, v, dec, groups, fin, **kw) for v in VARIANTS}
    # The ten-point grid costs ten times the size axis, so only score it when a
    # reliable-task table exists for at least one _flops variant to filter on.
    todo = [v for v in FLOPS_VARIANTS if load_reliable(out_dir, v, axes) is not None]
    if todo:
        grid = decisions(grid_frame(df, FRACS), groups, sizes, fin, FRACS)
        for v in todo:
            run(by, pool, tau, out_dir, v, grid, groups, fin, x="compute", **kw)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--by", default=["overall", "L", "transformation"], nargs="+",
                   choices=["overall", "L", "transformation"],
                   help="how the decisions are grouped into lines")
    p.add_argument("--pool", default=CANONICAL_POOL, help="the pool whose above-random gate applies")
    p.add_argument("--tau", type=float, default=TAU, help="the reliability threshold N_min is read at")
    p.add_argument("--axes", default="multi-axis", choices=["multi-axis", "mono-axis"],
                   help="the pair set (rule 15); mono-axis writes the `_one_axis` twins")
    p.add_argument("--langs", default="all", choices=list(LANG_SETS),
                   help="restrict the tasks to the languages of one L setting (stem token replaces `L`)")
    p.add_argument("--common-tasks", action="store_true",
                   help="restrict to the tasks with ≥ MIN_PAIRS pairs in every regime at every proxy size")
    args = p.parse_args()
    out = OUT_ROOT / load_pools()[args.pool].get("stage", "pretraining") / args.pool
    tables = {}
    for by in args.by:
        t = run_all(by, args.pool, args.tau, out, args.axes, args.langs, args.common_tasks)[""]
        if t is None:
            continue
        b = t[t["population"] == "all benchmarks"]
        tables[by] = (b.pivot_table(index="group", columns="size", values="reliability")
                      .reindex(columns=size_order(b["size"].unique())).round(2)
                      .join(b.groupby("group")["n_min_size"].first().rename(f"N_min(τ={args.tau:g})"))
                      .fillna("—").reset_index())
    # The block names all three groupings, so a partial run must not rewrite it
    # with a subset — that silently drops the other two tables from the README.
    if set(args.by) == set(BY_LINE) and args.axes == "multi-axis" and args.langs == "all" and not args.common_tasks:
        generate_readme(args.pool, out, tables)
    elif args.langs != "all" and args.pool == CANONICAL_POOL and args.axes == "multi-axis" and "L" in tables:
        # the restricted L figures get their own block, so a reader finds them
        stage = load_pools()[args.pool].get("stage", "pretraining")
        stem = stem_for("L", "above_66_size", args.axes, args.langs, args.common_tasks)
        body = "\n\n".join([
            f"## Scale convergence by language count, on the {args.langs} languages"
            + (", common tasks" if args.common_tasks else ""),
            f"The `--by L` lines read over the tasks in the {len(LANG_SETS[args.langs])} languages of the {args.langs} "
            f"setting ({', '.join(sorted(LANG_SETS[args.langs]))}), which every regime from {args.langs} up trains"
            + (f", and further over the tasks with ≥ {MIN_PAIRS} pairs in every regime at every proxy size — one task set "
               f"for the whole figure, so a gap between lines is a gap on the same benchmarks" if args.common_tasks else "")
            + f". What this cannot fix: a regime pools arch, list and temperature decisions at once and the mix differs "
            f"by regime (`share_*` in the CSV); rule 5 forbids holding it fixed. Under `--axes mono-axis` the regimes "
            f"keep only their one-axis pairs and rule 5 empties every proxy size but 1B, so the `_one_axis` L figures "
            f"draw the pooled line alone. Numbers below are the unfiltered population; the `above_66_size` twin "
            f"(`{stem}.png`) is the conditional one. Regenerate with `python analysis/rq02_decision_accuracy/"
            f"scale_convergence.py --by L --langs {args.langs}{' --common-tasks' if args.common_tasks else ''}`.",
            md_table(list(tables["L"].columns), tables["L"].values.tolist()),
            f"![Scale convergence, {args.langs}{' common tasks' if args.common_tasks else ''}]"
            f"({stage}/{args.pool}/{stem_for('L', '', args.axes, args.langs, args.common_tasks)}.png)"])
        replace_block(OUT_ROOT / "README.md", f"scale-convergence-{args.langs}{'-common' if args.common_tasks else ''}",
                      body, f"scale_convergence.py --by L --langs {args.langs}{' --common-tasks' if args.common_tasks else ''}")
    else:
        print(f"  (README block left alone: --by {' '.join(args.by)} is a subset of {sorted(BY_LINE)}, "
              f"or a restricted run)")
