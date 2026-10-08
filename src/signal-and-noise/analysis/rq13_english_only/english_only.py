"""English only: do the monolingual-English cells (L = 1) read the English
benchmarks better than the multilingual ones?

An L = 1 cell trains on English alone; every other language setting gives
English half of its tokens. If the trainings are right, an L1 model scores
higher on the English benchmarks than a multilingual model of the same size
and depth, clears chance on more of them at a smaller size, and ranks its
design variants at least as reliably. Each expectation is read with the
computation that already defines it, on the English benchmarks only: every
task the registry tags English (`utils.assign_language(task) == "en"`), every
variant (original, RF, LLM-RF; accuracy and bBPB), the per-language BPB of the
English validation set (`bpb_dclm`) kept apart.

Everywhere, a size whose task count falls below half the largest of its line
is `thin`: kept in the CSV, never drawn, quoted or counted in the verdict
(rule 13; the final evaluation of the L1 deep 350M cell covers few English
tasks).

  scores     the final score of the L1 baseline cell against the same-depth
             baseline cell at every other L, per English task and size,
             oriented so that a positive gap is L1 better (`utils.lower_is_better`
             flips bBPB); read against the seed noise of the deep L1 cell (the
             sample std over its replicate seeds, `effect_vs_noise`'s
             `seed_noise`) where the replicates exist (175M, 600M, 1B). A gap
             beyond BEYOND seed sds is a different model, as the noise-and-SNR
             analysis reads it. Rule 1: an at-chance (task, size) keeps its row
             and no gap.
  gate       per cell and size, the share of the English tasks with a chance
             level whose final run clears chance (the gate's run-level verdict,
             `above_chance`); per task, the smallest size from which the run
             stays above chance (`grids.smallest_safe`), L1 against every other L.
  scaling    the power-law error of the 1.7B English BPB per ladder top
             (`scaling_law_error`) and the R² of the log-N fit of every English
             accuracy task (`fit_table`) on the rungs the L1 deep cell has, so
             every L fits the same rungs. The fit includes the 1.7B reference:
             a reference-size quantity (rule 11), a property of the ladder and
             not a forecast of 1.7B from below (the BPB error is that).
  decisions  DA-size among L1's families against the 1.7B reference, next to
             the DA among the same number of families at every other L that has
             them. The families compared at an L MIRROR L1's: the deep and
             shallow cells of the baseline data build, plus every deep cell of
             another build (L1: DCLM without the edu filter, FineWeb; L2: the
             Chinese and Spanish second language; L15, L30: the B list and T = 3).
             The build is read off the family name, which the launcher writes
             from the scheme registry's label. Decisions, the gate at the proxy
             and the reference, the pair minimum and the leave-one-family-out band
             are the decision-accuracy computation's (`scale_convergence`:
             `decisions`, `reliability`, `aggregate`, `decorate`); both pair sets.
             A cell enters a line only where EVERY family of its group has the
             task (the group's full pair count), and a (task, size) only where
             every group has such a cell, so the lines of a size read one
             population. The groups' decisions differ (L1 moves the English
             corpus, L2 the second language, L15/L30 the list and T; only the
             depth pair is shared), so the decisions are also split by the axis
             a pair moves (`pair_axis`).
  SNR        per English task, size and family group, the noise-and-SNR headline
             definition (`rel_std` over the group's runs: `per_model_inputs` and
             the aggregator), at-chance cells blank; over the mirrored groups
             above and over the deep and shallow baseline cells of every L — the
             one decision every L shares, so the like-for-like reading (two
             families: rule 5 is about decision accuracy, an SNR needs two runs).
             A group's median reads only tasks all its families have, over the
             tasks every group of the set has.
  surrogate  Spearman rho of log10 SNR at the proxy against DA-size at the proxy
             over the English tasks, per group: the relation the surrogates
             analysis starts from. Its catalogue of ~210 statistics is not rerun:
             it ranks them per language over every pool, and here there is one
             language and one group of four families per L.
  AllenAI    our L1 DA-size and SNR next to DataDecide's (the external frameworks
             comparison's `allenai_snr_variants_per_task.csv`) on the shared
             English tasks matched by FORMAT (cloze against OLMES's RC, letter
             against `:mc`), at the matched sizes, with every difference of
             definition carried in the table. DataDecide's range at a size is
             taken over the tasks where L1 has a value there (ours is gated,
             theirs is not). The SNRs are not comparable (different signal and
             noise definitions) and are shown, not graded.
  verdict    one row per expectation: confirmed or not, with its number. "yes" =
             L1 ahead of (or within TIE of) every comparator at every size drawn,
             "partly" = behind somewhere and ahead somewhere; a reading over
             different decisions per L is "not like-for-like".

Outputs, `pretraining/<pool>/`:

    english_only_scores.png/.csv                 mean gap and win share per size, L1 against every other L
    english_only_scores_paper.png/.svg/.csv      the same for the paper: bare (rule 18); `--paper` redraws it
                                                 alone from english_only_scores_summary.csv
    english_only_scores_fixed_tasks_paper.png/.svg/.csv   the same on one population per line (rule 13), the
                                                 moving line dashed behind it; `--paper` redraws it too
    english_only_scores_by_benchmark.png/.csv    the same per benchmark (deep, every other L pooled)
    english_only_scores_per_task.csv             one row per (task, size, arch, comparator L)
    english_only_scores_summary.csv              per (scoring, arch, size, comparator), with `thin`
    english_only_above_random.png/.csv           share above chance per cell; which L crosses first
    english_only_above_random_per_task.csv       the run-level verdicts behind it
    english_only_scaling.png/.csv                English BPB prediction error; English benchmark R²
    english_only_da_size_by_L{_multi_axes,_mono_axis}.png/.csv   DA-size per family group (band, counts)
    english_only_da_size_by_L_both_axes.csv      the pooled lines in full: band, pair and task counts, `thin`
    english_only_da_size_by_pair_axis_both_axes.csv  the same decisions split by the axis a pair moves
    english_only_da_size_per_task_both_axes.csv  every (task, group, size) cell: gated and short ones NaN, pair count kept
    english_only_snr.png/.csv                    SNR per group, per task, and SNR against DA
    english_only_snr_per_task.csv                one row per (families, task, group, size)
    english_only_allenai.png/.csv                ours next to DataDecide on the shared tasks
    english_only_allenai_per_task.csv            the rows behind it, with L1's pair or run count
    english_only_verdict.csv                     one row per expectation

    python analysis/rq13_english_only/english_only.py --pool predictivity
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools, size_bucket  # noqa: E402
from pretrain.launch_trainings import LADDERS  # noqa: E402
from snr.download.ladder import ladder_dir  # noqa: E402
from snr.snr_variants import AGGREGATION_FUNCTIONS  # noqa: E402
from analysis import grids as G  # noqa: E402
from analysis import style as S  # noqa: E402
from analysis.autodoc import CANONICAL_POOL, fmt, md_table, replace_block  # noqa: E402
from analysis.paths import ENGLISH_ONLY  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import above_chance, load_mask, mask_pool  # noqa: E402
from analysis.rq01_scaling_predictability.analyze import fit_table  # noqa: E402
from analysis.rq01_scaling_predictability.scaling_law_error import scaling_law_error  # noqa: E402
from analysis.rq02_decision_accuracy.scale_convergence import (  # noqa: E402
    AXIS_LABEL, L_COLOUR, aggregate, decisions, decorate, group_order, keep_cells, pair_axis, reliability)
from analysis.rq03_noise_and_snr.effect_vs_noise import effect_vs_noise  # noqa: E402
from analysis.rq03_noise_and_snr.panels import VARIANT  # noqa: E402
from analysis.rq03_noise_and_snr.run_apertus_snr_variants import (  # noqa: E402
    per_model_inputs, variant_key, variant_signal_noise_snr)
from analysis.rq07_external_frameworks.analyze import ALLENAI_CSV, SIZE_PAIRS  # noqa: E402
from analysis.utils import (  # noqa: E402
    AXES_SUFFIX, GRID_SEED, MIN_PAIRS, RELIABLE_DA, SMALL_SIZES, TARGET_SIZE, assign_language,
    benchmark_family, design_axes, finals, fixed_population, ladder_frame, lower_is_better, pair_sets, passes_gate,
    size_order, variant)

EN = "en"
ENGLISH_BPB = "bpb_dclm"              # the English validation set's BPB: a measurement of its own, not a benchmark
ARCHS = ("deep", "shallow")           # the ladders every L trains (swiglu exists at three L's only)
SEEDS_POOL = "predictivity_seeds"     # every seed: the replicates the seed yardstick is made of
BEYOND = 2.0                          # |gap| / seed sd above this = a different model (the noise-and-SNR reading)
DEFICIT = 0.10                        # an L1 deficit this large (accuracy) is named in the README
TIE = 0.01                            # the verdict's tie margin: DA, R², share above chance, log10 SNR, % BPB error
JUMP = 0.3                            # bits: an L1 bBPB value this far above BOTH neighbouring sizes is a spike
AXES = ("multi-axis", "mono-axis")
# a group's data-build pairs are not one A/B/C comparison (L1 moves the English
# corpus, L2 the second language, L15 and L30 scheme A against B): a neutral label
EO_AXIS_LABEL = {**AXIS_LABEL, "scheme": "data build"}
MIRROR, DEPTH = "mirror", "depth"     # SNR family groups: L1's mirrored at each L; deep and shallow at every L
FAMILIES = {MIRROR: "the families mirroring L1's", DEPTH: "deep against shallow"}
SNR_FUNC = next(fd["func"] for fd in AGGREGATION_FUNCTIONS if variant_key(fd) == VARIANT)
# Our task -> DataDecide's, matched by FORMAT: lm-eval's ARC, HellaSwag and
# OpenBookQA score the answer strings, as OLMES's RC does; the `rf_` twins are
# the answer-string (cloze) rewrite of a letter task; the letter originals
# meet OLMES's `:mc`. DataDecide has no `mmlu:mc` aggregate, so the letter
# MMLU has no partner, and the Global-MMLU English split is not paired now that
# the original MMLU is evaluated.
ALLENAI_TASKS = {"arc_easy": "arc_easy", "arc_challenge": "arc_challenge", "hellaswag": "hellaswag",
                 "openbookqa": "openbookqa", "rf_commonsense_qa": "csqa", "rf_mmlu": "mmlu",
                 "commonsense_qa": "csqa:mc"}
ALLENAI_REF = "1B"                    # DataDecide's largest rung: its reference, so it has no DA-size
_FAMILY = re.compile(rf"lm-L\d+-(?:(.+)-)?(?:{'|'.join(LADDERS)})-seed\d+")
mpl.rcParams.update(S.RC)


# --- who is compared with whom ------------------------------------------------

def build_token(family: str) -> str:
    """The data build a family's name carries between its L and its ladder:
    `lm-L1-dclmP-deep-seed1904` -> `dclmP`, `lm-L8-schemeB-shallow-seed1904` ->
    `schemeB`, '' for the baseline build. The name is written by the launcher
    from the scheme registry's label and stays as trained, so it identifies a
    build whatever the frame's scheme columns are called."""
    return _FAMILY.fullmatch(family).group(1) or ""


def cell_table(fin: pd.DataFrame) -> pd.DataFrame:
    """One row per grid-seed family: L, ladder and data build."""
    c = fin.loc[fin["seed"] == GRID_SEED, ["family", "L", "ladder"]].drop_duplicates("family")
    return c.assign(L=c["L"].astype(int), build=c["family"].map(build_token)).reset_index(drop=True)


def mirror_groups(cells: pd.DataFrame) -> dict[int, list[str]]:
    """L -> the families that mirror L1's: the deep and shallow cells of the
    baseline build and every deep cell of another build. At L1 that is all
    four (deep, shallow, DCLM without the edu filter, FineWeb)."""
    keep = cells["ladder"].isin(ARCHS) & ((cells["build"] == "") | (cells["ladder"] == "deep"))
    return {int(L): sorted(g["family"]) for L, g in cells[keep].groupby("L")}


def depth_groups(cells: pd.DataFrame) -> dict[int, list[str]]:
    """L -> the deep and shallow cells of the baseline build, where both exist:
    the one decision every L shares."""
    b = cells[(cells["build"] == "") & cells["ladder"].isin(ARCHS)]
    return {int(L): sorted(g["family"]) for L, g in b.groupby("L") if len(g) == len(ARCHS)}


def gated(mask, tasks, sizes) -> np.ndarray:
    """True where rule 1 blanks a (task, size): `utils.passes_gate`, size by size."""
    tasks, sizes = list(tasks), list(sizes)
    ok = {s: passes_gate(mask, sorted(set(tasks)), s) for s in set(sizes)}
    return np.array([not ok[s][t] for t, s in zip(tasks, sizes)], dtype=bool)


def thin(n: pd.Series) -> pd.Series:
    """A point whose population is below half the largest of its line: kept in
    the CSV, never drawn, quoted or counted in the verdict (rule 13)."""
    return n < n.max() / 2


def scoring_of(task: str, kind: str) -> str:
    """acc / bbpb for a benchmark, the kind otherwise (`bpb_dclm`: bpb)."""
    return variant(task)[1] if kind == "benchmark" else kind


def population(fin: pd.DataFrame) -> str:
    """The English task population in one sentence, counted on the frame."""
    t = fin.drop_duplicates("task")
    b = t[t["kind"] == "benchmark"]
    acc = b[b["scoring"] == "acc"]["format"].value_counts()
    return (f"{len(t)} English tasks: {int(acc.sum())} accuracy-scored ("
            + ", ".join(f"{int(n)} {f}" for f, n in acc.items()) + f"), {int((b['scoring'] == 'bbpb').sum())} bBPB twins"
            + (f" and the English validation BPB `{ENGLISH_BPB}`" if ENGLISH_BPB in set(t["task"]) else "") + ".")


def coverage(fin: pd.DataFrame, families: list) -> list[str]:
    """What keeps a family's cells out of the shared population: a scoring it
    has no task of while the others do, and a size where its final covers
    fewer than half the English tasks it covers at its best size."""
    f = fin[fin["family"].isin(families)]
    out = []
    bb = f[f["task"].map(lambda t: variant(t)[1]) == "bbpb"].groupby("family")["task"].nunique()
    if len(bb):
        out += [f"`{fam}` has no bBPB twin (the per-item store predates it)" for fam in families if fam not in bb.index]
    n = f.groupby(["family", "size"])["task"].nunique()
    out += [f"`{fam}` at {s} has {int(v)} of the {int(n[fam].max())} English tasks it has elsewhere"
            for (fam, s), v in n.items() if v < n[fam].max() / 2]
    return out


def label(L) -> str:
    return f"L{int(L)}"


# --- 1. scores ------------------------------------------------------------------

def score_gaps(fin: pd.DataFrame, cells: pd.DataFrame, mask, seed_sd: pd.DataFrame) -> pd.DataFrame:
    """One row per (task, size, arch, comparator L): the L1 baseline cell's final
    score and the comparator's, the gap oriented so positive = L1 better, and
    the gap in units of the deep L1 cell's seed sd at that size."""
    base = fin.merge(cells.loc[cells["build"] == "", ["family"]], on="family")
    piv = base.pivot_table(index=["task", "size", "ladder"], columns="L", values="primary_score")
    if 1 not in piv.columns:
        return pd.DataFrame()
    out = pd.concat([piv[[1, L]].dropna().set_axis(["score_l1", "score_other"], axis=1).reset_index().assign(L=int(L))
                     for L in piv.columns if L != 1], ignore_index=True).rename(columns={"ladder": "arch"})
    out = out[out["arch"].isin(ARCHS)]
    sign = np.where(out["task"].map(lower_is_better), -1.0, 1.0)
    out["gap"] = sign * (out["score_l1"] - out["score_other"])
    kind = fin.groupby("task")["kind"].first()
    out["scoring"] = [scoring_of(t, kind[t]) for t in out["task"]]
    out["benchmark"] = out["task"].map(lambda t: G.display(benchmark_family(t)))
    out = out.merge(seed_sd, on=["task", "size"], how="left")
    out["gap_over_seed"] = out["gap"] / out["seed_sd"].where(out["seed_sd"] > 0)
    out["gated"] = gated(mask, out["task"], out["size"])
    out.loc[out["gated"], ["gap", "gap_over_seed"]] = np.nan          # rule 1: the row stays, the number goes
    return out


def _shares(d: pd.DataFrame) -> pd.Series:
    have = d.dropna(subset=["gap"])
    seed = have.dropna(subset=["gap_over_seed"])
    return pd.Series({"n_tasks": have["task"].nunique(), "n_comparisons": len(have),
                      "mean_l1": have["score_l1"].mean(), "mean_other": have["score_other"].mean(),
                      "mean_gap": have["gap"].mean(), "median_gap": have["gap"].median(),
                      "win_share": (have["gap"] > 0).mean() if len(have) else np.nan,
                      "n_with_seed": len(seed),
                      "ahead_beyond_seed": (seed["gap_over_seed"] > BEYOND).mean() if len(seed) else np.nan,
                      "behind_beyond_seed": (seed["gap_over_seed"] < -BEYOND).mean() if len(seed) else np.nan})


def score_summary(g: pd.DataFrame) -> pd.DataFrame:
    """Per (scoring, arch, size, comparator): the mean gap, the share of tasks
    L1 wins and the shares beyond the seed noise; `comparator` = one L or
    `every L > 1` (every (task, L) comparison pooled); `thin` per line."""
    keys = ["scoring", "arch", "size"]
    per_l = g.groupby(keys + ["L"]).apply(_shares, include_groups=False).reset_index()
    per_l["comparator"] = per_l.pop("L").map(label)
    pooled = g.groupby(keys).apply(_shares, include_groups=False).reset_index().assign(comparator="every L > 1")
    out = pd.concat([per_l, pooled], ignore_index=True)
    out["thin"] = out.groupby(["scoring", "arch", "comparator"])["n_tasks"].transform(thin)
    return out


def spikes(g: pd.DataFrame, sizes: list) -> tuple[str, pd.Index, int, int] | None:
    """The deep bBPB tasks whose L1 value sits more than JUMP bits above the
    value at BOTH neighbouring sizes, at the size where most do: (size, those
    tasks, how many (task, L) comparator cells spike there, of how many)."""
    b = g[(g["scoring"] == "bbpb") & (g["arch"] == "deep")]
    if b.empty:
        return None

    def mask(w):
        w = w.reindex(columns=sizes)
        return ((w - w.shift(1, axis=1)) > JUMP) & ((w - w.shift(-1, axis=1)) > JUMP)
    l1 = mask(b.drop_duplicates(["task", "size"]).pivot(index="task", columns="size", values="score_l1"))
    if not l1.to_numpy().any():
        return None
    at = l1.sum().idxmax()
    ow = b.pivot_table(index=["L", "task"], columns="size", values="score_other").reindex(columns=sizes)
    return at, l1.index[l1[at]], int(mask(ow)[at].sum()), int(ow[at].notna().sum())


# --- 2. the gate ----------------------------------------------------------------

def gate_runs(fin: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """The gate's run-level verdict (one-sided 95 % Wilson LCB > chance) of
    every grid-seed cell on every English task that has a chance level."""
    f = fin[fin["kind"] == "benchmark"].merge(cells[["family", "build"]], on="family")
    f = f.assign(above=above_chance(f["primary_score"].to_numpy(), f["task"].to_numpy()).to_numpy())
    return f.dropna(subset=["above"])[["family", "L", "ladder", "build", "task", "size", "primary_score", "above"]]


def share_above(runs: pd.DataFrame) -> pd.DataFrame:
    """Per drawn cell (the baseline build at every L, every L1 build) and size:
    the share of English tasks above chance, over the tasks EVERY drawn cell
    of that size scores, so the lines of a size read one population (rule 13;
    the L1 deep 350M cell has 17 of its English tasks evaluated)."""
    r = runs[runs["ladder"].isin(ARCHS) & ((runs["build"] == "") | (runs["L"] == 1))]
    n = r.groupby(["size", "task"])["family"].nunique()
    full = n[n == r.groupby("size")["family"].nunique().reindex(n.index.get_level_values("size")).to_numpy()]
    r = r.merge(full.reset_index()[["size", "task"]], on=["size", "task"])
    out = r.groupby(["family", "L", "ladder", "build", "size"])["above"].agg(share="mean", n_tasks="size").reset_index()
    return out.assign(thin=thin(out["n_tasks"]))


def first_above(runs: pd.DataFrame, family: str, sizes: list) -> pd.Series:
    """task -> index of the smallest size from which the run stays above chance
    (len(sizes) = never); NaN where the cell has no verdict at all."""
    p = runs[runs["family"] == family].pivot_table(index="task", columns="size", values="above").reindex(columns=sizes)
    lvl = G.smallest_safe(p)
    return lvl.where(lvl != G.NEVER_CODE, len(sizes))


def crossing(runs: pd.DataFrame, cells: pd.DataFrame, sizes: list) -> pd.DataFrame:
    """Per (arch, comparator L): over the tasks either cell ever clears, the share
    where L1 clears chance from a smaller size, the same size or a larger one."""
    rows = []
    base = cells[cells["build"] == ""].set_index(["L", "ladder"])["family"]
    for arch in ARCHS:
        if (1, arch) not in base.index:
            continue
        l1 = first_above(runs, base[(1, arch)], sizes)
        for L in sorted(x for x, a in base.index if a == arch and x != 1):
            o = first_above(runs, base[(L, arch)], sizes)
            both = pd.concat([l1.rename("l1"), o.rename("other")], axis=1).dropna()
            both = both[(both["l1"] < len(sizes)) | (both["other"] < len(sizes))]
            n = len(both)
            rows.append({"arch": arch, "comparator": label(L), "n_tasks": n,
                         "earlier": (both["l1"] < both["other"]).sum() / n if n else np.nan,
                         "same": (both["l1"] == both["other"]).sum() / n if n else np.nan,
                         "later": (both["l1"] > both["other"]).sum() / n if n else np.nan})
    return pd.DataFrame(rows)


# --- 4. decision accuracy, 5. SNR, 6. the surrogate --------------------------------

def da_tables(fin: pd.DataFrame, groups: dict, gate_pool: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The decision-accuracy computation over each group's own pairs, both pair
    sets. Returns (the pooled lines per (axes, population, group, size); every
    (axes, task, group, size) cell, gated and short ones NaN with their pair
    count (rules 1 and 5), `shared` marking the cells the lines read; the
    decisions of the drawn lines split by the axis each pair moves)."""
    sizes = size_order(fin["size"].unique())
    attrs = design_axes(fin[fin["family"].isin([f for fs in groups.values() for f in fs])])
    axis_of = pair_axis(attrs)
    outs, alls, mix = [], [], []
    for axes in AXES:
        pairs = {label(L): pair_sets(attrs.loc[f])[axes] for L, f in groups.items()}
        dec = decisions(fin.assign(frac=1.0), pairs, sizes, fin)
        cells = reliability(dec)
        # the line reads a cell only with every family of its group, and a (task, size) only where every group has one
        cells["all_families"] = cells["n_comparable"] == cells["group"].map({g: len(p) for g, p in pairs.items()})
        k = keep_cells(cells, gate_pool)
        n = k[k["all_families"]].groupby(["task", "size"])["group"].nunique()
        common = list(n[n == len(pairs)].index)
        cells["shared"] = pd.MultiIndex.from_frame(cells[["task", "size"]]).isin(common) & cells["all_families"].to_numpy()
        print(f"  [{axes}] {', '.join(f'{g}: {len(p)} pairs' for g, p in pairs.items())}; rule 5 empties "
              f"{int((cells['n_comparable'] < MIN_PAIRS).sum())} of {len(cells)} (task, group, size) cells "
              f"(< {MIN_PAIRS} pairs); {int((~cells['all_families']).sum())} lack a family of their group; "
              f"{int(cells['shared'].sum())} cells on (task, size) every group has")
        keys = ["population", "group", "size"]
        like = cells[cells["shared"]]
        out = decorate(aggregate(like, gate_pool, RELIABLE_DA), dec, like, gate_pool, keys)
        out["thin"] = out.groupby(["population", "group"])["n_tasks"].transform(thin)
        outs.append(out.assign(axes=axes))
        full = G.mark_gated(G.add_meta(cells), gate_pool, "size", "da", TARGET_SIZE)
        full.loc[full["n_comparable"] < MIN_PAIRS, "da"] = np.nan            # rule 5: NaN, the pair count stays
        full["population"] = np.where(full["family"] == "bpb", "bpb", "all benchmarks")
        full["scoring"] = [scoring_of(t, "bpb" if f == "bpb" else "benchmark") for t, f in zip(full["task"], full["family"])]
        alls.append(full.assign(axes=axes, n_pairs=full["n_comparable"]))
        # the drawn decisions (kept, proxies, not thin) by the axis their pair moves
        d = dec.astype({c: str for c in ("task", "group", "size", "family_a", "family_b")})
        d = d.merge(keep_cells(like, gate_pool)[["task", "group", "size", "population"]], on=["task", "group", "size"])
        d = d.merge(out.loc[~out["thin"], ["population", "group", "size"]], on=["population", "group", "size"])
        d = d[d["size"].isin(SMALL_SIZES)]
        d["axis"] = [axis_of.get((a, b), "multi") for a, b in zip(d["family_a"], d["family_b"])]
        m = d.groupby(["population", "group", "axis"])["match"].agg(n_matching="sum", n_decisions="size").reset_index()
        mix.append(m.assign(axes=axes, da=m["n_matching"] / m["n_decisions"]))
    out = pd.concat(outs, ignore_index=True)
    out = out.drop(columns=[c for c in ("frac", "tau", "reaches_tau", "n_min_size") if c in out])
    cols = ["axes", "task", "family", "language", "scoring", "population", "group", "size", "da", "n_pairs", "n_matching",
            "gated", "all_families", "shared"]
    return out, pd.concat(alls, ignore_index=True)[cols], pd.concat(mix, ignore_index=True)


def snr_table(df: pd.DataFrame, groups: dict, mask, families: str) -> pd.DataFrame:
    """Per (task, group, size): signal, noise and SNR (`VARIANT`) over the
    group's runs, the noise over rule 4's window; at-chance cells blank;
    `all_families` where every family of the group has a run."""
    d = df.assign(bucket=df["size"].map(size_bucket))
    rows = []
    for L, fams in groups.items():
        for (task, size), x in d[d["family"].isin(fams)].groupby(["task", "size"]):
            inputs = per_model_inputs(x, task, size)
            sig, noi, snr = variant_signal_noise_snr(inputs, SNR_FUNC)
            rows.append({"families": families, "task": task, "group": label(L), "size": size, "n_families": len(fams),
                         "n_runs": 0 if inputs is None else len(inputs[0]), "signal": sig, "noise": noi, "snr": snr})
    t = pd.DataFrame(rows)
    t["all_families"] = t["n_runs"] == t["n_families"]
    t["gated"] = gated(mask, t["task"], t["size"])
    t.loc[t["gated"], ["signal", "noise", "snr"]] = np.nan
    t["log10_snr"] = np.log10(t["snr"].where(t["snr"] > 0))
    t["benchmark"] = t["task"].map(lambda x: G.display(benchmark_family(x)))
    return t


def paired_snr(snr: pd.DataFrame) -> pd.DataFrame:
    """Per (families, group, size): the median log10 SNR over the English
    benchmarks that have an SNR from every family of EVERY group of the set
    at that size, so the groups read one population; `thin` per line."""
    out = []
    for fam, s in snr.groupby("families"):
        b = s[(s["task"] != ENGLISH_BPB) & s["all_families"]].dropna(subset=["log10_snr"])
        n = b.groupby(["size", "task"])["group"].nunique()
        b = b.merge(n[n == s["group"].nunique()].reset_index()[["size", "task"]], on=["size", "task"])
        out.append(b.groupby(["group", "size"])["log10_snr"].agg(median_log10_snr="median", n_tasks="size")
                   .reset_index().assign(families=fam))
    p = pd.concat(out, ignore_index=True)
    p["thin"] = p.groupby(["families", "group"])["n_tasks"].transform(thin)
    return p


def surrogate(snr: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Per (axes, group, proxy): Spearman rho of log10 SNR against DA-size over
    the English benchmarks the DA lines read (the gate and rule 5 applied)."""
    k = cells[(cells["population"] == "all benchmarks") & cells["size"].isin(SMALL_SIZES) & cells["shared"]].dropna(subset=["da"])
    s = snr[snr["families"] == MIRROR]
    m = k.merge(s[["task", "group", "size", "log10_snr"]], on=["task", "group", "size"]).dropna(subset=["log10_snr"])
    rows = []
    for (axes, grp, size), g in m.groupby(["axes", "group", "size"]):
        ok = len(g) >= 3 and g["da"].nunique() > 1 and g["log10_snr"].nunique() > 1
        rows.append({"axes": axes, "group": grp, "size": size, "n_tasks": len(g),
                     "rho": spearmanr(g["log10_snr"], g["da"]).statistic if ok else np.nan})
    out = pd.DataFrame(rows)
    return out.assign(thin=out.groupby(["axes", "group"])["n_tasks"].transform(thin)) if len(out) else out


# --- 7. AllenAI ---------------------------------------------------------------------

def allenai(snr: pd.DataFrame, cells: pd.DataFrame, others: list) -> pd.DataFrame | None:
    """Ours next to DataDecide's per shared task and matched size pair: DA-size
    (multi-axis, the only pair set DataDecide's 25 recipes have) and SNR. Ours
    for L1 (cells with every one of its families: `l1_n` pairs or runs) and the
    median over the other groups; theirs with the min and max over the shared
    tasks where L1 has a value at that size, the range L1 is read against
    (ours is gated, theirs is not: a range over tasks we blank would not be one)."""
    if not ALLENAI_CSV.is_file() or ALLENAI_CSV.read_bytes()[:40].startswith(b"version https://git-lfs"):
        print(f"  (no AllenAI table at {ALLENAI_CSV}: run build_allenai_variants.py or git lfs pull; comparison skipped)")
        return None
    al = pd.read_csv(ALLENAI_CSV, index_col="task")
    pairs = [(o, t) for o, t in ALLENAI_TASKS.items() if t in al.index]
    c = cells[cells["axes"] == "multi-axis"]
    s = snr[snr["families"] == MIRROR]
    ours = {"da_size": (c[c["all_families"]].set_index(["task", "group", "size"])[["da", "n_pairs"]],
                        c.drop_duplicates(["task", "size"]).set_index(["task", "size"])["gated"]),
            "snr": (s[s["all_families"]].set_index(["task", "group", "size"])[["snr", "n_runs"]],
                    s.drop_duplicates(["task", "size"]).set_index(["task", "size"])["gated"])}
    rows = []
    for o, theirs in pairs:
        for our_size, their_size in [p for p in SIZE_PAIRS if p[0] != TARGET_SIZE]:
            for q, col in (("da_size", f"decision_acc_size_{their_size}"), ("snr", f"snr_{VARIANT}_{their_size}")):
                if col not in al.columns:          # DataDecide's 1B is its reference: no DA-size there
                    continue
                vals, gate = ours[q]
                v = vals.iloc[:, 0]
                l1 = vals.loc[(o, "L1", our_size)] if (o, "L1", our_size) in vals.index else pd.Series([np.nan, np.nan])
                other = [v.get((o, g, our_size), np.nan) for g in others]
                rows.append({"task": o, "allenai_task": theirs, "quantity": q, "size": our_size,
                             "allenai_size": their_size, "l1": l1.iloc[0], "l1_n": l1.iloc[1],
                             "gated": bool(gate.get((o, our_size), False)),
                             "other_l_median": np.nanmedian(other) if np.isfinite(other).any() else np.nan,
                             "allenai": al.at[theirs, col]})
    t = pd.DataFrame(rows)
    rng = t.dropna(subset=["l1"]).groupby(["quantity", "size"])["allenai"].agg(allenai_min="min", allenai_max="max")
    t = t.join(rng, on=["quantity", "size"])
    t["l1_vs_allenai_range"] = np.select([t["l1"] < t["allenai_min"], t["l1"] > t["allenai_max"]], ["below", "above"],
                                         "inside")
    t.loc[t["l1"].isna(), "l1_vs_allenai_range"] = ""
    t["l1_over_allenai"] = t["l1"] / t["allenai"]          # the same task, the matched size
    return t


# --- figures ------------------------------------------------------------------------

def _x(ax, sizes):
    ax.set_xticks(range(len(sizes))); ax.set_xticklabels(sizes)
    ax.grid(color=S.GRID, lw=.6); S.clean(ax)


def _line(ax, xs, ys, panel, row, sizes, **kw) -> pd.DataFrame:
    ax.plot([sizes.index(s) for s in xs], ys, **({"marker": "o", "ms": 3.5} | kw))
    return pd.DataFrame({"panel": panel, "row": row, "col": list(xs), "value": list(ys)})


def _count_ticks(ax, n: pd.Series) -> None:
    """Each size's task count under its tick: the population is the same for every line of the panel."""
    ax.set_xticklabels([f"{s}\n{int(k)}" if k == k else s for s, k in n.items()])


def colour(grp: str) -> str:
    return L_COLOUR.get(grp, S.MUTED)


def _thin_note(t: pd.DataFrame, what: str) -> str:
    x = t[t["thin"]]
    return (f" Not drawn (fewer than half the largest {what}; in the CSV): "
            + ", ".join(sorted({f"{r} {int(n)}" for r, n in zip(x["where"], x["n_tasks"])})) + ".") if len(x) else ""


def fig_scores(summ: pd.DataFrame, out_dir: Path, sizes: list, where: str, paper: bool = False) -> None:
    """`paper`: english_only_scores_paper, the bare rule-18 version (capitalized
    labels, no title or caption, the line key under the panels), written
    through style.save_paper with the table it draws."""
    acc = summ[summ["scoring"] == "acc"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4) if paper else (15, 4.2))
    tabs = []
    for ax, arch in zip(axes[:2], ARCHS):
        a = acc[(acc["arch"] == arch) & ~acc["thin"]]
        for comp in group_order(a["comparator"].unique()):
            g = a[a["comparator"] == comp]
            g = g.set_index("size").reindex([s for s in sizes if s in set(g["size"])])
            pooled = comp == "every L > 1"
            tabs.append(_line(ax, g.index, 100 * g["mean_gap"], f"{arch}: mean gap", comp, sizes,
                              color=S.INK if pooled else colour(comp), lw=2.2 if pooled else 1.1, label=comp))
            if pooled:
                tabs.append(_line(ax, g.index, 100 * g["median_gap"], f"{arch}: median gap", comp, sizes, color=S.INK,
                                  lw=1.2, ls=":", label=f"{comp}, median"))
                for s, v, n in zip(g.index, 100 * g["mean_gap"], g["n_tasks"]):
                    ax.annotate(str(int(n)), (sizes.index(s), v), textcoords="offset points", xytext=(0, 5), fontsize=6,
                                ha="center")
        ax.axhline(0, color=S.MUTED, lw=.8)
        ax.set_title(f"({'ab'[ARCHS.index(arch)]}) {arch.capitalize()}: L1 minus L, accuracy points" if paper else
                     f"({'ab'[ARCHS.index(arch)]}) {arch}: L1 minus L, accuracy points (mean; pooled median dotted)",
                     loc="left", fontsize=8.5)
        ax.set_ylabel(f"{'M' if paper else 'm'}ean over the English tasks above chance"); _x(ax, sizes)
    if paper:                     # the line key under the panels, one row
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=7, frameon=False, ncol=len(handles), loc="upper center",
                   bbox_to_anchor=(0.5, 0.0))
    else:
        axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    ax = axes[2]
    p = acc[acc["comparator"] == "every L > 1"]
    tabs.append(pd.DataFrame({"panel": "tasks per size (every L > 1)", "row": p["arch"], "col": p["size"],
                              "value": p["n_tasks"]}))
    for arch, ls in zip(ARCHS, ("-", "--")):
        g = p[(p["arch"] == arch) & ~p["thin"]]
        g = g.set_index("size").reindex([s for s in sizes if s in set(g["size"])])
        tabs.append(_line(ax, g.index, g["win_share"], "shares", f"{arch}: L1 ahead", sizes, color=S.INK, ls=ls,
                          label=f"{arch}: L1 ahead"))
        if arch == "deep":
            s = g.dropna(subset=["ahead_beyond_seed"])
            tabs.append(_line(ax, s.index, s["ahead_beyond_seed"], "shares", f"ahead by > {BEYOND:g} seed sd", sizes,
                              color=S.SERIES[0], ls="", label=f"deep: ahead by > {BEYOND:g} seed sd"))
            tabs.append(_line(ax, s.index, s["behind_beyond_seed"], "shares", f"behind by > {BEYOND:g} seed sd", sizes,
                              color=S.SERIES[1], ls="", label=f"deep: behind by > {BEYOND:g} seed sd"))
    ax.axhline(0.5, color=S.MUTED, lw=.8, ls=":"); ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"(c) {'S' if paper else 's'}hare of (task, L) comparisons, every L > 1 pooled", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False); _x(ax, sizes)
    if paper:
        ax.set_ylabel("Share of comparisons")
        for a in axes:
            a.set_xlabel("Model size")
        fig.tight_layout()
        pd.concat(tabs, ignore_index=True).to_csv(out_dir / "english_only_scores_paper.csv", index=False)
        S.save_paper(fig, out_dir / "english_only_scores_paper")
        return
    G.save_highlights(fig, out_dir, "Do the English-only cells score higher on the English benchmarks?",
                      f"Final checkpoint of the L1 baseline cell against the same-depth baseline cell at each other L, seed "
                      f"{GRID_SEED}, {where}; accuracy-scored English tasks above chance at the size (rule 1) that both "
                      f"cells score; small number = tasks behind the pooled line (the population moves with the size)."
                      + _thin_note(p.assign(where=p["arch"] + " " + p["size"]), "count of the line")
                      + " Seed sd = sample std of the deep L1 cell over its replicate seeds, where they exist.",
                      tabs, name="english_only_scores")


def fig_scores_fixed(g: pd.DataFrame, out_dir: Path) -> None:
    """english_only_scores_paper on one population per line (rule 13): solid,
    the (task, L) comparisons with a value at every size the line is drawn at
    (`utils.fixed_population`); dashed behind it, the paper figure's moving
    population. Panel (c)'s shares are over the same moving comparisons, so they
    get the twin too; the pooled median line is left out."""
    acc = g[g["scoring"] == "acc"].dropna(subset=["gap"]).assign(unit=lambda d: d["task"] + "|" + d["L"].astype(str))
    seed = acc.dropna(subset=["gap_over_seed"])
    pooled = "every L > 1"
    cells = []
    for arch in ARCHS:
        d = acc[acc["arch"] == arch]
        cells += [c.assign(panel=f"{arch}: mean gap", line=comp, comparator=comp, value=100 * c["gap"])
                  for comp, c in [(label(L), d[d["L"] == L]) for L in sorted(d["L"].unique())] + [(pooled, d)]]
        cells.append(d.assign(panel="shares", line=f"{arch}: L1 ahead", comparator=pooled, value=(d["gap"] > 0) * 1.0))
    s = seed[seed["arch"] == "deep"]
    cells += [s.assign(panel="shares", line=f"{w} by > {BEYOND:g} seed sd", comparator=pooled, value=v * 1.0)
              for w, v in (("ahead", s["gap_over_seed"] > BEYOND), ("behind", s["gap_over_seed"] < -BEYOND))]
    summ = score_summary(g)
    drawn = summ.loc[(summ["scoring"] == "acc") & ~summ["thin"], ["arch", "comparator", "size"]]      # fig_scores' points
    sizes = size_order(g["size"].unique())
    cells = pd.concat(cells).merge(drawn, on=["arch", "comparator", "size"])
    cells["size"] = pd.Categorical(cells["size"], sizes, ordered=True)
    tab = pd.concat([d.groupby(["panel", "line", "size"], observed=True).agg(value=("value", "mean"), n_tasks=("task", "nunique"))
                     .reset_index().assign(reading=rd)
                     for rd, d in (("moving", cells), ("fixed", fixed_population(cells, ["panel", "line"], "size", "value",
                                                                                 task="unit")))]).rename(columns={"size": "x"})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    def draw(ax, panel, line, colour, fixed_kw):
        for rd in ("moving", "fixed"):
            t = tab[(tab["panel"] == panel) & (tab["line"] == line) & (tab["reading"] == rd)].set_index("x").reindex(sizes)
            kw = fixed_kw if rd == "fixed" else dict(ls=(0, (3, 2)) if fixed_kw.get("ls") != "" else "", lw=.9, alpha=.6,
                                                     marker=".", ms=3, zorder=1.8)
            ax.plot(range(len(sizes)), t["value"], color=colour, **({"marker": "o", "ms": 3.5} | kw))
        return t
    for ax, arch in zip(axes[:2], ARCHS):
        panel = f"{arch}: mean gap"
        for comp in group_order(tab.loc[tab["panel"] == panel, "line"].unique()):
            t = draw(ax, panel, comp, S.INK if comp == pooled else colour(comp),
                     dict(lw=2.2 if comp == pooled else 1.1, label=comp))
            if comp == pooled and t["value"].notna().any():
                last = t["value"].last_valid_index()
                ax.annotate(f"{int(t.loc[last, 'n_tasks'])} tasks", (sizes.index(last), t.loc[last, "value"]),
                            textcoords="offset points", xytext=(0, 5), fontsize=6, ha="center")
        ax.axhline(0, color=S.MUTED, lw=.8)
        ax.set_title(f"({'ab'[ARCHS.index(arch)]}) {arch.capitalize()}: L1 minus L, accuracy points", loc="left", fontsize=8.5)
        ax.set_ylabel("Mean over the English tasks above chance"); _x(ax, sizes)
    axes[0].plot([], [], color=S.MUTED, ls=(0, (3, 2)), lw=.9, label="Tasks above chance at each size")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 0.0))
    ax = axes[2]
    for arch, c in zip(ARCHS, (S.INK, S.MUTED)):
        draw(ax, "shares", f"{arch}: L1 ahead", c, dict(label=f"{arch}: L1 ahead"))
    for w, c in (("ahead", S.SERIES[0]), ("behind", S.SERIES[1])):
        draw(ax, "shares", f"{w} by > {BEYOND:g} seed sd", c, dict(ls="", label=f"deep: {w} by > {BEYOND:g} seed sd"))
    ax.axhline(0.5, color=S.MUTED, lw=.8, ls=":"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("(c) Share of (task, L) comparisons, every L > 1 pooled", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False); _x(ax, sizes); ax.set_ylabel("Share of comparisons")
    for a in axes:
        a.set_xlabel("Model size")
    fig.tight_layout()
    tab[["reading", "panel", "line", "x", "value", "n_tasks"]].to_csv(out_dir / "english_only_scores_fixed_tasks_paper.csv",
                                                                       index=False)
    S.save_paper(fig, out_dir / "english_only_scores_fixed_tasks_paper")


def fig_scores_by_benchmark(g: pd.DataFrame, out_dir: Path, sizes: list, where: str) -> None:
    d = g[(g["scoring"] == "acc") & (g["arch"] == "deep")]
    rows = sorted(d["benchmark"].unique())
    gap = d.pivot_table(index="benchmark", columns="size", values="gap", aggfunc="mean").reindex(index=rows, columns=sizes)
    win = d.assign(win=(d["gap"] > 0).where(d["gap"].notna())).pivot_table(
        index="benchmark", columns="size", values="win", aggfunc="mean").reindex(index=rows, columns=sizes)
    cnt = d.dropna(subset=["gap"]).pivot_table(index="benchmark", columns="size", values="task", aggfunc="nunique") \
        .reindex(index=rows, columns=sizes)
    grey = d.pivot_table(index="benchmark", columns="size", values="gated", aggfunc="all").reindex(index=rows, columns=sizes)
    grey = grey.fillna(False).astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.3 * len(rows) + 2.2), sharey=True)
    tabs = [G.matrix_ax(axes[0], 100 * gap, "mean gap, L1 minus L (accuracy points)", cnt=cnt, vmin=-10, vmax=10,
                        center=0, cmap=S.DIV, fmt="{:+.1f}", gated=grey),
            G.matrix_ax(axes[1], win, "share of (task, L) comparisons L1 wins", cnt=cnt, vmin=0, vmax=1, center=0.5,
                        cmap=S.DIV, fmt="{:.2f}", gated=grey)]
    G.save_highlights(fig, out_dir, "Per English benchmark: where the English-only deep cell is ahead",
                      f"Deep baseline cells, seed {GRID_SEED}, {where}, every L > 1 pooled; accuracy-scored English "
                      "tasks (twins as their own row); small number = tasks behind the cell; grey = every task of the row "
                      "at chance at that size (rule 1), white = no value.", tabs, name="english_only_scores_by_benchmark")


def fig_gate(share: pd.DataFrame, cross: pd.DataFrame, out_dir: Path, sizes: list, where: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    tabs = []
    n = share.groupby("size")["n_tasks"].first().reindex(sizes).dropna()
    thin_sizes = sorted(set(share.loc[share["thin"], "size"]), key=sizes.index)
    tabs.append(pd.DataFrame({"panel": "tasks per size", "row": "n_tasks", "col": n.index, "value": n.to_numpy()}))
    for ax, arch in zip(axes[:2], ARCHS):
        for (fam, L, build), g in share[share["ladder"] == arch].groupby(["family", "L", "build"]):
            tabs.append(pd.DataFrame({"panel": f"{arch}: share above chance", "row": label(L) + (f" {build}" if build else ""),
                                      "col": g["size"], "value": g["share"]}))
            g = g[~g["size"].isin(thin_sizes)]
            g = g.set_index("size").reindex([s for s in sizes if s in set(g["size"])])
            name = label(L) + (f" {build}" if build else "")
            _line(ax, g.index, g["share"], "", name, sizes, color=colour(label(L)), ls="-" if not build else "--",
                  lw=2.2 if L == 1 and not build else 1.1, label=name)
        ax.set_ylim(-0.02, 1.02); ax.set_title(f"({'ab'[ARCHS.index(arch)]}) {arch}: share of English tasks above chance",
                                               loc="left", fontsize=8.5)
        ax.set_ylabel("share of the cell's tasks with a chance level"); _x(ax, sizes)
    axes[0].legend(fontsize=6.5, frameon=False, ncol=2)
    ax = axes[2]
    c = cross.assign(row=cross["arch"] + " vs " + cross["comparator"])
    left = np.zeros(len(c))
    for col, colr in (("earlier", S.SERIES[0]), ("same", S.NODATA), ("later", S.SERIES[1])):
        ax.barh(range(len(c)), c[col].fillna(0), left=left, color=colr, label=f"L1 {col}", height=0.75)
        left += c[col].fillna(0).to_numpy()
        tabs.append(pd.DataFrame({"panel": "crossing", "row": c["row"], "col": col, "value": c[col]}))
    tabs.append(pd.DataFrame({"panel": "crossing", "row": c["row"], "col": "n_tasks", "value": c["n_tasks"]}))
    ax.set_yticks(range(len(c))); ax.set_yticklabels([f"{r} ({n})" for r, n in zip(c["row"], c["n_tasks"])], fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlim(0, 1); S.clean(ax)
    ax.set_title("(c) which cell clears chance from a smaller size", loc="left", fontsize=8.5)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    G.save_highlights(fig, out_dir, "Does the English-only cell clear chance earlier on the English benchmarks?",
                      f"Run-level verdict of the above-random gate (one-sided 95 % Wilson bound over the task's items above "
                      f"chance), final checkpoint, seed {GRID_SEED}, {where}; English tasks with a chance level (bBPB "
                      f"and BPB have none) that every drawn cell of the size scores: "
                      + ", ".join(f"{s} {int(k)}" for s, k in n.items())
                      + (f" (not drawn below half the largest: {', '.join(thin_sizes)}; in the CSV)" if thin_sizes else "")
                      + ". (c): per task, the smallest size from which the run stays above chance, L1 "
                      f"against the same-depth baseline cell at each L; tasks neither cell ever clears are left out "
                      f"(count in brackets).", tabs, name="english_only_above_random")


def fig_scaling(sle: pd.DataFrame, fits: pd.DataFrame, out_dir: Path, where: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    tabs = []
    ax = axes[0]
    tops = size_order(sle["ladder_top"].unique()) if len(sle) else []
    marks = dict(zip(sorted(sle["build"].unique()), "os^Dv"))          # one marker per data build (baseline first)
    for key, g in sorted(sle.groupby("chain"), key=lambda kv: (int(kv[1]["L"].iloc[0]), kv[0])):
        g = g.set_index("ladder_top").reindex([t for t in tops if t in set(g["ladder_top"])])
        L = int(g["L"].iloc[0])
        tabs.append(_line(ax, g.index, 100 * g["rel_error"].abs(), "English BPB error", key, tops, color=colour(label(L)),
                          ls="-" if g["ladder"].iloc[0] == "deep" else "--", lw=2.2 if L == 1 else 1.0,
                          marker=marks[g["build"].iloc[0]], label=key))
    ax.set_ylabel(f"|predicted − observed| / observed of the {TARGET_SIZE} BPB (%)")
    ax.set_xlabel("largest proxy rung in the fit"); ax.legend(fontsize=5.8, frameon=False, ncol=2)
    ax.set_title(f"(a) power-law prediction of the {TARGET_SIZE} English BPB (L1 thick; shallow dashed)", loc="left",
                 fontsize=8.5)
    _x(ax, tops)
    ax = axes[1]
    med = fits.groupby("L").agg(r2=("r2", "median"), rho=("rho", "median"), n=("r2", "count")).reset_index()
    ax.bar(range(len(med)), med["r2"], color=[colour(label(L)) for L in med["L"]])
    for i, (v, n) in enumerate(zip(med["r2"], med["n"])):
        ax.text(i, v, f"{v:.2f}\n{n}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(med))); ax.set_xticklabels([label(L) for L in med["L"]]); ax.set_ylim(0, 1.1)
    ax.set_ylabel(f"median R² of score vs log10 N, {TARGET_SIZE} included"); S.clean(ax)
    ax.set_title("(b) English accuracy tasks: how well a log-linear fit in size holds (n = fits)", loc="left", fontsize=8.5)
    tabs.append(pd.DataFrame({"panel": "benchmark R2", "row": med["L"].map(label), "col": "median r2", "value": med["r2"]}))
    tabs.append(pd.DataFrame({"panel": "benchmark R2", "row": med["L"].map(label), "col": "n fits", "value": med["n"]}))
    G.save_highlights(fig, out_dir, "Is the English-only ladder at least as predictable in size?",
                      f"Seed {GRID_SEED}, {where}. (a) log BPB = a − α log N fitted on the proxy rungs up to the x rung, "
                      f"error of the {TARGET_SIZE} prediction ({ENGLISH_BPB}, the English validation set — the baseline build's "
                      f"corpus, off-distribution for the DCLM-without-edu and FineWeb cells). (b) deep baseline cells, the "
                      f"rungs the L1 deep cell has (every L fits the same ones), above chance only (rule 1, ≥ 3 rungs per "
                      f"fit), the {TARGET_SIZE} reference included: a reference-size quantity describing the whole ladder, "
                      f"not a prediction of {TARGET_SIZE} from below (rule 11; (a) is that).", tabs,
                      name="english_only_scaling")


def fig_da(out: pd.DataFrame, groups: dict, out_dir: Path, axes_name: str, where: str) -> None:
    proxies = list(SMALL_SIZES)
    pops = ("all benchmarks", "bpb")
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    tabs = []
    for ax, pop in zip(axs, pops):
        d = out[(out["population"] == pop) & out["size"].isin(proxies)]
        for grp in group_order(d["group"].unique()):
            g = d[(d["group"] == grp) & ~d["thin"]]
            g = g.set_index("size").reindex([s for s in proxies if s in set(g["size"])])
            x = [proxies.index(s) for s in g.index]
            c = colour(grp)
            ax.fill_between(x, g["lo"].clip(0, 1), g["hi"].clip(0, 1), color=c, alpha=.10, lw=0)
            tabs.append(_line(ax, g.index, g["reliability"], pop, grp, proxies, color=c, lw=2.2 if grp == "L1" else 1.2,
                              label=f"{grp} ({len(groups[int(grp[1:])])} families)"))
            tabs += [pd.DataFrame({"panel": pop, "row": f"{grp} {b}", "col": g.index, "value": g[b].to_numpy()})
                     for b in ("lo", "hi", "n_comparable")]
        n = d.groupby("size")["n_tasks"].max().reindex(proxies)
        tabs.append(pd.DataFrame({"panel": pop, "row": "tasks (every group)", "col": n.index, "value": n.to_numpy()}))
        ax.axhline(0.5, color=S.MUTED, lw=.8, ls=":"); ax.axhline(RELIABLE_DA, color=S.MUTED, lw=.6, ls="--")
        ax.set_ylim(0.2, 1.02); ax.set_title(f"({'ab'[pops.index(pop)]}) " + ("English benchmarks" if pop != "bpb" else
                                             f"English BPB ({ENGLISH_BPB})"), loc="left", fontsize=8.5)
        _x(ax, proxies); _count_ticks(ax, n); ax.set_xlabel("proxy size (tasks, the same for every group)")
    axs[0].set_ylabel(f"DA-size against the {TARGET_SIZE} finals (pooled)"); axs[0].legend(fontsize=6.5, frameon=False)
    fams = "; ".join(f"{label(L)}: " + ", ".join(build_token(f) or f.split("-")[2] for f in fs) for L, fs in groups.items())
    t = out[out["size"].isin(proxies)].assign(where=lambda x: x["population"] + " " + x["size"]).drop_duplicates("where")
    G.save_highlights(fig, out_dir, "Do the English-only families rank like their reference at least as well?",
                      f"Decisions = {axes_name} pairs among each L's families ({fams}; baseline build named by its ladder), "
                      f"proxy final against the {TARGET_SIZE} final, pooled over the English tasks above chance at the proxy "
                      f"and the reference ({where}, rule 1) that every family of every group has (one population per size; "
                      f"rule 5's ≥ {MIN_PAIRS} pairs holds); band = 90 % leave-one-family-out jackknife; dotted = chance, "
                      f"dashed = τ = {RELIABLE_DA:g}. The groups' decisions differ (see the README)."
                      + _thin_note(t, "count"), tabs, name=f"english_only_da_size_by_L{AXES_SUFFIX[axes_name]}")


def fig_snr(snr: pd.DataFrame, paired: pd.DataFrame, sur: pd.DataFrame, out_dir: Path, sizes: list, where: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    tabs = []
    ax = axes[0]
    for fam, ls in ((MIRROR, "-"), (DEPTH, "--")):
        p = paired[paired["families"] == fam]
        for grp in group_order(p["group"].unique()):
            g = p[(p["group"] == grp) & ~p["thin"]]
            g = g.set_index("size").reindex([s for s in sizes if s in set(g["size"])])
            tabs.append(_line(ax, g.index, g["median_log10_snr"], f"median log10 SNR, {FAMILIES[fam]}", grp, sizes,
                              color=colour(grp), ls=ls, lw=2.2 if grp == "L1" else 1.0,
                              label=grp + ("" if fam == MIRROR else " deep–shallow")))
        n = p.groupby("size")["n_tasks"].max()
        tabs.append(pd.DataFrame({"panel": f"tasks, {FAMILIES[fam]}", "row": "n_tasks", "col": n.index, "value": n.to_numpy()}))
    ax.set_ylabel(f"median log10 SNR ({VARIANT}) over the paired tasks"); ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.set_title("(a) SNR per group: mirrored families solid, deep–shallow dashed", loc="left", fontsize=8.5); _x(ax, sizes)
    ax = axes[1]
    at = SMALL_SIZES[-1]
    s = snr[(snr["families"] == DEPTH) & snr["all_families"]]
    w = s[(s["size"] == at) & (s["task"] != ENGLISH_BPB)].pivot_table(index="task", columns="group", values="log10_snr")
    if "L1" in w:
        o = w.drop(columns="L1").median(axis=1)
        ok = w["L1"].notna() & o.notna()
        ax.scatter(o[ok], w.loc[ok, "L1"], s=10, color=colour("L1"), alpha=.7)
        lim = [np.nanmin([o[ok].min(), w.loc[ok, "L1"].min()]) - .1, np.nanmax([o[ok].max(), w.loc[ok, "L1"].max()]) + .1] \
            if ok.any() else [0, 1]
        ax.plot(lim, lim, color=S.MUTED, lw=.8, ls=":")
        tabs.append(pd.DataFrame({"panel": f"tasks at {at}, deep–shallow", "row": w.index[ok], "col": "L1 minus other L median",
                                  "value": (w.loc[ok, "L1"] - o[ok]).to_numpy()}))
        ax.set_title(f"(b) deep–shallow, per task at {at}: above the diagonal = L1 higher ({int(ok.sum())} tasks)",
                     loc="left", fontsize=8.5)
    ax.set_xlabel("log10 SNR, median over the other L's"); ax.set_ylabel("log10 SNR, L1"); S.clean(ax)
    ax = axes[2]
    sm = sur[(sur["axes"] == "multi-axis") & ~sur["thin"]] if len(sur) else sur
    for grp in group_order(sm["group"].unique()) if len(sm) else []:
        g = sm[sm["group"] == grp]
        g = g.set_index("size").reindex([x for x in SMALL_SIZES if x in set(g["size"])])
        tabs.append(_line(ax, g.index, g["rho"], "rho(SNR, DA-size)", grp, list(SMALL_SIZES), color=colour(grp),
                          lw=2.2 if grp == "L1" else 1.2, label=grp))
        tabs.append(pd.DataFrame({"panel": "rho(SNR, DA-size)", "row": f"{grp} n_tasks", "col": g.index,
                                  "value": g["n_tasks"].to_numpy()}))
    if not len(sm):
        ax.text(0.5, 0.5, "no decision accuracy in this pool", ha="center", va="center", transform=ax.transAxes, fontsize=8)
    ax.axhline(0, color=S.MUTED, lw=.8); ax.set_ylim(-1, 1); ax.set_ylabel("Spearman ρ over the English tasks")
    ax.set_title("(c) does SNR at the proxy predict DA-size there? (mirrored families)", loc="left", fontsize=8.5)
    _x(ax, list(SMALL_SIZES))
    t = paired.assign(where=lambda x: x["size"] + np.where(x["families"] == DEPTH, " deep–shallow", " mirrored"))
    G.save_highlights(fig, out_dir, "Do the English benchmarks have a higher SNR among the English-only families?",
                      f"SNR = {VARIANT} (the noise-and-SNR definition): spread of the final scores across a group's families "
                      f"over the checkpoint noise in the last 20 % of each run (rule 4); at-chance cells blank (rule 1; "
                      f"{where}). Groups: the families of the decision-accuracy figure (decisions differ by L), and the deep "
                      f"and shallow baseline cells of every L (the one shared decision). (a) tasks with an SNR from every "
                      f"family of every group of the set at that size (counts in the CSV); (c) multi-axis DA-size on the "
                      f"tasks the DA lines read." + _thin_note(t, "count of the line"), tabs, name="english_only_snr")


def fig_allenai(t: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 0.45 * t["task"].nunique() + 2.4))
    tabs = []
    for ax, (q, title, vals) in zip(axes, (("da_size", "DA-size", lambda v: v),
                                           ("snr", f"log10 SNR ({VARIANT})", lambda v: np.log10(v.where(v > 0))))):
        d = t[t["quantity"] == q]
        cols, grey = {}, {}
        for (s, a), g in d.groupby(["size", "allenai_size"], sort=False):
            g = g.set_index("task")
            cols[f"L1 {s}"], cols[f"other L {s}"], cols[f"DataDecide {a}"] = vals(g["l1"]), vals(g["other_l_median"]), \
                vals(g["allenai"])
            grey[f"L1 {s}"], grey[f"other L {s}"], grey[f"DataDecide {a}"] = g["gated"], g["gated"], g["gated"] & False
        mat, gmat = pd.DataFrame(cols), pd.DataFrame(grey)
        mat.index = gmat.index = [f"{ALLENAI_TASKS[o]} ↔ {o}" for o in mat.index]   # DataDecide's name first
        lo, hi = (0.3, 1.0) if q == "da_size" else (-0.5, 1.5)
        tabs.append(G.matrix_ax(ax, mat, title, vmin=lo, vmax=hi, fmt="{:.2f}", gated=gmat))
        ax.tick_params(axis="x", labelrotation=60)
    G.save_highlights(fig, out_dir, "The English-only cells next to AllenAI DataDecide on the shared English tasks",
                      f"Ours: L1's families (and the median over the other L groups), seed {GRID_SEED}, DA-size against "
                      f"{TARGET_SIZE}, multi-axis pairs, cells with every family of the group; DataDecide: 25 data recipes, "
                      f"DA-size against {ALLENAI_REF}. Tasks matched by format (cloze ↔ OLMES RC, letter ↔ :mc); sizes "
                      f"matched as the external frameworks comparison does. Grey = at chance on our side (rule 1; DataDecide "
                      f"is not gated), white = no value. Definitions differ (see the README): read levels, not decimals; "
                      f"the SNRs are not comparable.", tabs, name="english_only_allenai")


# --- 8. the verdict ------------------------------------------------------------------

def tally(d: pd.Series) -> tuple[int, int, int]:
    """(ahead, tied, behind) over signed differences in L1's favour; within TIE is a tie."""
    d = d.dropna()
    return int((d > TIE).sum()), int((d.abs() <= TIE).sum()), int((d < -TIE).sum())


def lead(wide: pd.DataFrame, higher: bool = True) -> tuple[int, int, int]:
    """`tally` of L1 against every other column of `wide`, in every row (a
    size, or one summary row)."""
    return tally((wide.drop(columns="L1").rsub(wide["L1"], axis=0) * (1 if higher else -1)).stack())


def grade(ahead: int, tied: int, behind: int) -> str:
    return "yes" if ahead and not behind else "partly" if ahead else "tie" if tied and not behind else "no"


def counts(a: int, t: int, b: int, unit: str) -> str:
    return f"ahead {a}, tied {t}, behind {b} of {a + t + b} {unit}"


def _means(wide: pd.DataFrame) -> str:
    m = wide.drop(columns="L1").mean()
    return ", ".join(f"{g} {fmt(m[g])}" for g in group_order(m.index))


def verdict(summ, cross, share, sle, fits, da, mix, paired, al, missing: str) -> pd.DataFrame:
    rows = []
    p = summ[~summ["thin"]]
    for scoring, what, unit, scale in (("acc", "higher English accuracy", "pp", 100), ("bpb", "lower English BPB", "bits", 1)):
        for arch in ARCHS:
            a = p[(p["scoring"] == scoring) & (p["arch"] == arch)]
            if len(a):
                ref = a[(a["size"] == TARGET_SIZE) & (a["comparator"] == "every L > 1")]
                n = tally(a.loc[a["comparator"] != "every L > 1", "win_share"] - 0.5)
                rows.append({"expectation": f"{what} ({arch})",
                             "measure": "share of the tasks L1 wins against each L at each size (thin sizes left out), "
                                        "against one half; mean gap at the reference, every L pooled (positive = L1 better)",
                             "l1": counts(*n, "(size, L) cells"),
                             "comparator": f"{fmt(scale * ref['mean_gap'].iloc[0], 2 if scale == 1 else 1)} {unit} at "
                                           f"{TARGET_SIZE}" if len(ref) else "",
                             "verdict": grade(*n)})
    if len(cross):
        c = cross[cross["arch"] == "deep"]
        per = int((c["earlier"] > c["later"]).sum())
        e, l = (c["earlier"] * c["n_tasks"]).sum(), (c["later"] * c["n_tasks"]).sum()
        sh = share[(share["build"] == "") & ~share["thin"] & (share["ladder"] == "deep")] \
            .pivot_table(index="size", columns="L", values="share").rename(columns=label)
        a, t, b = lead(sh) if "L1" in sh else (0, 0, 0)
        rows.append({"expectation": "clears chance earlier (deep)",
                     "measure": "comparators against which L1 stays above chance from a smaller size on more tasks than "
                                "from a larger one; L1's share of English tasks above chance against each L, per size",
                     "l1": f"{per} of {len(c)} comparators ({int(round(e))} earlier, {int(round(l))} later, (task, L) pooled)",
                     "comparator": "share: " + counts(a, t, b, "(size, L) cells"),
                     "verdict": "yes" if per == len(c) and not b else "partly" if per or a else "no"})
    if len(sle):
        top = sle[(sle["ladder_top"] == SMALL_SIZES[-1]) & (sle["build"] == "")].assign(err=lambda d: d["rel_error"].abs())
        m = top.groupby("L")["err"].median().rename(label).to_frame().T
        if "L1" in m:
            a, t, b = lead(100 * m, higher=False)              # in percentage points of error: TIE = 0.01 pp
            rows.append({"expectation": "English BPB at least as predictable in size",
                         "measure": f"median absolute relative error of the {TARGET_SIZE} prediction from {SMALL_SIZES[0]}–"
                                    f"{SMALL_SIZES[-1]} (baseline build, deep and shallow), L1 against each L",
                         "l1": f"{fmt(m['L1'].iloc[0] * 100, 1)} %",
                         "comparator": ", ".join(f"{k} {fmt(100 * v, 1)} %" for k, v in m.drop(columns="L1").iloc[0].items()),
                         "verdict": grade(a, t, b)})
    if len(fits):
        m = fits.groupby("L")["r2"].median().rename(label).to_frame().T
        if "L1" in m:
            a, t, b = lead(m)
            rows.append({"expectation": "English accuracy at least as predictable in size",
                         "measure": f"median R² of score vs log10 N (deep; the rungs the L1 deep cell has, {TARGET_SIZE} "
                                    "included: a reference-size quantity, rule 11), L1 against each L",
                         "l1": fmt(m["L1"].iloc[0]),
                         "comparator": ", ".join(f"{k} {fmt(v)}" for k, v in m.drop(columns="L1").iloc[0].items()),
                         "verdict": grade(a, t, b)})
    if missing:
        why = "not computed: " + missing.split(":")[0]
        rows += [{"expectation": e, "measure": why, "l1": "", "comparator": "", "verdict": "not computed"}
                 for e in ("higher decision accuracy (mirrored families)", "higher SNR (mirrored families)",
                           "L1 DA-size in DataDecide's range", "L1 SNR next to DataDecide")]
    else:
        d = da[(da["axes"] == "multi-axis") & (da["population"] == "all benchmarks") & da["size"].isin(SMALL_SIZES)
               & ~da["thin"]].pivot_table(index="size", columns="group", values="reliability")
        if "L1" in d:
            dp = mix[(mix["axes"] == "multi-axis") & (mix["population"] == "all benchmarks") & (mix["axis"] == "arch")]
            rows.append({"expectation": "higher decision accuracy (mirrored families)",
                         "measure": "pooled DA-size per proxy, multi-axis, " + counts(*lead(d), "(size, L) cells")
                                    + "; the decisions differ by L (at L1 three of six pairs move the English corpus)",
                         "l1": "mean " + fmt(d["L1"].mean()),
                         "comparator": _means(d)
                                       + "; on the depth pairs alone: "
                                       + ", ".join(f"{r.group} {fmt(r.da)}" for r in dp.set_index("group")
                                                   .loc[group_order(dp["group"])].reset_index().itertuples()),
                         "verdict": "not like-for-like"})
        s = paired[(paired["families"] == MIRROR) & ~paired["thin"]].pivot_table(index="size", columns="group",
                                                                                  values="median_log10_snr")
        if "L1" in s:
            rows.append({"expectation": "higher SNR (mirrored families)",
                         "measure": "median log10 SNR over the paired tasks per size, " + counts(*lead(s), "(size, L) cells")
                                    + "; the families' differences differ by L",
                         "l1": "mean " + fmt(s["L1"].mean()),
                         "comparator": _means(s),
                         "verdict": "not like-for-like"})
    s = paired[(paired["families"] == DEPTH) & ~paired["thin"]].pivot_table(index="size", columns="group",
                                                                             values="median_log10_snr")
    if "L1" in s:
        a, t, b = lead(s)
        rows.append({"expectation": "higher SNR (deep against shallow, the shared decision)",
                     "measure": "median log10 SNR over the paired tasks per size, " + counts(a, t, b, "(size, L) cells"),
                     "l1": "mean " + fmt(s["L1"].mean()),
                     "comparator": _means(s),
                     "verdict": grade(a, t, b)})
    if not missing and al is not None and len(al):
        x = al[al["quantity"] == "da_size"].dropna(subset=["l1"])
        if len(x):
            pos = x["l1_vs_allenai_range"].value_counts()
            ratio = x["l1_over_allenai"].median()
            ok = pos.get("below", 0) * 2 <= len(x)
            rows.append({"expectation": "L1 DA-size in DataDecide's range",
                         "measure": "(task, size) cells below / inside / above DataDecide's min–max over the shared tasks "
                                    "where L1 has a value at the matched size; median L1 / DataDecide on the same task "
                                    "(ours gated, decisions differ: see the README)",
                         "l1": " / ".join(str(int(pos.get(k, 0))) for k in ("below", "inside", "above")) + f" of {len(x)}",
                         "comparator": f"ratio {fmt(ratio)}",
                         "verdict": "yes" if ok and 0.5 <= ratio <= 2 else "partly" if ok else "no"})
        x = al[al["quantity"] == "snr"].dropna(subset=["l1"])
        if len(x):
            rows.append({"expectation": "L1 SNR next to DataDecide",
                         "measure": "not comparable: DataDecide's signal is the spread of 25 recipes, ours of L1's "
                                    "families; the noise uses different checkpoint windows (the like-for-like reading is "
                                    "the deep-against-shallow row above)",
                         "l1": f"{len(x)} cells", "comparator": f"ratio {fmt(x['l1_over_allenai'].median())} (shown, "
                                                                "not graded)",
                         "verdict": "not comparable"})
    return pd.DataFrame(rows)


# --- the README -------------------------------------------------------------------

def _ranges(x: pd.Series, p: int = 2) -> str:
    return f"{fmt(x.min(), p)}–{fmt(x.max(), p)}"


def readme(pool: str, gate_pool: str, rel: str, r: dict) -> None:
    path = ENGLISH_ONLY / "README.md"
    gen = f"english_only.py --pool {pool}"
    gh = f"https://github.com/mariagrandury/snr-multilingual/blob/main/src/signal-and-noise/analysis/{ENGLISH_ONLY.name}/{rel}"
    snap = datetime.fromtimestamp((ladder_dir() / "ladder_report.csv").stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    g, summ, cross, share, sle, fits = r["g"], r["summ"], r["cross"], r["share"], r["sle"], r["fits"]
    da, cells, mix, paired, sur, al, ver = r["da"], r["cells"], r["mix"], r["paired"], r["sur"], r["al"], r["ver"]
    groups, missing, sizes = r["groups"], r["missing"], r["sizes"]
    rule5 = f"**!!! RULE 5:** {missing}"

    def links(*names):
        return " · ".join(f"[{n}]({gh}/{n})" for n in names)

    # --- setup
    fams = "; ".join(f"{label(L)} — " + ", ".join(f"`{f}`" for f in fs) for L, fs in groups.items())
    left = ", ".join(f"{label(L)} ({len(f)})" for L, f in r["all_groups"].items() if L not in groups)
    lines = [f"Pool `{pool}`, ladder-report snapshot **{snap}**; seed {GRID_SEED} for every score, the replicate seeds of "
             f"`{SEEDS_POOL}` for the seed yardstick; gate `{gate_pool}`. {r['population']}", "",
             f"- **Mirrored family groups** (decision accuracy and the first SNR reading): {fams}."
             + (f" Left out, with another number of such families: {left}." if left else ""),
             "- **Deep against shallow** (the second SNR reading): the two baseline cells at "
             + ", ".join(label(L) for L in r["depth"]) + "."]
    if r["coverage"]:
        lines.append("- **Coverage that moves the populations**: " + "; ".join(r["coverage"]) + ". A decision-accuracy or "
                     "SNR cell missing a family of its group is left out of the lines (it stays in the per-task CSVs), so "
                     "a gap in one L1 family removes that (task, size) from every group's line.")
    if len(cells):
        k = cells[cells["shared"] & cells["da"].notna() & cells["size"].isin(SMALL_SIZES) & (cells["axes"] == "multi-axis")
                  & (cells["population"] == "all benchmarks")]
        n = k.drop_duplicates(["task", "size"]).groupby(["size", "scoring"]).size().unstack(fill_value=0) \
            .reindex(index=list(SMALL_SIZES), columns=["acc", "bbpb"]).fillna(0)
        th = da[(da["axes"] == "multi-axis") & (da["population"] == "all benchmarks") & da["thin"]
                & da["size"].isin(SMALL_SIZES)]["size"].unique()
        lines.append("- **The decision-accuracy population** (multi-axis; English benchmarks above chance at the proxy and "
                     "the reference that every family of every group has): " + ", ".join(
                         f"{s} {int(row.sum())} (" + ", ".join(f"{int(v)} {c}" for c, v in row.items()) + ")"
                         for s, row in n.iterrows())
                     + (f"; thin, so neither drawn nor quoted: {', '.join(th)}" if len(th) else "") + ".")
    if missing:
        lines += ["", rule5]
    replace_block(path, "english-setup", "\n".join(lines), gen)

    # --- scores
    a = summ[(summ["scoring"] == "acc") & (summ["comparator"] == "every L > 1")]
    bullets = []
    for arch in ARCHS:
        x = a[a["arch"] == arch].set_index("size")
        if x.empty:
            continue
        x = x.loc[size_order(x.index)]
        xs = x[~x["thin"]]
        bullets.append(f"- **{arch}**: L1 ahead in {_ranges(xs['win_share'])} of the (task, L) comparisons per size; mean "
                       f"gap {', '.join(f'{s} {fmt(100 * v, 1)}' for s, v in xs['mean_gap'].items())} accuracy points "
                       f"(positive = L1 better; tasks {', '.join(f'{s} {int(v)}' for s, v in x['n_tasks'].items())}"
                       + (f"; {', '.join(x.index[x['thin']])} thin, left out" if x["thin"].any() else "") + ").")
    s = a[(a["arch"] == "deep") & ~a["thin"]].dropna(subset=["ahead_beyond_seed"])
    s = s.set_index("size").loc[size_order(s["size"])].reset_index()
    if len(s):
        bullets.append(f"- **Against the seed noise** (deep, where replicates exist): L1 ahead by more than {BEYOND:g} seed sd "
                       f"in {', '.join(f'{r_.size} {fmt(r_.ahead_beyond_seed)}' for r_ in s.itertuples())} of the "
                       f"comparisons, behind by as much in {', '.join(f'{r_.size} {fmt(r_.behind_beyond_seed)}' for r_ in s.itertuples())}.")
    m = g[(g["scoring"] == "acc") & (g["arch"] == "deep")].groupby(["task", "size"])["gap"].mean()
    w = m[m < -DEFICIT].sort_values()
    if len(w):
        bullets.append(f"- **Where L1 is far behind** (deep, mean over every other L below −{100 * DEFICIT:g} pp): "
                       + ", ".join(f"`{t}` at {sz} ({fmt(100 * v, 0)} pp)" for (t, sz), v in w.items())
                       + ". A letter-format task can pass the gate on a constant answer that matches the majority gold "
                       "label (rule 1 does not test for it), so a cell where every L but one sits far above chance and "
                       "the other near zero reads a letter preference, not English skill; the medians are "
                       + ", ".join(f"{sz} {fmt(100 * v, 1)}" for sz, v in a[(a["arch"] == "deep") & ~a["thin"]]
                                   .set_index("size")["median_gap"].reindex(size_order(a["size"].unique())).dropna().items())
                       + " pp.")
    ebpb = summ[(summ["scoring"] == "bpb") & (summ["comparator"] == "every L > 1") & (summ["arch"] == "deep")]
    if len(ebpb):
        x = ebpb.set_index("size").reindex(size_order(ebpb["size"].unique()))
        bullets.append(f"- **English BPB** ({ENGLISH_BPB}, deep): L1 lower than the other L's in "
                       f"{_ranges(x['win_share'])} of the comparisons per size, by "
                       f"{', '.join(f'{s_} {fmt(v, 3)}' for s_, v in x['mean_gap'].items())} bits per byte "
                       f"(L1 at {TARGET_SIZE}: {fmt(x.at[TARGET_SIZE, 'mean_l1'], 3)} against "
                       f"{fmt(x.at[TARGET_SIZE, 'mean_other'], 3)}).")
    bb = summ[(summ["scoring"] == "bbpb") & (summ["comparator"] == "every L > 1") & (summ["arch"] == "deep")]
    if len(bb):
        x = bb.set_index("size")
        x = x.loc[size_order(x.index)]
        xs = x[~x["thin"]]
        line = (f"- **bBPB** (deep, lower is better, oriented): L1 ahead in {_ranges(xs['win_share'])} of the comparisons "
                f"per size; mean gap {', '.join(f'{s_} {fmt(v, 3)}' for s_, v in xs['mean_gap'].items())} bits, median "
                f"{', '.join(f'{s_} {fmt(v, 3)}' for s_, v in xs['median_gap'].items())} (positive = L1 better"
                + (f"; {', '.join(x.index[x['thin']])} thin: {', '.join(str(int(v)) for v in x.loc[x['thin'], 'n_tasks'])} "
                   f"tasks" if x["thin"].any() else "") + ").")
        sp = spikes(g, sizes)
        if sp:
            at, tasks, n_o, n_all = sp
            b1 = g[(g["scoring"] == "bbpb") & (g["arch"] == "deep") & (g["size"] == at)]
            fam = b1[b1["task"].isin(tasks)].drop_duplicates("task")["benchmark"].value_counts()
            line += (f" At {at} the L1 deep value sits more than {JUMP:g} bits above both neighbouring sizes on "
                     f"{len(tasks)} of {b1['task'].nunique()} tasks (" + ", ".join(f"{k} {v}" for k, v in fam.items())
                     + f"), against {n_o} of {n_all} (task, L) cells of the other L's deep cells; without those tasks "
                     f"the {at} mean gap is {fmt(b1.loc[~b1['task'].isin(tasks), 'gap'].mean(), 3)} instead of "
                     f"{fmt(b1['gap'].mean(), 3)}: a spike of that one cell, not of English (the store holds finals "
                     "only and no replicate seed, so it cannot be told from seed noise).")
        bullets.append(line)
    per_l = summ[(summ["scoring"] == "acc") & (summ["arch"] == "deep") & (summ["size"] == TARGET_SIZE)
                 & (summ["comparator"] != "every L > 1")]
    if len(per_l):
        bullets.append(f"- **By L at {TARGET_SIZE}** (deep, mean gap / win share): "
                       + "; ".join(f"{r_.comparator} {fmt(100 * r_.mean_gap, 1)} pp / {fmt(r_.win_share)}" for r_ in
                                   per_l.set_index("comparator").loc[group_order(per_l["comparator"])].reset_index()
                                   .itertuples()) + ".")
    replace_block(path, "english-scores", "\n".join([
        "![English scores](" + rel + "/english_only_scores.png)", "",
        "![English scores per benchmark](" + rel + "/english_only_scores_by_benchmark.png)", "",
        "Key findings (accuracy-scored English tasks above chance at the size; the population moves with the size, the "
        "counts are drawn on the pooled line and listed below):", "", *bullets, "",
        links("english_only_scores.png", "english_only_scores.csv", "english_only_scores_by_benchmark.png",
              "english_only_scores_by_benchmark.csv", "english_only_scores_per_task.csv",
              "english_only_scores_summary.csv")]), gen)

    # --- the gate
    n = share.groupby("size")["n_tasks"].first()
    bullets = [f"- **Population**: the English tasks with a chance level every drawn cell scores, "
               + ", ".join(f"{c} {int(n[c])}" for c in size_order(n.index)) + "; a size below half the largest population is "
               "left out of the lines and of the shares below (" + (", ".join(sorted(set(share.loc[share["thin"], "size"])))
                                                                    or "none") + ")."]
    sh = share[(share["build"] == "") & ~share["thin"]]
    for arch in ARCHS:
        x = sh[sh["ladder"] == arch].pivot_table(index="L", columns="size", values="share")
        if 1 in x.index:
            x = x.reindex(columns=[c for c in size_order(x.columns)])
            bullets.append(f"- **{arch}**: share of English tasks above chance, L1 "
                           f"{', '.join(f'{c} {fmt(v)}' for c, v in x.loc[1].items())}; other L's median "
                           f"{', '.join(f'{c} {fmt(v)}' for c, v in x.drop(1).median().items())}.")
    for r_ in cross[cross["arch"] == "deep"].itertuples():
        bullets.append(f"- **deep, L1 vs {r_.comparator}** ({r_.n_tasks} tasks either clears): L1 clears from a smaller size "
                       f"on {fmt(r_.earlier)}, the same on {fmt(r_.same)}, a larger one on {fmt(r_.later)}.")
    replace_block(path, "english-gate", "\n".join([
        "![Above random](" + rel + "/english_only_above_random.png)", "", "Key findings:", "", *bullets, "",
        links("english_only_above_random.png", "english_only_above_random.csv", "english_only_above_random_per_task.csv")]), gen)

    # --- scaling
    bullets = []
    if len(sle):
        top = sle[sle["ladder_top"] == SMALL_SIZES[-1]]
        bullets.append(f"- **English BPB, fit on {SMALL_SIZES[0]}–{SMALL_SIZES[-1]}** (a forecast from the proxies): "
                       f"|error| of the {TARGET_SIZE} prediction "
                       + "; ".join(f"{r_.chain} {fmt(100 * abs(r_.rel_error), 1)} %" for r_ in top.itertuples()) + ".")
    if len(fits):
        m = fits.groupby("L").agg(r2=("r2", "median"), n=("r2", "count"))
        bullets.append(f"- **English accuracy tasks** (deep, median R² of the log-N fit over the rungs the L1 deep cell "
                       f"has, {TARGET_SIZE} included — a reference-size quantity (rule 11) that describes the ladder, not a "
                       f"forecast; fits in brackets): " + "; ".join(f"{label(L)} {fmt(r_.r2)} ({int(r_.n)})"
                                                                     for L, r_ in m.iterrows()) + ".")
    replace_block(path, "english-scaling", "\n".join([
        "![Scaling](" + rel + "/english_only_scaling.png)", "", "Key findings:", "", *bullets, "",
        links("english_only_scaling.png", "english_only_scaling.csv")]), gen)

    # --- decision accuracy
    if missing:
        replace_block(path, "english-da", rule5, gen)
    else:
        bullets = []
        for axes in AXES:
            d = da[(da["axes"] == axes) & (da["population"] == "all benchmarks") & da["size"].isin(SMALL_SIZES) & ~da["thin"]]
            if d.empty:
                continue
            piv = d.pivot_table(index="group", columns="size", values="reliability")
            piv = piv.reindex(index=group_order(piv.index), columns=[c for c in SMALL_SIZES if c in piv.columns])
            half = ((d["hi"] - d["lo"]) / 2).groupby([d["group"], d["size"]]).first()
            bullets.append(f"- **{axes}**: " + "; ".join(
                f"{g_} " + ", ".join(fmt(v) + (f" ±{fmt(half[(g_, c)])}" if half.get((g_, c), np.nan) == half.get((g_, c), np.nan)
                                              else "") for c, v in r_.items() if v == v)
                for g_, r_ in piv.iterrows()) + f" ({', '.join(piv.columns)}).")
        mx = mix[(mix["axes"] == "multi-axis") & (mix["population"] == "all benchmarks")]
        if len(mx):
            bullets.append("- **By the axis a pair moves** (multi-axis, pooled over the proxies drawn; decisions in "
                           "brackets): " + "; ".join(
                               f"{g_} " + ", ".join(f"{EO_AXIS_LABEL.get(r_.axis, 'two axes at once')} {fmt(r_.da)} "
                                                    f"({int(r_.n_decisions)})" for r_ in x.sort_values("axis").itertuples())
                               for g_, x in ((k_, mx[mx["group"] == k_]) for k_ in group_order(mx["group"].unique())))
                           + ". The depth pair (deep against shallow) is the one decision every group shares.")
        bb = cells[(cells["axes"] == "multi-axis") & (cells["scoring"] == "bbpb") & cells["size"].isin(SMALL_SIZES)
                   & cells["da"].notna()]
        if len(bb):
            t = bb.groupby("group").agg(m=("n_matching", "sum"), n=("n_pairs", "sum"), p=("n_pairs", "median"))
            t = t.reindex(group_order(t.index))
            bullets.append("- **bBPB twins** (not in the lines: some family of a group has none; multi-axis, every cell "
                           "with ≥ 3 pairs, pooled over the proxies, NOT one population): " + "; ".join(
                               f"{k_} {fmt(v.m / v.n)} ({int(v.p)} pairs per task)" for k_, v in t.iterrows()) + ".")
        bp = da[(da["axes"] == "multi-axis") & (da["population"] == "bpb") & da["size"].isin(SMALL_SIZES) & ~da["thin"]]
        if len(bp):
            bullets.append(f"- **English BPB** (multi-axis, mean over the proxies): "
                           + "; ".join(f"{k_} {fmt(v)}" for k_, v in bp.groupby("group")["reliability"].mean()
                                       .reindex(group_order(bp["group"].unique())).items()) + ".")
        replace_block(path, "english-da", "\n".join([
            "![DA-size, multi-axis](" + rel + "/english_only_da_size_by_L_multi_axes.png)", "",
            "![DA-size, mono-axis](" + rel + "/english_only_da_size_by_L_mono_axis.png)", "",
            f"Key findings (DA-size against {TARGET_SIZE}, no reliability filter, pooled over the English benchmarks above "
            f"chance at the proxy and the reference (gate `{gate_pool}`) that every family of every group has, ≥ "
            f"{MIN_PAIRS} pairs per task, ± the 90 % leave-one-family-out half-width; the decisions differ by L):", "",
            *bullets, "",
            links("english_only_da_size_by_L_multi_axes.png", "english_only_da_size_by_L_multi_axes.csv",
                  "english_only_da_size_by_L_mono_axis.png", "english_only_da_size_by_L_mono_axis.csv",
                  "english_only_da_size_by_L_both_axes.csv", "english_only_da_size_by_pair_axis_both_axes.csv",
                  "english_only_da_size_per_task_both_axes.csv")]), gen)

    # --- SNR
    bullets = []
    for fam in (MIRROR, DEPTH):
        p = paired[(paired["families"] == fam)]
        if p.empty:
            if fam == MIRROR and missing:
                bullets.append(f"- **{FAMILIES[fam][0].upper() + FAMILIES[fam][1:]}**: not computed ({missing.split(':')[0]}).")
            continue
        piv = p[~p["thin"]].pivot_table(index="group", columns="size", values="median_log10_snr")
        piv = piv.reindex(index=group_order(piv.index), columns=[c for c in size_order(piv.columns)])
        n = p.groupby("size")["n_tasks"].max()
        th = sorted(set(p.loc[p["thin"], "size"]), key=sizes.index)
        bullets.append(f"- **{FAMILIES[fam][0].upper() + FAMILIES[fam][1:]}**, median log10 SNR over the paired tasks ("
                       + ", ".join(f"{c}: {int(n[c])}" for c in size_order(n.index)) + " tasks"
                       + (f"; thin: {', '.join(th)}" if th else "") + "): "
                       + "; ".join(f"{g_} " + ", ".join(fmt(v) for v in r_) for g_, r_ in piv.iterrows())
                       + f" ({', '.join(piv.columns)}).")
    s = sur[(sur["axes"] == "multi-axis") & ~sur["thin"]] if len(sur) else sur
    if len(s):
        at = [z for z in SMALL_SIZES if z in set(s["size"])]
        bullets.append("- **SNR → DA-size, Spearman ρ per proxy** (mirrored families, the DA lines' tasks): " + "; ".join(
            f"{g_} " + ", ".join(fmt(v) for v in x.set_index("size").reindex(at)["rho"])
            for g_, x in ((k_, s[s["group"] == k_]) for k_ in group_order(s["group"].unique()))) + f" ({', '.join(at)}).")
    replace_block(path, "english-snr", "\n".join([
        "![SNR](" + rel + "/english_only_snr.png)", "", "Key findings:", "", *bullets, "",
        links("english_only_snr.png", "english_only_snr.csv", "english_only_snr_per_task.csv")]), gen)

    # --- AllenAI
    if missing:
        replace_block(path, "english-allenai", rule5, gen)
    elif al is not None and len(al):
        n1 = len(groups[1])
        n_pairs = n1 * (n1 - 1) // 2
        x = al.dropna(subset=["l1"])
        defs = md_table(["", "ours (L1)", "DataDecide"], [
            ["models behind the signal", f"{n1} families (" + ", ".join(build_token(f) or f.split("-")[2] for f in groups[1])
             + ")", "25 data recipes"],
            ["DA pairs", f"{n_pairs} (multi-axis; a cell counts only with all {n1} families), so DA moves in steps of "
                         f"1/{n_pairs}", "300"],
            ["decisions", "depth and the English corpus", "the data recipe"],
            ["reference", TARGET_SIZE, f"{ALLENAI_REF} (so no DataDecide DA-size at {ALLENAI_REF})"],
            ["proxies", " / ".join(o for o, _ in SIZE_PAIRS if o != TARGET_SIZE) + " (non-embedding)",
             " / ".join(t for o, t in SIZE_PAIRS if o != TARGET_SIZE)],
            ["token budget", "100 N (5× Chinchilla)", "100 tokens per parameter (5× Chinchilla), per the DataDecide release"],
            ["checkpoint noise", "std over 80/85/90/95/100 % of the run (rule 4)",
             "std over the last five saved checkpoints (the table predates rule 4)"],
            ["metric", "lm-eval `acc` (originals), `acc_norm` (RF twins)",
             "the release's OLMES primary metric (not re-read here: the `core` parquet is not cached on the cluster)"],
            ["gate", "rule 1 at the proxy (and the reference for DA)", "none"]])
        rows = [[r_.task, r_.allenai_task, r_.quantity.replace("_", "-"), f"{r_.size} / {r_.allenai_size}", fmt(r_.l1),
                 int(r_.l1_n), fmt(r_.other_l_median), fmt(r_.allenai), f"{fmt(r_.allenai_min)}–{fmt(r_.allenai_max)}",
                 r_.l1_vs_allenai_range if r_.quantity == "da_size" else "(not graded)", fmt(r_.l1_over_allenai)]
                for r_ in x.itertuples()]
        none = al[al["l1"].isna()]
        why = ", ".join(f"`{t}` {', '.join(sorted(set(v['size']), key=sizes.index))}"
                        for t, v in none[none["quantity"] == "da_size"].groupby("task", sort=False))
        bullets = [f"- **{q.replace('_', '-')}**: L1 / DataDecide on the same task, median {fmt(g_['l1_over_allenai'].median())} "
                   f"(range {_ranges(g_['l1_over_allenai'])}); the other L groups' median over DataDecide "
                   f"{fmt((g_['other_l_median'] / g_['allenai']).median())}." for q, g_ in x.groupby("quantity")]
        if why:
            bullets.append(f"- **No L1 DA-size** (at chance on our side, rule 1, or a family missing): {why}. DataDecide is "
                           "not gated, so its range is taken only over the tasks where L1 has a value at that size.")
        reading = [
            "Reading the two sides:", "",
            f"- **DA-size shares the kernel, not the decisions.** Both sides count agreeing pairs; ours is conditional on "
            f"the task passing the gate at the proxy and the reference, theirs is not, and ours ranks {n1} families that "
            f"differ in depth and English corpus, theirs 25 recipes. A cell above DataDecide's range is a "
            f"{n_pairs}-of-{n_pairs} (DA moves in steps of 1/{n_pairs} against 1/300).",
            "- **The SNRs are not comparable** and are not graded: DataDecide's signal is the spread of 25 recipes over very "
            "different corpora, ours the spread of families of which two differ in depth only or in one filter, and the "
            "noise uses different checkpoint windows. The like-for-like SNR reading is deep against shallow at every L "
            "([Noise and SNR](#noise-and-snr)); the four-family direction against the other L groups is a within-ladder "
            "reading over different decisions per L (Setup)."]
        replace_block(path, "english-allenai", "\n".join([
            defs, "", "![AllenAI](" + rel + "/english_only_allenai.png)", "", "Key findings:", "", *bullets, "",
            md_table(["ours", "DataDecide", "quantity", "sizes (ours / theirs)", "L1", "L1 pairs / runs", "other L median",
                      "DataDecide", "DataDecide range (L1's tasks)", "L1 vs range", "L1 / DataDecide"], rows), "",
            *reading, "",
            links("english_only_allenai.png", "english_only_allenai.csv", "english_only_allenai_per_task.csv")]), gen)
    else:
        replace_block(path, "english-allenai",
                      "The AllenAI table is absent (`build_allenai_variants.py` / `git lfs pull`): no comparison written.", gen)

    replace_block(path, "english-verdict", md_table(["expectation", "measure", "L1", "comparator", "confirmed?"],
                                                    ver[["expectation", "measure", "l1", "comparator", "verdict"]].values.tolist())
                  + "\n\n" + links("english_only_verdict.csv"), gen)


# --- driver ------------------------------------------------------------------------

def main(pool: str, seeds_pool: str, write_readme: bool) -> None:
    gate_pool = mask_pool(pool)
    mask = load_mask(gate_pool)
    if mask is None:
        print(f"no above-random mask for `{pool}` or `{CANONICAL_POOL}` (above_random.py): nothing written")
        return
    df = ladder_frame(pool)
    df = df[df["task"].map(assign_language) == EN]
    fin = finals(df)
    cells = cell_table(fin)
    groups = mirror_groups(cells)
    if 1 not in groups:
        print(f"`{pool}` holds no L1 cell at seed {GRID_SEED}: nothing written")
        return
    n1 = len(groups[1])
    same = {L: f for L, f in groups.items() if len(f) == n1}
    depth = depth_groups(cells)
    missing = ""
    if n1 < 3:
        missing = (f"L1 has {n1} families in `{pool}`: decision accuracy needs three (rule 5, {MIN_PAIRS} pairs), so it is "
                   f"not computed ({', '.join(groups[1])}); the mirrored-family SNR, the surrogate and the comparison with "
                   "AllenAI's results are skipped with it, because this analysis compares L1 only with groups built like "
                   "its four families (a choice of the analysis, not rule 5). The deep-against-shallow SNR is computed. "
                   "They fill in once the pool holds the DCLM-without-edu and FineWeb cells.")
        print("!!! RULE 5: " + missing)
    print(f"{pool}: {fin['task'].nunique()} English tasks, gate `{gate_pool}`; groups with L1's {n1} families: "
          + "; ".join(f"{label(L)} {f}" for L, f in same.items()))
    print("  left out (another number of families): " + ", ".join(f"{label(L)} ({len(f)})" for L, f in groups.items()
                                                                     if L not in same))
    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = ENGLISH_ONLY / stage / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = size_order(fin["size"].unique())

    # 1. scores, against the seed noise of the deep L1 cell
    ds = ladder_frame(seeds_pool)
    ds = ds[ds["task"].map(assign_language) == EN]
    evn = effect_vs_noise(ds, finals(ds), seeds_pool)
    seed_sd = evn.loc[evn["L"] == 1, ["task", "size", "seed_noise", "n_seeds"]].rename(columns={"seed_noise": "seed_sd"})
    g = score_gaps(fin, cells, mask, seed_sd)
    g.to_csv(out_dir / "english_only_scores_per_task.csv", index=False)
    summ = score_summary(g)
    where = f"pool `{pool}`, gate `{gate_pool}`"
    fig_scores(summ, out_dir, sizes, where)
    fig_scores(summ, out_dir, sizes, where, paper=True)
    fig_scores_fixed(g, out_dir)
    summ.to_csv(out_dir / "english_only_scores_summary.csv", index=False)
    fig_scores_by_benchmark(g, out_dir, sizes, where)

    # 2. the gate
    runs = gate_runs(fin, cells)
    runs.to_csv(out_dir / "english_only_above_random_per_task.csv", index=False)
    share = share_above(runs)
    cross = crossing(runs, cells, sizes)
    fig_gate(share, cross, out_dir, sizes, where)

    # 3. scaling: the chains of the baseline build at every L and every L1 build
    sle, _ = scaling_law_error(fin)
    ids = fin.loc[fin["seed"] == GRID_SEED, ["family", "L", "ladder", "data"]].drop_duplicates()
    sle = sle[sle["task"] == ENGLISH_BPB].merge(ids.assign(L=ids["L"].astype(int)), on=["L", "ladder", "data"])
    sle["build"] = sle["family"].map(build_token)
    sle = sle[(sle["build"] == "") | (sle["L"] == 1)]
    sle["chain"] = [f"{label(L)} {a}" + (f" {b}" if b else "") for L, a, b in zip(sle["L"], sle["ladder"], sle["build"])]
    l1_deep = cells.loc[(cells["L"] == 1) & (cells["ladder"] == "deep") & (cells["build"] == ""), "family"]
    rungs = fin.loc[fin["family"].isin(l1_deep), ["task", "size"]].drop_duplicates()     # every L fits the same rungs
    fits, _ = fit_table(fin.merge(rungs, on=["task", "size"]), gate_pool)
    fits = fits[(fits["kind"] == "benchmark") & (fits["task"].map(lambda t: variant(t)[1]) == "acc")]
    fig_scaling(sle, fits, out_dir, where)

    # 4.-6. decisions, SNR, surrogate: the mirrored groups where L1 has three families, deep-shallow everywhere
    da = dcells = mix = sur = pd.DataFrame()
    al = None
    snr = snr_table(df, depth, mask, DEPTH)
    if not missing:
        da, dcells, mix = da_tables(fin, same, gate_pool)
        da.to_csv(out_dir / "english_only_da_size_by_L_both_axes.csv", index=False)
        mix.to_csv(out_dir / "english_only_da_size_by_pair_axis_both_axes.csv", index=False)
        dcells.to_csv(out_dir / "english_only_da_size_per_task_both_axes.csv", index=False)
        for axes in AXES:
            fig_da(da[da["axes"] == axes], same, out_dir, axes, where)
        snr = pd.concat([snr_table(df, same, mask, MIRROR), snr], ignore_index=True)
    snr.to_csv(out_dir / "english_only_snr_per_task.csv", index=False)
    paired = paired_snr(snr)
    if not missing:
        sur = surrogate(snr, dcells)
        # 7. AllenAI
        al = allenai(snr, dcells, [label(L) for L in same if L != 1])
        if al is not None and len(al):
            al.to_csv(out_dir / "english_only_allenai_per_task.csv", index=False)
            fig_allenai(al, out_dir)
    fig_snr(snr, paired, sur, out_dir, sizes, where)

    # 8. the verdict
    ver = verdict(summ, cross, share, sle, fits, da, mix, paired, al, missing)
    ver.to_csv(out_dir / "english_only_verdict.csv", index=False)
    print(ver.to_string(index=False))
    if write_readme or (pool == CANONICAL_POOL and not missing):
        readme(pool, gate_pool, f"{stage}/{pool}", dict(
            g=g, summ=summ, cross=cross, share=share, sle=sle, fits=fits, da=da, cells=dcells, mix=mix, paired=paired,
            sur=sur, al=al, ver=ver, groups=same, all_groups=groups, depth=depth, missing=missing, sizes=sizes,
            population=population(fin), coverage=coverage(fin, [f for fs in same.values() for f in fs])))
    elif pool == CANONICAL_POOL:
        print(f"  README left as it is: `{pool}` lacks L1's families, and the driver writes the README from a pool that "
              "has them (--readme)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default=CANONICAL_POOL)
    p.add_argument("--seeds-pool", default=SEEDS_POOL, help="the pool holding the replicate seeds of the L1 cell")
    p.add_argument("--readme", action="store_true",
                   help=f"write the README blocks from this pool (by default only `{CANONICAL_POOL}` writes them, and "
                        "only when L1 has three families there)")
    p.add_argument("--paper", action="store_true",
                   help="only the paper's bare figures (english_only_scores_paper and its fixed-tasks twin), from the tables on disk")
    a = p.parse_args()
    if a.paper:
        d = ENGLISH_ONLY / load_pools()[a.pool].get("stage", "pretraining") / a.pool
        summ = pd.read_csv(d / "english_only_scores_summary.csv")
        fig_scores(summ, d, size_order(summ["size"].unique()), "", paper=True)
        fig_scores_fixed(pd.read_csv(d / "english_only_scores_per_task.csv"), d)
    else:
        main(a.pool, a.seeds_pool, a.readme)
