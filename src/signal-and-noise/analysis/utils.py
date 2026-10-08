"""Shared analysis utilities — task-name helpers, the parent/aggregate filters,
the config-derived size params, and the SNR signal-pool loader.

These carry no research-question identity, so every ``rqNN_*`` script imports
them from here instead of from a sibling RQ, which keeps most of the numbered
run order a clean DAG.

It is not a strict one: rq03's ``compare_seed_splits`` imports rq04's
``analyze_snr_variants`` / ``snr_definition_postprocess``, and rq03's
``effect_vs_noise`` imports rq05's ``INTERVENTIONS``. Those are imports of
FUNCTIONS AND CONSTANTS only — neither module reads a file at import time — so
the run order is unaffected and a lower RQ never waits on a higher one's
tables. Keep it that way: anything that would make a lower-numbered RQ read a
higher-numbered one's OUTPUT belongs here instead.
"""

from __future__ import annotations

import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# utils.py lives at analysis/; scripts run from analysis/rqNN_*/ and add
# signal-and-noise + src to sys.path before importing this. Re-assert it so the
# module also imports cleanly on its own.
_SND = Path(__file__).resolve().parents[1]   # signal-and-noise
_SRC = Path(__file__).resolve().parents[2]   # src
for _p in (_SND, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from evals.scripts.utils.configs import (  # noqa: E402
    _TASK_LANG_ALIASES, bucket_order, expand_pool, fineweb_language,
    load_pools, load_snr_params, load_tasks, loader_for_source,
    pool_include_external, stage_external_models,
)
from snr.download.apertus import (  # noqa: E402
    load_a06_eval_results, load_apertus_eval_results,
    load_distillation_eval_results, load_posttraining_eval_results,
    load_reference_hf_eval_results,
)
from snr.download.ladder import load_predictivity_eval_results  # noqa: E402
from analysis.paths import SUBSET_SELECTION  # noqa: E402

# --- size params (single source of truth: configs/models.json) --------------
_SNR = load_snr_params()
SMALL_SIZES = _SNR["small_sizes"]
TARGET_SIZE = _SNR["target_size"]
CKPT_DA_EARLY_FRACS = _SNR["da_early_fracs"]   # the nine evaluated checkpoints before the final (analysis/RULES.md, rule 3)
NOISE_WINDOW = _SNR["noise_window"]            # noise = std over the shared checkpoints in this last share of a run (rule 4)
NOISE_GRID = _SNR["noise_grid"]                # the k/NOISE_GRID points the window is read on: 5 of them at 20 (rule 4)
MIN_PAIRS = _SNR["min_pairs"]                  # a decision-accuracy cell needs this many design-variant pairs (rule 5)
MIN_LANG_TASKS = _SNR["min_lang_tasks"]        # a per-language correlation needs this many distinct tasks (rule 8)
SHARED_FRACS = [k / 10 for k in range(1, 11)]  # the checkpoint grid every size was evaluated on
FRAC_TOL = 0.02                                # a checkpoint is at a tenth when within this share of the run of it
LANGUAGE_AGGREGATES = ("multi", "??")          # tags that are not a language (rule 7)
_BUCKETS = bucket_order()
# Longest-first alternation so "12-14B" matches before "1B" etc.
_BUCKET_RE = "|".join(sorted((re.escape(b) for b in _BUCKETS), key=len, reverse=True))

# --- task-name helpers ------------------------------------------------------
# configs/tasks.json tags every registered task with its language and
# benchmark family (116 languages for the predictivity ladder); it is the first
# stop for every helper below. The token-parsing fallbacks cover the names the
# 36-sweep parquet carries that were never registered (subject facets, the
# standalone English tasks) and stay byte-identical to the old behaviour.
_TASKS = load_tasks()

_LANG_MAP = {
    "ar": "ar", "arb": "ar",
    "de": "de",
    "es": "es", "spa": "es",
    "eu": "eu", "eus": "eu",
    "fr": "fr",
    "hi": "hi", "hin": "hi",
    "ru": "ru", "rus": "ru",
    "vi": "vi", "vie": "vi",
    "zh": "zh", "zho": "zh", "cmn": "zh",
    "ja": "ja", "jp": "ja", "jpn": "ja",
    "sw": "sw", "swh": "sw",
    "th": "th", "tha": "th",
    "tr": "tr", "tur": "tr",
    "en": "en", "eng": "en",
}

_ENGLISH_ONLY_TASKS = {
    "arc_challenge", "arc_easy", "commonsense_qa", "hellaswag", "mmlu",
    "openbookqa", "piqa", "truthfulqa_mc1",
}

# Tasks merged into one family even though their names don't share a
# prefix-up-to-language-token (only ARC's challenge/easy split matches).
_BENCHMARK_FAMILY_OVERRIDES = {
    "arc_challenge": "arc",
    "arc_easy": "arc",
}

# Trailing tokens allowed after the language token in a per-language aggregate
# (ISO-15924 script codes + lm-eval format suffixes).
_TRAILING_OK = {
    "arab", "latn", "cyrl", "hans", "hant", "deva", "jpan",
    "thai", "geor", "hebr", "beng", "knda", "tibt", "spai",
    "mc1", "mc2",
}


# A benchmark's BPB twin: `bbpb_<task>` is the item-mean bits per byte of the
# gold answer of <task> (build_per_item_store.bench_bpb), a third member of the
# family beside the original and its `rf_` twins. Not `bpb_`: that prefix is
# the per-language BPB of the validation sets.
BBPB = "bbpb_"
BENCH_BPB = SUBSET_SELECTION / "bench_bpb.csv"      # model, step, task, bbpb


# A benchmark task is one evaluation VARIANT of a benchmark: a `format` (how
# the items are posed: the original, its `rf_` cloze rewrite, its `rfgm_`
# LLM rewrite) and a `scoring` (how an answer becomes a number: accuracy, or
# the gold answer's bits per byte). The loader writes both as columns, so a
# table splits by variant with one groupby instead of parsing names.
FORMATS = ("original", "rf", "rfgm")
SCORINGS = ("acc", "bbpb")
# tau: a decision accuracy at or above it "reads like the reference" (the safe
# proxy sizes, the design-decision and language-transfer lines, the evaluation
# recipe). One constant, so a figure, a table and a README sentence that quote
# it cannot drift apart. Distinct from the above_66/above_80 reliability filters.
RELIABLE_DA = 0.75


def variant(task: str) -> tuple[str, str]:
    """(format, scoring) of a benchmark task: `bbpb_rf_belebele_spa_Latn` ->
    ("rf", "bbpb"), `include_base_44_es` -> ("original", "acc")."""
    scoring = "bbpb" if task.startswith(BBPB) else "acc"
    t = task[len(BBPB):] if scoring == "bbpb" else task
    return ("rfgm" if t.startswith("rfgm_") else "rf" if t.startswith("rf_") else "original"), scoring


def lower_is_better(task: str) -> bool:
    """Bits per byte (per-language, `bpb_*`, and a benchmark's, `bbpb_*`) and
    the training loss: the scores where a smaller value is the better model."""
    return task.startswith(("bpb_", BBPB)) or task == "train_loss"


def assign_language(task: str) -> str:
    """Project language tag of a task: ``multi`` for cross-language
    aggregates (`bpb_macro`, `train_loss`, `include_base_44`), ``??`` when
    unresolved; a BBPB twin has its original's."""
    if task.startswith(BBPB):
        return assign_language(task[len(BBPB):])
    if task in ("bpb_macro", "train_loss"):
        return "multi"
    if task.startswith("bpb_"):
        return fineweb_language(task[len("bpb_"):])
    if task in _ENGLISH_ONLY_TASKS:
        return "en"
    lang = _TASKS.get(task, {}).get("language")
    if lang:
        return _TASK_LANG_ALIASES.get(lang, lang)
    for tok in task.split("_"):
        if tok in _LANG_MAP:
            return _LANG_MAP[tok]
    return "??"


def benchmark_family(task: str) -> str:
    """Strip any language/script suffix, leaving the benchmark identifier.

    ``arc_challenge`` / ``arc_easy`` collapse to ``arc``; English
    ``truthfulqa_mc1`` is left alone so it doesn't collapse with the
    multilingual ``truthfulqa_<lang>_mc1`` variants. Per-language BPB tasks
    form the ``bpb`` family, the training loss the ``loss`` family; a BBPB
    twin the ``bbpb_<family>`` one, as an ``rf_`` twin's is ``rf_<family>``.
    """
    if task.startswith(BBPB):
        return BBPB + benchmark_family(task[len(BBPB):])
    if task == "train_loss":
        return "loss"
    if task.startswith("bpb_"):
        return "bpb"
    if task in _BENCHMARK_FAMILY_OVERRIDES:
        return _BENCHMARK_FAMILY_OVERRIDES[task]
    fam = _TASKS.get(task, {}).get("benchmark")
    if fam:
        return fam
    parts = task.split("_")
    out = []
    for p in parts:
        if p in _LANG_MAP:
            break
        out.append(p)
    return "_".join(out) if out else parts[0]


def _is_language_aggregate(task: str, family: str) -> bool:
    """Keep only the per-language aggregate of a family (``global_mmlu_ar``),
    not the per-(lang, subject) facet (``global_mmlu_ar_business``). A language
    aggregate's tokens after the family name start with one language token; any
    further trailing tokens must be in ``_TRAILING_OK``.
    """
    if task in _BENCHMARK_FAMILY_OVERRIDES:
        return True
    if task in _TASKS:
        # Registered tasks are per-language evaluations by construction
        # (subtopics are never registered); the cross-language aggregates
        # (`include_base_44`, tagged multi) are not one language's task.
        return assign_language(task) not in ("multi", "??")
    if not task.startswith(family + "_"):
        return False
    rest = task[len(family) + 1:].split("_")
    if not rest or rest[0] not in _LANG_MAP:
        return False
    return all(tok.lower() in _TRAILING_OK for tok in rest[1:])


def _is_parent_task(task: str) -> bool:
    """Mirror the cluster's ``aggregate_parents``: keep one row per real
    evaluation, dropping the per-(lang, subject) facets. English standalone
    tasks (``_ENGLISH_ONLY_TASKS``) plus multilingual per-language aggregates.
    """
    if task.startswith(BBPB):
        return _is_parent_task(task[len(BBPB):])
    if task in _ENGLISH_ONLY_TASKS or task.startswith("bpb_") or task == "train_loss":
        return True
    return _is_language_aggregate(task, benchmark_family(task))


# Ladder pools filter the loaded frame on its own columns (the scheme-B cells
# and the adopted off-grid seeds are real runs whether or not models.json
# lists them), so a member spec may carry any of these column filters, plus
# `cells_of: <pool>`: only the cells (CELL_KEYS) that other pool has, which is
# how the seed holdout's test split matches its train split without a
# hand-kept list of sizes, settings and ladders.
_LADDER_FILTERS = {"seeds": "seed", "sizes": "size", "L": "L", "ladder": "ladder",
                   "arch": "arch", "activation": "activation", "optimizer": "optimizer",
                   "data": "data", "scheme": "scheme", "T": "T"}
CELL_KEYS = ["size", "L", "ladder", "data"]     # a cell, apart from its seed


def _ladder_members(df: pd.DataFrame, pool: str) -> pd.DataFrame:
    """The rows of the loaded ladder frame `df` that `pool`'s members select."""
    frames = []
    for m in load_pools()[pool]["members"]:
        sub = df
        for key, col in _LADDER_FILTERS.items():
            if key in m:
                sub = sub[sub[col].isin(m[key])]
        if "cells_of" in m:
            cells = pd.MultiIndex.from_frame(_ladder_members(df, m["cells_of"])[CELL_KEYS])
            sub = sub[pd.MultiIndex.from_frame(sub[CELL_KEYS]).isin(cells)]
        frames.append(sub)
    return pd.concat(frames).drop_duplicates().reset_index(drop=True)


def _is_ladder_pool(pool: str) -> bool:
    members = load_pools()[pool].get("members", [])
    return bool(members) and all(
        loader_for_source(m["source"]) == "ladder" for m in members)


def pool_models(pool: str, df: pd.DataFrame) -> set[str]:
    """The pool's own (non-external) model names: every model of a ladder
    pool's frame, else the models.json expansion."""
    return set(df["model"]) if _is_ladder_pool(pool) else set(expand_pool(pool))


def build_snr_pool(pool: str, *, untrained: bool = False, facets: bool = False,
                   above_reference: bool = False) -> pd.DataFrame:
    """SNR signal-pool dataframe for the named pool. Apertus rows are filtered
    to the pool's `members` (via expand_pool); when the pool sets
    `include_external=true`, every external pretraining row (reference_hf, a06,
    distillation) declared at the pool's stage joins too. Externals have no
    `seed` and live only at their native sizes, but per_model_inputs groups by
    model name, so each adds a fresh signal/noise point at its size.

    Two analysis-wide rules (analysis/RULES.md) are applied here, at the one
    place every ladder analysis loads from, so a script has to opt out to
    break them: sub-benchmarks are folded into their per-language parent
    (`facets=True` keeps them, for rq08 alone) and a model's score on a
    language its mixture does not train is dropped (`untrained=True` keeps
    it, for rq06 alone and for the gate, which has to cover every task).

    Sizes are ANALYSIS_SIZES, 90M to the reference (rule 10);
    `above_reference=True` widens that to every evaluated size and belongs to
    the size-generalization question alone, which is what the 3B rung is for.
    """
    # The "external" tier pools every non-custom model across all four
    # external parquets (reference_hf + a06 + distillation + posttraining),
    # all models and all tasks, with no stage/name filtering. The instruct
    # models live in both reference_hf (pretraining tasks) and posttraining
    # (posttraining tasks) on disjoint task sets, so the concat has no
    # duplicate (model, task) rows.
    if load_pools()[pool].get("load_all_external"):
        frames = []
        for loader in (
            load_reference_hf_eval_results,
            load_a06_eval_results,
            load_distillation_eval_results,
            load_posttraining_eval_results,
        ):
            try:
                frames.append(loader())
            except FileNotFoundError:
                continue
        return pd.concat(frames, ignore_index=True)

    spec = load_pools()[pool]
    if _is_ladder_pool(pool):
        df = load_predictivity_eval_results(
            include_diverged=spec.get("include_diverged", False))
        df = _ladder_members(df[df["size"].isin(EVAL_SIZES if above_reference else ANALYSIS_SIZES)], pool)   # rule 10
        if not facets:
            df = parents_only(df)
        if not untrained:
            df = trained_only(df)
        return with_variant_columns(with_bbpb_twins(df))

    members = set(expand_pool(pool))
    df_a = load_apertus_eval_results()
    df_a = df_a[df_a["model"].isin(members)].copy()
    frames = [df_a]
    if pool_include_external(pool):
        stage = load_pools()[pool].get("stage", "pretraining")
        allowed = stage_external_models(stage)
        for loader in (
            load_reference_hf_eval_results,
            load_a06_eval_results,
            load_distillation_eval_results,
        ):
            try:
                df_e = loader()
            except FileNotFoundError:
                continue
            df_e = df_e[df_e["model"].isin(allowed)]
            if not df_e.empty:
                frames.append(df_e)
    return pd.concat(frames, ignore_index=True)

# --- ladder-frame helpers (rq00 curves, rq01, rq03, rq05, rq06) -----------------------------------------
# The ladder's size axis by non-embedding parameters, from the report module
# that defines it (CLAUDE.md #13: importable through src/).
from pretrain.ladder_report import NON_EMB  # noqa: E402
from pretrain.launch_trainings import EVAL_SIZES  # noqa: E402

LADDER_SIZES = sorted(NON_EMB, key=NON_EMB.get)
# Rule 10: every analysis reads the ladder from 90M up to the reference (the
# 90M and 175M rungs at their own batch, 2026-09-23; the loader keeps the
# diverged batch-504 runs out). The 3B rung sits ABOVE the reference — it exists
# for the size-generalization question, which opts in with `above_reference=True`.
# Everywhere else a size above the reference would quietly become one more
# column in a table whose reference is 1.7B. Derived from TARGET_SIZE, so
# moving the reference moves this with it.
ANALYSIS_SIZES = [s for s in EVAL_SIZES if NON_EMB[s] <= NON_EMB[TARGET_SIZE]]
GRID_SEED = 1904                      # the plan grid's seed
# What a cell is, apart from its size, as design levels (analysis/RULES.md,
# Definitions). The order is the one figures read in. A cell is named by its
# ladder token and its data BUILD (`ladder`, `data`); neither is an axis. The
# ladder is read as `arch` (the depth alone), `activation` and `optimizer`
# (launch_trainings.LADDERS), so a (deep, swiglu) pair is an activation choice
# and not a depth one. The build is read as `scheme`, the letter of the recipe
# it implements at its L, and `T` (DATA_SCHEMES `letter`, `temp`): A and AT3
# are scheme A at T=1 and T=3; B (L8-L30), ZH (L2) and DCLMP (L1) are scheme B;
# ES (L2) and FWEB (L1) are scheme C. The loader carries all of these as columns.
DESIGN_AXES = ["L", "arch", "activation", "optimizer", "scheme", "T", "seed"]
# The three pair sets a decision-accuracy table can be computed over (rule 15).
#   multi-axis  every pair of design variants: rq02's convention to date, and
#               two thirds of its pairs move more than one axis at once.
#   mono-axis   the pairs that move exactly ONE of the DESIGN_AXES above, the seed
#               held: the decision a practitioner actually makes, and what
#               upstream's "every pair" is by construction (DataDecide's recipes
#               differ in the data mix alone).
#   seed        two draws of ONE design. There is no right ordering, so this is
#               not a decision set but the null: what a benchmark with no signal
#               reads. Empty in a single-seed pool.
PAIR_AXES = ("multi-axis", "mono-axis", "seed")
# What a figure or table drawn over each pair set is called: every pair set
# carries its own suffix, so the two readings sit side by side in one folder
# and no file name leaves the pair set to be guessed.
AXES_SUFFIX = {"multi-axis": "_multi_axes", "mono-axis": "_mono_axis", "seed": "_seed_null"}


def design_axes(df: pd.DataFrame) -> pd.DataFrame:
    """family -> its DESIGN_AXES, and the `data` build `moved_axes` needs.

    `family` is the cell name with only the size token stripped, so the axes
    are a function of it; the assertion is what guarantees that.
    """
    a = (df[["family", *DESIGN_AXES, "data"]].drop_duplicates().set_index("family"))
    assert not a.index.duplicated().any(), "family does not determine its design axes"
    return a


def data_build(L: int, scheme: str, T: int) -> str | None:
    """The data build that implements (scheme, T) at L, or None: the inverse
    of the loader's build -> (scheme, T) map, one-to-one within an L."""
    from pretrain.launch_trainings import DATA_SCHEMES
    hits = [d for d, v in DATA_SCHEMES.items()
            if v["letter"] == scheme and int(v["temp"]) == T and L in v["langs"]]
    assert len(hits) <= 1, f"two builds implement scheme {scheme} at T={T}, L{L}: {hits}"
    return hits[0] if hits else None


def moved_axes(ra: pd.Series, rb: pd.Series) -> list[str]:
    """The DESIGN_AXES two families (rows of `design_axes`) differ on.

    `scheme` is the recipe AT THAT L, so across L the letter alone does not
    say whether the recipe moved: B is DCLMP at L1, ZH at L2 and the
    diversity-first list at L8-L30. A pair moves `scheme` when the letters
    differ or when it reads two builds at one temperature (A and AT3 are one
    recipe at two T); within one L the two readings agree.
    """
    def moves(k):
        if k == "scheme":
            return ra[k] != rb[k] or (ra["data"] != rb["data"] and ra["T"] == rb["T"])
        return ra[k] != rb[k]
    return [k for k in DESIGN_AXES if moves(k)]


def pair_sets(attrs: pd.DataFrame, seed: int | None = GRID_SEED) -> dict[str, list]:
    """The three pair sets of PAIR_AXES over `attrs`'s families.

    `seed` holds the design pairs at one seed, so a replicate never enters a
    decision set (it is a draw of one design, not a second design); pass None
    to let every seed in. The `seed` set is the complement: pairs identical on
    every axis but the seed.

    A pool holding no cell at `seed` — the holdout's train split is the
    replicate seeds by definition — would otherwise have no design pair at all, and
    every table built from it comes out empty (its SNR join, and the seed
    holdout downstream of that). The seed is then held at each seed the pool
    does have; a pair spanning two seeds is still the null's, as everywhere.
    """
    if seed is not None and seed not in set(attrs["seed"]):
        seed = None
    fams = sorted(attrs.index)
    multi, mono, null = [], [], []
    for i, a in enumerate(fams):
        ra = attrs.loc[a]
        for b in fams[i + 1:]:
            rb = attrs.loc[b]
            differ = moved_axes(ra, rb)
            if differ == ["seed"]:
                null.append((a, b))
                continue
            if "seed" in differ or (seed is not None and ra["seed"] != seed):
                continue          # mixes a replicate into a design decision
            multi.append((a, b))
            if len(differ) == 1:
                mono.append((a, b))
    return {"multi-axis": multi, "mono-axis": mono, "seed": null}


def one_axes(df: pd.DataFrame, axes: str = PAIR_AXES[0]) -> pd.DataFrame:
    """One pair set of a decision-accuracy table (rule 15), with the `axes`
    column dropped so the frame has the shape it had before that column
    existed. The default is the multi-axis reading, so a consumer that does not
    ask keeps the numbers it had; a table written before rule 15 has no such
    column and passes straight through.
    """
    return df if "axes" not in df.columns else df[df["axes"] == axes].drop(columns="axes")


def pair_agreement(proxy: dict, ref: dict, pairs=None) -> tuple[float, int]:
    """(decision accuracy, pairs) over the families `proxy` and `ref` share.

    The rule of `snr.metrics.decision_acc_fast`, applied to an explicit pair
    list rather than to two aligned vectors: the sign of the score difference
    on both sides, so a pair tied in both agrees and a pair tied in one is a
    miss, order-invariantly. That kernel cannot take a pair list, so the rule
    is stated twice — `tests/test_metrics.py::TestPairAgreement` is what keeps
    the two from drifting apart. `pairs` restricts the set (the mono-axis
    reading); None is every pair, which reproduces the kernel exactly.
    Returns (NaN, n) below MIN_PAIRS, as rule 5 requires.
    """
    common = set(proxy) & set(ref)
    pl = [(a, b) for a, b in pairs if a in common and b in common] if pairs is not None \
        else [(a, b) for i, a in enumerate(sorted(common)) for b in sorted(common)[i + 1:]]
    if len(pl) < MIN_PAIRS:                                   # rule 5
        return float("nan"), len(pl)
    agree = sum(np.sign(proxy[a] - proxy[b]) == np.sign(ref[a] - ref[b]) for a, b in pl)
    return agree / len(pl), len(pl)


def agreement_measures(proxy, ref) -> dict:
    """Every rank-agreement statistic of two aligned score vectors, with the
    pair counts they are all functions of — so a table can show WHY two of
    them differ rather than that they do.

    Over the n(n-1)/2 unordered pairs: C concordant (decided the same way on
    both sides), D discordant, T_both tied on both, T_one tied on one side.
    Then, with n_pairs = C + D + T_both + T_one:

        da              (C + T_both) / n_pairs      `decision_acc_fast`'s convention
        tau_a           (C - D) / n_pairs           Kendall, ties count for neither
        gamma           (C - D) / (C + D)           Goodman-Kruskal, ties dropped
        da_drop_ref_ties  C / (C + D + T_one_proxy) rq05's convention: a pair the
                                                    REFERENCE ties is no decision
        tau_b           scipy's Kendall, the tie-corrected denominator
        rho, pearson_r  Spearman on ranks, Pearson on the raw scores

    and the exact relation the pipeline's DA has to Kendall's tau, ties
    included, which `tests/test_metrics.py` pins:

        2 * da - 1 == tau_a + (T_both - T_one) / n_pairs

    Rank statistics are NaN below MIN_PAIRS (rule 5) or when a side is
    constant; the counts are always returned.
    """
    from scipy.stats import kendalltau, pearsonr, spearmanr
    s, t = np.asarray(proxy, float), np.asarray(ref, float)
    i, j = np.triu_indices(len(s), 1)
    ds, dt = np.sign(s[i] - s[j]), np.sign(t[i] - t[j])
    C = int(((ds == dt) & (ds != 0)).sum())
    D = int(((ds == -dt) & (ds != 0)).sum())
    T_both = int(((ds == 0) & (dt == 0)).sum())
    T_proxy, T_ref = int(((ds == 0) & (dt != 0)).sum()), int(((ds != 0) & (dt == 0)).sum())
    n = len(i)
    out = {"n_models": len(s), "n_pairs": n, "concordant": C, "discordant": D,
           "tied_both": T_both, "tied_one": T_proxy + T_ref}
    nan = float("nan")
    if n < MIN_PAIRS:                                                 # rule 5
        return out | {k: nan for k in ("da", "tau_a", "tau_b", "gamma", "da_drop_ref_ties", "rho", "pearson_r")}
    if np.std(s) == 0 or np.std(t) == 0:
        # a constant side has no rank correlation, but the pair counts are
        # still what the kernel reads: DA and tau_a stay, as decision_acc_fast has them
        return out | {"da": (C + T_both) / n, "tau_a": (C - D) / n,
                      "gamma": (C - D) / (C + D) if C + D else nan,
                      "da_drop_ref_ties": C / (C + D + T_proxy) if C + D + T_proxy else nan,
                      "tau_b": nan, "rho": nan, "pearson_r": nan}
    return out | {"da": (C + T_both) / n, "tau_a": (C - D) / n,
                  "tau_b": float(kendalltau(s, t).statistic),
                  "gamma": (C - D) / (C + D) if C + D else nan,
                  "da_drop_ref_ties": C / (C + D + T_proxy) if C + D + T_proxy else nan,
                  "rho": float(spearmanr(s, t).statistic), "pearson_r": float(pearsonr(s, t).statistic)}


JACKKNIFE_Z = 1.645          # the two-sided 90 % band every jackknife interval here is drawn at


def jackknife_ratio(decisions: pd.DataFrame, keys: list, a: str = "family_a", b: str = "family_b",
                    match: str = "match") -> pd.DataFrame:
    """A pooled decision ratio per group of `keys`, with its leave-one-FAMILY-out
    jackknife standard error and 90 % band.

    Decision accuracy pooled over pairs is a degree-2 U-statistic of the
    families, so the unit that gets resampled is the family, never the pair
    and never the task: pairs share families, tasks share families. The
    jackknife is used rather than a bootstrap because a family drawn twice
    forms a pair tied on both sides, which the kernel scores as an agreement —
    a resample that inflates the very number it estimates. Leave-one-out has
    no such pair.

    Two families whose removal deletes the same pairs are ONE unit. On an
    axis of disjoint pairs (deep vs swiglu at each L, deep vs shallow) both
    members of a pair delete exactly that pair, and counting the two equal
    leave-one-out values as independent overstates the variance by
    2(m-1)/m / ((n-1)/n) — x2.5 on three pairs. The units are therefore the
    distinct deletion sets: the pair on such an axis, the family wherever
    families are shared between pairs (the L regimes, the pooled line), where
    nothing changes. With n units the interval is over n values; at or below
    MIN_PAIRS units it is reported as NaN rather than drawn, and `n_families`
    and `n_units` are carried so a reader can see how much the band rests on.
    """
    rows = []
    for key, g in decisions.groupby(keys, sort=False):
        fams = sorted(set(g[a]) | set(g[b]))
        theta = g[match].mean()
        pair = pd.Series(list(zip(g[a], g[b])), index=g.index)
        units = {frozenset(q for q in set(pair) if f in q) for f in fams}
        n = len(units)
        if n <= MIN_PAIRS:
            rows.append(dict(zip(keys, key), reliability=theta, se=float("nan"), n_families=len(fams), n_units=n))
            continue
        loo = np.array([g.loc[~pair.isin(u), match].mean() for u in units])
        se = np.sqrt((n - 1) / n * ((loo - loo.mean()) ** 2).sum())
        rows.append(dict(zip(keys, key), reliability=theta, se=se, n_families=len(fams), n_units=n))
    out = pd.DataFrame(rows)
    out["lo"], out["hi"] = out["reliability"] - JACKKNIFE_Z * out["se"], out["reliability"] + JACKKNIFE_Z * out["se"]
    return out


@lru_cache(maxsize=None)
def language_token_share(L: int, data: str) -> dict[str, float] | None:
    """language -> share of a cell's training tokens, for the mixture (L, data
    build: the frame's `data`, never the scheme letter).

    English is the DCLM half, `EN_SHARE` % of every token (all of them at
    L = 1); the FineWeb-2 half is split by the builder's own plan
    (`<prefix>.plan.json` on DATA_MASTER, via data_progress.exact_tokens — the
    same record data_progress.py reports coverage from), so the share is the
    temperature-allocated one and not a byte estimate. None when the build's
    record is not reachable (capstor), so a caller can degrade and say so.
    The shares are the same at every size — only the token budget scales.
    """
    from pretrain.data.data_progress import exact_tokens, mixture_paths
    from pretrain.launch_trainings import DATA_MASTER, EN_SHARE, cell_fineweb_subsets
    en = EN_SHARE / 100
    if L == 1:
        return {"en": 1.0}
    subsets = cell_fineweb_subsets(L, data)
    if len(subsets) == 1:              # one language takes the whole half; scheme A's L2 predates plan files
        return {"en": en, fineweb_language(subsets[0]): 1 - en}
    got = exact_tokens(mixture_paths(DATA_MASTER, L, data))
    if got is None:
        return None
    tokens, _ = got
    total = sum(tokens.values())
    out = {"en": en}
    for subset, n in tokens.items():
        lang = fineweb_language(subset)
        out[lang] = out.get(lang, 0.0) + (1 - en) * n / total      # dialects fold into one tag
    return out


@lru_cache(maxsize=None)
def train_tokens(size: str, ladder: str) -> int:
    """The token budget D(N) a cell of `size` and `ladder` trains for."""
    import json
    from pretrain.launch_trainings import HYPERPARAMS
    return json.loads(Path(HYPERPARAMS[ladder]).read_text())["configs"][size]["predictivity"]["train_tokens"]


def language_tokens(L: int, data: str, size: str, ladder: str) -> dict[str, float] | None:
    """language -> training tokens of it a cell (size, L, ladder, data build) saw:
    `language_token_share` times the cell's budget. None where the share is."""
    share = language_token_share(L, data)
    return None if share is None else {k: v * train_tokens(size, ladder) for k, v in share.items()}


def size_order(sizes) -> list[str]:
    """The given sizes in ladder order (unknown sizes last, alphabetically)."""
    present = set(sizes)
    return ([s for s in LADDER_SIZES if s in present]
            + sorted(s for s in present if s not in NON_EMB))


def ladder_frame(pool: str, **kw) -> pd.DataFrame:
    """`build_snr_pool` plus `frac`, each checkpoint's position in its own run
    (step over the cell's last scored step, which `require_final` makes the
    target). `kw` = build_snr_pool's `untrained` / `facets` /
    `above_reference` opt-outs."""
    df = build_snr_pool(pool, **kw)
    df["frac"] = df["step"] / df.groupby("model")["step"].transform("max")
    return df


def on_shared_grid(df: pd.DataFrame) -> pd.Series:
    """Rows at one of the ten evaluated tenths of the run (`frac` within
    FRAC_TOL of k/10): the grid BPB and benchmarks share. BPB is also scored
    on the twentieths; those rows are left out wherever the two kinds are
    compared (rule 3)."""
    return ((df["frac"] * 10).round() / 10 - df["frac"]).abs() <= FRAC_TOL


def on_noise_grid(df: pd.DataFrame) -> pd.Series:
    """Rows at one of the k/NOISE_GRID points of the run. Inside the noise
    window this is the grid every kind of measurement shares: BPB has always
    been scored on the twentieths, and `launch_trainings.due_iters` now asks
    for the benchmarks there too."""
    return ((df["frac"] * NOISE_GRID).round() / NOISE_GRID - df["frac"]).abs() <= FRAC_TOL


def noise_checkpoints(df: pd.DataFrame) -> pd.DataFrame:
    """The rows the checkpoint-noise estimate is read on: the k/NOISE_GRID
    points in the last NOISE_WINDOW of each run, the same window and the same
    grid for every kind of measurement (rule 4). With NOISE_WINDOW = 0.2 and
    NOISE_GRID = 20 that is 0.80, 0.85, 0.90, 0.95 and 1.00 — five points.

    A run whose benchmark evals at 85 % and 95 % have not landed yet
    contributes three points, not five, and its noise is a noisier estimate
    of the same quantity; it is not a different definition.

    One row per grid point: a run that exited on SIGUSR2 has an extra save a
    few thousand iterations off the grid (lm-1.7B-L1-shallow-seed1904 saved at
    72,367 of 80,640, 89.7 %), and BPB scores every converted checkpoint, so
    without this the 90 % point is counted twice and the spread is read over
    six samples of five checkpoints."""
    sub = df[on_noise_grid(df) & (df["frac"] >= 1 - NOISE_WINDOW - FRAC_TOL)]
    if sub.empty:
        return sub
    point = (sub["frac"] * NOISE_GRID).round()
    keys = [sub[c] for c in ("model", "task") if c in sub.columns] + [point]
    nearest = (sub["frac"] - point / NOISE_GRID).abs().groupby(keys).idxmin()
    return sub.loc[sorted(nearest)]


def passes_gate(mask: pd.DataFrame | None, tasks, *sizes) -> pd.Series:
    """True where the above-random gate keeps a task at every one of `sizes`
    (rule 1): a mask of 0 rejects, 1 passes, and NA (no chance level: BPB, the
    loss, the generative tasks) passes — it is not a failed gate. A size the
    mask has no column for passes too."""
    ok = pd.Series(True, index=pd.Index(tasks))
    if mask is None:
        return ok
    for size in sizes:
        if size in mask.columns:
            ok &= (mask[size].reindex(ok.index) != 0).fillna(True).astype(bool)
    return ok


def parents_only(df: pd.DataFrame) -> pd.DataFrame:
    """One row per real evaluation: the per-language parent of every benchmark
    (`global_mmlu_full_ar`), never its subject facets (rule 6)."""
    keep = df["task"].map(_is_parent_task)
    return df[keep.to_numpy(dtype=bool)]


def languages_only(df: pd.DataFrame, col: str = "language") -> pd.DataFrame:
    """Drop the rows whose language tag is not a language: the cross-language
    aggregates (`multi`: bpb_macro, train_loss, include_base_44) and the
    unresolved (`??`). Every per-language table goes through this (rule 7)."""
    return df[~df[col].isin(LANGUAGE_AGGREGATES)]


@lru_cache(maxsize=None)
def trained_tasks(L: int, data: str) -> frozenset[str]:
    """Every task a (L, data build) cell is trained for: the benchmarks in its
    languages (English always), its languages' BPB, and the measurements that
    are not one language's (train_loss, bpb_macro)."""
    from pretrain.ladder_report import _trained_tasks
    return frozenset(_trained_tasks(L, data)) | (trained_bpb_tasks(L, data) or frozenset()) | {"train_loss", "bpb_macro"}


def is_trained(task: str, L: int, data: str) -> bool:
    return task in trained_tasks(int(L), data)


def trained_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a (model, task) row only when the model's mixture trains the
    task's language (rule 2). A score on an untrained language measures
    transfer, which is rq06's question and no other's."""
    keep = [is_trained(t, L, d) for t, L, d in zip(df["task"], df["L"], df["data"])]
    return df[np.asarray(keep, dtype=bool)]


def with_bbpb_twins(df: pd.DataFrame) -> pd.DataFrame:
    """`df` plus a `bbpb_<task>` row for every benchmark row the BBPB table
    (`BENCH_BPB`, written from the per-item store) has a value for: a copy of
    the row with the gold-answer bits per byte as its score. A copy, so the
    twin carries the row's model, size, step and design columns and inherits
    every rule the row already passed (the frame is filtered before this).
    A model with benchmark rows of a task family the table covers but no row
    at all in the table (a cell evaluated after the last store build) gets no
    twin: printed, since its twins' DA then runs over fewer families than the
    originals'."""
    if not BENCH_BPB.is_file():
        return df
    t = pd.read_csv(BENCH_BPB)
    bench = df[df["kind"] == "benchmark"]
    covered = bench["task"].isin(set(t["task"]))       # families with no bBPB anywhere are not a gap
    absent = sorted(set(bench.loc[covered, "model"]) - set(t["model"]))
    if absent:
        print(f"bBPB twins: {len(absent)} models not in the per-item store (rebuild with "
              f"build_per_item_store.sbatch): {', '.join(absent)}")
    twin = bench.merge(t, on=["model", "step", "task"])
    twin = twin.assign(task=BBPB + twin["task"], primary_score=twin["bbpb"]).drop(columns="bbpb")
    return pd.concat([df, twin], ignore_index=True)


def with_variant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """`format` and `scoring` on every benchmark row (`variant`); empty on the
    per-language BPB and the loss, which are not benchmark variants."""
    v = {t: variant(t) for t in df.loc[df["kind"] == "benchmark", "task"].unique()}
    return df.assign(format=df["task"].map(lambda t: v.get(t, (None, None))[0]),
                     scoring=df["task"].map(lambda t: v.get(t, (None, None))[1]))


def finals(df: pd.DataFrame) -> pd.DataFrame:
    """Each (model, task)'s last checkpoint."""
    return df.loc[df.groupby(["model", "task"])["step"].idxmax()]


def at_fraction(df: pd.DataFrame, f: float, tol: float = 0.06) -> pd.DataFrame:
    """Each (model, task)'s score at the checkpoint nearest fraction `f` of
    its run, within `tol` (half the k/10 benchmark grid spacing)."""
    d = df.assign(dist=(df["frac"] - f).abs())
    d = d.loc[d.groupby(["model", "task"])["dist"].idxmin()]
    return d[d["dist"] <= tol].drop(columns="dist")


def trained_bpb_tasks(L: int, data: str) -> set[str] | None:
    """The `bpb_<subset>` tasks of the languages a (L, data build) cell trains on
    (English always); None when the build defines no list at that L."""
    from pretrain.launch_trainings import cell_fineweb_subsets
    try:
        return {"bpb_dclm"} | {f"bpb_{s}" for s in cell_fineweb_subsets(L, data)}
    except KeyError:
        return None
