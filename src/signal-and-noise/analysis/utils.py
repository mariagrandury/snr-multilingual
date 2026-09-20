"""Shared analysis utilities — task-name helpers, the parent/aggregate filters,
the config-derived size params, and the SNR signal-pool loader.

These carry no research-question identity, so every ``rqNN_*`` script imports
them from here instead of from a sibling RQ. That keeps the numbered run order
a clean DAG: a lower-numbered RQ never imports a higher-numbered one.
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

# --- size params (single source of truth: configs/models.json) --------------
_SNR = load_snr_params()
SMALL_SIZES = _SNR["small_sizes"]
TARGET_SIZE = _SNR["target_size"]
CKPT_DA_EARLY_FRACS = _SNR["da_early_fracs"]   # the nine evaluated checkpoints before the final (analysis/RULES.md, rule 3)
NOISE_WINDOW = _SNR["noise_window"]            # noise = std over the shared checkpoints in this last share of a run (rule 4)
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


def assign_language(task: str) -> str:
    """Project language tag of a task: ``multi`` for cross-language
    aggregates (`bpb_macro`, `train_loss`, `include_base_44`), ``??`` when
    unresolved."""
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
    form the ``bpb`` family, the training loss the ``loss`` family.
    """
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
    if task in _ENGLISH_ONLY_TASKS or task.startswith("bpb_") or task == "train_loss":
        return True
    return _is_language_aggregate(task, benchmark_family(task))


# Ladder pools filter the loaded frame on its own columns (the scheme-B cells
# and the adopted off-grid seeds are real runs whether or not models.json
# lists them), so a member spec may carry any of these column filters.
_LADDER_FILTERS = {"seeds": "seed", "sizes": "size", "L": "L",
                   "arch": "arch", "scheme": "scheme"}


def _is_ladder_pool(pool: str) -> bool:
    members = load_pools()[pool].get("members", [])
    return bool(members) and all(
        loader_for_source(m["source"]) == "ladder" for m in members)


def pool_models(pool: str, df: pd.DataFrame) -> set[str]:
    """The pool's own (non-external) model names: every model of a ladder
    pool's frame, else the models.json expansion."""
    return set(df["model"]) if _is_ladder_pool(pool) else set(expand_pool(pool))


def build_snr_pool(pool: str, *, untrained: bool = False, facets: bool = False) -> pd.DataFrame:
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
        df = df[df["size"].isin(EVAL_SIZES)]      # 90M trains but is off the ladder
        frames = []
        for m in spec["members"]:
            sub = df
            for key, col in _LADDER_FILTERS.items():
                if key in m:
                    sub = sub[sub[col].isin(m[key])]
            frames.append(sub)
        df = pd.concat(frames).drop_duplicates().reset_index(drop=True)
        if not facets:
            df = parents_only(df)
        if not untrained:
            df = trained_only(df)
        return df

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
GRID_SEED = 1904                      # the plan grid's seed


def size_order(sizes) -> list[str]:
    """The given sizes in ladder order (unknown sizes last, alphabetically)."""
    present = set(sizes)
    return ([s for s in LADDER_SIZES if s in present]
            + sorted(s for s in present if s not in NON_EMB))


def ladder_frame(pool: str, **kw) -> pd.DataFrame:
    """`build_snr_pool` plus `frac`, each checkpoint's position in its own run
    (step over the cell's last scored step, which `require_final` makes the
    target). `kw` = build_snr_pool's `untrained` / `facets` opt-outs."""
    df = build_snr_pool(pool, **kw)
    df["frac"] = df["step"] / df.groupby("model")["step"].transform("max")
    return df


def on_shared_grid(df: pd.DataFrame) -> pd.Series:
    """Rows at one of the ten evaluated tenths of the run (`frac` within
    FRAC_TOL of k/10): the grid BPB and benchmarks share. BPB is also scored
    on the twentieths; those rows are left out wherever the two kinds are
    compared (rule 3)."""
    return ((df["frac"] * 10).round() / 10 - df["frac"]).abs() <= FRAC_TOL


def noise_checkpoints(df: pd.DataFrame) -> pd.DataFrame:
    """The rows the checkpoint-noise estimate is read on: the shared tenths in
    the last NOISE_WINDOW of each run, the same window for every kind of
    measurement (rule 4). With NOISE_WINDOW = 0.2 that is 0.8, 0.9 and 1.0."""
    return df[on_shared_grid(df) & (df["frac"] >= 1 - NOISE_WINDOW - FRAC_TOL)]


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
def trained_tasks(L: int, scheme: str) -> frozenset[str]:
    """Every task a (L, scheme) cell is trained for: the benchmarks in its
    languages (English always), its languages' BPB, and the measurements that
    are not one language's (train_loss, bpb_macro)."""
    from pretrain.ladder_report import _trained_tasks
    return frozenset(_trained_tasks(L, scheme)) | (trained_bpb_tasks(L, scheme) or frozenset()) | {"train_loss", "bpb_macro"}


def is_trained(task: str, L: int, scheme: str) -> bool:
    return task in trained_tasks(int(L), scheme)


def trained_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a (model, task) row only when the model's mixture trains the
    task's language (rule 2). A score on an untrained language measures
    transfer, which is rq06's question and no other's."""
    keep = [is_trained(t, L, s) for t, L, s in zip(df["task"], df["L"], df["scheme"])]
    return df[np.asarray(keep, dtype=bool)]


def finals(df: pd.DataFrame) -> pd.DataFrame:
    """Each (model, task)'s last checkpoint."""
    return df.loc[df.groupby(["model", "task"])["step"].idxmax()]


def at_fraction(df: pd.DataFrame, f: float, tol: float = 0.06) -> pd.DataFrame:
    """Each (model, task)'s score at the checkpoint nearest fraction `f` of
    its run, within `tol` (half the k/10 benchmark grid spacing)."""
    d = df.assign(dist=(df["frac"] - f).abs())
    d = d.loc[d.groupby(["model", "task"])["dist"].idxmin()]
    return d[d["dist"] <= tol].drop(columns="dist")


def trained_bpb_tasks(L: int, scheme: str) -> set[str] | None:
    """The `bpb_<subset>` tasks of the languages a (L, scheme) cell trains on
    (English always); None when the scheme defines no list at that L."""
    from pretrain.launch_trainings import cell_fineweb_subsets
    try:
        return {"bpb_dclm"} | {f"bpb_{s}" for s in cell_fineweb_subsets(L, scheme)}
    except KeyError:
        return None
