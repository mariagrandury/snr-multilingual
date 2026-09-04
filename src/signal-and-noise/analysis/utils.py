"""Shared analysis utilities — task-name helpers, the parent/aggregate filters,
the config-derived size params, and the SNR signal-pool loader.

These carry no research-question identity, so every ``rqNN_*`` script imports
them from here instead of from a sibling RQ. That keeps the numbered run order
a clean DAG: a lower-numbered RQ never imports a higher-numbered one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

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
    load_pools, load_snr_params, load_tasks, pool_include_external,
    stage_external_models,
)
from snr.download.apertus import (  # noqa: E402
    load_a06_eval_results, load_apertus_eval_results,
    load_distillation_eval_results, load_posttraining_eval_results,
    load_reference_hf_eval_results,
)

# --- size params (single source of truth: configs/models.json) --------------
_SNR = load_snr_params()
SMALL_SIZES = _SNR["small_sizes"]
TARGET_SIZE = _SNR["target_size"]
LAST_N = _SNR["last_n"]
CKPT_DA_EARLY_FRACS = _SNR["da_early_fracs"]
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


def build_snr_pool(pool: str) -> pd.DataFrame:
    """SNR signal-pool dataframe for the named pool. Apertus rows are filtered
    to the pool's `members` (via expand_pool); when the pool sets
    `include_external=true`, every external pretraining row (reference_hf, a06,
    distillation) declared at the pool's stage joins too. Externals have no
    `seed` and live only at their native sizes, but per_model_inputs groups by
    model name, so each adds a fresh signal/noise point at its size.
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

    pool_models = set(expand_pool(pool))
    df_a = load_apertus_eval_results()
    df_a = df_a[df_a["model"].isin(pool_models)].copy()
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
