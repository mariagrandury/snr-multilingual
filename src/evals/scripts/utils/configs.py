"""Shared loader for configs/models.json + configs/tasks.json + configs/hf_wandb.json.

Single source of truth for the eval / pretrain / signal-and-noise
pipelines. See .claude-shared/plans/models-tasks-json-refactor.md for
the full plan.

Path conventions:
- evals + pretrain code runs on the cluster (this file is read from
  /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/utils/).
- signal-and-noise code runs locally on the Mac
  (/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/...).
- configs/*.json live at the SHARED repo root — same content readable
  from either host.

What lives where:
- models.json   → `models` (each with a `stages` dict), `pools`,
                  `sources` (source→parquet split), `snr` (SNR analysis params)
- tasks.json    → `tasks`, `groups`
- hf_wandb.json → published-dataset + infra config (HF repo id, parquet
                  filename pattern, W&B entity/project, multilingual-evals
                  raw/ sources)
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

# Repo root resolution: this file is at <REPO>/src/evals/scripts/utils/configs.py
_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODELS_JSON = _REPO_ROOT / "configs" / "models.json"
DEFAULT_TASKS_JSON = _REPO_ROOT / "configs" / "tasks.json"
DEFAULT_HF_WANDB_JSON = _REPO_ROOT / "configs" / "hf_wandb.json"

# Size tokens recognised by the auto-derive branch of `family_of`. Add to
# this list when a new model size appears (extending here is cheaper than
# hand-curating a family in models.json for every dynamically-loaded row).
_SIZE_TOKENS = (
    "175M", "350M", "600M", "1B",                          # Apertus custom
    "150M", "300M", "750M", "190M",                        # AllenAI ladder
    "0.6B", "1.7B", "3B", "4B", "7B", "8B", "12B", "13B", "14B", "27B",
    "32B", "70B",
)


# --- Models -----------------------------------------------------------------

@lru_cache(maxsize=4)
def load_models(path: str | Path = DEFAULT_MODELS_JSON) -> dict[str, Any]:
    """{model_name: entry} dict (the `models` section of models.json)."""
    return json.loads(Path(path).read_text())["models"]


@lru_cache(maxsize=4)
def load_pools(path: str | Path = DEFAULT_MODELS_JSON) -> dict[str, Any]:
    """{pool_name: {description, stage, members, include_external?}}."""
    return json.loads(Path(path).read_text()).get("pools", {})


def get_model(name: str, path: str | Path = DEFAULT_MODELS_JSON) -> dict:
    return load_models(path)[name]


def filter_models(source: str | list[str] | None = None,
                  stage: str | None = None,
                  family: str | None = None,
                  size: str | None = None,
                  seeds: list[int] | None = None,
                  path: str | Path = DEFAULT_MODELS_JSON) -> list[str]:
    """Return the list of model names matching every non-None filter.

    `source` accepts a single value or a list of values (OR-ed). The
    other filters are single-valued. `seeds` is an OR over the int list.
    """
    src_set = ({source} if isinstance(source, str)
               else set(source) if source else None)
    seed_set = set(seeds) if seeds else None
    out = []
    for name, entry in load_models(path).items():
        if src_set is not None and entry.get("source") not in src_set:
            continue
        if stage is not None and stage not in entry.get("stages", {}):
            continue
        if family is not None and entry.get("family") != family:
            continue
        if size is not None and entry.get("size") != size:
            continue
        if seed_set is not None and entry.get("seed") not in seed_set:
            continue
        out.append(name)
    return out


# --- Pools ------------------------------------------------------------------

def expand_pool(name: str,
                path: str | Path = DEFAULT_MODELS_JSON) -> list[str]:
    """Pool name → ordered list of model names.

    A pool's `members` is a list of filter dicts (same kwargs as
    `filter_models`). The union of every member's matches is returned.
    External reference models (every row whose `source` isn't
    `snr-pretraining-custom` or `snr-pretraining-a06`) are appended when
    `include_external=true` for any matching member (caller-side
    behavior: signal-and-noise scripts read this flag and decide whether
    to include externals in the SNR signal pool).
    """
    pool = load_pools(path)[name]
    out: list[str] = []
    seen: set[str] = set()
    for member in pool["members"]:
        for m in filter_models(**member, path=path):
            if m not in seen:
                out.append(m)
                seen.add(m)
    return out


def pool_include_external(name: str,
                          path: str | Path = DEFAULT_MODELS_JSON) -> bool:
    return load_pools(path)[name].get("include_external", False)


# --- Sources → parquet split ------------------------------------------------

@lru_cache(maxsize=4)
def load_sources(path: str | Path = DEFAULT_MODELS_JSON) -> dict[str, Any]:
    """{source_name: {split: str | None}} from the `sources` section."""
    return json.loads(Path(path).read_text()).get("sources", {})


def split_for_source(source: str,
                     path: str | Path = DEFAULT_MODELS_JSON) -> str | None:
    """Parquet split for a model `source`. None means the source is not
    published to the dataset (build_hf_dataset.py skips its rows).
    KeyError if the source isn't declared."""
    return load_sources(path)[source]["split"]


# --- SNR analysis params ----------------------------------------------------

@lru_cache(maxsize=4)
def load_snr_params(path: str | Path = DEFAULT_MODELS_JSON) -> dict[str, Any]:
    """The `snr` section: small_sizes, target_size, plotted_mixes,
    da_early_fracs, last_n, size_buckets. Global — there is no per-pool
    override."""
    return json.loads(Path(path).read_text())["snr"]


@lru_cache(maxsize=4)
def _bucket_map(path: str | Path = DEFAULT_MODELS_JSON) -> dict[str, str]:
    """raw size label → bucket label, from snr.size_buckets (order preserved)."""
    out: dict[str, str] = {}
    for b in load_snr_params(path).get("size_buckets", []):
        for s in b["sizes"]:
            out[s] = b["label"]
    return out


def size_bucket(size: str, path: str | Path = DEFAULT_MODELS_JSON) -> str:
    """Map a raw size label (e.g. ``7B``, ``8B``) to its analysis bucket
    (``7-9B``). Nearby large sizes share a bucket so each has ≥2 models for
    cross-model signal/noise; custom sizes are singleton buckets. Unknown
    labels pass through unchanged."""
    return _bucket_map(path).get(size, size)


def bucket_order(path: str | Path = DEFAULT_MODELS_JSON) -> list[str]:
    """Bucket labels in ascending-size order (config order)."""
    return [b["label"] for b in load_snr_params(path).get("size_buckets", [])]


_EXTERNAL_SOURCES = (
    "huggingface-reference", "swiss-ai-reference",
    "snr-pretraining-a06", "distillation",
)


def stage_external_models(stage: str,
                          path: str | Path = DEFAULT_MODELS_JSON) -> set[str]:
    """Model names from the external sources (reference_hf / a06 / distillation)
    that have the given ``stage`` declared. Used to keep, e.g., instruct
    checkpoints out of the *pretraining* SNR pool when folding externals — the
    reference_hf parquet ships both pretraining and posttraining models."""
    out: set[str] = set()
    for src in _EXTERNAL_SOURCES:
        out |= set(filter_models(source=src, stage=stage, path=path))
    return out


# --- HF / W&B infra config (configs/hf_wandb.json) --------------------------

@lru_cache(maxsize=4)
def load_hf_wandb_config(path: str | Path = DEFAULT_HF_WANDB_JSON) -> dict[str, Any]:
    """Published-dataset + infra config: repo_id, parquet_pattern, wandb
    {entity, project}, multilingual_evals {raw_source_repos, model_dirs}."""
    return json.loads(Path(path).read_text())


def parquet_name(split: str, path: str | Path = DEFAULT_HF_WANDB_JSON) -> str:
    """Parquet filename for a split, e.g. `pretraining_custom` →
    `pretraining_custom-00000-of-00001.parquet`."""
    return load_hf_wandb_config(path)["parquet_pattern"].format(split=split)


# --- Family helpers ---------------------------------------------------------

def _strip_size_from_name(model: str, size: str) -> str:
    """Drop the ``size`` token (with adjacent dashes/slashes) from a
    model name to produce a cross-size identity. See run_apertus_snr_variants
    for the original implementation; this is the canonical home.

    Examples:
        apertus-175M-fwEdu30-fw270-seed1904 → apertus-fwEdu30-fw270-seed1904
        allenai/DataDecide-c4-150M          → allenai/DataDecide-c4
        SmolLM3-3B-Base                     → SmolLM3-Base
    """
    if not isinstance(model, str) or not isinstance(size, str):
        return model
    if size not in _SIZE_TOKENS:
        return model
    pattern = re.compile(rf"(?:^|(?<=[-/])){re.escape(size)}(?=[-/]|$)")
    new = pattern.sub("", model)
    new = re.sub(r"-{2,}", "-", new).strip("-/")
    return new or model


def family_of(model_name: str, size: str | None = None,
              path: str | Path = DEFAULT_MODELS_JSON) -> str:
    """Cross-size identity for a model.

    1. If `model_name` is in models.json → return its declared `family`.
    2. Otherwise → auto-derive via `_strip_size_from_name(model, size)`.
       The caller must supply `size` for this branch (it's needed to
       know which token to strip).
    """
    models = load_models(path)
    if model_name in models:
        return models[model_name]["family"]
    if size is None:
        raise ValueError(
            f"model {model_name!r} not in models.json; pass `size` to "
            "fall back to auto-derive."
        )
    return _strip_size_from_name(model_name, size)


def add_family_column(df, path: str | Path = DEFAULT_MODELS_JSON):
    """Attach `family` column to a per-(model, …) df. Idempotent.

    Used everywhere DA is computed — Apertus, AllenAI, HF reference all
    flow through this. Pandas is imported lazily so the cluster scripts
    can import this module without it on the side that just wants
    `iters_for` / `tokens_for`.
    """
    if "family" in df.columns:
        return df
    models = load_models(path)
    df = df.copy()
    df["family"] = [
        models[m]["family"] if m in models
        else _strip_size_from_name(m, s)
        for m, s in zip(df["model"], df["size"])
    ]
    return df


# --- Stages & checkpoints ---------------------------------------------------
#
# Each model has a `stages` dict — {<phase>: {tokens, num_iters,
# tokens_per_iter, checkpoints}} — where <phase> is pretraining /
# midtraining / posttraining. `checkpoints` holds `final` plus the subset
# lists below (ints for megatron_iter models, branch-name strings for
# hf_branch models).

_VALID_CKPT_SUBSETS = ("all", "dense_tail", "10_ckpts", "da_ckpts", "full_eval")

# step number embedded in an HF branch name (`stepN`, `step-N`,
# `stageK-step-N`, `stepN-tokensXXX`); `-tokensXXX(B|T)` explicit count.
_BRANCH_STEP_RE = re.compile(r"step-?(\d+)")
_BRANCH_TOKENS_RE = re.compile(r"-tokens([\d.]+)([BT])")


def stages_of(model_name: str,
              path: str | Path = DEFAULT_MODELS_JSON) -> dict:
    """The model's `stages` dict — {phase: {tokens, num_iters,
    tokens_per_iter, checkpoints}}."""
    return get_model(model_name, path)["stages"]


def _stage_containing(entry: dict, ckpt_id) -> dict:
    """The `stages.<phase>` dict whose `checkpoints.all` lists `ckpt_id`.
    Falls back to the sole stage when the model has exactly one (covers a
    megatron eval at a non-canonical iter — tokens = iter × tpi is still
    exact). Raises KeyError when no stage matches a multi-stage model."""
    stages = entry["stages"]
    for sdata in stages.values():
        if ckpt_id in sdata["checkpoints"].get("all", []):
            return sdata
    if len(stages) == 1:
        return next(iter(stages.values()))
    raise KeyError(f"ckpt {ckpt_id!r} not in any stage's checkpoints")


def iters_for(model_name: str, subset: str = "all", stage: str | None = None,
              path: str | Path = DEFAULT_MODELS_JSON) -> list:
    """Return the checkpoint list for `model_name`.

    For `checkpoint_kind="megatron_iter"`: returns `list[int]` of iters.
    For `checkpoint_kind="hf_branch"`: returns `list[str]` of branch names.

    `subset` selects the named subset under a stage's `checkpoints/` (falls
    back to `all` if absent). `stage` selects one phase; when None, the
    union across every stage is returned (in stage-declaration order) —
    single-stage models therefore just get that stage's subset.
    """
    if subset not in _VALID_CKPT_SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; "
                         f"valid: {_VALID_CKPT_SUBSETS}")
    stages = stages_of(model_name, path)
    if stage is not None:
        ck = stages[stage]["checkpoints"]
        return ck.get(subset, ck.get("all", []))
    out: list = []
    for sdata in stages.values():
        ck = sdata["checkpoints"]
        out.extend(ck.get(subset, ck.get("all", [])))
    return out


def _hf_branch_tokens(branch: str, sdata: dict) -> int | float | None:
    """Cumulative tokens for one HF branch within its stage:
    `main` → stage total; `-tokensXXX(B|T)` in the name → that explicit
    count; `stepN` → `step × stage.tokens_per_iter` (linear interp)."""
    if branch == "main":
        return sdata["tokens"]
    m = _BRANCH_TOKENS_RE.search(branch)
    if m:
        return float(m.group(1)) * (1e9 if m.group(2) == "B" else 1e12)
    m = _BRANCH_STEP_RE.search(branch)
    if m and sdata["tokens_per_iter"] is not None:
        return int(m.group(1)) * sdata["tokens_per_iter"]
    return None


def tokens_for(model_name: str, ckpt_id,
               path: str | Path = DEFAULT_MODELS_JSON) -> int | float | None:
    """Cumulative training tokens at `ckpt_id` (an iter int for
    megatron_iter models, a branch-name string for hf_branch models).

    - megatron_iter: `iter × stage.tokens_per_iter` — exact for any iter.
    - hf_branch / hf_local: `ckpt_id` must be a branch declared in some
      stage's `checkpoints.all`; raises KeyError otherwise (so callers can
      reject NAMEs that don't correspond to a canonical checkpoint).
    """
    entry = get_model(model_name, path)
    kind = entry["checkpoint_kind"]

    if kind == "megatron_iter":
        if not isinstance(ckpt_id, int):
            raise TypeError(f"megatron_iter expects int ckpt_id, "
                            f"got {ckpt_id!r}")
        tpi = _stage_containing(entry, ckpt_id)["tokens_per_iter"]
        return ckpt_id * tpi if tpi is not None else None

    for sdata in entry["stages"].values():
        if ckpt_id in sdata["checkpoints"].get("all", []):
            return _hf_branch_tokens(ckpt_id, sdata)
    raise KeyError(f"branch {ckpt_id!r} not in {model_name}'s checkpoints")


# --- Tasks ------------------------------------------------------------------

@lru_cache(maxsize=4)
def load_tasks(path: str | Path = DEFAULT_TASKS_JSON) -> dict[str, Any]:
    return json.loads(Path(path).read_text())["tasks"]


@lru_cache(maxsize=4)
def _load_groups(path: str | Path = DEFAULT_TASKS_JSON) -> dict[str, list[str]]:
    return json.loads(Path(path).read_text())["groups"]


def tasks_for_group(group: str,
                    path: str | Path = DEFAULT_TASKS_JSON) -> list[str]:
    return _load_groups(path)[group]


def metric_for(task: str,
               path: str | Path = DEFAULT_TASKS_JSON) -> str | None:
    """Task-declared metric override (e.g. ifeval → 'exact_match').
    None when the task doesn't pin a metric — callers fall back to the
    historical `acc` → `exact_match` heuristic."""
    return load_tasks(path).get(task, {}).get("metric")
