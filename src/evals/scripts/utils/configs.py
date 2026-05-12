"""Shared loader for configs/models.json + configs/tasks.json.

Single source of truth for the eval / pretrain / signal-and-noise
pipelines. See .claude-shared/plans/models-tasks-json-refactor.md for
the full plan.

Path conventions:
- evals + pretrain code runs on the cluster (this file is read from
  /iopsstor/scratch/cscs/mariagrandury/snr-multilingual/src/evals/scripts/utils/).
- signal-and-noise code runs locally on the Mac
  (/Users/mariagrandury/Projects/epfl/snr-multilingual/src/signal-and-noise/...).
- configs/models.json + configs/tasks.json live at the SHARED repo
  root — same content readable from either host.
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
        if stage is not None and entry.get("stage") != stage:
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


# --- Checkpoints ------------------------------------------------------------

_VALID_CKPT_SUBSETS = ("all", "dense_tail", "10_ckpts", "da_ckpts", "full_eval")


def iters_for(model_name: str, subset: str = "all",
              path: str | Path = DEFAULT_MODELS_JSON) -> list:
    """Return the checkpoint list for `model_name`.

    For `checkpoint_kind="megatron_iter"`: returns `list[int]` of iters.
    For `checkpoint_kind="hf_branch"`: returns `list[{"branch": str,
    "tokens": int | None}]`.

    `subset` selects the named subset under `checkpoints/`. `"full_eval"`
    is an alias for whichever subset the eval pipeline currently
    canonicalises on — default is `all`. Falls back to `all` if the
    requested subset is absent.
    """
    if subset not in _VALID_CKPT_SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; "
                         f"valid: {_VALID_CKPT_SUBSETS}")
    entry = get_model(model_name, path)
    ckpts = entry["checkpoints"]
    if subset == "full_eval":
        subset = "all"
    return ckpts.get(subset, ckpts.get("all", []))


def tokens_for(model_name: str, ckpt_id,
               hyperparams_path: str | Path | None = None,
               path: str | Path = DEFAULT_MODELS_JSON) -> int | None:
    """Tokens at `ckpt_id`. Branches on `checkpoint_kind`:

    - `megatron_iter`: ckpt_id is an iter (int). Tokens =
      iter × global_batch_size × seq_len, read from hyperparams_deep.json
      (default: <repo>/src/pretrain/hyperparams_deep.json).
    - `hf_branch`: ckpt_id is a branch string. Tokens is the explicit
      value stored on that branch entry (may be None).
    """
    entry = get_model(model_name, path)
    kind = entry["checkpoint_kind"]
    if kind == "megatron_iter":
        if not isinstance(ckpt_id, int):
            raise TypeError(f"megatron_iter expects int ckpt_id, "
                            f"got {ckpt_id!r}")
        return ckpt_id * _meg_tokens_per_step(hyperparams_path)
    if kind in ("hf_branch", "hf_local"):
        for branch_entry in entry["checkpoints"]["all"]:
            if branch_entry["branch"] == ckpt_id:
                return branch_entry.get("tokens")
        raise KeyError(f"branch {ckpt_id!r} not in {model_name}'s checkpoints")
    raise ValueError(f"Unknown checkpoint_kind {kind!r} for {model_name!r}")


@lru_cache(maxsize=2)
def _meg_tokens_per_step(path: str | Path | None = None) -> int:
    if path is None:
        path = _REPO_ROOT / "src" / "pretrain" / "hyperparams_deep.json"
    g = json.loads(Path(path).read_text())["global"]
    return g["global_batch_size"] * g["seq_len"]


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
