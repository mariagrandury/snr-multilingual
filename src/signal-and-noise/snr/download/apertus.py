"""Load eval results for the Apertus / a06 / reference-HF runs from local parquet.

The `multilingual-snr/multilingual-snr-eval-results` HuggingFace dataset
ships three parquet splits already in the schema expected by
`snr.dataloader.get_slice`:

  - data/pretraining_custom-*.parquet   (36 Apertus pretrained models;
        sizes 175M/350M/600M/1B; mixes fwEdu30/60/90; seeds 28/1797/1904)
  - data/pretraining_a06-*.parquet      (a06 main pretraining runs:
        apertus3-{1b,3b}-*-nodes)
  - data/reference_hf-*.parquet         (external HF reference models:
        SmolLM3-3B (incl. step-checkpoints), Olmo-3-7B, Apertus-8B)

The cluster-walking implementation that used to live here has been
superseded by these tables — every value the parser used to derive
(``model``, ``mix``, ``size``, ``step``, ``primary_score``, ``tokens``,
``compute``) is already a column.

Mix names in the parquet keep the full ``fwEduX-fwY`` form. The rest of
the pipeline (run_apertus.py, analyze_snr_variants.py, smooth_subtasks*)
only ever uses the FW-Edu ratio, so we strip the ``-fwY`` complement on
load to match the historical short form (``fwEdu30``).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

from snr.constants import DATA_DIR

# Shared configs loader (lives under src/evals/scripts/utils/). One-shot
# `src/` on sys.path so `from evals.scripts.utils.configs import …`
# resolves via Python's implicit namespace packages — same content
# whether the script is invoked from cluster or local.
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from evals.scripts.utils.configs import (  # noqa: E402
    add_family_column,
    parquet_name,
    split_for_source,
)

DEFAULT_DATA_DIR = Path(
    os.environ.get("SNR_MULTILINGUAL_DATA_DIR", DATA_DIR / "multilingual_snr" / "data")
)


def _parquet_path(data_dir: str | Path, source: str) -> Path:
    """<data_dir>/<split-for-source>-00000-of-00001.parquet — both the
    source→split map and the filename pattern come from configs/."""
    return Path(data_dir) / parquet_name(split_for_source(source))

# Used by smooth_subtasks_per_sample.py (which scans samples_*.jsonl
# directly — cluster-only path; not consumed by the parquet loaders
# below). Params/tokens-per-iter constants for the FLOPs / token axes
# moved to configs/models.json + src/evals/scripts/utils/configs.py
# (use `tokens_for(name, ckpt)` / `get_model(name)["params"]`).
DEFAULT_EVAL_ROOT = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/"
    "mariagrandury-epflnlp/snr-experiments"
)

# Match historical regex (still used by per-sample script when run on the
# cluster).
import re  # noqa: E402

_MODEL_RE = re.compile(
    r"^apertus-(?P<size>175M|350M|600M|1B)-fwEdu(?P<edu>30|60|90)-fw(?P<fw>270|240|210)-seed(?P<seed>\d+)-iter(?P<iter>\d+)$"
)


def _normalise_mix(mix: str) -> str:
    """fwEdu30-fw270 → fwEdu30  (drop the redundant complement)."""
    if isinstance(mix, str) and "-" in mix:
        return mix.split("-", 1)[0]
    return mix


# Size-label aliases — ``0.6B`` and ``600M`` are the same 600,000,000-param
# scale (the distilled ap-from8b-0.6b run and the custom 600M models). The
# DA / SNR pipeline groups on the exact `size` string, so canonicalise to the
# custom-grid label `600M` to let the distilled family join the 600M tier.
_SIZE_ALIASES = {"0.6B": "600M"}


def _normalise_size(s: str) -> str:
    return _SIZE_ALIASES.get(s, s) if isinstance(s, str) else s


def _size_key(s: str) -> float:
    """``175M`` → 0.175, ``1B`` → 1.0, ``7B`` → 7.0. Used for numeric
    size ordering — lexicographic sort would put ``1B`` before ``350M``."""
    if not isinstance(s, str):
        return float("nan")
    s = s.strip()
    try:
        if s.endswith("M"):
            return float(s[:-1]) / 1000.0
        if s.endswith("B"):
            return float(s[:-1])
    except ValueError:
        pass
    return float("nan")


def _read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Schema sanity: keep the columns the SNR pipeline expects, drop the
    # rest so downstream pivots stay cheap.
    keep = [
        "model",
        "size",
        "mix",
        "seed",
        "step",
        "task",
        "primary_score",
        "model_tokens",
        "flops",
    ]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["mix"] = df["mix"].map(_normalise_mix)
    df["size"] = df["size"].map(_normalise_size)
    df = df.rename(columns={"model_tokens": "tokens", "flops": "compute"})
    # Coerce numerics — the reference_hf split has NaN seeds.
    for c in ("step", "primary_score", "tokens", "compute"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df = df.dropna(subset=["primary_score", "step"])
    # Numeric size sort (string sort would give 175M, 1B, 350M, 600M).
    df = (
        df.assign(_size_num=df["size"].map(_size_key))
        .sort_values(["_size_num", "mix", "step", "task"])
        .drop(columns="_size_num")
        .reset_index(drop=True)
    )
    # Attach the cross-size identity once at load time so every consumer
    # sees `family` (matches configs/models.json declarations for known
    # models; falls back to size-stripping for dynamic rows e.g. AllenAI).
    return add_family_column(df)


def load_apertus_eval_results(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """One row per (model, ckpt, task) for the 36 Apertus custom pretrains
    (4 sizes × 3 mixes × 3 seeds)."""
    return _read_parquet(_parquet_path(data_dir, "snr-pretraining-custom"))


def load_a06_eval_results(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """a06 main pretraining runs (apertus3-{1b,3b}-*-nodes). Columns match
    `load_apertus_eval_results` — `mix` is always ``main``, `seed` is
    NaN, and `step` is the megatron iter count."""
    return _read_parquet(_parquet_path(data_dir, "snr-pretraining-a06"))


def load_distillation_eval_results(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Distillation runs (apertus-{0.6b,1b}-from8b-TOP256-long). Columns
    match `load_apertus_eval_results` — `mix` is always ``main``, `seed`
    is NaN, and `step` is the megatron iter count."""
    return _read_parquet(_parquet_path(data_dir, "distillation"))


def load_reference_hf_eval_results(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Reference HF runs (SmolLM3, Olmo-3, Apertus-8B). Columns match
    `load_apertus_eval_results` — `mix` is the HF "stage1"/"main", and
    `step` is the model's training step (not directly comparable to
    Apertus iter counts). `swiss-ai-reference` shares the same parquet
    split, so either source resolves to it."""
    return _read_parquet(_parquet_path(data_dir, "huggingface-reference"))


def load_all_eval_results(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Concatenated pretraining_custom + pretraining_a06 + reference_hf,
    with a `source` column to distinguish them. Use when an analysis
    benefits from pooling extra step-series data points (DA-ckpt, noise
    estimates, rank correlations) — the per-mix SNR computations themselves
    only apply within each source's own (size, mix) grid."""
    parts = []
    for src, loader in (
        ("apertus", load_apertus_eval_results),
        ("a06", load_a06_eval_results),
        ("reference_hf", load_reference_hf_eval_results),
    ):
        try:
            df = loader(data_dir)
        except FileNotFoundError:
            continue
        df["source"] = src
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out = (
        out.assign(_size_num=out["size"].map(_size_key))
        .sort_values(["source", "_size_num", "mix", "step", "task"])
        .drop(columns="_size_num")
        .reset_index(drop=True)
    )
    return out
