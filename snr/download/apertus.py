"""Load eval results for the Apertus / reference-HF runs from local parquet.

The `multilingual-snr/multilingual-snr-eval-results` HuggingFace dataset
ships two parquet splits already in the schema expected by
`snr.dataloader.get_slice`:

  - data/pretraining_custom-*.parquet   (12 Apertus pretrained models;
        sizes 175M/350M/600M/1B; mixes fwEdu30/60/90; seed 1904; 13 ckpts)
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
from pathlib import Path

import pandas as pd

from snr.constants import DATA_DIR

DEFAULT_DATA_DIR = Path(
    os.environ.get("SNR_MULTILINGUAL_DATA_DIR", DATA_DIR / "multilingual_snr" / "data")
)
PRETRAIN_PARQUET = "pretraining_custom-00000-of-00001.parquet"
REFERENCE_PARQUET = "reference_hf-00000-of-00001.parquet"

# Backwards-compat aliases used by smooth_subtasks_per_sample.py (which
# still scans samples_*.jsonl — that path only works on the cluster).
DEFAULT_EVAL_ROOT = Path(
    "/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/"
    "mariagrandury-epflnlp/snr-experiments"
)
_PARAMS = {"175M": 175e6, "350M": 350e6, "600M": 600e6, "1B": 1.0e9}
_TOKENS_PER_ITER = 504 * 4096

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
    keep = ["model", "size", "mix", "seed", "step", "task",
            "primary_score", "model_tokens", "flops"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["mix"] = df["mix"].map(_normalise_mix)
    df = df.rename(columns={"model_tokens": "tokens", "flops": "compute"})
    # Coerce numerics — the reference_hf split has NaN seeds.
    for c in ("step", "primary_score", "tokens", "compute"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    df = df.dropna(subset=["primary_score", "step"])
    # Numeric size sort (string sort would give 175M, 1B, 350M, 600M).
    df = df.assign(_size_num=df["size"].map(_size_key)).sort_values(
        ["_size_num", "mix", "step", "task"]
    ).drop(columns="_size_num").reset_index(drop=True)
    return df


def load_apertus_eval_results(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """One row per (model, ckpt, task) for the 12 Apertus custom pretrains."""
    return _read_parquet(Path(data_dir) / PRETRAIN_PARQUET)


def load_reference_hf_eval_results(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Reference HF runs (SmolLM3, Olmo-3, Apertus-8B). Columns match
    `load_apertus_eval_results` — `mix` is the HF "stage1"/"main", and
    `step` is the model's training step (not directly comparable to
    Apertus iter counts)."""
    return _read_parquet(Path(data_dir) / REFERENCE_PARQUET)


def load_all_eval_results(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> pd.DataFrame:
    """Concatenated pretraining_custom + reference_hf, with a `source`
    column to distinguish them. Use when an analysis benefits from
    pooling extra step-series data points (DA-ckpt, noise estimates,
    rank correlations) — the per-mix SNR computations themselves only
    apply within each source's own (size, mix) grid."""
    a = load_apertus_eval_results(data_dir)
    r = load_reference_hf_eval_results(data_dir)
    a["source"] = "apertus"
    r["source"] = "reference_hf"
    out = pd.concat([a, r], ignore_index=True)
    out = out.assign(_size_num=out["size"].map(_size_key)).sort_values(
        ["source", "_size_num", "mix", "step", "task"]
    ).drop(columns="_size_num").reset_index(drop=True)
    return out
