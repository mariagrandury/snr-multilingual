"""Recompute the 22 SNR variants on the AllenAI DataDecide / OLMo `core` split.

This is the AllenAI counterpart of `multilingual/run_apertus_snr_variants.py`.
We reuse all of that script's primitives — `per_mix_inputs`,
`variant_signal_noise_snr`, `variant_key`, `compute_size_decision_accuracy`,
the `AGGREGATION_FUNCTIONS` list — and only swap the DataFrame source.

Output: `allenai_snr_variants_per_task.csv` with the same wide format
(one row per task; columns `signal_<V>_<size>` / `noise_<V>_<size>` /
`snr_<V>_<size>` for every variant V × size, plus
`decision_acc_size_<small>` for each small size).

Sizes used here are the conventional DataDecide ladder small sizes
(150M / 300M / 750M) targeting 1B, all of which have 5 ckpts × 25 mixes.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd
from tqdm import tqdm

from snr.constants import PLOT_DIR
from snr.download.hf import pull_predictions_from_hf
from snr.snr_variants import AGGREGATION_FUNCTIONS

# Reuse the Apertus driver's helpers verbatim — same shape of inputs.
# The Apertus driver renamed `per_mix_inputs` → `per_model_inputs` when
# it generalised the signal pool to "any unique model at this size";
# AllenAI's per-(mix, seed) DataDecide runs are uniquely identified by
# their `model` column, so the same per-model grouping applies.
from analysis.rq02_snr_definition.run_apertus_snr_variants import (
    _safe,
    compute_size_decision_accuracy,
    per_model_inputs,
    variant_key,
    variant_signal_noise_snr,
)

SMALL_SIZES = ["150M", "300M", "750M"]
TARGET_SIZE = "1B"
ALL_SIZES = SMALL_SIZES + [TARGET_SIZE]
OUT_DIR = PLOT_DIR / "allenai_comparison"
OUT_CSV = OUT_DIR / "allenai_snr_variants_per_task.csv"


def load_allenai_core() -> pd.DataFrame:
    """Pull the AllenAI `core` parquet and shape it like the Apertus loader.

    Keep only `model_type='datadecide'` rows at sizes 150M/300M/750M/1B
    (the rows used in the small→target SNR computation). Drop the
    `peteish-ladder` and external rows — they have a different (mix,
    seed, ckpt) structure and the SNR variants don't apply.
    """
    path = pull_predictions_from_hf("allenai/signal-and-noise", split_name="core")
    df = pd.read_parquet(path)

    # Restrict to the DataDecide ladder at the four sizes we care about.
    # `peteish-ladder` rows (model_type=='ladder') and external rows are
    # dropped — they don't have the (mix × ckpt) grid the SNR variants
    # consume, and including them would let scaling-law ladder steps
    # leak into the per-mix groupings.
    df = df[df["model_type"] == "datadecide"]
    df = df[df["size"].isin(ALL_SIZES)].copy()

    df = df.dropna(subset=["primary_score", "step", "size", "mix", "task"])

    # Coerce to the same dtypes the Apertus loader uses.
    df["step"] = pd.to_numeric(df["step"], errors="coerce").astype("Int64")
    df["primary_score"] = pd.to_numeric(df["primary_score"], errors="coerce")
    df = df.dropna(subset=["step", "primary_score"]).copy()
    df["step"] = df["step"].astype(int)

    return df.sort_values(["size", "mix", "step", "task"]).reset_index(drop=True)


def _canonical_seed(df: pd.DataFrame) -> int | None:
    """Pick the most-common seed in the slice — used to pin per-mix
    aggregations to a single training seed on multi-seed corpora. Returns
    None if the seed column is absent or all-NaN; passing None to
    ``per_mix_inputs`` then disables seed filtering (legacy behavior)."""
    if "seed" not in df.columns:
        return None
    seeds = df["seed"].dropna()
    if seeds.empty:
        return None
    return int(seeds.mode().iloc[0])


def run() -> Path:
    df = load_allenai_core()
    tasks = sorted(df["task"].unique())
    seed = _canonical_seed(df)
    seed_counts = (df["seed"].value_counts(dropna=False).to_dict()
                   if "seed" in df.columns else {})
    print(
        f"Loaded {len(df):,} rows | "
        f"{df['model'].nunique()} models | "
        f"{df['mix'].nunique()} mixes | "
        f"{len(tasks)} tasks | "
        f"canonical seed: {seed} (counts: {seed_counts})"
    )

    # Pre-filter to the canonical seed once; `per_model_inputs` and
    # `compute_size_decision_accuracy` then operate on a single-seed slice
    # (matching the legacy behavior).
    df_seed = df[df["seed"] == seed].copy() if seed is not None else df

    # AllenAI rows aren't in configs/models.json, so the model_filter
    # equivalent for the new DA functions is the full set of AllenAI
    # model names — equivalent to no filter for this corpus.
    model_filter = set(df_seed["model"].unique())
    rows: list[dict] = []
    for task in tqdm(tasks, desc="Tasks"):
        row = {"task": task}

        # Size-DA: rank at small@last vs target@last.
        for s in SMALL_SIZES:
            row[f"decision_acc_size_{s}"] = _safe(
                compute_size_decision_accuracy, df_seed, task, s, TARGET_SIZE,
                model_filter=model_filter,
            )

        # 22 variants × 4 sizes × 3 stats.
        size_inputs = {s: per_model_inputs(df_seed, task, s) for s in ALL_SIZES}
        for fd in AGGREGATION_FUNCTIONS:
            key = variant_key(fd)
            for s in ALL_SIZES:
                sig, noi, snr = variant_signal_noise_snr(size_inputs[s], fd["func"])
                row[f"signal_{key}_{s}"] = sig
                row[f"noise_{key}_{s}"] = noi
                row[f"snr_{key}_{s}"] = snr
        rows.append(row)

    out = pd.DataFrame(rows).set_index("task").sort_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV)
    print(
        f"\nWrote CSV → {OUT_CSV}\n"
        f"  {len(out)} tasks × {len(out.columns)} columns "
        f"({len(AGGREGATION_FUNCTIONS)} variants × {len(ALL_SIZES)} sizes × 3 stats "
        f"+ {len(SMALL_SIZES)} size-DA)"
    )
    return OUT_CSV


if __name__ == "__main__":
    run()
