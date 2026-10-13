"""Recompute the 22 SNR variants on AllenAI's DataDecide ladder, like for like.

This is the AllenAI counterpart of `analysis/rq03_noise_and_snr/run_apertus_snr_variants.py`
and reuses its primitives (`per_model_inputs`, `ckpt_noise_columns`,
`variant_signal_noise_snr`, `variant_key`, `compute_size_decision_accuracy`,
`AGGREGATION_FUNCTIONS`); only the DataFrame source differs. Three choices
make the two sides comparable (rebuilt 2026-10-08; the June table read the
`core` split's last five checkpoints on each task's own primary metric):

  * The source is the `datadecide_intermediate` split, which holds every
    2,500-step checkpoint of each run, so the noise window is ours (rule 4,
    `utils.noise_checkpoints`): the k/20 points in the last 20 % of the run,
    5 points at 1B (55,000-69,369 of 69,369). The `core` split keeps only
    the last 5 checkpoints (86.5-100 % at 1B), 3-4 points on that grid.
  * The noise is ours: `utils.checkpoint_noise`, the residual SD around a
    line through the window (the raw SD is written beside it as
    `ckpt_noise_raw_<size>`, as in rq03).
  * The metric is ours: each DataDecide task is read on the column that
    matches the metric our matched task is scored on (`APERTUS_TASK`,
    `configs.accuracy_metric`): `acc_norm` (lm-eval's per-character
    normalisation) -> `acc_per_char`, `acc` -> `acc_raw`. DataDecide's own
    primary metric (`acc_uncond` on ARC-C, OBQA, CSQA) is not used.

The signal populations still differ: 25 pretraining corpora at one seed
(6198) against our design variants. The table covers DataDecide's ten
OLMES tasks.

Output: `allenai_snr_variants_per_task.csv`, one row per task, columns
`signal_<V>_<size>` / `noise_<V>_<size>` / `snr_<V>_<size>` for every
variant V and size, `ckpt_noise_detrended_<size>` / `ckpt_noise_raw_<size>`,
`decision_acc_size_<small>` / `n_pairs_size_<small>` for each small size, `metric` and `n_window`
(the median window points per run at 1B).

Sizes: the DataDecide small sizes 150M / 300M / 750M against 1B.

    python analysis/rq07_external_frameworks/build_allenai_variants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent   # signal-and-noise
_SRC = _REPO.parent                                       # src (for `evals`)
for _p in (_REPO, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd
from tqdm import tqdm

from snr.constants import PLOT_DIR
from analysis.paths import EXTERNAL_FRAMEWORKS
from snr.download.hf import pull_predictions_from_hf
from snr.snr_variants import AGGREGATION_FUNCTIONS
from evals.scripts.utils.configs import accuracy_metric, size_bucket
from analysis.utils import noise_checkpoints

# Reuse the Apertus driver's helpers verbatim — same shape of inputs.
# The Apertus driver renamed `per_mix_inputs` → `per_model_inputs` when
# it generalised the signal pool to "any unique model at this size";
# AllenAI's per-(mix, seed) DataDecide runs are uniquely identified by
# their `model` column, so the same per-model grouping applies.
from analysis.rq02_decision_accuracy.compute_da import (
    _safe,
    compute_size_decision_accuracy,
)
from analysis.rq03_noise_and_snr.run_apertus_snr_variants import (
    ckpt_noise_columns,
    per_model_inputs,
    variant_key,
    variant_signal_noise_snr,
)

SMALL_SIZES = ["150M", "300M", "750M"]
TARGET_SIZE = "1B"
ALL_SIZES = SMALL_SIZES + [TARGET_SIZE]
OUT_DIR = EXTERNAL_FRAMEWORKS
OUT_CSV = OUT_DIR / "allenai_snr_variants_per_task.csv"
SPLIT = "datadecide_intermediate"
# DataDecide task -> our task scored on the same items in the same (cloze)
# format; rq07's analyze.py reads the inverse to name our rows. MMLU and CSQA
# are matched to our cloze twins, since DataDecide's RC format is a cloze.
APERTUS_TASK = {"arc_challenge": "arc_challenge", "arc_easy": "arc_easy", "boolq": "boolq",
                "csqa": "rf_commonsense_qa", "hellaswag": "hellaswag", "mmlu": "rf_mmlu",
                "openbookqa": "openbookqa", "piqa": "piqa", "socialiqa": "social_iqa",
                "winogrande": "winogrande"}
DATADECIDE_METRIC = {"acc_norm": "acc_per_char", "acc": "acc_raw"}


def load_allenai() -> pd.DataFrame:
    """The DataDecide checkpoint series at the four sizes, shaped like the
    ladder loader: `primary_score` is the column matching our metric
    (module docstring), `frac` each checkpoint's share of its run."""
    path = pull_predictions_from_hf("allenai/signal-and-noise", split_name=SPLIT)
    df = pd.read_parquet(path)
    df = df[df["size"].isin(ALL_SIZES) & df["task"].isin(APERTUS_TASK)].copy()
    df["metric"] = df["task"].map(lambda t: DATADECIDE_METRIC[accuracy_metric(APERTUS_TASK[t]) or "acc"])
    df["primary_score"] = [r[m] for r, m in zip(df.to_dict("records"), df["metric"])]
    df = df.dropna(subset=["primary_score", "step"])
    df["step"] = df["step"].astype(int)
    df["frac"] = df["step"] / df.groupby(["model", "seed"])["step"].transform("max")
    # per_model_inputs / compute_size_decision_accuracy select on `bucket`;
    # the DataDecide sizes are not in the bucket map and pass through unchanged
    df["bucket"] = df["size"].map(size_bucket)
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
    df = load_allenai()
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
            row[f"decision_acc_size_{s}"], row[f"n_pairs_size_{s}"] = _safe(   # _safe returns (DA, pairs)
                compute_size_decision_accuracy, df_seed, task, s, TARGET_SIZE,
                model_filter=model_filter,
            )

        # 22 variants × 4 sizes × 3 stats, on the rule-4 window and noise.
        dft = df_seed[df_seed["task"] == task]
        row["metric"] = dft["metric"].iloc[0]
        row["n_window"] = float(noise_checkpoints(dft[dft["size"] == TARGET_SIZE]).groupby("model").size().median())
        size_inputs = {s: per_model_inputs(dft, task, s) for s in ALL_SIZES}
        for fd in AGGREGATION_FUNCTIONS:
            key = variant_key(fd)
            for s in ALL_SIZES:
                sig, noi, snr = variant_signal_noise_snr(size_inputs[s], fd["func"])
                row[f"signal_{key}_{s}"] = sig
                row[f"noise_{key}_{s}"] = noi
                row[f"snr_{key}_{s}"] = snr
        for s in ALL_SIZES:
            row |= ckpt_noise_columns(dft, task, s, size_inputs[s])
        rows.append(row)

    out = pd.DataFrame(rows).set_index("task").sort_index()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV)
    print(
        f"\nWrote CSV → {OUT_CSV}\n"
        f"  {len(out)} tasks × {len(out.columns)} columns "
        f"({len(AGGREGATION_FUNCTIONS)} variants × {len(ALL_SIZES)} sizes × 3 stats "
        f"+ 2 ckpt-noise cols per size + {len(SMALL_SIZES)} size-DA and pair counts)\n"
        f"  window points per run at {TARGET_SIZE} (median): {out['n_window'].to_dict()}"
    )
    return OUT_CSV


if __name__ == "__main__":
    run()
