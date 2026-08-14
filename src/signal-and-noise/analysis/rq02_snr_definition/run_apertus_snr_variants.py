"""Generate the per-task table consumed by analyze_snr_variants.py.

For every aggregator in snr.snr_variants.AGGREGATION_FUNCTIONS, store the
three return values (signal, noise, snr) at every model size. Also store
two definitions of decision accuracy:

  size DA  — (mix, seed) ranking at <small>'s last ckpt vs the ranking at
             TARGET_SIZE's last ckpt (the upstream allenai definition,
             generalised to (mix, seed) "models").
             3 cols: decision_acc_size_<175M|350M|600M>.

  ckpt DA  — within a single size, (mix, seed) ranking at an early ckpt vs the
             same size's last ckpt. 3 early ckpts × 4 sizes = 12 cols:
             decision_acc_ckpt_<early>_<size> for early in
             CKPT_DA_EARLY_STEPS and size in ALL_SIZES.

**Signal/noise pool per size = every unique training run at that size**:
Apertus (mix, seed) runs whose ``seed`` is in ``--seeds`` *and* any
external reference models loaded from ``reference_hf`` at that size
(e.g. Qwen3-0.6B at 600M, Apertus-v1.5 at 1B). Groupby is ``model``,
so each unique training run contributes one signal datapoint and one
trailing-N noise sample. This matches AllenAI's >1B SNR_MODELS recipe
(heterogeneous open-source models at a fixed size); we extend it to
small scales by pooling Apertus seeds alongside externals.

**DA stays on the (mix, seed) Apertus axis** — DA needs models that
span both sizes for a cross-size rank comparison, and only the
controlled Apertus (mix, seed) tuples span all four sizes consistently.

``--seeds`` selects the Apertus seed pool (default: 1904);
``--include-external`` (default on) folds the reference_hf parquet
rows into the SNR signal pool; ``--out-subdir`` routes outputs to a
seed-specific folder so multiple seed pools can coexist under
``results/snr_definition/``.

The CSV is the single source of truth for analyze_snr_variants.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Put `src/` on sys.path so `evals.scripts.utils.configs` imports
# resolve via implicit namespace packages.
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
from tqdm import tqdm

from evals.scripts.utils.configs import (  # noqa: E402
    add_family_column,
    bucket_order,
    expand_pool,
    load_pools,
    load_snr_params,
    pool_include_external,
    size_bucket,
    stage_external_models,
)
from analysis.utils import (
    _ENGLISH_ONLY_TASKS, _is_language_aggregate, _is_parent_task,
    assign_language, benchmark_family, build_snr_pool,
)
from analysis.rq00_acc_vs_flops.above_random import SIZES as AR_SIZES, scores_and_mask
from snr.constants import PLOT_DIR
from analysis.paths import SNR_DEFINITION, DECISION_ACCURACY
from snr.dataloader import get_slice
from snr.download.apertus import (
    load_a06_eval_results,
    load_apertus_eval_results,
    load_distillation_eval_results,
    load_reference_hf_eval_results,
)
from snr.metrics import decision_acc_fast
from snr.snr_variants import AGGREGATION_FUNCTIONS

# SNR analysis params — single source of truth in configs/models.json.
# The signal/noise + DA size axis is the *bucket* (size_bucket()), so nearby
# large sizes (7B/8B → "7-9B") pool to ≥2 models. The custom small sizes are
# singleton buckets, so SMALL_SIZES / TARGET_SIZE double as bucket labels for
# the core holdout size-DA.
from analysis.utils import (  # noqa: E402
    SMALL_SIZES, TARGET_SIZE, LAST_N, CKPT_DA_EARLY_FRACS)
OUT_ROOT = SNR_DEFINITION


# --- per-model arrays for snr_variants --------------------------------------


def per_model_inputs(df, task, size, last_n=LAST_N):
    """Build the four per-model arrays expected by snr_variants aggregators.

    ``size`` is a *bucket* label and rows are selected on the ``bucket``
    column, so nearby large sizes (e.g. 7B + 8B → "7-9B") pool together.
    Each unique value of the ``model`` column is a separate training run —
    so the signal pool combines Apertus (mix, seed) runs with external /
    a06 / distillation models present in the bucket. ``df`` is assumed to
    already be filtered to the desired pool / external inclusion and to
    carry a ``bucket`` column.

    Mirrors analysis/snr_variants.ipynb cells 5+7:
      step_noise         = per-model std of the last `last_n` ckpts
      data_scores        = per-model final-ckpt score
      data_noise         = cross-model std of `data_scores`, broadcast as
                           a constant array of the same length
      data_scores_last_n = per-model mean of the last `last_n` ckpts

    Models with fewer than 2 ckpts are dropped (they can't contribute
    step_noise — e.g. a single-revision external like Apertus-70B-2509).
    We require ≥ 2 surviving models overall.
    """
    sub = df[(df["bucket"] == size) & (df["task"] == task)]
    if sub.empty:
        return None
    sub = sub.sort_values("step")
    grouped = sub.groupby("model")["primary_score"].apply(list)
    last_arrays = [np.asarray(s[-last_n:], dtype=float) for s in grouped]
    last_arrays = [a for a in last_arrays if len(a) >= 2]
    if len(last_arrays) < 2:
        return None
    step_noise = np.array([np.std(a) for a in last_arrays])
    data_scores = np.array([a[-1] for a in last_arrays])
    data_scores_last_n = np.array([a.mean() for a in last_arrays])
    data_noise = np.full_like(data_scores, np.std(data_scores))
    return step_noise, data_scores, data_noise, data_scores_last_n


def variant_signal_noise_snr(inputs, agg_func):
    if inputs is None:
        return float("nan"), float("nan"), float("nan")
    try:
        signal, noise, snr = agg_func(*inputs)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    out = []
    for v in (signal, noise, snr):
        try:
            f = float(v)
            out.append(f if np.isfinite(f) else float("nan"))
        except Exception:
            out.append(float("nan"))
    return out[0], out[1], out[2]


def variant_key(func_dict):
    """Stable column-name token, e.g. 'rel_std' from 'rel_std_snr'."""
    name = func_dict["func"].__name__
    return name[:-4] if name.endswith("_snr") else name


# --- variants metadata table ------------------------------------------------


def variants_definitions_df() -> pd.DataFrame:
    """One row per aggregator describing what its signal/noise/snr mean."""
    rows = []
    for fd in AGGREGATION_FUNCTIONS:
        rows.append(
            {
                "variant": variant_key(fd),
                "title": fd["title"],
                "latex": fd["latex"],
                "signal_label": fd["signal_xlabel"],
                "noise_label": fd["noise_xlabel"],
                "snr_label": fd["snr_xlabel"],
            }
        )
    return pd.DataFrame(rows).set_index("variant")


def write_variants_definitions(out_dir: Path) -> Path:
    """Write the metadata table to CSV and print a readable version."""
    df_def = variants_definitions_df()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "snr_variants_definitions.csv"
    df_def.to_csv(csv_path)
    # Readable view (markdown table; falls back to .to_string() if tabulate is missing).
    try:
        printable = df_def[["title", "latex", "snr_label"]].to_markdown()
    except Exception:
        printable = df_def[["title", "latex", "snr_label"]].to_string()
    print("\nSNR variant definitions:")
    print(printable)
    return csv_path


# --- driver -----------------------------------------------------------------


# _is_parent_task and build_snr_pool now live in analysis/utils.py (imported above).



def run(pool: str, out_dir: Path):
    # The pool drives the SNR signal/noise (custom-in-pool rows + folded-in
    # externals when include_external). The size axis is the *bucket*
    # (size_bucket): custom small sizes are singletons; larger sizes pool so
    # each bucket has ≥2 models. Decision accuracy is computed upstream by
    # rq01 (compute_da.py); this step reads that table and appends the SNR
    # variant columns — DA first (the truth), SNR second (the proxies).
    df_pool = build_snr_pool(pool)
    df_pool["bucket"] = df_pool["size"].map(size_bucket)
    all_tasks = sorted(df_pool["task"].unique())
    tasks = [t for t in all_tasks if _is_parent_task(t)]

    pool_buckets = [b for b in bucket_order() if b in set(df_pool["bucket"].dropna())]

    pool_models = set(expand_pool(pool))
    all_models = set(df_pool["model"])
    is_external = ~df_pool["model"].isin(pool_models)

    # Above-random gate (raw-metric competence check): keep a (task, custom-size)
    # SNR cell only if its mean score beats chance by the margin. Random cells
    # carry no usable signal, so their signal/noise/snr are NaN'd here — the gate
    # propagates to every analysis that reads this CSV. External-size buckets
    # (3B, 7-9B, …) aren't gated (no custom random baseline at those sizes).
    _, _ar_mask, _ = scores_and_mask(df_pool[df_pool["model"].isin(pool_models)])
    above_random = {(t, s) for s in AR_SIZES if s in _ar_mask.columns
                    for t in _ar_mask.index[_ar_mask[s] == 1]}
    pool_n_models = df_pool.groupby("bucket")["model"].nunique().to_dict()
    print(
        f"Pool '{pool}': {len(all_models & pool_models)} custom + "
        f"{len(all_models - pool_models)} external model(s); "
        f"include_external={pool_include_external(pool)}"
    )
    print(
        f"Loaded {len(df_pool):,} rows ({int(is_external.sum()):,} external) | "
        f"{len(tasks)} parent tasks (filtered from {len(all_tasks)} total)"
    )
    print(f"  Buckets: {pool_buckets}")
    print(f"  Pool models per bucket: {pool_n_models}")

    write_variants_definitions(out_dir)

    # Pre-slice the pool by task once. per_model_inputs filters its `df` arg by
    # task, so passing the whole pool made each call re-scan all ~N rows.
    df_by_task = {t: g for t, g in df_pool.groupby("task", sort=False)}

    rows = []
    for task in tqdm(tasks, desc="SNR tasks"):
        row = {"task": task}
        dft = df_by_task[task]
        size_inputs = {b: per_model_inputs(dft, task, b) for b in pool_buckets}
        for fd in AGGREGATION_FUNCTIONS:
            key = variant_key(fd)
            for b in pool_buckets:
                # Gate custom-size cells: random benchmarks carry no signal.
                if b in AR_SIZES and (task, b) not in above_random:
                    sig = noi = snr = np.nan
                else:
                    sig, noi, snr = variant_signal_noise_snr(size_inputs[b], fd["func"])
                row[f"signal_{key}_{b}"] = sig
                row[f"noise_{key}_{b}"] = noi
                row[f"snr_{key}_{b}"] = snr
        rows.append(row)

    snr_df = pd.DataFrame(rows).set_index("task").sort_index()

    # Copy the DA ground truth (rq01) and append the SNR columns. DA is computed
    # before SNR, so the table must already exist.
    da_path = DECISION_ACCURACY / out_dir.parent.name / out_dir.name / "da_per_task.csv"
    if not da_path.exists():
        raise SystemExit(
            f"DA table missing: {da_path}\n"
            f"Run `compute_da.py --pool {pool}` first (DA is computed before SNR)."
        )
    da_df = pd.read_csv(da_path, index_col="task")
    combined = da_df.join(snr_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "snr_variants_per_task.csv"
    combined.to_csv(csv_path)
    print(f"\nWrote CSV → {csv_path}")
    print(
        f"  {len(combined)} tasks × {len(combined.columns)} columns "
        f"({len(AGGREGATION_FUNCTIONS)} variants × {len(pool_buckets)} buckets × 3 stats "
        f"+ {len(da_df.columns)} DA cols from rq01)"
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pool",
        required=True,
        help="Pool name from configs/models.json. Model-set tiers: "
        "seeds_1904, seeds_28_1797, seeds_28_1797_1904, custom_swissai_hf.",
    )
    p.add_argument(
        "--out-subdir",
        default=None,
        help="Subdir under results/<stage>/snr_definition/ (default: <pool>).",
    )
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; " f"available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    out_dir = SNR_DEFINITION / stage / (args.out_subdir or args.pool)
    run(pool=args.pool, out_dir=out_dir)


if __name__ == "__main__":
    main()
