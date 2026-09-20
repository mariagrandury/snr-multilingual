"""The per-task SNR table: 22 signal-to-noise definitions x size bucket, joined
to rq02's decision accuracy. `analysis/<stage>/<pool>/snr_variants_per_task.csv`
is the single source of truth rq04, rq07 and rq09 read.

For every parent task of the pool and every size bucket, `per_model_inputs`
builds the four per-model arrays the aggregators in snr/snr_variants.py take
(one model = one training run of the bucket, seeds included when the pool has
them) and each aggregator returns (signal, noise, snr), stored as
`signal_<variant>_<bucket>`, `noise_<variant>_<bucket>`, `snr_<variant>_<bucket>`.

  signal   the spread of the runs' final scores across the bucket
           (`data_scores`); `rel_std` divides by the cross-run std of the
           finals (`data_noise`, CLAUDE.md bug #4), the others by their own
           dispersion
  noise    the mean over runs of the checkpoint-to-checkpoint std inside the
           noise window (`step_noise`), relative to the window mean
           (`data_scores_last_n`)

The noise window is analysis/RULES.md rule 4: `utils.noise_checkpoints`, the
shared tenths in the last NOISE_WINDOW (20 %) of each run, 80 / 90 / 100 %,
the same rows for BPB (also scored on the twentieths, which are left out) and
for benchmarks. Before this the window was "the last five checkpoints of
whatever grid the task has", 80-100 % for BPB and 60-100 % for benchmarks.

Two families of cells are NaN and stay NaN downstream: (task, bucket) cells
the above-random gate puts at chance (rq00's committed mask, `load_mask`; the pool's own
runs; cells with no chance level, BPB and the loss, pass), and the discrepancy
family (`DISCREPANCY_UNIT_INTERVAL`: `discrepancy`, `star_discrepancy`,
`star_discrepancy_shifted`, `rel_star_discrepancy`) on every task that is not a
benchmark, because those aggregators read the scores as points of [0, 1] and
return finite but meaningless values on BPB and the loss.
`snr_variant_coverage.csv` counts the finite cells per (variant, bucket).

Decision accuracy is not computed here: rq02 `compute_da.py` writes
`da_per_task.csv` first and this script joins its columns in front of the SNR
ones. The pool's models come from `analysis.utils.build_snr_pool` (parent
tasks and trained languages only, 90M dropped); there is no seed or external
model option.

`snr_variants_definitions.csv` carries each aggregator's title and formula.
The `signal_xlabel` / `noise_xlabel` strings of AGGREGATION_FUNCTIONS are
swapped upstream ("Step-to-Step Rel. Std" is the noise, "Data Recipe ..." the
signal); snr/ is never edited, so they are un-swapped when written and no
figure here labels an axis with them.

    python analysis/rq03_noise_and_snr/run_apertus_snr_variants.py --pool predictivity
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
    bucket_order,
    load_pools,
    pool_include_external,
    size_bucket,
)
from analysis.utils import (  # noqa: E402
    NOISE_WINDOW, _is_parent_task, benchmark_family, ladder_frame,
    noise_checkpoints, pool_models,
)
from analysis.autodoc import CANONICAL_POOL  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask, scores_and_mask  # noqa: E402
from analysis.paths import NOISE_AND_SNR, DECISION_ACCURACY  # noqa: E402
from snr.snr_variants import AGGREGATION_FUNCTIONS  # noqa: E402

# The signal/noise size axis is the *bucket* (size_bucket()), so nearby large
# sizes (7B/8B → "7-9B") pool to ≥2 models; the ladder sizes are singletons.
OUT_ROOT = NOISE_AND_SNR
# Aggregators that read the scores as points of [0, 1]: finite but meaningless
# on BPB and the loss, so NaN on every task that is not a benchmark.
DISCREPANCY_UNIT_INTERVAL = {"discrepancy", "star_discrepancy", "star_discrepancy_shifted", "rel_star_discrepancy"}


# --- per-model arrays for snr_variants --------------------------------------


def per_model_inputs(df, task, size):
    """Build the four per-model arrays expected by snr_variants aggregators.

    ``size`` is a *bucket* label and rows are selected on the ``bucket``
    column, so nearby large sizes (e.g. 7B + 8B → "7-9B") pool together.
    Each unique value of the ``model`` column is a separate training run.
    ``df`` carries a ``bucket`` column; ``frac`` (each checkpoint's share of
    its run, as `utils.ladder_frame` defines it) is added when missing.

    The noise window is the one of RULES.md rule 4, `utils.noise_checkpoints`:
    the shared tenths in the last NOISE_WINDOW (20 %) of the run, 80 / 90 /
    100 %, the same rows for BPB and benchmarks.
      step_noise         = per-model std (ddof 0) over the window checkpoints
      data_scores        = per-model final-checkpoint score
      data_noise         = cross-model std of `data_scores`, broadcast as
                           a constant array of the same length (bug #4)
      data_scores_last_n = per-model mean over the window checkpoints

    Models with fewer than 2 window checkpoints are dropped (they can't
    contribute step_noise — e.g. a single-revision external model). We
    require ≥ 2 surviving models overall.
    """
    sub = df[(df["bucket"] == size) & (df["task"] == task)]
    if sub.empty:
        return None
    if "frac" not in sub:
        sub = sub.assign(frac=sub["step"] / sub.groupby("model")["step"].transform("max"))
    sub = sub.sort_values("step")
    window = noise_checkpoints(sub).groupby("model")["primary_score"].apply(np.asarray)
    window = window[window.map(len) >= 2]
    if len(window) < 2:
        return None
    step_noise = np.array([np.std(a) for a in window])
    data_scores = sub.groupby("model")["primary_score"].last().loc[window.index].to_numpy(dtype=float)
    data_scores_last_n = np.array([a.mean() for a in window])
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
    """One row per aggregator describing what its signal/noise/snr mean.
    Upstream's `signal_xlabel` / `noise_xlabel` are swapped (module
    docstring); they are written under the right name here."""
    rows = []
    for fd in AGGREGATION_FUNCTIONS:
        rows.append(
            {
                "variant": variant_key(fd),
                "title": fd["title"],
                "latex": fd["latex"],
                "signal_label": fd["noise_xlabel"],
                "noise_label": fd["signal_xlabel"],
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
    # rq02 (compute_da.py); this step reads that table and appends the SNR
    # variant columns — DA first (the truth), SNR second (the proxies).
    df_pool = ladder_frame(pool)                 # the pool plus `frac`, the noise window's axis
    df_pool["bucket"] = df_pool["size"].map(size_bucket)
    all_tasks = sorted(df_pool["task"].unique())
    tasks = [t for t in all_tasks if _is_parent_task(t)]

    pool_buckets = [b for b in bucket_order() if b in set(df_pool["bucket"].dropna())]

    own_models = pool_models(pool, df_pool)
    all_models = set(df_pool["model"])
    is_external = ~df_pool["model"].isin(own_models)

    # Above-random gate (raw-metric competence check): a (task, bucket) SNR
    # cell whose mean score sits at chance (within the margin) carries no
    # usable signal, so its signal/noise/snr are NaN'd here — the gate
    # propagates to every analysis that reads this CSV. Cells with no chance
    # level (per-language BPB, generative tasks) or no scores are left alone.
    # rule 1: the one canonical mask (rq00, computed on every task with untrained=True),
    # not a mask recomputed on this pool's trained-only frame, which could differ per cell
    _ar_mask = load_mask(pool) if load_mask(pool) is not None else load_mask(CANONICAL_POOL)
    if _ar_mask is None:
        _, _ar_mask, _ = scores_and_mask(df_pool[df_pool["model"].isin(own_models)])
        print("  (no committed above-random mask: gating with one recomputed on this pool)")
    at_chance = {(t, s) for s in _ar_mask.columns
                 for t in _ar_mask.index[(_ar_mask[s] == 0).fillna(False).to_numpy(bool)]}
    pool_n_models = df_pool.groupby("bucket")["model"].nunique().to_dict()
    print(
        f"Pool '{pool}': {len(all_models & own_models)} custom + "
        f"{len(all_models - own_models)} external model(s); "
        f"include_external={pool_include_external(pool)}"
    )
    print(
        f"Loaded {len(df_pool):,} rows ({int(is_external.sum()):,} external) | "
        f"{len(tasks)} parent tasks (filtered from {len(all_tasks)} total)"
    )
    print(f"  Buckets: {pool_buckets}")
    print(f"  Pool models per bucket: {pool_n_models}")
    print(f"  Noise window: the shared tenths in the last {NOISE_WINDOW:.0%} of each run (rule 4), "
          f"every kind of measurement; {sorted(DISCREPANCY_UNIT_INTERVAL)} are NaN outside benchmarks")

    write_variants_definitions(out_dir)

    # Pre-slice the pool by task once. per_model_inputs filters its `df` arg by
    # task, so passing the whole pool made each call re-scan all ~N rows.
    df_by_task = {t: g for t, g in df_pool.groupby("task", sort=False)}

    rows = []
    for task in tqdm(tasks, desc="SNR tasks"):
        row = {"task": task}
        dft = df_by_task[task]
        size_inputs = {b: per_model_inputs(dft, task, b) for b in pool_buckets}
        is_benchmark = benchmark_family(task) not in ("bpb", "loss")
        for fd in AGGREGATION_FUNCTIONS:
            key = variant_key(fd)
            for b in pool_buckets:
                # Gate at-chance cells: random benchmarks carry no signal.
                # The [0, 1] discrepancy family is undefined off the benchmarks.
                if (task, b) in at_chance or (key in DISCREPANCY_UNIT_INTERVAL and not is_benchmark):
                    sig = noi = snr = np.nan
                else:
                    sig, noi, snr = variant_signal_noise_snr(size_inputs[b], fd["func"])
                row[f"signal_{key}_{b}"] = sig
                row[f"noise_{key}_{b}"] = noi
                row[f"snr_{key}_{b}"] = snr
        rows.append(row)

    snr_df = pd.DataFrame(rows).set_index("task").sort_index()

    # Coverage per (variant, bucket): the discrepancy family is NaN on BPB and
    # the loss (above), so its cells are empty where every other variant has a
    # value. Ranking variants against DA on different task populations is not
    # a like-for-like comparison; the table makes the gap visible and the
    # postprocess reads it.
    cov = pd.DataFrame({b: {variant_key(fd): int(snr_df[f"snr_{variant_key(fd)}_{b}"].notna().sum())
                            for fd in AGGREGATION_FUNCTIONS} for b in pool_buckets})
    cov.index.name = "variant"
    out_dir.mkdir(parents=True, exist_ok=True)
    cov.to_csv(out_dir / "snr_variant_coverage.csv")
    short = cov.lt(0.9 * cov.max(axis=0), axis=1)
    for v in cov.index[short.any(axis=1)]:
        print(f"  coverage: `{v}` is finite on {cov.loc[v].to_dict()} cells vs "
              f"{cov.max(axis=0).to_dict()} for the best-covered variant")

    # Copy the DA ground truth (rq02) and append the SNR columns. DA is computed
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
        f"+ {len(da_df.columns)} DA cols from rq02)"
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
        help="Subdir under analysis/rq03_noise_and_snr/<stage>/ (default: <pool>).",
    )
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; " f"available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    out_dir = NOISE_AND_SNR / stage / (args.out_subdir or args.pool)
    run(pool=args.pool, out_dir=out_dir)


if __name__ == "__main__":
    main()
