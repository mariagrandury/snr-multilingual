"""Run the SNR pipeline on the 12 custom Apertus pretraining models.

Reuses the reference signal-and-noise code (Heineman et al., 2025) for
compute + rendering, and adds Apertus-specific I/O and plotting orchestration
here. signal-and-noise is made importable via the same sys.path pattern as
``src/snr/metrics.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Make signal-and-noise importable.
_SNR_ROOT = "/iopsstor/scratch/cscs/mariagrandury/signal-and-noise"
if _SNR_ROOT not in sys.path:
    sys.path.insert(0, _SNR_ROOT)

from snr.download.apertus import load_apertus_eval_results  # noqa: E402
from snr.snr_simple import main  # noqa: E402
from snr.plot import plot_snr_da_grid  # noqa: E402
from analysis.plotting.datadecide import plot_task_curves  # noqa: E402

SMALL_SIZES = ["175M", "350M", "600M"]
TARGET_SIZE = "1B"
PLOTTED_MIXES = ["fwEdu30", "fwEdu60", "fwEdu90"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
SEED = 1904
OUT_DIR = Path(_SNR_ROOT) / "img" / "apertus_multilingual"


def _results_to_per_task_df(results, sizes_for_da, sizes_for_snr):
    rows = []
    for r in results:
        row = {"task": r["Task"]}
        for s in sizes_for_da:
            row[f"decision_acc_{s}"] = r["Decision Accuracy"].get(s, float("nan"))
        for s in sizes_for_snr:
            row[f"snr_{s}"] = r["SNR"].get(s, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows).set_index("task")


def _plot_curves(df, tasks, all_sizes, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_plotted = 0
    for i, task in enumerate(tqdm(tasks, desc="Plotting curves")):
        fig, ax = plt.subplots(figsize=(6, 4))
        try:
            plot_task_curves(
                ax, task, signal_label=f"signal ({TARGET_SIZE})",
                plotted_sizes=all_sizes, plotted_mixes=PLOTTED_MIXES,
                metric="primary_score", df=df, colors=COLORS, SEED=SEED, task_idx=i,
                n_mixes_label=f"{len(PLOTTED_MIXES)} data mixtures",
                xc_label="", signal_size=TARGET_SIZE,
            )
            ax.set_title(task)
            fig.tight_layout()
            fig.savefig(out_dir / f"{task}.png", dpi=120)
            n_plotted += 1
        except Exception:
            # Tasks with incomplete (size, mix) coverage: skip.
            pass
        finally:
            plt.close(fig)
    return n_plotted


def run():
    df = load_apertus_eval_results()
    tasks = sorted(df["task"].unique())
    print(f"Loaded {len(df):,} rows | {df['model'].nunique()} models | {len(tasks)} tasks")

    all_sizes = SMALL_SIZES + [TARGET_SIZE]
    results = main(
        df=df, tasks=tasks,
        small_sizes=SMALL_SIZES,
        large_sizes_scaling=[],
        large_sizes_snr=[TARGET_SIZE],
        target_size=TARGET_SIZE,
        target_step=None,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_task = _results_to_per_task_df(results, SMALL_SIZES, all_sizes)
    csv_path = OUT_DIR / "snr_per_task.csv"
    per_task.sort_index().to_csv(csv_path)

    n_plotted = _plot_curves(df, tasks, all_sizes, OUT_DIR / "curves")

    grid_path = OUT_DIR / "snr_vs_decision_accuracy.png"
    plot_snr_da_grid(per_task, SMALL_SIZES, TARGET_SIZE, grid_path)

    print(f"\nWrote table CSV → {csv_path}")
    print(f"Wrote {n_plotted}/{len(tasks)} per-task curve PNGs → {OUT_DIR / 'curves'}")
    print(f"Wrote SNR vs decision-accuracy scatter → {grid_path}")


if __name__ == "__main__":
    run()
