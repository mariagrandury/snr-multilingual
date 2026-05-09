"""Render the per-benchmark and per-language acc-vs-FLOPs grids for the
12 custom Apertus pretraining checkpoints.

The signal/noise/decision-accuracy analysis lives in the sibling script
``run_apertus_snr_variants.py`` (writes the variants CSV) +
``analyze_snr_variants.py`` (renders the SNR-vs-DA grids and heatmaps).
This file is the curve-viewer.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `snr` and `analysis` importable when this file is run directly.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import math
from collections import defaultdict

import matplotlib.pyplot as plt
from tqdm import tqdm

from multilingual.analyze_snr_variants import assign_language, benchmark_family
from snr.constants import PLOT_DIR
from snr.download.apertus import load_apertus_eval_results
from analysis.plotting.datadecide import plot_task_curves

SMALL_SIZES = ["175M", "350M", "600M"]
TARGET_SIZE = "1B"
PLOTTED_MIXES = ["fwEdu30", "fwEdu60", "fwEdu90"]
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
SEED = 1904
OUT_DIR = PLOT_DIR


def _draw_task(ax, df, task, task_idx, all_sizes):
    plot_task_curves(
        ax, task, signal_label=f"signal ({TARGET_SIZE})",
        plotted_sizes=all_sizes, plotted_mixes=PLOTTED_MIXES,
        metric="primary_score", df=df, colors=COLORS, SEED=SEED, task_idx=task_idx,
        n_mixes_label=f"{len(PLOTTED_MIXES)} data mixtures",
        xc_label="", signal_size=TARGET_SIZE,
    )


def _plot_grid(df, group_label, tasks, subplot_titles, out_path, all_sizes, ncols=3):
    """Render `tasks` as subplots of one figure, saved to `out_path`."""
    n = len(tasks)
    if n == 0:
        return False
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    drawn = 0
    for idx, (task, subtitle) in enumerate(zip(tasks, subplot_titles)):
        ax = axes[idx // ncols][idx % ncols]
        try:
            _draw_task(ax, df, task, idx, all_sizes)
            ax.set_title(f"{subtitle} — {task}", fontsize=10)
            drawn += 1
        except Exception:
            ax.set_visible(False)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)
    if drawn == 0:
        plt.close(fig)
        return False
    fig.suptitle(group_label, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _plot_grouped_curves(df, tasks, all_sizes, out_dir):
    """Emit two combined views: one figure per benchmark family (subplots = languages)
    and one figure per language (subplots = benchmark families)."""
    by_family = defaultdict(list)
    by_language = defaultdict(list)
    for task in tasks:
        by_family[benchmark_family(task)].append(task)
        by_language[assign_language(task)].append(task)

    per_bench_dir = out_dir / "per_benchmark"
    per_lang_dir = out_dir / "per_language"
    per_bench_dir.mkdir(parents=True, exist_ok=True)
    per_lang_dir.mkdir(parents=True, exist_ok=True)

    n_bench = 0
    for family, fam_tasks in tqdm(sorted(by_family.items()), desc="Per-benchmark grids"):
        fam_tasks_sorted = sorted(fam_tasks, key=assign_language)
        subtitles = [assign_language(t) for t in fam_tasks_sorted]
        if _plot_grid(df, f"benchmark: {family}", fam_tasks_sorted, subtitles,
                      per_bench_dir / f"{family}.png", all_sizes):
            n_bench += 1

    n_lang = 0
    for lang, lang_tasks in tqdm(sorted(by_language.items()), desc="Per-language grids"):
        lang_tasks_sorted = sorted(lang_tasks, key=benchmark_family)
        subtitles = [benchmark_family(t) for t in lang_tasks_sorted]
        if _plot_grid(df, f"language: {lang}", lang_tasks_sorted, subtitles,
                      per_lang_dir / f"{lang}.png", all_sizes):
            n_lang += 1

    return n_bench, n_lang


def run():
    df = load_apertus_eval_results()
    tasks = sorted(df["task"].unique())
    print(f"Loaded {len(df):,} rows | {df['model'].nunique()} models | {len(tasks)} tasks")

    all_sizes = SMALL_SIZES + [TARGET_SIZE]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_bench, n_lang = _plot_grouped_curves(df, tasks, all_sizes, OUT_DIR / "acc_vs_flops")

    print(f"\nWrote {n_bench} per-benchmark grids → {OUT_DIR / 'acc_vs_flops' / 'per_benchmark'}")
    print(f"Wrote {n_lang} per-language grids → {OUT_DIR / 'acc_vs_flops' / 'per_language'}")


if __name__ == "__main__":
    run()
