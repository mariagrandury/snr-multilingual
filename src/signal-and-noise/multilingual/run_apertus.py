"""Render the per-benchmark and per-language acc-vs-FLOPs grids for a
custom-pretraining seed pool.

The signal/noise/decision-accuracy analysis lives in the sibling script
``run_apertus_snr_variants.py`` (writes the variants CSV) +
``analyze_snr_variants.py`` (renders the SNR-vs-DA grids and heatmaps).
This file is the curve-viewer.

``--pool`` (a ``seeds_*`` pool from configs/models.json) selects which
custom pretrains to draw: per-mix acc-vs-FLOPs curves per size, one figure
per (benchmark family) and per (language). One seed per pool — multi-seed
overlays get unreadable; ``--seed`` overrides the default.

SNR sizes / mixes come from the ``snr`` section of configs/models.json.

Output: ``results/acc_vs_flops/<pool>/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `snr` and `analysis` importable when this file is run directly.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Shared loader (cluster + local). configs lives at
# <repo>/src/evals/scripts/utils/configs.py — repo root is 3 parents above
# this file (multilingual/run_apertus.py → signal-and-noise → src → repo).
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
from evals.scripts.utils.configs import (  # noqa: E402
    expand_pool,
    load_pools,
    load_snr_params,
)

import math  # noqa: E402
from collections import defaultdict  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from tqdm import tqdm  # noqa: E402

from multilingual.analyze_snr_variants import assign_language, benchmark_family  # noqa: E402
from snr.constants import PLOT_DIR  # noqa: E402
from snr.download.apertus import load_apertus_eval_results  # noqa: E402
from analysis.plotting.datadecide import plot_task_curves  # noqa: E402

# SNR analysis params — single source of truth in configs/models.json.
_SNR = load_snr_params()
SMALL_SIZES = _SNR["small_sizes"]
TARGET_SIZE = _SNR["target_size"]
PLOTTED_MIXES = _SNR["plotted_mixes"]
ALL_SIZES = SMALL_SIZES + [TARGET_SIZE]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
OUT_ROOT = PLOT_DIR / "acc_vs_flops"


def _seed_of(pool: str) -> int:
    """Single canonical seed for a snr-pretraining-custom pool.

    Multi-seed pools pick the LAST seed listed in the pool's `seeds` field —
    the user can also pass ``--seed`` to override. Convention: 1904 is the
    fully-trained seed and is listed last in the multi-seed pools, so
    last-wins gives the most-complete curves by default.
    """
    members = load_pools()[pool]["members"]
    seeds: list[int] = []
    for m in members:
        seeds.extend(m.get("seeds", []))
    if not seeds:
        raise ValueError(f"pool {pool!r} has no seeds — pick a seeds_* pool")
    return seeds[-1]


def _draw_task(ax, df, task, task_idx, seed):
    plot_task_curves(
        ax, task, signal_label=f"signal ({TARGET_SIZE})",
        plotted_sizes=ALL_SIZES, plotted_mixes=PLOTTED_MIXES,
        metric="primary_score", df=df, colors=COLORS, SEED=seed, task_idx=task_idx,
        n_mixes_label=f"{len(PLOTTED_MIXES)} data mixtures",
        xc_label="", signal_size=TARGET_SIZE,
    )


def _plot_grid(df, group_label, tasks, subplot_titles, out_path, seed, ncols=3):
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
            _draw_task(ax, df, task, idx, seed)
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


def _plot_grouped_curves(df, tasks, out_dir, seed):
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
                      per_bench_dir / f"{family}.png", seed):
            n_bench += 1

    n_lang = 0
    for lang, lang_tasks in tqdm(sorted(by_language.items()), desc="Per-language grids"):
        lang_tasks_sorted = sorted(lang_tasks, key=benchmark_family)
        subtitles = [benchmark_family(t) for t in lang_tasks_sorted]
        if _plot_grid(df, f"language: {lang}", lang_tasks_sorted, subtitles,
                      per_lang_dir / f"{lang}.png", seed):
            n_lang += 1

    return n_bench, n_lang


def run(pool: str, out_dir: Path, seed: int | None = None):
    if seed is None:
        seed = _seed_of(pool)
    pool_models = set(expand_pool(pool))
    df = load_apertus_eval_results()
    df = df[df["model"].isin(pool_models)]
    tasks = sorted(df["task"].unique())
    print(f"Pool '{pool}': {len(pool_models)} model(s), seed={seed}, "
          f"{len(df):,} rows, {len(tasks)} tasks")

    out_dir.mkdir(parents=True, exist_ok=True)
    n_bench, n_lang = _plot_grouped_curves(df, tasks, out_dir, seed)
    print(f"\nWrote {n_bench} per-benchmark grids → {out_dir / 'per_benchmark'}")
    print(f"Wrote {n_lang} per-language grids → {out_dir / 'per_language'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pool", default="seeds_1904",
        help="seeds_* pool name from configs/models.json (e.g. seeds_1904, "
             "seeds_28_1797, seeds_28_1797_1904). Default: seeds_1904.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override the seed whose curves to draw "
             "(default: last seed listed in the pool).",
    )
    p.add_argument(
        "--out-subdir", default=None,
        help="Subdir under results/acc_vs_flops/ (default: <pool>).",
    )
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; "
                f"available: {sorted(load_pools().keys())}")
    run(pool=args.pool, out_dir=OUT_ROOT / (args.out_subdir or args.pool),
        seed=args.seed)


if __name__ == "__main__":
    main()
