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
    pool_include_external,
    stage_external_models,
)

import math  # noqa: E402
from collections import defaultdict  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from tqdm import tqdm  # noqa: E402

from multilingual.analyze_snr_variants import (  # noqa: E402
    _ENGLISH_ONLY_TASKS, assign_language, benchmark_family,
)
from multilingual.smooth_subtasks import _is_language_aggregate  # noqa: E402
from multilingual.autodoc import (  # noqa: E402
    CANONICAL_POOL, fmt, md_table, replace_block)
from snr.constants import PLOT_DIR  # noqa: E402
from snr.download.apertus import (  # noqa: E402
    load_a06_eval_results,
    load_apertus_eval_results,
    load_distillation_eval_results,
    load_reference_hf_eval_results,
)
from analysis.plotting.datadecide import plot_task_curves  # noqa: E402

# SNR analysis params — single source of truth in configs/models.json.
_SNR = load_snr_params()
SMALL_SIZES = _SNR["small_sizes"]
TARGET_SIZE = _SNR["target_size"]
PLOTTED_MIXES = _SNR["plotted_mixes"]
ALL_SIZES = SMALL_SIZES + [TARGET_SIZE]

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]
OUT_ROOT = PLOT_DIR / "acc_vs_flops"

# Only render curve grids for the top-N benchmark families by Signal (relative
# dispersion of final scores across mixtures at the target size — the project's
# "Signal" metric). Every task's signal is written to acc_vs_flops_signal.csv.
TOP_N = 3


def _is_parent_task(task: str) -> bool:
    """Aggregate subject-subtasks into their parent (mmlu_anatomy → mmlu,
    agieval_en_* → agieval_en, …) by keeping only parent rows. Language
    variants (arc_de, global_mmlu_full_ar) stay distinct. Mirrors
    run_apertus_snr_variants._is_parent_task."""
    if task in _ENGLISH_ONLY_TASKS:
        return True
    return _is_language_aggregate(task, benchmark_family(task))


def _task_signal(df, task, seed):
    """Signal = (max-min)/mean of the per-mix final-ckpt scores at the target
    size for ``seed`` (relative dispersion across data mixtures). NaN if <2
    mixes have data."""
    sub = df[(df["size"] == TARGET_SIZE) & (df["task"] == task)
             & (df["mix"].isin(PLOTTED_MIXES))]
    if "seed" in sub.columns:
        sub = sub[sub["seed"] == seed]
    if sub.empty:
        return float("nan")
    finals = sub.loc[sub.groupby("mix")["step"].idxmax(), "primary_score"]
    if len(finals) < 2 or finals.mean() == 0:
        return float("nan")
    return float((finals.max() - finals.min()) / finals.mean())

# Overlay style for external scaling models (one marker per source). These
# are single-mix runs at their native size (incl. >1B), drawn as final-ckpt
# points on the same FLOPs axis to show scaling past the custom 1B ceiling.
_EXT_STYLE = {
    "a06":       ("#d62728", "^", "a06 (apertus3)"),
    "distill":   ("#9467bd", "s", "distillation"),
    "reference": ("#7f7f7f", "o", "swiss-ai/HF ref"),
}


def _load_externals(pool: str) -> pd.DataFrame | None:
    """Folded external pretraining-checkpoint models for the overlay, or None
    when the pool doesn't include externals."""
    if not pool_include_external(pool):
        return None
    allowed = stage_external_models(load_pools()[pool].get("stage", "pretraining"))
    frames = []
    for src, loader in (("reference", load_reference_hf_eval_results),
                        ("a06", load_a06_eval_results),
                        ("distill", load_distillation_eval_results)):
        try:
            d = loader()
        except FileNotFoundError:
            continue
        d = d[d["model"].isin(allowed)]
        if not d.empty:
            d = d.copy()
            d["osource"] = src
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def _overlay_externals(ax, df_ext, task):
    """Scatter each external model's final-ckpt (compute, score) for `task`."""
    if df_ext is None:
        return
    sub = df_ext[df_ext["task"] == task]
    if sub.empty:
        return
    finals = sub.loc[sub.groupby("model")["step"].idxmax()]
    for src, g in finals.groupby("osource"):
        color, marker, _ = _EXT_STYLE.get(src, ("#333", "x", src))
        ax.scatter(g["compute"], g["primary_score"], s=42, marker=marker,
                   color=color, edgecolor="black", linewidth=0.4, zorder=5)


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


def _draw_task(ax, df, df_ext, task, task_idx, seed):
    plot_task_curves(
        ax, task, signal_label=f"signal ({TARGET_SIZE})",
        plotted_sizes=ALL_SIZES, plotted_mixes=PLOTTED_MIXES,
        metric="primary_score", df=df, colors=COLORS, SEED=seed, task_idx=task_idx,
        n_mixes_label=f"{len(PLOTTED_MIXES)} data mixtures",
        xc_label="", signal_size=TARGET_SIZE,
    )
    _overlay_externals(ax, df_ext, task)


def _plot_grid(df, df_ext, group_label, tasks, subplot_titles, out_path, seed, ncols=3):
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
            _draw_task(ax, df, df_ext, task, idx, seed)
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
    if df_ext is not None:
        handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor=c,
                          markeredgecolor="black", markersize=8, label=lab)
                   for c, m, lab in _EXT_STYLE.values()]
        fig.legend(handles=handles, loc="upper right", fontsize=9,
                   title="external (final ckpt)")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


def _plot_grouped_curves(df, df_ext, tasks, out_dir, seed):
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
        if _plot_grid(df, df_ext, f"benchmark: {family}", fam_tasks_sorted, subtitles,
                      per_bench_dir / f"{family}.png", seed):
            n_bench += 1

    n_lang = 0
    for lang, lang_tasks in tqdm(sorted(by_language.items()), desc="Per-language grids"):
        lang_tasks_sorted = sorted(lang_tasks, key=benchmark_family)
        subtitles = [benchmark_family(t) for t in lang_tasks_sorted]
        if _plot_grid(df, df_ext, f"language: {lang}", lang_tasks_sorted, subtitles,
                      per_lang_dir / f"{lang}.png", seed):
            n_lang += 1

    return n_bench, n_lang


def run(pool: str, out_dir: Path, seed: int | None = None, top_n: int = TOP_N):
    if seed is None:
        seed = _seed_of(pool)
    pool_models = set(expand_pool(pool))
    df = load_apertus_eval_results()
    df = df[df["model"].isin(pool_models)]
    df_ext = _load_externals(pool)  # overlay (scaling past 1B), or None
    # Aggregate subject-subtasks into parents (point 4); languages stay distinct.
    tasks = [t for t in sorted(df["task"].unique()) if _is_parent_task(t)]
    n_ext = df_ext["model"].nunique() if df_ext is not None else 0
    print(f"Pool '{pool}': {len(pool_models)} custom model(s), seed={seed}, "
          f"{len(df):,} rows, {len(tasks)} parent tasks; "
          f"+{n_ext} external scaling model(s) overlaid")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the full per-task Signal table; rank families by mean Signal.
    rows = [{"task": t, "family": benchmark_family(t), "language": assign_language(t),
             "signal": _task_signal(df, t, seed)} for t in tasks]
    sig_df = pd.DataFrame(rows)
    sig_df.to_csv(out_dir / "acc_vs_flops_signal.csv", index=False)
    fam_rank = (sig_df.groupby("family")["signal"].mean()
                .sort_values(ascending=False))
    top_families = list(fam_rank.head(top_n).index)
    print(f"  Wrote signal CSV ({len(sig_df)} tasks). Top-{top_n} families by "
          f"Signal: {top_families}")

    # Only plot the top-N families (both views) to keep the PNG count small.
    plot_tasks = [t for t in tasks if benchmark_family(t) in top_families]
    n_bench, n_lang = _plot_grouped_curves(df, df_ext, plot_tasks, out_dir, seed)
    print(f"Wrote {n_bench} per-benchmark grids → {out_dir / 'per_benchmark'}")
    print(f"Wrote {n_lang} per-language grids → {out_dir / 'per_language'}")

    # Auto-refresh the acc_vs_flops README (canonical pool only — no-op else).
    generate_readme(pool, out_dir)


# --- auto-generated README (acc-vs-FLOPs Signal + above-random gate) --------

def generate_readme(pool: str, out_dir: Path) -> None:
    """Rewrite the auto blocks of results/acc_vs_flops/README.md (canonical pool
    only): top mixture-Signal families + the above-random gate breakdown. The
    gate numbers come from the `custom` above-random report (custom pretrains,
    the SNR gate's domain)."""
    if pool != CANONICAL_POOL:
        return
    stage = load_pools()[pool].get("stage", "pretraining")
    sig = pd.read_csv(out_dir / "acc_vs_flops_signal.csv")
    fam_rank = (sig.groupby("family")["signal"].mean()
                .sort_values(ascending=False))
    top3 = list(fam_rank.head(3).index)
    top_sig = sig.sort_values("signal", ascending=False).head(5)

    # above-random gate from the `custom` report (buckets 175M…1B)
    mask = pd.read_csv(PLOT_DIR / "acc_vs_flops" / stage / "custom"
                       / "above_random_mask.csv")
    buckets = [c for c in ("175M", "350M", "600M", "1B") if c in mask.columns]
    above_any = (mask[buckets] == 1).any(axis=1)
    above_1b = (mask["1B"] == 1) if "1B" in mask.columns else above_any
    n_total, n_above = len(mask), int(above_any.sum())

    highlight = "\n".join([
        f"- **The benchmarks that separate data mixtures most: "
        f"`{'`, `'.join(top3)}`** — top-3 families by mixture-Signal "
        f"((max−min)/mean of per-mix final scores) at {TARGET_SIZE}.",
        f"- **Mixture-Signal ≠ reliability.** These top-Signal families are exactly "
        f"the ones the above-random gate **removes** — they sit at chance, so they "
        f"never enter the SNR analysis. Of **{n_total} benchmarks, {n_above} clear "
        f"chance at ≥1 size** ({n_total - n_above} are random everywhere) — almost "
        f"entirely an answer-count effect.",
    ])

    sig_rows = [[f"`{r.task}`", r.family, r.language, fmt(r.signal, 3)]
                for r in top_sig.itertuples()]
    t_signal = md_table(["task", "family", "lang", "Signal"], sig_rows)

    gate_rows = []
    for n_opt, g in mask.assign(_any=above_any, _1b=above_1b).groupby("n_options"):
        gate_rows.append([int(n_opt), fmt(1.0 / int(n_opt), 2),
                          f"{int(g['_any'].sum())} / {len(g)}",
                          f"{int(g['_1b'].sum())} / {len(g)}"])
    t_gate = md_table(["options", "chance", "above ≥1 size", "above @1B"], gate_rows)

    results = "\n\n".join([
        f"Headline numbers from the `{pool}` pool (Signal) and the `custom` "
        f"above-random report. Regenerate: "
        f"`python multilingual/run_apertus.py --pool {pool}` and "
        f"`python multilingual/above_random.py`.",
        f"**Top benchmarks by mixture-Signal** (full ranking in "
        f"`pretraining/{pool}/acc_vs_flops_signal.csv`):",
        t_signal,
        f"![top-Signal family accuracy vs FLOPs](pretraining/{pool}/per_benchmark/{top3[0]}.png)",
        f"**Above-random gate** — a benchmark must beat chance (`1/n_options`) by "
        f"+0.05; `run_apertus_snr_variants.py` NaN-s every random `(benchmark, size)` "
        f"SNR cell, so the gate propagates to all RQs. Almost entirely an "
        f"answer-count effect:",
        t_gate,
    ])

    readme = PLOT_DIR / "acc_vs_flops" / "README.md"
    gen = f"run_apertus.py --pool {pool}"
    replace_block(readme, "highlight", "## Highlighted result\n\n" + highlight, gen)
    replace_block(readme, "results", "## Results\n\n" + results, gen)
    print(f"Wrote auto README blocks → {readme}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pool", default="3seeds",
        help="Pool name from configs/models.json. Tiers: 1seed, 2seeds, "
             "3seeds, 3seeds_swissai_hf (the last overlays external scaling "
             "models past 1B). Default: 3seeds.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Override the seed whose curves to draw "
             "(default: last seed listed in the pool).",
    )
    p.add_argument(
        "--out-subdir", default=None,
        help="Subdir under results/<stage>/acc_vs_flops/ (default: <pool>).",
    )
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; "
                f"available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    run(pool=args.pool,
        out_dir=PLOT_DIR / "acc_vs_flops" / stage / (args.out_subdir or args.pool),
        seed=args.seed)


if __name__ == "__main__":
    main()
