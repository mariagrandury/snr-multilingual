"""Render the four publication figures for the report as vector PDFs.

Each figure reuses the loaders / CSVs the ``rqNN_*`` analysis scripts already
emit (no parser is re-implemented here):

* fig1 — per-task accuracy-vs-FLOPs trajectories, via ``get_slice`` on the
  ``rq00_acc_vs_flops`` signal pool (``seeds_28_1797_1904``, seed 1904).
* fig2 — SNR (``rel_mpd`` @ 1B) vs checkpoint decision accuracy (f56 @ 1B),
  from ``rq02_snr_definition``'s ``snr_variants_per_task.csv``.
* fig3 — family x language reliability map of the same DA values + gate mask.
* fig4 — cumulative subject-subset SNR sweep, straight from
  ``rq04_smooth_subtasks``'s ``global_mmlu_full.csv``.

Run from anywhere::

    python analysis/report_figures/make_figures.py

Output: ``analysis/report_figures/figures/fig{1..4}_*.pdf``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``snr``, ``evals`` and ``analysis`` importable when run directly.
_SND = Path(__file__).resolve().parents[2]          # signal-and-noise
_SRC = Path(__file__).resolve().parents[3]           # src
for _p in (_SND, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

from analysis.paths import (  # noqa: E402
    ACC_VS_FLOPS, DECISION_ACCURACY, SMOOTH_SUBTASKS, SNR_DEFINITION)
from analysis.utils import assign_language, benchmark_family  # noqa: E402
from evals.scripts.utils.configs import expand_pool  # noqa: E402
from snr.dataloader import get_slice  # noqa: E402
from snr.download.apertus import load_apertus_eval_results  # noqa: E402

# --- shared style -----------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # embed editable text, not outlines
    "savefig.bbox": "tight",
})

FULL_WIDTH = (5.5, 2.6)     # two-panel / full-width figures
SINGLE_COL = (3.4, 2.8)     # single-column figures

# 12-colour colourblind-safe palette (Okabe-Ito + Paul Tol), keyed by the
# project's 12 report languages. Shared by fig2 (point colour) so the legend
# matches fig3's column order.
LANGS = ["en", "es", "ru", "hi", "zh", "ja", "ar", "vi", "tr", "th", "sw", "eu"]
_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
    "#F0E442", "#44AA99", "#882255", "#117733", "#999999", "#000000",
]
LANG_COLOR = dict(zip(LANGS, _PALETTE))

# Data-mixture display order + colours (match rq00's tab10 trio).
MIX_ORDER = ["fwEdu90", "fwEdu60", "fwEdu30"]
MIX_LABEL = {"fwEdu90": "90/10", "fwEdu60": "60/40", "fwEdu30": "30/70"}
MIX_COLOR = {"fwEdu90": "#1f77b4", "fwEdu60": "#ff7f0e", "fwEdu30": "#2ca02c"}
SIZES = ["175M", "350M", "600M", "1B"]

POOL_CURVES = "seeds_28_1797_1904"   # fig1 acc-vs-FLOPs pool
POOL_SNR = "custom_swissai_hf"       # fig2/3/4 SNR pool
SEED = 1904

FIG_DIR = Path(__file__).resolve().parent / "figures"


# ===========================================================================
# Figure 1 — accuracy vs FLOPs gate (multiblimp_rus | belebele)
# ===========================================================================
def fig1_gate() -> None:
    df = load_apertus_eval_results()
    df = df[df["model"].isin(set(expand_pool(POOL_CURVES)))]

    scores = pd.read_csv(ACC_VS_FLOPS / "pretraining" / POOL_CURVES
                         / "above_random_scores.csv").set_index("task")

    panels = [("multiblimp_rus", "multiblimp (rus)"),
              ("belebele_rus_Cyrl", "belebele (rus)")]

    # Pre-collect every smoothed segment so panels can share y-limits.
    seg = {}            # (task, mix, size) -> (x, y_smoothed)
    finals = {}         # task -> list of final smoothed scores (one per mix)
    for task, _ in panels:
        finals[task] = []
        for mix in MIX_ORDER:
            for size in SIZES:
                cd = get_slice(df, mix=mix, task=task, size=size, seed=SEED)
                cd = cd[cd["compute"] > 0].sort_values("compute")
                if cd.empty:
                    continue
                y = cd["primary_score"].rolling(3, center=True, min_periods=1).mean()
                seg[(task, mix, size)] = (cd["compute"].to_numpy(), y.to_numpy())
                if size == "1B":
                    finals[task].append(y.iloc[-1])

    all_y = np.concatenate([y for _, y in seg.values()])
    base_vals = [scores.loc[t, "random_baseline"] for t, _ in panels]
    ylo = min(all_y.min(), min(base_vals)) - 0.02
    yhi = all_y.max() + 0.03

    fig, axes = plt.subplots(1, 2, figsize=FULL_WIDTH, sharey=True)
    for ax, (task, title) in zip(axes, panels):
        annotated = set()        # one 175M/1B tag per panel
        for mix in MIX_ORDER:
            for size in SIZES:
                if (task, mix, size) not in seg:
                    continue
                x, y = seg[(task, mix, size)]
                ax.plot(x, y, color=MIX_COLOR[mix], lw=1.0, alpha=0.9)
                ax.scatter(x[-1], y[-1], marker="x", s=16,
                           color=MIX_COLOR[mix], zorder=4)
                if size in ("175M", "1B") and size not in annotated:
                    ax.annotate(size, (x[-1], y[-1]), textcoords="offset points",
                                xytext=(2, 3), fontsize=7, ha="left", va="bottom")
                    annotated.add(size)
        base = float(scores.loc[task, "random_baseline"])
        ax.axhline(base, ls="--", lw=0.8, color="grey", zorder=1)
        ax.set_xscale("log")
        ax.set_xlabel("Training FLOPs")
        ax.set_title(title)
        ax.set_ylim(ylo, yhi)
    axes[0].set_ylabel("Accuracy")

    handles = [Line2D([0], [0], color=MIX_COLOR[m], lw=1.4, label=MIX_LABEL[m])
               for m in MIX_ORDER]
    handles.append(Line2D([0], [0], ls="--", color="grey", lw=0.8,
                          label="random baseline"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
               frameon=False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_gate.pdf")
    plt.close(fig)

    # --- sanity ----------------------------------------------------------
    mb = float(np.mean(finals["multiblimp_rus"]))
    bb = float(np.mean(finals["belebele_rus_Cyrl"]))
    print(f"[fig1] multiblimp_rus final mean={mb:.3f} (baseline "
          f"{scores.loc['multiblimp_rus', 'random_baseline']}) | "
          f"belebele_rus_Cyrl final mean={bb:.3f} (baseline "
          f"{scores.loc['belebele_rus_Cyrl', 'random_baseline']})")
    assert mb > scores.loc["multiblimp_rus", "random_baseline"] + 0.10, \
        "multiblimp_rus should sit well above its baseline"
    assert abs(bb - 0.25) < 0.05, "belebele_rus_Cyrl should hug 0.25"


# ===========================================================================
# Figure 2 — SNR (rel_mpd @ 1B) vs decision accuracy (ckpt f56 @ 1B)
#   fig2  : custom_swissai_hf pool (controlled data mixtures)
#   fig2b : all/external pool (heterogeneous open-source models)
# ===========================================================================
def fig2_snr_vs_da(csv_path: Path, out_name: str, tag: str) -> None:
    snr_col, da_col = "snr_rel_mpd_1B", "decision_acc_ckpt_f56_1B"
    df = pd.read_csv(csv_path)
    sub = df.dropna(subset=[snr_col, da_col])
    sub = sub[sub[snr_col] > 0].copy()
    sub["lang"] = sub["task"].map(assign_language)

    x = sub[snr_col].to_numpy()
    y = sub[da_col].to_numpy()
    logx = np.log10(x)
    n = len(sub)

    r, _ = pearsonr(logx, y)
    se_r = float(np.sqrt((1 - r ** 2) / (n - 2)))

    # OLS fit in log-x space + 95% bootstrap band.
    slope, intercept = np.polyfit(logx, y, 1)
    grid = np.linspace(logx.min(), logx.max(), 100)
    rng = np.random.default_rng(0)
    boot = np.empty((2000, grid.size))
    idx = np.arange(n)
    for b in range(boot.shape[0]):
        s = rng.choice(idx, n, replace=True)
        m, c = np.polyfit(logx[s], y[s], 1)
        boot[b] = m * grid + c
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    fig, ax = plt.subplots(figsize=SINGLE_COL)
    gx = 10 ** grid
    ax.fill_between(gx, lo, hi, color="0.8", alpha=0.6, lw=0, zorder=1)
    ax.plot(gx, slope * grid + intercept, color="0.25", lw=1.2, zorder=2)
    for lang, g in sub.groupby("lang"):
        ax.scatter(g[snr_col], g[da_col], s=18, color=LANG_COLOR.get(lang, "0.5"),
                   edgecolor="white", linewidth=0.3, label=lang, zorder=3)

    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([2, 3, 5, 8]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("SNR (rel_mpd, 1B; log scale)")
    ax.set_ylabel("Decision accuracy (ckpt, 1B)")
    ax.text(0.03, 0.97, f"R = {r:.2f}, R² = {r ** 2:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name)
    plt.close(fig)

    print(f"[{tag}] variant=rel_mpd @ f56/1B  n={n}  "
          f"Pearson R={r:.4f} (SE {se_r:.4f})  R²={r ** 2:.4f}")


# ===========================================================================
# Figure 3 — family x language reliability map (DA ckpt f56 @ 1B)
# ===========================================================================
def _render_reliability_heatmap(da, gate, families, cbar_label, out_name) -> None:
    """Shared imshow renderer for the reliability maps (fig3 / fig3b). ``gate``
    is the gate-failed mask (grey ×) or None when there is no gate concept;
    NaN cells with no gate flag are left white (missing)."""
    nrows = len(families)
    fig, ax = plt.subplots(figsize=(5.5, 0.18 * nrows + 1.1))
    im = ax.imshow(np.clip(da, 0.5, 1.0), aspect="auto", cmap="viridis",
                   vmin=0.5, vmax=1.0)
    for i in range(nrows):
        for j in range(len(LANGS)):
            if gate is not None and gate[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           color="0.85", zorder=2))
                ax.text(j, i, "×", ha="center", va="center", fontsize=6,
                        color="0.45", zorder=3)
            elif not np.isnan(da[i, j]):
                v = (np.clip(da[i, j], 0.5, 1.0) - 0.5) / 0.5
                txt = "white" if v < 0.5 else "black"   # viridis: dark at low end
                ax.text(j, i, f"{da[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color=txt, zorder=3)

    ax.set_xticks(range(len(LANGS)))
    ax.set_xticklabels(LANGS)
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(families)
    ax.set_xticks(np.arange(-0.5, len(LANGS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)

    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(cbar_label)

    legend = [Patch(facecolor="white", edgecolor="0.6", label="no task")]
    if gate is not None:
        legend.insert(0, Patch(facecolor="0.85", label="gate-failed"))
    ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0, 1.01),
              ncol=2, frameon=False, handlelength=1.0, fontsize=7)

    fig.tight_layout()
    fig.savefig(FIG_DIR / out_name)
    plt.close(fig)


def _fig3_family(fam: str) -> str | None:
    """Collapse/drop benchmark families for the reliability map: merge the
    agieval / truthfulqa / piqa variants into one row each, keep only the
    ``_full`` global-MMLU, and drop the arabic_leaderboard probes."""
    if fam.startswith("arabic_leaderboard"):
        return None
    if fam == "global_mmlu":            # keep global_mmlu_full only
        return None
    if fam.startswith("agieval"):
        return "agieval"
    if fam.startswith("truthfulqa"):
        return "truthfulqa"
    if fam in ("piqa", "global_piqa_completions"):
        return "piqa"
    return fam


def fig3_reliability_map() -> None:
    snr_col, da_col = "snr_rel_mpd_1B", "decision_acc_ckpt_f56_1B"
    df = pd.read_csv(SNR_DEFINITION / "pretraining" / POOL_SNR
                     / "snr_variants_per_task.csv")
    df = df.copy()
    df["family"] = df["task"].map(benchmark_family).map(_fig3_family)
    df["lang"] = df["task"].map(assign_language)
    df = df[df["lang"].isin(LANGS) & df["family"].notna()]

    families = sorted(df["family"].unique())
    da = np.full((len(families), len(LANGS)), np.nan)   # colour value
    gate = np.zeros_like(da, dtype=bool)                # gate-failed cell?
    for i, fam in enumerate(families):
        for j, lang in enumerate(LANGS):
            cell = df[(df["family"] == fam) & (df["lang"] == lang)]
            if cell.empty:
                continue                                 # missing -> white
            passed = cell[cell[snr_col].notna()]
            if not passed.empty:
                da[i, j] = passed[da_col].mean()         # colour
            else:
                gate[i, j] = True                        # gate-failed -> grey x

    # Sort rows by mean DA (most reliable on top); all-NaN rows sink down.
    rowmean = np.array([np.nanmean(row) if not np.isnan(row).all() else -np.inf
                        for row in da])
    order = np.argsort(rowmean)[::-1]
    da, gate = da[order], gate[order]
    families = [families[k] for k in order]

    _render_reliability_heatmap(da, gate, families, "Decision accuracy (ckpt, 1B)",
                                "fig3_reliability_map.pdf")

    sw = LANGS.index("sw")
    print(f"[fig3] {len(families)} families x {len(LANGS)} langs; top-3 rows: "
          f"{families[:3]}")
    print(f"[fig3] sw column coloured cells: "
          f"{int(np.sum(~np.isnan(da[:, sw])))} (expect few; custom-only "
          f"families grey/white)")


# ===========================================================================
# Figure 3b — DA-based reliability map on the EXTERNAL suite (small -> large)
# ===========================================================================
# Small / large size buckets for the cross-scale decision (analysis doc): the
# 4x3 product intersected with the columns that actually exist is the 9 pairs.
_DA_SMALL = ["270M", "600M", "1B", "1.7B"]
_DA_LARGE = ["7-9B", "12-14B", "27-32B"]
_DA_MIN_SUPPORT = 6


def fig3b_reliability_map_external() -> None:
    df = pd.read_csv(DECISION_ACCURACY / "all" / "external" / "da_per_task.csv")
    pairs = [f"decision_acc_size_{s}_to_{l}" for s in _DA_SMALL for l in _DA_LARGE]
    pairs = [c for c in pairs if c in df.columns]

    df = df.copy()
    df["family"] = df["task"].map(benchmark_family).map(_fig3_family)
    df["lang"] = df["task"].map(assign_language)
    df = df[df["lang"].isin(LANGS) & df["family"].notna()]
    # Average the 9 size-pair columns within a task, then across tasks of the
    # same (family, language); support = total non-NaN pairs in the cell.
    df["task_da"] = df[pairs].mean(axis=1)
    df["support"] = df[pairs].notna().sum(axis=1)

    families = sorted(df["family"].unique())
    da = np.full((len(families), len(LANGS)), np.nan)
    for i, fam in enumerate(families):
        for j, lang in enumerate(LANGS):
            cell = df[(df["family"] == fam) & (df["lang"] == lang)]
            if cell["support"].sum() >= _DA_MIN_SUPPORT:
                da[i, j] = cell["task_da"].mean()

    # Sanity-check before saving (analysis doc validation values).
    def _cell(fam, lang):
        return da[families.index(fam), LANGS.index(lang)] if fam in families else np.nan
    expected = {("xstorycloze", "ru"): 0.99, ("belebele", "sw"): 0.57,
                ("hellaswag", "es"): 0.97}
    for (fam, lang), want in expected.items():
        got = _cell(fam, lang)
        assert abs(got - want) <= 0.02, \
            f"fig3b {fam}_{lang}={got:.3f} differs from expected {want} by >0.02"

    rowmean = np.array([np.nanmean(r) if not np.isnan(r).all() else -np.inf
                        for r in da])
    order = np.argsort(rowmean)[::-1]
    da = da[order]
    families = [families[k] for k in order]

    _render_reliability_heatmap(
        da, None, families,
        "Decision accuracy (small to large, external suite)",
        "fig3b_reliability_map.pdf")

    print(f"[fig3b] {len(families)} families x {len(LANGS)} langs "
          f"({len(pairs)} size pairs); top-3 rows: {families[:3]}")


# ===========================================================================
# Figure 4 — cumulative subject-subset SNR sweep (global_mmlu_full_subjects)
# ===========================================================================
def fig4_subset_sweep() -> None:
    sweep = pd.read_csv(SMOOTH_SUBTASKS / "pretraining" / POOL_SNR
                        / "global_mmlu_full.csv")
    summary = pd.read_csv(SMOOTH_SUBTASKS / "pretraining" / POOL_SNR / "summary.csv")
    summary = summary[summary["case"] == "case2_global_mmlu_full_subjects"]

    colors = dict(zip(SIZES, plt.get_cmap("tab10").colors[:4]))

    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    print("[fig4] per-size best_n / best_snr / full_set_snr / gain:")
    for size in SIZES:
        row = sweep[sweep["size"] == size].iloc[0]
        cum = np.array([float(v) for v in row["cumulative_snrs"].split("|")])
        full = float(row["full_set_snr"])
        best_n, best_snr = int(row["best_n"]), float(row["best_snr"])
        xs = np.arange(1, len(cum) + 1)

        ax.plot(xs, cum, color=colors[size], lw=1.2, label=size)
        ax.axhline(full, color=colors[size], ls="--", lw=0.8, alpha=0.5)
        ax.scatter([best_n], [best_snr], color=colors[size], s=24, zorder=5)

        # Cross-check against rq04's summary.csv row.
        srow = summary[summary["size"] == size].iloc[0]
        assert abs(best_n - srow["best_n"]) == 0 and \
            abs(best_snr - srow["best_snr"]) < 1e-6 and \
            abs(full - srow["full_set_snr"]) < 1e-6, \
            f"fig4 sweep disagrees with summary.csv for {size}"
        print(f"  {size:>4}: best_n={best_n:2d}  best_snr={best_snr:.4f}  "
              f"full_set={full:.4f}  gain={best_snr - full:+.4f}")

    ax.set_xlabel("Number of subjects included")
    ax.set_ylabel("Combined SNR")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_subset_sweep.pdf")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_gate()
    fig2_snr_vs_da(SNR_DEFINITION / "pretraining" / POOL_SNR
                   / "snr_variants_per_task.csv", "fig2_snr_vs_da.pdf", "fig2")
    fig2_snr_vs_da(SNR_DEFINITION / "all" / "external"
                   / "snr_variants_per_task.csv", "fig2b_snr_vs_da.pdf", "fig2b")
    fig3_reliability_map()
    fig3b_reliability_map_external()
    fig4_subset_sweep()
    print(f"Wrote 6 figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
