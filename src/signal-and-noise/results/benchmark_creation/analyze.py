"""Per-family SNR vs benchmark-creation metadata analysis.

Reads:
  - ../snr_definition/snr_variants_per_task.csv  (Q1 output)
  - data_info.md                                 (this dir's metadata table)

Writes (this dir):
  - per_family_snr.csv             one row per family with SNR aggregates
                                   + curation/source category labels
  - per_task_snr.csv               one row per per-language aggregate task,
                                   carrying the per-task curation override
                                   (xnli_eu, paws_eu, xcopa_eu)
  - snr_by_curation_process.png    strip plot of family SNR by curation cat
  - snr_by_data_source.png         strip plot of family SNR by source-origin
  - snr_by_curation_per_task.png   per-task strip plot (catches the xnli_eu
                                   heterogeneity that family-level smears)
  - group_stats.csv                per-group n, mean, median, kruskal H, p

Q1's headline pick is `mpd` (mean pairwise distance, dispersion cluster);
SNR signal here is `snr_mpd_1B`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Reach the multilingual helpers without writing outside this dir.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from multilingual.analyze_snr_variants import assign_language, benchmark_family  # noqa: E402
from multilingual.smooth_subtasks import _is_language_aggregate  # noqa: E402

HERE = Path(__file__).resolve().parent
SNR_CSV = ROOT / "results" / "snr_definition" / "snr_variants_per_task.csv"
SNR_COL = "snr_mpd_1B"

# --- Categorical labels for grouping -----------------------------------------
# Keep these in sync with the per-family paragraphs in data_info.md. Two
# views:
#   curation_category — how items in the per-language eval set were produced.
#   source_origin     — whether the source benchmark was English-only and
#                       translated, or originally multilingual / aggregated.
# Task-format axes added in Q4-extension. `format`:
#   mcq_question_only — question + N letter-labeled options, no passage.
#   mrc_passage       — a passage to read + question + N options.
#   completion        — context + N candidate continuations, scored by LL.
#   minimal_pair      — two minimally-differing sentences, pick higher-LL.
#   classification    — premise + hypothesis (or sentence pair) -> N labels.
# `passage` is True iff the prompt contains a substantial passage / long
# context (a heuristic, formalised quantitatively in Phase B).
FAMILY_META: dict[str, dict] = {
    "arc": {
        "data_source": "ARC (Clark et al. 2018), Okapi-translated",
        "curation_process": "machine translation by ChatGPT",
        "curation_category": "machine_translation",
        "source_origin": "english_translated",
        "format": "mcq_question_only", "n_options": 4, "passage": False,
    },
    "belebele": {
        "data_source": "FLORES-200 passages, custom MRC questions",
        "curation_process": "human translation by bilingual experts",
        "curation_category": "human_translation",
        "source_origin": "english_translated",
        "format": "mrc_passage", "n_options": 4, "passage": True,
    },
    "global_mmlu": {
        "data_source": "MMLU (Hendrycks et al. 2021), Cohere Lite-style",
        "curation_process": "professional human translation + post-editing",
        "curation_category": "human_translation",
        "source_origin": "english_translated",
        "format": "mcq_question_only", "n_options": 4, "passage": False,
    },
    "global_mmlu_full": {
        "data_source": "MMLU (Hendrycks et al. 2021), Cohere Full",
        "curation_process": "machine translation + crowd / expert post-editing",
        "curation_category": "mt_post_edited",
        "source_origin": "english_translated",
        "format": "mcq_question_only", "n_options": 4, "passage": False,
    },
    "global_piqa_completions": {
        "data_source": "originally-multilingual native authoring (Arnett 2025)",
        "curation_process": "participatory native-speaker authoring (no translation)",
        "curation_category": "originally_multilingual",
        "source_origin": "originally_multilingual",
        "format": "completion", "n_options": 2, "passage": False,
    },
    "hellaswag": {
        "data_source": "HellaSwag (Zellers et al. 2019), Okapi-translated",
        "curation_process": "machine translation by ChatGPT",
        "curation_category": "machine_translation",
        "source_origin": "english_translated",
        "format": "completion", "n_options": 4, "passage": True,
    },
    "multiblimp": {
        "data_source": "Universal Dependencies + UniMorph (Jumelet 2025)",
        "curation_process": "template-based automatic generation from UD/UniMorph",
        "curation_category": "template_generated",
        "source_origin": "originally_multilingual",
        "format": "minimal_pair", "n_options": 2, "passage": False,
    },
    "paws": {
        "data_source": "PAWS (Zhang 2019); PAWS-X + HiTZ/PAWS-eu",
        "curation_process": "professional human translation (mixed sources for eu)",
        "curation_category": "human_translation",
        "source_origin": "english_translated",
        "format": "classification", "n_options": 2, "passage": False,
    },
    "xcopa": {
        "data_source": "COPA (Roemmele 2011); XCOPA + HiTZ/XCOPA-eu",
        "curation_process": "professional human translation + native re-annotation",
        "curation_category": "human_translation",
        "source_origin": "english_translated",
        "format": "completion", "n_options": 2, "passage": False,
    },
    "xnli": {
        "data_source": "MultiNLI (Williams 2018); XNLI + XNLIeu",
        "curation_process": "professional human translation (mt+post-edit for eu)",
        "curation_category": "human_translation",
        "source_origin": "english_translated",
        "format": "classification", "n_options": 3, "passage": False,
    },
    "xstorycloze": {
        "data_source": "Story Cloze Test (Mostafazadeh 2016), XStoryCloze",
        "curation_process": "professional human translation",
        "curation_category": "human_translation",
        "source_origin": "english_translated",
        "format": "completion", "n_options": 2, "passage": True,
    },
    "xwinograd": {
        "data_source": "aggregated native Winograd schemas",
        "curation_process": "originally-multilingual aggregation of native schemas",
        "curation_category": "originally_multilingual",
        "source_origin": "originally_multilingual",
        "format": "completion", "n_options": 2, "passage": False,
    },
}
# Derived: random baseline = 1 / n_options
for _f, _meta in FAMILY_META.items():
    _meta["random_baseline"] = round(1.0 / _meta["n_options"], 3)

# Per-task overrides: (family, lang) → curation_category. Used when an `_eu`
# subset comes from a different paper with a different curation method than
# the rest of the family.
PER_TASK_OVERRIDES: dict[tuple[str, str], str] = {
    ("xnli", "eu"): "mt_post_edited",   # XNLIeu (Heredia et al. 2024)
    # paws_eu and xcopa_eu also come from separate papers, but their curation
    # method (professional human translation) matches the family default.
}

CATEGORY_ORDER = [
    "originally_multilingual",
    "human_translation",
    "template_generated",
    "mt_post_edited",
    "machine_translation",
]
ORIGIN_ORDER = ["originally_multilingual", "english_translated"]


def load_per_task_snr() -> pd.DataFrame:
    df = pd.read_csv(SNR_CSV, usecols=["task", SNR_COL])
    df["family"] = df["task"].map(benchmark_family)
    df["language"] = df["task"].map(assign_language)
    keep = [
        _is_language_aggregate(t, f) and f in FAMILY_META
        for t, f in zip(df["task"], df["family"])
    ]
    df = df[keep].copy()
    df = df.dropna(subset=[SNR_COL])
    return df


def per_family_aggregate(per_task: pd.DataFrame) -> pd.DataFrame:
    g = per_task.groupby("family")[SNR_COL]
    out = pd.DataFrame({
        "n_tasks": g.size(),
        "snr_median": g.median(),
        "snr_mean": g.mean(),
        "snr_max": g.max(),
    }).reset_index()
    meta = pd.DataFrame.from_dict(FAMILY_META, orient="index").reset_index().rename(
        columns={"index": "family"}
    )
    merged = out.merge(meta, on="family", how="left")
    # Phase B: optional length features written by length_features.py.
    length_csv = HERE / "length_features.csv"
    if length_csv.exists():
        lf = pd.read_csv(length_csv)
        merged = merged.merge(lf, on="family", how="left")
    return merged.sort_values("snr_median", ascending=False)


def per_task_with_overrides(per_task: pd.DataFrame) -> pd.DataFrame:
    out = per_task.copy()
    out["curation_category"] = [
        FAMILY_META[f]["curation_category"] for f in out["family"]
    ]
    out["source_origin"] = [
        FAMILY_META[f]["source_origin"] for f in out["family"]
    ]
    for (fam, lang), cat in PER_TASK_OVERRIDES.items():
        mask = (out["family"] == fam) & (out["language"] == lang)
        out.loc[mask, "curation_category"] = cat
    return out.sort_values(["family", "language"]).reset_index(drop=True)


def kruskal_wallis(values_by_group: dict[str, np.ndarray]) -> tuple[float, float, int]:
    """Return (H, p, n_groups). Skip groups with n<2."""
    groups = [v for v in values_by_group.values() if len(v) >= 2]
    if len(groups) < 2:
        return (float("nan"), float("nan"), len(groups))
    H, p = stats.kruskal(*groups)
    return (float(H), float(p), len(groups))


def _strip_plot(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    order: list[str],
    label_col: str | None,
    title: str,
    out_path: Path,
) -> tuple[float, float, int]:
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(order) + 1.5))
    rng = np.random.default_rng(0)
    values_by_group: dict[str, np.ndarray] = {}
    for i, cat in enumerate(order):
        sub = df[df[group_col] == cat]
        vals = sub[value_col].to_numpy()
        values_by_group[cat] = vals
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(vals, np.full_like(vals, i, dtype=float) + jitter,
                   s=70, alpha=0.85, edgecolor="black", linewidth=0.6)
        if label_col is not None:
            for v, j, lbl in zip(vals, jitter, sub[label_col]):
                ax.annotate(lbl, (v, i + j), fontsize=7,
                            xytext=(4, 0), textcoords="offset points",
                            va="center")
        med = float(np.median(vals))
        ax.plot([med, med], [i - 0.32, i + 0.32], color="red", lw=1.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{c}\n(n={len(values_by_group[c])})" for c in order])
    ax.set_xscale("log")
    ax.set_xlabel(f"{value_col} (log scale)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    H, p, n_groups = kruskal_wallis(values_by_group)
    print(f"  Kruskal-Wallis: H = {H:.3f}, p = {p:.4f} ({n_groups} groups with n≥2)")
    return H, p, n_groups


_CATEGORY_COLORS = {
    "originally_multilingual": "#1f77b4",
    "human_translation": "#ff7f0e",
    "template_generated": "#2ca02c",
    "mt_post_edited": "#d62728",
    "machine_translation": "#9467bd",
}


def _scatter_baseline(per_family: pd.DataFrame, out_path: Path) -> None:
    """SNR vs random baseline (1/n_options) on log-y. Tests whether
    fewer-option tasks (higher baseline) systematically yield higher SNR."""
    fig, ax = plt.subplots(figsize=(8, 6))
    x = per_family["random_baseline"].to_numpy()
    y = per_family["snr_median"].to_numpy()
    colors = [_CATEGORY_COLORS[c] for c in per_family["curation_category"]]
    ax.scatter(x, y, c=colors, s=110, edgecolor="black", linewidth=0.7)
    for xi, yi, lbl in zip(x, y, per_family["family"]):
        ax.annotate(lbl, (xi, yi), fontsize=8,
                    xytext=(5, 3), textcoords="offset points")
    # Spearman + log-linear OLS.
    rho, p_rho = stats.spearmanr(x, y)
    log_y = np.log10(y)
    slope, intercept, r, p_lin, _ = stats.linregress(x, log_y)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, 10 ** (intercept + slope * xx), "k--", lw=1, alpha=0.6,
            label=f"OLS log10(SNR) = {intercept:.2f} + {slope:.2f}·baseline\nPearson r = {r:.2f}, p = {p_lin:.3f}")
    ax.set_yscale("log")
    ax.set_xlabel("random baseline (1 / n_options)")
    ax.set_ylabel("median snr_mpd_1B (log scale)")
    ax.set_title(f"SNR vs random baseline. Spearman ρ = {rho:.2f} (p={p_rho:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Spearman rho = {rho:.3f}, p = {p_rho:.4f}")


def _scatter_length(
    per_family: pd.DataFrame,
    x_col: str,
    out_path: Path,
    title: str,
) -> tuple[float, float]:
    """Scatter SNR (log) vs a continuous length feature (log).
    Returns (Spearman rho, p)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    df = per_family.dropna(subset=[x_col])
    x = df[x_col].to_numpy()
    y = df["snr_median"].to_numpy()
    colors = [_CATEGORY_COLORS[c] for c in df["curation_category"]]
    ax.scatter(x, y, c=colors, s=110, edgecolor="black", linewidth=0.7)
    for xi, yi, lbl in zip(x, y, df["family"]):
        ax.annotate(lbl, (xi, yi), fontsize=8,
                    xytext=(5, 3), textcoords="offset points")
    rho, p_rho = stats.spearmanr(x, y)
    log_x = np.log10(x)
    log_y = np.log10(y)
    slope, intercept, r, p_lin, _ = stats.linregress(log_x, log_y)
    xx = np.geomspace(x.min(), x.max(), 50)
    ax.plot(xx, 10 ** (intercept + slope * np.log10(xx)),
            "k--", lw=1, alpha=0.6,
            label=f"log-log OLS: slope={slope:.2f}\nPearson r = {r:.2f}, p = {p_lin:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{x_col} (log scale)")
    ax.set_ylabel("median snr_mpd_1B (log scale)")
    ax.set_title(f"{title}.  Spearman ρ = {rho:.2f} (p={p_rho:.3f})")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  {x_col}: ρ = {rho:.3f}, p = {p_rho:.4f}")
    return float(rho), float(p_rho)


def _length_grid(per_family: pd.DataFrame, out_path: Path) -> None:
    """Three-panel side-by-side: SNR vs context_len, option_len, ratio."""
    cols = [
        ("context_len_chars_median", "context length (chars)"),
        ("option_len_chars_median",  "option length (chars, avg over options)"),
        ("context_to_option_ratio",  "context : option ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (col, label) in zip(axes, cols):
        df = per_family.dropna(subset=[col])
        x = df[col].to_numpy()
        y = df["snr_median"].to_numpy()
        colors = [_CATEGORY_COLORS[c] for c in df["curation_category"]]
        ax.scatter(x, y, c=colors, s=80, edgecolor="black", linewidth=0.6)
        for xi, yi, lbl in zip(x, y, df["family"]):
            ax.annotate(lbl, (xi, yi), fontsize=7,
                        xytext=(4, 2), textcoords="offset points")
        rho, p = stats.spearmanr(x, y)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(label)
        ax.set_title(f"{col}\nρ = {rho:.2f} (p={p:.3f})")
        ax.grid(True, alpha=0.3, which="both")
    axes[0].set_ylabel("median snr_mpd_1B (log scale)")
    fig.suptitle("Phase B: SNR vs length features (color = curation category)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _ranked_bar(per_family: pd.DataFrame, out_path: Path) -> None:
    df = per_family.sort_values("snr_median", ascending=True)
    colors = [_CATEGORY_COLORS[c] for c in df["curation_category"]]
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(df) + 1.5))
    y = np.arange(len(df))
    ax.barh(y, df["snr_median"], color=colors, edgecolor="black", linewidth=0.6)
    for yi, (med, n) in enumerate(zip(df["snr_median"], df["n_tasks"])):
        ax.text(med, yi, f"  median={med:.2f}  (n={n})",
                va="center", ha="left", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["family"])
    ax.set_xscale("log")
    ax.set_xlabel("median snr_mpd_1B across the family's per-language tasks (log scale)")
    ax.set_title("Per-family SNR ranking (color = curation_category)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=col, ec="black", lw=0.6, label=cat)
        for cat, col in _CATEGORY_COLORS.items()
        if cat in set(df["curation_category"])
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    print(f"Loading {SNR_CSV.relative_to(ROOT)}")
    per_task = load_per_task_snr()
    print(f"  → {len(per_task)} per-language aggregate tasks across "
          f"{per_task['family'].nunique()} families")

    per_family = per_family_aggregate(per_task)
    out_csv = HERE / "per_family_snr.csv"
    per_family.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv.name}")

    per_task_overrides = per_task_with_overrides(per_task)
    per_task_csv = HERE / "per_task_snr.csv"
    per_task_overrides.to_csv(per_task_csv, index=False)
    print(f"Wrote {per_task_csv.name}")

    group_rows: list[dict] = []

    # 1) Family-level strip plot by curation_category
    print("\nFamily-level Kruskal-Wallis by curation_category:")
    H, p, ng = _strip_plot(
        per_family,
        group_col="curation_category",
        value_col="snr_median",
        order=CATEGORY_ORDER,
        label_col="family",
        title=f"Per-family median {SNR_COL} by curation process",
        out_path=HERE / "snr_by_curation_process.png",
    )
    group_rows.append({"view": "family/curation", "H": H, "p": p, "n_groups": ng})

    # 2) Family-level strip plot by source_origin
    print("\nFamily-level Kruskal-Wallis by source_origin:")
    H, p, ng = _strip_plot(
        per_family,
        group_col="source_origin",
        value_col="snr_median",
        order=ORIGIN_ORDER,
        label_col="family",
        title=f"Per-family median {SNR_COL} by source origin",
        out_path=HERE / "snr_by_data_source.png",
    )
    group_rows.append({"view": "family/source", "H": H, "p": p, "n_groups": ng})

    # 3) Per-task strip plot by curation_category — exposes within-family
    #    heterogeneity the family-level view smears. Drop labels: too many
    #    points to annotate readably.
    print("\nPer-task Kruskal-Wallis by curation_category (with overrides):")
    H, p, ng = _strip_plot(
        per_task_overrides,
        group_col="curation_category",
        value_col=SNR_COL,
        order=CATEGORY_ORDER,
        label_col=None,
        title=f"Per-task {SNR_COL} by curation process (xnli_eu re-tagged as MT+post-edit)",
        out_path=HERE / "snr_by_curation_per_task.png",
    )
    group_rows.append({"view": "task/curation", "H": H, "p": p, "n_groups": ng})

    # 4) Headline: ranked bar chart of family medians, colored by curation.
    _ranked_bar(per_family, HERE / "snr_per_family_ranked.png")
    print("Wrote snr_per_family_ranked.png")

    # --- Task-format axes (Phase A) ----------------------------------------
    print("\nFamily-level Kruskal-Wallis by n_options:")
    per_family["n_options_str"] = per_family["n_options"].astype(int).astype(str)
    H, p, ng = _strip_plot(
        per_family,
        group_col="n_options_str",
        value_col="snr_median",
        order=["2", "3", "4"],
        label_col="family",
        title=f"Per-family median {SNR_COL} by number of answer options",
        out_path=HERE / "snr_by_n_options.png",
    )
    group_rows.append({"view": "family/n_options", "H": H, "p": p, "n_groups": ng})

    print("\nFamily-level Kruskal-Wallis by format:")
    format_order = ["minimal_pair", "completion", "classification",
                    "mcq_question_only", "mrc_passage"]
    H, p, ng = _strip_plot(
        per_family,
        group_col="format",
        value_col="snr_median",
        order=format_order,
        label_col="family",
        title=f"Per-family median {SNR_COL} by task format",
        out_path=HERE / "snr_by_format.png",
    )
    group_rows.append({"view": "family/format", "H": H, "p": p, "n_groups": ng})

    print("\nFamily-level Kruskal-Wallis by passage flag:")
    per_family["passage_str"] = per_family["passage"].map({True: "passage", False: "no_passage"})
    H, p, ng = _strip_plot(
        per_family,
        group_col="passage_str",
        value_col="snr_median",
        order=["no_passage", "passage"],
        label_col="family",
        title=f"Per-family median {SNR_COL} by passage flag",
        out_path=HERE / "snr_by_passage.png",
    )
    group_rows.append({"view": "family/passage", "H": H, "p": p, "n_groups": ng})

    # Continuous: SNR vs random_baseline (n_options = 1/baseline)
    _scatter_baseline(per_family, HERE / "snr_vs_random_baseline.png")
    print("Wrote snr_vs_random_baseline.png")

    # --- Length features (Phase B) -----------------------------------------
    if "context_len_chars_median" in per_family.columns:
        print("\nLength-feature correlations (Spearman):")
        for col, fname in [
            ("context_len_chars_median", "snr_vs_context_len.png"),
            ("option_len_chars_median",  "snr_vs_option_len.png"),
            ("context_to_option_ratio",  "snr_vs_context_option_ratio.png"),
        ]:
            rho, p = _scatter_length(
                per_family, col, HERE / fname,
                title=f"SNR vs {col}",
            )
            group_rows.append({
                "view": f"family/{col}",
                "H": float("nan"), "p": p, "n_groups": len(per_family),
                "spearman_rho": rho,
            })
        _length_grid(per_family, HERE / "snr_vs_length_features.png")
        print("Wrote snr_vs_length_features.png")

    pd.DataFrame(group_rows).to_csv(HERE / "group_stats.csv", index=False)
    print(f"\nWrote group_stats.csv")

    print("\nPer-family table (sorted by snr_median desc):")
    cols = ["family", "n_tasks", "snr_median", "snr_mean", "snr_max",
            "curation_category", "source_origin", "format", "n_options",
            "passage"]
    with pd.option_context("display.max_colwidth", 36, "display.width", 160):
        print(per_family[cols].to_string(index=False))


if __name__ == "__main__":
    main()
