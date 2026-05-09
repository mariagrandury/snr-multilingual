"""Postprocessing for `snr_definition` — answers the four questions
laid out in the README.

Q1 (per language) — best variant under DA-size and DA-ckpt.
    → best_variant_per_language.csv
Q2 (within-cluster similarities) — language groups by best variant.
    → variant_clusters.csv  (printed; the qualitative interpretation
      lives in the README).
Q3 (across languages) — top variants by mean Pearson r.
    → top_variants_overall.csv  +  top_variants_overall.png
Q4 (top benchmarks per language) — under the global-best DA-size variant,
    rank benchmarks per language with both DA-size and DA-ckpt values.
    → top_benchmarks_per_language.csv  +  top_benchmarks_per_language.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from multilingual.analyze_snr_variants import (  # noqa: E402
    _per_language_pearson_table, assign_language, da_ckpt_pairs,
    da_size_pairs, list_variants,
)
from snr.constants import PLOT_DIR  # noqa: E402

OUT_DIR = PLOT_DIR / "snr_definition"
TARGET_SIZE = "1B"
TOP_K = 5


# --- per-DA helpers ---------------------------------------------------------

def _pairs_for(da_kind: str):
    """``da_size_pairs`` or ``da_ckpt_pairs`` (cross-size pooled)."""
    if da_kind == "size":
        return list(da_size_pairs())
    if da_kind == "ckpt":
        return list(da_ckpt_pairs())
    raise ValueError(da_kind)


def _table(df: pd.DataFrame, da_kind: str) -> pd.DataFrame:
    return _per_language_pearson_table(df, list_variants(df), _pairs_for(da_kind))


# --- Q1 + Q2: best variant per language under each DA flavor ---------------

def best_variant_per_language(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tables = {k: _table(df, k) for k in ("size", "ckpt")}
    langs = sorted(set(tables["size"].columns) | set(tables["ckpt"].columns))
    for lang in langs:
        row = {"language": lang}
        for kind in ("size", "ckpt"):
            col = tables[kind].get(lang)
            if col is None or col.dropna().empty:
                row[f"best_variant_da_{kind}"] = ""
                row[f"best_pearson_r_da_{kind}"] = np.nan
                row[f"runner_up_variant_da_{kind}"] = ""
                row[f"runner_up_pearson_r_da_{kind}"] = np.nan
                continue
            sorted_col = col.dropna().sort_values(ascending=False)
            row[f"best_variant_da_{kind}"] = sorted_col.index[0]
            row[f"best_pearson_r_da_{kind}"] = float(sorted_col.iloc[0])
            row[f"runner_up_variant_da_{kind}"] = (
                sorted_col.index[1] if len(sorted_col) > 1 else "")
            row[f"runner_up_pearson_r_da_{kind}"] = (
                float(sorted_col.iloc[1]) if len(sorted_col) > 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def render_best_variant_per_language(best_df: pd.DataFrame, save_path: Path):
    """Two-panel grouped bar: per language, r of the best DA-size and
    DA-ckpt variants, annotated with the variant name."""
    langs = best_df["language"].tolist()
    rs_size = best_df["best_pearson_r_da_size"].to_numpy()
    rs_ckpt = best_df["best_pearson_r_da_ckpt"].to_numpy()
    names_size = best_df["best_variant_da_size"].tolist()
    names_ckpt = best_df["best_variant_da_ckpt"].tolist()

    x = np.arange(len(langs))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(langs)), 5.2))
    b1 = ax.bar(x - w / 2, np.where(np.isfinite(rs_size), rs_size, 0),
                width=w, label="DA-size", color="#1f77b4", alpha=0.85)
    b2 = ax.bar(x + w / 2, np.where(np.isfinite(rs_ckpt), rs_ckpt, 0),
                width=w, label="DA-ckpt", color="#ff7f0e", alpha=0.85)
    for rect, name in zip(b1, names_size):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                name, ha="center", va="bottom", fontsize=7, rotation=90)
    for rect, name in zip(b2, names_ckpt):
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                name, ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("Pearson r — log10(SNR) vs DA")
    ax.set_title("Best SNR variant per language (DA-size vs DA-ckpt)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    ymax = max(0.85, np.nanmax([rs_size, rs_ckpt]) + 0.15)
    ax.set_ylim(min(-0.1, np.nanmin([rs_size, rs_ckpt]) - 0.05), ymax)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# Variant families by mathematical structure (mirrors the redundancy
# clusters in variant_correlation_matrix.png). Used to roll up Q2.
_VARIANT_FAMILY = {
    # Absolute-spread (dispersion redundancy cluster, near-equivalent at n_mixes=3)
    "rms_deviation":   "dispersion",
    "mpd":             "dispersion",
    "dispersion":      "dispersion",
    "range":           "dispersion",
    "quartile_deviation": "dispersion",
    "aad":             "dispersion",
    "mpsd":            "dispersion",
    "dist_std":        "dispersion",
    # Relative-spread (mean-normalised — own redundancy cluster)
    "rel_std":         "rel_spread",
    "rel_mpd":         "rel_spread",
    "rel_mpsd":        "rel_spread",
    "rel_dispersion":  "rel_spread",
    "iqr":             "rel_spread",
    # Discrepancy / CDF-based
    "discrepancy":              "discrepancy",
    "star_discrepancy":         "discrepancy",
    "star_discrepancy_shifted": "discrepancy",
    "rel_star_discrepancy":     "discrepancy",
    "dispersion_shifted":       "discrepancy",
    "gini":                     "discrepancy",
    # Robust (median / depth-based)
    "mad":         "robust",
    "tukey":       "depth",
    "projection":  "depth",
}


def render_variant_family_per_language(best_df: pd.DataFrame, save_path: Path):
    """Heatmap-ish grid: rows = languages, cols = (DA-size, DA-ckpt),
    cell = the variant family that wins. Makes Q2 visible at a glance."""
    family_palette = {
        "dispersion":  "#1f77b4",
        "rel_spread":  "#2ca02c",
        "discrepancy": "#ff7f0e",
        "robust":      "#9467bd",
        "depth":       "#8c564b",
    }
    langs = best_df["language"].tolist()
    cols = ["DA-size", "DA-ckpt"]
    fig, ax = plt.subplots(figsize=(4.0, 0.34 * len(langs) + 1.0))
    for i, lang in enumerate(langs):
        for j, kind in enumerate(["size", "ckpt"]):
            v = best_df.iloc[i][f"best_variant_da_{kind}"]
            r = best_df.iloc[i][f"best_pearson_r_da_{kind}"]
            family = _VARIANT_FAMILY.get(v, "??")
            color = family_palette.get(family, "#cccccc")
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color,
                                       alpha=0.85, edgecolor="white"))
            label = f"{family}\n{v}\n(r={r:+.2f})" if v else "—"
            txt_color = "white" if family in ("discrepancy", "robust", "depth") else "black"
            ax.text(j + 0.5, i + 0.5, label, ha="center", va="center",
                    fontsize=7, color=txt_color)
    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(langs))
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(cols, fontsize=10)
    ax.set_yticks([i + 0.5 for i in range(len(langs))])
    ax.set_yticklabels(langs, fontsize=10)
    ax.invert_yaxis()
    ax.set_aspect("auto")
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=family_palette[k],
                             alpha=0.85)
               for k in family_palette]
    ax.legend(handles, list(family_palette.keys()), title="Variant family",
              loc="upper right", bbox_to_anchor=(1.45, 1), fontsize=8,
              title_fontsize=9)
    ax.set_title("Best variant family per language", fontsize=11)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def variant_clusters(best_df: pd.DataFrame) -> pd.DataFrame:
    """Q2: which languages share the same best variant?"""
    rows = []
    for kind in ("size", "ckpt"):
        col = f"best_variant_da_{kind}"
        for variant, sub in best_df.groupby(col):
            if not variant:
                continue
            rows.append({
                "da_kind": kind,
                "variant": variant,
                "n_languages": len(sub),
                "languages": ",".join(sorted(sub["language"].tolist())),
                "mean_r": float(sub[f"best_pearson_r_da_{kind}"].mean()),
            })
    return pd.DataFrame(rows).sort_values(
        ["da_kind", "n_languages", "mean_r"], ascending=[True, False, False]
    )


# --- Q3: across languages, top variants ------------------------------------

def top_variants_overall(df: pd.DataFrame) -> pd.DataFrame:
    """Mean Pearson r across languages per variant, for both DA-size and
    DA-ckpt. Returns a long table sorted by mean DA-size r descending."""
    rows = []
    tables = {k: _table(df, k) for k in ("size", "ckpt")}
    variants = sorted(set(tables["size"].index) | set(tables["ckpt"].index))
    for v in variants:
        rows.append({
            "variant": v,
            "mean_r_da_size": float(tables["size"].loc[v].mean(skipna=True))
            if v in tables["size"].index else np.nan,
            "mean_r_da_ckpt": float(tables["ckpt"].loc[v].mean(skipna=True))
            if v in tables["ckpt"].index else np.nan,
        })
    out = pd.DataFrame(rows)
    out["mean_r_overall"] = out[["mean_r_da_size", "mean_r_da_ckpt"]].mean(
        axis=1, skipna=True)
    return out.sort_values("mean_r_da_size", ascending=False).reset_index(drop=True)


def render_top_variants_overall(tv_df: pd.DataFrame, save_path: Path):
    """Horizontal lollipop: per variant, mean r under DA-size and DA-ckpt."""
    tv_df = tv_df.sort_values("mean_r_overall", ascending=True)
    y = np.arange(len(tv_df))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(tv_df))))
    ax.hlines(y - 0.15, 0, tv_df["mean_r_da_size"], color="#1f77b4", linewidth=2)
    ax.scatter(tv_df["mean_r_da_size"], y - 0.15, color="#1f77b4",
               s=40, label="DA-size")
    ax.hlines(y + 0.15, 0, tv_df["mean_r_da_ckpt"], color="#ff7f0e", linewidth=2)
    ax.scatter(tv_df["mean_r_da_ckpt"], y + 0.15, color="#ff7f0e",
               s=40, label="DA-ckpt")
    ax.set_yticks(y)
    ax.set_yticklabels(tv_df["variant"], fontsize=9)
    ax.set_xlabel("Mean Pearson r across languages (log10 SNR ↔ DA)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("SNR variants ranked by cross-language correlation with DA")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# --- Q4: top benchmarks per language under the global-best variant ----------

def top_benchmarks_per_language(df: pd.DataFrame, variant: str,
                                size: str = TARGET_SIZE,
                                top_k: int = TOP_K) -> pd.DataFrame:
    snr_col = f"snr_{variant}_{size}"
    if snr_col not in df.columns:
        raise KeyError(snr_col)
    df = df.copy()
    df["language"] = [assign_language(t) for t in df.index]
    df = df[df["language"] != "??"]
    da_size_col = f"decision_acc_size_{size}"
    # Mean ckpt-DA across the 3 early-step pairs at the same model size
    # (1B; the same size we use for SNR ranking).
    ckpt_cols = [f"decision_acc_ckpt_{e}_{size}"
                 for e in (6000, 18000, 28000)]
    df["da_ckpt_mean"] = df[ckpt_cols].mean(axis=1, skipna=True)

    rows = []
    for lang, sub in df.groupby("language"):
        sub_sorted = sub.sort_values(snr_col, ascending=False)
        sub_sorted = sub_sorted[sub_sorted[snr_col].notna()]
        for rank, (task, row) in enumerate(sub_sorted.head(top_k).iterrows(), 1):
            rows.append({
                "language": lang,
                "rank": rank,
                "task": task,
                "snr": float(row[snr_col]),
                "da_size": float(row.get(da_size_col, np.nan)),
                "da_ckpt_1B_mean": float(row.get("da_ckpt_mean", np.nan)),
            })
    return pd.DataFrame(rows)


def render_top_benchmarks_grid(top_df: pd.DataFrame, variant: str,
                               save_path: Path):
    langs = sorted(top_df["language"].unique())
    n = len(langs)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.2 * nrows),
                             squeeze=False)
    for i, lang in enumerate(langs):
        ax = axes[i // ncols][i % ncols]
        sub = top_df[top_df["language"] == lang].sort_values("snr",
                                                             ascending=True)
        ax.barh(range(len(sub)), sub["snr"], color="#1f77b4", alpha=0.85)
        ax.set_yticks(range(len(sub)))
        ax.set_yticklabels(sub["task"], fontsize=8)
        for j, (_, row) in enumerate(sub.iterrows()):
            ax.text(row["snr"] + 0.02, j,
                    f"DA-s={row['da_size']:.2f} DA-c={row['da_ckpt_1B_mean']:.2f}",
                    fontsize=6, va="center")
        ax.set_xlabel(f"SNR ({variant} @ {TARGET_SIZE})", fontsize=8)
        ax.set_title(f"{lang}  (top {len(sub)})", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(f"Top-{TOP_K} benchmarks per language by SNR — variant "
                 f"`{variant}` @ {TARGET_SIZE}  (annotations: DA-size, DA-ckpt mean)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# --- driver ----------------------------------------------------------------

def main():
    csv_path = OUT_DIR / "snr_variants_per_task.csv"
    df = pd.read_csv(csv_path, index_col="task")

    # Q1 + Q2
    best_df = best_variant_per_language(df)
    best_path = OUT_DIR / "best_variant_per_language.csv"
    best_df.to_csv(best_path, index=False)
    print(f"Wrote → {best_path}")
    print(best_df.to_string(index=False))

    render_best_variant_per_language(
        best_df, OUT_DIR / "best_variant_per_language.png")
    print(f"Wrote → {OUT_DIR / 'best_variant_per_language.png'}")

    render_variant_family_per_language(
        best_df, OUT_DIR / "best_variant_family_per_language.png")
    print(f"Wrote → {OUT_DIR / 'best_variant_family_per_language.png'}")

    # Roll up best_df by family for the README narrative.
    fam_df = best_df.copy()
    fam_df["family_da_size"] = fam_df["best_variant_da_size"].map(
        _VARIANT_FAMILY).fillna("??")
    fam_df["family_da_ckpt"] = fam_df["best_variant_da_ckpt"].map(
        _VARIANT_FAMILY).fillna("??")
    fam_df.to_csv(OUT_DIR / "best_variant_family_per_language.csv", index=False)
    print(f"Wrote → {OUT_DIR / 'best_variant_family_per_language.csv'}")

    cluster_df = variant_clusters(best_df)
    cluster_path = OUT_DIR / "variant_clusters.csv"
    cluster_df.to_csv(cluster_path, index=False)
    print(f"\nWrote → {cluster_path}")
    print(cluster_df.to_string(index=False))

    # Q3
    tv_df = top_variants_overall(df)
    tv_path = OUT_DIR / "top_variants_overall.csv"
    tv_df.to_csv(tv_path, index=False)
    print(f"\nWrote → {tv_path}")
    print(tv_df.head(8).to_string(index=False))

    render_top_variants_overall(tv_df, OUT_DIR / "top_variants_overall.png")
    print(f"Wrote → {OUT_DIR / 'top_variants_overall.png'}")

    g_best = tv_df.iloc[0]["variant"]
    print(f"\nGlobal best variant (mean Pearson r across languages, DA-size): {g_best}")

    # Q4
    top_df = top_benchmarks_per_language(df, g_best)
    top_path = OUT_DIR / "top_benchmarks_per_language.csv"
    top_df.to_csv(top_path, index=False)
    print(f"\nWrote → {top_path}  ({len(top_df)} rows)")

    render_top_benchmarks_grid(
        top_df, g_best, OUT_DIR / "top_benchmarks_per_language.png")
    print(f"Wrote → {OUT_DIR / 'top_benchmarks_per_language.png'}")


if __name__ == "__main__":
    main()
