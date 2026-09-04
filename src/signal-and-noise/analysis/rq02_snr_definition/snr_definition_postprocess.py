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

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import load_pools, load_snr_params  # noqa: E402
from analysis.rq02_snr_definition.analyze_snr_variants import (  # noqa: E402
    _per_language_pearson_table, assign_language, benchmark_family,
    buckets_in_df, da_ckpt_pairs, da_size_pairs, list_variants,
)
from analysis.autodoc import (  # noqa: E402
    CANONICAL_POOL, HOLDOUT, SLIDES, fmt, md_table, replace_block)
from snr.constants import PLOT_DIR  # noqa: E402
from analysis.paths import SNR_DEFINITION

TARGET_SIZE = load_snr_params()["target_size"]
TOP_K = 5


# --- per-DA helpers ---------------------------------------------------------

def _pairs_for(df: pd.DataFrame, da_kind: str):
    """``da_size_pairs`` or ``da_ckpt_pairs`` (cross-size pooled), built from
    the CSV columns of ``df``."""
    if da_kind == "size":
        return list(da_size_pairs(df))
    if da_kind == "ckpt":
        return list(da_ckpt_pairs(df))
    raise ValueError(da_kind)


def _table(df: pd.DataFrame, da_kind: str) -> pd.DataFrame:
    return _per_language_pearson_table(df, list_variants(df), _pairs_for(df, da_kind))


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

def reference_bucket(df: pd.DataFrame) -> str:
    """The size the per-language benchmark ranking is read at: the configured
    target when its SNR columns exist, else the largest bucket in the CSV
    (the ladder is analysed while the reference rungs are still training)."""
    buckets = buckets_in_df(df)
    return TARGET_SIZE if TARGET_SIZE in buckets else buckets[-1]


def top_benchmarks_per_language(df: pd.DataFrame, variant: str,
                                size: str | None = None,
                                top_k: int = TOP_K) -> pd.DataFrame:
    size = size or reference_bucket(df)
    snr_col = f"snr_{variant}_{size}"
    if snr_col not in df.columns:
        raise KeyError(snr_col)
    df = df.copy()
    df["language"] = [assign_language(t) for t in df.index]
    df = df[df["language"] != "??"]
    da_size_col = f"decision_acc_size_{size}"
    # Mean ckpt-DA across the relative-fraction early ckpts at the same bucket
    # (1B; the same bucket we use for SNR ranking). Columns are named with the
    # fraction label (f12/f36/f56), so select them by pattern.
    import re as _re
    ckpt_cols = [c for c in df.columns
                 if _re.match(rf"^decision_acc_ckpt_f\d+_{_re.escape(size)}$", c)]
    df["da_ckpt_mean"] = (df[ckpt_cols].mean(axis=1, skipna=True)
                          if ckpt_cols else np.nan)

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
                "da_ckpt_mean": float(row.get("da_ckpt_mean", np.nan)),
                "size": size,
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
                    f"DA-s={row['da_size']:.2f} DA-c={row['da_ckpt_mean']:.2f}",
                    fontsize=6, va="center")
        ax.set_xlabel(f"SNR ({variant} @ {top_df['size'].iloc[0]})", fontsize=8)
        ax.set_title(f"{lang}  (top {len(sub)})", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    fig.suptitle(f"Top-{TOP_K} benchmarks per language by SNR — variant "
                 f"`{variant}` @ {top_df['size'].iloc[0]}  (annotations: DA-size, DA-ckpt mean)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# --- auto-generated README + slides (RQ1) ----------------------------------
# These rewrite the marker-delimited "Highlighted result" / "Results" blocks of
# results/snr_definition/README.md and the RQ1 results slide. They fire only for
# the canonical pool (so the per-tier pipeline loop writes the docs once, from
# the comprehensive pool, after the pure-pool CSVs already exist for the
# statistical-power table). RQ / setup / TODO prose lives outside the markers.

_POOL_TIERS = [
    ("predictivity", "grid, seed 1904"),
    ("predictivity_seeds", "all seeds"),
    ("predictivity_seeds_train", "holdout train (seeds 64/313)"),
    ("predictivity_seeds_test", "holdout test (seed 1904)"),
]


def _snr_dir(stage: str, pool: str) -> Path:
    return SNR_DEFINITION / stage / pool


def _read_tv(stage: str, pool: str) -> pd.DataFrame | None:
    p = _snr_dir(stage, pool) / "top_variants_overall.csv"
    return pd.read_csv(p) if p.exists() else None


def _anchor_rank1(stage: str, pool: str) -> pd.DataFrame:
    """Rank-1 benchmark per language from top_benchmarks_per_language.csv."""
    df = pd.read_csv(_snr_dir(stage, pool) / "top_benchmarks_per_language.csv")
    return df[df["rank"] == 1].sort_values("language").reset_index(drop=True)


def _holdout_metrics(stage: str) -> dict | None:
    p = _snr_dir(stage, f"{HOLDOUT[0]}__vs__{HOLDOUT[1]}") / "headline_metrics.csv"
    if not p.exists():
        return None
    hm = pd.read_csv(p)
    return {(r.metric, r.da_kind): float(r.value) for r in hm.itertuples()}


def _readme_blocks(stage: str, pool: str) -> tuple[str, str]:
    tv = _read_tv(stage, pool)
    g = tv.iloc[0]                                   # best by DA-size
    ckpt_sorted = tv.sort_values("mean_r_da_ckpt", ascending=False)
    ckpt_leaders = ckpt_sorted.head(3)["variant"].tolist()
    ckpt_r = ckpt_sorted.iloc[0]["mean_r_da_ckpt"]
    families = {_VARIANT_FAMILY.get(v, "??") for v in ckpt_leaders}
    worst = tv.sort_values("mean_r_overall").head(2)["variant"].tolist()
    anchor = _anchor_rank1(stage, pool)
    fam_counts = anchor["task"].map(benchmark_family).value_counts()
    fam_name, fam_n = fam_counts.index[0], int(fam_counts.iloc[0])
    n_langs = anchor["language"].nunique()
    ref = anchor["size"].iloc[0]
    hm = _holdout_metrics(stage)

    bullets = [
        f"- **Global-best SNR definition (`{pool}`): `{g.variant}`** — mean Pearson r of "
        f"log₁₀(SNR) vs decision accuracy **{fmt(g.mean_r_da_size)}** (DA-size), "
        f"**{fmt(g.mean_r_da_ckpt)}** (DA-ckpt), {fmt(g.mean_r_overall)} overall. "
        f"DA-ckpt is led by `{'`/`'.join(ckpt_leaders)}` (≈ {fmt(ckpt_r)}; "
        f"{'one family: ' + next(iter(families)) if len(families) == 1 else 'families: ' + ', '.join(sorted(families))}) — "
        f"recommend the *family*, not an exact variant.",
        f"- **Per-language anchor: `{fam_name}`** — the highest-SNR above-random benchmark "
        f"in **{fam_n} of {n_langs}** languages (`{g.variant}` SNR @ {ref}). "
        f"Weakest variants overall: `{'`, `'.join(worst)}`.",
    ]
    if hm is not None:
        bullets.append(
            f"- **Seed holdout ({HOLDOUT[0]} → {HOLDOUT[1]})**: Spearman ρ of the global "
            f"variant ranking **{fmt(hm[('spearman_rank_global','ckpt')])}** (DA-ckpt), "
            f"**{fmt(hm[('spearman_rank_global','size')])}** (DA-size); family-level "
            f"per-language agreement {hm[('family_agreement','ckpt')]:.0%} / "
            f"{hm[('family_agreement','size')]:.0%}. A ranking that does not survive the "
            f"seed swap is noise-dominated — only the *family* recommendation transfers.")
    highlight = "\n".join(bullets)

    # 1) variant ranking — top 7 + bottom 2
    vr = []
    for _, r in tv.head(7).iterrows():
        vr.append([f"`{r.variant}`", fmt(r.mean_r_da_size), fmt(r.mean_r_da_ckpt),
                   fmt(r.mean_r_overall)])
    vr.append(["…", "", "", ""])
    for _, r in tv.tail(2).iterrows():
        vr.append([f"`{r.variant}`", fmt(r.mean_r_da_size), fmt(r.mean_r_da_ckpt),
                   fmt(r.mean_r_overall)])
    t_variants = md_table(["variant", "DA-size r", "DA-ckpt r", "overall"], vr)

    # 2) statistical power by pool
    pw = []
    for p, lab in _POOL_TIERS:
        tvp = _read_tv(stage, p)
        if tvp is None:
            continue
        bp = tvp.iloc[0]
        pw.append([f"`{p}` ({lab})", f"`{bp.variant}`", fmt(bp.mean_r_da_size),
                   fmt(bp.mean_r_da_ckpt)])
    t_power = md_table(["pool", "best variant (DA-size)", "DA-size r", "DA-ckpt r"], pw)

    # 3) per-language anchor
    an = [[r.language, f"`{r.task}`", fmt(r.snr), fmt(r.da_ckpt_mean)]
          for _, r in anchor.iterrows()]
    t_anchor = md_table(["lang", "top benchmark", "SNR", f"DA-ckpt@{ref}"], an)

    results = [
        f"Headline numbers from the `{pool}` pool. Regenerate with "
        f"`python analysis/rq02_snr_definition/snr_definition_postprocess.py --pool {pool}`.",
        "**Global variant ranking** — mean Pearson r of log₁₀(SNR) vs DA across languages:",
        t_variants,
        f"![SNR variants ranked by correlation with DA]({stage}/{pool}/top_variants_overall.png)",
        "**Statistical power by pool** — each pool's best DA-size variant:",
        t_power,
        f"**Most reliable benchmark per language** — `{g.variant}` SNR @ {ref} over "
        f"above-random tasks (DA-size is undefined at the reference size itself, so "
        f"DA-ckpt@{ref} is shown):",
        t_anchor,
        f"![Top-5 benchmarks per language by SNR]({stage}/{pool}/top_benchmarks_per_language.png)",
    ]

    # 4) seed holdout (only once compare_seed_splits.py has run)
    if hm is not None:
        pct = lambda m, k: f"{hm[(m, k)]:.0%}"
        ho = [
            ["Spearman ρ on global variant ranking",
             fmt(hm[("spearman_rank_global", "size")]), fmt(hm[("spearman_rank_global", "ckpt")])],
            ["Pearson r between splits (all cells)",
             fmt(hm[("pearson_r_cells", "size")]), fmt(hm[("pearson_r_cells", "ckpt")])],
            ["Exact-variant agreement (per lang)",
             pct("exact_variant_agreement", "size"), pct("exact_variant_agreement", "ckpt")],
            ["Family-level agreement (per lang)",
             pct("family_agreement", "size"), pct("family_agreement", "ckpt")],
            ["Retention of train-best r on test",
             pct("retention", "size"), pct("retention", "ckpt")],
        ]
        results += [
            f"**Seed generalization** — holdout `{HOLDOUT[0]}` → `{HOLDOUT[1]}` "
            f"(the ×3 cells only). A variant ranking whose Spearman ρ is low here is "
            f"noise-dominated; recommend the family that transfers, not the argmax:",
            md_table(["metric", "DA-size", "DA-ckpt"], ho),
        ]
    return highlight, "\n\n".join(results)


def generate_readme(stage: str, pool: str) -> None:
    """Rewrite the auto blocks of results/snr_definition/README.md (canonical
    pool only)."""
    if pool != CANONICAL_POOL:
        return
    highlight, results = _readme_blocks(stage, pool)
    readme = SNR_DEFINITION / "README.md"
    gen = f"snr_definition_postprocess.py --pool {pool}"
    replace_block(readme, "highlight", "## Highlighted result\n\n" + highlight, gen)
    replace_block(readme, "results", "## Results\n\n" + results, gen)
    print(f"Wrote auto README blocks → {readme}")


def generate_slides(stage: str, pool: str) -> None:
    """Rewrite the RQ1 auto results slide in the deck (canonical pool only)."""
    if pool != CANONICAL_POOL:
        return
    anchor = _anchor_rank1(stage, pool)
    g_variant = _read_tv(stage, pool).iloc[0]["variant"]
    ref = anchor["size"].iloc[0]
    rows = [[r.language, f"`{r.task}`", fmt(r.snr, 1), fmt(r.da_ckpt_mean)]
            for _, r in anchor.iterrows()]
    slide = (
        "---\n"
        "title: RQ1 — SNR Definition\n"
        f"subtitle: \"Results (auto) — most reliable benchmark per language "
        f"(`{g_variant}` @ {ref})\"\n"
        "---\n\n"
        f"{md_table(['lang', 'top benchmark', 'SNR', f'DA-ckpt@{ref}'], rows)}\n\n"
        "<style>\n.slidev-layout table { font-size: 0.7em; }\n</style>"
    )
    replace_block(SLIDES, "rq1-results", slide,
                  "snr_definition_postprocess.py")
    print(f"Wrote RQ1 results slide → {SLIDES}")


# --- driver ----------------------------------------------------------------

def main(stage: str, pool: str, out_dir: Path):
    csv_path = out_dir / "snr_variants_per_task.csv"
    df = pd.read_csv(csv_path, index_col="task")

    # Q1 + Q2
    best_df = best_variant_per_language(df)
    best_path = out_dir / "best_variant_per_language.csv"
    best_df.to_csv(best_path, index=False)
    print(f"Wrote → {best_path}")
    print(best_df.to_string(index=False))

    render_best_variant_per_language(
        best_df, out_dir / "best_variant_per_language.png")
    print(f"Wrote → {out_dir / 'best_variant_per_language.png'}")

    render_variant_family_per_language(
        best_df, out_dir / "best_variant_family_per_language.png")
    print(f"Wrote → {out_dir / 'best_variant_family_per_language.png'}")

    # Roll up best_df by family for the README narrative.
    fam_df = best_df.copy()
    fam_df["family_da_size"] = fam_df["best_variant_da_size"].map(
        _VARIANT_FAMILY).fillna("??")
    fam_df["family_da_ckpt"] = fam_df["best_variant_da_ckpt"].map(
        _VARIANT_FAMILY).fillna("??")
    fam_df.to_csv(out_dir / "best_variant_family_per_language.csv", index=False)
    print(f"Wrote → {out_dir / 'best_variant_family_per_language.csv'}")

    cluster_df = variant_clusters(best_df)
    cluster_path = out_dir / "variant_clusters.csv"
    cluster_df.to_csv(cluster_path, index=False)
    print(f"\nWrote → {cluster_path}")
    print(cluster_df.to_string(index=False))

    # Q3
    tv_df = top_variants_overall(df)
    tv_path = out_dir / "top_variants_overall.csv"
    tv_df.to_csv(tv_path, index=False)
    print(f"\nWrote → {tv_path}")
    print(tv_df.head(8).to_string(index=False))

    render_top_variants_overall(tv_df, out_dir / "top_variants_overall.png")
    print(f"Wrote → {out_dir / 'top_variants_overall.png'}")

    g_best = tv_df.iloc[0]["variant"]
    print(f"\nGlobal best variant (mean Pearson r across languages, DA-size): {g_best}")

    # Q4
    top_df = top_benchmarks_per_language(df, g_best)
    top_path = out_dir / "top_benchmarks_per_language.csv"
    top_df.to_csv(top_path, index=False)
    print(f"\nWrote → {top_path}  ({len(top_df)} rows)")

    render_top_benchmarks_grid(
        top_df, g_best, out_dir / "top_benchmarks_per_language.png")
    print(f"Wrote → {out_dir / 'top_benchmarks_per_language.png'}")

    # Auto-refresh the README "Highlighted result" / "Results" blocks and the
    # RQ1 results slide (canonical pool only — no-op otherwise).
    generate_readme(stage, pool)
    generate_slides(stage, pool)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True,
                   help="Pool name from configs/models.json (tiers: 1seed, "
                        "2seeds, 3seeds, 3seeds_swissai_hf). Reads "
                        "results/<stage>/snr_definition/<pool>/snr_variants_per_task.csv.")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; "
                f"available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    main(stage=stage, pool=args.pool,
         out_dir=SNR_DEFINITION / stage / args.pool)
