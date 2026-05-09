"""Cross-corpus SNR transfer analysis: Apertus ↔ AllenAI DataDecide.

Inputs (must already exist):
  results/snr_definition/snr_variants_per_task.csv
  results/allenai_comparison/allenai_snr_variants_per_task.csv

Outputs (all in this directory):
  task_overlap.csv
  pearson_r_per_variant.csv
  snr_apertus_vs_snr_allenai_<best_variant>.png
  snr_apertus_vs_snr_allenai_grid.png
  top_apertus.csv  top_allenai.csv  agreement.md
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from snr.constants import PLOT_DIR
from snr.snr_variants import AGGREGATION_FUNCTIONS

from multilingual.run_apertus_snr_variants import variant_key

OUT_DIR = PLOT_DIR / "allenai_comparison"
APERTUS_CSV = PLOT_DIR / "snr_definition" / "snr_variants_per_task.csv"
ALLENAI_CSV = OUT_DIR / "allenai_snr_variants_per_task.csv"

APERTUS_SIZE = "1B"   # largest size in Apertus
ALLENAI_SIZE = "1B"   # matched: AllenAI also has 1B in DataDecide

# Approximate matched-size pairs for the size-sweep (Apertus → AllenAI).
SIZE_PAIRS = [
    ("175M", "150M"),
    ("350M", "300M"),
    ("600M", "750M"),
    ("1B",   "1B"),
]

VARIANTS = [variant_key(fd) for fd in AGGREGATION_FUNCTIONS]
VARIANT_TITLES = {variant_key(fd): fd["title"] for fd in AGGREGATION_FUNCTIONS}


# --- task name aliasing ----------------------------------------------------

# Direct (Apertus → AllenAI) aliases for tasks that are the same eval
# under different lm-eval task keys. The MMLU family is handled
# separately below since it's a per-subject prefix rewrite, not a single
# rename.
_TASK_ALIASES: dict[str, str] = {
    "commonsense_qa": "csqa",
    # Add more obvious mismatches here as we discover them.
}


def _canonicalise_apertus_task(t: str) -> str:
    """Map Apertus task names to their AllenAI equivalents.

    MMLU: Apertus only ran the multilingual ``global_mmlu_full_en_<subject>``
    view on a full ckpt-series — the vanilla ``mmlu_<subject>`` rows are
    single-shot and have no SNR. AllenAI exposes the same content under
    the vanilla ``mmlu_<subject>`` names, so we alias. Note: Apertus's
    ``global_mmlu_full_en`` is the **Cohere Full** translation of MMLU,
    not the original — see ``agreement.md`` for the methodological
    caveat.

    Other one-off renames live in ``_TASK_ALIASES``.
    """
    if t == "global_mmlu_full_en":
        return "mmlu"
    if t.startswith("global_mmlu_full_en_"):
        return "mmlu_" + t[len("global_mmlu_full_en_"):]
    return _TASK_ALIASES.get(t, t)


def _load_apertus_with_alias() -> pd.DataFrame:
    """Apply the canonicalisation, then drop duplicate canonical names by
    keeping the row with the most non-NaN values (i.e., the alias wins
    over the empty vanilla row whenever both exist)."""
    df = pd.read_csv(APERTUS_CSV, index_col="task")
    df = df.reset_index()
    df["_canonical"] = df["task"].map(_canonicalise_apertus_task)
    df["_n_finite"] = df.drop(columns=["task", "_canonical"]).notna().sum(axis=1)
    df = (
        df.sort_values("_n_finite", ascending=False)
          .drop_duplicates("_canonical", keep="first")
          .drop(columns=["task", "_n_finite"])
          .rename(columns={"_canonical": "task"})
          .set_index("task")
          .sort_index()
    )
    return df


# --- helpers ---------------------------------------------------------------

def _safe_log10(x: pd.Series) -> pd.Series:
    """log10 with non-positive values mapped to NaN."""
    arr = pd.to_numeric(x, errors="coerce")
    return np.log10(arr.where(arr > 0))


def _build_task_overlap(ap_tasks: set[str], al_tasks: set[str]) -> pd.DataFrame:
    all_tasks = sorted(ap_tasks | al_tasks)
    rows = [
        {
            "task": t,
            "in_apertus": t in ap_tasks,
            "in_allenai": t in al_tasks,
            "shared": (t in ap_tasks) and (t in al_tasks),
        }
        for t in all_tasks
    ]
    return pd.DataFrame(rows).set_index("task")


def _pearson_for_pair(
    ap_df: pd.DataFrame,
    al_df: pd.DataFrame,
    shared: list[str],
    ap_size: str,
    al_size: str,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """For each variant, Pearson r between log10(SNR) vectors over shared tasks.

    Returns:
      summary: DataFrame indexed by variant, with `n`, `r`, `p`.
      per_variant_xy: dict[variant -> 2-col DataFrame (apertus, allenai)] for
        plotting, restricted to tasks where both vectors are finite.
    """
    rows = []
    per_variant_xy: dict[str, pd.DataFrame] = {}
    for v in VARIANTS:
        ap_col = f"snr_{v}_{ap_size}"
        al_col = f"snr_{v}_{al_size}"
        if ap_col not in ap_df.columns or al_col not in al_df.columns:
            rows.append({"variant": v, "n": 0, "r": float("nan"), "p": float("nan")})
            continue

        ap = _safe_log10(ap_df.loc[shared, ap_col]).rename("apertus")
        al = _safe_log10(al_df.loc[shared, al_col]).rename("allenai")
        xy = pd.concat([ap, al], axis=1).dropna()
        per_variant_xy[v] = xy

        if len(xy) < 3:
            rows.append({"variant": v, "n": len(xy), "r": float("nan"), "p": float("nan")})
            continue
        r, p = pearsonr(xy["apertus"].to_numpy(), xy["allenai"].to_numpy())
        rows.append({"variant": v, "n": len(xy), "r": r, "p": p})
    return pd.DataFrame(rows).set_index("variant"), per_variant_xy


def _per_variant_pearson(
    ap_df: pd.DataFrame, al_df: pd.DataFrame, shared: list[str]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Default headline correlation: Apertus@1B ↔ AllenAI@1B."""
    return _pearson_for_pair(ap_df, al_df, shared, APERTUS_SIZE, ALLENAI_SIZE)


def _pearson_size_sweep(
    ap_df: pd.DataFrame, al_df: pd.DataFrame, shared: list[str]
) -> pd.DataFrame:
    """Per-variant Pearson r for every matched-size pair.

    Long-format table: columns = (variant, apertus_size, allenai_size, n, r, p).
    Lets the README show whether the cross-corpus correlation strengthens
    or weakens with model size.
    """
    rows = []
    for ap_size, al_size in SIZE_PAIRS:
        summary, _ = _pearson_for_pair(ap_df, al_df, shared, ap_size, al_size)
        for v in summary.index:
            rows.append({
                "variant": v,
                "apertus_size": ap_size,
                "allenai_size": al_size,
                "n": summary.loc[v, "n"],
                "r": summary.loc[v, "r"],
                "p": summary.loc[v, "p"],
            })
    return pd.DataFrame(rows)


def _plot_scatter(xy: pd.DataFrame, variant: str, r: float, n: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(xy["apertus"], xy["allenai"], alpha=0.6, s=30, edgecolors="none")
    # diagonal y=x reference
    lo = min(xy["apertus"].min(), xy["allenai"].min())
    hi = max(xy["apertus"].max(), xy["allenai"].max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray", lw=0.8, ls="--")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel(f"log10 SNR  —  Apertus {APERTUS_SIZE}")
    ax.set_ylabel(f"log10 SNR  —  AllenAI {ALLENAI_SIZE}")
    ax.set_title(
        f"{VARIANT_TITLES.get(variant, variant)}\n"
        f"shared tasks: n = {n}  |  Pearson r = {r:.3f}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_grid(
    per_variant_xy: dict[str, pd.DataFrame],
    summary: pd.DataFrame,
    path: Path,
) -> None:
    """Per-variant grid sorted top-to-bottom by Pearson r (descending)."""
    ordered = summary.sort_values("r", ascending=False).index.tolist()
    n = len(ordered)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.2))
    axes = np.atleast_2d(axes)

    for i, v in enumerate(ordered):
        ax = axes[i // ncols, i % ncols]
        xy = per_variant_xy.get(v, pd.DataFrame())
        if not xy.empty:
            ax.scatter(xy["apertus"], xy["allenai"], alpha=0.55, s=14, edgecolors="none")
            lo = min(xy["apertus"].min(), xy["allenai"].min())
            hi = max(xy["apertus"].max(), xy["allenai"].max())
            pad = 0.05 * (hi - lo if hi > lo else 1.0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="gray", lw=0.6, ls="--")
        r = summary.loc[v, "r"]
        n_v = int(summary.loc[v, "n"]) if not pd.isna(summary.loc[v, "n"]) else 0
        title = VARIANT_TITLES.get(v, v)
        ax.set_title(f"{title}\nr={r:.2f}  n={n_v}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.supxlabel(f"log10 SNR — Apertus {APERTUS_SIZE}", fontsize=10)
    fig.supylabel(f"log10 SNR — AllenAI {ALLENAI_SIZE}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --- top-K reliability agreement -------------------------------------------

def _top_k_table(df: pd.DataFrame, snr_col: str, k: int) -> pd.DataFrame:
    s = pd.to_numeric(df[snr_col], errors="coerce").dropna()
    return s.sort_values(ascending=False).head(k).rename("snr").to_frame()


def _agreement_at_k(top_a: pd.Index, top_b: pd.Index, k: int) -> dict:
    a = set(top_a)
    b = set(top_b)
    inter = a & b
    union = a | b
    return {
        "k": k,
        "n_intersection": len(inter),
        "intersection_over_k": len(inter) / k if k else float("nan"),
        "jaccard": len(inter) / len(union) if union else float("nan"),
        "shared_top_tasks": ", ".join(sorted(inter)),
    }


# --- driver ----------------------------------------------------------------

def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ap_df = _load_apertus_with_alias()
    al_df = pd.read_csv(ALLENAI_CSV, index_col="task")

    # Task overlap.
    overlap = _build_task_overlap(set(ap_df.index), set(al_df.index))
    overlap.to_csv(OUT_DIR / "task_overlap.csv")
    shared = overlap[overlap["shared"]].index.tolist()
    print(
        f"Apertus: {len(ap_df)} tasks  |  AllenAI: {len(al_df)} tasks  "
        f"|  shared: {len(shared)}"
    )

    # Per-variant Pearson r at the headline (Apertus@1B ↔ AllenAI@1B).
    summary, per_variant_xy = _per_variant_pearson(ap_df, al_df, shared)
    summary = summary.sort_values("r", ascending=False)
    summary.to_csv(OUT_DIR / "pearson_r_per_variant.csv")
    print("\nPer-variant Pearson r (top 5 / bottom 5):")
    print(summary.head(5).round(3))
    print("...")
    print(summary.tail(5).round(3))

    # Long-format size sweep: same r but at every matched-size pair.
    sweep = _pearson_size_sweep(ap_df, al_df, shared)
    sweep.to_csv(OUT_DIR / "pearson_r_size_sweep.csv", index=False)

    # Headline scatter on the best variant.
    best_variant = summary["r"].idxmax()
    best_r = summary.loc[best_variant, "r"]
    best_n = int(summary.loc[best_variant, "n"])
    headline_path = OUT_DIR / f"snr_apertus_vs_snr_allenai_{best_variant}.png"
    _plot_scatter(per_variant_xy[best_variant], best_variant, best_r, best_n, headline_path)
    print(f"\nHeadline scatter ({best_variant}, r={best_r:.3f}) → {headline_path.name}")

    # Per-variant grid.
    grid_path = OUT_DIR / "snr_apertus_vs_snr_allenai_grid.png"
    _plot_grid(per_variant_xy, summary, grid_path)
    print(f"Variant grid → {grid_path.name}")

    # Top-K reliability agreement (use best variant).
    snr_col_ap = f"snr_{best_variant}_{APERTUS_SIZE}"
    snr_col_al = f"snr_{best_variant}_{ALLENAI_SIZE}"
    # Restrict each side's ranking to the shared-task set so we measure
    # rank agreement on a comparable list. (If we ranked the union, only
    # AllenAI-only tasks would compete for AllenAI's slots.)
    ap_shared = ap_df.loc[shared]
    al_shared = al_df.loc[shared]

    K_LIST = [5, 10, 20]
    top_ap_full = _top_k_table(ap_shared, snr_col_ap, max(K_LIST))
    top_al_full = _top_k_table(al_shared, snr_col_al, max(K_LIST))
    top_ap_full.to_csv(OUT_DIR / "top_apertus.csv")
    top_al_full.to_csv(OUT_DIR / "top_allenai.csv")

    rows = []
    for k in K_LIST:
        ap_k = _top_k_table(ap_shared, snr_col_ap, k).index
        al_k = _top_k_table(al_shared, snr_col_al, k).index
        rows.append(_agreement_at_k(ap_k, al_k, k))
    agreement_df = pd.DataFrame(rows)

    # Write agreement.md
    aliased_mmlu = sorted(t for t in shared if t == "mmlu" or t.startswith("mmlu_"))
    other_aliases = sorted(
        f"{src} → {dst}" for src, dst in _TASK_ALIASES.items() if dst in shared
    )
    md_lines = [
        "# Top-K reliability agreement",
        "",
        f"Variant used: **{VARIANT_TITLES[best_variant]}** (`{best_variant}`)",
        f"Apertus SNR column: `snr_{best_variant}_{APERTUS_SIZE}`  ·  "
        f"AllenAI SNR column: `snr_{best_variant}_{ALLENAI_SIZE}`",
        f"Shared-task universe: **{len(shared)}** tasks.",
        "",
        "## ⚠️ Methodological caveat — MMLU aliasing",
        "",
        "Apertus's `global_mmlu_full_en[_<subject>]` rows are aliased to "
        "AllenAI's `mmlu[_<subject>]` rows so the cross-corpus comparison "
        "can use the ~60 MMLU subjects. **The two are not the same content.** "
        "Apertus runs the **Cohere Full** translation/post-edit of MMLU "
        "(English split), AllenAI runs the original Hendrycks et al. MMLU. "
        "Question wording, post-edits, and sample coverage may differ. "
        "Plan: re-run the original `mmlu` lm-eval task on the multilingual "
        "Apertus checkpoints; once that lands, drop the alias and compare "
        "like-for-like.",
        "",
        f"MMLU rows aliased into the shared set: **{len(aliased_mmlu)}** "
        f"of {len(shared)} total.",
        "",
        "Other Apertus → AllenAI aliases that hit the shared set: "
        + (", ".join(f"`{a}`" for a in other_aliases) if other_aliases else "_none_")
        + ".",
        "",
        "## Top-K agreement",
        "",
        "| K | n_intersection | intersection / K | Jaccard | Shared top-K tasks |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['k']} | {r['n_intersection']} | {r['intersection_over_k']:.2f} "
            f"| {r['jaccard']:.2f} | {r['shared_top_tasks'] or '—'} |"
        )
    md_lines += [
        "",
        f"## Top-{max(K_LIST)} per corpus",
        "",
        "### Apertus",
        "",
        top_ap_full.round(3).to_markdown(),
        "",
        "### AllenAI",
        "",
        top_al_full.round(3).to_markdown(),
        "",
    ]
    (OUT_DIR / "agreement.md").write_text("\n".join(md_lines))
    print("\nAgreement table:")
    print(agreement_df.to_string(index=False))


if __name__ == "__main__":
    run()
