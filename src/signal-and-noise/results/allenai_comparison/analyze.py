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

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from snr.constants import PLOT_DIR
from snr.snr_variants import AGGREGATION_FUNCTIONS

from evals.scripts.utils.configs import load_pools  # noqa: E402
from multilingual.autodoc import (  # noqa: E402
    ALLENAI_POOL, CANONICAL_POOL, SLIDES, fmt, md_table, replace_block)
from multilingual.run_apertus_snr_variants import variant_key

ROOT_OUT = PLOT_DIR / "allenai_comparison"
SNR_DEFINITION_ROOT = PLOT_DIR / "snr_definition"
ALLENAI_CSV = ROOT_OUT / "allenai_snr_variants_per_task.csv"

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


def _load_apertus_with_alias(apertus_csv: Path) -> pd.DataFrame:
    """Apply the canonicalisation, then drop duplicate canonical names by
    keeping the row with the most non-NaN values (i.e., the alias wins
    over the empty vanilla row whenever both exist)."""
    df = pd.read_csv(apertus_csv, index_col="task")
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


# --- auto-generated README (RQ3) -------------------------------------------
# Rewrites the marker-delimited "Highlighted result" / "Results" blocks of
# results/allenai_comparison/README.md. Fires only for the AllenAI canonical
# pool — the pure 3-seed pool, the like-for-like cross-corpus comparison
# (externals shift the shared-task SNR). The canonical-pool invocation runs
# last in the pipeline, so all four pools' shared_task_agreement.csv exist.
# RQ / setup / TODO prose lives outside the markers.

_POOL_TIERS = [
    ("seeds_1904", "1 seed"),
    ("seeds_28_1797", "2 seeds"),
    ("seeds_28_1797_1904", "3 seeds"),
    ("custom_swissai_hf", "+ externals"),
]


def _read_agreement(stage: str, pool: str):
    p = ROOT_OUT / stage / pool / "shared_task_agreement.csv"
    return pd.read_csv(p).iloc[0] if p.exists() else None


def _readme_blocks(stage: str, pool: str) -> tuple[str, str]:
    rows = {p: _read_agreement(stage, p) for p, _ in _POOL_TIERS}
    g = rows[ALLENAI_POOL]                       # canonical (pure 3-seed)
    ext = rows["custom_swissai_hf"]              # gated externals pool
    pure_r = " → ".join(
        fmt(rows[p]["pearson_log_snr"], 2)
        for p in ("seeds_1904", "seeds_28_1797", "seeds_28_1797_1904"))

    highlight = "\n".join([
        f"- **On the pure 3-seed pool (`{ALLENAI_POOL}`) both SNR values and rank order "
        f"agree across corpora** — best variant `{g.variant}`, Pearson r of log₁₀(SNR) "
        f"**{fmt(g.pearson_log_snr)}**, Spearman ρ of the rank order **{fmt(g.spearman_rank)}** "
        f"over the {int(g.n_shared)} shared English tasks.",
        f"- **The value correlation rises with seeds** — Pearson r {pure_r} "
        f"(1 → 2 → 3 seeds): more seeds tighten the cross-corpus SNR fit.",
        f"- **Dispersion + discrepancy families transfer; relative-spread does not** — "
        f"the cross-corpus winners are discrepancy/dispersion variants "
        f"(`{rows['seeds_1904'].variant}`, `{rows['seeds_28_1797'].variant}`, `{g.variant}`), "
        f"not the mean-normalised relative-spread family (incl. AllenAI's own `rel_std`).",
        f"- **Only 7 English tasks overlap, so the *correlation* is the result, not top-K "
        f"Jaccard** (any K ≥ 7 spans the whole universe → Jaccard ≡ 1.0). On "
        f"`custom_swissai_hf` the above-random gate shrinks the shared set to "
        f"n_shared = **{int(ext.n_shared)}**, so use the pure pool for the like-for-like fit.",
    ])

    rs = []
    for p, lab in _POOL_TIERS:
        r = rows[p]
        if r is None:
            continue
        rs.append([f"`{p}` ({lab})", f"`{r.variant}`", fmt(r.pearson_log_snr),
                   fmt(r.spearman_rank), int(r.n_shared)])
    t_pools = md_table(
        ["pool", "best variant", "Pearson r", "Spearman ρ", "n_shared"], rs)

    results = "\n\n".join([
        f"Cross-corpus agreement by pool (headline = the pure 3-seed pool "
        f"`{ALLENAI_POOL}`). Regenerate with "
        f"`python results/allenai_comparison/analyze.py --pool {CANONICAL_POOL}`.",
        "**Cross-corpus agreement over the shared English tasks** — Pearson r of "
        "log₁₀(SNR) (values) and Spearman ρ (rank), each pool's best cross-corpus "
        "variant. The pure pools share all 7 tasks; `custom_swissai_hf` shares fewer "
        "after the above-random gate, so it is indicative, not comparable:",
        t_pools,
        f"![Apertus vs AllenAI SNR — 3-seed pool, best variant]"
        f"(pretraining/{ALLENAI_POOL}/snr_apertus_vs_snr_allenai_{g.variant}.png)",
        f"![Apertus vs AllenAI SNR across variants]"
        f"(pretraining/{ALLENAI_POOL}/snr_apertus_vs_snr_allenai_grid.png)",
    ])
    return highlight, results


def generate_readme(stage: str, pool: str) -> None:
    """Rewrite the auto blocks of results/allenai_comparison/README.md.

    Fires on the LAST pipeline tier (``CANONICAL_POOL``) so every pool's
    ``shared_task_agreement.csv`` already exists for the by-pool table; the
    README content still features ``ALLENAI_POOL`` (the pure 3-seed pool) as the
    like-for-like cross-corpus headline."""
    if pool != CANONICAL_POOL:
        return
    highlight, results = _readme_blocks(stage, pool)
    readme = ROOT_OUT / "README.md"
    gen = f"analyze.py --pool {pool}"
    replace_block(readme, "highlight", "## Highlighted result\n\n" + highlight, gen)
    replace_block(readme, "results", "## Results\n\n" + results, gen)
    print(f"Wrote auto README blocks → {readme}")


def generate_slides(stage: str, pool: str) -> None:
    """Rewrite the RQ2 auto results slide (fires on the last tier; all pools'
    agreement CSVs then exist)."""
    if pool != CANONICAL_POOL:
        return
    rows = []
    for p, lab in _POOL_TIERS:
        r = _read_agreement(stage, p)
        if r is None:
            continue
        rows.append([f"`{p}`", f"`{r.variant}`", fmt(r.pearson_log_snr),
                     fmt(r.spearman_rank), int(r.n_shared)])
    slide = (
        "---\n"
        "title: RQ2 — Framework Generalization\n"
        "subtitle: \"Results (auto) — cross-corpus agreement with AllenAI by pool\"\n"
        "---\n\n"
        f"{md_table(['pool', 'best variant', 'Pearson r', 'Spearman ρ', 'n_shared'], rows)}\n\n"
        "Pure pools share all 7 English tasks; `custom_swissai_hf` shares fewer "
        "after the above-random gate — the pure 3-seed pool is the like-for-like fit.\n\n"
        "<style>\n.slidev-layout table { font-size: 0.7em; }\n</style>"
    )
    replace_block(SLIDES, "rq2-results", slide, "allenai_comparison/analyze.py")
    print(f"Wrote RQ2 results slide → {SLIDES}")


# --- driver ----------------------------------------------------------------

def run(stage: str, pool: str, apertus_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    apertus_csv = apertus_dir / "snr_variants_per_task.csv"
    print(f"Apertus pool : {apertus_dir.name}")
    print(f"Apertus CSV  : {apertus_csv}")
    print(f"AllenAI CSV  : {ALLENAI_CSV}")
    print(f"Output dir   : {out_dir}")

    ap_df = _load_apertus_with_alias(apertus_csv)
    al_df = pd.read_csv(ALLENAI_CSV, index_col="task")

    # Task overlap.
    overlap = _build_task_overlap(set(ap_df.index), set(al_df.index))
    overlap.to_csv(out_dir / "task_overlap.csv")
    shared = overlap[overlap["shared"]].index.tolist()
    print(
        f"Apertus: {len(ap_df)} tasks  |  AllenAI: {len(al_df)} tasks  "
        f"|  shared: {len(shared)}"
    )

    # Per-variant Pearson r at the headline (Apertus@1B ↔ AllenAI@1B).
    summary, per_variant_xy = _per_variant_pearson(ap_df, al_df, shared)
    summary = summary.sort_values("r", ascending=False)
    summary.to_csv(out_dir / "pearson_r_per_variant.csv")
    print("\nPer-variant Pearson r (top 5 / bottom 5):")
    print(summary.head(5).round(3))
    print("...")
    print(summary.tail(5).round(3))

    # Long-format size sweep: same r but at every matched-size pair.
    sweep = _pearson_size_sweep(ap_df, al_df, shared)
    sweep.to_csv(out_dir / "pearson_r_size_sweep.csv", index=False)

    # Headline scatter on the best variant.
    best_variant = summary["r"].idxmax()
    best_r = summary.loc[best_variant, "r"]
    best_n = int(summary.loc[best_variant, "n"])
    headline_path = out_dir / f"snr_apertus_vs_snr_allenai_{best_variant}.png"
    _plot_scatter(per_variant_xy[best_variant], best_variant, best_r, best_n, headline_path)
    print(f"\nHeadline scatter ({best_variant}, r={best_r:.3f}) → {headline_path.name}")

    # Per-variant grid.
    grid_path = out_dir / "snr_apertus_vs_snr_allenai_grid.png"
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

    # The shared universe is small (e.g. 7 English tasks). Set-overlap (top-K
    # Jaccard) is only meaningful for K < universe size — any K ≥ N returns the
    # whole universe on both sides → Jaccard ≡ 1.0 by construction. So the
    # headline agreement is the CORRELATION over the shared tasks (values:
    # Pearson on log10 SNR; ranking: Spearman ρ), and top-K is reported only at
    # non-trivial K.
    n_shared = len(shared)
    ap_vec = pd.to_numeric(ap_shared[snr_col_ap], errors="coerce")
    al_vec = pd.to_numeric(al_shared[snr_col_al], errors="coerce")
    xy = pd.concat([ap_vec, al_vec], axis=1, keys=["ap", "al"]).dropna()
    xy = xy[(xy["ap"] > 0) & (xy["al"] > 0)]
    rank_rho = float(spearmanr(xy["ap"], xy["al"]).correlation) if len(xy) >= 3 else float("nan")
    val_r = float(pearsonr(np.log10(xy["ap"]), np.log10(xy["al"]))[0]) if len(xy) >= 3 else float("nan")
    print(f"\nShared-task agreement ({best_variant}, n={len(xy)}): "
          f"Pearson(log SNR)={val_r:+.3f}  Spearman ρ={rank_rho:+.3f}")

    # Per-corpus full ranking over the shared set (all N tasks).
    top_ap_full = _top_k_table(ap_shared, snr_col_ap, n_shared)
    top_al_full = _top_k_table(al_shared, snr_col_al, n_shared)
    top_ap_full.to_csv(out_dir / "top_apertus.csv")
    top_al_full.to_csv(out_dir / "top_allenai.csv")

    # Only non-trivial K (K < universe); fall back to N-1 if all of 5/10/20 are ≥ N.
    K_LIST = [k for k in (5, 10, 20) if k < n_shared] or [max(2, n_shared - 1)]
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
        "## Cross-corpus agreement over the shared tasks (the result)",
        "",
        f"Best variant `{best_variant}`, n = {len(xy)} shared tasks:",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| **Pearson r** (log₁₀ SNR values) | **{val_r:+.3f}** |",
        f"| **Spearman ρ** (rank order) | **{rank_rho:+.3f}** |",
        "",
        f"> With only {n_shared} shared tasks, **top-K set overlap is NOT a "
        f"result** — any K ≥ {n_shared} spans the whole universe, so Jaccard is "
        f"trivially 1.0. Only K < {n_shared} is reported below.",
        "",
        "## Top-K agreement (non-trivial K only)",
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
        "## Full ranking per corpus (all shared tasks)",
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
    (out_dir / "agreement.md").write_text("\n".join(md_lines))
    agreement_df.to_csv(out_dir / "agreement.csv", index=False)
    pd.DataFrame([{
        "variant": best_variant, "n_shared": len(xy),
        "pearson_log_snr": val_r, "spearman_rank": rank_rho,
    }]).to_csv(out_dir / "shared_task_agreement.csv", index=False)
    print(f"Wrote → {out_dir / 'agreement.csv'}")
    print("\nAgreement table:")
    print(agreement_df.to_string(index=False))

    # Auto-refresh the README "Highlighted result" / "Results" blocks + RQ2
    # results slide (fires on the last tier — no-op otherwise).
    generate_readme(stage, pool)
    generate_slides(stage, pool)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True,
                   help="Apertus pool name (from configs/models.json). "
                        "Reads results/snr_definition/<pool>/, writes "
                        "results/allenai_comparison/<pool>/.")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; "
                f"available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    run(stage=stage, pool=args.pool,
        apertus_dir=SNR_DEFINITION_ROOT / stage / args.pool,
        out_dir=ROOT_OUT / stage / args.pool)


if __name__ == "__main__":
    main()
