"""Render SNR-vs-decision-accuracy results from snr_variants_per_task.csv.

The full per-variant Pearson r (log10 SNR vs DA) is written to a CSV — the
ranking for every variant, every DA definition, both overall and per
language — so nothing is lost. Scatter grids are then rendered only for the
TOP_N variants (by mean Pearson r) per DA definition, to keep the PNG count
small.

Per DA definition we rank SNR variants by mean Pearson r across cols:
  DA-size (3 cols): SNR(<small>) vs DA(<small>@last → 1B@last) for
                    small ∈ {175M, 350M, 600M}.
  DA-ckpt (3 cols): SNR(size) vs DA(<size>@<early> → <size>@max) for
                    early ∈ {6000, 18000, 28000}, pooling all 4 sizes
                    into one panel (color = size).

Outputs (per pool, under results/snr_definition/<pool>/):
  snr_variant_ranking.csv                     — all variants × DA-defs × scope
  <da_def>/snr_vs_decision_accuracy.png       — top-3 variants only
  <da_def>/heatmap_pearson_r.png              — all variants × language
  variant_correlation_matrix.png
  da_size_vs_da_ckpt.png

The CSV at results/snr_definition/<pool>/snr_variants_per_task.csv stays the
single source of truth for the raw per-task SNR/DA values.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from evals.scripts.utils.configs import (  # noqa: E402
    bucket_order,
    load_pools,
    load_snr_params,
)

from snr.constants import PLOT_DIR
from snr.plot import config_snr_ax

OUT_ROOT = PLOT_DIR / "snr_definition"

# SNR analysis params — single source of truth in configs/models.json.
# The size axis is the *bucket* (size_bucket); the actual buckets/targets to
# render are read from the per-task CSV columns, so this script adapts to
# whatever run_apertus_snr_variants produced (incl. >1B buckets + scaling-DA).
_SNR = load_snr_params()
SMALL_SIZES = _SNR["small_sizes"]
TARGET_SIZE = _SNR["target_size"]
_BUCKETS = bucket_order()
# Longest-first alternation so "12-14B" matches before "1B" etc.
_BUCKET_RE = "|".join(sorted((re.escape(b) for b in _BUCKETS), key=len, reverse=True))


def _bucket_color(bucket: str):
    """Stable color per bucket via the configured bucket order."""
    idx = _BUCKETS.index(bucket) if bucket in _BUCKETS else 0
    return plt.cm.viridis(idx / max(1, len(_BUCKETS) - 1))


# Minimum number of pooled (task, size) points to draw a regression line.
_MIN_FIT_POINTS = 5

# Only render scatter grids for the top-N variants (by mean Pearson r) per DA
# definition. The full ranking for every variant lands in snr_variant_ranking.csv.
TOP_N = 3


# --- language assignment -----------------------------------------------------

_LANG_MAP = {
    "ar": "ar", "arb": "ar",
    "de": "de",
    "es": "es", "spa": "es",
    "eu": "eu", "eus": "eu",
    "fr": "fr",
    "hi": "hi", "hin": "hi",
    "ru": "ru", "rus": "ru",
    "vi": "vi", "vie": "vi",
    "zh": "zh", "zho": "zh", "cmn": "zh",
    "ja": "ja", "jp": "ja", "jpn": "ja",
    "sw": "sw", "swh": "sw",
    "th": "th", "tha": "th",
    "tr": "tr", "tur": "tr",
    "en": "en", "eng": "en",
}

_ENGLISH_ONLY_TASKS = {
    "arc_challenge", "arc_easy", "commonsense_qa", "hellaswag", "mmlu",
    "openbookqa", "piqa", "truthfulqa_mc1",
}

# Tasks that should be merged into a single benchmark family even though
# their names don't share a prefix-up-to-language-token. Keep small and
# explicit; only ARC's challenge/easy split matches this pattern in the
# Apertus task list.
_BENCHMARK_FAMILY_OVERRIDES = {
    "arc_challenge": "arc",
    "arc_easy": "arc",
}


def assign_language(task: str) -> str:
    if task in _ENGLISH_ONLY_TASKS:
        return "en"
    for tok in task.split("_"):
        if tok in _LANG_MAP:
            return _LANG_MAP[tok]
    return "??"


def benchmark_family(task: str) -> str:
    """Strip any language/script suffix, leaving the benchmark identifier.

    Two explicit overrides via ``_BENCHMARK_FAMILY_OVERRIDES``:
      - ``arc_challenge`` and ``arc_easy`` collapse to ``arc`` so they
        end up in the same per-benchmark grid as ``arc_de``/``arc_es``/…
      - English ``truthfulqa_mc1`` is left alone so it does not collapse
        with the multilingual ``truthfulqa_<lang>_mc1`` variants (which
        are Spanish/Russian/etc. — they belong in their own family).
    """
    if task in _BENCHMARK_FAMILY_OVERRIDES:
        return _BENCHMARK_FAMILY_OVERRIDES[task]
    parts = task.split("_")
    out = []
    for p in parts:
        if p in _LANG_MAP:
            break
        out.append(p)
    return "_".join(out) if out else parts[0]


# --- column helpers ---------------------------------------------------------

def list_variants(df: pd.DataFrame) -> list[str]:
    """Variants are the unique tokens between ``snr_`` and the final
    ``_<bucket>`` (bucket labels can contain hyphens/dots, e.g. ``7-9B``)."""
    variants = set()
    pat = re.compile(rf"^snr_(.+?)_({_BUCKET_RE})$")
    for col in df.columns:
        m = pat.match(col)
        if m:
            variants.add(m.group(1))
    return sorted(variants)


def buckets_in_df(df: pd.DataFrame) -> list[str]:
    """Size buckets present in the CSV's ``snr_*`` columns, in size order."""
    pat = re.compile(rf"^snr_.+_({_BUCKET_RE})$")
    found = {m.group(1) for c in df.columns if (m := pat.match(c))}
    return [b for b in _BUCKETS if b in found]


def stat_col(stat: str, variant: str, size: str) -> str:
    return f"{stat}_{variant}_{size}"


# --- column iterator per DA definition --------------------------------------

def da_size_pairs(df: pd.DataFrame):
    """Yield (col_label, snr_buckets, da_col) per DA-size col found in the CSV.

    Canonical columns ``decision_acc_size_<bucket>`` are small→1B; scaling
    columns ``decision_acc_size_<small>_to_<target>`` carry their own target.
    Each pair plots SNR(small bucket) against that DA column — no pooling.
    """
    can = re.compile(rf"^decision_acc_size_({_BUCKET_RE})$")
    sca = re.compile(rf"^decision_acc_size_({_BUCKET_RE})_to_({_BUCKET_RE})$")
    canon, scaling = [], []
    for col in df.columns:
        if (m := sca.match(col)):
            scaling.append((f"{m.group(1)} → {m.group(2)}", [m.group(1)], col))
        elif (m := can.match(col)):
            canon.append((f"{m.group(1)} → {TARGET_SIZE}", [m.group(1)], col))
    canon.sort(key=lambda t: _BUCKETS.index(t[1][0]) if t[1][0] in _BUCKETS else 0)
    return canon + scaling


def da_ckpt_pairs(df: pd.DataFrame, sizes: list[str] = None):
    """Yield (col_label, snr_buckets, da_col_for_bucket) per ckpt-DA fraction
    found in the CSV. Pass ``sizes=[one_bucket]`` to restrict each panel to a
    single bucket (the ``da_ckpt/da_ckpt_<bucket>/`` subfolders); default pools
    every bucket present (cross-size view)."""
    pat = re.compile(rf"^decision_acc_ckpt_(f\d+)_({_BUCKET_RE})$")
    fracs, buckets = set(), set()
    for col in df.columns:
        if (m := pat.match(col)):
            fracs.add(m.group(1))
            buckets.add(m.group(2))
    frac_list = sorted(fracs, key=lambda f: int(f[1:]))
    all_b = [b for b in _BUCKETS if b in buckets]
    use = sizes if sizes is not None else all_b
    out = []
    for fl in frac_list:
        def _da_col(bucket, fl=fl):
            return f"decision_acc_ckpt_{fl}_{bucket}"
        out.append((f"ckpt {fl} → max", list(use), _da_col))
    return out


# --- data extraction --------------------------------------------------------

def _gather_points(df: pd.DataFrame, stat: str, variant: str,
                   snr_sizes: list[str], da_col_fn, log_x: bool):
    """For each size in ``snr_sizes`` collect (x, y, size). Skips rows where
    either coordinate is NaN (or x ≤ 0 when log_x). ``da_col_fn`` returns
    the DA column name for a given size (a string column for DA-size, or a
    different per-size column for DA-ckpt)."""
    data = {"x": [], "y": [], "size": []}
    for size in snr_sizes:
        x_c = stat_col(stat, variant, size)
        y_c = da_col_fn(size) if callable(da_col_fn) else da_col_fn
        if x_c not in df.columns or y_c not in df.columns:
            continue
        sub = df[[x_c, y_c]].dropna()
        if log_x:
            sub = sub[sub[x_c] > 0]
        if sub.empty:
            continue
        data["x"].extend(sub[x_c].to_numpy())
        data["y"].extend(sub[y_c].to_numpy())
        data["size"].extend([size] * len(sub))
    return data


def _pearson_r(xs, ys, log_x):
    if len(xs) < 3:
        return float("nan")
    x = np.log10(xs) if log_x else np.asarray(xs)
    y = np.asarray(ys)
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


# --- ranking ---------------------------------------------------------------

def variant_col_rs(df: pd.DataFrame, variant: str, da_pairs: list) -> list[float]:
    """Per-column Pearson r between log10(SNR) and DA, for one variant."""
    out = []
    for _label, sizes, da_fn in da_pairs:
        d = _gather_points(df, "snr", variant, sizes, da_fn, log_x=True)
        out.append(_pearson_r(d["x"], d["y"], log_x=True))
    return out


def rank_variants(df: pd.DataFrame, variants: list[str], da_pairs: list
                  ) -> list[tuple[str, list[float], float]]:
    """Order variants by mean Pearson r across cols (NaN ignored). Returns
    [(variant, per_col_rs, mean_r)]."""
    rows = []
    for v in variants:
        rs = variant_col_rs(df, v, da_pairs)
        finite = [r for r in rs if np.isfinite(r)]
        mean_r = float(np.mean(finite)) if finite else float("nan")
        rows.append((v, rs, mean_r))
    rows.sort(key=lambda t: -(t[2] if np.isfinite(t[2]) else -np.inf))
    return rows


# --- plotting --------------------------------------------------------------

def _scatter_panel(ax, data: dict, log_x: bool, plot_fit: bool, color_by_size: bool):
    if not data["x"]:
        ax.set_visible(False)
        return 0
    if color_by_size:
        for size in [b for b in _BUCKETS if b in set(data["size"])]:
            idx = [i for i, s in enumerate(data["size"]) if s == size]
            if not idx:
                continue
            xs = np.asarray(data["x"])[idx]
            ys = np.asarray(data["y"])[idx]
            ax.scatter(xs, ys, alpha=0.7, s=12, label=size, color=_bucket_color(size))
    else:
        # Single bucket per panel — color it by that bucket for continuity.
        size = data["size"][0] if data["size"] else None
        ax.scatter(data["x"], data["y"], alpha=0.7, s=12,
                   color=_bucket_color(size) if size else None)
    n = len(data["x"])
    config_snr_ax(
        ax, np.asarray(data["x"]), np.asarray(data["y"]), texts=[],
        xlabel="", plot_fit=plot_fit and n >= _MIN_FIT_POINTS, log_scale=log_x,
    )
    return n


def render_grid(df: pd.DataFrame, variants_ranked: list,
                da_pairs: list, save_path: Path, title: str,
                color_by_size: bool) -> bool:
    """Rows = variants, cols = DA pairs. Always log-x for SNR panels."""
    n_rows = len(variants_ranked)
    n_cols = len(da_pairs)
    if n_rows == 0:
        return False
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.5 * n_cols, 4 * n_rows), squeeze=False,
    )
    drawn = 0
    for r, (variant, rs, _mean) in enumerate(variants_ranked):
        for c, (col_label, sizes, da_fn) in enumerate(da_pairs):
            ax = axes[r][c]
            data = _gather_points(df, "snr", variant, sizes, da_fn, log_x=True)
            n = _scatter_panel(ax, data, log_x=True, plot_fit=True,
                               color_by_size=color_by_size)
            if n:
                drawn += 1
            r_text = f"  r={rs[c]:+.3f}" if np.isfinite(rs[c]) else ""
            ax.set_title(f"{variant} — {col_label}{r_text}  (n={n})",
                         fontsize=10)
            if c == n_cols - 1 and color_by_size and ax.get_visible():
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    seen = {}
                    for h, l in zip(handles, labels):
                        seen.setdefault(l, h)
                    ax.legend(seen.values(), seen.keys(), title="Size",
                              fontsize=8, title_fontsize=9, loc="lower right")
    if drawn == 0:
        plt.close(fig)
        return False
    # Reserve a fixed strip at the top of the figure for the suptitle so
    # that with tall figures (e.g. 22 rows × 4 inches) it doesn't end up
    # inside row 1 — the default y=0.98 is a fraction of figure height,
    # not a pixel offset.
    fig_h = fig.get_size_inches()[1]
    title_strip_in = 0.6
    title_y = 1 - 0.2 / fig_h  # baseline near the top edge
    fig.tight_layout(rect=(0, 0, 1, 1 - title_strip_in / fig_h))
    fig.suptitle(title, fontsize=14, y=title_y, va="top")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return True


# --- per-language gating ----------------------------------------------------

def _max_valid_da_per_pair(df: pd.DataFrame, da_pairs: list) -> int:
    """Largest count of non-NaN DA values pooled across each pair's sizes.

    For pooled cross-size views (``len(sizes) > 1``) the panel data are
    the *union* across sizes, so summing per pair (then taking the max
    across pairs) is the right gate; using the per-cell max would skip
    languages with enough total data spread across sizes.
    """
    best = 0
    for _, sizes, da_fn in da_pairs:
        total = 0
        for s in sizes:
            col = da_fn(s) if callable(da_fn) else da_fn
            if col in df.columns:
                total += int(df[col].notna().sum())
        best = max(best, total)
    return best


# --- heatmap visualizations ------------------------------------------------

def _per_language_pearson_table(df: pd.DataFrame, variants: list[str],
                                da_pairs: list) -> pd.DataFrame:
    """Build a (variant × language) DataFrame of pooled-across-cols Pearson r."""
    df_lang = df.copy()
    df_lang["language"] = [assign_language(t) for t in df_lang.index]
    langs = sorted(df_lang["language"].unique())
    table = pd.DataFrame(index=variants, columns=langs, dtype=float)
    for lang in langs:
        sub = df_lang[df_lang["language"] == lang].drop(columns=["language"])
        if len(sub) < 2:
            continue
        for v in variants:
            xs, ys = [], []
            for _, sizes, da_fn in da_pairs:
                d = _gather_points(sub, "snr", v, sizes, da_fn, log_x=True)
                xs.extend(d["x"])
                ys.extend(d["y"])
            table.loc[v, lang] = _pearson_r(xs, ys, log_x=True)
    return table


def _draw_heatmap(table: pd.DataFrame, save_path: Path, title: str,
                  vmin=-1.0, vmax=1.0, cmap="RdBu_r"):
    if table.empty:
        return False
    table = table.copy()
    # Order rows by mean r (most useful at top).
    table["_mean"] = table.mean(axis=1, skipna=True)
    table = table.sort_values("_mean", ascending=False).drop(columns=["_mean"])
    fig, ax = plt.subplots(figsize=(0.6 * len(table.columns) + 3,
                                    0.32 * len(table.index) + 2))
    arr = table.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(v) > 0.55 else "black")
    fig.colorbar(im, ax=ax, label="Pearson r (log10 SNR vs DA)")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return True


def _variant_corr_matrix(df: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """Pearson correlation between log10(SNR) values across variants, pooled
    over every (task, size) cell. Tells you which variants are
    algebraically redundant."""
    cols = {}
    buckets = buckets_in_df(df)
    for v in variants:
        all_x = []
        for s in buckets:
            c = stat_col("snr", v, s)
            if c not in df.columns:
                continue
            vals = df[c].to_numpy(dtype=float)
            mask = np.isfinite(vals) & (vals > 0)
            all_x.append(np.where(mask, np.log10(vals, where=mask, out=np.full_like(vals, np.nan)), np.nan))
        if all_x:
            cols[v] = np.concatenate(all_x)
    if not cols:
        return pd.DataFrame()
    return pd.DataFrame(cols).corr()


def _draw_corr_matrix(corr: pd.DataFrame, save_path: Path, title: str):
    if corr.empty:
        return False
    fig, ax = plt.subplots(figsize=(0.32 * len(corr.columns) + 3,
                                    0.32 * len(corr.index) + 2))
    arr = corr.to_numpy(dtype=float)
    im = ax.imshow(arr, aspect="auto", cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r between log10(SNR) variants")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return True


def _draw_da_size_vs_da_ckpt(df: pd.DataFrame, variants: list[str],
                             save_path: Path):
    """Per variant: x = mean(r) for DA-size, y = mean(r) for DA-ckpt.
    Above the diagonal: DA-ckpt agrees more with the variant; below: DA-size."""
    da_size = list(da_size_pairs(df))
    da_ckpt = list(da_ckpt_pairs(df))
    rows = []
    for v in variants:
        rs_size = [r for r in variant_col_rs(df, v, da_size) if np.isfinite(r)]
        rs_ckpt = [r for r in variant_col_rs(df, v, da_ckpt) if np.isfinite(r)]
        rows.append({
            "variant": v,
            "r_size": float(np.mean(rs_size)) if rs_size else np.nan,
            "r_ckpt": float(np.mean(rs_ckpt)) if rs_ckpt else np.nan,
        })
    pts = pd.DataFrame(rows).dropna()
    if pts.empty:
        return False
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pts["r_size"], pts["r_ckpt"], color="#1f77b4", s=24)
    for _, r in pts.iterrows():
        ax.annotate(r["variant"], (r["r_size"], r["r_ckpt"]),
                    fontsize=7, alpha=0.8, ha="left", va="bottom")
    lo = min(pts["r_size"].min(), pts["r_ckpt"].min(), -0.05) - 0.02
    hi = max(pts["r_size"].max(), pts["r_ckpt"].max(), 0.05) + 0.02
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=0.8)
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Pearson r — SNR vs DA-size  (mean across cols)")
    ax.set_ylabel("Pearson r — SNR vs DA-ckpt  (mean across cols)")
    ax.set_title("Variant agreement: DA-size vs DA-ckpt")
    ax.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return True


# --- driver ----------------------------------------------------------------

def _da_defs(df: pd.DataFrame) -> list[tuple]:
    """(subdir, da_kind, da_pairs, color_by_size, label), built from the CSV.

    DA-ckpt has one cross-bucket pooled view (``da_ckpt/da_ckpt_mix``,
    color=bucket) plus one mono-color view per bucket present. The per-bucket
    views remove the cross-size confound; the mix view is kept for comparison.
    Buckets with no ckpt-DA column (single-ckpt-only) are skipped."""
    defs = [
        ("da_size", "size", da_size_pairs(df), False, "all sizes"),
        ("da_ckpt/da_ckpt_mix", "ckpt", da_ckpt_pairs(df), True, "all sizes"),
    ]
    for b in buckets_in_df(df):
        pairs = da_ckpt_pairs(df, [b])
        if pairs and any(b in sizes for _, sizes, _ in pairs):
            defs.append((f"da_ckpt/da_ckpt_{b}", "ckpt", pairs, False, b))
    return defs


def _render_for_da(df: pd.DataFrame, variants: list[str], subdir: str,
                   da_kind: str, da_pairs: list, color_by_size: bool,
                   label: str, out_root: Path, csv_rows: list):
    """Rank variants for this DA definition, append the full ranking (overall
    per-column + per-language pooled r) to ``csv_rows``, and render the scatter
    grid for the TOP_N variants only plus the compact per-language heatmap."""
    out_dir = out_root / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    col_labels = [lab for lab, _, _ in da_pairs]

    ranked = rank_variants(df, variants, da_pairs)
    # Persist the overall ranking: one row per (variant, col) plus a mean row.
    for variant, rs, mean_r in ranked:
        for col_label, r in zip(col_labels, rs):
            csv_rows.append({"da_def": subdir, "scope": "all", "variant": variant,
                             "column": col_label, "pearson_r": r})
        csv_rows.append({"da_def": subdir, "scope": "all", "variant": variant,
                         "column": "mean", "pearson_r": mean_r})

    # Per-language pooled r (the heatmap data) — persist every cell.
    table = _per_language_pearson_table(df, variants, da_pairs)
    for variant in table.index:
        for lang in table.columns:
            csv_rows.append({"da_def": subdir, "scope": lang, "variant": variant,
                             "column": "pooled", "pearson_r": table.loc[variant, lang]})

    title = (f"SNR vs decision accuracy (DA-{da_kind}, {label}) — top {TOP_N} "
             f"variants by mean Pearson r")
    if render_grid(df, ranked[:TOP_N], da_pairs,
                   out_dir / "snr_vs_decision_accuracy.png", title,
                   color_by_size=color_by_size):
        print(f"Wrote → {out_dir / 'snr_vs_decision_accuracy.png'}")

    if _draw_heatmap(table, out_dir / "heatmap_pearson_r.png",
                     title=f"Pearson r — log10(SNR) vs DA-{da_kind} ({label}, per language)"):
        print(f"Wrote → {out_dir / 'heatmap_pearson_r.png'}")


def main(out_dir: Path):
    csv_path = out_dir / "snr_variants_per_task.csv"
    df = pd.read_csv(csv_path, index_col="task")
    variants = list_variants(df)
    buckets = buckets_in_df(df)
    n_size = sum(1 for c in df.columns if c.startswith("decision_acc_size_"))
    n_ckpt = sum(1 for c in df.columns if c.startswith("decision_acc_ckpt_"))
    print(f"Loaded {len(df)} tasks × {df.shape[1]} columns from {csv_path} "
          f"({len(variants)} variants × {len(buckets)} buckets × 3 stats "
          f"+ {n_size} size-DA + {n_ckpt} ckpt-DA)")
    print(f"  Buckets: {buckets}\n")

    csv_rows: list = []
    for subdir, da_kind, da_pairs, color_by_size, label in _da_defs(df):
        print(f"=== DA-{da_kind} ({label}) → {out_dir / subdir} ===")
        _render_for_da(df, variants, subdir, da_kind, da_pairs,
                       color_by_size, label, out_root=out_dir, csv_rows=csv_rows)
        print()

    ranking_path = out_dir / "snr_variant_ranking.csv"
    pd.DataFrame(csv_rows).to_csv(ranking_path, index=False)
    print(f"Wrote ranking CSV → {ranking_path} ({len(csv_rows)} rows)\n")

    # Variant correlation matrix (pool over all sizes/tasks).
    corr = _variant_corr_matrix(df, variants)
    if _draw_corr_matrix(corr, out_dir / "variant_correlation_matrix.png",
                         title="Inter-variant correlation of log10(SNR) "
                               "(pooled over tasks × sizes)"):
        print(f"Wrote → {out_dir / 'variant_correlation_matrix.png'}")

    # Per-variant scatter: r(SNR, DA-size) vs r(SNR, DA-ckpt).
    if _draw_da_size_vs_da_ckpt(df, variants,
                                out_dir / "da_size_vs_da_ckpt.png"):
        print(f"Wrote → {out_dir / 'da_size_vs_da_ckpt.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", required=True,
                   help="Pool name from configs/models.json (tiers: seeds_1904, "
                        "seeds_28_1797, seeds_28_1797_1904, custom_swissai_hf). "
                        "Output dir = results/snr_definition/<stage>/<pool>/.")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; "
                f"available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    main(out_dir=PLOT_DIR / "snr_definition" / stage / args.pool)
