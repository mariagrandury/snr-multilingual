"""Render SNR-vs-decision-accuracy plots from snr_variants_per_task.csv.

Per DA definition (and per language) we render one grid:
  rows = SNR variants (ordered by mean Pearson r across cols)
  cols = the three DA pairs that define the DA family

  DA-size (3 cols): SNR(<small>) vs DA(<small>@last → 1B@last) for
                    small ∈ {175M, 350M, 600M}.
  DA-ckpt (3 cols): SNR(size) vs DA(<size>@<early> → <size>@max) for
                    early ∈ {6000, 18000, 28000}, pooling all 4 sizes
                    into one panel (color = size).

Outputs:
  results/snr_definition/da_size/snr_vs_decision_accuracy.png
  results/snr_definition/da_size/snr_vs_decision_accuracy_<lang>.png
  results/snr_definition/da_size/heatmap_pearson_r.png
  results/snr_definition/da_size/variant_correlation_matrix.png
  (parallel set under da_ckpt/)
  results/snr_definition/da_size_vs_da_ckpt.png

The CSV at results/snr_definition/snr_variants_per_task.csv stays the
single source of truth.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from snr.constants import PLOT_DIR
from snr.plot import config_snr_ax

OUT_DIR = PLOT_DIR / "snr_definition"
CSV_PATH = OUT_DIR / "snr_variants_per_task.csv"

SMALL_SIZES = ["175M", "350M", "600M"]
TARGET_SIZE = "1B"
ALL_SIZES = SMALL_SIZES + [TARGET_SIZE]
CKPT_DA_EARLY_STEPS = [6000, 18000, 28000]

SIZE_PALETTE = {
    "175M": "#1f77b4",
    "350M": "#ff7f0e",
    "600M": "#2ca02c",
    "1B":   "#9467bd",
}

# Minimum number of pooled (task, size) points to draw a regression line.
_MIN_FIT_POINTS = 5


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
    """Variants are the unique tokens between ``snr_`` and the final ``_<size>``."""
    variants = set()
    pat = re.compile(r"^snr_(.+)_([0-9]+[MB])$")
    for col in df.columns:
        m = pat.match(col)
        if m:
            variants.add(m.group(1))
    return sorted(variants)


def stat_col(stat: str, variant: str, size: str) -> str:
    return f"{stat}_{variant}_{size}"


def da_size_col(size: str) -> str:
    return f"decision_acc_size_{size}"


def da_ckpt_col(early_step: int, size: str) -> str:
    return f"decision_acc_ckpt_{early_step}_{size}"


# --- column iterator per DA definition --------------------------------------

def da_size_pairs():
    """Yield (col_label, snr_sizes, da_col) per DA-size col.

    For DA-size each column has exactly one (small) size and we plot its
    SNR against its own DA — no pooling.
    """
    for s in SMALL_SIZES:
        yield (f"{s} → {TARGET_SIZE}", [s], da_size_col(s))


def da_ckpt_pairs(sizes: list[str] = None):
    """Yield (col_label, snr_sizes, da_col_for_size) per DA-ckpt col.

    Pass ``sizes=[one_size]`` to restrict each panel to a single model
    size (used by the ``da_ckpt/da_ckpt_<size>/`` subfolders); pass the
    default ``ALL_SIZES`` for the cross-size pooled view.
    """
    if sizes is None:
        sizes = ALL_SIZES
    for early in CKPT_DA_EARLY_STEPS:
        def _da_col(size, early=early):
            return da_ckpt_col(early, size)
        yield (f"ckpt {early} → max", list(sizes), _da_col)


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
        for size in ALL_SIZES:
            idx = [i for i, s in enumerate(data["size"]) if s == size]
            if not idx:
                continue
            xs = np.asarray(data["x"])[idx]
            ys = np.asarray(data["y"])[idx]
            ax.scatter(xs, ys, alpha=0.7, s=12, label=size, color=SIZE_PALETTE[size])
    else:
        # Single size per panel — color it by that size for visual continuity.
        size = data["size"][0] if data["size"] else None
        ax.scatter(data["x"], data["y"], alpha=0.7, s=12,
                   color=SIZE_PALETTE.get(size))
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
    for v in variants:
        all_x = []
        for s in ALL_SIZES:
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
    da_size = list(da_size_pairs())
    da_ckpt = list(da_ckpt_pairs())
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

# (subdir, da_kind, da_pairs_factory, color_by_size, label)
# DA-ckpt has one cross-size pooled view (``da_ckpt/da_ckpt_mix``, color=size)
# plus one mono-color view per model size. The per-size views remove the
# cross-size confound; the mix view is kept for cross-size comparison.
_DA_DEFS: list[tuple] = [
    ("da_size", "size", lambda: list(da_size_pairs()), False, "all sizes"),
    ("da_ckpt/da_ckpt_mix", "ckpt", lambda: list(da_ckpt_pairs()), True,
     "all sizes"),
]
for _s in ALL_SIZES:
    _DA_DEFS.append((
        f"da_ckpt/da_ckpt_{_s}", "ckpt",
        (lambda s=_s: list(da_ckpt_pairs([s]))),
        False, _s,
    ))


def _render_for_da(df: pd.DataFrame, variants: list[str], subdir: str,
                   da_kind: str, pairs_factory, color_by_size: bool,
                   label: str):
    out_dir = OUT_DIR / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    da_pairs = list(pairs_factory())

    ranked = rank_variants(df, variants, da_pairs)
    title = (f"SNR vs decision accuracy (DA-{da_kind}, {label}) — all benchmarks "
             f"(variants ordered by mean Pearson r)")
    if render_grid(df, ranked, da_pairs,
                   out_dir / "snr_vs_decision_accuracy.png", title,
                   color_by_size=color_by_size):
        print(f"Wrote → {out_dir / 'snr_vs_decision_accuracy.png'}")

    # Heatmap of (variant × language) Pearson r.
    table = _per_language_pearson_table(df, variants, da_pairs)
    if _draw_heatmap(table, out_dir / "heatmap_pearson_r.png",
                     title=f"Pearson r — log10(SNR) vs DA-{da_kind} ({label}, per language)"):
        print(f"Wrote → {out_dir / 'heatmap_pearson_r.png'}")

    # Per-language grids.
    df_lang = df.copy()
    df_lang["language"] = [assign_language(t) for t in df_lang.index]
    for lang, sub in sorted(df_lang.groupby("language")):
        sub = sub.drop(columns=["language"])
        if _max_valid_da_per_pair(sub, da_pairs) < _MIN_FIT_POINTS:
            print(f"  skip {lang}: too few valid DA-{da_kind} points "
                  f"(need ≥{_MIN_FIT_POINTS}).")
            continue
        ranked_lang = rank_variants(sub, variants, da_pairs)
        path = out_dir / f"snr_vs_decision_accuracy_{lang}.png"
        title_l = (f"SNR vs decision accuracy (DA-{da_kind}, {label}) — {lang} "
                   f"({len(sub)} benchmarks; variants ordered by mean Pearson r)")
        if render_grid(sub, ranked_lang, da_pairs, path, title_l,
                       color_by_size=color_by_size):
            print(f"Wrote → {path}")


def main():
    df = pd.read_csv(CSV_PATH, index_col="task")
    variants = list_variants(df)
    n_size = sum(1 for c in df.columns if c.startswith("decision_acc_size_"))
    n_ckpt = sum(1 for c in df.columns if c.startswith("decision_acc_ckpt_"))
    print(f"Loaded {len(df)} tasks × {df.shape[1]} columns "
          f"({len(variants)} variants × {len(ALL_SIZES)} sizes × 3 stats "
          f"+ {n_size} size-DA + {n_ckpt} ckpt-DA)\n")

    for subdir, da_kind, pairs_factory, color_by_size, label in _DA_DEFS:
        print(f"=== DA-{da_kind} ({label}) → {OUT_DIR / subdir} ===")
        _render_for_da(df, variants, subdir, da_kind, pairs_factory,
                       color_by_size, label)
        print()

    # Variant correlation matrix (pool over all sizes/tasks).
    corr = _variant_corr_matrix(df, variants)
    if _draw_corr_matrix(corr, OUT_DIR / "variant_correlation_matrix.png",
                         title="Inter-variant correlation of log10(SNR) "
                               "(pooled over tasks × sizes)"):
        print(f"Wrote → {OUT_DIR / 'variant_correlation_matrix.png'}")

    # Per-variant scatter: r(SNR, DA-size) vs r(SNR, DA-ckpt).
    if _draw_da_size_vs_da_ckpt(df, variants,
                                OUT_DIR / "da_size_vs_da_ckpt.png"):
        print(f"Wrote → {OUT_DIR / 'da_size_vs_da_ckpt.png'}")


if __name__ == "__main__":
    main()
