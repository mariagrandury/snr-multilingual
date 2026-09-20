"""Test framework generalization across seed splits.

Reads two pools' ``snr_variants_per_task.csv`` (``predictivity_seeds_train``,
the replicate seeds 64/313, as the "train" split and ``predictivity_seeds_test``,
seed 1904 on the same cells, as the "test" split) and asks:

  1. Does the per-language best variant agree between the two splits?
  2. For variants picked on the train split, what is their Pearson r
     on the test split?
  3. Across all languages × variants, how correlated are the per-cell
     Pearson r values between the two splits?

The per-language r is rq04's ``_per_language_pearson_table``: a language needs
MIN_LANG_TASKS (3) distinct tasks with a value, fewer is NaN (rule 8), and
``multi`` / ``??`` are never a language (rule 7, `utils.languages_only`). The
two pools must hold the same cells (size × L × arch × scheme); the script
checks that on the loaded pools and says so in the README block.

Outputs land under ``<rq03>/<stage>/<train_pool>__vs__<test_pool>/``:
  - ``per_language_agreement_da_<size|ckpt>.csv`` — for each language, the
    train-best variant and its r in both splits, plus the test-split's own best.
  - ``per_language_agreement_da_<size|ckpt>.png`` — bar chart of train-vs-test
    r per language under the train-best variant.
  - ``variant_r_train_vs_test.csv`` — long table of (language, variant,
    r_train, r_test) for every (lang, variant) cell.
  - ``variant_r_train_vs_test.png`` — scatter of r_train vs r_test for
    every (lang, variant) cell, coloured by DA flavor.
  - ``top_variants_train_vs_test.csv`` — mean r per variant and split, the
    Spearman ρ between the splits' global rankings.
  - ``summary.md``, ``headline_metrics.csv`` — the headline numbers.

CLI:
  python analysis/rq03_noise_and_snr/compare_seed_splits.py \
      --train-pool predictivity_seeds_train --test-pool predictivity_seeds_test
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

from evals.scripts.utils.configs import load_pools  # noqa: E402
from analysis.autodoc import fmt, replace_block  # noqa: E402
from analysis.rq04_surrogates.analyze_snr_variants import (  # noqa: E402
    _per_language_pearson_table, da_ckpt_pairs, da_size_pairs,
    list_variants, scaling_pairs,
)
from analysis.rq04_surrogates.snr_definition_postprocess import _VARIANT_FAMILY  # noqa: E402
from analysis.paths import NOISE_AND_SNR  # noqa: E402
from analysis.utils import MIN_LANG_TASKS, assign_language, build_snr_pool, languages_only  # noqa: E402

OUT_ROOT = NOISE_AND_SNR

# --- helpers ----------------------------------------------------------------

def _load(out_dir: Path) -> pd.DataFrame:
    csv = out_dir / "snr_variants_per_task.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found — run run_apertus_snr_variants.py for that split first."
        )
    return pd.read_csv(csv, index_col="task")


def _size_pairs(df: pd.DataFrame) -> list:
    """The holdout's DA-size: the reference columns when the pool has a value
    in them, else its `scaling_pairs` (175M → 600M on the ×3 cells, where
    1.7B is absent), named as such wherever the numbers are shown (rule 9)."""
    return [p for p in da_size_pairs(df) if df[p[2]].notna().any()] or scaling_pairs(df)


def _table_for(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """rq04's (variant × language) Pearson table (≥ MIN_LANG_TASKS distinct
    tasks per language, rule 8) on the language rows alone (rule 7)."""
    pairs = _size_pairs(df) if kind == "size" else list(da_ckpt_pairs(df))
    df = languages_only(df.assign(language=df.index.map(assign_language)).copy())
    return _per_language_pearson_table(df, list_variants(df), pairs)


def _best_per_lang(table: pd.DataFrame) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for lang in table.columns:
        col = table[lang].dropna()
        if col.empty:
            continue
        idx = col.idxmax()
        out[lang] = (str(idx), float(col.loc[idx]))
    return out


# --- per-language agreement --------------------------------------------------

def per_language_agreement(df_train: pd.DataFrame, df_test: pd.DataFrame,
                           kind: str) -> pd.DataFrame:
    """For each language, report the train-best variant's r in train and
    test, and the test-best variant for comparison."""
    t_train = _table_for(df_train, kind)
    t_test = _table_for(df_test, kind)
    best_train = _best_per_lang(t_train)
    best_test = _best_per_lang(t_test)

    rows = []
    for lang in sorted(set(t_train.columns) | set(t_test.columns)):
        v_t, r_t_train = best_train.get(lang, ("", np.nan))
        # The train-best variant's r in the test split:
        if v_t and lang in t_test.columns and v_t in t_test.index:
            r_t_test = float(t_test.loc[v_t, lang])
        else:
            r_t_test = np.nan
        v_test, r_test = best_test.get(lang, ("", np.nan))
        fam_t = _VARIANT_FAMILY.get(v_t, "??") if v_t else ""
        fam_test = _VARIANT_FAMILY.get(v_test, "??") if v_test else ""
        rows.append({
            "language": lang,
            "da_kind": kind,
            "train_best_variant": v_t,
            "train_best_family": fam_t,
            "train_best_r_train": r_t_train,
            "train_best_r_test": r_t_test,
            "test_best_variant": v_test,
            "test_best_family": fam_test,
            "test_best_r_test": r_test,
            "same_variant": bool(v_t and v_t == v_test),
            "same_family": bool(fam_t and fam_t != "??" and fam_t == fam_test),
        })
    return pd.DataFrame(rows)


def render_per_language(agreement: pd.DataFrame, save_path: Path,
                        title: str):
    """Bar chart per language: train-best r in train vs in test."""
    df = agreement.dropna(subset=["train_best_r_train", "train_best_r_test"])
    if df.empty:
        return False
    langs = df["language"].tolist()
    x = np.arange(len(langs))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(langs)), 5))
    ax.bar(x - w / 2, df["train_best_r_train"], width=w,
           label="r on train split", color="#1f77b4", alpha=0.85)
    ax.bar(x + w / 2, df["train_best_r_test"], width=w,
           label="r on test split", color="#ff7f0e", alpha=0.85)
    for i, (_, row) in enumerate(df.iterrows()):
        marker = "★" if row["same_variant"] else ""
        ax.text(i, max(row["train_best_r_train"], row["train_best_r_test"]) + 0.02,
                f"{row['train_best_variant']}{marker}",
                ha="center", va="bottom", fontsize=7, rotation=90)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_ylabel("Pearson r (log10 SNR vs DA)")
    ax.set_title(title + "  (★ = test-split also picks this variant)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return True


# --- variant-cell scatter ---------------------------------------------------

def variant_r_long(df_train: pd.DataFrame, df_test: pd.DataFrame,
                   kind: str) -> pd.DataFrame:
    t_train = _table_for(df_train, kind)
    t_test = _table_for(df_test, kind)
    common_vars = sorted(set(t_train.index) & set(t_test.index))
    common_langs = sorted(set(t_train.columns) & set(t_test.columns))
    rows = []
    for v in common_vars:
        for lang in common_langs:
            r_tr = float(t_train.loc[v, lang]) if v in t_train.index else np.nan
            r_te = float(t_test.loc[v, lang]) if v in t_test.index else np.nan
            rows.append({"da_kind": kind, "language": lang, "variant": v,
                         "r_train": r_tr, "r_test": r_te})
    return pd.DataFrame(rows)


def render_scatter(long_df: pd.DataFrame, save_path: Path, title: str):
    df = long_df.dropna(subset=["r_train", "r_test"])
    if df.empty:
        return False
    fig, ax = plt.subplots(figsize=(7, 7))
    palette = {"size": "#1f77b4", "ckpt": "#ff7f0e"}
    for kind, sub in df.groupby("da_kind"):
        ax.scatter(sub["r_train"], sub["r_test"], s=14, alpha=0.6,
                   color=palette.get(kind, "#777"), label=f"DA-{kind}")
    lo = min(df["r_train"].min(), df["r_test"].min()) - 0.05
    hi = max(df["r_train"].max(), df["r_test"].max()) + 0.05
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=0.8)
    ax.axhline(0, color="grey", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="grey", linewidth=0.4, alpha=0.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Pearson r — train split")
    ax.set_ylabel("Pearson r — test split")
    overall_r = float(np.corrcoef(df["r_train"], df["r_test"])[0, 1])
    ax.set_title(f"{title}  (overall r between splits = {overall_r:+.3f}, "
                 f"n = {len(df)})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return True


# --- summary md -------------------------------------------------------------

def write_summary(out_dir: Path, train_dir: Path, test_dir: Path,
                  agreement_size: pd.DataFrame,
                  agreement_ckpt: pd.DataFrame,
                  long_all: pd.DataFrame,
                  rank_corrs: dict | None = None, da_size_label: str = "DA-size"):
    def _agree_frac(df, col):
        if df.empty:
            return float("nan"), 0
        # A language with no data on a split carries "" (not NaN) in these
        # columns; counting it in the denominator understated agreement 10x.
        valid = df.replace({"train_best_variant": {"": None}, "test_best_variant": {"": None}}) \
                  .dropna(subset=["train_best_variant", "test_best_variant"])
        if valid.empty:
            return float("nan"), 0
        return float(valid[col].mean()), len(valid)

    def _block_r(long_df, kind):
        sub = long_df[(long_df["da_kind"] == kind)].dropna(
            subset=["r_train", "r_test"])
        if len(sub) < 3:
            return float("nan"), 0
        return float(np.corrcoef(sub["r_train"], sub["r_test"])[0, 1]), len(sub)

    def _retention(df):
        """Mean fraction r(train-best on test) / r(test-best on test)."""
        sub = df.dropna(subset=["train_best_r_test", "test_best_r_test"])
        sub = sub[(sub["test_best_r_test"] > 0)]
        if sub.empty:
            return float("nan"), 0
        ratios = sub["train_best_r_test"] / sub["test_best_r_test"]
        return float(ratios.clip(lower=0).mean()), len(sub)

    r_size, n_size_cells = _block_r(long_all, "size")
    r_ckpt, n_ckpt_cells = _block_r(long_all, "ckpt")
    a_var_size, n_size = _agree_frac(agreement_size, "same_variant")
    a_var_ckpt, n_ckpt = _agree_frac(agreement_ckpt, "same_variant")
    a_fam_size, _ = _agree_frac(agreement_size, "same_family")
    a_fam_ckpt, _ = _agree_frac(agreement_ckpt, "same_family")
    ret_size, n_ret_size = _retention(agreement_size)
    ret_ckpt, n_ret_ckpt = _retention(agreement_ckpt)
    rank_size = rank_corrs.get("size", float("nan")) if rank_corrs else float("nan")
    rank_ckpt = rank_corrs.get("ckpt", float("nan")) if rank_corrs else float("nan")

    def _cnt(a, n):        # languages agreeing, from the share; 0 when no language qualified (a is NaN)
        return round(a * n) if n else 0

    lines = []
    lines.append(f"# Seed-split generalization: `{train_dir.name}` → "
                 f"`{test_dir.name}`")
    lines.append("")
    lines.append(f"DA-size here is {da_size_label}; DA-ckpt is the within-size early → final ranking. "
                 f"A language's r needs ≥ {MIN_LANG_TASKS} distinct tasks with a value (rule 8).")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("|  | DA-size | DA-ckpt |")
    lines.append("|---|---:|---:|")
    lines.append(f"| Exact-variant agreement (lang-level) | "
                 f"{a_var_size:.0%} ({_cnt(a_var_size, n_size)}/{n_size}) | "
                 f"{a_var_ckpt:.0%} ({_cnt(a_var_ckpt, n_ckpt)}/{n_ckpt}) |")
    lines.append(f"| **Family-level agreement** (lang-level) | "
                 f"{a_fam_size:.0%} ({_cnt(a_fam_size, n_size)}/{n_size}) | "
                 f"{a_fam_ckpt:.0%} ({_cnt(a_fam_ckpt, n_ckpt)}/{n_ckpt}) |")
    lines.append(f"| Pearson r between splits (over all variant cells) | "
                 f"{r_size:+.3f} (n = {n_size_cells}) | "
                 f"{r_ckpt:+.3f} (n = {n_ckpt_cells}) |")
    lines.append(f"| **Spearman ρ on global variant ranking** | "
                 f"{rank_size:+.3f} | {rank_ckpt:+.3f} |")
    lines.append(f"| Retention of train-best (r_test / r_test_best, mean across langs) | "
                 f"{ret_size:.0%} (n = {n_ret_size}) | "
                 f"{ret_ckpt:.0%} (n = {n_ret_ckpt}) |")
    lines.append("")
    lines.append("**Family** groups together algebraically near-equivalent "
                 "variants (e.g. the dispersion cluster: `dispersion`/`mpd`/"
                 "`range`/`quartile_deviation`/`rms_deviation`/`aad`). At "
                 "n_mixes=3, members of a family correlate at r ≥ 0.999 so "
                 "exact-variant equality is overly strict.")
    lines.append("")
    lines.append("**Retention** is how much of the test-split's best r the "
                 "train-picked variant captures on the test split. 100% = "
                 "train-pick is also test-best; lower numbers mean we lose "
                 "predictive correlation by picking on the train split.")
    lines.append("")
    lines.append("## DA-size — per language")
    lines.append("")
    lines.append("| lang | train-best (family) | r (train) | r (test) | "
                 "test-best (family) | r (test) | same variant | same family |")
    lines.append("|---|---|---:|---:|---|---:|:---:|:---:|")
    for _, row in agreement_size.iterrows():
        s_v = "✅" if row["same_variant"] else ""
        s_f = "✅" if row["same_family"] else ""
        lines.append(
            f"| {row['language']} | `{row['train_best_variant']}` "
            f"({row['train_best_family']}) | "
            f"{row['train_best_r_train']:+.3f} | {row['train_best_r_test']:+.3f} | "
            f"`{row['test_best_variant']}` ({row['test_best_family']}) | "
            f"{row['test_best_r_test']:+.3f} | {s_v} | {s_f} |"
        )
    lines.append("")
    lines.append("## DA-ckpt — per language")
    lines.append("")
    lines.append("| lang | train-best (family) | r (train) | r (test) | "
                 "test-best (family) | r (test) | same variant | same family |")
    lines.append("|---|---|---:|---:|---|---:|:---:|:---:|")
    for _, row in agreement_ckpt.iterrows():
        s_v = "✅" if row["same_variant"] else ""
        s_f = "✅" if row["same_family"] else ""
        lines.append(
            f"| {row['language']} | `{row['train_best_variant']}` "
            f"({row['train_best_family']}) | "
            f"{row['train_best_r_train']:+.3f} | {row['train_best_r_test']:+.3f} | "
            f"`{row['test_best_variant']}` ({row['test_best_family']}) | "
            f"{row['test_best_r_test']:+.3f} | {s_v} | {s_f} |"
        )

    out_path = out_dir / "summary.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote → {out_path}")

    # Headline metrics as a machine-readable CSV — same values as the
    # markdown table at the top of summary.md, one row per metric × DA flavor.
    headline = pd.DataFrame([
        {"metric": "exact_variant_agreement",
         "da_kind": "size", "value": a_var_size, "n": n_size},
        {"metric": "exact_variant_agreement",
         "da_kind": "ckpt", "value": a_var_ckpt, "n": n_ckpt},
        {"metric": "family_agreement",
         "da_kind": "size", "value": a_fam_size, "n": n_size},
        {"metric": "family_agreement",
         "da_kind": "ckpt", "value": a_fam_ckpt, "n": n_ckpt},
        {"metric": "pearson_r_cells",
         "da_kind": "size", "value": r_size, "n": n_size_cells},
        {"metric": "pearson_r_cells",
         "da_kind": "ckpt", "value": r_ckpt, "n": n_ckpt_cells},
        {"metric": "spearman_rank_global",
         "da_kind": "size", "value": rank_size, "n": float("nan")},
        {"metric": "spearman_rank_global",
         "da_kind": "ckpt", "value": rank_ckpt, "n": float("nan")},
        {"metric": "retention",
         "da_kind": "size", "value": ret_size, "n": n_ret_size},
        {"metric": "retention",
         "da_kind": "ckpt", "value": ret_ckpt, "n": n_ret_ckpt},
    ])
    headline_path = out_dir / "headline_metrics.csv"
    headline.to_csv(headline_path, index=False)
    print(f"Wrote → {headline_path}")


# --- driver -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-pool", required=True,
                   help="Pool name (from configs/models.json) used as the "
                        "'train' split. Reads results/snr_definition/<pool>/.")
    p.add_argument("--test-pool", required=True,
                   help="Pool name used as the held-out 'test' split.")
    args = p.parse_args()

    pools = load_pools()
    for which, name in [("train", args.train_pool), ("test", args.test_pool)]:
        if name not in pools:
            p.error(f"unknown {which} pool {name!r}; "
                    f"available: {sorted(pools.keys())}")

    stage_train = pools[args.train_pool].get("stage", "pretraining")
    stage_test = pools[args.test_pool].get("stage", "pretraining")
    snr_root = OUT_ROOT / stage_train
    train_dir = snr_root / args.train_pool
    test_dir = OUT_ROOT / stage_test / args.test_pool
    out_dir = snr_root / f"{args.train_pool}__vs__{args.test_pool}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"train = {train_dir}")
    print(f"test  = {test_dir}")
    print(f"out   = {out_dir}")

    df_train = _load(train_dir)
    df_test = _load(test_dir)

    longs = []
    agreements = {}
    for kind in ("size", "ckpt"):
        agreement = per_language_agreement(df_train, df_test, kind)
        agreement_path = out_dir / f"per_language_agreement_da_{kind}.csv"
        agreement.to_csv(agreement_path, index=False)
        print(f"Wrote → {agreement_path}")

        render_per_language(
            agreement,
            out_dir / f"per_language_agreement_da_{kind}.png",
            title=f"Train-best variant: r on train vs test (DA-{kind})",
        )
        agreements[kind] = agreement

        long_df = variant_r_long(df_train, df_test, kind)
        longs.append(long_df)

    long_all = pd.concat(longs, ignore_index=True)
    long_path = out_dir / "variant_r_train_vs_test.csv"
    long_all.to_csv(long_path, index=False)
    print(f"Wrote → {long_path}")

    render_scatter(long_all, out_dir / "variant_r_train_vs_test.png",
                   title="Per-(language, variant) Pearson r — train vs test split")

    # Top-variants-overall comparison (Q3 from the README).
    tv_rows = []
    for split_name, df_split in (("train", df_train), ("test", df_test)):
        for kind in ("size", "ckpt"):
            table = _table_for(df_split, kind)
            for v in table.index:
                tv_rows.append({
                    "split": split_name, "da_kind": kind, "variant": v,
                    "mean_r": float(table.loc[v].mean(skipna=True)),
                })
    tv_long = pd.DataFrame(tv_rows)
    tv_wide = tv_long.pivot_table(
        index="variant", columns=["split", "da_kind"], values="mean_r"
    )
    tv_wide.columns = [f"{s}_{k}" for s, k in tv_wide.columns]
    # a DA kind with no language at MIN_LANG_TASKS tasks has no column: keep it, empty
    tv_wide = tv_wide.reindex(columns=[f"{s}_{k}" for s in ("train", "test") for k in ("size", "ckpt")])
    tv_wide = tv_wide.sort_values("train_size", ascending=False)
    # Rank-correlation between splits for each DA flavor (Spearman ρ).
    rank_corrs = {}
    for kind in ("size", "ckpt"):
        a = tv_wide[f"train_{kind}"].rank()
        b = tv_wide[f"test_{kind}"].rank()
        rank_corrs[kind] = float(a.corr(b))
    tv_path = out_dir / "top_variants_train_vs_test.csv"
    tv_wide.to_csv(tv_path)
    print(f"Wrote → {tv_path}")
    print(f"Spearman rank correlation across splits: "
          f"DA-size={rank_corrs['size']:+.3f}  DA-ckpt={rank_corrs['ckpt']:+.3f}")

    da_size_label = ", ".join(label for label, _, _ in _size_pairs(df_train)) or "absent"
    print(f"DA-size pairs on the holdout: {da_size_label}")
    for kind in ("size", "ckpt"):
        if tv_wide[f"train_{kind}"].isna().all() or tv_wide[f"test_{kind}"].isna().all():
            print(f"!!! RULE 8: no language has {MIN_LANG_TASKS} distinct tasks with a DA-{kind} value on both "
                  f"splits; the DA-{kind} holdout numbers are empty")
    write_summary(out_dir, train_dir, test_dir,
                  agreements["size"], agreements["ckpt"], long_all,
                  rank_corrs=rank_corrs, da_size_label=da_size_label)
    generate_readme(args.train_pool, args.test_pool, out_dir, agreements, rank_corrs, da_size_label)


def pool_cells(pool: str) -> tuple[set, list]:
    """The (size, L, arch, scheme) cells a pool holds, and its seeds."""
    df = build_snr_pool(pool)
    return set(map(tuple, df[["size", "L", "arch", "scheme"]].drop_duplicates().to_numpy())), sorted(df["seed"].unique())


def generate_readme(train_pool: str, test_pool: str, out_dir: Path, agreements: dict, rank_corrs: dict,
                    da_size_label: str) -> None:
    """The `seed-holdout` block of the rq03 README: the two pools' cells, the
    per-language rules and the headline numbers of `headline_metrics.csv`."""
    (cells_train, seeds_train), (cells_test, seeds_test) = pool_cells(train_pool), pool_cells(test_pool)
    same = cells_train == cells_test
    if not same:
        print(f"!!! the holdout pools hold different cells: train only {sorted(cells_train - cells_test)}, "
              f"test only {sorted(cells_test - cells_train)}")
    def _cells(cells):
        return ", ".join(f"{s} L{L} {a} scheme {sc}" for s, L, a, sc in sorted(cells, key=lambda c: (c[1], c[0], c[2], c[3])))
    rows = []
    for kind in ("size", "ckpt"):
        a = agreements[kind].replace({"train_best_variant": {"": None}, "test_best_variant": {"": None}}) \
                            .dropna(subset=["train_best_variant", "test_best_variant"])
        rows.append(f"| DA-{kind} | {len(a)} | {round(a['same_variant'].mean() * len(a)) if len(a) else 0} | "
                    f"{round(a['same_family'].mean() * len(a)) if len(a) else 0} | "
                    f"{fmt(rank_corrs[kind], 2) or 'no language at ' + str(MIN_LANG_TASKS) + ' tasks'} |")
    stage = load_pools()[train_pool].get("stage", "pretraining")
    body = "\n\n".join([
        "## Seed holdout",
        f"Regenerate with `python analysis/rq03_noise_and_snr/compare_seed_splits.py --train-pool {train_pool} "
        f"--test-pool {test_pool}`; the tables are under `{stage}/{out_dir.name}/`.",
        f"`{train_pool}` (seeds {', '.join(map(str, seeds_train))}) and `{test_pool}` (seeds "
        f"{', '.join(map(str, seeds_test))}) hold {'the same' if same else '**different**'} {len(cells_train)} cells: "
        f"{_cells(cells_train)}." + ("" if same else f" Train only: {_cells(cells_train - cells_test)}; test only: "
                                                     f"{_cells(cells_test - cells_train)}."),
        f"A language's r is rq04's Pearson r over its tasks' (log10 SNR, DA) points and needs at least "
        f"{MIN_LANG_TASKS} distinct tasks with a value (rule 8); `multi` and `??` are never a language (rule 7). "
        f"DA-size on the holdout is {da_size_label} (the pools stop at 600M, so the 1.7B reference never enters; "
        f"rule 9); DA-ckpt is the within-size early → final ranking. Agreement is counted over the languages with "
        f"a best variant on both splits.",
        "| DA | languages | same variant | same family | Spearman ρ of the variant ranking |\n"
        "|---|---|---|---|---|\n" + "\n".join(rows)])
    readme = OUT_ROOT / "README.md"
    replace_block(readme, "seed-holdout", body, f"compare_seed_splits.py --train-pool {train_pool} --test-pool {test_pool}")
    print(f"Wrote auto README block → {readme}")


if __name__ == "__main__":
    main()
