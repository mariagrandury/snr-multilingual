"""Apertus-multilingual port of analysis/smooth_subtasks.py.

Two views, mirroring the upstream "find the high-SNR subset" exercise:

  Case 1 — per_benchmark.csv
    "task"    = each multilingual benchmark family (arc, belebele,
                global_mmlu, xnli, ...).
    "subtask" = the per-language tasks in that family
                (arc_de, arc_es, ...).
    Goal: which language subset, ordered by per-language SNR, gives
    the highest combined SNR for the family?

  Case 2 — global_mmlu_full.csv  (special case)
    "task"    = global_mmlu_full (the multilingual MMLU benchmark).
    "subtask" = one subject (e.g., anatomy, philosophy, ...). Each
                subject's per-(model, ckpt) score is the mean across
                the 10 languages of global_mmlu_full.
    Goal: does the upstream finding (top-N subjects beat the full set)
    still hold on the multilingual Apertus models?

  Case 3 — global_mmlu_full_per_language.csv
    "task"    = global_mmlu_full_<lang> for each of the 10 languages.
    "subtask" = one subject within that language. No cross-language
                averaging — each language is treated independently.
    Goal: which subjects are most informative per language?

The upstream script uses instance-level data + IRT masks. Apertus only
has per-(model, ckpt, task) aggregate scores, so the SNR primitive is
``signal_to_noise_ratio`` over per-mix, last-5-ckpt arrays — same
formula as snr.snr_simple.compute_snr_small_scale. Combined-subset SNR
averages per-(mix, step) scores across the included subtasks before
applying that formula.
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from multilingual.analyze_snr_variants import (
    _BENCHMARK_FAMILY_OVERRIDES, _LANG_MAP, assign_language, benchmark_family,
)
from snr.constants import PLOT_DIR
from snr.download.apertus import load_apertus_eval_results
from snr.metrics import signal_to_noise_ratio

# 2-letter language codes used by global_mmlu_full subject keys.
_GMF_LANGS = ("ar", "en", "es", "hi", "ja", "ru", "sw", "tr", "vi", "zh")

ALL_SIZES = ["175M", "350M", "600M", "1B"]
LAST_N = 5
OUT_DIR = PLOT_DIR / "smooth_subtasks"


### SNR primitives (per-mix arrays, single subtask vs. averaged subset) ###


def _per_mix_last_n(scores_df: pd.DataFrame, last_n: int = LAST_N) -> list[np.ndarray]:
    """Sorted-by-step, grouped-by-mix list of last-n score arrays. Mirrors
    snr.snr_simple.compute_snr_small_scale (jagged-tolerant)."""
    scores_df = scores_df.sort_values("step")
    return [
        np.asarray(lst[-last_n:], dtype=float)
        for lst in scores_df.groupby("mix")["primary_score"].apply(list)
    ]


def snr_for_subset(df: pd.DataFrame, subtasks: list[str], size: str) -> float:
    """SNR after averaging per-(mix, step) scores across ``subtasks``.

    For one subtask this collapses to compute_snr_small_scale. With more
    subtasks we average across whichever subtasks are present at each
    (mix, step). A strict inner-join (require all subtasks at every
    kept (mix, step)) leaves arc/global_mmlu empty at most sizes
    because not every language is evaluated at every ckpt; the relaxed
    average is the pragmatic substitute and matches the intuition of
    "score on the multilingual subset = mean across the languages we
    have."
    """
    sub = df[(df["size"] == size) & (df["task"].isin(subtasks))]
    if sub.empty:
        return float("nan")
    if len(subtasks) == 1:
        arrays = _per_mix_last_n(sub)
    else:
        avg = (
            sub.groupby(["mix", "step"])["primary_score"]
            .mean()
            .reset_index()
        )
        arrays = _per_mix_last_n(avg)

    arrays = [a for a in arrays if a.size >= 2]
    if len(arrays) < 2:
        return float("nan")
    signal = [a.mean() for a in arrays]
    noise = np.concatenate(arrays)
    try:
        snr = signal_to_noise_ratio(signal, noise)
        return float(snr) if np.isfinite(snr) else float("nan")
    except Exception:
        return float("nan")


### Sweep one (task, list-of-subtasks) ###


def sweep_subset_snrs(
    df: pd.DataFrame,
    subtasks: list[str],
    size: str,
    rng: np.random.Generator | None = None,
) -> dict:
    """For each subtask compute its standalone SNR; sort descending; sweep
    cumulative subsets of size 1..N. Also compute a random-order baseline.
    Returns dict with sorted_subtasks, cumulative_snrs, random_cumulative_snrs.
    """
    per_subtask = {t: snr_for_subset(df, [t], size) for t in subtasks}
    sorted_subtasks = sorted(
        per_subtask.items(),
        key=lambda kv: -(kv[1] if np.isfinite(kv[1]) else -np.inf),
    )
    ordered = [t for t, _ in sorted_subtasks]

    cumulative = [
        snr_for_subset(df, ordered[: n + 1], size) for n in range(len(ordered))
    ]

    rng = rng or np.random.default_rng(0)
    rand_order = list(ordered)
    rng.shuffle(rand_order)
    rand_cumulative = [
        snr_for_subset(df, rand_order[: n + 1], size) for n in range(len(rand_order))
    ]

    return {
        "per_subtask": per_subtask,
        "sorted_subtasks": sorted_subtasks,
        "cumulative_snrs": cumulative,
        "random_subtasks": rand_order,
        "random_cumulative_snrs": rand_cumulative,
    }


def _argmax_safe(values: list[float]) -> int:
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return -1
    arr = np.where(np.isfinite(arr), arr, -np.inf)
    return int(np.argmax(arr))


### Plotting (one figure per family with size-rows) ###


def _plot_sweep(name: str, subtasks: list[str], per_size: dict[str, dict],
                save_path: Path):
    sizes = list(per_size.keys())
    if not sizes:
        return
    fig, axes = plt.subplots(len(sizes), 1, figsize=(max(6, 0.35 * len(subtasks)),
                                                     2.5 * len(sizes)),
                             sharex=True, squeeze=False)
    for i, size in enumerate(sizes):
        ax = axes[i][0]
        r = per_size[size]
        x = np.arange(1, len(r["cumulative_snrs"]) + 1)
        ax.plot(x, r["cumulative_snrs"], marker="o", markersize=3,
                linewidth=0.9, label="sorted by SNR")
        ax.plot(x, r["random_cumulative_snrs"], color="r", linewidth=0.7,
                alpha=0.8, label="random order")
        best_n = _argmax_safe(r["cumulative_snrs"]) + 1
        if best_n > 0:
            ax.axvline(best_n, color="grey", linestyle="--", linewidth=0.6)
            ax.set_title(f"{name} — {size}  (best subset = top {best_n} of "
                         f"{len(subtasks)})", fontsize=10)
        else:
            ax.set_title(f"{name} — {size}", fontsize=10)
        ax.set_ylabel("Combined SNR")
        ax.grid(True, linestyle="-", alpha=0.2)
    axes[-1][0].set_xlabel("Subset size (subtasks added in SNR order)")
    axes[0][0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


### CSV writer (one row per (task, size)) ###


def _result_row(task_name: str, size: str, sweep: dict) -> dict:
    cum = sweep["cumulative_snrs"]
    sorted_subs = [t for t, _ in sweep["sorted_subtasks"]]
    snrs = [s for _, s in sweep["sorted_subtasks"]]
    best_idx = _argmax_safe(cum)
    return {
        "task": task_name,
        "size": size,
        "n_subtasks": len(sorted_subs),
        "ranked_subtasks": "|".join(sorted_subs),
        "ranked_subtask_snrs": "|".join(
            f"{s:.4f}" if np.isfinite(s) else "nan" for s in snrs
        ),
        "cumulative_snrs": "|".join(
            f"{s:.4f}" if np.isfinite(s) else "nan" for s in cum
        ),
        "random_cumulative_snrs": "|".join(
            f"{s:.4f}" if np.isfinite(s) else "nan"
            for s in sweep["random_cumulative_snrs"]
        ),
        "full_set_snr": cum[-1] if cum else float("nan"),
        "best_n": best_idx + 1 if best_idx >= 0 else 0,
        "best_snr": cum[best_idx] if best_idx >= 0 else float("nan"),
        "best_subset": "|".join(sorted_subs[: best_idx + 1]) if best_idx >= 0 else "",
    }


### Case 1: per-benchmark (multilingual families) ###


# Trailing tokens that are allowed AFTER the language token in a
# language-aggregate task name. Anything else after the language is
# treated as a subject facet and the task is rejected.
#  - SCRIPTS: ISO 15924 codes seen in lm-eval task names, plus the
#    ``spai`` region marker that shows up in
#    ``global_piqa_completions_spa_latn_spai``.
#  - FORMATS: lm-eval format suffixes (``mc1`` / ``mc2`` for the
#    multiple-choice variants of TruthfulQA et al.).
_TRAILING_OK = {
    "arab", "latn", "cyrl", "hans", "hant", "deva", "jpan",
    "thai", "geor", "hebr", "beng", "knda", "tibt", "spai",
    "mc1", "mc2",
}


def _is_language_aggregate(task: str, family: str) -> bool:
    """Keep only the per-language aggregate of a family (e.g.,
    ``global_mmlu_ar``), not the per-(lang, subject) facet
    (``global_mmlu_ar_business``). The parquet ships both kinds of
    keys, so Case 1's "subtask = language" sweep needs this filter or
    each language counts multiple times.

    A "language aggregate" is a task whose tokens after the family name
    start with one language token; any further trailing tokens must be
    in ``_TRAILING_OK`` (script codes or known lm-eval format suffixes).
    Examples accepted: ``arc_de``, ``belebele_arb_Arab``,
    ``global_mmlu_ar``, ``global_piqa_completions_eng_latn``,
    ``global_piqa_completions_spa_latn_spai``, ``truthfulqa_eu_mc1``.
    Rejected: ``global_mmlu_ar_anatomy``, ``global_mmlu_es_social_sciences``.
    """
    if task in _BENCHMARK_FAMILY_OVERRIDES:
        return True
    if not task.startswith(family + "_"):
        return False
    rest = task[len(family) + 1:].split("_")
    if not rest or rest[0] not in _LANG_MAP:
        return False
    return all(tok.lower() in _TRAILING_OK for tok in rest[1:])


def collect_multilingual_families(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group tasks by benchmark_family, keeping only families with >1
    per-language aggregates (i.e., genuinely multilingual). Sort tasks
    in each family by language for stable output."""
    families: dict[str, list[str]] = defaultdict(list)
    for t in df["task"].unique():
        if assign_language(t) == "??":
            continue
        fam = benchmark_family(t)
        if not _is_language_aggregate(t, fam):
            continue
        families[fam].append(t)
    return {
        f: sorted(ts, key=assign_language)
        for f, ts in families.items()
        if len(ts) > 1
    }


def run_per_benchmark(df: pd.DataFrame, out_dir: Path) -> Path:
    families = collect_multilingual_families(df)
    print(f"Multilingual families: {len(families)} "
          f"({sum(len(v) for v in families.values())} subtasks)")

    rows = []
    plot_dir = out_dir / "per_benchmark_plots"
    for family, langs in tqdm(sorted(families.items()), desc="families"):
        per_size = {}
        for size in ALL_SIZES:
            sweep = sweep_subset_snrs(df, langs, size)
            per_size[size] = sweep
            rows.append(_result_row(family, size, sweep))
        _plot_sweep(family, langs, per_size, plot_dir / f"{family}.png")

    out = pd.DataFrame(rows)
    csv_path = out_dir / "per_benchmark.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    print(f"Wrote → {csv_path}")
    print(f"Wrote {len(families)} family plots → {plot_dir}")
    return csv_path


### Case 2: global_mmlu_full subjects (mean across the 10 languages) ###


def _parse_gmf_subject(task: str) -> str | None:
    """Return the subject token from ``global_mmlu_full_<lang>_<subject>``,
    or None if ``task`` isn't a per-subject key (e.g., the language
    aggregate ``global_mmlu_full_ar`` returns None)."""
    if not task.startswith("global_mmlu_full_"):
        return None
    rest = task[len("global_mmlu_full_"):]
    parts = rest.split("_", 1)
    if len(parts) < 2 or parts[0] not in _GMF_LANGS:
        return None
    return parts[1]


def _parse_gmf_lang_subject(task: str) -> tuple[str, str] | None:
    if not task.startswith("global_mmlu_full_"):
        return None
    rest = task[len("global_mmlu_full_"):]
    parts = rest.split("_", 1)
    if len(parts) < 2 or parts[0] not in _GMF_LANGS:
        return None
    return parts[0], parts[1]


def load_gmf_subjects_df(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-subject rows for ``global_mmlu_full``, averaged across the 10
    languages at each (model, ckpt). The output's ``task`` column carries
    the subject name (e.g., ``anatomy``).

    Reads from the per-(lang, subject) keys (``global_mmlu_full_<lang>_<subject>``)
    that already live in the parquet — no disk walking needed.
    """
    if df is None:
        df = load_apertus_eval_results()
    sub_rows = []
    for task in df["task"].unique():
        parsed = _parse_gmf_lang_subject(task)
        if parsed is None:
            continue
        lang, subject = parsed
        cur = df[df["task"] == task].copy()
        cur["language"] = lang
        cur["subject"] = subject
        sub_rows.append(cur)
    if not sub_rows:
        return pd.DataFrame()
    long = pd.concat(sub_rows, ignore_index=True)
    grouped = (
        long.groupby(["model", "mix", "size", "step", "subject"], as_index=False)
        .agg(primary_score=("primary_score", "mean"),
             n_languages=("language", "nunique"),
             seed=("seed", "first"),
             tokens=("tokens", "first"),
             compute=("compute", "first"))
        .rename(columns={"subject": "task"})
    )
    return grouped.sort_values(["size", "mix", "step", "task"]).reset_index(drop=True)


def run_gmf_subjects(out_dir: Path, df: pd.DataFrame | None = None) -> Path | None:
    df_gmf = load_gmf_subjects_df(df)
    if df_gmf.empty:
        print("No global_mmlu_full_<lang>_<subject> rows found — skipping.")
        return None
    subjects = sorted(df_gmf["task"].unique())
    coverage = df_gmf.groupby("size")["mix"].nunique()
    print(f"global_mmlu_full subjects: {len(subjects)}; "
          f"per-size #mixes with data: {coverage.to_dict()}")
    print(f"  mean #languages averaged per (mix, step, subject): "
          f"{df_gmf['n_languages'].mean():.2f} (max=10)")
    insufficient = [s for s, n in coverage.items() if n < 2]
    if insufficient:
        print(f"  warning: sizes {insufficient} have <2 mixes with per-subject "
              f"data; SNR is undefined there.")

    rows = []
    per_size = {}
    for size in ALL_SIZES:
        sweep = sweep_subset_snrs(df_gmf, subjects, size)
        per_size[size] = sweep
        rows.append(_result_row("global_mmlu_full", size, sweep))

    _plot_sweep("global_mmlu_full", subjects, per_size,
                out_dir / "global_mmlu_full_subjects.png")

    out = pd.DataFrame(rows)
    csv_path = out_dir / "global_mmlu_full.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    print(f"Wrote → {csv_path}")
    print(f"Wrote → {out_dir / 'global_mmlu_full_subjects.png'}")
    return csv_path


### Case 3: global_mmlu_full subjects per language (no cross-lang avg) ###


def load_gmf_per_language_df(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per (model, ckpt, lang, subject) rows for global_mmlu_full. ``task``
    is the subject; ``language`` is the language code.

    Reads the per-(lang, subject) task keys directly from the parquet.
    """
    if df is None:
        df = load_apertus_eval_results()
    rows = []
    for task in df["task"].unique():
        parsed = _parse_gmf_lang_subject(task)
        if parsed is None:
            continue
        lang, subject = parsed
        cur = df[df["task"] == task].copy()
        cur["language"] = lang
        cur["task"] = subject
        rows.append(cur)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["language", "size", "mix", "step", "task"]).reset_index(drop=True)


def run_gmf_subjects_per_language(out_dir: Path, df: pd.DataFrame | None = None) -> Path | None:
    df_lang = load_gmf_per_language_df(df)
    if df_lang.empty:
        print("No global_mmlu_full_<lang>_<subject> rows found — skipping.")
        return None
    languages = sorted(df_lang["language"].unique())
    print(f"global_mmlu_full languages: {len(languages)}; "
          f"subjects/lang: {df_lang.groupby('language')['task'].nunique().to_dict()}")

    rows = []
    plot_dir = out_dir / "global_mmlu_full_per_language_plots"
    for lang in tqdm(languages, desc="languages"):
        df_l = df_lang[df_lang["language"] == lang]
        subjects = sorted(df_l["task"].unique())
        per_size = {}
        for size in ALL_SIZES:
            sweep = sweep_subset_snrs(df_l, subjects, size)
            per_size[size] = sweep
            row = _result_row(f"global_mmlu_full_{lang}", size, sweep)
            row["language"] = lang
            rows.append(row)
        _plot_sweep(f"global_mmlu_full_{lang}", subjects, per_size,
                    plot_dir / f"{lang}.png")

    out = pd.DataFrame(rows)
    cols = ["language"] + [c for c in out.columns if c != "language"]
    out = out[cols]
    csv_path = out_dir / "global_mmlu_full_per_language.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(csv_path, index=False)
    print(f"Wrote → {csv_path}")
    print(f"Wrote {len(languages)} per-language plots → {plot_dir}")
    return csv_path


### Post-processing: unified summary across the three cases ###


_SUMMARY_COLS = [
    "case", "task", "size",
    "full_set_snr", "best_n", "best_snr", "snr_gain",
    "best_subset_short",
]


def _short_subset(s: str, max_items: int = 4) -> str:
    if not isinstance(s, str) or not s:
        return ""
    parts = s.split("|")
    if len(parts) <= max_items:
        return s
    return "|".join(parts[:max_items]) + f"|… (+{len(parts) - max_items})"


def build_summary(out_dir: Path) -> Path:
    """Read the three case CSVs from ``out_dir`` and emit ``summary.csv``
    ranking every (case, task, size) by ``snr_gain = best_snr -
    full_set_snr``. The three CSVs must already exist (run main() or
    the per-case run_* functions first).
    """
    case_files = {
        "case1_per_benchmark": out_dir / "per_benchmark.csv",
        "case2_global_mmlu_full_subjects": out_dir / "global_mmlu_full.csv",
        "case3_global_mmlu_full_per_language": out_dir / "global_mmlu_full_per_language.csv",
    }
    frames = []
    for case, path in case_files.items():
        if not path.exists():
            print(f"  skip {case}: {path} not found")
            continue
        df = pd.read_csv(path)
        df = df.assign(
            case=case,
            snr_gain=df["best_snr"] - df["full_set_snr"],
            best_subset_short=df["best_subset"].apply(_short_subset),
        )
        frames.append(df[_SUMMARY_COLS])
    if not frames:
        print("No case CSVs found — skipping summary.")
        return out_dir / "summary.csv"
    summary = pd.concat(frames, ignore_index=True)
    summary = summary.sort_values("snr_gain", ascending=False, na_position="last")
    csv_path = out_dir / "summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Wrote → {csv_path} ({len(summary)} rows)")
    return csv_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_apertus_eval_results()
    print(f"Loaded {len(df):,} rows | {df['model'].nunique()} models | "
          f"{df['task'].nunique()} tasks")

    print("\n=== Case 1: multilingual families (task = family, subtask = language) ===")
    run_per_benchmark(df, OUT_DIR)

    print("\n=== Case 2: global_mmlu_full subjects "
          "(task = global_mmlu_full, subtask = subject) ===")
    run_gmf_subjects(OUT_DIR, df=df)

    print("\n=== Case 3: global_mmlu_full subjects per language "
          "(task = global_mmlu_full_<lang>, subtask = subject) ===")
    run_gmf_subjects_per_language(OUT_DIR, df=df)

    print("\n=== Summary: snr_gain ranking across cases ===")
    build_summary(OUT_DIR)


if __name__ == "__main__":
    main()
