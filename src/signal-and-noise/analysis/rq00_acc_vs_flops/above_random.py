"""Above-random competence check, computed at the raw-metric level.

For every benchmark (parent task) and every model-size *bucket*, average the
raw `primary_score` across all models evaluated in that bucket (final
checkpoint per model — same level as the acc-vs-FLOPs curves). A benchmark is
**above random** at a bucket only if that mean beats chance (`1 / n_options`)
by a margin: `mean > random_baseline + MARGIN` (MARGIN = 0.05). Anything at or
below chance+margin is treated as random and carries no usable signal.

One report per gated pool is written, each a pair of CSVs under the pool's own
dir `acc_vs_flops/<stage>/<pool>/` (pool-named, matching every other RQ):
- `seeds_28_1797_1904/` — only the custom Apertus pretrains (buckets 175M…1B).
- `custom_swissai_hf/`  — every model in scope (custom + a06 + distill + Swiss-AI
                          / HF references), bucketed across the full ladder
                          (175M, 350M, 600M, 1B, 3B, 4B, 7-9B, … 70B).
- `external/`           — every non-custom model (all parquets, incl posttraining).

Each report writes:
- `above_random_scores.csv` — row per benchmark, column per size bucket, value =
  mean score across all models in that bucket.
- `above_random_mask.csv`   — same shape, value = 1 (above random) / 0 (random)
  / blank (no models in that bucket).

The mask is the gate: SNR and all downstream analyses only keep `(benchmark,
size)` cells with mask == 1 (`run_apertus_snr_variants.py` imports
`scores_and_mask` and applies it at the custom `SIZES`). This is a
*foundational* step — it depends ONLY on raw eval scores and the intrinsic
per-family answer-option counts (`N_OPTIONS` below, `random_baseline =
1 / n_options`); it never reads any RQ output, so every RQ depends on this
gate and not the reverse.

    python analysis/rq00_acc_vs_flops/above_random.py            # writes all reports
    python analysis/rq00_acc_vs_flops/above_random.py --only custom_swissai_hf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[3]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from evals.scripts.utils.configs import (  # noqa: E402
    bucket_order, load_pools, load_snr_params, load_tasks, size_bucket)
from analysis.utils import (  # noqa: E402
    assign_language, benchmark_family)
from analysis.utils import _is_parent_task  # noqa: E402
from snr.constants import PLOT_DIR  # noqa: E402
from analysis.paths import ACC_VS_FLOPS

# A benchmark must beat chance by more than this to count as "above random".
MARGIN = 0.05
_SNR = load_snr_params()
SIZES = _SNR["small_sizes"] + [_SNR["target_size"]]   # 90M, 175M, 350M, 600M, 1B

# Answer-option count per task. configs/tasks.json carries `n_options` where
# it was derived from the evaluated samples (derive_task_options.py); the
# per-family table below fills in the rest — intrinsic benchmark metadata (the
# number of MCQA / completion candidates the eval scores), NOT an analysis
# result. Together they are the ONLY source of the random baseline
# (= 1 / n_options); the gate depends only on raw eval scores + these counts,
# so the RQs depend on the gate and never the other way around. Families in
# `_APPROX` have a variable number of options per item, so their baseline is
# an approximation. A task with no option count anywhere (per-language BPB,
# generative tasks) has no chance level and is never gated.
N_OPTIONS = {
    # multilingual families (cf. lm-eval task specs)
    "arc": 4, "belebele": 4, "global_mmlu": 4, "global_mmlu_full": 4,
    "global_piqa_completions": 2, "global_piqa_parallel_cloze": 2,
    "global_piqa_nonparallel_cloze": 2, "hellaswag": 4, "multiblimp": 2, "paws": 2,
    "xcopa": 2, "xnli": 3, "xstorycloze": 2, "xwinograd": 2,
    "afrimmlu": 4, "afrixnli": 3, "include_base_44": 4, "truthfulqa-multi_mc1": 4,
    # standalone-English + extra MCQA families
    "mmlu": 4, "piqa": 2, "openbookqa": 4, "commonsense_qa": 5, "social_iqa": 3,
    "winogrande": 2, "ai2_arc": 4, "m_arc": 4, "m_hellaswag": 4,
    "include_base_44": 4,
    "agieval": 4, "agieval_logiqa": 4, "agieval_sat": 4, "agieval_lsat": 5,
    "truthfulqa": 4, "truthfulqa_mc1": 4,
    "arabic_leaderboard_alghafa_mcq_exams_test": 4,
}
_APPROX = {"truthfulqa", "truthfulqa_mc1", "truthfulqa-multi_mc1", "agieval",
           "agieval_logiqa", "agieval_sat", "agieval_lsat",
           "arabic_leaderboard_alghafa_mcq_exams_test"}


def task_n_options(task: str) -> float:
    """Option count of a task: tasks.json's derived value, else the family
    table, else NaN (no chance level)."""
    n = load_tasks().get(task, {}).get("n_options")
    return float(n) if n else float(N_OPTIONS.get(benchmark_family(task), float("nan")))


def scores_and_mask(df: pd.DataFrame, margin: float = MARGIN,
                    sizes: list[str] | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Core, reused by run() and by the SNR pipeline.

    `df` is the raw per-row eval frame for the models in scope. Rows are
    grouped by *bucket* (`size_bucket`), so nearby sizes pool (0.6B→600M,
    7/8/9B→7-9B). Returns (scores, mask, meta), each indexed by task.
    `scores`/`mask` have one column per bucket in `sizes` (default: every
    bucket present in `df`); `meta` carries family/language/n_options/
    random_baseline/options_exact. Depends only on raw scores + the option
    counts. A mask cell is 1 (above chance + margin), 0 (at chance) or NA
    (no score, or no chance level for the task).
    """
    df = df[df["task"].apply(_is_parent_task)].copy()
    df["family"] = df["task"].apply(benchmark_family)
    df["language"] = df["task"].apply(assign_language)
    df["bucket"] = df["size"].map(size_bucket)
    if sizes is None:
        sizes = [b for b in bucket_order() if b in set(df["bucket"].dropna())]

    # mean of each model's final-ckpt score, averaged over all models in a bucket
    finals = df.loc[df.groupby(["task", "model"])["step"].idxmax()]
    scores = (finals.pivot_table(index="task", columns="bucket",
                                 values="primary_score", aggfunc="mean")
              .reindex(columns=sizes))

    fam = df.groupby("task")["family"].first()
    lang = df.groupby("task")["language"].first()
    n_opt = pd.Series({t: task_n_options(t) for t in fam.index})   # NaN if unknown
    base_s = (1.0 / n_opt).round(3)

    diff = scores.sub(base_s, axis=0)                    # NaN where score or baseline absent
    mask = diff.gt(margin).where(diff.notna()).astype("Int64")  # 1=above chance+margin
    meta = pd.DataFrame({"family": fam, "language": lang,
                         "n_options": n_opt.astype("Int64"), "random_baseline": base_s,
                         "options_exact": ~fam.isin(_APPROX)})
    return scores, mask, meta


def load_mask(pool: str) -> pd.DataFrame | None:
    """Read the committed mask (task × bucket, Int64 0/1), or None if absent."""
    stage = load_pools()[pool].get("stage", "pretraining")
    path = ACC_VS_FLOPS / stage / pool / "above_random_mask.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path, index_col="task")
    return m[[c for c in m.columns if c in bucket_order()]].astype("Int64")


# Above-random reports, one per model-set pool we gate. Output dirs are
# pool-named (matching every other RQ — no separate label namespace), each with
# a caption for the appendix slide:
#   seeds_28_1797_1904 = pure custom pretrains (the SNR gate's domain, 175M…1B)
#   custom_swissai_hf   = custom + a06 + distill + Swiss-AI/HF refs (full ladder)
#   external            = every non-custom model (all parquets, incl posttraining)
REPORTS = [
    ("predictivity", "Predictivity ladder (90M–1.7B, seed 1904)"),
    ("seeds_28_1797_1904", "Custom Apertus pretrains only"),
    ("custom_swissai_hf", "All models (custom + Swiss-AI/HF refs)"),
    ("external", "All non-custom models (refs + a06 + distill + posttraining)"),
]


def run(pool: str) -> None:
    # build_snr_pool folds in the externals when the pool sets include_external
    # (custom_swissai_hf), is custom-only for seeds_*, and pools every non-custom
    # model for `external`. Lazy import: run_apertus_snr_variants imports
    # SIZES/scores_and_mask from here, so a module-top import would be circular.
    from analysis.utils import build_snr_pool

    df = build_snr_pool(pool)
    scores, mask, meta = scores_and_mask(df)
    buckets = list(scores.columns)

    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = ACC_VS_FLOPS / stage / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_out, mask_out = meta.join(scores), meta.join(mask)
    scores_out.index.name = mask_out.index.name = "task"
    scores_out.to_csv(out_dir / "above_random_scores.csv")
    mask_out.to_csv(out_dir / "above_random_mask.csv")

    above = (mask == 1).sum()
    have = mask.notna().sum()
    gated = mask.notna().any(axis=1)          # tasks with a chance level
    fully_random = ((mask == 0).sum(axis=1) == mask.notna().sum(axis=1))[gated].sum()
    print(f"[{pool}] {int(gated.sum())} benchmarks with a chance level "
          f"(of {len(scores)} tasks) × {len(buckets)} buckets (margin = +{MARGIN} over chance)")
    for s in buckets:
        print(f"  {s:>7}: above random {int(above[s])}/{int(have[s])}")
    print(f"  benchmarks random at EVERY available bucket (fully dropped): {int(fully_random)}")
    print(f"Wrote {out_dir/'above_random_scores.csv'} and {out_dir/'above_random_mask.csv'}\n")


# --- appendix slides --------------------------------------------------------
# Markdown-table helpers shared with da_per_benchmark.generate_slides (this is
# the lower-level module, so the slide primitives live here).

# Per-slide style so the wide tables fit a 16:9 Slidev frame.
TABLE_STYLE = (
    "<style>\n"
    ".slidev-layout table { font-size: 0.52em; line-height: 1.15; }\n"
    ".slidev-layout th, .slidev-layout td { padding: 1px 6px; }\n"
    "</style>"
)


def fmt_cell(v: float) -> str:
    return "" if pd.isna(v) else f"{v:.2f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _ar_slide(label: str, stage: str, caption: str) -> str:
    """One above-random slide: benchmark-family rows × size buckets, cell =
    mean score (bold = beats chance + MARGIN)."""
    df = pd.read_csv(ACC_VS_FLOPS / stage / label / "above_random_scores.csv")
    sizes = [c for c in bucket_order() if c in df.columns]
    base = df.groupby("family")["random_baseline"].first()
    means = df.groupby("family")[sizes].mean()
    order = means.mean(axis=1, skipna=True).sort_values(ascending=False).index

    header = ["benchmark", "rand"] + sizes
    rows = []
    for fam in order:
        cells = [f"`{fam}`", fmt_cell(base[fam])]
        for s in sizes:
            v = means.at[fam, s]
            txt = fmt_cell(v)
            if txt and v > base[fam] + MARGIN:
                txt = f"**{txt}**"
            cells.append(txt)
        rows.append(cells)

    return (
        f"---\n"
        f"title: Appendix — Above-random signal\n"
        f"subtitle: \"{caption} · mean score per family × size (bold = beats chance + {MARGIN})\"\n"
        f"---\n\n"
        f"{md_table(header, rows)}\n\n"
        f"{TABLE_STYLE}\n"
    )


def _report_exists(pool: str, stage: str) -> bool:
    """The pool's scores CSV is present and is not a git-lfs pointer (a plain
    clone holds pointers for the committed 36-sweep reports)."""
    p = ACC_VS_FLOPS / stage / pool / "above_random_scores.csv"
    return p.exists() and not p.read_text(errors="ignore").startswith("version https://git-lfs")


def above_random_slides(stage: str = "pretraining") -> list[str]:
    """Above-random appendix slides for the pretraining-stage pools whose report
    exists locally (the ladder, then the 36-sweep's custom-only and all-models
    pools). The `external` report lives at its own stage, so it isn't in the
    pretraining deck — its CSVs are written by `run("external")`."""
    return [
        _ar_slide(pool, stage, caption)
        for pool, caption in REPORTS
        if load_pools()[pool].get("stage", "pretraining") == stage
        and _report_exists(pool, stage)
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=[pool for pool, _ in REPORTS], default=None,
                   help="Generate only this pool's report (default: all).")
    args = p.parse_args()
    for pool, _caption in REPORTS:
        if args.only and pool != args.only:
            continue
        run(pool)


if __name__ == "__main__":
    main()
