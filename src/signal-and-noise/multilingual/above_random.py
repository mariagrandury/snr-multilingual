"""Above-random competence check, computed at the raw-metric level.

For every benchmark (parent task) and every model-size *bucket*, average the
raw `primary_score` across all models evaluated in that bucket (final
checkpoint per model — same level as the acc-vs-FLOPs curves). A benchmark is
**above random** at a bucket only if that mean beats chance (`1 / n_options`)
by a margin: `mean > random_baseline + MARGIN` (MARGIN = 0.05). Anything at or
below chance+margin is treated as random and carries no usable signal.

Two reports are written, each a pair of CSVs under
`results/acc_vs_flops/<stage>/<label>/`:
- `custom/`          — only the custom Apertus pretrains (buckets 175M…1B).
- `custom_swiss_hf/` — every model in scope (custom + a06 + distill + Swiss-AI
                       / HF references), bucketed across the full ladder
                       (175M, 350M, 600M, 1B, 3B, 4B, 7-9B, … 70B).

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

    python multilingual/above_random.py            # writes both reports
    python multilingual/above_random.py --only custom_swiss_hf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from evals.scripts.utils.configs import (  # noqa: E402
    bucket_order, load_pools, load_snr_params, size_bucket)
from multilingual.analyze_snr_variants import (  # noqa: E402
    assign_language, benchmark_family)
from multilingual.run_apertus import _is_parent_task  # noqa: E402
from snr.constants import PLOT_DIR  # noqa: E402

# A benchmark must beat chance by more than this to count as "above random".
MARGIN = 0.05
_SNR = load_snr_params()
SIZES = _SNR["small_sizes"] + [_SNR["target_size"]]   # 175M, 350M, 600M, 1B

# Answer-option count per benchmark family — intrinsic benchmark metadata (the
# number of MCQA / completion candidates the eval scores), NOT an analysis
# result. This is the ONLY source of the random baseline (= 1 / n_options); the
# gate depends only on raw eval scores + these counts, so the RQs depend on the
# gate and never the other way around. Families in `_APPROX` have a variable
# number of options per item, so their baseline is an approximation.
N_OPTIONS = {
    # multilingual families (cf. lm-eval task specs)
    "arc": 4, "belebele": 4, "global_mmlu": 4, "global_mmlu_full": 4,
    "global_piqa_completions": 2, "hellaswag": 4, "multiblimp": 2, "paws": 2,
    "xcopa": 2, "xnli": 3, "xstorycloze": 2, "xwinograd": 2,
    # standalone-English + extra MCQA families
    "mmlu": 4, "piqa": 2, "openbookqa": 4, "commonsense_qa": 5, "social_iqa": 3,
    "winogrande": 2, "ai2_arc": 4, "m_arc": 4, "m_hellaswag": 4,
    "include_base_44": 4,
    "agieval": 4, "agieval_logiqa": 4, "agieval_sat": 4, "agieval_lsat": 5,
    "truthfulqa": 4, "truthfulqa_mc1": 4,
    "arabic_leaderboard_alghafa_mcq_exams_test": 4,
}
_APPROX = {"truthfulqa", "truthfulqa_mc1", "agieval", "agieval_logiqa",
           "agieval_sat", "agieval_lsat",
           "arabic_leaderboard_alghafa_mcq_exams_test"}


def scores_and_mask(df: pd.DataFrame, margin: float = MARGIN, sizes: list[str] = SIZES
                    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Core, reused by run() and by the SNR pipeline.

    `df` is the raw per-row eval frame for the models in scope. Rows are
    grouped by *bucket* (`size_bucket`), so nearby sizes pool (0.6B→600M,
    7/8/9B→7-9B). Returns (scores, mask, meta), each indexed by task.
    `scores`/`mask` have one column per bucket in `sizes` (default the custom
    `SIZES`, which is what the SNR gate keys on); `meta` carries family/
    language/n_options/random_baseline/options_exact. Depends only on raw
    scores + the intrinsic N_OPTIONS counts.
    """
    df = df[df["task"].apply(_is_parent_task)].copy()
    df["family"] = df["task"].apply(benchmark_family)
    df["language"] = df["task"].apply(assign_language)
    df["bucket"] = df["size"].map(size_bucket)

    # mean of each model's final-ckpt score, averaged over all models in a bucket
    finals = df.loc[df.groupby(["task", "model"])["step"].idxmax()]
    scores = (finals.pivot_table(index="task", columns="bucket",
                                 values="primary_score", aggfunc="mean")
              .reindex(columns=sizes))

    fam = df.groupby("task")["family"].first()
    lang = df.groupby("task")["language"].first()
    n_opt = fam.map(N_OPTIONS)                            # NaN if family unknown
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
    path = PLOT_DIR / "acc_vs_flops" / stage / pool / "above_random_mask.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path, index_col="task")
    return m[[c for c in m.columns if c in bucket_order()]].astype("Int64")


# The two reports: (label, pool). `custom` = custom pretrains only (the SNR
# gate's domain, buckets 175M…1B); `custom_swiss_hf` = every model in scope
# (custom + a06 + distill + Swiss-AI/HF refs) across the full bucket ladder.
REPORTS = [("custom", "seeds_28_1797_1904"),
           ("custom_swiss_hf", "custom_swissai_hf")]


def run(label: str, pool: str) -> None:
    # build_snr_pool folds in the externals when the pool sets include_external
    # (custom_swissai_hf) and is custom-only otherwise (seeds_28_1797_1904).
    # Lazy import: run_apertus_snr_variants imports SIZES/scores_and_mask from
    # here, so a module-top import would be circular.
    from multilingual.run_apertus_snr_variants import build_snr_pool

    df = build_snr_pool(pool)
    buckets = [b for b in bucket_order() if b in set(df["size"].map(size_bucket).dropna())]
    scores, mask, meta = scores_and_mask(df, sizes=buckets)

    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = PLOT_DIR / "acc_vs_flops" / stage / label
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_out, mask_out = meta.join(scores), meta.join(mask)
    scores_out.index.name = mask_out.index.name = "task"
    scores_out.to_csv(out_dir / "above_random_scores.csv")
    mask_out.to_csv(out_dir / "above_random_mask.csv")

    above = (mask == 1).sum()
    have = mask.notna().sum()
    fully_random = ((mask == 0).sum(axis=1) == mask.notna().sum(axis=1)).sum()
    print(f"[{label}] pool '{pool}': {len(scores)} benchmarks × {len(buckets)} "
          f"buckets (margin = +{MARGIN} over chance)")
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
    df = pd.read_csv(PLOT_DIR / "acc_vs_flops" / stage / label / "above_random_scores.csv")
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


def above_random_slides(stage: str = "pretraining") -> list[str]:
    """The two above-random appendix slides (custom-only, then all models)."""
    return [
        _ar_slide("custom", stage, "Custom Apertus pretrains only"),
        _ar_slide("custom_swiss_hf", stage,
                  "All models (custom + Swiss-AI/HF refs)"),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=[label for label, _ in REPORTS], default=None,
                   help="Generate only this report (default: both).")
    args = p.parse_args()
    for label, pool in REPORTS:
        if args.only and label != args.only:
            continue
        run(label=label, pool=pool)


if __name__ == "__main__":
    main()
