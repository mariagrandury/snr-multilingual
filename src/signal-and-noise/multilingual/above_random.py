"""Above-random competence check, computed at the raw-metric level.

For every benchmark (parent task) and every custom model-size group, average
the raw `primary_score` across all models evaluated at that size (final
checkpoint per model — same level as the acc-vs-FLOPs curves). A benchmark is
**above random** at a size only if that mean beats chance (`1 / n_options`) by
a margin: `mean > random_baseline + MARGIN` (MARGIN = 0.05). Anything at or
below chance+margin is treated as random and carries no usable signal.

Two CSVs are written (under `results/acc_vs_flops/<stage>/<pool>/`):
- `above_random_scores.csv` — row per benchmark, column per size group, value =
  mean score across all models at that size.
- `above_random_mask.csv`   — same shape, value = 1 (above random) / 0 (random)
  / blank (no models at that size).

The mask is the gate: SNR and all downstream analyses only keep `(benchmark,
size)` cells with mask == 1 (`run_apertus_snr_variants.py` imports
`scores_and_mask` and applies it). This is a *foundational* step — it depends
ONLY on raw eval scores and the intrinsic per-family answer-option counts
(`N_OPTIONS` below, `random_baseline = 1 / n_options`); it never reads any RQ
output, so every RQ depends on this gate and not the reverse.

    python multilingual/above_random.py --pool custom_swissai_hf
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
    expand_pool, load_pools, load_snr_params)
from multilingual.analyze_snr_variants import (  # noqa: E402
    assign_language, benchmark_family)
from multilingual.run_apertus import _is_parent_task  # noqa: E402
from snr.constants import PLOT_DIR  # noqa: E402
from snr.download.apertus import load_apertus_eval_results  # noqa: E402

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


def scores_and_mask(df: pd.DataFrame,
                    margin: float = MARGIN) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Core, reused by run() and by the SNR pipeline.

    `df` is the raw per-row eval frame for the custom models in scope. Returns
    (scores, mask, meta), each indexed by task. `scores`/`mask` have one column
    per size in SIZES; `meta` carries family/language/n_options/random_baseline/
    options_exact. Depends only on raw scores + the intrinsic N_OPTIONS counts.
    """
    df = df[df["task"].apply(_is_parent_task)].copy()
    df["family"] = df["task"].apply(benchmark_family)
    df["language"] = df["task"].apply(assign_language)

    # mean of each model's final-ckpt score, averaged over all models at a size
    finals = df.loc[df.groupby(["task", "model"])["step"].idxmax()]
    scores = (finals.pivot_table(index="task", columns="size",
                                 values="primary_score", aggfunc="mean")
              .reindex(columns=SIZES))

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
    """Read the committed mask (task × size, Int64 0/1), or None if absent."""
    stage = load_pools()[pool].get("stage", "pretraining")
    path = PLOT_DIR / "acc_vs_flops" / stage / pool / "above_random_mask.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path, index_col="task")
    return m[[c for c in m.columns if c in SIZES]].astype("Int64")


def run(pool: str, out_dir: Path) -> None:
    df = load_apertus_eval_results()                       # custom pretrains
    df = df[df["model"].isin(set(expand_pool(pool)))]
    scores, mask, meta = scores_and_mask(df)

    out_dir.mkdir(parents=True, exist_ok=True)
    scores_out, mask_out = meta.join(scores), meta.join(mask)
    scores_out.index.name = mask_out.index.name = "task"
    scores_out.to_csv(out_dir / "above_random_scores.csv")
    mask_out.to_csv(out_dir / "above_random_mask.csv")

    above = (mask == 1).sum()
    have = mask.notna().sum()
    fully_random = ((mask == 0).sum(axis=1) == mask.notna().sum(axis=1)).sum()
    print(f"Pool '{pool}': {len(scores)} benchmarks × {len(SIZES)} size groups "
          f"(margin = +{MARGIN} over chance)")
    for s in SIZES:
        print(f"  {s:>5}: above random {int(above[s])}/{int(have[s])}")
    print(f"  benchmarks random at EVERY available size (fully dropped): {int(fully_random)}")
    print(f"Wrote {out_dir/'above_random_scores.csv'} and {out_dir/'above_random_mask.csv'}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", default="custom_swissai_hf",
                   help="Pool name from configs/models.json (default: custom_swissai_hf).")
    p.add_argument("--out-subdir", default=None,
                   help="Subdir under results/acc_vs_flops/<stage>/ (default: <pool>).")
    args = p.parse_args()
    if args.pool not in load_pools():
        p.error(f"unknown pool {args.pool!r}; available: {sorted(load_pools().keys())}")
    stage = load_pools()[args.pool].get("stage", "pretraining")
    run(pool=args.pool,
        out_dir=PLOT_DIR / "acc_vs_flops" / stage / (args.out_subdir or args.pool))


if __name__ == "__main__":
    main()
