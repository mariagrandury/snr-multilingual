"""Above-random competence check, computed at the raw-metric level.

For every benchmark (parent task) and every model-size *bucket*: each model's
final-checkpoint accuracy is a binomial proportion over the task's `n_items`
scored examples (`configs/tasks.json`, derived from the harness results by
`derive_task_options.py`), so it has a Wilson lower confidence bound
(`statsmodels.stats.proportion.proportion_confint`, ALPHA = 0.10 — the
two-sided 90 % interval, whose lower end is the **one-sided 95 %** bound
this asks for). A run is **confidently above chance** when that bound
clears chance (`LCB_95 > 1 / n_options`), and a (benchmark, bucket) cell is above random
when at least half of the bucket's runs are (`MIN_SHARE`). The old fixed
margin (+0.05 over chance, whatever the option count or the test-set size)
is gone: a 2-option task with 500 items and a 4-option task with 100 items
are now held to the same evidence, and a run's rounded accuracy is turned
back into its count of correct answers (`round(acc * n_items)`) for the
bound. Anything else is treated as random and carries no usable signal.

One report per gated pool is written, each a pair of CSVs under the pool's own
dir `acc_vs_flops/<stage>/<pool>/` (pool-named, matching every other RQ):
- `seeds_28_1797_1904/` — only the custom Apertus pretrains (buckets 175M…1B).
- `custom_swissai_hf/`  — every model in scope (custom + a06 + distill + Swiss-AI
                          / HF references), bucketed across the full ladder
                          (175M, 350M, 600M, 1B, 3B, 4B, 7-9B, … 70B).
- `external/`           — every non-custom model (all parquets, incl posttraining).

Each report writes:
- `above_random_scores.csv` — row per benchmark, column per size bucket, value =
  (on the ladder: mean over the cells that trained the benchmark's language)
  mean score across all models in that bucket.
- `above_random_share.csv`  — same shape, value = share of the bucket's runs
  whose Wilson lower bound clears chance, plus a `population` column per
  bucket saying whether the cell was read on the cells that trained the
  language (`trained`) or on every cell of the bucket (`all`).
- `above_random_mask.csv`   — same shape, value = 1 (above random: share ≥
  MIN_SHARE) / 0 (at chance) / blank (no models in that bucket, or no chance
  level or item count for the task).
- `above_random_runs.csv`   — one row per (task, model): accuracy, n_items,
  LCB_95 and the verdict, the evidence behind the share.

The mask is the gate: SNR and all downstream analyses drop the `(benchmark,
size)` cells whose mask is 0 (`run_apertus_snr_variants.py` imports
`scores_and_mask` and NaN-s those cells). A blank is not a drop — per-language
BPB and the generative tasks have no chance level. This is a
*foundational* step — it depends ONLY on raw eval scores and the intrinsic
per-family answer-option counts (`N_OPTIONS` below, `random_baseline =
1 / n_options`); it never reads any RQ output, so every RQ depends on this
gate and not the reverse. `above_chance(score, task)` is the one rule, for
scripts that gate a score of their own.

    python analysis/rq00_gate_and_curves/above_random.py            # writes all reports
    python analysis/rq00_gate_and_curves/above_random.py --only custom_swissai_hf
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from statsmodels.stats.proportion import proportion_confint  # noqa: E402

from evals.scripts.utils.configs import (  # noqa: E402
    bucket_order, load_pools, load_tasks, size_bucket)
from analysis.utils import (  # noqa: E402
    assign_language, benchmark_family)
from analysis.utils import _is_parent_task  # noqa: E402
from analysis.paths import GATE_AND_CURVES

# proportion_confint's alpha is the TWO-sided level, so 0.10 gives the
# one-sided 95 % lower bound ("is this run above chance?" is one-sided).
ALPHA = 0.10
MIN_SHARE = 0.5     # a cell is above random when at least this share of its runs is
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
    "global_piqa_completions": 2, "global_piqa_parallel_cloze": 4,  # solution0..3 in the harness template
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
           "arabic_leaderboard_alghafa_mcq_exams_test",
           # mc2 is in here for a second reason, and has no N_OPTIONS entry on
           # purpose: its score is the probability mass on ALL true answers,
           # not a pick-one accuracy, so its baseline is the average share of
           # true options per item (~0.4) and not 1/n. Without this line the
           # count derive_task_options reads off the samples (4, or 5 for the
           # vi/zh siblings) would become a 0.25 chance level that every model
           # clears at once.
           "truthfulqa_mc2"}


def task_n_options(task: str) -> float:
    """Option count of a task: tasks.json's derived value, else the family
    table, else NaN (no chance level). An `_APPROX` family keeps the table:
    its items have different option counts (TruthfulQA mc1 4–13), and the
    derived value is one item's count, not the mean the chance level needs."""
    fam = benchmark_family(task)
    n = None if fam in _APPROX else load_tasks().get(task, {}).get("n_options")
    return float(n) if n else float(N_OPTIONS.get(fam, float("nan")))


def task_n_items(task: str) -> float:
    """Scored examples of a task (tasks.json `n_items`), NaN if unknown."""
    return float(load_tasks().get(task, {}).get("n_items") or float("nan"))


def wilson_lcb(score, n_items):
    """One-sided 95 % Wilson lower bound of an accuracy over `n_items`
    examples (the accuracy is turned back into its count of correct answers).
    Elementwise; NaN where the score or the count is missing."""
    score, n = np.asarray(score, float), np.asarray(n_items, float)
    ok = np.isfinite(score) & np.isfinite(n) & (n > 0)
    lcb = np.full(score.shape, np.nan)
    if ok.any():
        lcb[ok] = proportion_confint(np.rint(score[ok] * n[ok]), n[ok], alpha=ALPHA, method="wilson")[0]
    return lcb


def above_chance(score, task) -> pd.Series:
    """Per element: 1.0 when the run's one-sided 95 % LCB clears chance, 0.0 when it
    does not, NaN when the task has no chance level or no item count."""
    task = pd.Series(task)
    chance = 1 / task.map(task_n_options).astype(float)
    lcb = wilson_lcb(score, task.map(task_n_items))
    out = pd.Series((lcb > chance.to_numpy()).astype(float), index=task.index)
    return out.mask(~np.isfinite(lcb) | chance.isna())


def scores_and_mask(df: pd.DataFrame, sizes: list[str] | None = None, runs: bool = False):
    """Core, reused by run() and by the SNR pipeline.

    `df` is the raw per-row eval frame for the models in scope. Rows are
    grouped by *bucket* (`size_bucket`), so nearby sizes pool (0.6B→600M,
    7/8/9B→7-9B). Returns (scores, mask, meta), each indexed by task — plus,
    with `runs=True`, the per-run table (task, model, bucket, score,
    n_items, lcb, above) and the per-cell share of runs above chance.
    `scores`/`mask` have one column per bucket in `sizes` (default: every
    bucket present in `df`); `meta` carries family/language/n_options/
    n_items/random_baseline/options_exact. Depends only on raw scores, the
    option counts and the item counts. A mask cell is 1 (at least MIN_SHARE
    of the runs confidently above chance), 0 (at chance) or NA (no score, or
    no chance level / item count for the task).
    """
    df = df[df["task"].apply(_is_parent_task)].copy()
    df["family"] = df["task"].apply(benchmark_family)
    df["language"] = df["task"].apply(assign_language)
    df["bucket"] = df["size"].map(size_bucket)
    if sizes is None:
        sizes = [b for b in bucket_order() if b in set(df["bucket"].dropna())]

    # mean of each model's final-ckpt score, averaged over the models in a bucket
    finals = df.loc[df.groupby(["task", "model"])["step"].idxmax()]
    scores = (finals.pivot_table(index="task", columns="bucket",
                                 values="primary_score", aggfunc="mean")
              .reindex(columns=sizes))
    # On the ladder a cell enters a benchmark's bucket mean only if it trained
    # the benchmark's language: the all-languages runs are scored on every
    # language, and an English-only cell averaged into `arc_th` drags the mean
    # to chance whatever a cell that trained Thai can do. A task no cell
    # trained keeps the plain mean.
    finals = finals.assign(above=above_chance(finals["primary_score"].to_numpy(), finals["task"].to_numpy()).to_numpy(),
                           n_items=finals["task"].map(task_n_items).to_numpy())
    finals["lcb"] = wilson_lcb(finals["primary_score"], finals["n_items"])
    share_of = lambda f: (f.pivot_table(index="task", columns="bucket", values="above", aggfunc="mean")
                          .reindex(index=scores.index, columns=sizes))
    share = share_of(finals)
    if {"L", "scheme"} <= set(finals.columns):
        from pretrain.ladder_report import _trained_tasks
        keep = pd.Series(False, index=finals.index)
        for (L, scheme), g in finals.groupby(["L", "scheme"]):
            keep.loc[g.index] = g["task"].isin(_trained_tasks(int(L), scheme))
        trained = (finals[keep].pivot_table(index="task", columns="bucket",
                                            values="primary_score", aggfunc="mean")
                   .reindex(index=scores.index, columns=sizes))
        scores = trained.combine_first(scores)
        share_tr = share_of(finals[keep])
        population = share_tr.notna().replace({True: "trained", False: "all"}).where(share.notna())
        share = share_tr.combine_first(share)
        finals["trained"] = keep
    else:
        population = share.notna().replace({True: "all"}).where(share.notna())

    fam = df.groupby("task")["family"].first()
    lang = df.groupby("task")["language"].first()
    n_opt = pd.Series({t: task_n_options(t) for t in fam.index})   # NaN if unknown
    base_s = (1.0 / n_opt).round(3)

    mask = share.ge(MIN_SHARE).where(share.notna()).astype("Int64")  # 1 = above chance
    meta = pd.DataFrame({"family": fam, "language": lang,
                         "n_options": n_opt.astype("Int64"), "random_baseline": base_s,
                         "n_items": pd.Series({t: task_n_items(t) for t in fam.index}).astype("Int64"),
                         "options_exact": ~fam.isin(_APPROX)})
    if runs:
        cols = ["task", "model", "bucket", "primary_score", "n_items", "lcb", "above"] + (["trained"] if "trained" in finals else [])
        return scores, mask, meta, finals[cols].rename(columns={"primary_score": "score"}), share, population
    return scores, mask, meta


def load_mask(pool: str) -> pd.DataFrame | None:
    """Read the committed mask (task × bucket, Int64 0/1), or None if absent."""
    stage = load_pools()[pool].get("stage", "pretraining")
    path = GATE_AND_CURVES / stage / pool / "above_random_mask.csv"
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
    ("predictivity", "Predictivity ladder (175M–1.7B, seed 1904)"),
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

    df = build_snr_pool(pool, untrained=True)   # the gate covers every task, trained or not: rq06 gates with it too
    scores, mask, meta, runs, share, population = scores_and_mask(df, runs=True)
    buckets = list(scores.columns)

    stage = load_pools()[pool].get("stage", "pretraining")
    out_dir = GATE_AND_CURVES / stage / pool
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_out, mask_out = meta.join(scores), meta.join(mask)
    scores_out.index.name = mask_out.index.name = "task"
    scores_out.to_csv(out_dir / "above_random_scores.csv")
    mask_out.to_csv(out_dir / "above_random_mask.csv")
    share_out = meta.join(share).join(population.add_suffix("__population"))
    share_out.index.name = "task"
    share_out.to_csv(out_dir / "above_random_share.csv")
    runs.sort_values(["task", "bucket", "model"]).to_csv(out_dir / "above_random_runs.csv", index=False)

    above = (mask == 1).sum()
    have = mask.notna().sum()
    gated = mask.notna().any(axis=1)          # tasks with a chance level
    fully_random = ((mask == 0).sum(axis=1) == mask.notna().sum(axis=1))[gated].sum()
    no_items = int((meta["n_options"].notna() & meta["n_items"].isna()).sum())
    print(f"[{pool}] {int(gated.sum())} benchmarks with a chance level "
          f"(of {len(scores)} tasks) × {len(buckets)} buckets (gate: one-sided 95 % Wilson LCB > chance "
          f"in ≥ {MIN_SHARE:.0%} of the runs); {no_items} tasks with a chance level but no item count")
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
    mean score (bold = above the gate in more than half of the family's tasks)."""
    df = pd.read_csv(GATE_AND_CURVES / stage / label / "above_random_scores.csv")
    mask = pd.read_csv(GATE_AND_CURVES / stage / label / "above_random_mask.csv")
    sizes = [c for c in bucket_order() if c in df.columns]
    base = df.groupby("family")["random_baseline"].first()
    means = df.groupby("family")[sizes].mean()
    above = mask.groupby("family")[sizes].mean()
    order = means.mean(axis=1, skipna=True).sort_values(ascending=False).index

    header = ["benchmark", "rand"] + sizes
    rows = []
    for fam in order:
        cells = [f"`{fam}`", fmt_cell(base[fam])]
        for s in sizes:
            v = means.at[fam, s]
            txt = fmt_cell(v)
            if txt and above.at[fam, s] > 0.5:
                txt = f"**{txt}**"
            cells.append(txt)
        rows.append(cells)

    return (
        f"---\n"
        f"title: Appendix — Above-random signal\n"
        f"subtitle: \"{caption} · mean score per family × size (bold = above chance in most of its tasks)\"\n"
        f"---\n\n"
        f"{md_table(header, rows)}\n\n"
        f"{TABLE_STYLE}\n"
    )


def _report_exists(pool: str, stage: str) -> bool:
    """The pool's scores CSV is present and is not a git-lfs pointer (a plain
    clone holds pointers for the committed 36-sweep reports)."""
    p = GATE_AND_CURVES / stage / pool / "above_random_scores.csv"
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
