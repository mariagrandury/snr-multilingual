"""Data of the Findings pages' figures, the interactive versions of the paper's
four main-text figures, each from the table the paper figure is drawn from:

- paper_rq0: the share of (language, K, checkpoint) evaluations above chance
  per benchmark, format and model size against the language's training tokens
  (pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper.csv, rq00, pool
  predictivity_seeds); a benchmark's formats are read off its name
  (original, rf_ = RF, rfgm_ = LLM-RF).
- paper_rq1: one row per task, the median R² and Spearman rho of its model-size
  fits and the median R² of its training fits, and its regime
  (scaling_regimes.csv, rq01, pool predictivity_seeds).
- paper_rq2: the three panels of rq2.png (rq2_da_all_above_66_either_transformation_mono_axis.csv):
  DA-size per design axis, DA-ckpt and DA-goal per proxy size.
- paper_rq3: the surrogate x proxy-size grid of rq3.png (rq3_surrogates_heatmap_paper.csv).
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
sys.path.insert(0, str(REPO / "src"))
from analysis import paths  # noqa: E402
from analysis.grids import paper_name  # noqa: E402

FORMATS = {"rf_": "RF", "rfgm_": "LLM-RF"}


def _r(x, d=4):
    return None if pd.isna(x) else round(float(x), d)


def _format(bench: str) -> tuple[str, str]:
    """(base benchmark, format) of a benchmark name of the tokens-seen table."""
    for prefix, name in FORMATS.items():
        if bench.startswith(prefix):
            return bench[len(prefix):], name
    return bench, "Original"


def build() -> dict:
    t = pd.read_csv(paths.CHANCE_VS_TRAIN_TOKENS / "pretraining/predictivity_seeds"
                    / "pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper.csv")
    rq0 = []
    for r in t.itertuples():
        base, fmt = _format(r.benchmark)
        rq0.append(dict(b=base, f=fmt, size=r.model_size, tokens=_r(r.tokens, 0), share=_r(r.share_above, 3),
                        n=int(r.n_cells)))
    names = {b: paper_name(b) for b in sorted({r["b"] for r in rq0})}

    reg = pd.read_csv(paths.SCALING_PREDICTABILITY / "pretraining/predictivity_seeds/scaling_regimes.csv")
    rq1 = [dict(task=r.task, family=r.family, name=paper_name(r.family), language=r.language, r2_size=_r(r.r2_size, 3),
                rho_size=_r(r.rho_size, 3), r2_train=_r(r.r2_trajectory, 3), regime=r.regime) for r in reg.itertuples()]

    d = pd.read_csv(paths.DECISION_ACCURACY / "pretraining/predictivity"
                    / "rq2_da_all_above_66_either_transformation_mono_axis.csv")
    size = d[d["panel"] == "DA-size"]
    run = d[d["panel"] != "DA-size"]
    rq2 = dict(size=[dict(group=r.group, size=r.size, da=_r(r.reliability_macro, 3), tasks=None if pd.isna(r.n_tasks) else int(r.n_tasks),
                          pairs=None if pd.isna(r.n_units) else int(r.n_units)) for r in size.itertuples()],
               run=[dict(panel=r.panel, size=r.proxy_size, x=_r(r.chinchilla, 2), da=_r(r.da, 3),
                         tasks=None if pd.isna(r.tasks) else int(r.tasks)) for r in run.itertuples()])

    h = pd.read_csv(paths.SURROGATES / "pretraining/predictivity/rq3_surrogates_heatmap_paper.csv")
    rq3 = [dict(proxy=r.proxy, metric=r.metric, rho=_r(r.rho, 2), n=int(r.n)) for r in h.itertuples()]
    return {"paper_rq0": dict(rows=rq0, names=names), "paper_rq1": rq1, "paper_rq2": rq2, "paper_rq3": rq3}
