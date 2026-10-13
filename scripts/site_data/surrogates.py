"""Data of the surrogates page's additional views (rq04_surrogates).

- surrogates_scoring: the surrogate x proxy-size grid of the paper
  (rq3_surrogates_heatmap_paper.csv: accuracy and bBPB twins pooled) and the
  same rows split by scoring (rq3_surrogates_by_scoring.csv: accuracy, bBPB
  twins, per-language BPB), only stacked under one `scoring` key.
- surrogates_points: per (task, proxy) the values behind those rho's, from
  rq03's snr_variants_per_task.csv on analyze.py's population (an SNR at the
  proxy and above chance at the 1.7B reference, rule 1): SNR, signal and noise
  (rel_std), the early-checkpoint agreement (DA-ckpt at 10 %) and DA-size.
- surrogates_variants: top_variants_overall.csv with each definition's family
  (snr_definition_postprocess._VARIANT_FAMILY) and the per-language winning
  family counts of best_variant_family_per_language.csv.
"""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
sys.path.insert(0, str(REPO / "src"))
from analysis import paths  # noqa: E402
from analysis.grids import paper_name  # noqa: E402
from analysis.rq00_gate_and_curves.above_random import load_mask  # noqa: E402
from analysis.rq04_surrogates.snr_definition_postprocess import _VARIANT_FAMILY  # noqa: E402
from analysis.utils import BBPB, TARGET_SIZE, assign_language, benchmark_family, passes_gate  # noqa: E402

PRED = "pretraining/predictivity"
PROXIES = ["90M", "175M", "350M", "600M", "1B"]
SCORING = {"benchmark tasks": "pooled", "benchmark accuracy": "accuracy", "benchmark bBPB": "bBPB",
           "per-language bits per byte": "BPB"}


def _r(x, d=4):
    return None if pd.isna(x) else round(float(x), d)


def build() -> dict:
    d = paths.SURROGATES / PRED
    grid = pd.concat([pd.read_csv(d / "rq3_surrogates_heatmap_paper.csv"),
                      pd.read_csv(d / "rq3_surrogates_by_scoring.csv")])
    scoring = [dict(scoring=SCORING[r.kind], proxy=r.proxy, metric=r.metric, rho=_r(r.rho, 3), p=float(f"{r.p:.3g}"),
                    n=int(r.n)) for r in grid.itertuples()]

    v = pd.read_csv(paths.NOISE_AND_SNR / PRED / "snr_variants_per_task.csv")
    v = v[~v["task"].isin(["bpb_macro", "train_loss"])].reset_index(drop=True)   # in no population (rule 7)
    mask = load_mask("predictivity")
    tasks = v["task"].tolist()
    meta = [dict(t=t, f=benchmark_family(t), l=assign_language(t),
                 k="BPB" if t.startswith("bpb_") else "bBPB" if t.startswith(BBPB) else "accuracy") for t in tasks]
    rows = []
    for s in PROXIES:
        keep = v[f"snr_rel_std_{s}"].notna() & passes_gate(mask, v["task"], s, TARGET_SIZE).to_numpy() \
            & v[f"decision_acc_size_{s}"].notna()
        for i in v.index[keep]:
            r = v.loc[i]
            rows.append([int(i), s, _r(r[f"snr_rel_std_{s}"]), _r(r[f"signal_rel_std_{s}"], 5),
                         _r(r[f"noise_rel_std_{s}"], 5), _r(r[f"decision_acc_ckpt_f10_{s}"], 3),
                         _r(r[f"decision_acc_size_{s}"], 3)])
    fams = sorted({m["f"] for m in meta})

    top = pd.read_csv(d / "top_variants_overall.csv")
    fam = pd.read_csv(d / "best_variant_family_per_language.csv")
    variants = [dict(variant=r.variant, family=_VARIANT_FAMILY.get(r.variant, "other"), da_size=_r(r.mean_r_da_size, 3),
                     da_ckpt=_r(r.mean_r_da_ckpt, 3), n_size=int(r.n_languages_da_size), n_ckpt=int(r.n_languages_da_ckpt))
                for r in top.itertuples()]
    wins = {k: fam[f"family_{k}"].value_counts().to_dict() for k in ("da_size", "da_ckpt")}
    return {"surrogates_scoring": scoring,
            "surrogates_points": dict(tasks=meta, names={f: paper_name(f) for f in fams},
                                      columns=["i", "proxy", "snr", "signal", "noise", "early", "da"], data=rows),
            "surrogates_variants": dict(variants=variants, wins=wins, n_languages=int(len(fam)))}
