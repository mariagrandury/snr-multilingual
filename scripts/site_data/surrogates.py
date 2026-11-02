"""Data of the surrogates page's additional views (rq04_surrogates).

- surrogates_scoring: the surrogate x proxy-size grid of the paper's Figure 4
  (rq3_surrogates_heatmap_paper.csv: accuracy and bBPB pooled, the table
  paper.py's paper_rq3 reads) and the same rows split by scoring, the paper's
  Figure 20 (rq3_surrogates_by_scoring_paper.csv: accuracy, bBPB, per-language
  BPB), only stacked under one `scoring` key. rho keeps four decimals, so the
  view's two-decimal labels round like the paper's figures.
- surrogates_points: per (task, proxy) the values behind those rho's, from
  rq03's snr_variants_per_task.csv on analyze.py's population (an SNR at the
  proxy and above chance at the 1.7B reference, rule 1): SNR, signal and noise
  (rel_std), the early-checkpoint agreement (DA-ckpt at 10 %) and DA-size.
- surrogates_variants: top_variants_overall.csv with each definition's family
  (snr_definition_postprocess._VARIANT_FAMILY) and the per-language winning
  family counts of best_variant_family_per_language.csv.
- surrogates_bench: per benchmark, the Spearman rho with DA-size within that
  benchmark (surrogate_correlations.csv: multi-axis pairs, every K, one point
  per (benchmark, language) cluster, proxies 90M-1B pooled) of the
  relative-std SNR over the relative checkpoint noise and of the strongest of
  the 22 x 6 signal x noise ratios; benchmarks with at least MIN_CLUSTERS
  clusters, as in the paper's stratum table (make_surrogate_tables.py). The
  source of section 6's "up to rho = 0.93 for HellaSwag".
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
SNR_REF = "snr__rel_std__ckpt_rel"   # Figure 4's "SNR, relative std"
MIN_CLUSTERS = 10
SCORING = {"benchmark tasks": "pooled", "benchmark accuracy": "accuracy", "benchmark bBPB": "bBPB",
           "per-language bits per byte": "BPB"}


def _r(x, d=4):
    return None if pd.isna(x) else round(float(x), d)


def build() -> dict:
    d = paths.SURROGATES / PRED
    grid = pd.concat([pd.read_csv(d / "rq3_surrogates_heatmap_paper.csv"),
                      pd.read_csv(d / "rq3_surrogates_by_scoring_paper.csv")])
    scoring = [dict(scoring=SCORING[r.kind], proxy=r.proxy, metric=r.metric, rho=_r(r.rho), p=float(f"{r.p:.3g}"),
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

    c = pd.read_csv(d / "surrogate_correlations.csv", low_memory=False,
                    usecols=["t_kind", "metric", "axes", "L", "subset_type", "subset", "surrogate", "rho", "q", "n_units"])
    c = c[(c["t_kind"] == "size") & (c["metric"] == "da") & (c["axes"] == "multi-axis") & (c["L"].astype(str) == "all")
          & c["subset_type"].isin(["benchmark", "all"]) & c["surrogate"].str.startswith("snr__")].dropna(subset=["rho"])
    bench = []
    for (st, b), g in c.groupby(["subset_type", "subset"]):
        if (st == "all" and b != "benchmarks") or g["n_units"].max() < MIN_CLUSTERS:
            continue
        best, ref = g.loc[g["rho"].idxmax()], g[g["surrogate"] == SNR_REF]
        bench.append(dict(b=paper_name(b) if st == "benchmark" else "every benchmark", all=st == "all",
                          k=None if st == "all" else "bBPB" if b.startswith(BBPB) else "accuracy", n=int(g["n_units"].max()),
                          ref=_r(ref["rho"].iloc[0]) if len(ref) else None,
                          best=_r(best["rho"]), best_def=best["surrogate"].split("__")[1:],
                          q=None if pd.isna(best["q"]) else float(f"{best['q']:.2g}")))
    return {"surrogates_scoring": scoring,
            "surrogates_points": dict(tasks=meta, names={f: paper_name(f) for f in fams},
                                      columns=["i", "proxy", "snr", "signal", "noise", "early", "da"], data=rows),
            "surrogates_variants": dict(variants=variants, wins=wins, n_languages=int(len(fam))),
            "surrogates_bench": sorted(bench, key=lambda r: -r["best"])}
