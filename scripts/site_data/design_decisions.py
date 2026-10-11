"""Data of the design-decisions page's additional views (rq05_design_decisions).

Per (decision, population, proxy size, checkpoint) the decision accuracy
against the 1.7B final, mean over L: rq05's `intervention_da_all_mono_axis`
re-keyed per data build by rq05's own `by_recipe` (so the final column is
`rq4_da_size_by_intervention_mono_axis`). It holds all six decisions, the
activation (xIELU vs SwiGLU) and optimizer (AdEMAMix vs Muon) ones included
since 2026-10-10. The optimizer has no 1.7B evaluation yet, so it has no row
until a refresh brings one (rq05 lists it in `facts.json` as
`pending_reference`), and meanwhile the page shows rq02's DA-ckpt for the
activation and optimizer axes (each proxy against its own final,
`early_small_da_ckpt_by_transformation_mono_axis`). The effect at the
reference in seed sds is `rq4_interventions.csv`, the median over L of the
per-L medians, as rq05's README quotes it. Rows only filtered, averaged as
rq05 does, and renamed."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
sys.path.insert(0, str(REPO / "src"))
from analysis import paths  # noqa: E402
from analysis.rq05_design_decisions.analyze import INTERVENTIONS, by_recipe  # noqa: E402

ALL = "pretraining/predictivity_seeds"
PRED = "pretraining/predictivity"
POPS = {"benchmark": "benchmark", "bpb_trained": "bpb_trained"}
EFFECT_POP = {"benchmarks": "benchmark", "bits per byte": "bpb_trained"}
# Plain-language row names; the analysis label stays in `names[key].long`.
SHORT = {
    "arch": "Depth: deep vs shallow",
    "scheme_B": "Language list: A vs B (L8–L30)",
    "scheme_B:ZH": "Second language: Russian vs Chinese (L2)",
    "scheme_C:ES": "Second language: Russian vs Spanish (L2)",
    "scheme_B:DCLMP": "English data: DCLM vs DCLM without the edu filter (L1)",
    "scheme_C:FWEB": "English data: DCLM vs FineWeb (L1)",
    "temperature": "Sampling temperature: T = 1 vs T = 3 (L15–L50)",
    "activation": "Activation: xIELU vs SwiGLU (L8–L30)",
    "optimizer": "Optimizer: AdEMAMix vs Muon (L8–L30)",
}
RQ02_AXIS = {"activation (XIELU vs SwiGLU)": "activation", "optimizer": "optimizer"}


def build() -> dict:
    d5 = paths.DESIGN_DECISIONS / ALL
    da = by_recipe(pd.read_csv(d5 / "intervention_da_all_mono_axis.csv"))
    da = da[da["population"].isin(POPS)]
    g = (da.groupby(["intervention", "label", "population", "proxy_size", "frac"], as_index=False)
           .agg(da=("decision_acc", "mean"), cells=("L", "nunique"), items=("n_items", "sum"),
                size_matched=("size_matched", "all")))
    g["source"] = "rq05"
    rows = g
    names = {k: dict(short=SHORT[k], long=lab) for k, lab in zip(rows["intervention"], rows["label"])}
    for k in ("activation", "optimizer"):      # listed even before their 1.7B evaluations land
        names.setdefault(k, dict(short=SHORT[k], long=INTERVENTIONS[k][0]))

    d2 = paths.DECISION_ACCURACY / PRED

    ckpt = pd.read_csv(d2 / "early_small_da_ckpt_by_transformation_mono_axis.csv")
    ckpt = ckpt[ckpt["axis"].isin(RQ02_AXIS)].assign(intervention=lambda x: x["axis"].map(RQ02_AXIS))

    eff = pd.read_csv(d5 / "rq4_interventions.csv")
    eff["population"] = eff["population"].map(EFFECT_POP)
    med = (eff.groupby(["intervention", "population"], as_index=False)
              .agg(ratio=("median_effect_over_seed_sd", "median"), Ls=("L", list),
                   per_L=("median_effect_over_seed_sd", list), share_above_2=("share_above_2", list)))

    def r(x):
        return round(float(x), 4)
    return {"design_decisions_ladder": dict(
        names=names,
        rows=[dict(key=x.intervention, pop=x.population, size=x.proxy_size,
                   frac=r(x.frac), da=r(x.da), items=int(x.items), size_matched=bool(x.size_matched),
                   source=x.source, cells=int(x.cells))
              for x in rows.itertuples()],
        ckpt=[dict(key=x.intervention, size=x.proxy_size, frac=r(x.frac), da=r(x.da), tasks=int(x.tasks))
              for x in ckpt.itertuples()],
        effect=[dict(key=x.intervention, pop=x.population, ratio=r(x.ratio), Ls=[int(L) for L in x.Ls],
                     per_L=[r(v) for v in x.per_L], share_above_2=[r(v) for v in x.share_above_2])
                for x in med.itertuples()],
    )}
