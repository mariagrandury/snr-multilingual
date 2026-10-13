"""Data of the external page's additional view (rq07_external_frameworks, the
ladder's headline pool `predictivity`): per SNR variant and size pair, the SNR
of every shared English task on our ladder and on DataDecide, read with rq07's
own task reconciliation (`_load_apertus_with_alias`) from the tables rq07
correlates (rq03's `snr_variants_per_task.csv`, `allenai_snr_variants_per_task.csv`),
plus rq07's Pearson r per (variant, size pair) and its headline row. Values
only selected and renamed; the view ranks them."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
sys.path.insert(0, str(REPO / "src"))
from analysis import paths  # noqa: E402
from analysis.grids import paper_name  # noqa: E402
from analysis.rq07_external_frameworks.analyze import (  # noqa: E402
    ALLENAI_CSV, SIZE_PAIRS, VARIANT_TITLES, VARIANTS, _load_apertus_with_alias)

POOL = "pretraining/predictivity"


def build() -> dict:
    out = paths.EXTERNAL_FRAMEWORKS / POOL
    ours = _load_apertus_with_alias(paths.NOISE_AND_SNR / POOL / "snr_variants_per_task.csv")
    theirs = pd.read_csv(ALLENAI_CSV, index_col="task")
    overlap = pd.read_csv(out / "task_overlap.csv")
    shared = overlap.loc[overlap["shared"], "task"].tolist()
    sweep = pd.read_csv(out / "pearson_r_size_sweep.csv")
    head = pd.read_csv(out / "shared_task_agreement.csv").iloc[0]

    snr = []
    for v in VARIANTS:
        for a, b in SIZE_PAIRS:
            ca, cb = f"snr_{v}_{a}", f"snr_{v}_{b}"
            if ca not in ours or cb not in theirs:
                continue
            for t in shared:
                x, y = pd.to_numeric(ours.at[t, ca], errors="coerce"), pd.to_numeric(theirs.at[t, cb], errors="coerce")
                if pd.notna(x) and pd.notna(y) and x > 0 and y > 0:
                    snr.append(dict(variant=v, ours=a, theirs=b, task=t, a=round(float(x), 4), b=round(float(y), 4)))
    r = [dict(variant=x.variant, ours=x.apertus_size, theirs=x.allenai_size, n=int(x.n),
              r=None if pd.isna(x.r) else round(float(x.r), 3), p=None if pd.isna(x.p) else round(float(x.p), 4))
         for x in sweep.itertuples()]
    names = {t: paper_name(t) for t in shared}
    names.update(arc_easy="ARC Easy", arc_challenge="ARC Challenge", csqa="CommonsenseQA",
                 hellaswag="HellaSwag", mmlu="MMLU", openbookqa="OpenBookQA",
                 piqa="PIQA", winogrande="Winogrande", boolq="BoolQ")   # the paper figure's names
    return {"external_ranks": dict(
        snr=snr, r=r, names=names, titles={v: VARIANT_TITLES[v] for v in VARIANTS},
        pairs=[f"{a}|{b}" for a, b in SIZE_PAIRS],
        headline=dict(variant=head["variant"], n=int(head["n_shared"]),
                      pearson=round(float(head["pearson_log_snr"]), 3), spearman=round(float(head["spearman_rank"]), 3)))}
