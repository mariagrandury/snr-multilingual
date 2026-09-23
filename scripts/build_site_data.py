"""Export the committed analysis tables as the compact JSON the site's
interactive views read (docs/interactive/data/). No statistic is computed
here — every number is a row of an rqNN_ output, only filtered and renamed.

    python3 scripts/build_site_data.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src/signal-and-noise"))
from analysis import paths  # noqa: E402

OUT = REPO / "docs/interactive/data"
PRED = "pretraining/predictivity"
ALL = "pretraining/predictivity_all"


def rows(df: pd.DataFrame) -> list[dict]:
    """Records with floats rounded to 4 digits and NaN dropped."""
    return [{k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in r.items() if not (isinstance(v, float) and v != v)}
            for r in df.to_dict("records")]


def table(df: pd.DataFrame) -> dict:
    """Columnar form for the large tables (keys are most of a record's bytes)."""
    df = df.round(4).astype(object).where(df.notna(), None)
    return dict(columns=list(df.columns), data=df.values.tolist())


def dump(name: str, obj) -> None:
    (OUT / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":")))
    print(f"{name}.json  {(OUT / f'{name}.json').stat().st_size // 1024} KB")


def ladder() -> dict:
    models = json.loads((REPO / "configs/models.json").read_text())["models"]
    runs = pd.read_csv(paths.GATE_AND_CURVES / PRED / "above_random_runs.csv", usecols=["model"])
    benchmarked = set(runs.model)
    cells = [dict(name=k, size=m["size"], params=m["params"], n_non_emb=m["n_non_emb"],
                  d_model=m["d_model"], L=m["L"], arch=m["arch"], scheme=m["scheme"],
                  T=m["temperature"], seed=m["seed"],
                  tokens=m["stages"]["pretraining"]["tokens"],
                  n_ckpts=len(m["stages"]["pretraining"]["checkpoints"]["all"]),
                  benchmarks=k in benchmarked)
             for k, m in models.items() if k.startswith("lm-")]
    bpb = pd.read_csv(paths.LANGUAGE_TRANSFER / ALL / "bpb_curves.csv")
    final = bpb[bpb.frac == bpb.groupby("model").frac.transform("max")]
    final_bpb = {m: dict(zip(g.task.str.removeprefix("bpb_"), g.primary_score.round(4)))
                 for m, g in final.groupby("model")}
    sets = {}
    for f in sorted((REPO / "src/pretrain/data").glob("language_sets_scheme*.json")):
        d = json.loads(f.read_text())
        sets[d["scheme"]] = d["sets"]
    langs = json.loads((REPO / "configs/languages.json").read_text())
    return dict(cells=cells, final_bpb=final_bpb, language_sets=sets,
                languages=langs["languages"], fineweb_iso2=langs["fineweb_iso2"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = REPO / "documents/paper/figures"
    dump("ladder", ladder())
    dump("facts", json.loads((REPO / "documents/ladder-facts.json").read_text()))

    share = pd.read_csv(paths.GATE_AND_CURVES / PRED / "above_random_share.csv")
    sizes = ["175M", "350M", "600M", "1B", "1.7B"]
    dump("gate", dict(
        sizes=sizes,
        tasks=table(share[["task", "family", "language", "n_options", "random_baseline", *sizes]]),
        reformulation=rows(pd.read_csv(paths.GATE_AND_CURVES.parent / "rq00_task_reformulation/rf_gate.csv"))))

    dump("scaling", dict(
        regimes=rows(pd.read_csv(paths.SCALING_PREDICTABILITY / ALL / "scaling_regimes.csv")),
        families=rows(pd.read_csv(paths.SCALING_PREDICTABILITY / ALL / "rq1_families.csv"))))

    da = pd.read_csv(paths.DECISION_ACCURACY / PRED / "da_per_benchmark.csv")
    dump("decision_accuracy", dict(
        early_small=rows(pd.read_csv(paths.DECISION_ACCURACY / PRED / "early_small_summary.csv")),
        per_task=table(da[["language", "benchmark", "task", "da_def", "comparison", "decision_acc"]])))

    dump("snr", dict(
        per_task=table(pd.read_csv(paths.NOISE_AND_SNR / PRED / "snr.csv")),
        effect_over_seed=table(pd.read_csv(paths.NOISE_AND_SNR / ALL / "effect_over_seed.csv"))))

    dump("surrogates", dict(
        rho=rows(pd.read_csv(fig / "rq3_surrogates.csv")),
        best_variant=rows(pd.read_csv(paths.SURROGATES / PRED / "best_variant_per_language.csv"))))

    dump("design_decisions", dict(
        da=rows(pd.read_csv(paths.DESIGN_DECISIONS / ALL / "rq4_da_by_intervention.csv")),
        effect=rows(pd.read_csv(paths.DESIGN_DECISIONS / ALL / "rq4_effect_vs_seed.csv")),
        early=rows(pd.read_csv(paths.DESIGN_DECISIONS / ALL / "rq2_early_small.csv"))))

    dump("transfer", dict(
        summary=rows(pd.read_csv(paths.LANGUAGE_TRANSFER / ALL / "rq5_transfer_summary.csv")),
        per_language=table(pd.read_csv(paths.LANGUAGE_TRANSFER / ALL / "rq5_transfer.csv",
                                      usecols=["L", "task", "trained", "reference_size", "k", "observed",
                                               "pred_transfer", "pred_last", "err_transfer", "err_last"]))))

    ext = paths.EXTERNAL_FRAMEWORKS / "all/external"
    dump("external", dict(
        ours=rows(pd.read_csv(ext / "top_apertus.csv")), allenai=rows(pd.read_csv(ext / "top_allenai.csv")),
        agreement=rows(pd.read_csv(ext / "shared_task_agreement.csv")),
        per_variant=rows(pd.read_csv(ext / "pearson_r_per_variant.csv"))))

    dump("subsets", rows(pd.read_csv(paths.SUBSET_SELECTION / "all/external/summary.csv")))
    dump("benchmark_design", rows(pd.read_csv(paths.BENCHMARK_DESIGN / PRED / "per_family_snr.csv")))


if __name__ == "__main__":
    main()
