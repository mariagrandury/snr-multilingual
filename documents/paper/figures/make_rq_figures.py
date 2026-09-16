"""Collect the paper's figures and tables from the analysis.

Every figure the paper embeds is produced by an rqNN script under
src/signal-and-noise/analysis/ (run_all_predictivity.sh runs them all); this
step only copies the selected files here, so no analysis code lives in the
paper folder and a figure can never be newer than the table it came from.

    RQ1 scaling        rq1_scaling        <- rq07_scaling_predictability (pool predictivity_all)
    RQ2 prediction     rq2_early_small    <- rq08_early_decision         (pool predictivity_all)
    RQ3 surrogates     rq3_surrogates     <- rq09_surrogates             (pool predictivity)
    RQ4 interventions  rq4_interventions  <- rq06_proxy_predictivity     (pool predictivity_all)
    RQ5 transfer       rq5_transfer       <- rq10_language_transfer      (pool predictivity_all)
    fig1-fig4          <- analysis/report_figures/figures (make_figures.py)

Each source's facts.json is merged into rq_facts.json, the file the analysis
section's numbers are checked against. A file whose bytes did not change is
not rewritten (the drivers render PDFs with SOURCE_DATE_EPOCH=0, so an
unchanged figure has unchanged bytes).

    python make_rq_figures.py
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ANALYSIS = REPO / "src" / "signal-and-noise" / "analysis"
ALL, HEAD = "pretraining/predictivity_all", "pretraining/predictivity"

# source directory -> (figure stems, table names)
SOURCES = {
    ANALYSIS / "rq07_scaling_predictability" / ALL: (["rq1_scaling"], ["rq1_fits.csv", "rq1_families.csv"]),
    ANALYSIS / "rq08_early_decision" / ALL: (["rq2_early_small"], ["rq2_decisions.csv", "rq2_early_small.csv"]),
    ANALYSIS / "rq09_surrogates" / HEAD: (["rq3_surrogates"], ["rq3_surrogates.csv"]),
    ANALYSIS / "rq06_proxy_predictivity" / ALL: (["rq4_interventions"],
                                                 ["rq4_effect_vs_seed.csv", "rq4_da_by_intervention.csv"]),
    ANALYSIS / "rq10_language_transfer" / ALL: (["rq5_transfer"], ["rq5_transfer.csv", "rq5_transfer_summary.csv"]),
    ANALYSIS / "report_figures" / "figures": (["fig1_gate", "fig2_snr_vs_da", "fig3_reliability_map",
                                              "fig4_subset_sweep"], []),
}


def main() -> int:
    facts, missing, copied = {}, [], 0
    for src, (figs, tables) in SOURCES.items():
        exts = ("pdf",) if src.name == "figures" else ("pdf", "png")   # make_figures.py writes PDFs only
        names = [f"{f}.{ext}" for f in figs for ext in exts] + tables
        for name in names:
            path = src / name
            if not path.is_file():
                missing.append(str(path.relative_to(REPO)))
                continue
            dst = HERE / name
            if dst.is_file() and dst.read_bytes() == path.read_bytes():
                continue                          # unchanged bytes: leave mtime alone
            shutil.copy2(path, dst)
            copied += 1
        fj = src / "facts.json"
        if fj.is_file():
            facts.update(json.loads(fj.read_text()))
    if facts:
        (HERE / "rq_facts.json").write_text(json.dumps(facts, indent=1, default=str))
    print(f"copied {copied} changed files into {HERE.relative_to(REPO)}; facts: {sorted(facts)}")
    for m in missing:
        print(f"  MISSING {m}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
