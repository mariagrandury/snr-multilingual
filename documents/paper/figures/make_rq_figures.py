"""Collect the paper's figures and tables from the analysis.

Every figure the paper embeds is produced by an rqNN script under
src/signal-and-noise/analysis/ (run_all_predictivity.sh runs them all); this
step only copies the selected files here, so no analysis code lives in the
paper folder and a figure can never be newer than the table it came from.

    RQ1 scaling        rq1_scaling        <- rq01_scaling_predictability/analyze.py      (pool predictivity_all)
    RQ2 prediction     rq2_early_small    <- rq05_design_decisions/early_decision.py     (pool predictivity_all)
    RQ3 surrogates     rq3_surrogates     <- rq04_surrogates/analyze.py                  (pool predictivity)
    RQ4 interventions  rq4_interventions  <- rq05_design_decisions/analyze.py            (pool predictivity_all)
    RQ5 transfer       rq5_transfer       <- rq06_language_transfer/analyze.py           (pool predictivity_all)
    fig1-fig4          <- analysis/report_figures/figures (make_figures.py)

Each source's facts file is merged into rq_facts.json, the file the analysis
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

# (source directory, figure stems, table names, facts file). A list, not a dict:
# one analysis folder can feed two of the paper's questions (rq05 holds both the
# interventions and the early-decision read), each with its own facts file.
SOURCES = [
    (ANALYSIS / "rq01_scaling_predictability" / ALL, ["rq1_scaling"], ["rq1_fits.csv", "rq1_families.csv"], "facts.json"),
    (ANALYSIS / "rq05_design_decisions" / ALL, ["rq2_early_small"], ["rq2_decisions.csv", "rq2_early_small.csv"],
     "early_decision_facts.json"),
    (ANALYSIS / "rq04_surrogates" / HEAD, ["rq3_surrogates"], ["rq3_surrogates.csv"], "facts.json"),
    (ANALYSIS / "rq05_design_decisions" / ALL, ["rq4_interventions"],
     ["rq4_effect_vs_seed.csv", "rq4_da_by_intervention.csv"], "facts.json"),
    (ANALYSIS / "rq06_language_transfer" / ALL, ["rq5_transfer"], ["rq5_transfer.csv", "rq5_transfer_summary.csv"],
     "facts.json"),
    (ANALYSIS / "report_figures" / "figures", ["fig1_gate", "fig2_snr_vs_da", "fig3_reliability_map",
                                               "fig4_subset_sweep"], [], None),
]


def main() -> int:
    facts, missing, copied = {}, [], 0
    for src, figs, tables, facts_name in SOURCES:
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
        fj = src / facts_name if facts_name else None
        if fj is not None and fj.is_file():
            facts.update(json.loads(fj.read_text()))
    if facts:
        (HERE / "rq_facts.json").write_text(json.dumps(facts, indent=1, default=str))
    print(f"copied {copied} changed files into {HERE.relative_to(REPO)}; facts: {sorted(facts)}")
    for m in missing:
        print(f"  MISSING {m}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
