"""Copy the paper's two figures from the analysis.

Every figure the paper embeds is produced by an rqNN script under
src/signal-and-noise/analysis/ (run_all_predictivity.sh runs them all); this
step only copies the selected files here, so no analysis code lives in the
paper folder and a figure can never be newer than the table it came from.

The paper includes each one as figures/rqN.png, so the copy renames the
analysis's descriptive stem to the paper's number. The .tex carries the source
path as a comment next to the \\includegraphics.

    rq1  scaling regimes  <- rq01_scaling_predictability/regimes.py  (pool predictivity_all)
                             scaling_regimes_outliers_paper.png
    rq2  decision accuracy<- rq02_decision_accuracy/paper_rq2.py     (pool predictivity, --axes mono-axis)
                             rq2_above_66_either_transformation_one_axis.png

A file whose bytes did not change is not rewritten, so an unchanged figure
keeps its mtime.

    python make_rq_figures.py
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ANALYSIS = REPO / "src" / "signal-and-noise" / "analysis"

# paper stem -> the analysis file it is a copy of
FIGURES = {
    "rq1": ANALYSIS / "rq01_scaling_predictability" / "pretraining" / "predictivity_all"
           / "scaling_regimes_outliers_paper.png",
    "rq2": ANALYSIS / "rq02_decision_accuracy" / "pretraining" / "predictivity"
           / "rq2_above_66_either_transformation_one_axis.png",
}


def main() -> int:
    missing, copied = [], 0
    for stem, path in FIGURES.items():
        if not path.is_file():
            missing.append(str(path.relative_to(REPO)))
            continue
        dst = HERE / f"{stem}.png"
        if dst.is_file() and dst.read_bytes() == path.read_bytes():
            continue                          # unchanged bytes: leave mtime alone
        shutil.copy2(path, dst)
        copied += 1
    print(f"copied {copied} changed files into {HERE.relative_to(REPO)}")
    for m in missing:
        print(f"  MISSING {m}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
