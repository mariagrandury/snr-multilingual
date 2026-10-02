"""Copy the paper's two figures from the analysis.

Every figure the paper embeds is produced by an rqNN script under
src/signal-and-noise/analysis/ (run_all_predictivity.sh runs them all); this
step only copies the selected files here, so no analysis code lives in the
paper folder and a figure can never be newer than the table it came from.

The copy renames the analysis's descriptive stem to the paper's name (rqN for
the main text, app_* for the appendix). The .tex carries the source path as a
comment next to the \\includegraphics.

    rq0  INCLUDE vs its RF twin <- rq01_scaling_predictability/tokens_seen.py --paper (pool predictivity_all)
                             pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical
    rq1  scaling regimes  <- rq01_scaling_predictability/regimes.py  (pool predictivity_all)
                             scaling_regimes_outliers_paper.png
    rq2  decision accuracy<- rq02_decision_accuracy/paper_rq2.py     (pool predictivity, --axes mono-axis)
                             rq2_above_66_either_transformation_mono_axis.png
    app_chance_share      <- rq00_gate_and_curves/panels.py --paper    first_size_share_paper
    app_chance_full       <- the same                                  first_size_above_random_paper
    app_chance_reformulation <- rq00_task_reformulation/reformulations_gate.py --paper  reformulations_gate_paper
    app_da_goal_multi_axes_bpb <- rq01_scaling_predictability/tokens_seen.py --paper  da_goal_multi_axes_across_langs_bpb_paper

The rq1 and rq2 copies are PNG; the others PNG and SVG.

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

GATE = ANALYSIS / "rq00_gate_and_curves" / "pretraining" / "predictivity"
RQ01 = ANALYSIS / "rq01_scaling_predictability" / "pretraining" / "predictivity_all"
# paper stem -> (the analysis stem it is a copy of, the formats copied)
FIGURES = {
    "rq0": (RQ01 / "pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical", ("png", "svg")),
    "rq1": (RQ01 / "scaling_regimes_outliers_paper", ("png",)),
    "rq2": (ANALYSIS / "rq02_decision_accuracy" / "pretraining" / "predictivity"
            / "rq2_above_66_either_transformation_mono_axis", ("png",)),
    "app_chance_share": (GATE / "first_size_share_paper", ("png", "svg")),
    "app_chance_full": (GATE / "first_size_above_random_paper", ("png", "svg")),
    "app_chance_reformulation": (ANALYSIS / "rq00_task_reformulation" / "reformulations_gate_paper", ("png", "svg")),
    "app_da_goal_multi_axes_bpb": (RQ01 / "da_goal_multi_axes_across_langs_bpb_paper", ("png", "svg")),
}


def main() -> int:
    missing, copied = [], 0
    for stem, (src, exts) in FIGURES.items():
        for ext in exts:
            path = src.with_suffix(f".{ext}")
            if not path.is_file():
                missing.append(str(path.relative_to(REPO)))
                continue
            dst = HERE / f"{stem}.{ext}"
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
