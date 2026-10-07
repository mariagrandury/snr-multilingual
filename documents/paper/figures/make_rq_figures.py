"""Copy the paper's figures (and one table) from the analysis.

Every figure the paper embeds is produced by an rqNN script under
src/signal-and-noise/analysis/ (run_all_predictivity.sh runs them all); this
step only copies the selected files here, so no analysis code lives in the
paper folder and a figure can never be newer than the table it came from.

The copy renames the analysis's descriptive stem to the paper's name (rqN for
the main text, app_* for the appendix). The .tex carries the source path as a
comment next to the \\includegraphics.

    rq0  INCLUDE vs its RF twin <- rq01_scaling_predictability/tokens_seen.py --paper (pool predictivity_seeds,
                             written to rq00_chance_vs_train_tokens/)
                             pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical
    rq1  scaling regimes  <- rq01_scaling_predictability/regimes.py  (pool predictivity_seeds)
                             scaling_regimes_outliers_paper.png
    rq2  decision accuracy<- rq02_decision_accuracy/paper_rq2.py     (pool predictivity, --axes mono-axis)
                             rq2_da_all_above_66_either_transformation_mono_axis.png
    rq3  surrogates       <- rq04_surrogates/analyze.py              (pool predictivity)
                             rq3_surrogates_paper.png
    app_chance_share      <- rq00_gate_and_curves/panels.py --paper    first_size_share_paper
    app_chance_full       <- the same                                  first_size_above_random_paper
    app_chance_reformulation <- rq00_task_reformulation/reformulations_gate.py --paper  reformulations_gate_paper
    app_da_goal_multi_axes_bpb <- rq01_scaling_predictability/tokens_seen.py --paper  da_goal_multi_axes_across_langs_bpb_paper
                             (written to rq02_da_vs_train_tokens/)
    app_chance_vs_train_tokens <- the same, rq00_chance_vs_train_tokens/      pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper
    app_benchmark_curves  <- rq00_gate_and_curves/curves.py --paper (pool predictivity_seeds)  benchmark_curves_paper
    app_benchmark_size_curves <- the same                                  benchmark_size_curves_paper
                             (these two: the extra figures of the scaling-predictability appendix page)
    app_above_random_external <- rq00_gate_and_curves/above_random_external.py  above_random_external_paper
    app_external_models   <- the same                                  above_random_external_models.tex (a LaTeX table)
    app_design_decisions  <- rq05_design_decisions/panels.py --paper (pool predictivity_seeds)  da_all_lines_mono_axis_paper
    app_language_transfer <- rq06_language_transfer/panels.py --paper (pool predictivity_seeds)  transfer_da_all_lines_mono_axis_paper
    app_external_frameworks <- rq07_external_frameworks/analyze.py --pool predictivity --paper  snr_apertus_vs_snr_allenai_paper
    app_subset_selection  <- rq08_subset_selection/panels.py --paper                gain_over_null_paper
    app_benchmark_design  <- rq09_benchmark_design/analyze.py --pool predictivity --paper  snr_per_family_ranked_paper
    app_size_generalisation <- rq10_size_generalisation/reference_consistency.py --paper
                             gate_share_and_da_size_mono_axis_paper
    app_evaluation_recipe <- rq11_evaluation_recipe/recipe.py --paper               recipe_da_size_variants_multi_axes_paper

    app_decision_accuracy <- rq02_decision_accuracy/scale_convergence.py --paper  scale_convergence_da_size_multi_axes_paper
    app_noise_and_snr     <- rq03_noise_and_snr/effect_vs_noise.py --paper (pool predictivity_seeds)  effect_vs_noise_paper
    app_surrogates        <- rq04_surrogates/snr_definition_postprocess.py --pool predictivity --paper  top_variants_overall_paper
    app_above_chance_items <- rq12_above_chance_items/above_chance_items.py --paper  above_chance_items_snr_paper
    app_english_only      <- rq13_english_only/english_only.py --paper      english_only_scores_paper
                             (these five: the main figure of an appendix page of make_rq_appendix.py)

The rq1, rq2 and rq3 copies and the last five are PNG, the table TeX; the others PNG and SVG. Every
figure is a bare paper figure written through `style.save_paper` (RULES.md rule 18): a `_paper`
stem, or rq2, whose writer paper_rq2.py draws only paper figures.

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
RQ01 = ANALYSIS / "rq01_scaling_predictability" / "pretraining" / "predictivity_seeds"
CHANCE_TOKENS = ANALYSIS / "rq00_chance_vs_train_tokens" / "pretraining" / "predictivity_seeds"
DA_TOKENS = ANALYSIS / "rq02_da_vs_train_tokens" / "pretraining" / "predictivity_seeds"
CURVES = ANALYSIS / "rq00_gate_and_curves" / "pretraining" / "predictivity_seeds"
PRED, SEEDS = ("pretraining", "predictivity"), ("pretraining", "predictivity_seeds")
# paper stem -> (the analysis stem it is a copy of, the formats copied)
FIGURES = {
    "rq0": (CHANCE_TOKENS / "pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical", ("png", "svg")),
    "rq1": (RQ01 / "scaling_regimes_outliers_paper", ("png",)),
    "rq2": (ANALYSIS / "rq02_decision_accuracy" / "pretraining" / "predictivity"
            / "rq2_da_all_above_66_either_transformation_mono_axis", ("png",)),
    "rq3": (ANALYSIS.joinpath("rq04_surrogates", *PRED, "rq3_surrogates_paper"), ("png",)),
    "app_chance_share": (GATE / "first_size_share_paper", ("png", "svg")),
    "app_chance_full": (GATE / "first_size_above_random_paper", ("png", "svg")),
    "app_chance_reformulation": (ANALYSIS / "rq00_task_reformulation" / "reformulations_gate_paper", ("png", "svg")),
    "app_da_goal_multi_axes_bpb": (DA_TOKENS / "da_goal_multi_axes_across_langs_bpb_paper", ("png", "svg")),
    "app_chance_vs_train_tokens": (CHANCE_TOKENS / "pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper", ("png", "svg")),
    "app_benchmark_curves": (CURVES / "benchmark_curves_paper", ("png", "svg")),
    "app_benchmark_size_curves": (CURVES / "benchmark_size_curves_paper", ("png", "svg")),
    "app_above_random_external": (GATE / "above_random_external_paper", ("png", "svg")),
    "app_external_models": (GATE / "above_random_external_models", ("tex",)),
    "app_design_decisions": (ANALYSIS.joinpath("rq05_design_decisions", *SEEDS, "da_all_lines_mono_axis_paper"), ("png", "svg")),
    "app_language_transfer": (ANALYSIS.joinpath("rq06_language_transfer", *SEEDS, "transfer_da_all_lines_mono_axis_paper"),
                              ("png", "svg")),
    "app_external_frameworks": (ANALYSIS.joinpath("rq07_external_frameworks", *PRED, "snr_apertus_vs_snr_allenai_paper"),
                                ("png", "svg")),
    "app_subset_selection": (ANALYSIS.joinpath("rq08_subset_selection", *PRED, "gain_over_null_paper"), ("png", "svg")),
    "app_benchmark_design": (ANALYSIS.joinpath("rq09_benchmark_design", *PRED, "snr_per_family_ranked_paper"), ("png", "svg")),
    "app_size_generalisation": (ANALYSIS.joinpath("rq10_size_generalisation", *PRED, "gate_share_and_da_size_mono_axis_paper"),
                                ("png", "svg")),
    "app_evaluation_recipe": (ANALYSIS.joinpath("rq11_evaluation_recipe", *PRED, "recipe_da_size_variants_multi_axes_paper"),
                              ("png", "svg")),
    # the main figures of the per-analysis appendix pages (make_rq_appendix.py)
    "app_decision_accuracy": (ANALYSIS.joinpath("rq02_decision_accuracy", *PRED, "scale_convergence_da_size_multi_axes_paper"),
                              ("png",)),
    "app_noise_and_snr": (ANALYSIS.joinpath("rq03_noise_and_snr", *SEEDS, "effect_vs_noise_paper"), ("png",)),
    "app_surrogates": (ANALYSIS.joinpath("rq04_surrogates", *PRED, "top_variants_overall_paper"), ("png",)),
    "app_above_chance_items": (ANALYSIS.joinpath("rq12_above_chance_items", *PRED, "above_chance_items_snr_paper"),
                               ("png",)),
    "app_english_only": (ANALYSIS.joinpath("rq13_english_only", *PRED, "english_only_scores_paper"), ("png",)),
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
