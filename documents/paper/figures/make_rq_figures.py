"""Copy the paper's figures (and one table) from the analysis.

Every figure the paper embeds is produced by an rqNN script under
src/signal-and-noise/analysis/ (run_all_predictivity.sh runs them all); this
step only copies the selected files here, so no analysis code lives in the
paper folder and a figure can never be newer than the table it came from.

The copy renames the analysis's descriptive stem to the paper's name (rqN for
the main text, app_rqNN_* for the appendix, NN the analysis folder the figure
comes from). The .tex carries the source path as a comment next to the
\\includegraphics.

    rq0  INCLUDE vs its RF twin <- rq01_scaling_predictability/tokens_seen.py --paper (pool predictivity_seeds,
                             written to rq00_chance_vs_train_tokens/)
                             pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical
    rq1  scaling regimes  <- rq01_scaling_predictability/regimes.py  (pool predictivity_seeds)
                             scaling_regimes_outliers_paper.png
    rq2  decision accuracy<- rq02_decision_accuracy/paper_rq2.py     (pool predictivity, --axes mono-axis)
                             rq2_da_all_above_66_either_transformation_mono_axis.png
    rq3  surrogates       <- rq04_surrogates/analyze.py              (pool predictivity)
                             rq3_surrogates_paper.png
    app_rq00_chance_share      <- rq00_gate_and_curves/panels.py --paper    first_size_share_paper
    app_rq00_chance_share_horizontal <- the same, first_size_share_paper_horizontal (its bars along the x axis)
    app_rq00_chance_full       <- the same                                  first_size_above_random_paper
    app_rq00_chance_reformulation <- rq00_task_reformulation/reformulations_gate.py --paper  reformulations_gate_paper
    app_rq02_da_goal_multi_axes_bpb <- rq01_scaling_predictability/tokens_seen.py --paper  da_goal_multi_axes_across_langs_bpb_paper
                             (written to rq02_da_vs_train_tokens/)
    app_rq00_chance_vs_train_tokens <- the same, rq00_chance_vs_train_tokens/      pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper
    app_rq00_benchmark_curves  <- rq00_gate_and_curves/curves.py --paper (pool predictivity_seeds)  benchmark_curves_paper
    app_rq00_benchmark_size_curves <- the same                                  benchmark_size_curves_paper
                             (these two: the extra figures of the scaling-predictability appendix page)
    app_rq00_above_random_external <- rq00_gate_and_curves/above_random_external.py  above_random_external_paper
    app_external_models   <- the same                                  above_random_external_models.tex (a LaTeX table)
    app_rq05_design_decisions  <- rq05_design_decisions/panels.py --paper (pool predictivity_seeds)  da_all_lines_mono_axis_paper
    app_rq06_language_transfer <- rq06_language_transfer/panels.py --paper (pool predictivity_seeds)  transfer_da_all_lines_mono_axis_paper
    app_rq06_cross_task_hellaswag <- rq06_language_transfer/cross_task_transfer.py  cross_task_da_size_by_language_hellaswag_all_languages_mono_axis_paper
    app_rq06_family_transfer   <- rq06_language_transfer/family_transfer.py  above_chance_untrained_by_family_paper
    app_rq06_family_lift       <- the same                                 above_chance_untrained_lift_by_subfamily_paper
                             (these three: the extra float pages of the language-transfer appendix page)
    app_rq07_external_frameworks <- rq07_external_frameworks/analyze.py --pool predictivity --paper  snr_apertus_vs_snr_allenai_paper
    app_rq08_subset_selection  <- rq08_subset_selection/panels.py --paper                gain_over_null_paper
    app_rq09_benchmark_design  <- rq09_benchmark_design/analyze.py --pool predictivity --paper  snr_per_family_ranked_paper
    app_rq09_design_da_correlation <- rq09_benchmark_design/design_da.py  design_da_size_correlation_above_66_either_mono_axis_paper
    app_rq09_design_da_by_level    <- the same                       design_da_size_by_level_above_66_either_mono_axis_paper
    app_rq09_design_da_quadrant    <- the same                       design_da_size_quadrant_mono_axis_paper (every task above chance)
    app_rq09_benchmark_characteristics <- the same                   benchmark_characteristics.tex (a LaTeX table)
                             (these four: the extra float pages of the benchmark-design appendix page)
    app_rq10_size_generalisation <- rq10_size_generalisation/reference_consistency.py --paper
                             gate_share_and_da_size_mono_axis_paper
    app_rq11_evaluation_recipe <- rq11_evaluation_recipe/recipe.py --paper               recipe_da_size_variants_multi_axes_paper
    app_rq11_recipe_by_benchmark <- the same                                    recipe_da_all_by_benchmark_multi_axes_paper
                             (the extra full-page figure of the evaluation-recipe appendix page)

    app_rq02_decision_accuracy <- rq02_decision_accuracy/scale_convergence.py --paper  scale_convergence_da_size_multi_axes_paper
    app_rq02_bbpb               <- rq02_decision_accuracy/paper_rq2.py --axes mono-axis  rq2_da_all_above_66_either_transformation_by_scoring_mono_axis
                             (rq2 in two rows: accuracy above, the bBPB twins below)
    app_rq02_da_by_language     <- the same                    rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis
    app_rq02_da_size_by_language_per_proxy <- the same         rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis
                             (these two: the extra full-page figures of the decision-accuracy appendix page)
    app_rq03_noise_and_snr     <- rq03_noise_and_snr/effect_vs_noise.py --paper (pool predictivity_seeds)  effect_vs_noise_paper
    app_rq04_surrogates        <- rq04_surrogates/snr_definition_postprocess.py --pool predictivity --paper  top_variants_overall_paper
    app_rq12_above_chance_items <- rq12_above_chance_items/above_chance_items.py --paper  above_chance_items_snr_paper
    app_rq13_english_only      <- rq13_english_only/english_only.py --paper      english_only_scores_paper
                             (these five: the main figure of an appendix page of make_rq_appendix.py)
    app_rq13_english_scaling_regimes <- rq13_english_only/english_regimes.py   english_only_scaling_regimes_paper
                             (the extra figure of the English-only appendix page)
    app_rq02_permutation_null  <- rq02_permutation_null/permutation_null.py       permutation_null_da_size_paper
    app_rq02_decisive_pairs    <- rq02_decisive_pairs/decisive_pairs.py           decisive_pairs_da_size_paper
    app_rq14_proxy_item_selection <- rq14_proxy_item_selection/proxy_item_selection.py  proxy_item_selection_da_size_multi_axes_paper
                             (these three: the main figure of an appendix page; the last needs the
                             cluster-only per-item store and is PENDING, not MISSING, until it is drawn)
    app_rq02_rq2_crossfit      <- rq02_decision_accuracy/crossfit_reliable.py  rq2_da_all_above_66_either_crossfit_transformation_mono_axis
    app_rq02_decision_accuracy_crossfit <- the same    scale_convergence_da_size_above_66_either_crossfit_multi_axes_paper
                             (these two: extra figures of the decision-accuracy appendix page)
    app_rqNN_fixed_<name>      <- the `_fixed_tasks_paper` twin of the paper figure <name> (rq2 and the app_rqNN_*
                             figures whose task set moves along x), written by the script that writes <name>
                             (rq02's by fixed_tasks.py); make_rq_appendix.py puts them in app_fixed_populations.tex

The copies are PNG, the table TeX. The SVG and PDF of every figure here are not copied:
`style.save` writes them under the paper name straight into ../figures_svg/ and ../figures_pdf/
(it reads FIGURES to know which analysis figures the paper embeds). Every
figure is a bare paper figure written through `style.save_paper` (RULES.md rule 18): a `_paper`
stem, or an rq2 stem, whose writers (paper_rq2.py, crossfit_reliable.py) draw only paper figures.

A file whose bytes did not change is not rewritten; its mtime is set to its
source's, as a copy would, so the refresh's orphan scan does not list it.

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
    "rq0": (CHANCE_TOKENS / "pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_include_rf_paper_vertical", ("png",)),
    "rq1": (RQ01 / "scaling_regimes_outliers_paper", ("png",)),
    "rq2": (ANALYSIS / "rq02_decision_accuracy" / "pretraining" / "predictivity"
            / "rq2_da_all_above_66_either_transformation_mono_axis", ("png",)),
    "rq3": (ANALYSIS.joinpath("rq04_surrogates", *PRED, "rq3_surrogates_paper"), ("png",)),
    "app_rq00_chance_share": (GATE / "first_size_share_paper", ("png",)),
    "app_rq00_chance_share_horizontal": (GATE / "first_size_share_paper_horizontal", ("png",)),
    "app_rq00_chance_full": (GATE / "first_size_above_random_paper", ("png",)),
    "app_rq00_chance_reformulation": (ANALYSIS / "rq00_task_reformulation" / "reformulations_gate_paper", ("png",)),
    "app_rq02_da_goal_multi_axes_bpb": (DA_TOKENS / "da_goal_multi_axes_across_langs_bpb_paper", ("png",)),
    "app_rq00_chance_vs_train_tokens": (CHANCE_TOKENS / "pass_prob_vs_train_tokens_by_benchmark_1904_ckpts_paper", ("png",)),
    "app_rq00_benchmark_curves": (CURVES / "benchmark_curves_paper", ("png",)),
    "app_rq00_benchmark_size_curves": (CURVES / "benchmark_size_curves_paper", ("png",)),
    "app_rq00_above_random_external": (GATE / "above_random_external_paper", ("png",)),
    "app_external_models": (GATE / "above_random_external_models", ("tex",)),
    "app_rq05_design_decisions": (ANALYSIS.joinpath("rq05_design_decisions", *SEEDS, "da_all_lines_mono_axis_paper"), ("png",)),
    "app_rq06_language_transfer": (ANALYSIS.joinpath("rq06_language_transfer", *SEEDS, "transfer_da_all_lines_mono_axis_paper"),
                              ("png",)),
    "app_rq06_cross_task_hellaswag": (ANALYSIS.joinpath("rq06_language_transfer", *PRED,
                                                         "cross_task_da_size_by_language_hellaswag_all_languages_mono_axis_paper"),
                                      ("png",)),
    "app_rq06_family_transfer": (ANALYSIS.joinpath("rq06_language_transfer", *PRED, "above_chance_untrained_by_family_paper"),
                                 ("png",)),
    "app_rq06_family_lift": (ANALYSIS.joinpath("rq06_language_transfer", *PRED, "above_chance_untrained_lift_by_subfamily_paper"),
                             ("png",)),
    "app_rq07_external_frameworks": (ANALYSIS.joinpath("rq07_external_frameworks", *PRED, "snr_apertus_vs_snr_allenai_paper"),
                                ("png",)),
    "app_rq08_subset_selection": (ANALYSIS.joinpath("rq08_subset_selection", *PRED, "gain_over_null_paper"), ("png",)),
    "app_rq09_benchmark_design": (ANALYSIS.joinpath("rq09_benchmark_design", *PRED, "snr_per_family_ranked_paper"), ("png",)),
    "app_rq09_design_da_correlation": (ANALYSIS.joinpath("rq09_benchmark_design", *PRED,
                                                          "design_da_size_correlation_above_66_either_mono_axis_paper"), ("png",)),
    "app_rq09_design_da_by_level": (ANALYSIS.joinpath("rq09_benchmark_design", *PRED,
                                                       "design_da_size_by_level_above_66_either_mono_axis_paper"), ("png",)),
    "app_rq09_design_da_quadrant": (ANALYSIS.joinpath("rq09_benchmark_design", *PRED,
                                                       "design_da_size_quadrant_mono_axis_paper"), ("png",)),
    "app_rq09_benchmark_characteristics": (ANALYSIS.joinpath("rq09_benchmark_design", *PRED, "benchmark_characteristics"),
                                           ("tex",)),
    "app_rq10_size_generalisation": (ANALYSIS.joinpath("rq10_size_generalisation", *PRED, "gate_share_and_da_size_mono_axis_paper"),
                                ("png",)),
    "app_rq11_evaluation_recipe": (ANALYSIS.joinpath("rq11_evaluation_recipe", *PRED, "recipe_da_size_variants_multi_axes_paper"),
                              ("png",)),
    "app_rq11_recipe_by_benchmark": (ANALYSIS.joinpath("rq11_evaluation_recipe", *PRED, "recipe_da_all_by_benchmark_multi_axes_paper"),
                                     ("png",)),
    # the main figures of the per-analysis appendix pages (make_rq_appendix.py)
    "app_rq02_bbpb": (ANALYSIS / "rq02_decision_accuracy" / "pretraining" / "predictivity"
                      / "rq2_da_all_above_66_either_transformation_by_scoring_mono_axis", ("png",)),
    "app_rq02_decision_accuracy": (ANALYSIS.joinpath("rq02_decision_accuracy", *PRED, "scale_convergence_da_size_multi_axes_paper"),
                              ("png",)),
    "app_rq02_da_by_language": (ANALYSIS.joinpath("rq02_decision_accuracy", *PRED, "rq2_da_all_by_benchmark_and_language_above_66_either_mono_axis"),
                                ("png",)),
    "app_rq02_da_size_by_language_per_proxy": (ANALYSIS.joinpath("rq02_decision_accuracy", *PRED,
                                                                 "rq2_da_size_by_benchmark_and_language_per_proxy_above_66_either_mono_axis"),
                                               ("png",)),
    "app_rq03_noise_and_snr": (ANALYSIS.joinpath("rq03_noise_and_snr", *SEEDS, "effect_vs_noise_paper"), ("png",)),
    "app_rq04_surrogates": (ANALYSIS.joinpath("rq04_surrogates", *PRED, "top_variants_overall_paper"), ("png",)),
    "app_rq12_above_chance_items": (ANALYSIS.joinpath("rq12_above_chance_items", *PRED, "above_chance_items_snr_paper"),
                               ("png",)),
    "app_rq13_english_only": (ANALYSIS.joinpath("rq13_english_only", *PRED, "english_only_scores_paper"), ("png",)),
    "app_rq13_english_scaling_regimes": (ANALYSIS.joinpath("rq13_english_only", *PRED, "english_only_scaling_regimes_paper"),
                                         ("png",)),
    "app_rq02_permutation_null": (ANALYSIS.joinpath("rq02_permutation_null", *PRED, "permutation_null_da_size_paper"), ("png",)),
    "app_rq02_decisive_pairs": (ANALYSIS.joinpath("rq02_decisive_pairs", *PRED, "decisive_pairs_da_size_paper"), ("png",)),
    "app_rq14_proxy_item_selection": (ANALYSIS.joinpath("rq14_proxy_item_selection", *PRED,
                                                        "proxy_item_selection_da_size_multi_axes_paper"), ("png",)),
    # extra figures of an appendix page
    "app_rq02_rq2_crossfit": (ANALYSIS.joinpath("rq02_decision_accuracy", *PRED,
                                                "rq2_da_all_above_66_either_crossfit_transformation_mono_axis"), ("png",)),
    "app_rq02_decision_accuracy_crossfit": (ANALYSIS.joinpath(
        "rq02_decision_accuracy", *PRED, "scale_convergence_da_size_above_66_either_crossfit_multi_axes_paper"), ("png",)),
    # the fixed-task twins (one task set per line, rule 13) of the appendix "Fixed task sets"
    "app_rq02_fixed_rq2": (ANALYSIS.joinpath("rq02_decision_accuracy", *PRED,
                                             "rq2_da_all_above_66_either_transformation_mono_axis_fixed_tasks_paper"), ("png",)),
    "app_rq02_fixed_decision_accuracy": (ANALYSIS.joinpath("rq02_decision_accuracy", *PRED,
                                                           "scale_convergence_da_size_multi_axes_fixed_tasks_paper"), ("png",)),
    "app_rq05_fixed_design_decisions": (ANALYSIS.joinpath("rq05_design_decisions", *SEEDS,
                                                          "da_all_lines_mono_axis_fixed_tasks_paper"), ("png",)),
    "app_rq10_fixed_size_generalisation": (ANALYSIS.joinpath("rq10_size_generalisation", *PRED,
                                                             "gate_share_and_da_size_mono_axis_fixed_tasks_paper"), ("png",)),
    "app_rq11_fixed_evaluation_recipe": (ANALYSIS.joinpath("rq11_evaluation_recipe", *PRED,
                                                           "recipe_da_size_variants_multi_axes_fixed_tasks_paper"), ("png",)),
    "app_rq12_fixed_above_chance_items": (ANALYSIS.joinpath("rq12_above_chance_items", *PRED,
                                                            "above_chance_items_snr_fixed_tasks_paper"), ("png",)),
    "app_rq13_fixed_english_only": (ANALYSIS.joinpath("rq13_english_only", *PRED, "english_only_scores_fixed_tasks_paper"),
                                    ("png",)),
}
# drawn only from the cluster-only per-item store: absent until that run, not an error
PENDING = {"app_rq14_proxy_item_selection"}


def main() -> int:
    missing, copied = [], 0
    for stem, (src, exts) in FIGURES.items():
        for ext in exts:
            path = src.with_suffix(f".{ext}")
            if not path.is_file():
                if stem in PENDING:
                    print(f"  PENDING {path.relative_to(REPO)} (needs the per-item store)")
                else:
                    missing.append(str(path.relative_to(REPO)))
                continue
            dst = HERE / f"{stem}.{ext}"
            if dst.is_file() and dst.read_bytes() == path.read_bytes():
                shutil.copystat(path, dst)        # unchanged bytes: not rewritten, but its mtime follows the source's
                continue                          # (as copy2 sets it), so the refresh's orphan scan sees it as written
            shutil.copy2(path, dst)
            copied += 1
    print(f"copied {copied} changed files into {HERE.relative_to(REPO)}")
    for m in missing:
        print(f"  MISSING {m}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
