"""Per-research-question output roots — the one place that knows the rqNN_
directory names.

Each analysis script writes its artifacts under ``<its RQ dir>/<stage>/<pool>/``
and reads sibling RQs' artifacts via these constants, so the numbered directory
names live in exactly one module.

The numbering follows the four themes of the study: A, is the evaluation
predictable (rq00–rq02); B, can it be measured cheaply (rq03–rq04); C, does the
framework generalise (rq05–rq07); D, can the benchmarks be improved (rq08–rq09).
"""

from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent

# A. predictivity and patterns in the evaluations
GATE_AND_CURVES = _ANALYSIS / "rq00_gate_and_curves"            # the above-random gate; score vs compute and vs training
SCALING_PREDICTABILITY = _ANALYSIS / "rq01_scaling_predictability"  # what moves with size; power-law prediction of the reference
DECISION_ACCURACY = _ANALYSIS / "rq02_decision_accuracy"        # does a small size / early checkpoint rank like the reference
# B. cheap measurements
NOISE_AND_SNR = _ANALYSIS / "rq03_noise_and_snr"                # seed vs checkpoint noise; the 22 SNR definitions; the seed holdout
SURROGATES = _ANALYSIS / "rq04_surrogates"                      # which cheap statistic predicts decision accuracy
# C. generalisation of the framework
DESIGN_DECISIONS = _ANALYSIS / "rq05_design_decisions"          # the five interventions; how small and how early
LANGUAGE_TRANSFER = _ANALYSIS / "rq06_language_transfer"        # unmeasured and never-trained languages
EXTERNAL_FRAMEWORKS = _ANALYSIS / "rq07_external_frameworks"    # agreement with AllenAI DataDecide
# D. benchmark improvement
SUBSET_SELECTION = _ANALYSIS / "rq08_subset_selection"          # can a subset beat the full set
BENCHMARK_DESIGN = _ANALYSIS / "rq09_benchmark_design"          # which design features predict reliability
