"""Per-research-question output roots — the one place that knows the rqNN_
directory names.

Each analysis script writes its artifacts under ``<its RQ dir>/<stage>/<pool>/``
and reads sibling RQs' artifacts via these constants, so the numbered directory
names live in exactly one module. Replaces the old ``PLOT_DIR / "<area>"``
roots (outputs used to live under ``results/``; they now live next to each RQ's
script).
"""

from pathlib import Path

_ANALYSIS = Path(__file__).resolve().parent

ACC_VS_FLOPS = _ANALYSIS / "rq00_acc_vs_flops"
DECISION_ACCURACY = _ANALYSIS / "rq01_decision_accuracy"
SNR_DEFINITION = _ANALYSIS / "rq02_snr_definition"
ALLENAI_COMPARISON = _ANALYSIS / "rq03_allenai_comparison"
SMOOTH_SUBTASKS = _ANALYSIS / "rq04_smooth_subtasks"
BENCHMARK_CREATION = _ANALYSIS / "rq05_benchmark_creation"
PROXY_PREDICTIVITY = _ANALYSIS / "rq06_proxy_predictivity"
