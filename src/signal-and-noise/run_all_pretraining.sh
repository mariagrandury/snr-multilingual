#!/usr/bin/env bash
# Full pretraining-stage analysis across the 4 model-set tiers.
# Idempotent: skips the per-task SNR compute when its CSV already exists.
# Output layout: analysis/<rqNN_name>/pretraining/<pool>/
#
# Three passes so dependencies resolve in one shot:
#   A. SNR compute per tier (cheap since the per-task pre-slice; ~35s/pool).
#   B. Seed-split holdout — needs the seeds_1904 + seeds_28_1797 CSVs from A,
#      and is read by the snr_definition README, so it runs before pass C.
#   C. Per-tier analysis + doc generation (the canonical pool, last, now sees
#      the fresh holdout — no redundant re-run).
set -uo pipefail
cd "$(dirname "$0")"
PY=python3
# The four custom-pretraining tiers, plus `external`: every non-custom model
# pooled across all parquets (reference_hf + a06 + distillation + posttraining),
# all models and all tasks. PASS A (DA+SNR) and PASS C (analysis) cover every
# tier; the custom-only PASS B and the rq00/above-random/smooth tail skip
# `external` (no mixture axis / custom random baseline there).
TIERS=(seeds_1904 seeds_28_1797 seeds_28_1797_1904 custom_swissai_hf external)

run() { echo; echo ">>> $*"; "$@" 2>&1 | grep -vE "RuntimeWarning|scores_shifted|scores = \(scores|depths|rel_noise|ckpt-DA: only one ckpt|Tasks:|families:|languages:|Per-benchmark grids|Per-language grids|projection |rms_deviation |range  |iqr  |tukey " | tail -18; }

# A tier's output stage (pretraining / external / …) comes from its pool config,
# so the DA/SNR cache paths must follow it — otherwise the `external` tier (a
# non-pretraining stage) never cache-hits and recomputes every run.
stage_of() { $PY -c "import sys,json; print(json.load(open('../../configs/models.json'))['pools'][sys.argv[1]].get('stage','pretraining'))" "$1"; }

echo "############################## PASS A — DA then SNR compute ##############################"
# Decision accuracy is the ground truth (rq01); SNR variants are the proxies
# (rq02) and read the DA table, so DA is computed first.
for t in "${TIERS[@]}"; do
  st=$(stage_of "$t")
  DA="analysis/rq01_decision_accuracy/$st"
  SNR="analysis/rq02_snr_definition/$st"
  if [ ! -f "$DA/$t/da_per_task.csv" ]; then
    run $PY analysis/rq01_decision_accuracy/compute_da.py --pool "$t"
  else
    echo "  (DA cached: $DA/$t/da_per_task.csv)"
  fi
  if [ ! -f "$SNR/$t/snr_variants_per_task.csv" ]; then
    run $PY analysis/rq02_snr_definition/run_apertus_snr_variants.py --pool "$t"
  else
    echo "  (SNR cached: $SNR/$t/snr_variants_per_task.csv)"
  fi
done

echo "############################## PASS B — seed-split holdout ##############################"
run $PY analysis/rq02_snr_definition/compare_seed_splits.py --train-pool seeds_28_1797 --test-pool seeds_1904

echo "############################## PASS C — analysis + docs ##############################"
# rq03 analyze reads the AllenAI-side SNR table; build it once (pool-independent,
# pulls the DataDecide `core` split from HF) before the per-tier rq03 runs.
ALLENAI_CSV=analysis/rq03_allenai_comparison/allenai_snr_variants_per_task.csv
if [ ! -f "$ALLENAI_CSV" ]; then
  run $PY analysis/rq03_allenai_comparison/build_allenai_variants.py
else
  echo "  (AllenAI CSV cached: $ALLENAI_CSV)"
fi
for t in "${TIERS[@]}"; do
  echo "############################## TIER $t ##############################"
  run $PY analysis/rq02_snr_definition/analyze_snr_variants.py --pool "$t"
  run $PY analysis/rq02_snr_definition/snr_definition_postprocess.py --pool "$t"
  run $PY analysis/rq05_benchmark_creation/analyze.py --pool "$t"
  run $PY analysis/rq03_allenai_comparison/analyze.py --pool "$t"
done

echo "############################## RQ4 smooth_subtasks ##############################"
run $PY analysis/rq04_smooth_subtasks/smooth_subtasks.py --pool seeds_28_1797_1904
run $PY analysis/rq04_smooth_subtasks/smooth_subtasks.py --pool custom_swissai_hf
run $PY analysis/rq04_smooth_subtasks/smooth_subtasks.py --pool external

echo "############################## RQ4 per-sample (local reuse) ##############################"
run $PY analysis/rq04_smooth_subtasks/analyze_per_sample_d.py

echo "############################## above-random gate + acc-vs-flops (top-N) #############"
# above_random writes the per-pool gate reports (seeds_28_1797_1904 /
# custom_swissai_hf / external) that run_apertus's README generator reads —
# run it first.
run $PY analysis/rq00_acc_vs_flops/above_random.py
run $PY analysis/rq00_acc_vs_flops/run_apertus.py --pool seeds_28_1797_1904
run $PY analysis/rq00_acc_vs_flops/run_apertus.py --pool custom_swissai_hf

echo "############################## ALL DONE ##############################"
