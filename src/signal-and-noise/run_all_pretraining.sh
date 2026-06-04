#!/usr/bin/env bash
# Full pretraining-stage analysis across the 4 model-set tiers.
# Idempotent: skips the per-task SNR compute when its CSV already exists.
# Output layout: results/<analysis>/pretraining/<pool>/
set -uo pipefail
cd "$(dirname "$0")"
PY=python3
TIERS=(seeds_1904 seeds_28_1797 seeds_28_1797_1904 custom_swissai_hf)
SNR=results/snr_definition/pretraining

run() { echo; echo ">>> $*"; "$@" 2>&1 | grep -vE "RuntimeWarning|scores_shifted|scores = \(scores|depths|rel_noise|ckpt-DA: only one ckpt|Tasks:|families:|languages:|Per-benchmark grids|Per-language grids|projection |rms_deviation |range  |iqr  |tukey " | tail -18; }

for t in "${TIERS[@]}"; do
  echo "############################## TIER $t ##############################"
  if [ ! -f "$SNR/$t/snr_variants_per_task.csv" ]; then
    run $PY multilingual/run_apertus_snr_variants.py --pool "$t"
  else
    echo "  (compute cached: $SNR/$t/snr_variants_per_task.csv)"
  fi
  run $PY multilingual/analyze_snr_variants.py --pool "$t"
  run $PY multilingual/snr_definition_postprocess.py --pool "$t"
  run $PY results/benchmark_creation/analyze.py --pool "$t"
  run $PY results/allenai_comparison/analyze.py --pool "$t"
done

echo "############################## SEED-SPLIT HOLDOUT ##############################"
run $PY multilingual/compare_seed_splits.py --train-pool seeds_28_1797 --test-pool seeds_1904

# The snr_definition README's seed-generalization table reads the holdout above,
# which runs after the tier loop — refresh the canonical RQ1 docs now that it
# exists (idempotent; rewrites only the auto:* blocks).
run $PY multilingual/snr_definition_postprocess.py --pool custom_swissai_hf

echo "############################## RQ4 smooth_subtasks ##############################"
run $PY multilingual/smooth_subtasks.py --pool seeds_28_1797_1904
run $PY multilingual/smooth_subtasks.py --pool custom_swissai_hf

echo "############################## RQ4 per-sample (local reuse) ##############################"
run $PY multilingual/analyze_per_sample_d.py

echo "############################## above-random gate + acc-vs-flops (top-N) #############"
# above_random writes the custom / custom_swiss_hf gate reports that
# run_apertus's README generator reads — run it first.
run $PY multilingual/above_random.py
run $PY multilingual/run_apertus.py --pool seeds_28_1797_1904
run $PY multilingual/run_apertus.py --pool custom_swissai_hf

echo "############################## ALL DONE ##############################"
