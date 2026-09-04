#!/usr/bin/env bash
# Full analysis of the predictivity ladder (90M–1.7B × L ∈ {1..100} × deep/shallow
# × scheme A/B × seeds) from the published ladder report — the wide CSV that
# `src/pretrain/ladder_report.py --plot --publish --push-hf` writes to the HF
# dataset named in configs/hf_wandb.json (`repo_id_ladder_report`). The loader
# downloads it on first use; point SNR_LADDER_DIR at a directory holding
# ladder_report.csv to use a local copy (the cluster's capstor copy, a fixture).
#
# Idempotent: the per-task DA and SNR tables are computed once per pool and
# reused. Output layout: analysis/<rqNN_name>/pretraining/<pool>/
#
#   A. The above-random gate (its report feeds the rq01 slides), then DA and SNR
#      per pool (DA is the truth rq02's variants are scored against).
#   B. Seed holdout — needs the train/test pool CSVs from A, read by rq02's README.
#   C. Per-pool analysis + docs (the canonical pool last so it sees the holdout).
#   D. rq06 (proxy size × L), rq04, the curve viewer, the figures.
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
# `predictivity` is the plan grid (seed 1904): the headline pool. The
# all-seeds pool feeds the seed-noise estimates (rq06); the two holdout pools are
# the ×3 cells split by seed.
POOLS=(predictivity_seeds predictivity_seeds_train predictivity_seeds_test predictivity)

run() { echo; echo ">>> $*"; "$@" 2>&1 | grep -vE "RuntimeWarning|scores_shifted|scores = \(scores|depths|rel_noise|ckpt-DA: only one ckpt|Tasks:|families:|languages:|Per-benchmark grids|Per-language grids|projection |rms_deviation |range  |iqr  |tukey " | tail -18; }
stage_of() { $PY -c "import sys,json; print(json.load(open('../../configs/models.json'))['pools'][sys.argv[1]].get('stage','pretraining'))" "$1"; }

echo "############################## PASS A — gate, DA, SNR compute ##############################"
run $PY analysis/rq00_acc_vs_flops/above_random.py --only predictivity
for t in "${POOLS[@]}"; do
  st=$(stage_of "$t")
  if [ ! -f "analysis/rq01_decision_accuracy/$st/$t/da_per_task.csv" ]; then
    run $PY analysis/rq01_decision_accuracy/compute_da.py --pool "$t"
  else
    echo "  (DA cached: analysis/rq01_decision_accuracy/$st/$t/da_per_task.csv)"
  fi
  if [ ! -f "analysis/rq02_snr_definition/$st/$t/snr_variants_per_task.csv" ]; then
    run $PY analysis/rq02_snr_definition/run_apertus_snr_variants.py --pool "$t"
  else
    echo "  (SNR cached: analysis/rq02_snr_definition/$st/$t/snr_variants_per_task.csv)"
  fi
done

echo "############################## PASS B — seed holdout ##############################"
run $PY analysis/rq02_snr_definition/compare_seed_splits.py \
    --train-pool predictivity_seeds_train --test-pool predictivity_seeds_test

echo "############################## PASS C — analysis + docs ##############################"
# rq03 needs the AllenAI-side SNR table (built once from the DataDecide `core`
# split on HF; a git-lfs pointer here means `git lfs pull` first).
ALLENAI_CSV=analysis/rq03_allenai_comparison/allenai_snr_variants_per_task.csv
if [ ! -f "$ALLENAI_CSV" ]; then
  run $PY analysis/rq03_allenai_comparison/build_allenai_variants.py
fi
for t in predictivity_seeds predictivity; do
  echo "############################## POOL $t ##############################"
  run $PY analysis/rq02_snr_definition/analyze_snr_variants.py --pool "$t"
  run $PY analysis/rq02_snr_definition/snr_definition_postprocess.py --pool "$t"
  run $PY analysis/rq01_decision_accuracy/da_per_benchmark.py --pool "$t"
  run $PY analysis/rq05_benchmark_creation/analyze.py --pool "$t"
  if grep -q "^version https://git-lfs" "$ALLENAI_CSV" 2>/dev/null; then
    echo "  (rq03 skipped: $ALLENAI_CSV is a git-lfs pointer — run git lfs pull)"
  else
    run $PY analysis/rq03_allenai_comparison/analyze.py --pool "$t"
  fi
done

echo "############################## PASS D — rq06, rq04, curves, figures ##############################"
run $PY analysis/rq06_proxy_predictivity/analyze.py --pool predictivity_seeds
run $PY analysis/rq04_smooth_subtasks/smooth_subtasks.py --pool predictivity
run $PY analysis/rq00_acc_vs_flops/run_apertus.py --pool predictivity
run $PY analysis/report_figures/make_figures.py

echo "############################## ALL DONE ##############################"
