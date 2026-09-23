#!/usr/bin/env bash
# Full analysis of the predictivity ladder (90M–1.7B × L ∈ {1..100} × deep/shallow
# × five data schemes × seeds) from the published ladder report — the wide CSV that
# `src/pretrain/ladder_report.py --plot --publish --push-hf` writes to the HF
# dataset named in configs/hf_wandb.json (`repo_id_ladder_report`). The loader
# downloads it on first use; point SNR_LADDER_DIR at a directory holding
# ladder_report.csv to use a local copy (the cluster's capstor copy, a fixture).
#
# Idempotent: the per-task DA and SNR tables are computed once per pool and
# reused, until the ladder report is newer than them (FORCE=1 recomputes them).
# CURVES=1 also redraws rq00's ~140 acc-vs-FLOPs grids, which nothing reads
# and which cost about an hour; the default leaves the ones on disk. Output layout:
# analysis/<rqNN_name>/pretraining/<pool>/
#
# The passes run IN RESEARCH-QUESTION ORDER, so rq00's tables and figures are
# final minutes into the run and rq01's and rq02's within the first hour,
# instead of at the end. The order is also the dependency order, with two
# exceptions kept next to their inputs and marked below: rq03's effect-vs-noise
# reads rq05's decision table, and rq07 reads rq04's variant ranking.
#
#   rq00  the above-random gate (every later step reads its mask), the curves
#   rq01  scaling predictability on every seed and scheme
#   rq02  decision accuracy: the per-pool tables, then every figure that reads them
#   rq03  the 22 SNR variants per pool (read rq02's DA), the seed holdout
#   rq04  surrogates: the variant ranking per pool (reads rq03), then the
#         analysis that reads rq00, rq01, rq02 and rq03 at once
#   rq05  design decisions on every seed and scheme; then rq03's effect-vs-noise
#   rq06  language transfer (reads rq05)
#   rq07  DataDecide agreement (reads rq03 and rq04)
#   rq08  subset selection; rq09 benchmark design (reads rq03 code)
#   then the report figures and the rules check over every table on disk.
#
# Themes: A predictivity of the evaluation (rq00-rq02), B cheap measurements
# (rq03-rq04), C generalisation (rq05-rq07), D benchmark improvement (rq08-rq09).
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
# Fixed PDF creation date: an unchanged figure re-renders to the same bytes,
# so git sees no diff (PNGs are already deterministic).
export SOURCE_DATE_EPOCH=0
# Steps that exited non-zero. Without this every stage failed silently and
# the script still printed ALL DONE, so a README could keep stale numbers.
FAILED=()
# `predictivity` is the plan grid (seed 1904): the headline pool. The
# all-seeds pool feeds the seed-noise estimates (rq03, rq05); the two holdout pools are
# the ×3 cells split by seed.
POOLS=(predictivity_seeds predictivity_seeds_train predictivity_seeds_test predictivity)
# The pools whose per-pool analysis and docs are written (the canonical one
# last, so it sees the holdout).
DOC_POOLS=(predictivity_seeds predictivity)

# Per-step wall time, so the next person can see where the hours go instead
# of inferring it from output timestamps.
run() { local t0=$SECONDS; echo; echo ">>> $*"; "$@" 2>&1 | grep -vE "RuntimeWarning|scores_shifted|scores = \(scores|depths|rel_noise|ckpt-DA: only one ckpt|Tasks:|families:|languages:|Per-benchmark grids|Per-language grids|projection |rms_deviation |range  |iqr  |tukey " | tail -18
       [ "${PIPESTATUS[0]}" -eq 0 ] || FAILED+=("$*")
       printf '    [%dm %02ds] %s\n' $(( (SECONDS - t0) / 60 )) $(( (SECONDS - t0) % 60 )) "${2##*/}"; }
pass() { echo; echo "############################## $* ##############################"; }
stage_of() { $PY -c "import sys,json; print(json.load(open('../../configs/models.json'))['pools'][sys.argv[1]].get('stage','pretraining'))" "$1"; }
# The ladder report is the only input, so a cached table older than it was built
# from data we no longer have. Reusing it lets a whole run finish on last
# night's numbers while every log line claims success.
LADDER_CSV=$($PY -c "from snr.download.ladder import ladder_dir; print(ladder_dir() / 'ladder_report.csv')")
# FORCE=1 recomputes the cached tables even when the report is not newer — after
# a change to the kernel, the gate or the loader, which the mtime cannot see.
fresh() { [ "${FORCE:-0}" != 1 ] && [ -f "$1" ] && [ ! "$LADDER_CSV" -nt "$1" ]; }
# The acc-vs-FLOPs grids (rq00) are ~140 figures nothing else reads and about
# an hour of this script; CURVES=1 redraws them. Every table, CSV and README
# block is written either way, so the default is a complete refresh of the
# numbers with a stale set of viewer figures.
CURVES=${CURVES:-0}
GRIDS=(--no-grids); [ "$CURVES" = 1 ] && GRIDS=()

pass "rq00 — the above-random gate and the curves"
# The gate first: every later step reads its mask, and the rq00 panels read
# it too, so they follow it here rather than at the end of the run.
run $PY analysis/rq00_gate_and_curves/above_random.py --only predictivity
run $PY analysis/rq00_gate_and_curves/run_apertus.py --pool predictivity ${GRIDS[@]+"${GRIDS[@]}"}
run $PY analysis/rq00_gate_and_curves/curves.py --pool predictivity_all
run $PY analysis/rq00_gate_and_curves/panels.py --pool predictivity
# the reformulated twins (rf_*) against the letter originals, through the rq00 gate
run $PY analysis/rq00_task_reformulation/compare.py
# the ladder's gate floor against the public models' (all/external mask): size floor or benchmark floor
run $PY analysis/rq00_gate_and_curves/benchmark_floor.py --pool predictivity

pass "rq01 — scaling predictability"
# The ladder-frame reads take every seed and scheme (`predictivity_all`); the
# scaling-law error reads every scheme.
run $PY analysis/rq01_scaling_predictability/analyze.py --pool predictivity_all
run $PY analysis/rq01_scaling_predictability/panels.py --pool predictivity_all
run $PY analysis/rq01_scaling_predictability/regimes.py --pool predictivity_all
# the same table with what the gate and the fit minimum removed put back, per family
run $PY analysis/rq01_scaling_predictability/regimes_survivorship.py --pool predictivity_all
run $PY analysis/rq01_scaling_predictability/scaling_law_error.py --pool predictivity_all

pass "rq02 — decision accuracy"
# The per-pool DA tables (the truth every later RQ reads), cached until the
# report is newer than them.
for t in "${POOLS[@]}"; do
  st=$(stage_of "$t")
  if fresh "analysis/rq02_decision_accuracy/$st/$t/da_per_task.csv"; then
    echo "  (DA cached: analysis/rq02_decision_accuracy/$st/$t/da_per_task.csv)"
  else
    run $PY analysis/rq02_decision_accuracy/compute_da.py --pool "$t"
  fi
done
# the scheme-inclusive DA table: every data scheme at the grid seed (AT3/BT3 =
# a temperature, ZH/ES = a second language), which is the population by_L and
# scale_convergence actually pair over. The headline `predictivity` pool keeps
# its A/B filter because its SNR signal would widen; decision accuracy is a rank
# agreement and has no such problem (plan/decision_accuracy.md).
run $PY analysis/rq02_decision_accuracy/compute_da.py --pool predictivity_schemes
for t in "${DOC_POOLS[@]}"; do
  run $PY analysis/rq02_decision_accuracy/da_per_benchmark.py --pool "$t"
  run $PY analysis/rq02_decision_accuracy/early_small.py --pool "$t"
done
# which (benchmark, language) cells rank reliably at all: the population every
# `above_*` figure below averages over. Reads rq02's da_per_task.csv, so it comes
# after compute_da and BEFORE everything that filters on it — by_L and
# scale_convergence both skip their filtered variants when it has not run yet,
# which silently costs the paper's rq2 figures.
run $PY analysis/rq02_decision_accuracy/reliable_tasks.py --pool predictivity
# per language count: pairs of design variants sharing the L (predictivity_all at the grid seed); rq04's panels read it
run $PY analysis/rq02_decision_accuracy/by_L.py --pool predictivity
# cross-task predictability: every parent task as the proxy for every other one (DA-size and DA-ckpt level maps)
run $PY analysis/rq02_decision_accuracy/cross_task.py --pool predictivity
# scale convergence: how small a FULLY TRAINED model still decides like the
# reference, by language count and by design axis (+ their above_80 variants)
run $PY analysis/rq02_decision_accuracy/scale_convergence.py --pool predictivity
# DA at all ten evaluated checkpoints (rq02's own table stops at da_early_fracs)
run $PY analysis/rq02_decision_accuracy/paper_ten_checkpoints.py
# the paper's RQ2 figure: composes the three panels from the CSVs above, so it
# runs LAST of the rq02 block — it derives nothing of its own
run $PY analysis/rq02_decision_accuracy/paper_rq2.py --pool predictivity
# the mono-axis twins (`_one_axis`): the same three definitions over the pairs
# that move ONE design axis, which is what upstream's "every pair" is by
# construction. Same folder, so the two readings compare without opening two.
run $PY analysis/rq02_decision_accuracy/by_L.py --pool predictivity --axes mono-axis
run $PY analysis/rq02_decision_accuracy/scale_convergence.py --pool predictivity --axes mono-axis
run $PY analysis/rq02_decision_accuracy/paper_rq2.py --pool predictivity --axes mono-axis
# rq02 extensions (plan/decision_accuracy.md): the per-L lines on the L8
# languages only and on the tasks every regime shares, one panel per L8
# language with a tokens-of-that-language axis, the DA ↔ Kendall τ identity and
# the tie-convention flip rates, and the seed-replicate uncertainty. All read
# da_reliable_tasks.csv, so they come after reliable_tasks.py.
run $PY analysis/rq02_decision_accuracy/scale_convergence.py --pool predictivity --by L --langs L8
run $PY analysis/rq02_decision_accuracy/scale_convergence.py --pool predictivity --by L --langs L8 --common-tasks
run $PY analysis/rq02_decision_accuracy/by_language.py --pool predictivity
run $PY analysis/rq02_decision_accuracy/agreement.py --pool predictivity
run $PY analysis/rq02_decision_accuracy/seed_uncertainty.py --pool predictivity
# do the high-resource languages rank more reliably: one pooled line per language tier, and per language against its share
run $PY analysis/rq02_decision_accuracy/language_tier.py --pool predictivity
# the three definitions on multi-axis against mono-axis pairs, same cells (exploratory)
run $PY analysis/rq02_decision_accuracy/pair_axes.py --pool predictivity

pass "rq03 — noise and the SNR variants"
# The 22 SNR variants per pool (they read rq02's DA), cached like the DA tables.
for t in "${POOLS[@]}"; do
  st=$(stage_of "$t")
  if fresh "analysis/rq03_noise_and_snr/$st/$t/snr_variants_per_task.csv"; then
    echo "  (SNR cached: analysis/rq03_noise_and_snr/$st/$t/snr_variants_per_task.csv)"
  else
    run $PY analysis/rq03_noise_and_snr/run_apertus_snr_variants.py --pool "$t"
  fi
done
# seed holdout — needs the train/test pool CSVs above
run $PY analysis/rq03_noise_and_snr/compare_seed_splits.py \
    --train-pool predictivity_seeds_train --test-pool predictivity_seeds_test
run $PY analysis/rq03_noise_and_snr/panels.py --pool predictivity
# effect_vs_noise reads rq05's decision table: it runs in the rq05 block below.

pass "rq04 — surrogates"
for t in "${DOC_POOLS[@]}"; do
  run $PY analysis/rq04_surrogates/analyze_snr_variants.py --pool "$t"
  run $PY analysis/rq04_surrogates/snr_definition_postprocess.py --pool "$t"
done
# surrogates read the headline pool's rq03 table, rq00's scores, rq01's fits and rq02's by_L
run $PY analysis/rq04_surrogates/analyze.py --pool predictivity
run $PY analysis/rq04_surrogates/panels.py --pool predictivity

pass "rq05 — design decisions"
# rq05 needs the five interventions and its early-decision read follows from
# its decision table; rq06 reads its table for the never-trained languages.
run $PY analysis/rq05_design_decisions/analyze.py --pool predictivity_all
run $PY analysis/rq05_design_decisions/early_decision.py --pool predictivity_all
run $PY analysis/rq05_design_decisions/panels.py --pool predictivity_all
run $PY analysis/rq05_design_decisions/transformations.py --pool predictivity_all
# rq03's effect-vs-noise reads rq05's interventions and the seed replicates
run $PY analysis/rq03_noise_and_snr/effect_vs_noise.py --pool predictivity_all

pass "rq06 — language transfer"
run $PY analysis/rq06_language_transfer/analyze.py --pool predictivity_all
run $PY analysis/rq06_language_transfer/panels.py --pool predictivity_all
# the minimal language panel: one language / English / the panel macro at the proxy against the 1.7B macro ranking
run $PY analysis/rq06_language_transfer/language_panel.py --pool predictivity

pass "rq07 — external frameworks"
# rq07 needs the AllenAI-side SNR table (built once from the DataDecide `core`
# split on HF; a git-lfs pointer here means `git lfs pull` first) and rq04's
# variant ranking per pool.
ALLENAI_CSV=analysis/rq07_external_frameworks/allenai_snr_variants_per_task.csv
if [ ! -f "$ALLENAI_CSV" ]; then
  run $PY analysis/rq07_external_frameworks/build_allenai_variants.py
fi
for t in "${DOC_POOLS[@]}"; do
  if grep -q "^version https://git-lfs" "$ALLENAI_CSV" 2>/dev/null; then
    echo "  (rq07 skipped: $ALLENAI_CSV is a git-lfs pointer — run git lfs pull)"
  else
    run $PY analysis/rq07_external_frameworks/analyze.py --pool "$t"
  fi
done

pass "rq08 — subset selection"
run $PY analysis/rq08_subset_selection/smooth_subtasks.py --pool predictivity
run $PY analysis/rq08_subset_selection/panels.py --pool predictivity

pass "rq09 — benchmark design"
for t in "${DOC_POOLS[@]}"; do
  run $PY analysis/rq09_benchmark_design/analyze.py --pool "$t"
done
run $PY analysis/rq09_benchmark_design/panels.py --pool predictivity
# FineTasks' four selection criteria computed on the ladder and judged by DA-size against the reference
run $PY analysis/rq09_benchmark_design/finetasks_criteria.py --pool predictivity

pass "report figures and the rules check"
run $PY analysis/report_figures/make_figures.py
# every table on disk against analysis/RULES.md (rule 14)
run $PY analysis/check_rules.py --quiet

if [ ${#FAILED[@]} -gt 0 ]; then
  echo "############################## FAILED ##############################"
  printf "  %s\n" "${FAILED[@]}"
  exit 1
fi
echo "############################## ALL DONE ##############################"
