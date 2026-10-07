#!/usr/bin/env bash
# Full analysis of the predictivity ladder (90M–1.7B × L ∈ {1..50} × deep/shallow/swiglu
# × seven data builds × seeds) from the published ladder report — the wide CSV that
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
#   rq01  scaling predictability on every seed and data build
#   rq02  decision accuracy: the per-pool tables, then every figure that reads them
#   rq03  the 22 SNR variants per pool (read rq02's DA), the seed holdout
#   rq04  surrogates: the variant ranking per pool (reads rq03), then the
#         analysis that reads rq00, rq01, rq02 and rq03 at once
#   rq05  design decisions on every seed and data build; then rq03's effect-vs-noise
#   rq06  language transfer (reads rq05)
#   rq07  DataDecide agreement (reads rq03 and rq04)
#         English only: the L1 cells against the multilingual ones (reads the gate and the
#         external frameworks comparison's AllenAI table)
#   rq08  subset selection; rq09 benchmark design (reads rq03 code)
#         the above-chance items (reads the per-item store and the gate's mask)
#   then the report figures and the rules check over every table on disk.
#
# Themes: A predictivity of the evaluation (rq00-rq02), B cheap measurements
# (rq03-rq04), C generalisation (rq05-rq07), D benchmark improvement (rq08-rq09),
# then rq10, the one question past the reference (the 3B rung).
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
# Fixed PDF creation date: an unchanged figure re-renders to the same bytes,
# so git sees no diff (PNGs are already deterministic).
export SOURCE_DATE_EPOCH=0
# Steps that exited non-zero. Without this every stage failed silently and
# the script still printed ALL DONE, so a README could keep stale numbers.
FAILED=()
# The four pools (analysis/RULES.md, Definitions): `predictivity` is the grid
# seed, every design — the headline pool and the gate's. `predictivity_seeds`
# is every seed, which feeds the seed-noise estimates (rq03, rq05) and the reads
# that take every run (rq01, rq06); the two holdout pools are the replicate seeds
# and seed 1904 on exactly their cells.
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
# The cached tables have two inputs, the ladder report and the bBPB twins' table
# (bench_bpb.csv, rewritten by the first pass only when its content changes), so
# a cached table older than either was built from data we no longer have.
# Reusing it lets a whole run finish on last night's numbers while every log
# line claims success.
LADDER_CSV=$($PY -c "from snr.download.ladder import ladder_dir; print(ladder_dir() / 'ladder_report.csv')")
BENCH_BPB_CSV=analysis/rq08_subset_selection/bench_bpb.csv
# FORCE=1 recomputes the cached tables even when neither input is newer — after
# a change to the kernel, the gate or the loader, which the mtime cannot see.
fresh() { [ "${FORCE:-0}" != 1 ] && [ -f "$1" ] && [ ! "$LADDER_CSV" -nt "$1" ] \
            && { [ ! -f "$BENCH_BPB_CSV" ] || [ ! "$BENCH_BPB_CSV" -nt "$1" ]; }; }
# The acc-vs-FLOPs grids (rq00) are ~140 figures nothing else reads and about
# an hour of this script; CURVES=1 redraws them. Every table, CSV and README
# block is written either way, so the default is a complete refresh of the
# numbers with a stale set of viewer figures.
CURVES=${CURVES:-0}
GRIDS=(--no-grids); [ "$CURVES" = 1 ] && GRIDS=()

pass "the benchmark BPB twins"
# every loader adds a `bbpb_<task>` row beside a benchmark row the table has a
# value for (utils.with_bbpb_twins), so it is (re)written before anything loads;
# without the cluster-only per-item store it writes nothing and the committed table stays
run $PY analysis/rq08_subset_selection/build_per_item_store.py --bench-bpb

pass "rq00 — the above-random gate and the curves"
# The gate first: every later step reads its mask, and the rq00 panels read
# it too, so they follow it here rather than at the end of the run.
run $PY analysis/rq00_gate_and_curves/above_random.py --only predictivity
run $PY analysis/rq00_gate_and_curves/run_apertus.py --pool predictivity ${GRIDS[@]+"${GRIDS[@]}"}
run $PY analysis/rq00_gate_and_curves/curves.py --pool predictivity_seeds
run $PY analysis/rq00_gate_and_curves/panels.py --pool predictivity
# the reformulated twins (rf_*) against the letter originals, through the rq00 gate
run $PY analysis/rq00_task_reformulation/compare.py
# ... and the probe families' twins (README block `rf-compare-probe`; probe.sh's FAMILIES)
run $PY analysis/rq00_task_reformulation/compare.py --tag probe --families mmlu,commonsense_qa,cultural_bench_easy,bbh_mcq,acp_bench_mcq
# which probe candidates survive the gate, per language: read off the committed mask (README block `probe-survivors`)
run $PY analysis/rq00_task_reformulation/probe_survivors.py
# the ladder's gate floor against the public models' (all/external mask): size floor or benchmark floor
run $PY analysis/rq00_gate_and_curves/above_random_external.py --pool predictivity

pass "rq01 — scaling predictability"
# The ladder-frame reads take every seed and data build (`predictivity_seeds`).
run $PY analysis/rq01_scaling_predictability/analyze.py --pool predictivity_seeds
run $PY analysis/rq01_scaling_predictability/panels.py --pool predictivity_seeds
run $PY analysis/rq01_scaling_predictability/regimes.py --pool predictivity_seeds
# the same table with what the gate and the fit minimum removed put back, per family
run $PY analysis/rq01_scaling_predictability/regimes_survivorship.py --pool predictivity_seeds
run $PY analysis/rq01_scaling_predictability/scaling_law_error.py --pool predictivity_seeds
# a language's evaluation against the tokens of that language the proxy saw (BPB DA-goal; the gate per benchmark)
run $PY analysis/rq01_scaling_predictability/tokens_seen.py --pool predictivity_seeds

pass "rq02 — decision accuracy"
# The per-pool DA tables (the truth every later RQ reads), cached until the
# report is newer than them.
for t in "${POOLS[@]}"; do
  st=$(stage_of "$t")
  if fresh "analysis/rq02_decision_accuracy/$st/$t/da_all_per_task_both_axes.csv"; then
    echo "  (DA cached: analysis/rq02_decision_accuracy/$st/$t/da_all_per_task_both_axes.csv)"
  else
    run $PY analysis/rq02_decision_accuracy/compute_da.py --pool "$t"
  fi
done
# DA of the benchmark BPB (bBPB) against accuracy on the grid pool; reads the
# per-item store, so off the cluster it writes nothing (README block `bench-bpb`).
# The store was built under the retired pool name and lacks some predictivity
# models (the swiglu and L1 FWEB cells; the script prints and names them): read it
# there until build_per_item_store is re-run for predictivity. Unreadable parts are skipped.
run $PY analysis/rq02_decision_accuracy/bench_bpb_da.py --pool predictivity --store predictivity_schemes
for t in "${DOC_POOLS[@]}"; do
  run $PY analysis/rq02_decision_accuracy/da_per_benchmark.py --pool "$t"
  run $PY analysis/rq02_decision_accuracy/early_small.py --pool "$t"
done
# which (benchmark, language) cells rank reliably at all: the population every
# `above_*` figure below averages over. Reads rq02's da_all_per_task_both_axes.csv, so it comes
# after compute_da and BEFORE everything that filters on it — by_L and
# scale_convergence both skip their filtered variants when it has not run yet,
# which silently costs the paper's rq2 figures.
run $PY analysis/rq02_decision_accuracy/reliable_tasks.py --pool predictivity
# two rq00 figures that READ rq01's regimes and the two rq02 tables above, so they run
# here and not in the rq00 block, where they drew the previous refresh's tables
# (CLAUDE.md bug #16): the gate on one task, run by run, with its rq02 cells;
run $PY analysis/rq00_gate_and_curves/above_random_example.py --pool predictivity --task include_v2_og_hungarian_hungary
# the twins' effect on the gate (McNemar) and on every headline reading with / without them
run $PY analysis/rq00_task_reformulation/reformulations_gate.py --pool predictivity
# the toy explainer of the three DA kinds, the pair sets and the value lattice (no measured number; README block)
run $PY analysis/rq02_decision_accuracy/da_explainer.py --pool predictivity
# per language count: pairs of design variants sharing the L (the grid seed of predictivity_seeds); rq04's panels read it
run $PY analysis/rq02_decision_accuracy/by_L.py --pool predictivity
# per design axis: the mono-axis pairs split by the one axis they move, one panel per axis (DA-ckpt and DA-goal)
run $PY analysis/rq02_decision_accuracy/by_L.py --pool predictivity --by transformation
# cross-task predictability: every parent task as the proxy for every other one (DA-size and DA-ckpt level maps)
run $PY analysis/rq02_decision_accuracy/cross_task.py --pool predictivity
# scale convergence: how small a FULLY TRAINED model still decides like the
# reference, by language count and by design axis (+ their above_80 variants);
# `--by transformation` also writes the one-panel-per-axis twin (`_transformation_panels`)
run $PY analysis/rq02_decision_accuracy/scale_convergence.py --pool predictivity
# the paper's RQ2 figure: composes the three panels from the CSVs above, so it
# runs LAST of the rq02 block — it derives nothing of its own
run $PY analysis/rq02_decision_accuracy/paper_rq2.py --pool predictivity
# the mono-axis twins (`_mono_axis`): the same three definitions over the pairs
# that move ONE design axis, which is what upstream's "every pair" is by
# construction. Same folder, so the two readings compare without opening two.
run $PY analysis/rq02_decision_accuracy/by_L.py --pool predictivity --axes mono-axis
run $PY analysis/rq02_decision_accuracy/scale_convergence.py --pool predictivity --axes mono-axis
run $PY analysis/rq02_decision_accuracy/paper_rq2.py --pool predictivity --axes mono-axis
# rq02 extensions (plan/decision_accuracy.md): the per-L lines on the L8
# languages only and on the tasks every regime shares, one panel per L8
# language with a tokens-of-that-language axis, the DA ↔ Kendall τ identity and
# the tie-convention flip rates, and the seed-replicate uncertainty. All read
# da_all_reliable_tasks_both_axes.csv, so they come after reliable_tasks.py.
run $PY analysis/rq02_decision_accuracy/scale_convergence.py --pool predictivity --by L --langs L8
run $PY analysis/rq02_decision_accuracy/scale_convergence.py --pool predictivity --by L --langs L8 --common-tasks
run $PY analysis/rq02_decision_accuracy/by_language.py --pool predictivity
run $PY analysis/rq02_decision_accuracy/agreement.py --pool predictivity
# rq01's scaling statistics against rq02's ranking statistics, per task
run $PY analysis/rq02_decision_accuracy/scaling_vs_ranking.py --pool predictivity
# DA one rung above the ladder: the public model lines at 1B-1.7B against their 12-14B siblings (external gate)
run $PY analysis/rq02_decision_accuracy/public_ladders.py --pool predictivity
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
# effect_vs_noise reads rq05's decision table and rq03's panels read effect_vs_noise:
# both run in the rq05 block below.

pass "rq04 — surrogates"
for t in "${DOC_POOLS[@]}"; do
  run $PY analysis/rq04_surrogates/analyze_snr_variants.py --pool "$t"
  run $PY analysis/rq04_surrogates/snr_definition_postprocess.py --pool "$t"
done
# surrogates read the headline pool's rq03 table, rq00's scores, rq01's fits and rq02's by_L
run $PY analysis/rq04_surrogates/analyze.py --pool predictivity
run $PY analysis/rq04_surrogates/panels.py --pool predictivity
# FineTasks' four selection criteria on the ladder, judged by DA-size, plus every surrogate against DA-size
run $PY analysis/rq04_surrogates/finetasks_criteria.py --pool predictivity
# the surrogate catalogue (literature.md) and the truths (rq02's DA, tau, rho), then the validated search
run $PY analysis/rq04_surrogates/catalogue.py --pool predictivity
run $PY analysis/rq04_surrogates/search.py --pool predictivity

pass "rq05 — design decisions"
# rq05 needs the four interventions and its early-decision read follows from
# its decision table; rq06 reads its table for the never-trained languages.
run $PY analysis/rq05_design_decisions/analyze.py --pool predictivity_seeds
run $PY analysis/rq05_design_decisions/early_decision.py --pool predictivity_seeds
run $PY analysis/rq05_design_decisions/panels.py --pool predictivity_seeds
run $PY analysis/rq05_design_decisions/transformations.py --pool predictivity_seeds
# rq03's effect-vs-noise reads rq05's interventions and the seed replicates;
# rq03's panels draw the effect-over-seed grids from it, so they come right after
run $PY analysis/rq03_noise_and_snr/effect_vs_noise.py --pool predictivity_seeds
run $PY analysis/rq03_noise_and_snr/panels.py --pool predictivity

pass "rq06 — language transfer"
run $PY analysis/rq06_language_transfer/analyze.py --pool predictivity_seeds
run $PY analysis/rq06_language_transfer/panels.py --pool predictivity_seeds
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

pass "English only — the monolingual-English cells against the multilingual ones"
# reads the gate's mask, the decision-accuracy and noise-and-SNR computations
# (as functions), the replicate seeds and the AllenAI table of the external
# frameworks comparison above; `predictivity` holds every L1 family (deep,
# shallow, DCLM without edu, FineWeb), so this run writes the README too
run $PY analysis/rq13_english_only/english_only.py --pool predictivity

pass "rq08 — subset selection"
run $PY analysis/rq08_subset_selection/smooth_subtasks.py --pool predictivity
run $PY analysis/rq08_subset_selection/panels.py --pool predictivity
# per-item view: reads the per-item store; the store is built by the sbatch
# (analysis/rq08_subset_selection/build_per_item_store.sbatch), not here
run $PY analysis/rq08_subset_selection/per_item_ladder.py --pool predictivity
# the items the 1.7B runs solve, chosen on half the designs, DA and SNR read on the rest (store finals; nothing without it)
run $PY analysis/rq08_subset_selection/reference_solved.py --pool predictivity --store predictivity_schemes   # the retired pool's store (a subset of the models): switch with bench_bpb_da

pass "the above-chance items"
# per benchmark-language task, only the items the 1.7B runs answer above chance, chosen in sample on purpose:
# how much DA-size, SNR, the gate and the scaling fit rise, under the gate first and under the items first.
# Reads the per-item store and the gate's mask (nothing without the store); DA-ckpt and the checkpoint SNR
# are computed where the store holds checkpoints and skipped (said so) on a finals-only store
run $PY analysis/rq12_above_chance_items/above_chance_items.py --pool predictivity --store-pool predictivity_schemes   # switch with bench_bpb_da

pass "rq09 — benchmark design"
for t in "${DOC_POOLS[@]}"; do
  run $PY analysis/rq09_benchmark_design/analyze.py --pool "$t"
done
run $PY analysis/rq09_benchmark_design/panels.py --pool predictivity

pass "rq10 — size generalisation (the 3B rung as the reference)"
# the one reader of above_reference=True: every family with a 3B final, read from
# every smaller rung; writes header-only tables and a placeholder figure until the
# 3B evaluations are in the report, so the block fills in by itself
run $PY analysis/rq10_size_generalisation/above_reference.py --pool predictivity
# today's preview: the same four families read to 1.7B (the comparison line of panel (a))
run $PY analysis/rq10_size_generalisation/above_reference.py --pool predictivity --reference 1.7B --design 3B
# the prior question to the ranking one: which benchmarks the 3B rung lifts above
# chance that the reference cannot resolve at all (both gate columns on the 3B families)
run $PY analysis/rq10_size_generalisation/gate_crossover.py --pool predictivity

pass "rq11 — the evaluation recipe"
# which benchmark, posed how (original, rf, rfgm) and scored how (accuracy, bBPB), reads the reference
# from the smallest proxy: reads rq02's per-task early-small table, so it runs after rq02 (tau = utils.RELIABLE_DA)
run $PY analysis/rq11_evaluation_recipe/recipe.py --pool predictivity

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
