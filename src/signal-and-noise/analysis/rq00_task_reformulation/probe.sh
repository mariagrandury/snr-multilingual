#!/usr/bin/env bash
# Once the probe evals are in (auto_evals_cscs.py --group auto_probe
# --final-only): do the candidate benchmarks carry signal, and does the
# reformulation help the ones that ask for a letter? Five existing steps in
# order, each reading the previous one's output:
#
#   derive_task_options   n_items for every task with a result -- the gate's
#                         Wilson bound has no denominator without it and drops
#                         the task in silence. --only-missing because a full
#                         pass opens 4M sample files (an hour); drop the flag
#                         when a task's answer format may have changed
#   ladder_report --plot  the wide CSV every analysis reads (src/pretrain/,
#                         where SNR_LADDER_DIR points below)
#   above_random          the gate, over every task in the report; with
#                         SNR_TRAINED_GROUPS the multilingual candidates are
#                         read on the cells that trained their language
#   compare --tag probe   original vs rf_ twin on the five probe pairs, kept
#                         apart from the three letter-format families' figures
#   probe_survivors       the per-language verdict, from the mask alone
#
# Nothing here submits a job. Run it from anywhere; it needs the snr env.
set -euo pipefail
ROOT=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual
PY=${PY:-/users/mariagrandury/miniconda3/envs/snr/bin/python}
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=2 SOURCE_DATE_EPOCH=0
# rule 2 for the probe alone: a candidate counts as trained in its language.
# Only this pass sets it; every other RQ keeps the `auto` population.
export SNR_TRAINED_GROUPS=auto,auto_probe
export SNR_LADDER_DIR=$ROOT/src/pretrain
# the pairs: the candidates that ask for a letter and so have an rf_ twin
FAMILIES=${FAMILIES:-mmlu,commonsense_qa,cultural_bench_easy,bbh_mcq,acp_bench_mcq}

cd "$ROOT"
echo ">>> n_items";            $PY src/evals/scripts/derive_task_options.py --only-missing
echo ">>> ladder report";      $PY src/pretrain/ladder_report.py --check benchmarks --plot
cd src/signal-and-noise
echo ">>> gate";               $PY analysis/rq00_gate_and_curves/above_random.py --only predictivity
echo ">>> original vs rf";     $PY analysis/rq00_task_reformulation/compare.py --tag probe --families "$FAMILIES"
echo ">>> survivors";          $PY analysis/rq00_task_reformulation/probe_survivors.py
echo ">>> rules";              $PY analysis/check_rules.py
