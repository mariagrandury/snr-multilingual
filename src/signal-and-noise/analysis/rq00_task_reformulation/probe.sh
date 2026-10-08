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
#                         read on the cells that trained their language.
#                         Written to $PROBE_GATE, NOT the canonical tables:
#                         this gate counts the probe as trained and reads the
#                         fresh local report, and until 2026-10-08 it replaced
#                         rq00_gate_and_curves/pretraining/predictivity/above_random_*.csv,
#                         which every other RQ reads (SNR_GATE_DIR, above_random.py)
#   compare --tag probe   original vs rf_ twin on every probe pair, kept
#                         apart from the three letter-format families' figures
#   probe_survivors       the per-language verdict, from the probe gate's mask
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
# the probe gate's tables, outside the repo (see above_random's SNR_GATE_DIR)
PROBE_GATE=${PROBE_GATE:-/iopsstor/scratch/cscs/$USER/snr-probe-gate}
# the pairs: the candidates that ask for a letter and so have an rf_ twin
# (the 2026-09-23 five, then the 2026-10-08 batch's seven)
FAMILIES=${FAMILIES:-mmlu,commonsense_qa,cultural_bench_easy,bbh_mcq,acp_bench_mcq,m_mmlu,mmmlu,arabicmmlu,oall_arabic_mmlu,oall_exams,careqa,french_bench_extra}

cd "$ROOT"
echo ">>> n_items";            $PY src/evals/scripts/derive_task_options.py --only-missing
echo ">>> ladder report";      $PY src/pretrain/ladder_report.py --check benchmarks --plot
cd src/signal-and-noise
echo ">>> gate";               SNR_GATE_DIR=$PROBE_GATE $PY analysis/rq00_gate_and_curves/above_random.py --only predictivity
echo ">>> original vs rf";     $PY analysis/rq00_task_reformulation/compare.py --tag probe --families "$FAMILIES"
echo ">>> survivors";          SNR_GATE_DIR=$PROBE_GATE $PY analysis/rq00_task_reformulation/probe_survivors.py
echo ">>> rules";              $PY analysis/check_rules.py
