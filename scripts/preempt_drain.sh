#!/bin/bash
# preempt_drain.sh — move my PENDING convert/eval jobs from `normal` to
# `preemptable`, keeping at most --max-nodes (default 50) nodes of mine there.
#
# Why: `normal` is capped by its QOS at 480 nodes for the WHOLE partition, so
# with the cluster full my jobs sit on QOSGrpNodeLimit for hours — on
# 2026-09-18 sbatch --test-only put the same 1-node eval at 13:34 on `normal`
# and 04:36 on `preemptable`. `preemptable` has all 1343 nodes and no group
# cap; the price is preemption (PreemptMode=REQUEUE, 4 min grace).
#
# That price is small for exactly these two kinds, and both RESUME:
#
#   convert-*  convert-snr.sh touches .hf_complete per iter and skips what
#              carries it, so a preemption costs at most the in-flight iter.
#   eval-*     scripts/eval_worker.py writes each task's results as it
#              finishes (per_task/<task>/), and the auto-eval watcher
#              resubmits with only the tasks still missing.
#
# Neither sets --no-requeue, so Slurm requeues them and they pick up where
# they stopped. Pretrain jobs are NEVER moved: 21 nodes lost to a preemption
# costs up to a save interval of training and there is no requeue for them
# (--no-requeue in launch_pretraining_cscs.sh). Unlike debug_drain.sh nothing
# is truncated here — preemptable's 24 h limit is above every eval walltime.
#
# Order — the priority ladder (2026-09-18):
#
#   0  convert-*                     the gate on every eval downstream
#   1  eval-* of the FINAL ckpt      the headline number for each model
#   2  eval-* at 20/40/60/80 %       the training-curve points
#   3  every other eval-*            the rest of the k/20 grid
#
# The fraction comes from the size's own schedule (launch_trainings.schedule_for,
# the same source the watcher's due_iters uses), not from the iter number, so
# 81000 at 1.7B and 28800 at 600M both read as 100 %. If that lookup fails the
# script still runs — every eval then ranks 3 and only converts keep priority.
#
# It does NOT submit anything: it only moves jobs that are already pending, so
# it cannot duplicate work. It also does not exit when the queue empties — it
# keeps watching, so conversions and evals submitted by a later auto_evals
# pass are moved too. Stop it with kill.
#
# Safe to run beside the other two drainers: jobs already placed in a
# reservation are skipped, and a job moved here leaves `normal`, which is the
# only queue debug_drain.sh reads.
#
# Usage:
#   bash scripts/preempt_drain.sh --dry-run         # show what it would move
#   bash scripts/preempt_drain.sh                   # loop (default 300 s)
#   bash scripts/preempt_drain.sh --once            # single pass
#   bash scripts/preempt_drain.sh --max-nodes 20 --interval 600
set -uo pipefail
MAX_NODES=50
INTERVAL=300; ONCE=0; DRY=0
PART=preemptable
PRETRAIN_DIR=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-nodes) MAX_NODES="$2"; shift 2 ;;
    --interval)  INTERVAL="$2"; shift 2 ;;
    --once)      ONCE=1; shift ;;
    --dry-run)   DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

sum() { awk '{s+=$1} END{print s+0}'; }

# size|arch -> train_iters, read once: the schedules are static. A failure here
# is not fatal (see the header), so the error is reported and the map left empty.
# python3.11, not python3: the login node's python3 is 3.6 and cannot even
# parse launch_trainings.py (walrus operators), which fails silently into an
# empty map and demotes every eval to rank 3.
PY_BIN=$(command -v python3.11 || command -v python3)
TARGETS=$(OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY_BIN" - "$PRETRAIN_DIR" <<'PY' 2>/dev/null
import json, sys
sys.path.insert(0, sys.argv[1])
from launch_trainings import HYPERPARAMS, schedule_for
for arch, path in HYPERPARAMS.items():
    for size, cfg in json.loads(path.read_text())["configs"].items():
        try:
            print(f"{size}|{arch}|{schedule_for(cfg)[0]}")
        except KeyError:            # a size with no predictivity block
            pass
PY
)
[ -z "$TARGETS" ] && echo "WARN: no schedules read from $PRETRAIN_DIR — evals rank by name only"

drain_once() {
    local q pend npend mine room
    # squeue's EXIT STATUS, not its output: a transient controller failure
    # prints nothing and would otherwise read as an empty queue (the trap
    # debug_drain.sh fell into on 2026-09-07).
    if ! q=$(squeue --me -h -p normal -t PD -o "%v|%i|%D|%j" 2>/dev/null); then
        echo "[$(date +%H:%M:%S)] squeue failed — retrying next tick"; return 0
    fi
    # Jobs already placed in a reservation (%v) stay there, and only convert/
    # eval are eligible — the pretrain filter is this line.
    pend=$(awk -F'|' '$1=="(null)" && $4 ~ /^(convert|eval)-/' <<<"$q")
    npend=$(grep -c . <<<"$pend")
    if ! mine=$(squeue --me -h -p "$PART" -t PD,R,CG -o "%D" 2>/dev/null | sum); then
        echo "[$(date +%H:%M:%S)] squeue on $PART failed — retrying next tick"; return 0
    fi
    room=$(( MAX_NODES - mine ))
    echo "[$(date +%H:%M:%S)] movable=$npend  mine on $PART=$mine/$MAX_NODES  room=$room"
    (( npend == 0 || room <= 0 )) && return 0

    # rank|jobid|nodes|name, then: rank, then oldest job id first.
    awk -F'|' '
        NR == FNR { target[$1 "|" $2] = $3; next }        # size|arch -> iters
        { name = $4; rank = 3
          if (name ~ /^convert-/) rank = 0
          else {
              split(name, f, "-")                         # eval-<size>-L..-...
              arch = (name ~ /-shallow-/) ? "shallow" : "deep"
              t = target[f[2] "|" arch]
              if (match(name, /-iter[0-9]+$/) && t > 0) {
                  pct = 100.0 * substr(name, RSTART + 5) / t
                  if (pct >= 99.0) rank = 1
                  else for (k = 20; k <= 80; k += 20)
                           if (pct > k - 1.0 && pct < k + 1.0) rank = 2
              }
          }
          printf "%d|%s|%s|%s\n", rank, $2, $3, name }' <(echo "$TARGETS") - <<<"$pend" \
    | LC_ALL=C sort -t'|' -k1,1n -k2,2n \
    | { local waiting=0
        while IFS='|' read -r rank jid n name; do
            if (( n > room )); then
                (( waiting++ == 0 )) && echo "  wait: $jid $name needs $n nodes, room=$room"
                continue
            fi
            if (( DRY )); then
                echo "  would move $jid $name (rank $rank, $n node)"
            elif scontrol update jobid="$jid" partition="$PART" 2>&1; then
                echo "  moved $jid $name (rank $rank)"
            else
                # continue, not break: the same job sorts first every tick, so
                # stopping here would let one unmovable job block the rest.
                echo "  WARN: failed to move $jid $name (trying the next job)"; continue
            fi
            room=$(( room - n ))
            (( room <= 0 )) && break
        done
        (( waiting > 1 )) && echo "  ($waiting jobs waiting for room)"; }
    return 0
}

if (( ONCE || DRY )); then drain_once; exit 0; fi

echo "[preempt-drain] loop start (interval ${INTERVAL}s, max-nodes $MAX_NODES); stop with kill."
while true; do
    drain_once
    sleep "$INTERVAL"
done
