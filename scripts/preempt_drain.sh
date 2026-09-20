#!/bin/bash
# preempt_drain.sh — move my PENDING convert/eval/bpb jobs from `normal` to
# `preemptable`, keeping at most --max-nodes (default 100) nodes of mine there.
#
# Why: `normal` is capped by its QOS at 480 nodes for the WHOLE partition, so
# with the cluster full my jobs sit on QOSGrpNodeLimit for hours — on
# 2026-09-18 sbatch --test-only put the same 1-node eval at 13:34 on `normal`
# and 04:36 on `preemptable`. `preemptable` has all 1343 nodes and no group
# cap; the price is preemption (PreemptMode=REQUEUE, 4 min grace).
#
# That price is small for exactly these kinds, and all RESUME:
#
#   build-*    NOT MOVED — they have to be SUBMITTED here instead
#              (`BUILD_PARTITION=preemptable data/launch_builds.sh`). The first
#              six moved (3448609-12, 3449776-7) were placed on ONE node that
#              already ran another user's job and were cancelled by uid 0
#              sixteen seconds after starting, before writing a line of output,
#              so no successor was ever queued and six chains died. A build
#              asks for 32 CPUs and `normal`, being OverSubscribe=EXCLUSIVE,
#              has always given it a whole node anyway (AllocCPUS=288);
#              `preemptable` is FORCE:1 and packs them. The launcher therefore
#              asks for --exclusive, and scontrol cannot add that to a job
#              already queued — hence submit, never move.
#   convert-*  convert-snr.sh touches .hf_complete per iter and skips what
#              carries it, so a preemption costs at most the in-flight iter.
#   eval-*     scripts/eval_worker.py writes each task's results as it
#              finishes (per_task/<task>/), and the auto-eval watcher
#              resubmits with only the tasks still missing.
#   pretrain-* ONLY the runs listed in the ladder below, and only when they
#              were submitted by launch_trainings.py --partition preemptable:
#              that adds --requeue, and the wrapper's SIGTERM trap forwards
#              SIGUSR2 so Megatron checkpoints inside the 240s grace and
#              resumes when the job comes back. A pretrain job submitted
#              WITHOUT those (the `normal` default) loses everything back to
#              its last save and never returns — moving one here is a loss,
#              which is why the filter takes only the named sizes.
#   *bpb*      score_bpb.py writes each checkpoint's bpb.json before starting
#              the next, and skips what is already written on re-run.
#
# Clariden has JobRequeue=0, so a preempted job is cancelled rather than
# requeued unless it asked for it (only the pretrain runs do, see above); the
# next watcher pass resubmits a convert or eval and it picks up where it
# stopped, and a build's successor is already queued. BPB is not the
# watcher's: score_bpb.sbatch
# chains itself, queuing its successor BEFORE scoring, so a preempted link's
# successor is already pending (and is moved here in turn). But a link killed
# before its FIRST checkpoint lands trips the chain's no-progress guard and the
# successor ends the chain — re-run launch_bpb.sh, which skips scored cells and
# cells with a job in flight. That, and their number, is why BPB goes last.
#
# A pretrain job below the named rungs is never moved, and neither is one
# without --requeue: 21 nodes lost to a preemption costs up to a save interval
# of training. Unlike debug_drain.sh nothing is truncated here — preemptable's
# 24 h limit is above every eval walltime.
#
# Order — the priority ladder (2026-09-20):
#
#   (build-* would head this list; it is excluded above until the 09-20
#    cancellations are explained)
#   1  convert-*                     the gate on every eval downstream
#   2  eval-* of the FINAL ckpt      the headline number for each model
#   3  pretrain-3B-*                 the extrapolation rung
#   4  pretrain-1.7B-*-deep-*        the reference rung
#   5  pretrain-1B-*-deep-*
#   6  eval-* at 20/40/60/80 %       the training-curve points
#   7  every other eval-*            the rest of the k/20 grid
#   8  any job with "bpb" in its name  last: hundreds of them, and they must
#                                      not starve convert -> eval
#
# Walltime is NOT touched: `preemptable` allows 24h where `normal` allows 12,
# but a job's limit can only be LOWERED after submission ("Access/permission
# denied" for anyone but an operator, verified 2026-09-20). A moved job
# therefore keeps the wall it was submitted with; to use the full 24h, submit
# there in the first place — `launch_trainings.py --partition preemptable`
# (23:59:00 + --requeue).
#
# The fraction comes from the size's own schedule (launch_trainings.schedule_for,
# the same source the watcher's due_iters uses), not from the iter number, so
# 81000 at 1.7B and 28800 at 600M both read as 100 %. If that lookup fails the
# script still runs — every eval then ranks 7 and only the kinds above them
# (builds, converts, the pretrain rungs) keep their priority.
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
MAX_NODES=100
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
    # Jobs already placed in a reservation (%v) stay there. Eligibility is this
    # line ("bpb" is matched anywhere in the name; every other kind is
    # anchored): build/convert/eval unconditionally, and of the pretrain runs
    # only 3B and the deep 1.7B/1B (builds excluded, see the header) — a shallow
    # or smaller rung is cheap enough
    # to wait for `normal`, and every pretrain job is checked for --requeue
    # again below before it is actually moved.
    pend=$(awk -F'|' '$1=="(null)" && ($4 ~ /^(convert|eval)-/ || $4 ~ /bpb/ \
                       || $4 ~ /^pretrain-3B-/ \
                       || ($4 ~ /^pretrain-(1\.7B|1B)-/ && $4 ~ /-deep-/))' <<<"$q")
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
        { name = $4; rank = 7                             # every other eval-*
          if (name ~ /bpb/) rank = 8
          else if (name ~ /^build-/) rank = 0
          else if (name ~ /^convert-/) rank = 1
          else if (name ~ /^pretrain-3B-/) rank = 3
          else if (name ~ /^pretrain-1\.7B-/) rank = 4
          else if (name ~ /^pretrain-1B-/) rank = 5
          else {
              split(name, f, "-")                         # eval-<size>-L..-...
              arch = (name ~ /-shallow-/) ? "shallow" : "deep"
              t = target[f[2] "|" arch]
              # Not anchored at the end: a reformulated eval carries a family
              # suffix (-rf, -rfgm) after the iter, and anchoring sent every
              # one of them to the bottom rank.
              if (match(name, /-iter[0-9]+/) && t > 0) {
                  it = substr(name, RSTART + 5); sub(/[^0-9].*/, "", it)
                  pct = 100.0 * it / t
                  if (pct >= 99.0) rank = 2
                  else for (k = 20; k <= 80; k += 20)
                           if (pct > k - 1.0 && pct < k + 1.0) rank = 6
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
            # A pretrain job is only preemption-safe if it can come back:
            # --requeue (launch_trainings.py --partition preemptable) plus the
            # wrapper's SIGTERM trap. Without it a preemption is a cancelled
            # run and a lost save interval on 21 nodes, so ask the controller
            # rather than assume — Requeue is 0/1 per job.
            if [[ $name == pretrain-* ]] \
               && [[ $(squeue -h -j "$jid" -O Requeue 2>/dev/null | tr -d ' ') != 1 ]]; then
                echo "  skip: $jid $name submitted without --requeue (relaunch with --partition $PART)"
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
