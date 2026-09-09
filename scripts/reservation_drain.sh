#!/bin/bash
# reservation_drain.sh — move my PENDING `normal` jobs into the reservation,
# within three budgets:
#
#   --max-nodes N (default MAX_MY_NODES below)  nodes my jobs may hold in the
#                 reservation at once. Counted over my running AND pending jobs
#                 already in it — a pending job starts the moment nodes free
#                 up, so it is committed.
#   MIN_FREE      nodes that must stay free for everyone else. "Free" is the
#                 reservation's size minus nodes down/in maintenance minus the
#                 nodes of every job (anyone's, running or pending) already in
#                 it — sinfo reports every node of an active reservation as
#                 `reserved`, idle or not, so it cannot be read off node states.
#   --hours H (default 12)  everything moved must be able to finish within H
#                 hours of the script's start: a job is moved only while its
#                 walltime still fits in the time left, so with 4h left a 5h
#                 job stays on `normal` even if nodes are free. The loop exits
#                 when the H hours are up. (A moved job may still pend inside
#                 the reservation for a while; the check is on walltime alone.)
#
# Order: jobs whose name contains --priority STR first (if given), then most
# node-hours (nodes x walltime) first, then job name alphabetically. A job
# that does not fit the remaining room is passed over and smaller ones behind
# it backfill the room (15 nodes left after three 21-node pretrains → 15
# 1-node evals/BPB go in). The big job is not starved by that: between ticks
# room only grows, and when a 21-node pretrain finishes the room jumps by 21
# at once, so the next tick finds the waiting pretrain — first in order —
# fitting before any eval is considered. A job that can never fit (more nodes
# than MAX_MY_NODES) is skipped with a message.
#
# Like debug_drain.sh it only MOVES already-pending jobs (scontrol update
# reservation=...), never submits, and the loop exits once nothing pending is
# left on `normal`. Where debug_drain.sh reroutes to the debug partition, this
# keeps the job on `normal` (the reservation's partition) and untouched
# otherwise: same walltime, same account.
#
# Usage:
#   bash scripts/reservation_drain.sh --dry-run          # plan only
#   bash scripts/reservation_drain.sh                    # loop (default 15 min)
#   bash scripts/reservation_drain.sh --once
#   bash scripts/reservation_drain.sh --priority eval    # evals first
#   bash scripts/reservation_drain.sh --interval 600 --priority L30
#   bash scripts/reservation_drain.sh --hours 6 --max-nodes 42
set -uo pipefail
RES=SD-69241-apertus-1-5-0
MAX_MY_NODES=63
MIN_FREE=50

INTERVAL=900; ONCE=0; DRY=0; PRIO=""; HOURS=12
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval)  INTERVAL="$2"; shift 2 ;;
    --priority)  PRIO="$2"; shift 2 ;;
    --hours)     HOURS="$2"; shift 2 ;;
    --max-nodes) MAX_MY_NODES="$2"; shift 2 ;;
    --once)      ONCE=1; shift ;;
    --dry-run)   DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
DEADLINE=$(( $(date +%s) + HOURS*3600 ))

sum() { awk '{s+=$1} END{print s+0}'; }

drain_once() {
    local q pend npend res size nodes unavail all_res my_res free room left
    left=$(( DEADLINE - $(date +%s) ))
    if (( left <= 0 )); then echo "[$(date +%H:%M:%S)] the $HOURS h window is over"; return 1; fi
    # squeue's exit status, not its output: a failed squeue must not read as an
    # empty queue (debug_drain.sh learned that on 2026-09-07).
    if ! q=$(squeue --me -h -p normal -t PD -o "%v|%i|%D|%l|%j" 2>/dev/null); then
        echo "[$(date +%H:%M:%S)] squeue failed — retrying next tick"; return 0
    fi
    pend=$(awk -F'|' '$1=="(null)"' <<<"$q")
    npend=$(grep -c . <<<"$pend")
    if (( npend == 0 )); then echo "[$(date +%H:%M:%S)] nothing pending left on normal"; return 1; fi

    res=$(scontrol show reservation "$RES" 2>/dev/null) || { echo "reservation $RES not found"; return 0; }
    size=$(grep -oP 'NodeCnt=\K\d+' <<<"$res")
    nodes=$(grep -oP 'Nodes=\K\S+' <<<"$res")
    unavail=$(sinfo -h -n "$nodes" -o "%T %D" 2>/dev/null | awk '/maint|down|drain|fail|unk/{s+=$2} END{print s+0}')
    all_res=$(squeue -h -R "$RES" -t PD,R,CG -o "%D" 2>/dev/null | sum)
    my_res=$(squeue --me -h -R "$RES" -t PD,R,CG -o "%D" 2>/dev/null | sum)
    free=$(( size - unavail - all_res ))
    room=$(( free - MIN_FREE ))
    (( MAX_MY_NODES - my_res < room )) && room=$(( MAX_MY_NODES - my_res ))
    echo "[$(date +%H:%M:%S)] pending=$npend  reservation: free=$free/$size (keep $MIN_FREE)  mine=$my_res/$MAX_MY_NODES  room=$room  time left=$(( left/3600 ))h$(( left%3600/60 ))m"

    # match|node_hours|name|jid|nodes|walltime|walltime_secs — priority match
    # first, then node-hours descending, then name.
    awk -F'|' -v prio="$PRIO" '
        function secs(t,  d, p, a, n) {
            d = 0; if (t ~ /-/) { split(t, p, "-"); d = p[1]; t = p[2] }
            n = split(t, a, ":")
            if (n == 3) return d*86400 + a[1]*3600 + a[2]*60 + a[3]
            if (n == 2) return d*86400 + a[1]*60 + a[2]
            return 0                       # UNLIMITED / NOT_SET
        }
        { m = (prio != "" && index($5, prio) > 0) ? 0 : 1
          printf "%d|%.2f|%s|%s|%s|%s|%d\n", m, $3*secs($4)/3600, $5, $2, $3, $4, secs($4) }' <<<"$pend" \
    | LC_ALL=C sort -t'|' -k1,1n -k2,2gr -k3,3 \
    | { waiting=0; toolong=0; while IFS='|' read -r _ nh name jid n wall wsecs; do
        if (( n > MAX_MY_NODES )); then
            echo "  skip $jid $name: $n nodes > max-nodes=$MAX_MY_NODES"; continue
        fi
        if (( wsecs > left )); then
            (( toolong++ == 0 )) && echo "  skip $jid $name: walltime $wall does not fit in the time left"
            continue
        fi
        if (( n > room )); then
            (( waiting++ == 0 )) && echo "  wait: $jid $name needs $n nodes (${nh} node-h), room=$room — backfilling with smaller jobs"
            continue
        fi
        if (( DRY )); then
            echo "  would move $jid $name ($n nodes x $wall = ${nh} node-h)"
        elif scontrol update jobid="$jid" reservation="$RES" 2>&1; then
            echo "  moved $jid $name ($n nodes x $wall = ${nh} node-h)"
        else
            echo "  WARN: failed to move $jid $name (stopping this tick)"; break
        fi
        room=$(( room - n ))
    done
      (( waiting > 1 )) && echo "  ($waiting jobs waiting for room)"
      (( toolong > 1 )) && echo "  ($toolong jobs too long for the time left)"; }
    return 0
}

if (( ONCE || DRY )); then drain_once; exit 0; fi

echo "[reservation-drain] loop start (interval ${INTERVAL}s, priority='${PRIO}', window ${HOURS}h, max-nodes $MAX_MY_NODES); stop with kill."
while true; do
    drain_once || { echo "[reservation-drain] done."; break; }
    sleep "$INTERVAL"
done
