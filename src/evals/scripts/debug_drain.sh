#!/bin/bash
# debug_drain.sh — feed PENDING normal-partition convert/eval jobs through the idle
# `debug` partition, shortest-walltime first, keeping it at the debug-qos cap
# of **1 running + 1 queued**. Each moved job is capped at debug's **1:30**
# wall via `scontrol update partition=debug timelimit=01:30:00`.
#
# Eval jobs are per-task idempotent, so a 1:30 debug chunk makes partial
# progress that survives on disk; jobs whose remaining work fits in 1:30
# (e.g. the ~1:17 anchors) finish outright, longer ones get a 1:30 chunk and
# their leftover tasks are picked up by a later launcher re-run.
#
# It does NOT submit new jobs — it only MOVES already-pending jobs, so there's
# no risk of duplicates. Stops when no pending normal eval/convert jobs remain.
#
# Conversions are drained too: they are short (~3 min) and are the gate on
# evaluating a cell, so a stuck normal queue blocks the whole eval pipeline.
#
# Usage:
#   bash scripts/debug_drain.sh --dry-run        # show what it would move
#   bash scripts/debug_drain.sh                  # loop (default 45s interval)
#   bash scripts/debug_drain.sh --once           # single pass
#   bash scripts/debug_drain.sh --interval 60
set -uo pipefail
INTERVAL=45; ONCE=0; DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --once)     ONCE=1; shift ;;
    --dry-run)  DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

drain_once() {
    local npend ndebug slots cand jid jn secs
    npend=$(squeue --me -h -p normal -t PD -o "%j" 2>/dev/null | grep -cE '^(eval|convert)-')
    ndebug=$(squeue --me -h -p debug -t PD,R,CG -o "%i" 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] pending(normal,eval)=$npend  debug=$ndebug/2"
    (( npend == 0 )) && return 1          # nothing left to drain → caller exits
    slots=$(( 2 - ndebug ))
    (( slots <= 0 )) && return 0          # debug full (1 run + 1 queued)
    for (( s=0; s<slots; s++ )); do
        # Conversions first, then evals; within a class, shortest walltime.
        # A cell cannot be evaluated until its HF snapshot exists, so a convert
        # stuck behind a queue of evals blocks everything downstream of it —
        # that is exactly how the pipeline stalled before. Rank 0 = convert,
        # 1 = eval, then seconds (HH:MM:SS or MM:SS).
        cand=$(squeue --me -h -p normal -t PD -o "%l|%i|%j" 2>/dev/null | grep -E '\|(eval|convert)-' | \
            awk -F'|' '{n=split($1,a,":"); sec=(n==3?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]);
                        rank=($3 ~ /^convert-/) ? 0 : 1; print rank"|"sec"|"$2"|"$3}' | \
            sort -t'|' -k1,1n -k2,2n | head -1 | cut -d'|' -f2-)
        [ -z "$cand" ] && return 0
        secs=${cand%%|*}; jid=$(echo "$cand"|cut -d'|' -f2); jn=$(echo "$cand"|cut -d'|' -f3)
        if (( DRY )); then
            echo "  would move $jid $jn (wall=${secs}s) -> debug @1:30"
            # in dry-run, can't actually free the slot; avoid infinite same-pick
            return 0
        fi
        if scontrol update jobid="$jid" partition=debug timelimit=01:30:00 2>/dev/null; then
            echo "  moved $jid $jn -> debug @1:30"
        else
            echo "  WARN: failed to move $jid $jn (skipping this tick)"
            return 0
        fi
        sleep 2
    done
    return 0
}

if (( ONCE || DRY )); then drain_once; exit 0; fi

echo "[debug-drain] loop start (interval ${INTERVAL}s); stop with TaskStop / kill."
while true; do
    drain_once || { echo "[debug-drain] no pending normal eval jobs left — done."; break; }
    sleep "$INTERVAL"
done
