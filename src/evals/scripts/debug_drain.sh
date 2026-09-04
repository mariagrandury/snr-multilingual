#!/bin/bash
# debug_drain.sh — feed PENDING normal-partition convert/eval/bpb jobs through
# the idle `debug` partition, keeping it at the debug-qos cap of **1 running +
# 1 queued**. Each moved job is capped at debug's **1:30** wall via
# `scontrol update partition=debug timelimit=01:30:00`.
#
# Truncating a job to 1:30 is only safe when it does not LOSE work, and all
# three kinds resume:
#
#   convert-*  convert-snr.sh touches .hf_complete per iter and skips what
#              already carries it, so a 1:30 kill costs at most the in-flight
#              iter. The launcher asks for 2h but the job actually runs 12-15
#              min, so filtering on the REQUESTED limit once left conversions
#              stranded on `normal` forever while this script reported
#              "nothing movable" — the exact stall it exists to prevent.
#   bpb-*      score_bpb.py writes each checkpoint's bpb.json before starting
#              the next, and skips what is already written on re-run.
#   eval-*     scripts/eval_worker.py writes each task's results the moment it
#              finishes (per_task/<task>/ in the eval dir), and the auto-eval
#              watcher resubmits a killed job with only the tasks still
#              missing. Until 2026-09-04 evals ran BATCH_TASKS=1 — one lm_eval
#              call writing everything in a single burst at the end, so a 1:30
#              kill of a 4h L100 job discarded 100% of it (../CLAUDE.md bug
#              13) — and were only moved when their own walltime already fit.
#
# Every kind is therefore moved regardless of its requested walltime. A
# truncated job costs one more cold start and one watcher pass to be
# resubmitted, against hours of queue wait on `normal`.
#
# It does NOT submit new jobs — it only MOVES already-pending jobs, so there's
# no risk of duplicates. Stops when nothing pending is left.
#
# Conversions are drained first: they run 12-15 min and are the gate on
# evaluating a cell, so a stuck normal queue blocks the whole eval pipeline.
# Evals next, BPB last — there are hundreds of BPB jobs and they must not
# starve the convert -> eval pipeline.
#
# The loop EXITS once nothing pending is left — it drains a batch, it does not
# stand guard. A job submitted after it exits will sit on `normal` until the
# drainer is started again.
#
# Usage:
#   bash scripts/debug_drain.sh --dry-run        # show what it would move
#   bash scripts/debug_drain.sh                  # loop (default 45s interval)
#   bash scripts/debug_drain.sh --once           # single pass
#   bash scripts/debug_drain.sh --interval 60
#   bash scripts/debug_drain.sh --ensure         # start one in the background
#                                                # if none is running; else no-op
set -uo pipefail
INTERVAL=45; ONCE=0; DRY=0; ENSURE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --once)     ONCE=1; shift ;;
    --dry-run)  DRY=1; shift ;;
    --ensure)   ENSURE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")

# --ensure: what job submitters call. The drainer's loop EXITS once nothing
# movable is left, so it has to be (re)started whenever new jobs are queued —
# but starting one per launch would leave several racing for the same two
# debug slots. So: start one only if none is running. "Running" = any
# debug_drain.sh process that is not itself an --ensure call, which covers
# both the form spawned below and one started by hand. A concurrent --dry-run
# also reads as running and skips the start; the next submitter's call picks
# it up, and erring towards "don't start a second one" is the safe direction.
if (( ENSURE )); then
    if pgrep -af "debug_drain\.sh" | grep -qv -- "--ensure"; then
        echo "[debug-drain] already running"
        exit 0
    fi
    LOGDIR=$(dirname "$(dirname "$SELF")")/logs   # src/evals/logs (gitignored)
    mkdir -p "$LOGDIR"
    LOG="$LOGDIR/debug_drain_$(date +%Y%m%d_%H%M%S).log"
    setsid bash "$SELF" --interval "$INTERVAL" >"$LOG" 2>&1 < /dev/null &
    echo "[debug-drain] started (pid $!) -> $LOG"
    exit 0
fi

drain_once() {
    local npend ndebug slots cand jid jn secs
    npend=$(squeue --me -h -p normal -t PD -o "%j" 2>/dev/null | grep -cE '^(eval|convert|bpb)-')
    ndebug=$(squeue --me -h -p debug -t PD,R,CG -o "%i" 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] pending=$npend  debug=$ndebug/2"
    (( npend == 0 )) && return 1          # nothing pending left → caller exits
    slots=$(( 2 - ndebug ))
    (( slots <= 0 )) && return 0          # debug full (1 run + 1 queued)
    for (( s=0; s<slots; s++ )); do
        # Rank 0 convert, 1 eval, 2 bpb (see the header); within a class,
        # shortest walltime first.
        cand=$(squeue --me -h -p normal -t PD -o "%l|%i|%j" 2>/dev/null | grep -E '\|(eval|convert|bpb)-' | \
            awk -F'|' '{n=split($1,a,":"); sec=(n==3?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]);
                        rank=($3 ~ /^convert-/) ? 0 : (($3 ~ /^eval-/) ? 1 : 2);
                        print rank"|"sec"|"$2"|"$3}' | \
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
    drain_once || { echo "[debug-drain] nothing pending left — done."; break; }
    sleep "$INTERVAL"
done
