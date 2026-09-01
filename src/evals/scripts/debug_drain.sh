#!/bin/bash
# debug_drain.sh — feed PENDING normal-partition convert/eval jobs through the idle
# `debug` partition, shortest-walltime first, keeping it at the debug-qos cap
# of **1 running + 1 queued**. Each moved job is capped at debug's **1:30**
# wall via `scontrol update partition=debug timelimit=01:30:00`.
#
# Whether an over-cap job may be moved depends on whether truncating it LOSES
# work, which differs per job kind:
#
#   convert-*  RESUMABLE — convert-snr.sh touches .hf_complete per iter and
#              skips what already carries it, so a 1:30 kill costs at most the
#              in-flight iter. Moved regardless of walltime: the launcher asks
#              for 2h but the job actually runs 12-15 min, so filtering on the
#              REQUESTED limit left conversions stranded on `normal` forever
#              while this script reported "nothing movable" — the exact stall
#              it exists to prevent.
#   bpb-*      RESUMABLE — score_bpb.py writes each checkpoint's bpb.json
#              before starting the next, and skips what is already written on
#              re-run. A 1:30 chunk therefore advances a cell by however many
#              checkpoints it got through, and the next chunk continues. These
#              are moved regardless of their requested walltime; that is the
#              whole point of draining them.
#   eval-*     NOT resumable under BATCH_TASKS=1 (see below), so only moved
#              when its own walltime already fits the cap.
#
# This used to move anything, assuming a truncated job leaves partial progress
# on disk. It does not, for the auto-eval path: that runs BATCH_TASKS=1, one
# lm_eval call for every task, and lm_eval writes in a single burst at the end
# — on a completed 45-minute job all 538 samples files AND the results file
# share one filename timestamp, with mtimes spanning 11 seconds, and per_task/
# is empty (../CLAUDE.md bug 13). So a 1:30 kill of a 4h L100 job discarded
# 100% of the work, and because nothing was written the next watcher pass
# resubmitted it to be killed again. Per-task idempotency is real, but it only
# helps ACROSS jobs that each finished something.
#
# With measured eval times (plan/compute-budget.md) only L1 (~8-13 min), L2
# (~14-23) and L8 (~40-70) fit 1:30; L15 upward cannot finish in a debug slot.
# Using debug for those needs the job SPLIT rather than truncated —
# evaluate.sbatch already has NUM_SPLITS/SPLIT_INDEX and aggregate_splits.sbatch
# — which belongs at submission time in auto_evals_cscs.py, not here: splitting
# a pending job means cancelling it and submitting N new ones, exactly the
# duplicate risk this script exists to avoid.
#
# It does NOT submit new jobs — it only MOVES already-pending jobs, so there's
# no risk of duplicates. Stops when nothing MOVABLE is left; over-cap jobs are
# reported as "too-long" each tick and stay on normal.
#
# Conversions are drained first: they run 12-15 min and are the gate on
# evaluating a cell, so a stuck normal queue blocks the whole eval pipeline.
#
# The loop EXITS once nothing movable is left — it drains a batch, it does not
# stand guard. A job submitted after it exits will sit on `normal` until the
# drainer is started again.
#
# Usage:
#   bash scripts/debug_drain.sh --dry-run        # show what it would move
#   bash scripts/debug_drain.sh                  # loop (default 45s interval)
#   bash scripts/debug_drain.sh --once           # single pass
#   bash scripts/debug_drain.sh --interval 60
set -uo pipefail
INTERVAL=45; ONCE=0; DRY=0
DEBUG_CAP_SEC=5400        # the debug partition's 1:30 wall (scontrol show partition debug)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --once)     ONCE=1; shift ;;
    --dry-run)  DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

drain_once() {
    local npend nlong ndebug slots cand jid jn secs
    # Count only what is MOVABLE (walltime <= the cap). Counting every pending
    # job would keep the loop spinning forever once the only ones left are the
    # long-L evals it must not touch.
    npend=$(squeue --me -h -p normal -t PD -o "%l|%j" 2>/dev/null | grep -E '\|(eval|convert|bpb)-' | \
        awk -F'|' -v cap="$DEBUG_CAP_SEC" '{n=split($1,a,":");
            sec=(n==3?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]);
            if (sec <= cap || $2 ~ /^(bpb|convert)-/) c++} END {print c+0}')
    nlong=$(squeue --me -h -p normal -t PD -o "%l|%j" 2>/dev/null | grep -E '\|(eval|convert|bpb)-' | \
        awk -F'|' -v cap="$DEBUG_CAP_SEC" '{n=split($1,a,":");
            sec=(n==3?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]);
            if (sec > cap && $2 !~ /^(bpb|convert)-/) c++} END {print c+0}')
    ndebug=$(squeue --me -h -p debug -t PD,R,CG -o "%i" 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] movable=$npend  too-long(eval)=$nlong  debug=$ndebug/2"
    (( npend == 0 )) && return 1          # nothing movable left → caller exits
    slots=$(( 2 - ndebug ))
    (( slots <= 0 )) && return 0          # debug full (1 run + 1 queued)
    for (( s=0; s<slots; s++ )); do
        # Rank 0 convert, 1 eval, 2 bpb; within a class, shortest walltime.
        # Converts first because a cell cannot be evaluated until its HF
        # snapshot exists, so one stuck behind a queue blocks everything
        # downstream — that is exactly how the pipeline stalled before. bpb
        # last because there are hundreds of them and they must not starve the
        # convert -> eval pipeline. Over-cap jobs are skipped unless resumable
        # (see the header).
        cand=$(squeue --me -h -p normal -t PD -o "%l|%i|%j" 2>/dev/null | grep -E '\|(eval|convert|bpb)-' | \
            awk -F'|' -v cap="$DEBUG_CAP_SEC" '{n=split($1,a,":"); sec=(n==3?a[1]*3600+a[2]*60+a[3]:a[1]*60+a[2]);
                        resumable=($3 ~ /^(bpb|convert)-/);
                        if (sec > cap && !resumable) next;
                        rank=($3 ~ /^convert-/) ? 0 : (resumable ? 2 : 1);
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
    drain_once || { echo "[debug-drain] nothing movable left (over-cap evals stay on normal) — done."; break; }
    sleep "$INTERVAL"
done
