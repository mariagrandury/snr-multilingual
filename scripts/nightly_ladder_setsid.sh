#!/bin/bash
# nightly_ladder_setsid.sh — arm nightly.sh without systemd. (The name
# predates the evals/ladder split; it arms either mode.)
#
# The fallback for when `loginctl enable-linger` is refused, which is what the
# systemd --user timer needs in order to outlive your login session. This
# survives logout, but NOT a login-node reboot: re-run it after one. The timer
# is the better answer where lingering is allowed — see nightly.sh.
#
# It sleeps until the mode's next hour (Europe/Zurich), runs that pass, and
# loops, so one invocation covers every night until the node restarts. Costs one sleeping
# pid against the 1000-per-user cap (645 in use when this was written).
#
# Usage:
#   bash scripts/nightly_ladder_setsid.sh ladder start   # arm (idempotent)
#   bash scripts/nightly_ladder_setsid.sh ladder status
#   bash scripts/nightly_ladder_setsid.sh evals  start   # the 00:00 pass
#
# The hour defaults to the mode's own (evals 00, ladder 04); HOUR=23 overrides.
# nightly.sh takes a cross-node lock, so arming this on two nodes is harmless.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
MODE=${1:-}
case "$MODE" in
  evals|ladder) shift ;;
  *) echo "usage: $0 {evals|ladder} {start|status|stop}" >&2; exit 2 ;;
esac
case "$MODE" in evals) DEFAULT_HOUR=00 ;; ladder) DEFAULT_HOUR=04 ;; esac
HOUR=${HOUR:-$DEFAULT_HOUR}
LOG_DIR=${NIGHTLY_LOG_DIR:-/iopsstor/scratch/cscs/$USER/logs/nightly}
PIDFILE=$LOG_DIR/setsid-$MODE.pid
mkdir -p "$LOG_DIR"

alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null; }

case "${1:-status}" in
  start)
    if alive; then echo "already armed (pid $(cat "$PIDFILE")) — nothing to do"; exit 0; fi
    # setsid detaches from this terminal's session, so a logout (SIGHUP to the
    # session) does not reach it.
    setsid nohup bash -c '
        while true; do
            now=$(date +%s)
            next=$(date -d "today '"$HOUR"':00" +%s)
            [ "$next" -le "$now" ] && next=$(date -d "tomorrow '"$HOUR"':00" +%s)
            sleep $(( next - now ))
            bash '"$REPO"'/scripts/nightly.sh '"$MODE"'
        done' >>"$LOG_DIR/setsid-$MODE.log" 2>&1 &
    echo $! > "$PIDFILE"
    echo "armed: pid $(cat "$PIDFILE"), next run $(date -d "today $HOUR:00" +'%F %T %Z' | sed "s/^/ /")"
    echo "  (if that is in the past it waits for tomorrow)"
    echo "  log: $LOG_DIR/setsid-$MODE.log   disarm: bash scripts/nightly_ladder_setsid.sh $MODE stop"
    ;;
  status)
    if alive; then
        echo "armed: pid $(cat "$PIDFILE") since $(ps -o lstart= -p "$(cat "$PIDFILE")" 2>/dev/null)"
        echo "last run: $(cat "$LOG_DIR/last-run-$MODE.txt" 2>/dev/null || echo 'none yet')"
    else
        echo "not armed"
    fi
    ;;
  stop)
    if alive; then kill "$(cat "$PIDFILE")" && echo "disarmed (pid $(cat "$PIDFILE"))"
    else echo "not armed"; fi
    ;;
  *) echo "usage: $0 {evals|ladder} {start|status|stop}" >&2; exit 2 ;;
esac
