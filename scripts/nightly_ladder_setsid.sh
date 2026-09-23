#!/bin/bash
# nightly_ladder_setsid.sh — arm nightly_ladder.sh without systemd.
#
# The fallback for when `loginctl enable-linger` is refused, which is what the
# systemd --user timer needs in order to outlive your login session. This
# survives logout, but NOT a login-node reboot: re-run it after one. The timer
# is the better answer where lingering is allowed — see nightly_ladder.sh.
#
# It sleeps until the next 06:00 Europe/Zurich, runs the refresh, and loops, so
# one invocation covers every night until the node restarts. Costs one sleeping
# pid against the 1000-per-user cap (645 in use when this was written).
#
# Usage:
#   bash scripts/nightly_ladder_setsid.sh start    # arm it (idempotent)
#   bash scripts/nightly_ladder_setsid.sh status   # is it armed, and for when
#   bash scripts/nightly_ladder_setsid.sh stop     # disarm
#
# HOUR=23 bash scripts/nightly_ladder_setsid.sh start   # a different hour

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
HOUR=${HOUR:-06}
LOG_DIR=${NIGHTLY_LOG_DIR:-/iopsstor/scratch/cscs/$USER/logs/nightly-ladder}
PIDFILE=$LOG_DIR/setsid.pid
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
            bash '"$REPO"'/scripts/nightly_ladder.sh
        done' >>"$LOG_DIR/setsid.log" 2>&1 &
    echo $! > "$PIDFILE"
    echo "armed: pid $(cat "$PIDFILE"), next run $(date -d "today $HOUR:00" +'%F %T %Z' | sed "s/^/ /")"
    echo "  (if that is in the past it waits for tomorrow)"
    echo "  log: $LOG_DIR/setsid.log    disarm: bash scripts/nightly_ladder_setsid.sh stop"
    ;;
  status)
    if alive; then
        echo "armed: pid $(cat "$PIDFILE") since $(ps -o lstart= -p "$(cat "$PIDFILE")" 2>/dev/null)"
        echo "last run: $(cat "$LOG_DIR/last-run.txt" 2>/dev/null || echo 'none yet')"
    else
        echo "not armed"
    fi
    ;;
  stop)
    if alive; then kill "$(cat "$PIDFILE")" && echo "disarmed (pid $(cat "$PIDFILE"))"
    else echo "not armed"; fi
    ;;
  *) echo "usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac
