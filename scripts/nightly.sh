#!/bin/bash
# nightly.sh {evals|ladder} — the two unattended nightly passes. One file,
# two modes, because they share all the awkward parts: the cross-node lock,
# the environment a systemd unit does not give you, and the logging.
#
#   evals   00:00  queue the bpb chain, then restart the auto-eval watcher
#   ladder  04:00  publish the ladder report, then submit the analysis and,
#                  after it, the curves-only job
#
# ONE INSTANCE, ACROSS EVERY LOGIN NODE. $HOME is shared NFS, so
# ~/.config/systemd/user/timers.target.wants/ is visible to EVERY login node's
# user manager and each one fires the timer. On 2026-10-02 ln001, ln002 and
# ln003 all ran at 06:00: three `ladder-refresh` jobs, and — worse — three
# concurrent `ladder_report.py --publish --push-git` writing the same CSVs and
# pushing the same branch. So every mode takes an atomic lock on the shared
# filesystem first. mkdir is the atomic primitive; the holder records
# host/pid/epoch so a crashed run cannot block tomorrow (NIGHTLY_LOCK_TTL,
# default 12 h). `ladder` also asks Slurm, right before it submits, whether a
# refresh is already in flight — belt and braces for a hand-run.
#
# Credentials, checked 2026-09-23:
#   - origin is git@github.com:…, and a timer has no SSH agent — but
#     ~/.ssh/id_ed25519 has no passphrase, so the push works unattended.
#   - HF_TOKEN comes from ~/.bashrc, which does NOT early-return for
#     non-interactive shells, so sourcing it below is enough.
#
# THE PUSH-FAILURE FALLBACK (`ladder`). ladder_report.publish() wraps both
# pushes in try/except and prints "[publish] … skipped" — it exits 0 either
# way. The report is never lost: publish() copies it to capstor BEFORE it
# pushes. But refresh_analysis.sh installs the report by fetching the orphan
# branch, so after a failed push the fetch SUCCEEDS and quietly hands the
# analysis yesterday's numbers. This script therefore reads publish()'s
# stderr, installs the freshly generated CSVs from disk when a push did not
# land, and then proves with cmp that what the analysis will read is
# byte-identical to what was just generated — refusing to spend hours if not.
#
# Usage:
#   bash scripts/nightly.sh ladder            # or: evals
#   bash scripts/nightly.sh ladder --dry-run  # print the plan, touch nothing
#   NIGHTLY_NO_SUBMIT=1 bash scripts/nightly.sh ladder   # stop before sbatch
#   touch $LOG_DIR/PAUSE_REFRESH   # ladder: publish, skip the analysis submit
#
# Arm both (once; the enable reaches every login node because $HOME is shared,
# and the lock is what keeps that from meaning three runs):
#   loginctl enable-linger
#   systemctl --user enable --now nightly-evals.timer nightly-ladder.timer
#   systemctl --user list-timers 'nightly-*'
# scripts/nightly_ladder_setsid.sh is the agent-free fallback if
# enable-linger is refused (its name predates the two modes) — it survives logout but not a login-node reboot.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD

MODE=${1:-}
DRY=0
[ "${2:-}" = "--dry-run" ] && DRY=1
case "$MODE" in
  evals|ladder) ;;
  *) echo "usage: $0 {evals|ladder} [--dry-run]" >&2; exit 2 ;;
esac

# Logs live OUTSIDE the repo: this runs nightly and must not churn git status.
LOG_DIR=${NIGHTLY_LOG_DIR:-/iopsstor/scratch/cscs/$USER/logs/nightly}
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y-%m-%d)
LOG=$LOG_DIR/$STAMP.$MODE.log
STATUS=$LOG_DIR/last-run-$MODE.txt

# A systemd user service starts with almost no environment: ~/.bashrc is what
# carries HF_TOKEN, and conda is what carries git-lfs and matplotlib.
source ~/.bashrc >/dev/null 2>&1 || true
source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr
# Overridable so the branches below can be exercised against a fixture with no
# remote, the way refresh_analysis.sh already takes PY.
PY=${PY:-python3.11}
GIT=${GIT:-git}
SBATCH=${SBATCH:-sbatch}
SQUEUE=${SQUEUE:-squeue}

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
FAILED=()
NOTES=()

finish() {
    local rc=${1:-0}
    if [ ${#FAILED[@]} -gt 0 ]; then
        say "FAILED: ${FAILED[*]}"
        { printf '%s  FAILED: %s\n' "$(date '+%F %T')" "${FAILED[*]}"
          printf '  %s\n' "${NOTES[@]+"${NOTES[@]}"}"
          printf '  log: %s\n' "$LOG"; } > "$STATUS"
        exit 1
    fi
    say "done"
    { printf '%s  OK\n' "$(date '+%F %T')"
      printf '  %s\n' "${NOTES[@]+"${NOTES[@]}"}"
      printf '  log: %s\n' "$LOG"; } > "$STATUS"
    exit "$rc"
}

# --- the cross-node lock -----------------------------------------------------
LOCK=$LOG_DIR/.$MODE.lock
take_lock() {
    local now owner_host owner_pid owner_epoch age
    now=$(date +%s)
    if mkdir "$LOCK" 2>/dev/null; then
        printf '%s %s %s\n' "$(hostname)" "$$" "$now" > "$LOCK/owner"
        trap drop_lock EXIT
        return 0
    fi
    owner_epoch=0
    read -r owner_host owner_pid owner_epoch < "$LOCK/owner" 2>/dev/null || true
    case ${owner_epoch:-0} in ''|*[!0-9]*) owner_epoch=0 ;; esac
    age=$(( now - owner_epoch ))
    if (( age > ${NIGHTLY_LOCK_TTL:-43200} )); then
        say "WARN: taking over a lock held by ${owner_host:-?}:${owner_pid:-?} for ${age}s"
        printf '%s %s %s\n' "$(hostname)" "$$" "$now" > "$LOCK/owner"
        trap drop_lock EXIT
        return 0
    fi
    say "another instance holds $MODE (${owner_host:-?}:${owner_pid:-?}, ${age}s ago) — nothing to do"
    return 1
}
# Removes only what take_lock created.
drop_lock() { rm -f "$LOCK/owner" 2>/dev/null; rmdir "$LOCK" 2>/dev/null; return 0; }

# --- modes -------------------------------------------------------------------
do_evals() {
    say "queueing the bpb chain"
    ( cd "$REPO/src" && bash evals/scripts/launch_bpb.sh ) >>"$LOG" 2>&1 \
        || FAILED+=("launch_bpb.sh")

    # Exactly one watcher, always running current code. A watcher holds its
    # module constants from import time, so editing a file cannot reach it: on
    # 2026-09-21 one started at 10:30 kept writing conversions to the
    # pre-move capstor path for three hours after the fix landed, splitting
    # msnr-hf-models across two trees. Restarting nightly is what guarantees
    # the running watcher matches the code on disk.
    # -u $USER, or pgrep matches a collaborator's watcher too. And the pattern
    # insists on a python process: a plain `-f "auto_evals_cscs.py --watch"`
    # also matches any SHELL whose command line happens to contain that string
    # — it matched this session's own `bash -c` while being tested, and killing
    # that is not what "restart the watcher" means.
    local pat=${WATCHER_PAT:-'python[0-9.]* +[^ ]*auto_evals_cscs\.py +--watch'}
    local olds
    olds=$(pgrep -u "$USER" -f "$pat" 2>/dev/null || true)
    if [ -n "$olds" ]; then
        say "stopping $(wc -w <<<"$olds") running watcher(s): $(tr '\n' ' ' <<<"$olds")"
        # shellcheck disable=SC2086
        kill $olds 2>/dev/null
        sleep 5
        olds=$(pgrep -u "$USER" -f "$pat" 2>/dev/null || true)
        if [ -n "$olds" ]; then
            # Do NOT start another: "restart" that leaves the old one running
            # is how you get two watchers, which is the thing this prevents.
            # Not escalating to SIGKILL either — a watcher mid-submit is the
            # user's to kill, and tomorrow's pass will try again.
            say "ERROR: still alive after SIGTERM: $(tr '\n' ' ' <<<"$olds") —"
            say "       refusing to start a second watcher; kill it by hand"
            FAILED+=("watcher would not stop"); return
        fi
    else
        say "no watcher was running"
    fi

    say "starting a fresh watcher (SBATCH_PARTITION=preemptable, --watch 1800)"
    # NOT `( cd X && setsid ... & )`: that backgrounds the whole `cd && setsid`
    # chain, so $! is a forked copy of THIS shell which then sits waiting on
    # the watcher and holding our stdout pipe open — the pid recorded is that
    # shell's, and a piped caller never sees EOF. Background setsid alone, and
    # disown it so nothing waits. setsid/nohup/env are all exec-chains, so $!
    # is the python process itself.
    local wlog=$LOG_DIR/$STAMP.watcher.log wpid
    cd "$REPO/src" || { FAILED+=("cannot cd to $REPO/src"); return; }
    setsid nohup env SBATCH_PARTITION=preemptable \
        "$PY" pretrain/auto_evals_cscs.py --watch 1800 >>"$wlog" 2>&1 &
    wpid=$!
    disown "$wpid" 2>/dev/null
    cd "$REPO" || true
    echo "$wpid" > "$LOG_DIR/watcher.pid"
    sleep 3
    if kill -0 "$wpid" 2>/dev/null; then
        NOTES+=("watcher pid $wpid, log $wlog")
        say "watcher up (pid $wpid): $(tr '\0' ' ' < /proc/$wpid/cmdline 2>/dev/null | cut -c1-70)"
    else
        say "ERROR: the watcher died within 3s — see $wlog"
        FAILED+=("watcher did not start")
    fi
}

do_ladder() {
    say "publishing the ladder report"
    local pub_err=$LOG_DIR/$STAMP.publish.err
    # stderr straight to its file (a `>(tee ...)` substitution is asynchronous: the grep below
    # could read it before it is flushed), then appended to the log once the command is done
    ( cd "$REPO/src" && $PY pretrain/ladder_report.py --plot --publish --push-hf --push-git ) \
        > >(tee -a "$LOG") 2> "$pub_err"
    local rc=$?
    cat "$pub_err" >> "$LOG"
    # A crash before publish() wrote its CSV leaves yesterday's file at $fresh,
    # which the cmp below would install and verify as "today's": stop here.
    (( rc == 0 )) || { FAILED+=("ladder_report.py (exit $rc)"); finish 1; }

    local git_pushed=1 hf_pushed=1
    grep -q "\[publish\] git push skipped"  "$pub_err" 2>/dev/null && git_pushed=0
    grep -q "\[publish\] HF upload skipped" "$pub_err" 2>/dev/null && hf_pushed=0
    (( hf_pushed ))  || { say "WARN: the Hub upload did not land (see $pub_err)"; FAILED+=("push-hf"); }
    (( git_pushed )) || say "WARN: the git push did not land — installing the local report instead"

    local fresh=$REPO/src/pretrain/ladder_report.csv
    local dests=("$REPO/src/signal-and-noise/data/ladder-report" "$REPO/data/ladder-report")
    [ -f "$fresh" ] || { say "ERROR: no report at $fresh — nothing to analyse"
                         FAILED+=("no ladder_report.csv"); finish 1; }

    if (( git_pushed )); then
        say "installing the report from origin/data/ladder-report"
        if $GIT fetch origin data/ladder-report >>"$LOG" 2>&1; then
            for d in "${dests[@]}"; do
                mkdir -p "$d" && $GIT archive origin/data/ladder-report | tar -x -C "$d"
            done
            say "installed $($GIT log -1 --format='%h %s' origin/data/ladder-report)"
        else
            say "WARN: fetch failed after a push that looked fine — using the local copy"
            git_pushed=0
        fi
    fi
    if (( ! git_pushed )); then
        say "installing the report from disk (no push, so the branch is stale)"
        for d in "${dests[@]}"; do
            mkdir -p "$d"
            for n in ladder_report.csv ladder_report_curve.csv ladder_report.md; do
                [ -f "$REPO/src/pretrain/$n" ] && cp -p "$REPO/src/pretrain/$n" "$d/$n"
            done
        done
    fi

    # Whatever route the report took, what the analysis is about to read must
    # be what was just generated. A stale branch is the one failure that
    # produces a complete, plausible, wrong run.
    local installed=${dests[0]}/ladder_report.csv
    if ! cmp -s "$fresh" "$installed"; then
        say "ERROR: $installed differs from the report just generated — refusing"
        say "       to spend hours on numbers that are not today's."
        FAILED+=("stale ladder report"); finish 1
    fi
    say "report verified: $(wc -l < "$installed") rows, identical to today's run"

    [ -n "${NIGHTLY_NO_SUBMIT:-}" ] && { say "NIGHTLY_NO_SUBMIT set — not submitting"; finish 0; }

    # Already in flight? Then a previous night is still working, or a node beat
    # us to the lock and finished: do not add a second. The EXIT STATUS, not
    # just the output — a controller hiccup prints nothing, which would read as
    # an empty queue and submit a duplicate. Same rule as
    # launch_trainings.active_slurm_jobs ("an unreachable controller must not
    # read as an empty queue") and preempt_drain.sh's squeue guard.
    local inflight
    if ! inflight=$($SQUEUE --me -h -n ladder-refresh,ladder-curves -o "%i %j %T %r" 2>>"$LOG"); then
        say "ERROR: squeue failed — not submitting without knowing what is already queued"
        FAILED+=("squeue failed"); finish 1
    fi
    # A job Slurm will never start is not in flight. An afterok child of a
    # failed refresh sat PENDING (DependencyNeverSatisfied) from 2026-10-04 and
    # made every later night skip its analysis (job 3574596).
    inflight=$(grep -v DependencyNeverSatisfied <<<"$inflight" || true)
    if [ -n "$inflight" ]; then
        say "a refresh is already queued/running — not submitting another:"
        say "  $(tr '\n' ';' <<<"$inflight")"
        NOTES+=("skipped submit, already in flight")
        finish 0
    fi

    # No --wait: the deck is not built here (node is not installed on the login
    # nodes), so there is nothing to do afterwards and the unit should not hold
    # for hours. The jobids go in the status file instead.
    # A manual pause (the report still publishes): `touch $LOG_DIR/PAUSE_REFRESH`
    # while the analysis code is being changed in the tree the job would run.
    if [ -e "$LOG_DIR/PAUSE_REFRESH" ]; then
        say "refresh paused ($LOG_DIR/PAUSE_REFRESH exists) — not submitting"
        NOTES+=("refresh paused"); finish 0
    fi
    say "submitting the analysis (FORCE=1, --no-fetch --no-deck)"
    local main_id
    main_id=$($SBATCH --parsable --account=infra01 --partition=normal --nodes=1 \
        --time=06:00:00 --job-name=ladder-refresh \
        --output="$LOG_DIR/$STAMP.refresh-%j.log" \
        --wrap="source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr \
                && cd $REPO \
                && FORCE=1 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 \
                   bash scripts/refresh_analysis.sh --no-fetch --no-deck" 2>>"$LOG")
    if [ -z "$main_id" ]; then
        say "ERROR: the analysis did not submit"; FAILED+=("sbatch ladder-refresh"); finish 1
    fi
    say "analysis job $main_id"
    NOTES+=("analysis job $main_id")

    # The ~140 acc-vs-FLOPs grids are about an hour and nothing downstream
    # reads them, so they go in their own job rather than lengthening the one
    # above. run_apertus.py without --no-grids is exactly what CURVES=1 does
    # inside run_all_predictivity.sh. After, not concurrent: the grids read the
    # above-random mask the main pipeline writes. afterany, not afterok: the
    # refresh exits non-zero when ANY one of its steps fails, so under afterok
    # the curves never ran (every refresh from 2026-09-28 to 10-04 ended FAILED)
    # and the never-satisfiable child then blocked the next night's submit.
    local curves_id
    curves_id=$($SBATCH --parsable --account=infra01 --partition=normal --nodes=1 \
        --time=04:00:00 --job-name=ladder-curves --dependency="afterany:$main_id" \
        --output="$LOG_DIR/$STAMP.curves-%j.log" \
        --wrap="source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr \
                && cd $REPO/src/signal-and-noise \
                && HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 \
                   python3 analysis/rq00_gate_and_curves/run_apertus.py --pool predictivity" \
        2>>"$LOG")
    if [ -n "$curves_id" ]; then
        say "curves job $curves_id (afterany:$main_id)"
        NOTES+=("curves job $curves_id, afterany:$main_id")
    else
        say "WARN: the curves job did not submit"; FAILED+=("sbatch ladder-curves")
    fi
}

# --- run ---------------------------------------------------------------------
if (( DRY )); then
    echo "mode:        $MODE"
    echo "lock:        $LOCK"
    echo "log:         $LOG"
    echo "status:      $STATUS"
    if [ "$MODE" = evals ]; then
        echo "would run    bash evals/scripts/launch_bpb.sh"
        echo "would kill   any auto_evals_cscs.py --watch of $USER, then start one fresh"
    else
        echo "would run    $PY pretrain/ladder_report.py --plot --publish --push-hf --push-git"
        echo "then         install the report (fetch if the push landed, local copy if not) + cmp"
        echo "then sbatch  ladder-refresh  FORCE=1 refresh_analysis.sh --no-fetch --no-deck"
        echo "then sbatch  ladder-curves   run_apertus.py --pool predictivity (afterany)"
    fi
    exit 0
fi

take_lock || exit 0
say "=== nightly $MODE on $(hostname) ==="
"do_$MODE"
finish 0
