#!/bin/bash
# nightly_ladder.sh — publish the ladder report, then rebuild every derived
# artefact, unattended. Armed by a systemd --user timer at 06:00 Europe/Zurich
# (see the bottom of this header); nothing about it depends on a live session.
#
# Why this is not one scrontab entry. Slurm's cron is enabled here, but it can
# only submit BATCH jobs, and the work straddles a hard boundary:
#
#   needs the internet   ladder_report.py --publish --push-hf --push-git,
#                        `git fetch`, and the slidev build
#   needs a Slurm job    refresh_analysis.sh, ~2 h of CPU that
#                        src/pretrain/CLAUDE.md forbids on a login node
#
# CSCS compute nodes have no internet and no HTTP proxy (verified 2026-09-23:
# no http_proxy/https_proxy in a running job's environment, none here either),
# and the `xfer` partition — the one place that might have had it — has had
# both its nodes DRAINED since 2026-09-22 ("workqueue lockup bug detected")
# with jobs queued behind it for weeks. So the network half runs here, on the
# login node, and only the heavy half is submitted. That is recipe 2b in
# ../CLAUDE.md, automated.
#
# Credentials, both checked before this was written:
#   - origin is git@github.com:..., i.e. SSH, and a timer has no agent — but
#     ~/.ssh/id_ed25519 has no passphrase, so the push works unattended.
#   - HF_TOKEN is exported from ~/.bashrc, which does NOT early-return for
#     non-interactive shells, so sourcing it below is enough.
#
# THE PUSH-FAILURE FALLBACK, which is the whole reason this is a script and
# not three lines in a crontab. ladder_report.publish() wraps both pushes in
# try/except and only prints "[publish] ... skipped" — it exits 0 either way.
# The report itself is never lost: publish() copies it to capstor BEFORE it
# attempts any push. But refresh_analysis.sh installs the report by fetching
# the orphan branch, so after a failed push the fetch SUCCEEDS and quietly
# hands the analysis yesterday's numbers. Nothing downstream would notice.
# This script therefore reads publish()'s stderr, and on any push failure
# installs the freshly generated CSVs from disk and passes --no-fetch. Either
# way it then proves, with cmp, that the installed report is byte-identical to
# the one just generated, and refuses to spend two hours if it is not.
#
# Usage:
#   bash scripts/nightly_ladder.sh              # the whole thing
#   bash scripts/nightly_ladder.sh --dry-run    # print the plan, touch nothing
#   NIGHTLY_NO_SUBMIT=1 bash scripts/nightly_ladder.sh   # stop before sbatch
#
# Install the timer (once, on the login node you want it on — it is pinned to
# that node, `clariden-ln003` when this was written):
#   loginctl enable-linger                       # or it dies at logout
#   systemctl --user enable --now ladder-nightly.timer
#   systemctl --user list-timers ladder-nightly  # confirm the next 06:00
# If enable-linger is refused, scripts/nightly_ladder_setsid.sh is the
# agent-free fallback; it survives logout but not a login-node reboot.

set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

# Logs live OUTSIDE the repo: this runs nightly and must not churn git status.
LOG_DIR=${NIGHTLY_LOG_DIR:-/iopsstor/scratch/cscs/$USER/logs/nightly-ladder}
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y-%m-%d)
LOG=$LOG_DIR/$STAMP.log
STATUS=$LOG_DIR/last-run.txt

# A systemd user service starts with almost no environment: ~/.bashrc is what
# carries HF_TOKEN, and conda is what carries the git-lfs filter and matplotlib.
source ~/.bashrc >/dev/null 2>&1 || true
source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr
# Overridable so the publish/install branches can be exercised against a
# fixture with no remote, the way refresh_analysis.sh already takes PY.
PY=${PY:-python3.11}
GIT=${GIT:-git}

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
FAILED=()

finish() {
    local rc=$1
    if [ ${#FAILED[@]} -gt 0 ]; then
        say "FAILED: ${FAILED[*]}"
        printf '%s  FAILED: %s\n  log: %s\n' "$(date '+%F %T')" "${FAILED[*]}" "$LOG" > "$STATUS"
        exit 1
    fi
    say "all refreshed"
    printf '%s  OK\n  log: %s\n' "$(date '+%F %T')" "$LOG" > "$STATUS"
    exit "$rc"
}

if (( DRY )); then
    echo "would log to      $LOG"
    echo "would run         $PY src/pretrain/ladder_report.py --plot --publish --push-hf --push-git"
    echo "then install the report (fetch if the push landed, local copy if not)"
    echo "then sbatch       FORCE=1 refresh_analysis.sh --no-fetch --no-deck"
    echo "then build the deck here, where there is network"
    exit 0
fi

say "=== nightly ladder refresh on $(hostname) ==="

# ---------------------------------------------------------------- 1. publish
# Stderr is captured, not just teed: the push outcome is only visible there.
say "publishing the ladder report"
PUB_ERR=$LOG_DIR/$STAMP.publish.err
( cd "$REPO/src" && $PY pretrain/ladder_report.py --plot --publish --push-hf --push-git ) \
    > >(tee -a "$LOG") 2> >(tee "$PUB_ERR" >>"$LOG")
LR_RC=$?
(( LR_RC == 0 )) || FAILED+=("ladder_report.py (exit $LR_RC)")

GIT_PUSHED=1; HF_PUSHED=1
grep -q "\[publish\] git push skipped"  "$PUB_ERR" 2>/dev/null && GIT_PUSHED=0
grep -q "\[publish\] HF upload skipped" "$PUB_ERR" 2>/dev/null && HF_PUSHED=0
(( HF_PUSHED ))  || { say "WARN: the Hub upload did not land (see $PUB_ERR)"; FAILED+=("push-hf"); }
(( GIT_PUSHED )) || say "WARN: the git push did not land — installing the local report instead"

# ------------------------------------------------- 2. install the report
# The freshly generated copy, which publish() wrote before it pushed anything.
FRESH=$REPO/src/pretrain/ladder_report.csv
DESTS=("$REPO/src/signal-and-noise/data/ladder-report" "$REPO/data/ladder-report")
if [ ! -f "$FRESH" ]; then
    say "ERROR: no report at $FRESH — nothing to analyse"
    FAILED+=("no ladder_report.csv"); finish 1
fi

if (( GIT_PUSHED )); then
    say "installing the report from origin/data/ladder-report"
    if $GIT fetch origin data/ladder-report >>"$LOG" 2>&1; then
        for d in "${DESTS[@]}"; do
            mkdir -p "$d" && $GIT archive origin/data/ladder-report | tar -x -C "$d"
        done
        say "installed $($GIT log -1 --format='%h %s' origin/data/ladder-report)"
    else
        say "WARN: fetch failed after a push that looked fine — falling back to the local copy"
        GIT_PUSHED=0
    fi
fi
if (( ! GIT_PUSHED )); then
    say "installing the report from disk (no push, so the branch is stale)"
    for d in "${DESTS[@]}"; do
        mkdir -p "$d"
        for n in ladder_report.csv ladder_report_curve.csv ladder_report.md; do
            [ -f "$REPO/src/pretrain/$n" ] && cp -p "$REPO/src/pretrain/$n" "$d/$n"
        done
    done
fi

# The invariant that makes the fallback safe: whatever route the report took,
# what the analysis is about to read must be what was just generated. A stale
# branch is the one failure that produces a complete, plausible, wrong run.
INSTALLED=${DESTS[0]}/ladder_report.csv
if ! cmp -s "$FRESH" "$INSTALLED"; then
    say "ERROR: $INSTALLED differs from the report just generated — refusing to"
    say "       spend two hours on numbers that are not today's."
    FAILED+=("stale ladder report"); finish 1
fi
say "report verified: $(wc -l < "$INSTALLED") rows, identical to today's run"

[ -n "${NIGHTLY_NO_SUBMIT:-}" ] && { say "NIGHTLY_NO_SUBMIT set — stopping before sbatch"; finish 0; }

# ------------------------------------------------------- 3. the heavy half
# --wait blocks this process (one sleeping pid, no CPU) so the deck below runs
# only after the tables it renders exist. --no-fetch because step 2 already
# installed the report and a compute node could not fetch anyway; --no-deck
# because slidev needs the network this node has and that one does not.
say "submitting the analysis (FORCE=1, --no-fetch --no-deck)"
sbatch --wait --account=infra01 --partition=normal --nodes=1 --time=06:00:00 \
       --job-name=ladder-refresh \
       --output="$LOG_DIR/$STAMP.analysis-%j.log" \
       --wrap="source ~/miniconda3/etc/profile.d/conda.sh && conda activate snr \
               && cd $REPO \
               && FORCE=1 HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=4 \
                  bash scripts/refresh_analysis.sh --no-fetch --no-deck" \
       >>"$LOG" 2>&1
SB_RC=$?
(( SB_RC == 0 )) || FAILED+=("refresh_analysis.sh (sbatch --wait exit $SB_RC)")
say "analysis job finished (exit $SB_RC)"

# ------------------------------------------------------------- 4. the deck
# Here, not in the job: `npx slidev build` reaches the network.
if command -v npx >/dev/null 2>&1; then
    say "building the deck"
    ( cd "$REPO/documents" && PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 \
        npx slidev build --out /tmp/slidev-nightly ) >>"$LOG" 2>&1 \
        || FAILED+=("slidev build")
else
    say "npx not found — deck not built"
fi

finish 0
