#!/bin/bash
# mirror_eval_logs.sbatch — copy the eval results tree to the capstor master.
#
# WHY: eval_logs lives on /iopsstor, which is purged ~30 days. It is 234 GB
# (2026-09-01) and irreplaceable — re-running it would cost ~940 node-hours.
# NOTHING SCHEDULES THIS: it is a manual rsync, so anything evaluated since the
# last run exists only on scratch. capstor is the
# durable store (same split as the data mixtures: capstor master, iopsstor
# working copy — see ../../pretrain/CLAUDE.md #8).
#
# COPY, NOT MOVE, deliberately. Moving would break two things at once:
#   * _eval_status.completed_tasks scans LOGS_ROOT to decide what is already
#     evaluated. Point that at an empty iopsstor tree and every watcher pass
#     re-submits every checkpoint — thousands of jobs.
#   * eval jobs would then WRITE results to capstor. A capstor blip (errno
#     108, seen twice this week) would fail the task being written, or the
#     job's final merge and W&B push.
# Keeping iopsstor as the working copy costs one rsync and no new failure mode.
#
# Idempotent and resumable: rsync skips files already there with the same size
# and mtime, so re-running after a wall-clock kill continues where it stopped.
#
#   sbatch scripts/mirror_eval_logs.sbatch          # the whole tree
#   DRY=1 bash scripts/mirror_eval_logs.sbatch      # show what would move, locally
#
#SBATCH --account=infra01
#SBATCH --job-name=mirror-eval-logs
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
# Logs go to capstor beside the data they describe: Slurm opens them before the
# script runs, so a swept scratch dir would mean no log at all.
#SBATCH --output=/capstor/store/cscs/swissai/infra01/msnr-eval-logs/logs/%x-%j.out
#SBATCH --error=/capstor/store/cscs/swissai/infra01/msnr-eval-logs/logs/%x-%j.out
set -uo pipefail

LOGS=${LOGS:-/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs}
CAP=${CAP:-/capstor/store/cscs/swissai/infra01}
DRY=${DRY:-0}

# The whole durable record of the sweep, not just the evals. The two training
# trees are ~2.6 GB against the eval tree's 336 GB, so they are effectively
# free — and until 2026-09-01 NOTHING about training reached capstor at all.
#
# The per-run glob is deliberately `*/logging`, not `lm-*/logging`: diagnostic
# cells are named `diag-*` precisely so the grid tooling ignores them, but
# they are the evidence behind plan/90M-rung-anomaly.md and must be kept. This
# is the one place the diag- prefix must NOT filter anything.
#
# checkpoints/, triggers/ and debug/ are excluded: checkpoints are terabytes
# and their durable form is already <CAP>/msnr-hf-models.
PAIRS=(
    "$LOGS/eval_logs|$CAP/msnr-eval-logs"
    "$LOGS/slurm/training|$CAP/msnr-train-logs"
    "$LOGS/Meg-Runs/msnr|$CAP/msnr-run-logging"
)

# --partial: a 2 MB samples file interrupted mid-write resumes rather than
#   restarting. --inplace is NOT used: a torn file would look complete.
# --no-perms/--no-group: capstor enforces its own ACLs; preserving the source
#   modes fails noisily on every file without changing anything.
ARGS=(-a --partial --no-perms --no-group --human-readable --stats
      --exclude=checkpoints/ --exclude=triggers/ --exclude=debug/)
(( DRY )) && ARGS+=(--dry-run --itemize-changes)

mkdir -p "$CAP/msnr-eval-logs/logs"
rc=0
for pair in "${PAIRS[@]}"; do
    src=${pair%%|*}; dst=${pair##*|}
    [ -d "$src" ] || { echo "[$(date)] SKIP $src (absent)"; continue; }
    mkdir -p "$dst"
    echo "[$(date)] mirroring $src -> $dst"
    du -sh "$src" 2>/dev/null
    rsync "${ARGS[@]}" "$src/" "$dst/"
    prc=$?
    # 23/24 = some files vanished or were unreadable mid-run, which is normal
    # while evals are writing; the next run picks them up. Keep going: one
    # busy tree must not stop the other two from being mirrored.
    if (( prc == 0 )); then
        echo "[$(date)] done $dst"; du -sh "$dst" 2>/dev/null
    else
        echo "[$(date)] rsync exited $prc on $src (23/24 = files changed" \
             "under us; re-run)" >&2
        rc=$prc
    fi
done
exit $rc
