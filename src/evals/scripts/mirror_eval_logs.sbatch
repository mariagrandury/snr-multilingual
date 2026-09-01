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
#   * eval jobs would then WRITE results to capstor at the end of a run. A
#     capstor blip (errno 108, seen twice this week) would lose a finished
#     45-minute job, because BATCH_TASKS=1 writes nothing until the very end.
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

SRC=${SRC:-/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs}
DST=${DST:-/capstor/store/cscs/swissai/infra01/msnr-eval-logs}
DRY=${DRY:-0}

mkdir -p "$DST/logs"
echo "[$(date)] mirroring $SRC -> $DST"
du -sh "$SRC" 2>/dev/null

# --partial: a 2 MB samples file interrupted mid-write resumes rather than
#   restarting. --inplace is NOT used: a torn file would look complete.
# --no-perms/--no-group: capstor enforces its own ACLs; preserving the source
#   modes fails noisily on every file without changing anything.
ARGS=(-a --partial --no-perms --no-group --human-readable --stats)
(( DRY )) && ARGS+=(--dry-run --itemize-changes)

rsync "${ARGS[@]}" "$SRC/" "$DST/"
rc=$?

if (( rc == 0 )); then
    echo "[$(date)] mirror complete"
    du -sh "$DST" 2>/dev/null
else
    # 23/24 = some files vanished or were unreadable mid-run, which is normal
    # while evals are writing; the next run picks them up.
    echo "[$(date)] rsync exited $rc (23/24 = files changed under us; re-run)" >&2
fi
exit $rc
