#!/bin/bash
#SBATCH --account=infra01
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8000
#SBATCH --output=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data/logs/%x-%j.out
#SBATCH --no-requeue
# Run the launcher's dry-run once a data build has fully finished and staged.
# A build (submit_build_one.sh) is a chain of 12h singleton segments, so a plain
# afterok on the segment you can see fires 12h in, mid-build. Instead: while any
# job of that name is still queued or running, requeue after the current one
# and exit; when none is left the build's last segment has run its staging.
#
#   sbatch --job-name=dryrun-3B --dependency=afterany:<segment id> \
#          --export=ALL,WAIT_FOR=build-a-L8-165b,ARGS="--size 3B" dryrun_after_build.sbatch
set -euo pipefail
: "${WAIT_FOR:?build job name}"; : "${ARGS:?launch_trainings.py args}"
SCRIPT=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/dryrun_after_build.sbatch  # not $0: Slurm runs a spool copy
cur=$(squeue -u "$USER" -h -n "$WAIT_FOR" -o %i | head -1)
if [ -n "$cur" ]; then
  echo "[$(date)] $WAIT_FOR still has job $cur — requeuing after it"
  sbatch --dependency=afterany:"$cur" --job-name="$SLURM_JOB_NAME" --export=ALL "$SCRIPT"
  exit 0
fi
echo "[$(date)] $WAIT_FOR has no job left — dry-run: launch_trainings.py cscs $ARGS --dry-run"
cd "$(dirname "$SCRIPT")"
PY=$(command -v python3.11 || true)
[ -n "$PY" ] || { source ~/.bashrc; conda activate snr; PY=python; }
$PY launch_trainings.py cscs $ARGS --dry-run
