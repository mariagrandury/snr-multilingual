set -euo pipefail
SLURM_JOB_NAME=build-x CHAIN_PART=preemptable CHAIN_TIME=1:00 SCRIPT=/x.sh
sleep() { :; }
  chain=(sbatch --dependency=singleton --job-name="$SLURM_JOB_NAME"
         --partition="$CHAIN_PART" --time="$CHAIN_TIME"
         ${BUILD_EXCLUSIVE:---exclusive} --export=ALL "$SCRIPT")
  "${chain[@]}" || { sleep 30; "${chain[@]}" || echo \
    "WARNING: sbatch refused the successor twice — this is the LAST attempt of" \
    "the chain; the build below still runs, then relaunch with ./launch_builds.sh" >&2; }
echo "BUILD RUNS"
