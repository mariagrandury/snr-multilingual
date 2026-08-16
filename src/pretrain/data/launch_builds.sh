#!/bin/bash
# Launch ONE Slurm job per data mixture so they all build in parallel (fastest
# path to having every mixture ready). Each job is self-chaining and idempotent
# (submit_build_one.sh). Run from the login node.
#
#   ./launch_builds.sh --dry-run   # print the sbatch commands, submit nothing
#   ./launch_builds.sh             # submit
#
# NOT launched here (already running as their own per-mix jobs):
#   - L2 (Russian, 1.7B-sized) -> submit_build_l2.sh
set -euo pipefail
DIR=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data
ONE=$DIR/submit_build_one.sh
OUT=/iopsstor/scratch/cscs/mariagrandury/data
OUT_B=$OUT/schemeB
DRY=${1:-}

mkdir -p "$OUT_B" "$OUT/logs"
# Scheme B is its own --data_dir (downstream keys off dir, fixed fineweb_L{L}
# name). Symlink the shared english build + validation manifest so schemeB/ is a
# complete data_dir; english may dangle until built (the B build needs only the
# manifest, which already exists).
ln -sfn "$OUT/english_dclm.bin"         "$OUT_B/english_dclm.bin"
ln -sfn "$OUT/english_dclm.idx"         "$OUT_B/english_dclm.idx"
ln -sfn "$OUT/validation.manifest.json" "$OUT_B/validation.manifest.json"

# english continues in its own job. If the old monolithic build-data-mix job is
# still running english, gate the english job to start after it ends so two jobs
# never write english_dclm at once; it resumes from the shared checkpoint.
MAIN=$(squeue -u "$USER" -h -n build-data-mix -t RUNNING -o %i 2>/dev/null | head -1 || true)
EN_DEP=(); [ -n "$MAIN" ] && EN_DEP=(--dependency=afterany:"$MAIN")

submit() { # name exportvars [extra sbatch args...]
  local name=$1 vars=$2; shift 2
  local cmd=(sbatch --job-name="$name" "$@" --export=ALL,"$vars" "$ONE")
  if [ "$DRY" = --dry-run ]; then echo "DRY: ${cmd[*]}"; else echo "  $("${cmd[@]}")  [$name]"; fi
}

echo "english gated after: ${MAIN:-<none running>}"
submit build-en "BUILD_SCHEME=A,BUILD_STAGE=english,BUILD_OUT=$OUT" "${EN_DEP[@]}"
for L in 8 15 30 50 100; do submit "build-a-L$L" "BUILD_SCHEME=A,BUILD_STAGE=fineweb,BUILD_SETTING=$L,BUILD_OUT=$OUT"; done
for L in 8 15 30;        do submit "build-b-L$L" "BUILD_SCHEME=B,BUILD_STAGE=fineweb,BUILD_SETTING=$L,BUILD_OUT=$OUT_B"; done
