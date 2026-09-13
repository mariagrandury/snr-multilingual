#!/bin/bash
# Launch ONE Slurm job per data mixture so they all build in parallel (fastest
# path to having every mixture ready). Each job is self-chaining and idempotent
# (submit_build_one.sh). Run from the login node.
#
#   ./launch_builds.sh --dry-run   # print the sbatch commands, submit nothing
#   ./launch_builds.sh             # submit
set -euo pipefail
DIR=/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain/data
ONE=$DIR/submit_build_one.sh
OUT=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data  # persistent capstor store
LOGS=/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data/logs   # logs live with the data on capstor (they carry the per-language plan); matches submit_build_one.sh's #SBATCH --output
DRY=${1:-}

# ---- the 92B rebuilds -------------------------------------------------------
# The 1.7B rung gained L15 and L50 on 2026-09-10, so those two scheme-A builds
# are now sized 92B where the finished ones on capstor are 52B. They cannot be
# rebuilt in place: submit_build_one.sh's idempotency guard keys on "<prefix>.idx
# present" and would skip them forever, and overwriting a dataset that trained
# (and training) cells read is a hard no. So the cells listed here build into
# REBUILD_ROOT instead — a parallel data root of the same shape (scheme subdirs
# + the shared english/validation symlinks) — and stage to their own iopsstor
# tree, so nothing under the 52B copies is touched. launch_trainings.py reads
# that stage (its CSCS_REBUILD_DATA_DIR) only for cells the 52B copy is too
# small for — the 1.7B at these settings — and every other rung keeps the 52B
# copy its peers trained on, so nothing is ever swapped in.
# B:15 is here for the same reason: schemeB's L15 is also a 52B build against a
# 92B target, and the B ladder is meant to reach 1.7B at L15 too (decided
# 2026-09-10). B's L8 and L30 are already 92B and need nothing.
REBUILD=(A:15 A:50 B:15)
REBUILD_ROOT=$OUT/rebuild-92B
REBUILD_DST=/iopsstor/scratch/cscs/mariagrandury/data-92B   # = launch_trainings.CSCS_REBUILD_DATA_DIR

mkdir -p "$OUT" "$LOGS"

# The scheme registry in ../launch_trainings.py owns which data schemes exist,
# where each one builds and which settings it defines — read it here rather than
# restating it, so a scheme added there gets its build jobs for free. One line
# per (scheme, setting) FineWeb-2 build; L=1 is 100% English and has none.
PLAN=$(python3.11 - <<'PY'
import sys
sys.path.insert(0, "/iopsstor/scratch/cscs/mariagrandury/Projects/snr-multilingual/src/pretrain")
from launch_trainings import DATA_SCHEMES
for name, v in DATA_SCHEMES.items():
    for L in sorted(v["langs"]):
        if L != 1:
            print(f"{name}:{v['subdir']}:{L}")
PY
)

# Every scheme is its own --data_dir (downstream keys off the dir; the mixture
# name is a fixed fineweb_L{L}). Symlink the shared english build + validation
# manifest into it so each scheme dir is a complete data_dir; english may
# dangle until built (a fineweb build needs only the manifest, which exists).
variant_dir() { # <root> <subdir> -> the scheme's data_dir, made complete
  local dir=$1${2:+/$2}
  if [ "$DRY" != --dry-run ]; then   # a dry run prints, it does not touch the store
    mkdir -p "$dir" || return 1
    if [ "$dir" != "$OUT" ]; then    # the master root IS where those two live
      for f in english_dclm.bin english_dclm.idx validation.manifest.json; do
        ln -sfn "$OUT/$f" "$dir/$f" || return 1
      done
    fi
  fi
  echo "$dir"
}

# english continues in its own job. If the old monolithic build-data-mix job is
# still running english, gate the english job to start after it ends so two jobs
# never write english_dclm at once; it resumes from the shared checkpoint.
MAIN=$(squeue -u "$USER" -h -n build-data-mix -t RUNNING -o %i 2>/dev/null | head -1 || true)
EN_DEP=(); [ -n "$MAIN" ] && EN_DEP=(--dependency=afterany:"$MAIN")

submit() { # name exportvars [--dependency=afterany:ID]
  local name=$1 vars=$2 dep=singleton a; shift 2
  # Always a singleton: a job of the same name still queued or running (an
  # earlier run of this script, a self-chained successor) would otherwise write
  # the same prefix concurrently — both resume from one checkpoint and append to
  # one .bin. An afterany gate, when given, is combined with it.
  for a in "$@"; do if [[ $a == --dependency=* ]]; then dep="${a#--dependency=},singleton"; fi; done
  local cmd=(sbatch --job-name="$name" --dependency="$dep" --export=ALL,"$vars" "$ONE")
  if [ "$DRY" = --dry-run ]; then echo "DRY: ${cmd[*]}"; else echo "  $("${cmd[@]}")  [$name]"; fi
}

# L2 (Russian, sized for its largest run, the 1.7B) goes through the same
# generic path as every other setting. If an old `build-l2` chain from the
# retired submit_build_l2.sh is still running, gate the new L2 job after it —
# both write the same fineweb_L2 prefix and must never run concurrently (the
# idempotency guard then no-ops whichever starts after the build finishes).
OLD_L2=$(squeue -u "$USER" -h -n build-l2 -o %i 2>/dev/null | head -1 || true)
L2_DEP=(); [ -n "$OLD_L2" ] && L2_DEP=(--dependency=afterany:"$OLD_L2")

echo "english gated after: ${MAIN:-<none running>}; L2 gated after: ${OLD_L2:-<none running>}"
submit build-en "BUILD_SCHEME=A,BUILD_STAGE=english,BUILD_OUT=$OUT" "${EN_DEP[@]}"

while IFS=: read -r scheme subdir L; do
  name="build-${scheme,,}-L$L"
  root=$OUT extra=
  # A rebuild keeps a name of its own: same-name jobs are a singleton chain
  # (submit_build_one.sh), and the attempt counter globs the log names, so
  # reusing the finished 52B build's name would chain onto its history.
  if [[ " ${REBUILD[*]} " == *" $scheme:$L "* ]]; then
    root=$REBUILD_ROOT; name="$name-92b"; extra=",BUILD_DST=$REBUILD_DST"
  fi
  dep=()
  if [ "$scheme:$L" = A:2 ]; then dep=("${L2_DEP[@]}"); fi
  # On its own line: a failure inside $(...) used as an argument is lost.
  vdir=$(variant_dir "$root" "$subdir") || { echo "cannot prepare $root/$subdir" >&2; exit 1; }
  submit "$name" "BUILD_SCHEME=$scheme,BUILD_STAGE=fineweb,BUILD_SETTING=$L,BUILD_OUT=$vdir$extra" "${dep[@]}"
done <<< "$PLAN"
