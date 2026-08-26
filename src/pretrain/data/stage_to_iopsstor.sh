#!/bin/bash
# stage_to_iopsstor.sh — copy finished mixtures from the capstor master to the
# iopsstor training stage.
#
# capstor is the durable master (iopsstor is swept ~30 days) and the builds
# write there, but training MUST read from iopsstor: Megatron memmaps the .bin
# and reads it shuffled, i.e. randomly, and capstor is ~28x slower per random
# read (see ../CLAUDE.md #8). A mixture that exists only on capstor makes every
# cell at that setting die on
#   AssertionError: One or both of the .idx and .bin files cannot be found
#
# Copies, never moves — the capstor copy is the one that survives a sweep.
#
# Idempotent: a mixture already staged at the right size is skipped, so this is
# safe to re-run and safe to call at the end of every build.
#
#   bash stage_to_iopsstor.sh                    # every completed mixture
#   bash stage_to_iopsstor.sh fineweb_L50        # one, scheme A
#   bash stage_to_iopsstor.sh schemeB/fineweb_L8 # one, scheme B
#   sbatch --time=04:00:00 --nodes=1 stage_to_iopsstor.sh   # ~2 TiB: not on the login node
#
#SBATCH --job-name=stage-data
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --output=/iopsstor/scratch/cscs/%u/data/logs/%x-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/%u/data/logs/%x-%j.err
set -uo pipefail

SRC=${SRC:-/capstor/store/cscs/swissai/infra01/multilingual_data_mixtures/predictivity-data}
DST=${DST:-/iopsstor/scratch/cscs/$USER/data}

stage_one() { # <relative prefix, e.g. fineweb_L50 or schemeB/fineweb_L8>
  local rel="$1" ok=1
  # A build in flight still has its checkpoint (create_data_mixture removes it
  # only on success) — staging a partial mixture would train on truncated data.
  if [ -f "$SRC/$rel.checkpoint.json" ]; then
    echo "  skip $rel — build still in progress"
    return 0
  fi
  mkdir -p "$(dirname "$DST/$rel")"
  for ext in bin idx; do
    local s="$SRC/$rel.$ext" d="$DST/$rel.$ext"
    [ -f "$s" ] || { echo "  skip $rel — no $ext on capstor"; return 0; }
    if [ -f "$d" ] && [ "$(stat -c%s "$s")" = "$(stat -c%s "$d")" ]; then
      continue                       # already staged, same size
    fi
    echo "  copy $rel.$ext ($(numfmt --to=iec "$(stat -c%s "$s")"))"
    # Copy to a temp name and rename: a killed copy must never leave a
    # short .bin behind, because Megatron would mmap it without complaint.
    if cp -f "$s" "$d.tmp" && mv -f "$d.tmp" "$d"; then :; else
      echo "  FAILED $rel.$ext"; rm -f "$d.tmp"; ok=0
    fi
  done
  [ "$ok" = 1 ] && echo "  staged $rel"
}

mkdir -p "$DST" "$DST/schemeB" "/iopsstor/scratch/cscs/$USER/data/logs"
echo "[$(date)] staging $SRC -> $DST"

if [ $# -gt 0 ]; then
  for rel in "$@"; do stage_one "$rel"; done
else
  # Every completed mixture: an .idx exists and no checkpoint is left behind.
  for idx in "$SRC"/*.idx "$SRC"/schemeB/*.idx; do
    [ -e "$idx" ] || continue
    case "$idx" in *validation*) continue;; esac
    rel="${idx#$SRC/}"; rel="${rel%.idx}"
    stage_one "$rel"
  done
fi
echo "[$(date)] done"
