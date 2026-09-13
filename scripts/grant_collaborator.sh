#!/bin/bash
# Give a collaborator access to the shared sweep trees.
#
#   scripts/grant_collaborator.sh <user> [plan|apply|verify|revoke]
#
# plan (default) prints what would change; apply sets the ACLs; verify shows
# the result; revoke undoes it. Safe to re-run -- apply is idempotent.
#
# Everything the pipeline reads or writes is pinned to mariagrandury's tree
# (launch_pretraining_cscs.sh, evaluate.sbatch, convert-snr.sh), so a
# collaborator needs no Megatron checkout, HF cache or data copy of their own
# -- only permission on these paths. They DO still need their own repo clone
# (evaluate.sbatch writes src/evals/logs/ relative to cwd) plus their own
# HF_TOKEN and WANDB_API_KEY.
#
# Access model: named-user POSIX ACLs, additive only -- group (a139/csstaff)
# and other permissions are never touched. Writable dirs also get the sticky
# bit, because POSIX has no "write but never delete" bit: directory write IS
# permission to delete anything inside it. With +t a collaborator can create
# freely and remove only what they own; your checkpoints and results are safe.
set -euo pipefail

U=${1:-}
MODE=${2:-plan}
[[ -n $U ]] || { echo "usage: $0 <user> [plan|apply|verify|revoke]" >&2; exit 2; }
id "$U" >/dev/null 2>&1 || { echo "no such user: $U" >&2; exit 2; }

# Absolute, not derived from $0: sessions run from arbitrary cluster cwds.
S=/iopsstor/scratch/cscs/mariagrandury
M=$S/data-mix-small/Megatron-LM/logs

# Dirs they must be able to create entries in. Non-recursive on purpose: the
# default ACL makes anything they create inherit the grant, while the 300k+
# existing files underneath stay untouched -- recursing is slow on Lustre and
# would hand out write access to already-trained checkpoints and results.
WRITE=(
  "$M/Meg-Runs/msnr"                        # checkpoints (new cells)
  "$M/eval_logs"                            # eval results root
  "$M/eval_logs/mariagrandury-epflnlp"      #   entity level
  "$M/eval_logs/mariagrandury-epflnlp/msnr" #   project level -> new <cell>-iter<N> dirs
  "$M/auto_evals"                           # auto-eval watcher stdout
  "$M/conversion-plans"                     # convert-snr.sh plans
  "$M/slurm/training"                       # sbatch --output for training
  "$M/slurm/conversion"                     # sbatch --output for conversion
  "$S/datasets/cache"                       # Megatron --data-cache-path (52G of prebuilt indices)
  "$S/hf_home/datasets"                     # pre-staged eval datasets; harness writes .lock files here
)

# Shared state files the pipeline rewrites IN PLACE. A directory ACL is not
# enough: write_text() opens the existing inode 'w', so without a file-level
# grant auto_evals_cscs.py dies with PermissionError on auto_eval_errors.json
# *after* submitting its jobs. write_text truncates rather than recreating, so
# the ACL survives every rewrite and granting once is durable.
WRITE_FILES=( "$M/eval_logs"/*.json "$M/eval_logs"/*.json.bak-*
              "$S/hf_home/datasets"/*.lock )

READ=( "$S/data" )        # tokenized training data, recursive (small tree)
TRAVERSE=( "$S/hf_home" ) # reach hf_home/datasets without reading the rest

case "$MODE" in
plan)
  echo "== $U: WRITE (rwx + default rwx + sticky) =="
  for d in "${WRITE[@]}"; do
    [[ -d $d ]] && printf '  %s  %s\n' "$(stat -c '%A' "$d")" "$d" || echo "  MISSING  $d"
  done
  echo "== $U: READ (r-x, recursive) =="
  for d in "${READ[@]}"; do
    printf '  %s  %s  (%s entries)\n' "$(stat -c '%A' "$d")" "$d" "$(find "$d" | wc -l)"
  done
  ;;

apply)
  for d in "${WRITE[@]}"; do
    [[ -d $d ]] || { echo "skip (missing): $d" >&2; continue; }
    setfacl -m "u:$U:rwx" "$d"      # can create entries here
    setfacl -d -m "u:$U:rwx" "$d"   # ...and in everything created under it
    chmod +t "$d"                   # ...but can only delete what they own
    echo "write  $d"
  done
  for f in "${WRITE_FILES[@]}"; do
    [[ -f $f ]] || continue
    # Pin the mask explicitly. Left to itself setfacl recomputes it as the
    # union with group::r-x -> rwx, which silently widens csstaff's *effective*
    # access on these files.
    setfacl -m "u:$U:rw-" -m m::rw- "$f"
    echo "write  $f"
  done
  for d in "${TRAVERSE[@]}"; do setfacl -m "u:$U:r-x" "$d"; echo "trav   $d"; done
  for d in "${READ[@]}"; do
    # Not `-R -m u:$U:rX`: the pre-existing group:csstaff:r-x entry makes X
    # resolve to x on every file, which also lifts the file mask from r-- to
    # r-x and widens csstaff. Split files from dirs and pin the mask back.
    find "$d" -type d -exec setfacl -m "u:$U:r-x" {} +
    find "$d" -type f -exec setfacl -m "u:$U:r--" -m m::r-- {} +
    setfacl -d -m "u:$U:r-x" "$d"
    echo "read   $d"
  done
  ;;

verify)
  for d in "${WRITE[@]}" "${READ[@]}" "${TRAVERSE[@]}"; do
    [[ -d $d ]] || continue
    printf '%s  %s\n' "$(stat -c '%A' "$d")" "$d"
    getfacl -p --omit-header "$d" 2>/dev/null | grep -E "^(default:)?user:$U:" | sed 's/^/    /'
  done
  ;;

revoke)
  for d in "${WRITE[@]}"; do
    [[ -d $d ]] || continue
    setfacl -x "u:$U" -d -x "u:$U" "$d" 2>/dev/null || true
    echo "revoked $d"
  done
  for f in "${WRITE_FILES[@]}"; do
    [[ -f $f ]] && { setfacl -x "u:$U" "$f" 2>/dev/null || true; echo "revoked $f"; }
  done
  for d in "${TRAVERSE[@]}"; do setfacl -x "u:$U" "$d" 2>/dev/null || true; echo "revoked $d"; done
  for d in "${READ[@]}"; do
    setfacl -R -x "u:$U" "$d" 2>/dev/null || true
    setfacl -d -x "u:$U" "$d" 2>/dev/null || true
    echo "revoked $d"
  done
  # The sticky bit is deliberately left set: it protects your files from every
  # other collaborator too, and revoking one person's ACL is no reason to drop it.
  ;;

*) echo "usage: $0 <user> [plan|apply|verify|revoke]" >&2; exit 2 ;;
esac
