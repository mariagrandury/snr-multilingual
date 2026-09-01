#!/bin/bash
# launch_bpb.sh — submit score_bpb.sbatch for every cell that still has
# unscored checkpoints. The one command that drives BPB forward.
#
#   bash scripts/launch_bpb.sh --dry-run           # show the plan
#   bash scripts/launch_bpb.sh                     # submit
#   bash scripts/launch_bpb.sh --filter '600M'     # only matching cells
#   bash scripts/launch_bpb.sh --filter '600M-L50' # only matching cells
#
# Idempotent at two layers, so re-running it is always safe:
#   * here — a cell whose converted checkpoints all have bpb.json is skipped,
#     and so is one that already has a job in flight. The latter matters more
#     than it looks: score_bpb.sbatch queues its own singleton successor, so a
#     cell mid-chain has a PENDING job that would otherwise be duplicated on
#     every run of this script.
#   * inside the job — score_bpb.sbatch re-derives the due set and exits
#     without chaining when nothing is left.
set -uo pipefail

STAGING=${STAGING:-/capstor/store/cscs/swissai/infra01/msnr-hf-models}
OUT_ROOT=${OUT_ROOT:-/iopsstor/scratch/cscs/mariagrandury/data-mix-small/Megatron-LM/logs/eval_logs/mariagrandury-epflnlp/msnr}
SBATCH=${SBATCH:-$(dirname "${BASH_SOURCE[0]}")/score_bpb.sbatch}
DRY=0; FILTER='.'

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --filter)  FILTER="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

n_sub=0; n_due=0
for cell in $(ls "$STAGING" 2>/dev/null | grep '^lm-' | grep -E "$FILTER"); do
    if squeue --me -h -n "bpb-$cell" -o %i 2>/dev/null | grep -q .; then
        echo "in-flight $cell"; continue
    fi
    due=0; conv=0
    # .hf_complete is convert-snr.sh's last write: a half-written snapshot has
    # a config.json but no marker, and must not be scored.
    for it in $(find "$STAGING/$cell" -name .hf_complete -printf '%h\n' 2>/dev/null \
                | xargs -r -n1 basename | sed 's/^iter_0*//'); do
        conv=$((conv + 1))
        [ -s "$OUT_ROOT/$cell-iter$it/bpb/bpb.json" ] || due=$((due + 1))
    done
    # "nothing converted" is not "nothing to do" — reporting it as complete
    # would hide a cell whose conversion never ran.
    (( conv == 0 )) && { echo "NO CKPTS  $cell (conversion has not run)"; continue; }
    (( due == 0 ))  && { echo "complete  $cell ($conv/$conv)"; continue; }
    n_due=$((n_due + due))
    if (( DRY )); then
        echo "would submit $cell ($due due)"
    else
        echo -n "submit    $cell ($due due) -> "
        sbatch --job-name="bpb-$cell" "$SBATCH" "$cell"
    fi
    n_sub=$((n_sub + 1))
done

echo "--- $n_sub cell(s), $n_due checkpoint(s) due$( (( DRY )) && echo ' (dry-run)')"
# Reminder rather than an auto-start: the drainer is a long-lived background
# process, and starting one per launch would leave several racing each other.
(( DRY )) || echo "if the debug queue is idle: nohup bash $(dirname "${BASH_SOURCE[0]}")/debug_drain.sh > logs/debug_drain_\$(date +%Y%m%d_%H%M%S).log 2>&1 &"
