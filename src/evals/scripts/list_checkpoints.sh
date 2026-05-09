#!/bin/bash
# Enumerate Megatron iters (multiples of 2000) or HF non-main branches for one
# model.
#
# Usage:
#   list_checkpoints.sh <model> [--include SUBSTR] (--last N | --total T) \
#                       [--dense-tail D] [--tail-pct P]
#
# --total T:        T checkpoints evenly spaced over the full training run.
# --last N:         the last N checkpoints (no spacing).
# --dense-tail D:   ALSO include up to D checkpoints evenly spaced within the
#                   last P% of training (default P=10). Combined with --total
#                   T, the union is emitted in training order without dupes.
#                   Use this to keep a single low-resolution full-run curve
#                   while getting a denser late-training picture for SNR work.
# --tail-pct P:     percentage of the training span counted as "tail" (default
#                   10). Ignored unless --dense-tail is set.
set -euo pipefail

MODEL="${1:-}"; shift || true
INCLUDE=""
if [[ "${1:-}" == "--include" ]]; then INCLUDE="$2"; shift 2; fi

MODE=""; COUNT=""; DENSE=0; TAIL_PCT=10
while [[ $# -gt 0 ]]; do
    case "$1" in
        --last|--total)
            MODE="${1#--}"; COUNT="$2"; shift 2 ;;
        --dense-tail)
            DENSE="$2"; shift 2 ;;
        --tail-pct)
            TAIL_PCT="$2"; shift 2 ;;
        *)
            echo "Usage: $0 <model> [--include SUBSTR] (--last N | --total T) [--dense-tail D] [--tail-pct P]" >&2; exit 1 ;;
    esac
done

[[ -n "$MODEL" && -n "$MODE" && "$COUNT" =~ ^[1-9][0-9]*$ ]] \
    || { echo "Usage: $0 <model> [--include SUBSTR] (--last N | --total T) [--dense-tail D] [--tail-pct P]" >&2; exit 1; }
[[ "$DENSE" =~ ^[0-9]+$ ]] \
    || { echo "Error: --dense-tail must be a non-negative integer (got '$DENSE')" >&2; exit 1; }
[[ "$TAIL_PCT" =~ ^[1-9][0-9]?$|^100$ ]] \
    || { echo "Error: --tail-pct must be 1-100 (got '$TAIL_PCT')" >&2; exit 1; }

filter_include() { [[ -z "$INCLUDE" ]] && cat || grep -F -- "$INCLUDE"; }

# Pick checkpoints from stdin: --total T evenly spaced, or --last N. Then
# optionally union with D evenly-spaced picks from the last P% of input.
# Output is in original (training) order, deduplicated.
select_lines() {
    awk -v m="$MODE" -v c="$COUNT" -v d="$DENSE" -v p="$TAIL_PCT" '
        { a[NR] = $0 }
        END {
            n = NR
            if (n == 0) { print "no candidates to select from" > "/dev/stderr"; exit 2 }

            # primary selection
            if (m == "last") {
                start = (n > c ? n - c + 1 : 1)
                for (i = start; i <= n; i++) sel[i] = 1
            } else if (c >= n) {
                for (i = 1; i <= n; i++) sel[i] = 1
            } else if (c == 1) {
                sel[n] = 1
            } else {
                for (i = 0; i < c; i++) sel[int(i * (n - 1) / (c - 1)) + 1] = 1
            }

            # dense tail (optional union with primary selection)
            if (d > 0) {
                tail_n = int(n * p / 100)
                if (tail_n < d) tail_n = d
                if (tail_n > n) tail_n = n
                tail_start = n - tail_n + 1
                if (d >= tail_n) {
                    for (i = tail_start; i <= n; i++) sel[i] = 1
                } else if (d == 1) {
                    sel[n] = 1
                } else {
                    for (i = 0; i < d; i++) sel[tail_start + int(i * (tail_n - 1) / (d - 1))] = 1
                }
            }

            for (i = 1; i <= n; i++) if (i in sel) print a[i]
        }'
}

case "$MODEL" in
    /iopsstor*|/capstor*)
        [[ -d "$MODEL" ]] || { echo "Not a directory: $MODEL" >&2; exit 2; }
        shopt -s nullglob; D=( "${MODEL%/}"/iter_* ); shopt -u nullglob
        (( ${#D[@]} )) || { echo "No iter_* in $MODEL" >&2; exit 2; }
        printf '%s\n' "${D[@]##*/iter_}" | sed 's/^0*//' | sort -n \
            | awk '$1 % 2000 == 0' | filter_include | select_lines
        ;;
    https://huggingface.co/*)
        # sort -V so step-40000 precedes step-1000000 (alphabetic sort would flip them,
        # making --total picks non-monotonic in step count).
        python3 - "${MODEL#https://huggingface.co/}" <<'PY' | sort -V | filter_include | select_lines
import json, sys, urllib.request
r = json.loads(urllib.request.urlopen(f'https://huggingface.co/api/models/{sys.argv[1]}/refs').read())
for b in (x['name'] for x in r['branches'] if x['name']!='main'): print(b)
PY
        ;;
    *) echo "Unrecognized format: $MODEL (want /iopsstor*, /capstor*, or https://huggingface.co/...)" >&2; exit 1 ;;
esac
