#!/bin/bash
# Generate a self-contained SNR stage runner from a models file (stdout).
# Usage: generate_snr_runner.sh --models <file> [--include SUBSTR] \
#                               (--last N | --total T) [--dense-tail D] [--tail-pct P] \
#                               > runners/snr_<stage>.sh
# Then: bash scripts/launch_evaluations.sh snr-<stage> --script runners/snr_<stage>.sh [flags]
#
# --dense-tail and --tail-pct are forwarded to list_checkpoints.sh; see
# its --help for semantics. Use them to add denser late-training picks on
# top of the evenly-spaced primary set.
#
# --unify-iters: take the longest per-model iter list as the canonical set
# and apply it to ALL Megatron entries. Use when a models file mixes
# fully-trained and half-trained models (e.g. multiple seeds at different
# training stages) — keeps the W&B x-axis grid identical across all of them.
# Half-trained models will list iters they don't have on disk yet; eval jobs
# targeting those fail at Megatron's --exit-on-missing-checkpoint and the
# per-task layer reschedules them next launch.
set -euo pipefail

LIST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/list_checkpoints.sh"

MODELS_FILE=""; MODE=""; COUNT=""; INCLUDE=""; DENSE=""; TAIL_PCT=""; UNIFY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)         MODELS_FILE="$2"; shift 2 ;;
        --include)        INCLUDE="$2"; shift 2 ;;
        --last|--total)   MODE="$1"; COUNT="$2"; shift 2 ;;
        --dense-tail)     DENSE="$2"; shift 2 ;;
        --tail-pct)       TAIL_PCT="$2"; shift 2 ;;
        --unify-iters)    UNIFY=1; shift ;;
        *) echo "Usage: $0 --models <file> [--include SUBSTR] (--last N | --total T) [--dense-tail D] [--tail-pct P] [--unify-iters]" >&2; exit 1 ;;
    esac
done
[[ -f "$MODELS_FILE" && -n "$MODE" ]] \
    || { echo "Usage: $0 --models <file> [--include SUBSTR] (--last N | --total T) [--dense-tail D] [--tail-pct P]" >&2; exit 1; }

LIST_FILTER=()
[[ -n "$INCLUDE" ]] && LIST_FILTER+=(--include "$INCLUDE")
LIST_TAIL=()
[[ -n "$DENSE" ]] && LIST_TAIL+=(--dense-tail "$DENSE")
[[ -n "$TAIL_PCT" ]] && LIST_TAIL+=(--tail-pct "$TAIL_PCT")

derive_name() {
    local m="${1#https://huggingface.co/}"
    [[ "$m" == /* ]] && { m="${m%/}"; m="${m%/checkpoints}"; basename "$m"; } || echo "${m##*/}"
}

MEG=$(mktemp); HF=$(mktemp)
trap 'rm -f "$MEG" "$HF"' EXIT

# Two-pass: collect Megatron entries (and their per-model iter lists), then
# either emit each model's own list or a unified canonical list across all of
# them (--unify-iters). HF entries always use their per-repo list.
MEG_BASES=(); MEG_PATHS=(); MEG_ITERS_LIST=()
while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"; line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue
    base=$(derive_name "$line")
    if ! selected=$(bash "$LIST" "$line" "${LIST_FILTER[@]}" "$MODE" "$COUNT" "${LIST_TAIL[@]}" 2>/dev/null); then
        echo "# WARNING: enumeration failed: $line" >&2; continue
    fi
    case "$line" in
        /iopsstor*|/capstor*)
            MEG_BASES+=("$base")
            MEG_PATHS+=("${line%/}")
            MEG_ITERS_LIST+=("$selected") ;;
        https://huggingface.co/*)
            repo="${line#https://huggingface.co/}"
            while IFS= read -r br; do
                printf '    "%s|%s|%s"\n' "$base" "$repo" "$br" >> "$HF"
            done <<< "$selected" ;;
        *) echo "# WARNING: unrecognized format: $line" >&2 ;;
    esac
done < "$MODELS_FILE"

# Canonical iter list: longest among collected Megatron entries (ties broken
# by max iter so a fully-trained model wins over a half-trained one).
CANONICAL_ITERS=""
if (( UNIFY )) && (( ${#MEG_ITERS_LIST[@]} > 0 )); then
    best_n=-1; best_max=-1; best_idx=0
    for i in "${!MEG_ITERS_LIST[@]}"; do
        its="${MEG_ITERS_LIST[$i]}"
        n=$(echo "$its" | grep -c .)
        max_it=$(echo "$its" | sort -n | tail -1)
        if (( n > best_n || (n == best_n && max_it > best_max) )); then
            best_n=$n; best_max=$max_it; best_idx=$i
        fi
    done
    CANONICAL_ITERS="${MEG_ITERS_LIST[$best_idx]}"
fi

for i in "${!MEG_BASES[@]}"; do
    base="${MEG_BASES[$i]}"
    ckpt="${MEG_PATHS[$i]}"
    iters="${CANONICAL_ITERS:-${MEG_ITERS_LIST[$i]}}"
    while IFS= read -r it; do
        [[ -z "$it" ]] && continue
        printf '    ["%s-iter%s"]="%s"\n' "$base" "$it" "$ckpt" >> "$MEG"
    done <<< "$iters"
done

[[ -s "$MEG" || -s "$HF" ]] || { echo "No checkpoints selected." >&2; exit 1; }

REGEN_INC=""
[[ -n "$INCLUDE" ]] && REGEN_INC=" --include $INCLUDE"
REGEN_TAIL=""
[[ -n "$DENSE" ]] && REGEN_TAIL+=" --dense-tail $DENSE"
[[ -n "$TAIL_PCT" ]] && REGEN_TAIL+=" --tail-pct $TAIL_PCT"
REGEN_UNIFY=""
(( UNIFY )) && REGEN_UNIFY=" --unify-iters"

echo "#!/bin/bash"
echo "# SNR stage runner - GENERATED from $MODELS_FILE (selection:$REGEN_INC $MODE $COUNT$REGEN_TAIL$REGEN_UNIFY)"
echo "# Regenerate: bash scripts/generate_snr_runner.sh --models $MODELS_FILE${REGEN_INC} $MODE $COUNT${REGEN_TAIL}${REGEN_UNIFY} > \$0"

if [[ -s "$MEG" ]]; then
    cat <<'EOF'

# ===== Megatron checkpoints (one sbatch per iter via MODEL_ITERATIONS) =====
export TOKENIZER=${TOKENIZER:-alehc/swissai-tokenizer}
export BOS=${BOS:-true}
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-megatron_lm}

declare -A MODEL_CHECKPOINTS=(
EOF
    cat "$MEG"
    echo ')'; echo ''; echo 'declare -A MODEL_ITERATIONS=('
    # MODEL_ITERATIONS key format: "<model>-iter"; value: iter number.
    sed -E 's/^    \["([^"]+)-iter([0-9]+)"\].*/    ["\1-iter\2-iter"]="\2"/' "$MEG"
    echo ')'; echo ''
    echo 'source runners/hf_base_runner.sh "SNR Megatron checkpoints"'
fi

if [[ -s "$HF" ]]; then
    cat <<'EOF'

# ===== HuggingFace revisions (REVISION is singleton; loop per branch) =====
unset MODEL_CHECKPOINTS MODEL_ITERATIONS
export APPLY_CHAT_TEMPLATE=${APPLY_CHAT_TEMPLATE:-false}
export LM_EVAL_BACKEND=${LM_EVAL_BACKEND:-vllm}

HF_ENTRIES=(
EOF
    cat "$HF"
    cat <<'EOF'
)

for ENTRY in "${HF_ENTRIES[@]}"; do
    IFS="|" read -r NAME REPO BRANCH <<< "$ENTRY"
    export REVISION="$BRANCH"
    unset MODEL_CHECKPOINTS
    declare -A MODEL_CHECKPOINTS=(["${NAME}-${BRANCH}"]="$REPO")
    source runners/hf_base_runner.sh "SNR HuggingFace checkpoints"
done
unset REVISION MODEL_CHECKPOINTS
EOF
fi
