#!/bin/bash
# Check which models from models_pretraining_custom_all.txt have checkpoints available.
# bash configs/signal_to_ratio/check_checkpoints.sh

MODELS_FILE="${1:-configs/signal_to_ratio/models_pretraining_custom_all.txt}"

if [[ ! -f "$MODELS_FILE" ]]; then
    echo "Error: Models file not found: $MODELS_FILE"
    exit 1
fi

echo "Checking checkpoints for models in: $MODELS_FILE"
echo ""

ready=0
missing=0

while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "$line" || "$line" == \#* ]] && continue

    ckpt_dir="${line%/}"
    model_name=$(basename "$(dirname "$ckpt_dir")")

    if [[ ! -d "$ckpt_dir" ]]; then
        echo "[MISSING DIR] $model_name"
        echo "              $ckpt_dir"
        missing=$((missing + 1))
        continue
    fi

    num_ckpts=$(ls -d "${ckpt_dir}"/iter_* 2>/dev/null | wc -l)

    if (( num_ckpts > 0 )); then
        last_iter=$(ls -d "${ckpt_dir}"/iter_* 2>/dev/null \
            | sed 's/.*iter_//' | sed 's/^0*//' \
            | sort -n | tail -n 1)
        echo "[READY]       $model_name  ($num_ckpts checkpoints, latest: iter $last_iter)"
        ready=$((ready + 1))
    else
        echo "[NO CKPTS]    $model_name"
        echo "              $ckpt_dir"
        missing=$((missing + 1))
    fi
done < "$MODELS_FILE"

echo ""
echo "Summary: $ready ready, $missing missing/empty"
