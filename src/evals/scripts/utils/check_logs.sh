#!/usr/bin/env bash
set -euo pipefail

# Usage: ./check_logs.sh [LOG_DIR]
# Default: current directory

LOG_DIR="${1:-.}"
if [[ ! -d "$LOG_DIR" ]]; then
  echo "Error: '$LOG_DIR' is not a directory" >&2
  exit 1
fi

shopt -s nullglob

# Sort files to group by model name
mapfile -t sorted_files < <(for f in "$LOG_DIR"/*.err; do [[ -e "$f" ]] && echo "$f"; done | sort)

# Track whether each model has at least one successful run
declare -A model_has_success
declare -A model_seen

prev_model=""
for f in "${sorted_files[@]}"; do
  [[ -e "$f" ]] || continue
  
  # Extract model name (remove final number and .err extension)
  filename=$(basename -- "$f")
  model_name=$(echo "$filename" | sed -E 's/_[0-9]+\.err$//')
  model_seen["$model_name"]=1
  
  # Add newline between different models
  if [[ "$prev_model" != "" && "$prev_model" != "$model_name" ]]; then
    echo ""
  fi
  prev_model="$model_name"

  found=0
  if grep -Fq 'EngingeDeadError' "$f"; then
    line="$(grep -F 'EngingeDeadError' -m1 -- "$f")"
    printf "%s: ❌ %s\n" "$(basename -- "$f")" "$line"
    found=1
  fi

  if grep -Fq 'Engine core initialization failed' "$f"; then
    line="$(grep -F 'Engine core initialization failed' -m1 -- "$f")"
    printf "%s: ❌ %s\n" "$(basename -- "$f")" "$line"
    found=1
  fi

  if grep -Fq "'float' object has no attribute 'keys'" "$f"; then
    line="$(grep -F "'float' object has no attribute 'keys'" -m1 -- "$f")"
    printf "%s: ❌ %s\n" "$(basename -- "$f")" "$line"
    found=1
  fi

  if grep -Fq '🚀 View run' "$f"; then
    line="$(grep -F '🚀 View run' -m1 -- "$f")"
    printf "%s: ✅ %s\n" "$(basename -- "$f")" "$line"
    found=1
    model_has_success["$model_name"]=1
  fi

  if [[ $found -eq 0 ]]; then
    if [[ -s "$f" ]]; then
      last_line="$(tail -n 1 -- "$f")"
    else
      last_line="(empty)"
    fi
    printf "%s: Unknown error 🟡 %s\n" "$(basename -- "$f")" "$last_line"
  fi
done

# Final summary: models with no successful runs
if [[ ${#model_seen[@]} -gt 0 ]]; then
  echo ""
  echo "========================================"
  echo "Models with NO successful runs"
  echo "========================================"
  # Collect and sort
  {
    for m in "${!model_seen[@]}"; do
      if [[ -z "${model_has_success[$m]+yes}" ]]; then
        echo "$m"
      fi
    done
  } | sort
fi


