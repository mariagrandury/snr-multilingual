#!/bin/bash

echo "Checking if the most recent job (highest job ID) for each model crashed..."
echo "========================================================================================"

# Get unique model names from all log files
models=$(find logs -name "*.err" | sed 's/.*eval-\([^_]*\)_.*\.err/\1/' | sort | uniq)

# Initialize tracking
total_models=0
crashed_models=""
successful_models=""
no_logs_models=""

for model in $models; do
    ((total_models++))
    echo -n "Checking $model: "
    
    # Find all .err files for this model and sort by job ID (highest first)
    err_files=$(find logs -name "eval-${model}_*.err" | while read file; do
        job_id=$(basename "$file" | sed 's/.*_\([0-9]*\)\.err/\1/')
        echo "$job_id:$file"
    done | sort -nr -t: -k1 | head -1 | cut -d: -f2)
    
    if [ -z "$err_files" ]; then
        echo "No .err files found"
        no_logs_models+="$model\n"
        continue
    fi
    
    # Get the most recent job ID
    most_recent_job_id=$(basename "$err_files" | sed 's/.*_\([0-9]*\)\.err/\1/')
    
    # Check if the most recent job crashed
    if grep -q "EngineDeadError\|Engine core initialization failed\|Traceback\|Error\|Exception\|Failed\|Killed\|Out of memory\|OOM\|Segmentation fault\|Core dumped" "$err_files"; then
        echo "CRASHED (job $most_recent_job_id)"
        crashed_models+="$model (job $most_recent_job_id)\n"
    elif grep -q "View project at: https://wandb.ai/apertus/swissai-evals-v0.1.9" "$err_files"; then
        echo "SUCCESS (job $most_recent_job_id)"
        successful_models+="$model (job $most_recent_job_id)\n"
    else
        # Check if job is still running or has other issues
        echo "UNCLEAR STATUS (job $most_recent_job_id)"
        crashed_models+="$model (job $most_recent_job_id) - unclear status\n"
    fi
done

echo ""
echo "========================================"
echo "MOST RECENT JOB STATUS SUMMARY"
echo "========================================"
echo "Total models found: $total_models"
echo "Successful models: $(echo -e "$successful_models" | wc -l)"
echo "Crashed/Unclear models: $(echo -e "$crashed_models" | wc -l)"
echo "Models with no logs: $(echo -e "$no_logs_models" | wc -l)"
echo ""
echo "SUCCESSFUL MODELS (most recent job completed):"
echo "=============================================="
echo -e "$successful_models" | sort -V
echo ""
echo "CRASHED/UNCLEAR MODELS (most recent job failed or unclear):"
echo "=========================================================="
echo -e "$crashed_models" | sort -V
echo ""
if [ -n "$no_logs_models" ]; then
    echo "MODELS WITH NO LOGS:"
    echo "==================="
    echo -e "$no_logs_models" | sort -V
    echo ""
fi
echo "========================================"
echo "Models requiring attention: $(echo -e "$crashed_models" | wc -l)"
echo "========================================"
