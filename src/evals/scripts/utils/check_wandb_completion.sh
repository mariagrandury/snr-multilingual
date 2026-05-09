#!/bin/bash

echo "Checking which models have completed successfully (contain wandb project view link)..."
echo "========================================================================================"

# Get unique model names from all log files
models=$(find logs -name "*.err" | sed 's/.*eval-\([^_]*\)_.*\.err/\1/' | sort | uniq)

# Initialize tracking
completed_models=""
incomplete_models=""
total_models=0
completed_count=0

for model in $models; do
    ((total_models++))
    echo -n "Checking $model: "
    
    # Find all .err files for this model
    err_files=$(find logs -name "eval-${model}_*.err" | sort -V)
    
    if [ -z "$err_files" ]; then
        echo "No .err files found"
        incomplete_models+="$model\n"
        continue
    fi
    
    model_completed=false
    
    # Check each .err file for the wandb completion message
    for err_file in $err_files; do
        if grep -q "View project at: https://wandb.ai/apertus/swissai-evals-v0.1.9" "$err_file"; then
            # Extract job ID from filename for reference
            job_id=$(basename "$err_file" | sed 's/.*_\([0-9]*\)\.err/\1/')
            echo "COMPLETED (job $job_id)"
            completed_models+="$model (job $job_id)\n"
            model_completed=true
            ((completed_count++))
            break
        fi
    done
    
    if [ "$model_completed" = false ]; then
        echo "NOT COMPLETED"
        incomplete_models+="$model\n"
    fi
done

echo ""
echo "========================================"
echo "WANDB COMPLETION SUMMARY"
echo "========================================"
echo "Total models found: $total_models"
echo "Completed models: $completed_count"
echo "Incomplete models: $((total_models - completed_count))"
echo ""
echo "COMPLETED MODELS:"
echo "================"
echo -e "$completed_models" | sort -V
echo ""
echo "INCOMPLETE MODELS (missing wandb completion):"
echo "============================================="
echo -e "$incomplete_models" | sort -V
echo ""
echo "========================================"
echo "Models requiring attention: $((total_models - completed_count))"
echo "========================================"
