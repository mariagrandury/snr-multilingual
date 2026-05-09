#!/bin/bash

echo "Checking log files with job ID > 657390 for EngineDeadError and Engine core initialization failed..."
echo "========================================================================================"

# Get unique model names from log files with job ID > 657390
models=$(find logs_xielu -name "*.err" | while read file; do
    job_id=$(basename "$file" | sed 's/.*_\([0-9]*\)\.err/\1/')
    if [ "$job_id" -gt 657390 ]; then
        echo "$file"
    fi
done | sed 's/.*eval-\([^_]*\)_.*\.err/\1/' | sort | uniq)

# Initialize counters and tracking
total_errors=0
engine_dead_errors=0
engine_init_errors=0
node_error_summary=""

for model in $models; do
    echo -n "Checking $model: "
    
    # Find log files for this model with job ID > 657390
    all_logs=$(find logs_xielu -name "eval-${model}_*.err" | while read file; do
        job_id=$(basename "$file" | sed 's/.*_\([0-9]*\)\.err/\1/')
        if [ "$job_id" -gt 657390 ]; then
            echo "$file"
        fi
    done | sort -V)
    
    if [ -z "$all_logs" ]; then
        echo "No log files found with job ID > 657390"
        continue
    fi
    
    model_has_errors=false
    model_errors=""
    
    # Check each log file for errors
    for log_file in $all_logs; do
        # Extract job ID from filename
        job_id=$(basename "$log_file" | sed 's/.*_\([0-9]*\)\.err/\1/')
        
        # Check if the file contains either error
        if grep -q "EngineDeadError\|Engine core initialization failed" "$log_file"; then
            # Extract node ID (nid...)
            node_id=$(grep -o "nid[0-9]*" "$log_file" | head -1)
            
            if [ -n "$node_id" ]; then
                # Check which specific error occurred
                if grep -q "EngineDeadError" "$log_file"; then
                    error_type="EngineDeadError"
                    ((engine_dead_errors++))
                    node_error_summary+="nid${node_id#nid}: EngineDeadError (${model}, job ${job_id})\n"
                elif grep -q "Engine core initialization failed" "$log_file"; then
                    error_type="Engine core initialization failed"
                    ((engine_init_errors++))
                    node_error_summary+="nid${node_id#nid}: Engine core initialization failed (${model}, job ${job_id})\n"
                fi
                
                model_errors+=" ${error_type} on ${node_id} (job ${job_id})"
                model_has_errors=true
                ((total_errors++))
            else
                if grep -q "EngineDeadError" "$log_file"; then
                    model_errors+=" EngineDeadError (job ${job_id}, no node ID)"
                    model_has_errors=true
                    ((total_errors++))
                    ((engine_dead_errors++))
                elif grep -q "Engine core initialization failed" "$log_file"; then
                    model_errors+=" Engine core initialization failed (job ${job_id}, no node ID)"
                    model_has_errors=true
                    ((total_errors++))
                    ((engine_init_errors++))
                fi
            fi
        fi
    done
    
    if [ "$model_has_errors" = true ]; then
        echo "FAILED:$model_errors"
    else
        echo "did not fail"
    fi
done

echo ""
echo "========================================"
echo "SUMMARY (Job ID > 657390)"
echo "========================================"
echo "Total errors found: $total_errors"
echo "EngineDeadError: $engine_dead_errors"
echo "Engine core initialization failed: $engine_init_errors"
echo ""
echo "Node Error Summary:"
echo "=================="
echo -e "$node_error_summary" | sort -V
