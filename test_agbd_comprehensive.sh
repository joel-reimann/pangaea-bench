#!/bin/bash

# Comprehensive AGBD Integration Test
# Tests all working encoder-decoder combinations with proper preprocessing

set -e

# Configuration
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="/scratch/reimannj5/pangaea-bench/agbd_final_results_${TIMESTAMP}"
LOG_DIR="${RESULTS_DIR}"
RESULTS_CSV="${RESULTS_DIR}/results.csv"
SUMMARY_LOG="${RESULTS_DIR}/summary.log"

mkdir -p "$RESULTS_DIR"

# Initialize files
echo "timestamp,encoder,decoder,status,error_type,duration_seconds,mse,rmse,notes" > "$RESULTS_CSV"
echo "AGBD Comprehensive Test - $(date)" > "$SUMMARY_LOG"
echo "Using agbd_regression preprocessing with ResizeToEncoder" >> "$SUMMARY_LOG"
echo "========================================" >> "$SUMMARY_LOG"

# Test combinations - all using agbd_regression preprocessing
declare -a COMBINATIONS=(
    "scalemae:reg_upernet"     # CONFIRMED WORKING ✅
    "scalemae:reg_fcn"
    "satmae:reg_upernet"
    "satmae:reg_fcn" 
    "dofa:reg_upernet"
    "dofa:reg_fcn"
    "prithvi:reg_upernet"
    "prithvi:reg_fcn"
    "croma_joint:reg_upernet"  # May need special handling
    "croma_joint:reg_fcn"
)

log_message() {
    echo "$1"
    echo "$(date '+%H:%M:%S') - $1" >> "$SUMMARY_LOG"
}

run_test() {
    local combination="$1"
    local start_time=$(date +%s)
    
    IFS=':' read -ra PARTS <<< "$combination"
    local encoder="${PARTS[0]}"
    local decoder="${PARTS[1]}"
    
    local run_name="agbd_${encoder}_${decoder}_${TIMESTAMP}"
    local log_file="${LOG_DIR}/${run_name}.log"
    
    log_message "Testing: $encoder + $decoder"
    
    # Run test
    local status="RUNNING"
    local error_type=""
    local mse=""
    local rmse=""
    local notes=""
    
    if timeout 900 bash -c "
        WANDB_MODE=offline \
        WANDB_RUN_NAME='$run_name' \
        HYDRA_FULL_ERROR=1 \
        torchrun --standalone --nproc_per_node=1 /scratch/reimannj5/pangaea-bench/pangaea/run.py \
            dataset=agbd \
            encoder=$encoder \
            decoder=$decoder \
            preprocessing=agbd_regression \
            criterion=mse \
            task=regression \
            task.trainer.n_epochs=1 \
            task.trainer.eval_interval=999 \
            use_wandb=false \
            task.trainer.use_wandb=false \
            task.evaluator.use_wandb=false \
            batch_size=8 \
            test_batch_size=8 \
            test_num_workers=2 \
            limited_label_train=0.0001 \
            limited_label_val=0.0001 \
            +limited_label_test=0.0001 \
            limited_label_strategy=random \
        > '$log_file' 2>&1
    "; then
        status="SUCCESS"
        
        # Extract metrics from log
        if grep -q "MSE.*[0-9]" "$log_file"; then
            mse=$(grep "MSE" "$log_file" | tail -1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
            rmse=$(grep "RMSE" "$log_file" | tail -1 | grep -o '[0-9]\+\.[0-9]\+' | head -1)
            notes="Training completed successfully"
        else
            notes="Completed but no metrics found"
        fi
        
        log_message "  ✅ SUCCESS (MSE: ${mse:-N/A}, RMSE: ${rmse:-N/A})"
    else
        status="FAILED"
        
        # Analyze error
        if grep -q "topology" "$log_file"; then
            error_type="TOPOLOGY_ERROR"
            notes="Missing topology attribute"
        elif grep -q "Shape mismatch.*chunks" "$log_file"; then
            error_type="SHAPE_MISMATCH"  
            notes="Patch size incompatibility"
        elif grep -q "CUDA out of memory" "$log_file"; then
            error_type="OOM"
            notes="Out of memory"
        elif grep -q "invalid for input of size" "$log_file"; then
            error_type="INPUT_SIZE_ERROR"
            notes="Input tensor size mismatch"
        else
            error_type="OTHER"
            notes="Unknown error"
        fi
        
        log_message "  ❌ FAILED: $error_type"
    fi
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    # Log to CSV
    echo "$(date -Iseconds),$encoder,$decoder,$status,$error_type,$duration,${mse:-},,${rmse:-},$notes" >> "$RESULTS_CSV"
    
    sleep 3
}

# Main execution
log_message "Starting comprehensive AGBD integration tests"
log_message "Testing ${#COMBINATIONS[@]} encoder-decoder combinations"

total_tests=0
successful_tests=0

for combination in "${COMBINATIONS[@]}"; do
    total_tests=$((total_tests + 1))
    
    log_message ""
    log_message "=== Test $total_tests/${#COMBINATIONS[@]} ==="
    
    run_test "$combination"
    
    if tail -1 "$RESULTS_CSV" | grep -q "SUCCESS"; then
        successful_tests=$((successful_tests + 1))
    fi
done

# Final summary
log_message ""
log_message "======================================="
log_message "AGBD Integration Test COMPLETE!"
log_message "Total tests: $total_tests"
log_message "Successful: $successful_tests"
log_message "Failed: $((total_tests - successful_tests))"
log_message "Success rate: $(( (successful_tests * 100) / total_tests ))%"

echo ""
echo "🎉 Testing complete!"
echo "📁 Results: $RESULTS_DIR"
echo "📊 Summary: $SUMMARY_LOG"
echo "📈 CSV: $RESULTS_CSV"

# Show summary table
if command -v column >/dev/null 2>&1; then
    echo ""
    echo "Results Summary:"
    column -t -s ',' "$RESULTS_CSV"
fi
