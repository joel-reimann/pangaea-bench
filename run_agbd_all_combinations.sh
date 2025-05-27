#!/bin/bash

# AGBD Dataset Training Script - All Valid Encoder/Decoder Combinations
# Runs comprehensive training experiments across all compatible architectures

# set -e  # Exit on any error

# Configuration
DATASET="agbd"
PREPROCESSING="agbd_regression"
CRITERION="mse"
TASK="regression"
BATCH_SIZE=32
TEST_BATCH_SIZE=32
TEST_NUM_WORKERS=4
N_EPOCHS=1
EVAL_INTERVAL=999

# Limited label settings for fast experimentation
LIMITED_LABEL_TRAIN=0.0001
LIMITED_LABEL_VAL=0.0001
LIMITED_LABEL_TEST=0.0001
LIMITED_LABEL_STRATEGY="random"

# Wandb settings
USE_WANDB=true
WANDB_PROJECT="agbd-pangaea-experiments"

# Base directory
PANGAEA_DIR="/scratch/reimannj5/pangaea-bench"
SCRIPT_DIR="$(dirname "$0")"

# Define all valid encoder/decoder combinations
# Based on successful testing from previous integration work
# Note: UNet decoders require encoders with 'topology' attribute (only UNet-family encoders)
# UperNet decoders work with most encoders using encoder.output_dim
declare -A VALID_COMBINATIONS
VALID_COMBINATIONS["remoteclip,reg_upernet"]="RemoteCLIP + UperNet"
VALID_COMBINATIONS["prithvi,reg_upernet"]="Prithvi + UperNet"
VALID_COMBINATIONS["scalemae,reg_upernet"]="ScaleMAE + UperNet"
VALID_COMBINATIONS["gfmswin,reg_upernet"]="GFMSwin + UperNet"
VALID_COMBINATIONS["croma_optical,reg_upernet"]="CROMA-Optical + UperNet"
VALID_COMBINATIONS["croma_sar,reg_upernet"]="CROMA-SAR + UperNet"
VALID_COMBINATIONS["croma_joint,reg_upernet"]="CROMA-Joint + UperNet"
# UNet encoders with UNet decoders (topology-compatible)
VALID_COMBINATIONS["unet_encoder,reg_unet"]="UNet-Encoder + UNet"
# Note: UNet-Encoder + UperNet removed due to initialization incompatibility issues
# Note: UNet decoders removed for non-UNet encoders due to topology attribute requirement

# Results tracking
RESULTS_DIR="${SCRIPT_DIR}/agbd_training_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
LOG_FILE="${RESULTS_DIR}/training_summary.log"
RESULTS_CSV="${RESULTS_DIR}/training_results.csv"

# Initialize results CSV
echo "timestamp,encoder,decoder,description,status,duration_minutes,wandb_run_name,log_file" > "$RESULTS_CSV"

# Function to log messages
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

# Function to run single training experiment
run_training() {
    local encoder="$1"
    local decoder="$2"
    local description="$3"
    
    local start_time=$(date +%s)
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local run_name="agbd_${encoder}_${decoder}_${timestamp}"
    local experiment_log="${RESULTS_DIR}/${run_name}.log"
    
    log_message "Starting: $description (Run: $run_name)"
    
    # Set up environment variables
    export WANDB_RUN_NAME="$run_name"
    export HYDRA_FULL_ERROR=1
    export WANDB_PROJECT="$WANDB_PROJECT"
    
    # Build command
    local cmd="torchrun --standalone --nproc_per_node=1 ${PANGAEA_DIR}/pangaea/run.py \
        dataset=$DATASET \
        encoder=$encoder \
        decoder=$decoder \
        preprocessing=$PREPROCESSING \
        criterion=$CRITERION \
        task=$TASK \
        task.trainer.n_epochs=$N_EPOCHS \
        task.trainer.eval_interval=$EVAL_INTERVAL \
        use_wandb=$USE_WANDB \
        task.trainer.use_wandb=$USE_WANDB \
        task.evaluator.use_wandb=$USE_WANDB \
        batch_size=$BATCH_SIZE \
        test_batch_size=$TEST_BATCH_SIZE \
        test_num_workers=$TEST_NUM_WORKERS \
        limited_label_train=$LIMITED_LABEL_TRAIN \
        limited_label_val=$LIMITED_LABEL_VAL \
        +limited_label_test=$LIMITED_LABEL_TEST \
        limited_label_strategy=$LIMITED_LABEL_STRATEGY"
    
    # Run the experiment
    local status="SUCCESS"
    if ! $cmd > "$experiment_log" 2>&1; then
        status="FAILED"
        log_message "FAILED: $description - Check $experiment_log for details"
    else
        log_message "SUCCESS: $description"
    fi
    
    # Calculate duration
    local end_time=$(date +%s)
    local duration_minutes=$(( (end_time - start_time) / 60 ))
    
    # Record results
    echo "${timestamp},${encoder},${decoder},\"${description}\",${status},${duration_minutes},${run_name},${experiment_log}" >> "$RESULTS_CSV"
    
    return $([ "$status" = "SUCCESS" ] && echo 0 || echo 1)
}

# Main execution
main() {
    log_message "Starting AGBD training experiments across all valid combinations"
    log_message "Results will be saved to: $RESULTS_DIR"
    log_message "Total combinations to test: ${#VALID_COMBINATIONS[@]}"
    
    local success_count=0
    local failure_count=0
    local total_start_time=$(date +%s)
    
    # Run all combinations
    for combination in "${!VALID_COMBINATIONS[@]}"; do
        IFS=',' read -r encoder decoder <<< "$combination"
        description="${VALID_COMBINATIONS[$combination]}"
        
        if run_training "$encoder" "$decoder" "$description"; then
            ((success_count++))
        else
            ((failure_count++))
        fi
        
        # Brief pause between experiments
        sleep 5
    done
    
    # Calculate total duration
    local total_end_time=$(date +%s)
    local total_duration_minutes=$(( (total_end_time - total_start_time) / 60 ))
    
    # Summary
    log_message "================================================"
    log_message "AGBD Training Experiments Complete!"
    log_message "Total combinations: ${#VALID_COMBINATIONS[@]}"
    log_message "Successful runs: $success_count"
    log_message "Failed runs: $failure_count"
    log_message "Total duration: ${total_duration_minutes} minutes"
    log_message "Results saved to: $RESULTS_DIR"
    log_message "Detailed logs: $LOG_FILE"
    log_message "CSV summary: $RESULTS_CSV"
    log_message "================================================"
    
    # Display results table
    echo ""
    echo "Training Results Summary:"
    echo "=========================="
    column -t -s ',' "$RESULTS_CSV"
    
    if [ $failure_count -eq 0 ]; then
        log_message "🎉 All experiments completed successfully!"
        return 0
    else
        log_message "⚠️  Some experiments failed. Check individual logs for details."
        return 1
    fi
}

# Execute main function
main "$@"
