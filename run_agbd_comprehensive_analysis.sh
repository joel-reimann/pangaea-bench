#!/bin/bash

# AGBD Comprehensive Analysis Script - All Models, Features & Configurations
# Maximizes AGBD dataset utilization with complete model coverage and advanced features

set -e  # Exit on any error

# Configuration
DATASET="agbd"
CRITERION="mse"
TASK="regression"
BATCH_SIZE=32
TEST_BATCH_SIZE=32
TEST_NUM_WORKERS=4
N_EPOCHS=1  # Increased for more meaningful results
EVAL_INTERVAL=999

# Limited label settings for comprehensive testing
LIMITED_LABEL_TRAIN=1   # Increased for better model evaluation
LIMITED_LABEL_VAL=1
LIMITED_LABEL_TEST=1
LIMITED_LABEL_STRATEGY="random"

# Wandb settings
USE_WANDB=true
WANDB_PROJECT="agbd-comprehensive-analysis"

# Base directory
PANGAEA_DIR="/scratch/reimannj5/pangaea-bench"
SCRIPT_DIR="$(dirname "$0")"

# Results tracking
RESULTS_DIR="${SCRIPT_DIR}/agbd_comprehensive_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"
LOG_FILE="${RESULTS_DIR}/comprehensive_analysis.log"
RESULTS_CSV="${RESULTS_DIR}/comprehensive_results.csv"

# Initialize results CSV with extended columns
echo "timestamp,encoder,decoder,preprocessing,description,status,duration_minutes,wandb_run_name,log_file,features_used" > "$RESULTS_CSV"

# ============================================================================
# COMPREHENSIVE MODEL COMBINATIONS
# ============================================================================

# Define all encoder/decoder/preprocessing combinations
declare -A COMPREHENSIVE_COMBINATIONS

# === FOUNDATION MODELS WITH STANDARD REGRESSION ===
COMPREHENSIVE_COMBINATIONS["remoteclip,reg_upernet,agbd_regression"]="RemoteCLIP + UperNet (Standard)"
COMPREHENSIVE_COMBINATIONS["prithvi,reg_upernet,agbd_regression"]="Prithvi + UperNet (Standard)"
COMPREHENSIVE_COMBINATIONS["scalemae,reg_upernet,agbd_regression"]="ScaleMAE + UperNet (Standard)"
COMPREHENSIVE_COMBINATIONS["croma_optical,reg_upernet,agbd_regression"]="CROMA-Optical + UperNet (Standard)"
COMPREHENSIVE_COMBINATIONS["croma_sar,reg_upernet,agbd_regression"]="CROMA-SAR + UperNet (Standard)"
COMPREHENSIVE_COMBINATIONS["croma_joint,reg_upernet,agbd_regression"]="CROMA-Joint + UperNet (Standard)"

# === SSL4EO FAMILY (MISSING FROM ORIGINAL) ===
COMPREHENSIVE_COMBINATIONS["ssl4eo_dino,reg_upernet,agbd_regression"]="SSL4EO-DINO + UperNet"
COMPREHENSIVE_COMBINATIONS["ssl4eo_mae_optical,reg_upernet,agbd_regression"]="SSL4EO-MAE-Optical + UperNet"
COMPREHENSIVE_COMBINATIONS["ssl4eo_data2vec,reg_upernet,agbd_regression"]="SSL4EO-Data2Vec + UperNet"
COMPREHENSIVE_COMBINATIONS["ssl4eo_moco,reg_upernet,agbd_regression"]="SSL4EO-MoCo + UperNet"

# === ADVANCED MODELS (MISSING FROM ORIGINAL) ===
COMPREHENSIVE_COMBINATIONS["dofa,reg_upernet,agbd_regression"]="DOFA + UperNet (Dynamic Optical)"
COMPREHENSIVE_COMBINATIONS["spectralgpt,reg_upernet,agbd_regression"]="SpectralGPT + UperNet"
COMPREHENSIVE_COMBINATIONS["satlasnet_si,reg_upernet,agbd_regression"]="SatlasNet-SI + UperNet"
COMPREHENSIVE_COMBINATIONS["satlasnet_mi,reg_upernet,agbd_regression"]="SatlasNet-MI + UperNet"

# === BASELINE MODELS (MISSING FROM ORIGINAL) ===
COMPREHENSIVE_COMBINATIONS["vit_scratch,reg_upernet,agbd_regression"]="ViT-Scratch + UperNet (Baseline)"
COMPREHENSIVE_COMBINATIONS["resnet50_scratch,reg_upernet,agbd_regression"]="ResNet50-Scratch + UperNet (Baseline)"
COMPREHENSIVE_COMBINATIONS["resnet50_pretrained,reg_upernet,agbd_regression"]="ResNet50-Pretrained + UperNet (Baseline)"

# === TOPOLOGY-COMPATIBLE UNET COMBINATIONS ===
COMPREHENSIVE_COMBINATIONS["unet_encoder,reg_unet,agbd_regression"]="UNet-Encoder + UNet (Standard)"

# === MULTI-TEMPORAL EXPERIMENTS (UNEXPLORED) ===
COMPREHENSIVE_COMBINATIONS["remoteclip,reg_upernet_mt_ltae,agbd_regression"]="RemoteCLIP + MT-UperNet-LTAE"
COMPREHENSIVE_COMBINATIONS["prithvi,reg_upernet_mt_ltae,agbd_regression"]="Prithvi + MT-UperNet-LTAE"
COMPREHENSIVE_COMBINATIONS["scalemae,reg_upernet_mt_linear,agbd_regression"]="ScaleMAE + MT-UperNet-Linear"
COMPREHENSIVE_COMBINATIONS["croma_joint,reg_upernet_mt_ltae,agbd_regression"]="CROMA-Joint + MT-UperNet-LTAE"

# === ALTERNATIVE PREPROCESSING EXPERIMENTS ===
COMPREHENSIVE_COMBINATIONS["remoteclip,reg_upernet,reg_default"]="RemoteCLIP + UperNet (Alt Preprocessing)"
COMPREHENSIVE_COMBINATIONS["prithvi,reg_upernet,reg_default"]="Prithvi + UperNet (Alt Preprocessing)"

# Function to log messages
log_message() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" | tee -a "$LOG_FILE"
}

# Function to check if pretrained model exists
check_model_availability() {
    local encoder="$1"
    local model_paths=(
        "./pretrained_models/${encoder}*.pth"
        "./pretrained_models/B13_vits16_${encoder}*.pth"
        "./pretrained_models/${encoder}_*.pth"
    )
    
    for pattern in "${model_paths[@]}"; do
        if compgen -G "$pattern" > /dev/null; then
            return 0
        fi
    done
    return 1
}

# Function to run single comprehensive experiment
run_comprehensive_experiment() {
    local encoder="$1"
    local decoder="$2"
    local preprocessing="$3"
    local description="$4"
    
    local start_time=$(date +%s)
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local run_name="agbd_comprehensive_${encoder}_${decoder}_${preprocessing}_${timestamp}"
    local experiment_log="${RESULTS_DIR}/${run_name}.log"
    
    log_message "Starting: $description (Run: $run_name)"
    
    # Check model availability for some encoders
    if [[ "$encoder" =~ ^(ssl4eo|dofa|spectralgpt|gfmswin)_ ]] || [[ "$encoder" == "gfmswin" ]]; then
        if ! check_model_availability "$encoder"; then
            log_message "WARNING: Pretrained model for $encoder may not be available, attempting anyway..."
        fi
    fi
    
    # Set up environment variables
    export WANDB_RUN_NAME="$run_name"
    export HYDRA_FULL_ERROR=1
    export WANDB_PROJECT="$WANDB_PROJECT"
    
    # Determine multi-temporal setting based on decoder
    local multi_temporal="false"
    if [[ "$decoder" =~ mt_ ]]; then
        multi_temporal="6"  # Use 6 temporal frames for multi-temporal
    fi
    
    # Build command with comprehensive parameters
    local cmd="torchrun --standalone --nproc_per_node=1 ${PANGAEA_DIR}/pangaea/run.py \
        dataset=$DATASET \
        dataset.multi_temporal=$multi_temporal \
        encoder=$encoder \
        decoder=$decoder \
        preprocessing=$preprocessing \
        criterion=$CRITERION \
        task=$TASK \
        task.trainer.n_epochs=$N_EPOCHS \
        task.trainer.eval_interval=$EVAL_INTERVAL \
        use_wandb=$USE_WANDB \
        batch_size=$BATCH_SIZE \
        test_batch_size=$TEST_BATCH_SIZE \
        test_num_workers=$TEST_NUM_WORKERS \
        limited_label_train=$LIMITED_LABEL_TRAIN \
        limited_label_val=$LIMITED_LABEL_VAL \
        +limited_label_test=$LIMITED_LABEL_TEST \
        limited_label_strategy=$LIMITED_LABEL_STRATEGY \
        hydra.job.chdir=False \
        hydra.run.dir=${RESULTS_DIR}/${run_name}"
    
    log_message "Command: $cmd"
    
    # Execute experiment
    local status="FAILED"
    local features_used="standard"
    
    # Determine features being used
    if [[ "$multi_temporal" != "false" ]]; then
        features_used="multi-temporal,$features_used"
    fi
    if [[ "$preprocessing" != "agbd_regression" ]]; then
        features_used="alt-preprocessing,$features_used"
    fi
    
    if timeout 3600 bash -c "$cmd" >> "$experiment_log" 2>&1; then
        status="SUCCESS"
        log_message "✅ SUCCESS: $description"
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            status="TIMEOUT"
            log_message "⏰ TIMEOUT: $description (1 hour limit)"
        else
            status="FAILED"
            log_message "❌ FAILED: $description (exit code: $exit_code)"
        fi
        
        # Log error details
        log_message "Error details for $description:"
        tail -20 "$experiment_log" | tee -a "$LOG_FILE"
    fi
    
    # Calculate duration
    local end_time=$(date +%s)
    local duration_minutes=$(( (end_time - start_time) / 60 ))
    
    # Record results
    echo "$timestamp,$encoder,$decoder,$preprocessing,\"$description\",$status,$duration_minutes,$run_name,${experiment_log##*/},$features_used" >> "$RESULTS_CSV"
    
    log_message "Completed: $description (Duration: ${duration_minutes}m, Status: $status)"
    echo "----------------------------------------" | tee -a "$LOG_FILE"
}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

log_message "🚀 Starting AGBD Comprehensive Analysis"
log_message "📊 Total combinations to test: ${#COMPREHENSIVE_COMBINATIONS[@]}"
log_message "📁 Results directory: $RESULTS_DIR"

# Execute all combinations
combination_count=0
total_combinations=${#COMPREHENSIVE_COMBINATIONS[@]}

for combination_key in "${!COMPREHENSIVE_COMBINATIONS[@]}"; do
    combination_count=$((combination_count + 1))
    
    IFS=',' read -r encoder decoder preprocessing <<< "$combination_key"
    description="${COMPREHENSIVE_COMBINATIONS[$combination_key]}"
    
    log_message "🔄 Running combination $combination_count/$total_combinations"
    run_comprehensive_experiment "$encoder" "$decoder" "$preprocessing" "$description"
    
    # Brief pause between experiments
    sleep 5
done

# ============================================================================
# ANALYSIS SUMMARY
# ============================================================================

log_message "📋 COMPREHENSIVE ANALYSIS COMPLETE"
log_message "📊 Generating summary report..."

# Generate summary
{
    echo "# AGBD Comprehensive Analysis Summary"
    echo "Generated: $(date)"
    echo ""
    echo "## Overview"
    echo "- Total combinations tested: $total_combinations"
    echo "- Results directory: $RESULTS_DIR"
    echo ""
    echo "## Results Summary"
    echo "\`\`\`"
    echo "Status Distribution:"
    cut -d',' -f5 "$RESULTS_CSV" | tail -n +2 | sort | uniq -c
    echo ""
    echo "Model Family Performance:"
    cut -d',' -f2,5 "$RESULTS_CSV" | tail -n +2 | sort
    echo ""
    echo "Decoder Performance:"
    cut -d',' -f3,5 "$RESULTS_CSV" | tail -n +2 | sort
    echo "\`\`\`"
    echo ""
    echo "## Detailed Results"
    echo "See: $RESULTS_CSV"
    echo ""
    echo "## Key Findings"
    echo "1. SSL4EO family compatibility with AGBD"
    echo "2. Advanced model (DOFA, SpectralGPT, SatlasNet) performance"
    echo "3. Multi-temporal decoder effectiveness"
    echo "4. Baseline model comparison"
    echo "5. Alternative preprocessing impact"
} > "${RESULTS_DIR}/analysis_summary.md"

log_message "📄 Summary report saved: ${RESULTS_DIR}/analysis_summary.md"
log_message "✅ COMPREHENSIVE ANALYSIS COMPLETED"

# Display final summary
echo ""
echo "🎯 FINAL SUMMARY:"
echo "=================="
echo "📁 Results: $RESULTS_DIR"
echo "📊 CSV: $RESULTS_CSV"
echo "📄 Summary: ${RESULTS_DIR}/analysis_summary.md"
echo "📋 Log: $LOG_FILE"
echo ""
echo "🔬 Analysis focused on:"
echo "• Missing SSL4EO family models"
echo "• Advanced models (DOFA, SpectralGPT, SatlasNet)"
echo "• Multi-temporal regression capabilities"
echo "• Baseline model comparisons"
echo "• Alternative preprocessing strategies"
echo "• Complete AGBD feature utilization"