#!/bin/bash

# AGBD Integration Test - PROVEN WORKING VERSION
# This exact configuration worked in agbd_scalemae_test_20250526_144953.log

set -e

echo "🎯 Testing PROVEN WORKING AGBD configuration..."
echo "   Based on: agbd_scalemae_test_20250526_144953.log (SUCCESS)"
echo "   Key fix: Using preprocessing=agbd_regression instead of reg_default"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/scratch/reimannj5/pangaea-bench/agbd_proven_working_${TIMESTAMP}.log"

# EXACT configuration from the successful log - DO NOT CHANGE
WANDB_MODE=offline \
WANDB_RUN_NAME="agbd_proven_working_${TIMESTAMP}" \
HYDRA_FULL_ERROR=1 \
torchrun --standalone --nproc_per_node=1 /scratch/reimannj5/pangaea-bench/pangaea/run.py \
    dataset=agbd \
    encoder=scalemae \
    decoder=reg_upernet \
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
    > "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "🎉 SUCCESS! AGBD integration is WORKING!"
    echo "📄 Log: $LOG_FILE"
    echo ""
    echo "✅ Proven working configuration:"
    echo "   - ScaleMAE encoder"
    echo "   - UPerNet decoder (NOT UNet)" 
    echo "   - agbd_regression preprocessing (NOT reg_default)"
    echo "   - This handles 25×25 → 224×224 resizing automatically"
    echo ""
    echo "🚀 Ready to fix the comprehensive script!"
else
    echo "❌ ERROR: Even the proven configuration failed!"
    echo "📄 Log: $LOG_FILE"
    echo ""
    echo "🔍 Critical error analysis:"
    if grep -q "topology.*not.*attribute" "$LOG_FILE"; then
        echo "   🐛 TOPOLOGY: UNet decoder incompatible - use reg_upernet"
    elif grep -q "chunks of 8" "$LOG_FILE"; then
        echo "   🐛 CROMA: 25÷8 doesn't work - use reg_upernet or resize to 24"
    elif grep -q "invalid for input of size" "$LOG_FILE"; then
        echo "   🐛 SIZE: ResizeToEncoder not working - check implementation"
    elif grep -q "initialize_index" "$LOG_FILE"; then
        echo "   🐛 DATASET: Missing function in AGBD class"
    fi
    
    echo ""
    tail -15 "$LOG_FILE"
fi
