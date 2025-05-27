#!/bin/bash

# Quick AGBD test with known working configuration
set -e

echo "🚀 Quick AGBD integration test..."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/scratch/reimannj5/pangaea-bench/quick_agbd_test_${TIMESTAMP}.log"

echo "Running test with ScaleMAE + UNet (should work)..."

WANDB_MODE=offline \
HYDRA_FULL_ERROR=1 \
torchrun --standalone --nproc_per_node=1 /scratch/reimannj5/pangaea-bench/pangaea/run.py \
    dataset=agbd \
    encoder=scalemae \
    decoder=reg_unet \
    preprocessing=reg_default \
    criterion=mse \
    task=regression \
    task.trainer.n_epochs=1 \
    task.trainer.eval_interval=999 \
    use_wandb=false \
    task.trainer.use_wandb=false \
    task.evaluator.use_wandb=false \
    dataset.img_size=32 \
    batch_size=8 \
    test_batch_size=8 \
    test_num_workers=2 \
    limited_label_train=0.0001 \
    limited_label_val=0.0001 \
    +limited_label_test=0.0001 \
    limited_label_strategy=random \
    > "$LOG_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Quick test PASSED! AGBD integration is working."
    echo "📄 Log saved to: $LOG_FILE"
else
    echo "❌ Quick test FAILED. Check log for details:"
    echo "📄 Log file: $LOG_FILE"
    echo ""
    echo "Last 20 lines of error log:"
    tail -20 "$LOG_FILE"
fi
