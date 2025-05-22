#!/bin/bash
#SBATCH --account=es_schin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=30:00:00
#SBATCH --job-name=pangaea_agbd_full
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/cluster/scratch/reimannj/agbd_logs/pangaea_agbd-%j.txt
#SBATCH --error=/cluster/scratch/reimannj/agbd_logs/pangaea_agbd-%j.txt

################################################################################
# PANGAEA AGBD FULL PIPELINE SLURM JOB SCRIPT
#
# This script runs the full Pangaea AGBD pipeline with job-specific output dirs.
# - All outputs (tensors, images, logs) are organized by SLURM job/array ID.
# - Logs system and GPU memory usage for debugging.
# - Follows best practices for cluster jobs and output hygiene.
#
# Author: <your_name> | Date: <today's date>
################################################################################


# Notes: On euler, avoid using any GPU below 20GB. Also avoid the V100 32GB GPUs because they are compute capable 7.0. 

# Activate environment
source ~/pangaea-bench-venv/bin/activate

# Print debug info
set -euo pipefail

echo "========== JOB INFO =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node: $(hostname)"
echo "Working Dir: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "hydra-core: $(pip show hydra-core | grep Version)"
echo "PYTHONPATH: ${PYTHONPATH:-}"
echo "=============================="

cd /cluster/home/reimannj/pangaea-bench/

export HYDRA_FULL_ERROR=1

# Output directories (job-specific)
export JOB_ID=${SLURM_JOB_ID}
export TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
TENSOR_DIR=/cluster/scratch/reimannj/agbd_tensors/job_${JOB_ID}_${TASK_ID}/
IMAGE_DIR=/cluster/scratch/reimannj/agbd_images/job_${JOB_ID}_${TASK_ID}/
LOG_DIR=/cluster/scratch/reimannj/agbd_logs/

mkdir -p "$TENSOR_DIR" "$IMAGE_DIR" "$LOG_DIR"

# Check and fix permissions for LOG_DIR to avoid permission denied errors
if ! touch "$LOG_DIR/test_write.txt" 2>/dev/null; then
  echo "[WARN] Cannot write to $LOG_DIR, attempting to chmod..."
  chmod -R u+w "$LOG_DIR"
  if ! touch "$LOG_DIR/test_write.txt" 2>/dev/null; then
    echo "[ERROR] Still cannot write to $LOG_DIR. Logging will be disabled."
    export LOGGING_DISABLED=1
  else
    export LOGGING_DISABLED=0
  fi
else
  export LOGGING_DISABLED=0
fi

# Log system RAM usage every 60s (only if logging enabled)
# if [ "$LOGGING_DISABLED" = "0" ]; then
#   nohup bash -c 'while true; do echo "$(date) | $(free -h)" >> $LOG_DIR/memlog_${JOB_ID}_${TASK_ID}.txt; sleep 60; done' &
#   nohup bash -c 'while true; do echo "$(date) | $(nvidia-smi --query-gpu=memory.used --format=csv)" >> $LOG_DIR/gpumemlog_${JOB_ID}_${TASK_ID}.txt; sleep 60; done' &
# fi

echo "========== GPU INFO =========="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}" | tee -a "$LOG_DIR/gpuinfo_${JOB_ID}_${TASK_ID}.txt"
echo "nvidia-smi output:" | tee -a "$LOG_DIR/gpuinfo_${JOB_ID}_${TASK_ID}.txt"
nvidia-smi | tee -a "$LOG_DIR/gpuinfo_${JOB_ID}_${TASK_ID}.txt"
echo "=============================="

################################################################################
# MAIN PANGAEA AGBD RUN
################################################################################

/cluster/home/reimannj/pangaea-bench-venv/bin/torchrun --standalone --nproc_per_node=1 pangaea/run.py \
  dataset=agbd \
  encoder=prithvi \
  decoder=reg_upernet \
  preprocessing=agbd_reg_resize \
  criterion=mse \
  task=regression \
  task.trainer.n_epochs=10 \
  task.trainer.eval_interval=1 \
  use_wandb=True \
  task.trainer.use_wandb=True \
  batch_size=16 \
  test_batch_size=16 \
  test_num_workers=8 \
  +task.trainer.save_tensors=True \
  +task.trainer.tensor_save_dir=$TENSOR_DIR \
  +task.evaluator.save_tensors=True \
  +task.evaluator.tensor_save_dir=$TENSOR_DIR \
  +task.trainer.max_saved_tensors=100 \
  +task.evaluator.max_saved_tensors=100

################################################################################
# POSTPROCESSING: GENERATE IMAGES FROM SAVED TENSORS
################################################################################

python save_agbd_images.py --input_dir "$TENSOR_DIR" --output_dir "$IMAGE_DIR" --max_images 1000

################################################################################
# END OF JOB
################################################################################
echo "Job $SLURM_JOB_ID finished at $(date)"
