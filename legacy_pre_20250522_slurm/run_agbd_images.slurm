#!/bin/bash
#SBATCH --account=es_schin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=04:00:00
#SBATCH --job-name=agbd_images
#SBATCH --output=/cluster/scratch/reimannj/agbd_logs/agbd_images-%j.txt
#SBATCH --error=/cluster/scratch/reimannj/agbd_logs/agbd_images-%j.txt

################################################################################
# AGBD IMAGE GENERATION SLURM JOB SCRIPT
#
# This script generates images from saved AGBD tensors using save_agbd_images.py.
# - Input/output directories are set per job.
# - Logging and output hygiene are enforced.
#
# Author: <your_name> | Date: 2025-05-21
################################################################################

# Activate environment
source ~/pangaea-bench-venv/bin/activate

set -euo pipefail

# Print job info
echo "========== JOB INFO =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Working Dir: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "=============================="

cd /cluster/home/reimannj/pangaea-bench/

# Output directories (job-specific)
# Always require user to specify the tensor/image job IDs for input/output
if [ -z "${TENSOR_JOB_ID:-}" ] || [ -z "${TENSOR_TASK_ID:-}" ]; then
  echo "[ERROR] You must set TENSOR_JOB_ID and TENSOR_TASK_ID before running this script."
  echo "Example: TENSOR_JOB_ID=32709439 TENSOR_TASK_ID=0 sbatch run_agbd_images.slurm"
  exit 1
fi
TENSOR_DIR=/cluster/scratch/reimannj/agbd_tensors/job_${TENSOR_JOB_ID}_${TENSOR_TASK_ID}/
IMAGE_DIR=/cluster/scratch/reimannj/agbd_images/job_${TENSOR_JOB_ID}_${TENSOR_TASK_ID}/
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

################################################################################
# MAIN IMAGE GENERATION
################################################################################

python save_agbd_images.py --input_dir "$TENSOR_DIR" --output_dir "$IMAGE_DIR" --max_images 1000

################################################################################
# END OF JOB
################################################################################
echo "Job $SLURM_JOB_ID finished at $(date)"
