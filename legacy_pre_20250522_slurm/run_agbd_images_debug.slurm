#!/bin/bash
#SBATCH --account=es_schin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8000
#SBATCH --time=01:00:00
#SBATCH --job-name=agbd_images_debug
#SBATCH --output=/cluster/scratch/reimannj/agbd_logs/agbd_images_debug-%j.txt
#SBATCH --error=/cluster/scratch/reimannj/agbd_logs/agbd_images_debug-%j.txt

################################################################################
# AGBD IMAGE GENERATION DEBUG SLURM JOB SCRIPT
#
# This script generates images from saved AGBD tensors in the debug/val directory.
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

TENSOR_DIR=/cluster/scratch/reimannj/agbd_tensors/val/
IMAGE_DIR=/cluster/scratch/reimannj/agbd_images/val/
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
