#!/bin/bash
#SBATCH --account=es_schin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=2:00:00
#SBATCH --job-name=agbd_debug
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/cluster/scratch/reimannj/agbd_logs/agbd_debug-%j.txt
#SBATCH --error=/cluster/scratch/reimannj/agbd_logs/agbd_debug-%j.txt

# Activate environment
source ~/pangaea-bench-venv/bin/activate

# Print debug info
set -euo pipefail

echo "========== JOB INFO =========="
echo "Job ID: $SLURM_JOB_ID"
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
TENSOR_DIR=/cluster/scratch/reimannj/agbd_tensors/debug_${JOB_ID}/
IMAGE_DIR=/cluster/scratch/reimannj/agbd_images/debug_${JOB_ID}/
LOG_DIR=/cluster/scratch/reimannj/agbd_logs/

mkdir -p "$TENSOR_DIR" "$IMAGE_DIR" "$LOG_DIR"

# Log system RAM and GPU memory usage every 60s
# nohup bash -c 'while true; do echo "$(date) | $(free -h)" >> $LOG_DIR/memlog_debug_${JOB_ID}.txt; sleep 60; done' &
# nohup bash -c 'while true; do echo "$(date) | $(nvidia-smi --query-gpu=memory.used --format=csv)" >> $LOG_DIR/gpumemlog_debug_${JOB_ID}.txt; sleep 60; done' &

echo "========== GPU INFO =========="
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}" | tee -a "$LOG_DIR/gpuinfo_debug_${JOB_ID}.txt"
echo "nvidia-smi output:" | tee -a "$LOG_DIR/gpuinfo_debug_${JOB_ID}.txt"
nvidia-smi | tee -a "$LOG_DIR/gpuinfo_debug_${JOB_ID}.txt"
echo "=============================="

################################################################################
# MAIN PANGAEA AGBD DEBUG RUN (small, fast, for pipeline/image validation)
################################################################################

/cluster/home/reimannj/pangaea-bench-venv/bin/torchrun --standalone --nproc_per_node=1 pangaea/run.py \
  dataset=agbd \
  encoder=prithvi \
  decoder=reg_upernet \
  preprocessing=agbd_reg_resize \
  criterion=mse \
  task=regression \
  task.trainer.n_epochs=1 \
  task.trainer.eval_interval=999 \
  use_wandb=False \
  task.trainer.use_wandb=False \
  batch_size=8 \
  test_batch_size=8 \
  test_num_workers=4 \
  limited_label_train=0.0001 \
  limited_label_val=0.0001 \
  +limited_label_test=0.001 \
  limited_label_strategy=random \
  +task.trainer.save_tensors=True \
  +task.trainer.tensor_save_dir=$TENSOR_DIR \
  +task.evaluator.save_tensors=True \
  +task.evaluator.tensor_save_dir=$TENSOR_DIR \
  +task.trainer.max_saved_tensors=10 \
  +task.evaluator.max_saved_tensors=10

################################################################################
# POSTPROCESSING: GENERATE IMAGES FROM SAVED TENSORS
################################################################################

python save_agbd_images.py --input_dir "$TENSOR_DIR" --output_dir "$IMAGE_DIR" --max_images 50

echo "Debug job $SLURM_JOB_ID finished at $(date)"
