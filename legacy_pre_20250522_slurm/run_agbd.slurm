#!/bin/bash
#SBATCH --account=es_schin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
#SBATCH --time=16:00:00
#SBATCH --time=4:00:00
#SBATCH --job-name=agbd_test
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=BEGIN,END

# module load stack/2024-06
# module load python/3.11.6
source ~/pangaea-bench-venv/bin/activate

# Debugging: print python and hydra info
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"
echo "hydra-core version: $(pip show hydra-core | grep Version)"
echo "PYTHONPATH: $PYTHONPATH"

cd /cluster/home/reimannj/pangaea-bench/


export HYDRA_FULL_ERROR=1

# Print GPU info for debugging
echo "GPU info:" && nvidia-smi

# Log system RAM usage every 60s
# while true; do
#   echo "$(date) | $(free -h)" >> memlog.txt
#   sleep 60
# done &

# # Log GPU memory usage every 60s
# while true; do
#   echo "$(date) | $(nvidia-smi --query-gpu=memory.used --format=csv)" >> gpumemlog.txt
#   sleep 60
# done &

# --- FULL RUN (commented out for now) ---
# /cluster/home/reimannj/pangaea-bench-venv/bin/torchrun --standalone --nproc_per_node=1 pangaea/run.py \
#   dataset=agbd \
#   encoder=prithvi \
#   decoder=reg_upernet \
#   preprocessing=agbd_reg_resize \
#   criterion=mse \
#   task=regression \
#   task.trainer.n_epochs=2 \
#   task.trainer.eval_interval=999 \
#   use_wandb=True \
#   task.trainer.use_wandb=True \
#   test_batch_size=32 \
#   test_num_workers=8

# --- FAST SUBSET RUN (should finish in <4h) ---
# /cluster/home/reimannj/pangaea-bench-venv/bin/torchrun --standalone --nproc_per_node=1 pangaea/run.py \
#   dataset=agbd \
#   encoder=prithvi \
#   decoder=reg_upernet \
#   preprocessing=agbd_reg_resize \
#   criterion=mse \
#   task=regression \
#   task.trainer.n_epochs=2 \
#   task.trainer.eval_interval=999 \
#   use_wandb=True \
#   task.trainer.use_wandb=True \
#   test_batch_size=32 \
#   test_num_workers=8 \
#   limited_label_train=0.0001 \
#   limited_label_val=0.0001 \
#   +limited_label_test=0.001 \
#   limited_label_strategy=random


# Corrected run with the right path
/cluster/home/reimannj/pangaea-bench-venv/bin/torchrun --standalone --nproc_per_node=1 pangaea/run.py \
  dataset=agbd \
  encoder=prithvi \
  decoder=reg_upernet \
  preprocessing=agbd_reg_resize \
  criterion=mse \
  task=regression \
  task.trainer.n_epochs=2 \
  task.trainer.eval_interval=999 \
  use_wandb=False \
  task.trainer.use_wandb=False \
  test_batch_size=32 \
  test_num_workers=8 \
  limited_label_train=0.0001 \
  limited_label_val=0.0001 \
  +limited_label_test=0.001 \
  limited_label_strategy=random \
  +task.trainer.save_tensors=True \
  +task.trainer.tensor_save_dir=/cluster/scratch/reimannj/agbd_tensors/val/ \
  +task.evaluator.save_tensors=True \
  +task.evaluator.tensor_save_dir=/cluster/scratch/reimannj/agbd_tensors/val/