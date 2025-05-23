#!/bin/bash
#SBATCH --account=es_schin
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --job-name=debug_agbd_sample
#SBATCH --mem-per-cpu=2048
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=/cluster/scratch/reimannj/agbd_logs/debug_agbd_sample-%j.txt
#SBATCH --error=/cluster/scratch/reimannj/agbd_logs/debug_agbd_sample-%j.txt

# Activate environment
source ~/pangaea-bench-venv/bin/activate

set -euo pipefail

cd /cluster/home/reimannj/pangaea-bench-20250522/

# Print debug info
python --version
which python
pip freeze | grep -E 'numpy|torch|h5py'

# Run the debug script
python debug_agbd_sample.py --num_samples 10 --output_dir /cluster/scratch/reimannj/agbd_debug/euler_test_$(date +%Y%m%d_%H%M%S)

echo "Debug script finished at $(date)"
