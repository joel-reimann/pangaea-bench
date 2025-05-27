#!/bin/bash

# Clean up old AGBD test files and logs

echo "🧹 Cleaning up AGBD test artifacts..."

# Remove old log files
echo "Removing old log files..."
find /scratch/reimannj5/pangaea-bench -name "agbd_*test*.log" -mtime +1 -delete
find /scratch/reimannj5/pangaea-bench -name "quick_agbd_test*.log" -mtime +1 -delete

# Remove old result directories (keep most recent 3)
echo "Cleaning old result directories..."
find /scratch/reimannj5/pangaea-bench -name "agbd_training_results_*" -type d | sort | head -n -3 | xargs rm -rf

# Remove temporary files
echo "Removing temporary files..."
find /scratch/reimannj5/pangaea-bench -name "*.pyc" -delete
find /scratch/reimannj5/pangaea-bench -name "__pycache__" -type d -exec rm -rf {} +

echo "✅ Cleanup complete!"

# Show remaining AGBD-related files
echo ""
echo "Remaining AGBD files:"
find /scratch/reimannj5/pangaea-bench -name "*agbd*" -type f | head -10
