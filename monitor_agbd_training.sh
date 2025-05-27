#!/bin/bash

# AGBD Training Monitor - Real-time progress tracking
# Use this script to monitor the progress of the main training script

SCRIPT_DIR="$(dirname "$0")"

# Find the most recent results directory
LATEST_RESULTS_DIR=$(find "$SCRIPT_DIR" -name "agbd_training_results_*" -type d | sort | tail -1)

if [ -z "$LATEST_RESULTS_DIR" ]; then
    echo "No training results directory found. Make sure to run the training script first."
    exit 1
fi

LOG_FILE="${LATEST_RESULTS_DIR}/training_summary.log"
RESULTS_CSV="${LATEST_RESULTS_DIR}/training_results.csv"

echo "Monitoring AGBD training progress..."
echo "Results directory: $LATEST_RESULTS_DIR"
echo "Log file: $LOG_FILE"
echo "Results CSV: $RESULTS_CSV"
echo "=================================="

# Function to show current status
show_status() {
    if [ -f "$LOG_FILE" ]; then
        echo ""
        echo "Latest log entries:"
        echo "==================="
        tail -10 "$LOG_FILE"
    fi
    
    if [ -f "$RESULTS_CSV" ] && [ $(wc -l < "$RESULTS_CSV") -gt 1 ]; then
        echo ""
        echo "Current results summary:"
        echo "========================"
        column -t -s ',' "$RESULTS_CSV"
        
        # Count status
        local total=$(tail -n +2 "$RESULTS_CSV" | wc -l)
        local success=$(tail -n +2 "$RESULTS_CSV" | grep -c "SUCCESS" || echo 0)
        local failed=$(tail -n +2 "$RESULTS_CSV" | grep -c "FAILED" || echo 0)
        
        echo ""
        echo "Progress: $total/14 completed | Success: $success | Failed: $failed"
    fi
}

# Show initial status
show_status

# Watch for changes if requested
if [ "$1" = "--watch" ] || [ "$1" = "-w" ]; then
    echo ""
    echo "Watching for updates (Ctrl+C to stop)..."
    echo "========================================="
    
    while true; do
        sleep 30
        clear
        echo "AGBD Training Monitor - $(date)"
        echo "Results directory: $LATEST_RESULTS_DIR"
        echo "=================================="
        show_status
        
        # Check if training is complete
        if [ -f "$RESULTS_CSV" ]; then
            local total_lines=$(wc -l < "$RESULTS_CSV")
            if [ $total_lines -eq 15 ]; then  # Header + 14 combinations
                echo ""
                echo "🎉 Training complete! All 14 combinations finished."
                break
            fi
        fi
    done
fi
