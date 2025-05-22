# AGBD Pipeline Debugging, Refactoring, and Data Normalization Diary (2025-05-22)

## Executive Summary
This document provides a comprehensive, step-by-step account of the debugging, refactoring, and validation process for the AGBD dataset integration in the PANGAEA benchmark pipeline, following the events of 2025-05-21. It includes references to all relevant scripts, configuration files, and logs, and documents both the technical rationale and the practical workflow for future reproducibility and troubleshooting.

---

## 1. Context and Motivation

After a successful round of debugging and normalization on 2025-05-21 (see `diary/what I did 21.05.txt`), the goal was to:
- Ensure robust, efficient, and future-proof integration of the AGBD dataset and all related scripts in the PANGAEA benchmark.
- Make all dataset, debug, and image generation scripts robust to missing/malformed modalities and compatible with downstream tasks.
- Enable and review tensor logging for later image generation, ensuring it is efficient, mode-aware, and does not slow down training/evaluation.
- Perform a comprehensive, step-by-step review of all relevant files for consistency, robustness, and best practices.

---

## 2. Refactoring and Robustification

### 2.1 Dataset and Preprocessing Logic
- **Files:**
  - `pangaea-bench/pangaea/datasets/agbd.py` (main AGBD dataset class)
  - `AGBD/Models/dataset.py` (reference/original logic)
  - `pangaea-bench/configs/dataset/agbd.yaml` (dataset config)
  - `pangaea-bench/configs/preprocessing/agbd_reg_resize.yaml` (preprocessing config)
- **Actions:**
  - Compared and reviewed dataset logic for robust handling of modalities, normalization, and metadata.
  - Confirmed compatibility of dataset and preprocessing configs.
  - Ensured correct band order and normalization for Prithvi bands: `['B02', 'B03', 'B04', 'B8A', 'B11', 'B12']`.
  - Added/checked error handling for missing or malformed modalities.

### 2.2 Debug and Visualization Scripts
- **Files:**
  - `pangaea-bench/debug_agbd_sample.py` (debug script)
  - `pangaea-bench/save_agbd_images.py` (image generation and batch summary script)
- **Actions:**
  - Enhanced scripts for robust visualization, batch summary, and outlier flagging.
  - Ensured scripts can handle multi-modal and multi-band input robustly.
  - Added batch-wise stats and warnings for outlier or all-constant data.

### 2.3 Tensor Logging Refactor
- **Files:**
  - `pangaea-bench/pangaea/engine/trainer.py` (training logic, tensor logging)
  - `pangaea-bench/pangaea/engine/evaluator.py` (evaluation logic, tensor logging)
- **Actions:**
  - Refactored `maybe_save_tensor` to be DDP-safe, metadata-rich, error-tolerant, and compatible with downstream scripts.
  - Integrated efficient, representative batch selection and logging into both training and evaluation loops.
  - Ensured unified filename conventions and robust error handling.
  - Validated that tensor logging is efficient and does not slow down training/evaluation.

---

## 3. Debugging and Validation Pipeline

### 3.1 Debug SLURM Script
- **File:** `pangaea-bench/run_agbd_debug.slurm`
- **Actions:**
  - Created a new SLURM script for a fast, small-scale debug run.
  - Configured to save tensors and images to job-specific debug directories (e.g., `/cluster/scratch/reimannj/agbd_tensors/debug_${JOB_ID}/`).
  - Limited tensor/image saving to 10 batches for rapid inspection.
  - Logged system and GPU memory for debugging.
  - Ran the debug job and inspected the output images and logs.

### 3.2 Diagnosing Data Issues
- **Symptoms:** Output images were all red/purple or blank, indicating a data or normalization issue.
- **Actions:**
  - Used a custom script (`/scratch/reimannj3/debug_tensors.py`) to inspect the saved tensor files directly.
  - Discovered that the input tensors were all 1.0 (fully saturated), and GT/pred were not meaningful.
  - Compared the debug script (which worked) and the main pipeline (which was broken).
  - Realized that the normalization logic or stats file in the main pipeline had reverted or was mismatched.

### 3.3 Root Cause and Resolution
- **Root Cause:** The main pipeline's `agbd.py` had reverted to an old version, losing the correct normalization logic.
- **Actions:**
  - Compared the current `agbd.py` to the working version from conversation history.
  - Restored the correct version of `agbd.py` with the proper normalization logic (see lines 1-66 and beyond for correct normalization and band handling).
  - Re-ran the debug script and confirmed that the images and data were now correct and meaningful.

---

## 4. Best Practices and Lessons Learned

- **Version Control:**
  - Noted the importance of using `git` or another version control system to prevent accidental file reverts or overwrites.
  - Recommended regular backups and integrity checks after major edits or merges.

- **Data Validation:**
  - Emphasized the value of direct tensor inspection and debug scripts for diagnosing pipeline issues.
  - Used scripts like `debug_tensors.py` to print stats for saved tensor files and catch all-constant or malformed data early.

- **Documentation:**
  - Documented all changes, rationale, and debugging steps for future reference and reproducibility.
  - Maintained a diary (`diary/what I did 21.05.txt`) and this detailed summary for transparency.

---

## 5. Next Steps

- **Re-run the pipeline on Euler** with the restored and correct normalization logic.
- **Monitor outputs and logs** to ensure the pipeline is functioning as expected.
- **(Optional) Automate file integrity checks** or integrate version control to prevent similar issues in the future.
- **Continue to use debug and validation scripts** to catch issues early in the workflow.

---

## 6. References and File Map

- **Dataset and Preprocessing:**
  - `pangaea-bench/pangaea/datasets/agbd.py`
  - `AGBD/Models/dataset.py`
  - `pangaea-bench/configs/dataset/agbd.yaml`
  - `pangaea-bench/configs/preprocessing/agbd_reg_resize.yaml`
- **Debug and Visualization:**
  - `pangaea-bench/debug_agbd_sample.py`
  - `pangaea-bench/save_agbd_images.py`
  - `pangaea-bench/diary/what I did 21.05.txt`
- **Tensor Logging and Training/Evaluation:**
  - `pangaea-bench/pangaea/engine/trainer.py`
  - `pangaea-bench/pangaea/engine/evaluator.py`
- **SLURM and Job Scripts:**
  - `pangaea-bench/run_agbd_debug.slurm`
  - `pangaea-bench/run_agbd.slurm`
  - `pangaea-bench/run_pangaea_agbd_job.slurm`
- **Debugging and Inspection:**
  - `/scratch/reimannj3/debug_tensors.py`
  - `/scratch/reimannj3/agbd_debug/20250522_004210_val_B02_B03_B04_B8A_B11_B12/debug_log.txt`
  - `/scratch/reimannj3/agbd_debug/20250522_014244_val_B02_B03_B04_B8A_B11_B12/debug_log.txt`

---

## 7. Timeline of Key Events

| Date         | Step/Action                                                                                 |
|--------------|--------------------------------------------------------------------------------------------|
| 2025-05-21   | Diagnosed normalization issue, generated new stats, updated pipeline, documented process.   |
| 2025-05-22   | Refactored tensor logging and evaluation logic.                                            |
| 2025-05-22   | Created and ran debug jobs, inspected outputs.                                             |
| 2025-05-22   | Diagnosed and fixed a critical regression in `agbd.py` normalization logic.                |
| 2025-05-22   | Restored correct code, validated with debug runs, and prepared for full rerun.             |

---

## 8. Lessons for Future Work
- Always validate data at every stage (raw, normalized, tensor, image).
- Use version control for all critical pipeline files.
- Keep debug scripts and logs for reproducibility.
- Document every change and rationale for future reference.

---

## 9. Additional Technical Details and Commands Used

### 9.1. Rsync and Data Management
- To keep the workspace clean and avoid syncing legacy or output folders, the following `rsync` command was used to copy the project to Euler, excluding legacy and output directories:

```bash
rsync -Pav \
  --exclude '20*/' \
  --exclude 'outputs/' \
  --exclude '.git/' \
  --exclude '.github/' \
  --exclude 'AGBD/' \
  --exclude 'legacy_pre_20250522_outputs/' \
  --exclude 'legacy_pre_20250522_slurm/' \
  --exclude 'legacy_pre_20250522_py/' \
  . reimannj@euler.ethz.ch:/cluster/home/reimannj/pangaea-bench-20250522/
```
- This ensured that only the relevant, up-to-date code and configuration files were transferred to the new working directory on Euler, avoiding clutter and confusion from old runs.

### 9.2. Organizing Legacy Files
- To keep the main directory clean, old output and SLURM script folders were moved to legacy directories using:

```bash
mkdir -p legacy_pre_20250522_outputs legacy_pre_20250522_slurm
mv 2025* legacy_pre_20250522_outputs/
mv run_agbd*.slurm run_pangaea_agbd_job.slurm legacy_pre_20250522_slurm/
```
- This made it easy to distinguish between current and legacy files, and to restore or reference old runs if needed.

### 9.3. Debugging Tensor Files
- To inspect the contents of saved tensor files and diagnose data issues, the following script was used:

```python
import torch
import os

tensor_dir = "/scratch/reimannj3/agbd_tensors/debug_32733932"
files = [f for f in os.listdir(tensor_dir) if f.endswith('.pt')]
files.sort()

def print_stats(arr, name):
    arr = arr.cpu().numpy() if hasattr(arr, 'cpu') else arr
    print(f"{name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()}, mean={arr.mean()}, std={arr.std()}")

for fname in files[:5]:
    print(f"\n--- {fname} ---")
    d = torch.load(os.path.join(tensor_dir, fname))
    for k in ['input', 'pred', 'gt', 'logits']:
        if k in d:
            if isinstance(d[k], dict):
                for mod, arr in d[k].items():
                    print_stats(arr, f"input[{mod}]")
            elif d[k] is not None:
                print_stats(d[k], k)
    if 'meta' in d:
        print("meta:", d['meta'])
```
- This script was critical for quickly identifying that the input tensors were all 1.0 due to a normalization bug.

### 9.4. SLURM Job Submission
- Debug and main jobs were submitted using:

```bash
sbatch run_agbd_debug.slurm
sbatch run_pangaea_agbd_job.slurm
```
- The debug SLURM script was specifically designed for rapid iteration and validation, with small batch sizes and limited tensor/image saving.

### 9.5. Debugging and Validation Scripts
- The following scripts were used for in-depth debugging and validation:
  - `debug_agbd_sample.py`: For visualizing and summarizing individual samples from the dataset.
  - `save_agbd_images.py`: For generating images from saved tensor files, with robust handling of multi-band and multi-modal data.
  - `debug_tensors.py`: For printing statistics of saved tensor files.

### 9.6. Normalization and Data Integrity
- After discovering the normalization bug, the correct normalization logic was restored in `pangaea/datasets/agbd.py`.
- The normalization stats file was regenerated using a custom script (`compute_agbd_stats.py`) to ensure it matched the raw HDF5 data scale and reflectance conversion.
- The new stats file (`agbd_stats_from_raw.pkl`) was copied to the correct location and referenced in the pipeline.

### 9.7. Documentation and Diary
- All steps, rationale, and technical details were documented in:
  - `diary/what I did 21.05.txt`
  - `diary/what_we_did_2025-05-22_full_pipeline_debug_and_fix.md` (this file)
- This ensures full reproducibility and transparency for future work and collaboration.

---

**Prepared by:** [Your Name or Team]
**Date:** 2025-05-22
