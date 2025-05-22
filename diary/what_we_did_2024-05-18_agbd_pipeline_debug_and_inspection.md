# AGBD Pipeline Debugging and Inspection Diary (2024-05-18)

## Executive Summary
This entry documents the initial investigation into the AGBD data pipeline and visualization issues. The focus was on identifying why model input, prediction, and ground truth (GT) images were not meaningful, and on gathering evidence about the data pipeline, tensor shapes, and file contents.

---

## 1. Context and Motivation
- **Goal:** Diagnose why AGBD model input images appeared black/red, GT was flat purple, and predictions were tiny dots.
- **Motivation:** Ensure the data pipeline, tensor saving, and visualization scripts are correct and produce interpretable outputs for downstream analysis and model debugging.

---

## 2. Initial Symptom and Hypothesis
- **Symptoms:**
  - Input images were all black/red.
  - GT images were flat purple.
  - Prediction images were tiny dots.
- **Hypotheses:**
  - Data normalization or loading bug.
  - Tensor shape mismatch.
  - Visualization script not handling data correctly.

---

## 3. Inspection and Evidence Gathering
### 3.1. Code and File Review
- Located and reviewed the following files:
  - `save_agbd_images.py` (visualization script)
  - `test.py` (batch inspection script)
  - `/cluster/scratch/reimannj/agbd_tensors/val/val_batch_0.pt` (example batch file)
  - Mapping and HDF5 files for AGBD data

### 3.2. Debug Logging and Stats
- Added debug logging to `save_agbd_images.py` to print tensor stats before saving images.
- Ran the visualization script and confirmed:
  - Input tensors were all 0/1.
  - GT was constant per patch.
  - Prediction had some variation but was tiny.
- Inspected mapping and HDF5 files:
  - Confirmed mapping and HDF5 structure were correct and contained real data.

### 3.3. Batch Inspection Script
- Wrote and ran `test.py` to print input, GT, and pred tensor shapes and stats.
- Discovered input tensor in the batch file was a dict (with key 'optical') and had shape `[32, 6, 1, 224, 224]` (extra singleton dimension).
- Confirmed GT and pred were `[32, 25, 25]` and GT was constant per patch.

---

## 4. Next Steps
- Trace the source of the extra singleton dimension in the input tensor (likely in `pangaea/datasets/agbd.py`).
- Update scripts to handle the correct input shape.
- Continue debugging to ensure meaningful visualizations.

---

## 5. Lessons Learned
- Direct inspection of tensors and batch files is critical for diagnosing data pipeline issues.
- Debug logging in visualization scripts is essential for catching silent errors.

---

**Prepared by:** [Your Name or Team]
**Date:** 2024-05-18
