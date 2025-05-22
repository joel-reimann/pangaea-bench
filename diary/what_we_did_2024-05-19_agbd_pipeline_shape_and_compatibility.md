# AGBD Pipeline Shape and Compatibility Diary (2024-05-19)

## Executive Summary
This entry details the investigation into the extra singleton dimension in the AGBD input tensor, the rationale for its presence, and the steps taken to ensure compatibility with the PANGAEA pipeline. It also documents the decision to robustify scripts and document the workaround.

---

## 1. Context and Motivation
- **Goal:** Understand and address the `[B, C, 1, H, W]` input shape in AGBD batches.
- **Motivation:** Ensure compatibility with the PANGAEA pipeline, which expects a temporal dimension, and make all scripts robust to this shape.

---

## 2. Investigation and Discussion
### 2.1. Source of Singleton Dimension
- Located `unsqueeze(1)` operation in `pangaea/datasets/agbd.py` as the source of the extra dimension.
- Confirmed that AGBD has no temporal dimension, so a singleton T=1 is added for compatibility.

### 2.2. Documentation and Best Practice
- Decided to document this workaround in the dataset code.
- Downstream scripts (visualization, inspection) should handle both `[B, C, 1, H, W]` and `[B, C, H, W]` gracefully, squeezing the singleton dimension for visualization/stats.

---

## 3. Script Refactoring
- Updated `test.py` to automatically squeeze the singleton dimension and log detailed stats, with clear info/warning/error messages.
- Planned to update `save_agbd_images.py` to:
  - Squeeze singleton temporal dimension if present.
  - Add clear logging for each image, including tensor stats and shape info.
  - Warn if input channels are all 0 or 1.
  - Upsample GT/pred to input image size for visualization if needed.
  - Add comments explaining the singleton dimension and normalization.

---

## 4. Lessons Learned
- Workarounds for data shape compatibility should be clearly documented in code and diaries.
- All scripts should be robust to expected and edge-case tensor shapes.

---

**Prepared by:** [Your Name or Team]
**Date:** 2024-05-19
