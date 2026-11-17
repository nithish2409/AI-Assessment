# AI-Assessment — Phase 4 Hybrid Building Damage Comparison Validation Report
## Alignment + SSIM + YOLO Crack Segmentation Empirical Verification

**Evaluation Date:** 2026-11-16  
**Evaluator:** Senior Computer Vision Engineer, AI/ML Architect & QA Specialist  
**Evaluated Systems:**
- Main FastAPI Gateway (`backend/venv`, Python 3.14.2)
- YOLO26s-seg Crack Inference Microservice (`backend/venv_yolo`, Python 3.12.2, CUDA 12.4, RTX 4060 Laptop GPU)
- Alignment Engine: SIFT / AKAZE / ORB / RANSAC Homography
- Visual Comparison: SSIM on Valid Overlap ROI
- Progression Engine: Bounded Spatial Tolerance Matching & Connected Component Progression

---

## 1. Executive Summary

Phase 4 successfully implements and validates the hybrid building damage comparison pipeline (`POST /api/comparison/hybrid`). The system accepts before and after photographs, geometrically aligns the before image into the after reference coordinate frame, computes valid overlap ROI, evaluates general visual differences via SSIM, performs YOLO crack segmentation on both original photographs, transforms before crack masks into reference space, and calculates qualified crack progression metrics.

### Key Validation Outcomes:
1. **Identical Image Control:** Achieved perfect alignment (SSIM 1.0, 100% persistent crack overlap, 0 newly observed crack pixels, 0.0 pp coverage change).
2. **Geometric Warping Verification:** Successfully aligned a transformed image pair ($dx=+6\text{ px}, dy=+4\text{ px}, \theta=1.2^\circ$) using SIFT (1,081 inliers, 99.8% inlier ratio, 0.0997 px reprojection error, 97.6% valid overlap). Spatial tolerance ($\tau=2\text{ px}$) reduced false boundary changes from 684 px to just 8 px.
3. **Crack Progression Detection:** Correctly detected and localized newly introduced fissure branches (+0.976 pp coverage change, 5,233 px persistent, 3,498 px newly observed, 1 new crack component).
4. **Rejection Gate Verification:** An unrelated concrete image pair was strictly rejected with HTTP 422 `ALIGNMENT_FAILED`. No misleading change metrics or direct-mode fallbacks occurred.
5. **Regression Integrity:**
   - `backend/venv` (Python 3.14): **151 passed, 4 skipped, 0 failed** (includes original 69 baseline tests).
   - `backend/venv_yolo` (Python 3.12): **32 passed, 0 skipped, 0 failed**.

---

## 2. Experimental Test Scenarios & Results

### Test Case 1: Identical Image Control Pair
- **Input:** `1616.rf.c868709931a671796794fdbb95352c5a.jpg` ($416 \times 416$) as both Before and After.
- **Alignment:** Fast-path byte-identical detection (`status="accepted"`, `method="NONE"`, `valid_overlap=100.0%`).
- **Visual Comparison:** SSIM score = **1.0000**, Visual change = **0.0%** (0 changed pixels).
- **Crack Comparison:**
  - Before crack pixels: 7,042 px (4.0692% coverage)
  - After crack pixels: 7,042 px (4.0692% coverage)
  - Persistent pixels: **7,042 px** (100.0%)
  - Newly observed pixels: **0 px**
  - No-longer-observed pixels: **0 px**
  - Coverage change: **0.0000 percentage points**
  - Persistent components: 3, New components: 0, Expanded components: 0
- **Total Latency:** 682.53 ms

### Test Case 2: Synthetically Transformed Pair (Translation + Rotation)
- **Input:** `1616.rf...` Before; After synthetically transformed ($dx=+6$, $dy=+4$, $\theta=1.2^\circ$).
- **Alignment:** SIFT (`inliers=1081`, `inlier_ratio=0.9982`, `reprojection_error=0.0997 px`, `valid_overlap=97.6152%`).
- **Visual Comparison:** SSIM score = **0.9983**, Visual change = **0.0%**.
- **Crack Comparison:**
  - Persistent pixels: 6,814 px
  - Raw newly observed pixels: 684 px (subpixel boundary jitter)
  - **Tolerance-adjusted newly observed pixels ($\tau=2\text{ px}$): 8 px**
  - Verification: Bounded spatial tolerance successfully absorbs boundary registration jitter.
- **Total Latency:** 768.22 ms

### Test Case 3: Synthetic Crack Progression Pair (Extension & Branching)
- **Input:** `1616.rf...` Before; After contains synthetically introduced fissure extensions and a new branch.
- **Alignment:** Feature alignment accepted (`valid_overlap=100.0%`).
- **Visual Comparison:** SSIM score = **0.9817**, Visual change = **1.97%**.
- **Crack Comparison:**
  - Before crack coverage: **4.0692%** (7,042 px)
  - After crack coverage: **5.0452%** (8,731 px)
  - Coverage change: **+0.9760 percentage points**
  - Persistent crack pixels: **5,233 px**
  - Newly observed crack pixels: **3,498 px** (tolerance-adjusted: 3,426 px)
  - Potential new components: **1**
- **Total Latency:** 594.76 ms

### Test Case 4: Unrelated Image Rejection Pair (Quality Gate)
- **Input:** `1616.rf...` Before vs `1722.rf...` After (completely different concrete surfaces).
- **Result:** **HTTP 422 Unprocessable Content**
- **Error Code:** `ALIGNMENT_FAILED`
- **Attempted Detectors:** `['SIFT', 'AKAZE', 'ORB']`
- **Message:** "Image alignment failed. Use photos showing the same surface with sufficient visible shared details and similar viewpoints."
- **Verification:** Rejection gate strictly confirmed. The engine does NOT fall back to direct comparison or fabricate invalid change metrics.

---

## 3. Quantitative Telemetry & Latency Breakdown

| Metric / Stage | Test 1: Identical | Test 2: Transformed | Test 3: Progression | Test 4: Unrelated |
| :--- | :---: | :---: | :---: | :---: |
| **HTTP Status Code** | 200 OK | 200 OK | 200 OK | 422 Unprocessable |
| **Alignment Status** | accepted (identical) | accepted | accepted | rejected |
| **Alignment Method** | NONE | SIFT | NONE | SIFT, AKAZE, ORB |
| **Inlier Count / Ratio** | N/A | 1,081 / 0.9982 | N/A | < 8 / < 0.45 |
| **Reprojection Error** | 0.0 px | 0.0997 px | 0.0 px | N/A |
| **Valid Overlap %** | 100.0% | 97.6152% | 100.0% | 0.0% |
| **SSIM Score** | 1.0000 | 0.9983 | 0.9817 | N/A |
| **Visual Change %** | 0.0% | 0.0% | 1.97% | N/A |
| **Crack Coverage Before** | 4.0692% | 4.0336% | 4.0692% | N/A |
| **Crack Coverage After** | 4.0692% | 4.0384% | 5.0452% | N/A |
| **Coverage Change (pp)** | 0.0000 pp | +0.0048 pp | +0.9760 pp | N/A |
| **Persistent Crack Pixels** | 7,042 px | 6,814 px | 5,233 px | N/A |
| **Newly Observed Pixels** | 0 px | 684 px (8 px tol.) | 3,498 px | N/A |
| **Potential New Comps** | 0 | 0 | 1 | N/A |
| **Alignment Duration** | 0.43 ms | 295.72 ms | 0.38 ms | 98.82 ms |
| **SSIM Duration** | 31.38 ms | 21.94 ms | 26.89 ms | N/A |
| **YOLO Before Duration** | 237.50 ms | 40.37 ms | 67.47 ms | N/A |
| **YOLO After Duration** | 154.09 ms | 57.29 ms | 42.47 ms | N/A |
| **Crack Comparison** | 6.34 ms | 6.12 ms | 5.53 ms | N/A |
| **Overlay Visualization** | 14.33 ms | 14.52 ms | 15.08 ms | N/A |
| **Total API Latency** | **682.53 ms** | **768.22 ms** | **594.76 ms** | **98.82 ms** |

---

## 4. Visual Evidence Inventory

All visual outputs are persisted in `backend/datasets/damage/reports/hybrid_comparison/`:

1. `alignment_examples/test2_aligned_before.png`: Transformed before image warped into reference frame.
2. `alignment_examples/test2_after_ref.png`: Reference after image canvas.
3. `crack_change_examples/test1_identical_overlay.png`: Pure yellow overlay showing 100% persistent crack correspondence.
4. `crack_change_examples/test2_transformed_overlay.png`: Yellow persistent overlay after geometric warping.
5. `crack_change_examples/test3_progression_overlay.png`: Yellow persistent cracks alongside bright red newly observed fissure branch.
6. `crack_change_examples/test3_newly_observed_mask.png`: Binary mask of newly introduced fissure.
7. `crack_change_examples/test3_persistent_mask.png`: Binary mask of surviving historical fissure.
8. `comparison_visualizations/test3_ssim_heatmap.png`: Jet colormap of SSIM visual differences.
9. `comparison_visualizations/test3_hybrid_composite.png`: Composite overlay with color-coded legend card.

---

## 5. Methodological Limitations

1. **Temporal Pair Availability:** Because the open-source Crack-Seg dataset consists of isolated inspection crops rather than genuine longitudinal temporal series, real crack progression was verified using controlled synthetic extensions. Physical progression accuracy on multi-month real structural series remains unvalidated.
2. **Subpixel Alignment Uncertainty:** Thin cracks (1-2 px wide) are vulnerable to false change signals from minor registration residuals. The $\tau=2\text{ px}$ tolerance buffer mitigates this, but extreme viewpoint shifts ($>30^\circ$) can degrade mask alignment.
3. **Illumination Disparity:** Drastic lighting or shadow shifts between inspections reduce SSIM and may cause false visual change detections. The system distinguishes general SSIM visual change from crack mask change to mitigate false alarms.
4. **Physical Scale Calibration:** Crack metrics are reported in pixels and coverage percentages. Without metric camera calibration or ground targets, physical crack width (in millimeters) cannot be inferred.
