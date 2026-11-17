# AI-Assessment — Phase 4 Hybrid Building Damage Comparison Guide
## Image Alignment + SSIM + YOLO Crack Segmentation

---

## 1. System Architecture Overview

The Hybrid Building Damage Comparison system integrates geometric image registration, structural similarity analysis, and deep-learning crack instance segmentation to perform robust before/after damage assessments of concrete surfaces.

```text
Before Photograph (Moving)        After Photograph (Reference)
         |                                     |
         +------------------+------------------+
                            |
                            v
               Image Upload Validation
            (Format, 10MB, 20MP limits)
                            |
                            v
             Feature Alignment Engine
         (SIFT / AKAZE / ORB / Direct)
                            |
                            v
            Homography H (Before -> After)
                 + Quality Gates
                            |
            +---------------+---------------+
            |                               |
        [Rejected]                      [Accepted]
            |                               |
    HTTP 422 Structured Error               v
                                   Valid Overlap ROI (V)
                               (Nearest-Neighbor Warp)
                                            |
                    +-----------------------+-----------------------+
                    |                                               |
                    v                                               v
            SSIM Visual Comparison                        Parallel YOLO Segmentation
        (Similarity Heatmap & Diff)                     (Original Before & After Photos)
                    |                                               |
                    |                                               v
                    |                                    Warp Before Crack Mask
                    |                                     (Nearest-Neighbor H)
                    |                                               |
                    +-----------------------+-----------------------+
                                            |
                                            v
                                  Spatial Mask Analysis
                            (Persistent, New, No-Longer-Observed)
                                            |
                                            v
                               Spatial Tolerance Adjustment
                              (Bounded Dilation Buffer tau)
                                            |
                                            v
                            Connected Component Progression
                              (New vs Expanded Components)
                                            |
                                            v
                            Composite Hybrid Visual Overlay
                             (Yellow=Persistent, Red=New,
                              Blue=No-Longer-Observed)
                                            |
                                            v
                               FastAPI JSON API Response
                               (POST /api/comparison/hybrid)
```

---

## 2. Coordinate Conventions

1. **Reference Coordinate Frame:** The **After Image** defines the world reference plane $(x_A, y_A)$ with dimensions $(W_A, H_A)$.
2. **Alignment Transformation:** The estimated homography matrix $H \in \mathbb{R}^{3 \times 3}$ maps coordinates from the Before Image into the After Image:
   $$\begin{bmatrix} x_A \\ y_A \\ 1 \end{bmatrix} \sim H \begin{bmatrix} x_B \\ y_B \\ 1 \end{bmatrix}$$
3. **Binary Mask Warping:** To prevent partial-pixel blurring and interpolation artifacts, all binary segmentation masks are transformed using **nearest-neighbor interpolation** (`cv2.INTER_NEAREST`) with constant border value $0$:
   ```python
   aligned_before_mask = cv2.warpPerspective(
       before_mask,
       H,
       (after_width, after_height),
       flags=cv2.INTER_NEAREST,
       borderMode=cv2.BORDER_CONSTANT,
       borderValue=0,
   )
   ```
4. **Common Canvas:** All subsequent operations (SSIM, difference masks, crack masks, overlays) exist strictly in $(W_A, H_A)$ space.

---

## 3. Alignment Quality Gates & Acceptance Criteria

To avoid comparing unrelated scenes or establishing false progression metrics from distorted transformations, homographies must satisfy strict geometric criteria:

| Quality Signal | Threshold | Rationale |
| :--- | :---: | :--- |
| **Minimum Matches** | $\ge 10$ | Ensures sufficient keypoint correspondences |
| **Minimum Inliers** | $\ge 8$ | Verifies geometric consistency under RANSAC |
| **Inlier Ratio** | $\ge 0.45$ | Guards against random or repetitive feature matching |
| **Reprojection Error** | $\le 5.0\text{ px}$ | Caps inlier residual alignment distortion |
| **Determinant ($\det(H)$)** | $> 10^{-6}$ | Rejects degenerate matrices and flips/reflections |
| **Convexity Check** | True | Confirms transformed 4-corner quad is convex and non-self-intersecting |
| **Scale Change Limit** | $\le 4.0\times$ | Rejects extreme artificial enlargement or shrinkage |
| **Perspective Shear** | $|H_{2,0}| \cdot W \le 0.8$ | Rejects severe out-of-plane projective distortion |
| **Valid Overlap Ratio** | $\ge 0.50$ | Rejects pairs with insufficient shared visible scene area |

### Classification:
- **`accepted`**: All quality gates satisfied, confidence $\ge 0.40$.
- **`uncertain`**: Quality gates satisfied, but confidence $< 0.40$ (triggers caveat warning).
- **`direct`**: Explicitly requested; geometric alignment assumed without verification.
- **`rejected`**: Fails quality checks; returns structured HTTP 422 `ALIGNMENT_FAILED` error. No direct fallback is attempted.

---

## 4. Valid Overlap ROI Calculation

The valid comparison region $V$ is computed by transforming a full-ones footprint of the Before Image into the After Image canvas:
$$V(x, y) = \text{warpPerspective}\left(\mathbf{1}_{H_B \times W_B},\ H,\ (W_A, H_A),\ \text{INTER\_NEAREST}\right)$$

- Total reference canvas: $N_{\text{canvas}} = W_A \times H_A$
- Valid overlap pixels: $N_{\text{valid}} = \sum_{(x,y)} [V(x, y) > 0]$
- Valid overlap percentage:
  $$\text{Overlap \%} = \frac{N_{\text{valid}}}{N_{\text{canvas}}} \times 100\%$$

**Universal Denominator Rule:** All damage percentages (crack coverage, SSIM visual change, newly observed cracks) use $N_{\text{valid}}$ as their denominator. Warp borders and invalid regions outside $V$ are completely excluded from calculations.

---

## 5. SSIM Visual Comparison

Visual differences are evaluated between the aligned before image and the reference after image:
1. Both images converted to grayscale and smoothed with Gaussian filter ($5 \times 5$, $\sigma = 0$).
2. Structural Similarity computed using `skimage.metrics.structural_similarity(..., full=True, data_range=255)`.
3. To eliminate window boundary artifacts from perspective warping, $V$ is eroded by a $9 \times 9$ kernel:
   $$V_{\text{safe}} = \text{erode}(V, \mathbf{1}_{9 \times 9})$$
4. Mean SSIM is computed strictly over $V_{\text{safe}} > 0$.
5. Visual difference intensity: $D = (1.0 - \text{SSIM Map}) \times 255$.
6. Otsu thresholding generates binary visual change contours. Contours smaller than 100 px are discarded as noise.

---

## 6. YOLO Crack Segmentation & Mask Alignment

1. **Segment Original Images:**
   To prevent warp-induced artifacting or crack thinning, YOLO26s-seg is executed on the **original untransformed before and after photographs**.
2. **Inference Parameters:**
   - Architecture: `YOLO26s-seg`
   - Confidence: $0.25$
   - IoU: $0.45$
   - Input size: $640$
3. **Coordinate Transformation:**
   The decoded before crack mask $M_B$ is projected into the after image coordinate system using homography $H$ with `INTER_NEAREST`:
   $$B = \text{warpPerspective}(M_B, H, (W_A, H_A)) \ \& \ V$$
   $$A = M_A \ \& \ V$$

---

## 7. Spatial Crack Comparison Equations

Let $B$ be the aligned before crack mask and $A$ be the after crack mask, both restricted to valid overlap $V$:

| Category | Boolean Formulation | Interpretation |
| :--- | :---: | :--- |
| **Persistent Cracks ($P$)** | $B \ \& \ A$ | Cracks detected in both photographs |
| **Newly Observed Cracks ($N$)** | $A \ \& \ (\sim B)$ | Cracks detected only in after photograph |
| **No-Longer-Observed Cracks ($L$)** | $B \ \& \ (\sim A)$ | Cracks detected in before but not after |
| **Raw Crack Change ($C$)** | $B \oplus A = N \mid L$ | Total symmetric difference |

### Spatial Tolerance Adjustment ($\tau$):
Because cracks are only 1-3 pixels wide, minor subpixel alignment residuals can cause boundary pixels to register as false changes.
A bounded morphological structuring element of radius $\tau$ (default $\tau = 2\text{ px}$) is applied:
$$B_{\tau} = \text{dilate}(B, \text{SE}_{\tau})$$
$$A_{\tau} = \text{dilate}(A, \text{SE}_{\tau})$$
- **Tolerance-Adjusted New Pixels:** $N_{\tau} = A \ \& \ (\sim B_{\tau})$
- **Tolerance-Adjusted No-Longer-Observed:** $L_{\tau} = B \ \& \ (\sim A_{\tau})$

---

## 8. Connected Component Analysis for Crack Progression

Connected components are identified via 8-connectivity:
1. **Potential New Crack Components:** After-image crack components that have zero spatial overlap or proximity with any before-image cracks ($C_A \cap B_{\tau} = \emptyset$).
2. **Potential Expanded Components:** After-image components overlapping before-image cracks whose area grew by $\ge 15\%$ and at least 20 pixels:
   $$\text{Area}(C_A) - \sum \text{Area}(C_{B, \text{matched}}) \ge 20 \text{ px} \quad \text{and} \quad \frac{\Delta \text{Area}}{\text{Area}_{\text{matched}}} \ge 0.15$$
3. **Persistent Components:** Crack components with stable area ($\Delta \text{Area} < 15\%$).
4. **No-Longer-Observed Components:** Before-image crack components that disappeared in the after image ($C_B \cap A_{\tau} = \emptyset$).

---

## 9. API Request and Response Reference

### Endpoint
`POST /api/comparison/hybrid`

### Request (Multipart Form-Data)
- `before`: Image file (JPEG, PNG, WebP, max 10MB) [Required]
- `after`: Image file (JPEG, PNG, WebP, max 10MB) [Required]
- `alignment_mode`: `auto` (default) | `sift` | `akaze` | `orb` | `direct`
- `confidence`: float in $(0, 1]$ (default: `0.25`)
- `iou`: float in $(0, 1]$ (default: `0.45`)
- `tolerance_px`: integer in $[0, 10]$ (default: `2`)

### Response (200 OK)
```json
{
  "status": "completed",
  "image_width": 416,
  "image_height": 416,
  "alignment": {
    "status": "accepted",
    "method": "SIFT",
    "confidence": 0.942,
    "inlier_count": 1081,
    "inlier_ratio": 0.9982,
    "reprojection_error": 0.0997,
    "valid_overlap_pixels": 168929,
    "valid_overlap_percentage": 97.6152,
    "warnings": []
  },
  "visual_comparison": {
    "ssim_score": 0.9983,
    "visual_changed_pixels": 0,
    "visual_change_percentage": 0.0,
    "changed_regions_count": 0
  },
  "crack_comparison": {
    "before_crack_pixels": 6814,
    "after_crack_pixels": 6822,
    "before_crack_coverage_percentage": 4.0336,
    "after_crack_coverage_percentage": 4.0384,
    "coverage_change_percentage_points": 0.0048,
    "persistent_pixels": 6814,
    "newly_observed_pixels": 8,
    "no_longer_observed_pixels": 0,
    "raw_change_pixels": 8,
    "raw_change_percentage": 0.0047,
    "tolerance_px": 2,
    "tolerance_adjusted_new_pixels": 0,
    "tolerance_adjusted_no_longer_observed_pixels": 0,
    "potential_new_components": 0,
    "potential_expanded_components": 0,
    "persistent_components": 3,
    "no_longer_observed_components": 0
  },
  "model": {
    "name": "YOLO26s-seg",
    "class_names": ["crack"],
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45
  },
  "images": {
    "aligned_before": "<BASE64_PNG>",
    "visual_difference": "<BASE64_PNG>",
    "before_crack_mask": "<BASE64_PNG>",
    "after_crack_mask": "<BASE64_PNG>",
    "newly_observed_mask": "<BASE64_PNG>",
    "persistent_mask": "<BASE64_PNG>",
    "no_longer_observed_mask": "<BASE64_PNG>",
    "hybrid_overlay": "<BASE64_PNG>"
  },
  "timing": {
    "total_duration_ms": 763.36,
    "alignment_ms": 295.72,
    "ssim_ms": 21.94,
    "yolo_before_ms": 40.37,
    "yolo_after_ms": 57.29,
    "crack_comparison_ms": 6.12,
    "visualization_ms": 14.52
  },
  "warnings": [
    "Crack-region differences represent model predictions and do not establish physical structural deterioration or repair.",
    "Visual change (SSIM) encompasses lighting, surface texture, staining, and camera variations in addition to physical alterations."
  ]
}
```

---

## 10. Operational PowerShell Commands

```powershell
# 1. Start isolated YOLO GPU microservice (port 8001)
backend\venv_yolo\Scripts\python.exe -m uvicorn damage_inference.app:app --host 127.0.0.1 --port 8001 --app-dir backend

# 2. Start main FastAPI backend (port 8000)
backend\venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --app-dir backend

# 3. Execute automated test suites
backend\venv\Scripts\python.exe -m pytest backend/tests/ -o pythonpath=backend -v
backend\venv_yolo\Scripts\python.exe -m pytest backend/tests/test_damage_service.py backend/tests/test_damage_geometry.py -v

# 4. Run real end-to-end validation benchmark
backend\venv\Scripts\python.exe backend/scripts/run_hybrid_validation.py
```

---

## 11. Troubleshooting

1. **`ALIGNMENT_FAILED (422)`**:
   - Cause: Photographs lack sufficient shared keypoints or have severe perspective disparity ($>45^\circ$).
   - Solution: Capture images from similar vantage points with visible concrete texture. If photos are already aligned, use `alignment_mode="direct"`.
2. **`INSUFFICIENT_OVERLAP (422)`**:
   - Cause: Common visible overlap between images is less than 50%.
   - Solution: Ensure both captures center on the same wall region.
3. **`DAMAGE_SERVICE_UNAVAILABLE (503)`**:
   - Cause: YOLO microservice on `127.0.0.1:8001` is not running.
   - Solution: Launch `backend/venv_yolo/Scripts/python.exe -m uvicorn damage_inference.app:app --port 8001`.
4. **`INFERENCE_TIMEOUT (504)`**:
   - Cause: GPU forward pass or mask extraction exceeded the 30-second timeout.
   - Solution: Verify GPU utilization and system RAM.
