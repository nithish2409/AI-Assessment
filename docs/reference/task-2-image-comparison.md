# TASK 2 — BEFORE/AFTER IMAGE COMPARISON

**Project:** AI-Assessment  
**Module:** Task 2 — Image Comparison  
**Document Type:** Technical Implementation and Completion Documentation  
**Status:** COMPLETED — FINAL ACCEPTANCE PASSED  
**Date:** 16 November 2026

---

## 1. Introduction

The Before/After Image Comparison module is a computer vision-based system developed to identify and quantify visual differences between two building inspection photographs.

The system accepts a reference photograph (Before) and a subsequent photograph (After), performs geometric image alignment where possible, calculates structural similarity, identifies changed regions, and generates visual representations of the detected differences.

The module uses classical computer vision and image processing techniques rather than a trained damage classification model. It is designed to support visual inspection by identifying areas that may require further investigation.

The system is integrated into a full-stack application using FastAPI for backend processing and React for the user interface.

## 2. Problem Statement

Manual comparison of building inspection photographs is time-consuming and prone to human error, especially when the differences are small or distributed across a large surface.

The objective is to develop a system capable of comparing two images, highlighting visual changes, quantifying their extent, and presenting interpretable results.

A major technical challenge is that photographs may differ in resolution, camera viewpoint, lighting, and perspective. Direct pixel-by-pixel comparison of unaligned images can produce misleading results.

The system therefore incorporates feature-based image registration, alignment verification, similarity measurement, and structured error handling.

## 3. Objectives

The primary objectives of Task 2 are:

1. Accept two building inspection photographs representing Before and After conditions.
2. Validate the uploaded images and their properties.
3. Perform feature-based image alignment using SIFT, AKAZE, and ORB.
4. Support explicit Direct comparison for images already known to be geometrically aligned.
5. Measure image similarity using the Structural Similarity Index (SSIM).
6. Identify regions containing visual differences.
7. Calculate the percentage of the image exhibiting detected changes.
8. Generate visual comparison outputs, including heatmaps, masks, and highlighted regions.
9. Display alignment diagnostics and comparison statistics.
10. Handle invalid images, incompatible photographs, and failed alignment safely.
11. Provide an interactive frontend for uploading, comparing, and inspecting images.
12. Verify functionality through automated tests and controlled evaluation scenarios.

## 4. Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python 3.14.2 |
| Backend framework | FastAPI |
| Computer vision | OpenCV |
| Image processing | NumPy |
| Structural similarity | scikit-image |
| Image registration | SIFT, AKAZE, ORB |
| Geometric transformation | Homography and RANSAC |
| Frontend | React 19 |
| Frontend build tool | Vite 8 |
| Styling | Tailwind CSS 4 |
| API communication | Axios |
| Backend testing | pytest |
| Frontend testing | Node.js test runner |
| Development environment | Windows 11, VS Code |

Task 2 uses the main FastAPI backend and does not require the separate YOLO inference microservice for its general visual comparison functionality.

## 5. System Architecture

The system follows a frontend-backend architecture.

### 5.1 Architecture Overview

```text
USER
 |
 v
REACT FRONTEND
 |
 | Upload Before and After Images
 | Select Alignment Mode
 | Validate / Compare
 v
AXIOS HTTP CLIENT
 |
 v
FASTAPI BACKEND
 |
 +-- Image Validation
 |
 +-- Image Decoding and Normalization
 |
 +-- Geometric Alignment
 |    |
 |    +-- SIFT
 |    +-- AKAZE
 |    +-- ORB
 |    +-- RANSAC Homography
 |    +-- Alignment Verification
 |
 +-- SSIM Analysis
 |
 +-- Difference Mask Generation
 |
 +-- Changed Region Detection
 |
 +-- Change Percentage Calculation
 |
 v
STRUCTURED JSON RESPONSE
 |
 v
REACT RESULTS DASHBOARD
 |
 +-- Similarity Score
 +-- Changed Area Percentage
 +-- Changed Region Count
 +-- Alignment Diagnostics
 +-- SSIM Heatmap
 +-- Highlighted Changes
 +-- Binary Change Mask
 +-- Image Comparison Viewer
```

### 5.2 Architectural Responsibilities

**Frontend**

The frontend manages image uploads, user input, comparison requests, loading states, errors, result visualization, and user interaction.

**Backend API**

The API receives uploaded files, validates requests, invokes comparison services, and returns structured results.

**Image Alignment Service**

The alignment service estimates geometric correspondence between images using local feature matching and homography.

**Image Comparison Engine**

The comparison engine calculates SSIM, generates difference images, extracts changed regions, and calculates change statistics.

**Error Handling Layer**

The error handling layer communicates alignment failures, invalid inputs, and operational failures through structured API responses.

## 6. Image Comparison Methodology

### 6.1 Image Upload and Validation

The system accepts two images:

- **Before Image:** The reference inspection photograph.
- **After Image:** The photograph being compared against the reference.

The validation stage checks that both files can be decoded as supported images and extracts their dimensions and channel information.

The frontend provides a dedicated Validate Pair action before comparison.

Validation confirms that the files are technically suitable for processing. It does not guarantee that the photographs depict the same physical scene.

### 6.2 Image Preprocessing

Images are decoded and prepared for processing.

Depending on the selected alignment mode and input dimensions, preprocessing may include:

- Image decoding.
- Color-space conversion.
- Resolution normalization.
- Grayscale conversion for feature extraction and similarity analysis.
- Preparation of a common coordinate frame.

Image dimension normalization is recorded in the response metadata.

### 6.3 Geometric Image Alignment

Image alignment is necessary when photographs have small changes in viewpoint, orientation, position, or scale.

The system supports two alignment modes.

#### A. Automatic Feature-Based Alignment

Automatic alignment attempts feature detectors in the following order:

1. SIFT — Scale-Invariant Feature Transform.
2. AKAZE — Accelerated-KAZE.
3. ORB — Oriented FAST and Rotated BRIEF.

The system extracts local image features, matches corresponding descriptors, filters ambiguous matches, and estimates a geometric transformation.

Lowe's ratio test is used to reject ambiguous descriptor matches.

RANSAC is applied to estimate a robust homography while rejecting inconsistent correspondences.

The resulting transformation maps the moving image into the reference coordinate system.

The estimated alignment must satisfy geometric acceptance checks before the comparison proceeds.

**Important:** The system does not silently switch to Direct mode when automatic alignment fails.

#### B. Direct Alignment

Direct mode bypasses feature-based geometric registration.

It assumes that corresponding pixels represent the same scene locations.

This mode is appropriate for:

- Images captured using a fixed camera setup.
- Precisely registered image pairs.
- Controlled synthetic test images.
- Images known to share the same pixel coordinate system.

Direct mode is not suitable for unrelated photographs or images with uncorrected camera shifts.

When image dimensions differ, the system normalizes them and reports the operation. Resizing alone does not establish geometric correspondence.

### 6.4 Alignment Acceptance Criteria

The implementation uses the following reported configuration:

| Parameter | Value |
|---|---|
| Lowe ratio | 0.75 |
| Minimum good matches | 10 |
| RANSAC reprojection threshold | 3.0 pixels |
| Minimum inliers | 10 |
| Minimum inlier ratio | 0.15 |
| Maximum scale change | 2.5 |
| Maximum perspective/skew parameter | 0.002 |
| Maximum mean reprojection error | 4.0 pixels |
| Minimum valid overlap ratio | 0.65 |

These values are engineering acceptance thresholds for the implemented registration algorithm. They are not guarantees that every accepted pair is perfectly aligned.

## 7. Structural Similarity Index (SSIM)

The Structural Similarity Index measures the similarity between two images by considering luminance, contrast, and structural information.

SSIM is defined as:

\[
SSIM(x,y)=\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
\]

Where:

- \(\mu_x,\mu_y\) represent the mean intensities.
- \(\sigma_x^2,\sigma_y^2\) represent image variances.
- \(\sigma_{xy}\) represents covariance.
- \(C_1,C_2\) are stability constants.

A score of 1.0 indicates identical images under the evaluated comparison.

Lower scores indicate increasing structural or photometric differences.

### 7.1 SSIM Difference Map

The system generates a spatial difference representation using the SSIM map.

Conceptually:

\[
D(x,y)=1-SSIM_{map}(x,y)
\]

Higher difference values indicate regions with greater local dissimilarity.

The difference map is converted into a visual heatmap to help users locate changes.

### 7.2 Important Interpretation

SSIM detects general visual differences, including changes caused by:

- Cracks or surface alterations.
- Stains and discoloration.
- Lighting variations.
- Texture changes.
- Shadows.
- Image compression and noise.

Therefore, SSIM differences must not automatically be interpreted as structural damage.

## 8. Change Detection and Quantification

### 8.1 Difference Mask Generation

The SSIM difference map is processed to identify pixels exhibiting meaningful visual differences.

The image processing pipeline uses thresholding and morphological operations to produce a binary change mask.

The mask separates candidate changed pixels from relatively unchanged pixels.

### 8.2 Connected Region Detection

Connected regions are extracted from the binary change mask.

Small or insignificant regions may be filtered according to the configured area criteria.

For each retained region, the system reports spatial information such as:

- Position (X, Y).
- Bounding box width and height.
- Region area in pixels.

### 8.3 Changed Area Percentage

The changed area percentage is calculated relative to the valid comparison area.

\[
ChangedArea(\%)=
\frac{N_{changed}}{N_{valid}}\times100
\]

Where:

- \(N_{changed}\) is the number of pixels classified as changed within the valid region.
- \(N_{valid}\) is the number of pixels in the valid comparison region.

This metric represents the percentage of the analyzed image exhibiting detected visual changes.

It does not represent the percentage of physical structural damage.

## 9. Alignment Reliability and Verification

The system includes explicit alignment status metadata to distinguish verified registration from assumed correspondence.

### 9.1 Automatic Alignment

For accepted feature-based alignment:

```json
{
  "geometric_status": "VERIFIED_FEATURE_REGISTRATION",
  "method_used": "SIFT",
  "confidence": 0.9142,
  "matches": 142,
  "inliers": 108,
  "inlier_ratio": 0.7606,
  "overlap_ratio": 0.9654,
  "reprojection_error": 1.18
}
```

The frontend displays the alignment method, quality metric, matching statistics, reprojection error, and valid overlap.

The values above illustrate the response structure; they are not universal performance figures.

### 9.2 Direct Alignment

For Direct mode:

```json
{
  "geometric_status": "ASSUMED_UNVERIFIED",
  "method_used": "DIRECT",
  "confidence": null,
  "matches": null,
  "inliers": null,
  "inlier_ratio": null,
  "reprojection_error": null,
  "overlap_ratio": 1.0
}
```

The frontend displays:

- Geometric Registration: N/A.
- Status: Unverified.
- Inliers / Matches: N/A.
- Reprojection Error: N/A.
- Direct Alignment Notice.

This avoids incorrectly displaying 0% alignment quality when no feature registration was attempted.

The 100% overlap value in Direct mode represents the normalized canvas overlap, not independently verified scene overlap.

### 9.3 Identical Images

Byte-identical inputs are handled through an identity fast path.

The system can report:

- SSIM: 1.0000.
- Changed area: 0%.
- Changed regions: 0.
- Identity status.

This is an exact-input identity check rather than an independently estimated feature-based registration.

## 10. Dimension Normalization

The system supports input images with different resolutions.

When normalization occurs, the API records source and processing dimensions.

Example:

```json
{
  "dimension_normalized": true,
  "dimension_details": {
    "before_source": {
      "width": 2048,
      "height": 1536,
      "aspect_ratio": 1.333
    },
    "after_source": {
      "width": 1448,
      "height": 1086,
      "aspect_ratio": 1.333
    },
    "processing_dimensions": {
      "width": 1200,
      "height": 900
    },
    "aspect_ratio_match": true
  }
}
```

When source aspect ratios differ significantly, the frontend displays a distortion warning.

The reported warning threshold is an absolute aspect-ratio difference greater than 0.02.

Dimension normalization ensures compatible processing dimensions but cannot independently verify that the photographs show identical scene coordinates.

## 11. Backend API Endpoints

### 11.1 Validate Image Pair

**Endpoint:**

`POST /api/comparison/validate`

**Purpose:**

Validate uploaded image files and return information about their dimensions and compatibility for processing.

**Request:**

Multipart form data containing the Before and After image files.

**Expected result:**

A structured response describing the uploaded images and their validation status.

### 11.2 Compare Images

**Endpoint:**

`POST /api/comparison/compare`

**Purpose:**

Perform alignment, SSIM analysis, change detection, and comparison result generation.

**Request parameters:**

- Before image.
- After image.
- Alignment mode.

**Supported alignment modes:**

- `auto`
- `direct`
- Explicit feature-based modes supported by the implementation.

**Response information includes:**

- Processing status.
- Alignment metadata.
- SSIM score.
- Changed area percentage.
- Changed region count.
- Changed region details.
- Generated comparison images.
- Warnings and diagnostic information.

### 11.3 Alignment Failure Response

When geometric alignment cannot be established, the backend returns HTTP 422.

Example:

```json
{
  "detail": {
    "error": "ALIGNMENT_FAILED",
    "message": "Image alignment failed. Use photos with similar viewpoints and visible common details.",
    "attempted_methods": [
      "SIFT",
      "AKAZE",
      "ORB"
    ],
    "failure_reasons": [
      "Insufficient matching features between the two images."
    ],
    "guidance": "Ensure both images show the same scene from a similar viewpoint with visible shared details. If the images are already pixel-aligned, you may use alignment_mode='direct'."
  }
}
```

The system intentionally rejects unreliable automatic alignment rather than producing potentially misleading comparison results.

## 12. Frontend Implementation

The React frontend provides an interactive Before/After Image Comparison page.

**Frontend route:**

`/image-comparison`

### 12.1 Image Upload Interface

The interface contains two independent upload panels:

- Before Inspection Photograph.
- After Inspection Photograph.

Each panel displays an image preview, file information, image dimensions, and controls to replace or remove the image.

### 12.2 Alignment Selection

The user can select the required alignment method.

Automatic alignment attempts the available feature detectors.

Direct mode is available only through explicit user selection.

### 12.3 Validation and Comparison Controls

The interface includes:

- Validate Pair.
- Compare Images.
- Retry Operation.
- Reset Comparison.

### 12.4 Comparison Metrics

The results dashboard displays:

| Metric | Description |
|---|---|
| Mean SSIM Score | Average structural similarity over the valid comparison region |
| Changed Area Percentage | Percentage of analyzed pixels classified as changed |
| Changed Regions Count | Number of retained connected change regions |
| Alignment Quality / Geometric Registration | Registration quality or verification status |
| Inliers / Matches | Feature correspondence statistics |
| Reprojection Error | Geometric registration error |
| Valid Overlap | Fraction of the reference frame available for comparison |

### 12.5 Visualization Features

The frontend supports multiple visual outputs:

1. SSIM Difference Heatmap.
2. Highlighted Changes.
3. Binary Change Mask.
4. Aligned Reference.
5. Normalized Before Image.
6. Before/After Slider.
7. Side-by-Side Comparison.
8. Expanded Image Viewer.

These visualizations allow users to inspect the comparison results beyond numerical statistics.

### 12.6 Detected Change Regions Table

Detected regions are presented with:

- Region number.
- Position.
- Bounding box dimensions.
- Area in pixels.

This helps identify where visual differences occur.

## 13. Error Handling

The frontend uses a centralized API error parser.

A previous defect caused HTTP 422 alignment failures to be incorrectly displayed as network errors.

The parser was corrected to preserve structured backend error codes and to support repeated parsing without losing the original error information.

### 13.1 Supported Error Categories

| Error | Handling |
|---|---|
| ALIGNMENT_FAILED | Amber warning with alignment guidance |
| INSUFFICIENT_OVERLAP | Warning indicating insufficient shared image area |
| INVALID_ALIGNMENT_MODE | Warning explaining invalid mode selection |
| NETWORK_ERROR | Used when no HTTP response is received |
| HTTP 503 | Service unavailable message |
| HTTP 504 | Timeout message |
| Validation errors | Displays relevant input validation information |

### 13.2 Alignment Failure UI

The warning panel displays:

- Image Alignment Failed.
- Error code.
- Attempted alignment methods.
- Failure reasons.
- Recommended action.
- Retry Operation button.

This makes alignment failures understandable without incorrectly instructing the user to restart a functioning backend.

## 14. Implementation Files

The main Task 2 implementation is organized within the existing AI-Assessment project.

### Backend

```text
backend/
└── app/
    ├── api/
    │   └── comparison.py
    ├── services/
    │   ├── comparison_service.py
    │   ├── image_alignment.py
    │   └── image_comparison_engine.py
    └── schemas/
        └── [comparison-related schemas]
```

### Frontend

```text
frontend/
├── src/
│   ├── pages/
│   │   └── ImageComparison.jsx
│   ├── components/
│   │   └── ErrorState.jsx
│   └── services/
│       └── api.js
└── tests/
    └── frontend_validation.test.js
```

### Tests

The reported backend test files include:

```text
backend/tests/
├── test_alignment.py
└── test_direct_alignment_telemetry.py
```

Other existing comparison and regression tests are retained in the project.

The exact directory tree should be treated as a description of the reported implementation rather than an independently inspected filesystem inventory.

## 15. Testing and Validation

Task 2 was validated using automated tests, controlled comparison scenarios, and manual frontend acceptance testing.

### 15.1 Automated Test Results

The final engineering report records:

| Test Suite | Passed | Skipped |
|---|---:|---:|
| Main Backend | 204 | 4 |
| YOLO Microservice | 32 | 0 |
| Frontend | 31 | 0 |
| **Total** | **267** | **4** |

All reported executed tests passed.

The YOLO tests are included as cross-module regression checks; the general Task 2 SSIM pipeline does not depend on YOLO inference.

The frontend production build also completed successfully.

### 15.2 Controlled Validation Scenarios

**Scenario A — Identical Images**

- Alignment mode: Auto.
- SSIM: 1.0000.
- Valid overlap: 100%.
- Changed area: 0%.
- Changed regions: 0.
- Result: Passed.

**Scenario B — Controlled Crack Addition**

- Alignment mode: Direct.
- SSIM: 0.9838.
- Valid overlap: 100%.
- Changed area: 1.83%.
- Changed regions: 1.
- Registration status: ASSUMED_UNVERIFIED.
- Result: Passed.

**Scenario C — Geometrically Transformed Image**

- Alignment mode: Auto.
- Detector selected: AKAZE.
- SSIM: 0.9995.
- Valid overlap: 96.92%.
- Alignment confidence: 0.9923.
- Inliers / Matches: 266 / 267.
- Reprojection error: 0.0084 pixels.
- Registration status: VERIFIED_FEATURE_REGISTRATION.
- Result: Passed.

**Scenario D — Unrelated Photographs**

- Alignment mode: Auto.
- Response: HTTP 422.
- Error code: ALIGNMENT_FAILED.
- Attempted methods: SIFT, AKAZE, ORB.
- Result: Correctly rejected.

**Scenario E — Different Image Dimensions**

- Input dimensions: 600 × 450 and 400 × 300.
- Alignment mode: Direct.
- SSIM: 0.9785.
- Valid canvas overlap: 100%.
- Dimension normalization: True.
- Alignment confidence: N/A.
- Warnings: Displayed.
- Result: Passed.

### 15.3 Manual Acceptance Testing

The completed manual verification covered:

- Direct mode telemetry.
- Automatic alignment diagnostics.
- Identical image comparison.
- Controlled crack addition.
- Unrelated photograph rejection.
- Different-resolution image warnings.
- Regression checks for other dashboard modules.

All checks were confirmed successful during final acceptance testing.

## 16. Observed Results

The implemented system demonstrates the ability to:

1. Accept and validate two inspection photographs.
2. Align photographs using local feature-based methods.
3. Reject pairs where reliable alignment cannot be established.
4. Compare explicitly pre-aligned images without feature matching.
5. Calculate SSIM and changed area statistics.
6. Generate localized difference heatmaps and binary masks.
7. Extract and report changed image regions.
8. Display interpretable geometric alignment diagnostics.
9. Communicate processing limitations and errors.
10. Present results through an interactive web dashboard.

The controlled crack-addition test demonstrated successful localization of a newly introduced visual feature.

The transformed-image test demonstrated accurate registration under the tested geometric transformation.

These results establish functional correctness for the evaluated scenarios, not universal accuracy across all real-world inspection conditions.

## 17. Limitations

### 17.1 Dependence on Common Visual Features

Automatic alignment requires sufficient corresponding visual details.

Smooth, featureless, or substantially different photographs may fail registration.

### 17.2 Sensitivity to Viewpoint Changes

Large perspective changes, extreme rotation, or substantial scale differences can prevent reliable homography estimation.

### 17.3 Planar Transformation Assumption

A homography is most appropriate for approximately planar surfaces or scenes adequately described by a projective transformation.

Complex three-dimensional changes and parallax can reduce alignment accuracy.

### 17.4 Sensitivity to Lighting and Surface Appearance

SSIM may respond to lighting changes, shadows, stains, weathering, and other non-damage variations.

These effects can contribute to detected change regions.

### 17.5 Direct Alignment Is Not Verified

Direct mode assumes correct pixel correspondence.

It does not estimate or verify the physical alignment of the two photographs.

### 17.6 Resolution Normalization Limitations

Resizing images to compatible dimensions does not guarantee identical framing.

Different aspect ratios can introduce geometric distortion.

### 17.7 No Damage Severity Classification

Task 2 measures general image differences.

It does not independently determine:

- Structural safety.
- Crack severity.
- Physical crack width.
- Engineering risk.
- The cause of observed changes.

### 17.8 Controlled Evaluation Scope

The final alignment and UI verification includes controlled test scenarios.

These tests do not establish performance guarantees for all real-world, longitudinal building inspection photographs.

## 18. Relationship to Other Assessment Modules

### Task 1 — Building Crack Detection

Task 1 uses a YOLO segmentation model to identify concrete cracks in individual photographs.

Task 1 is responsible for crack-specific detection and segmentation.

### Task 2 — Before/After Image Comparison

Task 2 uses image alignment, SSIM, and classical image processing to identify general visual differences between two photographs.

It is not dependent on crack detection.

### Hybrid Progression

The separate Hybrid Progression module combines image registration and comparison with YOLO crack segmentation to investigate potential changes in detected crack regions.

This module provides crack-specific temporal comparison capabilities beyond the general visual difference scope of Task 2.

### Task 3 — NLP News Classification

Task 3 performs text classification and NLP feature analysis.

It is independent of the computer vision processing pipeline.

All modules are accessible through the shared React dashboard.

## 19. Final Completion Checklist

| Requirement | Status |
|---|---|
| Upload two photographs | Completed |
| Validate input images | Completed |
| Automatic feature-based alignment | Completed |
| SIFT, AKAZE, ORB support | Completed |
| Homography estimation and verification | Completed |
| Explicit Direct comparison mode | Completed |
| SSIM calculation | Completed |
| Difference heatmap | Completed |
| Binary change mask | Completed |
| Highlight changed regions | Completed |
| Changed area percentage | Completed |
| Changed regions and coordinates | Completed |
| Alignment quality telemetry | Completed |
| Direct mode verification warnings | Completed |
| Dimension normalization metadata | Completed |
| Structured error handling | Completed |
| React frontend integration | Completed |
| Backend automated tests | Passed |
| Frontend automated tests | Passed |
| Production frontend build | Passed |
| Manual acceptance testing | Passed |
| Technical documentation | Completed |

## 20. Conclusion

Task 2 — Before/After Image Comparison has been successfully implemented and accepted as part of the AI-Assessment technical assessment.

The module integrates classical computer vision techniques, including SIFT, AKAZE, ORB, RANSAC homography, Structural Similarity Index analysis, image thresholding, and connected region detection.

The final implementation provides image validation, geometric registration, visual difference detection, changed area quantification, structured error handling, and an interactive React dashboard.

Additional reliability improvements ensure that automatic alignment is distinguished from unverified Direct comparison, preventing misleading alignment-quality reporting.

Automated regression tests and controlled validation scenarios passed, and the completed frontend was manually verified.

**Final Status: TASK 2 COMPLETED — IMPLEMENTED, TESTED, AND ACCEPTED.**

The system is suitable for demonstrating image registration and visual change detection in the technical assessment. It should be regarded as an inspection-support prototype rather than a certified structural damage assessment system.

---

**End of Task 2 Technical Documentation**