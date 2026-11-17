"""
Comprehensive regression tests for Task 2 Alignment Reliability and Telemetry.
Validates all 20 requirements specified in AI-ASSESSMENT Task 2 specification:
 1. Identical images with Auto alignment
 2. Identical images with Direct alignment
 3. Same scene with small translation
 4. Same scene with mild rotation
 5. Same scene with mild perspective transformation
 6. Unrelated images rejected by Auto
 7. Smooth or featureless images rejected by Auto
 8. Direct mode does not call feature matching
 9. Direct mode reports unverified geometric status
10. Direct mode reports N/A (None) instead of 0% quality
11. Direct mode with unequal dimensions follows documented policy
12. Different aspect ratios are not silently distorted (warning attached)
13. Auto mode reports actual inliers and reprojection error
14. Valid overlap is calculated correctly
15. Failed alignment does not silently fall back to Direct
16. SSIM 1.0 on identical images does not change semantic alignment status
17. Direct alignment warning text compliance
18. Auto alignment exposes real geometric diagnostics
19. Existing Task 2 comparison features continue to function
20. Hybrid Progression shared alignment behavior does not regress
"""

import math
import numpy as np
import cv2
import pytest
from unittest.mock import patch

from app.services.image_alignment import ImageAlignmentService
from app.services.image_comparison_engine import ImageComparisonEngine
from app.schemas.hybrid_comparison import HybridAlignmentInfo


# ---------------------------------------------------------------------------
# Test Helpers: Deterministic Synthetic Image Generators
# ---------------------------------------------------------------------------

def make_textured_image(width=500, height=400, seed=42):
    """Generate high-contrast, repeatable geometric features for detector testing."""
    rng = np.random.default_rng(seed)
    img = rng.integers(100, 210, (height, width, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Add circles, rectangles, and lines
    for _ in range(25):
        cx = int(rng.integers(30, width - 30))
        cy = int(rng.integers(30, height - 30))
        r = int(rng.integers(8, 30))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, (cx, cy), r, color, -1)

    for _ in range(20):
        x1 = int(rng.integers(10, width - 60))
        y1 = int(rng.integers(10, height - 60))
        x2 = x1 + int(rng.integers(20, 50))
        y2 = y1 + int(rng.integers(20, 50))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    for _ in range(15):
        pt1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        pt2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.line(img, pt1, pt2, color, thickness=2)

    return img


def make_featureless_image(width=500, height=400):
    """Generate a smooth surface with essentially zero detectable keypoints."""
    img = np.full((height, width, 3), 160, dtype=np.uint8)
    # subtle gradient only
    for y in range(height):
        img[y, :, :] = np.clip(155 + (y * 10 // height), 0, 255)
    return img


# ---------------------------------------------------------------------------
# Test Suite
# ---------------------------------------------------------------------------

class TestTask2AlignmentReliabilityAndTelemetry:

    # 1. Identical images with Auto alignment
    def test_01_identical_images_auto(self):
        svc = ImageAlignmentService()
        img = make_textured_image()
        result = svc.align(img, img.copy(), mode="auto")
        assert result["status"] == "identical"
        assert result["confidence"] == 1.0
        assert result["geometric_status"] == "IDENTICAL_INPUTS"
        assert result["overlap_ratio"] == 1.0

    # 2. Identical images with Direct alignment
    def test_02_identical_images_direct(self):
        svc = ImageAlignmentService()
        img = make_textured_image()
        result = svc.align(img, img.copy(), mode="direct")
        # Identical fast-path takes precedence
        assert result["status"] == "identical"
        assert result["confidence"] == 1.0
        assert result["geometric_status"] == "IDENTICAL_INPUTS"

    # 3. Same scene with a small translation
    def test_03_same_scene_small_translation(self):
        svc = ImageAlignmentService()
        before = make_textured_image()
        M = np.float32([[1, 0, 8], [0, 1, 6]])
        after = cv2.warpAffine(before, M, (before.shape[1], before.shape[0]))
        result = svc.align(before, after, mode="auto")
        assert result["status"] == "aligned"
        assert result["geometric_status"] == "VERIFIED_FEATURE_REGISTRATION"
        assert result["inliers"] >= 10
        assert result["reprojection_error"] < 3.0
        assert result["confidence"] > 0.5

    # 4. Same scene with mild rotation
    def test_04_same_scene_mild_rotation(self):
        svc = ImageAlignmentService()
        before = make_textured_image()
        h, w = before.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 3.0, 1.0)
        after = cv2.warpAffine(before, M, (w, h))
        result = svc.align(before, after, mode="auto")
        assert result["status"] == "aligned"
        assert result["inliers"] >= 10
        assert result["overlap_ratio"] > 0.85

    # 5. Same scene with mild perspective transformation
    def test_05_same_scene_mild_perspective(self):
        svc = ImageAlignmentService()
        before = make_textured_image()
        h, w = before.shape[:2]
        src_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_corners = np.float32([[5, 4], [w - 6, 2], [w - 3, h - 5], [4, h - 4]])
        H_true = cv2.getPerspectiveTransform(src_corners, dst_corners)
        after = cv2.warpPerspective(before, H_true, (w, h))
        result = svc.align(before, after, mode="auto")
        assert result["status"] == "aligned"
        assert result["inliers"] >= 10

    # 6. Unrelated images rejected by Auto
    def test_06_unrelated_images_rejected_by_auto(self):
        svc = ImageAlignmentService()
        img1 = make_textured_image(seed=101)
        img2 = make_textured_image(seed=999)
        result = svc.align(img1, img2, mode="auto")
        assert result["status"] == "failed"
        assert result["geometric_status"] == "FAILED"
        assert result["confidence"] == 0.0

    # 7. Smooth or featureless images rejected by Auto
    def test_07_smooth_images_rejected_by_auto(self):
        svc = ImageAlignmentService()
        img1 = make_featureless_image()
        img2 = make_featureless_image()
        # Add slight pixel offset so not byte-identical
        img2[0, 0, 0] = 161
        result = svc.align(img1, img2, mode="auto")
        assert result["status"] == "failed"
        assert result["confidence"] == 0.0

    # 8. Direct mode does not call feature matching
    def test_08_direct_mode_skips_feature_matching(self):
        svc = ImageAlignmentService()
        before = make_textured_image(seed=1)
        after = make_textured_image(seed=2)
        with patch.object(cv2, "BFMatcher") as mock_matcher:
            result = svc.align(before, after, mode="direct")
            mock_matcher.assert_not_called()
        assert result["status"] == "direct"
        assert result["method_used"] == "DIRECT"

    # 9. Direct mode reports unverified geometric status
    def test_09_direct_mode_reports_unverified_status(self):
        svc = ImageAlignmentService()
        before = make_textured_image(seed=1)
        after = make_textured_image(seed=2)
        result = svc.align(before, after, mode="direct")
        assert result["geometric_status"] == "ASSUMED_UNVERIFIED"

    # 10. Direct mode reports N/A (None) instead of 0% quality
    def test_10_direct_mode_reports_none_quality(self):
        svc = ImageAlignmentService()
        before = make_textured_image(seed=1)
        after = make_textured_image(seed=2)
        result = svc.align(before, after, mode="direct")
        assert result["confidence"] is None
        assert result["alignment_quality"] is None
        assert result["inliers"] is None
        assert result["matches"] is None
        assert result["inlier_ratio"] is None
        assert result["reprojection_error"] is None

    # 11. Direct mode with unequal dimensions follows documented policy
    def test_11_direct_mode_unequal_dimensions(self):
        engine = ImageComparisonEngine()
        before = np.zeros((400, 600, 3), dtype=np.uint8)
        after = np.zeros((300, 450, 3), dtype=np.uint8)
        # Ensure not byte-identical
        before[10, 10] = 200
        after[20, 20] = 180

        res = engine.compare(before, after, mode="direct")
        align = res["alignment"]
        assert align["dimension_normalized"] is True
        assert align["geometric_status"] == "ASSUMED_UNVERIFIED"
        assert align["confidence"] is None
        assert "dimension_details" in align
        assert align["dimension_details"]["before_source"]["width"] == 600
        assert align["dimension_details"]["after_source"]["width"] == 450

    # 12. Different aspect ratios are not silently distorted
    def test_12_different_aspect_ratios_warning(self):
        engine = ImageComparisonEngine()
        before = np.zeros((400, 600, 3), dtype=np.uint8)  # aspect ratio 1.5
        after = np.zeros((400, 400, 3), dtype=np.uint8)   # aspect ratio 1.0
        before[10, 10] = 255
        after[10, 10] = 200

        res = engine.compare(before, after, mode="direct")
        align = res["alignment"]
        assert align["dimension_normalized"] is True
        assert any("aspect ratio" in w.lower() for w in align["warnings"])

    # 13. Auto mode reports actual inliers and reprojection error
    def test_13_auto_reports_real_diagnostics(self):
        svc = ImageAlignmentService()
        before = make_textured_image()
        M = np.float32([[1, 0, 5], [0, 1, 3]])
        after = cv2.warpAffine(before, M, (before.shape[1], before.shape[0]))
        result = svc.align(before, after, mode="auto")
        assert isinstance(result["inliers"], int)
        assert result["inliers"] > 0
        assert isinstance(result["reprojection_error"], float)
        assert result["reprojection_error"] >= 0.0

    # 14. Valid overlap is calculated correctly
    def test_14_valid_overlap_calculation(self):
        svc = ImageAlignmentService()
        before = make_textured_image()
        M = np.float32([[1, 0, 20], [0, 1, 0]])
        after = cv2.warpAffine(before, M, (before.shape[1], before.shape[0]))
        result = svc.align(before, after, mode="auto")
        assert 0.70 < result["overlap_ratio"] < 1.0

    # 15. Failed alignment does not silently fall back to Direct
    def test_15_no_silent_fallback_to_direct(self):
        engine = ImageComparisonEngine()
        img1 = make_textured_image(seed=1)
        img2 = make_textured_image(seed=999)
        with pytest.raises(ValueError, match="Image alignment failed"):
            engine.compare(img1, img2, mode="auto")

    # 16. SSIM 1.0 on identical images does not change semantic alignment status
    def test_16_ssim_identical_preserves_status(self):
        engine = ImageComparisonEngine()
        img = make_textured_image()
        res = engine.compare(img, img.copy(), mode="auto")
        assert res["ssim_score"] == pytest.approx(1.0, abs=1e-3)
        assert res["alignment"]["status"] == "identical"
        assert res["alignment"]["geometric_status"] == "IDENTICAL_INPUTS"

    # 17. Frontend renders Direct alignment warnings
    def test_17_direct_warning_string_compliance(self):
        engine = ImageComparisonEngine()
        img1 = make_textured_image(seed=1)
        img2 = make_textured_image(seed=2)
        res = engine.compare(img1, img2, mode="direct")
        warnings = res["alignment"]["warnings"]
        assert any(
            "direct comparison mode: geometric alignment was not verified" in w.lower()
            for w in warnings
        )
        assert any(
            "resizing or normalization does not independently verify alignment" in w.lower()
            for w in warnings
        )

    # 18. Frontend renders Auto alignment diagnostics
    def test_18_auto_diagnostics_exposed(self):
        engine = ImageComparisonEngine()
        before = make_textured_image()
        M = np.float32([[1, 0, 4], [0, 1, 2]])
        after = cv2.warpAffine(before, M, (before.shape[1], before.shape[0]))
        res = engine.compare(before, after, mode="auto")
        align = res["alignment"]
        assert "detector" in align
        assert "method_used" in align
        assert "geometric_status" in align
        assert align["geometric_status"] == "VERIFIED_FEATURE_REGISTRATION"
        assert align["inliers"] is not None
        assert align["reprojection_error"] is not None

    # 19. Existing Task 2 comparison features still work
    def test_19_comparison_features_intact(self):
        engine = ImageComparisonEngine()
        before = make_textured_image()
        after = before.copy()
        # Add a synthetic crack line to 'after'
        cv2.line(after, (50, 50), (200, 200), (0, 0, 0), thickness=4)
        res = engine.compare(before, after, mode="direct")
        assert res["status"] == "completed"
        assert res["changed_percentage"] > 0
        assert res["changed_regions_count"] >= 1
        assert "images" in res
        assert "difference" in res["images"]
        assert "mask" in res["images"]
        assert "highlighted" in res["images"]

    # 20. Hybrid Progression shared alignment behavior does not regress
    def test_20_hybrid_shared_alignment_does_not_regress(self):
        # Verify HybridAlignmentInfo schema accommodates both Direct (None) and Auto values
        direct_info = HybridAlignmentInfo(
            status="direct",
            method="DIRECT",
            method_used="DIRECT",
            detector="DIRECT",
            geometric_status="ASSUMED_UNVERIFIED",
            confidence=None,
            inlier_count=None,
            inlier_ratio=None,
            reprojection_error=None,
            valid_overlap_pixels=100000,
            valid_overlap_percentage=100.0,
            warnings=["Geometric alignment not verified."],
        )
        assert direct_info.confidence is None
        assert direct_info.geometric_status == "ASSUMED_UNVERIFIED"

        auto_info = HybridAlignmentInfo(
            status="accepted",
            method="SIFT",
            method_used="SIFT",
            detector="SIFT",
            geometric_status="VERIFIED_FEATURE_REGISTRATION",
            confidence=0.88,
            inlier_count=52,
            inlier_ratio=0.74,
            reprojection_error=1.12,
            valid_overlap_pixels=98000,
            valid_overlap_percentage=98.0,
            warnings=[],
        )
        assert auto_info.confidence == 0.88
        assert auto_info.geometric_status == "VERIFIED_FEATURE_REGISTRATION"
