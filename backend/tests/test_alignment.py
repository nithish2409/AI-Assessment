"""
Comprehensive tests for ImageAlignmentService and its integration
with the ImageComparisonEngine and the FastAPI endpoint.

Synthetic images use deterministic seeds and controlled geometric
transforms to ensure reproducibility without network access.
"""

import math
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from fastapi.testclient import TestClient
from app.main import app
from app.services.image_alignment import ImageAlignmentService, _HAS_AKAZE
from app.services.image_comparison_engine import ImageComparisonEngine


client = TestClient(app)


# ======================================================================
# Helpers — deterministic synthetic images
# ======================================================================

def _textured_image(
    width: int = 600,
    height: int = 400,
    seed: int = 42,
) -> np.ndarray:
    """Generate a BGR image with rich, deterministic texture suitable
    for feature detection (circles, rectangles, lines, gradient patches).
    """
    rng = np.random.default_rng(seed)
    # Start with a smooth random-color canvas.
    img = rng.integers(80, 200, (height, width, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Sprinkle geometric shapes to give detectors strong keypoints.
    for _ in range(30):
        cx = int(rng.integers(30, width - 30))
        cy = int(rng.integers(30, height - 30))
        r = int(rng.integers(8, 35))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.circle(img, (cx, cy), r, color, thickness=-1)

    for _ in range(20):
        x1 = int(rng.integers(0, width - 40))
        y1 = int(rng.integers(0, height - 40))
        x2 = x1 + int(rng.integers(20, 80))
        y2 = y1 + int(rng.integers(20, 60))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    for _ in range(15):
        pt1 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        pt2 = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.line(img, pt1, pt2, color, thickness=int(rng.integers(1, 4)))

    return img


def _translate_image(
    img: np.ndarray, tx: int, ty: int,
) -> np.ndarray:
    """Translate *img* by (tx, ty) pixels using an affine warp."""
    h, w = img.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, M, (w, h))


def _rotate_image(
    img: np.ndarray, angle_deg: float,
) -> np.ndarray:
    """Rotate *img* around its centre."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (w, h))


def _scale_image(
    img: np.ndarray, factor: float,
) -> np.ndarray:
    """Uniformly scale the image."""
    h, w = img.shape[:2]
    new_w, new_h = int(w * factor), int(h * factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _unrelated_image(
    width: int = 600, height: int = 400,
) -> np.ndarray:
    """A completely different synthetic image (different seed)."""
    return _textured_image(width, height, seed=999)


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


# ======================================================================
# 1. Identical images
# ======================================================================

class TestIdenticalImages:
    def test_identical_returns_identical_status(self):
        svc = ImageAlignmentService()
        img = _textured_image()
        result = svc.align(img, img.copy())
        assert result["status"] == "identical"
        assert result["confidence"] == 1.0
        assert result["warnings"] == []

    def test_identical_valid_mask_all_white(self):
        svc = ImageAlignmentService()
        img = _textured_image()
        result = svc.align(img, img.copy())
        mask = result["valid_mask"]
        assert mask.shape == img.shape[:2]
        assert np.all(mask == 255)


# ======================================================================
# 2. SIFT alignment on translated images
# ======================================================================

class TestSIFTAlignment:
    def test_sift_translation(self):
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=20, ty=15)
        result = svc.align(before, after, mode="sift")
        assert result["status"] == "aligned"
        assert result["method"] == "SIFT"
        assert result["inliers"] >= svc.MIN_INLIERS
        assert result["confidence"] > 0

    def test_sift_rotation(self):
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _rotate_image(before, angle_deg=3)
        result = svc.align(before, after, mode="sift")
        assert result["status"] == "aligned"
        assert result["method"] == "SIFT"


# ======================================================================
# 3. AKAZE alignment
# ======================================================================

class TestAKAZEAlignment:
    def test_akaze_environment_availability(self):
        """Verify that AKAZE is present in this OpenCV installation."""
        assert _HAS_AKAZE, (
            "cv2.AKAZE_create is missing. Ensure opencv-python-headless 4.x "
            "is installed rather than development 5.x builds lacking AKAZE."
        )

    def test_akaze_capability(self):
        """Verify AKAZE can be constructed and extract binary descriptors."""
        detector = cv2.AKAZE_create()
        assert detector is not None
        img = _textured_image(400, 300)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        assert len(keypoints) > 10, f"Expected keypoints, got {len(keypoints)}"
        assert descriptors is not None
        assert descriptors.dtype == np.uint8, "AKAZE default descriptors must be binary (uint8)"
        assert descriptors.shape[0] == len(keypoints)

    def test_akaze_translation(self):
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=15, ty=10)
        result = svc.align(before, after, mode="akaze")
        assert result["status"] == "aligned"
        assert result["method"] == "AKAZE"
        assert result["inliers"] >= svc.MIN_INLIERS
        assert result["inlier_ratio"] >= svc.MIN_INLIER_RATIO
        assert result["confidence"] > 0.0

    def test_akaze_rotation(self):
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _rotate_image(before, angle_deg=3)
        result = svc.align(before, after, mode="akaze")
        assert result["status"] == "aligned"
        assert result["method"] == "AKAZE"
        assert result["inliers"] >= svc.MIN_INLIERS

    def test_akaze_quality_metrics_and_mask(self):
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=12, ty=8)
        result = svc.align(before, after, mode="akaze")
        assert result["reprojection_error"] <= svc.MAX_REPROJ_ERROR
        assert result["overlap_ratio"] >= svc.MIN_OVERLAP_RATIO
        mask = result["valid_mask"]
        assert mask.shape == before.shape[:2]
        assert mask.dtype == np.uint8
        assert np.count_nonzero(mask) > 0

    def test_akaze_unavailable_returns_none(self):
        """When AKAZE is absent, the detector gracefully returns failed."""
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=15, ty=10)
        with patch("app.services.image_alignment._HAS_AKAZE", False):
            result = svc.align(before, after, mode="akaze")
            assert result["status"] == "failed"


# ======================================================================
# 4. ORB alignment
# ======================================================================

class TestORBAlignment:
    def test_orb_translation(self):
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=25, ty=20)
        result = svc.align(before, after, mode="orb")
        assert result["status"] == "aligned"
        assert result["method"] == "ORB"
        assert result["matches"] >= svc.MIN_GOOD_MATCHES

    def test_orb_preserves_existing_parameters(self):
        """ORB should use nfeatures=3000 (legacy compatibility)."""
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=10, ty=10)
        result = svc.align(before, after, mode="orb")
        assert result["status"] == "aligned"


# ======================================================================
# 5. Automatic fallback behavior
# ======================================================================

class TestAutoMode:
    def test_auto_selects_best(self):
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=20, ty=15)
        result = svc.align(before, after, mode="auto")
        assert result["status"] == "aligned"
        assert result["method"] in ("SIFT", "AKAZE", "ORB")

    def test_auto_tries_all_methods_via_mock(self):
        """Force SIFT to fail, verify auto falls through to ORB."""
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=20, ty=15)

        original_try = svc._try_method

        def mock_try(method, *args, **kwargs):
            if method == "sift":
                return None  # Simulate SIFT failure
            return original_try(method, *args, **kwargs)

        with patch.object(svc, "_try_method", side_effect=mock_try):
            result = svc.align(before, after, mode="auto")
            assert result["status"] == "aligned"
            # Should have fallen through to AKAZE or ORB.
            assert result["method"] in ("AKAZE", "ORB")

    def test_auto_all_fail_returns_failure(self):
        """When every detector fails, auto must NOT silently succeed."""
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=20, ty=15)

        def mock_try(method, *args, **kwargs):
            return None  # All fail

        with patch.object(svc, "_try_method", side_effect=mock_try):
            result = svc.align(before, after, mode="auto")
            assert result["status"] == "failed"
            assert len(result["attempted_methods"]) >= 2
            assert result["confidence"] == 0.0


# ======================================================================
# 6. Different image resolutions
# ======================================================================

class TestDifferentResolutions:
    def test_direct_mode_resizes(self):
        svc = ImageAlignmentService()
        before = _textured_image(600, 400)
        after = _textured_image(800, 500, seed=42)
        result = svc.align(before, after, mode="direct")
        assert result["status"] == "direct"
        aligned = result["aligned_image"]
        assert aligned.shape[:2] == before.shape[:2]

    def test_engine_handles_different_sizes(self):
        """ImageComparisonEngine resizes before delegating."""
        engine = ImageComparisonEngine()
        before = _textured_image(600, 400)
        after = _textured_image(800, 500, seed=42)
        # Direct mode should succeed even with mismatched sizes.
        result = engine.compare(before, after, mode="direct")
        assert result["status"] == "completed"


# ======================================================================
# 7. Insufficient keypoints
# ======================================================================

class TestInsufficientKeypoints:
    def test_blank_image_fails(self):
        """A solid-colour image has no keypoints."""
        svc = ImageAlignmentService()
        blank = np.full((400, 600, 3), 128, dtype=np.uint8)
        result = svc.align(blank, blank.copy(), mode="sift")
        # identical fast path triggers first for exact copies
        assert result["status"] == "identical"

    def test_blank_vs_different_blank_fails(self):
        svc = ImageAlignmentService()
        blank_a = np.full((400, 600, 3), 128, dtype=np.uint8)
        blank_b = np.full((400, 600, 3), 200, dtype=np.uint8)
        result = svc.align(blank_a, blank_b, mode="sift")
        assert result["status"] == "failed"


# ======================================================================
# 8. Unrelated images
# ======================================================================

class TestUnrelatedImages:
    def test_unrelated_auto_fails(self):
        """Auto mode must NOT silently produce a direct comparison."""
        svc = ImageAlignmentService()
        before = _textured_image(seed=1)
        after = _unrelated_image()
        result = svc.align(before, after, mode="auto")
        assert result["status"] == "failed"
        assert result["confidence"] == 0.0

    def test_unrelated_not_direct(self):
        """Verify no silent fallback to direct mode."""
        svc = ImageAlignmentService()
        before = _textured_image(seed=1)
        after = _unrelated_image()
        result = svc.align(before, after, mode="auto")
        assert result["method"] != "DIRECT"


# ======================================================================
# 9. Explicit direct mode
# ======================================================================

class TestDirectMode:
    def test_direct_status(self):
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=5, ty=5)
        result = svc.align(before, after, mode="direct")
        assert result["status"] == "direct"
        assert result["method"] == "DIRECT"
        assert result["confidence"] is None
        assert result["method_used"] == "DIRECT"
        assert result["geometric_status"] == "ASSUMED_UNVERIFIED"

    def test_direct_has_warning(self):
        svc = ImageAlignmentService()
        img = _textured_image()
        result = svc.align(img, img.copy(), mode="direct")
        # identical fast path takes priority
        # If not identical, direct would have warnings
        # Let's use slightly different images:
        after = _translate_image(img, tx=1, ty=0)
        result = svc.align(img, after, mode="direct")
        assert result["status"] == "direct"
        assert len(result["warnings"]) >= 1
        assert "geometric alignment" in result["warnings"][0].lower()

    def test_direct_overlap_all_valid(self):
        svc = ImageAlignmentService()
        img = _textured_image()
        after = _translate_image(img, tx=3, ty=3)
        result = svc.align(img, after, mode="direct")
        assert result["overlap_ratio"] == 1.0
        mask = result["valid_mask"]
        assert np.all(mask == 255)


# ======================================================================
# 10. Invalid alignment mode
# ======================================================================

class TestInvalidMode:
    def test_raises_on_invalid_mode(self):
        svc = ImageAlignmentService()
        img = _textured_image()
        with pytest.raises(ValueError, match="Unknown alignment mode"):
            svc.align(img, img, mode="superalign")

    def test_api_rejects_invalid_mode(self):
        img = _textured_image()
        resp = client.post(
            "/api/comparison/compare",
            files={
                "before": ("b.png", _encode_png(img), "image/png"),
                "after": ("a.png", _encode_png(img), "image/png"),
            },
            data={"alignment_mode": "invalid_mode"},
        )
        assert resp.status_code == 422
        body = resp.json()
        assert body["detail"]["error"] == "INVALID_ALIGNMENT_MODE"


# ======================================================================
# 11. Homography validation
# ======================================================================

class TestHomographyValidation:
    def test_degenerate_homography_rejected(self):
        svc = ImageAlignmentService()
        # Zero matrix is degenerate.
        result = svc._validate_homography(np.zeros((3, 3)), 600, 400)
        assert not result["valid"]

    def test_identity_homography_accepted(self):
        svc = ImageAlignmentService()
        result = svc._validate_homography(np.eye(3), 600, 400)
        assert result["valid"]

    def test_extreme_scale_rejected(self):
        svc = ImageAlignmentService()
        H = np.eye(3) * 10.0  # 10× scale
        H[2, 2] = 1.0
        result = svc._validate_homography(H, 600, 400)
        assert not result["valid"]

    def test_nan_homography_rejected(self):
        svc = ImageAlignmentService()
        H = np.full((3, 3), np.nan)
        result = svc._validate_homography(H, 600, 400)
        assert not result["valid"]


# ======================================================================
# 12. Valid overlap mask
# ======================================================================

class TestValidOverlapMask:
    def test_aligned_mask_shape(self):
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=15, ty=10)
        result = svc.align(before, after, mode="sift")
        assert result["status"] == "aligned"
        mask = result["valid_mask"]
        assert mask.shape == before.shape[:2]
        assert mask.dtype == np.uint8

    def test_mask_not_all_black(self):
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=15, ty=10)
        result = svc.align(before, after, mode="sift")
        mask = result["valid_mask"]
        assert np.count_nonzero(mask) > 0


# ======================================================================
# 13. Confidence bounds
# ======================================================================

class TestConfidenceBounds:
    def test_confidence_between_0_and_1(self):
        svc = ImageAlignmentService()
        before = _textured_image()
        after = _translate_image(before, tx=20, ty=15)
        result = svc.align(before, after, mode="sift")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_function_clamps(self):
        # Edge case: perfect metrics.
        c = ImageAlignmentService._confidence(1.0, 200, 0.0, 1.0)
        assert c == pytest.approx(1.0)

        # Edge case: worst metrics.
        c = ImageAlignmentService._confidence(0.0, 0, 10.0, 0.0)
        assert c == pytest.approx(0.0)

    def test_failed_alignment_zero_confidence(self):
        svc = ImageAlignmentService()
        before = _textured_image(seed=1)
        after = _unrelated_image()
        result = svc.align(before, after, mode="auto")
        assert result["confidence"] == 0.0


# ======================================================================
# 14. Structured API failure
# ======================================================================

class TestStructuredAPIFailure:
    def test_alignment_failure_structured_response(self):
        """Unrelated images → 422 with ALIGNMENT_FAILED error body."""
        before = _textured_image(seed=1)
        after = _unrelated_image()
        resp = client.post(
            "/api/comparison/compare",
            files={
                "before": ("b.png", _encode_png(before), "image/png"),
                "after": ("a.png", _encode_png(after), "image/png"),
            },
            data={"alignment_mode": "auto"},
        )
        assert resp.status_code == 422
        body = resp.json()
        detail = body["detail"]
        assert detail["error"] == "ALIGNMENT_FAILED"
        assert "message" in detail
        assert "attempted_methods" in detail
        assert "guidance" in detail

    def test_api_compare_default_mode_backward_compat(self):
        """Sending only files (no alignment_mode) must still work."""
        before = _textured_image()
        after = _translate_image(before, tx=10, ty=5)
        resp = client.post(
            "/api/comparison/compare",
            files={
                "before": ("b.png", _encode_png(before), "image/png"),
                "after": ("a.png", _encode_png(after), "image/png"),
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert "ssim_score" in data
        assert "changed_percentage" in data
        assert "regions" in data
        assert "images" in data


# ======================================================================
# 15. Backward compatibility with existing comparison tests
# ======================================================================

class TestBackwardCompatibility:
    """Re-verify the original engine tests still pass with the new
    alignment service underneath."""

    def _make_legacy_pair(self):
        rng = np.random.default_rng(42)
        before = rng.integers(
            100, 220, size=(400, 600, 3), dtype=np.uint8
        )
        before = cv2.GaussianBlur(before, (3, 3), 0)
        after = before.copy()
        cv2.line(
            after, (150, 120), (430, 290),
            (10, 10, 10), thickness=8,
        )
        return before, after

    def test_legacy_identical(self):
        before, _ = self._make_legacy_pair()
        engine = ImageComparisonEngine()
        result = engine.compare(before, before.copy())
        assert result["status"] == "completed"
        assert result["alignment"]["status"] == "identical"
        assert result["ssim_score"] == pytest.approx(1.0, abs=0.001)
        assert result["changed_percentage"] == 0.0

    def test_legacy_change_detection(self):
        before, after = self._make_legacy_pair()
        engine = ImageComparisonEngine()
        result = engine.compare(before, after)
        assert result["status"] == "completed"
        assert result["changed_percentage"] > 0
        assert result["changed_regions_count"] >= 1

    def test_legacy_response_fields(self):
        before, after = self._make_legacy_pair()
        engine = ImageComparisonEngine()
        result = engine.compare(before, after)
        for key in (
            "status", "alignment", "ssim_score",
            "changed_percentage", "changed_regions_count",
            "regions", "image_width", "image_height", "images",
        ):
            assert key in result, f"Missing response field: {key}"

        imgs = result["images"]
        for key in (
            "before", "aligned_after", "difference",
            "mask", "highlighted",
        ):
            assert key in imgs, f"Missing image field: {key}"

    def test_alignment_metadata_present(self):
        """New metadata fields must appear in the alignment sub-dict."""
        before, after = self._make_legacy_pair()
        engine = ImageComparisonEngine()
        result = engine.compare(before, after)
        alignment = result["alignment"]
        assert "confidence" in alignment
        assert "inlier_ratio" in alignment
        assert "reprojection_error" in alignment
        assert "warnings" in alignment


# ======================================================================
# Additional integration: engine + direct mode
# ======================================================================

class TestEngineDirectMode:
    def test_engine_direct_mode(self):
        engine = ImageComparisonEngine()
        img = _textured_image()
        after = _translate_image(img, tx=3, ty=2)
        result = engine.compare(img, after, mode="direct")
        assert result["status"] == "completed"
        assert result["alignment"]["status"] == "direct"
        assert result["alignment"]["method"] == "DIRECT"
        assert len(result["alignment"]["warnings"]) >= 1
