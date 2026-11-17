"""
Integration tests for HybridComparisonService.
Verifies alignment integration, homography calculation, SSIM on valid overlap,
and mock YOLO microservice aggregation.
"""
import asyncio
import base64
import cv2
import numpy as np
import pytest
from unittest.mock import AsyncMock, patch

from app.services.hybrid_comparison_service import HybridComparisonService, hybrid_comparison_service
from app.services.crack_change_engine import crack_change_engine


def run_async(coro):
    return asyncio.run(coro)


def make_textured_surface(width: int = 400, height: int = 400, seed: int = 42) -> np.ndarray:
    """Generate reproducible textured concrete-like surface rich in keypoints."""
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), 140, dtype=np.uint8)
    # Add random high-frequency texture blocks and speckles
    noise = rng.integers(-40, 40, (height, width, 3), dtype=np.int16)
    textured = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw distinctive geometric markings for guaranteed robust feature matching
    for i in range(10, width - 20, 35):
        for j in range(10, height - 20, 35):
            cv2.circle(textured, (i, j), 4, (30, 30, 30), -1)
            cv2.rectangle(textured, (i + 8, j + 8), (i + 22, j + 22), (200, 200, 200), 2)
            cv2.line(textured, (i, j + 15), (i + 25, j), (80, 80, 80), 1)

    return textured


def encode_img_bytes(img: np.ndarray, ext: str = ".jpg") -> bytes:
    _, buf = cv2.imencode(ext, img)
    return buf.tobytes()


def make_mock_yolo_response(mask_np: np.ndarray, conf_val: float = 0.5) -> dict:
    """Construct mock microservice response with base64 mask."""
    b64_mask = crack_change_engine.encode_image_png(mask_np)
    return {
        "status": "completed",
        "model": "YOLO26s-seg",
        "class_names": ["crack"],
        "image_width": mask_np.shape[1],
        "image_height": mask_np.shape[0],
        "detections_count": 1 if np.count_nonzero(mask_np) > 0 else 0,
        "crack_coverage_percentage": round(100.0 * np.count_nonzero(mask_np) / (mask_np.shape[0] * mask_np.shape[1]), 4),
        "detections": [],
        "images": {
            "original": b64_mask,
            "mask": b64_mask,
            "overlay": b64_mask,
        },
        "inference": {
            "device": "mock_gpu",
            "duration_ms": 15.0,
        },
        "warnings": [],
    }


class TestHybridComparisonService:

    def test_identical_image_pair_comparison(self):
        """Identical image pair: perfect alignment, SSIM ~ 1.0, 0% change."""
        img = make_textured_surface(400, 400)
        img_bytes = encode_img_bytes(img)

        # Mock crack masks (identical crack present)
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[150:250, 198:202] = 255  # 100 * 4 = 400 px
        mock_resp = make_mock_yolo_response(mask)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as mock_seg:
            mock_seg.return_value = mock_resp

            result = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=img_bytes,
                before_type="image/jpeg",
                after_bytes=img_bytes,
                after_type="image/jpeg",
                alignment_mode="auto",
            ))

            assert result.status == "completed"
            assert result.alignment.status in ("accepted", "identical")
            assert result.alignment.valid_overlap_percentage >= 99.0
            assert result.visual_comparison.ssim_score >= 0.98
            assert result.crack_comparison.before_crack_pixels == 400
            assert result.crack_comparison.after_crack_pixels == 400
            assert result.crack_comparison.persistent_pixels == 400
            assert result.crack_comparison.newly_observed_pixels == 0
            assert result.crack_comparison.no_longer_observed_pixels == 0
            assert result.crack_comparison.coverage_change_percentage_points == 0.0
            assert result.crack_comparison.persistent_components == 1

    def test_known_translation_alignment(self):
        """Before image translated by dx=+8, dy=+6 aligned into after image coordinate frame."""
        before = make_textured_surface(400, 400, seed=10)
        # Translate to create after image: shift by +8, +6
        M = np.float32([[1, 0, 8], [0, 1, 6]])
        after = cv2.warpAffine(before, M, (400, 400))

        before_bytes = encode_img_bytes(before)
        after_bytes = encode_img_bytes(after)

        # Empty crack masks for pure alignment & SSIM test
        empty_mask = np.zeros((400, 400), dtype=np.uint8)
        mock_resp = make_mock_yolo_response(empty_mask)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as mock_seg:
            mock_seg.return_value = mock_resp

            result = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=before_bytes,
                before_type="image/jpeg",
                after_bytes=after_bytes,
                after_type="image/jpeg",
                alignment_mode="sift",
            ))

            assert result.status == "completed"
            assert result.alignment.status == "accepted"
            assert result.alignment.inlier_count >= 15
            assert result.alignment.reprojection_error < 3.0
            assert result.alignment.valid_overlap_percentage > 85.0
            assert result.visual_comparison.ssim_score > 0.85

    def test_known_rotation_alignment(self):
        """Small rotation (3 degrees) aligned into reference frame."""
        before = make_textured_surface(400, 400, seed=15)
        center = (200, 200)
        rot_mat = cv2.getRotationMatrix2D(center, 3.0, 1.0)
        after = cv2.warpAffine(before, rot_mat, (400, 400))

        before_bytes = encode_img_bytes(before)
        after_bytes = encode_img_bytes(after)

        empty_mask = np.zeros((400, 400), dtype=np.uint8)
        mock_resp = make_mock_yolo_response(empty_mask)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as mock_seg:
            mock_seg.return_value = mock_resp

            result = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=before_bytes,
                before_type="image/jpeg",
                after_bytes=after_bytes,
                after_type="image/jpeg",
                alignment_mode="auto",
            ))

            assert result.status == "completed"
            assert result.alignment.status == "accepted"
            assert result.alignment.valid_overlap_percentage > 80.0

    def test_unrelated_images_rejected(self):
        """Unrelated images must be rejected with ValueError; NO silent fallback."""
        img1 = make_textured_surface(400, 400, seed=1)
        # Completely different scene: random inverted noise patterns
        img2 = np.zeros((400, 400, 3), dtype=np.uint8)
        for i in range(0, 400, 20):
            cv2.line(img2, (0, i), (400, 400 - i), (255, 255, 255), 3)

        b1 = encode_img_bytes(img1)
        b2 = encode_img_bytes(img2)

        with pytest.raises(ValueError) as excinfo:
            run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=b1,
                before_type="image/jpeg",
                after_bytes=b2,
                after_type="image/jpeg",
                alignment_mode="auto",
            ))
        assert "alignment failed" in str(excinfo.value).lower()

    def test_insufficient_features_blank_image_rejected(self):
        """Solid blank images with 0 features must fail alignment."""
        blank1 = np.full((300, 300, 3), 50, dtype=np.uint8)
        blank2 = np.full((300, 300, 3), 200, dtype=np.uint8)

        b1 = encode_img_bytes(blank1, ".png")
        b2 = encode_img_bytes(blank2, ".png")

        empty_mask = np.zeros((300, 300), dtype=np.uint8)
        mock_resp = make_mock_yolo_response(empty_mask)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as mock_seg:
            mock_seg.return_value = mock_resp

            with pytest.raises(ValueError) as excinfo:
                run_async(hybrid_comparison_service.compare_hybrid(
                    before_bytes=b1,
                    before_type="image/png",
                    after_bytes=b2,
                    after_type="image/png",
                    alignment_mode="sift",
                ))
            assert "alignment failed" in str(excinfo.value).lower()

    def test_direct_mode_explicitly_requested(self):
        """Direct mode bypasses feature matching and assigns DIRECT method."""
        img1 = make_textured_surface(300, 300, seed=3)
        img2 = img1.copy()
        # Non-identical so it doesn't trigger the byte-identical fast path
        img2[0:10, 0:10] = 0

        b1 = encode_img_bytes(img1, ".png")
        b2 = encode_img_bytes(img2, ".png")

        empty_mask = np.zeros((300, 300), dtype=np.uint8)
        mock_resp = make_mock_yolo_response(empty_mask)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as mock_seg:
            mock_seg.return_value = mock_resp

            result = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=b1,
                before_type="image/png",
                after_bytes=b2,
                after_type="image/png",
                alignment_mode="direct",
            ))

            assert result.status == "completed"
            assert result.alignment.method == "DIRECT"
            assert result.alignment.status == "direct"
            assert any("direct comparison" in w.lower() for w in result.alignment.warnings)

    def test_crack_progression_with_expanded_crack(self):
        """Verify crack progression detection when crack grows."""
        img = make_textured_surface(400, 400, seed=5)
        img_bytes = encode_img_bytes(img)

        # Before mask: crack from y=100..200 (100 px * 3 px = 300 px)
        before_mask = np.zeros((400, 400), dtype=np.uint8)
        before_mask[100:200, 150:153] = 255
        resp_before = make_mock_yolo_response(before_mask)

        # After mask: crack lengthened to y=100..320 (220 px * 3 px = 660 px) + new crack
        after_mask = np.zeros((400, 400), dtype=np.uint8)
        after_mask[100:320, 150:153] = 255  # Expanded component
        after_mask[300:350, 300:304] = 255  # Brand new component (50*4 = 200 px)
        resp_after = make_mock_yolo_response(after_mask)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as mock_seg:
            mock_seg.side_effect = [resp_before, resp_after]

            result = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=img_bytes,
                before_type="image/jpeg",
                after_bytes=img_bytes,
                after_type="image/jpeg",
                alignment_mode="auto",
                tolerance_px=2,
            ))

            assert result.crack_comparison.before_crack_pixels == 300
            assert result.crack_comparison.after_crack_pixels == 660 + 200
            assert result.crack_comparison.persistent_pixels == 300
            assert result.crack_comparison.newly_observed_pixels == 360 + 200
            assert result.crack_comparison.potential_expanded_components == 1
            assert result.crack_comparison.potential_new_components == 1
            assert result.crack_comparison.coverage_change_percentage_points > 0.0
