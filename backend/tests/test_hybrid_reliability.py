"""
Deterministic Reliability and Regression Tests for Phase 4.1.

Validates 20 specific scenarios:
 1. Identical no-change pair
 2. Translation-only pair
 3. Rotation-only pair
 4. Perspective transformation
 5. Brightness variation
 6. Blur variation
 7. JPEG compression
 8. Small synthetic crack addition
 9. Large synthetic crack addition
 10. Crack extension
 11. Component expansion
 12. No-longer-observed crack
 13. Unrelated image rejection
 14. Low-overlap rejection
 15. Spatial tolerance boundaries
 16. Coverage-change threshold behavior
 17. Empty-mask handling
 18. No division by zero
 19. Backward-compatible API response
 20. Reliability warnings
"""
import asyncio
import cv2
import numpy as np
import pytest
from unittest.mock import AsyncMock, patch

from app.services.hybrid_comparison_service import HybridComparisonService, hybrid_comparison_service
from app.services.crack_change_engine import crack_change_engine, CrackChangeEngine
from app.schemas.hybrid_comparison import HybridComparisonResponse


def run_async(coro):
    return asyncio.run(coro)


def make_test_texture(width: int = 400, height: int = 400, seed: int = 101) -> np.ndarray:
    """Generate high-contrast textured surface rich in SIFT/ORB keypoints."""
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), 130, dtype=np.uint8)
    noise = rng.integers(-30, 30, (height, width, 3), dtype=np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Distinct geometric landmarks for robust feature tracking
    for y in range(20, height - 30, 40):
        for x in range(20, width - 30, 40):
            cv2.circle(img, (x, y), 5, (20, 20, 20), -1)
            cv2.rectangle(img, (x + 8, y + 8), (x + 24, y + 24), (220, 220, 220), 2)
            cv2.line(img, (x, y + 15), (x + 20, y), (70, 70, 70), 2)
    return img


def encode_image(img: np.ndarray, ext: str = ".jpg", quality: int = 95) -> bytes:
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    _, buf = cv2.imencode(ext, img, params)
    return buf.tobytes()


def make_mock_yolo_resp(mask: np.ndarray) -> dict:
    b64 = crack_change_engine.encode_image_png(mask)
    h, w = mask.shape[:2]
    pts = np.count_nonzero(mask)
    return {
        "status": "completed",
        "model": "YOLO26s-seg",
        "class_names": ["crack"],
        "image_width": w,
        "image_height": h,
        "detections_count": 1 if pts > 0 else 0,
        "crack_coverage_percentage": round(100.0 * pts / (h * w), 4),
        "detections": [],
        "images": {"original": b64, "mask": b64, "overlay": b64},
        "inference": {"device": "mock_gpu", "duration_ms": 10.0},
        "warnings": [],
    }


class TestHybridReliabilityScenarios:
    """Complete 20-scenario reliability and noise-floor test suite."""

    # Scenario 1: Identical no-change pair
    def test_01_identical_no_change_pair(self):
        img = make_test_texture(400, 400)
        img_bytes = encode_image(img)
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[150:250, 198:202] = 255  # 400 pixels

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.return_value = make_mock_yolo_resp(mask)
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=img_bytes, before_type="image/jpeg",
                after_bytes=img_bytes, after_type="image/jpeg",
            ))
            assert res.crack_comparison.coverage_change_percentage_points == 0.0
            assert res.crack_comparison.tolerance_adjusted_new_pixels == 0
            assert res.crack_comparison.progression_label == "No substantial predicted change detected"
            assert res.crack_comparison.is_above_noise_floor is False
            assert res.crack_comparison.uncertainty_level == "low"

    # Scenario 2: Translation-only pair
    def test_02_translation_only_pair(self):
        before = make_test_texture(400, 400)
        M = np.float32([[1, 0, 15], [0, 1, 10]])
        after = cv2.warpAffine(before, M, (400, 400), borderMode=cv2.BORDER_REFLECT)

        before_mask = np.zeros((400, 400), dtype=np.uint8)
        before_mask[150:250, 198:202] = 255
        after_mask = cv2.warpAffine(before_mask, M, (400, 400), flags=cv2.INTER_NEAREST)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.side_effect = [make_mock_yolo_resp(before_mask), make_mock_yolo_resp(after_mask)]
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(before), before_type="image/jpeg",
                after_bytes=encode_image(after), after_type="image/jpeg",
            ))
            assert res.alignment.status == "accepted"
            assert res.crack_comparison.tolerance_adjusted_new_pixels == 0
            assert res.crack_comparison.progression_label == "No substantial predicted change detected"
            assert res.crack_comparison.is_above_noise_floor is False

    # Scenario 3: Rotation-only pair
    def test_03_rotation_only_pair(self):
        before = make_test_texture(400, 400)
        center = (200, 200)
        rot_mat = cv2.getRotationMatrix2D(center, 2.0, 1.0)
        after = cv2.warpAffine(before, rot_mat, (400, 400), borderMode=cv2.BORDER_REFLECT)

        before_mask = np.zeros((400, 400), dtype=np.uint8)
        before_mask[160:240, 198:202] = 255
        after_mask = cv2.warpAffine(before_mask, rot_mat, (400, 400), flags=cv2.INTER_NEAREST)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.side_effect = [make_mock_yolo_resp(before_mask), make_mock_yolo_resp(after_mask)]
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(before), before_type="image/jpeg",
                after_bytes=encode_image(after), after_type="image/jpeg",
            ))
            assert res.alignment.status == "accepted"
            assert res.crack_comparison.potential_new_components == 0
            assert res.crack_comparison.progression_label == "No substantial predicted change detected"

    # Scenario 4: Perspective transformation
    def test_04_perspective_transformation(self):
        before = make_test_texture(400, 400)
        pts1 = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
        pts2 = np.float32([[5, 4], [394, 2], [396, 396], [3, 395]])
        H_true = cv2.getPerspectiveTransform(pts1, pts2)
        after = cv2.warpPerspective(before, H_true, (400, 400), borderMode=cv2.BORDER_REFLECT)

        before_mask = np.zeros((400, 400), dtype=np.uint8)
        before_mask[160:240, 198:202] = 255
        after_mask = cv2.warpPerspective(before_mask, H_true, (400, 400), flags=cv2.INTER_NEAREST)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.side_effect = [make_mock_yolo_resp(before_mask), make_mock_yolo_resp(after_mask)]
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(before), before_type="image/jpeg",
                after_bytes=encode_image(after), after_type="image/jpeg",
            ))
            assert res.alignment.status == "accepted"
            assert abs(res.crack_comparison.coverage_change_percentage_points) <= 0.89
            assert res.crack_comparison.is_above_noise_floor is False

    # Scenario 5: Brightness variation
    def test_05_brightness_variation(self):
        before = make_test_texture(400, 400)
        after = cv2.convertScaleAbs(before, alpha=1.0, beta=35)
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[150:250, 198:202] = 255

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.return_value = make_mock_yolo_resp(mask)
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(before), before_type="image/jpeg",
                after_bytes=encode_image(after), after_type="image/jpeg",
            ))
            assert res.crack_comparison.progression_label == "No substantial predicted change detected"
            assert res.crack_comparison.coverage_change_percentage_points == 0.0

    # Scenario 6: Blur variation
    def test_06_blur_variation(self):
        before = make_test_texture(400, 400)
        after = cv2.GaussianBlur(before, (5, 5), 1.5)
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[150:250, 198:202] = 255

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.return_value = make_mock_yolo_resp(mask)
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(before), before_type="image/jpeg",
                after_bytes=encode_image(after), after_type="image/jpeg",
            ))
            assert res.visual_comparison.ssim_score < 0.95
            assert res.crack_comparison.progression_label == "No substantial predicted change detected"

    # Scenario 7: JPEG compression
    def test_07_jpeg_compression(self):
        before = make_test_texture(400, 400)
        after_bytes = encode_image(before, ext=".jpg", quality=25)
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[150:250, 198:202] = 255

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.return_value = make_mock_yolo_resp(mask)
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(before), before_type="image/jpeg",
                after_bytes=after_bytes, after_type="image/jpeg",
            ))
            assert res.crack_comparison.coverage_change_percentage_points == 0.0
            assert res.crack_comparison.is_above_noise_floor is False

    # Scenario 8: Small synthetic crack addition
    def test_08_small_synthetic_crack_addition(self):
        img = make_test_texture(400, 400)
        before_mask = np.zeros((400, 400), dtype=np.uint8)
        before_mask[150:250, 198:202] = 255

        after_mask = before_mask.copy()
        # Add small disjoint crack: 5x6 = 30 px (>= 20 px threshold)
        after_mask[50:56, 50:55] = 255

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.side_effect = [make_mock_yolo_resp(before_mask), make_mock_yolo_resp(after_mask)]
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(img), before_type="image/jpeg",
                after_bytes=encode_image(img), after_type="image/jpeg",
            ))
            assert res.crack_comparison.potential_new_components == 1
            assert res.crack_comparison.tolerance_adjusted_new_pixels >= 20
            assert res.crack_comparison.progression_label == "Potential newly observed crack regions"

    # Scenario 9: Large synthetic crack addition
    def test_09_large_synthetic_crack_addition(self):
        img = make_test_texture(400, 400)
        before_mask = np.zeros((400, 400), dtype=np.uint8)
        before_mask[150:250, 198:202] = 255

        after_mask = before_mask.copy()
        # Add large disjoint crack: 300x5 = 1500 px (> 0.89 pp coverage change)
        after_mask[50:350, 100:105] = 255

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.side_effect = [make_mock_yolo_resp(before_mask), make_mock_yolo_resp(after_mask)]
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(img), before_type="image/jpeg",
                after_bytes=encode_image(img), after_type="image/jpeg",
            ))
            assert res.crack_comparison.potential_new_components == 1
            assert res.crack_comparison.is_above_noise_floor is True
            assert res.crack_comparison.progression_label == "Potential newly observed crack regions"
            assert res.crack_comparison.uncertainty_level == "low"

    # Scenario 10: Crack extension
    def test_10_crack_extension(self):
        img = make_test_texture(400, 400)
        before_mask = np.zeros((400, 400), dtype=np.uint8)
        before_mask[150:200, 198:202] = 255  # 50 * 4 = 200 px

        after_mask = np.zeros((400, 400), dtype=np.uint8)
        after_mask[150:280, 198:202] = 255   # 130 * 4 = 520 px (>15% growth and >20 px)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.side_effect = [make_mock_yolo_resp(before_mask), make_mock_yolo_resp(after_mask)]
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(img), before_type="image/jpeg",
                after_bytes=encode_image(img), after_type="image/jpeg",
            ))
            assert res.crack_comparison.potential_expanded_components == 1
            assert res.crack_comparison.progression_label == "Potential expansion of detected crack regions"

    # Scenario 11: Component expansion
    def test_11_component_expansion(self):
        img = make_test_texture(400, 400)
        before_mask = np.zeros((400, 400), dtype=np.uint8)
        before_mask[100:150, 100:104] = 255  # 200 px

        after_mask = np.zeros((400, 400), dtype=np.uint8)
        after_mask[95:160, 98:106] = 255     # 65 * 8 = 520 px (>150% growth)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.side_effect = [make_mock_yolo_resp(before_mask), make_mock_yolo_resp(after_mask)]
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(img), before_type="image/jpeg",
                after_bytes=encode_image(img), after_type="image/jpeg",
            ))
            assert res.crack_comparison.potential_expanded_components >= 1
            assert res.crack_comparison.progression_label == "Potential expansion of detected crack regions"

    # Scenario 12: No-longer-observed crack
    def test_12_no_longer_observed_crack(self):
        img = make_test_texture(400, 400)
        before_mask = np.zeros((400, 400), dtype=np.uint8)
        before_mask[150:250, 198:202] = 255
        after_mask = np.zeros((400, 400), dtype=np.uint8)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.side_effect = [make_mock_yolo_resp(before_mask), make_mock_yolo_resp(after_mask)]
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(img), before_type="image/jpeg",
                after_bytes=encode_image(img), after_type="image/jpeg",
            ))
            assert res.crack_comparison.after_crack_pixels == 0
            assert res.crack_comparison.no_longer_observed_pixels == 400
            assert res.crack_comparison.no_longer_observed_components >= 1

    # Scenario 13: Unrelated image rejection
    def test_13_unrelated_image_rejection(self):
        img1 = make_test_texture(400, 400, seed=1)
        # Completely different non-overlapping pattern
        img2 = np.zeros((400, 400, 3), dtype=np.uint8)
        for i in range(10):
            cv2.line(img2, (0, i * 40), (400, i * 40), (255, 255, 255), 2)

        with pytest.raises(ValueError) as exc:
            run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(img1), before_type="image/jpeg",
                after_bytes=encode_image(img2), after_type="image/jpeg",
            ))
        assert "Image alignment failed" in str(exc.value)

    # Scenario 14: Low-overlap rejection
    def test_14_low_overlap_rejection(self):
        img1 = make_test_texture(400, 400, seed=10)
        # Severe shift yielding < 50% overlap
        M = np.float32([[1, 0, 250], [0, 1, 0]])
        img2 = cv2.warpAffine(img1, M, (400, 400))

        with pytest.raises(ValueError) as exc:
            run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(img1), before_type="image/jpeg",
                after_bytes=encode_image(img2), after_type="image/jpeg",
            ))
        assert "insufficient" in str(exc.value).lower() or "alignment failed" in str(exc.value).lower()

    # Scenario 15: Spatial tolerance boundaries
    def test_15_spatial_tolerance_boundaries(self):
        valid = np.full((100, 100), 255, dtype=np.uint8)
        B = np.zeros((100, 100), dtype=np.uint8)
        B[40:60, 50] = 255  # 1px line

        # Displacement 1 px
        A1 = np.zeros((100, 100), dtype=np.uint8)
        A1[40:60, 51] = 255
        res1 = crack_change_engine.analyze_crack_change(B, A1, valid, tolerance_px=2)
        assert res1["tolerance_adjusted_new_pixels"] == 0

        # Displacement 2 px
        A2 = np.zeros((100, 100), dtype=np.uint8)
        A2[40:60, 52] = 255
        res2 = crack_change_engine.analyze_crack_change(B, A2, valid, tolerance_px=2)
        assert res2["tolerance_adjusted_new_pixels"] == 0

        # Displacement 3 px (outside 2px radius)
        A3 = np.zeros((100, 100), dtype=np.uint8)
        A3[40:60, 53] = 255
        res3 = crack_change_engine.analyze_crack_change(B, A3, valid, tolerance_px=2)
        assert res3["tolerance_adjusted_new_pixels"] > 0

    # Scenario 16: Coverage-change threshold behavior
    def test_16_coverage_change_threshold_behavior(self):
        valid = np.full((100, 100), 255, dtype=np.uint8)  # 10,000 px
        B = np.zeros((100, 100), dtype=np.uint8)

        # 50 pixels = 0.50 pp change (< 0.89 pp calibrated noise floor)
        A_sub = np.zeros((100, 100), dtype=np.uint8)
        A_sub[10:20, 10:15] = 255  # 50 px
        res_sub = crack_change_engine.analyze_crack_change(B, A_sub, valid, calibrated_noise_floor_pp=0.89)
        assert res_sub["is_above_noise_floor"] is False

        # 120 pixels = 1.20 pp change (> 0.89 pp calibrated noise floor)
        A_sup = np.zeros((100, 100), dtype=np.uint8)
        A_sup[10:34, 10:15] = 255  # 120 px
        res_sup = crack_change_engine.analyze_crack_change(B, A_sup, valid, calibrated_noise_floor_pp=0.89)
        assert res_sup["is_above_noise_floor"] is True

    # Scenario 17: Empty-mask handling
    def test_17_empty_mask_handling(self):
        img = make_test_texture(400, 400)
        empty_mask = np.zeros((400, 400), dtype=np.uint8)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.return_value = make_mock_yolo_resp(empty_mask)
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(img), before_type="image/jpeg",
                after_bytes=encode_image(img), after_type="image/jpeg",
            ))
            assert res.crack_comparison.before_crack_pixels == 0
            assert res.crack_comparison.after_crack_pixels == 0
            assert res.crack_comparison.coverage_change_percentage_points == 0.0
            assert res.crack_comparison.progression_label == "No substantial predicted change detected"
            assert res.crack_comparison.is_above_noise_floor is False

    # Scenario 18: No division by zero
    def test_18_no_division_by_zero(self):
        empty = np.zeros((50, 50), dtype=np.uint8)
        # valid mask is all zeros
        res = crack_change_engine.analyze_crack_change(empty, empty, empty)
        assert res["valid_overlap_pixels"] == 0
        assert res["before_crack_coverage_percentage"] == 0.0
        assert res["after_crack_coverage_percentage"] == 0.0
        assert res["coverage_change_percentage_points"] == 0.0

    # Scenario 19: Backward-compatible API response
    def test_19_backward_compatible_api_response(self):
        img = make_test_texture(400, 400)
        mask = np.zeros((400, 400), dtype=np.uint8)
        mask[150:200, 198:202] = 255

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.return_value = make_mock_yolo_resp(mask)
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(img), before_type="image/jpeg",
                after_bytes=encode_image(img), after_type="image/jpeg",
            ))
            # Verify all historical Phase 4 fields exist
            d = res.crack_comparison.model_dump()
            required_keys = [
                "before_crack_pixels", "after_crack_pixels",
                "before_crack_coverage_percentage", "after_crack_coverage_percentage",
                "coverage_change_percentage_points", "persistent_pixels",
                "newly_observed_pixels", "no_longer_observed_pixels",
                "raw_change_pixels", "raw_change_percentage", "tolerance_px",
                "tolerance_adjusted_new_pixels", "tolerance_adjusted_no_longer_observed_pixels",
                "potential_new_components", "potential_expanded_components",
                "persistent_components", "no_longer_observed_components",
                # Phase 4.1 additions
                "progression_label", "calibrated_noise_floor_pp",
                "is_above_noise_floor", "uncertainty_level", "interpretation_summary",
            ]
            for k in required_keys:
                assert k in d, f"Missing field {k} in crack_comparison"

    # Scenario 20: Reliability warnings
    def test_20_reliability_warnings(self):
        img = make_test_texture(400, 400)
        mask = np.zeros((400, 400), dtype=np.uint8)

        with patch("app.services.hybrid_comparison_service.damage_client.segment_image", new_callable=AsyncMock) as m:
            m.return_value = make_mock_yolo_resp(mask)
            res = run_async(hybrid_comparison_service.compare_hybrid(
                before_bytes=encode_image(img), before_type="image/jpeg",
                after_bytes=encode_image(img), after_type="image/jpeg",
            ))
            warnings = res.warnings
            assert len(warnings) >= 3
            # Check warning content
            assert any("not establish physical structural deterioration" in w for w in warnings)
            assert any("Visual change (SSIM)" in w for w in warnings)
            assert any("empirical noise floor" in w for w in warnings)
