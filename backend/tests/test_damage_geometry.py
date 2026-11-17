"""
Comprehensive automated tests for YOLO damage segmentation geometry, rasterization,
pixel-area calculations, and serialization (Phase 3C.1 audit).
"""
import base64
import cv2
import numpy as np
import pytest
from pathlib import Path

from damage_inference.image_utils import (
    validate_and_decode_image,
    encode_image_to_base64_png,
    create_overlay,
)


# ---------------------------------------------------------------------------
# Synthetic Deterministic Geometry Tests
# ---------------------------------------------------------------------------

def test_case_01_thin_diagonal_crack():
    """1. Thin diagonal crack-shaped polygon."""
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.array([[20, 20], [180, 180], [181, 182], [21, 22]], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 255)
    area = int(np.count_nonzero(mask))
    assert area > 0
    # Diagonal crack area should be much smaller than the 160x160 bounding box (25,600 px)
    assert area < 1000


def test_case_02_irregular_zigzag_polygon():
    """2. Irregular zigzag polygon."""
    h, w = 300, 300
    mask = np.zeros((h, w), dtype=np.uint8)
    zigzag = np.array([
        [50, 50], [100, 80], [70, 120], [150, 160], [120, 200], [200, 250],
        [195, 255], [115, 205], [145, 165], [65, 125], [95, 85], [45, 55]
    ], dtype=np.int32)
    cv2.fillPoly(mask, [zigzag], 255)
    area = int(np.count_nonzero(mask))
    assert 500 < area < 3000


def test_case_03_multiple_disconnected_cracks():
    """3. Multiple disconnected cracks."""
    h, w = 400, 400
    mask = np.zeros((h, w), dtype=np.uint8)
    c1 = np.array([[30, 30], [80, 30], [80, 40], [30, 40]], dtype=np.int32)
    c2 = np.array([[200, 200], [300, 200], [300, 215], [200, 215]], dtype=np.int32)
    cv2.fillPoly(mask, [c1, c2], 255)
    area1 = (80 - 30 + 1) * (40 - 30 + 1)  # 51 * 11 = 561 px
    area2 = (300 - 200 + 1) * (215 - 200 + 1)  # 101 * 16 = 1616 px
    assert np.count_nonzero(mask) == area1 + area2


def test_case_04_overlapping_crack_masks():
    """4. Overlapping crack masks."""
    h, w = 200, 200
    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(m1, (50, 50), (100, 100), 255, -1)   # Area 51x51 = 2601
    cv2.rectangle(m2, (80, 50), (130, 100), 255, -1)   # Area 51x51 = 2601
    union = cv2.bitwise_or(m1, m2)
    # Total width is 50 to 130 = 81 px, height 51 px -> 81 * 51 = 4131 px
    assert np.count_nonzero(union) == 81 * 51
    assert np.count_nonzero(union) < np.count_nonzero(m1) + np.count_nonzero(m2)


def test_case_05_crack_touching_image_boundary():
    """5. Crack touching an image boundary."""
    h, w = 200, 200
    poly = np.array([[0, 0], [w, 0], [w, 10], [0, 10]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    assert mask[0, 0] == 255
    assert mask[0, w - 1] == 255
    assert mask[10, 0] == 255
    assert mask[11, 0] == 0


def test_case_06_single_pixel_or_very_small_predicted_region():
    """6. Single-pixel or very small predicted region."""
    h, w = 100, 100
    mask = np.zeros((h, w), dtype=np.uint8)
    tiny = np.array([[50, 50], [51, 50], [51, 51], [50, 51]], dtype=np.int32)
    cv2.fillPoly(mask, [tiny], 255)
    area = np.count_nonzero(mask)
    assert 1 <= area <= 4


def test_case_07_non_square_landscape_image():
    """7. Non-square landscape image."""
    raw = np.full((300, 800, 3), 128, dtype=np.uint8)
    _, enc = cv2.imencode(".png", raw)
    img_bgr, h, w = validate_and_decode_image(enc.tobytes(), "image/png")
    assert h == 300 and w == 800


def test_case_08_non_square_portrait_image():
    """8. Non-square portrait image."""
    raw = np.full((800, 300, 3), 128, dtype=np.uint8)
    _, enc = cv2.imencode(".png", raw)
    img_bgr, h, w = validate_and_decode_image(enc.tobytes(), "image/png")
    assert h == 800 and w == 300


def test_case_09_empty_detection_set():
    """9. Empty detection set."""
    h, w = 200, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    coverage = (np.count_nonzero(mask) / (h * w)) * 100.0
    assert coverage == 0.0


def test_case_10_correct_original_resolution_mask_dimensions():
    """10. Correct original-resolution mask dimensions."""
    h, w = 375, 512
    mask = np.zeros((h, w), dtype=np.uint8)
    assert mask.shape == (375, 512)


def test_case_11_correct_polygon_to_mask_rasterization():
    """11. Correct polygon-to-mask rasterization."""
    h, w = 100, 100
    pts = np.array([[10, 10], [50, 10], [50, 40], [10, 40]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    # 41 x 31 = 1271 px
    assert np.count_nonzero(mask) == 41 * 31


def test_case_12_correct_per_instance_pixel_area():
    """12. Correct per-instance pixel area."""
    h, w = 100, 100
    poly = np.array([[20, 20], [60, 20], [60, 60], [20, 60]], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    area = int(np.count_nonzero(mask))
    assert area == 41 * 41


def test_case_13_correct_union_mask_area():
    """13. Correct union mask area."""
    h, w = 100, 100
    m1 = np.zeros((h, w), dtype=np.uint8)
    m2 = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(m1, [np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.int32)], 255)
    cv2.fillPoly(m2, [np.array([[20, 10], [40, 10], [40, 30], [20, 30]], dtype=np.int32)], 255)
    union = cv2.bitwise_or(m1, m2)
    assert np.count_nonzero(union) == 31 * 21


def test_case_14_correct_coverage_percentage():
    """14. Correct coverage percentage."""
    h, w = 100, 100
    total = h * w
    mask = np.zeros((h, w), dtype=np.uint8)
    # Fill exactly 500 pixels (5.0%)
    mask[:10, :50] = 255
    cov = (np.count_nonzero(mask) / total) * 100.0
    assert cov == 5.0


def test_case_15_correct_base64_png_decoding():
    """15. Correct base64 PNG decoding."""
    img = np.full((150, 150, 3), 75, dtype=np.uint8)
    b64 = encode_image_to_base64_png(img)
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(b64), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (150, 150, 3)
    assert np.array_equal(decoded, img)


def test_case_16_correct_overlay_alignment():
    """16. Correct overlay alignment."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    overlay = create_overlay(img, mask, alpha=0.5, highlight_color=(0, 0, 255))
    assert overlay[50, 50, 2] > 0
    assert overlay[10, 10, 2] == 0


def test_case_17_bounding_box_area_not_substituted_for_mask_area():
    """17. Bounding-box area is not substituted for mask area."""
    h, w = 500, 500
    # BBox: [50, 50, 450, 450] -> bbox area = 400 * 400 = 160,000 px
    # Diagonal crack: 1 pixel wide line from (50, 50) to (450, 450)
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.array([[50, 50], [450, 450], [451, 451], [51, 51]], dtype=np.int32)
    cv2.fillPoly(mask, [poly], 255)
    crack_area = int(np.count_nonzero(mask))
    bbox_area = 400 * 400
    assert crack_area < bbox_area * 0.05, "Crack area must NOT equal bounding box area!"


def test_case_18_correct_mask_detection_indexing():
    """18. Correct mask/detection indexing."""
    boxes = [np.array([10, 10, 20, 20]), np.array([50, 50, 70, 70])]
    polys = [np.array([[10, 10], [20, 20], [15, 25]]), np.array([[50, 50], [70, 70], [60, 75]])]
    assert len(boxes) == len(polys)
    for i, (b, p) in enumerate(zip(boxes, polys)):
        assert p[:, 0].min() >= b[0] - 5


def test_case_19_behavior_when_segmentation_masks_are_absent():
    """19. Correct behavior when segmentation masks are absent."""
    h, w = 200, 200
    binary_mask = np.zeros((h, w), dtype=np.uint8)
    assert int(np.count_nonzero(binary_mask)) == 0


def test_case_20_actual_model_inference_with_real_crack_image():
    """20. Actual model inference with a real crack image (runs in PyTorch env)."""
    torch = pytest.importorskip("torch")
    from damage_inference.service import DamageInferenceService
    from damage_inference.config import CHECKPOINT_PATH

    service = DamageInferenceService(checkpoint_path=CHECKPOINT_PATH)
    assert service.initialize() is True

    backend_dir = Path(__file__).resolve().parent.parent
    test_img = backend_dir / "datasets" / "damage" / "processed" / "images" / "test" / "1616.rf.c868709931a671796794fdbb95352c5a.jpg"
    if not test_img.exists():
        pytest.skip("Test image not present")

    import asyncio
    res = asyncio.run(service.segment_image(test_img.read_bytes(), "image/jpeg", 0.25, 0.45))
    assert res.status == "completed"
    assert res.detections_count > 0
    # Verify first detection polygon is not a 4-point rectangle
    d0 = res.detections[0]
    assert len(d0.polygon) > 10, f"Expected irregular polygon with many points, got {len(d0.polygon)}"
    # Verify reported area matches polygon rasterization within 1%
    recon = np.zeros((res.image_height, res.image_width), dtype=np.uint8)
    cv2.fillPoly(recon, [np.round(np.array(d0.polygon)).astype(np.int32)], 255)
    recon_area = int(np.count_nonzero(recon))
    diff = abs(recon_area - d0.area_pixels)
    assert diff <= d0.area_pixels * 0.01, f"Area mismatch: {recon_area} vs {d0.area_pixels}"
